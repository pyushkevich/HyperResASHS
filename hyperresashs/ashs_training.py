import os
from os.path import join
from .utils.upsample_inr_method import create_link
from .utils.upsample_linear_method import linear_isotropic_upsampling, pad_image_with_world_alignment_in_memory
from .utils.trim_neck import trim_neck_in_memory
from picsl_greedy import Greedy3D
from picsl_c3d import Convert3D
import yaml
from types import SimpleNamespace
import torch
import time
import shutil
import tempfile
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Protocol, Dict, Literal, Type, Callable, Any
from .preprocessing import *
from .ashs_inference import ASHSExperimentBase, ASHSProcessor
from .utils.tool import copy_or_link_file
import pandas as pd


def process_manifest(manifest_csv: str) -> pd.DataFrame:
    """
    Validate the manifest CSV file for training preprocessing, replacing 
    relative paths with absolute paths, and return a DataFrame.
    
    Raises an error if the manifest is invalid
    
    Parameters:
    -----------
    manifest_csv : str
        Path to the manifest CSV file to validate.
    
    The manifest CSV should have the following columns:
        - 'id': Unique identifier for each case (e.g., subject ID)
        - 'date': [Optional] date for the case (e.g., scan date)
        - 'tse': Path to the TSE image for the case
        - 'mprage': Path to the MPRAGE image for the case
        - 'seg_left': Path to the left hemisphere segmentation for the case
        - 'seg_right': Path to the right hemisphere segmentation for the case
    """
    required_columns = {'id', 'tse', 'mprage', 'seg_left', 'seg_right'}
    
    df = pd.read_csv(manifest_csv)
    
    # Check for required columns
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    # Check that file paths exist as either an absolute path or relative to the manifest file
    manifest_dir = os.path.dirname(manifest_csv)
    def fix_filename(path):
        if not os.path.isabs(path):
            path = os.path.join(manifest_dir, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File does not exist at row {i+1}, column '{col}': {path}")
        return path
        
    for col in ['tse', 'mprage', 'seg_left', 'seg_right']:
        df[col] = df[col].apply(fix_filename)
        
    print("Manifest validation successful.")
    return df


class HyperASHSTraining:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # TODO: this is redundant with the ASHSInference class. Consider refactoring to avoid this redundancy.
        # Should really load the namespace and pass as part of the config
        with open(config['FILE_NAME_CONFIG']) as f:
            settings = yaml.safe_load(f)
            self.nm = SimpleNamespace(**settings)

    def preprocess_from_manifest_file(self, manifest_file: str, output_dir: str, 
                                      overwrite_existing=False, save_intermediates=False, create_links=True):
        """
        Preprocess data for training from a manifest file. Performs cropping and registration to prepare
        data for INR upsampling and nnUNet training. 
        
        Parameters:
        -----------
        manifest_file : str
            Path to the manifest CSV file. See below
        output_dir : str
            Directory where the preprocessed data will be saved. The function will create subdirectories for each case.
    
        Manifest file format:
        ---------------
        The manifest file should be a CSV with the following columns:
            - 'id': Unique identifier for each case (e.g., subject ID)
            - 'date': [Optional] date for the case (e.g., scan date)
            - 'tse': Path to the TSE image for the case
            - 'mprage': Path to the MPRAGE image for the case
            - 'seg_left': Path to the left hemisphere segmentation for the case
            - 'seg_right': Path to the right hemisphere segmentation for the case
        """
        df = process_manifest(manifest_file)

        # Create a preprocessing/registration worker
        reg = ASHSProcessor(self.config, training_mode=True,
                            overwrite_existing=overwrite_existing, 
                            save_intermediates=save_intermediates) 
        
        for i, (_,row) in enumerate(df.iterrows()):
            
            subject, date = str(row['id']), str(row['date']) if 'date' in row else 'nodate'
            
            # Create a folder in the output directory for this case
            case_path = join(output_dir, subject, date, 'preproc')
            os.makedirs(case_path, exist_ok=True)
            
            # TODO: redirect stdout/stderr to a log file for this specific case
            
            # Create the ASHS experiment representation
            exp = ASHSExperimentBase(self.config, case_path, self.nm, subject=subject, date=date)
            
            # Link or copy the input files to the working directory folder
            for col, dest in [('mprage', exp.gpe.t1_native), 
                              ('tse', exp.gpe.t2_whole_img),
                              ('seg_left', exp.lpe['left'].input_seg),
                              ('seg_right', exp.lpe['right'].input_seg)]:
                copy_or_link_file(row[col], dest.filename, 
                                  create_links=create_links, force_overwrite=overwrite_existing)
            
            # Execute the registration and preprocessing steps (neck trimming, global and local registration, ROI cropping)
            print('=' * 40)
            print(f'Preprocessing case {i+1}/{len(df)}: {subject} - {date}')
            print('=' * 40)
            reg.preprocess(exp)
            
            # At this point, we have the images registered, cropped, and prepared for both INR and nnUNet training.
             
            
