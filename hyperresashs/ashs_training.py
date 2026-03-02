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
from .ashs_exp import ASHSExperimentBase
from .ashs_preproc import ASHSProcessor, Timer
from .utils.tool import copy_or_link_file
import pandas as pd
import re
from importlib.resources import files
from argparse import Namespace
import sys


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
            raise FileNotFoundError(f"File specified in manifest .csv does not exist': {path}")
        return path
        
    for col in ['tse', 'mprage', 'seg_left', 'seg_right']:
        df[col] = df[col].apply(fix_filename)
        
    print("Manifest validation successful.")
    return df


class HyperASHSTraining:
    """
    This class is used to organize HyperASHS model training 
    
    Parameters:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing settings for the ASHS processing and training. 
        This should include at least the 'FILE_NAME_CONFIG' key pointing to the YAML file with filename settings.
    manifest_file : str
        Path to the manifest CSV file. See below
    output_dir : str
        Directory where the preprocessed data will be saved. The function will create subdirectories for each case.
    overwrite_existing : bool, optional
        Whether to overwrite existing preprocessed files if they already exist. Default is False.
    save_intermediates : bool, optional
        Whether to save intermediate files generated during preprocessing steps. Default is False.
    create_links : bool, optional
        Whether to create symbolic links to the original files in the working directory instead of copying them. Default is True.

    Manifest file format:
    ---------------
    The manifest file should be a CSV with the following columns:
        - 'id': Unique identifier for each case (e.g., subject ID)
        - 'date': [Optional] date for the case (e.g., scan date)
        - 'tse': Path to the TSE image for the case
        - 'mprage': Path to the MPRAGE image for the case
        - 'seg_left': Path to the left hemisphere segmentation for the case
        - 'seg_right': Path to the right hemisphere segmentation for the case
        
    Output directory structure:
    ---------------
    The function will create a subdirectory for each case in the output directory, with the following structure:
        output_dir/
            preproc/
                subject1/
                    date1/
                        Preprocessed files for subject1 date1 (e.g., registered TSE, MPRAGE, segmentations) 
            inr_training/
                subject1_date1/
                    Files prepared for INR training (e.g., cropped TSE, MPRAGE, segmentations)
                    
    """
    
    def __init__(self, config: Dict[str, Any], manifest_file: str, output_dir: str, 
                 overwrite_existing=False, save_intermediates=False, create_links=True):
        self.config = config
        self.manifest_file = manifest_file
        self.output_dir = output_dir
        self.overwrite_existing = overwrite_existing
        self.save_intermediates = save_intermediates
        self.create_links = create_links

        # TODO: this is redundant with the ASHSInference class. Consider refactoring to avoid this redundancy.
        # Should really load the namespace and pass as part of the config
        with open(config['FILE_NAME_CONFIG']) as f:
            settings = yaml.safe_load(f)
            self.nm = SimpleNamespace(**settings)
            
        # Load the manifest file and validate it, replacing relative paths with absolute paths
        self.df = process_manifest(manifest_file)

        # Define top-level folders for preprocessed data and INR training data
        self.dir_preproc = join(output_dir, 'preproc')
        self.dir_inr_training = join(output_dir, 'inr_training')
        
        # Initialize the experiment dictionary to store the ASHSExperimentBase objects for each case
        self.d_exp = {}
        for i, (_,row) in enumerate(self.df.iterrows()):
            
            subject, date = str(row['id']), str(row['date']) if 'date' in row else 'nodate'
            
            # Create a folder in the output directory for this case
            case_path = join(self.dir_preproc, subject, date)
            inr_path_map = {side: join(self.dir_inr_training, f'{subject}_{date}_{side}') for side in ['left', 'right']}
            os.makedirs(case_path, exist_ok=True)
            
            # Create the ASHS experiment representation
            exp = self.d_exp[(subject,date)] = ASHSExperimentBase(self.config, case_path, self.nm, 
                                                                  subject=subject, date=date, 
                                                                  inr_path=inr_path_map)
            
            # Link or copy the input files to the working directory folder
            for col, dest in [('mprage', exp.gpe.t1_native), 
                              ('tse', exp.gpe.t2_whole_img),
                              ('seg_left', exp.lpe['left'].input_seg),
                              ('seg_right', exp.lpe['right'].input_seg)]:
                copy_or_link_file(row[col], dest.filename, 
                                  create_links=create_links, force_overwrite=overwrite_existing)

    def preprocess(self, filter=None):
        """
        Execute the preprocessing steps for each case in the manifest file, including 
        registration, neck trimming, and preparation for INR training.
        """ 

        # Create a preprocessing/registration worker
        reg = ASHSProcessor(self.config, training_mode=True,
                            overwrite_existing=self.overwrite_existing, 
                            save_intermediates=self.save_intermediates) 
        
        # Perform initial processing steps for each case (registration, INR preprocessing, and nnUNet preprocessing)
        for i, ((subject, date), exp) in enumerate(self.d_exp.items()):
            # Execute the registration and preprocessing steps (neck trimming, global and local registration, ROI cropping)
            if filter is not None:
                case_id = f'{subject}_{date}'
                if not re.search(filter, case_id):
                    print(f'Skipping case {case_id} due to filter "{filter}"')
                    continue

            print('=' * 40)
            print(f'Preprocessing case {i+1}/{len(self.df)}: {subject} - {date}')
            print('=' * 40)
            reg.preprocess(exp)
            reg.prepare_inr(exp)
            
    def train_inr(self, filter=None, device='cuda'):
        """
        Train the INR model for each case in the manifest file using the preprocessed data.
        """ 
        # Import the INR trainig code
        from .submodules.multi_contrast_inr.main import main as inr_main
        from .submodules.multi_contrast_inr.main import parse_args as inr_parse_args
        
        # Read the template YAML file
        config_temp = files('hyperresashs').joinpath('config_templates/config_inr_template.yaml')
        with config_temp.open('r') as f:
            config = yaml.safe_load(f)
            
        # Set common fields in the config file
        config["SETTINGS"]["DIRECTORY"] = self.dir_inr_training
        config["MODEL"]["MODEL_CLASS"] = 'MLPv2WithEarlySeg'
        config["TRAINING"]["EPOCHS"] = 60

        # Run INR for each subject            
        for i, ((subject, date), exp) in enumerate(self.d_exp.items()):
            for side in ['left', 'right']:
                case_id = f'{subject}_{date}_{side}'
                
                # Check filter
                if filter is not None:
                    if not re.search(filter, case_id):
                        print(f'Skipping case {case_id} due to filter "{filter}"')
                        continue
                
                # Check if the file can be skipped because the output already exists
                if self.overwrite_existing and exp.lpe[side].t2_patch_hyperres_seg.exists():
                    print(f'Skipping case {case_id} because output already exists and overwrite_existing=True')
                    continue
                
                print('=' * 40)
                print(f'Training INR for case {i+1}/{len(self.df)}: {subject} - {date} - {side}')
                
                # Create a config for this case        
                inr_work_dir = exp.lpe[side].dir_inr_train_input     
                inr_result_dir = join(inr_work_dir, 'result')   
                config["DATASET"]["SUBJECT_ID"] = case_id
                config["SETTINGS"]["SAVE_PATH"] = inr_result_dir
                
                # Write the config file
                inr_config = join(inr_work_dir, 'config.yaml')
                with open(inr_config, 'w') as f:
                    yaml.safe_dump(config, f, sort_keys=False)
                                
                saved_args = sys.argv
                sys.argv = ['test', '--config', inr_config]
                if device is not None:
                    sys.argv.extend(['--device', device])

                # Time the INR
                with Timer() as tm_inr:            
                    inr_main(inr_parse_args())
                    
                sys.argv = saved_args
                print(f'INR training completed for case {case_id} in {tm_inr.total:.1f} seconds.')                
                
                # Finally, copy the trained model to the case folder and perform any necessary cleanup.
                # Sicne the INR ROI does not have the same full context as the ROI defined based on the 
                # template, the segmentaton needs to be resampled to the original ROI space.
                
                # Find the INR output. TODO: this hard-codes 60 epochs!
                inr_result_img_dir = join(inr_result_dir, 'images', case_id, config["MODEL"]["MODEL_CLASS"])
                last_epoch = config["TRAINING"]["EPOCHS"]-1
                inr_files = os.listdir(inr_result_img_dir)
                fn_inr_final_seg = join(inr_result_img_dir, [x for x in inr_files if x.endswith(f'e{last_epoch}__seg.nii.gz')][0])
                
                c3d = Convert3D()
                c3d.push(exp.lpe[side].t2_patch_hyperres.data)      # hyper_primary
                c3d.execute(f"{fn_inr_final_seg} -int 0 -reslice-identity")
                exp.lpe[side].t2_patch_hyperres_seg.data = c3d.peek(-1)                
                
                print('=' * 40)
                
                         
            
