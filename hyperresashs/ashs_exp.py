import os
from os.path import join
from types import SimpleNamespace
import SimpleITK as sitk
from matplotlib.gridspec import GridSpec
from typing import Dict, Type, Callable, Any

class LazyPipelineElement:
    """
    A generic lazy-loading pipeline element that defers reading from disk until needed.
    
    Supports multiple object types via a registration system. New types can be registered
    using LazyPipelineElement.register() with custom loader/saver functions.
    
    Example:
        LazyPipelineElement.register(
            sitk.Transform,
            loader=lambda path: sitk.ReadTransform(path),
            saver=lambda transform, path: sitk.WriteTransform(transform, path)
        )
        lazy_xfm = LazyPipelineElement("transform.mat", sitk.Transform)
        transform = lazy_xfm.data  # Loads on first access
    """
    def __init__(self, filename, obj_type: Type):
        self.filename = filename
        self._data = None
        self.obj_type = obj_type
        
    @classmethod
    def register(cls, obj_type: Type, loader:Callable[[str], Any], saver: Callable[[Any, str], None]):
        """Register a new type of pipeline element with a loader and writer function"""
        def get(self):
            if self._data is None and os.path.exists(self.filename):
                self._data = loader(self.filename)
            return self._data
        
        def set(self, value):
            self._data = value
            saver(value, self.filename)
        
        setattr(cls, f'get_{obj_type.__name__}', get)
        setattr(cls, f'set_{obj_type.__name__}', set)
        
    @property
    def data_or_none(self) -> Any:
        """Read the object from disk if it hasn't been read yet, and return it. Returns None if the file does not exist."""
        if self._data is None and os.path.exists(self.filename):
            # Get the loader function for this type
            loader = getattr(self, f'get_{self.obj_type.__name__}', None)
            if loader is None:
                raise ValueError(f"No loader registered for type {self.obj_type}")
            self._data = loader()
            
        return self._data
    
    @property
    def data(self) -> Any:
        """Read the object from disk if it hasn't been read yet, and return it. Raises exception if data does not exist."""
        if self._data is None:
            if not os.path.exists(self.filename):
                raise FileNotFoundError(f"File does not exist: {self.filename}")

            # Get the loader function for this type
            loader = getattr(self, f'get_{self.obj_type.__name__}', None)
            if loader is None:
                raise ValueError(f"No loader registered for type {self.obj_type}")
            self._data = loader()
            
        return self._data
    
    @data.setter
    def data(self, value:Any):
        """Set the object and write it to disk immediately"""
        self._data = value
        if self._data is not None:
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)
            saver = getattr(self, f'set_{self.obj_type.__name__}', None)
            if saver is None:
                raise ValueError(f"No saver registered for type {self.obj_type}")
            saver(value)
        
    def exists(self) -> bool:
        """Check if the image file exists on disk"""
        return os.path.exists(self.filename)
    
    def __str__(self) -> str:
        return f"LazyPipelineElement(filename='{self.filename}', type={self.obj_type.__name__}, exists={self.exists()}, loaded={self._data is not None})"
        

LazyPipelineElement.register(sitk.Image, 
                             lambda path: sitk.ReadImage(path),
                             lambda img, path: sitk.WriteImage(img, path))
        
class LazyImage(LazyPipelineElement):
    """
    Convenience wrapper for lazy-loading SimpleITK images.
    
    Specialized version of LazyPipelineElement for sitk.Image objects. Automatically
    defers reading from disk until accessed via the .data property, and immediately
    writes to disk when set.
    
    Example:
        lazy_img = LazyImage("path/to/image.nii.gz")
        if lazy_img.exists():
            img = lazy_img.data  # Loads on first access
            # Process img...
            lazy_img.data = processed_img  # Saves immediately
    """
    def __init__(self, filename):
        super().__init__(filename, sitk.Image)        
 
 
# This class stores the images generated using processing
class GlobalPipelineElements:
    def __init__(self, case_path:str, nm: SimpleNamespace):
        # Registration matrices
        self.fn_save_mat_path_t2_to_t1_global = join(case_path, 'global_matrix_3tt2_to_3tt1.mat')
        self.fn_template_to_3tt1_affine_matrix = join(case_path, nm.affine_matrix)

        # Raw inputs                
        self.t2_whole_img = LazyImage(join(case_path, nm.t2_whole_img))
        self.t1_native = LazyImage(join(case_path, nm.t1_whole_img_before_registeration))
        
        # Processed images
        self.t1_neck_trim = LazyImage(join(case_path, nm.t1_name_after_triming_neck))
        self.t1_reg_to_t2 = LazyImage(join(case_path, nm.t1_whole_img))
        self.template_to_3tt1 = LazyImage(join(case_path, nm.template_to_3tt1))
        self.t2_padded_img = LazyImage(join(case_path, nm.t2_padded_img))
    

class TemplatePipelineElements:
    def __init__(self, template_path:str, nm: SimpleNamespace):
        self.template_3tt1 = LazyImage(join(template_path, nm.template))
        self.template_roi_left = LazyImage(join(template_path, nm.left_roi_file))
        self.template_roi_right = LazyImage(join(template_path, nm.right_roi_file))


class LocalPipelineElements:
    def __init__(self, case_path:str, side:str, test_folder: str, nm: SimpleNamespace, 
                 inr_path: str|None=None, nnunet_train_id: int|None = None):
        # Folders
        self.dir_local = join(case_path, test_folder, side)
        self.dir_nnunet_input = join(self.dir_local, 'input')
        self.dir_nnunet_output = join(self.dir_local, 'output')
        
        # Images and matrices generated during preprocessing
        self.roi_in_t1_space = LazyImage(join(case_path, nm.global_roi_in_3tt1_XYZ.replace('XYZ', side)))
        self.fn_save_mat_path_t2_to_t1_local = join(self.dir_local, nm.reg_mat)
        self.t2_patch_hyperres = LazyImage(join(self.dir_local, nm.hyper_primary))
        self.t1_patch_warped_hyperres = LazyImage(join(self.dir_local, nm.hyper_secondary_after_registertion))
        self.fn_registration_qc = join(self.dir_local, f'registration_qc.png')
        
        # NNUNet input and output
        self.nnunet_train_id = nnunet_train_id
        self.hl_nnunet_t2_input = join(self.dir_nnunet_input, 'MTL_000_0000.nii.gz')
        self.hl_nnunet_t1_input = join(self.dir_nnunet_input, 'MTL_000_0001.nii.gz')
        self.nnunet_seg = LazyImage(join(self.dir_nnunet_output, 'MTL_000.nii.gz'))
        self.fn_segmentation_qc = join(self.dir_local, f'segmentation_qc.png')
        
        # Final post-processed segmentation
        self.t2_seg_native = LazyImage(join(self.dir_local, f'hyperashs_seg_{side}_to_t2orig.nii.gz'))

        # The input path for INR training is different since INR code expects a certain directory structure
        self.dir_inr_train_input = inr_path
        
        # For training/INR, the primary and secondary modalities cropped at native resolution
        self.input_seg = LazyImage(join(self.dir_local, nm.seg))
        self.inr_primary = LazyImage(join(self.dir_local, nm.inr_primary))
        self.inr_primary_mask = LazyImage(join(self.dir_local, 'primary_mask.nii.gz'))
        self.inr_primary_gt = LazyImage(join(self.dir_local, 'primary_gt.nii.gz'))
        self.inr_secondary = LazyImage(join(self.dir_local, nm.inr_secondary))
        self.inr_secondary_mask = LazyImage(join(self.dir_local, 'secondary_mask.nii.gz'))
        self.inr_secondary_gt = LazyImage(join(self.dir_local, 'secondary_gt.nii.gz'))
        self.inr_seg = LazyImage(join(self.dir_local, nm.inr_seg))
        self.inr_inference_mask = LazyImage(join(self.dir_local, 'inr_inference_mask.nii.gz'))
        
        # After INR upsampling is done, the upsampled segmentation is resampled into the space of the
        # T2 hyper-resolution patch.
        self.t2_patch_hyperres_seg = LazyImage(join(self.dir_local, nm.hyper_primary_seg))
                
class ASHSExperimentBase:
    def __init__(self, config: Dict[str, Any], case_path:str, nm: SimpleNamespace, 
                 sides=['left', 'right'], subject:str|None=None, date:str|None=None,
                 inr_path: Dict[str,str]|None=None, 
                 nnunet_train_id: Dict[str,int]|None=None):
        self.config = config
        self.case_path = case_path
        self.inr_path = inr_path
        self.nnunet_train_id = nnunet_train_id
        self.test_folder = 'Dataset{}_{}'.format(config['EXP_NUM'], config['MODEL_NAME'])
        self.gpe = GlobalPipelineElements(case_path, nm)
        self.lpe = { side: LocalPipelineElements(case_path, side, self.test_folder, nm, 
                                                 inr_path=inr_path[side] if inr_path is not None else None,
                                                 nnunet_train_id=nnunet_train_id[side] if nnunet_train_id is not None else None) 
                    for side in sides }
        self.tpe = TemplatePipelineElements(config['TEMPLATE_PATH'], nm)
        self.subject = subject
        self.date = date
        self.qc_title = f'{subject} - {date}' if (subject and date) else f'{subject}' if subject else ''

