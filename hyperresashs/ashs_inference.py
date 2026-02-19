import os
from os.path import join
from .utils.upsample_inr_method import create_link
from .utils.upsample_linear_method import linear_isotropic_upsampling, pad_image_with_world_alignment_in_memory
from .utils.trim_neck import trim_neck_in_memory
from picsl_greedy import Greedy3D
from picsl_c3d import Convert3D
import yaml
from types import SimpleNamespace
from batchgenerators.utilities.file_and_folder_operations import *
import torch
import time
import shutil
import tempfile
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Protocol, Dict, Literal, Type, Callable, Any



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
    def data(self) -> Any:
        """Read the object from disk if it hasn't been read yet, and return it. Returns None if the file does not exist."""
        if self._data is None and os.path.exists(self.filename):
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
        

# Normalize a simple ITK image to 0-255 range
def normalize_intensity(img: sitk.Image, qtile=0.995) -> sitk.Image:
    c3d = Convert3D()
    c3d.push(img)
    c3d.execute(f'-stretch 0 {qtile*100}% 0 255 -clip 0 255')
    return c3d.peek(-1)

# Crop segmentation by margin
def trim_segmentation(seg: sitk.Image, margin_mm:float=10) -> sitk.Image:
    c3d = Convert3D()
    c3d.push(seg)
    c3d.execute(f'-trim {margin_mm}mm')
    return c3d.peek(-1)

# Crop to segmentation and normalize intensity of images
def normalize_and_reslice(img: sitk.Image, ref_img: sitk.Image, qtile=0.995) -> sitk.Image:
    c3d = Convert3D()
    c3d.push(ref_img)
    c3d.push(img)
    c3d.execute(f'-stretch 0 {qtile*100}% 0 255 -clip 0 255 -reslice-identity')
    return c3d.peek(-1)

# Overlay segmentation on top of an image with transparency for QC visualization
def overlay_segmentation_on_image(image: sitk.Image, segmentation: sitk.Image, label_file: str, alpha=0.5) -> sitk.Image:
    
    c3d = Convert3D()
    c3d.push(image)
    c3d.push(segmentation)
    c3d.execute(f'-oli {label_file} {alpha}')
    return sitk.Compose([c3d.peek(i) for i in range(3)])


def generate_ashs_qc(
    volumes: Dict[str, sitk.Image],
    row_labels: Dict[str, str],
    output_path: str,
    title: str|None = None):
    
    # Initialize c3d converter
    c3d = Convert3D()
    
    # Define slice specifications (percentage positions)
    cor_slice_spec = np.linspace(0.0, 1.0, 7)[1:-1] * 100    
    sag_slice_spec = np.linspace(0.0, 1.0, 5)[1:-1] * 100   
    
    # Normalize each of the images
    def normalize(img):
        c3d.push(img)
        c3d.execute('-stretch 0 99.5% 0 255 -clip 0 255 -type uchar')
        return c3d.peek(-1)
    
    # Function to extract slices using c3d
    def extract_slices(img, raicode, slice_spec):
        # Extract the separate components
        n_comp = img.GetNumberOfComponentsPerPixel()
        i_comp = [img] if n_comp == 1 else [ sitk.VectorIndexSelectionCast(img, i) for i in range(n_comp) ]

        # Extract the slices for each component separately
        slices = []
        slice_spec_str = ','.join([f'{pct}%' for pct in slice_spec])
        for i, comp_img in enumerate(i_comp):
            c3d.execute('-clear')
            c3d.push(comp_img)
            c3d.execute(f'-swapdim {raicode} -slice z {slice_spec_str}')
            slices.append([ c3d.peek(i) for i in range(len(slice_spec)) ])
        
        # Slices are now grouped by component, then slice
        return [ sitk.Compose([ slices[c][i] for c in range(n_comp) ]) for i in range(len(slice_spec)) ]        
    
    # Extract the slices in coronal and sagittal views
    slices_cor = { k: extract_slices(v, 'RSA', cor_slice_spec) for k,v in volumes.items() }
    slices_sag = { k: extract_slices(v, 'ASR', sag_slice_spec) for k,v in volumes.items() }
            
    # Create the montage using matplotlib
    n_cor = len(cor_slice_spec)
    n_sag = len(sag_slice_spec)
    n_cols = n_cor + n_sag
    n_rows = len(volumes)
    
    # Compute the extents of different slices
    k1 = next(iter(volumes.keys()))
    w_cor = slices_cor[k1][0].GetSpacing()[0] * slices_cor[k1][0].GetSize()[0]
    w_sag = slices_sag[k1][0].GetSpacing()[0] * slices_sag[k1][0].GetSize()[0]
    h_cor = slices_cor[k1][0].GetSpacing()[1] * slices_cor[k1][0].GetSize()[1]
    aspect = n_rows * h_cor / (n_cor * w_cor + n_sag * w_sag)
    
    fig = plt.figure(figsize=(16, 16 * aspect))
    gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.02, hspace=0.02, 
                  width_ratios=[w_cor]*n_cor + [w_sag]*n_sag)
    
    def plot_slice(ax, img, label=None):
        arr = sitk.GetArrayFromImage(img).squeeze().astype(np.uint8)
        ext = np.array(img.GetSize()) * np.array(img.GetSpacing())
        if img.GetNumberOfComponentsPerPixel() == 1:
           ax.imshow(arr, cmap='gray', aspect=1, extent=[0, ext[0], 0, ext[1]])
        else:
           ax.imshow(arr, aspect=1, extent=[0, ext[0], 0, ext[1]])
        ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.6)
        ax.set_xticks(np.linspace(0, ext[0], 7))
        ax.set_yticks(np.linspace(0, ext[1], 7))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params('both', length=0, width=1, which='major')
        if label:
            ax.text(0.02, 0.98, label, transform=ax.transAxes, 
                fontsize=10, va='top', ha='left', color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    # Row 0: Template
    for i, (k,lab) in enumerate(row_labels.items()):
        for j, slice_img in enumerate(slices_cor[k]):
            ax = fig.add_subplot(gs[i, j])
            plot_slice(ax, slice_img, lab if j == 0 else None)
        
        for j, slice_img in enumerate(slices_sag[k]):
            ax = fig.add_subplot(gs[i, n_cor + j])
            plot_slice(ax, slice_img)
            
    # Add title
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    # Save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=96, bbox_inches='tight', facecolor='white', edgecolor='black', pad_inches=0.1)
    plt.close(fig)
    
    
def generate_ashs_segmentation_qc(
    seg: sitk.Image, 
    t1: sitk.Image, 
    t2:sitk.Image, 
    label_file: str, 
    output_path: str,
    title: str|None = None,
    trim_margin_mm: float = 3.0):
    
    seg = trim_segmentation(seg, trim_margin_mm)
    t2 = normalize_and_reslice(t2, seg)
    t1 = normalize_and_reslice(t1, seg)
    
    # Deal here with missing labelfile
    fn_cleanup = None
    if label_file is None:
        print("Warning: No label file provided for QC generation, using default colors")
        # Create a temporary label file with random colors
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.txt') as tmp_label_file:
            # Get all the unique labels in the segmentation
            arr = sitk.GetArrayFromImage(seg)
            labels = np.unique(arr)
            
            # Cycle through the matplotlib tab20 colormap for colors
            cmap = plt.get_cmap('tab20')
            for i, label in enumerate(labels):
                if label == 0:
                    color, alpha = (0,0,0), 0.0  # Background is transparent
                else:
                    color, alpha = [int(c*255) for c in cmap(i % 20)[:3]], 1
                tmp_label_file.write(f'{label} {color[0]} {color[1]} {color[2]} {alpha} {alpha} {alpha} "Label {label}"\n')
            label_file = tmp_label_file.name    
            fn_cleanup = label_file  # Remember to delete this file later
    
    vols = {'t2': t2, 't1': t1, 
            't2s': overlay_segmentation_on_image(t2, seg, label_file), 
            't1s': overlay_segmentation_on_image(t1, seg, label_file)}
    
    generate_ashs_qc(vols, 
                     row_labels={'t2': 'T2 (native)', 't1': 'T1 to T2', 't2s': 'T2-seg', 't1s': 'T1-seg'},
                     output_path=output_path, title=title)
    
    # Delete the temporary label file if we created one
    if fn_cleanup is not None:
        os.remove(fn_cleanup)
    

def generate_ashs_registration_qc(
    template_img: sitk.Image,
    t1_to_t2: sitk.Image,
    t2_img: sitk.Image,
    output_path: str,
    title: str|None = None):
    
    vols = {'t2': normalize_intensity(t2_img),
            't1': normalize_intensity(t1_to_t2),
            'temp': normalize_intensity(template_img)}

    generate_ashs_qc(vols, 
                     row_labels={'t2': 'T2 (native)', 't1': 'T1 to T2', 'temp': 'Template to T2'},
                     output_path=output_path, title=title)
    

class Timer:
    def __init__(self) -> None:
        self.t_elapsed = 0
        self.n = 0
        
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        
    def start(self):
        self.start_time = time.time()
        
    def stop(self):
        if self.start_time is None:
            raise RuntimeError("Timer was not started.")
        self.t_elapsed += time.time() - self.start_time
        self.n += 1
        self.start_time = None
        
    @property
    def total(self):
        return self.t_elapsed
    
    @property
    def average(self):
        return self.t_elapsed / self.n if self.n > 0 else np.NAN
    
    
# Define a signature for the callback function that can be used to report progress and send 
# updates to the caller during processing. The callback function is meant to be used in 
# conjunction with ITK-SNAP DSS, but not limited to it
class ProgressCallbackType(Protocol):
    def __call__(self, 
                 progress: float|None=None, 
                 progress_chunk_start: float=0.0, 
                 progress_chunk_end: float=1.0, 
                 attachments: Dict[str,str]|None = None,
                 message: str|None = None) -> None: 
        ...
        
def default_progress_callback(progress: float|None=None,
                              progress_chunk_start: float=0.0, 
                              progress_chunk_end: float=1.0, 
                              attachments: Dict[str,str]|None = None,
                              message: str|None = None) -> None:
        pass
    

class HyperASHSInference():
    def __init__(self, config):
        self.config = config

        with open(config['FILE_NAME_CONFIG']) as f:
            settings = yaml.safe_load(f)
            self.nm = SimpleNamespace(**settings)

        self.test_path = config['TEST_PATH']
        self.test_folder = 'Dataset{}_{}'.format(config['EXP_NUM'], config['MODEL_NAME'])
        self.upsampling_method = config['UPSAMPLING_METHOD']
        self.dataset_id = config['EXP_NUM']
        self.trainer = config['TRAINER']

        # ROI determination (template path)
        self.template_3tt1 = join(config['TEMPLATE_PATH'], self.nm.template)
        self.template_roi_left = join(config['TEMPLATE_PATH'], self.nm.left_roi_file)
        self.template_roi_right = join(config['TEMPLATE_PATH'], self.nm.right_roi_file)
        
        # Number of threads for Greedy
        self.greedy_num_threads = config.get('GREEDY_NUM_THREADS', 0)
        
        # Optional cropping applied to the T2 image before registration with T1. Cropping
        # in the coronal plane helps with registration speed and accuracy.
        self.t2_cropping = config.get('ASHS_TSE_REGION_CROP', 0.2)
        
        # Optional ITK-SNAP label file
        self.label_file = config.get('ITKSNAP_LABEL_FILE')
        print(f"Using label file for QC visualization: {self.label_file}")

    def download_model_from_huggingface(self, hf_repo_id, target_path):
        try:
            from huggingface_hub import snapshot_download
            import requests
        except ImportError:
            raise ImportError("huggingface_hub is not installed. Please install it with: pip install huggingface_hub")
        
        print(f"Model not found locally. Downloading from Hugging Face: {hf_repo_id}")
        
        os.makedirs(target_path, exist_ok=True)
        temp_dir = tempfile.mkdtemp()
        try:
            print(f"Downloading model to temporary location: {temp_dir}")
            
            disable_ssl = os.environ.get('HF_HUB_DISABLE_SSL_VERIFY', '0').lower() in ('1', 'true', 'yes')
            original_request = None
            if disable_ssl:
                print("warning: ssl verification is disabled (hf_hub_disable_ssl_verify=1)")
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                original_request = requests.Session.request
                def patched_request(self, method, url, **kwargs):
                    kwargs['verify'] = False
                    return original_request(self, method, url, **kwargs)
                requests.Session.request = patched_request # type: ignore
            
            try:
                downloaded_path = snapshot_download(
                    repo_id=hf_repo_id,
                    local_dir=temp_dir,
                    repo_type="model"
                )
            except Exception as ssl_error:
                error_str = str(ssl_error).lower()
                if 'ssl' in error_str or 'certificate' in error_str or 'cert' in error_str:
                    if original_request:
                        requests.Session.request = original_request
                    raise RuntimeError(
                        f"ssl certificate verification failed. this often happens in corporate networks with proxies.\n"
                        f"to disable ssl verification (less secure), set environment variable:\n"
                        f"  export HF_HUB_DISABLE_SSL_VERIFY=1\n"
                        f"then run your command again.\n"
                        f"original error: {str(ssl_error)}"
                    )
                raise
            finally:
                if original_request:
                    requests.Session.request = original_request
            
            downloaded_contents = os.listdir(temp_dir)
            
            model_folder = None
            for item in downloaded_contents:
                item_path = join(temp_dir, item)
                if os.path.isdir(item_path):
                    if os.path.exists(join(item_path, 'dataset.json')) and os.path.exists(join(item_path, 'plans.json')):
                        model_folder = item_path
                        break
            
            if model_folder is None:
                if os.path.exists(join(temp_dir, 'dataset.json')) and os.path.exists(join(temp_dir, 'plans.json')):
                    model_folder = temp_dir
                else:
                    raise ValueError(f"Could not find model structure in downloaded repository. Contents: {downloaded_contents}")
            
            parent_dir = os.path.dirname(target_path)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            
            for item in os.listdir(model_folder):
                src = join(model_folder, item)
                dst = join(target_path, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
            
            required_files = ['dataset.json', 'plans.json']
            for req_file in required_files:
                if not os.path.exists(join(target_path, req_file)):
                    raise ValueError(f"Required file {req_file} not found in downloaded model")
            
            fold_found = False
            for item in os.listdir(target_path):
                if item.startswith('fold_') and os.path.isdir(join(target_path, item)):
                    if os.path.exists(join(target_path, item, 'checkpoint_final.pth')):
                        fold_found = True
                        break
            
            if not fold_found:
                raise ValueError("No fold checkpoints found in downloaded model")
            
            print(f"Successfully downloaded and installed model to: {target_path}")
            
        except Exception as e:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(f"Failed to download model from Hugging Face ({hf_repo_id}): {str(e)}")
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def resample_test_with_date(self, subject_id=None):
        subject_list = os.listdir(self.test_path)
        if subject_id:
            if subject_id not in subject_list:
                raise ValueError(f'subject_id {subject_id} not found in test path {self.test_path}')
            subject_list = [subject_id]
        
        for subject_ in subject_list:
            subject_path = join(self.test_path, subject_)
            date_list = os.listdir(subject_path)

            for date_ in date_list:
                date_path = join(subject_path, date_)

                self.run_inference_for_one_case(date_path, subject=subject_, date=date_)
    
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
        
    class LocalPipelineElements:
        def __init__(self, case_path:str, side:str, test_folder: str, nm: SimpleNamespace):
            # Folders
            self.dir_local = join(case_path, test_folder, side)
            self.dir_nnunet_input = join(self.dir_local, 'input')
            self.dir_nnunet_output = join(self.dir_local, 'output')
            
            # Images and matrices generated during preprocessing
            self.roi_in_t1_space = LazyImage(join(case_path, nm.global_roi_in_3tt1_XYZ.replace('XYZ', side)))
            self.fn_save_mat_path_t2_to_t1_local = join(self.dir_local, nm.reg_mat)
            self.t2_patch_native = LazyImage(join(self.dir_local, nm.hyper_primary))
            self.t1_patch_warped = LazyImage(join(self.dir_local, nm.hyper_secondary_after_registertion))
            self.fn_registration_qc = join(self.dir_local, f'registration_qc.png')
            
            # NNUNet input and output
            self.hl_nnunet_t2_input = join(self.dir_nnunet_input, 'MTL_000_0000.nii.gz')
            self.hl_nnunet_t1_input = join(self.dir_nnunet_input, 'MTL_000_0001.nii.gz')
            self.nnunet_seg = LazyImage(join(self.dir_nnunet_output, 'MTL_000.nii.gz'))
            self.fn_segmentation_qc = join(self.dir_local, f'segmentation_qc.png')
            
            # Final post-processed segmentation
            self.t2_seg_native = LazyImage(join(self.dir_local, f'hyperashs_seg_{side}_to_t2orig.nii.gz'))
                    
    def get_pipeline_elements(self, case_path:str):
        return self.GlobalPipelineElements(case_path, self.nm), { side: self.LocalPipelineElements(case_path, side, self.test_folder, self.nm) for side in ['left', 'right'] }
    
    def run_inference_for_one_case(self, case_path, subject:str|None=None, date:str|None=None,
                                   save_intermediates: bool = True, overwrite_existing: bool = False,
                                   callback: ProgressCallbackType = default_progress_callback):
   
        nt = self.greedy_num_threads
        
        # Generate a title for QC images based on subject and date if provided
        qc_title = f'{subject} - {date}' if (subject and date) else f'{subject}' if subject else ''
        
        # create the folder for hyper-resolution inference
        hyper_test_path = join(case_path, self.test_folder)
        os.makedirs(hyper_test_path, exist_ok=True)
        
        # Initialize the pipeline elements
        gpe, lpe = self.get_pipeline_elements(case_path)
    
        # Create timers for all elements of the pipeline            
        tm_all, tm_neck, tm_reg_t1_t2_whole, tm_reg_t1_temp, tm_reg_t1_t2_local, tm_nnunet, tm_finalize = Timer(), Timer(), Timer(), Timer(), Timer(), Timer(), Timer()
        with tm_all:
            
            # Perform neck trimming if necessary
            if overwrite_existing or not gpe.t1_neck_trim.exists():
                with tm_neck:
                    gpe.t1_neck_trim.data = trim_neck_in_memory(gpe.t1_native.data, verbose=True)
                        
            callback(progress=0.05, message=f"Neck trimming completed in {tm_neck.total:.1f} s.")
                    
            # Check if nnUNet inputs already exist, and if not, perform the registration steps
            if overwrite_existing or not all(lp.t2_patch_native.exists() and lp.roi_in_t1_space.exists() for lp in lpe.values()):
                
                # ------- global T1 to T2 registration using ASHS pipeline -------
                with tm_reg_t1_t2_whole:

                    # If specified, crop the T2 image before registration      
                    t2_cropped_img = gpe.t2_whole_img.data
                    if self.t2_cropping > 0:
                        # Create the cropping command for c3d based on the specified cropping fraction
                        c3d  = Convert3D()
                        c3d.push(gpe.t2_whole_img.data)
                        c = self.t2_cropping * 100
                        c3d.execute(f'-swapdim RSA -region {c}x{c}x0% {100-2*c}x{100-2*c}x100%')
                        t2_cropped_img = c3d.peek(-1)
                    
                    # Perform the affine registration
                    g = Greedy3D()
                    g.execute(f'-threads {nt} -z -a -dof 6 -ia-identity -m NMI '
                            f'-i t2 t1 -n 100x100x10 -o {gpe.fn_save_mat_path_t2_to_t1_global} ', 
                            t2=t2_cropped_img, t1=gpe.t1_neck_trim.data)
                    
                    # Apply the registration 
                    g.execute(f'-threads {nt} -rf t2 -rm t1 t1_reg_to_t2 '
                            f'-r {gpe.fn_save_mat_path_t2_to_t1_global}', t1_reg_to_t2=None)
                    
                    if save_intermediates:
                        gpe.t1_reg_to_t2.data = g['t1_reg_to_t2']
                
                # ------- global T1 to template registration using ASHS pipeline -------
                with tm_reg_t1_temp:
                    template_3tt1 = sitk.ReadImage(self.template_3tt1)

                    # 1. rigid
                    g.execute(f'-threads {nt} -a -dof 6 -m NCC 2x2x2 '
                            f"-i template_3tt1 trim_t1_image "
                            f"-o rigid -n 400x0x0x0 "
                            f"-ia-image-centers -search 400 5 5", 
                            template_3tt1=template_3tt1, trim_t1_image=gpe.t1_neck_trim.data, rigid=None)
                    
                    # 2. affine
                    g.execute(f'-threads {nt} -a -m NCC 2x2x2 '
                            f'-i template_3tt1 trim_t1_image '
                            f'-o {gpe.fn_template_to_3tt1_affine_matrix} -n 400x80x40x0 '
                            f'-ia rigid')
                    
                    # 3. deformable
                    g.execute(f'-threads {nt} -m NCC 2x2x2 -e 0.5 -n 60x20x0 -sv '
                            f'-i template_3tt1 trim_t1_image -it {gpe.fn_template_to_3tt1_affine_matrix} '
                            f'-o warpfwd -oinv warpinv', 
                            warpfwd=None, warpinv=None)
                    
                    # 4. apply
                    g.execute(f"-threads {nt} -rf trim_t1_image "
                            f"-rm template_3tt1 template_to_3tt1 "
                            f"-rm {self.template_roi_left} t1_roi_left "
                            f"-rm {self.template_roi_right} t1_roi_right "
                            f"-r {gpe.fn_template_to_3tt1_affine_matrix},-1 warpinv", 
                            template_to_3tt1=None, t1_roi_left=None, t1_roi_right=None)
                    
                    # Read off the ROI images
                    t1_roi = { 'left': g['t1_roi_left'], 'right': g['t1_roi_right'] }

                    if save_intermediates:
                        gpe.template_to_3tt1.data = g['template_to_3tt1']
                        for k, v in t1_roi.items():
                            lpe[k].roi_in_t1_space.data = v
            
                # ------- Perform the cropping based on the ROIs  ------- 
                with tm_reg_t1_t2_local:
                    # Pad the T2 image with world alignment
                    t2_padded_img = pad_image_with_world_alignment_in_memory(t2_cropped_img, [40, 40, 40], [40, 40, 40])
                    
                    for side_, lp in lpe.items():
                                                
                        # Determine the target spacing for the T2 upsampling (replace the largest spacing with the second largest one)
                        t2_original_spacing = np.array(t2_cropped_img.GetSpacing())
                        spc_order = np.argsort(t2_original_spacing)
                        spc_sorted = np.array([t2_original_spacing[i] for i in spc_order])
                        spc_var = np.array([
                            np.std(np.array([spc_sorted[0], spc_sorted[1], spc_sorted[2] / (k+1)])) for k in range(10)])
                        scaling = np.ones(3)
                        scaling[spc_order[-1]] = np.argmin(spc_var)+1
                        scaling_str = 'x'.join([f'{100*s}' for s in scaling]) + '%'
                        print(f'Original T2 spacing: {t2_original_spacing}, Scaling factors: {scaling_str}')              
                                        
                        # Crop the T2 using the T1 ROI and apply the new spacing
                        c3d = Convert3D()
                        c3d.push(t2_padded_img)
                        c3d.push(t1_roi[side_])
                        c3d.execute(f'-popas ROI_T1 -as T2 -push ROI_T1 -reslice-matrix {gpe.fn_save_mat_path_t2_to_t1_global} -trim 5vox '
                                    f'-resample {scaling_str} -as ROI_T2 -dup -push T2 -reslice-identity -swapdim RPI')
                        lp.t2_patch_native.data = c3d.peek(-1)
                        roi_t2 = c3d.peek(-2)

                        # Write out the cropped T2 image and create links for nnUNet input
                        create_link(lp.t2_patch_native.filename, lp.hl_nnunet_t2_input)
                                        
                        # Compute local registration between T2 and T1
                        g.execute(f'-threads {nt} -z -a -dof 6 -ia {gpe.fn_save_mat_path_t2_to_t1_global} -m NMI '
                                f'-i t2 t1 -gm mask -n 100x50 -o {lp.fn_save_mat_path_t2_to_t1_local} ', 
                                t2=lp.t2_patch_native.data, t1=gpe.t1_neck_trim.data, mask=roi_t2)
                        
                        # Apply the registration to the T1 and resample it into the isotropic space
                        g.execute(f'-threads {nt} -rf t2 -rm t1 t1_reg_to_t2 '
                                f'-r {lp.fn_save_mat_path_t2_to_t1_local}', t1_reg_to_t2=None)
                        lp.t1_patch_warped.data = g['t1_reg_to_t2']

                        # Write out the cropped T1 and create link for nnUNet input
                        create_link(lp.t1_patch_warped.filename, lp.hl_nnunet_t1_input)
                        
                        # For QC purposes, map template all the way to T2 space
                        g.execute(f"-threads {nt} -rf t2 -rm template_3tt1 template_to_t2 "
                                  f"-r {lp.fn_save_mat_path_t2_to_t1_local} {gpe.fn_template_to_3tt1_affine_matrix},-1 warpinv",
                                  template_to_t2=None)

                        # Generate registration QC screenshot
                        generate_ashs_registration_qc(
                            template_img=g['template_to_t2'],
                            t1_to_t2=lp.t1_patch_warped.data,
                            t2_img=lp.t2_patch_native.data,
                            output_path=lp.fn_registration_qc,
                            title=f"{qc_title} Registration QC - {side_.capitalize()}")
            
            callback(progress=0.25,
                     attachments={f'{side_.capitalize()} Registration QC': lp.fn_registration_qc for side_, lp in lpe.items()},
                     message=f"Registration and ROI cropping completed in {tm_reg_t1_t2_whole.total + tm_reg_t1_temp.total + tm_reg_t1_t2_local.total:.1f} s.")

            # ------- nnunet inference -------
            with tm_nnunet:
                for i_side_, (side_, lp) in enumerate(lpe.items()):

                    # command
                    print(f'start running inference for {lp.dir_local}')
                    start = time.time()

                    # nnUNet prediction
                    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
                    from batchgenerators.utilities.file_and_folder_operations import load_json
                    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
                    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
                    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

                    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                    predictor = nnUNetPredictor(verbose=True, device=device)
                    nnunet_model = join(os.environ.get("nnUNet_results",""), 
                                        f"Dataset{self.config['EXP_NUM']}_{self.config['MODEL_NAME']}", f"{self.config['TRAINER']}__nnUNetPlans__3d_fullres")
                    
                    dataset_json_path = join(nnunet_model, 'dataset.json')
                    plans_json_path = join(nnunet_model, 'plans.json')
                    model_complete = (os.path.exists(dataset_json_path) and 
                                    os.path.exists(plans_json_path))
                    
                    if model_complete:
                        fold_folders = [item for item in os.listdir(nnunet_model) 
                                        if item.startswith('fold_') and os.path.isdir(join(nnunet_model, item))]
                        if fold_folders:
                            for fold_folder in fold_folders:
                                checkpoint_path = join(nnunet_model, fold_folder, 'checkpoint_final.pth')
                                if not os.path.exists(checkpoint_path):
                                    model_complete = False
                                    break
                        else:
                            model_complete = False
                    
                    if not model_complete:
                        hf_repo_id = self.config.get('HF_MODEL_REPO')
                        if hf_repo_id:
                            print(f"Model not found or incomplete at {nnunet_model}, downloading from Hugging Face...")
                            self.download_model_from_huggingface(hf_repo_id, nnunet_model)
                        else:
                            raise FileNotFoundError(
                                f"Model not found or incomplete at {nnunet_model} and HF_MODEL_REPO not specified in config. "
                                f"Please either:\n"
                                f"  1. Place the trained model at the expected path, or\n"
                                f"  2. Add HF_MODEL_REPO to your config file to enable automatic download from Hugging Face."
                            )
                    
                    use_folds = predictor.auto_detect_available_folds(nnunet_model, 'checkpoint_final.pth')
                    dataset_json = load_json(join(nnunet_model, 'dataset.json'))
                    plans = load_json(join(nnunet_model, 'plans.json'))
                    plans_manager = PlansManager(plans)

                    parameters = []
                    configuration_name = None
                    inference_allowed_mirroring_axes = None
                    for i, f in enumerate(use_folds):
                        f = int(f) if f != 'all' else f
                        checkpoint = torch.load(join(nnunet_model, f'fold_{f}', 'checkpoint_final.pth'),
                                                map_location=torch.device('cpu'), weights_only=False)
                        if i == 0:
                            configuration_name = checkpoint['init_args']['configuration']
                            inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                                'inference_allowed_mirroring_axes' in checkpoint.keys() else None

                        parameters.append(checkpoint['network_weights'])
                        
                    configuration_manager = plans_manager.get_configuration(configuration_name)

                    # restore network
                    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
                    network = nnUNetTrainer.build_network_architecture(
                        configuration_manager.network_arch_class_name,
                        configuration_manager.network_arch_init_kwargs,
                        configuration_manager.network_arch_init_kwargs_req_import,
                        num_input_channels,
                        plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
                        enable_deep_supervision=False
                    )

                    predictor.manual_initialization(
                        network, plans_manager, configuration_manager, parameters,
                        dataset_json, self.config['TRAINER'],
                        inference_allowed_mirroring_axes)

                    # predictor.initialize_from_trained_model_folder(nnunet_model, None)
                    predictor.predict_from_files(
                        lp.dir_nnunet_input,
                        lp.dir_nnunet_output,
                        save_probabilities=False,
                        overwrite=True, num_processes_preprocessing=1, num_processes_segmentation_export=1)

                    end = time.time()  # end counting the time
                    elapsed_time = end - start
                    print(f'Inference time for {lp.dir_local}: {elapsed_time:.2f} seconds')
                    with open(join(lp.dir_nnunet_output, "elapsed_time.txt"), "w") as f_:
                        f_.write(str(elapsed_time))
                        
                    # Generate a QC screenshot for the nnUNet output
                    generate_ashs_segmentation_qc(
                        seg = lp.nnunet_seg.data,
                        t1 = lp.t1_patch_warped.data,
                        t2 = lp.t2_patch_native.data,
                        label_file = self.label_file,
                        output_path = lp.fn_segmentation_qc,
                        title = f"{qc_title} Segmentation QC - {side_.capitalize()}")
                        
                    # Report being done with part of the nnUNet processing 
                    callback(progress=0.5 * (i_side_+1), 
                             message=f"nnUNet Inference for {side_} completed in {elapsed_time:.1f} s.",
                             attachments={f'{side_.capitalize()} Segmentation QC': lp.fn_segmentation_qc},
                             progress_chunk_start=0.25, progress_chunk_end=0.95)
                        
            with tm_finalize:

                # Create a new greedy instance for the registration
                g = Greedy3D()
                
                # Apply resampling to original T2 space for each side
                for side_, lp in lpe.items():
                    g.execute(f'-threads {nt} -rf t2 -ri LABEL 0.2vox '
                              f'-rm final_seg {lp.t2_seg_native.filename} -r ',
                              t2=gpe.t2_whole_img.data, final_seg=lp.nnunet_seg.data)
                    
            callback(progress=1.0, message=f"Post-processing completed in {tm_finalize.total:.1f} s.")
                    
        print(f'ASHS inference completed for case: {case_path}')
        print(f'  Time in neck trimming: {tm_neck.total:.2f}s')
        print(f'  Time in global T1/T2 reg: {tm_reg_t1_t2_whole.total:.2f}s')
        print(f'  Time in T1/template reg: {tm_reg_t1_temp.total:.2f}s')
        print(f'  Time in local T1/T2 reg: {tm_reg_t1_t2_local.total:.2f}s')
        print(f'  Time in nnUNet: {tm_nnunet.total:.2f}s')
        print(f'  Total time: {tm_all.total:.2f}s')

    def execute(self, subject_id=None):
        # --- run inference
        self.resample_test_with_date(subject_id=subject_id)