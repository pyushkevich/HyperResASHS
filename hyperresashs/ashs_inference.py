import os
from os.path import join
from .ashs_exp import ASHSExperimentBase
from .ashs_preproc import ASHSProcessor, generate_ashs_segmentation_qc, ProgressCallbackType, default_progress_callback, Timer
import yaml
from types import SimpleNamespace
import torch
import time
import shutil
import tempfile

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
       
        # Number of threads for Greedy
        self.greedy_num_threads = config.get('GREEDY_NUM_THREADS', 0)
        
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
    

    def run_inference_for_one_case(self, case_path, subject:str|None=None, date:str|None=None,
                                   save_intermediates: bool = True, overwrite_existing: bool = False,
                                   callback: ProgressCallbackType = default_progress_callback, 
                                   device: str|None = None):
        
        # Create the ASHS experiment representation
        exp = ASHSExperimentBase(self.config, case_path, self.nm, subject=subject, date=date)

        # Create a preprocessing/registration worker
        reg = ASHSProcessor(self.config, 
                            overwrite_existing=overwrite_existing, 
                            save_intermediates=save_intermediates) 

        # create the folder for hyper-resolution inference
        hyper_test_path = join(case_path, self.test_folder)
        os.makedirs(hyper_test_path, exist_ok=True)
        
        # Create timers for all elements of the pipeline            
        tm_all, tm_nnunet = Timer(), Timer()
        with tm_all:
            
            # Execute the registration and preprocessing steps (neck trimming, global and local registration, ROI cropping)
            reg.preprocess(exp, callback=callback, progress_range=(0.0, 0.25))

            # ------- nnunet inference -------
            with tm_nnunet:
                for i_side_, (side_, lp) in enumerate(exp.lpe.items()):

                    # command
                    print(f'start running inference for {lp.dir_local}')
                    start = time.time()

                    # nnUNet prediction
                    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
                    from batchgenerators.utilities.file_and_folder_operations import load_json
                    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
                    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
                    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

                    torch_device = torch.device(device) if device else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
                    predictor = nnUNetPredictor(verbose=True, device=torch_device)
                    
                    # Check if the path to the model is specified in config
                    nnunet_model = self.config.get('MODEL_PATH')
                    if not nnunet_model:                    
                    
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
                        t1 = lp.t1_patch_warped_hyperres.data,
                        t2 = lp.t2_patch_hyperres.data,
                        label_file = self.label_file,
                        output_path = lp.fn_segmentation_qc,
                        title = f"{exp.qc_title} Segmentation QC - {side_.capitalize()}")
                        
                    # Report being done with part of the nnUNet processing 
                    callback(progress=0.5 * (i_side_+1), progress_range=(0.25, 0.75),
                             message=f"nnUNet Inference for {side_} completed in {elapsed_time:.1f} s.",
                             attachments={f'{side_.capitalize()} Segmentation QC': lp.fn_segmentation_qc})
            
            # ------- final post-processing -------
            reg.postprocess(exp, callback=callback, progress_range=(0.95, 1.0))
                    
        print(f'ASHS inference completed for case: {case_path}')
        print(f'  Time in neck trimming: {reg.tm_neck.total:.2f}s')
        print(f'  Time in global T1/T2 reg: {reg.tm_reg_t1_t2_whole.total:.2f}s')
        print(f'  Time in T1/template reg: {reg.tm_reg_t1_temp.total:.2f}s')
        print(f'  Time in local T1/T2 reg: {reg.tm_reg_t1_t2_local.total:.2f}s')
        print(f'  Time in nnUNet: {tm_nnunet.total:.2f}s')
        print(f'  Total time: {tm_all.total:.2f}s')

    def execute(self, subject_id=None):
        # --- run inference
        self.resample_test_with_date(subject_id=subject_id)