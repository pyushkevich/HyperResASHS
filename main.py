import yaml
import argparse
from pathlib import Path
import os
from preprocessing import *
from testing import *
from prepare_inr import INRPreprocess


def search_config_name(config_id):
    curr_path = Path.cwd()
    searched_file = None

    for folder_name in ['config', 'config_test']:
        config_path = os.path.join(curr_path, folder_name)
        if not os.path.exists(config_path):
            continue

        file_list = os.listdir(config_path)
        for file_ in file_list:
            if int(file_.split('_')[1]) == config_id:
                return os.path.join(config_path, file_)

    if searched_file == None:
        raise ValueError('There is no config file with id {}'.format(config_id))


def extract_id_from_filename(filename):
    basename = os.path.basename(filename)
    parts = basename.split('_')
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def check_id_in_existing_configs(config_id, config_file_path):
    curr_path = Path.cwd()
    config_file_abs_path = Path(config_file_path).resolve()
    
    for folder_name in ['config', 'config_test']:
        config_path = os.path.join(curr_path, folder_name)
        if not os.path.exists(config_path):
            continue

        file_list = os.listdir(config_path)
        for file_ in file_list:
            if not file_.endswith('.yaml'):
                continue
            
            existing_file_path = Path(config_path) / file_
            if not existing_file_path.is_file():
                continue
                
            existing_file_abs_path = existing_file_path.resolve()
            
            if existing_file_abs_path == config_file_abs_path:
                continue
            
            parts = file_.split('_')
            if len(parts) < 2:
                continue
            
            try:
                existing_id = int(parts[1])
            except ValueError:
                continue
            
            if existing_id == config_id:
                raise ValueError(f'config id {config_id} already exists in {folder_name}/{file_}. please rename your config file to use a different id.')


def check_nnunet_dataset_exists(config_id, nnunet_raw_path):
    paths_to_check = []
    
    if nnunet_raw_path and os.path.exists(nnunet_raw_path):
        paths_to_check.append(nnunet_raw_path)
    
    env_nnunet_raw = os.environ.get('nnUNet_raw')
    if env_nnunet_raw and os.path.exists(env_nnunet_raw):
        if env_nnunet_raw not in paths_to_check:
            paths_to_check.append(env_nnunet_raw)
    
    for path in paths_to_check:
        if not os.path.exists(path):
            continue
        
        try:
            dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        except (OSError, PermissionError):
            continue
        
        for dir_name in dirs:
            if not dir_name.startswith('Dataset'):
                continue
            
            parts = dir_name.split('_')
            if len(parts) >= 1 and parts[0].startswith('Dataset'):
                id_str = parts[0].replace('Dataset', '')
                try:
                    existing_id = int(id_str)
                except ValueError:
                    continue
                
                if existing_id == config_id:
                    raise ValueError(f'nnunet dataset {dir_name} already exists at {os.path.join(path, dir_name)}. please rename your config file to use a different exp_num.')


def validate_config_file(config_file, stage='prepare'):
    filename_id = extract_id_from_filename(config_file)
    if filename_id is None:
        raise ValueError(f'cannot extract id from config filename: {config_file}')
    
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    exp_num = config.get('EXP_NUM')
    model_name = config.get('MODEL_NAME')
    nnunet_raw_path = config.get('NNUNET_RAW_PATH')
    
    if stage == 'test':
        return
    
    # 1. check if filename id matches exp_num
    if exp_num != filename_id:
        raise ValueError(f'id in filename {filename_id} does not match exp_num in config {exp_num}. please rename your config file to match the exp_num.')
    
    # 2. check if id exists in existing config files
    check_id_in_existing_configs(filename_id, config_file)
    
    # 3. check if nnunet dataset exists
    check_nnunet_dataset_exists(exp_num, nnunet_raw_path)


if __name__ == '__main__':
    print('')

    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config_id', default='392', help='configure id (integer) or path to config file')
    parser.add_argument('-s','--stage', default='prepare', type=str, help='Set pipeline stage')
    parser.add_argument('--subject_id', type=str, default=None, help='optional: test only this specific subject id (only used when stage is test)')
    args = parser.parse_args()

    # determine if config_id is a path or an id
    config_file = None
    if os.path.exists(args.config_id):
        config_file = args.config_id
        if not os.path.isfile(config_file):
            raise ValueError(f'config path is not a file: {config_file}')
        validate_config_file(config_file, args.stage)
    else:
        try:
            config_id = int(args.config_id)
            config_file = search_config_name(config_id)
        except ValueError:
            raise ValueError(f'invalid config_id: {args.config_id}. it should be an integer id or a valid file path.')

    config = yaml.safe_load(open(config_file, 'r'))
    
    # set default FILE_NAME_CONFIG based on stage
    if 'FILE_NAME_CONFIG' not in config:
        if args.stage == 'test':
            config['FILE_NAME_CONFIG'] = 'config_test/global_0000_filenames.yaml'
        else:
            config['FILE_NAME_CONFIG'] = 'config/global_000_finenames.yaml'

    if args.stage == 'prepare':
        preparer = PreprocessorInVivo(config)
        preparer.prepare_patch_data_from_ashs_package()

    if args.stage == 'prepare_inr':
        inr_executor = INRPreprocess(config)
        inr_executor.execute()

    if args.stage == 'run_inr':
        inr_executor = INRPreprocess(config)
        inr_executor.run_inr_upsampling()

    if args.stage == 'preprocess':
        preparer = PreprocessorInVivo(config)
        preparer.execute()
    
    if args.stage == 'train':
        preparer = PreprocessorInVivo(config)
        preparer.run_nnunet_training()
    
    if args.stage == 'test':
        tester = ModelTester(config)
        if args.subject_id:
            tester.execute(subject_id=args.subject_id)
        else:
            tester.execute()