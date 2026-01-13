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
    # extract id from filename like config_292_IsotropicSeg.yaml -> 292
    basename = os.path.basename(filename)
    parts = basename.split('_')
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def check_id_in_existing_configs(config_id, config_file_path):
    # check if config_id exists in config/ or config_test/ folders
    curr_path = Path.cwd()
    for folder_name in ['config', 'config_test']:
        config_path = os.path.join(curr_path, folder_name)
        if not os.path.exists(config_path):
            continue

        file_list = os.listdir(config_path)
        for file_ in file_list:
            if file_ == os.path.basename(config_file_path):
                continue  # skip the current file
            try:
                existing_id = int(file_.split('_')[1])
                if existing_id == config_id:
                    raise ValueError(f'config id {config_id} already exists in {folder_name}/{file_}. please change the id in your config filename.')
            except (ValueError, IndexError):
                continue


def check_nnunet_dataset_exists(config_id, model_name, nnunet_raw_path):
    # check if Dataset{config_id}_{model_name} exists in nnUNet raw path
    # check both config path and environment variable
    paths_to_check = []
    
    if nnunet_raw_path and os.path.exists(nnunet_raw_path):
        paths_to_check.append(nnunet_raw_path)
    
    env_nnunet_raw = os.environ.get('nnUNet_raw')
    if env_nnunet_raw and os.path.exists(env_nnunet_raw):
        if env_nnunet_raw not in paths_to_check:
            paths_to_check.append(env_nnunet_raw)
    
    dataset_name = f'Dataset{config_id}_{model_name}'
    for path in paths_to_check:
        dataset_path = os.path.join(path, dataset_name)
        if os.path.exists(dataset_path):
            raise ValueError(f'nnunet dataset {dataset_name} already exists at {dataset_path}. please change the exp_num in your config file.')


def validate_config_file(config_file):
    # extract id from filename
    filename_id = extract_id_from_filename(config_file)
    if filename_id is None:
        raise ValueError(f'cannot extract id from config filename: {config_file}')
    
    # load config to get exp_num
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    exp_num = config.get('EXP_NUM')
    model_name = config.get('MODEL_NAME')
    nnunet_raw_path = config.get('NNUNET_RAW_PATH')
    
    # check if filename id matches exp_num
    if exp_num != filename_id:
        print(f'warning: id in filename ({filename_id}) does not match exp_num in config ({exp_num}). please ensure they are consistent.')
    
    # check if id exists in existing config files
    check_id_in_existing_configs(filename_id, config_file)
    
    # check if nnunet dataset exists
    if nnunet_raw_path and model_name:
        check_nnunet_dataset_exists(exp_num, model_name, nnunet_raw_path)


def config_comparison(template_config, user_config):
    user_keys = set(user_config)
    template_keys = set(template_config)

    missing_keys = template_keys - user_keys
    extra_keys = user_keys - template_keys

    if missing_keys or extra_keys:
        error_msg = ""
        if missing_keys:
            error_msg += f"Missing keys: {sorted(missing_keys)}\n"
        if extra_keys:
            error_msg += f"Extra keys: {sorted(extra_keys)}\n"
        raise ValueError(f"Config mismatch:\n{error_msg}")


def check_template_alignment(input_config):
    parent_path = Path(input_config).parent
    template_path = None
    for file_ in os.listdir(parent_path):
        if 'template' in file_:
            template_path = os.path.join(parent_path, file_)

    with open(input_config) as f:
        user_config = yaml.safe_load(f)

    with open(template_path) as f:
        template_config = yaml.safe_load(f)
    
    config_comparison(template_config, user_config)


if __name__ == '__main__':
    print('')

    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config_id', default='392', help='configure id (integer) or path to config file')
    parser.add_argument('-s','--stage', default='prepare', type=str, help='Set pipeline stage')
    args = parser.parse_args()

    # determine if config_id is a path or an id
    config_file = None
    if os.path.exists(args.config_id):
        config_file = args.config_id
        if not os.path.isfile(config_file):
            raise ValueError(f'config path is not a file: {config_file}')
        validate_config_file(config_file)
    else:
        try:
            config_id = int(args.config_id)
            config_file = search_config_name(config_id)
        except ValueError:
            raise ValueError(f'invalid config_id: {args.config_id}. it should be an integer id or a valid file path.')

    check_template_alignment(config_file)
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
        tester.execute()