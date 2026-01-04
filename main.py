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

        file_list = os.listdir(config_path)
        for file_ in file_list:
            if int(file_.split('_')[1]) == config_id:
                return os.path.join(config_path, file_)

    if searched_file == None:
        raise ValueError('There is no config file with id {}'.format(config_id))


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
    parser.add_argument('-c','--config_id', default=295, type=int, help='Add configure id')
    parser.add_argument('-s','--stage', default='collect_result', type=str, help='Set pipeline stage')
    args = parser.parse_args()

    config_file = search_config_name(args.config_id)
    check_template_alignment(config_file)
    config = yaml.safe_load(open(config_file, 'r'))

    if args.stage == 'prepare':
        preparer = PreprocessorInVivo(config)
        preparer.prepare_patch_data_from_ashs_package()

    if args.stage == 'prepare_inr':
        inr_executor = INRPreprocess(config)
        inr_executor.execute()

    if args.stage == 'preprocess':
        preparer = PreprocessorInVivo(config)
        preparer.execute()
    
    if args.stage == 'test':
        tester = ModelTester(config)
        tester.execute()