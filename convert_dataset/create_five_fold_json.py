#!/usr/bin/env python3

import os
import json
import re
import argparse
import random
from pathlib import Path


def extract_numeric_id(case_name):
    # extract all digits from the case name
    case_number = ''.join(filter(str.isdigit, case_name))
    if case_number:
        # format as 3-digit zero-padded to match train{xxx} format
        return case_number.zfill(3)
    return None


def create_five_fold_json(input_dir, output_path, seed=42):
    # get all case names
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"input directory not found: {input_dir}")
    
    case_names = [d for d in os.listdir(input_dir) 
                  if os.path.isdir(os.path.join(input_dir, d))]
    case_names.sort()
    
    # extract numeric ids
    numeric_ids = []
    for case_name in case_names:
        numeric_id = extract_numeric_id(case_name)
        if numeric_id:
            numeric_ids.append(numeric_id)
        else:
            print(f"warning: could not extract numeric id from case name: {case_name}")
    
    if len(numeric_ids) == 0:
        raise ValueError("no valid numeric ids found in case names")
    
    print(f"found {len(numeric_ids)} cases")
    print(f"numeric ids: {numeric_ids}")
    
    # create 5-fold split
    random.seed(seed)
    shuffled_ids = numeric_ids.copy()
    random.shuffle(shuffled_ids)
    
    # distribute cases into 5 folds
    folds = {f'fold_{i}': [] for i in range(5)}
    
    for i, case_id in enumerate(shuffled_ids):
        fold_idx = i % 5
        folds[f'fold_{fold_idx}'].append(case_id)
    
    # sort each fold for consistency
    for fold in folds.values():
        fold.sort()
    
    # print the result
    print('\nfive-fold split:')
    for fold_name, cases in folds.items():
        print(f'  {fold_name}: {cases} ({len(cases)} cases)')
    
    # save to json file
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(folds, f, indent=2)
    
    print(f'\n✓ saved five_fold.json to: {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Create five_fold.json from case names in a directory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python convert_dataset/create_five_fold_json.py \\
      --input-dir /path/to/t2_ashs \\
      --output convert_dataset/five_fold.json
        """
    )
    parser.add_argument('--input-dir', required=True,
                        help='Path to directory containing case folders')
    parser.add_argument('--output', required=True,
                        help='Output path for five_fold.json')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    create_five_fold_json(args.input_dir, args.output, args.seed)


if __name__ == '__main__':
    main()

