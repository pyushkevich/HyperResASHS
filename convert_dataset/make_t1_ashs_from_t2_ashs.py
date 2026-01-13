#!/usr/bin/env python3

import os
import argparse
import SimpleITK as sitk
import numpy as np
from os.path import join


def resample_roi_to_t1_space(roi_mask_path, t1_reference_path):
    roi_mask = sitk.ReadImage(roi_mask_path)
    t1_ref = sitk.ReadImage(t1_reference_path)
    
    # resample roi to t1 space using identity transform
    # since images are aligned in physical space, we use nearest neighbor interpolation
    # this is only to find where the roi is in t1's index space
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(t1_ref)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetTransform(sitk.Transform())
    
    roi_resampled = resampler.Execute(roi_mask)
    
    return roi_resampled


def get_bounding_box_from_mask(roi_mask):
    # convert to numpy array to find non-zero region
    mask_array = sitk.GetArrayFromImage(roi_mask)
    
    # find non-zero indices
    nonzero_indices = np.nonzero(mask_array)
    
    if len(nonzero_indices[0]) == 0:
        raise ValueError("No non-zero voxels found in ROI mask")
    
    # get bounding box in array coordinates (z, y, x)
    z_min, z_max = nonzero_indices[0].min(), nonzero_indices[0].max()
    y_min, y_max = nonzero_indices[1].min(), nonzero_indices[1].max()
    x_min, x_max = nonzero_indices[2].min(), nonzero_indices[2].max()
    
    # convert to simpleitk index coordinates (x, y, z)
    start_idx = [x_min, y_min, z_min]
    size_idx = [x_max - x_min + 1, y_max - y_min + 1, z_max - z_min + 1]
    
    return start_idx, size_idx


def crop_t1_with_roi_mask(t1_path, roi_resampled, output_path):
    t1_image = sitk.ReadImage(t1_path)
    
    # store original t1 spacing for verification
    original_spacing = t1_image.GetSpacing()
    original_size = t1_image.GetSize()
    
    # get bounding box from resampled roi
    start_idx, size_idx = get_bounding_box_from_mask(roi_resampled)
    
    # ensure indices are within image bounds and convert to unsigned int tuples
    image_size = t1_image.GetSize()
    start_idx = tuple(int(max(0, start_idx[i])) for i in range(3))
    size_idx = tuple(int(min(size_idx[i], image_size[i] - start_idx[i])) for i in range(3))
    
    # extract region using extractimagefilter - this does not resample, just extracts
    # a subregion maintaining the original spacing
    extract_filter = sitk.ExtractImageFilter()
    extract_filter.SetSize(size_idx)
    extract_filter.SetIndex(start_idx)
    
    cropped_image = extract_filter.Execute(t1_image)
    
    # update origin to reflect the new physical position
    origin = np.array(t1_image.GetOrigin())
    spacing = np.array(t1_image.GetSpacing())
    direction = np.array(t1_image.GetDirection()).reshape(3, 3)
    
    # calculate new origin in physical space
    start_idx_array = np.array([int(x) for x in start_idx])
    new_origin_physical = origin + direction @ (start_idx_array * spacing)
    cropped_image.SetOrigin(tuple(new_origin_physical))
    
    # maintain original spacing and direction (this preserves t1's resolution)
    cropped_image.SetSpacing(t1_image.GetSpacing())
    cropped_image.SetDirection(t1_image.GetDirection())
    
    # verify spacing is preserved
    cropped_spacing = cropped_image.GetSpacing()
    if cropped_spacing != original_spacing:
        print(f"warning: spacing changed from {original_spacing} to {cropped_spacing}")
    else:
        print(f"t1 resolution preserved: spacing = {original_spacing}")
    
    sitk.WriteImage(cropped_image, output_path)
    print(f"cropped image saved to: {output_path}")
    print(f"original size: {original_size}, cropped size: {cropped_image.GetSize()}")


def extract_numeric_id(case_name):
    # extract all digits from the case name
    numeric_part = ''.join(filter(str.isdigit, case_name))
    if numeric_part:
        return numeric_part.zfill(3)
    return None


def rename_case_to_train_format(case_name):
    numeric_id = extract_numeric_id(case_name)
    if numeric_id:
        return f'train{numeric_id}'
    return None


def make_t1_ashs_from_t2_ashs(t2_ashs_path, output_ashs_path, 
                            t2_roi_name='tse_native_chunk_{side}.nii.gz',
                            t1_whole_name='mprage.nii.gz',
                            t1_output_name='tse_native_chunk_{side}.nii.gz'):
    os.makedirs(output_ashs_path, exist_ok=True)
    
    case_list = [d for d in os.listdir(t2_ashs_path) 
                 if os.path.isdir(join(t2_ashs_path, d))]
    case_list.sort()
    
    print(f"found {len(case_list)} cases in t2 ashs path: {t2_ashs_path}")
    print(f"output t1 ashs path: {output_ashs_path}")
    print(f"note: t2 ashs folders will be renamed in place to train{{xxx}} format")
    print(f"\nprocessing cases...")
    
    for case_idx, original_case_id in enumerate(case_list, 1):
        train_case_id = rename_case_to_train_format(original_case_id)
        if train_case_id is None:
            print(f"\n[{case_idx}/{len(case_list)}] warning: could not extract numeric id from case: {original_case_id}")
            print(f"  skipping this case...")
            continue
        
        print(f"\n[{case_idx}/{len(case_list)}] processing case: {original_case_id} -> {train_case_id}")
        
        original_case_path = join(t2_ashs_path, original_case_id)
        renamed_t2_case_path = join(t2_ashs_path, train_case_id)
        
        if original_case_id != train_case_id:
            if os.path.exists(renamed_t2_case_path):
                print(f"  warning: t2 ashs folder already exists: {train_case_id}, skipping rename")
                t2_case_path = renamed_t2_case_path
            else:
                os.rename(original_case_path, renamed_t2_case_path)
                print(f"  renamed t2 ashs folder: {original_case_id} -> {train_case_id}")
                t2_case_path = renamed_t2_case_path
        else:
            t2_case_path = original_case_path
            print(f"  t2 ashs folder already in correct format: {train_case_id}")
        
        output_case_path = join(output_ashs_path, train_case_id)
        os.makedirs(output_case_path, exist_ok=True)
        
        for side in ['left', 'right']:
            print(f"  processing {side} side...")
            
            t2_roi_path = join(t2_case_path, t2_roi_name.format(side=side))
            t1_whole_path = join(t2_case_path, t1_whole_name)
            t1_output_path = join(output_case_path, t1_output_name.format(side=side))
            
            if not os.path.exists(t2_roi_path):
                print(f"    warning: t2 roi not found: {t2_roi_path}")
                continue
            
            if not os.path.exists(t1_whole_path):
                print(f"    warning: whole-brain t1 not found: {t1_whole_path}")
                continue
            
            try:
                roi_resampled = resample_roi_to_t1_space(t2_roi_path, t1_whole_path)
                start_idx, size_idx = get_bounding_box_from_mask(roi_resampled)
                crop_t1_with_roi_mask(t1_whole_path, roi_resampled, t1_output_path)
                
                print(f"    ✓ successfully created: {t1_output_path}")
                
            except Exception as e:
                print(f"    error processing {side} side for case {train_case_id}: {str(e)}")
                continue
    
    print(f"\n✓ completed! t1 ashs dataset created at: {output_ashs_path}")
    print(f"✓ t2 ashs folders renamed in place at: {t2_ashs_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Make T1 ASHS format dataset from T2 ASHS path and whole-brain T1 images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python convert_dataset/make_t1_ashs_from_t2_ashs.py \\
      --t2-ashs-path /path/to/t2/ashs \\
      --output-ashs-path /path/to/output/t1/ashs

Input structure (T2 ASHS path - contains both T2 ROIs and whole-brain T1):
  {t2_ashs_path}/
    {case_id}/
      tse_native_chunk_left.nii.gz    (T2 ROI)
      tse_native_chunk_right.nii.gz   (T2 ROI)
      mprage.nii.gz                    (whole-brain T1)

Output structure (T1 ASHS path):
  {output_ashs_path}/
    train{xxx}/
      tse_native_chunk_left.nii.gz    (cropped T1 ROI)
      tse_native_chunk_right.nii.gz   (cropped T1 ROI)

Note: Case folders are automatically renamed to train{xxx} format where xxx is 3-digit zero-padded number.
        """
    )
    parser.add_argument('--t2-ashs-path', required=True,
                        help='Path to T2 ASHS directory (contains case folders with T2 ROIs and mprage.nii.gz)')
    parser.add_argument('--output-ashs-path', required=True,
                        help='Output path for T1 ASHS format dataset')
    parser.add_argument('--t2-roi-name', default='tse_native_chunk_{side}.nii.gz',
                        help='Filename pattern for T2 ROI (default: tse_native_chunk_{side}.nii.gz)')
    parser.add_argument('--t1-whole-name', default='mprage.nii.gz',
                        help='Filename for whole-brain T1 (default: mprage.nii.gz)')
    parser.add_argument('--t1-output-name', default='tse_native_chunk_{side}.nii.gz',
                        help='Output filename pattern for T1 ROI (default: tse_native_chunk_{side}.nii.gz)')
    args = parser.parse_args()
    
    if not os.path.exists(args.t2_ashs_path):
        raise FileNotFoundError(f"t2 ashs path not found: {args.t2_ashs_path}")
    
    make_t1_ashs_from_t2_ashs(
        t2_ashs_path=args.t2_ashs_path,
        output_ashs_path=args.output_ashs_path,
        t2_roi_name=args.t2_roi_name,
        t1_whole_name=args.t1_whole_name,
        t1_output_name=args.t1_output_name
    )


if __name__ == '__main__':
    main()

