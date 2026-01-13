#!/usr/bin/env python3
"""
trim_neck: Brain MRI neck removal tool

Python implementation of the trim_neck.sh script.
Uses SimpleITK and Convert3D (picsl_c3d) for image processing.

Script by Paul Yushkevich and Sandhitsu Das. Modified by Philip Cook.
Python port for HyperResASHS.
"""

import argparse
import os
import sys
import tempfile
import numpy as np
import SimpleITK as sitk
from picsl_c3d import Convert3D


def get_image_info(image_path):
    """Get image dimensions using SimpleITK."""
    img = sitk.ReadImage(image_path)
    size = img.GetSize()
    return [size[0], size[1], size[2]]


def trim_neck(input_image, output_image, headlen=150, clearance=10, 
               mask_out=None, mask_trim=False, working_dir=None, verbose=False):
    """
    Trim neck from brain MRI image.
    
    Parameters:
    -----------
    input_image : str
        Path to input T1 image
    output_image : str
        Path to output trimmed image
    headlen : float
        Length (sup-inf) of the head that should be captured in mm [default: 150]
    clearance : float
        Clearance above head that should be captured in mm [default: 10]
    mask_out : str, optional
        Path to save mask indicating trimmed region
    mask_trim : bool
        If True, replace trimmed region with zeros rather than cropping [default: False]
    working_dir : str, optional
        Directory for intermediate files. If None, uses temp directory.
    verbose : bool
        Enable verbose/debug output [default: False]
    """
    
    # create working directory
    if working_dir is None:
        if 'TMPDIR' in os.environ:
            working_dir = os.environ['TMPDIR']
        else:
            working_dir = tempfile.mkdtemp()
    else:
        os.makedirs(working_dir, exist_ok=True)
    
    # intermediate file paths
    ras_image = os.path.join(working_dir, 'source_ras.nii.gz')
    dsample_ras = os.path.join(working_dir, 'dsample_ras.nii.gz')
    landmarks_file = os.path.join(working_dir, 'landmarks.txt')
    samples_file = os.path.join(working_dir, 'samples.nii.gz')
    rfmap_file = os.path.join(working_dir, 'rfmap.nii.gz')
    forest_file = os.path.join(working_dir, 'myforest.rf')
    levelset_file = os.path.join(working_dir, 'levelset.nii.gz')
    mask_file = os.path.join(working_dir, 'mask.nii.gz')
    slab_file = os.path.join(working_dir, 'slab.nii.gz')
    slab_src_file = os.path.join(working_dir, 'slab_src.nii.gz')
    trimmed_input_file = os.path.join(working_dir, 'trimmed_input.nii.gz')
    trim_region_mask_file = os.path.join(working_dir, 'trim_mask.nii.gz')
    
    # create landmarks file
    landmarks_content = """40x40x40% 1
60x40x40% 1
40x60x40% 1
40x40x60% 1
60x60x40% 1
60x40x60% 1
40x60x60% 1
60x60x60% 1
3x3x3% 2
97x3x3% 2
3x97x3% 2
3x3x97% 2
97x97x3% 2
97x3x97% 2
3x97x97% 2
97x97x97% 2
"""
    with open(landmarks_file, 'w') as f:
        f.write(landmarks_content)
    
    c3d = Convert3D()
    
    # step 1: swap dimensions to RAS, smooth, resample, compute structure tensor, train RF
    verbose_prefix = '-verbose ' if verbose else ''
    cmd1 = f'{verbose_prefix}{input_image} -swapdim RAS -o {ras_image} ' \
           f'-smooth-fast 1vox -resample-mm 2x2x2mm -o {dsample_ras} -as T1 ' \
           f'-dup -steig 2.0 4x4x4 ' \
           f'-push T1 -dup -scale 0 -lts {landmarks_file} 15 -o {samples_file} ' \
           f'-rf-param-patch 1x1x1 -rf-train {forest_file} -pop -rf-apply {forest_file} ' \
           f'-o {rfmap_file}'
    
    if verbose:
        print(f'Running: {cmd1}')
    c3d.execute(cmd1)
    
    # step 2: level set segmentation
    cmd2 = f'{verbose_prefix}{rfmap_file} -as R -smooth-fast 1vox -resample 50% -stretch 0 1 -1 1 -dup ' \
           f'{samples_file} -thresh 1 1 1 -1 -reslice-identity ' \
           f'-levelset-curvature 5.0 -levelset 300 -o {levelset_file} ' \
           f'-insert R 1 -reslice-identity -thresh 0 inf 1 0 -o {mask_file}'
    
    if verbose:
        print(f'Running: {cmd2}')
    c3d.execute(cmd2)
    
    # step 3: get dimensions and compute trim region
    dims = get_image_info(mask_file)
    dimx, dimy, dimz = dims[0], dims[1], dims[2]
    
    regz = int(headlen / 2 + clearance / 2)
    trim_amount = int(clearance / 2)
    
    # step 4: perform trimming
    cmd3 = f'{verbose_prefix}{mask_file} ' \
           f'-dilate 1 {dimx}x{dimy}x0vox -trim 0x0x{trim_amount}vox ' \
           f'-region 0x0x0vox {dimx}x{dimy}x{regz}vox -thresh -inf inf 1 0 -o {slab_file} -popas S ' \
           f'{input_image} -as I -int 0 -push S -reslice-identity -trim 0vox -as SS -o {slab_src_file} ' \
           f'-push I -reslice-identity -o {trimmed_input_file}'
    
    if verbose:
        print(f'Running: {cmd3}')
    c3d.execute(cmd3)
    
    # step 5: create trim region mask
    cmd4 = f'{input_image} -as I {trimmed_input_file} -thresh 0 0 1 1 -reslice-identity ' \
           f'-thresh 0.5 inf 1 0 -o {trim_region_mask_file}'
    
    if verbose:
        print(f'Running: {cmd4}')
    c3d.execute(cmd4)
    
    # step 6: generate final output
    if mask_trim:
        cmd5 = f'{input_image} {trim_region_mask_file} -multiply -o {output_image}'
    else:
        cmd5 = f'{trimmed_input_file} -o {output_image}'
    
    if verbose:
        print(f'Running: {cmd5}')
    c3d.execute(cmd5)
    
    # step 7: save mask if requested
    if mask_out:
        cmd6 = f'{trim_region_mask_file} -o {mask_out}'
        if verbose:
            print(f'Running: {cmd6}')
        c3d.execute(cmd6)
    
    print('finish trimming neck')


def main():
    parser = argparse.ArgumentParser(
        description='trim_neck: Brain MRI neck removal tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Script by Paul Yushkevich and Sandhitsu Das. Modified by Philip Cook.
Python port for HyperResASHS.
        """
    )
    
    parser.add_argument('input_image', help='Input T1 image path')
    parser.add_argument('output_image', help='Output trimmed image path')
    parser.add_argument('-l', '--headlen', type=float, default=150,
                        help='Length (sup-inf) of the head that should be captured in mm [default: 150]')
    parser.add_argument('-c', '--clearance', type=float, default=10,
                        help='Clearance above head that should be captured in mm [default: 10]')
    parser.add_argument('-m', '--mask-out', dest='mask_out', type=str, default=None,
                        help='Location to save mask indicating trimmed region')
    parser.add_argument('-r', '--mask-trim', dest='mask_trim', action='store_true',
                        help='Output replaces trimmed region with zeros, rather than cropping input')
    parser.add_argument('-w', '--working-dir', dest='working_dir', type=str, default=None,
                        help='Location to store intermediate files. Default: [TMPDIR]')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Verbose/debug mode')
    
    args = parser.parse_args()
    
    trim_neck(
        args.input_image,
        args.output_image,
        headlen=args.headlen,
        clearance=args.clearance,
        mask_out=args.mask_out,
        mask_trim=args.mask_trim,
        working_dir=args.working_dir,
        verbose=args.debug
    )


if __name__ == '__main__':
    main()

