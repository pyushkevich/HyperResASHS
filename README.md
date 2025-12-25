# HyperResASHS

[![arXiv](https://img.shields.io/badge/arXiv-2508.17171-b31b1b.svg)](https://doi.org/10.48550/arXiv.2508.17171)

**HyperResASHS** is a deep learning pipeline for isotropic segmentation of medial temporal lobe (MTL) subregions from multi-modality 3T MRI (T1w and T2w). This repository implements the method described in our paper for achieving high-resolution, isotropic segmentation of brain structures.

## Overview

This project addresses the challenge of segmenting MTL subregions from anisotropic MRI data by:
1. Building an isotropic atlas using Implicit Neural Representations (INR)
2. Training a multi-modality segmentation model with nnU-Net
3. Performing inference on test data with automatic preprocessing

The pipeline handles the entire workflow from raw multi-modality MRI images to final segmentation results, including registration, ROI extraction, upsampling, and model inference.

## Setup

This repository uses git submodules for dependencies. When cloning, use:

```bash
git clone --recursive <repository-url>
```

If you've already cloned without submodules, initialize them with:

```bash
git submodule update --init --recursive
```

**Submodules:**
- `submodules/multi_contrast_inr`: INR repository (tracking `main` branch)
- `submodules/nnUNet`: Modified nnUNet repository (tracking `mmseg` branch) - [https://github.com/liyue3780/nnUNet/tree/mmseg](https://github.com/liyue3780/nnUNet/tree/mmseg)

**Note**: The modified nnUNet includes Modality Augmentation methods for multi-modality brain MRI segmentation. Make sure to use the `mmseg` branch when running nnU-Net training.

## Pipeline Details

The pipeline consists of five main steps:

### Step 1: Preprocessing (`stage = preprocess`)

Run preprocessing to create the experiment folder by copying images and segmentations from the two atlas folders (T1w and T2w ASHS atlases). 

- If the upsampling method is **not INR** (e.g., `GreedyUpsampling` or `None`), the preprocessing will complete all steps including upsampling, registration, and nnU-Net dataset preparation.
- If the upsampling method is **INR**, the preprocessing will only prepare the patch data and stop, waiting for the user to complete INR upsampling (proceed to Step 2).

**Note**: After INR upsampling is completed (Step 2), you need to **run Step 1 again** (`stage = preprocess`) to complete the remaining preprocessing steps. This second run becomes Step 3.

### Step 2: INR Upsampling (`stage = prepare_inr`)

Use INR to upsample the atlas segmentation. This step requires the copied images and folders from Step 1 as input. The `prepare_inr` stage will:
- Prepare the data in the format expected by the INR submodule
- Generate INR configuration files for each case
- Create a shell script `scripts/run_inr_upsampling_{EXP_NUM}{MODEL_NAME}.sh` with paths automatically filled from your config

**Folder structure created:**

The `prepare_inr` stage creates the following folder structure under `{INR_PATH}/{EXP_NUM}{MODEL_NAME}/`:
- `preprocess/`: Contains case folders with input data and config files for INR
- `training_preparation/`: Contains case folders with prepared data ready for INR training
- `training_output/`: Will contain INR training outputs (created after INR training completes)

**To run INR upsampling:**

1. **Update the INR repository path** in the generated script:
   ```bash
   # Edit scripts/run_inr_upsampling_{EXP_NUM}{MODEL_NAME}.sh
   # For example: scripts/run_inr_upsampling_292IsotropiSeg.sh
   # Update INR_REPO_PATH to point to the INR submodule
   INR_REPO_PATH="submodules/multi_contrast_inr"
   ```

2. **Run the generated script**:
   ```bash
   ./scripts/run_inr_upsampling_{EXP_NUM}{MODEL_NAME}.sh
   # For example: ./scripts/run_inr_upsampling_292IsotropiSeg.sh
   ```

   The script will automatically:
   - Find all cases in the `training_preparation` folder (created in Step 2)
   - Run INR training for each case using the generated config files from the `preprocess` folder
   - Process cases in batches (default: start=0, count=60, can be modified in the script)

**Note**: The script uses the INR repository's `main.py` to train each case. Make sure the INR repository is properly set up and accessible.

### Step 3: Complete Preprocessing (`stage = preprocess` - second run)

After INR upsampling is finished, **run Step 1 again** (`stage = preprocess`). This time, since the INR output exists, the preprocessing will complete all remaining steps:
- Copy INR upsampled results
- Register secondary modality (T1w) to primary (T2w)
- Prepare nnU-Net dataset
- Remove outer segmentation artifacts
- Convert labels to continuous format
- Create cross-validation splits
- Run nnU-Net experiment planning

### Step 4: nnU-Net Training

Step 3 creates the nnU-Net dataset, runs experiment planning, and creates five-fold cross-validation splits. However, **nnU-Net training must be run manually**. 

**Note**: 
- Use the modified nnUNet from `submodules/nnUNet` (mmseg branch) which includes Modality Augmentation methods
- When running nnU-Net training, ensure that the trainer matches the `TRAINER` specified in your configuration file (e.g., `ModAugUNetTrainer`)

### Step 5: Testing (`stage = test`)

When processing test data, run the test stage. This stage performs:
- Whole-brain registration (T1w to T2w)
- ROI extraction using ASHS template
- Patch cropping and upsampling
- Local registration for fine alignment
- nnU-Net inference for segmentation
- Output of segmentation results

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{hyperresashs2024,
  title={HyperResASHS: Isotropic Segmentation of MTL Subregions from Multi-modality 3T MRI},
  author={[Authors]},
  journal={arXiv preprint arXiv:2508.17171},
  year={2024},
  url={https://doi.org/10.48550/arXiv.2508.17171}
}
```

## Changelog

### 12/22/2025
- Updated README.md with comprehensive documentation
- Added data structure documentation slots for atlas and test data

### 11/20/2025
- Added main pipeline of preprocessing

### 10/27/2025
- Initial release of isotropic segmentation pipeline for MTL subregions
- Support for 3T-T2w and 3T-T1w multi-modality MRI

## Contact

For questions or support, please open an issue or contact [liyue3780@gmail.com](mailto:liyue3780@gmail.com).
