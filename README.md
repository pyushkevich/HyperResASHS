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

Follow these steps to set up the repository in a fresh environment:

### 1. Create a Conda Environment

Create a new conda environment with Python 3.10 or higher:

```bash
conda create -n hyperresashs python=3.10
conda activate hyperresashs
```

### 2. Clone the Repository

Clone the repository with submodules:

```bash
git clone --recursive https://github.com/liyue3780/HyperResASHS.git
cd HyperResASHS
```

If you've already cloned without submodules, initialize them with:

```bash
git submodule update --init --recursive
```

### 3. Install Python Dependencies

**Important**: PyTorch version compatibility is critical. This pipeline requires PyTorch 2.5.x (tested with 2.5.1). Newer versions (e.g., 2.9) may cause compatibility issues.

```bash
# First, install PyTorch with CUDA support (adjust CUDA version as needed)
# For CUDA 11.8:
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
# pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# Then install the package and remaining dependencies
pip install -e .
```

**Note**: Additional dependencies may be required by the submodules. See the submodule setup instructions below.

### 4. Install ITK-SNAP

Download ITK-SNAP from the [official website](http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.SNAP3) and add its bin directory to your PATH.

**For Linux:**

```bash
export PATH=$PATH:/path/to/itksnap-4.0.1-20230320-Linux-gcc64/bin
```

Add this line to your `~/.bashrc` (or `~/.zshrc` for zsh) for permanent setup, then reload your shell: `source ~/.bashrc`

For other platforms, adapt the path according to your installation location and operating system.

### 5. Set Up Submodules

This repository uses git submodules for dependencies:

- **`submodules/multi_contrast_inr`**: INR repository (tracking `main` branch)
- **`submodules/nnUNet`**: Modified nnUNet repository (tracking `mmseg` branch) - [https://github.com/liyue3780/nnUNet/tree/mmseg](https://github.com/liyue3780/nnUNet/tree/mmseg)

**Install nnUNet submodule:**

```bash
cd submodules/nnUNet
pip install -e .
cd ../..
```

**Set nnU-Net Environment Variables:**

After installing nnUNet, you must set the following environment variables:

```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

For detailed setup instructions (including Linux, MacOS, and Windows), see the [nnU-Net environment variables documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/set_environment_variables.md).

**Install INR submodule dependencies:**

```bash
pip install tensorboard==2.20.0 lpips==0.1.4
```

Refer to the INR repository's documentation for specific installation requirements. The INR submodule may require additional dependencies such as PyTorch, nibabel, and other packages.

**Note**: The modified nnUNet includes Modality Augmentation methods for multi-modality brain MRI segmentation. Make sure to use the `mmseg` branch when running nnU-Net training.

### 6. Verify Installation

Verify that the main pipeline can be imported:

```bash
python -c "from preprocessing import PreprocessorInVivo; from testing import ModelTester; from prepare_inr import INRPreprocess; print('Installation successful!')"
```

## Configuration

For detailed configuration information, including config file format and cross-validation file format, see [Configuration Guide](docs/configuration.md).

## Pipeline Details

The pipeline consists of six main steps that can be run in linear order. Each step assumes the previous steps have been completed:

1. **Prepare** → 2. **Prepare INR** → 3. **Run INR Upsampling** → 4. **Preprocess** → 5. **Train** → 6. **Test**

**Note**: Steps 2-3 are only needed if using INR upsampling. For other upsampling methods (e.g., `GreedyUpsampling` or `None`), you can skip Steps 2-3 and go directly from Step 1 to Step 4.

### Step 1: Prepare Patch Data (`stage = prepare`)

Run this step first to create the experiment folder by copying images and segmentations from the two atlas folders (T1w and T2w ASHS atlases). This step:
- Copies primary and secondary modality images from ASHS packages
- Copies segmentation files
- Performs coordinate system transformations (swapdim RPI)
- Creates the folder structure in `{PREPARE_RAW_PATH}/{EXP_NUM}{MODEL_NAME}/images/`

**Usage:**
```bash
python main.py -s prepare -c {CONFIG_ID}
```

### Step 2: Prepare INR Data (`stage = prepare_inr`)

This step prepares the data for INR upsampling. It requires the prepared patch data from Step 1. The `prepare_inr` stage will:
- Prepare the data in the format expected by the INR submodule
- Generate INR configuration files for each case
- Create a shell script `scripts/run_inr_upsampling_{EXP_NUM}{MODEL_NAME}.sh` with paths automatically filled from your config

**Usage:**
```bash
python main.py -s prepare_inr -c {CONFIG_ID}
```

**Folder structure created:**

The `prepare_inr` stage creates the following folder structure under `{INR_PATH}/{EXP_NUM}{MODEL_NAME}/`:
- `preprocess/`: Contains case folders with input data and config files for INR
- `training_preparation/`: Contains case folders with prepared data ready for INR training
- `training_output/`: Will contain INR training outputs (created after INR training completes)

### Step 3: Run INR Upsampling (`stage = run_inr`)

After preparing INR data, you have two options to run INR upsampling:

#### Option 1: Run via Python (Recommended for single GPU)

Run INR upsampling directly using Python:

```bash
python main.py -s run_inr -c {CONFIG_ID}
```

This will process all cases sequentially. The script automatically:
- Finds all cases in the `training_preparation` folder (created in Step 2)
- Runs INR training for each case using the generated config files from the `preprocess` folder

#### Option 2: Run via Shell Script (Recommended for multi-GPU)

For multi-GPU setups, you can use the generated shell script and modify it to run different batches on different GPUs. See [INR Upsampling Shell Script Guide](docs/inr_upsampling_shell_script.md) for detailed instructions.

### Step 4: Complete Preprocessing (`stage = preprocess`)

After INR upsampling is finished, run the preprocessing stage to complete all remaining steps. This step assumes the patch data preparation (Step 1) was already completed. It will:
- Copy INR upsampled results (if using INR upsampling method)
- Perform resampling/upsampling based on the configured method
- Register secondary modality (T1w) to primary (T2w)
- Prepare nnU-Net dataset
- Remove outer segmentation artifacts
- Convert labels to continuous format
- Create cross-validation splits
- Run nnU-Net experiment planning
- Generate nnU-Net training script: `scripts/train_nnunet_{EXP_NUM}{MODEL_NAME}.sh`

**Usage:**
```bash
python main.py -s preprocess -c {CONFIG_ID}
```

**Outputs from Step 4:**
- nnU-Net dataset in `{NNUNET_RAW_PATH}/Dataset{EXP_NUM}_{MODEL_NAME}/`
- Preprocessed data in `{NNUNET_RAW_PATH}/../nnUNet_preprocessed/Dataset{EXP_NUM}_{MODEL_NAME}/`
- Cross-validation splits file: `splits_final.json`
- Training script: `scripts/train_nnunet_{EXP_NUM}{MODEL_NAME}.sh`

**Note**: If you're using a non-INR upsampling method (e.g., `GreedyUpsampling` or `None`), you can skip Steps 2-3 and go directly from Step 1 to Step 4.

### Step 5: nnU-Net Training (`stage = train`)

Step 4 creates the nnU-Net dataset, runs experiment planning, and creates five-fold cross-validation splits. A training script is automatically generated for convenience.

You have two options to run nnU-Net training:

#### Option 1: Run via Python (Recommended)

Run nnU-Net training directly using Python:

```bash
python main.py -s train -c {CONFIG_ID}
```

This will train all 5 folds (fold 0-4) sequentially using the `TRAINER` specified in your configuration file (e.g., `ModAugUNetTrainer`).

**Note**: Ensure you're using the modified nnUNet from `submodules/nnUNet` (mmseg branch) which includes Modality Augmentation methods. The `nnUNetv2_train` command should be available after installing the modified nnUNet.

#### Option 2: Run via Shell Script

Alternatively, you can run the generated training script manually. See [nnU-Net Training Shell Script Guide](docs/nnunet_training_shell_script.md) for detailed instructions.

### Step 6: Testing (`stage = test`)

Testing is independent from Steps 1-5 and uses its own configuration files in the `config_test/` directory. Each test configuration has its own ID that links to a trained model.

**Test Configuration ID Convention:**
- The test config ID links to the training model ID. For example:
  - `2921` links to model `292` (the `1` represents the first test set)
  - `2922`, `2923`, `2924`, etc. can be used for different test sets of the same model
- The first digits match the `EXP_NUM` of the trained model you want to use

**Usage:**
```bash
python main.py -s test -c {TEST_CONFIG_ID}
# For example: python main.py -s test -c 2921
```

**Configuration Requirements:**
- `EXP_NUM`: Must match the training model's `EXP_NUM` (e.g., `292`)
- `MODEL_NAME`: Must match the training model's `MODEL_NAME` (e.g., `TestPipeline`)
- `TRAINER`: Must match the training model's `TRAINER`
- `CONDITION`: Must match the training model's `CONDITION` (e.g., `in_vivo`)
- `UPSAMPLING_METHOD`: Must match the training model's `UPSAMPLING_METHOD`
- `TEST_PATH`: Path to test data (see [Configuration Guide](docs/configuration.md) for structure)
- `TEMPLATE_PATH`: Path to ASHS template for MTL ROI cropping (downloadable from [DOI: 10.5061/dryad.k6djh9wmn](https://doi.org/10.5061/dryad.k6djh9wmn))

**This stage performs:**
- Whole-brain registration (T1w to T2w)
- ROI extraction using ASHS template
- Patch cropping and upsampling
- Local registration for fine alignment
- nnU-Net inference for segmentation
- Output of segmentation results

For detailed test configuration information, see [Configuration Guide](docs/configuration.md).

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

### 01/04/2026
- Refactored pipeline to support linear execution order
- Added separate `prepare` stage for patch data preparation
- Simplified `execute()` method in preprocessing to remove conditional checks
- Updated pipeline documentation to reflect linear execution flow
- Added `.gitignore` to exclude Python cache files and build artifacts

### 12/24/2025
- Added INR and modified nnUNet as git submodules
- Added INR preparation module with config generation
- Added INR upsampling script template and generation
- Added nnUNet training script template and generation
- Updated README with submodule setup and pipeline documentation

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
