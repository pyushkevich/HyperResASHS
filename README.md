# HyperResASHS

[![arXiv](https://img.shields.io/badge/arXiv-2508.17171-b31b1b.svg)](https://doi.org/10.48550/arXiv.2508.17171)

**HyperResASHS** is a deep learning pipeline for isotropic segmentation of medial temporal lobe (MTL) subregions from multi-modality 3T MRI (T1w and T2w). This repository implements the method described in our paper for achieving high-resolution, isotropic segmentation of brain structures.

## Overview

This project addresses the challenge of segmenting MTL subregions from anisotropic MRI data by:
1. Building an isotropic atlas using Implicit Neural Representations (INR)
2. Training a multi-modality segmentation model with nnU-Net
3. Performing inference on test data with automatic preprocessing

The pipeline handles the entire workflow from raw multi-modality MRI images to final segmentation results, including registration, ROI extraction, upsampling, and model inference.

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- Modified nnU-Net (to be added as submodule)
- Modified Implicit Neural Representation (to be added as submodule)

### Dependencies

TBD

### Submodules

This project requires the following submodules:

- **Modified Implicit Neural Representation (INR)**: Required for the `prepare_inr` stage output processing and when using `INRUpsampling` method in `preprocess` and `test` stages.
- **Modified nnU-Net**: Required for the `preprocess` stage (training data preparation) and the `test` stage (model inference).

**Note**: The `prepare_inr` stage does not require the INR submodule to be installed, as it only prepares the input data for the INR submodule.

### Setup

TBD



## Usage

The main pipeline is controlled through `main.py` with command-line arguments:

```bash
python main.py -c <config_id> -s <stage>
```

### Arguments

- `-c, --config_id`: Configuration ID that corresponds to different experiments
- `-s, --stage`: Pipeline stage to execute. Options: `prepare_inr`, `preprocess`, `test` (see Stages below)

### Pipeline Stages

#### 1. `prepare_inr` - Build Isotropic Atlas

This stage prepares the input data for the Implicit Neural Representation (INR) submodule to create an isotropic atlas.

```bash
python main.py -c 292 -s prepare_inr
```

#### 2. `preprocess` - Training Data Preparation

Prepares training data for nnU-Net by:
- Extracting MTL patches from ASHS atlas data
- Upsampling to isotropic resolution
- Registering multi-modality images
- Formatting data for nnU-Net training

```bash
python main.py -c 292 -s preprocess
```

**Atlas Data Structure**:

TBD

#### 3. `test` - Model Inference

Runs inference on test data with automatic preprocessing:
- Whole-brain registration (T1w to T2w)
- ROI extraction using ASHS template
- Patch cropping and upsampling
- Local registration for fine alignment
- nnU-Net inference

```bash
python main.py -c 2921 -s test
```

**Test Data Structure**:

Test data should be organized as follows:

```
TEST_PATH/
├── participant_id_1/
│   ├── scan_date_1/
│   │   ├── t1_img.nii.gz
│   │   └── t2_img.nii.gz
│   └── scan_date_2/
│       ├── t1_img.nii.gz
│       └── t2_img.nii.gz
└── participant_id_2/
    └── ...
```

## Configuration

Configuration files are located in `config/` (for training) and `config_test/` (for testing).

### Training Configuration

Example: `config/config_292_IsotropicSeg.yaml`

Key parameters:
- `EXP_NUM`: Experiment number
- `MODEL_NAME`: Model identifier
- `UPSAMPLING_METHOD`: One of `'INRUpsampling'`, `'GreedyUpsampling'`, or `'None'`
- `PRIMARY_ASHS_PATH`: Path to 3T-T2w ASHS atlas
- `SECOND_ASHS_PATH`: Path to 3T-T1w ASHS atlas
- `NNUNET_RAW_PATH`: Path to nnU-Net raw data directory

### Test Configuration

Example: `config_test/configtest_2921_IsotropicSeg.yaml`

Key parameters:
- `TEST_PATH`: Root path to test data
- `TEMPLATE_PATH`: Path to ASHS template
- `NECK_SHELL`: Path to neck trimming shell script

## Pipeline Details

### Training Pipeline (`preprocess`)

1. **Patch Extraction**: Extracts left/right MTL patches from ASHS atlas data
2. **Upsampling**: Converts anisotropic patches to isotropic resolution using selected method
3. **Registration**: Registers secondary modality (T1w) to primary (T2w)
4. **Data Formatting**: 
   - Flips right-side images for consistency
   - Removes outer segmentation artifacts
   - Converts labels to continuous format
   - Creates cross-validation splits
5. **nnU-Net Preparation**: Runs nnU-Net experiment planning and preprocessing

### Inference Pipeline (`test`)

1. **Global Registration**: Registers whole-brain T1w to T2w images
2. **ROI Extraction**: 
   - Trims neck from T1w image
   - Performs rigid, affine, and deformable registration to ASHS template
   - Extracts left and right MTL ROIs
3. **Patch Cropping**: Crops MTL patches from both modalities using extracted ROIs
4. **Upsampling**: Converts patches to isotropic space
5. **Local Registration**: Fine-tunes alignment between modalities
6. **Inference**: Runs trained nnU-Net model for segmentation
7. **Output**: Saves segmentation results and processing time

## Upsampling Methods

The pipeline supports three upsampling methods:

- **INRUpsampling**: Uses Implicit Neural Representation for high-quality upsampling (recommended)
- **GreedyUpsampling**: Uses Greedy-based method for segmentation upsampling
- **None**: No upsampling (uses original resolution)

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

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Issues

If you encounter any problems or have questions, please open an issue on GitHub.

## Changelog

### 12/22/2025
- Updated README.md with comprehensive documentation
- Added data structure documentation slots for atlas and test data

### 11/20/2025
- Added main pipeline of preprocessing

### 10/27/2025
- Initial release of isotropic segmentation pipeline for MTL subregions
- Support for 3T-T2w and 3T-T1w multi-modality MRI

## Acknowledgments

- ASHS (Automatic Segmentation of Hippocampal Subfields) for template and atlas
- nnU-Net framework for segmentation
- PICSL tools for image processing

## Contact

For questions or support, please open an issue or contact [liyue3780@gmail.com](mailto:liyue3780@gmail.com).
