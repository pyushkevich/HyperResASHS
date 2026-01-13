# Configuration Guide

## Config File Format

The pipeline uses YAML configuration files located in the `config/` or `config_test/` directories. Each config file specifies paths, experiment parameters, and other settings. See `config/config_000_template.yaml` for a template.

## Training Configuration (`config/`)

Training configurations are located in the `config/` directory and define all parameters needed for the training pipeline.

### Training Configuration Parameters

#### Global Parameters

- **`EXP_NUM`**: Experiment number (e.g., `292`). This will be used as the nnU-Net dataset ID and must be unique for each experiment.
- **`MODEL_NAME`**: Model name identifier (e.g., `TestPipeline`). Combined with `EXP_NUM` to create unique experiment identifiers.
- **`TRAINER`**: nnU-Net trainer class to use (e.g., `nnUNetTrainerAnyNumModAugNumEpoch10`, `ModAugUNetTrainer`). Must be available in the modified nnUNet submodule.
- **`CONDITION`**: Experimental condition (e.g., `in_vivo`). Used to specify the type of data being processed.

#### Upsampling Parameters

- **`UPSAMPLING_METHOD`**: Method for upsampling segmentation (options: `None`, `INRUpsampling`, `GreedyUpsampling`)
  - `None`: No upsampling, use original resolution
  - `INRUpsampling`: Use Implicit Neural Representation for upsampling (requires Steps 2-3)
  - `GreedyUpsampling`: Linear interpolation implemented by Greedy

#### Preprocessing Paths

- **`PREPARE_RAW_PATH`**: Base path where prepared patch data will be stored
- **`PRIMARY_ASHS_PATH`**: Path to 3T-T2w ASHS atlas directory. ASHS atlases can be downloaded from [NITRC ASHS project page](https://www.nitrc.org/projects/ashs)
- **`SECOND_ASHS_PATH`**: Path to 3T-T1w ASHS atlas directory. ASHS atlases can be downloaded from [NITRC ASHS project page](https://www.nitrc.org/projects/ashs)
- **`SNAP_LABEL_PATH`**: Path to SNAP label file for label mapping. This file is available in the T2 atlas directory
- **`CV_FILE`**: Path to JSON file defining cross-validation folds (see [Cross-Validation File Format](#cross-validation-file-format-cv_file))

#### INR Parameters (if using INR upsampling)

- **`INR_PATH`**: Base path where INR processing results will be stored
- **`INR_CORRECTION_PARAM`**: Registration correction method (options: `None`, `rigid`, `affine`)

#### nnU-Net Parameters

- **`NNUNET_RAW_PATH`**: Path to nnU-Net raw data directory (should match `nnUNet_raw` environment variable)
- **`NNUNET_PREPROSSOR`**: Preprocessor identifier for nnU-Net experiment planning

#### File Configuration

- **`FILE_NAME_CONFIG`**: Path to the global filenames configuration file (e.g., `config/global_000_finenames.yaml`)

### Example Training Configuration

```yaml
# global configure
EXP_NUM: 292
MODEL_NAME: 'TestPipeline'
TRAINER: 'nnUNetTrainerAnyNumModAugNumEpoch10'
CONDITION: 'in_vivo'

# preprocessing configure
PREPARE_RAW_PATH: '/path/to/prepare/raw'
PRIMARY_ASHS_PATH: '/path/to/t2w/ashs/atlas'
SECOND_ASHS_PATH: '/path/to/t1w/ashs/atlas'
SNAP_LABEL_PATH: '/path/to/snap/labels.txt'
CV_FILE: '/path/to/cross_validation.json'

# Upsampling
UPSAMPLING_METHOD: 'INRUpsampling'

# INR
INR_PATH: '/path/to/inr/results'
INR_CORRECTION_PARAM: 'rigid'

# nnU-Net
NNUNET_RAW_PATH: '/path/to/nnunet/raw'
NNUNET_PREPROSSOR: 'DefaultPreprocessor'

# file or folder name config file
FILE_NAME_CONFIG: 'config/global_000_finenames.yaml'
```

## Cross-Validation File Format (`CV_FILE`)

The `CV_FILE` parameter in your config specifies a JSON file that defines the cross-validation folds for nnU-Net training. This file determines how cases are split into training and validation sets for 5-fold cross-validation.

### Format

The JSON file should contain exactly 5 folds (`fold_0` through `fold_4`), where each fold is a list of case IDs (as strings). Case IDs should match the numeric part of your case folder names (e.g., if your case folder is `000_left`, the case ID is `"000"`).

### Example

```json
{
    "fold_0": [
        "000",
        "001",
        "002",
        "014",
        "021",
        "022"
    ],
    "fold_1": [
        "007",
        "009",
        "011",
        "024",
        "026",
        "027"
    ],
    "fold_2": [
        "008",
        "013",
        "016",
        "017",
        "018",
        "020"
    ],
    "fold_3": [
        "004",
        "005",
        "006",
        "010",
        "015",
        "025"
    ],
    "fold_4": [
        "003",
        "012",
        "019",
        "023",
        "028"
    ]
}
```

### How It Works

- During preprocessing (Step 4), the pipeline reads this file and maps each case ID to its corresponding nnU-Net dataset ID
- For each fold, the cases in that fold become the validation set, while cases from all other folds become the training set
- This creates 5-fold cross-validation splits automatically

## Test Configuration (`config_test/`)

Test configurations are independent from training configurations and use their own ID system. Test configs are located in the `config_test/` directory.

### Test Configuration ID Convention

The test configuration ID links to a trained model:
- **Format**: `{MODEL_EXP_NUM}{TEST_SET_NUMBER}`
- **Example**: 
  - `2921` links to model `292` (the `1` represents the first test set)
  - `2922`, `2923`, `2924`, etc. can be used for different test sets of the same model
- The first digits must match the `EXP_NUM` of the trained model you want to use

### Test Configuration Parameters

The following parameters must match the training configuration:

- **`EXP_NUM`**: Must match the training model's `EXP_NUM` (e.g., `292`)
- **`MODEL_NAME`**: Must match the training model's `MODEL_NAME` (e.g., `TestPipeline`)
- **`TRAINER`**: Must match the training model's `TRAINER` (e.g., `nnUNetTrainerAnyNumModAugNumEpoch10`)
- **`CONDITION`**: Must match the training model's `CONDITION` (e.g., `in_vivo`)
- **`UPSAMPLING_METHOD`**: Must match the training model's `UPSAMPLING_METHOD` (e.g., `INRUpsampling`)

### Test Data Path Structure (`TEST_PATH`)

The `TEST_PATH` should point to a directory with the following structure:

```
TEST_PATH/
├── subj001/
│   ├── 20260107/          # or 2026-01-07
│   │   ├── t1_img.nii.gz  # Whole-brain T1 image
│   │   └── t2_img.nii.gz  # Dedicated T2 image for MTL
│   └── 20251231/          # or 2025-12-31
│       ├── t1_img.nii.gz
│       └── t2_img.nii.gz
├── subj002/
│   └── ...
└── ...
```

**Structure Requirements:**
- **First level**: Image/subject ID (e.g., `subj001`, `subj002`, etc.)
- **Second level**: Scanning date (e.g., `20260107`, `20251231`, `2026-01-07`, `2025-12-31`, etc.)
- **Files**: 
  - `t1_img.nii.gz`: Whole-brain T1-weighted image
  - `t2_img.nii.gz`: Dedicated T2-weighted image for MTL region

### Template Path (`TEMPLATE_PATH`)

The `TEMPLATE_PATH` should point to the ASHS template directory used for cropping the MTL ROI. This template can be downloaded from:

**DOI**: [10.5061/dryad.k6djh9wmn](https://doi.org/10.5061/dryad.k6djh9wmn)

The template is required for ROI extraction during the testing stage.

### Example Test Configuration

```yaml
# global configure
EXP_NUM: 292                    # Must match training model EXP_NUM
MODEL_NAME: 'TestPipeline'      # Must match training model MODEL_NAME
TRAINER: 'nnUNetTrainerAnyNumModAugNumEpoch10'  # Must match training TRAINER
CONDITION: 'in_vivo'            # Must match training CONDITION
UPSAMPLING_METHOD: "INRUpsampling"  # Must match training UPSAMPLING_METHOD

# test 
TEST_PATH: '/path/to/test/data'

# trim neck shell script path
NECK_SHELL: 'shell/trim_neck.sh'
TEMPLATE_PATH: '/path/to/ashs/template'  # Download from DOI: 10.5061/dryad.k6djh9wmn

# file that saves global file names
FILE_NAME_CONFIG: "config_test/global_0000_filenames.yaml"
```

