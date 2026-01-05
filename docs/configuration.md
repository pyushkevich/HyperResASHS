# Configuration Guide

## Config File Format

The pipeline uses YAML configuration files located in the `config/` or `config_test/` directories. Each config file specifies paths, experiment parameters, and other settings. See `config/config_000_template.yaml` for a template.

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

