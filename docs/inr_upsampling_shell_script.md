# INR Upsampling via Shell Script (Multi-GPU Setup)

For multi-GPU setups, you can use the generated shell script and modify it to run different batches on different GPUs.

## Usage

### 1. Edit the Generated Script

Edit the generated script to set `start` and `count` parameters:

```bash
# Edit scripts/run_inr_upsampling_{EXP_NUM}{MODEL_NAME}.sh
# For example: scripts/run_inr_upsampling_292IsotropiSeg.sh
# Modify these variables:
start=0      # Starting index of cases to process
count=60     # Number of cases to process
```

### 2. Run the Script

Run the script:

```bash
./scripts/run_inr_upsampling_{EXP_NUM}{MODEL_NAME}.sh
# For example: ./scripts/run_inr_upsampling_292IsotropiSeg.sh
```

## Multi-GPU Usage Example

To distribute cases across multiple GPUs:

- **GPU 0**: Edit script with `start=0, count=30`, run on GPU 0
- **GPU 1**: Edit script with `start=30, count=30`, run on GPU 1
- **GPU 2**: Edit script with `start=60, count=30`, run on GPU 2
- etc.

## Notes

- The generated script automatically sets `INR_REPO_PATH` to point to the `submodules/multi_contrast_inr` submodule, so no manual configuration is needed
- The script uses the INR repository's `main.py` to train each case
- Make sure the INR repository is properly set up and accessible

