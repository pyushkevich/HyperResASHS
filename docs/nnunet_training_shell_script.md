# nnU-Net Training via Shell Script

Alternatively, you can run the generated training script manually instead of using the Python interface.

## Usage

Run the generated training script:

```bash
./scripts/train_nnunet_{EXP_NUM}{MODEL_NAME}.sh
# For example: ./scripts/train_nnunet_292IsotropiSeg.sh
```

## Training Parameters

The script trains all 5 folds (fold 0-4) with the following parameters:

- **Dataset ID**: `{EXP_NUM}` (from your config)
- **Configuration**: `3d_fullres`
- **Fold**: `0-4` (all 5 folds)
- **Trainer**: `{TRAINER}` (from your config, e.g., `ModAugUNetTrainer`)

## Notes

- Ensure you're using the modified nnUNet from `submodules/nnUNet` (mmseg branch) which includes Modality Augmentation methods
- The `nnUNetv2_train` command should be available after installing the modified nnUNet
- The script is automatically generated during preprocessing (Step 4) with paths and parameters filled from your configuration file

