# HyperResASHS

## Changelog

- 11/20/2025: add main pipeline of preprocessing
- 10/27/2025: This is the code for project of isotropic segmentation for MTL subregions in 3T-T2w and 3T-T1w multi-modality MRI. The code will be checked and released here step-by-step. Coming soon!


## Pipeline
The `main.py` file defines the pipeline of our manuscript [https://doi.org/10.48550/arXiv.2508.17171](https://doi.org/10.48550/arXiv.2508.17171).

It needs a modified nnU-Net and a modified Implicit Neural Representation. I will added these two models as submodules soon.

There are three main steps as shown in the manuscript: 1. build the isotropic atlas; 2. training multi-modality segmentation model; 3. run inference.

In `main.py`, the option flags include `-c` and `-s`. The -c controls the config ID that corresponds to different experiments and -s represents different steps.

For `-s`, `prepare_inr` represents step 1 (build the isotropic atlas), `preprocess` represents the preparation of step 2 (the training process is handled by nnU-Net, not here), and `test` represents the model inference.

## Test
This step contains more details then the other two because we suppose the test data is not preprocessed. We need to 1. register whole brain T1w and T2w images; 2. locate the left and right MTL ROIs using ASHS template and deformable registration; 3. crop the MTL patches using extracted ROIs; 4. upsample the patches to isotropic space; 5. register local patches to fix tiny misalignment; 6. run the inference of trained nnU-Net

The input path of this step is set by variable `TEST_PATH` in test config in `config_test`. The folder should have three levels: 1. participant anonymous ID； 2. scanning date (one participant can have more than one scan); 3. T1w and T2w MR images with file names `t1_img.nii.gz` and `t2_img.nii.gz`