# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

HyperResASHS segments medial temporal lobe (MTL) subregions from paired T1w (MPRAGE) and T2w (TSE) 3T MRI. It produces isotropic segmentations using nnU-Net and trains on segmentations that were upsampled with an Implicit Neural Representation (INR). It is distributed on PyPI as `hyperresashs`, and its CLI is `hrashs`.

## Setup / commands

- Clone with submodules: `git clone --recursive ...` or `git submodule update --init --recursive`. Both submodules are required at import time.
- Install for development: install `torch==2.5.1` (plus matching torchvision/torchaudio) for the local CUDA version first, then run `pip install -e .`. Newer PyTorch versions are known to break the code.
- Smoke test: `hrashs check`. It checks the Hugging Face connection, imports the vendored nnU-Net and INR code, and checks torch/CUDA.
- Inference: `hrashs run -a <atlas_id_or_dir> -g t1.nii.gz -f t2.nii.gz -w <workdir> [-I subj] [-t N] [--device cpu]`. If `-f` is omitted, it runs in T1-only mode.
- Training: `hrashs train -c atlas.yaml -m manifest.csv -l labels.txt -w <workdir> [-s 1-3] [-F regex]`. `-s` selects stages: 1 preprocess, 2 INR, 3 nnU-Net prep, 4 nnU-Net train, 5 finalize.
- Other subcommands: `list`, `desc <id>`, `upload`. You can also run the CLI with `python -m hyperresashs`.
- Build: `python -m build`. There are no tests and no linter configured. Verify changes by running `hrashs check` and a real `run`/`train` on sample data. A GPU is needed for practical runtimes.

## Architecture

### Two generations of code
- **Current (`hrashs` CLI):** `ashs_cli.py` → `ashs_inference.py` (`HyperASHSInference`) and `ashs_training.py` (`HyperASHSTraining`). Both are built on `ashs_preproc.py` (`ASHSProcessor`) and `ashs_exp.py`. New work goes here.
- **Legacy (Yue Li's original research pipeline):** `main.py`, `preprocessing.py`, `prepare_inr.py`, `testing.py`, and `shell/*.sh`. They are driven by numbered YAML files in `config/` and `config_test/` (`config_<EXP_NUM>_<name>.yaml`) and by `-s prepare|prepare_inr|run_inr|preprocess|train|test`. `main.py` is not a packaged entry point. `docs/configuration.md` and the `docs/*shell_script.md` files describe this legacy flow. These modules are not imported by the CLI.

### Config bridging
`ashs_cli.write_config_files()` converts an atlas config (`atlas.yaml` from a Hugging Face repo or a local dir, with `metadata`, `config`, `template`, and optional `cleanup` sections) into the flat uppercase-key dict that the pipeline expects (`EXP_NUM`, `MODEL_NAME`, `TRAINER`, `TEMPLATE_PATH`, `ATLAS_PATH`, `SIDES`, `FILE_NAME_CONFIG`, ...). It writes that dict to `<workdir>/config/`. The comments call this "the way Yue's code expects it". The nnU-Net dataset id is `Dataset{EXP_NUM}_{MODEL_NAME}`.

### Filename scheme and lazy images
- Every intermediate and output path comes from a YAML filename scheme in `hyperresashs/config_templates/` (`ashs_filename_scheme.yaml`, plus a BIDS variant). Templates use `{prefix}`, `{side}`, and `{dataset}` placeholders. To rename or add an output, edit the scheme and add the matching attribute in `ashs_exp.py`.
- `ashs_exp.py` binds the scheme to a case directory:
  - `GlobalPipelineElements` (`gpe`) holds per-subject items.
  - `LocalPipelineElements` (`lpe[side]`) holds per-hemisphere items.
  - `TemplatePipelineElements` (`tpe`) holds template/ROI images.
  - Images are wrapped in `LazyImage`/`LazyInt16Image`: reading `.data` loads from disk, and assigning `.data` saves immediately. `exists()` lets steps be skipped when `overwrite_existing` is off.
- `ASHSProcessor` contains the shared steps:
  - `preprocess`: neck trim, then T1↔T2 and template→T1 registration via `picsl_greedy`/`picsl_c3d`, then ROI cropping, then hyper-res patch creation.
  - `prepare_inr`
  - `postprocess`: back to native space, volumes CSV, QC PNGs.
- Inference and training both drive these steps through an `ASHSExperimentBase`.

### Sides
Most structures are segmented per hemisphere, with a separate nnU-Net model for each value in `SIDES` (`left`/`right`). Non-lateralized structures use `mid`.

### Vendored submodules
Both submodules are forks maintained by pyushkevich:
- `hyperresashs/submodules/nnUNet`: the `nnUNet_modaug` fork, `mmseg` branch. It provides custom trainers such as `ModAugUNetTrainer` and `ModAugPartialCoverageUNetTrainer` under `nnunetv2/training/nnUNetTrainer/variants/data_augmentation/` (per the README; the submodules may not be checked out in a given clone).
- `hyperresashs/submodules/multi_contrast_inr`

`hyperresashs/__init__.py` puts both submodules at the front of `sys.path` and evicts any already-imported `nnunetv2`, so the vendored copies win over a pip-installed nnU-Net. Import `nnunetv2` lazily inside functions after `hyperresashs` has been imported. `HyperASHSTraining` sets the `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` env vars to directories under the training workdir, and calls the nnU-Net APIs (plan/preprocess/run_training) in-process rather than through the CLI.

### Atlases / models
Pre-trained atlases are Hugging Face Hub repos, listed through an atlas-directory repo (`get_atlas_listing` in `ashs_cli.py`). `hrashs train` stage 5 (`finalize`) writes a deployable atlas dir to `<workdir>/final/atlas/`, and `hrashs upload` pushes it to the Hub. `-k` disables SSL verification for Hugging Face and torch hub downloads, for use behind corporate proxies.

## Release
Bump `version` in `pyproject.toml`. Publishing a GitHub release triggers the PyPI upload workflow, which checks out the submodules recursively.
