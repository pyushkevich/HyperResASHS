## Tools

### 1. `make_t1_ashs_from_t2_ashs.py`

Converts a T2 ASHS format dataset to T1 ASHS format by cropping whole-brain T1 images using T2 ROI bounding boxes.

**Input structure (T2 ASHS path):**
```
{t2_ashs_path}/
  {case_id}/
    tse_native_chunk_left.nii.gz    (T2 ROI)
    tse_native_chunk_right.nii.gz   (T2 ROI)
    mprage.nii.gz                    (whole-brain T1)
```

**Output structure (T1 ASHS path):**
```
{output_ashs_path}/
  train{xxx}/
    tse_native_chunk_left.nii.gz    (cropped T1 ROI)
    tse_native_chunk_right.nii.gz   (cropped T1 ROI)
```

**Note:** Case folders are automatically renamed to `train{xxx}` format where `xxx` is a 3-digit zero-padded number (e.g., `train001`, `train002`)

**Usage:**
```bash
python convert_dataset/make_t1_ashs_from_t2_ashs.py \
    --t2-ashs-path /path/to/t2/ashs \
    --output-ashs-path /path/to/output/t1/ashs
```

**Optional arguments:**
- `--t2-roi-name`: Filename pattern for T2 ROI (default: `tse_native_chunk_{side}.nii.gz`)
- `--t1-whole-name`: Filename for whole-brain T1 (default: `mprage.nii.gz`)
- `--t1-output-name`: Output filename pattern for T1 ROI (default: `tse_native_chunk_{side}.nii.gz`)

**Example:**
```bash
python convert_dataset/make_t1_ashs_from_t2_ashs.py \
    --t2-ashs-path /data/t2_ashs \
    --output-ashs-path /data/t1_ashs
```

### 2. `create_five_fold_json.py`

Creates a `five_fold.json` file for 5-fold cross-validation from case folder names in a directory.

**Usage:**
```bash
python convert_dataset/create_five_fold_json.py \
    --input-dir /path/to/t2_ashs \
    --output /path/to/five_fold.json
```

**Optional arguments:**
- `--seed`: Random seed for reproducibility (default: `42`)

**Example:**
```bash
python convert_dataset/create_five_fold_json.py \
    --input-dir /data/t2_ashs \
    --output /data/five_fold.json \
    --seed 42
```

**Output format:**
The script extracts numeric IDs from case folder names and creates a JSON file with 5-fold splits:

```json
{
  "fold_0": ["001", "005", "009"],
  "fold_1": ["002", "006", "010"],
  "fold_2": ["003", "007", "011"],
  "fold_3": ["004", "008", "012"],
  "fold_4": ["013", "014", "015"]
}
```

