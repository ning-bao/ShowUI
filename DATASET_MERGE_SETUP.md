# Dataset Merge Setup Guide

This guide explains how to prepare the merged ShowUI-desktop + ScreenSpot dataset on any remote instance.

## Prerequisites

1. **ShowUI codebase** - Clone or copy the ShowUI repository
2. **Datasets** - Download the required datasets:
   - `ScreenSpot` dataset
   - `ShowUI-desktop` dataset

## Quick Setup

### Option 1: Automated Setup (Recommended)

```bash
# Navigate to ShowUI root directory
cd /path/to/ShowUI

# Run the setup script
./setup_merged_dataset.sh ~/showui_data ~/showui_data/ShowUI-desktop-merged
```

### Option 2: Manual Setup

```bash
# Navigate to ShowUI root directory
cd /path/to/ShowUI

# Run the preparation script directly
python3 prepare_merged_dataset.py \
    --data_dir ~/showui_data \
    --output_dir ~/showui_data/ShowUI-desktop-merged \
    --screenspot_splits screenspot_desktop
```

## Dataset Structure Expected

```
~/showui_data/
├── ScreenSpot/
│   ├── images/
│   └── metadata/
│       ├── screenspot_desktop.json
│       ├── screenspot_mobile.json
│       └── screenspot_web.json
└── ShowUI-desktop/
    ├── images/
    └── metadata/
        └── hf_train.json
```

## Output Structure

After running the script, you'll get:

```
~/showui_data/ShowUI-desktop-merged/
├── images/
│   ├── showui -> ~/showui_data/ShowUI-desktop/images
│   └── screenspot -> ~/showui_data/ScreenSpot/images
└── metadata/
    └── hf_train.json
```

## Using the Merged Dataset

### 1. Update Dataset Mapping

Edit `data/dset_shared_grounding.py` and add the new dataset mapping:

```python
dataset_mapping = {
    "showui-desktop": "ShowUI-desktop",
    "showui-desktop-merged": "ShowUI-desktop-merged",  # Add this line
    "showui-web": "ShowUI-web",
    "amex": "AMEX",
    "rico": "RICO",
    "ricosca": "RICO",
    "widget": "RICO",
}
```

### 2. Run Training

Use the merged dataset in your training commands:

```bash
# For RL training
python rl_train.py \
    --dataset_dir ~/showui_data \
    --train_dataset showui-desktop-merged \
    --train_json hf_train \
    [other training arguments...]

# For regular training
python train.py \
    --dataset_dir ~/showui_data \
    --showui_data hf_train \
    [other training arguments...]
```

## Script Options

The `prepare_merged_dataset.py` script supports several options:

```bash
python3 prepare_merged_dataset.py \
    --data_dir ~/showui_data \                    # Base directory with datasets
    --output_dir ~/showui_data/ShowUI-desktop-merged \  # Output directory
    --showui_split hf_train \                     # ShowUI-desktop split to use
    --screenspot_splits screenspot_desktop \      # ScreenSpot splits to include
```

### Available ScreenSpot Splits

- `screenspot_desktop` - Desktop screenshots
- `screenspot_mobile` - Mobile screenshots  
- `screenspot_web` - Web screenshots

You can include multiple splits:

```bash
python3 prepare_merged_dataset.py \
    --data_dir ~/showui_data \
    --output_dir ~/showui_data/ShowUI-desktop-merged \
    --screenspot_splits screenspot_desktop screenspot_mobile
```

## Troubleshooting

### PIL/Pillow Not Available

If you get PIL import errors, the script will fall back to using default image dimensions (1920x1080). This is usually fine for most use cases.

### Permission Errors

Make sure you have write permissions to the output directory:

```bash
chmod -R 755 ~/showui_data/
```

### Symlink Issues

If symlinks don't work on your system, you can copy the images instead:

```bash
# Copy instead of symlink
cp -r ~/showui_data/ShowUI-desktop/images ~/showui_data/ShowUI-desktop-merged/images/showui
cp -r ~/showui_data/ScreenSpot/images ~/showui_data/ShowUI-desktop-merged/images/screenspot
```

## Verification

After setup, verify the dataset:

```bash
# Check structure
ls -la ~/showui_data/ShowUI-desktop-merged/

# Check metadata
wc -l ~/showui_data/ShowUI-desktop-merged/metadata/hf_train.json

# Check symlinks
ls -la ~/showui_data/ShowUI-desktop-merged/images/
```

## Expected Results

- **Total samples**: ~311 (varies based on splits included)
- **ShowUI-desktop samples**: ~101 (original)
- **ScreenSpot samples**: ~210 (converted from screenspot_desktop)

The merged dataset combines both datasets with proper image path prefixes to avoid conflicts.
