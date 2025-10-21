#!/usr/bin/env python3
"""
Portable script to prepare merged ShowUI-desktop + ScreenSpot dataset.
Run this on any instance where you have the ShowUI codebase and datasets.

Usage:
    python3 prepare_merged_dataset.py --data_dir ~/showui_data --output_dir ~/showui_data/ShowUI-desktop-merged
"""

import os
import json
import sys
from pathlib import Path
import argparse

# Try to import PIL, fallback to default size if not available
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available, using default image size (1920x1080)")


def convert_bbox_to_normalized(bbox, img_width, img_height):
    """Convert absolute bbox to normalized coordinates."""
    x, y, width, height = bbox
    
    # Convert to [x1, y1, x2, y2] format
    x1 = x
    y1 = y
    x2 = x + width
    y2 = y + height
    
    # Normalize to [0, 1]
    x1_norm = x1 / img_width
    y1_norm = y1 / img_height
    x2_norm = x2 / img_width
    y2_norm = y2 / img_height
    
    return [x1_norm, y1_norm, x2_norm, y2_norm]


def get_center_point(bbox):
    """Get center point from bbox."""
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return [center_x, center_y]


def get_image_size(image_path):
    """Get image dimensions."""
    if PIL_AVAILABLE:
        try:
            with Image.open(image_path) as img:
                return img.size  # Returns (width, height)
        except Exception as e:
            print(f"Warning: Could not get image size for {image_path}: {e}")
            return (1920, 1080)  # Default size
    else:
        # Use default size when PIL is not available
        return (1920, 1080)


def convert_screenspot_to_showui(screenspot_data, images_dir):
    """Convert ScreenSpot data to ShowUI format."""
    
    # Group data by image filename
    image_groups = {}
    for item in screenspot_data:
        img_filename = item['img_filename']
        if img_filename not in image_groups:
            image_groups[img_filename] = []
        image_groups[img_filename].append(item)
    
    converted_data = []
    
    for img_filename, items in image_groups.items():
        # Get image path and size
        img_path = os.path.join(images_dir, img_filename)
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
            
        img_width, img_height = get_image_size(img_path)
        
        # Convert elements
        elements = []
        for item in items:
            # Convert bbox to normalized coordinates
            bbox_norm = convert_bbox_to_normalized(
                item['bbox'], img_width, img_height
            )
            
            # Get center point
            point = get_center_point(bbox_norm)
            
            element = {
                'instruction': item['instruction'],
                'bbox': bbox_norm,
                'point': point
            }
            elements.append(element)
        
        # Create ShowUI format entry
        showui_entry = {
            'img_url': img_filename,
            'img_size': [img_width, img_height],
            'element': elements,
            'element_size': len(elements)
        }
        
        converted_data.append(showui_entry)
    
    return converted_data


def load_showui_desktop(showui_dir: Path, split: str):
    """Load ShowUI-desktop dataset."""
    meta_file = showui_dir / 'metadata' / f'{split}.json'
    if not meta_file.exists():
        raise FileNotFoundError(f'Metadata not found: {meta_file}')
    with open(meta_file, 'r') as f:
        data = json.load(f)
    return data


def normalize_img_url_prefix(samples, prefix: str):
    """Add prefix to img_url paths."""
    normalized = []
    for item in samples:
        new_item = dict(item)
        # ensure images are referenced via subfolder prefix
        new_item['img_url'] = str(Path(prefix) / item['img_url'])
        normalized.append(new_item)
    return normalized


def main():
    parser = argparse.ArgumentParser(description='Prepare merged ShowUI-desktop + ScreenSpot dataset')
    parser.add_argument('--data_dir', required=True, help='Base directory containing ScreenSpot and ShowUI-desktop')
    parser.add_argument('--output_dir', required=True, help='Output merged dataset directory')
    parser.add_argument('--showui_split', default='hf_train', help='ShowUI-desktop split to use')
    parser.add_argument('--screenspot_splits', nargs='+', default=['screenspot_desktop'], 
                       help='ScreenSpot splits to convert and merge')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Check required directories exist
    screenspot_dir = data_dir / 'ScreenSpot'
    showui_dir = data_dir / 'ShowUI-desktop'
    
    if not screenspot_dir.exists():
        raise FileNotFoundError(f"ScreenSpot directory not found: {screenspot_dir}")
    if not showui_dir.exists():
        raise FileNotFoundError(f"ShowUI-desktop directory not found: {showui_dir}")
    
    print(f"Converting ScreenSpot dataset from {screenspot_dir}")
    print(f"Loading ShowUI-desktop from {showui_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'images').mkdir(exist_ok=True)
    (output_dir / 'metadata').mkdir(exist_ok=True)
    
    # Load and convert ScreenSpot data
    all_screenspot_data = []
    for split in args.screenspot_splits:
        print(f"\nProcessing ScreenSpot {split}...")
        
        screenspot_file = screenspot_dir / 'metadata' / f'{split}.json'
        if not screenspot_file.exists():
            print(f"Warning: {screenspot_file} not found, skipping...")
            continue
            
        with open(screenspot_file, 'r') as f:
            screenspot_data = json.load(f)
        
        print(f"Loaded {len(screenspot_data)} items from {split}")
        
        # Convert to ShowUI format
        images_dir = screenspot_dir / 'images'
        converted_data = convert_screenspot_to_showui(
            screenspot_data, str(images_dir)
        )
        
        print(f"Converted to {len(converted_data)} ShowUI entries")
        all_screenspot_data.extend(converted_data)
    
    # Load ShowUI-desktop data
    print(f"\nLoading ShowUI-desktop {args.showui_split}...")
    showui_samples = load_showui_desktop(showui_dir, args.showui_split)
    print(f"Loaded {len(showui_samples)} ShowUI-desktop entries")
    
    # Prefix img_url so both can live under merged images/
    showui_prefixed = normalize_img_url_prefix(showui_samples, 'showui')
    screenspot_prefixed = normalize_img_url_prefix(all_screenspot_data, 'screenspot')
    
    # Merge datasets
    merged = showui_prefixed + screenspot_prefixed
    
    # Write merged metadata
    merged_meta = output_dir / 'metadata' / 'hf_train.json'
    with open(merged_meta, 'w') as f:
        json.dump(merged, f, indent=2)
    
    # Create subfolder symlinks in images/
    showui_link = output_dir / 'images' / 'showui'
    screenspot_link = output_dir / 'images' / 'screenspot'
    
    def ensure_symlink(link_path: Path, target: Path):
        if link_path.exists() or link_path.is_symlink():
            if link_path.is_symlink() or link_path.is_file():
                link_path.unlink()
            else:
                # it's a directory
                import shutil
                shutil.rmtree(link_path)
        link_path.symlink_to(target)
    
    ensure_symlink(showui_link, showui_dir / 'images')
    ensure_symlink(screenspot_link, screenspot_dir / 'images')
    
    print(f"\n✅ Merged dataset created successfully!")
    print(f"📁 Location: {output_dir}")
    print(f"📊 Total samples: {len(merged)} ({len(showui_prefixed)} ShowUI + {len(screenspot_prefixed)} ScreenSpot)")
    print(f"📁 Structure:")
    print(f"  {output_dir}/")
    print(f"  ├── images/")
    print(f"  │   ├── showui -> {showui_dir / 'images'}")
    print(f"  │   └── screenspot -> {screenspot_dir / 'images'}")
    print(f"  └── metadata/hf_train.json")
    
    print(f"\n🚀 To use this dataset:")
    print(f"1. Update dataset mapping in data/dset_shared_grounding.py:")
    print(f'   "showui-desktop-merged": "ShowUI-desktop-merged"')
    print(f"2. Run training with:")
    print(f"   --dataset_dir {data_dir}")
    print(f"   --train_dataset showui-desktop-merged")


if __name__ == '__main__':
    main()
