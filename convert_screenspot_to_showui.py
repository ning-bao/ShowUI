#!/usr/bin/env python3
"""
Convert ScreenSpot dataset to ShowUI format for RL training.

ScreenSpot format:
- img_filename: image filename
- bbox: [x, y, width, height] in absolute pixels
- instruction: text instruction
- data_type: type of element (icon, text, etc.)
- data_source: source platform (windows, macos, etc.)

ShowUI format:
- img_url: path to image
- img_size: [width, height] in pixels
- element: array of elements
  - instruction: text instruction
  - bbox: [x1, y1, x2, y2] normalized (0-1)
  - point: [x, y] normalized (0-1)
- element_size: number of elements
"""

import json
import os
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


def convert_screenspot_to_showui(screenspot_data, images_dir, output_dir):
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


def main():
    parser = argparse.ArgumentParser(description='Convert ScreenSpot to ShowUI format')
    parser.add_argument('--screenspot_dir', required=True, 
                       help='Path to ScreenSpot dataset directory')
    parser.add_argument('--output_dir', required=True,
                       help='Path to output ShowUI-RL directory')
    parser.add_argument('--splits', nargs='+', default=['screenspot_desktop', 'screenspot_mobile', 'screenspot_web'],
                       help='ScreenSpot splits to convert')
    
    args = parser.parse_args()
    
    screenspot_dir = Path(args.screenspot_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'images').mkdir(exist_ok=True)
    (output_dir / 'metadata').mkdir(exist_ok=True)
    
    print(f"Converting ScreenSpot dataset from {screenspot_dir}")
    print(f"Output directory: {output_dir}")
    
    for split in args.splits:
        print(f"\nProcessing {split}...")
        
        # Load ScreenSpot data
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
            screenspot_data, str(images_dir), str(output_dir)
        )
        
        print(f"Converted to {len(converted_data)} ShowUI entries")
        
        # Save converted data
        output_file = output_dir / 'metadata' / f'{split}.json'
        with open(output_file, 'w') as f:
            json.dump(converted_data, f, indent=2)
        
        print(f"Saved converted data to {output_file}")
        
        # Copy images (create symlinks to save space)
        images_src = screenspot_dir / 'images'
        images_dst = output_dir / 'images'
        
        if images_src.exists():
            # Create symlink to images directory
            images_link = output_dir / 'images'
            if images_link.exists():
                if images_link.is_symlink():
                    images_link.unlink()
                else:
                    # Remove directory and create symlink
                    import shutil
                    shutil.rmtree(images_link)
            images_link.symlink_to(images_src)
            print(f"Created symlink to images: {images_link} -> {images_src}")
    
    print(f"\nConversion complete! ShowUI-RL dataset saved to {output_dir}")
    print(f"Dataset structure:")
    print(f"  {output_dir}/")
    print(f"  ├── images/ -> {screenspot_dir}/images")
    print(f"  └── metadata/")
    for split in args.splits:
        print(f"      ├── {split}.json")


if __name__ == '__main__':
    main()
