#!/usr/bin/env python3
"""
Export annotation pipeline data to ShowUI-desktop format.
Outputs parquet files and images in the standard ShowUI-desktop structure.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import db as dbm


def normalize_bbox(bbox_abs, img_width, img_height):
    """Convert absolute bbox to normalized [0-1] coordinates."""
    x1, y1, x2, y2 = bbox_abs
    return [
        x1 / img_width,
        y1 / img_height,
        x2 / img_width,
        y2 / img_height
    ]


def normalize_point(point_abs, img_width, img_height):
    """Convert absolute point to normalized [0-1] coordinates."""
    x, y = point_abs
    return [x / img_width, y / img_height]


def export_to_showui_desktop(images_root, annotations_root, output_root, split='train', filenames_filter=None):
    """
    Export annotations to ShowUI-desktop format.
    
    Args:
        images_root: Path to images folder
        annotations_root: Path to annotations folder
        output_root: Path to output ShowUI-desktop folder
        split: Dataset split name (train/val/test)
        filenames_filter: Optional list of filenames to export (if None, export all)
    """
    images_root = Path(images_root)
    annotations_root = Path(annotations_root)
    output_root = Path(output_root)
    
    # Create output directories
    output_images = output_root / 'images'
    output_data = output_root / 'data'
    output_metadata = output_root / 'metadata'
    
    output_images.mkdir(parents=True, exist_ok=True)
    output_data.mkdir(parents=True, exist_ok=True)
    output_metadata.mkdir(parents=True, exist_ok=True)
    
    # Get all images from DB
    all_images = dbm.list_images(limit=100000)
    
    # Filter only annotated images
    annotated_images = [img for img in all_images if img['has_annotation']]
    
    # Apply filename filter if provided
    if filenames_filter:
        filenames_set = set(filenames_filter)
        annotated_images = [img for img in annotated_images if img['filename'] in filenames_set]
        print(f"Filtering to {len(annotated_images)} selected images")
    
    print(f"Found {len(annotated_images)} annotated images to export")
    
    # Prepare export data
    export_records = []
    copied_images = 0
    skipped = 0
    
    for img_record in annotated_images:
        filename = img_record['filename']
        img_path = images_root / filename
        
        # Construct correct annotation path (respecting folder structure)
        filename_path = Path(filename)
        ann_path = annotations_root / filename_path.parent / f"{filename_path.stem}.json"
        
        if not img_path.exists():
            print(f"⚠️  Image not found: {filename}")
            skipped += 1
            continue
            
        if not ann_path.exists():
            print(f"⚠️  Annotation not found: {filename}")
            skipped += 1
            continue
        
        # Load annotation
        try:
            with open(ann_path, 'r') as f:
                annotation = json.load(f)
        except Exception as e:
            print(f"⚠️  Failed to load annotation for {filename}: {e}")
            skipped += 1
            continue
        
        # Validate annotation structure
        if 'img_size' not in annotation or 'element' not in annotation:
            print(f"⚠️  Invalid annotation structure for {filename}")
            skipped += 1
            continue
        
        img_width, img_height = annotation['img_size']
        
        # Convert to normalized coordinates
        normalized_elements = []
        for elem in annotation['element']:
            try:
                # Get bbox and point in absolute coordinates
                bbox_abs = elem['bbox']
                point_abs = elem['point']
                instruction = elem.get('instruction', '')
                
                # Normalize
                bbox_norm = normalize_bbox(bbox_abs, img_width, img_height)
                point_norm = normalize_point(point_abs, img_width, img_height)
                
                normalized_elements.append({
                    'instruction': instruction,
                    'bbox': bbox_norm,
                    'point': point_norm
                })
            except Exception as e:
                print(f"⚠️  Failed to process element in {filename}: {e}")
                continue
        
        if not normalized_elements:
            print(f"⚠️  No valid elements for {filename}")
            skipped += 1
            continue
        
        # Copy image to output folder
        output_img_path = output_images / filename
        output_img_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(img_path, output_img_path)
            copied_images += 1
        except Exception as e:
            print(f"⚠️  Failed to copy image {filename}: {e}")
            skipped += 1
            continue
        
        # Add to export records
        export_records.append({
            'img_url': filename,
            'img_size': [img_width, img_height],
            'element': normalized_elements,
            'element_size': len(normalized_elements)
        })
    
    # Write metadata JSON (one record per line)
    metadata_file = output_metadata / f'hf_{split}.json'
    print(f"\nWriting metadata to {metadata_file}...")
    # Write as a valid JSON array (instead of JSONL)
    with open(metadata_file, 'w') as f:
        json.dump(export_records, f, ensure_ascii=False)
    
    # Try to write parquet if pandas/pyarrow available
    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        # Convert to DataFrame
        df = pd.DataFrame(export_records)
        
        # Write as parquet (single file for simplicity, can be sharded later)
        parquet_file = output_data / f'{split}-00000-of-00001.parquet'
        df.to_parquet(parquet_file, index=False, engine='pyarrow')
        print(f"✓ Wrote parquet: {parquet_file}")
    except ImportError:
        print("⚠️  pandas/pyarrow not available, skipping parquet export")
        print("   Install with: pip install pandas pyarrow")
    except Exception as e:
        print(f"⚠️  Failed to write parquet: {e}")
    
    # Write summary
    print(f"\n{'='*60}")
    print(f"Export Summary:")
    print(f"  Exported images: {copied_images}")
    print(f"  Skipped: {skipped}")
    print(f"  Total records: {len(export_records)}")
    print(f"  Output location: {output_root}")
    print(f"{'='*60}")
    
    # Write README
    readme_path = output_root / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(f"""# ShowUI-desktop Dataset Export

Exported from AccuAnnotate annotation pipeline.

## Structure

```
{output_root.name}/
├── images/           # Image files (preserves folder structure)
├── data/             # Parquet files with annotations
├── metadata/         # JSON metadata (one record per line)
└── README.md         # This file
```

## Format

- **images/**: Original images with preserved paths
- **metadata/hf_{split}.json**: JSON Lines format with normalized coordinates (0-1)
- **data/{split}-*.parquet**: Parquet files with the same structure

## Statistics

- Total images: {copied_images}
- Total annotations: {sum(r['element_size'] for r in export_records)}
- Average elements per image: {(sum(r['element_size'] for r in export_records) / len(export_records) if export_records else 0):.1f}

## Coordinate Format

All bounding boxes and points use normalized coordinates [0-1]:
- bbox: [x1_norm, y1_norm, x2_norm, y2_norm]
- point: [x_norm, y_norm]

To convert back to absolute coordinates:
```python
x_abs = x_norm * img_width
y_abs = y_norm * img_height
```

Generated by AccuAnnotate Export Tool
""")
    
    print(f"\n✓ Export complete!")
    print(f"  Location: {output_root}")


def main():
    parser = argparse.ArgumentParser(description='Export annotations to ShowUI-desktop format')
    parser.add_argument('--images', default='data/images', help='Path to images folder')
    parser.add_argument('--annotations', default='data/annotations', help='Path to annotations folder')
    parser.add_argument('--output', required=True, help='Output directory for ShowUI-desktop format')
    parser.add_argument('--split', default='train', help='Dataset split name (train/val/test)')
    parser.add_argument('--filenames', default=None, help='JSON array of filenames to export (optional)')
    args = parser.parse_args()
    
    # Initialize DB
    dbm.init_db()
    
    # Parse filenames filter if provided
    filenames_filter = None
    if args.filenames:
        try:
            filenames_filter = json.loads(args.filenames)
        except json.JSONDecodeError as e:
            print(f"Error parsing filenames: {e}")
            sys.exit(1)
    
    export_to_showui_desktop(
        args.images,
        args.annotations,
        args.output,
        args.split,
        filenames_filter
    )


if __name__ == '__main__':
    main()
