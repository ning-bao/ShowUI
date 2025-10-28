# Adding New Export Formats

This guide explains how to add new dataset export formats to the annotation pipeline.

## Overview

The export system is designed to be extensible. Currently supported:
- **ShowUI-Desktop**: The default format

## How to Add a New Format

### 1. Create Export Script

Create a new Python script in `scripts/` (e.g., `export_coco.py`):

```python
#!/usr/bin/env python3
"""
Export annotations to COCO format.
"""

import argparse
import json
from pathlib import Path

def export_to_coco(images_dir, annotations_dir, output_dir, split_name, filenames):
    """
    Convert annotations to COCO format.
    
    Args:
        images_dir: Path to images folder
        annotations_dir: Path to annotations folder
        output_dir: Output directory
        split_name: Dataset split name (train/val/test)
        filenames: List of image filenames to export
    """
    # Your export logic here
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', required=True)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', default='train')
    parser.add_argument('--filenames', default='[]')
    
    args = parser.parse_args()
    filenames = json.loads(args.filenames) if args.filenames else []
    
    export_to_coco(
        Path(args.images),
        Path(args.annotations),
        Path(args.output),
        args.split,
        filenames
    )
```

### 2. Update Frontend (HTML)

Add the new format option in `templates/index.html`:

```html
<select id="exportFormat" class="filter-select">
    <option value="showui-desktop">ShowUI-Desktop</option>
    <option value="coco">COCO Format</option>  <!-- ADD THIS -->
    <option value="yolo">YOLO Format</option>  <!-- Or this -->
</select>
```

### 3. Update Backend (app.py)

Add the format handler in the `/api/export` route:

```python
@app.route('/api/export', methods=['POST'])
def export_dataset():
    # ... existing code ...
    
    # Route to appropriate export script based on format
    if export_format == 'showui-desktop':
        script_path = Path(__file__).parent / 'scripts' / 'export_showui_desktop.py'
        # ... existing ShowUI export code ...
    
    elif export_format == 'coco':  # ADD THIS BLOCK
        script_path = Path(__file__).parent / 'scripts' / 'export_coco.py'
        images_path = Path(app.config['UPLOAD_FOLDER'])
        annotations_path = Path(app.config['ANNOTATION_FOLDER'])
        
        result = subprocess.run([
            'python3', str(script_path),
            '--images', str(images_path),
            '--annotations', str(annotations_path),
            '--output', str(output_dir),
            '--split', split_name,
            '--filenames', json.dumps(filenames)
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            return jsonify({'error': 'Export failed', 'details': result.stderr}), 500
    
    else:
        return jsonify({'error': f'Export format "{export_format}" not implemented yet'}), 400
    
    # ... rest of the code ...
```

## Current Annotation Format

Our internal annotation format:

```json
{
  "img_size": [width, height],
  "element": [
    {
      "instruction": "Click the submit button",
      "bbox": [x1, y1, x2, y2],
      "point": [center_x, center_y]
    }
  ]
}
```

- Coordinates are in absolute pixels
- `bbox`: [left, top, right, bottom]
- `point`: [x, y] center point

## Export Features

All export formats automatically support:
- **Selective Export**: Only export selected images
- **Split Names**: Organize by train/val/test
- **ZIP Download**: Automatic compression for easy download
- **Progress Tracking**: Real-time progress updates

## Testing

After adding a new format, test it by:

1. Select some images in the UI
2. Click "📦 Export Dataset"
3. Choose your new format from the dropdown
4. Enter a split name
5. Click "Export"
6. Download the ZIP file

The system will automatically:
- Run your export script
- Count exported images
- Create a ZIP file
- Provide download link


