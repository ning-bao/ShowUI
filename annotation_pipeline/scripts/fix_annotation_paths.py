#!/usr/bin/env python3
"""
Fix annotation file paths by moving them to match their image folder structure.
"""
import sys
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import db as dbm

def main():
    dbm.init_db()
    images = dbm.list_images(limit=100000)
    annotated = [img for img in images if img['has_annotation']]
    
    images_root = Path('data/images')
    annotations_root = Path('data/annotations')
    
    moved = 0
    errors = []
    
    for img in annotated:
        filename = img['filename']
        filename_path = Path(filename)
        
        # Expected annotation path (with subfolder)
        correct_ann_path = annotations_root / filename_path.parent / f"{filename_path.stem}.json"
        
        # Old annotation path (in root)
        old_ann_path = annotations_root / f"{filename_path.stem}.json"
        
        # If annotation exists in root but not in correct subfolder
        if old_ann_path.exists() and not correct_ann_path.exists():
            try:
                # Create subfolder if needed
                correct_ann_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Move annotation
                shutil.move(str(old_ann_path), str(correct_ann_path))
                moved += 1
                print(f"✓ Moved: {old_ann_path.name} → {correct_ann_path.relative_to(annotations_root)}")
                
            except Exception as e:
                errors.append(f"Failed to move {filename}: {e}")
                print(f"✗ Error: {filename} - {e}")
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Moved: {moved} annotations")
    print(f"  Errors: {len(errors)}")
    print(f"{'='*60}")
    
    if errors:
        print("\nErrors:")
        for err in errors[:10]:
            print(f"  - {err}")

if __name__ == '__main__':
    main()


