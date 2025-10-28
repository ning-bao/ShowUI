#!/usr/bin/env python3
"""
Migrate annotation files from root annotations folder to match folder structure.
This is a one-time migration script.
"""

import sys
from pathlib import Path
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import db as dbm

def migrate_annotations():
    """Migrate annotations to match image folder structure."""
    
    annotation_folder = Path('data/annotations')
    migrated = 0
    skipped = 0
    errors = []
    
    print("Starting annotation migration...")
    print("-" * 60)
    
    # Get all images from database
    images = dbm.list_images(limit=10000)
    
    for img in images:
        filename = img['filename']
        
        # Skip if image is in root (no folder)
        if '/' not in filename:
            continue
        
        filename_path = Path(filename)
        stem = filename_path.stem
        
        # Old location (root of annotations)
        old_path = annotation_folder / f"{stem}.json"
        
        # New location (preserving folder structure)
        new_path = annotation_folder / filename_path.parent / f"{stem}.json"
        
        # Skip if already in correct location
        if new_path.exists():
            skipped += 1
            continue
        
        # Migrate if old location exists
        if old_path.exists():
            try:
                new_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old_path), str(new_path))
                print(f"✓ Migrated: {old_path.name} -> {new_path.relative_to(annotation_folder)}")
                migrated += 1
            except Exception as e:
                error_msg = f"Failed to migrate {old_path.name}: {e}"
                print(f"✗ {error_msg}")
                errors.append(error_msg)
        else:
            # Annotation doesn't exist in old location either
            pass
    
    print("-" * 60)
    print(f"\nMigration complete!")
    print(f"  Migrated: {migrated}")
    print(f"  Skipped (already correct): {skipped}")
    print(f"  Errors: {len(errors)}")
    
    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  - {error}")
    
    return migrated, skipped, errors

if __name__ == '__main__':
    migrate_annotations()


