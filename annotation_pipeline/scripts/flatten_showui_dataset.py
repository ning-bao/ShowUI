#!/usr/bin/env python3
"""
Flatten ShowUI-desktop dataset folder structure.
Converts folder/subfolder/image.png -> folder_subfolder_image.png
"""

import os
import shutil
from pathlib import Path
import argparse

def flatten_dataset(source_dir, output_dir=None, dry_run=False):
    """
    Flatten folder structure by renaming files.
    
    Args:
        source_dir: Source directory (e.g., ~/showui_data/ShowUI-desktop)
        output_dir: Output directory (if None, creates 'flattened' in source parent)
        dry_run: If True, only print what would be done
    """
    source_path = Path(source_dir).expanduser().resolve()
    
    if not source_path.exists():
        print(f"Error: Source directory not found: {source_path}")
        return
    
    # Default output directory
    if output_dir:
        output_path = Path(output_dir).expanduser().resolve()
    else:
        output_path = source_path.parent / f"{source_path.name}_flattened"
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    print(f"Dry run: {dry_run}")
    print("-" * 60)
    
    # Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
    
    copied_count = 0
    skipped_count = 0
    error_count = 0
    
    # Walk through all files in source directory
    for file_path in source_path.rglob('*'):
        if not file_path.is_file():
            continue
        
        # Check if it's an image
        if file_path.suffix.lower() not in image_extensions:
            continue
        
        # Get relative path from source
        rel_path = file_path.relative_to(source_path)
        
        # If file is directly in source (no subfolder), keep original name
        if len(rel_path.parts) == 1:
            new_name = rel_path.name
        else:
            # Replace path separators with underscores
            # e.g., app_store/screen_1.png -> app_store_screen_1.png
            parts = list(rel_path.parts[:-1]) + [rel_path.stem]
            new_name = '_'.join(parts) + rel_path.suffix
        
        dest_path = output_path / new_name
        
        # Check if destination already exists
        if dest_path.exists() and not dry_run:
            # Add counter to make unique
            counter = 1
            while dest_path.exists():
                stem = '_'.join(parts) + f"_{counter}"
                dest_path = output_path / (stem + rel_path.suffix)
                counter += 1
            print(f"  ⚠️  Duplicate: {rel_path} -> {dest_path.name}")
        
        if dry_run:
            print(f"  Would copy: {rel_path} -> {new_name}")
            copied_count += 1
        else:
            try:
                shutil.copy2(file_path, dest_path)
                print(f"  ✓ Copied: {rel_path} -> {dest_path.name}")
                copied_count += 1
            except Exception as e:
                print(f"  ✗ Error: {rel_path} - {e}")
                error_count += 1
    
    print("-" * 60)
    print(f"\nSummary:")
    print(f"  Copied: {copied_count}")
    print(f"  Errors: {error_count}")
    print(f"\nOutput directory: {output_path}")
    
    if dry_run:
        print("\n⚠️  This was a dry run. Use --execute to actually copy files.")

def main():
    parser = argparse.ArgumentParser(description='Flatten ShowUI-desktop folder structure')
    parser.add_argument('--source', default='~/showui_data/ShowUI-desktop',
                        help='Source directory (default: ~/showui_data/ShowUI-desktop)')
    parser.add_argument('--output', default=None,
                        help='Output directory (default: source_flattened)')
    parser.add_argument('--execute', action='store_true',
                        help='Actually perform the operation (default is dry run)')
    
    args = parser.parse_args()
    
    flatten_dataset(
        source_dir=args.source,
        output_dir=args.output,
        dry_run=not args.execute
    )

if __name__ == '__main__':
    main()


