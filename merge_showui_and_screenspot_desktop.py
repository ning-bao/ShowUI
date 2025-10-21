#!/usr/bin/env python3
import os
import json
from pathlib import Path
import argparse


def load_showui_desktop(showui_dir: Path, split: str):
    meta_file = showui_dir / 'metadata' / f'{split}.json'
    if not meta_file.exists():
        raise FileNotFoundError(f'Metadata not found: {meta_file}')
    with open(meta_file, 'r') as f:
        data = json.load(f)
    return data


def normalize_img_url_prefix(samples, prefix: str):
    normalized = []
    for item in samples:
        new_item = dict(item)
        # ensure images are referenced via subfolder prefix
        new_item['img_url'] = str(Path(prefix) / item['img_url'])
        normalized.append(new_item)
    return normalized


def main():
    parser = argparse.ArgumentParser(description='Merge ShowUI-desktop and ScreenSpot desktop (converted)')
    parser.add_argument('--showui_desktop', required=True, help='Path to ShowUI-desktop root')
    parser.add_argument('--screenspot_rl', required=True, help='Path to converted ScreenSpot ShowUI-RL root')
    parser.add_argument('--out_dir', required=True, help='Output merged dataset directory')
    parser.add_argument('--showui_split', default='hf_train', help='ShowUI-desktop split to use (default: hf_train)')
    parser.add_argument('--screenspot_split', default='screenspot_desktop', help='ScreenSpot split to use (default: screenspot_desktop)')
    args = parser.parse_args()

    showui_dir = Path(args.showui_desktop)
    screenspot_dir = Path(args.screenspot_rl)
    out_dir = Path(args.out_dir)
    (out_dir / 'metadata').mkdir(parents=True, exist_ok=True)
    images_out = out_dir / 'images'
    images_out.mkdir(exist_ok=True)

    # Load datasets
    showui_samples = load_showui_desktop(showui_dir, args.showui_split)
    screenspot_samples = load_showui_desktop(screenspot_dir, args.screenspot_split)

    # Prefix img_url so both can live under merged images/
    showui_prefixed = normalize_img_url_prefix(showui_samples, 'showui')
    screenspot_prefixed = normalize_img_url_prefix(screenspot_samples, 'screenspot')

    merged = showui_prefixed + screenspot_prefixed

    # Write merged metadata
    merged_meta = out_dir / 'metadata' / 'hf_train.json'
    with open(merged_meta, 'w') as f:
        json.dump(merged, f)

    # Create subfolder symlinks in images/
    # images/showui -> <ShowUI-desktop>/images
    # images/screenspot -> <ShowUI-RL>/images
    showui_link = images_out / 'showui'
    screenspot_link = images_out / 'screenspot'

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

    print('Merged dataset created:')
    print(f'  {out_dir}')
    print(f'  ├── images/')
    print(f'  │   ├── showui -> {showui_dir / "images"}')
    print(f'  │   └── screenspot -> {screenspot_dir / "images"}')
    print(f'  └── metadata/hf_train.json  (#samples={len(merged)})')


if __name__ == '__main__':
    main()
