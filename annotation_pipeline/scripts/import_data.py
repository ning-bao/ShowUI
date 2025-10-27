import argparse
import os
import sys
from pathlib import Path
import json

# Add parent directory to path so we can import db
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import db as dbm


def main():
    parser = argparse.ArgumentParser(description='Import existing images and annotations into SQLite metadata DB')
    parser.add_argument('--images', default='data/images', help='Path to images root folder')
    parser.add_argument('--annotations', default='data/annotations', help='Path to annotations root folder')
    args = parser.parse_args()

    images_root = Path(args.images)
    ann_root = Path(args.annotations)
    images_root.mkdir(parents=True, exist_ok=True)
    ann_root.mkdir(parents=True, exist_ok=True)

    dbm.init_db()

    total = 0
    updated = 0
    for p in images_root.rglob('*'):
        if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            rel = str(p.relative_to(images_root)).replace('\\', '/')
            ann = ann_root / f"{p.stem}.json"
            has_ann = ann.exists()
            try:
                size_b = p.stat().st_size
            except Exception:
                size_b = None
            dbm.upsert_image(rel, has_annotation=has_ann, size_bytes=size_b)
            total += 1
            if has_ann:
                updated += 1

    # also ensure folders from filesystem in DB (including empties)
    for dirpath, dirnames, filenames in os.walk(images_root):
        if Path(dirpath) == images_root:
            continue
        rel = Path(dirpath).relative_to(images_root)
        dbm.ensure_folder_chain(str(rel).replace('\\', '/'))

    print(f"Imported {total} images ({updated} with annotations).")


if __name__ == '__main__':
    main()


