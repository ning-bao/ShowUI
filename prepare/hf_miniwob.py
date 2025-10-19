#!/usr/bin/env python3
"""
Preprocess MiniWob++ dataset from HF parquet format to ShowUI-compatible JSON.
Download first with:
  huggingface-cli download LucasThil/miniwob_plusplus_v2_raw --repo-type dataset --local-dir $DATA_DIR/MiniWob

Usage:
  python prepare/hf_miniwob.py --data_root $DATA_DIR/MiniWob
"""
import os
import json
import glob
import argparse
from typing import List, Dict, Tuple
import ast

try:
    import pyarrow.parquet as pq
except ImportError:
    raise ImportError("pyarrow required. Install: pip install pyarrow")


def pick_node_from_dom(dom_tree: dict) -> dict:
    """Pick the best node from DOM tree: focused > clickable > first."""
    try:
        q = [dom_tree]
        candidates = []
        while q:
            n = q.pop(0)
            if isinstance(n, dict):
                if n.get("focused") is True:
                    return n
                candidates.append(n)
                for c in n.get("children", []) or []:
                    q.append(c)
        # Prefer clickable-like nodes
        for n in candidates:
            classes = str(n.get("classes", "")).lower()
            if any(k in classes for k in ["button", "link", "click", "reply", "like", "submit"]):
                return n
        return candidates[0] if candidates else {}
    except Exception:
        return {}


def process_miniwob_item(item: dict, default_viewport: Tuple[float, float] = (160.0, 210.0)) -> dict:
    """Convert a MiniWob++ parquet row into ShowUI multi-turn format."""
    task_name = item.get("task_name", "") or item.get("subdomain", "")
    processed_states_raw = item.get("processed_states", "")
    
    # Parse processed_states (Python repr string)
    if isinstance(processed_states_raw, str):
        try:
            processed_states = ast.literal_eval(processed_states_raw)
        except Exception:
            return None
    else:
        processed_states = processed_states_raw
    
    if not processed_states or not isinstance(processed_states, list):
        return None
    
    # Infer viewport from first DOM tree
    img_w, img_h = default_viewport
    try:
        first_dom = processed_states[0].get("dom", {})
        if isinstance(first_dom, dict):
            w = first_dom.get("width")
            h = first_dom.get("height")
            if isinstance(w, (int, float)) and isinstance(h, (int, float)) and w > 0 and h > 0:
                img_w, img_h = float(w), float(h)
    except Exception:
        pass
    
    # Extract steps
    steps = []
    for i, step_dict in enumerate(processed_states):
        action_type = step_dict.get("action_type", "click")
        dom_tree = step_dict.get("dom", {})
        
        # Pick target node from DOM
        node = pick_node_from_dom(dom_tree)
        if not node or not isinstance(node, dict):
            continue
        
        try:
            left = float(node.get("left", 0.0))
            top = float(node.get("top", 0.0))
            width = float(node.get("width", 0.0))
            height = float(node.get("height", 0.0))
            if width <= 0 or height <= 0:
                continue
            
            # Normalize to [0, 1]
            cx = (left + width / 2.0) / img_w
            cy = (top + height / 2.0) / img_h
            cx = min(1.0, max(0.0, cx))
            cy = min(1.0, max(0.0, cy))
            
            steps.append({
                "instruction": f"{task_name} [{action_type}]",
                "point": [cx, cy],
                "img_url": f"step_{i}.png"  # placeholder
            })
        except Exception:
            continue
    
    if len(steps) == 0:
        return None
    
    return {
        "task_name": task_name,
        "base_img_url": None,
        "steps": steps
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess MiniWob++ dataset")
    parser.add_argument("--data_root", type=str, required=True, help="Path to MiniWob dataset root")
    parser.add_argument("--split", type=str, default="train", help="Split to process (train/test)")
    parser.add_argument("--max_items", type=int, default=None, help="Max items to process (for debugging)")
    args = parser.parse_args()
    
    data_dir = os.path.join(args.data_root, "data")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"MiniWob++ data folder not found: {data_dir}")
    
    # Find parquet files for split
    pattern = f"{args.split}-*.parquet"
    shard_files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not shard_files:
        raise FileNotFoundError(f"No parquet shards matching {pattern} in {data_dir}")
    
    print(f"Found {len(shard_files)} parquet shards for split '{args.split}'")
    
    # Read and process
    raw_items = []
    for fp in shard_files:
        print(f"Reading {os.path.basename(fp)}...")
        table = pq.read_table(fp)
        raw_items.extend(table.to_pylist())
        if args.max_items and len(raw_items) >= args.max_items:
            raw_items = raw_items[:args.max_items]
            break
    
    print(f"Processing {len(raw_items)} raw items...")
    processed = []
    skipped = 0
    
    for idx, item in enumerate(raw_items):
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(raw_items)}...")
        
        result = process_miniwob_item(item)
        if result:
            processed.append(result)
        else:
            skipped += 1
    
    print(f"Processed {len(processed)} valid items, skipped {skipped}")
    
    # Save to metadata folder
    metadata_dir = os.path.join(args.data_root, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    
    output_file = os.path.join(metadata_dir, f"miniwob_{args.split}.json")
    with open(output_file, "w") as f:
        json.dump(processed, f, indent=2)
    
    print(f"Saved {len(processed)} items to {output_file}")
    print(f"\nTo use in training, load with:")
    print(f"  --train_dataset miniwob --train_json miniwob_{args.split}")


if __name__ == "__main__":
    main()
