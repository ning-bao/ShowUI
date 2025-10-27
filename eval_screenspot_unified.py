#!/usr/bin/env python3
"""
Unified evaluation script for ScreenSpot V1, V2, and Pro datasets.
Automatically detects dataset format and evaluates accordingly.

Usage:
  # V1 (original ScreenSpot)
  python eval_screenspot_unified.py --dataset_path ~/showui_data/ScreenSpot --split hf_test_full

  # V2
  python eval_screenspot_unified.py --dataset_path ~/showui_data/ScreenSpot-v2 --split desktop

  # Pro
  python eval_screenspot_unified.py --dataset_path ~/showui_data/ScreenSpot-Pro
"""

import os
import ast
import re
import json
import argparse
from typing import List, Dict, Any, Optional, Set, Tuple
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


def parse_coord(output_text: str) -> Tuple[float, float]:
    """Parse model output to extract [x, y] coordinates."""
    try:
        xy = ast.literal_eval(output_text)
        if isinstance(xy, (list, tuple)) and len(xy) == 2:
            return float(xy[0]), float(xy[1])
    except Exception:
        pass
    try:
        m = re.search(r"[\[\(]?\s*([-+]?[0-9]*\.?[0-9]+)\s*,\s*([-+]?[0-9]*\.?[0-9]+)\s*[\]\)]?", output_text)
        if m:
            return float(m.group(1)), float(m.group(2))
    except Exception:
        pass
    return float("nan"), float("nan")


def detect_dataset_format(dataset_path: str) -> str:
    """Detect whether dataset is V1, V2, or Pro format."""
    # Check for Pro format (FiftyOne)
    if os.path.exists(os.path.join(dataset_path, "samples.json")) and \
       os.path.exists(os.path.join(dataset_path, "metadata.json")):
        return "pro"
    
    # Check for V2 format (separate JSON files)
    v2_files = ["screenspot_desktop_v2.json", "screenspot_mobile_v2.json", "screenspot_web_v2.json"]
    if any(os.path.exists(os.path.join(dataset_path, f)) for f in v2_files):
        return "v2"
    
    # Check for V1 format (metadata/ folder)
    if os.path.exists(os.path.join(dataset_path, "metadata")):
        return "v1"
    
    raise ValueError(f"Cannot detect dataset format for {dataset_path}")


def load_v1_items(dataset_path: str, split: str) -> Tuple[List[dict], str]:
    """Load V1 format (original ScreenSpot)."""
    meta_path = os.path.join(dataset_path, "metadata", f"{split}.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"V1 metadata not found: {meta_path}")
    
    with open(meta_path) as f:
        items = json.load(f)
    
    images_root = os.path.join(dataset_path, "images")
    
    # Normalize to common format
    for item in items:
        if "img_url" in item:
            item["img_filename"] = item["img_url"]
        if "task" in item:
            item["instruction"] = item["task"]
        # bbox is already [x, y, w, h]
        # Add split/data_type if missing
        if "split" not in item:
            item["split"] = "unknown"
        if "data_type" not in item:
            item["data_type"] = "unknown"
    
    return items, images_root


def load_v2_items(dataset_path: str, split: str) -> Tuple[List[dict], str]:
    """Load V2 format (separate JSON files per environment)."""
    split_map = {
        "desktop": "screenspot_desktop_v2.json",
        "mobile": "screenspot_mobile_v2.json",
        "web": "screenspot_web_v2.json",
        "all": None,  # Load all three
    }
    
    if split not in split_map:
        raise ValueError(f"V2 split must be one of: {list(split_map.keys())}")
    
    items = []
    if split == "all":
        for env, filename in [("desktop", "screenspot_desktop_v2.json"),
                               ("mobile", "screenspot_mobile_v2.json"),
                               ("web", "screenspot_web_v2.json")]:
            path = os.path.join(dataset_path, filename)
            if os.path.exists(path):
                with open(path) as f:
                    env_items = json.load(f)
                for item in env_items:
                    item["split"] = env
                items.extend(env_items)
    else:
        json_path = os.path.join(dataset_path, split_map[split])
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"V2 file not found: {json_path}")
        with open(json_path) as f:
            items = json.load(f)
        for item in items:
            item["split"] = split
    
    images_root = os.path.join(dataset_path, "screenspotv2_image")
    
    # Normalize format: bbox is already [x, y, w, h], instruction exists
    # data_type exists, just ensure split is set
    return items, images_root


def load_pro_items(dataset_path: str) -> Tuple[List[dict], str]:
    """Load Pro format (FiftyOne MongoDB-style JSON)."""
    samples_path = os.path.join(dataset_path, "samples.json")
    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"Pro samples.json not found: {samples_path}")
    
    with open(samples_path) as f:
        # samples.json is a JSON array of documents
        content = f.read().strip()
        # Handle both array format and newline-delimited JSON
        if content.startswith('['):
            samples = json.loads(content)
        else:
            # Newline-delimited JSON
            samples = [json.loads(line) for line in content.split('\n') if line.strip()]
    
    items = []
    for sample in samples:
        # Extract fields from FiftyOne format
        bbox_obj = sample.get("action_detection", {})
        if not bbox_obj or "bounding_box" not in bbox_obj:
            continue
        
        # FiftyOne bbox is normalized [x_center, y_center, width, height]
        bbox_norm = bbox_obj["bounding_box"]
        # We'll store as normalized and convert during evaluation
        
        item = {
            "img_filename": sample.get("filepath", "").replace("data/", ""),
            "instruction": sample.get("instruction", ""),
            "data_type": bbox_obj.get("label", "unknown"),
            "split": sample.get("platform", {}).get("label", "unknown"),
            "bbox_normalized": bbox_norm,  # [x_center, y_center, w, h] normalized
            "img_width": sample.get("metadata", {}).get("width"),
            "img_height": sample.get("metadata", {}).get("height"),
        }
        items.append(item)
    
    images_root = os.path.join(dataset_path, "data")
    return items, images_root


def evaluate_items(
    processor,
    model,
    items: List[dict],
    images_root: str,
    device: str,
    min_pixels: int,
    max_pixels: int,
    limit: Optional[int] = None,
    envs: Optional[Set[str]] = None,
    types: Optional[Set[str]] = None,
    dataset_format: str = "v1",
) -> Dict[str, Dict[str, List[dict]]]:
    """Evaluate items regardless of format."""
    # Filter by environment and type
    if envs:
        envs = {e.lower() for e in envs}
        items = [it for it in items if str(it.get("split", "")).lower() in envs]
    if types:
        types = {t.lower() for t in types}
        items = [it for it in items if str(it.get("data_type", "")).lower() in types]
    
    N = min(limit, len(items)) if limit and limit > 0 else len(items)
    print(f"Evaluating {N} samples from {dataset_format.upper()} dataset")
    
    model.eval()
    results: Dict[str, Dict[str, List[dict]]] = {}
    
    for i in tqdm(range(N), desc="Evaluating"):
        item = items[i]
        img_path = os.path.join(images_root, item["img_filename"])
        if not os.path.exists(img_path):
            continue
        
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        
        # Override with stored dimensions if available (Pro format)
        if "img_width" in item and item["img_width"]:
            img_w = item["img_width"]
        if "img_height" in item and item["img_height"]:
            img_h = item["img_height"]
        
        split_name = item.get("split", "unknown")
        data_type = item.get("data_type", "unknown")
        
        if split_name not in results:
            results[split_name] = {}
        if data_type not in results[split_name]:
            results[split_name][data_type] = []
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Based on the screenshot of the page, I give a text description and you give its corresponding "
                            "location. The coordinate represents a clickable location [x, y] for an element, which is a "
                            "relative coordinate on the screenshot, scaled from 0 to 1."
                        ),
                    },
                    {"type": "image", "image": img, "min_pixels": min_pixels, "max_pixels": max_pixels},
                    {"type": "text", "text": item["instruction"]},
                ],
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt").to(device)
        
        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )
            gen = out[:, inputs.input_ids.shape[1] :]
            pred_str = processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            pred_xy = parse_coord(pred_str)
            
            # Convert ground truth bbox to normalized [x1, y1, x2, y2]
            if "bbox_normalized" in item:
                # Pro format: [x_center, y_center, w, h] normalized
                x_c, y_c, w, h = item["bbox_normalized"]
                gt_bbox = [x_c - w/2, y_c - h/2, x_c + w/2, y_c + h/2]
            else:
                # V1/V2 format: [x, y, w, h] in pixels
                x, y, w, h = item["bbox"]
                gt_bbox = [x / img_w, y / img_h, (x + w) / img_w, (y + h) / img_h]
            
            valid = not any(v != v for v in pred_xy)  # check for NaNs
            acc = 1 if (valid and (gt_bbox[0] <= pred_xy[0] <= gt_bbox[2]) and (gt_bbox[1] <= pred_xy[1] <= gt_bbox[3])) else 0
            
            results[split_name][data_type].append({
                "anno_id": item.get("id", item.get("ui_id", i)),
                "instruction": item["instruction"],
                "pred": list(pred_xy) if valid else None,
                "gt_bbox": gt_bbox,
                "acc": acc,
            })
        except Exception as e:
            results[split_name][data_type].append({
                "anno_id": item.get("id", item.get("ui_id", i)),
                "instruction": item["instruction"],
                "pred": None,
                "gt_bbox": None,
                "acc": 0,
                "error": str(e),
            })
    
    return results


def compute_metrics(results: Dict[str, Dict[str, List[dict]]]) -> Dict[str, Any]:
    """Compute per-split, per-type success rates and overall aggregate."""
    metrics: Dict[str, Any] = {}
    overall_total = 0
    overall_success = 0
    
    for split_name, split_data in results.items():
        metrics[split_name] = {}
        for data_type, samples in split_data.items():
            total = len(samples)
            success = sum(s.get("acc", 0) for s in samples)
            sr = success / total if total > 0 else 0.0
            metrics[split_name][data_type] = {"success_rate": sr, "total": total}
            overall_total += total
            overall_success += success
    
    metrics["overall"] = {
        "success_rate": (overall_success / overall_total) if overall_total > 0 else 0.0,
        "total": overall_total,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Unified ScreenSpot evaluation (V1/V2/Pro)")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset (V1/V2/Pro)")
    parser.add_argument("--model_id", type=str, default="showlab/ShowUI-2B", help="Model ID or checkpoint path")
    parser.add_argument("--split", type=str, default=None, help="Split to evaluate (V1: hf_test_full; V2: desktop/mobile/web/all; Pro: ignored)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--min_visual_tokens", type=int, default=256, help="Min visual tokens")
    parser.add_argument("--max_visual_tokens", type=int, default=1344, help="Max visual tokens")
    parser.add_argument("--output", type=str, default="eval_unified_results.json", help="Output JSON file")
    parser.add_argument("--envs", type=str, nargs="*", default=None, help="Environment filter")
    parser.add_argument("--types", type=str, nargs="*", default=None, help="Data type filter")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    min_pixels = args.min_visual_tokens * 28 * 28
    max_pixels = args.max_visual_tokens * 28 * 28
    
    # Detect dataset format
    dataset_format = detect_dataset_format(args.dataset_path)
    print(f"Detected dataset format: {dataset_format.upper()}")
    
    # Load items based on format
    if dataset_format == "v1":
        if not args.split:
            args.split = "hf_test_full"
        items, images_root = load_v1_items(args.dataset_path, args.split)
    elif dataset_format == "v2":
        if not args.split:
            args.split = "all"
        items, images_root = load_v2_items(args.dataset_path, args.split)
    elif dataset_format == "pro":
        items, images_root = load_pro_items(args.dataset_path)
    else:
        raise ValueError(f"Unknown dataset format: {dataset_format}")
    
    print(f"Loaded {len(items)} items from {args.dataset_path}")
    
    # Load model
    print(f"Loading model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id, min_pixels=min_pixels, max_pixels=max_pixels)
    model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=dtype, device_map="auto")
    
    # Evaluate
    results = evaluate_items(
        processor=processor,
        model=model,
        items=items,
        images_root=images_root,
        device=device,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        limit=args.limit,
        envs=set(args.envs) if args.envs else None,
        types=set(args.types) if args.types else None,
        dataset_format=dataset_format,
    )
    
    metrics = compute_metrics(results)
    
    print("\n=== Evaluation Results ===")
    for split_name, split_metrics in metrics.items():
        if split_name == "overall":
            print(f"\noverall: {split_metrics['success_rate']:.4f} ({split_metrics['total']} samples)")
            continue
        print(f"\n{split_name}:")
        for data_type, m in split_metrics.items():
            print(f"  {data_type}: {m['success_rate']:.4f} ({m['total']} samples)")
    
    # Save results
    output_data = {
        "dataset_path": os.path.abspath(args.dataset_path),
        "dataset_format": dataset_format,
        "model_id": args.model_id,
        "split": args.split,
        "envs": args.envs,
        "types": args.types,
        "metrics": metrics,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved detailed results to {args.output}")


if __name__ == "__main__":
    main()




