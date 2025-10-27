#!/usr/bin/env python3
"""
Unified checkpoint evaluation script for ScreenSpot V1, V2, and Pro datasets.
Evaluates multiple checkpoints and tracks the best performing one.

Usage:
  # V1 - all environments
  python eval_checkpoints_unified.py \
    --dataset_path ~/showui_data/ScreenSpot \
    --split hf_test_full \
    --models_root . --model_glob 'rl_ckpt_*'

  # V2 - desktop only
  python eval_checkpoints_unified.py \
    --dataset_path ~/showui_data/ScreenSpot-v2 \
    --split desktop \
    --models_root . --model_glob 'rl_ckpt_*'

  # V2 - desktop + web
  python eval_checkpoints_unified.py \
    --dataset_path ~/showui_data/ScreenSpot-v2 \
    --split all \
    --envs desktop web \
    --models_root . --model_glob 'rl_ckpt_*'

  # Pro - all
  python eval_checkpoints_unified.py \
    --dataset_path ~/showui_data/ScreenSpot-Pro \
    --models showlab/ShowUI-2B ./rl_ckpt_best
"""

import os
import ast
import re
import json
import glob as glob_module
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
    
    model.eval()
    results: Dict[str, Dict[str, List[dict]]] = {}
    
    for i in tqdm(range(N), desc="Evaluating", leave=False):
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


def discover_model_dirs(models_root: str, model_glob: str) -> List[str]:
    """Return a sorted list of checkpoint directories matching the glob pattern.
    Only include directories containing a config.json (saved by training).
    """
    candidates: List[str] = []
    for path in glob_module.glob(os.path.join(models_root, model_glob)):
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
            candidates.append(os.path.abspath(path))
    
    # sort for stable ordering: best first, then epoch numbers
    def sort_key(p: str) -> Tuple[int, int]:
        base = os.path.basename(p)
        if base == "rl_ckpt_best":
            return (0, 0)
        m = re.match(r"rl_ckpt_epoch(\d+)$", base)
        if m:
            return (1, int(m.group(1)))
        return (2, 0)
    
    candidates.sort(key=sort_key)
    return candidates


def evaluate_checkpoint(
    checkpoint_path: str,
    dataset_path: str,
    dataset_format: str,
    split: Optional[str],
    items: List[dict],
    images_root: str,
    device: str,
    dtype: torch.dtype,
    min_pixels: int,
    max_pixels: int,
    limit: Optional[int] = None,
    envs: Optional[Set[str]] = None,
    types: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Load processor+model from checkpoint and evaluate."""
    print(f"\n{'='*80}")
    print(f"Evaluating: {os.path.basename(checkpoint_path)}")
    print(f"{'='*80}")
    
    try:
        processor = AutoProcessor.from_pretrained(
            checkpoint_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            checkpoint_path,
            torch_dtype=dtype,
            device_map="auto",
        )
        
        results = evaluate_items(
            processor=processor,
            model=model,
            items=items,
            images_root=images_root,
            device=device,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            limit=limit,
            envs=envs,
            types=types,
            dataset_format=dataset_format,
        )
        
        metrics = compute_metrics(results)
        
        # Print summary
        overall_sr = metrics["overall"]["success_rate"]
        print(f"\n✓ Overall Success Rate: {overall_sr:.4f} ({metrics['overall']['total']} samples)")
        
        return {
            "checkpoint_path": checkpoint_path,
            "split": split,
            "metrics": metrics,
            "results": results,
            "success": True,
        }
    
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        return {
            "checkpoint_path": checkpoint_path,
            "split": split,
            "error": str(e),
            "success": False,
        }


def main():
    parser = argparse.ArgumentParser(description="Unified checkpoint evaluation (V1/V2/Pro)")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset (V1/V2/Pro)")
    parser.add_argument("--models_root", type=str, default=".", help="Directory containing checkpoint folders")
    parser.add_argument("--model_glob", type=str, default="rl_ckpt_*", help="Glob to select checkpoint folders")
    parser.add_argument("--models", type=str, nargs="*", default=None, help="Explicit list of model IDs or checkpoint paths")
    parser.add_argument("--split", type=str, default=None, help="Split to evaluate (V1: hf_test_full; V2: desktop/mobile/web/all; Pro: ignored)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--min_visual_tokens", type=int, default=256, help="Min visual tokens")
    parser.add_argument("--max_visual_tokens", type=int, default=1344, help="Max visual tokens")
    parser.add_argument("--output", type=str, default="eval_checkpoints_unified_results.json", help="Output JSON file")
    parser.add_argument("--envs", type=str, nargs="*", default=None, help="Environment filter (e.g., desktop web)")
    parser.add_argument("--types", type=str, nargs="*", default=None, help="Data type filter (e.g., icon text)")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    min_pixels = args.min_visual_tokens * 28 * 28
    max_pixels = args.max_visual_tokens * 28 * 28
    
    # Detect dataset format
    dataset_format = detect_dataset_format(args.dataset_path)
    print(f"Detected dataset format: {dataset_format.upper()}")
    
    # Load items based on format (once, reused for all checkpoints)
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
    if args.envs:
        print(f"Environment filter: {args.envs}")
    if args.types:
        print(f"Type filter: {args.types}")
    
    # Discover checkpoints
    if args.models and len(args.models) > 0:
        checkpoint_paths = []
        for m in args.models:
            if os.path.isdir(m):
                checkpoint_paths.append(os.path.abspath(m))
            else:
                # Assume it's a HuggingFace model ID
                checkpoint_paths.append(m)
    else:
        checkpoint_paths = discover_model_dirs(args.models_root, args.model_glob)
    
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No checkpoints found. Tried models={args.models} or {os.path.join(args.models_root, args.model_glob)}"
        )
    
    print(f"\nFound {len(checkpoint_paths)} checkpoint(s):")
    for ck in checkpoint_paths:
        print(f"  - {ck}")
    
    # Evaluate all checkpoints
    all_results: Dict[str, Any] = {
        "dataset_path": os.path.abspath(args.dataset_path),
        "dataset_format": dataset_format,
        "split": args.split,
        "envs": args.envs,
        "types": args.types,
        "min_visual_tokens": args.min_visual_tokens,
        "max_visual_tokens": args.max_visual_tokens,
        "device": device,
        "checkpoints": {},
    }
    
    best_name = None
    best_sr = -1.0
    
    for ck_path in checkpoint_paths:
        result = evaluate_checkpoint(
            checkpoint_path=ck_path,
            dataset_path=args.dataset_path,
            dataset_format=dataset_format,
            split=args.split,
            items=items,
            images_root=images_root,
            device=device,
            dtype=dtype,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            limit=args.limit,
            envs=set(args.envs) if args.envs else None,
            types=set(args.types) if args.types else None,
        )
        
        all_results["checkpoints"][ck_path] = result
        
        if result.get("success") and "metrics" in result:
            overall_sr = result["metrics"]["overall"]["success_rate"]
            if overall_sr > best_sr:
                best_sr = overall_sr
                best_name = ck_path
    
    all_results["best_checkpoint"] = {
        "path": best_name,
        "overall_success_rate": best_sr,
    }
    
    # Save results
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Best checkpoint: {best_name}")
    print(f"Best success rate: {best_sr:.4f}")
    print(f"\nSaved detailed results to {args.output}")


if __name__ == "__main__":
    main()




