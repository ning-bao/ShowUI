#!/usr/bin/env python3
"""
Evaluate multiple checkpoints (e.g., rl_ckpt_best, rl_ckpt_epoch*) on a large
evaluation split (default: ScreenSpot hf_test_full). Saves detailed results and
aggregated metrics per checkpoint into a single JSON.

Example:
  python eval_checkpoints.py \
    --dataset_dir /path/to/datasets \
    --models_root . \
    --model_glob 'rl_ckpt_*' \
    --split hf_test_full \
    --output eval_checkpoints_results.json
"""

import os
import re
import ast
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional, Set

import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


def parse_coord(output_text: str) -> Tuple[float, float]:
    """Parse model output to extract [x, y] coordinates.
    Returns (nan, nan) on failure.
    """
    try:
        xy = ast.literal_eval(output_text)
        if isinstance(xy, (list, tuple)) and len(xy) == 2:
            return float(xy[0]), float(xy[1])
    except Exception:
        pass
    # Fallback regex to catch loose formats like "(0.12, 0.34)" or "[0.12,0.34]"
    try:
        m = re.search(r"[\[\(]?\s*([-+]?[0-9]*\.?[0-9]+)\s*,\s*([-+]?[0-9]*\.?[0-9]+)\s*[\]\)]?", output_text)
        if m:
            return float(m.group(1)), float(m.group(2))
    except Exception:
        pass
    return float("nan"), float("nan")


def resolve_screenspot_paths(dataset_dir: str, split: str, dataset_variant: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """Resolve metadata path and images root, handling V1/V2/Pro layouts.
    Returns (metadata_path, images_root). images_root may be None if not found.
    """
    # If explicit variant provided, try that first
    if dataset_variant:
        explicit_meta = os.path.join(dataset_dir, dataset_variant, "metadata", f"{split}.json")
        if os.path.exists(explicit_meta):
            variant_root = os.path.dirname(os.path.dirname(explicit_meta))
            images_root = os.path.join(variant_root, "images")
            if not os.path.isdir(images_root):
                for img_dir in ["image", "imgs", "Images", "IMAGES"]:
                    alt = os.path.join(variant_root, img_dir)
                    if os.path.isdir(alt):
                        images_root = alt
                        break
                else:
                    images_root = None
            return explicit_meta, images_root

    # Fast-path common layout
    meta_path = os.path.join(dataset_dir, "ScreenSpot", "metadata", f"{split}.json")
    if os.path.exists(meta_path):
        variant_root = os.path.dirname(os.path.dirname(meta_path))  # .../ScreenSpot
        images_root = os.path.join(variant_root, "images")
        return meta_path, images_root if os.path.isdir(images_root) else None

    # Try common alternative variant folder names
    candidate_roots = [
        "ScreenSpotV2",
        "ScreenSpot-v2",
        "ScreenSpotV1",
        "ScreenSpot-v1",
        "ScreenSpotPro",
        "ScreenSpot-Pro",
        "ScreenSpot_Pro",
        "ScreenSpot",
    ]
    for name in candidate_roots:
        mp = os.path.join(dataset_dir, name, "metadata", f"{split}.json")
        if os.path.exists(mp):
            variant_root = os.path.dirname(os.path.dirname(mp))
            ir = os.path.join(variant_root, "images")
            if not os.path.isdir(ir):
                # try a few common image dir variants
                for img_dir in ["image", "imgs", "Images", "IMAGES"]:
                    alt = os.path.join(variant_root, img_dir)
                    if os.path.isdir(alt):
                        ir = alt
                        break
                else:
                    ir = None
            return mp, ir

    # Fallback: walk to find any metadata/<split>.json and infer images sibling
    for dirpath, dirnames, filenames in os.walk(dataset_dir):
        if os.path.basename(dirpath) == "metadata" and f"{split}.json" in filenames:
            mp = os.path.join(dirpath, f"{split}.json")
            variant_root = os.path.dirname(dirpath)
            ir = os.path.join(variant_root, "images")
            if not os.path.isdir(ir):
                for img_dir in ["image", "imgs", "Images", "IMAGES"]:
                    alt = os.path.join(variant_root, img_dir)
                    if os.path.isdir(alt):
                        ir = alt
                        break
                else:
                    ir = None
            return mp, ir

    raise FileNotFoundError(f"Could not locate metadata for split '{split}' under {dataset_dir}")


def load_screenspot_items(
    dataset_dir: str,
    split: str,
    envs: Optional[Set[str]] = None,
    types: Optional[Set[str]] = None,
    dataset_variant: Optional[str] = None,
) -> Tuple[List[dict], Optional[str]]:
    meta_path, images_root = resolve_screenspot_paths(dataset_dir, split, dataset_variant)
    with open(meta_path) as f:
        items = json.load(f)

    # Optional filtering by environment and type
    if envs:
        envs = {e.lower() for e in envs}
    if types:
        types = {t.lower() for t in types}
    if envs or types:
        def _keep(it: dict) -> bool:
            if envs is not None and str(it.get("split", "")).lower() not in envs:
                return False
            if types is not None and str(it.get("data_type", "")).lower() not in types:
                return False
            return True
        items = [it for it in items if _keep(it)]
    return items, images_root


def evaluate_screenspot_items(
    processor,
    model,
    items: List[dict],
    images_root: Optional[str],
    device: str,
    min_pixels: int,
    max_pixels: int,
    limit: int = None,
) -> Dict[str, Dict[str, List[dict]]]:
    """Run evaluation and return nested results[split_name][data_type] = list of sample dicts.
    Each sample dict contains anno_id, task, pred, gt_bbox, acc, and optional error.
    """
    N = min(limit, len(items)) if limit and limit > 0 else len(items)
    results: Dict[str, Dict[str, List[dict]]] = {}

    model.eval()

    for i in tqdm(range(N), desc="Evaluating ScreenSpot"):
        item = items[i]
        img_path = os.path.join(images_root, item["img_url"]) if images_root else item.get("img_path", "")
        if not os.path.exists(img_path):
            continue

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = (item["img_size"][0], item["img_size"][1]) if "img_size" in item else img.size
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
                    {"type": "text", "text": item["task"]},
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

            # Convert ScreenSpot bbox (x, y, w, h in pixels) to normalized [x1, y1, x2, y2]
            x, y, w, h = item["bbox"]
            gt_bbox = [x / img_w, y / img_h, (x + w) / img_w, (y + h) / img_h]

            valid = not any(v != v for v in pred_xy)  # check for NaNs
            acc = 1 if (valid and (gt_bbox[0] <= pred_xy[0] <= gt_bbox[2]) and (gt_bbox[1] <= pred_xy[1] <= gt_bbox[3])) else 0

            results[split_name][data_type].append(
                {
                    "anno_id": item.get("id", i),
                    "task": item["task"],
                    "pred": list(pred_xy) if valid else None,
                    "gt_bbox": gt_bbox,
                    "acc": acc,
                }
            )
        except Exception as e:
            results[split_name][data_type].append(
                {
                    "anno_id": item.get("id", i),
                    "task": item["task"],
                    "pred": None,
                    "gt_bbox": None,
                    "acc": 0,
                    "error": str(e),
                }
            )

    return results


def compute_metrics(results: Dict[str, Dict[str, List[dict]]]) -> Dict[str, Any]:
    """Compute per-split and per-type success rates plus an overall summary."""
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
    try:
        import glob
    except Exception:
        glob = None

    candidates: List[str] = []
    if glob is not None:
        for path in glob.glob(os.path.join(models_root, model_glob)):
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
    checkpoint_dir: str,
    dataset_dir: str,
    split: str,
    device: str,
    dtype: torch.dtype,
    min_pixels: int,
    max_pixels: int,
    limit: int = None,
    envs: Optional[Set[str]] = None,
    types: Optional[Set[str]] = None,
    dataset_variant: Optional[str] = None,
) -> Dict[str, Any]:
    """Load processor+model from checkpoint_dir and evaluate on ScreenSpot split."""
    processor = AutoProcessor.from_pretrained(
        checkpoint_dir,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        checkpoint_dir,
        torch_dtype=dtype,
        device_map="auto",
    )

    items, images_root = load_screenspot_items(dataset_dir, split, envs=envs, types=types, dataset_variant=dataset_variant)
    results = evaluate_screenspot_items(
        processor=processor,
        model=model,
        items=items,
        images_root=images_root,
        device=device,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        limit=limit,
    )
    metrics = compute_metrics(results)
    return {"checkpoint_dir": checkpoint_dir, "split": split, "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple RL checkpoints on ScreenSpot")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset root (contains ScreenSpot/)")
    parser.add_argument("--models_root", type=str, default=".", help="Directory containing checkpoint folders")
    parser.add_argument("--model_glob", type=str, default="rl_ckpt_*", help="Glob to select checkpoint folders")
    parser.add_argument("--models", type=str, nargs="*", default=None, help="Explicit list of checkpoint dirs")
    parser.add_argument("--split", type=str, default="hf_test_full", help="ScreenSpot split to evaluate")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of samples to evaluate")
    parser.add_argument("--min_visual_tokens", type=int, default=256, help="Min visual tokens")
    parser.add_argument("--max_visual_tokens", type=int, default=1344, help="Max visual tokens")
    parser.add_argument("--output", type=str, default="eval_checkpoints_results.json", help="Output JSON path")
    parser.add_argument("--envs", type=str, nargs="*", default=None, help="Environment filter: e.g., desktop mobile web")
    parser.add_argument("--types", type=str, nargs="*", default=None, help="Data type filter: e.g., icon text")
    parser.add_argument("--dataset_variant", type=str, default=None, help="Explicit variant folder under dataset_dir, e.g., ScreenSpotV2, ScreenSpotPro")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    min_pixels = args.min_visual_tokens * 28 * 28
    max_pixels = args.max_visual_tokens * 28 * 28

    if args.models and len(args.models) > 0:
        checkpoint_dirs = [os.path.abspath(p) for p in args.models]
    else:
        checkpoint_dirs = discover_model_dirs(args.models_root, args.model_glob)

    if not checkpoint_dirs:
        raise FileNotFoundError(
            f"No checkpoints found. Tried models={args.models} or {os.path.join(args.models_root, args.model_glob)}"
        )

    print("Found checkpoints:")
    for ck in checkpoint_dirs:
        print(f"  - {ck}")

    all_results: Dict[str, Any] = {
        "dataset_dir": os.path.abspath(args.dataset_dir),
        "split": args.split,
        "min_visual_tokens": args.min_visual_tokens,
        "max_visual_tokens": args.max_visual_tokens,
        "device": device,
        "checkpoints": {},
    }

    best_name = None
    best_sr = -1.0

    for ck_dir in checkpoint_dirs:
        print(f"\nEvaluating checkpoint: {ck_dir}")
        try:
            res = evaluate_checkpoint(
                checkpoint_dir=ck_dir,
                dataset_dir=args.dataset_dir,
                split=args.split,
                device=device,
                dtype=dtype,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                limit=args.limit,
                envs=set(args.envs) if args.envs else None,
                types=set(args.types) if args.types else None,
                dataset_variant=args.dataset_variant,
            )
            all_results["checkpoints"][ck_dir] = res
            overall_sr = res["metrics"].get("overall", {}).get("success_rate", 0.0)
            print(f"Overall success rate: {overall_sr:.4f}")
            if overall_sr > best_sr:
                best_sr = overall_sr
                best_name = ck_dir
        except Exception as e:
            all_results["checkpoints"][ck_dir] = {"error": str(e)}
            print(f"Error evaluating {ck_dir}: {e}")

    all_results["best_checkpoint"] = {"path": best_name, "overall_success_rate": best_sr}

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved aggregated results to {args.output}")


if __name__ == "__main__":
    main()



