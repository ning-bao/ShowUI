#!/usr/bin/env python3
"""
Evaluate base model (or any checkpoint) on ScreenSpot full test set.
Outputs detailed per-split success rates and saves results to JSON.
"""
import os
import ast
import re
import json
import argparse
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


def within_bbox(point, bbox):
    """Check if normalized point [x, y] is within normalized bbox [x1, y1, x2, y2]."""
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def parse_coord(output_text):
    """Parse model output to extract [x, y] coordinates with a regex fallback."""
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


def evaluate_screenspot(
    processor,
    model,
    dataset_dir,
    split="hf_test_full",
    min_pixels=256 * 28 * 28,
    max_pixels=1344 * 28 * 28,
    device="cuda",
    limit=None,
):
    """
    Evaluate on ScreenSpot dataset.
    Returns dict: {split_name: {data_type: [sample_results], ...}, ...}
    """
    meta_path = os.path.join(dataset_dir, "ScreenSpot", "metadata", f"{split}.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"ScreenSpot metadata not found: {meta_path}")

    with open(meta_path) as f:
        items = json.load(f)

    N = min(limit, len(items)) if limit and limit > 0 else len(items)
    print(f"Evaluating {N} samples from ScreenSpot/{split}")

    model.eval()
    results = {}

    for i in tqdm(range(N), desc=f"Evaluating {split}"):
        item = items[i]
        img_path = os.path.join(dataset_dir, "ScreenSpot", "images", item["img_url"])
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
                        "text": "Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1.",
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
            pred = parse_coord(pred_str)

            # Convert ScreenSpot bbox (x, y, w, h in pixels) to normalized [x1, y1, x2, y2]
            x, y, w, h = item["bbox"]
            gt = [x / img_w, y / img_h, (x + w) / img_w, (y + h) / img_h]

            acc = (
                1
                if (not any(float("nan") == v or v != v for v in pred))
                and (gt[0] <= pred[0] <= gt[2])
                and (gt[1] <= pred[1] <= gt[3])
                else 0
            )

            results[split_name][data_type].append(
                {
                    "anno_id": item.get("id", i),
                    "task": item["task"],
                    "pred": list(pred) if not any(float("nan") == v or v != v for v in pred) else None,
                    "gt_bbox": gt,
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


def compute_metrics(results):
    """Compute per-split and per-type success rates from results dict."""
    metrics = {}
    for split_name, split_data in results.items():
        metrics[split_name] = {}
        for data_type, samples in split_data.items():
            total = len(samples)
            success = sum(s["acc"] for s in samples)
            metrics[split_name][data_type] = {"success_rate": success / total if total > 0 else 0.0, "total": total}
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on ScreenSpot")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset root (contains ScreenSpot/)")
    parser.add_argument(
        "--model_id", type=str, default="showlab/ShowUI-2B", help="Model ID or local checkpoint path"
    )
    parser.add_argument("--split", type=str, default="hf_test_full", help="ScreenSpot split to evaluate")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples (optional)")
    parser.add_argument("--min_visual_tokens", type=int, default=256, help="Min visual tokens")
    parser.add_argument("--max_visual_tokens", type=int, default=1344, help="Max visual tokens")
    parser.add_argument("--output", type=str, default="eval_results.json", help="Output JSON file for results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    min_pixels = args.min_visual_tokens * 28 * 28
    max_pixels = args.max_visual_tokens * 28 * 28

    print(f"Loading model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id, min_pixels=min_pixels, max_pixels=max_pixels)
    model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=dtype, device_map="auto")

    results = evaluate_screenspot(
        processor,
        model,
        args.dataset_dir,
        split=args.split,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        device=device,
        limit=args.limit,
    )

    metrics = compute_metrics(results)

    print("\n=== Evaluation Results ===")
    for split_name, split_metrics in metrics.items():
        print(f"\n{split_name}:")
        for data_type, m in split_metrics.items():
            print(f"  {data_type}: {m['success_rate']:.4f} ({m['total']} samples)")

    # Save detailed results
    output_data = {"model_id": args.model_id, "split": args.split, "metrics": metrics, "results": results}
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved detailed results to {args.output}")


if __name__ == "__main__":
    main()

