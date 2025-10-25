import os
import ast
import json
import argparse
from typing import List, Tuple, Dict, Any

import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


def parse_pred_coord(text: str, img_w: int, img_h: int) -> Tuple[float, float]:
    """Parse model output into normalized [x,y] in [0,1].

    Strategy:
    1) Try strict Python literal list/tuple (e.g., [0.1, 0.2])
    2) Fallback regex: extract first two numbers; if > 1, treat as pixels and normalize
    3) Clamp to [0,1]
    """
    # 1) strict literal
    try:
        val = ast.literal_eval(text)
        if (
            isinstance(val, (list, tuple))
            and len(val) == 2
            and all(isinstance(v, (int, float)) for v in val)
        ):
            x, y = float(val[0]), float(val[1])
            if x > 1.5 or y > 1.5:
                # likely pixels
                if img_w > 0 and img_h > 0:
                    x /= float(img_w)
                    y /= float(img_h)
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            return x, y
    except Exception:
        pass

    # 2) regex fallback
    import re
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if len(nums) >= 2:
        try:
            x = float(nums[0])
            y = float(nums[1])
            if x > 1.5 or y > 1.5:
                if img_w > 0 and img_h > 0:
                    x /= float(img_w)
                    y /= float(img_h)
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            return x, y
        except Exception:
            pass
    return float("nan"), float("nan")


def load_screenspot_v2_desktop(v2_dir: str) -> List[Dict[str, Any]]:
    json_path = os.path.join(v2_dir, "screenspot_desktop_v2.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"ScreenSpot-v2 file not found: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    # Expected keys: img_filename, instruction, bbox (x,y,w,h), data_type
    # Images located under: v2_dir/screenspotv2_image/<img_filename>
    image_root = os.path.join(v2_dir, "screenspotv2_image")

    items = []
    for it in data:
        img_file = it.get("img_filename")
        if not img_file:
            continue
        img_path = os.path.join(image_root, img_file)
        bbox = it.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        items.append(
            dict(
                image_path=img_path,
                instruction=it.get("instruction", ""),
                bbox=bbox,
                meta=dict(data_type=it.get("data_type", ""), source="v2"),
            )
        )
    return items


def load_screenspot_pro_desktop(pro_dir: str) -> List[Dict[str, Any]]:
    # ScreenSpot-Pro uses FiftyOne export with samples in samples.json
    samples_path = os.path.join(pro_dir, "samples.json")
    if not os.path.exists(samples_path):
        raise FileNotFoundError(
            f"ScreenSpot-Pro samples not found: {samples_path}. "
            "Ensure you extracted the dataset correctly."
        )

    # samples.json appears to be a single JSON object containing a list under a key or a large array
    # Try both possibilities robustly
    with open(samples_path, "r") as f:
        text = f.read().strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            # Attempt common keys
            if "samples" in data and isinstance(data["samples"], list):
                records = data["samples"]
            else:
                # Fallback: if dict but no "samples" key, try to find a list value
                records = None
                for v in data.values():
                    if isinstance(v, list):
                        records = v
                        break
                if records is None:
                    raise ValueError("Unable to locate sample list in ScreenSpot-Pro JSON")
        elif isinstance(data, list):
            records = data
        else:
            raise ValueError("Unexpected ScreenSpot-Pro JSON structure")
    except Exception as e:
        raise RuntimeError(f"Failed to parse ScreenSpot-Pro samples.json: {e}")

    items = []
    for rec in records:
        # Required fields
        rel_path = rec.get("filepath")  # e.g. data/screenshot_*.png
        instruction = rec.get("instruction", "")
        # Normalized bounding box (cx, cy, w, h) or (x, y, w, h)? From head: looks like [x, y, w, h] normalized 0-1
        det = rec.get("action_detection") or {}
        bbox = det.get("bounding_box")
        if not (rel_path and isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue

        image_path = os.path.join(pro_dir, rel_path)

        # Convert normalized [x, y, w, h] to absolute bbox needs image size; but for evaluation we only need GT region as normalized [x1,y1,x2,y2]
        # We will convert at evaluation time using actual image size
        items.append(
            dict(
                image_path=image_path,
                instruction=instruction,
                bbox_norm=bbox,  # normalized [x, y, w, h]
                meta=dict(
                    data_type=det.get("label", ""),
                    platform=(rec.get("platform") or {}).get("label", ""),
                    source="pro",
                ),
            )
        )
    return items


def within_box_norm(pred_xy: Tuple[float, float], gt_xyxy: Tuple[float, float, float, float]) -> bool:
    return (
        (not any(map(lambda v: v != v, pred_xy)))
        and gt_xyxy[0] <= pred_xy[0] <= gt_xyxy[2]
        and gt_xyxy[1] <= pred_xy[1] <= gt_xyxy[3]
    )


def evaluate_items(
    items: List[Dict[str, Any]],
    processor: AutoProcessor,
    model: Qwen2VLForConditionalGeneration,
    max_new_tokens: int,
    batch_size: int,
    min_pixels: int,
    max_pixels: int,
    *,
    pro_center_bbox: bool = False,
    viz_dir: str | None = None,
    viz_samples: int = 0,
) -> float:
    ok = 0
    N = len(items)
    parsed = 0
    saved = 0
    if viz_dir:
        os.makedirs(viz_dir, exist_ok=True)
    for start in tqdm(range(0, N, batch_size), desc="Eval"):
        end = min(start + batch_size, N)
        batch = items[start:end]

        images = []
        messages = []
        img_sizes = []
        raw_imgs = []
        for it in batch:
            img = Image.open(it["image_path"]).convert("RGB")
            w, h = img.size
            img_sizes.append((w, h))
            images.append(img)
            raw_imgs.append((img.copy(), it["image_path"]))
            messages.append([
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1.",
                        },
                        {"type": "image", "image": img, "min_pixels": min_pixels, "max_pixels": max_pixels},
                        {"type": "text", "text": it.get("instruction", "")},
                    ],
                }
            ])

        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
        inputs = processor(text=texts, images=images, padding=True, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        gen = out[:, inputs.input_ids.shape[1]:]
        pred_strs = processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        for pred_str, it, (w, h), (raw_img, img_path) in zip(pred_strs, batch, img_sizes, raw_imgs):
            px, py = parse_pred_coord(pred_str, w, h)
            if not (px != px or py != py):
                parsed += 1

            if "bbox" in it:
                x, y, bw, bh = it["bbox"]
                gt = (x / w, y / h, (x + bw) / w, (y + bh) / h)
            else:
                nx, ny, nw, nh = it["bbox_norm"]  # normalized in [0,1]
                if pro_center_bbox:
                    # interpret as [cx, cy, w, h]
                    x1 = nx - nw / 2.0
                    y1 = ny - nh / 2.0
                    x2 = nx + nw / 2.0
                    y2 = ny + nh / 2.0
                    gt = (x1, y1, x2, y2)
                else:
                    # interpret as [x, y, w, h] top-left
                    gt = (nx, ny, nx + nw, ny + nh)

            ok += 1 if within_box_norm((px, py), gt) else 0

            # optional visualization for quick auditing
            if viz_dir and saved < viz_samples:
                draw = ImageDraw.Draw(raw_img)
                gx1 = int(max(0.0, min(1.0, gt[0])) * w)
                gy1 = int(max(0.0, min(1.0, gt[1])) * h)
                gx2 = int(max(0.0, min(1.0, gt[2])) * w)
                gy2 = int(max(0.0, min(1.0, gt[3])) * h)
                px_i = int(max(0.0, min(1.0, px)) * w) if not (px != px) else -1
                py_i = int(max(0.0, min(1.0, py)) * h) if not (py != py) else -1
                draw.rectangle([gx1, gy1, gx2, gy2], outline="red", width=3)
                if px_i >= 0 and py_i >= 0:
                    r = 6
                    draw.ellipse([px_i - r, py_i - r, px_i + r, py_i + r], fill="cyan", outline="cyan")
                base = os.path.basename(img_path)
                raw_img.save(os.path.join(viz_dir, f"viz_{saved:04d}_{base}"))
                saved += 1

    print(f"Parsed predictions: {parsed}/{N} ({(parsed/max(1,N)):.2%})")
    return ok / max(1, N)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on ScreenSpot V2 and Pro (desktop only)")
    parser.add_argument("--data_root", required=True, help="Path to ~/showui_data containing ScreenSpot-v2 and/or ScreenSpot-Pro")
    parser.add_argument("--model_id", default="showlab/ShowUI-2B", help="HF model id or local path")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--limit_v2", type=int, default=0, help="Optional cap on number of V2 samples (0=all)")
    parser.add_argument("--limit_pro", type=int, default=0, help="Optional cap on number of Pro samples (0=all)")
    parser.add_argument("--pro_center_bbox", action="store_true", help="Interpret Pro boxes as [cx,cy,w,h] instead of [x,y,w,h]")
    parser.add_argument("--viz_dir", default=None, help="Optional directory to save N visualized samples")
    parser.add_argument("--viz_samples", type=int, default=0, help="Number of samples to visualize")
    parser.add_argument("--eval_v2", action="store_true", help="Only evaluate ScreenSpot-V2")
    parser.add_argument("--eval_pro", action="store_true", help="Only evaluate ScreenSpot-Pro")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    min_pixels = 256 * 28 * 28
    max_pixels = 1344 * 28 * 28

    processor = AutoProcessor.from_pretrained(
        args.model_id if os.path.isdir(args.model_id) else "showlab/ShowUI-2B",
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=dtype,
    )
    model.eval()

    v2_dir = os.path.join(args.data_root, "ScreenSpot-v2")
    pro_dir = os.path.join(args.data_root, "ScreenSpot-Pro")

    has_v2 = os.path.isdir(v2_dir)
    has_pro = os.path.isdir(pro_dir)

    # Selection logic: if neither flag is set, run both available
    run_v2 = args.eval_v2 or (not args.eval_v2 and not args.eval_pro)
    run_pro = args.eval_pro or (not args.eval_v2 and not args.eval_pro)

    if not (has_v2 or has_pro):
        raise FileNotFoundError("Neither ScreenSpot-v2 nor ScreenSpot-Pro directories were found under data_root")

    if has_v2 and run_v2:
        v2_items = load_screenspot_v2_desktop(v2_dir)
        if args.limit_v2 and args.limit_v2 > 0:
            v2_items = v2_items[: args.limit_v2]
        v2_score = evaluate_items(
            v2_items, processor, model, args.max_new_tokens, args.batch_size, min_pixels, max_pixels,
            pro_center_bbox=False, viz_dir=args.viz_dir, viz_samples=args.viz_samples,
        )
        print(f"ScreenSpot-V2 Desktop Success Rate ({len(v2_items)}): {v2_score:.3f}")

    if has_pro and run_pro:
        pro_items = load_screenspot_pro_desktop(pro_dir)
        # Filter to desktop OS platforms; ScreenSpot-Pro has platform labels like 'macos', 'windows'
        pro_items = [it for it in pro_items if (it.get("meta", {}).get("platform", "").lower() in ("macos", "windows"))]
        if args.limit_pro and args.limit_pro > 0:
            pro_items = pro_items[: args.limit_pro]
        pro_score = evaluate_items(
            pro_items, processor, model, args.max_new_tokens, args.batch_size, min_pixels, max_pixels,
            pro_center_bbox=args.pro_center_bbox, viz_dir=args.viz_dir, viz_samples=args.viz_samples,
        )
        print(f"ScreenSpot-Pro Desktop Success Rate ({len(pro_items)}): {pro_score:.3f}")


if __name__ == "__main__":
    main()


