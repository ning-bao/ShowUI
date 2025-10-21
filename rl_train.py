import os
import io
import hashlib
import ast
import math
import time
import json
import random
import re
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from torch.utils.tensorboard import SummaryWriter

from data.dset_shared_grounding import dataset_mapping
from data.template.shared_grounding import grounding_to_qwen
from data.data_utils import IGNORE_INDEX


@dataclass
class RLArgs:
    dataset_dir: str
    train_dataset: str = "showui-desktop"
    train_json: str = "hf_train"
    min_visual_tokens: int = 256
    max_visual_tokens: int = 1344
    model_id: str = "showlab/ShowUI-2B"
    lr: float = 5e-6
    batch_size: int = 1
    steps_per_epoch: int = 200
    epochs: int = 1
    tau_success: float = 0.06
    tau_success_end: float = 0.06
    alpha_dist: float = 1.0
    max_new_tokens: int = 64
    temperature: float = 0.7
    temperature_end: float = 0.7
    top_p: float = 0.0
    top_k: int = 0
    do_sample: bool = True
    num_beams: int = 1
    grad_accum_steps: int = 1
    entropy_coef_start: float = 0.01
    entropy_coef_end: float = 0.0
    kl_coef: float = 0.0
    ref_model_id: str = ""
    ref_model_8bit: bool = True
    seed: int = 42
    gradient_checkpointing: bool = False
    load_in_8bit: bool = False
    log_dir: str = "./runs/rl"
    eval_subset_limit: int = 200
    eval_every_steps: int = 200
    save_every_epochs: int = 1
    resume_from: str = ""
    save_optimizer: bool = True
    eval_split: str = "hf_test_full"
    log_samples_every: int = 100
    log_hist_every: int = 100
    save_best: bool = True
    warmup_steps: int = 200
    reward_ema_beta: float = 0.9
    stats_jsonl: str = ""


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Single-GPU only; no distributed initialization


def find_latest_epoch_checkpoint(base_dir: str) -> tuple:
    """Scan base_dir for rl_ckpt_epoch{N} directories and return (path, N) of the latest.
    Returns (None, -1) if none found."""
    if not os.path.isdir(base_dir):
        return None, -1
    latest_n = -1
    latest_path = None
    try:
        for name in os.listdir(base_dir):
            if not name.startswith("rl_ckpt_epoch"):
                continue
            try:
                n = int(name.replace("rl_ckpt_epoch", ""))
            except Exception:
                continue
            if n > latest_n:
                latest_n = n
                latest_path = os.path.join(base_dir, name)
    except Exception:
        pass
    return latest_path, latest_n


def load_split_items(dataset_dir: str, dataset: str, split: str) -> Tuple[str, List[dict]]:
    # Special handling for Novelis-style JSON: free-form JSON file with bbox targets
    if dataset.lower() == "novelis":
        # Resolve JSON path: allow absolute path, relative to CWD, or under dataset_dir
        json_path = split
        if not json_path.endswith(".json"):
            json_path = f"{json_path}.json"
        if not os.path.isabs(json_path):
            cand = os.path.join(dataset_dir, json_path)
            json_path = cand if os.path.exists(cand) else json_path

        with open(json_path) as f:
            raw_items = json.load(f)

        samples: List[dict] = []
        for it in raw_items:
            img_rel_or_abs = it.get("image_context", "")
            if not img_rel_or_abs:
                continue
            img_path = img_rel_or_abs if os.path.isabs(img_rel_or_abs) else os.path.join(dataset_dir, img_rel_or_abs)
            if not os.path.exists(img_path):
                # Try without dataset_dir if already combined incorrectly
                if os.path.isabs(img_rel_or_abs) and os.path.exists(img_rel_or_abs):
                    img_path = img_rel_or_abs
                else:
                    continue

            # Read size to normalize the bbox center
            try:
                with Image.open(img_path) as im:
                    width, height = im.size
            except Exception:
                continue

            bbox = it.get("target_bbox", None)
            if not bbox or len(bbox) != 4:
                continue
            x, y, w, h = bbox
            cx = (float(x) + float(w) / 2.0) / max(1.0, float(width))
            cy = (float(y) + float(h) / 2.0) / max(1.0, float(height))
            cx = min(1.0, max(0.0, cx))
            cy = min(1.0, max(0.0, cy))

            instruction = it.get("goal") or it.get("rubric") or ""
            if not instruction:
                # Fallback to app/id description
                instruction = f"Locate target for {it.get('app', 'unknown')} - {it.get('id', '')}"

            samples.append({
                # Use absolute path directly; we'll set img_dir to empty string
                "img_url": os.path.abspath(img_path),
                "element": [
                    {
                        "instruction": instruction,
                        "point": [cx, cy],
                    }
                ],
            })

        # Return empty img_dir so that downstream os.path.join("", abs_path) yields abs_path
        return "", samples

    # Salesforce parquet-based grounding datasets (load all .parquet files)
    if dataset.lower() in ("salesforce", "salesforce-parquet", "screenspot-parquet", "screenspot-parquet"):
        # Resolve 'split' to either a parquet file or a directory containing parquets
        base_path = split
        if not os.path.isabs(base_path):
            cand = os.path.join(dataset_dir, base_path)
            base_path = cand if os.path.exists(cand) else base_path

        parq_files: List[str] = []
        if os.path.isdir(base_path):
            for root, _, files in os.walk(base_path):
                for name in files:
                    if name.lower().endswith(".parquet") or name.lower().endswith(".parq"):
                        parq_files.append(os.path.join(root, name))
        elif os.path.isfile(base_path) and (base_path.lower().endswith(".parquet") or base_path.lower().endswith(".parq")):
            parq_files.append(base_path)
        else:
            # If split is not a path, fallback to searching under dataset_dir/split
            search_dir = os.path.join(dataset_dir, str(split))
            if os.path.isdir(search_dir):
                for root, _, files in os.walk(search_dir):
                    for name in files:
                        if name.lower().endswith(".parquet") or name.lower().endswith(".parq"):
                            parq_files.append(os.path.join(root, name))

        if not parq_files:
            return "", []

        # Lazy imports to avoid hard dependency if not used
        pd = None
        pq = None
        try:
            import pandas as pd  # type: ignore
        except Exception:
            pd = None
        if pd is None:
            try:
                import pyarrow.parquet as pq  # type: ignore
            except Exception:
                pq = None

        samples: List[dict] = []

        def resolve_image_path(path_value: str, parq_path: str) -> str:
            if not path_value:
                return ""
            if os.path.isabs(path_value) and os.path.exists(path_value):
                return path_value
            cand1 = os.path.join(dataset_dir, path_value)
            if os.path.exists(cand1):
                return cand1
            base_dir = os.path.dirname(parq_path)
            cand2 = os.path.join(base_dir, path_value)
            if os.path.exists(cand2):
                return cand2
            return ""

        def coerce_list(val):
            if isinstance(val, (list, tuple)):
                return list(val)
            # numpy array support
            try:
                import numpy as _np  # local alias to avoid shadowing
                if isinstance(val, _np.ndarray):
                    return val.tolist()
            except Exception:
                pass
            return None

        for pfile in parq_files:
            try:
                if pd is not None:
                    try:
                        df = pd.read_parquet(pfile, engine="pyarrow")
                    except Exception:
                        try:
                            df = pd.read_parquet(pfile, engine="fastparquet")
                        except Exception:
                            if pq is not None:
                                df = pq.read_table(pfile).to_pandas()
                            else:
                                continue
                elif pq is not None:
                    df = pq.read_table(pfile).to_pandas()
                else:
                    continue
            except Exception:
                continue

            # Normalize column names to simplify matching
            cols = {str(c): c for c in df.columns}
            def get_val(row, names):
                for n in names:
                    if n in row and row[n] is not None:
                        return row[n]
                return None

            for _, row in df.iterrows():
                # Image path candidates
                img_val = get_val(row, [
                    "image_path", "img_path", "image", "img", "screenshot_path", "image_file", "image_url"
                ])
                img_bytes = None
                if isinstance(img_val, dict):
                    # HF Datasets image struct {"path": str, "bytes": optional}
                    if "path" in img_val and img_val["path"]:
                        img_val = img_val["path"]
                    elif "bytes" in img_val and img_val["bytes"]:
                        img_bytes = img_val["bytes"]
                        img_val = ""
                if isinstance(img_val, (bytes, bytearray)):
                    try:
                        img_val = img_val.decode("utf-8", errors="ignore")
                    except Exception:
                        img_val = str(img_val)
                if not isinstance(img_val, str):
                    img_val = str(img_val) if img_val is not None else ""
                abs_img = resolve_image_path(img_val, pfile)
                iw = ih = None
                if not abs_img and img_bytes:
                    try:
                        # Create cache path under dataset_dir
                        cache_dir = os.path.join(dataset_dir, ".sf_parquet_img_cache")
                        os.makedirs(cache_dir, exist_ok=True)
                        sha = hashlib.sha1(img_bytes).hexdigest()
                        abs_img = os.path.join(cache_dir, f"{sha}.png")
                        if not os.path.exists(abs_img):
                            with Image.open(io.BytesIO(img_bytes)) as im:
                                iw, ih = im.size
                                im.convert("RGB").save(abs_img, format="PNG")
                        if iw is None or ih is None:
                            with Image.open(abs_img) as im2:
                                iw, ih = im2.size
                    except Exception:
                        abs_img = ""
                        iw = ih = None
                if not abs_img:
                    continue

                # Determine image size if still unknown
                if iw is None or ih is None:
                    try:
                        with Image.open(abs_img) as im:
                            iw, ih = im.size
                    except Exception:
                        continue

                # Instruction candidates
                instr = get_val(row, ["task", "instruction", "query", "goal", "caption", "description", "text"])
                if instr is None:
                    instr = "Locate the target region"
                else:
                    instr = str(instr)

                # Target: prefer bbox, else point
                bbox = get_val(row, ["bbox", "target_bbox", "box", "rect"])
                point_x = get_val(row, ["point_x", "cx", "center_x", "x_center", "target_x", "x"])
                point_y = get_val(row, ["point_y", "cy", "center_y", "y_center", "target_y", "y"])

                cx_norm = None
                cy_norm = None

                # Parse bbox if available
                blist = coerce_list(bbox)
                if blist is None and isinstance(bbox, str):
                    try:
                        blist = ast.literal_eval(bbox)
                        blist = blist if isinstance(blist, (list, tuple)) else None
                    except Exception:
                        blist = None
                if blist is not None and len(blist) >= 4:
                    try:
                        b0, b1, b2, b3 = float(blist[0]), float(blist[1]), float(blist[2]), float(blist[3])
                        # Detect format: [x1,y1,x2,y2] vs [x,y,w,h]
                        if (b2 > b0 and b3 > b1):
                            # likely corners
                            if max(abs(b0), abs(b1), abs(b2), abs(b3)) > 1.0001:
                                # pixels
                                cx = (b0 + b2) / 2.0
                                cy = (b1 + b3) / 2.0
                                cx_norm = cx / max(1.0, float(iw))
                                cy_norm = cy / max(1.0, float(ih))
                            else:
                                # normalized corners
                                cx_norm = (b0 + b2) / 2.0
                                cy_norm = (b1 + b3) / 2.0
                        else:
                            # treat as [x, y, w, h]
                            if max(abs(b0), abs(b1), abs(b2), abs(b3)) > 1.0001:
                                cx_norm = (b0 + b2 / 2.0) / max(1.0, float(iw))
                                cy_norm = (b1 + b3 / 2.0) / max(1.0, float(ih))
                            else:
                                cx_norm = b0 + b2 / 2.0
                                cy_norm = b1 + b3 / 2.0
                    except Exception:
                        cx_norm = None
                        cy_norm = None

                # Fallback to point
                if (cx_norm is None or cy_norm is None) and (point_x is not None and point_y is not None):
                    try:
                        px = float(point_x)
                        py = float(point_y)
                        if max(abs(px), abs(py)) > 1.0001:
                            cx_norm = px / max(1.0, float(iw))
                            cy_norm = py / max(1.0, float(ih))
                        else:
                            cx_norm = px
                            cy_norm = py
                    except Exception:
                        cx_norm = None
                        cy_norm = None

                if cx_norm is None or cy_norm is None:
                    continue

                cx_norm = min(1.0, max(0.0, float(cx_norm)))
                cy_norm = min(1.0, max(0.0, float(cy_norm)))

                samples.append({
                    "img_url": os.path.abspath(abs_img),
                    "element": [
                        {
                            "instruction": instr,
                            "point": [cx_norm, cy_norm],
                        }
                    ],
                })

        return "", samples

    # Default ShowUI-style datasets
    base_image_dir = os.path.join(dataset_dir, dataset_mapping[dataset])
    meta_dir = os.path.join(base_image_dir, "metadata")
    img_dir = os.path.join(base_image_dir, "images")
    with open(os.path.join(meta_dir, f"{split}.json")) as f:
        samples = json.load(f)
    return img_dir, samples


def build_prompt(processor, element_name: str, image_path: str, min_pixels: int, max_pixels: int):
    img = Image.open(image_path).convert("RGB")
    img_dict = {"type": "image", "min_pixels": min_pixels, "max_pixels": max_pixels}
    messages = grounding_to_qwen(element_name, img_dict, sample_io=0, user_prompt_random=False, xy_int=False, uniform_prompt=True)
    # Strong instruction to return only [x, y]
    if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], dict):
        try:
            messages[0]["content"].append({
                "type":"text",
                "text":"Return exactly two numbers in square brackets like [x, y] with both x and y in [0,1]. Do not include any other text, units, or explanation."
            })
        except Exception:
            pass
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt")
    return text, inputs


def parse_coord(output_text: str, img_size: Tuple[int, int] = None) -> Tuple[float, float]:
    """Parse a coordinate string into normalized [0,1] x,y.
    Supports:
      - [x, y] or (x, y)
      - 'x: a, y: b'
      - percentages like '30%, 40%'
      - pixel values if img_size is provided
      - 4-value bbox 'x, y, w, h' -> center normalized using img_size
    Returns (nan, nan) if parsing fails.
    """
    text = (output_text or "").strip()
    # If there are multiple bracketed sections, pick the first [ ... ] segment
    bracket = re.search(r"\[[^\]]+\]", text)
    if bracket:
        text = bracket.group(0)
    # First, try safe literal eval for simple list/tuple cases
    try:
        xy = ast.literal_eval(text)
        if isinstance(xy, (list, tuple)):
            nums = [float(v) for v in xy if isinstance(v, (int, float)) or (isinstance(v, str) and re.match(r"^[-+]?[0-9]*\.?[0-9]+%?$", v.strip()))]
            # handle percent tokens embedded in list
            def to_num(v):
                if isinstance(v, str) and v.strip().endswith('%'):
                    return float(v.strip()[:-1]) / 100.0
                return float(v)
            nums = [to_num(v) for v in xy if isinstance(v, (int, float, str))]
            if len(nums) >= 2:
                if len(nums) >= 4 and img_size is not None:
                    # treat as bbox center
                    x, y, w, h = nums[:4]
                    iw, ih = max(1.0, float(img_size[0])), max(1.0, float(img_size[1]))
                    # interpret as pixels if any dimension > 1, else normalized
                    if max(x, y, w, h) > 1.0001:
                        cx = (x + w / 2.0) / iw
                        cy = (y + h / 2.0) / ih
                    else:
                        cx = x + w / 2.0
                        cy = y + h / 2.0
                    return min(1.0, max(0.0, cx)), min(1.0, max(0.0, cy))
                # 2-value
                x, y = nums[0], nums[1]
                if img_size is not None and (abs(x) > 1.0001 or abs(y) > 1.0001):
                    iw, ih = max(1.0, float(img_size[0])), max(1.0, float(img_size[1]))
                    x = x / iw
                    y = y / ih
                return min(1.0, max(0.0, x)), min(1.0, max(0.0, y))
    except Exception:
        pass

    # Regex: extract up to 4 numeric tokens with optional %
    try:
        tokens = re.findall(r"[-+]?\d*\.?\d+%?", text)
        if len(tokens) >= 2:
            # convert tokens
            vals = []
            for t in tokens[:4]:
                if t.endswith('%'):
                    vals.append(float(t[:-1]) / 100.0)
                else:
                    vals.append(float(t))
            if len(vals) >= 4 and img_size is not None:
                x, y, w, h = vals[:4]
                iw, ih = max(1.0, float(img_size[0])), max(1.0, float(img_size[1]))
                if max(x, y, w, h) > 1.0001:
                    cx = (x + w / 2.0) / iw
                    cy = (y + h / 2.0) / ih
                else:
                    cx = x + w / 2.0
                    cy = y + h / 2.0
                return min(1.0, max(0.0, cx)), min(1.0, max(0.0, cy))
            # 2-value
            x, y = vals[0], vals[1]
            if img_size is not None and (abs(x) > 1.0001 or abs(y) > 1.0001):
                iw, ih = max(1.0, float(img_size[0])), max(1.0, float(img_size[1]))
                x = x / iw
                y = y / ih
            return min(1.0, max(0.0, x)), min(1.0, max(0.0, y))
    except Exception:
        pass

    return float("nan"), float("nan")


def l2_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    dx = (p1[0] - p2[0])
    dy = (p1[1] - p2[1])
    return math.sqrt(dx * dx + dy * dy)


def compute_reward(pred_xy: Tuple[float, float], tgt_xy: Tuple[float, float], tau: float, alpha: float) -> float:
    if any(math.isnan(v) for v in pred_xy):
        return -1.0
    d = l2_distance(pred_xy, tgt_xy)
    r = (1.0 if d < tau else 0.0) - alpha * min(d, 0.5)
    return max(-1.0, min(1.0, r))


@torch.no_grad()
def evaluate_screenspot_subset(processor, model, dataset_dir: str, limit: int, min_pixels: int, max_pixels: int, device: str) -> float:
    """Lightweight subset eval on ScreenSpot. Returns success rate in [0,1]."""
    meta_path = os.path.join(dataset_dir, "ScreenSpot", "metadata", "hf_test_full.json")
    if not os.path.exists(meta_path):
        return 0.0
    try:
        with open(meta_path) as f:
            items = json.load(f)
    except Exception:
        return 0.0

    N = min(limit, len(items)) if limit and limit > 0 else len(items)
    if N == 0:
        return 0.0

    ok = 0
    model_unwrapped = model.module if hasattr(model, "module") else model
    model_unwrapped.eval()

    for i in range(N):
        item = items[i]
        img_path = os.path.join(dataset_dir, "ScreenSpot", "images", item["img_url"]) 
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = (item["img_size"][0], item["img_size"][1]) if "img_size" in item else img.size

        messages = [
            {"role":"user","content":[
                {"type":"text","text":"Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1. Return exactly two numbers in square brackets like [x, y] with both x and y in [0,1]. Do not include any other text, units, or explanation."},
                {"type":"image","image":img,"min_pixels":min_pixels,"max_pixels":max_pixels},
                {"type":"text","text":item["task"]},
            ]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt")
        inputs = inputs.to(device)

        try:
            out = model_unwrapped.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                num_beams=1,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=False,
            )
            gen = out[:, inputs.input_ids.shape[1]:]
            pred_str = processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            pred = parse_coord(pred_str, img_size=(img_w, img_h))
            x, y, w, h = item["bbox"]
            gt = [x / img_w, y / img_h, (x + w) / img_w, (y + h) / img_h]
            ok += 1 if (not any(math.isnan(v) for v in pred)) and (gt[0] <= pred[0] <= gt[2]) and (gt[1] <= pred[1] <= gt[3]) else 0
        except Exception:
            continue

    model_unwrapped.train()
    return ok / N


def reinforce_step(model, processor, device, batch, args: RLArgs):
    model.train()

    texts: List[str] = []
    processor_inputs = {}
    meta_list = []

    for element_name, image_path, tgt_xy in batch:
        text, inputs = build_prompt(processor, element_name, image_path, args.min_visual_tokens * 28 * 28, args.max_visual_tokens * 28 * 28)
        texts.append(text)
        meta_list.append((tgt_xy, image_path))
        # accumulate processor batch
        for k, v in inputs.items():
            processor_inputs.setdefault(k, []).append(v)

    # stack batch
    for k in list(processor_inputs.keys()):
        processor_inputs[k] = torch.cat(processor_inputs[k], dim=0).to(device)

    # Unwrap DDP for generation if wrapped
    model_unwrapped = model.module if hasattr(model, "module") else model
    # Ensure float inputs match model dtype
    try:
        param_dtype = next(model_unwrapped.parameters()).dtype
    except StopIteration:
        param_dtype = torch.float32

    # Cast pixel values to model dtype to avoid dtype-induced NaNs
    if "pixel_values" in processor_inputs and processor_inputs["pixel_values"] is not None:
        processor_inputs["pixel_values"] = processor_inputs["pixel_values"].to(dtype=param_dtype)

    with torch.no_grad():
        safe_temperature = float(max(args.temperature, 1e-4)) if args.do_sample else 1.0
        gen_kwargs = {
            "max_new_tokens": int(max(1, args.max_new_tokens)),
            "do_sample": bool(args.do_sample),
            "temperature": safe_temperature,
            "num_beams": int(max(1, args.num_beams)),
            "eos_token_id": processor.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if args.top_p and args.top_p > 0.0:
            gen_kwargs["top_p"] = float(min(1.0, max(1e-6, args.top_p)))
        if args.top_k and args.top_k > 0:
            gen_kwargs["top_k"] = int(max(1, args.top_k))
        try:
            # Warmup: force greedy decoding for first N steps to reduce invalid parses
            force_greedy = getattr(args, "_global_step", 0) < int(max(0, args.warmup_steps))
            if force_greedy:
                generated = model_unwrapped.generate(
                    **processor_inputs,
                    max_new_tokens=int(max(1, args.max_new_tokens)),
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    use_cache=True,
                )
            else:
                generated = model_unwrapped.generate(
                    **processor_inputs,
                    **gen_kwargs,
                )
        except Exception:
            # Fallback to deterministic greedy decoding if sampler fails
            generated = model_unwrapped.generate(
                **processor_inputs,
                max_new_tokens=int(max(1, args.max_new_tokens)),
                do_sample=False,
                num_beams=1,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
            )

    # trim prompt tokens
    gen_trimmed = []
    for in_ids, out_ids in zip(processor_inputs["input_ids"], generated):
        trim = out_ids[len(in_ids) :]
        gen_trimmed.append(trim)
    gen_trimmed = torch.nn.utils.rnn.pad_sequence(gen_trimmed, batch_first=True, padding_value=processor.tokenizer.pad_token_id)

    # decode and compute rewards
    decoded = processor.batch_decode(gen_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    rewards = []
    for out_text, (tgt_xy, image_path) in zip(decoded, meta_list):
        # read image size for robust parsing (pixel/percent support)
        img_w, img_h = None, None
        try:
            with Image.open(image_path) as im:
                img_w, img_h = im.size
        except Exception:
            pass
        pred_xy = parse_coord(out_text, img_size=(img_w, img_h) if (img_w and img_h) else None)
        rewards.append(compute_reward(pred_xy, tgt_xy, args.tau_success, args.alpha_dist))
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)

    # Build inputs for policy loss (prompt + generated) and labels only on generated tokens
    input_ids_full = torch.cat([processor_inputs["input_ids"], gen_trimmed], dim=1)
    attention_mask_full = (input_ids_full != processor.tokenizer.pad_token_id).long()
    labels = input_ids_full.clone()
    # mask prompt tokens
    prompt_len = processor_inputs["input_ids"].shape[1]
    labels[:, :prompt_len] = IGNORE_INDEX

    outputs = model(
        input_ids=input_ids_full.to(device),
        attention_mask=attention_mask_full.to(device),
        pixel_values=processor_inputs.get("pixel_values"),
        image_grid_thw=processor_inputs.get("image_grid_thw"),
        labels=None,
    )

    # compute per-token NLL over generated tokens only
    vocab = outputs.logits.size(-1)
    logits = outputs.logits[:, prompt_len - 1 : -1, :].contiguous()
    # sanitize logits to avoid NaNs/Infs
    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
    target = input_ids_full[:, prompt_len:].contiguous()
    nll = F.cross_entropy(logits.view(-1, vocab), target.view(-1), reduction="none")
    nll = nll.view(target.size(0), target.size(1))

    token_mask = (target != processor.tokenizer.pad_token_id).float()
    seq_nll = (nll * token_mask).sum(dim=1) / (token_mask.sum(dim=1) + 1e-6)

    # Advantage using EMA baseline/std (robust for small batches)
    baseline = getattr(args, "_reward_baseline_for_adv", 0.0)
    std_est = getattr(args, "_reward_std_for_adv", 1.0)
    adv = rewards_tensor - float(baseline)
    if float(std_est) > 1e-6:
        adv = adv / float(std_est)
    else:
        adv = torch.zeros_like(adv)

    # Entropy bonus on generated tokens
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    # sanitize probs
    probs = torch.nan_to_num(probs, nan=0.0)
    token_entropy = -(probs * log_probs).sum(dim=-1)  # [B, T]
    entropy_per_seq = (token_entropy * token_mask).sum(dim=1) / (token_mask.sum(dim=1) + 1e-6)
    entropy_mean = entropy_per_seq.mean()

    # Optional KL to reference policy (PPO-style safety)
    kl_loss = torch.tensor(0.0, device=device)
    if getattr(args, "_ref_logits_fn", None) is not None and args.kl_coef > 0.0:
        with torch.no_grad():
            ref_logits = args._ref_logits_fn(
                input_ids=input_ids_full.to(device),
                attention_mask=attention_mask_full.to(device),
                pixel_values=processor_inputs.get("pixel_values"),
                image_grid_thw=processor_inputs.get("image_grid_thw"),
            )
        ref_logits = ref_logits[:, prompt_len - 1 : -1, :].contiguous()
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        # KL(policy || ref) per token
        kl_token = (probs * (log_probs - ref_log_probs)).sum(dim=-1)
        kl_per_seq = (kl_token * token_mask).sum(dim=1) / (token_mask.sum(dim=1) + 1e-6)
        kl_loss = kl_per_seq.mean()

    # REINFORCE objective with entropy and KL
    policy_loss = (adv * seq_nll).mean()
    loss = policy_loss - args.entropy_coef_start * entropy_mean + args.kl_coef * kl_loss
    if not torch.isfinite(loss):
        # fallback to finite surrogate
        loss = torch.nan_to_num(policy_loss, nan=0.0, posinf=1e4, neginf=1e4)

    # Prepare sample texts for optional logging
    sample_pairs = []
    try:
        for i, out_text in enumerate(decoded[: min(2, len(decoded))]):
            elem_name = batch[i][0] if i < len(batch) else ""
            sample_pairs.append((elem_name, out_text))
    except Exception:
        pass

    # lightweight debug: print first couple of step outputs for visibility
    try:
        if getattr(args, "_global_step", 0) < 3 and len(decoded) > 0:
            print(f"[DBG] raw='{decoded[0]}'")
    except Exception:
        pass

    # Step stats for JSONL logging
    try:
        invalid_rate = ((rewards_tensor <= -0.999).float().mean().item())
    except Exception:
        invalid_rate = 0.0
    step_stats = {
        "reward_mean": float(rewards_tensor.mean().item()),
        "reward_min": float(rewards_tensor.min().item()),
        "reward_max": float(rewards_tensor.max().item()),
        "adv_mean": float(adv.mean().item()),
        "adv_std": float(adv.std(unbiased=False).item()),
        "seq_nll_mean": float(seq_nll.mean().item()),
        "invalid_rate": float(invalid_rate),
    }

    return loss, float(rewards_tensor.mean().item()), float(entropy_mean.item()), float(kl_loss.item()), sample_pairs, step_stats


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--train_dataset", type=str, default="showui-desktop")
    parser.add_argument("--train_json", type=str, default="hf_train")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps_per_epoch", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--tau_success", type=float, default=0.06)
    parser.add_argument("--tau_success_end", type=float, default=0.06)
    parser.add_argument("--alpha_dist", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--temperature_end", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--entropy_coef_start", type=float, default=0.01)
    parser.add_argument("--entropy_coef_end", type=float, default=0.0)
    parser.add_argument("--kl_coef", type=float, default=0.0)
    parser.add_argument("--ref_model_id", type=str, default="")
    parser.add_argument("--ref_model_8bit", action="store_true")
    parser.add_argument("--model_id", type=str, default="showlab/ShowUI-2B")
    parser.add_argument("--min_visual_tokens", type=int, default=256)
    parser.add_argument("--max_visual_tokens", type=int, default=896)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit (requires bitsandbytes)")
    parser.add_argument("--log_dir", type=str, default="./runs/rl")
    parser.add_argument("--eval_subset_limit", type=int, default=200)
    parser.add_argument("--eval_every_steps", type=int, default=200)
    parser.add_argument("--save_every_epochs", type=int, default=1)
    parser.add_argument("--resume_from", type=str, default="", help="Resume from checkpoint directory")
    parser.add_argument("--save_optimizer", action="store_true")
    parser.add_argument("--eval_split", type=str, default="hf_test_full")
    parser.add_argument("--log_samples_every", type=int, default=100)
    parser.add_argument("--log_hist_every", type=int, default=100)
    parser.add_argument("--save_best", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--stats_jsonl", type=str, default="", help="Path to JSONL file to append per-step stats")
    args_ns = parser.parse_args()

    args = RLArgs(
        dataset_dir=args_ns.dataset_dir,
        train_dataset=args_ns.train_dataset,
        train_json=args_ns.train_json,
        steps_per_epoch=args_ns.steps_per_epoch,
        epochs=args_ns.epochs,
        batch_size=args_ns.batch_size,
        lr=args_ns.lr,
        tau_success=args_ns.tau_success,
        tau_success_end=args_ns.tau_success_end,
        alpha_dist=args_ns.alpha_dist,
        max_new_tokens=args_ns.max_new_tokens,
        temperature=args_ns.temperature,
        temperature_end=args_ns.temperature_end,
        top_p=args_ns.top_p,
        top_k=args_ns.top_k,
        do_sample=args_ns.do_sample,
        num_beams=args_ns.num_beams,
        grad_accum_steps=args_ns.grad_accum_steps,
        entropy_coef_start=args_ns.entropy_coef_start,
        entropy_coef_end=args_ns.entropy_coef_end,
        kl_coef=args_ns.kl_coef,
        ref_model_id=args_ns.ref_model_id,
        ref_model_8bit=args_ns.ref_model_8bit,
        model_id=args_ns.model_id,
        min_visual_tokens=args_ns.min_visual_tokens,
        max_visual_tokens=args_ns.max_visual_tokens,
        gradient_checkpointing=args_ns.gradient_checkpointing,
        load_in_8bit=args_ns.load_in_8bit,
        log_dir=args_ns.log_dir,
        eval_subset_limit=args_ns.eval_subset_limit,
        eval_every_steps=args_ns.eval_every_steps,
        save_every_epochs=args_ns.save_every_epochs,
        resume_from=args_ns.resume_from,
        save_optimizer=args_ns.save_optimizer,
        eval_split=args_ns.eval_split,
        log_samples_every=args_ns.log_samples_every,
        log_hist_every=args_ns.log_hist_every,
        save_best=args_ns.save_best,
        warmup_steps=args_ns.warmup_steps,
        stats_jsonl=args_ns.stats_jsonl,
    )

    set_seed(args.seed)

    # Single-GPU setup
    global_rank = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    # speed optimizations
    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    min_pixels = args.min_visual_tokens * 28 * 28
    max_pixels = args.max_visual_tokens * 28 * 28

    processor = AutoProcessor.from_pretrained(args.model_id, min_pixels=min_pixels, max_pixels=max_pixels)
    # Load model on single device
    if args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_id,
            quantization_config=quantization_config,
            device_map="auto",
        )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=torch_dtype)
        model.to(device)
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    # No DDP wrapping

    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Optional reference model for KL regularization
    ref_model = None
    if args.kl_coef > 0.0:
        try:
            ref_id = args.ref_model_id if args.ref_model_id else args.model_id
            if args.ref_model_8bit:
                ref_qconf = BitsAndBytesConfig(load_in_8bit=True)
                ref_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    ref_id,
                    quantization_config=ref_qconf,
                    device_map="auto",
                )
            else:
                ref_model = Qwen2VLForConditionalGeneration.from_pretrained(ref_id, torch_dtype=torch_dtype)
                ref_model.to(device)
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad_(False)

            def _ref_logits_fn(input_ids, attention_mask, pixel_values=None, image_grid_thw=None):
                out = ref_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    labels=None,
                )
                return out.logits

            args._ref_logits_fn = _ref_logits_fn
        except Exception as e:
            print(f"Reference model load failed, disabling KL: {e}")
            args.kl_coef = 0.0

    # Resume support (by directory naming or explicit path)
    start_epoch = 0
    global_step = 0
    resume_dir = None
    if args.resume_from:
        # If resume_from points to a base dir with epoch subdirs, pick latest
        if os.path.isdir(args.resume_from) and not os.path.exists(os.path.join(args.resume_from, "pytorch_model.bin")):
            cand_path, cand_epoch = find_latest_epoch_checkpoint(args.resume_from)
            if cand_epoch >= 0:
                resume_dir = cand_path
                start_epoch = cand_epoch
        else:
            resume_dir = args.resume_from
    else:
        # Auto-resume from CWD if any epoch checkpoints exist
        cand_path, cand_epoch = find_latest_epoch_checkpoint(os.getcwd())
        if cand_epoch >= 0:
            resume_dir = cand_path
            start_epoch = cand_epoch

    if resume_dir:
        try:
            # Load model weights
            if os.path.exists(os.path.join(resume_dir, "pytorch_model.bin")):
                sd = torch.load(os.path.join(resume_dir, "pytorch_model.bin"), map_location="cpu")
                model.load_state_dict(sd, strict=False)
            # Load optimizer and training meta if present
            state_path = os.path.join(resume_dir, "optimizer.pt")
            meta_path = os.path.join(resume_dir, "training_state.json")
            if os.path.exists(state_path):
                optimizer.load_state_dict(torch.load(state_path, map_location="cpu"))
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                start_epoch = int(meta.get("epoch", start_epoch))
                global_step = int(meta.get("global_step", 0))
            print(f"Resumed from {resume_dir} (epoch {start_epoch})")
        except Exception as e:
            print(f"Resume failed: {e}")

    img_dir, samples = load_split_items(args.dataset_dir, args.train_dataset, args.train_json)
    if global_rank == 0:
        print(f"Loaded {len(samples)} samples from {args.train_dataset}/{args.train_json}")

    writer = None
    if global_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.log_dir)
    # running baseline stats for batch_size=1 stability
    reward_baseline = 0.0
    reward_var = 1e-6

    # Utility: epoch-wise shuffled iterator over samples
    def sample_epoch_iterator(all_samples):
        idxs = list(range(len(all_samples)))
        random.shuffle(idxs)
        for i in idxs:
            yield all_samples[i]

    total_optim_steps = args.epochs * args.steps_per_epoch
    best_sr = -1.0

    # Initialize schedules from fixed starting points
    if not hasattr(args, "_sched_init"):
        object.__setattr__(args, "_temp_start", float(args.temperature))
        object.__setattr__(args, "_tau_start", float(args.tau_success))
        object.__setattr__(args, "_entropy_start", float(args.entropy_coef_start))
        object.__setattr__(args, "_sched_init", True)

    for epoch in range(start_epoch, args.epochs):
        running_loss = 0.0
        running_reward = 0.0
        running_entropy = 0.0
        running_kl = 0.0
        start = time.time()

        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch+1}/{args.epochs}")
        for _ in pbar:
            # schedules
            progress = (global_step + 1) / max(1, total_optim_steps)
            # temperature schedule
            temperature_now = float(args._temp_start + (args.temperature_end - args._temp_start) * progress)
            object.__setattr__(args, "temperature", temperature_now)
            # entropy schedule
            entropy_now = float(args._entropy_start + (args.entropy_coef_end - args._entropy_start) * progress)
            object.__setattr__(args, "entropy_coef_start", entropy_now)
            # tau schedule
            tau_now = float(args._tau_start + (args.tau_success_end - args._tau_start) * progress)
            object.__setattr__(args, "tau_success", tau_now)

            accum_loss = 0.0
            accum_reward = 0.0
            accum_entropy = 0.0
            accum_kl = 0.0

            optimizer.zero_grad(set_to_none=True)
            for micro in range(max(1, args.grad_accum_steps)):
                batch_items = []
                # shuffled iterator per epoch
                for _ in range(args.batch_size):
                    # try to draw a valid sample with at least one element
                    # fall back to random choice if needed
                    item = None
                    for cand in sample_epoch_iterator(samples):
                        if cand.get("element"):
                            item = cand
                            break
                    if item is None:
                        item = random.choice(samples)
                        if not item.get("element"):
                            continue
                    image_path = os.path.join(img_dir, item["img_url"]) if "img_url" in item else ""
                    element = random.choice(item["element"]) if item.get("element") else None
                    if element is None:
                        continue
                    element_name = element["instruction"]
                    tgt_xy = (float(element["point"][0]), float(element["point"][1]))
                    batch_items.append((element_name, image_path, tgt_xy))

                if not batch_items:
                    continue

                loss_tensor, reward, ent, kl, sample_pairs, step_stats = reinforce_step(model, processor, device, batch_items, args)
                (loss_tensor / max(1, args.grad_accum_steps)).backward()
                accum_loss += float(loss_tensor.item())
                accum_reward += reward
                accum_entropy += ent
                accum_kl += kl

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss = accum_loss / max(1, args.grad_accum_steps)
            reward = accum_reward / max(1, args.grad_accum_steps)
            ent = accum_entropy / max(1, args.grad_accum_steps)
            kl = accum_kl / max(1, args.grad_accum_steps)

            running_loss += loss
            running_reward += reward
            running_entropy += ent
            running_kl += kl
            pbar.set_postfix({"loss": f"{loss:.4f}", "reward": f"{reward:.3f}", "H": f"{ent:.3f}", "KL": f"{kl:.3f}"})

            # running baseline diagnostics (always update EMA for advantage, even without writer)
            try:
                reward_baseline = args.reward_ema_beta * reward_baseline + (1.0 - args.reward_ema_beta) * reward
                diff = reward - reward_baseline
                reward_var = args.reward_ema_beta * reward_var + (1.0 - args.reward_ema_beta) * (diff * diff)
                # expose EMA baseline/std for advantage computation
                try:
                    object.__setattr__(args, "_reward_baseline_for_adv", float(reward_baseline))
                    object.__setattr__(args, "_reward_std_for_adv", math.sqrt(max(1e-8, reward_var)))
                except Exception:
                    pass
            except Exception:
                pass

            if writer:
                writer.add_scalar("train/loss", loss, global_step)
                writer.add_scalar("train/reward", reward, global_step)
                writer.add_scalar("train/entropy", ent, global_step)
                if args.kl_coef > 0.0:
                    writer.add_scalar("train/kl", kl, global_step)
                # log EMA baseline/std to TB
                try:
                    writer.add_scalar("train/reward_baseline", reward_baseline, global_step)
                    writer.add_scalar("train/reward_std", math.sqrt(max(1e-8, reward_var)), global_step)
                except Exception:
                    pass
                # invalid parse rate approximation: 1.0 if reward == -1 and seq_nll finite
                try:
                    invalid_rate = 1.0 if reward <= -0.999 else 0.0
                    writer.add_scalar("train/invalid_parse_rate", invalid_rate, global_step)
                except Exception:
                    pass

            # Append per-step stats to JSONL if requested
            try:
                if args.stats_jsonl:
                    rec = {
                        "type": "train_step",
                        "step": int(global_step),
                        "epoch": int(epoch),
                        "time": float(time.time()),
                        "loss": float(loss),
                        "entropy_mean": float(ent),
                        "kl": float(kl),
                        "reward_mean": float(reward),
                        "reward_baseline": float(reward_baseline),
                        "reward_std_ema": float(math.sqrt(max(1e-8, reward_var))),
                        "adv_mean": float(step_stats.get("adv_mean", 0.0)),
                        "adv_std": float(step_stats.get("adv_std", 0.0)),
                        "seq_nll_mean": float(step_stats.get("seq_nll_mean", 0.0)),
                        "invalid_rate": float(step_stats.get("invalid_rate", 0.0)),
                        "temperature": float(args.temperature),
                        "tau_success": float(args.tau_success),
                        "entropy_coef": float(args.entropy_coef_start),
                        "lr": float(optimizer.param_groups[0].get("lr", 0.0)),
                    }
                    dname = os.path.dirname(os.path.abspath(args.stats_jsonl))
                    if dname:
                        os.makedirs(dname, exist_ok=True)
                    with open(args.stats_jsonl, "a") as f:
                        f.write(json.dumps(rec) + "\n")
            except Exception:
                pass

            # periodic subset eval
            if ((global_step + 1) % args.eval_every_steps == 0):
                min_pixels = args.min_visual_tokens * 28 * 28
                max_pixels = args.max_visual_tokens * 28 * 28
                sr = evaluate_screenspot_subset(processor, model, args.dataset_dir, args.eval_subset_limit, min_pixels, max_pixels, device)
                if writer:
                    writer.add_scalar("eval/screenspot_subset_success", sr, global_step)
                # Append eval record
                try:
                    if args.stats_jsonl:
                        rec = {
                            "type": "eval",
                            "step": int(global_step),
                            "epoch": int(epoch),
                            "time": float(time.time()),
                            "eval_subset_success": float(sr),
                            "eval_subset_limit": int(args.eval_subset_limit),
                            "min_visual_tokens": int(args.min_visual_tokens),
                            "max_visual_tokens": int(args.max_visual_tokens),
                        }
                        dname = os.path.dirname(os.path.abspath(args.stats_jsonl))
                        if dname:
                            os.makedirs(dname, exist_ok=True)
                        with open(args.stats_jsonl, "a") as f:
                            f.write(json.dumps(rec) + "\n")
                except Exception:
                    pass
                if args.save_best and sr > best_sr:
                    best_sr = sr
                    save_dir = os.path.join(os.getcwd(), f"rl_ckpt_best")
                    os.makedirs(save_dir, exist_ok=True)
                    model_to_save = model.module if hasattr(model, "module") else model
                    try:
                        torch.save(model_to_save.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
                        model_to_save.config.to_json_file(os.path.join(save_dir, "config.json"))
                    except Exception as e:
                        print(f"Best save failed: {e}")
                    processor.save_pretrained(save_dir)

            # optional histograms and text samples logging cadence
            if writer and args.log_hist_every > 0 and ((global_step + 1) % args.log_hist_every == 0):
                # Note: with current API we only have mean reward here; histogram will be sparse
                writer.add_histogram("train/reward_hist_mean", np.array([reward], dtype=np.float32), global_step)
            if writer and args.log_samples_every > 0 and ((global_step + 1) % args.log_samples_every == 0):
                try:
                    for i, (elem_name, out_text) in enumerate(sample_pairs):
                        writer.add_text(f"train/sample_{i}", f"Instruction: {elem_name}\nOutput: {out_text}", global_step)
                except Exception:
                    pass

            # debug prints moved into reinforce_step where decoded/meta_list are in scope

            global_step += 1
            try:
                object.__setattr__(args, "_global_step", int(global_step))
            except Exception:
                pass

        duration = time.time() - start
        print(f"Epoch {epoch+1} done in {duration:.1f}s | avg loss {running_loss/args.steps_per_epoch:.4f} | avg reward {running_reward/args.steps_per_epoch:.3f}")

        # end-of-epoch eval on subset
        min_pixels = args.min_visual_tokens * 28 * 28
        max_pixels = args.max_visual_tokens * 28 * 28
        sr = evaluate_screenspot_subset(processor, model, args.dataset_dir, args.eval_subset_limit, min_pixels, max_pixels, device)
        if writer:
            writer.add_scalar("eval/screenspot_subset_success_epoch", sr, epoch)
        print(f"Epoch {epoch+1} eval subset success: {sr:.4f}")

        # periodic save
        if ((epoch + 1) % args.save_every_epochs == 0):
            save_dir = os.path.join(os.getcwd(), f"rl_ckpt_epoch{epoch+1}")
            os.makedirs(save_dir, exist_ok=True)
            model_to_save = model.module if hasattr(model, "module") else model
            try:
                torch.save(model_to_save.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
                model_to_save.config.to_json_file(os.path.join(save_dir, "config.json"))
            except Exception as e:
                print(f"Fallback save failed: {e}")
            processor.save_pretrained(save_dir)
            # Save optimizer and training state
            if args.save_optimizer:
                try:
                    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
                    with open(os.path.join(save_dir, "training_state.json"), "w") as f:
                        json.dump({"epoch": epoch + 1, "global_step": global_step}, f)
                except Exception as e:
                    print(f"Saving optimizer failed: {e}")
            print(f"Saved RL checkpoint to {save_dir}")

    # No distributed cleanup needed


if __name__ == "__main__":
    main()


