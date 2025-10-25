#!/usr/bin/env python3
"""
rl_train_optimized.py

"""

import os
import io
import re
import ast
import sys
import json
import math
import time
import random
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from torch.utils.tensorboard import SummaryWriter

# Optional: hook in your ShowUI prompt builder if available
try:
    from data.template.shared_grounding import grounding_to_qwen  # type: ignore
    from data.dset_shared_grounding import dataset_mapping  # type: ignore
except Exception:
    grounding_to_qwen = None
    dataset_mapping = {}

IGNORE_INDEX = -100


# ------------------------------
# Utilities
# ------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def l2_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    dx = (p1[0] - p2[0])
    dy = (p1[1] - p2[1])
    return math.sqrt(dx * dx + dy * dy)


def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


# ------------------------------
# Upgraded reward utilities
# ------------------------------
def build_well_formed_checks(output_text: str, parsed_xy: Tuple[float, float]) -> Dict[str, bool]:
    """
    Return a dict of format checks used by `upgraded_reward`:
    - brackets: output contains [] or ()
    - two_nums: parse produced two numbers (not NaN)
    - in01: numbers are in [0,1]
    - no_extra: output looks like ONLY a bracketed pair (tolerates whitespace)
    """
    text = (output_text or "").strip()
    has_brackets = ('[' in text and ']' in text) or ('(' in text and ')' in text)
    two_nums = not any(math.isnan(v) for v in parsed_xy)
    in01 = (0.0 <= parsed_xy[0] <= 1.0) and (0.0 <= parsed_xy[1] <= 1.0)
    m = re.search(r"^[^\d\[\(\-\+]*([\[\(]?\s*[-+]?\d*\.?\d+\s*,\s*[-+]?\d*\.?\d+\s*[\]\)]?)\s*$", text)
    no_extra = m is not None
    return {"brackets": has_brackets, "two_nums": two_nums, "in01": in01, "no_extra": no_extra}


def upgraded_reward(
    pred_xy: Tuple[float, float], tgt_xy: Tuple[float, float],
    bbox_xyxy: Optional[Tuple[float, float, float, float]] = None,
    well_formed_checks: Optional[Dict[str, bool]] = None,
    sigma_scale: float = 0.35,
    beta_point: float = 1.0,
    alpha_format: float = 0.2,
    inside_bonus: float = 0.5,
    lin_penalty: float = 0.5,
) -> float:
    """
    Distance-shaped reward with:
      - Scale-aware Gaussian term around target (sigma from bbox diagonal if available)
      - Linear distance penalty
      - Inside-box bonus (if predicted point falls inside GT bbox)
      - Formatting score encouraging well-formed outputs
    All coordinates are expected normalized to [0,1].
    """
    # invalid prediction penalty
    if any(math.isnan(v) for v in pred_xy):
        return -1.0

    px, py = pred_xy
    cx, cy = tgt_xy

    # sigma from bbox diagonal if available
    if bbox_xyxy is not None:
        x1, y1, x2, y2 = bbox_xyxy
        bw, bh = max(1e-6, x2 - x1), max(1e-6, y2 - y1)
        diag = 0.5 * math.sqrt(bw * bw + bh * bh)
        sigma = max(1e-3, sigma_scale * diag)
        inside = (x1 <= px <= x2) and (y1 <= py <= y2)
    else:
        sigma = max(1e-3, sigma_scale * 0.25)  # fallback
        inside = False

    d = l2_distance((px, py), (cx, cy))
    gaussian = math.exp(- (d / sigma) ** 2)
    point = gaussian - lin_penalty * min(d, 1.0)
    if inside:
        point += inside_bonus

    wf = well_formed_checks or {}
    fscore = sum(1.0 for k in ("brackets", "two_nums", "in01", "no_extra") if wf.get(k, False)) / 4.0

    R = beta_point * point + alpha_format * fscore
    return float(max(-1.0, min(1.5, R)))


# ------------------------------
# Data loading (ShowUI/ScreenSpot + flexible parquet/Novelis styles)
# ------------------------------
def _resolve_abs(path_value: str, dataset_dir: str, parq_path: Optional[str] = None) -> str:
    if not path_value:
        return ""
    if os.path.isabs(path_value) and os.path.exists(path_value):
        return path_value
    cand1 = os.path.join(dataset_dir, path_value)
    if os.path.exists(cand1):
        return cand1
    if parq_path:
        base_dir = os.path.dirname(parq_path)
        cand2 = os.path.join(base_dir, path_value)
        if os.path.exists(cand2):
            return cand2
    return ""


def load_split_items(dataset_dir: str, dataset: str, split: str) -> Tuple[str, List[dict]]:
    """Return (img_dir, samples) where each sample resembles ShowUI format:
        {
          "img_url": "/abs/path/to/image.png",
          "element": [{"instruction": str, "point": [cx, cy], "bbox": [x,y,w,h] (optional)}]
        }
    """
    # Novelis-style JSON
    if dataset.lower() == "novelis":
        json_path = split if split.endswith(".json") else f"{split}.json"
        if not os.path.isabs(json_path):
            cand = os.path.join(dataset_dir, json_path)
            json_path = cand if os.path.exists(cand) else json_path
        with open(json_path) as f:
            items = json.load(f)
        out = []
        for it in items:
            img_path = _resolve_abs(it.get("image_context", ""), dataset_dir)
            if not os.path.exists(img_path):
                continue
            try:
                with Image.open(img_path) as im:
                    iw, ih = im.size
            except Exception:
                continue
            bbox = it.get("target_bbox")
            cx = cy = None
            if bbox and len(bbox) == 4:
                x, y, w, h = map(float, bbox)
                cx = (x + w / 2.0) / max(1.0, float(iw))
                cy = (y + h / 2.0) / max(1.0, float(ih))
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
            if cx is None:
                continue
            instr = it.get("goal") or it.get("rubric") or f"Locate target for {it.get('app','unknown')}"
            elem = {"instruction": instr, "point": [cx, cy]}
            # keep bbox (normalized) if present for reward scaling (optional)
            if bbox and len(bbox) == 4:
                elem["bbox_abs"] = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
                elem["img_size"] = [iw, ih]
            out.append({"img_url": os.path.abspath(img_path), "element": [elem]})
        return "", out

    # Parquet-style flexible loader
    if dataset.lower() in ("salesforce", "salesforce-parquet", "screenspot-parquet", "parquet"):
        try:
            import pandas as pd  # type: ignore
        except Exception:
            pd = None
        try:
            import pyarrow.parquet as pq  # type: ignore
        except Exception:
            pq = None

        base_path = split
        if not os.path.isabs(base_path):
            cand = os.path.join(dataset_dir, base_path)
            base_path = cand if os.path.exists(cand) else base_path

        parq_files: List[str] = []
        if os.path.isdir(base_path):
            for root, _, files in os.walk(base_path):
                for n in files:
                    if n.lower().endswith((".parquet", ".parq")):
                        parq_files.append(os.path.join(root, n))
        elif os.path.isfile(base_path) and base_path.lower().endswith((".parquet", ".parq")):
            parq_files.append(base_path)

        samples: List[dict] = []
        for pfile in parq_files:
            try:
                if pd is not None:
                    try:
                        df = pd.read_parquet(pfile, engine="pyarrow")
                    except Exception:
                        df = pd.read_parquet(pfile, engine="fastparquet")
                else:
                    df = pq.read_table(pfile).to_pandas() if pq is not None else None
                if df is None:
                    continue
            except Exception:
                continue

            def get(row, names: List[str]):
                for n in names:
                    if n in row and row[n] is not None:
                        return row[n]
                return None

            for _, row in df.iterrows():
                img_val = get(row, ["image_path","img_path","image","img","screenshot_path","image_file","image_url"]) or ""
                img_bytes = None
                if isinstance(img_val, dict):
                    if img_val.get("path"):
                        img_val = img_val["path"]
                    elif img_val.get("bytes"):
                        img_bytes = img_val["bytes"]
                        img_val = ""
                if isinstance(img_val, (bytes, bytearray)):
                    try:
                        img_val = img_val.decode("utf-8", errors="ignore")
                    except Exception:
                        img_val = str(img_val)

                abs_img = _resolve_abs(str(img_val), dataset_dir, parq_path=pfile)
                iw = ih = None
                if not abs_img and img_bytes:
                    try:
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
                if iw is None or ih is None:
                    try:
                        with Image.open(abs_img) as im:
                            iw, ih = im.size
                    except Exception:
                        continue

                instr = str(get(row, ["task","instruction","query","goal","caption","description","text"]) or "Locate the target region")

                bbox = get(row, ["bbox","target_bbox","box","rect"])
                px = get(row, ["point_x","cx","center_x","x_center","target_x","x"]) or get(row, ["x_center_norm"])
                py = get(row, ["point_y","cy","center_y","y_center","target_y","y"]) or get(row, ["y_center_norm"])
                cx = cy = None
                bx = by = bw = bh = None

                def _as_list(v):
                    if isinstance(v, (list, tuple)):
                        return list(v)
                    try:
                        import numpy as _np
                        if isinstance(v, _np.ndarray):
                            return v.tolist()
                    except Exception:
                        pass
                    if isinstance(v, str):
                        try:
                            vv = ast.literal_eval(v)
                            return vv if isinstance(vv, (list, tuple)) else None
                        except Exception:
                            return None
                    return None

                bl = _as_list(bbox)
                if bl and len(bl) >= 4:
                    b0, b1, b2, b3 = map(safe_float, bl[:4])
                    # corners vs [x,y,w,h]
                    if (b2 > b0 and b3 > b1):
                        # corners
                        if max(abs(b0),abs(b1),abs(b2),abs(b3)) > 1.0001:
                            cx = (b0 + b2) / 2.0 / max(iw,1)
                            cy = (b1 + b3) / 2.0 / max(ih,1)
                            bx, by, bw, bh = b0, b1, (b2-b0), (b3-b1)
                        else:
                            cx = (b0 + b2) / 2.0
                            cy = (b1 + b3) / 2.0
                            bx, by, bw, bh = b0, b1, (b2-b0), (b3-b1)
                    else:
                        # [x,y,w,h]
                        if max(abs(b0),abs(b1),abs(b2),abs(b3)) > 1.0001:
                            cx = (b0 + b2/2.0) / max(iw,1)
                            cy = (b1 + b3/2.0) / max(ih,1)
                            bx, by, bw, bh = b0, b1, b2, b3
                        else:
                            cx = b0 + b2/2.0
                            cy = b1 + b3/2.0
                            bx, by, bw, bh = b0, b1, b2, b3
                if (cx is None or cy is None) and (px is not None and py is not None):
                    px = safe_float(px); py = safe_float(py)
                    if max(abs(px),abs(py)) > 1.0001:
                        cx = px / max(iw,1)
                        cy = py / max(ih,1)
                    else:
                        cx, cy = px, py
                if cx is None or cy is None:
                    continue
                cx = max(0.0, min(1.0, float(cx)))
                cy = max(0.0, min(1.0, float(cy)))

                elem: Dict[str, Any] = {"instruction": instr, "point": [cx, cy]}
                if bx is not None:
                    elem["bbox_abs"] = [bx, by, bw, bh]
                    elem["img_size"] = [iw, ih]
                samples.append({"img_url": os.path.abspath(abs_img), "element": [elem]})
        return "", samples

    # Default: ShowUI-style directory
    base_image_dir = os.path.join(dataset_dir, dataset_mapping.get(dataset, dataset))
    meta_dir = os.path.join(base_image_dir, "metadata")
    img_dir = os.path.join(base_image_dir, "images")
    with open(os.path.join(meta_dir, f"{split}.json")) as f:
        samples = json.load(f)
    return img_dir, samples


# ------------------------------
# Prompt / Parsing
# ------------------------------
def build_prompt(processor, instruction: str, image_path: str, min_pixels: int, max_pixels: int):
    img = Image.open(image_path).convert("RGB")
    img_dict = {"type": "image", "min_pixels": min_pixels, "max_pixels": max_pixels}
    if grounding_to_qwen is not None:
        messages = grounding_to_qwen(instruction, img_dict, sample_io=0, user_prompt_random=False, xy_int=False, uniform_prompt=True)
    else:
        messages = [{"role":"user","content":[
            {"type":"text","text":"Given a screenshot and a text description, return the clickable location as normalized [x, y] in [0,1]."},
            {"type":"image","image": img, "min_pixels": min_pixels, "max_pixels": max_pixels},
            {"type":"text","text": instruction},
        ]}]
    # Strong formatting instruction
    if isinstance(messages, list) and messages and isinstance(messages[0], dict):
        try:
            messages[0]["content"].append({"type":"text","text":"Return exactly two numbers in square brackets like [x, y], both in [0,1]. No extra text."})
        except Exception:
            pass
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text, img


def normalize_fullwidth(s: str) -> str:
    # Map common full-width punctuation to ASCII
    return s.replace('，', ',').replace('［','[').replace('］',']')


def parse_coord(output_text: str, img_size: Optional[Tuple[int, int]] = None) -> Tuple[Tuple[float, float], bool]:
    """Parse model text into (x,y) in [0,1] and a well_formed flag.
    Supports [x,y], (x,y), percent tokens, pixel values (if img_size is provided), and four-value bbox -> center.
    """
    text = normalize_fullwidth((output_text or "").strip())
    # select first bracketed span if present
    bracket = re.search(r"\[[^\]]+\]", text)
    if bracket:
        text = bracket.group(0)
    well_formed = False

    # Try literal eval first
    try:
        xy = ast.literal_eval(text)
        if isinstance(xy, (list, tuple)) and len(xy) >= 2:
            well_formed = True
            toks: List[float] = []
            for v in xy[:4]:
                if isinstance(v, str) and v.strip().endswith('%'):
                    toks.append(float(v.strip()[:-1]) / 100.0)
                else:
                    toks.append(float(v))
            if len(toks) >= 4 and img_size is not None:
                x,y,w,h = toks[:4]
                iw, ih = (max(1.0, float(img_size[0])), max(1.0, float(img_size[1])))
                if max(x,y,w,h) > 1.0001:
                    cx = (x + w/2.0) / iw
                    cy = (y + h/2.0) / ih
                else:
                    cx = x + w/2.0
                    cy = y + h/2.0
                return (max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))), well_formed
            x, y = toks[0], toks[1]
            if img_size is not None and (abs(x) > 1.0001 or abs(y) > 1.0001):
                iw, ih = (max(1.0, float(img_size[0])), max(1.0, float(img_size[1])))
                x, y = x/iw, y/ih
            return (max(0.0, min(1.0, x)), max(0.0, min(1.0, y))), well_formed
    except Exception:
        pass

    # Regex fallback
    try:
        tokens = re.findall(r"[-+]?\d*\.?\d+%?", text)
        if len(tokens) >= 2:
            vals: List[float] = []
            for t in tokens[:4]:
                vals.append(float(t[:-1]) / 100.0 if t.endswith('%') else float(t))
            well_formed = True
            if len(vals) >= 4 and img_size is not None:
                x,y,w,h = vals[:4]
                iw, ih = (max(1.0, float(img_size[0])), max(1.0, float(img_size[1])))
                if max(x,y,w,h) > 1.0001:
                    cx = (x + w/2.0) / iw
                    cy = (y + h/2.0) / ih
                else:
                    cx = x + w/2.0
                    cy = y + h/2.0
                return (max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))), well_formed
            x, y = vals[0], vals[1]
            if img_size is not None and (abs(x) > 1.0001 or abs(y) > 1.0001):
                iw, ih = (max(1.0, float(img_size[0])), max(1.0, float(img_size[1])))
                x, y = x/iw, y/ih
            return (max(0.0, min(1.0, x)), max(0.0, min(1.0, y))), well_formed
    except Exception:
        pass

    return (float('nan'), float('nan')), False


# ------------------------------
# (Legacy) Reward kept for ablations
# ------------------------------
def compute_reward(pred_xy: Tuple[float,float], tgt_xy: Tuple[float,float], tau: float, alpha: float,
                    bbox_wh: Optional[Tuple[float,float]] = None, well_formed: bool = False) -> float:
    if any(math.isnan(v) for v in pred_xy):
        return -1.0 if not well_formed else -0.8
    d = l2_distance(pred_xy, tgt_xy)
    if bbox_wh:
        w,h = bbox_wh
        scale = 0.5 * math.sqrt(w*w + h*h) + 1e-6
        d = d / scale
    sigma = max(tau * 0.5, 1e-3)
    soft = math.exp(- (d / sigma)**2)
    grammar_bonus = 0.05 if well_formed else 0.0
    r = soft + grammar_bonus - alpha * min(d, 1.0)
    return max(-1.0, min(1.0, float(r)))


# ------------------------------
# Prefix-constrained decoding FSM (optional)
# ------------------------------
class XYFSM:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        # Build robust sets by scanning vocab for substrings (since many tokenizers use multi-char pieces)
        self.vocab = self.tok.get_vocab()
        def ids_with(chars: str):
            wanted = set()
            for tok, idx in self.vocab.items():
                if any(ch in tok for ch in chars):
                    wanted.add(idx)
            return wanted
        self.lbr_set = ids_with('[')
        self.rbr_set = ids_with(']')
        self.comma_set = ids_with(',')
        self.space_set = ids_with(' ')
        self.digit_set = ids_with('0123456789-+.')
        self.eos_id = self.tok.eos_token_id
        self.fallback_all = set(self.vocab.values())
        self.reset()

    def reset(self):
        self.state = 'S0'

    def allowed(self, generated_ids: List[int]) -> Set[int]:
        # Decode small tail for state update (robust to merged tokens)
        tail = generated_ids[-32:] if len(generated_ids) > 32 else generated_ids
        txt = self.tok.decode(tail, skip_special_tokens=True)
        txt = normalize_fullwidth(txt)
        if self.state == 'S0':
            self.state = 'Sx' if '[' in txt else 'S0'
            # If we can find any '['-containing tokens, constrain to them; otherwise, don't constrain at all
            return set(self.lbr_set) if len(self.lbr_set) > 0 else set(self.fallback_all)
        if self.state == 'Sx':
            if ']' in txt: self.state = 'Sdone'
            if ',' in txt: self.state = 'Sy'
            # allow digits, dot, sign, comma, space
            allowed: Set[int] = set()
            allowed |= self.digit_set
            allowed |= self.comma_set
            allowed |= self.space_set
            allowed |= self.rbr_set
            return allowed if len(allowed) > 0 else set(self.fallback_all)
        if self.state == 'Sy':
            if ']' in txt: self.state = 'Sdone'
            allowed: Set[int] = set()
            allowed |= self.digit_set
            allowed |= self.space_set
            allowed |= self.rbr_set
            return allowed if len(allowed) > 0 else set(self.fallback_all)
        if self.state == 'Sdone':
            return set([self.eos_id]) if self.eos_id is not None else set(self.fallback_all)
        return set()


def make_prefix_allowed_tokens_fn(tokenizer):
    fsm = XYFSM(tokenizer)
    def fn(batch_id: int, input_ids: torch.LongTensor):
        return list(fsm.allowed(input_ids.tolist()))
    return fn


# ------------------------------
# Coordinate-token masking
# ------------------------------
def coord_token_mask(tokenizer, ids_row: torch.Tensor) -> torch.Tensor:
    toks = tokenizer.convert_ids_to_tokens(ids_row.tolist())
    mask = []
    inside = False
    for t in toks:
        s = t.replace('▁', '')  # common marker from sentencepiece-like vocab
        if '[' in s:
            inside = True
        is_num = any(ch in '0123456789-+.' for ch in s)
        is_comma = ',' in s
        m = bool(inside and (is_num or is_comma or ']' in s))
        mask.append(1.0 if m else 0.0)
        if ']' in s:
            inside = False
    return torch.tensor(mask, dtype=torch.float32)


# ------------------------------
# Evaluation (ScreenSpot subset)
# ------------------------------
@torch.no_grad()
def evaluate_screenspot_subset(processor, model, dataset_dir: str, limit: int, min_pixels: int, max_pixels: int,
                               device: str, envs: Optional[Set[str]] = None, types: Optional[Set[str]] = None,
                               seed: int = 42) -> Dict[str, float]:
    meta_path = os.path.join(dataset_dir, "ScreenSpot", "metadata", "hf_test_full.json")
    if not os.path.exists(meta_path):
        return {"success": 0.0, "mean_dist": 1.0, "mean_dist_scaled": 1.0, "parse_valid": 0.0}
    try:
        with open(meta_path) as f:
            items = json.load(f)
    except Exception:
        return {"success": 0.0, "mean_dist": 1.0, "mean_dist_scaled": 1.0, "parse_valid": 0.0}

    if envs is not None:
        envs = {e.lower() for e in envs}
        items = [it for it in items if str(it.get("split","")).lower() in envs]
    if types is not None:
        types = {t.lower() for t in types}
        items = [it for it in items if str(it.get("data_type","")).lower() in types]

    rng = random.Random(seed)
    rng.shuffle(items)
    N = min(limit, len(items)) if limit and limit > 0 else len(items)
    items = items[:N]
    if N == 0:
        return {"success": 0.0, "mean_dist": 1.0, "mean_dist_scaled": 1.0, "parse_valid": 0.0}

    ok = 0
    dists = []
    dists_scaled = []
    parse_ok = 0

    model_eval = model.module if hasattr(model, "module") else model
    model_eval.eval()

    for it in items:
        img_path = os.path.join(dataset_dir, "ScreenSpot", "images", it["img_url"]) 
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path).convert("RGB")
        iw, ih = (it.get("img_size", [img.size[0], img.size[1]])[0], it.get("img_size", [img.size[0], img.size[1]])[1])
        messages = [{"role":"user","content":[
            {"type":"text","text":"Return the element location as normalized [x,y] in [0,1]. No extra text."},
            {"type":"image","image": img, "min_pixels": min_pixels, "max_pixels": max_pixels},
            {"type":"text","text": it["task"]},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt").to(device)
        out = model_eval.generate(**inputs, max_new_tokens=64, do_sample=False, num_beams=1,
                                  eos_token_id=processor.tokenizer.eos_token_id, use_cache=False)
        gen = out[:, inputs.input_ids.shape[1]:]
        pred_str = processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        (px,py), wf = parse_coord(pred_str, img_size=(iw,ih))
        parse_ok += int(wf)
        x,y,w,h = it["bbox"]
        cx = x + w/2.0
        cy = y + h/2.0
        gt = (cx/iw, cy/ih)
        d = l2_distance((px,py), gt) if not any(math.isnan(v) for v in (px,py)) else 1.0
        dists.append(d)
        scale = 0.5 * math.sqrt((w/iw)**2 + (h/ih)**2) + 1e-6
        dists_scaled.append(min(5.0, d/scale))
        ok += int((not any(math.isnan(v) for v in (px,py))) and (x/iw <= px <= (x+w)/iw) and (y/ih <= py <= (y+h)/ih))

    model_eval.train()
    return {
        "success": ok / max(1, N),
        "mean_dist": float(np.mean(dists)) if dists else 1.0,
        "mean_dist_scaled": float(np.mean(dists_scaled)) if dists_scaled else 1.0,
        "parse_valid": parse_ok / max(1, N),
    }


# ------------------------------
# One RL step (masked policy loss, masked entropy, masked KL)
# ------------------------------
def reinforce_step(model, processor, device, batch, args, ref_logits_fn=None):
    model.train()

    # Build prompts/images
    texts: List[str] = []
    images: List[Image.Image] = []
    metas: List[Dict[str, Any]] = []

    min_pixels = args.min_visual_tokens * 28 * 28
    max_pixels = args.max_visual_tokens * 28 * 28

    for element_name, image_path, tgt_xy, bbox_xyxy in batch:
        text, img = build_prompt(processor, element_name, image_path, min_pixels, max_pixels)
        texts.append(text)
        images.append(img)
        metas.append({"tgt_xy": tgt_xy, "image_path": image_path, "bbox_xyxy": bbox_xyxy})

    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt").to(device)

    model_unwrapped = model.module if hasattr(model, "module") else model
    param_dtype = next(model_unwrapped.parameters()).dtype
    if "pixel_values" in inputs and inputs["pixel_values"] is not None:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=param_dtype)

    # Decide greedy vs sampling
    force_greedy = bool(getting := getattr(args, "_force_greedy_steps", 0)) and int(getting) > 0
    if not force_greedy and args.warmup_steps > 0 and int(getattr(args, "_global_step", 0)) < int(args.warmup_steps):
        force_greedy = True

    gen_kwargs = {
        "max_new_tokens": int(max(1, args.max_new_tokens)),
        "do_sample": (args.do_sample and not force_greedy),
        "temperature": float(max(args.temperature, 1e-4)) if (args.do_sample and not force_greedy) else 1.0,
        "num_beams": int(1 if (args.do_sample and not force_greedy) else args.num_beams),
        "eos_token_id": processor.tokenizer.eos_token_id,
        "use_cache": True,
    }
    if gen_kwargs["do_sample"]:
        if args.top_p and args.top_p > 0.0:
            gen_kwargs["top_p"] = float(min(1.0, max(1e-6, args.top_p)))
        if args.top_k and args.top_k > 0:
            gen_kwargs["top_k"] = int(max(1, args.top_k))

    # Optional prefix constraint
    prefix_allowed_tokens_fn = None
    if args.constrained_decode:
        try:
            prefix_allowed_tokens_fn = make_prefix_allowed_tokens_fn(processor.tokenizer)
        except Exception:
            prefix_allowed_tokens_fn = None

    with torch.no_grad():
        try:
            generated = model_unwrapped.generate(
                **inputs,
                **gen_kwargs,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )
        except Exception:
            generated = model_unwrapped.generate(
                **inputs,
                max_new_tokens=int(max(1, args.max_new_tokens)),
                do_sample=False, num_beams=1,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
            )

    # Trim prompt
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        # Many causal LMs use EOS as PAD; ensures masks are well-defined
        pad_id = processor.tokenizer.eos_token_id
        if pad_id is None:
            # Last resort
            pad_id = 0
    gen_trimmed = []
    for in_ids, out_ids in zip(inputs["input_ids"], generated):
        gen_trimmed.append(out_ids[len(in_ids):])
    gen_trimmed = torch.nn.utils.rnn.pad_sequence(gen_trimmed, batch_first=True,
                                                  padding_value=pad_id)

    # Decode, compute rewards
    decoded = processor.batch_decode(gen_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    rewards = []
    parsed_ok = []

    for out_text, meta in zip(decoded, metas):
        try:
            with Image.open(meta["image_path"]) as im:
                iw, ih = im.size
        except Exception:
            iw = ih = None
        (px,py), wf = parse_coord(out_text, img_size=(iw,ih) if (iw and ih) else None)
        parsed_ok.append(1.0 if wf else 0.0)
        checks = build_well_formed_checks(out_text, (px,py))
        r = upgraded_reward(
            (px,py), meta["tgt_xy"],
            bbox_xyxy=meta.get("bbox_xyxy"),
            well_formed_checks=checks,
        )
        rewards.append(r)

    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    parsed_ok_tensor = torch.tensor(parsed_ok, dtype=torch.float32, device=device)

    # Prepare inputs for re-scoring
    input_ids_full = torch.cat([inputs["input_ids"], gen_trimmed], dim=1)
    attention_mask_full = (input_ids_full != pad_id).long()
    labels = input_ids_full.clone()
    prompt_len = inputs["input_ids"].shape[1]
    labels[:, :prompt_len] = IGNORE_INDEX

    outputs = model(
        input_ids=input_ids_full.to(device),
        attention_mask=attention_mask_full.to(device),
        pixel_values=inputs.get("pixel_values"),
        image_grid_thw=inputs.get("image_grid_thw"),
        labels=None,
    )

    V = outputs.logits.size(-1)
    logits = outputs.logits[:, prompt_len - 1 : -1, :].contiguous()
    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
    target = input_ids_full[:, prompt_len:].contiguous()

    # Coordinate-token mask
    coord_masks = []
    for row in input_ids_full[:, prompt_len:]:
        coord_masks.append(coord_token_mask(processor.tokenizer, row))
    coord_mask = torch.stack(coord_masks, dim=0).to(logits.device)

    token_ce = F.cross_entropy(logits.view(-1, V), target.view(-1), reduction="none").view_as(coord_mask)
    token_mask = (target != pad_id).float()

    # restrict loss to coord tokens (and valid positions)
    # Fallback: if no coord tokens detected for a sample, use all non-pad generated tokens
    coord_mask = coord_mask.to(logits.device)
    no_coord = (coord_mask.sum(dim=1) == 0)
    if no_coord.any():
        coord_mask[no_coord] = token_mask[no_coord]
    active_mask = (coord_mask * token_mask)
    seq_loss = (token_ce * active_mask).sum(dim=1) / (active_mask.sum(dim=1) + 1e-6)

    # Advantage: per-batch whitening
    adv = rewards_tensor.clone()
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-6)

    # Entropy (masked)
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    ent_token = -(probs * log_probs).sum(dim=-1)
    ent_masked = (ent_token * active_mask).sum(dim=1) / (active_mask.sum(dim=1) + 1e-6)
    entropy_mean = ent_masked.mean()

    # KL to reference (masked)
    kl_loss = torch.tensor(0.0, device=logits.device)
    if ref_logits_fn is not None and args.kl_coef > 0.0:
        with torch.no_grad():
            ref_logits = ref_logits_fn(
                input_ids=input_ids_full.to(device),
                attention_mask=attention_mask_full.to(device),
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
            )
        ref_logits = ref_logits[:, prompt_len - 1 : -1, :].contiguous()
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        kl_token = (probs * (log_probs - ref_log_probs)).sum(dim=-1)
        kl_masked = (kl_token * active_mask).sum(dim=1) / (active_mask.sum(dim=1) + 1e-6)
        kl_loss = kl_masked.mean()

    policy_loss = (adv * seq_loss).mean()
    loss = policy_loss - args.entropy_coef * entropy_mean + args.kl_coef * kl_loss

    if not torch.isfinite(loss):
        # Skip this batch; return safe numbers
        return None, float(rewards_tensor.mean().item()), float(entropy_mean.item()), float(kl_loss.item()), decoded, {
            "invalid": True
        }

    # Stats
    stats = {
        "reward_mean": float(rewards_tensor.mean().item()),
        "reward_min": float(rewards_tensor.min().item()),
        "reward_max": float(rewards_tensor.max().item()),
        "seq_loss_mean": float(seq_loss.mean().item()),
        "entropy_mean": float(entropy_mean.item()),
        "kl": float(kl_loss.item()),
        "parse_valid": float(parsed_ok_tensor.mean().item()),
    }

    return loss, stats["reward_mean"], stats["entropy_mean"], stats["kl"], decoded, stats


# ------------------------------
# Main training
# ------------------------------
@dataclass
class Args:
    dataset_dir: str
    train_dataset: str = "showui-desktop"
    train_json: str = "hf_train"
    epochs: int = 1
    steps_per_epoch: int = 200
    batch_size: int = 1
    grad_accum_steps: int = 1

    model_id: str = "showlab/ShowUI-2B"
    load_in_8bit: bool = False
    gradient_checkpointing: bool = False

    lr: float = 5e-6
    weight_decay: float = 0.01
    warmup_steps: int = 200

    tau_success: float = 0.06
    alpha_dist: float = 1.0

    max_new_tokens: int = 64
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.0
    top_k: int = 0
    num_beams: int = 1
    constrained_decode: bool = True

    entropy_coef: float = 0.01
    kl_coef: float = 1e-4
    target_kl: float = 0.08
    kl_adapt_rate: float = 1.5
    kl_adapt_every: int = 50
    min_kl_coef: float = 1e-4
    max_kl_coef: float = 5e-1

    min_visual_tokens: int = 256
    max_visual_tokens: int = 896

    log_dir: str = "./runs/rl_opt"
    save_every_epochs: int = 1
    save_best: bool = True
    eval_every_steps: int = 200
    eval_subset_limit: int = 400
    eval_envs: Optional[List[str]] = None
    eval_types: Optional[List[str]] = None
    stats_jsonl: str = ""

    safety_cooldown_steps: int = 20
    safety_entropy_threshold: float = 3.0
    safety_kl_multiplier: float = 5.0
    safety_temp_floor: float = 0.3
    safety_temp_decay: float = 0.5

    seed: int = 42

    ref_model_id: str = ""
    ref_model_8bit: bool = True


def lr_lambda_builder(total_steps: int, warmup_steps: int):
    warm = max(1, warmup_steps)
    def fn(step: int):
        if step < warm:
            return step / float(warm)
        prog = (step - warm) / max(1.0, total_steps - warm)
        # cosine decays from 1.0 to 0.2
        return 0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, prog))))
    return fn


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_dir', type=str, required=True)
    p.add_argument('--train_dataset', type=str, default='showui-desktop')
    p.add_argument('--train_json', type=str, default='hf_train')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--steps_per_epoch', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--grad_accum_steps', type=int, default=1)

    p.add_argument('--model_id', type=str, default='showlab/ShowUI-2B')
    p.add_argument('--load_in_8bit', action='store_true')
    p.add_argument('--gradient_checkpointing', action='store_true')

    p.add_argument('--lr', type=float, default=5e-6)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_steps', type=int, default=400)

    p.add_argument('--tau_success', type=float, default=0.06)
    p.add_argument('--alpha_dist', type=float, default=1.0)

    p.add_argument('--max_new_tokens', type=int, default=64)
    p.add_argument('--do_sample', action='store_true')
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top_p', type=float, default=0.0)
    p.add_argument('--top_k', type=int, default=0)
    p.add_argument('--num_beams', type=int, default=1)
    p.add_argument('--constrained_decode', action='store_true')

    p.add_argument('--entropy_coef', type=float, default=0.01)
    p.add_argument('--kl_coef', type=float, default=1e-4)
    p.add_argument('--target_kl', type=float, default=0.08)
    p.add_argument('--kl_adapt_rate', type=float, default=1.5)
    p.add_argument('--kl_adapt_every', type=int, default=50)
    p.add_argument('--min_kl_coef', type=float, default=1e-4)
    p.add_argument('--max_kl_coef', type=float, default=5e-1)

    p.add_argument('--min_visual_tokens', type=int, default=256)
    p.add_argument('--max_visual_tokens', type=int, default=896)

    p.add_argument('--log_dir', type=str, default='./runs/rl_opt')
    p.add_argument('--save_every_epochs', type=int, default=1)
    p.add_argument('--save_best', action='store_true')
    p.add_argument('--eval_every_steps', type=int, default=200)
    p.add_argument('--eval_subset_limit', type=int, default=400)
    p.add_argument('--eval_envs', type=str, nargs='*', default=['desktop'])
    p.add_argument('--eval_types', type=str, nargs='*', default=None)
    p.add_argument('--stats_jsonl', type=str, default='')

    p.add_argument('--safety_cooldown_steps', type=int, default=20)
    p.add_argument('--safety_entropy_threshold', type=float, default=3.0)
    p.add_argument('--safety_kl_multiplier', type=float, default=5.0)
    p.add_argument('--safety_temp_floor', type=float, default=0.3)
    p.add_argument('--safety_temp_decay', type=float, default=0.5)

    p.add_argument('--seed', type=int, default=42)

    p.add_argument('--ref_model_id', type=str, default='')
    p.add_argument('--ref_model_8bit', action='store_true')

    ns = p.parse_args()
    args = Args(**vars(ns))

    # Setup
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_dtype = torch.bfloat16 if device == 'cuda' else torch.float32

    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    # Processor
    min_pixels = args.min_visual_tokens * 28 * 28
    max_pixels = args.max_visual_tokens * 28 * 28
    processor = AutoProcessor.from_pretrained(args.model_id, min_pixels=min_pixels, max_pixels=max_pixels)

    # Model
    if args.load_in_8bit:
        qconf = BitsAndBytesConfig(load_in_8bit=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_id, quantization_config=qconf, device_map="auto")
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=torch_dtype)
        model.to(device)
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    # Ref model for KL
    ref_model = None
    ref_logits_fn = None
    if args.kl_coef > 0.0:
        try:
            ref_id = args.ref_model_id if args.ref_model_id else args.model_id
            if args.ref_model_8bit:
                rq = BitsAndBytesConfig(load_in_8bit=True)
                ref_model = Qwen2VLForConditionalGeneration.from_pretrained(ref_id, quantization_config=rq, device_map="auto")
            else:
                ref_model = Qwen2VLForConditionalGeneration.from_pretrained(ref_id, torch_dtype=torch_dtype)
                ref_model.to(device)
            ref_model.eval()
            for p_ in ref_model.parameters():
                p_.requires_grad_(False)

            def _ref_logits_fn(input_ids, attention_mask, pixel_values=None, image_grid_thw=None):
                out = ref_model(input_ids=input_ids, attention_mask=attention_mask,
                                pixel_values=pixel_values, image_grid_thw=image_grid_thw, labels=None)
                return out.logits
            ref_logits_fn = _ref_logits_fn
        except Exception as e:
            print(f"[WARN] ref model load failed, disabling KL: {e}")
            args.kl_coef = 0.0

    # Optimizer (WD excluding bias/LayerNorm)
    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(grouped, lr=args.lr)

    total_steps = args.epochs * args.steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_builder(total_steps, args.warmup_steps))

    # Data
    img_dir, samples = load_split_items(args.dataset_dir, args.train_dataset, args.train_json)
    print(f"Loaded {len(samples)} samples from {args.train_dataset}/{args.train_json}")

    # Logging
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    # Helper: epoch iterator with pointer
    def make_epoch_stream(all_samples):
        idxs = list(range(len(all_samples)))
        random.shuffle(idxs)
        return idxs, 0

    idxs, ptr = make_epoch_stream(samples)

    def next_sample():
        nonlocal idxs, ptr
        if ptr >= len(idxs):
            idxs, ptr = make_epoch_stream(samples)
        it = samples[idxs[ptr]]
        ptr += 1
        return it

    # Training loop
    best_metric = -1.0
    global_step = 0

    for epoch in range(args.epochs):
        running = {"loss": 0.0, "reward": 0.0, "entropy": 0.0, "kl": 0.0}
        t0 = time.time()

        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch+1}/{args.epochs}")
        for _ in pbar:
            accum = {"loss": 0.0, "reward": 0.0, "entropy": 0.0, "kl": 0.0}
            optimizer.zero_grad(set_to_none=True)

            for micro in range(max(1, args.grad_accum_steps)):
                # Build batch
                batch_items = []
                tries = 0
                while len(batch_items) < args.batch_size and tries < args.batch_size * 4:
                    item = next_sample()
                    tries += 1
                    if not item.get("element"): continue
                    element = random.choice(item["element"])
                    image_path = os.path.join(img_dir, item["img_url"]) if img_dir else item.get("img_url", "")
                    if not image_path: continue
                    tgt_xy = (float(element["point"][0]), float(element["point"][1]))
                    bbox_xyxy = None
                    if element.get("bbox_abs") and element.get("img_size"):
                        bx,by,bw,bh = element["bbox_abs"]
                        iw,ih = element["img_size"]
                        iwf, ihf = max(1.0, float(iw)), max(1.0, float(ih))
                        x1 = float(bx) / iwf
                        y1 = float(by) / ihf
                        x2 = float(bx + bw) / iwf
                        y2 = float(by + bh) / ihf
                        # clamp to [0,1]
                        x1 = max(0.0, min(1.0, x1)); y1 = max(0.0, min(1.0, y1))
                        x2 = max(0.0, min(1.0, x2)); y2 = max(0.0, min(1.0, y2))
                        if x2 > x1 and y2 > y1:
                            bbox_xyxy = (x1, y1, x2, y2)
                    batch_items.append((str(element["instruction"]), image_path, tgt_xy, bbox_xyxy))

                if not batch_items:
                    continue

                res = reinforce_step(model, processor, device, batch_items, args, ref_logits_fn)
                if res[0] is None:  # NaN/Inf step skipped
                    continue
                loss_tensor, reward, ent, kl, decoded, stats = res
                (loss_tensor / max(1, args.grad_accum_steps)).backward()

                accum["loss"] += float(loss_tensor.item())
                accum["reward"] += float(reward)
                accum["entropy"] += float(ent)
                accum["kl"] += float(kl)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Metrics
            loss = accum["loss"] / max(1, args.grad_accum_steps)
            reward = accum["reward"] / max(1, args.grad_accum_steps)
            ent = accum["entropy"] / max(1, args.grad_accum_steps)
            kl = accum["kl"] / max(1, args.grad_accum_steps)

            running["loss"] += loss
            running["reward"] += reward
            running["entropy"] += ent
            running["kl"] += kl

            pbar.set_postfix({"loss": f"{loss:.4f}", "R": f"{reward:.3f}", "H": f"{ent:.3f}", "KL": f"{kl:.3f}", "klc": f"{args.kl_coef:.3g}"})

            # Adaptive KL control (masked KL surrogate)
            try:
                if args.kl_adapt_every > 0 and ((global_step + 1) % args.kl_adapt_every == 0):
                    target = float(args.target_kl)
                    rate = float(args.kl_adapt_rate)
                    if target > 0.0 and rate > 1.0:
                        new_coef = float(args.kl_coef)
                        if kl > target * 1.3:
                            new_coef = new_coef * rate if new_coef > 0.0 else max(1e-4, float(args.min_kl_coef))
                        elif kl < target / 1.3:
                            new_coef = new_coef / rate
                        args.kl_coef = float(min(args.max_kl_coef, max(args.min_kl_coef, new_coef)))
            except Exception:
                pass

            # Safety cooldown (finite)
            try:
                meltdown = False
                if ent > args.safety_entropy_threshold:
                    meltdown = True
                if args.target_kl > 0.0 and kl > args.target_kl * args.safety_kl_multiplier:
                    meltdown = True
                if meltdown:
                    steps = max(int(getattr(args, "_force_greedy_steps", 0)), int(args.safety_cooldown_steps))
                    setattr(args, "_force_greedy_steps", steps)
                    args.temperature = max(args.safety_temp_floor, args.temperature * args.safety_temp_decay)
            except Exception:
                pass

            # Decrement cooldown counter
            fs = int(getattr(args, "_force_greedy_steps", 0))
            if fs > 0:
                setattr(args, "_force_greedy_steps", fs - 1)

            # Logging
            if writer:
                writer.add_scalar("train/loss", loss, global_step)
                writer.add_scalar("train/reward", reward, global_step)
                writer.add_scalar("train/entropy_masked", ent, global_step)
                writer.add_scalar("train/kl_masked", kl, global_step)
                writer.add_scalar("train/kl_coef", float(args.kl_coef), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar("train/cooldown_active", 1.0 if fs>0 else 0.0, global_step)

            # Stats JSONL
            if args.stats_jsonl:
                try:
                    rec = {
                        "type": "train_step",
                        "step": int(global_step),
                        "epoch": int(epoch),
                        "time": float(time.time()),
                        "loss": float(loss),
                        "reward": float(reward),
                        "entropy": float(ent),
                        "kl": float(kl),
                        "kl_coef": float(args.kl_coef),
                        "lr": float(optimizer.param_groups[0]['lr']),
                        "cooldown": int(fs),
                    }
                    dname = os.path.dirname(os.path.abspath(args.stats_jsonl))
                    if dname:
                        os.makedirs(dname, exist_ok=True)
                    with open(args.stats_jsonl, 'a') as f:
                        f.write(json.dumps(rec) + "\n")
                except Exception:
                    pass

            # Periodic eval
            if args.eval_every_steps > 0 and ((global_step + 1) % args.eval_every_steps == 0):
                ev = evaluate_screenspot_subset(
                    processor, model, args.dataset_dir, args.eval_subset_limit,
                    args.min_visual_tokens*28*28, args.max_visual_tokens*28*28,
                    device,
                    envs=set(args.eval_envs) if args.eval_envs else None,
                    types=set(args.eval_types) if args.eval_types else None,
                )
                if writer:
                    writer.add_scalar("eval/success", ev["success"], global_step)
                    writer.add_scalar("eval/mean_dist", ev["mean_dist"], global_step)
                    writer.add_scalar("eval/mean_dist_scaled", ev["mean_dist_scaled"], global_step)
                    writer.add_scalar("eval/parse_valid", ev["parse_valid"], global_step)
                if args.stats_jsonl:
                    try:
                        rec = {"type": "eval", "step": int(global_step), "epoch": int(epoch), "time": float(time.time()), **ev}
                        with open(args.stats_jsonl, 'a') as f:
                            f.write(json.dumps(rec) + "\n")
                    except Exception:
                        pass
                if args.save_best and ev["success"] > best_metric:
                    best_metric = ev["success"]
                    save_dir = os.path.join(os.getcwd(), "rl_ckpt_best")
                    os.makedirs(save_dir, exist_ok=True)
                    model_to_save = model.module if hasattr(model, "module") else model
                    torch.save(model_to_save.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
                    model_to_save.config.to_json_file(os.path.join(save_dir, "config.json"))
                    processor.save_pretrained(save_dir)
                    with open(os.path.join(save_dir, "meta.json"), 'w') as f:
                        json.dump({"args": vars(args), "time": time.time(), "best_success": best_metric}, f, indent=2)

            global_step += 1
            setattr(args, "_global_step", int(global_step))

        dur = time.time() - t0
        print(f"Epoch {epoch+1} done in {dur:.1f}s | avg loss {running['loss']/args.steps_per_epoch:.4f} | avg R {running['reward']/args.steps_per_epoch:.3f}")

        # End-of-epoch eval and checkpoint
        ev = evaluate_screenspot_subset(
            processor, model, args.dataset_dir, args.eval_subset_limit,
            args.min_visual_tokens*28*28, args.max_visual_tokens*28*28, device,
            envs=set(args.eval_envs) if args.eval_envs else None,
            types=set(args.eval_types) if args.eval_types else None,
        )
        print(f"Epoch {epoch+1} eval: success={ev['success']:.4f} dist={ev['mean_dist']:.4f} d/scale={ev['mean_dist_scaled']:.4f} parse={ev['parse_valid']:.4f}")
        if writer:
            writer.add_scalar("eval_epoch/success", ev["success"], epoch)
            writer.add_scalar("eval_epoch/mean_dist", ev["mean_dist"], epoch)
            writer.add_scalar("eval_epoch/mean_dist_scaled", ev["mean_dist_scaled"], epoch)
            writer.add_scalar("eval_epoch/parse_valid", ev["parse_valid"], epoch)

        if ((epoch + 1) % args.save_every_epochs == 0):
            save_dir = os.path.join(os.getcwd(), f"rl_ckpt_epoch{epoch+1}")
            os.makedirs(save_dir, exist_ok=True)
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save(model_to_save.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
            model_to_save.config.to_json_file(os.path.join(save_dir, "config.json"))
            processor.save_pretrained(save_dir)
            with open(os.path.join(save_dir, "meta.json"), 'w') as f:
                json.dump({"args": vars(args), "time": time.time(), "epoch": epoch+1}, f, indent=2)
            print(f"Saved checkpoint to {save_dir}")

    print("Training complete.")


if __name__ == "__main__":
    main()
