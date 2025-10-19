import os
import ast
import math
import time
import json
import random
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
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

# Prefer memory-efficient SDPA kernels where available to lower peak memory
try:
    from torch.backends.cuda import sdp_kernel
    sdp_kernel.enable_flash_sdp(False)
    sdp_kernel.enable_mem_efficient_sdp(True)
    sdp_kernel.enable_math_sdp(True)
except Exception:
    pass


@dataclass
class MTRLArgs:
    dataset_dir: str
    train_dataset: str = "showui-desktop"
    train_json: str = "hf_train"
    min_visual_tokens: int = 192
    max_visual_tokens: int = 640
    model_id: str = "showlab/ShowUI-2B"
    lr: float = 5e-6
    batch_size: int = 1
    steps_per_epoch: int = 200
    epochs: int = 1
    # decoding
    max_new_tokens: int = 24
    temperature: float = 0.7
    top_p: float = 0.0
    top_k: int = 0
    do_sample: bool = True
    num_beams: int = 1
    # schedules
    temperature_end: float = 0.7
    entropy_coef_start: float = 0.01
    entropy_coef_end: float = 0.0
    # multi-turn params
    turns_per_traj: int = 2
    gamma: float = 0.99
    # reward shaping
    tau_success: float = 0.08
    tau_success_end: float = 0.06
    alpha_dist: float = 0.5
    alpha_dist_end: float = 1.0
    # training
    grad_accum_steps: int = 1
    gradient_checkpointing: bool = False
    load_in_8bit: bool = False
    # logging
    log_dir: str = "./runs/rl-mt"
    eval_subset_limit: int = 1000
    eval_every_steps: int = 200
    eval_subset_limit_train: int = 200
    eval_subset_limit_epoch: int = 1000
    full_eval_every_epochs: int = 5
    eval_max_new_tokens: int = 24
    eval_disable_mid: bool = False
    log_samples_every: int = 100
    log_hist_every: int = 100
    save_every_epochs: int = 10
    save_best: bool = True
    # KL regularization
    kl_coef: float = 0.0
    ref_model_id: str = ""
    ref_model_8bit: bool = True
    # misc
    seed: int = 42
    warmup_steps: int = 50
    resume_from: str = ""
    save_optimizer: bool = True
    eval_split: str = "hf_test_full"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_latest_epoch_checkpoint(base_dir: str) -> tuple:
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
    base_image_dir = os.path.join(dataset_dir, dataset_mapping[dataset])
    meta_dir = os.path.join(base_image_dir, "metadata")
    img_dir = os.path.join(base_image_dir, "images")
    with open(os.path.join(meta_dir, f"{split}.json")) as f:
        samples = json.load(f)
    return img_dir, samples


def load_osworld_items(dataset_dir: str, split: str) -> Tuple[str, List[dict]]:
    """Load OSWorld-style multi-turn items.
    Expected structure (flexible): each item has a list of steps, where each step contains
    an instruction and a target (point or bbox). Optional per-step image URLs.
    """
    base_dir = os.path.join(dataset_dir, "OSWorld")
    meta_dir = os.path.join(base_dir, "metadata")
    img_dir = os.path.join(base_dir, "images")
    meta_path = os.path.join(meta_dir, f"{split}.json")
    with open(meta_path) as f:
        raw = json.load(f)

    def normalize_point(step, img_w=None, img_h=None):
        # prefer normalized point
        if "point" in step and isinstance(step["point"], (list, tuple)) and len(step["point"]) == 2:
            try:
                x, y = float(step["point"][0]), float(step["point"][1])
                return [min(1.0, max(0.0, x)), min(1.0, max(0.0, y))]
            except Exception:
                pass
        # derive from bbox center if available
        if "bbox" in step and isinstance(step["bbox"], (list, tuple)) and len(step["bbox"]) == 4 and img_w and img_h:
            try:
                x, y, w, h = step["bbox"]
                cx = (x + w / 2.0) / float(img_w)
                cy = (y + h / 2.0) / float(img_h)
                return [min(1.0, max(0.0, cx)), min(1.0, max(0.0, cy))]
            except Exception:
                pass
        return None

    samples: List[dict] = []
    for item in raw:
        steps_src = item.get("steps") or item.get("trajectory") or []
        base_img_url = item.get("img_url") or item.get("base_img_url") or None
        img_size = item.get("img_size")
        img_w, img_h = (img_size[0], img_size[1]) if isinstance(img_size, (list, tuple)) and len(img_size) == 2 else (None, None)
        norm_steps = []
        for st in steps_src:
            instr = st.get("instruction") or st.get("query") or st.get("text") or ""
            s_img = st.get("img_url") or base_img_url
            pt = normalize_point(st, img_w=img_w, img_h=img_h)
            if pt is None:
                # skip steps without a usable target
                continue
            norm_steps.append({"instruction": instr, "point": pt, "img_url": s_img})
        if len(norm_steps) == 0:
            continue
        samples.append({"base_img_url": base_img_url, "steps": norm_steps})

    return img_dir, samples


def load_mind2web_items(dataset_dir: str, split: str) -> Tuple[str, List[dict]]:
    """Load Mind2Web from HF shards data/train/*.json (raw osunlp/Mind2Web format).
    Reference: https://huggingface.co/datasets/osunlp/Mind2Web
    Maps action sequences into multi-turn steps with per-step instructions and targets.
    """
    import glob
    base_dir = os.path.join(dataset_dir, "Mind2Web")
    data_dir = os.path.join(base_dir, "data")
    # Map split name to HF folder (train, test_task, test_website, test_domain)
    split_map = {"train": "train", "hf_train": "train", "test": "test_task", "test_task": "test_task", "test_website": "test_website", "test_domain": "test_domain"}
    split_dir = os.path.join(data_dir, split_map.get(split, split))
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Mind2Web split folder not found: {split_dir}. Download with: huggingface-cli download osunlp/Mind2Web --repo-type dataset --local-dir {base_dir}")
    
    shard_files = sorted(glob.glob(os.path.join(split_dir, "*.json")))
    if not shard_files:
        raise FileNotFoundError(f"No JSON shards in {split_dir}")
    
    raw = []
    for fp in shard_files:
        with open(fp) as f:
            raw.extend(json.load(f))
    
    # For now, no images; render on-the-fly or skip img_url
    # If you have a render script, run it first or point img_dir to a rendered screenshot folder
    img_dir = os.path.join(base_dir, "screenshots")
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(base_dir, "images")

    def normalize_point(bbox, img_w=None, img_h=None):
        if bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4 and img_w and img_h:
            try:
                x, y, w, h = bbox
                cx = (x + w / 2.0) / float(img_w)
                cy = (y + h / 2.0) / float(img_h)
                return [min(1.0, max(0.0, cx)), min(1.0, max(0.0, cy))]
            except Exception:
                return None
        return None

    samples: List[dict] = []
    for item in raw:
        # Mind2Web raw schema: "confirmed_task", "actions": [{"action_uid", "operation": {"op", "value"}, "pos_candidates": [...]}]
        task = item.get("confirmed_task", "")
        actions = item.get("actions", [])
        if not actions:
            continue
        norm_steps = []
        for act in actions:
            op = act.get("operation", {})
            op_type = op.get("op", "CLICK")
            op_val = op.get("value", "")
            # Instruction is task + action type [+ value if TYPE/SELECT]
            if op_type == "TYPE":
                instr = f"{task} [Type: {op_val}]"
            elif op_type == "SELECT":
                instr = f"{task} [Select: {op_val}]"
            else:
                instr = f"{task} [Click]"
            # Extract target from pos_candidates (first positive element's attributes → bbox)
            pos_cands = act.get("pos_candidates", [])
            if not pos_cands:
                continue
            # Parse attributes JSON string to get bbox
            try:
                attrs = json.loads(pos_cands[0].get("attributes", "{}"))
                bbox_str = attrs.get("bounding_box_rect") or attrs.get("rect") or None
                if bbox_str and isinstance(bbox_str, str):
                    # Format like "x,y,width,height"
                    parts = [float(v.strip()) for v in bbox_str.split(",")]
                    if len(parts) == 4:
                        x, y, w, h = parts
                        # Derive img_w/img_h from viewport or heuristic (1920x1080 default)
                        img_w, img_h = 1920, 1080
                        pt = normalize_point([x, y, w, h], img_w=img_w, img_h=img_h)
                    else:
                        pt = None
                else:
                    pt = None
            except Exception:
                pt = None
            if pt is None:
                continue
            # Image rendering: you'd use action_uid or cleaned_html to render a screenshot
            # For now, placeholder; if you have a render script, save to img_dir/<action_uid>.png
            s_img = f"{act.get('action_uid', 'unknown')}.png"
            norm_steps.append({"instruction": instr, "point": pt, "img_url": s_img})
        if len(norm_steps) == 0:
            continue
        samples.append({"base_img_url": None, "steps": norm_steps})

    print(f"Mind2Web loaded {len(samples)} multi-turn items from {len(shard_files)} shards")
    return img_dir, samples


def load_miniwob_items(dataset_dir: str, split: str) -> Tuple[str, List[dict]]:
    """Load MiniWob++ from HF parquet shards (LucasThil/miniwob_plusplus_v2_raw format).
    Reference: https://huggingface.co/datasets/LucasThil/miniwob_plusplus_v2_raw/tree/main/data
    Maps episodes into multi-turn steps with per-step actions and targets.
    """
    import glob
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pyarrow is required to read MiniWob++ parquet. Install: pip install pyarrow")
    
    base_dir = os.path.join(dataset_dir, "MiniWob")
    data_dir = os.path.join(base_dir, "data")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"MiniWob++ data folder not found: {data_dir}. Download with: huggingface-cli download LucasThil/miniwob_plusplus_v2_raw --repo-type dataset --local-dir {base_dir}")
    
    # Read all train-*.parquet files
    shard_files = sorted(glob.glob(os.path.join(data_dir, "train-*.parquet")))
    if not shard_files:
        raise FileNotFoundError(f"No parquet shards in {data_dir}")
    
    raw = []
    for fp in shard_files:
        table = pq.read_table(fp)
        raw.extend(table.to_pylist())
    
    # Debug: print schema from first item
    if len(raw) > 0:
        print(f"DEBUG: MiniWob++ schema keys: {list(raw[0].keys())}")
        if "processed_states" in raw[0]:
            ps = raw[0]["processed_states"]
            if isinstance(ps, list) and len(ps) > 0:
                print(f"DEBUG: processed_states has {len(ps)} items, first keys: {list(ps[0].keys())}")
    
    # Images: render on-the-fly or skip; for now placeholder
    img_dir = os.path.join(base_dir, "screenshots")
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(base_dir, "images")

    samples: List[dict] = []
    for item in raw:
        # MiniWob++ actual schema: "task_name", "utterance", "reward", "raw_reward", "processed_states"
        # processed_states is a list of state dicts, each potentially containing action info
        task_name = item.get("task_name", "") or item.get("subdomain", "") or item.get("task", "")
        processed_states = item.get("processed_states", [])
        
        # New schema: processed_states contains trajectory steps with embedded action/state info
        # Old schema fallback: separate states and actions
        if processed_states and isinstance(processed_states, list) and len(processed_states) > 0:
            # Use processed_states (new schema)
            use_states = processed_states
        else:
            # Fallback to old schema
            states_old = item.get("states", [])
            actions_old = item.get("actions", [])
            if not actions_old or not states_old:
                continue
            # Merge into processed_states-like structure
            use_states = []
            for i, (st, act) in enumerate(zip(states_old, actions_old)):
                combined = {"state": st, "action": act}
                use_states.append(combined)
        
        if not use_states:
            continue
        
        # Build per-step instructions and targets
        norm_steps = []
        img_w, img_h = 160, 210  # default MiniWob viewport
        
        # Try to infer viewport from first state
        try:
            first_state = use_states[0].get("state") or use_states[0].get("tree") or use_states[0]
            if isinstance(first_state, dict):
                rw = first_state.get("width")
                rh = first_state.get("height")
                if isinstance(rw, (int, float)) and isinstance(rh, (int, float)) and rw > 0 and rh > 0:
                    img_w, img_h = float(rw), float(rh)
        except Exception:
            pass

        def pick_node_from_state(state_dict: dict) -> dict:
            # Prefer focused node, else clickable, else first
            try:
                tree = state_dict.get("tree") or state_dict.get("state") or state_dict
                q = [tree]
                candidates = []
                while q:
                    n = q.pop(0)
                    if isinstance(n, dict):
                        if n.get("focused") is True:
                            return n
                        candidates.append(n)
                        for c in n.get("children", []) or []:
                            q.append(c)
                # Clickable-like
                for n in candidates:
                    classes = str(n.get("classes", "")).lower()
                    if any(k in classes for k in ["button", "link", "click", "reply", "like"]):
                        return n
                return candidates[0] if candidates else {}
            except Exception:
                return {}

        for i, step_dict in enumerate(use_states):
            # Extract action info (may be embedded in step_dict or separate field)
            act = step_dict.get("action", {})
            act_type = act.get("action_type", "") or act.get("type", "") or step_dict.get("action_type", "") or "click"
            
            # Flexible coord extraction
            coords = None
            # Try action subdict first
            for kpair in (("coords", None), ("x", "y"), ("mouseX", "mouseY"), ("clickX", "clickY")):
                if kpair[1] is None:
                    v = act.get(kpair[0]) or step_dict.get(kpair[0])
                    if isinstance(v, (list, tuple)) and len(v) == 2:
                        coords = [v[0], v[1]]
                        break
                else:
                    x = act.get(kpair[0]) or step_dict.get(kpair[0])
                    y = act.get(kpair[1]) or step_dict.get(kpair[1])
                    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                        coords = [x, y]
                        break
            
            instr = f"{task_name} [{act_type}]"
            if coords is None:
                # Derive from state tree
                node = pick_node_from_state(step_dict)
                try:
                    left = float(node.get("left", 0.0))
                    top = float(node.get("top", 0.0))
                    width = float(node.get("width", 0.0))
                    height = float(node.get("height", 0.0))
                    coords = [left + width / 2.0, top + height / 2.0]
                except Exception:
                    pass
            
            if coords is None:
                continue
            
            try:
                px = min(1.0, max(0.0, float(coords[0]) / img_w))
                py = min(1.0, max(0.0, float(coords[1]) / img_h))
                pt = [px, py]
            except Exception:
                continue
            
            s_img = f"step_{i}.png"
            norm_steps.append({"instruction": instr, "point": pt, "img_url": s_img})
        
        if len(norm_steps) == 0:
            continue
        samples.append({"base_img_url": None, "steps": norm_steps})

    print(f"MiniWob++ loaded {len(samples)} multi-turn items from {len(shard_files)} shards")
    return img_dir, samples


def build_messages_for_turn(processor, instruction: str, img: Image.Image, min_pixels: int, max_pixels: int, history: Optional[List[dict]]) -> Tuple[str, dict]:
    img_dict = {"type": "image", "min_pixels": min_pixels, "max_pixels": max_pixels}
    if history is None:
        # first turn: base grounding template
        messages = grounding_to_qwen(instruction, img_dict, sample_io=0, user_prompt_random=False, xy_int=False, uniform_prompt=True)
        if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], dict):
            try:
                messages[0]["content"].append({"type":"text","text":"Output only [x, y] where both are between 0 and 1."})
            except Exception:
                pass
    else:
        # subsequent turns: add feedback and keep image
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "We will refine the clickable point prediction over multiple turns."},
                {"type": "image", "image": img, "min_pixels": min_pixels, "max_pixels": max_pixels},
                {"type": "text", "text": instruction},
            ]}
        ]
        for fb in history:
            messages.append(fb)
        # final user reminder
        messages.append({"role": "user", "content": [{"type": "text", "text": "Respond only with [x, y]."}]})

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt")
    return text, inputs


def parse_coord(output_text: str) -> Tuple[float, float]:
    try:
        xy = ast.literal_eval(output_text)
        if isinstance(xy, (list, tuple)) and len(xy) == 2:
            x, y = float(xy[0]), float(xy[1])
            x = min(1.0, max(0.0, x))
            y = min(1.0, max(0.0, y))
            return x, y
    except Exception:
        pass
    try:
        m = re.search(r"[\[\(]?\s*([-+]?[0-9]*\.?[0-9]+)\s*,\s*([-+]?[0-9]*\.?[0-9]+)\s*[\]\)]?", output_text)
        if m:
            x, y = float(m.group(1)), float(m.group(2))
            x = min(1.0, max(0.0, x))
            y = min(1.0, max(0.0, y))
            return x, y
    except Exception:
        pass
    return float("nan"), float("nan")


def l2_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    dx = (p1[0] - p2[0])
    dy = (p1[1] - p2[1])
    return math.sqrt(dx * dx + dy * dy)


def compute_turn_reward(pred_xy: Tuple[float, float], tgt_xy: Tuple[float, float], tau: float, alpha: float) -> float:
    if any(math.isnan(v) for v in pred_xy):
        return -1.0
    d = l2_distance(pred_xy, tgt_xy)
    r = (1.0 if d < tau else 0.0) - alpha * min(d, 0.5)
    return max(-1.0, min(1.0, r))


@torch.no_grad()
def evaluate_screenspot_subset_multiturn(processor, model, dataset_dir: str, limit: int, min_pixels: int, max_pixels: int, device: str, turns: int, eval_max_new_tokens: int) -> float:
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

        instruction = item["task"]
        history: List[dict] = None
        pred_xy = (float("nan"), float("nan"))
        for t in range(turns):
            text = processor.apply_chat_template([
                {"role":"user","content":[
                    {"type":"text","text":"Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1."},
                    {"type":"image","image":img,"min_pixels":min_pixels,"max_pixels":max_pixels},
                    {"type":"text","text":instruction},
                ]}
            ], tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt").to(device)
            try:
                out = model_unwrapped.generate(
                    **inputs,
                    max_new_tokens=int(max(8, eval_max_new_tokens)),
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    use_cache=False,
                )
                gen = out[:, inputs.input_ids.shape[1]:]
                pred_str = processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
                pred_xy = parse_coord(pred_str)
            except Exception:
                pass
            # simple history: append assistant feedback as text
            if not any(math.isnan(v) for v in pred_xy):
                fb_text = f"Assistant turn {t+1} predicted {pred_xy}"
            else:
                fb_text = f"Assistant turn {t+1} produced invalid output"
            msg = {"role": "assistant", "content": [{"type": "text", "text": fb_text}]}
            history = [] if history is None else history
            history.append(msg)
        x, y, w, h = item["bbox"]
        gt = [x / img_w, y / img_h, (x + w) / img_w, (y + h) / img_h]
        ok += 1 if (not any(math.isnan(v) for v in pred_xy)) and (gt[0] <= pred_xy[0] <= gt[2]) and (gt[1] <= pred_xy[1] <= gt[3]) else 0

    model_unwrapped.train()
    return ok / N


def generate_multiturn_trajectory(model, processor, device, instruction: str, image_path: str, tgt_xy: Tuple[float, float], args: MTRLArgs, osworld_steps: Optional[List[dict]] = None):
    img = Image.open(image_path).convert("RGB")
    min_pixels = args.min_visual_tokens * 28 * 28
    max_pixels = args.max_visual_tokens * 28 * 28

    history: Optional[List[dict]] = None
    turns = max(1, int(args.turns_per_traj))

    texts: List[str] = []
    processor_batches: List[dict] = []
    generated_token_ids: List[torch.Tensor] = []
    decoded_texts: List[str] = []
    rewards: List[float] = []

    model_unwrapped = model.module if hasattr(model, "module") else model

    try:
        param_dtype = next(model_unwrapped.parameters()).dtype
    except StopIteration:
        param_dtype = torch.float32

    for t in range(turns):
        # Select per-turn instruction/target/image for OSWorld, otherwise reuse single-turn inputs
        if osworld_steps and t < len(osworld_steps):
            instr_t = osworld_steps[t].get("instruction", instruction)
            tgt_t = osworld_steps[t].get("point", tgt_xy)
            img_path_t = osworld_steps[t].get("img_url") or image_path
            try:
                img_t = Image.open(img_path_t).convert("RGB") if os.path.exists(img_path_t) else img
            except Exception:
                img_t = img
        else:
            instr_t = instruction
            tgt_t = tgt_xy
            img_t = img

        text, inputs = build_messages_for_turn(processor, instr_t, img_t, min_pixels, max_pixels, history)
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
        if "pixel_values" in inputs and inputs["pixel_values"] is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=param_dtype)

        texts.append(text)
        processor_batches.append(inputs)

        with torch.no_grad():
            safe_temperature = float(max(args.temperature, 1e-4)) if args.do_sample else 1.0
            gen_kwargs = {
                "max_new_tokens": int(max(1, args.max_new_tokens)),
                "do_sample": bool(args.do_sample),
                "temperature": safe_temperature,
                "num_beams": int(max(1, args.num_beams)),
                "eos_token_id": processor.tokenizer.eos_token_id,
                "use_cache": False,
            }
            if args.top_p and args.top_p > 0.0:
                gen_kwargs["top_p"] = float(min(1.0, max(1e-6, args.top_p)))
            if args.top_k and args.top_k > 0:
                gen_kwargs["top_k"] = int(max(1, args.top_k))
            try:
                force_greedy = getattr(args, "_global_step", 0) < int(max(0, args.warmup_steps))
                if force_greedy:
                    out = model_unwrapped.generate(
                        **inputs,
                        max_new_tokens=int(max(1, args.max_new_tokens)),
                        do_sample=False,
                        num_beams=1,
                        eos_token_id=processor.tokenizer.eos_token_id,
                        use_cache=False,
                    )
                else:
                    out = model_unwrapped.generate(
                        **inputs,
                        **gen_kwargs,
                    )
            except Exception:
                out = model_unwrapped.generate(
                    **inputs,
                    max_new_tokens=int(max(1, args.max_new_tokens)),
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    use_cache=False,
                )

        gen = out[:, inputs["input_ids"].shape[1]:]
        generated_token_ids.append(gen)

        pred_text = processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        decoded_texts.append(pred_text)
        pred_xy = parse_coord(pred_text)
        # reward against per-turn target (OSWorld) or single target
        rew = compute_turn_reward(pred_xy, tgt_t, args.tau_success, args.alpha_dist)
        rewards.append(rew)

        if not any(math.isnan(v) for v in pred_xy):
            fb_text = f"Predicted {pred_xy}."
        else:
            fb_text = "Invalid output. Please output [x, y] in [0,1]."
        fb_msg = {"role": "assistant", "content": [{"type": "text", "text": fb_text}]}
        history = [] if history is None else history
        history.append(fb_msg)

    return texts, processor_batches, generated_token_ids, decoded_texts, rewards


def reinforce_step_multiturn(model, processor, device, batch, args: MTRLArgs):
    model.train()

    gamma = float(args.gamma)

    losses_per_item: List[torch.Tensor] = []
    entropies: List[float] = []
    kl_terms: List[float] = []

    sample_pairs = []
    reward_means: List[float] = []
    all_turn_rewards: List[float] = []

    has_ref = getattr(args, "_ref_logits_fn", None) is not None and args.kl_coef > 0.0

    for i, item in enumerate(batch):
        instruction = item[0]
        image_path = item[1]
        tgt_xy = item[2]
        osworld_steps = item[3] if len(item) > 3 else None
        texts, proc_batches, gen_ids, decoded, rewards = generate_multiturn_trajectory(
            model, processor, device, instruction, image_path, tgt_xy, args, osworld_steps=osworld_steps
        )
        # track mean per-turn reward for this trajectory
        try:
            reward_means.append(float(np.mean(rewards)) if len(rewards) else 0.0)
        except Exception:
            reward_means.append(0.0)
        # collect for global stats
        all_turn_rewards.extend(rewards)

        # discounted returns and advantages for this trajectory
        G: List[float] = [0.0 for _ in rewards]
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = float(rewards[t] + gamma * running)
            G[t] = running
        mean_G = float(np.mean(G)) if len(G) else 0.0
        std_G = float(np.std(G)) if len(G) else 0.0
        if std_G > 1e-6:
            traj_adv = [(g - mean_G) / std_G for g in G]
        else:
            traj_adv = [0.0 for _ in G]

        per_turn_losses: List[torch.Tensor] = []
        per_turn_entropy: List[float] = []
        per_turn_kl: List[float] = []

        for t in range(len(gen_ids)):
            inputs = proc_batches[t]
            gen_trimmed = gen_ids[t]

            input_ids_full = torch.cat([inputs["input_ids"], gen_trimmed], dim=1)
            attention_mask_full = (input_ids_full != processor.tokenizer.pad_token_id).long()
            prompt_len = inputs["input_ids"].shape[1]

            if torch.cuda.is_available():
                autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            else:
                autocast_ctx = torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False)
            with autocast_ctx:
                outputs = model(
                    input_ids=input_ids_full.to(device),
                    attention_mask=attention_mask_full.to(device),
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                    use_cache=False,
                    labels=None,
                )
                vocab = outputs.logits.size(-1)
                logits = outputs.logits[:, prompt_len - 1 : -1, :].contiguous()
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
                target = input_ids_full[:, prompt_len:].contiguous()
                nll = F.cross_entropy(logits.view(-1, vocab), target.view(-1), reduction="none")
                nll = nll.view(target.size(0), target.size(1))
                token_mask = (target != processor.tokenizer.pad_token_id).float()
                turn_nll = (nll * token_mask).sum(dim=1) / (token_mask.sum(dim=1) + 1e-6)

                log_probs = F.log_softmax(logits, dim=-1)
                probs = log_probs.exp()
                probs = torch.nan_to_num(probs, nan=0.0)
                token_entropy = -(probs * log_probs).sum(dim=-1)
                entropy_per_seq = (token_entropy * token_mask).sum(dim=1) / (token_mask.sum(dim=1) + 1e-6)
                per_turn_entropy.append(float(entropy_per_seq.mean().item()))

                kl_val = 0.0
                if has_ref:
                    with torch.no_grad():
                        ref_logits = args._ref_logits_fn(
                            input_ids=input_ids_full.to(device),
                            attention_mask=attention_mask_full.to(device),
                            pixel_values=inputs.get("pixel_values"),
                            image_grid_thw=inputs.get("image_grid_thw"),
                        )
                    ref_logits = ref_logits[:, prompt_len - 1 : -1, :].contiguous()
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                    kl_token = (probs * (log_probs - ref_log_probs)).sum(dim=-1)
                    kl_per_seq = (kl_token * token_mask).sum(dim=1) / (token_mask.sum(dim=1) + 1e-6)
                    kl_val = float(kl_per_seq.mean().item())
                per_turn_kl.append(kl_val)

            adv_t = torch.tensor(traj_adv[t], dtype=torch.float32, device=device)
            loss_t = (adv_t * turn_nll).mean()
            per_turn_losses.append(loss_t)

            # free per-turn tensors early
            del input_ids_full, attention_mask_full, target, logits, nll, token_mask, turn_nll
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        policy_loss_item = torch.stack(per_turn_losses).mean() if per_turn_losses else torch.tensor(0.0, device=device)
        losses_per_item.append(policy_loss_item)
        entropies.append(float(np.mean(per_turn_entropy)) if len(per_turn_entropy) else 0.0)
        kl_terms.append(float(np.mean(per_turn_kl)) if len(per_turn_kl) else 0.0)

        if i < 2:
            sample_pairs.append((instruction, " | ".join(decoded)))

        # free trajectory tensors
        del proc_batches, gen_ids
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    policy_loss = torch.stack(losses_per_item).mean() if len(losses_per_item) else torch.tensor(0.0, device=device)
    entropy_mean = float(np.mean(entropies)) if len(entropies) else 0.0
    kl_mean = float(np.mean(kl_terms)) if len(kl_terms) else 0.0

    loss = policy_loss - args.entropy_coef_start * torch.tensor(entropy_mean, device=device) + args.kl_coef * torch.tensor(kl_mean, device=device)
    if not torch.isfinite(loss):
        loss = torch.nan_to_num(policy_loss, nan=0.0, posinf=1e4, neginf=1e4)

    # Average of mean per-turn rewards across batch
    avg_reward = float(np.mean(reward_means)) if len(reward_means) else 0.0
    # Stats for logging
    if len(all_turn_rewards):
        rmin = float(np.min(all_turn_rewards))
        rmax = float(np.max(all_turn_rewards))
        succ_rate = float(np.mean([1.0 if r > 0.0 else 0.0 for r in all_turn_rewards]))
    else:
        rmin, rmax, succ_rate = 0.0, 0.0, 0.0

    return loss, avg_reward, entropy_mean, kl_mean, sample_pairs, rmin, rmax, succ_rate


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
    parser.add_argument("--tau_success", type=float, default=0.08)
    parser.add_argument("--tau_success_end", type=float, default=0.06)
    parser.add_argument("--alpha_dist", type=float, default=0.5)
    parser.add_argument("--alpha_dist_end", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=24)
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
    parser.add_argument("--min_visual_tokens", type=int, default=192)
    parser.add_argument("--max_visual_tokens", type=int, default=640)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--log_dir", type=str, default="./runs/rl-mt")
    parser.add_argument("--eval_subset_limit", type=int, default=1000)
    parser.add_argument("--eval_every_steps", type=int, default=200)
    parser.add_argument("--eval_subset_limit_train", type=int, default=200)
    parser.add_argument("--eval_subset_limit_epoch", type=int, default=1000)
    parser.add_argument("--full_eval_every_epochs", type=int, default=5)
    parser.add_argument("--eval_max_new_tokens", type=int, default=24)
    parser.add_argument("--eval_disable_mid", action="store_true")
    parser.add_argument("--save_every_epochs", type=int, default=10)
    parser.add_argument("--resume_from", type=str, default="")
    parser.add_argument("--save_optimizer", action="store_true")
    parser.add_argument("--eval_split", type=str, default="hf_test_full")
    parser.add_argument("--log_samples_every", type=int, default=100)
    parser.add_argument("--log_hist_every", type=int, default=100)
    parser.add_argument("--save_best", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--turns_per_traj", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    args_ns = parser.parse_args()

    args = MTRLArgs(
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
        alpha_dist_end=args_ns.alpha_dist_end,
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
        eval_subset_limit_train=args_ns.eval_subset_limit_train,
        eval_subset_limit_epoch=args_ns.eval_subset_limit_epoch,
        full_eval_every_epochs=args_ns.full_eval_every_epochs,
        eval_max_new_tokens=args_ns.eval_max_new_tokens,
        eval_disable_mid=args_ns.eval_disable_mid,
        save_every_epochs=args_ns.save_every_epochs,
        resume_from=args_ns.resume_from,
        save_optimizer=args_ns.save_optimizer,
        eval_split=args_ns.eval_split,
        warmup_steps=args_ns.warmup_steps,
        turns_per_traj=args_ns.turns_per_traj,
        gamma=args_ns.gamma,
    )

    set_seed(args.seed)
    global_rank = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    min_pixels = args.min_visual_tokens * 28 * 28
    max_pixels = args.max_visual_tokens * 28 * 28

    processor = None
    try:
        # Prefer fast processor which supports min/max pixel overrides
        processor = AutoProcessor.from_pretrained(
            args.model_id,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    except Exception as e:
        print(f"Processor load with min/max pixels failed ({e}); retrying with defaults.")
        processor = AutoProcessor.from_pretrained(args.model_id)
    if args.load_in_8bit:
        try:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                args.model_id,
                quantization_config=quantization_config,
                device_map="auto",
            )
        except Exception as e:
            print(f"8-bit load failed ({e}); falling back to non-quantized load.")
            args.load_in_8bit = False
            model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=torch_dtype)
            model.to(device)
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=torch_dtype)
        model.to(device)
    # Disable KV cache to reduce memory during training/inference loops
    try:
        model.config.use_cache = False
    except Exception:
        pass
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    optimizer = AdamW(model.parameters(), lr=args.lr)

    if args.kl_coef > 0.0:
        try:
            ref_id = args.ref_model_id if args.ref_model_id else args.model_id
            if args.ref_model_8bit:
                try:
                    ref_qconf = BitsAndBytesConfig(load_in_8bit=True)
                    ref_model = Qwen2VLForConditionalGeneration.from_pretrained(
                        ref_id,
                        quantization_config=ref_qconf,
                        device_map="auto",
                    )
                except Exception as e:
                    print(f"Ref 8-bit load failed ({e}); falling back to non-quantized ref.")
                    args.ref_model_8bit = False
                    ref_model = Qwen2VLForConditionalGeneration.from_pretrained(ref_id, torch_dtype=torch_dtype)
                    ref_model.to(device)
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

    start_epoch = 0
    global_step = 0
    resume_dir = None
    if args.resume_from:
        if os.path.isdir(args.resume_from) and not os.path.exists(os.path.join(args.resume_from, "pytorch_model.bin")):
            cand_path, cand_epoch = find_latest_epoch_checkpoint(args.resume_from)
            if cand_epoch >= 0:
                resume_dir = cand_path
                start_epoch = cand_epoch
        else:
            resume_dir = args.resume_from
    else:
        cand_path, cand_epoch = find_latest_epoch_checkpoint(os.getcwd())
        if cand_epoch >= 0:
            resume_dir = cand_path
            start_epoch = cand_epoch

    if resume_dir:
        try:
            if os.path.exists(os.path.join(resume_dir, "pytorch_model.bin")):
                sd = torch.load(os.path.join(resume_dir, "pytorch_model.bin"), map_location="cpu")
                model.load_state_dict(sd, strict=False)
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

    # Dataset loader selection (OSWorld / Mind2Web / MiniWob / shared grounding)
    ds_name = str(args.train_dataset).lower()
    is_osworld = ("osworld" in ds_name)
    is_mind2web = ("mind2web" in ds_name)
    is_miniwob = ("miniwob" in ds_name) or ("miniwob++" in ds_name)
    
    if is_osworld:
        img_dir, samples = load_osworld_items(args.dataset_dir, args.train_json)
    elif is_mind2web:
        img_dir, samples = load_mind2web_items(args.dataset_dir, args.train_json)
    elif is_miniwob:
        img_dir, samples = load_miniwob_items(args.dataset_dir, args.train_json)
    else:
        img_dir, samples = load_split_items(args.dataset_dir, args.train_dataset, args.train_json)
    if global_rank == 0:
        print(f"Loaded {len(samples)} samples from {args.train_dataset}/{args.train_json}")

    writer = None
    if global_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.log_dir)

    def sample_epoch_iterator(all_samples):
        idxs = list(range(len(all_samples)))
        random.shuffle(idxs)
        for i in idxs:
            yield all_samples[i]

    total_optim_steps = args.epochs * args.steps_per_epoch
    best_sr = -1.0

    if not hasattr(args, "_sched_init"):
        object.__setattr__(args, "_temp_start", float(args.temperature))
        object.__setattr__(args, "_entropy_start", float(args.entropy_coef_start))
        object.__setattr__(args, "_tau_start", float(args.tau_success))
        object.__setattr__(args, "_alpha_start", float(args.alpha_dist))
        object.__setattr__(args, "_sched_init", True)

    for epoch in range(start_epoch, args.epochs):
        running_loss = 0.0
        running_reward = 0.0
        running_entropy = 0.0
        running_kl = 0.0
        start = time.time()

        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch+1}/{args.epochs}")
        for _ in pbar:
            progress = (global_step + 1) / max(1, total_optim_steps)
            temperature_now = float(args._temp_start + (args.temperature_end - args._temp_start) * progress)
            object.__setattr__(args, "temperature", temperature_now)
            entropy_now = float(args._entropy_start + (args.entropy_coef_end - args._entropy_start) * progress)
            object.__setattr__(args, "entropy_coef_start", entropy_now)
            # reward schedules
            tau_now = float(args._tau_start + (args.tau_success_end - args._tau_start) * progress)
            alpha_now = float(args._alpha_start + (args.alpha_dist_end - args._alpha_start) * progress)
            object.__setattr__(args, "tau_success", tau_now)
            object.__setattr__(args, "alpha_dist", alpha_now)

            accum_loss = 0.0
            accum_reward = 0.0
            accum_entropy = 0.0
            accum_kl = 0.0
            # stats across micro-steps
            batch_reward_min = float("inf")
            batch_reward_max = float("-inf")
            batch_success_rate = []

            optimizer.zero_grad(set_to_none=True)
            for micro in range(max(1, args.grad_accum_steps)):
                batch_items = []
                for _ in range(args.batch_size):
                    if is_osworld or is_mind2web or is_miniwob:
                        item = random.choice(samples)
                        steps = item.get("steps", [])
                        if not steps:
                            continue
                        # trim to desired turns and ensure targets exist
                        use_steps = [s for s in steps if isinstance(s.get("point"), (list, tuple)) and len(s.get("point")) == 2]
                        if not use_steps:
                            continue
                        use_steps = use_steps[: max(1, int(args.turns_per_traj))]
                        base_img_url = item.get("base_img_url") or (use_steps[0].get("img_url") if use_steps else None)
                        image_path = os.path.join(img_dir, base_img_url) if base_img_url else ""
                        first_instr = use_steps[0].get("instruction", "")
                        first_pt = use_steps[0].get("point", [float("nan"), float("nan")])
                        # expand step image paths to absolute
                        steps_abs = []
                        for s in use_steps:
                            s_img = s.get("img_url")
                            s_abs = dict(s)
                            if s_img:
                                s_abs["img_url"] = os.path.join(img_dir, s_img)
                            steps_abs.append(s_abs)
                        batch_items.append((first_instr, image_path, (float(first_pt[0]), float(first_pt[1])), steps_abs))
                    else:
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
                        instruction = element["instruction"]
                        tgt_xy = (float(element["point"][0]), float(element["point"][1]))
                        batch_items.append((instruction, image_path, tgt_xy))

                if not batch_items:
                    continue

                loss_tensor, reward, ent, kl, sample_pairs, rmin, rmax, succ_rate = reinforce_step_multiturn(model, processor, device, batch_items, args)
                (loss_tensor / max(1, args.grad_accum_steps)).backward()
                accum_loss += float(loss_tensor.item())
                accum_reward += reward
                accum_entropy += ent
                accum_kl += kl
                batch_reward_min = min(batch_reward_min, rmin)
                batch_reward_max = max(batch_reward_max, rmax)
                batch_success_rate.append(succ_rate)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss = accum_loss / max(1, args.grad_accum_steps)
            reward = accum_reward / max(1, args.grad_accum_steps)
            ent = accum_entropy / max(1, args.grad_accum_steps)
            kl = accum_kl / max(1, args.grad_accum_steps)
            succ = float(np.mean(batch_success_rate)) if len(batch_success_rate) else 0.0

            running_loss += loss
            running_reward += reward
            running_entropy += ent
            running_kl += kl
            pbar.set_postfix({"loss": f"{loss:.4f}", "reward": f"{reward:.3f}", "H": f"{ent:.3f}", "succ": f"{succ:.2f}"})
            if writer:
                writer.add_scalar("train/loss", loss, global_step)
                writer.add_scalar("train/reward", reward, global_step)
                writer.add_scalar("train/reward_min", float(batch_reward_min if batch_reward_min != float("inf") else 0.0), global_step)
                writer.add_scalar("train/reward_max", float(batch_reward_max if batch_reward_max != float("-inf") else 0.0), global_step)
                writer.add_scalar("train/success_turn_rate", succ, global_step)
                writer.add_scalar("train/entropy", ent, global_step)
                if args.kl_coef > 0.0:
                    writer.add_scalar("train/kl", kl, global_step)
                try:
                    invalid_rate = 1.0 if reward <= -0.999 else 0.0
                    writer.add_scalar("train/invalid_parse_rate", invalid_rate, global_step)
                except Exception:
                    pass
                if args.log_samples_every > 0 and ((global_step + 1) % args.log_samples_every == 0):
                    try:
                        for i, (instr, out_texts) in enumerate(sample_pairs):
                            writer.add_text(f"train/sample_{i}", f"Instruction: {instr}\nTurns: {out_texts}", global_step)
                    except Exception:
                        pass

            if ((global_step + 1) % args.eval_every_steps == 0) and (not args.eval_disable_mid):
                sr = evaluate_screenspot_subset_multiturn(
                    processor, model, args.dataset_dir,
                    args.eval_subset_limit_train,
                    min_pixels, max_pixels, device,
                    max(1, int(args.turns_per_traj)),
                    int(max(8, args.eval_max_new_tokens))
                )
                if writer:
                    writer.add_scalar("eval/screenspot_subset_success", sr, global_step)
                if sr > best_sr:
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

            if writer and args.log_hist_every > 0 and ((global_step + 1) % args.log_hist_every == 0):
                writer.add_histogram("train/reward_hist_mean", np.array([reward], dtype=np.float32), global_step)

            global_step += 1
            try:
                object.__setattr__(args, "_global_step", int(global_step))
            except Exception:
                pass

        duration = time.time() - start
        print(f"Epoch {epoch+1} done in {duration:.1f}s | avg loss {running_loss/args.steps_per_epoch:.4f} | avg reward {running_reward/args.steps_per_epoch:.3f}")

        end_limit = args.eval_subset_limit_epoch if ((epoch + 1) % max(1, args.full_eval_every_epochs) == 0) else args.eval_subset_limit_train
        sr = evaluate_screenspot_subset_multiturn(
            processor, model, args.dataset_dir,
            end_limit,
            min_pixels, max_pixels, device,
            max(1, int(args.turns_per_traj)),
            int(max(8, args.eval_max_new_tokens))
        )
        if writer:
            writer.add_scalar("eval/screenspot_subset_success_epoch", sr, epoch)
        print(f"Epoch {epoch+1} eval subset success (multi-turn): {sr:.4f}")

        # Always save best at end of epoch
        if sr > best_sr:
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
            if args.save_optimizer:
                try:
                    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
                    with open(os.path.join(save_dir, "training_state.json"), "w") as f:
                        json.dump({"epoch": epoch + 1, "global_step": global_step}, f)
                except Exception as e:
                    print(f"Saving optimizer failed: {e}")
            print(f"Saved RL checkpoint to {save_dir}")


if __name__ == "__main__":
    main()
