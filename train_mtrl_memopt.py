#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Memory-optimized Multi-Turn RL (MTRL) trainer for ShowUI-2B (Qwen2VL) with PPO-Clip.

Key memory tactics:
- Rollout buffers store tensors on CPU (move to CUDA only when computing).
- Generation with use_cache toggled (default off) to reduce KV cache.
- Smaller default image token budgets + short outputs.
- Optional QLoRA (4-bit) + LoRA adapters (train only small adapter weights).
- Gradient checkpointing supported.

Fixes:
- Manual construction of Qwen2-VL processor (tokenizer + image processor) with
  size={'shortest_edge': ..., 'longest_edge': ...} to satisfy new transformers versions.
- Gated sampling flags (only when do_sample=True).
- dtype/device alignment for value head.
- StepBuf uses CPU storage and 1-D shapes where needed.
"""

import os, re, ast, math, time, json, random, gc
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from PIL import Image
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    Qwen2VLImageProcessor,
    Qwen2VLProcessor,
    Qwen2VLForConditionalGeneration,
    BitsAndBytesConfig,
)
from torch.utils.tensorboard import SummaryWriter

# Project utils (must exist in your repo)
from data.dset_shared_grounding import dataset_mapping
from data.template.shared_grounding import grounding_to_qwen  # not used directly here but kept for parity
from data.data_utils import IGNORE_INDEX  # noqa: F401 (kept for parity)

# Optional PEFT (for LoRA/QLoRA)
try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


# -----------------------------
# Args
# -----------------------------
@dataclass
class Args:
    dataset_dir: str
    train_dataset: str = "showui-desktop"
    train_json: str = "hf_train"
    eval_split: str = "hf_test_full"

    model_id: str = "showlab/ShowUI-2B"
    ref_model_id: str = ""

    # Quantization / adapters
    load_in_8bit: bool = False
    load_in_4bit: bool = False       # QLoRA path
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules_csv: str = "q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj"

    gradient_checkpointing: bool = False

    # Optim
    lr: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    # Multi-turn
    horizon: int = 3
    step_penalty: float = 0.01
    tau_success: float = 0.06
    tau_success_end: float = 0.04
    improvement_scale: float = 0.5
    clip_improvement: float = 0.15

    # Generation (short to save mem)
    max_new_tokens: int = 16
    do_sample: bool = True
    temperature: float = 0.7
    temperature_end: float = 0.5
    top_p: float = 0.9
    top_k: int = 0
    num_beams: int = 1
    warmup_steps: int = 100
    gen_use_cache: bool = False      # smaller memory by default

    # Visual token budgets (smaller)
    min_visual_tokens: int = 160
    max_visual_tokens: int = 640

    # PPO
    ppo_epochs: int = 2
    ppo_clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.001
    kl_coef: float = 0.0
    gae_gamma: float = 0.99
    gae_lambda: float = 0.95
    batch_size_episodes: int = 2
    minibatches: int = 2

    # Training schedule
    epochs: int = 1
    steps_per_epoch: int = 20
    seed: int = 42

    # Logging / saving
    log_dir: str = "./runs/mtrl_memopt"
    log_samples_every: int = 50
    eval_every: int = 0               # off by default to save mem
    eval_subset_limit: int = 200
    save_every_epochs: int = 1
    save_best: bool = False

    # Resume
    resume_from: str = ""
    save_optimizer: bool = True

    # Processor / tokenizer speed
    use_fast_processor: bool = False  # False keeps slow tokenizer (avoids behavior drift)


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def parse_coord(output_text: str) -> Tuple[float, float]:
    try:
        xy = ast.literal_eval(output_text)
        if isinstance(xy, (list, tuple)) and len(xy) == 2:
            x, y = float(xy[0]), float(xy[1])
            return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))
    except Exception:
        pass
    try:
        m = re.search(r"[\[\(]?\s*([-+]?[0-9]*\.?[0-9]+)\s*,\s*([-+]?[0-9]*\.?[0-9]+)\s*[\]\)]?", output_text)
        if m:
            x, y = float(m.group(1)), float(m.group(2))
            return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))
    except Exception:
        pass
    return float("nan"), float("nan")


def center_of_bbox_rel(bbox, img_w, img_h):
    x, y, w, h = bbox
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    return cx, cy


def l2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def in_bbox(pred_xy: Tuple[float, float], bbox_rel: Tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = bbox_rel
    return (x1 <= pred_xy[0] <= x2) and (y1 <= pred_xy[1] <= y2)


def dir_feedback(pred: Tuple[float, float], bbox_center: Tuple[float, float]) -> str:
    dx = bbox_center[0] - pred[0]
    dy = bbox_center[1] - pred[1]
    def bucket(v):
        av = abs(v)
        if av < 0.02: return "slightly"
        if av < 0.07: return "a bit"
        return "far"
    msg_x = "right" if dx > 0 else "left"
    msg_y = "down" if dy > 0 else "up"
    return f"Feedback: move {bucket(dx)} {msg_x} and {bucket(dy)} {msg_y}. Output only [x, y]."


def load_split_items(dataset_dir: str, dataset: str, split: str) -> Tuple[str, List[dict]]:
    base_image_dir = os.path.join(dataset_dir, dataset_mapping[dataset])
    meta_dir = os.path.join(base_image_dir, "metadata")
    img_dir = os.path.join(base_image_dir, "images")
    with open(os.path.join(meta_dir, f"{split}.json"), "r") as f:
        samples = json.load(f)
    return img_dir, samples


def make_messages(processor, img: Image.Image, instruction: str, history, min_pixels, max_pixels):
    content = [
        {"type": "text",
         "text": "Based on the screenshot, output only [x, y] in [0,1] for the described clickable element."},
        {"type": "image", "image": img, "min_pixels": min_pixels, "max_pixels": max_pixels},
        {"type": "text", "text": instruction},
    ]
    messages = [{"role": "user", "content": content}]
    for (assistant_text, feedback_text) in history:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_text}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": feedback_text}]})
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# -----------------------------
# Model wrapper (policy + value)
# -----------------------------
class PVModel(nn.Module):
    def __init__(self, base: Qwen2VLForConditionalGeneration, hidden_size: int):
        super().__init__()
        self.base = base
        ref = next(self.base.parameters())
        self.v_head = nn.Linear(hidden_size, 1, bias=True)
        self.v_head.to(device=ref.device, dtype=ref.dtype)

    def forward(self, *args, **kwargs):
        return self.base(*args, **kwargs)

    def value_from_hidden(self, hidden_states: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        B = hidden_states.size(0)
        gather = hidden_states[torch.arange(B, device=hidden_states.device), idx, :]
        gather = gather.to(self.v_head.weight.dtype)
        return self.v_head(gather).squeeze(-1)


# -----------------------------
# Rollout buffers (CPU storage)
# -----------------------------
class StepBuf:
    __slots__ = (
        "input_ids_full", "attention_mask_full",  # torch.Long on CPU
        "pixel_values_cpu",                       # torch.Float on CPU or None
        "prompt_len",                             # int
        "image_grid_thw",                         # metadata (small)
        "action_ids",                             # torch.Long on CPU
        "old_logp", "value", "reward", "done", "entropy"  # 1D tensors on CPU
    )
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


# -----------------------------
# Evaluation (no grads)
# -----------------------------
@torch.no_grad()
def evaluate_subset(processor, model, dataset_dir: str, limit: int, device: str, min_pixels: int, max_pixels: int) -> float:
    meta_path = os.path.join(dataset_dir, "ScreenSpot", "metadata", "hf_test_full.json")
    if not os.path.exists(meta_path):
        return 0.0
    try:
        with open(meta_path, "r") as f:
            items = json.load(f)
    except Exception:
        return 0.0
    N = min(limit, len(items)) if limit and limit > 0 else len(items)
    if N == 0: return 0.0
    ok = 0
    m = model.module if hasattr(model, "module") else model
    m.eval()
    for i in range(N):
        item = items[i]
        img_path = os.path.join(dataset_dir, "ScreenSpot", "images", item["img_url"])
        if not os.path.exists(img_path): continue
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = item.get("img_size", img.size)
        messages = [{"role":"user","content":[
            {"type":"text","text":"Return only [x, y] in [0,1]."},
            {"type":"image","image":img,"min_pixels":min_pixels,"max_pixels":max_pixels},
            {"type":"text","text":item["task"]},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt").to(device)
        if "pixel_values" in inputs and inputs["pixel_values"] is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(next(m.base.parameters()).dtype)
        out = m.base.generate(
            **inputs, max_new_tokens=16, do_sample=False, num_beams=1,
            eos_token_id=processor.tokenizer.eos_token_id, use_cache=False
        )
        gen = out[:, inputs["input_ids"].shape[1]:]
        pred_str = processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        pred = parse_coord(pred_str)
        x, y, w, h = item["bbox"]
        gt = [x / img_w, y / img_h, (x + w) / img_w, (y + h) / img_h]
        ok += 1 if (not any(math.isnan(v) for v in pred)) and (gt[0] <= pred[0] <= gt[2]) and (gt[1] <= pred[1] <= gt[3]) else 0
    m.train()
    return ok / N


# -----------------------------
# Trainer
# -----------------------------
def main():
    import argparse
    p = argparse.ArgumentParser()
    # Dataset / model
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--train_dataset", type=str, default="showui-desktop")
    p.add_argument("--train_json", type=str, default="hf_train")
    p.add_argument("--eval_split", type=str, default="hf_test_full")
    p.add_argument("--model_id", type=str, default="showlab/ShowUI-2B")
    p.add_argument("--ref_model_id", type=str, default="")

    # Quant/adapters
    p.add_argument("--load_in_8bit", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules_csv", type=str, default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj")

    p.add_argument("--gradient_checkpointing", action="store_true")

    # Multi-turn / rewards
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--step_penalty", type=float, default=0.01)
    p.add_argument("--tau_success", type=float, default=0.06)
    p.add_argument("--tau_success_end", type=float, default=0.04)
    p.add_argument("--improvement_scale", type=float, default=0.5)
    p.add_argument("--clip_improvement", type=float, default=0.15)

    # Gen
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--temperature_end", type=float, default=0.5)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--gen_use_cache", action="store_true")

    # Visual res
    p.add_argument("--min_visual_tokens", type=int, default=160)
    p.add_argument("--max_visual_tokens", type=int, default=640)

    # PPO / opt
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--ppo_epochs", type=int, default=2)
    p.add_argument("--ppo_clip", type=float, default=0.2)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--ent_coef", type=float, default=0.001)
    p.add_argument("--kl_coef", type=float, default=0.0)
    p.add_argument("--gae_gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--batch_size_episodes", type=int, default=2)
    p.add_argument("--minibatches", type=int, default=2)

    # Schedule
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--steps_per_epoch", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)

    # Logging
    p.add_argument("--log_dir", type=str, default="./runs/mtrl_memopt")
    p.add_argument("--log_samples_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=0)
    p.add_argument("--eval_subset_limit", type=int, default=200)
    p.add_argument("--save_every_epochs", type=int, default=1)
    p.add_argument("--save_best", action="store_true")

    # Resume
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--save_optimizer", action="store_true")

    # Processor / tokenizer speed
    p.add_argument("--use_fast_processor", action="store_true")

    args = Args(**vars(p.parse_args()))
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Speed flags
    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    # --- Build processor manually (fix for newer transformers) ---
    # Map token budgets (tokens ~ (edge/28)^2) to edges; clamp to >=224 px
    def _edge_from_tokens(n_tokens: int) -> int:
        e = 28 * math.ceil(math.sqrt(max(1, n_tokens)))
        return max(224, int(e))

    shortest_edge = _edge_from_tokens(args.min_visual_tokens)
    longest_edge  = _edge_from_tokens(args.max_visual_tokens)

    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=args.use_fast_processor)
    # pad token guard (some Qwen configs don't set pad)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    img_proc = Qwen2VLImageProcessor(
        size={"shortest_edge": shortest_edge, "longest_edge": longest_edge}
    )
    processor = Qwen2VLProcessor(image_processor=img_proc, tokenizer=tok)

    # Also keep these numbers to pass in messages (used by Qwen2VL for tiling hints)
    min_pixels = args.min_visual_tokens * 28 * 28
    max_pixels = args.max_visual_tokens * 28 * 28

    # --- Load base (with quant if requested) ---
    peft_used = False
    if args.load_in_4bit:
        qconf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            bnb_4bit_use_double_quant=True
        )
        base = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_id, quantization_config=qconf, device_map="auto"
        )
        if args.use_lora and PEFT_AVAILABLE:
            base = prepare_model_for_kbit_training(base)
            target_modules = [m.strip() for m in args.target_modules_csv.split(",") if m.strip()]
            lconf = LoraConfig(
                r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                bias="none", task_type="CAUSAL_LM", target_modules=target_modules
            )
            base = get_peft_model(base, lconf); peft_used = True
        hidden = base.config.hidden_size
    elif args.load_in_8bit:
        qconf = BitsAndBytesConfig(load_in_8bit=True)
        base = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_id, quantization_config=qconf, device_map="auto"
        )
        if args.use_lora and PEFT_AVAILABLE:
            base = prepare_model_for_kbit_training(base)
            target_modules = [m.strip() for m in args.target_modules_csv.split(",") if m.strip()]
            lconf = LoraConfig(
                r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                bias="none", task_type="CAUSAL_LM", target_modules=target_modules
            )
            base = get_peft_model(base, lconf); peft_used = True
        hidden = base.config.hidden_size
    else:
        base = Qwen2VLForConditionalGeneration.from_pretrained(args.model_id, dtype=model_dtype)
        base.to(device)
        if args.use_lora and PEFT_AVAILABLE:
            target_modules = [m.strip() for m in args.target_modules_csv.split(",") if m.strip()]
            lconf = LoraConfig(
                r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                bias="none", task_type="CAUSAL_LM", target_modules=target_modules
            )
            base = get_peft_model(base, lconf); peft_used = True
        hidden = base.config.hidden_size

    if args.gradient_checkpointing:
        try: base.gradient_checkpointing_enable()
        except Exception: pass

    model = PVModel(base, hidden)
    if not (args.load_in_4bit or args.load_in_8bit):
        model.to(device)

    # Optimizer (only train adapters + v_head if LoRA)
    train_params = list(model.v_head.parameters()) + [p for p in model.base.parameters() if p.requires_grad]
    optimizer = AdamW(train_params, lr=args.lr, weight_decay=args.weight_decay)

    # Data
    img_dir, samples = load_split_items(args.dataset_dir, args.train_dataset, args.train_json)
    print(f"Loaded {len(samples)} samples from {args.train_dataset}/{args.train_json}")

    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    total_updates = args.epochs * args.steps_per_epoch
    temp_start = float(args.temperature)
    tau_start = float(args.tau_success)
    global_step = 0
    best_sr = -1.0

    def sample_item_with_element():
        item = None
        while item is None:
            cand = random.choice(samples)
            if cand.get("element"): item = cand
        return item

    def rollout_batch():
        """
        Collect episodes; store buffers on CPU to avoid VRAM spikes.
        """
        bufs: List[StepBuf] = []

        for _ in range(args.batch_size_episodes):
            item = sample_item_with_element()
            image_path = os.path.join(img_dir, item["img_url"])
            img = Image.open(image_path).convert("RGB")
            img_w, img_h = item.get("img_size", img.size)
            element = random.choice(item["element"])
            instruction = element["instruction"]
            x, y, w, h = element["bbox"]
            bbox_rel = (x / img_w, y / img_h, (x + w) / img_w, (y + h) / img_h)
            center_rel = center_of_bbox_rel(element["bbox"], img_w, img_h)

            history: List[Tuple[str, str]] = []
            prev_dist = None

            for t in range(args.horizon):
                text = make_messages(processor, img, instruction, history, min_pixels, max_pixels)
                inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt")

                # Move minimal tensors to device; keep no-long-lived CUDA tensors
                for k in list(inputs.keys()):
                    if isinstance(inputs[k], torch.Tensor):
                        inputs[k] = inputs[k].to(device)
                prompt_len = inputs["input_ids"].shape[1]

                progress = (global_step + 1) / max(1, total_updates)
                temperature_now = temp_start + (args.temperature_end - temp_start) * progress
                tau_now = tau_start + (args.tau_success_end - tau_start) * progress
                do_sample = args.do_sample and (global_step >= args.warmup_steps)

                gen_kwargs = dict(
                    max_new_tokens=args.max_new_tokens,
                    do_sample=do_sample,
                    num_beams=args.num_beams,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    use_cache=args.gen_use_cache,
                )
                if do_sample:
                    gen_kwargs["temperature"] = max(1e-4, float(temperature_now))
                    if args.top_p and args.top_p > 0: gen_kwargs["top_p"] = float(args.top_p)
                    if args.top_k and args.top_k > 0: gen_kwargs["top_k"] = int(args.top_k)

                # dtype align
                if "pixel_values" in inputs and inputs["pixel_values"] is not None:
                    inputs["pixel_values"] = inputs["pixel_values"].to(next(model.base.parameters()).dtype)

                with torch.no_grad():
                    gen_out = model.base.generate(**inputs, **gen_kwargs)

                action_ids = gen_out[:, prompt_len:]  # [1, A]
                decoded = processor.batch_decode(action_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                pred = parse_coord(decoded)

                # Reward
                if any(math.isnan(v) for v in pred):
                    cur_dist = 1.0
                    success = False
                    rew = -0.5
                else:
                    cur_dist = l2(pred, center_rel)
                    success = in_bbox(pred, bbox_rel) or (cur_dist < tau_now)
                    if success:
                        rew = 1.0
                    else:
                        if prev_dist is None:
                            rew = -cur_dist
                        else:
                            improv = max(-args.clip_improvement, min(args.clip_improvement, (prev_dist - cur_dist)))
                            rew = args.improvement_scale * improv - 0.1 * cur_dist
                rew -= args.step_penalty

                # Build full input for scoring (and immediately move to CPU)
                input_ids_full = torch.cat([inputs["input_ids"], action_ids], dim=1).cpu()
                attention_mask_full = (input_ids_full != processor.tokenizer.pad_token_id).long().cpu()
                pixel_values_cpu = inputs.get("pixel_values", None)
                if pixel_values_cpu is not None:
                    pixel_values_cpu = pixel_values_cpu.detach().cpu()

                # Compute old logp/value/entropy (single-sample) then move to CPU
                with torch.no_grad():
                    out = model.forward(
                        input_ids=input_ids_full.to(device),
                        attention_mask=attention_mask_full.to(device),
                        pixel_values=pixel_values_cpu.to(device) if pixel_values_cpu is not None else None,
                        image_grid_thw=inputs.get("image_grid_thw"),
                        output_hidden_states=True,
                    )
                    logits = out.logits[:, :-1, :]
                    hidden = out.hidden_states[-1]
                    alen = action_ids.size(1)
                    logits_gen = logits[0, prompt_len-1: prompt_len-1+alen, :]
                    action_ids_dev = action_ids[0].to(device)
                    log_probs = F.log_softmax(logits_gen, dim=-1)
                    probs = log_probs.exp()
                    tok_logp = log_probs.gather(1, action_ids_dev.view(-1,1)).squeeze(1)
                    seq_logp = tok_logp.sum().detach().cpu().view(1)
                    tok_ent = (-(probs * log_probs).sum(dim=-1).mean()).detach().cpu().view(1)
                    v = model.value_from_hidden(hidden, torch.tensor([prompt_len-1], device=hidden.device)).detach().cpu().view(1)

                bufs.append(StepBuf(
                    input_ids_full=input_ids_full,                      # CPU
                    attention_mask_full=attention_mask_full,            # CPU
                    pixel_values_cpu=pixel_values_cpu,                  # CPU or None
                    image_grid_thw=inputs.get("image_grid_thw"),
                    prompt_len=prompt_len,
                    action_ids=action_ids[0].detach().cpu(),            # CPU
                    old_logp=seq_logp, value=v,
                    reward=torch.tensor([float(rew)], dtype=torch.float32),
                    done=torch.tensor([1.0 if success or t == args.horizon - 1 else 0.0], dtype=torch.float32),
                    entropy=tok_ent,
                ))

                # Free CUDA asap
                del out, logits, hidden, logits_gen, action_ids_dev, log_probs, probs, tok_logp
                torch.cuda.empty_cache()

                feedback = "Success. Stop." if success else dir_feedback(pred, center_rel)
                history.append((decoded, feedback))
                prev_dist = cur_dist
                if success:
                    break

            img.close()

        torch.cuda.empty_cache(); gc.collect()
        return bufs

    def ppo_update(bufs: List[StepBuf]):
        # Build flat tensors (CPU), GAE on CPU, then per-sample GPU forwards
        rewards = torch.cat([b.reward for b in bufs], dim=0).view(-1)     # CPU [N]
        dones   = torch.cat([b.done   for b in bufs], dim=0).view(-1)     # CPU [N]
        values  = torch.cat([b.value  for b in bufs], dim=0).view(-1)     # CPU [N]
        entropies = torch.cat([b.entropy for b in bufs], dim=0).view(-1)  # CPU [N]

        advantages = torch.zeros_like(rewards)
        lastgaelam = 0.0; next_value = 0.0
        for t in reversed(range(rewards.size(0))):
            mask = 1.0 - float(dones[t].item())
            delta = rewards[t].item() + args.gae_gamma * next_value * mask - values[t].item()
            lastgaelam = delta + args.gae_gamma * args.gae_lambda * mask * lastgaelam
            advantages[t] = lastgaelam
            next_value = values[t].item()
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        old_logps = torch.cat([b.old_logp for b in bufs], dim=0).view(-1)  # CPU [N]

        N = len(bufs)
        idxs = np.arange(N)
        mb_size = max(1, N // args.minibatches)

        policy_losses, value_losses, ent_losses, kl_losses = [], [], [], []

        for _ in range(args.ppo_epochs):
            np.random.shuffle(idxs)
            for mb_start in range(0, N, mb_size):
                mb_idx = idxs[mb_start: mb_start + mb_size]
                if len(mb_idx) == 0: continue

                mb_adv = advantages[mb_idx].to(device)
                mb_ret = returns[mb_idx].to(device)
                mb_old = old_logps[mb_idx].to(device)

                new_logps, new_vals, new_ents = [], [], []
                kl_term_list = []

                for i in mb_idx:
                    b = bufs[i]
                    input_ids_full = b.input_ids_full.to(device, non_blocking=True)
                    attention_mask_full = b.attention_mask_full.to(device, non_blocking=True)
                    pixel_values = b.pixel_values_cpu.to(device) if b.pixel_values_cpu is not None else None
                    prompt_len = int(b.prompt_len)
                    action_ids = b.action_ids.to(device)

                    out = model.forward(
                        input_ids=input_ids_full,
                        attention_mask=attention_mask_full,
                        pixel_values=pixel_values,
                        image_grid_thw=b.image_grid_thw,
                        output_hidden_states=True,
                    )
                    logits = out.logits[:, :-1, :]
                    hidden = out.hidden_states[-1]
                    alen = action_ids.size(0)

                    logits_gen = logits[0, prompt_len-1: prompt_len-1+alen, :]
                    log_probs = F.log_softmax(logits_gen, dim=-1)
                    probs = log_probs.exp()
                    tok_logp = log_probs.gather(1, action_ids.view(-1,1)).squeeze(1)
                    seq_logp = tok_logp.sum()

                    tok_ent = -(probs * log_probs).sum(dim=-1).mean()

                    v = model.value_from_hidden(hidden, torch.tensor([prompt_len-1], device=hidden.device))

                    new_logps.append(seq_logp)
                    new_vals.append(v.squeeze(0))
                    new_ents.append(tok_ent)

                    # Free per-sample quickly
                    del out, logits, hidden, logits_gen, log_probs, probs, tok_logp, v
                    torch.cuda.empty_cache()

                new_logps = torch.stack(new_logps).to(device)
                new_vals  = torch.stack(new_vals).to(device)
                new_ents  = torch.stack(new_ents).to(device)
                kl_loss = torch.tensor(0.0, device=device)

                ratio = torch.exp(new_logps - mb_old)
                clipped = torch.clamp(ratio, 1.0 - args.ppo_clip, 1.0 + args.ppo_clip) * mb_adv
                policy_loss = -(torch.min(ratio * mb_adv, clipped)).mean()

                value_loss = F.mse_loss(new_vals, mb_ret)
                ent_loss = -new_ents.mean()

                loss = policy_loss + args.vf_coef * value_loss + args.ent_coef * ent_loss + args.kl_coef * kl_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                ent_losses.append(new_ents.mean().item())
                kl_losses.append(float(kl_loss.item()))

                del new_logps, new_vals, new_ents, kl_loss
                torch.cuda.empty_cache()

        return (
            float(np.mean(policy_losses)) if policy_losses else 0.0,
            float(np.mean(value_losses)) if value_losses else 0.0,
            float(np.mean(ent_losses)) if ent_losses else 0.0,
            float(np.mean(kl_losses)) if kl_losses else 0.0,
        )

    # --- Training loop ---
    for epoch in range(args.epochs):
        model.train()
        epoch_pl, epoch_vl, epoch_ent, epoch_kl = [], [], [], []
        pbar = tqdm(range(args.steps_per_epoch), desc=f"[MTRL-MEM] Epoch {epoch+1}/{args.epochs}")

        for _ in pbar:
            bufs = rollout_batch()
            pl, vl, ent, kl = ppo_update(bufs)
            epoch_pl.append(pl); epoch_vl.append(vl); epoch_ent.append(ent); epoch_kl.append(kl)

            writer.add_scalar("train/policy_loss", pl, global_step)
            writer.add_scalar("train/value_loss", vl, global_step)
            writer.add_scalar("train/entropy", ent, global_step)

            if args.log_samples_every and (global_step % args.log_samples_every == 0):
                writer.add_text("train/note", "Mem-optimized PPO step complete.", global_step)

            if args.eval_every and (global_step % args.eval_every == 0):
                sr = evaluate_subset(processor, model, args.dataset_dir, args.eval_subset_limit, device, min_pixels, max_pixels)
                writer.add_scalar("eval/screenspot_subset_success", sr, global_step)
                pbar.set_postfix({"SR": f"{sr:.3f}"})

            global_step += 1

            del bufs
            torch.cuda.empty_cache()
            gc.collect()

        writer.add_scalar("epoch/policy_loss", float(np.mean(epoch_pl)), epoch)
        writer.add_scalar("epoch/value_loss", float(np.mean(epoch_vl)), epoch)
        writer.add_scalar("epoch/entropy", float(np.mean(epoch_ent)), epoch)

        if args.eval_every == 0:
            sr = evaluate_subset(processor, model, args.dataset_dir, args.eval_subset_limit, device, min_pixels, max_pixels)
            print(f"[MTRL-MEM] Epoch {epoch+1} subset SR: {sr:.4f}")
            writer.add_scalar("eval/screenspot_subset_success_epoch", sr, epoch)

        if (epoch + 1) % args.save_every_epochs == 0:
            save_dir = os.path.join(os.getcwd(), f"mtrl_mem_ckpt_epoch{epoch+1}")
            os.makedirs(save_dir, exist_ok=True)
            try:
                if peft_used:
                    # Save adapters (and base config)
                    model.base.save_pretrained(save_dir)
                else:
                    torch.save(model.base.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
                    model.base.config.to_json_file(os.path.join(save_dir, "config.json"))
            except Exception as e:
                print(f"[save] {e}")
            # Save processor components
            try:
                processor.save_pretrained(save_dir)
            except Exception:
                tok.save_pretrained(save_dir)
                img_proc.save_pretrained(save_dir)
            if args.save_optimizer:
                try:
                    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
                    with open(os.path.join(save_dir, "training_state.json"), "w") as f:
                        json.dump({"epoch": epoch + 1, "global_step": global_step}, f)
                except Exception as e:
                    print(f"[save_opt] {e}")

        torch.cuda.empty_cache(); gc.collect()

    print("Done.")


if __name__ == "__main__":
    main()
