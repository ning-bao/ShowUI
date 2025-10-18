#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-Turn RL (MTRL) trainer for ShowUI-2B (Qwen2VL) with PPO-Clip.

Key features:
- Multi-turn episodes with synthetic directional feedback (offline from bbox).
- Shaped rewards (success, improvement, step penalty; curriculum on tau).
- PPO-Clip with GAE, value head on top of the base model.
- Optional KL tether to a reference policy (set --kl_coef > 0).
- Robust dtype/device handling (no BF16/FP32 mismatches).
- Greedy warmup; sampling args passed only when do_sample=True.
- Evaluation on ScreenSpot subset (if present).

Run (smoke test):
  python train_mtrl.py \
    --dataset_dir "$DATA_DIR" \
    --train_dataset showui-desktop \
    --train_json hf_train \
    --epochs 1 --steps_per_epoch 5 \
    --batch_size_episodes 2 --minibatches 2 \
    --horizon 3 --improvement_scale 0.6 \
    --temperature 0.7 --temperature_end 0.6 \
    --tau_success 0.06 --tau_success_end 0.05
"""

import os, re, ast, math, time, json, random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from torch.utils.tensorboard import SummaryWriter

# === Project utilities (must exist in your repo) ===
from data.dset_shared_grounding import dataset_mapping
from data.template.shared_grounding import grounding_to_qwen
from data.data_utils import IGNORE_INDEX


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
    ref_model_id: str = ""           # default: same as model_id
    load_in_8bit: bool = False
    gradient_checkpointing: bool = False

    lr: float = 5e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    # Multi-turn
    horizon: int = 4                  # max steps per episode
    step_penalty: float = 0.01        # small negative each step
    tau_success: float = 0.06         # success if inside bbox or close to center
    tau_success_end: float = 0.04     # tighten during training
    improvement_scale: float = 0.5    # weight for distance improvement reward
    clip_improvement: float = 0.15    # clip per-step improvement bonus

    # Generation
    max_new_tokens: int = 32
    do_sample: bool = True
    temperature: float = 0.7
    temperature_end: float = 0.5
    top_p: float = 0.9
    top_k: int = 0
    num_beams: int = 1
    warmup_steps: int = 300           # greedy warmup

    # Visual resolution
    min_visual_tokens: int = 256
    max_visual_tokens: int = 1344

    # PPO
    ppo_epochs: int = 2
    ppo_clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.001
    kl_coef: float = 0.0              # set >0 to tether to ref policy
    gae_gamma: float = 0.99
    gae_lambda: float = 0.95
    batch_size_episodes: int = 8      # episodes per rollout batch
    minibatches: int = 4              # PPO SGD splits

    # Training schedule
    epochs: int = 3
    steps_per_epoch: int = 200        # PPO updates per epoch (each does one rollout batch)
    seed: int = 42

    # Logging / saving
    log_dir: str = "./runs/mtrl"
    log_samples_every: int = 100
    eval_every: int = 200
    eval_subset_limit: int = 200
    save_every_epochs: int = 1
    save_best: bool = True

    # Resume
    resume_from: str = ""
    save_optimizer: bool = True

    # Quant ref
    ref_model_8bit: bool = True


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split_items(dataset_dir: str, dataset: str, split: str) -> Tuple[str, List[dict]]:
    base_image_dir = os.path.join(dataset_dir, dataset_mapping[dataset])
    meta_dir = os.path.join(base_image_dir, "metadata")
    img_dir = os.path.join(base_image_dir, "images")
    with open(os.path.join(meta_dir, f"{split}.json"), "r") as f:
        samples = json.load(f)
    return img_dir, samples


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


# -----------------------------
# Conversation builder (MTRL)
# -----------------------------
def make_messages(processor, img: Image.Image, instruction: str, history: List[Tuple[str, str]], min_pixels: int, max_pixels: int):
    """
    history: list of (assistant_text, user_feedback_text) for previous turns.
    Image is included once in the first user turn.
    """
    content = [
        {"type": "text",
         "text": "Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] in [0,1]. Return only [x, y]."},
        {"type": "image", "image": img, "min_pixels": min_pixels, "max_pixels": max_pixels},
        {"type": "text", "text": instruction},
    ]
    messages = [{"role": "user", "content": content}]
    for (assistant_text, feedback_text) in history:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_text}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": feedback_text}]})
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text


# -----------------------------
# Model wrapper (policy + value)
# -----------------------------
class PVModel(nn.Module):
    def __init__(self, base: Qwen2VLForConditionalGeneration, hidden_size: int):
        super().__init__()
        self.base = base
        ref = next(self.base.parameters())
        self.v_head = nn.Linear(hidden_size, 1, bias=True)
        # align value head with base dtype/device
        self.v_head.to(device=ref.device, dtype=ref.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
    ):
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=output_hidden_states,
            use_cache=False,
        )
        return out

    def value_from_hidden(self, hidden_states: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, T, H], idx: [B] (positions to read state value)
        """
        B = hidden_states.size(0)
        gather = hidden_states[torch.arange(B, device=hidden_states.device), idx, :]  # [B, H]
        gather = gather.to(self.v_head.weight.dtype)
        return self.v_head(gather).squeeze(-1)  # [B]


# -----------------------------
# Rollout buffers
# -----------------------------
class StepBuf:
    __slots__ = (
        "input_ids_full", "attention_mask_full", "pixel_values", "image_grid_thw",
        "prompt_len", "action_ids", "old_logp", "value", "reward", "done", "entropy"
    )
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


# -----------------------------
# Evaluation
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
        messages = [
            {"role":"user","content":[
                {"type":"text","text":"Return only [x, y] in [0,1]."},
                {"type":"image","image":img,"min_pixels":min_pixels,"max_pixels":max_pixels},
                {"type":"text","text":item["task"]},
            ]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt").to(device)
        # dtype align
        if "pixel_values" in inputs and inputs["pixel_values"] is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(next(m.base.parameters()).dtype)
        out = m.base.generate(
            **inputs, max_new_tokens=32, do_sample=False, num_beams=1,
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
# Single-sample logp/value (robust to pixel shapes)
# -----------------------------
@torch.no_grad()
def compute_logprob_value_entropy_single(model: PVModel,
                                         input_ids_full: torch.Tensor,
                                         attention_mask_full: torch.Tensor,
                                         pixel_values: Optional[torch.Tensor],
                                         image_grid_thw: Optional[torch.Tensor],
                                         prompt_len: int,
                                         action_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (seq_logp, value, entropy) for one sample.
    """
    out = model.forward(
        input_ids=input_ids_full,
        attention_mask=attention_mask_full,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        output_hidden_states=True,
    )
    logits = out.logits[:, :-1, :]           # [1, T-1, V]
    hidden = out.hidden_states[-1]           # [1, T, H]

    pl = int(prompt_len)
    alen = action_ids.size(0)
    logits_gen = logits[0, pl-1: pl-1+alen, :]              # [A, V]
    target_gen = action_ids.view(-1).to(logits_gen.device)  # [A]
    log_probs = F.log_softmax(logits_gen, dim=-1)
    probs = log_probs.exp()
    tok_logp = log_probs.gather(1, target_gen.view(-1, 1)).squeeze(1)  # [A]
    seq_logp = tok_logp.sum()                                          # scalar

    tok_ent = -(probs * log_probs).sum(dim=-1).mean()                  # scalar

    v = model.value_from_hidden(hidden, torch.tensor([pl-1], device=hidden.device))  # [1]
    return seq_logp.detach(), v.detach().squeeze(0), tok_ent.detach()


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
    p.add_argument("--load_in_8bit", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")

    # Multi-turn / rewards
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--step_penalty", type=float, default=0.01)
    p.add_argument("--tau_success", type=float, default=0.06)
    p.add_argument("--tau_success_end", type=float, default=0.04)
    p.add_argument("--improvement_scale", type=float, default=0.5)
    p.add_argument("--clip_improvement", type=float, default=0.15)

    # Gen
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--temperature_end", type=float, default=0.5)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--warmup_steps", type=int, default=300)

    # Visual res
    p.add_argument("--min_visual_tokens", type=int, default=256)
    p.add_argument("--max_visual_tokens", type=int, default=1344)

    # PPO / opt
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--ppo_epochs", type=int, default=2)
    p.add_argument("--ppo_clip", type=float, default=0.2)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--ent_coef", type=float, default=0.001)
    p.add_argument("--kl_coef", type=float, default=0.0)
    p.add_argument("--gae_gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--batch_size_episodes", type=int, default=8)
    p.add_argument("--minibatches", type=int, default=4)

    # Schedule
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--steps_per_epoch", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)

    # Logging
    p.add_argument("--log_dir", type=str, default="./runs/mtrl")
    p.add_argument("--log_samples_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--eval_subset_limit", type=int, default=200)
    p.add_argument("--save_every_epochs", type=int, default=1)
    p.add_argument("--save_best", action="store_true")

    # Resume
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--save_optimizer", action="store_true")

    args = Args(**vars(p.parse_args()))
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Speeds
    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    min_pixels = args.min_visual_tokens * 28 * 28
    max_pixels = args.max_visual_tokens * 28 * 28
    processor = AutoProcessor.from_pretrained(args.model_id, min_pixels=min_pixels, max_pixels=max_pixels)

    # Base policy
    if args.load_in_8bit:
        qconf = BitsAndBytesConfig(load_in_8bit=True)
        base = Qwen2VLForConditionalGeneration.from_pretrained(args.model_id, quantization_config=qconf, device_map="auto")
        hidden = base.config.hidden_size
    else:
        base = Qwen2VLForConditionalGeneration.from_pretrained(args.model_id, dtype=model_dtype)
        base.to(device)
        hidden = base.config.hidden_size

    if args.gradient_checkpointing:
        try: base.gradient_checkpointing_enable()
        except Exception: pass

    model = PVModel(base, hidden)
    if not args.load_in_8bit:
        model.to(device)

    # Optional ref policy for KL tethering
    ref_model = None
    if args.kl_coef > 0.0:
        try:
            rid = args.ref_model_id if args.ref_model_id else args.model_id
            if args.ref_model_8bit:
                rq = BitsAndBytesConfig(load_in_8bit=True)
                ref_model = Qwen2VLForConditionalGeneration.from_pretrained(rid, quantization_config=rq, device_map="auto")
            else:
                ref_model = Qwen2VLForConditionalGeneration.from_pretrained(rid, dtype=model_dtype).to(device)
            ref_model.eval()
            for p_ in ref_model.parameters(): p_.requires_grad_(False)
        except Exception as e:
            print(f"[KL] ref model load failed, disabling KL: {e}")
            args.kl_coef = 0.0

    # Data
    img_dir, samples = load_split_items(args.dataset_dir, args.train_dataset, args.train_json)
    print(f"Loaded {len(samples)} samples from {args.train_dataset}/{args.train_json}")

    # Opt / logs
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    # Schedules (linear)
    total_updates = args.epochs * args.steps_per_epoch
    temp_start = float(args.temperature)
    tau_start = float(args.tau_success)

    global_step = 0
    best_sr = -1.0

    # --- helpers ---
    def sample_item_with_element():
        item = None
        while item is None:
            cand = random.choice(samples)
            if cand.get("element"): item = cand
        return item

    def rollout_batch():
        """
        Collect args.batch_size_episodes episodes with up to args.horizon steps each.
        Returns list[StepBuf]
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
                # Build conversation
                text = make_messages(processor, img, instruction, history, min_pixels, max_pixels)
                inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt")
                for k in inputs:
                    if isinstance(inputs[k], torch.Tensor):
                        inputs[k] = inputs[k].to(device)
                prompt_len = inputs["input_ids"].shape[1]

                # Gen kwargs (greedy warmup)
                progress = (global_step + 1) / max(1, total_updates)
                temperature_now = temp_start + (args.temperature_end - temp_start) * progress
                tau_now = tau_start + (args.tau_success_end - tau_start) * progress
                do_sample = args.do_sample and (global_step >= args.warmup_steps)

                gen_kwargs = dict(
                    max_new_tokens=args.max_new_tokens,
                    do_sample=do_sample,
                    num_beams=args.num_beams,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    use_cache=True,
                )
                if do_sample:
                    gen_kwargs["temperature"] = max(1e-4, float(temperature_now))
                    if args.top_p and args.top_p > 0: gen_kwargs["top_p"] = float(args.top_p)
                    if args.top_k and args.top_k > 0: gen_kwargs["top_k"] = int(args.top_k)

                # dtype align for pixels
                if "pixel_values" in inputs and inputs["pixel_values"] is not None:
                    inputs["pixel_values"] = inputs["pixel_values"].to(next(model.base.parameters()).dtype)

                with torch.no_grad():
                    gen_out = model.base.generate(**inputs, **gen_kwargs)

                action_ids = gen_out[:, prompt_len:]  # [1, A]
                decoded = processor.batch_decode(action_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                pred = parse_coord(decoded)

                # Reward shaping
                if any(math.isnan(v) for v in pred):
                    cur_dist = 1.0
                    success = False
                    rew = -0.5  # invalid
                else:
                    cur_dist = l2(pred, center_rel)
                    x1, y1, x2, y2 = bbox_rel
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

                # Build full input for scoring
                input_ids_full = torch.cat([inputs["input_ids"], action_ids], dim=1)
                attention_mask_full = (input_ids_full != processor.tokenizer.pad_token_id).long()

                pixel_values = inputs.get("pixel_values", None)
                if pixel_values is not None:
                    pixel_values = pixel_values.to(next(model.base.parameters()).dtype)

                # Old logp / value / entropy (single-sample)
                old_logp, value, entropy = compute_logprob_value_entropy_single(
                    model=model,
                    input_ids_full=input_ids_full,
                    attention_mask_full=attention_mask_full,
                    pixel_values=pixel_values,
                    image_grid_thw=inputs.get("image_grid_thw"),
                    prompt_len=prompt_len,
                    action_ids=action_ids[0].detach().cpu()
                )

                bufs.append(StepBuf(
                    input_ids_full=input_ids_full.detach(),
                    attention_mask_full=attention_mask_full.detach(),
                    pixel_values=pixel_values.detach() if pixel_values is not None else None,
                    image_grid_thw=inputs.get("image_grid_thw"),
                    prompt_len=torch.tensor([prompt_len], device=device),
                    action_ids=action_ids[0].detach().cpu(),
                    old_logp=old_logp.detach().view(1),
                    value=value.detach().unsqueeze(0),
                    reward=torch.tensor([float(rew)], device=device),
                    done=torch.tensor([1.0 if success or t == args.horizon - 1 else 0.0], device=device),
                    entropy=entropy.detach().unsqueeze(0),
                ))

                # Prepare feedback & continue
                feedback = "Success. Stop." if success else dir_feedback(pred, center_rel)
                history.append((decoded, feedback))
                prev_dist = cur_dist
                if success:
                    break

        return bufs

    def ppo_update(bufs: List[StepBuf]):
        # Flatten sequences; compute GAE with episode boundaries indicated by done flags
        rewards = torch.cat([b.reward for b in bufs]).to(device)        # [N]
        dones   = torch.cat([b.done   for b in bufs]).to(device)         # [N]
        values  = torch.cat([b.value  for b in bufs]).to(device).squeeze(-1)  # [N]
        entropies = torch.cat([b.entropy for b in bufs]).to(device).squeeze(-1)  # [N]

        advantages = torch.zeros_like(rewards)
        lastgaelam = 0.0
        next_value = 0.0
        for t in reversed(range(rewards.size(0))):
            mask = 1.0 - float(dones[t].item())
            delta = rewards[t] + args.gae_gamma * next_value * mask - values[t]
            lastgaelam = delta + args.gae_gamma * args.gae_lambda * mask * lastgaelam
            advantages[t] = lastgaelam
            next_value = values[t].item()
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        old_logps = torch.cat([b.old_logp for b in bufs]).to(device).view(-1)  # [N]

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

                # Compute new logp/value/entropy per sample (robust to pixel shape)
                new_logps, new_vals, new_ents = [], [], []
                kl_term_list = []

                for i in mb_idx:
                    b = bufs[i]
                    # pad as batch=1
                    input_ids_full = b.input_ids_full.to(device)
                    attention_mask_full = b.attention_mask_full.to(device)
                    pixel_values = b.pixel_values.to(device) if b.pixel_values is not None else None
                    prompt_len = int(b.prompt_len.item())
                    action_ids = b.action_ids.to(device)

                    # forward
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

                    logits_gen = logits[0, prompt_len-1: prompt_len-1+alen, :]  # [A, V]
                    log_probs = F.log_softmax(logits_gen, dim=-1)
                    probs = log_probs.exp()
                    tok_logp = log_probs.gather(1, action_ids.view(-1,1)).squeeze(1)
                    seq_logp = tok_logp.sum()

                    tok_ent = -(probs * log_probs).sum(dim=-1).mean()

                    v = model.value_from_hidden(hidden, torch.tensor([prompt_len-1], device=hidden.device))

                    new_logps.append(seq_logp)
                    new_vals.append(v.squeeze(0))
                    new_ents.append(tok_ent)

                    # Optional KL regularization w.r.t ref
                    if args.kl_coef > 0.0 and (ref_model is not None):
                        with torch.no_grad():
                            ref_out = ref_model(
                                input_ids=input_ids_full,
                                attention_mask=attention_mask_full,
                                pixel_values=pixel_values,
                                use_cache=False
                            )
                            ref_logits = ref_out.logits[:, :-1, :]
                        p = F.log_softmax(logits[0, prompt_len-1:prompt_len-1+alen, :], dim=-1).exp()
                        ql = F.log_softmax(ref_logits[0, prompt_len-1:prompt_len-1+alen, :], dim=-1)
                        kl_tok = (p * (torch.log(p + 1e-12) - ql)).sum(dim=-1).mean()
                        kl_term_list.append(kl_tok)

                new_logps = torch.stack(new_logps).to(device)
                new_vals  = torch.stack(new_vals).to(device)
                new_ents  = torch.stack(new_ents).to(device)
                kl_loss = torch.stack(kl_term_list).mean() if kl_term_list else torch.tensor(0.0, device=device)

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
                kl_losses.append(kl_loss.item() if isinstance(kl_loss, torch.Tensor) else float(kl_loss))

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
        pbar = tqdm(range(args.steps_per_epoch), desc=f"[MTRL] Epoch {epoch+1}/{args.epochs}")

        for _ in pbar:
            bufs = rollout_batch()
            pl, vl, ent, kl = ppo_update(bufs)
            epoch_pl.append(pl); epoch_vl.append(vl); epoch_ent.append(ent); epoch_kl.append(kl)

            writer.add_scalar("train/policy_loss", pl, global_step)
            writer.add_scalar("train/value_loss", vl, global_step)
            writer.add_scalar("train/entropy", ent, global_step)
            if args.kl_coef > 0: writer.add_scalar("train/kl", kl, global_step)

            if args.log_samples_every and (global_step % args.log_samples_every == 0):
                writer.add_text("train/hint", "Multi-turn rollout collected; PPO updated.", global_step)

            if args.eval_every and (global_step % args.eval_every == 0):
                sr = evaluate_subset(processor, model, args.dataset_dir, args.eval_subset_limit, device, min_pixels, max_pixels)
                writer.add_scalar("eval/screenspot_subset_success", sr, global_step)
                pbar.set_postfix({"SR": f"{sr:.3f}"})
                if args.save_best and sr > best_sr:
                    best_sr = sr
                    save_dir = os.path.join(os.getcwd(), "mtrl_ckpt_best")
                    os.makedirs(save_dir, exist_ok=True)
                    m_to_save = model.module if hasattr(model, "module") else model
                    try:
                        torch.save(m_to_save.base.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
                        m_to_save.base.config.to_json_file(os.path.join(save_dir, "config.json"))
                        with open(os.path.join(save_dir, "training_state.json"), "w") as f:
                            json.dump({"epoch": epoch + 1, "global_step": global_step, "best_sr": best_sr}, f)
                    except Exception as e:
                        print(f"[save_best] {e}")
                    processor.save_pretrained(save_dir)

            global_step += 1

        # End-of-epoch logs
        try:
            writer.add_scalar("epoch/policy_loss", np.mean(epoch_pl), epoch)
            writer.add_scalar("epoch/value_loss", np.mean(epoch_vl), epoch)
            writer.add_scalar("epoch/entropy", np.mean(epoch_ent), epoch)
            if args.kl_coef > 0: writer.add_scalar("epoch/kl", np.mean(epoch_kl), epoch)
        except Exception:
            pass

        # Epoch eval + save
        sr = evaluate_subset(processor, model, args.dataset_dir, args.eval_subset_limit, device, min_pixels, max_pixels)
        print(f"[MTRL] Epoch {epoch+1} subset SR: {sr:.4f}")
        writer.add_scalar("eval/screenspot_subset_success_epoch", sr, epoch)

        if (epoch + 1) % args.save_every_epochs == 0:
            save_dir = os.path.join(os.getcwd(), f"mtrl_ckpt_epoch{epoch+1}")
            os.makedirs(save_dir, exist_ok=True)
            m_to_save = model.module if hasattr(model, "module") else model
            try:
                torch.save(m_to_save.base.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
                m_to_save.base.config.to_json_file(os.path.join(save_dir, "config.json"))
            except Exception as e:
                print(f"[save_epoch] {e}")
            processor.save_pretrained(save_dir)
            if args.save_optimizer:
                try:
                    optimizer_state = optimizer.state_dict()
                    torch.save(optimizer_state, os.path.join(save_dir, "optimizer.pt"))
                    with open(os.path.join(save_dir, "training_state.json"), "w") as f:
                        json.dump({"epoch": epoch + 1, "global_step": global_step}, f)
                except Exception as e:
                    print(f"[save_opt] {e}")

    print("Done.")


if __name__ == "__main__":
    main()
