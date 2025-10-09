import os
import ast
import math
import time
import json
import random
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
    alpha_dist: float = 1.0
    max_new_tokens: int = 64
    temperature: float = 0.7
    seed: int = 42
    gradient_checkpointing: bool = False
    load_in_8bit: bool = False
    log_dir: str = "./runs/rl"
    eval_subset_limit: int = 200
    eval_every_steps: int = 200
    save_every_epochs: int = 1


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Single-GPU only; no distributed initialization


def load_split_items(dataset_dir: str, dataset: str, split: str) -> Tuple[str, List[dict]]:
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
    text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt")
    return text, inputs


def parse_coord(output_text: str) -> Tuple[float, float]:
    try:
        xy = ast.literal_eval(output_text)
        if isinstance(xy, (list, tuple)) and len(xy) == 2:
            x, y = float(xy[0]), float(xy[1])
            return x, y
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
                {"type":"text","text":"Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1."},
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
            pred = parse_coord(pred_str)
            x, y, w, h = item["bbox"]
            gt = [x / img_w, y / img_h, (x + w) / img_w, (y + h) / img_h]
            ok += 1 if (not any(math.isnan(v) for v in pred)) and (gt[0] <= pred[0] <= gt[2]) and (gt[1] <= pred[1] <= gt[3]) else 0
        except Exception:
            continue

    model_unwrapped.train()
    return ok / N


def reinforce_step(model, processor, device, batch, args: RLArgs, optimizer):
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

    with torch.no_grad():
        generated = model_unwrapped.generate(
            **processor_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            use_cache=False,
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
    for out_text, (tgt_xy, _) in zip(decoded, meta_list):
        pred_xy = parse_coord(out_text)
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
    target = input_ids_full[:, prompt_len:].contiguous()
    nll = F.cross_entropy(logits.view(-1, vocab), target.view(-1), reduction="none")
    nll = nll.view(target.size(0), target.size(1))

    token_mask = (target != processor.tokenizer.pad_token_id).float()
    seq_nll = (nll * token_mask).sum(dim=1) / (token_mask.sum(dim=1) + 1e-6)

    # REINFORCE objective: minimize (-reward * logprob) = reward * nll
    loss = (rewards_tensor * seq_nll).mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return float(loss.item()), float(rewards_tensor.mean().item())


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
    parser.add_argument("--alpha_dist", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--model_id", type=str, default="showlab/ShowUI-2B")
    parser.add_argument("--min_visual_tokens", type=int, default=256)
    parser.add_argument("--max_visual_tokens", type=int, default=896)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit (requires bitsandbytes)")
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
        alpha_dist=args_ns.alpha_dist,
        max_new_tokens=args_ns.max_new_tokens,
        temperature=args_ns.temperature,
        model_id=args_ns.model_id,
        min_visual_tokens=args_ns.min_visual_tokens,
        max_visual_tokens=args_ns.max_visual_tokens,
        gradient_checkpointing=args_ns.gradient_checkpointing,
        load_in_8bit=args_ns.load_in_8bit,
    )

    set_seed(args.seed)

    # Single-GPU setup
    global_rank = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    min_pixels = args.min_visual_tokens * 28 * 28
    max_pixels = args.max_visual_tokens * 28 * 28

    processor = AutoProcessor.from_pretrained(args.model_id, min_pixels=min_pixels, max_pixels=max_pixels)
    # Load model on single device
    if args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_id,
            quantization_config=quantization_config,
            device_map={"": local_rank} if torch.cuda.is_available() else "auto",
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

    img_dir, samples = load_split_items(args.dataset_dir, args.train_dataset, args.train_json)
    if global_rank == 0:
        print(f"Loaded {len(samples)} samples from {args.train_dataset}/{args.train_json}")

    writer = None
    if global_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.log_dir)

    global_step = 0
    for epoch in range(args.epochs):
        running_loss = 0.0
        running_reward = 0.0
        start = time.time()

        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch+1}/{args.epochs}")
        for _ in pbar:
            batch_items = []
            for _ in range(args.batch_size):
                item = random.choice(samples)
                image_path = os.path.join(img_dir, item["img_url"]) if "img_url" in item else ""
                element = random.choice(item["element"]) if item.get("element") else None
                if element is None:
                    continue
                element_name = element["instruction"]
                tgt_xy = (float(element["point"][0]), float(element["point"][1]))
                batch_items.append((element_name, image_path, tgt_xy))

            if not batch_items:
                continue

            loss, reward = reinforce_step(model, processor, device, batch_items, args, optimizer)
            running_loss += loss
            running_reward += reward
            pbar.set_postfix({"loss": f"{loss:.4f}", "reward": f"{reward:.3f}"})
            if writer:
                writer.add_scalar("train/loss", loss, global_step)
                writer.add_scalar("train/reward", reward, global_step)

            # periodic subset eval
            if ((global_step + 1) % args.eval_every_steps == 0):
                min_pixels = args.min_visual_tokens * 28 * 28
                max_pixels = args.max_visual_tokens * 28 * 28
                sr = evaluate_screenspot_subset(processor, model, args.dataset_dir, args.eval_subset_limit, min_pixels, max_pixels, device)
                if writer:
                    writer.add_scalar("eval/screenspot_subset_success", sr, global_step)

            global_step += 1

        duration = time.time() - start
        print(f"Epoch {epoch+1} done in {duration:.1f}s | avg loss {running_loss/args.steps_per_epoch:.4f} | avg reward {running_reward/args.steps_per_epoch:.3f}")

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
            print(f"Saved RL checkpoint to {save_dir}")

    # No distributed cleanup needed


if __name__ == "__main__":
    main()


