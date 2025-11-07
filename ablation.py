#!/usr/bin/env python3
"""
Run a complete, sequential ablation study for your ShowUI RL fine‑tuning code, evaluate each run,
aggregate across seeds, and save CSV/JSON/LaTeX + a compact Markdown report.

Usage (example):
    python ablation.py \
      --train_script /mnt/f/USYD/Research/ShowUI/rl_train_optimized.py \
      --dataset_dir /mnt/f/USYD/Research/DATASETS/ShowUI \
      --base_outdir /mnt/f/USYD/Research/ShowUI/ablation_runs \
      --train_dataset showui-train --train_json hf_train \
      --epochs 2 --steps_per_epoch 1000 --eval_subset_limit 400 \
      --seeds 42 2025

This script detects which flags your training script supports (via --help) and only passes those flags.
Unsupported variants (e.g., --reward_ema_beta) are skipped automatically.
"""

from __future__ import annotations
import argparse
import ast
import csv
import json
import math
import os
import re
import shutil
import statistics as stats
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# --------------------------- Utilities --------------------------- #

def run(cmd: List[str], cwd: Path) -> int:
    print("\n[RUN]", " ".join(cmd))
    cwd.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, cwd=str(cwd))
    return proc.returncode


def supports_flag(train_script: Path, flag: str) -> bool:
    try:
        out = subprocess.run([sys.executable, str(train_script), "--help"],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
        return flag in out.stdout
    except Exception:
        return False


# Copied from your training code (kept in sync):
_NUM_RE = r"[-+]?\d*\.?\d+%?"

def parse_coord(output_text: str, img_size: Tuple[int, int] | None = None) -> Tuple[float, float]:
    text = (output_text or "").strip()
    m = re.search(r"\[[^\]]+\]", text)
    if m:
        text = m.group(0)
    try:
        xy = ast.literal_eval(text)
        if isinstance(xy, (list, tuple)) and len(xy) >= 2:
            def to_num(v):
                if isinstance(v, str) and v.strip().endswith('%'):
                    return float(v.strip()[:-1]) / 100.0
                return float(v)
            vals = [to_num(v) for v in xy[:4]]
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
            x, y = vals[0], vals[1]
            if img_size is not None and (abs(x) > 1.0001 or abs(y) > 1.0001):
                iw, ih = max(1.0, float(img_size[0])), max(1.0, float(img_size[1]))
                x = x / iw
                y = y / ih
            return min(1.0, max(0.0, x)), min(1.0, max(0.0, y))
    except Exception:
        pass
    try:
        tokens = re.findall(_NUM_RE, text)
        if len(tokens) >= 2:
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
            x, y = vals[0], vals[1]
            if img_size is not None and (abs(x) > 1.0001 or abs(y) > 1.0001):
                iw, ih = max(1.0, float(img_size[0])), max(1.0, float(img_size[1]))
                x = x / iw
                y = y / ih
            return min(1.0, max(0.0, x)), min(1.0, max(0.0, y))
    except Exception:
        pass
    return float('nan'), float('nan')


# --------------------------- Evaluation --------------------------- #

def evaluate_checkpoint(model_dir: Path, dataset_dir: Path, limit: int,
                        min_visual_tokens: int, max_visual_tokens: int,
                        env_filter: str = "desktop") -> Dict[str, float]:
    """
    Deterministic greedy eval on ScreenSpot hf_test_full (filtered to env_filter).
    Returns dict with: succ_pct, invalid_pct, l2_mean, n_items, n_valid, n_succ, n_invalid
    """
    from PIL import Image
    import json as _json
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    import torch

    meta_path = dataset_dir / "ScreenSpot" / "metadata" / "hf_test_full.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing eval metadata: {meta_path}")

    items = _json.load(open(meta_path))
    items = [it for it in items if str(it.get("split", "")).lower() == env_filter]
    if limit and limit > 0:
        items = items[:limit]

    if not items:
        raise RuntimeError("No items to evaluate after filtering.")

    min_px = min_visual_tokens * 28 * 28
    max_px = max_visual_tokens * 28 * 28

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(str(model_dir), min_pixels=min_px, max_pixels=max_px)
    model = Qwen2VLForConditionalGeneration.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16 if device=="cuda" else torch.float32)
    model.to(device)
    model.eval()

    succ = 0
    invalid = 0
    l2_sum = 0.0
    valid_count = 0

    def center_from_bbox(bbox, w, h):
        x, y, bw, bh = bbox
        return (x + bw/2)/w, (y + bh/2)/h

    for it in tqdm(items, desc="Evaluating samples", leave=False):
        img_path = dataset_dir / "ScreenSpot" / "images" / it["img_url"]
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        if "img_size" in it and isinstance(it["img_size"], (list, tuple)):
            W, H = it["img_size"][0], it["img_size"][1]
        else:
            W, H = img.size

        messages = [{
            "role": "user",
            "content": [
                {"type":"text","text":"Return exactly two numbers in square brackets like [x, y] with both x and y in [0,1]."},
                {"type":"image","image": img, "min_pixels": min_px, "max_pixels": max_px},
                {"type":"text","text": it["task"]}
            ]
        }]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False, num_beams=1,
                                 eos_token_id=proc.tokenizer.eos_token_id, use_cache=False)
        gen = out[:, inputs.input_ids.shape[1]:]
        pred_str = proc.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        px, py = parse_coord(pred_str, img_size=(W, H))
        if any(math.isnan(v) for v in (px, py)) or px < 0 or px > 1 or py < 0 or py > 1:
            invalid += 1
            continue
        cx, cy = center_from_bbox(it["bbox"], W, H)
        l2 = math.sqrt((px-cx)**2 + (py-cy)**2)
        l2_sum += l2
        valid_count += 1
        x, y, bw, bh = it["bbox"]
        x1, y1, x2, y2 = x/W, y/H, (x+bw)/W, (y+bh)/H
        succ += 1 if (x1 <= px <= x2 and y1 <= py <= y2) else 0

    n_items = len(items)
    succ_pct = 100.0 * succ / max(1, n_items)
    invalid_pct = 100.0 * invalid / max(1, n_items)
    l2_mean = l2_sum / max(1, valid_count)
    
    # Cleanup GPU memory
    del model
    del proc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    return {
        "succ_pct": succ_pct,
        "invalid_pct": invalid_pct,
        "l2_mean": l2_mean,
        "n_items": n_items,
        "n_valid": valid_count,
        "n_succ": succ,
        "n_invalid": invalid,
    }


# --------------------------- Cleanup Helpers --------------------------- #

def cleanup_run_artifacts(run_dir: Path, keep_best: bool = True, remove_tb: bool = False) -> None:
    """Remove large intermediate artifacts to save disk.
    - Deletes all rl_ckpt_epoch* directories
    - Deletes optimizer.pt and training_state.json if present
    - Optionally removes TensorBoard logs
    Keeps rl_ckpt_best by default.
    """
    try:
        # Remove per-epoch checkpoints
        for p in run_dir.glob("rl_ckpt_epoch*"):
            shutil.rmtree(p, ignore_errors=True)
        # Remove optimizer/training state files wherever present under run_dir
        for p in run_dir.rglob("optimizer.pt"):
            try:
                p.unlink()
            except Exception:
                pass
        for p in run_dir.rglob("training_state.json"):
            try:
                p.unlink()
            except Exception:
                pass
        if remove_tb:
            tb = run_dir / "tb"
            if tb.exists():
                shutil.rmtree(tb, ignore_errors=True)
    except Exception:
        pass


# --------------------------- Experiment Plan --------------------------- #

FULL_BASE = {
    # Data & schedule (optimized baseline)
    "--epochs": 20,
    "--steps_per_epoch": 200,
    "--batch_size": 1,
    "--lr": 5e-6,
    "--max_new_tokens": 32,
    "--min_visual_tokens": 256,
    "--max_visual_tokens": 896,
    # Eval cadence
    "--eval_subset_limit": 400,
    "--eval_every_steps": 10000,  # effectively eval at epoch end only
    # RL knobs (rl_train_optimized.py style)
    # Fixed tau (both start and end at same value)
    "--tau_success": 0.06,
    "--tau_success_end": 0.06,
    "--alpha_dist": 1.0,
    "--entropy_coef": 0.01,
    "--kl_coef": 0.02,
    "--target_kl": 0.08,
    "--kl_adapt_every": 50,
    "--kl_adapt_rate": 1.5,
    "--min_kl_coef": 1e-4,
    "--max_kl_coef": 5e-1,
    "--warmup_steps": 200,
    # Safety
    "--safety_cooldown_steps": 20,
    "--safety_entropy_threshold": 3.0,
    "--safety_kl_multiplier": 5.0,
    "--safety_temp_floor": 0.3,
    "--safety_temp_decay": 0.5,
    # Storage control: rely on best-only
    "--save_every_epochs": 0,
}

FULL_FLAGS_TRUE = {
    "--save_best": True,
    "--do_sample": True,
    "--constrained_decode": True,
}

# Variant definitions as overrides relative to FULL
VARIANTS = {
    "full": ({}, {}),
    "no_distance": ({"--alpha_dist": 0.0}, {}),
    # EMA baseline toggle (best-effort; only applied if train script supports --reward_ema_beta)
    "no_ema": ({"--reward_ema_beta": 0.0}, {}),
    "no_entropy": ({"--entropy_coef": 0.0}, {}),
    "no_warmup": ({"--warmup_steps": 0}, {}),
    "no_safety": ({"--safety_cooldown_steps": 0}, {}),
    "adaptive_tau": ({"--tau_success": 0.06, "--tau_success_end": 0.02}, {}),
    "no_tau": ({"--tau_success": 0.0, "--tau_success_end": 0.0}, {}),
    "no_kl": ({"--kl_coef": 0.0}, {}),
    "greedy_only": ({}, {"--do_sample": False}),
}

# Pretty names for LaTeX/CSV
PRETTY = {
    "full": "Full (ours)",
    "no_distance": "\\hspace{1em}-- distance term",
    "no_ema": "\\hspace{1em}-- EMA baseline",
    "no_entropy": "\\hspace{1em}-- entropy",
    "no_warmup": "\\hspace{1em}-- warm-up",
    "no_safety": "\\hspace{1em}-- safety cd.",
    "adaptive_tau": "\\hspace{1em}-- fixed \\tau (adaptive instead)",
    "no_tau": "\\hspace{1em}-- \\tau (set to 0)",
    "no_kl": "\\hspace{1em}-- adaptive KL",
    "greedy_only": "\\hspace{1em} greedy only",
}

# --------------------------- Orchestrator --------------------------- #

def build_cmd(train_script: Path, dataset_dir: Path, outdir: Path, seed: int,
              base: Dict[str, object], flags_true: Dict[str, bool], overrides: Dict[str, object],
              overrides_true: Dict[str, bool], train_dataset: str, train_json: str) -> List[str]:
    cmd = [sys.executable, str(train_script.resolve()),
           "--dataset_dir", str(dataset_dir.resolve()),
           "--train_dataset", str(train_dataset), "--train_json", str(train_json),
           "--model_id", "showlab/ShowUI-2B",
           "--log_dir", str(outdir / "tb"),
           "--eval_envs", "desktop",
           "--stats_jsonl", str(outdir / "stats.jsonl"),
          ]

    def maybe_add_flag(k: str, v: object):
        nonlocal cmd
        if supports_flag(train_script, k):
            if isinstance(v, bool):
                if v:
                    cmd.append(k)
            else:
                cmd += [k, str(v)]

    # Compose flags with support checks
    for k, v in base.items():
        maybe_add_flag(k, v)
    # Decoding defaults (guarded)
    maybe_add_flag("--top_p", 0.9)
    maybe_add_flag("--temperature", 0.7)
    maybe_add_flag("--num_beams", 1)
    for k, v in flags_true.items():
        maybe_add_flag(k, v)
    for k, v in overrides.items():
        maybe_add_flag(k, v)
    for k, v in overrides_true.items():
        maybe_add_flag(k, v)

    # Ensure batch size present (last wins)
    maybe_add_flag("--batch_size", base.get("--batch_size", 1))
    # Seed if supported
    if supports_flag(train_script, "--seed"):
        cmd += ["--seed", str(seed)]
    return cmd


def find_ckpt(run_dir: Path) -> Optional[Path]:
    best = run_dir / "rl_ckpt_best"
    if best.exists():
        return best
    # else pick latest epoch
    epochs = []
    for p in run_dir.glob("rl_ckpt_epoch*"):
        try:
            n = int(p.name.replace("rl_ckpt_epoch", ""))
            epochs.append((n, p))
        except Exception:
            pass
    if epochs:
        epochs.sort(key=lambda x: x[0], reverse=True)
        return epochs[0][1]
    return None


def ci95(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    m = stats.mean(values)
    sd = stats.stdev(values)
    half = 1.96 * sd / math.sqrt(len(values))
    return m, half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_script", type=Path, required=True, help="Path to your train_rl.py")
    ap.add_argument("--dataset_dir", type=Path, required=True)
    ap.add_argument("--base_outdir", type=Path, default=Path("./ablation_runs"))
    ap.add_argument("--train_dataset", type=str, default="showui-train")
    ap.add_argument("--train_json", type=str, default="hf_train")
    ap.add_argument("--epochs", type=int, default=FULL_BASE["--epochs"])  # allow override
    ap.add_argument("--steps_per_epoch", type=int, default=FULL_BASE["--steps_per_epoch"])  # allow override
    ap.add_argument("--eval_subset_limit", type=int, default=FULL_BASE["--eval_subset_limit"])  # allow override
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2025])
    args = ap.parse_args()

    # Update plan from CLI overrides
    FULL_BASE["--epochs"] = args.epochs
    FULL_BASE["--steps_per_epoch"] = args.steps_per_epoch
    FULL_BASE["--eval_subset_limit"] = args.eval_subset_limit

    # Detect unsupported flags and adjust the variants list accordingly
    has_reward_ema_beta = supports_flag(args.train_script, "--reward_ema_beta")
    if not has_reward_ema_beta and "no_ema" in VARIANTS:
        print("[WARN] --reward_ema_beta not supported by the training script; skipping 'no_ema' ablation.")

    # Load existing results if they exist (for resuming/appending)
    per_run_csv = args.base_outdir / "per_run_results.csv"
    existing_results = {}  # key: (variant, seed) -> row dict
    if per_run_csv.exists():
        try:
            with open(per_run_csv, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert numeric fields back
                    for k in ["seed", "n_items", "n_valid", "n_succ", "n_invalid"]:
                        if k in row:
                            row[k] = int(row[k])
                    for k in ["succ_pct", "invalid_pct", "l2_mean", "train_time_sec"]:
                        if k in row:
                            row[k] = float(row[k])
                    key = (row["variant"], row["seed"])
                    existing_results[key] = row
            print(f"[LOAD] Found {len(existing_results)} existing results in {per_run_csv}")
        except Exception as e:
            print(f"[WARN] Could not load existing results: {e}")

    results_rows = []  # per-run rows
    summary_rows = []  # per-variant mean±CI

    args.base_outdir.mkdir(parents=True, exist_ok=True)

    # Save a copy of the plan
    plan_path = args.base_outdir / "plan.json"
    json.dump({
        "full_base": FULL_BASE,
        "full_flags_true": FULL_FLAGS_TRUE,
        "variants": list(VARIANTS.keys()),
        "seeds": args.seeds,
    }, open(plan_path, "w"), indent=2)

    # Filter out unsupported variants
    variants_to_run = [(vname, over, over_true) for vname, (over, over_true) in VARIANTS.items()
                       if not (vname == "no_ema" and not has_reward_ema_beta)]
    
    print(f"\n{'='*80}")
    print(f"Running {len(variants_to_run)} variants × {len(args.seeds)} seeds = {len(variants_to_run) * len(args.seeds)} total runs")
    print(f"{'='*80}\n")
    
    for vname, over, over_true in tqdm(variants_to_run, desc="Ablation Variants", position=0, leave=True):
        vdir = args.base_outdir / vname
        vdir.mkdir(parents=True, exist_ok=True)
        per_seed_metrics = {"succ": [], "invalid": [], "l2": []}

        for seed in tqdm(args.seeds, desc=f"  {vname} seeds", position=1, leave=False):
            run_dir = vdir / f"seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if already completed
            eval_json = run_dir / "eval.json"
            if eval_json.exists():
                print(f"[SKIP] {vname} seed {seed} already evaluated, loading results...")
                try:
                    with open(eval_json) as f:
                        row = json.load(f)
                    results_rows.append(row)
                    per_seed_metrics["succ"].append(row["succ_pct"])
                    per_seed_metrics["invalid"].append(row["invalid_pct"])
                    per_seed_metrics["l2"].append(row["l2_mean"])
                    print(f"[LOADED] {vname} seed {seed}: succ={row['succ_pct']:.2f}%, l2={row['l2_mean']:.4f}")
                    continue
                except Exception as e:
                    print(f"[WARN] Failed to load {eval_json}: {e}. Re-running...")

            # Compose command
            cmd = build_cmd(args.train_script, args.dataset_dir, run_dir, seed,
                            FULL_BASE, FULL_FLAGS_TRUE, over, over_true,
                            args.train_dataset, args.train_json)

            # Train (sequential)
            tqdm.write(f"\n[TRAIN] Starting {vname} seed {seed}...")
            t0 = time.time()
            rc = run(cmd, cwd=run_dir)
            t_train = time.time() - t0
            tqdm.write(f"[TRAIN] Completed in {t_train/60:.1f} minutes")
            
            # Wait for GPU memory to be fully released by the subprocess
            tqdm.write("[GPU] Waiting 5 seconds for GPU memory cleanup...")
            time.sleep(5)
            
            if rc != 0:
                tqdm.write(f"[ERROR] Training failed for {vname} seed {seed} (rc={rc}). Skipping eval.")
                continue

            # Find checkpoint
            ckpt = find_ckpt(run_dir)
            if not ckpt:
                tqdm.write(f"[ERROR] No checkpoint found in {run_dir}.")
                continue

            # Evaluate
            tqdm.write(f"[EVAL] Evaluating {vname} seed {seed} @ {ckpt.name}")
            eval_stats = evaluate_checkpoint(
                model_dir=ckpt,
                dataset_dir=args.dataset_dir,
                limit=FULL_BASE["--eval_subset_limit"],
                min_visual_tokens=int(FULL_BASE["--min_visual_tokens"]),
                max_visual_tokens=int(FULL_BASE["--max_visual_tokens"]),
                env_filter="desktop",
            )
            
            # Wait after evaluation for GPU cleanup
            tqdm.write("[GPU] Waiting 3 seconds after evaluation...")
            time.sleep(3)

            row = {
                "variant": vname,
                "seed": seed,
                "succ_pct": eval_stats["succ_pct"],
                "invalid_pct": eval_stats["invalid_pct"],
                "l2_mean": eval_stats["l2_mean"],
                "n_items": eval_stats["n_items"],
                "n_valid": eval_stats["n_valid"],
                "n_succ": eval_stats["n_succ"],
                "n_invalid": eval_stats["n_invalid"],
                "train_time_sec": t_train,
            }
            results_rows.append(row)
            json.dump(row, open(run_dir / "eval.json", "w"), indent=2)
            tqdm.write(f"[RESULT] {vname} seed {seed}: succ={row['succ_pct']:.2f}%, l2={row['l2_mean']:.4f}, invalid={row['invalid_pct']:.2f}%")

            # For summary
            per_seed_metrics["succ"].append(row["succ_pct"])  # % values
            per_seed_metrics["invalid"].append(row["invalid_pct"])  # % values
            per_seed_metrics["l2"].append(row["l2_mean"])  # absolute

            # Storage cleanup: remove intermediate checkpoints, optimizer files, and TB logs
            tqdm.write(f"[CLEANUP] Removing intermediate checkpoints for {vname} seed {seed}...")
            cleanup_run_artifacts(run_dir, keep_best=True, remove_tb=True)

        # Aggregate over seeds
        if per_seed_metrics["succ"]:
            m_succ, ci_succ = ci95(per_seed_metrics["succ"])  # %
            m_inv, ci_inv = ci95(per_seed_metrics["invalid"])  # %
            m_l2, ci_l2 = ci95(per_seed_metrics["l2"])        # absolute
            summary_rows.append({
                "variant": vname,
                "variant_pretty": PRETTY.get(vname, vname),
                "succ_pct_mean": m_succ,
                "succ_pct_ci": ci_succ,
                "l2_mean": m_l2,
                "l2_ci": ci_l2,
                "invalid_pct_mean": m_inv,
                "invalid_pct_ci": ci_inv,
                "n_seeds": len(per_seed_metrics["succ"]),
            })
            tqdm.write(f"\n[SUMMARY] {vname}: succ={m_succ:.2f}±{ci_succ:.2f}%, l2={m_l2:.4f}±{ci_l2:.4f}, invalid={m_inv:.2f}±{ci_inv:.2f}%\n")

    # Merge new results with existing results
    all_results = dict(existing_results)  # Start with existing
    for row in results_rows:
        key = (row["variant"], row["seed"])
        all_results[key] = row  # Update/add new results
    
    # Convert to list for CSV
    all_results_list = sorted(all_results.values(), key=lambda x: (x["variant"], x["seed"]))
    
    # Recompute summary from ALL available results (not just this run)
    summary_by_variant = {}
    for row in all_results_list:
        vname = row["variant"]
        if vname not in summary_by_variant:
            summary_by_variant[vname] = {"succ": [], "invalid": [], "l2": []}
        summary_by_variant[vname]["succ"].append(row["succ_pct"])
        summary_by_variant[vname]["invalid"].append(row["invalid_pct"])
        summary_by_variant[vname]["l2"].append(row["l2_mean"])
    
    # Build new summary rows from complete data
    summary_rows_complete = []
    for vname, metrics in summary_by_variant.items():
        if metrics["succ"]:
            m_succ, ci_succ = ci95(metrics["succ"])
            m_inv, ci_inv = ci95(metrics["invalid"])
            m_l2, ci_l2 = ci95(metrics["l2"])
            summary_rows_complete.append({
                "variant": vname,
                "variant_pretty": PRETTY.get(vname, vname),
                "succ_pct_mean": m_succ,
                "succ_pct_ci": ci_succ,
                "l2_mean": m_l2,
                "l2_ci": ci_l2,
                "invalid_pct_mean": m_inv,
                "invalid_pct_ci": ci_inv,
                "n_seeds": len(metrics["succ"]),
            })
    
    # Write per-run CSV with ALL results
    if all_results_list:
        with open(per_run_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_results_list[0].keys()))
            w.writeheader(); w.writerows(all_results_list)
        print(f"[WRITE] {per_run_csv} ({len(all_results_list)} runs total)")

    # Write summary CSV + JSON with complete statistics
    summary_csv = args.base_outdir / "summary.csv"
    summary_json = args.base_outdir / "summary.json"
    if summary_rows_complete:
        with open(summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows_complete[0].keys()))
            w.writeheader(); w.writerows(summary_rows_complete)
        json.dump(summary_rows_complete, open(summary_json, "w"), indent=2)
        print(f"[WRITE] {summary_csv}\n[WRITE] {summary_json}")

    # LaTeX table (use complete summary with all seeds)
    if summary_rows_complete:
        # order rows as defined in PRETTY mapping
        def sort_key(r):
            order = list(PRETTY.keys())
            try:
                return order.index(r["variant"]) if r["variant"] in order else 999
            except Exception:
                return 999
        summary_rows_sorted = sorted(summary_rows_complete, key=sort_key)

        def fmt_pm(m, ci, prec=2):
            return f"{m:.{prec}f} \\pm {ci:.{prec}f}"

        lines = [
            "\\begin{table}[!htbp]",
            "\\centering",
            "\\footnotesize",
            "\\setlength{\\tabcolsep}{4.5pt}",
            "\\renewcommand{\\arraystretch}{1.08}",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "\\textbf{Variant} & \\textbf{Succ (\\%)} & \\textbf{L2} & \\textbf{Invalid (\\%)} \\\\",
            "\\midrule",
        ]
        for r in summary_rows_sorted:
            lines.append(
                f"{r['variant_pretty']} & "
                f"{fmt_pm(r['succ_pct_mean'], r['succ_pct_ci'], 2)} & "
                f"{fmt_pm(r['l2_mean'], r['l2_ci'], 4)} & "
                f"{fmt_pm(r['invalid_pct_mean'], r['invalid_pct_ci'], 2)} \\\\"
            )
        lines += [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Ablations on the desktop subset. Each row toggles one component relative to Full.}",
            "\\label{tab:ablation}",
            "\\end{table}",
        ]
        tex_path = args.base_outdir / "ablation_table.tex"
        open(tex_path, "w").write("\n".join(lines))
        print(f"[WRITE] {tex_path}")

    # Markdown report
    report_md = args.base_outdir / "Ablation_Report.md"
    entropy_str = f"entropy={FULL_BASE.get('--entropy_coef', 'N/A')}"
    open(report_md, "w").write(textwrap.dedent(f"""
        # Ablation Study Report

        **Data**: `showui-desktop/hf_train` → eval on `ScreenSpot/hf_test_full` (desktop); limit={FULL_BASE['--eval_subset_limit']}

        **Training budget**: epochs={FULL_BASE['--epochs']}, steps_per_epoch={FULL_BASE['--steps_per_epoch']}, batch_size=1

        **Full (ours)** key knobs: tau={FULL_BASE['--tau_success']} (fixed at {FULL_BASE['--tau_success_end']}), alpha_dist={FULL_BASE['--alpha_dist']}, {entropy_str},
        KL (coef={FULL_BASE['--kl_coef']}, target={FULL_BASE['--target_kl']}), warmup={FULL_BASE['--warmup_steps']}, safety cooldown steps={FULL_BASE['--safety_cooldown_steps']}.

        See `summary.csv` and `ablation_table.tex` for final numbers (mean ± 95% CI over seeds={args.seeds}).
    """))
    print(f"[WRITE] {report_md}")
    
    print(f"\n{'='*80}")
    print("✓ Ablation study complete!")
    print(f"  Results saved to: {args.base_outdir}")
    print(f"  Per-run CSV: {per_run_csv}")
    print(f"  Summary CSV: {summary_csv}")
    print(f"  LaTeX table: {args.base_outdir / 'ablation_table.tex'}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
