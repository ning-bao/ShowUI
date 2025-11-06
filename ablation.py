#!/usr/bin/env python3
"""
ablation_best.py — Storage-aware, resume-safe ablation study orchestrator for rl_train_optimized.py
Now with:
- Global progress bar (tqdm if available; graceful fallback otherwise)
- Clear step logging (baselines, per-run train/eval/cleanup, writes)
- Plan/variant listing at start; --list_only to print plan and exit
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

# -------- tqdm (progress bars) with graceful fallback -------- #
try:
    from tqdm import tqdm
except Exception:
    class _DummyTQDM:
        def __init__(self, total=None, desc=None, unit=None, leave=True): self.n=0
        def update(self, n=1): self.n += n
        def close(self): pass
    def tqdm(*a, **k): return _DummyTQDM()

# --------------------------- Helpers --------------------------- #

def run(cmd: List[str], cwd: Path) -> int:
    print("\n[RUN]", " ".join(cmd))
    cwd.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, cwd=str(cwd))
    return proc.returncode

def _get_help_flags(train_script: Path) -> str:
    try:
        out = subprocess.run([sys.executable, str(train_script), "--help"],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, timeout=90)
        return out.stdout or ""
    except Exception:
        return ""

def supports_flag(help_text: str, flag: str) -> bool:
    pat = rf"(?:^|\s){re.escape(flag)}(?:[=\s]|$)"
    return re.search(pat, help_text) is not None

def human_size(bytes_: int) -> str:
    units = ["B","KB","MB","GB","TB"]
    size = float(bytes_)
    for u in units:
        if size < 1024.0:
            return f"{size:.1f}{u}"
        size /= 1024.0
    return f"{size:.1f}PB"

def dir_size_bytes(p: Path) -> int:
    total = 0
    if p.exists():
        for root, _, files in os.walk(p):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except Exception:
                    pass
    return total

# --------------------------- Parsing coords (eval) --------------------------- #

_NUM_RE = r"[-+]?\d*\.?\d+%?"

def parse_coord_text(output_text: str, img_size: Tuple[int, int] | None = None) -> Tuple[float, float]:
    text = (output_text or "").strip()
    m = re.search(r"\[[^\]]+\]", text)
    if m:
        text = m.group(0)
    # literal eval
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
    # regex fallback
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

# --------------------------- Deterministic greedy eval --------------------------- #

def evaluate_checkpoint(model_dir_or_id: str, dataset_dir: Path, limit: int,
                        min_visual_tokens: int, max_visual_tokens: int,
                        env_filter: str = "desktop") -> Dict[str, float]:
    """
    Deterministic greedy eval on ScreenSpot hf_test_full (filtered to env_filter).
    Returns: succ_pct, invalid_pct, l2_mean, n_items, n_valid, n_succ, n_invalid
    """
    from PIL import Image
    import json as _json
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

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
    proc = AutoProcessor.from_pretrained(model_dir_or_id, min_pixels=min_px, max_pixels=max_px)
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_dir_or_id,
                                                            torch_dtype=torch.bfloat16 if device=="cuda" else torch.float32)
    model.to(device)
    model.eval()

    succ = 0
    invalid = 0
    l2_sum = 0.0
    valid_count = 0

    def center_from_bbox(bbox, w, h):
        x, y, bw, bh = bbox
        return (x + bw/2)/w, (y + bh/2)/h

    for it in items:
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
        px, py = parse_coord_text(pred_str, img_size=(W, H))
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

    del model
    del proc
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass

    return {
        "succ_pct": succ_pct,
        "invalid_pct": invalid_pct,
        "l2_mean": l2_mean,
        "n_items": n_items,
        "n_valid": valid_count,
        "n_succ": succ,
        "n_invalid": invalid,
    }

# --------------------------- Disk cleanup --------------------------- #

def cleanup_run_artifacts(run_dir: Path, keep_best_dir: bool = True, remove_tb: bool = True) -> None:
    try:
        for p in run_dir.glob("rl_ckpt_epoch*"):
            shutil.rmtree(p, ignore_errors=True)
        for p in run_dir.rglob("optimizer.pt"):
            try: p.unlink()
            except Exception: pass
        for p in run_dir.rglob("training_state.json"):
            try: p.unlink()
            except Exception: pass
        if remove_tb:
            tb = run_dir / "tb"
            if tb.exists():
                shutil.rmtree(tb, ignore_errors=True)
    except Exception:
        pass

def find_ckpt(run_dir: Path) -> Optional[Path]:
    best = run_dir / "rl_ckpt_best"
    if best.exists():
        return best
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

# --------------------------- Stats utils --------------------------- #

def ci95(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    m = stats.mean(values)
    sd = stats.stdev(values)
    half = 1.96 * sd / math.sqrt(len(values))
    return m, half

# --------------------------- Experiment Plan --------------------------- #

FULL_BASE = {
    "--epochs": 20,
    "--steps_per_epoch": 200,
    "--batch_size": 1,
    "--lr": 5e-6,
    "--max_new_tokens": 32,
    "--min_visual_tokens": 256,
    "--max_visual_tokens": 896,
    "--temperature": 0.7,
    "--top_p": 0.9,
    "--eval_subset_limit": 200,         # in-training eval (fast)
    "--save_every_epochs": 1000000000,  # disable per-epoch save safely
    "--entropy_coef": 0.01,
    "--kl_coef": 0.02,
    "--target_kl": 0.08,
    "--kl_adapt_every": 50,
    "--kl_adapt_rate": 1.5,
    "--min_kl_coef": 1e-4,
    "--max_kl_coef": 5e-1,
    "--warmup_steps": 200,
    "--safety_cooldown_steps": 20,
    "--safety_entropy_threshold": 3.0,
    "--safety_kl_multiplier": 5.0,
    "--safety_temp_floor": 0.3,
    "--safety_temp_decay": 0.5,
    "--save_best": True,
}

VARIANTS: Dict[str, Tuple[Dict[str, object], Dict[str, bool], Dict[str, bool]]] = {
    "full": ({}, {"--do_sample": True, "--constrained_decode": True}, {}),
    "greedy_only": ({}, {"--constrained_decode": True}, {"--do_sample": True}),
    "no_constrained": ({}, {}, {"--constrained_decode": True}),
    "no_kl": ({"--kl_coef": 0.0}, {"--constrained_decode": True}, {"--do_sample": False}),
    "low_entropy": ({"--entropy_coef": 0.003}, {"--do_sample": True, "--constrained_decode": True}, {}),
    "no_warmup": ({"--warmup_steps": 0}, {"--do_sample": True, "--constrained_decode": True}, {}),
    "no_safety": ({"--safety_cooldown_steps": 0}, {"--do_sample": True, "--constrained_decode": True}, {}),
    "large_visual_tokens": ({"--min_visual_tokens": 384, "--max_visual_tokens": 1024},
                            {"--do_sample": True, "--constrained_decode": True}, {}),
}

PRETTY = {
    "full": "Full (sampling + FSM)",
    "greedy_only": "\\hspace{1em} greedy only",
    "no_constrained": "\\hspace{1em}-- constrained decode",
    "no_kl": "\\hspace{1em}-- KL",
    "low_entropy": "\\hspace{1em} entropy=0.003",
    "no_warmup": "\\hspace{1em}-- warm-up",
    "no_safety": "\\hspace{1em}-- safety cooldown",
    "large_visual_tokens": "\\hspace{1em} larger visual tokens",
}

# --------------------------- Orchestrator --------------------------- #

def build_cmd(help_text: str, train_script: Path, dataset_dir: Path, outdir: Path, seed: int,
              base: Dict[str, object], flags_true: Dict[str, bool],
              flags_false: Dict[str, bool], overrides: Dict[str, object],
              train_dataset: str, train_json: str,
              steps_per_epoch: int) -> List[str]:
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
        if supports_flag(help_text, k):
            if isinstance(v, bool):
                if v:
                    cmd.append(k)
            else:
                cmd += [k, str(v)]

    for k, v in base.items():
        maybe_add_flag(k, v)
    # ensure save_best happens once per epoch
    maybe_add_flag("--eval_every_steps", steps_per_epoch)
    maybe_add_flag("--num_beams", 1)

    for k, v in flags_true.items():
        maybe_add_flag(k, v)
    for k, v in overrides.items():
        maybe_add_flag(k, v)

    maybe_add_flag("--batch_size", base.get("--batch_size", 1))

    if supports_flag(help_text, "--seed"):
        cmd += ["--seed", str(seed)]
    return cmd

def print_plan(args, help_text: str):
    seeds = args.seeds
    variants = list(VARIANTS.keys())
    runs = len(variants) * len(seeds)
    baselines = 1 + (1 if args.user_baseline_dir else 0)

    print("\n===== ABLATION PLAN =====")
    print(f"Dataset dir       : {args.dataset_dir}")
    print(f"Train dataset/json: {args.train_dataset} / {args.train_json}")
    print(f"Epochs x Steps    : {args.epochs} x {args.steps_per_epoch}")
    print(f"In-train eval N   : {FULL_BASE['--eval_subset_limit']}")
    print(f"Final eval N      : {args.final_eval_limit}")
    print(f"Seeds             : {seeds}")
    print(f"Variants ({len(variants)}): {variants}")
    print(f"Total runs        : {runs} (variants × seeds)")
    print(f"Baselines         : {baselines} (zero-shot + {'user' if args.user_baseline_dir else 'no user'})")
    print("Disk policy       : delete per-epoch ckpts & TB; "
          + ("keep ONE global best ckpt" if args.keep_global_best_ckpt else "delete all ckpts after eval"))
    print("\n-- Variant specs --")
    for name, (ov, ft, ff) in VARIANTS.items():
        def fmt(d): 
            if not d: return "{}"
            return "{ " + ", ".join(f"{k}={v}" for k,v in d.items()) + " }"
        def fmtb(d): 
            if not d: return "{}"
            return "{ " + ", ".join(k for k in d.keys()) + " }"
        print(f" * {name}: overrides={fmt(ov)} | flags_true={fmtb(ft)} | flags_false={fmtb(ff)}")
    print("=========================\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_script", type=Path, required=True)
    ap.add_argument("--dataset_dir", type=Path, required=True)
    ap.add_argument("--base_outdir", type=Path, default=Path("./ablation_runs"))
    ap.add_argument("--train_dataset", type=str, default="showui-train")
    ap.add_argument("--train_json", type=str, default="hf_train")
    ap.add_argument("--epochs", type=int, default=FULL_BASE["--epochs"])
    ap.add_argument("--steps_per_epoch", type=int, default=FULL_BASE["--steps_per_epoch"])
    ap.add_argument("--final_eval_limit", type=int, default=334)
    ap.add_argument("--intrain_eval_limit", type=int, default=FULL_BASE["--eval_subset_limit"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 2025])
    ap.add_argument("--keep_global_best_ckpt", action="store_true")
    ap.add_argument("--user_baseline_dir", type=str, default="")
    ap.add_argument("--list_only", action="store_true", help="Print variants/plan and exit.")
    args = ap.parse_args()

    FULL_BASE["--epochs"] = args.epochs
    FULL_BASE["--steps_per_epoch"] = args.steps_per_epoch
    FULL_BASE["--eval_subset_limit"] = max(10, int(args.intrain_eval_limit))

    help_text = _get_help_flags(args.train_script)

    args.base_outdir.mkdir(parents=True, exist_ok=True)
    per_run_csv = args.base_outdir / "per_run_results.csv"
    summary_csv = args.base_outdir / "summary.csv"
    summary_json = args.base_outdir / "summary.json"
    tex_path = args.base_outdir / "ablation_table.tex"
    plan_path = args.base_outdir / "plan.json"

    # Save & show plan
    json.dump({
        "full_base": FULL_BASE,
        "variants": list(VARIANTS.keys()),
        "seeds": args.seeds,
        "final_eval_limit": args.final_eval_limit,
        "intrain_eval_limit": args.intrain_eval_limit,
    }, open(plan_path, "w"), indent=2)
    print_plan(args, help_text)
    if args.list_only:
        print("[INFO] --list_only given; exiting without running.")
        return

    # Load existing rows (resume)
    existing: Dict[Tuple[str,int], dict] = {}
    if per_run_csv.exists():
        try:
            with open(per_run_csv, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for k in ["seed", "n_items", "n_valid", "n_succ", "n_invalid"]:
                        if k in row and row[k] != "":
                            row[k] = int(row[k])
                    for k in ["succ_pct", "invalid_pct", "l2_mean", "train_time_sec"]:
                        if k in row and row[k] != "":
                            row[k] = float(row[k])
                    existing[(row["variant"], row["seed"])] = row
            print(f"[LOAD] Found {len(existing)} rows in {per_run_csv}")
        except Exception as e:
            print(f"[WARN] Could not load existing {per_run_csv}: {e}")

    per_run_rows: List[dict] = []
    global_best: Tuple[float, Optional[Path]] = (-1.0, None)  # (succ_pct, path)

    # Compute overall steps for progress bar
    variants = list(VARIANTS.keys())
    total_runs = len(variants) * len(args.seeds)
    baseline_steps = 1 + (1 if args.user_baseline_dir else 0)   # each is one eval
    steps_per_run = 3  # train -> eval -> cleanup
    write_steps = 3    # per_run_csv + summary csv/json + latex
    total_steps = baseline_steps + total_runs * steps_per_run + write_steps

    overall = tqdm(total=total_steps, desc="Overall", unit="step")
    runs_bar = tqdm(total=total_runs, desc="Ablation runs", unit="run")

    # --- Baselines ---
    print("[STEP] Baseline: zero-shot ShowUI-2B")
    try:
        bstats = evaluate_checkpoint("showlab/ShowUI-2B", args.dataset_dir, args.final_eval_limit,
                                     int(FULL_BASE["--min_visual_tokens"]), int(FULL_BASE["--max_visual_tokens"]),
                                     env_filter="desktop")
        row = {"variant": "baseline_zero_shot", "seed": 0, "train_time_sec": 0.0, **bstats}
        per_run_rows.append(row)
        print(f"[BASELINE] Zero-shot: succ={row['succ_pct']:.2f}% l2={row['l2_mean']:.4f} invalid={row['invalid_pct']:.2f}%")
    except Exception as e:
        print(f"[WARN] Zero-shot baseline failed: {e}")
    overall.update(1)

    if args.user_baseline_dir:
        print(f"[STEP] Baseline: user model @ {args.user_baseline_dir}")
        try:
            bstats = evaluate_checkpoint(args.user_baseline_dir, args.dataset_dir, args.final_eval_limit,
                                         int(FULL_BASE["--min_visual_tokens"]), int(FULL_BASE["--max_visual_tokens"]),
                                         env_filter="desktop")
            row = {"variant": "user_baseline", "seed": 0, "train_time_sec": 0.0, **bstats}
            per_run_rows.append(row)
            print(f"[BASELINE] User baseline: succ={row['succ_pct']:.2f}%")
        except Exception as e:
            print(f"[WARN] User baseline eval failed: {e}")
        overall.update(1)

    # --- Variants × Seeds ---
    for vname in variants:
        for seed in args.seeds:
            rundir = args.base_outdir / vname / f"seed{seed}"
            rundir.mkdir(parents=True, exist_ok=True)

            eval_json = rundir / "eval.json"
            if eval_json.exists():
                try:
                    row = json.load(open(eval_json))
                    per_run_rows.append(row)
                    print(f"[SKIP] {vname}/seed{seed} already evaluated: succ={row['succ_pct']:.2f}%")
                    runs_bar.update(1)
                    overall.update(steps_per_run)  # count train+eval+cleanup as done
                    continue
                except Exception as e:
                    print(f"[WARN] Failed to load {eval_json}: {e}. Re-running...")

            # TRAIN
            print(f"[STEP] Train: variant={vname}, seed={seed}")
            overrides, flags_true, flags_false = VARIANTS[vname]
            cmd = build_cmd(help_text, args.train_script, args.dataset_dir, rundir, seed,
                            FULL_BASE, flags_true, flags_false, overrides,
                            args.train_dataset, args.train_json, args.steps_per_epoch)
            t0 = time.time()
            rc = run(cmd, cwd=rundir)
            t_train = time.time() - t0
            print(f"[DONE] Train in {t_train:.1f}s (rc={rc})")
            overall.update(1)
            print("[GPU] Sleeping 5s for cleanup...")
            time.sleep(5)

            if rc != 0:
                print(f"[ERROR] Training failed for {vname} seed {seed} (rc={rc}). Skipping eval/cleanup.")
                runs_bar.update(1)
                overall.update(2)  # still tick eval+cleanup to keep overall bar moving
                continue

            # EVAL
            ckpt = find_ckpt(rundir)
            if not ckpt:
                print(f"[ERROR] No checkpoint found in {rundir}. Skipping eval/cleanup.")
                runs_bar.update(1)
                overall.update(2)
                continue

            print(f"[STEP] Eval: {vname}/seed{seed} @ {ckpt}")
            estats = evaluate_checkpoint(
                model_dir_or_id=str(ckpt),
                dataset_dir=args.dataset_dir,
                limit=int(args.final_eval_limit),
                min_visual_tokens=int(FULL_BASE["--min_visual_tokens"]),
                max_visual_tokens=int(FULL_BASE["--max_visual_tokens"]),
                env_filter="desktop",
            )
            row = {
                "variant": vname,
                "seed": seed,
                **estats,
                "train_time_sec": t_train,
            }
            per_run_rows.append(row)
            json.dump(row, open(eval_json, "w"), indent=2)
            print(f"[RESULT] {vname}/seed{seed}: succ={row['succ_pct']:.2f}% l2={row['l2_mean']:.4f} invalid={row['invalid_pct']:.2f}%")
            overall.update(1)

            # CLEANUP
            if args.keep_global_best_ckpt:
                # keep only the best overall
                # compare on succ_pct; delete non-best ckpts
                # NOTE: we can’t compare across different limits here, but all runs share same limit
                if row["succ_pct"] > (global_best[0] if global_best[0] is not None else -1):
                    # delete previous best if any
                    if global_best[1] is not None and global_best[1].exists():
                        try:
                            print(f"[CLEAN] Deleting previous global best: {global_best[1]}")
                            shutil.rmtree(global_best[1], ignore_errors=True)
                        except Exception:
                            pass
                    global_best = (row["succ_pct"], ckpt)
                    print(f"[BEST] New global best {row['succ_pct']:.2f}% @ {ckpt}")
                else:
                    try:
                        sz = human_size(dir_size_bytes(ckpt))
                        print(f"[CLEAN] Removing ckpt ({sz}) {ckpt}")
                        shutil.rmtree(ckpt, ignore_errors=True)
                    except Exception:
                        pass
            else:
                try:
                    sz = human_size(dir_size_bytes(ckpt))
                    print(f"[CLEAN] Removing ckpt ({sz}) {ckpt}")
                    shutil.rmtree(ckpt, ignore_errors=True)
                except Exception:
                    pass

            cleanup_run_artifacts(rundir, keep_best_dir=False, remove_tb=True)
            overall.update(1)
            runs_bar.update(1)

    # --- Write outputs ---
    # merge with existing, de-dup
    all_rows = list(per_run_rows)
    seen = set((r["variant"], r["seed"]) for r in all_rows)
    for (v, s), r in existing.items():
        if (v, s) not in seen:
            all_rows.append(r)
    if all_rows:
        all_rows = sorted(all_rows, key=lambda x: (x["variant"], x["seed"]))
        with open(per_run_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader(); w.writerows(all_rows)
        print(f"[WRITE] {per_run_csv} ({len(all_rows)} runs total)")
    overall.update(1)

    # summaries
    by_variant: Dict[str, Dict[str, List[float]]] = {}
    for r in all_rows:
        v = r["variant"]
        if v.startswith("baseline"):
            continue
        by_variant.setdefault(v, {"succ": [], "inv": [], "l2": []})
        by_variant[v]["succ"].append(r["succ_pct"])
        by_variant[v]["inv"].append(r["invalid_pct"])
        by_variant[v]["l2"].append(r["l2_mean"])

    summary_rows = []
    for v, d in by_variant.items():
        if d["succ"]:
            m_succ, ci_succ = ci95(d["succ"])
            m_inv, ci_inv = ci95(d["inv"])
            m_l2, ci_l2 = ci95(d["l2"])
            summary_rows.append({
                "variant": v,
                "variant_pretty": PRETTY.get(v, v),
                "succ_pct_mean": m_succ,
                "succ_pct_ci": ci_succ,
                "l2_mean": m_l2,
                "l2_ci": ci_l2,
                "invalid_pct_mean": m_inv,
                "invalid_pct_ci": ci_inv,
                "n_seeds": len(d["succ"]),
            })

    if summary_rows:
        with open(summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader(); w.writerows(summary_rows)
        json.dump(summary_rows, open(summary_json, "w"), indent=2)
        print(f"[WRITE] {summary_csv}\n[WRITE] {summary_json}")
    overall.update(1)

    # LaTeX
    if summary_rows:
        def sort_key(r):
            order = list(PRETTY.keys())
            try:
                return order.index(r["variant"]) if r["variant"] in order else 999
            except Exception:
                return 999
        rows_sorted = sorted(summary_rows, key=sort_key)

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
        for r in rows_sorted:
            lines.append(
                f"{r['variant_pretty']} & "
                f"{fmt_pm(r['succ_pct_mean'], r['succ_pct_ci'], 2)} & "
                f"{fmt_pm(r['l2_mean'], r['l2_ci'], 4)} & "
                f"{fmt_pm(r['invalid_pct_mean'], r['invalid_pct_ci'], 2)} \\\\"
            )
        lines += [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Ablations on the ScreenSpot desktop subset. Mean $\\pm$ 95\\% CI over seeds. Greedy final evaluation with identical protocol across variants.}",
            "\\label{tab:ablation}",
            "\\end{table}",
        ]
        open(tex_path, "w").write("\n".join(lines))
        print(f"[WRITE] {tex_path}")
    overall.update(1)

    # Keep-best report
    if args.keep_global_best_ckpt:
        if global_best[1] is not None:
            sz = human_size(dir_size_bytes(global_best[1]))
            print(f"[KEEP] Global best kept: {global_best[1]} ({sz}), succ={global_best[0]:.2f}%")
        else:
            print("[KEEP] No ckpt kept.")
    overall.close(); runs_bar.close()

if __name__ == "__main__":
    main()
