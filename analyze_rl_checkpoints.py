import os, re, json, math, argparse

from pathlib import Path

from typing import List, Dict, Tuple


import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


# ---------- Utility: coordinate parsing (copied/adapted from training script) ----------
import ast
def parse_coord(output_text: str, img_size: Tuple[int, int] = None) -> Tuple[float, float]:
    text = (output_text or "").strip()
    bracket = re.search(r"\[[^\]]+\]", text)
    if bracket:
        text = bracket.group(0)
    try:
        xy = ast.literal_eval(text)
        if isinstance(xy, (list, tuple)):
            def to_num(v):
                if isinstance(v, str) and v.strip().endswith('%'):
                    return float(v.strip()[:-1]) / 100.0
                return float(v)
            nums = [to_num(v) for v in xy if isinstance(v, (int, float, str))]
            if len(nums) >= 4 and img_size is not None:
                x, y, w, h = nums[:4]
                iw, ih = max(1.0, float(img_size[0])), max(1.0, float(img_size[1]))
                if max(x, y, w, h) > 1.0001:
                    cx = (x + w / 2.0) / iw
                    cy = (y + h / 2.0) / ih
                else:
                    cx = x + w / 2.0
                    cy = y + h / 2.0
                return min(1.0, max(0.0, cx)), min(1.0, max(0.0, cy))
            if len(nums) >= 2:
                x, y = nums[0], nums[1]
                if img_size is not None and (abs(x) > 1.0001 or abs(y) > 1.0001):
                    iw, ih = max(1.0, float(img_size[0])), max(1.0, float(img_size[1]))
                    x, y = x/iw, y/ih
                return min(1.0, max(0.0, x)), min(1.0, max(0.0, y))
    except Exception:
        pass
    try:
        tokens = re.findall(r"[-+]?\d*\.?\d+%?", text)
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
                x, y = x/iw, y/ih
            return min(1.0, max(0.0, x)), min(1.0, max(0.0, y))
    except Exception:
        pass
    return float("nan"), float("nan")


# ---------- Metrics helpers ----------
def inside_bbox(px: float, py: float, bbox: Tuple[int,int,int,int], img_w: int, img_h: int) -> bool:
    x, y, w, h = bbox
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    cx, cy = px * img_w, py * img_h
    return (x1 <= cx <= x2) and (y1 <= cy <= y2)


def dist_to_box(px: float, py: float, bbox: Tuple[int,int,int,int], img_w: int, img_h: int) -> float:
    """Euclidean distance in pixels from point to nearest point on (or inside) bbox. 0 if inside."""
    x, y, w, h = bbox
    x1, y1, x2, y2 = x, y, x + w, y + h
    cx, cy = px * img_w, py * img_h
    dx = 0.0 if x1 <= cx <= x2 else (x1 - cx) if cx < x1 else (cx - x2)
    dy = 0.0 if y1 <= cy <= y2 else (y1 - cy) if cy < y1 else (cy - y2)
    return math.hypot(dx, dy)


def within_r_or_bbox(px: float, py: float, bbox: Tuple[int,int,int,int], img_w: int, img_h: int, r_px: float) -> bool:
    if inside_bbox(px, py, bbox, img_w, img_h):
        return True
    # center-distance tolerance
    x, y, w, h = bbox
    cx0, cy0 = x + w/2.0, y + h/2.0
    cx, cy = px * img_w, py * img_h
    return math.hypot(cx - cx0, cy - cy0) <= r_px


# ---------- Data loading ----------
def load_screenspot_items(dataset_dir: str) -> List[dict]:
    meta_path = Path(dataset_dir) / "ScreenSpot" / "metadata" / "hf_test_full.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")
    with open(meta_path) as f:
        items = json.load(f)
    return items


# ---------- Evaluation on a checkpoint ----------
@torch.no_grad()
def evaluate_checkpoint(ckpt_dir: Path, dataset_dir: str, limit: int, device: str,
                        min_pixels: int, max_pixels: int) -> Dict[str, float]:
    # Load processor/model from checkpoint folder
    processor = AutoProcessor.from_pretrained(str(ckpt_dir), min_pixels=min_pixels, max_pixels=max_pixels)
    model = Qwen2VLForConditionalGeneration.from_pretrained(str(ckpt_dir),
                                                            torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32))
    model.to(device)
    model.eval()


    items = load_screenspot_items(dataset_dir)
    # Prefer desktop-only if available
    items = [it for it in items if str(it.get("split","")).lower() == "desktop"] or items
    N = min(limit, len(items)) if (limit and limit > 0) else len(items)

    succ_bbox = 0
    succ_r = {4:0, 8:0, 16:0, 32:0}
    dtb_all = []

    for i in range(N):
        it = items[i]
        img_path = Path(dataset_dir) / "ScreenSpot" / "images" / it["img_url"]
        if not img_path.exists():
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        img_w, img_h = (it["img_size"][0], it["img_size"][1]) if "img_size" in it else img.size
        bbox = tuple(it["bbox"])  # [x,y,w,h]

        # Prepare chat
        messages = [
            {"role":"user","content":[
                {"type":"text","text":"Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1. Return exactly two numbers in square brackets like [x, y] with both x and y in [0,1]. Do not include any other text."},
                {"type":"image","image":img,"min_pixels":min_pixels,"max_pixels":max_pixels},
                {"type":"text","text":it["task"]}
            ]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt").to(device)

        try:
            out = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                num_beams=1,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
            )
            gen = out[:, inputs.input_ids.shape[1]:]
            pred_str = processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            px, py = parse_coord(pred_str, img_size=(img_w, img_h))
            if any(map(lambda v: math.isnan(v), [px,py])):
                continue
        except Exception:
            continue

        if inside_bbox(px, py, bbox, img_w, img_h):
            succ_bbox += 1
        for r in succ_r.keys():
            if within_r_or_bbox(px, py, bbox, img_w, img_h, r_px=float(r)):
                succ_r[r] += 1
        dtb_all.append(dist_to_box(px, py, bbox, img_w, img_h))

    eps = max(1, N)
    result = {
        "n": N,
        "success_bbox": succ_bbox / eps,
        "success_r4": succ_r[4] / eps,
        "success_r8": succ_r[8] / eps,
        "success_r16": succ_r[16] / eps,
        "success_r32": succ_r[32] / eps,
        "dtb_px_mean": float(np.mean(dtb_all)) if dtb_all else float("nan"),
        "auc_hit": float(np.mean([
            succ_bbox/eps,
            succ_r[4]/eps, succ_r[8]/eps, succ_r[16]/eps, succ_r[32]/eps
        ])),
        "_dtb_list": dtb_all,  # used for the CDF plot
    }
    return result


# ---------- Plotting ----------
def plot_learning_curve(df: pd.DataFrame, outpath: Path):
    plt.figure(figsize=(7,4))
    plt.plot(df["epoch"], df["success_bbox"])
    plt.xlabel("Epoch")
    plt.ylabel("Success@bbox")
    plt.title("Validation Success@bbox across epochs")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_success_at_r(df: pd.DataFrame, outpath: Path):
    plt.figure(figsize=(7,4))
    for r in [4,8,16,32]:
        plt.plot(df["epoch"], df[f"success_r{r}"], label=f"@{r}px")
    plt.xlabel("Epoch")
    plt.ylabel("Success@r")
    plt.title("Success@r across epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_dtb_cdf(dtb_a: List[float], dtb_b: List[float], label_a: str, label_b: str, outpath: Path):
    def cdf(data):
        xs = np.sort(np.array(data, dtype=np.float32))
        ys = np.arange(1, len(xs)+1)/len(xs) if len(xs) else np.array([])
        return xs, ys
    plt.figure(figsize=(6,4))
    if dtb_a:
        xs, ys = cdf(dtb_a)
        plt.plot(xs, ys, label=label_a)
    if dtb_b:
        xs, ys = cdf(dtb_b)
        plt.plot(xs, ys, label=label_b)
    plt.xlabel("Distance-to-Box (px)")
    plt.ylabel("CDF")
    plt.title("Pointing Error Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# ---------- Main ----------
def discover_checkpoints(root: Path) -> List[Tuple[int, Path]]:
    out = []
    for p in root.iterdir():
        if p.is_dir():
            m = re.match(r"rl_ckpt_epoch(\d+)", p.name)
            if m and (p / "pytorch_model.bin").exists():
                out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints_root", type=str, default="/root/ShowUI/training")
    ap.add_argument("--dataset_dir", type=str, required=True, help="Root directory that contains ScreenSpot/{images,metadata}")
    ap.add_argument("--limit", type=int, default=200, help="Max examples to evaluate per checkpoint (0 = all)")
    ap.add_argument("--min_visual_tokens", type=int, default=256)
    ap.add_argument("--max_visual_tokens", type=int, default=1344)
    ap.add_argument("--out_dir", type=str, default="/root/ShowUI/training/analysis")
    ap.add_argument("--only_base", action="store_true", help="Evaluate only the base model at --base_model_dir")
    ap.add_argument("--base_model_dir", type=str, default=None, help="Directory of the base model to evaluate (requires config/tokenizer/model files)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    min_pixels = args.min_visual_tokens * 28 * 28
    max_pixels = args.max_visual_tokens * 28 * 28

    if args.only_base:
        if not args.base_model_dir:
            raise RuntimeError("--only_base requires --base_model_dir to be set")
        ckpts = [(0, Path(args.base_model_dir))]
    else:
        ckpts = discover_checkpoints(Path(args.checkpoints_root))
        if not ckpts:
            raise RuntimeError(f"No checkpoints found in {args.checkpoints_root}")
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    dtb_by_epoch = {}
    for ep, cdir in ckpts:
        print(f"Evaluating epoch {ep} at {cdir} ...")
        res = evaluate_checkpoint(cdir, args.dataset_dir, args.limit, device, min_pixels, max_pixels)
        row = {
            "epoch": ep,
            "n": res["n"],
            "success_bbox": res["success_bbox"],
            "success_r4": res["success_r4"],
            "success_r8": res["success_r8"],
            "success_r16": res["success_r16"],
            "success_r32": res["success_r32"],
            "auc_hit": res["auc_hit"],
            "dtb_px_mean": res["dtb_px_mean"],
        }
        rows.append(row)
        dtb_by_epoch[ep] = res.get("_dtb_list", [])

    df = pd.DataFrame(rows).sort_values("epoch")
    csv_path = Path(args.out_dir) / "metrics_by_epoch.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    plot_learning_curve(df, Path(args.out_dir) / "learning_curve_success_bbox.png")
    plot_success_at_r(df, Path(args.out_dir) / "success_at_r_by_epoch.png")

    # CDF: compare first vs best epoch
    first_ep = df["epoch"].iloc[0]
    best_idx = df["success_bbox"].idxmax()
    best_ep = int(df.loc[best_idx, "epoch"])
    plot_dtb_cdf(
        dtb_by_epoch.get(first_ep, []),
        dtb_by_epoch.get(best_ep, []),
        label_a=f"Epoch {first_ep}",
        label_b=f"Epoch {best_ep} (best)",
        outpath=Path(args.out_dir) / "dtb_cdf_first_vs_best.png"
    )
    print(f"Figures saved under {args.out_dir}")


if __name__ == "__main__":
    main()


