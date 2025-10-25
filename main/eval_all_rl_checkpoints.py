import os
import json
import argparse
import subprocess
from pathlib import Path


def discover_checkpoints(root: str) -> list[str]:
    root_path = Path(root)
    ckpts = []
    for p in root_path.iterdir():
        if p.is_dir() and p.name.startswith("rl_ckpt"):
            ckpts.append(str(p.resolve()))
    # Sort numerically if suffix is a number
    def _key(s: str) -> tuple:
        name = Path(s).name
        try:
            n = int(name.replace("rl_ckpt", ""))
        except Exception:
            n = 1_000_000
        return (n, name)
    ckpts.sort(key=_key)
    return ckpts


def run_eval(evaluator: str, data_root: str, model_path: str, eval_flags: list[str], extra: list[str]) -> tuple[bool, str]:
    cmd = [
        "python3",
        evaluator,
        "--data_root",
        data_root,
        "--model_id",
        model_path,
    ] + eval_flags + extra
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return True, out
    except subprocess.CalledProcessError as e:
        return False, e.output


def main():
    parser = argparse.ArgumentParser(description="Evaluate all rl_ckpt* on ScreenSpot-V2 and Pro (desktop-only)")
    parser.add_argument("--repo_root", default=str(Path(__file__).resolve().parents[1]), help="ShowUI repo root")
    parser.add_argument("--data_root", required=True, help="Path to ~/showui_data")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--limit_v2", type=int, default=0)
    parser.add_argument("--limit_pro", type=int, default=0)
    parser.add_argument("--output", default="eval_rl_ckpts_results.json")
    args = parser.parse_args()

    evaluator = str(Path(args.repo_root) / "main" / "eval_screenspot_v2_pro.py")
    if not os.path.exists(evaluator):
        raise FileNotFoundError(f"Evaluator not found: {evaluator}")

    ckpts = discover_checkpoints(args.repo_root)
    results = []
    for ckpt in ckpts:
        record = {"checkpoint": ckpt}
        flags_common = [
            "--batch_size", str(args.batch_size),
            "--max_new_tokens", str(args.max_new_tokens),
            "--limit_v2", str(args.limit_v2),
            "--limit_pro", str(args.limit_pro),
        ]

        ok_v2, out_v2 = run_eval(
            evaluator,
            args.data_root,
            ckpt,
            ["--eval_v2"],
            flags_common,
        )
        record["v2_ok"] = ok_v2
        record["v2_log"] = out_v2

        ok_pro, out_pro = run_eval(
            evaluator,
            args.data_root,
            ckpt,
            ["--eval_pro"],
            flags_common,
        )
        record["pro_ok"] = ok_pro
        record["pro_log"] = out_pro

        results.append(record)
        print(f"Evaluated {ckpt}: V2 ok={ok_v2}, Pro ok={ok_pro}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()


