import os
import ast
import json
import argparse
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


def within_bbox(point, bbox):
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, help="Path to DATA_DIR containing ScreenSpot/")
    parser.add_argument("--model_id", default="showlab/ShowUI-2B", help="Model id or local ckpt dir")
    parser.add_argument("--split", default="hf_test_full", help="ScreenSpot split json name")
    parser.add_argument("--limit", type=int, default=200, help="Max samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for batched generation")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max new tokens during generation")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for model (GPU/PyTorch 2.1+)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Inference-friendly settings
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    min_pixels = 256*28*28
    max_pixels = 1344*28*28

    processor = AutoProcessor.from_pretrained(
        args.model_id if os.path.isdir(args.model_id) else "showlab/ShowUI-2B",
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=dtype,
    )
    model.eval()

    # Optional compile (helps when evaluating many steps)
    if args.compile and device == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="max-autotune")
        except Exception:
            pass

    meta_path = os.path.join(args.dataset_dir, "ScreenSpot", "metadata", f"{args.split}.json")
    with open(meta_path) as f:
        items = json.load(f)

    N = min(args.limit, len(items))
    ok = 0

    for start in tqdm(range(0, N, args.batch_size), desc="ScreenSpot eval"):
        end = min(start + args.batch_size, N)
        batch = items[start:end]

        # Prepare batch inputs
        imgs = []
        img_sizes = []
        messages_list = []
        for item in batch:
            img = Image.open(os.path.join(args.dataset_dir, "ScreenSpot", "images", item["img_url"]))
            img = img.convert("RGB")
            if "img_size" in item and isinstance(item["img_size"], (list, tuple)) and len(item["img_size"]) == 2:
                img_w, img_h = item["img_size"][0], item["img_size"][1]
            else:
                img_w, img_h = img.size
            img_sizes.append((img_w, img_h))
            imgs.append(img)
            messages_list.append([
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1.",
                        },
                        {"type": "image", "image": img, "min_pixels": min_pixels, "max_pixels": max_pixels},
                        {"type": "text", "text": item["task"]},
                    ],
                }
            ])

        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
        inputs = processor(text=texts, images=imgs, padding=True, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                num_beams=1,
                do_sample=False,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        gen = out[:, inputs.input_ids.shape[1]:]
        pred_strs = processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        # Compute accuracy per sample
        for pred_str, item, (img_w, img_h) in zip(pred_strs, batch, img_sizes):
            try:
                pred = ast.literal_eval(pred_str)
                x, y, w, h = item["bbox"]
                gt = [x / img_w, y / img_h, (x + w) / img_w, (y + h) / img_h]
                ok += 1 if within_bbox(pred, gt) else 0
            except Exception:
                pass

    print(f"ScreenSpot Success Rate ({N} samples): {ok/N:.3f}")


if __name__ == "__main__":
    main()


