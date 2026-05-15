"""
Compute prompt token-length statistics for SpotAgent / GSV RL datasets.

Usage:
    python stat_prompt_length.py \
        --parquet /mnt/sh/mmvision/home/jonahli/data_agent/rl/SpotAgent/train.parquet \
        --model   /mnt/sh/mmvision/home/jonahli/init_ckpt/vllm/Qwen2.5-VL-3B-Instruct \
        --max_length 4096 \
        --num_workers 64

The script tokenizes {system + user + image} prompts exactly as rl_dataset._build_messages does,
and prints percentile / over-limit statistics.
"""

import argparse
import os
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _open_image(image_entry) -> Image.Image:
    """Open a single image entry (dict with 'bytes', 'image_url', or 'path')."""
    if isinstance(image_entry, dict):
        if "bytes" in image_entry:
            return Image.open(BytesIO(image_entry["bytes"])).convert("RGB")
        if "image_url" in image_entry:
            return Image.open(image_entry["image_url"]).convert("RGB")
        if "path" in image_entry:
            return Image.open(image_entry["path"]).convert("RGB")
        if "image" in image_entry and isinstance(image_entry["image"], Image.Image):
            return image_entry["image"].convert("RGB")
    if isinstance(image_entry, Image.Image):
        return image_entry.convert("RGB")
    raise TypeError(f"Unsupported image entry type: {type(image_entry)}")


def build_structured_messages(prompt: list[dict], pil_images: list[Image.Image]) -> list[dict]:
    """Convert messages with '<image>' text placeholders to structured content lists.

    Qwen2.5-VL apply_chat_template needs:
        {"type": "image", "image": PIL_image}  ← actual PIL so patch count is correct
    """
    import copy, re
    messages = copy.deepcopy(prompt)
    image_offset = 0
    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        segments = re.split(r"(<image>)", content)
        segments = [s for s in segments if s]
        content_list = []
        for seg in segments:
            if seg == "<image>" and image_offset < len(pil_images):
                content_list.append({"type": "image", "image": pil_images[image_offset]})
                image_offset += 1
            else:
                content_list.append({"type": "text", "text": seg})
        msg["content"] = content_list
    return messages


# ---------------------------------------------------------------------------
# worker initialiser (load processor once per process)
# ---------------------------------------------------------------------------

_processor = None

def _init_worker(model_path: str):
    global _processor
    _processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)


def _worker(row: dict) -> int:
    global _processor
    try:
        prompt = row["prompt"]  # original messages with "<image>" as plain text
        pil_images = load_pil_images(row.get("images") or [])

        # Pass original messages (with "<image>" text) to apply_chat_template.
        # Qwen2.5-VL's chat template inserts vision token placeholders when it
        # sees image content or when images are provided to the processor call.
        raw = _processor.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = _processor(
            text=[raw],
            images=pil_images if pil_images else None,
            return_tensors="pt",
        )
        return int(inputs["input_ids"].shape[1])
    except Exception as e:
        return -1  # mark as failed


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        nargs="+",
        default=["/mnt/sh/mmvision/home/jonahli/data_agent/rl/SpotAgent/train.parquet"],
        help="One or more parquet file paths.",
    )
    parser.add_argument(
        "--model",
        default="/mnt/sh/mmvision/home/jonahli/init_ckpt/vllm/Qwen2.5-VL-3B-Instruct",
        help="HuggingFace model path for tokenizer/processor.",
    )
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="If set, randomly sample this many rows for faster estimation.",
    )
    args = parser.parse_args()

    # load all parquets
    dfs = [pd.read_parquet(p) for p in args.parquet]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(df)}")

    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=42).reset_index(drop=True)
        print(f"Sampled {len(df)} rows for estimation.")

    rows = df.to_dict("records")

    lengths = []
    failed = 0

    from multiprocessing import Pool
    with Pool(processes=args.num_workers, initializer=_init_worker, initargs=(args.model,)) as pool:
        for length in tqdm(pool.imap(_worker, rows, chunksize=8), total=len(rows), desc="tokenizing"):
            if length < 0:
                failed += 1
            else:
                lengths.append(length)

    lengths = np.array(lengths)
    print(f"\n{'='*50}")
    print(f"Processed:  {len(lengths)}  |  Failed: {failed}")
    print(f"{'='*50}")
    print(f"min        : {lengths.min()}")
    print(f"max        : {lengths.max()}")
    print(f"mean       : {lengths.mean():.1f}")
    print(f"median (p50): {np.percentile(lengths, 50):.0f}")
    print(f"p75        : {np.percentile(lengths, 75):.0f}")
    print(f"p90        : {np.percentile(lengths, 90):.0f}")
    print(f"p95        : {np.percentile(lengths, 95):.0f}")
    print(f"p99        : {np.percentile(lengths, 99):.0f}")
    print(f"{'='*50}")
    over = (lengths > args.max_length).sum()
    print(f"Over {args.max_length} tokens: {over} / {len(lengths)} ({100*over/len(lengths):.1f}%)")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
