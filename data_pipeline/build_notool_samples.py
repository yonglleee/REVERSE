#!/usr/bin/env python3
"""
build_notool_samples.py — Build "think → answer" notool SFT samples from MP16-Pro
data, using deterministic templates for the <think> segment.

Purpose (§6.4 v5 plan)
----------------------
Coldstart v4 parquet has 0 notool samples, so the model learns "always call a tool".
This script generates N notool samples with the format:

    system: <same coldstart v5 system prompt>
    user:   <image> + same user prompt as coldstart
    assistant: <think>...visual-cue-based reasoning...</think>
               <answer>Country, City, Lat, Lon</answer>

These samples are mixed into coldstart v5 training data alongside the cleaned
with-tool samples, teaching the model the "self-confident → skip tools" behavior.

Samples are drawn from MP16-Pro coldstart CSVs (test_filtered_part*.csv), which
contain ground-truth city/country/lat/lon. We do NOT filter by "MP16pro SFT can
answer <1km" because the think we generate is deterministic/templated and does
not depend on the model's ability — we're teaching the FORMAT, not the knowledge.

Usage
-----
    python3 data_pipeline/build_notool_samples.py \
        --csv_dir  /mnt/sh/mmvision/home/jonahli/data_agent/coldstart/raw \
        --parts    00 01 02 \
        --n_samples 1500 \
        --output   /mnt/sh/mmvision/home/jonahli/data_agent/sft/coldstart/train_notool_templated.parquet

Schema (matches coldstart train parquet: [messages, images]):
    messages: list[{role, content}]  (system, user, assistant)
    images:   list[{image: path}]    (1 image per sample)
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys
from typing import Dict, List

import pandas as pd

# ── Coldstart v5 system prompt (copy of build_sft_coldstart.py SYSTEM_PROMPT) ────
# Keep in sync with data_pipeline/build_sft_coldstart.py.
SYSTEM_PROMPT = (
    "You are a geolocation expert. Given an image, identify its location.\n"
    "You have three tools:\n"
    "1. `image_search_tool`: Reverse image search using a cropped region. Best for "
    "distinctive landmarks, buildings, or scenes. Returns matching web pages.\n"
    "2. `text_search_tool`: Search the web with natural language queries. Use for "
    "visible text/signs, landmark names, or any clues found from image search results.\n"
    "3. `image_zoom_in_tool`: Zoom into a region to read text/inscriptions that are "
    "too small at full scale.\n\n"
    "Decision rules:\n"
    "  • If you are HIGHLY CONFIDENT of the exact location (world-famous landmark, "
    "legible place-name text, familiar scene), provide your final <answer> DIRECTLY "
    "without any tool call. Explicitly state in <think> why no tool is needed.\n"
    "  • Distinctive landmark or scene visible (but uncertain of exact coords) → use `image_search_tool`\n"
    "  • Text/signs already legible → use `text_search_tool` directly\n"
    "  • Text/signs too small to read → use `image_zoom_in_tool` first, then `text_search_tool`\n"
    "  • image_search returns a landmark/location name → follow up with `text_search_tool`\n"
    "  • Do NOT use `image_zoom_in_tool` before `image_search_tool` — zoom does not improve image search\n"
    "  • When in doubt, use a tool. Only skip tools when you are certain.\n\n"
    "For EVERY response, first enclose your reasoning in <think> </think> tags, then output EXACTLY ONE of:\n"
    '<tool_call>{"name": "image_search_tool", "arguments": {"bbox_2d": [x1, y1, x2, y2], "goal": "..."}}</tool_call>\n'
    "or:\n"
    '<tool_call>{"name": "text_search_tool", "arguments": {"query": "your query"}}</tool_call>\n'
    "or (parallel):\n"
    '<tool_call>{"name": "text_search_tool", "arguments": {"query": ["query one", "query two"]}}   </tool_call>\n'
    "or:\n"
    '<tool_call>{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [x1, y1, x2, y2]}}</tool_call>\n'
    "or your final answer in <answer> tags (when confident without needing a tool).\n\n"
    "After receiving tool results, output on its own line: <useful>[i, j, ...]</useful> "
    "listing the 1-based indices of results that match this specific image (i.e., mention "
    "the actual location, landmark, or geographic region shown in the image). Results about "
    "a different place are NOT useful even if they contain geographic information. "
    "Output <useful>[]</useful> if none match. Example: after receiving search results, your "
    "response must start with <think>...</think> then immediately <useful>[1, 3]</useful>.\n\n"
    "Final answer format: <answer>Country, City, Latitude, Longitude</answer>. "
    "e.g. <answer>Italy, Golfo Arnaci, 40.9606, 9.5873</answer>"
)

# User prompt template (matches first user message in coldstart v4 parquet).
USER_PROMPT = (
    "<image>\n"
    "Analyze the architectural styles, vegetation, street infrastructure, and cultural markers in this image. "
    "Based on these visual cues, determine the location.\n\n"
    "Answer strictly in the following format:\n"
    "Country, City, Latitude, Longitude. You FIRST think about the reasoning process as an internal monologue "
    "and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. "
    "Wrap your final answer in <answer> tags in the format: <answer>Country, City, Latitude, Longitude</answer>. "
    "e.g. <answer>Italy, Golfo Arnaci, 40.9606, 9.5873</answer>"
)

# Think templates — deterministic, diverse, emphasizing "confident, no tool needed".
# Each template takes (city, country, region, continent) via .format().
_THINK_TEMPLATES = [
    "Analyzing the visual cues in this image — the architecture, vegetation, street "
    "infrastructure, and cultural markers — I can identify this as a scene from {city}, {country}. "
    "The characteristic features are consistent with the {region} region. "
    "I am confident in this identification based on visual priors; no tool call needed.",

    "The image shows distinctive features characteristic of {city}, {country}. "
    "The combination of architectural style, environmental context, and regional cultural "
    "markers matches this location in {continent}. I can provide the answer directly without "
    "needing reverse image search.",

    "Based on the visual composition — including the buildings, terrain, and overall ambience — "
    "this scene is from {country}, specifically the {city} area. "
    "The features align with what I know about this location in {region}. "
    "No tool call needed; I can answer directly.",

    "I recognize this as {city}, {country}. The visual style of the architecture and "
    "surroundings, together with the regional context of {region}, {continent}, gives me "
    "high confidence in this identification without needing to verify with external search.",

    "Looking at the image, the scenery, vegetation pattern, and any visible infrastructure "
    "details point to {city}, {country}. This is consistent with the {continent} region "
    "known as {region}. I will answer directly based on this visual assessment.",
]

_THINK_TEMPLATES_NO_CITY = [
    # For samples missing a specific city — use country/region only.
    "Analyzing the image — the architecture, vegetation, and environmental context — "
    "I can identify this as a scene from {country}, likely in the {region} region of {continent}. "
    "I am confident enough to provide the answer based on visual priors without a tool call.",

    "The visual features of this image — landscape, buildings, and cultural markers — are "
    "consistent with {country} in {continent}. Based on my recognition of this region, "
    "I can provide coordinates directly.",
]


def _fmt_think(row: pd.Series, rng: random.Random) -> str:
    city = str(row.get("city", "") or "").strip()
    country = str(row.get("country", "") or "").strip()
    region = str(row.get("region", "") or "").strip() or country
    continent = str(row.get("continent", "") or "").strip() or "the world"

    if city and city.lower() not in ("none", "nan", "null"):
        tmpl = rng.choice(_THINK_TEMPLATES)
        return tmpl.format(city=city, country=country, region=region, continent=continent)
    else:
        tmpl = rng.choice(_THINK_TEMPLATES_NO_CITY)
        return tmpl.format(country=country, region=region, continent=continent)


def _fmt_answer(row: pd.Series) -> str:
    country = str(row.get("country", "") or "Unknown").strip()
    city = str(row.get("city", "") or "").strip() or country
    lat = float(row.get("LAT", 0))
    lon = float(row.get("LON", 0))
    # Round to 4 decimals, matching Kimi coldstart answer format
    return f"<answer>{country}, {city}, {lat:.4f}, {lon:.4f}</answer>"


def build_sample(row: pd.Series, rng: random.Random) -> Dict:
    think = _fmt_think(row, rng)
    answer = _fmt_answer(row)
    assistant_content = f"<think>\n{think}\n</think>\n\n{answer}"
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": USER_PROMPT},
        {"role": "assistant", "content": assistant_content},
    ]
    image_path = str(row.get("path", "")).strip()
    images = [{"image": image_path}]
    return {"messages": messages, "images": images}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv_dir", default="/mnt/sh/mmvision/home/jonahli/data_agent/coldstart/raw",
                    help="Directory containing test_filtered_part*.csv")
    ap.add_argument("--parts", nargs="*", default=None,
                    help="Parts to include (e.g. 00 01 02). Default: all found.")
    ap.add_argument("--n_samples", type=int, default=1500,
                    help="Target number of notool samples to generate")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for sampling + template choice")
    ap.add_argument("--output", required=True, help="Output parquet path")
    ap.add_argument("--check_images", action="store_true",
                    help="Skip samples whose image file doesn't exist")
    args = ap.parse_args()

    # Auto-detect parts
    if args.parts:
        parts = args.parts
    else:
        parts = sorted([
            os.path.basename(f).replace("test_filtered_part", "").replace(".csv", "")
            for f in glob.glob(os.path.join(args.csv_dir, "test_filtered_part*.csv"))
        ])
    print(f"[notool] Using parts: {parts}")

    # Load all CSVs
    dfs = []
    for p in parts:
        csv_path = os.path.join(args.csv_dir, f"test_filtered_part{p}.csv")
        if not os.path.exists(csv_path):
            print(f"  SKIP: {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        df["part"] = p
        dfs.append(df)
        print(f"  loaded part{p}: {len(df)} rows")
    if not dfs:
        print("No CSVs loaded. Aborting.")
        sys.exit(1)

    all_df = pd.concat(dfs, ignore_index=True)
    print(f"[notool] Total rows: {len(all_df)}")

    # Filter rows missing critical fields
    required = ["LAT", "LON", "country", "path"]
    before = len(all_df)
    all_df = all_df.dropna(subset=required)
    all_df = all_df[all_df["country"].astype(str).str.strip().str.lower().isin(["", "none", "null", "nan"]) == False]
    print(f"[notool] After dropping rows w/ missing fields: {len(all_df)} (removed {before - len(all_df)})")

    # Optional: check image existence
    if args.check_images:
        before = len(all_df)
        all_df = all_df[all_df["path"].apply(os.path.exists)]
        print(f"[notool] After image-existence check: {len(all_df)} (removed {before - len(all_df)})")

    # Random sample
    rng = random.Random(args.seed)
    n_target = min(args.n_samples, len(all_df))
    sampled = all_df.sample(n=n_target, random_state=args.seed).reset_index(drop=True)
    print(f"[notool] Sampled {len(sampled)} rows")

    # Build SFT samples
    rows_out = []
    for _, row in sampled.iterrows():
        try:
            rows_out.append(build_sample(row, rng))
        except Exception as e:
            print(f"  WARN: build_sample failed for IMG_ID={row.get('IMG_ID')}: {e}")
            continue

    out_df = pd.DataFrame(rows_out)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df.to_parquet(args.output, index=False)
    print(f"[notool] Wrote {len(out_df)} samples to {args.output}")

    # Show one example
    if len(out_df):
        print("\n[notool] Example sample:")
        ex = out_df.iloc[0]
        for m in ex["messages"]:
            content = m["content"]
            if len(content) > 300:
                content = content[:300] + "..."
            print(f"  [{m['role']}] {content}")
        print(f"  images: {ex['images']}")


if __name__ == "__main__":
    main()
