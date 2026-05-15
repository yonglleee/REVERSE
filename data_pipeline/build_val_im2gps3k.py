#!/usr/bin/env python3
"""
build_val_im2gps3k.py — Build fixed 50-row RL val parquet from im2gps3k_50.

Val is permanently decoupled from the coldstart train pool:
  - Source: eval/im2gps3k_50.csv (ground truth) + kimi_im2gps3k_val_v6/G_val.jsonl (annotations)
  - 50 rows, never overlaps with coldstart train, stable forever
  - data_source = "im2gps3k"
  - Requires useful_results in JSONL (re-annotated with SYSTEM_TOOL_LLMCLIENT_ALL3_V2)

Schema identical to train rows except:
  - data_source = "im2gps3k"
  - split = "val"

Usage:
  python data_pipeline/build_val_im2gps3k.py \\
    --csv     /mnt/sh/mmvision/home/jonahli/projects/tusou/eval/im2gps3k_50.csv \\
    --jsonl   /mnt/sh/mmvision/home/jonahli/save/agent/eval/im2gps3k/kimi_im2gps3k_val_v6/G_val.jsonl \\
    --out_dir /mnt/sh/mmvision/home/jonahli/data_agent/rl/coldstart
"""

from __future__ import annotations

import argparse
import json
import os

SYSTEM_PROMPT = (
    "You are a geolocation expert. Given an image, identify its location.\n"
    "You have three tools:\n"
    "1. `image_search_tool`: Reverse image search using a cropped region. Best for distinctive landmarks, buildings, or scenes. Returns matching web pages.\n"
    "2. `text_search_tool`: Search the web with natural language queries. Use for visible text/signs, landmark names, or any clues found from image search results.\n"
    "3. `image_zoom_in_tool`: Zoom into a region to read text/inscriptions that are too small at full scale.\n"
    "\n"
    "Decision rules:\n"
    "  \u2022 Distinctive landmark or scene visible \u2192 use `image_search_tool`\n"
    "  \u2022 Text/signs already legible \u2192 use `text_search_tool` directly\n"
    "  \u2022 Text/signs too small to read \u2192 use `image_zoom_in_tool` first, then `text_search_tool`\n"
    "  \u2022 image_search returns a landmark/location name \u2192 follow up with `text_search_tool`\n"
    "  \u2022 Do NOT use `image_zoom_in_tool` before `image_search_tool` \u2014 zoom does not improve image search\n"
    "\n"
    "Workflow:\n"
    "1. Analyze the image and pick the best tool based on the decision rules above.\n"
    "2. Use results to refine your understanding. Search additional regions or queries if uncertain.\n"
    "3. Once confident, provide your final answer \u2014 do not over-search.\n"
    "\n"
    "For EVERY response, first enclose your reasoning in <think> </think> tags, then output EXACTLY ONE of:\n"
    '<tool_call>{"name": "image_search_tool", "arguments": {"bbox_2d": [x1, y1, x2, y2]}}</tool_call>\n'
    "or:\n"
    '<tool_call>{"name": "text_search_tool", "arguments": {"query": "your query"}}</tool_call>\n'
    "or:\n"
    '<tool_call>{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [x1, y1, x2, y2]}}</tool_call>\n'
    "where bbox_2d is [x1, y1, x2, y2] in [0, 1000] normalized coordinates.\n"
    "Output ONLY ONE tool call per response. Wait for the result before calling again.\n"
    "\n"
    "After receiving results from image_search_tool or text_search_tool, you MUST output on its own line BEFORE any further reasoning: "
    "<useful>[i, j, ...]</useful> listing the 1-based indices of results that are useful for geolocation. "
    "Output <useful>[]</useful> if none are useful. "
    "Example: after receiving search results, your response must start with <think>...</think> then immediately <useful>[1, 3]</useful>.\n"
    "\n"
    "Final answer format: <answer>Country, City, Latitude, Longitude</answer>. "
    "e.g. <answer>Italy, Golfo Arnaci, 40.9606, 9.5873</answer>"
)

USER_PROMPT = """\
<image>Analyze the architectural styles, vegetation, street infrastructure, and cultural markers \
in this image. Based on these visual cues, determine the location.

Answer strictly in the following format:
Country, City, Latitude, Longitude. You FIRST think about the reasoning process as an internal \
monologue and then provide the final answer. The reasoning process MUST BE enclosed within \
<think> </think> tags. Wrap your final answer in <answer> tags in the format: \
<answer>Country, City, Latitude, Longitude</answer>. \
e.g. <answer>Italy, Golfo Arnaci, 40.9606, 9.5873</answer>"""

# Sanity check: ensure user prompt uses <answer> format, not \boxed{}
assert '<answer>' in USER_PROMPT, "USER_PROMPT must use <answer> format, not \\boxed{}"
assert r'\boxed' not in USER_PROMPT, "USER_PROMPT must NOT use \\boxed{} format"


def build_prompt(system: str, user: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


def build_ground_truth(lat: float, lon: float, country: str, city: str) -> str:
    return f"{lat:.4f}, {lon:.4f}, {country}, {city}"


def km_to_difficulty(km) -> str:
    if km is None:
        return 'unknown'
    km = float(km)
    if km <= 25:
        return 'easy'
    elif km <= 200:
        return 'medium'
    else:
        return 'hard'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv',
                    default='/mnt/sh/mmvision/home/jonahli/projects/tusou/eval/im2gps3k_50.csv')
    ap.add_argument('--jsonl',
                    default='/mnt/sh/mmvision/home/jonahli/save/agent/eval/im2gps3k/'
                            'kimi_im2gps3k_val_v6/G_val.jsonl')
    ap.add_argument('--out_dir',
                    default='/mnt/sh/mmvision/home/jonahli/data_agent/rl/coldstart')
    args = ap.parse_args()

    import pandas as pd

    os.makedirs(args.out_dir, exist_ok=True)

    # Load ground truth CSV
    df_csv = pd.read_csv(args.csv)
    # id, latitude, longitude, country, state, city
    gt_map = {}
    for _, row in df_csv.iterrows():
        img_id = str(int(row['id']))
        gt_map[img_id] = {
            'lat': float(row['latitude']),
            'lon': float(row['longitude']),
            'country': str(row['country']) if row['country'] == row['country'] else 'Unknown',
            'city': str(row['city']) if row['city'] == row['city'] else 'Unknown',
        }
    print(f"Loaded {len(gt_map)} rows from CSV")

    # Load JSONL annotation
    jsonl_entries = {}
    if not os.path.exists(args.jsonl):
        print(f"ERROR: JSONL not found: {args.jsonl}")
        print("Run eval_im2gps3k.py with kimi_k2d6 + zoom,image_search,text_search first.")
        return

    with open(args.jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            img_id = str(d.get('id', ''))
            if img_id:
                jsonl_entries[img_id] = d

    print(f"Loaded {len(jsonl_entries)} entries from JSONL")

    # Check useful_results coverage
    n_useful = sum(
        1 for d in jsonl_entries.values()
        if d.get('useful_results') is not None and len(d.get('useful_results', [])) > 0
    )
    print(f"  useful_results non-empty: {n_useful}/{len(jsonl_entries)}")
    if n_useful == 0:
        print("  WARNING: no useful_results found — negmix reward will be disabled for val.")
        print("  Consider re-annotating with SYSTEM_TOOL_LLMCLIENT_ALL3_V2.")

    rows = []
    n_skip_no_image = 0
    n_skip_no_jsonl = 0

    for img_id, gt in gt_map.items():
        if img_id not in jsonl_entries:
            print(f"  WARN: {img_id} not in JSONL, skipping")
            n_skip_no_jsonl += 1
            continue

        d = jsonl_entries[img_id]

        # Get annotation-time image path
        images = d.get('images', [])
        ann_path = None
        if images:
            img0 = images[0]
            if isinstance(img0, dict):
                ann_path = img0.get('image_url', '') or img0.get('path', '')
            else:
                ann_path = str(img0)

        if not ann_path or not os.path.exists(ann_path):
            print(f"  WARN: image not found for {img_id}: {ann_path}, skipping")
            n_skip_no_image += 1
            continue

        km_val = d.get('km')
        difficulty = km_to_difficulty(km_val)
        useful_results = d.get('useful_results')  # may be None or []

        lat = gt['lat']
        lon = gt['lon']
        country = gt['country']
        city = gt['city']
        ground_truth = build_ground_truth(lat, lon, country, city)

        tools_kwargs = {
            "image_zoom_in_tool": {"create_kwargs": {"image": ann_path}},
            "image_search_tool":  {"create_kwargs": {"image": ann_path}},
            "text_search_tool":   {"create_kwargs": {"image": ann_path}},
        }

        extra_info = {
            "answer": ground_truth,
            "index": len(rows),
            "img_id": img_id,
            "part": "im2gps3k",
            "difficulty": difficulty,
            "km": km_val,
            "need_tools_kwargs": True,
            "split": "val",
            "tools_kwargs": tools_kwargs,
        }
        if useful_results is not None:
            extra_info["useful_results"] = useful_results

        rows.append({
            "data_source": "im2gps3k",
            "prompt": build_prompt(SYSTEM_PROMPT, USER_PROMPT),
            "images": [{"image_url": ann_path}],
            "ability": "geoloc",
            "reward_model": {"ground_truth": ground_truth, "style": "rule"},
            "agent_name": "tool_agent",
            "extra_info": extra_info,
        })

    if not rows:
        print("No rows built — check JSONL and CSV overlap.")
        return

    print(f"\nBuilt {len(rows)} val rows "
          f"(skipped {n_skip_no_jsonl} no-jsonl, {n_skip_no_image} no-image)")

    val_path = os.path.join(args.out_dir, 'val_coldstart_v4.parquet')

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_parquet(val_path, index=False)

    print(f"Val: {len(df)} rows → {val_path}")
    print(f"Val size: {os.path.getsize(val_path)/1024:.1f} KB")

    # Difficulty distribution
    diff_counts = df['extra_info'].apply(lambda x: x.get('difficulty', '?')).value_counts()
    print("Difficulty distribution:")
    for diff, cnt in diff_counts.items():
        print(f"  {diff}: {cnt}")

    # useful_results coverage
    n_with_useful = df['extra_info'].apply(
        lambda x: x.get('useful_results') is not None and len(x.get('useful_results', [])) > 0
    ).sum()
    print(f"useful_results coverage: {n_with_useful}/{len(df)}")


if __name__ == '__main__':
    main()
