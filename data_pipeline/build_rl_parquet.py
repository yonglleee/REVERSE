#!/usr/bin/env python3
"""
build_rl_parquet.py — Build RL training parquet from coldstart CSV + JSONL.

Each row = one training sample with:
  - data_source: "coldstart"
  - prompt: [system, user] messages in VERL multi-turn chat format
  - images: [{"image_url": annotation_time_image_path}]
  - ability: "geoloc"
  - reward_model: {"ground_truth": "lat, lon, country, city", "style": "rule"}
  - agent_name: "tool_agent"
  - extra_info: {
        "answer": "lat, lon, country, city",
        "index": row_index,
        "img_id": img_id,
        "part": partNN,
        "difficulty": "easy" | "medium" | "hard" | "unknown",
        "km": float | None,
        "need_tools_kwargs": True,
        "split": "train",
        "tools_kwargs": {
            "image_zoom_in_tool":   {"create_kwargs": {"image": annotation_time_path}},
            "image_search_tool":    {"create_kwargs": {"image": annotation_time_path}},
            "text_search_tool":     {"create_kwargs": {"image": annotation_time_path}},
        }
    }

Filter rules:
  1. Skip rows with no JSONL annotation entry (no annotation-time image path, no cache).
  2. Skip masked rows with zero tool calls (no learning signal).
  3. Keep masked-with-tools rows (km=null, CSV LAT/LON valid for RL reward).

Difficulty levels (based on km from JSONL):
  - easy:    km <= 25
  - medium:  25 < km <= 200
  - hard:    km > 200
  - unknown: masked (km=null) but has tool calls

Curriculum sampling CLI args:
  --difficulty_filter   comma-sep subset: easy,medium,hard,unknown  (default: all four)
  --max_per_class       cap per difficulty bucket (default: unlimited)
  --suffix              output filename suffix, e.g. "_easy"

Val is always built separately by build_val_im2gps3k.py (fixed 50-row im2gps3k benchmark).

Usage:
  # Baseline (all difficulties)
  python data_pipeline/build_rl_parquet.py \\
    --parts 00 01 \\
    --out_dir /mnt/sh/mmvision/home/jonahli/data_agent/rl/coldstart

  # Easy only
  python data_pipeline/build_rl_parquet.py \\
    --parts 00 01 \\
    --out_dir /mnt/sh/mmvision/home/jonahli/data_agent/rl/coldstart \\
    --difficulty_filter easy --suffix _easy
"""

from __future__ import annotations

import argparse
import json
import os
import random

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
    "<useful>[i, j, ...]</useful> listing the 1-based indices of results that match this specific image "
    "(i.e., mention the actual location, landmark, or geographic region shown in the image). "
    "Results about a different place are NOT useful even if they contain geographic information. "
    "Output <useful>[]</useful> if none match. "
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


def make_relative(path: str, data_root: str) -> str:
    """Convert absolute path to relative path under data_root.
    If data_root is empty or path doesn't start with data_root, return as-is."""
    if not data_root or not path:
        return path
    data_root = data_root.rstrip('/')
    if path.startswith(data_root + '/'):
        return path[len(data_root) + 1:]
    return path


def _has_api_fail_rl(messages: list) -> bool:
    import json as _json
    for m in messages:
        c = m.get('content', '') or ''
        if isinstance(c, list):
            c = ' '.join(x.get('text', '') for x in c if isinstance(x, dict))
        if ('Image search failed' in c or 'Tavily search failed' in c or
                'all API keys exhausted' in c or 'COS upload' in c):
            return True
    return False


def _has_full_img_search_rl(messages: list) -> bool:
    import json as _json, re as _re
    TC_RE = _re.compile(r'<tool_call>(.*?)</tool_call>', _re.DOTALL)
    for m in messages:
        if m.get('role') != 'assistant':
            continue
        c = m.get('content', '') or ''
        if isinstance(c, list):
            c = ' '.join(x.get('text', '') for x in c if isinstance(x, dict))
        for tc_m in TC_RE.finditer(c):
            try:
                tc = _json.loads(tc_m.group(1))
                if tc.get('name') == 'image_search_tool':
                    b = tc.get('arguments', {}).get('bbox_2d', [])
                    if b and b[0] <= 5 and b[1] <= 5 and b[2] >= 995 and b[3] >= 995:
                        return True
            except Exception:
                pass
    return False


def process_csv(csv_path: str, jsonl_path: str, part: str,
                start_idx: int = 0, require_fullcov: bool = False,
                data_root: str = '', drop_api_errors: bool = False,
                crop_filter: bool = False) -> list[dict]:
    """
    Build one row per image from the coldstart CSV.

    Filter rules:
    - Only keep rows with JSONL annotation entry (annotation-time image path required for cache).
    - Skip masked rows with zero tool calls (no learning signal).
    - If require_fullcov=True: skip rows where any search call (image_search/text_search)
      lacks a useful_results annotation (ensures discrimination reward is fully defined).

    Returns list of row dicts ready for pd.DataFrame.
    """
    import pandas as pd
    import re

    TOOL_CALL_RE = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)

    df_csv = pd.read_csv(csv_path)
    # Drop rows without valid lat/lon
    df_csv = df_csv.dropna(subset=["LAT", "LON"])

    # Build image_id → annotation_time_path, masked, km, n_tool_calls from JSONL
    image_path_map: dict[str, str] = {}
    masked_map: dict[str, float] = {}
    km_map: dict[str, object] = {}
    n_tool_map: dict[str, int] = {}
    fullcov_map: dict[str, bool] = {}  # img_id → all search calls have useful annotation

    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                img_id = d.get('id', '')
                if not img_id:
                    continue

                # Optional filters on trajectory quality
                msgs = d.get('messages', [])
                if drop_api_errors and _has_api_fail_rl(msgs):
                    continue
                if crop_filter and _has_full_img_search_rl(msgs):
                    continue

                images = d.get('images', [])
                if images:
                    img0 = images[0]
                    if isinstance(img0, dict):
                        ann_path = img0.get('image_url', '') or img0.get('path', '')
                    else:
                        ann_path = str(img0)
                    if ann_path and os.path.exists(ann_path):
                        image_path_map[img_id] = ann_path

                masked_map[img_id] = float(d.get('masked', 0.0))
                km_map[img_id] = d.get('km')  # may be None for masked rows
                n_tool_map[img_id] = int(
                    d.get('n_tool_calls', 0) or
                    (d.get('n_crop_calls', 0) or 0) +
                    (d.get('n_search_calls', 0) or 0) +
                    (d.get('n_tavily_calls', 0) or 0)
                )

                # Count search calls vs annotated useful_results for fullcov check
                if require_fullcov:
                    msgs = d.get('messages', [])
                    # Count search calls in messages
                    n_search = 0
                    for m in msgs:
                        if m.get('role') != 'assistant':
                            continue
                        c = m.get('content', '')
                        if isinstance(c, list):
                            c = ' '.join(x.get('text', '') for x in c if isinstance(x, dict))
                        for tc_m in TOOL_CALL_RE.finditer(c):
                            try:
                                tc = json.loads(tc_m.group(1))
                                if tc.get('name') in ('image_search_tool', 'text_search_tool'):
                                    n_search += 1
                            except Exception:
                                pass
                    # Count annotated useful_results for search tools
                    n_annotated = sum(
                        1 for u in d.get('useful_results', [])
                        if u.get('tool') in ('image_search_tool', 'text_search_tool')
                    )
                    # Full coverage: no search calls, or every search call has an annotation
                    # Using count comparison (n_annotated >= n_search) as proxy for per-call coverage.
                    # Strict per-call matching is not feasible since useful_results uses message
                    # absolute indices which may not align 1:1 with call order across reruns.
                    fullcov_map[img_id] = (n_search == 0) or (n_annotated >= n_search)
    else:
        print(f"  WARN: JSONL not found: {jsonl_path}")

    rows = []
    n_skip_no_jsonl = 0
    n_skip_masked_no_tools = 0
    n_skip_no_fullcov = 0

    for i, csv_row in enumerate(df_csv.itertuples(index=False)):
        img_id = csv_row.IMG_ID
        lat = float(csv_row.LAT)
        lon = float(csv_row.LON)
        country = (str(csv_row.country)
                   if hasattr(csv_row, 'country') and csv_row.country == csv_row.country
                   else "Unknown")
        city = (str(csv_row.city)
                if hasattr(csv_row, 'city') and csv_row.city == csv_row.city
                else "Unknown")

        # FILTER 1: Only keep rows with JSONL annotation entry
        if img_id not in image_path_map:
            n_skip_no_jsonl += 1
            continue

        ann_path = image_path_map[img_id]

        # FILTER 2: Skip masked rows with zero tool calls (no learning signal)
        is_masked = masked_map.get(img_id, 0.0) != 0.0
        n_tools = n_tool_map.get(img_id, 0)
        if is_masked and n_tools == 0:
            n_skip_masked_no_tools += 1
            continue

        # FILTER 3: Skip rows without full useful_results coverage (if require_fullcov)
        if require_fullcov and not fullcov_map.get(img_id, True):
            n_skip_no_fullcov += 1
            continue

        km_val = km_map.get(img_id)
        difficulty = km_to_difficulty(km_val)

        ground_truth = build_ground_truth(lat, lon, country, city)
        global_idx = start_idx + len(rows)

        tools_kwargs = {
            "image_zoom_in_tool": {"create_kwargs": {"image": make_relative(ann_path, data_root)}},
            "image_search_tool":  {"create_kwargs": {"image": make_relative(ann_path, data_root)}},
            "text_search_tool":   {"create_kwargs": {"image": make_relative(ann_path, data_root)}},
        }

        rows.append({
            "data_source": "coldstart",
            "prompt": build_prompt(SYSTEM_PROMPT, USER_PROMPT),
            "images": [{"image_url": make_relative(ann_path, data_root)}],
            "ability": "geoloc",
            "reward_model": {"ground_truth": ground_truth, "style": "rule"},
            "agent_name": "tool_agent",
            "extra_info": {
                "answer": ground_truth,
                "index": global_idx,
                "img_id": img_id,
                "part": part,
                "difficulty": difficulty,
                "km": km_val,
                "need_tools_kwargs": True,
                "split": "train",
                "tools_kwargs": tools_kwargs,
            },
        })

    print(f"  part{part}: {len(rows)} rows kept "
          f"(skipped {n_skip_no_jsonl} no-jsonl, {n_skip_masked_no_tools} masked-no-tools"
          f"{', ' + str(n_skip_no_fullcov) + ' no-fullcov' if require_fullcov else ''})")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv_dir',
                    default='/mnt/sh/mmvision/home/jonahli/data_agent/coldstart/raw')
    ap.add_argument('--jsonl_dir',
                    default='/mnt/sh/mmvision/home/jonahli/save/agent/annotate/coldstart')
    ap.add_argument('--out_dir',
                    default='/mnt/sh/mmvision/home/jonahli/data_agent/rl/coldstart')
    ap.add_argument('--fullcov', action='store_true',
                    help='Only keep samples where all search calls have useful_results annotations')
    ap.add_argument('--parts', nargs='*', default=None,
                    help='Parts to include (e.g. 00 01). Default: auto-detect all available.')
    ap.add_argument('--seed', type=int, default=42)
    # Curriculum sampling
    ap.add_argument('--difficulty_filter', type=str, default=None,
                    help='Comma-sep subset to keep: easy,medium,hard,unknown (default: all four)')
    ap.add_argument('--max_per_class', type=int, default=None,
                    help='Cap number of rows per difficulty class (default: unlimited)')
    ap.add_argument('--suffix', type=str, default='',
                    help='Output filename suffix, e.g. "_easy" → train_coldstart_v4_easy.parquet')
    ap.add_argument('--data_root', type=str, default='',
                    help='If set, strip this prefix from image paths to make them relative. '
                         'E.g. /mnt/sh/mmvision/home/jonahli/data_agent/REVERSE')
    ap.add_argument('--drop_api_errors', action='store_true',
                    help='Drop trajectories with API failure markers (Tavily/Image/COS).')
    ap.add_argument('--crop_filter', action='store_true',
                    help='Drop trajectories where any image_search uses near-full-image bbox.')
    args = ap.parse_args()

    import pandas as pd

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    # Auto-detect available parts
    if args.parts is None:
        import glob
        csv_files = sorted(glob.glob(os.path.join(args.csv_dir, 'test_filtered_part*.csv')))
        parts = [os.path.basename(f).replace('test_filtered_part', '').replace('.csv', '')
                 for f in csv_files]
        print(f"Auto-detected parts: {parts}")
    else:
        parts = args.parts

    all_rows = []
    start_idx = 0

    for part in parts:
        csv_path = os.path.join(args.csv_dir, f'test_filtered_part{part}.csv')
        jsonl_path = os.path.join(args.jsonl_dir, f'part{part}.jsonl')

        if not os.path.exists(csv_path):
            print(f"  SKIP: CSV not found: {csv_path}")
            continue

        print(f"\nProcessing part{part} ...")
        rows = process_csv(csv_path, jsonl_path, part=part, start_idx=start_idx,
                           require_fullcov=args.fullcov, data_root=args.data_root,
                           drop_api_errors=args.drop_api_errors,
                           crop_filter=args.crop_filter)
        all_rows.extend(rows)
        start_idx += len(rows)

    if not all_rows:
        print("No rows extracted.")
        return

    df = pd.DataFrame(all_rows)

    # Print difficulty distribution before filtering
    difficulty_counts = df['extra_info'].apply(lambda x: x.get('difficulty', '?')).value_counts()
    print(f"\nTotal rows before filter: {len(df)}")
    print("Difficulty distribution:")
    for diff, cnt in difficulty_counts.items():
        print(f"  {diff}: {cnt}")

    # Apply difficulty filter
    if args.difficulty_filter:
        keep_set = set(d.strip() for d in args.difficulty_filter.split(','))
        mask = df['extra_info'].apply(lambda x: x.get('difficulty', '?')).isin(keep_set)
        df = df[mask].reset_index(drop=True)
        print(f"\nAfter difficulty_filter={args.difficulty_filter}: {len(df)} rows")

    # Apply per-class cap
    if args.max_per_class is not None:
        diff_col = df['extra_info'].apply(lambda x: x.get('difficulty', '?'))
        groups = []
        for diff_val, grp in df.groupby(diff_col):
            if len(grp) > args.max_per_class:
                grp = grp.sample(args.max_per_class, random_state=args.seed)
            groups.append(grp)
        df = pd.concat(groups, ignore_index=True)
        # Shuffle after capping
        df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        print(f"After max_per_class={args.max_per_class}: {len(df)} rows")

    # Re-index
    for i, row in df.iterrows():
        row['extra_info']['index'] = i

    train_path = os.path.join(args.out_dir, f'train_coldstart_v4{args.suffix}.parquet')
    df.to_parquet(train_path, index=False)

    print(f"\nTrain: {len(df)} rows → {train_path}")
    print(f"Train size: {os.path.getsize(train_path)/1024:.1f} KB")

    # Final difficulty distribution
    final_diff = df['extra_info'].apply(lambda x: x.get('difficulty', '?')).value_counts()
    print("Final difficulty distribution:")
    for diff, cnt in final_diff.items():
        print(f"  {diff}: {cnt}")


if __name__ == '__main__':
    main()
