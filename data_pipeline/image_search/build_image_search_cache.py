#!/usr/bin/env python3
"""
build_image_search_cache.py — Extract image_search_tool calls + Kimi <useful> labels
from coldstart annotation JSONL, write a parquet cache usable by ImageSearchTool.

Key differences from SpotAgent's search_cache_labeled.parquet:
  - gt_bbox comes from the FIRST image_search call's bbox_2d (from the annotation itself)
    NOT from a separate labeling pipeline
  - image_path = annotation-time path (e.g. part00_images/IMG_ID_0.jpg)
  - Only the first call's bbox is stored as gt_bbox (used for zoom IOU check)
  - Results: result_title, result_link (from "Link:"), result_source (from "Source:")
  - is_geo_useful: True if result index in Kimi's useful_results[tool='image_search_tool']

Output schema (per result row, compatible with ImageSearchTool._load_cache):
  image_id       (str)  — IMG_ID basename
  image_path     (str)  — annotation-time image path (cache key for ImageSearchTool)
  bbox           (str)  — JSON [x1,y1,x2,y2] of the gt bbox (from first search call)
  call_idx       (int)  — 0-based call index
  result_pos     (int)  — 1-based result position
  result_title   (str)
  result_link    (str)
  result_source  (str)
  result_thumbnail (str) — empty (not available from annotation)
  is_geo_useful  (bool)
  useful_indices (str)  — JSON list, e.g. "[1,3,5]"
  n_useful       (int)
  call_turn      (int)
  part           (str)  — source jsonl basename

ImageSearchTool._load_cache groups by (image_path, bbox), so storing bbox per row
is required. We use the first call's bbox as gt_bbox for the whole image.

Usage:
  python data_pipeline/image_search/build_image_search_cache.py \\
    --jsonl /mnt/sh/mmvision/home/jonahli/save/agent/annotate/coldstart/part00.jsonl \\
    --out /mnt/sh/mmvision/home/jonahli/data_agent/rl/coldstart/image_search_cache_part00.parquet

Multiple parts:
  for part in 00 01; do
    python ... --jsonl ...part${part}.jsonl --out ...part${part}.parquet
  done
  # then merge with: python -c "
  #   import pandas as pd, glob
  #   dfs = [pd.read_parquet(f) for f in sorted(glob.glob('...part*.parquet'))]
  #   pd.concat(dfs).to_parquet('image_search_cache_merged.parquet', index=False)
  # "
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

TOOL_CALL_RE = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)

# Response format from annotation:
# Image search results for region [x1, y1, x2, y2] (goal: ...):
# [1] Title
#     Source: SourceName
#     Link: https://...
RESULT_BLOCK_RE = re.compile(
    r'\[(\d+)\]\s*(.*?)\n\s*Source:\s*(.*?)\n\s*Link:\s*(https?://\S+)',
    re.DOTALL
)


def _content_str(content) -> str:
    if isinstance(content, list):
        return ' '.join(
            x.get('text', '') if isinstance(x, dict) else str(x)
            for x in content
        )
    return content or ''


def _parse_image_search_results(resp_text: str) -> list[dict]:
    """
    Parse image search tool_response text into list of result dicts.
    Each dict: {pos: int, title: str, source: str, link: str}
    """
    results = []
    for m in RESULT_BLOCK_RE.finditer(resp_text):
        pos = int(m.group(1))
        title = m.group(2).strip()
        source = m.group(3).strip()
        link = m.group(4).strip()
        results.append({'pos': pos, 'title': title, 'source': source, 'link': link})
    return results


def _extract_image_search_calls(messages: list) -> list[dict]:
    """
    Walk messages, pair each image_search_tool call with its response.

    Returns list of:
        {
          msg_idx:   int,
          queries:   list[dict] with bbox_2d
          bbox_2d:   list[float]   (from tool call arguments)
          results:   list[dict]    (parsed from response)
          raw_resp:  str
        }
    """
    pairs = []
    pending = None

    for i, m in enumerate(messages):
        role = m.get('role', '')
        c = _content_str(m.get('content', ''))

        if role == 'assistant':
            for tc_m in TOOL_CALL_RE.finditer(c):
                try:
                    tc = json.loads(tc_m.group(1))
                except Exception:
                    continue
                if tc.get('name') != 'image_search_tool':
                    continue
                bbox_2d = tc.get('arguments', {}).get('bbox_2d', [])
                if bbox_2d and len(bbox_2d) == 4:
                    pending = {'msg_idx': i, 'bbox_2d': [float(v) for v in bbox_2d]}
                    break  # only first call per assistant message

        elif role == 'user' and pending is not None and 'Image search results' in c:
            results = _parse_image_search_results(c)
            pairs.append({
                **pending,
                'resp_msg_idx': i,
                'results': results,
                'raw_resp': c,
            })
            pending = None

    return pairs


def process_entry(d: dict, part: str) -> list[dict]:
    """Convert one annotation entry to a list of cache rows."""
    image_id = d.get('id', '')
    images = d.get('images', [])
    if images:
        img0 = images[0]
        if isinstance(img0, dict):
            image_path = img0.get('image_url', '') or img0.get('path', '')
        else:
            image_path = str(img0)
    else:
        image_path = ''

    messages = d.get('messages', [])
    useful_results = d.get('useful_results', [])

    # Build ordered list of (turn, useful_indices) for image_search_tool
    img_useful_list = [u for u in useful_results if u.get('tool') == 'image_search_tool']

    pairs = _extract_image_search_calls(messages)
    if not pairs:
        return []

    rows = []
    for call_idx, pair in enumerate(pairs):
        # Each call uses its OWN bbox as gt_bbox (not just the first call's)
        call_gt_bbox = pair['bbox_2d']
        call_gt_bbox_json = json.dumps(call_gt_bbox)

        # Match useful_indices by call order
        if call_idx < len(img_useful_list):
            useful_indices = img_useful_list[call_idx].get('indices', [])
            call_turn = img_useful_list[call_idx].get('turn', -1)
        else:
            useful_indices = []
            call_turn = -1

        useful_indices_json = json.dumps(useful_indices)
        n_useful = len(useful_indices)
        results = pair['results']

        if not results:
            # Record call with no parseable results
            rows.append({
                'image_id':       image_id,
                'image_path':     image_path,
                'bbox':           call_gt_bbox_json,
                'call_idx':       call_idx,
                'result_pos':     -1,
                'result_title':   '',
                'result_link':    '',
                'result_source':  '',
                'result_thumbnail': '',
                'is_geo_useful':  False,
                'useful_indices': useful_indices_json,
                'n_useful':       n_useful,
                'call_turn':      call_turn,
                'part':           part,
            })
            continue

        for r in results:
            rows.append({
                'image_id':       image_id,
                'image_path':     image_path,
                'bbox':           call_gt_bbox_json,
                'call_idx':       call_idx,
                'result_pos':     r['pos'],
                'result_title':   r['title'],
                'result_link':    r['link'],
                'result_source':  r['source'],
                'result_thumbnail': '',
                'is_geo_useful':  r['pos'] in useful_indices,
                'useful_indices': useful_indices_json,
                'n_useful':       n_useful,
                'call_turn':      call_turn,
                'part':           part,
            })

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', required=True, help='Path to coldstart annotation JSONL')
    ap.add_argument('--out', required=True, help='Output parquet path')
    ap.add_argument('--min_n_useful', type=int, default=0,
                    help='Only include images with >= this many useful results (0 = include all)')
    args = ap.parse_args()

    import pandas as pd

    part = os.path.splitext(os.path.basename(args.jsonl))[0]
    print(f"Processing {args.jsonl} (part={part}) ...")

    all_rows = []
    n_entries = 0
    n_with_imgsearch = 0

    with open(args.jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception as e:
                print(f"  WARN: json parse error: {e}")
                continue
            n_entries += 1
            has_is = any(u.get('tool') == 'image_search_tool' for u in d.get('useful_results', []))
            if has_is:
                n_with_imgsearch += 1
            rows = process_entry(d, part)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    print(f"  Total entries processed: {n_entries}")
    print(f"  Entries with image_search useful labels: {n_with_imgsearch}")
    print(f"  Total rows: {len(df)}")

    if len(df) == 0:
        print("  No rows extracted. Check JSONL format.")
        sys.exit(0)

    print(f"  is_geo_useful=True: {df['is_geo_useful'].sum()}")
    print(f"  Unique images: {df['image_id'].nunique()}")
    print(f"  Calls per image (mean): {df.groupby('image_id')['call_idx'].max().mean():.2f}")
    print(f"  Results per call (mean): {df.groupby(['image_id','call_idx'])['result_pos'].count().mean():.2f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"\nSaved: {args.out}  ({os.path.getsize(args.out)/1024:.1f} KB)")


if __name__ == '__main__':
    main()
