#!/usr/bin/env python3
"""
build_text_search_cache.py — Extract text_search_tool calls + Kimi <useful> labels
from part00 (or any) coldstart annotation JSONL, write a parquet cache usable by TextSearchTool.

Output schema (per result row):
  image_id     (str)  — basename of the original image, e.g. "3d_ca_5136986794.jpg"
  image_path   (str)  — full path from d['images'][0] (the actual file used in RL)
  call_idx     (int)  — 0-based tool call index within this image
  query        (str)  — single search query string (multi-query calls are split into rows)
  result_index (int)  — 1-based result index (as numbered in the response)
  result_title (str)
  result_url   (str)
  result_content (str)
  is_geo_useful (bool) — True if result_index in Kimi's useful indices for this call
  useful_indices (str) — JSON list of useful indices for this call, e.g. "[1,3,5]"
  n_useful     (int)
  call_turn    (int)  — turn number from useful_results annotation
  part         (str)  — source jsonl basename (e.g. "part00")

The tool call / result mapping is based on message ordering:
  assistant message with <tool_call>{"name":"text_search_tool",...}</tool_call>
  followed by user message with <tool_response>Web search results...</tool_response>

Usage:
  python data_pipeline/text_search/build_text_search_cache.py \\
    --jsonl /mnt/sh/mmvision/home/jonahli/save/agent/annotate/coldstart/part00.jsonl \\
    --out /mnt/sh/mmvision/home/jonahli/data_agent/rl/coldstart/text_search_cache_part00.parquet

Multiple parts:
  for part in 00 01 02; do
    python ... --jsonl ...part${part}.jsonl --out ...part${part}.parquet
  done
  # then merge with merge_text_search_caches.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

TOOL_CALL_RE = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
# Each query's results block: "Web search results for: QUERY\n\n[1] ...\n..."
QUERY_BLOCK_RE = re.compile(
    r'Web search results for:\s*(.*?)\n+(.*?)(?=\nWeb search results for:|\Z)',
    re.DOTALL
)
RESULT_BLOCK_RE = re.compile(
    r'\[(\d+)\]\s*(.*?)\nURL:\s*(https?://\S+)\n?(.*?)(?=\n\[\d+\]|\Z)',
    re.DOTALL
)


def _content_str(content) -> str:
    """Extract text from a message content field (handles str and list-of-dicts)."""
    if isinstance(content, list):
        return ' '.join(
            x.get('text', '') if isinstance(x, dict) else str(x)
            for x in content
        )
    return content or ''


def _parse_query_field(query_raw) -> list[str]:
    """Normalise the `query` argument (str or list) to a list of strings."""
    if isinstance(query_raw, list):
        return [str(q) for q in query_raw if q]
    if isinstance(query_raw, str) and query_raw.strip():
        return [query_raw.strip()]
    return []


def _parse_results_from_response(resp_text: str) -> dict[str, list[dict]]:
    """
    Parse tool_response text into {query_str: [result_dicts]}.

    Each result_dict: {index (int), title, url, content}

    Handles multi-query responses (multiple "Web search results for: X" blocks).
    If the response has no per-query header, all results are grouped under key "".
    """
    out: dict[str, list[dict]] = {}

    if 'Web search results for:' in resp_text:
        for m in QUERY_BLOCK_RE.finditer(resp_text):
            query_key = m.group(1).strip()
            block = m.group(2)
            results = []
            for rm in RESULT_BLOCK_RE.finditer(block):
                idx = int(rm.group(1))
                title = rm.group(2).strip()
                url = rm.group(3).strip()
                content = rm.group(4).strip()
                results.append({'index': idx, 'title': title, 'url': url, 'content': content})
            out[query_key] = results
    else:
        # Fallback: no per-query header, parse all [N] blocks
        results = []
        for rm in RESULT_BLOCK_RE.finditer(resp_text):
            idx = int(rm.group(1))
            title = rm.group(2).strip()
            url = rm.group(3).strip()
            content = rm.group(4).strip()
            results.append({'index': idx, 'title': title, 'url': url, 'content': content})
        out[''] = results

    return out


def _extract_tool_calls(messages: list) -> list[dict]:
    """
    Walk messages, pair each text_search_tool call with its response.

    Returns list of:
        {
          msg_idx:      int,   # index of assistant message
          resp_msg_idx: int,   # index of user message with tool_response
          queries:      list[str],
          results_by_query: dict[str, list[dict]],
          raw_resp_text: str,
        }
    """
    pairs = []
    pending: dict | None = None  # waiting for response

    for i, m in enumerate(messages):
        role = m.get('role', '')
        c = _content_str(m.get('content', ''))

        if role == 'assistant':
            # Possibly starts new text_search_tool calls
            new_queries = []
            for tc_m in TOOL_CALL_RE.finditer(c):
                try:
                    tc = json.loads(tc_m.group(1))
                except Exception:
                    continue
                if tc.get('name') != 'text_search_tool':
                    continue
                qs = _parse_query_field(tc.get('arguments', {}).get('query', ''))
                new_queries.extend(qs)
            if new_queries:
                pending = {'msg_idx': i, 'queries': new_queries}

        elif role == 'user' and pending is not None and 'Web search results' in c:
            results_by_query = _parse_results_from_response(c)
            pairs.append({
                **pending,
                'resp_msg_idx': i,
                'results_by_query': results_by_query,
                'raw_resp_text': c,
            })
            pending = None

    return pairs


def process_entry(d: dict, part: str) -> list[dict]:
    """Convert one annotation entry to a list of cache rows."""
    image_id = d.get('id', '')
    images = d.get('images', [])
    # images[0] may be a path string or a dict like {'image_url': '...'}
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

    # Build turn → useful_indices map for text_search_tool
    # Note: turn in useful_results is the assistant-turn number (1-based in eval loop)
    # We match by call order, not by message index.
    text_useful_map: dict[int, list[int]] = {}
    text_useful_list = [u for u in useful_results if u.get('tool') == 'text_search_tool']
    for u in text_useful_list:
        turn = u.get('turn', -1)
        indices = u.get('indices', [])
        text_useful_map[turn] = indices

    pairs = _extract_tool_calls(messages)

    rows = []
    for call_idx, pair in enumerate(pairs):
        queries = pair['queries']
        results_by_query = pair['results_by_query']

        # Associate with useful_results by call_idx order
        # The useful_results turn is the turn counter from eval loop.
        # We align call_idx (0-based) to text_useful_list (ordered).
        useful_indices: list[int] = []
        if call_idx < len(text_useful_list):
            useful_indices = text_useful_list[call_idx].get('indices', [])
            call_turn = text_useful_list[call_idx].get('turn', -1)
        else:
            call_turn = -1

        useful_indices_json = json.dumps(useful_indices)
        n_useful = len(useful_indices)

        # If multi-query: results are shared across all queries in this call.
        # We create one row per (query, result) combination.
        # For reward: all queries in a call share the same useful_indices
        # (Kimi marks results useful at the call level, not per-query).

        # Flatten all results across all query blocks; assign global result_index.
        # The [N] numbering in the response is already global (1..M across all queries).
        all_results: dict[int, dict] = {}  # index → result
        for qkey, rlist in results_by_query.items():
            for r in rlist:
                if r['index'] not in all_results:
                    all_results[r['index']] = r

        if not all_results:
            # Fallback: create a synthetic row so the call is still recorded
            rows.append({
                'image_id': image_id,
                'image_path': image_path,
                'call_idx': call_idx,
                'query': ' | '.join(queries),
                'result_index': -1,
                'result_title': '',
                'result_url': '',
                'result_content': '',
                'is_geo_useful': False,
                'useful_indices': useful_indices_json,
                'n_useful': n_useful,
                'call_turn': call_turn,
                'part': part,
            })
            continue

        for ridx, r in sorted(all_results.items()):
            rows.append({
                'image_id': image_id,
                'image_path': image_path,
                'call_idx': call_idx,
                'query': ' | '.join(queries),
                'result_index': ridx,
                'result_title': r['title'],
                'result_url': r['url'],
                'result_content': r['content'],
                'is_geo_useful': ridx in useful_indices,
                'useful_indices': useful_indices_json,
                'n_useful': n_useful,
                'call_turn': call_turn,
                'part': part,
            })

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', required=True, help='Path to coldstart annotation JSONL')
    ap.add_argument('--out', required=True, help='Output parquet path')
    ap.add_argument('--min_n_useful', type=int, default=0,
                    help='Only include calls with >= this many useful results (0 = include all)')
    args = ap.parse_args()

    import pandas as pd

    part = os.path.splitext(os.path.basename(args.jsonl))[0]
    print(f"Processing {args.jsonl} (part={part}) ...")

    all_rows = []
    n_entries = 0
    n_with_text_search = 0

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
            has_ts = any(u.get('tool') == 'text_search_tool' for u in d.get('useful_results', []))
            if has_ts:
                n_with_text_search += 1
            rows = process_entry(d, part)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    print(f"  Total entries processed: {n_entries}")
    print(f"  Entries with text_search useful labels: {n_with_text_search}")
    print(f"  Total rows: {len(df)}")

    if len(df) == 0:
        print("  No rows extracted. Check JSONL format.")
        sys.exit(0)

    print(f"  is_geo_useful=True: {df['is_geo_useful'].sum()}")
    print(f"  Unique images: {df['image_id'].nunique()}")
    print(f"  Calls per image (mean): {df.groupby('image_id')['call_idx'].max().mean():.2f}")
    print(f"  Results per call (mean): {df.groupby(['image_id','call_idx'])['result_index'].count().mean():.2f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"\nSaved: {args.out}  ({os.path.getsize(args.out)/1024:.1f} KB)")


if __name__ == '__main__':
    main()
