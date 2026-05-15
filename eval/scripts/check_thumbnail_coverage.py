#!/usr/bin/env python3
"""
check_thumbnail_coverage.py — Check thumbnail coverage in image search cache.

Checks how many image_search calls in the coldstart/RL data have thumbnails
stored in the eval SQLite cache (imsearch_cache.db).

Usage:
  python eval/scripts/check_thumbnail_coverage.py
  python eval/scripts/check_thumbnail_coverage.py --parquet /path/to/train_coldstart_v4.parquet
"""

import argparse
import json
import os
import re
import sqlite3
import hashlib
import sys
from collections import defaultdict


def _cache_key(img_path: str, bbox_2d: list) -> str:
    bbox_str = ",".join(f"{v:.1f}" for v in bbox_2d)
    raw = f"{img_path}|{bbox_str}"
    return hashlib.sha256(raw.encode()).hexdigest()


def check_jsonl_thumbnail_coverage(jsonl_path: str, cache_db_path: str, difficulty_filter: str = None):
    """Check thumbnail coverage for image_search calls in a JSONL file."""
    conn = sqlite3.connect(cache_db_path, timeout=30)

    stats = defaultdict(lambda: {
        "total_samples": 0,
        "samples_with_search": 0,
        "total_search_calls": 0,
        "calls_in_cache": 0,
        "calls_with_thumbnail": 0,
        "calls_without_thumbnail": 0,
    })

    TOOL_CALL_RE = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)

    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            km = entry.get("km")
            masked = entry.get("masked", 0)

            # Determine difficulty
            if masked != 0 or km is None:
                difficulty = "unknown"
            elif km <= 25:
                difficulty = "easy"
            elif km <= 200:
                difficulty = "medium"
            else:
                difficulty = "hard"

            if difficulty_filter and difficulty != difficulty_filter:
                continue

            stats[difficulty]["total_samples"] += 1

            # Find image_search tool calls
            msgs = entry.get("messages", [])
            images = entry.get("images", [])
            image_path = images[0] if images else ""

            has_search = False
            for m in msgs:
                content = str(m.get("content", ""))
                for tc_match in TOOL_CALL_RE.finditer(content):
                    try:
                        tc = json.loads(tc_match.group(1))
                    except json.JSONDecodeError:
                        continue

                    if tc.get("name") != "image_search_tool":
                        continue

                    has_search = True
                    bbox = tc.get("arguments", {}).get("bbox_2d", [0, 0, 1000, 1000])
                    stats[difficulty]["total_search_calls"] += 1

                    # Check cache
                    key = _cache_key(image_path, bbox)
                    row = conn.execute(
                        "SELECT result FROM imsearch_cache WHERE key = ?", (key,)
                    ).fetchone()

                    if row:
                        stats[difficulty]["calls_in_cache"] += 1
                        data = json.loads(row[0])
                        results = data.get("results", [])
                        has_thumb = any(r.get("thumbnail", "") for r in results)
                        if has_thumb:
                            stats[difficulty]["calls_with_thumbnail"] += 1
                        else:
                            stats[difficulty]["calls_without_thumbnail"] += 1
                    else:
                        stats[difficulty]["calls_without_thumbnail"] += 1

            if has_search:
                stats[difficulty]["samples_with_search"] += 1

    conn.close()
    return dict(stats)


def check_parquet_thumbnail_coverage(parquet_path: str, cache_parquet_path: str):
    """Check thumbnail coverage in RL parquet's image_search cache."""
    import pandas as pd

    cache_df = pd.read_parquet(cache_parquet_path)
    print(f"Cache parquet: {len(cache_df)} rows")
    print(f"Columns: {list(cache_df.columns)}")

    if "result_thumbnail" in cache_df.columns:
        has_thumb = cache_df["result_thumbnail"].fillna("").astype(str).str.len() > 0
        print(f"  With thumbnail: {has_thumb.sum()} ({has_thumb.mean()*100:.1f}%)")
        print(f"  Without thumbnail: {(~has_thumb).sum()} ({(~has_thumb).mean()*100:.1f}%)")

        # Per image breakdown
        grouped = cache_df.groupby("image_path").apply(
            lambda g: g["result_thumbnail"].fillna("").astype(str).str.len().gt(0).any()
        )
        print(f"  Images with any thumbnail: {grouped.sum()}/{len(grouped)}")
    else:
        print("  No result_thumbnail column in cache parquet")


def main():
    parser = argparse.ArgumentParser(description="Check thumbnail coverage in image search cache")
    parser.add_argument("--jsonl", nargs="*",
                        default=[
                            "/mnt/sh/mmvision/home/jonahli/save/agent/annotate/coldstart_relabeled/part00.jsonl",
                            "/mnt/sh/mmvision/home/jonahli/save/agent/annotate/coldstart_relabeled/part01.jsonl",
                        ])
    parser.add_argument("--cache_db",
                        default="/mnt/sh/mmvision/home/jonahli/projects/tusou/eval/.cache/imsearch_cache.db")
    parser.add_argument("--cache_parquet",
                        default="/mnt/sh/mmvision/home/jonahli/data_agent/rl/coldstart/image_search_cache_merged_v2.parquet")
    parser.add_argument("--difficulty", default=None, choices=["easy", "medium", "hard", "unknown"])
    args = parser.parse_args()

    # Check SQLite cache
    print("=" * 70)
    print("  SQLite Cache Thumbnail Coverage (eval/.cache/imsearch_cache.db)")
    print("=" * 70)

    all_stats = defaultdict(lambda: {
        "total_samples": 0,
        "samples_with_search": 0,
        "total_search_calls": 0,
        "calls_in_cache": 0,
        "calls_with_thumbnail": 0,
        "calls_without_thumbnail": 0,
    })

    for jsonl_path in args.jsonl:
        if not os.path.exists(jsonl_path):
            print(f"  SKIP: {jsonl_path} not found")
            continue
        print(f"\n  Processing {os.path.basename(jsonl_path)}...")
        stats = check_jsonl_thumbnail_coverage(jsonl_path, args.cache_db, args.difficulty)
        for diff, s in stats.items():
            for k, v in s.items():
                all_stats[diff][k] += v

    # Print results
    print(f"\n  {'Difficulty':<12} {'Samples':>8} {'w/Search':>10} {'SearchCalls':>12} {'InCache':>8} {'w/Thumb':>8} {'no/Thumb':>9} {'ThumbRate':>10}")
    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*12} {'-'*8} {'-'*8} {'-'*9} {'-'*10}")

    total = {"total_samples": 0, "samples_with_search": 0, "total_search_calls": 0,
             "calls_in_cache": 0, "calls_with_thumbnail": 0, "calls_without_thumbnail": 0}

    for diff in ["easy", "medium", "hard", "unknown"]:
        s = all_stats.get(diff, {})
        if not s.get("total_samples"):
            continue
        rate = f"{s['calls_with_thumbnail']/s['total_search_calls']*100:.1f}%" if s['total_search_calls'] > 0 else "N/A"
        print(f"  {diff:<12} {s['total_samples']:>8} {s['samples_with_search']:>10} {s['total_search_calls']:>12} {s['calls_in_cache']:>8} {s['calls_with_thumbnail']:>8} {s['calls_without_thumbnail']:>9} {rate:>10}")
        for k in total:
            total[k] += s.get(k, 0)

    rate = f"{total['calls_with_thumbnail']/total['total_search_calls']*100:.1f}%" if total['total_search_calls'] > 0 else "N/A"
    print(f"  {'TOTAL':<12} {total['total_samples']:>8} {total['samples_with_search']:>10} {total['total_search_calls']:>12} {total['calls_in_cache']:>8} {total['calls_with_thumbnail']:>8} {total['calls_without_thumbnail']:>9} {rate:>10}")

    # Check parquet cache
    print(f"\n{'=' * 70}")
    print("  Parquet Cache Thumbnail Coverage (RL training cache)")
    print("=" * 70)
    if os.path.exists(args.cache_parquet):
        check_parquet_thumbnail_coverage(None, args.cache_parquet)
    else:
        print(f"  SKIP: {args.cache_parquet} not found")


if __name__ == "__main__":
    main()
