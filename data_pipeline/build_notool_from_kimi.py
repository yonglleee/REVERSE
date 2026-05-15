#!/usr/bin/env python3
"""
build_notool_from_kimi.py — Construct high-quality notool SFT samples by
extracting Kimi's real <think> segments from coldstart annotation JSONL,
truncating at the "let me verify / use tool" cues, then appending <answer>.

Motivation (§6.10 v5.1)
-----------------------
v5.0 used 5 fixed templates for the <think> segment → the model learned
"templated filling" rather than real recognition. v5.0 agent eval: only
3.4% of samples attempted notool (17/500, all failed).

v5.1 uses Kimi's actual think segments from samples where:
- Kimi predicted correctly (masked=0, km<=1)
- Kimi's first think showed HIGH CONFIDENCE (regex signals)
- Kimi's first think mentioned the final answer's CITY name (strong evidence)

These samples mostly come from part01/02 where Tavily was flaky, so Kimi had
to rely on visual priors alone — exactly the behavior we want to distill.

Construction rules:
1. Keep system + first user (image) message.
2. Replace all subsequent assistant/user messages with ONE assistant message:
   <think>truncated first think</think>
   <answer>Country, City, Lat, Lon</answer>
3. Truncation: cut the first think at the first occurrence of
   "Let me search/verify/check/use" or "I should search/verify/use", or
   any <tool_call>-like language. Keep only the "analysis → identification" part.

Usage:
    python3 data_pipeline/build_notool_from_kimi.py \
        --jsonl_dir /mnt/sh/mmvision/home/jonahli/save/agent/annotate/coldstart \
        --parts 00 01 02 \
        --output /mnt/sh/mmvision/home/jonahli/data_agent/sft/coldstart/train_notool_real_think.parquet \
        --max_samples 800 \
        --seed 42
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
from typing import Dict, List, Optional

import pandas as pd

# Same SYSTEM_PROMPT + USER_PROMPT as v5 / coldstart v4.
# Keep in sync with data_pipeline/build_sft_coldstart.py and build_notool_samples.py.
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

# Confidence signals in think (from previous analysis in prototype 50 discussion).
STRONG_POS = re.compile(
    r"(this is clearly|this is definitely|this is the famous|this is the iconic|"
    r"this is the|unmistakabl|clearly identif|strongly (?:suggests|resembles|indicates)|"
    r"signature (?:feature|architecture|colou?r)|distinctive (?:feature|landmark|architecture)|"
    r"the famous|iconic (?:building|landmark|structure)|"
    r"I recognize|I can identify|I know (?:this|that)|familiar (?:building|place|scene))",
    re.I,
)

# Truncation cues: cut at the first occurrence of these phrases to strip
# "let me verify with a tool" / "I should search" / tool_call-intent sentences.
TRUNCATE_CUES = re.compile(
    r"(?:\n+|\.\s+|^)("
    r"Let me\s+(?:search|verify|check|use|try|do|confirm)|"
    r"I(?:'ll| will)\s+(?:search|verify|check|use|try|do|confirm)|"
    r"I\s+should\s+(?:search|verify|check|use|try|call|do|confirm|utilize)|"
    r"I\s+need\s+to\s+(?:search|verify|check|use|try|confirm)|"
    r"I\s+could\s+use\s+(?:the\s+)?(?:image_search|text_search|image_zoom)|"
    r"I\s+can\s+use\s+(?:the\s+)?(?:image_search|text_search|image_zoom)|"
    r"I\s+(?:might|could|will)\s+(?:use|try|search|verify)\s+(?:the\s+)?(?:tool|image_search|text_search)|"
    r"Since\s+this\s+is\s+a\s+distinctive\s+landmark,?\s+I\s+should|"
    r"To\s+(?:verify|confirm|check|be certain|be sure)|"
    r"For\s+verification|"
    r"Let'?s\s+(?:search|verify|check|try|use)|"
    r"Now\s+I(?:'ll| will)\s+(?:search|verify|check|use|call)|"
    r"However,?\s+(?:to verify|to confirm|for verification|I should)|"
    r"But\s+(?:to verify|to confirm|let me|I should)"
    r")",
    re.IGNORECASE,
)

# Also strip any <tool_call> block that might have leaked into the think string.
TOOL_CALL_RE = re.compile(r"<tool_call>.*?(?:</tool_call>|$)", re.DOTALL)


def _extract_first_think(messages: List[dict]) -> Optional[str]:
    """Return the first <think>...</think> body from the first assistant message."""
    for m in messages:
        if m.get("role") != "assistant":
            continue
        c = m.get("content", "")
        if isinstance(c, list):
            c = "\n".join(
                str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in c
            )
        c = str(c)
        tm = re.search(r"<think>(.*?)</think>", c, re.DOTALL)
        if tm:
            return tm.group(1).strip()
    return None


def _extract_final_answer(messages: List[dict]) -> Optional[str]:
    """Return the final <answer>...</answer> body."""
    for m in reversed(messages):
        if m.get("role") != "assistant":
            continue
        c = m.get("content", "")
        if isinstance(c, list):
            c = "\n".join(
                str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in c
            )
        c = str(c)
        am = re.search(r"<answer>(.*?)</answer>", c, re.DOTALL)
        if am:
            return am.group(1).strip()
    return None


def _truncate_think(think: str) -> str:
    """
    Strip the "let me verify" tail and any tool_call remnants from the think.
    Return the analysis/identification part only.
    """
    # 1. Remove any inline <tool_call> blocks
    think = TOOL_CALL_RE.sub("", think)

    # 2. Cut at the first TRUNCATE_CUE occurrence
    m = TRUNCATE_CUES.search(think)
    if m:
        think = think[: m.start()].rstrip(" .")
        think += "."  # close sentence neatly

    # 3. If the think ends with "So the location is ..." or similar, keep it.
    # Otherwise, append a minimal confidence statement.
    tail = think[-200:].lower()
    if not any(
        kw in tail
        for kw in (
            "the location is",
            "coordinates are",
            "i can identify",
            "no tool",
            "without needing",
            "directly",
            "confident",
            "i recognize",
        )
    ):
        think += (
            "\n\nBased on these visual cues, I am confident in identifying "
            "this location directly without needing external verification."
        )

    return think.strip()


def _has_api_error(messages: List[dict]) -> bool:
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, list):
            c = " ".join(
                str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in c
            )
        s = str(c)
        if "search failed" in s.lower() or "api keys exhausted" in s.lower():
            return True
    return False


def _first_image_path(entry: dict) -> Optional[str]:
    """Get the original image path from entry.images[0]."""
    imgs = entry.get("images", [])
    if not imgs:
        return None
    first = imgs[0]
    if isinstance(first, dict):
        return str(first.get("image", "") or first.get("path", "")).strip() or None
    if isinstance(first, str):
        return first
    return None


def process_entry(entry: dict) -> Optional[Dict]:
    """Extract a notool sample from a Kimi annotation entry, or None if unusable."""
    masked = entry.get("masked", 1)
    km = entry.get("km")
    if masked != 0:
        return None
    if km is None or km >= 1:  # only very-easy samples
        return None

    messages = entry.get("messages", [])
    if not messages:
        return None

    think = _extract_first_think(messages)
    answer = _extract_final_answer(messages)
    if not think or not answer:
        return None

    # Confidence filter: must have a positive confidence signal in first think
    n_pos = len(STRONG_POS.findall(think))
    if n_pos < 1:
        return None

    # Answer must contain a city token (at least 2 commas → "Country, City, Lat, Lon")
    parts = [p.strip() for p in answer.split(",")]
    if len(parts) < 4:
        return None
    city = parts[1]
    if len(city) < 2 or city.lower() in ("none", "unknown", "null"):
        return None

    # Require: city name appears in think (strong evidence Kimi actually identified it)
    if not re.search(r"\b" + re.escape(city) + r"\b", think, re.I):
        return None

    truncated = _truncate_think(think)

    # Final sanity: reject if the truncated think still discusses tools
    # (Kimi's "should I use image_search or text_search?" meta-reasoning —
    # we want pure visual identification, not tool deliberation).
    _TOOL_MENTION = re.compile(
        r"(image_search|text_search|image_zoom|tool_call|bbox_2d|let me search|let me verify|call.*tool|use.*tool)",
        re.IGNORECASE,
    )
    if _TOOL_MENTION.search(truncated):
        return None

    # Find image path — look for it in entry.images
    image_path = _first_image_path(entry)
    if not image_path:
        return None

    assistant_content = f"<think>\n{truncated}\n</think>\n\n<answer>{answer}</answer>"
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": USER_PROMPT},
            {"role": "assistant", "content": assistant_content},
        ],
        "images": [{"image": image_path}],
        "_meta_has_api_err": _has_api_error(messages),
        "_meta_km": km,
        "_meta_id": entry.get("id", ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--jsonl_dir",
        default="/mnt/sh/mmvision/home/jonahli/save/agent/annotate/coldstart",
    )
    ap.add_argument("--parts", nargs="*", default=None,
                    help="Parts to include. Default: all found (00, 01, 02, …).")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_samples", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Auto-detect
    if args.parts is None:
        files = sorted(glob.glob(os.path.join(args.jsonl_dir, "part*.jsonl")))
        parts = [
            os.path.basename(f).replace("part", "").replace(".jsonl", "") for f in files
        ]
    else:
        parts = args.parts
    print(f"[notool-real] parts: {parts}")

    rng = random.Random(args.seed)

    all_candidates = []
    stats = {"total": 0, "usable": 0}
    per_part = {}
    per_source_err = {"clean": 0, "has_err": 0}
    for p in parts:
        path = os.path.join(args.jsonl_dir, f"part{p}.jsonl")
        if not os.path.exists(path):
            print(f"  SKIP: {path} not found")
            continue
        n_part = 0
        n_usable = 0
        with open(path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                stats["total"] += 1
                n_part += 1
                row = process_entry(entry)
                if row is None:
                    continue
                stats["usable"] += 1
                n_usable += 1
                if row["_meta_has_api_err"]:
                    per_source_err["has_err"] += 1
                else:
                    per_source_err["clean"] += 1
                all_candidates.append(row)
        per_part[p] = (n_part, n_usable)
        print(f"  part{p}: {n_part} entries → {n_usable} usable notool candidates")

    print(f"\n[notool-real] total: {stats['total']}, usable: {stats['usable']}")
    print(f"  clean (no API err): {per_source_err['clean']}")
    print(f"  has_err:            {per_source_err['has_err']}")

    if not all_candidates:
        print("No usable candidates!")
        return

    rng.shuffle(all_candidates)
    chosen = all_candidates[: args.max_samples]
    print(f"[notool-real] sampled {len(chosen)} out of {len(all_candidates)}")

    # Drop meta fields before writing
    out_rows = [
        {"messages": c["messages"], "images": c["images"]} for c in chosen
    ]
    out_df = pd.DataFrame(out_rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df.to_parquet(args.output, index=False)
    print(f"[notool-real] wrote {len(out_df)} samples to {args.output}")

    # Show 2 examples
    print("\n[notool-real] sample previews:")
    for ex in chosen[:2]:
        print("  ===")
        print(
            f"  id={ex['_meta_id']} km={ex['_meta_km']:.3f} "
            f"has_err={ex['_meta_has_api_err']}"
        )
        asst = ex["messages"][-1]["content"]
        if len(asst) > 500:
            asst = asst[:500] + "..."
        print(f"  assistant: {asst}")


if __name__ == "__main__":
    main()
