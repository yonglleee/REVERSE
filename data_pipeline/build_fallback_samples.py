#!/usr/bin/env python3
"""
build_fallback_samples.py — Build "tool-failed → fallback answer" teaching
samples from Kimi coldstart annotations where ALL tool calls failed but Kimi
still gave a correct answer based on visual priors.

Motivation (§6.13 → v5.3)
-------------------------
v5.2 model has 100/500 max_turns_exceeded failures: when tools return empty or
exceed call limits, the model doesn't know how to gracefully exit with a
fallback answer. It just keeps calling tools until max_turns.

v4 accidentally taught "tool failed → hallucinate answer" (bad), but we threw
away all those samples when building v4-clean / v5.x. We need to put back a
small, CLEAN subset: samples where Kimi correctly identified the location from
visual priors alone, with tools just failing (not giving bad info).

Sample trajectory:
    system + user(image)
    → assistant: <think>visual analysis, try tool</think> <tool_call>...</tool_call>
    → user: <tool_response>Tool error: XXX</tool_response>
    → assistant: <think>Tool failed. Based on visual priors, this is YYY</think> <answer>...</answer>

Selection criteria:
- masked=0, km<=200  (Kimi was correct)
- n_tool_calls>=1
- ALL tool_responses contain error markers (no successful responses)
- First <think> mentions the final answer's city (real visual identification,
  not lucky guess)

Construction:
1. Keep system + user (first image message)
2. Assistant turn 1: use Kimi's first <think> + first <tool_call>
3. User turn 2: canonical "Tool error: search returned no useful information"
4. Assistant turn 2: short fallback think + <answer>
   - If Kimi's final think (after last nudge) is good (mentions city + priors), use it trimmed
   - Otherwise build a minimal fallback think: "The tool failed. Based on visual cues in the image, I can identify this as CITY, COUNTRY."

Usage:
    python3 data_pipeline/build_fallback_samples.py \
        --jsonl_dir /mnt/sh/mmvision/home/jonahli/save/agent/annotate/coldstart \
        --parts 00 01 02 \
        --output /mnt/sh/mmvision/home/jonahli/data_agent/sft/coldstart/train_fallback_samples.parquet
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Optional

import pandas as pd


# Canonical SYSTEM_PROMPT — extract from build_sft_coldstart.py at runtime
def _load_canonical_prompt() -> str:
    path = "/mnt/sh/mmvision/home/jonahli/projects/tusou/data_pipeline/build_sft_coldstart.py"
    with open(path) as f:
        src = f.read()
    m = re.search(r"SYSTEM_PROMPT = \((.*?)\)\n\n", src, re.DOTALL)
    if not m:
        raise RuntimeError("Could not find SYSTEM_PROMPT in build_sft_coldstart.py")
    return eval(m.group(0).split("= ", 1)[1].rstrip())


SYSTEM_PROMPT = _load_canonical_prompt()

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

# Regexes
FIRST_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

# Canonical fallback tool_response (replaces varied original failure messages)
TOOL_ERROR_RESPONSE = (
    "<tool_response>\n"
    "Tool error: the search returned no useful information. "
    "Please provide your best answer based on visual analysis.\n"
    "</tool_response>"
)


def _flatten(content) -> str:
    if isinstance(content, list):
        parts = []
        for x in content:
            if isinstance(x, dict):
                parts.append(str(x.get("text", "")))
            else:
                parts.append(str(x))
        return "\n".join(parts)
    return str(content)


def _extract_first_think_and_toolcall(messages: list) -> Optional[tuple[str, str]]:
    """Return (first_think, first_tool_call_json_str) or None."""
    for m in messages:
        if m.get("role") != "assistant":
            continue
        c = _flatten(m.get("content", ""))
        tm = THINK_RE.search(c)
        tcm = FIRST_TOOL_CALL_RE.search(c)
        if tm and tcm:
            think = tm.group(1).strip()
            tc_json = tcm.group(1).strip()
            # Validate json parses
            try:
                json.loads(tc_json)
            except Exception:
                return None
            return (think, tc_json)
    return None


def _extract_final_answer_and_closing_think(messages: list) -> Optional[tuple[str, str]]:
    """Walk from the end, find the last assistant msg with <answer>; also
    return the think from the same msg if present (Kimi's final fallback reasoning).
    """
    for m in reversed(messages):
        if m.get("role") != "assistant":
            continue
        c = _flatten(m.get("content", ""))
        am = ANSWER_RE.search(c)
        if am:
            ans = am.group(1).strip()
            # Get the think from this same message (final fallback reasoning)
            tm = THINK_RE.search(c)
            closing_think = tm.group(1).strip() if tm else ""
            return (ans, closing_think)
    return None


def _trim_closing_think(think: str, city: str) -> Optional[str]:
    """Clean up Kimi's final fallback think:
    - Remove sentences saying 'let me try another search / tool'
    - Ensure it mentions the city
    - Keep only visual-prior reasoning
    """
    # Drop sentences starting with "Let me try" / "Let me search" / similar
    cue = re.compile(
        r"(?:\n+|\.\s+|^)("
        r"Let me (?:try|search|verify|check|use|do)|"
        r"I(?:'ll| will) (?:try|search|verify|check|use|do)|"
        r"I should (?:try|search|verify|check|use|call)|"
        r"I need to (?:try|search|verify|check|use)|"
        r"Let'?s (?:try|search|verify|check|use)|"
        r"Now I(?:'ll| will)"
        r")",
        re.IGNORECASE,
    )
    m = cue.search(think)
    if m:
        think = think[: m.start()].rstrip(" .\n") + "."

    # Must mention the city
    if not re.search(r"\b" + re.escape(city) + r"\b", think, re.I):
        return None

    # Remove <useful> tags remnants if present
    think = re.sub(r"<useful>.*?</useful>", "", think, flags=re.DOTALL).strip()

    # Remove any tool_call leak
    think = re.sub(r"<tool_call>.*?(?:</tool_call>|$)", "", think, flags=re.DOTALL).strip()

    # Append fallback closing line
    if "no tool" not in think.lower() and "without" not in think.lower():
        think += (
            "\n\nThe tool could not provide additional information, but based on these visual "
            "cues I am confident in my identification."
        )
    return think


def _has_api_error(messages: list) -> bool:
    for m in messages:
        c = _flatten(m.get("content", ""))
        if "search failed" in c.lower() or "api keys exhausted" in c.lower() or "API key not set" in c:
            return True
    return False


def _has_successful_tool_response(messages: list) -> bool:
    """Check if ANY tool_response was successful (has results, not an error)."""
    for m in messages:
        if m.get("role") != "user":
            continue
        c = _flatten(m.get("content", ""))
        cl = c.lower()
        if "<tool_response>" not in cl and "zoomed in" not in cl and "search results" not in cl:
            continue
        # Is it a success?
        is_err = (
            "search failed" in cl
            or "api keys exhausted" in cl
            or "API key not set" in c
            or "tool call error" in cl
        )
        if is_err:
            continue
        if "search results" in cl or "zoomed in on the image" in cl:
            return True
    return False


def _first_image_path(entry: dict) -> Optional[str]:
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
    """Construct a fallback SFT sample. Return None if unusable."""
    masked = entry.get("masked", 1)
    km = entry.get("km")
    n_tool = entry.get("n_tool_calls", 0)
    if masked != 0 or km is None or km > 200 or n_tool < 1:
        return None

    messages = entry.get("messages", [])
    if not messages:
        return None

    # MUST have API error AND no successful tool response
    if not _has_api_error(messages):
        return None
    if _has_successful_tool_response(messages):
        return None

    first = _extract_first_think_and_toolcall(messages)
    if not first:
        return None
    first_think, first_tool_call = first

    final = _extract_final_answer_and_closing_think(messages)
    if not final:
        return None
    final_answer, closing_think_raw = final

    # Parse city from answer
    parts = [p.strip() for p in final_answer.split(",")]
    if len(parts) < 4:
        return None
    city = parts[1]
    if len(city) < 2 or city.lower() in ("none", "unknown", "null"):
        return None

    # First think should also mention the city (stronger signal of true identification)
    if not re.search(r"\b" + re.escape(city) + r"\b", first_think, re.I):
        return None

    # Trim closing think
    closing = _trim_closing_think(closing_think_raw, city)
    if closing is None:
        # Build minimal fallback think
        country = parts[0]
        closing = (
            f"The tool did not return useful information. Based on the visual cues I "
            f"analyzed earlier (architectural style, vegetation, cultural markers), I am "
            f"confident this is {city}, {country}. I will provide my best answer based on "
            f"these visual priors."
        )

    # Clean first_think of any trailing "let me search" lines
    first_think_clean = first_think.strip()
    # Remove <useful> if leaked
    first_think_clean = re.sub(r"<useful>.*?</useful>", "", first_think_clean, flags=re.DOTALL).strip()
    # Strip trailing tool_call-intention lines from first_think (we keep <tool_call> tag outside)
    tail_cue = re.compile(
        r"\n+\s*(Let me (?:try|search|verify|check|use|do)|"
        r"I(?:'ll| will) (?:try|search|verify|check|use|do)|"
        r"I should (?:try|search|verify|check|use|call)|"
        r"Let'?s (?:try|search|verify|check|use)).*$",
        re.IGNORECASE | re.DOTALL,
    )
    first_think_clean = tail_cue.sub("", first_think_clean).rstrip(" .") + "."

    # Build assistant messages
    asst1 = f"<think>\n{first_think_clean}\n</think>\n<tool_call>{first_tool_call}</tool_call>"
    asst2 = f"<think>\n{closing}\n</think>\n\n<answer>{final_answer}</answer>"

    img_path = _first_image_path(entry)
    if not img_path:
        return None

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
            {"role": "assistant", "content": asst1},
            {"role": "user", "content": TOOL_ERROR_RESPONSE},
            {"role": "assistant", "content": asst2},
        ],
        "images": [{"image": img_path}],
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--jsonl_dir",
        default="/mnt/sh/mmvision/home/jonahli/save/agent/annotate/coldstart",
    )
    ap.add_argument("--parts", nargs="*", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_samples", type=int, default=1000)
    args = ap.parse_args()

    if args.parts is None:
        parts = sorted(
            [
                os.path.basename(f).replace("part", "").replace(".jsonl", "")
                for f in glob.glob(os.path.join(args.jsonl_dir, "part*.jsonl"))
            ]
        )
    else:
        parts = args.parts
    print(f"[fallback] parts: {parts}")

    out_rows = []
    per_part = {}
    for p in parts:
        path = os.path.join(args.jsonl_dir, f"part{p}.jsonl")
        if not os.path.exists(path):
            print(f"  SKIP: {path}")
            continue
        n_in, n_out = 0, 0
        with open(path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                n_in += 1
                row = process_entry(entry)
                if row is not None:
                    out_rows.append(row)
                    n_out += 1
        per_part[p] = (n_in, n_out)
        print(f"  part{p}: {n_in} entries → {n_out} fallback samples")

    print(f"\n[fallback] total: {len(out_rows)}")
    if not out_rows:
        print("  no usable samples")
        return

    import random

    random.shuffle(out_rows)
    out_rows = out_rows[: args.max_samples]

    df = pd.DataFrame(out_rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"[fallback] wrote {len(df)} samples to {args.output}")

    # Show example
    if out_rows:
        ex = out_rows[0]
        print("\n[fallback] example sample:")
        for m in ex["messages"]:
            c = m["content"]
            if len(c) > 300:
                c = c[:250] + "..."
            print(f"  [{m['role']}] {c}")


if __name__ == "__main__":
    main()
