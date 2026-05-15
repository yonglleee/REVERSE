#!/usr/bin/env python3
"""
build_sft_coldstart.py — Build SFT cold-start parquet from Kimi K2d6 annotation JSONL.

Each row in the output parquet is one multi-turn conversation:
  - messages: list of {role, content} dicts (full Kimi trajectory)
      - system: system prompt (zoom + image_search + text_search, with <tool_call> format examples)
      - user: original question with <image> placeholder
      - assistant: <think>...</think> + <useful>[...]</useful> + <tool_call>...</tool_call>  (or final answer)
      - user: <tool_response>...</tool_response>  (tool outputs)
      - ... (repeated as needed)
      - assistant: final answer with <useful>[...]</useful> + <answer>Country, City, Lat, Lon</answer>
  - images: list of {"image": path} dicts (original + zoom crops, in order)

Filtering:
  - masked == 0  (Kimi predicted correctly)
  - km <= 200    (within 200 km of ground truth → high quality trajectory)
  - n_tool_calls >= 1  (has at least one tool use)
  - n_nudges <= 5  (filter overly confused trajectories)
  - final answer is not "None, None"
  - [optional, --drop_api_errors] no <tool_response> contains API failure strings
    (e.g. "Tavily search failed: all API keys exhausted", "Image search failed",
    "COS upload error"). Trajectories with such errors teach the model bad
    "tool failed → hallucinate answer" behavior.

Image alignment:
  - The JSONL images list contains paths: img_0.jpg (original), img_1..img_N.jpg (zoom crops).
  - Messages reference images via `<image>\\nimage_id: N` placeholders.
  - For SFT, we replace `image_id: N` with the N-th consecutive <image> token, matching
    the MultiTurnSFTDataset's sequential image list lookup.
  - The `images` column in the output parquet lists image paths in the order they appear.

Answer format:
  - Kimi outputs: <answer>Country, City, Lat, Lon</answer>
  - SFT target: <answer>Country, City, Lat, Lon</answer>  (keep as-is)
  - <useful>[...]</useful> tags are KEPT as supervision signal for learning to judge useful results.

Usage:
  python data_pipeline/build_sft_coldstart.py \\
    --jsonl_dir /mnt/sh/mmvision/home/jonahli/save/agent/annotate/coldstart \\
    --out_dir /mnt/sh/mmvision/home/jonahli/data_agent/sft/coldstart \\
    --parts 00 01 \\
    --val_ratio 0.05 \\
    --seed 42

  # Auto-detect all parts:
  python data_pipeline/build_sft_coldstart.py
"""

import argparse
import json
import os
import re
import random
from typing import Optional, List, Tuple


# ── System prompt (zoom + image_search + text_search) ────────────────────────
SYSTEM_PROMPT = (
    "You are a geolocation expert. Given an image, identify its location.\n"
    "You have three tools:\n"
    "1. `image_search_tool`: Reverse image search using a cropped region. Best for distinctive landmarks, buildings, or scenes. Returns matching web pages.\n"
    "2. `text_search_tool`: Search the web with natural language queries. Use for visible text/signs, landmark names, or any clues found from image search results.\n"
    "3. `image_zoom_in_tool`: Zoom into a region to read text/inscriptions that are too small at full scale.\n"
    "\n"
    "Decision rules:\n"
    "  \u2022 If you are HIGHLY CONFIDENT of the exact location (world-famous landmark, legible place-name text, familiar scene), provide your final <answer> DIRECTLY without any tool call. Explicitly state in <think> why no tool is needed.\n"
    "  \u2022 Distinctive landmark or scene visible (but uncertain of exact coords) \u2192 use `image_search_tool`\n"
    "  \u2022 Text/signs already legible \u2192 use `text_search_tool` directly\n"
    "  \u2022 Text/signs too small to read \u2192 use `image_zoom_in_tool` first, then `text_search_tool`\n"
    "  \u2022 image_search returns a landmark/location name \u2192 follow up with `text_search_tool`\n"
    "  \u2022 Do NOT use `image_zoom_in_tool` before `image_search_tool` \u2014 zoom does not improve image search\n"
    "  \u2022 When in doubt, use a tool. Only skip tools when you are certain.\n"
    "  \u2022 **Fallback rule**: If tools have failed, returned empty results, or exceeded their call limits, DO NOT keep retrying. Instead, provide your best `<answer>` based on visual priors in the image. Never finish a response without either a `<tool_call>` or an `<answer>`.\n"
    "\n"
    "For EVERY response, first enclose your reasoning in <think> </think> tags, then output EXACTLY ONE of:\n"
    '<tool_call>{"name": "image_search_tool", "arguments": {"bbox_2d": [x1, y1, x2, y2], "goal": "..."}}</tool_call>\n'
    "or:\n"
    '<tool_call>{"name": "text_search_tool", "arguments": {"query": "your query"}}</tool_call>\n'
    "or (parallel):\n"
    '<tool_call>{"name": "text_search_tool", "arguments": {"query": ["query one", "query two"]}}   </tool_call>\n'
    "or:\n"
    '<tool_call>{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [x1, y1, x2, y2]}}</tool_call>\n'
    "or your final answer in <answer> tags (when confident without needing a tool).\n"
    "\n"
    "After receiving tool results, output on its own line: "
    "<useful>[i, j, ...]</useful> listing the 1-based indices of results that match this specific image "
    "(i.e., mention the actual location, landmark, or geographic region shown in the image). "
    "Results about a different place are NOT useful even if they contain geographic information. "
    "Output <useful>[]</useful> if none match. "
    "Example: after receiving search results, your response must start with <think>...</think> then immediately <useful>[1, 3]</useful>.\n"
    "\n"
    "Final answer format: <answer>Country, City, Latitude, Longitude</answer>. "
    "e.g. <answer>Italy, Golfo Arnaci, 40.9606, 9.5873</answer>"
)

# User prompt with <image> placeholder
USER_PROMPT = (
    "<image>\n"
    "Analyze the architectural styles, vegetation, street infrastructure, and cultural markers "
    "in this image. Based on these visual cues, determine the location.\n\n"
    "Answer strictly in the following format:\n"
    "Country, City, Latitude, Longitude. "
    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    "The reasoning process MUST BE enclosed within <think> </think> tags. "
    "Wrap your final answer in <answer> tags in the format: <answer>Country, City, Latitude, Longitude</answer>. "
    "e.g. <answer>Italy, Golfo Arnaci, 40.9606, 9.5873</answer>"
)

# Nudge message prefix (used to detect and remove nudge messages)
_NUDGE_PREFIX = "Your previous response ended without a tool call or final answer"

# API failure markers injected into <tool_response> when the annotator's tool call failed.
# Trajectories containing these are polluted: the model learns to "hallucinate an answer
# after tool failure" because Kimi gave a correct answer anyway based on its own priors.
_API_ERROR_PATTERNS = [
    "Tavily search failed",
    "Image search failed",
    "all API keys exhausted",
    "COS upload error",
    "qcloud_cos",
]


def make_relative(path: str, data_root: str) -> str:
    """Convert absolute path to relative path under data_root."""
    if not data_root or not path:
        return path
    data_root = data_root.rstrip('/')
    if path.startswith(data_root + '/'):
        return path[len(data_root) + 1:]
    return path


_TOOL_CALL_RE = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)


def _has_full_image_search(messages: list) -> bool:
    """Return True if any image_search_tool call uses near-full-image bbox [0,0,1000,1000]."""
    import json as _json
    for m in messages:
        if m.get('role') != 'assistant':
            continue
        c = m.get('content', '') or ''
        if isinstance(c, list):
            c = ' '.join(x.get('text', '') for x in c if isinstance(x, dict))
        for tc_m in _TOOL_CALL_RE.finditer(c):
            try:
                tc = _json.loads(tc_m.group(1))
                if tc.get('name') == 'image_search_tool':
                    b = tc.get('arguments', {}).get('bbox_2d', [])
                    if b and b[0] <= 5 and b[1] <= 5 and b[2] >= 995 and b[3] >= 995:
                        return True
            except Exception:
                pass
    return False

def _has_api_error(messages: list) -> bool:
    """Return True if any message content contains an API failure marker."""
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, list):
            c = "\n".join(
                str(x.get("text", "")) if isinstance(x, dict) else str(x)
                for x in c
            )
        text = str(c)
        for pat in _API_ERROR_PATTERNS:
            if pat in text:
                return True
    return False


def _clean_final_answer(text: str, gt_lat: float, gt_lon: float,
                         gt_country: str, gt_city: str) -> Optional[str]:
    """
    Clean the final assistant message, keeping <answer>Country, City, Lat, Lon</answer> as-is.

    - Keep <useful>[...]</useful> tags as supervision signal.
    - Skip "None, None" answers.
    - If no <answer> tag found, return None (skip entry).
    """
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if not m:
        return None  # No answer tag — skip entry

    ans_str = m.group(1).strip()
    if ans_str.lower().startswith("none"):
        return None  # "None, None" — skip entry

    return text.rstrip()


def _is_nudge_msg(msg: dict) -> bool:
    """Return True if this user message is a nudge (prompt to continue)."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content.strip().startswith(_NUDGE_PREFIX)
    return False


def _rewrite_user_msg1(msg: dict) -> dict:
    """
    Rewrite the first user message (which has content as a list of text parts)
    to use the new USER_PROMPT with <image> placeholder.

    Original format (content is a list):
        [{"type": "text", "text": "<image>\\nimage_id: 0"},
         {"type": "text", "text": "Analyze..."}]

    Output format (content is a string):
        "<image>\\nAnalyze... (new prompt)"
    """
    return {"role": "user", "content": USER_PROMPT}


def _clean_assistant_msg(msg: dict) -> dict:
    """
    Clean an intermediate assistant message:
    - Keep <useful>[...]</useful> tags as supervision signal for learning to judge useful results
    - Keep <think>...</think> and <tool_call>...</tool_call> as-is
    """
    content = msg.get("content", "")
    if isinstance(content, str):
        content = content.rstrip()
    return {"role": "assistant", "content": content}


def _build_image_list(jsonl_images: List[str], used_image_ids: List[int]) -> List[dict]:
    """
    Build the images list for MultiTurnSFTDataset.

    Only include images that are actually referenced in the messages (via image_id).
    Returns list of {"image": path} dicts in the order they appear in messages.
    """
    images = []
    for img_id in used_image_ids:
        if img_id < len(jsonl_images):
            images.append({"image": jsonl_images[img_id]})
    return images


def _replace_image_id_in_content(content: str) -> Tuple[str, Optional[int]]:
    """
    Replace `<image>\\nimage_id: N` with just `<image>`.

    Returns (new_content, image_id) where image_id is the integer N found,
    or (new_content, None) if no image_id found.
    """
    m = re.search(r'<image>\s*\nimage_id:\s*(\d+)', content)
    if m:
        image_id = int(m.group(1))
        new_content = re.sub(r'<image>\s*\nimage_id:\s*\d+', '<image>', content)
        return new_content, image_id
    # Also handle standalone <image> (no image_id annotation)
    if '<image>' in content:
        return content, None
    return content, None


def process_entry(entry: dict, data_root: str = '') -> Optional[dict]:
    """
    Convert one JSONL entry to a SFT parquet row.

    Returns None if the entry should be skipped.
    """
    gt_lat = float(entry.get("gt_lat", 0))
    gt_lon = float(entry.get("gt_lon", 0))
    gt_country = str(entry.get("gt_country", "Unknown"))
    gt_city = str(entry.get("gt_city", "Unknown"))

    # Build useful_results lookup: (tool, turn) → indices
    # Used to inject <useful> tags into assistant messages that lack them
    useful_by_turn: dict = {}
    for u in entry.get("useful_results", []):
        key = (u.get("tool", ""), int(u.get("turn", -1)))
        useful_by_turn[key] = u.get("indices", [])
    jsonl_images = entry.get("images", [])

    raw_messages = entry.get("messages", [])
    if not raw_messages:
        return None

    # ── Build new messages list ──────────────────────────────────────────────
    new_messages = []
    used_image_ids = []  # image_ids referenced in order of <image> tokens
    skip_next_assistant = False  # used to skip assistant responses to nudges
    last_assistant_raw_idx = -1  # absolute index of last assistant message in raw_messages
    prev_search_tool: Optional[str] = None   # tool that produced the last search response
    prev_tool_caller_idx: int = -1           # raw_messages index of the assistant that called it

    for i, msg in enumerate(raw_messages):
        role = msg.get("role", "")

        if role == "system":
            # Replace with new v4 system prompt
            new_messages.append({"role": "system", "content": SYSTEM_PROMPT})
            continue

        if role == "user":
            # Check if this is a nudge message
            if _is_nudge_msg(msg):
                # Skip nudge + the following assistant response
                skip_next_assistant = True
                continue

            if i == 1:
                # First user message: replace with new USER_PROMPT
                new_messages.append(_rewrite_user_msg1(msg))
                # Track image_id: the original image is always images[0]
                used_image_ids.append(0)
                continue

            # Other user messages: tool responses and zoom image references
            content = msg.get("content", "")
            if isinstance(content, list):
                # Multi-part content (shouldn't happen for non-first user msgs, but handle it)
                parts_text = "\n".join(p.get("text", "") for p in content if p.get("type") == "text")
                content = parts_text

            # Track which search tool responded and which assistant called it
            # (for <useful> injection in next assistant turn)
            if "Image search results" in content:
                prev_search_tool = "image_search_tool"
                prev_tool_caller_idx = last_assistant_raw_idx
            elif "Web search results" in content:
                prev_search_tool = "text_search_tool"
                prev_tool_caller_idx = last_assistant_raw_idx
            else:
                prev_search_tool = None
                prev_tool_caller_idx = -1

            # Replace image_id references with sequential <image> tokens
            new_content, img_id = _replace_image_id_in_content(content)
            if img_id is not None:
                used_image_ids.append(img_id)

            new_messages.append({"role": "user", "content": new_content})
            continue

        if role == "assistant":
            if skip_next_assistant:
                skip_next_assistant = False
                prev_search_tool = None
                prev_tool_caller_idx = -1
                continue

            last_assistant_raw_idx = i  # track absolute index
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "\n".join(p.get("text", "") for p in content if p.get("type") == "text")

            # Check if this is the last assistant message (final answer)
            is_last = (i == len(raw_messages) - 1)
            if is_last:
                # Clean final answer (keep <answer> tag as-is, remove artifacts)
                new_content = _clean_final_answer(content, gt_lat, gt_lon, gt_country, gt_city)
                if new_content is None:
                    return None  # Skip "None, None" or missing answer tag
                new_messages.append({"role": "assistant", "content": new_content})
            else:
                # Intermediate assistant message — clean up annotation artifacts
                cleaned = _clean_assistant_msg({"role": "assistant", "content": content})

                # Inject <useful>[...]</useful> if:
                # 1. Previous user turn was a search tool response
                # 2. This assistant message doesn't already have <useful>
                # 3. We have a useful_results annotation for (tool, caller_msg_idx)
                #    where caller_msg_idx = absolute index of the assistant that made the call
                c = cleaned["content"]
                if (prev_search_tool is not None
                        and prev_tool_caller_idx >= 0
                        and "<useful>" not in c
                        and (prev_search_tool, prev_tool_caller_idx) in useful_by_turn):
                    indices = useful_by_turn[(prev_search_tool, prev_tool_caller_idx)]
                    useful_tag = f" <useful>{json.dumps(indices)}</useful>"
                    if "</think>" in c:
                        c = c.replace("</think>", f"</think>\n{useful_tag}", 1)
                    else:
                        c = c + f"\n{useful_tag}"
                    cleaned["content"] = c

                new_messages.append(cleaned)

            prev_search_tool = None  # reset after each assistant turn
            prev_tool_caller_idx = -1
            continue

    # ── Sanity checks ────────────────────────────────────────────────────────
    if len(new_messages) < 3:
        return None  # Need at least system + user + assistant

    # Filter out excessively long trajectories (>=13 turns = >5 tool calls, likely confused)
    if len(new_messages) >= 13:
        return None

    # Final message must be assistant with an <answer> tag
    final_msg = new_messages[-1]
    if final_msg["role"] != "assistant":
        return None
    if "<answer>" not in final_msg["content"]:
        return None

    # ── Build images list ────────────────────────────────────────────────────
    # used_image_ids are in the order they appear in messages (sequential <image> tokens)
    # Verify all referenced images exist
    images = []
    for img_id in used_image_ids:
        if img_id < len(jsonl_images):
            img_path = jsonl_images[img_id]
            if os.path.exists(img_path):
                images.append({"image": make_relative(img_path, data_root)})
            else:
                # Image file missing — skip entry
                return None
        else:
            return None

    # Count <image> tokens in messages to ensure alignment
    n_image_tokens = sum(
        msg["content"].count("<image>")
        for msg in new_messages
        if isinstance(msg.get("content"), str)
    )
    if n_image_tokens != len(images):
        # Mismatch — skip to avoid training errors
        return None

    return {
        "messages": new_messages,
        "images": images,
        # Extra metadata (not used by trainer, but useful for debugging)
        "img_id": entry.get("id", ""),
        "km": entry.get("km"),
        "n_tool_calls": entry.get("n_tool_calls", 0),
    }


def process_jsonl(jsonl_path: str, max_nudges: int = 5,
                  drop_api_errors: bool = False, crop_filter: bool = False,
                  data_root: str = '') -> List[dict]:
    """
    Process one JSONL file and return list of SFT row dicts.

    Filters:
      - masked == 0
      - km <= 200 (or km is None but masked==0 means km is available)
      - n_tool_calls >= 1
      - n_nudges <= max_nudges
      - [if drop_api_errors] messages contain no API failure markers
    """
    if not os.path.exists(jsonl_path):
        print(f"  SKIP: JSONL not found: {jsonl_path}")
        return []

    with open(jsonl_path) as f:
        entries = [json.loads(l) for l in f if l.strip()]

    rows = []
    skipped_filter = 0
    skipped_process = 0
    skipped_api_err = 0

    for entry in entries:
        masked = entry.get("masked", 1)
        km = entry.get("km")
        n_tool_calls = entry.get("n_tool_calls", 0)
        n_nudges = entry.get("n_nudges", 0)

        # Quality filter
        if masked != 0:
            skipped_filter += 1
            continue
        if km is None or km > 200:
            skipped_filter += 1
            continue
        if n_tool_calls < 1:
            skipped_filter += 1
            continue
        if n_nudges > max_nudges:
            skipped_filter += 1
            continue
        if drop_api_errors and _has_api_error(entry.get("messages", [])):
            skipped_api_err += 1
            continue
        if crop_filter and _has_full_image_search(entry.get("messages", [])):
            skipped_api_err += 1
            continue

        row = process_entry(entry, data_root=data_root)
        if row is None:
            skipped_process += 1
            continue

        rows.append(row)

    msg = (f"  Total: {len(entries)}, filtered_out: {skipped_filter}, "
           f"process_skipped: {skipped_process}, accepted: {len(rows)}")
    if drop_api_errors:
        msg += f", api_err_dropped: {skipped_api_err}"
    print(msg)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl_dir",
                    default="/mnt/sh/mmvision/home/jonahli/save/agent/annotate/coldstart")
    ap.add_argument("--out_dir",
                    default="/mnt/sh/mmvision/home/jonahli/data_agent/sft/coldstart")
    ap.add_argument("--parts", nargs="*", default=None,
                    help="Parts to include (e.g. 00 01). Default: auto-detect all available.")
    ap.add_argument("--val_ratio", type=float, default=0.05,
                    help="Fraction of data for validation (default 0.05 = 5%%)")
    ap.add_argument("--max_nudges", type=int, default=5,
                    help="Max allowed nudge messages per trajectory (default 5)")
    ap.add_argument("--drop_api_errors", action="store_true",
                    help="Drop trajectories with API failure markers (Tavily/Image/COS).")
    ap.add_argument("--crop_filter", action="store_true",
                    help="Drop trajectories where any image_search uses near-full-image bbox.")
    ap.add_argument("--out_suffix", default="",
                    help="Suffix appended to output parquet filename "
                         "(e.g. '_clean' → train_sft_coldstart_clean.parquet)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_root", type=str, default="",
                    help="If set, strip this prefix from image paths to make them relative. "
                         "E.g. /mnt/sh/mmvision/home/jonahli/data_agent/REVERSE")
    args = ap.parse_args()

    import pandas as pd

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    # Auto-detect available parts
    if args.parts is None:
        import glob as _glob
        jsonl_files = sorted(_glob.glob(os.path.join(args.jsonl_dir, "part*.jsonl")))
        parts = [os.path.basename(f).replace("part", "").replace(".jsonl", "")
                 for f in jsonl_files]
        print(f"Auto-detected parts: {parts}")
    else:
        parts = args.parts

    all_rows = []
    for part in parts:
        jsonl_path = os.path.join(args.jsonl_dir, f"part{part}.jsonl")
        print(f"\nProcessing part{part} ...")
        rows = process_jsonl(jsonl_path, max_nudges=args.max_nudges,
                             drop_api_errors=args.drop_api_errors,
                             crop_filter=args.crop_filter,
                             data_root=args.data_root)
        all_rows.extend(rows)

    if not all_rows:
        print("No rows extracted. Check JSONL paths and filter criteria.")
        return

    print(f"\nTotal accepted: {len(all_rows)}")

    # Split train / val
    indices = list(range(len(all_rows)))
    random.shuffle(indices)
    n_val = max(1, int(len(all_rows) * args.val_ratio))
    val_indices = set(indices[:n_val])

    train_rows = [all_rows[i] for i in range(len(all_rows)) if i not in val_indices]
    val_rows = [all_rows[i] for i in range(len(all_rows)) if i in val_indices]

    # Drop extra metadata cols from val/train for cleaner parquet
    def to_sft_row(r):
        return {"messages": r["messages"], "images": r["images"]}

    train_df = pd.DataFrame([to_sft_row(r) for r in train_rows])
    val_df = pd.DataFrame([to_sft_row(r) for r in val_rows])

    train_path = os.path.join(args.out_dir, f"train_sft_coldstart{args.out_suffix}.parquet")
    val_path = os.path.join(args.out_dir, f"val_sft_coldstart{args.out_suffix}.parquet")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"\nTrain: {len(train_rows)} rows → {train_path}")
    print(f"Val:   {len(val_rows)} rows  → {val_path}")
    print(f"Train size: {os.path.getsize(train_path)/1024:.1f} KB")
    print(f"Val size:   {os.path.getsize(val_path)/1024:.1f} KB")

    # Quick sanity: print first row's messages summary
    if train_rows:
        r = train_rows[0]
        print(f"\n[Sample] img_id={r['img_id']} km={r['km']:.1f} n_tool_calls={r['n_tool_calls']}")
        print(f"  messages: {len(r['messages'])} turns")
        print(f"  images:   {len(r['images'])} images")
        for j, m in enumerate(r["messages"]):
            c = m["content"]
            print(f"  [{j}] {m['role']}: {str(c)[:100]}")


if __name__ == "__main__":
    main()
