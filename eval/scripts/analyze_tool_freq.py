#!/usr/bin/env python3
"""
Analyze tool call frequency from validation JSONL files and optionally update proposal.md.

Usage:
    python3 analyze_tool_freq.py --data_dir <dir>              # print table
    python3 analyze_tool_freq.py --data_dir <dir> --update_proposal  # update proposal.md
    python3 analyze_tool_freq.py --all                         # analyze v1 + v2 + v3 and update proposal
"""
import json
import os
import re
import argparse
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE = "/mnt/sh/mmvision/home/jonahli/save/agent/rollout_output/multiturn"
PROPOSAL = "/mnt/sh/mmvision/home/jonahli/projects/tusou/proposal.md"
CHANGELOG = "/mnt/sh/mmvision/home/jonahli/projects/CHANGELOG.md"

EXPERIMENT_DIRS = {
    "v1": "Qwen3-VL-4B-Instruct-geoloc-zoom-imgsearch-searchreward-NNODES4",
    "v2": "Qwen3-VL-4B-Instruct-geoloc-zoom-imgsearch-basereward-v2-NNODES4",
}


def count_tool_calls_in_text(text, tool_name):
    """Count actual tool invocations via <tool_call> XML blocks in the conversation string."""
    # Match <tool_call> blocks containing the tool name
    pattern = r'<tool_call>\s*\{[^<]*"name"\s*:\s*"' + re.escape(tool_name) + r'"'
    return len(re.findall(pattern, text, re.DOTALL))


def analyze_step(jsonl_path):
    """Analyze a single step's JSONL file.

    Records have keys: input (str), output (str), reward (float), score (float), ...
    Tool calls appear as <tool_call>{"name": "..."}...</tool_call> inside the 'input' field
    which stores the full conversation history including tool responses.
    """
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue

    if not records:
        return None

    zooms, searches, rewards = [], [], []
    zoom_search, zoom_only, search_only, neither = 0, 0, 0, 0

    for r in records:
        # reward
        reward = r.get("reward", r.get("score", None))
        if reward is not None:
            rewards.append(float(reward))

        # The full conversation (including tool calls) is stored in 'input'
        # The final assistant turn is in 'output'
        inp = r.get("input", "")
        out = r.get("output", "")
        combined = (inp if isinstance(inp, str) else json.dumps(inp)) + \
                   (out if isinstance(out, str) else json.dumps(out))

        n_zoom = count_tool_calls_in_text(combined, "image_zoom_in_tool")
        n_search = count_tool_calls_in_text(combined, "image_search_tool")

        zooms.append(n_zoom)
        searches.append(n_search)

        has_zoom = n_zoom > 0
        has_search = n_search > 0
        if has_zoom and has_search:
            zoom_search += 1
        elif has_zoom:
            zoom_only += 1
        elif has_search:
            search_only += 1
        else:
            neither += 1

    n = len(records)
    avg_zoom = sum(zooms) / n if n else 0
    avg_search = sum(searches) / n if n else 0
    avg_reward = sum(rewards) / len(rewards) if rewards else 0
    return {
        "n": n,
        "avg_zoom": avg_zoom,
        "avg_search": avg_search,
        "zoom_search_pct": zoom_search / n * 100 if n else 0,
        "zoom_only_pct": zoom_only / n * 100 if n else 0,
        "search_only_pct": search_only / n * 100 if n else 0,
        "neither_pct": neither / n * 100 if n else 0,
        "avg_reward": avg_reward,
        "zoom_search": zoom_search,
        "zoom_only": zoom_only,
        "search_only": search_only,
        "neither": neither,
    }


def analyze_dir(data_dir):
    """Analyze all step jsonl files in a directory."""
    data_dir = Path(data_dir)
    results = {}
    for jsonl in sorted(data_dir.glob("*.jsonl"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0):
        step = int(jsonl.stem) if jsonl.stem.isdigit() else jsonl.stem
        stats = analyze_step(jsonl)
        if stats:
            results[step] = stats
    return results


def format_table(results, title=""):
    """Format results as a markdown table."""
    lines = []
    if title:
        lines.append(f"\n**{title}**\n")
    lines.append("| Step | n | avg_zoom | avg_search | zoom+search% | zoom-only% | search-only% | neither% | avg_reward |")
    lines.append("|------|---|----------|------------|--------------|------------|--------------|----------|------------|")
    for step in sorted(k for k in results if isinstance(k, int)):
        s = results[step]
        lines.append(
            f"| {step} | {s['n']} | {s['avg_zoom']:.3f} | {s['avg_search']:.3f} "
            f"| {s['zoom_search_pct']:.1f}% | {s['zoom_only_pct']:.1f}% "
            f"| {s['search_only_pct']:.1f}% | {s['neither_pct']:.1f}% "
            f"| {s['avg_reward']:.4f} |"
        )
    return "\n".join(lines)


def build_v1_section(v1_results):
    table = format_table(v1_results)
    return f"""\n#### v1 工具调用频次演变（validation set，step 10-117）

{table}

**关键观察**：v1（+SearchReward，`base_call_reward=0`）从第10步开始 search 调用频次迅速归零，
模型完全放弃 search 工具，最终 avg_search ≈ 0。这是因为 search 只有在返回正样本时才有奖励，
而 zoom（IOU ≥ 0.7 时给 +0.1）相对容易获得正奖励，GRPO 直接抑制了 search。
"""


def build_v2_section(v2_results):
    table = format_table(v2_results)
    return f"""\n#### v2 工具调用频次演变（validation set，262条，step 10-50）

{table}

**关键观察**：v2（BaseReward，`base_call_reward=0.25`）成功使 search 在训练过程中保持活跃，
但 zoom 逐步被 GRPO 抑制（72%→44%）。分析 step 50 奖励分布发现，
search-only rollout 平均奖励（0.6755）> zoom+search（0.5700），
GRPO 正是利用这一奖励差异抑制了 zoom。
"""


SECTION_ANCHOR_V1 = "#### v1 工具调用频次演变"
SECTION_ANCHOR_V2 = "#### v2 工具调用频次演变"


def update_proposal(v1_results, v2_results):
    """Update proposal.md with v1 and v2 tool frequency tables."""
    with open(PROPOSAL, "r") as f:
        content = f.read()

    # Build new sections
    v1_section = build_v1_section(v1_results)
    v2_section = build_v2_section(v2_results)

    # Replace or insert v1 section
    if SECTION_ANCHOR_V1 in content:
        # Replace existing v1 section up to next #### or ###
        pattern = r"#### v1 工具调用频次演变.*?(?=\n#### |\n### |\Z)"
        new_content = re.sub(pattern, v1_section.strip(), content, flags=re.DOTALL)
        if new_content == content:
            print("[WARN] v1 section replacement had no effect, appending")
        else:
            content = new_content
            print("[OK] Replaced v1 tool frequency section")
    else:
        # Insert before v2 section or at end of Phase 3
        if SECTION_ANCHOR_V2 in content:
            content = content.replace(SECTION_ANCHOR_V2,
                                       v1_section.strip() + "\n\n" + SECTION_ANCHOR_V2)
            print("[OK] Inserted v1 tool frequency section before v2 section")
        else:
            print("[WARN] Could not find v2 section anchor, appending v1 section to end")
            content = content + "\n" + v1_section

    # Replace or insert v2 section
    if SECTION_ANCHOR_V2 in content:
        pattern = r"#### v2 工具调用频次演变.*?(?=\n#### |\n### |\Z)"
        new_content = re.sub(pattern, v2_section.strip(), content, flags=re.DOTALL)
        if new_content == content:
            print("[WARN] v2 section replacement had no effect")
        else:
            content = new_content
            print("[OK] Replaced v2 tool frequency section")
    else:
        content = content + "\n" + v2_section
        print("[OK] Appended v2 tool frequency section")

    with open(PROPOSAL, "w") as f:
        f.write(content)
    print(f"[OK] Written to {PROPOSAL}")

    # Update CHANGELOG
    from datetime import date
    today = date.today().strftime("%Y-%m-%d")
    entry = f"\n## {today}\n\n- [EDIT] `tusou/proposal.md` — 补充 v1/v2 工具调用频次演变表格\n"
    with open(CHANGELOG, "a") as f:
        f.write(entry)
    print(f"[OK] CHANGELOG updated")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Path to validation_output_test2 dir for a single experiment")
    parser.add_argument("--all", action="store_true", help="Analyze v1 + v2 and update proposal")
    parser.add_argument("--update_proposal", action="store_true", help="Update proposal.md after analysis")
    parser.add_argument("--version", choices=["v1", "v2"], help="Experiment version when using --data_dir")
    args = parser.parse_args()

    if args.all:
        print("=" * 60)
        print("Analyzing v1 (searchreward)...")
        v1_dir = Path(BASE) / EXPERIMENT_DIRS["v1"] / "validation_output_test2"
        v1_results = analyze_dir(v1_dir)
        print(format_table(v1_results, "v1 (SearchReward)"))

        print("\n" + "=" * 60)
        print("Analyzing v2 (basereward-v2)...")
        v2_dir = Path(BASE) / EXPERIMENT_DIRS["v2"] / "validation_output_test2"
        v2_results = analyze_dir(v2_dir)
        print(format_table(v2_results, "v2 (BaseReward)"))

        print("\n" + "=" * 60)
        print("Updating proposal.md...")
        update_proposal(v1_results, v2_results)

    elif args.data_dir:
        results = analyze_dir(args.data_dir)
        version = args.version or "unknown"
        print(format_table(results, f"{version} ({args.data_dir})"))
        if args.update_proposal:
            if version == "v1":
                v2_dir = Path(BASE) / EXPERIMENT_DIRS["v2"] / "validation_output_test2"
                v2_results = analyze_dir(v2_dir)
                update_proposal(results, v2_results)
            elif version == "v2":
                v1_dir = Path(BASE) / EXPERIMENT_DIRS["v1"] / "validation_output_test2"
                v1_results = analyze_dir(v1_dir)
                update_proposal(v1_results, results)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
