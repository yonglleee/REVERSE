"""
annotate_useful_results.py — 用 Kimi 补全 coldstart JSONL 里缺失的 useful_results 标注。

标注策略（方案B + fallback）：
  1. 从搜索结果后的 assistant <think> 块里解析引用了哪些 [N] 结果索引
  2. 如果 <think> 里有明确引用（如 [1]、[3] 等） → 直接用这些 index
  3. 如果 <think> 里没有引用 → fallback 到方案A：让 Kimi 看搜索结果+GT 判别
     - 如果 <think> 里明确说"no results"/"not helpful"等 → 直接返回 []，跳过 Kimi
  4. 将 {turn, tool, indices} 追加到 useful_results 字段

输出：在原 JSONL 基础上补全 useful_results，写到新文件。

用法：
  python3 annotate_useful_results.py --parts 00 01 --concurrency 32 --resume

  # 只标注 RL parquet 用到的图片：
  python3 annotate_useful_results.py --parts 00 01 \\
    --rl_parquet /mnt/sh/mmvision/home/jonahli/data_agent/rl/coldstart/train_coldstart_v4.parquet \\
    --concurrency 32 --resume

  # Dry run：
  python3 annotate_useful_results.py --dry_run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "data_pipeline" / "server"))

# ── Kimi fallback system prompt ────────────────────────────────────────────────

ANNOTATE_SYSTEM = (
    "You are a geolocation annotation expert.\n"
    "Given a list of web search results and the ground truth location, "
    "identify which results are USEFUL for geolocation.\n\n"
    "A result is USEFUL if it mentions the specific location, landmark, city, "
    "or geographic region shown in the image.\n"
    "A result is NOT USEFUL if it is unrelated or mentions a different place.\n\n"
    "Output ONLY a JSON array of 1-based indices of useful results.\n"
    "Examples: [1, 3] or [2] or []\n"
    "Do NOT include any explanation."
)

# Phrases in <think> that indicate model found no useful results
_NO_RESULT_PHRASES = [
    'no results', 'not helpful', 'not useful', 'no useful', 'no relevant',
    'no information', 'not related', 'unrelated', 'no match', 'no matching',
    'none of', 'none are', 'nothing useful', 'not geo', 'no geo',
    'no location', 'no specific location', 'cannot determine',
    '没有有用', '没有相关', '没有帮助', '无法', '不相关',
]

TOOL_CALL_RE = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL)
# Match [N] references in think block: [1], [2,3], [1, 3], result [3], result #3
BRACKET_REF_RE = re.compile(r'\[(\d+)\]')


# ── Parse search results from tool response ────────────────────────────────────

IMG_RESULT_RE = re.compile(
    r'\[(\d+)\]\s*(.*?)\n\s*Source:\s*(.*?)\n\s*Link:\s*(https?://\S+)',
    re.DOTALL
)
WEB_RESULT_RE = re.compile(
    r'\[(\d+)\]\s*(.*?)(?:\n\s*URL:\s*(\S+))?(?:\n\s*(.*?))?(?=\[\d+\]|$)',
    re.DOTALL
)


def _content_str(content) -> str:
    if isinstance(content, list):
        return ' '.join(
            x.get('text', '') if isinstance(x, dict) else str(x)
            for x in content
        )
    return content or ''


def _parse_img_results(text: str) -> list[dict]:
    results = []
    for m in IMG_RESULT_RE.finditer(text):
        results.append({
            'pos': int(m.group(1)),
            'title': m.group(2).strip(),
            'source': m.group(3).strip(),
            'link': m.group(4).strip(),
        })
    return results


def _parse_web_results(text: str) -> list[dict]:
    results = []
    for m in WEB_RESULT_RE.finditer(text):
        snippet = (m.group(4) or '').strip()[:200]
        results.append({
            'pos': int(m.group(1)),
            'title': m.group(2).strip().split('\n')[0],
            'url': (m.group(3) or '').strip(),
            'snippet': snippet,
        })
    return results


# ── Extract calls and their following <think> blocks ──────────────────────────

def _extract_calls(messages: list) -> list[dict]:
    """
    Walk messages, extract all image_search and text_search calls with:
      - their search results (from user tool response)
      - the <think> content of the NEXT assistant message (method B source)

    Returns list of:
      {tool, call_idx, turn, results, query, next_think}
      where turn = assistant turn index (0-based among all assistants)
            next_think = <think>...</think> content of the assistant after the tool response
    """
    calls = []
    img_call_idx = 0
    txt_call_idx = 0
    pending = None  # {tool, turn, query}
    # turn = absolute message index of the assistant that made the tool call
    # This matches the original JSONL useful_results format (e.g., turn=2 means msg[2])

    for i, msg in enumerate(messages):
        role = msg.get('role', '')
        c = _content_str(msg.get('content', ''))

        if role == 'assistant':
            # If there's a pending call waiting for its next-think, capture it now
            if pending and pending.get('_waiting_for_think'):
                think_m = THINK_RE.search(c)
                pending['next_think'] = think_m.group(1).strip() if think_m else ''
                calls.append({k: v for k, v in pending.items() if not k.startswith('_')})
                pending = None

            # Look for tool calls in this assistant message
            for tc_m in TOOL_CALL_RE.finditer(c):
                try:
                    tc = json.loads(tc_m.group(1))
                except Exception:
                    continue
                name = tc.get('name', '')
                if name == 'image_search_tool':
                    pending = {
                        'tool': 'image_search_tool',
                        'call_idx': img_call_idx,
                        'turn': i,  # absolute message index, matches original JSONL format
                        'query': '',
                        'results': [],
                        'next_think': '',
                        '_waiting_for_think': False,
                    }
                    img_call_idx += 1
                    break
                elif name == 'text_search_tool':
                    q = tc.get('arguments', {}).get('query', '')
                    if isinstance(q, list):
                        q = ' | '.join(q)
                    pending = {
                        'tool': 'text_search_tool',
                        'call_idx': txt_call_idx,
                        'turn': i,  # absolute message index, matches original JSONL format
                        'query': q,
                        'results': [],
                        'next_think': '',
                        '_waiting_for_think': False,
                    }
                    txt_call_idx += 1
                    break

        elif role == 'user' and pending is not None and not pending.get('_waiting_for_think'):
            if pending['tool'] == 'image_search_tool' and 'Image search results' in c:
                results = _parse_img_results(c)
                if not results:
                    qm = re.search(r'Image search results for region.*?:\s*\n?(.*)', c, re.DOTALL)
                pending['results'] = results
                pending['_waiting_for_think'] = True  # wait for next assistant <think>

            elif pending['tool'] == 'text_search_tool' and 'Web search results' in c:
                results = _parse_web_results(c)
                query = pending.get('query', '')
                if not query:
                    qm = re.search(r'Web search results for:\s*(.+?)(?:\n|$)', c)
                    if qm:
                        query = qm.group(1).strip()
                pending['results'] = results
                pending['query'] = query
                pending['_waiting_for_think'] = True

    # Handle pending call at end of conversation (no more assistant messages)
    if pending and pending.get('_waiting_for_think'):
        pending['next_think'] = ''
        calls.append({k: v for k, v in pending.items() if not k.startswith('_')})

    return calls


# ── Method B: extract indices from <think> block ──────────────────────────────

def _extract_from_think(think: str, n_results: int) -> Optional[list[int]]:
    """
    Try to extract referenced result indices from a <think> block.

    Returns:
      - list[int]: indices found (may be empty [] if think says no results)
      - None: think doesn't mention results → fallback to Kimi
    """
    if not think:
        return None

    think_lower = think.lower()

    # Check if model explicitly says results are not useful
    for phrase in _NO_RESULT_PHRASES:
        if phrase in think_lower:
            return []  # model said nothing useful → []

    # Extract [N] bracket references
    refs = set()
    for m in BRACKET_REF_RE.finditer(think):
        idx = int(m.group(1))
        if 1 <= idx <= n_results:
            refs.add(idx)

    if refs:
        return sorted(refs)

    # No bracket references found → cannot determine from think → fallback
    return None


# ── Method A fallback: Kimi GT-based annotation ───────────────────────────────

def _build_kimi_prompt(call: dict, gt_country: str, gt_city: str,
                       gt_lat: float, gt_lon: float) -> str:
    lines = [
        f"Ground truth: {gt_country}, {gt_city} ({gt_lat:.4f}, {gt_lon:.4f})",
        f"Tool: {call['tool']}",
    ]
    if call.get('query'):
        lines.append(f"Query: {call['query']}")
    lines.append("")
    lines.append("Search results:")
    for r in call['results']:
        pos = r['pos']
        title = r.get('title', '')
        source = r.get('source', r.get('url', ''))
        snippet = r.get('snippet', '')
        line = f"[{pos}] {title}"
        if source:
            line += f"  |  {source}"
        if snippet:
            line += f"\n    {snippet[:150]}"
        lines.append(line)
    return '\n'.join(lines)


def _call_kimi_fallback(client, call: dict, gt_country: str, gt_city: str,
                        gt_lat: float, gt_lon: float) -> list[int]:
    """Fallback: ask Kimi to judge based on GT location."""
    if not call['results']:
        return []

    prompt = _build_kimi_prompt(call, gt_country, gt_city, gt_lat, gt_lon)
    n = len(call['results'])

    # system prompt as first message (chat_messages doesn't accept system kwarg)
    messages = [
        {"role": "system", "content": ANNOTATE_SYSTEM},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(3):
        try:
            resp = client.chat_messages(messages, max_tokens=2048)
            text = (resp or '').strip()
            m = re.search(r'\[.*?\]', text, re.DOTALL)
            if m:
                raw = json.loads(m.group(0))
                return [int(i) for i in raw
                        if isinstance(i, (int, float)) and 1 <= int(i) <= n]
            return []
        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)
    return []


# ── Annotate one call (method B + fallback) ────────────────────────────────────

def _annotate_call(client, call: dict, gt_country: str, gt_city: str,
                   gt_lat: float, gt_lon: float) -> tuple[list[int], str]:
    """
    Returns (indices, method) where method is 'think'/'think_empty'/'kimi'/'empty_results'.
    """
    if not call['results']:
        return [], 'empty_results'

    n = len(call['results'])
    think = call.get('next_think', '')

    # Method B: try to extract from <think>
    think_result = _extract_from_think(think, n)

    if think_result is not None:
        # think gave us a definitive answer (could be [] if model said nothing useful)
        method = 'think_empty' if not think_result else 'think'
        return think_result, method

    # Fallback: Kimi GT-based
    indices = _call_kimi_fallback(client, call, gt_country, gt_city, gt_lat, gt_lon)
    return indices, 'kimi'


# ── Process one JSONL entry ───────────────────────────────────────────────────

def process_entry(entry: dict, client, tools: set) -> dict:
    """Annotate missing useful_results for one JSONL entry."""
    gt_lat = float(entry.get('gt_lat', 0))
    gt_lon = float(entry.get('gt_lon', 0))
    gt_country = str(entry.get('gt_country', ''))
    gt_city = str(entry.get('gt_city', ''))

    # Existing annotations keyed by (tool, turn) where turn = absolute message index
    existing = {}
    for u in entry.get('useful_results', []):
        try:
            tool = u.get('tool')
            turn = u.get('turn')
            if tool is None or turn is None:
                continue
            existing[(tool, int(turn))] = u
        except (ValueError, TypeError):
            continue

    messages = entry.get('messages', [])
    calls = _extract_calls(messages)

    new_annotations = []
    for call in calls:
        if call['tool'] not in tools:
            continue
        key = (call['tool'], call['turn'])
        if key in existing:
            continue  # already annotated

        indices, method = _annotate_call(
            client, call, gt_country, gt_city, gt_lat, gt_lon
        )
        new_annotations.append({
            'turn': call['turn'],
            'tool': call['tool'],
            'indices': indices,
            '_method': method,  # debug field, stripped before saving
        })

    if new_annotations:
        entry = dict(entry)
        combined = list(entry.get('useful_results', []))
        # Strip debug fields before saving
        for ann in new_annotations:
            combined.append({k: v for k, v in ann.items() if not k.startswith('_')})
        entry['useful_results'] = combined

    return entry


# ── File-level processing ──────────────────────────────────────────────────────

def process_file(input_path: str, output_path: str, client,
                 tools: set, concurrency: int, resume: bool,
                 rl_paths: Optional[set]) -> dict:
    stats = {
        'total': 0, 'annotated': 0, 'skipped': 0,
        'method_think': 0, 'method_think_empty': 0,
        'method_kimi': 0, 'method_empty_results': 0,
    }

    # Load all IDs already in output (for resume) — only skip if already fully annotated
    done_entries = {}  # id → entry (from output, with annotations)
    done_ids = set()
    if resume and os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    calls = _extract_calls(d.get('messages', []))
                    existing = {(u['tool'], int(u['turn'])) for u in d.get('useful_results', [])}
                    needs = [c for c in calls if c['tool'] in tools and (c['tool'], c['turn']) not in existing]
                    if not needs:
                        done_ids.add(str(d.get('id', '')))
                        done_entries[str(d.get('id', ''))] = d
                except Exception:
                    pass
        print(f"  Resume: {len(done_ids)} entries already fully annotated in output")

    # Load all entries
    entries = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
    stats['total'] = len(entries)

    # Classify entries
    to_write_as_is = []  # no annotation needed or already done
    to_annotate = []

    for entry in entries:
        eid = str(entry.get('id', ''))

        # Already fully annotated in output → use the output version (has annotations)
        if eid in done_ids:
            to_write_as_is.append(done_entries[eid])
            stats['skipped'] += 1
            continue

        # Filter by rl_paths if provided
        if rl_paths is not None:
            images = entry.get('images', [])
            img0 = images[0] if images else ''
            if isinstance(img0, dict):
                img0 = img0.get('image_url', '') or img0.get('path', '')
            if img0 not in rl_paths:
                to_write_as_is.append(entry)
                stats['skipped'] += 1
                continue

        # Check if needs annotation
        existing = {(u['tool'], int(u['turn'])) for u in entry.get('useful_results', [])}
        calls = _extract_calls(entry.get('messages', []))
        needs = [c for c in calls if c['tool'] in tools and (c['tool'], c['turn']) not in existing]

        if not needs:
            to_write_as_is.append(entry)
            stats['skipped'] += 1
        else:
            to_annotate.append(entry)

    print(f"  Need annotation: {len(to_annotate)} entries")
    print(f"  Write as-is: {len(to_write_as_is)} entries")

    # Always write mode: resume rewrites the full file with fixed entries
    # (append would create duplicate rows for re-annotated entries)
    with open(output_path, 'w') as out_f:
        # Write entries that need no annotation
        for entry in to_write_as_is:
            out_f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        # Annotate in parallel
        done = 0

        def _annotate_and_track(entry):
            result = process_entry(entry, client, tools)
            # Count methods used
            for ann in result.get('useful_results', []):
                # We can't easily track methods here since they're stripped
                pass
            return result

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(process_entry, entry, client, tools): entry
                for entry in to_annotate
            }
            for future in as_completed(futures):
                result = future.result()
                out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                stats['annotated'] += 1
                done += 1
                if done % 100 == 0:
                    out_f.flush()
                    print(f"  {done}/{len(to_annotate)} annotated...")

    return stats


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir',
                    default='/mnt/sh/mmvision/home/jonahli/save/agent/annotate/coldstart')
    ap.add_argument('--parts', nargs='+', default=['00', '01'])
    ap.add_argument('--output_dir',
                    default='/mnt/sh/mmvision/home/jonahli/save/agent/annotate/coldstart_relabeled')
    ap.add_argument('--tools', nargs='+', default=['image_search_tool', 'text_search_tool'],
                    help='Which tools to annotate useful_results for')
    ap.add_argument('--rl_parquet', default='',
                    help='Only annotate entries whose image_path is in this RL parquet. '
                         'Leave empty to annotate all entries.')
    ap.add_argument('--model', default='kimi_k2d6',
                    help='LLM model for Kimi fallback annotation (kimi_k2d6 or kimi_k2d5)')
    ap.add_argument('--concurrency', type=int, default=16)
    ap.add_argument('--resume', action='store_true',
                    help='Resume from existing output (skip already-processed entries)')
    ap.add_argument('--dry_run', action='store_true',
                    help='Count what needs annotation without calling API')
    args = ap.parse_args()

    tools = set(args.tools)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load RL parquet filter
    rl_paths = None
    if args.rl_parquet:
        import pandas as pd
        df = pd.read_parquet(args.rl_parquet)
        rl_paths = set()
        for ei in df['extra_info']:
            tw = ei.get('tools_kwargs', {})
            for tool_name in tools:
                img = tw.get(tool_name, {}).get('create_kwargs', {}).get('image', '')
                if img:
                    rl_paths.add(img)
        print(f"RL filter: {len(rl_paths)} unique image paths from {args.rl_parquet}")

    if args.dry_run:
        total_entries = total_need = total_kimi_calls = total_think_calls = 0
        for part in args.parts:
            path = os.path.join(args.input_dir, f'part{part}.jsonl')
            n_entries = n_need = n_kimi = n_think = 0
            with open(path) as f:
                for line in f:
                    try:
                        d = json.loads(line.strip())
                    except Exception:
                        continue
                    n_entries += 1

                    # RL filter
                    if rl_paths is not None:
                        images = d.get('images', [])
                        img0 = images[0] if images else ''
                        if isinstance(img0, dict):
                            img0 = img0.get('image_url', '') or img0.get('path', '')
                        if img0 not in rl_paths:
                            continue

                    existing = {(u['tool'], int(u['turn'])) for u in d.get('useful_results', [])}
                    calls = _extract_calls(d.get('messages', []))
                    needs = [c for c in calls if c['tool'] in tools
                             and (c['tool'], c['turn']) not in existing]
                    if needs:
                        n_need += 1
                        for call in needs:
                            think_result = _extract_from_think(
                                call.get('next_think', ''), len(call['results'])
                            )
                            if think_result is not None:
                                n_think += 1
                            else:
                                n_kimi += 1
            print(f"part{part}: {n_entries} entries, {n_need} need annotation "
                  f"(~{n_think} from think, ~{n_kimi} need Kimi API)")
            total_entries += n_entries
            total_need += n_need
            total_kimi_calls += n_kimi
            total_think_calls += n_think
        print(f"\nTotal: {total_entries} entries, {total_need} need annotation")
        print(f"  ~{total_think_calls} extractable from <think> (no API needed)")
        print(f"  ~{total_kimi_calls} need Kimi fallback API calls")
        return

    from llm_client import LLMClient
    client = LLMClient(model=args.model)

    for part in args.parts:
        input_path = os.path.join(args.input_dir, f'part{part}.jsonl')
        output_path = os.path.join(args.output_dir, f'part{part}.jsonl')
        print(f"\n[part{part}] {input_path} -> {output_path}")
        stats = process_file(input_path, output_path, client, tools,
                             args.concurrency, args.resume, rl_paths)
        print(f"[part{part}] done: {stats}")

    print("\nAll parts done. Next steps:")
    print("  1. Rebuild image_search_cache (build_image_search_cache.py)")
    print("  2. Rebuild SFT coldstart parquet (build_sft_coldstart.py)")
    print("  3. Re-run SFT coldstart training")


if __name__ == '__main__':
    main()
