"""
eval_landmark.py — Evaluate landmark recognition on the internal landmark benchmark.

Dataset: /mnt/sh/mmvision/home/jonahli/data_agent/benchmark/landmark_eval.parquet
  Columns: idx, image_path, gt_description, domestic

Eval flow:
  1. Model sees the image, predicts the landmark name / location as free-form text.
  2. An LLM judge (qwen122b via LLMClient by default) compares the prediction
     to the ground-truth 详细位置描述 and returns a binary score (1 = correct, 0 = wrong).
  3. Report overall accuracy and domestic / foreign breakdown.

Usage:
  # no-tool mode (default)
  python3 eval_landmark.py --sglang_url http://127.0.0.1:31054 --tag 4b_notool

  # zoom-only mode
  python3 eval_landmark.py --sglang_url http://127.0.0.1:31054 --tag 4b_zoom --mode zoom

  # zoom+search mode
  python3 eval_landmark.py --sglang_url http://127.0.0.1:31054 --tag 4b_zoom_search --mode zoom_search

  # judge uses qwen122b by default; set to '' to use --judge_url SGLang endpoint
  python3 eval_landmark.py --judge_model '' --judge_url http://127.0.0.1:31054 --tag test
"""

import argparse
import asyncio
import base64
import io
import json
import os
import re
import sys
from pathlib import Path

import aiohttp
import pandas as pd
from PIL import Image
from tqdm.asyncio import tqdm as atqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_agent_tool import parse_tool_calls_flexible, ToolCallManager

# llm_client.py (qwen122b / 235b) — lazy import inside run() when --judge_model is set
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "../data_pipeline/server"))

# ── Paths ──────────────────────────────────────────────────────────────────────
BENCHMARK_DIR  = "/mnt/sh/mmvision/home/jonahli/data_agent/benchmark"
PARQUET_PATH   = f"{BENCHMARK_DIR}/landmark_eval.parquet"
OUTPUT_DIR     = "/mnt/sh/mmvision/home/jonahli/save/agent/eval/landmark"

# ── Tool schemas ────────────────────────────────────────────────────────────────
TOOL_SCHEMA_ZOOM = {
    "type": "function",
    "function": {
        "name": "image_zoom_in_tool",
        "description": (
            "Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label. "
            "Use this to examine details more closely before making a landmark prediction.\n"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "bbox_2d": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "The bounding box of the region to zoom in, as [x1, y1, x2, y2], "
                        "where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner. "
                        "Values are normalized coordinates in the range [0, 1000] relative to the current image.\n"
                    )
                },
                "label": {
                    "type": "string",
                    "description": "The name or label of the object in the bounding box (optional)."
                }
            },
            "required": ["bbox_2d"]
        }
    }
}

TOOL_SCHEMA_SEARCH = {
    "type": "function",
    "function": {
        "name": "image_search_tool",
        "description": (
            "Search the internet using a cropped region of the image as a query. "
            "Use this after zooming in on a region of interest (e.g. signs, landmarks, storefronts) "
            "to find matching web pages that may reveal the landmark name and location.\n"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "bbox_2d": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "The bounding box of the region to search with, as [x1, y1, x2, y2], "
                        "where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner. "
                        "Values are normalized coordinates in the range [0, 1000] relative to the original image. "
                        "Use the same bbox as image_zoom_in_tool to search the zoomed region.\n"
                    )
                }
            },
            "required": ["bbox_2d"]
        }
    }
}

TOOL_SCHEMAS_ZOOM_ONLY   = [TOOL_SCHEMA_ZOOM]
TOOL_SCHEMAS_ALL         = [TOOL_SCHEMA_ZOOM, TOOL_SCHEMA_SEARCH]

# ── Prompts ─────────────────────────────────────────────────────────────────────
SYSTEM_NOTOOL = (
    "You are a landmark recognition expert. "
    "Given an image, identify the landmark or location shown as precisely as possible. "
    "Describe the name of the landmark, its city, and country if known. "
    "Think step by step before answering.\n\n"
    "Provide your final answer inside \\boxed{}, e.g. \\boxed{埃菲尔铁塔, 法国巴黎}"
)

SYSTEM_TOOL = (
    "You are a landmark recognition expert with access to image tools.\n"
    "Given an image, identify the landmark or location shown as precisely as possible.\n\n"
    "You may use the following tools to assist:\n"
    "  - image_zoom_in_tool: zoom in on a region to read signs, plaques, or architectural details\n"
    "  - image_search_tool: reverse-image-search a region to find matching web pages\n\n"
    "Strategy:\n"
    "  1. Examine the image carefully.\n"
    "  2. If you see text, signs, or distinctive features, zoom in to read them.\n"
    "  3. Use image_search_tool on distinctive regions to find location clues from the web.\n"
    "  4. Combine all evidence to determine the landmark name and location.\n\n"
    "Provide your final answer inside \\boxed{}, e.g. \\boxed{埃菲尔铁塔, 法国巴黎}"
)

SYSTEM_TOOL_ZOOM_ONLY = (
    "You are a landmark recognition expert with access to an image zoom tool.\n"
    "Given an image, identify the landmark or location shown as precisely as possible.\n\n"
    "You may use image_zoom_in_tool to zoom in on regions to read signs, plaques, or architectural details.\n\n"
    "Provide your final answer inside \\boxed{}, e.g. \\boxed{埃菲尔铁塔, 法国巴黎}"
)

USER_PRED = (
    "Please identify the landmark or location shown in this image. "
    "What is the name of this place, and where is it located?\n\n"
    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    r"The reasoning process MUST BE enclosed within <think> </think> tags. "
    r"The final answer MUST BE put in \boxed{} with the landmark name and location, "
    r"e.g. \boxed{埃菲尔铁塔, 法国巴黎}"
)

SYSTEM_JUDGE = (
    "You are a strict judge evaluating whether a model's landmark prediction is correct.\n"
    "You will be given:\n"
    "  - Ground truth: the detailed location description of the image\n"
    "  - Prediction: the model's answer\n\n"
    "Scoring rules:\n"
    "  - Score 1 if the prediction correctly identifies the landmark/location "
    "(same building/place/street, even if phrased differently or partially correct)\n"
    "  - Score 0 if the prediction is wrong, vague, or identifies the wrong location\n\n"
    "Respond with ONLY a JSON object: {\"score\": 1} or {\"score\": 0}\n"
    "No explanation, no other text."
)


def judge_user_prompt(gt: str, pred: str) -> str:
    return (
        f"Ground truth: {gt}\n\n"
        f"Prediction: {pred}\n\n"
        "Score (1=correct, 0=wrong):"
    )


# ── Image helpers ───────────────────────────────────────────────────────────────
def encode_image(img: Image.Image, max_pixels: int = 2048 * 1024) -> str:
    w, h = img.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def load_encode(img_path: str) -> str:
    return encode_image(Image.open(img_path).convert("RGB"))


# ── SGLang API ──────────────────────────────────────────────────────────────────
async def chat(session, url, messages, tools=None, max_tokens=1024, temperature=0.0,
               no_thinking=False):
    payload = {
        "model":       "default",
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools
    if no_thinking:
        payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    for attempt in range(3):
        try:
            async with session.post(
                f"{url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                data    = await resp.json()
                msg     = data["choices"][0]["message"]
                content = msg.get("content", "") or ""
                return content
        except Exception:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
    return ""


# ── Parse prediction from boxed ─────────────────────────────────────────────────
BOXED_RE = re.compile(r'\\boxed\{([^}]+)\}')

def parse_pred(text: str) -> str:
    """Extract content inside \\boxed{} as the prediction string."""
    matches = BOXED_RE.findall(text)
    if matches:
        return matches[-1].strip()
    # Fallback: return the last non-empty line of the response
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


# ── LLMClient async wrapper ─────────────────────────────────────────────────────
async def _chat_llmclient(llm_client, messages: list, max_tokens: int):
    """Async wrapper around synchronous LLMClient.chat_messages()."""
    loop = asyncio.get_event_loop()
    def _call():
        return llm_client.chat_messages(messages, max_tokens=max_tokens)
    result = await loop.run_in_executor(None, _call)
    return result or ""


# ── Judge ───────────────────────────────────────────────────────────────────────
async def judge(session, url, gt: str, pred: str, max_tokens: int = 64,
                no_thinking: bool = True, llm_client=None) -> int:
    """Call the LLM judge and return 0 or 1."""
    messages = [
        {"role": "system", "content": SYSTEM_JUDGE},
        {"role": "user",   "content": judge_user_prompt(gt, pred)},
    ]
    if llm_client is not None:
        resp = await _chat_llmclient(llm_client, messages, max_tokens=max_tokens)
    else:
        resp = await chat(session, url, messages, max_tokens=max_tokens,
                          temperature=0.0, no_thinking=no_thinking)
    # Parse {"score": 1} or {"score": 0}
    try:
        j = json.loads(resp.strip())
        return int(j.get("score", 0))
    except Exception:
        # Fallback: look for digit 1 or 0
        m = re.search(r'"score"\s*:\s*([01])', resp)
        if m:
            return int(m.group(1))
        if "1" in resp:
            return 1
    return 0


# ── No-tool single-turn eval ────────────────────────────────────────────────────
async def eval_one_notool(row, session, pred_url, judge_url, sem, args, llm_client=None):
    async with sem:
        img_path = row.image_path
        gt       = row.gt_description or ""
        idx      = row.idx
        domestic = row.domestic

        try:
            b64 = load_encode(img_path)
        except Exception as e:
            return {
                "idx": idx, "domestic": domestic, "gt": gt,
                "pred": "", "score": 0, "error": str(e),
                "n_tool_calls": 0,
            }

        messages = [
            {"role": "system", "content": SYSTEM_NOTOOL},
            {"role": "user",   "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text",      "text": USER_PRED},
            ]},
        ]
        response = await chat(session, pred_url, messages,
                              max_tokens=args.max_tokens,
                              temperature=0.0,
                              no_thinking=args.no_thinking)
        pred_text = parse_pred(response)

        score = await judge(session, judge_url, gt, pred_text,
                            no_thinking=args.no_thinking,
                            llm_client=llm_client)

        return {
            "idx":          idx,
            "domestic":     domestic,
            "gt":           gt,
            "pred":         pred_text,
            "response":     response,
            "score":        score,
            "n_tool_calls": 0,
        }


# ── Multi-turn agent loop ────────────────────────────────────────────────────────
async def eval_agent_loop(row, session, pred_url, judge_url, sem, args,
                          llm_client=None, search_sem=None):
    """Multi-turn agent loop for landmark eval (zoom / zoom+search)."""
    use_search = (args.mode == "zoom_search")
    api_tools  = TOOL_SCHEMAS_ALL if use_search else TOOL_SCHEMAS_ZOOM_ONLY
    system_prompt = SYSTEM_TOOL if use_search else SYSTEM_TOOL_ZOOM_ONLY

    async with sem:
        img_path = row.image_path
        gt       = row.gt_description or ""
        idx      = row.idx
        domestic = row.domestic

        try:
            orig_b64 = load_encode(img_path)
        except Exception as e:
            return {
                "idx": idx, "domestic": domestic, "gt": gt,
                "pred": "", "score": 0, "error": str(e),
                "n_tool_calls": 0,
            }

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{orig_b64}"}},
                {"type": "text",      "text": USER_PRED},
            ]},
        ]

        manager       = ToolCallManager(img_path)
        all_responses = []
        n_tool_calls  = 0

        for turn in range(args.max_turns + 1):
            response = await chat(session, pred_url, messages,
                                  tools=api_tools,
                                  max_tokens=args.max_tokens,
                                  temperature=0.0,
                                  no_thinking=args.no_thinking)
            all_responses.append(response)

            tool_calls = parse_tool_calls_flexible(response)
            tool_call  = tool_calls[0] if tool_calls else None

            if tool_call is None or turn == args.max_turns:
                # If max_turns hit but no boxed answer yet, force a final no-tool turn
                if turn == args.max_turns and tool_call is not None and not parse_pred(response):
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content":
                        "Based on your analysis so far, provide your final answer now. "
                        "You MUST put the final answer in \\boxed{} with the landmark name and location, "
                        r"e.g. \boxed{埃菲尔铁塔, 法国巴黎}"
                    })
                    final_resp = await chat(session, pred_url, messages,
                                            tools=None, max_tokens=512,
                                            temperature=0.0,
                                            no_thinking=args.no_thinking)
                    all_responses.append(final_resp)
                break

            name  = tool_call.get("name", "")
            args_ = tool_call.get("arguments", {})
            if isinstance(args_, str):
                try:
                    args_ = json.loads(args_)
                except Exception:
                    args_ = {}

            try:
                loop = asyncio.get_event_loop()
                if name == "image_search_tool" and search_sem is not None:
                    async with search_sem:
                        result = await loop.run_in_executor(None, manager.execute, name, args_)
                else:
                    result = await loop.run_in_executor(None, manager.execute, name, args_)

                img_b64     = result.get("crop_b64")
                search_text = result.get("text")

                if name == "image_zoom_in_tool":
                    bbox  = args_.get("bbox_2d", [])
                    label = args_.get("label", "")
                    tool_text = (f"Zoomed in on the image to the region {bbox} with label {label}."
                                 if label else
                                 f"Zoomed in on the image to the region {bbox}.")
                elif name == "image_search_tool":
                    tool_text = search_text or "No search results available."
                    img_b64   = None
                else:
                    tool_text = str(result)
                    img_b64   = None

                n_tool_calls += 1

            except Exception as e:
                img_b64   = None
                tool_text = f"Tool call error: {e}"

            # Build assistant + tool_response messages
            # Use role="tool" to match VERL training format (jinja wraps as <tool_response>)
            assistant_content = response
            if "<tool_call>" in assistant_content and "</tool_call>" not in assistant_content:
                tc_start   = assistant_content.find("<tool_call>")
                json_start = assistant_content.find("{", tc_start)
                if json_start != -1:
                    depth, json_end = 0, json_start
                    for ci, ch in enumerate(assistant_content[json_start:], json_start):
                        if ch == "{":
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0:
                                json_end = ci
                                break
                    assistant_content = (assistant_content[:tc_start] +
                                         "<tool_call>\n" +
                                         assistant_content[json_start:json_end+1] +
                                         "\n</tool_call>")
                else:
                    assistant_content = assistant_content.rstrip() + "\n</tool_call>"

            messages.append({"role": "assistant", "content": assistant_content})

            if img_b64:
                # Image FIRST, text second — matches VERL training format
                tool_content = [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text",      "text": tool_text},
                ]
            else:
                tool_content = tool_text
            messages.append({"role": "tool", "content": tool_content})

        # Extract final prediction from last response
        full_response = "\n".join(all_responses)
        pred_text     = parse_pred(full_response)

        score = await judge(session, judge_url, gt, pred_text,
                            no_thinking=args.no_thinking,
                            llm_client=llm_client)

        return {
            "idx":          idx,
            "domestic":     domestic,
            "gt":           gt,
            "pred":         pred_text,
            "response":     full_response,
            "score":        score,
            "n_tool_calls": n_tool_calls,
        }


# ── Main ────────────────────────────────────────────────────────────────────────
async def run(args):
    df = pd.read_parquet(PARQUET_PATH)
    if args.max_samples > 0:
        df = df.head(args.max_samples)

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if os.path.exists(args.output_jsonl):
        with open(args.output_jsonl) as f:
            for line in f:
                try:
                    done_ids.add(int(json.loads(line)["idx"]))
                except Exception:
                    pass
    print(f"Total: {len(df)}, already done: {len(done_ids)}, "
          f"to process: {len(df) - len(done_ids)}")

    pred_url  = args.sglang_url
    judge_url = args.judge_url or args.sglang_url
    print(f"Mode:              {args.mode}")
    print(f"Prediction server: {pred_url}")

    llm_judge = None
    if args.judge_model:
        from llm_client import LLMClient
        llm_judge = LLMClient(model=args.judge_model)
        print(f"Judge: LLMClient(model={args.judge_model!r})")
    else:
        print(f"Judge server:      {judge_url}")

    async with aiohttp.ClientSession() as s:
        for _ in range(60):
            try:
                async with s.get(f"{pred_url}/health",
                                 timeout=aiohttp.ClientTimeout(total=5)) as r:
                    if r.status == 200:
                        print("SGLang ready.")
                        break
            except Exception:
                pass
            await asyncio.sleep(2)

    sem        = asyncio.Semaphore(args.concurrency)
    search_sem = asyncio.Semaphore(args.search_concurrency) if args.mode == "zoom_search" else None

    async with aiohttp.ClientSession() as session:
        tasks = []
        for row in df.itertuples():
            if row.idx in done_ids:
                continue
            if args.mode == "notool":
                tasks.append(eval_one_notool(row, session, pred_url, judge_url,
                                             sem, args, llm_client=llm_judge))
            else:
                tasks.append(eval_agent_loop(row, session, pred_url, judge_url,
                                             sem, args, llm_client=llm_judge,
                                             search_sem=search_sem))

        results = []
        with open(args.output_jsonl, "a") as out_f:
            for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks),
                              desc=f"eval_landmark[{args.mode}]"):
                res = await coro
                results.append(res)
                out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                out_f.flush()

    # ── Aggregate ──────────────────────────────────────────────────────────────
    all_results = []
    with open(args.output_jsonl) as f:
        for line in f:
            try:
                all_results.append(json.loads(line))
            except Exception:
                pass

    n_total   = len(all_results)
    n_correct = sum(r["score"] for r in all_results)
    acc       = n_correct / n_total if n_total else 0.0

    domestic_res = [r for r in all_results if r.get("domestic") == "国内"]
    foreign_res  = [r for r in all_results if r.get("domestic") == "国外"]
    acc_dom = sum(r["score"] for r in domestic_res) / len(domestic_res) if domestic_res else 0
    acc_for = sum(r["score"] for r in foreign_res)  / len(foreign_res)  if foreign_res  else 0

    avg_tools = (sum(r.get("n_tool_calls", 0) for r in all_results) / n_total
                 if n_total else 0)

    summary = {
        "tag":          args.tag,
        "mode":         args.mode,
        "model":        pred_url,
        "judge":        args.judge_model or judge_url,
        "n_total":      n_total,
        "n_correct":    n_correct,
        "accuracy":     round(acc,     4),
        "acc_domestic": round(acc_dom, 4),
        "acc_foreign":  round(acc_for, 4),
        "n_domestic":   len(domestic_res),
        "n_foreign":    len(foreign_res),
        "avg_tool_calls": round(avg_tools, 2),
    }
    print("\n── Results ────────────────────────────────────────────")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    summary_path = args.output_jsonl.replace(".jsonl", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sglang_url",  default="http://127.0.0.1:31050",
                        help="SGLang server URL for prediction")
    parser.add_argument("--judge_url",   default=None,
                        help="SGLang server URL for judge (defaults to --sglang_url)")
    parser.add_argument("--judge_model", default="qwen122b",
                        help="Use LLMClient for judging: 'qwen122b', '235b', etc. "
                             "Set to '' to use --judge_url SGLang endpoint instead.")
    parser.add_argument("--mode",        default="notool",
                        choices=["notool", "zoom", "zoom_search"],
                        help="Eval mode: notool / zoom / zoom_search")
    parser.add_argument("--tag",         default="run", help="Experiment tag")
    parser.add_argument("--output_jsonl", default=None,
                        help="Output JSONL path (auto-generated from tag if not set)")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples (0 = all)")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Max concurrent requests")
    parser.add_argument("--search_concurrency", type=int, default=4,
                        help="Max concurrent image_search_tool calls")
    parser.add_argument("--max_tokens",  type=int, default=1024,
                        help="Max tokens for prediction")
    parser.add_argument("--max_turns",   type=int, default=5,
                        help="Max agent turns (agent modes only)")
    parser.add_argument("--no_thinking", action="store_true",
                        help="Disable thinking (enable_thinking=False)")
    args = parser.parse_args()

    if args.output_jsonl is None:
        args.output_jsonl = f"{OUTPUT_DIR}/{args.tag}.jsonl"

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
