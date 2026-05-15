"""
eval_sft_granularity.py — Evaluate SFT label-format granularity ablation on Im2GPS3K.

Evaluates models trained with l0/l2/l3/l4 label granularity in single-turn mode
(no tool use). Each format uses a matching answer-format instruction at inference time.

Label formats:
  l0 — Latitude, Longitude only
  l2 — Country, City, Latitude, Longitude          (default SpotSFT format)
  l3 — Country, Region, City, Latitude, Longitude
  l4 — Country, Region, City, Neighbourhood, Latitude, Longitude

Usage:
  # single model via SGLang
  python3 eval_sft_granularity.py --label_format l2 --tag sft_4b_l2_step600

  # or use the shell script (runs all 4 in parallel, each on its own GPU):
  bash run_granularity_eval.sh
"""

import argparse
import asyncio
import base64
import io
import json
import math
import os
import re
from pathlib import Path

import aiohttp
import pandas as pd
from PIL import Image
from tqdm.asyncio import tqdm as atqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
BENCHMARK_DIR = "/mnt/sh/mmvision/home/jonahli/data_agent/benchmark"
CSV_PATH      = f"{BENCHMARK_DIR}/im2gps3k_CLEAN.csv"
IMG_DIR       = f"{BENCHMARK_DIR}/im2gps3ktest"
OUTPUT_DIR    = "/mnt/sh/mmvision/home/jonahli/save/agent/eval/im2gps3k"

# ── Prompt constants ────────────────────────────────────────────────────────────
# SFT models are trained WITHOUT a system prompt.
# The geolocation instruction is embedded directly in the user message,
# matching the exact training-time format from MP16-Pro/sft data.
SYSTEM_NOTOOL = ""  # no system prompt for SFT models

_SFT_PREFIX = (
    "You are a geolocation expert. Given an image, analyze the visual cues such as "
    "architecture, vegetation, road signs, landscape, and cultural elements to determine the location. "
)

# Each prompt matches the training-time answer format for its label granularity level.
LABEL_FORMAT_PROMPTS = {
    "l0": (
        _SFT_PREFIX
        + "The final answer MUST BE enclosed in <answer></answer> tags with the format: "
        "Latitude, Longitude. e.g. <answer>40.9606, 9.5873</answer>\n\n"
        "Look at the visual clues in this photo -- architecture, signs, vegetation, terrain. "
        "Where was it taken? Answer: Latitude, Longitude."
    ),
    "l2": (
        _SFT_PREFIX
        + "The final answer MUST BE enclosed in <answer></answer> tags with the format: "
        "Country, City, Latitude, Longitude. e.g. <answer>Italy, Golfo Arnaci, 40.9606, 9.5873</answer>\n\n"
        "Look at the visual clues in this photo -- architecture, signs, vegetation, terrain. "
        "Where was it taken? Answer: Country, City, Latitude, Longitude."
    ),
    # l2_no_prefix: same answer format as l2, but WITHOUT the "You are a geolocation expert..."
    # system prefix. Required for SGLang inference with --chat-template when the full prefix
    # triggers "!!!!" (token-sequence mismatch vs. training-time processor-expanded <image>).
    "l2_no_prefix": (
        "The final answer MUST BE enclosed in <answer></answer> tags with the format: "
        "Country, City, Latitude, Longitude. e.g. <answer>Italy, Golfo Arnaci, 40.9606, 9.5873</answer>\n\n"
        "Look at the visual clues in this photo -- architecture, signs, vegetation, terrain. "
        "Where was it taken? Answer: Country, City, Latitude, Longitude."
    ),
    "l3": (
        _SFT_PREFIX
        + "The final answer MUST BE enclosed in <answer></answer> tags with the format: "
        "Country, Region, City, Latitude, Longitude. e.g. <answer>Italy, Sardinia, Golfo Arnaci, 40.9606, 9.5873</answer>\n\n"
        "Look at the visual clues in this photo -- architecture, signs, vegetation, terrain. "
        "Where was it taken? Answer: Country, Region, City, Latitude, Longitude."
    ),
    "l4": (
        _SFT_PREFIX
        + "The final answer MUST BE enclosed in <answer></answer> tags with the format: "
        "Country, Region, City, Neighbourhood, Latitude, Longitude. "
        "e.g. <answer>Italy, Sardinia, Golfo Arnaci, Costa Smeralda, 40.9606, 9.5873</answer>\n\n"
        "Look at the visual clues in this photo -- architecture, signs, vegetation, terrain. "
        "Where was it taken? Answer: Country, Region, City, Neighbourhood, Latitude, Longitude."
    ),
}

# ── Image helpers ──────────────────────────────────────────────────────────────
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

# ── Geo helpers ────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(min(a, 1.0)))

ANSWER_RE      = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
BOXED_RE       = re.compile(r'\\boxed\{([^}]+)\}')   # legacy fallback
BOXED_TRUNC_RE = re.compile(r'\\boxed\{([^}]{3,}?)$')  # legacy fallback
DEGREE_RE      = re.compile(
    r'([-+]?\d{1,3}(?:\.\d+)?)\s*°?\s*([NS])[\s,;/]+?([-+]?\d{1,3}(?:\.\d+)?)\s*°?\s*([EW])',
    re.IGNORECASE,
)
FLOAT_PAIR_RE  = re.compile(r'([-+]?\d{1,3}(?:\.\d+)?)\s*,\s*([-+]?\d{1,3}(?:\.\d+)?)')

def parse_pred(text: str):
    # 1. <answer>...</answer>  — new format, lat/lon are the LAST two comma-separated fields
    answer_m = ANSWER_RE.search(text)
    if answer_m:
        parts = [p.strip() for p in answer_m.group(1).split(",")]
        try:
            return float(parts[-2]), float(parts[-1])
        except Exception:
            pass

    # 2. \boxed{...} — legacy fallback
    matches = BOXED_RE.findall(text)
    if not matches:
        m = BOXED_TRUNC_RE.search(text)
        if m:
            matches = [m.group(1)]
    if matches:
        parts = matches[-1].split(",")
        try:
            return float(parts[0].strip()), float(parts[1].strip())
        except Exception:
            try:
                return float(parts[-2].strip()), float(parts[-1].strip())
            except Exception:
                pass

    # 3. Degree-symbol format
    degree_matches = DEGREE_RE.findall(text)
    if degree_matches:
        lat_s, lat_dir, lon_s, lon_dir = degree_matches[-1]
        try:
            lat = float(lat_s) * (-1 if lat_dir.upper() == 'S' else 1)
            lon = float(lon_s) * (-1 if lon_dir.upper() == 'W' else 1)
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                return lat, lon
        except Exception:
            pass

    # 4. Plain float pair
    all_pairs = FLOAT_PAIR_RE.findall(text)
    for lat_s, lon_s in reversed(all_pairs):
        try:
            lat, lon = float(lat_s), float(lon_s)
            if abs(lat) <= 1.0 and abs(lon) <= 1.0:
                continue
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                return lat, lon
        except Exception:
            continue
    return None

# ── SGLang API ─────────────────────────────────────────────────────────────────
async def chat(session, url, messages, max_tokens=4096, temperature=0.0,
               no_thinking=False):
    payload = {
        "model":       "default",
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": temperature,
    }
    if no_thinking:
        payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    for attempt in range(3):
        try:
            async with session.post(
                f"{url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=600),
            ) as resp:
                data    = await resp.json()
                content = data["choices"][0]["message"].get("content", "") or ""
                return content
        except Exception:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
    return ""

# ── Single-turn eval ──────────────────────────────────────────────────────────
async def eval_single_turn(row, img_path, session, url, sem, args):
    async with sem:
        orig_b64 = load_encode(img_path)
        messages = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{orig_b64}"}},
            {"type": "text",      "text": args.user_prompt},
        ]})
        response = await chat(session, url, messages,
                              max_tokens=args.max_tokens,
                              temperature=args.temperature,
                              no_thinking=args.no_thinking)
        pred = parse_pred(response)
        km   = haversine(pred[0], pred[1], row.latitude, row.longitude) if pred else None
        return {
            "id":       row.id,
            "gt_lat":   row.latitude,  "gt_lon":  row.longitude,
            "pred_lat": pred[0] if pred else None,
            "pred_lon": pred[1] if pred else None,
            "km":       km,
            "masked":   1.0 if pred is None else 0.0,
            "response": response,
        }

# ── Main ───────────────────────────────────────────────────────────────────────
async def run(args):
    csv_file = args.csv_path or CSV_PATH
    df = pd.read_csv(csv_file)
    if args.max_samples > 0:
        if args.subset_seed is not None:
            df = df.sample(n=min(args.max_samples, len(df)), random_state=args.subset_seed).reset_index(drop=True)
        else:
            df = df.head(args.max_samples)

    imgs = {f.split("_")[0]: f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")}

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if os.path.exists(args.output_jsonl):
        with open(args.output_jsonl) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    # Only count as done if response is non-empty;
                    # empty responses indicate a mid-run GPU disconnect.
                    if d.get("response", ""):
                        done_ids.add(d["id"])
                except Exception:
                    pass
    print(f"Total: {len(df)}, already done: {len(done_ids)}, to process: {len(df) - len(done_ids)}")

    print(f"Waiting for SGLang at {args.sglang_url}...")
    async with aiohttp.ClientSession() as s:
        for _ in range(60):
            try:
                async with s.get(f"{args.sglang_url}/health",
                                 timeout=aiohttp.ClientTimeout(total=5)) as r:
                    if r.status == 200:
                        print("SGLang ready.")
                        break
            except Exception:
                pass
            await asyncio.sleep(2)

    sem = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for row in df.itertuples():
            if row.id in done_ids:
                continue
            img_path = os.path.join(IMG_DIR, imgs.get(str(row.id), ""))
            if not os.path.exists(img_path):
                continue
            tasks.append(eval_single_turn(row, img_path, session, args.sglang_url, sem, args))

        with open(args.output_jsonl, "a", buffering=1) as f_out:
            async for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc=args.tag):
                result = await coro
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

    all_results = []
    with open(args.output_jsonl) as f:
        for line in f:
            try:
                all_results.append(json.loads(line))
            except Exception:
                pass

    km_list   = [r["km"] for r in all_results if r.get("km") is not None]
    n         = len(km_list)
    n_total   = len(all_results)
    n_no_pred = sum(1 for r in all_results if r.get("km") is None)

    # Standard Im2GPS accuracy: denominator = n_total (unparsed = wrong)
    def acc(thresh):
        return round(sum(1 for k in km_list if k <= thresh) / n_total, 4) if n_total else 0

    summary = {
        "tag":        args.tag,
        "label_format": args.label_format,
        "model":      args.sglang_url,
        "n_total":    n_total,
        "n_parsed":   n,
        "n_no_pred":  n_no_pred,
        "parse_rate": round(n / n_total, 4) if n_total else 0,
        "acc_1km":    acc(1),
        "acc_25km":   acc(25),
        "acc_200km":  acc(200),
        "acc_750km":  acc(750),
        "acc_2500km": acc(2500),
    }

    summary_path = args.output_jsonl.replace(".jsonl", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 65)
    print(f"  Im2GPS3K SFT Granularity Eval — {args.tag}  [{args.label_format}]")
    print("=" * 65)
    print(f"  Samples:    {n}/{n_total} parsed  ({n_no_pred} no prediction)  parse_rate={summary['parse_rate']:.3f}")
    print(f"  Acc@1km:    {summary['acc_1km']:.3f}  ({int(summary['acc_1km'] * n_total)}/{n_total})")
    print(f"  Acc@25km:   {summary['acc_25km']:.3f}  ({int(summary['acc_25km'] * n_total)}/{n_total})")
    print(f"  Acc@200km:  {summary['acc_200km']:.3f}  ({int(summary['acc_200km'] * n_total)}/{n_total})")
    print(f"  Acc@750km:  {summary['acc_750km']:.3f}  ({int(summary['acc_750km'] * n_total)}/{n_total})")
    print(f"  Acc@2500km: {summary['acc_2500km']:.3f}  ({int(summary['acc_2500km'] * n_total)}/{n_total})")
    print("=" * 65)
    print(f"  Results → {args.output_jsonl}")
    print(f"  Summary → {summary_path}")
    return summary


def _load_prompt_arg(s: str) -> str:
    if s and s.startswith("@"):
        return Path(s[1:]).read_text().strip()
    return s


def main():
    p = argparse.ArgumentParser(
        description="Evaluate SFT label-format granularity ablation on Im2GPS3K (single-turn, no tools)."
    )
    p.add_argument("--label_format", default="l2", choices=["l0", "l2", "l2_no_prefix", "l3", "l4"],
                   help="SFT granularity label format: "
                        "l0=coords only, l2=country+city+coords (default), "
                        "l2_no_prefix=l2 without system prefix (for SGLang+chat_template), "
                        "l3=country+region+city+coords, l4=country+region+city+neighbourhood+coords")
    p.add_argument("--system_prompt", default=None, type=_load_prompt_arg,
                   help="Override system prompt (string or @/path/to/file). Default: SYSTEM_NOTOOL")
    p.add_argument("--user_prompt",   default=None, type=_load_prompt_arg,
                   help="Override user prompt (string or @/path/to/file). Default: LABEL_FORMAT_PROMPTS[label_format]")
    p.add_argument("--sglang_url",    default="http://127.0.0.1:30000")
    p.add_argument("--tag",           default=None,
                   help="Run tag for output filenames. Defaults to sft_{label_format}")
    p.add_argument("--output_jsonl",  default=None)
    p.add_argument("--csv_path",      default=None)
    p.add_argument("--max_samples",   type=int, default=-1)
    p.add_argument("--subset_seed",   type=int, default=None)
    p.add_argument("--concurrency",   type=int, default=32)
    p.add_argument("--max_tokens",    type=int, default=8192)
    p.add_argument("--temperature",   type=float, default=0.0)
    p.add_argument("--no_thinking",   action="store_true",
                   help="Disable thinking for reasoning models (passes enable_thinking=false to SGLang)")
    args = p.parse_args()

    if args.tag is None:
        args.tag = f"sft_{args.label_format}"
    if args.system_prompt is None:
        args.system_prompt = SYSTEM_NOTOOL
    if args.user_prompt is None:
        args.user_prompt = LABEL_FORMAT_PROMPTS[args.label_format]
    if args.output_jsonl is None:
        args.output_jsonl = f"{OUTPUT_DIR}/{args.tag}.jsonl"

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
