"""
eval_im2gps3k_235b.py — Evaluate Qwen3-VL-235B on Im2GPS3K via py3meshkit.

Uses same prompts and metrics as eval_im2gps3k.py (--notool mode).
Runs with multiprocessing to parallelize 235B API calls.

Usage:
    python3 eval_im2gps3k_235b.py \
        --num_processes 10 \
        --tag qwen3vl235b_notool_v3
"""

import sys
import json
import os
import base64
import io
import math
import re
import time
import multiprocessing
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
import tqdm

sys.path.insert(0, "/mnt/sh/mmvision/home/jonahli/projects/SpatialVL/data_preprocess/server/qwen3vl235b")
from Qwen3_VL_235B import Qwen3VL235B

# ── Paths ──────────────────────────────────────────────────────────────────────
BENCHMARK_DIR = "/mnt/sh/mmvision/home/jonahli/data_agent/benchmark"
CSV_PATH      = f"{BENCHMARK_DIR}/im2gps3k_CLEAN.csv"
IMG_DIR       = f"{BENCHMARK_DIR}/im2gps3ktest"
OUTPUT_DIR    = "/mnt/sh/mmvision/home/jonahli/save/agent/eval/im2gps3k"

# ── Prompts (verbatim from eval_im2gps3k.py --notool) ─────────────────────────
SYSTEM_PROMPT = (
    "You are a geolocation expert. You are given an image and you need to identify its location. "
    "Reason step by step before making your prediction.\n\n"
    "Provide your final answer in the format: Latitude, Longitude, Country, City. "
    "e.g. 40.9606, 9.5873, Italy, Golfo Arnaci"
)

USER_PROMPT = (
    "Analyze the architectural styles, vegetation, street infrastructure, and cultural markers in this image. "
    "Based on these visual cues, determine the location."
    r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    r"The reasoning process MUST BE enclosed within <think> </think> tags. "
    r"The final answer MUST BE put in \boxed{} in the format: Latitude, Longitude, Country, City. "
    r"e.g. \boxed{40.9606, 9.5873, Italy, Golfo Arnaci}"
)

BOXED_RE = re.compile(r'\\boxed\{([^}]+)\}')

# ── Helpers ────────────────────────────────────────────────────────────────────

def encode_image(img_path: str, max_pixels: int = 2048 * 1024) -> str:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(min(a, 1.0)))


def parse_pred(text: str):
    matches = BOXED_RE.findall(text)
    if not matches:
        return None
    parts = matches[-1].split(",")
    try:
        return float(parts[0].strip()), float(parts[1].strip())
    except Exception:
        return None


def call_235b(img_b64: str, retry: int = 3):
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        {"type": "text", "text": USER_PROMPT},
        {"type": "text", "text": SYSTEM_PROMPT},
    ]
    for attempt in range(retry):
        try:
            model = Qwen3VL235B()
            return model(content)
        except Exception as e:
            print(f"[235B] attempt {attempt+1}/{retry}: {e}")
            if attempt < retry - 1:
                time.sleep(2 ** attempt)
    return None


# ── Worker ─────────────────────────────────────────────────────────────────────

def process_batch(batch, result_queue, worker_id):
    pid = os.getpid()
    print(f"[Worker {worker_id} / PID {pid}] batch_size={len(batch)}")
    ok = fail = 0
    for i, row in enumerate(batch):
        try:
            img_b64 = encode_image(row["img_path"])
            response = call_235b(img_b64)
            pred = parse_pred(response or "")
            km = haversine(pred[0], pred[1], row["gt_lat"], row["gt_lon"]) if pred else None
            result = {
                "id":       row["id"],
                "gt_lat":   row["gt_lat"],
                "gt_lon":   row["gt_lon"],
                "pred_lat": pred[0] if pred else None,
                "pred_lon": pred[1] if pred else None,
                "km":       km,
                "response": response or "",
            }
            result_queue.put({"type": "result", "data": result})
            ok += 1
        except Exception as e:
            print(f"[Worker {worker_id}] exception on id={row['id']}: {e}")
            result_queue.put({"type": "failure", "data": {"id": row["id"]}})
            fail += 1
        if (i + 1) % 20 == 0:
            print(f"[Worker {worker_id}] {i+1}/{len(batch)}, ok={ok}, fail={fail}")
    result_queue.put(None)
    print(f"[Worker {worker_id}] done. ok={ok}, fail={fail}")


# ── Writer ─────────────────────────────────────────────────────────────────────

def writer_process(result_queue, output_jsonl, fail_jsonl, total_tasks, num_workers, already_done):
    workers_done = 0
    n_ok = n_fail = 0
    print(f"[Writer PID {os.getpid()}] started, {num_workers} workers")

    with open(output_jsonl, "a", buffering=1) as f_out, \
         open(fail_jsonl, "a", buffering=1) as f_fail, \
         tqdm.tqdm(total=total_tasks + already_done, initial=already_done, desc="Evaluating 235B") as pbar:

        while workers_done < num_workers:
            try:
                msg = result_queue.get(timeout=120)
            except Exception:
                print(f"[Writer] queue timeout, workers_done={workers_done}/{num_workers}")
                continue
            if msg is None:
                workers_done += 1
                continue
            if msg["type"] == "result":
                f_out.write(json.dumps(msg["data"], ensure_ascii=False) + "\n")
                n_ok += 1
            else:
                f_fail.write(json.dumps(msg["data"], ensure_ascii=False) + "\n")
                n_fail += 1
            pbar.update(1)

    print(f"[Writer] finished. ok={n_ok}, fail={n_fail}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    df = pd.read_csv(CSV_PATH)
    print(f"Total rows: {len(df)}")

    imgs = {f.split("_")[0]: f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")}

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    fail_jsonl = args.output_jsonl.replace(".jsonl", ".failures.jsonl")

    # Resume support
    done_ids = set()
    if os.path.exists(args.output_jsonl):
        with open(args.output_jsonl) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
    print(f"Already done: {len(done_ids)}")

    # Build work items
    items = []
    for row in df.itertuples():
        if row.id in done_ids:
            continue
        img_file = imgs.get(str(row.id), "")
        img_path = os.path.join(IMG_DIR, img_file)
        if not os.path.exists(img_path):
            continue
        items.append({
            "id":     row.id,
            "gt_lat": row.latitude,
            "gt_lon": row.longitude,
            "img_path": img_path,
        })

    print(f"To process: {len(items)}")
    if not items:
        print("Nothing to do.")
        return

    n_proc = min(args.num_processes, len(items))
    batch_size = max(1, len(items) // n_proc)
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    print(f"Processes: {len(batches)}  batch_size~{batch_size}")

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    from multiprocessing import Process, Manager
    manager = Manager()
    q = manager.Queue(maxsize=min(len(batches) * 20, 2000))

    writer = Process(
        target=writer_process,
        args=(q, args.output_jsonl, fail_jsonl, len(items), len(batches), len(done_ids)),
    )
    writer.start()

    workers = []
    for wid, batch in enumerate(batches):
        p = Process(target=process_batch, args=(batch, q, wid))
        p.start()
        workers.append(p)
        print(f"  Started worker {wid} (PID {p.pid})")

    for p in workers:
        p.join()
    writer.join()

    # ── Compute summary ────────────────────────────────────────────────────────
    all_results = []
    with open(args.output_jsonl) as f:
        for line in f:
            try:
                all_results.append(json.loads(line))
            except Exception:
                pass

    km_list = [r["km"] for r in all_results if r.get("km") is not None]
    n = len(km_list)
    n_no_pred = sum(1 for r in all_results if r.get("km") is None)

    def acc(t):
        return round(sum(1 for k in km_list if k <= t) / n, 4) if n else 0

    summary = {
        "tag":        args.tag,
        "n_total":    len(all_results),
        "n_parsed":   n,
        "n_no_pred":  n_no_pred,
        "acc_1km":    acc(1),
        "acc_25km":   acc(25),
        "acc_200km":  acc(200),
        "acc_750km":  acc(750),
        "acc_2500km": acc(2500),
    }

    summary_path = args.output_jsonl.replace(".jsonl", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  Im2GPS3K — {args.tag}  [Qwen3-VL-235B]")
    print("=" * 60)
    print(f"  Samples:    {n}/{len(all_results)} parsed  ({n_no_pred} no pred)")
    print(f"  Acc@1km:    {summary['acc_1km']:.3f}")
    print(f"  Acc@25km:   {summary['acc_25km']:.3f}")
    print(f"  Acc@200km:  {summary['acc_200km']:.3f}")
    print(f"  Acc@750km:  {summary['acc_750km']:.3f}")
    print(f"  Acc@2500km: {summary['acc_2500km']:.3f}")
    print("=" * 60)
    print(f"  Output  → {args.output_jsonl}")
    print(f"  Summary → {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval Qwen3-VL-235B on Im2GPS3K")
    parser.add_argument("--num_processes", type=int, default=10)
    parser.add_argument("--tag", default="qwen3vl235b_notool_v3")
    parser.add_argument(
        "--output_jsonl",
        default=f"{OUTPUT_DIR}/qwen3vl235b_notool_v3.jsonl",
    )
    parser.add_argument("--max_samples", type=int, default=-1)
    args = parser.parse_args()
    if args.max_samples > 0:
        import pandas as _pd
        _df = _pd.read_csv(CSV_PATH).head(args.max_samples)
        _df.to_csv(CSV_PATH + ".tmp", index=False)
    main(args)
