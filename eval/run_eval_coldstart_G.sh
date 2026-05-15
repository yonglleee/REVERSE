#!/usr/bin/env bash
# =============================================================================
# run_eval_coldstart_G.sh — Eval coldstart SFT model on Im2GPS3K
#   G: zoom + image_search + text_search (50 samples)
#   Uses the coldstart training system prompt (3-tool version)
#
# Usage:
#   bash eval/run_eval_coldstart_G.sh
#   CKPT=.../huggingface bash eval/run_eval_coldstart_G.sh
# =============================================================================

set -euo pipefail

CKPT="${CKPT:-/mnt/sh/mmvision/home/jonahli/save/tusou/sft/coldstart-multiturn-sft-qwen3-4b-fsdp-fsdp2-sp1-n1-lr2e-5-bs128-v3/global_step_25/huggingface}"
EVAL_SCRIPT="/mnt/sh/mmvision/home/jonahli/projects/tusou/eval/eval_benchmark.py"
ENV_FILE="/mnt/sh/mmvision/home/jonahli/projects/tusou/eval/.env"
LOG_DIR="/mnt/sh/mmvision/home/jonahli/save/agent/eval/im2gps3k/logs"
CHAT_TEMPLATE="${CKPT}/chat_template.jinja"
SYSTEM_PROMPT_FILE="/mnt/sh/mmvision/home/jonahli/projects/tusou/eval/coldstart_system_prompt.txt"

BASE_GPU=${BASE_GPU:-0}
BASE_PORT=${BASE_PORT:-31000}
MAX_SAMPLES="${MAX_SAMPLES:-50}"
SUBSET_SEED="${SUBSET_SEED:-42}"
CONCURRENCY=${CONCURRENCY:-8}

mkdir -p "$LOG_DIR"

# Load TAVILY keys
if [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
fi

# Write coldstart system prompt to file if not already there
python3 -c "
import re
with open('/mnt/sh/mmvision/home/jonahli/projects/tusou/data_pipeline/build_sft_coldstart.py') as f:
    src = f.read()
m = re.search(r'SYSTEM_PROMPT = \((.*?)\)\n\n', src, re.DOTALL)
prompt = eval(m.group(0).split('= ', 1)[1].rstrip())
with open('$SYSTEM_PROMPT_FILE', 'w') as f:
    f.write(prompt)
print('System prompt written:', len(prompt), 'chars')
"

CKPT_STEP=$(basename "$(dirname "$CKPT")")   # e.g. global_step_25
CKPT_NAME=$(basename "$(dirname "$(dirname "$CKPT")")")  # e.g. coldstart-multiturn-sft-...
# derive a short version tag: coldstart_v3_step25 (override via TAG env var)
TAG="${TAG:-coldstart_v3_G_zoom_imgsearch_textsearch_${CKPT_STEP}}"
OUT_DIR="/mnt/sh/mmvision/home/jonahli/save/agent/eval/im2gps3k/coldstart"
mkdir -p "$OUT_DIR"
OUTPUT_JSONL="${OUTPUT_JSONL:-${OUT_DIR}/${TAG}.jsonl}"
TOOLS="zoom,image_search,text_search"
GPU="$BASE_GPU"
PORT="$BASE_PORT"
SGLANG_LOG="$LOG_DIR/sglang_${TAG}.log"
EVAL_LOG="$LOG_DIR/${TAG}.log"
SGLANG_URL="http://localhost:${PORT}"

echo "========================================"
echo " Coldstart SFT Eval: G (zoom+imgsearch+textsearch)"
echo " CKPT: $CKPT"
echo " TAG:  $TAG"
echo " OUT:  $OUTPUT_JSONL"
echo " System prompt: $SYSTEM_PROMPT_FILE"
echo "========================================"

echo ">>> [$TAG] GPU=$GPU port=$PORT tools=$TOOLS"

CUDA_VISIBLE_DEVICES=$GPU python3 -m sglang.launch_server \
    --model-path "$CKPT" \
    --host 0.0.0.0 --port "$PORT" \
    --tp 1 --trust-remote-code \
    --chat-template "$CHAT_TEMPLATE" \
    --context-length 65536 \
    > "$SGLANG_LOG" 2>&1 &
SGLANG_PID=$!

echo -n "    [$TAG] waiting for SGLang..."
for i in $(seq 1 300); do
    code=$(curl -s -o /dev/null -w '%{http_code}' "$SGLANG_URL/health" 2>/dev/null || echo "000")
    if [ "$code" = "200" ]; then
        echo " ready (${i}x2s)"; break
    fi
    if [ $i -eq 120 ]; then
        echo " TIMEOUT"; kill "$SGLANG_PID" 2>/dev/null || true; exit 1
    fi
    sleep 2; echo -n "."
done

python3 "$EVAL_SCRIPT" \
    --tag "$TAG" \
    --output_jsonl "$OUTPUT_JSONL" \
    --mode agent \
    --tools "$TOOLS" \
    --system_prompt "@${SYSTEM_PROMPT_FILE}" \
    --no_api_tools \
    --no_save_images \
    --max_turns 10 \
    --max_tokens 8192 \
    --sglang_url "$SGLANG_URL" \
    --concurrency "$CONCURRENCY" \
    $( [ "$MAX_SAMPLES" != "-1" ] && echo "--max_samples $MAX_SAMPLES --subset_seed $SUBSET_SEED" ) \
    > "$EVAL_LOG" 2>&1
RC=$?

kill "$SGLANG_PID" 2>/dev/null || true
wait "$SGLANG_PID" 2>/dev/null || true

if [[ $RC -eq 0 ]]; then
    echo "    [$TAG] DONE"
    grep -E "acc_1km|acc_25km|acc_200km|parsed" "$EVAL_LOG" | tail -5
else
    echo "    [$TAG] FAILED (rc=$RC) → $EVAL_LOG"
    tail -20 "$EVAL_LOG"
fi

echo ""
echo "=================================================================="
echo "  Results"
echo "=================================================================="
EVAL_DIR="$OUT_DIR"
SF="${OUT_DIR}/${TAG}_summary.json"
[ -f "$SF" ] && python3 -c "
import json, re
s = json.load(open('$SF'))
tag = s['tag']
a1, a25, a200 = s['acc_1km'], s['acc_25km'], s['acc_200km']
tc = s.get('avg_tool_calls', 0)
sc = s.get('avg_search_calls', 0)
# Cache hit rate from jsonl
img_hits = img_total = txt_hits = txt_total = 0
try:
    with open('${OUTPUT_JSONL}') as f:
        for line in f:
            d = json.loads(line)
            out = d.get('output', '') or ''
            # image search cache hits from tool responses in messages
            for m in re.finditer(r'Image search results', out):
                img_total += 1
            for m in re.finditer(r'cache_hit.*?True', out):
                img_hits += 1
except: pass
cache_str = f'img_cache={img_hits}/{img_total}' if img_total else ''
print(f'  {tag:<50} @1km={a1:.3f}  @25km={a25:.3f}  @200km={a200:.3f}  tools={tc:.2f}  search={sc:.2f}  {cache_str}')
" || echo "  $TAG — no summary found"
echo "=================================================================="
