#!/usr/bin/env bash
# =============================================================================
# run_eval_rl_checkpoint.sh — Eval RL checkpoint on Im2GPS3K
#
# Launches SGLang with the checkpoint's own chat_template.jinja,
# uses coldstart system prompt (from build_sft_coldstart.py),
# and runs eval with zoom+image_search+text_search (G config).
#
# Usage:
#   # 50 samples (quick check)
#   CKPT=/path/to/global_step_80/actor/huggingface bash eval/run_eval_rl_checkpoint.sh
#
#   # Full 2997 samples
#   CKPT=/path/to/global_step_80/actor/huggingface MAX_SAMPLES=-1 bash eval/run_eval_rl_checkpoint.sh
#
#   # Custom tag and GPU
#   CKPT=... TAG=my_run GPU=2 PORT=31002 bash eval/run_eval_rl_checkpoint.sh
#
# IMPORTANT:
#   - Must use --chat-template from the checkpoint (${CKPT}/chat_template.jinja)
#     SGLang default Qwen3 template uses hermes tool_calls format, but our model
#     was trained with <tool_call> XML format. Without this, tools=0 in eval.
#   - Uses --no_api_tools: tools hit SQLite cache only, no live API calls.
#   - System prompt from build_sft_coldstart.py (contains <useful> tag instruction).
# =============================================================================

set -euo pipefail

# ── Required ─────────────────────────────────────────────────────────────────
CKPT="${CKPT:?Set CKPT=/path/to/actor/huggingface}"

# ── Optional (with defaults) ─────────────────────────────────────────────────
GPU="${GPU:-0}"
PORT="${PORT:-31000}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"
SUBSET_SEED="${SUBSET_SEED:-42}"
MAX_TURNS="${MAX_TURNS:-10}"
CONCURRENCY="${CONCURRENCY:-8}"
BENCHMARK="${BENCHMARK:-im2gps3k}"

EVAL_SCRIPT="/mnt/sh/mmvision/home/jonahli/projects/tusou/eval/eval_benchmark.py"
ENV_FILE="/mnt/sh/mmvision/home/jonahli/projects/tusou/eval/.env"
if [ "$BENCHMARK" = "yfcc4k" ]; then
    OUT_DIR="/mnt/sh/mmvision/home/jonahli/save/agent/eval/yfcc4k"
    LOG_DIR="/mnt/sh/mmvision/home/jonahli/save/agent/eval/yfcc4k/logs"
else
    OUT_DIR="/mnt/sh/mmvision/home/jonahli/save/agent/eval/im2gps3k/coldstart"
    LOG_DIR="/mnt/sh/mmvision/home/jonahli/save/agent/eval/im2gps3k/logs"
fi
TOOLS="zoom,image_search,text_search"

# ── Chat template: MUST use checkpoint's own template ────────────────────────
CHAT_TEMPLATE="${CKPT}/chat_template.jinja"
if [ ! -f "$CHAT_TEMPLATE" ]; then
    echo "ERROR: chat_template.jinja not found in checkpoint: $CHAT_TEMPLATE"
    echo "  This is required for correct <tool_call> XML format."
    echo "  Without it, SGLang uses hermes format and model won't call tools."
    exit 1
fi

# ── System prompt: extract from build_sft_coldstart.py ───────────────────────
SYS_PROMPT_FILE="/tmp/coldstart_sys_prompt_$(date +%s).txt"
python3 -c "
import re
with open('/mnt/sh/mmvision/home/jonahli/projects/tusou/data_pipeline/build_sft_coldstart.py') as f:
    src = f.read()
m = re.search(r'SYSTEM_PROMPT = \((.*?)\)\n\n', src, re.DOTALL)
prompt = eval(m.group(0).split('= ', 1)[1].rstrip())
with open('$SYS_PROMPT_FILE', 'w') as f:
    f.write(prompt)
print('System prompt written:', len(prompt), 'chars')
"

# ── Derive tag from checkpoint path ──────────────────────────────────────────
# e.g. .../NNODES4-base-easy/global_step_80/actor/huggingface → rl_base_step80
STEP_DIR=$(basename "$(dirname "$(dirname "$CKPT")")")  # global_step_80
STEP_NUM=$(echo "$STEP_DIR" | grep -oP '\d+' | tail -1)
EXP_DIR=$(basename "$(dirname "$(dirname "$(dirname "$CKPT")")")")  # NNODES4-base-easy
# Extract variant: base or mp16pro
if echo "$EXP_DIR" | grep -q "mp16pro"; then
    VARIANT="mp16pro"
else
    VARIANT="base"
fi

if [ "$MAX_SAMPLES" = "-1" ]; then
    SAMPLE_TAG="full"
else
    SAMPLE_TAG="${MAX_SAMPLES}"
fi
TAG="${TAG:-rl_${VARIANT}_step${STEP_NUM}_G_${SAMPLE_TAG}}"

mkdir -p "$OUT_DIR" "$LOG_DIR"
OUTPUT_JSONL="${OUT_DIR}/${TAG}.jsonl"
SGLANG_LOG="$LOG_DIR/sglang_${TAG}.log"
EVAL_LOG="$LOG_DIR/${TAG}.log"
SGLANG_URL="http://localhost:${PORT}"

# Load env
if [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
fi

echo "========================================"
echo "  RL Checkpoint Eval"
echo "  CKPT:     $CKPT"
echo "  VARIANT:  $VARIANT"
echo "  STEP:     $STEP_NUM"
echo "  TAG:      $TAG"
echo "  TEMPLATE: $CHAT_TEMPLATE"
echo "  SAMPLES:  $MAX_SAMPLES"
echo "  GPU:      $GPU"
echo "  OUT:      $OUTPUT_JSONL"
echo "========================================"

# ── Launch SGLang ────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=$GPU python3 -m sglang.launch_server \
    --model-path "$CKPT" \
    --host 0.0.0.0 --port "$PORT" \
    --tp 1 --trust-remote-code \
    --chat-template "$CHAT_TEMPLATE" \
    --context-length 65536 \
    > "$SGLANG_LOG" 2>&1 &
SGLANG_PID=$!

echo -n "Waiting for SGLang..."
for i in $(seq 1 300); do
    code=$(curl -s -o /dev/null -w '%{http_code}' "$SGLANG_URL/health" 2>/dev/null || echo "000")
    if [ "$code" = "200" ]; then
        echo " ready (${i}x2s)"; break
    fi
    if [ $i -eq 120 ]; then
        echo " TIMEOUT"
        tail -20 "$SGLANG_LOG"
        kill "$SGLANG_PID" 2>/dev/null || true
        exit 1
    fi
    sleep 2; echo -n "."
done

# ── Run eval ─────────────────────────────────────────────────────────────────
EVAL_ARGS=(
    --tag "$TAG"
    --output_jsonl "$OUTPUT_JSONL"
    --benchmark "$BENCHMARK"
    --mode agent
    --tools "$TOOLS"
    --system_prompt "@${SYS_PROMPT_FILE}"
    --no_api_tools
    --no_save_images
    --max_turns "$MAX_TURNS"
    --max_tokens 32768
    --sglang_url "$SGLANG_URL"
    --concurrency "$CONCURRENCY"
)
if [ "$MAX_SAMPLES" != "-1" ]; then
    EVAL_ARGS+=(--max_samples "$MAX_SAMPLES" --subset_seed "$SUBSET_SEED")
fi

python3 "$EVAL_SCRIPT" "${EVAL_ARGS[@]}" > "$EVAL_LOG" 2>&1
RC=$?

kill "$SGLANG_PID" 2>/dev/null || true
wait "$SGLANG_PID" 2>/dev/null || true
rm -f "$SYS_PROMPT_FILE"

# ── Report ───────────────────────────────────────────────────────────────────
if [[ $RC -eq 0 ]]; then
    echo "DONE"
    SF="${OUT_DIR}/${TAG}_summary.json"
    [ -f "$SF" ] && python3 -c "
import json; s=json.load(open('$SF'))
print(f\"  {s['tag']:<50} @1km={s['acc_1km']:.3f}  @25km={s['acc_25km']:.3f}  @200km={s['acc_200km']:.3f}  @750km={s.get('acc_750km',0):.3f}  tools={s.get('avg_tool_calls',0):.1f}\")
" || echo "  $TAG — no summary found"
else
    echo "FAILED (rc=$RC)"
    tail -20 "$EVAL_LOG"
fi
