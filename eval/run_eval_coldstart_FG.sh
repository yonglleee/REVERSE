#!/usr/bin/env bash
# =============================================================================
# run_eval_coldstart_FG.sh — Eval coldstart SFT model on Im2GPS3K
#   F: image_search + text_search (50 samples)
#   G: zoom + image_search + text_search (50 samples)
#
# Usage:
#   bash eval/run_eval_coldstart_FG.sh
#   CKPT=.../huggingface bash eval/run_eval_coldstart_FG.sh  # override checkpoint
# =============================================================================

set -euo pipefail

CKPT="${CKPT:-/mnt/sh/mmvision/home/jonahli/save/tusou/sft/coldstart-multiturn-sft-qwen3-4b-fsdp-fsdp2-sp1-n1-lr2e-5-bs128-from-mp16pro-step7800-v2/global_step_24/huggingface}"
EVAL_SCRIPT="/mnt/sh/mmvision/home/jonahli/projects/tusou/eval/eval_benchmark.py"
ENV_FILE="/mnt/sh/mmvision/home/jonahli/projects/tusou/eval/.env"
LOG_DIR="/mnt/sh/mmvision/home/jonahli/save/agent/eval/im2gps3k/logs"
CHAT_TEMPLATE="${CKPT}/chat_template.jinja"

BASE_GPU=${BASE_GPU:-0}
BASE_PORT=${BASE_PORT:-31000}
CONCURRENCY=${CONCURRENCY:-8}

mkdir -p "$LOG_DIR"

# Load TAVILY keys
if [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
fi

echo "========================================"
echo " Coldstart SFT Eval: F & G"
echo " CKPT: $CKPT"
echo "========================================"

launch_one() {
    local tag=$1
    local tools=$2
    local gpu=$3
    local port=$4
    local sglang_log="$LOG_DIR/sglang_${tag}.log"
    local eval_log="$LOG_DIR/${tag}.log"
    local sglang_url="http://localhost:${port}"

    echo ">>> [$tag] GPU=$gpu port=$port tools=$tools"

    CUDA_VISIBLE_DEVICES=$gpu python3 -m sglang.launch_server \
        --model-path "$CKPT" \
        --host 0.0.0.0 --port "$port" \
        --tp 1 --trust-remote-code \
        --chat-template "$CHAT_TEMPLATE" \
        > "$sglang_log" 2>&1 &
    local sglang_pid=$!

    echo -n "    [$tag] waiting for SGLang..."
    for i in $(seq 1 120); do
        code=$(curl -s -o /dev/null -w '%{http_code}' "$sglang_url/health" 2>/dev/null || echo "000")
        if [ "$code" = "200" ]; then
            echo " ready (${i}x2s)"; break
        fi
        if [ $i -eq 120 ]; then
            echo " TIMEOUT"; kill "$sglang_pid" 2>/dev/null || true; return 1
        fi
        sleep 2; echo -n "."
    done

    python3 "$EVAL_SCRIPT" \
        --tag "$tag" \
        --mode agent \
        --tools "$tools" \
        --max_samples 50 \
        --subset_seed 42 \
        --max_turns 10 \
        --max_tokens 8192 \
        --sglang_url "$sglang_url" \
        --concurrency "$CONCURRENCY" \
        > "$eval_log" 2>&1
    local rc=$?

    kill "$sglang_pid" 2>/dev/null || true
    wait "$sglang_pid" 2>/dev/null || true

    if [[ $rc -eq 0 ]]; then
        echo "    [$tag] DONE"
        grep -E "acc_1km|acc_25km|acc_200km|parsed" "$eval_log" | tail -5
    else
        echo "    [$tag] FAILED (rc=$rc) → $eval_log"
        tail -20 "$eval_log"
    fi
    return $rc
}

# Run F and G sequentially (single node, share GPUs safely)
launch_one "coldstart_v2_F_imgsearch_textsearch" "image_search,text_search" "$BASE_GPU" "$BASE_PORT"
launch_one "coldstart_v2_G_zoom_imgsearch_textsearch" "zoom,image_search,text_search" "$BASE_GPU" "$BASE_PORT"

echo ""
echo "=================================================================="
echo "  Results"
echo "=================================================================="
EVAL_DIR="/mnt/sh/mmvision/home/jonahli/save/agent/eval/im2gps3k"
for tag in "coldstart_v2_F_imgsearch_textsearch" "coldstart_v2_G_zoom_imgsearch_textsearch"; do
    sf="$EVAL_DIR/${tag}_summary.json"
    [ -f "$sf" ] && python3 -c "
import json; s=json.load(open('$sf'))
print(f\"  {s['tag']:<45} @1km={s['acc_1km']:.3f}  @25km={s['acc_25km']:.3f}  @200km={s['acc_200km']:.3f}  tool_calls={s.get('avg_tool_calls',0):.2f}\")
" || echo "  $tag — no summary found"
done
echo "=================================================================="
