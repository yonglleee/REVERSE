#!/usr/bin/env bash
# =============================================================================
# run_8b_eval_all_steps.sh
# Evaluate 8b SFT model across ALL 39 checkpoints on Im2GPS3K.
#
# 8 GPUs available → run 8 jobs in parallel (5 rounds: 8+8+8+8+7).
# Each job: launch SGLang on one GPU, run eval, kill SGLang.
#
# Model:  mp16pro-geo-qwen3-8b-v2-fsdp-fsdp2-sp1-n8-lr2e-5-bs512
# Steps:  200 400 ... 7800 7842
#
# USAGE:
#   bash run_8b_eval_all_steps.sh
#   BASE_GPU=0 BASE_PORT=31000 LABEL_FORMAT=l2 bash run_8b_eval_all_steps.sh
# =============================================================================

set -euo pipefail

EVAL_SCRIPT="/mnt/sh/mmvision/home/jonahli/projects/tusou/eval/eval_sft_granularity.py"
SFT_DIR="/mnt/sh/mmvision/home/jonahli/save/tusou/sft/mp16pro-geo-qwen3-8b-v2-fsdp-fsdp2-sp1-n8-lr2e-5-bs512"
LOG_DIR="/mnt/sh/mmvision/home/jonahli/save/tusou/eval/8b_all_steps/logs"
EVAL_DIR="/mnt/sh/mmvision/home/jonahli/save/tusou/eval/8b_all_steps"

BASE_GPU="${BASE_GPU:-0}"
BASE_PORT="${BASE_PORT:-31000}"
CONCURRENCY="${CONCURRENCY:-32}"
LABEL_FORMAT="${LABEL_FORMAT:-l2_no_prefix}"
N_GPU=8

mkdir -p "$LOG_DIR" "$EVAL_DIR"

STEPS=(200 400 600 800 1000 1200 1400 1600 1800 2000 2200 2400 2600 2800 3000 3200 3400 3600 3800 4000 4200 4400 4600 4800 5000 5200 5400 5600 5800 6000 6200 6400 6600 6800 7000 7200 7400 7600 7800 7842)
TOTAL=${#STEPS[@]}  # 39 (including 7842)

# ---------------------------------------------------------------------------
# launch_one <step> <gpu_abs> <port>
# ---------------------------------------------------------------------------
launch_one() {
    local step=$1
    local gpu=$2
    local port=$3

    local ckpt="${SFT_DIR}/global_step_${step}/huggingface"
    local tag="sft_8b_${LABEL_FORMAT}_step${step}"
    local sglang_url="http://localhost:${port}"
    local sglang_log="${LOG_DIR}/sglang_${tag}.log"
    local eval_log="${LOG_DIR}/${tag}.log"
    local out_jsonl="${EVAL_DIR}/${tag}.jsonl"

    # Skip if valid summary already exists (n_parsed > 0)
    if [ -f "${EVAL_DIR}/${tag}_summary.json" ]; then
        local n_parsed
        n_parsed=$(python3 -c "import json; s=json.load(open('${EVAL_DIR}/${tag}_summary.json')); print(s.get('n_parsed',0))" 2>/dev/null || echo "0")
        if [ "$n_parsed" -gt 0 ]; then
            echo "  [SKIP] $tag — already done (n_parsed=$n_parsed)"
            return 0
        else
            echo "  [REDO] $tag — summary exists but n_parsed=0, re-running"
            rm -f "${EVAL_DIR}/${tag}_summary.json" "${out_jsonl}"
        fi
    fi

    if [ ! -d "$ckpt" ]; then
        echo "  [MISS] $tag — checkpoint not found: $ckpt"
        return 1
    fi

    echo "  [START] $tag  GPU=$gpu port=$port"

    # Launch SGLang with the checkpoint's own chat_template.jinja (120-line full Qwen3-VL
    # template with vision token support). Without this, "You are a geolocation expert..."
    # prefix triggers "!!!!" for step1800+ due to token-sequence mismatch.
    # We use label_format=l2_no_prefix (no "You are..." prefix) which works correctly.
    CUDA_VISIBLE_DEVICES=$gpu python3 -m sglang.launch_server \
        --model-path "$ckpt" \
        --host 0.0.0.0 --port "$port" \
        --tp 1 --trust-remote-code \
        --chat-template "${ckpt}/chat_template.jinja" \
        > "$sglang_log" 2>&1 &
    local sglang_pid=$!

    # Wait up to 5 min for SGLang (8b model loads slower)
    local ready=0
    for i in $(seq 1 150); do
        local code
        code=$(curl -s -o /dev/null -w '%{http_code}' "${sglang_url}/health" 2>/dev/null || echo "000")
        if [ "$code" = "200" ]; then
            ready=1; break
        fi
        if ! kill -0 "$sglang_pid" 2>/dev/null; then
            echo "  [FAIL] $tag — SGLang died. See $sglang_log"
            return 1
        fi
        sleep 2
    done

    if [ "$ready" -eq 0 ]; then
        echo "  [FAIL] $tag — SGLang timed out. See $sglang_log"
        kill "$sglang_pid" 2>/dev/null || true
        return 1
    fi

    # Run eval
    python3 "$EVAL_SCRIPT" \
        --label_format "$LABEL_FORMAT" \
        --tag "$tag" \
        --sglang_url "$sglang_url" \
        --concurrency "$CONCURRENCY" \
        --output_jsonl "$out_jsonl" \
        > "$eval_log" 2>&1
    local rc=$?

    kill "$sglang_pid" 2>/dev/null || true
    wait "$sglang_pid" 2>/dev/null || true

    if [ $rc -eq 0 ]; then
        echo "  [DONE] $tag"
        local sf="${EVAL_DIR}/${tag}_summary.json"
        if [ -f "$sf" ]; then
            python3 -c "
import json; s=json.load(open('$sf'))
print(f\"  {s['tag']}: @1km={s['acc_1km']:.3f}  @25km={s['acc_25km']:.3f}  @200km={s['acc_200km']:.3f}  @750km={s['acc_750km']:.3f}  @2500km={s['acc_2500km']:.3f}\")
" 2>/dev/null || true
        fi
    else
        echo "  [FAIL] $tag (rc=$rc) — see $eval_log"
    fi
    return $rc
}

export -f launch_one
export EVAL_SCRIPT SFT_DIR LOG_DIR EVAL_DIR LABEL_FORMAT CONCURRENCY

# ---------------------------------------------------------------------------
# Run in batches of N_GPU
# ---------------------------------------------------------------------------
echo "======================================================================"
echo "  8B SFT All-Steps Eval  |  label_format=${LABEL_FORMAT}"
echo "  Total: $TOTAL steps  |  GPUs: $N_GPU  |  Rounds: $(( (TOTAL + N_GPU - 1) / N_GPU ))"
echo "======================================================================"
echo ""

round=1
for (( batch_start=0; batch_start<TOTAL; batch_start+=N_GPU )); do
    batch_end=$(( batch_start + N_GPU - 1 ))
    [ $batch_end -ge $TOTAL ] && batch_end=$(( TOTAL - 1 ))
    batch_size=$(( batch_end - batch_start + 1 ))

    echo "--- Round $round (steps $((batch_start+1))–$((batch_end+1)) / $TOTAL) ---"

    pids=()
    for (( slot=0; slot<batch_size; slot++ )); do
        idx=$(( batch_start + slot ))
        step="${STEPS[$idx]}"
        gpu=$(( BASE_GPU + slot ))
        port=$(( BASE_PORT + slot ))
        launch_one "$step" "$gpu" "$port" &
        pids+=($!)
    done

    failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=$(( failed + 1 ))
    done
    [ $failed -gt 0 ] && echo "  WARNING: $failed job(s) failed in round $round."

    round=$(( round + 1 ))
    echo ""
done

# Kill any stray SGLang servers
pkill -f "sglang.launch_server" 2>/dev/null || true

# ---------------------------------------------------------------------------
# Final summary table
# ---------------------------------------------------------------------------
echo "======================================================================"
echo "  Im2GPS3K — 8B SFT All Steps  (label_format=${LABEL_FORMAT})"
printf "  %-44s %6s %7s %8s %8s %9s\n" "Tag" "@1km" "@25km" "@200km" "@750km" "@2500km"
printf "  %-44s %6s %7s %8s %8s %9s\n" "---" "----" "-----" "------" "------" "-------"

for step in "${STEPS[@]}"; do
    tag="sft_8b_${LABEL_FORMAT}_step${step}"
    sf="${EVAL_DIR}/${tag}_summary.json"
    if [ -f "$sf" ]; then
        python3 -c "
import json; s=json.load(open('$sf'))
print(f\"  {s['tag']:<44} {s['acc_1km']:>6.3f} {s['acc_25km']:>7.3f} {s['acc_200km']:>8.3f} {s['acc_750km']:>8.3f} {s['acc_2500km']:>9.3f}\")
" 2>/dev/null || printf "  %-44s  (parse error)\n" "$tag"
    else
        printf "  %-44s  (no result)\n" "$tag"
    fi
done

echo "======================================================================"
echo "Logs    → $LOG_DIR"
echo "Results → $EVAL_DIR"
