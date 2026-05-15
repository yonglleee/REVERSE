#!/usr/bin/env bash
# =============================================================================
# run_eval_kimi_imgsearch.sh — Eval Kimi K2d6 on Im2GPS3K
#   Tools: image_search only
#   Flickr domains excluded to prevent GT leakage
#   Uses a separate cache dir (no cross-contamination with other runs)
#
# Usage:
#   bash eval/run_eval_kimi_imgsearch.sh
#   MAX_SAMPLES=2997 bash eval/run_eval_kimi_imgsearch.sh
# =============================================================================

set -euo pipefail

MAX_SAMPLES="${MAX_SAMPLES:-50}"
SUBSET_SEED="${SUBSET_SEED:-42}"
MAX_TURNS="${MAX_TURNS:-10}"
CONCURRENCY="${CONCURRENCY:-8}"
TAG="${TAG:-kimi_k2d6_imgsearch_noflickr_${MAX_SAMPLES}}"

EVAL_SCRIPT="/mnt/sh/mmvision/home/jonahli/projects/tusou/eval/eval_benchmark.py"
ENV_FILE="/mnt/sh/mmvision/home/jonahli/projects/tusou/eval/.env"
OUT_DIR="/mnt/sh/mmvision/home/jonahli/save/agent/eval/im2gps3k/kimi"
LOG_DIR="/mnt/sh/mmvision/home/jonahli/save/agent/eval/im2gps3k/logs"
CACHE_DIR="/mnt/sh/mmvision/home/jonahli/save/agent/eval/cache_kimi_noflickr"

mkdir -p "$OUT_DIR" "$LOG_DIR" "$CACHE_DIR"

OUTPUT_JSONL="${OUT_DIR}/${TAG}.jsonl"
LOG_FILE="${LOG_DIR}/${TAG}.log"

# Load TAVILY keys
if [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
fi

echo "========================================"
echo " Kimi K2d6 Eval: image_search only"
echo " TAG:     $TAG"
echo " Samples: $MAX_SAMPLES (seed=$SUBSET_SEED)"
echo " Cache:   $CACHE_DIR"
echo " Out:     $OUTPUT_JSONL"
echo "========================================"

EVAL_CACHE_DIR="$CACHE_DIR" \
python3 "$EVAL_SCRIPT" \
    --model kimi_k2d6 \
    --mode agent \
    --tools image_search \
    --tag "$TAG" \
    --output_jsonl "$OUTPUT_JSONL" \
    --max_turns "$MAX_TURNS" \
    --max_tokens 8192 \
    --concurrency "$CONCURRENCY" \
    --exclude_domains flickr.com \
    $( [ "$MAX_SAMPLES" != "-1" ] && echo "--max_samples $MAX_SAMPLES --subset_seed $SUBSET_SEED" ) \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=================================================================="
echo "  Results — $TAG"
echo "=================================================================="
SF="${OUT_DIR}/${TAG}_summary.json"
[ -f "$SF" ] && python3 -c "
import json
s = json.load(open('$SF'))
print(f'  @1km={s[\"acc_1km\"]:.3f}  @25km={s[\"acc_25km\"]:.3f}  @200km={s[\"acc_200km\"]:.3f}  @750km={s[\"acc_750km\"]:.3f}  @2500km={s[\"acc_2500km\"]:.3f}')
print(f'  parsed={s[\"n_parsed\"]}/{s[\"n_total\"]}  avg_tool={s.get(\"avg_tool_calls\",0):.2f}')
" || echo "  no summary found"
echo "=================================================================="
