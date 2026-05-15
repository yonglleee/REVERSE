#!/usr/bin/env bash
# =============================================================================
# run_annotate_coldstart.sh — 用 Exp G (zoom + image_search + text_search) 对
# coldstart 数据做地理标注，生成带 useful_results 的 jsonl 供训练数据构建。
#
# 输入 CSV: /mnt/sh/mmvision/home/jonahli/data_agent/coldstart/raw/test_filtered_part*.csv
#   字段: IMG_ID, LAT, LON, city, country, path（完整图片路径）
#
# 输出: /mnt/sh/mmvision/home/jonahli/save/agent/annotate/coldstart/part{NN}.jsonl
#
# 用法:
#   bash eval/run_annotate_coldstart.sh            # 跑全部 part00~part11
#   bash eval/run_annotate_coldstart.sh 00         # 只跑 part00
#   bash eval/run_annotate_coldstart.sh 00 01 02   # 只跑指定 part
# =============================================================================

set -uo pipefail

# ── 依赖检查由 eval_benchmark.py 内置 _ensure_deps() 自动处理 ─────────────────

# ── 加载 API keys ─────────────────────────────────────────────────────────────
ENV_FILE="$(dirname "${BASH_SOURCE[0]}")/.env"
if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
    echo "TAVILY_API_KEY set: ${TAVILY_API_KEY:+yes (${#TAVILY_API_KEY} chars)}"
else
    echo "ERROR: .env not found at $ENV_FILE"
    exit 1
fi

# ── 启动前 Tavily quota 健康检查 ──────────────────────────────────────────────
# 自动注释耗尽的 key + 恢复可用的 key，保证 pool 里只有活的 key。
# 注意：被注释（即行首是 #）的 key 被 loader 作为 fallback 加载，
# 所以即使不手动操作，挂的 key 也会被尝试。quota check 能提前淘汰。
echo "==> Pre-flight: running check_api_quota.py --auto-manage --only-tavily"
python3 "$(dirname "${BASH_SOURCE[0]}")/scripts/check_api_quota.py" --auto-manage --only-tavily 2>&1 \
    | tail -5
echo "==> Pre-flight: done"

# ── 路径 ──────────────────────────────────────────────────────────────────────
EVAL=/mnt/sh/mmvision/home/jonahli/projects/tusou/eval/eval_benchmark.py
CSV_DIR=/mnt/sh/mmvision/home/jonahli/data_agent/coldstart/raw
OUT_DIR=/mnt/sh/mmvision/home/jonahli/save/agent/annotate/coldstart
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# ── 公共参数 ──────────────────────────────────────────────────────────────────
COMMON_ARGS=(
    --mode agent
    --model kimi_k2d6
    --tools image_search,text_search
    --max_samples -1        # 全量
    --max_turns 10
    --max_tokens 8192
    --concurrency 8
    --search_concurrency 8
    --tavily_concurrency 8
    --image_store dir
    --exclude_domains flickr.com
)

# ── 确定要跑哪些 part ─────────────────────────────────────────────────────────
if [[ $# -gt 0 ]]; then
    PARTS=("$@")
else
    PARTS=(02 03 04 05 06 07 08 09 10 11)
fi

# ── 逐 part 后台运行 ──────────────────────────────────────────────────────────
for part in "${PARTS[@]}"; do
    csv="${CSV_DIR}/test_filtered_part${part}.csv"
    if [[ ! -f "$csv" ]]; then
        echo "WARN: $csv not found, skipping."
        continue
    fi
    out_jsonl="${OUT_DIR}/part${part}.jsonl"
    log="${LOG_DIR}/part${part}.log"
    tag="annotate_part${part}"

    echo ""
    echo "========================================================"
    echo "  [BG] part${part}"
    echo "  csv    → ${csv}"
    echo "  output → ${out_jsonl}"
    echo "  log    → ${log}"
    echo "========================================================"
    python3 "$EVAL" \
        "${COMMON_ARGS[@]}" \
        --csv_path "$csv" \
        --tag "$tag" \
        --output_jsonl "$out_jsonl" \
        > "$log" 2>&1 &
    echo "  PID: $!"
done

echo ""
echo "Waiting for all background jobs to finish..."
wait
echo "All annotation jobs done."
echo "Output → $OUT_DIR"
