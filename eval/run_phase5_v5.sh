#!/usr/bin/env bash
# =============================================================================
# run_phase5_v5.sh — Phase 5 v5 实验 (Kimi K2d6, Im2GPS3K)
#
# 与 v4 相同设置，以下改动：
#   1. think-truncation nudge: 当模型 think 完后忘记输出 tool_call 或 <answer>
#      时，注入 user 提醒消息，让模型继续（不再直接 break）
#
# 7 个实验 (全部重跑):
#   A. no-tool baseline (single-turn)
#   B. zoom-only
#   C. image_search-only
#   D. text_search-only
#   E. zoom + text_search
#   F. image_search + text_search
#   G. zoom + image_search + text_search (all tools)
#
# 统一参数:
#   model=kimi_k2d6, n=50, seed=42, max_tokens=8192, concurrency=2
#   max_turns=10 (B~G)
#
# 用法:
#   cd /mnt/sh/mmvision/home/jonahli/projects/tusou
#   bash eval/run_phase5_v5.sh           # 跑全部
#   bash eval/run_phase5_v5.sh B         # 只跑 B
#   bash eval/run_phase5_v5.sh D E       # 只跑 D E
# =============================================================================

set -euo pipefail

# ── 依赖检查由 eval_benchmark.py 内置 _ensure_deps() 自动处理 ─────────────────

# ── 加载 API keys (.env 含 TAVILY_API_KEY) ──────────────────────────────────
ENV_FILE="$(dirname "${BASH_SOURCE[0]}")/.env"
if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
    echo "TAVILY_API_KEY set: ${TAVILY_API_KEY:+yes (${#TAVILY_API_KEY} chars)}"
else
    echo "ERROR: .env not found at $ENV_FILE — text_search will not work!"
    exit 1
fi

# ── 路径 ────────────────────────────────────────────────────────────────────
EVAL=/mnt/sh/mmvision/home/jonahli/projects/tusou/eval/eval_benchmark.py
OUT=/mnt/sh/mmvision/home/jonahli/save/agent/eval/im2gps3k/kimi_phase5_v5
LOG_DIR="${OUT}/logs"
mkdir -p "$OUT" "$LOG_DIR"

# ── 公共参数 ─────────────────────────────────────────────────────────────────
COMMON_ARGS=(
    --mode agent
    --model kimi_k2d6
    --max_samples 50
    --subset_seed 42
    --max_turns 10
    --max_tokens 8192
    --concurrency 2
    --image_store dir
)

# ── 选择运行哪些实验 ──────────────────────────────────────────────────────────
RUN_ALL=true
TARGETS=()
if [[ $# -gt 0 ]]; then
    RUN_ALL=false
    TARGETS=("$@")
fi

should_run() {
    $RUN_ALL && return 0
    for t in "${TARGETS[@]}"; do
        [[ "$t" == "$1" ]] && return 0
    done
    return 1
}

# ── 后台运行单个实验 ──────────────────────────────────────────────────────────
run_exp_bg() {
    local label=$1 tools=$2 tag=$3
    local out_jsonl="${OUT}/${tag}.jsonl"
    local log="${LOG_DIR}/${tag}.log"
    echo ""
    echo "========================================================"
    echo "  [BG] Exp ${label}: --tools ${tools}"
    echo "  output → ${out_jsonl}"
    echo "  log    → ${log}"
    echo "========================================================"
    python3 "$EVAL" \
        "${COMMON_ARGS[@]}" \
        --tools "$tools" \
        --tag "$tag" \
        --output_jsonl "$out_jsonl" \
        > "$log" 2>&1 &
    echo "  PID: $!"
}

# ── 实验 A: no-tool (single-turn) ────────────────────────────────────────────
if should_run A; then
    tag="A_notool"
    out_jsonl="${OUT}/${tag}.jsonl"
    log="${LOG_DIR}/${tag}.log"
    echo ""
    echo "========================================================"
    echo "  [BG] Exp A: no-tool single-turn"
    echo "  output → ${out_jsonl}"
    echo "  log    → ${log}"
    echo "========================================================"
    python3 "$EVAL" \
        --mode single \
        --notool \
        --model kimi_k2d6 \
        --max_samples 50 \
        --subset_seed 42 \
        --max_tokens 8192 \
        --concurrency 2 \
        --tag "$tag" \
        --output_jsonl "$out_jsonl" \
        > "$log" 2>&1 &
    echo "  PID: $!"
fi

# ── 实验 B: zoom-only ────────────────────────────────────────────────────────
should_run B && run_exp_bg "B" "zoom" "B_zoom_only"

# ── 实验 C: image_search-only ────────────────────────────────────────────────
should_run C && run_exp_bg "C" "image_search" "C_imgsearch_only"

# ── 实验 D: text_search-only ─────────────────────────────────────────────────
should_run D && run_exp_bg "D" "text_search" "D_textsearch_only"

# ── 实验 E: zoom + text_search ───────────────────────────────────────────────
should_run E && run_exp_bg "E" "zoom,text_search" "E_zoom_textsearch"

# ── 实验 F: image_search + text_search ──────────────────────────────────────
should_run F && run_exp_bg "F" "image_search,text_search" "F_imgsearch_textsearch"

# ── 实验 G: all tools ────────────────────────────────────────────────────────
should_run G && run_exp_bg "G" "zoom,image_search,text_search" "G_imgsearch_textsearch_zoom"

# ── 等待所有后台任务完成 ───────────────────────────────────────────────────────
echo ""
echo "Waiting for all background experiments to finish..."
wait
echo "All experiments done."

# ── 汇总结果 ─────────────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "  Phase 5 v5 — Im2GPS3K Results"
echo "=================================================================="
printf "  %-8s %-36s %6s %7s %8s %8s %9s\n" "Exp" "Tag" "@1km" "@25km" "@200km" "@750km" "@2500km"
printf "  %-8s %-36s %6s %7s %8s %8s %9s\n" "---" "---" "----" "-----" "------" "------" "-------"

for row in "A A_notool" "B B_zoom_only" "C C_imgsearch_only" "D D_textsearch_only" "E E_zoom_textsearch" "F F_imgsearch_textsearch" "G G_imgsearch_textsearch_zoom"; do
    exp=$(echo "$row" | cut -d' ' -f1)
    tag=$(echo "$row" | cut -d' ' -f2)
    sf="${OUT}/${tag}_summary.json"
    if [[ -f "$sf" ]]; then
        python3 -c "
import json; s=json.load(open('$sf'))
print(f\"  $exp       {s.get('tag','$tag'):<36} {s['acc_1km']:>6.3f} {s['acc_25km']:>7.3f} {s['acc_200km']:>8.3f} {s['acc_750km']:>8.3f} {s['acc_2500km']:>9.3f}\")
" 2>/dev/null || printf "  %-8s %-36s  (parse error)\n" "$exp" "$tag"
    else
        printf "  %-8s %-36s  (no result yet)\n" "$exp" "$tag"
    fi
done
echo "=================================================================="
echo "  Logs → $LOG_DIR"
echo "=================================================================="
