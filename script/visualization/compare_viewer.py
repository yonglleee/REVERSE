"""
compare_viewer.py — 两个 JSONL 实验结果对比查看器

用法:
    streamlit run script/visualization/compare_viewer.py

功能:
  - 加载两个 JSONL 文件（File A / File B，如 v1 vs v2）
  - 按阈值计算每个样本的 "改善" / "退步" 分类
  - 筛选分类后的样本列表，逐样本左右对比展示完整多轮对话
"""

import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

# ── 默认路径 ────────────────────────────────────────────────────────────────────
DEFAULT_FILE_A = "/mnt/sh/mmvision/home/jonahli/save/agent/eval/im2gps3k/kimi_phase5/kimi_zoom_only.jsonl"
DEFAULT_FILE_B = "/mnt/sh/mmvision/home/jonahli/save/agent/eval/im2gps3k/kimi_phase5/kimi_zoom_only_v2.jsonl"

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
THINK_RE     = re.compile(r"<think>(.*?)</think>", re.DOTALL)
IMAGE_ID_RE  = re.compile(r"image_id:\s*(\d+)")

DATASET_CONFIGS = {
    "im2gps3k": {
        "img_dir": "/mnt/sh/mmvision/home/jonahli/data_agent/benchmark/im2gps3ktest",
        "id_field": "id",
    },
    "自动检测": {"img_dir": None, "id_field": None},
}

THRESHOLDS = [1, 25, 200, 750, 2500]

# ── 页面配置 ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="实验对比查看器", layout="wide")
st.markdown(
    """
    <style>
    .stCodeBlock pre, .stCode pre,
    div[data-testid="stCodeBlock"] pre, div[data-testid="stCode"] pre,
    div[data-testid="stCodeBlock"] code, div[data-testid="stCode"] code {
        white-space: pre-wrap !important;
        overflow-wrap: anywhere !important;
        word-break: break-word !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_jsonl(path: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rec["_source_file"] = path
                rows.append(rec)
            except json.JSONDecodeError:
                rows.append({"_raw": line, "_parse_error": True, "_source_file": path})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def normalize_str(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False)
    return str(val)


def to_image_path_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        if raw.startswith("[") and raw.endswith("]"):
            try:
                parsed = json.loads(raw)
                items = parsed if isinstance(parsed, list) else [raw]
            except Exception:
                items = [raw]
        else:
            items = [raw]
    else:
        items = [value]
    return [str(x) for x in items if x]


def get_primary_image_path(row: pd.Series, dataset_cfg: Optional[Dict] = None) -> Optional[str]:
    for field in ("image_paths", "image_path", "images"):
        paths = to_image_path_list(row.get(field))
        if paths:
            return paths[0]
    if dataset_cfg and dataset_cfg.get("img_dir") and dataset_cfg.get("id_field"):
        img_dir  = dataset_cfg["img_dir"]
        id_field = dataset_cfg["id_field"]
        img_id   = row.get(id_field)
        if img_id is not None:
            img_id_str = str(int(img_id)) if isinstance(img_id, float) else str(img_id)
            try:
                candidates = [
                    f for f in os.listdir(img_dir)
                    if f.split("_")[0] == img_id_str and f.endswith(".jpg")
                ]
                if candidates:
                    return os.path.join(img_dir, candidates[0])
            except Exception:
                pass
    return None


def extract_zoom_bboxes(output_text: str) -> List[Dict[str, Any]]:
    bboxes = []
    for payload in TOOL_CALL_RE.findall(output_text or ""):
        try:
            tc = json.loads(payload)
        except Exception:
            continue
        if tc.get("name") != "image_zoom_in_tool":
            continue
        args = tc.get("arguments", {}) or {}
        bbox = args.get("bbox_2d")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except Exception:
            continue
        bboxes.append({"bbox_2d": [x1, y1, x2, y2],
                        "label": normalize_str(args.get("label", "")).strip()})
    return bboxes


def draw_bbox_over_image(image_path: str, bboxes: List[Dict]) -> Optional[Image.Image]:
    if not bboxes or not image_path or image_path.startswith("http"):
        return None
    if not os.path.exists(image_path):
        return None
    image = Image.open(image_path).convert("RGB")
    draw  = ImageDraw.Draw(image)
    w, h  = image.size
    lw    = max(2, min(w, h) // 250)
    for idx, item in enumerate(bboxes, 1):
        x1, y1, x2, y2 = item["bbox_2d"]
        mv = max(abs(x1), abs(y1), abs(x2), abs(y2))
        if mv <= 1.5:
            l, t, r, b = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
        elif mv <= 1000.0:
            l, t, r, b = int(x1/1000*w), int(y1/1000*h), int(x2/1000*w), int(y2/1000*h)
        else:
            l, t, r, b = int(x1), int(y1), int(x2), int(y2)
        l, t, r, b = max(0,min(w,l)), max(0,min(h,t)), max(0,min(w,r)), max(0,min(h,b))
        if r > l and b > t:
            draw.rectangle([(l,t),(r,b)], outline=(255,0,0), width=lw)
            draw.text((l+4, max(0,t-16)), item.get("label") or f"#{idx}", fill=(255,0,0))
    return image


def render_messages(row: pd.Series, img_width: int = 320) -> None:
    """渲染 messages 字段（多轮对话）。"""
    raw = row.get("messages")
    if raw is None:
        st.caption("（无 messages 字段）")
        return
    if isinstance(raw, str):
        try:
            msgs = json.loads(raw)
        except Exception:
            st.code(raw)
            return
    else:
        msgs = raw
    if not isinstance(msgs, list):
        st.code(normalize_str(msgs))
        return

    images_raw = row.get("images", [])
    if isinstance(images_raw, str):
        try:
            images_raw = json.loads(images_raw)
        except Exception:
            images_raw = []

    _src = row.get("_source_file") or ""
    _jsonl_dir = os.path.dirname(_src) if _src and os.path.isfile(_src) else None

    img_map: Dict[int, str] = {}
    if isinstance(images_raw, list):
        for idx, p in enumerate(images_raw):
            if not p or not isinstance(p, str):
                continue
            if os.path.isabs(p) and os.path.exists(p):
                img_map[idx] = p
                continue
            if _jsonl_dir:
                for cand in (os.path.join(_jsonl_dir, "images", p),
                             os.path.join(_jsonl_dir, p)):
                    if os.path.exists(cand):
                        img_map[idx] = cand
                        break
            if idx not in img_map:
                img_map[idx] = p

    n_tool = sum(
        1 for m in msgs
        if m.get("role") == "assistant" and "<tool_call>" in str(m.get("content", ""))
    )
    st.caption(f"💬 {len(msgs)} 条消息，{n_tool} 次工具调用")

    turn = 0
    for msg in msgs:
        role    = msg.get("role", "?")
        content = msg.get("content", "")

        if role == "system":
            with st.expander("⚙️ System Prompt", expanded=False):
                st.markdown(normalize_str(content))
            continue

        if role == "user":
            if isinstance(content, list):
                question_parts = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "text":
                        clean = "\n".join(
                            ln for ln in item["text"].splitlines()
                            if not IMAGE_ID_RE.search(ln) and ln.strip() != "<image>"
                        ).strip()
                        if clean:
                            question_parts.append(clean)
                if 0 in img_map and os.path.exists(img_map[0]):
                    st.image(img_map[0], caption="原始图片", width=img_width)
                if question_parts:
                    with st.expander("❓ 用户问题", expanded=True):
                        st.markdown("\n\n".join(question_parts))
            else:
                content_str = normalize_str(content)
                image_ids   = [int(m.group(1)) for m in IMAGE_ID_RE.finditer(content_str)]
                tr_m   = re.search(r"<tool_response>(.*?)</tool_response>", content_str, re.DOTALL)
                tr_txt = tr_m.group(1).strip() if tr_m else ""
                if tr_txt:
                    st.markdown(f"🛠️ **Tool Response**: {tr_txt}")
                for iid in image_ids:
                    if iid in img_map and os.path.exists(img_map[iid]):
                        st.image(img_map[iid], caption=f"Zoom 图 (id={iid})", width=img_width)
            continue

        if role == "assistant":
            content_str  = normalize_str(content)
            is_tool_turn = "<tool_call>" in content_str
            if is_tool_turn:
                turn += 1
                st.markdown(f"**🤖 [第 {turn} 轮工具调用]**")
            else:
                st.markdown("**🤖 [最终回答]**")
            # 思考折叠
            think_m = THINK_RE.search(content_str)
            if think_m:
                with st.expander("💭 思考过程", expanded=False):
                    st.markdown(think_m.group(1).strip())
                # 去掉 think 块后显示剩余内容
                visible = THINK_RE.sub("", content_str).strip()
            else:
                visible = content_str
            st.code(visible, language=None)
            # 工具轮次展示 bbox
            if is_tool_turn:
                bboxes = extract_zoom_bboxes(content_str)
                if bboxes:
                    primary = get_primary_image_path(row, None)
                    if primary and os.path.exists(primary):
                        ann = draw_bbox_over_image(primary, bboxes)
                        if ann:
                            st.image(ann, caption=f"第 {turn} 轮 zoom 框图", width=img_width)
            continue

        st.markdown(f"**[{role}]**")
        st.code(normalize_str(content))


# ═══════════════════════════════════════════════════════════════════════════════
# 对比分析
# ═══════════════════════════════════════════════════════════════════════════════

def compute_diff(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """
    按 id 字段合并两份结果，计算每个阈值的改善/退步情况。
    返回 merged DataFrame，含 km_a, km_b, km_delta 及各阈值的 pass_a, pass_b, category。
    """
    def _to_km(df: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(df["km"], errors="coerce")

    id_col = "id"
    if id_col not in df_a.columns or id_col not in df_b.columns:
        st.error("两个文件都必须包含 'id' 字段才能合并对比。")
        st.stop()

    # 若有重复 id，保留最后一条
    df_a = df_a.drop_duplicates(subset=[id_col], keep="last")
    df_b = df_b.drop_duplicates(subset=[id_col], keep="last")

    # 以 id 为 key 合并
    merged = pd.merge(
        df_a.rename(columns=lambda c: f"{c}_a" if c != id_col else c),
        df_b.rename(columns=lambda c: f"{c}_b" if c != id_col else c),
        on=id_col,
        how="inner",
    )

    merged["km_a"] = pd.to_numeric(merged.get("km_a"), errors="coerce")
    merged["km_b"] = pd.to_numeric(merged.get("km_b"), errors="coerce")
    # km_delta: 正值 = B 更远（退步），负值 = B 更近（改善）
    merged["km_delta"] = merged["km_b"] - merged["km_a"]

    for t in THRESHOLDS:
        merged[f"pass_a_{t}"] = merged["km_a"] <= t
        merged[f"pass_b_{t}"] = merged["km_b"] <= t

    # 汇总文字标签：收集所有阈值下的变化
    def _label(row):
        tags = []
        for t in THRESHOLDS:
            pa = row.get(f"pass_a_{t}", False)
            pb = row.get(f"pass_b_{t}", False)
            if not pa and pb:
                tags.append(f"↑@{t}km")
            elif pa and not pb:
                tags.append(f"↓@{t}km")
        return " | ".join(tags) if tags else "—"

    merged["change_summary"] = merged.apply(_label, axis=1)
    return merged


def _category_mask(merged: pd.DataFrame, category: str) -> pd.Series:
    """根据分类返回 boolean mask。"""
    if category == "全部样本":
        return pd.Series([True] * len(merged), index=merged.index)
    if category == "有任何变化":
        return merged["change_summary"] != "—"

    # "@25km 改善": A 未过 25km，B 过了
    if category == "@25km 改善":
        return (~merged["pass_a_25"].fillna(False)) & (merged["pass_b_25"].fillna(False))
    if category == "@25km 退步":
        return (merged["pass_a_25"].fillna(False)) & (~merged["pass_b_25"].fillna(False))
    if category == "@2500km 改善":
        return (~merged["pass_a_2500"].fillna(False)) & (merged["pass_b_2500"].fillna(False))
    if category == "@2500km 退步":
        return (merged["pass_a_2500"].fillna(False)) & (~merged["pass_b_2500"].fillna(False))
    if category == "@200km 改善":
        return (~merged["pass_a_200"].fillna(False)) & (merged["pass_b_200"].fillna(False))
    if category == "@200km 退步":
        return (merged["pass_a_200"].fillna(False)) & (~merged["pass_b_200"].fillna(False))
    if category == "@750km 改善":
        return (~merged["pass_a_750"].fillna(False)) & (merged["pass_b_750"].fillna(False))
    if category == "@750km 退步":
        return (merged["pass_a_750"].fillna(False)) & (~merged["pass_b_750"].fillna(False))
    if category == "B 无预测 (km=None)":
        return merged["km_b"].isna()
    if category == "A 无预测 (km=None)":
        return merged["km_a"].isna()

    return pd.Series([True] * len(merged), index=merged.index)


def acc_line(km_series: pd.Series, n_total: int) -> str:
    parts = []
    for t in THRESHOLDS:
        n = int((km_series <= t).sum())
        parts.append(f"@{t}km={n/n_total:.3f}({n}/{n_total})")
    return "  ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# 主界面
# ═══════════════════════════════════════════════════════════════════════════════

st.title("🔍 实验对比查看器")

with st.sidebar:
    st.subheader("📂 数据源")
    file_a = st.text_input("File A (基准)", value=DEFAULT_FILE_A)
    label_a = st.text_input("File A 标签", value="v1 (zoom-only)")
    file_b = st.text_input("File B (对比)", value=DEFAULT_FILE_B)
    label_b = st.text_input("File B 标签", value="v2 (zoom-only v2)")

    dataset_name = st.selectbox("数据集", options=list(DATASET_CONFIGS.keys()), index=0)
    dataset_cfg  = DATASET_CONFIGS[dataset_name]

    reload_btn = st.button("🔄 重新加载")
    if reload_btn:
        load_jsonl.clear()

    st.subheader("🔎 分类筛选")
    CATEGORIES = [
        "全部样本",
        "有任何变化",
        "@25km 改善",
        "@25km 退步",
        "@200km 改善",
        "@200km 退步",
        "@750km 改善",
        "@750km 退步",
        "@2500km 改善",
        "@2500km 退步",
        "B 无预测 (km=None)",
        "A 无预测 (km=None)",
    ]
    category = st.selectbox("样本分类", options=CATEGORIES, index=0)

    sort_by = st.selectbox("排序依据", options=["km_delta", "km_a", "km_b", "id"], index=0)
    sort_asc = st.checkbox("升序", value=False)

# ── 加载文件 ────────────────────────────────────────────────────────────────────
for path, label in [(file_a, "File A"), (file_b, "File B")]:
    if not os.path.exists(path):
        st.error(f"{label} 文件不存在：{path}")
        st.stop()

with st.spinner("加载数据..."):
    df_a = load_jsonl(file_a)
    df_b = load_jsonl(file_b)

st.caption(f"File A: {file_a}  →  {len(df_a)} 条")
st.caption(f"File B: {file_b}  →  {len(df_b)} 条")

# ── 合并计算 ────────────────────────────────────────────────────────────────────
merged = compute_diff(df_a, df_b)
n_common = len(merged)

# ── 总体指标对比 ────────────────────────────────────────────────────────────────
st.subheader("📊 总体指标对比")

km_a_all = merged["km_a"].dropna()
km_b_all = merged["km_b"].dropna()
n_a  = len(km_a_all)
n_b  = len(km_b_all)

col1, col2, col3 = st.columns(3)
col1.metric("共同样本数", n_common)
col2.metric(f"{label_a} 有预测", n_a)
col3.metric(f"{label_b} 有预测", n_b)

metric_data = []
for t in THRESHOLDS:
    acc_a = int((km_a_all <= t).sum()) / n_common if n_common else 0
    acc_b = int((km_b_all <= t).sum()) / n_common if n_common else 0
    delta = acc_b - acc_a
    metric_data.append({"阈值": f"@{t}km",
                        f"File A ({label_a})": f"{acc_a:.3f}",
                        f"File B ({label_b})": f"{acc_b:.3f}",
                        "Δ(B-A)": f"{delta:+.3f}"})

st.dataframe(pd.DataFrame(metric_data).set_index("阈值"), use_container_width=True)

# ── 样本变化汇总表 ──────────────────────────────────────────────────────────────
st.subheader("📋 样本变化统计")

change_counts = []
for cat in CATEGORIES[2:]:  # 跳过"全部"和"有任何变化"
    mask = _category_mask(merged, cat)
    change_counts.append({"分类": cat, "样本数": int(mask.sum())})

st.dataframe(pd.DataFrame(change_counts).set_index("分类"), use_container_width=True)

# ── 筛选样本列表 ────────────────────────────────────────────────────────────────
st.subheader(f"🗂️ 样本列表 — {category}")

mask = _category_mask(merged, category)
view = merged[mask].copy()
if sort_by in view.columns:
    view = view.sort_values(by=sort_by, ascending=sort_asc, na_position="last")

# 显示摘要列
display_cols = ["id", "km_a", "km_b", "km_delta", "change_summary"]
for col in display_cols:
    if col not in view.columns:
        display_cols.remove(col)

# 加入 gt_country / gt_city（优先从 A）
for gc in ("gt_country_a", "gt_city_a"):
    if gc in view.columns:
        display_cols.append(gc)

st.dataframe(
    view[display_cols].rename(columns={
        "km_a": f"km_A",
        "km_b": f"km_B",
        "km_delta": "Δkm(B-A)",
        "change_summary": "变化",
        "gt_country_a": "国家",
        "gt_city_a": "城市",
    }),
    use_container_width=True,
    height=300,
)
st.caption(f"共 {len(view)} 条（共同样本 {n_common} 条）")

# ── 详情面板 ────────────────────────────────────────────────────────────────────
st.subheader("🔬 逐样本对比")

if view.empty:
    st.info("当前分类下无样本。")
    st.stop()

# session state 导航
if "cmp_pos" not in st.session_state:
    st.session_state.cmp_pos = 0
if st.session_state.cmp_pos >= len(view):
    st.session_state.cmp_pos = 0

view_ids = view["id"].tolist()

with st.sidebar:
    st.subheader("🧭 导航")
    nav_c1, nav_c2 = st.columns(2)
    prev_btn = nav_c1.button("◀ 上一条")
    next_btn = nav_c2.button("▶ 下一条")
    if prev_btn:
        st.session_state.cmp_pos = (st.session_state.cmp_pos - 1) % len(view)
    if next_btn:
        st.session_state.cmp_pos = (st.session_state.cmp_pos + 1) % len(view)

    # 样本选择器
    sel_id = st.selectbox(
        "选择样本 ID",
        options=view_ids,
        index=st.session_state.cmp_pos,
    )
    if sel_id in view_ids:
        st.session_state.cmp_pos = view_ids.index(sel_id)

    st.caption(f"进度：{st.session_state.cmp_pos + 1} / {len(view)}")

current_id = view_ids[st.session_state.cmp_pos]
row_merged = view[view["id"] == current_id].iloc[0]

# 从原始 df 取回完整 row
row_a_list = df_a[df_a["id"] == current_id]
row_b_list = df_b[df_b["id"] == current_id]
row_a = row_a_list.iloc[-1] if not row_a_list.empty else None
row_b = row_b_list.iloc[-1] if not row_b_list.empty else None

# ── 基本信息 ─────────────────────────────────────────────────────────────────
km_a_val = row_merged.get("km_a")
km_b_val = row_merged.get("km_b")
gt_lat   = row_merged.get("gt_lat_a") or row_merged.get("gt_lat_b")
gt_lon   = row_merged.get("gt_lon_a") or row_merged.get("gt_lon_b")
gt_country = row_merged.get("gt_country_a") or row_merged.get("gt_country_b")
gt_city    = row_merged.get("gt_city_a")    or row_merged.get("gt_city_b")

info_cols = st.columns(4)
info_cols[0].metric("样本 ID", current_id)
info_cols[1].metric("GT 坐标", f"({gt_lat:.4f}, {gt_lon:.4f})" if gt_lat is not None else "N/A")
info_cols[2].metric("国家/城市", f"{gt_country or '?'} / {gt_city or '?'}")
info_cols[3].metric("变化", row_merged.get("change_summary", "—"))

km_cols = st.columns(3)
km_cols[0].metric(f"km_A ({label_a})",
                  f"{km_a_val:.1f} km" if pd.notna(km_a_val) else "None (无预测)")
km_cols[1].metric(f"km_B ({label_b})",
                  f"{km_b_val:.1f} km" if pd.notna(km_b_val) else "None (无预测)")
delta_val = row_merged.get("km_delta")
km_cols[2].metric("Δ = B - A",
                  f"{delta_val:+.1f} km" if pd.notna(delta_val) else "N/A",
                  delta_color="inverse" if pd.notna(delta_val) else "off")

# ── 原始图片 ─────────────────────────────────────────────────────────────────
img_path = None
if row_a is not None:
    img_path = get_primary_image_path(row_a, dataset_cfg)
if img_path and os.path.exists(img_path):
    st.image(img_path, caption=f"原始图片 (id={current_id})", width=400)

# ── 左右对比 ─────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.markdown(f"### 🅰️ {label_a}")
    pred_lat_a = row_merged.get("pred_lat_a")
    pred_lon_a = row_merged.get("pred_lon_a")
    n_tc_a     = row_merged.get("n_tool_calls_a")
    st.caption(
        f"km={km_a_val:.1f}" if pd.notna(km_a_val) else "km=None"
        + (f"  |  pred=({pred_lat_a:.4f},{pred_lon_a:.4f})" if pd.notna(pred_lat_a) else "  |  pred=None")
        + (f"  |  tool_calls={int(n_tc_a)}" if pd.notna(n_tc_a) else "")
    )
    if row_a is not None:
        if "messages" in row_a and row_a["messages"] is not None:
            render_messages(row_a, img_width=280)
        elif "response" in row_a:
            st.code(normalize_str(row_a.get("response")), language=None)
        else:
            st.info("无 messages / response 字段")
    else:
        st.warning("File A 中无此样本")

with col_right:
    st.markdown(f"### 🅱️ {label_b}")
    pred_lat_b = row_merged.get("pred_lat_b")
    pred_lon_b = row_merged.get("pred_lon_b")
    n_tc_b     = row_merged.get("n_tool_calls_b")
    st.caption(
        f"km={km_b_val:.1f}" if pd.notna(km_b_val) else "km=None"
        + (f"  |  pred=({pred_lat_b:.4f},{pred_lon_b:.4f})" if pd.notna(pred_lat_b) else "  |  pred=None")
        + (f"  |  tool_calls={int(n_tc_b)}" if pd.notna(n_tc_b) else "")
    )
    if row_b is not None:
        if "messages" in row_b and row_b["messages"] is not None:
            render_messages(row_b, img_width=280)
        elif "response" in row_b:
            st.code(normalize_str(row_b.get("response")), language=None)
        else:
            st.info("无 messages / response 字段")
    else:
        st.warning("File B 中无此样本")
