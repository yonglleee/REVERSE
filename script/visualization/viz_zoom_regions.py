"""
Zoom Region 标注可视化工具

展示 step1_bboxes_v2.jsonl 和 step2_verified_v2_*.jsonl 的标注结果：
  - 原图 + bbox 框
  - 每个 bbox 的 label / reason / gain / is_useful
  - 235B 预测 vs GT 的距离（pred_km）
  - 可按 has_zoom_region / has_useful_region / pred_reward 筛选

Usage:
    streamlit run /mnt/sh/mmvision/home/jonahli/projects/tusou/visualization/viz_zoom_regions.py
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

# ── 默认文件路径 ────────────────────────────────────────────────────────────────
DEFAULT_STEP1 = "/mnt/sh/mmvision/home/jonahli/data_agent/rl/SpotAgent/zoom_regions/step1_bboxes_v2.jsonl"
DEFAULT_STEP2_4B = "/mnt/sh/mmvision/home/jonahli/data_agent/rl/SpotAgent/zoom_regions/step2_verified_v2_4b.jsonl"
DEFAULT_STEP2_8B = "/mnt/sh/mmvision/home/jonahli/data_agent/rl/SpotAgent/zoom_regions/step2_verified_v2_8b.jsonl"

BBOX_COLORS = ["#FF4444", "#44BB44", "#4488FF", "#FF8800"]


# ── 数据加载 ────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
    return entries


# ── bbox 绘制 ──────────────────────────────────────────────────────────────────

def draw_bboxes(img: Image.Image, regions: List[Dict], mode: str = "step1") -> Image.Image:
    """
    Draw bboxes on image. Coords normalized to [0, 1000].
    mode: 'step1' uses zoom_regions, 'step2' uses verified_regions with gain info.
    """
    img = img.copy().convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    for i, region in enumerate(regions):
        bbox = region.get("bbox", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        # denormalize from [0,1000] to pixel coords
        px1 = int(x1 / 1000 * w)
        py1 = int(y1 / 1000 * h)
        px2 = int(x2 / 1000 * w)
        py2 = int(y2 / 1000 * h)

        color = BBOX_COLORS[i % len(BBOX_COLORS)]
        if mode == "step2":
            is_useful = region.get("is_useful", None)
            if is_useful is True:
                color = "#00FF00"
            elif is_useful is False:
                color = "#FF4444"

        draw.rectangle([px1, py1, px2, py2], outline=color, width=3)
        label = region.get("label", f"region {i}")
        draw.text((px1 + 3, py1 + 3), f"{i+1}. {label}", fill=color)

    return img


def crop_region(img: Image.Image, bbox: List[float]) -> Image.Image:
    """Crop a bbox region (coords in [0,1000]) from image."""
    w, h = img.size
    x1, y1, x2, y2 = bbox
    px1 = int(x1 / 1000 * w)
    py1 = int(y1 / 1000 * h)
    px2 = int(x2 / 1000 * w)
    py2 = int(y2 / 1000 * h)
    px1, px2 = max(0, px1), min(w, px2)
    py1, py2 = max(0, py1), min(h, py2)
    return img.crop((px1, py1, px2, py2))


# ── Streamlit UI ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Zoom Region 可视化", layout="wide")
st.title("🔍 Zoom Region 标注可视化")

# ── 侧边栏：文件选择和筛选 ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("数据文件")
    data_source = st.selectbox("数据来源", ["Step 1 (bbox 标注)", "Step 2 - 4B (增益验证)", "Step 2 - 8B (增益验证)"])

    if data_source == "Step 1 (bbox 标注)":
        jsonl_path = st.text_input("JSONL 路径", DEFAULT_STEP1)
        mode = "step1"
    elif data_source == "Step 2 - 4B (增益验证)":
        jsonl_path = st.text_input("JSONL 路径", DEFAULT_STEP2_4B)
        mode = "step2"
    else:
        jsonl_path = st.text_input("JSONL 路径", DEFAULT_STEP2_8B)
        mode = "step2"

    st.divider()
    st.header("筛选")
    filter_has_zoom = st.checkbox("只看有 zoom_regions 的样本", value=True)
    if mode == "step2":
        filter_useful = st.checkbox("只看 has_useful_region=True", value=False)
        filter_gain_pos = st.checkbox("只看至少一个 gain>0 的 region", value=False)
    filter_pred_km = st.slider("pred_km 最大值（235B 预测误差）", 0, 20000, 20000, step=100)

    st.divider()
    st.header("显示设置")
    show_crops = st.checkbox("显示各 bbox crop", value=True)
    max_display = st.slider("最多显示样本数", 10, 200, 50)

# ── 加载数据 ───────────────────────────────────────────────────────────────────
if not os.path.exists(jsonl_path):
    st.error(f"文件不存在: {jsonl_path}")
    st.stop()

with st.spinner("加载数据..."):
    entries = load_jsonl(jsonl_path)

st.caption(f"共加载 {len(entries)} 条记录 from `{jsonl_path}`")

# ── 聚合统计 ───────────────────────────────────────────────────────────────────
total = len(entries)
has_zoom = sum(1 for e in entries if e.get("zoom_regions") or e.get("verified_regions"))
if mode == "step2":
    useful = sum(1 for e in entries if e.get("has_useful_region"))
    gains = [r.get("gain") for e in entries for r in e.get("verified_regions", []) if r.get("gain") is not None]
    pos_gains = [g for g in gains if g > 0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("总样本", total)
col2.metric("有 zoom_regions", f"{has_zoom} ({100*has_zoom/total:.1f}%)")
if mode == "step2":
    col3.metric("has_useful_region", f"{useful} ({100*useful/total:.1f}%)")
    col4.metric("正增益 region", f"{len(pos_gains)}/{len(gains)} ({100*len(pos_gains)/max(len(gains),1):.1f}%)")
elif mode == "step1":
    pred_kms = [e.get("pred_km") for e in entries if e.get("pred_km") is not None]
    if pred_kms:
        col3.metric("235B mean pred_km", f"{sum(pred_kms)/len(pred_kms):.0f} km")
        col4.metric("235B pred_km < 25km", f"{sum(1 for k in pred_kms if k < 25)} ({100*sum(1 for k in pred_kms if k < 25)/len(pred_kms):.1f}%)")

st.divider()

# ── 筛选 ──────────────────────────────────────────────────────────────────────
filtered = entries
if filter_has_zoom:
    filtered = [e for e in filtered if e.get("zoom_regions") or e.get("verified_regions")]
if mode == "step2":
    if filter_useful:
        filtered = [e for e in filtered if e.get("has_useful_region")]
    if filter_gain_pos:
        filtered = [e for e in filtered if any(r.get("gain", 0) > 0 for r in e.get("verified_regions", []))]
if filter_pred_km < 20000:
    filtered = [e for e in filtered if (e.get("pred_km") or 99999) <= filter_pred_km]

st.caption(f"筛选后：{len(filtered)} 条 | 显示前 {min(max_display, len(filtered))} 条")
filtered = filtered[:max_display]

# ── 按样本展示 ────────────────────────────────────────────────────────────────
for idx, entry in enumerate(filtered):
    img_path = entry.get("image_path", "")
    gt = entry.get("ground_truth", "")
    pred_km = entry.get("pred_km")
    pred_reward = entry.get("pred_reward")
    pred_city = entry.get("pred_city", "")
    pred_country = entry.get("pred_country", "")

    regions = entry.get("verified_regions") if mode == "step2" else entry.get("zoom_regions", [])
    has_useful = entry.get("has_useful_region", False)

    title = f"**#{entry.get('index', idx)}** — GT: `{gt}`"
    if pred_km is not None:
        title += f" | 235B pred: {pred_country}/{pred_city} ({pred_km:.0f} km, reward={pred_reward:.2f})"
    if mode == "step2":
        title += f" | {'✅ useful' if has_useful else '❌ no gain'}"

    with st.expander(title, expanded=(idx < 3)):
        if not os.path.exists(img_path):
            st.warning(f"图片不存在: {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")

        if regions:
            img_with_boxes = draw_bboxes(img, regions, mode=mode)
            st.image(img_with_boxes, caption=f"原图 + bbox（共 {len(regions)} 个）", use_container_width=True)
        else:
            st.image(img, caption="原图（无 zoom_regions）", use_container_width=True)

        # ── Region 详情 + crop ─────────────────────────────────────────────────
        if regions:
            st.markdown("**Region 详情：**")
            for i, region in enumerate(regions):
                color = BBOX_COLORS[i % len(BBOX_COLORS)]
                label = region.get("label", "?")
                reason = region.get("reason", "")
                bbox = region.get("bbox", [])

                if mode == "step2":
                    is_useful = region.get("is_useful", None)
                    gain = region.get("gain")
                    r_orig = region.get("reward_original")
                    r_zoom = region.get("reward_with_zoom")
                    useful_tag = "✅" if is_useful else "❌"
                    gain_str = f"gain={gain:.4f}" if gain is not None else "gain=N/A"
                    reward_str = f"reward: {r_orig:.3f} → {r_zoom:.3f}" if (r_orig is not None and r_zoom is not None) else ""
                    header = f"{useful_tag} **{i+1}. {label}** | {gain_str} | {reward_str}"
                else:
                    header = f"**{i+1}. {label}**"

                st.markdown(f"<span style='color:{color}'>{header}</span>", unsafe_allow_html=True)
                st.caption(f"bbox: {bbox} | {reason}")

                if show_crops and len(bbox) == 4:
                    crop = crop_region(img, bbox)
                    if crop.size[0] > 0 and crop.size[1] > 0:
                        st.image(crop, caption=f"Crop {i+1}: {label}", width=300)

        st.divider()
