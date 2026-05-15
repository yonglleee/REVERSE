import json
import math
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw


DEFAULT_JSONL = "/mnt/sh/mmvision/home/jonahli/save/tusou/eval/tool_combo_50_v5/combo_zoom.jsonl"
PRIORITY_FIELDS = ["messages", "input", "output", "gts", "reward", "score", "step"]
TOOL_CALL_RE   = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
THINK_RE       = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_RE_V    = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
IMAGE_ID_RE    = re.compile(r"image_id:\s*(\d+)")

# ── Dataset configs ────────────────────────────────────────────────────────────
DATASET_CONFIGS = {
    "自动检测": {"img_dir": None, "id_field": None},
    "im2gps3k": {
        "img_dir": "/mnt/sh/mmvision/home/jonahli/data_agent/benchmark/im2gps3ktest",
        "id_field": "id",
    },
}

st.set_page_config(page_title="JSONL 可视化", layout="wide")
st.markdown(
    """
    <style>
    .stCodeBlock pre,
    .stCode pre,
    div[data-testid="stCodeBlock"] pre,
    div[data-testid="stCode"] pre,
    div[data-testid="stCodeBlock"] code,
    div[data-testid="stCode"] code {
        white-space: pre-wrap !important;
        overflow-wrap: anywhere !important;
        word-break: break-word !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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
                rec["_source_file"] = path  # 注入来源路径，供图片相对路径解析使用
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


def render_wrapped_code(text: str) -> None:
    st.code(text)


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
    image_paths: List[str] = []
    if "image_paths" in row:
        image_paths.extend(to_image_path_list(row.get("image_paths")))
    if not image_paths and "image_path" in row:
        image_paths.extend(to_image_path_list(row.get("image_path")))
    # combo format: images list field (index 0 = original image)
    if not image_paths and "images" in row:
        image_paths.extend(to_image_path_list(row.get("images")))

    image_paths = [p for p in image_paths if p]
    if image_paths:
        return image_paths[0]

    # Dataset-based fallback: look up image by id field
    if dataset_cfg and dataset_cfg.get("img_dir") and dataset_cfg.get("id_field"):
        img_dir = dataset_cfg["img_dir"]
        id_field = dataset_cfg["id_field"]
        img_id = row.get(id_field)
        if img_id is not None:
            img_id_str = str(int(img_id)) if isinstance(img_id, float) else str(img_id)
            # im2gps3k filenames: {id}_*.jpg  (e.g. 12345_abc.jpg)
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
    if not output_text:
        return []
    bboxes: List[Dict[str, Any]] = []
    for payload in TOOL_CALL_RE.findall(output_text):
        try:
            tool_call = json.loads(payload)
        except Exception:
            continue
        if tool_call.get("name") != "image_zoom_in_tool":
            continue
        args = tool_call.get("arguments", {}) or {}
        bbox = args.get("bbox_2d")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except Exception:
            continue
        bboxes.append(
            {
                "bbox_2d": [x1, y1, x2, y2],
                "label": normalize_str(args.get("label", "")).strip(),
            }
        )
    return bboxes


def draw_bbox_over_image(image_path: str, bboxes: List[Dict[str, Any]]) -> Optional[Image.Image]:
    if not bboxes or image_path.startswith("http://") or image_path.startswith("https://"):
        return None
    if not os.path.exists(image_path):
        return None

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    line_w = max(2, min(width, height) // 250)

    for idx, item in enumerate(bboxes, start=1):
        x1, y1, x2, y2 = item["bbox_2d"]
        max_v = max(abs(x1), abs(y1), abs(x2), abs(y2))

        if max_v <= 1.5:
            left = int(round(x1 * width))
            top = int(round(y1 * height))
            right = int(round(x2 * width))
            bottom = int(round(y2 * height))
        elif max_v <= 1000.0:
            left = int(round(x1 / 1000.0 * width))
            top = int(round(y1 / 1000.0 * height))
            right = int(round(x2 / 1000.0 * width))
            bottom = int(round(y2 / 1000.0 * height))
        else:
            left = int(round(x1))
            top = int(round(y1))
            right = int(round(x2))
            bottom = int(round(y2))

        left = max(0, min(width, left))
        top = max(0, min(height, top))
        right = max(0, min(width, right))
        bottom = max(0, min(height, bottom))
        if right <= left or bottom <= top:
            continue

        draw.rectangle([(left, top), (right, bottom)], outline=(255, 0, 0), width=line_w)
        label = item.get("label") or f"bbox_{idx}"
        draw.text((left + 4, max(0, top - 16)), label, fill=(255, 0, 0))

    return image

def render_image_preview(row: pd.Series, dataset_cfg: Optional[Dict] = None) -> None:
    # messages 格式由 render_messages 自己显示图片，不重复渲染
    if "messages" in row and row.get("messages") is not None:
        return

    image_paths: List[str] = []
    if "image_paths" in row:
        image_paths.extend(to_image_path_list(row.get("image_paths")))
    if not image_paths and "image_path" in row:
        image_paths.extend(to_image_path_list(row.get("image_path")))
    if not image_paths and "images" in row:
        image_paths.extend(to_image_path_list(row.get("images")))

    image_paths = [p for p in image_paths if p]

    # Dataset-based fallback
    if not image_paths and dataset_cfg and dataset_cfg.get("img_dir") and dataset_cfg.get("id_field"):
        path = get_primary_image_path(row, dataset_cfg)
        if path:
            image_paths = [path]

    if not image_paths:
        return

    # Try both "output" and "response" fields for tool calls
    output_text = normalize_str(row.get("output", "") or row.get("response", ""))
    bboxes = extract_zoom_bboxes(output_text)

    st.markdown("#### 图片预览")
    st.image(image_paths[0], caption=image_paths[0], width=320)

    if bboxes:
        annotated = draw_bbox_over_image(image_paths[0], bboxes)
        if annotated is not None:
            st.image(annotated, caption=f"image_zoom_in_tool bbox 覆盖图（{len(bboxes)}个）", width=320)
            st.caption("bbox 坐标按自适应规则解析：优先 [0,1]，其次 [0,1000]，否则按像素坐标。")
        else:
            st.caption("检测到 bbox，但当前图片无法本地绘制（可能是远程 URL 或路径不存在）。")



def split_output_rounds(text: str) -> List[str]:
    if not text:
        return []
    normalized = text.replace("</tool_call>user", "</tool_call>\nuser").replace(
        "</tool_response>assistant", "</tool_response>\nassistant"
    )
    if "<tool_call>" not in normalized:
        return []

    raw_parts = normalized.split("<tool_call>")
    rounds: List[str] = []
    preface = raw_parts[0].strip()
    if preface:
        rounds.append(preface)

    for part in raw_parts[1:]:
        chunk = f"<tool_call>{part}".strip()
        if chunk:
            rounds.append(chunk)
    return rounds


def render_rich_text(label: str, value: Any, row: Optional[pd.Series] = None, dataset_cfg: Optional[Dict] = None) -> None:
    text = normalize_str(value)
    st.markdown(f"#### {label}")
    if label in ("output", "response") and "<tool_call>" in text:
        rounds = split_output_rounds(text)
        if len(rounds) > 1:
            tool_rounds = max(1, len(rounds) - 1)
            st.caption(f"工具轮次：{tool_rounds}")
            current_image_path = get_primary_image_path(row, dataset_cfg) if row is not None else None
            for idx, segment in enumerate(rounds, start=1):
                is_preface = idx == 1 and not segment.startswith("<tool_call>")
                round_no: Optional[int] = None
                if is_preface:
                    st.markdown("**[前置分析]**")
                else:
                    round_no = idx if rounds[0].startswith("<tool_call>") else idx - 1
                    st.markdown(f"**[第 {round_no} 轮]**")
                render_wrapped_code(segment)

                if is_preface or not current_image_path:
                    continue
                round_bboxes = extract_zoom_bboxes(segment)
                if not round_bboxes:
                    continue
                annotated = draw_bbox_over_image(current_image_path, round_bboxes)
                round_display = round_no if round_no is not None else idx
                if annotated is not None:
                    st.image(annotated, caption=f"第 {round_display} 轮 zoom 框图", width=320)
                else:
                    st.caption(f"第 {round_display} 轮检测到 bbox，但当前图片无法本地绘制。")
            return

    render_wrapped_code(text)


def _render_think_block(text: str) -> None:
    """折叠展示 <think>...</think> 内容。"""
    m = THINK_RE.search(text)
    if m:
        think_content = m.group(1).strip()
        with st.expander("💭 思考过程", expanded=False):
            st.markdown(think_content)


def _render_assistant_text(text: str) -> None:
    """渲染 assistant 消息：直接显示原始内容（含所有标签）。"""
    st.code(text, language=None)


def render_messages(row: pd.Series) -> None:
    """渲染 messages 字段（完整多轮对话）。

    格式：messages 是 list of {role, content}，images 是路径列表，
    user 消息中 image_id: N 对应 images[N]。
    """
    raw = row.get("messages")
    if raw is None:
        return

    # 解析 messages
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

    # 解析 images 列表
    images_raw = row.get("images", [])
    if isinstance(images_raw, str):
        try:
            images_raw = json.loads(images_raw)
        except Exception:
            images_raw = []

    # 尝试从 row 反推 JSONL 所在目录，用于拼接相对路径图片
    _jsonl_dir: Optional[str] = None
    try:
        _src = row.get("_source_file") or ""
        if _src and os.path.isfile(_src):
            _jsonl_dir = os.path.dirname(_src)
    except Exception:
        pass

    img_map: dict[int, str] = {}  # image_id -> path
    if isinstance(images_raw, list):
        for idx, p in enumerate(images_raw):
            if not p or not isinstance(p, str):
                continue
            # 如果已经是绝对路径且存在，直接用
            if os.path.isabs(p) and os.path.exists(p):
                img_map[idx] = p
                continue
            # 尝试裸文件名 → 同 JSONL 目录下的 images/ 子目录
            if _jsonl_dir:
                candidate = os.path.join(_jsonl_dir, "images", p)
                if os.path.exists(candidate):
                    img_map[idx] = candidate
                    continue
                # 也尝试直接拼同级目录
                candidate2 = os.path.join(_jsonl_dir, p)
                if os.path.exists(candidate2):
                    img_map[idx] = candidate2
                    continue
            # fallback: 原始值（可能本身就是完整路径）
            img_map[idx] = p

    # 统计 tool 轮次
    n_tool = sum(
        1 for m in msgs
        if m.get("role") == "assistant" and "<tool_call>" in str(m.get("content", ""))
    )
    st.markdown(f"#### 💬 多轮对话（{len(msgs)} 条消息，{n_tool} 次工具调用）")

    turn = 0  # assistant turn counter
    for i, msg in enumerate(msgs):
        role = msg.get("role", "?")
        content = msg.get("content", "")

        # ── system ──────────────────────────────────────────────────────────
        if role == "system":
            with st.expander("⚙️ System Prompt", expanded=False):
                st.markdown(normalize_str(content))
            continue

        # ── user ────────────────────────────────────────────────────────────
        if role == "user":
            # content 可能是 list（第一轮原图）或 str（tool_response）
            if isinstance(content, list):
                # 第一轮 user：提取 image_id 和 question
                image_ids = []
                question_parts = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    t = item.get("type", "")
                    txt = item.get("text", "")
                    if t == "text":
                        # 检查是否含 image_id
                        for m2 in IMAGE_ID_RE.finditer(txt):
                            image_ids.append(int(m2.group(1)))
                        # 去掉 image_id 行保留 question
                        clean = "\n".join(
                            ln for ln in txt.splitlines()
                            if not IMAGE_ID_RE.search(ln) and ln.strip() not in ("<image>",)
                        ).strip()
                        if clean:
                            question_parts.append(clean)
                    elif t in ("image_url", "image"):
                        pass  # 图片由 images 字段处理

                # 展示原图（image_id: 0）
                if 0 in img_map and os.path.exists(img_map[0]):
                    st.image(img_map[0], caption="原始图片", width=360)
                elif image_ids:
                    for iid in image_ids:
                        if iid in img_map and os.path.exists(img_map[iid]):
                            st.image(img_map[iid], caption=f"image_id={iid}", width=360)

                if question_parts:
                    with st.expander("❓ 用户问题", expanded=True):
                        st.markdown("\n\n".join(question_parts))

            else:
                # tool_response 消息（含 image_id: N + <tool_response>）
                content_str = normalize_str(content)
                image_ids = [int(m2.group(1)) for m2 in IMAGE_ID_RE.finditer(content_str)]

                # 提取 tool_response 文本
                tr_m = re.search(r"<tool_response>(.*?)</tool_response>", content_str, re.DOTALL)
                tr_text = tr_m.group(1).strip() if tr_m else ""

                cols_left, cols_right = st.columns([1, 1])
                with cols_left:
                    if tr_text:
                        st.markdown("🛠️ **Tool Response**")
                        st.code(tr_text, language=None)
                    for iid in image_ids:
                        if iid in img_map and os.path.exists(img_map[iid]):
                            st.image(img_map[iid], caption=f"Zoom 图 (image_id={iid})", width=320)
                # 无需 cols_right，保持对称留白
            continue

        # ── assistant ───────────────────────────────────────────────────────
        if role == "assistant":
            content_str = normalize_str(content)
            is_tool_turn = "<tool_call>" in content_str

            if is_tool_turn:
                turn += 1
                st.markdown(f"---\n**🤖 Assistant（第 {turn} 轮工具调用）**")
            else:
                st.markdown("---\n**🤖 Assistant（最终回答）**")

            _render_assistant_text(content_str)
            continue

        # ── 其他 role（fallback）────────────────────────────────────────────
        st.markdown(f"**[{role}]**")
        st.code(normalize_str(content))


def add_search_filter(df: pd.DataFrame, query: str) -> pd.DataFrame:
    if not query:
        return df
    query = query.strip().lower()
    if not query:
        return df
    mask = []
    for _, row in df.iterrows():
        combined = " ".join(normalize_str(v).lower() for v in row.values)
        mask.append(query in combined)
    return df[pd.Series(mask, index=df.index)]


st.title("JSONL 可视化查看器")

with st.sidebar:
    st.subheader("数据源")
    jsonl_path = st.text_input("JSONL 路径", value=DEFAULT_JSONL)
    dataset_name = st.selectbox("数据集", options=list(DATASET_CONFIGS.keys()), index=0)
    dataset_cfg = DATASET_CONFIGS[dataset_name]
    reload_clicked = st.button("重新加载")

if reload_clicked:
    clear_cache = getattr(load_jsonl, "clear", None)
    if callable(clear_cache):
        clear_cache()

if not os.path.exists(jsonl_path):
    st.error("文件不存在，请检查路径。")
    st.stop()

with st.spinner("加载数据中..."):
    df = load_jsonl(jsonl_path)

st.caption(f"当前文件：{jsonl_path} · 总行数：{len(df)}")

if df.empty:
    st.warning("数据为空或无法解析。")
    st.stop()

all_columns = list(df.columns)
priority_cols = [c for c in PRIORITY_FIELDS if c in all_columns]
other_cols = [c for c in all_columns if c not in priority_cols]

with st.sidebar:
    st.subheader("表格设置")
    visible_cols = st.multiselect(
        "显示字段",
        options=priority_cols + other_cols,
        default=priority_cols,
    )
    search_query = st.text_input("全文搜索")
    sort_field = st.selectbox("排序字段", options=["(无)"] + all_columns)
    sort_order = st.selectbox("排序顺序", options=["降序", "升序"])

    st.subheader("筛选条件")
    numeric_cols = [c for c in all_columns if pd.api.types.is_numeric_dtype(df[c])]
    filter_col = st.selectbox("数值字段", options=["(无)"] + numeric_cols)
    min_val = None
    max_val = None
    if filter_col != "(无)":
        col_min = float(pd.to_numeric(df[filter_col], errors="coerce").min())
        col_max = float(pd.to_numeric(df[filter_col], errors="coerce").max())
        if col_min < col_max:
            min_val, max_val = st.slider(
                "数值范围",
                min_value=col_min,
                max_value=col_max,
                value=(col_min, col_max),
            )
        else:
            min_val, max_val = col_min, col_max
            st.caption(f"该字段只有单一取值：{col_min}")

    st.subheader("分页")
    page_size = st.selectbox("每页条数", options=[20, 50, 100, 200], index=1)
    page_num = st.number_input("页码", min_value=1, value=1, step=1)

    st.subheader("详情导航")
    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        prev_clicked = st.button("上一条")
    with nav_col2:
        next_clicked = st.button("下一条")

filtered_df = df.copy()
if filter_col != "(无)" and min_val is not None:
    numeric_series = pd.to_numeric(filtered_df[filter_col], errors="coerce")
    filtered_df = filtered_df[(numeric_series >= min_val) & (numeric_series <= max_val)]

filtered_df = add_search_filter(filtered_df, search_query)

if sort_field != "(无)":
    ascending = sort_order == "升序"
    filtered_df = filtered_df.sort_values(by=sort_field, ascending=ascending, na_position="last")

st.subheader("数据表")
if not visible_cols:
    st.info("请在左侧选择需要显示的字段。")
else:
    total_pages = max(1, math.ceil(len(filtered_df) / page_size))
    page_num = min(page_num, total_pages)
    start = (page_num - 1) * page_size
    end = start + page_size
    page_df = filtered_df.iloc[start:end]
    # pyarrow cannot serialize mixed list/non-list columns (e.g. "messages", "images")
    # — stringify any column that contains list or dict values before rendering
    display_df = page_df[visible_cols].copy()
    for col in display_df.columns:
        if display_df[col].apply(lambda x: isinstance(x, (list, dict))).any():
            display_df[col] = display_df[col].apply(
                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x
            )
    st.dataframe(display_df, width="stretch", height=420)
    st.caption(f"第 {page_num}/{total_pages} 页 · 当前筛选后 {len(filtered_df)} 条")

st.subheader("统计汇总")
stat_cols = [c for c in ["reward", "step", "km", "n_tool_calls"] if c in filtered_df.columns]
if not stat_cols:
    st.info("未检测到可统计的数值字段。")
else:
    stats = filtered_df[stat_cols].describe().T
    st.dataframe(stats, width="stretch")
    dist_cols = [c for c in stat_cols if c != "step"]
    if not dist_cols:
        st.info("暂无可展示的分布图。")
    else:
        for col in dist_cols:
            st.markdown(f"**{col} 分布**")
            st.bar_chart(filtered_df[col].dropna())

# ── Im2GPS3K accuracy metrics ──────────────────────────────────────────────────
if "km" in filtered_df.columns:
    st.subheader("Im2GPS3K 准确率指标")
    km_series = pd.to_numeric(filtered_df["km"], errors="coerce").dropna()
    n_total = len(filtered_df)
    n_parsed = len(km_series)
    n_no_pred = n_total - n_parsed

    thresholds = [1, 25, 200, 750, 2500]
    accs = {t: round((km_series <= t).sum() / n_parsed, 4) if n_parsed else 0.0 for t in thresholds}

    col_info, col_metrics = st.columns([1, 2])
    with col_info:
        st.metric("总样本", n_total)
        st.metric("有预测", n_parsed)
        st.metric("无预测", n_no_pred)
        if "n_tool_calls" in filtered_df.columns:
            avg_calls = pd.to_numeric(filtered_df["n_tool_calls"], errors="coerce").mean()
            st.metric("平均 zoom 次数", f"{avg_calls:.2f}")
    with col_metrics:
        metric_cols = st.columns(len(thresholds))
        for col, t in zip(metric_cols, thresholds):
            col.metric(f"Acc@{t}km", f"{accs[t]:.3f}", f"{int(accs[t]*n_parsed)}/{n_parsed}")

st.subheader("详情面板")
if filtered_df.empty:
    st.info("暂无可展示的数据。")
else:
    detail_indices = filtered_df.index.tolist()
    if "detail_pos" not in st.session_state:
        st.session_state.detail_pos = 0
    if st.session_state.detail_pos >= len(detail_indices):
        st.session_state.detail_pos = 0

    if prev_clicked:
        st.session_state.detail_pos = (st.session_state.detail_pos - 1) % len(detail_indices)
    elif next_clicked:
        st.session_state.detail_pos = (st.session_state.detail_pos + 1) % len(detail_indices)

    current_pos = st.session_state.detail_pos + 1
    total_details = len(detail_indices)

    with st.sidebar:
        st.caption(f"详情进度：{current_pos}/{total_details}")
        if total_details > 1:
            slider_pos = st.slider(
                "跳转到第几条",
                min_value=1,
                max_value=total_details,
                value=current_pos,
                step=1,
            )
            if slider_pos != current_pos:
                st.session_state.detail_pos = slider_pos - 1
        else:
            st.caption("当前仅有 1 条数据，无需跳转。")

        selected_index = st.selectbox(
            "选择行索引",
            options=detail_indices,
            index=st.session_state.detail_pos,
        )

    st.session_state.detail_pos = detail_indices.index(selected_index)
    row = filtered_df.loc[selected_index]
    render_image_preview(row, dataset_cfg)
    for field in PRIORITY_FIELDS:
        if field in row:
            if field == "messages":
                render_messages(row)
            else:
                render_rich_text(field, row[field], row=row, dataset_cfg=dataset_cfg)
    remaining_fields = [
        c for c in all_columns if c not in PRIORITY_FIELDS and c not in {"image_paths", "image_path"}
    ]
    if remaining_fields:
        st.markdown("#### 其他字段")
        for field in remaining_fields:
            render_rich_text(field, row[field], row=row, dataset_cfg=dataset_cfg)
