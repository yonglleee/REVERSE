"""
image_search_tool.py — 本地图搜缓存工具，用于 RL 训练中的 image search 工具调用。

流程：
  1. create()：从 create_kwargs 获取 image_path（SpotSFT 原始路径，与缓存 key 一致）
  2. execute()：接收 bbox_2d（归一化到 [0,1000]），先做 IOU 检查：
     - IOU(model_bbox, gt_bbox) >= iou_threshold → 命中，返回混合结果（正样本 + 跨组负样本）
     - 否则 → "No search results available for this region."
     这使得 image_search 的增益与 zoom 精度绑定：精准 zoom 才能触发有效图搜。
  3. 缓存在 __init__ 时一次性加载到内存 dict，按 image_path 索引，O(1) 查找

方案B（混合正负样本）：
  - IOU 命中后，固定返回 max_results 条结果（全部正样本 + 负样本补满）
  - 剩余 N-n_pos 条从其他图的负样本池中随机填充
  - 打乱顺序后返回，让模型自己判断哪些结果有用
  - reward = base_call_reward（保底）+ n_pos_returned × reward_per_pos（判别信号）

Cache key 设计：
  - RL tools_kwargs 里传入的 image 即为 SpotSFT 原始路径
  - parquet 里的 image_path 同为 SpotSFT 原始路径，两者直接对齐

缓存 parquet schema（search_cache_labeled.parquet）:
  index, image_path, ground_truth, filter_tier,
  bbox_idx, bbox (gt best_bbox, JSON string), turn, gcs_url,
  result_pos, result_title, result_link, result_source, result_thumbnail, result_image,
  is_geo_useful, is_positive, useful_indices, label_reason
"""

import base64
import io
import json
import logging
import os
import random
from typing import Any, Optional
from uuid import uuid4

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# 默认缓存路径（使用带质量标注的 labeled 版本）
_DEFAULT_CACHE_PATH = (
    "/mnt/sh/mmvision/home/jonahli/data_agent/rl/coldstart/image_search_cache_merged.parquet"
)

# 默认 IOU 阈值：模型 bbox 与 gt bbox 的 IOU 达到此值才返回搜索结果
_DEFAULT_IOU_THRESHOLD = 0.5

# 方案B 默认参数
_DEFAULT_N_POS_SAMPLES = 3       # 每次返回的正样本数上限
_DEFAULT_N_NEG_SAMPLES = 3       # 每次返回的本组负样本数
_DEFAULT_REWARD_PER_POS = 0.3    # pos_recall × reward_per_pos 的乘数（归一化后最高 +0.3）
_DEFAULT_MAX_RESULTS = 10        # 每次返回的最大结果数（正+负之和）
_DEFAULT_ENABLE_NEGMIX = True    # True=方案C（本组负样本+归一化reward），False=baseline
_DEFAULT_MAX_THUMBNAIL_IMAGES = 5  # 每次最多随结果返回的缩略图张数
_DEFAULT_BASE_CALL_REWARD = 0.1  # IOU 命中的保底奖励
_DEFAULT_DISCRIMINATION_PENALTY = 0.05  # 每个误标 neg 为 useful 的惩罚


def _b64_to_pil(b64_str: str):
    """将 base64 字符串（可带 data URI 前缀）解码为 PIL.Image，失败返回 None。"""
    try:
        from PIL import Image as _PIL_Image
        if b64_str.startswith("data:"):
            b64_str = b64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_str)
        return _PIL_Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return None


def _bbox_iou(a: list[float], b: list[float]) -> float:
    """计算两个 bbox 的 IOU，坐标格式 [x1, y1, x2, y2]。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    if inter == 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0.0 else 0.0


# Process-level memoization: avoid re-loading the ~50MB pickle for every
# ToolAgentLoop instance (see text_search_tool.py for same comment).
_CACHE_MEMO: dict[str, tuple[dict, list]] = {}


def _load_cache_uncached(parquet_path: str) -> tuple[dict, list]:
    """
    将 parquet 加载为内存 dict，并构建全局负样本池。
    首次加载后生成 pickle 缓存，后续直接读 pickle（速度提升 10x+）。

    Returns:
        cache: dict
            key:   image_path (str)
            value: list of per-call dicts, sorted by call_idx:
                {
                    "call_idx":    int,
                    "gt_bbox":     [x1, y1, x2, y2],   # 该次调用的 gt bbox
                    "pos_results": list of {title, link, source},
                    "neg_results": list of {title, link, source},
                }
        global_neg_pool: list of {title, link, source, image_path}
    """
    import pandas as pd
    import pickle

    if not os.path.exists(parquet_path):
        logger.warning(f"Search cache not found: {parquet_path}")
        return {}, []

    # Try loading from pickle cache (much faster)
    pickle_path = parquet_path + ".cache.pkl"
    parquet_mtime = os.path.getmtime(parquet_path)
    if os.path.exists(pickle_path) and os.path.getmtime(pickle_path) >= parquet_mtime:
        try:
            with open(pickle_path, "rb") as f:
                result = pickle.load(f)
            logger.info(f"Loaded cache from pickle: {pickle_path} ({len(result[0])} entries)")
            return result
        except Exception as e:
            logger.warning(f"Failed to load pickle cache: {e}, rebuilding from parquet")

    df = pd.read_parquet(parquet_path)
    needed = [
        "image_path", "bbox", "call_idx",
        "result_pos", "result_title", "result_link", "result_source",
        "is_geo_useful", "result_thumbnail",
    ]
    df = df[[c for c in needed if c in df.columns]]

    if "is_geo_useful" not in df.columns:
        df["is_geo_useful"] = True
    if "call_idx" not in df.columns:
        df["call_idx"] = 0

    # 去重：同一 (image_path, call_idx, result_pos) 出现多次时保留 is_geo_useful=True 的
    # （两个 part 合并时可能产生重复行，标注结果不同时优先取 True）
    if "result_pos" in df.columns:
        df = df.sort_values(["image_path", "call_idx", "result_pos", "is_geo_useful"],
                            ascending=[True, True, True, False])
        df = df.drop_duplicates(subset=["image_path", "call_idx", "result_pos"], keep="first")

    # 预排序：image_path → call_idx → result_pos
    sort_cols = ["image_path", "call_idx"] + (["result_pos"] if "result_pos" in df.columns else [])
    df = df.sort_values(sort_cols)

    titles    = df["result_title"].fillna("").astype(str).values
    links     = df["result_link"].fillna("").astype(str).values
    sources   = df["result_source"].fillna("").astype(str).values
    thumbs    = df["result_thumbnail"].fillna("").astype(str).values
    useful    = df["is_geo_useful"].astype(bool).values
    img_paths = df["image_path"].astype(str).values
    bboxes    = df["bbox"].astype(str).values
    call_idxs = df["call_idx"].values

    cache: dict = {}
    global_neg_pool: list = []
    i, n = 0, len(df)

    while i < n:
        ip = img_paths[i]
        ci = call_idxs[i]
        bb = bboxes[i]
        j = i
        while j < n and img_paths[j] == ip and call_idxs[j] == ci:
            j += 1

        try:
            gt_bbox = json.loads(bb)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Failed to parse gt bbox: {bb!r}, skipping")
            i = j
            continue

        pos_results: list = []
        neg_results: list = []
        for k in range(i, j):
            result = {
                "title":     titles[k],
                "link":      links[k],
                "source":    sources[k],
                "thumbnail": thumbs[k],
            }
            if useful[k]:
                pos_results.append(result)
            else:
                neg_results.append(result)
                global_neg_pool.append({**result, "image_path": ip})

        call_dict = {
            "call_idx":    int(ci),
            "gt_bbox":     gt_bbox,
            "all_results": pos_results + neg_results,
            "pos_results": pos_results,
            "neg_results": neg_results,
        }
        if ip not in cache:
            cache[ip] = []
        cache[ip].append(call_dict)
        i = j

    # sort calls by call_idx within each image
    for ip in cache:
        cache[ip].sort(key=lambda x: x["call_idx"])

    logger.info(
        f"Loaded {len(cache)} cache entries, global_neg_pool size={len(global_neg_pool)}: {parquet_path}"
    )

    # Save pickle cache for faster subsequent loads
    try:
        with open(pickle_path, "wb") as f:
            pickle.dump((cache, global_neg_pool), f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved pickle cache: {pickle_path}")
    except Exception as e:
        logger.warning(f"Failed to save pickle cache: {e}")

    return cache, global_neg_pool


def _load_cache(parquet_path: str) -> tuple[dict, list]:
    """Memoized wrapper around _load_cache_uncached.

    Per-process cache: first call loads (~50MB pickle → dict + list);
    subsequent calls with the same path return the cached object in O(1).
    """
    if parquet_path in _CACHE_MEMO:
        return _CACHE_MEMO[parquet_path]
    result = _load_cache_uncached(parquet_path)
    _CACHE_MEMO[parquet_path] = result
    logger.info(f"ImageSearch cache memoized in-process: {parquet_path}")
    return result


class ImageSearchTool(BaseTool):
    """本地图搜缓存工具（方案B：随机混合正负样本）。

    模型调用示例：
        <tool_call>
        {"name": "image_search_tool", "arguments": {"bbox_2d": [200, 300, 600, 700]}}
        </tool_call>

    IOU 命中时，固定返回 max_results 条结果（全部正样本 + 负样本补满），
    剩余名额用跨组负样本填充，打乱后呈现。reward = n_pos × reward_per_pos（线性累加）。

    随机正负比例防止模型记住固定格式，迫使模型真正判断每条结果的有用性。
    """

    NORMALIZED_COORD_MAX = 1000.0

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict] = {}

        cache_path = config.get("cache_path", _DEFAULT_CACHE_PATH)
        self._iou_threshold: float = float(config.get("iou_threshold", _DEFAULT_IOU_THRESHOLD))

        # 方案B 参数
        self._reward_per_pos: float = float(
            config.get("reward_per_pos", _DEFAULT_REWARD_PER_POS)
        )
        self._max_results: int = int(config.get("max_results", _DEFAULT_MAX_RESULTS))
        self._n_pos_limit: int = int(config.get("n_pos_limit", _DEFAULT_N_POS_SAMPLES))
        self._enable_negmix: bool = bool(config.get("enable_negmix", _DEFAULT_ENABLE_NEGMIX))
        self._base_call_reward: float = float(
            config.get("base_call_reward", _DEFAULT_BASE_CALL_REWARD)
        )
        self._discrimination_penalty: float = float(
            config.get("discrimination_penalty", _DEFAULT_DISCRIMINATION_PENALTY)
        )
        self._iou_reward_coeff: float = float(config.get("iou_reward_coeff", 0.2))
        self._max_thumbnail_images: int = int(
            config.get("max_thumbnail_images", _DEFAULT_MAX_THUMBNAIL_IMAGES)
        )

        self._cache, self._global_neg_pool = _load_cache(cache_path)
        logger.info(
            f"ImageSearchTool initialized, cache_size={len(self._cache)}, "
            f"iou_threshold={self._iou_threshold}, "
            f"enable_negmix={self._enable_negmix}, "
            f"base_call_reward={self._base_call_reward}, reward_per_pos={self._reward_per_pos}, "
            f"discrimination_penalty={self._discrimination_penalty}, "
            f"max_results={self._max_results}, n_pos_limit={self._n_pos_limit}"
        )

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """创建实例，记录 image_path（与 image_zoom_in_tool 保持一致）。"""
        if instance_id is None:
            instance_id = str(uuid4())

        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            kwargs.update(create_kwargs)

        image_path = kwargs.get("image")  # create_kwargs 里传的是 "image" key（文件路径）
        self._instance_dict[instance_id] = {
            "image_path": image_path or "",
            "call_cursor": 0,  # cycles through per-call gt_bbox entries
            "last_pos_indices": None,  # for F1 reward computation
        }
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """
        根据 bbox_2d 查本地缓存，返回混合正负样本搜索结果（方案B）。

        1. 验证 bbox_2d 格式和范围
        2. 用 image_path 找到缓存条目（gt_bbox + pos/neg_results）
        3. 计算 IOU(model_bbox, gt_bbox)：
           - IOU >= iou_threshold →
               * 随机采样 n_pos 条正样本（当前组）
               * 从全局负样本池中采样 n_neg 条（排除当前图）
               * 打乱后呈现，给 discrimination_reward
           - IOU <  iou_threshold → "No search results available."
        """
        bbox_2d = parameters.get("bbox_2d")

        if not bbox_2d or len(bbox_2d) != 4:
            return (
                ToolResponse(text="Error: bbox_2d parameter is missing or not a list of 4 numbers."),
                -0.05,
                {"success": False},
            )

        # 验证坐标范围
        try:
            bbox_floats = [float(v) for v in bbox_2d]
        except (TypeError, ValueError):
            return (
                ToolResponse(text="Error: bbox_2d contains non-numeric values."),
                -0.05,
                {"success": False},
            )

        if any(v < 0 or v > self.NORMALIZED_COORD_MAX for v in bbox_floats):
            return (
                ToolResponse(
                    text=f"Error: bbox_2d must be normalized in [0, {int(self.NORMALIZED_COORD_MAX)}]."
                ),
                -0.05,
                {"success": False},
            )

        image_path = self._instance_dict.get(instance_id, {}).get("image_path", "")

        # cache is now image_path → list[call_dict], one per cached search call
        calls = self._cache.get(image_path) if image_path else None

        if not calls:
            logger.info(f"image_search cache miss: image_path={image_path}")
            text = "No search results available for this region."
            return ToolResponse(text=text), 0.0, {"success": False, "cache_hit": False, "iou": None}

        # Use call_cursor to pick which cached call to compare against
        call_cursor = self._instance_dict.get(instance_id, {}).get("call_cursor", 0)
        entry = calls[call_cursor % len(calls)]
        self._instance_dict[instance_id]["call_cursor"] = call_cursor + 1

        gt_bbox = entry["gt_bbox"]
        pos_results = entry["pos_results"]

        iou = _bbox_iou(bbox_floats, [float(v) for v in gt_bbox])
        logger.info(
            f"image_search image_path={os.path.basename(image_path)} call={call_cursor} IOU={iou:.3f} "
            f"(threshold={self._iou_threshold}), model_bbox={bbox_floats}, gt_bbox={gt_bbox}, "
            f"pos={len(pos_results)}"
        )

        if iou < self._iou_threshold:
            text = "No search results available for this region."
            return ToolResponse(text=text), 0.0, {"success": False, "cache_hit": False, "iou": iou}

        # ── IOU 达标：根据 enable_negmix 选择返回策略 ──────────────────────────
        if not self._enable_negmix:
            # Baseline：返回全部结果，按 result_pos 排序，reward=base_call_reward
            all_results = entry.get("all_results", pos_results + entry.get("neg_results", []))
            bbox_ints = [int(round(v)) for v in bbox_floats]
            lines = [f"Image search results for region {bbox_ints}:\n"]
            for idx, r in enumerate(all_results, start=1):
                lines.append(f"[{idx}] {r['title']}")
                lines.append(f"    Source: {r['source']}")
                lines.append(f"    Link: {r['link']}")
            text = "\n".join(lines)
            return (
                ToolResponse(text=text),
                self._base_call_reward,
                {"success": True, "cache_hit": True, "iou": iou, "negmix": False},
            )

        # ── 方案C：本组负样本 + 归一化 reward + 判别惩罚 ──────────────────────
        # 负样本来自同一次搜索的本组 neg（与当前图相关），而非跨组随机采样。
        # reward = base_call_reward + (n_pos_returned / max(n_pos_total, 1)) * reward_per_pos
        # 判别惩罚：若上一轮模型标注了 <useful>[]，与真实 pos 比对，误标 neg 为 useful → 惩罚
        neg_results = entry.get("neg_results", [])
        n_pos_total = len(pos_results)

        # 固定返回 max_results 条：全部正样本 + 负样本补满
        n_pos = min(n_pos_total, self._max_results, self._n_pos_limit)
        n_neg = min(len(neg_results), self._max_results - n_pos)
        total = n_pos + n_neg

        if total == 0:
            text = "No search results available for this region."
            return ToolResponse(text=text), 0.0, {"success": False, "cache_hit": False, "iou": iou}

        # 1. 正样本：本组 is_geo_useful=True
        sampled_pos = random.sample(pos_results, n_pos) if n_pos > 0 else []

        # 2. 本组负样本：本组 is_geo_useful=False（与当前图语义相关，判别难度更高）
        sampled_neg = random.sample(neg_results, n_neg) if n_neg > 0 else []

        # 3. 用 (is_pos, result) 标记后打乱，shuffle 不影响标记
        tagged = [(True, r) for r in sampled_pos] + [(False, r) for r in sampled_neg]
        random.shuffle(tagged)

        # 正样本在 mixed 里的 1-based index（打乱后，序号和标记严格对齐）
        pos_indices_in_mixed = {i + 1 for i, (is_pos, _) in enumerate(tagged) if is_pos}

        # 4. 格式化结果（展示给模型的序号和 pos_indices_in_mixed 完全对应）
        bbox_ints = [int(round(v)) for v in bbox_floats]
        lines = [f"Image search results for region {bbox_ints}:\n"]
        for idx, (_, r) in enumerate(tagged, start=1):
            lines.append(f"[{idx}] {r['title']}")
            lines.append(f"    Source: {r['source']}")
            lines.append(f"    Link: {r['link']}")
        text = "\n".join(lines)

        # 5. IOU reward only (F1 is computed in agent_loop, not here)
        #    iou_reward = iou × coeff  (default 0.2: IOU=0.5→0.1, IOU=1.0→0.2)
        iou_reward = iou * self._iou_reward_coeff

        final_reward = iou_reward

        return (
            ToolResponse(text=text),
            final_reward,
            {
                "success": True,
                "cache_hit": True,
                "iou": iou,
                "iou_reward": iou_reward,
                "reward": final_reward,
                "reward": final_reward,
                "_pos_indices_in_mixed": pos_indices_in_mixed,  # for agent_loop to persist
                "_n_total_results": total,  # total results returned (for MCC computation)
            },
        )

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """image search 自身奖励已在 execute() 中返回。"""
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
