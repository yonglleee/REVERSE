"""
text_search_tool.py — Cache-backed web-text-search tool for RL training.

Design mirrors ImageSearchTool (method-B negmix + reward):
  1. create()  : record image_path (same key as zoom/image_search tools)
  2. execute() : lookup the pre-built text_search cache by image_path
                 → return a mix of geo-useful + cross-image non-useful results
                 → reward = base_call_reward + n_pos × reward_per_pos
  3. Cache key : image_path (coldstart annotation image path)

Cache parquet schema (text_search_cache_part*.parquet, merged into one):
  image_id      (str)   — filename basename
  image_path    (str)   — full path (matches tools_kwargs 'image' key)
  call_idx      (int)   — 0-based tool call index per image
  query         (str)   — query string (multi-query calls joined with " | ")
  result_index  (int)   — 1-based result position in original response
  result_title  (str)
  result_url    (str)
  result_content(str)
  is_geo_useful (bool)  — True if this result was in Kimi's <useful>[] annotation
  useful_indices(str)   — JSON list, e.g. "[1,3,5]"
  n_useful      (int)
  call_turn     (int)
  part          (str)

Negmix strategy (same as ImageSearchTool method-B):
  - Fixed total results = max_results (all pos + neg fill)
  - Pick n_pos = min(n_pos_available, n_pos_limit, max_results), n_neg fills remaining
  - Fill remaining with cross-image non-useful results from global_neg_pool
  - Shuffle, return as numbered list [1] ... [N]
  - reward = base_call_reward + n_pos × reward_per_pos   (no zoom-before-search enforcement)

Config keys:
  cache_path        (str)   : path to merged text_search_cache parquet
  base_call_reward  (float) : reward for any valid call (cache hit). Default 0.1
  reward_per_pos    (float) : reward per geo-useful result returned. Default 0.1
  n_pos_limit       (int)   : max positive results to return per call. Default 3
  max_results       (int)   : total results per call. Default 10
  enable_negmix     (bool)  : True = method-B, False = return pos-only baseline. Default True
  no_cache_reward   (float) : reward when image_path is not in cache. Default 0.0
"""

import json
import logging
import os
import random
from typing import Any, Optional
from uuid import uuid4

from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# ── defaults ──────────────────────────────────────────────────────────────────
_DEFAULT_CACHE_PATH = (
    "/mnt/sh/mmvision/home/jonahli/data_agent/rl/coldstart/text_search_cache_merged.parquet"
)
_DEFAULT_BASE_CALL_REWARD = 0.1
_DEFAULT_REWARD_PER_POS = 0.1
_DEFAULT_N_POS_LIMIT = 3
_DEFAULT_MAX_RESULTS = 10
_DEFAULT_ENABLE_NEGMIX = True
_DEFAULT_NO_CACHE_REWARD = 0.0


# Process-level memoization: avoid re-loading the ~40MB pickle for every
# ToolAgentLoop instance. VERL's agent_loop.py instantiates a new
# ToolAgentLoop per trajectory (hydra.utils.instantiate on every
# _run_agent_loop call), which triggers TextSearchTool.__init__, which
# calls _load_cache. Without this memoization, each rollout batch of 256
# samples × rollout_n triggers 256+ pickle loads per worker process,
# causing GPU idle / CPU 100% during rollout (2026-05-04 diagnosis).
_CACHE_MEMO: dict[str, tuple[dict, list]] = {}


def _load_cache_uncached(parquet_path: str) -> tuple[dict, list]:
    """
    Load the text_search cache parquet into memory.
    首次加载后生成 pickle 缓存，后续直接读 pickle（速度提升 10x+）。

    Returns
    -------
    cache : dict
        image_path → list of per-call dicts:
            {
                "call_idx":   int,
                "query":      str,
                "pos_results": list of {title, url, content},
                "neg_results": list of {title, url, content},
            }
    global_neg_pool : list of {title, url, content, image_path}
        All non-useful results across all images, for cross-image negmix sampling.
    """
    import pandas as pd
    import pickle

    if not os.path.exists(parquet_path):
        logger.warning(f"TextSearch cache not found: {parquet_path}")
        return {}, []

    # Try loading from pickle cache (much faster)
    pickle_path = parquet_path + ".cache.pkl"
    parquet_mtime = os.path.getmtime(parquet_path)
    if os.path.exists(pickle_path) and os.path.getmtime(pickle_path) >= parquet_mtime:
        try:
            with open(pickle_path, "rb") as f:
                result = pickle.load(f)
            logger.info(f"Loaded TextSearch cache from pickle: {pickle_path} ({len(result[0])} entries)")
            return result
        except Exception as e:
            logger.warning(f"Failed to load pickle cache: {e}, rebuilding from parquet")

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        logger.warning(f"Failed to read TextSearch cache: {e}")
        return {}, []

    needed = ["image_path", "call_idx", "query",
              "result_title", "result_url", "result_content", "result_index",
              "is_geo_useful"]
    df = df[[c for c in needed if c in df.columns]]
    if "is_geo_useful" not in df.columns:
        df["is_geo_useful"] = False

    # 去重：同一 (image_path, call_idx, result_index) 保留 is_geo_useful=True 的
    if "result_index" in df.columns:
        df = df.sort_values(["image_path", "call_idx", "result_index", "is_geo_useful"],
                            ascending=[True, True, True, False])
        df = df.drop_duplicates(subset=["image_path", "call_idx", "result_index"], keep="first")

    # 预排序，使相同 (image_path, call_idx, result_index) 的行相邻
    sort_cols = ["image_path", "call_idx"] + (["result_index"] if "result_index" in df.columns else [])
    df = df.sort_values(sort_cols)

    # 提取 numpy 数组，避免 iterrows（快 ~20x）
    img_paths   = df["image_path"].astype(str).values
    call_idxs   = df["call_idx"].values
    queries     = df["query"].fillna("").astype(str).values
    titles      = df["result_title"].fillna("").astype(str).values
    urls        = df["result_url"].fillna("").astype(str).values
    contents    = df["result_content"].fillna("").astype(str).values
    useful      = df["is_geo_useful"].astype(bool).values

    cache: dict = {}
    global_neg_pool: list = []
    i, n = 0, len(df)

    while i < n:
        ip = img_paths[i]
        ci = call_idxs[i]
        j = i
        while j < n and img_paths[j] == ip and call_idxs[j] == ci:
            j += 1

        query_str = queries[i]
        pos_results: list = []
        neg_results: list = []
        for k in range(i, j):
            r = {
                "title":   titles[k],
                "url":     urls[k],
                "content": contents[k],
            }
            if useful[k]:
                pos_results.append(r)
            else:
                neg_results.append(r)
                global_neg_pool.append({**r, "image_path": ip})

        call_dict = {
            "call_idx":    int(ci),
            "query":       query_str,
            "pos_results": pos_results,
            "neg_results": neg_results,
        }

        if ip not in cache:
            cache[ip] = []
        cache[ip].append(call_dict)
        i = j

    # ensure calls are sorted by call_idx within each image
    for ip in cache:
        cache[ip].sort(key=lambda x: x["call_idx"])

    logger.info(
        f"Loaded TextSearch cache: {len(cache)} images, "
        f"global_neg_pool={len(global_neg_pool)}: {parquet_path}"
    )

    # Save pickle cache for faster subsequent loads
    try:
        with open(pickle_path, "wb") as f:
            pickle.dump((cache, global_neg_pool), f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved TextSearch pickle cache: {pickle_path}")
    except Exception as e:
        logger.warning(f"Failed to save pickle cache: {e}")

    return cache, global_neg_pool


def _load_cache(parquet_path: str) -> tuple[dict, list]:
    """Memoized wrapper around _load_cache_uncached.

    Per-process cache: first call loads (~40MB pickle → dict + list);
    subsequent calls with the same path return the cached object in O(1).
    """
    if parquet_path in _CACHE_MEMO:
        return _CACHE_MEMO[parquet_path]
    result = _load_cache_uncached(parquet_path)
    _CACHE_MEMO[parquet_path] = result
    logger.info(f"TextSearch cache memoized in-process: {parquet_path}")
    return result


def _format_results(results: list[dict]) -> str:
    """Format a list of result dicts into numbered text."""
    lines = []
    for idx, r in enumerate(results, start=1):
        lines.append(f"[{idx}] {r['title']}")
        if r.get("url"):
            lines.append(f"    URL: {r['url']}")
        if r.get("content"):
            lines.append(f"    {r['content'][:400]}")
    return "\n".join(lines)


class TextSearchTool(BaseTool):
    """Cache-backed text search tool with negmix reward for RL training.

    Each image has one or more recorded text_search calls (from coldstart annotation).
    On execute(), the tool cycles through calls in order, returning a negmix of
    geo-useful + cross-image non-useful results.

    Per-instance state:
        image_path  — matched against cache key
        call_cursor — which cached call to use next (0-based)
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict] = {}

        cache_path = config.get("cache_path", _DEFAULT_CACHE_PATH)
        self._base_call_reward: float = float(config.get("base_call_reward", _DEFAULT_BASE_CALL_REWARD))
        self._reward_per_pos: float = float(config.get("reward_per_pos", _DEFAULT_REWARD_PER_POS))
        self._n_pos_limit: int = int(config.get("n_pos_limit", _DEFAULT_N_POS_LIMIT))
        self._max_results: int = int(config.get("max_results", _DEFAULT_MAX_RESULTS))
        self._enable_negmix: bool = bool(config.get("enable_negmix", _DEFAULT_ENABLE_NEGMIX))
        self._no_cache_reward: float = float(config.get("no_cache_reward", _DEFAULT_NO_CACHE_REWARD))

        self._cache, self._global_neg_pool = _load_cache(cache_path)
        logger.info(
            f"TextSearchTool initialized: cache_size={len(self._cache)}, "
            f"base_call_reward={self._base_call_reward}, reward_per_pos={self._reward_per_pos}, "
            f"n_pos_limit={self._n_pos_limit}, max_results={self._max_results}, "
            f"enable_negmix={self._enable_negmix}"
        )

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())

        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            kwargs.update(create_kwargs)

        image_path = kwargs.get("image", "")
        self._instance_dict[instance_id] = {
            "image_path": image_path or "",
            "call_cursor": 0,   # which cached call to serve next
            "last_pos_indices": None,  # for F1 reward computation
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """
        Return text search results from the cache for this image.

        The `query` parameter from the model is ignored for result content (results
        are pre-cached from coldstart annotation). The query is still included in the
        response header for readability.

        Cycles through cached calls in order (call_cursor). Once all calls are
        exhausted, returns an empty / cache-miss response.
        """
        query_raw = parameters.get("query", "")
        if isinstance(query_raw, list):
            query_display = " | ".join(str(q) for q in query_raw if q)
        else:
            query_display = str(query_raw).strip()

        instance = self._instance_dict.get(instance_id, {})
        image_path = instance.get("image_path", "")

        calls = self._cache.get(image_path) if image_path else None

        if not calls:
            logger.info(f"TextSearch cache miss: image_path={image_path}")
            text = f"Web search results for: {query_display}\n\nNo results found."
            return (
                ToolResponse(text=text),
                self._no_cache_reward,
                {"success": False, "cache_hit": False},
            )

        # Pick the best-matching cached call by query similarity (Jaccard on word tokens)
        # Falls back to call_cursor order if similarity is uniformly 0 (e.g., empty query)
        def _jaccard(a: str, b: str) -> float:
            ta = set(a.lower().split())
            tb = set(b.lower().split())
            if not ta and not tb:
                return 1.0
            if not ta or not tb:
                return 0.0
            return len(ta & tb) / len(ta | tb)

        used_cursors = instance.get("used_cursors", set())
        # Score each cached call; prefer unused calls, then best similarity
        best_score, best_call = -1.0, None
        for c in calls:
            ci = c["call_idx"]
            score = _jaccard(query_display, c["query"])
            # Penalize already-used calls slightly to encourage diversity
            if ci in used_cursors:
                score -= 0.5
            if score > best_score:
                best_score = score
                best_call = c
        # If all are equally bad (all used, sim=0), fall back to cursor order
        if best_call is None:
            call_cursor = instance.get("call_cursor", 0)
            best_call = calls[call_cursor % len(calls)]
            self._instance_dict[instance_id]["call_cursor"] = call_cursor + 1
        else:
            used_cursors.add(best_call["call_idx"])
            self._instance_dict[instance_id]["used_cursors"] = used_cursors

        call = best_call
        pos_results = call["pos_results"]
        neg_results = call["neg_results"]
        cached_query = call["query"]

        # Build response header using cached query
        response_query = cached_query if cached_query else query_display

        if not self._enable_negmix:
            # Baseline: return all pos results, no negmix
            text = f"Web search results for: {response_query}\n\n"
            text += _format_results(pos_results[:self._max_results])
            n_pos_returned = min(len(pos_results), self._max_results)
            reward = self._base_call_reward + n_pos_returned * self._reward_per_pos
            return (
                ToolResponse(text=text),
                reward,
                {"success": True, "cache_hit": True, "n_pos": n_pos_returned, "negmix": False},
            )

        # ── 方案C negmix：本组负样本 + pos_recall 归一化 reward ──────────────────
        # 本组负样本：同一次搜索调用里 is_geo_useful=False 的结果（语义更相关，判别难度更高）
        n_pos_total = len(pos_results)

        # 固定返回 max_results 条：全部正样本 + 负样本补满
        n_pos = min(n_pos_total, self._max_results, self._n_pos_limit)
        n_neg = min(len(neg_results), self._max_results - n_pos)
        total = n_pos + n_neg

        if total == 0:
            text = f"Web search results for: {response_query}\n\nNo results found."
            return (
                ToolResponse(text=text),
                self._no_cache_reward,
                {"success": False, "cache_hit": True, "n_pos": 0},
            )

        # tagged tuple 确保 shuffle 后序号和正负标记严格对齐
        sampled_pos = random.sample(pos_results, n_pos) if n_pos > 0 else []
        sampled_neg = random.sample(neg_results, n_neg) if n_neg > 0 else []

        tagged = [(True, r) for r in sampled_pos] + [(False, r) for r in sampled_neg]
        random.shuffle(tagged)

        # 正样本在 mixed 里的 1-based index
        pos_indices_in_mixed = {i + 1 for i, (is_pos, _) in enumerate(tagged) if is_pos}

        text = f"Web search results for: {response_query}\n\n"
        # format with 1-based index
        lines = []
        for idx, (_, r) in enumerate(tagged, start=1):
            lines.append(f"[{idx}] {r.get('title','')}")
            if r.get('url'):
                lines.append(f"    URL: {r['url']}")
            if r.get('content'):
                lines.append(f"    {r['content'][:200]}")
        text += "\n".join(lines)

        # F1 reward：基于模型上一轮的 <useful> 标签与真实正样本的匹配度
        # F1 is computed in agent_loop, not here
        reward = self._base_call_reward

        return (
            ToolResponse(text=text),
            reward,
            {
                "success": True,
                "cache_hit": True,
                "reward": reward,
                "_pos_indices_in_mixed": pos_indices_in_mixed,
                "_n_total_results": total,  # total results returned (for MCC computation)
            },
        )

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """TextSearch reward is already returned from execute(); no episode-level reward."""
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
