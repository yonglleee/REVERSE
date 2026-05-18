"""
utils_agent_tool.py — Tool utilities for geo-localization eval agent loop.

Public API:
  - parse_tool_calls_flexible(text)        → List[dict]   # parse model tool call output
  - crop_tool_core(img_path, bbox_2d, ...) → dict         # crop image region
  - image_search_tool_core(img_path, bbox_2d) → dict      # crop + GCS upload + oxylabs reverse image search
  - tavily_search_tool_core(query, api_key) → dict        # Tavily web text search
  - ToolCallManager(img_path)              → manager      # per-image tool state tracker

Adding a new tool (e.g. search):
  1. Add search_tool_core() function below
  2. Add execute_search() method to ToolCallManager
  3. In eval_benchmark.py agent loop, add elif name == "search_web" branch
"""

import base64
import concurrent.futures
import hashlib
import io
import json
import math
import os
import random
import re
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

# ── Constants ──────────────────────────────────────────────────────────────────
IMAGE_FACTOR    = 28
MIN_PIXELS      = 256 * 256
MAX_PIXELS      = 2048 * 1024

MAX_CROP_CALLS  = 5
MAX_SEARCH_CALLS = 3
MAX_TAVILY_CALLS = 5

# ── SQLite cache helpers ───────────────────────────────────────────────────────
# v2: excludes flickr from image search results (both live and cached)
_CACHE_DIR = os.environ.get("EVAL_CACHE_DIR",
             "/mnt/sh/mmvision/home/jonahli/save/agent/cache_v2")
_TAVILY_CACHE_DB = os.path.join(_CACHE_DIR, "tavily_cache.db")
_IMSEARCH_CACHE_DB = os.path.join(_CACHE_DIR, "imsearch_cache.db")
_db_lock = threading.Lock()


def _get_db(path: str, ddl: str) -> sqlite3.Connection:
    """Open (or create) a SQLite DB in WAL mode and run the DDL."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(ddl)
    conn.commit()
    return conn


def _get_tavily_db() -> sqlite3.Connection:
    return _get_db(_TAVILY_CACHE_DB, """
        CREATE TABLE IF NOT EXISTS tavily_cache (
            key       TEXT PRIMARY KEY,
            query     TEXT NOT NULL,
            result    TEXT NOT NULL,
            created   INTEGER NOT NULL
        )
    """)


def _get_imsearch_db() -> sqlite3.Connection:
    return _get_db(_IMSEARCH_CACHE_DB, """
        CREATE TABLE IF NOT EXISTS imsearch_cache (
            key       TEXT PRIMARY KEY,
            img_path  TEXT NOT NULL,
            bbox      TEXT NOT NULL,
            result    TEXT NOT NULL,
            created   INTEGER NOT NULL
        )
    """)


# ── Tavily cache ───────────────────────────────────────────────────────────────
def _tavily_cache_key(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()


def _tavily_cache_get(query: str) -> Optional[Dict]:
    key = _tavily_cache_key(query)
    try:
        with _db_lock:
            conn = _get_tavily_db()
            row = conn.execute(
                "SELECT result FROM tavily_cache WHERE key=?", (key,)
            ).fetchone()
            conn.close()
        if row:
            return json.loads(row[0])
    except Exception:
        pass
    return None


def _tavily_cache_set(query: str, result: Dict) -> None:
    key = _tavily_cache_key(query)
    try:
        with _db_lock:
            conn = _get_tavily_db()
            conn.execute(
                "INSERT OR REPLACE INTO tavily_cache (key, query, result, created) VALUES (?,?,?,?)",
                (key, query, json.dumps(result, ensure_ascii=False), int(time.time())),
            )
            conn.commit()
            conn.close()
    except Exception:
        pass


# ── Image search cache ─────────────────────────────────────────────────────────
def _imsearch_cache_key(img_path: str, bbox_2d: List[float]) -> str:
    bbox_str = ",".join(f"{v:.1f}" for v in bbox_2d)
    raw = f"{img_path}|{bbox_str}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _imsearch_cache_get(img_path: str, bbox_2d: List[float]) -> Optional[Dict]:
    key = _imsearch_cache_key(img_path, bbox_2d)
    try:
        with _db_lock:
            conn = _get_imsearch_db()
            row = conn.execute(
                "SELECT result FROM imsearch_cache WHERE key=?", (key,)
            ).fetchone()
            conn.close()
        if row:
            return json.loads(row[0])
    except Exception:
        pass
    return None


def _imsearch_cache_set(img_path: str, bbox_2d: List[float], result: Dict) -> None:
    key = _imsearch_cache_key(img_path, bbox_2d)
    bbox_str = ",".join(f"{v:.1f}" for v in bbox_2d)
    try:
        with _db_lock:
            conn = _get_imsearch_db()
            conn.execute(
                "INSERT OR REPLACE INTO imsearch_cache (key, img_path, bbox, result, created) VALUES (?,?,?,?,?)",
                (key, img_path, bbox_str, json.dumps(result, ensure_ascii=False), int(time.time())),
            )
            conn.commit()
            conn.close()
    except Exception:
        pass

# ── Tavily key pool ────────────────────────────────────────────────────────────
def _load_tavily_keys_from_env(env_file: Optional[str] = None) -> List[str]:
    """
    Load all Tavily keys from .env file + environment variable.
    .env file is the primary source (supports multiple TAVILY_API_KEY lines).
    Process environment variable is added only if not already in .env.
    Active lines (no #) come first; commented lines are kept as fallback.
    """
    keys: List[str] = []
    seen: set = set()

    # 1. From .env file (primary — supports multiple active lines)
    if env_file is None:
        env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_file):
        active, commented = [], []
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith("TAVILY_API_KEY="):
                    k = line[len("TAVILY_API_KEY="):].split("#")[0].strip().strip('"').strip("'")
                    if k:
                        active.append(k)
                elif line.startswith("# TAVILY_API_KEY="):
                    k = line[len("# TAVILY_API_KEY="):].split("#")[0].strip().strip('"').strip("'")
                    if k:
                        commented.append(k)
        for k in active + commented:
            if k not in seen:
                keys.append(k)
                seen.add(k)

    # 2. From process environment (fallback, e.g. passed directly via export)
    env_val = os.environ.get("TAVILY_API_KEY", "")
    for k in re.split(r"[,\s]+", env_val):
        k = k.strip()
        if k and k not in seen:
            keys.append(k)
            seen.add(k)

    return keys


def _check_tavily_key_quota(key: str, timeout: int = 10) -> int:
    """
    Query Tavily usage API.
    Returns remaining quota (plan_limit - plan_usage), or -1 on error.
    """
    try:
        import requests as _req
        _proxies = _get_proxies()
        resp = _req.get(
            "https://api.tavily.com/usage",
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
            proxies=_proxies,
        )
        if resp.status_code == 200:
            acct = resp.json().get("account", {})
            limit = acct.get("plan_limit", 0)
            usage = acct.get("plan_usage", 0)
            if isinstance(limit, int) and isinstance(usage, int):
                return max(0, limit - usage)
        return -1
    except Exception:
        return -1


class _TavilyKeyPool:
    """
    Thread-safe Tavily API key pool.
    On first use, pre-checks quota for all keys and only admits those with
    remaining quota > 0. Automatically rotates to next key on 429 / 432.
    """
    def __init__(self):
        self._lock    = threading.Lock()
        self._keys:      List[str] = []
        self._exhausted: set       = set()
        self._idx = 0
        self._loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            all_keys = _load_tavily_keys_from_env()
            # Check all keys concurrently to avoid serial timeout delays
            from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
            results = {}
            with ThreadPoolExecutor(max_workers=min(len(all_keys), 20)) as executor:
                future_to_key = {executor.submit(_check_tavily_key_quota, k): k for k in all_keys}
                for fut in _as_completed(future_to_key):
                    k = future_to_key[fut]
                    results[k] = fut.result()
            valid = []
            for k in all_keys:
                remaining = results.get(k, -1)
                short = k[:16] + "..."
                if remaining > 0:
                    print(f"[TavilyPool] ✅ {short}  remaining={remaining}", flush=True)
                    valid.append(k)
                elif remaining == 0:
                    print(f"[TavilyPool] ❌ {short}  exhausted (skipped)", flush=True)
                else:
                    # quota check failed (network error etc.) — admit key, let runtime decide
                    print(f"[TavilyPool] ⚠️  {short}  quota check failed, admitted anyway", flush=True)
                    valid.append(k)
            self._keys   = valid
            self._loaded = True
            print(f"[TavilyPool] {len(self._keys)}/{len(all_keys)} key(s) admitted to pool", flush=True)

    def get_key(self) -> str:
        """Return next available key (round-robin), or '' if all exhausted."""
        with self._lock:
            self._ensure_loaded()
            n = len(self._keys)
            for i in range(n):
                k = self._keys[(self._idx + i) % n]
                if k not in self._exhausted:
                    self._idx = (self._idx + i + 1) % n  # advance for next caller
                    return k
            return ""

    def mark_exhausted(self, key: str):
        """Mark a key as exhausted (429/432) and rotate to next."""
        with self._lock:
            self._ensure_loaded()
            if key not in self._exhausted:
                self._exhausted.add(key)
                short = key[:12] + "..."
                remaining = [k for k in self._keys if k not in self._exhausted]
                print(f"[TavilyPool] Key {short} exhausted. "
                      f"{len(remaining)}/{len(self._keys)} key(s) remaining.", flush=True)
                # Advance index past exhausted key
                if key in self._keys:
                    self._idx = (self._keys.index(key) + 1) % max(len(self._keys), 1)

    def reload(self):
        """Force reload keys from .env (e.g. after adding a new key)."""
        with self._lock:
            self._loaded = False
            self._exhausted.clear()
            self._idx = 0


_tavily_pool = _TavilyKeyPool()

# Keep for backward compat (single-key callers)
_TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

# ── Serper cache DB ────────────────────────────────────────────────────────────
_SERPER_CACHE_DB = os.path.join(_CACHE_DIR, "serper_imsearch_cache.db")


def _get_serper_db() -> sqlite3.Connection:
    return _get_db(_SERPER_CACHE_DB, """
        CREATE TABLE IF NOT EXISTS serper_imsearch_cache (
            key       TEXT PRIMARY KEY,
            img_path  TEXT NOT NULL,
            bbox      TEXT NOT NULL,
            result    TEXT NOT NULL,
            created   INTEGER NOT NULL
        )
    """)


def _serper_imsearch_cache_key(img_path: str, bbox_2d: List[float]) -> str:
    bbox_str = ",".join(f"{v:.1f}" for v in bbox_2d)
    return hashlib.sha256(f"{img_path}|{bbox_str}".encode()).hexdigest()


def _serper_imsearch_cache_get(img_path: str, bbox_2d: List[float]) -> Optional[Dict]:
    key = _serper_imsearch_cache_key(img_path, bbox_2d)
    try:
        with _db_lock:
            conn = _get_serper_db()
            row = conn.execute(
                "SELECT result FROM serper_imsearch_cache WHERE key=?", (key,)
            ).fetchone()
            conn.close()
        if row:
            return json.loads(row[0])
    except Exception:
        pass
    return None


def _serper_imsearch_cache_set(img_path: str, bbox_2d: List[float], result: Dict) -> None:
    key = _serper_imsearch_cache_key(img_path, bbox_2d)
    bbox_str = ",".join(f"{v:.1f}" for v in bbox_2d)
    try:
        with _db_lock:
            conn = _get_serper_db()
            conn.execute(
                "INSERT OR REPLACE INTO serper_imsearch_cache "
                "(key, img_path, bbox, result, created) VALUES (?,?,?,?,?)",
                (key, img_path, bbox_str, json.dumps(result, ensure_ascii=False), int(time.time())),
            )
            conn.commit()
            conn.close()
    except Exception:
        pass


# ── Serper key pool ────────────────────────────────────────────────────────────
def _load_serper_keys_from_env(env_file: Optional[str] = None) -> List[str]:
    """Load all SERPER_API_KEY lines from .env (active first, then commented)."""
    keys: List[str] = []
    seen: set = set()

    if env_file is None:
        env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_file):
        active, commented = [], []
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith("SERPER_API_KEY="):
                    k = line[len("SERPER_API_KEY="):].split("#")[0].strip().strip('"').strip("'")
                    if k:
                        active.append(k)
                elif line.startswith("# SERPER_API_KEY="):
                    k = line[len("# SERPER_API_KEY="):].split("#")[0].strip().strip('"').strip("'")
                    if k:
                        commented.append(k)
        for k in active + commented:
            if k not in seen:
                keys.append(k)
                seen.add(k)

    env_val = os.environ.get("SERPER_API_KEY", "")
    for k in re.split(r"[,\s]+", env_val):
        k = k.strip()
        if k and k not in seen:
            keys.append(k)
            seen.add(k)

    return keys


class _SerperKeyPool:
    """
    Thread-safe round-robin Serper API key pool.
    Serper has no public quota-check endpoint, so we skip pre-validation
    and rotate on HTTP 429 at runtime.
    """
    def __init__(self):
        self._lock    = threading.Lock()
        self._keys:      List[str] = []
        self._exhausted: set       = set()
        self._idx = 0
        self._loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            self._keys = _load_serper_keys_from_env()
            self._loaded = True
            print(f"[SerperPool] {len(self._keys)} key(s) loaded", flush=True)

    def get_key(self) -> str:
        with self._lock:
            self._ensure_loaded()
            n = len(self._keys)
            if n == 0:
                return ""
            for i in range(n):
                k = self._keys[(self._idx + i) % n]
                if k not in self._exhausted:
                    return k
            return ""

    def mark_exhausted(self, key: str):
        with self._lock:
            self._ensure_loaded()
            if key not in self._exhausted:
                self._exhausted.add(key)
                short = key[:12] + "..."
                remaining = [k for k in self._keys if k not in self._exhausted]
                print(f"[SerperPool] Key {short} exhausted. "
                      f"{len(remaining)}/{len(self._keys)} key(s) remaining.", flush=True)
                if key in self._keys:
                    self._idx = (self._keys.index(key) + 1) % max(len(self._keys), 1)

    def reload(self):
        with self._lock:
            self._loaded = False
            self._exhausted.clear()
            self._idx = 0


_serper_pool = _SerperKeyPool()

# ── Oxylabs / COS config ───────────────────────────────────────────────────────
_OXYLABS_USER = os.environ.get("OXYLABS_USER", "")
_OXYLABS_PASS = os.environ.get("OXYLABS_PASS", "")
_COS_SECRET_ID  = os.environ.get("COS_SECRET_ID", "")
_COS_SECRET_KEY = os.environ.get("COS_SECRET_KEY", "")
_COS_BUCKET     = "search-agent-1259723048"
_COS_REGION     = "ap-tokyo"

# ── Proxy config — read from env, fallback to internal proxy, None = direct ───
def _get_proxies() -> dict:
    """Return proxy dict. Reads HTTPS_PROXY/HTTP_PROXY env vars first;
    if not set, use internal proxy; if internal proxy env var GEO_NO_PROXY=1,
    return empty dict (direct connection)."""
    if os.environ.get("GEO_NO_PROXY", ""):
        return {}
    proxy = (os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
             or os.environ.get("https_proxy") or os.environ.get("http_proxy"))
    if proxy:
        return {"https": proxy, "http": proxy}

def _get_direct_proxies() -> dict:
    """For COS / Oxylabs: use hk proxy which allows myqcloud.com access."""
    if os.environ.get("GEO_NO_PROXY", ""):
        return {}
    proxy = (os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
             or os.environ.get("https_proxy") or os.environ.get("http_proxy"))
    if proxy:
        return {"https": proxy, "http": proxy}

# ── Tool call parsing ──────────────────────────────────────────────────────────
_STRICT_RE = re.compile(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>", re.MULTILINE)
_LAZY_RE   = re.compile(r"<tool_call>\s*(\{.*)", re.DOTALL)

def _normalize_tool_call(call: dict) -> dict:
    """Infer missing 'name' or 'arguments' fields from common argument keys."""
    if not isinstance(call, dict):
        return call
    if "name" not in call:
        if "bbox_2d" in call:
            return {"name": "image_zoom_in_tool", "arguments": {"bbox_2d": call["bbox_2d"]}}
        if "query" in call:
            return {"name": "text_search_tool", "arguments": {"query": call["query"]}}
    if "arguments" not in call:
        if call.get("name") == "image_zoom_in_tool" and "bbox_2d" in call:
            call["arguments"] = {"bbox_2d": call["bbox_2d"]}
        elif call.get("name") == "image_search_tool" and "bbox_2d" in call:
            call["arguments"] = {"bbox_2d": call["bbox_2d"]}
        elif call.get("name") == "text_search_tool" and "query" in call:
            call["arguments"] = {"query": call["query"]}
        elif call.get("name") == "search_web" and "query" in call:
            call["arguments"] = {"query": call["query"]}
    return call

def parse_tool_calls_flexible(text: str) -> List[Dict[str, Any]]:
    """
    Parse <tool_call> JSON from model output.
    1. Strict: <tool_call>{...}</tool_call>
    2. Fallback: <tool_call>{...   (model stopped mid-generation, unclosed tag)
    """
    if not text:
        return []
    calls = []
    for m in _STRICT_RE.finditer(text):
        try:
            calls.append(_normalize_tool_call(json.loads(m.group(1))))
        except Exception:
            pass
    if calls:
        return calls
    m = _LAZY_RE.search(text)
    if m:
        raw = m.group(1)
        candidate = ""
        for ch in raw:
            candidate += ch
            if ch == "}":
                try:
                    calls.append(_normalize_tool_call(
                        json.loads(candidate.split("<")[0].strip())
                    ))
                    return calls
                except Exception:
                    pass
    return calls

# ── Image resize helpers ───────────────────────────────────────────────────────
def _round_by_factor(n: int, f: int) -> int:
    return round(n / f) * f

def _ceil_by_factor(n: int, f: int) -> int:
    return math.ceil(n / f) * f

def _floor_by_factor(n: int, f: int) -> int:
    return math.floor(n / f) * f

def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> Tuple[int, int]:
    """Resize to token-friendly dimensions for Qwen2-VL / Qwen3-VL."""
    h = max(factor, _round_by_factor(height, factor))
    w = max(factor, _round_by_factor(width, factor))
    if h * w > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h = _floor_by_factor(height / beta, factor)
        w = _floor_by_factor(width / beta, factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = _ceil_by_factor(height * beta, factor)
        w = _ceil_by_factor(width * beta, factor)
    return h, w

# ── Tool core functions ────────────────────────────────────────────────────────
def crop_tool_core(
    original_image_path: str,
    bbox_2d: List[float],
    label: Optional[str] = None,
    abs_scaling: float = 1.0,
    bbox_normalize: bool = False,
    crop_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Crop and resize an image region.

    bbox_2d: [x1, y1, x2, y2]
      bbox_normalize=True  → coords in [0, 1000], scaled to image pixels
      bbox_normalize=False → coords in pixels, multiplied by abs_scaling

    Returns dict: {success, crop_path, crop_b64, label, bbox}
    """
    left, top, right, bottom = bbox_2d

    with Image.open(original_image_path) as img:
        img = img.convert("RGB")
        W, H = img.size
        if bbox_normalize:
            left_px   = int(round(left   / 1000.0 * W))
            top_px    = int(round(top    / 1000.0 * H))
            right_px  = int(round(right  / 1000.0 * W))
            bottom_px = int(round(bottom / 1000.0 * H))
        else:
            left_px   = int(left   * abs_scaling)
            top_px    = int(top    * abs_scaling)
            right_px  = int(right  * abs_scaling)
            bottom_px = int(bottom * abs_scaling)
        cropped = img.crop((left_px, top_px, right_px, bottom_px))

    new_h, new_w = smart_resize(bottom_px - top_px, right_px - left_px)
    cropped = cropped.resize((new_w, new_h), resample=Image.BICUBIC)

    if crop_path is None:
        tmpdir = os.path.abspath(".temp/eval_crops")
        os.makedirs(tmpdir, exist_ok=True)
        crop_path = os.path.join(tmpdir, f"crop_{uuid.uuid4().hex}.jpg")
    os.makedirs(os.path.dirname(os.path.abspath(crop_path)), exist_ok=True)
    cropped.save(crop_path, format="JPEG")

    buf = io.BytesIO()
    cropped.save(buf, format="JPEG")
    crop_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {"success": True, "crop_path": crop_path, "crop_b64": crop_b64,
            "label": label, "bbox": bbox_2d}


def image_search_tool_core(
    original_image_path: str,
    bbox_2d: List[float],
    goal: Optional[str] = None,
    exclude_domains: list = None,
) -> Dict[str, Any]:
    """
    Reverse image search via oxylabs Google Lens API, with SQLite cache.

    Flow:
      1. Check SQLite cache (key = sha256(img_path|bbox))
      2. Crop the region
      3. Upload cropped image to COS (public URL needed by oxylabs)
      4. Call oxylabs reverse image search with the COS URL
      5. Cache and return top results (text + thumbnail base64 list)

    Args:
        original_image_path: Path to the source image.
        bbox_2d: [x1, y1, x2, y2] in [0,1000] normalized coords.
        goal: Optional description of what to look for.

    Returns dict: {success, text, results, thumbnails}
      text:       formatted search results string
      results:    list of {pos, title, link, source, thumbnail}
      thumbnails: list of base64-encoded thumbnail strings (may be empty strings)
    """
    import requests

    # 1. Check cache
    cached = _imsearch_cache_get(original_image_path, bbox_2d)
    if cached is not None:
        # Re-format text with goal if provided (results already cached)
        results = cached.get("results", [])
        # Apply exclude_domains filter even on cached results
        if exclude_domains:
            _excl = [d.lower() for d in exclude_domains]
            results = [r for r in results
                       if not any(e in (r.get("source", "") or r.get("link", "")).lower()
                                  for e in _excl)]
        bbox_ints = [int(round(v)) for v in bbox_2d]
        lines = []
        if goal:
            lines.append(f"Image search results for region {bbox_ints} (goal: {goal}):\n")
        else:
            lines.append(f"Image search results for region {bbox_ints}:\n")
        for r in results:
            lines.append(f"[{r['pos']}] {r['title']}")
            lines.append(f"    Source: {r['source']}")
            lines.append(f"    Link: {r['link']}")
        text = "\n".join(lines)
        return {"success": True, "text": text, "results": results,
                "thumbnails": [r.get("thumbnail", "") for r in results]}

    # 2. Crop the region
    crop_result = crop_tool_core(original_image_path, bbox_2d, bbox_normalize=True)
    crop_path = crop_result["crop_path"]

    # 3. Upload to COS
    try:
        from qcloud_cos import CosConfig, CosS3Client
        cos_config = CosConfig(Region=_COS_REGION, SecretId=_COS_SECRET_ID, SecretKey=_COS_SECRET_KEY,
                               Proxies=_get_direct_proxies() or None)
        cos_client = CosS3Client(cos_config)
        blob_name = f"eval_crops/{uuid.uuid4().hex}.jpg"
        with open(crop_path, "rb") as fp:
            cos_client.put_object(
                Bucket=_COS_BUCKET,
                Body=fp,
                Key=blob_name,
                ContentType="image/jpeg",
            )
        cos_url = f"https://{_COS_BUCKET}.cos.{_COS_REGION}.myqcloud.com/{blob_name}"
    except Exception as e:
        print(f"[ImageSearch ERROR] COS upload error: {e}", flush=True)
        return {"success": False, "text": f"Image search failed (COS upload error): {e}", "results": [], "thumbnails": []}

    # 4. Oxylabs reverse image search (with retry on proxy/5xx errors)
    _IMSEARCH_RETRIES = 3
    _last_err = None
    for _attempt in range(_IMSEARCH_RETRIES + 1):
        try:
            payload = {"source": "google_lens", "query": cos_url, "parse": True}
            response = requests.post(
                "https://realtime.oxylabs.io/v1/queries",
                auth=(_OXYLABS_USER, _OXYLABS_PASS),
                json=payload,
                timeout=60,
                proxies=_get_direct_proxies(),
            )
            # Transient 5xx — retry
            if 500 <= response.status_code < 600 and _attempt < _IMSEARCH_RETRIES:
                sleep_s = min(30, (2 ** (_attempt + 1)) + random.uniform(0, 1))
                print(f"[ImageSearch WARN] HTTP {response.status_code} "
                      f"retry {_attempt+1}/{_IMSEARCH_RETRIES} in {sleep_s:.1f}s", flush=True)
                time.sleep(sleep_s)
                continue
            response.raise_for_status()
            result_raw = response.json()
            organic = result_raw["results"][0]["content"]["results"].get("organic", [])[:15]
            # Filter out excluded domains from results
            if exclude_domains:
                _excl = [d.lower() for d in exclude_domains]
                organic = [x for x in organic
                           if not any(e in (x.get("domain", "") or x.get("url", "")).lower()
                                      for e in _excl)]
            results = [
                {
                    "pos":       x.get("pos", idx + 1),
                    "title":     x.get("title", ""),
                    "link":      x.get("url", ""),
                    "source":    x.get("domain", ""),
                    "thumbnail": x.get("url_thumbnail") or x.get("thumbnail", ""),
                }
                for idx, x in enumerate(organic)
            ]
            break  # success
        except (requests.exceptions.ProxyError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            _last_err = e
            if _attempt < _IMSEARCH_RETRIES:
                sleep_s = min(30, (2 ** (_attempt + 1)) + random.uniform(0, 1))
                print(f"[ImageSearch WARN] {type(e).__name__} "
                      f"retry {_attempt+1}/{_IMSEARCH_RETRIES} in {sleep_s:.1f}s", flush=True)
                time.sleep(sleep_s)
                continue
            print(f"[ImageSearch ERROR] Oxylabs API error (exhausted retries): {e}", flush=True)
            return {"success": False, "text": f"Image search failed (API error): {e}",
                    "results": [], "thumbnails": []}
        except Exception as e:
            print(f"[ImageSearch ERROR] Oxylabs API error: {e}", flush=True)
            return {"success": False, "text": f"Image search failed (API error): {e}",
                    "results": [], "thumbnails": []}
    else:
        # All retries exhausted without break
        print(f"[ImageSearch ERROR] Oxylabs API error (retries exhausted): {_last_err}", flush=True)
        return {"success": False, "text": f"Image search failed (API error): {_last_err}",
                "results": [], "thumbnails": []}

    # 5. Cache result (without goal, so it can be reused for any goal)
    _imsearch_cache_set(original_image_path, bbox_2d, {"results": results})

    # 6. Format output text
    bbox_ints = [int(round(v)) for v in bbox_2d]
    lines = []
    if goal:
        lines.append(f"Image search results for region {bbox_ints} (goal: {goal}):\n")
    else:
        lines.append(f"Image search results for region {bbox_ints}:\n")
    for r in results:
        lines.append(f"[{r['pos']}] {r['title']}")
        lines.append(f"    Source: {r['source']}")
        lines.append(f"    Link: {r['link']}")
    text = "\n".join(lines)

    return {"success": True, "text": text, "results": results,
            "thumbnails": [r.get("thumbnail", "") for r in results]}


def serper_image_search_tool_core(
    original_image_path: str,
    bbox_2d: List[float],
    goal: Optional[str] = None,
    api_key: str = "",
    exclude_domains: list = None,
) -> Dict[str, Any]:
    """
    Reverse image search via Serper Google Lens API (/lens), with SQLite cache.

    Flow:
      1. Check SQLite cache
      2. Crop the region
      3. Upload cropped image to COS (public URL)
      4. POST to https://google.serper.dev/lens with X-API-KEY header
      5. Cache and return top results

    Args:
        original_image_path: Path to the source image.
        bbox_2d: [x1, y1, x2, y2] in [0,1000] normalized coords.
        goal: Optional description of what to look for.
        api_key: Serper API key. Falls back to pool then SERPER_API_KEY env var.

    Returns dict: {success, text, results, thumbnails}
    """
    import requests

    # 1. Check cache
    cached = _serper_imsearch_cache_get(original_image_path, bbox_2d)
    if cached is not None:
        results = cached.get("results", [])
        # Apply exclude_domains filter on cached results too
        if exclude_domains:
            _excl = [d.lower() for d in exclude_domains]
            results = [r for r in results if not any(e in r.get("link", "").lower() for e in _excl)]
        # Re-number pos
        for i, r in enumerate(results):
            r["pos"] = i + 1
        bbox_ints = [int(round(v)) for v in bbox_2d]
        header = f"Image search results for region {bbox_ints}" + (f" (goal: {goal})" if goal else "") + ":\n"
        lines = [header]
        for r in results:
            lines.append(f"[{r['pos']}] {r['title']}")
            lines.append(f"    Source: {r['source']}")
            lines.append(f"    Link: {r['link']}")
        return {"success": True, "text": "\n".join(lines),
                "results": results,
                "thumbnails": [r.get("thumbnail", "") for r in results]}

    # 2. Crop
    crop_result = crop_tool_core(original_image_path, bbox_2d, bbox_normalize=True)
    crop_path = crop_result["crop_path"]

    # 3. Upload to COS
    try:
        from qcloud_cos import CosConfig, CosS3Client
        cos_config = CosConfig(Region=_COS_REGION, SecretId=_COS_SECRET_ID, SecretKey=_COS_SECRET_KEY,
                               Proxies=_get_direct_proxies() or None)
        cos_client = CosS3Client(cos_config)
        blob_name = f"eval_crops/{uuid.uuid4().hex}.jpg"
        with open(crop_path, "rb") as fp:
            cos_client.put_object(
                Bucket=_COS_BUCKET,
                Body=fp,
                Key=blob_name,
                ContentType="image/jpeg",
            )
        cos_url = f"https://{_COS_BUCKET}.cos.{_COS_REGION}.myqcloud.com/{blob_name}"
    except Exception as e:
        print(f"[SerperSearch ERROR] COS upload error: {e}", flush=True)
        return {"success": False, "text": f"Serper image search failed (COS upload error): {e}",
                "results": [], "thumbnails": []}

    # 4. Call Serper Lens API (with key rotation on 429)
    _proxies = _get_proxies()
    tried: set = set()
    while True:
        cur_key = api_key or _serper_pool.get_key() or os.environ.get("SERPER_API_KEY", "")
        if not cur_key or cur_key in tried:
            print(f"[SerperSearch ERROR] All keys exhausted for image={original_image_path}", flush=True)
            return {"success": False,
                    "text": "Serper image search failed: all API keys exhausted",
                    "results": [], "thumbnails": []}
        tried.add(cur_key)
        try:
            resp = requests.post(
                "https://google.serper.dev/lens",
                headers={"X-API-KEY": cur_key, "Content-Type": "application/json"},
                json={"url": cos_url},
                timeout=30,
                proxies=_proxies,
            )
            if resp.status_code == 429:
                _serper_pool.mark_exhausted(cur_key)
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:
            print(f"[SerperSearch ERROR] API error key={cur_key[:12]}... error={e}", flush=True)
            return {"success": False, "text": f"Serper image search failed: {e}",
                    "results": [], "thumbnails": []}

    # 5. Parse results
    # Serper Lens returns: {"organic": [{"title","link","source","imageUrl",...}], ...}
    organic = data.get("organic", [])[:15]
    # Apply exclude_domains filter
    if exclude_domains:
        _excl = [d.lower() for d in exclude_domains]
        organic = [x for x in organic if not any(e in x.get("link", "").lower() for e in _excl)]
    results = [
        {
            "pos":       idx + 1,
            "title":     x.get("title", ""),
            "link":      x.get("link", ""),
            "source":    x.get("source", ""),
            "thumbnail": x.get("imageUrl", ""),
        }
        for idx, x in enumerate(organic)
    ]

    # 6. Cache
    _serper_imsearch_cache_set(original_image_path, bbox_2d, {"results": results})

    # 7. Format
    bbox_ints = [int(round(v)) for v in bbox_2d]
    header = f"Image search results for region {bbox_ints}" + (f" (goal: {goal})" if goal else "") + ":\n"
    lines = [header]
    for r in results:
        lines.append(f"[{r['pos']}] {r['title']}")
        lines.append(f"    Source: {r['source']}")
        lines.append(f"    Link: {r['link']}")

    return {"success": True, "text": "\n".join(lines),
            "results": results,
            "thumbnails": [r.get("thumbnail", "") for r in results]}


def tavily_search_tool_core(
    query,
    api_key: str = "",
    topk: int = 10,
    timeout: int = 30,
    exclude_domains: list = None,
) -> Dict[str, Any]:
    """
    Web text search via Tavily API.

    Args:
        query:   Natural-language search query (string) or list of queries (parallel execution).
        api_key: Tavily API key. Falls back to _TAVILY_API_KEY module-level constant.
        topk:    Number of results to request per query (default 5).
        timeout: HTTP timeout in seconds.

    Returns dict: {success, text, results}
      text: formatted search results string ready for tool_response injection

    When query is a list, all queries are executed in parallel and results are
    concatenated. SQLite WAL cache is checked/written for each individual query.
    """
    import requests

    key = api_key or _tavily_pool.get_key() or _TAVILY_API_KEY
    if not key:
        return {"success": False, "text": "Tavily API key not set (env TAVILY_API_KEY)", "results": []}

    # Normalize to list
    if isinstance(query, str):
        queries = [query] if query.strip() else []
    elif isinstance(query, list):
        queries = [q for q in query if isinstance(q, str) and q.strip()]
    else:
        queries = []

    if not queries:
        return {"success": False, "text": "Empty query", "results": []}

    def _fetch_one(q: str) -> Dict[str, Any]:
        """Fetch a single query, checking cache first. Auto-rotates key on quota errors."""
        cached = _tavily_cache_get(q)
        if cached is not None:
            return cached

        _proxies = _get_proxies()

        # Try each available key (pool rotation on 429/432).
        # For transient proxy / network / 5xx errors: retry same key with backoff
        # (these errors are NOT key-related — the proxy is flaky).
        tried: set = set()
        while True:
            cur_key = api_key or _tavily_pool.get_key() or _TAVILY_API_KEY
            if not cur_key or cur_key in tried:
                print(f"[Tavily ERROR] All keys exhausted for query={q!r:.80}", flush=True)
                return {"success": False, "query": q,
                        "text": "Tavily search failed: all API keys exhausted", "results": []}
            tried.add(cur_key)

            # Per-key retry budget for transient errors (proxy 502, ConnectionError, 5xx).
            _TRANSIENT_RETRIES = 3
            transient_attempt = 0
            give_up_this_key = False
            while True:
                try:
                    payload = {
                        "query": q,
                        "api_key": cur_key,
                        "max_results": topk,
                        "include_answer": False,
                        "include_raw_content": False,
                    }
                    if exclude_domains:
                        payload["exclude_domains"] = exclude_domains
                    resp = requests.post(
                        "https://api.tavily.com/search",
                        json=payload,
                        timeout=timeout,
                        proxies=_proxies,
                    )
                    if resp.status_code in (429, 432):
                        # Quota exceeded — mark and retry with next key
                        _tavily_pool.mark_exhausted(cur_key)
                        give_up_this_key = True
                        break
                    # Transient 5xx: retry same key with backoff
                    if 500 <= resp.status_code < 600 and transient_attempt < _TRANSIENT_RETRIES:
                        transient_attempt += 1
                        sleep_s = min(30, (2 ** transient_attempt) + random.uniform(0, 1))
                        print(f"[Tavily WARN] HTTP {resp.status_code} query={q!r:.40} "
                              f"key={cur_key[:12]}... retry {transient_attempt}/{_TRANSIENT_RETRIES} "
                              f"in {sleep_s:.1f}s", flush=True)
                        time.sleep(sleep_s)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    raw_results = data.get("results", [])
                except (requests.exceptions.ProxyError,
                        requests.exceptions.ConnectionError,
                        requests.exceptions.Timeout) as e:
                    # Transient network / proxy error: retry same key with backoff
                    if transient_attempt < _TRANSIENT_RETRIES:
                        transient_attempt += 1
                        sleep_s = min(30, (2 ** transient_attempt) + random.uniform(0, 1))
                        print(f"[Tavily WARN] {type(e).__name__} query={q!r:.40} "
                              f"key={cur_key[:12]}... retry {transient_attempt}/{_TRANSIENT_RETRIES} "
                              f"in {sleep_s:.1f}s", flush=True)
                        time.sleep(sleep_s)
                        continue
                    print(f"[Tavily ERROR] query={q!r:.80} key={cur_key[:12]}... "
                          f"network_error={e}", flush=True)
                    return {"success": False, "query": q,
                            "text": f"Tavily search failed: {e}", "results": []}
                except Exception as e:
                    print(f"[Tavily ERROR] query={q!r:.80} key={cur_key[:12]}... error={e}", flush=True)
                    return {"success": False, "query": q, "text": f"Tavily search failed: {e}", "results": []}

                result = {"success": True, "query": q, "results": raw_results}
                _tavily_cache_set(q, result)
                return result

            if give_up_this_key:
                # break inner → continue outer while (try next key)
                continue

        # unreachable

    # Parallel execution for multiple queries
    if len(queries) == 1:
        all_results_data = [_fetch_one(queries[0])]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries)) as pool:
            all_results_data = list(pool.map(_fetch_one, queries))

    # Merge results
    all_raw_results = []
    parts = []
    overall_success = False
    for item in all_results_data:
        if item.get("success"):
            overall_success = True
        q = item.get("query", "")
        raw = item.get("results", [])
        all_raw_results.extend(raw)
        if raw:
            parts.append(f"Web search results for: {q}\n")
            for i, r in enumerate(raw, 1):
                title   = r.get("title", "")
                url     = r.get("url", "")
                content = r.get("content", "")
                parts.append(f"[{i}] {title}\nURL: {url}\n{content}\n")
        elif item.get("success"):
            parts.append(f"No results found for query: {q}\n")
        else:
            parts.append(item.get("text", f"Search failed for: {q}") + "\n")

    text = "\n".join(parts).strip() if parts else "No search results available."

    return {"success": overall_success, "text": text, "results": all_raw_results}


class ToolCallManager:
    """
    Per-image tool state tracker for the agent loop.

    Tracks call counts and enforces per-tool limits.
    The agent loop (round iteration) lives outside this class.

    Usage:
        manager = ToolCallManager(img_path)
        result  = manager.execute("image_zoom_in_tool", {"bbox_2d": [...]})
        # result["crop_b64"] → base64 cropped image

    Adding a new tool:
        1. Add execute_<tool>() method
        2. Add branch in execute() dispatch
    """

    def __init__(self, image_path: str, tavily_api_key: str = "", serper_api_key: str = "",
                 exclude_domains: list = None):
        self.image_path = image_path
        self.tavily_api_key = tavily_api_key or _TAVILY_API_KEY
        self.serper_api_key = serper_api_key or os.environ.get("SERPER_API_KEY", "")
        self.exclude_domains = exclude_domains or []
        self.crop_call_count   = 0
        self.search_call_count = 0
        self.tavily_call_count = 0
        self.serper_call_count = 0

    # ── unified dispatch ──
    def execute(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dispatch a tool call by name.
        Raises ValueError for unknown tools, Exception if limit exceeded.
        """
        if name == "image_zoom_in_tool":
            return self.execute_crop(arguments)
        elif name == "image_search_tool":
            return self.execute_search(arguments)
        elif name == "serper_search_tool":
            return self.execute_serper_search(arguments)
        elif name == "text_search_tool":
            return self.execute_text_search(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

    # ── zoom / crop ──
    def execute_crop(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if self.crop_call_count >= MAX_CROP_CALLS:
            raise Exception(f"Maximum crop calls ({MAX_CROP_CALLS}) exceeded")
        bbox = arguments.get("bbox_2d")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Invalid bbox_2d: {bbox}")
        # Handle nested [[x,y,x,y]] → [x,y,x,y]
        if isinstance(bbox[0], list):
            bbox = bbox[0]
        bbox = [int(float(v)) for v in bbox]
        result = crop_tool_core(self.image_path, bbox, bbox_normalize=True)
        self.crop_call_count += 1
        return result

    # ── image search ──
    def execute_search(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if self.search_call_count >= MAX_SEARCH_CALLS:
            raise Exception(f"Maximum search calls ({MAX_SEARCH_CALLS}) exceeded")
        goal = arguments.get("goal")

        # Support both single bbox and list of bboxes
        raw_bbox = arguments.get("bbox_2d")
        if not raw_bbox:
            raise ValueError(f"Invalid bbox_2d: {raw_bbox}")

        # Detect multi-bbox: [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]
        if isinstance(raw_bbox[0], list):
            bboxes = raw_bbox
        else:
            bboxes = [raw_bbox]

        bboxes = [[int(float(v)) for v in b] for b in bboxes]

        if len(bboxes) == 1:
            result = image_search_tool_core(self.image_path, bboxes[0], goal=goal,
                                            exclude_domains=self.exclude_domains if self.exclude_domains else None)
        else:
            # Parallel execution for multiple bboxes
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(bboxes)) as pool:
                futures = [pool.submit(image_search_tool_core, self.image_path, b, goal) for b in bboxes]
                sub_results = [f.result() for f in futures]
            # Merge
            all_results = []
            texts = []
            overall_success = False
            for sr in sub_results:
                if sr.get("success"):
                    overall_success = True
                all_results.extend(sr.get("results", []))
                texts.append(sr.get("text", ""))
            result = {
                "success": overall_success,
                "text": "\n\n".join(t for t in texts if t),
                "results": all_results,
                "crop_b64": sub_results[0].get("crop_b64") if sub_results else None,
            }

        self.search_call_count += 1
        return result

    # ── serper image search ──
    def execute_serper_search(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        MAX_SERPER_CALLS = 3
        if self.serper_call_count >= MAX_SERPER_CALLS:
            raise Exception(f"Maximum serper search calls ({MAX_SERPER_CALLS}) exceeded")
        goal = arguments.get("goal")

        raw_bbox = arguments.get("bbox_2d")
        if not raw_bbox:
            raise ValueError(f"Invalid bbox_2d: {raw_bbox}")

        if isinstance(raw_bbox[0], list):
            bboxes = raw_bbox
        else:
            bboxes = [raw_bbox]

        bboxes = [[int(float(v)) for v in b] for b in bboxes]

        if len(bboxes) == 1:
            result = serper_image_search_tool_core(
                self.image_path, bboxes[0], goal=goal, api_key=self.serper_api_key,
                exclude_domains=self.exclude_domains if self.exclude_domains else None,
            )
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(bboxes)) as pool:
                futures = [
                    pool.submit(serper_image_search_tool_core, self.image_path, b, goal, self.serper_api_key,
                                self.exclude_domains if self.exclude_domains else None)
                    for b in bboxes
                ]
                sub_results = [f.result() for f in futures]
            all_results = []
            texts = []
            overall_success = False
            for sr in sub_results:
                if sr.get("success"):
                    overall_success = True
                all_results.extend(sr.get("results", []))
                texts.append(sr.get("text", ""))
            result = {
                "success": overall_success,
                "text": "\n\n".join(t for t in texts if t),
                "results": all_results,
                "thumbnails": [],
            }

        self.serper_call_count += 1
        return result

    # ── text search (Tavily) ──
    def execute_text_search(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if self.tavily_call_count >= MAX_TAVILY_CALLS:
            raise Exception(f"Maximum tavily calls ({MAX_TAVILY_CALLS}) exceeded")
        query = arguments.get("query", "")
        # Support both single string and list of queries
        if isinstance(query, list):
            queries = [q for q in query if isinstance(q, str) and q.strip()]
        elif isinstance(query, str) and query.strip():
            queries = [query]
        else:
            raise ValueError(f"Invalid query: {query!r}")
        # Always use pool for key rotation; single-key fallback only if pool is empty
        result = tavily_search_tool_core(queries, api_key="", exclude_domains=self.exclude_domains or None)
        self.tavily_call_count += 1
        return result

    # ── stats ──
    @property
    def total_tool_calls(self) -> int:
        return self.crop_call_count + self.search_call_count + self.tavily_call_count + self.serper_call_count
