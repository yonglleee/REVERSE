#!/usr/bin/env python3
from __future__ import annotations

"""
check_api_quota.py — 检查 Tavily / Oxylabs / COS / Serper API 的可用额度与连通性

直接复用 utils_agent_tool.py 中的常量、key-loader 和 quota-checker，无重复代码。

用法:
    python3 eval/scripts/check_api_quota.py           s/check_api_quota.py --auto-manage      # 自动注释耗尽 + 恢复可用 key
    python3 eval/scripts/check_api_quota.py --mark-exhausted   # 只注释耗尽的 key
    python3 eval/scripts/check_api_quota.py --restore          # 只取消注释已恢复的 key
    python3 eval/scripts/check_api_quota.py --only-tavily      # 只查 Tavily
    python3 eval/scripts/check_api_quota.py --show-exhausted   # 也打印耗尽的 key
    python3 eval/scripts/check_api_quota.py --workers 40       # 并发数（默认 20）

自动管理模式 (--auto-manage):
    - 每次运行检查所有 active key，耗尽的自动加 # 注释
    - 同时检查所有被 # 注释的 key，配额恢复后自动取消注释
    - 建议配合 cron 定时运行，实现全自动管理
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ── 把 eval/ 加入 sys.path，直接 import utils_agent_tool ──────────────────────
_EVAL_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

import utils_agent_tool as _u  # noqa: E402  # 复用所有常量与工具函数

# ── 默认 .env 路径 ─────────────────────────────────────────────────────────────
_DEFAULT_ENV = os.path.join(_EVAL_DIR, ".env")

# ── 代理（与 utils_agent_tool.py 一致）────────────────────────────────────────
def _build_proxies() -> dict:
    if os.environ.get("GEO_NO_PROXY", ""):
        return {}
    proxy = (os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
             or os.environ.get("https_proxy") or os.environ.get("http_proxy"))
    if proxy:
        return {"https": proxy, "http": proxy}
    return {"https": "http://REMOVED_PROXY",
            "http":  "http://REMOVED_PROXY"}

_PROXIES = _build_proxies()

# ── 测试图片（Oxylabs / Serper Google Lens 测试用）────────────────────────────
_TEST_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/"
    "PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
)


# =============================================================================
# Tavily — 直接复用 _check_tavily_key_quota + _load_tavily_keys_from_env
# =============================================================================

def check_tavily(key: str) -> dict:
    """
    调用 utils_agent_tool._check_tavily_key_quota，
    同时返回 plan_usage / plan_limit / plan_name / reset 信息供展示。
    """
    from datetime import datetime, timedelta
    short = key[:20] + "..."
    try:
        resp = requests.get(
            "https://api.tavily.com/usage",
            headers={"Authorization": f"Bearer {key}"},
            timeout=15,
            proxies=_PROXIES,
        )
        if resp.status_code == 200:
            acct   = resp.json().get("account", {})
            usage  = acct.get("plan_usage", 0)
            limit  = acct.get("plan_limit", 0)
            plan   = acct.get("current_plan", "?")
            remain = (limit - usage) if isinstance(limit, int) and isinstance(usage, int) else -1
            status = "exhausted" if remain <= 0 else "ok"
            icon   = "❌ EXHAUSTED" if remain <= 0 else "✅ OK"
            # Tavily free plans reset monthly on the 1st
            now = datetime.now()
            if now.month == 12:
                reset_date = datetime(now.year + 1, 1, 1)
            else:
                reset_date = datetime(now.year, now.month + 1, 1)
            days_to_reset = (reset_date - now).days
            reset_str = f"resets {reset_date.strftime('%Y-%m-%d')} ({days_to_reset}d)"
            detail = f"plan={plan}  used={usage}/{limit}  remaining={remain}  {reset_str}"
            return dict(key=key, key_short=short, icon=icon, status=status,
                        remaining=remain, days_to_reset=days_to_reset,
                        reset_date=reset_date.strftime('%Y-%m-%d'), detail=detail)
        elif resp.status_code in (429, 432):
            return dict(key=key, key_short=short, icon="❌ QUOTA EXCEEDED",
                        status="exhausted", remaining=0, detail=resp.text[:100])
        elif resp.status_code == 401:
            return dict(key=key, key_short=short, icon="❌ 401 INVALID",
                        status="invalid",   remaining=-1, detail=resp.text[:100])
        else:
            return dict(key=key, key_short=short, icon=f"⚠️  HTTP {resp.status_code}",
                        status="error",     remaining=-1, detail=resp.text[:100])
    except Exception as e:
        return dict(key=key, key_short=short, icon="❌ ERROR",
                    status="error", remaining=-1, detail=str(e))


# =============================================================================
# Oxylabs — 用 utils_agent_tool._OXYLABS_USER/PASS 常量
# =============================================================================

def check_oxylabs() -> dict:
    try:
        resp = requests.post(
            "https://realtime.oxylabs.io/v1/queries",
            auth=(_u._OXYLABS_USER, _u._OXYLABS_PASS),
            json={"source": "google_lens", "query": _TEST_IMAGE_URL, "parse": True},
            timeout=30,
            proxies=_PROXIES,
        )
        if resp.status_code == 200:
            n = len(resp.json().get("results", []))
            return {"icon": "✅ OK", "detail": f"results={n}"}
        return {"icon": f"❌ HTTP {resp.status_code}", "detail": resp.text[:200]}
    except Exception as e:
        return {"icon": "❌ ERROR", "detail": str(e)}


# =============================================================================
# COS — 用 utils_agent_tool._COS_* 常量
# =============================================================================

def check_cos() -> dict:
    try:
        from qcloud_cos import CosConfig, CosS3Client  # type: ignore
        cfg = CosConfig(
            Region    =_u._COS_REGION,
            SecretId  =_u._COS_SECRET_ID,
            SecretKey =_u._COS_SECRET_KEY,
            Proxies   =_PROXIES or None,
        )
        CosS3Client(cfg).list_objects(Bucket=_u._COS_BUCKET, MaxKeys=1)
        return {"icon": "✅ OK",
                "detail": f"bucket={_u._COS_BUCKET}  region={_u._COS_REGION}"}
    except ImportError:
        return {
            "icon": "⚠️  NOT INSTALLED",
            "detail": (
                "qcloud_cos 未安装，image_search 无法上传图片到 COS。\n"
                "  修复：pip install cos-python-sdk-v5"
            ),
        }
    except Exception as e:
        return {"icon": "❌ ERROR", "detail": str(e)}


# =============================================================================
# Serper — 用 utils_agent_tool._load_serper_keys_from_env
# =============================================================================

def check_serper(key: str) -> dict:
    short = key[:20] + "..."
    try:
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": key, "Content-Type": "application/json"},
            json={"q": "eiffel tower location", "num": 1},
            timeout=15,
            proxies=_PROXIES,
        )
        if resp.status_code == 200:
            data   = resp.json()
            n      = len(data.get("organic", []))
            credit = data.get("credits", "?")
            return {"key_short": short, "icon": "✅ OK",
                    "detail": f"organic_results={n}  credits_used={credit}"}
        elif resp.status_code == 403:
            return {"key_short": short, "icon": "❌ 403 INVALID KEY", "detail": resp.text[:100]}
        elif resp.status_code == 429:
            return {"key_short": short, "icon": "❌ 429 QUOTA EXCEEDED", "detail": resp.text[:100]}
        return {"key_short": short, "icon": f"⚠️  HTTP {resp.status_code}", "detail": resp.text[:100]}
    except Exception as e:
        return {"key_short": short, "icon": "❌ ERROR", "detail": str(e)}


# =============================================================================
# --mark-exhausted / --restore：将耗尽的 key 注释、将恢复的 key 取消注释
# =============================================================================

def _load_all_tavily_keys_from_env(env_path: str) -> tuple[list[str], list[tuple[str, str]]]:
    """
    返回 (active_keys, commented_pairs)。
    active_keys: 未被注释的 key 列表。
    commented_pairs: [(full_line, raw_key), ...] 所有被 # TAVILY_API_KEY= 注释掉的行及其原始 key。
    """
    with open(env_path) as f:
        lines = f.readlines()
    active, commented = [], []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("TAVILY_API_KEY="):
            k = stripped[len("TAVILY_API_KEY="):].split("#")[0].strip().strip('"').strip("'")
            if k:
                active.append(k)
        elif stripped.startswith("#") and "TAVILY_API_KEY=" in stripped:
            # 去掉开头的 # 和可能的空格，提取 key
            inner = stripped.lstrip("#").strip()
            if inner.startswith("TAVILY_API_KEY="):
                k = inner[len("TAVILY_API_KEY="):].split("#")[0].strip().strip('"').strip("'")
                if k:
                    commented.append((stripped, k))
    return active, commented


def mark_exhausted_keys(env_path: str, exhausted_keys: set) -> int:
    """注释掉 .env 里所有在 exhausted_keys 中的活跃 TAVILY_API_KEY，返回修改行数。"""
    with open(env_path) as f:
        lines = f.readlines()
    n_changed, new_lines = 0, []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("TAVILY_API_KEY="):
            k = stripped[len("TAVILY_API_KEY="):].split("#")[0].strip().strip('"').strip("'")
            if k and k in exhausted_keys:
                new_lines.append("# " + line.lstrip())
                n_changed += 1
                continue
        new_lines.append(line)
    with open(env_path, "w") as f:
        f.writelines(new_lines)
    return n_changed


def restore_available_keys(env_path: str, restored_keys: set) -> int:
    """取消 .env 里已注释但配额已恢复的 TAVILY_API_KEY 的注释，返回修改行数。"""
    with open(env_path) as f:
        lines = f.readlines()
    n_changed, new_lines = 0, []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") and "TAVILY_API_KEY=" in stripped:
            inner = stripped.lstrip("#").strip()
            if inner.startswith("TAVILY_API_KEY="):
                k = inner[len("TAVILY_API_KEY="):].split("#")[0].strip().strip('"').strip("'")
                if k and k in restored_keys:
                    # 去掉注释前缀，恢复为活跃行；保留原始缩进风格（无缩进）
                    new_lines.append(inner + "\n")
                    n_changed += 1
                    continue
        new_lines.append(line)
    with open(env_path, "w") as f:
        f.writelines(new_lines)
    return n_changed


# =============================================================================
# main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Check Tavily / Oxylabs / COS / Serper API quota",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--env", default=_DEFAULT_ENV,
                        help=f"Path to .env (default: {_DEFAULT_ENV})")
    parser.add_argument("--mark-exhausted", action="store_true",
                        help="Auto-comment exhausted Tavily keys in .env")
    parser.add_argument("--restore", dest="restore", action="store_true", default=True,
                        help="Auto-uncomment restored Tavily keys in .env (keys that were "
                             "commented but now have available quota). Default: ON.")
    parser.add_argument("--no-restore", dest="restore", action="store_false",
                        help="Disable the default --restore behavior.")
    parser.add_argument("--auto-manage", action="store_true",
                        help="Shorthand for --mark-exhausted --restore: both comment out "
                             "exhausted AND restore recovered keys")
    parser.add_argument("--show-exhausted", action="store_true",
                        help="Also print exhausted/invalid keys (default: only show ok)")
    parser.add_argument("--only-tavily", action="store_true",
                        help="Skip Oxylabs / COS / Serper checks")
    parser.add_argument("--no-oxylabs", action="store_true")
    parser.add_argument("--no-cos",     action="store_true")
    parser.add_argument("--no-serper",  action="store_true")
    parser.add_argument("--workers", type=int, default=20,
                        help="Parallel workers for Tavily checks (default: 20)")
    args = parser.parse_args()

    if args.only_tavily:
        args.no_oxylabs = args.no_cos = args.no_serper = True

    print(f"  .env: {args.env}\n")

    # ── Tavily ────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  Tavily API Keys")
    print("=" * 70)

    # 复用 utils_agent_tool 的 key loader（只取 active 行）
    all_keys = _u._load_tavily_keys_from_env(args.env)

    # 同时加载被注释的 key（用于 --restore 模式）
    _, commented_pairs = _load_all_tavily_keys_from_env(args.env)
    commented_keys = [k for (_, k) in commented_pairs]

    if not all_keys and not (args.restore or args.auto_manage):
        print("  ⚠️  No active Tavily keys found in .env\n")
    else:
        results_map: dict = {}
        with ThreadPoolExecutor(max_workers=min(len(all_keys), args.workers)) as ex:
            fut_map = {ex.submit(check_tavily, k): k for k in all_keys}
            for fut in as_completed(fut_map):
                results_map[fut_map[fut]] = fut.result()

        n_ok = n_exhausted = n_error = 0
        total_remaining   = 0
        exhausted_keys    = set()
        ok_results        = []

        for key in all_keys:
            r = results_map[key]
            if r["status"] == "ok":
                n_ok += 1
                total_remaining += max(r["remaining"], 0)
                ok_results.append(r)
            elif r["status"] == "exhausted":
                n_exhausted += 1
                exhausted_keys.add(key)
            else:
                n_error += 1

        if ok_results:
            print(f"\n  ── ✅ Available ({n_ok}) ──")
            for r in sorted(ok_results, key=lambda x: -x["remaining"]):
                print(f"  {r['key_short']:<28}  {r['detail']}")

        if args.show_exhausted:
            exhausted_results = [results_map[k] for k in all_keys if results_map[k]["status"] == "exhausted"]
            if exhausted_results:
                print(f"\n  ── ❌ Exhausted ({n_exhausted}) ──")
                for r in exhausted_results:
                    print(f"  {r['key_short']:<28}  {r['detail']}")
            error_results = [results_map[k] for k in all_keys if results_map[k]["status"] == "error"]
            if error_results:
                print(f"\n  ── ⚠️  Error ({n_error}) ──")
                for r in error_results:
                    print(f"  {r['key_short']:<28}  {r['detail']}")

        print()
        print(f"  Active keys    : {len(all_keys)}")
        print(f"  ✅ OK          : {n_ok}  (total remaining quota: {total_remaining})")
        print(f"  ❌ Exhausted   : {n_exhausted}")
        if n_error:
            print(f"  ⚠️  Error      : {n_error}")

        # Reset date info
        if ok_results and ok_results[0].get("days_to_reset") is not None:
            d = ok_results[0]["days_to_reset"]
            rd = ok_results[0]["reset_date"]
            print(f"  🔄 Next reset  : {rd} ({d} days from now)")

        # ── 自动管理：注释耗尽 + 恢复可用 ─────────────────────────────────
        do_mark = args.mark_exhausted or args.auto_manage
        do_restore = args.restore or args.auto_manage

        if do_mark:
            n = mark_exhausted_keys(args.env, exhausted_keys)
            if n:
                print(f"\n  📝 --mark-exhausted: commented out {n} key(s) in {args.env}")
            else:
                print("\n  📝 --mark-exhausted: nothing to comment out")

        if do_restore and commented_keys:
            print(f"\n  🔍 Checking {len(commented_keys)} commented key(s) for restoration ...")
            restored_set: set = set()
            with ThreadPoolExecutor(max_workers=min(len(commented_keys), args.workers)) as ex:
                fut_map = {ex.submit(check_tavily, k): k for k in commented_keys}
                for fut in as_completed(fut_map):
                    r = fut.result()
                    if r["status"] == "ok":
                        restored_set.add(fut_map[fut])
                        print(f"    ✅ {r['key_short']:<28}  restored!  ({r['detail']})")
            if restored_set:
                n = restore_available_keys(args.env, restored_set)
                print(f"\n  📝 --restore: uncommented {n} key(s) in {args.env}")
            else:
                print("    (none recovered yet)")
        elif do_restore and not commented_keys:
            print("\n  📝 --restore: no commented keys found")

    print()

    # ── Oxylabs ───────────────────────────────────────────────────────────────
    if not args.no_oxylabs:
        print("=" * 70)
        print("  Oxylabs Google Lens")
        print("=" * 70)
        r = check_oxylabs()
        print(f"  {r['icon']}")
        if r["detail"]:
            print(f"  {r['detail']}")
        print()

    # ── COS ───────────────────────────────────────────────────────────────────
    if not args.no_cos:
        print("=" * 70)
        print("  Tencent COS  (image upload for image_search)")
        print("=" * 70)
        r = check_cos()
        print(f"  {r['icon']}")
        for line in r["detail"].splitlines():
            print(f"  {line}")
        print()

    # ── Serper ────────────────────────────────────────────────────────────────
    if not args.no_serper:
        print("=" * 70)
        print("  Serper Google Lens")
        print("=" * 70)
        # 复用 utils_agent_tool 的 key loader
        serper_keys = _u._load_serper_keys_from_env(args.env)
        if not serper_keys:
            print("  ⚠️  No SERPER_API_KEY found in .env")
        else:
            for key in serper_keys:
                r = check_serper(key)
                print(f"  {r['icon']}")
                print(f"  {r['key_short']}")
                if r["detail"]:
                    print(f"  {r['detail']}")
        print()


if __name__ == "__main__":
    main()
