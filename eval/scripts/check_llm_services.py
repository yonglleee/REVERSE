#!/usr/bin/env python3
"""
check_llm_services.py — Check availability and health of all LLM services
used by the eval pipeline (Kimi K2d6, Qwen3.5-122B, etc.)

For each model:
  1. Query Starlink broker to discover all deployed machines
  2. Check vLLM /health endpoint on each machine
  3. Send a simple test prompt to verify generation works
  4. Report status, latency, and GPU utilization (if available)

Usage:
  python eval/scripts/check_llm_services.py
  python eval/scripts/check_llm_services.py --model kimi_k2d6
  python eval/scripts/check_llm_services.py --model all --verbose
"""

import argparse
import json
import random
import socket
import struct
import sys
import time
from datetime import datetime
from typing import Dict, List

try:
    import requests
except ImportError:
    print("pip install requests")
    sys.exit(1)

# ── Broker config (from llm_client.py) ──────────────────────────────────────
BROKER_URL = "http://astrastarlinkbroker.production.polaris:12534/OPEN_API_GetMachineListByApp"
VLLM_PORT = 8000

BROKER_APPIDS = {
    "kimi_k2d6": "kimi_k2d6",
    "kimi_k2d5": "kimi_k2d5",
    "qwen122b":  "Qwen3d5_122B_A10B",
    "qwen397b":  "qwen3d5_397b_a17b",
}

ALL_MODELS = list(BROKER_APPIDS.keys())


def int_to_ip(ip_int: int) -> str:
    return socket.inet_ntoa(struct.pack("!I", ip_int))


def get_machines(appid: str) -> List[Dict]:
    """Query broker for all machines of a given appid."""
    try:
        res = requests.post(BROKER_URL, json={"appid": appid}, timeout=10)
        res.raise_for_status()
        data = res.json()
        machines = data.get("machine_list", {}).get("list", [])
        return machines
    except Exception as e:
        return []


def check_health(ip: str, port: int = VLLM_PORT, timeout: float = 5.0) -> dict:
    """Check vLLM /health endpoint."""
    url = f"http://{ip}:{port}/health"
    try:
        t0 = time.time()
        r = requests.get(url, timeout=timeout)
        latency = time.time() - t0
        return {"ok": r.status_code == 200, "status": r.status_code, "latency_ms": round(latency * 1000)}
    except requests.exceptions.ConnectTimeout:
        return {"ok": False, "status": "timeout", "latency_ms": -1}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "status": "conn_refused", "latency_ms": -1}
    except Exception as e:
        return {"ok": False, "status": str(e)[:60], "latency_ms": -1}


def check_generate(ip: str, model_name: str, port: int = VLLM_PORT, timeout: float = 30.0) -> dict:
    """Send a simple generation request to verify the model works."""
    url = f"http://{ip}:{port}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Say 'hello' in one word."}],
        "max_tokens": 16,
        "temperature": 0,
    }
    # Qwen3 系列需关闭 thinking 模式，否则 content 为 None
    if "qwen" in model_name.lower():
        payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    try:
        t0 = time.time()
        r = requests.post(url, json=payload, timeout=timeout)
        latency = time.time() - t0
        if r.status_code == 200:
            data = r.json()
            content = data["choices"][0]["message"].get("content") or \
                      data["choices"][0]["message"].get("reasoning_content") or ""
            return {"ok": True, "text": content.strip()[:50], "latency_ms": round(latency * 1000)}
        else:
            return {"ok": False, "text": f"HTTP {r.status_code}: {r.text[:100]}", "latency_ms": round(latency * 1000)}
    except Exception as e:
        return {"ok": False, "text": str(e)[:80], "latency_ms": -1}


def check_model_info(ip: str, port: int = VLLM_PORT, timeout: float = 5.0) -> dict:
    """Get model info from vLLM /v1/models endpoint."""
    url = f"http://{ip}:{port}/v1/models"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            models = [m["id"] for m in data.get("data", [])]
            return {"ok": True, "models": models}
        return {"ok": False, "models": []}
    except Exception:
        return {"ok": False, "models": []}


def check_one_model(model: str, verbose: bool = False):
    appid = BROKER_APPIDS[model]
    print(f"\n{'='*60}")
    print(f"  Model: {model}  (appid: {appid})")
    print(f"{'='*60}")

    # 1. Query broker
    machines = get_machines(appid)
    if not machines:
        print(f"  [BROKER] FAIL — no machines returned (service may be down)")
        return {"model": model, "n_machines": 0, "n_healthy": 0, "n_generate_ok": 0}

    ips = []
    for m in machines:
        try:
            ip = int_to_ip(m["attr"]["ip"])
            ips.append(ip)
        except (KeyError, struct.error):
            pass

    ips = sorted(set(ips))
    print(f"  [BROKER] OK — {len(ips)} machine(s) found")

    # 2. Check each machine
    n_healthy = 0
    n_generate_ok = 0

    for ip in ips:
        health = check_health(ip)
        status_str = "OK" if health["ok"] else f"FAIL ({health['status']})"
        line = f"  [{ip}] health: {status_str} ({health['latency_ms']}ms)"

        if health["ok"]:
            n_healthy += 1
            # Check model info
            info = check_model_info(ip)
            if info["ok"] and info["models"]:
                line += f"  model: {info['models'][0]}"

            # Test generation
            actual_model = info["models"][0] if info["ok"] and info["models"] else appid
            gen = check_generate(ip, model_name=actual_model)
            gen_str = "OK" if gen["ok"] else "FAIL"
            line += f"  generate: {gen_str} ({gen['latency_ms']}ms)"
            if gen["ok"]:
                n_generate_ok += 1
                if verbose:
                    line += f"  response: \"{gen['text']}\""
            else:
                line += f"  error: {gen['text']}"

        print(line)

    # Summary
    print(f"\n  Summary: {n_healthy}/{len(ips)} healthy, {n_generate_ok}/{len(ips)} generating OK")
    return {"model": model, "n_machines": len(ips), "n_healthy": n_healthy, "n_generate_ok": n_generate_ok}


def main():
    parser = argparse.ArgumentParser(description="Check LLM service availability")
    parser.add_argument("--model", default="all",
                        help=f"Model to check: {', '.join(ALL_MODELS)} or 'all' (default: all)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show response text")
    args = parser.parse_args()

    print(f"LLM Service Health Check — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.model == "all":
        models = ALL_MODELS
    else:
        models = [m.strip() for m in args.model.split(",")]
        for m in models:
            if m not in BROKER_APPIDS:
                print(f"Unknown model: {m}. Available: {ALL_MODELS}")
                sys.exit(1)

    results = []
    for model in models:
        r = check_one_model(model, verbose=args.verbose)
        results.append(r)

    # Final summary table
    print(f"\n{'='*60}")
    print(f"  {'Model':<15} {'Machines':>8} {'Healthy':>8} {'GenOK':>8}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8}")
    for r in results:
        print(f"  {r['model']:<15} {r['n_machines']:>8} {r['n_healthy']:>8} {r['n_generate_ok']:>8}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
