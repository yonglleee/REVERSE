#!/usr/bin/env python3
"""
test_serper.py — 测试 Serper Google Lens API
用法:
    python3 eval/test_serper.py
    python3 eval/test_serper.py --key <your_key>
    python3 eval/test_serper.py --image /path/to/image.jpg --bbox 200 200 800 800
"""
import argparse
import json
import os
import sys

# 确保能 import utils_agent_tool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_raw_api(key: str, image_url: str):
    """直接测试 Serper Lens API，不走 COS 流程，用公开图片 URL"""
    import requests
    proxies = {
        "https": "http://REMOVED_PROXY",
        "http":  "http://REMOVED_PROXY",
    }
    print(f"\n=== Raw Serper /lens test ===")
    print(f"  Key  : {key[:12]}...")
    print(f"  Image: {image_url}")
    resp = requests.post(
        "https://google.serper.dev/lens",
        headers={"X-API-KEY": key, "Content-Type": "application/json"},
        json={"url": image_url},
        timeout=30,
        proxies=proxies,
    )
    print(f"  HTTP : {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        organic = data.get("organic", [])
        print(f"  Organic results: {len(organic)}")
        for i, r in enumerate(organic[:5], 1):
            print(f"  [{i}] {r.get('title','')}")
            print(f"       {r.get('link','')}")
        return True
    else:
        print(f"  Error: {resp.text[:300]}")
        return False


def test_tool_core(key: str, image_path: str, bbox: list):
    """测试 serper_image_search_tool_core（走 COS 上传流程）"""
    from utils_agent_tool import serper_image_search_tool_core
    print(f"\n=== serper_image_search_tool_core test ===")
    print(f"  Image: {image_path}")
    print(f"  Bbox : {bbox}")
    result = serper_image_search_tool_core(image_path, bbox, goal="test location", api_key=key)
    print(f"  Success : {result['success']}")
    print(f"  Results : {len(result.get('results', []))}")
    print(f"  Text preview:\n{result.get('text','')[:500]}")
    return result["success"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", default="", help="Serper API key (default: from .env)")
    parser.add_argument("--image", default="", help="Local image path for full tool test")
    parser.add_argument("--bbox", nargs=4, type=float, default=[100, 100, 900, 900],
                        metavar=("X1", "Y1", "X2", "Y2"), help="Bbox in [0,1000] coords")
    parser.add_argument("--raw-only", action="store_true", help="Only run raw API test")
    args = parser.parse_args()

    # Load key from .env if not provided
    if not args.key:
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("SERPER_API_KEY="):
                        args.key = line[len("SERPER_API_KEY="):].split("#")[0].strip()
                        break
    if not args.key:
        print("ERROR: No Serper API key found. Use --key or add SERPER_API_KEY= to eval/.env")
        sys.exit(1)

    # Always run raw API test (uses a public Wikipedia image)
    TEST_IMAGE_URL = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/"
        "Eiffel_Tower%2C_Paris_21_October_2011.jpg/800px-Eiffel_Tower%2C_Paris_21_October_2011.jpg"
    )
    ok = test_raw_api(args.key, TEST_IMAGE_URL)

    if not args.raw_only and args.image:
        ok2 = test_tool_core(args.key, args.image, args.bbox)
        ok = ok and ok2

    print(f"\n{'✅ All tests passed' if ok else '❌ Some tests failed'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
