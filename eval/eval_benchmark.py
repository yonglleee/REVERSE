"""
eval_benchmark.py — Evaluate geo-localization on Im2GPS3K / YFCC4K benchmarks.

Two evaluation modes (--mode):
  single  — single-turn, one inference call, no tool use
  agent   — multi-turn agent loop with ToolCallManager tool dispatch

Two built-in prompt variants (select with --notool):
  default  — tool prompt: verbatim system from spotsft_multiturn_w_tool.py
             (build_system_prompt(with_search=False) + instruction_following)
  --notool — no-tool prompt: same system without tool section, same user

Override either with --system_prompt / --user_prompt (@/path/to/file supported).

Usage:
  # no-tool baseline
  python3 eval_benchmark.py --mode single --notool --tag qwen3vl4b_notool

  # trained agent (default prompts + tool schema)
  python3 eval_benchmark.py --mode agent --use_tools --tag qwen3vl4b_trained

  # Or use shell scripts (recommended):
  bash run_eval_benchmark.sh
  bash run_eval_rlvr_steps.sh
"""

import argparse
import asyncio
import base64
import io
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path


def _ensure_deps():
    """Auto-install missing Python dependencies (mirrors Tencent PyPI for speed)."""
    _pip = [
        sys.executable, "-m", "pip", "install", "-q",
        "-i", "https://mirrors.tencent.com/pypi/simple",
        "--extra-index-url", "https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple",
    ]
    _deps = [
        ("qcloud_cos",  "cos-python-sdk-v5"),
        ("aiohttp",     "aiohttp"),
        ("pandas",      "pandas"),
        ("PIL",         "Pillow"),
        ("tqdm",        "tqdm"),
        ("openai",      "openai"),
        ("requests",    "requests"),
    ]
    for import_name, pkg_name in _deps:
        try:
            __import__(import_name)
        except ImportError:
            print(f"[eval_benchmark] Installing {pkg_name} ...", flush=True)
            subprocess.check_call(_pip + [pkg_name])

_ensure_deps()


import aiohttp
import pandas as pd
from PIL import Image
from tqdm.asyncio import tqdm as atqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_agent_tool import parse_tool_calls_flexible, ToolCallManager

# llm_client.py (235B / Qwen3.5) — lazy import inside run() when --model is set
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "../data_pipeline/server"))

# ── Paths ──────────────────────────────────────────────────────────────────────
BENCHMARK_DIR  = "/mnt/sh/mmvision/home/jonahli/data_agent/benchmark"
CSV_PATH       = f"{BENCHMARK_DIR}/im2gps3k_CLEAN.csv"
IMG_DIR        = f"{BENCHMARK_DIR}/im2gps3ktest"
YFCC4K_TXT     = f"{BENCHMARK_DIR}/yfcc4k.txt"
YFCC4K_IMG_DIR = f"{BENCHMARK_DIR}/yfcc4k"
OUTPUT_DIR     = "/mnt/sh/mmvision/home/jonahli/save/agent/eval"

# ── Tool schema ────────────────────────────────────────────────────────────────
# Passed to SGLang API as tools= to trigger {%- if tools %} in custom_chat_template
# Descriptions match training tool_config exactly (geoloc_spot_zoom_imgsearch_tool_config_searchreward.yaml)
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "image_zoom_in_tool",
        "description": (
            "Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label. "
            "Use this to examine details more closely before making a geolocation prediction.\n"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "bbox_2d": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "The bounding box of the region to zoom in, as [x1, y1, x2, y2], "
                        "where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner. "
                        "Values are normalized coordinates in the range [0, 1000] relative to the current image.\n"
                    )
                },
                "label": {
                    "type": "string",
                    "description": "The name or label of the object in the bounding box (optional)."
                }
            },
            "required": ["bbox_2d"]
        }
    }
}


TOOL_SCHEMA_SEARCH = {
    "type": "function",
    "function": {
        "name": "image_search_tool",
        "description": (
            "Reverse image search using a cropped region of the image. "
            "Crop to the most geographically distinctive subregion (landmark, building facade, sign, or unique scene element) when possible. "
            "Use the full image only if the entire scene is the landmark. "
            "Returns matching web pages with titles and sources that may reveal the location.\n"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "bbox_2d": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "The bounding box of the region to search, as [x1, y1, x2, y2], "
                        "where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner. "
                        "Values are normalized coordinates in the range [0, 1000] relative to the original image.\n"
                    )
                },
                "goal": {
                    "type": "string",
                    "description": (
                        "Optional description of what you are trying to identify with this search, "
                        "e.g. 'identify this landmark' or 'find the location of this building'.\n"
                    )
                }
            },
            "required": ["bbox_2d"]
        }
    }
}

TOOL_SCHEMAS_ALL = [TOOL_SCHEMA, TOOL_SCHEMA_SEARCH]

TOOL_SCHEMA_TAVILY = {
    "type": "function",
    "function": {
        "name": "text_search_tool",
        "description": (
            "Search the web using natural language text query or queries. "
            "Use this to look up location names, landmarks, signs, or any text clues "
            "observed in the image to find information about the location.\n"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "oneOf": [
                        {
                            "type": "string",
                            "description": (
                                "A natural language search query. "
                                "e.g. 'Eiffel Tower Paris location' or 'street sign \"Rue de Rivoli\" Paris'.\n"
                            )
                        },
                        {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Multiple search queries to run in parallel. "
                                "e.g. [\"Arc de Triomphe location\", \"Champs-Élysées Paris\"].\n"
                            )
                        }
                    ],
                    "description": "Search query string or list of query strings (parallel execution)."
                }
            },
            "required": ["query"]
        }
    }
}

TOOL_SCHEMAS_TAVILY = [TOOL_SCHEMA, TOOL_SCHEMA_SEARCH, TOOL_SCHEMA_TAVILY]

# ── Prompt constants ────────────────────────────────────────────────────────────
# Verbatim from spotsft_multiturn_w_tool.py — keeps eval in-distribution with training.
#
# SYSTEM_TOOL   = build_system_prompt(with_search=False)
# SYSTEM_NOTOOL = same base without tool section
# USER_PROMPT   = user_q + ". " + instruction_following
#   user_q is the SpotSFT dataset user turn, confirmed from validation_output/0.jsonl:
#   "Analyze the architectural styles, vegetation, street infrastructure,
#    and cultural markers in this image. Based on these visual cues, determine the location."

SYSTEM_TOOL = (
    "You are a geolocation expert. You are given an image and you need to identify its location. "
    "Reason step by step before making your prediction. "
    "You have access to the following tools:\n"
    "1. `image_zoom_in_tool`: Zoom in on a specific region of the image to examine details such as "
    "signs, license plates, storefronts, or architectural features more closely.\n"
    "2. `image_search_tool`: Perform a reverse image search on a specific region of the image "
    "(same bbox as zoom). Returns matching web pages that may reveal the location. "
    "Use this after zooming in on a distinctive region.\n\n"
    "You SHOULD use `image_zoom_in_tool` to inspect key details before predicting. "
    "You SHOULD use `image_search_tool` on regions with distinctive landmarks, signs, or text.\n\n"
    "IMPORTANT: Output only ONE tool call per response. Wait for the result before calling the next tool.\n\n"
    "Provide your final answer in the format: Country, City, Latitude, Longitude. "
    "e.g. Italy, Golfo Arnaci, 40.9606, 9.5873\n"
    "Wrap your final answer in <answer> tags: <answer>Country, City, Latitude, Longitude</answer>. "
    "e.g. <answer>Italy, Golfo Arnaci, 40.9606, 9.5873</answer>"
)

# Variant for LLMClient (235B / Qwen3.5): no native tool schema injection,
# so we provide explicit <tool_call> format instructions in the system prompt.
SYSTEM_TOOL_LLMCLIENT = (
    "You are a geolocation expert. You are given an image and you need to identify its location. "
    "You have access to the following tools:\n"
    "1. `image_zoom_in_tool`: Zoom in on a specific region of the image to examine details such as signs, "
    "license plates, storefronts, or architectural features more closely.\n"
    "2. `image_search_tool`: Search the internet using a cropped region of the image as a query. "
    "Use this after zooming in to find matching web pages that may reveal the location. "
    "Optionally provide a `goal` field to describe what you are trying to identify.\n"
    "\nWorkflow: First analyze the image and reason about what regions contain location clues. "
    "Then call image_zoom_in_tool to zoom in on ONE specific region. After seeing the zoomed result, "
    "call image_search_tool with the same bbox to search the internet. "
    "Continue reasoning and repeat if needed. Finally provide your answer.\n\n"
    "For EVERY response, first enclose your full reasoning in <think> </think> tags, then on its own line output EXACTLY:\n"
    "<tool_call>{\"name\": \"image_zoom_in_tool\", \"arguments\": {\"bbox_2d\": [x1, y1, x2, y2]}}</tool_call>\n"
    "or:\n"
    "<tool_call>{\"name\": \"image_search_tool\", \"arguments\": {\"bbox_2d\": [x1, y1, x2, y2], \"goal\": \"identify the country and city\"}}</tool_call>\n"
    "where bbox_2d is [x1, y1, x2, y2] in [0, 1000] pixel coordinates of the region.\n"
    "Output ONLY ONE tool call per response. Wait for the result before calling again.\n\n"
    "Provide your final answer in the format: Country, City, Latitude, Longitude. "
    "e.g. Italy, Golfo Arnaci, 40.9606, 9.5873\n"
    "Wrap your final answer in <answer> tags: <answer>Country, City, Latitude, Longitude</answer>. "
    "e.g. <answer>Italy, Golfo Arnaci, 40.9606, 9.5873</answer>"
)

# V2: decoupled zoom + search — each tool independently selects the best region.
# Removes the "search same bbox as zoom" constraint from the original SYSTEM_TOOL_LLMCLIENT.
# Motivation: experiment B showed avg_search=0.64 vs D (imgsearch-only) avg_search=1.40 @25km=0.540,
# suggesting that forcing search to follow zoom bbox severely limits search quality.
SYSTEM_TOOL_LLMCLIENT_ZOOM_SEARCH_V2 = (
    "You are a geolocation expert. You are given an image and you need to identify its location.\n"
    "You have access to the following tools:\n"
    "1. `image_zoom_in_tool`: Zoom in on a specific region to examine fine details more closely "
    "(signs, license plates, inscriptions, logos, architectural ornaments).\n"
    "2. `image_search_tool`: Search the internet using a cropped region of the image as a visual query. "
    "Use this on regions containing visually distinctive landmarks, storefronts, or monuments to find "
    "matching web pages. Optionally provide a `goal` field to describe what you are searching for.\n\n"
    "Workflow:\n"
    "Step 1 — Analyze the image: identify all regions with location clues, prioritized as:\n"
    "  • Text/signage (highest priority): street signs, shop names, billboards, license plates\n"
    "  • Distinctive architecture: churches, temples, hotels, commercial buildings with recognizable style\n"
    "  • Sculptures/monuments: statues, memorials, fountains, art installations\n"
    "  • Natural landmarks: unique mountain peaks, rock formations, coastline shapes\n"
    "  • Infrastructure: distinctive bridges, lighthouses, docks, transit facilities\n"
    "  • Other geographically distinctive visual elements\n\n"
    "Step 2 — Use tools strategically and independently:\n"
    "  • Use `image_zoom_in_tool` when close-up detail adds value: to read text, identify inscriptions, "
    "logos, license plates, or fine architectural features. "
    "Do NOT zoom on wide-scene elements better assessed at full scale (skylines, landscapes, vegetation).\n"
    "  • Use `image_search_tool` on any visually distinctive region where image matching may identify "
    "a landmark or location. Choose the BEST region for image search independently — it does NOT need "
    "to be the same region you zoomed into. A wide landmark crop often works better for image search.\n\n"
    "Step 3 — Provide your answer once you have gathered enough evidence. "
    "You do not need to use all available turns — if the location is already clear, answer directly.\n\n"
    "For EVERY tool call response, first enclose your full reasoning in <think> </think> tags, "
    "then on its own line output EXACTLY ONE of:\n"
    "<tool_call>{\"name\": \"image_zoom_in_tool\", \"arguments\": {\"bbox_2d\": [x1, y1, x2, y2]}}</tool_call>\n"
    "or:\n"
    "<tool_call>{\"name\": \"image_search_tool\", \"arguments\": {\"bbox_2d\": [x1, y1, x2, y2], \"goal\": \"identify the landmark\"}}</tool_call>\n"
    "where bbox_2d is [x1, y1, x2, y2] in [0, 1000] pixel coordinates of the region.\n"
    "Output ONLY ONE tool call per response. Wait for the result before calling again.\n\n"
    "After receiving results from image_search_tool, inside your <think> block identify which results match this specific image "
    "(compare result titles/snippets against the original image — not just against the query). "
    "Then immediately after </think> output on its own line: "
    "<useful>[i, j, ...]</useful> listing the 1-based indices of results that match "
    "(i.e. the result explicitly mentions the actual location, landmark, or geographic region shown in the image). "
    "Results about a different place are NOT useful even if they contain geographic information. "
    "Output <useful>[]</useful> if none match. "
    "Then on the next line output another <tool_call> or your final <answer>.\n\n"
    "Final answer format: <answer>Country, City, Latitude, Longitude</answer>. "
    "e.g. <answer>Italy, Golfo Arnaci, 40.9606, 9.5873</answer>"
)

SYSTEM_TOOL_LLMCLIENT_ZOOM_ONLY = (
    "You are a geolocation expert. You are given an image and you need to identify its location.\n"
    "You have access to the following tool:\n"
    "1. `image_zoom_in_tool`: Zoom in on a specific region of the image to examine details more closely.\n\n"
    "Workflow:\n"
    "Step 1 — Analyze the image: identify all key visual regions that could help determine the shooting "
    "location, prioritized as follows:\n"
    "  • Text/signage (highest priority): street signs, shop names, billboards, license plates\n"
    "  • Distinctive architecture: churches, temples, hotels, commercial buildings with recognizable style\n"
    "  • Sculptures/monuments: statues, memorials, fountains, art installations\n"
    "  • Natural landmarks: unique mountain peaks, rock formations, coastline shapes\n"
    "  • Infrastructure: distinctive bridges, lighthouses, docks, transit facilities\n"
    "  • Other geographically distinctive visual elements\n\n"
    "Step 2 — Decide what to zoom: use image_zoom_in_tool ONLY on regions where close-up detail adds "
    "value (text, architectural ornaments, monument inscriptions, distinctive logos, etc.). "
    "Do NOT zoom on wide-scene elements better assessed at full scale "
    "(overall skyline, landscape shapes, vegetation, general urban layout).\n\n"
    "Step 3 — Provide your answer once you have gathered enough evidence. "
    "You do not need to use all available turns — if the image is already clear enough, answer directly.\n\n"
    "For EVERY tool call response, first enclose your full reasoning in <think> </think> tags, "
    "then on its own line output EXACTLY:\n"
    "<tool_call>{\"name\": \"image_zoom_in_tool\", \"arguments\": {\"bbox_2d\": [x1, y1, x2, y2]}}</tool_call>\n"
    "where bbox_2d is [x1, y1, x2, y2] in [0, 1000] normalized coordinates of the region.\n"
    "Output ONLY ONE tool call per response. Wait for the result before calling again.\n\n"
    "Provide your final answer in the format: Country, City, Latitude, Longitude. "
    "e.g. Italy, Golfo Arnaci, 40.9606, 9.5873\n"
    "Wrap your final answer in <answer> tags: <answer>Country, City, Latitude, Longitude</answer>. "
    "e.g. <answer>Italy, Golfo Arnaci, 40.9606, 9.5873</answer>"
)

SYSTEM_TOOL_LLMCLIENT_SEARCH_ONLY = (
    "You are a geolocation expert. You are given an image and you need to identify its location. "
    "You have access to the following tool:\n"
    "1. `image_search_tool`: Search the internet using a cropped region of the image as a query. "
    "Use this on regions containing distinctive landmarks, signs, or text to find matching web pages. "
    "Optionally provide a `goal` field to describe what you are trying to identify.\n"
    "\nWorkflow: First analyze the image and identify regions with strong location clues. "
    "Call image_search_tool on those regions to find matching web pages. "
    "Continue reasoning and repeat if needed. Finally provide your answer.\n\n"
    "For EVERY response, first enclose your full reasoning in <think> </think> tags, then on its own line output EXACTLY:\n"
    "<tool_call>{\"name\": \"image_search_tool\", \"arguments\": {\"bbox_2d\": [x1, y1, x2, y2], \"goal\": \"identify the landmark\"}}</tool_call>\n"
    "where bbox_2d is [x1, y1, x2, y2] in [0, 1000] pixel coordinates of the region.\n"
    "Output ONLY ONE tool call per response. Wait for the result before calling again.\n\n"
    "After receiving results from image_search_tool, inside your <think> block identify which results match this specific image "
    "(compare result titles/snippets against the original image — not just against the query). "
    "Then immediately after </think> output on its own line: "
    "<useful>[i, j, ...]</useful> listing the 1-based indices of results that match "
    "(i.e. the result explicitly mentions the actual location, landmark, or geographic region shown in the image). "
    "Results about a different place are NOT useful even if they contain geographic information. "
    "Output <useful>[]</useful> if none match. "
    "Then on the next line output another <tool_call> or your final <answer>.\n\n"
    "Final answer format: <answer>Country, City, Latitude, Longitude</answer>. "
    "e.g. <answer>Italy, Golfo Arnaci, 40.9606, 9.5873</answer>"
)

# ── v2 prompts (phase5_v2/v3) — combo labels: A=notool, B=zoom, C=imgsearch, D=textsearch, E=zoom+text, F=imgsearch+text, G=all3 ──

SYSTEM_TOOL_LLMCLIENT_TAVILY_ONLY_V2 = (
    "You are a geolocation expert. Given an image, identify its location.\n"
    "You have one tool: `text_search_tool` — search the web with natural language queries.\n\n"
    "Workflow:\n"
    "1. Carefully analyze the image. Look for: text/signs/billboards, landmark names, architectural style, "
    "vegetation, vehicles, language, flags, infrastructure — anything geographically distinctive.\n"
    "2. Call `text_search_tool` with a specific query. Prefer combining multiple clues "
    "(e.g. 'Gothic cathedral red brick tower Germany', 'street sign Cyrillic mountains Serbia'). "
    "You can pass a list of queries for parallel search when you have multiple independent clues.\n"
    "3. If results are unclear or too generic, reformulate with a different angle and search again.\n"
    "4. Once you have enough evidence, provide your final answer directly — do not search unnecessarily.\n\n"
    "For EVERY response, first enclose your reasoning in <think> </think> tags, then output EXACTLY ONE of:\n"
    "<tool_call>{\"name\": \"text_search_tool\", \"arguments\": {\"query\": \"your query\"}}</tool_call>\n"
    "or (parallel):\n"
    "<tool_call>{\"name\": \"text_search_tool\", \"arguments\": {\"query\": [\"query one\", \"query two\"]}}</tool_call>\n"
    "or your final answer in <answer> tags.\n\n"
    "After receiving results from text_search_tool, inside your <think> block identify which results match this specific image "
    "(compare result titles/snippets against the original image — not just against the query). "
    "Then immediately after </think> output on its own line: "
    "<useful>[i, j, ...]</useful> listing the 1-based indices of results that match "
    "(i.e. the result explicitly mentions the actual location, landmark, or geographic region shown in the image). "
    "Results about a different place are NOT useful even if they contain geographic information. "
    "Output <useful>[]</useful> if none match. "
    "Then on the next line output another <tool_call> or your final <answer>.\n\n"
    "Final answer format: <answer>Country, City, Latitude, Longitude</answer>. "
    "e.g. <answer>Italy, Golfo Arnaci, 40.9606, 9.5873</answer>"
)

SYSTEM_TOOL_LLMCLIENT_ZOOM_TAVILY_V2 = (
    "You are a geolocation expert. Given an image, identify its location.\n"
    "You have two tools:\n"
    "1. `image_zoom_in_tool`: Zoom into a region to read fine details (text, signs, inscriptions, logos, "
    "license plates). Use ONLY when the detail is too small to read at full scale.\n"
    "2. `text_search_tool`: Search the web with natural language queries. Use to look up any visible or "
    "zoomed-in text, landmark names, architectural styles, or geographic clues.\n\n"
    "Workflow:\n"
    "1. Analyze the image. Identify location clues — text, landmarks, architecture, vegetation, etc.\n"
    "2. If useful text/signs are already legible, go directly to `text_search_tool`.\n"
    "   If text exists but is too small to read, use `image_zoom_in_tool` first, then `text_search_tool`.\n"
    "3. After seeing search results, if uncertain, you may zoom another region or search with a refined query.\n"
    "4. Once confident, provide your final answer.\n\n"
    "For EVERY response, first enclose your reasoning in <think> </think> tags, then output EXACTLY ONE of:\n"
    "<tool_call>{\"name\": \"image_zoom_in_tool\", \"arguments\": {\"bbox_2d\": [x1, y1, x2, y2]}}</tool_call>\n"
    "or:\n"
    "<tool_call>{\"name\": \"text_search_tool\", \"arguments\": {\"query\": \"your query\"}}</tool_call>\n"
    "or (parallel):\n"
    "<tool_call>{\"name\": \"text_search_tool\", \"arguments\": {\"query\": [\"query one\", \"query two\"]}}</tool_call>\n"
    "where bbox_2d is [x1, y1, x2, y2] in [0, 1000] normalized coordinates.\n"
    "Output ONLY ONE tool call per response. Wait for the result before calling again.\n\n"
    "After receiving results from text_search_tool, inside your <think> block identify which results match this specific image "
    "(compare result titles/snippets against the original image — not just against the query). "
    "Then immediately after </think> output on its own line: "
    "<useful>[i, j, ...]</useful> listing the 1-based indices of results that match "
    "(i.e. the result explicitly mentions the actual location, landmark, or geographic region shown in the image). "
    "Results about a different place are NOT useful even if they contain geographic information. "
    "Output <useful>[]</useful> if none match. "
    "Then on the next line output another <tool_call> or your final <answer>.\n\n"
    "Final answer format: <answer>Country, City, Latitude, Longitude</answer>. "
    "e.g. <answer>Italy, Golfo Arnaci, 40.9606, 9.5873</answer>"
)

SYSTEM_TOOL_LLMCLIENT_SEARCH_TAVILY_V2 = (
    "You are a geolocation expert. Given an image, identify its location.\n"
    "You have two tools:\n"
    "1. `image_search_tool`: Reverse image search using a cropped region of the image. "
    "Crop the most distinctive region (landmark, building, sign). Returns matching web pages with titles and sources.\n"
    "2. `text_search_tool`: Search the web with natural language queries. Use to look up landmark names, "
    "location names, or any text clues — including names or places found in image_search results.\n\n"
    "Workflow:\n"
    "1. **First, analyze the image and assess your own confidence.**\n"
    "   - If you are HIGHLY CONFIDENT of the exact location (a world-famous landmark like the Eiffel Tower, "
    "Colosseum, Taj Mahal; or clearly visible text/signs that name the place; or a scene you already know "
    "the coordinates for), provide your final `<answer>` DIRECTLY without any tool call. "
    "Your `<think>` should explicitly state why no tool is needed "
    "(e.g. 'This is clearly the Eiffel Tower in Paris, France — no search needed.').\n"
    "   - Only call a tool when you are UNCERTAIN of the exact city/country/coordinates. When in doubt, use a tool.\n"
    "2. If you need a tool: identify the most geographically distinctive region (a specific landmark, sign, building facade, or unique scene element), then call `image_search_tool` on that region. "
    "Crop to a specific subregion when possible; use the full image only if the entire scene is the landmark. "
    "Optionally set `goal` to describe what you're searching for.\n"
    "3. Read the results. If you find a landmark name, location, or any useful text, call `text_search_tool` "
    "to gather more details (description, city, country, coordinates).\n"
    "4. If visible text or signs are already identifiable, you may call `text_search_tool` directly.\n"
    "5. Once you have enough evidence, provide your final answer.\n\n"
    "For EVERY response, first enclose your reasoning in <think> </think> tags, then output EXACTLY ONE of:\n"
    "<tool_call>{\"name\": \"image_search_tool\", \"arguments\": {\"bbox_2d\": [x1, y1, x2, y2], \"goal\": \"...\"}}</tool_call>\n"
    "or:\n"
    "<tool_call>{\"name\": \"text_search_tool\", \"arguments\": {\"query\": \"your query\"}}</tool_call>\n"
    "or (parallel):\n"
    "<tool_call>{\"name\": \"text_search_tool\", \"arguments\": {\"query\": [\"query one\", \"query two\"]}}</tool_call>\n"
    "or your final answer in <answer> tags (when confident without needing a tool).\n"
    "where bbox_2d is [x1, y1, x2, y2] in [0, 1000] normalized coordinates.\n"
    "Output ONLY ONE tool call per response. Wait for the result before calling again.\n\n"
    "After receiving results from image_search_tool or text_search_tool, inside your <think> block identify which results match this specific image "
    "(compare result titles/snippets against the original image — not just against the query). "
    "Then immediately after </think> output on its own line: "
    "<useful>[i, j, ...]</useful> listing the 1-based indices of results that match "
    "(i.e. the result explicitly mentions the actual location, landmark, or geographic region shown). "
    "Results about a different place are NOT useful even if they contain geographic information. "
    "Output <useful>[]</useful> if none match. "
    "Then on the next line output another <tool_call> or your final <answer>.\n\n"
    "Final answer format: <answer>Country, City, Latitude, Longitude</answer>. "
    "e.g. <answer>Italy, Golfo Arnaci, 40.9606, 9.5873</answer>"
)

SYSTEM_TOOL_LLMCLIENT_ALL3_V2 = (
    "You are a geolocation expert. Given an image, identify its location.\n"
    "You have three tools:\n"
    "1. `image_search_tool`: Reverse image search using a cropped region. Best for distinctive landmarks, "
    "buildings, or scenes. Crop to the most informative subregion when possible; use the full image only if the entire scene is the landmark. "
    "Returns matching web pages.\n"
    "2. `text_search_tool`: Search the web with natural language queries. Use for visible text/signs, "
    "landmark names, or any clues found from image search results.\n"
    "3. `image_zoom_in_tool`: Zoom into a region to read text/inscriptions that are too small at full scale.\n\n"
    "Decision rules:\n"
    "  • If HIGHLY CONFIDENT of the exact location (world-famous landmark, clearly legible place-name) → provide <answer> directly.\n"
    "  • Distinctive landmark or scene visible → use `image_search_tool`\n"
    "  • Text/signs already legible → use `text_search_tool` directly\n"
    "  • Text/signs too small to read → use `image_zoom_in_tool` first, then `text_search_tool`\n"
    "  • image_search returns a landmark/location name → follow up with `text_search_tool`\n"
    "  • Do NOT use `image_zoom_in_tool` before `image_search_tool` — zoom does not improve image search\n"
    "  • Fallback: if tools have failed or returned empty results, provide your best <answer> from visual priors. "
    "Never end a response without a <tool_call> or <answer>.\n\n"
    "For EVERY response, first enclose your reasoning in <think> </think> tags, then output EXACTLY ONE of:\n"
    "<tool_call>{\"name\": \"image_search_tool\", \"arguments\": {\"bbox_2d\": [x1, y1, x2, y2], \"goal\": \"...\"}}</tool_call>\n"
    "or:\n"
    "<tool_call>{\"name\": \"text_search_tool\", \"arguments\": {\"query\": \"your query\"}}</tool_call>\n"
    "or (parallel):\n"
    "<tool_call>{\"name\": \"text_search_tool\", \"arguments\": {\"query\": [\"query one\", \"query two\"]}}</tool_call>\n"
    "or:\n"
    "<tool_call>{\"name\": \"image_zoom_in_tool\", \"arguments\": {\"bbox_2d\": [x1, y1, x2, y2]}}</tool_call>\n"
    "where bbox_2d is [x1, y1, x2, y2] in [0, 1000] normalized coordinates.\n"
    "Output ONLY ONE tool call per response. Wait for the result before calling again.\n\n"
    "After receiving results from image_search_tool or text_search_tool, inside your <think> block identify which results match this specific image "
    "(compare result titles/snippets against the original image — not just against the query). "
    "Then immediately after </think> output on its own line: "
    "<useful>[i, j, ...]</useful> listing the 1-based indices of results that match "
    "(i.e. the result explicitly mentions the actual location, landmark, or geographic region shown in the image). "
    "Results about a different place are NOT useful even if they contain geographic information. "
    "Output <useful>[]</useful> if none match. "
    "Then on the next line output another <tool_call> or your final <answer>.\n\n"
    "Final answer format: <answer>Country, City, Latitude, Longitude</answer>. "
    "e.g. <answer>Italy, Golfo Arnaci, 40.9606, 9.5873</answer>"
)

SYSTEM_NOTOOL = (
    "You are a geolocation expert. You are given an image and you need to identify its location. "
    "Reason step by step before making your prediction.\n\n"
    "Provide your final answer in the format: Country, City, Latitude, Longitude. "
    "e.g. Italy, Golfo Arnaci, 40.9606, 9.5873\n"
    "Wrap your final answer in <answer> tags: <answer>Country, City, Latitude, Longitude</answer>. "
    "e.g. <answer>Italy, Golfo Arnaci, 40.9606, 9.5873</answer>"
)

# user_q verbatim from SpotSFT dataset (confirmed in validation_output/0.jsonl)
_user_q = (
    "Analyze the architectural styles, vegetation, street infrastructure, and cultural markers in this image. "
    "Based on these visual cues, determine the location."
)
# instruction_following: <answer> format (SFT-compatible)
_instruction_following = (
    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    "The reasoning process MUST BE enclosed within <think> </think> tags. "
    "Wrap your final answer in <answer> tags in the format: <answer>Country, City, Latitude, Longitude</answer>. "
    "e.g. <answer>Italy, Golfo Arnaci, 40.9606, 9.5873</answer>"
)
USER_PROMPT = _user_q + "\n\nAnswer strictly in the following format:\nCountry, City, Latitude, Longitude. " + _instruction_following


# ── Image helpers ──────────────────────────────────────────────────────────────
def encode_image(img: Image.Image, max_pixels: int = 2048 * 1024) -> str:
    w, h = img.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()

def load_encode(img_path: str) -> str:
    return encode_image(Image.open(img_path).convert("RGB"))

# ── Geo helpers ────────────────────────────────────────────────────────────────
_B64_DATA_RE = re.compile(r'^data:image/([^;]+);base64,([A-Za-z0-9+/=]+)$')

def _b64_url_to_bytes(b64_url: str):
    """Convert a data:image/...;base64,... URL to raw bytes. Returns None on failure."""
    m = _B64_DATA_RE.match(b64_url)
    if not m:
        return None
    try:
        return base64.b64decode(m.group(2))
    except Exception:
        return None


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(min(a, 1.0)))

ANSWER_RE     = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
USEFUL_RE     = re.compile(r'<useful>\s*(\[.*?\])\s*</useful>', re.DOTALL)
BOXED_RE = re.compile(r'\\boxed\{([^}]+)\}')
# Fallback: \boxed{ with no closing brace (response truncated at max_tokens)
BOXED_TRUNC_RE = re.compile(r'\\boxed\{([^}]{3,}?)$')
# Match floats with optional degree symbol and N/S/E/W suffix (e.g. 39.9448° N, 32.8545° E)
DEGREE_RE = re.compile(
    r'([-+]?\d{1,3}(?:\.\d+)?)\s*°?\s*([NS])[\s,;/]+?([-+]?\d{1,3}(?:\.\d+)?)\s*°?\s*([EW])',
    re.IGNORECASE,
)
# Match any pair of floats separated by comma (for plain-text Country, City, lat, lon format)
FLOAT_PAIR_RE = re.compile(r'([-+]?\d{1,3}(?:\.\d+)?)\s*,\s*([-+]?\d{1,3}(?:\.\d+)?)')

def parse_pred(text: str):
    answer_matches = ANSWER_RE.findall(text)
    if answer_matches:
        parts = answer_matches[-1].split(",")
        try:
            lat = float(parts[-2].strip())
            lon = float(parts[-1].strip())
            if lat == 0.0 and lon == 0.0:
                return None  # filter Gulf of Guinea default (model uncertainty fallback)
            return lat, lon
        except Exception:
            pass
    return None

# ── LLMClient helpers (235B / Qwen3.5) ────────────────────────────────────────

def _linearize_messages_for_235b(messages: list):
    """
    Flatten a multi-turn messages list into a single content list for 235B.
    235B only accepts one content list; we prefix role labels so the model
    understands turn boundaries. Returns (content_list, system_str_or_None).

    Special handling: the last assistant message's <tool_call> blocks are
    stripped to avoid 235B mimicking the tool-call format in its final reply.
    """
    system = None
    content = []
    for msg in messages:
        role = msg["role"]
        body = msg["content"]
        if role == "system":
            system = body if isinstance(body, str) else None
            continue
        role_tag = "Assistant: " if role == "assistant" else "User: "

        # Strip <tool_call>...</tool_call> from ALL assistant messages so 235B
        # doesn't mimic the format and keep outputting tool calls in final reply.
        if role == "assistant":
            if isinstance(body, str):
                body = re.sub(r'<tool_call>[\s\S]*?</tool_call>', '', body).strip()

        if isinstance(body, str):
            # Always emit the role tag even if body is empty (e.g. pure tool_call stripped),
            # so 235B sees the correct User/Assistant turn boundaries.
            content.append({"type": "text", "text": role_tag + body})
        elif isinstance(body, list):
            first = True
            for part in body:
                if part["type"] == "image_url":
                    if first:
                        content.append({"type": "text", "text": role_tag})
                        first = False
                    content.append(part)
                elif part["type"] == "text":
                    prefix = role_tag if first else ""
                    first = False
                    content.append({"type": "text", "text": prefix + part["text"]})
    return content, system


async def _chat_llmclient(llm_client, messages: list, max_tokens: int,
                          tools: list = None):
    """
    Async wrapper around the synchronous LLMClient.
    Runs in a thread executor so the event loop stays unblocked.
    Returns (content_str, finish_reason) — same shape as chat().

    Both 235B and qwen35vl go through chat_messages() which supports
    native multi-turn messages lists.
    For 235B, tools are passed via OpenAI tool_calls API (native, no prompt hack).
    """
    loop = asyncio.get_event_loop()
    def _call():
        return llm_client.chat_messages(messages, max_tokens=max_tokens, tools=tools)
    # Extra outer retry: llm_client already retries 3×, but single-machine models
    # (e.g. kimi_k2d6) can still return None under high concurrency. Retry up to 5
    # more times with exponential back-off before giving up.
    for outer in range(5):
        result = await loop.run_in_executor(None, _call)
        if result:
            return (result, "")
        await asyncio.sleep(2 ** outer)
    return ("", "")


# ── SGLang API ─────────────────────────────────────────────────────────────────
async def chat(session, url, messages, tools=None, max_tokens=None,
               temperature=0.0, llm_client=None, no_thinking=False):
    if llm_client is not None:
        return await _chat_llmclient(llm_client, messages, max_tokens, tools=tools)
    payload = {
        "model":       "default",
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools
        payload["stop"] = ["</tool_call>"]
    if no_thinking:
        payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    for attempt in range(3):
        try:
            async with session.post(
                f"{url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                data          = await resp.json()
                msg           = data["choices"][0]["message"]
                content       = msg.get("content", "") or ""
                finish_reason = data["choices"][0].get("finish_reason", "")
                return content, finish_reason
        except Exception:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
    return "", ""

# ── Single-turn eval ──────────────────────────────────────────────────────────
async def eval_single_turn(row, img_path, session, url, sem, args, llm_client=None):
    """Single-turn inference — no tool use."""
    async with sem:
        orig_b64 = load_encode(img_path)
        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{orig_b64}"}},
                {"type": "text",      "text": args.user_prompt},
            ]},
        ]
        response, _ = await chat(session, url, messages,
                                 max_tokens=args.max_tokens,
                                 temperature=getattr(args, 'temperature', 0.0),
                                 llm_client=llm_client,
                                 no_thinking=getattr(args, 'no_thinking', False))
        pred = parse_pred(response)
        km   = haversine(pred[0], pred[1], row.latitude, row.longitude) if pred else None
        return {
            "id":           row.id,
            "gt_lat":       row.latitude, "gt_lon":   row.longitude,
            "gt_country":   getattr(row, "country", None),
            "gt_city":      getattr(row, "city", None),
            "pred_lat":     pred[0] if pred else None,
            "pred_lon":     pred[1] if pred else None,
            "km":           km,
            "response":     response,
            "n_tool_calls": 0,
        }

# ── Multi-turn agent loop ──────────────────────────────────────────────────────
async def eval_agent_loop(row, img_path, session, url, sem, args, llm_client=None,
                          search_sem=None, tavily_sem=None):
    """
    Multi-turn agent loop producing DeepResearch-compatible output.

    Message format:
      system  → plain string
      user    → [{"type":"text","text":"<image>\\nimage_id: 0"}, {"type":"text","text":question}]
      assistant → reasoning + <tool_call>...</tool_call>  (or <answer>lat,lon</answer> for final)
      user    → "<tool_response>\\n...result...\\n</tool_response>"
                (+ "<image>\\nimage_id: N" prepended when a crop image exists)
      ...
      assistant → <answer>lat, lon</answer>

    Images are tracked in images_list (list of bytes).  Messages reference them via
    "<image>\\nimage_id: N" placeholder (index into images_list).

    Returns: eval metrics + messages (clean, no inline base64) + images (bytes or file paths).
    """
    # Determine active tool set (from --tools flag or legacy flags)
    _tools_set = set()
    if hasattr(args, 'tools') and args.tools:
        for _t in args.tools.split(","):
            _t = _t.strip().lower()
            if _t in ("zoom", "image_zoom_in_tool"):
                _tools_set.add("zoom")
            elif _t in ("image_search", "image_search_tool"):
                _tools_set.add("image_search")
            elif _t in ("text_search", "text_search_tool", "tavily"):
                _tools_set.add("text_search")
    elif getattr(args, 'use_tavily', False):
        _tools_set = {"zoom", "image_search", "text_search"}
    elif getattr(args, 'zoom_only', False):
        _tools_set = {"zoom"}
    elif getattr(args, 'use_tools', False):
        _tools_set = {"zoom", "image_search"}

    api_tools = None
    if llm_client is None and not getattr(args, 'no_api_tools', False):
        schema_list = []
        if "zoom"         in _tools_set: schema_list.append(TOOL_SCHEMA)
        if "image_search" in _tools_set: schema_list.append(TOOL_SCHEMA_SEARCH)
        if "text_search"  in _tools_set: schema_list.append(TOOL_SCHEMA_TAVILY)
        api_tools = schema_list or None
    elif args.mode == "agent" and _tools_set:
        # kimi / 235b / qwen35vl: pass native tool schemas so model uses tool_calls API
        # (avoids relying on text-format <tool_call> parsing with no stop token)
        schema_list = []
        if "zoom"         in _tools_set: schema_list.append(TOOL_SCHEMA)
        if "image_search" in _tools_set: schema_list.append(TOOL_SCHEMA_SEARCH)
        if "text_search"  in _tools_set: schema_list.append(TOOL_SCHEMA_TAVILY)
        api_tools = schema_list or None

    async with sem:
        # ── images_list: all images as bytes, messages reference by index ────────
        images_list = []  # list of raw bytes

        def _register_image(b64_url: str) -> int:
            """Decode b64_url, append bytes to images_list, return index."""
            raw = _b64_url_to_bytes(b64_url)
            if raw is None:
                raw = b""
            images_list.append(raw)
            return len(images_list) - 1

        orig_b64 = load_encode(img_path)
        img_idx  = _register_image(f"data:image/jpeg;base64,{orig_b64}")

        # Build initial messages in DeepResearch format
        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": f"<image>\nimage_id: {img_idx}"},
                {"type": "text", "text": args.user_prompt},
            ]},
        ]

        # For the actual API calls we pass the real b64 image inline; messages above are the
        # "clean" record. We keep a separate api_messages list that mirrors messages but with
        # real base64 content for the API.
        api_messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{orig_b64}"}},
                {"type": "text",      "text": args.user_prompt},
            ]},
        ]

        manager = ToolCallManager(img_path, tavily_api_key=os.environ.get("TAVILY_API_KEY", ""),
                                  exclude_domains=getattr(args, 'exclude_domains', None) or [])

        _tool_name_map = {
            "image_zoom_in_tool": "zoom",
            "image_search_tool":  "image_search",
            "text_search_tool":   "text_search",
        }

        last_assistant_content = ""
        turn = 0
        n_nudges = 0
        max_turns_exceeded = False
        useful_results = []  # [{turn, tool, indices}] — parsed <useful> annotations

        _prev_search_tools = []  # tool names that ran in the previous turn (for <useful> annotation)

        while turn <= args.max_turns:
            response, _ = await chat(session, url, api_messages,
                                     tools=api_tools, max_tokens=args.max_tokens,
                                     temperature=getattr(args, 'temperature', 0.0),
                                     llm_client=llm_client,
                                     no_thinking=getattr(args, 'no_thinking', False))
            last_assistant_content = response

            # ── Parse <useful> annotation from this response ──────────────────────
            # Model outputs <useful>[i, j, ...]</useful> after receiving search results.
            # Associate with the tools that ran in the previous turn.
            if _prev_search_tools:
                useful_m = USEFUL_RE.search(response)
                if useful_m:
                    try:
                        indices = json.loads(useful_m.group(1))
                    except Exception:
                        indices = []
                    for _tool_name in _prev_search_tools:
                        useful_results.append({
                            "turn": turn,
                            "tool": _tool_name,
                            "indices": indices,
                        })
            _prev_search_tools = []  # reset for this turn

            tool_calls = parse_tool_calls_flexible(response)

            if not tool_calls:
                # Check if model produced a complete think block but forgot to output
                # a tool_call or <answer> — "think-truncation" pattern.
                has_answer = bool(ANSWER_RE.search(response))
                think_truncated = (
                    not has_answer
                    and response.strip()  # non-empty response
                    and re.search(r'</think>', response)  # has a complete think block
                    and turn < args.max_turns  # still have turns left
                )
                if think_truncated:
                    # Append the incomplete assistant turn, then inject a user nudge.
                    messages.append({"role": "assistant", "content": response})
                    api_messages.append({"role": "assistant", "content": response})
                    nudge = ("Your previous response ended without a tool call or final answer. "
                             "Please either call a tool (e.g. image_zoom_in_tool) or provide your "
                             "final answer using <answer>Country, City, Latitude, Longitude</answer>. "
                             "If the location truly cannot be determined, output <answer>None, None</answer>.")
                    messages.append({"role": "user", "content": nudge})
                    api_messages.append({"role": "user", "content": nudge})
                    n_nudges += 1
                    turn += 1  # nudge DOES count as a turn (bounded by max_turns)
                    continue
                messages.append({"role": "assistant", "content": response})
                api_messages.append({"role": "assistant", "content": response})
                break

            if turn == args.max_turns:
                messages.append({"role": "assistant", "content": response})
                api_messages.append({"role": "assistant", "content": response})
                max_turns_exceeded = True
                break

            # ── native tool role path (235b and kimi both use OpenAI tool_calls format) ─
            use_native_tool_role = (llm_client is not None and
                                    getattr(llm_client, "model", None) in ("235b", "kimi_k2d6"))

            # Execute tool calls
            loop = asyncio.get_event_loop()
            tc_results = []  # (name, args_, img_b64_or_None, tool_text, extra_b64_list)
            for tool_call in tool_calls:
                name = tool_call.get("name", "")
                if _tools_set and _tool_name_map.get(name) not in _tools_set:
                    continue
                args_ = tool_call.get("arguments", {})
                if isinstance(args_, str):
                    try:
                        args_ = json.loads(args_)
                    except Exception:
                        args_ = {}
                try:
                    if name == "image_search_tool" and search_sem is not None:
                        async with search_sem:
                            result = await loop.run_in_executor(None, manager.execute, name, args_)
                    elif name == "text_search_tool" and tavily_sem is not None:
                        async with tavily_sem:
                            result = await loop.run_in_executor(None, manager.execute, name, args_)
                    else:
                        result = await loop.run_in_executor(None, manager.execute, name, args_)
                    img_b64     = result.get("crop_b64")
                    search_text = result.get("text")
                    extra_b64_list = []  # extra images to store (not sent to model)
                    if name == "image_zoom_in_tool":
                        bbox  = args_.get("bbox_2d", [])
                        label = args_.get("label", "")
                        if label:
                            tool_text = f"Zoomed in on the image to the region {bbox} with label {label}."
                        else:
                            tool_text = f"Zoomed in on the image to the region {bbox}."
                    elif name == "image_search_tool":
                        tool_text = search_text or "No search results available."
                        img_b64   = None  # crop_b64 not set by image_search_tool_core
                        # collect all returned thumbnail images for saving
                        for thumb in result.get("thumbnails", []):
                            if thumb:
                                extra_b64_list.append(thumb)
                    elif name == "text_search_tool":
                        tool_text = search_text or "No search results available."
                        img_b64   = None
                    else:
                        tool_text = str(result)
                        img_b64   = None
                except Exception as e:
                    img_b64   = None
                    extra_b64_list = []
                    tool_text = f"Tool call error: {e}"
                tc_results.append((name, args_, img_b64, tool_text, extra_b64_list))

            if use_native_tool_role:
                # 235B / kimi: standard OpenAI tool_calls format for api_messages.
                # For clean messages record, still use <tool_response> format.
                _is_kimi = (llm_client is not None and getattr(llm_client, "model", None) == "kimi_k2d6")
                tool_calls_payload = [
                    {
                        "id": f"call_{turn}_{i}",
                        "type": "function",
                        "function": {"name": tc_name, "arguments": json.dumps(tc_args)},
                    }
                    for i, (tc_name, tc_args, _, _, _) in enumerate(tc_results)
                ]
                api_messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tool_calls_payload,
                })
                for i, (tc_name, tc_args, img_b64, tool_text, extra_b64_list) in enumerate(tc_results):
                    if img_b64 and not _is_kimi:
                        # 235B supports list content in role=tool
                        tool_content = [
                            {"type": "text", "text": tool_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                        ]
                        api_messages.append({
                            "role": "tool",
                            "tool_call_id": f"call_{turn}_{i}",
                            "content": tool_content,
                        })
                    elif img_b64 and _is_kimi:
                        # kimi: role=tool must be plain text; send image as follow-up role=user
                        api_messages.append({
                            "role": "tool",
                            "tool_call_id": f"call_{turn}_{i}",
                            "content": tool_text,
                        })
                        api_messages.append({"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                            {"type": "text", "text": "(zoomed image for the tool result above)"},
                        ]})
                    else:
                        api_messages.append({
                            "role": "tool",
                            "tool_call_id": f"call_{turn}_{i}",
                            "content": tool_text,
                        })
                        # --thumbnails: send image_search thumbnail images to model
                        if getattr(args, 'thumbnails', False) and tc_name == "image_search_tool" and extra_b64_list:
                            thumb_content = [{"type": "text", "text": "(thumbnail images for the search results above)"}]
                            for thumb_b64 in extra_b64_list:
                                if thumb_b64:
                                    url = thumb_b64 if thumb_b64.startswith("data:") else f"data:image/jpeg;base64,{thumb_b64}"
                                    thumb_content.append({"type": "image_url", "image_url": {"url": url}})
                            if len(thumb_content) > 1:  # has at least one thumbnail
                                api_messages.append({"role": "user", "content": thumb_content})
                    # save thumbnail images from image_search (not sent to model)
                    # thumbnails are already full data-URLs: "data:image/jpeg;base64,..."
                    for thumb_b64 in extra_b64_list:
                        _register_image(thumb_b64)
                # Clean messages: assistant + user <tool_response>
                messages.append({"role": "assistant", "content": response})
                for tc_name, tc_args, img_b64, tool_text, extra_b64_list in tc_results:
                    if img_b64:
                        crop_idx  = _register_image(f"data:image/jpeg;base64,{img_b64}")
                        resp_content = (
                            f"<image>\nimage_id: {crop_idx}\n"
                            f"<tool_response>\n{tool_text}\n</tool_response>"
                        )
                    else:
                        resp_content = f"<tool_response>\n{tool_text}\n</tool_response>"
                    messages.append({"role": "user", "content": resp_content})
            else:
                # ── Non-235B path: <tool_call> in assistant content ────────────
                # Restore closing tag if stop= stripped it
                assistant_content = response
                if "<tool_call>" in assistant_content and not assistant_content.rstrip().endswith("</tool_call>"):
                    tc_start   = assistant_content.rfind("<tool_call>")
                    json_start = assistant_content.find("{", tc_start)
                    if json_start != -1:
                        depth    = 0
                        json_end = json_start
                        for ci, ch in enumerate(assistant_content[json_start:], json_start):
                            if ch == "{":
                                depth += 1
                            elif ch == "}":
                                depth -= 1
                                if depth == 0:
                                    json_end = ci
                                    break
                        assistant_content = assistant_content[:json_end + 1] + "\n</tool_call>"
                    else:
                        assistant_content = assistant_content.rstrip() + "\n</tool_call>"

                messages.append({"role": "assistant", "content": assistant_content})
                api_messages.append({"role": "assistant", "content": assistant_content})

                # Inject tool results as role=user with <tool_response>
                for tc_name, tc_args, img_b64, tool_text, extra_b64_list in tc_results:
                    if img_b64:
                        crop_idx = _register_image(f"data:image/jpeg;base64,{img_b64}")
                        # Clean message: image placeholder + tool_response text
                        clean_content = (
                            f"<image>\nimage_id: {crop_idx}\n"
                            f"<tool_response>\n{tool_text}\n</tool_response>"
                        )
                        messages.append({"role": "user", "content": clean_content})
                        # API message: real image + text (matches VERL training format: image first)
                        api_messages.append({"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                            {"type": "text",      "text": f"<tool_response>\n{tool_text}\n</tool_response>"},
                        ]})
                    else:
                        resp_text = f"<tool_response>\n{tool_text}\n</tool_response>"
                        messages.append({"role": "user", "content": resp_text})
                        api_messages.append({"role": "user", "content": resp_text})
                    # save thumbnail images from image_search (not sent to model)
                    # thumbnails are already full data-URLs: "data:image/jpeg;base64,..."
                    for thumb_b64 in extra_b64_list:
                        _register_image(thumb_b64)

                # force_search: inject image_search_tool call if zoom was just called
                if (len(tc_results) == 1
                        and tc_results[0][0] == "image_zoom_in_tool"
                        and getattr(args, 'force_search', False)
                        and manager.search_call_count == 0):
                    _fs_bbox = tc_results[0][1].get("bbox_2d", [])
                    try:
                        if search_sem is not None:
                            async with search_sem:
                                _fs_result = await loop.run_in_executor(
                                    None, manager.execute, "image_search_tool",
                                    {"bbox_2d": _fs_bbox})
                        else:
                            _fs_result = await loop.run_in_executor(
                                None, manager.execute, "image_search_tool",
                                {"bbox_2d": _fs_bbox})
                        _fs_text = _fs_result.get("text") or "No search results available."
                    except Exception as _fs_e:
                        _fs_text = f"Image search failed: {_fs_e}"
                    _fs_call_str = (
                        f'<tool_call>\n{{"name": "image_search_tool", "arguments": {{"bbox_2d": {_fs_bbox}}}}}\n</tool_call>'
                    )
                    messages.append({"role": "assistant", "content": _fs_call_str})
                    api_messages.append({"role": "assistant", "content": _fs_call_str})
                    _fs_resp = f"<tool_response>\n{_fs_text}\n</tool_response>"
                    messages.append({"role": "user", "content": _fs_resp})
                    api_messages.append({"role": "user", "content": _fs_resp})

            turn += 1  # only increment after processing real tool calls, not nudges
            # Record which search tools ran this turn so we can correlate <useful> next turn
            _prev_search_tools = [
                tc_name for tc_name, _, _, _, _ in tc_results
                if tc_name in ("image_search_tool", "text_search_tool")
            ]

        pred = parse_pred(last_assistant_content)
        km   = haversine(pred[0], pred[1], row.latitude, row.longitude) if pred else None
        return {
            "id":           row.id,
            "gt_lat":       row.latitude, "gt_lon":   row.longitude,
            "gt_country":   getattr(row, "country", None),
            "gt_city":      getattr(row, "city", None),
            "pred_lat":     pred[0] if pred else None,
            "pred_lon":     pred[1] if pred else None,
            "km":           km,
            "masked":       1.0 if pred is None else 0.0,
            "n_tool_calls": manager.total_tool_calls,
            "n_crop_calls": manager.crop_call_count,
            "n_search_calls": manager.search_call_count,
            "n_tavily_calls": manager.tavily_call_count,
            "n_nudges":     n_nudges,
            "max_turns_exceeded": max_turns_exceeded,
            "useful_results": useful_results,  # [{turn, tool, indices}] from <useful> annotations
            "messages":     messages,
            "images":       images_list,  # list of raw bytes; saved per --image_store in run()
        }

# ── Main ───────────────────────────────────────────────────────────────────────
async def run(args):
    # ── Load benchmark data ──────────────────────────────────────────────────
    if args.benchmark == "yfcc4k":
        csv_file = getattr(args, 'csv_path', None) or YFCC4K_TXT
        df = pd.read_csv(csv_file, sep='\t', header=None)
        # yfcc4k.txt columns: 0=idx, 1=photo_id, ..., 12=longitude, 13=latitude
        df = df.rename(columns={1: 'id', 13: 'latitude', 12: 'longitude'})
        df['id'] = df['id'].astype(str)
        img_dir = YFCC4K_IMG_DIR
    else:
        csv_file = getattr(args, 'csv_path', None) or CSV_PATH
        df = pd.read_csv(csv_file)
        # ── Normalize column names across different CSV schemas ──────────────
        # im2gps3k_CLEAN.csv:  id, latitude, longitude, [country, city]
        # MP16-Pro / coldstart: IMG_ID, LAT, LON, city, country, path
        col_rename = {}
        if "IMG_ID" in df.columns and "id" not in df.columns:
            col_rename["IMG_ID"] = "id"
        if "LAT" in df.columns and "latitude" not in df.columns:
            col_rename["LAT"] = "latitude"
        if "LON" in df.columns and "longitude" not in df.columns:
            col_rename["LON"] = "longitude"
        if col_rename:
            df = df.rename(columns=col_rename)
        img_dir = IMG_DIR

    if args.max_samples > 0:
        subset_seed = getattr(args, 'subset_seed', None)
        if subset_seed is not None:
            df = df.sample(n=min(args.max_samples, len(df)), random_state=subset_seed).reset_index(drop=True)
        else:
            df = df.head(args.max_samples)

    # img_path: prefer inline `path` column; fall back to img_dir lookup
    _has_path_col = "path" in df.columns
    if _has_path_col:
        imgs = {}
    elif args.benchmark == "yfcc4k":
        imgs = {f.replace(".jpg", ""): f for f in os.listdir(img_dir) if f.endswith(".jpg")}
    else:
        imgs = {f.split("_")[0]: f for f in os.listdir(img_dir) if f.endswith(".jpg")}

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if os.path.exists(args.output_jsonl):
        with open(args.output_jsonl) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    # Skip any sample that was already processed (has a response),
                    # regardless of whether a prediction was extracted.
                    if r.get("response") or r.get("messages"):
                        done_ids.add(r["id"])
                except Exception:
                    pass
    print(f"Total: {len(df)}, already done: {len(done_ids)}, to process: {len(df) - len(done_ids)}")

    # ── Init LLMClient if --model is set (skip SGLang) ─────────────────────────
    llm_client = None
    if args.model:
        from llm_client import LLMClient
        llm_client = LLMClient(model=args.model)
        print(f"Using LLMClient(model={args.model!r}), skipping SGLang health check.")
    else:
        print(f"Waiting for SGLang at {args.sglang_url}...")
        async with aiohttp.ClientSession() as s:
            for _ in range(60):
                try:
                    async with s.get(f"{args.sglang_url}/health",
                                     timeout=aiohttp.ClientTimeout(total=5)) as r:
                        if r.status == 200:
                            print("SGLang ready.")
                            break
                except Exception:
                    pass
                await asyncio.sleep(2)

    sem = asyncio.Semaphore(args.concurrency)
    search_sem = asyncio.Semaphore(args.search_concurrency)
    tavily_sem = asyncio.Semaphore(getattr(args, 'tavily_concurrency', 5))

    async with aiohttp.ClientSession() as session:
        tasks = []
        for row in df.itertuples():
            if row.id in done_ids:
                continue
            if _has_path_col:
                img_path = str(getattr(row, "path", ""))
            else:
                img_path = os.path.join(img_dir, imgs.get(str(row.id), ""))
            if not os.path.exists(img_path):
                continue
            if args.mode == "agent":
                tasks.append(eval_agent_loop(row, img_path, session, args.sglang_url, sem, args,
                                             llm_client=llm_client, search_sem=search_sem,
                                             tavily_sem=tavily_sem))
            else:
                tasks.append(eval_single_turn(row, img_path, session, args.sglang_url, sem, args,
                                              llm_client=llm_client))

        with open(args.output_jsonl, "a", buffering=1) as f_out:
            # Images storage: determine mode from --image_store flag
            image_store = getattr(args, 'image_store', 'bytes')
            no_save_images = getattr(args, 'no_save_images', False)
            if image_store == "dir":
                images_dir = args.output_jsonl.replace(".jsonl", "_images")
                os.makedirs(images_dir, exist_ok=True)
            async for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc=args.tag):
                result = await coro
                result = dict(result)
                if "images" in result:
                    images_bytes = result.pop("images")  # list of raw bytes
                    if no_save_images:
                        # Drop images entirely (reduces jsonl size ~100x)
                        pass
                    elif image_store == "dir":
                        # Save each image to {images_dir}/{id}_{idx}.jpg, store path list
                        sample_id  = str(result.get("id", "unk"))
                        saved_paths = []
                        for img_idx, img_raw in enumerate(images_bytes):
                            if img_raw:
                                fpath = os.path.join(images_dir, f"{sample_id}_{img_idx}.jpg")
                                with open(fpath, "wb") as _f:
                                    _f.write(img_raw)
                                saved_paths.append(fpath)
                            else:
                                saved_paths.append(None)
                        result["images"] = saved_paths
                    else:
                        # Store as base64 strings for direct JSON serialization
                        result["images"] = [
                            base64.b64encode(b).decode() if b else None
                            for b in images_bytes
                        ]
                # Skip empty responses (SGLang timeout/error) so they can be retried on re-run
                if not result.get("response") and not result.get("messages"):
                    continue
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                _km      = result.get("km")
                _pred    = f"{result.get('pred_lat','?')},{result.get('pred_lon','?')}"
                _tools   = result.get("n_tool_calls", "?")
                _km_str  = f"{_km:.1f}km" if isinstance(_km, (int, float)) else "N/A"
                print(f"[{result.get('id','?')}] dist={_km_str}  tools={_tools}  pred={_pred}", flush=True)

    all_results = []
    with open(args.output_jsonl) as f:
        for line in f:
            try:
                all_results.append(json.loads(line))
            except Exception:
                pass

    km_list   = [r["km"] for r in all_results if r.get("km") is not None]
    n         = len(km_list)
    n_total   = len(all_results)
    n_no_pred = sum(1 for r in all_results if r.get("km") is None)

    # Standard Im2GPS accuracy: denominator = n_total (unparsed = wrong)
    def acc(thresh):
        return round(sum(1 for k in km_list if k <= thresh) / n_total, 4) if n_total else 0

    summary = {
        "tag":            args.tag,
        "mode":           args.mode,
        "model":          args.sglang_url,
        "n_total":        n_total,
        "n_parsed":       n,
        "n_no_pred":      n_no_pred,
        "parse_rate":     round(n / n_total, 4) if n_total else 0,
        "acc_1km":        acc(1),
        "acc_25km":       acc(25),
        "acc_200km":      acc(200),
        "acc_750km":      acc(750),
        "acc_2500km":     acc(2500),
        "avg_tool_calls": round(
            sum(r.get("n_tool_calls", 0) for r in all_results) / max(n_total, 1), 3
        ),
        "avg_search_calls": round(
            sum(r.get("n_search_calls", 0) for r in all_results) / max(n_total, 1), 3
        ),
        "avg_tavily_calls": round(
            sum(r.get("n_tavily_calls", 0) for r in all_results) / max(n_total, 1), 3
        ),
    }

    summary_path = args.output_jsonl.replace(".jsonl", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 65)
    print(f"  Im2GPS3K Eval — {args.tag}  [mode={args.mode}]")
    print("=" * 65)
    print(f"  Samples:    {n}/{n_total} parsed  ({n_no_pred} no prediction)  parse_rate={summary['parse_rate']:.3f}")
    print(f"  Acc@1km:    {summary['acc_1km']:.3f}  ({int(summary['acc_1km'] * n_total)}/{n_total})")
    print(f"  Acc@25km:   {summary['acc_25km']:.3f}  ({int(summary['acc_25km'] * n_total)}/{n_total})")
    print(f"  Acc@200km:  {summary['acc_200km']:.3f}  ({int(summary['acc_200km'] * n_total)}/{n_total})")
    print(f"  Acc@750km:  {summary['acc_750km']:.3f}  ({int(summary['acc_750km'] * n_total)}/{n_total})")
    print(f"  Acc@2500km: {summary['acc_2500km']:.3f}  ({int(summary['acc_2500km'] * n_total)}/{n_total})")
    if args.mode == "agent":
        print(f"  Avg tool calls:   {summary['avg_tool_calls']:.2f}")
        print(f"  Avg search calls: {summary['avg_search_calls']:.2f}")
        print(f"  Avg tavily calls: {summary['avg_tavily_calls']:.2f}")
    print("=" * 65)
    print(f"  Results → {args.output_jsonl}")
    print(f"  Summary → {summary_path}")
    return summary


def _load_prompt_arg(s: str) -> str:
    """Allow passing prompt as @/path/to/file or inline string."""
    if s and s.startswith("@"):
        return Path(s[1:]).read_text().strip()
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",        choices=["single", "agent"], default="single",
                   help="single=one-shot, agent=multi-turn tool loop")
    p.add_argument("--notool",      action="store_true",
                   help="Use SYSTEM_NOTOOL prompt (default: SYSTEM_TOOL)")
    p.add_argument("--system_prompt", default=None, type=_load_prompt_arg,
                   help="Override system prompt (string or @/path/to/file)")
    p.add_argument("--user_prompt",   default=None, type=_load_prompt_arg,
                   help="Override user prompt (string or @/path/to/file)")
    p.add_argument("--use_tools",   action="store_true",
                   help="Pass TOOL_SCHEMA to SGLang (triggers custom_chat_template tools branch)")
    p.add_argument("--zoom_only",   action="store_true",
                   help="LLMClient agent mode: restrict to zoom tool only (Combo B: zoom-only)")
    p.add_argument("--use_tavily",  action="store_true",
                   help="Enable all 3 tools: zoom + image_search + text_search (Tavily)")
    p.add_argument("--tools",       default=None,
                   help="Comma-separated tool set for LLMClient agent mode. "
                        "Supported names: zoom, image_search, text_search. "
                        "e.g. --tools zoom (B)  --tools image_search (C)  --tools text_search (D)  "
                        "--tools zoom,text_search (E)  --tools image_search,text_search (F)  "
                        "--tools zoom,image_search,text_search (G, all tools). "
                        "Overrides --use_tools/--use_tavily/--zoom_only when model is set.")
    p.add_argument("--max_tokens",  type=int, default=8192)
    p.add_argument("--sglang_url",  default="http://127.0.0.1:30000")
    p.add_argument("--model",       default=None, choices=["235b", "qwen35vl", "kimi_k2d5", "kimi_k2d6"],
                   help="Use LLMClient instead of SGLang: '235b', 'qwen35vl', or 'kimi'")
    p.add_argument("--tag",         default="unnamed")
    p.add_argument("--output_jsonl",default=None)
    p.add_argument("--benchmark",   default="im2gps3k", choices=["im2gps3k", "yfcc4k"],
                   help="Benchmark dataset to evaluate on (default: im2gps3k)")
    p.add_argument("--csv_path",    default=None,
                   help="Override benchmark data path (default: auto from --benchmark)")
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--subset_seed", type=int, default=None,
                   help="Random seed for reproducible subset sampling (uses df.sample instead of df.head)")
    p.add_argument("--max_turns",   type=int, default=3,
                   help="Max agent loop turns (agent mode only)")
    p.add_argument("--concurrency", type=int, default=32,
                   help="Async concurrency for SGLang; for LLMClient use 4-8")
    p.add_argument("--search_concurrency", type=int, default=5,
                   help="Max simultaneous image_search_tool calls (oxylabs rate limit guard)")
    p.add_argument("--tavily_concurrency", type=int, default=5,
                   help="Max simultaneous text_search_tool calls (Tavily rate limit guard)")
    p.add_argument("--no_thinking", action="store_true",
                   help="Disable thinking for reasoning models (Qwen3.5 etc.), "
                        "passes chat_template_kwargs={enable_thinking:false} to SGLang")
    p.add_argument("--no_api_tools", action="store_true",
                   help="Do NOT pass OpenAI tool schemas to SGLang. Use this for SFT models "
                        "trained with text-format <tool_call> system prompts (e.g. coldstart SFT). "
                        "Without this flag, SGLang chat_template injects <tools> XML which conflicts "
                        "with the coldstart training format.")
    p.add_argument("--exclude_domains", nargs="*", default=[],
                   help="Domains to exclude from Tavily text search (e.g. flickr.com). "
                        "Use to prevent ground-truth leakage during benchmark eval.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature for generation (default: 0.0 = greedy)")
    p.add_argument("--image_store",  choices=["bytes", "dir"], default="bytes",
                   help="How to store images in output jsonl. "
                        "'bytes' = base64-encoded inline in 'images' list (default, self-contained). "
                        "'dir'   = save image files to {output_jsonl_stem}_images/, store paths in 'images' list.")
    p.add_argument("--no_save_images", action="store_true",
                   help="Do not save images at all (reduces jsonl size ~100x). "
                        "Overrides --image_store.")
    p.add_argument("--force_search", action="store_true",
                   help="方案B: if zoom was called but search has not been called, inject a "
                        "user nudge prompting the model to call image_search_tool before answering")
    p.add_argument("--thumbnails", action="store_true",
                   help="Send image_search thumbnail images to the model alongside text results. "
                        "Thumbnails are sent as base64 images in a follow-up user message (Kimi) "
                        "or inline in tool content (235B). Increases context but improves <useful> accuracy.")
    args = p.parse_args()

    # --tools flag: parse comma-separated tool names and set derived flags
    # Supported: zoom, image_search, text_search
    # This overrides --use_tools/--use_tavily/--zoom_only when --model is set.
    _tools_set = set()
    if args.tools:
        for _t in args.tools.split(","):
            _t = _t.strip().lower()
            if _t in ("zoom", "image_zoom_in_tool"):
                _tools_set.add("zoom")
            elif _t in ("image_search", "image_search_tool"):
                _tools_set.add("image_search")
            elif _t in ("text_search", "text_search_tool", "tavily"):
                _tools_set.add("text_search")
            else:
                print(f"WARNING: unknown tool name '{_t}', ignoring. "
                      f"Valid names: zoom, image_search, text_search")
        # Map tool set → legacy flags
        args.zoom_only  = (_tools_set == {"zoom"})
        args.use_tools  = bool(_tools_set & {"zoom", "image_search"})
        args.use_tavily = "text_search" in _tools_set

    # Apply built-in prompt defaults; CLI overrides take precedence
    if args.system_prompt is None:
        if args.notool:
            args.system_prompt = SYSTEM_NOTOOL
        elif args.model and args.mode == "agent":
            # LLMClient agent mode: pick system prompt based on active tool set
            has_zoom   = "zoom"         in _tools_set if _tools_set else getattr(args, 'zoom_only', False) or getattr(args, 'use_tools', False)
            has_search = "image_search" in _tools_set if _tools_set else (getattr(args, 'use_tools', False) and not getattr(args, 'zoom_only', False))
            has_tavily = "text_search"  in _tools_set if _tools_set else getattr(args, 'use_tavily', False)
            if has_zoom and has_search and has_tavily:
                args.system_prompt = SYSTEM_TOOL_LLMCLIENT_ALL3_V2
            elif has_zoom and has_search:
                args.system_prompt = SYSTEM_TOOL_LLMCLIENT_ZOOM_SEARCH_V2
            elif has_zoom and has_tavily:
                args.system_prompt = SYSTEM_TOOL_LLMCLIENT_ZOOM_TAVILY_V2
            elif has_search and has_tavily:
                args.system_prompt = SYSTEM_TOOL_LLMCLIENT_SEARCH_TAVILY_V2
            elif has_zoom:
                args.system_prompt = SYSTEM_TOOL_LLMCLIENT_ZOOM_ONLY
            elif has_search:
                args.system_prompt = SYSTEM_TOOL_LLMCLIENT_SEARCH_ONLY
            elif has_tavily:
                args.system_prompt = SYSTEM_TOOL_LLMCLIENT_TAVILY_ONLY_V2
            else:
                args.system_prompt = SYSTEM_TOOL_LLMCLIENT
        else:
            args.system_prompt = SYSTEM_TOOL
    # --use_tavily implicitly enables tool use
    if getattr(args, 'use_tavily', False):
        args.use_tools = True
    if args.user_prompt is None:
        args.user_prompt = USER_PROMPT

    if args.output_jsonl is None:
        args.output_jsonl = f"{OUTPUT_DIR}/{args.benchmark}/{args.tag}.jsonl"

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
