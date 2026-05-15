# -*- coding: utf-8 -*-
"""
llm_client.py - 统一大模型调用客户端

支持四个模型：
  - 235b      : Qwen3-VL-235B，py3meshkit gRPC，需在 gemini container 运行
  - qwen122b  : Qwen3.5-122B，Astra Starlink broker + 直连 vllm
  - qwen397b  : Qwen3.5-397B，Astra Starlink broker + 直连 vllm
  - kimi      : Kimi K2，Astra Starlink broker + 直连 vllm

使用示例：

    from llm_client import LLMClient

    client = LLMClient(model="235b")      # Qwen3-VL-235B
    client = LLMClient(model="qwen122b")  # Qwen3.5-122B
    client = LLMClient(model="qwen397b")  # Qwen3.5-397B
    client = LLMClient(model="kimi")      # Kimi K2

    # 纯文本
    resp = client.chat("你好")

    # 图片 + 文本（本地路径）
    resp = client.chat("这张图片在哪里拍的？", images=["path/to/img.jpg"])

    # 图片 + 文本（已有 base64）
    resp = client.chat("Where is this?", images_b64=[b64_str])

    # 完全自定义 content list
    resp = client.chat_raw(
        content=[
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": "Where is this?"},
        ],
        system="You are a geolocation expert.",
    )

    # 多轮对话
    resp = client.chat_messages([
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "你叫什么"},
    ])
"""

import io
import sys
import base64
import time
import random
import socket
import struct
import requests as _requests
from pathlib import Path
from typing import Optional
from PIL import Image
from openai import OpenAI


# ── 235B 配置（py3meshkit gRPC） ───────────────────────────────────────────────
_235B_SERVICE  = "GLM_4_5"
_235B_PORT     = 8080
_235B_LIB_PATH = "/mnt/sh/mmvision/home/jonahli/projects/UUT/data_preprocess/server/qwen3vl235b"

# ── Astra Starlink broker 配置 ─────────────────────────────────────────────────
_BROKER_URL  = "http://astrastarlinkbroker.production.polaris:12534/OPEN_API_GetMachineListByApp"
_VLLM_PORT   = 8000

# broker appid 映射
_BROKER_APPIDS = {
    "qwen122b":  "Qwen3d5_122B_A10B",
    "qwen397b":  "qwen3d5_397b_a17b",
    "kimi_k2d5": "kimi_k2d5",
    "kimi_k2d6": "kimi_k2d6",
}

MODELS = ("235b", "qwen122b", "qwen397b", "kimi_k2d6", "kimi_k2d5")


# ── 图片编码 ───────────────────────────────────────────────────────────────────

def encode_image(path: str, max_pixels: int = 2048 * 1024) -> str:
    """本地图片路径 -> base64 JPEG 字符串。"""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


# ── 235B 懒加载 ────────────────────────────────────────────────────────────────

def _load_235b_class():
    """懒加载 Qwen3VL235B（需要 py3meshkit，仅 gemini container）。"""
    sys.path.insert(0, _235B_LIB_PATH)
    try:
        from Qwen3_VL_235B import Qwen3VL235B
        return Qwen3VL235B
    except ImportError as e:
        raise RuntimeError("py3meshkit 未安装，235B 只能在 gemini container 运行: {}".format(e))


# ── Starlink broker IP 获取 ────────────────────────────────────────────────────

def _get_vllm_ip(appid: str) -> str:
    """从 Astra Starlink broker 获取一个可用 vllm 机器 IP。"""
    res = _requests.post(_BROKER_URL, json={"appid": appid}, timeout=5)
    machines = res.json()["machine_list"]["list"]
    if not machines:
        raise RuntimeError("broker 返回空机器列表，appid={}".format(appid))
    ips = [socket.inet_ntoa(struct.pack("!I", x["attr"]["ip"])) for x in machines]
    return random.choice(ips)


# ── 统一客户端 ─────────────────────────────────────────────────────────────────

class LLMClient:
    """
    统一调用接口。

    model: "235b"      - Qwen3-VL-235B，py3meshkit gRPC，gemini container 专用
           "qwen122b"  - Qwen3.5-122B，Starlink broker + 直连 vllm
           "qwen397b"  - Qwen3.5-397B，Starlink broker + 直连 vllm
           "kimi_k2d6" - Kimi K2.6，Starlink broker + 直连 vllm
    """

    def __init__(self, model: str = "235b", retry: int = 3):
        assert model in MODELS, \
            "model 须为 {} 之一，got: {}".format(MODELS, model)
        self.model    = model
        self.retry    = retry
        self._cls235b = None  # 懒加载

    # ── 内部调用 ───────────────────────────────────────────────────────────────

    def _call_235b(self, content: list) -> Optional[str]:
        if self._cls235b is None:
            self._cls235b = _load_235b_class()
        for attempt in range(self.retry):
            try:
                return self._cls235b(service=_235B_SERVICE)(content)
            except Exception as e:
                print("[235B] attempt {}/{}: {}".format(attempt + 1, self.retry, e))
                if attempt < self.retry - 1:
                    time.sleep(2 ** attempt)
        return None

    def _call_235b_multiturn(self, messages: list, max_tokens: int = 2048,
                              tools: Optional[list] = None) -> Optional[str]:
        """多轮对话，直接用 OpenAI client 发 messages，支持 tools。"""
        if self._cls235b is None:
            self._cls235b = _load_235b_class()
        for attempt in range(self.retry):
            try:
                import asyncio
                from Qwen3_VL_235B import schedule_fast_agent_job
                ip = asyncio.run(schedule_fast_agent_job(_235B_SERVICE, "test", 60000))
                print("[235B multiturn] ip={}".format(ip))
                client = OpenAI(api_key="EMPTY", base_url="http://{}:{}/v1".format(ip, _235B_PORT))
                model_name = client.models.list().data[0].id

                kwargs = dict(model=model_name, messages=messages,
                              temperature=1e-6, max_tokens=max_tokens)
                if tools:
                    kwargs["tools"] = tools

                resp = client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message

                if msg.tool_calls:
                    tc = msg.tool_calls[0]
                    tool_str = '<tool_call>{{"name": "{}", "arguments": {}}}</tool_call>'.format(
                        tc.function.name, tc.function.arguments)
                    prefix = (msg.content or "").rstrip()
                    return (prefix + "\n" + tool_str) if prefix else tool_str
                return msg.content or ""
            except Exception as e:
                print("[235B multiturn] attempt {}/{}: {}".format(attempt + 1, self.retry, e))
                if attempt < self.retry - 1:
                    time.sleep(2 ** attempt)
        return None

    def _call_vllm(self, messages: list, max_tokens: int = 2048,
                   tools: Optional[list] = None,
                   extra_body: Optional[dict] = None) -> Optional[str]:
        """通过 Starlink broker 发现 IP，直连 vllm（qwen122b / qwen397b / kimi）。"""
        appid = _BROKER_APPIDS[self.model]
        for attempt in range(self.retry):
            try:
                ip = _get_vllm_ip(appid)
                client = OpenAI(api_key="EMPTY",
                                base_url="http://{}:{}/v1".format(ip, _VLLM_PORT))
                model_name = client.models.list().data[0].id

                kwargs = dict(
                    model=model_name,
                    messages=messages,
                    temperature=1e-6,
                    max_tokens=max_tokens,
                )
                # Qwen 思考模型关掉 thinking；kimi 不传该参数
                if self.model in ("qwen122b", "qwen397b"):
                    kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
                if extra_body:
                    kwargs["extra_body"] = {**kwargs.get("extra_body", {}), **extra_body}
                if tools:
                    kwargs["tools"] = tools

                resp = client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message

                # 处理 tool_calls（kimi native tool call）
                if msg.tool_calls:
                    tc = msg.tool_calls[0]
                    tool_str = '<tool_call>{{"name": "{}", "arguments": {}}}</tool_call>'.format(
                        tc.function.name, tc.function.arguments)
                    # 拼接 reasoning_content（如有）
                    thinking = getattr(msg, "reasoning_content", None) or ""
                    prefix = (msg.content or "").rstrip()
                    if thinking:
                        prefix = "<think>\n{}\n</think>".format(thinking.strip()) + (
                            "\n" + prefix if prefix else "")
                    return (prefix + "\n" + tool_str) if prefix else tool_str

                # 处理 reasoning_content（kimi 推理内容）
                thinking = getattr(msg, "reasoning_content", None) or ""
                content = msg.content or ""
                if thinking:
                    return "<think>\n{}\n</think>\n{}".format(thinking.strip(), content)
                return content

            except Exception as e:
                print("[{}] attempt {}/{}: {}".format(self.model, attempt + 1, self.retry, e))
                if attempt < self.retry - 1:
                    time.sleep(2 ** attempt)
        return None

    # ── 公开接口 ───────────────────────────────────────────────────────────────

    def chat_raw(self, content: list,
                 system: Optional[str] = None,
                 temperature: float = 1e-6,
                 max_tokens: int = 2048,
                 extra_body: Optional[dict] = None) -> Optional[str]:
        """传入 OpenAI content list 直接调用（单轮）。"""
        if self.model == "235b":
            full = list(content)
            if system:
                full.append({"type": "text", "text": system})
            return self._call_235b(full)
        else:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": content})
            return self._call_vllm(messages, max_tokens=max_tokens, extra_body=extra_body)

    def chat_messages(self, messages: list, max_tokens: int = 2048,
                      tools: Optional[list] = None) -> Optional[str]:
        """多轮对话：直接传入 OpenAI messages 列表。"""
        if self.model == "235b":
            return self._call_235b_multiturn(messages, max_tokens=max_tokens, tools=tools)
        else:
            return self._call_vllm(messages, max_tokens=max_tokens, tools=tools)

    def chat(self,
             prompt: str,
             images: Optional[list] = None,
             images_b64: Optional[list] = None,
             system: Optional[str] = None,
             temperature: float = 1e-6,
             max_tokens: int = 2048) -> Optional[str]:
        """
        便捷接口：图片（路径或 base64）+ 文本 prompt。

        images:     本地图片路径列表，自动编码为 base64
        images_b64: base64 字符串列表（带或不带 data: 前缀均可）
        """
        content = []
        for path in (images or []):
            b64 = encode_image(path)
            content.append({"type": "image_url",
                             "image_url": {"url": "data:image/jpeg;base64,{}".format(b64)}})
        for b64 in (images_b64 or []):
            url = b64 if b64.startswith("data:") else "data:image/jpeg;base64,{}".format(b64)
            content.append({"type": "image_url", "image_url": {"url": url}})
        content.append({"type": "text", "text": prompt})

        return self.chat_raw(content, system=system,
                             temperature=temperature, max_tokens=max_tokens)
