# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
from typing import Any, Optional
from uuid import uuid4

import ray
import ray.actor

from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _call_tavily(query: str, api_key: str, max_results: int = 3, timeout: int = 30) -> list[dict]:
    """Call Tavily search API for a single query.

    Returns list of {"title": ..., "url": ..., "content": ...} dicts.
    """
    import requests

    url = "https://api.tavily.com/search"
    payload = {
        "query": query,
        "api_key": api_key,
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data.get("results", [])


def _format_results(query: str, results: list[dict]) -> str:
    """Format Tavily results into readable text, matching SearchTool output style."""
    if not results:
        return f"No results found for query: {query}"
    parts = [f"Search results for: {query}\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "")
        parts.append(f"[{i}] {title}\nURL: {url}\n{content}\n")
    return "\n".join(parts)


class TavilySearchTool(BaseTool):
    """Search tool backed by Tavily API.

    Config keys:
        api_key (str): Tavily API key. Falls back to env var TAVILY_API_KEY.
        topk (int): Number of results per query. Default: 3.
        timeout (int): HTTP timeout in seconds. Default: 30.
        num_workers (int): Ray actor max concurrency. Default: 50.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        self.api_key = config.get("api_key") or os.environ.get("TAVILY_API_KEY", "")
        assert self.api_key, "Tavily API key must be set via config.api_key or env TAVILY_API_KEY"

        self.topk = config.get("topk", 3)
        self.timeout = config.get("timeout", 30)
        num_workers = config.get("num_workers", 50)

        # Ray actor for concurrent execution
        self._pool = (
            ray.remote(_TavilyWorker)
            .options(max_concurrency=num_workers)
            .remote(api_key=self.api_key, topk=self.topk, timeout=self.timeout)
        )

        logger.info(f"Initialized TavilySearchTool with topk={self.topk}, timeout={self.timeout}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"results": []}
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        query_list = parameters.get("query_list")
        if not query_list or not isinstance(query_list, list):
            err = "Error: 'query_list' is missing, empty, or not a list."
            logger.error(f"[TavilySearchTool] {err}")
            return ToolResponse(text=json.dumps({"result": err})), 0.0, {}

        try:
            result_text, metadata = await self._pool.search_batch.remote(query_list)
            self._instance_dict[instance_id]["results"].append(result_text.strip())
            metrics = {
                "query_count": metadata.get("query_count", 0),
                "total_results": metadata.get("total_results", 0),
                "status": metadata.get("status", "unknown"),
                "api_request_error": metadata.get("api_request_error", ""),
            }
            return ToolResponse(text=result_text), 0.0, metrics

        except Exception as e:
            err = f"Search execution failed: {e}"
            logger.error(f"[TavilySearchTool] {err}")
            return ToolResponse(text=json.dumps({"result": err})), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs):
        return self._instance_dict[instance_id]["results"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


@ray.remote
class _TavilyWorker:
    """Ray actor that executes Tavily searches."""

    def __init__(self, api_key: str, topk: int, timeout: int):
        self.api_key = api_key
        self.topk = topk
        self.timeout = timeout

    def search_batch(self, query_list: list[str]) -> tuple[str, dict]:
        """Search all queries and return combined text + metadata."""
        all_text = []
        total_results = 0
        errors = []

        for query in query_list:
            try:
                results = _call_tavily(query, self.api_key, self.topk, self.timeout)
                total_results += len(results)
                all_text.append(_format_results(query, results))
            except Exception as e:
                errors.append(str(e))
                logger.warning(f"[TavilyWorker] Query '{query}' failed: {e}")
                all_text.append(f"Error searching for '{query}': {e}")

        metadata = {
            "query_count": len(query_list),
            "total_results": total_results,
            "status": "error" if errors else "success",
            "api_request_error": "; ".join(errors),
        }
        return "\n\n".join(all_text), metadata
