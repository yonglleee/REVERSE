# Copyright 2025 Reallm Labs Ltd. or its affiliates
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

"""
VERL tool for geo-localization reward (SpotSFT-200k).

The model calls `calc_geoloc_reward(answer=<str>)` to submit its
location prediction. The answer should be enclosed in <answer></answer> tags with
format: "Country, City, Latitude, Longitude".
The tool computes a haversine-distance-based reward and returns feedback.
"""

import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.utils.reward_score import geoloc
from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SpotSFTGeolocTool(BaseTool):
    """Tool for geo-localization reward in SpotSFT RL training.

    Schema (registered name: ``calc_geoloc_reward``):

    .. code-block:: json

        {
          "type": "function",
          "function": {
            "name": "calc_geoloc_reward",
            "description": "Submit your geolocation prediction. The answer must be in <answer>Country, City, Latitude, Longitude</answer> format.",
            "parameters": {
              "type": "object",
              "properties": {
                "answer": {
                  "type": "string",
                  "description": "The predicted location in <answer></answer> tags, e.g. <answer>USA, New York, 40.71, -74.01</answer>"
                }
              },
              "required": ["answer"]
            }
          }
        }
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict] = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        **kwargs,
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        if ground_truth is None:
            ground_truth = kwargs.get("create_kwargs", {}).get("ground_truth", "")
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth or "",
            "best_reward": 0.0,
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs,
    ) -> tuple[ToolResponse, float, dict]:
        answer = parameters.get("answer", "")
        if not isinstance(answer, str):
            answer = str(answer)

        self._instance_dict[instance_id]["response"] = answer
        reward = await self.calc_reward(instance_id)

        # Penalty for non-improved answer submission (same as geo3k)
        best = self._instance_dict[instance_id]["best_reward"]
        tool_reward = 0.0 if reward > best else -0.05
        self._instance_dict[instance_id]["best_reward"] = max(reward, best)

        return ToolResponse(text=f"Current parsed {answer=} {reward=}"), tool_reward, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        state = self._instance_dict[instance_id]
        # print(state["response"], state["ground_truth"])
        return geoloc._compute_score(
            predict_str=state["response"],
            ground_truth=state["ground_truth"],
            format_score=0.1,
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
