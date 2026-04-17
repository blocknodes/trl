"""共享的数据模型和工具函数。"""

import json
import logging
import re

from pydantic import BaseModel

logger = logging.getLogger("planning_server.helpers")

# 自定义 VERBOSE 级别 (低于 DEBUG)
VERBOSE = 5
logging.addLevelName(VERBOSE, "VERBOSE")


# ── Request / Response Models ───────────────────────────────────

class RetrievalSetting(BaseModel):
    top_k: int = 1
    score_threshold: float = 0
    search_mode: str = "hybrid"
    search_strategy: str = "precise"


class SubQueryItem(BaseModel):
    sub_query: str
    tool_use: str
    topk: int


class PlanningRequest(BaseModel):
    query: str
    retrieval_setting: RetrievalSetting
    turn: int
    max_turn: int = 3
    max_top_k: int = 3
    max_context_size: int = 3
    tool_hub: str = "es,graph,web"
    tool_hub_optional: str = ""
    history: dict | None = None
    deep_thinking: bool = False
    dynamic_thinking: bool = False


class PlanningResponse(BaseModel):
    is_off_topic: bool = False
    status: str = "running"
    turn: int = 1
    current: list[SubQueryItem] | None = None
    history: dict | None = None
    final: dict | None = None
    step_summary: str | None = None
    answer: str | None = None
    reference_tree: list | None = None


# ── JSON 解析 ───────────────────────────────────────────────────

def _parse_llm_json(content: str) -> dict | None:
    """从 LLM 输出中解析 JSON，支持 markdown 包裹和 think 块。"""
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned.rstrip()).strip()
        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            pass

    m = re.search(r"\{[\s\S]*\}", cleaned)
    if m:
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            pass

    return None
