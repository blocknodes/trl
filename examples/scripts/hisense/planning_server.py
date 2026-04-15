"""
Planning Server: FastAPI 服务，实现两轮 planning 协议。

协议流程 (最多两轮):
- Round 1: 调用 LLM 将 query 分解为 sub_queries，分配 tool_hub 中的工具，返回 status="running"
- Round 2: 纯规则判断 - 遍历 history 中检索结果的分数，低于阈值的 sub_query 用 tool_hub_optional
  补充工具调用 (status="running")；若所有分数都达标或无 optional tools，直接 status="stop" + final

用法:
    python examples/scripts/hisense/planning_server.py \
        --base-url http://localhost:8078/v1 \
        --model qwen4b \
        --port 9100
"""

import argparse
import json
import logging
import re

from fastapi import FastAPI
from pydantic import BaseModel
from openai import AsyncOpenAI

logger = logging.getLogger("planning_server")

# 自定义 VERBOSE 级别 (低于 DEBUG)，用于打印完整输入输出内容
VERBOSE = 5
logging.addLevelName(VERBOSE, "VERBOSE")


# ── System Prompt: 和 query_rewriter.py 完全一致 ────────────────

SYSTEM_PROMPT = """\
You are a query rewriter. Given a user query and a topk number, rewrite the query into a list of atomic sub-queries for search.

Rules:
1. Each sub-query should be atomic: it asks about exactly one thing.
2. The number of sub-queries must NOT exceed the given topk.
3. All sub-queries together should fully cover the intent of the original query.
4. If the original query is not suitable for search (e.g. casual chat, greetings), return an empty list.
5. For comparison queries involving multiple subjects, decompose by subject first, NOT by attribute. \
For example, "410和510有啥区别" should be decomposed into "410特点", "510特点", \
NOT "410和510性能区别", "410和510价格区别".
6. If the user query is already atomic (asks about exactly one thing), do NOT split it. \
At most normalize it: convert verbose colloquial expressions into concise standard form.

You MUST respond with a JSON object in the following format:
{"sub_queries": ["query1", "query2", ...]}

If the query is not suitable for search:
{"sub_queries": []}

Examples:

User: query: "E8Q电视的像素和刷新率分别是多少？", topk: 3
Output: {"sub_queries": ["E8Q电视的像素是多少", "E8Q电视的刷新率是多少"]}

User: query: "410和510有啥区别", topk: 3
Output: {"sub_queries": ["410特点", "510特点"]}

User: query: "对比U8Q和小米S pro Mini LED的画质和价格", topk: 2
Output: {"sub_queries": ["U8Q画质和价格", "小米S pro Mini LED画质和价格"]}

User: query: "你好呀", topk: 3
Output: {"sub_queries": []}
"""



# ── FastAPI App ─────────────────────────────────────────────────

app = FastAPI(title="Planning Server")

_llm_client: AsyncOpenAI | None = None
_llm_model: str = ""


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


class PlanningResponse(BaseModel):
    is_off_topic: bool = False
    status: str = "running"
    turn: int = 1
    current: list[SubQueryItem] | None = None
    history: dict | None = None
    final: dict | None = None


# ── Helper functions ────────────────────────────────────────────

def _parse_llm_json(content: str) -> dict | None:
    """从 LLM 输出中解析 JSON，支持 markdown 包裹和 think 块。"""
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    # Direct parse
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    # ```json ... ```
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned.rstrip()).strip()
        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            pass

    # JSON embedded in text
    m = re.search(r"\{[\s\S]*\}", cleaned)
    if m:
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _collect_all_results_from_history(history: dict | None) -> dict:
    """从 history 中收集所有检索结果，按工具类型分组去重，用于 final 输出。"""
    if not history:
        return {}
    results_by_tool: dict[str, list[dict]] = {}
    seen: dict[str, set] = {}
    for turn_key in sorted(history.keys()):
        turn_data = history[turn_key]
        contents = turn_data.get("retrieval_contents", [])
        for item in contents:
            result = item.get("result", {})
            for tool_type, records in result.items():
                if tool_type not in results_by_tool:
                    results_by_tool[tool_type] = []
                    seen[tool_type] = set()
                for record in records:
                    key = (record.get("title", ""), record.get("content", "")[:100])
                    if key not in seen[tool_type]:
                        seen[tool_type].add(key)
                        results_by_tool[tool_type].append(record)
    return results_by_tool


def _get_sub_queries_with_used_tools(history: dict | None) -> list[dict]:
    """遍历 history，收集所有 sub_query 及其已使用过的工具。

    返回 [{"sub_query": str, "topk": int, "used_tools": set}, ...] 列表。
    """
    if not history:
        return []

    agg: dict[str, dict] = {}
    for turn_key in sorted(history.keys()):
        turn_data = history[turn_key]
        contents = turn_data.get("retrieval_contents", [])
        for item in contents:
            sq = item.get("sub_query", "")
            if sq not in agg:
                agg[sq] = {"topk": item.get("topk", 3), "used_tools": set()}
            tools = item.get("tool_use", "")
            for t in tools.split(","):
                t = t.strip()
                if t:
                    agg[sq]["used_tools"].add(t)

    return [{"sub_query": sq, "topk": info["topk"], "used_tools": info["used_tools"]} for sq, info in agg.items()]


def _trim_history(history: dict | None, max_context_size: int) -> dict | None:
    """只保留最近 max_context_size 轮的 history。"""
    if not history:
        return None
    turns = sorted(history.keys())
    if len(turns) <= max_context_size:
        return history
    recent = turns[-max_context_size:]
    return {k: history[k] for k in recent}


# ── API Endpoint ────────────────────────────────────────────────

@app.post("/planner", response_model=PlanningResponse)
async def planner(req: PlanningRequest):
    """单一 planning 端点。

    Round 1 (turn==1): 调用 LLM 分解 query → sub_queries + 原始 query，用 tool_hub 所有工具搜索
    Round 2+ (turn>1): 纯规则 - 检查每个 sub_query 达标结果数是否满足 top_k，
                       不足的用尚未使用的工具（含 tool_hub_optional）补充
    停止条件: 达到 max_turn / 所有 sub_query 达标 / 所有工具已用完
    """
    logger.log(VERBOSE, "Request input:\n%s", json.dumps(req.model_dump(), ensure_ascii=False, indent=2, default=str))
    logger.debug("Request summary: turn=%d, query=%r, tool_hub=%s", req.turn, req.query, req.tool_hub)

    def _respond(resp: PlanningResponse) -> PlanningResponse:
        logger.log(VERBOSE, "Response output:\n%s", json.dumps(resp.model_dump(), ensure_ascii=False, indent=2, default=str))
        logger.debug("Response summary: status=%s, turn=%d, current=%d items",
                      resp.status, resp.turn, len(resp.current) if resp.current else 0)
        return resp

    if req.turn == 1:
        # ── 首轮: 调用 LLM 做 query rewrite，得到 sub_queries ──
        logger.info("Turn 1 | query=%r, max_top_k=%d, tool_hub=%s", req.query, req.max_top_k, req.tool_hub)
        user_content = f'query: "{req.query}"\ntopk: {req.max_top_k}'

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        completion = await _llm_client.chat.completions.create(
            model=_llm_model,
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
        )

        content = completion.choices[0].message.content or ""
        logger.log(VERBOSE, "LLM raw response:\n%s", content)
        logger.debug("LLM response length: %d chars", len(content))
        parsed = _parse_llm_json(content)

        if parsed is None:
            logger.warning("Failed to parse LLM JSON output, returning stop")
            return _respond(PlanningResponse(
                is_off_topic=False,
                status="stop",
                turn=req.turn,
                final={},
            ))

        sub_queries = parsed.get("sub_queries", [])

        # 空列表 = 闲聊 / 不适合搜索
        if not sub_queries:
            logger.info("Off-topic detected, no sub_queries")
            return _respond(PlanningResponse(
                is_off_topic=True,
                status="stop",
                turn=req.turn,
                current=[],
            ))

        # 将原始 query 加入 sub_queries（如果不在其中）
        if req.query not in sub_queries:
            sub_queries.insert(0, req.query)

        logger.info("Turn 1 | sub_queries=%s", sub_queries)

        # 每个 sub_query 用 tool_hub 中的所有工具
        current_items = []
        for sq in sub_queries:
            current_items.append(SubQueryItem(
                sub_query=sq,
                tool_use=req.tool_hub,
                topk=req.max_top_k,
            ))

        return _respond(PlanningResponse(
            is_off_topic=False,
            status="running",
            turn=req.turn,
            current=current_items,
        ))

    else:
        # ── 非首轮: 纯规则，不调 LLM ──
        logger.info("Turn %d | rule-based, tool_hub=%s, tool_hub_optional=%s",
                     req.turn, req.tool_hub, req.tool_hub_optional)

        # 所有可用工具 = tool_hub + tool_hub_optional
        all_tools: set[str] = set()
        for t in req.tool_hub.split(","):
            t = t.strip()
            if t:
                all_tools.add(t)
        if req.tool_hub_optional:
            for t in req.tool_hub_optional.split(","):
                t = t.strip()
                if t:
                    all_tools.add(t)

        # 收集所有 sub_query 及其已用工具
        all_items = _get_sub_queries_with_used_tools(req.history)
        logger.debug("Sub-queries with used tools: %s",
                      [(i["sub_query"], i["used_tools"]) for i in all_items])

        # 对所有 sub_query，找出尚未使用的工具
        current_items = []
        if req.turn < req.max_turn:
            for item in all_items:
                remaining_tools = all_tools - item["used_tools"]
                if remaining_tools:
                    logger.debug("sub_query=%r remaining_tools=%s", item["sub_query"], remaining_tools)
                    current_items.append(SubQueryItem(
                        sub_query=item["sub_query"],
                        tool_use=",".join(sorted(remaining_tools)),
                        topk=min(item["topk"], req.max_top_k),
                    ))

        # 停止条件: 达到 max_turn / 所有 sub_query 的工具都已用完
        if not current_items:
            final = _collect_all_results_from_history(req.history)
            logger.info("Turn %d | stop, final has %d tool types", req.turn, len(final))
            return _respond(PlanningResponse(
                is_off_topic=False,
                status="stop",
                turn=req.turn,
                final=final,
                history=_trim_history(req.history, req.max_context_size),
            ))

        logger.info("Turn %d | running, %d sub_queries to search", req.turn, len(current_items))
        return _respond(PlanningResponse(
            is_off_topic=False,
            status="running",
            turn=req.turn,
            current=current_items,
            history=_trim_history(req.history, req.max_context_size),
        ))


# ── 启动入口 ────────────────────────────────────────────────────

def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Planning Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--base-url", default="http://localhost:8078/v1", help="vLLM server base URL")
    parser.add_argument("--model", default="qwen4b", help="Model name")
    parser.add_argument("--log-level", default="INFO",
                        choices=["VERBOSE", "DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Console logging level (default: INFO)")
    parser.add_argument("--file-log-level", default="DEBUG",
                        choices=["VERBOSE", "DEBUG", "INFO", "WARNING", "ERROR"],
                        help="File logging level (default: DEBUG)")
    args = parser.parse_args()

    import datetime as _dt
    _log_file = f"planning_server_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    _log_fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    _console = logging.StreamHandler()
    _console.setLevel(VERBOSE if args.log_level == "VERBOSE" else getattr(logging, args.log_level))
    _fh = logging.FileHandler(_log_file)
    _fh.setLevel(VERBOSE if args.file_log_level == "VERBOSE" else getattr(logging, args.file_log_level))
    logging.basicConfig(level=VERBOSE, format=_log_fmt, handlers=[_fh, _console])

    global _llm_client, _llm_model
    _llm_client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")
    _llm_model = args.model

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
