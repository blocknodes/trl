"""
Planning Server: FastAPI 服务，实现两轮 planning 协议。

协议流程 (最多两轮):
- Round 1: 调用 LLM 将 query 分解为 sub_queries，分配 tool_hub 中的工具，返回 status="running"
- Round 2: 纯规则判断 - 遍历 history 中检索结果的分数，低于阈值的 sub_query 用 tool_hub_optional
  补充工具调用 (status="running")；若所有分数都达标或无 optional tools，直接 status="stop" + final

用法:
    python examples/scripts/hisense/planning_server.py \\
        --base-url http://localhost:8078/v1 \\
        --model qwen4b \\
        --port 9100
"""

import argparse
import json
import logging

from fastapi import FastAPI
from openai import AsyncOpenAI

from helpers import VERBOSE, PlanningRequest, PlanningResponse, SubQueryItem
from query_rewrite import rewrite_query
from deep_thinking import handle_deep_thinking, set_session_ttl
from dynamic_thinking import handle_dynamic_thinking
from workflow import (
    collect_all_results_from_history,
    count_total_qualified,
    get_sub_queries_with_used_tools,
    trim_history,
)

logger = logging.getLogger("planning_server")

app = FastAPI(title="Planning Server")

_llm_client: AsyncOpenAI | None = None
_llm_model: str = ""
_planner_client: AsyncOpenAI | None = None
_planner_model: str = ""


@app.post("/planner", response_model=PlanningResponse)
async def planner(req: PlanningRequest):
    """单一 planning 端点。"""
    logger.log(VERBOSE, "Request input:\n%s", json.dumps(req.model_dump(), ensure_ascii=False, indent=2, default=str))
    logger.debug("Request summary: turn=%d, query=%r, tool_hub=%s, tool_hub_optional=%s, "
                 "score_threshold=%.4f, top_k=%d, max_top_k=%d, deep_thinking=%s",
                 req.turn, req.query, req.tool_hub, req.tool_hub_optional,
                 req.retrieval_setting.score_threshold, req.retrieval_setting.top_k,
                 req.max_top_k, req.deep_thinking)

    def _respond(resp: PlanningResponse) -> PlanningResponse:
        logger.log(VERBOSE, "Response output:\n%s", json.dumps(resp.model_dump(), ensure_ascii=False, indent=2, default=str))
        current_summary = ""
        if resp.current:
            parts = [f"({c.sub_query} -> [{c.tool_use}] topk={c.topk})" for c in resp.current]
            current_summary = ", ".join(parts)
        final_tools = list(resp.final.keys()) if resp.final else []
        logger.debug("Response summary: status=%s, turn=%d, current=[%s], final_tools=%s",
                      resp.status, resp.turn, current_summary, final_tools)
        return resp

    # ── Deep Thinking 模式 ──
    if req.deep_thinking:
        return await handle_deep_thinking(
            req, _llm_client, _llm_model, _planner_client, _planner_model, trim_history,
        )

    # ── Dynamic Thinking (React) 模式 ──
    if req.dynamic_thinking:
        return await handle_dynamic_thinking(
            req, _llm_client, _llm_model, _planner_client, _planner_model, trim_history,
        )

    if req.turn == 1:
        # ── 首轮: 调用 LLM 做 query rewrite ──
        logger.info("Turn 1 | query=%r, max_top_k=%d, tool_hub=%s", req.query, req.max_top_k, req.tool_hub)

        sub_queries = await rewrite_query(_llm_client, _llm_model, req.query, req.max_top_k)

        if sub_queries is None:
            return _respond(PlanningResponse(is_off_topic=False, status="stop", turn=req.turn, final={}))

        if not sub_queries:
            logger.info("Off-topic detected, no sub_queries")
            return _respond(PlanningResponse(is_off_topic=True, status="stop", turn=req.turn, current=[]))

        if req.query not in sub_queries:
            sub_queries.insert(0, req.query)

        logger.info("Turn 1 | sub_queries=%s", sub_queries)
        current_items = [SubQueryItem(sub_query=sq, tool_use=req.tool_hub, topk=req.max_top_k) for sq in sub_queries]

        return _respond(PlanningResponse(is_off_topic=False, status="running", turn=req.turn, current=current_items))

    else:
        # ── 非首轮: 纯规则判断 ──
        logger.info("Turn %d | rule-based, tool_hub=%s, tool_hub_optional=%s",
                     req.turn, req.tool_hub, req.tool_hub_optional)

        score_threshold = req.retrieval_setting.score_threshold
        top_k = req.retrieval_setting.top_k

        total_qualified = count_total_qualified(req.history, score_threshold)
        logger.debug("Total qualified (threshold=%.4f): %d, top_k=%d", score_threshold, total_qualified, top_k)

        if total_qualified >= top_k:
            final = collect_all_results_from_history(req.history, original_query=req.query, top_k=top_k)
            logger.info("Turn %d | total qualified=%d >= top_k=%d, stop.", req.turn, total_qualified, top_k)
            return _respond(PlanningResponse(
                is_off_topic=False, status="stop", turn=req.turn, final=final,
                history=trim_history(req.history, req.max_context_size),
            ))

        optional_tools: set[str] = set()
        if req.tool_hub_optional:
            for t in req.tool_hub_optional.split(","):
                t = t.strip()
                if t:
                    optional_tools.add(t)

        all_items = get_sub_queries_with_used_tools(req.history)
        current_items = []
        if req.turn < req.max_turn and optional_tools:
            for item in all_items:
                remaining_tools = optional_tools - item["used_tools"]
                if remaining_tools:
                    logger.debug("sub_query=%r optional remaining_tools=%s", item["sub_query"], remaining_tools)
                    current_items.append(SubQueryItem(
                        sub_query=item["sub_query"],
                        tool_use=",".join(sorted(remaining_tools)),
                        topk=min(item["topk"], req.max_top_k),
                    ))

        if not current_items:
            final = collect_all_results_from_history(req.history, original_query=req.query, top_k=top_k)
            logger.info("Turn %d | no optional tools to try, stop.", req.turn)
            return _respond(PlanningResponse(
                is_off_topic=False, status="stop", turn=req.turn, final=final,
                history=trim_history(req.history, req.max_context_size),
            ))

        logger.info("Turn %d | running, %d sub_queries to search", req.turn, len(current_items))
        return _respond(PlanningResponse(
            is_off_topic=False, status="running", turn=req.turn,
            current=current_items, history=trim_history(req.history, req.max_context_size),
        ))


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Planning Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--base-url", default="http://localhost:8078/v1", help="vLLM server base URL")
    parser.add_argument("--model", default="qwen4b", help="Model name")
    parser.add_argument("--planner-base-url", default="https://aix-backup.hismarttv.com/v1",
                        help="Planner LLM base URL for deep_thinking")
    parser.add_argument("--planner-model", default="deepseek-v3",
                        help="Planner model name for deep_thinking")
    parser.add_argument("--planner-api-key", default="<api-key>",
                        help="Planner LLM API key")
    parser.add_argument("--session-ttl", type=int, default=600,
                        help="Deep thinking session TTL in seconds (default: 600)")
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

    global _llm_client, _llm_model, _planner_client, _planner_model
    _llm_client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")
    _llm_model = args.model
    _planner_model = args.planner_model
    _planner_client = AsyncOpenAI(base_url=args.planner_base_url, api_key=args.planner_api_key)
    set_session_ttl(args.session_ttl)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
