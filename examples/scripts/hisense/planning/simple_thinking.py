"""Simple thinking: 两轮 planning 协议。

- Round 1: 调用 LLM 将 query 分解为 sub_queries，分配工具
- Round 2: 规则判断分数，补充工具或 stop
"""

import asyncio
import logging

from openai import AsyncOpenAI

from helpers import PlanningResponse, SubQueryItem
from query_rewrite import rewrite_query
from tool_selection import select_optional_tools
from workflow import (
    collect_all_results_from_history,
    count_total_qualified,
    get_sub_queries_with_used_tools,
    trim_history,
)

logger = logging.getLogger("planning_server.simple_thinking")


async def handle_simple_thinking(req, llm_client: AsyncOpenAI, llm_model: str,
                                 ts_client: AsyncOpenAI | None = None, ts_model: str = "",
                                 ts_threshold: float = 0) -> PlanningResponse:
    """Simple thinking 模式处理。"""

    if req.turn == 1:
        # ── 首轮: 调用 LLM 做 query rewrite ──
        logger.info("Turn 1 | query=%r, max_top_k=%d, tool_hub=%s", req.query, req.max_top_k, req.tool_hub)

        sub_queries = await rewrite_query(llm_client, llm_model, req.query, req.max_top_k)

        if sub_queries is None:
            return PlanningResponse(is_off_topic=False, status="stop", turn=req.turn, final={})

        if not sub_queries:
            logger.info("Off-topic detected, no sub_queries")
            return PlanningResponse(is_off_topic=True, status="stop", turn=req.turn, current=[])

        if req.query not in sub_queries:
            sub_queries.insert(0, req.query)

        logger.info("Turn 1 | sub_queries=%s", sub_queries)

        # ── tool_hub_optional: 根据 tool_selection_mode 处理 ──
        if req.tool_selection_mode == "model" and req.tool_hub_optional:
            threshold = req.tool_selection_threshold if req.tool_selection_threshold is not None else ts_threshold
            tool_hubs = await asyncio.gather(*(
                select_optional_tools(
                    ts_client, ts_model, sq, req.tool_hub, req.tool_hub_optional, threshold=threshold,
                ) for sq in sub_queries
            ))
            logger.info("Turn 1 | tool_selection_mode=model, per-subquery tool_hubs=%s", tool_hubs)
            current_items = [SubQueryItem(sub_query=sq, tool_use=th, topk=req.max_top_k) for sq, th in zip(sub_queries, tool_hubs)]
        else:
            current_items = [SubQueryItem(sub_query=sq, tool_use=req.tool_hub, topk=req.max_top_k) for sq in sub_queries]

        return PlanningResponse(is_off_topic=False, status="running", turn=req.turn, current=current_items)

    else:
        # ── 非首轮 ──
        logger.info("Turn %d | tool_selection_mode=%s, tool_hub=%s, tool_hub_optional=%s",
                     req.turn, req.tool_selection_mode, req.tool_hub, req.tool_hub_optional)

        score_threshold = req.retrieval_setting.score_threshold
        top_k = req.retrieval_setting.top_k

        total_qualified = count_total_qualified(req.history, score_threshold)
        logger.debug("Total qualified (threshold=%.4f): %d, top_k=%d", score_threshold, total_qualified, top_k)

        if total_qualified >= top_k:
            final = collect_all_results_from_history(req.history, original_query=req.query, top_k=top_k)
            logger.info("Turn %d | total qualified=%d >= top_k=%d, stop.", req.turn, total_qualified, top_k)
            return PlanningResponse(
                is_off_topic=False, status="stop", turn=req.turn, final=final,
                history=trim_history(req.history, req.max_context_size),
            )

        current_items = []

        if req.tool_selection_mode == "model" and req.tool_hub_optional and req.turn < req.max_turn:
            all_items = get_sub_queries_with_used_tools(req.history)
            sqs = [item["sub_query"] for item in all_items]
            if sqs:
                threshold = req.tool_selection_threshold if req.tool_selection_threshold is not None else ts_threshold
                tool_hubs = await asyncio.gather(*(
                    select_optional_tools(
                        ts_client, ts_model, sq, req.tool_hub, req.tool_hub_optional, threshold=threshold,
                    ) for sq in sqs
                ))
                for item, th in zip(all_items, tool_hubs):
                    new_tools = set(th.split(",")) - item["used_tools"]
                    if new_tools:
                        current_items.append(SubQueryItem(
                            sub_query=item["sub_query"],
                            tool_use=",".join(sorted(new_tools)),
                            topk=min(item["topk"], req.max_top_k),
                        ))
                logger.info("Turn %d | model mode, per-subquery tool_hubs=%s", req.turn, tool_hubs)

        elif req.tool_selection_mode == "rule" and req.tool_hub_optional and req.turn < req.max_turn:
            optional_tools = {t.strip() for t in req.tool_hub_optional.split(",") if t.strip()}
            all_items = get_sub_queries_with_used_tools(req.history)
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
            logger.info("Turn %d | no more tools to try, stop.", req.turn)
            return PlanningResponse(
                is_off_topic=False, status="stop", turn=req.turn, final=final,
                history=trim_history(req.history, req.max_context_size),
            )

        logger.info("Turn %d | running, %d sub_queries to search", req.turn, len(current_items))
        return PlanningResponse(
            is_off_topic=False, status="running", turn=req.turn,
            current=current_items, history=trim_history(req.history, req.max_context_size),
        )
