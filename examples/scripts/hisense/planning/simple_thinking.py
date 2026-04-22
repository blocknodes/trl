"""Simple thinking: 两轮 planning 协议。

- Round 1: 调用 LLM 将 query 分解为 sub_queries，分配工具
- Round 2: 规则判断分数，补充工具或 stop

tool_select_enable=True 时:
  若 tool_hub_optional 含 graph/es/struct/unstruct，调用 tool_selection model
  判断 query 是否适合结构化搜索 → 适合用 graph，否则用 es
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

# struct/unstruct -> graph/es 别名映射
_TOOL_ALIAS = {"struct": "graph", "unstruct": "es"}
_TOOL_ALIAS_REV = {v: k for k, v in _TOOL_ALIAS.items()}


def _normalize_tool(t: str) -> str:
    return _TOOL_ALIAS.get(t, t)


def _normalize_tools(tools_str: str) -> str:
    return ",".join(_normalize_tool(t.strip()) for t in tools_str.split(",") if t.strip())


def _restore_tool(t: str, original_tools: set[str]) -> str:
    """将内部名还原为原始输入名。如果原始输入用的是 struct/unstruct，还原回去。"""
    if t in _TOOL_ALIAS_REV and _TOOL_ALIAS_REV[t] in original_tools:
        return _TOOL_ALIAS_REV[t]
    return t


def _restore_tools(tools_str: str, original_tools: set[str]) -> str:
    return ",".join(_restore_tool(t.strip(), original_tools) for t in tools_str.split(",") if t.strip())


async def _select_graph_or_es(ts_client: AsyncOpenAI, ts_model: str, query: str,
                              threshold: float) -> str:
    """调用 tool_selection 判断 query 适合 graph 还是 es。"""
    # 用 graph 做判定：in_scope=True → graph，否则 es
    result = await select_optional_tools(
        ts_client, ts_model, query,
        tool_hub="", tool_hub_optional="graph", threshold=threshold,
    )
    # select_optional_tools 返回合并后的 tool_hub 字符串
    # 如果 graph 被选中，result 包含 "graph"；否则为空
    if "graph" in result:
        return "graph"
    return "es"


async def _apply_tool_select(ts_client: AsyncOpenAI, ts_model: str, sub_queries: list[str],
                             tool_hub: str, tool_hub_optional: str,
                             threshold: float) -> list[str]:
    """对每个 sub_query，从 tool_hub_optional 中判断选 graph 还是 es，追加到 tool_hub。

    tool_hub 原样保留，仅当 tool_hub_optional 含 graph/es/struct/unstruct 时触发判定。
    """
    optional_normalized = {_normalize_tool(t.strip()) for t in tool_hub_optional.split(",") if t.strip()}
    need_select = bool(optional_normalized & {"graph", "es"})
    if not need_select:
        return [tool_hub] * len(sub_queries)

    # 并发判定每个 sub_query
    selections = await asyncio.gather(*(
        _select_graph_or_es(ts_client, ts_model, sq, threshold) for sq in sub_queries
    ))

    results = []
    for selected in selections:
        # tool_hub 保持不变，追加判定结果
        results.append(tool_hub + "," + selected if tool_hub else selected)

    logger.info("tool_select per-subquery: %s", list(zip(sub_queries, results)))
    return results


async def handle_simple_thinking(req, llm_client: AsyncOpenAI, llm_model: str,
                                 ts_client: AsyncOpenAI | None = None, ts_model: str = "",
                                 ts_threshold: float = 0) -> PlanningResponse:
    """Simple thinking 模式处理。"""

    if req.tool_select_enable:
        _optional_set = {t.strip() for t in req.tool_hub_optional.split(",") if t.strip()}
        if req.tool_hub:
            raise ValueError("tool_select_enable=True requires tool_hub to be empty")
        if _optional_set != {"struct", "unstruct"}:
            raise ValueError(f"tool_select_enable=True requires tool_hub_optional='struct,unstruct', got: {req.tool_hub_optional}")

    # 只对 tool_hub_optional 中的别名做还原（tool_hub 里的原样保留）
    _orig_optional = {t.strip() for t in req.tool_hub_optional.split(",") if t.strip()}
    # 内部统一用 graph/es
    req_tool_hub = _normalize_tools(req.tool_hub)
    req_tool_hub_optional = _normalize_tools(req.tool_hub_optional)

    def _restore_items(items: list[SubQueryItem]) -> list[SubQueryItem]:
        return [SubQueryItem(sub_query=it.sub_query,
                             tool_use=_restore_tools(it.tool_use, _orig_optional),
                             topk=it.topk) for it in items]

    if req.turn == 1:
        logger.info("Turn 1 | query=%r, max_top_k=%d, tool_hub=%s", req.query, req.max_top_k, req_tool_hub)

        sub_queries = await rewrite_query(llm_client, llm_model, req.query, req.max_top_k)

        if sub_queries is None:
            return PlanningResponse(is_off_topic=False, status="stop", turn=req.turn, final={})

        if not sub_queries:
            logger.info("Off-topic detected, no sub_queries")
            return PlanningResponse(is_off_topic=True, status="stop", turn=req.turn, current=[])

        if req.query not in sub_queries:
            sub_queries.insert(0, req.query)

        logger.info("Turn 1 | sub_queries=%s", sub_queries)

        # ── tool_select_enable: 用 model 判断 graph vs es ──
        if req.tool_select_enable and ts_client and req_tool_hub_optional:
            threshold = req.tool_selection_threshold if req.tool_selection_threshold is not None else ts_threshold
            tool_hubs = await _apply_tool_select(
                ts_client, ts_model, sub_queries, req_tool_hub, req_tool_hub_optional, threshold,
            )
            current_items = [SubQueryItem(sub_query=sq, tool_use=th, topk=req.max_top_k)
                             for sq, th in zip(sub_queries, tool_hubs)]

        # ── tool_selection_mode=model (旧逻辑，tool_select_enable=False) ──
        elif req.tool_selection_mode == "model" and req_tool_hub_optional:
            threshold = req.tool_selection_threshold if req.tool_selection_threshold is not None else ts_threshold
            tool_hubs = await asyncio.gather(*(
                select_optional_tools(
                    ts_client, ts_model, sq, req_tool_hub, req_tool_hub_optional, threshold=threshold,
                ) for sq in sub_queries
            ))
            logger.info("Turn 1 | tool_selection_mode=model, per-subquery tool_hubs=%s", tool_hubs)
            current_items = [SubQueryItem(sub_query=sq, tool_use=th, topk=req.max_top_k)
                             for sq, th in zip(sub_queries, tool_hubs)]
        else:
            current_items = [SubQueryItem(sub_query=sq, tool_use=req_tool_hub, topk=req.max_top_k)
                             for sq in sub_queries]

        return PlanningResponse(is_off_topic=False, status="running", turn=req.turn, current=_restore_items(current_items))

    else:
        # ── 非首轮 ──
        logger.info("Turn %d | tool_select_enable=%s, tool_hub=%s, tool_hub_optional=%s",
                     req.turn, req.tool_select_enable, req_tool_hub, req_tool_hub_optional)

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

        if req.tool_select_enable and ts_client and req_tool_hub_optional and req.turn < req.max_turn:
            # tool_select_enable 模式: 对未达标的 sub_query 重新判定 graph/es
            all_items = get_sub_queries_with_used_tools(req.history)
            sqs = [item["sub_query"] for item in all_items]
            if sqs:
                threshold = req.tool_selection_threshold if req.tool_selection_threshold is not None else ts_threshold
                tool_hubs = await _apply_tool_select(
                    ts_client, ts_model, sqs, req_tool_hub, req_tool_hub_optional, threshold,
                )
                for item, th in zip(all_items, tool_hubs):
                    new_tools = set(th.split(",")) - item["used_tools"]
                    if new_tools:
                        current_items.append(SubQueryItem(
                            sub_query=item["sub_query"],
                            tool_use=",".join(sorted(new_tools)),
                            topk=min(item["topk"], req.max_top_k),
                        ))

        elif req.tool_selection_mode == "model" and req_tool_hub_optional and req.turn < req.max_turn:
            all_items = get_sub_queries_with_used_tools(req.history)
            sqs = [item["sub_query"] for item in all_items]
            if sqs:
                threshold = req.tool_selection_threshold if req.tool_selection_threshold is not None else ts_threshold
                tool_hubs = await asyncio.gather(*(
                    select_optional_tools(
                        ts_client, ts_model, sq, req_tool_hub, req_tool_hub_optional, threshold=threshold,
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

        elif req.tool_selection_mode == "rule" and req_tool_hub_optional and req.turn < req.max_turn:
            optional_tools = {_normalize_tool(t.strip()) for t in req_tool_hub_optional.split(",") if t.strip()}
            all_items = get_sub_queries_with_used_tools(req.history)
            for item in all_items:
                remaining_tools = optional_tools - item["used_tools"]
                if remaining_tools:
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
            current=_restore_items(current_items), history=trim_history(req.history, req.max_context_size),
        )
