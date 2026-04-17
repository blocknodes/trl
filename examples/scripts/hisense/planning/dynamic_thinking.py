"""Dynamic thinking: React 式循环 — 搜索 → 总结+判断 → 决定是否继续。

与 deep_thinking 的区别：不预先生成 plan，每一步搜完后由 LLM 决定是否需要下一步。
"""

import hashlib
import json
import logging
import re
import time

from openai import AsyncOpenAI

from helpers import _parse_llm_json
from prompts import QUERY_REWRITE_PROMPT, DEEP_FINAL_PROMPT, DEEP_RESOLVE_PROMPT, DYNAMIC_REACT_PROMPT

logger = logging.getLogger("planning_server.dynamic_thinking")

# 复用 deep_thinking 的 session 存储和工具函数
from deep_thinking import (
    _deep_sessions,
    _session_ttl,
    _cleanup_expired_sessions,
    _make_session_id,
    _results_to_text,
    _collect_referenced_records,
    _deep_rewrite_query,
    _deep_final_answer,
)


async def _react_step(client: AsyncOpenAI, model: str, query: str, results_text: str, prev_context: str) -> dict:
    """总结搜索结果，同时判断是否需要继续搜索。"""
    prompt = DYNAMIC_REACT_PROMPT % (query, prev_context or "(无)", results_text[:4000])
    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a research assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3, max_tokens=2048,
    )
    content = completion.choices[0].message.content or ""
    logger.debug("React step raw response length: %d", len(content))
    parsed = _parse_llm_json(content)
    if parsed:
        return parsed
    return {"status": "stop", "summary": content[:500], "key_facts": [], "references": []}


async def handle_dynamic_thinking(req, llm_client, llm_model, planner_client, planner_model, trim_history_fn):
    """Dynamic thinking (react) 模式处理。"""
    from helpers import PlanningResponse, SubQueryItem

    _cleanup_expired_sessions()
    session_id = _make_session_id(req.query)

    if req.turn == 1:
        # ── 首轮: 直接对原始 query 做 rewrite，发起第一次搜索 ──
        logger.info("Dynamic thinking Turn 1 | query=%r", req.query)

        _deep_sessions[session_id] = {
            "plan": [],  # dynamic 模式不预生成 plan
            "summaries": [],
            "step_idx": 0,
            "query": req.query,
            "created_at": time.time(),
        }

        sub_queries = await _deep_rewrite_query(
            llm_client, llm_model, planner_client, planner_model,
            req.query, "", [], req.max_top_k,
        )
        logger.info("Dynamic thinking Step 1 | sub_queries=%s", sub_queries)

        if not sub_queries:
            _deep_sessions.pop(session_id, None)
            return PlanningResponse(is_off_topic=True, status="stop", turn=req.turn, current=[])

        current_items = [SubQueryItem(sub_query=sq, tool_use=req.tool_hub, topk=req.max_top_k) for sq in sub_queries]
        return PlanningResponse(is_off_topic=False, status="running", turn=req.turn, current=current_items)

    else:
        # ── 后续轮: 总结上一步结果 + 判断是否继续 ──
        if session_id not in _deep_sessions:
            logger.warning("Dynamic thinking: session not found: %s", session_id)
            return PlanningResponse(status="stop", turn=req.turn, final={})

        session = _deep_sessions[session_id]
        query = session["query"]
        step_idx = session["step_idx"]

        prev_context = "\n".join(
            f"Step {i + 1}: {s.get('summary', '')}" for i, s in enumerate(session["summaries"])
        )
        latest_turn_key = f"turn_{req.turn - 1}"
        results_text = ""
        if req.history and latest_turn_key in req.history:
            results_text = _results_to_text(req.history[latest_turn_key])

        # React: 总结 + 判断
        react_result = await _react_step(planner_client, planner_model, query, results_text, prev_context)

        session["summaries"].append({
            "step": step_idx + 1,
            "goal": react_result.get("next_goal", query if step_idx == 0 else ""),
            **react_result,
        })
        session["step_idx"] = step_idx + 1

        logger.info("Dynamic thinking Step %d | status=%s, summary=%s",
                     step_idx + 1, react_result.get("status"), react_result.get("summary", "")[:200])

        should_stop = react_result.get("status") != "continue" or req.turn >= req.max_turn
        next_goal = react_result.get("next_goal", "")

        if should_stop or not next_goal:
            # ── 结束: 生成最终答案 ──
            logger.info("Dynamic thinking: generating final answer (%d summaries)", len(session["summaries"]))
            final_result = await _deep_final_answer(planner_client, planner_model, query, session["summaries"])
            final = _collect_referenced_records(session["summaries"])
            _deep_sessions.pop(session_id, None)

            return PlanningResponse(
                is_off_topic=False, status="stop", turn=req.turn, final=final,
                step_summary=react_result.get("summary", ""),
                answer=final_result.get("answer", ""),
                reference_tree=final_result.get("reference_tree", []),
                history=trim_history_fn(req.history, req.max_context_size),
            )

        # ── 继续: 对 next_goal 做 rewrite ──
        all_key_facts = []
        for s in session["summaries"]:
            all_key_facts.extend(s.get("key_facts", []))

        accumulated_context = "\n".join(
            f"Step {s.get('step', i+1)}: {s.get('summary', '')}" for i, s in enumerate(session["summaries"])
        )

        sub_queries = await _deep_rewrite_query(
            llm_client, llm_model, planner_client, planner_model,
            next_goal, accumulated_context, all_key_facts, req.max_top_k,
        )
        logger.info("Dynamic thinking Step %d | next_goal=%r, sub_queries=%s", step_idx + 2, next_goal, sub_queries)

        current_items = [SubQueryItem(sub_query=sq, tool_use=req.tool_hub, topk=req.max_top_k) for sq in sub_queries]
        return PlanningResponse(
            is_off_topic=False, status="running", turn=req.turn,
            current=current_items,
            step_summary=react_result.get("summary", ""),
            history=trim_history_fn(req.history, req.max_context_size),
        )
