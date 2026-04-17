"""Deep thinking: 多步研究计划 + 逐步搜索总结 + 最终答案生成。"""

import hashlib
import json
import logging
import re
import time

from openai import AsyncOpenAI

from helpers import _parse_llm_json
from prompts import QUERY_REWRITE_PROMPT, DEEP_PLAN_PROMPT, DEEP_SUMMARY_PROMPT, DEEP_FINAL_PROMPT, DEEP_RESOLVE_PROMPT

logger = logging.getLogger("planning_server.deep_thinking")

# ── Session 管理 ────────────────────────────────────────────────

# session_id -> {"plan": [...], "summaries": [...], "step_idx": int, "query": str, "created_at": float}
_deep_sessions: dict[str, dict] = {}
_session_ttl: int = 600


def set_session_ttl(ttl: int):
    global _session_ttl
    _session_ttl = ttl


def _make_session_id(query: str) -> str:
    return hashlib.md5(query.encode("utf-8")).hexdigest()


def _cleanup_expired_sessions():
    now = time.time()
    expired = [sid for sid, s in _deep_sessions.items() if now - s.get("created_at", 0) > _session_ttl]
    for sid in expired:
        logger.info("Session expired, cleaning up: %s", sid)
        _deep_sessions.pop(sid, None)


# ── 内部 LLM 调用 ──────────────────────────────────────────────

async def _deep_create_plan(client: AsyncOpenAI, model: str, query: str, max_steps: int) -> list[dict]:
    messages = [
        {"role": "system", "content": DEEP_PLAN_PROMPT},
        {"role": "user", "content": f'query: "{query}"\nmax_steps: {max_steps}'},
    ]
    completion = await client.chat.completions.create(
        model=model, messages=messages, temperature=0.7, max_tokens=2048,
    )
    content = completion.choices[0].message.content or ""
    parsed = _parse_llm_json(content)
    if parsed and "steps" in parsed:
        return parsed["steps"]
    return []


async def _deep_rewrite_query(
    llm_client: AsyncOpenAI, llm_model: str,
    planner_client: AsyncOpenAI, planner_model: str,
    goal: str, context: str, key_facts: list[str], max_top_k: int,
) -> list[str]:
    """将 step goal 结合前面轮次的搜索结果具体化，再调用 query_rewriter 分解为 sub_queries。"""
    resolved_goal = goal

    if context or key_facts:
        facts_text = "\n".join(f"- {f}" for f in key_facts)
        resolve_prompt = DEEP_RESOLVE_PROMPT % (facts_text, context, goal)
        completion = await planner_client.chat.completions.create(
            model=planner_model,
            messages=[{"role": "user", "content": resolve_prompt}],
            temperature=0.1, max_tokens=512,
        )
        resolved = (completion.choices[0].message.content or "").strip()
        resolved = re.sub(r"<think>.*?</think>", "", resolved, flags=re.DOTALL).strip()
        resolved = resolved.strip('"').strip("'").strip("\u201c").strip("\u201d")
        if resolved:
            logger.info("Deep thinking: resolved goal %r -> %r", goal, resolved)
            resolved_goal = resolved

    messages = [
        {"role": "system", "content": QUERY_REWRITE_PROMPT},
        {"role": "user", "content": f'query: "{resolved_goal}"\ntopk: {max_top_k}'},
    ]
    completion = await llm_client.chat.completions.create(
        model=llm_model, messages=messages, temperature=0.7, max_tokens=2048,
    )
    content = completion.choices[0].message.content or ""
    parsed = _parse_llm_json(content)
    if parsed and "sub_queries" in parsed:
        return parsed["sub_queries"]
    return [goal]


async def _deep_summarize(client: AsyncOpenAI, model: str, goal: str, results_text: str, prev_context: str) -> dict:
    prompt = DEEP_SUMMARY_PROMPT % (prev_context or "(无)", goal, results_text[:4000])
    messages = [
        {"role": "system", "content": "You are a research assistant."},
        {"role": "user", "content": prompt},
    ]
    completion = await client.chat.completions.create(
        model=model, messages=messages, temperature=0.3, max_tokens=2048,
    )
    content = completion.choices[0].message.content or ""
    parsed = _parse_llm_json(content)
    if parsed:
        return parsed
    return {"summary": content[:500], "references": [], "key_facts": []}


async def _deep_final_answer(client: AsyncOpenAI, model: str, query: str, all_summaries: list[dict]) -> dict:
    summaries_text = json.dumps(all_summaries, ensure_ascii=False, indent=2)
    prompt = DEEP_FINAL_PROMPT % (query, summaries_text[:6000])
    messages = [
        {"role": "system", "content": "You are a research assistant."},
        {"role": "user", "content": prompt},
    ]
    completion = await client.chat.completions.create(
        model=model, messages=messages, temperature=0.3, max_tokens=4096,
    )
    content = completion.choices[0].message.content or ""
    parsed = _parse_llm_json(content)
    if parsed:
        return parsed
    return {"answer": content[:2000], "reference_tree": []}


def _results_to_text(history_turn: dict) -> str:
    lines = []
    for item in history_turn.get("retrieval_contents", []):
        sq = item.get("sub_query", "")
        for tool_type, records in item.get("result", {}).items():
            for r in records[:5]:
                title = r.get("title", "")
                content = r.get("content", "")[:300]
                score = r.get("score", "")
                lines.append(f"[{tool_type}] query={sq} | title={title} | score={score}\n{content}")
    return "\n---\n".join(lines) if lines else "(无结果)"


def _collect_referenced_records(summaries: list[dict]) -> dict:
    result: dict[str, list[dict]] = {}
    seen: set[str] = set()
    for s in summaries:
        step_key = f"step_{s.get('step', 0)}"
        refs = s.get("references", [])
        deduped = []
        for ref in refs:
            if isinstance(ref, str):
                ref = {"title": ref}
            if not isinstance(ref, dict):
                continue
            key = ref.get("content_snippet", ref.get("title", ""))
            if key and key not in seen:
                seen.add(key)
                deduped.append(ref)
        if deduped:
            result[step_key] = deduped
    return result


# ── 主入口 ──────────────────────────────────────────────────────

async def handle_deep_thinking(req, llm_client, llm_model, planner_client, planner_model, trim_history_fn):
    """Deep thinking 模式处理。返回 PlanningResponse dict。"""
    from helpers import PlanningResponse, SubQueryItem

    _cleanup_expired_sessions()
    session_id = _make_session_id(req.query)

    if req.turn == 1:
        logger.info("Deep thinking Turn 1 | query=%r, session_id=%s", req.query, session_id)
        steps = await _deep_create_plan(planner_client, planner_model, req.query, max_steps=req.max_turn)

        if not steps:
            logger.info("Deep thinking: off-topic or empty plan")
            return PlanningResponse(is_off_topic=True, status="stop", turn=req.turn, current=[])

        logger.info("Deep thinking plan: %d steps", len(steps))
        for i, s in enumerate(steps):
            logger.info("  Step %d: %s", i + 1, s.get("goal", ""))

        _deep_sessions[session_id] = {
            "plan": steps, "summaries": [], "step_idx": 0,
            "query": req.query, "created_at": time.time(),
        }

        step = steps[0]
        sub_queries = await _deep_rewrite_query(
            llm_client, llm_model, planner_client, planner_model,
            step["goal"], "", [], req.max_top_k,
        )
        logger.info("Deep thinking Step 1 | sub_queries=%s", sub_queries)

        current_items = [SubQueryItem(sub_query=sq, tool_use=req.tool_hub, topk=req.max_top_k) for sq in sub_queries]
        return PlanningResponse(is_off_topic=False, status="running", turn=req.turn, current=current_items)

    else:
        if session_id not in _deep_sessions:
            logger.warning("Deep thinking: session not found: %s", session_id)
            return PlanningResponse(status="stop", turn=req.turn, final={})

        session = _deep_sessions[session_id]
        plan = session["plan"]
        step_idx = session["step_idx"]
        query = session["query"]

        prev_context = "\n".join(
            f"Step {i + 1}: {s.get('summary', '')}" for i, s in enumerate(session["summaries"])
        )
        latest_turn_key = f"turn_{req.turn - 1}"
        results_text = ""
        if req.history and latest_turn_key in req.history:
            results_text = _results_to_text(req.history[latest_turn_key])

        current_step = plan[step_idx]
        logger.info("Deep thinking Turn %d | summarizing step %d: %s",
                     req.turn, step_idx + 1, current_step.get("goal", ""))

        summary_result = await _deep_summarize(
            planner_client, planner_model, current_step["goal"], results_text, prev_context,
        )
        session["summaries"].append({
            "step": step_idx + 1,
            "goal": current_step.get("goal", ""),
            "reason": current_step.get("reason", ""),
            **summary_result,
        })
        session["step_idx"] = step_idx + 1

        logger.info("Deep thinking Step %d summary: %s",
                     step_idx + 1, summary_result.get("summary", "")[:200])

        next_idx = step_idx + 1
        if next_idx >= len(plan) or req.turn >= req.max_turn:
            logger.info("Deep thinking: generating final answer (%d summaries)", len(session["summaries"]))
            final_result = await _deep_final_answer(planner_client, planner_model, query, session["summaries"])
            final = _collect_referenced_records(session["summaries"])
            _deep_sessions.pop(session_id, None)

            return PlanningResponse(
                is_off_topic=False, status="stop", turn=req.turn, final=final,
                step_summary=summary_result.get("summary", ""),
                answer=final_result.get("answer", ""),
                reference_tree=final_result.get("reference_tree", []),
                history=trim_history_fn(req.history, req.max_context_size),
            )

        next_step = plan[next_idx]
        accumulated_context = "\n".join(
            f"Step {s['step']}: {s.get('summary', '')}" for s in session["summaries"]
        )
        all_key_facts = []
        for s in session["summaries"]:
            all_key_facts.extend(s.get("key_facts", []))

        sub_queries = await _deep_rewrite_query(
            llm_client, llm_model, planner_client, planner_model,
            next_step["goal"], accumulated_context, all_key_facts, req.max_top_k,
        )
        logger.info("Deep thinking Step %d | sub_queries=%s", next_idx + 1, sub_queries)

        current_items = [SubQueryItem(sub_query=sq, tool_use=req.tool_hub, topk=req.max_top_k) for sq in sub_queries]
        return PlanningResponse(
            is_off_topic=False, status="running", turn=req.turn,
            current=current_items,
            step_summary=summary_result.get("summary", ""),
            history=trim_history_fn(req.history, req.max_context_size),
        )
