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
import hashlib
import json
import logging
import re
import time

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
_planner_client: AsyncOpenAI | None = None
_planner_model: str = ""


# ── Deep Thinking: session 存储和 prompts ───────────────────────

# session_id -> {"plan": [...], "summaries": [...], "step_idx": int, "query": str, "created_at": float}
_deep_sessions: dict[str, dict] = {}
_session_ttl: int = 600  # 默认 10 分钟，可通过 --session-ttl 配置


def _make_session_id(query: str) -> str:
    """根据 query 生成确定性的 session_id。"""
    return hashlib.md5(query.encode("utf-8")).hexdigest()


def _cleanup_expired_sessions():
    """清理超时的 session。"""
    now = time.time()
    expired = [sid for sid, s in _deep_sessions.items() if now - s.get("created_at", 0) > _session_ttl]
    for sid in expired:
        logger.info("Session expired, cleaning up: %s", sid)
        _deep_sessions.pop(sid, None)

DEEP_PLAN_PROMPT = """\
You are a research planning assistant. Given a user query and a maximum number of steps, \
create a step-by-step research plan. Each step should specify what to search for and why.

Rules:
1. Each step has a clear search goal.
2. Later steps can build on information gathered in earlier steps.
3. Total steps must NOT exceed max_steps.
4. If the query is simple, fewer steps are fine.
5. Each step should have a "goal" (what to find) and "reason" (why this is needed).

You MUST respond with a JSON object:
{"steps": [{"goal": "...", "reason": "..."}, ...]}

If the query is casual chat:
{"steps": []}
"""

DEEP_SUMMARY_PROMPT = """\
You are a research assistant. Given a search query, search results, and previous research context, \
provide a concise summary of the findings and list the key references.

Previous context:
%s

Current step goal: %s

Search results:
%s

Respond with a JSON object:
{"summary": "concise summary of findings", "references": [{"title": "...", "content_snippet": "...", "source": "...", "query": "the sub_query that produced this result"}], "key_facts": ["fact1", "fact2"]}
"""

DEEP_FINAL_PROMPT = """\
You are a research assistant. Given all the research summaries collected across multiple steps, \
synthesize a final comprehensive answer to the original question.

Original question: %s

Research summaries:
%s

Respond with a JSON object:
{"answer": "comprehensive final answer", "reference_tree": [{"step": 1, "goal": "...", "summary": "...", "references": [...]}]}
"""


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


class PlanningResponse(BaseModel):
    is_off_topic: bool = False
    status: str = "running"
    turn: int = 1
    current: list[SubQueryItem] | None = None
    history: dict | None = None
    final: dict | None = None
    # deep_thinking 模式额外字段
    step_summary: str | None = None
    answer: str | None = None
    reference_tree: list | None = None


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


def _collect_all_results_from_history(history: dict | None, original_query: str = "", top_k: int = 3) -> dict:
    """从 history 中组装 final 结果，按工具类型分组。

    组装规则:
    1. 按 sub_query 分组，每个 sub_query 内按 score 降序排列
    2. 原始 query 对应的那路放第一位
    3. 每路至少入围 1 条结果
    4. 轮询取结果直到凑够 top_k，跳过已选过的重复记录
    """
    if not history:
        return {}

    # 按 (sub_query, tool_type) 收集结果，每路内部按 score 降序
    # streams[tool_type] = [(sub_query, [record, ...]), ...]
    streams_by_tool: dict[str, dict[str, list[dict]]] = {}

    for turn_key in sorted(history.keys()):
        turn_data = history[turn_key]
        contents = turn_data.get("retrieval_contents", [])
        for item in contents:
            sq = item.get("sub_query", "")
            result = item.get("result", {})
            for tool_type, records in result.items():
                if tool_type not in streams_by_tool:
                    streams_by_tool[tool_type] = {}
                if sq not in streams_by_tool[tool_type]:
                    streams_by_tool[tool_type][sq] = []
                streams_by_tool[tool_type][sq].extend(records)

    # 每路内部按 score 降序排列
    for tool_type in streams_by_tool:
        for sq in streams_by_tool[tool_type]:
            streams_by_tool[tool_type][sq].sort(
                key=lambda r: r.get("score", 0) if isinstance(r.get("score", 0), (int, float)) else 0,
                reverse=True,
            )

    final: dict[str, list[dict]] = {}

    for tool_type, sq_map in streams_by_tool.items():
        # 排序 sub_query 列表：原始 query 放第一位，其余保持插入顺序
        sq_keys = list(sq_map.keys())
        if original_query in sq_keys:
            sq_keys.remove(original_query)
            sq_keys.insert(0, original_query)

        # 构建每路的候选流（带指针）
        streams: list[tuple[str, list[dict]]] = [(sq, sq_map[sq]) for sq in sq_keys]

        selected: list[dict] = []
        seen: set[tuple] = set()
        pointers = [0] * len(streams)  # 每路当前取到的位置

        def _record_key(record: dict) -> str:
            return record.get("content", "")

        def _pick_next(stream_idx: int) -> dict | None:
            """从指定路中取下一个不重复的记录。"""
            sq, records = streams[stream_idx]
            while pointers[stream_idx] < len(records):
                r = records[pointers[stream_idx]]
                pointers[stream_idx] += 1
                if _record_key(r) not in seen:
                    return r
            return None

        # 第一轮：每路至少取 1 条，保证每路入围
        for i in range(len(streams)):
            r = _pick_next(i)
            if r is not None:
                seen.add(_record_key(r))
                selected.append(r)

        # 后续轮：轮询取，直到凑够 top_k
        while len(selected) < top_k:
            added = False
            for i in range(len(streams)):
                if len(selected) >= top_k:
                    break
                r = _pick_next(i)
                if r is not None:
                    seen.add(_record_key(r))
                    selected.append(r)
                    added = True
            if not added:
                break  # 所有路都取完了

        final[tool_type] = selected

    return final


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


def _count_total_qualified(history: dict | None, score_threshold: float) -> int:
    """统计 history 中所有 tool 类型、所有 sub_query 达到 score_threshold 的结果总数。"""
    count = 0
    if not history:
        return count
    for turn_key in sorted(history.keys()):
        turn_data = history[turn_key]
        for item in turn_data.get("retrieval_contents", []):
            for tool_type, records in item.get("result", {}).items():
                for r in records:
                    score = r.get("score", 0)
                    if isinstance(score, (int, float)) and score >= score_threshold:
                        count += 1
    return count


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

async def _deep_create_plan(query: str, max_steps: int) -> list[dict]:
    """调用 planner 模型生成研究计划。"""
    messages = [
        {"role": "system", "content": DEEP_PLAN_PROMPT},
        {"role": "user", "content": f'query: "{query}"\nmax_steps: {max_steps}'},
    ]
    completion = await _planner_client.chat.completions.create(
        model=_planner_model,
        messages=messages,
        temperature=0.7,
        max_tokens=2048,
    )
    content = completion.choices[0].message.content or ""
    logger.log(VERBOSE, "Deep plan raw response:\n%s", content)
    parsed = _parse_llm_json(content)
    if parsed and "steps" in parsed:
        return parsed["steps"]
    return []


async def _deep_rewrite_query(goal: str, context: str, key_facts: list[str], max_top_k: int) -> list[str]:
    """将 step goal 结合前面轮次的搜索结果具体化，再调用 query_rewriter 分解为 sub_queries。

    关键：KBP 没有记忆，所以 goal 中任何依赖前面轮次的引用（如"在价格和匹数范围内"、
    "初始结果中"、"上述型号"等）都必须替换为具体的数值、型号名、参数等，
    生成的 query 必须是完全自包含的，可以直接丢给搜索引擎。
    """
    resolved_goal = goal

    # 如果有前面轮次的 context 或 key_facts，先让 planner 模型把 goal 具体化
    if context or key_facts:
        facts_text = "\n".join(f"- {f}" for f in key_facts)
        resolve_prompt = (
            "你的任务：将搜索目标改写为一个完全自包含的搜索 query。\n\n"
            "规则（必须严格遵守）：\n"
            "1. 搜索引擎没有任何记忆，不知道之前搜过什么，所以输出的 query 必须包含所有必要信息。\n"
            "2. 禁止出现任何模糊引用，包括但不限于：'初始结果'、'上述'、'前面提到的'、"
            "'符合条件的'、'在...范围内'（不带具体数值）、'相关型号'等。\n"
            "3. 所有引用必须替换为具体的数值、型号名、品牌名、参数值。\n"
            "4. 如果已知事实中没有对应的具体值，则删除该限定条件，不要用模糊表述代替。\n"
            "5. 只输出改写后的 query 文本，不要解释、不要加引号。\n\n"
            f"已知事实（来自前面的搜索结果）:\n{facts_text}\n\n"
            f"前面的研究摘要:\n{context}\n\n"
            f"原始搜索目标: {goal}\n\n"
            "改写后的搜索 query:"
        )
        completion = await _planner_client.chat.completions.create(
            model=_planner_model,
            messages=[{"role": "user", "content": resolve_prompt}],
            temperature=0.1,
            max_tokens=512,
        )
        resolved = (completion.choices[0].message.content or "").strip()
        resolved = re.sub(r"<think>.*?</think>", "", resolved, flags=re.DOTALL).strip()
        # 去掉可能的引号包裹
        resolved = resolved.strip('"').strip("'").strip(""").strip(""")
        if resolved:
            logger.info("Deep thinking: resolved goal %r -> %r", goal, resolved)
            resolved_goal = resolved

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f'query: "{resolved_goal}"\ntopk: {max_top_k}'},
    ]
    completion = await _llm_client.chat.completions.create(
        model=_llm_model,
        messages=messages,
        temperature=0.7,
        max_tokens=2048,
    )
    content = completion.choices[0].message.content or ""
    parsed = _parse_llm_json(content)
    if parsed and "sub_queries" in parsed:
        return parsed["sub_queries"]
    return [goal]


async def _deep_summarize(goal: str, results_text: str, prev_context: str) -> dict:
    """调用 planner 模型对搜索结果做总结。"""
    prompt = DEEP_SUMMARY_PROMPT % (prev_context or "(无)", goal, results_text[:4000])
    messages = [
        {"role": "system", "content": "You are a research assistant."},
        {"role": "user", "content": prompt},
    ]
    completion = await _planner_client.chat.completions.create(
        model=_planner_model,
        messages=messages,
        temperature=0.3,
        max_tokens=2048,
    )
    content = completion.choices[0].message.content or ""
    logger.log(VERBOSE, "Deep summary raw response:\n%s", content)
    parsed = _parse_llm_json(content)
    if parsed:
        return parsed
    return {"summary": content[:500], "references": [], "key_facts": []}


async def _deep_final_answer(query: str, all_summaries: list[dict]) -> dict:
    """调用 planner 模型生成最终答案和引用树。"""
    summaries_text = json.dumps(all_summaries, ensure_ascii=False, indent=2)
    prompt = DEEP_FINAL_PROMPT % (query, summaries_text[:6000])
    messages = [
        {"role": "system", "content": "You are a research assistant."},
        {"role": "user", "content": prompt},
    ]
    completion = await _planner_client.chat.completions.create(
        model=_planner_model,
        messages=messages,
        temperature=0.3,
        max_tokens=4096,
    )
    content = completion.choices[0].message.content or ""
    logger.log(VERBOSE, "Deep final answer raw response:\n%s", content)
    parsed = _parse_llm_json(content)
    if parsed:
        return parsed
    return {"answer": content[:2000], "reference_tree": []}


def _results_to_text(history_turn: dict) -> str:
    """将一轮检索结果转为文本摘要供 LLM 使用。"""
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
    """从 deep_thinking 各步骤的 summaries 中提取被引用的 records，按步骤组织。

    每个 summary 中的 references 列表即为该步骤引用的记录，
    返回 {"step_1": [...], "step_2": [...]} 格式，去重。
    """
    result: dict[str, list[dict]] = {}
    seen: set[str] = set()
    for s in summaries:
        step_key = f"step_{s.get('step', 0)}"
        refs = s.get("references", [])
        deduped = []
        for ref in refs:
            key = ref.get("content_snippet", ref.get("title", ""))
            if key and key not in seen:
                seen.add(key)
                deduped.append(ref)
        if deduped:
            result[step_key] = deduped
    return result

async def _handle_deep_thinking(req: PlanningRequest) -> PlanningResponse:
    """Deep thinking 模式处理。

    Turn 1: 调用 planner 模型生成研究计划，然后对第一步做 query rewrite，返回搜索任务
    Turn 2+: 接收上一步搜索结果，调用 planner 模型做总结，然后对下一步做 query rewrite
    最后一步或所有步骤完成: 生成最终 answer + reference_tree
    """
    # 清理过期 session
    _cleanup_expired_sessions()

    session_id = _make_session_id(req.query)

    if req.turn == 1:
        # ── 首轮: 生成 plan，执行第一步 ──
        logger.info("Deep thinking Turn 1 | query=%r, session_id=%s", req.query, session_id)
        steps = await _deep_create_plan(req.query, max_steps=req.max_turn)

        if not steps:
            logger.info("Deep thinking: off-topic or empty plan")
            return PlanningResponse(is_off_topic=True, status="stop", turn=req.turn, current=[])

        logger.info("Deep thinking plan: %d steps", len(steps))
        for i, s in enumerate(steps):
            logger.info("  Step %d: %s", i + 1, s.get("goal", ""))

        # 创建 session（如果已存在则覆盖）
        _deep_sessions[session_id] = {
            "plan": steps,
            "summaries": [],
            "step_idx": 0,
            "query": req.query,
            "created_at": time.time(),
        }

        # 对第一步做 query rewrite
        step = steps[0]
        sub_queries = await _deep_rewrite_query(step["goal"], "", [], req.max_top_k)
        # deep_thinking 模式下每步有自己的 goal，不插入最初的 query

        logger.info("Deep thinking Step 1 | sub_queries=%s", sub_queries)

        current_items = [SubQueryItem(sub_query=sq, tool_use=req.tool_hub, topk=req.max_top_k) for sq in sub_queries]

        return PlanningResponse(
            is_off_topic=False,
            status="running",
            turn=req.turn,
            current=current_items,
        )

    else:
        # ── 后续轮: 总结上一步结果，执行下一步或结束 ──
        if session_id not in _deep_sessions:
            logger.warning("Deep thinking: session not found: %s", session_id)
            return PlanningResponse(status="stop", turn=req.turn, final={})

        session = _deep_sessions[session_id]
        plan = session["plan"]
        step_idx = session["step_idx"]
        query = session["query"]

        # 总结上一步的搜索结果
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

        summary_result = await _deep_summarize(current_step["goal"], results_text, prev_context)
        session["summaries"].append({
            "step": step_idx + 1,
            "goal": current_step.get("goal", ""),
            "reason": current_step.get("reason", ""),
            **summary_result,
        })
        session["step_idx"] = step_idx + 1

        logger.info("Deep thinking Step %d summary: %s",
                     step_idx + 1, summary_result.get("summary", "")[:200])

        # 检查是否还有下一步
        next_idx = step_idx + 1
        if next_idx >= len(plan) or req.turn >= req.max_turn:
            # ── 所有步骤完成或达到 max_turn: 生成最终答案 ──
            logger.info("Deep thinking: generating final answer (%d summaries)", len(session["summaries"]))
            final_result = await _deep_final_answer(query, session["summaries"])

            # final 只放各步骤 summaries 中引用到的 records，不放全量
            final = _collect_referenced_records(session["summaries"])

            # 清理 session
            _deep_sessions.pop(session_id, None)

            return PlanningResponse(
                is_off_topic=False,
                status="stop",
                turn=req.turn,
                final=final,
                step_summary=summary_result.get("summary", ""),
                answer=final_result.get("answer", ""),
                reference_tree=final_result.get("reference_tree", []),
                history=_trim_history(req.history, req.max_context_size),
            )

        # ── 还有下一步: 用之前的总结作为 context，对下一步做 query rewrite ──
        next_step = plan[next_idx]
        accumulated_context = "\n".join(
            f"Step {s['step']}: {s.get('summary', '')}" for s in session["summaries"]
        )
        # 收集所有步骤的 key_facts，供 goal 具体化使用
        all_key_facts = []
        for s in session["summaries"]:
            all_key_facts.extend(s.get("key_facts", []))

        sub_queries = await _deep_rewrite_query(next_step["goal"], accumulated_context, all_key_facts, req.max_top_k)
        logger.info("Deep thinking Step %d | sub_queries=%s", next_idx + 1, sub_queries)

        current_items = [SubQueryItem(sub_query=sq, tool_use=req.tool_hub, topk=req.max_top_k) for sq in sub_queries]

        return PlanningResponse(
            is_off_topic=False,
            status="running",
            turn=req.turn,
            current=current_items,
            step_summary=summary_result.get("summary", ""),
            history=_trim_history(req.history, req.max_context_size),
        )


@app.post("/planner", response_model=PlanningResponse)
async def planner(req: PlanningRequest):
    """单一 planning 端点。

    Round 1 (turn==1): 调用 LLM 分解 query → sub_queries + 原始 query，用 tool_hub 所有工具搜索
    Round 2+ (turn>1): 纯规则 - 检查每个 sub_query 达标结果数是否满足 top_k，
                       不足的用尚未使用的工具（含 tool_hub_optional）补充
    停止条件: 达到 max_turn / 所有 sub_query 达标 / 所有工具已用完
    """
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

    # ── Deep Thinking 模式 ──────────────────────────────────────
    if req.deep_thinking:
        return await _handle_deep_thinking(req)

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
        # ── 非首轮: 纯规则，按阈值和 top_k 判断是否需要补搜 ──
        logger.info("Turn %d | rule-based, tool_hub=%s, tool_hub_optional=%s",
                     req.turn, req.tool_hub, req.tool_hub_optional)

        score_threshold = req.retrieval_setting.score_threshold
        top_k = req.retrieval_setting.top_k

        # 统计所有 tool、所有 sub_query 达标结果的总数
        total_qualified = _count_total_qualified(req.history, score_threshold)
        logger.debug("Total qualified (threshold=%.4f): %d, top_k=%d", score_threshold, total_qualified, top_k)

        # 总数凑满 top_k，直接结束
        if total_qualified >= top_k:
            final = _collect_all_results_from_history(req.history, original_query=req.query, top_k=top_k)
            logger.info("Turn %d | total qualified=%d >= top_k=%d, stop. final has %d tool types",
                        req.turn, total_qualified, top_k, len(final))
            return _respond(PlanningResponse(
                is_off_topic=False,
                status="stop",
                turn=req.turn,
                final=final,
                history=_trim_history(req.history, req.max_context_size),
            ))

        # 凑不满，用 tool_hub_optional 中尚未使用的工具对所有 sub_query 补搜
        optional_tools: set[str] = set()
        if req.tool_hub_optional:
            for t in req.tool_hub_optional.split(","):
                t = t.strip()
                if t:
                    optional_tools.add(t)

        all_items = _get_sub_queries_with_used_tools(req.history)
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

        # 没有可补搜的任务，直接结束
        if not current_items:
            final = _collect_all_results_from_history(req.history, original_query=req.query, top_k=top_k)
            logger.info("Turn %d | no optional tools to try, stop. final has %d tool types", req.turn, len(final))
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
    parser.add_argument("--planner-base-url", default="https://aix-backup.hismarttv.com/v1",
                        help="Planner LLM base URL for deep_thinking")
    parser.add_argument("--planner-model", default="deepseek-v3",
                        help="Planner model name for deep_thinking")
    parser.add_argument("--planner-api-key", default="x31ctKZ0ONfi1jkO",
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

    global _llm_client, _llm_model, _planner_client, _planner_model, _session_ttl
    _llm_client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")
    _llm_model = args.model
    _planner_model = args.planner_model
    _planner_client = AsyncOpenAI(base_url=args.planner_base_url, api_key=args.planner_api_key)
    _session_ttl = args.session_ttl

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
