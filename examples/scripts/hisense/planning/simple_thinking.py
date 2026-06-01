"""Simple thinking: 两轮 planning 协议。

- Round 1: 调用 LLM 将 query 分解为 sub_queries，分配工具
- Round 2: 规则判断分数，补充工具或 stop

tool_select_enable=True 时:
  若 tool_hub_optional 含 graph/es/struct/unstruct，调用 tool_selection model
  判断 query 是否适合结构化搜索 → 适合用 graph，否则用 es
"""

import asyncio
import hashlib
import json
import logging
import os
import time

from openai import AsyncOpenAI

from helpers import DomainItem, PlanningResponse, RetrievalSetting, SubQueryItem, _parse_llm_json
from helpers import extract_entity_tables, build_keywords_from_entity_tables, score_query_against_keywords
from query_rewrite import rewrite_query
from prompts import DOMAIN_SELECT_PROMPT, RDF_SUMMARIZE_PROMPT
from tool_selection import select_optional_tools
from workflow import (
    collect_all_results_from_history,
    count_total_qualified,
    get_sub_queries_with_used_tools,
    trim_history,
)

logger = logging.getLogger("planning_server.simple_thinking")

# ── Session 管理 ────────────────────────────────────────────────
# session_id -> {"history": dict, "created_at": float}
_simple_sessions: dict[str, dict] = {}
_simple_session_ttl: int = 600


def set_simple_session_ttl(ttl: int):
    global _simple_session_ttl
    _simple_session_ttl = ttl


def _make_session_id(query: str) -> str:
    return hashlib.md5(query.encode("utf-8")).hexdigest()


def _cleanup_expired_sessions():
    now = time.time()
    expired = [sid for sid, s in _simple_sessions.items()
               if now - s.get("created_at", 0) > _simple_session_ttl]
    for sid in expired:
        logger.info("Simple session expired, cleaning up: %s", sid)
        _simple_sessions.pop(sid, None)


# 工具别名映射（保留 unstruct -> es 的兼容别名；struct 和 graph 是两种独立工具）
_TOOL_ALIAS = {"unstruct": "es"}
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


async def _summarize_rdf_domain(client: AsyncOpenAI, model: str,
                               domain: str, rdf: list) -> str:
    """调用 LLM 将 domain 的子 domain 定义压缩为一句话摘要。"""
    # 拼接所有子 domain 的 desc 和 info
    parts = []
    for sub in rdf:
        if sub.desc:
            parts.append(f"{sub.scene}: {sub.desc}")
        elif sub.info:
            info_str = json.dumps(sub.info, ensure_ascii=False)[:500]
            parts.append(f"{sub.scene}: {info_str}")
        else:
            parts.append(sub.scene)
    define = "; ".join(parts)
    prompt = RDF_SUMMARIZE_PROMPT % (domain, define[:4000])
    completion = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3, max_tokens=256,
    )
    summary = (completion.choices[0].message.content or "").strip()
    logger.debug("RDF summarize: domain=%r -> %s", domain, summary[:200])
    return summary


async def _select_domains(client: AsyncOpenAI, model: str, query: str,
                         domains: list[DomainItem]) -> list[str]:
    """从 domains 中选出与 query 相关的领域。

    1. 过滤出有 description 或 rdf 的 domain 作为候选集
    2. 候选集为空 → 返回所有 domain
    3. 有 description 的直接用，没有的调 LLM 压缩 rdf
    4. 调 LLM 从候选集中选择，选不出来 → 返回整个候选集
    """
    all_names = [d.domain for d in domains]
    candidates = [d for d in domains if d.desc.strip() or d.rdf_list]
    if not candidates:
        logger.info("No domain has desc or rdf, returning all domains")
        return all_names

    candidate_names = [d.domain for d in candidates]

    # 获取每个候选 domain 的摘要：有 desc 直接用，否则调 LLM 压缩 rdf
    async def _get_summary(d: DomainItem) -> str:
        if d.desc.strip():
            return d.desc.strip()
        return await _summarize_rdf_domain(client, model, d.domain, d.rdf_list)

    summaries = await asyncio.gather(*(_get_summary(d) for d in candidates))
    domain_desc = "\n".join(f"- {d.domain}: {s}" for d, s in zip(candidates, summaries))

    # 用摘要做 domain 选择
    prompt = DOMAIN_SELECT_PROMPT % (domain_desc, query)
    completion = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3, max_tokens=512,
    )
    content = completion.choices[0].message.content or ""
    parsed = _parse_llm_json(content)
    if parsed and "domains" in parsed:
        valid = [d for d in parsed["domains"] if d in candidate_names]
        if valid:
            logger.debug("Domain selection: query=%r, selected=%s", query, valid)
            return valid
        logger.warning("Domain selection returned empty, falling back to all candidates")
    else:
        logger.warning("Domain selection parse failed, falling back to all candidates")
    return candidate_names


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


def _select_tool_by_keyword(query: str, tool_hub_optional: str, domains: list | None,
                            keyword_threshold: float) -> tuple[str, list[str]]:
    """基于关键词匹配从 tool_hub_optional 中选择工具。

    工具类型:
    - struct: 结构化检索（带 domain，由关键词匹配触发）
    - graph: 查询工具（不带 domain，类似 web）
    - es: 必选工具（当环境变量 ES_REQUIRED=1 时）
    - web: fallback 工具

    规则:
    - 若环境变量 ES_REQUIRED=1 且 tool_hub_optional 含 es，es 必选
    - 若 tool_hub_optional 含 struct 且关键词命中 → 加上 struct + matched domains
    - 若 es 和 struct 都不可用/未命中，按优先级 fallback 到 es > graph > web
    - 若 struct 是唯一选项且未命中，仍使用 struct（带所有 domains）

    返回: (选中的工具字符串, 匹配的 domain 列表; 仅 struct 时有效)
    """
    es_required = os.environ.get("ES_REQUIRED", "0") == "1"

    optional_normalized = [_normalize_tool(t.strip()) for t in tool_hub_optional.split(",") if t.strip()]

    # 关键词匹配：判断 struct 是否合适
    matched_domains: list[str] = []
    if "struct" in optional_normalized and domains:
        for domain_item in domains:
            for sub_domain in domain_item.rdf_list:
                if sub_domain.info:
                    entity_tables = extract_entity_tables(sub_domain.info)
                    if entity_tables:
                        keywords = build_keywords_from_entity_tables(entity_tables)
                        hit_ratio, hit_count, token_count, hits = score_query_against_keywords(query, keywords)
                        logger.info("Keyword tool select: domain=%s, scene=%s, query=%r, "
                                    "hit_ratio=%.2f, hit_count=%d, token_count=%d, threshold=%.2f, hits=%s",
                                    domain_item.domain, sub_domain.scene, query,
                                    hit_ratio, hit_count, token_count, keyword_threshold, hits)
                        if hit_ratio >= keyword_threshold:
                            if domain_item.domain not in matched_domains:
                                matched_domains.append(domain_item.domain)

    # 组装首轮工具：struct（命中时） + es（ES_REQUIRED=1 且 es 在 optional 中时必选）
    selected: list[str] = []
    if matched_domains:
        selected.append("struct")
    if es_required and "es" in optional_normalized:
        selected.append("es")

    if selected:
        tool_use = ",".join(selected)
        logger.info("Keyword tool select: selected=%s, matched_domains=%s, es_required=%s",
                    tool_use, matched_domains, es_required)
        return tool_use, matched_domains

    # ── 未命中且不强制 es：按优先级 fallback (es > graph > web) ──
    priority_order = ["es", "graph", "web"]
    available = [t for t in priority_order if t in optional_normalized]
    available += [t for t in optional_normalized if t not in priority_order and t != "struct"]

    if available:
        fallback = available[0]
        logger.info("Keyword tool select: struct not matched, fallback to %s", fallback)
        return fallback, []

    # 只有 struct 可用，未命中也使用，带所有 domains
    if "struct" in optional_normalized:
        all_domains = [d.domain for d in domains] if domains else []
        logger.info("Keyword tool select: struct not matched but is only option, using all domains=%s", all_domains)
        return "struct", all_domains

    return "", []


async def handle_simple_thinking(req, llm_client: AsyncOpenAI, llm_model: str,
                                 ts_client: AsyncOpenAI | None = None, ts_model: str = "",
                                 ts_threshold: float = 0) -> PlanningResponse:
    """Simple thinking 模式处理。"""

    if req.tool_select_enable:
        _optional_set = {t.strip() for t in req.tool_hub_optional.split(",") if t.strip()}
        if req.tool_hub:
            raise ValueError("tool_select_enable=True requires tool_hub to be empty")
        if _optional_set != {"struct", "es"}:
            raise ValueError(f"tool_select_enable=True requires tool_hub_optional='struct,es', got: {req.tool_hub_optional}")

    # ── Session 管理：内部维护 history ──
    _cleanup_expired_sessions()
    session_id = _make_session_id(req.query)

    if req.turn == 1:
        # 首轮：初始化 session
        _simple_sessions[session_id] = {"history": {}, "created_at": time.time()}
    else:
        # 非首轮：合并调用方传入的 history 到 session
        if session_id not in _simple_sessions:
            _simple_sessions[session_id] = {"history": {}, "created_at": time.time()}
        session = _simple_sessions[session_id]
        if req.history:
            session["history"].update(req.history)

    history = _simple_sessions[session_id]["history"]

    # 收集原始别名（tool_hub + tool_hub_optional），用于输出时还原
    _orig_hub = {t.strip() for t in req.tool_hub.split(",") if t.strip()}
    _orig_optional = {t.strip() for t in req.tool_hub_optional.split(",") if t.strip()}
    _orig_all = _orig_hub | _orig_optional
    # 内部统一用 graph/es
    req_tool_hub = _normalize_tools(req.tool_hub)
    req_tool_hub_optional = _normalize_tools(req.tool_hub_optional)

    def _restore_items(items: list[SubQueryItem]) -> list[SubQueryItem]:
        return [SubQueryItem(sub_query=it.sub_query,
                             tool_use=_restore_tools(it.tool_use, _orig_all),
                             topk=it.topk, domain=it.domain) for it in items]

    # ── 单轮直检快捷路径：tool_hub 为纯 struct 或 es 且 max_turn=1 ──
    if req.max_turn == 1 and req_tool_hub in ("struct", "es"):
        is_struct = req_tool_hub == "struct"
        tool_label = _restore_tools(req_tool_hub, _orig_all)

        sub_queries = await rewrite_query(llm_client, llm_model, req.query, req.max_top_k)
        if sub_queries is None:
            return PlanningResponse(is_off_topic=False, status="stop", turn=req.turn, final={})
        if not sub_queries:
            return PlanningResponse(is_off_topic=True, status="stop", turn=req.turn, current=[])
        if req.query not in sub_queries:
            sub_queries.insert(0, req.query)

        if is_struct:
            # domains 仅用于结构化检索，LLM 选择相关 domain；无 domains 时无法选择
            if req.domains:
                domains = await _select_domains(llm_client, llm_model, req.query, req.domains)
            else:
                domains = []
            current_items = [
                SubQueryItem(sub_query=sq, tool_use=tool_label, topk=req.max_top_k, domain=domains)
                for sq in sub_queries
            ]
            logger.info("Single-turn struct | query=%r, sub_queries=%s, domains=%s",
                         req.query, sub_queries, domains)
        else:
            current_items = [
                SubQueryItem(sub_query=sq, tool_use=tool_label, topk=req.max_top_k)
                for sq in sub_queries
            ]
            logger.info("Single-turn unstruct | query=%r, sub_queries=%s", req.query, sub_queries)
        return PlanningResponse(is_off_topic=False, status="stop", turn=req.turn, current=current_items)

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

        # ── tool_selection_mode=keyword: 关键词匹配决定工具优先级 ──
        elif req.tool_selection_mode == "keyword" and not req_tool_hub and req_tool_hub_optional:
            kw_threshold = float(req.tool_selection_threshold) if req.tool_selection_threshold is not None else 0.5
            selected_tool, matched_domains = _select_tool_by_keyword(
                req.query, req_tool_hub_optional, req.domains, kw_threshold,
            )
            if selected_tool:
                # 若选中工具含 struct 且有 matched_domains，附上 domain 字段
                tool_set = {t.strip() for t in selected_tool.split(",") if t.strip()}
                use_domain = matched_domains if "struct" in tool_set and matched_domains else None
                current_items = [SubQueryItem(sub_query=sq, tool_use=selected_tool, topk=req.max_top_k,
                                             domain=use_domain) for sq in sub_queries]
            else:
                # 无可用工具，fallback 到所有 optional
                current_items = [SubQueryItem(sub_query=sq, tool_use=req_tool_hub_optional, topk=req.max_top_k)
                                 for sq in sub_queries]
            logger.info("Turn 1 | keyword mode: selected_tool=%s, matched_domains=%s",
                         selected_tool, matched_domains)

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

        rs = req.retrieval_setting or RetrievalSetting()
        score_threshold = rs.score_threshold
        top_k = rs.top_k

        total_qualified = count_total_qualified(history, score_threshold)
        logger.debug("Total qualified (threshold=%.4f): %d, top_k=%d", score_threshold, total_qualified, top_k)

        if total_qualified >= top_k:
            final = collect_all_results_from_history(history, original_query=req.query, top_k=top_k)
            logger.info("Turn %d | total qualified=%d >= top_k=%d, stop.", req.turn, total_qualified, top_k)
            _simple_sessions.pop(session_id, None)
            return PlanningResponse(
                is_off_topic=False, status="stop", turn=req.turn, final=final,
                history=trim_history(history, req.max_context_size),
            )

        current_items = []

        if req.tool_select_enable and ts_client and req_tool_hub_optional and req.turn < req.max_turn:
            all_items = get_sub_queries_with_used_tools(history)
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
            all_items = get_sub_queries_with_used_tools(history)
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

        elif req.tool_selection_mode == "keyword" and req_tool_hub_optional and req.turn < req.max_turn:
            # keyword 模式非首轮：检查 struct 是否已有结果，有则直接 stop
            has_struct_results = False
            if history:
                for turn_key in history.values():
                    for item in turn_key.get("retrieval_contents", []):
                        for tool_type, records in item.get("result", {}).items():
                            if tool_type == "struct" and records:
                                has_struct_results = True
                                break

            if has_struct_results:
                # struct 有结果，直接 stop，不再补充其他工具
                final = collect_all_results_from_history(history, original_query=req.query, top_k=top_k)
                logger.info("Turn %d | keyword mode: struct has results, stop.", req.turn)
                _simple_sessions.pop(session_id, None)
                return PlanningResponse(
                    is_off_topic=False, status="stop", turn=req.turn, final=final,
                    history=trim_history(history, req.max_context_size),
                )

            # struct 无结果，使用所有 tool_hub_optional + 所有 domains（仅给 struct）
            all_tools = req_tool_hub_optional
            all_domains = [d.domain for d in req.domains] if req.domains else []
            all_items = get_sub_queries_with_used_tools(history)
            for item in all_items:
                used = item["used_tools"]
                all_tool_set = {t.strip() for t in all_tools.split(",") if t.strip()}
                new_tools = all_tool_set - used
                if new_tools:
                    current_items.append(SubQueryItem(
                        sub_query=item["sub_query"],
                        tool_use=",".join(sorted(new_tools)),
                        topk=min(item["topk"], req.max_top_k),
                        domain=all_domains if "struct" in new_tools else None,
                    ))
            logger.info("Turn %d | keyword mode fallback: all tools=%s, all domains=%s",
                         req.turn, all_tools, all_domains)

        elif req.tool_selection_mode == "rule" and req_tool_hub_optional and req.turn < req.max_turn:
            optional_tools = {_normalize_tool(t.strip()) for t in req_tool_hub_optional.split(",") if t.strip()}
            all_items = get_sub_queries_with_used_tools(history)
            for item in all_items:
                remaining_tools = optional_tools - item["used_tools"]
                if remaining_tools:
                    current_items.append(SubQueryItem(
                        sub_query=item["sub_query"],
                        tool_use=",".join(sorted(remaining_tools)),
                        topk=min(item["topk"], req.max_top_k),
                    ))

        if not current_items:
            final = collect_all_results_from_history(history, original_query=req.query, top_k=top_k)
            logger.info("Turn %d | no more tools to try, stop.", req.turn)
            _simple_sessions.pop(session_id, None)
            return PlanningResponse(
                is_off_topic=False, status="stop", turn=req.turn, final=final,
                history=trim_history(history, req.max_context_size),
            )

        logger.info("Turn %d | running, %d sub_queries to search", req.turn, len(current_items))
        return PlanningResponse(
            is_off_topic=False, status="running", turn=req.turn,
            current=_restore_items(current_items), history=trim_history(history, req.max_context_size),
        )
