"""Workflow: 多轮 planning 的规则判断 + history 操作 helper。"""

import logging

logger = logging.getLogger("planning_server.workflow")


def collect_all_results_from_history(history: dict | None, original_query: str = "", top_k: int = 3) -> dict:
    """从 history 中组装 final 结果，按工具类型分组。

    组装规则:
    1. 按 sub_query 分组，每个 sub_query 内按 score 降序排列
    2. 原始 query 对应的那路放第一位
    3. 每路至少入围 1 条结果
    4. 轮询取结果直到凑够 top_k，跳过已选过的重复记录
    """
    if not history:
        return {}

    streams_by_tool: dict[str, dict[str, list[dict]]] = {}
    for turn_key in sorted(history.keys()):
        turn_data = history[turn_key]
        for item in turn_data.get("retrieval_contents", []):
            sq = item.get("sub_query", "")
            result = item.get("result", {})
            for tool_type, records in result.items():
                if tool_type not in streams_by_tool:
                    streams_by_tool[tool_type] = {}
                if sq not in streams_by_tool[tool_type]:
                    streams_by_tool[tool_type][sq] = []
                streams_by_tool[tool_type][sq].extend(records)

    for tool_type in streams_by_tool:
        for sq in streams_by_tool[tool_type]:
            streams_by_tool[tool_type][sq].sort(
                key=lambda r: r.get("score", 0) if isinstance(r.get("score", 0), (int, float)) else 0,
                reverse=True,
            )

    final: dict[str, list[dict]] = {}
    for tool_type, sq_map in streams_by_tool.items():
        # struct/graph: 全量保留（每路 sub_query 都保留，不受 top_k 截断）
        if tool_type in ("struct", "graph"):
            all_records: list[dict] = []
            seen_u: set[str] = set()
            for records in sq_map.values():
                for r in records:
                    # struct 用 subQuery/sparqlResult 去重，graph 用 content 去重
                    if tool_type == "struct":
                        key = r.get("subQuery", "") or r.get("sparqlResult", "")
                    else:
                        key = r.get("content", "")
                    if not key:
                        all_records.append(r)
                    elif key not in seen_u:
                        seen_u.add(key)
                        all_records.append(r)
            final[tool_type] = all_records
            continue

        sq_keys = list(sq_map.keys())
        if original_query in sq_keys:
            sq_keys.remove(original_query)
            sq_keys.insert(0, original_query)

        streams: list[tuple[str, list[dict]]] = [(sq, sq_map[sq]) for sq in sq_keys]
        selected: list[dict] = []
        seen: set[str] = set()
        pointers = [0] * len(streams)

        def _record_key(record: dict) -> str:
            return record.get("content", "")

        def _pick_next(stream_idx: int) -> dict | None:
            _, records = streams[stream_idx]
            while pointers[stream_idx] < len(records):
                r = records[pointers[stream_idx]]
                pointers[stream_idx] += 1
                if _record_key(r) not in seen:
                    return r
            return None

        for i in range(len(streams)):
            r = _pick_next(i)
            if r is not None:
                seen.add(_record_key(r))
                selected.append(r)

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
                break

        final[tool_type] = selected
    return final


def get_sub_queries_with_used_tools(history: dict | None) -> list[dict]:
    """遍历 history，收集所有 sub_query 及其已使用过的工具。"""
    if not history:
        return []

    agg: dict[str, dict] = {}
    for turn_key in sorted(history.keys()):
        turn_data = history[turn_key]
        for item in turn_data.get("retrieval_contents", []):
            sq = item.get("sub_query", "")
            if sq not in agg:
                agg[sq] = {"topk": item.get("topk", 3), "used_tools": set()}
            for t in item.get("tool_use", "").split(","):
                t = t.strip()
                if t:
                    agg[sq]["used_tools"].add(t)

    return [{"sub_query": sq, "topk": info["topk"], "used_tools": info["used_tools"]} for sq, info in agg.items()]


def count_total_qualified(history: dict | None, score_threshold: float) -> int:
    """统计 history 中所有达到 score_threshold 的结果总数。

    仅对 es 生效，struct/graph/web 等其他工具结果不参与计数。
    """
    count = 0
    if not history:
        return count
    for turn_key in sorted(history.keys()):
        turn_data = history[turn_key]
        for item in turn_data.get("retrieval_contents", []):
            for tool_type, records in item.get("result", {}).items():
                if tool_type != "es":
                    continue
                for r in records:
                    score = r.get("score", 0)
                    if isinstance(score, (int, float)) and score >= score_threshold:
                        count += 1
    return count


def trim_history(history: dict | None, max_context_size: int) -> dict | None:
    """只保留最近 max_context_size 轮的 history。"""
    if not history:
        return None
    turns = sorted(history.keys())
    if len(turns) <= max_context_size:
        return history
    recent = turns[-max_context_size:]
    return {k: history[k] for k in recent}
