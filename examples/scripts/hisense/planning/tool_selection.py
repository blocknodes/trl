"""Tool Selection: 调用 LLM 判断 tool_hub_optional 中的工具（如 graph）是否适用于当前 query。"""

import logging

from openai import AsyncOpenAI

from helpers import _parse_llm_json
from prompts import GRAPH_TOOL_SELECTION_PROMPT

logger = logging.getLogger("planning_server.tool_selection")

# 需要 LLM 判定的工具 -> 对应的判定 prompt 模板（必须含 {query} 占位符）
TOOL_SELECTION_PROMPTS: dict[str, str] = {
    "graph": GRAPH_TOOL_SELECTION_PROMPT,
}


async def select_optional_tools(
    client: AsyncOpenAI,
    model: str,
    query: str,
    tool_hub: str,
    tool_hub_optional: str,
    threshold: float = 0,
) -> str:
    """对 tool_hub_optional 中有判定 prompt 的工具调用 LLM，in_scope 且 confidence >= threshold 的合并进 tool_hub。

    返回合并后的 tool_hub 字符串。
    """
    optional = [t.strip() for t in tool_hub_optional.split(",") if t.strip()]
    to_check = [t for t in optional if t in TOOL_SELECTION_PROMPTS]

    if not to_check:
        return tool_hub

    hub_tools = [t.strip() for t in tool_hub.split(",") if t.strip()]
    for tool in to_check:
        prompt = TOOL_SELECTION_PROMPTS[tool].replace("{query}", query)
        messages = [{"role": "user", "content": prompt}]
        try:
            logger.debug("Tool selection [%s] query=%r, prompt length=%d", tool, query, len(prompt))
            completion = await client.chat.completions.create(
                model=model, messages=messages, temperature=0, max_tokens=512,
            )
            content = completion.choices[0].message.content or ""
            logger.debug("Tool selection [%s] raw response: %s", tool, content[:500])
            parsed = _parse_llm_json(content)
            in_scope = parsed.get("in_scope", False) if parsed else False
            confidence = float(parsed.get("confidence_score", 0.0)) if parsed else 0.0
            reason = parsed.get("reason", "") if parsed else "parse failed"
            logger.info("Tool selection [%s] query=%r -> in_scope=%s, confidence=%.2f, threshold=%.2f, reason=%s",
                        tool, query, in_scope, confidence, threshold, reason)
            if in_scope and confidence >= threshold and tool not in hub_tools:
                hub_tools.append(tool)
        except Exception:
            logger.exception("Tool selection [%s] failed, skipping", tool)

    result = ",".join(hub_tools)
    logger.info("Tool hub after selection: %s", result)
    return result
