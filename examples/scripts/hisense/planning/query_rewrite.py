"""Query rewrite: 调用 LLM 将 query 分解为 sub_queries。"""

import logging
from openai import AsyncOpenAI

from helpers import _parse_llm_json
from prompts import QUERY_REWRITE_PROMPT

logger = logging.getLogger("planning_server.query_rewrite")


async def rewrite_query(client: AsyncOpenAI, model: str, query: str, max_top_k: int) -> list[str] | None:
    """调用 LLM 做 query rewrite，返回 sub_queries 列表。

    返回 None 表示 LLM 输出解析失败；返回空列表表示闲聊/不适合搜索。
    """
    messages = [
        {"role": "system", "content": QUERY_REWRITE_PROMPT},
        {"role": "user", "content": f'query: "{query}"\ntopk: {max_top_k}'},
    ]
    completion = await client.chat.completions.create(
        model=model, messages=messages, temperature=0.7, max_tokens=2048,
    )
    content = completion.choices[0].message.content or ""
    logger.debug("LLM response length: %d chars", len(content))
    parsed = _parse_llm_json(content)
    if parsed is None:
        logger.warning("Failed to parse LLM JSON output")
        return None
    return parsed.get("sub_queries", [])
