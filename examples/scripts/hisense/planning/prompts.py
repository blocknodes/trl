"""所有 LLM prompt 模板集中管理。"""

# ── Query Rewrite ───────────────────────────────────────────────

QUERY_REWRITE_PROMPT = """\
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

# ── Deep Thinking: Plan ─────────────────────────────────────────

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

# ── Deep Thinking: Summary ──────────────────────────────────────

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

# ── Deep Thinking: Final Answer ─────────────────────────────────

DEEP_FINAL_PROMPT = """\
You are a research assistant. Given all the research summaries collected across multiple steps, \
synthesize a final comprehensive answer to the original question.

Original question: %s

Research summaries:
%s

Respond with a JSON object:
{"answer": "comprehensive final answer", "reference_tree": [{"step": 1, "goal": "...", "summary": "...", "references": [...]}]}
"""

# ── Deep Thinking: Goal Resolve ─────────────────────────────────

DEEP_RESOLVE_PROMPT = """\
你的任务：将搜索目标改写为一个完全自包含的搜索 query。

规则（必须严格遵守）：
1. 搜索引擎没有任何记忆，不知道之前搜过什么，所以输出的 query 必须包含所有必要信息。
2. 禁止出现任何模糊引用，包括但不限于：'初始结果'、'上述'、'前面提到的'、\
'符合条件的'、'在...范围内'（不带具体数值）、'相关型号'等。
3. 所有引用必须替换为具体的数值、型号名、品牌名、参数值。
4. 如果已知事实中没有对应的具体值，则删除该限定条件，不要用模糊表述代替。
5. 只输出改写后的 query 文本，不要解释、不要加引号。

已知事实（来自前面的搜索结果）:
%s

前面的研究摘要:
%s

原始搜索目标: %s

改写后的搜索 query:\
"""

# ── Dynamic Thinking: React (搜索后判断是否继续) ────────────────

DYNAMIC_REACT_PROMPT = """\
You are a research assistant using a React-style loop: Search → Summarize → Decide.

You just received search results for a user question. Your tasks:
1. Summarize the findings so far.
2. Decide whether the information is sufficient to answer the original question, or if further searching is needed.
3. If further searching is needed, provide the next search goal.

Original question: %s

Previous research context:
%s

Current search results:
%s

Respond with a JSON object:
- If sufficient: {"status": "stop", "summary": "...", "key_facts": ["..."], "references": [...]}
- If more search needed: {"status": "continue", "summary": "...", "key_facts": ["..."], "references": [...], "next_goal": "what to search next and why"}
"""

# ── Tool Selection: Graph ───────────────────────────────────────

GRAPH_TOOL_SELECTION_PROMPT = """\
判断以下用户查询是否适合使用知识图谱(graph)工具进行检索。

知识图谱适用于：实体关系查询、属性对比、型号参数查询、产品关联关系等结构化问题。
知识图谱不适用于：闲聊、主观评价、操作指南、故障排查等非结构化问题。

用户查询: {query}

请用 JSON 格式回答:
{"in_scope": true/false, "confidence_score": 0.0-1.0, "reason": "简要说明"}
"""
