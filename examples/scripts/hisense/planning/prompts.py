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
7. Every sub-query MUST be self-contained and independently searchable. \
NEVER use pronouns or references like "这些", "那些", "上述", "它们", "其中", "从结果中". \
Each sub-query must include all necessary context (brand, category, parameters, price range, etc.).
8. Sub-queries must be parallelizable with NO dependencies between them. \
Do NOT split a single condition into "find X" then "check if X meets Y". \
Instead combine all conditions into one query, e.g. "1匹低端挂机空调 价格1300-1700元".

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

User: query: "适合7-10平米的1匹低端挂机空调 价格1300-1700元", topk: 3
Output: {"sub_queries": ["1匹低端挂机空调 适合7-10平米 价格1300-1700元"]}

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
2. 绝对禁止出现任何指代词或模糊引用，包括但不限于：'这些'、'那些'、'上述'、'前面提到的'、\
'初始结果'、'符合条件的'、'查询结果中'、'在...范围内'（不带具体数值）、'相关型号'、\
'这些型号'、'这些产品'、'它们'等。违反此规则视为失败。
3. 所有引用必须替换为具体的数值、型号名、品牌名、参数值。
4. 如果已知事实中没有对应的具体值，则删除该限定条件，不要用模糊表述代替。
5. 输出的 query 应该简洁直接，像用户直接在搜索框输入的那样。
6. 只输出改写后的 query 文本，不要解释、不要加引号。

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

# ── Domain Selection (结构化检索) ──────────────────────────────

RDF_SUMMARIZE_PROMPT = """\
你是一个 schema 摘要助手。给定一个领域名称和它的 JSON 定义，用一句话概括该领域包含哪些核心实体和属性。

领域名称: %s

JSON 定义:
%s

请直接输出一句话摘要，不要加任何格式或前缀。
"""

DOMAIN_SELECT_PROMPT = """\
你是一个领域分类助手。给定用户查询和一组可用的领域（domain），选出与查询相关的领域。

可用领域及其描述:
%s

规则:
1. 只从给定的领域列表中选择，不要编造新领域。
2. 选出所有与查询相关的领域，可以是一个或多个。
3. 如果没有任何领域与查询相关，返回空列表。

用户查询: %s

请用 JSON 格式回答:
{"domains": ["domain1", "domain2"]}
"""

# ── Tool Selection: Graph ───────────────────────────────────────

GRAPH_TOOL_SELECTION_PROMPT = """\
判断以下用户查询是否适合使用知识图谱(graph/结构化检索)工具进行检索。

适合结构化检索的查询（应返回 in_scope=true）：
- 产品参数、规格、型号查询（如价格、尺寸、重量、功率、分辨率等）
- 实体属性查询（如"XX的价格是多少"、"XX的参数"）
- 实体关系查询（如"XX和YY的区别"、"XX属于什么系列"）
- 产品对比（如"A和B哪个好"）
- 分类、系列、品牌等层级关系查询

不适合结构化检索的查询（应返回 in_scope=false）：
- 闲聊、问候
- 主观评价、用户体验（如"XX好不好用"、"推荐一下"）
- 操作指南、使用教程（如"怎么连接WiFi"、"如何安装"）
- 故障排查、维修方法（如"黑屏怎么办"、"不制冷怎么修"）

用户查询: {query}

请用 JSON 格式回答:
{"in_scope": true/false, "confidence_score": 0.0-1.0, "reason": "简要说明"}
"""
