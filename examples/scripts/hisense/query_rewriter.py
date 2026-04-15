# query_rewriter.py
import asyncio
import re
import json
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

from ruler_score import score_group


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

User: query: "闺蜜机有防蓝光吗", topk: 2
Output: {"sub_queries": [闺蜜机有防蓝光吗]}

User: query: "请问一下，那个我的冰箱长期不用了，我应该怎么保养才对啊", topk: 2
Output: {"sub_queries": [如何进行长期停用冰箱的保养]}
"""


def _replace_system_prompt(example):
    """Replace the system prompt and append topk to user message."""
    prompt = example["prompt"]
    for msg in prompt:
        if msg["role"] == "system":
            msg["content"] = SYSTEM_PROMPT
        elif msg["role"] == "user":
            msg["content"] = msg["content"].rstrip() + "\ntopk: 2"
    example["prompt"] = prompt
    return example


dataset = load_dataset("./DeepMath-103K", split="train")
dataset = dataset.map(_replace_system_prompt)
dataset = dataset.shuffle()


def _parse_sub_queries(content: str) -> list[str] | None:
    """Parse sub_queries from completion content. Returns None on failure.

    Supports two formats:
    1. Direct JSON: {"sub_queries": ["q1", "q2"]}
    2. tool_call format: <tool_call>{"name": "...", "arguments": {"query": ["q1", "q2"]}}</tool_call>
    """
    # Strip <think>...</think> blocks (Qwen3 thinking mode)
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    # Format 1: <tool_call> blocks — extract query lists from arguments.query
    tool_call_pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(tool_call_pattern, cleaned, re.DOTALL)
    if matches:
        all_queries = []
        for match in matches:
            try:
                obj = json.loads(match)
                queries = obj.get("arguments", {}).get("query", [])
                if isinstance(queries, str):
                    queries = [queries]
                all_queries.extend(queries)
            except (json.JSONDecodeError, ValueError):
                return None
        return all_queries if all_queries else []

    # Format 2: direct JSON {"sub_queries": [...]}
    # Also handle ```json{...}``` without newline
    json_candidate = cleaned
    if json_candidate.startswith("```"):
        json_candidate = re.sub(r"^```(?:json)?", "", json_candidate)
        json_candidate = re.sub(r"```$", "", json_candidate.rstrip()).strip()
    try:
        obj = json.loads(json_candidate)
        if isinstance(obj, dict) and "sub_queries" in obj:
            return obj["sub_queries"]
    except (json.JSONDecodeError, ValueError):
        pass

    # Format 3: JSON embedded in markdown or surrounding text
    m = re.search(r'\{[^{}]*"sub_queries"\s*:\s*\[.*?\][^{}]*\}', cleaned, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "sub_queries" in obj:
                return obj["sub_queries"]
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _extract_topk(prompts) -> int:
    """Extract topk from the user message in prompts."""
    for msg in reversed(prompts[0]):
        if msg.get("role") == "user":
            m = re.search(r"topk:\s*(\d+)", msg.get("content", ""))
            if m:
                return int(m.group(1))
    return 3  # default


def _extract_question(prompts) -> str:
    """Extract the original user query from prompts."""
    for msg in reversed(prompts[0]):
        if msg.get("role") == "user":
            m = re.search(r'query:\s*"(.*?)"', msg.get("content", ""))
            if m:
                return m.group(1)
            return msg.get("content", "")
    return ""


def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion is valid JSON with sub_queries.

    Returns -1 if the format is wrong, 0 otherwise.
    """
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content in completion_contents:
        sub_queries = _parse_sub_queries(content)
        if sub_queries is None:
            rewards.append(-1.0)
        else:
            rewards.append(0.0)
    return rewards


def topk_reward_func(completions, prompts, **kwargs):
    """Reward function that checks the number of sub_queries does not exceed topk.

    Returns -1 if sub_queries count exceeds topk, 0 otherwise.
    """
    topk = _extract_topk(prompts)
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content in completion_contents:
        sub_queries = _parse_sub_queries(content)
        if sub_queries is None:
            rewards.append(-1.0)
        elif len(sub_queries) > topk:
            rewards.append(-1.0)
        else:
            rewards.append(0.0)
    return rewards


def query_length_reward_func(completions, **kwargs):
    """Reward function that penalizes long sub-queries.

    Returns -avg_len/100 as the reward. Shorter queries get scores closer to 0.
    Returns -1 if parsing fails.
    """
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content in completion_contents:
        sub_queries = _parse_sub_queries(content)
        if sub_queries is None or not sub_queries:
            # Empty list is valid (not suitable for search), no length penalty
            rewards.append(0.0 if sub_queries is not None else -1.0)
        else:
            avg_len = sum(len(q) for q in sub_queries) / len(sub_queries)
            rewards.append(-avg_len / 100.0)
    return rewards


async def _async_ruler_reward(completions, prompts, **kwargs):
    """Async implementation: use RULER judge to score sub-query quality.

    Scoring criteria:
    1. Whether each sub-query is sufficiently atomic
    2. Whether all sub-queries together fully cover the original question
    """
    completion_contents = [c[0]["content"] for c in completions]
    question = _extract_question(prompts)
    topk = _extract_topk(prompts)

    trajectories = []
    traj_indices = []
    for i, content in enumerate(completion_contents):
        sub_queries = _parse_sub_queries(content)
        if sub_queries is None:
            print(f"[rewriter_reward] Completion {i} | PARSE FAILED | Raw content:\n{content[:500]}")
            continue

        print(f"[rewriter_reward] Completion {i} | Question: {question}")
        print(f"[rewriter_reward] Sub-queries: {sub_queries}")

        trajectory = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"原始问题: {question}\ntopk: {topk}"},
            {"role": "assistant", "content": json.dumps({"sub_queries": sub_queries}, ensure_ascii=False)},
        ]
        trajectories.append(trajectory)
        traj_indices.append(i)

    rewards = [0.0] * len(completion_contents)
    if trajectories:
        try:
            scores = await _score_with_auto_split(trajectories, question)
            for idx, score in zip(traj_indices, scores):
                rewards[idx] = score
                print(f"[rewriter_reward] Completion {idx} | RULER score: {score:.3f}")
        except Exception as e:
            print(f"[rewriter_reward] RULER scoring failed after all retries: {e}")

    print(f"[rewriter_reward] Final rewards: {rewards}")
    return rewards


async def _score_with_auto_split(trajectories: list, question: str) -> list[float]:
    """Score trajectories with RULER, automatically splitting into smaller chunks on token limit errors."""
    goal = (
        f"评估query改写质量。原始问题: {question}\n"
        "评分要点:\n"
        "1. 每个sub-query是否足够原子化（每个子查询只问一件事）\n"
        "2. 所有sub-queries合在一起是否能够完整覆盖原问题的所有意图\n"
        "3. 如果原问题不适合搜索，sub_queries应为空列表\n"
        "4. 复合类型（如对比类问题）应优先按主语分解，而非按属性分解。"
        "例如'410和510有啥区别'应分解为'410特点','510特点'，"
        "而不是'410和510性能区别','410和510价格区别'"
    )
    for num_chunks in (1, 2, 4):
        try:
            if num_chunks == 1:
                results = await score_group(
                    message_lists=trajectories,
                    #goal=goal,
                )
                return [r.score for r in results]

            chunk_size = (len(trajectories) + num_chunks - 1) // num_chunks
            all_scores = []
            for i in range(0, len(trajectories), chunk_size):
                chunk = trajectories[i : i + chunk_size]
                print(f"[rewriter_reward] Scoring chunk {i // chunk_size + 1}/{num_chunks} ({len(chunk)} trajectories)")
                results = await score_group(
                    message_lists=chunk,
                    goal=goal,
                )
                all_scores.extend(r.score for r in results)
            return all_scores

        except ValueError as e:
            if "token_limit_exceeded" in str(e) or "request_too_large" in str(e):
                next_chunks = {1: 2, 2: 4}.get(num_chunks)
                if next_chunks:
                    print(f"[rewriter_reward] Token limit exceeded with {num_chunks} chunk(s), retrying with {next_chunks}")
                    continue
            raise

    raise ValueError("Token limit exceeded even after splitting into 4 chunks")


async def ruler_reward_func(completions, prompts, **kwargs):
    """Reward function: use RULER LLM judge to evaluate sub-query rewriting quality.

    Scoring criteria:
    1. Whether each sub-query is sufficiently atomic (asks about one thing)
    2. Whether all sub-queries together fully cover the original question
    """
    return await _async_ruler_reward(completions, prompts, **kwargs)


training_args = GRPOConfig(
    output_dir="./query_rewriter_output_4B_multi",
    logging_steps=1,
    use_vllm=True,
    vllm_mode="server",
    log_completions=True,
    save_strategy="steps",
    save_steps=20,
    chat_template_kwargs={"enable_thinking": False},
)

trainer = GRPOTrainer(
    model="../../../../../models/Qwen3-4B-Instruct-2507/",
    reward_funcs=[topk_reward_func, ruler_reward_func],
    train_dataset=dataset,
    args=training_args,
)
trainer.train()
