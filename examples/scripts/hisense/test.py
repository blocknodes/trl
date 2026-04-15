# train_grpo.py
import asyncio
import re
import json
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

from planner_client import kbp_search
from ruler_score import score_group


dataset = load_dataset("./DeepMath-103K", split="train")


def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion follows the tool_call format.

    Expected format:
        <tool_call>
        {"name": "...", "arguments": {"query": [...]}}
        </tool_call>

    Returns -1 if the format is wrong, 0 otherwise.
    """
    pattern = r"<tool_call>\s*\{.*?\}\s*</tool_call>"
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content in completion_contents:
        matches = re.findall(pattern, content, re.DOTALL)
        if not matches:
            rewards.append(-1.0)
            continue
        valid = True
        for match in matches:
            json_str = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", match, re.DOTALL)
            if not json_str:
                valid = False
                break
            try:
                obj = json.loads(json_str.group(1))
                if "name" not in obj or "arguments" not in obj:
                    valid = False
                    break
            except (json.JSONDecodeError, ValueError):
                valid = False
                break
        rewards.append(0.0 if valid else -1.0)
    return rewards


def query_count_reward_func(completions, **kwargs):
    """Reward function that checks the number of queries in tool_call arguments.

    Returns -1 if any tool_call block has fewer than 2 or more than 3 queries, 0 otherwise.
    """
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content in completion_contents:
        matches = re.findall(pattern, content, re.DOTALL)
        if not matches:
            rewards.append(-1.0)
            continue
        min_query_count = float("inf")
        valid = True
        for match in matches:
            try:
                obj = json.loads(match)
                queries = obj.get("arguments", {}).get("query", [])
                min_query_count = min(min_query_count, len(queries))
            except (json.JSONDecodeError, ValueError):
                valid = False
                break
        if not valid or min_query_count < 2 or min_query_count > 3:
            rewards.append(-1.0)
        else:
            rewards.append(0.0)
    return rewards


def query_length_reward_func(completions, **kwargs):
    """Reward function that penalizes long queries. Score is inversely proportional to query length.

    For each completion, computes the average length of all queries across all tool_call blocks,
    then returns -1/avg_len as the reward (shorter queries get scores closer to 0, longer ones
    get more negative scores). Returns -1 if parsing fails.
    """
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content in completion_contents:
        matches = re.findall(pattern, content, re.DOTALL)
        if not matches:
            rewards.append(-1.0)
            continue
        all_lengths = []
        valid = True
        for match in matches:
            try:
                obj = json.loads(match)
                queries = obj.get("arguments", {}).get("query", [])
                all_lengths.extend(len(q) for q in queries)
            except (json.JSONDecodeError, ValueError):
                valid = False
                break
        if not valid or not all_lengths:
            rewards.append(-1.0)
        else:
            avg_len = sum(all_lengths) / len(all_lengths)
            rewards.append(-avg_len / 100.0)  # normalize: 10 chars -> -0.1, 50 chars -> -0.5
    return rewards


def _parse_inner_search_queries(content: str) -> list[list[str]]:
    """Extract query lists from all inner_search tool_call blocks in a completion.

    Returns a list of query lists (one per inner_search tool_call block).
    Returns an empty list if no valid inner_search blocks are found.
    """
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, content, re.DOTALL)
    all_queries = []
    for match in matches:
        try:
            obj = json.loads(match)
            if obj.get("name") == "inner_search":
                queries = obj.get("arguments", {}).get("query", [])
                if isinstance(queries, str):
                    queries = [queries]
                if queries:
                    all_queries.append(queries)
        except (json.JSONDecodeError, ValueError):
            continue
    return all_queries


async def _async_kb_reward(completions, prompts, **kwargs):
    """Async implementation: call KBP search for inner_search queries, then use RULER to score the group."""
    completion_contents = [c[0]["content"] for c in completions]
    # All completions in a batch share the same prompt, take the question from the first one
    question = ""
    for msg in reversed(prompts[0]):
        if msg.get("role") == "user":
            question = msg.get("content", "")
            break

    # Step 1: for each completion, parse inner_search queries and call KBP concurrently
    # Each completion gets at most 3 results total, distributed evenly across subqueries
    MAX_RESULTS_PER_COMPLETION = 3
    all_search_tasks = []  # (index, query, top_k)
    query_map = {}  # index -> list of queries
    for i, content in enumerate(completion_contents):
        query_groups = _parse_inner_search_queries(content)
        if not query_groups:
            query_map[i] = None
            continue
        flat_queries = [q for group in query_groups for q in group]
        query_map[i] = flat_queries
        n_queries = len(flat_queries)
        per_query = max(1, MAX_RESULTS_PER_COMPLETION // n_queries)
        for q in flat_queries:
            all_search_tasks.append((i, q, per_query))

    # Execute all KBP searches concurrently
    search_results_by_idx: dict[int, list[str]] = {}
    if all_search_tasks:
        tasks = [kbp_search(q, top_k=top_k) for _, q, top_k in all_search_tasks]
        results = await asyncio.gather(*tasks)
        for (idx, _, _), result in zip(all_search_tasks, results):
            search_results_by_idx.setdefault(idx, []).append(result)

    # Step 2: build trajectories for all completions that have inner_search
    trajectories = []
    traj_indices = []  # which completion index each trajectory corresponds to
    for i, content in enumerate(completion_contents):
        if query_map[i] is None:
            continue
        combined = "\n=======\n".join(search_results_by_idx.get(i, []))

        print(f"[kb_reward] Completion {i} | Question: {question}")
        print(f"[kb_reward] Queries: {query_map[i]}")
        print(f"[kb_reward] Search results:\n{combined[:2000]}{'... (truncated)' if len(combined) > 2000 else ''}")
        print()

        trajectory = [
            {"role": "system", "content": "You are a research assistant. Answer the user's question based on search results."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": combined},
        ]
        trajectories.append(trajectory)
        traj_indices.append(i)

    # Step 3: score the whole group with RULER, auto-split if token limit exceeded
    rewards = [0.0] * len(completion_contents)
    if trajectories:
        try:
            scores = await _score_with_auto_split(trajectories, question)
            for idx, score in zip(traj_indices, scores):
                rewards[idx] = score
        except Exception as e:
            print(f"[kb_reward] RULER scoring failed after all retries: {e}")

    return rewards


async def _score_with_auto_split(trajectories: list, question: str) -> list[float]:
    """Score trajectories with RULER, automatically splitting into smaller chunks on token limit errors."""
    for num_chunks in (1, 2, 4):
        try:
            if num_chunks == 1:
                results = await score_group(
                    message_lists=trajectories,
                    goal=f"根据搜索结果，能否全面准确的回答用户问题: {question}",
                )
                return [r.score for r in results]

            # Split into chunks and score each chunk separately
            chunk_size = (len(trajectories) + num_chunks - 1) // num_chunks
            all_scores = []
            for i in range(0, len(trajectories), chunk_size):
                chunk = trajectories[i : i + chunk_size]
                print(f"[kb_reward] Scoring chunk {i // chunk_size + 1}/{num_chunks} ({len(chunk)} trajectories)")
                results = await score_group(
                    message_lists=chunk,
                    goal=f"根据搜索结果，能否全面准确的回答用户问题: {question}",
                )
                all_scores.extend(r.score for r in results)
            return all_scores

        except ValueError as e:
            if "token_limit_exceeded" in str(e) or "request_too_large" in str(e):
                next_chunks = {1: 2, 2: 4}.get(num_chunks)
                if next_chunks:
                    print(f"[kb_reward] Token limit exceeded with {num_chunks} chunk(s), retrying with {next_chunks}")
                    continue
            raise

    raise ValueError("Token limit exceeded even after splitting into 4 chunks")


async def kb_reward_func(completions, prompts, **kwargs):
    """Reward function: for inner_search tool_calls, call KBP search and use RULER to score results.

    Parses inner_search tool_call blocks from completions, executes real KBP searches,
    then uses the RULER LLM judge to evaluate the quality of search results against the
    original user question. Returns a score between 0 and 1 for completions with
    inner_search calls, or 0.0 if no inner_search is present.
    """
    return await _async_kb_reward(completions, prompts, **kwargs)


training_args = GRPOConfig(
    output_dir="./new_output1",
    logging_steps=1,
    use_vllm=True,
    vllm_mode="server",
    log_completions=True,
    save_strategy="steps",
    save_steps=20,
)

trainer = GRPOTrainer(
    model="../../../../../models/Qwen3-4B-Instruct-2507/",
    reward_funcs=[kb_reward_func],
    train_dataset=dataset,
    args=training_args,
)
trainer.train()
