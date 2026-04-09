# train_grpo.py
import re
import json
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig


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
        # Check all tool_call blocks in the content
        matches = re.findall(pattern, content, re.DOTALL)
        if not matches:
            rewards.append(-1.0)
            continue
        valid = True
        for match in matches:
            # Extract the JSON part between <tool_call> and </tool_call>
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


training_args = GRPOConfig(
    output_dir="./grpo_results",
    logging_steps=1,
)

trainer = GRPOTrainer(
    model="../../../../../models/Qwen3-4B-Instruct-2507/",
    reward_funcs=[format_reward_func, query_count_reward_func, query_length_reward_func],
    train_dataset=dataset,
    args=training_args,
)
trainer.train()
