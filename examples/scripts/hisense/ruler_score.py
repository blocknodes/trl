"""
Standalone RULER scoring function.

Extracts the core LLM-as-judge logic from ruler.py into a single function
that takes a group of message trajectories and returns scores + reasons,
without depending on art.TrajectoryGroup.
"""

import json
import re
from dataclasses import dataclass
from textwrap import dedent
from typing import List

import aiohttp
from pydantic import BaseModel, Field


# ---------- response schema for the LLM judge ----------

class _TrajectoryScore(BaseModel):
    trajectory_id: str = Field(description="The id of the trajectory being scored.")
    explanation: str = Field(description="A short description of the trajectory's performance.")
    score: float = Field(description="A score between 0 and 1.")


class _JudgeResponse(BaseModel):
    scores: List[_TrajectoryScore] = Field(description="The scores for each trajectory.")


# ---------- public return type ----------

@dataclass
class ScoreResult:
    """Single trajectory scoring result."""
    score: float
    reason: str


# ---------- default rubric ----------

DEFAULT_RUBRIC = dedent("""\
    - A trajectory that achieves its goal should always get a significantly higher score than a trajectory that does not achieve its goal.
    - A trajectory that achieves its goal more efficiently (eg. by avoiding unproductive detours) should get a higher score than a trajectory that achieves its goal less efficiently.
    - If one trajectory is only slightly better than another, the difference in scores should be small. If it is significantly better, the difference in scores should be large.
    - You may give some partial credit for a trajectory that makes progress towards its goal but does not complete it.
""")


# ---------- API config ----------

JUDGE_API_URL = "https://aix-backup.hismarttv.com/v1/chat/completions"
JUDGE_API_KEY = "x31ctKZ0ONfi1jkO"
JUDGE_MODEL = "deepseek-v3"


# ---------- main function ----------

async def score_group(
    message_lists: list[list[dict]],
    goal: str | None = None,
    rubric: str = DEFAULT_RUBRIC,
    tools: list | None = None,
    debug: bool = False,
) -> list[ScoreResult]:
    """Score a group of trajectories using an LLM judge (RULER).

    This is a standalone function extracted from ruler.py. It takes raw message
    lists (one per trajectory), sends them to an LLM judge for relative scoring,
    and returns a list of (score, reason) pairs.

    Args:
        message_lists: Each element is a conversation (list of message dicts)
            representing one trajectory in the group.
        goal: Explicit description of the task goal. If provided, the judge
            will use it directly; if None, the judge infers the goal from
            the system/user messages in the trajectories.
        rubric: Grading rubric text. The default works well for most tasks.
        tools: Optional tool definitions the agent had access to.
        debug: If True, pretty-print the raw judge response.

    Returns:
        A list of ScoreResult, one per trajectory, in the same order as
        message_lists. Each contains a float score (0-1) and a reason string.
    """
    if not message_lists:
        return []

    # ---- find common prefix to save tokens ----
    common_prefix_len = 0
    for idx, msg in enumerate(message_lists[0]):
        if all(len(ml) > idx and ml[idx] == msg for ml in message_lists):
            common_prefix_len += 1
        else:
            break

    all_identical = all(len(ml) == common_prefix_len for ml in message_lists)

    # ---- build user prompt ----
    user_text = ""

    if common_prefix_len > 0 and not all_identical:
        common_prefix = message_lists[0][:common_prefix_len]
        user_text += "<context>\n" + json.dumps(common_prefix) + "\n</context>\n\n"

    if tools:
        user_text += "<available_tools>\n" + json.dumps(tools) + "\n</available_tools>\n\n"

    serialized: list[str] = []
    if all_identical:
        serialized.append(
            f'<trajectory id="1">\n' + json.dumps(message_lists[0]) + "\n</trajectory>"
        )
    else:
        for i, full_msgs in enumerate(message_lists, start=1):
            trimmed = full_msgs[common_prefix_len:]
            serialized.append(
                f'<trajectory id="{i}">\n' + json.dumps(trimmed) + "\n</trajectory>"
            )

    user_text += "Trajectories:\n\n" + "\n\n".join(serialized)

    if goal:
        goal_section = f"The goal of the agent is:\n{goal}"
    else:
        goal_section = (
            "The goal is not explicitly provided. Infer it from the system/user "
            "messages in the trajectories."
        )

    system_prompt = dedent(f"""\
        All of the trajectories below have been given the same goal. Your job is to \
        consider each of them and give them a score between 0 and 1.

        {goal_section}

        Grading standards:
        {rubric}
    """)

    # Append format instruction so the model returns valid JSON
    format_instruction = (
        "\n\nYou MUST respond with a JSON object matching this schema:\n"
        + json.dumps(_JudgeResponse.model_json_schema(), ensure_ascii=False)
        + "\nReturn ONLY the JSON, no extra text."
    )
    messages = [
        {"role": "system", "content": system_prompt + format_instruction},
        {"role": "user", "content": user_text},
    ]

    # ---- call LLM judge via HTTP ----
    payload = {
        "model": JUDGE_MODEL,
        "messages": messages,
        "stream": False,
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {JUDGE_API_KEY}",
    }

    print(f"[RULER] Scoring {len(message_lists)} trajectories | Goal: {goal}")

    async with aiohttp.ClientSession() as session:
        async with session.post(JUDGE_API_URL, json=payload, headers=headers,
                                timeout=aiohttp.ClientTimeout(total=120)) as resp:
            resp_data = await resp.json()

    choices = resp_data.get("choices", [])
    if not choices:
        raise ValueError(f"No choices in response: {resp_data}")

    content = choices[0].get("message", {}).get("content", "{}")

    # Strip markdown code fences if the model wrapped JSON in ```json ... ```
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1] if "\n" in stripped else stripped[3:]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3].rstrip()
        content = stripped

    print(f"[RULER] Raw judge response:\n{content[:3000]}{'... (truncated)' if len(content) > 3000 else ''}")

    if debug:
        try:
            print("[RULER] Parsed judge response:")
            print(json.dumps(json.loads(content), indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print(f"[RULER] Failed to parse as JSON: {content}")

    try:
        parsed = _JudgeResponse.model_validate_json(content)
    except Exception:
        # Fallback: try to fix common JSON issues (unescaped quotes in explanation)
        try:
            # Extract score fields with regex as last resort
            scores = []
            for m in re.finditer(
                r'"trajectory_id"\s*:\s*"(\d+)".*?"score"\s*:\s*([\d.]+)',
                content,
                re.DOTALL,
            ):
                scores.append(_TrajectoryScore(
                    trajectory_id=m.group(1),
                    explanation="(parse fallback)",
                    score=float(m.group(2)),
                ))
            if not scores:
                raise ValueError(f"Could not extract any scores from: {content[:500]}")
            parsed = _JudgeResponse(scores=scores)
            print(f"[RULER] Used regex fallback, extracted {len(scores)} scores")
        except Exception as e2:
            raise ValueError(f"JSON parse failed and regex fallback failed: {e2}\nRaw: {content[:500]}")

    # ---- build results ----
    for s in parsed.scores:
        print(f"[RULER] Trajectory {s.trajectory_id}: score={s.score:.3f} | {s.explanation}")
    if all_identical and len(message_lists) > 1:
        if len(parsed.scores) != 1:
            raise ValueError(
                f"Expected 1 score for identical trajectories, got {len(parsed.scores)}"
            )
        s = parsed.scores[0]
        return [ScoreResult(score=s.score, reason=s.explanation)] * len(message_lists)

    if len(parsed.scores) != len(message_lists):
        raise ValueError(
            f"Expected {len(message_lists)} scores, got {len(parsed.scores)}"
        )

    return [
        ScoreResult(score=s.score, reason=s.explanation) for s in parsed.scores
    ]
