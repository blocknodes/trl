"""
Planner Server: FastAPI 服务，封装 planner.py 中的 LLM 交互逻辑。

Server 维护每个会话的 messages 状态，每次请求调用 LLM 一轮，
使用和 planner.py 完全一致的 SYSTEM_PROMPT、guided_regex 约束解码、stop_tokens，
解析 tool_call / answer / chat 后返回给 client。

用法:
    python examples/scripts/hisense/planner_server.py \
        --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen3-1.7B \
        --port 9000
"""

import argparse
import datetime
import json
import uuid

from fastapi import FastAPI
from pydantic import BaseModel
from openai import AsyncOpenAI

# ── 和 planner.py 完全一致的 SYSTEM_PROMPT ──────────────────────
SYSTEM_PROMPT = """You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response.
# Tools
You may call one or more functions to assist with the user query, but you should prefer inner_search than web_search.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "inner_search", "description": "Perform semantic search from inner knowledge base then returns a string of the top search results. Accepts multiple queries.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries."}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "web_search", "description": "Perform web search (not keywords search) then returns a string of the top search results. Accepts multiple queries.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries."}}, "required": ["query"]}}}
</tools>

For each function call, return a list of json objects with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Example1:
User query: 请问下E8Q这款电视的像素是多少？
Output: <tool_call>
{"name": "inner_search", "arguments": {"query": ["E8Q电视的像素是多少"]}}
</tool_call>

Example2:
User query: 海尔H5C保修政策
Output: <tool_call>
{"name": "web_search", "arguments": {"query": ["海尔H5C保修政策"]}}
</tool_call>

Example3:
User query: 查询KFR-26GW/QS1Lite-X1、KFR-26GW/QS1Lite-B1、KFR-26GW/QS1Lite-C1能效的卖点
Output: <tool_call>
{"name": "inner_search", "arguments": {"query": ["KFR-26GW/QS1Lite-X1能效的卖点","KFR-26GW/QS1Lite-B1能效的卖点","KFR-26GW/QS1Lite-C1能效的卖点"]}}
</tool_call>

Example4:
User query: U8Q和小米S pro Mini LED哪个画质好？
Output: <tool_call>
{"name": "inner_search", "arguments": {"query": ["U8Q画质表现"]}}
</tool_call>
<tool_call>
{"name": "web_search", "arguments": {"query": ["小米S pro Mini LED画质"]}}
</tool_call>

注意：
1. query中品牌不是海信/hisense，比如海尔/小米/创维等，需要使用web_search
2. 多主语需要分开
3. Your FIRST response MUST start with one of: `<tool_call>` or `<chat>`. All subsequent responses MUST start with either `<answer>` or `<tool_call>` only.

When you are ready to provide your final answer, you MUST start your response with `<answer>` and end with `</answer>`.

If the user's FIRST message is casual conversation or chitchat (not a research question), respond with `<chat>` at the beginning and `</chat>` at the end.

Current date: """


# ── 和 planner.py 完全一致的 stop tokens 和约束解码 regex ───────
STOP_TOKENS = ["\n<tool_response>", "<tool_response>"]
GUIDED_REGEX_FIRST = r"(<tool_call>[\s\S]*</tool_call>|<chat>[\s\S]*</chat>)"
GUIDED_REGEX_EXEC = r"(<answer>[\s\S]*</answer>|<tool_call>[\s\S]*</tool_call>)"


def build_messages(question: str) -> list[dict]:
    """构造和 planner.py 一致的 messages 结构。"""
    cur_date = datetime.date.today().strftime("%Y-%m-%d")
    return [
        {"role": "system", "content": SYSTEM_PROMPT + cur_date},
        {"role": "user", "content": question},
    ]


def parse_tool_calls(content: str) -> list[tuple[str, dict]]:
    """从模型输出中解析所有 <tool_call>...</tool_call> 块。和 planner.py 完全一致。"""
    results = []
    if "<tool_call>" not in content or "</tool_call>" not in content:
        return results
    parts = content.split("<tool_call>")[1:]  # skip before first tag
    for part in parts:
        if "</tool_call>" not in part:
            continue
        raw = part.split("</tool_call>")[0].strip()
        try:
            import json5
            parsed = json5.loads(raw)
            results.append((parsed["name"], parsed.get("arguments", {})))
        except Exception:
            try:
                parsed = json.loads(raw)
                results.append((parsed["name"], parsed.get("arguments", {})))
            except Exception:
                continue
    return results


# ── FastAPI App ─────────────────────────────────────────────────

app = FastAPI(title="Planner Server")

# 运行时由 main() 设置
_llm_client: AsyncOpenAI | None = None
_llm_model: str = ""
_stop_at_answer: bool = True

# 会话存储: session_id -> messages list
_sessions: dict[str, list[dict]] = {}


# ── Request / Response Models ───────────────────────────────────

class StartRequest(BaseModel):
    question: str
    stop_at_answer: bool = True


class StartResponse(BaseModel):
    session_id: str


class StepRequest(BaseModel):
    session_id: str
    tool_response: str | None = None


class ToolCall(BaseModel):
    name: str
    arguments: dict


class StepResponse(BaseModel):
    status: str  # "tool_call" | "answer" | "chat" | "none"
    content: str  # LLM 原始输出
    tool_calls: list[ToolCall] = []
    answer: str = ""
    chat: str = ""
    turn: int = 0
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ── API Endpoints ───────────────────────────────────────────────

@app.post("/start", response_model=StartResponse)
async def start_session(req: StartRequest):
    """创建新会话，用 question 初始化 messages。"""
    session_id = str(uuid.uuid4())
    messages = build_messages(req.question)
    _sessions[session_id] = messages
    return StartResponse(session_id=session_id)


@app.post("/step", response_model=StepResponse)
async def step(req: StepRequest):
    """执行一轮 LLM 调用。

    如果 tool_response 不为空，先将其作为 <tool_response> 追加到 messages，
    然后调用 LLM，解析输出，返回 tool_calls / answer / chat。
    """
    messages = _sessions.get(req.session_id)
    if messages is None:
        return StepResponse(status="none", content="Session not found")

    # 如果 client 传回了工具执行结果，追加到 messages
    if req.tool_response is not None:
        tool_response_msg = f"<tool_response>\n{req.tool_response}\n</tool_response>"
        messages.append({"role": "user", "content": tool_response_msg})

    # 判断当前轮次（通过 messages 中 assistant 消息数量）
    turn = sum(1 for m in messages if m["role"] == "assistant") + 1

    # 和 planner.py 一致：首轮用 guided_regex_first，后续用 guided_regex_exec
    guided_regex = GUIDED_REGEX_FIRST if turn == 1 else GUIDED_REGEX_EXEC

    # 和 planner.py 一致的 stop tokens
    stop_tokens = list(STOP_TOKENS)
    if _stop_at_answer:
        stop_tokens.append("<answer>")

    # 调用 LLM，参数和 planner.py 的 run_test 完全一致
    completion = await _llm_client.chat.completions.create(
        model=_llm_model,
        messages=messages,
        temperature=1.0,
        stop=stop_tokens,
        max_tokens=4096,
        extra_body={"guided_regex": guided_regex},
    )

    content = completion.choices[0].message.content or ""
    finish_reason = completion.choices[0].finish_reason or ""
    usage = completion.usage

    # 追加 assistant 回复到 messages
    messages.append({"role": "assistant", "content": content})

    # 解析输出，和 planner.py 的 run_test 逻辑完全一致
    # 1. 闲聊
    if "<chat>" in content and "</chat>" in content:
        chat_msg = content.split("<chat>")[1].split("</chat>")[0]
        return StepResponse(
            status="chat", content=content, chat=chat_msg, turn=turn,
            finish_reason=finish_reason,
            prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
        )

    # 2. 最终回答
    if "<answer>" in content and "</answer>" in content:
        answer = content.split("<answer>")[1].split("</answer>")[0]
        return StepResponse(
            status="answer", content=content, answer=answer, turn=turn,
            finish_reason=finish_reason,
            prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
        )

    # 3. 工具调用
    tool_calls = parse_tool_calls(content)
    if tool_calls:
        return StepResponse(
            status="tool_call", content=content,
            tool_calls=[ToolCall(name=name, arguments=args) for name, args in tool_calls],
            turn=turn, finish_reason=finish_reason,
            prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
        )

    # 4. 无法识别
    return StepResponse(
        status="none", content=content, turn=turn,
        finish_reason=finish_reason,
        prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
    )


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """清理会话。"""
    _sessions.pop(session_id, None)
    return {"ok": True}


# ── 启动入口 ────────────────────────────────────────────────────

def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Planner Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="vLLM server base URL")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B", help="Model name")
    parser.add_argument("--no-stop-at-answer", action="store_true", help="Do not add <answer> to stop tokens")
    args = parser.parse_args()

    global _llm_client, _llm_model, _stop_at_answer
    _llm_client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")
    _llm_model = args.model
    _stop_at_answer = not args.no_stop_at_answer

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
