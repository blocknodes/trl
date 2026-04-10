"""
单脚本测试：用 tongyi_deepresearch 的 prompt 验证 vLLM 模型的输入输出。

用法:
    python examples/search_agent/test_vllm_search_agent.py \
        --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen3-1.7B \
        --question "What is the population of Tokyo in 2024?"

前置条件: 先用 vLLM 起好模型服务，例如:
    vllm serve Qwen/Qwen3-1.7B --port 8000
"""

import argparse
import asyncio
import datetime
import json
import sys

import aiohttp
from openai import AsyncOpenAI

# ── 海信 KBP 检索 API 配置 ──────────────────────────────────────
KBP_API_URL = "https://inner-apisix-test.hisense.com/kbp-test/openapi/kbp/mix/retrieval"
KBP_USER_KEY = "qimfvt7lwtqeyangfl259vjg8fzdhh5l"
KBP_API_KEY = "a5198155-607d-4e4b-b92c-f04259964c93"

# ── Bocha Web Search API 配置 ───────────────────────────────────
BOCHA_API_URL = "https://api.bochaai.com/v1/web-search"
BOCHA_API_KEY = "sk-2ad764f6f07041ee99eca00816403d2b"

# ── 直接内联 prompt，和项目保持一致 ──────────────────────────────
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


def build_messages(question: str) -> list[dict]:
    """构造和 react_agent.py 一致的 messages 结构。"""
    cur_date = datetime.date.today().strftime("%Y-%m-%d")
    return [
        {"role": "system", "content": SYSTEM_PROMPT + cur_date},
        {"role": "user", "content": question},
    ]


async def kbp_search(query: str, top_k: int = 10) -> str:
    """调用海信 KBP 检索 API 进行搜索。"""
    payload = {
        "query": query,
        "retrieval_setting": {
            "top_k": top_k,
            "score_threshold": 0,
            "search_mode": "hybrid",
            "search_strategy": "broad",
        },
        "tracingModel": False,
    }
    headers = {
        "Content-Type": "application/json",
        "api-key": KBP_API_KEY,
    }
    url = f"{KBP_API_URL}?user_key={KBP_USER_KEY}"

    async with aiohttp.ClientSession() as session:
        for attempt in range(3):
            try:
                async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    text = await resp.text()
                    try:
                        data = json.loads(text)
                    except Exception:
                        return f"[KBP Search] Failed to parse response for '{query}': {text[:200]}"

                    # 解析 KBP API 返回结果，组装为搜索结果格式
                    records = data.get("records", data.get("data", []))
                    if isinstance(data, dict) and not records:
                        # 尝试兼容不同返回结构
                        for key in data:
                            if isinstance(data[key], list):
                                records = data[key]
                                break

                    if not records:
                        return f"No results found for query: '{query}'. Raw response keys: {list(data.keys()) if isinstance(data, dict) else type(data).__name__}"

                    web_snippets = []
                    for idx, item in enumerate(records, start=1):
                        title = item.get("title", item.get("metadata", {}).get("title", f"Result {idx}"))
                        content = item.get("content", item.get("text", item.get("segment", "")))
                        score = item.get("score", "")
                        source = item.get("metadata", {}).get("source", item.get("source", ""))

                        snippet = f"{idx}. [{title}]"
                        if source:
                            snippet += f"({source})"
                        if score:
                            snippet += f"\nScore: {score}"
                        if content:
                            # 截取前 500 字符避免过长
                            snippet += f"\n{content[:500]}"
                        web_snippets.append(snippet)

                    return (
                        f"A KBP search for '{query}' found {len(web_snippets)} results:\n\n## Search Results\n"
                        + "\n\n".join(web_snippets)
                    )
            except asyncio.TimeoutError:
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
                return f"[KBP Search] Timeout for query: '{query}'"
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(0.5)
                    continue
                return f"[KBP Search] Error for query '{query}': {e}"

    return f"[KBP Search] All retries failed for query: '{query}'"


async def bocha_web_search(query: str, count: int = 50) -> str:
    """调用 Bocha AI Web Search API 进行互联网搜索。"""
    payload = {
        "query": query,
        "summary": True,
        "freshness": "noLimit",
        "count": count,
    }
    headers = {
        "Authorization": f"Bearer {BOCHA_API_KEY}",
        "Content-Type": "application/json",
        "Accept-Encoding": "gzip, deflate",
    }

    async with aiohttp.ClientSession() as session:
        for attempt in range(3):
            try:
                async with session.post(
                    BOCHA_API_URL, json=payload, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    text = await resp.text()
                    try:
                        data = json.loads(text)
                    except Exception:
                        return f"[Bocha Search] Failed to parse response for '{query}': {text[:200]}"

                    # 解析 Bocha API 返回结果
                    web_pages = data.get("data", {}).get("webPages", {}).get("value", [])
                    if not web_pages:
                        # 尝试兼容其他返回结构
                        if isinstance(data, dict) and "webPages" in data:
                            web_pages = data["webPages"].get("value", [])

                    if not web_pages:
                        summary = data.get("data", {}).get("summary", "")
                        if summary:
                            return f"Bocha search for '{query}':\n\n{summary}"
                        return f"No results found for query: '{query}'."

                    snippets = []
                    for idx, item in enumerate(web_pages, start=1):
                        name = item.get("name", f"Result {idx}")
                        url = item.get("url", "")
                        snippet = item.get("snippet", "")
                        site_name = item.get("siteName", "")

                        entry = f"{idx}. [{name}]({url})"
                        if site_name:
                            entry += f" - {site_name}"
                        if snippet:
                            entry += f"\n{snippet[:500]}"
                        snippets.append(entry)

                    result = (
                        f"Bocha web search for '{query}' found {len(snippets)} results:\n\n"
                        + "\n\n".join(snippets)
                    )

                    # 附加 summary（如果有）
                    summary = data.get("data", {}).get("summary", "")
                    if summary:
                        result += f"\n\n## Summary\n{summary}"

                    return result
            except asyncio.TimeoutError:
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
                return f"[Bocha Search] Timeout for query: '{query}'"
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(0.5)
                    continue
                return f"[Bocha Search] Error for query '{query}': {e}"

    return f"[Bocha Search] All retries failed for query: '{query}'"


async def real_tool_result(tool_name: str, tool_args: dict) -> str:
    """调用真实 API 获取搜索结果。"""
    if tool_name == "inner_search":
        queries = tool_args.get("query", [])
        if isinstance(queries, str):
            queries = [queries]
        tasks = [kbp_search(q) for q in queries]
        responses = await asyncio.gather(*tasks)
        return "\n=======\n".join(responses)
    elif tool_name == "web_search":
        queries = tool_args.get("query", [])
        if isinstance(queries, str):
            queries = [queries]
        tasks = [bocha_web_search(q) for q in queries]
        responses = await asyncio.gather(*tasks)
        return "\n=======\n".join(responses)
    else:
        return f"Error: Tool {tool_name} not found"


def parse_tool_calls(content: str) -> list[tuple[str, dict]]:
    """从模型输出中解析所有 <tool_call>...</tool_call> 块。"""
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


async def run_test(base_url: str, model: str, question: str, max_turns: int, debug: bool = False, stop_at_answer: bool = True):
    """直接执行循环：每轮根据结果决定下一步，无需先 plan。"""
    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
    messages = build_messages(question)

    # 和 react_agent.py 一致的 stop tokens
    stop_tokens = ["\n<tool_response>", "<tool_response>"]
    if stop_at_answer:
        stop_tokens.append("<answer>")

    # 约束解码 regex（首轮允许 chat，后续只允许 tool_call 或 answer）
    guided_regex_first = r"(<tool_call>[\s\S]*</tool_call>|<chat>[\s\S]*</chat>)"
    guided_regex_exec = r"(<answer>[\s\S]*</answer>|<tool_call>[\s\S]*</tool_call>)"

    print("=" * 70)
    print(f"Model: {model}")
    print(f"Question: {question}")
    print(f"Max turns: {max_turns}")
    print("=" * 70)

    for turn in range(1, max_turns + 1):
        print(f"\n{'─' * 50}")
        print(f"Turn {turn}: Calling LLM...")
        print(f"{'─' * 50}")
        print(f"[Input messages count]: {len(messages)}")

        guided_regex = guided_regex_first if turn == 1 else guided_regex_exec

        request_body = {
            "model": model,
            "messages": messages,
            "temperature": 1.0,
            "stop": stop_tokens,
            "max_tokens": 4096,
            "extra_body": {"guided_regex": guided_regex},
        }
        if debug:
            print(f"\n[DEBUG REQUEST]:\n{json.dumps(request_body, ensure_ascii=False, indent=2)}")

        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1.0,
            stop=stop_tokens,
            max_tokens=4096,
            extra_body={"guided_regex": guided_regex},
        )

        content = completion.choices[0].message.content or ""
        finish_reason = completion.choices[0].finish_reason
        usage = completion.usage

        if debug:
            print(f"\n[DEBUG RESPONSE]:\n{completion.model_dump_json(indent=2)}")

        print(f"[Finish reason]: {finish_reason}")
        print(f"[Usage]: prompt_tokens={usage.prompt_tokens}, completion_tokens={usage.completion_tokens}")
        print(f"[Response]:\n{content[:2000]}{'... (truncated)' if len(content) > 2000 else ''}")

        messages.append({"role": "assistant", "content": content})

        # 如果是闲聊，直接返回
        if "<chat>" in content and "</chat>" in content:
            chat_msg = content.split("<chat>")[1].split("</chat>")[0]
            print(f"\n[Chat response]: {chat_msg}")
            return

        # 检查是否有 <answer>
        if "<answer>" in content and "</answer>" in content:
            answer = content.split("<answer>")[1].split("</answer>")[0]
            print(f"\n{'=' * 70}")
            print(f"Agent finished with answer:\n{answer}")
            print(f"{'=' * 70}")
            return

        # 检查是否有 tool_call
        tool_calls = parse_tool_calls(content)
        if tool_calls:
            # 并发执行所有 tool calls
            all_results = []
            for tool_name, tool_args in tool_calls:
                print(f"\n[Tool call detected]: {tool_name}")
                print(f"[Tool args]: {json.dumps(tool_args, ensure_ascii=False, indent=2)}")

            tasks = [real_tool_result(name, args) for name, args in tool_calls]
            results = await asyncio.gather(*tasks)

            for (tool_name, _), result in zip(tool_calls, results):
                if debug:
                    print(f"\n[DEBUG TOOL RESPONSE ({tool_name})]:\n{result}")
                else:
                    print(f"[Tool response ({tool_name})]: {result[:500]}{'...' if len(result) > 500 else ''}")
                all_results.append(result)

            combined = "\n=======\n".join(all_results)
            tool_response = f"<tool_response>\n{combined}\n</tool_response>"
            messages.append({"role": "user", "content": tool_response})
        else:
            print("[No tool call or answer detected, quiting...]")
            sys.exit(0)

    print(f"\n{'=' * 70}")
    print(f"Reached max turns ({max_turns}) without final answer.")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Test vLLM model with search agent prompt")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="vLLM server base URL")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B", help="Model name")
    parser.add_argument("--question", default="What is the population of Tokyo in 2024?", help="Test question")
    parser.add_argument("--max-turns", type=int, default=5, help="Max conversation turns")
    parser.add_argument("--debug", action="store_true", help="Print full LLM request/response without truncation")
    parser.add_argument("--no-stop-at-answer", action="store_true", help="Do not add <answer> to stop tokens")
    args = parser.parse_args()

    asyncio.run(run_test(args.base_url, args.model, args.question, args.max_turns, args.debug, not args.no_stop_at_answer))


if __name__ == "__main__":
    main()
