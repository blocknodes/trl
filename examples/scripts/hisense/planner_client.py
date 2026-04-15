"""
Planner Client: 调用 planner_server 获取 LLM 的 tool_call 指令，
执行真实的搜索工具（inner_search -> KBP, web_search -> Bocha），
将结果回传 server，循环直到 LLM 输出 <answer> 或 <chat>。

用法:
    python examples/scripts/hisense/planner_client.py \
        --server-url http://localhost:9000 \
        --question "海信冰箱"
"""

import argparse
import asyncio
import json
import sys

import aiohttp

# ── 海信 KBP 检索 API 配置（和 planner.py 完全一致）─────────────
KBP_API_URL = "https://inner-apisix-test.hisense.com/kbp-test/openapi/kbp/mix/retrieval"
KBP_USER_KEY = "qimfvt7lwtqeyangfl259vjg8fzdhh5l"
KBP_API_KEY = "a5198155-607d-4e4b-b92c-f04259964c93"

# ── Bocha Web Search API 配置（和 planner.py 完全一致）──────────
BOCHA_API_URL = "https://api.bochaai.com/v1/web-search"
BOCHA_API_KEY = "sk-2ad764f6f07041ee99eca00816403d2b"


# ── 工具实现：和 planner.py 完全一致 ────────────────────────────

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


async def real_tool_result(tool_name: str, tool_args: dict, top_k: int = 10) -> str:
    """调用真实 API 获取搜索结果。"""
    if tool_name == "inner_search":
        queries = tool_args.get("query", [])
        if isinstance(queries, str):
            queries = [queries]
        tasks = [kbp_search(q, top_k=top_k) for q in queries]
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


# ── 主循环：调用 server，执行工具，回传结果 ─────────────────────

async def run(server_url: str, question: str, max_turns: int, top_k: int = 10, debug: bool = False):
    """循环调用 planner_server，执行工具，直到得到 answer 或 chat。"""
    print("=" * 70)
    print(f"Server: {server_url}")
    print(f"Question: {question}")
    print(f"Max turns: {max_turns}")
    print("=" * 70)

    async with aiohttp.ClientSession() as http:
        # 1. 创建会话
        async with http.post(f"{server_url}/start", json={"question": question}) as resp:
            start_data = await resp.json()
        session_id = start_data["session_id"]
        print(f"[Session created]: {session_id}")

        try:
            tool_response = None

            for turn in range(1, max_turns + 1):
                print(f"\n{'─' * 50}")
                print(f"Turn {turn}: Calling server /step ...")
                print(f"{'─' * 50}")

                # 2. 调用 server /step
                step_payload = {"session_id": session_id, "max_turns": max_turns, "top_k": top_k}
                if tool_response is not None:
                    step_payload["tool_response"] = tool_response

                async with http.post(f"{server_url}/step", json=step_payload,
                                     timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    step_data = await resp.json()

                status = step_data["status"]
                content = step_data["content"]
                step_turn = step_data.get("turn", turn)

                print(f"[Status]: {status}")
                print(f"[Finish reason]: {step_data.get('finish_reason', '')}")
                print(f"[Usage]: prompt_tokens={step_data.get('prompt_tokens', 0)}, "
                      f"completion_tokens={step_data.get('completion_tokens', 0)}")
                print(f"[Response]:\n{content[:2000]}{'... (truncated)' if len(content) > 2000 else ''}")

                # 3. 根据 status 处理
                if status == "chat":
                    print(f"\n[Chat response]: {step_data.get('chat', '')}")
                    return

                if status == "answer":
                    print(f"\n{'=' * 70}")
                    print(f"Agent finished with answer:\n{step_data.get('answer', '')}")
                    merged = step_data.get("merged_results", {})
                    if merged:
                        print(f"\n[Merged results by tool type]:")
                        print(json.dumps(merged, ensure_ascii=False, indent=2))
                    print(f"{'=' * 70}")
                    return

                if status == "tool_call":
                    tool_calls = step_data.get("tool_calls", [])
                    all_results = []

                    for tc in tool_calls:
                        tool_name = tc["name"]
                        tool_args = tc["arguments"]
                        print(f"\n[Tool call detected]: {tool_name}")
                        print(f"[Tool args]: {json.dumps(tool_args, ensure_ascii=False, indent=2)}")

                    # 并发执行所有 tool calls
                    tasks = [real_tool_result(tc["name"], tc["arguments"], top_k=top_k) for tc in tool_calls]
                    results = await asyncio.gather(*tasks)

                    for tc, result in zip(tool_calls, results):
                        if debug:
                            print(f"\n[DEBUG TOOL RESPONSE ({tc['name']})]:\n{result}")
                        else:
                            print(f"[Tool response ({tc['name']})]: {result[:500]}{'...' if len(result) > 500 else ''}")
                        all_results.append(result)

                    # 和 planner.py 一致的格式拼接
                    tool_response = "\n=======\n".join(all_results)
                    continue

                # status == "none"
                print("[No tool call or answer detected, quiting...]")
                sys.exit(0)

            print(f"\n{'=' * 70}")
            print(f"Reached max turns ({max_turns}) without final answer.")
            print(f"{'=' * 70}")

        finally:
            # 清理会话
            try:
                await http.delete(f"{server_url}/session/{session_id}")
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Planner Client")
    parser.add_argument("--server-url", default="http://localhost:9000", help="Planner server URL")
    parser.add_argument("--question", default="海信冰箱", help="Test question")
    parser.add_argument("--max-turns", type=int, default=5, help="Max conversation turns")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k for inner KB (KBP) retrieval")
    parser.add_argument("--debug", action="store_true", help="Print full tool responses without truncation")
    args = parser.parse_args()

    asyncio.run(run(args.server_url, args.question, args.max_turns, top_k=args.top_k, debug=args.debug))


if __name__ == "__main__":
    main()
