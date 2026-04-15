"""
Planning Client: 调用 planning_server 的 /planner 端点，
执行真实的搜索工具（es -> KBP, web -> Bocha, graph -> 暂未实现），
将结果回传 server，循环直到 server 返回 status="stop"。

用法:
    python examples/scripts/hisense/planning_client.py \
        --server-url http://localhost:9100 \
        --question "海信冰箱" \
        --top-k 1 \
        --score-threshold 0.8
"""

import argparse
import asyncio
import json
import logging
import sys

import aiohttp

logger = logging.getLogger("planning_client")

# 自定义 VERBOSE 级别 (低于 DEBUG)，用于打印完整输入输出内容
VERBOSE = 5
logging.addLevelName(VERBOSE, "VERBOSE")

# ── 海信 KBP 检索 API 配置 ──────────────────────────────────────
KBP_API_URL = "https://inner-apisix-test.hisense.com/kbp-test/openapi/kbp/mix/retrieval"
KBP_USER_KEY = "qimfvt7lwtqeyangfl259vjg8fzdhh5l"
KBP_API_KEY = "a5198155-607d-4e4b-b92c-f04259964c93"

# ── Bocha Web Search API 配置 ───────────────────────────────────
BOCHA_API_URL = "https://api.bochaai.com/v1/web-search"
BOCHA_API_KEY = "sk-2ad764f6f07041ee99eca00816403d2b"

# ── 小素 (Xiaosu) Web Search API 配置 ──────────────────────────
XIAOSU_API_URL = "https://inner-apisix.hisense.com/hiagent/v1/chat-messages"
XIAOSU_USER_KEY = "bt5ix3u9szdexwlvpmsxcavl3hnnyytf"
XIAOSU_API_KEY = "app-YPgOiXUZm9fFIb2RksDCRoiS"

# 运行时由 main() 设置
_web_search_backend: str = "xiaosu"  # "bocha" or "xiaosu"


# ── 工具实现 ────────────────────────────────────────────────────

async def kbp_search(query: str, top_k: int = 10) -> list[dict]:
    """调用海信 KBP 检索 API，返回结构化结果列表。"""
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
    headers = {"Content-Type": "application/json", "api-key": KBP_API_KEY}
    url = f"{KBP_API_URL}?user_key={KBP_USER_KEY}"

    async with aiohttp.ClientSession() as session:
        for attempt in range(3):
            try:
                async with session.post(url, json=payload, headers=headers,
                                        timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    text = await resp.text()
                    try:
                        data = json.loads(text)
                    except Exception:
                        return []

                    records = data.get("records", data.get("data", []))
                    if isinstance(data, dict) and not records:
                        for key in data:
                            if isinstance(data[key], list):
                                records = data[key]
                                break

                    results = []
                    for item in records:
                        results.append({
                            "file_name": item.get("metadata", {}).get("source", ""),
                            "title": item.get("title", item.get("metadata", {}).get("title", "")),
                            "content": item.get("content", item.get("text", item.get("segment", "")))[:500],
                            "score": item.get("score", 0),
                            "category_path": item.get("metadata", {}).get("category_path", ""),
                        })
                    return results
            except asyncio.TimeoutError:
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
                return []
            except Exception:
                if attempt < 2:
                    await asyncio.sleep(0.5)
                    continue
                return []
    return []


async def bocha_web_search(query: str, count: int = 50) -> list[dict]:
    """调用 Bocha AI Web Search API，返回结构化结果列表。"""
    payload = {"query": query, "summary": True, "freshness": "noLimit", "count": count}
    headers = {
        "Authorization": f"Bearer {BOCHA_API_KEY}",
        "Content-Type": "application/json",
        "Accept-Encoding": "gzip, deflate",
    }

    async with aiohttp.ClientSession() as session:
        for attempt in range(3):
            try:
                async with session.post(BOCHA_API_URL, json=payload, headers=headers,
                                        timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    text = await resp.text()
                    try:
                        data = json.loads(text)
                    except Exception:
                        return []

                    web_pages = data.get("data", {}).get("webPages", {}).get("value", [])
                    if not web_pages and isinstance(data, dict) and "webPages" in data:
                        web_pages = data["webPages"].get("value", [])

                    results = []
                    for item in web_pages:
                        results.append({
                            "title": item.get("name", ""),
                            "content": item.get("snippet", "")[:500],
                        })
                    return results
            except asyncio.TimeoutError:
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
                return []
            except Exception:
                if attempt < 2:
                    await asyncio.sleep(0.5)
                    continue
                return []
    return []


async def xiaosu_web_search(query: str, topk: int = 1) -> list[dict]:
    """调用小素 (Xiaosu) Web Search API，返回结构化结果列表。"""
    payload = {
        "user": "planning_client",
        "inputs": {"topk": topk},
        "query": query,
        "response_mode": "blocking",
    }
    headers = {
        "Authorization": f"Bearer {XIAOSU_API_KEY}",
        "Content-Type": "application/json",
    }
    url = f"{XIAOSU_API_URL}?user_key={XIAOSU_USER_KEY}"

    async with aiohttp.ClientSession() as session:
        for attempt in range(3):
            try:
                async with session.post(url, json=payload, headers=headers,
                                        timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    text = await resp.text()
                    try:
                        data = json.loads(text)
                    except Exception:
                        return []

                    # 解析小素返回: answer 字段是一个 JSON 字符串
                    answer_str = data.get("answer", "")
                    try:
                        answer_data = json.loads(answer_str)
                    except (json.JSONDecodeError, TypeError):
                        answer_data = {}

                    content_datas = answer_data.get("data", {}).get("contentDatas", [])
                    results = []
                    for item in content_datas:
                        results.append({
                            "title": item.get("webTitle", ""),
                            "content": item.get("content", "")[:500],
                            "source": item.get("source", ""),
                            "webUrl": item.get("webUrl", ""),
                            "time": item.get("time", ""),
                        })
                    return results
            except asyncio.TimeoutError:
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
                return []
            except Exception:
                if attempt < 2:
                    await asyncio.sleep(0.5)
                    continue
                return []
    return []


async def web_search(query: str, topk: int = 10) -> list[dict]:
    """根据配置的后端调用对应的 web 搜索。"""
    if _web_search_backend == "xiaosu":
        return await xiaosu_web_search(query, topk=topk)
    return await bocha_web_search(query, count=topk)


async def execute_search(sub_query: str, tool_use: str, topk: int) -> dict:
    """对一个 sub_query 执行指定工具的搜索，返回 planning 协议格式的结果。"""
    tools = [t.strip() for t in tool_use.split(",") if t.strip()]
    result: dict[str, list[dict]] = {}

    tasks = []
    tool_names = []
    for tool in tools:
        if tool == "es":
            tasks.append(kbp_search(sub_query, top_k=topk))
            tool_names.append("es")
        elif tool == "web":
            tasks.append(web_search(sub_query, topk=topk))
            tool_names.append("web")
        elif tool == "graph":
            # graph 暂未实现，直接置空
            result["graph"] = []

    if tasks:
        responses = await asyncio.gather(*tasks)
        for name, resp in zip(tool_names, responses):
            result[name] = resp

    return {
        "sub_query": sub_query,
        "tool_use": tool_use,
        "topk": topk,
        "result": result,
    }


# ── 主循环 ──────────────────────────────────────────────────────

async def run(server_url: str, question: str, max_turn: int, top_k: int,
              score_threshold: float, max_top_k: int, tool_hub: str,
              tool_hub_optional: str, debug: bool = False):
    """循环调用 planning_server /planner，执行搜索，直到 status="stop"。"""
    logger.info("Server: %s", server_url)
    logger.info("Question: %s", question)
    logger.info("Max turns: %d, top_k: %d, threshold: %.2f", max_turn, top_k, score_threshold)
    logger.info("Tools: %s | Optional: %s", tool_hub, tool_hub_optional)

    history: dict = {}

    async with aiohttp.ClientSession() as http:
        for turn in range(1, max_turn + 1):
            logger.info("Turn %d: Calling /planner ...", turn)

            # 构造请求
            payload = {
                "query": question,
                "retrieval_setting": {
                    "top_k": top_k,
                    "score_threshold": score_threshold,
                    "search_mode": "hybrid",
                    "search_strategy": "precise",
                },
                "turn": turn,
                "max_turn": max_turn,
                "max_top_k": max_top_k,
                "max_context_size": 3,
                "tool_hub": tool_hub,
                "tool_hub_optional": tool_hub_optional,
            }
            if history:
                payload["history"] = history

            logger.log(VERBOSE, "Request payload:\n%s", json.dumps(payload, ensure_ascii=False, indent=2))
            logger.debug("Request summary: turn=%d, query=%r, tool_hub=%s", turn, question, tool_hub)

            async with http.post(f"{server_url}/planner", json=payload,
                                 timeout=aiohttp.ClientTimeout(total=120)) as resp:
                resp_data = await resp.json()

            logger.log(VERBOSE, "Response data:\n%s", json.dumps(resp_data, ensure_ascii=False, indent=2))
            logger.debug("Response summary: status=%s, current=%d items",
                         resp_data.get("status"), len(resp_data.get("current") or []))

            status = resp_data.get("status", "stop")
            is_off_topic = resp_data.get("is_off_topic", False)
            current = resp_data.get("current", [])
            final = resp_data.get("final")

            logger.info("Turn %d | status=%s, is_off_topic=%s", turn, status, is_off_topic)

            if is_off_topic:
                logger.info("Off-topic query, skipping retrieval")
                return

            if status == "stop":
                logger.info("Planning complete")
                if final:
                    logger.info("Final results:\n%s", json.dumps(final, ensure_ascii=False, indent=2))
                else:
                    logger.info("No results")
                return

            # status == "running"，执行搜索
            logger.info("Sub-queries to search: %d", len(current))
            for item in current:
                logger.info("  sub_query=%r tools=%s topk=%d", item["sub_query"], item["tool_use"], item["topk"])

            # 并发执行所有 sub_query 的搜索
            tasks = [execute_search(item["sub_query"], item["tool_use"], item["topk"]) for item in current]
            retrieval_contents = await asyncio.gather(*tasks)

            # 打印搜索结果摘要
            for rc in retrieval_contents:
                sq = rc["sub_query"]
                for tool_type, records in rc["result"].items():
                    scores = [r.get("score", "N/A") for r in records]
                    logger.info("  [%s] %r -> %d results, scores=%s", tool_type, sq, len(records), scores)

            if debug:
                logger.log(VERBOSE, "retrieval_contents:\n%s", json.dumps(list(retrieval_contents), ensure_ascii=False, indent=2))

            # 将本轮结果写入 history
            history[f"turn_{turn}"] = {"retrieval_contents": list(retrieval_contents)}

    logger.warning("Reached max turns (%d) without stop", max_turn)


def main():
    parser = argparse.ArgumentParser(description="Planning Client")
    parser.add_argument("--server-url", default="http://localhost:9100", help="Planning server URL")
    parser.add_argument("--question", default="海信冰箱", help="User query")
    parser.add_argument("--max-turn", type=int, default=3, help="Max planning turns")
    parser.add_argument("--top-k", type=int, default=1, help="Required top_k for retrieval_setting")
    parser.add_argument("--score-threshold", type=float, default=0.8, help="Score threshold")
    parser.add_argument("--max-top-k", type=int, default=3, help="Max top_k per sub-query")
    parser.add_argument("--tool-hub", default="es,graph,web", help="Available tools")
    parser.add_argument("--tool-hub-optional", default="", help="Optional tools")
    parser.add_argument("--debug", action="store_true", help="Print full retrieval results")
    parser.add_argument("--web-backend", default="xiaosu", choices=["bocha", "xiaosu"],
                        help="Web search backend: bocha or xiaosu (default: bocha)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["VERBOSE", "DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Console logging level (default: INFO)")
    parser.add_argument("--file-log-level", default="DEBUG",
                        choices=["VERBOSE", "DEBUG", "INFO", "WARNING", "ERROR"],
                        help="File logging level (default: DEBUG)")
    args = parser.parse_args()

    import datetime as _dt
    _log_file = f"planning_client_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    _log_fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    _console = logging.StreamHandler()
    _console.setLevel(VERBOSE if args.log_level == "VERBOSE" else getattr(logging, args.log_level))
    _fh = logging.FileHandler(_log_file)
    _fh.setLevel(VERBOSE if args.file_log_level == "VERBOSE" else getattr(logging, args.file_log_level))
    logging.basicConfig(level=VERBOSE, format=_log_fmt, handlers=[_fh, _console])

    global _web_search_backend
    _web_search_backend = args.web_backend

    asyncio.run(run(
        server_url=args.server_url,
        question=args.question,
        max_turn=args.max_turn,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        max_top_k=args.max_top_k,
        tool_hub=args.tool_hub,
        tool_hub_optional=args.tool_hub_optional,
        debug=args.debug,
    ))


if __name__ == "__main__":
    main()
