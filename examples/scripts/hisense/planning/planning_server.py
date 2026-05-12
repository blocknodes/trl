"""
Planning Server: FastAPI 服务，实现 planning 协议。

支持三种 thinking 模式:
- simple (默认): 两轮 planning，query rewrite + 规则判断
- dynamic: React 式循环，搜索 → 总结+判断 → 决定是否继续
- deep: 多步研究计划 + 逐步搜索总结 + 最终答案生成

用法:
    python examples/scripts/hisense/planning_server.py \\
        --base-url http://localhost:8078/v1 \\
        --model qwen4b \\
        --port 9100
"""

import argparse
import json
import logging
import os

from fastapi import FastAPI
from openai import AsyncOpenAI

from helpers import VERBOSE, PlanningRequest, PlanningResponse, RetrievalSetting
from workflow import trim_history

logger = logging.getLogger("planning_server")

app = FastAPI(title="Planning Server")


@app.get("/health")
async def health():
    return {"status": "ok"}

_llm_client: AsyncOpenAI | None = None
_llm_model: str = ""
_planner_client: AsyncOpenAI | None = None
_planner_model: str = ""
_ts_client: AsyncOpenAI | None = None
_ts_model: str = ""
_ts_threshold: float = 0


@app.post("/planner", response_model=PlanningResponse, response_model_exclude_none=True)
async def planner(req: PlanningRequest):
    """单一 planning 端点。"""
    logger.log(VERBOSE, "Request input:\n%s", json.dumps(req.model_dump(), ensure_ascii=False, indent=2, default=str))
    rs = req.retrieval_setting or RetrievalSetting()
    logger.debug("Request summary: turn=%d, query=%r, tool_hub=%s, tool_hub_optional=%s, "
                 "score_threshold=%.4f, top_k=%d, max_top_k=%d, thinking=%s",
                 req.turn, req.query, req.tool_hub, req.tool_hub_optional,
                 rs.score_threshold, rs.top_k,
                 req.max_top_k, req.thinking)

    _handlers = {
        "deep": ("deep_thinking", "handle_deep_thinking"),
        "dynamic": ("dynamic_thinking", "handle_dynamic_thinking"),
        "simple": ("simple_thinking", "handle_simple_thinking"),
    }
    mode = req.thinking if req.thinking in _handlers else "simple"
    mod_name, func_name = _handlers[mode]

    try:
        import importlib
        handler = getattr(importlib.import_module(mod_name), func_name)
    except (ImportError, AttributeError):
        logger.error("Thinking mode %r unavailable (module %s not found), falling back to stop", mode, mod_name)
        return PlanningResponse(status="stop", turn=req.turn, final={})

    if mode in ("deep", "dynamic"):
        resp = await handler(req, _llm_client, _llm_model, _planner_client, _planner_model, trim_history)
    else:
        resp = await handler(req, _llm_client, _llm_model, _ts_client, _ts_model, _ts_threshold)

    logger.log(VERBOSE, "Response output:\n%s", json.dumps(resp.model_dump(), ensure_ascii=False, indent=2, default=str))
    current_summary = ""
    if resp.current:
        parts = [f"({c.sub_query} -> [{c.tool_use}] topk={c.topk})" for c in resp.current]
        current_summary = ", ".join(parts)
    final_tools = list(resp.final.keys()) if resp.final else []
    logger.debug("Response summary: status=%s, turn=%d, current=[%s], final_tools=%s",
                  resp.status, resp.turn, current_summary, final_tools)
    return resp


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Planning Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--base-url", default="http://localhost:8078/v1", help="vLLM server base URL")
    parser.add_argument("--model", default="qwen4b", help="Model name")
    parser.add_argument("--planner-base-url", default="https://aix-backup.hismarttv.com/v1",
                        help="Planner LLM base URL for deep/dynamic thinking")
    parser.add_argument("--planner-model", default="deepseek-v3",
                        help="Planner model name for deep/dynamic thinking")
    parser.add_argument("--planner-api-key", default="EMPTY",
                        help="Planner LLM API key")
    parser.add_argument("--session-ttl", type=int, default=600,
                        help="Deep thinking session TTL in seconds (default: 600)")
    parser.add_argument("--tool-select-base-url", default="http://localhost:8071/v1",
                        help="Tool selection LLM base URL (default: http://localhost:8071/v1)")
    parser.add_argument("--tool-select-model", default="qwen4b",
                        help="Tool selection model name (default: qwen4b)")
    parser.add_argument("--tool-select-api-key", default="EMPTY",
                        help="Tool selection LLM API key")
    parser.add_argument("--tool-selection-threshold", type=float, default=0,
                        help="Default threshold for model-based tool selection (default: 0)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["VERBOSE", "DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Console logging level (default: INFO)")
    parser.add_argument("--file-log-level", default="DEBUG",
                        choices=["VERBOSE", "DEBUG", "INFO", "WARNING", "ERROR"],
                        help="File logging level (default: DEBUG)")
    args = parser.parse_args()

    import datetime as _dt
    _log_dir = os.environ.get("LOG_DIR", "log")
    os.makedirs(_log_dir, exist_ok=True)
    _log_file = os.path.join(_log_dir, f"planning_server_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    _log_fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    _console = logging.StreamHandler()
    _console.setLevel(VERBOSE if args.log_level == "VERBOSE" else getattr(logging, args.log_level))
    _fh = logging.FileHandler(_log_file)
    _fh.setLevel(VERBOSE if args.file_log_level == "VERBOSE" else getattr(logging, args.file_log_level))
    logging.basicConfig(level=VERBOSE, format=_log_fmt, handlers=[_fh, _console])
    logger.info("Log directory: %s", os.path.abspath(_log_dir))

    global _llm_client, _llm_model, _planner_client, _planner_model, _ts_client, _ts_model, _ts_threshold

    llm_base_url = os.environ.get("LLM_BASE_URL", args.base_url)
    llm_model = os.environ.get("LLM_MODEL", args.model)
    planner_base_url = os.environ.get("PLANNER_BASE_URL", args.planner_base_url)
    planner_model = os.environ.get("PLANNER_MODEL", args.planner_model)
    planner_api_key = os.environ.get("PLANNER_API_KEY", args.planner_api_key)
    ts_base_url = os.environ.get("TOOL_SELECT_BASE_URL", args.tool_select_base_url)
    ts_model = os.environ.get("TOOL_SELECT_MODEL", args.tool_select_model)
    ts_api_key = os.environ.get("TOOL_SELECT_API_KEY", args.tool_select_api_key)

    _llm_client = AsyncOpenAI(base_url=llm_base_url, api_key="EMPTY")
    _llm_model = llm_model
    _planner_model = planner_model
    _planner_client = AsyncOpenAI(base_url=planner_base_url, api_key=planner_api_key)
    _ts_client = AsyncOpenAI(base_url=ts_base_url, api_key=ts_api_key)
    _ts_model = ts_model
    _ts_threshold = args.tool_selection_threshold
    try:
        from deep_thinking import set_session_ttl
        set_session_ttl(args.session_ttl)
    except ImportError:
        logger.warning("deep_thinking module not found, session TTL not set")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
