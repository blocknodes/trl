# Planning Module

An LLM-based multi-turn retrieval planning system supporting query decomposition, tool selection, and multi-step reasoning. Uses a Client-Server architecture where the Server handles planning decisions and the Client executes actual retrieval.

## Architecture

```
┌──────────────────┐         ┌──────────────────┐         ┌─────────────┐
│  Planning Client │ ──────► │  Planning Server │ ──────► │   LLM API   │
│  (retrieval)     │ ◄────── │  (planning)      │ ◄────── │  (inference) │
└──────────────────┘         └──────────────────┘         └─────────────┘
        │
        ├──► KBP (Knowledge Base Retrieval)
        ├──► Web Search (Bocha / Xiaosu)
        └──► Graph (Structured Retrieval)
```

## File Overview

| File | Description |
|------|-------------|
| `planning_server.py` | FastAPI server exposing `/planner` and `/health` endpoints |
| `planning_client.py` | Client that loops calling the server and executes actual retrieval |
| `helpers.py` | Data models (Request/Response) and utility functions |
| `prompts.py` | All LLM prompt templates |
| `query_rewrite.py` | Query decomposition: splits user query into multiple sub_queries |
| `simple_thinking.py` | Simple mode: two-round planning (rewrite + rule-based supplement) |
| `dynamic_thinking.py` | Dynamic mode: React-style loop, LLM decides whether to continue after each search |
| `deep_thinking.py` | Deep mode: pre-generates a multi-step research plan, searches and summarizes step by step |
| `tool_selection.py` | Tool selection: LLM determines whether a query suits graph or es |
| `workflow.py` | Multi-turn history management, result aggregation, score evaluation |

## Thinking Modes

### Simple (default)

Two-round planning protocol:
1. Round 1: LLM decomposes the query into sub_queries and assigns retrieval tools
2. Round 2: Evaluates scores against threshold, supplements with additional tools or returns results

Best for straightforward queries with fast response time.

### Dynamic

React-style loop:
1. Rewrite the query and search
2. LLM summarizes search results and evaluates information sufficiency
3. If insufficient, generates the next goal and continues searching
4. If sufficient, generates the final answer

Best for complex questions requiring multi-round exploration.

### Deep

Multi-step research plan:
1. LLM pre-generates a complete research plan (multiple steps)
2. Executes step by step: rewrite → search → summarize
3. After all steps complete, generates a final answer with reference tree

Best for deep questions requiring systematic research.

## Quick Start

### Start the Server

```bash
python planning_server.py \
  --port 8083 \
  --base-url http://localhost:8078/v1 \
  --model qwen4b \
  --planner-base-url https://aix-backup.hismarttv.com/v1 \
  --planner-model deepseek-v3 \
  --planner-api-key your-api-key
```

### Health Check

```bash
curl http://localhost:8083/health
```

### Using the Client

```bash
# Simple mode
python planning_client.py \
  --server-url http://localhost:8083 \
  --question "海信冰箱怎么样" \
  --top-k 3 \
  --tool-hub "es" \
  --max-turn 3

# Deep mode
python planning_client.py \
  --server-url http://localhost:8083 \
  --question "65E8Q怎么样" \
  --top-k 3 \
  --score-threshold 0.1 \
  --tool-hub "es" \
  --max-turn 10 \
  --thinking deep

# Structured retrieval with domains
python planning_client.py \
  --server-url http://localhost:8083 \
  --question "E8Q和E7Q空调的匹数率是多少" \
  --tool-hub "struct" \
  --max-turn 1 \
  --domains '[{"domain":"SupplyChain","desc":"Supply chain knowledge graph","rdf_list":[{"scene":"material","desc":"","info":[]}]}]'
```

### Using curl Directly

```bash
curl -X POST http://localhost:8083/planner \
  -H "Content-Type: application/json; charset=utf-8" \
  -d '{
    "query": "E8Q和E7Q空调的匹数率是多少",
    "turn": 1,
    "max_turn": 1,
    "tool_hub": "struct",
    "domains": [
      {
        "domain": "SupplyChain",
        "desc": "Knowledge graph ontology for TV backplane costing",
        "rdf_list": [
          {
            "scene": "material",
            "desc": "",
            "info": [
              {
                "@id": "http://www.jhk.com/supply-chain/material/property/sprayingUnitPrice",
                "@type": ["http://www.w3.org/2002/07/owl#DatatypeProperty"]
              }
            ]
          }
        ]
      }
    ]
  }'
```

## Server Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--host` | `0.0.0.0` | Listen address |
| `--port` | `9100` | Listen port |
| `--base-url` | `http://localhost:8078/v1` | LLM endpoint for query rewrite |
| `--model` | `qwen4b` | Model for query rewrite |
| `--planner-base-url` | `https://aix-backup.hismarttv.com/v1` | LLM endpoint for deep/dynamic mode |
| `--planner-model` | `deepseek-v3` | Model for deep/dynamic mode |
| `--planner-api-key` | `EMPTY` | Planner LLM API key |
| `--session-ttl` | `600` | Deep thinking session expiry (seconds) |
| `--tool-select-base-url` | `http://localhost:8071/v1` | LLM endpoint for tool selection |
| `--tool-select-model` | `qwen4b` | Model for tool selection |
| `--log-level` | `INFO` | Log level |

Environment variable overrides: `LLM_BASE_URL`, `LLM_MODEL`, `PLANNER_BASE_URL`, `PLANNER_MODEL`, `PLANNER_API_KEY`

## Client Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--server-url` | `http://localhost:9100` | Server URL |
| `--question` | `海信冰箱` | User query |
| `--max-turn` | `3` | Maximum planning turns |
| `--top-k` | `1` | Expected number of results |
| `--score-threshold` | `0.8` | Score threshold |
| `--max-top-k` | `3` | Max top_k per sub_query |
| `--tool-hub` | `es,graph,web` | Available tools |
| `--tool-hub-optional` | `` | Optional tools (LLM decides whether to use) |
| `--thinking` | `simple` | Thinking mode: simple / dynamic / deep |
| `--search-strategy` | `broad` | KBP search strategy: broad / precise |
| `--domains` | `` | Domains JSON string (for structured retrieval) |
| `--web-backend` | `xiaosu` | Web search backend: bocha / xiaosu |
| `--tool-select-enable` | `false` | Enable model-based tool selection |
| `--debug` | `false` | Print full retrieval results |

## Protocol

### Request

```json
{
  "query": "user question",
  "turn": 1,
  "max_turn": 3,
  "tool_hub": "es,graph,web",
  "tool_hub_optional": "",
  "retrieval_setting": {
    "top_k": 3,
    "score_threshold": 0.8,
    "search_mode": "hybrid",
    "search_strategy": "broad"
  },
  "max_top_k": 3,
  "thinking": "simple",
  "history": {},
  "domains": []
}
```

### Response

```json
{
  "is_off_topic": false,
  "status": "running",
  "turn": 1,
  "current": [
    {"sub_query": "sub query", "tool_use": "es", "topk": 3}
  ],
  "final": null,
  "plan": null,
  "step_summary": null,
  "answer": null,
  "reference_tree": null
}
```

- `status="running"`: Execute retrieval for items in `current`, put results into `history`, then call again
- `status="stop"`: Planning complete, `final` contains the aggregated results
