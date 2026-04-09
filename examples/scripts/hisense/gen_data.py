import json
import sys

def batch_convert_to_prompt_solution_jsonl(
    input_jsonl: str = "input_queries.jsonl",    # 你的 query 输入文件
    output_jsonl: str = "sft_prompt_dataset.jsonl",  # 输出文件
    default_solution: str = "Yes"  # 默认 solution（可改）
):
    """
    从输入 jsonl 批量读取 query，转换成你需要的格式：
    {
      "prompt": [{"content": query, "role": "user"}],
      "solution": "Yes"
    }
    """
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

    with open(input_jsonl, "r", encoding="utf-8") as f_in, \
         open(output_jsonl, "w", encoding="utf-8") as f_out:

        count = 0
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            # 兼容两种输入格式：{"query":"..."} 或 纯字符串行
            try:
                data = json.loads(line)
                if isinstance(data, dict) and "query" in data:
                    query = data["query"]
                else:
                    query = str(data)
            except:
                query = line  # 纯文本行

            # 构建你要的最终格式
            output_data = {
                "prompt": [
                    {
                        "content": SYSTEM_PROMPT,
                        "role": "system"
                    },
                    {
                        "content": query,
                        "role": "user"
                    }
                ],
                "solution": default_solution
            }

            # 写入 jsonl
            f_out.write(json.dumps(output_data, ensure_ascii=False) + "\n")
            count += 1

    print(f"✅ 转换完成！共处理 {count} 条数据")
    print(f"📁 输出文件：{output_jsonl}")


# ------------------- 直接运行 -------------------
if __name__ == "__main__":
    batch_convert_to_prompt_solution_jsonl(
        input_jsonl=sys.argv[1],
        output_jsonl="sft_prompt_dataset.jsonl",
        default_solution="Yes"  # 你可以改成任意默认答案
    )