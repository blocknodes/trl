import os
import json
import ssl
import warnings
import time
import random
import requests  # 新增：KbpRetrievalClient 需要用到
from dotenv import load_dotenv
from openai import OpenAI

# ====================== 1. KBP 检索客户端类 ======================
class KbpRetrievalClient:
    """
    封装 KBP 混合检索 API 的客户端类
    """

    def __init__(self, base_url="https://inner-apisix-test.hisense.com",
                 user_key="qimfvt7lwtqeyangfl259vjg8fzdhh5l",
                 api_key="83dd8d9d-6a77-4954-9071-aa195fb6b406"):
        """
        初始化客户端

        :param base_url: API 基础地址
        :param user_key: 用户 key
        :param api_key: API 密钥
        """
        self.base_url = base_url.rstrip('/')
        self.user_key = user_key
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'api-key': self.api_key
        })

    def retrieval(self, query, top_k=5, score_threshold=0, search_mode='hybrid',
                 tracing_model=False, max_retries=10, initial_delay=0.5, backoff_factor=2.0):
        """
        执行检索请求（带退火重试机制）

        :param query: 查询文本
        :param top_k: 返回结果数量
        :param score_threshold: 分数阈值
        :param search_mode: 搜索模式
        :param tracing_model: 是否追踪模型
        :param max_retries: 最大重试次数
        :param initial_delay: 初始延迟时间（秒）
        :param backoff_factor: 退避因子
        :return: API 响应结果（字典）
        """
        url = f"{self.base_url}/kbp-test/openapi/kbp/mix/retrieval?user_key={self.user_key}"

        payload = {
            "retrieval_setting": {
                "top_k": top_k,
                "score_threshold": score_threshold,
                "search_mode": search_mode,
                "search_strategy":"precise",
            },
            "query": query,
            "tracingModel": tracing_model
        }

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                response = self.session.post(url, data=json.dumps(payload))
                response.raise_for_status()  # 如果状态码不是 200, 则引发 HTTPError 异常
                response = response.json()
                responses = [f"{item['title']}\n{item['content']}" for item in response['records']]

                return responses
            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < max_retries:
                    # 计算退避时间（带随机抖动）
                    delay = initial_delay * (backoff_factor ** attempt)
                    delay += random.uniform(0, 0.5 * delay)  # 添加随机抖动
                    print(f"KBP检索请求失败（第 {attempt+1} 次）: {e}，将在 {delay:.2f} 秒后重试...")
                    time.sleep(delay)
                else:
                    print(f"KBP检索已达到最大重试次数 ({max_retries})，请求最终失败: {e}")

        return None

# ====================== 2. 全局配置 & SSL 修复 ======================
# 加载环境变量
load_dotenv()

# 禁用 SSL 证书验证（适配本地/内网接口）
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# 禁用所有 SSL 警告（避免终端刷屏）
warnings.filterwarnings("ignore")

# ====================== 3. 基础配置 ======================
class Config:
    """配置类：管理 API 密钥和模型参数"""
    def __init__(self):
        # OpenAI 配置（支持 Qwen OpenAI 兼容接口）
        self.api_key = os.getenv("OPENAI_API_KEY","empty")
        self.base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:8088/v1")
        self.model = os.getenv("OPENAI_MODEL_NAME", "qwen4b")

        # 智能体参数
        self.temperature = 0.5
        self.max_tokens = 4096
        self.max_iterations = 2

        # KBP 检索配置
        self.kbp_base_url = os.getenv("KBP_BASE_URL", "https://inner-apisix-test.hisense.com")
        self.kbp_user_key = os.getenv("KBP_USER_KEY", "qimfvt7lwtqeyangfl259vjg8fzdhh5l")
        self.kbp_api_key = os.getenv("KBP_API_KEY", "83dd8d9d-6a77-4954-9071-aa195fb6b406")

# ====================== 4. 工具定义（替换为 KBP 检索） ======================
class ToolRegistry:
    """工具注册表：管理智能体可调用的工具（替换为 KBP 检索）"""
    def __init__(self, config):
        self.config = config
        self.tools = {}
        # 初始化 KBP 检索客户端
        self.kbp_client = KbpRetrievalClient(
            base_url=config.kbp_base_url,
            user_key=config.kbp_user_key,
            api_key=config.kbp_api_key
        )
        self._register_builtin_tools()

    def _register_builtin_tools(self):
        """注册内置工具（替换为 KBP 检索）"""
        # 工具1：KBP 混合检索（替代原 call_local_api）
        def kbp_retrieval(query: str, top_k: int = 5, score_threshold: float = 0,
                         search_mode: str = 'hybrid') -> str:
            """
            KBP 混合检索工具
            :param query: 检索查询文本
            :param top_k: 返回结果数量（默认5）
            :param score_threshold: 分数阈值（默认0）
            :param search_mode: 搜索模式（默认hybrid，可选：keyword/semantic/hybrid）
            :return: 检索结果（格式化字符串）
            """
            print(f"\n🔍 调用 KBP 检索：查询='{query}'，top_k={top_k}，模式={search_mode}")
            result = self.kbp_client.retrieval(
                query=query,
                top_k=top_k,
                score_threshold=score_threshold,
                search_mode=search_mode
            )

            if result is None:
                return "KBP 检索失败：请求超时或达到最大重试次数"

            # 格式化返回结果
            formatted_result = json.dumps(result, ensure_ascii=False, indent=2)
            return f"KBP 检索结果：\n{formatted_result}"

        # 工具2：加法计算（保留原有功能）
        def add(a: float, b: float) -> str:
            """加法计算工具"""
            return f"{a} + {b} = {a + b}"

        # 工具3：乘法计算（保留原有功能）
        def multiply(a: float, b: float) -> str:
            """乘法计算工具"""
            return f"{a} × {b} = {a * b}"

        # 注册工具（替换 call_local_api 为 kbp_retrieval）
        self.tools = {
            "kbp_retrieval": (
                kbp_retrieval,
                "KBP 混合检索，参数：query（检索文本）、top_k（返回数量）、score_threshold（分数阈值）、search_mode（搜索模式）"
            ),
            "add": (
                add,
                "加法计算，参数：a（数字）、b（数字）"
            ),
            "multiply": (
                multiply,
                "乘法计算，参数：a（数字）、b（数字）"
            )
        }

    def get_tool(self, tool_name: str):
        """获取工具函数和描述"""
        if tool_name not in self.tools:
            return None, f"工具 {tool_name} 不存在"
        return self.tools[tool_name]

    def list_tools(self):
        """列出所有可用工具"""
        tool_descs = []
        for name, (_, desc) in self.tools.items():
            tool_descs.append(f"{name}: {desc}")
        return "\n".join(tool_descs)

# ====================== 5. Deep Agent 核心实现 ======================
class SimpleDeepAgent:
    """简易版 Deep Agent（集成 KBP 检索工具）"""
    def __init__(self):
        self.config = Config()
        self.tool_registry = ToolRegistry(self.config)

        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )

        # 智能体状态
        self.thought_history = []

    def _get_agent_prompt(self, user_input: str):
        """生成智能体提示词（更新工具描述）"""
        tools_desc = self.tool_registry.list_tools()
        thought_history = "\n".join(self.thought_history)

        prompt = f"""
你是一个具备多步推理能力的 Deep Agent，专注于调用 KBP 混合检索接口解决问题，遵循以下流程：
1. 分析用户问题，判断是否需要调用 KBP 检索/计算工具
2. 如需调用工具：输出严格的 JSON 格式，包含 tool（工具名）、params（参数）、reason（调用理由）
3. 如无需工具/已获取足够信息：输出 FINAL_ANSWER: 你的最终答案
4. 复杂问题拆解为多步，逐步调用工具获取信息后推理

可用工具列表：
{tools_desc}

思考历史（已执行步骤）：
{thought_history if thought_history else "无"}

用户问题：{user_input}

注意：
- 调用 KBP 检索时，query 参数必须填写完整的检索文本
- 参数必须严格匹配工具要求，top_k 为整数，score_threshold 为浮点数
- 工具调用仅返回 JSON，无其他多余文字
- 多步推理循序渐进，每一步只解决一个子问题
"""
        return prompt

    def _call_llm(self, prompt: str) -> str:
        """调用 LLM（本地 OpenAI 兼容接口）"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            print(f'\n=======\nprompt:{prompt}\nresponse:{response.choices[0].message.content.strip()}\n===!!====\n')
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"LLM 调用失败：{str(e)}"

    def _call_tool(self, tool_name: str, params: dict) -> str:
        """调用工具"""
        tool_func, _ = self.tool_registry.get_tool(tool_name)
        if not tool_func:
            return f"工具调用失败：{tool_name} 不存在"

        try:
            result = tool_func(** params)
            return f"工具调用成功：\n{result}"
        except Exception as e:
            return f"工具调用出错：{str(e)}"

    def run(self, user_input: str) -> str:
        """运行智能体（核心入口）"""
        self.thought_history = []
        iterations = 0

        while iterations < self.config.max_iterations:
            iterations += 1

            # 1. 生成提示词并调用 LLM
            prompt = self._get_agent_prompt(user_input)
            llm_response = self._call_llm(prompt)

            # 2. 处理最终答案
            if llm_response.startswith("FINAL_ANSWER:"):
                final_answer = llm_response.replace("FINAL_ANSWER:", "").strip()
                self.thought_history.append(f"步骤 {iterations}：返回最终答案 - {final_answer}")
                return final_answer

            # 3. 处理工具调用
            try:
                tool_call = json.loads(llm_response)
                tool_name = tool_call.get("tool")
                params = tool_call.get("params", {})
                reason = tool_call.get("reason", "无")

                # 记录思考历史
                self.thought_history.append(
                    f"步骤 {iterations}：调用工具 {tool_name} - 理由：{reason} - 参数：{params}"
                )

                # 调用工具并获取结果
                tool_result = self._call_tool(tool_name, params)
                self.thought_history.append(f"步骤 {iterations} 工具结果：{tool_result}")

                # 打印中间过程
                print(f"\n=== 推理步骤 {iterations} ===")
                print(f"调用工具：{tool_name}")
                print(f"参数：{params}")
                print(f"结果：{tool_result[:200]}...")

            except json.JSONDecodeError:
                self.thought_history.append(f"步骤 {iterations}：LLM 响应格式错误 - {llm_response}")
                continue
            except Exception as e:
                error_msg = f"步骤 {iterations} 出错：{str(e)}"
                self.thought_history.append(error_msg)
                continue

        return f"推理超时（最大 {self.config.max_iterations} 步），历史：\n{self.thought_history}"

# ====================== 6. 运行示例（测试 KBP 检索） ======================
if __name__ == "__main__":
    # 初始化智能体
    agent = SimpleDeepAgent()
    print("简易版 Deep Agent 已初始化（集成 KBP 混合检索）！\n")

    # 测试用例1：KBP 检索
    test_input1 = "检索关于海信电视的产品信息"
    print(f"用户问题：{test_input1}\n")
    result1 = agent.run(test_input1)
    print(f"\n最终答案：{result1}")

    # 测试用例2：简单计算（保留原有功能）
    test_input2 = "计算 100 * 25 + 500 的结果"
    print(f"\n用户问题：{test_input2}\n")
    result2 = agent.run(test_input2)
    print(f"\n最终答案：{result2}")