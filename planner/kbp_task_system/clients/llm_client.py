import requests
import random
import time
from typing import List, Dict, Optional, Tuple

class SimpleLLMClient:
    """
    精简版LLM客户端，只保留自动重试功能
    """

    def __init__(self, llm_configs: Dict[str, Dict], default_llm: Optional[str] = None):
        """
        初始化LLM客户端
        """
        self.llm_configs = llm_configs
        self.default_llm = default_llm or next(iter(llm_configs.keys()))

        if self.default_llm not in self.llm_configs:
            raise ValueError(f"默认模型 {self.default_llm} 不在配置中")

    def _prepare_request_parameters(self, llm_name: str) -> tuple:
        """准备LLM API请求的URL和headers"""
        config = self.llm_configs[llm_name]

        # 处理URL参数
        url_params = config["url_params"]
        if url_params:
            formatted_params = {k: v.format(key=config["key"]) for k, v in url_params.items()}
            query_string = "&".join([f"{k}={v}" for k, v in formatted_params.items()])
            request_url = f"{config['url']}?{query_string}"
        else:
            request_url = config["url"]

        # 处理请求头
        headers = {k: v.format(key=config["key"]) for k, v in config["headers"].items()}

        return request_url, headers

    def _create_payload(self, llm_name: str, messages: List[Dict[str, str]],
                       temperature: float = 0, n: int = 1, **kwargs) -> Dict:
        """创建LLM API请求的payload"""
        return {
            "model": self.llm_configs[llm_name]["model"],
            "messages": messages,
            "temperature": temperature,
            "n": n,** kwargs
        }

    def chat_completion(self, messages: List[Dict[str, str]], llm_name: Optional[str] = None,
                       temperature: float = 0, n: int = 1, max_retries: int = 3,
                       initial_delay: float = 1.0, debug: bool = True, **kwargs) -> Dict:
        """
        调用LLM的聊天接口，带自动重试功能
        """
        llm_name = llm_name or self.default_llm
        if llm_name not in self.llm_configs:
            raise ValueError(f"未知的LLM模型: {llm_name}")

        payload = self._create_payload(llm_name, messages, temperature, n,** kwargs)
        request_url, headers = self._prepare_request_parameters(llm_name)

        if debug:
            print(f"\n[DEBUG] LLM请求 - 模型: {llm_name}")
            print(f"[DEBUG] 请求URL: {request_url}")
            print(f"[DEBUG] 请求头: {headers}")
            print(f"[DEBUG] 请求消息: {messages}")
            print(f"[DEBUG] 请求参数: {payload}")

        # 带指数退避的重试机制
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(request_url, json=payload, headers=headers, timeout=30)

                if debug:
                    print(f"[DEBUG] LLM响应状态码: {response.status_code}")

                if response.status_code == 200:
                    result = response.json()

                    if debug:
                        print(f"[DEBUG] LLM响应内容: {json.dumps(result, indent=2, ensure_ascii=False)[:500]}...")

                    return result

                # 非200状态码，准备重试
                if debug:
                    print(f"[DEBUG] LLM请求失败 - 状态码: {response.status_code}, 响应: {response.text[:300]}...")

                if attempt < max_retries:
                    delay = initial_delay * (2 ** attempt) + random.uniform(0, 0.5)
                    time.sleep(delay)
                else:
                    error_response = {"error": f"API请求失败，状态码: {response.status_code}", "details": response.text}
                    if debug:
                        print(f"[DEBUG] LLM请求最终失败: {error_response}")
                    return error_response

            except Exception as e:
                # 发生异常，准备重试
                if debug:
                    print(f"[DEBUG] LLM请求异常: {str(e)}")

                if attempt < max_retries:
                    delay = initial_delay * (2 ** attempt) + random.uniform(0, 0.5)
                    time.sleep(delay)
                else:
                    error_response = {"error": "调用LLM时发生错误", "details": str(e)}
                    if debug:
                        print(f"[DEBUG] LLM请求最终异常: {error_response}")
                    return error_response

        return {"error": "达到最大重试次数"}