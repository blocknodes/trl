import mysql.connector
from mysql.connector import Error
from typing import List, Dict, Optional, Tuple, Union
import random
import time
import requests
import json
import time
import random
import requests
from typing import Dict, List, Optional
from typing import Callable, Dict, Any, Optional, List, Tuple
import os
from tqdm import tqdm
import sys
import copy


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
                records = response.json()['records']
                records = [item['content'] for item in records]
                return records
            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < max_retries:
                    # 计算退避时间（带随机抖动）
                    delay = initial_delay * (backoff_factor ** attempt)
                    delay += random.uniform(0, 0.5 * delay)  # 添加随机抖动
                    print(f"请求失败（第 {attempt+1} 次）: {e}，将在 {delay:.2f} 秒后重试...")
                    time.sleep(delay)
                else:
                    print(f"已达到最大重试次数 ({max_retries})，请求最终失败: {e}")

        return None

class SimpleLLMClient:
    """
    精简版LLM客户端，只保留自动重试功能
    """

    def __init__(self, llm_configs: Dict[str, Dict], default_llm: Optional[str] = None):
        """
        初始化LLM客户端

        Args:
            llm_configs: LLM模型配置字典
            default_llm: 默认使用的LLM模型名称
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
            "n": n,
            **kwargs
        }

    def chat_completion(self, messages: List[Dict[str, str]], llm_name: Optional[str] = None,
                       temperature: float = 0, n: int = 1, max_retries: int = 3,
                       initial_delay: float = 1.0, **kwargs) -> Dict:
        """
        调用LLM的聊天接口，带自动重试功能

        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "..."}, ...]
            llm_name: LLM模型名称，不提供则使用默认模型
            temperature: 温度参数
            n: 返回结果数量
            max_retries: 最大重试次数
            initial_delay: 初始延迟时间（秒）
            **kwargs: 其他payload参数

        Returns:
            LLM返回的JSON响应
        """
        llm_name = llm_name or self.default_llm
        if llm_name not in self.llm_configs:
            raise ValueError(f"未知的LLM模型: {llm_name}")

        payload = self._create_payload(llm_name, messages, temperature, n, **kwargs)
        request_url, headers = self._prepare_request_parameters(llm_name)

        # 带指数退避的重试机制
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(request_url, json=payload, headers=headers, timeout=30)

                if response.status_code == 200:
                    return response.json()

                # 非200状态码，准备重试
                if attempt < max_retries:
                    delay = initial_delay * (2 ** attempt) + random.uniform(0, 0.5)
                    time.sleep(delay)
                else:
                    return {"error": f"API请求失败，状态码: {response.status_code}", "details": response.text}

            except Exception as e:
                # 发生异常，准备重试
                if attempt < max_retries:
                    delay = initial_delay * (2 ** attempt) + random.uniform(0, 0.5)
                    time.sleep(delay)
                else:
                    return {"error": "调用LLM时发生错误", "details": str(e)}

        return {"error": "达到最大重试次数"}


# ------------------- 使用示例 -------------------
if __name__ == "__main__":
    # LLM配置
    LLM_CONFIGS = {
        "deepseek-v3": {
            "url": "https://aix-backup.hismarttv.com/v1/chat/completions",
            "headers": {"Content-Type": "application/json", "Authorization": "Bearer {key}"},
            "key": "x31ctKZ0ONfi1jkO",
            "model": "deepseek-v3",
            "url_params": {}
        },
        "gpt-4": {
            "url": "https://inner-apisix.hisense.com/openai/deployments/gpt-4-1/chat/completions",
            "headers": {"Content-Type": "application/json", "api-key": "Oi4rzFyLbMOmqVn8YYEyT2Pt0mkr3lgU"},
            "key": "nregzh6g2oviajyjstgzlhjsjmp9rtql",
            "model": "gpt-4-1",
            "url_params": {"user_key": "{key}"}
        },
        "qwen3-4b": {
            "url": "http://localhost:8088/v1/chat/completions",
            "headers": {"Content-Type": "application/json", "Authorization": "Bearer {key}"},
            "key": "x31ctKZ0ONfi1jkO",
            "model": "qwen4b",
            "url_params": {}
        }
    }

    llm_client = SimpleLLMClient(llm_configs=LLM_CONFIGS, default_llm="qwen3-4b")
    kbp_client = KbpRetrievalClient()

    query = sys.argv[1]

    queue = []

    node = {'orig_query':query,'current_query':query,'refs':[], 'debug':{}, 'level':0}

    queue.append(node)

    while len(queue) > 0:
        queue_copied = copy.deepcopy(queue)
        queue = []

        for node in queue_copied:
            orig_query = node['orig_query']
            query = node['current_query']
            refs = node['refs']
            level = node['level']
            if level >5:
                print(f'level oveflow {level}')
                sys.exit(0)

            results = kbp_client.retrieval(
                query=query,
                top_k=3,
                score_threshold=0,
                search_mode="hybrid",
                tracing_model=False
            )
            result=''
            merged_refs = refs + results
            for i in range(len(merged_refs)):
                cleaned_text = merged_refs[i].replace('\n', ' ')
                result += f"{i}. {cleaned_text}\n\n"

            prompt = f"""任务描述： 你是一个问答专家：以下是用户问题和召回的documents:
用户问题：{orig_query}
documents:
{result}
请根据当前召回的documents输出对应的结果，以json表示，具体要求如下：
1. 如果当前召回结果满足用户的query，直接输出对应的reference和answer:如：{{"ref":[0,3],"answer": answer}}
2. 如果当前召回结果不满足用户的query，输出有用的reference，并给予前面输出的reference，输出以及需要补充的subquery：{{"ref":[1],"need_retrieve":[subquery1,subquer2...]}}
3. ref要严格对应序号，没有的话置[],不要输出多余字符，仅json
"""
            # 发送请求
            messages = [{"role": "user", "content": prompt}]
            response = llm_client.chat_completion(
                messages=messages,
                temperature=0.8,
                max_retries=20,
                n=1,
                chat_template_kwargs = {
                    "thinking": True
                },
            )
            response = response['choices'][0]['message']['content']
            print(f'【轮数 {level}】query:{query}\n【轮数 {level}】prompt:{prompt}\n【轮数 {level}】LLM响应:{response}')
            ## parse the response and continue to
            #response = json.loads(response.split('</think>\n\n')[1])
            #response = response.split('</think>\n\n')[1]
            #print(f'!!!!!\n{response}')
            response = json.loads(response)
            refs = response['ref']

            refs = [merged_refs[i] for i in refs if i <len(merged_refs)]

            ref_string = ''
            if 'need_retrieve' not in response.keys():
                print(f'##### done #####')
                for i in range(len(refs)):
                    ref_string += f"{i}. {refs[i]}\n"
                #ref_string = "\n".join(refs)
                answer = response['answer']
                print(f'origal query:{orig_query}\nrefs:\n{ref_string}\nanswer:{answer}')
                print(f'##### done #####!!!!!')
                sys.exit(0)

            subqueries = response['need_retrieve']



            for subquery in subqueries:
                queue.append({'orig_query':orig_query, 'current_query':subquery, 'refs':refs, 'level': level+1, 'debug':{}})




