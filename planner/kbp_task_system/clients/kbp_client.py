import requests
import json
import time
import random
from typing import List, Dict, Optional

class KbpRetrievalClient:
    """
    封装 KBP 混合检索 API 的客户端类
    """

    def __init__(self, base_url: str, user_key: str, api_key: str):
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

    def retrieval(self, query: str, top_k: int = 5, score_threshold: int = 0,
                 search_mode: str = 'hybrid', tracing_model: bool = False,
                 max_retries: int = 10, initial_delay: float = 0.5,
                 backoff_factor: float = 2.0) -> Optional[List[Dict]]:
        """
        执行检索请求（带退火重试机制）
        """
        url = f"{self.base_url}/kbp-test/openapi/kbp/mix/retrieval?user_key={self.user_key}"

        payload = {
            "retrieval_setting": {
                "top_k": top_k,
                "score_threshold": score_threshold,
                "search_mode": search_mode,
                "search_strategy": "precise",
            },
            "query": query,
            "tracingModel": tracing_model
        }

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                response = self.session.post(url, data=json.dumps(payload))
                response.raise_for_status()
                records = response.json()['records']

                retrieval_results = []
                for idx, item in enumerate(records):
                    item['index'] = idx + 1
                    retrieval_results.append(item)

                return retrieval_results
            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < max_retries:
                    delay = initial_delay * (backoff_factor ** attempt)
                    delay += random.uniform(0, 0.5 * delay)
                    print(f"请求失败（第 {attempt+1} 次）: {e}，将在 {delay:.2f} 秒后重试...")
                    time.sleep(delay)
                else:
                    print(f"已达到最大重试次数 ({max_retries})，请求最终失败: {e}")

        return None