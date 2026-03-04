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

                # 保存原始检索结果，包含序号信息
                retrieval_results = []
                for idx, item in enumerate(records):
                    retrieval_results.append({
                        'content': item['content'],
                        'index': idx + 1  # 添加序号信息
                    })

                return retrieval_results
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
                       initial_delay: float = 1.0, debug: bool = True, **kwargs) -> Dict:
        """
        调用LLM的聊天接口，带自动重试功能

        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "..."}, ...]
            llm_name: LLM模型名称，不提供则使用默认模型
            temperature: 温度参数
            n: 返回结果数量
            max_retries: 最大重试次数
            initial_delay: 初始延迟时间（秒）
            debug: 是否打印调试信息
            **kwargs: 其他payload参数

        Returns:
            LLM返回的JSON响应
        """
        llm_name = llm_name or self.default_llm
        if llm_name not in self.llm_configs:
            raise ValueError(f"未知的LLM模型: {llm_name}")

        payload = self._create_payload(llm_name, messages, temperature, n, **kwargs)
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


class TaskExecutionSystem:
    """
    任务执行系统，支持多步骤、多子步骤的任务执行
    """

    def __init__(self, llm_client: SimpleLLMClient, kbp_client: KbpRetrievalClient, debug: bool = True):
        self.llm_client = llm_client
        self.kbp_client = kbp_client
        self.debug = debug
        self.citation_chain = {}  # 存储引用链

    def generate_execution_plan(self, query: str) -> List[Dict]:
        """
        生成执行计划

        Args:
            query: 用户的查询或任务

        Returns:
            执行计划，格式为：
            [
              {
                "step_id": 1,
                "description": "步骤描述",
                "sub_steps": [
                  {
                    "sub_step_id": 1,
                    "action": "retrieval|generation",
                    "parameters": {...}
                  }
                ]
              }
            ]
        """
        plan_prompt = f"""
        请为以下任务生成详细的执行计划，计划应该分为多个步骤，每个步骤内可以有多个并行的子步骤。

        任务: {query}

        输出格式严格按照以下JSON格式:
        {{
          "steps": [
            {{
              "step_id": 1,
              "description": "步骤描述",
              "sub_steps": [
                {{
                  "sub_step_id": 1,
                  "action": "retrieval|generation",
                  "parameters": {{
                    "query": "检索查询词或生成提示词"
                  }}
                }}
              ]
            }}
          ]
        }}

        要求：
        1. 步骤之间是串行关系，后面的步骤可以依赖前面步骤的结果
        2. 同一步骤内的子步骤可以并行执行，无需相互依赖
        3. 每个子步骤都有明确的action类型和参数
        4. action类型包括：retrieval(信息检索), generation(LLM生成)
        5. 在后续步骤的query中，可以用 {{prev_step_result}} 占位符来表示上一步的结果
        """

        messages = [{"role": "user", "content": plan_prompt}]

        if self.debug:
            print(f"\n[DEBUG] 生成执行计划的输入: {plan_prompt[:200]}...")

        response = self.llm_client.chat_completion(messages, debug=False)

        if "error" in response:
            print(f"生成执行计划失败: {response['error']}")
            return []

        try:
            content = response['choices'][0]['message']['content']

            if self.debug:
                print(f"[DEBUG] 生成执行计划的输出: {content[:500]}...")

            # 提取JSON部分
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            json_str = content[start_idx:end_idx]
            plan_data = json.loads(json_str)
            return plan_data.get('steps', [])
        except Exception as e:
            print(f"解析执行计划失败: {e}")
            print(f"原始内容: {content}")
            return []

    def execute_sub_step(self, sub_step: Dict, prev_step_result: Optional[str] = None,
                         prev_citations: Optional[List[int]] = None,
                         prev_action_type: Optional[str] = None) -> Dict:
        """
        执行单个子步骤

        Args:
            sub_step: 子步骤定义
            prev_step_result: 上一步骤的结果，用于替换占位符
            prev_citations: 上一步骤的引用列表
            prev_action_type: 上一步骤的操作类型（retrieval 或 generation）

        Returns:
            子步骤执行结果
        """
        action = sub_step['action']
        parameters = sub_step.get('parameters', {}).copy()

        # 如果参数中有占位符，替换为实际值
        if 'query' in parameters and prev_step_result:
            query = parameters['query']
            query = query.replace('{{prev_step_result}}', str(prev_step_result))
            parameters['query'] = query

        if self.debug:
            print(f"[DEBUG] 执行子步骤 - 动作: {action}, 参数: {parameters}")

        if action == 'retrieval':
            query = parameters.get('query', '')
            top_k = parameters.get('top_k', 5)
            if self.debug:
                print(f"[DEBUG] KBP检索 - 查询: {query[:100]}..., top_k: {top_k}")

            results = self.kbp_client.retrieval(query, top_k=top_k)

            if self.debug:
                print(f"[DEBUG] KBP检索结果: {len(results) if results else 0} 条记录")

            # 为检索结果分配新的全局索引
            global_indices = []
            for idx, item in enumerate(results):
                global_index = len(self.citation_chain) + 1
                self.citation_chain[global_index] = {
                    'type': 'retrieval',
                    'content': item['content'],
                    'original_index': item['index'],
                    'source': f"检索结果[{item['index']}]"
                }
                item['global_index'] = global_index
                global_indices.append(global_index)

            return {
                'status': 'success',
                'data': results,
                'citations': global_indices,  # 返回新生成的全局引用索引
                'message': f"检索到 {len(results) if results else 0} 条结果，分配引用索引: {global_indices}"
            }
        elif action == 'generation':
            prompt = parameters.get('query', '')

            # 构建上下文，包含之前的引用信息
            context_parts = []

            # 如果上一步是检索，则强制要求引用其中的部分条目
            if prev_action_type == 'retrieval' and prev_citations:
                context_parts.append("您必须从以下检索结果中引用相关条目来回答问题:")
                for citation_idx in prev_citations:
                    if citation_idx in self.citation_chain:
                        citation_info = self.citation_chain[citation_idx]
                        context_parts.append(f"[{citation_idx}] {citation_info['content']}")

                # 添加特殊指令，要求明确引用
                context_parts.append("请在您的回答中明确引用上述检索结果中的相关条目，使用方括号标注引用编号（例如：[1]、[2]等）。")

            if prev_step_result:
                context_parts.append(f"上一步结果: {prev_step_result}")

            full_prompt = "\n".join(context_parts) + f"\n\n当前任务: {prompt}"

            messages = [{"role": "user", "content": full_prompt}]

            if self.debug:
                print(f"[DEBUG] LLM生成 - 提示: {full_prompt[:500]}...")

            result = self.llm_client.chat_completion(messages, debug=False)

            if "error" in result:
                return {
                    'status': 'error',
                    'data': None,
                    'citations': [],
                    'message': result['error']
                }

            content = result['choices'][0]['message']['content']

            # 为生成的内容分配新的全局索引
            global_index = len(self.citation_chain) + 1
            # 分析生成内容中的引用，找出它引用了哪些检索结果
            used_citations = []
            if prev_citations:
                for citation_idx in prev_citations:
                    if f"[{citation_idx}]" in content or f" [{citation_idx}]" in content or f"[{citation_idx}]" in content:
                        used_citations.append(citation_idx)

            self.citation_chain[global_index] = {
                'type': 'generation',
                'content': content,
                'source': f"LLM生成结果",
                'parent_citations': prev_citations or [],
                'used_citations': used_citations  # 实际使用的引用
            }

            if self.debug:
                print(f"[DEBUG] LLM生成结果: {content[:500]}...")

            return {
                'status': 'success',
                'data': content,
                'citations': [global_index],  # 返回新生成的全局引用索引
                'used_citations': used_citations,  # 实际使用的引用
                'message': f"LLM生成完成，分配引用索引: [{global_index}]，使用了引用: {used_citations}"
            }
        else:
            return {
                'status': 'error',
                'data': None,
                'citations': [],
                'message': f"未知的动作类型: {action}"
            }

    def execute_step(self, step: Dict, prev_step_result: Optional[str] = None,
                     prev_citations: Optional[List[int]] = None,
                     prev_action_type: Optional[str] = None) -> Dict:
        """
        执行单个步骤（包含多个子步骤）

        Args:
            step: 步骤定义
            prev_step_result: 上一步骤的结果
            prev_citations: 上一步骤的引用列表
            prev_action_type: 上一步骤的操作类型

        Returns:
            步骤执行结果
        """
        print(f"\n执行步骤 {step['step_id']}: {step['description']}")

        sub_steps = step['sub_steps']
        step_results = {}

        # 并行执行子步骤（这里简化为顺序执行，但逻辑上是并行的）
        all_step_citations = []
        all_used_citations = []
        for sub_step in sub_steps:
            print(f"  执行子步骤 {sub_step['sub_step_id']}: {sub_step['action']}")

            # 执行子步骤
            result = self.execute_sub_step(sub_step, prev_step_result, prev_citations, prev_action_type)
            step_results[str(sub_step['sub_step_id'])] = result

            print(f"    - {result['message']}")

            # 收集当前步骤的所有引用
            if result.get('citations'):
                all_step_citations.extend(result['citations'])
            if result.get('used_citations'):
                all_used_citations.extend(result['used_citations'])

        # 整合步骤结果
        combined_result = ""
        relevant_indices = []  # 存储相关的检索结果索引
        retrieval_contents = []  # 存储检索内容以便后续显示

        for sub_step_id, sub_result in step_results.items():
            if sub_result['status'] == 'success' and sub_result['data']:
                if isinstance(sub_result['data'], list):
                    # 检查是否是检索结果（包含索引信息）
                    if sub_result['data'] and 'index' in sub_result['data'][0]:
                        # 这是一个检索结果，包含索引信息
                        for item in sub_result['data']:
                            global_idx = item.get('global_index', item['index'])
                            combined_result += f"[{global_idx}] {item['content']} "
                            relevant_indices.append(global_idx)
                            retrieval_contents.append({
                                'index': global_idx,
                                'content': item['content']
                            })
                    else:
                        combined_result += " ".join(str(item) for item in sub_result['data'])
                else:
                    combined_result += str(sub_result['data'])

        return {
            'step_id': step['step_id'],
            'results': step_results,
            'combined_result': combined_result,
            'relevant_indices': relevant_indices,  # 添加相关的检索结果索引
            'retrieval_contents': retrieval_contents,  # 添加检索内容
            'citations': sorted(list(set(all_step_citations))),  # 当前步骤的引用列表
            'used_citations': sorted(list(set(all_used_citations))),  # 实际使用的引用列表
            'action_type': 'generation' if any('generation' in str(res.get('data', '')) for res in step_results.values()) else 'retrieval',
            'summary': f"步骤 {step['step_id']} 完成，共执行了 {len(sub_steps)} 个子步骤"
        }

    def execute_task(self, query: str) -> Dict:
        """
        执行完整任务

        Args:
            query: 用户的查询或任务

        Returns:
            任务执行结果
        """
        print(f"开始执行任务: {query}")

        # 重置引用链
        self.citation_chain = {}

        # 1. 生成执行计划
        print("\n正在生成执行计划...")
        execution_plan = self.generate_execution_plan(query)

        if not execution_plan:
            return {
                'status': 'error',
                'message': '无法生成执行计划',
                'final_answer': '抱歉，无法处理您的请求'
            }

        print(f"生成了 {len(execution_plan)} 个步骤的执行计划")

        # 2. 按步骤执行
        prev_step_result = None
        prev_citations = []
        prev_action_type = None
        all_results = {}
        all_relevant_indices = []  # 存储所有步骤的相关索引
        all_retrieval_contents = []  # 存储所有检索内容

        for i, step in enumerate(execution_plan):
            step_result = self.execute_step(step, prev_step_result, prev_citations, prev_action_type)
            all_results[step['step_id']] = step_result

            print(f"  {step_result['summary']}")

            # 更新前一步骤结果和引用列表，供下一步使用
            prev_step_result = step_result['combined_result']
            prev_citations = step_result['citations']
            prev_action_type = step_result['action_type']

            # 收集所有相关索引和内容
            all_relevant_indices.extend(step_result.get('relevant_indices', []))
            all_retrieval_contents.extend(step_result.get('retrieval_contents', []))

        # 3. 生成最终答案及相关的索引
        print("\n正在生成最终答案和相关索引...")
        final_answer, relevant_indices = self.generate_final_answer_and_relevant_indices(query, all_results)

        return {
            'status': 'success',
            'execution_plan': execution_plan,
            'all_results': all_results,
            'final_answer': final_answer,
            'relevant_indices': relevant_indices,  # 由大模型识别的相关索引
            'all_retrieval_contents': all_retrieval_contents,  # 包含所有检索内容
            'citation_chain': self.citation_chain,  # 包含完整的引用链
            'message': '任务执行完成'
        }

    def generate_final_answer_and_relevant_indices(self, original_query: str, all_results: Dict) -> Tuple[str, List[int]]:
        """
        根据所有步骤结果生成最终答案，并让大模型识别最相关的检索结果索引

        Args:
            original_query: 原始查询
            all_results: 所有步骤执行结果

        Returns:
            (最终答案, 相关索引列表)
        """
        # 构建上下文
        context_parts = []
        all_contents = []

        for step_id, step_result in all_results.items():
            # 提取内容并构建带索引的上下文
            combined_content = step_result['combined_result']
            context_parts.append(f"步骤 {step_id} 结果: {combined_content}")

            # 解析检索结果，提取所有带索引的内容
            for sub_step_id, sub_result in step_result['results'].items():
                if sub_result['status'] == 'success' and isinstance(sub_result['data'], list):
                    for item in sub_result['data']:
                        if 'index' in item and 'content' in item:
                            all_contents.append((item['global_index'], item['content']))

        context = "\n".join(context_parts)

        # 按索引排序所有内容
        sorted_contents = sorted(all_contents, key=lambda x: x[0])

        # 构建详细上下文，包含所有检索结果及其编号
        detailed_context = "所有检索结果:\n"
        for idx, content in sorted_contents:
            detailed_context += f"[{idx}] {content}\n\n"

        # 构建完整的引用链上下文
        citation_context = "完整的引用链信息:\n"
        for idx in sorted(self.citation_chain.keys()):
            citation_info = self.citation_chain[idx]
            source_info = citation_info.get('source', 'Unknown')
            parent_citations = citation_info.get('parent_citations', [])
            used_citations = citation_info.get('used_citations', [])
            parent_refs = f" (来自引用: {parent_citations})" if parent_citations else ""
            used_refs = f" (使用了引用: {used_citations})" if used_citations else ""
            citation_context += f"[{idx}] {citation_info['content']} [来源: {source_info}{parent_refs}{used_refs}]\n\n"

        final_prompt = f"""
        基于以下任务执行结果，为原始查询生成最终答案，并在答案中适当位置引用相关信息源的编号。
        同时，请分析并列出最相关的检索结果索引。

        原始查询: {original_query}

        {citation_context}

        执行结果:
        {context}

        请按以下格式输出：

        最终答案:
        [在此处生成最终答案，并在引用信息时使用方括号标注引用编号，如[1]、[2]等]

        相关索引:
        [在此处列出最相关的检索结果索引，以逗号分隔，例如: 1,3,5]
        """

        messages = [{"role": "user", "content": final_prompt}]

        if self.debug:
            print(f"[DEBUG] 生成最终答案和相关索引的输入: {final_prompt[:500]}...")

        response = self.llm_client.chat_completion(messages, debug=False)

        if "error" in response:
            return f"生成最终答案时出错: {response['error']}", []

        try:
            content = response['choices'][0]['message']['content']

            if self.debug:
                print(f"[DEBUG] 生成最终答案和相关索引的输出: {content[:500]}...")

            # 解析大模型的输出，提取最终答案和相关索引
            final_answer = ""
            relevant_indices = []

            lines = content.split('\n')
            current_section = None
            for line in lines:
                if line.startswith('最终答案:'):
                    current_section = 'answer'
                    continue
                elif line.startswith('相关索引:'):
                    current_section = 'indices'
                    continue

                if current_section == 'answer':
                    final_answer += line + '\n'
                elif current_section == 'indices':
                    # 提取数字索引
                    import re
                    numbers = re.findall(r'\d+', line)
                    relevant_indices.extend([int(num) for num in numbers if num.strip()])

            # 清理最终答案
            final_answer = final_answer.strip()

            if self.debug:
                print(f"[DEBUG] 解析出的最终答案: {final_answer[:200]}...")
                print(f"[DEBUG] 解析出的相关索引: {relevant_indices}")

            return final_answer, sorted(list(set(relevant_indices)))  # 去重并排序
        except Exception as e:
            return f"生成最终答案时出错: {str(e)}", []


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

    # 创建任务执行系统
    task_system = TaskExecutionSystem(llm_client, kbp_client, debug=True)

    # 示例查询
    user_query = sys.argv[1] if len(sys.argv) > 1 else "请告诉我海信的历史"

    # 执行任务
    result = task_system.execute_task(user_query)

    print("\n" + "="*50)
    print("任务执行完成！")
    print("="*50)
    print(f"最终答案:\n{result['final_answer']}")

    # 打印执行计划
    print("\n执行计划详情:")
    for step in result['execution_plan']:
        print(f"  步骤 {step['step_id']}: {step['description']}")
        for sub_step in step['sub_steps']:
            print(f"    - 子步骤 {sub_step['sub_step_id']}: {sub_step['action']}, 参数: {sub_step.get('parameters', {})}")

    # 打印由大模型生成的相关检索结果的序号
    print(f"\n大模型识别的相关检索条目序号: {result.get('relevant_indices', [])}")

    # 打印所有检索结果及其索引
    print(f"\n\n\n\n\n所有检索结果:")
    for item in result.get('all_retrieval_contents', []):
        print(f"[{item['index']}] {item['content']}")

    # 打印完整的引用链
    print(f"\n\n\n\n\n完整的引用链:")
    for idx in sorted(result.get('citation_chain', {}).keys()):
        citation_info = result['citation_chain'][idx]
        print(f"[{idx}] {citation_info['content'][:200]}... [类型: {citation_info['type']}, 来源: {citation_info.get('source', 'Unknown')}, 使用引用: {citation_info.get('used_citations', [])}]")
