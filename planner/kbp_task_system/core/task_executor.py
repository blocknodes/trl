import json
import re
from typing import List, Dict, Optional, Tuple, Union
from clients.llm_client import SimpleLLMClient
from clients.kbp_client import KbpRetrievalClient

class TaskExecutionSystem:
    """
    任务执行系统，支持多步骤、多子步骤的任务执行
    """

    def __init__(self, llm_client: SimpleLLMClient, kbp_client: KbpRetrievalClient, debug: bool = True, max_rounds: int = 10):
        """
        初始化任务执行系统
        """
        self.llm_client = llm_client
        self.kbp_client = kbp_client
        self.debug = debug
        self.citation_chain = {}  # 存储引用链
        self.max_rounds = max_rounds  # 最大轮次限制

    def analyze_next_steps(self, remaining_plan: List[Dict]) -> bool:
        """
        分析剩余计划中是否还有检索任务
        """
        has_retrieval = False
        for step in remaining_plan:
            for sub_step in step.get('sub_steps', []):
                if sub_step.get('action') == 'retrieval':
                    has_retrieval = True
                    break
            if has_retrieval:
                break
        return not has_retrieval  # 如果没有检索任务，返回True

    def generate_execution_plan(self, query: str) -> List[Dict]:
        """
        生成执行计划
        """
        plan_prompt = f"""
        请为以下任务生成详细的执行计划，计划应该分为多个步骤，每个步骤内可以有多个并行的子步骤。
        注意：步骤总数**不能超过{self.max_rounds}个**，请合理规划步骤数量。

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
        6. 步骤总数必须控制在{self.max_rounds}个以内
        7. 同一步骤内只能是一种action type
        8. 只在必要时做generation(LLM生成)，且尽量简洁
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
                print(f"[DEBUG] 生成执行计划的输出: {content[:]}")

            # 提取JSON部分
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            json_str = content[start_idx:end_idx]
            plan_data = json.loads(json_str)
            steps = plan_data.get('steps', [])

            # 强制截断超出最大轮次的步骤
            if len(steps) > self.max_rounds:
                print(f"警告：生成的执行计划包含 {len(steps)} 个步骤，超过最大轮次限制 {self.max_rounds}，已自动截断")
                steps = steps[:self.max_rounds]

            return steps
        except Exception as e:
            print(f"解析执行计划失败: {e}")
            print(f"原始内容: {content}")
            return []

    def execute_sub_step(self, sub_step: Dict, prev_step_result: Optional[str] = None,
                         prev_citations: Optional[List[int]] = None,
                         prev_action_type: Optional[str] = None,
                         round_number: Optional[int] = None) -> Dict:
        """
        执行单个子步骤
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
                    'source': f"检索结果[{item['index']}]",
                    'round_number': round_number,  # 添加轮次信息
                    'orig_item':item
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
                context_parts.append("您必须从以下检索结果中引用相关条目来回答问题，注意尽量简洁:")
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
                'used_citations': used_citations,  # 实际使用的引用
                'round_number': round_number  # 添加轮次信息
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
                     prev_action_type: Optional[str] = None,
                     round_number: Optional[int] = None) -> Dict:
        """
        执行单个步骤（包含多个子步骤）
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
            result = self.execute_sub_step(sub_step, prev_step_result, prev_citations, prev_action_type, round_number)
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
            'summary': f"步骤 {step['step_id']} 完成，共执行了 {len(sub_steps)} 个子步骤",
            'round_number': round_number  # 添加轮次信息
        }

    def get_current_citation_indices(self) -> Tuple[List[int], List[int], List[int]]:
        """
        获取当前保留的所有引用序号
        """
        # 所有引用序号
        all_indices = sorted(list(self.citation_chain.keys()))

        # 区分检索和生成类型的引用
        retrieval_indices = []
        generation_indices = []

        for idx in all_indices:
            citation_type = self.citation_chain[idx].get('type', '')
            if citation_type == 'retrieval':
                retrieval_indices.append(idx)
            elif citation_type == 'generation':
                generation_indices.append(idx)

        return all_indices, retrieval_indices, generation_indices

    def trace_used_citations_backwards(self, starting_citation_ids: List[int]) -> List[int]:
        """
        从给定的引用ID开始，回溯到所有相关的retrieval节点
        """
        visited = set()
        result_indices = set()

        def dfs(citation_id):
            if citation_id in visited:
                return
            visited.add(citation_id)

            if citation_id not in self.citation_chain:
                return

            citation_info = self.citation_chain[citation_id]

            # 如果是检索类型，加入结果
            if citation_info.get('type') == 'retrieval':
                result_indices.add(citation_id)
                return  # 到达叶子节点，停止回溯

            # 如果是生成类型，继续回溯其使用的引用
            used_citations = citation_info.get('used_citations', [])
            for used_id in used_citations:
                dfs(used_id)

        for start_id in starting_citation_ids:
            dfs(start_id)

        return sorted(list(result_indices))

    def execute_task(self, query: str, mode: str = 'train') -> Dict:
        """
        执行完整任务
        """
        print(f"开始执行任务: {query}")
        print(f"执行模式: {mode}")
        print(f"最大执行轮次限制: {self.max_rounds}")

        # 重置引用链
        self.citation_chain = {}

        # 1. 生成执行计划
        print("\n正在生成执行计划...")
        execution_plan = self.generate_execution_plan(query)

        if not execution_plan:
            return {
                'status': 'error',
                'message': '无法生成执行计划',
                'final_answer': '抱歉，无法处理您的请求',
                'last_round_result': '',
                'all_citations': {},
                'relevant_indices': [],
                'execution_info': {}
            }

        print(f"生成了 {len(execution_plan)} 个步骤的执行计划（最大限制 {self.max_rounds} 个）")

        # 2. 按步骤执行
        prev_step_result = None
        prev_citations = []
        prev_action_type = None
        all_results = {}
        all_relevant_indices = []  # 存储所有步骤的相关索引
        all_retrieval_contents = []  # 存储所有检索内容
        reached_max_rounds = False  # 标记是否达到最大轮次
        early_termination = False  # 标记是否提前终止

        for i, step in enumerate(execution_plan):
            # 记录当前轮次
            current_round = i + 1

            # 检查是否超出最大轮次
            if i + 1 > self.max_rounds:
                reached_max_rounds = True
                print(f"\n⚠️  已达到最大执行轮次限制 ({self.max_rounds})，终止执行")

                # 获取并输出当前保留的所有引用序号
                all_indices, retrieval_indices, generation_indices = self.get_current_citation_indices()
                print(f"\n📋 截止到最大轮次保留的引用序号信息:")
                print(f"   - 所有引用序号: {all_indices}")
                print(f"   - 检索类型引用序号: {retrieval_indices}")
                print(f"   - 生成类型引用序号: {generation_indices}")

                # 输出检索类型引用的具体内容
                if retrieval_indices:
                    print(f"\n📄 截止到最大轮次的检索结果详情:")
                    for idx in retrieval_indices:
                        citation_info = self.citation_chain[idx]
                        print(f"   [{idx}] {citation_info['content'][:200]}...")

                break

            step_result = self.execute_step(step, prev_step_result, prev_citations, prev_action_type, current_round)
            all_results[step['step_id']] = step_result

            print(f"  {step_result['summary']}")

            # 更新前一步骤结果和引用列表，供下一步使用
            prev_step_result = step_result['combined_result']
            prev_citations = step_result['citations']
            prev_action_type = step_result['action_type']

            # 收集所有相关索引和内容
            all_relevant_indices.extend(step_result.get('relevant_indices', []))
            all_retrieval_contents.extend(step_result.get('retrieval_contents', []))

            # 推理模式下的提前终止检查
            if mode == 'infer':
                remaining_plan = execution_plan[i+1:]  # 剩余计划
                if len(remaining_plan) > 0:  # 如果还有剩余步骤
                    # 检查当前步骤是否只有LLM生成任务
                    current_has_only_gen = True
                    for sub_step in step.get('sub_steps', []):
                        if sub_step.get('action') == 'retrieval':
                            current_has_only_gen = False
                            break

                    # 如果当前步骤只有生成任务，且后续没有检索任务，则提前终止
                    if current_has_only_gen and self.analyze_next_steps(remaining_plan):
                        print(f"\n🔍 推理模式检测到当前步骤后只有LLM生成任务，无检索任务，提前终止执行")
                        early_termination = True

                        # 获取最后一轮的引用ID
                        last_round_number = max(all_results.keys()) if all_results else 0
                        last_round_result = all_results.get(last_round_number, {})

                        # 从最后一轮的引用链中找到所有generation节点
                        last_generation_citation_ids = []
                        for citation_id, citation_info in self.citation_chain.items():
                            if citation_info.get('type') == 'generation' and citation_info.get('round_number') == last_round_number:
                                last_generation_citation_ids.append(citation_id)

                        # 对最后一轮的每个generation节点进行回溯
                        true_retrieved_indices = []
                        if last_generation_citation_ids:
                            for gen_citation_id in last_generation_citation_ids:
                                citation_info = self.citation_chain[gen_citation_id]
                                starting_citations = citation_info.get('used_citations', [])

                                if starting_citations:
                                    # 对每个generation节点的引用进行回溯
                                    retrieved_for_this_gen = self.trace_used_citations_backwards(starting_citations)
                                    true_retrieved_indices.extend(retrieved_for_this_gen)

                        # 去重
                        true_retrieved_indices = sorted(list(set(true_retrieved_indices)))

                        # 如果没有任何回溯到的检索片段，查找所有检索类型的节点
                        if not true_retrieved_indices:
                            _, retrieval_indices, _ = self.get_current_citation_indices()
                            true_retrieved_indices = retrieval_indices

                        # 生成当前结果的引用信息
                        print("\n正在生成当前结果的引用信息...")
                        final_answer, relevant_indices = self.generate_final_answer_and_relevant_indices(query, all_results, true_retrieved_indices, use_llm=False)

                        # 获取当前的引用序号信息
                        all_indices, retrieval_indices, generation_indices = self.get_current_citation_indices()

                        # 获取最后一轮结果
                        last_round_number = max(all_results.keys()) if all_results else 0
                        last_round_result = all_results.get(last_round_number, {}).get('combined_result', '')

                        return {
                            'status': 'success',
                            'execution_plan': execution_plan,
                            'all_results': all_results,
                            'final_answer': final_answer,
                            'last_round_result': last_round_result,
                            'relevant_indices': relevant_indices,
                            'true_retrieved_indices': true_retrieved_indices,  # 新增：真实引用的检索片段索引
                            'all_citations': self.citation_chain,
                            'execution_info': {
                                'max_rounds': self.max_rounds,
                                'executed_rounds': len(all_results),
                                'reached_max_rounds': reached_max_rounds,
                                'early_termination': early_termination,
                                'current_citation_indices': {
                                    'all_indices': all_indices,
                                    'retrieval_indices': retrieval_indices,
                                    'generation_indices': generation_indices
                                }
                            },
                            'message': f'推理模式提前终止，已完成 {len(all_results)} 轮'
                        }

        # 3. 生成最终答案及相关的索引
        print("\n正在生成最终答案和相关索引...")

        # 获取最后一轮的引用ID
        last_round_number = max(all_results.keys()) if all_results else 0
        last_round_result = all_results.get(last_round_number, {})

        # 从最后一轮的引用链中找到所有generation节点
        last_generation_citation_ids = []
        for citation_id, citation_info in self.citation_chain.items():
            if citation_info.get('type') == 'generation' and citation_info.get('round_number') == last_round_number:
                last_generation_citation_ids.append(citation_id)

        # 对最后一轮的每个generation节点进行回溯
        true_retrieved_indices = []
        if last_generation_citation_ids:
            for gen_citation_id in last_generation_citation_ids:
                citation_info = self.citation_chain[gen_citation_id]
                starting_citations = citation_info.get('used_citations', [])

                if starting_citations:
                    # 对每个generation节点的引用进行回溯
                    retrieved_for_this_gen = self.trace_used_citations_backwards(starting_citations)
                    true_retrieved_indices.extend(retrieved_for_this_gen)

        # 去重
        true_retrieved_indices = sorted(list(set(true_retrieved_indices)))

        # 如果没有任何回溯到的检索片段，查找所有检索类型的节点
        if not true_retrieved_indices:
            _, retrieval_indices, _ = self.get_current_citation_indices()
            true_retrieved_indices = retrieval_indices

        final_answer, relevant_indices = self.generate_final_answer_and_relevant_indices(query, all_results, true_retrieved_indices)

        # 获取最终的引用序号信息
        all_indices, retrieval_indices, generation_indices = self.get_current_citation_indices()

        # 获取最后一轮结果
        last_round_number = max(all_results.keys()) if all_results else 0
        last_round_result = all_results.get(last_round_number, {}).get('combined_result', '')

        return {
            'status': 'success',
            'execution_plan': execution_plan,
            'all_results': all_results,
            'final_answer': final_answer,
            'last_round_result': last_round_result,
            'relevant_indices': relevant_indices,
            'true_retrieved_indices': true_retrieved_indices,  # 新增：真实引用的检索片段索引
            'all_citations': self.citation_chain,
            'execution_info': {
                'max_rounds': self.max_rounds,
                'executed_rounds': len(all_results),
                'reached_max_rounds': reached_max_rounds,
                'early_termination': early_termination,
                'current_citation_indices': {
                    'all_indices': all_indices,
                    'retrieval_indices': retrieval_indices,
                    'generation_indices': generation_indices
                }
            },
            'message': '任务执行完成' if len(all_results) <= self.max_rounds and not reached_max_rounds
                       else f'任务执行终止（已达到最大轮次限制 {self.max_rounds}）'
        }

    def generate_final_answer_and_relevant_indices(self, original_query: str, all_results: Dict, true_retrieved_indices: List[int] = None, use_llm: bool = True) -> Tuple[str, List[int]]:
        """
        根据所有步骤结果生成最终答案，并返回最相关的检索结果索引
        :param original_query: 原始查询
        :param all_results: 所有步骤结果
        :param true_retrieved_indices: 真实引用的检索片段索引（由回溯算法得出）
        :param use_llm: 是否调用LLM生成（infer模式最后一轮设为False）
        :return: 最终答案, 相关索引列表
        """
        # ===== 新增：infer模式跳过LLM，直接基于引用链生成 =====
        if not use_llm:
            # 1. 提取所有检索类型的引用索引（即relevant_indices）
            _, retrieval_indices, _ = self.get_current_citation_indices()
            relevant_indices = retrieval_indices

            # 2. 拼接所有相关检索内容作为最终答案
            final_answer_parts = [f"以下是与「{original_query}」相关的检索内容："]
            for idx in relevant_indices:
                citation_info = self.citation_chain[idx]
                final_answer_parts.append(f"[{idx}] {citation_info['content']}")

            final_answer = "\n".join(final_answer_parts)
            return final_answer, relevant_indices
        # ===== 原有LLM调用逻辑（保留）=====
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
            round_number = citation_info.get('round_number', 'Unknown')
            parent_refs = f" (来自引用: {parent_citations})" if parent_citations else ""
            used_refs = f" (使用了引用: {used_citations})" if used_citations else ""
            round_info = f" (第{round_number}轮)" if round_number != 'Unknown' else ""
            citation_context += f"[{idx}] {citation_info['content']} [来源: {source_info}{parent_refs}{used_refs}{round_info}]\n\n"

        # 构建最终提示
        if true_retrieved_indices:
            # 如果提供了真实引用的检索片段索引，强调这些内容
            true_retrieved_context = "真实引用的检索片段:\n"
            for idx in true_retrieved_indices:
                if idx in self.citation_chain:
                    citation_info = self.citation_chain[idx]
                    true_retrieved_context += f"[{idx}] {citation_info['content']}\n\n"

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
