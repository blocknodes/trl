from pydantic import BaseModel
from typing import List, Dict, Optional

# 请求体模型
class TaskRequest(BaseModel):
    query: str
    max_rounds: int = 2
    mode: str = "infer"

# 响应体模型
class TaskResponse(BaseModel):
    status: str
    final_answer: str
    last_round_result: str
    all_citations: Dict[int, Dict]
    relevant_indices: List[int]
    execution_info: Dict
    message: str