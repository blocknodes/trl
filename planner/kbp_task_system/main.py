from fastapi import FastAPI, HTTPException, Header
import uvicorn
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from schemas.models import TaskRequest, TaskResponse
from clients.llm_client import SimpleLLMClient
from clients.kbp_client import KbpRetrievalClient
from core.task_executor import TaskExecutionSystem
from config.settings import LLM_CONFIGS, KBP_CONFIG, SERVER_CONFIG, DEFAULT_LLM_MODEL, DEFAULT_MAX_ROUNDS

# 创建 FastAPI 应用实例
app = FastAPI(
    title="KBP 混合检索任务执行系统",
    description="基于 KBP 检索和 LLM 生成的多步骤任务执行接口",
    version="1.0.0"
)

# 初始化客户端
llm_client = SimpleLLMClient(llm_configs=LLM_CONFIGS, default_llm=DEFAULT_LLM_MODEL)


# 定义请求模型
class RetrievalSetting(BaseModel):
    top_k: int = 10
    score_threshold: float = 0.0
    search_mode: str = "hybrid"
    search_strategy: str = "broad"

class TaskRequest(BaseModel):
    query: str
    retrieval_setting: RetrievalSetting = RetrievalSetting()
    tracingModel: bool = False
    max_rounds: int = 2
    mode: str = "infer"

# 定义响应模型
class Metadata(BaseModel):
    path: str
    kind: str
    description: str
    document_id: str
    tagInfo: str

class Record(BaseModel):
    score: float
    from_query: str
    metadata: Metadata
    category_path: str
    file_name: str
    knowledge_path: List[str]
    category_type: str
    source_id: str
    knowledge_id: str
    title: str
    content: str
    url: str

class TaskResponse(BaseModel):
    ts: int = 1
    records: List[Record]
    code : str = '0'
    msg : str = '操作成功'
    alert: str = '0'




# API 接口
@app.post("/execute_task", response_model=Dict[str, Any])
async def execute_task_api(
        request: TaskRequest,
        api_key: str = Header(..., alias="api-key")
    ):
    """
    执行 KBP 混合检索任务

    - **query**: 用户的查询语句
    - **retrieval_setting**: 检索设置（top_k, score_threshold, search_mode, search_strategy）
    - **tracingModel**: 是否追踪模型（默认False）
    - **max_rounds**: 最大执行轮次（默认2）
    - **mode**: 执行模式（infer/train，默认infer）
    """
    try:
        # 创建任务执行系统实例
        # 初始化客户端
        kbp_client = KbpRetrievalClient(
            base_url=KBP_CONFIG["base_url"],
            user_key=KBP_CONFIG["user_key"],
            api_key=api_key
        )

        task_system = TaskExecutionSystem(
            llm_client=llm_client,
            kbp_client=kbp_client,
            debug=True,
            max_rounds=request.max_rounds
        )

        # 执行任务
        result = task_system.execute_task(
            query=request.query,
            mode=request.mode
        )
        #import pdb;pdb.set_trace()
        records = []
        true_retrieved_indices = result['true_retrieved_indices']
        for idx in true_retrieved_indices:
            item = result['all_citations'][idx]
            records.append(item['orig_item'])
        #import pdb;pdb.set_trace()
        response = {
            "ts": -100,
            "code": '0',
            "msg": '操作成功',
            "alert": '0',
            "records":records,
            "final_answer": result["final_answer"],
        }
        # 构造响应




        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"执行任务时发生错误: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy", "service": "KBP Task Execution System"}

# 启动服务
if __name__ == "__main__":
    uvicorn.run(
        app="main:app",
        host=SERVER_CONFIG["host"],
        port=SERVER_CONFIG["port"],
        reload=SERVER_CONFIG["reload"],
        log_level=SERVER_CONFIG["log_level"]
    )
