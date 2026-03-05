from fastapi import FastAPI, HTTPException
import uvicorn
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
kbp_client = KbpRetrievalClient(
    base_url=KBP_CONFIG["base_url"],
    user_key=KBP_CONFIG["user_key"],
    api_key=KBP_CONFIG["api_key"]
)

# API 接口
@app.post("/execute_task", response_model=TaskResponse)
async def execute_task_api(request: TaskRequest):
    """
    执行 KBP 混合检索任务

    - **query**: 用户的查询语句
    - **max_rounds**: 最大执行轮次（默认2）
    - **mode**: 执行模式（infer/train，默认infer）
    """
    try:
        # 创建任务执行系统实例
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

        # 构造简化的响应（重点返回最后一轮结果和所有引用）
        response = {
            "status": result["status"],
            "final_answer": result["final_answer"],
            "last_round_result": result["last_round_result"],
            "all_citations": result["all_citations"],
            "relevant_indices": result["relevant_indices"],
            "execution_info": result["execution_info"],
            "message": result["message"]
        }

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