# app.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# 导入你的认知核心
from cogni_memory import CogniCore

# ---------------------------------------------------------
# 1. 准备本地 Embedding 模型和 API 配置
# ---------------------------------------------------------
# 这里我们用开源极速模型，保证本地隐私和零成本运行
from sentence_transformers import SentenceTransformer
print("加载本地海马体晶体 (Embedding Model)...")
local_emb_model = SentenceTransformer('BAAI/bge-small-zh-v1.5') 

# 配置你的商业大模型底座 API (可换成千问、Kimi、DeepSeek等)
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# 初始化唯一的 CogniCore 实例 (全局单例)
engine = CogniCore(
    api_endpoint=API_ENDPOINT,
    api_key=API_KEY,
    embedding_model=local_emb_model
)

# ---------------------------------------------------------
# 2. 定义 FastAPI 应用与数据模型 (RESTful 规范)
# ---------------------------------------------------------
app = FastAPI(title="CogniCore Memory API", description="小乐的数字生命记忆中间件", version="1.0.0")

class PerceiveRequest(BaseModel):
    user_text: str
    ai_text: str

class RecallRequest(BaseModel):
    query: str
    top_k: Optional[int] = 2

class ConfigUpdateRequest(BaseModel):
    w_c: Optional[float] = None
    w_s: Optional[float] = None
    w_n: Optional[float] = None
    w_m: Optional[float] = None

# ---------------------------------------------------------
# 3. 对外暴露的 HTTP 接口
# ---------------------------------------------------------

@app.post("/api/v1/memory/perceive")
async def api_perceive(req: PerceiveRequest, background_tasks: BackgroundTasks):
    """
    感知接口：将对话压入待处理队列，并利用后台任务执行深度睡眠(打分入库)，不阻塞前端响应。
    """
    engine.perceive(req.user_text, req.ai_text)
    
    # 商业化技巧：用 FastAPI 的后台任务执行代谢，让接口毫秒级返回！
    background_tasks.add_task(engine.sleep)
    return {"status": "success", "message": "记忆已捕获，正在潜意识中新陈代谢"}

@app.post("/api/v1/memory/recall")
async def api_recall(req: RecallRequest):
    """
    回想接口：极速抽取当前对话相关的高维记忆。
    """
    memories = engine.recall(req.query, req.top_k)
    return {"status": "success", "data": memories}

@app.post("/api/v1/memory/config")
async def api_update_config(req: ConfigUpdateRequest):
    """
    调参接口：提供给前端 UI 滑条，动态改变 AI 的性格和固执度。
    """
    # 过滤掉为 None 的参数
    # 新代码（完美适配 Pydantic V2）
    update_data = req.model_dump(exclude_none=True)
    if update_data:
        engine.update_config(update_data)
    return {"status": "success", "message": "内核参数已热更新", "current_config": update_data}

@app.delete("/api/v1/memory/wipe")
async def api_wipe():
    """
    格式化接口：慎用！
    """
    engine.wipe()
    return {"status": "success", "message": "全部记忆已抹除"}

# 启动命令: uvicorn app:app --reload --port 8000