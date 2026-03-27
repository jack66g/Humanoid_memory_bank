# __init__.py
import time
import requests
from typing import List, Dict, Any

# 导入所有拆解好的核心器官
from .config import CogniConfig
from .memory_store import MemoryStore
from .evaluator import CSNEvaluator
from .physics import CognitivePhysics
from .engine import CognitiveEngine
from .utils.math_utils import calculate_v_score

class CogniCore:
    def __init__(
        self, 
        api_endpoint: str,       # 【改动】：接收大模型 API 请求地址
        api_key: str,            # 【改动】：接收大模型 API 密钥
        embedding_model: Any,    # 专属的海马体（向量）模型插槽（保持本地运行）
        model_name: str = "gpt-3.5-turbo", # 【新增】：动态接收模型名字！默认兜底 gpt-3.5-turbo
        db_path: str = "cogni_memory_store", 
        custom_s_desc: str = None,
        custom_config: CogniConfig = None
    ):
        """
        认知皮层引擎 V3 - 商业化 RESTful API 版。
        彻底解耦大模型底座，支持动态调参和任意兼容 OpenAI 格式的云端大模型。
        """
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_name = model_name # 【新增】：把模型名字存下来
        
        # 如果用户传了配置，就用用户的；没传就用默认的。
        if custom_config is not None:
            self.config = custom_config
            self.config.db_store_path = db_path
        else:
            self.config = CogniConfig(db_store_path=db_path)
            
        self.store = MemoryStore(self.config)
        self.physics = CognitivePhysics(self.config, embedding_model=embedding_model)
        
        # ==========================================
        # 【改动】：重写为标准的 HTTP 请求发送器
        # ==========================================
        def api_llm_generate(prompt: str) -> str:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model_name, # <--- 【核心改动】：动态读取！
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 10
            }
            try:
                response = requests.post(self.api_endpoint, headers=headers, json=payload, timeout=15)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                print(f"⚠️ [API 调用失败]: {e}")
                return "5, 1, 8" # 降级容错
            
        # 把网络请求函数透传给评估器
        self.evaluator = CSNEvaluator(
            llm_generate_fn=api_llm_generate, 
            custom_s_desc=custom_s_desc
        )
        
        def calc_v_wrapper(C: float, S: float, N: float, M: float) -> float:
            return calculate_v_score(
                C, S, N, M, 
                self.config.w_c, self.config.w_s, self.config.w_n, self.config.w_m
            )
            
        self.engine = CognitiveEngine(
            config=self.config,
            db_collection=self.store.get_collection(),
            get_emb_fn=self.physics.get_emb,
            eval_csn_fn=self.evaluator.evaluate,
            calc_m_fn=self.physics.calculate_physical_m,
            calc_v_fn=calc_v_wrapper
        )

    # ==========================================
    # 动态调参接口 (为后续前端 UI 滑条准备)
    # ==========================================
    def update_config(self, new_params: dict):
        """【新增】：支持外部随时修改底层参数，例如改变 AI 的性格和固执度"""
        for k, v in new_params.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        print(f"⚙️ [内核配置已动态热更新]: {new_params}")

    def update_evaluator_desc(self, new_desc: str):
        """【新增】：支持外部动态修改 S 绝对价值的提示词（价值观重塑）"""
        self.evaluator.update_s_desc(new_desc)
        print(f"🧠 [价值观已重塑]: {self.evaluator.s_desc}")

    # ==========================================
    # 对外暴露的极简 API
    # ==========================================
    def perceive(self, user_text: str, ai_text: str):
        self.engine.add_to_pending(user_text, ai_text)
        
    def recall(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        return self.engine.retrieve_memory(query, top_k)
        
    def sleep(self):
        self.engine.process_pending_queue()
        
    def wipe(self):
        if self.store.wipe_all_memories():
            print("💣 轰！所有长期记忆已被强行抹除。")

    # ==========================================
    # 交互终端 (也已替换为调用 API)
    # ==========================================
    def run_cli(self):
        import datetime
        print("\n" + "="*65)
        print("🚀 认知皮层引擎 V3 (全自动记忆代谢 API 版) 已启动！")
        print("="*65 + "\n")

        while True:
            user_input = input("\n🧑 用户: ")
            if not user_input.strip(): continue
            if user_input.strip().lower() in ['q', 'quit', 'exit']: break

            retrieved_mems = self.recall(user_input, top_k=2)
            memory_context = ""
            if retrieved_mems:
                print(f"  [🧠 提取了 {len(retrieved_mems)} 条高维关联记忆]")
                for m in retrieved_mems:
                    dt_str = datetime.datetime.fromtimestamp(m['meta']['timestamp']).strftime('%Y-%m-%d %H:%M')
                    ai_reply = m['meta'].get('ai_text', '无记录')
                    memory_context += f"[{dt_str}]\n用户当时说：{m['text']}\n你当时回答：{ai_reply}\n\n"
            else:
                memory_context = "无。"

            prompt = (
                f"你是一个有自我意志的实体。请根据历史记忆回答。\n"
                f"【触发的长期记忆】：\n{memory_context}\n"
                f"【用户当前输入】：{user_input}"
            )
            
            # 使用云端大模型 API 生成回复
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            payload = {
                "model": self.model_name, # <--- 【核心改动】：动态读取！
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 200
            }
            try:
                res = requests.post(self.api_endpoint, headers=headers, json=payload)
                res.raise_for_status()
                response_text = res.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                response_text = f"[API 生成异常]: {e}"
            
            print(f"🤖 AI: {response_text}")
            
            self.perceive(user_text=user_input, ai_text=response_text)
            self.sleep()