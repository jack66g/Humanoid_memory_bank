# __init__.py
import time
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
        model: Any, 
        tokenizer: Any, 
        embedding_model: Any,  # 【改动】：新增专属的海马体（向量）模型插槽
        db_path: str = "cogni_memory_store", 
        custom_s_desc: str = None,
        custom_config: CogniConfig = None
    ):
        """
        认知皮层引擎 V3 - 顶级 API 入口。
        只需要传入大模型底座和分词器，系统会自动完成所有物理器官的组装。
        """
        # 如果用户传了配置，就用用户的；没传就用默认的。
        if custom_config is not None:
            self.config = custom_config
            self.config.db_store_path = db_path
        else:
            self.config = CogniConfig(db_store_path=db_path)
            
        self.store = MemoryStore(self.config)
        
        # 【改动】：初始化物理引擎（只挂载专业的 embedding_model 提取向量）
        self.physics = CognitivePhysics(self.config, embedding_model=embedding_model)
        
        # 封装默认的大模型生成函数（用于给评估器打分，这里依然用 Qwen 底座）
        def default_llm_generate(prompt: str) -> str:
            import torch
            device = next(model.parameters()).device
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=10, temperature=0.1)
            return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
        # 把参数透传给评估器
        self.evaluator = CSNEvaluator(
            llm_generate_fn=default_llm_generate, 
            custom_s_desc=custom_s_desc
        )
        
        # 包装纯数学公式
        def calc_v_wrapper(C: float, S: float, N: float, M: float) -> float:
            return calculate_v_score(
                C, S, N, M, 
                self.config.w_c, self.config.w_s, self.config.w_n, self.config.w_m
            )
            
        # 组装调度引擎（大脑中枢）
        self.engine = CognitiveEngine(
            config=self.config,
            db_collection=self.store.get_collection(),
            get_emb_fn=self.physics.get_emb,
            eval_csn_fn=self.evaluator.evaluate,
            calc_m_fn=self.physics.calculate_physical_m,
            calc_v_fn=calc_v_wrapper
        )
        
        # 保存底座，供外部聊天生成使用
        self.model = model
        self.tokenizer = tokenizer

    # ==========================================
    # 对外暴露的极简 API (开发者直接调用的方法)
    # ==========================================
    # 【保留】：接收双轨数据（分离用户文本和AI文本）
    def perceive(self, user_text: str, ai_text: str):
        """感知：将用户的输入和AI的回答分离存入待处理队列"""
        self.engine.add_to_pending(user_text, ai_text)
        
    def recall(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """回想：提取高维关联记忆"""
        return self.engine.retrieve_memory(query, top_k)
        
    def sleep(self):
        """代谢：触发底层的清洗、打分、张量共鸣与融合"""
        self.engine.process_pending_queue()
        
    def wipe(self):
        """格式化：切除所有长期记忆 (保留此 API 供代码级调用，CLI 中不再暴露)"""
        if self.store.wipe_all_memories():
            print("💣 轰！所有长期记忆已被强行抹除。")

    # ==========================================
    # 原汁原味的交互终端 (全自动无感记忆版)
    # ==========================================
    def run_cli(self):
        import torch
        import datetime  # 引入 datetime 模块处理时间戳
        
        print("\n" + "="*65)
        print("🚀 认知皮层引擎 V3 (全自动记忆代谢版 - 键值分离架构) 已启动！")
        print("💡 对话结束后系统将自动进行记忆打分、张量共鸣与艾宾浩斯代谢。")
        print("="*65 + "\n")

        while True:
            user_input = input("\n🧑 用户: ")
            
            # 【保留】：防御空字符串刺客！
            if not user_input.strip(): continue
            if user_input.strip().lower() in ['q', 'quit', 'exit']: break

            # 检索记忆
            retrieved_mems = self.recall(user_input, top_k=2)
            memory_context = ""
            if retrieved_mems:
                print(f"  [🧠 提取了 {len(retrieved_mems)} 条高维关联记忆并执行巩固]")
                for m in retrieved_mems:
                    # 动态提取时间戳并拼接为可读前缀
                    dt_str = datetime.datetime.fromtimestamp(m['meta']['timestamp']).strftime('%Y-%m-%d %H:%M')
                    
                    # 【保留】：完美还原历史对话场景（从负荷中解包 AI 的话）
                    ai_reply = m['meta'].get('ai_text', '无记录')
                    memory_context += f"[{dt_str}]\n用户当时说：{m['text']}\n你当时回答：{ai_reply}\n\n"
            else:
                memory_context = "无。"

            # 组装 Prompt 并生成回复
            prompt = (
                f"<|im_start|>system\n你是一个有自我意志的实体。\n"
                f"【触发的长期记忆】：\n{memory_context}\n"
                f"请结合记忆回答，若无关则忽略。<|im_end|>\n"
                f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
            )
            
            device = next(self.model.parameters()).device
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=200, temperature=0.7)
            response_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            print(f"🤖 AI: {response_text}")
            
            # ==========================================
            # 【保留】：调用双轨感知，分别存入 Key 和 Value
            # ==========================================
            self.perceive(user_text=user_input, ai_text=response_text)
            
            # 对话完毕后，立刻全自动触发记忆新陈代谢！
            self.sleep()