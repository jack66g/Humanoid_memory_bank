# physics.py
import math
import torch
import torch.nn.functional as F
from typing import List, Any
from .config import CogniConfig

class CognitivePhysics:
    def __init__(self, config: CogniConfig, tokenizer: Any = None, base_llm: Any = None):
        """
        专职负责核心算法与纯物理数学计算。
        包含：Mean Pooling 全局语义提取、M值物理共鸣映射、V值数学坍缩。
        如果用户自带独立的 Embedding 模型，可以不传 tokenizer 和 base_llm。
        """
        self.config = config
        self.tokenizer = tokenizer
        self.base_llm = base_llm

    def get_emb(self, text: str) -> List[float]:
        """
        【保留核心】使用 Mean Pooling 获取全局语义张量
        """
        if not self.tokenizer or not self.base_llm:
            raise ValueError("未注入大模型底座，无法执行原生 Mean Pooling 提取。")

        # 动态获取模型所在设备，代替硬编码的 "cuda"，增强插件的跨设备兼容性
        device = next(self.base_llm.parameters()).device
        
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            # 提取最后一层隐藏状态
            hidden_states = self.base_llm(**inputs, output_hidden_states=True).hidden_states[-1]
            
            # [Batch, Seq_len, Hidden_dim] -> Mean Pooling 沿 Seq_len 维度求平均
            sentence_emb = hidden_states.mean(dim=1).squeeze(0)
            
            # 必须进行 L2 归一化，保证计算余弦相似度的绝对精准
            sentence_emb = F.normalize(sentence_emb, p=2, dim=0)
            
        return sentence_emb.to(torch.float32).cpu().numpy().tolist()

    def calculate_physical_m(self, new_vec: List[float], retrieved_vecs: List[List[float]]) -> float:
        """
        纯物理共鸣计算：张量夹角映射
        【保留核心】避开多维张量隐式布尔判断
        """
        if retrieved_vecs is None or len(retrieved_vecs) == 0:
            return 5.0 
            
        t_new = torch.tensor(new_vec)
        t_olds = torch.tensor(retrieved_vecs)
        
        # 批量计算余弦相似度
        sims = F.cosine_similarity(t_new.unsqueeze(0), t_olds)
        max_sim = sims.max().item()
        
        # 映射公式：[-1, 1] 的相似度映射到 [0, 10] 的 M 评分
        m_score = (max_sim + 1.0) * 5.0 
        
        return max(0.0, min(10.0, m_score))

    def calculate_v_score(self, C: float, S: float, N: float, M: float) -> float:
        """
        核心公式：V = 1 / (1 + e^-(wc*C + ws*S + wm*M - wn*N))
        【保留核心】绝对保留了极高/极低 M 值的非线性惩罚/奖励逻辑
        """
        norm_c = (C - 5) / 2.0
        norm_s = (S - 5) / 2.0
        norm_n = (N - 5) / 2.0
        
        # 这里是你设计的非常精妙的“反回声室”非线性惩罚逻辑，原封不动！
        if M > 9.0:
            norm_m = -2.0  
        elif M < 3.0:
            norm_m = -1.0  
        else:
            norm_m = (M - 5) / 2.0 
        
        # 动态读取 config.py 中的权重参数
        logit = (self.config.w_c * norm_c) + \
                (self.config.w_s * norm_s) + \
                (self.config.w_m * norm_m) - \
                (self.config.w_n * norm_n)
                
        return 1.0 / (1.0 + math.exp(-logit))