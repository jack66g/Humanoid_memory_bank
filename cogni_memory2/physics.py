# physics.py
import math
import torch
import torch.nn.functional as F
from typing import List, Any
from .config import CogniConfig

class CognitivePhysics:
    def __init__(self, config: CogniConfig, embedding_model: Any = None):
        """
        专职负责核心算法与纯物理数学计算。
        【改动】：剥离了生成式大模型，现在专属接收轻量级的 Embedding 模型（海马体晶体）。
        """
        self.config = config
        self.embedding_model = embedding_model

    def get_emb(self, text: str) -> List[float]:
        """
        【改动】：使用专属 Embedding 模型获取纯净的高维向量。
        极大提升了语义区分度，彻底解决了大模型的“表示退化”和向量空间挤压问题。
        """
        if not self.embedding_model:
            raise ValueError("未注入 Embedding 模型，无法执行向量提取。")

        # BGE 极速编码，自带 L2 归一化，确保余弦相似度计算绝对精准
        vec = self.embedding_model.encode(text, normalize_embeddings=True)
        
        # 转换回标准的 Python List[float] 格式
        return vec.tolist()

    def calculate_physical_m(self, new_vec: List[float], retrieved_vecs: List[List[float]]) -> float:
        """
        纯物理共鸣计算：张量夹角映射
        【微调】：适配专业 Embedding 模型的相似度分布
        """
        if retrieved_vecs is None or len(retrieved_vecs) == 0:
            return 5.0 
            
        t_new = torch.tensor(new_vec)
        t_olds = torch.tensor(retrieved_vecs)
        
        # 批量计算余弦相似度
        sims = F.cosine_similarity(t_new.unsqueeze(0), t_olds)
        max_sim = sims.max().item()
        
        # 映射公式：专业模型的余弦相似度通常在 0 到 1 之间区分度极大。
        # 所以直接将其乘以 10 映射到 [0, 10] 的 M 评分，完美契合 config 的阈值设计。
        m_score = max_sim * 10.0 
        
        return max(0.0, min(10.0, m_score))

    def calculate_v_score(self, C: float, S: float, N: float, M: float) -> float:
        """
        核心公式：V = 1 / (1 + e^-(wc*C + ws*S + wm*M - wn*N))
        【完全保留】：你原汁原味的数学坍缩公式与非线性惩罚逻辑，一行未动！
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