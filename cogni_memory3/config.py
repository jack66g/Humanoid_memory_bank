# config.py
import os
from dataclasses import dataclass

@dataclass
class CogniConfig:
    """
    认知皮层引擎 V3 - 核心配置中心
    （注：默认参数严格遵循原始物理与数学规则，请谨慎修改核心权重）
    """
    
    # ==========================================
    # 1. 存储配置
    # ==========================================
    db_store_path: str = "cogni_memory_store"
    collection_name: str = "episodic_memory"
    
    # ==========================================
    # 2. 修正版权重：打破“回声室”，价值 S 称王！
    # ==========================================
    w_c: float = 1.0  # C (Context) 通顺度
    w_s: float = 1.8  # S (Significance) 绝对价值 (最高权重，允许新知识强行突围)
    w_n: float = 1.2  # N (Noise) 噪声惩罚
    w_m: float = 0.8  # M (Memory Resonance) 物理共鸣附加分 (调低，防止故步自封)
    
    # ==========================================
    # 3. 物理代谢与阈值参数
    # ==========================================
    v_threshold: float = 0.50          # 入库及格线微调 (V_score 大于此值才存入长期记忆)
    forget_rate: float = 0.005         # 衰减系数 (控制艾宾浩斯遗忘曲线的陡峭程度)
    death_threshold: float = 0.1       # 物理删除阈值 (初始价值 * 衰减因子 小于此值将被物理抹除)
    adulthood_threshold: int = 3       # 开启共鸣计算的阈值
    
    # ==========================================
    # 4. 张量共鸣 (M) 判定逻辑阈值 (提取自代码核心循环)
    # ==========================================
    m_consolidation_threshold: float = 8.5  # 达到此值触发“巩固”或“融合”
    m_extreme_repeat_threshold: float = 9.5 # 达到此值判定为“极度重复”，只巩固不新增文本