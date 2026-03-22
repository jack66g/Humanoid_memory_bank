# utils/math_utils.py
import math

def calculate_v_score(C: float, S: float, N: float, M: float, 
                      w_c: float, w_s: float, w_n: float, w_m: float) -> float:
    """
    核心公式：V = 1 / (1 + e^-(wc*C + ws*S + wm*M - wn*N))
    纯粹的标量数学运算，完全脱离张量与框架依赖。
    【保留核心】绝对保留了极高/极低 M 值的非线性惩罚/打破回声室逻辑。
    """
    norm_c = (C - 5) / 2.0
    norm_s = (S - 5) / 2.0
    norm_n = (N - 5) / 2.0
    
    # 极度精妙的反回声室惩罚
    if M > 9.0:
        norm_m = -2.0  
    elif M < 3.0:
        norm_m = -1.0  
    else:
        norm_m = (M - 5) / 2.0 
    
    logit = (w_c * norm_c) + (w_s * norm_s) + (w_m * norm_m) - (w_n * norm_n)
    return 1.0 / (1.0 + math.exp(-logit))

def calculate_ebbinghaus_decay_factor(dt_hours: float, forget_rate: float) -> float:
    """
    艾宾浩斯遗忘曲线衰减因子计算
    提取自原逻辑: math.exp(-self.forget_rate * dt_hours)
    """
    return math.exp(-forget_rate * dt_hours)

def calculate_sim_score(distance: float) -> float:
    """
    距离到相似度的转换公式
    提取自原逻辑: 1.0 / (1.0 + dist)
    """
    return 1.0 / (1.0 + distance)