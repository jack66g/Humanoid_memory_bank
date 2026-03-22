# evaluator.py
import re
from typing import Callable, Tuple

class CSNEvaluator:
    def __init__(self, llm_generate_fn: Callable[[str], str], custom_s_desc: str = None):
        """
        专职负责 C(通顺度)、S(价值)、N(噪声) 的客观评估。
        解耦了具体的 LLM 底座，只需传入一个能接收 prompt 并返回文本结果的函数即可。
        """
        self.llm_generate_fn = llm_generate_fn

        # ==========================================
        # 【核心修改】：按你的要求设定 S 的判定逻辑
        # ==========================================
        if custom_s_desc is None or custom_s_desc.strip() == "":
            # 1. 如果没有提示词（没有设定），直接让大模型打 1 分
            self.s_desc = "用户未设定具体判定标准，请固定给此项打 1 分。"
        else:
            # 2. 如果有提示词设定，就是原来的样子（按用户的提示词去评估）
            self.s_desc = custom_s_desc

    def evaluate(self, text: str) -> Tuple[float, float, float]:
        """
        【重大修复】剥离 M 因子，让 LLM 只做客观的 C、S、N 评价
        原汁原味保留了系统指令和严格的输出格式约束。
        """
        prompt = (
            f"<|im_start|>system\n"
            f"你是一个客观、冷酷的数据评估器。请对下面这段文本进行评分（0到10分）：\n"
            f"C (通顺度): 语句是否通顺且逻辑连贯。\n"
            f"S (绝对价值): {self.s_desc}\n"  # <=== 动态插入指令
            f"N (噪声): 是否包含无意义的废话、单纯的语气词或乱码（越高代表越废话）。\n"
            f"请严格只输出三个数字（C,S,N），用英文逗号分隔，例如：8,9,2。\n"
            f"绝对不要输出任何其他解释或符号！<|im_end|>\n"
            f"<|im_start|>user\n文本：\n{text}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        # 调用外部注入的大模型生成能力
        try:
            res = self.llm_generate_fn(prompt)
        except Exception as e:
            print(f"⚠️ [评估器异常] 模型调用失败: {e}")
            res = "" # 触发底层的容错兜底
        
        # 核心逻辑完全保留：精准的正则提取
        match = re.search(r'(\d+)\s*,\s*(\d+)\s*,\s*(\d+)', res)
        if match:
            return float(match.group(1)), float(match.group(2)), float(match.group(3))
            
        # 容错兜底值：如果 LLM 抽风乱答，默认判定为高噪声、低价值，防止垃圾数据污染数据库
        return 5.0, 1.0, 8.0