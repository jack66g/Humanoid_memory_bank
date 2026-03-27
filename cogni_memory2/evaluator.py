# evaluator.py
import re
from typing import Callable, Tuple

class CSNEvaluator:
    def __init__(self, llm_generate_fn: Callable[[str], str], custom_s_desc: str = None):
        """
        专职负责 C(通顺度)、S(价值)、N(噪声) 的客观评估。
        """
        self.llm_generate_fn = llm_generate_fn
        # ==========================================
        # 【物理逻辑】：S 默认留空，不再硬塞提示词
        # ==========================================
        self.s_desc = custom_s_desc if custom_s_desc else ""

    def update_s_desc(self, new_desc: str):
        """
        支持外部动态修改 S。如果传入空，则物理关闭 S 评估。
        """
        self.s_desc = new_desc if new_desc else ""

    def detect_conflict(self, old_text: str, new_text: str) -> bool:
        """
        逻辑裁判员。判断新输入是否与旧记忆产生逻辑冲突。
        """
        prompt = (
            f"<|im_start|>system\n"
            f"你是一个极其冷酷的逻辑裁判员。你的唯一任务是判断【新输入】是否在推翻、修改或否定【旧记忆】。\n"
            f"规则：\n"
            f"1. 如果新输入表达了与旧记忆相反的观点、状态改变或明确的修改意图，输出 1。\n"
            f"2. 如果新输入只是补充细节，或者两者不冲突，输出 0。\n"
            f"绝对不要输出任何多余的废话，只输出 1 或 0。<|im_end|>\n"
            f"<|im_start|>user\n"
            f"【旧记忆】：{old_text}\n"
            f"【新输入】：{new_text}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        try:
            res = self.llm_generate_fn(prompt).strip()
            return "1" in res 
        except Exception as e:
            print(f"⚠️ [冲突检测异常]: {e}")
            return False

    def evaluate(self, text: str) -> Tuple[float, float, float]:
        """
        【物理开关逻辑】：
        - 如果 s_desc 为空：不评估 S，模型只输出 (C, N)，S 在程序里固定为 1.0。
        - 如果 s_desc 不空：模型输出 (C, S, N)，S 由模型评估。
        """
        
        if not self.s_desc:
            # -------------------------------------------------------
            # 模式 A：平权模式（不评估价值，S 固定为 1.0）
            # -------------------------------------------------------
            prompt = (
                f"<|im_start|>system\n"
                f"你是一个客观、冷酷的数据评估器。请对下面这段文本进行评分（0到10分）：\n"
                f"C (通顺度): 语句是否通顺且逻辑连贯。\n"
                f"N (噪声): 是否包含无意义的废话、单纯的语气词或乱码（越高代表越废话）。\n"
                f"请严格只输出两个数字（C,N），用英文逗号分隔，例如：8,2。\n"
                f"绝对不要输出任何其他解释或符号！<|im_end|>\n"
                f"<|im_start|>user\n文本：\n{text}\n<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            try:
                res = self.llm_generate_fn(prompt)
                match = re.search(r'(\d+)\s*,\s*(\d+)', res)
                if match:
                    # 返回时物理强制 S = 1.0
                    return float(match.group(1)), 1.0, float(match.group(2))
            except Exception as e:
                print(f"⚠️ [评估器模式A异常]: {e}")
            return 5.0, 1.0, 8.0 # 兜底

        else:
            # -------------------------------------------------------
            # 模式 B：评估模式（S 参与评估）
            # -------------------------------------------------------
            prompt = (
                f"<|im_start|>system\n"
                f"你是一个客观、冷酷的数据评估器。请对下面这段文本进行评分（0到10分）：\n"
                f"C (通顺度): 语句是否通顺且逻辑连贯。\n"
                f"S (绝对价值): {self.s_desc}\n"
                f"N (噪声): 是否包含无意义的废话、单纯的语气词或乱码（越高代表越废话）。\n"
                f"请严格只输出三个数字（C,S,N），用英文逗号分隔，例如：8,9,2。\n"
                f"绝对不要输出任何其他解释或符号！<|im_end|>\n"
                f"<|im_start|>user\n文本：\n{text}\n<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            try:
                res = self.llm_generate_fn(prompt)
                match = re.search(r'(\d+)\s*,\s*(\d+)\s*,\s*(\d+)', res)
                if match:
                    return float(match.group(1)), float(match.group(2)), float(match.group(3))
            except Exception as e:
                print(f"⚠️ [评估器模式B异常]: {e}")
            return 5.0, 1.0, 8.0 # 兜底