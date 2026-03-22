# test_run.py
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 导入我们自己写的插件！
from cogni_memory import CogniCore

def main():
    # 填入你电脑上 Qwen 真实的本地路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "..", "mode") # 替换成你的真实路径
    
    print("正在加载底座模型，准备注入海马体...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_llm = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.float16
    ).eval()
    
    # 冻结参数，省显存
    for param in base_llm.parameters(): 
        param.requires_grad = False

    print("✅ 底座加载完成！开始组装认知引擎...")
    
    # ==========================================
    # 【核心修改】：定义“闲聊陪伴”专属的 S (绝对价值) 评价标准
    # ==========================================
    chat_s_desc = "是否包含用户的个人情绪、生活琐事、日常闲聊、人际关系或偏好习惯。只要是自然的日常交流和情感表达，请务必给予高分。"
    
    # 2. 一键实例化你的认知引擎（注入闲聊专属的提示词插槽）
    brain = CogniCore(
        model=base_llm, 
        tokenizer=tokenizer, 
        db_path="./test_memory_db",
        custom_s_desc=chat_s_desc   # <=== 这里！把闲聊设定传进去！
    )
    
    # 3. 启动交互终端
    brain.run_cli()

if __name__ == "__main__":
    main()