# webui.py
import os
import sys
# 【保留】：强制使用国内镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import io
import time
import requests
import gradio as gr
from sentence_transformers import SentenceTransformer

# 导入你的认知皮层核心
from cogni_memory2 import CogniCore
from cogni_memory2.config import CogniConfig

# ==========================================
# 1. 核心引擎初始化 (本地挂载)
# ==========================================
print("🚀 正在加载本地海马体晶体 (BGE-Small)...")
local_emb_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

# 【注意】：请在这里填入你的大模型 API 密钥和地址
API_ENDPOINT = "https://api.deepseek.com/chat/completions" 
API_KEY = "你的API密钥" 
MODEL_NAME = "deepseek-chat" 

# 实例化引擎，获取默认配置
default_cfg = CogniConfig()
engine = CogniCore(
    api_endpoint=API_ENDPOINT,
    api_key=API_KEY,
    embedding_model=local_emb_model,
    model_name=MODEL_NAME,
    custom_s_desc="用户未设定具体判定标准，请固定给此项打 1 分。"
)

# ==========================================
# 2. 前端交互逻辑函数
# ==========================================
def apply_config(s_desc, w_s_val, w_m_val, top_k_val):
    """点击‘应用配置’按钮时触发"""
    engine.update_config({"w_s": w_s_val, "w_m": w_m_val})
    engine.update_evaluator_desc(s_desc)
    
    log_msg = (
        f"⚙️ [内核配置已热更新]\n"
        f"  - 价值观 (S_desc): {engine.evaluator.s_desc}\n"
        f"  - 性格权重 (w_s): {w_s_val}\n"
        f"  - 固执权重 (w_m): {w_m_val}\n"
        f"  - 回想广度 (top_k): {top_k_val}"
    )
    return log_msg

def chat_with_cognicore(user_message, history, top_k_val):
    """处理用户输入，获取大模型回复，并拦截底层代谢日志"""
    if not user_message.strip():
        return "", history, "⚠️ 输入不能为空"

    # 魔法拦截：捕获底层 engine 所有的 print 输出
    old_stdout = sys.stdout
    captured_logs = io.StringIO()
    sys.stdout = captured_logs
    
    try:
        # 1. 提取记忆
        retrieved_mems = engine.recall(user_message, top_k=int(top_k_val))
        memory_context = ""
        if retrieved_mems:
            print(f"🧠 [海马体检索] 提取了 {len(retrieved_mems)} 条高维关联记忆")
            for m in retrieved_mems:
                import datetime
                dt_str = datetime.datetime.fromtimestamp(m['meta']['timestamp']).strftime('%m-%d %H:%M')
                ai_reply = m['meta'].get('ai_text', '无记录')
                memory_context += f"[{dt_str}]\n用户当时说：{m['text']}\n你当时回答：{ai_reply}\n\n"
        else:
            memory_context = "无。"

        # 2. 组装 Prompt
        prompt = (
            f"你是一个有自我意志的实体。请严格根据历史记忆回答，如果有矛盾，以新记忆为准。\n"
            f"【触发的长期记忆】：\n{memory_context}\n"
            f"【用户当前输入】：{user_message}"
        )

        # 3. 调用云端 API 生成回复
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 300
        }
        
        res = requests.post(API_ENDPOINT, headers=headers, json=payload, timeout=15)
        res.raise_for_status()
        ai_reply = res.json()["choices"][0]["message"]["content"].strip()

        # 4. 感知与物理代谢！
        engine.perceive(user_message, ai_reply)
        engine.sleep()
        
    except Exception as e:
        ai_reply = f"系统异常: {e}"
        print(f"❌ 发生错误: {e}")
    finally:
        # 恢复系统的正常打印功能
        sys.stdout = old_stdout

    # 获取拦截到的全部底层日志
    logs = captured_logs.getvalue()
    
    # 【核心修复】：适配 Gradio 6.0 的消息字典格式
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": ai_reply})
    
    return "", history, logs

# ==========================================
# 3. 极客风 UI 界面布局
# ==========================================
# 【修正】：theme 参数已从 Blocks 移至 launch()
with gr.Blocks(title="CogniCore 数字生命控制台") as demo:
    gr.Markdown("# 🚀 CogniCore 数字生命控制台 V3 (张量突触版)")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎛️ 认知内核参数 (Control Panel)")
            
            s_desc_input = gr.Textbox(
                label="价值观评判标准 (S)", 
                value="用户未设定具体判定标准，请固定给此项打 1 分。",
                lines=3,
                info="定义 AI 认为什么是'有价值的'。例如：你需要极度偏向技术细节。"
            )
            
            w_s_slider = gr.Slider(
                minimum=0.1, maximum=3.0, step=0.1, 
                value=default_cfg.w_s, 
                label="性格权重 (w_s)", 
                info="越大越容易记住符合价值观的硬核知识"
            )
            
            w_m_slider = gr.Slider(
                minimum=0.1, maximum=3.0, step=0.1, 
                value=default_cfg.w_m, 
                label="固执程度 (w_m)", 
                info="越大越容易陷入过去的回忆（张量共鸣强化）"
            )
            
            top_k_slider = gr.Slider(
                minimum=1, maximum=5, step=1, 
                value=2, 
                label="回忆广度 (top_k)", 
                info="单次检索提取的关联记忆数量"
            )
            
            apply_btn = gr.Button("⚡ 烧录配置至内核", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("### 💬 交互与潜意识透视")
            
            # 【核心修正】：删掉 type="messages"，Gradio 6.0 不再支持此参数且默认就是字典格式
            chatbot = gr.Chatbot(label="与数字生命对话", height=400)
            
            with gr.Row():
                user_input = gr.Textbox(show_label=False, placeholder="输入消息...", scale=4)
                send_btn = gr.Button("发送 / 回车", variant="primary", scale=1)
                
            log_output = gr.Textbox(
                label="记忆透视窗 (Log Output)", 
                lines=8, 
                interactive=False,
                info="实时监控底层张量计算、CSN 打分与艾宾浩斯代谢过程"
            )

    # ==========================================
    # 4. 事件绑定
    # ==========================================
    apply_btn.click(
        fn=apply_config,
        inputs=[s_desc_input, w_s_slider, w_m_slider, top_k_slider],
        outputs=[log_output]
    )
    
    user_input.submit(
        fn=chat_with_cognicore,
        inputs=[user_input, chatbot, top_k_slider],
        outputs=[user_input, chatbot, log_output]
    )
    
    send_btn.click(
        fn=chat_with_cognicore,
        inputs=[user_input, chatbot, top_k_slider],
        outputs=[user_input, chatbot, log_output]
    )

# 启动！
if __name__ == "__main__":
    # 【修正】：theme 参数放在 launch 中以适配最新规范
    demo.launch(inbrowser=True, server_port=7860, theme=gr.themes.Monochrome())