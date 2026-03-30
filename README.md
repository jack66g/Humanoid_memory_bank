# 🚀 CogniCore V3 - 数字生命认知与记忆中间件

CogniCore V3 是一套彻底跳出传统 RAG（检索增强生成）框架的**动态记忆与认知新陈代谢系统**。

它不只是一个死板的“历史聊天记事本”，而是一颗拥有独立运转逻辑的“数字大脑”。无论是为视觉小说中的专属女主角赋予会随时间流逝而变化的情感流转，还是为复杂的全栈 Web 业务打造具备自我纠偏能力的智能体，CogniCore 都能提供极致的底层支撑。

## ✨ 核心特性

* **🧠 认知新陈代谢 (Ebbinghaus Decay)**：内置艾宾浩斯遗忘曲线。毫无营养的闲聊会随时间自然枯萎并被物理抹除，而深刻的记忆会被长期保留。
* **⚖️ CSN 客观价值评估**：引入大模型作为“潜意识裁判”。自动对每条输入进行 C(通顺度)、S(绝对价值)、N(噪声) 打分，拒绝垃圾数据污染海马体。
* **🌊 物理张量共鸣 (Tensor Resonance)**：完全解耦底层存储。利用本地极速 Embedding 模型（BGE-Small）进行纯物理高维空间比对，精准触发记忆的巩固与融合。
* **⚡ 逻辑裁判员 2.0 (Conflict Resolution)**：自带冲突消解机制。当新事实与旧记忆冲突时，系统能自动覆写底层认知，彻底打破 AI 的“回声室效应”。
* **🔌 极简 RESTful API**：开箱即用。支持通过 FastAPI 快速接入任何前端业务或游戏引擎后台。

---

## 🛠️ 快速配置指南

系统采用**“云端大模型 API + 本地海马体晶体 (Embedding)”**的双轨架构，既保证了逻辑推理的强大，又实现了记忆向量的极速、免费和隐私安全。

### 1. 环境依赖安装
确保你的机器上安装了 Python 3.8+，然后安装核心依赖：
```bash
pip install fastapi uvicorn requests chromadb sentence-transformers torch gradio

2. 挂载本地海马体晶体 (Embedding 模型)
系统默认使用开源的 BAAI/bge-small-zh-v1.5 作为特征提取器。首次运行时，代码会自动从 HuggingFace 镜像站下载该轻量级模型。
(注：已在代码中配置 HF_ENDPOINT 强制使用国内镜像，确保下载顺畅。)

3. 配置大模型 API (核心底座)
你需要准备一个兼容 OpenAI 格式的云端大模型 API（如 DeepSeek、Kimi 或通义千问）。

打开 UI2.py (或你的入口文件 app.py)，找到以下区域并填入你的密钥：

# ---------------------------------------------------------
# 请在这里填入你的大模型 API 密钥和地址
# ---------------------------------------------------------
API_ENDPOINT = "[https://api.deepseek.com/chat/completions](https://api.deepseek.com/chat/completions)" # 替换为你的 API 地址
API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"          # 替换为你的真实 API Key
MODEL_NAME = "deepseek-chat"                               # 替换为你要调用的模型名称

4. 启动系统
方式一：启动可视化控制台 (Gradio UI)
运行 UI2，你可以直接在界面上调整 AI 的“性格权重 (w_s)”和“固执程度 (w_m)”：

python UI2.py

方式二：启动 RESTful API 服务 (FastAPI)
用于将记忆系统接入你自己的前台应用：
uvicorn app:app --reload --port 8000

