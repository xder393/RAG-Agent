# 🤖 知识库问答 Agent（RAG）

> 基于**检索增强生成（RAG）**的知识库问答系统：上传资料即可提问，Agent 只依据你的资料回答，并附带来源引用。支持资料管理、多对话历史、Web 界面。

## ✨ 功能亮点

- 📄 **资料上传即问**：支持 `.txt` / `.pdf`，拖拽上传，自动切块入库
- 🧠 **基于 RAG 精准回答**：检索相关资料 → 大模型生成 → 附来源引用，可有效防止幻觉
- 🔍 **手写向量检索**：余弦相似度检索 + 排序，不依赖重型向量数据库
- 💬 **多对话管理**：新建 / 切换 / 删除对话，历史记录持久化（重启不丢）
- 🖥️ **现代 Web 界面**：双标签页（资料管理 / 问答），拖拽动画、流式交互
- 🔐 **本地 Embedding**：使用本地模型（BGE），资料不出本机、零向量化成本

## 🛠️ 技术栈

| 层 | 技术 |
|----|------|
| 前端 | HTML + CSS + JavaScript（原生，零依赖） |
| 后端 | Python + FastAPI + Uvicorn |
| 检索 | sentence-transformers（BAAI/bge-small-zh-v1.5）+ 手写余弦相似度 |
| 大模型 | DeepSeek API（兼容 OpenAI SDK） |
| 存储 | 资料文件（data/）+ 对话历史（JSON） |

## 🏗️ 架构

```
用户提问
   │
   ▼
┌─────────────────────────────────────────┐
│ ① 向量化问题（本地 BGE 模型）              │
│ ② 向量检索：与知识库所有块算余弦相似度     │
│ ③ 取 Top-K 相关资料块                     │
│ ④ 拼装 Prompt（资料 + 问题 + 防幻觉指令） │
│ ⑤ 调用 DeepSeek 生成回答                  │
│ ⑥ 返回回答 + 来源引用                     │
└─────────────────────────────────────────┘
   │
   ▼
用户得到带依据的回答
```

**为什么用 RAG 而不是直接问大模型？**
大模型不了解你的私有资料，直接问会"编造"（幻觉）。RAG 先检索你的资料再让模型"看着资料回答"，准确、可溯源、资料可随时更新。

## 📁 项目结构

```
agent/
├── src/
│   ├── config.py          # 配置读取（.env）
│   ├── ingest.py          # 资料读取（txt/pdf）+ 切块
│   ├── embeddings.py      # 文本向量化 + 余弦相似度
│   ├── vector_store.py    # 向量存储 + 检索
│   ├── rag.py             # RAG 问答管道（核心）
│   └── app.py             # FastAPI 后端（上传/问答/管理/对话）
├── web/
│   └── index.html         # 前端页面
├── data/                  # 上传的资料与对话记录（不入库）
├── requirements.txt
└── .env.example           # 配置模板
```

## 🚀 快速开始

### 1. 环境准备（Python 3.9+）

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置

复制 `.env.example` 为 `.env` 并填写：

```bash
cp .env.example .env
```

```ini
# .env
OPENAI_API_KEY=你的API Key
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
OPENAI_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
```

> 兼容所有 OpenAI 格式 API（DeepSeek / 通义 / 本地 Ollama 等），改 `OPENAI_BASE_URL` 和 `OPENAI_MODEL` 即可。
> 首次运行会自动下载本地向量模型（约 95MB）。国内网络若下载失败，可设置镜像：`HF_ENDPOINT=https://hf-mirror.com`

### 3. 启动

```bash
uvicorn src.app:app --host 127.0.0.1 --port 8000
```

浏览器打开 <http://127.0.0.1:8000> 🎉

1. 📚 **资料管理**页 → 拖拽上传 `.txt` / `.pdf`
2. 💬 **问答**页 → 新建对话 → 提问
3. 回答下方可展开 **📎 参考来源**

## 🔌 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/upload` | 上传资料（multipart） |
| GET | `/api/files` | 资料列表 |
| DELETE | `/api/files/{filename}` | 删除资料并重建知识库 |
| GET | `/api/status` | 知识库状态（块数） |
| GET | `/api/conversations` | 对话列表 |
| POST | `/api/conversations` | 新建对话 |
| GET | `/api/conversations/{id}` | 获取对话历史 |
| POST | `/api/conversations/{id}/ask` | 在对话中提问 |
| DELETE | `/api/conversations/{id}` | 删除对话 |

交互式 API 文档：启动后访问 <http://127.0.0.1:8000/docs>（FastAPI 自带 Swagger UI）

## 🗺️ Roadmap

- [ ] 多轮对话上下文（历史注入）
- [ ] 流式输出（SSE 打字机效果）
- [ ] 混合检索（BM25 关键词 + 向量）
- [ ] 单元测试（pytest）
- [ ] Docker 一键部署

## 📜 说明

- 本项目为个人学习实践项目，用于深入理解 RAG 与 Agent 的完整链路。
- `.env` 含密钥，已被 `.gitignore` 忽略，**切勿提交**。
