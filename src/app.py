import os
import json
import uuid
import time
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from src.ingest import load_text_file, load_pdf_file, chunk_text
from src.vector_store import VectorStore
from src.rag import rag_answer, rag_answer_with_sources
app = FastAPI(title="RAG Knowledge Base API", description="基于 RAG 的知识库问答 API", version="1.0.0")
store=VectorStore()
def load_existing_data():
    """加载已有的文本数据并构建向量存储"""
    if not os.path.exists("data"):
        print("未找到 data 文件夹，请先创建。")
        return
    for filename in os.listdir("data"):
        file_path = os.path.join("data", filename)
        if filename.endswith(".txt"):
            content = load_text_file(file_path)
        elif filename.endswith(".pdf"):
            content = load_pdf_file(file_path)
        else:
            continue
        chunks = chunk_text(content, chunk_size=100, overlap=20)
        store.add(chunks)
        print(f"已加载 {filename}，创建了 {len(chunks)} 个文本块。")
@app.on_event("startup")
async def startup():
    """应用启动时加载已有数据"""
    load_existing_data()
@app.get("/",response_class=HTMLResponse)
async def root():
    """① 返回前端页面（浏览器访问根网址时）"""
    with open("web/index.html", "r", encoding="utf-8") as f:
        return f.read()
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """② 上传文件接口（前端上传文件时调用）"""
    filename = file.filename
    if not (filename.endswith(".txt") or filename.endswith(".pdf")):
        return JSONResponse({"ok": False, "message": "仅支持 txt 或 pdf 文件"}, status_code=400)

    # 保存文件到 data/ 文件夹
    filepath = os.path.join("data", filename)
    content = await file.read()
    with open(filepath, "wb") as f:
        f.write(content)

    # 按类型读取文字
    if filename.endswith(".txt"):
        text = content.decode("utf-8")       # 直接解码字节
    else:
        text = load_pdf_file(filepath)       # PDF 用 pypdf 读

    # 切块并入库
    chunks = chunk_text(text)
    store.add(chunks)
    return {"ok": True, "message": f"「{filename}」上传成功，已加入 {len(chunks)} 块资料"}
@app.post("/api/chat")
async def chat(request: dict):
    """③ 接收问题，调用 RAG 返回回答和来源题时调用）"""
    question = request.get("question", "").strip()
    if not question:
        return JSONResponse({"ok": False, "message": "问题不能为空"}, status_code=400)
    answer, sources = rag_answer_with_sources(store, question, top_k=3)
    return {"ok": True, "answer": answer, "sources": sources}
@app.get("/api/status")
async def status():
    """④ 返回当前知识库状态"""
    return {"ok": True, "total_chunks": len(store.chunks)}


@app.get("/api/files")
async def list_files():
    """⑤ 列出 data/ 里所有资料文件"""
    files = []
    if os.path.exists("data"):
        for filename in os.listdir("data"):
            filepath = os.path.join("data", filename)
            size = os.path.getsize(filepath)
            files.append({"name": filename, "size": size})
    return {"ok": True, "files": files}


@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    """⑥ 删除一个资料文件，并重建知识库"""
    filepath = os.path.join("data", filename)
    if not os.path.exists(filepath):
        return JSONResponse({"ok": False, "message": "文件不存在"}, status_code=404)

    os.remove(filepath)                       # 删除物理文件

    # 重建知识库：清空内存里的旧数据，重新加载剩下的文件
    store.chunks = []
    store.vectors = []
    load_existing_data()

    return {"ok": True, "message": f"已删除「{filename}」，知识库已重建"}


# ========== 对话存储（JSON 文件） ==========

CONVERSATIONS_FILE = "data/conversations.json"


def load_conversations():
    """读取所有对话"""
    if not os.path.exists(CONVERSATIONS_FILE):
        return []
    with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_conversations(conversations):
    """把对话列表保存到文件"""
    with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)


# ========== 对话接口 ==========

@app.get("/api/conversations")
async def list_conversations():
    """⑦ 列出所有对话（只返回概要，不含消息）"""
    convs = load_conversations()
    return {"ok": True, "conversations": [
        {"id": c["id"], "title": c["title"], "created_at": c["created_at"]}
        for c in convs
    ]}


@app.post("/api/conversations")
async def create_conversation():
    """⑧ 新建一个空对话"""
    conv = {
        "id": uuid.uuid4().hex[:8],
        "title": "新对话",
        "created_at": time.strftime("%Y-%m-%d %H:%M"),
        "messages": [],
    }
    convs = load_conversations()
    convs.insert(0, conv)          # 新的对话放最前面
    save_conversations(convs)
    return {"ok": True, "conversation": conv}


@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    """⑨ 获取某个对话的完整消息"""
    convs = load_conversations()
    for c in convs:
        if c["id"] == conv_id:
            return {"ok": True, "conversation": c}
    return JSONResponse({"ok": False, "message": "对话不存在"}, status_code=404)


@app.post("/api/conversations/{conv_id}/ask")
async def ask_in_conversation(conv_id: str, request: dict):
    """⑩ 在指定对话里提问：存用户问题 → 生成回答 → 存回答"""
    question = request.get("question", "").strip()
    if not question:
        return JSONResponse({"ok": False, "message": "问题不能为空"}, status_code=400)

    convs = load_conversations()
    conv = None
    for c in convs:
        if c["id"] == conv_id:
            conv = c
            break
    if not conv:
        return JSONResponse({"ok": False, "message": "对话不存在"}, status_code=404)

    # 1. 保存用户的消息
    conv["messages"].append({"role": "user", "content": question})

    # 2. 调用 RAG 生成回答（复用之前写好的函数）
    answer, sources = rag_answer_with_sources(store, question, top_k=3)

    # 3. 保存助手的消息（附带来源）
    conv["messages"].append({"role": "assistant", "content": answer, "sources": sources})

    # 4. 如果还是默认标题，就用第一个问题当标题（截取前 20 字）
    if conv["title"] == "新对话":
        conv["title"] = question[:20]

    save_conversations(convs)
    return {"ok": True, "answer": answer, "sources": sources}


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    """⑪ 删除一个对话"""
    convs = load_conversations()
    new_convs = [c for c in convs if c["id"] != conv_id]
    if len(new_convs) == len(convs):          # 长度没变 = 没删到 = 不存在
        return JSONResponse({"ok": False, "message": "对话不存在"}, status_code=404)
    save_conversations(new_convs)
    return {"ok": True, "message": "对话已删除"}