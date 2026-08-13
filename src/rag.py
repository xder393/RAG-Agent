from openai import OpenAI
from src.config import API_KEY, BASE_URL, CHAT_MODEL, EMBEDDING_MODEL   
from src.vector_store import VectorStore
from src.ingest import load_text_file, chunk_text
def build_prompt(question,contexts):
    """构建 prompt，包含问题和相关的上下文"""
    context_text="\n\n".join(contexts)
    prompt=f"""你是一个知识库问答助手，以下是相关的上下文信息：
{context_text}
请根据以上内容回答用户的问题。如果无法从上下文中找到答案，请如实说明，不要编造答案。请直接回答问题：
用户问题：{question}"""
    return prompt
def ask_llm(prompt):
    """调用 LLM 接口获取回答"""
    client=OpenAI(api_key=API_KEY,base_url=BASE_URL)    
    response=client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"system","content":"你是一个知识库问答助手"},
                {"role":"user","content":prompt}],
        temperature=0.2,)
    return response.choices[0].message.content

def rag_answer(store,question,top_k=3):
    """基于 RAG 的问答流程"""
    # 1. 检索相关的文本块
    results=store.search(question,top_k=top_k)
    contexts=[chunk for score,chunk in results]
    # 2. 构建 prompt
    prompt=build_prompt(question,contexts)
    # 3. 调用 LLM 获取回答
    answer=ask_llm(prompt)
    return answer

def rag_answer_with_sources(store, question, top_k=3):
    """和 rag_answer 一样，但额外返回回答所依据的资料来源"""
    results = store.search(question, top_k=top_k)
    contexts = [chunk for score, chunk in results]
    prompt = build_prompt(question, contexts)
    answer = ask_llm(prompt)
    sources = [chunk[:100] for chunk in contexts]   # 每块截取前 100 字作为来源预览
    return answer, sources

if __name__=="__main__":
    # 1. 加载文本并构建向量存储
    content=load_text_file("data/sample.txt")
    chunks=chunk_text(content,chunk_size=100,overlap=20)
    store=VectorStore()
    store.add(chunks)
    # 2. 循环提问
    print("\n===== 知识库问答测试 =====")
    while True:
        question=input("\n请输入你的问题（输入 q 退出）：")
        if question.lower()=="q":
            break
        answer=rag_answer(store,question,top_k=3)
        print(f"\n=== 回答 ===\n{answer}")