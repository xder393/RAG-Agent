from src.embeddings import embed_texts, cosine_similarity
class VectorStore:
    def __init__(self):
        self.chunks = []  # 存储文本块
        self.vectors = []  # 存储对应的向量
    def add(self,chunks):
        new_vectors=embed_texts(chunks)
        self.chunks.extend(chunks)
        self.vectors.extend(new_vectors)
        print(f"已添加 {len(chunks)} 个文本块，当前总数：{len(self.chunks)}")
    def search(self,query,top_k=3):
        query_vector=embed_texts([query])[0]
        ranked=[]
        for i in range(len(self.vectors)):
            score=cosine_similarity(query_vector,self.vectors[i])
            ranked.append((score,self.chunks[i]))
        ranked.sort(key=lambda x:x[0],reverse=True)
        return ranked[:top_k]
if __name__=="__main__":
    from src.ingest import load_text_file, chunk_text
    content=load_text_file("data/sample.txt")
    chunks=chunk_text(content,chunk_size=100,overlap=20)
    print(f"总共创建了 {len(chunks)} 个文本块")
    store=VectorStore()
    store.add(chunks)
    print("\n===== 图书馆检索测试 =====")
    while True:
        query = input("\n请输入你的问题（输入 q 退出）：")
        if query.lower() == "q":
            break
        results = store.search(query, top_k=2)
        print(f"\n=== 与「{query}」最相关的 {len(results)} 块 ===")
        for score, chunk in results:
            print(f"[相似度 {score:.3f}] {chunk}...")