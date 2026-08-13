# embeddings.py - 把文字变成向量（数字列表）

from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL

# 模型只加载一次，之后复用（懒加载）
_model = None


def get_model():
    """第一次调用时加载模型，之后直接返回已加载的模型"""
    global _model
    if _model is None:
        print(f"首次使用，正在下载/加载模型：{EMBEDDING_MODEL} ...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts):
    """把一组文字变成向量，返回 [向量1, 向量2, ...]"""
    model = get_model()
    vectors = model.encode(texts)     # 结果是 numpy 数组
    return vectors.tolist()           # 转成普通 Python 列表，方便使用


def cosine_similarity(v1, v2):
    """计算两个向量的相似度，返回 0~1 之间的数（越接近 1 越相似）"""
    dot = sum(a * b for a, b in zip(v1, v2))      # 点积：对应位置相乘再相加
    norm1 = sum(a * a for a in v1) ** 0.5         # 向量1 的长度
    norm2 = sum(b * b for b in v2) ** 0.5         # 向量2 的长度
    return dot / (norm1 * norm2)                  # 余弦相似度公式


if __name__ == "__main__":
    # 测试：把三句不同的话变成向量，看看效果
    sentences = [
        "公司的年假制度是什么？",
        "员工每年有几天带薪年假？",
        "今天天气真不错",
    ]
    vectors = embed_texts(sentences)

    for s, v in zip(sentences, vectors):
        print(f"句子：{s}")
        print(f"向量维度：{len(v)}  （每个向量有 {len(v)} 个数字）")
        print(f"前 8 个数字：{[round(x, 3) for x in v[:8]]}")
        print()

    print("=== 相似度预览（越接近 1 越相似）===")
    print(f"句1 vs 句2：{cosine_similarity(vectors[0], vectors[1]):.3f}  ← 都是问年假")
    print(f"句1 vs 句3：{cosine_similarity(vectors[0], vectors[2]):.3f}  ← 一个是年假一个是天气")