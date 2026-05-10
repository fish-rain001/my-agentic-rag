# my-agentic-rag

Agentic RAG 库 - 混合向量检索（BGE + BM25）+ LangGraph 状态机

## 核心功能

- **混合向量检索**：BGE 稠密向量 + BM25 稀疏向量
- **三级分块**：L1/L2/L3 父子块结构 + Auto-merging
- **LLM 循环评分**：检索结果相关性判断 + 查询重写
- **条件路由**：基于评分动态决定下一步（生成/重写检索）

## 技术栈

- LangGraph - 状态机
- LangChain - 集成
- BGE - 稠密向量
- Milvus/ChromaDB - 向量数据库
- MiniMax API - LLM

## 安装

```bash
pip install my-agentic-rag
```

或从源码：

```bash
pip install -e .
```

## 快速开始

```python
from my_agentic_rag import get_embedding_service, build_rag_graph

# 获取 embedding 服务
embedding_svc = get_embedding_service()
vec = embedding_svc.get_dense_embedding("your text")

# 构建 RAG 图
rag_graph = build_rag_graph()
result = rag_graph.invoke({
    "question": "your question",
    "query": "",
    "context": "",
    "docs": [],
    "relevant_docs": [],
    "route": "",
    "answer": ""
})
```

## API

### get_embedding_service()

返回 `EmbeddingService` 实例，提供：
- `get_dense_embedding(text)` - BGE 稠密向量
- `get_sparse_embedding(text)` - BM25 稀疏向量

### build_rag_graph()

返回 LangGraph CompiledStateGraph，包含节点：
- retrieve_initial / retrieve_expanded
- grade_documents
- rewrite_question
- generate

## License

MIT