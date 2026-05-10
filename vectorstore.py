"""
============================================================
代码故事（从程序启动到回答用户问题的完整过程）
============================================================
程序启动后，当需要检索文档时...
【vectorstore.py】的【retrieve()】被调用...
它尝试连接 Milvus，如果成功则使用混合检索...
如果 Milvus 不可用，自动降级到 ChromaDB...

最终返回最相关的文档列表。
============================================================
"""

"""
============================================================
面试版：向量存储抽象层（支持双模式）
============================================================
1. Milvus 模式：稠密+稀疏混合检索 + Rerank + Auto-merging
2. ChromaDB 模式（降级）：纯向量检索
============================================================
"""

import os
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from . import config

# 尝试导入 Milvus 相关模块
_milvus_available = False
try:
    from pymilvus import connections
    from .rag_utils import retrieve_documents
    from .document_loader import get_document_loader
    _milvus_available = True
except ImportError:
    pass


class VectorStore:
    """
    向量存储管理器

    支持两种模式：
    1. Milvus 模式（优先）：需要 Milvus 服务运行
    2. ChromaDB 模式（降级）：纯向量检索，无需额外服务
    """

    def __init__(self):
        self._initialized = False
        self._use_milvus = False
        self._chroma_db: Optional[Chroma] = None

        # 检查 Milvus 是否可用
        if _milvus_available:
            try:
                connections.connect(
                    host=config.MILVUS_HOST,
                    port=config.MILVUS_PORT,
                    alias="default",
                    timeout=5,
                    user=config.MILVUS_USER if config.MILVUS_USER else None,
                    password=config.MILVUS_PASSWORD if config.MILVUS_PASSWORD else None,
                    secure=True
                )
                self._use_milvus = True
                print("Milvus 连接成功，使用 Milvus 模式")
            except Exception:
                print("Milvus 不可用，自动降级到 ChromaDB 模式")
                self._use_milvus = False
        else:
            print("Milvus 模块未安装，使用 ChromaDB 模式")

    def initialize(self, data_dir=None) -> None:
        """初始化向量数据库"""
        if data_dir is None:
            data_dir = config.DATA_DIR

        if self._use_milvus:
            self._initialize_milvus(data_dir)
        else:
            self._initialize_chroma(data_dir)

        self._initialized = True

    def _initialize_milvus(self, data_dir) -> None:
        """使用 Milvus 初始化"""
        loader = get_document_loader()
        chunks = loader.load_and_chunk_all(data_dir)
        loader.persist_chunks(chunks)
        print(f"Milvus 向量数据库初始化完成，共 {len(chunks)} 个块")

    def _initialize_chroma(self, data_dir) -> None:
        """使用 ChromaDB 初始化"""
        from pathlib import Path

        # 加载文档
        loader = DirectoryLoader(
            str(data_dir),
            glob="*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        documents = loader.load()
        print(f"加载了 {len(documents)} 个文档")

        # 分块
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        splits = text_splitter.split_documents(documents)
        print(f"文档切分成 {len(splits)} 个文本块")

        # 创建 ChromaDB
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={"device": "cpu"},
        )

        chroma_dir = config.BASE_DIR / "chroma_db"
        chroma_dir.mkdir(parents=True, exist_ok=True)

        self._chroma_db = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=str(chroma_dir)
        )
        print(f"ChromaDB 向量数据库初始化完成，共 {len(splits)} 个块")

    def load_and_index(self, data_dir=None) -> None:
        """初始化（兼容旧接口）"""
        self.initialize(data_dir)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """检索最相关的文档"""
        if not self._initialized:
            self.initialize()

        if self._use_milvus:
            return self._retrieve_milvus(query, top_k)
        else:
            return self._retrieve_chroma(query, top_k)

    def _retrieve_milvus(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Milvus 检索"""
        result = retrieve_documents(query, top_k)
        docs = result["docs"]

        formatted = []
        for doc in docs:
            formatted.append({
                "content": doc.get("text", ""),
                "metadata": {
                    "source": doc.get("source", "unknown"),
                    "chunk_level": doc.get("chunk_level", 0),
                    "chunk_id": doc.get("chunk_id", ""),
                    "parent_chunk_id": doc.get("parent_chunk_id", ""),
                },
                "score": doc.get("score", 0.0)
            })

        return formatted

    def _retrieve_chroma(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """ChromaDB 检索"""
        if self._chroma_db is None:
            self.initialize()

        results = self._chroma_db.similarity_search_with_score(
            query=query,
            k=top_k
        )

        docs = []
        for doc, score in results:
            docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            })

        return docs


# 全局单例
_vectorstore: Optional[VectorStore] = None


def get_vectorstore() -> VectorStore:
    """获取 VectorStore 单例"""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = VectorStore()
    return _vectorstore
