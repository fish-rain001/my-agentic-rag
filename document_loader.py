"""
============================================================
代码故事（从程序启动到回答用户问题的完整过程）
============================================================
程序启动后，如果需要初始化向量数据库...
【document_loader.py】的【load_and_chunk_all()】被调用...

它会：
1. 扫描 data 目录下的所有 .md 文件
2. 对每个文档进行三级分块：
   - L1: 1200+ tokens 的父块
   - L2: 600+ tokens 的中间块
   - L3: 300+ tokens 的叶子块

3. 同时计算每个块的稠密向量和稀疏向量
4. 将所有块写入 Milvus 数据库

这样检索时可以从 L3 叶子开始，如果多个叶子属于同一个父块，
可以自动合并到 L1，提供更完整的上下文。
============================================================
"""

"""
============================================================
面试版：三级分块 + 父子关系
============================================================
三级分块是 RAG 优化的重要技巧...

L3（叶子块）：最小的检索单位，精确匹配
L2（中间块）：L3 合并后的父块
L1（父块）：最大的块，提供完整上下文

Auto-merging：如果多个 L3 命中同一个 L2，或多个 L2 命中同一个 L1，
就自动用父块替换，保持上下文的完整性。
============================================================
"""

import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import DirectoryLoader, TextLoader

from . import config
from .embedding import get_embedding_service
from .milvus_client import get_milvus_client


class DocumentLoader:
    """
    文档加载 + 三级分块

    将文档切分为 L1/L2/L3 三级层级结构
    """

    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.milvus_client = get_milvus_client()

    def load_documents(self, data_dir: Path) -> List[str]:
        """加载 data 目录下的所有 .md 文档"""
        loader = DirectoryLoader(
            str(data_dir),
            glob="*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        documents = loader.load()
        return [doc.page_content for doc in documents]

    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """
        简单文本分块（按字符 + 段落）

        1. 先按换行分段
        2. 如果段落过长，再按句子分
        """
        # 先按段落分割
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 如果当前块加上这个段落超过大小
            if len(current_chunk) + len(para) + 2 > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # 如果单个段落就超过大小，按句子分割
                if len(para) > chunk_size:
                    sentences = para.split("。")
                    current_chunk = ""
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) + 1 > chunk_size:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence
                        else:
                            current_chunk = current_chunk + "。" + sentence if current_chunk else sentence
                else:
                    current_chunk = para
            else:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _create_chunk(
        self,
        text: str,
        chunk_level: int,
        parent_chunk_id: str,
        source: str,
        chunk_index: int
    ) -> Dict[str, Any]:
        """创建单个块"""
        # 生成 chunk_id
        chunk_id = f"{source}_{chunk_level}_{chunk_index}_{uuid.uuid4().hex[:8]}"

        # 计算向量
        dense_embedding = self.embedding_service.get_dense_embedding(text)
        sparse_embedding = self.embedding_service.get_sparse_embedding(text)

        return {
            "chunk_id": chunk_id,
            "parent_chunk_id": parent_chunk_id,
            "chunk_level": chunk_level,
            "text": text,
            "source": source,
            "dense_embedding": dense_embedding,
            "sparse_embedding": sparse_embedding,
        }

    def chunk_document(self, text: str, source: str) -> List[Dict[str, Any]]:
        """
        对单个文档进行三级分块

        流程：
        1. L3 切分：按 300 token 切分成最小块
        2. L2 切分：按 600 token 切分
        3. L1 切分：按 1200 token 切分
        """
        all_chunks = []

        # L3 叶子块 (300 token ≈ 600 字符)
        l3_texts = self._split_text(text, config.L3_CHUNK_SIZE)
        l3_chunks = []
        for i, text in enumerate(l3_texts):
            chunk = self._create_chunk(text, config.CHUNK_LEVEL_L3, "", source, i)
            l3_chunks.append(chunk)

        # L2 中间块 (600 token ≈ 1200 字符)
        l2_texts = self._split_text(text, config.L2_CHUNK_SIZE)
        l2_chunks = []
        l2_parent_map = {}  # L2 index -> L1 parent id

        for i, text in enumerate(l2_texts):
            # L2 的 parent 暂时为空，后续 L1 创建后补充
            chunk = self._create_chunk(text, config.CHUNK_LEVEL_L2, "", source, i)
            l2_chunks.append(chunk)

        # L1 父块 (1200 token ≈ 2400 字符)
        l1_texts = self._split_text(text, config.L1_CHUNK_SIZE)
        l1_chunks = []

        for i, text in enumerate(l1_texts):
            chunk = self._create_chunk(text, config.CHUNK_LEVEL_L1, "", source, i)
            l1_chunks.append(chunk)

        # 建立父子关系
        # L3 -> L2：按顺序，L3 的 parent_chunk_id 指向同一个 L2
        l3_per_l2 = max(1, len(l3_chunks) // max(1, len(l2_chunks)))
        for i, l3_chunk in enumerate(l3_chunks):
            l2_index = min(i // l3_per_l2, len(l2_chunks) - 1)
            l3_chunk["parent_chunk_id"] = l2_chunks[l2_index]["chunk_id"]

        # L2 -> L1：按顺序，L2 的 parent_chunk_id 指向同一个 L1
        l2_per_l1 = max(1, len(l2_chunks) // max(1, len(l1_chunks)))
        for i, l2_chunk in enumerate(l2_chunks):
            l1_index = min(i // l2_per_l1, len(l1_chunks) - 1)
            l2_chunk["parent_chunk_id"] = l1_chunks[l1_index]["chunk_id"]

        all_chunks.extend(l1_chunks)
        all_chunks.extend(l2_chunks)
        all_chunks.extend(l3_chunks)

        return all_chunks

    def load_and_chunk_all(self, data_dir: Path) -> List[Dict[str, Any]]:
        """
        加载目录下所有文档并分块

        返回所有层级的块列表
        """
        documents = self.load_documents(data_dir)

        all_chunks = []
        for doc_text in documents:
            source = f"doc_{len(all_chunks)}"
            chunks = self.chunk_document(doc_text, source)
            all_chunks.extend(chunks)

        print(f"文档加载完成，共 {len(documents)} 个文档，{len(all_chunks)} 个块")
        return all_chunks

    def persist_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """将块写入 Milvus"""
        if not chunks:
            return

        # 先构建 BM25 索引
        texts = [c["text"] for c in chunks]
        self.embedding_service.build_index(texts)

        # 写入 Milvus
        self.milvus_client.insert(chunks)
        print(f"已写入 {len(chunks)} 个块到 Milvus")


# 全局单例
_document_loader: Optional[DocumentLoader] = None


def get_document_loader() -> DocumentLoader:
    """获取 DocumentLoader 单例"""
    global _document_loader
    if _document_loader is None:
        _document_loader = DocumentLoader()
    return _document_loader
