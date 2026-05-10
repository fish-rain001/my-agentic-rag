"""
============================================================
代码故事（从程序启动到回答用户问题的完整过程）
============================================================
当用户提问时...
【milvus_client.py】的【hybrid_retrieve()】被调用...
它接收两个向量：
1. dense_embedding（稠密向量）- 用于语义相似性检索
2. sparse_embedding（稀疏向量）- 用于关键词匹配检索

它会：
1. 分别在 Milvus 中检索稠密向量和稀疏向量
2. 使用 RRF（倒数排名融合）算法合并两个结果
3. 返回融合后的 top_k 结果
============================================================
"""

"""
============================================================
面试版：Milvus 混合检索
============================================================
Milvus 是一个生产级向量数据库...
本模块实现稠密+稀疏的混合检索...
核心是 RRF（Reciprocal Rank Fusion）算法...
将不同检索方法的结果按排名融合...
============================================================
"""

from typing import List, Dict, Any, Optional
import numpy as np

from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    AnnSearchRequest,
    RRFRanker,
)

from . import config
from .embedding import get_embedding_service


class MilvusClient:
    """
    Milvus 向量数据库客户端

    支持：
    - 创建集合（稠密向量 + 稀疏向量 + 层级字段）
    - 插入文档
    - 混合检索（稠密 + 稀疏 RRF 融合）
    """

    def __init__(self):
        self._collection: Optional[Collection] = None
        self._embedding_service = get_embedding_service()
        self._connect()

    def _connect(self) -> None:
        """连接到 Milvus（支持本地或云端）"""
        if config.MILVUS_USER and config.MILVUS_PASSWORD:
            # 云端连接（Zilliz Cloud）
            connections.connect(
                host=config.MILVUS_HOST,
                port=config.MILVUS_PORT,
                user=config.MILVUS_USER,
                password=config.MILVUS_PASSWORD,
                alias="default",
                secure=True
            )
        else:
            # 本地连接
            connections.connect(
                host=config.MILVUS_HOST,
                port=config.MILVUS_PORT,
                alias="default"
            )

    def _create_collection(self) -> Collection:
        """创建集合"""
        # 字段定义
        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=512, is_primary=True),
            FieldSchema(name="parent_chunk_id", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="chunk_level", dtype=DataType.INT64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
            # 稠密向量字段
            FieldSchema(name="dense_embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
            # 稀疏向量字段
            FieldSchema(name="sparse_embedding", dtype=DataType.SPARSE_FLOAT_VECTOR),
        ]

        schema = CollectionSchema(fields=fields, description="Agentic RAG Documents")

        collection = Collection(name=config.MILVUS_COLLECTION, schema=schema)

        # 创建索引
        # 稠密向量索引 (HNSW)
        dense_index_params = {
            "metric_type": "IP",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 256}
        }
        collection.create_index(
            field_name="dense_embedding",
            index_params=dense_index_params
        )

        # 稀疏向量索引 (SPARSE_INVERTED_INDEX)
        sparse_index_params = {
            "metric_type": "IP",
            "index_type": "SPARSE_INVERTED_INDEX",
            "params": {"drop_ratio_build": 0.2}
        }
        collection.create_index(
            field_name="sparse_embedding",
            index_params=sparse_index_params
        )

        return collection

    def get_collection(self) -> Collection:
        """获取集合，不存在则创建"""
        try:
            collection = Collection(name=config.MILVUS_COLLECTION)
            collection.load()
            self._collection = collection
            return collection
        except Exception:
            self._collection = self._create_collection()
            self._collection.load()
            return self._collection

    def insert(self, chunks: List[Dict[str, Any]]) -> None:
        """
        批量插入文档块

        chunks 格式：
        [{
            "chunk_id": str,
            "parent_chunk_id": str,
            "chunk_level": int,
            "text": str,
            "source": str,
            "dense_embedding": List[float],
            "sparse_embedding": Dict[int, float]
        }]
        """
        collection = self.get_collection()

        entities = [
            [c["chunk_id"] for c in chunks],
            [c.get("parent_chunk_id", "") for c in chunks],
            [c["chunk_level"] for c in chunks],
            [c["text"] for c in chunks],
            [c["source"] for c in chunks],
            [c["dense_embedding"] for c in chunks],
            [c["sparse_embedding"] for c in chunks],
        ]

        collection.insert(entities)
        collection.flush()

    def hybrid_retrieve(
        self,
        dense_embedding: List[float],
        sparse_embedding: Dict[int, float],
        top_k: int = 5,
        rrf_k: int = 60,
        filter_expr: str = ""
    ) -> List[Dict[str, Any]]:
        """
        混合检索：稠密 + 稀疏 RRF 融合

        1. 稠密向量搜索
        2. 稀疏向量搜索
        3. RRF 融合两个结果
        """
        collection = self.get_collection()

        candidate_k = top_k * config.CANDIDATE_MULTIPLIER

        # 稠密向量搜索请求
        dense_search = AnnSearchRequest(
            data=[dense_embedding],
            anns_field="dense_embedding",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=candidate_k,
            expr=filter_expr if filter_expr else None,
            output_fields=["chunk_id", "parent_chunk_id", "chunk_level", "text", "source"]
        )

        # 稀疏向量搜索请求
        sparse_search = AnnSearchRequest(
            data=[sparse_embedding],
            anns_field="sparse_embedding",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=candidate_k,
            expr=filter_expr if filter_expr else None,
            output_fields=["chunk_id", "parent_chunk_id", "chunk_level", "text", "source"]
        )

        # RRF 融合
        ranker = RRFRanker(k=rrf_k)

        results = collection.hybrid_search(
            reqs=[dense_search, sparse_search],
            ranker=ranker,
            limit=top_k,
            output_fields=["chunk_id", "parent_chunk_id", "chunk_level", "text", "source"]
        )

        # 格式化结果
        docs = []
        for hits in results:
            for hit in hits:
                docs.append({
                    "chunk_id": hit.entity.get("chunk_id"),
                    "parent_chunk_id": hit.entity.get("parent_chunk_id"),
                    "chunk_level": hit.entity.get("chunk_level"),
                    "text": hit.entity.get("text"),
                    "source": hit.entity.get("source"),
                    "score": hit.score
                })

        return docs

    def dense_retrieve(
        self,
        dense_embedding: List[float],
        top_k: int = 5,
        filter_expr: str = ""
    ) -> List[Dict[str, Any]]:
        """
        仅稠密向量检索（备用）
        """
        collection = self.get_collection()

        results = collection.search(
            data=[dense_embedding],
            anns_field="dense_embedding",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=top_k,
            expr=filter_expr if filter_expr else None,
            output_fields=["chunk_id", "parent_chunk_id", "chunk_level", "text", "source"]
        )

        docs = []
        for hits in results:
            for hit in hits:
                docs.append({
                    "chunk_id": hit.entity.get("chunk_id"),
                    "parent_chunk_id": hit.entity.get("parent_chunk_id"),
                    "chunk_level": hit.entity.get("chunk_level"),
                    "text": hit.entity.get("text"),
                    "source": hit.entity.get("source"),
                    "score": hit.score
                })

        return docs


# 全局单例
_milvus_client: Optional[MilvusClient] = None


def get_milvus_client() -> MilvusClient:
    """获取 MilvusClient 单例"""
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = MilvusClient()
    return _milvus_client
