"""
============================================================
代码故事（从程序启动到回答用户问题的完整过程）
============================================================
当 RAG 流程执行到检索步骤时...
【retrieve_documents()】被调用...

它做了四件事：
1. 获取查询的稠密和稀疏向量
2. 调用 Milvus 混合检索
3. 如果配置了 Rerank API，调用重排
4. 执行 Auto-merging，把多个叶子块合并为父块

最终返回更精准、上下文更完整的检索结果。
============================================================
"""

"""
============================================================
面试版：Rerank + Auto-merging
============================================================
Rerank：在初始检索后，用更精准的模型重新排序结果
Auto-merging：如果多个相关小块都属于同一个父块，
就用父块替换这些小块，提供更完整的上下文
============================================================
"""

import requests
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional

from . import config
from .embedding import get_embedding_service
from .milvus_client import get_milvus_client


# ================================================================
# Rerank
# ================================================================
def rerank_documents(
    query: str,
    docs: List[Dict[str, Any]],
    top_k: int
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    调用外部 Rerank API 重排文档

    返回：(重新排序的文档列表, 元信息)
    """
    if not docs or not config.RERANK_API_URL:
        return docs, {"rerank_applied": False}

    try:
        headers = {
            "Authorization": f"Bearer {config.RERANK_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": config.RERANK_MODEL,
            "query": query,
            "documents": [doc.get("text", "") for doc in docs],
            "top_n": top_k,
            "return_documents": True
        }

        response = requests.post(
            config.RERANK_API_URL,
            json=payload,
            headers=headers,
            timeout=30
        )

        if response.status_code != 200:
            return docs, {"rerank_applied": False, "rerank_error": "api_error"}

        result = response.json()

        # 按 rerank 结果重新排序
        reranked = []
        for item in result.get("results", []):
            idx = item["index"]
            reranked.append({
                **docs[idx],
                "rerank_score": item.get("relevance_score", 0)
            })

        return reranked[:top_k], {
            "rerank_applied": True,
            "rerank_model": config.RERANK_MODEL
        }

    except Exception as e:
        return docs, {"rerank_applied": False, "rerank_error": str(e)}


# ================================================================
# Auto-merging
# ================================================================
def _merge_to_parent_level(
    docs: List[Dict[str, Any]],
    threshold: int = 2
) -> Tuple[List[Dict[str, Any]], int]:
    """
    将子块合并到父块

    规则：如果同一个父块的子块数量 >= threshold，
    则用父块替换这些子块

    返回：(合并后的文档列表, 合并次数)
    """
    # 按 parent_chunk_id 分组
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if parent_id:
            groups[parent_id].append(doc)

    # 找出需要合并的父块 ID
    merge_parent_ids = [
        parent_id
        for parent_id, children in groups.items()
        if len(children) >= threshold
    ]

    if not merge_parent_ids:
        return docs, 0

    # 这里需要获取父块内容
    # 简化：直接用第一个子块的信息标记为已合并
    # 实际应该从 Milvus/数据库查询父块内容
    merged_docs = []
    merged_count = 0

    # 追踪已处理的父块
    processed_parents = set()

    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()

        if parent_id in merge_parent_ids and parent_id not in processed_parents:
            # 合并这个父块的所有子块
            children = groups[parent_id]
            merged_doc = {
                **children[0],
                "merged_from_children": True,
                "merged_child_count": len(children),
                "score": max(c.get("score", 0) for c in children)
            }
            merged_docs.append(merged_doc)
            processed_parents.add(parent_id)
            merged_count += 1
        elif not parent_id or parent_id not in merge_parent_ids:
            merged_docs.append(doc)

    return merged_docs, merged_count


def auto_merge_documents(
    docs: List[Dict[str, Any]],
    top_k: int,
    threshold: int = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    两阶段自动合并：L3 -> L2，然后 L2 -> L1

    返回：(合并后的文档列表, 元信息)
    """
    if threshold is None:
        threshold = config.AUTO_MERGE_THRESHOLD

    meta = {
        "auto_merge_enabled": True,
        "auto_merge_threshold": threshold,
        "auto_merge_steps": 0,
        "auto_merge_replaced_chunks": 0
    }

    if not docs:
        return docs, meta

    # 第一阶段：L3 -> L2
    merged_docs, merged_count_l3_l2 = _merge_to_parent_level(docs, threshold)

    # 第二阶段：L2 -> L1
    merged_docs, merged_count_l2_l1 = _merge_to_parent_level(merged_docs, threshold)

    total_merged = merged_count_l3_l2 + merged_count_l2_l1

    # 排序并截取 top_k
    merged_docs.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    merged_docs = merged_docs[:top_k]

    meta["auto_merge_steps"] = 2
    meta["auto_merge_replaced_chunks"] = total_merged

    return merged_docs, meta


# ================================================================
# 完整检索流程
# ================================================================
def retrieve_documents(
    query: str,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    完整检索流程

    1. 获取查询的稠密 + 稀疏向量
    2. Milvus 混合检索 (RRF 融合)
    3. Rerank 重排（如配置）
    4. Auto-merging 自动合并

    返回：{"docs": [...], "meta": {...}}
    """
    embedding_service = get_embedding_service()
    milvus_client = get_milvus_client()

    candidate_k = max(top_k * config.CANDIDATE_MULTIPLIER, top_k)

    # 1. 获取查询向量
    dense_embedding = embedding_service.get_dense_embedding(query)
    sparse_embedding = embedding_service.get_sparse_embedding(query)

    # 2. Milvus 混合检索
    try:
        retrieved = milvus_client.hybrid_retrieve(
            dense_embedding=dense_embedding,
            sparse_embedding=sparse_embedding,
            top_k=candidate_k,
            rrf_k=config.RRF_K,
            filter_expr=f"chunk_level == {config.LEAF_RETRIEVE_LEVEL}"
        )
        retrieval_mode = "hybrid"
    except Exception:
        # 降级：仅稠密检索
        retrieved = milvus_client.dense_retrieve(
            dense_embedding=dense_embedding,
            top_k=candidate_k,
            filter_expr=f"chunk_level == {config.LEAF_RETRIEVE_LEVEL}"
        )
        retrieval_mode = "dense_fallback"

    # 3. Rerank
    reranked, rerank_meta = rerank_documents(query, retrieved, top_k)
    rerank_meta["retrieval_mode"] = retrieval_mode

    # 4. Auto-merging
    merged, merge_meta = auto_merge_documents(reranked, top_k)

    # 合并元信息
    meta = {
        **rerank_meta,
        **merge_meta,
        "candidate_k": candidate_k,
        "leaf_retrieve_level": config.LEAF_RETRIEVE_LEVEL,
        "total_retrieved": len(retrieved)
    }

    return {"docs": merged, "meta": meta}
