"""
============================================================
代码故事（从程序启动到回答用户问题的完整过程）
============================================================
程序启动后，【config.py】被加载，读取各类配置：
- MiniMax API 配置（用于 LLM 调用）
- Milvus 向量数据库配置（用于存储和检索文档向量）
- Rerank API 配置（用于重排检索结果）
- BM25 参数配置（用于稀疏向量检索）
============================================================
"""

"""
============================================================
面试版：配置管理
============================================================
采用集中式配置设计：
1. MiniMax API：LLM 调用
2. Milvus：向量数据库（稠密+稀疏混合检索）
3. Rerank API：结果重排
4. BM25 参数：稀疏向量检索调优
============================================================
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 项目根目录
BASE_DIR = Path(__file__).parent

load_dotenv(BASE_DIR / ".env")

# 文档目录
DATA_DIR = BASE_DIR / "data"

# ================================================================
# MiniMax API 配置（LLM）
# ================================================================
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.minimaxi.com/anthropic")
MODEL_ID = os.getenv("MODEL_ID", "MiniMax-M2.7")

# ================================================================
# Milvus 向量数据库配置
# ================================================================
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "agentic_rag_docs")
MILVUS_USER = os.getenv("MILVUS_USER", "")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "")

# ================================================================
# Rerank API 配置
# ================================================================
RERANK_API_KEY = os.getenv("RERANK_API_KEY", "")
RERANK_MODEL = os.getenv("RERANK_MODEL", "jina-reranker")
RERANK_API_URL = os.getenv("RERANK_API_URL", "https://api.jina.ai/rerank")

# ================================================================
# BM25 稀疏向量配置
# ================================================================
BM25_STATE_PATH = BASE_DIR / "data" / "bm25_state.json"

# ================================================================
# 三级分块配置
# ================================================================
CHUNK_LEVEL_L1 = 1  # 父块层级
CHUNK_LEVEL_L2 = 2  # 中间块层级
CHUNK_LEVEL_L3 = 3  # 叶子块层级
L1_CHUNK_SIZE = 1200  # L1 块大小（token）
L2_CHUNK_SIZE = 600  # L2 块大小（token）
L3_CHUNK_SIZE = 300  # L3 块大小（token）
LEAF_RETRIEVE_LEVEL = CHUNK_LEVEL_L3  # 检索从 L3 开始

# ================================================================
# Auto-merging 配置
# ================================================================
AUTO_MERGE_THRESHOLD = 2  # 合并阈值：子块数量 >= threshold 时合并到父块

# ================================================================
# Embedding 模型配置
# ================================================================
EMBEDDING_MODEL = "BAAI/bge-m3"  # BGE 稠密向量模型

# ================================================================
# RAG 参数
# ================================================================
RETRIEVAL_TOP_K = 5       # 初始检索数量
RERANK_TOP_K = 3          # 重排后返回数量
CANDIDATE_MULTIPLIER = 3  # 候选文档倍数（用于混合检索）
RRF_K = 60                # RRF 融合参数
