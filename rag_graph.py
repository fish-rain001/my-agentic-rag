"""
============================================================
代码故事（从程序启动到回答用户问题的完整过程）
============================================================
用户提问后，【rag_graph.py】启动，首先进入【retrieve_initial()】节点...
这个节点调用向量数据库进行初次检索...
检索结果传给【grade_documents()】节点...
这个节点用 LLM 判断每个文档是否与问题相关...
根据评分结果决定路由：
  - 如果有相关文档 → 进入【generate()】节点 → 结束
  - 如果没有相关文档 → 进入【rewrite_question()】节点 → 【retrieve_expanded()】→ 结束
============================================================
"""

"""
============================================================
面试版：Agentic RAG 的核心流程
============================================================
Agentic RAG 使用 LangGraph 实现条件路由...
核心是【grade_documents()】节点，它用 LLM 评分...
评分结果决定下一步：
  - 评分通过 → 直接生成答案
  - 评分不通过 → 重写问题 + 扩展检索
这与普通 RAG 的一旦式检索完全不同...
============================================================
"""

from typing import TypedDict, List, Optional, Union, Any
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic


def get_response_text(response: Union[str, Any]) -> str:
    """从 LLM 响应中提取文本内容"""
    if isinstance(response, str):
        return response
    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, str):
            return content
        if isinstance(content, list) and len(content) > 0:
            # 处理 content 是列表的情况（如 MiniMax 返回的 thinking + text）
            texts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        texts.append(item.get("text", ""))
                    elif "text" in item:
                        texts.append(item["text"])
                elif hasattr(item, "text"):
                    texts.append(item.text)
            if texts:
                return "\n".join(texts)
    if isinstance(response, list) and len(response) > 0:
        # 处理响应本身就是列表的情况
        texts = []
        for item in response:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    texts.append(item.get("text", ""))
                elif "text" in item:
                    texts.append(item["text"])
            elif hasattr(item, "text"):
                texts.append(item.text)
        if texts:
            return "\n".join(texts)
    return str(response)

from . import config
from .vectorstore import get_vectorstore
from .prompts import (
    GENERATE_PROMPT,
    REWRITE_QUERY_PROMPT,
    GRADE_DOCUMENTS_PROMPT,
)


# ================================================================
# RAG 状态定义
# ================================================================

class RAGState(TypedDict):
    """
    RAG 流程状态类

    用于在整个 RAG 流程中传递和共享数据。
    每个节点都可以读取和写入这些字段。
    """
    question: str              # 用户原始问题
    query: str                 # 当前使用的查询（可能是重写后的）
    context: str               # 拼接后的上下文（用于 LLM 生成答案）
    docs: List[dict]           # 检索到的所有文档列表
    relevant_docs: List[dict] # 经过 LLM 评分判定为相关的文档
    route: str                 # 路由决定，决定下一步走哪条分支
    answer: str                # 最终生成的答案


# ================================================================
# LLM 模型
# ================================================================

def get_llm():
    """
    获取 LLM 模型实例（单例模式）

    Returns:
        ChatAnthropic: 配置好的 MiniMax API 客户端

    说明：
        - 使用单例模式避免重复创建模型实例
        - 配置 temperature=0.3，平衡创造性和确定性
        - base_url 指向 MiniMax 的 Anthropic 兼容接口
    """
    return ChatAnthropic(
        model=config.MODEL_ID,
        api_key=config.ANTHROPIC_API_KEY,
        base_url=config.ANTHROPIC_BASE_URL,
        temperature=0.3
    )


# ================================================================
# RAG 图节点
# ================================================================

def retrieve_initial(state: RAGState) -> RAGState:
    """
    节点1：初次检索

    Args:
        state: 包含用户问题的 RAG 状态

    Returns:
        更新后的 state，包含检索到的文档列表

    功能说明：
        1. 从 state 中取出用户问题
        2. 调用向量数据库进行相似度检索
        3. 将检索结果存入 state.docs
        4. 打印检索进度信息

    检索流程：
        用户问题 → 向量化 → ChromaDB 相似度搜索 → Top-K 文档
    """
    # 从状态中获取用户问题
    question = state["question"]
    print(f"\n[Step 1] 初次检索: {question[:50]}...")

    # 获取向量数据库实例并进行检索
    vectorstore = get_vectorstore()
    docs = vectorstore.retrieve(question, top_k=config.RETRIEVAL_TOP_K)

    # 更新状态：保存原始问题作为当前查询，保存检索结果
    state["query"] = question
    state["docs"] = docs

    print(f"[Step 1] 检索到 {len(docs)} 个文档")
    return state


def grade_documents(state: RAGState) -> RAGState:
    """
    节点2：相关性评分（Agentic RAG 核心）

    Args:
        state: 包含检索文档的 RAG 状态

    Returns:
        更新后的 state，包含相关文档和路由决定

    功能说明：
        1. 遍历每个检索到的文档
        2. 使用 LLM 判断文档是否与用户问题相关
        3. 根据判断结果分为"相关"和"不相关"两组
        4. 根据相关文档数量决定路由：
           - 有相关文档 → 路由到 "generate"（生成答案）
           - 无相关文档 → 路由到 "rewrite"（重写问题）

    为什么是核心：
        - 普通 RAG 使用向量相似度阈值判断（不准确）
        - Agentic RAG 使用 LLM 语义理解判断（更智能）
        - 这是实现"条件路由"的关键节点
    """
    question = state["question"]
    docs = state["docs"]
    llm = get_llm()

    print(f"\n[Step 2] 正在评估 {len(docs)} 个文档的相关性...")

    relevant_docs = []

    # 逐个文档进行 LLM 评分
    for i, doc in enumerate(docs):
        # 构建评分提示词，让 LLM 判断文档是否与问题相关
        prompt = GRADE_DOCUMENTS_PROMPT.format(
            question=question,
            document=doc["content"]
        )
        # 调用 LLM 获取评分结果
        response = llm.invoke(prompt)
        grade = get_response_text(response).strip().lower()

        # 根据 LLM 返回内容判断是否相关
        if "相关" in grade:
            relevant_docs.append(doc)
            print(f"  文档 {i+1}: [相关] (score={doc['score']:.4f})")
        else:
            print(f"  文档 {i+1}: [不相关]")

    # 将评分结果存入状态
    state["relevant_docs"] = relevant_docs

    # 根据评分结果决定路由：是否有相关文档
    if len(relevant_docs) >= 1:
        state["route"] = "generate"  # 有相关文档，直接生成答案
        print(f"[Step 2] 评分通过 ({len(relevant_docs)} 个相关文档)")
    else:
        state["route"] = "rewrite"   # 没有相关文档，需要重写问题
        print(f"[Step 2] 评分不通过，需要重写问题")

    return state


def rewrite_question(state: RAGState) -> RAGState:
    """
    节点3：查询重写（Agentic RAG 核心）

    Args:
        state: 包含原始问题的 RAG 状态

    Returns:
        更新后的 state，包含重写后的查询

    功能说明：
        当 Step 2 评分发现没有相关文档时触发此节点
        1. 获取用户原始问题
        2. 使用 LLM 将问题改写得更清晰、更适合检索
        3. 保存重写后的查询到 state.query

    重写策略：
        - 去除模糊表述，使问题更具体
        - 补充必要的上下文信息
        - 使用更精确的专业术语
        - 保持原问题的核心意图

    为什么需要重写：
        - 用户问题可能表达不清
        - 问题中的术语可能与文档不匹配
        - 重写可以提高二次检索的质量
    """
    question = state["question"]
    llm = get_llm()

    print(f"\n[Step 3] 重写问题...")

    # 使用提示词让 LLM 重写问题
    prompt = REWRITE_QUERY_PROMPT.format(question=question)
    response = llm.invoke(prompt)
    rewritten_query = get_response_text(response).strip()

    # 更新状态：用重写后的问题替换原查询
    state["query"] = rewritten_query
    print(f"[Step 3] 重写后: {rewritten_query[:50]}...")

    return state


def retrieve_expanded(state: RAGState) -> RAGState:
    """
    节点4：扩展检索

    Args:
        state: 包含重写后查询的 RAG 状态

    Returns:
        更新后的 state，包含扩展检索到的相关文档

    功能说明：
        1. 使用重写后的问题进行第二次向量检索
        2. 检索结果重新经过 LLM 评分
        3. 保存评分后的相关文档

    与 Step 1 的区别：
        - Step 1 使用用户原始问题检索
        - Step 4 使用 LLM 重写后的问题检索
        - 重写后的问题更清晰，检索结果可能更好
    """
    query = state["query"]
    print(f"\n[Step 4] 使用重写后的问题进行扩展检索...")

    # 使用重写后的问题进行第二次检索
    vectorstore = get_vectorstore()
    docs = vectorstore.retrieve(query, top_k=config.RETRIEVAL_TOP_K)

    state["docs"] = docs

    # 对扩展检索结果重新进行 LLM 评分
    question = state["question"]
    llm = get_llm()

    relevant_docs = []
    for i, doc in enumerate(docs):
        prompt = GRADE_DOCUMENTS_PROMPT.format(
            question=question,
            document=doc["content"]
        )
        response = llm.invoke(prompt)
        grade = get_response_text(response).strip().lower()

        if "相关" in grade:
            relevant_docs.append(doc)

    state["relevant_docs"] = relevant_docs
    print(f"[Step 4] 扩展检索到 {len(relevant_docs)} 个相关文档")

    return state


def generate(state: RAGState) -> RAGState:
    """
    节点5：生成答案

    Args:
        state: 包含相关文档的 RAG 状态

    Returns:
        更新后的 state，包含最终答案

    功能说明：
        1. 将相关文档拼装成上下文
        2. 构建包含上下文和问题的提示词
        3. 调用 LLM 生成最终答案
        4. 保存答案到 state.answer

    上下文构建格式：
        [文档 1] (来源: xxx.md)
        文档内容...

        [文档 2] (来源: yyy.md)
        文档内容...
    """
    question = state["question"]
    relevant_docs = state["relevant_docs"]
    llm = get_llm()

    print(f"\n[Step 5] 生成答案...")

    # 构建上下文：将相关文档格式化为字符串
    context_parts = []
    for i, doc in enumerate(relevant_docs):
        # 获取文档来源信息（文件名）
        source = doc["metadata"].get("source", "unknown")
        context_parts.append(f"[文档 {i+1}] (来源: {source})\n{doc['content']}")

    # 用双换行连接各文档
    context = "\n\n".join(context_parts)

    # 构建生成提示词，让 LLM 根据上下文回答问题
    prompt = GENERATE_PROMPT.format(
        context=context,
        question=question
    )
    # 调用 LLM 生成答案
    response = llm.invoke(prompt)
    answer = get_response_text(response).strip()

    # 保存上下文和答案到状态
    state["context"] = context
    state["answer"] = answer

    print(f"[Step 5] 答案生成完成 ({len(answer)} 字符)")
    return state


# ================================================================
# 构建 RAG 图
# ================================================================

def build_rag_graph():
    """
    构建 RAG 流程图

    Returns:
        CompiledStateGraph: 编译后的 LangGraph 图

    流程图结构：
        retrieve_initial → grade_documents
                                      ↓
                              [条件路由判断]
                                 ↓       ↓
                            generate   rewrite_question
                                          ↓
                                    retrieve_expanded
                                          ↓
                                          ↓
                                        generate

    说明：
        - 使用 StateGraph 定义状态和节点
        - 使用 add_conditional_edges 实现条件路由
        - 路由规则由 grade_documents 节点的 route 字段决定
    """
    graph = StateGraph(RAGState)

    # 添加 5 个节点到图中
    graph.add_node("retrieve_initial", retrieve_initial)  # 初次检索
    graph.add_node("grade_documents", grade_documents)     # LLM 评分
    graph.add_node("rewrite_question", rewrite_question)   # 查询重写
    graph.add_node("retrieve_expanded", retrieve_expanded) # 扩展检索
    graph.add_node("generate", generate)                   # 生成答案

    # 设置入口节点（第一个执行的节点）
    graph.set_entry_point("retrieve_initial")

    # 添加边：retrieve_initial 之后执行 grade_documents
    graph.add_edge("retrieve_initial", "grade_documents")

    # 添加条件边：根据评分结果决定下一步
    def route_decision(state: RAGState) -> str:
        """
        路由决策函数

        Args:
            state: 当前 RAG 状态

        Returns:
            str: 路由目标节点名称
        """
        return state.get("route", "rewrite")

    # 当 grade_documents 完成后，根据 route 字段决定下一步
    graph.add_conditional_edges(
        "grade_documents",      # 从评分节点出发
        route_decision,         # 根据 route 字段路由
        {
            "generate": "generate",           # route="generate" → 生成答案
            "rewrite": "rewrite_question"     # route="rewrite" → 重写问题
        }
    )

    # rewrite_question 之后执行 retrieve_expanded
    graph.add_edge("rewrite_question", "retrieve_expanded")

    # retrieve_expanded 之后执行 generate
    graph.add_edge("retrieve_expanded", "generate")

    # generate 是终节点，连接到 END
    graph.add_edge("generate", END)

    # 编译图，返回可执行的图实例
    return graph.compile()


# ================================================================
# 执行 RAG
# ================================================================

def run_rag(question: str) -> dict:
    """
    运行完整的 RAG 流程

    Args:
        question: 用户问题

    Returns:
        dict: 包含以下字段的字典
            - question: 原始问题
            - answer: 生成的答案
            - relevant_docs: 用到的相关文档
            - query: 最终使用的查询（可能经过重写）

    使用示例：
        result = run_rag("Python 列表如何切片？")
        print(result["answer"])
    """
    # 构建 RAG 图
    rag_graph = build_rag_graph()

    # 初始化状态
    initial_state = RAGState(
        question=question,
        query="",
        context="",
        docs=[],
        relevant_docs=[],
        route="",
        answer=""
    )

    # 执行图，获取最终状态
    result = rag_graph.invoke(initial_state)

    # 提取关键结果返回
    return {
        "question": result["question"],
        "answer": result["answer"],
        "relevant_docs": result["relevant_docs"],
        "query": result["query"]
    }


# ================================================================
# 调试：打印图结构
# ================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("RAG 图结构")
    print("=" * 50)

    rag_graph = build_rag_graph()

    # 打印图的可视化描述
    print("""
    流程说明：
    1. retrieve_initial: 初次向量检索
    2. grade_documents: LLM 评分文档相关性
    3. [条件路由]
       - 有相关文档 → generate
       - 无相关文档 → rewrite_question
    4. rewrite_question: 重写问题
    5. retrieve_expanded: 扩展检索
    6. generate: 生成答案
    """)
