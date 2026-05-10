"""
============================================================
代码故事（从程序启动到回答用户问题的完整过程）
============================================================
用户启动程序后，【main.py】被运行...
首先调用【initialize_vectorstore()】初始化向量数据库...
向量数据库会自动检测并选择 Milvus 或 ChromaDB 模式...
初始化完成后，程序进入交互循环...
用户可以输入问题，每次问题都会被传给【run_rag()】...
【run_rag()】会调用 RAG 流程，得到答案后显示给用户...
============================================================
"""

"""
============================================================
面试版：最简 Agentic RAG 的入口
============================================================
main.py 是整个 Agentic RAG 的入口...
它做了三件事：
1. 初始化向量数据库
2. 加载文档
3. 启动问答循环
核心问答逻辑委托给 rag_graph.py
============================================================
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from .vectorstore import get_vectorstore
from .rag_graph import run_rag


def initialize_vectorstore():
    """初始化向量数据库"""
    print("=" * 60)
    print("初始化向量数据库")
    print("=" * 60)

    vectorstore = get_vectorstore()
    vectorstore.initialize()

    print("初始化完成！\n")


def chat_loop():
    """交互式问答循环"""
    print("=" * 60)
    print("Agentic RAG 问答系统")
    print("=" * 60)
    print("输入问题进行咨询，或输入 'quit' 退出")
    print("输入 'reset' 重新初始化向量数据库")
    print("=" * 60)

    vectorstore = get_vectorstore()

    while True:
        try:
            question = input("\n你: ").strip()

            if not question:
                continue

            if question.lower() == "quit":
                print("再见！")
                break

            if question.lower() == "reset":
                # 重新初始化
                global _vectorstore
                import vectorstore as vs
                vs._vectorstore = None
                initialize_vectorstore()
                continue

            # 运行 RAG
            print("\n正在思考...")
            result = run_rag(question)

            print("\n" + "=" * 60)
            print("答案:")
            print("=" * 60)
            print(result["answer"])
            print("=" * 60)

            # 显示参考文档
            if result["relevant_docs"]:
                print(f"\n参考了 {len(result['relevant_docs'])} 个文档:")
                for i, doc in enumerate(result["relevant_docs"], 1):
                    source = doc["metadata"].get("source", "unknown")
                    print(f"  {i}. {source}")

        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}")


def main():
    """主函数"""
    initialize_vectorstore()
    chat_loop()


if __name__ == "__main__":
    main()
