"""
单独运行此脚本以初始化向量数据库（无需进入问答循环）
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from .vectorstore import get_vectorstore


def main():
    print("=" * 60)
    print("初始化向量数据库")
    print("=" * 60)

    vectorstore = get_vectorstore()

    # 初始化（加载文档 -> 三级分块 -> 写入 Milvus）
    vectorstore.initialize()

    print("初始化完成！")


if __name__ == "__main__":
    main()
