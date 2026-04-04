#!/usr/bin/env python3
"""
数学知识图谱系统 - 启动脚本
提供统一的命令行界面来运行系统的不同功能
"""

import argparse
import sys
import os
from knowledge_graph_builder import KnowledgeGraphBuilder
from app import app

def build_knowledge_graph():
    """构建知识图谱"""
    print("开始构建数学知识图谱...")
    builder = KnowledgeGraphBuilder()
    
    try:
        entities, relations = builder.build_knowledge_graph()
        print(f"构建完成! 共识别 {len(entities)} 个实体, {len(relations)} 个关系")
        return True
    except Exception as e:
        print(f"构建过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def start_web_server():
    """启动Web服务器"""
    print("启动知识图谱Web服务器...")
    print("访问地址: http://localhost:5000")
    print("按 Ctrl+C 停止服务器")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"服务器启动失败: {e}")

def test_system():
    """运行系统测试"""
    print("运行系统测试...")
    os.system("python test_build.py")

def clear_database():
    """清空图数据库"""
    confirm = input("确定要清空图数据库吗？此操作不可撤销！(y/N): ")
    if confirm.lower() == 'y':
        builder = KnowledgeGraphBuilder()
        builder.clear_graph()
        print("图数据库已清空")
    else:
        print("操作已取消")

def show_status():
    """显示系统状态"""
    from neo4j_manager import Neo4jManager
    
    print("系统状态检查:")
    print("-" * 40)
    
    # 检查Neo4j连接
    try:
        manager = Neo4jManager()
        result = manager.graph.run("MATCH (n) RETURN count(n) as node_count")
        node_count = result.data()[0]['node_count']
        print(f"✓ Neo4j连接正常，当前节点数: {node_count}")
    except Exception as e:
        print(f"✗ Neo4j连接失败: {e}")
    
    # 检查数据目录
    from config import Config
    if os.path.exists(Config.DATA_DIR):
        files = [f for f in os.listdir(Config.DATA_DIR) if f.endswith(('.docx', '.docm'))]
        print(f"✓ 数据目录存在，找到 {len(files)} 个文档文件")
    else:
        print(f"✗ 数据目录不存在: {Config.DATA_DIR}")
    
    # 检查输出目录
    if os.path.exists(Config.OUTPUT_DIR):
        print("✓ 输出目录存在")
    else:
        print("✗ 输出目录不存在")
    
    print("-" * 40)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='数学知识图谱系统')
    parser.add_argument('command', choices=['build', 'web', 'test', 'clear', 'status', 'all'],
                       help='运行命令: build-构建图谱, web-启动Web, test-运行测试, clear-清空数据库, status-系统状态, all-完整流程')
    
    args = parser.parse_args()
    
    # 添加当前目录到Python路径
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    if args.command == 'build':
        build_knowledge_graph()
    elif args.command == 'web':
        start_web_server()
    elif args.command == 'test':
        test_system()
    elif args.command == 'clear':
        clear_database()
    elif args.command == 'status':
        show_status()
    elif args.command == 'all':
        # 完整流程: 构建图谱 -> 启动Web
        if build_knowledge_graph():
            start_web_server()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()