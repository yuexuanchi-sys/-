#!/usr/bin/env python3
"""
测试脚本：验证搜索功能和节点重复问题是否已修复
"""

import requests
import json

BASE_URL = "http://localhost:5001"

def test_search_api():
    """测试搜索API"""
    print("=== 测试搜索API ===")
    
    # 测试搜索"一元"
    response = requests.get(f"{BASE_URL}/api/search?q=一元")
    if response.status_code == 200:
        data = response.json()
        print(f"搜索'一元'结果: {json.dumps(data, ensure_ascii=False, indent=2)}")
        
        if data['success']:
            print(f"找到 {len(data['data'])} 个结果")
            for item in data['data']:
                print(f"  - {item['name']} (类型: {item['labels']})")
        else:
            print(f"搜索失败: {data['error']}")
    else:
        print(f"API请求失败: {response.status_code}")

def test_branch_api():
    """测试分支API"""
    print("\n=== 测试分支API ===")
    
    # 测试一个已知节点的分支
    test_node = "一元一次方程"  # 替换为实际存在的节点名称
    response = requests.get(f"{BASE_URL}/api/branch/{test_node}")
    
    if response.status_code == 200:
        data = response.json()
        if data['success']:
            nodes = data['data']['nodes']
            relationships = data['data']['relationships']
            print(f"节点 '{test_node}' 的分支:")
            print(f"  节点数量: {len(nodes)}")
            print(f"  关系数量: {len(relationships)}")
            
            # 检查节点重复
            node_ids = [node['id'] for node in nodes]
            unique_node_ids = set(node_ids)
            if len(node_ids) == len(unique_node_ids):
                print("  ✅ 节点无重复")
            else:
                print("  ❌ 发现重复节点")
                duplicates = [node_id for node_id in node_ids if node_ids.count(node_id) > 1]
                print(f"  重复的节点ID: {duplicates}")
        else:
            print(f"分支加载失败: {data['error']}")
    else:
        print(f"API请求失败: {response.status_code}")

def test_full_graph():
    """测试完整图谱API"""
    print("\n=== 测试完整图谱API ===")
    
    response = requests.get(f"{BASE_URL}/api/full-graph")
    if response.status_code == 200:
        data = response.json()
        if data['success']:
            nodes = data['data']['nodes']
            relationships = data['data']['relationships']
            print(f"完整图谱:")
            print(f"  节点数量: {len(nodes)}")
            print(f"  关系数量: {len(relationships)}")
            
            # 检查节点重复
            node_ids = [node['id'] for node in nodes]
            unique_node_ids = set(node_ids)
            if len(node_ids) == len(unique_node_ids):
                print("  ✅ 节点无重复")
            else:
                print("  ❌ 发现重复节点")
                duplicates = [node_id for node_id in node_ids if node_ids.count(node_id) > 1]
                print(f"  重复的节点ID: {duplicates[:5]}...")  # 只显示前5个重复
        else:
            print(f"完整图谱加载失败: {data['error']}")
    else:
        print(f"API请求失败: {response.status_code}")

def test_health():
    """测试健康检查"""
    print("\n=== 测试健康检查 ===")
    
    response = requests.get(f"{BASE_URL}/api/health")
    if response.status_code == 200:
        data = response.json()
        print(f"健康状态: {data['status']}")
        print(f"数据库: {data.get('database', 'unknown')}")
    else:
        print(f"健康检查失败: {response.status_code}")

if __name__ == "__main__":
    print("开始测试数学知识图谱应用...")
    
    try:
        test_health()
        test_search_api()
        test_branch_api()
        test_full_graph()
        
        print("\n=== 测试完成 ===")
        
    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到服务器，请确保应用正在运行")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")