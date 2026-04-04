#!/usr/bin/env python3
"""
测试Volcengine ARK API连接和功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kggen_client import KGGenClient
from config import Config
import json

def test_ark_api():
    """测试Volcengine ARK API连接"""
    print("=== 测试Volcengine ARK API连接 ===")
    
    # 显示当前配置
    print(f"API URL: {Config.KGGEN_API_URL}")
    print(f"API Key: {Config.KGGen_API_KEY[:10]}...{Config.KGGen_API_KEY[-4:]}")
    print(f"Model: {Config.KGGen_MODEL}")
    
    # 创建客户端
    client = KGGenClient()
    
    # 测试文本
    test_text = "勾股定理指出：在直角三角形中，两直角边的平方和等于斜边的平方，即 a² + b² = c²"
    
    print(f"\n测试文本: {test_text}")
    print("\n发送请求到Volcengine ARK API...")
    
    try:
        # 提取实体和关系
        result = client.extract_entities_and_relations(test_text)
        
        print("\n=== 提取结果 ===")
        print(f"实体数量: {len(result['entities'])}")
        print(f"关系数量: {len(result['relations'])}")
        
        if result['entities']:
            print("\n实体:")
            for entity in result['entities']:
                print(f"  {entity['type']}: {entity['text']} (位置: {entity['start']}-{entity['end']})")
        
        if result['relations']:
            print("\n关系:")
            for relation in result['relations']:
                print(f"  {relation['subject']} --{relation['relation']}--> {relation['object']}")
        
        # 保存详细结果
        with open('ark_api_test_result.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n详细结果已保存到: ark_api_test_result.json")
        
        return True
        
    except Exception as e:
        print(f"API测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ark_api()
    if success:
        print("\n✅ Volcengine ARK API测试成功!")
    else:
        print("\n❌ Volcengine ARK API测试失败!")
        print("\n请检查:")
        print("1. API密钥是否正确")
        print("2. 网络连接是否正常") 
        print("3. API服务是否可用")