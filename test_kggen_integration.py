#!/usr/bin/env python3
"""
KGGen集成测试脚本
测试KGGen与现有实体抽取和关系提取的集成效果
"""

import os
import json
import time
from typing import Dict, List
from config import Config
from entity_extractor_kggen import KGGenEnhancedEntityExtractor
from relation_extractor_kggen import KGGenEnhancedRelationExtractor
from kggen_client import KGGenClient

def test_kggen_client():
    """测试KGGen客户端功能"""
    print("=" * 60)
    print("测试KGGen客户端功能")
    print("=" * 60)
    
    try:
        client = KGGenClient()
        print("[OK] KGGen客户端初始化成功")
        
        # 测试文本
        test_texts = [
            "勾股定理指出：在直角三角形中，两直角边的平方和等于斜边的平方，即 a² + b² = c²",
            "二次函数的一般形式是 y = ax² + bx + c，其中a、b、c是常数且a ≠ 0",
            "解一元二次方程可以使用求根公式 x = [-b ± √(b² - 4ac)] / 2a",
            "微积分基本定理建立了微分和积分之间的关系"
        ]
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n测试文本 {i}: {text}")
            result = client.extract_entities_and_relations(text)
            
            print("提取到的实体:")
            for entity in result["entities"]:
                # 处理Unicode字符编码问题
                entity_text = entity['text'].encode('gbk', errors='replace').decode('gbk')
                print(f"  {entity['type']}: {entity_text} (位置: {entity['start']}-{entity['end']})")
            
            print("提取到的关系:")
            for relation in result["relations"]:
                # 处理Unicode字符编码问题
                subject = relation['subject'].encode('gbk', errors='replace').decode('gbk')
                obj = relation['object'].encode('gbk', errors='replace').decode('gbk')
                print(f"  {subject} --{relation['relation']}--> {obj}")
            
            time.sleep(1)  # 避免速率限制
        
        return True
        
    except Exception as e:
        print(f"[ERROR] KGGen客户端测试失败: {e}")
        return False

def test_entity_extractor_with_kggen():
    """测试KGGen增强的实体提取器"""
    print("\n" + "=" * 60)
    print("测试KGGen增强的实体提取器")
    print("=" * 60)
    
    try:
        extractor = KGGenEnhancedEntityExtractor(use_kggen=True)
        print("[OK] KGGen增强实体提取器初始化成功")
        
        test_texts = [
            "勾股定理是直角三角形的重要性质，它推导自平方差公式",
            "二次函数的图像是一个抛物线，其顶点坐标为 (-b/2a, c - b²/4a)",
            "等差数列的通项公式是 an = a1 + (n-1)d，其中d是公差"
        ]
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n测试文本 {i}: {text}")
            
            # 提取实体
            entities = extractor.extract_entities(text)
            print("提取到的实体:")
            for entity in entities:
                # 处理Unicode字符编码问题
                entity_text = entity['text'].encode('gbk', errors='replace').decode('gbk')
                print(f"  {entity['type']}: '{entity_text}' (来源: {entity['source']}, 置信度: {entity.get('confidence', 1.0):.2f})")
            
            # 提取实体和关系
            result = extractor.extract_entities_and_relations(text)
            print("提取到的关系:")
            for relation in result["relations"]:
                # 处理Unicode字符编码问题
                subject = relation['subject'].encode('gbk', errors='replace').decode('gbk')
                obj = relation['object'].encode('gbk', errors='replace').decode('gbk')
                print(f"  {subject} --{relation['relation']}--> {obj} (来源: {relation['source']})")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 实体提取器测试失败: {e}")
        return False

def test_relation_extractor_with_kggen():
    """测试KGGen增强的关系提取器"""
    print("\n" + "=" * 60)
    print("测试KGGen增强的关系提取器")
    print("=" * 60)
    
    try:
        extractor = KGGenEnhancedRelationExtractor()
        print("[OK] KGGen增强关系提取器初始化成功")
        
        test_text = "勾股定理是直角三角形的重要性质，它推导自平方差公式"
        test_entities = [
            {'text': '勾股定理', 'type': 'THEOREM', 'start': 0, 'end': 4},
            {'text': '直角三角形', 'type': 'CONCEPT', 'start': 6, 'end': 10},
            {'text': '平方差公式', 'type': 'FORMULA', 'start': 16, 'end': 20}
        ]
        
        print(f"测试文本: {test_text}")
        print("测试实体:")
        for entity in test_entities:
            # 处理Unicode字符编码问题
            entity_text = entity['text'].encode('gbk', errors='replace').decode('gbk')
            print(f"  {entity['type']}: {entity_text}")
        
        relations = extractor.extract_relations(test_text, test_entities)
        print("提取到的关系:")
        for rel in relations:
            # 处理Unicode字符编码问题
            subject = rel['subject'].encode('gbk', errors='replace').decode('gbk')
            obj = rel['object'].encode('gbk', errors='replace').decode('gbk')
            print(f"  {subject} --{rel['relation']}--> {obj} (来源: {rel['source']}, 置信度: {rel.get('confidence', 1.0):.2f})")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 关系提取器测试失败: {e}")
        return False

def compare_extraction_methods():
    """比较不同提取方法的性能"""
    print("\n" + "=" * 60)
    print("比较不同提取方法的性能")
    print("=" * 60)
    
    test_text = "二次函数 y = ax² + bx + c 的图像是一个抛物线，其顶点坐标可以通过公式 x = -b/2a 计算得到"
    
    # 1. 仅使用规则方法
    print("1. 仅使用规则方法:")
    extractor_rule = KGGenEnhancedEntityExtractor(use_kggen=False)
    entities_rule = extractor_rule.extract_entities(test_text, use_ensemble=False)
    print(f"   提取到 {len(entities_rule)} 个实体")
    for entity in entities_rule:
        # 处理Unicode字符编码问题
        entity_text = entity['text'].encode('gbk', errors='replace').decode('gbk')
        print(f"     {entity['type']}: {entity_text} (来源: {entity['source']})")
    
    # 2. 使用所有本地方法（规则+词性+模型）
    print("\n2. 使用所有本地方法:")
    extractor_local = KGGenEnhancedEntityExtractor(use_kggen=False)
    entities_local = extractor_local.extract_entities(test_text, use_ensemble=True)
    print(f"   提取到 {len(entities_local)} 个实体")
    for entity in entities_local:
        # 处理Unicode字符编码问题
        entity_text = entity['text'].encode('gbk', errors='replace').decode('gbk')
        print(f"     {entity['type']}: {entity_text} (来源: {entity['source']})")
    
    # 3. 使用KGGen增强方法
    print("\n3. 使用KGGen增强方法:")
    extractor_kggen = KGGenEnhancedEntityExtractor(use_kggen=True)
    entities_kggen = extractor_kggen.extract_entities(test_text, use_ensemble=True)
    print(f"   提取到 {len(entities_kggen)} 个实体")
    for entity in entities_kggen:
        # 处理Unicode字符编码问题
        entity_text = entity['text'].encode('gbk', errors='replace').decode('gbk')
        print(f"     {entity['type']}: {entity_text} (来源: {entity['source']})")
    
    return {
        "rule_only": len(entities_rule),
        "local_all": len(entities_local),
        "kggen_enhanced": len(entities_kggen)
    }

def benchmark_performance():
    """性能基准测试"""
    print("\n" + "=" * 60)
    print("性能基准测试")
    print("=" * 60)
    
    test_texts = [
        "勾股定理: a² + b² = c²",
        "二次函数: y = ax² + bx + c",
        "等差数列求和公式: S_n = n/2 * (a1 + an)",
        "圆的面积公式: S = πr²",
        "三角函数: sin²θ + cos²θ = 1"
    ]
    
    # 测试本地方法
    print("测试本地方法性能...")
    extractor_local = KGGenEnhancedEntityExtractor(use_kggen=False)
    start_time = time.time()
    
    for text in test_texts:
        entities = extractor_local.extract_entities(text)
        # 处理Unicode字符编码问题
        text_encoded = text.encode('gbk', errors='replace').decode('gbk')
        print(f"文本: {text_encoded} -> 实体数: {len(entities)}")
    
    local_time = time.time() - start_time
    print(f"本地方法总耗时: {local_time:.2f}秒")
    
    # 测试KGGen方法
    print("\n测试KGGen方法性能...")
    extractor_kggen = KGGenEnhancedEntityExtractor(use_kggen=True)
    start_time = time.time()
    
    for text in test_texts:
        entities = extractor_kggen.extract_entities(text)
        # 处理Unicode字符编码问题
        text_encoded = text.encode('gbk', errors='replace').decode('gbk')
        print(f"文本: {text_encoded} -> 实体数: {len(entities)}")
        time.sleep(0.5)  # 避免API速率限制
    
    kggen_time = time.time() - start_time
    print(f"KGGen方法总耗时: {kggen_time:.2f}秒")
    
    return {
        "local_time": local_time,
        "kggen_time": kggen_time
    }

def main():
    """主测试函数"""
    print("KGGen集成测试开始")
    print("=" * 60)
    
    results = {}
    
    # 运行各项测试
    results['kggen_client'] = test_kggen_client()
    results['entity_extractor'] = test_entity_extractor_with_kggen()
    results['relation_extractor'] = test_relation_extractor_with_kggen()
    results['comparison'] = compare_extraction_methods()
    results['performance'] = benchmark_performance()
    
    # 输出测试总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    success_count = sum(1 for result in results.values() if result is not False)
    total_count = len(results)
    
    print(f"测试通过率: {success_count}/{total_count}")
    
    if 'comparison' in results:
        comp = results['comparison']
        print(f"\n方法比较结果:")
        print(f"  仅规则方法: {comp['rule_only']} 个实体")
        print(f"  所有本地方法: {comp['local_all']} 个实体")
        print(f"  KGGen增强方法: {comp['kggen_enhanced']} 个实体")
    
    if 'performance' in results:
        perf = results['performance']
        print(f"\n性能测试结果:")
        print(f"  本地方法耗时: {perf['local_time']:.2f}秒")
        print(f"  KGGen方法耗时: {perf['kggen_time']:.2f}秒")
        if perf['local_time'] > 0:
            print(f"  速度比: {perf['kggen_time']/perf['local_time']:.1f}x")
        else:
            print("  速度比: 无限 (本地方法时间接近零)")
    
    print("\n测试完成！")

if __name__ == "__main__":
    main()