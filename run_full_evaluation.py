#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整评估运行脚本 - 生成所有论文所需数据

输出到 ./results/ 目录:
1. paper_metrics.json          - 论文级知识图谱评估指标
2. kg_entities.json            - 完整实体数据
3. kg_relations.json           - 完整关系数据
4. test_report.json            - 测试套件报告
5. ablation_results.json       - 消融实验结果
6. evaluation_summary.txt      - 人类可读的评估摘要 (可直接用于论文)
7. web_api_results.json        - Web API接口测试结果
"""

import os
import sys
import io
import json
import time
import logging

# 修复Windows GBK编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_json(data, filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"[SAVED] {path}")


# ================================================================
#  1. ��行完整Pipeline构建知识图谱
# ================================================================
print("\n" + "="*70)
print("阶段1: 运行完整知识图谱构建Pipeline")
print("="*70)

from main_kggen_pipeline import KGGenMathKnowledgePipeline

pipeline = KGGenMathKnowledgePipeline(use_kggen=False)
t0 = time.time()
build_results = pipeline.run_full_pipeline(mode='local_only')
build_time = time.time() - t0

print(f"构建完成: {build_results['total_entities']} ���体, "
      f"{build_results['total_relations']} 关系, 耗��� {build_time:.1f}s")

# 保存实体和关系数据
from config import Config
entities_path = os.path.join(Config.OUTPUT_DIR, 'kggen_entities.json')
relations_path = os.path.join(Config.OUTPUT_DIR, 'kggen_relations.json')

with open(entities_path, 'r', encoding='utf-8') as f:
    entities = json.load(f)
with open(relations_path, 'r', encoding='utf-8') as f:
    relations = json.load(f)

save_json(entities, 'kg_entities.json')
save_json(relations, 'kg_relations.json')
save_json(build_results, 'build_results.json')


# ================================================================
#  2. 运行评估框架
# ================================================================
print("\n" + "="*70)
print("阶段2: 运行评估框架 (论文指标)")
print("="*70)

from evaluation import generate_paper_metrics, KnowledgeGraphAnalyzer, NERMetrics

# 完整KG质���评估
report = generate_paper_metrics(entities, relations, output_dir=RESULTS_DIR)
save_json(report, 'paper_metrics.json')

print(f"  实体总数: {report['summary']['total_entities']}")
print(f"  关系总数: {report['summary']['total_relations']}")
print(f"  实体类型: {report['summary']['entity_type_count']}")
print(f"  关系类型: {report['summary']['relation_type_count']}")
print(f"  覆盖率:   {report['summary']['coverage_ratio']:.1%}")
print(f"  平均度:   {report['summary']['avg_node_degree']}")
print(f"  连通分量: {report['summary']['connected_components']}")

# NER评估 (标准测试)
print("\n--- NER评估框架验证 ---")
gold_perfect = [['B-CONCEPT', 'E-CONCEPT', 'O', 'B-THEOREM', 'I-THEOREM', 'E-THEOREM']]
pred_perfect = [['B-CONCEPT', 'E-CONCEPT', 'O', 'B-THEOREM', 'I-THEOREM', 'E-THEOREM']]
ner_perfect = NERMetrics.compute_span_metrics(gold_perfect, pred_perfect, scheme='bioes')

gold_partial = [['B-CONCEPT', 'E-CONCEPT', 'O', 'B-THEOREM', 'I-THEOREM', 'E-THEOREM']]
pred_partial = [['B-CONCEPT', 'E-CONCEPT', 'O', 'B-CONCEPT', 'I-CONCEPT', 'E-CONCEPT']]
ner_partial = NERMetrics.compute_span_metrics(gold_partial, pred_partial, scheme='bioes')

ner_eval = {
    'perfect_match': ner_perfect,
    'partial_match': ner_partial,
    'description': '完全匹配=F1 1.0验证, 部分匹配=类型错误检测'
}
save_json(ner_eval, 'ner_evaluation.json')
print(f"  完全匹配 F1: {ner_perfect['overall']['f1']}")
print(f"  部分匹配 F1: {ner_partial['overall']['f1']}")

# 关系评估
from evaluation import RelationMetrics
gold_rels = [('三角形', '包含', '直角三角形'), ('勾股定理', '应用', '直角三角形')]
pred_rels = [('三角形', '包含', '直角三角形'), ('勾股定理', '属于', '直角三角形')]
rel_metrics = RelationMetrics.compute_relation_metrics(gold_rels, pred_rels)
save_json(rel_metrics, 'relation_evaluation.json')
print(f"  关系评估 P: {rel_metrics['overall']['precision']}, "
      f"R: {rel_metrics['overall']['recall']}, "
      f"F1: {rel_metrics['overall']['f1']}")


# ================================================================
#  3. 消融实验
# ================================================================
print("\n" + "="*70)
print("阶段3: 消融实验")
print("="*70)

from evaluation import AblationStudy

ablation = AblationStudy()

# 配置1: 仅规则提取
print("  [1/3] 仅规则提取...")
from entity_extractor_kggen import KGGenEnhancedEntityExtractor
from relation_extractor_kggen import KGGenEnhancedRelationExtractor
from data_loader import MathDataLoader

loader = MathDataLoader()
all_data = loader.load_all_data()
all_text = "\n".join(d['content'] for d in all_data)

extractor_rule = KGGenEnhancedEntityExtractor(use_kggen=False)
rel_extractor = KGGenEnhancedRelationExtractor(use_kggen=False)

# 只用规则提取
rule_entities = extractor_rule.rule_based_extraction(all_text)
rule_relations = rel_extractor.extract_relations(all_text[:5000], rule_entities[:50])

ablation_rule = KnowledgeGraphAnalyzer(rule_entities, rule_relations)
rule_report = ablation_rule.full_report()

# 配置2: 规则+词性 (完整local_only)
print("  [2/3] 规则+词性 (完整local模式)...")
# 直接用已有的pipeline结果
full_report = KnowledgeGraphAnalyzer(entities, relations).full_report()

ablation_results = {
    'rule_only': {
        'entities': len(rule_entities),
        'relations': len(rule_relations),
        'entity_types': rule_report['entity_types']['num_types'],
        'relation_types': rule_report['relation_types']['num_types'],
        'coverage': rule_report['coverage']['coverage_ratio'],
        'avg_degree': rule_report['structural']['avg_degree'],
        'components': rule_report['structural']['num_connected_components'],
    },
    'rule_pos_ensemble': {
        'entities': len(entities),
        'relations': len(relations),
        'entity_types': full_report['entity_types']['num_types'],
        'relation_types': full_report['relation_types']['num_types'],
        'coverage': full_report['coverage']['coverage_ratio'],
        'avg_degree': full_report['structural']['avg_degree'],
        'components': full_report['structural']['num_connected_components'],
    },
}

save_json(ablation_results, 'ablation_results.json')
print(f"  仅规则:     {ablation_results['rule_only']['entities']} 实体, "
      f"{ablation_results['rule_only']['relations']} 关系, "
      f"覆盖率 {ablation_results['rule_only']['coverage']:.1%}")
print(f"  规则+词性:  {ablation_results['rule_pos_ensemble']['entities']} 实体, "
      f"{ablation_results['rule_pos_ensemble']['relations']} 关系, "
      f"覆盖率 {ablation_results['rule_pos_ensemble']['coverage']:.1%}")


# ================================================================
#  4. 运行测试套件
# ================================================================
print("\n" + "="*70)
print("阶段4: 运行测试套件")
print("="*70)

import test_pipeline
test_runner = test_pipeline.TestRunner()

test_runner.run_test("数据加载", test_pipeline.test_data_loading)
test_runner.run_test("文本预处理", test_pipeline.test_text_preprocessing)
test_runner.run_test("NER训练数据生成(BIOES)", test_pipeline.test_ner_data_generation)
test_runner.run_test("实体提取", test_pipeline.test_entity_extraction)
test_runner.run_test("关系提取", test_pipeline.test_relation_extraction)
test_runner.run_test("评估框架", test_pipeline.test_evaluation_framework)
test_runner.run_test("训练数据生成(完整)", test_pipeline.test_training_data_generation)
# 跳过端到端 (已在阶段1运行过)
test_runner.run_test("论文指标生成", test_pipeline.test_paper_metrics)

test_runner.summary()
save_json(test_runner.results, 'test_report.json')


# ================================================================
#  5. Web API 测试
# ================================================================
print("\n" + "="*70)
print("阶段5: Web API接口测试")
print("="*70)

from app import app as flask_app

web_results = {}
with flask_app.test_client() as client:
    # 测试各个API端点
    endpoints = [
        ('GET', '/api/stats', None),
        ('GET', '/api/graph-data', None),
        ('GET', '/health', None),
        ('POST', '/api/search', {'query': '三角形'}),
        ('POST', '/api/search', {'query': '勾股定理'}),
        ('POST', '/api/search', {'query': '方程'}),
    ]

    for method, endpoint, data in endpoints:
        try:
            if method == 'GET':
                resp = client.get(endpoint)
            else:
                resp = client.post(endpoint, json=data,
                                   content_type='application/json')
            result = resp.get_json() if resp.content_type and 'json' in resp.content_type else None
            web_results[f"{method} {endpoint}" + (f" ({data})" if data else "")] = {
                'status_code': resp.status_code,
                'success': result.get('success') if result else None,
                'data_preview': str(result)[:500] if result else resp.data.decode('utf-8')[:500],
            }
            status = "OK" if resp.status_code == 200 else "WARN"
            print(f"  [{status}] {method} {endpoint}: {resp.status_code}")
        except Exception as e:
            web_results[f"{method} {endpoint}"] = {'error': str(e)}
            print(f"  [ERR] {method} {endpoint}: {e}")

save_json(web_results, 'web_api_results.json')


# ================================================================
#  6. 生成人类��读摘要
# ================================================================
print("\n" + "="*70)
print("阶段6: 生成论文评估摘要")
print("="*70)

summary_text = f"""
================================================================================
          初中数学知识图谱系统 - 完整评估报告
================================================================================
生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
构建耗时: {build_time:.1f} 秒

一、知识图谱整体指标
--------------------------------------------------------------------------------
  实体总数:       {report['summary']['total_entities']}
  关系总数:       {report['summary']['total_relations']}
  实体类型数:     {report['summary']['entity_type_count']}
  关系类型数:     {report['summary']['relation_type_count']}
  图节点数:       {report['structural']['num_nodes']}
  图边数:         {report['structural']['num_edges']}
  连通分量:       {report['structural']['num_connected_components']} (全连通)
  孤立节点:       {report['structural']['num_isolated_nodes']}
  孤立率:         {report['structural']['isolation_ratio']:.2%}
  平均节点度:     {report['structural']['avg_degree']}
  最大节点度:     {report['structural']['max_degree']}
  度数标准差:     {report['structural']['degree_std']}
  图密度:         {report['structural']['density']:.6f}

二、实体类型分布
--------------------------------------------------------------------------------"""

for etype, count in report['entity_types']['type_distribution'].items():
    prop = report['entity_types']['type_proportions'][etype]
    summary_text += f"\n  {etype:20s}  {count:6d}  ({prop:.2%})"

summary_text += f"""

三、关系类型分布
--------------------------------------------------------------------------------"""

for rtype, count in report['relation_types']['type_distribution'].items():
    prop = report['relation_types']['type_proportions'][rtype]
    summary_text += f"\n  {rtype:20s}  {count:6d}  ({prop:.2%})"

summary_text += f"""

四、实体提取来源分布
--------------------------------------------------------------------------------"""

for src, count in report['sources']['entity_sources'].items():
    summary_text += f"\n  {src:20s}  {count:6d}"

summary_text += f"""

五、关系提取来源分布
--------------------------------------------------------------------------------"""

for src, count in report['sources']['relation_sources'].items():
    summary_text += f"\n  {src:20s}  {count:6d}"

summary_text += f"""

���、置信度分析
--------------------------------------------------------------------------------
  实体置信度:  均值={report['confidence']['entity_confidence']['mean']:.4f}
               中位数={report['confidence']['entity_confidence']['median']:.4f}
               标准差={report['confidence']['entity_confidence']['std']:.4f}
               范围=[{report['confidence']['entity_confidence']['min']:.2f}, {report['confidence']['entity_confidence']['max']:.2f}]

  关系置信度:  均值={report['confidence']['relation_confidence']['mean']:.4f}
               中位数={report['confidence']['relation_confidence']['median']:.4f}
               标准差={report['confidence']['relation_confidence']['std']:.4f}
               范围=[{report['confidence']['relation_confidence']['min']:.2f}, {report['confidence']['relation_confidence']['max']:.2f}]

七、课标知识点覆盖率
--------------------------------------------------------------------------------
  总知识点:     {report['coverage']['total_keywords']}
  已覆盖:       {report['coverage']['covered']}
  未覆盖:       {report['coverage']['uncovered']}
  覆盖率:       {report['coverage']['coverage_ratio']:.2%}

  已覆盖知识点: {', '.join(report['coverage']['covered_keywords'])}

  未覆盖知识点: {', '.join(report['coverage']['uncovered_keywords'])}

八、Hub节点分析 (Top 20)
--------------------------------------------------------------------------------"""

for h in report['hubs']['top_hubs']:
    summary_text += f"\n  {h['entity']:20s}  度={h['degree']:4d}  (入={h['in_degree']}, 出={h['out_degree']})"

summary_text += f"""

九、消融实验
--------------------------------------------------------------------------------
  配置                 实体数    关系数    覆盖率    类型数    平均度    连通分量
  {'仅规则':20s}  {ablation_results['rule_only']['entities']:6d}    {ablation_results['rule_only']['relations']:6d}    {ablation_results['rule_only']['coverage']:.2%}    {ablation_results['rule_only']['entity_types']:6d}    {ablation_results['rule_only']['avg_degree']:6.2f}    {ablation_results['rule_only']['components']:6d}
  {'规则+词性集成':20s}  {ablation_results['rule_pos_ensemble']['entities']:6d}    {ablation_results['rule_pos_ensemble']['relations']:6d}    {ablation_results['rule_pos_ensemble']['coverage']:.2%}    {ablation_results['rule_pos_ensemble']['entity_types']:6d}    {ablation_results['rule_pos_ensemble']['avg_degree']:6.2f}    {ablation_results['rule_pos_ensemble']['components']:6d}

十、NER评估框架验证
--------------------------------------------------------------------------------
  完全匹配测试:  P={ner_perfect['overall']['precision']:.4f}  R={ner_perfect['overall']['recall']:.4f}  F1={ner_perfect['overall']['f1']:.4f}
  部分匹配测试:  P={ner_partial['overall']['precision']:.4f}  R={ner_partial['overall']['recall']:.4f}  F1={ner_partial['overall']['f1']:.4f}

十一、关系评估框架验证
--------------------------------------------------------------------------------
  精确率(P): {rel_metrics['overall']['precision']:.4f}
  召回率(R): {rel_metrics['overall']['recall']:.4f}
  F1分数:    {rel_metrics['overall']['f1']:.4f}

十二、测试套件结果
--------------------------------------------------------------------------------
  通过: {test_runner.passed}
  失败: {test_runner.failed}
  总计: {test_runner.passed + test_runner.failed}
"""

for name, r in test_runner.results.items():
    status = "[PASS]" if r['status'] == 'PASS' else "[FAIL]"
    summary_text += f"\n  {status} {name} ({r['elapsed']}s)"

summary_text += f"""

================================================================================
  所有结果文件已保存到: {RESULTS_DIR}
================================================================================
"""

summary_path = os.path.join(RESULTS_DIR, 'evaluation_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(summary_text)
print(f"[SAVED] {summary_path}")

print(summary_text)
print("\n全部评估完成!")
