#!/usr/bin/env python3
"""
综合测试脚本 - 验证所有模块功能

测试项:
1. 数据加载
2. 文本预处理
3. NER 训练数据生成 (BIOES)
4. 实体提取 (规则+词性)
5. 关系提取 (规则+共现+定义模式)
6. 知识图谱构建 (端到端)
7. 评估框架
8. 消融实验
"""

import os
import sys
import json
import time
import logging
import traceback
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestRunner:
    """测试运行器"""

    def __init__(self):
        self.results = {}
        self.passed = 0
        self.failed = 0

    def run_test(self, name: str, test_fn):
        """运行单个测试"""
        print(f"\n{'='*60}")
        print(f"测试: {name}")
        print(f"{'='*60}")

        start = time.time()
        try:
            result = test_fn()
            elapsed = time.time() - start
            self.results[name] = {'status': 'PASS', 'elapsed': round(elapsed, 2), 'detail': result}
            self.passed += 1
            print(f"  [PASS] ({elapsed:.2f}s)")
            return True
        except Exception as e:
            elapsed = time.time() - start
            self.results[name] = {'status': 'FAIL', 'elapsed': round(elapsed, 2), 'error': str(e)}
            self.failed += 1
            print(f"  [FAIL] ({elapsed:.2f}s): {e}")
            traceback.print_exc()
            return False

    def summary(self):
        print(f"\n{'='*60}")
        print(f"测试结果: {self.passed} 通过, {self.failed} 失败, 共 {self.passed + self.failed} 项")
        print(f"{'='*60}")
        for name, r in self.results.items():
            status = "[PASS]" if r['status'] == 'PASS' else "[FAIL]"
            print(f"  {status} {name} ({r['elapsed']}s)")
        return self.failed == 0


# ================================================================
#  测试用例
# ================================================================

def test_data_loading():
    """测试数据加载"""
    from data_loader import MathDataLoader
    loader = MathDataLoader()
    files = loader.get_file_list()
    assert len(files) > 0, "没有找到数据文件"

    data = loader.load_all_data()
    assert len(data) > 0, "没有加载到数据"

    for item in data:
        assert 'content' in item, "数据缺少 content 字段"
        assert len(item['content']) > 100, f"文件内容太短: {item.get('file_name')}"
        assert 'grade' in item, "数据缺少 grade 字段"

    return f"加载 {len(data)} 个文件, 总字符: {sum(len(d['content']) for d in data)}"


def test_text_preprocessing():
    """测试文本预处理"""
    from data_loader import MathDataLoader
    loader = MathDataLoader()

    test_texts = [
        "勾股  定理 指出：a²+b²=c²",
        "第一章\n\n丰富的 图形 世界",
        "已知∠A=30°，求∠B的度数",
    ]
    for text in test_texts:
        cleaned = loader.preprocess_text(text)
        assert len(cleaned) > 0, f"预处理后为空: {text}"
        # 数学符号应保留
        if '²' in text:
            assert '²' in cleaned, "数学符号被删除"

    return "文本预处理正常"


def test_ner_data_generation():
    """测试 NER 训练数据生成 (BIOES)"""
    from enhanced_data_processor import EnhancedDataProcessor

    processor = EnhancedDataProcessor()

    # 测试单句标注
    sentence = "勾股定理指出在直角三角形中两直角边的平方和等于斜边的平方"
    labels = processor._annotate_sentence(sentence, scheme='bioes')

    assert len(labels) == len(sentence), f"标签长度不匹配: {len(labels)} vs {len(sentence)}"

    # 验证 "勾股定理" 被标注
    entity_spans = []
    i = 0
    while i < len(labels):
        if labels[i].startswith('B-') or labels[i].startswith('S-'):
            start = i
            etype = labels[i].split('-', 1)[1]
            if labels[i].startswith('S-'):
                entity_spans.append((sentence[start:start+1], etype))
                i += 1
            else:
                i += 1
                while i < len(labels) and (labels[i].startswith('I-') or labels[i].startswith('E-')):
                    i += 1
                entity_spans.append((sentence[start:i], etype))
        else:
            i += 1

    entity_texts = [s[0] for s in entity_spans]
    assert '勾股定理' in entity_texts, f"勾股定理未被标注, 实体: {entity_spans}"
    assert '直角三角形' in entity_texts, f"直角三角形未被标注, ��体: {entity_spans}"

    # 测试批量生成
    test_data = [{'content': '有理数包括整数和分数。勾股定理是直角三角形的重要性质。',
                  'grade': 7, 'chapter': 1, 'file_name': 'test.docx'}]
    df = processor.create_char_level_ner_training_data(test_data, scheme='bioes')
    assert len(df) > 0, "未生成训练样本"

    return f"生成 {len(df)} 条样本, 单句标注实体: {entity_spans}"


def test_entity_extraction():
    """测试实体提取"""
    from entity_extractor_kggen import KGGenEnhancedEntityExtractor

    extractor = KGGenEnhancedEntityExtractor(use_kggen=False)

    test_text = """
    三角形是由三条线段首尾顺次连接所组成的封闭图形。
    直角三角形中，两个锐角互为余角。
    勾股定理：直角三角形两直角边的平方和等于斜边的平方。
    """

    entities = extractor.extract_entities(test_text, use_ensemble=True)
    assert len(entities) > 0, "未提取到实体"

    entity_texts = [e['text'] for e in entities]
    # 至少应该提取到一些数学概念
    math_found = [t for t in entity_texts if any(kw in t for kw in
                  ['三角', '直角', '锐角', '余角', '勾股', '线段', '图形'])]
    assert len(math_found) > 0, f"未提取到数学实体, 实体: {entity_texts[:10]}"

    return f"提取 {len(entities)} 个实体, 数学相关: {math_found[:5]}"


def test_relation_extraction():
    """测试关系提取"""
    from relation_extractor_kggen import KGGenEnhancedRelationExtractor

    extractor = KGGenEnhancedRelationExtractor(use_kggen=False)

    test_text = "等腰三角形属于三角形。学习勾股定理之前需要掌握直角三角形。平行四边形包含矩形和菱形。"
    test_entities = [
        {'text': '等腰三角形', 'type': 'CONCEPT', 'start': 0, 'end': 5},
        {'text': '三角形', 'type': 'CONCEPT', 'start': 7, 'end': 10},
        {'text': '勾股定理', 'type': 'THEOREM', 'start': 12, 'end': 16},
        {'text': '直角三角形', 'type': 'CONCEPT', 'start': 22, 'end': 27},
        {'text': '平行四边形', 'type': 'CONCEPT', 'start': 29, 'end': 34},
        {'text': '矩形', 'type': 'CONCEPT', 'start': 36, 'end': 38},
        {'text': '��形', 'type': 'CONCEPT', 'start': 39, 'end': 41},
    ]

    relations = extractor.extract_relations(test_text, test_entities)
    assert len(relations) > 0, "未提取到关系"

    rel_types = set(r['relation'] for r in relations)
    sources = set(r['source'] for r in relations)

    return (f"提取 {len(relations)} 个关系, "
            f"类型: {rel_types}, 来源: {sources}")


def test_knowledge_graph_building():
    """测试知识图谱构建 (端到端)"""
    from main_kggen_pipeline import KGGenMathKnowledgePipeline

    pipeline = KGGenMathKnowledgePipeline(use_kggen=False)
    results = pipeline.run_full_pipeline(mode='local_only')

    assert 'error' not in results, f"构建失败: {results.get('error')}"
    assert results['total_entities'] > 0, "没有实体"
    assert results['total_relations'] > 0, "没有关系"

    return (f"实体: {results['total_entities']}, "
            f"关系: {results['total_relations']}, "
            f"类型: {results['entity_type_distribution']}")


def test_evaluation_framework():
    """测试评估框架"""
    from evaluation import NERMetrics, RelationMetrics, KnowledgeGraphAnalyzer

    # NER 指标测试
    gold = [['B-CONCEPT', 'E-CONCEPT', 'O', 'B-THEOREM', 'I-THEOREM', 'E-THEOREM']]
    pred = [['B-CONCEPT', 'E-CONCEPT', 'O', 'B-THEOREM', 'I-THEOREM', 'E-THEOREM']]
    metrics = NERMetrics.compute_span_metrics(gold, pred, scheme='bioes')
    assert metrics['overall']['f1'] == 1.0, f"完全匹配应该 F1=1.0, 得到 {metrics}"

    # 关系指标测试
    gold_rels = [('三角形', '包含', '直角三角形'), ('勾股定理', '应用', '直角三角形')]
    pred_rels = [('三角形', '包含', '直角三角形'), ('勾股定理', '属于', '直角三角形')]
    rel_metrics = RelationMetrics.compute_relation_metrics(gold_rels, pred_rels)
    assert rel_metrics['overall']['precision'] == 0.5, f"关系P应为0.5, 得到 {rel_metrics}"

    # KG 分析测试
    entities = [
        {'text': '三角形', 'type': 'CONCEPT', 'confidence': 0.9},
        {'text': '直角三角形', 'type': 'CONCEPT', 'confidence': 0.85},
        {'text': '勾股定理', 'type': 'THEOREM', 'confidence': 0.95},
    ]
    relations = [
        {'subject': '三角形', 'relation': '包含', 'object': '直角三角形', 'confidence': 0.9},
        {'subject': '勾股定���', 'relation': '应用', 'object': '直角三角形', 'confidence': 0.85},
    ]
    analyzer = KnowledgeGraphAnalyzer(entities, relations)
    report = analyzer.full_report()

    assert report['structural']['num_nodes'] == 3
    assert report['structural']['num_edges'] == 2
    assert report['coverage']['coverage_ratio'] > 0, "覆盖率应大于0"

    return f"NER F1=1.0, 关系P=0.5, 图节点={report['structural']['num_nodes']}"


def test_paper_metrics():
    """测试论文指标生成"""
    output_dir = './output'
    entities_path = os.path.join(output_dir, 'kggen_entities.json')
    relations_path = os.path.join(output_dir, 'kggen_relations.json')

    if not os.path.exists(entities_path):
        return "SKIP: 无已有输出文件"
    if not os.path.exists(relations_path):
        return "SKIP: 无关系输出文件"

    from evaluation import generate_paper_metrics

    with open(entities_path, 'r', encoding='utf-8') as f:
        entities = json.load(f)
    with open(relations_path, 'r', encoding='utf-8') as f:
        relations = json.load(f)

    report = generate_paper_metrics(entities, relations)

    assert 'summary' in report
    assert report['summary']['total_entities'] > 0
    assert report['coverage']['coverage_ratio'] > 0

    return (f"覆盖率: {report['coverage']['coverage_ratio']:.1%}, "
            f"Hub节点: {[h['entity'] for h in report['hubs']['top_hubs'][:5]]}")


def test_training_data_generation():
    """测试完整的训练数据生成流��"""
    from data_loader import MathDataLoader
    from enhanced_data_processor import EnhancedDataProcessor

    loader = MathDataLoader()
    data = loader.load_all_data()

    processor = EnhancedDataProcessor()
    df = processor.create_char_level_ner_training_data(data, scheme='bioes')

    assert len(df) > 100, f"训练样本太少: {len(df)}"

    # 统计标注分布
    label_counter = Counter()
    for _, row in df.iterrows():
        label_counter.update(row['char_labels'])

    non_o_labels = {k: v for k, v in label_counter.items() if k != 'O'}
    assert len(non_o_labels) > 0, "没有非O标签"

    # 保存
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    processor.save_char_level_training_data(df, os.path.join(output_dir, 'ner_train_bioes.jsonl'))
    processor.save_conll_format_data(df, os.path.join(output_dir, 'ner_train_bioes.conll'), level='char')

    return (f"样本: {len(df)}, 标注分布: "
            f"{dict(Counter(k.split('-')[0] if '-' in k else k for k in label_counter.keys()))}")


# ================================================================
#  主程序
# ================================================================

def main():
    runner = TestRunner()

    # 基础测试
    runner.run_test("数据加载", test_data_loading)
    runner.run_test("文本预处理", test_text_preprocessing)
    runner.run_test("NER训练数据生成(BIOES)", test_ner_data_generation)
    runner.run_test("实体提取", test_entity_extraction)
    runner.run_test("关系提取", test_relation_extraction)
    runner.run_test("评估框架", test_evaluation_framework)

    # 端到端测试
    runner.run_test("训练数据生成(完整)", test_training_data_generation)
    runner.run_test("知识图谱构建(端到端)", test_knowledge_graph_building)
    runner.run_test("论文指标生成", test_paper_metrics)

    success = runner.summary()

    # 保存测试报告
    report_path = os.path.join('./output', 'test_report.json')
    os.makedirs('./output', exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(runner.results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n测试报告已保存到: {report_path}")

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
