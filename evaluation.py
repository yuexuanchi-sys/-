"""
评估框架 - 面向论文发表的全面评估模块

包含:
1. NER 实体识别评估 (span-level P/R/F1)
2. 关系提取评估 (triple-level P/R/F1)
3. 知识图谱质量分析 (图结构指标)
4. 消融实验支持
"""

import json
import os
import re
import logging
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, Counter
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# 1. NER 实体识别评估
# ============================================================

class NERBioesDecoder:
    """将 BIOES 标签序列解码为 span 集合"""

    @staticmethod
    def decode_bioes(labels: List[str]) -> List[Tuple[int, int, str]]:
        """
        解码 BIOES 标签为 (start, end, type) 列表。
        end 为 exclusive。
        """
        spans = []
        i = 0
        n = len(labels)
        while i < n:
            label = labels[i]
            if label.startswith('S-'):
                entity_type = label[2:]
                spans.append((i, i + 1, entity_type))
                i += 1
            elif label.startswith('B-'):
                entity_type = label[2:]
                start = i
                i += 1
                while i < n and labels[i].startswith('I-') and labels[i][2:] == entity_type:
                    i += 1
                if i < n and labels[i].startswith('E-') and labels[i][2:] == entity_type:
                    spans.append((start, i + 1, entity_type))
                    i += 1
                else:
                    # 不完整的 BIOES 序列，仍记录
                    spans.append((start, i, entity_type))
            else:
                i += 1
        return spans

    @staticmethod
    def decode_bio(labels: List[str]) -> List[Tuple[int, int, str]]:
        """解码 BIO 标签为 (start, end, type) 列表"""
        spans = []
        i = 0
        n = len(labels)
        while i < n:
            label = labels[i]
            if label.startswith('B-'):
                entity_type = label[2:]
                start = i
                i += 1
                while i < n and labels[i] == f'I-{entity_type}':
                    i += 1
                spans.append((start, i, entity_type))
            else:
                i += 1
        return spans


class NERMetrics:
    """NER span-level 评估指标"""

    @staticmethod
    def compute_span_metrics(
        gold_labels_list: List[List[str]],
        pred_labels_list: List[List[str]],
        scheme: str = 'bioes'
    ) -> Dict:
        """
        计算 span-level precision / recall / F1

        Args:
            gold_labels_list: 金标准标签序列列表
            pred_labels_list: 预测标签序列列表
            scheme: 'bioes' 或 'bio'

        Returns:
            包含 overall 和 per-type 指标的字典
        """
        decoder = NERBioesDecoder.decode_bioes if scheme == 'bioes' else NERBioesDecoder.decode_bio

        type_tp = defaultdict(int)
        type_fp = defaultdict(int)
        type_fn = defaultdict(int)

        for gold_labels, pred_labels in zip(gold_labels_list, pred_labels_list):
            gold_spans = set(decoder(gold_labels))
            pred_spans = set(decoder(pred_labels))

            for span in pred_spans:
                entity_type = span[2]
                if span in gold_spans:
                    type_tp[entity_type] += 1
                else:
                    type_fp[entity_type] += 1

            for span in gold_spans:
                entity_type = span[2]
                if span not in pred_spans:
                    type_fn[entity_type] += 1

        # 汇总
        all_types = set(type_tp.keys()) | set(type_fp.keys()) | set(type_fn.keys())
        per_type = {}

        total_tp = total_fp = total_fn = 0
        for t in sorted(all_types):
            tp = type_tp[t]
            fp = type_fp[t]
            fn = type_fn[t]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            per_type[t] = {
                'precision': round(p, 4),
                'recall': round(r, 4),
                'f1': round(f1, 4),
                'support': tp + fn
            }
            total_tp += tp
            total_fp += fp
            total_fn += fn

        overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0.0

        return {
            'overall': {
                'precision': round(overall_p, 4),
                'recall': round(overall_r, 4),
                'f1': round(overall_f1, 4),
                'support': total_tp + total_fn
            },
            'per_type': per_type
        }

    @staticmethod
    def compute_partial_match_metrics(
        gold_labels_list: List[List[str]],
        pred_labels_list: List[List[str]],
        scheme: str = 'bioes'
    ) -> Dict:
        """
        计算部分匹配指标 (SemEval 标准: exact + partial + type)

        返回 exact match, type match, partial match 三种口径。
        """
        decoder = NERBioesDecoder.decode_bioes if scheme == 'bioes' else NERBioesDecoder.decode_bio

        exact_tp = partial_tp = type_tp = 0
        total_pred = total_gold = 0

        for gold_labels, pred_labels in zip(gold_labels_list, pred_labels_list):
            gold_spans = decoder(gold_labels)
            pred_spans = decoder(pred_labels)
            total_pred += len(pred_spans)
            total_gold += len(gold_spans)

            matched_gold = set()
            for ps_start, ps_end, ps_type in pred_spans:
                best_match = None
                best_overlap = 0
                for gi, (gs_start, gs_end, gs_type) in enumerate(gold_spans):
                    if gi in matched_gold:
                        continue
                    overlap_start = max(ps_start, gs_start)
                    overlap_end = min(ps_end, gs_end)
                    overlap = max(0, overlap_end - overlap_start)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match = (gi, gs_start, gs_end, gs_type)

                if best_match is not None and best_overlap > 0:
                    gi, gs_start, gs_end, gs_type = best_match
                    matched_gold.add(gi)
                    # exact match
                    if ps_start == gs_start and ps_end == gs_end and ps_type == gs_type:
                        exact_tp += 1
                    # type match (type correct, span overlaps)
                    if ps_type == gs_type:
                        type_tp += 1
                    # partial match (any overlap)
                    partial_tp += 1

        def _prf(tp, pred_count, gold_count):
            p = tp / pred_count if pred_count > 0 else 0.0
            r = tp / gold_count if gold_count > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            return {'precision': round(p, 4), 'recall': round(r, 4), 'f1': round(f1, 4)}

        return {
            'exact_match': _prf(exact_tp, total_pred, total_gold),
            'type_match': _prf(type_tp, total_pred, total_gold),
            'partial_match': _prf(partial_tp, total_pred, total_gold),
            'total_predicted': total_pred,
            'total_gold': total_gold,
        }

    @staticmethod
    def error_analysis(
        texts: List[str],
        gold_labels_list: List[List[str]],
        pred_labels_list: List[List[str]],
        scheme: str = 'bioes',
        top_k: int = 20
    ) -> Dict:
        """
        错误分析: 统计常见的 FP / FN 模式

        Returns:
            false_positives: [(entity_text, type, count), ...]
            false_negatives: [(entity_text, type, count), ...]
            type_confusion: {(pred_type, gold_type): count}
        """
        decoder = NERBioesDecoder.decode_bioes if scheme == 'bioes' else NERBioesDecoder.decode_bio

        fp_counter = Counter()
        fn_counter = Counter()
        type_confusion = Counter()

        for text, gold_labels, pred_labels in zip(texts, gold_labels_list, pred_labels_list):
            gold_spans = set(decoder(gold_labels))
            pred_spans = set(decoder(pred_labels))

            for start, end, etype in pred_spans - gold_spans:
                entity_text = text[start:end] if end <= len(text) else text[start:]
                fp_counter[(entity_text, etype)] += 1

            for start, end, etype in gold_spans - pred_spans:
                entity_text = text[start:end] if end <= len(text) else text[start:]
                fn_counter[(entity_text, etype)] += 1

            # type confusion
            for ps in pred_spans:
                for gs in gold_spans:
                    if ps[0] == gs[0] and ps[1] == gs[1] and ps[2] != gs[2]:
                        type_confusion[(ps[2], gs[2])] += 1

        return {
            'false_positives': fp_counter.most_common(top_k),
            'false_negatives': fn_counter.most_common(top_k),
            'type_confusion': dict(type_confusion.most_common(top_k)),
        }


# ============================================================
# 2. 关系提取评估
# ============================================================

class RelationMetrics:
    """关系提取评估指标"""

    @staticmethod
    def compute_relation_metrics(
        gold_triples: List[Tuple[str, str, str]],
        pred_triples: List[Tuple[str, str, str]],
    ) -> Dict:
        """
        计算关系提取的 P / R / F1

        Args:
            gold_triples: [(subject, relation, object), ...]
            pred_triples: [(subject, relation, object), ...]
        """
        gold_set = set(gold_triples)
        pred_set = set(pred_triples)

        tp = len(gold_set & pred_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        # per-relation-type
        rel_types = set(t[1] for t in gold_set | pred_set)
        per_type = {}
        for rt in sorted(rel_types):
            rt_gold = {t for t in gold_set if t[1] == rt}
            rt_pred = {t for t in pred_set if t[1] == rt}
            rt_tp = len(rt_gold & rt_pred)
            rt_fp = len(rt_pred - rt_gold)
            rt_fn = len(rt_gold - rt_pred)
            rt_p = rt_tp / (rt_tp + rt_fp) if (rt_tp + rt_fp) > 0 else 0.0
            rt_r = rt_tp / (rt_tp + rt_fn) if (rt_tp + rt_fn) > 0 else 0.0
            rt_f1 = 2 * rt_p * rt_r / (rt_p + rt_r) if (rt_p + rt_r) > 0 else 0.0
            per_type[rt] = {
                'precision': round(rt_p, 4), 'recall': round(rt_r, 4),
                'f1': round(rt_f1, 4), 'support': len(rt_gold)
            }

        return {
            'overall': {'precision': round(p, 4), 'recall': round(r, 4),
                        'f1': round(f1, 4), 'support': len(gold_set)},
            'per_type': per_type
        }

    @staticmethod
    def compute_relaxed_relation_metrics(
        gold_triples: List[Tuple[str, str, str]],
        pred_triples: List[Tuple[str, str, str]],
    ) -> Dict:
        """
        宽松匹配: subject/object 允许部分匹配 (包含关系)
        """
        tp = 0
        matched_gold = set()

        for ps, pr, po in pred_triples:
            for gi, (gs, gr, go) in enumerate(gold_triples):
                if gi in matched_gold:
                    continue
                # 关系类型必须匹配
                if pr != gr:
                    continue
                # subject/object 允许包含匹配
                if (ps in gs or gs in ps) and (po in go or go in po):
                    tp += 1
                    matched_gold.add(gi)
                    break

        fp = len(pred_triples) - tp
        fn = len(gold_triples) - len(matched_gold)

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        return {
            'relaxed_match': {
                'precision': round(p, 4), 'recall': round(r, 4),
                'f1': round(f1, 4)
            }
        }


# ============================================================
# 3. 知识图谱质量分析
# ============================================================

class KnowledgeGraphAnalyzer:
    """知识图谱结构与质量分析"""

    def __init__(self, entities: List[Dict], relations: List[Dict]):
        self.entities = entities
        self.relations = relations
        self._build_graph()

    def _build_graph(self):
        """构建邻接表"""
        self.adj = defaultdict(set)       # 有向
        self.adj_undirected = defaultdict(set)  # 无向 (用于连通分量)
        self.in_degree = defaultdict(int)
        self.out_degree = defaultdict(int)
        self.nodes = {e['text'] for e in self.entities}

        for r in self.relations:
            subj = r.get('subject', '')
            obj = r.get('object', '')
            if subj and obj:
                self.adj[subj].add(obj)
                self.adj_undirected[subj].add(obj)
                self.adj_undirected[obj].add(subj)
                self.out_degree[subj] += 1
                self.in_degree[obj] += 1

    def structural_analysis(self) -> Dict:
        """图结构分析"""
        num_nodes = len(self.nodes)
        num_edges = len(self.relations)

        # 度分布
        degrees = []
        for node in self.nodes:
            d = self.in_degree.get(node, 0) + self.out_degree.get(node, 0)
            degrees.append(d)

        degrees = np.array(degrees) if degrees else np.array([0])

        # 连通分量 (BFS)
        visited = set()
        components = 0
        for node in self.nodes:
            if node not in visited:
                components += 1
                queue = [node]
                while queue:
                    current = queue.pop(0)
                    if current in visited:
                        continue
                    visited.add(current)
                    for neighbor in self.adj_undirected.get(current, set()):
                        if neighbor not in visited:
                            queue.append(neighbor)

        # 孤立节点 (degree = 0)
        isolated = sum(1 for d in degrees if d == 0)

        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': round(num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0, 6),
            'avg_degree': round(float(degrees.mean()), 2),
            'max_degree': int(degrees.max()),
            'min_degree': int(degrees.min()),
            'degree_std': round(float(degrees.std()), 2),
            'num_connected_components': components,
            'num_isolated_nodes': isolated,
            'isolation_ratio': round(isolated / num_nodes if num_nodes > 0 else 0, 4),
        }

    def entity_type_analysis(self) -> Dict:
        """实体类型分布分析"""
        type_counts = Counter(e.get('type', 'UNKNOWN') for e in self.entities)
        total = len(self.entities)
        return {
            'type_distribution': dict(type_counts),
            'type_proportions': {t: round(c / total, 4) for t, c in type_counts.items()},
            'num_types': len(type_counts),
            'total_entities': total,
        }

    def relation_type_analysis(self) -> Dict:
        """关系类型分布分析"""
        type_counts = Counter(r.get('relation', 'UNKNOWN') for r in self.relations)
        total = len(self.relations)
        return {
            'type_distribution': dict(type_counts),
            'type_proportions': {t: round(c / total, 4) for t, c in type_counts.items()},
            'num_types': len(type_counts),
            'total_relations': total,
        }

    def source_analysis(self) -> Dict:
        """来源分布分析 (用于论文中比较不同提取方法的贡献)"""
        entity_sources = Counter(e.get('source', 'unknown') for e in self.entities)
        relation_sources = Counter(r.get('source', 'unknown') for r in self.relations)
        return {
            'entity_sources': dict(entity_sources),
            'relation_sources': dict(relation_sources),
        }

    def confidence_analysis(self) -> Dict:
        """置信度分布分析"""
        entity_confs = [e.get('confidence', 0) for e in self.entities if 'confidence' in e]
        relation_confs = [r.get('confidence', 0) for r in self.relations if 'confidence' in r]

        def _stats(values):
            if not values:
                return {'count': 0}
            arr = np.array(values)
            return {
                'count': len(arr),
                'mean': round(float(arr.mean()), 4),
                'std': round(float(arr.std()), 4),
                'min': round(float(arr.min()), 4),
                'max': round(float(arr.max()), 4),
                'median': round(float(np.median(arr)), 4),
                'q25': round(float(np.percentile(arr, 25)), 4),
                'q75': round(float(np.percentile(arr, 75)), 4),
            }

        return {
            'entity_confidence': _stats(entity_confs),
            'relation_confidence': _stats(relation_confs),
        }

    def coverage_analysis(self, textbook_keywords: List[str] = None) -> Dict:
        """
        覆盖率分析: 实体是否覆盖了教材的核心知识点

        Args:
            textbook_keywords: 教材核心关键词列表 (可选)
        """
        if textbook_keywords is None:
            # 初中数学核心概念列表 (可扩展)
            textbook_keywords = [
                # 七年级
                '有理数', '整数', '分数', '负数', '正数', '数轴', '绝对值', '相反数',
                '加法', '减法', '乘法', '除法', '乘方', '代数式', '方程', '一元一次方程',
                '几何体', '棱柱', '圆柱', '圆锥', '直线', '射线', '线段', '角',
                '平行线', '垂直', '三角形', '全等三角形',
                # 八年级
                '勾股定理', '实数', '平方根', '立方根', '二次根式',
                '一次函数', '正比例函数', '平行四边形', '矩形', '菱形', '正方形',
                '分式', '分式方程', '数据分析', '中位数', '众数', '方差',
                # 九年级
                '一元二次方程', '二次函数', '反比例函数', '相似三角形',
                '锐角三角函数', '圆', '弧', '圆心角', '概率', '随机事件',
                '投影', '视图',
            ]

        entity_texts = {e['text'] for e in self.entities}
        covered = []
        uncovered = []

        for kw in textbook_keywords:
            found = any(kw in et or et in kw for et in entity_texts)
            if found:
                covered.append(kw)
            else:
                uncovered.append(kw)

        return {
            'total_keywords': len(textbook_keywords),
            'covered': len(covered),
            'uncovered': len(uncovered),
            'coverage_ratio': round(len(covered) / len(textbook_keywords), 4) if textbook_keywords else 0,
            'covered_keywords': covered,
            'uncovered_keywords': uncovered,
        }

    def hub_analysis(self, top_k: int = 20) -> Dict:
        """Hub 节点分析: 找出连接最多的核心知识点"""
        degree_map = {}
        for node in self.nodes:
            d = self.in_degree.get(node, 0) + self.out_degree.get(node, 0)
            degree_map[node] = d

        sorted_nodes = sorted(degree_map.items(), key=lambda x: -x[1])[:top_k]
        return {
            'top_hubs': [{'entity': n, 'degree': d,
                          'in_degree': self.in_degree.get(n, 0),
                          'out_degree': self.out_degree.get(n, 0)}
                         for n, d in sorted_nodes]
        }

    def full_report(self) -> Dict:
        """生成完整的 KG 质量报告"""
        return {
            'structural': self.structural_analysis(),
            'entity_types': self.entity_type_analysis(),
            'relation_types': self.relation_type_analysis(),
            'sources': self.source_analysis(),
            'confidence': self.confidence_analysis(),
            'coverage': self.coverage_analysis(),
            'hubs': self.hub_analysis(),
        }


# ============================================================
# 4. 消融实验支持
# ============================================================

class AblationStudy:
    """消融实验框架"""

    def __init__(self):
        self.results = {}

    def run_config(self, config_name: str, pipeline_fn, **kwargs) -> Dict:
        """
        运行单个配置并记录结果

        Args:
            config_name: 配置名称 (如 "bert+kggen", "rule_only", "bert_only")
            pipeline_fn: 管道执行函数，返回 (entities, relations)
            **kwargs: 额外参数
        """
        logger.info(f"消融实验: 运行配置 [{config_name}]")
        entities, relations = pipeline_fn(**kwargs)

        analyzer = KnowledgeGraphAnalyzer(entities, relations)
        report = analyzer.full_report()

        self.results[config_name] = {
            'num_entities': len(entities),
            'num_relations': len(relations),
            'report': report,
        }
        return self.results[config_name]

    def compare(self) -> Dict:
        """比较所有配置的结果"""
        if len(self.results) < 2:
            return {'message': '需要至少两个配置才能比较'}

        comparison = {}
        for name, result in self.results.items():
            comparison[name] = {
                'entities': result['num_entities'],
                'relations': result['num_relations'],
                'coverage': result['report']['coverage']['coverage_ratio'],
                'avg_degree': result['report']['structural']['avg_degree'],
                'components': result['report']['structural']['num_connected_components'],
                'isolation_ratio': result['report']['structural']['isolation_ratio'],
                'entity_types': result['report']['entity_types']['num_types'],
                'relation_types': result['report']['relation_types']['num_types'],
            }

        return comparison

    def save_results(self, output_path: str):
        """保存消融实验结果"""
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'ablation_results': {
                    name: {
                        'num_entities': r['num_entities'],
                        'num_relations': r['num_relations'],
                        'report': r['report'],
                    } for name, r in self.results.items()
                },
                'comparison': self.compare(),
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"消融实验结果已保存到: {output_path}")


# ============================================================
# 5. 论文指标汇总
# ============================================================

def generate_paper_metrics(entities: List[Dict], relations: List[Dict],
                           output_dir: str = None) -> Dict:
    """
    一键生成论文所需的所有指标

    Returns:
        完整的评估报告字典
    """
    if output_dir is None:
        output_dir = Config.OUTPUT_DIR

    analyzer = KnowledgeGraphAnalyzer(entities, relations)
    report = analyzer.full_report()

    # 添加汇总统计
    report['summary'] = {
        'total_entities': len(entities),
        'total_relations': len(relations),
        'entity_type_count': report['entity_types']['num_types'],
        'relation_type_count': report['relation_types']['num_types'],
        'coverage_ratio': report['coverage']['coverage_ratio'],
        'avg_node_degree': report['structural']['avg_degree'],
        'graph_density': report['structural']['density'],
        'connected_components': report['structural']['num_connected_components'],
    }

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'paper_metrics.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"论文指标已保存到: {report_path}")

    return report


if __name__ == '__main__':
    # 从已有输出加载
    entities_path = os.path.join(Config.OUTPUT_DIR, 'kggen_entities.json')
    relations_path = os.path.join(Config.OUTPUT_DIR, 'kggen_relations.json')

    if os.path.exists(entities_path) and os.path.exists(relations_path):
        with open(entities_path, 'r', encoding='utf-8') as f:
            entities = json.load(f)
        with open(relations_path, 'r', encoding='utf-8') as f:
            relations = json.load(f)

        report = generate_paper_metrics(entities, relations)
        print(json.dumps(report['summary'], ensure_ascii=False, indent=2))
        print(f"\n覆盖率: {report['coverage']['coverage_ratio']:.1%}")
        print(f"未覆盖关键词: {report['coverage']['uncovered_keywords']}")
    else:
        print("请先运行管道生成实体和关系数据")
