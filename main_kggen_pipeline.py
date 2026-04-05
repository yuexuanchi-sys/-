#!/usr/bin/env python3
"""
KGGen增强的数学知识图谱构建主程序

支持模式:
  hybrid       - 本地BERT + KGGen API 混合提取
  kggen_direct - 仅使用 KGGen API
  local_only   - 仅使用本地模型 (规则+词性+BERT)
  train        - 生成训练数据 + BERT NER 训练
  evaluate     - 评估已有结果 (KG质量分析)
  ablation     - 消融实验 (对比不同配置)
"""

import os
import sys
import random
import argparse
import json
import time
import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from config import Config

# ── 可复现性: 固定随机种子 ──────────────────────
SEED = 42

def set_seed(seed: int = SEED):
    """固定所有随机源，保证可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed()

# ── 日志 ──────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("kggen_pipeline.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class KGGenMathKnowledgePipeline:
    """KGGen增强的数学知识图谱构建管道"""

    def __init__(self, use_kggen: bool = True):
        from knowledge_graph_builder_kggen import KGGenEnhancedKnowledgeGraphBuilder

        self.use_kggen = use_kggen
        self.knowledge_graph_builder = KGGenEnhancedKnowledgeGraphBuilder(use_kggen=use_kggen)
        self.entity_extractor = self.knowledge_graph_builder.entity_extractor
        self.relation_extractor = self.knowledge_graph_builder.relation_extractor
        self.kggen_client = self.knowledge_graph_builder.kggen_client
        logger.info(f"KGGen集成: {'启用' if use_kggen else '禁用'}")

    # ================================================================
    #  知识图谱构建
    # ================================================================

    def run_full_pipeline(self, mode: str = "hybrid") -> Dict:
        """运行完整的知识图谱构建管道"""
        logger.info("=" * 60)
        logger.info("启动KGGen增强的数学知识图谱构建管道")
        logger.info("=" * 60)

        try:
            if mode == "kggen_direct":
                logger.info("使用KGGen直接模式构建知识图谱...")
                entities, relations = self.knowledge_graph_builder.build_knowledge_graph_with_kggen_direct()
            elif mode == "local_only":
                logger.info("使用仅本地模式构建知识图谱...")
                self.knowledge_graph_builder.use_kggen = False
                entities, relations = self.knowledge_graph_builder.build_knowledge_graph()
            else:
                logger.info("使用混合模式构建知识图谱...")
                entities, relations = self.knowledge_graph_builder.build_knowledge_graph()

            results = self._analyze_results(entities, relations)
            self._log_results(results)

            # 自动生成论文指标
            from evaluation import generate_paper_metrics
            paper_report = generate_paper_metrics(entities, relations)
            results['paper_metrics'] = paper_report.get('summary', {})

            return results

        except Exception as e:
            logger.error(f"管道执行失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": str(e)}

    # ================================================================
    #  NER 训练数据生成 + BERT 训练
    # ================================================================

    def run_training_pipeline(self, epochs: int = 10, batch_size: int = 16,
                               learning_rate: float = 2e-5) -> Dict:
        """
        端到端训练管道:
        1. 从教材生成 NER 训练数据 (BIOES 自动标注)
        2. 训练 BERT+BiLSTM+CRF 模型
        3. 在测试集上评估
        """
        logger.info("=" * 60)
        logger.info("启动 NER 训练管道")
        logger.info("=" * 60)

        # Step 1: 生成训练数据
        logger.info("步骤1: 生成 BIOES 训练数据...")
        from data_loader import MathDataLoader
        from enhanced_data_processor import EnhancedDataProcessor

        loader = MathDataLoader()
        data = loader.load_all_data()
        logger.info(f"加载 {len(data)} 个教材文件")

        processor = EnhancedDataProcessor()
        train_df = processor.create_char_level_ner_training_data(data, scheme='bioes')
        logger.info(f"生成 {len(train_df)} 条 BIOES 训练样本")

        # 保存训练数据
        output_dir = Config.OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)
        train_data_path = os.path.join(output_dir, 'ner_train_bioes.jsonl')
        processor.save_char_level_training_data(train_df, train_data_path)

        # Step 2: 训练 BERT+BiLSTM+CRF
        logger.info("步骤2: 训练 BERT+BiLSTM+CRF 模型...")
        from bert_trainer_v2 import BERTMathNERTrainer

        # 将 DataFrame 转为训练器期望的格式
        training_data = []
        for _, row in train_df.iterrows():
            training_data.append({
                'text': row['text'],
                'char_labels': row['char_labels'],
            })

        trainer = BERTMathNERTrainer(
            model_dir=Config.MODEL_DIR,
            max_length=Config.MAX_SEQ_LENGTH,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=epochs,
        )

        train_results = trainer.train(training_data)
        logger.info(f"训练完成! 最佳 F1: {train_results.get('best_f1', 'N/A')}")

        # Step 3: 评估
        logger.info("步骤3: 在测试集上评估...")
        test_results = trainer.evaluate()

        return {
            'train_samples': len(train_df),
            'training': train_results,
            'evaluation': test_results,
        }

    # ================================================================
    #  消融实验
    # ================================================================

    def run_ablation_study(self) -> Dict:
        """
        运行消融实验，比较不同配置的效果:
        1. local_only (规则+词性)
        2. local_only + 共现关系
        3. hybrid (本地+KGGen)  — 需要 API
        """
        from evaluation import AblationStudy

        ablation = AblationStudy()

        # 配置 1: 仅本地
        logger.info("消融实验 [1/2]: local_only")
        self.knowledge_graph_builder.use_kggen = False
        ablation.run_config(
            'local_only',
            lambda: self.knowledge_graph_builder.build_knowledge_graph()
        )

        # 配置 2: hybrid (如果 KGGen 可用)
        if self.kggen_client:
            logger.info("消融实验 [2/2]: hybrid")
            self.knowledge_graph_builder.use_kggen = True
            ablation.run_config(
                'hybrid',
                lambda: self.knowledge_graph_builder.build_knowledge_graph()
            )
        else:
            logger.info("KGGen 不可用，跳过 hybrid 配置")

        comparison = ablation.compare()
        ablation.save_results(os.path.join(Config.OUTPUT_DIR, 'ablation_results.json'))

        logger.info("消融实验完成!")
        for name, metrics in comparison.items():
            logger.info(f"  [{name}] 实体: {metrics['entities']}, "
                        f"关系: {metrics['relations']}, "
                        f"覆盖率: {metrics['coverage']:.1%}")

        return comparison

    # ================================================================
    #  评估已有结果
    # ================================================================

    def run_evaluation(self) -> Dict:
        """评估已有的知识图谱输出"""
        from evaluation import generate_paper_metrics

        entities_path = os.path.join(Config.OUTPUT_DIR, 'kggen_entities.json')
        relations_path = os.path.join(Config.OUTPUT_DIR, 'kggen_relations.json')

        if not os.path.exists(entities_path) or not os.path.exists(relations_path):
            logger.error("未找到已有输出文件，请先运行构建管道")
            return {"error": "output files not found"}

        with open(entities_path, 'r', encoding='utf-8') as f:
            entities = json.load(f)
        with open(relations_path, 'r', encoding='utf-8') as f:
            relations = json.load(f)

        report = generate_paper_metrics(entities, relations)
        logger.info(f"评估完成: {report['summary']}")
        return report

    # ================================================================
    #  辅助
    # ================================================================

    def extract_from_text(self, text: str) -> Dict:
        """从单个文本提取实体和关系"""
        entities = self.entity_extractor.extract_entities(text)
        relations = self.relation_extractor.extract_relations(text, entities)
        return {"text": text, "entities": entities, "relations": relations}

    def batch_extract(self, texts: List[str]) -> List[Dict]:
        """批量提取"""
        results = []
        for i, text in enumerate(texts):
            logger.info(f"处理第 {i+1}/{len(texts)} 个文本")
            results.append(self.extract_from_text(text))
            if self.use_kggen:
                time.sleep(0.5)
        return results

    def _analyze_results(self, entities: List[Dict], relations: List[Dict]) -> Dict:
        """分析构建结果"""
        from collections import Counter
        analysis = {
            "total_entities": len(entities),
            "total_relations": len(relations),
            "entity_type_distribution": dict(Counter(e['type'] for e in entities)),
            "relation_type_distribution": dict(Counter(r['relation'] for r in relations)),
            "source_distribution_entities": dict(Counter(e.get('source', 'unknown') for e in entities)),
            "source_distribution_relations": dict(Counter(r.get('source', 'unknown') for r in relations)),
        }

        entity_confs = [e.get('confidence', 0.5) for e in entities]
        relation_confs = [r.get('confidence', 0.5) for r in relations]
        analysis["confidence_stats"] = {
            "entities": {"min": min(entity_confs, default=0), "max": max(entity_confs, default=0),
                         "avg": sum(entity_confs) / len(entity_confs) if entity_confs else 0},
            "relations": {"min": min(relation_confs, default=0), "max": max(relation_confs, default=0),
                          "avg": sum(relation_confs) / len(relation_confs) if relation_confs else 0},
        }
        return analysis

    def _log_results(self, results: Dict):
        logger.info("知识图谱构建完成！")
        logger.info(f"实体总数: {results['total_entities']}")
        logger.info(f"关系总数: {results['total_relations']}")
        logger.info(f"实体类型分布: {results['entity_type_distribution']}")
        logger.info(f"关系类型分布: {results['relation_type_distribution']}")
        logger.info(f"来源分布 - 实体: {results['source_distribution_entities']}")
        logger.info(f"来源分布 - 关系: {results['source_distribution_relations']}")

    def export_results(self, results: Dict, output_dir: str = None):
        if output_dir is None:
            output_dir = Config.OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "kggen_detailed_results.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"详细结果已保存到: {path}")


# ================================================================
#  命令行入口
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='KGGen增强的数学知识图谱构建管道')
    parser.add_argument('--mode', type=str,
                        choices=['hybrid', 'kggen_direct', 'local_only',
                                 'train', 'evaluate', 'ablation', 'extract'],
                        default='hybrid', help='运行模式')
    parser.add_argument('--no-kggen', action='store_true', help='禁用KGGen集成')
    parser.add_argument('--text', type=str, help='要处理的文本（extract模式）')
    parser.add_argument('--input-file', type=str, help='输入文件路径')
    parser.add_argument('--output-dir', type=str, default=Config.OUTPUT_DIR, help='输出目录')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='训练批次大小')
    parser.add_argument('--lr', type=float, default=2e-5, help='学习率')
    parser.add_argument('--seed', type=int, default=SEED, help='随机种子')

    args = parser.parse_args()
    set_seed(args.seed)

    if args.mode == 'train':
        pipeline = KGGenMathKnowledgePipeline(use_kggen=False)
        results = pipeline.run_training_pipeline(
            epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr
        )
        print(json.dumps(results, ensure_ascii=False, indent=2, default=str))

    elif args.mode == 'evaluate':
        pipeline = KGGenMathKnowledgePipeline(use_kggen=False)
        report = pipeline.run_evaluation()
        print(json.dumps(report.get('summary', report), ensure_ascii=False, indent=2))

    elif args.mode == 'ablation':
        pipeline = KGGenMathKnowledgePipeline(use_kggen=not args.no_kggen)
        comparison = pipeline.run_ablation_study()
        print(json.dumps(comparison, ensure_ascii=False, indent=2))

    elif args.mode == 'extract':
        pipeline = KGGenMathKnowledgePipeline(use_kggen=not args.no_kggen)
        if args.text:
            result = pipeline.extract_from_text(args.text)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.input_file:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            results = pipeline.batch_extract(texts)
            pipeline.export_results(results, args.output_dir)
        else:
            print("请提供 --text 或 --input-file 参数")

    else:
        pipeline = KGGenMathKnowledgePipeline(use_kggen=not args.no_kggen)
        results = pipeline.run_full_pipeline(mode=args.mode)

        if "error" not in results:
            pipeline.export_results(results, args.output_dir)
            print(f"\n{'='*60}")
            print("构建结果摘要")
            print(f"{'='*60}")
            print(f"实体总数: {results['total_entities']}")
            print(f"关系总数: {results['total_relations']}")
            for k in ['entity_type_distribution', 'relation_type_distribution']:
                print(f"\n{k}:")
                for t, c in results[k].items():
                    print(f"  {t}: {c}")
            if 'paper_metrics' in results:
                pm = results['paper_metrics']
                print(f"\n论文核心指标:")
                print(f"  覆盖率: {pm.get('coverage_ratio', 0):.1%}")
                print(f"  平均节点度: {pm.get('avg_node_degree', 0):.2f}")
                print(f"  图密度: {pm.get('graph_density', 0):.6f}")
        else:
            print(f"构建失败: {results['error']}")


if __name__ == "__main__":
    main()
