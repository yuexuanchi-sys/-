#!/usr/bin/env python3
"""
KGGen增强的数学知识图谱构建主程序
集成本地模型和KGGen API的完整知识图谱构建管道
"""

import os
import argparse
import json
from typing import Dict, List
from config import Config
from knowledge_graph_builder_kggen import KGGenEnhancedKnowledgeGraphBuilder
from entity_extractor_kggen import KGGenEnhancedEntityExtractor
from relation_extractor_kggen import KGGenEnhancedRelationExtractor
from kggen_client import KGGenClient
import logging

# 配置日志
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
        self.use_kggen = use_kggen
        self.kggen_client = KGGenClient() if use_kggen else None
        self.entity_extractor = KGGenEnhancedEntityExtractor(use_kggen=use_kggen)
        self.relation_extractor = KGGenEnhancedRelationExtractor()
        self.knowledge_graph_builder = KGGenEnhancedKnowledgeGraphBuilder(use_kggen=use_kggen)
        
        logger.info(f"KGGen集成: {'启用' if use_kggen else '禁用'}")
    
    def run_full_pipeline(self, mode: str = "hybrid") -> Dict:
        """
        运行完整的知识图谱构建管道
        
        Args:
            mode: 构建模式 - "hybrid"(混合), "kggen_direct"(KGGen直接), "local_only"(仅本地)
            
        Returns:
            Dict: 构建结果统计
        """
        logger.info("=" * 60)
        logger.info("启动KGGen增强的数学知识图谱构建管道")
        logger.info("=" * 60)
        
        results = {}
        
        try:
            if mode == "kggen_direct":
                logger.info("使用KGGen直接模式构建知识图谱...")
                entities, relations = self.knowledge_graph_builder.build_knowledge_graph_with_kggen_direct()
            elif mode == "local_only":
                logger.info("使用仅本地模式构建知识图谱...")
                # 临时禁用KGGen
                self.knowledge_graph_builder.use_kggen = False
                entities, relations = self.knowledge_graph_builder.build_knowledge_graph()
            else:
                logger.info("使用混合模式构建知识图谱...")
                entities, relations = self.knowledge_graph_builder.build_knowledge_graph()
            
            # 统计结果
            results = self._analyze_results(entities, relations)
            
            logger.info("知识图谱构建完成！")
            logger.info(f"实体总数: {results['total_entities']}")
            logger.info(f"关系总数: {results['total_relations']}")
            logger.info(f"实体类型分布: {results['entity_type_distribution']}")
            logger.info(f"关系类型分布: {results['relation_type_distribution']}")
            logger.info(f"来源分布 - 实体: {results['source_distribution_entities']}")
            logger.info(f"来源分布 - 关系: {results['source_distribution_relations']}")
            
        except Exception as e:
            logger.error(f"管道执行失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results = {"error": str(e)}
        
        return results
    
    def extract_from_text(self, text: str) -> Dict:
        """从单个文本中提取实体和关系"""
        logger.info(f"处理文本: {text[:100]}...")
        
        try:
            # 提取实体
            entities = self.entity_extractor.extract_entities(text)
            
            # 提取关系
            relations = self.relation_extractor.extract_relations(text, entities)
            
            result = {
                "text": text,
                "entities": entities,
                "relations": relations,
                "stats": {
                    "entity_count": len(entities),
                    "relation_count": len(relations),
                    "entity_types": {},
                    "relation_types": {},
                    "sources": {
                        "entities": {},
                        "relations": {}
                    }
                }
            }
            
            # 统计信息
            for entity in entities:
                result["stats"]["entity_types"][entity["type"]] = result["stats"]["entity_types"].get(entity["type"], 0) + 1
                result["stats"]["sources"]["entities"][entity.get("source", "unknown")] = result["stats"]["sources"]["entities"].get(entity.get("source", "unknown"), 0) + 1
            
            for relation in relations:
                result["stats"]["relation_types"][relation["relation"]] = result["stats"]["relation_types"].get(relation["relation"], 0) + 1
                result["stats"]["sources"]["relations"][relation.get("source", "unknown")] = result["stats"]["sources"]["relations"].get(relation.get("source", "unknown"), 0) + 1
            
            return result
            
        except Exception as e:
            logger.error(f"文本处理失败: {e}")
            return {"error": str(e), "text": text}
    
    def batch_extract(self, texts: List[str]) -> List[Dict]:
        """批量处理多个文本"""
        results = []
        
        for i, text in enumerate(texts):
            logger.info(f"处理第 {i+1}/{len(texts)} 个文本")
            result = self.extract_from_text(text)
            results.append(result)
            
            # 避免速率限制
            if self.use_kggen:
                import time
                time.sleep(0.5)
        
        return results
    
    def _analyze_results(self, entities: List[Dict], relations: List[Dict]) -> Dict:
        """分析构建结果"""
        analysis = {
            "total_entities": len(entities),
            "total_relations": len(relations),
            "entity_type_distribution": {},
            "relation_type_distribution": {},
            "source_distribution_entities": {},
            "source_distribution_relations": {},
            "confidence_stats": {
                "entities": {"min": 1.0, "max": 0.0, "avg": 0.0},
                "relations": {"min": 1.0, "max": 0.0, "avg": 0.0}
            }
        }
        
        # 实体统计
        entity_confidences = []
        for entity in entities:
            entity_type = entity["type"]
            analysis["entity_type_distribution"][entity_type] = analysis["entity_type_distribution"].get(entity_type, 0) + 1
            
            source = entity.get("source", "unknown")
            analysis["source_distribution_entities"][source] = analysis["source_distribution_entities"].get(source, 0) + 1
            
            confidence = entity.get("confidence", 0.5)
            entity_confidences.append(confidence)
        
        # 关系统计
        relation_confidences = []
        for relation in relations:
            relation_type = relation["relation"]
            analysis["relation_type_distribution"][relation_type] = analysis["relation_type_distribution"].get(relation_type, 0) + 1
            
            source = relation.get("source", "unknown")
            analysis["source_distribution_relations"][source] = analysis["source_distribution_relations"].get(source, 0) + 1
            
            confidence = relation.get("confidence", 0.5)
            relation_confidences.append(confidence)
        
        # 置信度统计
        if entity_confidences:
            analysis["confidence_stats"]["entities"]["min"] = min(entity_confidences)
            analysis["confidence_stats"]["entities"]["max"] = max(entity_confidences)
            analysis["confidence_stats"]["entities"]["avg"] = sum(entity_confidences) / len(entity_confidences)
        
        if relation_confidences:
            analysis["confidence_stats"]["relations"]["min"] = min(relation_confidences)
            analysis["confidence_stats"]["relations"]["max"] = max(relation_confidences)
            analysis["confidence_stats"]["relations"]["avg"] = sum(relation_confidences) / len(relation_confidences)
        
        return analysis
    
    def export_results(self, results: Dict, output_dir: str = None):
        """导出结果到文件"""
        if output_dir is None:
            output_dir = Config.OUTPUT_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存详细结果
        detailed_path = os.path.join(output_dir, "kggen_detailed_results.json")
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"详细结果已保存到: {detailed_path}")
        
        return detailed_path

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='KGGen增强的数学知识图谱构建管道')
    parser.add_argument('--mode', type=str, choices=['hybrid', 'kggen_direct', 'local_only', 'extract'], 
                       default='hybrid', help='运行模式')
    parser.add_argument('--no-kggen', action='store_true', help='禁用KGGen集成')
    parser.add_argument('--text', type=str, help='要处理的文本（extract模式使用）')
    parser.add_argument('--input-file', type=str, help='包含多个文本的文件路径（每行一个文本）')
    parser.add_argument('--output-dir', type=str, default=Config.OUTPUT_DIR, help='输出目录')
    
    args = parser.parse_args()
    
    # 创建管道
    pipeline = KGGenMathKnowledgePipeline(use_kggen=not args.no_kggen)
    
    if args.mode == 'extract':
        # 单个文本提取模式
        if args.text:
            result = pipeline.extract_from_text(args.text)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.input_file:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            results = pipeline.batch_extract(texts)
            pipeline.export_results(results, args.output_dir)
            print(f"处理完成，结果已保存到 {args.output_dir}")
        else:
            print("请提供 --text 或 --input-file 参数")
    
    else:
        # 知识图谱构建模式
        results = pipeline.run_full_pipeline(mode=args.mode)
        
        if "error" not in results:
            # 导出结果
            pipeline.export_results(results, args.output_dir)
            
            # 打印摘要
            print("\n" + "=" * 60)
            print("构建结果摘要")
            print("=" * 60)
            print(f"实体总数: {results['total_entities']}")
            print(f"关系总数: {results['total_relations']}")
            print(f"\n实体类型分布:")
            for entity_type, count in results['entity_type_distribution'].items():
                print(f"  {entity_type}: {count}")
            print(f"\n关系类型分布:")
            for relation_type, count in results['relation_type_distribution'].items():
                print(f"  {relation_type}: {count}")
            print(f"\n来源分布 - 实体:")
            for source, count in results['source_distribution_entities'].items():
                print(f"  {source}: {count}")
            print(f"\n来源分布 - 关系:")
            for source, count in results['source_distribution_relations'].items():
                print(f"  {source}: {count}")
            
            print(f"\n结果已保存到: {args.output_dir}")
        else:
            print(f"构建失败: {results['error']}")

if __name__ == "__main__":
    main()