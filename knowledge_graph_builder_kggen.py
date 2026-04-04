import os
import json
import re
from typing import List, Dict
from tqdm import tqdm
from data_loader import MathDataLoader
from entity_extractor_kggen import KGGenEnhancedEntityExtractor
from relation_extractor_kggen import KGGenEnhancedRelationExtractor
from neo4j_manager import Neo4jManager
from config import Config
from performance_optimizer import PerformanceMonitor, time_it
import torch
from torch import nn
from kggen_client import KGGenClient
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KGGenEnhancedKnowledgeGraphBuilder:
    """KGGen增强的知识图谱构建器"""
    
    def __init__(self, device: str = None, use_kggen: bool = True):
        if device is None:
            cuda_available = torch.cuda.is_available()
            logger.info(f"检测到CUDA可用: {cuda_available}")
            device = 'cuda' if cuda_available else 'cpu'
        
        self.data_loader = MathDataLoader()
        self.entity_extractor = KGGenEnhancedEntityExtractor(use_kggen=use_kggen)
        self.relation_extractor = KGGenEnhancedRelationExtractor()
        self.neo4j_manager = Neo4jManager()
        self.device = device
        self.use_kggen = use_kggen
        
        if use_kggen:
            try:
                self.kggen_client = KGGenClient()
                logger.info("[OK] KGGen客户端初始化成功")
            except Exception as e:
                logger.warning(f"KGGen客户端初始化失败: {e}")
                self.use_kggen = False
        
        # 显示设备信息
        logger.info(f"运行设备: {device}")
        if device == 'cuda':
            logger.info(f"GPU名称: {torch.cuda.get_device_name()}")
        else:
            logger.info("使用CPU运行")

    def build_knowledge_graph(self) -> tuple:
        """构建完整的知识图谱（混合模式）"""
        monitor = PerformanceMonitor()
        monitor.start_timer('total_build_process')
        
        logger.info("开始构建数学知识图谱...")
        
        # 1. 加载数据
        logger.info("步骤1: 加载数学教材数据")
        monitor.start_timer('data_loading')
        data = self.data_loader.load_all_data()
        monitor.end_timer('data_loading')
        logger.info(f"成功加载 {len(data)} 个文件")
        
        # 2. 实体识别和关系抽取
        logger.info("步骤2: 实体识别和关系抽取")
        all_entities = []
        all_relations = []
        
        monitor.start_timer('entity_relation_extraction')
        for item in tqdm(data, desc="处理文件"):
            try:
                text = item['content']
                grade = item.get('grade')
                chapter = item.get('chapter')
                file_name = item.get('file_name')
                
                # 使用KGGen增强的提取
                extraction_result = self._extract_with_kggen_enhancement(text)
                
                # 添加元数据
                for entity in extraction_result["entities"]:
                    entity.update({
                        'grade': grade,
                        'chapter': chapter,
                        'source_file': file_name
                    })
                
                for relation in extraction_result["relations"]:
                    relation.update({
                        'grade': grade,
                        'chapter': chapter,
                        'source_file': file_name
                    })
                
                all_entities.extend(extraction_result["entities"])
                all_relations.extend(extraction_result["relations"])
                
            except Exception as e:
                logger.error(f"处理文件失败: {e}")
                continue
        
        monitor.end_timer('entity_relation_extraction')
        logger.info(f"识别到 {len(all_entities)} 个实体")
        logger.info(f"抽取到 {len(all_relations)} 个关系")
        
        # 3. 后处理
        logger.info("步骤3: 数据后处理")
        monitor.start_timer('postprocessing')
        processed_entities = self._postprocess_entities(all_entities)
        processed_relations = self._postprocess_relations(all_relations)
        monitor.end_timer('postprocessing')
        
        # 4. 数据去重
        logger.info("步骤4: 数据去重")
        monitor.start_timer('deduplication')
        unique_entities = self._deduplicate_entities(processed_entities)
        unique_relations = self._deduplicate_relations(processed_relations)
        monitor.end_timer('deduplication')
        
        logger.info(f"去重后: {len(unique_entities)} 个实体, {len(unique_relations)} 个关系")
        
        # 5. 添加层级关系
        logger.info("步骤5: 构建层级结构")
        monitor.start_timer('hierarchy_building')
        self._add_hierarchical_relations(unique_entities, unique_relations)
        monitor.end_timer('hierarchy_building')
        
        # 6. 导入Neo4j
        logger.info("步骤6: 导入Neo4j图数据库")
        monitor.start_timer('neo4j_import')
        self.neo4j_manager.import_knowledge_graph(unique_entities, unique_relations)
        monitor.end_timer('neo4j_import')
        
        # 7. 保存中间结果
        logger.info("步骤7: 保存处理结果")
        monitor.start_timer('saving_results')
        self._save_results(unique_entities, unique_relations)
        monitor.end_timer('saving_results')
        
        monitor.end_timer('total_build_process')
        
        # 打印性能统计
        monitor.print_stats()
        
        logger.info("知识图谱构建完成！")
        
        return unique_entities, unique_relations

    def build_knowledge_graph_with_kggen_direct(self) -> tuple:
        """使用KGGen直接模式构建知识图谱"""
        logger.info("使用KGGen直接模式构建知识图谱...")
        
        monitor = PerformanceMonitor()
        monitor.start_timer('kggen_direct_build')
        
        # 1. 加载数据
        data = self.data_loader.load_all_data()
        logger.info(f"成功加载 {len(data)} 个文件")
        
        all_entities = []
        all_relations = []
        
        for item in tqdm(data, desc="KGGen直接处理"):
            try:
                text = item['content']
                grade = item.get('grade')
                chapter = item.get('chapter')
                file_name = item.get('file_name')
                
                # 直接使用KGGen API提取
                if self.use_kggen and self.kggen_client:
                    result = self.kggen_client.extract_entities_and_relations(text)
                    
                    # 添加元数据
                    for entity in result["entities"]:
                        entity.update({
                            'grade': grade,
                            'chapter': chapter,
                            'source_file': file_name,
                            'source': 'kggen_direct'
                        })
                    
                    for relation in result["relations"]:
                        relation.update({
                            'grade': grade,
                            'chapter': chapter,
                            'source_file': file_name,
                            'source': 'kggen_direct'
                        })
                    
                    all_entities.extend(result["entities"])
                    all_relations.extend(result["relations"])
                
            except Exception as e:
                logger.error(f"KGGen直接处理失败: {e}")
                continue
        
        # 后处理和去重
        processed_entities = self._postprocess_entities(all_entities)
        processed_relations = self._postprocess_relations(all_relations)
        unique_entities = self._deduplicate_entities(processed_entities)
        unique_relations = self._deduplicate_relations(processed_relations)
        
        # 导入Neo4j
        self.neo4j_manager.import_knowledge_graph(unique_entities, unique_relations)
        self._save_results(unique_entities, unique_relations)
        
        monitor.end_timer('kggen_direct_build')
        monitor.print_stats()
        
        return unique_entities, unique_relations

    def _extract_with_kggen_enhancement(self, text: str) -> Dict:
        """使用KGGen增强的实体和关系提取"""
        result = {"entities": [], "relations": []}
        
        try:
            # 首先使用本地方法提取
            entities = self.entity_extractor.extract_entities(text)
            
            # 使用KGGen提取关系（如果可用）
            if self.use_kggen and self.kggen_client:
                kggen_result = self.kggen_client.extract_entities_and_relations(text)
                
                # 合并实体（优先保留本地提取的实体）
                merged_entities = self._merge_entities(entities, kggen_result.get("entities", []))
                
                # 验证和合并关系
                valid_relations = []
                entity_texts = {e["text"] for e in merged_entities}
                
                for rel in kggen_result.get("relations", []):
                    if (rel["subject"] in entity_texts and 
                        rel["object"] in entity_texts):
                        rel["source"] = "kggen"
                        rel["confidence"] = rel.get("confidence", 0.9)
                        valid_relations.append(rel)
                
                result["entities"] = merged_entities
                result["relations"] = valid_relations
            else:
                # 仅使用本地方法
                relations = self.relation_extractor.extract_relations(text, entities)
                result["entities"] = entities
                result["relations"] = relations
                
        except Exception as e:
            logger.error(f"KGGen增强提取失败: {e}")
            # 回退到本地方法
            entities = self.entity_extractor.extract_entities(text)
            relations = self.relation_extractor.extract_relations(text, entities)
            result["entities"] = entities
            result["relations"] = relations
        
        return result

    def _merge_entities(self, local_entities: List[Dict], kggen_entities: List[Dict]) -> List[Dict]:
        """合并本地和KGGen提取的实体"""
        merged = local_entities.copy()
        local_entity_map = {e["text"]: e for e in local_entities}
        
        for kggen_entity in kggen_entities:
            if kggen_entity["text"] not in local_entity_map:
                # KGGen发现了新的实体
                kggen_entity["source"] = "kggen"
                kggen_entity["confidence"] = kggen_entity.get("confidence", 0.8)
                merged.append(kggen_entity)
            else:
                # 实体已存在，可以选择性地更新信息
                existing_entity = local_entity_map[kggen_entity["text"]]
                # 可以在这里实现更复杂的合并逻辑
        
        return merged

    def _postprocess_entities(self, entities: List[Dict]) -> List[Dict]:
        """实体后处理"""
        processed_entities = []
        
        for entity in entities:
            # 过滤太短的实体
            if len(entity['text'].strip()) < 2:
                continue
                
            # 验证实体类型
            if not self._validate_entity_type(entity['text'], entity['type']):
                # 尝试重新分类
                new_type = self._reclassify_entity(entity['text'])
                if new_type:
                    entity['type'] = new_type
                else:
                    continue
            
            processed_entities.append(entity)
        
        return processed_entities

    def _postprocess_relations(self, relations: List[Dict]) -> List[Dict]:
        """关系后处理"""
        processed_relations = []
        
        for relation in relations:
            # 过滤无效的关系
            if (not relation['subject'] or not relation['object'] or 
                relation['subject'] == relation['object']):
                continue
                
            processed_relations.append(relation)
        
        return processed_relations

    def _validate_entity_type(self, text: str, entity_type: str) -> bool:
        """验证实体类型是否合理"""
        # 简单的验证逻辑
        if entity_type == 'FORMULA':
            return any(c in '²³⁴⁵⁶⁷⁸⁹⁰⁺⁻⁼⁽⁾∀∃∫∑∏√∞πθφΔ∇∂' for c in text) or any(c.isalpha() for c in text)
        
        if entity_type == 'THEOREM':
            return any(keyword in text for keyword in ['定理', '公式', '法则', '原理'])
        
        return True

    def _reclassify_entity(self, text: str) -> str:
        """重新分类实体"""
        if any(c in '²³⁴⁵⁶⁷⁸⁹⁰⁺⁻⁼⁽⁾∀∃∫∑∏√∞πθφΔ∇∂' for c in text):
            return 'FORMULA'
        
        if any(keyword in text for keyword in ['定理', '公式', '法则', '原理']):
            return 'THEOREM'
        
        if any(keyword in text for keyword in ['方法', '解法', '算法', '步骤']):
            return 'METHOD'
        
        return 'CONCEPT'

    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """实体去重"""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            key = (
                entity['text'],
                entity['type'],
                entity.get('grade'),
                entity.get('chapter')
            )
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities

    def _deduplicate_relations(self, relations: List[Dict]) -> List[Dict]:
        """关系去重"""
        seen = set()
        unique_relations = []
        
        for rel in relations:
            key = (rel['subject'], rel['relation'], rel['object'])
            if key not in seen:
                seen.add(key)
                unique_relations.append(rel)
        
        return unique_relations

    def _add_hierarchical_relations(self, entities: List[Dict], relations: List[Dict]):
        """添加教材结构层级关系"""
        # 创建章节节点映射
        chapter_map = {
            (e['grade'], e['chapter']): e['text']
            for e in entities if e['type'] == 'CHAPTER_TITLE'
        }
        
        # 添加隶属关系
        for entity in entities:
            if entity['type'] != 'CHAPTER_TITLE' and 'grade' in entity:
                chapter_key = (entity['grade'], entity['chapter'])
                if chapter_key in chapter_map:
                    relations.append({
                        'subject': chapter_map[chapter_key],
                        'relation': 'BELONGS_TO',
                        'object': entity['text'],
                        'source': 'hierarchy',
                        'confidence': 1.0
                    })

    def _save_results(self, entities: List[Dict], relations: List[Dict]):
        """保存处理结果"""
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        
        # 保存实体数据
        entities_path = os.path.join(Config.OUTPUT_DIR, "kggen_entities.json")
        with open(entities_path, 'w', encoding='utf-8') as f:
            json.dump(entities, f, ensure_ascii=False, indent=2)
        
        # 保存关系数据
        relations_path = os.path.join(Config.OUTPUT_DIR, "kggen_relations.json")
        with open(relations_path, 'w', encoding='utf-8') as f:
            json.dump(relations, f, ensure_ascii=False, indent=2)
        
        # 保存统计信息
        stats = {
            'total_entities': len(entities),
            'total_relations': len(relations),
            'entity_types': {},
            'relation_types': {},
            'sources': {
                'entities': {},
                'relations': {}
            }
        }
        
        # 统计实体类型
        for entity in entities:
            entity_type = entity['type']
            stats['entity_types'][entity_type] = stats['entity_types'].get(entity_type, 0) + 1
            source = entity.get('source', 'unknown')
            stats['sources']['entities'][source] = stats['sources']['entities'].get(source, 0) + 1
        
        # 统计关系类型
        for rel in relations:
            rel_type = rel['relation']
            stats['relation_types'][rel_type] = stats['relation_types'].get(rel_type, 0) + 1
            source = rel.get('source', 'unknown')
            stats['sources']['relations'][source] = stats['sources']['relations'].get(source, 0) + 1
        
        stats_path = os.path.join(Config.OUTPUT_DIR, "kggen_statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"KGGen增强结果已保存到 {Config.OUTPUT_DIR} 目录")

# 使用示例
if __name__ == "__main__":
    builder = KGGenEnhancedKnowledgeGraphBuilder()
    
    try:
        # 构建知识图谱
        entities, relations = builder.build_knowledge_graph()
        
        # 示例查询
        print("\n构建完成统计:")
        print(f"实体总数: {len(entities)}")
        print(f"关系总数: {len(relations)}")
        
    except Exception as e:
        print(f"构建过程中出错: {e}")
        import traceback
        traceback.print_exc()