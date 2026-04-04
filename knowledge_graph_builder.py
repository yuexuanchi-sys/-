import os
import json
import re
from typing import List, Dict
from tqdm import tqdm
from data_loader import MathDataLoader
from entity_recognizer import EntityRecognizer
from relation_extractor import RelationExtractor
from neo4j_manager import Neo4jManager
from config import Config
from performance_optimizer import PerformanceMonitor, time_it
import torch
from torch import nn

# Add at the top with other imports
class KnowledgeGraphBuilder:
    def __init__(self, device: str = None):
        if device is None:
            cuda_available = torch.cuda.is_available()
            print(f"检测到CUDA可用: {cuda_available}")
            device = 'cuda' if cuda_available else 'cpu'
        self.data_loader = MathDataLoader()
        self.entity_recognizer = EntityRecognizer().to(device)  # <mcsymbol name="EntityRecognizer" filename="entity_recognizer.py" path="c:\Users\xiejiang\Desktop\knowledge_graph_project\entity_recognizer.py" startline="1" type="class"></mcsymbol>
        self.relation_extractor = RelationExtractor()  # RelationExtractor内部已处理设备移动
        self.neo4j_manager = Neo4jManager()
        self.device = device
        
        # 显示设备信息
        print(f"运行设备: {device}")
        if device == 'cuda':
            print(f"GPU名称: {torch.cuda.get_device_name()}")
        else:
            print("使用CPU运行")

    def build_knowledge_graph(self):
        """构建完整的知识图谱"""
        monitor = PerformanceMonitor()
        monitor.start_timer('total_build_process')
        
        print("开始构建数学知识图谱...")
        
        # 1. 加载数据
        print("步骤1: 加载数学教材数据")
        monitor.start_timer('data_loading')
        data = self.data_loader.load_all_data()  # 这里可以正常访问data_loader
        monitor.end_timer('data_loading')
        print(f"成功加载 {len(data)} 个文件")
        
        # 2. 实体识别和关系抽取
        print("步骤2: 实体识别和关系抽取")
        all_entities = []
        all_relations = []
        
        monitor.start_timer('entity_relation_extraction')
        for item in tqdm(data, desc="处理文件"):
            try:
                text = item['content']
                grade = item.get('grade')
                chapter = item.get('chapter')
                file_name = item.get('file_name')
                
                # 文本处理不再限制长度，直接分割成句子处理
                sentences = self.data_loader.split_into_sentences(text)
                for sentence in tqdm(sentences, desc=f"处理句子", leave=False):
                    # 处理所有句子，不再限制长度
                    # 实体识别
                    entities = self.entity_recognizer.predict_entities(sentence)
                    # 关系抽取
                    relations = self.relation_extractor.extract_relations(sentence, entities)
                    
                    # 添加元数据
                    for entity in entities:
                        entity.update({
                            'grade': grade,
                            'chapter': chapter,
                            'source_file': file_name
                        })
                    
                    all_entities.extend(entities)
                    
                    # 关系抽取（只在有实体时进行）
                    if entities:
                        try:
                            relations = self.relation_extractor.extract_relations(sentence, entities)
                            all_relations.extend(relations)
                        except Exception as e:
                            print(f"  关系抽取失败: {e}")
                            continue
                    # 实体识别
                    entities = self.entity_recognizer.predict_entities(text)
                    
                    # 添加元数据
                    for entity in entities:
                        entity.update({
                            'grade': grade,
                            'chapter': chapter,
                            'source_file': file_name
                        })
                    
                    all_entities.extend(entities)
                    
                    # 关系抽取（只在有实体时进行）
                    if entities:
                        try:
                            relations = self.relation_extractor.extract_relations(text, entities)
                            all_relations.extend(relations)
                        except Exception as e:
                            print(f"  关系抽取失败: {e}")
                            continue
            except Exception as e:
                print(f"处理文件失败: {e}")
                continue
        
        monitor.end_timer('entity_relation_extraction')
        print(f"识别到 {len(all_entities)} 个实体")
        print(f"抽取到 {len(all_relations)} 个关系")
        
        # 2.5 实体后处理
        print("步骤2.5: 实体后处理")
        monitor.start_timer('entity_postprocessing')
        all_entities = self._postprocess_entities(all_entities)
        monitor.end_timer('entity_postprocessing')
        print(f"后处理后实体数: {len(all_entities)}")
        
        # 3. 数据去重
        print("步骤3: 数据去重")
        monitor.start_timer('deduplication')
        unique_entities = self._deduplicate_entities(all_entities)
        unique_relations = self._deduplicate_relations(all_relations)
        monitor.end_timer('deduplication')
        
        print(f"去重后: {len(unique_entities)} 个实体, {len(unique_relations)} 个关系")
        
        # 3.5 添加层级关系
        print("步骤3.5: 构建层级结构")
        monitor.start_timer('hierarchy_building')
        self._add_hierarchical_relations(unique_entities, unique_relations)
        monitor.end_timer('hierarchy_building')
        
        # 4. 导入Neo4j
        print("步骤4: 导入Neo4j图数据库")
        monitor.start_timer('neo4j_import')
        self.neo4j_manager.import_knowledge_graph(unique_entities, unique_relations)
        monitor.end_timer('neo4j_import')
        
        # 5. 保存中间结果
        print("步骤5: 保存处理结果")
        monitor.start_timer('saving_results')
        self._save_results(unique_entities, unique_relations)
        monitor.end_timer('saving_results')
        
        monitor.end_timer('total_build_process')
        
        # 打印性能统计
        monitor.print_stats()
        
        print("知识图谱构建完成！")
        
        return unique_entities, unique_relations
    
    def _postprocess_entities(self, entities: List[Dict]) -> List[Dict]:
        """实体后处理"""
        processed_entities = []
        
        for entity in entities:
            if len(entity['text'].strip()) < 2:
                continue
                
            if self._is_likely_misclassified(entity):
                entity['type'] = self._correct_entity_type(entity)
                
            # 添加章节唯一标识
            if entity['type'] == 'CHAPTER_TITLE':
                entity['chapter_id'] = f"CHAPTER_{entity['grade']}_{entity['chapter']}"
                
            processed_entities.append(entity)
        
        return processed_entities

    def _is_likely_misclassified(self, entity: Dict) -> bool:
        """判断实体是否可能被错误分类"""
        text = entity['text']
        entity_type = entity['type']
        
        # 增强判断逻辑
        if entity_type == "THEOREM" and ("章" in text or "节" in text or "第" in text and "章" in text):
            return True
        if entity_type == "METHOD" and (text.endswith("了") or "步骤" in text) and not text.endswith("方法"):
            return True
        return False

    def _correct_entity_type(self, entity: Dict) -> str:
        """修正实体类型（增强版）"""
        text = entity['text']
        if "章" in text or "节" in text or re.match(r'第[一二三四五六七八九十]+章', text):
            return "CHAPTER_TITLE"
        elif "方法" in text or "解法" in text:
            return "METHOD"
        elif "定理" in text or "定律" in text:
            return "THEOREM"
        elif "公式" in text or "表达式" in text:
            return "FORMULA"
        else:
            return "CONCEPT"  # 默认归类为概念

    def _add_hierarchical_relations(self, entities: List[Dict], relations: List[Dict]):
        """添加教材结构层级关系"""
        # 创建章节节点映射
        chapter_map = {
            (e['grade'], e['chapter']): e['chapter_id']
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
                        'hierarchy': 'chapter'
                    })

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
    
    def _save_results(self, entities: List[Dict], relations: List[Dict]):
        """保存处理结果"""
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        
        # 保存实体数据
        entities_path = os.path.join(Config.OUTPUT_DIR, "entities.json")
        with open(entities_path, 'w', encoding='utf-8') as f:
            json.dump(entities, f, ensure_ascii=False, indent=2)
        
        # 保存关系数据
        relations_path = os.path.join(Config.OUTPUT_DIR, "relations.json")
        with open(relations_path, 'w', encoding='utf-8') as f:
            json.dump(relations, f, ensure_ascii=False, indent=2)
        
        # 保存统计信息
        stats = {
            'total_entities': len(entities),
            'total_relations': len(relations),
            'entity_types': {},
            'relation_types': {}
        }
        
        # 统计实体类型
        for entity in entities:
            entity_type = entity['type']
            stats['entity_types'][entity_type] = stats['entity_types'].get(entity_type, 0) + 1
        
        # 统计关系类型
        for rel in relations:
            rel_type = rel['relation']
            stats['relation_types'][rel_type] = stats['relation_types'].get(rel_type, 0) + 1
        
        stats_path = os.path.join(Config.OUTPUT_DIR, "statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"结果已保存到 {Config.OUTPUT_DIR} 目录")
    
    def query_knowledge_graph(self, query_type: str, **kwargs):
        """查询知识图谱"""
        if query_type == 'entity':
            return self.neo4j_manager.query_entities(**kwargs)
        elif query_type == 'relation':
            return self.neo4j_manager.query_relations(**kwargs)
        elif query_type == 'path':
            return self.neo4j_manager.get_knowledge_path(**kwargs)
        else:
            raise ValueError("不支持的查询类型")
    
    def export_graph(self, output_path: str):
        """导出图数据"""
        self.neo4j_manager.export_graph_data(output_path)
    
    def clear_graph(self):
        """清空图数据库"""
        self.neo4j_manager.clear_database()
        print("图数据库已清空")

class LearningPathRecommender:
    def __init__(self, neo4j_manager: Neo4jManager):
        self.neo4j_manager = neo4j_manager
    
    def recommend_learning_path(self, target_concept: str, prerequisite_depth: int = 2) -> List[Dict]:
        """推荐学习路径"""
        # 获取所有前置知识
        prerequisites = self._find_prerequisites(target_concept, prerequisite_depth)
        
        # 构建学习路径
        learning_path = self._build_learning_path(prerequisites, target_concept)
        
        return learning_path
    
    def _find_prerequisites(self, concept: str, depth: int) -> List[Dict]:
        """查找前置知识"""
        query = """
        MATCH path = (start:CONCEPT {name: $concept})<-[:PREREQUISITE*1..$depth]-(prereq)
        RETURN prereq.name as name, length(path) as depth
        ORDER BY depth DESC
        """
        result = self.neo4j_manager.graph.run(query, concept=concept, depth=depth)
        return [dict(record) for record in result]
    
    def _build_learning_path(self, prerequisites: List[Dict], target: str) -> List[Dict]:
        """构建学习路径"""
        # 按深度排序
        prerequisites.sort(key=lambda x: x['depth'])
        
        learning_path = []
        visited = set()
        
        # 添加前置知识
        for prereq in prerequisites:
            if prereq['name'] not in visited:
                learning_path.append({
                    'concept': prereq['name'],
                    'type': 'prerequisite',
                    'order': len(learning_path) + 1
                })
                visited.add(prereq['name'])
        
        # 添加目标概念
        learning_path.append({
            'concept': target,
            'type': 'target',
            'order': len(learning_path) + 1
        })
        
        return learning_path

# 使用示例
if __name__ == "__main__":
    builder = KnowledgeGraphBuilder()
    
    try:
        # 构建知识图谱
        entities, relations = builder.build_knowledge_graph()
        
        # 示例查询
        print("\n示例查询:")
        concepts = builder.query_knowledge_graph('entity', entity_type='CONCEPT')
        print(f"找到 {len(concepts)} 个数学概念")
        
        # 学习路径推荐示例
        recommender = LearningPathRecommender(builder.neo4j_manager)
        path = recommender.recommend_learning_path("二次函数")
        print(f"学习路径推荐: {[item['concept'] for item in path]}")
        
    except Exception as e:
        print(f"构建过程中出错: {e}")
        import traceback
        traceback.print_exc()
