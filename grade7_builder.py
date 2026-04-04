"""七年级上册知识图谱构建器
专门处理七年级上册数学教材内容的知识图谱构建
"""

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

class Grade7KnowledgeGraphBuilder:
    def __init__(self):
        self.data_loader = MathDataLoader()
        self.entity_recognizer = EntityRecognizer()
        self.relation_extractor = RelationExtractor()
        self.neo4j_manager = Neo4jManager()
        
    def get_grade7_files(self) -> List[str]:
        """获取七年级上册文件列表"""
        if not os.path.exists(Config.DATA_DIR):
            raise FileNotFoundError(f"数据目录不存在: {Config.DATA_DIR}")
            
        grade7_files = []
        for file_name in os.listdir(Config.DATA_DIR):
            # 匹配7.1, 7.2等七年级上册文件
            if re.match(r'^7\.\d+', file_name):
                file_path = os.path.join(Config.DATA_DIR, file_name)
                grade7_files.append(file_path)
        
        print(f"找到 {len(grade7_files)} 个七年级上册文件")
        return sorted(grade7_files)
    
    def load_grade7_data(self) -> List[Dict]:
        """加载七年级上册数据"""
        files = self.get_grade7_files()
        all_data = []
        
        for file_path in files:
            content = self.data_loader.read_file_content(file_path)
            metadata = self.data_loader.extract_metadata(file_path)
            
            data_item = {
                'content': content,
                **metadata
            }
            all_data.append(data_item)
            
            print(f"已加载七年级上册: {metadata.get('file_name', '未知文件')}")
        
        return all_data
    
    @time_it
    def build_grade7_knowledge_graph(self):
        """构建七年级上册知识图谱"""
        monitor = PerformanceMonitor()
        monitor.start_timer('grade7_build_process')
        
        print("开始构建七年级上册数学知识图谱...")
        
        # 1. 加载七年级上册数据
        print("步骤1: 加载七年级上册数学教材数据")
        monitor.start_timer('grade7_data_loading')
        data = self.load_grade7_data()
        monitor.end_timer('grade7_data_loading')
        print(f"成功加载 {len(data)} 个七年级上册文件")
        
        # 2. 实体识别和关系抽取
        print("步骤2: 实体识别和关系抽取")
        all_entities = []
        all_relations = []
        
        monitor.start_timer('grade7_entity_relation_extraction')
        for item in tqdm(data, desc="处理七年级上册文件"):
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
            except Exception as e:
                print(f"处理文件失败: {e}")
                continue
        
        monitor.end_timer('grade7_entity_relation_extraction')
        print(f"识别到 {len(all_entities)} 个实体")
        print(f"抽取到 {len(all_relations)} 个关系")
        
        # 3. 数据去重
        print("步骤3: 数据去重")
        monitor.start_timer('grade7_deduplication')
        unique_entities = self._deduplicate_entities(all_entities)
        unique_relations = self._deduplicate_relations(all_relations)
        monitor.end_timer('grade7_deduplication')
        
        print(f"去重后: {len(unique_entities)} 个实体, {len(unique_relations)} 个关系")
        
        # 4. 导入Neo4j（如果连接成功）
        print("步骤4: 导入Neo4j图数据库")
        monitor.start_timer('grade7_neo4j_import')
        try:
            self.neo4j_manager.import_knowledge_graph(unique_entities, unique_relations)
        except Exception as e:
            print(f"Neo4j导入失败: {e}")
            print("继续保存文件结果...")
        monitor.end_timer('grade7_neo4j_import')
        
        # 5. 保存中间结果
        print("步骤5: 保存处理结果")
        monitor.start_timer('grade7_saving_results')
        self._save_grade7_results(unique_entities, unique_relations)
        monitor.end_timer('grade7_saving_results')
        
        monitor.end_timer('grade7_build_process')
        
        # 打印性能统计
        monitor.print_stats()
        
        print("七年级上册知识图谱构建完成！")
        
        return unique_entities, unique_relations
    
    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """实体去重"""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            key = (entity['text'], entity['type'])
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
    
    def _save_grade7_results(self, entities: List[Dict], relations: List[Dict]):
        """保存七年级上册处理结果"""
        # 创建七年级专用输出目录
        grade7_output_dir = os.path.join(Config.OUTPUT_DIR, "grade7")
        os.makedirs(grade7_output_dir, exist_ok=True)
        
        # 保存实体数据
        entities_path = os.path.join(grade7_output_dir, "entities.json")
        with open(entities_path, 'w', encoding='utf-8') as f:
            json.dump(entities, f, ensure_ascii=False, indent=2)
        
        # 保存关系数据
        relations_path = os.path.join(grade7_output_dir, "relations.json")
        with open(relations_path, 'w', encoding='utf-8') as f:
            json.dump(relations, f, ensure_ascii=False, indent=2)
        
        # 保存统计信息
        stats = {
            'total_entities': len(entities),
            'total_relations': len(relations),
            'entity_types': {},
            'relation_types': {},
            'grade': 7,
            'semester': '上册'
        }
        
        # 统计实体类型
        for entity in entities:
            entity_type = entity['type']
            stats['entity_types'][entity_type] = stats['entity_types'].get(entity_type, 0) + 1
        
        # 统计关系类型
        for rel in relations:
            rel_type = rel['relation']
            stats['relation_types'][rel_type] = stats['relation_types'].get(rel_type, 0) + 1
        
        stats_path = os.path.join(grade7_output_dir, "statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"七年级上册结果已保存到 {grade7_output_dir} 目录")
    
    def query_grade7_entities(self):
        """查询七年级上册实体"""
        if not self.neo4j_manager.connected:
            print("离线模式: 无法查询实体")
            return []
        
        # 简单的查询所有实体
        return self.neo4j_manager.query_entities()
    
    def clear_database(self):
        """清空数据库（谨慎使用）"""
        self.neo4j_manager.clear_database()

# 使用示例
if __name__ == "__main__":
    builder = Grade7KnowledgeGraphBuilder()
    
    try:
        # 构建七年级上册知识图谱
        print("开始构建七年级上册知识图谱...")
        entities, relations = builder.build_grade7_knowledge_graph()
        
        print(f"\n七年级上册处理完成!")
        print(f"识别到 {len(entities)} 个实体")
        print(f"抽取到 {len(relations)} 个关系")
        
        # 显示实体类型统计
        entity_types = {}
        for entity in entities:
            entity_type = entity['type']
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        print("\n实体类型统计:")
        for entity_type, count in entity_types.items():
            print(f"  {entity_type}: {count} 个")
        
        # 示例：显示前几个实体
        if entities:
            print(f"\n前5个实体示例:")
            for entity in entities[:5]:
                print(f"  - {entity['text']} ({entity['type']})")
        
    except Exception as e:
        print(f"构建过程中出错: {e}")
        import traceback
        traceback.print_exc()