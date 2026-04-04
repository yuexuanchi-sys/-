import re
import json
from typing import List, Dict, Set, Optional, Union
from collections import defaultdict
from kggen_client import KGGenClient  # 暂时保持原客户端名称
from config import Config
import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class KnowledgeGraph:
    """知识图谱数据类"""
    entities: Set[str]
    edges: Set[str]
    relations: Set[tuple]  # (subject, relation, object)
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "entities": list(self.entities),
            "edges": list(self.edges),
            "relations": [list(rel) for rel in self.relations]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'KnowledgeGraph':
        """从字典创建KnowledgeGraph"""
        return cls(
            entities=set(data.get("entities", [])),
            edges=set(data.get("edges", [])),
            relations={tuple(rel) for rel in data.get("relations", [])}
        )

class ARKKnowledgeGraph:
    """ARKKnowledgeGraph类，基于Volcengine ARK API的知识图谱生成、聚合和聚类功能"""
    
    def __init__(self, model: str = None, temperature: float = 0.0, api_key: str = None):
        """
        初始化ARKKnowledgeGraph
        
        Args:
            model: 模型名称，默认为config中的设置
            temperature: 温度参数
            api_key: API密钥，默认为config中的设置
        """
        self.model = model or Config.KGGen_MODEL
        self.temperature = temperature
        self.api_key = api_key or Config.KGGen_API_KEY
        
        # 初始化API客户端
        self.client = KGGenClient()  # 暂时使用原客户端
        logger.info(f"ARKKnowledgeGraph初始化完成，模型: {self.model}, 温度: {self.temperature}")
    
    def generate(self, input_data: Union[str, List[Dict]], context: str = None, 
                 chunk_size: int = None, cluster: bool = False) -> KnowledgeGraph:
        """
        从输入数据生成知识图谱
        
        Args:
            input_data: 输入文本或消息数组
            context: 上下文信息
            chunk_size: 分块大小（字符数），用于处理长文本
            cluster: 是否进行聚类
            
        Returns:
            KnowledgeGraph: 生成的知识图谱
        """
        if isinstance(input_data, list):
            # 处理消息数组
            text = self._messages_to_text(input_data)
        else:
            text = input_data
        
        # 长文本处理：如果指定了分块大小且文本长度超过分块大小，则进行分块处理
        if chunk_size and len(text) > chunk_size:
            return self._process_long_text(text, context, chunk_size, cluster)
        else:
            # 单块处理
            graph = self._process_single_chunk(text, context)
            if cluster:
                graph = self.cluster(graph, context)
            return graph
    
    def _process_long_text(self, text: str, context: str, chunk_size: int, cluster: bool) -> KnowledgeGraph:
        """处理长文本，通过分块方式"""
        logger.info(f"处理长文本，长度: {len(text)} 字符，分块大小: {chunk_size}")
        
        chunks = self._chunk_text(text, chunk_size)
        graphs = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"处理第 {i+1}/{len(chunks)} 块，长度: {len(chunk)} 字符")
            graph = self._process_single_chunk(chunk, context)
            graphs.append(graph)
        
        # 聚合所有块的结果
        combined_graph = self.aggregate(graphs)
        
        if cluster:
            combined_graph = self.cluster(combined_graph, context)
        
        return combined_graph
    
    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """将文本分块，尽量在句子边界分割"""
        chunks = []
        start = 0
        
        while start < len(text):
            # 计算当前块的结束位置
            end = start + chunk_size
            
            # 如果还有剩余文本，尝试在句子边界分割
            if end < len(text):
                # 查找最近的句子结束位置（句号、问号、感叹号等）
                sentence_end = self._find_sentence_boundary(text, end)
                if sentence_end > start and sentence_end < len(text):
                    end = sentence_end + 1  # 包括结束标点
            
            chunk = text[start:end].strip()
            if chunk:  # 确保块不为空
                chunks.append(chunk)
            
            start = end  # 移动到下一个块
        
        return chunks
    
    def _find_sentence_boundary(self, text: str, position: int) -> int:
        """查找最近的句子边界"""
        # 从指定位置向前查找句子结束标点
        for i in range(position, max(0, position - 100), -1):
            if i < len(text) and text[i] in '.!?。！？\n':
                return i
        
        # 如果没找到，从指定位置向后查找
        for i in range(position, min(len(text), position + 100)):
            if i < len(text) and text[i] in '.!?。！？\n':
                return i
        
        return position  # 没找到边界，返回原位置
    
    def _messages_to_text(self, messages: List[Dict]) -> str:
        """将消息数组转换为文本"""
        text = ""
        for msg in messages:
            if msg.get("role") == "user" or msg.get("role") == "assistant":
                text += msg.get("content", "") + "\n"
        return text.strip()
    
    def _process_single_chunk(self, text: str, context: str = None) -> KnowledgeGraph:
        """处理单个文本块"""
        try:
            # 使用API客户端提取实体和关系
            result = self.client.extract_entities_and_relations(text)
            
            # 转换为KnowledgeGraph格式
            entities = set()
            edges = set()
            relations = set()
            
            for entity in result.get("entities", []):
                entities.add(entity["text"])
            
            for relation in result.get("relations", []):
                subject = relation["subject"]
                rel_type = relation["relation"]
                obj = relation["object"]
                
                entities.add(subject)
                entities.add(obj)
                edges.add(rel_type)
                relations.add((subject, rel_type, obj))
            
            return KnowledgeGraph(entities, edges, relations)
            
        except Exception as e:
            logger.error(f"处理文本块失败: {e}")
            return KnowledgeGraph(set(), set(), set())
    
    def aggregate(self, graphs: List[KnowledgeGraph]) -> KnowledgeGraph:
        """聚合多个知识图谱"""
        all_entities = set()
        all_edges = set()
        all_relations = set()
        
        for graph in graphs:
            all_entities |= graph.entities
            all_edges |= graph.edges
            all_relations |= graph.relations
        
        return KnowledgeGraph(all_entities, all_edges, all_relations)
    
    def cluster(self, graph: KnowledgeGraph, context: str = None) -> KnowledgeGraph:
        """
        对知识图谱进行聚类，合并相似的实体和关系
        """
        if not graph.entities:
            return graph
        
        # 实体聚类（基于文本相似度）
        entity_clusters = self._cluster_entities(list(graph.entities))
        
        # 创建实体映射：原始实体 -> 聚类代表实体
        entity_mapping = {}
        for cluster in entity_clusters:
            representative = self._choose_representative_entity(cluster, context)
            for entity in cluster:
                entity_mapping[entity] = representative
        
        # 更新关系中的实体
        new_relations = set()
        for subject, relation, obj in graph.relations:
            new_subject = entity_mapping.get(subject, subject)
            new_obj = entity_mapping.get(obj, obj)
            # 只有当事物不同时才保留关系
            if new_subject != new_obj:
                new_relations.add((new_subject, relation, new_obj))
        
        # 更新实体集合
        new_entities = set(entity_mapping.values()) if entity_mapping else graph.entities
        
        return KnowledgeGraph(new_entities, graph.edges, new_relations)
    
    def _cluster_entities(self, entities: List[str]) -> List[List[str]]:
        """基于文本相似度对实体进行聚类"""
        if len(entities) <= 1:
            return [entities]
        
        try:
            # 使用TF-IDF向量化
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(entities)
            
            # 计算相似度矩阵
            similarity_matrix = cosine_similarity(X)
            
            # 使用DBSCAN聚类
            dbscan = DBSCAN(eps=0.5, min_samples=1, metric='precomputed')
            # 将相似度转换为距离
            distance_matrix = 1 - similarity_matrix
            clusters = dbscan.fit_predict(distance_matrix)
            
            # 分组实体
            entity_clusters = defaultdict(list)
            for i, cluster_id in enumerate(clusters):
                entity_clusters[cluster_id].append(entities[i])
            
            return list(entity_clusters.values())
            
        except Exception as e:
            logger.warning(f"实体聚类失败，使用简单方法: {e}")
            # 回退到基于字符串相似度的简单聚类
            return self._simple_entity_clustering(entities)
    
    def _simple_entity_clustering(self, entities: List[str]) -> List[List[str]]:
        """基于字符串相似度的简单实体聚类"""
        clusters = []
        used_entities = set()
        
        for i, entity1 in enumerate(entities):
            if entity1 in used_entities:
                continue
                
            cluster = [entity1]
            used_entities.add(entity1)
            
            for j, entity2 in enumerate(entities[i+1:], start=i+1):
                if entity2 in used_entities:
                    continue
                    
                # 计算字符串相似度
                similarity = SequenceMatcher(None, entity1, entity2).ratio()
                if similarity > 0.7:  # 相似度阈值
                    cluster.append(entity2)
                    used_entities.add(entity2)
            
            clusters.append(cluster)
        
        return clusters
    
    def _choose_representative_entity(self, cluster: List[str], context: str = None) -> str:
        """从实体簇中选择代表实体"""
        if not cluster:
            return ""
        
        # 简单策略：选择最长的实体，或者根据上下文选择
        if context and any(keyword in context for keyword in ["家庭", "人物", "人名"]):
            # 对于家庭关系，优先选择全名或更正式的称呼
            for entity in cluster:
                if len(entity) > 3 and " " in entity:  # 假设全名有空格
                    return entity
        elif context and any(keyword in context for keyword in ["技术", "科学", "术语"]):
            # 对于技术术语，选择最具体的
            for entity in cluster:
                if len(entity) > 5:  # 假设较长的术语更具体
                    return entity
        
        # 默认选择最长的实体
        return max(cluster, key=len)

# 使用示例
if __name__ == "__main__":
    # 初始化ARKKnowledgeGraph
    ark_graph = ARKKnowledgeGraph(
        model=Config.KGGen_MODEL,
        temperature=0.0,
        api_key=Config.KGGen_API_KEY
    )
    
    # 示例1：单个字符串
    text_input = "Linda 是 Josh 的母亲。Ben 是 Josh 的兄弟。Andrew 是 Josh 的父亲。"
    graph_1 = ark_graph.generate(
        input_data=text_input,
        context="家庭关系"
    )
    print("示例1结果:")
    print(f"entities={graph_1.entities}")
    print(f"edges={graph_1.edges}")
    print(f"relations={graph_1.relations}")
    
    # 示例2：长文本处理
    long_text = "数学是研究数量、结构、变化、空间以及信息等概念的学科。代数主要研究数字和符号的运算规则。几何研究形状、大小、相对位置等空间性质。微积分研究变化和累积的概念。"
    graph_2 = ark_graph.generate(
        input_data=long_text,
        context="数学概念",
        chunk_size=50,  # 小分块用于演示
        cluster=True
    )
    print("\n示例2结果:")
    print(f"entities={graph_2.entities}")
    print(f"edges={graph_2.edges}")
    print(f"relations={graph_2.relations}")