try:
    from py2neo import Graph, Node, Relationship
    _py2neo_available = True
except ImportError:
    _py2neo_available = False
import pandas as pd
from typing import List, Dict
from config import Config

class Neo4jManager:
    def __init__(self, max_retries=3, retry_delay=2):
        self.connected = False
        self.graph = None
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        if _py2neo_available:
            self._connect_with_retry()
        else:
            print("[WARN] py2neo 未安装，使用离线模式")

    def _connect_with_retry(self):
        """带重试机制的连接方法"""
        for attempt in range(self.max_retries):
            try:
                print(f"尝试连接Neo4j数据库 (尝试 {attempt + 1}/{self.max_retries})...")
                self.graph = Graph(
                    Config.NEO4J_URI,
                    auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD)
                )
                # 测试连接
                result = self.graph.run("RETURN 1 as test")
                test_value = result.data()[0]['test']
                
                self._create_constraints()
                self.connected = True
                print("[OK] Neo4j连接成功")
                return
                
            except Exception as e:
                print(f"[WARN] 连接失败 (尝试 {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    print(f"等待 {self.retry_delay} 秒后重试...")
                    import time
                    time.sleep(self.retry_delay)
                else:
                    print("[WARN] 所有重试尝试均失败，将使用离线模式")
                    print("[WARN] 数据不会保存到数据库，但程序可以继续运行")
                    print("请检查:")
                    print("1. Neo4j Desktop是否运行且数据库已启动")
                    print("2. 连接信息是否正确 (URI: neo4j://localhost:7687)")
                    print("3. 用户名密码是否正确")
                    print("4. 防火墙是否阻止了7687端口")
    
    def _create_constraints(self):
        """创建唯一性约束"""
        try:
            # 为每种实体类型创建唯一性约束
            for entity_type in Config.ENTITY_TYPES.keys():
                self.graph.run(
                    f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{entity_type}) REQUIRE n.name IS UNIQUE"
                )
            print("Neo4j约束创建成功")
        except Exception as e:
            print(f"创建约束时出错: {e}")
    
    def create_entity_node(self, entity: Dict) -> Node:
        """创建实体节点"""
        if not self.connected:
            # 离线模式下返回一个模拟的节点对象
            entity_type = entity['type']
            properties = {
                'name': entity['text'],
                'source': entity.get('source', 'unknown'),
                'grade': entity.get('grade'),
                'chapter': entity.get('chapter')
            }
            properties = {k: v for k, v in properties.items() if v is not None}
            
            # 创建一个简单的对象模拟Node
            class MockNode:
                def __init__(self, labels, **kwargs):
                    self.labels = labels
                    self.properties = kwargs
                
                def __getitem__(self, key):
                    return self.properties.get(key)
            
            return MockNode([entity_type], **properties)
        
        entity_type = entity['type']
        properties = {
            'name': entity['text'],
            'source': entity.get('source', 'unknown'),
            'grade': entity.get('grade'),
            'chapter': entity.get('chapter')
        }
        
        # 清理None值
        properties = {k: v for k, v in properties.items() if v is not None}
        
        node = Node(entity_type, **properties)
        self.graph.merge(node, entity_type, "name")
        return node
    
    def create_relation(self, relation: Dict, subject_node: Node, object_node: Node) -> Relationship:
        """创建关系边"""
        if not self.connected:
            # 离线模式下返回一个模拟的关系对象
            class MockRelationship:
                def __init__(self, start, rel_type, end, **kwargs):
                    self.start_node = start
                    self.end_node = end
                    self.type = rel_type
                    self.properties = kwargs
                
                def __getitem__(self, key):
                    return self.properties.get(key)
            
            rel_type = relation['relation']
            properties = {
                'source': relation.get('source', 'unknown'),
                'confidence': relation.get('confidence', 1.0)
            }
            return MockRelationship(subject_node, rel_type, object_node, **properties)
        
        rel_type = relation['relation']
        properties = {
            'source': relation.get('source', 'unknown'),
            'confidence': relation.get('confidence', 1.0)
        }
        
        relationship = Relationship(subject_node, rel_type, object_node, **properties)
        self.graph.merge(relationship)
        return relationship
    
    def import_knowledge_graph(self, entities: List[Dict], relations: List[Dict]):
        """导入知识图谱数据"""
        if not self.connected:
            print("离线模式: 知识图谱数据不会保存到数据库")
            print("识别到的实体:")
            for entity in entities:
                print(f"  {entity['type']}: {entity['text']} (来源: {entity.get('source', 'unknown')})")
            
            print("\n识别到的关系:")
            for relation in relations:
                print(f"  {relation['subject']} --{relation['relation']}--> {relation['object']}")
            
            print(f"\n总计: {len(entities)} 个实体, {len(relations)} 个关系")
            return
        
        print("开始导入知识图谱数据...")
        
        # 创建所有实体节点
        entity_nodes = {}
        for entity in entities:
            node = self.create_entity_node(entity)
            entity_nodes[entity['text']] = node
            print(f"创建实体: {entity['text']} ({entity['type']})")
        
        # 创建所有关系
        for relation in relations:
            subject = relation['subject']
            object_ = relation['object']
            
            if subject in entity_nodes and object_ in entity_nodes:
                self.create_relation(relation, entity_nodes[subject], entity_nodes[object_])
                print(f"创建关系: {subject} --{relation['relation']}--> {object_}")
        
        print("知识图谱导入完成！")
    
    def query_entities(self, entity_type: str = None, name: str = None) -> List[Node]:
        """查询实体"""
        if not self.connected:
            print("离线模式: 无法查询实体，返回空列表")
            return []
            
        if entity_type and name:
            query = f"MATCH (n:{entity_type} {{name: $name}}) RETURN n"
            result = self.graph.run(query, name=name)
        elif entity_type:
            query = f"MATCH (n:{entity_type}) RETURN n LIMIT 100"
            result = self.graph.run(query)
        elif name:
            query = "MATCH (n {name: $name}) RETURN n"
            result = self.graph.run(query, name=name)
        else:
            query = "MATCH (n) RETURN n LIMIT 100"
            result = self.graph.run(query)
        
        return [record['n'] for record in result]
    
    def query_relations(self, subject: str = None, relation_type: str = None, object_: str = None) -> List[Dict]:
        """查询关系"""
        if not self.connected:
            print("离线模式: 无法查询关系，返回空列表")
            return []
            
        if subject and relation_type and object_:
            query = f"""
            MATCH (s {{name: $subject}})-[r:{relation_type}]->(o {{name: $object}})
            RETURN s, r, o
            """
            result = self.graph.run(query, subject=subject, object=object_)
        elif subject and relation_type:
            query = f"""
            MATCH (s {{name: $subject}})-[r:{relation_type}]->(o)
            RETURN s, r, o
            """
            result = self.graph.run(query, subject=subject)
        elif subject:
            query = """
            MATCH (s {name: $subject})-[r]->(o)
            RETURN s, r, o
            """
            result = self.graph.run(query, subject=subject)
        else:
            query = "MATCH (s)-[r]->(o) RETURN s, r, o LIMIT 100"
            result = self.graph.run(query)
        
        return [{
            'subject': record['s'],
            'relation': record['r'],
            'object': record['o']
        } for record in result]
    
    def get_knowledge_path(self, start_entity: str, end_entity: str, max_depth: int = 3) -> List[Dict]:
        """获取知识路径"""
        if not self.connected:
            print("离线模式: 无法获取知识路径，返回空列表")
            return []
            
        query = """
        MATCH path = shortestPath((s {name: $start})-[*1..$max_depth]-(e {name: $end}))
        RETURN path
        """
        result = self.graph.run(query, start=start_entity, end=end_entity, max_depth=max_depth)
        
        paths = []
        for record in result:
            path = record['path']
            nodes = path.nodes
            relationships = path.relationships
            
            path_info = {
                'nodes': [{'name': node['name'], 'type': list(node.labels)[0]} for node in nodes],
                'relationships': [{
                    'type': rel.type,
                    'start': rel.start_node['name'],
                    'end': rel.end_node['name']
                } for rel in relationships]
            }
            paths.append(path_info)
        
        return paths
    
    def clear_database(self):
        """清空数据库"""
        if not self.connected:
            print("离线模式: 无法清空数据库")
            return
            
        self.graph.run("MATCH (n) DETACH DELETE n")
        print("数据库已清空")
    
    def export_graph_data(self, output_path: str):
        """导出图数据"""
        if not self.connected:
            print("离线模式: 无法导出图数据")
            return
            
        # 导出所有节点
        nodes_query = "MATCH (n) RETURN n.name as name, labels(n)[0] as type, properties(n) as properties"
        nodes_result = self.graph.run(nodes_query)
        nodes_data = [dict(record) for record in nodes_result]
        
        # 导出所有关系
        rels_query = """
        MATCH (s)-[r]->(o)
        RETURN s.name as subject, type(r) as relation, o.name as object, properties(r) as properties
        """
        rels_result = self.graph.run(rels_query)
        rels_data = [dict(record) for record in rels_result]
        
        # 保存到文件
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'nodes': nodes_data,
                'relationships': rels_data
            }, f, ensure_ascii=False, indent=2)
        
        print(f"图数据已导出到: {output_path}")

# 使用示例
if __name__ == "__main__":
    try:
        manager = Neo4jManager()
        print("Neo4j连接成功")
        
        # 测试查询
        entities = manager.query_entities("CONCEPT")
        print(f"找到 {len(entities)} 个概念实体")
        
    except Exception as e:
        print(f"Neo4j连接失败: {e}")
        print("请确保Neo4j服务正在运行，并检查配置中的连接信息")