"""Neo4j数据导入工具
用于将处理好的知识图谱数据导入到Neo4j数据库中
"""

import json
import os
from typing import List, Dict
from neo4j_manager import Neo4jManager
from config import Config

class Neo4jDataImporter:
    def __init__(self):
        self.neo4j_manager = Neo4jManager()
    
    def load_grade7_data(self) -> tuple[List[Dict], List[Dict]]:
        """加载七年级上册处理结果"""
        grade7_dir = os.path.join(Config.OUTPUT_DIR, "grade7")
        
        # 加载实体数据
        entities_path = os.path.join(grade7_dir, "entities.json")
        if not os.path.exists(entities_path):
            raise FileNotFoundError(f"实体文件不存在: {entities_path}")
        
        with open(entities_path, 'r', encoding='utf-8') as f:
            entities = json.load(f)
        
        # 加载关系数据
        relations_path = os.path.join(grade7_dir, "relations.json")
        if not os.path.exists(relations_path):
            raise FileNotFoundError(f"关系文件不存在: {relations_path}")
        
        with open(relations_path, 'r', encoding='utf-8') as f:
            relations = json.load(f)
        
        print(f"加载成功: {len(entities)} 个实体, {len(relations)} 个关系")
        return entities, relations
    
    def import_grade7_to_neo4j(self):
        """将七年级上册数据导入Neo4j"""
        if not self.neo4j_manager.connected:
            print("Neo4j连接失败，无法导入数据")
            return False
        
        try:
            print("开始加载七年级上册数据...")
            entities, relations = self.load_grade7_data()
            
            print("开始导入Neo4j数据库...")
            self.neo4j_manager.import_knowledge_graph(entities, relations)
            
            print("七年级上册数据导入完成！")
            return True
            
        except Exception as e:
            print(f"导入过程中出错: {e}")
            return False
    
    def clear_neo4j_database(self):
        """清空Neo4j数据库"""
        if not self.neo4j_manager.connected:
            print("Neo4j连接失败，无法清空数据库")
            return False
        
        try:
            print("正在清空Neo4j数据库...")
            self.neo4j_manager.clear_database()
            print("数据库已清空")
            return True
        except Exception as e:
            print(f"清空数据库时出错: {e}")
            return False
    
    def export_neo4j_data(self, output_path: str):
        """导出Neo4j中的数据"""
        if not self.neo4j_manager.connected:
            print("Neo4j连接失败，无法导出数据")
            return False
        
        try:
            print(f"正在导出数据到: {output_path}")
            self.neo4j_manager.export_graph_data(output_path)
            print("数据导出完成")
            return True
        except Exception as e:
            print(f"导出数据时出错: {e}")
            return False
    
    def query_statistics(self):
        """查询数据库统计信息"""
        if not self.neo4j_manager.connected:
            print("Neo4j连接失败，无法查询统计信息")
            return
        
        try:
            # 查询节点数量
            node_query = "MATCH (n) RETURN count(n) as node_count"
            node_result = self.neo4j_manager.graph.run(node_query)
            node_count = [record['node_count'] for record in node_result][0]
            
            # 查询关系数量
            rel_query = "MATCH ()-[r]->() RETURN count(r) as rel_count"
            rel_result = self.neo4j_manager.graph.run(rel_query)
            rel_count = [record['rel_count'] for record in rel_result][0]
            
            # 查询实体类型分布
            entity_type_query = """
            MATCH (n) 
            RETURN labels(n)[0] as entity_type, count(n) as count 
            ORDER BY count DESC
            """
            entity_type_result = self.neo4j_manager.graph.run(entity_type_query)
            
            print("\n=== Neo4j数据库统计 ===")
            print(f"节点总数: {node_count}")
            print(f"关系总数: {rel_count}")
            print("\n实体类型分布:")
            for record in entity_type_result:
                print(f"  {record['entity_type']}: {record['count']} 个")
            
            # 查询关系类型分布
            rel_type_query = """
            MATCH ()-[r]->() 
            RETURN type(r) as rel_type, count(r) as count 
            ORDER BY count DESC
            """
            rel_type_result = self.neo4j_manager.graph.run(rel_type_query)
            
            print("\n关系类型分布:")
            for record in rel_type_result:
                print(f"  {record['rel_type']}: {record['count']} 个")
                
        except Exception as e:
            print(f"查询统计信息时出错: {e}")

# 使用示例
if __name__ == "__main__":
    importer = Neo4jDataImporter()
    
    print("Neo4j数据导入工具")
    print("==================")
    
    if not importer.neo4j_manager.connected:
        print("警告: Neo4j连接失败，请确保Neo4j服务正在运行")
        print("Neo4j配置:")
        print(f"  URI: {Config.NEO4J_URI}")
        print(f"  用户: {Config.NEO4J_USER}")
        print("请检查Neo4j服务状态和连接配置")
        exit(1)
    
    print("Neo4j连接成功")
    
    # 显示菜单
    while True:
        print("\n请选择操作:")
        print("1. 导入七年级上册数据到Neo4j")
        print("2. 清空Neo4j数据库")
        print("3. 导出Neo4j数据到文件")
        print("4. 查询数据库统计信息")
        print("5. 退出")
        
        choice = input("请输入选项 (1-5): ").strip()
        
        if choice == "1":
            success = importer.import_grade7_to_neo4j()
            if success:
                importer.query_statistics()
                
        elif choice == "2":
            confirm = input("确定要清空整个数据库吗？此操作不可逆！(y/N): ").strip().lower()
            if confirm == 'y':
                importer.clear_neo4j_database()
                
        elif choice == "3":
            output_path = input("请输入导出文件路径 (默认: ./exported_graph.json): ").strip()
            if not output_path:
                output_path = "./exported_graph.json"
            importer.export_neo4j_data(output_path)
            
        elif choice == "4":
            importer.query_statistics()
            
        elif choice == "5":
            print("退出程序")
            break
            
        else:
            print("无效选项，请重新选择")