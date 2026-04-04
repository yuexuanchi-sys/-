from flask import Flask, request, jsonify, render_template
from neo4j import GraphDatabase
import json

app = Flask(__name__)

# Neo4j配置
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "BMtanwang7546"

class SimpleMathKnowledgeGraph:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    def close(self):
        self.driver.close()
    
    def get_root_nodes(self):
        """获取根节点"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n:根分类)
                RETURN n.name as name, n.id as id, n.grade as grade
                LIMIT 10
            """)
            return [dict(record) for record in result]
    
    def get_chapters(self):
        """获取所有章节"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n:一级分类)
                RETURN n.chapter as chapter, n.name as name, n.id as id
                ORDER BY n.chapter
                LIMIT 20
            """)
            return [dict(record) for record in result]

    def get_sections(self):
        """获取所有二级分类（小节）"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n:二级分类)
                RETURN n.name as name, n.id as id, n.section as section
                ORDER BY n.section
                LIMIT 200
            """)
            return [dict(record) for record in result]

    def get_knowledges(self):
        """获取所有知识点"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n:knowledges7_entities)
                RETURN n.name as name, n.id as id
                LIMIT 300
            """)
            return [dict(record) for record in result]
    
    def find_knowledge_branch(self, node_name, max_depth=3):
        """查找知识节点的完整分支，只包含指定类型的节点"""
        with self.driver.session() as session:
            # 首先获取起始节点（只从指定类型中查找）
            start_result = session.run("""
                MATCH (start)
                WHERE (start:根分类 OR start:一级分类 OR start:二级分类 OR start:knowledges7_entities)
                AND start.name = $node_name
                RETURN start
                LIMIT 1
            """, node_name=node_name)
            
            nodes = []
            relationships = []
            node_map = {}
            
            # 处理起始节点
            start_record = start_result.single()
            if not start_record or not start_record['start']:
                return {"nodes": [], "relationships": []}
            
            start_node = start_record['start']
            start_id = str(start_node.element_id)
            node_map[start_id] = {
                "id": start_id,
                "label": start_node.get('name', 'Unknown'),
                "title": self._get_node_title(start_node),
                "color": self._get_node_color(start_node.labels),
                "level": self._get_node_level(start_node.labels),
                "properties": self._convert_properties(start_node)
            }
            
            # 然后获取所有相关的子节点和关系（只包含指定类型的节点）
            result = session.run("""
                MATCH (start)
                WHERE (start:根分类 OR start:一级分类 OR start:二级分类 OR start:knowledges7_entities)
                AND start.name = $node_name
                OPTIONAL MATCH (start)-[r:包含]->(child)
                WHERE (child:根分类 OR child:一级分类 OR child:二级分类 OR child:knowledges7_entities)
                RETURN start, child, r
                LIMIT 300
            """, node_name=node_name)
            
            relationship_set = set()
            
            for record in result:
                # 处理起始节点（确保在映射中）
                start_node = record['start']
                if start_node:
                    start_id = str(start_node.element_id)
                    if start_id not in node_map:
                        node_map[start_id] = {
                            "id": start_id,
                            "label": start_node.get('name', 'Unknown'),
                            "title": self._get_node_title(start_node),
                            "color": self._get_node_color(start_node.labels),
                            "level": self._get_node_level(start_node.labels),
                            "properties": self._convert_properties(start_node)
                        }
                
                # 处理子节点
                child_node = record['child']
                if child_node:
                    child_id = str(child_node.element_id)
                    if child_id not in node_map:
                        node_map[child_id] = {
                            "id": child_id,
                            "label": child_node.get('name', 'Unknown'),
                            "title": self._get_node_title(child_node),
                            "color": self._get_node_color(child_node.labels),
                            "level": self._get_node_level(child_node.labels),
                            "properties": self._convert_properties(child_node)
                        }
                
                # 处理关系
                rel = record['r']
                if rel:
                    rel_id = str(rel.element_id)
                    if rel_id not in relationship_set:
                        relationship_set.add(rel_id)
                        relationships.append({
                            "id": rel_id,
                            "source": str(rel.start_node.element_id),
                            "target": str(rel.end_node.element_id),
                            "label": rel.type,
                            "title": rel.type,
                            "color": self._get_relationship_color(rel.type)
                        })
            
            # 将节点映射转换为列表
            nodes = list(node_map.values())
            
            return {"nodes": nodes, "relationships": relationships}

    def get_full_graph(self):
        """获取完整的知识图谱，包含所有层级的节点和关系"""
        with self.driver.session() as session:
            # 首先获取所有节点，确保每个节点只出现一次
            nodes_result = session.run("""
                MATCH (n)
                WHERE n:根分类 OR n:一级分类 OR n:二级分类 OR n:knowledges7_entities
                RETURN n
                LIMIT 300
            """)
            
            nodes = []
            node_map = {}  # 用于存储节点ID到节点数据的映射
            
            for record in nodes_result:
                node = record['n']
                if node:
                    node_id = str(node.element_id)
                    if node_id not in node_map:
                        node_map[node_id] = {
                            "id": node_id,
                            "label": node.get('name', 'Unknown'),
                            "title": self._get_node_title(node),
                            "color": self._get_node_color(node.labels),
                            "level": self._get_node_level(node.labels),
                            "properties": self._convert_properties(node)
                        }
            
            # 然后获取所有关系
            relationships_result = session.run("""
                MATCH (n)-[r:包含]->(m)
                WHERE n:根分类 OR n:一级分类 OR n:二级分类 OR n:knowledges7_entities
                RETURN r
                LIMIT 300
            """)
            
            relationships = []
            relationship_set = set()  # 用于避免重复关系
            
            for record in relationships_result:
                rel = record['r']
                if rel:
                    rel_id = str(rel.element_id)
                    if rel_id not in relationship_set:
                        relationship_set.add(rel_id)
                        relationships.append({
                            "id": rel_id,
                            "source": str(rel.start_node.element_id),
                            "target": str(rel.end_node.element_id),
                            "label": rel.type,
                            "title": rel.type,
                            "color": self._get_relationship_color(rel.type)
                        })
            
            # 将节点映射转换为列表
            nodes = list(node_map.values())
            
            return {"nodes": nodes, "relationships": relationships}

    def _convert_properties(self, node):
        """转换节点属性为可JSON序列化的格式"""
        properties = {}
        for key, value in node.items():
            try:
                # 尝试JSON序列化来检测不可序列化的对象
                json.dumps(value)
                properties[key] = value
            except (TypeError, ValueError):
                # 如果不可序列化，转换为字符串
                properties[key] = str(value)
        return properties

    def _format_graph_data_simple(self, record):
        """简化版图数据格式化"""
        nodes = []
        relationships = []
        node_set = set()
        
        # 处理节点
        for node_dict in record['nodes']:
            if not node_dict or 'node' not in node_dict:
                continue
                
            node = node_dict['node']
            if not node:
                continue
                
            node_id = str(node.element_id)
            if node_id in node_set:
                continue
            node_set.add(node_id)
            
            node_data = {
                "id": node_id,
                "label": node.get('name', 'Unknown'),
                "title": self._get_node_title(node),
                "color": self._get_node_color(node.labels),
                "level": self._get_node_level(node.labels),
                "properties": dict(node)
            }
            nodes.append(node_data)
        
        # 处理关系
        for rel in record['relationships']:
            if not rel:
                continue
                
            relationships.append({
                "id": str(rel.element_id),
                "source": str(rel.start_node.element_id),
                "target": str(rel.end_node.element_id),
                "label": rel.type,
                "title": rel.type,
                "color": self._get_relationship_color(rel.type)
            })
        
        return {"nodes": nodes, "relationships": relationships}

    def _format_graph_data(self, record):
        """格式化图数据"""
        nodes = []
        relationships = []
        node_set = set()
        
        # 处理节点
        for node in record['nodes']:
            node_id = str(node.element_id)
            if node_id in node_set:
                continue
            node_set.add(node_id)
            
            node_data = {
                "id": node_id,
                "label": node.get('name', 'Unknown'),
                "title": self._get_node_title(node),
                "color": self._get_node_color(node.labels),
                "level": self._get_node_level(node.labels),
                "properties": dict(node)
            }
            nodes.append(node_data)
        
        # 处理关系
        for rel in record['relationships']:
            relationships.append({
                "id": str(rel.element_id),
                "source": str(rel.start_node.element_id),
                "target": str(rel.end_node.element_id),
                "label": rel.type,
                "title": rel.type,
                "color": self._get_relationship_color(rel.type)
            })
        
        return {"nodes": nodes, "relationships": relationships}
    
    def search_knowledge(self, keyword):
        """搜索知识点 - 只搜索指定的节点类型"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                WHERE (n:根分类 OR n:一级分类 OR n:二级分类 OR n:knowledges7_entities)
                AND (n.name CONTAINS $keyword OR n.id CONTAINS $keyword)
                RETURN
                    n.name as name,
                    n.id as id,
                    labels(n) as labels,
                    n.chapter as chapter,
                    n.section as section,
                    n.grade as grade,
                    elementId(n) as element_id
                LIMIT 220
            """, keyword=keyword)
            
            results = []
            for record in result:
                data = dict(record)
                # 确保element_id字段存在
                if 'element_id' not in data:
                    data['element_id'] = str(data.get('id', ''))
                results.append(data)
            return results
    
    def _get_node_title(self, node):
        """获取节点标题"""
        labels = list(node.labels)
        name = node.get('name', 'Unknown')
        
        if '根分类' in labels:
            return f"根分类: {name}"
        elif '一级分类' in labels:
            return f"章节: {name}"
        elif '二级分类' in labels:
            return f"小节: {name}"
        else:
            return f"知识点: {name}"
    
    def _get_node_color(self, labels):
        """根据节点类型获取颜色"""
        color_map = {
            '根分类': '#FF6B6B',
            '一级分类': '#4ECDC4',
            '二级分类': '#45B7D1',
            'knowledges7_entities': '#96CEB4'
        }
        
        for label in labels:
            if label in color_map:
                return color_map[label]
        return '#95A5A6'
    
    def _get_relationship_color(self, rel_type):
        """根据关系类型获取颜色"""
        return '#3498DB'
    
    def _get_node_level(self, labels):
        """获取节点层级"""
        level_map = {
            '根分类': 0,
            '一级分类': 1,
            '二级分类': 2,
            'knowledges7_entities': 3
        }
        
        for label in labels:
            if label in level_map:
                return level_map[label]
        return 4

# 初始化知识图谱连接
math_kg = SimpleMathKnowledgeGraph()

@app.route('/')
def index():
    return render_template('simple_math_knowledge.html')

@app.route('/api/roots')
def get_roots():
    """获取根节点"""
    try:
        roots = math_kg.get_root_nodes()
        return jsonify({"success": True, "data": roots})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/chapters')
def get_chapters():
    """获取所有章节"""
    try:
        chapters = math_kg.get_chapters()
        return jsonify({"success": True, "data": chapters})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/sections')
def get_sections():
    """获取所有二级分类（小节）"""
    try:
        sections = math_kg.get_sections()
        return jsonify({"success": True, "data": sections})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/knowledges')
def get_knowledges():
    """获取所有知识点"""
    try:
        knowledges = math_kg.get_knowledges()
        return jsonify({"success": True, "data": knowledges})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/branch/<node_name>')
def get_branch(node_name):
    """获取知识分支"""
    depth = request.args.get('depth', 3, type=int)
    try:
        data = math_kg.find_knowledge_branch(node_name, depth)
        return jsonify({"success": True, "data": data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/search')
def search_knowledge():
    """搜索知识点"""
    keyword = request.args.get('q', '')
    if not keyword:
        return jsonify({"success": False, "error": "搜索关键词不能为空"})
    
    try:
        results = math_kg.search_knowledge(keyword)
        return jsonify({"success": True, "data": results})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/chapter/<chapter_id>')
def get_chapter_structure(chapter_id):
    """获取章节结构"""
    try:
        with math_kg.driver.session() as session:
            # 修复查询语法，避免嵌套聚合函数
            result = session.run("""
                MATCH (chapter:一级分类 {id: $chapter_id})
                OPTIONAL MATCH (chapter)-[:包含]->(section:二级分类)
                WITH chapter, section
                OPTIONAL MATCH (section)-[:包含]->(knowledge:knowledges7_entities)
                WITH chapter, section, collect(knowledge) as knowledges_list
                RETURN
                    chapter.name as chapter_name,
                    chapter.chapter as chapter_num,
                    collect({
                        name: section.name,
                        id: section.id,
                        section: section.section,
                        knowledges: [k in knowledges_list | {name: k.name, id: k.id}]
                    }) as sections
            """, chapter_id=chapter_id)
            
            data = result.single()
            if data:
                return jsonify({"success": True, "data": data})
            else:
                return jsonify({"success": False, "error": "章节未找到"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/full-graph')
def get_full_graph():
    """获取完整的知识图谱"""
    try:
        data = math_kg.get_full_graph()
        return jsonify({"success": True, "data": data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/health')
def health_check():
    """健康检查"""
    try:
        with math_kg.driver.session() as session:
            session.run("RETURN 1 as test")
        return jsonify({"status": "healthy", "database": "connected"})
    except Exception as e:
        return jsonify({"status": "unhealthy", "database": "disconnected", "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)