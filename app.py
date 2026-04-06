from flask import Flask, request, jsonify, render_template
import json
import re
from config import Config

try:
    from neo4j import GraphDatabase
    _neo4j_available = True
except ImportError:
    _neo4j_available = False

app = Flask(__name__)

# ================================================================
# 离线数据源: 从 JSON 文件加载, 不依赖 Neo4j
# ================================================================

class OfflineGraphDataSource:
    """从 output/kggen_entities.json + kggen_relations.json 加载图数据"""

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = Config.OUTPUT_DIR
        self.entities = []
        self.relations = []
        self._load(output_dir)

    def _load(self, output_dir):
        import os
        ent_path = os.path.join(output_dir, 'kggen_entities.json')
        rel_path = os.path.join(output_dir, 'kggen_relations.json')
        if os.path.exists(ent_path):
            with open(ent_path, 'r', encoding='utf-8') as f:
                self.entities = json.load(f)
        if os.path.exists(rel_path):
            with open(rel_path, 'r', encoding='utf-8') as f:
                self.relations = json.load(f)

    # ------------ 颜色/类型映射 (与 Neo4j 版保持一致) --------
    TYPE_COLORS = {
        'CONCEPT': '#4A90D9', 'FORMULA': '#E8744F', 'THEOREM': '#67C23A',
        'METHOD': '#F7BA2A', 'EXAMPLE': '#909399', 'PROPERTY': '#E6A23C',
        'MODULE': '#5470c6', 'CHAPTER': '#91cc75',
    }

    def _node(self, e):
        return {
            'id': e['text'],
            'label': e['text'],
            'title': f"{e.get('type','')}: {e['text']}",
            'color': self.TYPE_COLORS.get(e.get('type', ''), '#999'),
            'level': 'concept',
            'properties': {k: str(v) for k, v in e.items()},
        }

    def _edge(self, r):
        return {
            'from': r['subject'],
            'to': r['object'],
            'label': r['relation'],
            'title': f"{r['subject']} —{r['relation']}→ {r['object']}",
        }

    def get_full_graph(self):
        nodes_map = {}
        for e in self.entities:
            nodes_map[e['text']] = self._node(e)
        edges = [self._edge(r) for r in self.relations
                 if r['subject'] in nodes_map and r['object'] in nodes_map]
        return {'nodes': list(nodes_map.values()), 'relationships': edges}

    def get_all_chapters(self):
        return [e['text'] for e in self.entities if e.get('type') == 'CHAPTER']

    def find_knowledge_branch(self, node_name, max_depth=3):
        children = set()
        for r in self.relations:
            if r['subject'] == node_name and r['relation'] == '包含':
                children.add(r['object'])
        nodes_map = {}
        for e in self.entities:
            if e['text'] == node_name or e['text'] in children:
                nodes_map[e['text']] = self._node(e)
        edges = [self._edge(r) for r in self.relations
                 if r['subject'] in nodes_map and r['object'] in nodes_map]
        return {'nodes': list(nodes_map.values()), 'relationships': edges}

    def search_knowledge(self, keyword):
        nodes_map = {}
        for e in self.entities:
            if keyword in e['text']:
                nodes_map[e['text']] = self._node(e)
        edges = [self._edge(r) for r in self.relations
                 if r['subject'] in nodes_map and r['object'] in nodes_map]
        return {'nodes': list(nodes_map.values()), 'relationships': edges}

    def find_knowledge_path(self, start, end):
        # BFS shortest path
        from collections import deque
        adj = {}
        for r in self.relations:
            adj.setdefault(r['subject'], []).append((r['object'], r))
            adj.setdefault(r['object'], []).append((r['subject'], r))
        visited = {start}
        queue = deque([(start, [])])
        while queue:
            cur, path = queue.popleft()
            if cur == end:
                nodes_map = {}
                for e in self.entities:
                    if e['text'] in {start, end} or any(e['text'] in (r['subject'], r['object']) for r in path):
                        nodes_map[e['text']] = self._node(e)
                return {'nodes': list(nodes_map.values()), 'relationships': [self._edge(r) for r in path]}
            for nbr, rel in adj.get(cur, []):
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, path + [rel]))
        return {'nodes': [], 'relationships': []}


class MathKnowledgeGraph:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            Config.NEO4J_URI,
            auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD)
        )
    
    def close(self):
        self.driver.close()
    
    def find_knowledge_branch(self, node_name, max_depth=3):
        """查找知识节点的完整分支"""
        with self.driver.session() as session:
            # 使用更直接的方法：查询节点及其直接子节点
            result = session.run("""
                MATCH (start {name: $node_name})
                OPTIONAL MATCH (start)-[r:包含]->(child)
                RETURN start, child, r
            """, node_name=node_name)
            
            nodes = []
            relationships = []
            node_set = set()
            
            for record in result:
                # 处理起始节点
                start_node = record['start']
                if start_node:
                    start_id = str(start_node.element_id)
                    if start_id not in node_set:
                        node_set.add(start_id)
                        nodes.append({
                            "id": start_id,
                            "label": start_node.get('name', 'Unknown'),
                            "title": self._get_node_title(start_node),
                            "color": self._get_node_color(start_node.labels),
                            "level": self._get_node_level(start_node.labels),
                            "properties": self._serialize_properties(start_node)
                        })
                
                # 处理子节点
                child_node = record['child']
                if child_node:
                    child_id = str(child_node.element_id)
                    if child_id not in node_set:
                        node_set.add(child_id)
                        nodes.append({
                            "id": child_id,
                            "label": child_node.get('name', 'Unknown'),
                            "title": self._get_node_title(child_node),
                            "color": self._get_node_color(child_node.labels),
                            "level": self._get_node_level(child_node.labels),
                            "properties": self._serialize_properties(child_node)
                        })
                
                # 处理关系
                rel = record['r']
                if rel:
                    relationships.append({
                        "id": str(rel.element_id),
                        "source": str(rel.start_node.element_id),
                        "target": str(rel.end_node.element_id),
                        "label": rel.type,
                        "title": rel.type,
                        "color": self._get_relationship_color(rel.type)
                    })
            
            return {"nodes": nodes, "relationships": relationships}
    
    def find_knowledge_path(self, start_node, end_node):
        """查找两个知识节点间的路径"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (start {name: $start_node}), (end {name: $end_node})
                MATCH path = shortestPath((start)-[:包含*..10]-(end))
                RETURN path
            """, start_node=start_node, end_node=end_node)
            
            records = result.data()
            if not records or not records[0]['path']:
                return {"nodes": [], "relationships": []}
            
            return self._format_path_data(records[0]['path'])
    
    def get_all_chapters(self):
        """获取所有章节信息"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n:一级分类)
                RETURN n.chapter as chapter, n.name as name, n.id as id
                ORDER BY n.chapter
            """)
            return [dict(record) for record in result]
    
    def get_knowledge_by_level(self, level):
        """根据层级获取知识点"""
        with self.driver.session() as session:
            if level == "根分类":
                query = "MATCH (n:根分类) RETURN n.name as name, n.id as id, n.grade as grade"
            elif level == "一级分类":
                query = "MATCH (n:一级分类) RETURN n.name as name, n.id as id, n.chapter as chapter"
            elif level == "二级分类":
                query = "MATCH (n:二级分类) RETURN n.name as name, n.id as id, n.section as section"
            else:  # 知识点
                query = "MATCH (n:knowledges7_entities) RETURN n.name as name, n.id as id"
            
            result = session.run(query)
            return [dict(record) for record in result]
    
    def search_knowledge(self, keyword):
        """搜索知识点"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                WHERE n.name CONTAINS $keyword OR n.id CONTAINS $keyword
                RETURN
                    n.name as name,
                    n.id as id,
                    labels(n) as labels,
                    n.chapter as chapter,
                    n.section as section,
                    n.grade as grade
                LIMIT 50
            """, keyword=keyword)
            return [dict(record) for record in result]
    
    def get_chapter_structure(self, chapter_id):
        """获取章节的完整结构"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (chapter:一级分类 {id: $chapter_id})
                OPTIONAL MATCH (chapter)-[:包含]->(section:二级分类)
                OPTIONAL MATCH (section)-[:包含]->(knowledge:knowledges7_entities)
                RETURN
                    chapter.name as chapter_name,
                    chapter.chapter as chapter_num,
                    collect(DISTINCT {
                        name: section.name,
                        id: section.id,
                        section: section.section,
                        knowledges: collect(DISTINCT {
                            name: knowledge.name,
                            id: knowledge.id
                        })
                    }) as sections
            """, chapter_id=chapter_id)
            
            return result.single()
    
    def _format_simple_graph_data(self, record):
        """简化版本的图数据格式化，兼容Node对象和字典"""
        nodes = []
        relationships = []
        node_set = set()
        
        # 添加起始节点
        start_node = record['start']
        
        # 处理Node对象或字典
        if hasattr(start_node, 'id'):
            # Neo4j Node对象
            start_id = str(start_node.id)
            start_name = start_node.get('name', 'Unknown')
            start_labels = list(start_node.labels) if hasattr(start_node, 'labels') else []
            start_properties = dict(start_node)
        else:
            # 字典对象
            start_id = str(start_node.get('id', '0'))
            start_name = start_node.get('name', 'Unknown')
            start_labels = start_node.get('labels', [])
            start_properties = start_node
        
        node_set.add(start_id)
        
        nodes.append({
            "id": start_id,
            "label": start_name,
            "title": self._get_node_title_from_dict({'name': start_name, 'labels': start_labels, **start_properties}),
            "color": self._get_node_color(start_labels),
            "level": self._get_node_level(start_labels),
            "properties": start_properties
        })
        
        # 处理子节点
        children = record['children'] or []
        for child in children:
            if hasattr(child, 'id'):
                child_id = str(child.id)
                child_name = child.get('name', 'Unknown')
                child_labels = list(child.labels) if hasattr(child, 'labels') else []
                child_properties = dict(child)
            else:
                child_id = str(child.get('id', '0'))
                child_name = child.get('name', 'Unknown')
                child_labels = child.get('labels', [])
                child_properties = child
                
            if child_id in node_set:
                continue
            node_set.add(child_id)
            
            nodes.append({
                "id": child_id,
                "label": child_name,
                "title": self._get_node_title_from_dict({'name': child_name, 'labels': child_labels, **child_properties}),
                "color": self._get_node_color(child_labels),
                "level": self._get_node_level(child_labels),
                "properties": child_properties
            })
        
        # 处理关系
        rels = record['relationships'] or []
        for rel in rels:
            if hasattr(rel, 'id'):
                rel_id = str(rel.id)
                rel_type = rel.type
                source_id = str(rel.start_node.id)
                target_id = str(rel.end_node.id)
            else:
                rel_id = str(rel.get('id', '0'))
                rel_type = rel.get('type', '包含')
                source_id = str(rel.get('source', '0'))
                target_id = str(rel.get('target', '0'))
            
            relationships.append({
                "id": rel_id,
                "source": source_id,
                "target": target_id,
                "label": rel_type,
                "title": rel_type,
                "color": self._get_relationship_color(rel_type)
            })
        
        return {"nodes": nodes, "relationships": relationships}
    
    def _get_node_title_from_dict(self, node_dict):
        """从字典获取节点标题信息"""
        labels = node_dict.get('labels', [])
        name = node_dict.get('name', 'Unknown')
        
        if '根分类' in labels:
            return f"根分类: {name}\n年级: {node_dict.get('grade', '')}"
        elif '一级分类' in labels:
            return f"章节{node_dict.get('chapter', '')}: {name}"
        elif '二级分类' in labels:
            return f"小节{node_dict.get('section', '')}: {name}"
        else:
            return f"知识点: {name}"
    
    def _serialize_properties(self, node):
        """序列化节点属性，处理不可序列化的对象"""
        properties = {}
        for key, value in node.items():
            try:
                # 尝试JSON序列化来检测不可序列化的对象
                json.dumps(value)
                properties[key] = value
            except (TypeError, ValueError):
                # 如果无法序列化，转换为字符串
                properties[key] = str(value)
        return properties
    
    def _format_path_data(self, path):
        """格式化路径数据"""
        nodes = []
        relationships = []
        node_set = set()
        
        # 添加起始节点
        start_node = path.start_node
        start_id = str(start_node.id)
        node_set.add(start_id)
        
        nodes.append({
            "id": start_id,
            "label": start_node.get('name', 'Unknown'),
            "title": self._get_node_title(start_node),
            "color": self._get_node_color(start_node.labels),
            "level": self._get_node_level(start_node.labels)
        })
        
        # 处理路径中的关系和节点
        for rel in path.relationships:
            end_node = rel.end_node
            end_id = str(end_node.id)
            
            if end_id not in node_set:
                node_set.add(end_id)
                nodes.append({
                    "id": end_id,
                    "label": end_node.get('name', 'Unknown'),
                    "title": self._get_node_title(end_node),
                    "color": self._get_node_color(end_node.labels),
                    "level": self._get_node_level(end_node.labels)
                })
            
            relationships.append({
                "id": str(rel.id),
                "source": str(rel.start_node.id),
                "target": end_id,
                "label": rel.type,
                "title": rel.type,
                "color": self._get_relationship_color(rel.type)
            })
        
        return {"nodes": nodes, "relationships": relationships}
    
    def _get_node_title(self, node):
        """获取节点标题信息"""
        labels = list(node.labels)
        name = node.get('name', 'Unknown')
        
        if '根分类' in labels:
            return f"根分类: {name}\n年级: {node.get('grade', '')}"
        elif '一级分类' in labels:
            return f"章节{node.get('chapter', '')}: {name}"
        elif '二级分类' in labels:
            return f"小节{node.get('section', '')}: {name}"
        else:
            return f"知识点: {name}"
    
    def _get_node_color(self, labels):
        """根据节点类型获取颜色"""
        color_map = {
            '根分类': '#FF6B6B',      # 红色
            '一级分类': '#4ECDC4',     # 青色
            '二级分类': '#45B7D1',     # 蓝色
            'knowledges7_entities': '#96CEB4'  # 绿色
        }
        
        for label in labels:
            if label in color_map:
                return color_map[label]
        return '#95A5A6'  # 默认灰色
    
    def _get_relationship_color(self, rel_type):
        """根据关系类型获取颜色"""
        return {
            '包含': '#3498DB',
            '分为': '#9B59B6',
            '具有': '#E74C3C',
            '包括': '#F39C12'
        }.get(rel_type, '#95A5A6')
    
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

    def get_mock_data(self):
        """获取模拟数据用于演示"""
        nodes = [
            {
                "id": "1",
                "label": "初中数学",
                "title": "根分类: 初中数学\n年级: 初中",
                "color": "#FF6B6B",
                "level": 0,
                "properties": {"name": "初中数学", "grade": "初中"}
            },
            {
                "id": "2",
                "label": "有理数",
                "title": "章节1: 有理数",
                "color": "#4ECDC4",
                "level": 1,
                "properties": {"name": "有理数", "chapter": "1"}
            },
            {
                "id": "3",
                "label": "整式的加减",
                "title": "章节2: 整式的加减",
                "color": "#4ECDC4",
                "level": 1,
                "properties": {"name": "整式的加减", "chapter": "2"}
            },
            {
                "id": "4",
                "label": "一元一次方程",
                "title": "章节3: 一元一次方程",
                "color": "#4ECDC4",
                "level": 1,
                "properties": {"name": "一元一次方程", "chapter": "3"}
            },
            {
                "id": "5",
                "label": "正数和负数",
                "title": "小节1.1: 正数和负数",
                "color": "#45B7D1",
                "level": 2,
                "properties": {"name": "正数和负数", "section": "1.1"}
            },
            {
                "id": "6",
                "label": "有理数的加减",
                "title": "小节1.2: 有理数的加减",
                "color": "#45B7D1",
                "level": 2,
                "properties": {"name": "有理数的加减", "section": "1.2"}
            },
            {
                "id": "7",
                "label": "单项式",
                "title": "小节2.1: 单项式",
                "color": "#45B7D1",
                "level": 2,
                "properties": {"name": "单项式", "section": "2.1"}
            },
            {
                "id": "8",
                "label": "正数的概念",
                "title": "知识点: 正数的概念",
                "color": "#96CEB4",
                "level": 3,
                "properties": {"name": "正数的概念"}
            },
            {
                "id": "9",
                "label": "负数的概念",
                "title": "知识点: 负数的概念",
                "color": "#96CEB4",
                "level": 3,
                "properties": {"name": "负数的概念"}
            },
            {
                "id": "10",
                "label": "有理数加法法则",
                "title": "知识点: 有理数加法法则",
                "color": "#96CEB4",
                "level": 3,
                "properties": {"name": "有理数加法法则"}
            }
        ]
        
        relationships = [
            {"id": "1", "source": "1", "target": "2", "label": "包含", "title": "包含", "color": "#3498DB"},
            {"id": "2", "source": "1", "target": "3", "label": "包含", "title": "包含", "color": "#3498DB"},
            {"id": "3", "source": "1", "target": "4", "label": "包含", "title": "包含", "color": "#3498DB"},
            {"id": "4", "source": "2", "target": "5", "label": "包含", "title": "包含", "color": "#3498DB"},
            {"id": "5", "source": "2", "target": "6", "label": "包含", "title": "包含", "color": "#3498DB"},
            {"id": "6", "source": "3", "target": "7", "label": "包含", "title": "包含", "color": "#3498DB"},
            {"id": "7", "source": "5", "target": "8", "label": "包含", "title": "包含", "color": "#3498DB"},
            {"id": "8", "source": "5", "target": "9", "label": "包含", "title": "包含", "color": "#3498DB"},
            {"id": "9", "source": "6", "target": "10", "label": "包含", "title": "包含", "color": "#3498DB"}
        ]
        
        return {"nodes": nodes, "relationships": relationships}

    def get_full_graph(self):
        """获取完整的知识图谱（所有节点和关系）"""
        with self.driver.session() as session:
            # 获取所有节点
            result = session.run("""
                MATCH (n)
                RETURN n
            """)
            
            nodes = []
            node_set = set()
            
            for record in result:
                node = record['n']
                node_id = str(node.element_id)
                
                if node_id not in node_set:
                    node_set.add(node_id)
                    nodes.append({
                        "id": node_id,
                        "label": node.get('name', 'Unknown'),
                        "title": self._get_node_title(node),
                        "color": self._get_node_color(node.labels),
                        "level": self._get_node_level(node.labels),
                        "properties": self._serialize_properties(node)
                    })
            
            # 获取所有关系
            result = session.run("""
                MATCH ()-[r]->()
                RETURN r
            """)
            
            relationships = []
            
            for record in result:
                rel = record['r']
                relationships.append({
                    "id": str(rel.element_id),
                    "source": str(rel.start_node.element_id),
                    "target": str(rel.end_node.element_id),
                    "label": rel.type,
                    "title": rel.type,
                    "color": self._get_relationship_color(rel.type)
                })
            
            return {"nodes": nodes, "relationships": relationships}

# 初始化: 优先 Neo4j, 不可用时自动切换离线模式
_offline = True
_offline_ds = None
math_kg = None

if _neo4j_available:
    try:
        math_kg = MathKnowledgeGraph()
        with math_kg.driver.session() as session:
            session.run("RETURN 1")
        _offline = False
        print("[OK] Neo4j 连接成功，使用在线模式")
    except Exception as e:
        math_kg = None
        print(f"[WARN] Neo4j 连接失败 ({e})，切换离线模式")
else:
    print("[INFO] neo4j 模块未安装，使用离线模式")

if _offline:
    _offline_ds = OfflineGraphDataSource()
    print(f"[OK] 离线模式已就绪 (实体: {len(_offline_ds.entities)}, 关系: {len(_offline_ds.relations)})")

@app.route('/')
def index():
    return render_template('math_knowledge.html')

def _ds():
    """返回当前可用的数据源 (Neo4j 或 离线JSON)"""
    return _offline_ds if _offline else math_kg

@app.route('/api/full-graph')
def get_full_graph():
    """获取完整的知识图谱"""
    try:
        data = _ds().get_full_graph()
        return jsonify({"success": True, "data": data, "offline": _offline})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/chapters')
def get_chapters():
    """获取所有章节"""
    try:
        chapters = _ds().get_all_chapters()
        return jsonify({"success": True, "data": chapters})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/branch/<node_name>')
def get_branch(node_name):
    """获取知识分支"""
    depth = request.args.get('depth', 3, type=int)
    try:
        data = _ds().find_knowledge_branch(node_name, depth)
        return jsonify({"success": True, "data": data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/path')
def get_path():
    """查找知识路径"""
    start_node = request.args.get('start')
    end_node = request.args.get('end')

    if not start_node or not end_node:
        return jsonify({"success": False, "error": "起始节点和目标节点不能为空"})

    try:
        data = _ds().find_knowledge_path(start_node, end_node)
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
        results = _ds().search_knowledge(keyword)
        return jsonify({"success": True, "data": results})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/level/<level_type>')
def get_by_level(level_type):
    """按层级获取知识"""
    if _offline:
        return jsonify({"success": False, "error": "离线模式不支持层级查询"})
    try:
        results = math_kg.get_knowledge_by_level(level_type)
        return jsonify({"success": True, "data": results})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/chapter/<chapter_id>')
def get_chapter_structure(chapter_id):
    """获取章节结构"""
    if _offline:
        return jsonify({"success": False, "error": "离线模式不支持章节查询"})
    try:
        structure = math_kg.get_chapter_structure(chapter_id)
        return jsonify({"success": True, "data": structure})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/related-nodes/<node_name>')
def get_related_nodes(node_name):
    """获取相关节点"""
    if _offline:
        return jsonify({"success": False, "error": "离线模式不支持关联查询"})
    try:
        with math_kg.driver.session() as session:
            result = session.run("""
                MATCH (start {name: $node_name})--(related)
                RETURN DISTINCT related.name as name, labels(related) as labels
                LIMIT 10
            """, node_name=node_name)
            
            related_nodes = [dict(record) for record in result]
            return jsonify({"success": True, "data": related_nodes})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/health')
def health_check():
    """健康检查端点"""
    if _offline:
        return jsonify({
            'status': 'healthy',
            'mode': 'offline',
            'entities': len(_offline_ds.entities),
            'relations': len(_offline_ds.relations),
        })
    try:
        with math_kg.driver.session() as session:
            session.run("RETURN 1 as test")
        return jsonify({
            'status': 'healthy',
            'mode': 'neo4j',
            'database': 'connected'
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'database': 'disconnected',
            'error': str(e)
        }), 500

@app.route('/api/test-data')
def test_data():
    """测试数据端点，用于验证数据格式"""
    if _offline:
        return jsonify({"success": False, "error": "离线模式不支持测试数据查询"})
    try:
        # 测试获取根分类数据
        with math_kg.driver.session() as session:
            result = session.run("""
                MATCH (n:根分类)
                RETURN n.name as name, n.id as id, n.grade as grade
                LIMIT 5
            """)
            root_nodes = [dict(record) for record in result]
            
            # 测试获取一级分类数据
            result = session.run("""
                MATCH (n:一级分类)
                RETURN n.name as name, n.id as id, n.chapter as chapter
                LIMIT 5
            """)
            chapter_nodes = [dict(record) for record in result]
            
            # 测试获取节点总数
            result = session.run("""
                MATCH (n)
                RETURN count(n) as total_nodes
            """)
            total_nodes = result.single()['total_nodes']
            
        return jsonify({
            'success': True,
            'data': {
                'root_nodes': root_nodes,
                'chapter_nodes': chapter_nodes,
                'total_nodes': total_nodes,
                'message': f'数据库连接正常，共找到 {total_nodes} 个节点'
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.route('/test')
def test_chart():
    """测试图表页面"""
    return render_template('test_chart.html')

@app.route('/simple-test')
def simple_test():
    """简单测试页面"""
    return render_template('simple_test.html')

@app.route('/debug-test')
def debug_test():
    """调试测试页面"""
    return render_template('debug_test.html')
if __name__ == '__main__':
    app.run(debug=True, port=5000)