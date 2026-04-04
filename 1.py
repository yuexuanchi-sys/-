import os
import docx  
from kg_gen import KGGen
import re

# 1. 基础配置
os.environ["DEEPSEEK_API_KEY"] = "sk-43738d2df5e84ea2b86d43fd53bf044b"

kg = KGGen(
    model="deepseek/deepseek-chat", 
    temperature=0.0 # 知识抽取必须保持0.0，减少大模型的幻觉
)

# ==================== 核心优化：定义极其严格的上下文规则 ====================
SYSTEM_CONTEXT = """
你是一个严谨的初中数学教育专家。当前输入的是初中七年级数学教材。
请基于文本抽取核心的数学知识图谱，必须严格遵守以下规则：

【规则 1：实体极简与去变量】
实体必须是数学概念名词。绝对不要提取具体的点、线、面字母（如 A, B, O, OA, 射线AB, x, y）。
- 正确实体：有理数、乘法分配律、绝对值、圆心、扇形。
- 错误实体：点O、OA、小明、甲乙、a、b。

【规则 2：关系（边）必须标准化】
边的名称必须极度精简，**只能**从以下词库中选择或组合：
[包含, 属于, 定义为, 计算法则, 性质, 前提条件, 符号表示, 等于]
绝对不要自己编造长句子的关系！

【示范样例】
输入：“像5, 1.2, 1/2这样大于0的数叫做正数。它们前面可以加上正号+。”
正确输出：
实体：正数, 0
关系：('正数', '定义为', '大于0的数'), ('正数', '符号表示', '正号+')

输入：“如图，以点O为圆心，线段OA为半径画圆。”
正确输出：(不提取任何关系，因为包含特定变量O和OA)
"""
# =========================================================================

print("正在读取 Word 文档...")
doc = docx.Document('D:\\数据\\7.11.docx')

# 简单的清洗：去掉多余的空格、制表符等
paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
full_text = '\n'.join(paragraphs)
# 清洗掉一些明显的无用标题（可根据实际情况增加）
full_text = re.sub(r'活动[一二三四五].*?\n', '', full_text)
full_text = re.sub(r'小组成员进行充分的讨论.*?\n', '', full_text)

# ==================== 核心优化：带重叠的滑动切片 ====================
chunk_size = 500  # 稍微放大一点，因为去掉了噪音
overlap = 50      # 前后重叠 50 个字符，防止定义被腰斩
chunks = []

# 步长为 chunk_size - overlap
for i in range(0, len(full_text), chunk_size - overlap):
    chunks.append(full_text[i : i + chunk_size])

print(f"文档预处理完成！总字符数: {len(full_text)}。")
print(f"采用滑动窗口切分为 {len(chunks)} 个片段（包含重叠区）。开始高精度抽取...\n")

global_entities = set()
global_edges = set()
global_relations = set()

# 遍历每一个文本块
for i, chunk in enumerate(chunks):
    print(f"[{i+1}/{len(chunks)}] 正在抽取...", end=" ", flush=True)
    try:
        # 【关键】：在此处传入强大的 context 约束！
        graph = kg.generate(input_data=chunk, context=SYSTEM_CONTEXT)
        
        if graph and hasattr(graph, 'relations'):
            global_entities.update(graph.entities)
            global_edges.update(graph.edges)
            global_relations.update(graph.relations)
            print(f"成功! 获取关系: {len(graph.relations)} 条")
        else:
            print("成功! 本段无核心知识。")
            
    except Exception as e:
        print(f"失败 (但不影响全局): {e}")
# ==================== 终极图谱后处理清洗 ====================
import re

clean_relations = set()
clean_entities = set()
clean_edges = set()

def is_garbage_entity(entity_name):
    # 如果实体长度只有1个或2个英文字母（如 O, a, AB, OA），认为是垃圾变量
    if re.fullmatch(r'[A-Za-z]{1,2}', entity_name):
        return True
    return False

for subj, edge, obj in global_relations:
    # 过滤掉包含垃圾变量的关系
    if is_garbage_entity(subj) or is_garbage_entity(obj):
        continue
    
    # 过滤掉关系词太长的废话（强制收敛）
    if len(edge) > 6:
        continue
        
    clean_relations.add((subj, edge, obj))
    clean_entities.add(subj)
    clean_entities.add(obj)
    clean_edges.add(edge)

# 替换回全局变量
global_entities = clean_entities
global_edges = clean_edges
global_relations = clean_relations
# ==========================================================

# ==================== 最终结果展示 ====================
print("\n" + "="*50)
print("🏆 高质量长文本知识图谱抽取完成！")
print(f"🎯 总计提纯实体 (Entities): {len(global_entities)} 个")
print(f"🎯 总计规范边 (Edges): {len(global_edges)} 种")
print(f"🎯 总计核心关系 (Relations): {len(global_relations)} 条")

# 预览前 15 条关系
relations_list = list(global_relations)
print("\n🔍 高质量关系预览 (前15条):")
for r in relations_list[:15]:
    print(r)
print("="*50)
from pyvis.network import Network

# ==================== 终极边关系清洗与合并 ====================
# 定义我们允许的标准关系字典（强制把发散的关系映射到标准词汇上）
# 你可以根据课本内容自己增删这个字典
edge_mapping = {
    "包括": "包含", "含有": "包含", "可以含有": "包含", "分为": "包含",
    "积为": "计算结果", "存在则积为": "计算结果", "等于": "计算结果",
    "再算": "运算顺序", "最后算": "运算顺序",
    "描述": "定义为", "应用于": "适用条件", "能表示": "性质",
    "≥": "数值关系", "<": "数值关系", ">": "数值关系"
}

final_relations = set()

# 强行清洗边，把相似的边合并，不认识的边统一叫“相关”
for subj, edge, obj in global_relations:
    # 尝试在映射表中找，找不到的看长度，太长的废话直接叫“相关属性”
    if edge in edge_mapping:
        standard_edge = edge_mapping[edge]
    elif len(edge) > 4:
        standard_edge = "相关属性"
    else:
        standard_edge = edge
        
    final_relations.add((subj, standard_edge, obj))

print("\n🚀 正在生成交互式可视化网页...")

# ==================== 生成交互式 HTML 图谱 ====================
# 初始化网络画布，开启物理引擎让节点自动排布
net = Network(height='800px', width='100%', bgcolor='#222222', font_color='white', directed=True)
net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08)

# 遍历最终的纯净关系集合，添加节点和边
for subj, edge, obj in final_relations:
    # 自动去重添加节点
    net.add_node(subj, label=subj, title=subj, color='#00a8ff', size=20)
    net.add_node(obj, label=obj, title=obj, color='#4cd137', size=20)
    # 添加带箭头的边
    net.add_edge(subj, obj, title=edge, label=edge, color='#7f8fa6')

# 保存为 HTML 文件
html_file_name = "math_knowledge_graph.html"
net.show(html_file_name, notebook=False)

print(f"🎉 成功！请在浏览器中打开文件: {html_file_name}")