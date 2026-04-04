# KGGen使用示例
from kggen_client import KGGenClient
from config import Config

# 初始化客户端
client = KGGenClient()

# 示例：从文本提取实体和关系
def extract_entities_relations(text):
    """从文本提取实体和关系"""
    try:
        # 调用KGGen API
        result = client.extract_entities_and_relations(text)
        
        # 提取实体和关系
        entities = result.get('entities', [])
        relations = result.get('relations', [])
        
        print(f"提取到 {len(entities)} 个实体和 {len(relations)} 个关系")
        
        # 打印实体
        if entities:
            print("\n实体列表:")
            for entity in entities:
                print(f"  {entity['type']}: {entity['text']} (位置: {entity['start']}-{entity['end']})")
        
        # 打印关系
        if relations:
            print("\n关系列表:")
            for relation in relations:
                print(f"  {relation['subject']} --{relation['relation']}--> {relation['object']}")
        
        return result
        
    except Exception as e:
        print(f"提取失败: {e}")
        return None

# 使用示例
if __name__ == "__main__":
    text = "勾股定理指出：在直角三角形中，两直角边的平方和等于斜边的平方"
    result = extract_entities_relations(text)
    if result:
        print("\n提取成功！")