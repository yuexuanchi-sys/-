

#!/usr/bin/env python3
"""
KGGen安装和配置脚本
这个脚本帮助安装和配置KGGen包，并验证配置是否正确
"""

import os
import sys
import subprocess
from config import Config

def install_kggen():
    """KGGen API服务设置（无需安装包，只需配置API密钥）"""
    print("[INFO] KGGen 是一个API服务，无需安装Python包")
    print("[INFO] 您只需要设置API密钥即可使用")
    return True

def setup_environment():
    """设置环境变量"""
    print("\n=== 环境变量设置 ===")
    
    # 获取API密钥
    api_key = input("请输入您的KGGen API密钥（如果已设置环境变量可直接回车）: ").strip()
    
    if api_key:
        # 设置环境变量（当前会话）
        os.environ['KGGEN_API_KEY'] = api_key
        print("[OK] API密钥已设置为当前会话的环境变量")
        
        # 提示用户永久设置
        print("\n提示: 永久设置环境变量方法:")
        print("Windows (CMD):")
        print("  setx KGGEN_API_KEY your_api_key")
        print("Windows (PowerShell):")
        print("  [Environment]::SetEnvironmentVariable('KGGEN_API_KEY', 'your_api_key', 'User')")
        print("Linux/macOS:")
        print("  echo 'export KGGEN_API_KEY=your_api_key' >> ~/.bashrc")
        print("  echo 'export KGGEN_API_KEY=your_api_key' >> ~/.zshrc")
        print("  然后运行: source ~/.bashrc 或 source ~/.zshrc")
    else:
        # 检查是否已设置
        if 'KGGEN_API_KEY' in os.environ:
            print("[OK] 使用已设置的环境变量 KGGEN_API_KEY")
        else:
            print("[ERROR] 未提供API密钥且环境变量未设置")
            return False
    
    return True

def test_kggen_connection():
    """测试KGGen连接"""
    print("\n=== 测试KGGen连接 ===")
    
    try:
        from kggen_client import KGGenClient
        from config import Config
        
        # 验证配置
        if not Config.validate_kggen_config():
            print("[ERROR] KGGen配置验证失败")
            return False
        
        # 创建客户端
        client = KGGenClient()
        
        # 简单测试
        print("[OK] KGGen客户端创建成功")
        print(f"API URL: {Config.KGGEN_API_URL}")
        print(f"超时设置: {Config.KGGEN_TIMEOUT}秒")
        
        # 测试API连接
        test_text = "勾股定理"
        print(f"测试文本: {test_text}")
        
        result = client.extract_entities_and_relations(test_text)
        if result:
            print("[OK] API连接测试成功")
            print(f"提取到 {len(result.get('entities', []))} 个实体")
            print(f"提取到 {len(result.get('relations', []))} 个关系")
        else:
            print("[WARN] API调用未返回结果，可能是网络或配置问题")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] KGGen连接测试失败: {e}")
        return False

def create_example_usage():
    """创建使用示例"""
    example_code = '''
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
            print("\\n实体列表:")
            for entity in entities:
                print(f"  {entity['type']}: {entity['text']} (位置: {entity['start']}-{entity['end']})")
        
        # 打印关系
        if relations:
            print("\\n关系列表:")
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
        print("\\n提取成功！")
'''.strip()

    # 保存示例文件
    example_path = os.path.join(os.path.dirname(__file__), "kggen_example.py")
    with open(example_path, 'w', encoding='utf-8') as f:
        f.write(example_code)
    
    print(f"✅ 使用示例已保存到: {example_path}")

def main():
    """主函数"""
    print("KGGen API服务配置工具")
    print("=" * 50)
    
    # 安装KGGen（实际上只是配置）
    if not install_kggen():
        sys.exit(1)
    
    # 设置环境变量
    if not setup_environment():
        sys.exit(1)
    
    # 测试连接
    if not test_kggen_connection():
        print("[WARN] 连接测试失败，请检查API密钥和网络连接")
        # 不退出，可能只是网络问题
    
    # 创建使用示例
    create_example_usage()
    
    print("\n" + "=" * 50)
    print("[OK] KGGen API服务配置完成！")
    print("下一步:")
    print("1. 确保设置永久环境变量")
    print("2. 查看 kggen_example.py 文件了解使用方法")
    print("3. 运行 main_kggen_pipeline.py 开始构建知识图谱")

if __name__ == "__main__":
    main()