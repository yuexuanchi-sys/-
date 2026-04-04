#!/usr/bin/env python3
"""
测试修复的脚本
1. 测试Word文档读取功能（包括宏启用文档）
2. 测试KGGen API连接（可选）
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_word_document_reading():
    """测试Word文档读取功能"""
    from data_loader import MathDataLoader
    
    logger.info("测试Word文档读取功能...")
    
    # 创建数据加载器
    loader = MathDataLoader()
    
    # 测试文件路径（根据用户提供的文件列表）
    test_files = [
        "D:\\数据\\8.1.docx",  # 宏启用文档
        "D:\\数据\\8.11.docx",
        "D:\\数据\\8.2.docx",
        "D:\\数据\\9.1.docx", 
        "D:\\数据\\9.11.docx",
        "D:\\数据\\9.2.docx"
    ]
    
    success_count = 0
    for file_path in test_files:
        if not os.path.exists(file_path):
            logger.warning(f"文件不存在: {file_path}")
            continue
            
        try:
            logger.info(f"读取文件: {file_path}")
            content = loader.read_file_content(file_path)
            
            if content and content.strip():
                logger.info(f"成功读取文件，内容长度: {len(content)} 字符")
                logger.info(f"内容预览: {content[:200]}...")
                success_count += 1
            else:
                logger.warning(f"文件内容为空: {file_path}")
                
        except Exception as e:
            logger.error(f"读取文件失败 {file_path}: {e}")
    
    logger.info(f"Word文档读取测试完成: {success_count}/{len(test_files)} 个文件成功读取")

def test_kggen_connection():
    """测试KGGen API连接"""
    from kggen_client import KGGenClient
    from config import Config
    
    logger.info("测试KGGen API连接...")
    
    # 验证配置
    if not Config.validate_kggen_config():
        logger.warning("KGGen配置未正确设置，跳过API测试")
        return
    
    # 创建客户端
    client = KGGenClient()
    
    # 测试文本
    test_text = "勾股定理指出：在直角三角形中，两直角边的平方和等于斜边的平方，即 a² + b² = c²"
    
    try:
        logger.info(f"发送测试请求，文本长度: {len(test_text)} 字符")
        result = client.extract_entities_and_relations(test_text)
        
        if result and "entities" in result:
            logger.info("KGGen API连接成功!")
            logger.info(f"提取到 {len(result['entities'])} 个实体")
            logger.info(f"提取到 {len(result['relations'])} 个关系")
            
            # 打印实体
            for entity in result["entities"]:
                logger.info(f"实体: {entity['type']} - {entity['text']}")
                
            # 打印关系
            for relation in result["relations"]:
                logger.info(f"关系: {relation['subject']} --{relation['relation']}--> {relation['object']}")
        else:
            logger.warning("KGGen API返回空结果")
            
    except Exception as e:
        logger.error(f"KGGen API测试失败: {e}")

if __name__ == "__main__":
    logger.info("开始测试修复...")
    
    # 测试Word文档读取
    test_word_document_reading()
    
    # 测试KGGen API连接（可选，如果需要测试API）
    # test_kggen_connection()
    
    logger.info("测试完成!")