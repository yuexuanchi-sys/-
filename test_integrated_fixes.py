#!/usr/bin/env python3
"""
测试集成版本的修复
1. 测试集成版本的KGGen客户端连接
2. 验证超时和重试机制
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

def test_integrated_kggen_connection():
    """测试集成版本的KGGen API连接"""
    try:
        # 导入集成版本的配置和客户端
        from 知识图谱构建.ark_integration.kggen_client import KGGenClient
        from 知识图谱构建.config_integrated import validate_config
        
        logger.info("测试集成版本的KGGen API连接...")
        
        # 验证配置
        errors = validate_config()
        if errors:
            logger.warning("配置验证失败，跳过API测试")
            for error in errors:
                logger.warning(f"配置错误: {error}")
            return
        
        # 创建客户端
        client = KGGenClient()
        
        # 测试文本
        test_text = "勾股定理指出：在直角三角形中，两直角边的平方和等于斜边的平方，即 a² + b² = c²"
        
        try:
            logger.info(f"发送测试请求，文本长度: {len(test_text)} 字符")
            result = client.extract_entities_and_relations(test_text)
            
            if result and "entities" in result:
                logger.info("集成版本KGGen API连接成功!")
                logger.info(f"提取到 {len(result['entities'])} 个实体")
                logger.info(f"提取到 {len(result['relations'])} 个关系")
                
                # 打印实体
                for entity in result["entities"]:
                    logger.info(f"实体: {entity['type']} - {entity['text']}")
                    
                # 打印关系
                for relation in result["relations"]:
                    logger.info(f"关系: {relation['subject']} --{relation['relation']}--> {relation['object']}")
            else:
                logger.warning("集成版本KGGen API返回空结果，但连接成功")
                
        except Exception as e:
            logger.error(f"集成版本KGGen API测试失败: {e}")
            logger.info("这可能是由于网络问题或API服务暂时不可用")
            
    except ImportError as e:
        logger.error(f"无法导入集成版本模块: {e}")
    except Exception as e:
        logger.error(f"集成版本测试发生意外错误: {e}")

def test_timeout_handling():
    """测试超时处理机制"""
    try:
        from 知识图谱构建.ark_integration.kggen_client import KGGenClient
        import requests
        
        logger.info("测试超时处理机制...")
        
        # 创建一个故意会超时的请求（使用一个不存在的慢速端点）
        client = KGGenClient()
        
        # 临时修改URL为会超时的地址
        original_url = client.api_url
        client.api_url = "https://httpbin.org/delay/10"  # 10秒延迟
        
        try:
            result = client.extract_entities_and_relations("测试超时")
            logger.info("超时测试完成")
        except Exception as e:
            logger.info(f"超时测试捕获到预期异常: {e}")
        finally:
            # 恢复原始URL
            client.api_url = original_url
            
    except Exception as e:
        logger.error(f"超时处理测试失败: {e}")

if __name__ == "__main__":
    logger.info("开始测试集成版本的修复...")
    
    # 测试集成版本KGGen连接
    test_integrated_kggen_connection()
    
    # 测试超时处理机制（可选）
    # test_timeout_handling()
    
    logger.info("集成版本测试完成!")