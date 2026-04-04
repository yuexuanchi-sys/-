#!/usr/bin/env python3
"""
检查集成版本的超时设置
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from 知识图谱构建.ark_integration.kggen_client import KGGenClient
    from 知识图谱构建.config_integrated import KGGEN_TIMEOUT, KGGEN_MAX_RETRIES
    
    print(f"集成版本超时设置: {KGGEN_TIMEOUT}秒")
    print(f"集成版本最大重试次数: {KGGEN_MAX_RETRIES}")
    
    client = KGGenClient()
    print(f"客户端超时设置: {client.timeout}秒")
    print(f"客户端最大重试次数: {client.max_retries}")
    
except ImportError as e:
    print(f"导入错误: {e}")
except Exception as e:
    print(f"错误: {e}")