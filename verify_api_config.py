#!/usr/bin/env python3
"""
验证API配置是否正确设置
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def verify_main_config():
    """验证主配置文件"""
    try:
        from config import Config
        
        print("=== 主配置文件验证 ===")
        print(f"API密钥: {Config.KGGen_API_KEY}")
        print(f"超时时间: {Config.KGGEN_TIMEOUT}秒")
        print(f"最大重试次数: {Config.KGGEN_MAX_RETRIES}")
        print(f"API URL: {Config.KGGEN_API_URL}")
        
        # 验证API密钥格式
        if Config.KGGen_API_KEY == "b05528ee-73ba-4d5f-b15e-937b5993b53b":
            print("✅ API密钥正确设置")
        else:
            print("❌ API密钥设置错误")
            
        if Config.KGGEN_TIMEOUT >= 60:
            print("✅ 超时时间设置合理")
        else:
            print("❌ 超时时间可能需要调整")
            
        if Config.KGGEN_MAX_RETRIES >= 5:
            print("✅ 重试次数设置合理")
        else:
            print("❌ 重试次数可能需要增加")
            
    except ImportError as e:
        print(f"无法导入主配置: {e}")
    except Exception as e:
        print(f"主配置验证错误: {e}")

def verify_integrated_config():
    """验证集成配置文件"""
    try:
        from 知识图谱构建.config_integrated import KGGen_API_KEY, KGGEN_TIMEOUT, KGGEN_MAX_RETRIES, KGGEN_API_URL
        
        print("\n=== 集成配置文件验证 ===")
        print(f"API密钥: {KGGen_API_KEY}")
        print(f"超时时间: {KGGEN_TIMEOUT}秒")
        print(f"最大重试次数: {KGGEN_MAX_RETRIES}")
        print(f"API URL: {KGGEN_API_URL}")
        
        # 验证API密钥格式
        if KGGen_API_KEY == "b05528ee-73ba-4d5f-b15e-937b5993b53b":
            print("✅ API密钥正确设置")
        else:
            print("❌ API密钥设置错误")
            
        if KGGEN_TIMEOUT >= 60:
            print("✅ 超时时间设置合理")
        else:
            print("❌ 超时时间可能需要调整")
            
        if KGGEN_MAX_RETRIES >= 5:
            print("✅ 重试次数设置合理")
        else:
            print("❌ 重试次数可能需要增加")
            
    except ImportError as e:
        print(f"无法导入集成配置: {e}")
    except Exception as e:
        print(f"集成配置验证错误: {e}")

if __name__ == "__main__":
    print("开始验证API配置...")
    verify_main_config()
    verify_integrated_config()
    print("\n验证完成!")