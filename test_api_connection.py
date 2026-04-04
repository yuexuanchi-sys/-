#!/usr/bin/env python3
"""
简单的API连接测试脚本
用于验证DeepSeek API连接是否正常
"""

import os
import requests
import json
from config import Config

def test_deepseek_api():
    """测试DeepSeek API连接"""
    print("Testing DeepSeek API connection...")
    
    # 设置API密钥
    api_key = os.getenv('KGGEN_API_KEY', Config.KGGen_API_KEY)
    api_url = Config.KGGEN_API_URL
    
    print(f"API URL: {api_url}")
    print(f"API Key: {api_key[:10]}...{api_key[-4:]}")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 简单的测试请求
    payload = {
        "model": Config.KGGen_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "你是一个有帮助的AI助手。"
            },
            {
                "role": "user", 
                "content": "你好，请回复'API测试成功'"
            }
        ],
        "max_tokens": 50,
        "temperature": 0.1,
        "stream": False
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("API connection successful!")
            print(f"Response: {result}")
            return True
        else:
            print(f"API request failed: {response.status_code}")
            print(f"Error message: {response.text}")
            return False
            
    except Exception as e:
        print(f"API request exception: {e}")
        return False

if __name__ == "__main__":
    success = test_deepseek_api()
    if success:
        print("\nAPI connection test passed")
    else:
        print("\nAPI connection test failed")