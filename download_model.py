#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BERT模型下载脚本
用于在能够访问网络时下载所需的BERT模型
"""

import os
import sys
from transformers import BertTokenizer, BertModel

def download_bert_model(model_name="bert-base-chinese", local_dir="./models/bert-base-chinese"):
    """下载BERT模型到本地目录"""
    print(f"开始下载BERT模型: {model_name}")
    print(f"目标目录: {os.path.abspath(local_dir)}")
    
    # 创建目录
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        # 下载tokenizer
        print("下载tokenizer...")
        tokenizer = BertTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(local_dir)
        print("✓ tokenizer下载完成")
        
        # 下载模型
        print("下载模型...")
        model = BertModel.from_pretrained(model_name)
        model.save_pretrained(local_dir)
        print("✓ 模型下载完成")
        
        print(f"\n模型已成功下载到: {os.path.abspath(local_dir)}")
        return True
        
    except Exception as e:
        print(f"下载失败: {e}")
        print("\n建议解决方案:")
        print("1. 检查网络连接")
        print("2. 使用VPN或代理访问Hugging Face")
        print("3. 手动下载模型文件并放置到相应目录")
        return False

def check_model_exists(local_dir="./models/bert-base-chinese"):
    """检查本地模型是否存在"""
    required_files = [
        "config.json",
        "pytorch_model.bin",
        "vocab.txt",
        "tokenizer_config.json",
        "special_tokens_map.json"
    ]
    
    if not os.path.exists(local_dir):
        return False
        
    existing_files = os.listdir(local_dir)
    missing_files = [f for f in required_files if f not in existing_files]
    
    if missing_files:
        print(f"缺失文件: {missing_files}")
        return False
        
    return True

def main():
    """主函数"""
    print("=" * 50)
    print("BERT模型下载工具")
    print("=" * 50)
    
    local_dir = "./models/bert-base-chinese"
    
    # 检查是否已存在模型
    if check_model_exists(local_dir):
        print("✓ 本地模型已存在")
        print(f"位置: {os.path.abspath(local_dir)}")
        return True
    
    # 尝试下载模型
    print("本地模型不存在，开始下载...")
    success = download_bert_model(local_dir=local_dir)
    
    if success:
        # 更新配置文件使用本地模型
        update_config(local_dir)
        print("\n✓ 配置已更新为使用本地模型")
    
    return success

def update_config(local_dir):
    """更新配置文件使用本地模型路径"""
    config_file = os.path.join(os.path.dirname(__file__), "config.py")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换模型路径为本地路径
        new_content = content.replace(
            'BERT_MODEL = "bert-base-chinese"',
            f'BERT_MODEL = "{os.path.abspath(local_dir)}"'
        )
        
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
            
        print(f"已更新 {config_file} 使用本地模型路径")
        
    except Exception as e:
        print(f"更新配置文件失败: {e}")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)