#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LTP 模型自动修复脚本
自动下载并替换损坏的 LTP 模型文件
"""

import os
import sys
import requests
import logging
from pathlib import Path
import shutil

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LTPModelFixer:
    """LTP 模型修复器"""
    
    def __init__(self):
        self.model_dir = Path("small-main")
        self.temp_dir = Path("ltp_temp_fix")
        
        # LTP small 模型下载链接
        self.model_url = "https://huggingface.co/LTP/small/resolve/main/pytorch_model.bin"
        self.config_urls = {
            "config.json": "https://huggingface.co/LTP/small/resolve/main/config.json",
            "vocab.txt": "https://huggingface.co/LTP/small/resolve/main/vocab.txt",
            "tokenizer_config.json": "https://huggingface.co/LTP/small/resolve/main/tokenizer_config.json",
            "special_tokens_map.json": "https://huggingface.co/LTP/small/resolve/main/special_tokens_map.json"
        }
    
    def check_current_model(self):
        """检查当前模型状态"""
        model_path = self.model_dir / "pytorch_model.bin"
        
        if not model_path.exists():
            logger.error("模型文件不存在")
            return False
        
        size = os.path.getsize(model_path)
        logger.info(f"当前模型文件大小: {size} 字节")
        
        if size < 1000000:  # 小于 1MB 认为是损坏的
            logger.error("模型文件已损坏，需要重新下载")
            return False
        else:
            logger.info("模型文件正常")
            return True
    
    def create_directories(self):
        """创建必要的目录"""
        self.model_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        logger.info("目录创建完成")
    
    def download_model_file(self):
        """下载模型文件"""
        model_path = self.temp_dir / "pytorch_model.bin"
        
        try:
            logger.info("开始下载 LTP small 模型...")
            response = requests.get(self.model_url, stream=True, timeout=300)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            sys.stdout.write(f"\r下载进度: {progress:.1f}% ({downloaded}/{total_size} bytes)")
                            sys.stdout.flush()
            
            print()  # 换行
            logger.info(f"模型下载完成: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"下载失败: {e}")
            return False
    
    def download_config_files(self):
        """下载配置文件"""
        for filename, url in self.config_urls.items():
            target_path = self.temp_dir / filename
            try:
                logger.info(f"下载 {filename}...")
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                
                with open(target_path, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"✓ {filename} 下载完成")
                
            except Exception as e:
                logger.warning(f"下载 {filename} 失败: {e}")
    
    def verify_downloaded_model(self):
        """验证下载的模型文件"""
        model_path = self.temp_dir / "pytorch_model.bin"
        
        try:
            size = os.path.getsize(model_path)
            logger.info(f"下载的模型文件大小: {size} 字节")
            
            if size < 1000000:
                logger.error("下载的模型文件仍然太小，可能下载失败")
                return False
            
            logger.info("模型文件验证通过")
            return True
            
        except Exception as e:
            logger.error(f"验证失败: {e}")
            return False
    
    def replace_model_files(self):
        """替换模型文件"""
        try:
            # 备份原有文件
            if (self.model_dir / "pytorch_model.bin").exists():
                backup_dir = Path("small-main-backup")
                backup_dir.mkdir(exist_ok=True)
                
                for file in self.model_dir.glob("*"):
                    if file.is_file():
                        shutil.copy2(file, backup_dir / file.name)
                
                logger.info(f"原有文件已备份到: {backup_dir}")
            
            # 复制新文件
            for file in self.temp_dir.glob("*"):
                if file.is_file():
                    shutil.copy2(file, self.model_dir / file.name)
            
            logger.info("新模型文件已安装")
            return True
            
        except Exception as e:
            logger.error(f"文件替换失败: {e}")
            return False
    
    def cleanup(self):
        """清理临时文件"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info("临时文件已清理")
        except Exception as e:
            logger.warning(f"清理临时文件失败: {e}")
    
    def run(self):
        """运行修复流程"""
        logger.info("开始 LTP 模型修复...")
        
        try:
            # 检查当前模型状态
            if self.check_current_model():
                logger.info("模型文件正常，无需修复")
                return True
            
            # 创建目录
            self.create_directories()
            
            # 下载模型文件
            if not self.download_model_file():
                logger.error("模型下载失败")
                return False
            
            # 下载配置文件
            self.download_config_files()
            
            # 验证下载的模型
            if not self.verify_downloaded_model():
                logger.error("下载的模型验证失败")
                return False
            
            # 替换文件
            if not self.replace_model_files():
                logger.error("文件替换失败")
                return False
            
            # 清理
            self.cleanup()
            
            logger.info("✓ LTP 模型修复完成！")
            logger.info(f"模型目录: {self.model_dir.absolute()}")
            return True
            
        except Exception as e:
            logger.error(f"修复过程出错: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """主函数"""
    fixer = LTPModelFixer()
    
    if fixer.run():
        print("\n" + "="*50)
        print("✓ LTP 模型修复成功！")
        print("="*50)
        print("接下来请运行测试验证 LTP 是否正常工作:")
        print("python 知识图谱构建/ltp_fix.py")
    else:
        print("\n[错误] LTP 模型修复失败")
        print("请检查网络连接后重试")

if __name__ == "__main__":
    main()