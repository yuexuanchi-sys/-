#!/usr/bin/env python3
"""
数学知识图谱系统启动脚本
确保使用兼容的依赖版本启动应用
"""

import sys
import subprocess
import os

def check_dependencies():
    """检查并安装正确的依赖版本"""
    try:
        import flask
        import werkzeug
        
        # 检查当前安装的版本
        flask_version = flask.__version__
        werkzeug_version = werkzeug.__version__
        
        print(f"当前Flask版本: {flask_version}")
        print(f"当前Werkzeug版本: {werkzeug_version}")
        
        # 检查是否是需要兼容的版本
        if flask_version.startswith('3.') or werkzeug_version.startswith('3.'):
            print("检测到不兼容的版本，正在安装兼容版本...")
            
            # 安装兼容版本
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "Flask==2.3.0", "Werkzeug==2.3.0"
            ])
            
            print("依赖更新完成，请重新运行此脚本")
            return False
            
        return True
        
    except ImportError as e:
        print(f"依赖未安装: {e}")
        print("正在安装依赖...")
        
        # 安装依赖
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "Flask==2.3.0", "Werkzeug==2.3.0", "neo4j==5.20.0"
        ])
        
        print("依赖安装完成，请重新运行此脚本")
        return False

def main():
    """主启动函数"""
    print("=== 数学知识图谱系统启动器 ===")
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 导入并启动应用
    try:
        from app import app
        print("应用启动成功！")
        print("访问地址: http://localhost:5000")
        print("按 Ctrl+C 停止服务器")
        
        # 启动Flask开发服务器
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except ImportError as e:
        print(f"导入应用失败: {e}")
        print("请确保 app.py 文件存在且语法正确")
    except Exception as e:
        print(f"启动应用时出错: {e}")

if __name__ == '__main__':
    main()