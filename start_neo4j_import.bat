@echo off
chcp 65001 >nul
echo ========================================
echo   七年级上册知识图谱数据导入工具
echo ========================================
echo.

REM 检查Neo4j导入器文件是否存在
if not exist "neo4j_importer.py" (
    echo 错误: neo4j_importer.py 文件不存在
    pause
    exit /b 1
)

REM 检查输出目录是否存在
if not exist "output\grade7\entities.json" (
    echo 错误: 七年级上册处理结果不存在
    echo 请先运行 grade7_builder.py 处理七年级上册内容
    pause
    exit /b 1
)

echo 正在启动Neo4j数据导入工具...
echo.
echo 请确保Neo4j服务正在运行
echo Neo4j配置信息:
echo   默认URI: bolt://localhost:7687
echo   默认用户: neo4j
echo   默认密码: BMtanwang7546
echo.
echo 如果需要修改配置，请编辑 config.py 文件
echo.

REM 运行Neo4j导入器
python neo4j_importer.py

echo.
echo ========================================
echo   导入完成，按任意键退出...
echo ========================================
pause