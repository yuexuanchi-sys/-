@echo off
chcp 65001 >nul
echo ========================================
echo   知识图谱构建系统 - 智能启动脚本
echo ========================================
echo.

echo 检查Neo4j连接状态...
python simple_neo4j_test.py

echo.
echo 如果显示连接失败，请:
echo 1. 确保Neo4j Desktop已安装并运行
echo 2. 在Neo4j Desktop中启动数据库
echo 3. 访问 http://localhost:7474 确认数据库运行
echo.
echo 即使Neo4j连接失败，程序也会在离线模式下运行
echo.

echo 正在启动知识图谱构建程序...
python knowledge_graph_builder.py

pause