@echo off
chcp 65001 >nul
echo 正在启动七年级上册知识图谱构建器...
cd /d %~dp0
python grade7_builder.py
pause