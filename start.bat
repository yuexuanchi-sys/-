@echo off
chcp 65001 >nul
echo.
echo ========================================
echo   数学知识图谱系统 - 快速启动脚本
echo ========================================
echo.

:menu
echo 请选择要执行的操作:
echo 1. 构建知识图谱
echo 2. 启动Web服务器
echo 3. 运行系统测试
echo 4. 显示系统状态
echo 5. 完整流程（构建+启动Web）
echo 6. 清空数据库
echo 7. 退出
echo.

set /p choice="请输入选项 (1-7): "

if "%choice%"=="1" goto build
if "%choice%"=="2" goto web
if "%choice%"=="3" goto test
if "%choice%"=="4" goto status
if "%choice%"=="5" goto all
if "%choice%"=="6" goto clear
if "%choice%"=="7" goto exit

echo 无效选项，请重新输入
goto menu

:build
echo 开始构建知识图谱...
python run.py build
pause
goto menu

:web
echo 启动Web服务器...
python run.py web
pause
goto menu

:test
echo 运行系统测试...
python run.py test
pause
goto menu

:status
echo 显示系统状态...
python run.py status
pause
goto menu

:all
echo 运行完整流程...
python run.py all
pause
goto menu

:clear
echo 清空数据库...
python run.py clear
pause
goto menu

:exit
echo 谢谢使用！
pause