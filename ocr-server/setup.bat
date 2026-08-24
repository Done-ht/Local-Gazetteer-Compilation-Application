@echo off
chcp 65001 >nul
REM ============================================================
REM  server-paddle 环境搭建脚本
REM  使用 Python 3.10+ 创建虚拟环境并安装全部依赖
REM  服务端版：FastAPI + Uvicorn + 本地 PaddleOCR（纯 CPU）
REM ============================================================
setlocal
cd /d "%~dp0"

REM 查找 Python 3.10+：优先 py 启动器，其次 PATH 中的 python
set "PYTHON="
py -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1 && set "PYTHON=py"
if not defined PYTHON (
    python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1 && set "PYTHON=python"
)
if not defined PYTHON (
    echo [错误] 未找到 Python 3.10 及以上版本，请先安装并加入 PATH
    pause
    exit /b 1
)

echo [1/3] 使用以下解释器创建虚拟环境 .venv：
%PYTHON% -c "import sys; print('      ', sys.executable)"
%PYTHON% -m venv .venv
if errorlevel 1 (
    echo [错误] 创建虚拟环境失败
    pause
    exit /b 1
)

echo [2/3] 升级 pip ...
".venv\Scripts\python.exe" -m pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/

echo [3/3] 按 requirements.txt 安装依赖（阿里云源）...
".venv\Scripts\python.exe" -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
if errorlevel 1 (
    echo [错误] 依赖安装失败，请检查网络后重试
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  环境搭建完成！双击 run.bat 启动服务
echo  启动后控制台会显示局域网访问地址与二维码
echo ============================================================
pause
