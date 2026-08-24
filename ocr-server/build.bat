@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================================
echo  server-paddle 打包脚本（服务端版）
echo ============================================================

REM 检查 .venv
if not exist ".venv\Scripts\python.exe" (
    echo [错误] 未找到 .venv 虚拟环境，请先运行 setup.bat
    pause
    exit /b 1
)

REM 清理旧产物
echo [1/3] 清理旧产物...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"

REM 打包
echo [2/3] 开始打包（首次约 5-10 分钟，请耐心等待）...
".venv\Scripts\python.exe" -m PyInstaller main.spec --noconfirm
if errorlevel 1 (
    echo [失败] 打包出错
    pause
    exit /b 1
)

REM 完成
echo [3/3] 打包完成
echo ============================================================
echo  输出目录: %cd%\dist\server-paddle\
echo  入口程序: %cd%\dist\server-paddle\server-paddle.exe
echo ============================================================
echo  可将 dist\server-paddle 整个文件夹复制给用户使用。
echo  双击 server-paddle.exe 启动（控制台显示局域网访问地址）。
echo  首次识别会加载 PaddleOCR 模型，约 30-60 秒。
echo ============================================================
pause
