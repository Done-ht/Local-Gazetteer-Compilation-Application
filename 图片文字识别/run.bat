@echo off
chcp 65001 >nul
REM ============================================================
REM  server-paddle �����ű���ʹ����Ŀ�� .venv ���⻷����
REM  ��������� 0.0.0.0:8000���Զ����������������ҳ
REM ============================================================
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    echo [����] ʹ����Ŀ���⻷��: %cd%\.venv
    ".venv\Scripts\python.exe" main.py
) else (
    echo [����] δ�ҵ� .venv ���⻷�������˵�ϵͳ python
    echo        ���������� setup.bat �������⻷������װ����
    python main.py
)
pause
