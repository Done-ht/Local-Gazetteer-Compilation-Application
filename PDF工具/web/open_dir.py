"""打开输出目录（本机模式专用）。

任务完成后，前端调用此模块在系统文件管理器中打开输出目录，
对应桌面版"完成后弹窗询问是否打开输出目录"的体验。
"""
import os
import sys
import subprocess


def open_directory(path: str) -> bool:
    """在系统文件管理器中打开指定目录，成功返回 True。

    Args:
        path: 要打开的目录绝对路径

    Returns:
        bool: 是否成功唤起文件管理器
    """
    if not path or not os.path.isdir(path):
        return False
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # Windows 资源管理器
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])  # macOS Finder
        else:
            subprocess.Popen(["xdg-open", path])  # Linux
        return True
    except Exception:
        return False
