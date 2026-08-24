"""Web 版日志模块：实时日志写入文件（5MB 轮转保留 3 份），不占用控制台。

分工：
- 控制台：仅输出启动基础信息（访问 URL 等），由 run_web.py 用 print 输出；
- 日志文件：请求日志、运行/调试信息（werkzeug 等）统一写入 logs/app.log。
"""
import os
import sys
import logging
from logging.handlers import RotatingFileHandler


def _resolve_log_dir() -> str:
    """确定日志目录。

    优先级：
    1. exe 同目录下的 logs/（打包后可写、方便随程序查看）
    2. 回退到 %LOCALAPPDATA%/pdf_tool/logs
    """
    if getattr(sys, "frozen", False):
        base_dir = os.path.dirname(sys.executable)
    else:
        # 开发模式：web/web_logger.py 上溯两级即项目根目录
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    log_dir = os.path.join(base_dir, "logs")
    try:
        os.makedirs(log_dir, exist_ok=True)
        test_file = os.path.join(log_dir, ".write_test")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("t")
        os.remove(test_file)
        return log_dir
    except (PermissionError, OSError):
        local_app_data = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        fallback_dir = os.path.join(local_app_data, "pdf_tool", "logs")
        os.makedirs(fallback_dir, exist_ok=True)
        return fallback_dir


def setup_logging() -> str:
    """初始化 Web 版文件日志（5MB 轮转），并让 werkzeug 只写文件不写控制台。

    返回：日志文件的完整路径。
    """
    log_dir = _resolve_log_dir()
    log_file = os.path.join(log_dir, "app.log")

    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding="utf-8",
        delay=True,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    # root logger：写文件，不添加控制台 handler（避免污染控制台）
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)

    # werkzeug（Flask 开发服务器）：请求/运行日志只写文件，不上控制台
    wz_logger = logging.getLogger("werkzeug")
    wz_logger.setLevel(logging.INFO)
    wz_logger.handlers = [file_handler]
    wz_logger.propagate = False

    return log_file
