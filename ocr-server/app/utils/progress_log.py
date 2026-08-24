"""进度日志：直接追加写入文件，绕开 logging 系统。

uvicorn.run() 会覆盖 logging 配置导致 FileHandler 失效，
因此这里用 open() 直接追加写入文件，确保日志一定落地。

每行格式：时间 | 内容 | 内存占用
psutil 不可用时仍输出内容（不含内存数值）。

日志文件大小限制：超过 10MB 时自动清空（保留最近一条作为时间标记），
避免长期运行导致文件无限增长。每页 OCR 都会写入一行，376 页文档约 50KB，
正常使用不会触发清空；但长期累积（数千页）需要限制大小。

调用方：
  - pdf_handler.py：每页 OCR 完成时记录"第 X/Y 页 耗时 Xs 识别 X 行"
  - tasks.py：任务级事件（上传/排队/开始/完成/失败/预约）
  - main.py：服务启动/关闭等里程碑事件
"""
from __future__ import annotations

import os
import time
from typing import Optional

# 进度日志文件路径（延迟初始化，放在项目根目录 main.py 同级）
_PROGRESS_LOG_PATH: Optional[str] = None

# 日志文件大小上限：超过此值自动清空（保留最近一条作为时间标记）
_MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB


def _resolve_path() -> str:
    """延迟解析日志文件路径，避免模块加载时计算。

    日志统一放到 output/log/ 目录下，与 task_dirs.LOG_DIR 保持一致。
    """
    global _PROGRESS_LOG_PATH
    if _PROGRESS_LOG_PATH is None:
        from .task_dirs import progress_log_path, LOG_DIR
        os.makedirs(LOG_DIR, exist_ok=True)
        _PROGRESS_LOG_PATH = progress_log_path()
    return _PROGRESS_LOG_PATH


def _trim_if_too_large(path: str) -> None:
    """日志文件超过大小上限时清空，保留最近一行作为时间标记。

    避免长期运行（数千页 OCR）导致文件无限增长占用磁盘。
    """
    try:
        size = os.path.getsize(path)
        if size <= _MAX_LOG_SIZE:
            return
        # 读取最后一条记录作为时间标记
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        last_line = lines[-1] if lines else ""
        # 清空文件，只保留最近一条 + 清空说明
        with open(path, "w", encoding="utf-8") as f:
            if last_line:
                f.write(last_line)
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | [日志已清空，原大小 {size / 1024 / 1024:.1f} MB]\n")
    except Exception:
        pass


def log_progress(tag: str) -> None:
    """追加一行进度日志到 ocr_progress.log。

    参数:
        tag: 事件描述，例如 "任务完成: xxx.pdf | 10 页 | 耗时 30s"
    """
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    try:
        import psutil
        mem_mb = psutil.Process().memory_info().rss / 1024 / 1024
        line = f"{ts} | {tag} | 内存: {mem_mb:.0f} MB\n"
    except Exception:
        line = f"{ts} | {tag}\n"
    try:
        path = _resolve_path()
        _trim_if_too_large(path)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass
