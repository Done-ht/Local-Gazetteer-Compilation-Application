"""
PDF 工具 - 日志模块

功能：
- 日志写入文件，不在控制台输出日志内容
- 5MB 轮转循环，保留 3 个历史文件（app.log / app.log.1 / app.log.2 / app.log.3）
- 捕获 stdout / stderr（包括第三方库的 print 输出与异常栈），统一写入日志文件
- 最新日志始终追加在 app.log 末尾（查看时滚动到底部即可看到最新内容）

使用方式：
    在 main.py 启动早期调用 setup_logging()，之后全局 logging 与 print 均自动写入文件。
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler


def _resolve_log_dir() -> str:
    """
    确定日志目录。

    优先级：
    1. exe 同目录下的 logs/（便携，开发与打包后均可见）
    2. 回退到 %LOCALAPPDATA%/pdf_tool/logs（exe 同目录无写权限时）
    """
    if getattr(sys, 'frozen', False):
        # PyInstaller 打包后的 exe
        base_dir = os.path.dirname(sys.executable)
    else:
        # 开发模式：项目根目录（app/utils/logger.py 上溯两级）
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

    log_dir = os.path.join(base_dir, 'logs')

    try:
        os.makedirs(log_dir, exist_ok=True)
        # 测试可写性
        test_file = os.path.join(log_dir, '.write_test')
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write('t')
        os.remove(test_file)
        return log_dir
    except (PermissionError, OSError):
        # 回退到用户应用数据目录
        local_app_data = os.environ.get('LOCALAPPDATA') or os.path.expanduser('~')
        fallback_dir = os.path.join(local_app_data, 'pdf_tool', 'logs')
        os.makedirs(fallback_dir, exist_ok=True)
        return fallback_dir


class _StdoutToLogger:
    """
    将 stdout / stderr 重定向到 logger 的流适配器。

    采用按行缓冲：只有遇到换行符才产生一条日志，避免每条 print 产生多条记录。
    """

    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self._buffer = ''

    def write(self, message: str) -> None:
        if not message:
            return
        self._buffer += message
        while '\n' in self._buffer:
            line, self._buffer = self._buffer.split('\n', 1)
            if line.strip():  # 忽略空行
                self.logger.log(self.level, line)

    def flush(self) -> None:
        if self._buffer.strip():
            self.logger.log(self.level, self._buffer)
        self._buffer = ''

    def isatty(self) -> bool:
        return False


def setup_logging() -> str:
    """
    初始化全局日志系统。

    - 配置 root logger 使用 RotatingFileHandler（5MB 轮转，保留 3 个备份）
    - 重定向 sys.stdout / sys.stderr 到 logger（捕获 print 与异常）
    - 不向控制台输出日志内容；仅在控制台输出一行日志文件路径，方便定位

    返回：
        日志文件的完整路径
    """
    log_dir = _resolve_log_dir()
    log_file = os.path.join(log_dir, 'app.log')

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    # 清除可能存在的默认 handler，避免重复输出
    root_logger.handlers.clear()

    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding='utf-8',
        delay=True,  # 延迟创建文件，避免目录创建失败时崩溃
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    root_logger.addHandler(file_handler)

    # 在重定向前，用原始 stdout 输出一行路径提示（开发模式可见，打包后无控制台也无害）
    try:
        sys.__stdout__.write(f'\n[日志] 日志文件路径: {log_file}\n')
        sys.__stdout__.write(f'[日志] 轮转策略: 单文件 5MB，保留 3 个历史文件\n')
        sys.__stdout__.write(f'[日志] 最新内容追加在文件末尾，请滚动到底部查看\n\n')
        sys.__stdout__.flush()
    except Exception:
        pass

    # 重定向 stdout / stderr，捕获所有 print 与异常栈
    sys.stdout = _StdoutToLogger(root_logger, logging.INFO)
    sys.stderr = _StdoutToLogger(root_logger, logging.ERROR)

    logging.info('===== 日志系统初始化完成 =====')
    logging.info('日志文件路径: %s', log_file)
    logging.info('轮转策略: 单文件 5MB，保留 3 个历史文件')

    return log_file
