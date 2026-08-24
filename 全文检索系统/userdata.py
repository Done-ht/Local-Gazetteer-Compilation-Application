"""用户登录相关数据的存储目录约定。

Windows 下默认存到 <用户主目录>/biaoshifu 文件夹（如 C:\\Users\\<用户名>\\biaoshifu），
跨应用共用同一组账号密码登录。可用环境变量 BIAOSHIFU_DIR 覆盖（例如测试时指向临时目录）。

存放的文件（由 auth.py / settings.py 读写）：
    _users.json     用户表（PBKDF2 密码哈希，绝不明文落盘）
    _sessions.json  登录会话 token（7 天有效）
    _secrets.json   敏感设置（如 DeepSeek API Key，加密存储）
    _settings.json  系统设置（普通参数）

注意：目录名固定为 biaoshifu，不叫 search，避免与项目名混淆。
"""
from __future__ import annotations

import os

# 默认目录名
DEFAULT_DIR_NAME = "biaoshifu"

# 环境变量覆盖：BIAOSHIFU_DIR=<绝对路径>
ENV_VAR = "BIAOSHIFU_DIR"

_AUTH_DIR: str | None = None


def auth_base_dir() -> str:
    """解析并创建用户登录数据目录（惰性初始化，进程内缓存）。

    优先级：环境变量 BIAOSHIFU_DIR > <用户主目录>/biaoshifu。
    """
    global _AUTH_DIR
    if _AUTH_DIR is None:
        env = os.environ.get(ENV_VAR, "").strip()
        if env:
            d = os.path.abspath(os.path.expanduser(env))
        else:
            d = os.path.join(os.path.expanduser("~"), DEFAULT_DIR_NAME)
        os.makedirs(d, exist_ok=True)
        _AUTH_DIR = d
    return _AUTH_DIR
