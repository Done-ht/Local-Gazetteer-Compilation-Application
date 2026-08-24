# -*- coding: utf-8 -*-
"""滑动式纠错进度持久化。

进度文件放在源文档同目录，命名 _stem.docproof.json（下划线开头排在顶部）。
保存内容：当前页索引、页码顺序、被修改页的文本快照、错误面板状态
（已提交修正 / 当前页未确认 / 当前页已确认）。
通过 source_mtime 校验源文档未被外部修改，避免恢复到失效的偏移。
"""
import json
import os
from datetime import datetime

PROGRESS_VERSION = 1


def progress_path(source_path: str) -> str:
    """进度文件路径：源文档同目录，_stem.docproof.json"""
    d = os.path.dirname(os.path.abspath(source_path))
    stem, _ = os.path.splitext(os.path.basename(source_path))
    return os.path.join(d, f"_{stem}.docproof.json")


def save_progress(source_path: str, data: dict) -> str:
    """保存进度到文件，返回文件路径"""
    path = progress_path(source_path)
    payload = dict(data)
    payload["version"] = PROGRESS_VERSION
    payload["source_path"] = source_path
    try:
        payload["source_mtime"] = os.path.getmtime(source_path)
    except OSError:
        payload["source_mtime"] = None
    payload["saved_at"] = datetime.now().isoformat(timespec="seconds")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def load_progress(source_path: str) -> dict:
    """加载进度文件；不存在 / 损坏 / 源文档已改动 时返回 None"""
    path = progress_path(source_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("version") != PROGRESS_VERSION:
        return None
    # 校验源文档未被外部修改，避免恢复到失效偏移
    try:
        cur_mtime = os.path.getmtime(source_path)
    except OSError:
        return None
    if data.get("source_mtime") != cur_mtime:
        return None
    return data


def delete_progress(source_path: str):
    """删除进度文件（已导出或用户放弃时调用）"""
    path = progress_path(source_path)
    try:
        os.remove(path)
    except OSError:
        pass
