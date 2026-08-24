"""去重索引：基于源文件 SHA256 的全局去重。

对应需求 8 表格第 2 行：
- 导入前计算源文件 SHA256
- 全局去重索引 _dedup_index.json
- 命中 -> 跳过 + 显示已导入信息
- --force 参数可绕过去重
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from typing import Optional, Dict


def compute_file_sha256(path: str, buf_size: int = 1 << 20) -> str:
    """流式计算文件 SHA256，内存只保留 buf_size 缓冲。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(buf_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


class DedupIndex:
    """全局去重索引，存储在 store_root/_dedup_index.json。

    结构：
    {
        "<sha256>": {
            "file_name": "...",
            "file_path": "...",
            "zone_id": "zone_001",
            "imported_at": "2026-07-23T...",
            "char_count": 12345
        },
        ...
    }
    """

    def __init__(self, store_root: str):
        self.store_root = store_root
        self.path = os.path.join(store_root, "_dedup_index.json")
        self._data: Optional[Dict] = None
        # 库级去重文件并发写互斥：多用户同时导入同一/不同库时,
        # 防止“读-改-写”交错导致去重索引丢失更新（读旧写旧）。
        self._lock = threading.RLock()

    def _load(self) -> Dict:
        with self._lock:
            if self._data is None:
                if os.path.isfile(self.path):
                    with open(self.path, "r", encoding="utf-8") as f:
                        self._data = json.load(f)
                else:
                    self._data = {}
            return self._data

    def _save(self) -> None:
        if self._data is None:
            return
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def check(self, sha256: str) -> Optional[dict]:
        """返回已导入信息，未导入返回 None。"""
        with self._lock:
            data = self._load()
            return data.get(sha256)

    def add(
        self,
        sha256: str,
        file_name: str,
        file_path: str,
        zone_id: str,
        char_count: int,
    ) -> None:
        with self._lock:
            data = self._load()
            data[sha256] = {
                "file_name": file_name,
                "file_path": file_path,
                "zone_id": zone_id,
                "imported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "char_count": char_count,
            }
            self._save()

    def remove(self, sha256: str) -> bool:
        with self._lock:
            data = self._load()
            if sha256 in data:
                del data[sha256]
                self._save()
                return True
            return False

    def all(self) -> Dict:
        with self._lock:
            return self._load()

    def count(self) -> int:
        with self._lock:
            return len(self._load())
