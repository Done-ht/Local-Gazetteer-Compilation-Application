"""存储区管理：分 zone、五千万字上限、chunk JSON 写入。

每个 zone 目录结构：
    zone_xxx/
        _zone.json          # zone 元数据（字符计数、chunk 计数等）
        _transaction.json   # 事务状态（仅在事务进行时存在）
        _staging/           # 暂存区（事务写入时使用）
        chunks/
            chunk_000001.json
            ...
        _index/
            postings/       # 倒排索引分桶文件
            _manifest.json

本模块仅负责存储布局与 chunk 文件读写，不关心事务/去重/索引语义。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Iterator, Optional

# 每个 zone 的字符上限：五千万字
ZONE_CHAR_LIMIT = 50_000_000

# 全局元数据文件
GLOBAL_META_FILE = "_global.json"
# 全局去重索引文件
DEDUP_INDEX_FILE = "_dedup_index.json"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ---- Windows 文件占用重试工具 ----
# 杀软扫描、Windows 索引服务、并发进程可能短暂占用 JSON 文件，
# 导致 os.replace / os.remove 立即抛 WinError 5（拒绝访问）。
# 这里做指数退避重试，避免假性失败导致导入被误回滚或 chunk 写入丢失。

_FILE_OP_MAX_RETRIES = 5
_FILE_OP_BASE_DELAY = 0.1  # 秒，指数退避基数


def _replace_with_retry(src: str, dst: str) -> None:
    """os.replace 带重试：遇到 PermissionError/OSError 时退避后重试。"""
    last_err: Optional[Exception] = None
    for attempt in range(_FILE_OP_MAX_RETRIES):
        try:
            os.replace(src, dst)
            return
        except (PermissionError, OSError) as e:
            last_err = e
            time.sleep(_FILE_OP_BASE_DELAY * (2 ** attempt))
    raise last_err  # type: ignore[misc]


def _remove_with_retry(path: str) -> None:
    """os.remove 带重试：遇到 PermissionError/OSError 时退避后重试。"""
    last_err: Optional[Exception] = None
    for attempt in range(_FILE_OP_MAX_RETRIES):
        try:
            os.remove(path)
            return
        except (PermissionError, OSError) as e:
            last_err = e
            time.sleep(_FILE_OP_BASE_DELAY * (2 ** attempt))
    raise last_err  # type: ignore[misc]


def _read_json(path: str, default=None):
    if not os.path.isfile(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, data) -> None:
    _ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    _replace_with_retry(tmp, path)


@dataclass
class ZoneMeta:
    zone_id: str
    char_count: int = 0           # 已提交字符数
    chunk_count: int = 0          # 已提交 chunk 数
    source_count: int = 0         # 已导入源文件数
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ZoneMeta":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})


class Zone:
    """单个数据存储区。"""

    def __init__(self, root: str, zone_id: str):
        self.root = root
        self.zone_id = zone_id
        self.path = os.path.join(root, zone_id)
        self.meta_path = os.path.join(self.path, "_zone.json")
        self.chunks_dir = os.path.join(self.path, "chunks")
        self.staging_dir = os.path.join(self.path, "_staging")
        self.index_dir = os.path.join(self.path, "_index")
        self.postings_dir = os.path.join(self.path, "_index", "postings")
        self.tx_path = os.path.join(self.path, "_transaction.json")
        self._meta: Optional[ZoneMeta] = None

    @property
    def meta(self) -> ZoneMeta:
        if self._meta is None:
            d = _read_json(self.meta_path, default=None)
            if d is None:
                self._meta = ZoneMeta(zone_id=self.zone_id)
            else:
                self._meta = ZoneMeta.from_dict(d)
        return self._meta

    def save_meta(self) -> None:
        _write_json(self.meta_path, self.meta.to_dict())

    def ensure_dirs(self) -> None:
        _ensure_dir(self.path)
        _ensure_dir(self.chunks_dir)
        _ensure_dir(self.staging_dir)
        _ensure_dir(self.postings_dir)

    def remaining(self) -> int:
        return ZONE_CHAR_LIMIT - self.meta.char_count

    def chunk_path(self, seq: int, staging: bool = False) -> str:
        name = f"chunk_{seq:06d}.json"
        base = self.staging_dir if staging else self.chunks_dir
        return os.path.join(base, name)

    def iter_chunk_files(self) -> Iterator[str]:
        if not os.path.isdir(self.chunks_dir):
            return
        for name in sorted(os.listdir(self.chunks_dir)):
            if name.startswith("chunk_") and name.endswith(".json"):
                yield os.path.join(self.chunks_dir, name)

    def read_chunk(self, seq: int) -> dict:
        path = self.chunk_path(seq)
        return _read_json(path)

    def write_chunk(self, chunk: dict, staging: bool = True) -> str:
        """写入 chunk 文件。默认写到暂存区，提交时再 move 到正式目录。"""
        seq = chunk["chunk_seq"]
        path = self.chunk_path(seq, staging=staging)
        _write_json(path, chunk)
        return path

    def commit_staging(self) -> int:
        """将暂存区的 chunk 文件移动到正式目录，返回移动的文件数。

        同时移动 .json（chunk 数据）和 .idx（per-chunk 索引）。
        """
        _ensure_dir(self.chunks_dir)
        moved = 0
        if not os.path.isdir(self.staging_dir):
            return 0
        for name in sorted(os.listdir(self.staging_dir)):
            if not name.startswith("chunk_"):
                continue
            if not (name.endswith(".json") or name.endswith(".idx")):
                continue
            src = os.path.join(self.staging_dir, name)
            dst = os.path.join(self.chunks_dir, name)
            if os.path.isfile(src):
                _replace_with_retry(src, dst)
                moved += 1
        return moved

    def clean_staging(self) -> None:
        """清空暂存区（回滚时调用）。"""
        if not os.path.isdir(self.staging_dir):
            return
        for name in os.listdir(self.staging_dir):
            p = os.path.join(self.staging_dir, name)
            if os.path.isfile(p):
                _remove_with_retry(p)


class ZoneManager:
    """管理所有 zone 的创建与选取。"""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        _ensure_dir(self.root)
        self.global_meta_path = os.path.join(self.root, GLOBAL_META_FILE)

    @property
    def dedup_path(self) -> str:
        return os.path.join(self.root, DEDUP_INDEX_FILE)

    def list_zones(self) -> list[Zone]:
        zones = []
        for name in sorted(os.listdir(self.root)):
            full = os.path.join(self.root, name)
            if os.path.isdir(full) and name.startswith("zone_"):
                zones.append(Zone(self.root, name))
        return zones

    def get_zone(self, zone_id: str) -> Zone:
        return Zone(self.root, zone_id)

    def _next_zone_id(self) -> str:
        zones = self.list_zones()
        if not zones:
            return "zone_001"
        last = zones[-1].zone_id
        try:
            n = int(last.split("_")[1]) + 1
        except Exception:
            n = len(zones) + 1
        return f"zone_{n:03d}"

    def create_zone(self) -> Zone:
        zone = Zone(self.root, self._next_zone_id())
        zone.ensure_dirs()
        zone.save_meta()
        return zone

    def select_zone_for_chars(self, need_chars: int) -> Zone:
        """选取一个能容纳 need_chars 字符的 zone；都不够则新建。"""
        for z in self.list_zones():
            if z.remaining() >= need_chars:
                return z
        # 现有 zone 都不够，新建
        z = self.create_zone()
        return z

    def stats(self) -> dict:
        zones = self.list_zones()
        total_chars = 0
        total_chunks = 0
        total_sources = 0
        zone_info = []
        for z in zones:
            m = z.meta
            total_chars += m.char_count
            total_chunks += m.chunk_count
            total_sources += m.source_count
            zone_info.append({
                "zone_id": m.zone_id,
                "char_count": m.char_count,
                "chunk_count": m.chunk_count,
                "source_count": m.source_count,
                "remaining": z.remaining(),
            })
        return {
            "store_root": self.root,
            "zone_count": len(zones),
            "total_chars": total_chars,
            "total_chunks": total_chunks,
            "total_sources": total_sources,
            "zones": zone_info,
        }
