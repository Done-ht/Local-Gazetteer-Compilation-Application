"""事务管理：两阶段提交 + 崩溃恢复。

对应需求 8 表格第 3 行：
- 两阶段提交：先写暂存区(staging)，再 move 到 zone
- _transaction.json 记录状态 + 提交进度
- staging 状态崩溃 -> 回滚清理
- committing 状态崩溃 -> 恢复命令继续完成
- 每次 import 自动检测并处理残留事务

状态机：
    (无事务文件)
        | begin()
        v
    staging       <-- 写 chunk + .idx 到 _staging/
        | set_commit_stats() + prepare_commit()
        v
    committing    <-- move _staging/ -> chunks/，更新元数据/去重/索引
        | finish()
        v
    (无事务文件)

事务文件 _transaction.json：
{
    "state": "staging" | "committing",
    "zone_id": "zone_001",
    "source_file": "...",
    "source_sha256": "...",
    "chunk_count": 0,        # staging 已写 chunk 数
    "committed_chunks": 0,   # committing 已 move 文件数
    "char_count": 0,         # 实际字符数（set_commit_stats 更新）
    "source_count_delta": 1, # source_count 增量
    "started_at": "...",
    "updated_at": "..."
}
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Optional

from storage import Zone, ZoneManager, _replace_with_retry, _remove_with_retry


# 重试工具已移至 storage.py（底层模块），供 storage/transaction/web_api 共用，
# 避免杀软扫描或 Windows 索引服务短暂占用 JSON 文件时 os.replace/os.remove
# 立即抛 WinError 5 导致导入被误回滚。


# ============================================================
#  zone 级事务互斥锁
# ============================================================
# 同一 zone 的 _transaction.json 不并发安全：两个导入并发操作同一 zone 时，
# 一方 os.replace/清理会与另一方交错（FileNotFoundError / WinError 5 / staging
# 被清空等）。以事务文件路径为粒度加锁：同一 zone 的事务串行执行，不同 zone
# 互不影响（同库多 zone 仍可并行）。
_zone_tx_locks: dict = {}
_zone_tx_locks_guard = threading.Lock()


def _zone_tx_lock(tx_path: str) -> threading.Lock:
    with _zone_tx_locks_guard:
        if tx_path not in _zone_tx_locks:
            _zone_tx_locks[tx_path] = threading.Lock()
        return _zone_tx_locks[tx_path]


class Transaction:
    """单个 zone 上的导入事务。"""

    def __init__(self, zone: Zone):
        self.zone = zone
        self.data: dict = {}
        self._zone_lock = _zone_tx_lock(zone.tx_path)
        self._locked = False

    # ---- 锁管理：begin 获取，finish/rollback 释放 ----
    def _acquire(self) -> None:
        if not self._locked:
            self._zone_lock.acquire()
            self._locked = True

    def _release(self) -> None:
        if self._locked:
            self._locked = False
            self._zone_lock.release()

    def _read(self) -> Optional[dict]:
        if not os.path.isfile(self.zone.tx_path):
            return None
        try:
            with open(self.zone.tx_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def _write(self) -> None:
        self.data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        tmp = self.zone.tx_path + ".tmp"
        os.makedirs(os.path.dirname(self.zone.tx_path), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        # Windows 上事务文件可能被其他进程（杀软扫描、索引服务、并发导入）短暂占用，
        # os.replace 会立即抛 WinError 5。这里做有限次重试，避免假性失败导致回滚。
        _replace_with_retry(tmp, self.zone.tx_path)

    def _clear(self) -> None:
        if os.path.isfile(self.zone.tx_path):
            _remove_with_retry(self.zone.tx_path)

    # ---- 状态查询 ----

    def current_state(self) -> Optional[str]:
        d = self._read()
        return d.get("state") if d else None

    def current(self) -> Optional[dict]:
        return self._read()

    # ---- 生命周期 ----

    def begin(
        self,
        source_file: str,
        source_sha256: str,
        char_count: int,
    ) -> None:
        """开启事务：状态 -> staging。"""
        self._acquire()
        try:
            self.zone.ensure_dirs()
            self.zone.clean_staging()
            self.data = {
                "state": "staging",
                "zone_id": self.zone.zone_id,
                "source_file": source_file,
                "source_sha256": source_sha256,
                "chunk_count": 0,
                "committed_chunks": 0,
                "char_count": char_count,
                "source_count_delta": 1,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            self._write()
        except Exception:
            self._release()
            raise

    def record_chunk(self, count: int = 1) -> None:
        """staging 阶段每写一个 chunk 调用，更新计数。"""
        d = self._read()
        if d is None:
            return
        d["chunk_count"] = d.get("chunk_count", 0) + count
        self.data = d
        self._write()

    def set_commit_stats(self, char_count: int, chunk_count: int) -> None:
        """staging 完成、prepare_commit 前调用，写入实际统计值。"""
        d = self._read()
        if d is None:
            return
        d["char_count"] = char_count
        d["chunk_count"] = chunk_count
        self.data = d
        self._write()

    def prepare_commit(self) -> None:
        """staging -> committing。"""
        d = self._read()
        if d is None:
            raise RuntimeError("无活动事务，无法 prepare_commit")
        d["state"] = "committing"
        self.data = d
        self._write()

    def commit_step(self, moved: int) -> None:
        """committing 阶段记录已移动的文件数。"""
        d = self._read()
        if d is None:
            return
        d["committed_chunks"] = d.get("committed_chunks", 0) + moved
        self.data = d
        self._write()

    def finish(self) -> None:
        """事务完成，清除事务文件。"""
        try:
            self._clear()
        finally:
            self._release()

    def rollback(self) -> None:
        """回滚：清空暂存区 + 清除事务文件。"""
        try:
            self.zone.clean_staging()
            self._clear()
        finally:
            self._release()

    # ---- 恢复 ----

    def recover(self) -> str:
        """检测并处理残留事务，返回描述字符串。

        - 无事务：'no_transaction'
        - staging：回滚清理 -> 'rolled_back'
        - committing：继续完成提交（move 剩余 + 更新元数据 + 去重 + 索引合并）-> 'committed'
        """
        self._acquire()
        try:
            d = self._read()
            if d is None:
                return "no_transaction"
            state = d.get("state")
            if state == "staging":
                self.rollback()
                return "rolled_back"
            elif state == "committing":
                return self._recover_committing(d)
            else:
                self.rollback()
                return f"rolled_back_unknown_state:{state}"
        finally:
            self._release()

    def _recover_committing(self, d: dict) -> str:
        """恢复 committing 状态：完成剩余提交步骤。"""
        # 1. move 剩余 staging 文件
        moved = self.zone.commit_staging()
        if moved > 0:
            self.commit_step(moved)

        # 2. 更新 zone 元数据（用 tx 中记录的实际值）。
        #    幂等保护：崩溃恢复路径可能重跑，已应用过则跳过，避免计数重复累加。
        if not d.get("recover_meta_applied"):
            m = self.zone.meta
            m.char_count += d.get("char_count", 0)
            m.chunk_count += d.get("chunk_count", 0)
            m.source_count += d.get("source_count_delta", 1)
            self.zone.save_meta()
            d["recover_meta_applied"] = True
            self.data = d
            self._write()  # 持久化标志，防止下次 recover 重复累加

        # 3. 写入去重索引（同 key 覆盖写，幂等）
        from dedup import DedupIndex
        dedup = DedupIndex(self.zone.root)
        dedup.add(
            d.get("source_sha256", ""),
            os.path.basename(d.get("source_file", "")),
            d.get("source_file", ""),
            self.zone.zone_id,
            d.get("char_count", 0),
        )

        # 4. 合并索引（幂等，重复运行安全）
        from indexer import ZoneIndex
        zone_index = ZoneIndex(self.zone.index_dir)
        zone_index.merge_zone_chunks(self.zone.chunks_dir, self.zone.zone_id)
        zone_index.cleanup_merged_idx(self.zone.chunks_dir)

        # 5. 完成
        self.finish()
        return "committed"


def recover_all_zones(mgr: ZoneManager) -> list[tuple[str, str]]:
    """扫描所有 zone，处理残留事务。返回 [(zone_id, result), ...]。"""
    results = []
    for zone in mgr.list_zones():
        tx = Transaction(zone)
        res = tx.recover()
        if res != "no_transaction":
            results.append((zone.zone_id, res))
    return results
