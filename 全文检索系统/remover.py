"""数据删除器：按扩展名/文件名/SHA 删除已导入的数据。

删除流程：
1. 遍历所有 zone 的 chunk，匹配要删除的源文件
2. 删除对应的 chunk JSON（及残留 .idx）
3. 从去重索引删除记录
4. 重算受影响 zone 的元数据（char_count/chunk_count/source_count）
5. 全量重建受影响 zone 的倒排索引

本模块与 importer.py 对称，互不依赖。
"""
from __future__ import annotations

import json
import os
from typing import List

from storage import Zone, ZoneManager
from dedup import DedupIndex


def _remove_source_copy(mgr: ZoneManager, file_path_rel: str) -> bool:
    """删除 _sources 目录下的源文件副本。

    file_path_rel 是相对 BASE_DIR 的路径（即 dedup 记录里的 file_path）。
    """
    if not file_path_rel:
        return False
    # BASE_DIR 是 mgr.root 的父目录
    base_dir = os.path.dirname(mgr.root)
    abs_path = os.path.join(base_dir, file_path_rel)
    if os.path.isfile(abs_path):
        try:
            os.remove(abs_path)
            return True
        except OSError:
            return False
    return False


def _iter_chunks(zone: Zone):
    """遍历 zone 的所有 chunk JSON，返回 (path, chunk_dict)。"""
    if not os.path.isdir(zone.chunks_dir):
        return
    for name in sorted(os.listdir(zone.chunks_dir)):
        if not (name.startswith("chunk_") and name.endswith(".json")):
            continue
        path = os.path.join(zone.chunks_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                yield path, json.load(f)
        except (json.JSONDecodeError, OSError):
            continue


def _recompute_zone_meta(zone: Zone) -> None:
    """从现有 chunk 文件重算 zone 元数据。"""
    total_chars = 0
    total_chunks = 0
    seen_shas = set()
    for _path, chunk in _iter_chunks(zone):
        total_chars += chunk.get("text_length", 0)
        total_chunks += 1
        sha = chunk.get("source", {}).get("source_sha256", "")
        if sha:
            seen_shas.add(sha)
    m = zone.meta
    m.char_count = total_chars
    m.chunk_count = total_chunks
    m.source_count = len(seen_shas)
    zone.save_meta()


def _renumber_chunks(zone: Zone) -> int:
    """删除后对剩余 chunk 重新编号（从 1 开始），保持存储紧凑与连续性。

    更新 chunk_id / chunk_seq 并重命名文件。返回重编号的文件数。
    """
    chunks: List[tuple] = []  # (old_path, chunk_dict)
    for path, chunk in _iter_chunks(zone):
        chunks.append((path, chunk))
    if not chunks:
        return 0
    # 按原 seq 排序
    chunks.sort(key=lambda x: x[1].get("chunk_seq", 0))

    new_seq = 0
    renamed = 0
    for path, chunk in chunks:
        new_seq += 1
        old_seq = chunk.get("chunk_seq", 0)
        if old_seq == new_seq:
            continue
        # 更新 chunk_id 与 chunk_seq
        chunk["chunk_seq"] = new_seq
        chunk["chunk_id"] = f"{zone.zone_id}/chunk_{new_seq:06d}"
        new_path = os.path.join(zone.chunks_dir, f"chunk_{new_seq:06d}.json")
        # 先写新文件，再删旧文件（避免同名覆盖问题）
        tmp = new_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(chunk, f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, new_path)
        if os.path.abspath(path) != os.path.abspath(new_path):
            os.remove(path)
        renamed += 1
    return renamed


def remove_by_ext(mgr: ZoneManager, ext: str) -> dict:
    """删除所有指定扩展名的源文件对应的 chunk。

    ext 形如 ".txt"（不带点也会自动补）。
    返回统计 dict。
    """
    if not ext.startswith("."):
        ext = "." + ext
    ext_lower = ext.lower()

    removed_chunks = 0
    removed_chars = 0
    removed_shas = set()
    zones_affected = set()

    for zone in mgr.list_zones():
        to_delete: List[tuple] = []  # (chunk_path, sha, text_length)
        for path, chunk in _iter_chunks(zone):
            file_name = chunk.get("source", {}).get("file_name", "")
            name_ext = os.path.splitext(file_name)[1].lower()
            if name_ext == ext_lower:
                to_delete.append(
                    (path, chunk.get("source", {}).get("source_sha256", ""),
                     chunk.get("text_length", 0))
                )

        if not to_delete:
            continue

        for path, sha, tlen in to_delete:
            if os.path.isfile(path):
                os.remove(path)
                removed_chunks += 1
                removed_chars += tlen
            idx_path = os.path.splitext(path)[0] + ".idx"
            if os.path.isfile(idx_path):
                os.remove(idx_path)
            if sha:
                removed_shas.add(sha)

        zones_affected.add(zone.zone_id)
        _renumber_chunks(zone)
        _recompute_zone_meta(zone)

    # 从去重索引删除（先读出 file_path 删除 _sources 副本）
    dedup = DedupIndex(mgr.root)
    removed_sources = 0
    for sha in removed_shas:
        if not sha:
            continue
        info = dedup.check(sha)
        if info:
            _remove_source_copy(mgr, info.get("file_path", ""))
        if dedup.remove(sha):
            removed_sources += 1

    # 全量重建受影响 zone 的倒排索引
    from indexer import ZoneIndex
    for zone in mgr.list_zones():
        if zone.zone_id in zones_affected:
            zi = ZoneIndex(zone.index_dir)
            zi.rebuild(zone.chunks_dir, zone.zone_id)

    return {
        "ext": ext_lower,
        "removed_chunks": removed_chunks,
        "removed_sources": removed_sources,
        "removed_chars": removed_chars,
        "zones_affected": sorted(zones_affected),
    }


def remove_by_sha(mgr: ZoneManager, sha256: str) -> dict:
    """按源文件 SHA256 删除对应的所有 chunk。"""
    removed_chunks = 0
    removed_chars = 0
    zones_affected = set()

    for zone in mgr.list_zones():
        to_delete: List[tuple] = []
        for path, chunk in _iter_chunks(zone):
            if chunk.get("source", {}).get("source_sha256", "") == sha256:
                to_delete.append((path, chunk.get("text_length", 0)))

        if not to_delete:
            continue

        for path, tlen in to_delete:
            if os.path.isfile(path):
                os.remove(path)
                removed_chunks += 1
                removed_chars += tlen
            idx_path = os.path.splitext(path)[0] + ".idx"
            if os.path.isfile(idx_path):
                os.remove(idx_path)

        zones_affected.add(zone.zone_id)
        _renumber_chunks(zone)
        _recompute_zone_meta(zone)

    # 从去重索引删除（先读出 file_path 删除 _sources 副本）
    dedup = DedupIndex(mgr.root)
    info = dedup.check(sha256)
    if info:
        _remove_source_copy(mgr, info.get("file_path", ""))
    dedup.remove(sha256)

    from indexer import ZoneIndex
    for zone in mgr.list_zones():
        if zone.zone_id in zones_affected:
            zi = ZoneIndex(zone.index_dir)
            zi.rebuild(zone.chunks_dir, zone.zone_id)

    return {
        "sha256": sha256,
        "removed_chunks": removed_chunks,
        "removed_chars": removed_chars,
        "zones_affected": sorted(zones_affected),
    }
