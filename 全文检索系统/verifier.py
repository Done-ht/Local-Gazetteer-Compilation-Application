"""校验器：数据完整性校验 + 修复建议。

校验维度：
1. chunk 内容哈希（content_hash）完整性
2. zone 偏移连续性
3. 源文件 SHA256 一致性
4. 标签覆盖率（缺失 tags 的 chunk）
5. 倒排索引一致性（索引与 chunk 数量是否匹配）
6. 向量索引状态（是否存在、向量数是否匹配）
7. 元数据统计覆盖率（缺失 stats 的 chunk）

每项检测附带修复建议，支持逐项确认或一键修复。
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Optional, Dict, Any, List

from storage import Zone, ZoneManager
from dedup import compute_file_sha256


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_chunk(chunk: dict) -> tuple[bool, str]:
    """校验单个 chunk：重算 content_hash 并比对。返回 (ok, message)。"""
    text = chunk.get("text", "")
    expected = chunk.get("content_hash", "")
    actual = content_hash(text)
    if actual != expected:
        return False, f"content_hash mismatch: expected={expected} actual={actual}"
    return True, "ok"


def verify_zone_continuity(zone: Zone) -> tuple[bool, list[str]]:
    """检查 zone 内 chunk 序列偏移量连续性。

    规则：
    - chunk_seq 从 1 开始递增（跨源文件连续）
    - text_offset 按"导入批次"检查连续性：同一批次内，
      本块 text_offset - overlap_prev == 上一块 text_offset + 上一块 text_length
    - 新批次的第一个 chunk 应 text_offset=0, overlap_prev=0
    - 批次判定：源文件 SHA256 变化，或 (text_offset=0 且 overlap_prev=0)
      （后者覆盖 --force 重复导入同一文件的场景）
    """
    import json
    issues: list[str] = []
    prev_end: Optional[int] = None
    prev_seq = 0
    prev_source: Optional[str] = None
    for path in zone.iter_chunk_files():
        with open(path, "r", encoding="utf-8") as f:
            chunk = json.load(f)
        seq = chunk.get("chunk_seq", 0)
        off = chunk.get("text_offset", 0)
        length = chunk.get("text_length", 0)
        overlap = chunk.get("overlap_prev", 0)
        src = chunk.get("source", {}).get("source_sha256", "")
        # 序号连续（跨源文件）
        if seq != prev_seq + 1:
            issues.append(
                f"{zone.zone_id} chunk_seq 跳跃: 期望 {prev_seq + 1}, 实际 {seq}"
            )
        # 新批次判定：源文件变化 OR (offset=0 且 overlap=0)
        # 后者覆盖 --force 重复导入同一文件的情况
        is_new_batch = (src != prev_source) or (off == 0 and overlap == 0)
        if is_new_batch:
            # 新批次首块应 offset=0, overlap=0
            if off != 0:
                issues.append(
                    f"{zone.zone_id} chunk_{seq:06d} 新批次首块 text_offset 应为 0, 实际 {off}"
                )
            if overlap != 0:
                issues.append(
                    f"{zone.zone_id} chunk_{seq:06d} 新批次首块 overlap_prev 应为 0, 实际 {overlap}"
                )
            prev_end = None
        else:
            # 同一批次：检查偏移连续性
            if prev_end is not None:
                expected_start = prev_end - overlap
                if off != expected_start:
                    issues.append(
                        f"{zone.zone_id} chunk_{seq:06d} 偏移不连续: "
                        f"期望起始 {expected_start} (上块末 {prev_end} - overlap {overlap}), 实际 {off}"
                    )
        prev_end = off + length
        prev_seq = seq
        prev_source = src
    return (len(issues) == 0), issues


def verify_zone(zone: Zone, source_file_map: Optional[dict] = None) -> dict:
    """校验整个 zone：逐块哈希 + 偏移连续性 + （可选）源文件 SHA256。

    source_file_map: {source_sha256: abs_path} 用于源文件级校验。
    返回：
    {
        "zone_id": ...,
        "chunk_total": N,
        "chunk_ok": M,
        "chunk_bad": [...],
        "continuity_ok": bool,
        "continuity_issues": [...],
        "source_ok": bool,
        "source_issues": [...]
    }
    """
    import json
    chunk_total = 0
    chunk_ok = 0
    chunk_bad = []
    source_issues = []
    seen_source_hashes = set()

    for path in zone.iter_chunk_files():
        chunk_total += 1
        with open(path, "r", encoding="utf-8") as f:
            chunk = json.load(f)
        ok, msg = verify_chunk(chunk)
        if ok:
            chunk_ok += 1
        else:
            chunk_bad.append({"chunk": os.path.basename(path), "reason": msg})
        # 收集源 SHA256
        src = chunk.get("source", {})
        sha = src.get("source_sha256", "")
        if sha:
            seen_source_hashes.add(sha)

    cont_ok, cont_issues = verify_zone_continuity(zone)

    # 源文件级校验
    source_ok = True
    if source_file_map:
        for sha, abs_path in source_file_map.items():
            if sha in seen_source_hashes:
                if not os.path.isfile(abs_path):
                    source_ok = False
                    source_issues.append(f"源文件不存在: {abs_path}")
                    continue
                actual = compute_file_sha256(abs_path)
                if actual != sha:
                    source_ok = False
                    source_issues.append(
                        f"源文件 SHA256 不匹配: {abs_path} "
                        f"(期望 {sha}, 实际 {actual})"
                    )

    return {
        "zone_id": zone.zone_id,
        "chunk_total": chunk_total,
        "chunk_ok": chunk_ok,
        "chunk_bad": chunk_bad,
        "continuity_ok": cont_ok,
        "continuity_issues": cont_issues,
        "source_ok": source_ok,
        "source_issues": source_issues,
    }


def verify_all(mgr: ZoneManager, source_files: Optional[list[str]] = None) -> dict:
    """校验所有 zone + 生成修复建议（单次遍历优化）。

    将 chunk 哈希校验、偏移连续性、标签覆盖率、元数据统计覆盖率合并为
    单次遍历，避免对大库（数千 chunk）重复 I/O 导致前端超时。

    返回：
    {
        "zones": [zone_result, ...],
        "fix_suggestions": [
            {"type": "tags", "title": "...", "description": "...", "affected_count": N, "fixable": True, "severity": "low"},
            ...
        ],
        "summary": {"total_chunks": N, "total_zones": N, "healthy": bool}
    }
    """
    source_file_map = {}
    if source_files:
        for p in source_files:
            if os.path.isfile(p):
                sha = compute_file_sha256(p)
                source_file_map[sha] = os.path.abspath(p)

    zone_results = []
    # 跨 zone 累计的覆盖率统计
    tags_total = 0
    tags_missing = 0
    stats_total = 0
    stats_missing = 0
    total_chunks_all = 0

    for zone in mgr.list_zones():
        chunk_total = 0
        chunk_ok = 0
        chunk_bad = []
        source_issues = []
        seen_source_hashes = set()
        # 偏移连续性检查状态
        prev_end: Optional[int] = None
        prev_seq = 0
        prev_source: Optional[str] = None
        continuity_issues: list[str] = []

        for path in zone.iter_chunk_files():
            chunk_total += 1
            try:
                with open(path, "r", encoding="utf-8") as f:
                    chunk = json.load(f)
            except Exception:
                chunk_bad.append({"chunk": os.path.basename(path), "reason": "读取失败"})
                # 读取失败的 chunk 属于"数据完整性"问题，不计入 tags/stats 缺失统计
                # 否则会导致检测永远报缺失，但修复却跳过该 chunk，形成"1 项→0 项"死循环
                continue

            # 1. chunk 内容哈希校验
            ok, msg = verify_chunk(chunk)
            if ok:
                chunk_ok += 1
            else:
                chunk_bad.append({"chunk": os.path.basename(path), "reason": msg})

            # 2. 收集 source SHA256（供源文件级校验）
            src = chunk.get("source", {})
            sha = src.get("source_sha256", "")
            if sha:
                seen_source_hashes.add(sha)

            # 3. 偏移连续性检查（原 verify_zone_continuity 逻辑）
            seq = chunk.get("chunk_seq", 0)
            off = chunk.get("text_offset", 0)
            length = chunk.get("text_length", 0)
            overlap = chunk.get("overlap_prev", 0)
            src_sha = src.get("source_sha256", "")
            if seq != prev_seq + 1:
                continuity_issues.append(
                    f"{zone.zone_id} chunk_seq 跳跃: 期望 {prev_seq + 1}, 实际 {seq}"
                )
            is_new_batch = (src_sha != prev_source) or (off == 0 and overlap == 0)
            if is_new_batch:
                if off != 0:
                    continuity_issues.append(
                        f"{zone.zone_id} chunk_{seq:06d} 新批次首块 text_offset 应为 0, 实际 {off}"
                    )
                if overlap != 0:
                    continuity_issues.append(
                        f"{zone.zone_id} chunk_{seq:06d} 新批次首块 overlap_prev 应为 0, 实际 {overlap}"
                    )
                prev_end = None
            else:
                if prev_end is not None:
                    expected_start = prev_end - overlap
                    if off != expected_start:
                        continuity_issues.append(
                            f"{zone.zone_id} chunk_{seq:06d} 偏移不连续: "
                            f"期望起始 {expected_start} (上块末 {prev_end} - overlap {overlap}), 实际 {off}"
                        )
            prev_end = off + length
            prev_seq = seq
            prev_source = src_sha

            # 4. 标签覆盖率（原 _verify_tags 逻辑）
            # 用 is None 区分"未抽取"(None/缺键)与"抽取后为空"([])
            # 否则空标签 chunk 会被反复检测为缺失 → 修复写入 [] → 仍判为缺失 → 死循环
            tags_total += 1
            if chunk.get("tags") is None:
                tags_missing += 1

            # 5. 元数据统计覆盖率（原 _verify_stats 逻辑）
            stats_total += 1
            if not src.get("stats"):
                stats_missing += 1

        total_chunks_all += chunk_total

        # 源文件级校验
        source_ok = True
        if source_file_map:
            for sha, abs_path in source_file_map.items():
                if sha in seen_source_hashes:
                    if not os.path.isfile(abs_path):
                        source_ok = False
                        source_issues.append(f"源文件不存在: {abs_path}")
                        continue
                    actual = compute_file_sha256(abs_path)
                    if actual != sha:
                        source_ok = False
                        source_issues.append(
                            f"源文件 SHA256 不匹配: {abs_path} "
                            f"(期望 {sha}, 实际 {actual})"
                        )

        zone_results.append({
            "zone_id": zone.zone_id,
            "chunk_total": chunk_total,
            "chunk_ok": chunk_ok,
            "chunk_bad": chunk_bad,
            "continuity_ok": len(continuity_issues) == 0,
            "continuity_issues": continuity_issues,
            "source_ok": source_ok,
            "source_issues": source_issues,
        })

    # 倒排索引一致性（仅读 manifest，不遍历 chunk 内容）
    index_info = _verify_index(mgr)
    # 向量索引状态（仅查状态）
    semantic_info = _verify_semantic(mgr)

    # 生成修复建议
    # 始终列出所有修复项，允许用户主动执行（即使无问题）
    # affected_count=0 表示当前无问题，但仍可主动执行（如重建索引、重新生成元数据）
    # severity 根据 needs_fix 动态调整：需修复时用原级别，已就绪时降为 "info"，避免误导用户
    fix_suggestions = []
    tags_need = tags_missing > 0
    fix_suggestions.append({
        "type": "tags",
        "title": "补抽标签",
        "description": f"{tags_missing}/{tags_total} 个 chunk 缺失标签（jieba 词性过滤）" if tags_need
                       else f"全部 {tags_total} 个 chunk 标签完整（可重新生成）",
        "affected_count": tags_missing,
        "fixable": True,
        "severity": "low" if tags_need else "info",
        "needs_fix": tags_need,
    })
    index_need = not index_info["ok"]
    fix_suggestions.append({
        "type": "index",
        "title": "重建倒排索引",
        "description": index_info["description"] if index_need
                       else f"索引完整（{index_info['affected']} chunk 已索引，可全量重建）",
        "affected_count": index_info["affected"],
        "fixable": True,
        "severity": "high" if index_need else "info",
        "needs_fix": index_need,
    })
    semantic_need = not semantic_info["ok"]
    fix_suggestions.append({
        "type": "semantic",
        "title": "构建向量索引",
        "description": semantic_info["description"] if semantic_need
                       else "向量索引已就绪（可强制重建）",
        "affected_count": semantic_info["affected"],
        "fixable": True,
        "severity": "medium" if semantic_need else "info",
        "needs_fix": semantic_need,
    })
    stats_need = stats_missing > 0
    fix_suggestions.append({
        "type": "stats",
        "title": "补充元数据统计",
        "description": f"{stats_missing}/{stats_total} 个 chunk 缺失元数据统计（朝代/主题/实体密度）" if stats_need
                       else f"全部 {stats_total} 个 chunk 元数据完整（可重新生成）",
        "affected_count": stats_missing,
        "fixable": True,
        "severity": "low" if stats_need else "info",
        "needs_fix": stats_need,
    })

    # 汇总
    total_chunks = sum(z["chunk_total"] for z in zone_results)
    total_zones = len(zone_results)
    # healthy 判断：所有 chunk 完整 + 没有任何需要修复的项
    healthy = all(
        z["chunk_ok"] == z["chunk_total"] and z["continuity_ok"] and z["source_ok"]
        for z in zone_results
    ) and all(not s["needs_fix"] for s in fix_suggestions)

    return {
        "zones": zone_results,
        "fix_suggestions": fix_suggestions,
        "summary": {
            "total_chunks": total_chunks,
            "total_zones": total_zones,
            "healthy": healthy,
        },
    }


def _verify_tags(mgr: ZoneManager) -> dict:
    """检测缺失 tags 的 chunk 数量。"""
    total = 0
    missing = 0
    for zone in mgr.list_zones():
        for path in zone.iter_chunk_files():
            total += 1
            try:
                with open(path, "r", encoding="utf-8") as f:
                    chunk = json.load(f)
                if not chunk.get("tags"):
                    missing += 1
            except Exception:
                missing += 1
    return {"total": total, "missing": missing}


def _verify_index(mgr: ZoneManager) -> dict:
    """检测倒排索引与 chunk 数量是否一致。"""
    try:
        from indexer import ZoneIndex
    except ImportError:
        return {"ok": True, "description": "", "affected": 0}

    total_chunks = 0
    indexed_chunks = 0
    for zone in mgr.list_zones():
        for _ in zone.iter_chunk_files():
            total_chunks += 1
        try:
            zi = ZoneIndex.get(zone.index_dir)
            manifest = zi._load_manifest()
            indexed_chunks += len(manifest)
        except Exception:
            pass

    if total_chunks == 0:
        return {"ok": True, "description": "", "affected": 0}
    if indexed_chunks == 0:
        return {
            "ok": False,
            "description": f"倒排索引为空（{total_chunks} 个 chunk 未索引）",
            "affected": total_chunks,
        }
    if indexed_chunks < total_chunks:
        diff = total_chunks - indexed_chunks
        return {
            "ok": False,
            "description": f"索引覆盖不全：{indexed_chunks}/{total_chunks}（差 {diff}）",
            "affected": diff,
        }
    return {"ok": True, "description": "", "affected": 0}


def _verify_semantic(mgr: ZoneManager) -> dict:
    """检测向量索引状态。"""
    lib_root = mgr.root
    total_chunks = sum(z.meta.chunk_count for z in mgr.list_zones())
    try:
        from semantic_manager import get_manager
        mgr_sem = get_manager()
        if not mgr_sem.available():
            return {
                "ok": True,  # 向量索引是可选功能，不可用不算问题
                "description": "",
                "affected": 0,
            }
        status = mgr_sem.status(lib_root)
        st = status.get("status", "")
        vec_count = status.get("vector_count", 0)
        if st == "unavailable":
            return {"ok": True, "description": "", "affected": 0}
        if st in ("building", "pending"):
            return {
                "ok": True,  # 正在构建中，不算问题
                "description": "",
                "affected": 0,
            }
        if total_chunks == 0:
            # 存储区为空：没有任何 chunk，自然无向量可言，不是问题
            return {"ok": True, "description": "", "affected": 0}
        if st == "idle" or vec_count == 0:
            return {
                "ok": False,
                "description": f"向量索引未构建（{total_chunks} 个 chunk 待向量化）",
                "affected": total_chunks,
            }
        if vec_count < total_chunks:
            diff = total_chunks - vec_count
            return {
                "ok": False,
                "description": f"向量索引覆盖不全：{vec_count}/{total_chunks}（差 {diff}）",
                "affected": diff,
            }
        return {"ok": True, "description": "", "affected": 0}
    except Exception:
        return {"ok": True, "description": "", "affected": 0}


def _verify_stats(mgr: ZoneManager) -> dict:
    """检测缺失元数据统计的 chunk 数量。"""
    total = 0
    missing = 0
    for zone in mgr.list_zones():
        for path in zone.iter_chunk_files():
            total += 1
            try:
                with open(path, "r", encoding="utf-8") as f:
                    chunk = json.load(f)
                if not chunk.get("source", {}).get("stats"):
                    missing += 1
            except Exception:
                missing += 1
    return {"total": total, "missing": missing}
