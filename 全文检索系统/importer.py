"""导入编排器：串联提取、分块、索引、事务、去重、校验。

单遍流式导入流程：
1. 校验文件类型
2. 流式计算源文件 SHA256
3. 查去重索引；命中且非 --force -> 跳过并提示
4. 自动恢复目标 zone 的残留事务
5. 选取 zone（剩余空间最大；不足则新建）
6. 开启事务（staging）
7. 流式：extract -> chunk -> 写 staging JSON + per-chunk .idx
   - 边写边计字符数；若超出 zone 剩余空间 -> 回滚并报错（禁止导入）
8. prepare_commit（staging -> committing）
9. move staging -> chunks/，更新 zone 元数据，写入去重索引
10. 合并新 chunk 的 .idx 到 zone 级倒排索引（幂等）
11. finish（清除事务文件）
"""
from __future__ import annotations

import os
import shutil
import time
from typing import Optional

from userdata import auth_base_dir as _auth_base_dir

from storage import Zone, ZoneManager, ZONE_CHAR_LIMIT
from extractor import extract, supported
from chunker import Chunker, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
from indexer import build_postings, write_chunk_index, ZoneIndex
from transaction import Transaction
from dedup import DedupIndex, compute_file_sha256
from verifier import content_hash
from tagger import extract_tags


def _tagger_top_k(base_dir: Optional[str] = None) -> int:
    """从 settings 读取标签提取数量，默认 10，范围 0-30。
    设为 0 时禁用标签提取。

    base_dir: 数据根目录（--data-dir）。不传时回退代码目录（旧行为）。
    """
    try:
        from settings import SettingsStore
        store = SettingsStore(_auth_base_dir())
        v = int(store.get("tag_top_k", 10))
        return max(0, min(v, 30))
    except Exception:
        return 10


def _copy_source_to_lib(mgr: ZoneManager, file_path: str, file_name: str,
                        source_sha: str) -> str:
    """复制源文件到库工作目录 _sources/ 下，返回副本的相对路径（相对 BASE_DIR）。

    这样即使原始文件被移动/删除/临时目录被清理，溯源时仍能找到副本。
    重名处理：同名不同 SHA 的文件加序号后缀；同 SHA 则复用已有副本。
    """
    sources_dir = os.path.join(mgr.root, "_sources")
    os.makedirs(sources_dir, exist_ok=True)
    copy_name = file_name
    copy_dest = os.path.join(sources_dir, copy_name)
    counter = 1
    while os.path.exists(copy_dest):
        # 已存在且内容相同（SHA 一致），直接复用，不重复复制
        try:
            if compute_file_sha256(copy_dest) == source_sha:
                break
        except OSError:
            pass
        # 同名但内容不同，加序号
        base_name, ext_name = os.path.splitext(file_name)
        copy_name = f"{base_name}_{counter}{ext_name}"
        copy_dest = os.path.join(sources_dir, copy_name)
        counter += 1
    if not os.path.exists(copy_dest):
        shutil.copy2(file_path, copy_dest)
    # 返回相对 BASE_DIR（即 mgr.root 的父目录）的路径
    return os.path.relpath(copy_dest, os.path.dirname(mgr.root))


def _estimate_chars(file_path: str) -> int:
    """保守估计字符数上界。

    对 UTF-8 中文文本文件，1 字符最多 3/4 字节，字节数是字符数的上界。
    对 PDF 等二进制格式，文件字节数远大于实际文本字符数（含字体、图片、布局等），
    因此用随机采样页提取文本算平均字数，再乘以总页数，能准确反映
    目录/彩页/正文混合的情况。
    """
    import random

    try:
        # PDF 文件：随机采样页提取文本，按平均字数 × 总页数估计
        lower = file_path.lower()
        if lower.endswith(".pdf"):
            try:
                from pypdf import PdfReader  # type: ignore
            except ImportError:
                try:
                    from PyPDF2 import PdfReader  # type: ignore
                except ImportError:
                    PdfReader = None

            if PdfReader is not None:
                try:
                    reader = PdfReader(file_path)
                    page_count = len(reader.pages)
                    if page_count == 0:
                        return 0

                    # 随机采样：小文件全采，大文件最多采 20 页
                    # 采样数取 page_count 和 20 的较小值，至少 3 页
                    sample_size = min(page_count, 20)
                    if sample_size < 3:
                        sample_indices = list(range(page_count))
                    else:
                        # 随机采样，覆盖目录/彩页/正文等不同区域
                        sample_indices = sorted(random.sample(range(page_count), sample_size))

                    # 提取采样页文本，统计字符数
                    total_sample_chars = 0
                    sampled = 0
                    for idx in sample_indices:
                        try:
                            text = reader.pages[idx].extract_text() or ""
                            # 去除空白字符后计数（与实际入库字符数一致）
                            text = text.strip()
                            total_sample_chars += len(text)
                            sampled += 1
                        except Exception:
                            continue

                    if sampled == 0:
                        # 全部采样页提取失败，回退到字节数的 1/10
                        return os.path.getsize(file_path) // 10

                    # 图片型 PDF（扫描件）：采样页能解析但提取到 0 字
                    # 此时实际可提取文本为 0，需要 OCR 才能获得文本
                    # 返回 0 让导入流程后续检测并提示用户
                    if total_sample_chars == 0:
                        return 0

                    # 平均每页字符数 × 总页数
                    avg_chars_per_page = total_sample_chars / sampled
                    estimated = int(avg_chars_per_page * page_count)
                    # 加 20% 余量（采样可能漏掉字数多的页）
                    estimated = int(estimated * 1.2)
                    return estimated
                except Exception:
                    pass
            # pypdf 不可用或解析失败，回退到字节数的 1/10（PDF 二进制开销约 90%）
            return os.path.getsize(file_path) // 10

        if lower.endswith(".docx"):
            # docx 是 zip 压缩格式，文件字节数远大于实际文本量。
            # 统计 word/document.xml 中 <w:t> 文本节点的真实字符数，
            # 与 extractor._stream_docx 提取的段落文本量级一致（含表格略偏大，作上界安全）。
            try:
                import re
                import zipfile

                with zipfile.ZipFile(file_path) as zf:
                    if "word/document.xml" in zf.namelist():
                        data = zf.read("word/document.xml")
                        texts = re.findall(rb"<w:t[^>]*>(.*?)</w:t>", data, re.S)
                        return sum(len(re.sub(rb"[\s]", b"", t)) for t in texts)
            except Exception:
                pass
            return os.path.getsize(file_path)

        # 其他文件类型：用字节数作为上界（UTF-8 中文 1 字符最多 3/4 字节）
        return os.path.getsize(file_path)
    except OSError:
        return 0


def select_target_zone(mgr: ZoneManager, file_path: str) -> Zone:
    """选取目标 zone：剩余空间最大者；若其剩余 < 估计字符数则新建。"""
    est = _estimate_chars(file_path)
    zones = mgr.list_zones()
    best: Optional[Zone] = None
    best_rem = -1
    for z in zones:
        r = z.remaining()
        if r > best_rem:
            best_rem = r
            best = z
    if best is None or best_rem < est:
        return mgr.create_zone()
    return best


def import_file(
    mgr: ZoneManager,
    file_path: str,
    force: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    base_dir: Optional[str] = None,
    skip_merge: bool = False,
    import_root: Optional[str] = None,
) -> dict:
    """导入单个文件。返回结果摘要 dict。

    skip_merge=True 时跳过索引合并步骤，用于批量导入场景。
    调用方需在批量结束后统一调一次 ZoneIndex.merge_zone_chunks。
    import_root: 导入根目录（用户选择的文件夹路径），用于计算 relative_dir 保留目录结构元数据。
                 若提供，会计算文件相对导入根的目录路径并存入 chunk 元数据。
    """
    if not os.path.isfile(file_path):
        return {"ok": False, "error": f"文件不存在: {file_path}"}
    if not supported(file_path):
        return {"ok": False, "error": f"不支持的文件类型: {file_path}"}

    base = base_dir or os.getcwd()
    file_path = os.path.abspath(file_path)
    file_name = os.path.basename(file_path)

    # 计算相对导入根的目录路径，保留文件夹结构信息
    relative_dir = ""
    if import_root:
        import_root_abs = os.path.abspath(import_root)
        try:
            relative_dir = os.path.relpath(os.path.dirname(file_path), import_root_abs)
            # 若文件就在导入根目录下，relpath 返回 "."，置空
            if relative_dir == ".":
                relative_dir = ""
        except ValueError:
            # Windows 下跨盘符 relpath 会失败
            relative_dir = ""

    # 1. 流式计算源文件 SHA256
    source_sha = compute_file_sha256(file_path)

    # 2. 去重检查
    dedup = DedupIndex(mgr.root)
    if not force:
        existing = dedup.check(source_sha)
        if existing:
            return {
                "ok": False,
                "skipped": True,
                "reason": "重复导入",
                "source_sha256": source_sha,
                "existing": existing,
            }

    # 2.5 复制源文件到库工作目录 _sources/，确保溯源时能找到
    rel_path = _copy_source_to_lib(mgr, file_path, file_name, source_sha)

    # 3. 选取 zone
    zone = select_target_zone(mgr, file_path)
    zone.ensure_dirs()

    # 4. 恢复残留事务
    tx = Transaction(zone)
    if tx.current_state() is not None:
        tx.recover()

    # 5. 容量预检
    est = _estimate_chars(file_path)
    if est > ZONE_CHAR_LIMIT:
        return {
            "ok": False,
            "error": (
                f"文件估计字符数 {est} 超过单 zone 上限 {ZONE_CHAR_LIMIT}，禁止导入"
            ),
        }

    # 5.5 图片型 PDF 检测：采样提取到 0 字，说明是扫描件，pypdf 无法提取文本
    # 此时导入会得到空内容，应提前提示用户做 OCR 处理
    if est == 0 and file_path.lower().endswith(".pdf"):
        # 清理已复制的源文件
        try:
            src_in_lib = os.path.join(mgr.root, rel_path)
            if os.path.isfile(src_in_lib):
                os.remove(src_in_lib)
        except Exception:
            pass
        return {
            "ok": False,
            "error": (
                "此 PDF 为图片型扫描件，无法提取文本内容。"
                "请先使用 OCR 工具（如 PaddleOCR、ABBYY、Adobe Acrobat）"
                "将其转换为可检索的文本 PDF 后再导入。"
            ),
        }

    # 6. 开启事务（用估计值占位，提交时用实际值）
    tx.begin(file_path, source_sha, est)

    try:
        # 7. 流式提取 -> 分块 -> 写 staging
        chunker = Chunker(chunk_size=chunk_size, overlap=overlap)
        seq_start = zone.meta.chunk_count
        total_chars = 0
        chunks_written = 0
        chunk_seq = seq_start

        # heading 继承：extract_title 返回空时，继承前一个 chunk 的 heading
        # 解决史书类文档中，卷目标题只在某个 chunk 出现，但后续 chunk 也应该归属同一卷
        prev_heading = ""
        # 缓存 chunk_data 用于统计平滑（循环后做滑动平均再批量写文件）
        _pending_chunks = []

        for chunk in chunker.chunk(extract(file_path)):
            # 容量检查：超出剩余空间则回滚
            if total_chars + chunk["text_length"] > zone.remaining():
                tx.rollback()
                return {
                    "ok": False,
                    "error": (
                        f"导入过程中超出 zone {zone.zone_id} 剩余空间 "
                        f"(已写 {total_chars} 字, 本块 {chunk['text_length']} 字, "
                        f"剩余 {zone.remaining()} 字)，已回滚"
                    ),
                }

            chunk_seq += 1
            text = chunk["text"]
            chunk_id = f"{zone.zone_id}/chunk_{chunk_seq:06d}"
            # heading 为空时（docx/pdf/txt 无 Markdown # 标题）用算法兜底提取
            heading = chunk.get("heading", "")
            if not heading:
                try:
                    from title_extract import extract_title
                    heading = extract_title(text)
                except Exception:
                    pass  # 算法异常时保持空 heading，不阻断导入
            # heading 继承：如果 extract_title 仍然返回空，继承前一个 chunk 的 heading
            # 这解决了史书类文档中，某些 chunk 是纯表格/正文，无法提取标题，但应归属前一个有标题的 chunk
            if not heading and prev_heading:
                heading = prev_heading
            # 更新 prev_heading 供下一个 chunk 使用
            if heading:
                prev_heading = heading
            chunk_data = {
                "chunk_id": chunk_id,
                "zone_id": zone.zone_id,
                "chunk_seq": chunk_seq,
                "text": text,
                "text_offset": chunk["text_offset"],
                "text_length": chunk["text_length"],
                "overlap_prev": chunk["overlap_prev"],
                "heading": heading,
                "content_hash": content_hash(text),
                "source": {
                    "file_path": rel_path,
                    "file_name": file_name,
                    "source_sha256": source_sha,
                    "byte_offset": chunk["source_byte_offset"],
                    "byte_length": chunk["source_byte_length"],
                    "original_path": file_path,
                    "relative_dir": relative_dir,
                },
                "chunk_size": chunk_size,
                "overlap": overlap,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            # 提取 chunk 标签（jieba 词性过滤，纯 CPU）
            tag_top_k = _tagger_top_k(base)
            if tag_top_k > 0:
                try:
                    chunk_data["tags"] = extract_tags(text, top_k=tag_top_k)
                except Exception:
                    chunk_data["tags"] = []
            # 元数据统计：朝代/主题/实体密度（jieba 分词，纯 CPU）
            try:
                from metadata_stats import compute_chunk_stats
                chunk_data["source"]["stats"] = compute_chunk_stats(text)
            except Exception:
                pass
            # 缓存 chunk_data，循环后做滑动平均再批量写文件
            _pending_chunks.append((chunk_data, chunk))

            chunks_written += 1
            total_chars += chunk["text_length"]
            tx.record_chunk(1)

            # CPU 让步：每处理 8 个 chunk 主动 sleep(0) 让出时间片，
            # 避免 jieba 分词长时间 100% 占用 CPU 导致前端请求卡顿。
            # sleep(0) 不引入实际延迟，仅让调度器把 CPU 分给其他线程/进程。
            if chunks_written % 8 == 0:
                time.sleep(0)

        # 滑动平均 + 文档级汇总（同文档 chunk 间平滑，抵消单 chunk 统计波动）
        if _pending_chunks:
            try:
                from metadata_stats import sliding_window_average, summarize_document_stats
                stats_list = [cd["source"].get("stats", {}) for cd, _ in _pending_chunks]
                smoothed = sliding_window_average(stats_list, window=1)
                doc_stats = summarize_document_stats(stats_list)
                for (cd, _), s in zip(_pending_chunks, smoothed):
                    cd["source"]["stats"] = s
                    cd["source"]["doc_stats"] = doc_stats
            except Exception:
                pass

        # 批量写 chunk JSON + 索引到 staging
        for cd, chunk in _pending_chunks:
            chunk_json_path = zone.write_chunk(cd, staging=True)
            idx_path = os.path.splitext(chunk_json_path)[0] + ".idx"
            write_chunk_index(idx_path, build_postings(cd["text"]))

        # 8. 记录实际统计值 + prepare_commit
        tx.set_commit_stats(total_chars, chunks_written)
        tx.prepare_commit()

        # 9. move staging -> chunks
        moved = zone.commit_staging()
        tx.commit_step(moved)

        # 更新 zone 元数据
        m = zone.meta
        m.char_count += total_chars
        m.chunk_count += chunks_written
        m.source_count += 1
        zone.save_meta()

        # 写入去重索引
        dedup.add(source_sha, file_name, rel_path, zone.zone_id, total_chars)

        # 写入文档级元数据缓存（供统计界面展示朝代/主题/实体密度）
        if _pending_chunks and _pending_chunks[0][0]["source"].get("doc_stats"):
            try:
                import json as _json
                doc_stats_path = os.path.join(mgr.root, "_doc_stats.json")
                existing = {}
                if os.path.isfile(doc_stats_path):
                    with open(doc_stats_path, "r", encoding="utf-8") as f:
                        existing = _json.load(f)
                doc_stats = _pending_chunks[0][0]["source"]["doc_stats"]
                existing[file_name] = {
                    **doc_stats,
                    "relative_dir": relative_dir,
                    "char_count": total_chars,
                    "chunk_count": chunks_written,
                }
                with open(doc_stats_path, "w", encoding="utf-8") as f:
                    _json.dump(existing, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        # 10. 合并新 chunk 的 .idx 到 zone 级倒排索引（幂等）
        # skip_merge=True 时跳过，由批量调用方统一合并
        zone_index = ZoneIndex.get(zone.index_dir)
        if skip_merge:
            merge_stat = {"merged": 0, "skipped": 0, "batch_skipped": True}
        else:
            merge_stat = zone_index.merge_zone_chunks(zone.chunks_dir, zone.zone_id)
            # 清理已合并的 per-chunk .idx 文件，释放存储空间
            zone_index.cleanup_merged_idx(zone.chunks_dir)

        # 11. finish
        tx.finish()

        return {
            "ok": True,
            "zone_id": zone.zone_id,
            "source_sha256": source_sha,
            "file_name": file_name,
            "chunks_written": chunks_written,
            "char_count": total_chars,
            "index_merged": merge_stat.get("merged", 0),
        }
    except Exception as e:
        tx.rollback()
        return {"ok": False, "error": f"导入异常已回滚: {e}"}
