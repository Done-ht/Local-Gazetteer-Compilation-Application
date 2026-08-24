"""任务处理流程编排器。

实现规范化的 PDF OCR 处理流程：
  1. 收到文件先拆分到 pdf_pages/，删除原件
  2. 对比 ocr_pages 和 pdf_pages 得到未完成页号数组
  3. 根据进程数分配任务，启动转化
  4. 单页 OCR 完整才保存（提前终止自动舍弃，不留空文件）
  5. 每进程 10 页后重新获取未完成数组，重新分配任务
  6. 最后循环检查未完成数组，直到为空
  7. 合并 ocr_pages 中的单页 PDF 为最终可编辑 PDF（统一叠加文字层）

进度计算基于 ocr_pages 数量 / pdf_pages 总数。
"""
from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import fitz

from ..utils import task_dirs
from ..utils.progress_log import log_progress

logger = logging.getLogger(__name__)

# 每个进程处理多少页后重新分配任务
REASSIGN_EVERY_PAGES = 10

# 任务后全量重启子进程的累计页数阈值。
# 旧逻辑：每个任务（每轮）结束后无条件 _rebuild_ocr_instances 重启所有 worker，
# 导致每个任务都从"冷启动"开始（模型加载+首推理≈60s），用户看到"页已处理完但
# 状态仍 running 数十秒"。根因：子进程 batch_size（_page_counts 池级跨任务计数）
# 本身已能在累计处理 N 页后自动重启单个 worker 释放内存，任务后的全量重启是冗余。
# naive_best_fit + eager_delete（官方内存方案）已让内存可控（实测峰值后回落、不累积），
# 故此处改为按累计页数触发：累计处理满 REBUILD_AFTER_PAGES 页才全量重启一次，
# 中间小任务复用热 worker，消除冷启动税。
REBUILD_AFTER_PAGES = 15
# 模块级累计计数器：自上次全量重启以来已处理的总页数（跨任务累计）
_pages_since_rebuild: int = 0

# 单页最大失败次数：超过此值后跳过该页（插入空白页由合并阶段处理）
# 防止因 ocr_pages 目录缺失/子进程持续 crash 等导致同一页被无限重试
MAX_PAGE_FAILS = 3

# 任务级熔断：已尝试页数达到 CIRCUIT_MIN_ATTEMPTS 且失败率超过 CIRCUIT_FAIL_RATIO 时，
# 判定为引擎级故障并中止整个任务。避免旧版"每页静默失败→整本输出空白但任务完成"的问题
# （2026-08-23 完整版年鉴 368 页全空事故的根因之一）。8 页的小任务也能覆盖
CIRCUIT_MIN_ATTEMPTS = 8
CIRCUIT_FAIL_RATIO = 0.5


def split_pdf_to_pages(
    pdf_path: str,
    task_id: str,
    progress_cb: Optional[Callable] = None,
) -> int:
    """拆分 PDF 到 pdf_pages/page_XXXX.pdf，返回总页数。

    拆分完成后删除原 PDF 文件（节省空间）。
    如果 pdf_pages 已有文件（断点续传），跳过已拆分的页。

    参数:
        pdf_path: 原 PDF 文件路径
        task_id: 任务 ID
        progress_cb: 进度回调 (current, total, message)

    返回:
        总页数
    """
    pdf_pages_dir = task_dirs.pdf_pages_dir(task_id)
    os.makedirs(pdf_pages_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    total = doc.page_count

    # 检查已拆分的页，跳过已有的
    existing = set(task_dirs.list_pdf_pages(task_id))

    for i in range(total):
        page_no = i + 1
        if page_no in existing:
            continue
        out_path = task_dirs.page_pdf_path(task_id, page_no)
        # 拆分为单页 PDF（保留原始压缩）
        single = fitz.open()
        single.insert_pdf(doc, from_page=i, to_page=i)
        single.save(out_path, garbage=3, deflate=True)
        single.close()
        if progress_cb and (i + 1) % 10 == 0:
            progress_cb(i + 1, total, f"拆分 PDF 第 {i + 1}/{total} 页")

    doc.close()

    # 删除原 PDF 文件
    try:
        os.remove(pdf_path)
        logger.info("已删除原 PDF: %s", pdf_path)
    except Exception as e:
        logger.warning("删除原 PDF 失败: %s", e)

    log_progress(f"任务 {task_id}: 拆分完成，共 {total} 页")
    return total


def process_single_page(
    task_id: str,
    page_no: int,
    pipeline: Any,
    slot: int,
    source_name: str,
) -> bool:
    """处理单个页面的 OCR，原子保存到 ocr_pages。

    流程:
      1. 从 pdf_pages 读取单页 PDF
      2. 渲染为图片
      3. OCR 识别（含版面分析）
      4. 原子写入 ocr_pages：
         - 先写临时 JSON 文件
         - 成功后复制单页 PDF 到 ocr_pages
         - rename 临时 JSON 为最终 JSON
      5. 任何步骤失败都返回 False，不留下任何半成品文件

    注意：单页 PDF 不叠加文字层（文字层在合并阶段统一叠加），
    ocr_pages/page_XXXX.pdf 只是 pdf_pages/page_XXXX.pdf 的副本，
    ocr_pages/page_XXXX.json 保存 OCR 结果。

    返回:
        True 表示成功，False 表示失败
    """
    src_pdf_path = task_dirs.page_pdf_path(task_id, page_no)
    if not os.path.isfile(src_pdf_path):
        logger.error("任务 %s 页 %d: pdf_pages 中找不到源文件 %s",
                     task_id, page_no, src_pdf_path)
        return False

    # 确保 ocr_pages 目录存在：断点续传场景下目录可能缺失
    # 不创建会导致 open(tmp_json, "w") 报 No such file or directory
    ocr_dir = task_dirs.ocr_pages_dir(task_id)
    os.makedirs(ocr_dir, exist_ok=True)

    # 输出路径
    final_pdf = task_dirs.ocr_pdf_path(task_id, page_no)
    final_json = task_dirs.ocr_json_path(task_id, page_no)

    # 如果已完成，跳过
    if os.path.isfile(final_pdf) and os.path.isfile(final_json):
        return True

    # 临时文件路径（处理失败时自动舍弃）
    tmp_json = final_json + ".tmp"
    tmp_pdf = final_pdf + ".tmp"

    try:
        # 1. 读取单页 PDF 并渲染为图片
        doc = fitz.open(src_pdf_path)
        page = doc.load_page(0)
        dpi = pipeline.cfg.get("render_dpi", 200)
        matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)

        # 0. 检查页面是否已有文本层：有则无需 OCR（原样保留即可）
        #    无文本层（扫描件/图片页）必须 OCR，且跳过 L1/L2 图像启发式过滤，
        #    避免低边缘密度的扫描件被 L1 误判为"文字页"而漏识。
        has_text_layer = False
        try:
            has_text_layer = bool(page.get_text().strip())
        except Exception:
            pass

        if has_text_layer:
            from ..core.document.base import PageResult
            page_result = PageResult(
                page_no=page_no,
                image=None,
                skipped=True,
                reason="已有文本层，无需OCR",
            )
        else:
            pix = page.get_pixmap(matrix=matrix, alpha=False)

            # 转换为 OpenCV BGR 数组
            import numpy as np
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = img[:, :, :3]  # RGBA -> RGB
            elif pix.n == 1:
                img = np.stack([img[:, :, 0]] * 3, axis=2)  # 灰度 -> RGB
            img = img[:, :, ::-1].copy()  # RGB -> BGR

            # 2. OCR 识别（通过 pipeline 的 handler）
            # Pipeline 有 _handlers 列表，找到 PdfHandler
            handler = None
            for h in pipeline._handlers:
                if h.__class__.__name__ == "PdfHandler":
                    handler = h
                    break
            if handler is None:
                raise RuntimeError("找不到 PdfHandler")
            page_result = handler._ocr_image(img, slot=slot, force_ocr=True,
                                               page_no=page_no)
            page_result.page_no = page_no
            # handler 层会把 OCR 引擎异常转成 skipped 页（reason="OCR 异常: ..."），
            # 这里必须转回失败，否则环境故障会被伪装成"成功但无文字"的页
            # （2026-08-23 整本空输出事故的成因之一）
            if page_result.skipped and (page_result.reason or "").startswith("OCR 异常"):
                raise RuntimeError(page_result.reason)
            # 版面诊断日志（栏缝/各栏行数/y范围/跨栏行/版面类型）
            if (page_result.ocr_result and page_result.ocr_result.lines
                    and img is not None):
                try:
                    handler._log_layout_diagnostic(
                        page_result.ocr_result.lines,
                        img.shape[1], img.shape[0], page_no,
                    )
                except Exception:
                    pass

            # 释放图片内存：DPI=200 下每页图片约 11.6MB，必须及时回收
            # gc.collect(0) 只回收 generation 0，开销极小（<1ms），避免瞬时累积
            del img
            del pix
            gc.collect(0)

        # 统一释放页面对象与文档句柄（两个分支都要关闭）
        try:
            del page
        except Exception:
            pass
        try:
            doc.close()
        except Exception:
            pass

        # 3. 保存 OCR JSON 到临时文件
        json_data = _page_result_to_json(page_result, page_no)
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False)

        # 4. 复制单页 PDF 到临时文件
        import shutil
        shutil.copy2(src_pdf_path, tmp_pdf)

        # 5. 原子 rename：临时文件 -> 最终文件
        os.replace(tmp_pdf, final_pdf)
        os.replace(tmp_json, final_json)

        # 释放 OCRResult 内存
        page_result.image = None
        page_result.ocr_result = None
        gc.collect(0)

        return True

    except Exception as e:
        logger.error("任务 %s 页 %d OCR 失败: %s", task_id, page_no, e)
        # 清理临时文件（不留半成品，连空 JSON 都不留）
        for tmp in [tmp_pdf, tmp_json]:
            try:
                if os.path.isfile(tmp):
                    os.remove(tmp)
            except Exception:
                pass
        return False


def _page_result_to_json(page_result: Any, page_no: int) -> dict:
    """把 PageResult 转换为 JSON 可序列化的字典。"""
    lines_data = []
    if page_result.ocr_result and page_result.ocr_result.lines:
        for line in page_result.ocr_result.lines:
            lines_data.append({
                "text": line.text,
                "coords": line.coords if hasattr(line, "coords") else [],
            })
    return {
        "page_no": page_no,
        "lines": lines_data,
        "skipped": page_result.skipped,
        "reason": page_result.reason or "",
    }


def build_document_result_from_ocr_pages(
    task_id: str, source_name: str,
) -> "DocumentResult":
    """从 ocr_pages 目录下的 JSON 重建 DocumentResult。

    用于 PDF 源文件输出 docx/txt/markdown/json 格式：PDF 流程逐页处理时
    已把每页 OCR 结果序列化到 ocr_pages/page_XXXX.json，本函数反向重建
    为 DocumentResult，复用 output.save_output 的文本类格式输出路径。

    遍历全部 pdf_pages 页号（保证顺序且包含缺失页），JSON 缺失的页标记
    为 skipped。ocr_json 仅持久化了 text + coords，tables / line_groups /
    confidence 会丢失（表格内容仍以文本行形式存在于 lines 中）。

    参数:
        task_id: 任务 ID
        source_name: 源文件名（用于构造 DocumentResult.source_path）

    返回:
        重建的 DocumentResult
    """
    # 懒导入避免跨模块依赖
    from ..core.document.base import DocumentResult, PageResult
    from ..providers.base import OCRLine, OCRResult

    source_path = os.path.join(task_dirs.source_dir(task_id), source_name)

    # 优先用 pdf_pages 页号（包含未完成页），退化为 ocr_pages 页号
    all_pages = task_dirs.list_pdf_pages(task_id)
    if not all_pages:
        all_pages = task_dirs.list_ocr_pages(task_id)

    page_results: List[PageResult] = []
    for page_no in all_pages:
        json_path = task_dirs.ocr_json_path(task_id, page_no)
        if not os.path.isfile(json_path):
            page_results.append(PageResult(
                page_no=page_no,
                skipped=True,
                reason="OCR 结果缺失",
            ))
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("任务 %s 页 %d 读取 JSON 失败: %s", task_id, page_no, e)
            page_results.append(PageResult(
                page_no=page_no,
                skipped=True,
                reason=f"JSON 读取失败: {e}",
            ))
            continue

        lines_data = data.get("lines", [])
        ocr_lines = [
            OCRLine(
                text=ln.get("text", ""),
                coords=[tuple(p) for p in ln.get("coords", [])],
            )
            for ln in lines_data if ln.get("text")
        ]
        ocr_result = OCRResult(lines=ocr_lines, provider="reconstructed")
        page_results.append(PageResult(
            page_no=page_no,
            ocr_result=ocr_result,
            skipped=data.get("skipped", False),
            reason=data.get("reason", ""),
        ))

    logger.info(
        "任务 %s: 从 ocr_pages 重建 DocumentResult，共 %d 页（其中 %d 页缺失）",
        task_id, len(page_results),
        sum(1 for p in page_results if p.skipped),
    )
    return DocumentResult(
        source_path=source_path,
        pages=page_results,
        native_text="",
    )


def process_task_pages(
    task_id: str,
    pipeline: Any,
    slots: List[int],
    source_name: str,
    progress_cb: Optional[Callable] = None,
    total_pages: Optional[int] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    current_page_cb: Optional[Callable[[str, int], None]] = None,
) -> Tuple[bool, int]:
    """处理任务的所有未完成页。

    实现规范化流程:
      1. 获取未完成页号数组
      2. 根据进程数分配任务
      3. 每进程 REASSIGN_EVERY_PAGES 页后重新获取未完成数组，重新分配
      4. 循环直到未完成数组为空

    参数:
        task_id: 任务 ID
        pipeline: Pipeline 实例
        slots: 可用的槽位列表
        source_name: 源文件名
        progress_cb: 进度回调 (current, total, message)
        total_pages: 总页数（None 则自动从 pdf_pages 获取）
        cancel_check: 无参回调，返回 True 表示任务已被取消/删除，
                      检测到后提前退出循环，释放文件句柄
        current_page_cb: 回调 (task_id, page_no) 上报当前正在处理的页号，
                         stall 触发时用于定位卡死页；page_no=-1 表示该页处理结束
                         多进程并行时只上报最后一个 slot 的页号（足够定位）

    返回:
        (success, completed_count)
    """
    if total_pages is None:
        total_pages = task_dirs.get_total_pages(task_id)

    if total_pages == 0:
        logger.error("任务 %s: pdf_pages 为空，无法处理", task_id)
        return False, 0

    n_workers = max(1, len(slots))
    completed = task_dirs.get_completed_pages(task_id)

    logger.info(
        "任务 %s 开始处理: 总 %d 页, 已完成 %d 页, %d 进程并行",
        task_id, total_pages, completed, n_workers,
    )

    # 单页失败计数：超过 MAX_PAGE_FAILS 的页将被跳过，避免无限重试
    # 跨 while True 轮次保持，确保同一页在多次轮次中累计失败次数
    page_fail_counts: Dict[int, int] = {}

    # 熔断统计：本轮调用中尝试过/成功的页号（跨轮次累计）。
    # 正常扫描件的"空白页"也会成功返回（True），只有引擎级故障才会大面积 False，
    # 因此用"尝试页数 + 失败率"判定环境问题，而不是统计空结果页。
    _attempted_pages: set = set()
    _ok_pages: set = set()

    def _circuit_error() -> str:
        """失败率熔断判定；触发时返回错误消息，否则返回空串。"""
        n_att = len(_attempted_pages)
        if n_att < CIRCUIT_MIN_ATTEMPTS:
            return ""
        n_fail = n_att - len(_ok_pages)
        if n_fail / n_att <= CIRCUIT_FAIL_RATIO:
            return ""
        return (
            f"{n_fail}/{n_att} 页识别失败（失败率 {n_fail / n_att:.0%}），"
            "判定为 OCR 引擎级故障，任务中止。"
            "常见原因：运行环境的 paddleocr 与代码 API 不匹配"
            "（如 \"'PaddleOCR' object has no attribute 'predict'\"）或模型目录缺失；"
            "请用项目自带 .venv 启动服务后重新提交"
        )

    # 断点续传预检查：验证 pending 页的 pdf_pages 源文件是否存在
    # 服务重启恢复任务时，pdf_pages 目录可能因上次异常退出而损坏
    # 大量源文件缺失时直接标记任务失败，避免每页都跑 3 次失败才跳过
    _pending_check = task_dirs.get_pending_pages(task_id)
    if _pending_check:
        _missing = [
            p for p in _pending_check
            if not os.path.isfile(task_dirs.page_pdf_path(task_id, p))
        ]
        if _missing:
            _missing_ratio = len(_missing) / len(_pending_check)
            if _missing_ratio > 0.5:
                logger.error(
                    "任务 %s: %d/%d 页源文件缺失（%.0f%%），任务目录损坏，标记失败",
                    task_id, len(_missing), len(_pending_check), _missing_ratio * 100,
                )
                task_dirs.log_error(
                    task_id,
                    f"任务目录损坏: {len(_missing)}/{len(_pending_check)} 页源文件缺失"
                    f"（>{int(_missing_ratio * 100)}%），请重新上传",
                )
                return False, completed
            else:
                logger.warning(
                    "任务 %s: %d 页源文件缺失（<50%%），将在失败后被跳过: %s",
                    task_id, len(_missing), _missing[:20],
                )

    # 主循环：每次获取未完成页，分配任务，处理 10 页后重新分配
    while True:
        # 取消检查：任务被删除/暂停时提前退出，释放文件句柄
        if cancel_check is not None and cancel_check():
            logger.warning("任务 %s 已取消，提前退出处理循环（已完成 %d 页）", task_id, completed)
            return False, completed

        pending = task_dirs.get_pending_pages(task_id)
        if not pending:
            break

        # 过滤掉超过最大失败次数的页：这些页已确认无法处理，跳过避免无限重试
        # 跳过的页号记录到错误日志，合并阶段会插入空白页
        skipped = [p for p in pending if page_fail_counts.get(p, 0) >= MAX_PAGE_FAILS]
        if skipped:
            for p in skipped:
                task_dirs.log_error(
                    task_id,
                    f"页 {p} 连续失败 {MAX_PAGE_FAILS} 次，跳过（合并阶段插入空白页）",
                )
            logger.warning(
                "任务 %s: %d 页超过失败阈值 %d 次，跳过: %s",
                task_id, len(skipped), MAX_PAGE_FAILS, skipped[:20],
            )
            pending = [p for p in pending if page_fail_counts.get(p, 0) < MAX_PAGE_FAILS]
            if not pending:
                logger.warning("任务 %s: 所有 pending 页均已跳过，退出主循环", task_id)
                break

        # 检查页数异常
        actual_completed = task_dirs.get_completed_pages(task_id)
        if actual_completed + len(pending) > total_pages:
            task_dirs.log_error(
                task_id,
                f"页数异常: 已完成 {actual_completed} + 未完成 {len(pending)} > 总页数 {total_pages}",
            )

        # 分配本轮处理的页（每个进程 REASSIGN_EVERY_PAGES 页）
        batch_size = n_workers * REASSIGN_EVERY_PAGES
        batch = pending[:batch_size]

        # 按进程数切分
        chunks = _split_pages_to_chunks(batch, n_workers)

        # 多线程处理
        if n_workers == 1:
            # 单进程：直接在当前线程处理
            for page_no in chunks[0]:
                # 取消检查：每页处理前检查，避免被删除后继续持有文件句柄
                if cancel_check is not None and cancel_check():
                    logger.warning("任务 %s 已取消，提前退出单进程循环（已完成 %d 页）",
                                   task_id, completed)
                    return False, completed
                # 上报当前页号：stall 触发时可定位卡死页
                if current_page_cb is not None:
                    current_page_cb(task_id, page_no)
                success = process_single_page(
                    task_id, page_no, pipeline, slots[0], source_name,
                )
                # 该页处理结束，清除标记（-1 表示空闲）
                if current_page_cb is not None:
                    current_page_cb(task_id, -1)
                _attempted_pages.add(page_no)
                if success:
                    _ok_pages.add(page_no)
                    completed += 1
                else:
                    page_fail_counts[page_no] = page_fail_counts.get(page_no, 0) + 1
                    task_dirs.log_error(
                        task_id,
                        f"页 {page_no} OCR 失败（第 {page_fail_counts[page_no]}/{MAX_PAGE_FAILS} 次）",
                    )
                    _cerr = _circuit_error()
                    if _cerr:
                        if cancel_check is not None and cancel_check():
                            return False, completed
                        task_dirs.log_error(task_id, _cerr)
                        raise RuntimeError(_cerr)

                # 更新进度
                if progress_cb:
                    progress_cb(completed, total_pages, f"已识别 {completed}/{total_pages} 页")
        else:
            # 多进程并行
            # 用 holder+lock 共享 completed：让 worker 线程每页完成后立即上报进度，
            # 避免第一波 future 全部完成前前端长时间显示 0/N
            completed_lock = threading.Lock()
            completed_holder = [completed]
            with ThreadPoolExecutor(
                max_workers=n_workers,
                thread_name_prefix=f"ocr-{task_id[:8]}",
            ) as executor:
                futures = {}
                for i, chunk in enumerate(chunks):
                    if not chunk:
                        continue
                    slot = slots[i]
                    future = executor.submit(
                        _process_chunk,
                        task_id, chunk, pipeline, slot, source_name,
                        cancel_check,
                        current_page_cb,
                        progress_cb,
                        total_pages,
                        completed_lock,
                        completed_holder,
                    )
                    futures[future] = (slot, chunk)

                # as_completed 不设超时：依赖 TaskManager 的 stall 检测（300s 无进度 kill 子进程）
                # 处理卡死。原来的 chunk_timeout=600s 与 stall_timeout=300s 竞态：
                # stall kill 子进程后 worker 重试，但 chunk_timeout 到期又 cancel future，
                # 两个超时机制互相干扰导致进度永远推不动。
                # stall 检测是独立后台监控，kill 子进程后 ocr() 抛 RuntimeError，
                # _process_chunk 正常返回失败结果，future 正常完成，无需 chunk 级超时
                try:
                    for future in as_completed(futures, timeout=None):
                        slot, chunk = futures[future]
                        try:
                            results = future.result()
                            for page_no, success in results:
                                _attempted_pages.add(page_no)
                                if not success:
                                    page_fail_counts[page_no] = page_fail_counts.get(page_no, 0) + 1
                                    task_dirs.log_error(
                                        task_id,
                                        f"页 {page_no} OCR 失败（第 {page_fail_counts[page_no]}/{MAX_PAGE_FAILS} 次）",
                                    )
                                else:
                                    _ok_pages.add(page_no)
                        except Exception as e:
                            logger.error("任务 %s slot=%d 处理异常: %s", task_id, slot, e)
                            task_dirs.log_error(task_id, f"slot={slot} 处理异常: {e}")
                        # 熔断检查放在内层 except 之外：触发时必须中止任务而不是被吞进日志。
                        # 放在 as_completed 循环体内，每个 chunk 完成即评估，尽早止损
                        _cerr = _circuit_error()
                        if _cerr:
                            if cancel_check is not None and cancel_check():
                                return False, completed
                            task_dirs.log_error(task_id, _cerr)
                            raise RuntimeError(_cerr)
                        # 进度已在 _process_chunk 内实时上报，此处不再重复调用 progress_cb
                        # （避免与 worker 线程并发调用导致进度回退）
                except TimeoutError:
                    # timeout=None 不会触发 TimeoutError，保留 except 以防万一
                    logger.error("任务 %s 多进程并行意外超时", task_id)
                # 同步 holder 中的 completed 回主线程局部变量
                completed = completed_holder[0]

        # 取消检查：每轮结束后再次检查，避免被删除后继续重建 OCR 实例
        if cancel_check is not None and cancel_check():
            logger.warning("任务 %s 已取消，本轮结束后退出（已完成 %d 页）", task_id, completed)
            return False, completed

        # 任务后重建：按累计页数触发，避免每个任务都付冷启动税。
        # 旧逻辑无条件每轮重启所有 worker → 每个任务都从冷启动开始
        # （模型加载+首推理≈60s），用户看到"页已处理完但状态仍 running 数十秒"。
        # 子进程 batch_size（_page_counts 池级跨任务计数）已能自动重启单个 worker
        # 释放内存；naive_best_fit+eager_delete 让内存可控（实测峰值后回落、不累积）。
        # 故仅当累计处理页数达阈值时才全量重启一次，中间小任务复用热 worker。
        global _pages_since_rebuild
        _pages_since_rebuild += len(batch)
        if _pages_since_rebuild >= REBUILD_AFTER_PAGES:
            _rebuild_ocr_instances(slots, pipeline)
            _pages_since_rebuild = 0
            logger.info(
                "累计处理 %d 页，已全量重启子进程释放内存（计数器归零）",
                REBUILD_AFTER_PAGES,
            )
        else:
            logger.debug(
                "本轮 %d 页，累计 %d/%d 页未触发全量重启（复用热 worker）",
                len(batch), _pages_since_rebuild, REBUILD_AFTER_PAGES,
            )

    # 取消检查：重试循环前检查
    if cancel_check is not None and cancel_check():
        logger.warning("任务 %s 已取消，跳过重试循环（已完成 %d 页）", task_id, completed)
        return False, completed

    # 最终检查：循环直到没有未完成页
    # 用户要求：最后完成前再次获取未完成的数组，没完成就完成，此为循环，直到没有
    retry_count = 0
    max_retries = 3
    while retry_count < max_retries:
        pending = task_dirs.get_pending_pages(task_id)
        if not pending:
            break
        # 重试循环同样过滤超过失败阈值的页
        skipped = [p for p in pending if page_fail_counts.get(p, 0) >= MAX_PAGE_FAILS]
        if skipped:
            pending = [p for p in pending if page_fail_counts.get(p, 0) < MAX_PAGE_FAILS]
            if not pending:
                logger.warning("任务 %s: 重试阶段所有 pending 页均已跳过", task_id)
                break
        retry_count += 1
        logger.warning(
            "任务 %s: 第 %d 次重试 %d 页未完成: %s",
            task_id, retry_count, len(pending), pending[:20],
        )
        task_dirs.log_error(
            task_id,
            f"第 {retry_count} 次重试 {len(pending)} 页: {pending[:20]}",
        )
        # 重试失败的页
        for page_no in pending:
            # 取消检查：重试阶段也要检查
            if cancel_check is not None and cancel_check():
                logger.warning("任务 %s 已取消，退出重试循环（已完成 %d 页）", task_id, completed)
                return False, completed
            # 上报当前页号：重试阶段 stall 同样需要定位
            if current_page_cb is not None:
                current_page_cb(task_id, page_no)
            success = process_single_page(
                task_id, page_no, pipeline, slots[0], source_name,
            )
            if current_page_cb is not None:
                current_page_cb(task_id, -1)
            if success:
                completed += 1
            else:
                page_fail_counts[page_no] = page_fail_counts.get(page_no, 0) + 1
            if progress_cb:
                progress_cb(completed, total_pages, f"重试 {completed}/{total_pages} 页")

    # 最终检查
    pending = task_dirs.get_pending_pages(task_id)
    if pending:
        # 区分"跳过的页"（超过失败阈值）和"真正未完成的页"
        skipped_pages = [p for p in pending if page_fail_counts.get(p, 0) >= MAX_PAGE_FAILS]
        truly_pending = [p for p in pending if page_fail_counts.get(p, 0) < MAX_PAGE_FAILS]
        if truly_pending:
            logger.warning("任务 %s 处理结束但有 %d 页未完成: %s",
                           task_id, len(truly_pending), truly_pending[:20])
            task_dirs.log_error(task_id, f"处理结束但有 {len(truly_pending)} 页未完成")
            return False, completed
        # 仅有跳过的页：任务算成功（合并阶段对缺失页插入空白页）
        if skipped_pages:
            logger.warning(
                "任务 %s: %d 页因连续失败被跳过，合并阶段将插入空白页: %s",
                task_id, len(skipped_pages), skipped_pages[:20],
            )
            task_dirs.log_error(
                task_id,
                f"以下 {len(skipped_pages)} 页被跳过（合并时插入空白页）: {skipped_pages[:20]}",
            )

    logger.info("任务 %s 全部 %d 页处理完成（含跳过页）", task_id, total_pages)
    return True, completed


def _split_pages_to_chunks(pages: List[int], n: int) -> List[List[int]]:
    """把页号列表均匀切分为 n 个块。

    余数分给前几个块。例如 [1,2,3,4,5,6,7,8,9,10] 切 3 份 → [[1,2,3,4],[5,6,7],[8,9,10]]
    """
    if n <= 0:
        return [pages]
    chunks = [[] for _ in range(n)]
    for i, page_no in enumerate(pages):
        chunks[i % n].append(page_no)
    return chunks


def _process_chunk(
    task_id: str,
    pages: List[int],
    pipeline: Any,
    slot: int,
    source_name: str,
    cancel_check: Optional[Callable[[], bool]] = None,
    current_page_cb: Optional[Callable[[str, int], None]] = None,
    progress_cb: Optional[Callable] = None,
    total_pages: Optional[int] = None,
    completed_lock: Optional[threading.Lock] = None,
    completed_holder: Optional[List[int]] = None,
) -> List[Tuple[int, bool]]:
    """处理一组页面，返回每页的处理结果。

    每 5 页重建一次 PPStructure 实例，释放版面分析的 paddle 内存池。
    PPStructure 在主进程中运行（无子进程隔离），长文档会累积内存。
    子进程 OCR 由 SubprocessOCRPool 自动重启，这里只处理 PPStructure。

    参数:
        cancel_check: 无参回调，返回 True 表示任务已被取消/删除，
                      检测到后停止处理剩余页，已处理的结果照常返回
        current_page_cb: 回调 (task_id, page_no) 上报当前正在处理的页号，
                         stall 触发时用于定位卡死页；page_no=-1 表示该页处理结束
                         多进程并行时多个 slot 会并发写，最后写入的为当前可见页号
        progress_cb: 进度回调，多进程并行时每页完成后立即上报，避免长时间 0/N
        total_pages: 总页数，配合 progress_cb 使用
        completed_lock: 保护 completed_holder 的锁（多线程并发递增）
        completed_holder: 共享的 completed 计数器 [count]，单元素列表模拟可变引用
    """
    results = []
    for i, page_no in enumerate(pages):
        # 取消检查：多进程并行时每页处理前检查
        # 命中后立即停止处理剩余页，返回已处理结果
        if cancel_check is not None and cancel_check():
            logger.warning("任务 %s slot=%d 已取消，停止处理剩余 %d 页",
                           task_id, slot, len(pages) - i)
            break
        # 上报当前页号：stall 触发时可定位卡死页
        if current_page_cb is not None:
            current_page_cb(task_id, page_no)
        success = process_single_page(task_id, page_no, pipeline, slot, source_name)
        if current_page_cb is not None:
            current_page_cb(task_id, -1)
        results.append((page_no, success))
        # 实时更新进度（多进程并行）：每页完成后立即上报，避免前端长时间 0/N
        if progress_cb is not None and completed_holder is not None and completed_lock is not None:
            with completed_lock:
                if success:
                    completed_holder[0] += 1
                cur = completed_holder[0]
            progress_cb(cur, total_pages or 0, f"已识别 {cur}/{total_pages} 页")
        # 每 5 页重建版面分析实例，释放内存
        # 阶段4：PPStructureV3 已移入子进程，由 SubprocessLayoutPool 自动每
        # batch_size 页重启子进程，无需在此手动重建。
        # 仅同进程模式（use_subprocess_layout=false）需要手动重建。
        if (i + 1) % 5 == 0 and (i + 1) < len(pages):
            layout_pool = getattr(pipeline, "_layout_pool", None)
            if layout_pool is None:
                # 同进程模式：重建主进程 PPStructureV3 实例
                try:
                    from ..core.layout import rebuild_layout_instance
                    rebuild_layout_instance(slot)
                    gc.collect()
                    logger.info(
                        "任务 %s slot=%d: 已处理 %d/%d 页，重建 PPStructureV3 实例（同进程模式）",
                        task_id, slot, i + 1, len(pages),
                    )
                except Exception as e:
                    logger.warning("重建 PPStructureV3 实例失败: %s", e)
            # 子进程模式：由 SubprocessLayoutPool 内部自动重启，无需处理
    return results


def _rebuild_ocr_instances(slots: List[int], pipeline: Any = None) -> None:
    """重建所有槽位的 OCR 实例，释放 paddle 内存池。

    子进程模式（默认）：重启子进程 OCR worker，OS 强制回收所有内存。
    同进程模式：重建主进程内的 PaddleOCR 实例池。

    无论哪种模式，都同时重建 PPStructureV3（版面分析）实例，
    因为版面分析始终在主进程中运行，没有子进程隔离。

    参数:
        slots: 需要重建的槽位列表
        pipeline: Pipeline 实例，用于获取子进程池。None 时回退同进程模式。
    """
    try:
        # 优先尝试子进程模式：通过 pipeline 获取子进程池并重启
        if pipeline is not None and hasattr(pipeline, "_paddle"):
            paddle_provider = pipeline._paddle
            # SubprocessOCRProvider 有 _pool 属性指向 SubprocessOCRPool
            if hasattr(paddle_provider, "_pool") and paddle_provider._pool is not None:
                for slot in slots:
                    try:
                        paddle_provider._pool._restart_proc(slot)
                    except Exception as e:
                        logger.warning("重启子进程 slot=%d 失败: %s", slot, e)
                logger.info("已重启所有子进程 OCR worker（释放内存）")
            else:
                # 同进程模式：重建主进程 PaddleOCR 实例池
                from ..providers.paddle_local import rebuild_ocr_instance
                for slot in slots:
                    rebuild_ocr_instance(slot)
                logger.info("已重建主进程 PaddleOCR 实例池（释放内存）")
        else:
            # 没有 pipeline 引用，回退到同进程模式重建
            from ..providers.paddle_local import rebuild_ocr_instance
            for slot in slots:
                rebuild_ocr_instance(slot)

        # 同时重启版面分析引擎（阶段4：PPStructure 已移入子进程）
        # 子进程模式下：重启 layout 子进程；同进程模式下：重建实例
        layout_pool = getattr(pipeline, "_layout_pool", None) if pipeline is not None else None
        if layout_pool is not None:
            for slot in slots:
                try:
                    layout_pool._restart_proc(slot)
                except Exception as e:
                    logger.warning("重启版面分析子进程 slot=%d 失败: %s", slot, e)
            logger.info("已重启所有版面分析子进程（释放内存）")
        else:
            # 同进程模式：重建主进程 PPStructure 实例池
            try:
                from ..core.layout import rebuild_layout_instance
                for slot in slots:
                    rebuild_layout_instance(slot)
            except Exception:
                pass
        gc.collect()
    except Exception as e:
        logger.warning("重建 OCR 实例失败: %s", e)


def merge_ocr_pages(
    task_id: str,
    source_name: str,
    output_path: Optional[str] = None,
    keep_ocr_pages: bool = True,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[str, int]:
    """合并 ocr_pages 中的单页 PDF 为最终可编辑 PDF。

    采用“分块合并”策略解决大文档（200+ 页）单文件逐页插入越来越慢、
    甚至看起来像卡死的问题：
      1. 每 CHUNK_SIZE 页先合并成一个 chunk PDF（含文字层）
      2. 最后把若干 chunk PDF 一次性合并为最终文件
      3. 这样每个 fitz 文档在任意时刻都不会超过 CHUNK_SIZE 页，
         insert_pdf 耗时保持恒定，400 页合并从数十分钟降到几分钟

    参数:
        task_id: 任务 ID
        source_name: 源文件名（用于生成输出文件名）
        output_path: 自定义输出路径，None 则用默认路径
        keep_ocr_pages: 是否保留 ocr_pages 目录（True 保留，False 删除）
        progress_cb: 进度回调 (current, total, message)

    返回:
        (out_path, merged_count)
    """
    if output_path is None:
        output_path = task_dirs.final_pdf_path(task_id, source_name)

    ocr_pages = task_dirs.list_ocr_pages(task_id)
    pdf_pages = task_dirs.list_pdf_pages(task_id)
    total = len(pdf_pages)

    if not ocr_pages:
        logger.error("任务 %s: ocr_pages 为空，无法合并", task_id)
        raise RuntimeError("ocr_pages 为空，无法合并")

    logger.info("任务 %s: 合并 %d/%d 页到 %s",
                task_id, len(ocr_pages), total, os.path.basename(output_path))
    log_progress(f"任务 {task_id}: 开始合并 {len(ocr_pages)}/{total} 页")

    # 获取字体配置
    font_kwargs = _get_font_kwargs()
    dpi = 200  # 默认 DPI，实际应从配置获取

    # 预创建 fitz.Font 对象，供 TextWriter 复用（比每次 insert_text 重新解析字体快得多）
    text_font: Optional[Any] = None
    if font_kwargs:
        try:
            if font_kwargs.get("fontfile"):
                text_font = fitz.Font(fontfile=font_kwargs["fontfile"])
            elif font_kwargs.get("fontname"):
                text_font = fitz.Font(fontname=font_kwargs["fontname"])
        except Exception as e:
            logger.warning("任务 %s: 创建字体对象失败: %s", task_id, e)
            text_font = None

    # 大文档合并优化：
    #   - 超过 200 页时跳过 subset_fonts()：实测子集化在大文档上可能耗时数分钟
    #     甚至假死，跳过可显著降低导出时间，文件稍大但可接受
    #   - .ttc 字体（simsun/msyh）不支持有效子集化，直接跳过
    skip_subset = False
    if font_kwargs:
        font_path = font_kwargs.get("fontfile", "")
        if font_path.lower().endswith(".ttc") or total > 200:
            skip_subset = True
            logger.info(
                "任务 %s: %s，跳过字体子集化以避免导出卡死",
                task_id,
                "文档超过 200 页" if total > 200 else f"字体 {font_path} 为 .ttc 格式",
            )

    # 分块大小：50 页为一个 chunk。经验值，平衡 chunk 数量和单 chunk 大小。
    CHUNK_SIZE = 50

    t_merge_start = time.time()
    task_dir = task_dirs.task_dir(task_id)
    chunks_dir = os.path.join(task_dir, "merge_chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    merged = 0
    missing = 0
    chunk_files: List[str] = []

    def _report_progress(page_no: int, stage: str) -> None:
        elapsed = time.time() - t_merge_start
        msg = f"{stage} {page_no}/{total} 页，已用 {elapsed:.1f}s"
        logger.info("任务 %s: %s", task_id, msg)
        log_progress(f"任务 {task_id}: {msg}")
        if progress_cb:
            progress_cb(page_no, total, msg)

    # 步骤 1：生成 chunk PDF
    chunk_idx = 0
    for start in range(1, total + 1, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE - 1, total)
        chunk_doc = fitz.open()
        chunk_merged = 0
        chunk_missing = 0

        for page_no in range(start, end + 1):
            ocr_pdf = task_dirs.ocr_pdf_path(task_id, page_no)
            ocr_json = task_dirs.ocr_json_path(task_id, page_no)

            if os.path.isfile(ocr_pdf) and os.path.isfile(ocr_json):
                src = fitz.open(ocr_pdf)
                chunk_doc.insert_pdf(src)
                src.close()
                try:
                    page = chunk_doc.load_page(chunk_doc.page_count - 1)
                    _overlay_text_from_json(page, ocr_json, dpi, text_font)
                except Exception as e:
                    logger.warning("任务 %s 页 %d 叠加文字层失败: %s", task_id, page_no, e)
                chunk_merged += 1
            else:
                chunk_doc.new_page(width=595, height=842)
                chunk_missing += 1
                logger.warning("任务 %s: 页 %d 丢失，插入空白页", task_id, page_no)

        merged += chunk_merged
        missing += chunk_missing

        # 保存 chunk 文件
        chunk_path = os.path.join(chunks_dir, f"chunk_{chunk_idx:04d}.pdf")
        chunk_doc.save(chunk_path, garbage=4, deflate=True)
        chunk_doc.close()
        chunk_files.append(chunk_path)
        chunk_idx += 1

        # 每处理完一个 chunk（约 50 页）上报一次进度
        _report_progress(end, "合并中")

    # 步骤 2：把所有 chunk PDF 合并为最终文件
    # 此时 chunk 数量很少（400 页 => 8 个 chunk），insert_pdf 很快
    logger.info("任务 %s: 合并 %d 个 chunk 到最终文件...", task_id, len(chunk_files))
    final_doc = fitz.open()
    for chunk_path in chunk_files:
        chunk_src = fitz.open(chunk_path)
        final_doc.insert_pdf(chunk_src)
        chunk_src.close()

    # 字体子集化（仅小文档且使用 .ttf 字体时）
    if not skip_subset:
        try:
            logger.info("任务 %s: 开始字体子集化...", task_id)
            t_subset = time.time()
            final_doc.subset_fonts()
            logger.info(
                "任务 %s: 字体子集化完成，耗时 %.1fs",
                task_id, time.time() - t_subset,
            )
        except Exception as e:
            logger.warning("任务 %s: 字体子集化失败: %s", task_id, e)
    else:
        logger.info("任务 %s: 已跳过字体子集化", task_id)

    # 保存到临时文件再替换
    tmp_output = output_path + ".tmp"
    try:
        logger.info("任务 %s: 开始保存最终 PDF...", task_id)
        t_save = time.time()
        final_doc.save(tmp_output, garbage=4, deflate=True)
        final_doc.close()
        try:
            os.replace(tmp_output, output_path)
        except PermissionError:
            # Windows 下目标文件可能被旧服务进程占用（如崩溃后残留句柄）
            # 先尝试删除目标文件再替换；仍失败则保留 .tmp 文件并抛出，避免数据丢失
            logger.warning(
                "任务 %s: 替换最终文件失败，尝试删除被占用的旧文件后重试",
                task_id,
            )
            try:
                os.remove(output_path)
                os.replace(tmp_output, output_path)
            except Exception:
                # 删除/替换都失败：保留 .tmp 文件，用户可手动处理占用
                logger.error(
                    "任务 %s: 无法写入最终文件 %s，临时文件保留在 %s",
                    task_id, output_path, tmp_output,
                )
                raise
        logger.info(
            "任务 %s: 保存完成，耗时 %.1fs",
            task_id, time.time() - t_save,
        )
    except Exception:
        # 非 PermissionError 的其他异常：清理临时文件
        if os.path.isfile(tmp_output):
            try:
                os.remove(tmp_output)
            except Exception:
                pass
        raise
    finally:
        # 清理 chunk 临时文件
        try:
            shutil.rmtree(chunks_dir, ignore_errors=True)
        except Exception:
            pass

    total_elapsed = time.time() - t_merge_start
    if missing > 0:
        task_dirs.log_error(task_id, f"合并完成但有 {missing} 页丢失（已插入空白页）")
    log_progress(
        f"任务 {task_id}: 合并完成，{merged} 页正常，"
        f"{missing} 页丢失，总耗时 {total_elapsed:.1f}s"
    )
    logger.info(
        "任务 %s: 合并完成，%d 页正常，%d 页丢失，总耗时 %.1fs",
        task_id, merged, missing, total_elapsed,
    )

    # 清理 ocr_pages 目录（仅在非 partial 模式下）
    if not keep_ocr_pages:
        try:
            shutil.rmtree(task_dirs.ocr_pages_dir(task_id), ignore_errors=True)
        except Exception:
            pass

    return output_path, merged


def _get_font_kwargs() -> Optional[dict]:
    """查找可用的中文字体。"""
    import sys as _sys
    if getattr(_sys, "frozen", False):
        base = os.path.dirname(_sys.executable)
        font_dirs = [
            os.path.join(base, "_internal", "fonts"),
            os.path.join(base, "fonts"),
            r"C:\Windows\Fonts",
        ]
    else:
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        font_dirs = [
            os.path.join(base, "fonts"),
            r"C:\Windows\Fonts",
        ]

    font_files = {
        "simhei": "simhei.ttf",
        "simsun": "simsun.ttc",
        "msyh": "msyh.ttc",
    }

    for font_name, font_file in font_files.items():
        for font_dir in font_dirs:
            path = os.path.join(font_dir, font_file)
            if os.path.isfile(path):
                return {"fontfile": path, "fontname": font_name}

    return None


def _overlay_text_from_json(
    page: fitz.Page,
    json_path: str,
    dpi: int,
    font: Optional[Any],
) -> None:
    """从 JSON 读取 OCR 结果，在页面上叠加隐形文字层。

    使用 fitz.TextWriter 批量写入：
      - 相比逐行 page.insert_text，TextWriter 把文字先收集到内存，
        最后一次性写入页面，实测大页可提速 5~10 倍。
      - 仍保持 render_mode=3（隐形可搜索）。

    参数:
        page: 目标 PDF 页
        json_path: OCR 结果 JSON 路径
        dpi: 渲染 DPI，用于像素 -> PDF 点坐标转换
        font: fitz.Font 对象；None 表示不叠加文字层
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines = data.get("lines", [])
    if not lines or font is None:
        return

    scale = 72.0 / dpi
    tw = fitz.TextWriter(page.rect)
    failed = 0

    for line in lines:
        text = line.get("text", "")
        coords = line.get("coords", [])
        if not text or not coords:
            continue
        if len(coords) >= 4:
            # coords 是 4 个顶点 [x1,y1], [x2,y2], [x3,y3], [x4,y4]
            # 取左下角作为文字起点
            x = coords[3][0] * scale
            y = coords[3][1] * scale
            # 字号取行高
            y_top = coords[0][1] * scale
            fontsize = max(4, abs(y - y_top) * 0.8)
            try:
                tw.append((x, y), text, font=font, fontsize=fontsize)
            except Exception:
                failed += 1

    if failed > 0 and failed == len(lines):
        # 全部失败时不再执行 write_text（避免空操作异常）
        return

    try:
        tw.write_text(page, render_mode=3)  # 隐形（可搜索不可见）
    except Exception:
        pass


def merge_partial(
    task_id: str,
    source_name: str,
    dpi: int = 200,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[str, int]:
    """提前导出：合并当前已完成的 ocr_pages。

    与 merge_ocr_pages 的区别：
      - 使用 partial_pdf_path 作为输出路径
      - 保留 ocr_pages 目录（任务可继续运行）
    """
    output_path = task_dirs.partial_pdf_path(task_id, source_name)
    return merge_ocr_pages(
        task_id, source_name, output_path,
        keep_ocr_pages=True, progress_cb=progress_cb,
    )
