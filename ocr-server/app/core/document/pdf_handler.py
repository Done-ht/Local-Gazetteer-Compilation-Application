"""PDF 文档处理器。

使用 PyMuPDF (fitz) 渲染每页为图片，逐页过滤 + OCR。
支持生成「可搜索 PDF」：原图层 + 隐形文字层。
"""
from __future__ import annotations

import io
import logging
import os
import time
from typing import Optional

import cv2
import numpy as np

from ...utils.progress_log import log_progress
from .base import BaseHandler, DocumentResult, PageResult

logger = logging.getLogger(__name__)


class PdfHandler(BaseHandler):
    """PDF 处理器。"""

    extensions = [".pdf"]

    def __init__(
        self,
        ocr_provider,
        image_filter,
        render_dpi: int = 200,
        layout_analyzer=None,
        rebuild_every_pages: int = 0,
        supplement_ocr: bool = True,
    ) -> None:
        super().__init__(ocr_provider, image_filter, layout_analyzer=layout_analyzer,
                         supplement_ocr=supplement_ocr)
        self.render_dpi = render_dpi
        # 每 N 页重建 OCR 实例，释放 paddle 内存池累积的中间张量
        # 0 表示不重建（适合小文件，避免重建开销）
        self.rebuild_every_pages = max(0, int(rebuild_every_pages))

    def process(self, path: str, progress_cb=None, page_cb=None, slot: int = 0,
                start_page: int = 0, end_page: Optional[int] = None) -> DocumentResult:
        """处理 PDF。

        参数:
            start_page: 从第几页开始（0基），用于断点续传。
                        前 start_page 页视为已处理，直接跳过。
            end_page: 处理到第几页（0基，不含），None表示处理到末尾。
                      用于多进程并行处理时切分页范围。
        """
        import fitz  # PyMuPDF
        import gc

        doc = fitz.open(path)
        total = doc.page_count
        if end_page is None or end_page > total:
            end_page = total
        pages: list[PageResult] = []
        zoom = self.render_dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        # 增量模式下 page_cb 会把每页写入磁盘，写完后 image 不再需要
        # 释放 image 可避免 200 页 PDF 累积 2GB+ 内存
        incremental = page_cb is not None
        # 断点续传：跳过已处理页
        start_page = max(0, min(start_page, end_page))
        if start_page > 0:
            log_progress(f"PDF 断点续传: {os.path.basename(path)}, 跳过前 {start_page} 页，从第 {start_page + 1} 页开始")
        log_progress(f"PDF 开始处理: {os.path.basename(path)}, 共 {total} 页")
        # 每页统计：用于完成后输出瓶颈分析汇总
        # 字段: page, render, filter, ocr_or_layout, write, total, lines, skipped, reason
        page_stats: list[dict] = []
        pdf_start_ts = time.time()
        for i in range(start_page, end_page):
            # === 阶段0：检查页面是否已有文本层 ===
            # 有文本层的 PDF 页（如 Word/LaTeX 导出的）本就是可编辑 PDF，
            # 无需 OCR，原样保留即可；无文本层（扫描件/图片页）必须 OCR，
            # 且跳过 L1/L2 图像启发式过滤（避免低边缘密度扫描件被误判跳过）。
            page = doc.load_page(i)
            has_text_layer = False
            try:
                has_text_layer = bool(page.get_text().strip())
            except Exception:
                pass

            # === 阶段1：渲染 ===
            t_render_start = time.time()
            if has_text_layer:
                # 有文本层：无需渲染与 OCR（原样保留，输出即为可编辑 PDF）
                from .base import PageResult
                page_result = PageResult(
                    page_no=i + 1,
                    image=None,
                    skipped=True,
                    reason="已有文本层，无需OCR",
                )
                img = None
                pix = None
                t_render = 0.0
            else:
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                img = self._pix_to_cv2(pix)
                t_render = time.time() - t_render_start

            if progress_cb:
                progress_cb(i + 1, total, f"识别 PDF 第 {i + 1}/{total} 页")
            # === 阶段2：OCR（含过滤+版面分析+识别） ===
            # 单页失败重试：最多重试 1 次（共 2 次尝试），仍失败则插入空白页并继续下一页
            # 子进程 OCR 内部已有 2 次重试（subprocess_ocr.ocr），这里再做 1 次外层重试
            # 总重试链路：pdf_handler 2次 × subprocess_ocr 2次 = 4 次，足够覆盖偶发崩溃
            # 旧值 3 次外层重试 × 3 次内层重试 × 180s 超时 = 1620s/页，远超 stall_timeout
            if not has_text_layer:
                page_result = None
                ocr_error = None
                t_ocr = 0.0
                for attempt in range(2):
                    try:
                        t_ocr_start = time.time()
                        page_result = self._ocr_image(img, slot=slot, force_ocr=True,
                                                       page_no=i + 1)
                        t_ocr = time.time() - t_ocr_start
                        break
                    except Exception as e:
                        ocr_error = e
                        logger.warning(
                            "第 %d/%d 页 OCR 第 %d 次尝试失败: %s",
                            i + 1, total, attempt + 1, e,
                        )
                        # 子进程崩溃后由 ocr() 内部自动重启，这里等待 1 秒后重试
                        if attempt < 1:
                            import time as _sleep_t
                            _sleep_t.sleep(1)
                if page_result is None:
                    # 2 次都失败：插入空白页结果，继续下一页
                    logger.error(
                        "第 %d/%d 页 OCR 连续 2 次失败，跳过此页: %s",
                        i + 1, total, ocr_error,
                    )
                    log_progress(
                        f"第 {i + 1}/{total} 页 [失败] OCR 连续 2 次失败，跳过此页"
                    )
                    from .base import PageResult, OCRResult
                    page_result = PageResult(
                        page_no=i + 1,
                        image=None,
                        ocr_result=OCRResult(lines=[], raw=[]),
                    )
                    page_result.reason = f"OCR失败: {ocr_error}"
            page_result.page_no = i + 1
            # 版面诊断日志（栏缝/各栏行数/y范围/跨栏行/版面类型）
            if (page_result.ocr_result and page_result.ocr_result.lines
                    and img is not None):
                try:
                    BaseHandler._log_layout_diagnostic(
                        page_result.ocr_result.lines,
                        img.shape[1], img.shape[0], i + 1,
                    )
                except Exception:
                    pass
            # 在写入磁盘前记录行数与原因（写入后 ocr_result 会被置 None）
            line_count = len(page_result.ocr_result.lines) if page_result.ocr_result else 0
            page_reason = page_result.reason or ""
            page_skipped = page_result.skipped
            # 从 PageResult.timings 提取各子阶段耗时
            # filter: 双层过滤 / seg: 版面切割 / recog: 区域OCR / sup: 补充识别 / reorder: 多栏排序
            # ocr: 整页OCR（无版面分析时）
            # layout_failed: 版面分析回退耗时（尝试了版面分析但覆盖率不足，回退整页OCR）
            tm = page_result.timings
            t_filter = tm.get("filter", 0.0)
            t_seg = tm.get("seg", 0.0)
            t_recog = tm.get("recog", 0.0)
            t_sup = tm.get("sup", 0.0)
            t_reorder = tm.get("reorder", 0.0)
            t_full_ocr = tm.get("ocr", 0.0)  # 整页OCR路径（无版面分析）
            t_layout_failed = tm.get("layout_failed", 0.0)  # 版面分析回退耗时
            use_layout = t_seg > 0 or t_recog > 0 or t_sup > 0
            # 增量模式下不保留 pages：page_cb 已把结果写入磁盘，
            # 保留 OCRResult（含 raw 原始字典 + lines.coords）会导致
            # 28 页累积 2.8GB+ 内存（日志实测 416MB→3259MB）
            if not incremental:
                pages.append(page_result)
            # === 阶段3：增量写入 ===
            t_write = 0.0
            if page_cb is not None:
                t_write_start = time.time()
                # 文件占用等致命错误直接抛出中断，避免后续页白跑
                page_cb(page_result)
                # 写入磁盘后 image 和 ocr_result 都不再需要，释放内存
                page_result.image = None
                page_result.ocr_result = None
                t_write = time.time() - t_write_start
            # 释放本页图片资源（pixmap + cv2 数组都是大对象）
            # DPI=200 下 A4 页约 11.6MB/页，不显式释放会累积
            del img
            del pix
            # 释放 fitz 页面对象的内部缓存
            # fitz.Page 对象在 get_pixmap 后会保留渲染数据，不显式释放会累积
            del page
            # 每5页显式 GC，避免 Python GC 滞后导致内存累积
            # 日志实测：不显式 GC 时 28 页累积到 3.2GB，每页涨约 100MB
            if (i - start_page + 1) % 5 == 0:
                gc.collect()
            # 每 N 页重建 OCR 实例，释放 paddle 内存池
            # paddle 的 C++ 内存池会缓存中间张量不归还 OS，长文档会累积到 GB 级
            # 重建实例 = 销毁旧实例 + 创建新实例，强制释放缓存
            # 用本次已处理页数判断（断点续传时绝对页码不连续）
            processed = i - start_page + 1
            if (
                self.rebuild_every_pages > 0
                and processed % self.rebuild_every_pages == 0
                and (i + 1) < total  # 最后一页不重建，避免无谓开销
            ):
                try:
                    from ...providers.paddle_local import rebuild_ocr_instance
                    rebuild_ocr_instance(slot)
                    # 同步重建版面分析实例（与 OCR 共用 slot）
                    # 阶段4：子进程 layout 模式下由 SubprocessLayoutPool 自动重启，
                    # 无需在此手动重建；仅同进程模式需要 rebuild_layout_instance
                    layout_subprocess_pool = getattr(
                        self.layout_analyzer, "_subprocess_pool", None
                    ) if self.layout_analyzer is not None else None
                    if layout_subprocess_pool is None:
                        from ..layout import rebuild_layout_instance
                        rebuild_layout_instance(slot)
                    gc.collect()
                    log_progress(f"PDF 第 {i + 1}/{total} 页后重建 OCR 实例")
                except Exception as e:
                    logger.warning("重建 OCR 实例失败（继续处理）: %s", e)
            # 每页进度日志：输出耗时分解，让用户看清瓶颈环节
            # 渲染/过滤/版面切割/区域OCR/补充识别/多栏排序/写入/版面回退 各阶段耗时
            # t_ocr 是 _ocr_image 总耗时（含 filter + 版面分析 + 整页OCR）
            # total_elapsed = t_render + t_ocr + t_write，确保总时间正确
            total_elapsed = t_render + t_ocr + t_write
            status = "跳过" if page_skipped else "完成"
            # 慢页标记：OCR 相关阶段超过 30s 时加 [慢] 提示
            ocr_related = t_seg + t_recog + t_sup + t_reorder + t_full_ocr + t_layout_failed
            slow_mark = " [慢]" if ocr_related > 30 else ""
            # 根据路径构造 OCR 阶段描述
            if use_layout:
                # 版面分析路径：拆分 seg/recog/sup/reorder
                ocr_stage = (
                    f"切割{t_seg:.1f} 识别{t_recog:.1f} 补充{t_sup:.1f} 排序{t_reorder:.1f}"
                )
            elif t_full_ocr > 0:
                # 整页OCR路径：若有版面回退，单独标注（避免时间"消失"）
                if t_layout_failed > 0:
                    ocr_stage = f"版面回退{t_layout_failed:.1f} 整页OCR{t_full_ocr:.1f}"
                else:
                    ocr_stage = f"整页OCR{t_full_ocr:.1f}"
            else:
                ocr_stage = "OCR 0.0"
            log_progress(
                f"第 {i + 1}/{total} 页 [{status}]{slow_mark} "
                f"渲染{t_render:.1f} 过滤{t_filter:.1f} {ocr_stage}s "
                f"写入{t_write:.1f} = {total_elapsed:.1f}s | "
                f"{line_count} 行 | {page_reason}"
            )
            # 详细技术日志写到 ocr_service.log
            logger.info(
                "第 %d/%d 页 渲染%.1fs 过滤%.1fs 切割%.1fs 识别%.1fs 补充%.1fs 排序%.1fs 版面回退%.1fs 整页OCR%.1fs 写入%.1fs = %.1fs | %d 行 | %s",
                i + 1, total, t_render, t_filter, t_seg, t_recog, t_sup, t_reorder,
                t_layout_failed, t_full_ocr,
                t_write, total_elapsed, line_count, page_reason,
            )
            # 收集统计：用于完成后汇总（含版面子阶段细分）
            page_stats.append({
                "page": i + 1,
                "render": t_render,
                "filter": t_filter,
                "seg": t_seg,
                "recog": t_recog,
                "sup": t_sup,
                "reorder": t_reorder,
                "full_ocr": t_full_ocr,
                "layout_failed": t_layout_failed,
                "ocr_or_layout": ocr_related,
                "write": t_write,
                "total": total_elapsed,
                "lines": line_count,
                "skipped": page_skipped,
                "reason": page_reason,
            })
        doc.close()
        # 处理完强制 GC，释放可能的循环引用
        gc.collect()
        # 输出处理汇总：总耗时、平均、各环节占比、最慢页排名
        # 让用户能判断瓶颈环节、向使用者解释耗时原因
        self._log_summary(page_stats, path, total, pdf_start_ts)
        return DocumentResult(source_path=path, pages=pages)

    @staticmethod
    def _log_summary(stats: list[dict], path: str, total: int,
                     start_ts: float) -> None:
        """输出 PDF 处理汇总到进度日志。

        包含总耗时、各环节占比（含版面子阶段）、最慢 5 页排名，
        帮助用户定位瓶颈环节、解释耗时原因。
        """
        if not stats:
            return
        elapsed = time.time() - start_ts
        elapsed_min = int(elapsed // 60)
        elapsed_sec = elapsed - elapsed_min * 60
        # 各环节累计耗时
        sum_render = sum(s["render"] for s in stats)
        sum_filter = sum(s["filter"] for s in stats)
        sum_seg = sum(s["seg"] for s in stats)
        sum_recog = sum(s["recog"] for s in stats)
        sum_sup = sum(s["sup"] for s in stats)
        sum_reorder = sum(s["reorder"] for s in stats)
        sum_full_ocr = sum(s["full_ocr"] for s in stats)
        sum_layout_failed = sum(s.get("layout_failed", 0) for s in stats)
        sum_write = sum(s["write"] for s in stats)
        sum_total = sum(s["total"] for s in stats)
        skipped_count = sum(1 for s in stats if s["skipped"])
        done_count = len(stats) - skipped_count
        total_lines = sum(s["lines"] for s in stats)
        # 占比百分比（避免除零）
        pct = lambda x: (x / sum_total * 100) if sum_total > 0 else 0
        log_progress(
            f"=== PDF 处理汇总: {os.path.basename(path)} ==="
        )
        log_progress(
            f"  总页数 {total} | 已处理 {len(stats)} 页 "
            f"(完成 {done_count}, 跳过 {skipped_count}) | "
            f"识别 {total_lines} 行 | 总耗时 {elapsed_min}分{elapsed_sec:.0f}秒"
        )
        # 主环节占比（含版面回退，避免时间"消失"）
        log_progress(
            f"  主环节占比: 渲染 {pct(sum_render):.0f}% | "
            f"过滤 {pct(sum_filter):.0f}% | "
            f"OCR/版面 {pct(sum_seg + sum_recog + sum_sup + sum_reorder + sum_full_ocr):.0f}% | "
            f"版面回退 {pct(sum_layout_failed):.0f}% | "
            f"写入 {pct(sum_write):.0f}%"
        )
        # OCR/版面子环节占比（仅在用了版面分析时显示）
        if sum_seg + sum_recog + sum_sup + sum_reorder > 0:
            log_progress(
                f"  版面子环节: 切割 {pct(sum_seg):.0f}% | "
                f"区域识别 {pct(sum_recog):.0f}% | "
                f"补充识别 {pct(sum_sup):.0f}% | "
                f"多栏排序 {pct(sum_reorder):.0f}%"
                + (f" | 整页OCR {pct(sum_full_ocr):.0f}%" if sum_full_ocr > 0 else "")
            )
        if stats:
            avg = sum_total / len(stats)
            log_progress(f"  平均 {avg:.1f}s/页")
        # 最慢 5 页排名：按总耗时倒序，输出页号/耗时/环节/原因
        slowest = sorted(stats, key=lambda s: s["total"], reverse=True)[:5]
        if slowest:
            log_progress("  最慢 5 页:")
            for s in slowest:
                # 找出该页耗时最长的环节（含版面子阶段和版面回退）
                stages = {
                    "渲染": s["render"],
                    "过滤": s["filter"],
                    "切割": s["seg"],
                    "区域识别": s["recog"],
                    "补充识别": s["sup"],
                    "多栏排序": s["reorder"],
                    "整页OCR": s["full_ocr"],
                    "版面回退": s.get("layout_failed", 0),
                    "写入": s["write"],
                }
                # 过滤掉耗时为 0 的环节
                stages = {k: v for k, v in stages.items() if v > 0}
                if stages:
                    main_stage = max(stages, key=stages.get)
                    main_time = stages[main_stage]
                else:
                    main_stage = "?"
                    main_time = 0
                log_progress(
                    f"    第 {s['page']} 页 {s['total']:.1f}s "
                    f"({main_stage} {main_time:.1f}s) | "
                    f"{s['lines']} 行 | {s['reason']}"
                )

    @staticmethod
    def _pix_to_cv2(pix) -> np.ndarray:
        """将 fitz.Pixmap 转为 OpenCV BGR 数组。"""
        arr = np.frombuffer(pix.samples, dtype=np.uint8)
        arr = arr.reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif pix.n == 1:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return np.ascontiguousarray(arr)
