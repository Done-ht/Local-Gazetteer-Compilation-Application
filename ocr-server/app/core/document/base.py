"""文档处理器基类与数据结构。

每种文档类型实现 BaseHandler.process()，返回统一的 DocumentResult。
流水线负责调度，输出转换器负责格式化。
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ...providers.base import OCRLine, OCRResult

logger = logging.getLogger(__name__)

# 文本纠错器懒加载单例（只在首次被调用时创建，避免未启用时的导入开销）
_TEXT_CORRECTOR_SINGLETON = None
_TEXT_CORRECTOR_LOCK = None  # 用普通锁避免多线程并发创建（可选）


def _get_text_corrector():
    """获取文本纠错器单例（懒加载）。"""
    global _TEXT_CORRECTOR_SINGLETON, _TEXT_CORRECTOR_LOCK
    if _TEXT_CORRECTOR_SINGLETON is None:
        try:
            import threading as _t
            if _TEXT_CORRECTOR_LOCK is None:
                _TEXT_CORRECTOR_LOCK = _t.Lock()
            with _TEXT_CORRECTOR_LOCK:
                if _TEXT_CORRECTOR_SINGLETON is None:
                    from ..text_corrector import TextCorrector
                    _TEXT_CORRECTOR_SINGLETON = TextCorrector()
                    logger.info("文本纠错器已初始化（懒加载）")
        except Exception as e:
            logger.warning("文本纠错器初始化失败，将跳过纠错: %s", e)
            _TEXT_CORRECTOR_SINGLETON = False  # 用 False 标记"不可用"，避免反复重试
    return _TEXT_CORRECTOR_SINGLETON if _TEXT_CORRECTOR_SINGLETON is not False else None


def _html_table_to_markdown(html: str) -> str:
    """把 SLANet 识别的 HTML 表格转换为 Markdown 表格文本。

    用于文本输出（txt/markdown），让表格在文本中保持结构。
    PDF 渲染时直接用原始 HTML（insert_htmlbox），不走此函数。
    """
    if not html:
        return ""
    # 提取所有行
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL | re.IGNORECASE)
    if not rows:
        return ""
    md_rows: List[List[str]] = []
    for row in rows:
        # 提取单元格（th 或 td）
        cells = re.findall(
            r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.DOTALL | re.IGNORECASE
        )
        # 去除单元格内的 HTML 标签和空白
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        # 跳过完全空的行（SLANet 偶尔输出空 <tr></tr>）
        if any(c for c in cells):
            md_rows.append(cells)
    if not md_rows:
        return ""
    # 第一行作为表头
    header = md_rows[0]
    col_count = len(header)
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join("---" for _ in range(col_count)) + " |")
    for row in md_rows[1:]:
        # 补齐列数（合并单元格可能导致行内 cell 数不一致）
        while len(row) < col_count:
            row.append("")
        lines.append("| " + " | ".join(row[:col_count]) + " |")
    return "\n".join(lines)


@dataclass
class PageResult:
    """单页处理结果。"""

    # 页面渲染图（用于生成可搜索 PDF 的原图层），可能为 None（如 DOCX 段落）
    image: Optional[np.ndarray] = None
    # OCR 结果（若该页被过滤跳过，lines 为空）
    ocr_result: OCRResult = field(default_factory=lambda: OCRResult(provider=""))
    # 是否被过滤跳过
    skipped: bool = False
    # 过滤 / 处理原因（用于日志）
    reason: str = ""
    # 页码（从 1 开始）
    page_no: int = 1
    # 各阶段耗时（秒），用于日志诊断瓶颈环节
    # 可含: filter(双层过滤) / layout(版面分析+区域OCR) / ocr(整页OCR)
    timings: Dict[str, float] = field(default_factory=dict)


@dataclass
class DocumentResult:
    """整份文档处理结果。"""

    source_path: str
    pages: List[PageResult] = field(default_factory=list)
    # 额外文本（DOCX 原生段落文字等）
    native_text: str = ""


class BaseHandler:
    """文档处理器基类。"""

    # 支持的扩展名（小写，含点）
    extensions: List[str] = []

    def __init__(self, ocr_provider, image_filter, layout_analyzer=None,
                 supplement_ocr: bool = True) -> None:
        """
        参数:
            ocr_provider: OCR 提供商实例（实现 recognize）
            image_filter: DualLayerFilter 实例
            layout_analyzer: 版面分析器实例（可选，启用后按区域分别 OCR）
            supplement_ocr: 版面分析后是否做整页补充识别（默认 True）。
                True=补回 PPStructureV3 漏切的区域（质量优先，每页多一次整页 OCR）；
                False=跳过补充识别（速度优先，省约 6-10s/页，对版面规整文档影响小）。
        """
        self.ocr = ocr_provider
        self.flt = image_filter
        self.layout = layout_analyzer
        self.supplement_ocr = supplement_ocr

    @classmethod
    def can_handle(cls, path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in cls.extensions

    def process(self, path: str, progress_cb=None, page_cb=None, slot: int = 0) -> DocumentResult:
        """处理一份文档，返回 DocumentResult。

        progress_cb(current, total, message) 用于上报进度。
        page_cb(page_result) 每页处理完后调用，用于增量写入输出。
        slot: 实例池槽位号，传给底层 OCR 引擎选择对应 PaddleOCR 实例（默认 0）。
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 工具：对单张图片执行 过滤 + OCR（含版面分析）
    # ------------------------------------------------------------------
    def _ocr_image(self, image: np.ndarray, slot: int = 0,
                   force_ocr: bool = False, page_no=None) -> PageResult:
        if image is None or image.size == 0:
            return PageResult(skipped=True, reason="空图片")

        h, w = image.shape[:2]

        import time as _time
        # Layer1/2 过滤：整页判断是否需要 OCR
        # force_ocr=True（PDF 页无文本层时传入）：跳过过滤直接 OCR，
        # 避免低边缘密度的扫描件被 L1 误判为"文字页"而漏识。
        t_flt_start = _time.time()
        if force_ocr:
            reason = "无文本层强制OCR"
        else:
            should, reason = self.flt.should_ocr(image, slot=slot)
            if not should:
                return PageResult(
                    image=image, skipped=True, reason=reason,
                    timings={"filter": t_flt},
                )
        t_flt = _time.time() - t_flt_start

        # 版面分析路径：整页 → 区域切割 → 按区域 OCR → 坐标偏移合并
        # 若版面分析回退（覆盖率不足/无文字行），记录耗时避免"时间消失"
        t_layout_failed = 0.0
        layout_failed_timings: Dict[str, float] = {}
        if self.layout is not None and self.layout.is_available():
            t_layout_start = _time.time()
            result, layout_timings = self._ocr_with_layout(image, slot, force_ocr, page_no=page_no)
            t_layout = _time.time() - t_layout_start
            if result is not None:
                # 合并耗时: filter + 版面子阶段(seg/recog/sup/reorder)
                # layout 子阶段在 _ocr_with_layout 内部已记录到 layout_timings
                merged_timings = {"filter": t_flt}
                merged_timings.update(layout_timings)
                # === 行级文本纠错（版面分析路径） ===
                # 在 OCRLine 上直接改 text，使 searchable PDF 隐形文字层、
                # txt/markdown/docx 所有输出格式同步受益。
                t_correct_start = _time.time()
                fix_count = self._apply_line_level_corrections(result)
                t_correct = _time.time() - t_correct_start
                if t_correct > 0:
                    merged_timings["text_correct"] = t_correct
                if fix_count > 0:
                    logger.info("行级文本纠错[版面路径]: 修复 %d 处，耗时 %.1fs", fix_count, t_correct)
                logger.info(
                    "耗时分解: 过滤 %.1fs + 版面[切割%.1f 识别%.1f 补充%.1f 排序%.1f] + 纠错%.1fs = %.1fs | 行数 %d",
                    t_flt,
                    layout_timings.get("seg", 0),
                    layout_timings.get("recog", 0),
                    layout_timings.get("sup", 0),
                    layout_timings.get("reorder", 0),
                    t_correct,
                    t_flt + t_layout + t_correct, len(result.lines),
                )
                return PageResult(
                    image=image, ocr_result=result, reason=f"{reason} + 版面分析",
                    timings=merged_timings,
                )
            # 版面分析回退：记录耗时，避免"时间消失"
            # _ocr_with_layout 可能花了大量时间做PPStructure切割+区域OCR，
            # 然后因覆盖率不足返回 None，这些时间必须保留到 timings 中
            t_layout_failed = t_layout
            layout_failed_timings = layout_timings
            logger.info(
                "版面分析回退（耗时 %.1fs: 切割%.1f 识别%.1f 补充%.1f），回退整页OCR",
                t_layout,
                layout_timings.get("seg", 0),
                layout_timings.get("recog", 0),
                layout_timings.get("sup", 0),
            )

        # 回退路径：整页 OCR
        # 【关键修复 V2 - 强制分栏兜底（物理防串栏）】
        # 用户正确指出"长方形版面不该串栏"，但在本回退路径中我们根本没版面框，
        # 只能对整图识别 → PaddleOCR 内部按 y 逐行，天然左右栏交错！
        # 修复：先用一次轻量「中线双栏试探」：若大部分文字行中心 cx 的分布
        # 在中线两侧形成双峰（行中心左/右数量均 >= 25%），就认为是双栏文档，
        # 改为【按中线裁剪左右半页 → 分别 OCR → 左栏 lines + 右栏 lines】，
        # 物理上保证左栏整栏读完再读右栏，100% 避免一行左一行右的 y 交错串栏！
        # 若非双栏则仍用原整页 OCR（单栏/正文不受影响）。
        # 注意：此双栏特征检测必须很快 → 用一次 detect 级 PaddleOCR（无识别）
        # 或直接用"整页 OCR 一次 + 检查行 cx 分布"。为减少复杂度我们先做
        # 整页 OCR（为了拿行坐标），再决定是否重跑"左半+右半"版本：
        #   - 若单栏 → 直接用整页结果（开销 1 次 OCR）
        #   - 若双栏 → 再跑 2 次半页 OCR 拼接（开销 3 次 OCR，但质量远胜）
        t_ocr_start = _time.time()
        try:
            full_page_result = self.ocr.recognize(image, slot=slot, new_page=True)
        except Exception as e:
            return PageResult(
                image=image,
                skipped=True,
                reason=f"OCR 异常: {e}",
                timings={"filter": t_flt, "ocr": _time.time() - t_ocr_start},
            )
        t_ocr = _time.time() - t_ocr_start

        # ==== 强制分栏检测：按栏缝切分为 N 栏，各栏分别 OCR（物理防串栏） ====
        # 旧版只按"中线"左/右双栏试探，会把标准三栏误判为双栏（中栏行中心落在
        # 中线±5%内不计入任一侧），再从中线一刀切 → 中栏被拦腰截断，左半页混入
        # 第1、2栏、右半页混入第2、3栏，即"第二栏末尾和第三栏混在一起"。
        #
        # 修复 V3（首选策略：全部文本框 + KMeans 聚类 — 基于三栏页诊断验证）：
        #   用 full_page_result.lines 的全部文本框 cx 做 1D KMeans（不筛选行宽，
        #   避免"正文宽行"被误判成跨栏候选而剔除 → 反而使推断失真），
        #   K=3 聚类相邻簇中点即为栏缝，实测误差 4-8px（<0.5% 页宽，非常精确）。
        #
        # 回退链：
        #   ① KMeans(全部文本框 cx)  → 投影列像素空白带  → ③ 行 cx 间隙启发
        force_split_done = False
        split_timing = 0.0
        if full_page_result.lines and w > 0 and h > 0:
            total_lines = len(full_page_result.lines)
            # 0) 首选：全部OCR文本框 + KMeans 聚类（V3 高精度方法）
            #    此方法对非均分三栏（如样例年鉴 w=1678, 缝=[618,1082]）
            #    精度最高：平均误差仅 4px，远优于投影法/间隙法
            valid_gaps = BaseHandler._detect_column_gutters_kmeans(
                full_page_result.lines, w, h
            )
            if valid_gaps:
                logger.info(
                    "栏缝检测①[KMeans-全部文本框]: 检测到 %d 栏（缝 x=%s，%d行）",
                    len(valid_gaps) + 1, valid_gaps, total_lines,
                )
            else:
                # ① 回退1：图像列投影检测栏缝（次优，与行长短无关）
                valid_gaps = self._detect_column_gutters_projection(image, w, h)
                if valid_gaps:
                    logger.info(
                        "栏缝检测②[列投影法]: 检测到 %d 栏（缝 x=%s，%d行）",
                        len(valid_gaps) + 1, valid_gaps, total_lines,
                    )
                else:
                    # ② 回退2：行中心 cx 分布启发（与 _reorder_by_columns 方法2一致）
                    line_cxs_fb = sorted(
                        (min(p[0] for p in ln.coords) + max(p[0] for p in ln.coords)) / 2
                        for ln in full_page_result.lines
                    )
                    gap_threshold = max(50, w * 0.06)
                    all_gap_xs = []
                    for i in range(len(line_cxs_fb) - 1):
                        g = line_cxs_fb[i + 1] - line_cxs_fb[i]
                        if g >= gap_threshold:
                            gap_x = (line_cxs_fb[i] + line_cxs_fb[i + 1]) / 2
                            all_gap_xs.append(gap_x)
                    all_gap_xs = sorted(set(round(g) for g in all_gap_xs))
                    all_gap_xs = [g for g in all_gap_xs if w * 0.08 <= g <= w * 0.92]
                    valid_gaps = all_gap_xs
                    logger.info(
                        "栏缝检测③[行cx间隙启发]（KMeans/投影均未命中）: 缝 x=%s",
                        valid_gaps,
                    )
            # 1) 用 OCR 行 cx 验证每个栏缝两侧都有足够文本行（>=15%），过滤噪声
            if valid_gaps:
                line_cxs_sorted = sorted(
                    (min(p[0] for p in ln.coords) + max(p[0] for p in ln.coords)) / 2
                    for ln in full_page_result.lines
                )
                validated = []
                for gap_x in valid_gaps:
                    left = sum(1 for cx in line_cxs_sorted if cx < gap_x)
                    right = total_lines - left
                    if left >= total_lines * 0.15 and right >= total_lines * 0.15:
                        validated.append(gap_x)
                valid_gaps = validated
            num_cols = len(valid_gaps) + 1
            # 仅当确认为 2 栏及以上、且栏缝清晰时才物理切分
            if num_cols >= 2:
                logger.info(
                    "整页OCR回退: 检测到 %d 栏（栏缝 x=%s，总%d行）"
                    " → 按栏缝切分各栏分别OCR（物理防串栏）",
                    num_cols, valid_gaps, total_lines,
                )
                t_split_start = _time.time()
                try:
                    OVERLAP_PX = 30
                    # 按栏缝切分为 N 条：[(x1, x2), ...]
                    strips = []
                    prev = 0
                    for gx in valid_gaps:
                        strips.append((prev, gx))
                        prev = gx
                    strips.append((prev, w))
                    # 每条各扩展 OVERLAP_PX 到相邻栏缝重叠区（栏缝附近的字避免被截断）
                    strip_results = []
                    for si, (sx1, sx2) in enumerate(strips):
                        crop_x1 = max(0, sx1 - (OVERLAP_PX if si > 0 else 0))
                        crop_x2 = min(w, sx2 + (OVERLAP_PX if si < len(strips) - 1 else 0))
                        strip_img = image[:, crop_x1:crop_x2]
                        res = self.ocr.recognize(strip_img, slot=slot, new_page=False)
                        strip_results.append((crop_x1, res))
                    # 各栏 OCR 行坐标整体偏移回原页坐标系，按左→右顺序拼接
                    new_lines = []
                    for crop_x1, res in strip_results:
                        for ln in res.lines:
                            new_coords = [(x + crop_x1, y) for x, y in ln.coords]
                            new_lines.append(
                                OCRLine(text=ln.text, coords=new_coords,
                                        confidence=ln.confidence)
                            )
                    # 覆盖整页结果：第1栏lines + 第2栏lines + ...（已按阅读顺序）
                    full_page_result.lines = new_lines
                    # 行分组清空（重新从空建）
                    full_page_result.line_groups = []
                    # 栏缝重叠区去重：仅当两行文本完全相同且都落在栏缝重叠带内时去重，
                    # 避免 OVERLAP 区域的行被相邻两栏各识别一次
                    if OVERLAP_PX > 0 and len(strip_results) > 1:
                        boundary_xs = [gx for gx in valid_gaps]
                        dedup_lines = []
                        seen_in_overlap = set()
                        for ln in new_lines:
                            xs = [p[0] for p in ln.coords]
                            cx = (min(xs) + max(xs)) / 2
                            near_boundary = any(abs(cx - bx) <= OVERLAP_PX
                                                for bx in boundary_xs)
                            cand = ln.text.strip()
                            if near_boundary and cand and cand in seen_in_overlap:
                                continue  # 栏缝重叠带内的完全重复行，丢弃
                            if near_boundary and cand:
                                seen_in_overlap.add(cand)
                            dedup_lines.append(ln)
                        full_page_result.lines = dedup_lines
                    force_split_done = True
                    split_timing = _time.time() - t_split_start
                    logger.info(
                        "强制分栏兜底完成: %d 栏 → 各栏 %s 行 → 拼接后共%d行（去重后） 耗时%.1fs",
                        num_cols,
                        [len(r.lines) for _, r in strip_results],
                        len(full_page_result.lines), split_timing,
                    )
                except Exception as _spl_e:
                    logger.warning(
                        "强制分栏兜底异常（回退用整页OCR结果 + 后续reorder）: %s", _spl_e
                    )
                    force_split_done = False
        result = full_page_result
        # 构建 timings：若版面分析回退，保留版面分析耗时避免"时间消失"
        # layout_failed 记录版面分析总耗时，layout子阶段也保留供诊断
        timings = {"filter": t_flt, "ocr": t_ocr}
        if t_layout_failed > 0:
            timings["layout_failed"] = t_layout_failed
            # 保留版面分析子阶段耗时（前缀加 layout_ 避免与正常路径冲突）
            for k, v in layout_failed_timings.items():
                timings[f"layout_{k}"] = v
        if force_split_done and split_timing > 0:
            timings["force_split_double_col"] = split_timing
        # 如果强制分栏成功完成，把后续的 fallback_reorder 记为"已完成"（不重复做），
        # 因为左栏lines+右栏lines的拼接已经保证了正确的阅读顺序。
        _force_split_already_ordered = force_split_done
        logger.info(
            "耗时分解: 过滤 %.1fs + 整页OCR %.1fs%s = %.1fs | 行数 %d",
            t_flt, t_ocr,
            f" + 版面回退{t_layout_failed:.1f}" if t_layout_failed > 0 else "",
            t_flt + t_ocr + t_layout_failed, len(result.lines),
        )

        # === 整页 OCR 路径：多栏排序（除非强制分栏已物理拼接好） ===
        # 只有在"强制分栏兜底"未触发或失败时，才继续用 _reorder_by_columns 的
        # 数据驱动分栏检测做最后的 reorder；若强制分栏已把左半页 OCR 结果 + 右半
        # 页 OCR 结果按阅读顺序拼接，则 reorder 已在裁剪阶段物理完成，再跑统计
        # reorder 反而可能因左右栏 y 绝对差造成误排序（打乱拼接好的正确顺序）。
        if not _force_split_already_ordered:
            t_reorder_start = _time.time()
            result.lines, result.line_groups = self._reorder_by_columns(
                result.lines, result.line_groups,
                regions=[],
                page_width=w,
                skip_orders=set(),
                text_types=set(),
            )
            t_reorder = _time.time() - t_reorder_start
            if t_reorder > 0.01:
                timings["fallback_reorder"] = t_reorder
                logger.info("整页OCR回退路径: 多栏排序耗时 %.1fs", t_reorder)
        else:
            # 强制分栏已完成正确顺序拼接，记录一个"0耗时"以便日志区分两条路径
            logger.info(
                "整页OCR回退路径: 强制分栏兜底已按左→右顺序拼接，跳过多栏排序reorder",
            )

        # 目录页碎片合并：孤立页码归位 + 散字标题合并（整页 OCR 路径同样适用）
        result.lines, result.line_groups = self._merge_toc_fragments(
            result.lines, result.line_groups, w, h
        )

        # === 行级文本纠错（整页OCR回退路径） ===
        t_correct_start2 = _time.time()
        fix_count2 = self._apply_line_level_corrections(result)
        t_correct2 = _time.time() - t_correct_start2
        if t_correct2 > 0:
            timings["text_correct"] = t_correct2
        if fix_count2 > 0:
            logger.info("行级文本纠错[整页OCR路径]: 修复 %d 处，耗时 %.1fs", fix_count2, t_correct2)

        return PageResult(
            image=image, ocr_result=result, reason=reason,
            timings=timings,
        )

    # ------------------------------------------------------------------
    # 工具：行级文本纠错（写回 OCRLine.text，所有输出格式同步受益）
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_line_level_corrections(result: OCRResult) -> int:
        """对 OCRResult 的每一行做文本纠错，直接修改 OCRLine.text。

        返回纠错处数（仅用于日志统计）。
        纠错器不可用时直接返回 0，不阻塞主流程。

        ===== 关键修复（V2 写回逻辑） =====
        旧版 bug：当 merge_to_paragraphs=True 时，corrector.correct_lines 会把
        相邻的短行合并为更少的逻辑行，导致 len(corrected.lines) < len(result.lines)。
        旧代码按索引对位写回 for i,new in enumerate(corrected.lines): result.lines[i].text = new
        → 只有前 N 行（N=合并后行数）被正确写回，后面全部行的纠错"消失"。
        这就是用户反馈"人学率依然是 人学率"的根因：因为「人学率」那行在被合并
        行的后面部分，没被写回。

        V2 修复策略（两条通路互补，保证纠错结果全部落地）：
          A) 逐行单条纠错（保证每行 OCRLine.text 至少被词典+模式扫过一遍）
             —— 不做跨行合并，仅做行内替换（merge_to_paragraphs=False），
                因此输出行数恒等于输入行数，写回能按索引严格对位。
          B) 合并后的段落级结果写回 result.text 字段（若 OCRResult 有 text 字段）
             用于 output.py 的全文档级再利用（OCR 行仍按物理行，
             搜索 PDF 隐形文字层可正常按坐标定位）。
        """
        if not result.lines:
            return 0
        corrector = _get_text_corrector()
        if corrector is None:
            return 0
        try:
            # ====== 通路 A：逐行单条纠错（行数严格相等，保证写回对位） ======
            raw_texts = [ln.text for ln in result.lines]
            fixed = 0
            for i, raw in enumerate(raw_texts):
                if not raw:
                    continue
                # 对单行调用 correct_lines (单元素列表 + 不允许跨行合并)
                # 保证输出行数=1，与 OCRLine 一一对应
                single_res = corrector.correct_lines([raw], merge_to_paragraphs=False)
                if not single_res.lines:
                    continue
                new_text = single_res.lines[0]
                if new_text and new_text != raw:
                    result.lines[i].text = new_text
                    fixed += 1
            # ====== 边界叠字回退 ======
            # 纠错规则按单行生效、看不到下一物理行。原文跨行断词时
            # （上行末尾"…不断得"、下行开头"到加强…"），行尾补全规则会把
            # 下一行行首已有的字再补到上行末尾 → 拼读起来出现"得到到"式叠字。
            # 仅当追加确由纠错引入（原始行末尾不是该字、纠错只是向后追加）、
            # 且下一行以该字开头时，撤销追加，把断字还给下一行。
            for i in range(len(result.lines) - 1):
                raw_i = raw_texts[i]
                fixed_i = result.lines[i].text
                nxt = result.lines[i + 1].text
                while (
                    fixed_i
                    and raw_i
                    and nxt
                    and len(fixed_i) > len(raw_i)
                    and fixed_i.startswith(raw_i)
                    and fixed_i[-1] == nxt[0]
                    and not raw_i.endswith(fixed_i[-1])
                ):
                    fixed_i = fixed_i[:-1]
                if fixed_i != result.lines[i].text:
                    logger.info(
                        "边界叠字回退: %r → %r（下一行以 %r 开头）",
                        result.lines[i].text, fixed_i, nxt[0],
                    )
                    result.lines[i].text = fixed_i
            logger.debug(
                "_apply_line_level_corrections(V2): 逐行模式 %d 行 → 修复 %d 处",
                len(raw_texts), fixed,
            )
            return fixed
        except Exception as e:
            logger.warning("行级文本纠错(V2)异常（跳过，不阻断）: %s", e)
            return 0

    def _ocr_with_layout(self, image: np.ndarray, slot: int = 0,
                         force_ocr: bool = False, page_no=None) -> tuple:
        """版面分析 + 按区域 OCR。

        流程:
          1. PPStructureV3 切割页面为多个区域（text/title/...）
          2. 预计算 title 区域的扩展范围（底部多扩 50px，用于识别跨行标题续行）
          3. 对每个文本类区域单独过滤 + OCR
             - title 区域用大 padding（x=20, 顶=20, 底=15），只覆盖第一行
             - text 区域用小 padding（10px），续行由 text 区域识别（更完整）
          4. 将区域内的行坐标偏移回原页坐标系
          5. 跳过 text 区域中与 title 第一行重叠的行（避免碎片重复）
          6. 跨行标题合并：title 第一行 + text 区域续行 → 同一逻辑行
          7. 覆盖率回退：若文本类区域面积覆盖率 < 15%，回退整页 OCR
          8. 补充识别：整页 OCR 后把不在任何已处理区域内的行补充进来
          9. 多栏排序：检测双栏布局并按"左栏从上到下 → 右栏从上到下"重排

        返回 (OCRResult, timings_dict)，失败时返回 (None, timings_dict)。
        timings 含: seg(版面切割) / recog(区域OCR) / sup(补充识别) / reorder(多栏排序)
        """
        import time as _time
        timings: Dict[str, float] = {}
        h, w = image.shape[:2]
        # 版面分析（PPStructureV3 切割页面为区域）
        t_la_start = _time.time()
        layout_res = self.layout.analyze(image, slot=slot, page_no=page_no)
        t_la = _time.time() - t_la_start
        timings["seg"] = t_la
        logger.info("版面分析(PPStructure): %.1fs, 区域数=%d", t_la, len(layout_res.regions))
        if not layout_res.available:
            logger.info("版面分析不可用，回退整页 OCR: %s", layout_res.reason)
            return None, timings

        # ===== 原生 OCR 快速路径 =====
        # PPStructureV3.predict() 一次调用即完成整页 OCR + XY-cut 阅读顺序排序，
        # layout_res.native_ocr_lines 已含全部文本行（text/bbox/score，阅读顺序）。
        # 直接使用可跳过逐区域 OCR（34 次→0 次）和自定义栏分析（KMeans/force_split/reorder），
        # 复现官方 ~200ms 级性能。仅在原生行可用时走此路径，否则回退下方逐区域 OCR。
        if layout_res.native_ocr_lines:
            native_lines: List[OCRLine] = []
            for ln in layout_res.native_ocr_lines:
                txt = ln.get("text", "") or ""
                bbox = ln.get("bbox") or [0, 0, 0, 0]
                if len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = [int(v) for v in bbox]
                # 跳过空文本行（PPStructure 偶尔产出空行）
                if not txt.strip():
                    continue
                native_lines.append(OCRLine(
                    text=txt,
                    coords=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
                    confidence=float(ln.get("score", 0.0)),
                ))
            # 提取表格 HTML（供 searchable PDF 用 insert_htmlbox 渲染表格结构）
            native_tables: List[dict] = []
            for r in layout_res.regions:
                if r.type == "table" and r.html:
                    native_tables.append({"bbox": r.bbox, "html": r.html})
            if native_lines:
                result = OCRResult(
                    lines=native_lines, width=w, height=h,
                    provider="paddleocr-native", tables=native_tables,
                )
                timings["recog"] = 0.0  # OCR 已在 predict() 内完成，无额外区域 OCR
                timings["sup"] = 0.0
                timings["reorder"] = 0.0
                logger.info(
                    "原生OCR快速路径: %d 行（PPStructureV3 整页OCR+XY-cut），"
                    "跳过逐区域OCR与自定义栏分析",
                    len(native_lines),
                )
                return result, timings
            # 原生行提取失败（空行过多或异常），回退逐区域 OCR
            logger.info("原生OCR行为空，回退逐区域 OCR 路径")

        all_lines: List[OCRLine] = []
        line_groups: List[List[int]] = []
        # 只对文本类区域做 OCR，跳过 figure 等纯图区域
        # table 区域也做 OCR（表格内容需要识别，只是不还原表格结构）
        text_types = {"text", "title", "header", "footer", "reference", "equation", "table"}
        # 被拆分的跨栏大框 / 被覆盖的重复区域的 order 集合，用于后续跳过这些区域不 OCR
        skip_orders: set = set()

        # 统计文本类区域面积，用于判断版面分析是否覆盖了足够的内容
        total_area = h * w
        text_region_area = 0

        # 预计算 title 区域的扩展范围
        # 版面分析对跨行标题常只切出第一行，第二行被误归到下方 text 区域。
        # 策略：title 区域只 OCR 第一行（底padding=15，不含第二行），
        # 第二行由 text 区域识别（OCR 更完整，因为 text 区域有完整上下文）。
        # 然后将 text 区域中属于标题的续行加入 title 的 line_groups。
        #
        # title_expanded 用于两个目的：
        #   1. _find_title_continuation: 判断 text 行是否是标题续行（y 在 title 下方 50px 内）
        #   2. _is_title_fragment: 判断 text 行是否是 title 第一行的碎片（y 与 title 重叠）
        TITLE_PAD_X = 20    # title 区域 x 方向 padding（补偿 bbox 起点偏移）
        TITLE_PAD_TOP = 20   # title 区域顶部 padding
        TITLE_PAD_BOT = 15   # title 区域底部 padding（只覆盖第一行，不含续行）
        TITLE_CONT_MAX_GAP = 50  # 标题续行与 title 底部的最大间距（px）
        title_bboxes: List[List[int]] = []        # 原 title bbox [x1, y1, x2, y2]
        title_expanded: List[List[int]] = []      # title bbox + padding，用于碎片检测
        title_cont_ranges: List[List[int]] = []   # [x1, y_bottom, x2, y_bottom+GAP]，用于续行检测
        for region in layout_res.regions:
            if region.type == "title":
                rx1, ry1, rx2, ry2 = region.bbox
                title_bboxes.append([rx1, ry1, rx2, ry2])
                title_expanded.append([
                    max(0, rx1 - TITLE_PAD_X),
                    max(0, ry1 - TITLE_PAD_TOP),
                    min(w, rx2 + TITLE_PAD_X),
                    min(h, ry2 + TITLE_PAD_BOT),
                ])
                # 续行检测范围：title 底部下方 GAP 像素，x 范围扩展
                title_cont_ranges.append([
                    max(0, rx1 - TITLE_PAD_X),
                    ry2,
                    min(w, rx2 + TITLE_PAD_X),
                    min(h, ry2 + TITLE_CONT_MAX_GAP),
                ])

        def _line_center(line_coords: List, x_off: int, y_off: int):
            """返回行的中心坐标 (cx, cy) 和 x/y 范围，基于原图坐标系。"""
            xs = [p[0] + x_off for p in line_coords]
            ys = [p[1] + y_off for p in line_coords]
            return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, min(xs), max(xs), min(ys), max(ys)

        def _is_title_fragment(line_coords: List, x_off: int, y_off: int) -> bool:
            """判断 text 行是否是 title 第一行的碎片（y 与 title 重叠）。
            这些碎片是 text 区域 padding 裁剪到 title 第一行底部造成的，必须跳过。
            """
            cx, cy, _, _, _, _ = _line_center(line_coords, x_off, y_off)
            for tx1, ty1, tx2, ty2 in title_expanded:
                if tx1 <= cx <= tx2 and ty1 <= cy <= ty2:
                    return True
            return False

        def _find_title_continuation(line_coords: List, x_off: int, y_off: int) -> int:
            """判断 text 行是否是某个 title 的续行，返回 title 索引或 -1。
            续行条件：y 在 title 底部下方 GAP 内，且 x 范围与 title 重叠。
            """
            cx, cy, lx1, lx2, _, _ = _line_center(line_coords, x_off, y_off)
            for i, (tx1, ty1, tx2, ty2) in enumerate(title_cont_ranges):
                if not (ty1 <= cy <= ty2):
                    continue
                # x 范围需有重叠（标题续行通常与标题左对齐或在其 x 范围内）
                if lx2 < tx1 or lx1 > tx2:
                    continue
                return i
            return -1

        # 记录每个 title 的行索引（用于后续合并续行到 line_groups）
        title_line_indices: List[List[int]] = [[] for _ in title_bboxes]
        # 每个 title 只匹配第一个续行，避免把后续正文行误判为续行
        title_has_cont: List[bool] = [False] * len(title_bboxes)

        # ==================================================================
        # 【修复 V2 补 1 前置】processed_bboxes：记录"真实裁剪范围"
        # 必须在 region OCR 循环之前声明为空列表，循环内每成功处理一个 region
        # 就把 (px1,py1,px2,py2) append 进来。用于 supplement_ocr 的"已处理
        # 区域"判断——必须用真实裁剪范围（含 padding），不能用原始 bbox
        # （旧 bug：用原始 bbox 导致 pad 外的文字被重复加入）
        # ==================================================================
        processed_bboxes: List[List[int]] = []

        # ==================================================================
        # 【关键修复 V2】跨栏大框拆分 + 去重前置（根治串栏 + 内容重复 2-3 次）
        # ==================================================================
        # 用户怀疑正确：PPStructure 偶尔会切出一个"宽度 = 页宽 70%+ 且横跨中线
        # （x1<page_mid<x2）的大 text 框"。对这种框直接裁剪 OCR → PaddleOCR
        # 内部按 y 坐标逐行读 → 左栏y=100 + 右栏y=110 + 左栏y=120 被依次读出
        # → 左右栏内容一行行硬拼（即用户看到的"硬件基础设施不断得→院投资500万"）
        # 同时旁边如果有单栏小区域（重叠度约 0.5），两者各 OCR 一次 + 补充 OCR
        # 整页再读一次 → 同一段话碎片重复出现 2-3 次（截断位置不同）。
        #
        # 修复：在做任何 OCR 之前，先扫描所有文本类 region：
        #   a) 若 region 横跨页面中线 + 宽度 > 页宽 65% → 标记为「跨栏嫌疑框」
        #      → 在中线处 SPLIT 为两个子 region（左半、右半），避免单个框内混读
        #   b) 若 region 与其他 region 有双向高覆盖 → 跳过大的（保留小的精确）
        # ==================================================================
        page_mid_x = w / 2  # 页面水平中线
        COL_SPAN_W_RATIO = 0.65  # 跨栏判定阈值：宽度>页宽65%且横跨中线
        # 拆分后的 region 要加入 layout_res.regions，因此需要构造新的 LayoutRegion 对象。
        # 先获取 LayoutRegion 类（从 providers 中通过实例动态取，避免硬依赖模块路径）
        if layout_res.regions:
            _LR_Cls = type(layout_res.regions[0])
            _has_attr_bbox = all(hasattr(r, "bbox") for r in layout_res.regions)
            _has_attr_type = all(hasattr(r, "type") for r in layout_res.regions)
            _has_attr_order = all(hasattr(r, "order") for r in layout_res.regions)
        else:
            _LR_Cls = None
            _has_attr_bbox = _has_attr_type = _has_attr_order = False

        # 阶段 A：扫描并拆分跨栏大框（把拆分结果追加到 layout_res.regions 新列表
        # 中，替换原列表，同时在 skip_orders 中标记原大框不被 OCR）
        if _LR_Cls is not None and _has_attr_bbox and _has_attr_type and _has_attr_order:
            new_regions = []
            split_count = 0

            # 【新增】拆分前先检测真正的"栏缝"坐标。旧逻辑只在中线 w/2 一刀切，
            # 而中线往往不是栏缝，导致"横跨第2、3栏"的大框右半仍含两栏内容 → 串栏。
            # 这里从"非跨栏 text 区域"的 x 范围合并后求间隙，得出各栏缝 x。
            # 注意必须排除跨栏大框自身（它会桥接栏缝，破坏间隙检测）。
            def _compute_column_gap_xs():
                _ranges = []
                for _r in layout_res.regions:
                    if _r.type not in text_types:
                        continue
                    if _r.type in ("table", "title"):
                        continue
                    _rx1, _, _rx2, _ = _r.bbox
                    _rw = _rx2 - _rx1
                    # 跳过本身是"跨栏大框"的区域（横跨中线且较宽）
                    if _rx1 < page_mid_x < _rx2 and _rw / w > 0.5:
                        continue
                    _ranges.append((_rx1, _rx2))
                if len(_ranges) < 2:
                    return []
                _ranges.sort()
                _merged = []
                _OV = 5  # 小重叠容忍（px）
                for _x1, _x2 in _ranges:
                    if _merged and _x1 <= _merged[-1][1] + _OV:
                        _pw = _merged[-1][1] - _merged[-1][0]
                        _cw = _x2 - _x1
                        _ov = _merged[-1][1] - _x1
                        _q = w / 4
                        if _ov <= _OV and _pw > _q and _cw > _q:
                            # 两个宽区的微重叠 → 是两栏，不合并
                            _merged.append((_x1, _x2))
                        else:
                            _merged[-1] = (_merged[-1][0], max(_merged[-1][1], _x2))
                    else:
                        _merged.append((_x1, _x2))
                _gaps = []
                for _i in range(len(_merged) - 1):
                    _g = _merged[_i + 1][0] - _merged[_i][1]
                    if _g >= 30:
                        _gaps.append(round((_merged[_i][1] + _merged[_i + 1][0]) / 2))
                return sorted(_gaps)

            _col_gap_xs = _compute_column_gap_xs()
            if _col_gap_xs:
                logger.info(
                    "跨栏拆分-栏缝检测: 页面栏缝 x=%s（用于切分跨栏大框）", _col_gap_xs,
                )

            # 负整数序号发生器，保证拆分出的子 region order 全局唯一
            _neg_seq = 1

            for region in layout_res.regions:
                if region.type not in text_types:
                    new_regions.append(region)
                    continue
                rx1, ry1, rx2, ry2 = region.bbox
                rw = rx2 - rx1
                rw_ratio = rw / w if w > 0 else 0
                cross_midline = (rx1 < page_mid_x < rx2)
                is_cross_col_span = cross_midline and rw_ratio > COL_SPAN_W_RATIO
                # 特例：table 区域不拆（表格本身就是多列，SLANet 内部处理）；
                # title 区域也不拆（标题通常居中跨两栏是合法的）
                if region.type in ("table", "title"):
                    is_cross_col_span = False

                # 落在该大框内的栏缝切点（真正的栏缝，而非中线）
                inner_gaps = [gx for gx in _col_gap_xs if rx1 < gx < rx2]
                if is_cross_col_span and inner_gaps:
                    # 按栏缝切成多段（如三栏横条 → 左/中/右三栏独立子框）
                    cuts = sorted(inner_gaps)
                else:
                    # 未在框内检测到真实栏缝时，视为单栏正文（如"编辑说明"整页
                    # 通栏排版的散文），不拆分。旧逻辑会回退到中线 w/2 一刀切，
                    # 导致通栏散文每行被拦腰截断、左右半边分列读出 → 内容错乱。
                    new_regions.append(region)
                    continue

                # 过滤掉距左右边界过近的切点，避免切出 0 宽/极窄片段
                cuts = [c for c in cuts if rx1 + 40 <= c <= rx2 - 40]
                if not cuts:
                    new_regions.append(region)
                    continue

                # 按切点把原框切成多段子框
                sub_boxes = []
                _prev = rx1
                for c in cuts:
                    sub_boxes.append([_prev, ry1, c, ry2])
                    _prev = c
                sub_boxes.append([_prev, ry1, rx2, ry2])

                # 构造子 region（继承原 type），order 用负整数，与原 order 不冲突
                try:
                    for sb in sub_boxes:
                        new_regions.append(_LR_Cls(
                            type=region.type,
                            bbox=sb,
                            order=-_neg_seq,
                            html=getattr(region, "html", None),
                            crop=None,
                        ))
                        _neg_seq += 1
                    # 标记原 region.order 跳过（不 OCR 原大框）
                    if isinstance(region.order, int):
                        skip_orders.add(region.order)
                    split_count += 1
                    logger.info(
                        "拆分跨栏大框[%s] order=%s bbox=%s(w_ratio=%.2f,跨中线=%s) → %d 个子框 %s",
                        region.type, region.order, region.bbox, rw_ratio, cross_midline,
                        len(sub_boxes), sub_boxes,
                    )
                except Exception as _e:
                    # 构造失败（例如 _LR_Cls 构造签名不同）：回退不拆
                    logger.warning(
                        "跨栏大框拆分失败（回退不拆，保留原框可能串栏）: %s", _e
                    )
                    new_regions.append(region)
            if split_count > 0:
                layout_res.regions = new_regions
                logger.info(
                    "跨栏大框拆分完成: %d 个大框 → %d 个子框（按栏缝切分，避免单框内左右栏混读）",
                    split_count, len(new_regions),
                )

        # 预计算文本类区域的重叠关系，跳过被高度覆盖的重复区域
        # 版面分析有时会返回重叠区域对：
        #   1. 同一区域同时识别为 text 和 reference（bbox 几乎完全重叠）
        #   2. 横跨多栏的大区域 + 单栏小区域（部分重叠，IoU 0.3~0.5）
        # 第 2 种情况原阈值 iou>0.5/cover_j>0.7 漏判，导致同一块文字被两个区域
        # 各 OCR 一次：小区域识别出完整文本，大区域因左/右边界被切识别出被
        # 裁剪的子串，最终产生"嵌套子串"或"完全重影"重复（详见 _diag_dup_bug.py）。
        # 修复：降低阈值 + 加 cover_i（原代码只看 j 被 i 覆盖，漏判 i 被 j 大量
        # 包含）+ 保留更小的区域（更精确，由补充 OCR 补回大区域独有部分）。
        text_regions = [r for r in layout_res.regions if r.type in text_types]
        # 注意：skip_orders 已在上方跨栏拆分阶段写入部分大框的 order
        for i, ri in enumerate(text_regions):
            if ri.order in skip_orders:
                continue
            ix1, iy1, ix2, iy2 = ri.bbox
            i_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if i_area == 0:
                continue
            for j in range(i + 1, len(text_regions)):
                rj = text_regions[j]
                if rj.order in skip_orders:
                    continue
                jx1, jy1, jx2, jy2 = rj.bbox
                # 交集
                ox1 = max(ix1, jx1)
                oy1 = max(iy1, jy1)
                ox2 = min(ix2, jx2)
                oy2 = min(iy2, jy2)
                if ox2 <= ox1 or oy2 <= oy1:
                    continue
                inter_area = (ox2 - ox1) * (oy2 - oy1)
                j_area = max(0, jx2 - jx1) * max(0, jy2 - jy1)
                if j_area == 0:
                    continue
                # IoU 和双向覆盖比例
                iou = inter_area / (i_area + j_area - inter_area)
                cover_i = inter_area / i_area  # i 被 j 覆盖的比例
                cover_j = inter_area / j_area  # j 被 i 覆盖的比例
                # 阈值降低 + 加 cover_i 判断（覆盖小区域被大区域包含的情况）
                if iou > 0.3 or cover_j > 0.5 or cover_i > 0.5:
                    # title 区域优先级最高，不被跳过
                    # 选择跳过更大的区域（横跨多栏易产生被裁剪的子串），
                    # 保留更小的区域（更精确），让补充 OCR 补回大区域独有部分
                    if rj.type == "title" and ri.type != "title":
                        # j 是 title 优先保留，跳过 i
                        skip_orders.add(ri.order)
                        logger.info(
                            "跳过重叠区域: order=%d type=%s bbox=%s "
                            "(与 title order=%d bbox=%s IoU=%.2f cover_i=%.2f cover_j=%.2f)",
                            ri.order, ri.type, ri.bbox,
                            rj.order, rj.bbox, iou, cover_i, cover_j,
                        )
                        break  # i 已被跳过，跳出内层循环
                    elif ri.type == "title" and rj.type != "title":
                        # i 是 title 优先保留，跳过 j
                        skip_orders.add(rj.order)
                        logger.info(
                            "跳过重叠区域: order=%d type=%s bbox=%s "
                            "(与 title order=%d bbox=%s IoU=%.2f cover_i=%.2f cover_j=%.2f)",
                            rj.order, rj.type, rj.bbox,
                            ri.order, ri.bbox, iou, cover_i, cover_j,
                        )
                    elif i_area >= j_area:
                        # 都非 title 或都是 title：跳过更大的 i
                        skip_orders.add(ri.order)
                        logger.info(
                            "跳过重叠区域(大): order=%d type=%s bbox=%s 面积=%d "
                            "(与 order=%d type=%s bbox=%s 面积=%d IoU=%.2f cover_i=%.2f cover_j=%.2f，"
                            "保留更小区域，由补充OCR补回大区域独有部分)",
                            ri.order, ri.type, ri.bbox, i_area,
                            rj.order, rj.type, rj.bbox, j_area,
                            iou, cover_i, cover_j,
                        )
                        break  # i 已被跳过，跳出内层循环
                    else:
                        # 跳过更大的 j
                        skip_orders.add(rj.order)
                        logger.info(
                            "跳过重叠区域(大): order=%d type=%s bbox=%s 面积=%d "
                            "(与 order=%d type=%s bbox=%s 面积=%d IoU=%.2f cover_i=%.2f cover_j=%.2f，"
                            "保留更小区域，由补充OCR补回大区域独有部分)",
                            rj.order, rj.type, rj.bbox, j_area,
                            ri.order, ri.type, ri.bbox, i_area,
                            iou, cover_i, cover_j,
                        )

        # 区域 OCR 循环：对每个文本类区域做过滤 + OCR
        # 记录被过滤跳过 / OCR 异常的区域 order：这些区域的内容没有被识别，
        # 不能算作"已处理区域"，否则补充整页 OCR 识别出的行会被误判为
        # 已处理而跳过（稀疏文字区被 L1 误杀后的漏识 bug）。
        skipped_region_orders: set = set()
        t_reg_start = _time.time()
        reg_ocr_count = 0
        # 收集表格区域（bbox + html），供 PDF 渲染用 insert_htmlbox
        page_tables: List[dict] = []
        # 页级边界标记：每页只允许一次 new_page=True（传给子进程 OCR 计页）。
        # 第一次 OCR 调用（第一个区域）标记新页；补充整页 OCR 若在补充开启且
        # 前面已标记过（区域>=1）则不再标记，避免一页计数两次。
        _page_marked = False
        for region in layout_res.regions:
            if region.type not in text_types:
                logger.info("跳过非文本区域: type=%s bbox=%s", region.type, region.bbox)
                continue
            if region.order in skip_orders:
                continue

            # 累加文本区域面积
            rx1, ry1, rx2, ry2 = region.bbox
            text_region_area += max(0, (rx2 - rx1)) * max(0, (ry2 - ry1))

            # ==================================================================
            # 【修复 V2 补 1】预计算"真实裁剪范围（含 padding）"
            # 必须放在最前面，保证两条路径（table SLANet / 普通区域 OCR）都能
            # 正确取到同一个 padded bbox 用于 supplement_ocr processed_bbox。
            # ==================================================================
            is_title = region.type == "title"
            is_table = region.type == "table"
            if is_title:
                pad_x, pad_top, pad_bot = TITLE_PAD_X, TITLE_PAD_TOP, TITLE_PAD_BOT
            elif is_table:
                pad_x, pad_top, pad_bot = 100, 20, 20
            else:
                pad_x, pad_top, pad_bot = 10, 10, 10
            px1 = max(0, rx1 - pad_x)
            py1 = max(0, ry1 - pad_top)
            px2 = min(w, rx2 + pad_x)
            py2 = min(h, ry2 + pad_bot)
            _this_region_padded_bbox = [px1, py1, px2, py2]
            _region_processed_ok = False  # 成功后置 True

            # 表格区域特殊处理：SLANet 已识别 HTML 结构
            # 修复：过大的 table 区域（极可能是目录页被误判为表格）
            # 跳过 SLANet → Markdown 单一行合并路径，改用普通逐行 OCR
            # 判定标准：区域高度 > 页面 50%，或 (宽度 > 80% 且 高度 > 30%)
            is_suspicious_fake_table = False
            if region.type == "table":
                region_h_ratio = (ry2 - ry1) / h if h > 0 else 0
                region_w_ratio = (rx2 - rx1) / w if w > 0 else 0
                if (region_h_ratio > 0.5
                        or (region_w_ratio > 0.8 and region_h_ratio > 0.3)):
                    is_suspicious_fake_table = True
                    logger.info(
                        "疑似假表格(目录页)跳过SLANet: order=%d bbox=%s "
                        "h_ratio=%.2f w_ratio=%.2f → 改用普通逐行OCR",
                        region.order, region.bbox,
                        region_h_ratio, region_w_ratio,
                    )

            if region.type == "table" and region.html and not is_suspicious_fake_table:
                md_text = _html_table_to_markdown(region.html)
                if md_text:
                    # 修复：不再把整个表格作为单一 OCRLine（导致 PDF 文字层定位错误）
                    # 改为按 Markdown 行拆分，每行按高度均分坐标，
                    # 保证 PDF 可搜索层 / TXT 输出正常。
                    md_lines = [ln for ln in md_text.split("\n") if ln.strip()]
                    added_lines_before = len(all_lines)
                    if md_lines:
                        # 按行数均分表格高度，每行一段 y 区间
                        tbl_h = ry2 - ry1
                        n = len(md_lines)
                        line_h = tbl_h / max(n, 1)
                        for li, md_line in enumerate(md_lines):
                            ly1 = ry1 + int(li * line_h)
                            ly2 = ry1 + int((li + 1) * line_h)
                            coords = [
                                (rx1, ly1), (rx2, ly1),
                                (rx2, ly2), (rx1, ly2),
                            ]
                            all_lines.append(
                                OCRLine(text=md_line, coords=coords, confidence=0.95)
                            )
                    page_tables.append({
                        "bbox": [rx1, ry1, rx2, ry2],
                        "html": region.html,
                    })
                    added_count = len(all_lines) - added_lines_before
                    logger.info(
                        "表格区域(SLANet) order=%d bbox=%s | HTML %d 字符 → 拆分为 %d 个独立OCRLine",
                        region.order, region.bbox,
                        len(region.html), added_count,
                    )
                    _region_processed_ok = True
                    # 表格 Markdown 拆分成功：把真实 padded bbox 加入 processed_bboxes
                    processed_bboxes.append(_this_region_padded_bbox)
                    continue

            region_img = image[py1:py2, px1:px2]
            if region_img is None or region_img.size == 0:
                continue

            # 对区域裁剪图做过滤 + OCR
            # force_ocr=True（无文本层扫描件整页强制 OCR）：区域级同样跳过
            # L1/L2 过滤，避免稀疏文字区域（大 bbox 含大片空白导致边缘密度低）
            # 被 L1 误判为"无文字"而漏识。整页已判定必须识别，其子区域亦然。
            if not force_ocr:
                should, sub_reason = self.flt.should_ocr(region_img, slot=slot)
                if not should:
                    logger.info(
                        "区域过滤跳过 [%s] order=%d: %s",
                        region.type, region.order, sub_reason,
                    )
                    skipped_region_orders.add(region.order)
                    continue

            try:
                sub_result = self.ocr.recognize(
                    region_img, slot=slot, new_page=not _page_marked,
                )
                if not _page_marked:
                    _page_marked = True
                reg_ocr_count += 1
            except Exception as e:
                logger.warning("区域 OCR 异常 [%s] order=%d: %s", region.type, region.order, e)
                skipped_region_orders.add(region.order)
                continue

            # 记录该区域在 all_lines 中的起始索引（用于跨行标题分组）
            start_idx = len(all_lines)

            # 坐标偏移：把区域内的坐标转换回原页坐标系
            x_off, y_off = px1, py1
            for line in sub_result.lines:
                shifted_coords = [(x + x_off, y + y_off) for x, y in line.coords]

                if is_title:
                    # title 区域：直接添加（只含第一行）
                    all_lines.append(
                        OCRLine(text=line.text, coords=shifted_coords, confidence=line.confidence)
                    )
                else:
                    # text 区域：跳过 title 第一行碎片（padding 裁剪造成的）
                    if _is_title_fragment(line.coords, x_off, y_off):
                        logger.info("text 行是 title 碎片，跳过: %r", line.text)
                        continue
                    # 检测是否是标题续行（每个 title 只匹配第一个续行）
                    title_idx = _find_title_continuation(line.coords, x_off, y_off)
                    if title_idx >= 0 and title_has_cont[title_idx]:
                        title_idx = -1  # 该 title 已有续行，不再匹配
                    line_idx = len(all_lines)
                    all_lines.append(
                        OCRLine(text=line.text, coords=shifted_coords, confidence=line.confidence)
                    )
                    if title_idx >= 0:
                        # 标记为标题续行：记录行索引，加入对应 title 的分组
                        title_line_indices[title_idx].append(line_idx)
                        title_has_cont[title_idx] = True
                        logger.info(
                            "检测到标题续行 (title#%d): %r",
                            title_idx, line.text,
                        )

            # 记录 title 区域自身的行索引
            end_idx = len(all_lines)
            if is_title:
                # 找到对应的 title 索引（通过 bbox 匹配）
                for ti, tb in enumerate(title_bboxes):
                    if tb[0] == rx1 and tb[1] == ry1 and tb[2] == rx2 and tb[3] == ry2:
                        # title 行插到分组开头（续行已在循环中追加到末尾）
                        title_line_indices[ti] = list(range(start_idx, end_idx)) + title_line_indices[ti]
                        break

            # ==================================================================
            # 【修复 V2 补 1 续】—— 普通区域 OCR 成功：把 padded bbox 登记为 processed
            # 注意：table HTML 路径已在 continue 前登记；本路径能走到这里，说明
            # 前面的 try/except OCR 调用已成功（否则会走 continue/break）。
            # ==================================================================
            processed_bboxes.append(_this_region_padded_bbox)

        # 构建 line_groups：title 有续行时（行数 > 1），合并为同一逻辑行
        # 文本输出（txt/markdown）合并为一行；searchable PDF 仍按物理行定位
        for indices in title_line_indices:
            if len(indices) > 1:
                line_groups.append(indices)
                logger.info("跨行标题合并: 行索引=%s 物理行=%d", indices, len(indices))

        # 覆盖率回退：文本区域面积占比过低，说明版面分析未识别出足够的文字区域
        # 典型场景：屏幕截图、UI 截图、海报等非文档图片
        # 正常文档扫描件的文本区域覆盖率通常 >30%，15% 是安全下限
        coverage = text_region_area / total_area if total_area > 0 else 0
        if coverage < 0.15:
            logger.info(
                "版面分析文本区域覆盖率 %.1f%% 过低（阈值 15%%），回退整页 OCR",
                coverage * 100,
            )
            return None, timings

        if not all_lines:
            logger.info("版面分析未识别到文字行，回退整页 OCR")
            return None, timings

        # 补充识别：版面分析可能遗漏部分区域（如双栏文档只切出一栏）
        # 对整页做一次 OCR，把不在任何已处理区域内的行补充进来
        #
        # 【修复 V2 - processed_bbox 是真实裁剪范围（含 padding）】
        # processed_bboxes 已在上方 region 循环前声明为空，并在循环内每次成功
        # 处理一个 region 后 append(px1,py1,px2,py2)。这里直接复用，不再重置。

        # 整页 OCR 补充
        t_reg_total = _time.time() - t_reg_start
        timings["recog"] = t_reg_total
        t_sup_start = _time.time()
        if self.supplement_ocr:
            try:
                # 补充整页 OCR 是一页最后一次（也常是唯一一次）整图识别，
                # 若前面的区域 OCR 尚未标记新页（区域全被过滤/跳过），
                # 在此补上页级边界标记，保证每页恰好计数一次
                full_result = self.ocr.recognize(
                    image, slot=slot, new_page=not _page_marked,
                )
            except Exception as e:
                logger.warning("补充识别整页 OCR 异常: %s", e)
                full_result = None
        else:
            # 已关闭补充识别（layout_supplement_ocr=false）：跳过整页 OCR
            full_result = None
            logger.info(
                "已关闭补充整页 OCR（layout_supplement_ocr=false）"
                "| 区域OCR×%d %.1fs，跳过整页补充 %.1fs 预算",
                reg_ocr_count, t_reg_total, 0.0,
            )
        t_sup = _time.time() - t_sup_start
        timings["sup"] = t_sup
        logger.info(
            "版面分析耗时分解: PPStructure %.1fs + 区域OCR×%d %.1fs + 补充整页OCR %.1fs = %.1fs",
            t_la, reg_ocr_count, t_reg_total, t_sup, t_la + t_reg_total + t_sup,
        )

        if full_result and full_result.lines:
            # 已有行的文本列表（用于相似度+子串去重，不能只用 exact match）
            # 用户反馈同一段话出现 2-3 次（每次截断位置不同），这就是原因：
            # supplement 的整页 OCR 结果与区域 OCR 结果可能差几个字但本质是
            # 同一句话碎片（例："县二院为病房大楼；" vs "病房大楼；县中医院..."）
            existing_texts_stripped = [ln.text.strip() for ln in all_lines]
            MIN_COMMON_CHARS = 10  # 10 个连续公共汉字 → 极大概率来自同一句原文

            def _has_long_common_substr(a: str, b: str) -> bool:
                """判断两个字符串是否有 >= MIN_COMMON_CHARS 的连续公共子串。

                用 O(n*m) 的二维 DP，实际 n,m < 200（单行 OCR 文本不长），
                supplement_ocr 每次最多 50-100 行，计算量可忽略。
                """
                if not a or not b:
                    return False
                la, lb = len(a), len(b)
                if la < MIN_COMMON_CHARS or lb < MIN_COMMON_CHARS:
                    return False
                # 早期跳出：任一方向的子串包含直接返回 True
                # （快速路径，避免跑 DP）
                if a in b or b in a:
                    return max(la, lb) >= MIN_COMMON_CHARS
                # 枚举较短字符串的所有长度=MIN_COMMON_CHARS 的窗口，检查在另一串是否存在
                # 比经典 DP 快（实际 OCR 行中，只要有一个窗口命中就说明高度同源）
                short, long = (a, b) if la <= lb else (b, a)
                for i in range(0, len(short) - MIN_COMMON_CHARS + 1):
                    window = short[i:i + MIN_COMMON_CHARS]
                    if window in long:
                        return True
                return False

            added = 0
            skipped_bbox = 0
            skipped_similar = 0
            for line in full_result.lines:
                # ---- 判定 1：bbox 中心点是否落入 processed_bboxes ----
                xs = [p[0] for p in line.coords]
                ys = [p[1] for p in line.coords]
                cx = (min(xs) + max(xs)) / 2
                cy = (min(ys) + max(ys)) / 2
                in_processed = False
                for bx1, by1, bx2, by2 in processed_bboxes:
                    # 【V2 修复 1】padding 容差从 20px 提升到 40px
                    # 因为 processed_bboxes 现在本身就含 pad_x/pad_top/pad_bot，
                    # 基本覆盖了区域；再给 40px 容忍相邻栏的边界字符错位。
                    if (bx1 - 40 <= cx <= bx2 + 40 and
                            by1 - 40 <= cy <= by2 + 40):
                        in_processed = True
                        break
                if in_processed:
                    skipped_bbox += 1
                    continue
                # ---- 判定 2：相似度去重（同源句子碎片必命中）----
                cand = line.text.strip()
                if not cand:
                    continue
                is_dup = False
                for exist in existing_texts_stripped:
                    if not exist:
                        continue
                    # 2a. 快速完全相等（老逻辑保留）
                    if cand == exist:
                        is_dup = True
                        break
                    # 2b. 长公共子串：10+ 连续汉字重合 → 同一句子不同截断
                    if _has_long_common_substr(cand, exist):
                        is_dup = True
                        break
                if is_dup:
                    skipped_similar += 1
                    continue
                # ---- 通过，追加为新行 ----
                all_lines.append(
                    OCRLine(
                        text=line.text,
                        coords=line.coords,
                        confidence=line.confidence,
                    )
                )
                existing_texts_stripped.append(cand)
                added += 1
            if added > 0 or skipped_bbox > 0 or skipped_similar > 0:
                logger.info(
                    "补充识别(V2): 新增%d行 | 已在processed区域跳过%d行 | 相似度去重跳过%d行",
                    added, skipped_bbox, skipped_similar,
                )

        # 多栏排序：双栏文档的版面分析路径行 + 补充行混合后，按分栏阅读顺序重排
        # 检测分栏：用已处理区域的 x 范围，找不重叠的栏
        t_reorder_start = _time.time()
        all_lines, line_groups = self._reorder_by_columns(
            all_lines, line_groups, layout_res.regions, w, skip_orders, text_types
        )
        timings["reorder"] = _time.time() - t_reorder_start

        # 目录页碎片合并：孤立页码归位 + 散字标题合并（修复虚线引导符断行）
        all_lines, line_groups = self._merge_toc_fragments(
            all_lines, line_groups, w, h
        )

        return OCRResult(
            lines=all_lines,
            width=w,
            height=h,
            provider=self.ocr.name,
            line_groups=line_groups,
            tables=page_tables,
        ), timings

    # ------------------------------------------------------------------
    # 工具：全部文本框 + KMeans 聚类 检测栏缝（首选高精度方法）
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_column_gutters_kmeans(lines, page_w, page_h):
        """用**全部**OCR文本框的中心x坐标做 KMeans 聚类推断栏缝。

        诊断验证（样例年鉴三栏页 w=1678, 实际缝=[618,1082]）:
          - 全部文本框 + K=3 聚类: 推断缝=[612,1080], 均值误差 4px（<0.3%页宽）
          - 比"投影法"和"筛选窄框后聚类"精度高 10-30 倍。

        策略：
          1. 取全部 lines 的 cx=(x1+x2)/2 做 1D KMeans（不筛选行宽，避免误删正文）
          2. 尝试 K∈[2,3,4]，计算每个 K 的轮廓系数近似 + 簇间距合理性，选最优 K
          3. 相邻簇中心的中点即为栏缝
          4. 验证：每栏中心之间距离 >= 页宽*15%，且每簇样本数 >= 总样本10%

        返回: 栏缝x坐标列表（按升序，已过滤 8%~92% 页宽范围）。
        """
        if not lines or page_w <= 0:
            return []
        # 提取行 cx，并区分"栏内窄行"与"跨栏宽行"。
        # 跨栏行（页眉/页脚/跨栏标题，行宽>=50%页宽）的 cx 落在栏缝或跨栏位置，
        # 会把 K=3 的簇中心拉偏导致误判 K=2（实测单页三栏页 13 跨栏行 → 误判 2 栏，
        # 中栏+右栏合并按 y 混排 → PDF 复制跨栏）。
        # 修复：KMeans 仅用栏内窄行聚类（栏内行宽约 20-33% 页宽，<50%），
        # 跨栏行不参与栏缝推断。窄行不足 8 时回退全部行（保持原逻辑）。
        narrow_cxs = []   # 栏内窄行 cx（用于 KMeans）
        all_cxs = []      # 全部行 cx（回退用）
        WIDE_THRESHOLD = page_w * 0.50  # 行宽 >= 50% 页宽视为跨栏行
        for ln in lines:
            xs = [p[0] for p in ln.coords]
            if not xs:
                continue
            lx1, lx2 = min(xs), max(xs)
            cx = (lx1 + lx2) / 2
            all_cxs.append(cx)
            if (lx2 - lx1) < WIDE_THRESHOLD:
                narrow_cxs.append(cx)
        # 优先用窄行；不足 8 则回退全部行
        cxs = narrow_cxs if len(narrow_cxs) >= 8 else all_cxs
        if len(cxs) < 8:
            return []
        data = np.array(cxs, dtype=np.float64).reshape(-1, 1)
        total = len(cxs)

        # 尝试 sklearn KMeans，不可用时退化为等间距启发（兜底）
        try:
            from sklearn.cluster import KMeans
        except Exception:
            # sklearn 不可用：退化为"按分位数均分"启发式（非理想但可用）
            # 尝试找双峰/三峰：按 cx 排序后找最大间隙
            # 顺序必须是 (3, 2)——先试 K=3，不通过簇间距/样本数校验再试 K=2
            # （与 sklearn 路径一致，符合项目硬约束；旧版 (2,3) 会让 K=2 先命中
            #   把三栏页从中间栏劈成两半，缝偏离 200+px）
            sorted_cxs = sorted(cxs)
            best_gaps = []
            for test_k in (3, 2):
                if total < test_k * 3:
                    continue
                splits = np.array_split(sorted_cxs, test_k)
                centers = [float(np.mean(s)) for s in splits if len(s) > 0]
                centers.sort()
                cand_gaps = []
                valid = True
                for i in range(len(centers) - 1):
                    if centers[i + 1] - centers[i] < page_w * 0.15:
                        valid = False
                        break
                    cand_gaps.append(round((centers[i] + centers[i + 1]) / 2))
                if valid:
                    # 检查每簇数量
                    col_bounds = [0] + cand_gaps + [page_w]
                    col_counts_ok = True
                    for bi in range(len(col_bounds) - 1):
                        cnt = sum(1 for cxv in sorted_cxs
                                  if col_bounds[bi] <= cxv < col_bounds[bi + 1])
                        if cnt < total * 0.08:
                            col_counts_ok = False
                            break
                    if col_counts_ok:
                        best_gaps = cand_gaps
                        break
            return [gx for gx in best_gaps
                    if page_w * 0.08 <= gx <= page_w * 0.92]

        best_gaps = []
        best_score = -1.0  # 越高越好
        # 只尝试 K=2 和 K=3（年鉴/报纸常见栏数，K=4极少且易噪声）
        for K in (3, 2):
            if total < K * 3:
                continue
            try:
                km = KMeans(n_clusters=K, n_init=5, random_state=42, max_iter=200)
                labels = km.fit_predict(data)
                centers = km.cluster_centers_.flatten().tolist()
                # 按中心 x 升序重排
                order = sorted(range(K), key=lambda i: centers[i])
                centers_sorted = [centers[i] for i in order]
                # 1. 簇间距检查：相邻中心距离 >= 页宽*15%（栏不能太窄）
                spacing_ok = True
                for i in range(K - 1):
                    if centers_sorted[i + 1] - centers_sorted[i] < page_w * 0.15:
                        spacing_ok = False
                        break
                if not spacing_ok:
                    continue
                # 2. 每簇样本数 >= 8%（避免把页眉/页脚1-2行当成独立栏）
                cluster_sizes = [0] * K
                for li in range(K):
                    cluster_sizes[li] = int(np.sum(labels == li))
                size_ok = all(cs >= total * 0.08 for cs in cluster_sizes)
                if not size_ok:
                    continue
                # 3. 簇内紧凑度评分（簇内平方和倒数归一化，越大越好）
                #    + 簇间间距奖励（中心间距乘积，越大越好）
                inertia = float(km.inertia_) / total  # 平均每点到中心的距离平方
                spread_prod = 1.0
                for i in range(K - 1):
                    spread_prod *= (centers_sorted[i + 1] - centers_sorted[i]) / page_w
                # 紧凑度：inertia 越小越好 → 评分用 1/(1+inertia)
                compactness = 1.0 / (1.0 + inertia / 1000.0)  # /1000归一化px²
                # 综合分数（紧凑度*0.5 + 间距乘积*0.5）
                score = compactness * 0.5 + spread_prod ** (1.0 / (K - 1)) * 0.5
                # 4. 栏缝中点验证：两侧至少各 8% 行
                cand_gaps = [round((centers_sorted[i] + centers_sorted[i + 1]) / 2)
                             for i in range(K - 1)]
                cand_gaps_filtered = [gx for gx in cand_gaps
                                      if page_w * 0.08 <= gx <= page_w * 0.92]
                if len(cand_gaps_filtered) != len(cand_gaps):
                    continue
                # 额外验证：对每条缝，左/右侧 cx 计数均 >=8%
                cols_bounds = [0] + cand_gaps_filtered + [page_w]
                per_col_counts_ok = True
                for bi in range(len(cols_bounds) - 1):
                    cnt = sum(1 for cxv in cxs
                              if cols_bounds[bi] <= cxv < cols_bounds[bi + 1])
                    if cnt < total * 0.08:
                        per_col_counts_ok = False
                        break
                if not per_col_counts_ok:
                    continue
                if score > best_score:
                    best_score = score
                    best_gaps = cand_gaps_filtered
            except Exception:
                continue
        return best_gaps

    # ------------------------------------------------------------------
    # 工具：整页 OCR 完成后输出版面诊断摘要日志（服务运行时可见）
    # ------------------------------------------------------------------
    @staticmethod
    def _log_layout_diagnostic(lines, page_w, page_h, page_no):
        """输出版面诊断摘要：栏缝/各栏行数/y范围/跨栏行/版面类型。

        在 PdfHandler.process 每页 OCR 完成后调用，以 INFO 日志输出，
        服务运行时可直接在控制台/日志文件中观察每页版面结构。
        跨栏行判定阈值=行宽≥45%页宽（正常栏内行宽约24-27%，跨栏行约72%）。
        诊断异常不影响主流程。
        """
        if not lines or page_w <= 0 or page_h <= 0:
            return
        try:
            # 1) 栏缝检测
            gaps = BaseHandler._detect_column_gutters_kmeans(lines, page_w, page_h)
            num_cols = len(gaps) + 1 if gaps else 1
            col_bounds = [0] + list(gaps) + [page_w]

            # 2) 按栏分组
            col_idxs = [[] for _ in range(num_cols)]
            for i, ln in enumerate(lines):
                xs = [p[0] for p in ln.coords]
                cx = (min(xs) + max(xs)) / 2.0
                ci = 0
                for gx in gaps:
                    if cx < gx:
                        break
                    ci += 1
                col_idxs[ci].append(i)

            # 3) 各栏行数 + y范围
            col_counts = []
            col_yranges = []
            for ci in range(num_cols):
                idxs = col_idxs[ci]
                col_counts.append(len(idxs))
                if idxs:
                    ys_min = [min(p[1] for p in lines[i].coords) for i in idxs]
                    ys_max = [max(p[1] for p in lines[i].coords) for i in idxs]
                    col_yranges.append(f"{min(ys_min)}-{max(ys_max)}")
                else:
                    col_yranges.append("-")

            # 4) 跨栏行检测（行宽≥45%页宽）
            width_thresh = page_w * 0.45
            fw_lines = []
            for i, ln in enumerate(lines):
                xs = [p[0] for p in ln.coords]
                lw = max(xs) - min(xs)
                if lw >= width_thresh:
                    ys = [p[1] for p in ln.coords]
                    cy = (min(ys) + max(ys)) / 2.0
                    fw_lines.append((cy, min(ys), max(ys), ln.text))
            fw_lines.sort(key=lambda t: t[0])

            # 5) 版面类型判定
            top_band = page_h / 3.0
            bot_band = page_h * 2.0 / 3.0
            has_top = any(cy < top_band for cy, _, _, _ in fw_lines)
            has_mid = any(top_band <= cy <= bot_band for cy, _, _, _ in fw_lines)
            has_bot = any(cy > bot_band for cy, _, _, _ in fw_lines)
            if num_cols <= 1:
                layout = "单栏"
            elif not fw_lines:
                layout = "标准三栏" if num_cols == 3 else f"{num_cols}栏(无跨栏)"
            else:
                parts = [f"{num_cols}栏"]
                if has_top:
                    parts.append("顶部跨栏")
                if has_mid:
                    parts.append("中部跨栏")
                if has_bot:
                    parts.append("底部跨栏")
                layout = " + ".join(parts)

            # 6) 输出摘要日志
            yranges_str = "|".join(col_yranges)
            if fw_lines:
                top_n = sum(1 for cy, _, _, _ in fw_lines if cy < top_band)
                mid_n = sum(1 for cy, _, _, _ in fw_lines if top_band <= cy <= bot_band)
                bot_n = sum(1 for cy, _, _, _ in fw_lines if cy > bot_band)
                pos_parts = []
                if top_n:
                    pos_parts.append(f"顶{top_n}")
                if mid_n:
                    pos_parts.append(f"中{mid_n}")
                if bot_n:
                    pos_parts.append(f"底{bot_n}")
                logger.info(
                    "[版面诊断] P%d: %d栏 缝x=%s 各栏=%s行 y=[%s] 跨栏=%d(%s) → %s",
                    page_no, num_cols, gaps, col_counts, yranges_str,
                    len(fw_lines), "/".join(pos_parts), layout,
                )
                # 跨栏行明细（最多5条）
                for cy, y_min, y_max, text in fw_lines[:5]:
                    y_pct = cy / page_h * 100
                    pos_tag = "顶" if cy < top_band else ("底" if cy > bot_band else "中")
                    snippet = text[:25] + "…" if len(text) > 25 else text
                    logger.info(
                        "[版面诊断]   跨栏行 y=%d-%d(%.0f%%)[%s] \"%s\"",
                        y_min, y_max, y_pct, pos_tag, snippet,
                    )
                if len(fw_lines) > 5:
                    logger.info("[版面诊断]   ...另有 %d 条跨栏行", len(fw_lines) - 5)
            else:
                logger.info(
                    "[版面诊断] P%d: %d栏 缝x=%s 各栏=%s行 y=[%s] 跨栏=0 → %s",
                    page_no, num_cols, gaps, col_counts, yranges_str, layout,
                )
        except Exception as _diag_e:
            logger.debug("[版面诊断] P%d 诊断异常(不影响主流程): %s", page_no, _diag_e)

    # ------------------------------------------------------------------
    # 工具：图像列投影检测栏缝（物理防串栏的核心）
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_column_gutters_projection(image, page_w, page_h):
        """用图像列投影检测栏缝（文字区之间的空白竖带）。

        比"OCR 行中心分布"更鲁棒：行中心法会把右栏内长短不一的行
        误判成两个簇，从而在栏内产生假栏缝（把标准三栏误判成四栏，
        第二/三栏被拦腰截断）。投影法直接统计每 x 列的暗像素数，
        栏缝 = 几乎没有文字的空白竖带，与行的长短无关，三栏年鉴页
        能稳定找到 2 条真栏缝。

        返回栏缝的 x 坐标列表（已按页面 8%~92% 过滤页眉/页脚/页边）。
        """
        gray = image
        if gray.ndim == 3:
            gray = np.mean(image, axis=2)
        text_mask = gray < 128  # 文字为暗像素
        col_proj = text_mask.sum(axis=0).astype(float)
        max_proj = float(col_proj.max()) if col_proj.size else 0.0
        # 栏缝阈值：远低于栏内峰值（栏内每列有大量文字像素，栏缝接近 0）
        gutter_thresh = max(2.0, max_proj * 0.02)
        min_gutter_w = max(15, int(page_w * 0.01))
        gutters = []
        in_gutter = False
        start = 0
        for x in range(page_w):
            low = col_proj[x] < gutter_thresh
            if low and not in_gutter:
                in_gutter = True
                start = x
            elif not low and in_gutter:
                in_gutter = False
                if x - start >= min_gutter_w:
                    gutters.append(((start + x) // 2, x - start))
        if in_gutter and page_w - start >= min_gutter_w:
            gutters.append(((start + page_w) // 2, page_w - start))
        # 约束在页面 8%~92% 内（排除页眉/页脚/页面边缘误判）
        return [gx for gx, _gw in gutters
                if page_w * 0.08 <= gx <= page_w * 0.92]

    # ------------------------------------------------------------------
    # 工具：多栏排序（支持双栏/三栏/N 栏的智能检测）
    # ------------------------------------------------------------------
    @staticmethod
    def _reorder_by_columns(
        all_lines: List[OCRLine],
        line_groups: List[List[int]],
        regions,
        page_width: int,
        skip_orders: set,
        text_types: set,
    ) -> tuple:
        """智能检测多栏布局并按阅读顺序重排行。

        支持 2 栏、3 栏乃至更多栏（年鉴/报纸常见三版拼接）。
        按"左栏从上到下 → 中栏从上到下 → 右栏从上到下"重排。

        检测方法（三级，逐步回退 + 相互验证）：
          1. 优先用版面分析区域的 x 范围，找不重叠的 x 区段作为栏
             （适用于普通页面，PPStructure 能正确切出栏块）
          2. 行 x1 分布：用 OCR 行的 x1（行起点）分布找所有显著间隙
             （版面分析失败/整页OCR回退路径的主方法）
          3. 行中心 x 坐标直方图双峰聚类（方法2阈值过严时的兜底）
             对年鉴类双栏文档特别有效：行中心天然形成两个高斯簇

        检测到 N 栏后，按栏分组并各自按 y 排序，左栏在前。
        单栏或无法判断时保持原顺序。

        line_groups 的索引会跟随重排更新。
        """
        if not all_lines:
            return all_lines, line_groups

        # 栏分界线列表（每条线是一个 x 坐标）
        column_gaps: List[tuple] = []  # [(gap_x, gap_width), ...]
        detect_source = ""

        # === 方法1：用版面分析区域 x 范围检测分栏 ===
        region_x_ranges = []
        for region in regions:
            if region.type not in text_types:
                continue
            if region.order in skip_orders:
                continue
            rx1, _, rx2, _ = region.bbox
            region_x_ranges.append((rx1, rx2))

        if region_x_ranges:
            # 区间合并找不重叠的 x 区段。
            # 改进：合并时增加"小重叠容忍"（容忍 <= 5px 的相互穿插）
            # 因为 PPStructure 有时对左右栏相邻的两个区域会切出微小重叠，
            # 导致本应分开的两栏被错误合并为一个 x 区段。
            region_x_ranges.sort()
            merged = []
            OVERLAP_TOL = 5  # 像素：<=5px 的重叠视为"几乎不重叠"
            for x1, x2 in region_x_ranges:
                if merged and x1 <= merged[-1][1] + OVERLAP_TOL:
                    # 进一步检查：若重叠 <= OVERLAP_TOL 且 两区段本身都较宽（>页面1/4），
                    # 则不合并（它们本来就是两个栏）
                    prev_w = merged[-1][1] - merged[-1][0]
                    cur_w = x2 - x1
                    overlap = merged[-1][1] - x1
                    quarter = page_width / 4
                    if overlap <= OVERLAP_TOL and prev_w > quarter and cur_w > quarter:
                        # 两个宽区的微重叠 → 是两栏，不合并
                        merged.append((x1, x2))
                    else:
                        merged[-1] = (merged[-1][0], max(merged[-1][1], x2))
                else:
                    merged.append((x1, x2))

            # 找所有显著间隙（>= 30px）作为分栏线
            if len(merged) >= 2:
                for i in range(len(merged) - 1):
                    gap = merged[i + 1][0] - merged[i][1]
                    if gap >= 30:
                        gap_x = (merged[i][1] + merged[i + 1][0]) / 2
                        column_gaps.append((gap_x, gap))
                if column_gaps:
                    detect_source = "区域x范围(改进)"

        # === 方法2（V3 首选）：全部 OCR 文本框 cx + KMeans 聚类 ===
        # 基于三栏页诊断验证：全部文本框 + K=3聚类 精度最高（误差仅4-8px），
        # 远优于行cx间隙法、投影法、双峰直方图。
        # 关键：**不筛选行宽**，保留所有行（含"正文宽行"——这是精度关键，
        # 旧版"先剔除宽行再推断"会误删70%正文行，导致推断失真）。
        if not column_gaps and len(all_lines) >= 8:
            kmeans_gaps = BaseHandler._detect_column_gutters_kmeans(
                all_lines, page_width, 0
            )
            if kmeans_gaps:
                for gx in kmeans_gaps:
                    # 估计缝宽：找缝两侧最近的 cx 差
                    lcxs = sorted(
                        (min(p[0] for p in ln.coords) + max(p[0] for p in ln.coords)) / 2
                        for ln in all_lines
                    )
                    est_w = 0
                    for i in range(1, len(lcxs)):
                        if lcxs[i - 1] < gx <= lcxs[i]:
                            est_w = max(1, int(lcxs[i] - lcxs[i - 1]))
                            break
                    column_gaps.append((gx, est_w if est_w > 0 else 50))
                if column_gaps:
                    detect_source = "KMeans-全部文本框cx"
                    logger.info(
                        "分栏检测(KMeans-全部cx): 检测到 %d 条分栏线 %s",
                        len(column_gaps),
                        [f"x={g[0]:.0f}({g[1]:.0f}px)" for g in column_gaps],
                    )

        # === 方法3：用 OCR 行的 x 中心（而非 x1）分布检测分栏 ===
        # x1 是行起点，行尾的短缩进会干扰分布；改用行中心 cx = (x1+x2)/2，
        # 更能反映"该行属于哪一栏"的真实位置。
        if not column_gaps and len(all_lines) >= 6:
            # 计算每行的中心 cx
            line_cxs = sorted(
                (min(p[0] for p in ln.coords) + max(p[0] for p in ln.coords)) / 2
                for ln in all_lines
            )

            # 间隙阈值：比旧版更激进，页面宽度的 8%（原为 12%）
            # 年鉴两栏间的栏间隙常见 60-120px，过严阈值会漏检
            threshold2 = max(50, page_width * 0.08)
            candidate_gaps = []
            for i in range(len(line_cxs) - 1):
                g = line_cxs[i + 1] - line_cxs[i]
                if g >= threshold2:
                    gap_x2 = (line_cxs[i] + line_cxs[i + 1]) / 2
                    # 间隙位置约束放宽（原为 15%-85%，改为 10%-90%）
                    # 有些书的边距很窄，栏缝略偏左/右
                    if page_width * 0.10 <= gap_x2 <= page_width * 0.90:
                        candidate_gaps.append((gap_x2, g, i + 1))

            # 验证每个候选分栏线：两侧至少各有 20% 的行（原为硬性 3 行）
            # 避免页眉/页脚的 3 行造成假分栏
            total = len(line_cxs)
            min_rows_per_col = max(3, int(total * 0.20))
            for gap_x2, g, split_idx in candidate_gaps:
                left_count = split_idx
                right_count = total - split_idx
                if left_count >= min_rows_per_col and right_count >= min_rows_per_col:
                    column_gaps.append((gap_x2, g))

            if column_gaps:
                detect_source = "行cx分布"
                logger.info(
                    "分栏检测(行cx分布): 检测到 %d 条分栏线 %s",
                    len(column_gaps),
                    [f"x={g[0]:.0f}({g[1]:.0f}px)" for g in column_gaps],
                )

        # === 方法4：直方图双峰聚类（方法3阈值过严时的兜底） ===
        # 思路：把所有行 cx 放入 binned 直方图（bin=30px），找"两个高度相近、
        # 中间有明显波谷"的双峰结构 → 两波峰的中线就是分栏线。
        # 对年鉴双栏文档特别鲁棒，因为行中心天然聚集在两个窄区间内。
        if not column_gaps and len(all_lines) >= 8 and page_width > 0:
            line_cxs_raw = [
                (min(p[0] for p in ln.coords) + max(p[0] for p in ln.coords)) / 2
                for ln in all_lines
            ]
            BIN = 30  # 每 bin 30 像素
            nbins = page_width // BIN + 1
            hist = [0] * nbins
            for cx in line_cxs_raw:
                b = int(cx // BIN)
                if 0 <= b < nbins:
                    hist[b] += 1
            # 平滑（3-bin 移动平均）以消除毛刺
            smoothed = hist[:]
            for b in range(1, nbins - 1):
                smoothed[b] = (hist[b - 1] + hist[b] + hist[b + 1]) / 3
            # 找两个主波峰：各自占比 >= 25%，且波峰间距 >= 页面宽度的 25%
            peak_bins = []
            for b in range(1, nbins - 1):
                if smoothed[b] >= smoothed[b - 1] and smoothed[b] >= smoothed[b + 1]:
                    if smoothed[b] >= len(all_lines) * 0.10:  # 至少 10% 的行
                        peak_bins.append((b, smoothed[b]))
            # 按高度选前 2 名
            peak_bins.sort(key=lambda x: -x[1])
            if len(peak_bins) >= 2:
                pb1, pv1 = peak_bins[0]
                pb2, pv2 = peak_bins[1]
                left_bin = min(pb1, pb2)
                right_bin = max(pb1, pb2)
                dist = (right_bin - left_bin) * BIN
                # 双峰间距 >= 页面 20%，两峰高度比例 >= 0.4（矮峰不低于高峰的 40%）
                if (dist >= page_width * 0.20 and
                        min(pv1, pv2) / max(pv1, pv2) >= 0.4):
                    # 取两峰之间的最小波谷为分栏线
                    valley_bin = left_bin
                    valley_val = smoothed[left_bin]
                    for b in range(left_bin + 1, right_bin):
                        if smoothed[b] < valley_val:
                            valley_val = smoothed[b]
                            valley_bin = b
                    gap_x = (valley_bin + 0.5) * BIN
                    # 验证分栏线两侧行数
                    left_c = sum(1 for cx in line_cxs_raw if cx < gap_x)
                    right_c = sum(1 for cx in line_cxs_raw if cx >= gap_x)
                    min_rc = max(3, int(len(all_lines) * 0.15))
                    if left_c >= min_rc and right_c >= min_rc:
                        column_gaps.append((gap_x, dist))
                        detect_source = "直方图双峰聚类"
                        logger.info(
                            "分栏检测(双峰聚类): 波峰bin %d/%d 谷bin %d "
                            "分栏线x=%.0f 左%d行 右%d行",
                            left_bin, right_bin, valley_bin,
                            gap_x, left_c, right_c,
                        )

        if not column_gaps:
            return all_lines, line_groups

        # 用分栏线把行分为 N 栏（按中心点归类）
        # 分栏线排序后，行根据中心点落在哪个区间决定属于哪一栏
        sorted_gaps = sorted(column_gaps, key=lambda g: g[0])
        gap_xs = [g[0] for g in sorted_gaps]
        # 栏数 = 分栏线数 + 1
        num_cols = len(gap_xs) + 1

        # 每栏收集 (y_center, original_index, line)
        columns: List[List[tuple]] = [[] for _ in range(num_cols)]
        # 跨栏行：横跨多栏的页眉/标题（中心点在分栏线但本身很宽，接近页面宽度）
        # 这些行如果简单归到左栏，会打乱右栏的顺序；正确做法是放在其纵向位置
        # 所对应的局部阅读流中。
        cross_lines: List[tuple] = []  # (cy, original_index, line)

        for i, line in enumerate(all_lines):
            xs = [p[0] for p in line.coords]
            ys = [p[1] for p in line.coords]
            lx1, lx2 = min(xs), max(xs)
            cx = (lx1 + lx2) / 2
            cy = (min(ys) + max(ys)) / 2
            # 跨栏判定：行宽 >= 页面 70%，且跨越至少 1 条分栏线
            line_w = lx2 - lx1
            if line_w >= page_width * 0.70 and num_cols >= 2:
                # 检查是否至少跨越一条 gap_x（即 gap_x 在 lx1 和 lx2 之间）
                crosses_any = any(lx1 < gx < lx2 for gx in gap_xs)
                if crosses_any:
                    cross_lines.append((cy, i, line))
                    continue
            # 找到 cx 落在第几栏
            col_idx = 0
            for gx in gap_xs:
                if cx < gx:
                    break
                col_idx += 1
            columns[col_idx].append((cy, i, line))

        # 各栏排序后按栏顺序合并（左→右）。
        # 栏内用"行带感知"排序：同一水平行带（y 区间明显重叠）的碎片按 x 排序，
        # 行带之间按 y 排序。修复页眉"·4· 样例年鉴"这类同一行被拆成多个框、
        # y 相差仅几像素时纯按 cy 排序导致同行左右碎片颠倒的问题。
        ordered = []
        col_counts = []
        for col in columns:
            ordered.extend(BaseHandler._sort_column_rows(col))
            col_counts.append(len(col))

        # 跨栏行：按其 y 值插入到 ordered 中合适的位置
        # （跨栏标题/页眉通常出现在与其纵向位置对应的阅读流中）
        if cross_lines:
            cross_lines.sort(key=lambda x: x[0])
            # 插入算法：对每个 cross_line，找到 ordered 中第一个
            # y 比它大的行，插到其前面。
            for cy_cross, old_i_cross, ln_cross in cross_lines:
                inserted = False
                for pos in range(len(ordered)):
                    if ordered[pos][0] > cy_cross:
                        ordered.insert(pos, (cy_cross, old_i_cross, ln_cross))
                        inserted = True
                        break
                if not inserted:
                    ordered.append((cy_cross, old_i_cross, ln_cross))

        # 构建索引映射：old_index -> new_index
        old_to_new = {}
        new_all_lines = []
        for new_idx, (_, old_idx, line) in enumerate(ordered):
            old_to_new[old_idx] = new_idx
            new_all_lines.append(line)

        # 更新 line_groups 索引
        new_line_groups = []
        for group in line_groups:
            new_group = [old_to_new[i] for i in group if i in old_to_new]
            if len(new_group) > 1:
                new_line_groups.append(new_group)

        cross_note = f" + {len(cross_lines)}个跨栏行" if cross_lines else ""
        logger.info(
            "多栏排序[%s]: 检测到 %d 栏（分栏线 %s），各栏行数 %s%s",
            detect_source, num_cols,
            [f"x={g:.0f}" for g in gap_xs], col_counts, cross_note,
        )

        return new_all_lines, new_line_groups

    # ------------------------------------------------------------------
    # 工具：栏内行带感知排序（同一水平带内的碎片按 x 排序）
    # ------------------------------------------------------------------
    @staticmethod
    def _sort_column_rows(col: List[tuple]) -> List[tuple]:
        """栏内排序：同一水平行带内的碎片按 x1 升序，行带之间按 cy 升序。

        col 元素为 (cy, original_index, line)。
        行带判定：按 cy 升序扫描，若该行 y 区间与当前行带的 y 区间
        重叠超过该行高度的 40%，视为同一行带（如同一行被拆成的
        "·4·" 与 "样例年鉴" 两个框，y 只差几像素）。
        正文逐行排列（行距 > 行高）时每行自成一个行带，行为与纯 y 排序一致。
        """
        if len(col) <= 1:
            return list(col)

        def _yrange(item) -> tuple:
            ys = [p[1] for p in item[2].coords]
            return min(ys), max(ys)

        items = sorted(col, key=lambda x: x[0])
        rows: List[List[tuple]] = []
        row_y1 = row_y2 = 0
        for item in items:
            y1, y2 = _yrange(item)
            lh = max(1, y2 - y1)
            if rows and y1 < row_y2 - 0.4 * lh and y2 > row_y1 + 0.4 * lh:
                rows[-1].append(item)
                row_y1 = min(row_y1, y1)
                row_y2 = max(row_y2, y2)
            else:
                rows.append([item])
                row_y1, row_y2 = y1, y2

        out: List[tuple] = []
        for row in rows:
            # 行带内按 x1（行起点）升序，恢复同行的左→右阅读顺序
            row.sort(key=lambda x: min(p[0] for p in x[2].coords))
            out.extend(row)
        return out

    # ------------------------------------------------------------------
    # 工具：目录页碎片合并（孤立页码归位 + 散字标题合并）
    # ------------------------------------------------------------------
    @staticmethod
    def _merge_toc_fragments(
        all_lines: List[OCRLine],
        line_groups: List[List[int]],
        page_width: int,
        page_height: int,
    ) -> tuple:
        """合并目录页碎片，修复 PaddleOCR 在虚线引导符处的断行。

        目录页（年鉴/书籍目录）有两个共性碎片问题：
          1. 孤立页码行：条目与页码之间的长虚线"…"被检测器视为行间隙，
             页码被单独切成一行（如"自然资源" / "41" 分成两行）。
          2. 散字标题行：板块标题（如"特载""专记"）字间距偏大，每个字
             被切成独立行。

        本方法在 _reorder_by_columns 之后对物理行做合并：
          - 孤立页码（纯数字、len<=3）→ 找同 y（±25px）左侧的条目行，
            拼到条目 text 末尾，页码行删除。
          - 散字（单字、中文/字母数字）→ 同 y（±15px）且 x 间隙 <100px
            的连续单字按 x 升序合并为一行，取首字坐标。

        目录页判定：可合并的孤立页码 >=3 个才执行页码合并，避免正文页
        偶发的数字行被误合并。散字合并条件严格，正文页几乎不触发。

        物理合并 all_lines（删除被合并行）并重建 line_groups 索引。
        PDF 文字层与 txt/markdown 输出同步生效。
        """
        import re as _re

        if not all_lines:
            return all_lines, line_groups

        def _geom(line: OCRLine):
            xs = [p[0] for p in line.coords]
            ys = [p[1] for p in line.coords]
            return (min(xs), max(xs), (min(xs) + max(xs)) / 2,
                    (min(ys) + max(ys)) / 2)

        def _is_pageno(line: OCRLine) -> bool:
            t = line.text.strip()
            if not t or len(t) > 3:
                return False
            return t.isdigit()

        def _is_single_char(line: OCRLine) -> bool:
            t = line.text.strip()
            if len(t) != 1:
                return False
            # 排除标点/符号，只合并中文、字母、数字
            return _re.match(r"[\u4e00-\u9fffA-Za-z0-9]", t) is not None

        to_delete: set = set()

        # ---- 1. 孤立页码合并 ----
        # 找所有"纯数字短行 + 同 y 左侧有条目"的配对
        pageno_pairs: List[tuple] = []  # (page_idx, target_idx)
        page_mid = page_width / 2
        for i, ln in enumerate(all_lines):
            if not _is_pageno(ln):
                continue
            x1i, x2i, cxi, cyi = _geom(ln)
            pageno_in_right = cxi > page_mid
            best_j = -1
            best_dy = 1e9
            for j, lj in enumerate(all_lines):
                if j == i:
                    continue
                if _is_pageno(lj):
                    continue
                x1j, x2j, cxj, cyj = _geom(lj)
                if x2j >= x1i:
                    continue  # 条目必须在页码左侧
                # 同栏约束：页码在右栏时只匹配右栏条目，左栏同理。
                # 阈值用 page_width*0.45 / 0.55 收紧交叉容差，避免左栏
                # 续行（作者+页码换行，cx 落在栏边界附近）被误判为右栏
                # 条目，抢走右栏条目的页码
                if pageno_in_right and cxj < page_width * 0.45:
                    continue  # 右栏页码不匹配左栏条目
                if not pageno_in_right and cxj > page_width * 0.55:
                    continue  # 左栏页码不匹配右栏条目
                dy = abs(cyj - cyi)
                if dy < 25 and dy < best_dy:
                    best_dy = dy
                    best_j = j
            if best_j >= 0:
                pageno_pairs.append((i, best_j))

        # 目录页判定：可合并页码 >=3 个才执行，避免正文页误合并
        if len(pageno_pairs) >= 3:
            for page_idx, target_idx in pageno_pairs:
                if page_idx in to_delete or target_idx in to_delete:
                    continue
                target = all_lines[target_idx]
                page_ln = all_lines[page_idx]
                sep = "" if target.text.rstrip().endswith(
                    ("…", ".", "·", "、", "—", "-")
                ) else " "
                target.text = target.text.rstrip() + sep + page_ln.text.strip()
                to_delete.add(page_idx)
            logger.info(
                "目录页碎片合并: 孤立页码归位 %d 个", len(pageno_pairs)
            )

        # ---- 2. 散字标题合并 ----
        # 同 y（±15px）且 x 间隙 <100px 的连续单字按 x 升序合并
        single_indices = [
            i for i in range(len(all_lines))
            if i not in to_delete and _is_single_char(all_lines[i])
        ]
        char_used: set = set()
        char_groups: List[List[int]] = []
        for i in single_indices:
            if i in char_used:
                continue
            _, _, cxi, cyi = _geom(all_lines[i])
            same_row = [i]
            for j in single_indices:
                if j == i or j in char_used:
                    continue
                _, _, cxj, cyj = _geom(all_lines[j])
                if abs(cyj - cyi) < 15:
                    same_row.append(j)
            if len(same_row) < 2:
                continue
            same_row.sort(key=lambda k: _geom(all_lines[k])[2])  # 按 cx 升序
            group = [same_row[0]]
            for k in same_row[1:]:
                prev_x2 = _geom(all_lines[group[-1]])[1]
                cur_x1 = _geom(all_lines[k])[0]
                if 0 < cur_x1 - prev_x2 < 100:
                    group.append(k)
                else:
                    if len(group) >= 2:
                        char_groups.append(list(group))
                    group = [k]
            if len(group) >= 2:
                char_groups.append(list(group))
            for k in same_row:
                char_used.add(k)

        if char_groups:
            for group in char_groups:
                first = group[0]
                merged_text = "".join(
                    all_lines[k].text.strip() for k in group
                )
                all_lines[first].text = merged_text
                for k in group[1:]:
                    to_delete.add(k)
            logger.info(
                "目录页碎片合并: 散字标题合并 %d 组", len(char_groups)
            )

        # ---- 3. 重建 all_lines 与 line_groups（索引重映射）----
        if not to_delete:
            return all_lines, line_groups
        old_to_new: dict = {}
        new_lines: List[OCRLine] = []
        for old_i in range(len(all_lines)):
            if old_i in to_delete:
                continue
            old_to_new[old_i] = len(new_lines)
            new_lines.append(all_lines[old_i])
        new_groups: List[List[int]] = []
        for g in line_groups:
            ng = [old_to_new[i] for i in g if i in old_to_new]
            if len(ng) > 1:
                new_groups.append(ng)
        return new_lines, new_groups
