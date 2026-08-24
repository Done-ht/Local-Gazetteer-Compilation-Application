"""单页 PDF 写入器：每页保存为独立小文件，最后合并。

替代 IncrementalPdfWriter 的流式写入方案。

方案设计（解决文件大小 + 写入耗时两个问题）：

问题1 - 文件大小膨胀（100MB→3.1GB）：
  旧方案把渲染图用 PNG 编码插入，PNG 无损压缩对扫描图极差。
  新方案直接从源 PDF 复制原页面，保留原始 JPEG/JBIG2 压缩数据。

问题2 - 字体重复嵌入（每页10MB字体→272页2.7GB）：
  若每页单文件都 insert_text 叠加文字层，simsun.ttc（~10MB）会被
  嵌入每个单页文件，272 页 = 2.7GB 仅为字体。
  新方案分两阶段：
    1. 单页阶段：只复制源页面（不叠加文字层），OCR lines 存 JSON
    2. 合并阶段：合并所有单页后，在最终文件上统一叠加文字层
       字体只嵌入一次（~10MB），与页数无关

问题3 - 写入耗时随页数增长（末尾页15.5s）：
  旧 IncrementalPdfWriter 每5页 flush，close/reopen 加载全PDF元数据。
  新方案每页独立文件，写入开销恒定（~13ms），合并用 insert_pdf 0.6s。

最终输出大小 ≈ 源PDF + 10MB字体 + 几MB文字层 ≈ 源PDF + 15MB
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, List, Optional, Tuple

from ..core.document.base import PageResult

logger = logging.getLogger(__name__)


def _output_path(source_path: str, ext: str, output_dir: str,
                 suffix: str = "_OCR") -> str:
    """生成输出文件路径，文件名以 _ 开头（排在目录顶部，用户偏好）。"""
    base = os.path.basename(source_path)
    name, _ = os.path.splitext(base)
    out_dir = output_dir if output_dir else os.path.dirname(source_path)
    out_name = f"_{name}{suffix}.{ext.lstrip('.')}"
    return os.path.join(out_dir, out_name)


class PagePdfWriter:
    """单页 PDF 写入器：复制源PDF页面，合并后叠加文字层。

    两阶段设计：
      阶段1（append_page）：从源PDF复制原页面到独立小文件（保留原始压缩），
                            OCR lines 保存到 JSON 文件供阶段2使用
      阶段2（close）：合并所有单页 → 在合并文件上统一叠加文字层（字体只嵌入一次）

    用法：
        writer = PagePdfWriter(source_path, work_dir, render_dpi=200)
        start_page = writer.existing_pages()  # 断点续传
        for page_result in process_pages(start_page=start_page):
            writer.append_page(page_result)
        out_path = writer.close()  # 合并 + 叠加文字层
    """

    def __init__(
        self,
        source_path: str,
        work_dir: str,
        render_dpi: int = 200,
    ) -> None:
        """
        参数:
            source_path: 源 PDF 路径（用于复制原页面 + 生成输出文件名）
            work_dir: 任务工作目录，单页文件存放在其下 pages/ 子目录
            render_dpi: 渲染 DPI（用于像素→PDF 点坐标转换）
        """
        import fitz

        self._fitz = fitz
        self.source_path = source_path
        self.render_dpi = render_dpi
        self.scale = 72.0 / render_dpi
        # 最终合并输出路径（与 IncrementalPdfWriter 一致，确保兼容）
        self.out_path = _output_path(source_path, "pdf", work_dir)
        # 单页文件目录：work_dir/pages/page_XXXX.pdf
        self.pages_dir = os.path.join(work_dir, "pages")
        os.makedirs(self.pages_dir, exist_ok=True)
        self._page_count = 0
        # 中文字体缓存（合并阶段使用，只嵌入一次）
        self._font_kwargs: Optional[dict] = None
        # 源 PDF 文档句柄（延迟打开，append_page 时首次打开）
        self._src_doc: Optional[Any] = None

    def _ensure_src_doc(self):
        """延迟打开源 PDF，避免构造函数阶段 IO。"""
        if self._src_doc is None:
            self._src_doc = self._fitz.open(self.source_path)
        return self._src_doc

    def _get_font_kwargs(self) -> dict:
        """获取中文字体参数（延迟初始化）。

        合并阶段调用，字体只嵌入一次到最终文件，且用 subset_fonts() 子集化。
        只嵌入实际用到的字符，文件大小几乎不增长。

        字体查找优先级：
          1. 打包内置字体 _internal/fonts/（确保任意 Windows 可用，含英文版）
          2. Windows 系统字体 C:/Windows/Fonts/
          3. PyMuPDF 内置 china-s（最后 fallback）

        字体选择（关键）：
          - simhei.ttf（.ttf 格式）支持 subset_fonts() 子集化，子集化后
            从 9.3MB 降到几KB（只含用到的字符）
          - simsun.ttc / msyh.ttc（.ttc 格式）不支持子集化，会完整嵌入
            9-18MB，导致输出文件膨胀
          - 因此优先用 simhei.ttf，而非传统的 simsun.ttc
        """
        if self._font_kwargs is not None:
            return self._font_kwargs

        import sys

        # 打包内置字体目录（spec 中打包到 _internal/fonts/）
        bundled_font_dir = None
        if getattr(sys, "frozen", False):
            exe_dir = os.path.dirname(sys.executable)
            bundled_font_dir = os.path.join(exe_dir, "_internal", "fonts")
        else:
            # 开发环境：项目根目录下的 fonts/
            root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            bundled_font_dir = os.path.join(root, "fonts")

        # 字体候选（优先 .ttf 支持子集化，避免 .ttc 不支持）
        # 每项: (bundled_name, system_path, fontname)
        candidates = [
            ("simhei.ttf",   "C:/Windows/Fonts/simhei.ttf",   "simhei"),
            ("simkai.ttf",   "C:/Windows/Fonts/simkai.ttf",   "simkai"),
            ("STKAITI.TTF",  "C:/Windows/Fonts/STKAITI.TTF",  "stkaiti"),
            # .ttc 格式不支持子集化，仅作为最后 fallback
            ("simsun.ttc",   "C:/Windows/Fonts/simsun.ttc",   "simsun"),
            ("msyh.ttc",     "C:/Windows/Fonts/msyh.ttc",     "msyh"),
        ]
        for bundled_name, sys_path, fontname in candidates:
            # 1. 打包内置字体
            if bundled_font_dir:
                bundled_path = os.path.join(bundled_font_dir, bundled_name)
                if os.path.exists(bundled_path):
                    self._font_kwargs = {"fontfile": bundled_path, "fontname": fontname}
                    logger.info("使用中文字体(打包内置): %s (子集化=%s)",
                                fontname,
                                "是" if bundled_name.endswith(".ttf") else "否")
                    return self._font_kwargs
            # 2. 系统字体
            if os.path.exists(sys_path):
                self._font_kwargs = {"fontfile": sys_path, "fontname": fontname}
                logger.info("使用中文字体(系统): %s (子集化=%s)",
                            fontname,
                            "是" if sys_path.endswith(".ttf") else "否")
                return self._font_kwargs
        logger.warning("未找到系统中文字体，回退内置 china-s（打包后可能失效）")
        self._font_kwargs = {"fontname": "china-s"}
        return self._font_kwargs

    def existing_pages(self) -> int:
        """扫描 pages_dir 已有的单页文件数（断点续传）。

        文件名格式 page_XXXX.pdf，XXXX 为 4 位零填充页号（从 1 开始）。
        返回已有页数，上层据此跳过已处理页。
        """
        if not os.path.isdir(self.pages_dir):
            return 0
        max_page = 0
        for name in os.listdir(self.pages_dir):
            if not name.startswith("page_") or not name.endswith(".pdf"):
                continue
            try:
                num = int(name[5:-4])
                if num > max_page:
                    max_page = num
            except ValueError:
                continue
        return max_page

    def _page_file(self, page_no: int) -> str:
        """返回指定页号的单页 PDF 文件路径。"""
        return os.path.join(self.pages_dir, f"page_{page_no:04d}.pdf")

    def _ocr_file(self, page_no: int) -> str:
        """返回指定页号的 OCR 结果 JSON 文件路径。"""
        return os.path.join(self.pages_dir, f"ocr_{page_no:04d}.json")

    def append_page(self, page: PageResult, page_no: Optional[int] = None) -> None:
        """阶段1：从源PDF复制原页面到独立文件，OCR lines 存 JSON。

        不叠加文字层（避免字体嵌入每个单页文件导致膨胀）。
        文字层在 close() 合并后统一叠加，字体只嵌入一次。

        参数:
            page: 单页处理结果（page.page_no 为源PDF中的1基页号）
            page_no: 显式指定输出页号（多线程并行处理时必传，避免_page_count竞争）
                     None时自动递增 _page_count（单线程兼容模式）
        """
        if page_no is None:
            self._page_count += 1
            page_no = self._page_count
        else:
            # 显式页号模式：更新 _page_count 为最大值（供 close 合并统计）
            if page_no > self._page_count:
                self._page_count = page_no

        fitz = self._fitz
        src_doc = self._ensure_src_doc()
        src_page_idx = page.page_no - 1
        if src_page_idx >= src_doc.page_count:
            logger.warning("源PDF页号越界: %d >= %d", src_page_idx, src_doc.page_count)
            doc = fitz.open()
            doc.new_page(width=595, height=842)
            doc.save(self._page_file(page_no))
            doc.close()
            # 保存空 OCR 结果
            self._save_ocr_json(page_no, page)
            return

        # 从源 PDF 复制该页（保留原始图片压缩数据）
        doc = fitz.open()
        try:
            doc.insert_pdf(src_doc, from_page=src_page_idx, to_page=src_page_idx)
            # 不在此叠加文字层，避免字体嵌入每个单页文件
            doc.save(self._page_file(page_no), garbage=3, deflate=True)
        finally:
            doc.close()

        # 保存 OCR 结果到 JSON，供 close() 合并后叠加文字层
        self._save_ocr_json(page_no, page)

    def _save_ocr_json(self, page_no: int, page: PageResult) -> None:
        """把 OCR lines 保存到 JSON 文件，供合并后叠加文字层。

        只保存文字层需要的字段：text + coords（已转 PDF 坐标）。
        跳过的页保存空 lines（标记 skipped）。
        表格区域保存 bbox + html，渲染时用 insert_htmlbox（非 insert_text）。
        """
        data = {
            "page_no": page_no,
            "skipped": page.skipped,
            "lines": [],
            "tables": [],
        }
        # 调试日志：记录传入时 ocr_result 的状态，定位"0 行"问题
        # 之前发现所有 JSON 都是 0 行，需确认是 OCR 真没识别到还是保存逻辑问题
        ocr_line_count = 0
        # 收集表格 bbox（原图像素坐标），用于跳过 table 区域内的行
        # table 区域的文字由 insert_htmlbox 渲染，不需要 insert_text 叠加
        table_bboxes: List[List[int]] = []
        if page.ocr_result and not page.skipped:
            # 保存表格区域（bbox + html）
            for tbl in page.ocr_result.tables:
                tbl_bbox = tbl.get("bbox")
                tbl_html = tbl.get("html", "")
                if tbl_bbox and tbl_html:
                    # bbox 转换到 PDF 坐标系
                    tx0, ty0, tx1, ty1 = tbl_bbox
                    data["tables"].append({
                        "rect": [
                            tx0 * self.scale,
                            ty0 * self.scale,
                            tx1 * self.scale,
                            ty1 * self.scale,
                        ],
                        "html": tbl_html,
                    })
                    table_bboxes.append(tbl_bbox)

            ocr_line_count = len(page.ocr_result.lines)
            for line in page.ocr_result.lines:
                if not line.coords or not line.text:
                    continue
                xs = [p[0] for p in line.coords]
                ys = [p[1] for p in line.coords]
                lx0, ly0 = min(xs), min(ys)
                lx1, ly1 = max(xs), max(ys)
                # 跳过 table 区域内的行（表格文字由 insert_htmlbox 渲染）
                # 检查行中心点是否在某个 table bbox 内
                cx = (lx0 + lx1) / 2
                cy = (ly0 + ly1) / 2
                in_table = False
                for tx0, ty0, tx1, ty1 in table_bboxes:
                    if tx0 <= cx <= tx1 and ty0 <= cy <= ty1:
                        in_table = True
                        break
                if in_table:
                    continue
                # 坐标转换到 PDF 坐标系（原图像素 → PDF 点）
                x0 = lx0 * self.scale
                y1 = ly1 * self.scale  # 基线（底部）
                y0 = ly0 * self.scale
                font_size = max((y1 - y0) * 0.9, 6)
                data["lines"].append({
                    "text": line.text,
                    "x": x0,
                    "y": y1,  # insert_text 的 origin 是基线左下角
                    "size": font_size,
                })
        # 调试：保存时记录原始行数和实际写入行数，便于定位差异
        if ocr_line_count > 0 or page.skipped:
            logger.debug(
                "保存OCR JSON 页 %d: skipped=%s, ocr_result.lines=%d, 写入=%d, 表格=%d",
                page_no, page.skipped, ocr_line_count, len(data["lines"]),
                len(data["tables"]),
            )
        elif ocr_line_count == 0 and not page.skipped:
            # 有 ocr_result 但 0 行：可能是版面回退后整页OCR也返回空
            logger.warning(
                "保存OCR JSON 页 %d: ocr_result 存在但 0 行（版面回退+整页OCR均无结果）",
                page_no,
            )
        try:
            with open(self._ocr_file(page_no), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            logger.warning("保存OCR结果失败 (页 %d): %s", page_no, e)

    def close(self) -> str:
        """阶段2：合并所有单页 → 统一叠加文字层 → 保存最终文件。

        字体在合并文件中只嵌入一次，与页数无关。
        清理单页文件目录并关闭源 PDF 句柄。

        返回: 最终输出文件路径（self.out_path）
        """
        # _merge_and_save 返回页数（int），不是路径
        # 输出路径就是 self.out_path，直接返回它
        self._merge_and_save(self.out_path, keep_pages=False)
        self._cleanup_pages_dir()
        self.close_src()
        return self.out_path

    def finalize_partial(self, out_path: Optional[str] = None) -> Tuple[str, int]:
        """提前合并当前已处理页为最终 PDF，保留单页文件以便任务继续运行。

        用于运行中任务「提前导出已识别部分」：
          - 不清理 pages_dir（任务可能还在写入新页）
          - 不关闭源 PDF 句柄（任务还需要复制后续页）
          - 输出路径默认 out_path 同目录下的 _name_partial.pdf

        参数:
            out_path: 自定义输出路径，None 则生成默认 partial 路径

        返回:
            (输出文件路径, 已合并页数)
        """
        if out_path is None:
            # 默认输出路径：与 out_path 同目录，文件名加 _partial 后缀
            base = os.path.dirname(self.out_path)
            src_basename = os.path.basename(self.source_path)
            name, _ = os.path.splitext(src_basename)
            out_path = os.path.join(base, f"_{name}_partial_OCR.pdf")
        merged = self._merge_and_save(out_path, keep_pages=True)
        return out_path, merged

    def _merge_and_save(self, out_path: str, keep_pages: bool) -> int:
        """合并所有单页 → 统一叠加文字层 → 保存到指定路径。

        参数:
            out_path: 最终输出文件路径
            keep_pages: True 保留 pages_dir（用于提前导出，任务继续运行）
                       False 由调用方清理（close 正常完成时）

        返回:
            实际合并的页数
        """
        fitz = self._fitz
        total_pages = self.existing_pages()

        # 无单页文件：检查是否已有合并完成的输出文件
        if total_pages == 0:
            if os.path.isfile(out_path):
                logger.info("已有合并文件，跳过合并: %s", out_path)
                return 0
            logger.warning("无单页文件可合并，输出空 PDF: %s", out_path)
            doc = fitz.open()
            doc.new_page(width=595, height=842)
            doc.save(out_path)
            doc.close()
            return 0

        # 步骤1: 合并所有单页 PDF（保留原始图片压缩）
        out_doc = fitz.open()
        merged = 0
        missing_pages = []  # 记录丢失的页号
        for page_no in range(1, total_pages + 1):
            page_file = self._page_file(page_no)
            if not os.path.isfile(page_file):
                logger.warning("单页文件丢失，插入空白页: %s", page_file)
                missing_pages.append(page_no)
                out_doc.new_page(width=595, height=842)
                continue
            src = fitz.open(page_file)
            out_doc.insert_pdf(src)
            src.close()
            merged += 1
        logger.info("已合并 %d 页", merged)
        # 有丢失页时记录警告（仍生成残缺 PDF，但让用户知道有问题）
        if missing_pages:
            missing_count = len(missing_pages)
            # 只显示前 10 个页号，避免日志过长
            preview = missing_pages[:10]
            preview_str = ",".join(str(p) for p in preview)
            if missing_count > 10:
                preview_str += f"... 等共 {missing_count} 页"
            logger.warning(
                "合并完成但有 %d 页丢失（已插入空白页）: %s",
                missing_count, preview_str,
            )
            # 写入进度日志，便于事后排查
            from ..utils.progress_log import log_progress
            try:
                log_progress(
                    f"⚠ 合并警告: {missing_count} 页丢失已补空白页 ({preview_str})"
                )
            except Exception:
                pass

        # 步骤2: 在合并文件上统一叠加文字层（字体只嵌入一次）
        font_kwargs = self._get_font_kwargs()
        total_lines = 0
        total_tables = 0
        for page_no in range(1, merged + 1):
            ocr_file = self._ocr_file(page_no)
            if not os.path.isfile(ocr_file):
                continue
            try:
                with open(ocr_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning("读取OCR结果失败 (页 %d): %s", page_no, e)
                continue
            if data.get("skipped"):
                continue
            pdf_page = out_doc[page_no - 1]

            # 2a. 普通文字层：insert_text 叠加隐形文字（可搜索可选中）
            if data.get("lines"):
                # 修复"三合一"问题：多栏文档底部最后一行 Y 坐标相同时，
                # PDF 阅读器（Adobe/Foxit/PDF.js）复制时按 Y 坐标聚类为同一行，
                # 再按 X 顺序拼接，导致三栏底部被合并为一行（仅最后一行出错）。
                # 解决：对同 Y（容差 0.5pt）的多个文本块按 X 顺序累加微小 Y 偏移，
                # 使各栏 Y 产生足够差异，破坏阅读器的同行聚类。
                # 文字层为隐形（render_mode=3），Y 偏移不影响视觉，仅影响搜索高亮位置。
                # 注意：不再做全局 (y,x) 重排，否则会把"左栏整栏→右栏整栏"的正确
                # 阅读顺序重排回"先 y 后 x"，导致多栏文档左右栏按行交错（串栏）。
                # 此处仅保留下游传入的阅读顺序，只对同 Y 带内的行累加 X/Y 偏移。
                page_lines = [dict(ln) for ln in data["lines"]]
                Y_TOL = 0.5  # 同行判定容差（pt）
                # 用 y 排序副本做带聚类分组（不改变 page_lines 的插入顺序），
                # 判断哪些行属于同一水平行带，以便对同带内的行累加 Y 偏移。
                y_groups: List[List[dict]] = []
                for ln in sorted(page_lines, key=lambda l: l["y"]):
                    if y_groups and abs(ln["y"] - y_groups[-1][0]["y"]) <= Y_TOL:
                        y_groups[-1].append(ln)
                    else:
                        y_groups.append([ln])
                # 同带内按 X 排序，累加 Y 偏移破坏阅读器聚类（只改 y，不重排 page_lines）
                # 步长 = 组内最大字号 / 3，确保超过阅读器同行阈值（约字号 50%）
                # 最大累计偏移限制为一行高，避免搜索高亮跨行
                for grp in y_groups:
                    if len(grp) < 2:
                        continue
                    grp.sort(key=lambda l: l["x"])
                    n = len(grp)
                    max_size = max(l["size"] for l in grp)
                    step = min(max_size / 3, max_size / (n - 1))
                    for idx, ln in enumerate(grp):
                        ln["y"] = ln["y"] + idx * step

                for ln in page_lines:
                    try:
                        pdf_page.insert_text(
                            (ln["x"], ln["y"]),
                            ln["text"],
                            fontsize=ln["size"],
                            render_mode=3,  # 不可见但可搜索
                            **font_kwargs,
                        )
                        total_lines += 1
                    except Exception as e:
                        logger.debug("文字插入失败 (页 %d): %s", page_no, e)

            # 2b. 表格区域：insert_htmlbox 渲染 HTML 表格（保持表格结构可复制）
            # 用白色背景覆盖原图片的表格区域，然后渲染带边框的表格
            # 这样复制时能保持行列结构，而非按位置顺序的纯文本
            if data.get("tables"):
                for tbl in data["tables"]:
                    rect = fitz.Rect(tbl["rect"])
                    html = tbl["html"]
                    try:
                        # CSS: 白色背景 + 黑色边框 + 紧凑布局
                        # SLANet 返回的 HTML 已含 <table> 标签，CSS 控制样式
                        css = (
                            "body { background-color: #ffffff; margin: 0; }"
                            "table { border-collapse: collapse; width: 100%;"
                            "  font-family: simsun, serif; font-size: 8px; }"
                            "td, th { border: 0.5px solid #000; padding: 1px;"
                            "  text-align: left; }"
                        )
                        pdf_page.insert_htmlbox(rect, html, css=css, overlay=True)
                        total_tables += 1
                    except Exception as e:
                        logger.warning(
                            "表格HTML渲染失败 (页 %d, rect=%s): %s",
                            page_no, list(rect), e,
                        )
        logger.info("文字层叠加: %d 行 | 表格渲染: %d 个", total_lines, total_tables)

        # 步骤3: 字体子集化（只嵌入实际用到的字符）
        # 关键优化：simhei.ttf 子集化后从 9.3MB 降到几KB
        # 必须在 save 前调用，对已 insert_text 的字体生效
        try:
            out_doc.subset_fonts()
            logger.info("字体子集化完成")
        except Exception as e:
            logger.warning("字体子集化失败（继续保存）: %s", e)

        # 步骤4: 保存最终文件（全量保存，garbage=4 压缩）
        # keep_pages=True 时保存到临时文件再替换，避免占用正在写入的 pages_dir
        if keep_pages:
            tmp_out = out_path + ".tmp"
            out_doc.save(tmp_out, garbage=4, deflate=True)
            out_doc.close()
            os.replace(tmp_out, out_path)
        else:
            out_doc.save(out_path, garbage=4, deflate=True)
            out_doc.close()
        logger.info("最终输出: %s (%d 页)", out_path, merged)

        self._page_count = merged
        return merged

    def _cleanup_pages_dir(self) -> None:
        """清理单页文件目录（含 PDF 和 JSON）。"""
        try:
            import shutil

            shutil.rmtree(self.pages_dir, ignore_errors=True)
        except Exception:
            pass

    def close_src(self) -> None:
        """关闭源 PDF 文档句柄（处理完成后调用释放资源）。"""
        if self._src_doc is not None:
            try:
                self._src_doc.close()
            except Exception:
                pass
            self._src_doc = None

    @property
    def page_count(self) -> int:
        return self._page_count
