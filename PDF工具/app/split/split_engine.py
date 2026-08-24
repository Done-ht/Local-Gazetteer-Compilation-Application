import fitz
import os
from ..utils.pdf_utils import (
    get_pdf_page_count,
    get_pdf_bookmarks,
    validate_page_range,
)


class CancelledException(Exception):
    """操作被取消"""
    pass


class SplitEngine:
    """PDF 拆分引擎"""

    SPLIT_MODE_SINGLE = "single"       # 单页拆分
    SPLIT_MODE_RANGE = "range"         # 按范围拆分
    SPLIT_MODE_EXTRACT = "extract"     # 提取指定页

    def __init__(self, progress_callback=None, cancel_check=None):
        self._progress_callback = progress_callback
        self._cancel_check = cancel_check

    def _check_cancelled(self):
        if self._cancel_check and self._cancel_check():
            raise CancelledException("操作已被取消")

    def _report(self, current: int, total: int, message: str = ""):
        self._check_cancelled()
        if self._progress_callback:
            self._progress_callback(current, total, message)

    def split(self, input_path: str, output_dir: str, mode: str, **kwargs) -> list[str]:
        """
        拆分 PDF

        Args:
            input_path: 输入 PDF 文件路径
            output_dir: 输出目录
            mode: 拆分模式 (SPLIT_MODE_*)
            **kwargs:
                - range_text: 范围模式下的页码字符串 "1-3,5,8-10"
                - extract_pages: 提取模式下的页码列表 [0, 2, 5]

        Returns:
            输出文件路径列表

        Raises:
            FileNotFoundError: 输入文件不存在
            ValueError: 参数无效
        """
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"文件不存在: {input_path}")

        os.makedirs(output_dir, exist_ok=True)
        total_pages = get_pdf_page_count(input_path)

        if mode == self.SPLIT_MODE_SINGLE:
            return self._split_single(input_path, output_dir, total_pages)
        elif mode == self.SPLIT_MODE_RANGE:
            range_text = kwargs.get("range_text", "")
            return self._split_range(input_path, output_dir, total_pages, range_text)
        elif mode == self.SPLIT_MODE_EXTRACT:
            extract_pages = kwargs.get("extract_pages", [])
            return self._split_extract(input_path, output_dir, total_pages, extract_pages)
        else:
            raise ValueError(f"未知的拆分模式: {mode}")

    def _split_single(self, input_path: str, output_dir: str, total_pages: int) -> list[str]:
        """按单页拆分"""
        basename = os.path.splitext(os.path.basename(input_path))[0]
        output_files = []

        self._report(0, total_pages, "开始单页拆分...")

        src_doc = fitz.open(input_path)
        try:
            for i in range(total_pages):
                self._check_cancelled()
                self._report(i + 1, total_pages, f"正在拆分第 {i + 1} 页...")
                out_doc = fitz.open()
                out_doc.insert_pdf(src_doc, from_page=i, to_page=i)
                out_path = os.path.join(output_dir, f"_{basename}_第{i + 1}页.pdf")
                out_doc.save(out_path, garbage=4, deflate=True)
                out_doc.close()
                output_files.append(os.path.abspath(out_path))
        finally:
            src_doc.close()

        self._report(total_pages, total_pages, f"拆分完成，共 {len(output_files)} 个文件")
        return output_files

    def _split_range(self, input_path: str, output_dir: str, total_pages: int, range_text: str) -> list[str]:
        """按范围提取：将所有指定页（含范围）合并到同一个 PDF 文件。

        说明：早期版本对每个范围生成独立 PDF，导致用户输入 ``1-5`` 时
        虽然内部只有 1 个范围，但因为 ``output_is_dir=True`` 被打包成 zip
        下载，与用户"得到一个多页 PDF"的预期不符。现统一为：把所有指定
        页（按用户输入顺序去重后）合并为单一 PDF。
        """
        ranges = validate_page_range(range_text, total_pages)
        if not ranges:
            raise ValueError("请输入有效的页码范围")

        basename = os.path.splitext(os.path.basename(input_path))[0]
        output_files = []

        # 总页数（去重后）用于进度统计
        unique_pages: list[int] = []
        seen: set[int] = set()
        for start, end in ranges:
            for p in range(start, end + 1):
                if p not in seen:
                    seen.add(p)
                    unique_pages.append(p)
        total = len(unique_pages)

        self._report(0, total, "开始按范围提取...")

        src_doc = fitz.open(input_path)
        try:
            out_doc = fitz.open()
            for idx, page_num in enumerate(unique_pages):
                self._check_cancelled()
                self._report(idx + 1, total, f"正在提取第 {page_num + 1} 页...")
                out_doc.insert_pdf(src_doc, from_page=page_num, to_page=page_num)
            # 文件名：若只有一个范围用 "M-N页"，否则用 "提取范围"
            if len(ranges) == 1:
                start, end = ranges[0]
                page_label = f"{start + 1}-{end + 1}页" if start != end else f"第{start + 1}页"
            else:
                page_label = f"提取范围_{total}页"
            out_path = os.path.join(output_dir, f"_{basename}_{page_label}.pdf")
            out_doc.save(out_path, garbage=4, deflate=True)
            out_doc.close()
            output_files.append(os.path.abspath(out_path))
        finally:
            src_doc.close()

        self._report(total, total, f"提取完成，共 {total} 页 → 1 个文件")
        return output_files

    def _split_extract(self, input_path: str, output_dir: str, total_pages: int, extract_pages: list[int]) -> list[str]:
        """提取指定页合并为一个 PDF"""
        if not extract_pages:
            raise ValueError("请选择要提取的页码")

        for p in extract_pages:
            if p < 0 or p >= total_pages:
                raise ValueError(f"页码 {p + 1} 超出范围（文档共 {total_pages} 页）")

        basename = os.path.splitext(os.path.basename(input_path))[0]
        output_files = []

        self._report(0, 1, "正在提取指定页...")

        src_doc = fitz.open(input_path)
        try:
            out_doc = fitz.open()
            for page_num in sorted(set(extract_pages)):
                self._check_cancelled()
                out_doc.insert_pdf(src_doc, from_page=page_num, to_page=page_num)
            out_path = os.path.join(output_dir, f"_{basename}_提取页.pdf")
            out_doc.save(out_path, garbage=4, deflate=True)
            out_doc.close()
            output_files.append(os.path.abspath(out_path))
        finally:
            src_doc.close()

        self._report(1, 1, "提取完成")
        return output_files
