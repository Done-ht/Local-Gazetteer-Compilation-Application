import fitz
import os
from ..compose.compose_engine import SUPPORTED_IMAGE_EXTS


class CancelledException(Exception):
    """操作被取消"""
    pass


class InsertEngine:
    """向 PDF 指定位置插入内容引擎"""

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

    def insert(self, base_pdf: str, insert_files: list[str],
               insert_page: int, output_path: str) -> str:
        """
        向 PDF 的指定位置插入图片或 PDF

        Args:
            base_pdf: 基础 PDF 文件路径
            insert_files: 要插入的文件路径列表（图片 + PDF）
            insert_page: 插入位置（0-based，第 0 页表示插在最前面）
            output_path: 输出 PDF 文件路径

        Returns:
            输出文件的绝对路径

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 参数无效
        """
        if not os.path.isfile(base_pdf):
            raise FileNotFoundError(f"文件不存在: {base_pdf}")

        if not insert_files:
            raise ValueError("没有选择要插入的内容")

        self._report(0, 1, "正在打开基础文档...")

        # 验证基础 PDF
        base_doc = fitz.open(base_pdf)
        total_base_pages = base_doc.page_count

        if insert_page < 0 or insert_page > total_base_pages:
            base_doc.close()
            raise ValueError(f"插入位置无效（文档共 {total_base_pages} 页，页码范围 1-{total_base_pages}）")

        # 验证插入文件
        valid_files = []
        for f in insert_files:
            self._check_cancelled()
            if not os.path.isfile(f):
                base_doc.close()
                raise FileNotFoundError(f"文件不存在: {f}")
            ext = os.path.splitext(f)[1].lower()
            if ext == ".pdf":
                valid_files.append(("pdf", f))
            elif ext in SUPPORTED_IMAGE_EXTS:
                valid_files.append(("image", f))
            else:
                base_doc.close()
                raise ValueError(f"不支持的文件格式: {f}")

        total_insert = len(valid_files)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self._report(0, total_base_pages + total_insert + 1, "正在构建新文档...")

        doc = fitz.open()
        try:
            step = 0

            # 1. 复制插入位置之前的基础页
            for i in range(insert_page):
                self._check_cancelled()
                doc.insert_pdf(base_doc, from_page=i, to_page=i)
                step += 1
                self._report(step, total_base_pages + total_insert + 1,
                             f"正在复制基础文档第 {i + 1}/{total_base_pages} 页...")

            # 2. 插入新内容
            for file_type, file_path in valid_files:
                self._check_cancelled()
                filename = os.path.basename(file_path)
                step += 1
                self._report(step, total_base_pages + total_insert + 1,
                             f"正在插入: {filename}")

                if file_type == "pdf":
                    src_doc = fitz.open(file_path)
                    try:
                        doc.insert_pdf(src_doc)
                    finally:
                        src_doc.close()
                else:
                    page = doc.new_page()
                    try:
                        pix = fitz.Pixmap(file_path)
                        img_w, img_h = pix.width, pix.height
                        pix = None
                    except Exception:
                        img_w, img_h = page.rect.width, page.rect.height

                    page_w, page_h = page.rect.width, page.rect.height
                    scale = min(page_w / img_w, page_h / img_h, 1.0)
                    if scale < 1.0:
                        display_w = img_w * scale
                        display_h = img_h * scale
                        x_offset = (page_w - display_w) / 2
                        y_offset = (page_h - display_h) / 2
                        rect = fitz.Rect(x_offset, y_offset, x_offset + display_w, y_offset + display_h)
                    else:
                        rect = page.rect

                    page.insert_image(rect, filename=file_path)

            # 3. 复制插入位置之后的基础页
            for i in range(insert_page, total_base_pages):
                self._check_cancelled()
                doc.insert_pdf(base_doc, from_page=i, to_page=i)
                step += 1
                self._report(step, total_base_pages + total_insert + 1,
                             f"正在复制基础文档第 {i + 1}/{total_base_pages} 页...")

            base_doc.close()

            self._report(step, total_base_pages + total_insert + 1, "正在保存 PDF...")
            doc.save(output_path, garbage=4, deflate=True)
        finally:
            doc.close()

        abs_path = os.path.abspath(output_path)
        self._report(total_base_pages + total_insert + 1,
                     total_base_pages + total_insert + 1,
                     f"插入完成: {os.path.basename(abs_path)}")
        return abs_path