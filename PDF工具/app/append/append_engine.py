import fitz
import os
from ..compose.compose_engine import SUPPORTED_IMAGE_EXTS


class CancelledException(Exception):
    """操作被取消"""
    pass


class AppendEngine:
    """图片 + PDF 拼接引擎"""

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

    def append(self, input_files: list[str], output_path: str) -> str:
        """
        拼接图片和 PDF 文件为一个 PDF

        - 图片 → 每页一图
        - PDF → 直接追加页面

        Args:
            input_files: 按顺序排列的文件路径列表（图片 + PDF 混合）
            output_path: 输出 PDF 文件路径

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式不支持
        """
        if not input_files:
            raise ValueError("没有选择要拼接的文件")

        self._report(0, len(input_files), "正在验证文件...")

        valid_files = []
        for f in input_files:
            self._check_cancelled()
            if not os.path.isfile(f):
                raise FileNotFoundError(f"文件不存在: {f}")
            ext = os.path.splitext(f)[1].lower()
            if ext == ".pdf":
                valid_files.append(("pdf", f))
            elif ext in SUPPORTED_IMAGE_EXTS:
                valid_files.append(("image", f))
            else:
                raise ValueError(f"不支持的文件格式: {f}")

        total = len(valid_files)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self._report(0, total, "开始拼接...")

        doc = fitz.open()
        try:
            for i, (file_type, file_path) in enumerate(valid_files):
                self._check_cancelled()
                filename = os.path.basename(file_path)
                self._report(i + 1, total, f"正在处理: {filename}")

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

            self._report(total, total, "正在保存 PDF...")
            doc.save(output_path, garbage=4, deflate=True)
        finally:
            doc.close()

        abs_path = os.path.abspath(output_path)
        self._report(total, total, f"拼接完成: {os.path.basename(abs_path)}")
        return abs_path