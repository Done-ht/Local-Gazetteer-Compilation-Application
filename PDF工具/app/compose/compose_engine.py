import fitz
import os


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp"}


class CancelledException(Exception):
    """操作被取消"""
    pass


class ComposeEngine:
    """图片合成 PDF 引擎"""

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

    def compose(self, image_files: list[str], output_path: str) -> str:
        """
        将多张图片合成为一个 PDF（每张图片一页）

        Args:
            image_files: 按顺序排列的图片文件路径列表
            output_path: 输出 PDF 文件路径

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式不支持
        """
        if not image_files:
            raise ValueError("没有选择要合成的图片")

        self._report(0, len(image_files), "正在验证图片...")

        valid_files = []
        for f in image_files:
            self._check_cancelled()
            if not os.path.isfile(f):
                raise FileNotFoundError(f"文件不存在: {f}")
            ext = os.path.splitext(f)[1].lower()
            if ext not in SUPPORTED_IMAGE_EXTS:
                raise ValueError(f"不支持的图片格式: {f}")
            valid_files.append(f)

        total = len(valid_files)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self._report(0, total, "开始合成...")

        doc = fitz.open()
        try:
            for i, img_path in enumerate(valid_files):
                self._check_cancelled()
                filename = os.path.basename(img_path)
                self._report(i + 1, total, f"正在处理: {filename}")

                # 创建新页面，使用图片原始尺寸（A4 以内）
                page = doc.new_page()

                # 获取图片尺寸
                try:
                    pix = fitz.Pixmap(img_path)
                    img_w, img_h = pix.width, pix.height
                    pix = None
                except Exception:
                    img_w, img_h = page.rect.width, page.rect.height

                # 计算缩放，使图片适配页面
                page_w, page_h = page.rect.width, page.rect.height
                scale = min(page_w / img_w, page_h / img_h, 1.0)
                if scale < 1.0:
                    # 图片太大，居中放置
                    display_w = img_w * scale
                    display_h = img_h * scale
                    x_offset = (page_w - display_w) / 2
                    y_offset = (page_h - display_h) / 2
                    rect = fitz.Rect(x_offset, y_offset, x_offset + display_w, y_offset + display_h)
                else:
                    rect = page.rect

                page.insert_image(rect, filename=img_path)

            self._report(total, total, "正在保存 PDF...")
            doc.save(output_path, garbage=4, deflate=True)
        finally:
            doc.close()

        abs_path = os.path.abspath(output_path)
        self._report(total, total, f"合成完成: {os.path.basename(abs_path)}")
        return abs_path