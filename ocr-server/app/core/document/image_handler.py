"""图片文档处理器。

支持 jpg/jpeg/png/bmp，单张图片作为一页处理。
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from .base import BaseHandler, DocumentResult, PageResult


class ImageHandler(BaseHandler):
    """图片处理器。"""

    extensions = [".jpg", ".jpeg", ".png", ".bmp"]

    def process(self, path: str, progress_cb=None, page_cb=None, slot: int = 0) -> DocumentResult:
        # cv2.imread 不支持中文路径，用 numpy + imdecode 兜底
        try:
            data = np.fromfile(path, dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            return DocumentResult(source_path=path, pages=[])
        if progress_cb:
            progress_cb(1, 1, f"识别图片: {os.path.basename(path)}")
        page = self._ocr_image(image, slot=slot)
        page.page_no = 1
        return DocumentResult(source_path=path, pages=[page])
