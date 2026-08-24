"""DOCX 文档处理器。

处理两种情况：
1. 扫描型 DOCX：正文段落为空或占位，内容以图片形式嵌入 —— OCR 提取图片文字。
2. 混合型 DOCX：既有原生文字段落，又含图片 —— 保留原生文字，图片走 OCR。

输出策略：
- 原生文字段落直接收集（保留样式由输出层负责）
- 嵌入图片逐张过滤 + OCR
"""
from __future__ import annotations

import io
import os
from typing import List

import cv2
import numpy as np

from .base import BaseHandler, DocumentResult, PageResult


class DocxHandler(BaseHandler):
    """DOCX 处理器。"""

    extensions = [".docx"]

    def process(self, path: str, progress_cb=None, page_cb=None, slot: int = 0) -> DocumentResult:
        from docx import Document
        from docx.opc.constants import RELATIONSHIP_TYPE as RT

        doc = Document(path)
        # 1. 收集原生段落文字
        native_lines: List[str] = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                native_lines.append(text)
        native_text = "\n".join(native_lines)

        # 2. 提取所有嵌入图片
        image_parts = []
        for rel in doc.part.rels.values():
            if rel.reltype == RT.IMAGE:
                blob = rel.target_part.blob
                image_parts.append(blob)

        total = len(image_parts)
        pages: List[PageResult] = []
        for idx, blob in enumerate(image_parts):
            arr = np.frombuffer(blob, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            if progress_cb:
                progress_cb(
                    idx + 1,
                    max(total, 1),
                    f"识别 DOCX 内嵌图片 {idx + 1}/{total}",
                )
            page = self._ocr_image(img, slot=slot)
            page.page_no = idx + 1
            pages.append(page)
            # 增量回调：致命错误直接抛出中断
            if page_cb is not None:
                page_cb(page)

        return DocumentResult(
            source_path=path, pages=pages, native_text=native_text
        )
