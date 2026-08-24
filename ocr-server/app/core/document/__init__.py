"""文档处理器模块。

每种文档类型（图片 / PDF / DOCX）对应一个独立 Handler，
通过 BaseHandler 统一接口，便于扩展。
"""
from .base import BaseHandler, DocumentResult
from .image_handler import ImageHandler
from .pdf_handler import PdfHandler
from .docx_handler import DocxHandler

__all__ = [
    "BaseHandler",
    "DocumentResult",
    "ImageHandler",
    "PdfHandler",
    "DocxHandler",
]
