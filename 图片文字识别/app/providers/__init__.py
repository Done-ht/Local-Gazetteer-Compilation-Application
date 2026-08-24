"""OCR 提供商模块（服务端版，仅本地 PaddleOCR）。"""
from .base import BaseProvider, OCRResult, OCRLine
from .paddle_local import PaddleLocalProvider

__all__ = [
    "BaseProvider",
    "OCRResult",
    "OCRLine",
    "PaddleLocalProvider",
]
