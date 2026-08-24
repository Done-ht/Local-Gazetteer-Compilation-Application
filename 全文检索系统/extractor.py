"""文本提取器：从 txt/md/docx/pdf 中流式提取纯文本。

设计要点：
- 自动过滤非文字型内容（图片、PDF 中的图片部分、docx 中的图片部分）。
- 保留基本结构标记（Markdown 风格的 #、-、>），用于标题/列表/引用。
- 流式产出 (text_segment, source_byte_offset) 元组，内存只保留当前段。
- 对未安装的可选依赖（python-docx / pypdf）做优雅降级。

本模块对外只暴露 extract(file_path) 生成器。
"""
from __future__ import annotations

import os
import re
from typing import Iterator, Tuple

TextSegment = Tuple[str, int]  # (text, source_byte_offset)


# ---------------- 通用工具 ----------------

def _detect_encoding(path: str) -> str:
    """编码探测：优先 utf-8-sig，回退 gb18030 / gbk / big5。

    绝不使用 latin-1 作为兜底——它对所有字节都"成功"解码，
    会把 GBK 等多字节编码无声地变成乱码。
    若全部候选都失败，返回 utf-8 并由调用方用 errors="replace" 处理。
    """
    # 读取前 64KB 做编码判断（比 4KB 更可靠）
    try:
        with open(path, "rb") as f:
            sample = f.read(65536)
    except OSError:
        return "utf-8"
    if not sample:
        return "utf-8"

    # 检查 BOM
    if sample.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if sample.startswith(b"\xff\xfe") or sample.startswith(b"\xfe\xff"):
        return "utf-16"

    # 候选编码列表：gb18030 是 GBK 的超集，放在 gbk 之前
    for enc in ("utf-8", "gb18030", "gbk", "big5"):
        try:
            sample.decode(enc)
            return enc
        except (UnicodeDecodeError, LookupError):
            continue
    # 全部失败，用 utf-8 + replace 兜底（让损坏字符显式暴露为 \ufffd）
    return "utf-8"


def _stream_text_file(path: str, chunk_size: int = 8192) -> Iterator[TextSegment]:
    """流式读取纯文本文件，产出 (text, byte_offset)。

    byte_offset 为该段文本对应源文件的起始字节偏移。
    为处理多字节字符在块边界被截断的情况，遇到解码失败时按字节回退直至可解码。
    换行符统一标准化为 \\n（\\r\\n / \\r -> \\n）。
    """
    enc = _detect_encoding(path)
    with open(path, "rb") as f:
        byte_offset = 0
        while True:
            raw = f.read(chunk_size)
            if not raw:
                return
            seg_start = byte_offset
            # 尝试解码；若失败说明多字节字符被截断，回退文件指针直到可解码
            text = None
            cut = 0
            for _ in range(4):
                try:
                    text = raw[: len(raw) - cut].decode(enc)
                    break
                except UnicodeDecodeError:
                    cut += 1
            if text is None:
                text = raw.decode(enc, errors="replace")
                consumed = len(raw)
            else:
                consumed = len(raw) - cut
                if cut > 0:
                    # 把多读的尾部字节还回去
                    f.seek(seg_start + consumed)
            byte_offset += consumed
            # 标准化换行符
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            yield text, seg_start


# ---------------- Markdown ----------------

def _stream_markdown(path: str) -> Iterator[TextSegment]:
    """Markdown 直接按文本流式读取（已是结构化文本）。"""
    yield from _stream_text_file(path)


# ---------------- TXT ----------------

def _stream_txt(path: str) -> Iterator[TextSegment]:
    yield from _stream_text_file(path)


# ---------------- DOCX ----------------

def _stream_docx(path: str) -> Iterator[TextSegment]:
    """从 docx 提取段落文本，保留标题层级为 Markdown # 标记。

    python-docx 不暴露精确字节偏移，这里用段落序号 * 0 作为近似，
    并在 chunk 元数据中标注 source_byte_offset=0（docx 无意义字节偏移）。
    """
    try:
        from docx import Document  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "解析 .docx 需要安装 python-docx：pip install python-docx"
        ) from e

    doc = Document(path)
    # docx 是 zip，无法给出有意义的源字节偏移；用 0 占位
    for para in doc.paragraphs:
        style = (para.style.name or "").lower() if para.style else ""
        text = para.text or ""
        if not text:
            continue
        # 标题样式映射为 Markdown #
        if style.startswith("heading"):
            m = re.search(r"(\d+)", style)
            level = int(m.group(1)) if m else 1
            level = max(1, min(level, 6))
            text = "#" * level + " " + text
        elif style.startswith("title"):
            text = "# " + text
        elif style.startswith("list") or style.startswith("bullet"):
            text = "- " + text
        elif style.startswith("quote"):
            text = "> " + text
        yield text + "\n", 0


# ---------------- PDF ----------------

def _stream_pdf(path: str) -> Iterator[TextSegment]:
    """从可编辑 PDF 提取文本（图片型 PDF 会被自然过滤，因提取不到文字）。

    使用 pypdf；page index * 1_000_000 + char offset 作为伪字节偏移占位。
    """
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "解析 .pdf 需要安装 pypdf：pip install pypdf"
            ) from e

    reader = PdfReader(path)
    for page_idx, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if not text:
            continue
        # 伪字节偏移：page_idx * 1_000_000 便于区分页
        yield text + "\n", page_idx * 1_000_000


# ---------------- 调度 ----------------

_HANDLERS = {
    ".txt": _stream_txt,
    ".md": _stream_markdown,
    ".markdown": _stream_markdown,
    ".docx": _stream_docx,
    ".pdf": _stream_pdf,
}

SUPPORTED_EXTS = set(_HANDLERS.keys())


def supported(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_EXTS


def extract(path: str) -> Iterator[TextSegment]:
    """主入口：根据扩展名分发，流式产出 (text, source_byte_offset)。

    每段 text 长度不固定（受底层读取块大小/段落/页决定），由上层 chunker 合并/切分。
    """
    ext = os.path.splitext(path)[1].lower()
    handler = _HANDLERS.get(ext)
    if handler is None:
        raise ValueError(f"不支持的文件类型: {ext}（支持: {sorted(SUPPORTED_EXTS)}）")
    yield from handler(path)
