# -*- coding: utf-8 -*-
"""统一文档加载：txt / docx / pdf / 图片 -> list[Page] + DocContext

分页规则：约 2000 字一页，按段落边界切分（段落不拆半）。
docx 会维护“页内偏移 -> 段落对象”的映射，供导出时原位改写；
正文段落与表格单元格段落均参与校对，表格单元格的修正可写回原 cell。
扫描版 PDF 与图片不做 OCR，仅打标记由编排层处理。
"""
import os
from dataclasses import dataclass, field
from typing import Optional

from .models import Page

PAGE_SIZE = 2000  # 每页目标字数
TXT_EXTS = {".txt"}
DOCX_EXTS = {".docx"}
PDF_EXTS = {".pdf"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
# PDF 整篇可提取文本少于该阈值则判定为扫描版
SCANNED_PDF_TEXT_THRESHOLD = 50


@dataclass
class ParaSpan:
    """页内一段文本对应的 docx 段落位置。

    paragraph 直接持有 python-docx 的 Paragraph 对象引用
    （正文段落或表格单元格段落均可），导出时无需再按 index 反查。
    """
    paragraph: object  # python-docx Paragraph 对象
    start: int         # 页内偏移（含）
    end: int           # 页内偏移（不含）


@dataclass
class DocContext:
    """加载结果上下文"""
    source_path: str
    file_type: str                      # txt / docx / pdf / image / scanned_pdf
    doc: object = None                  # docx 时的 python-docx Document，供导出用
    page_maps: dict = field(default_factory=dict)  # page_num -> list[ParaSpan]
    has_tables: bool = False            # docx 是否含表格（已纳入校对，仅作提示）
    needs_ocr: bool = False             # 是否需要走 OCR


def needs_ocr(path: str) -> bool:
    """判断该文件是否需要走 OCR（图片，或无可提取文本的扫描版 PDF）"""
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTS:
        return True
    if ext in PDF_EXTS:
        return _pdf_is_scanned(path)
    return False


def load_document(path: str):
    """加载文档，返回 (pages, doc_ctx)。扫描版 PDF / 图片返回空 pages + 标记"""
    ext = os.path.splitext(path)[1].lower()
    if ext in TXT_EXTS:
        return _load_txt(path)
    if ext in DOCX_EXTS:
        return _load_docx(path)
    if ext in PDF_EXTS:
        return _load_pdf(path)
    if ext in IMAGE_EXTS:
        ctx = DocContext(source_path=path, file_type="image", needs_ocr=True)
        return [], ctx
    raise ValueError(f"不支持的文件类型: {ext}")


def _paginate_paragraphs(paras: list) -> list:
    """把段落列表按约 PAGE_SIZE 字分页（段落不拆半），返回 [ [段落, ...], ... ]"""
    pages, current, size = [], [], 0
    for para in paras:
        n = len(para) + 1  # +1 为换行
        if current and size + n > PAGE_SIZE:
            pages.append(current)
            current, size = [], 0
        current.append(para)
        size += n
    if current:
        pages.append(current)
    return pages


def _load_txt(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="gbk", errors="replace") as f:
            text = f.read()
    paras = [p for p in text.splitlines() if p.strip()]
    pages = [Page(page_num=i + 1, text="\n".join(group))
             for i, group in enumerate(_paginate_paragraphs(paras))]
    ctx = DocContext(source_path=path, file_type="txt")
    return pages, ctx


def _load_docx(path: str):
    import docx
    doc = docx.Document(path)
    # 收集正文段落 + 表格单元格段落。
    # 顺序：先按文档顺序遍历 body 子元素，遇到 <w:p> 取正文段落，
    # 遇到 <w:tbl> 把表格里每个单元格的段落按行单元格顺序追加。
    # 这样表格内容会出现在其正文位置之后，便于 LLM 结合上下文校对。
    from docx.oxml.ns import qn
    from docx.text.paragraph import Paragraph
    from docx.table import Table

    body = doc.element.body
    paras = []
    has_tables = False
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            paras.append(Paragraph(child, doc.part))
        elif child.tag == qn("w:tbl"):
            has_tables = True
            table = Table(child, doc.part)
            for row in table.rows:
                # row.cells 在合并单元格时可能重复返回同一 cell，用 id 去重
                seen_cell_ids = set()
                for cell in row.cells:
                    cid = id(cell._tc)
                    if cid in seen_cell_ids:
                        continue
                    seen_cell_ids.add(cid)
                    for para in cell.paragraphs:
                        if para.text.strip():
                            paras.append(para)

    pages, page_maps = [], {}
    page_num, current_text, current_spans, size = 1, [], [], 0

    def flush():
        nonlocal page_num, current_text, current_spans, size
        if not current_text:
            return
        pages.append(Page(page_num=page_num, text="".join(current_text)))
        page_maps[page_num] = current_spans
        page_num += 1
        current_text, current_spans, size = [], [], 0

    for para in paras:
        t = para.text
        if not t.strip():
            continue
        piece = ("\n" if current_text else "") + t
        if current_text and size + len(piece) > PAGE_SIZE:
            flush()
            piece = t
        start = size + (1 if current_text else 0)
        current_text.append(piece)
        current_spans.append(ParaSpan(paragraph=para, start=start, end=start + len(t)))
        size += len(piece)
    flush()

    ctx = DocContext(
        source_path=path,
        file_type="docx",
        doc=doc,
        page_maps=page_maps,
        has_tables=has_tables,
    )
    return pages, ctx


def _pdf_is_scanned(path: str) -> bool:
    import fitz
    with fitz.open(path) as pdf:
        total = sum(len(page.get_text().strip()) for page in pdf)
    return total < SCANNED_PDF_TEXT_THRESHOLD


def _load_pdf(path: str):
    import fitz
    ctx = DocContext(source_path=path, file_type="pdf")
    pages = []
    with fitz.open(path) as pdf:
        for i, page in enumerate(pdf):
            pages.append(Page(page_num=i + 1, text=page.get_text()))
    if sum(len(p.text.strip()) for p in pages) < SCANNED_PDF_TEXT_THRESHOLD:
        # 扫描版：打 OCR 标记，pages 视为空
        ctx.file_type = "scanned_pdf"
        ctx.needs_ocr = True
        return [], ctx
    return pages, ctx
