# -*- coding: utf-8 -*-
"""导出：txt / docx 原位改写（保留格式）/ PDF 来源转修正后 docx"""
import os

from .docload import DocContext


def export_txt(pages: list, out_path: str) -> None:
    """合并页文本写出为 txt"""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(p.text for p in pages))


def _replace_in_paragraph(para, local_start: int, local_end: int, replacement: str) -> None:
    """在段落内做局部替换：区间合并进第一个涉及的 run，清空其余 run 的对应部分，
    只改 run 文本，保留加粗等格式。"""
    pos = 0
    first = True
    for run in para.runs:
        rlen = len(run.text)
        rstart, rend = pos, pos + rlen
        pos = rend
        if rend <= local_start or rstart >= local_end:
            continue
        s = max(local_start - rstart, 0)
        e = min(local_end - rstart, rlen)
        if first:
            run.text = run.text[:s] + replacement + run.text[e:]
            first = False
        else:
            run.text = run.text[:s] + run.text[e:]


def _apply_corrections_docx(doc_ctx: DocContext, corrections: list) -> None:
    """把 correction 应用到 docx Document。corrections: (page_num, start, end, replacement)

    ParaSpan 直接持有 Paragraph 对象引用，正文段落与表格单元格段落
    均可原位改写（保留加粗等格式）。
    """
    # 从后往前替换，避免偏移失效
    ordered = sorted(corrections, key=lambda c: (c[0], c[1]), reverse=True)
    for page_num, start, end, replacement in ordered:
        spans = doc_ctx.page_maps.get(page_num, [])
        # 找出与 [start, end) 相交的段落（通常只有一个）
        hit = [sp for sp in spans if sp.start < end and sp.end > start]
        if not hit:
            continue
        for i, sp in enumerate(hit):
            para = sp.paragraph  # 直接用段落对象引用（正文或表格单元格）
            local_start = max(start - sp.start, 0)
            local_end = min(end - sp.start, sp.end - sp.start)
            # 替换文本只写进第一个段落，跨段落的部分删除
            _replace_in_paragraph(para, local_start, local_end, replacement if i == 0 else "")


def export_docx(doc_ctx: DocContext, corrections: list, out_path: str) -> None:
    """基于 docload 保留的 Document 原位改写并另存。不会改动源文件。"""
    if doc_ctx.doc is None:
        raise ValueError("DocContext 中没有 docx Document 对象，无法导出 docx")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    _apply_corrections_docx(doc_ctx, corrections)
    doc_ctx.doc.save(out_path)


def _apply_corrections_text(text: str, corrections: list) -> str:
    """在纯文本上应用 correction（从后往前）"""
    for _, start, end, replacement in sorted(corrections, key=lambda c: c[1], reverse=True):
        text = text[:start] + replacement + text[end:]
    return text


def export_pdf_as_docx(pages: list, corrections: list, out_path: str) -> None:
    """PDF / OCR 来源的导出：新建 docx，按页写入修正后的文本（按行分段）"""
    import docx
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    corr_by_page = {}
    for c in corrections:
        corr_by_page.setdefault(c[0], []).append(c)

    document = docx.Document()
    for i, page in enumerate(pages):
        text = _apply_corrections_text(page.text, corr_by_page.get(page.page_num, []))
        for line in text.splitlines():
            if line.strip():
                document.add_paragraph(line)
        if i < len(pages) - 1:
            document.add_page_break()
    document.save(out_path)
