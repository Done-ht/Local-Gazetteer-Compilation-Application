# -*- coding: utf-8 -*-
"""去除 OCR 文本中的硬换行（版面排版导致的段内换行），保留正常换行。

背景：
  OCR 引擎按图像中的物理行输出文本，每行一个换行。但文档排版时一句话
  会在版面中折成多行（尤其在窄栏、双栏文档中），这些"硬换行"把一个
  完整的句子/段落切得支离破碎，严重影响可读性。

  本模块把"续行"合并回所属段落，仅在结构性边界处保留换行：
    1. 空行 —— 段落 / 页面分隔
    2. 日期 / 时间开头的行 —— 大事记、编年体条目（1978年、2月、9月24日）
    3. 序号开头的行 —— 列表条目（1、 (一) （1） 一、 第一条）
    4. 话题标签开头的行 —— "文化教育：""卫生："等短词 + 冒号
    5. 独立短标题行 —— 短且无标点、前导为空行或句末标点（如"县名的由来"）
    6. Markdown 表格行 —— 以 | 开头（表格结构需保持逐行）
    7. 纯数字短行 —— 页码等独立标记

  合并时直接拼接（不加空格），适合中文连续文本。

设计原则：
  - 高度独立：不依赖项目其他模块，可被 output.py / xfyun.exporters 共用
  - 保守保留：宁可多合并一行（可读性仍可接受），也不误拆段落
  - 仅做"去硬换行"，不做文本纠错（纠错由 text_corrector 负责）
"""
from __future__ import annotations

import re
from typing import List

# 句末标点：行尾出现这些 → 句子已完整（用于判定前一行为段落结束）
_SENT_END = set("。！？!?…")
# 冒号结尾（标题 / 标签）：行尾是 ：或 :
_COLON_END = set("：:")

# 日期 / 时间开头：1978年 / 2月 / 9月24日 / 十一月
_DATE_HEAD = re.compile(
    r"^\s*\d{1,4}\s*年"
    r"|^\s*\d{1,2}\s*月(\s*\d{1,2}\s*日?)?"
    r"|^\s*\d{1,2}\s*日"
    r"|^\s*[一二三四五六七八九十百]+月"
)
# 序号开头：1、 2. (一) （1） 一、 第一条
# 注意：数字 + 点后必须跟空格或非数字，避免误匹配小数（48.4%）
_LIST_HEAD = re.compile(
    r"^\s*\d+\s*[、]"
    r"|^\s*\d+\s*[.．]\s"
    r"|^\s*[（(]\s*[一二三四五六七八九十\d]+\s*[)）]"
    r"|^\s*[一二三四五六七八九十]+\s*[、.．]"
    r"|^\s*第\s*[一二三四五六七八九十\d]+\s*[章节条款步]"
)
# 话题标签开头：1-6 个汉字（含、·）+ 冒号，如"文化教育：""文体、科技："
_TOPIC_LABEL = re.compile(r"^[\u4e00-\u9fff、·]{1,6}[：:]")
# 纯数字短行（页码）：1-3 位纯数字
_PAGENO = re.compile(r"^\s*\d{1,3}\s*$")
# Markdown 表格行
_MD_TABLE = re.compile(r"^\s*\|")
# 短标题最大长度（汉字计）
_HEADING_MAX_LEN = 12


def _starts_entry(line: str) -> bool:
    """该行是否以结构性标记开头（→ 新条目 / 新段落，保留换行）。"""
    s = line.lstrip()
    if not s:
        return False
    if _DATE_HEAD.match(s):
        return True
    if _LIST_HEAD.match(s):
        return True
    if _TOPIC_LABEL.match(s):
        return True
    if _MD_TABLE.match(s):
        return True
    if _PAGENO.match(s):
        return True
    return False


def _is_heading(line: str, prev: str) -> bool:
    """该行是否是独立短标题（→ 新段落，且下一行也另起）。

    条件：行较短、不含句中/句末标点、前导为空行或句末标点 / 冒号结尾。
    用于识别"县名的由来""一步跨三县"这类独立成行的标题。
    """
    s = line.strip()
    if not (2 <= len(s) <= _HEADING_MAX_LEN):
        return False
    # 含句中 / 句末标点 → 不是纯标题
    if any(p in s for p in "。！？；，、：!?;,:"):
        return False
    # 日期 / 序号 / 话题标签等交给 _starts_entry 处理，这里不算标题
    if _starts_entry(s):
        return False
    pv = prev.rstrip()
    if not pv:
        return True
    return pv[-1] in _SENT_END or pv[-1] in _COLON_END


def merge_lines_to_paragraphs(lines: List[str]) -> List[str]:
    """把物理行列表合并为逻辑段落列表（去除段内硬换行）。

    输入：OCR 物理行（每行一个元素，可能含空行）。
    输出：逻辑段落（每个元素是一个段落，内部续行已直接拼接）。
    空行保留为单独的 "" 元素，作为段落分隔符（连续空行折叠为一个）。
    """
    paragraphs: List[str] = []
    cur: List[str] = []
    prev_nonblank = ""
    prev_was_heading = False

    def flush():
        """冲掉当前累积段落。空段落不追加。"""
        nonlocal cur
        if cur:
            paragraphs.append("".join(cur))
            cur = []

    for raw in lines:
        s = raw.strip()
        if not s:
            # 空行：先冲掉当前段落，再追加一个空行分隔符（连续空行折叠）
            flush()
            if not paragraphs or paragraphs[-1] != "":
                paragraphs.append("")
            prev_nonblank = ""
            prev_was_heading = False
            continue

        is_head = _is_heading(s, prev_nonblank)
        starts_entry = _starts_entry(s)

        # 判定是否新开段落
        start_new = False
        if not cur:
            start_new = True  # 段落开头
        elif prev_was_heading:
            start_new = True  # 上一行是独立标题 → 本行另起
        elif starts_entry:
            start_new = True  # 日期 / 序号 / 话题标签 → 新条目
        elif is_head:
            start_new = True  # 独立短标题 → 新段落

        if start_new:
            flush()
            cur = [s]
            # 标题行（非结构性条目）→ 下一行也另起
            prev_was_heading = is_head and not starts_entry
        else:
            # 续行：直接拼接到当前段落（不加空格，适合中文连续文本）
            cur.append(s)
            prev_was_heading = False
        prev_nonblank = s

    flush()
    # 去掉末尾 / 开头多余空行
    while paragraphs and paragraphs[-1] == "":
        paragraphs.pop()
    while paragraphs and paragraphs[0] == "":
        paragraphs.pop(0)
    return paragraphs


def dehard_wrap(text: str, paragraph_sep: str = "\n") -> str:
    """去除文本中的硬换行，返回段落文本。

    参数:
        text: 含硬换行的原始文本（OCR 物理行用 \\n 拼接）。
        paragraph_sep: 段落之间的分隔符，默认单换行。
            传 "\\n\\n" 可得到空行分隔的段落（更易读）。

    示例:
        >>> dehard_wrap("文化教育：解放前...1998年\\n全县小学171所...在校\\n学生26487人...")
        '文化教育：解放前...1998年全县小学171所...在校学生26487人...'
    """
    if not text:
        return text
    lines = text.split("\n")
    paragraphs = merge_lines_to_paragraphs(lines)
    return paragraph_sep.join(paragraphs)
