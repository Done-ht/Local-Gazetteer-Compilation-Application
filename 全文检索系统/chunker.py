"""流式分块器：边读边分块，1w 字一块、20 字重叠。

规则（对应需求 4、6）：
- 累积文本，当累积 >= chunk_size（默认 10000 字）时切出一个 chunk。
- 下一个 chunk 的开头要包含前一个 chunk 末尾 overlap（默认 20）字。
  例：chunk1 = 文本[0:10000]，则 chunk2 起始于 10000-20=9980。
- 最后剩余不足 chunk_size 的文本作为一个尾 chunk 输出（若非空）。
- 内存中只保留当前 chunk 缓冲区，不缓存全文。
- 标题追踪：扫描 chunk 文本中的 Markdown 风格 # 标题，
  维护 current_heading 状态，写入每个 chunk 的 heading 字段。
  语义：chunk.heading = 该 chunk 主要内容所属的最近标题；
       首个标题之前的 chunk，heading 为空字符串。

本模块只负责把 (text_segment, source_byte_offset) 流切分成带偏移的 chunk 字典，
不负责落盘（落盘由 storage 模块完成）。
"""
from __future__ import annotations

import re
from typing import Iterator, Tuple, List

DEFAULT_CHUNK_SIZE = 10_000
DEFAULT_OVERLAP = 20

# extractor 产出的段
TextSegment = Tuple[str, int]

# Markdown 风格标题：行首 1-6 个 # 后跟标题文本
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)


def _first_heading(text: str) -> str:
    """返回文本中第一个标题的文本，无则返回 None。"""
    m = _HEADING_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


def _last_heading(text: str) -> str:
    """返回文本中最后一个标题的文本，无则返回 None。"""
    matches = _HEADING_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return None


class Chunker:
    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须 > 0")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap 必须 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        # step = chunk_size - overlap
        self.step = chunk_size - overlap

    def chunk(self, segments: Iterator[TextSegment]) -> Iterator[dict]:
        """把段流切成 chunk。

        每个 chunk 字典包含：
            text:               该块文本
            text_offset:        在提取后纯文本中的起始字符偏移
            text_length:        该块字符数
            overlap_prev:       与上一块重叠的字符数
            source_byte_offset: 该块起始文本对应的源文件字节偏移（取段的偏移）
            source_byte_length: 近似字节长度（用 text 编码长度近似）

        注意：source_byte_offset 采用"段级"映射——记录该 chunk 起始字符
        所在段的 source_byte_offset。对 txt/md 较精确；对 docx/pdf 为占位。
        """
        # 缓冲：把所有段拼成 (char, byte_offset) 列表太耗内存，
        # 这里采用"分段拼接 + 滑动窗口"策略：维护一个文本缓冲与对应段偏移表。
        buf_text: List[str] = []
        buf_byte_offsets: List[int] = []  # 每个段在 buf 中的起始字符索引对应的源字节偏移
        buf_char_start: List[int] = []     # 每个段在全局提取文本中的起始字符偏移
        buf_len = 0
        global_char_pos = 0   # 已产出（含已切出）的全局字符游标
        # 上一切出的 chunk 末尾在全局文本中的位置
        last_chunk_end = 0
        chunk_seq = 0
        # 标题追踪：最近经过的标题（供下一 chunk 继承）
        current_heading = ""

        # 为支持 overlap，切出一个 chunk 后保留末尾 overlap 字符在缓冲中。
        # 实现上：缓冲是"尚未切出"的文本。切出 chunk_size 后，
        # 把缓冲前 (chunk_size - overlap) 字符丢弃，保留末尾 overlap 字符作为下一块开头。
        # 但段级字节偏移映射在丢弃后需要同步平移。

        # 为简化映射，我们维护一个"段列表"，每段记录 (text, seg_char_start_in_buf, seg_byte_offset, seg_global_char_start)
        # 切出后整体平移 seg_char_start_in_buf。
        segs: List[List] = []  # [text, start_in_buf, byte_offset, global_char_start]

        def flush_to_size(target: int) -> bool:
            """尝试把缓冲填到至少 target 长度。返回是否还有输入。"""
            nonlocal buf_len
            # 通过拉取 segment 实现
            return True  # 实际拉取在主循环

        for text, byte_off in segments:
            if not text:
                continue
            segs.append([text, buf_len, byte_off, global_char_pos])
            buf_text.append(text)
            buf_len += len(text)
            global_char_pos += len(text)

            # 只要缓冲 >= chunk_size，就切出
            while buf_len >= self.chunk_size:
                chunk_seq += 1
                # 拼接缓冲文本
                full = "".join(buf_text)
                chunk_text = full[: self.chunk_size]
                # 起始全局字符偏移 = 第一个段的全局起始
                chunk_global_start = segs[0][3]
                # overlap_prev：如果是第二块及以后，且上一块结束位置 > 本块起始，则有重叠
                overlap_prev = 0
                if last_chunk_end > chunk_global_start:
                    overlap_prev = last_chunk_end - chunk_global_start
                # 起始源字节偏移：用第一个段
                chunk_byte_offset = segs[0][2]

                # 标题追踪：本 chunk 的 heading = chunk 内首个标题（若有），
                # 否则继承 current_heading
                first_h = _first_heading(chunk_text)
                if first_h is not None:
                    heading = first_h
                    # 更新 current_heading 为本 chunk 中最后一个标题（供下个 chunk 用）
                    last_h = _last_heading(chunk_text)
                    if last_h:
                        current_heading = last_h
                else:
                    heading = current_heading

                yield {
                    "text": chunk_text,
                    "text_offset": chunk_global_start,
                    "text_length": len(chunk_text),
                    "overlap_prev": overlap_prev,
                    "source_byte_offset": chunk_byte_offset,
                    "source_byte_length": len(chunk_text.encode("utf-8")),
                    "heading": heading,
                }
                last_chunk_end = chunk_global_start + len(chunk_text)

                # 保留末尾 overlap 字符作为下一块开头
                keep_from = self.chunk_size - self.overlap
                if self.overlap == 0:
                    # 全部丢弃
                    buf_text = []
                    buf_len = 0
                    segs = []
                else:
                    kept = full[keep_from:]
                    buf_len = len(kept)
                    # 重新构建段表：保留的字符来自原缓冲末尾
                    # 计算保留部分起始的全局偏移与字节偏移
                    kept_global_start = chunk_global_start + keep_from
                    # 找到 kept 起始落在哪个原段
                    new_byte_off = chunk_byte_offset
                    acc = chunk_global_start
                    for s in segs:
                        s_text, s_start_in_buf, s_byte, s_global = s
                        s_end_global = s_global + len(s_text)
                        if kept_global_start < s_end_global:
                            new_byte_off = s_byte
                            break
                        acc = s_end_global
                    buf_text = [kept]
                    segs = [[kept, 0, new_byte_off, kept_global_start]]

        # 尾部剩余
        if buf_len > 0:
            chunk_seq += 1
            full = "".join(buf_text)
            chunk_global_start = segs[0][3] if segs else global_char_pos
            overlap_prev = 0
            if last_chunk_end > chunk_global_start:
                overlap_prev = last_chunk_end - chunk_global_start
            chunk_byte_offset = segs[0][2] if segs else 0
            # 标题追踪（尾部 chunk）
            first_h = _first_heading(full)
            if first_h is not None:
                heading = first_h
            else:
                heading = current_heading
            yield {
                "text": full,
                "text_offset": chunk_global_start,
                "text_length": len(full),
                "overlap_prev": overlap_prev,
                "source_byte_offset": chunk_byte_offset,
                "source_byte_length": len(full.encode("utf-8")),
                "heading": heading,
            }
