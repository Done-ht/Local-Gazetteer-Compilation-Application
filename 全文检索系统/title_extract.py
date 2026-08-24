# -*- coding: utf-8 -*-
"""Chunk 标题提取模块。

当 chunker 产出的 heading 为空时（docx/pdf/txt 无 Markdown # 标题），
用本模块从 chunk 文本内容中识别标题，填补 heading 字段。

算法流程：
  Step 1 逐行扫描，收集候选行 + 直接命中
  Step 2 取得分最高候选
  Step 3 fallback：高频实词拼接 -> 首非空行截断
"""
from __future__ import annotations

import re
from collections import Counter

# ---------- 配置 ----------
LEN_SHORT = 10      # 短标题阈值
LEN_NORMAL = 20     # 常规标题阈值
LEN_MAX = 30        # 候选行最大长度（超过直接淘汰）

# 章节格式正则（命中即"强候选"）
PATTERN_CHAPTER = re.compile(
    r'^\s*('
    r'第[一二三四五六七八九十百千零\d]+[部章节回卷集篇]'   # 第一章 / 第二部
    r'|[Cc]hapter\s+\d+'                                    # Chapter 1
    r'|^\d{1,4}[\.\-]\d{1,4}\s*$'                            # 1.1 / 1-2 / 12.34（整行匹配，避免匹配数据行如 23.0 宣城市 1407380）
    r'|序\s*$|前言\s*$|后记\s*$|附录\s*$|楔子\s*$|引子\s*$'  # 单字标题
    r')'
)
# 纯数字章节编号单独处理（需配合"前后空行 + 范围 1~500"约束）
PATTERN_NUM_ONLY = re.compile(r'^\d{1,3}$')

# 年鉴类条目标记：【XXX】格式（年鉴/方志常用条目标题）
PATTERN_YEARBOOK_MARK = re.compile(r'【([^】]{1,30})】')
# 年鉴类条目行：整行是 【XXX】 或 【XXX】后跟少量说明文字
PATTERN_YEARBOOK_MARK_LINE = re.compile(r'^\s*【([^】]{1,30})】\s*.{0,20}$')

# 行尾终止标点（出现则强烈提示"这是正文不是标题"）
END_PUNCT = set('。！？!?…；;：:，,、')
# 行尾顿号（强正文信号，年鉴正文常以顿号分隔并列项）
END_COMMA = set('、，,')
# 行尾闭合引号（对白结尾强信号）
END_QUOTE = set('""』」』')
# 行首闭合符号（chunk 边界残片）
START_CLOSE = set('》」』）)】｝}’\'')
# 说话动词+冒号（强对白信号）
PATTERN_SPEECH_VERB = re.compile(r'[说问道笑喊叫哭骂唱嘀咕哼答劝][：:]')
# 行内对白（含 "..." 或 「...」）
PATTERN_DIALOGUE_INLINE = re.compile(r'[""][^""]{1,80}[""]')
# 页码混入模式：标题末尾的 ·数字 或 ·数字·数字（如"县情概览·65"）
PATTERN_PAGE_NUM_SUFFIX = re.compile(r'[·•・]\d{1,4}$')
# 纯数字+单位开头（强正文信号，如"年12月31日"、"50万元"）
PATTERN_NUM_UNIT_START = re.compile(r'^\d{1,4}\s*(年|月|日|时|分|秒|元|万|亿|斤|公斤|吨|亩|户|人|名|个|所|家|类|项|次|轮|期|届|件|起|处|座|台|套|条|本|册|卷|篇|章|节)')
# 正文动词开头模式（强正文信号，年鉴/方志类常见句首）
# 如"坚持举全县之力"、"按照企业向园区集中"、"距还是干部思想观念"
PATTERN_BODY_TEXT_START = re.compile(r'^(坚持|按照|推进|加强|促进|推动|加快|推进|落实|深化|完善|强化|突出|抓住|围绕|立足|着眼|基于|鉴于|尽管|虽然|但是|然而|因为|由于|所以|因此|并且|而且|不仅|不但|无论|不管|除了|除非|随着|经过|通过|根据|按照|依据|鉴于|就|距|还有|还有|也是|也是|已是|已是|不是|不是|已是|仍是|仍是|更有|更有|更有|且|且|且)')

# 中文停用词（高频但无信息量）
STOP_WORDS = {
    '的', '了', '是', '在', '我', '有', '和', '就', '不', '再', '也', '都', '还', '又',
    '很', '最', '太', '只', '更', '已', '正', '将', '被', '把', '让', '使', '到', '向',
    '着', '往', '从', '对', '跟', '与', '及', '或', '但', '即', '则', '而', '之', '其',
    '此', '这', '那', '他', '她', '它', '你', '们', '您', '个', '些', '一', '上', '下',
    '里', '中', '人', '为', '以', '可', '能', '要', '会', '说', '道', '问', '看', '想',
    '一个', '一些', '这种', '这那', '我们', '他们', '她们', '它们', '你们', '这个', '那个',
    '什么', '怎么', '为何', '哪里', '怎样', '谁', '能够', '可以', '可能', '应该', '需要',
    '已经', '正在', '于是', '然后', '接着', '后来', '因为', '所以', '因此', '并且', '而且',
    '但是', '然而', '虽然', '尽管', '如果', '只有', '只要', '一下', '一起', '上来', '下去',
    '出来', '回来', '过来', '起来', '现在', '当时', '后来', '最后', '首先', '然后', '不过',
    '不要', '不能', '不会', '没有', '没什么', '什么的', '这样', '那样', '怎么', '怎样',
    '说道', '问道', '喊道', '叫道', '写道', '笑道', '哭道', '骂道', '哼道', '唱道', '念道',
    '嘀咕', '回答', '答应', '劝说', '大叫', '大喊', '大笑', '点头', '摇头', '抬头', '低头',
    '这个', '这些', '这种', '这里', '那是', '那是', '那些', '那种', '那里', '自己', '别人',
    '他们', '我们', '你们', '她们', '它们', '大家', '咱们', '先生', '女士', '同志',
}


# ---------- 工具函数 ----------
def _is_blank(line: str) -> bool:
    return not line or line.strip() == '' or line.isspace()


def _is_meaningful_word(w: str) -> bool:
    if len(w) < 2:
        return False
    if w in STOP_WORDS:
        return False
    if not re.search(r'[\u4e00-\u9fff]', w):
        return False
    return True


def _is_pure_english(s: str) -> bool:
    return not re.search(r'[\u4e00-\u9fff]', s)


def _content_words(text: str) -> list:
    """简易分词：2-4字滑窗 + 过滤停用词。"""
    s = re.sub(r'[\s\W_]+', '', text)
    words = []
    for n in (2, 3, 4):
        for i in range(len(s) - n + 1):
            w = s[i:i + n]
            if _is_meaningful_word(w):
                words.append(w)
    return words


def _ends_with_punct(line: str) -> bool:
    s = line.rstrip()
    if not s:
        return False
    return s[-1] in END_PUNCT


def _position_score(idx: int, total: int) -> float:
    if total <= 0:
        return 0.0
    if idx < total * 0.2:
        return 2.0
    if idx < total * 0.5:
        return 1.0
    return 0.0


def _length_score(length: int) -> float:
    if length <= 0:
        return -5.0
    if 10 <= length <= 20:
        return 3.0
    if 5 <= length < 10:
        return 2.0
    if 20 < length <= 30:
        return 1.0
    if length < 5:
        return 0.5
    return -2.0


def _structure_score(lines: list, idx: int) -> float:
    s = 0.0
    if idx > 0 and _is_blank(lines[idx - 1]):
        s += 1.0
    if idx < len(lines) - 1 and _is_blank(lines[idx + 1]):
        s += 1.0
    if not _ends_with_punct(lines[idx]):
        s += 2.0
    return s


def _format_score(line: str) -> float:
    if PATTERN_CHAPTER.search(line):
        return 5.0
    s = line.strip()
    if re.match(r'^(《[^》]{1,20}》|"[^"]{1,20}"|\'[^\']{1,20}\')\s*$', s):
        return 2.0
    return 0.0


def _is_chapter_number_only(s: str, lines: list, idx: int) -> bool:
    if not PATTERN_NUM_ONLY.match(s):
        return False
    try:
        n = int(s)
    except ValueError:
        return False
    if not (1 <= n <= 500):
        return False
    prev_blank = idx > 0 and _is_blank(lines[idx - 1])
    next_blank = idx < len(lines) - 1 and _is_blank(lines[idx + 1])
    return prev_blank or next_blank


def _score_line(line: str, lines: list, idx: int, chunk_freq: Counter) -> float:
    s = line.strip()
    L = len(s)
    if L == 0 or L > LEN_MAX:
        return -999.0
    if s[0] in '的了是在我有和就也不再也都还又':
        return -999.0
    if s[0] in START_CLOSE:
        return -999.0
    if s[0] in '""「『':
        return -999.0
    if _ends_with_punct(s):
        if PATTERN_CHAPTER.search(s):
            pass
        else:
            return -999.0
    if s[-1] in END_QUOTE and not PATTERN_CHAPTER.search(s):
        return -999.0
    if '《' in s and '》' not in s:
        return -999.0
    if s.count('"') != s.count('"') and '"' in s:
        return -999.0
    # 正文动词开头 → 强烈正文信号，直接淘汰
    if PATTERN_BODY_TEXT_START.match(s):
        return -999.0

    total_lines = len(lines)
    sc = 0.0
    sc += _length_score(L)
    sc += _structure_score(lines, idx)
    sc += _format_score(s)
    sc += _position_score(idx, total_lines)
    # 词汇分
    words = _content_words(s)
    hit = 0
    for w in words:
        if chunk_freq.get(w, 0) >= 2:
            hit += 1
            if hit >= 3:
                break
    sc += float(hit)

    if PATTERN_DIALOGUE_INLINE.search(s):
        sc -= 5.0
    if PATTERN_SPEECH_VERB.search(s):
        sc -= 5.0
    if _is_pure_english(s):
        sc -= 3.0
    return sc


# ---------- 主入口 ----------
def _is_yearbook_index_chunk(lines: list) -> bool:
    """检测是否为年鉴索引页 chunk。

    年鉴末尾索引页特征：大量行形如"词条+页码+栏号"（如"实战化训练111b"），
    行尾匹配 \\d{1,4}[ab]$ 的行占比 > 30%。
    """
    if len(lines) < 5:
        return False
    index_line_count = 0
    non_blank = 0
    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        non_blank += 1
        if re.search(r'\d{1,4}[ab]$', s):
            index_line_count += 1
    return non_blank > 0 and index_line_count / non_blank > 0.3


def _is_yearbook_toc_chunk(lines: list) -> bool:
    """检测是否为年鉴纯目录页 chunk。

    纯目录页特征（非编辑说明，但全是目录条目）：
    - 前 30 行含大量【】条目标记（≥8 个）
    - 同时含乡镇名称（以"镇""乡"结尾，≥3 个）
    - 常夹杂页码（独立数字行或行尾数字）

    例如：南丰镇\\n【概况】\\n【工业经济...】\\n190\\n...
    """
    if len(lines) < 5:
        return False
    head_lines = lines[:30]
    mark_count = 0
    town_count = 0
    for raw in head_lines:
        s = raw.strip()
        if not s:
            continue
        # 统计【】条目
        marks = PATTERN_YEARBOOK_MARK.findall(s)
        if marks:
            mark_count += len(marks)
        # 统计乡镇名（整行是"XX镇"或"XX乡"）
        if re.match(r'^[\u4e00-\u9fff]{2,6}[镇乡]$', s):
            town_count += 1
    return mark_count >= 8 and town_count >= 3


def _is_yearbook_editorial_chunk(lines: list) -> bool:
    """检测是否为年鉴编辑说明/目录页 chunk。

    特征：前 50 行含"编辑说明""编 明 辑说""目\n特载"等年鉴目录特征词，
    或前 30 行含大量"一、""二、""三、"等编辑说明编号开头。
    """
    head_text = '\n'.join(lines[:50])
    # 编辑说明特征词（含 OCR 乱序变体）
    editorial_markers = ['编辑说明', '编 明 辑说', '凡例', '编纂说明']
    for marker in editorial_markers:
        if marker in head_text:
            return True
    # 目录特征：前 50 行同时含"目"和"特载"或"类目"
    if '目' in head_text and ('特载' in head_text or '类目' in head_text):
        # 进一步确认：前 50 行含"一、《"或"二、《"等目录编号
        if re.search(r'[一二三四五六七八九十]、', head_text):
            return True
    return False


def extract_title(chunk_text: str) -> str:
    """从 chunk 文本中提取标题。

    返回标题字符串（已 strip）。无法提取时返回空字符串。

    识别顺序（优先级从高到低）：
    1. 章节格式（第X章/附录/序言等）
    2. 年鉴类【】条目标记（年鉴/方志常用）
    3. 史书卷目格式（开头连续短行）
    4. 独立短行（前后空行包围）
    5. 候选行打分
    6. fallback：高频实词拼接 / 首非空行截断

    特殊处理：
    - 年鉴编辑说明/目录页 → 跳过【】提取，返回"编辑说明"或"目录"
    - 年鉴索引页 → 返回"索引"
    - 年鉴纯目录页 → 返回"目录"
    """
    if not chunk_text or not chunk_text.strip():
        return ""
    lines = chunk_text.split('\n')

    # 年鉴索引页检测：直接返回"索引"
    if _is_yearbook_index_chunk(lines):
        return "索引"

    # 年鉴编辑说明/目录页检测：提前返回，避免目录中的章节号/【】被误识别
    if _is_yearbook_editorial_chunk(lines):
        return "编辑说明与目录"

    # 年鉴纯目录页检测：全是【】条目+乡镇名+页码，无正文内容
    if _is_yearbook_toc_chunk(lines):
        return "目录"

    chunk_freq = Counter(_content_words(chunk_text))

    direct_hits = []
    strong_hits = []
    candidates = []

    # 史书卷目格式识别：开头几行是短行（2-10字），行尾无标点，
    # 形如 "纪第一\n武帝操" 或 "列传第二十三\n华佗" 这种"卷名+章节名"
    # 优先作为 strong_hit
    head_short_lines = []  # 收集开头连续的短行
    for idx, raw in enumerate(lines[:5]):
        s = raw.strip()
        if not s:
            if head_short_lines:
                break  # 短行后的空行表示短行组结束
            continue
        if len(s) > 12 or _ends_with_punct(s):
            break
        # 排除首行是"了是在我"等开头词
        if s[0] in '的了是在我有和就也不再也都还又':
            break
        head_short_lines.append((idx, s))
    # 如果开头有 2-3 个连续短行，且首行不是完整句子（无标点结尾），
    # 视为史书卷目格式
    if len(head_short_lines) >= 2:
        # 验证：最后一个短行的下一行应该是长行（正文）
        last_idx = head_short_lines[-1][0]
        if last_idx + 1 < len(lines):
            next_line = lines[last_idx + 1].strip()
            if len(next_line) > 12 or _ends_with_punct(next_line):
                # 确认是卷目格式：合并短行作为标题
                title_parts = [s for _, s in head_short_lines[:3]]
                merged = ' · '.join(title_parts)
                return _clean_title(merged)

    for idx, raw in enumerate(lines):
        s = raw.strip()
        L = len(s)
        if L == 0 or L > LEN_MAX:
            continue
        if PATTERN_CHAPTER.search(s):
            strong_hits.append((idx, s))
            continue
        if _is_chapter_number_only(s, lines, idx):
            strong_hits.append((idx, s))
            continue
        if L < LEN_SHORT:
            prev_blank = idx > 0 and _is_blank(lines[idx - 1])
            next_blank = idx < len(lines) - 1 and _is_blank(lines[idx + 1])
            has_cw = any(_is_meaningful_word(w) for w in _content_words(s))
            # 放宽条件：首行（idx==0）或末行也算"边界"
            is_boundary = (idx == 0 or idx == len(lines) - 1 or prev_blank or next_blank)
            if is_boundary and has_cw and not _is_text_fragment(s):
                direct_hits.append((idx, s))
                continue
        # 正文片段直接跳过打分，避免进入 candidates
        if _is_text_fragment(s):
            continue
        # 年鉴类密集正文过滤：如果行前后都没有空行，且不在 chunk 边界，
        # 说明是密集正文的一部分，降权处理（减5分，避免被选为标题）
        prev_blank = idx > 0 and _is_blank(lines[idx - 1])
        next_blank = idx < len(lines) - 1 and _is_blank(lines[idx + 1])
        is_boundary = (idx == 0 or idx == len(lines) - 1)
        dense_text_penalty = 0.0
        if not (prev_blank or next_blank or is_boundary):
            dense_text_penalty = -5.0
        sc = _score_line(raw, lines, idx, chunk_freq) + dense_text_penalty
        if sc > -50:
            candidates.append((sc, idx, s))

    if strong_hits:
        return _clean_title(strong_hits[0][1])

    # 年鉴类【】条目识别：扫描前 30 行，提取【XXX】作为标题
    # 年鉴/方志用【】标记条目，如【概况】【工业经济】【价格认证】等
    # 这是年鉴类文档最强的结构信号
    yearbook_marks = _extract_yearbook_marks(lines[:30])
    if yearbook_marks:
        return yearbook_marks

    if direct_hits:
        return _clean_title(direct_hits[0][1])
    if candidates:
        candidates.sort(reverse=True)
        # 检测是否为密集文本（无空行分隔的 PDF 提取文本）
        # 密集文本中行间没有结构信号，候选行大概率是正文片段，需提高门槛
        blank_count = sum(1 for raw in lines if _is_blank(raw))
        is_dense_text = blank_count == 0
        # 密集文本要求更高分数（3.0），普通文本保持 1.0
        min_score = 3.0 if is_dense_text else 1.0
        # 从高分到低分找第一个"不是正文片段"的标题
        for sc, idx, title in candidates:
            if sc < min_score:
                break
            # 最终过滤：排除正文片段、纯数字、混合内容等
            if _is_text_fragment(title):
                continue
            # 排除纯数字（如"3502646.8"）
            if re.match(r'^[\d.\-]+$', title):
                continue
            # 排除含大量数字的行（如"第34号提案147b 房地产管理93"）
            digit_ratio = sum(1 for c in title if c.isdigit()) / max(len(title), 1)
            if digit_ratio > 0.3:
                continue
            # 排除以括号开头的人名列表（如"（张启文）十字分局..."）
            if title.startswith('（') or title.startswith('('):
                continue
            # 密集文本额外过滤：候选行必须前后有空行或处于边界
            # （密集文本中无空行分隔的行大概率是正文）
            if is_dense_text:
                prev_blank = idx > 0 and _is_blank(lines[idx - 1])
                next_blank = idx < len(lines) - 1 and _is_blank(lines[idx + 1])
                is_boundary = (idx == 0 or idx == len(lines) - 1)
                if not (prev_blank or next_blank or is_boundary):
                    continue
            return _clean_title(title)

    # fallback：高频实词拼接
    seen = set()
    top_words = []
    for w, c in chunk_freq.most_common(50):
        if not _is_meaningful_word(w):
            continue
        if w in seen:
            continue
        if any(w in sw or sw in w for sw in seen):
            continue
        seen.add(w)
        top_words.append(w)
        if len(top_words) >= 2:
            break
    if top_words:
        joined = ''.join(top_words)
        if 3 <= len(joined) <= 8:
            return _clean_title(joined)

    # fallback：扫描所有行，找第一个"像标题"的短行
    # 年鉴类 PDF 提取的文本常是密集正文（每行 10-15 字），首行往往是正文片段
    # 策略：严格要求前后有空行包围（或处于行首/行尾边界），避免取到正文片段
    for idx, raw in enumerate(lines):
        s = raw.strip()
        if not s:
            continue
        # 跳过正文片段：行尾有标点、单字、数字+单位开头、页码混入等
        if _is_text_fragment(s):
            continue
        if s[0] in '的了是在我有和就也不再也都还又':
            continue
        if s[0] in '""「『':
            continue
        if len(s) < 2:
            continue
        # 行尾是"年/月/日"等时间词 → 正文片段
        if s[-1] in '年月日时分秒':
            continue
        # 偏好短行（标题通常 ≤20 字）
        L = len(s)
        if L > 20:
            continue
        # 严格要求前后至少有一个空行（或处于 chunk 边界）
        # 年鉴正文是密集文本，没有空行分隔；标题行通常前后有空行
        prev_blank = idx > 0 and _is_blank(lines[idx - 1])
        next_blank = idx < len(lines) - 1 and _is_blank(lines[idx + 1])
        is_chunk_boundary = (idx == 0 or idx == len(lines) - 1)
        # 必须前后有空行或处于 chunk 边界
        if not (prev_blank or next_blank or is_chunk_boundary):
            continue
        # 如果是 chunk 边界（首行/末行），还需要验证下一行/上一行是长行（正文）
        # 避免取到被截断的正文首行
        if idx == 0 and not next_blank:
            # 首行且下一行不是空行 → 可能是密集正文的首行
            # 只有当首行很短（≤8字）时才认为是标题
            if L > 8:
                continue
        return _clean_title(s[:20])

    # 最终 fallback：返回空，不让正文片段当标题
    return ""


def _is_text_fragment(s: str) -> bool:
    """判断一行是否是正文片段而非标题。

    年鉴类 PDF 提取的文本常在句子中间换行，导致首行是半句话。
    这些片段有明显特征：行尾标点、数字+单位开头、页码混入等。
    """
    if not s:
        return True
    # 行尾标点（句号、逗号、顿号等）→ 正文片段
    if _ends_with_punct(s):
        return True
    # 行尾顿号/逗号 → 正文并列项
    if s[-1] in END_COMMA:
        return True
    # 单字 → 信息量不足
    if len(s) < 2:
        return True
    # 纯数字/数字+小数点（如"3502646.8"、"85.23"）→ 正文数据
    if re.match(r'^[\d.\-,\s]+$', s):
        return True
    # 数字+单位开头 → 正文（如"年12月31日"、"50万元"）
    if PATTERN_NUM_UNIT_START.search(s):
        return True
    # 页码混入 → 正文（如"县情概览·65"）
    if PATTERN_PAGE_NUM_SUFFIX.search(s):
        return True
    # 行尾省略号 → 正文片段
    if s.endswith('...') or s.endswith('……') or s.endswith('....'):
        return True
    # 以闭合括号开头 → chunk 边界残片
    if s[0] in START_CLOSE:
        return True
    # 数字占比过高（>40%）→ 正文数据片段
    digit_count = sum(1 for c in s if c.isdigit())
    if digit_count / max(len(s), 1) > 0.4:
        return True
    # 行首是【但没有闭合的】→ 残缺的条目标记（如"【民生工"）
    if s.startswith('【') and '】' not in s:
        return True
    # 行首是《但没有闭合的》→ 残缺的书名号
    if s.startswith('《') and '》' not in s:
        return True
    # 短行（≤6字）中间含顿号 → 正文并列项残片（如"集中、商"、"办、烟草专"）
    if len(s) <= 6 and '、' in s:
        return True
    # 行首是"一二三四..."等中文数字+顿号 → 正文列表项（如"一、xxx"）
    if re.match(r'^[一二三四五六七八九十]+、', s):
        return True
    # 行中间含句号/问号/感叹号/逗号/分号/顿号 → 正文片段（标题不会在中间有这些标点）
    # 如"该公司投保。不仅对2007年"是正文，"约的缘故，故而责罚右贤王"是正文
    # "地保护工作；建成昌明220千伏输变"是正文（含分号）
    # "按照企业向园区集中、居民向小区"是正文（含顿号）
    # "庆祝建党87周年"是标题（无中间标点）
    if any(p in s[1:-1] for p in '。！？!?，,；;、'):
        return True
    # 行中间含闭合括号（)）】》」』）→ 正文片段残片
    # 如"中随迁2名）移民来本县水鸣乡安家"是正文（含）在中间）
    if any(p in s[1:-1] for p in '）)】》」』'):
        return True
    # 行内嵌入3位以上数字（且非年份）→ 正文数据片段
    # 如"户402名移民来本县十字镇安家落"含"402"→正文
    # 如"建成昌明220千伏输变"含"220"→正文
    # 年份（19xx/20xx）不在此列，因为标题可能含年份如"2008年工作回顾"
    embedded_nums = re.findall(r'\d{3,}', s)
    if embedded_nums:
        for n in embedded_nums:
            if not (n.startswith('19') and len(n) == 4) and not (n.startswith('20') and len(n) == 4):
                return True
    # 行尾是"年/月/日"等时间词且长度>5 → 正文片段（如"已超过一百二十年"）
    # 标题不会以时间词结尾，除非是短标题如"元年"（≤5字）
    if s[-1] in '年月日时分秒' and len(s) > 5:
        return True
    # 表格行：含表格框线符号（┃━╋┫┣┗┛┓┏）→ 表格数据，不是标题
    if any(c in s for c in '┃━╋┫┣┗┛┓┏┳┻╂'):
        return True
    # 索引行：行尾是"数字+字母"（如"实战化训练111b"、"信贷规模169a"）→ 索引条目，不是标题
    # 年鉴末尾索引页每行形如"词条+页码+栏号(a/b)"
    if re.search(r'\d{1,4}[ab]$', s):
        return True
    # 纯页码行（如"164.164"）→ 页码片段
    if re.match(r'^[\d.\s]+$', s) and len(s) < 15:
        return True
    # 正文动词开头 → 正文片段（年鉴/方志类常见句首）
    # 如"坚持举全"、"按照企业向园区集中"、"距还是干部思想观念"
    if PATTERN_BODY_TEXT_START.match(s):
        return True
    return False


def _clean_title(title: str) -> str:
    """清洗标题：去除页码后缀、尾部省略号、多余空白。"""
    if not title:
        return ""
    s = title.strip()
    # 去除页码后缀（如"县情概览·65" → "县情概览"）
    s = PATTERN_PAGE_NUM_SUFFIX.sub('', s)
    # 去除尾部省略号
    while s.endswith('...') or s.endswith('……') or s.endswith('....') or s.endswith('.'):
        s = s.rstrip('.…·')
    return s.strip()


def _extract_yearbook_marks(lines: list, max_marks: int = 3) -> str:
    """从年鉴类文本中提取【】条目标题。

    年鉴/方志用【XXX】标记条目，如【概况】【工业经济】【价格认证】等。
    扫描前几行，提取【】内的文字作为标题。

    如果有多个【】条目，取前 max_marks 个用 · 拼接。
    如果只有一个【】条目，直接返回其内容。
    """
    marks = []
    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        # 整行是【XXX】或【XXX】+ 少量说明
        m = PATTERN_YEARBOOK_MARK_LINE.match(s)
        if m:
            mark_text = m.group(1).strip()
            if mark_text and mark_text not in marks:
                marks.append(mark_text)
                if len(marks) >= max_marks:
                    break
            continue
        # 行内含【XXX】但不是整行（可能是正文中的条目引用）
        # 只有当行首就是【时才识别，避免正文中的【】被误提取
        if s.startswith('【'):
            for m in PATTERN_YEARBOOK_MARK.finditer(s):
                mark_text = m.group(1).strip()
                if mark_text and mark_text not in marks:
                    marks.append(mark_text)
                    if len(marks) >= max_marks:
                        break
            if len(marks) >= max_marks:
                break

    if not marks:
        return ""
    if len(marks) == 1:
        return marks[0]
    return ' · '.join(marks)
