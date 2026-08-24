# -*- coding: utf-8 -*-
"""纠错编排：分块调用 DeepSeek 校对，定位错误，token 熔断。

纯逻辑、不依赖 Qt；通过回调报告进度，便于 QThread / CLI 复用。
"""
import json
import logging
import uuid

from .deepseek import DeepSeekClient
from .models import ErrorItem, Page, TokenUsage

logger = logging.getLogger(__name__)

CHUNK_SIZE = 2000        # 单块最大字数
VALID_TYPES = {"错别字", "语法", "标点", "逻辑"}

# 预置开关对应的 prompt 说明：开启时“需要检查”，关闭时“不要检查”
# 关闭项会显式告诉模型把该类问题视为可接受，避免它自由发挥误报。
_PRESET_PROMPT = {
    "check_cn_number_space": "数字与汉字之间的空格（如“销售 额”或“100 万”中间的空格）",
    "check_en_number_space": "英文单词与汉字之间的空格",
    "check_full_half_punct": "全角与半角标点混用（如中文段落里出现半角逗号“,”）",
    "check_redundant_space": "句中多余空格、连续空格",
    "check_number_format": "数字格式问题（千分位、单位书写不规范）",
}

_BASE_PROMPT = (
    "你是一名严谨的中文公文/年鉴校对专家。请校对用户给出的文本块，找出以下四类问题：\n"
    "1. 错别字（含专有名词漏字、多字）；\n"
    "2. 语法语病（成分残缺、搭配不当、句式杂糅等）；\n"
    "3. 标点误用；\n"
    "4. 数据/逻辑矛盾（前后数据冲突、计算不符、表述不合公文规范）。\n"
    "要求：\n"
    "- 只报确实有问题的地方，不要报可改可不改的风格问题；\n"
    "- 年份、数据、人名、地名必须与上下文一致才可判错；若原文年份/数据与"
    "前后文一致（如全文均在描述 2025 年的工作，则文中的 2025 年不应判为错误），"
    "严禁凭空臆断为错误年份并要求改成其他年份；\n"
    "- 每条错误的 quote 字段必须是原文中连续的一小段原句（逐字照抄，含标点），用于在原文中定位；\n"
    "- original 为其中有误的部分，suggestion 为修改后的对应部分；\n"
    "- 顺带提取本块中出现的人名、地名、机构名等实体放入 entities；\n"
    "- 严格输出 JSON，格式：\n"
    '{"errors": [{"quote": "...", "original": "...", "suggestion": "...", '
    '"error_type": "错别字|语法|标点|逻辑", "reason": "..."}], "entities": ["..."]}\n'
    "没有错误时 errors 返回空数组。"
)

# 向后兼容：旧代码若直接 import SYSTEM_PROMPT，仍可拿到一份可用 prompt
SYSTEM_PROMPT = _BASE_PROMPT

# ---------- OCR 识别纠错专用提示词 ----------
# 用户输入的文件已是 OCR 识别产物（由扫描件/图片识别而来），可能存在识别错误。
# 该模式聚焦于发现并修正这些 OCR 特有的错误，而非公文语病审查。
OCR_PROMPT = (
    "你是一名中文古籍/公文 OCR 识别结果校对专家。用户给你的文本是 OCR 识别的产物，"
    "可能存在各种识别错误。请重点排查并修正以下 OCR 特有问题：\n"
    "1. 形近字误识：字形相近导致的错字，如 已/己/巳、未/末、土/士、日/曰、"
    "干/千、人/入、甲/由/申、戊/戌/戍、帅/师、刺/剌、享/亨、"
    "荼/茶、赢/嬴、汨/汩、崇/祟等；\n"
    "2. 数字误识：0/O/o、1/l/I/|、2/Z、5/S、6/8、3/8/9 混淆；"
    "中文数字与阿拉伯数字混用错乱（如“三”识为“王”、“五”识为“丑”）；\n"
    "3. 标点误识：句号/逗号混淆（。/，）、引号方向错误（“”识反或识为「」）、"
    "顿号/逗号错位、冒号/分号混淆、破折号识为连字符等；\n"
    "4. 漏字漏行漏句：OCR 漏识整行/整句导致上下文断裂，表现为句子突然中断、"
    "前后不衔接、段落中间出现残句；\n"
    "5. 重复识别：同一字/词/句被重复识别多次（如“于于”“的的工作”）；\n"
    "6. 乱码字符：识别出无意义的符号、外文字符或 Unicode 怪字符；\n"
    "7. 版式错位：表格/分栏/标题层级识别错位，导致内容串行或归属错误；\n"
    "8. 专名错识：人名、地名、官职名、年号、书名号内的字被识别成同形异义字"
    "（如“李鸿章”识为“李鸿童”、“光绪”识为“光堵”、“荆州”识为“刑州”等需结合上下文判断）。\n"
    "要求：\n"
    "- 仅报告确有 OCR 识别错误的片段，不要把原文的文言用法、古字、异体字当作错误"
    "（如“尅”“俾”“厥”等为文言常用字，非 OCR 错误）；\n"
    "- 漏行漏句类问题：若发现句子明显断裂或上下文不衔接，请报告，"
    "original 填写断裂处可见文字，suggestion 给出可能的补全（如无法确定则填“[疑似漏字]”）；\n"
    "- 重复识别：original 填写重复片段，suggestion 填写去重后的正确片段；\n"
    "- 乱码字符：original 填写乱码，suggestion 填写应识别的正确字符或“[无法识别]”；\n"
    "- quote 字段必须是原文中连续的一小段原句（逐字照抄），用于定位；\n"
    "- 顺带提取本块中出现的人名、地名、机构名、年号等实体放入 entities；\n"
    "- 严格输出 JSON：\n"
    '{"errors": [{"quote": "...", "original": "...", "suggestion": "...", '
    '"error_type": "错别字|语法|标点|逻辑", "reason": "..."}], "entities": ["..."]}\n'
    "没有错误时 errors 返回空数组。"
)


def build_ocr_system_prompt(rules: dict = None) -> str:
    """OCR 识别纠错模式的 system prompt 组装。

    与普通模式一样支持预置开关与自定义规则，但基础提示词替换为 OCR_PROMPT。
    """
    if not rules:
        rules = {"switches": {}, "custom": []}
    switches = rules.get("switches") or {}
    custom = [r for r in (rules.get("custom") or []) if isinstance(r, str) and r.strip()]

    parts = [OCR_PROMPT]

    off_items, on_items = [], []
    for key, desc in _PRESET_PROMPT.items():
        if switches.get(key):
            on_items.append(desc)
        else:
            off_items.append(desc)

    if off_items:
        parts.append("【以下风格类问题视为可接受，不要报告】\n" + "；".join(off_items) + "。")
    if on_items:
        parts.append("【以下风格类问题需要检查并报告】\n" + "；".join(on_items) + "。")

    if custom:
        parts.append("【附加校对规则】\n" + "\n".join(f"- {r}" for r in custom))

    return "\n\n".join(parts)


def build_system_prompt(rules: dict = None) -> str:
    """根据规则配置组装 system prompt。

    rules: {"switches": {key: bool}, "custom": [str, ...]}
    - 预置开关关闭的项，显式告诉模型“不要检查”，避免模型自由发挥误报；
    - 预置开关开启的项，告诉模型“需要检查”；
    - custom 中的每条规则作为附加检查要求追加。
    缺省（rules 为空）时等价于所有预置开关关闭——即不检查这些风格问题。
    """
    if not rules:
        rules = {"switches": {}, "custom": []}
    switches = rules.get("switches") or {}
    custom = [r for r in (rules.get("custom") or []) if isinstance(r, str) and r.strip()]

    parts = [_BASE_PROMPT]

    off_items, on_items = [], []
    for key, desc in _PRESET_PROMPT.items():
        if switches.get(key):
            on_items.append(desc)
        else:
            off_items.append(desc)

    if off_items:
        parts.append("【以下风格类问题视为可接受，不要报告】\n" + "；".join(off_items) + "。")
    if on_items:
        parts.append("【以下风格类问题需要检查并报告】\n" + "；".join(on_items) + "。")

    if custom:
        parts.append("【附加校对规则】\n" + "\n".join(f"- {r}" for r in custom))

    return "\n\n".join(parts)


# ---------- 二次复核 ----------
REVIEW_SYSTEM = (
    "你是校对复核员。下面给你一段原文和一组校对意见。请逐条判断每条意见是否成立。\n"
    "判断标准：\n"
    "- confirm：原文确实有误，且修改建议正确；\n"
    "- uncertain：无法确定，或属于可改可不改的风格偏好；\n"
    "- reject：原文其实正确，校对意见属于误报。\n"
    "重要原则：宁可保留为 uncertain，也不要轻易判 reject。"
    "只有当你有充分证据（如前后文年份/数据明确冲突）证明原文正确、修改会引入错误时，才判 reject。"
    "拿不准时一律判 uncertain。\n"
    "重点核对：前后文出现的年份、数据、人名、地名、机构名是否一致。"
    "例如前文均为 2025 年，而某条意见称“此处应为 2024 年”，则该条可判 reject。\n"
    "严格输出 JSON：\n"
    '{"reviews": [{"quote": "对应原校对意见中的 original", '
    '"verdict": "confirm|reject|uncertain", "reason": "..."}]}'
)


def _parse_reviews(content: str):
    """解析复核响应，返回 reviews 列表"""
    text = content.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0].strip()
    data = json.loads(text)
    return data.get("reviews", [])


def _review_chunk(client: DeepSeekClient, chunk_text: str, chunk_errors: list,
                  prev_tail: str, next_head: str, review_context: int):
    """对单块的错误做二次复核。

    返回 (matched, usage)，matched: [(ErrorItem, verdict:str, reason:str), ...]
    verdict 为 confirm/reject/uncertain；未匹配到复核意见时为空串（由调用方保留启发式置信度）。
    """
    if not chunk_errors:
        return [], TokenUsage()
    prev = prev_tail[-review_context:] if prev_tail else ""
    nxt = next_head[:review_context] if next_head else ""
    errs_brief = [
        {"quote": e.original, "original": e.original, "suggestion": e.suggestion,
         "error_type": e.error_type, "reason": e.reason}
        for e in chunk_errors
    ]
    parts = [f"【原文】\n{chunk_text}"]
    if prev:
        parts.append(f"【前文摘要】\n…{prev}")
    if nxt:
        parts.append(f"【后文摘要】\n…{nxt}")
    parts.append(f"【校对意见】\n{json.dumps(errs_brief, ensure_ascii=False)}")
    messages = [
        {"role": "system", "content": REVIEW_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
    content, usage = client.chat(messages, temperature=0.0)
    try:
        reviews = _parse_reviews(content)
    except (json.JSONDecodeError, AttributeError) as e:
        logger.warning("复核响应 JSON 解析失败: %s", e)
        reviews = []

    matched, used = [], set()
    for e in chunk_errors:
        verdict, reason = "", ""
        for j, r in enumerate(reviews):
            if j in used:
                continue
            rq = (r.get("quote") or "").strip()
            if rq and (rq == e.original or rq in e.original or e.original in rq):
                verdict = (r.get("verdict") or "").lower()
                reason = (r.get("reason") or "").strip()
                used.add(j)
                break
        matched.append((e, verdict, reason))
    return matched, usage


def _split_chunks(pages: list):
    """把页切成 ≤CHUNK_SIZE 的块（按行/段边界），返回 [(page_num, chunk_text), ...]"""
    chunks = []
    for page in pages:
        text = page.text
        if len(text) <= CHUNK_SIZE:
            if text.strip():
                chunks.append((page.page_num, text))
            continue
        # 过长页按行拆分
        current, size = [], 0
        for line in text.splitlines(keepends=True):
            if current and size + len(line) > CHUNK_SIZE:
                chunks.append((page.page_num, "".join(current)))
                current, size = [], 0
            current.append(line)
            size += len(line)
        if current:
            chunks.append((page.page_num, "".join(current)))
    return chunks


def _build_user_prompt(chunk_text: str, entities: list, prev_tail: str,
                       next_head: str = "") -> str:
    """组装用户消息：前文摘要 + 待校对文本 + 后文摘要

    next_head 让模型在校对当前块时也能看到后文开头，便于跨页一致性核对
    （例如第3页末尾与第4页开头的年份/数据是否一致）。
    """
    parts = []
    if entities:
        parts.append("【已出现的实体（人名/地名/机构名，供核对一致性）】\n" + "、".join(entities[-50:]))
    if prev_tail:
        parts.append(f"【前文结尾摘录】\n…{prev_tail}")
    parts.append(f"【待校对文本】\n{chunk_text}")
    if next_head:
        parts.append(f"【后文开头摘录】\n{next_head}…")
    return "\n\n".join(parts)


def _parse_response(content: str):
    """解析模型 JSON 响应。

    容忍：
    - markdown 代码块 ```json ... ``` 包裹
    - 模型输出被 max_tokens 截断导致 JSON 不完整（Unterminated string / EOF）：
      逐步回退最后一个 } 或 ] 之前的内容，丢弃未完成的最后一条错误，
      只要能解析出已完整的部分就返回，避免整块丢弃。
    """
    text = content.strip()
    if text.startswith("```"):
        # 去掉 ```json ... ``` 包裹（截断时可能没有结尾 ```）
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        if "```" in text:
            text = text.rsplit("```", 1)[0].strip()

    try:
        data = json.loads(text)
        return data.get("errors", []), data.get("entities", [])
    except json.JSONDecodeError:
        # 截断容错：逐步砍掉末尾，尝试补齐闭合括号
        # 找到最后一个完整的 } (一条错误结束) 或 ] (errors 数组结束)
        # 策略：从末尾向前找最后一个 '}'，截到那里再加 ']' 闭合数组、加 '}' 闭合对象
        last_obj_close = text.rfind("}")
        if last_obj_close < 0:
            # 连一个完整对象都没有，无法挽救
            logger.warning("JSON 解析失败且无法挽救（无完整对象）: %s", text[:200])
            return [], []
        truncated = text[:last_obj_close + 1]
        # 数组可能已经在 } 后闭合了（如 }]}），检查是否需要补 ]
        # 简单做法：尝试在 } 后加 ]}，若还不行就逐步回退到上一个 }
        for suffix in ["]}", "]}]", "}"]:
            try:
                data = json.loads(truncated + suffix)
                logger.warning(
                    "JSON 截断容错：丢弃未完成最后一条，恢复 %d 条错误",
                    len(data.get("errors", [])))
                return data.get("errors", []), data.get("entities", [])
            except json.JSONDecodeError:
                continue
        # 再回退到上一个 }
        prev_close = text.rfind("}", 0, last_obj_close)
        while prev_close >= 0:
            for suffix in ["]}", "]}]", "}"]:
                try:
                    data = json.loads(text[:prev_close + 1] + suffix)
                    logger.warning(
                        "JSON 截断容错（回退）：恢复 %d 条错误",
                        len(data.get("errors", [])))
                    return data.get("errors", []), data.get("entities", [])
                except json.JSONDecodeError:
                    continue
            prev_close = text.rfind("}", 0, prev_close)
        logger.warning("JSON 解析失败且所有容错均失败: %s", text[:200])
        return [], []


# ---------- 规则硬过滤 ----------
# 常见全角→半角标点映射
_FULL_HALF = {
    "，": ",", "。": ".", "；": ";", "：": ":", "？": "?", "！": "!",
    "（": "(", "）": ")", "“": '"', "”": '"', "‘": "'", "’": "'",
    "【": "[", "】": "]", "《": "<", "》": ">",
}


def _normalize_punct(s: str) -> str:
    for f, h in _FULL_HALF.items():
        s = s.replace(f, h)
    return s


def _is_space_only_diff(original: str, suggestion: str) -> bool:
    """两者仅空格（含全角空格）有差异"""
    if original == suggestion:
        return False
    a = original.replace(" ", "").replace("\u3000", "")
    b = suggestion.replace(" ", "").replace("\u3000", "")
    return a == b


def _is_punct_fullhalf_diff(original: str, suggestion: str) -> bool:
    """两者仅全角/半角标点有差异"""
    if original == suggestion:
        return False
    return _normalize_punct(original) == _normalize_punct(suggestion)


def _is_excluded_by_rule(item: ErrorItem, rules: dict) -> bool:
    """判断一条检出是否属于「已被规则关闭」的风格类问题，应硬排除。
    - 任意空格开关关闭且差异仅为空格 → 排除
    - 全半角标点开关关闭且差异仅为全半角标点 → 排除
    """
    if not rules:
        return False
    sw = rules.get("switches") or {}
    o, s = item.original, item.suggestion
    space_off = (not sw.get("check_cn_number_space")
                 or not sw.get("check_en_number_space")
                 or not sw.get("check_redundant_space"))
    if space_off and _is_space_only_diff(o, s):
        return True
    if not sw.get("check_full_half_punct") and _is_punct_fullhalf_diff(o, s):
        return True
    return False


def proofread(pages: list, client: DeepSeekClient, token_limit: int = 1000000,
              progress_cb=None, on_limit_exceeded=None, rules: dict = None,
              review: bool = False, review_context: int = 800, review_cb=None,
              status_cb=None, context_prev: str = "", context_next: str = "",
              ocr_mode: bool = False):
    """对 pages 跑完整纠错。

    参数：
        progress_cb(chunk_index, total_chunks, chunk_usage, total_usage)
        on_limit_exceeded(total_usage) -> bool；返回 False 表示中止
        rules: {"switches": {key: bool}, "custom": [str, ...]}；为空时使用默认策略
        review: 是否对每块检出的错误做二次复核（降低误报、区分明显/存疑）
        review_context: 复核时给模型参考的前/后文摘要字数
        review_cb(chunk_index, total_chunks, confirmed, uncertain, rejected, excluded)
        status_cb(message): 每次 API 请求前回调，用于 UI 展示“正在分析…”避免误以为卡住
        context_prev / context_next: 单页纠错模式下的前文/后文摘要（滑动式翻页用），
            传入后首轮 prev_tail 与末轮 next_head 会使用它们，让复核能拿到跨页上下文
        ocr_mode: OCR 识别纠错模式，使用 OCR 专用系统提示词，聚焦 OCR 特有错误
    返回：(list[ErrorItem], TokenUsage)
    """
    page_texts = {p.page_num: p.text for p in pages}
    chunks = _split_chunks(pages)
    total_chunks = len(chunks)

    system_prompt = build_ocr_system_prompt(rules) if ocr_mode else build_system_prompt(rules)

    errors, entities = [], []
    prev_tail = context_prev or ""
    total_usage = TokenUsage()
    consumed = {}  # page_num -> 已消耗到的偏移，保证重复 quote 取未消耗的第一个

    for i, (page_num, chunk_text) in enumerate(chunks, start=1):
        # 后文摘要：下一块开头 200 字；末块用 context_next（跨页后文）
        if i < len(chunks):
            next_head = chunks[i][1][:200]
        else:
            next_head = context_next or ""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _build_user_prompt(
                chunk_text, entities, prev_tail, next_head)},
        ]
        if status_cb:
            status_cb(f"正在分析第 {i}/{total_chunks} 块（第 {page_num} 页）…")
        content, usage = client.chat(messages)
        total_usage.add(usage)

        # _parse_response 内部已做截断容错，不会再抛 JSONDecodeError；
        # 仅返回 ([], []) 时说明该块完全无法挽救
        raw_errors, new_entities = _parse_response(content)

        for e in new_entities:
            if isinstance(e, str) and e and e not in entities:
                entities.append(e)

        # 先定位本块所有错误
        chunk_errors = []
        excluded = 0
        for raw in raw_errors:
            item = _locate(raw, page_num, page_texts, consumed)
            if item is None:
                continue
            # 规则硬过滤：已被规则关闭的风格类问题（如空格）直接排查出去，不进复核也不进结果
            if _is_excluded_by_rule(item, rules):
                excluded += 1
                logger.info("规则排除(不报): %s → %s", item.original[:30], item.suggestion[:30])
                continue
            chunk_errors.append(item)

        # 二次复核
        chunk_usage = usage
        confirmed = uncertain = rejected = 0
        if review and chunk_errors:
            next_head = chunks[i][1] if i < len(chunks) else (context_next or "")
            if status_cb:
                status_cb(f"正在复核第 {i}/{total_chunks} 块的 {len(chunk_errors)} 条检出…")
            matched, rusage = _review_chunk(
                client, chunk_text, chunk_errors, prev_tail, next_head, review_context)
            total_usage.add(rusage)
            # 合并本轮 token 用于进度展示
            chunk_usage = TokenUsage(
                usage.prompt_tokens + rusage.prompt_tokens,
                usage.completion_tokens + rusage.completion_tokens,
                usage.total + rusage.total,
            )
            for item, verdict, vreason in matched:
                v = (verdict or "").lower()
                if v == "confirm":
                    item.confidence = "明确"
                    errors.append(item)
                    confirmed += 1
                elif v == "reject":
                    # 不丢弃：降级为存疑保留，让用户自行判断，避免漏掉有价值检出
                    item.confidence = "存疑"
                    if vreason:
                        item.reason = f"{item.reason}（复核疑似误报：{vreason}）"
                    else:
                        item.reason = f"{item.reason}（复核疑似误报）"
                    errors.append(item)
                    rejected += 1
                elif v == "uncertain":
                    item.confidence = "存疑"
                    if vreason:
                        item.reason = f"{item.reason}（复核存疑：{vreason}）"
                    errors.append(item)
                    uncertain += 1
                else:
                    # 未匹配到复核意见，保留启发式置信度
                    errors.append(item)
                    uncertain += 1
            if review_cb:
                review_cb(i, total_chunks, confirmed, uncertain, rejected, excluded)
        else:
            errors.extend(chunk_errors)
            confirmed = len(chunk_errors)
            if review_cb and (chunk_errors or excluded):
                review_cb(i, total_chunks, confirmed, 0, 0, excluded)

        prev_tail = chunk_text[-200:]
        if progress_cb:
            progress_cb(i, total_chunks, chunk_usage, total_usage)

        if token_limit and total_usage.total > token_limit:
            logger.warning("token 累计 %d 超过限额 %d", total_usage.total, token_limit)
            if on_limit_exceeded and on_limit_exceeded(total_usage) is False:
                break

    return errors, total_usage


# 中文数字 ↔ 阿拉伯数字 映射（用于 _locate 数字归一化匹配）
_CN_NUM_MAP = {"零": "0", "〇": "0", "一": "1", "二": "2", "两": "2", "三": "3",
               "四": "4", "五": "5", "六": "6", "七": "7", "八": "8", "九": "9",
               "十": "10", "百": "100", "千": "1000", "万": "10000", "亿": "100000000"}


def _normalize_spaces(s: str) -> str:
    """去掉所有空白字符（半角空格、全角空格、tab、换行），用于模糊匹配。
    注意：归一化后字符数变化，无法直接回映位置，必须配合 _find_with_offset 使用。
    """
    import re
    return re.sub(r"\s+", "", s).replace("\u3000", "")


def _build_space_map(s: str):
    """构建「去空白后字符 → 原字符串位置」的索引。
    返回 (no_space_str, list[int])：list[i] = no_space_str[i] 在原串中的下标。
    """
    no_space = []
    idx_map = []
    for i, ch in enumerate(s):
        if ch.isspace() or ch == "\u3000":
            continue
        no_space.append(ch)
        idx_map.append(i)
    return "".join(no_space), idx_map


def _find_in_text(text: str, candidates: list, start_from: int):
    """在 text 中查找候选字符串，返回 (idx, matched_text)。

    依次尝试：
      1. 精确匹配（先从 start_from 起，再退回全文）
      2. 标点全/半角归一化后模糊匹配（1:1 替换，位置一一对应可安全回映）
      3. 长串（≥12 字）前缀匹配（应对模型把原文截短或加了省略号）
      4. 去空白后子串匹配（应对"覆盖 372 名" vs "覆盖372名"）：
         去掉双方所有空白后匹配，再用字符位置映射表回映到原文位置
      5. 最长子串回退：取 candidate 中最长的、能在原文（去空白后）找到的连续子串

    全部失败返回 (-1, "")。
    """
    # 1. 精确匹配（consumed 之后优先；找不到则全文回退，不强制 idx>=start_from）
    #    原因：模型返回的错误顺序可能与原文不一致（跨句引用、乱序返回），
    #    若强制 idx>=start_from 会丢弃所有位置在 consumed 之前的错误。
    for cand in candidates:
        if not cand:
            continue
        idx = text.find(cand, start_from)
        if idx >= 0:
            logger.info("定位[1-精确] cand=%r → idx=%d", cand[:40], idx)
            return idx, text[idx:idx + len(cand)]
    # 全文回退：允许命中 consumed 之前的位置（优于直接丢弃错误）
    for cand in candidates:
        if not cand:
            continue
        idx = text.find(cand)
        if idx >= 0:
            logger.info("定位[1-精确全文] cand=%r → idx=%d (start_from=%d)", cand[:40], idx, start_from)
            return idx, text[idx:idx + len(cand)]

    # 2. 标点归一化模糊匹配（1:1 替换，长度不变）
    norm_text = _normalize_punct(text)
    for cand in candidates:
        if not cand:
            continue
        norm_cand = _normalize_punct(cand)
        if not norm_cand or len(norm_cand) != len(cand):
            continue
        ni = norm_text.find(norm_cand, start_from)
        if ni < 0:
            ni = norm_text.find(norm_cand)
        if ni >= 0:
            logger.info("定位[2-标点] cand=%r → idx=%d", cand[:40], ni)
            return ni, text[ni:ni + len(cand)]

    # 3. 长串前缀匹配（≥12 字）：模型有时把原文截短或加省略号
    for cand in candidates:
        if not cand or len(cand) < 12:
            continue
        prefix = cand[:12]
        idx = text.find(prefix, start_from)
        if idx < 0:
            idx = text.find(prefix)
        if idx >= 0:
            logger.info("定位[3-前缀] cand=%r → idx=%d", cand[:40], idx)
            end = min(idx + len(cand), len(text))
            return idx, text[idx:end]

    # 4. 去空白后子串匹配：应对"覆盖 372 名育儿家庭" vs "覆盖372名育儿家庭"
    #    原文与候选都去掉所有空白，匹配后用位置映射表回映
    no_space_text, idx_map = _build_space_map(text)
    # 同时把 start_from 转换到 no_space 坐标系
    ns_start = 0
    for i, orig_idx in enumerate(idx_map):
        if orig_idx >= start_from:
            ns_start = i
            break
    for cand in candidates:
        if not cand:
            continue
        no_space_cand = _normalize_spaces(cand)
        if not no_space_cand:
            continue
        ni = no_space_text.find(no_space_cand, ns_start)
        if ni < 0:
            ni = no_space_text.find(no_space_cand)  # 全文回退
        if ni >= 0:
            orig_start = idx_map[ni]
            orig_end = idx_map[ni + len(no_space_cand) - 1] + 1
            logger.info("定位[4-去空白] cand=%r → idx=%d", cand[:40], orig_start)
            return orig_start, text[orig_start:orig_end]

    # 5. 最长子串回退：candidate 去空白后从长到短试，找到原文（去空白）里的子串
    #    限制：≥8 字、≥75% 长度。优先从 consumed 之后找，找不到则全文回退（不丢弃错误）。
    #    应对模型在 original 里改了字、加了引号导致整体匹配失败，
    #    但其中较长的连续片段仍能在原文定位。
    for cand in candidates:
        if not cand:
            continue
        no_space_cand = _normalize_spaces(cand)
        if len(no_space_cand) < 8:
            continue  # 太短的子串容易误命中，跳过
        L = len(no_space_cand)
        for sub_len in range(L, 7, -1):
            if sub_len < L * 0.75:
                break  # 不再缩短到 75% 以下
            for sub_start in range(0, L - sub_len + 1):
                sub = no_space_cand[sub_start:sub_start + sub_len]
                # 先从 consumed 之后找
                ni = no_space_text.find(sub, ns_start)
                if ni < 0:
                    ni = no_space_text.find(sub)  # 全文回退，避免丢弃
                if ni >= 0:
                    orig_start = idx_map[ni]
                    orig_end = idx_map[ni + sub_len - 1] + 1
                    logger.info("定位[5-子串] cand=%r sub=%r → idx=%d", cand[:40], sub, orig_start)
                    return orig_start, text[orig_start:orig_end]

    logger.info("定位[失败] candidates=%r start_from=%d", [c[:30] for c in candidates if c], start_from)
    return -1, ""


def _shrink_to_original(matched: str, original: str):
    """在 matched（quote 命中范围）内定位 original 的实际子区间。

    返回 (start_in_matched, length)，表示 original 在 matched 中的起始与长度；
    找不到时返回 (0, len(matched))（退回整个 matched，consumed 用 quote 末尾）。

    多级匹配：
    1. 精确子串 find
    2. 去空白后子串 find + 位置映射回映
    3. 标点归一化后 find
    4. 最长公共子串（应对模型改了 original 里的个别字）
    """
    if not original or not matched:
        return 0, len(matched)
    # 1. 精确子串
    sub = matched.find(original)
    if sub >= 0:
        return sub, len(original)
    # 2. 去空白后子串 + 位置映射
    ns_matched, m_map = _build_space_map(matched)
    ns_original = _normalize_spaces(original)
    if ns_original and len(ns_original) <= len(ns_matched):
        sub = ns_matched.find(ns_original)
        if sub >= 0:
            return m_map[sub], m_map[sub + len(ns_original) - 1] + 1 - m_map[sub]
    # 3. 标点归一化后 find（1:1 替换，位置对齐）
    if len(matched) == len(original):
        sub = _normalize_punct(matched).find(_normalize_punct(original))
        if sub >= 0:
            return sub, len(original)
    # 4. 最长公共子串：模型可能在 original 里改了 1-2 个字
    #    找 original 与 matched 的最长连续公共子串，长度 ≥ original 的 60% 才接受
    best_start, best_len = 0, 0
    o_len = len(original)
    for i in range(o_len):
        for j in range(len(matched)):
            k = 0
            while (i + k < o_len and j + k < len(matched)
                   and original[i + k] == matched[j + k]):
                k += 1
            if k > best_len:
                best_len, best_start = k, j
    if best_len >= max(4, o_len * 0.6):
        return best_start, best_len
    return 0, len(matched)


def _locate(raw: dict, page_num: int, page_texts: dict, consumed: dict):
    """把模型返回的错误定位到页内偏移；定位失败返回 None

    关键设计：
    - consumed 推进到 original 末尾（而非 quote 末尾），避免长 quote 上下文
      占用过大范围导致后续重复短语被错误跳过。例如 quote="18.6亿元,同比增长9.3%"
      original="同比增长"，consumed 只推进到"同比增长"末尾，下次"同比增长"
      仍能命中后文出现位置。
    - located_original 始终取自原文实际切片（含原文的空格/标点），保证与
      suggestion 比较时反映真实差异。
    """
    if not isinstance(raw, dict):
        return None
    quote = (raw.get("quote") or "").strip()
    original = (raw.get("original") or "").strip()
    suggestion = raw.get("suggestion") or ""
    error_type = raw.get("error_type") or "错别字"
    reason = raw.get("reason") or ""
    if error_type not in VALID_TYPES:
        error_type = "错别字"

    text = page_texts.get(page_num, "")
    start_from = consumed.get(page_num, 0)

    # 候选定位字符串：quote 优先（更长上下文），其次 original
    candidates = [quote, original] if quote else ([original] if original else [])
    idx, matched = _find_in_text(text, candidates, start_from)

    if idx < 0:
        logger.warning("错误定位失败，丢弃: original=%s reason=%s", original[:30], reason[:50])
        return None

    # 在 matched（quote 命中范围）内缩小到 original 的实际子区间。
    # 这样 offset_start/offset_end 精准包裹错误本身，consumed 也只推进到
    # original 末尾，避免长 quote 上下文占用过大范围。
    sub_start, sub_len = _shrink_to_original(matched, original)
    offset_start = idx + sub_start
    offset_end = offset_start + sub_len
    located_original = text[offset_start:offset_end]

    # consumed 推进到 original 子区间末尾（而非整个 quote 末尾）
    consumed[page_num] = offset_end

    # 启发式置信度：原文与建议完全一致 = 无实质修改，视为存疑
    confidence = "存疑" if located_original == suggestion else "明确"

    # 调试日志：定位到的原文与模型 original 差异较大时记录，便于排查定位不准
    if original and _normalize_spaces(located_original) != _normalize_spaces(original):
        logger.debug(
            "定位偏差: model_original=%r located=%r quote=%r",
            original[:40], located_original[:40], quote[:40])

    return ErrorItem(
        id=uuid.uuid4().hex[:8],
        page_num=page_num,
        offset_start=offset_start,
        offset_end=offset_end,
        original=located_original,
        suggestion=suggestion,
        error_type=error_type,
        reason=reason,
        confidence=confidence,
    )
