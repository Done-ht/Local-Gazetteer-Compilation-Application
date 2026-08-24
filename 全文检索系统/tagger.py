"""chunk 标签提取：基于 jieba.analyse 的 tf-idf + 词性过滤。

策略：
  - 用 jieba.analyse.extract_tags 提取 top_k 个关键词
  - allowPOS 限定为名词类（n/ns/nt/nr/nz），过滤掉"的/是/在"等虚词
  - 标签存入 chunk JSON 的 tags 字段，导入时自动生成

性能：
  - 单 chunk 约 1-3ms（纯 CPU，10 万 chunk 约 3-5 分钟）
  - 不调用 LLM，零网络成本
  - jieba 词典首次加载约 0.8s，之后常驻内存

POS 说明（jieba 词性标注集）：
  n  = 普通名词（如"商税"、"职能"）
  ns = 地名（如"涿县"、"郎溪县"）
  nt = 机构名（如"人民政府"、"厘金局"）
  nr = 人名（如"刘备"、"诸葛亮"）
  nz = 其他专有名词（如"市场准入"）
"""
from __future__ import annotations

import os
from typing import List

# jieba 首次 import 较慢（加载词典），延迟到首次调用时加载
_jieba_loaded = False


def _ensure_jieba():
    """延迟加载 jieba 词典。"""
    global _jieba_loaded
    if not _jieba_loaded:
        import jieba  # noqa: F401
        import jieba.analyse  # noqa: F401
        _jieba_loaded = True


# 允许的词性（名词类）
ALLOW_POS = ('n', 'ns', 'nt', 'nr', 'nz')

# 标签最小长度：单字标签多为噪声（如"年"、"月"），要求 ≥2 字
MIN_TAG_LEN = 2

# 显式停用词：jieba 词性过滤后的兜底
# 包含：时间词、量词、文言虚词残留、常见但无意义的双字词
_STOP_WORDS = {
    # 时间/年代
    "今年", "去年", "明年", "前年", "当时", "此时", "届时", "早年", "晚年",
    "年初", "年末", "年底", "年初", "月中", "月度",
    # 称谓泛词
    "先生", "夫人", "氏族", "族人",
    # 文言虚词双字
    "于是", "然后", "然而", "所以", "因为", "由于", "至于", "虽然",
    "但是", "如果", "即使", "无论", "不但", "而且", "并且", "或者",
    # 通用泛词
    "一切", "所有", "其他", "另外", "某些", "有些", "同一", "同样",
    "可能", "应该", "必须", "可以", "能够", "需要",
    "进行", "开始", "结束", "完成", "实现", "继续",
    "情况", "状况", "方面", "问题", "事情", "事物", "内容", "形式",
    "方法", "方式", "手段", "途径", "过程", "步骤",
    "结果", "效果", "影响", "作用", "意义", "目的", "目标",
    "时期", "阶段", "时期", "期间", "中间", "其中",
    # 通用动作
    "进行", "实行", "实施", "执行", "开展", "推进", "推动",
}


def extract_tags(text: str, top_k: int = 10) -> List[str]:
    """从文本中提取 top_k 个标签。

    参数：
        text: chunk 文本
        top_k: 返回标签数量上限，默认 10

    返回：
        标签列表，按 tf-idf 权重降序

    性能：
        单次调用约 1-3ms（10 万字以内），首次调用多约 0.8s（加载词典）
    """
    if not text or len(text) < 10:
        return []

    _ensure_jieba()
    import jieba.analyse

    # 取 top_k * 2 个候选，过滤后取前 top_k
    # 多取一些是为了应对停用词/单字过滤后数量不足
    candidates = jieba.analyse.extract_tags(
        text,
        topK=top_k * 2,
        allowPOS=ALLOW_POS,
        withWeight=False,
    )

    tags: List[str] = []
    seen = set()
    for w in candidates:
        # 长度过滤
        if len(w) < MIN_TAG_LEN:
            continue
        # 停用词过滤
        if w in _STOP_WORDS:
            continue
        # 去重
        if w in seen:
            continue
        seen.add(w)
        tags.append(w)
        if len(tags) >= top_k:
            break

    return tags


def extract_tags_for_chunk(chunk_text: str, top_k: int = 10) -> List[str]:
    """对 chunk 文本提取标签（与 extract_tags 相同，命名上区分用途）。"""
    return extract_tags(chunk_text, top_k=top_k)


def batch_extract_tags(
    chunks_iter,
    top_k: int = 10,
    progress_callback=None,
):
    """批量提取标签（生成器）。

    参数：
        chunks_iter: 迭代器，每个元素是 (chunk_id, text) 元组
        top_k: 每个 chunk 的标签数上限
        progress_callback: 可选回调 (current, total) -> None

    生成：
        (chunk_id, tags) 元组
    """
    _ensure_jieba()  # 预加载，避免首个 chunk 计入加载耗时

    total = 0
    for item in chunks_iter:
        if isinstance(item, tuple) and len(item) == 2:
            chunk_id, text = item
        else:
            # 兼容 dict 输入
            chunk_id = item.get("chunk_id", "")
            text = item.get("text", "")

        tags = extract_tags(text, top_k=top_k)
        yield (chunk_id, tags)

        total += 1
        if progress_callback:
            progress_callback(total, None)
