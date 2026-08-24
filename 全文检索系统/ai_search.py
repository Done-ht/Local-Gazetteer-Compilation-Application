"""智能检索模块（依赖 deepseek + searcher + library + storage）。

工作流程：
    1. 用现有搜索引擎（parallel_search）检索与问题相关的 chunk；
    2. 读取命中 chunk 的完整文本，按相关度排序后拼接为上下文；
    3. 将上下文交给 DeepSeek 生成答案（支持流式）；
    4. 返回答案 + 引用来源（命中的 chunk 列表）。

返回结构：
    {
        "question": "用户问题",
        "model": "deepseek-v4-pro",
        "answer": "答案文本",
        "retrieval": { ... 搜索引擎统计 ... },
        "references": [
            {
                "index": 1,
                "library": "郎溪县志",
                "chunk_id": "zone_001/chunk_000123",
                "source_file": "第一节　农业资源.docx",
                "hit_count": 5,
                "snippet": "..."
            }
        ]
    }

流式版本 ai_search_stream 会先 yield 检索阶段事件，再 yield 模型生成事件：
    {"phase":"retrieval", "retrieval":{...}, "references":[...]}
    {"phase":"reasoning", "delta":"思考片段"}    # 思考模式才有
    {"phase":"content", "delta":"答案片段"}
    {"phase":"done", "usage":{...}}
"""
from __future__ import annotations

import json
import os
from typing import Iterator, List, Dict, Optional, Any

from library import LibraryRegistry
from searcher import parallel_search, search_related_keywords, search_semantic
from deepseek import DeepSeekClient, DeepSeekError, V4_FLASH, V4_PRO
from settings import SettingsStore
from userdata import auth_base_dir as _auth_base_dir


# ============================================================
#  从设置读取检索参数（替代硬编码常量）
# ============================================================


def _setting(key: str, default: Any = None) -> Any:
    """从 SettingsStore 读取单个参数（设置存于 biaoshifu 用户登录数据目录）。"""
    store = SettingsStore(_auth_base_dir())
    return store.get(key, default)


# 控制 context 总长度，避免超出模型上下文窗口
DEFAULT_MAX_CHUNKS = 8          # 最多取前 N 个命中 chunk（默认值，被 top_k 覆盖）
DEFAULT_MAX_CHARS_PER_CHUNK = 800  # 单个 chunk 最多取前 800 字
DEFAULT_MAX_TOTAL_CHARS = 6000  # context 总字数上限（默认值，按 top_k 动态扩展）
# 单条 chunk 平均字符估算，用于按 top_k 推算 context 总字数上限
CHARS_PER_CHUNK_BUDGET = 800

# ============================================================
#  搜索力度档位（替代固定 top_k 的截断策略）
# ============================================================
# 单个查询词的引用上限（避免高频词吃掉所有名额）
MAX_CHUNKS_PER_QUERY = 15
# 模型最多规划的查询词数
MAX_QUERIES = 5
# 召回总数超过此阈值时前端预警（性能/精度风险）
EFFORT_WARN_THRESHOLD = 50
# 精读模式：小 chunk 窗口（命中位置 ± N 字）
DEEPREAD_SNIPPET_WINDOW = 100
# 精读模式：expand 工具单次展开字数上限
DEEPREAD_EXPAND_MAX_CHARS = 10000
# 精读模式：每个查询词最多 expand 轮数
DEEPREAD_EXPAND_ROUNDS_PER_QUERY = 2
# 精读模式：默认小 chunk 数（200字×100=2万字，约等于原10个大chunk）
DEEPREAD_DEFAULT_MAX_MINI_CHUNKS = 100


def _compute_total_chars_budget(top_k: int) -> int:
    """根据 top_k 推算 context 总字数上限。

    保证 top_k 条引用都能进入 context（不被总字数提前截断），
    同时设上限避免 context 过长拖慢生成 / 超出模型上下文窗口。
    """
    # 至少容纳 top_k 条 × 单条预算；上限 32000（约 8k token，给问题+答案留余地）
    return min(32000, max(DEFAULT_MAX_TOTAL_CHARS, top_k * CHARS_PER_CHUNK_BUDGET))


def _load_chunk_text(base_dir: str, library_name: str, chunk_id: str) -> str:
    """根据 library + chunk_id 读取 chunk 的完整文本。"""
    try:
        reg = LibraryRegistry(base_dir)
        lib = reg.get_library(library_name)
        if lib is None:
            return ""
        mgr = lib.manager(base_dir)
        parts = chunk_id.split("/")
        if len(parts) != 2:
            return ""
        zone_id, chunk_name = parts
        zone = mgr.get_zone(zone_id)
        if zone is None:
            return ""
        chunk_path = os.path.join(zone.chunks_dir, f"{chunk_name}.json")
        if not os.path.isfile(chunk_path):
            return ""
        with open(chunk_path, "r", encoding="utf-8") as f:
            chunk = json.load(f)
        return chunk.get("text", "") or ""
    except Exception:
        return ""


def _truncate(text: str, max_chars: int) -> str:
    """截断文本到指定长度，超出部分用省略号标记。"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "……"


def _format_metadata_for_prompt(metadata: Optional[dict]) -> str:
    """生成给 LLM 看的紧凑元数据摘要。

    格式示例：· 时代:西汉 · 主题:传记 · 人物:刘邦/项羽/韩信
    - 人物只显示前3个，避免 prompt 膨胀
    - entity_density 不显示（数值对 LLM 无意义）
    - era_names 在古代时显示，现代时已置空（由 _extract_metadata 控制）
    - 无 metadata 时返回空字符串
    """
    if not metadata:
        return ""
    parts = []
    if metadata.get("era"):
        parts.append(f"时代:{metadata['era']}")
    if metadata.get("topic"):
        parts.append(f"主题:{metadata['topic']}")
    persons = metadata.get("top_persons") or []
    person_names = [p.get("name") for p in persons[:3] if p.get("name")]
    if person_names:
        parts.append(f"人物:{'/'.join(person_names)}")
    return " · ".join(parts)


def _build_context(
    base_dir: str,
    search_results: List[Dict[str, Any]],
    max_chunks: int,
    max_chars_per_chunk: int,
    max_total_chars: int,
) -> tuple[str, List[Dict[str, Any]]]:
    """把命中 chunk 拼成带编号的 context，返回 (context, references)。"""
    context_parts: List[str] = []
    references: List[Dict[str, Any]] = []
    total = 0

    for i, r in enumerate(search_results[:max_chunks], 1):
        lib_name = r.get("library", "")
        chunk_id = r.get("chunk_id", "")
        full_text = _load_chunk_text(base_dir, lib_name, chunk_id)
        snippet = r.get("snippet", "") or full_text[:120]
        # 优先用完整文本，截断到上限
        text = full_text if full_text else snippet
        text = _truncate(text, max_chars_per_chunk)

        remaining = max_total_chars - total
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = _truncate(text, remaining)

        # 注入元数据摘要，让 LLM 在生成回答时能利用朝代/主题/人物信息
        # 格式：[1] 来源：汉书.txt（二十四史）· 时代:西汉 · 主题:传记 · 人物:刘邦/项羽/韩信
        md_summary = _format_metadata_for_prompt(r.get("metadata"))
        source_line = f"[{i}] 来源：{r.get('source_file','')}（{lib_name}）"
        if md_summary:
            source_line += f" · {md_summary}"
        context_parts.append(f"{source_line}\n{text}")
        total += len(text)

        references.append({
            "index": i,
            "library": lib_name,
            "library_note": r.get("library_note", ""),
            "chunk_id": chunk_id,
            "source_file": r.get("source_file", ""),
            "source_file_path": r.get("source_file_path", ""),
            "source_sha256": r.get("source_sha256", ""),
            "heading": r.get("heading", ""),
            "hit_count": r.get("hit_count", 0),
            "matched_words": r.get("matched_words", []),
            "snippet": snippet,
            "metadata": r.get("metadata"),
        })

    context = "\n\n".join(context_parts)
    return context, references


def _get_registry_model(settings_store) -> tuple[str, str]:
    """从设置中读取模型与 base_url，model_id 与 display_name。"""
    model_id = settings_store.get("deepseek_model") if settings_store else V4_FLASH
    base_url = settings_store.get("deepseek_base_url") if settings_store else None
    if not model_id:
        model_id = V4_FLASH
    return model_id, base_url


def ai_search(
    question: str,
    registry: LibraryRegistry,
    client: DeepSeekClient,
    base_dir: str,
    library_names: Optional[List[str]] = None,
    parallel: int = 4,
    top_k: int = 20,
    max_chunks: Optional[int] = None,
    max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
    max_total_chars: Optional[int] = None,
    temperature: float = 0.3,
) -> Dict[str, Any]:
    """智能检索 + 一次性生成答案。"""
    # top_k 同时控制检索结果数与 context 引用条数，避免设置 20 却只取 8
    if max_chunks is None:
        max_chunks = top_k
    if max_total_chars is None:
        max_total_chars = _compute_total_chars_budget(top_k)
    # 1. 检索
    search_result = parallel_search(
        registry, question,
        library_names=library_names,
        parallel=parallel,
        base_dir=base_dir,
    )
    hits = search_result["results"][:top_k]

    # 2. 构造 context
    context, references = _build_context(
        base_dir, hits, max_chunks, max_chars_per_chunk, max_total_chars,
    )

    retrieval_summary = {
        "total_hits": search_result["total_hits"],
        "used_hits": len(references),
        "searched_libraries": search_result["searched_libraries"],
    }

    if not references:
        return {
            "question": question,
            "model": client.model,
            "answer": "未检索到相关资料，无法生成回答。",
            "retrieval": retrieval_summary,
            "references": [],
        }

    # 3. 调用模型
    answer = client.ask(
        question, context=context,
        model=client.model, temperature=temperature,
    )

    return {
        "question": question,
        "model": client.model,
        "answer": answer,
        "retrieval": retrieval_summary,
        "references": references,
    }


def ai_search_stream(
    question: str,
    registry: LibraryRegistry,
    client: DeepSeekClient,
    base_dir: str,
    library_names: Optional[List[str]] = None,
    parallel: int = 4,
    top_k: int = 20,
    max_chunks: Optional[int] = None,
    max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
    max_total_chars: Optional[int] = None,
    temperature: float = 0.3,
) -> Iterator[Dict[str, Any]]:
    """智能检索 + 流式生成答案。

    事件序列：
        {"phase":"retrieval","retrieval":{...},"references":[...]}
        {"phase":"reasoning","delta":"..."}   # 思考模式才有
        {"phase":"content","delta":"..."}
        {"phase":"done","usage":{...}}
        或
        {"phase":"error","error":"...","stage":"retrieval|generation"}
    """
    # top_k 同时控制检索结果数与 context 引用条数
    if max_chunks is None:
        max_chunks = top_k
    if max_total_chars is None:
        max_total_chars = _compute_total_chars_budget(top_k)
    try:
        # 1. 检索
        search_result = parallel_search(
            registry, question,
            library_names=library_names,
            parallel=parallel,
            base_dir=base_dir,
        )
        hits = search_result["results"][:top_k]
        # 关键词检索完成后，注入向量召回（解决同义表述、语义分散等召回盲区）
        vr = _inject_vector_recall(
            question, registry, base_dir, library_names,
            hits, top_k=top_k,
        )
        yield {
            "phase": "vector_recall",
            "available": vr["available"],
            "new_hits": vr["new_hits"],
            "total_hits": vr["total_hits"],
            "reason": vr["reason"],
        }

        # 2. 构造 context
        context, references = _build_context(
            base_dir, hits, max_chunks, max_chars_per_chunk, max_total_chars,
        )

        retrieval_summary = {
            "total_hits": search_result["total_hits"],
            "used_hits": len(references),
            "searched_libraries": search_result["searched_libraries"],
        }

        # 先把检索结果发出去（前端可以先渲染引用）
        yield {
            "phase": "retrieval",
            "retrieval": retrieval_summary,
            "references": references,
        }

        if not references:
            yield {"phase": "content", "delta": "未检索到相关资料，无法生成回答。"}
            yield {"phase": "done", "usage": None}
            return

        # 3. 流式调用模型
        for event in client.ask_stream(
            question, context=context,
            model=client.model, temperature=temperature,
        ):
            etype = event.get("type")
            if etype == "reasoning":
                yield {"phase": "reasoning", "delta": event.get("delta", "")}
            elif etype == "content":
                yield {"phase": "content", "delta": event.get("delta", "")}
            elif etype in ("finish", "done"):
                yield {"phase": "done", "usage": event.get("usage")}

    except DeepSeekError as e:
        yield {"phase": "error", "stage": "generation",
               "error": f"DeepSeek 调用失败: {e}"}
    except Exception as e:
        yield {"phase": "error", "stage": "retrieval",
               "error": f"检索失败: {e}"}


# ============================================================
#  Agent 模式：让 LLM 主导检索（规划 → 检索 → 评估 → 重试 → 生成）
# ============================================================
#
# 解决的问题：
#   原直接把整句"介绍郎溪县的经济发展情况"丢进 tokenize() 后被拆成
#   单字 token，"郎""溪"在档案库中频率极高导致命中数压倒"经济发展"
#   连续匹配的 phrase_bonus，结果排在前面的都是"郎溪"密集条目而非
#   真正讲经济的章节。
#
# Agent 流程：
#   1. 规划：LLM 分析问题 → 输出结构化查询词列表（保留词组、扩展同义领域）
#   2. 检索：对每个查询词独立调用 parallel_search，合并去重，按
#      phrase_bonus*2 + hit_count + 多查询命中加分 重新排序
#   3. 评估：LLM 判断资料是否足够；若不足，生成补充查询词再试一轮（最多 2 轮）
#   4. 生成：用最终 context 流式生成答案
# ============================================================

# 多查询命中加成：被 N 个查询词命中的 chunk 加 (N-1)*MULTI_QUERY_BONUS
MULTI_QUERY_BONUS = 80
# phrase_bonus 在 agent 排序中的放大系数（让连续匹配权重远高于分散命中）
PHRASE_WEIGHT_MULTIPLIER = 3
# 最大重试轮数（含首轮）
DEFAULT_MAX_ROUNDS = 2
# 单轮最多查询词
DEFAULT_MAX_QUERIES = 5

# ===== 向量召回（Faiss + bge-small-zh）加成系数 =====
# 双通道命中（关键词 + 向量同时命中）：大幅加成，说明该 chunk 与问题强相关
SEMANTIC_BOTH_BONUS = 500
# 仅向量召回命中（无关键词命中）：较小加成，避免噪声 chunk 挤占关键词命中
SEMANTIC_ONLY_BONUS = 60
# 语义相似度（0~1）转换为排序分的系数
SEMANTIC_SCORE_WEIGHT = 200


_PLANNER_SYSTEM = """你是一名档案检索规划师。用户提出一个自然语言问题，你需要把它分解为多个高效的检索查询词。

【极重要·第一步：理解问题】
在规划查询词之前，必须先理解问题的真实意图：
1. 识别问题中的核心实体和概念（人物、地点、事件、时代、主题等）
2. 识别问题中的限定条件（时间、地点、文体、领域等）
3. 识别问题中可能存在的错别字或非标准表述，理解其真实含义
4. 若问题含代词（他/她/它），需结合历史上下文确定指代对象

【极重要·查询词切分规则】
查询词必须保持语义完整性，绝对不能把一个完整概念拆开：
- "七世纪"是一个完整的时间概念，不能拆成"七世"+"纪"，更不能只取"七世"
- "碳基生物"是一个完整的科学概念，不能拆成"碳基"+"生物"
- "经济发展"是一个完整的词组，不能拆成"经济"+"发展"
- "最强碳基生物"作为完整短语优先保留，不要拆分
正确做法：把完整概念作为一个查询词，如"七世纪""最强碳基生物""碳基生命"

背景：底层是按字符切分的倒排索引，对完整词组（如"经济发展"）通过 bigram 连续匹配给高分。
因此查询词应当：
1. 抽取问题中的【核心概念词组】保持完整（如"经济发展""农业区划""工商"），不要拆成单字
2. 地名、人名等专有名词单独作为一个查询词
3. 围绕核心概念扩展同义/相关领域词（例如问"经济发展"可补充"工商""财政""税收""计划管理"等可能涉及的章节）
4. 排除"介绍""情况""如何""怎样"等无信息量词

【极重要】朝代/历史人物检索经验：
当问题涉及历史人物、朝代、史书文献时（如"汉武帝""孝武皇帝""三国志"等），极易出现跨朝代同名干扰：
- "武帝""孝武皇帝"等谥号/庙号在多个朝代重复出现（汉武帝、晋武帝、宋孝武帝、北魏太武帝等）
- "本纪""列传"等篇目名在所有正史中都存在
- "脱脱"在北魏/辽/金/元等多个朝代都有同名人物，必须用 title_filter 限定到《元史》
- "李密"在三国（蜀汉）和隋末都有，必须根据问题上下文限定朝代
- 若不限定朝代/文献，检索结果会被其他朝代的内容严重污染
因此规划时必须：
  a. 在 queries 中加入朝代/文献限定词（如"汉书""史记""后汉书""三国志"等）
  b. 在 title_filter 中填入朝代/书名/篇目名，用于在 chunk 标题(heading)中做精确过滤
  c. title_filter 的 contains 是 OR 关系——例如查汉武帝，填 ["汉书","武帝纪"] 表示保留标题含"汉书"或"武帝纪"的 chunk
  d. 若问题明确指向某部史书的某篇章，title_filter 应同时包含书名和篇目名
  e. 【必须】当问题涉及历史人物时，必须根据你对该人物所属朝代的知识，在 title_filter 中填入对应史书名。
     即使问题文本中没有出现朝代名，只要你识别出人物属于某朝代，就必须填 title_filter。
     例：问"脱脱"→识别为元朝宰相→title_filter: {"contains":["元史"],"excludes":["晋","魏","南","北"]}
     例：问"于谦"→识别为明朝→title_filter: {"contains":["明史"],"excludes":["元","宋","金","清"]}
     例：问"李牧"→识别为战国赵国→史记有廉颇蔺相如列传含李牧→title_filter: {"contains":["史记"],"excludes":["汉书","后汉","三国","晋","宋","魏","隋","唐","宋史","辽","金","元","明"]}

title_filter 字段说明：
  - contains: 标题包含任一关键词的 chunk 保留（OR 关系）
  - contains_all: 标题必须同时包含所有关键词的 chunk 保留（AND 关系）
  - excludes: 标题包含这些关键词的 chunk 排除
  - 若问题不涉及历史/朝代/文献，title_filter 留空 null

输出严格的 JSON（不要 markdown 代码块、不要注释）：
{
  "queries": ["经济发展", "郎溪", "工商", "财政", "计划管理"],
  "title_filter": null,
  "exclude": ["介绍", "情况"],
  "rationale": "经济发展是核心词组需整体检索；郎溪作为地名限定；扩展工商/财政等同义领域以覆盖不同章节"
}

历史人物检索示例：
问题："汉武帝的武功成就有哪些？"
{
  "queries": ["武帝纪", "匈奴", "西域", "河西四郡", "汉书"],
  "title_filter": {"contains": ["汉书", "武帝纪"], "excludes": ["晋", "宋", "北魏", "南"]},
  "exclude": [],
  "rationale": "汉武帝本纪在《汉书》中；用 title_filter 锁定汉书/武帝纪，排除晋宋等同名干扰；武功涉及匈奴、西域、河西四郡等"
}

queries 最多 5 个，按重要性从高到低排序。"""


# 问题预检：识别奇怪输入（错别字、歧义、信息不足），避免浪费算力
_QUESTION_CHECK_SYSTEM = """你是一名问题理解助手。判断用户问题是否存在以下问题：

1. **疑似错别字**：如"原理是玄宗的称号"中的"原理"可能是"原来"的笔误
2. **歧义/指代不清**：含代词但缺少上下文，无法确定指代对象（如"他是谁"且无上下文）
3. **信息不足**：缺少关键信息，无法有效检索（如"那个事怎么样了"）
4. **无意义输入**：纯符号、乱码、测试性内容

判断标准：
- 能从问题中提取出可检索的实体/概念（人名、地名、事件、概念等）→ ok=true
- 问题完全无法理解或明显有错别字影响意图 → ok=false，给出澄清建议

【极重要】结合上下文判断指代消解：
1. 历史问答：若问题含代词（他/她/它/此），但历史问答中已明确指代对象，则 ok=true
2. 资料库范围：若用户已选定资料库（会在【资料库范围】中列出数量和名称），问题中的指代词按以下规则消解：
   - "这本书"/"该书" → 当仅选中1个库时，明确指代该库，ok=true
   - "两本书"/"这两本书"/"这两本" → 当选中2个库时，明确指代这两个库，ok=true
   - "这些书"/"这几本书"/"各书" → 当选中N个库时，明确指代所有选中的库，ok=true
   - 仅当选中的库数量与指代词的数量不匹配（如选中1个库却问"两本书"）时才 ok=false
3. 仅当无法从问题、历史和资料库范围中推断意图时才 ok=false

输出严格 JSON（不要 markdown 代码块）：
{
  "ok": true或false,
  "reason": "判断理由（简短）",
  "clarify": "当 ok=false 时，向用户的澄清建议（友好、具体，指出可能的笔误或缺失信息）"
}"""


def _check_question(
    client: DeepSeekClient,
    question: str,
    history: Optional[List[Dict[str, Any]]] = None,
    library_context: str = "",
) -> Dict[str, Any]:
    """预检用户问题，识别错别字/歧义/信息不足。

    返回 {"ok": bool, "reason": str, "clarify": str}。
    失败时返回 {"ok": true} 不阻断流程。

    library_context 用于注入资料库范围信息，帮助判断"这本书"等指代是否可消解。
    """
    try:
        history_hint = ""
        if history:
            history_text = _build_history_text(history)
            if history_text:
                history_hint = f"\n\n{history_text}"
        lib_hint = ""
        if library_context:
            lib_hint = f"\n\n【资料库范围】{library_context}"
        messages = [
            {"role": "system", "content": _QUESTION_CHECK_SYSTEM},
            {"role": "user", "content": f"用户问题：{question}{history_hint}{lib_hint}"},
        ]
        resp = client.chat(messages, temperature=0.1, max_tokens=300)
        result = _extract_json(resp.get("content", "")) or {}
        return {
            "ok": bool(result.get("ok", True)),
            "reason": result.get("reason", ""),
            "clarify": result.get("clarify", ""),
        }
    except Exception:
        # 预检失败不阻断流程
        return {"ok": True, "reason": "预检失败", "clarify": ""}

_EVALUATOR_SYSTEM = """你是一名检索结果评估师。基于用户问题和当前已检索到的资料片段，判断这些资料是否足够回答问题。

判断标准：
- sufficient=true: 资料直接包含回答问题所需的信息（不只是话题相关）
- sufficient=false: 资料偏题、信息稀薄、或缺少问题中关键维度

若不足，请给出 1~3 个补充查询词（next_queries），用于下一轮检索。这些词应当：
- 避开已经用过的查询方向
- 针对资料缺失的具体维度
- 仍是完整词组而非单字

输出严格的 JSON（不要 markdown 代码块）：
{
  "sufficient": false,
  "missing_aspect": "缺少改革开放后工业发展的具体数据",
  "next_queries": ["工业", "改革开放", "乡镇企业"],
  "confidence": 0.4
}"""


def _extract_json(content: str) -> Optional[Dict[str, Any]]:
    """从可能含 markdown 围栏或前后噪声的文本中提取首个 JSON 对象。

    若 JSON 被截断（max_tokens 不足导致输出未完成），尝试自动补全后解析。
    """
    if not content:
        return None
    s = content.strip()
    # 去除 markdown 代码围栏
    if s.startswith("```"):
        # 去掉首行 ```xxx
        s = s.split("\n", 1)[1] if "\n" in s else ""
        # 去掉末尾 ```
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]
        s = s.strip()
    # 直接定位首个 { 到末尾 }
    start = s.find("{")
    if start < 0:
        return None
    # 用栈匹配最外层 }
    depth = 0
    in_str = False
    esc = False
    last_complete_end = -1
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    last_complete_end = i
    # 优先用完整匹配
    if last_complete_end > 0:
        try:
            return json.loads(s[start:last_complete_end + 1])
        except json.JSONDecodeError:
            pass
    # JSON 被截断：尝试自动补全（depth > 0 表示 { 没闭合，或字符串没闭合）
    truncated = s[start:]
    # 若字符串未闭合，先补 "
    if in_str:
        truncated += '"'
    # 补足缺少的 }
    # 重新计算深度
    depth2 = 0
    in_str2 = False
    esc2 = False
    for c in truncated:
        if in_str2:
            if esc2:
                esc2 = False
            elif c == "\\":
                esc2 = True
            elif c == '"':
                in_str2 = False
        else:
            if c == '"':
                in_str2 = True
            elif c == "{":
                depth2 += 1
            elif c == "}":
                depth2 -= 1
    if depth2 > 0:
        # 可能最后一个字段后面多了个逗号或冒号，先去掉
        truncated = truncated.rstrip()
        if truncated.endswith(','):
            truncated = truncated[:-1]
        elif truncated.endswith(':'):
            # 字段名后没值，补 null
            truncated += ' null'
        elif truncated.endswith('"' ):
            # 字段值后没逗号，补 ,
            pass
        truncated += '}' * depth2
        try:
            return json.loads(truncated)
        except json.JSONDecodeError:
            # 再尝试：去掉最后一个不完整的字段
            # 找最后一个逗号
            last_comma = truncated.rfind(',')
            if last_comma > 0:
                candidate = truncated[:last_comma] + '}' * depth2
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
    return None


def _build_history_text(history: Optional[List[Dict[str, Any]]]) -> str:
    """把历史对话构造为纯文本上下文（仅问题和最终回答，不含 reasoning/工具调用细节）。

    history 元素结构：{role: "user"|"assistant", content: str, references?: list}
    """
    if not history:
        return ""
    parts: List[str] = []
    for m in history:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            parts.append(f"用户：{content}")
        elif role == "assistant":
            parts.append(f"助手：{content}")
    return "\n\n".join(parts)


def _build_library_context(registry: LibraryRegistry, library_names: Optional[List[str]]) -> str:
    """构造库背景信息字符串，注入提示词避免大模型重复搜索主体地名。

    按文件夹分组输出，格式：
        [县志类]《郎溪县志》、《XX县志》；[政策类]《XX政策》；《无文件夹的库》
    无 folder 字段的库直接列在末尾（不加方括号前缀）。
    """
    if registry is None:
        return ""
    try:
        all_libs = registry.list_libraries()
        if library_names:
            libs = [l for l in all_libs if l.name in library_names]
        else:
            libs = all_libs
        if not libs:
            return ""

        def _lib_desc(lib):
            desc = f"《{lib.name}》"
            if lib.note:
                desc += f"（{lib.note}）"
            return desc

        # 按 folder 分组（空字符串 = 根级）
        by_folder: Dict[str, List[str]] = {}
        root_descs: List[str] = []
        for lib in libs:
            f = (lib.folder or "").strip()
            if f:
                by_folder.setdefault(f, []).append(_lib_desc(lib))
            else:
                root_descs.append(_lib_desc(lib))

        groups: List[str] = []
        # 文件夹分组按名字排序，保证提示词稳定
        for f in sorted(by_folder.keys()):
            groups.append(f"[{f}]" + "、".join(by_folder[f]))
        if root_descs:
            groups.append("、".join(root_descs))
        # 在开头标注库数量，帮助预检模型判断"两本书"等指代是否可消解
        count_hint = f"共 {len(libs)} 个库：" if len(libs) > 1 else ""
        return count_hint + "；".join(groups)
    except Exception:
        return ""


# 规划回退用的停用字（bigram 首尾字命中这些字时丢弃，避免无信息量查询词）
_PLAN_STOP_CHARS = set("介绍情况如何怎样哪些请问一下的了是在有和与及或这那")


# ============================================================
#  语义检索（同义词组 + 跨组共现，可叠加到三种问答模式）
# ============================================================

# 语义检索规划器：把问题分解为 must/should 同义词组
_SEMANTIC_PLANNER_SYSTEM = """你是一个语义检索规划器。基于用户问题，生成 1-3 组同义词组，用于关联检索。

【输出格式】严格 JSON：
{
  "groups": [
    {"words": ["刘备","先主","先帝","昭烈帝","玄德"], "required": true, "label": "问题主体"},
    {"words": ["崩","殂","薨","卒","殁","死","病逝"], "required": false, "label": "事件概念"},
    {"words": ["永安","白帝","章武"], "required": false, "label": "相关时地"}
  ],
  "rationale": "...",
  "should_continue": true
}

【组的设计原则】
1. 每组是同义/等价词集合：人物别名、字号、庙号、谥号、官职、相关地名、同义动词
2. 必须区分 required（必需组）与 should（加分组）：
   - required 组：问题主体（人物全名、专有地名、特定事件名），chunk 必须命中至少一个 required 组才算相关。问题主体词必须放在 required 组。
   - should 组：同义动词、时间、地点、相关概念等，跨组共现越多加分越高
3. required 组的词必须是高辨识度专有名词（人名/地名/事件名），避免使用历代通用的泛称：
   - ✅ 推荐：人物全名（刘备、张飞）、专属字号（玄德、翼德）、专属庙号/谥号（昭烈帝、汉烈祖）
   - ⚠️ 谨慎：泛称（先主、先帝、太祖）在多朝代史书中都指前代君主，会引入跨朝代噪声，仅当无更优词时使用
   - ❌ 禁止：单字通用词（如"死""卒""崩"）放入 required 组
4. should 组的同义词优先选古籍/正史中的原始表述（如"崩/殂/薨/卒"比"驾崩"更常见）
5. 1-3 组为宜，required 组通常 1 个，should 组 1-2 个
6. 组内词不要重复，不要包含空格或标点，单组同义词不超过 8 个
7. 不要把问题原文整句作为某一组；要拆解为概念单元

【should_continue 字段】
- true：还需要继续下一轮检索（当前规划可能不够充分）
- false：当前规划已充分覆盖问题，无需下一轮
- 【强制】首轮必须为 true（首轮规划尚未验证效果，无法判断是否充分）
- 第 2 轮起，若 required 组命中多且 should 组共现充分，可设为 false

【问题类型示例】
- "刘备去世的时间" → groups=[
    {"words":["刘备","先主","先帝","昭烈帝","玄德"],"required":true,"label":"问题主体"},
    {"words":["崩","殂","薨","卒","殁","病逝"],"required":false,"label":"事件概念"}
  ]
- "张飞字什么" → groups=[
    {"words":["张飞","翼德","益德"],"required":true,"label":"问题主体"},
    {"words":["字"],"required":false,"label":"询问概念"}
  ]
- "永安托孤" → groups=[
    {"words":["刘备","先主"],"required":true,"label":"问题主体"},
    {"words":["永安","白帝"],"required":false,"label":"地点"},
    {"words":["托孤","托"],"required":false,"label":"事件"}
  ]
- "郎溪经济发展" → groups=[
    {"words":["郎溪"],"required":true,"label":"问题主体"},
    {"words":["经济","发展","产业"],"required":false,"label":"概念"}
  ]

只输出 JSON，不要解释。"""

# 语义检索重规划器：基于上一轮命中情况调整
_SEMANTIC_REPLANNER_SYSTEM = """你是一个语义检索规划器。基于用户问题和上一轮检索反馈，调整生成新的同义词组。

【上一轮反馈】
- 已使用的查询组（含 must/should 标记）
- 命中数量、共现情况、required 组命中率
- 评估上一轮覆盖问题的程度

【调整原则】
1. 每轮关键词必须与上轮不同（更换同义词、调整组结构、增减概念维度），禁止重复输出相同 groups
2. 若上一轮 required 组命中少 → 扩展人物/地名的更多别名（如新增字号、官职、谥号变体）
3. 若上一轮 should 组命中少 → 替换为更古风/更专业的同义词（如"死"→"崩/殂/薨/卒/殒/殁"）
4. 若 required 组命中多但 should 组共现少 → 调整 should 组为更具体的概念（如把"死"换为"病重""临终""遗诏"等更具上下文的词）
5. 若上一轮已充分命中（required 组高命中 + should 组高共现）→ should_continue 设为 false，结束检索
6. 不要超过 3 组，组内同义词不超过 8 个
7. required 组的词必须是专有名词，禁止单字通用词（如"死""卒"）放入 required 组

【should_continue 字段】
- true：当前规划不充分，需要继续下一轮
- false：当前轮已找到充分信息，可结束检索（前端将基于已有命中生成答案）

【输出格式】严格 JSON：
{
  "groups": [
    {"words": [...], "required": true/false, "label": "..."},
    ...
  ],
  "rationale": "...",
  "coverage_assessment": "上一轮覆盖度评估",
  "should_continue": true/false
}

只输出 JSON。"""


def _plan_semantic_groups(
    client: DeepSeekClient,
    question: str,
    library_context: str = "",
    prev_rounds: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.3,
    max_tokens: int = 1200,
) -> Dict[str, Any]:
    """规划同义词组（含 must/should 标记）。首轮用 _SEMANTIC_PLANNER_SYSTEM，重规划用 _SEMANTIC_REPLANNER_SYSTEM。

    Args:
        prev_rounds: 上一轮规划与命中反馈列表，每项形如
            {"groups": [{"words":[...],"required":bool,"label":str}],
             "total_hits": N, "cooccur_max": M, "required_hit_rate": float, "rationale": "..."}
        若为空或 None 视为首轮。

    Returns:
        {
            "groups": [{"words":[...],"required":bool,"label":str}, ...],
            "rationale": "...",
            "round": int,
            "should_continue": bool,  # LLM 主动建议是否继续下一轮
        }
    """
    lib_hint = ""
    if library_context:
        lib_hint = f"\n\n【数据来源】{library_context}\n资料主体已限定，规划时不要把主体本身拆出来单独成组。"

    is_replan = bool(prev_rounds)
    if is_replan:
        # 构造上一轮反馈摘要（含 must/should 标记和 required 组命中率）
        feedback_parts = []
        for i, r in enumerate(prev_rounds, 1):
            parts = []
            for g in r.get("groups", []):
                if isinstance(g, dict):
                    tag = "required" if g.get("required") else "should"
                    label = g.get("label", "")
                    words = g.get("words", [])
                    parts.append(f"[{tag}|{label}] {'|'.join(words)}")
                else:
                    parts.append(f"[should] {'|'.join(g) if isinstance(g, list) else str(g)}")
            groups_str = " ∩ ".join(parts)
            req_rate = r.get("required_hit_rate", 0.0)
            feedback_parts.append(
                f"第{i}轮 groups=[{groups_str}]  命中 {r.get('total_hits', 0)} 条，"
                f"最高共现 {r.get('cooccur_max', 0)} 组，required 组命中率 {req_rate:.0%}，"
                f"评估：{r.get('rationale', '')}"
            )
        feedback = "\n".join(feedback_parts)
        messages = [
            {"role": "system", "content": _SEMANTIC_REPLANNER_SYSTEM},
            {"role": "user", "content":
                f"用户问题：{question}{lib_hint}\n\n【上一轮反馈】\n{feedback}\n\n请调整生成新的 groups（必须与上轮不同）。"},
        ]
    else:
        messages = [
            {"role": "system", "content": _SEMANTIC_PLANNER_SYSTEM},
            {"role": "user", "content": f"用户问题：{question}{lib_hint}\n\n请生成同义词组。"},
        ]

    try:
        resp = client.chat(messages, temperature=temperature, max_tokens=max_tokens)
        plan = _extract_json(resp.get("content", "")) or {}
    except Exception as e:
        import sys as _sys
        print(f"[warn] _plan_semantic_groups API 异常: {e}", file=_sys.stderr)
        plan = {}

    # 解析新格式 groups：[{words, required, label}, ...]；兼容旧格式 [[...], ...]
    raw_groups = plan.get("groups") or []
    cleaned_groups: List[Dict[str, Any]] = []
    seen_words: set = set()
    for g in raw_groups:
        if isinstance(g, dict):
            words = g.get("words") or []
            required = bool(g.get("required", False))
            label = g.get("label", "") or ""
        elif isinstance(g, list):
            words = g
            required = False
            label = ""
        else:
            continue

        gs: List[str] = []
        for w in words:
            if isinstance(w, str):
                w = w.strip()
                if w and w not in seen_words and " " not in w and not any(c in w for c in "，。；：、"):
                    seen_words.add(w)
                    gs.append(w)
        if gs:
            cleaned_groups.append({"words": gs, "required": required, "label": label})
        if len(cleaned_groups) >= 3:
            break

    # 强制规则：若所有组都是 should（无 required），把第一组提升为 required（问题主体组）
    # —— 用户问"刘备死"，必需命中"刘备"组才算相关，否则跨朝代混入
    if cleaned_groups and not any(g["required"] for g in cleaned_groups):
        cleaned_groups[0]["required"] = True
        if not cleaned_groups[0]["label"]:
            cleaned_groups[0]["label"] = "问题主体"

    # 失败回退：用 bigram 拆问题，第一个 bigram 作为 required 组
    if not cleaned_groups:
        fallback = _fallback_queries_from_question(question, library_context)
        if fallback:
            cleaned_groups = [
                {"words": [fallback[0]], "required": True, "label": "问题主体"},
            ]
            for w in fallback[1:3]:
                cleaned_groups.append({"words": [w], "required": False, "label": ""})

    # 解析 should_continue（默认 true，让多轮机制有机会运行）
    should_continue_raw = plan.get("should_continue")
    if isinstance(should_continue_raw, bool):
        should_continue = should_continue_raw
    elif isinstance(should_continue_raw, str):
        should_continue = should_continue_raw.strip().lower() in ("true", "1", "yes", "是")
    else:
        should_continue = True  # 默认继续

    return {
        "groups": cleaned_groups,
        "rationale": plan.get("rationale", ""),
        "round": (len(prev_rounds) + 1) if prev_rounds else 1,
        "should_continue": should_continue,
    }


def _semantic_retrieval_rounds(
    question: str,
    registry: LibraryRegistry,
    client: DeepSeekClient,
    base_dir: str,
    library_names: Optional[List[str]] = None,
    parallel: int = 4,
    top_k: int = 20,
    temperature: float = 0.3,
    max_rounds: int = 3,
    library_context: str = "",
) -> Iterator[Dict[str, Any]]:
    """语义检索：多轮（默认 3 轮）同义词组关联检索。

    每轮让 LLM 规划 1-3 组同义词组（含 must/should 标记），
    调用 search_related_keywords 检索（必需组强制过滤），合并去重后评估是否充分。
    LLM 可通过 should_continue=false 主动结束；多轮关键词必须不同。

    事件序列：
        {"phase":"semantic_plan","round":R,
         "groups":[{"words":[...],"required":bool,"label":str},...],"rationale":"..."}
        {"phase":"semantic_searching","round":R,"groups":[...],
         "total_hits":N,"new_hits":M,"cooccur_max":K,"filtered_by_required":F,
         "required_hit_rate":float}
        {"phase":"semantic_round_done","round":R,"total_unique":N,"cooccur_max":K,
         "should_continue":bool,"reason":"..."}
        {"phase":"semantic_done","total_unique":N,"rounds":R,"all_groups":[...]}
    最后返回合并去重后的 hits 列表（与 parallel_search 输出兼容）。
    """
    all_hits: List[Dict[str, Any]] = []
    used_chunk_ids = set()
    used_words_per_round: List[set] = []  # 每轮已用关键词集合，强制每轮不同
    all_groups_history: List[Dict[str, Any]] = []

    for round_idx in range(max_rounds):
        round_num = round_idx + 1
        # 规划
        plan = _plan_semantic_groups(
            client, question, library_context=library_context,
            prev_rounds=all_groups_history if all_groups_history else None,
            temperature=temperature,
        )
        groups = plan.get("groups", [])
        rationale = plan.get("rationale", "")
        should_continue = plan.get("should_continue", True)

        # 提取本轮所有关键词用于去重检查
        round_words = set()
        for g in groups:
            round_words.update(g.get("words", []))

        # 跳过：本轮所有关键词与已有某轮完全相同
        if round_words and round_words in used_words_per_round:
            yield {"phase": "semantic_round_done", "round": round_num,
                   "total_unique": len(all_hits), "cooccur_max": 0,
                   "skipped": True, "reason": "关键词与上一轮完全相同，结束检索"}
            break
        used_words_per_round.append(round_words)

        yield {
            "phase": "semantic_plan",
            "round": round_num,
            "groups": groups,
            "rationale": rationale,
            "should_continue": should_continue,
        }

        # 执行本轮检索（search_related_keywords 内部已处理 required 过滤）
        round_total_hits = 0
        round_cooccur_max = 0
        round_new_hits = 0
        round_filtered = 0
        # 单次 search_related_keywords 调用即可处理多组（组内 OR + 组间共现加分）
        try:
            sr = search_related_keywords(
                registry, groups,
                library_names=library_names,
                parallel=parallel,
                base_dir=base_dir,
                top_k=top_k * 3,  # 多取一些用于合并去重
            )
            round_filtered = sr.get("filtered_by_required", 0)
        except Exception as e:
            yield {"phase": "semantic_searching", "round": round_num,
                   "groups": groups,
                   "error": f"语义检索失败: {e}"}
            continue

        # 合并 hits
        for h in sr.get("results", []):
            cid = h.get("chunk_id")
            if cid and cid not in used_chunk_ids:
                used_chunk_ids.add(cid)
                all_hits.append(h)
                round_new_hits += 1
            elif cid:
                # 已有：更新 hit_count 与 matched_words
                for existing in all_hits:
                    if existing.get("chunk_id") == cid:
                        existing["hit_count"] = max(existing.get("hit_count", 0),
                                                    h.get("hit_count", 0))
                        mw = set(existing.get("matched_words", [])) | set(h.get("matched_words", []))
                        existing["matched_words"] = sorted(mw)
                        # 取较大的 related_score 和 should_cooccur_groups
                        if h.get("related_score", 0) > existing.get("related_score", 0):
                            existing["related_score"] = h["related_score"]
                            existing["cooccur_groups"] = h.get("cooccur_groups", 0)
                            existing["should_cooccur_groups"] = h.get("should_cooccur_groups", 0)
                        break

        round_total_hits = sr.get("total_hits", 0)
        round_cooccur_max = max((h.get("should_cooccur_groups", 0) for h in sr.get("results", [])),
                                default=0)

        # 计算 required 组命中率（用于反馈给 LLM）
        required_indices = sr.get("required_indices", [])
        required_hit_rate = 1.0
        if required_indices and sr.get("results"):
            # results 已经过滤掉未命中 required 组的 chunk，所以这里反映的是过滤后命中占比
            # 真正的 required 组命中率应基于过滤前后比
            total_candidates = round_total_hits + round_filtered
            required_hit_rate = (round_total_hits / total_candidates) if total_candidates > 0 else 0.0

        # 记录本轮历史（供下一轮重规划参考）
        all_groups_history.append({
            "groups": groups,
            "total_hits": round_total_hits,
            "cooccur_max": round_cooccur_max,
            "required_hit_rate": required_hit_rate,
            "rationale": rationale,
        })

        yield {
            "phase": "semantic_searching",
            "round": round_num,
            "groups": groups,
            "total_hits": round_total_hits,
            "new_hits": round_new_hits,
            "cooccur_max": round_cooccur_max,
            "filtered_by_required": round_filtered,
            "required_hit_rate": required_hit_rate,
        }

        # 提前终止判定（收紧版，避免过早结束）：
        # 1. LLM 主动结束（should_continue=false）+ 已有命中 + 非首轮
        #    【首轮强制不结束】—— LLM 在尚未看到检索效果的情况下判定"充分"不可信
        if not should_continue and len(all_hits) > 0 and round_num >= 2:
            yield {
                "phase": "semantic_round_done", "round": round_num,
                "total_unique": len(all_hits), "cooccur_max": round_cooccur_max,
                "should_continue": False, "reason": "LLM 判定已充分覆盖，主动结束",
            }
            break

        # 2. should 共现 >= 2 且总命中 >= top_k（充分覆盖）
        if round_cooccur_max >= 2 and len(all_hits) >= top_k:
            yield {
                "phase": "semantic_round_done", "round": round_num,
                "total_unique": len(all_hits), "cooccur_max": round_cooccur_max,
                "should_continue": should_continue, "reason": "should 共现充分且命中数达标",
            }
            break

        # 3. 连续 2 轮 0 新增命中
        if round_new_hits == 0 and round_num >= 2:
            yield {
                "phase": "semantic_round_done", "round": round_num,
                "total_unique": len(all_hits), "cooccur_max": round_cooccur_max,
                "should_continue": should_continue, "reason": "连续 0 新增命中，结束检索",
            }
            break

        # 非最后一轮：发送 round_done 事件
        if round_num < max_rounds:
            yield {
                "phase": "semantic_round_done", "round": round_num,
                "total_unique": len(all_hits), "cooccur_max": round_cooccur_max,
                "should_continue": should_continue, "reason": "继续下一轮",
            }
        else:
            yield {
                "phase": "semantic_round_done", "round": round_num,
                "total_unique": len(all_hits), "cooccur_max": round_cooccur_max,
                "should_continue": False, "reason": "已达最大轮数",
            }

    # 按 should_cooccur_groups 优先 + related_score 降序排序
    all_hits.sort(key=lambda x: (
        x.get("should_cooccur_groups", 0),
        x.get("related_score", x.get("score", 0)),
        x.get("cooccur_groups", 0),
        x.get("hit_count", 0),
    ), reverse=True)

    yield {
        "phase": "semantic_done",
        "total_unique": len(all_hits),
        "rounds": len(all_groups_history),
        "all_groups": [r["groups"] for r in all_groups_history],
    }

    # 内部事件：把 hits 传给调用方（不发给前端）
    yield {
        "phase": "_semantic_hits",
        "hits": all_hits,
    }


def _fallback_queries_from_question(question: str, library_context: str = "") -> List[str]:
    """规划失败时的回退：从问题中提取有意义的查询词。

    改进版：
    1. 优先保留完整中文短语（2-6字），避免把"七世纪"拆成"七世"+"纪"
    2. 仅对超过6字的长片段做 bigram 拆分
    3. 过滤停用字和库背景中的地名用字

    返回查询词列表（最多5个）；若问题中无任何有效短语，回退到原问题。
    """
    import re
    # 复用搜索引擎的连续中文片段提取规则
    from searcher import _HAN_PHRASE_RE

    # 从库背景中收集地名用字（如《郎溪县志》→ 郎/溪/县/志），回退时避免再搜地名
    geo_chars: set = set()
    if library_context:
        for p in _HAN_PHRASE_RE.findall(library_context):
            geo_chars.update(p)

    queries: List[str] = []
    seen: set = set()
    for phrase in _HAN_PHRASE_RE.findall(question):
        if len(phrase) < 2:
            continue
        # 2-6字短语：直接保留完整短语，不拆分（避免"七世纪"被拆成"七世"）
        if len(phrase) <= 6:
            if (phrase[0] not in _PLAN_STOP_CHARS and phrase[-1] not in _PLAN_STOP_CHARS
                    and phrase not in seen):
                # 首尾字都是地名用字 → 丢弃（避免搜"郎溪""溪县"等高频地名）
                if phrase[0] in geo_chars and phrase[-1] in geo_chars and len(phrase) <= 3:
                    continue
                queries.append(phrase)
                seen.add(phrase)
            continue
        # 超过6字的长片段：先保留完整短语，再取 bigram 补充
        if phrase not in seen and phrase[0] not in _PLAN_STOP_CHARS:
            queries.append(phrase)
            seen.add(phrase)
        for i in range(len(phrase) - 1):
            bg = phrase[i:i + 2]
            if bg in seen:
                continue
            # 首尾字任一是停用字 → 丢弃
            if bg[0] in _PLAN_STOP_CHARS or bg[1] in _PLAN_STOP_CHARS:
                continue
            # 首尾字都是地名用字 → 丢弃（避免搜"郎溪""溪县"等高频地名）
            if bg[0] in geo_chars and bg[1] in geo_chars:
                continue
            queries.append(bg)
            seen.add(bg)
            if len(queries) >= 5:
                break
        if len(queries) >= 5:
            break

    return queries[:5] if queries else [question]


def _plan_queries(
    client: DeepSeekClient,
    question: str,
    max_queries: int = None,
    temperature: float = 0.2,
    library_context: str = "",
    max_tokens: int = None,
    retry: int = None,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """让 LLM 把问题分解为查询词列表。

    带重试和 bigram 回退：API 偶发失败/返回空内容时，先重试；
    重试仍失败则从问题中提取 bigram 作为查询词，避免整句0命中。
    参数未传时从 settings 读取默认值。

    history: 本会话之前的问答（user/assistant 列表），用于解析代词、省略主语等。
    规划阶段会基于历史上下文理解问题真实意图，避免把"他"等代词直接当查询词。
    """
    # 参数兜底：未传则用默认值
    if max_queries is None:
        max_queries = 7
    if max_tokens is None:
        max_tokens = 1000
    if retry is None:
        retry = 2

    lib_hint = ""
    if library_context:
        lib_hint = (
            f"\n\n【数据来源】{library_context}\n"
            "注意：上述资料库已限定了主体（如地名、机构名等），"
            "规划查询词时【不要】把主体本身作为查询词去搜索以确认主体，"
            "应聚焦于问题中的核心概念和相关领域。"
        )
    # 构造历史上下文提示（让规划阶段理解代词/省略主语）
    history_hint = ""
    if history:
        history_text = _build_history_text(history)
        if history_text:
            history_hint = (
                f"\n\n{history_text}\n"
                "【极重要】当前问题可能含代词（他/她/它/此/这）或省略主语，"
                "请结合上述历史问答理解问题真实意图，"
                "把代词解析为具体的人物/事件/概念后再规划查询词，"
                "不要把代词本身作为查询词。"
            )
    messages = [
        {"role": "system", "content": _PLANNER_SYSTEM},
        {"role": "user", "content": f"用户问题：{question}{lib_hint}{history_hint}\n\n请生成检索计划。"
                                  f"\n（queries 最多 {max_queries} 个，按重要性从高到低排序）"},
    ]

    plan: Dict[str, Any] = {}
    for attempt in range(retry):
        try:
            resp = client.chat(messages, temperature=temperature, max_tokens=max_tokens)
            plan = _extract_json(resp.get("content", "")) or {}
        except Exception as e:
            import sys as _sys
            print(f"[warn] _plan_queries 第{attempt+1}次 API 异常: {e}", file=_sys.stderr)
            plan = {}
        queries = plan.get("queries") or []
        # 清洗：去空、去重
        seen = set()
        cleaned: List[str] = []
        for q in queries:
            if isinstance(q, str):
                q = q.strip()
                if q and q not in seen:
                    seen.add(q)
                    cleaned.append(q)
        if cleaned:
            plan["queries"] = cleaned[:max_queries]
            plan.setdefault("exclude", [])
            plan.setdefault("rationale", "")
            return plan
        if attempt < retry - 1:
            import sys as _sys
            print(f"[warn] _plan_queries 第{attempt+1}次返回空 queries，重试。content={resp.get('content','')[:200] if 'resp' in dir() else 'N/A'}", file=_sys.stderr)

    # 全部重试失败 → bigram 回退
    import sys as _sys
    print(f"[warn] _plan_queries {retry}次均失败，回退到 bigram 分词。question={question}", file=_sys.stderr)
    fallback = _fallback_queries_from_question(question, library_context)
    plan = {
        "queries": fallback[:max_queries],
        "exclude": [],
        "rationale": "规划 API 失败，已回退到问题分词",
    }
    return plan


def _evaluate_results(
    client: DeepSeekClient,
    question: str,
    references: List[Dict[str, Any]],
    used_queries: List[str],
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """让 LLM 评估检索结果是否足够回答问题。"""
    if not references:
        return {
            "sufficient": False,
            "missing_aspect": "无任何命中资料",
            "next_queries": [],
            "confidence": 0.0,
        }
    # 把已有资料的标题/片段拼接给模型
    max_refs = 8
    snippet_chars = 160
    eval_max_tokens = 400
    next_q_limit = 3
    parts = []
    for r in references[:max_refs]:
        heading = r.get("heading") or ""
        sf = r.get("source_file") or ""
        snippet = (r.get("snippet") or "").replace("\n", " ")[:snippet_chars]
        parts.append(f"[{r['index']}] {sf} · {heading}\n{snippet}")
    ctx = "\n\n".join(parts)
    messages = [
        {"role": "system", "content": _EVALUATOR_SYSTEM},
        {"role": "user", "content":
            f"用户问题：{question}\n\n已用查询词：{', '.join(used_queries)}\n\n"
            f"已检索到的资料片段：\n{ctx}\n\n请评估是否足够回答问题。"},
    ]
    try:
        resp = client.chat(messages, temperature=temperature, max_tokens=eval_max_tokens)
        ev = _extract_json(resp.get("content", "")) or {}
    except Exception:
        ev = {}
    # 清洗 next_queries
    nq = ev.get("next_queries") or []
    cleaned: List[str] = []
    seen = set(used_queries)
    for q in nq:
        if isinstance(q, str):
            q = q.strip()
            if q and q not in seen:
                seen.add(q)
                cleaned.append(q)
    ev["next_queries"] = cleaned[:next_q_limit]
    ev.setdefault("sufficient", False)
    ev.setdefault("missing_aspect", "")
    ev.setdefault("confidence", 0.5)
    return ev


def _apply_title_filter_to_hits(
    hits: List[Dict[str, Any]],
    tf: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """对检索结果按标题(heading)和源文件名(source_file)做二次过滤。

    heading 是 chunk 级篇名（如"先主传""刘袁吕传"），不含书名；
    source_file 是源文件名（如"二十四史全译04 三国志.txt"），含书名。
    过滤时同时检查两个字段：任一字段匹配即通过。

    与 agent_workflow 的 _apply_title_filter 逻辑一致：
      contains:     heading 或 source_file 含任一关键词（OR）
      contains_all: heading 或 source_file 含所有关键词（AND）
      excludes:     heading 和 source_file 都不含这些关键词
      regex:        heading 或 source_file 正则匹配（忽略大小写）
    """
    if not tf:
        return hits
    contains = tf.get("contains") or []
    contains_all = tf.get("contains_all") or []
    excludes = tf.get("excludes") or []
    regex_pat = tf.get("regex") or ""

    regex_re = None
    if regex_pat:
        try:
            import re as _re
            regex_re = _re.compile(regex_pat, _re.IGNORECASE)
        except Exception:
            regex_re = None

    def _match_field(text: str) -> bool:
        """检查单个字段是否通过过滤条件。"""
        if contains and not any(kw in text for kw in contains):
            return False
        if contains_all and not all(kw in text for kw in contains_all):
            return False
        if excludes and any(kw in text for kw in excludes):
            return False
        if regex_re and not regex_re.search(text):
            return False
        return True

    filtered = []
    for h in hits:
        heading = (h.get("heading") or "").strip()
        # source_file 可能是完整路径，取 basename 做匹配更精准
        sf = h.get("source_file") or ""
        if sf:
            sf = os.path.basename(sf)
        # heading 或 source_file 任一匹配即通过
        if _match_field(heading) or _match_field(sf):
            filtered.append(h)
    return filtered


# 二十四史书名映射（用于启发式 title_filter 兜底）
_DYNASTY_BOOK_MAP = {
    "史记": {"contains": ["史记"], "excludes": ["汉书", "后汉书", "三国志", "晋书", "宋书", "南齐书", "梁书", "陈书", "魏书", "北齐", "周书", "隋书", "南史", "北史", "旧唐", "新唐", "旧五代", "新五代", "宋史", "辽史", "金史", "元史", "明史"]},
    "汉书": {"contains": ["汉书"], "excludes": ["史记", "后汉书", "三国志", "晋书", "宋书", "南齐书", "梁书", "陈书", "魏书", "北齐", "周书", "隋书", "南史", "北史", "旧唐", "新唐", "旧五代", "新五代", "宋史", "辽史", "金史", "元史", "明史"]},
    "后汉书": {"contains": ["后汉书"], "excludes": ["史记", "汉书", "三国志", "晋书", "宋书", "南齐书", "梁书", "陈书", "魏书", "北齐", "周书", "隋书", "南史", "北史", "旧唐", "新唐", "旧五代", "新五代", "宋史", "辽史", "金史", "元史", "明史"]},
    "三国志": {"contains": ["三国志"], "excludes": ["史记", "汉书", "后汉书", "晋书", "宋书", "南齐书", "梁书", "陈书", "魏书", "北齐", "周书", "隋书", "南史", "北史", "旧唐", "新唐", "旧五代", "新五代", "宋史", "辽史", "金史", "元史", "明史"]},
    "晋书": {"contains": ["晋书"], "excludes": ["史记", "汉书", "后汉书", "三国志", "宋书", "南齐书", "梁书", "陈书", "魏书", "北齐", "周书", "隋书", "南史", "北史", "旧唐", "新唐", "旧五代", "新五代", "宋史", "辽史", "金史", "元史", "明史"]},
    "宋书": {"contains": ["宋书"], "excludes": ["史记", "汉书", "后汉书", "三国志", "晋书", "南齐书", "梁书", "陈书", "魏书", "北齐", "周书", "隋书", "南史", "北史", "旧唐", "新唐", "旧五代", "新五代", "宋史", "辽史", "金史", "元史", "明史"]},
    "南齐书": {"contains": ["南齐书"], "excludes": ["史记", "汉书", "后汉书", "三国志", "晋书", "宋书", "梁书", "陈书", "魏书", "北齐", "周书", "隋书", "南史", "北史", "旧唐", "新唐", "旧五代", "新五代", "宋史", "辽史", "金史", "元史", "明史"]},
    "梁书": {"contains": ["梁书"], "excludes": ["史记", "汉书", "后汉书", "三国志", "晋书", "宋书", "南齐书", "陈书", "魏书", "北齐", "周书", "隋书", "南史", "北史", "旧唐", "新唐", "旧五代", "新五代", "宋史", "辽史", "金史", "元史", "明史"]},
    "陈书": {"contains": ["陈书"], "excludes": ["史记", "汉书", "后汉书", "三国志", "晋书", "宋书", "南齐书", "梁书", "魏书", "北齐", "周书", "隋书", "南史", "北史", "旧唐", "新唐", "旧五代", "新五代", "宋史", "辽史", "金史", "元史", "明史"]},
    "魏书": {"contains": ["魏书"], "excludes": ["史记", "汉书", "后汉书", "三国志", "晋书", "宋书", "南齐书", "梁书", "陈书", "北齐", "周书", "隋书", "南史", "北史", "旧唐", "新唐", "旧五代", "新五代", "宋史", "辽史", "金史", "元史", "明史"]},
    "北齐书": {"contains": ["北齐书"], "excludes": ["史记", "汉书", "后汉书", "三国志", "晋书", "宋书", "南齐书", "梁书", "陈书", "魏书", "周书", "隋书", "南史", "北史", "旧唐", "新唐", "旧五代", "新五代", "宋史", "辽史", "金史", "元史", "明史"]},
    "周书": {"contains": ["周书"], "excludes": ["史记", "汉书", "后汉书", "三国志", "晋书", "宋书", "南齐书", "梁书", "陈书", "魏书", "北齐", "隋书", "南史", "北史", "旧唐", "新唐", "旧五代", "新五代", "宋史", "辽史", "金史", "元史", "明史"]},
    "隋书": {"contains": ["隋书"], "excludes": ["史记", "汉书", "后汉书", "三国志", "晋书", "宋书", "南齐书", "梁书", "陈书", "魏书", "北齐", "周书", "南史", "北史", "旧唐", "新唐", "旧五代", "新五代", "宋史", "辽史", "金史", "元史", "明史"]},
    "南史": {"contains": ["南史"], "excludes": ["史记", "汉书", "后汉书", "三国志", "晋书", "宋书", "南齐书", "梁书", "陈书", "魏书", "北齐", "周书", "隋书", "北史", "旧唐", "新唐", "旧五代", "新五代", "宋史", "辽史", "金史", "元史", "明史"]},
    "北史": {"contains": ["北史"], "excludes": ["史记", "汉书", "后汉书", "三国志", "晋书", "宋书", "南齐书", "梁书", "陈书", "魏书", "北齐", "周书", "隋书", "南史", "旧唐", "新唐", "旧五代", "新五代", "宋史", "辽史", "金史", "元史", "明史"]},
    "旧唐书": {"contains": ["旧唐书"], "excludes": ["史记", "汉书", "后汉书", "三国志", "晋书", "宋书", "南齐书", "梁书", "陈书", "魏书", "北齐", "周书", "隋书", "南史", "北史", "新唐", "旧五代", "新五代", "宋史", "辽史", "金史", "元史", "明史"]},
    "新唐书": {"contains": ["新唐书"], "excludes": ["史记", "汉书", "后汉书", "三国志", "晋书", "宋书", "南齐书", "梁书", "陈书", "魏书", "北齐", "周书", "隋书", "南史", "北史", "旧唐", "旧五代", "新五代", "宋史", "辽史", "金史", "元史", "明史"]},
    "旧五代史": {"contains": ["旧五代史"], "excludes": ["史记", "汉书", "后汉书", "三国志", "晋书", "宋书", "南齐书", "梁书", "陈书", "魏书", "北齐", "周书", "隋书", "南史", "北史", "旧唐", "新唐", "新五代", "宋史", "辽史", "金史", "元史", "明史"]},
    "新五代史": {"contains": ["新五代史"], "excludes": ["史记", "汉书", "后汉书", "三国志", "晋书", "宋书", "南齐书", "梁书", "陈书", "魏书", "北齐", "周书", "隋书", "南史", "北史", "旧唐", "新唐", "旧五代", "宋史", "辽史", "金史", "元史", "明史"]},
    "宋史": {"contains": ["宋史"], "excludes": ["史记", "汉书", "后汉书", "三国志", "晋书", "宋书", "南齐书", "梁书", "陈书", "魏书", "北齐", "周书", "隋书", "南史", "北史", "旧唐", "新唐", "旧五代", "新五代", "辽史", "金史", "元史", "明史"]},
    "辽史": {"contains": ["辽史"], "excludes": ["史记", "汉书", "后汉书", "三国志", "晋书", "宋书", "南齐书", "梁书", "陈书", "魏书", "北齐", "周书", "隋书", "南史", "北史", "旧唐", "新唐", "旧五代", "新五代", "宋史", "金史", "元史", "明史"]},
    "金史": {"contains": ["金史"], "excludes": ["史记", "汉书", "后汉书", "三国志", "晋书", "宋书", "南齐书", "梁书", "陈书", "魏书", "北齐", "周书", "隋书", "南史", "北史", "旧唐", "新唐", "旧五代", "新五代", "宋史", "辽史", "元史", "明史"]},
    "元史": {"contains": ["元史"], "excludes": ["史记", "汉书", "后汉书", "三国志", "晋书", "宋书", "南齐书", "梁书", "陈书", "魏书", "北齐", "周书", "隋书", "南史", "北史", "旧唐", "新唐", "旧五代", "新五代", "宋史", "辽史", "金史", "明史"]},
    "明史": {"contains": ["明史"], "excludes": ["史记", "汉书", "后汉书", "三国志", "晋书", "宋书", "南齐书", "梁书", "陈书", "魏书", "北齐", "周书", "隋书", "南史", "北史", "旧唐", "新唐", "旧五代", "新五代", "宋史", "辽史", "金史", "元史"]},
}

# 朝代名 → 对应史书书名（用于从朝代名推断 title_filter）
_DYNASTY_NAME_TO_BOOK = {
    "西汉": "汉书", "前汉": "汉书", "汉武帝": "汉书", "汉昭帝": "汉书", "汉宣帝": "汉书",
    "东汉": "后汉书", "后汉": "后汉书", "光武帝": "后汉书", "光武": "后汉书",
    "三国": "三国志", "蜀汉": "三国志", "曹魏": "三国志", "东吴": "三国志", "孙吴": "三国志",
    "晋代": "晋书", "西晋": "晋书", "东晋": "晋书",
    "南朝宋": "宋书", "刘宋": "宋书",
    "南齐": "南齐书", "萧齐": "南齐书",
    "南梁": "梁书", "萧梁": "梁书",
    "南陈": "陈书",
    "北魏": "魏书", "拓跋": "魏书",
    "北齐": "北齐书", "高齐": "北齐书",
    "北周": "周书", "宇文": "周书",
    "隋朝": "隋书", "隋代": "隋书", "隋文帝": "隋书", "隋炀帝": "隋书",
    "唐朝": "旧唐书", "唐代": "旧唐书", "唐太宗": "旧唐书", "唐高宗": "旧唐书", "唐玄宗": "旧唐书",
    "北宋": "宋史", "南宋": "宋史", "宋代": "宋史", "宋太祖": "宋史", "宋仁宗": "宋史",
    "辽代": "辽史", "辽朝": "辽史",
    "金代": "金史", "金朝": "金史",
    "元代": "元史", "元朝": "元史", "忽必烈": "元史",
    "明代": "明史", "明朝": "明史", "明太祖": "明史", "明成祖": "明史", "朱元璋": "明史",
    "春秋": None, "左传": None, "战国": None,  # 先秦无二十四史对应
}


def _infer_title_filter_from_question(
    question: str,
    queries: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """启发式兜底：从问题文本或查询词中检测朝代/史书关键词，构造 title_filter。

    当规划器未返回 title_filter 时调用，避免跨朝代召回污染。
    同时检查问题文本和规划器生成的查询词（queries）——
    规划器常在 queries 中加入朝代/史书限定词（如"元史 脱脱"），
    但未必填 title_filter 字段，这里从 queries 中提取信号。

    仅在检测到明确的朝代/史书信号时返回过滤条件，否则返回 None。
    """
    # 合并问题文本和查询词，统一检测
    texts = [question or ""]
    if queries:
        texts.extend(queries)
    combined = " ".join(t for t in texts if t)
    if not combined.strip():
        return None
    # 1. 直接匹配史书书名（在合并文本中查找）
    for book, tf in _DYNASTY_BOOK_MAP.items():
        if book in combined:
            return tf
    # 2. 通过朝代名/帝王名推断
    for name, book in _DYNASTY_NAME_TO_BOOK.items():
        if name in combined and book:
            return _DYNASTY_BOOK_MAP.get(book)
    return None


def _merge_hits(
    accumulated: List[Dict[str, Any]],
    new_hits: List[Dict[str, Any]],
    query: str,
) -> None:
    """把新一轮命中的 chunk 合并进累积列表（同 chunk_id 累加命中信息）。"""
    by_id = {h["chunk_id"]: h for h in accumulated}
    for hit in new_hits:
        cid = hit["chunk_id"]
        if cid in by_id:
            existing = by_id[cid]
            existing["hit_count"] = existing.get("hit_count", 0) + hit.get("hit_count", 0)
            existing["phrase_bonus"] = max(
                existing.get("phrase_bonus", 0), hit.get("phrase_bonus", 0)
            )
            mqs = existing.setdefault("matched_queries", [])
            if query not in mqs:
                mqs.append(query)
            # 合并 matched_words
            mw = set(existing.get("matched_words", [])) | set(hit.get("matched_words", []))
            existing["matched_words"] = sorted(mw)
        else:
            # 新 chunk，附上 matched_queries
            hit = dict(hit)
            hit["matched_queries"] = [query]
            accumulated.append(hit)
            by_id[cid] = hit


def _rescore_agent_hits(hits: List[Dict[str, Any]]) -> None:
    """Agent 模式综合评分：phrase_bonus*放大 + hit_count + 多查询命中加成 + 向量召回加成。

    就地修改 hits 中每个元素的 'score' 字段，并按 score 降序排好。

    评分构成：
      - phrase_bonus × 放大系数（连续词组匹配，最强相关性信号）
      - hit_count（命中次数）
      - 多查询命中加成（被多个查询词命中，加 (N-1) × 加成）
      - 向量召回加成：
        * 双通道命中（关键词 + 向量）：大幅加成（强相关性）
        * 仅向量召回：较小加成 + 语义相似度 × 系数
    """
    for h in hits:
        pb = h.get("phrase_bonus", 0) * PHRASE_WEIGHT_MULTIPLIER
        hc = h.get("hit_count", 0)
        mq = max(0, len(h.get("matched_queries", [])) - 1) * MULTI_QUERY_BONUS
        # 向量召回加成
        channels = h.get("channels") or []
        has_semantic = "semantic" in channels
        has_keyword = "keyword" in channels or not channels  # 无 channels 字段视为关键词命中
        sem_bonus = 0
        if has_semantic and has_keyword:
            sem_bonus = SEMANTIC_BOTH_BONUS
        elif has_semantic:
            sem_bonus = SEMANTIC_ONLY_BONUS
        sem_score = h.get("semantic_score", 0) * SEMANTIC_SCORE_WEIGHT if has_semantic else 0
        h["score"] = pb + hc + mq + sem_bonus + sem_score
    hits.sort(key=lambda x: (x["score"], x.get("phrase_bonus", 0), x["hit_count"]),
              reverse=True)


def _inject_vector_recall(
    question: str,
    registry: LibraryRegistry,
    base_dir: str,
    library_names: Optional[List[str]],
    all_hits: List[Dict[str, Any]],
    top_k: int = 30,
    chunk_filter: Optional[set] = None,
    title_filter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """在已有关键词/关联词命中基础上，注入向量召回结果。

    工作流程：
      1. 调用 search_semantic 在指定库范围内做向量近邻查询
      2. 应用 title_filter（若提供）：按标题过滤掉不相关朝代/文献的干扰内容
         与关键词检索阶段保持一致的过滤口径，避免向量召回引入噪声
      3. 把向量召回结果合并进 all_hits：
         - 已有关键词命中：标注双通道（keyword + semantic），累加 hit_count，
           保留 semantic_score 供 _rescore_agent_hits 加成
         - 仅向量召回的新 chunk：作为新 hit 加入，channels=["semantic"]，
           matched_queries=["[向量召回]"]
      4. 返回统计信息（前端展示用）

    不可用情况（依赖未装 / 索引未就绪 / 设置关闭）：
      返回 available=False，all_hits 不变，调用方应继续走原流程

    Args:
        question: 用户原始问题（用作向量查询文本）
        registry: 库注册表
        base_dir: 工作根目录
        library_names: 限定查询的库名；None 表示全部
        all_hits: 累积命中列表（会被就地修改）
        top_k: 向量召回条数上限
        chunk_filter: 若提供，只召回该集合内的 chunk（分块检索时用）
        title_filter: 若提供，对向量召回结果按标题做二次过滤（与关键词阶段一致）

    Returns:
        {
            "available": bool,       # 向量通道是否实际参与
            "new_hits": int,         # 新增的 chunk 数（去重后）
            "total_hits": int,       # 向量召回总条数（过滤后、含已有关键词命中的）
            "reason": str,           # 不可用时返回原因
        }
    """
    try:
        sr = search_semantic(
            registry, question, base_dir,
            library_names=library_names,
            top_k=top_k,
            chunk_filter=chunk_filter,
        )
    except Exception as e:
        return {"available": False, "new_hits": 0, "total_hits": 0,
                "reason": f"向量召回异常: {e}"}

    if not sr.get("semantic_available", False):
        return {"available": False, "new_hits": 0, "total_hits": 0,
                "reason": sr.get("semantic_reason", "向量索引未就绪")}

    sem_hits = sr.get("results", [])
    if not sem_hits:
        return {"available": True, "new_hits": 0, "total_hits": 0, "reason": ""}

    # 应用 title_filter：与关键词检索阶段保持一致的过滤口径
    # 向量召回可能召回其他朝代/文献的语义相近 chunk（如查"刘备"召回《汉书》内容），
    # 用 title_filter 过滤掉，避免噪声进入后续筛选阶段
    # 【关键修复】过滤后为空时不回退：回退会引入大量跨朝代噪声
    # （如查"脱脱"时向量召回全是晋书/魏书内容，回退后全部进入待筛选）
    filtered_count_before = len(sem_hits)
    if title_filter:
        sem_hits = _apply_title_filter_to_hits(sem_hits, title_filter)
    filtered_count_after = len(sem_hits)

    # 合并到 all_hits（按 chunk_id 去重）
    existing_map = {h["chunk_id"]: h for h in all_hits}
    new_count = 0
    for hit in sem_hits:
        cid = hit["chunk_id"]
        if cid in existing_map:
            # 已有关键词命中：标注双通道，保留语义分
            existing = existing_map[cid]
            channels = existing.setdefault("channels", ["keyword"])
            if "semantic" not in channels:
                channels.append("semantic")
            existing["semantic_score"] = hit.get("semantic_score", 0)
            # 双通道命中视为一次额外"命中"，提升 hit_count
            existing["hit_count"] = existing.get("hit_count", 0) + 1
        else:
            # 仅向量召回的新 chunk
            new_hit = dict(hit)
            new_hit["matched_queries"] = ["[向量召回]"]
            new_hit["channels"] = ["semantic"]
            new_hit["hit_count"] = 0  # 向量召回无关键词命中次数
            new_hit["phrase_bonus"] = 0
            all_hits.append(new_hit)
            existing_map[cid] = new_hit
            new_count += 1

    return {"available": True, "new_hits": new_count,
            "total_hits": len(sem_hits), "reason": ""}


def ai_search_agent_stream(
    question: str,
    registry: LibraryRegistry,
    client: DeepSeekClient,
    base_dir: str,
    library_names: Optional[List[str]] = None,
    parallel: int = 4,
    top_k: int = 20,
    max_chunks: Optional[int] = None,
    max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
    max_total_chars: Optional[int] = None,
    max_rounds: int = None,
    temperature: float = 0.3,
) -> Iterator[Dict[str, Any]]:
    """Agent 模式智能检索（流式）。

    事件序列：
        {"phase":"plan","queries":[...],"rationale":"..."}
        {"phase":"round_start","round":1,"queries":[...]}
        {"phase":"searching","query":"...","total_hits":N,"new_hits":M}
        {"phase":"retrieval","round":R,"queries":[...],"references":[...],
         "total_unique":N,"used_queries":[...]}
        {"phase":"evaluating","round":R}
        {"phase":"evaluated","round":R,"evaluation":{...}}
        {"phase":"generating"}
        {"phase":"reasoning","delta":"..."}
        {"phase":"content","delta":"..."}
        {"phase":"done","usage":{...}}
    {"phase":"error","stage":"...","error":"..."}
    """
    # top_k 同时控制检索结果数与 context 引用条数
    if max_chunks is None:
        max_chunks = top_k
    if max_total_chars is None:
        max_total_chars = _compute_total_chars_budget(top_k)
    if max_rounds is None:
        max_rounds = 3
    try:
        # ---------- 阶段 1：规划 ----------
        # 构造库背景信息注入规划提示词，避免在不知道库背景的情况下
        # 规划出与库内容脱节的关键词（如古文库却规划现代汉语词）
        library_context = _build_library_context(registry, library_names)
        plan = _plan_queries(client, question, library_context=library_context)
        queries: List[str] = list(plan.get("queries", []))
        yield {
            "phase": "plan",
            "queries": queries,
            "rationale": plan.get("rationale", ""),
            "exclude": plan.get("exclude", []),
        }

        all_hits: List[Dict[str, Any]] = []
        used_queries: List[str] = []
        references: List[Dict[str, Any]] = []

        # ---------- 阶段 2 & 3：多轮检索 + 评估 ----------
        current_queries = queries
        for round_idx in range(max_rounds):
            round_num = round_idx + 1
            yield {"phase": "round_start", "round": round_num, "queries": current_queries}

            # 对本轮每个查询词独立检索
            existing_ids = {h["chunk_id"] for h in all_hits}
            for q in current_queries:
                if q in used_queries:
                    continue
                try:
                    sr = parallel_search(
                        registry, q,
                        library_names=library_names,
                        parallel=parallel,
                        base_dir=base_dir,
                    )
                except Exception as e:
                    yield {"phase": "searching", "query": q,
                           "error": f"检索失败: {e}"}
                    continue
                raw_hits = sr["results"][:top_k]
                # 区分：该查询词命中的 chunk 中，有多少是之前已命中的，多少是新增的
                truly_new = [h for h in raw_hits if h["chunk_id"] not in existing_ids]
                _merge_hits(all_hits, raw_hits, q)
                # 合并后更新 existing_ids
                existing_ids = {h["chunk_id"] for h in all_hits}
                used_queries.append(q)
                yield {
                    "phase": "searching",
                    "query": q,
                    "total_hits": sr.get("total_hits", 0),
                    "new_hits": len(truly_new),  # 真正新增的 chunk 数
                    "repeat_hits": len(raw_hits) - len(truly_new),  # 已命中过的
                    "round": round_num,
                }

            # 重新评分排序
            _rescore_agent_hits(all_hits)
            top_hits = all_hits[:top_k]
            context, references = _build_context(
                base_dir, top_hits, max_chunks, max_chars_per_chunk, max_total_chars,
            )

            yield {
                "phase": "retrieval",
                "round": round_num,
                "queries": current_queries,
                "references": references,
                "total_unique": len(all_hits),
                "used_queries": list(used_queries),
            }

            # 若已是最后一轮，不再评估
            if round_idx >= max_rounds - 1:
                break

            # 评估
            if not references:
                # 没有任何命中，直接进入下一轮（若有补充词）或结束
                yield {"phase": "evaluating", "round": round_num}
                ev = {
                    "sufficient": False,
                    "missing_aspect": "无命中资料",
                    "next_queries": [],
                    "confidence": 0.0,
                }
                yield {"phase": "evaluated", "round": round_num, "evaluation": ev}
                if not ev["next_queries"]:
                    break
                current_queries = ev["next_queries"]
                continue

            yield {"phase": "evaluating", "round": round_num}
            ev = _evaluate_results(client, question, references, used_queries)
            yield {"phase": "evaluated", "round": round_num, "evaluation": ev}

            if ev.get("sufficient"):
                break
            next_q = ev.get("next_queries") or []
            if not next_q:
                break
            current_queries = next_q

        # ---------- 阶段 3.5：注入向量召回（多轮检索完成后） ----------
        # 用原始问题做向量近邻查询，补充关键词检索的召回盲区
        # （同义表述、语义分散、概念隐含）
        vr = _inject_vector_recall(
            question, registry, base_dir, library_names,
            all_hits, top_k=top_k,
        )
        yield {
            "phase": "vector_recall",
            "available": vr["available"],
            "new_hits": vr["new_hits"],
            "total_hits": vr["total_hits"],
            "reason": vr["reason"],
        }
        # 重新评分（含向量召回加成）+ 重建 context
        _rescore_agent_hits(all_hits)
        top_hits = all_hits[:top_k]
        context, references = _build_context(
            base_dir, top_hits, max_chunks, max_chars_per_chunk, max_total_chars,
        )

        # ---------- 阶段 4：生成答案 ----------
        if not references:
            yield {"phase": "content",
                   "delta": "经过多轮检索仍未找到与问题相关的资料，无法生成回答。"}
            yield {"phase": "done", "usage": None}
            return

        yield {"phase": "generating"}
        for event in client.ask_stream(
            question, context=context,
            model=client.model, temperature=temperature,
        ):
            etype = event.get("type")
            if etype == "reasoning":
                yield {"phase": "reasoning", "delta": event.get("delta", "")}
            elif etype == "content":
                yield {"phase": "content", "delta": event.get("delta", "")}
            elif etype in ("finish", "done"):
                yield {"phase": "done", "usage": event.get("usage")}

    except DeepSeekError as e:
        yield {"phase": "error", "stage": "generation",
               "error": f"DeepSeek 调用失败: {e}"}
    except Exception as e:
        yield {"phase": "error", "stage": "agent",
               "error": f"Agent 执行失败: {e}"}


# ============================================================
#  多轮对话流式（会话模式）
# ============================================================
#
# 支持基于历史消息"接着问"：把历史 user/assistant 消息作为对话上下文
# 传给模型，让模型能引用前文。每次新问题仍会独立检索资料。
# ============================================================


def _build_history_messages(
    question: str,
    context: str,
    history: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """把历史问答 + 新问题拼成 chat messages。

    history 中的 assistant 消息只取最终 content（不含 reasoning 思考过程），
    避免把内部推理塞进对话历史污染上下文。
    """
    if context:
        system = (
            "你是一名严谨的资料分析助手。请仅根据下方提供的【参考资料】回答问题。"
            "若资料不足以回答，请明确说明「资料中未提及」，不要编造内容。"
            "回答时在关键信息后用 [n] 标注引用的第 n 条资料。\n"
            "这是多轮对话，你可以引用前文已讨论过的内容，但新问题若涉及新事实，"
            "请以本次提供的【参考资料】为准。"
        )
    else:
        system = (
            "你是一名严谨的资料分析助手。请清晰、准确地回答问题。\n"
            "这是多轮对话，你可以引用前文已讨论过的内容。"
        )
    messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
    # 拼接历史（默认最近 6 轮 = 12 条消息，可设置 gen_history_rounds）
    hist_rounds = _setting("gen_history_rounds", 6)
    recent = history[-(hist_rounds * 2):]
    for m in recent:
        role = m.get("role", "")
        content = m.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    # 当前问题 + 资料
    if context:
        messages.append({
            "role": "user",
            "content": f"【参考资料】\n{context}\n\n【问题】\n{question}",
        })
    else:
        messages.append({"role": "user", "content": question})
    return messages


def simple_chat_stream(
    question: str,
    history: List[Dict[str, Any]],
    client: DeepSeekClient,
    temperature: float = 0.5,
    extra_context: Optional[List[Dict[str, Any]]] = None,
) -> Iterator[Dict[str, Any]]:
    """纯对话模式（不检索），直接把问题发给大模型，带历史上下文。

    若提供 extra_context（来自检索结果"加到上下文"的 chunk 列表），
    则将其作为系统参考资料注入，让模型基于这些内容回答。

    事件序列：
        {"phase":"content","delta":"..."}
        {"phase":"reasoning","delta":"..."}   # 思考模式
        {"phase":"done","usage":{...}}
        {"phase":"error","stage":"chat","error":"..."}
    """
    system = (
        "你是一名乐于助人的中文对话助手。请清晰、准确地回答用户问题。\n"
        "这是多轮对话，你可以引用前文已讨论过的内容。"
        "若不确定，请坦诚说明，不要编造。"
    )
    # 注入额外上下文（来自检索结果"加到上下文"）
    if extra_context:
        parts = []
        for i, c in enumerate(extra_context, 1):
            heading = c.get("heading") or ""
            fp = c.get("file_path") or ""
            text = c.get("text") or ""
            tag = f"[资料 {i}]"
            if heading:
                tag += f" {heading}"
            if fp:
                tag += f"（来源: {os.path.basename(fp)}）"
            parts.append(f"{tag}\n{text}")
        ctx_text = "\n\n".join(parts)
        system = (
            system + "\n\n"
            "以下是用户从资料库中挑选的参考内容，请优先基于这些内容回答用户问题。"
            "若参考内容不足以下结论，可结合你的知识补充，但需明确区分。"
            f"\n\n=== 参考资料开始 ===\n{ctx_text}\n=== 参考资料结束 ==="
        )
    messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
    # 拼接历史（默认最近 6 轮，可设置 gen_history_rounds）
    hist_rounds = _setting("gen_history_rounds", 6)
    recent = history[-(hist_rounds * 2):]
    for m in recent:
        role = m.get("role", "")
        content = m.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})

    try:
        for event in client.chat_stream(
            messages, model=client.model, temperature=temperature,
        ):
            etype = event.get("type")
            if etype == "reasoning":
                yield {"phase": "reasoning", "delta": event.get("delta", "")}
            elif etype == "content":
                yield {"phase": "content", "delta": event.get("delta", "")}
            elif etype in ("finish", "done"):
                yield {"phase": "done", "usage": event.get("usage")}
    except DeepSeekError as e:
        yield {"phase": "error", "stage": "chat",
               "error": f"DeepSeek 调用失败: {e}"}
    except Exception as e:
        yield {"phase": "error", "stage": "chat",
               "error": f"对话失败: {e}"}


def chat_stream(
    question: str,
    history: List[Dict[str, Any]],
    registry: LibraryRegistry,
    client: DeepSeekClient,
    base_dir: str,
    library_names: Optional[List[str]] = None,
    parallel: int = 4,
    top_k: int = 20,
    max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
    temperature: float = 0.3,
) -> Iterator[Dict[str, Any]]:
    """多轮对话流式检索 + 生成。

    与 ai_search_stream 的区别：把 history 拼进对话 messages，让模型能"接着问"。
    每次新问题仍独立检索（保证资料时效性）。

    事件序列同 ai_search_stream，额外无 history 字段（前端自己有）。
    """
    max_chunks = top_k
    max_total_chars = _compute_total_chars_budget(top_k)
    try:
        # 1. 检索（独立检索，不依赖历史）
        search_result = parallel_search(
            registry, question,
            library_names=library_names,
            parallel=parallel,
            base_dir=base_dir,
        )
        hits = search_result["results"][:top_k]

        # 2. 构造 context
        context, references = _build_context(
            base_dir, hits, max_chunks, max_chars_per_chunk, max_total_chars,
        )

        retrieval_summary = {
            "total_hits": search_result["total_hits"],
            "used_hits": len(references),
            "searched_libraries": search_result["searched_libraries"],
        }

        yield {
            "phase": "retrieval",
            "retrieval": retrieval_summary,
            "references": references,
        }

        if not references:
            yield {"phase": "content",
                   "delta": "未检索到与问题相关的资料，无法生成回答。"}
            yield {"phase": "done", "usage": None}
            return

        # 3. 流式生成（带历史）
        messages = _build_history_messages(question, context, history)
        for event in client.chat_stream(
            messages, model=client.model, temperature=temperature,
        ):
            etype = event.get("type")
            if etype == "reasoning":
                yield {"phase": "reasoning", "delta": event.get("delta", "")}
            elif etype == "content":
                yield {"phase": "content", "delta": event.get("delta", "")}
            elif etype in ("finish", "done"):
                yield {"phase": "done", "usage": event.get("usage")}

    except DeepSeekError as e:
        yield {"phase": "error", "stage": "generation",
               "error": f"DeepSeek 调用失败: {e}"}
    except Exception as e:
        yield {"phase": "error", "stage": "retrieval",
               "error": f"检索失败: {e}"}


# ============================================================
#  精读模式（小 chunk + 工具展开）
# ============================================================
#
# 与直接/Agent 模式的区别：
#   - 不把整 chunk 塞进 context，而是取命中位置附近 200 字作为"小 chunk"
#   - 大模型看到小 chunk 后，用 expand_chunks 工具自主展开需要的 chunk
#   - 每个查询词最多 2 轮 expand，每轮可批量展开多个 chunk
#   - 工具阶段非流式（等完整 tool_calls），生成阶段流式
#
# 优点：初始 context 很小（200字 × N），能塞很多候选；大模型自主决定展开谁，省 token
# ============================================================

# expand_chunks 工具定义（OpenAI function calling 格式）
_EXPAND_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "expand_chunks",
            "description": (
                "展开指定的小资料，获取其完整文本。"
                "当你认为某个小资料的片段值得深入阅读时，调用此工具获取更多上下文。"
                "一次可以指定多个资料编号批量展开。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ref_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "要展开的资料编号列表（[n] 编号，1 到 资料总数），如 [1, 3, 5]",
                    },
                    "length": {
                        "type": "integer",
                        "description": f"每个资料取前多少字，范围 1~{DEEPREAD_EXPAND_MAX_CHARS}",
                    },
                },
                "required": ["ref_ids"],
            },
        },
    }
]

_DEEPREAD_SYSTEM = """你是一名严谨的档案精读分析助手。工作流程：

1. 你会收到若干【小资料】（关键词命中位置附近 200 字），每个带编号 [n]（n=1..N）
2. 阅读这些小资料，判断哪些值得深入
3. 调用 expand_chunks 工具展开你认为重要的小资料，获取完整文本（参数 ref_ids 为资料编号列表）
4. 每个查询词方向最多展开 2 轮，每轮可批量指定多个资料编号
5. 收集够信息后，直接给出最终答案（不要再调用工具）

回答要求：
- 仅根据展开后的资料回答，不要编造
- 关键信息后用 [n] 标注引用的第 n 条小资料（n 必须在 1 到 资料总数 范围内）
- 若资料不足，明确说明「资料中未提及」
- 你只能使用我们提供给你的 [1]、[2]...[N] 编号，绝不能使用超出范围的数字
- 原文中的 [数字] 注解（如 [91]）是史书脚注号，不是你的引用标记，不要复用

注意：expand_chunks 的 length 参数表示想看每个资料的前多少字（最多 10000）。"""


def _build_mini_snippets(
    base_dir: str,
    hits: List[Dict[str, Any]],
    window: int = None,
) -> List[Dict[str, Any]]:
    """把命中 chunk 转成 200 字小 chunk 列表（命中位置附近 ±window 字）。

    返回结构：
        [
            {
                "index": 1,
                "chunk_id": "zone_001/chunk_000123",
                "library": "郎溪县志",
                "source_file": "第一节 计划管理.docx",
                "heading": "第三节 计划管理",
                "matched_words": ["经济发展", "计划"],
                "hit_count": 5,
                "mini_snippet": "...命中位置附近200字...",
            },
            ...
        ]
    """
    mini_chunks: List[Dict[str, Any]] = []
    for i, r in enumerate(hits, 1):
        lib_name = r.get("library", "")
        chunk_id = r.get("chunk_id", "")
        full_text = _load_chunk_text(base_dir, lib_name, chunk_id)
        if not full_text:
            # 回退到搜索返回的 snippet
            full_text = r.get("snippet", "") or ""
        if window is None:
            window = 100

        # 找命中位置：用 matched_words 在 full_text 中定位
        matched_words = r.get("matched_words", [])
        pos = -1
        for w in matched_words:
            p = full_text.find(w)
            if p >= 0:
                pos = p
                break

        if pos < 0:
            pos = 0  # 找不到就从头取

        start = max(0, pos - window)
        end = min(len(full_text), pos + window)
        snippet = full_text[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(full_text):
            snippet = snippet + "..."

        mini_chunks.append({
            "index": i,
            "chunk_id": chunk_id,
            "library": lib_name,
            "source_file": r.get("source_file", ""),
            "source_file_path": r.get("source_file_path", ""),
            "heading": r.get("heading", ""),
            "matched_words": matched_words,
            "hit_count": r.get("hit_count", 0),
            "matched_queries": r.get("matched_queries", []),
            "mini_snippet": snippet.replace("\n", " "),
            "metadata": r.get("metadata"),
        })
    return mini_chunks


def _format_mini_chunks_for_prompt(mini_chunks: List[Dict[str, Any]]) -> str:
    """把小 chunk 列表格式化成给大模型的文本。

    注意：不显示 chunk_id，避免模型误用 chunk_id 中的数字（如 chunk_000035 中的 35）
    作为引用编号。模型应使用 [n] 编号（n=1..N）调用工具和标注引用。

    元数据注入：在 heading 后追加时代/主题/人物摘要，让模型在筛选阶段
    就能基于朝代/主题判断 keep/drop，减少跨朝代干扰。
    """
    parts = []
    for mc in mini_chunks:
        md_summary = _format_metadata_for_prompt(mc.get("metadata"))
        header = f"[{mc['index']}] 来源：{mc['source_file']}（{mc['library']}） · {mc['heading']}"
        if md_summary:
            header += f" · {md_summary}"
        parts.append(
            f"{header}\n"
            f"命中词: {', '.join(mc['matched_words'])}（{mc['hit_count']}处）\n"
            f"片段: {mc['mini_snippet']}"
        )
    return "\n\n".join(parts)


# ============================================================
#  纯向量检索模式（global / precise / complex 的向量版本）
# ============================================================
#
# 与混合词模式的区别：
#   - 混合词模式：关键词检索（parallel_search）为主 + 向量召回补充
#   - 纯向量模式：全程向量检索，不依赖关键词命中
#     * global：多轮规划 → 大chunk→小chunk递进（search_semantic_progressive）
#     * precise：多轮规划 → 小chunk→二次切分→二次向量（search_semantic_precise）
#     * complex：拆分子问题 → 按粒度路由到 global_vector / precise_vector
#
# 多轮检索规划器（_plan_vector_queries_round）：
#   每轮生成3个查询词，根据已有结果决定是否继续检索
# ============================================================

# 向量检索规划器系统提示词
_VECTOR_PLANNER_SYSTEM = """你是一名检索规划专家，为向量检索生成查询词。

任务：
- 根据用户问题，生成简短的查询词供向量检索使用
- 每轮生成恰好3个查询词，覆盖问题的不同方面
- 查询词长度2~6字，简洁有力，是核心概念词而非完整句子
- 不要拆分专有名词（如"官渡之战"不能拆成"官渡"+"之战"，应整体使用或换角度表述）

多轮检索规则：
- 首轮：基于问题生成初始查询词，覆盖问题的主要方面
- 后续轮：根据已有检索结果决定是否需要继续检索
  * 若已找到足够信息回答问题 → should_continue 设为 false
  * 若仍有未覆盖的方面 → should_continue 设为 true，并生成新的查询词
- 新查询词应探索之前未覆盖的角度，避免与已有查询词重复

返回严格JSON格式（不要加markdown围栏，不要输出多余文字）：
{
  "queries": ["查询词1", "查询词2", "查询词3"],
  "should_continue": true,
  "rationale": "简述生成理由或终止原因"
}"""

# 子问题粒度分类系统提示词（complex 向量模式用）
_VECTOR_GRANULARITY_SYSTEM = """你是一名检索策略专家。根据每个子问题的特征，为其选择最合适的向量检索粒度。

规则：
- "global"：适合宽泛、概述性、多方面的问题（如"介绍XX的历史发展"、"XX的整体概况"）
- "precise"：适合具体、细节性、定位精准的问题（如"XX事件发生在哪一年"、"XX的具体数值是多少"）

返回严格JSON（不要加markdown围栏）：
{
  "granularities": ["global", "precise", ...]
}"""


def _plan_vector_queries_round(
    client: DeepSeekClient,
    question: str,
    library_context: str = "",
    round_num: int = 1,
    max_rounds: int = 3,
    queries_per_round: int = 3,
    prev_rounds: Optional[List[Dict]] = None,
    temperature: float = 0.3,
) -> Dict[str, Any]:
    """向量检索多轮规划器：每轮生成查询词并决定是否继续。

    首轮（round_num=1）基于问题生成初始查询词；
    后续轮接收 prev_rounds（之前轮次的查询词和命中摘要），决定是否继续 + 生成新词。

    Args:
        prev_rounds: 之前轮次的信息列表，每项含：
            {"round": int, "queries": [str], "hits_summary": str}

    Returns:
        {
            "queries": ["词1", "词2", "词3"],
            "should_continue": bool,
            "rationale": str,
            "round": int,
        }
    """
    lib_hint = f"\n\n【数据来源】{library_context}" if library_context else ""

    # 构造用户消息：首轮只给问题，后续轮附带已有检索结果
    if round_num == 1 or not prev_rounds:
        user_content = (
            f"用户问题：{question}{lib_hint}\n\n"
            f"请生成 {queries_per_round} 个查询词用于向量检索。"
        )
    else:
        prev_text = "\n".join(
            f"第{r.get('round', i + 1)}轮查询词：{'、'.join(r.get('queries', []))}\n"
            f"命中概况：{r.get('hits_summary', '无')}"
            for i, r in enumerate(prev_rounds)
        )
        user_content = (
            f"用户问题：{question}{lib_hint}\n\n"
            f"【已有检索结果】\n{prev_text}\n\n"
            f"当前是第{round_num}轮（最多{max_rounds}轮）。"
            f"请判断是否需要继续检索，若需要则生成 {queries_per_round} 个新的查询词。"
        )

    messages = [
        {"role": "system", "content": _VECTOR_PLANNER_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    try:
        resp = client.chat(messages, temperature=temperature, max_tokens=600)
        plan = _extract_json(resp.get("content", "")) or {}
    except Exception as e:
        import sys as _sys
        print(f"[warn] _plan_vector_queries_round 第{round_num}轮 API 异常: {e}",
              file=_sys.stderr)
        plan = {}

    # 清洗查询词：去空、去重、限制数量
    raw_queries = plan.get("queries") or []
    seen: set = set()
    cleaned: List[str] = []
    for q in raw_queries:
        if isinstance(q, str):
            q = q.strip()
            if q and q not in seen:
                seen.add(q)
                cleaned.append(q)

    if cleaned:
        plan["queries"] = cleaned[:queries_per_round]
        plan.setdefault("should_continue", round_num < max_rounds)
        plan.setdefault("rationale", "")
        plan["round"] = round_num
        return plan

    # 失败回退：从问题中提取 bigram 作为查询词，不继续后续轮次
    import sys as _sys
    print(f"[warn] _plan_vector_queries_round 第{round_num}轮返回空 queries，回退到 bigram。"
          f"question={question}", file=_sys.stderr)
    return {
        "queries": _fallback_queries_from_question(question, library_context)[:queries_per_round],
        "should_continue": False,
        "rationale": "规划 API 失败，已回退到问题分词",
        "round": round_num,
    }


_CHUNK_FILTER_SYSTEM = """你是一名资料筛选专家，负责判断检索到的 chunk 是否与用户问题相关。

任务：
- 阅读每个 chunk 的摘要（标题+片段），判断它与用户问题的相关性
- 保留与问题直接相关、含关键信息的 chunk
- 排除明显无关、仅字面巧合命中、内容空洞的 chunk
- 严格但不过度：只要 chunk 含有与问题相关的事实、数据、描述，就保留

返回严格JSON格式（不要加markdown围栏，不要输出多余文字）：
{
  "keep": ["chunk_id1", "chunk_id2"],
  "drop": ["chunk_id3"],
  "rationale": "简述筛选理由"
}"""

def _filter_chunks_round(
    client: DeepSeekClient,
    question: str,
    candidates: List[Dict[str, Any]],
    round_num: int = 1,
    temperature: float = 0.3,
) -> Dict[str, Any]:
    """让模型筛选本轮检索到的 chunk，返回保留/排除的 chunk_id 列表。

    Args:
        candidates: 候选 chunk 列表（已按评分取前N），每项需含 chunk_id、heading、snippet

    Returns:
        {"keep": [chunk_id,...], "drop": [chunk_id,...], "rationale": str}
    """
    if not candidates:
        return {"keep": [], "drop": [], "rationale": "无候选 chunk"}

    # 构造候选列表文本
    lines = []
    for i, c in enumerate(candidates, 1):
        cid = c.get("chunk_id", "")
        heading = c.get("heading", "") or c.get("source_file", "")
        snippet = c.get("snippet", "") or c.get("mini_snippet", "")
        snippet = snippet[:200]
        lines.append(f"{i}. chunk_id={cid} | 标题={heading} | 片段={snippet}")
    cand_text = "\n".join(lines)

    user_content = (
        f"用户问题：{question}\n\n"
        f"【候选 chunk 列表】（共{len(candidates)}条）\n{cand_text}\n\n"
        f"请判断每个 chunk 是否与问题相关，返回保留和排除的 chunk_id 列表。"
    )

    messages = [
        {"role": "system", "content": _CHUNK_FILTER_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    try:
        resp = client.chat(messages, temperature=temperature, max_tokens=1000)
        result = _extract_json(resp.get("content", "")) or {}
    except Exception as e:
        import sys as _sys
        print(f"[warn] _filter_chunks_round 第{round_num}轮筛选异常: {e}",
              file=_sys.stderr)
        return {"keep": [c["chunk_id"] for c in candidates], "drop": [],
                "rationale": "筛选API异常，全部保留"}

    keep = result.get("keep", [])
    drop = result.get("drop", [])
    # 清洗：确保 keep/drop 中的 id 确实在候选中
    cand_ids = {c.get("chunk_id", "") for c in candidates}
    keep = [cid for cid in keep if cid in cand_ids]
    drop = [cid for cid in drop if cid in cand_ids]
    # 未明确归类的 chunk 默认保留
    classified = set(keep) | set(drop)
    for c in candidates:
        cid = c.get("chunk_id", "")
        if cid and cid not in classified:
            keep.append(cid)

    return {
        "keep": keep,
        "drop": drop,
        "rationale": result.get("rationale", ""),
    }


def _dedup_vector_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """向量命中去重：按 chunk_id 合并，保留最高分，累积 matched_queries。

    与 _merge_hits 的区别：
    - _merge_hits 保留首次命中的分数（关键词模式按命中次数排序）
    - 本函数保留最高分（向量模式按语义相似度排序）
    """
    by_id: Dict[str, Dict[str, Any]] = {}
    for h in hits:
        cid = h.get("chunk_id", "")
        if not cid:
            continue
        if cid not in by_id:
            by_id[cid] = dict(h)
            by_id[cid]["matched_queries"] = list(h.get("matched_queries", []))
        else:
            ex = by_id[cid]
            # 保留最高分的版本
            if h.get("score", 0) > ex.get("score", 0):
                mqs = ex.get("matched_queries", [])
                ex.update(h)
                ex["matched_queries"] = mqs
            # 合并 matched_queries
            for q in h.get("matched_queries", []):
                if q not in ex["matched_queries"]:
                    ex["matched_queries"].append(q)
    return list(by_id.values())


def _build_vector_mini_chunks(
    hits: List[Dict[str, Any]],
    text_field: str = "sub_text",
) -> List[Dict[str, Any]]:
    """从向量检索结果构造小 chunk 列表（用向量命中的子片段作为 mini_snippet）。

    与 _build_mini_snippets 的区别：
    - _build_mini_snippets 用 matched_words 定位命中位置（关键词模式）
    - 本函数直接用向量检索返回的子片段文本（sub_text / subchunk_text）

    Args:
        text_field: 用哪个字段作为命中片段
            - "sub_text"：global 模式（progressive 返回的子片段）
            - "subchunk_text"：precise 模式（二次切分命中的子片段，更精确）
    """
    mini_chunks: List[Dict[str, Any]] = []
    for i, h in enumerate(hits, 1):
        # 优先用向量命中的子片段文本，回退到 snippet
        hit_text = h.get(text_field, "") or ""
        if not hit_text:
            hit_text = h.get("snippet", "") or ""
        # 截断到 200 字（与 _build_mini_snippets 一致）
        snippet = hit_text[:200].replace("\n", " ")
        if len(hit_text) > 200:
            snippet = snippet + "..."
        mini_chunks.append({
            "index": i,
            "chunk_id": h.get("chunk_id", ""),
            "library": h.get("library", ""),
            "source_file": h.get("source_file", ""),
            "source_file_path": h.get("source_file_path", ""),
            "heading": h.get("heading", ""),
            "matched_words": h.get("matched_words", []),
            "hit_count": h.get("hit_count", 0),
            "matched_queries": h.get("matched_queries", []),
            "mini_snippet": snippet,
        })
    return mini_chunks


# ============================================================
#  向量模式共用阶段（global / precise / complex 复用）
# ============================================================

def _build_extra_ctx_text(extra_context: Optional[List[Dict]]) -> str:
    """把用户从检索结果挑选的额外上下文格式化为注入提示词的文本块。"""
    if not extra_context:
        return ""
    parts = []
    for i, c in enumerate(extra_context, 1):
        heading = c.get("heading") or ""
        fp = c.get("file_path") or ""
        text = c.get("text") or ""
        tag = f"[用户选定资料 {i}]"
        if heading:
            tag += f" {heading}"
        if fp:
            tag += f"（来源: {os.path.basename(fp)}）"
        parts.append(f"{tag}\n{text}")
    return (
        "【用户选定资料】（用户从检索结果中挑选的参考内容，请重点参考）\n"
        + "\n\n".join(parts) + "\n\n"
    )


def _summarize_round_hits(hits: List[Dict[str, Any]]) -> str:
    """构造一轮检索命中的摘要文本（供下轮规划器参考）。"""
    if not hits:
        return "本轮无命中"
    top_headings = [
        h.get("heading", "") or h.get("source_file", "")
        for h in hits[:5] if h.get("heading") or h.get("source_file")
    ]
    summary = f"共{len(hits)}条命中"
    if top_headings:
        summary += f"，涉及：{'、'.join(top_headings)}"
    return summary


def _plan_vector_round(
    client: DeepSeekClient,
    question: str,
    library_context: str,
    round_num: int,
    max_rounds: int,
    queries_per_round: int,
    prev_rounds: List[Dict[str, Any]],
    temperature: float,
) -> Iterator[Dict[str, Any]]:
    """规划一轮向量查询词并 yield plan 事件。

    返回 (queries, proceed)：proceed=False 表示非首轮且规划器认为
    无需继续，调用方应中断检索循环。
    """
    plan = _plan_vector_queries_round(
        client, question, library_context=library_context,
        round_num=round_num, max_rounds=max_rounds,
        queries_per_round=queries_per_round,
        prev_rounds=prev_rounds if prev_rounds else None,
        temperature=temperature,
    )
    queries: List[str] = plan.get("queries", [])
    should_continue = plan.get("should_continue", round_num < max_rounds)
    yield {
        "phase": "plan",
        "round": round_num,
        "queries": queries,
        "rationale": plan.get("rationale", ""),
        "should_continue": should_continue,
        "vector_mode": True,
    }
    if round_num > 1 and not should_continue:
        return [], False
    return queries, True


def _finalize_vector_hits(
    raw_all_hits: List[Dict[str, Any]],
    max_mini_chunks: int = 500,
) -> tuple:
    """合并去重、排序、截断。返回 (truncated, total_unique)。"""
    all_hits = _dedup_vector_hits(raw_all_hits)
    # 按分数降序，多查询命中的 chunk 优先
    all_hits.sort(
        key=lambda h: (h.get("score", 0), len(h.get("matched_queries", []))),
        reverse=True,
    )
    return all_hits[:max_mini_chunks], len(all_hits)


def _build_vector_references(mini_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把 mini_chunks 转为 retrieval 事件使用的 references 结构。"""
    return [{
        "index": mc["index"],
        "library": mc["library"],
        "library_note": "",
        "chunk_id": mc["chunk_id"],
        "source_file": mc["source_file"],
        "source_file_path": mc["source_file_path"],
        "source_sha256": "",
        "heading": mc["heading"],
        "hit_count": mc["hit_count"],
        "matched_words": mc["matched_words"],
        "snippet": mc["mini_snippet"],
    } for mc in mini_chunks]


def _warn_if_noisy(total_unique: int, truncated_len: int) -> Iterator[Dict[str, Any]]:
    """命中数超过精度预警阈值时 yield warn 事件。"""
    if total_unique > EFFORT_WARN_THRESHOLD:
        yield {
            "phase": "warn",
            "message": f"命中 {total_unique} 条较多，可能含噪声"
                       f"（精度预警阈值 {EFFORT_WARN_THRESHOLD}）",
            "total_unique": total_unique,
            "truncated_to": truncated_len,
        }


def _stream_llm_answer(
    client: DeepSeekClient,
    messages: List[Dict[str, Any]],
    temperature: float,
) -> Iterator[Dict[str, Any]]:
    """不带工具的纯流式生成，yield reasoning/content/done 事件。"""
    for event in client.chat_stream(messages, model=client.model,
                                    temperature=temperature):
        etype = event.get("type")
        if etype == "reasoning":
            yield {"phase": "reasoning", "delta": event.get("delta", "")}
        elif etype == "content":
            yield {"phase": "content", "delta": event.get("delta", "")}
        elif etype in ("finish", "done"):
            yield {"phase": "done", "usage": event.get("usage")}


def _run_expand_chunks_loop(
    client: DeepSeekClient,
    conversation: List[Dict[str, Any]],
    mini_chunks: List[Dict[str, Any]],
    base_dir: str,
    all_used_queries: List[str],
    temperature: float,
    expand_rounds_per_query: int = 2,
    expand_max: int = 10000,
) -> Iterator[Dict[str, Any]]:
    """expand_chunks 工具循环（global / precise 向量模式共用）。

    content 分片立即 yield 给前端（真正的流式输出）；tool_calls 分片由
    chat_stream 累积，流结束后一次性处理。

    返回 True 表示模型已不再调用工具、最终答案已流式输出完毕（已 yield
    done 事件），调用方直接返回；返回 False 表示展开轮数耗尽，调用方进入
    阶段 4 兜底生成。
    """
    import json as _json
    expanded_cache: Dict[str, str] = {}
    query_expand_rounds: Dict[str, int] = {q: 0 for q in all_used_queries}

    max_total_rounds = max(1, len(all_used_queries)) * expand_rounds_per_query
    for round_idx in range(max_total_rounds):
        tool_calls = None
        content_buffer = ""
        usage = None
        for event in client.chat_stream(
            conversation,
            model=client.model,
            temperature=temperature,
            tools=_EXPAND_TOOL_SCHEMA,
            tool_choice="auto",
        ):
            etype = event.get("type")
            if etype == "reasoning":
                yield {"phase": "reasoning", "delta": event.get("delta", "")}
            elif etype == "content":
                delta = event.get("delta", "")
                content_buffer += delta
                yield {"phase": "content", "delta": delta}
            elif etype == "tool_calls":
                tool_calls = event.get("tool_calls", [])
            elif etype in ("finish", "done"):
                usage = event.get("usage")

        if not tool_calls:
            # 模型未调用工具，content 已流式输出完毕
            yield {"phase": "done", "usage": usage}
            return True

        conversation.append({
            "role": "assistant",
            "content": content_buffer,
            "tool_calls": tool_calls,
        })

        for tc in tool_calls:
            func = tc.get("function", {})
            fn_name = func.get("name", "")
            args_str = func.get("arguments", "{}")
            try:
                args = _json.loads(args_str) if isinstance(args_str, str) else args_str
            except _json.JSONDecodeError:
                args = {}

            if fn_name != "expand_chunks":
                continue

            ref_ids = args.get("ref_ids", []) or []
            # 兼容旧参数 chunk_ids
            if not ref_ids:
                legacy_cids = args.get("chunk_ids", []) or []
                for cid in legacy_cids:
                    mc = next((m for m in mini_chunks if m["chunk_id"] == cid), None)
                    if mc:
                        ref_ids.append(mc["index"])
            length = args.get("length", expand_max)
            length = max(1, min(int(length), expand_max))

            expanded_texts = []
            resolved_chunk_ids = []
            for rid in ref_ids:
                try:
                    idx = int(rid)
                except (TypeError, ValueError):
                    continue
                mc = next((m for m in mini_chunks if m["index"] == idx), None)
                if mc is None:
                    continue
                cid = mc["chunk_id"]
                resolved_chunk_ids.append(cid)
                if cid in expanded_cache:
                    full = expanded_cache[cid]
                else:
                    full = _load_chunk_text(base_dir, mc["library"], cid)
                    expanded_cache[cid] = full
                text = full[:length]
                expanded_texts.append({
                    "ref_id": idx,
                    "source_file": mc["source_file"],
                    "heading": mc["heading"],
                    "text": text,
                })
                for mq in mc.get("matched_queries", []):
                    if mq in query_expand_rounds:
                        query_expand_rounds[mq] += 1

            yield {
                "phase": "expanding",
                "chunk_ids": resolved_chunk_ids,
                "ref_ids": ref_ids,
                "length": length,
                "round": round_idx + 1,
            }
            yield {
                "phase": "expanded",
                "expanded_texts": expanded_texts,
                "round": round_idx + 1,
            }

            tool_result = "\n\n".join(
                f"[资料 [{et['ref_id']}]]\n"
                f"来源: {et['source_file']} · {et['heading']}\n"
                f"完整文本（前 {length} 字）:\n{et['text']}"
                for et in expanded_texts
            )
            conversation.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": tool_result or "未找到对应资料",
            })

        all_exhausted = all(
            v >= expand_rounds_per_query for v in query_expand_rounds.values()
        ) if query_expand_rounds else True
        if all_exhausted:
            conversation.append({
                "role": "user",
                "content": "展开轮数已用完，请根据已获取的资料直接给出最终答案。",
            })
    return False


def ai_search_global_vector_stream(
    question: str,
    registry: LibraryRegistry,
    client: DeepSeekClient,
    base_dir: str,
    library_names: Optional[List[str]] = None,
    parallel: int = 4,
    temperature: float = 0.3,
    extra_context: Optional[List[Dict]] = None,
    max_rounds: int = None,
    queries_per_round: int = None,
    parent_top_k: int = None,
    child_top_k: int = None,
) -> Iterator[Dict[str, Any]]:
    """global 向量模式流式检索 + 生成（多轮规划 → 大chunk→小chunk递进）。

    流程：
        1. 多轮向量检索（每轮规划3个查询词，对每个查询词调 search_semantic_progressive）
        2. 合并去重、排序、截断
        3. 构造小 chunk（用 sub_text 作为命中片段）
        4. expand_chunks 工具循环（复用精读模式逻辑）
        5. 流式生成最终答案

    事件序列：
        {"phase":"plan","round":R,"queries":[...],"should_continue":bool}
        {"phase":"searching","round":R,"query":"...","total_hits":N,"new_hits":M}
        {"phase":"retrieval","references":[...],"total_unique":N,"mini_count":M}
        {"phase":"expanding","chunk_ids":[...],"length":L,"round":R}
        {"phase":"expanded","expanded_texts":[...]}
        {"phase":"generating"}
        {"phase":"reasoning","delta":"..."}
        {"phase":"content","delta":"..."}
        {"phase":"done","usage":{...}}
        {"phase":"error","stage":"...","error":"..."}
    """
    # 局部导入向量检索函数（不修改文件顶部的现有 import）
    from searcher import search_semantic_progressive
    try:
        # 参数兜底：未传则用默认值
        if max_rounds is None:
            max_rounds = 3
        if queries_per_round is None:
            queries_per_round = 3
        if parent_top_k is None:
            parent_top_k = 20
        if child_top_k is None:
            child_top_k = 5
        max_mini_chunks = 500
        expand_rounds_per_query = 2

        # 构造库背景信息
        library_context = _build_library_context(registry, library_names)

        # ---------- 阶段 1：多轮向量检索 ----------
        raw_all_hits: List[Dict[str, Any]] = []
        prev_rounds: List[Dict[str, Any]] = []
        all_used_queries: List[str] = []

        for round_num in range(1, max_rounds + 1):
            # 规划查询词
            queries, proceed = yield from _plan_vector_round(
                client, question, library_context, round_num, max_rounds,
                queries_per_round, prev_rounds, temperature,
            )
            if not proceed or not queries:
                break

            # 对每个查询词调 search_semantic_progressive（大chunk→小chunk递进）
            round_hits: List[Dict[str, Any]] = []
            for q in queries:
                if q not in all_used_queries:
                    all_used_queries.append(q)
                try:
                    sr = search_semantic_progressive(
                        registry, q, base_dir,
                        library_names=library_names, parallel=parallel,
                        parent_top_k=parent_top_k, child_top_k=child_top_k,
                    )
                except Exception as e:
                    yield {"phase": "searching", "round": round_num, "query": q,
                           "error": f"向量检索失败: {e}"}
                    continue
                raw_hits = sr.get("results", [])
                # 标记每个命中来自哪个查询词
                for r in raw_hits:
                    r = dict(r)
                    r.setdefault("matched_queries", []).append(q)
                    round_hits.append(r)
                yield {
                    "phase": "searching",
                    "round": round_num,
                    "query": q,
                    "total_hits": sr.get("total_hits", 0),
                    "new_hits": len(raw_hits),
                    "semantic_available": sr.get("semantic_available", True),
                }

            # 记录本轮命中摘要（供下轮规划器参考）
            prev_rounds.append({
                "round": round_num,
                "queries": queries,
                "hits_summary": _summarize_round_hits(round_hits),
            })
            raw_all_hits.extend(round_hits)

        # ---------- 阶段 2：合并去重、排序、截断 ----------
        truncated, total_unique = _finalize_vector_hits(raw_all_hits, max_mini_chunks)
        yield from _warn_if_noisy(total_unique, len(truncated))

        if not truncated:
            yield {"phase": "content",
                   "delta": "未检索到与问题相关的资料，无法生成回答。"}
            yield {"phase": "done", "usage": None}
            return

        # 构造小 chunk（用 sub_text 作为命中片段）
        mini_chunks = _build_vector_mini_chunks(truncated, text_field="sub_text")

        yield {
            "phase": "retrieval",
            "queries": all_used_queries,
            "references": _build_vector_references(mini_chunks),
            "total_unique": total_unique,
            "mini_count": len(mini_chunks),
            "vector_mode": True,
        }

        # ---------- 阶段 3：expand_chunks 工具循环 ----------
        mini_text = _format_mini_chunks_for_prompt(mini_chunks)
        lib_prefix = f"【数据来源】{library_context}\n\n" if library_context else ""
        extra_ctx_text = _build_extra_ctx_text(extra_context)
        conversation: List[Dict[str, Any]] = [
            {"role": "system", "content": _DEEPREAD_SYSTEM},
            {"role": "user", "content":
                f"{lib_prefix}{extra_ctx_text}【小 chunk 列表】（共 {len(mini_chunks)} 条，向量检索命中）\n\n{mini_text}\n\n"
                f"【问题】\n{question}\n\n"
                f"请先阅读小 chunk，用 expand_chunks 工具展开你认为重要的 chunk，"
                f"然后给出最终答案。"},
        ]

        finished = yield from _run_expand_chunks_loop(
            client, conversation, mini_chunks, base_dir,
            all_used_queries, temperature,
            expand_rounds_per_query=expand_rounds_per_query,
        )
        if finished:
            return

        # ---------- 阶段 4：最终生成（流式） ----------
        yield {"phase": "generating"}
        yield from _stream_llm_answer(client, conversation, temperature)

    except DeepSeekError as e:
        yield {"phase": "error", "stage": "global_vector",
               "error": f"DeepSeek 调用失败: {e}"}
    except Exception as e:
        yield {"phase": "error", "stage": "global_vector",
               "error": f"global 向量模式执行失败: {e}"}


def ai_search_precise_vector_stream(
    question: str,
    registry: LibraryRegistry,
    client: DeepSeekClient,
    base_dir: str,
    library_names: Optional[List[str]] = None,
    parallel: int = 4,
    temperature: float = 0.3,
    extra_context: Optional[List[Dict]] = None,
    max_rounds: int = None,
    queries_per_round: int = None,
    initial_top_k: int = None,
    subchunk_parts: int = None,
    subchunk_top_k: int = None,
) -> Iterator[Dict[str, Any]]:
    """precise 向量模式流式检索 + 生成（多轮规划 → 小chunk→二次切分→二次向量）。

    流程：
        1. 多轮向量检索（每轮规划3个查询词，对每轮所有查询词调 search_semantic_precise）
        2. 合并去重、排序、截断
        3. 构造小 chunk（用 subchunk_text 作为命中片段，比 sub_text 更精确）
        4. expand_chunks 工具循环
        5. 流式生成最终答案

    事件序列与 ai_search_global_vector_stream 一致。
    """
    from searcher import search_semantic_precise
    try:
        # 参数兜底
        if max_rounds is None:
            max_rounds = 3
        if queries_per_round is None:
            queries_per_round = 3
        if initial_top_k is None:
            initial_top_k = 20
        if subchunk_parts is None:
            subchunk_parts = 10
        if subchunk_top_k is None:
            subchunk_top_k = 3
        max_mini_chunks = 500
        expand_rounds_per_query = 2

        # 构造库背景信息
        library_context = _build_library_context(registry, library_names)

        # ---------- 阶段 1：多轮向量检索 ----------
        raw_all_hits: List[Dict[str, Any]] = []
        prev_rounds: List[Dict[str, Any]] = []
        all_used_queries: List[str] = []

        for round_num in range(1, max_rounds + 1):
            # 规划查询词
            queries, proceed = yield from _plan_vector_round(
                client, question, library_context, round_num, max_rounds,
                queries_per_round, prev_rounds, temperature,
            )
            if not proceed or not queries:
                break

            for q in queries:
                if q not in all_used_queries:
                    all_used_queries.append(q)

            # 对每轮的所有查询词调 search_semantic_precise（小chunk→二次切分→二次向量）
            try:
                sr = search_semantic_precise(
                    registry, queries, base_dir,
                    library_names=library_names, parallel=parallel,
                    initial_top_k=initial_top_k,
                    subchunk_parts=subchunk_parts,
                    subchunk_top_k=subchunk_top_k,
                )
            except Exception as e:
                yield {"phase": "searching", "round": round_num,
                       "error": f"向量检索失败: {e}"}
                continue

            raw_hits = sr.get("results", [])
            raw_all_hits.extend(raw_hits)
            yield {
                "phase": "searching",
                "round": round_num,
                "queries": queries,
                "total_hits": sr.get("total_hits", 0),
                "new_hits": len(raw_hits),
                "semantic_available": sr.get("semantic_available", True),
            }

            # 记录本轮命中摘要
            prev_rounds.append({
                "round": round_num,
                "queries": queries,
                "hits_summary": _summarize_round_hits(raw_hits),
            })

        # ---------- 阶段 2：合并去重、排序、截断 ----------
        truncated, total_unique = _finalize_vector_hits(raw_all_hits, max_mini_chunks)
        yield from _warn_if_noisy(total_unique, len(truncated))

        if not truncated:
            yield {"phase": "content",
                   "delta": "未检索到与问题相关的资料，无法生成回答。"}
            yield {"phase": "done", "usage": None}
            return

        # 构造小 chunk（用 subchunk_text 作为命中片段，比 sub_text 更精确）
        mini_chunks = _build_vector_mini_chunks(truncated, text_field="subchunk_text")

        yield {
            "phase": "retrieval",
            "queries": all_used_queries,
            "references": _build_vector_references(mini_chunks),
            "total_unique": total_unique,
            "mini_count": len(mini_chunks),
            "vector_mode": True,
        }

        # ---------- 阶段 3：expand_chunks 工具循环 ----------
        mini_text = _format_mini_chunks_for_prompt(mini_chunks)
        lib_prefix = f"【数据来源】{library_context}\n\n" if library_context else ""
        extra_ctx_text = _build_extra_ctx_text(extra_context)
        conversation: List[Dict[str, Any]] = [
            {"role": "system", "content": _DEEPREAD_SYSTEM},
            {"role": "user", "content":
                f"{lib_prefix}{extra_ctx_text}【小 chunk 列表】（共 {len(mini_chunks)} 条，向量精准检索命中）\n\n{mini_text}\n\n"
                f"【问题】\n{question}\n\n"
                f"请用 expand_chunks 工具展开你认为重要的 chunk，然后给出最终答案。"},
        ]

        finished = yield from _run_expand_chunks_loop(
            client, conversation, mini_chunks, base_dir,
            all_used_queries, temperature,
            expand_rounds_per_query=expand_rounds_per_query,
        )
        if finished:
            return

        # ---------- 阶段 4：最终生成（流式） ----------
        yield {"phase": "generating"}
        yield from _stream_llm_answer(client, conversation, temperature)

    except DeepSeekError as e:
        yield {"phase": "error", "stage": "precise_vector",
               "error": f"DeepSeek 调用失败: {e}"}
    except Exception as e:
        yield {"phase": "error", "stage": "precise_vector",
               "error": f"precise 向量模式执行失败: {e}"}


def ai_search_complex_vector_stream(
    question: str,
    registry: LibraryRegistry,
    client: DeepSeekClient,
    base_dir: str,
    library_names: Optional[List[str]] = None,
    parallel: int = 4,
    temperature: float = 0.3,
    max_subquestions: int = None,
) -> Iterator[Dict[str, Any]]:
    """complex 向量模式流式检索 + 生成（拆分子问题 → 按粒度路由 → 汇总）。

    流程：
        1. 调 _split_question 拆分子问题（复用现有函数）
        2. LLM 为每个子问题选择粒度（global 或 precise）
        3. 对每个子问题调用对应的向量模式子流，拦截事件加 sub_ 前缀
        4. 合并所有子答案
        5. 流式生成最终总结

    事件序列：
        {"phase":"split","subquestions":[...],"granularities":[...]}
        {"phase":"sub_plan","index":I,"queries":[...]}
        {"phase":"sub_searching","index":I,"query":"..."}
        {"phase":"sub_retrieval","index":I,"references":[...]}
        {"phase":"sub_content","index":I,"delta":"..."}
        {"phase":"sub_done","index":I,"answer":"..."}
        {"phase":"merging","sub_count":N}
        {"phase":"retrieval","references":[...]}
        {"phase":"reasoning","delta":"..."}
        {"phase":"content","delta":"..."}
        {"phase":"done","usage":{...}}
        {"phase":"error","stage":"...","error":"..."}
    """
    try:
        # 参数兜底
        if max_subquestions is None:
            max_subquestions = 5

        # 构造库背景信息
        library_context = _build_library_context(registry, library_names)

        # ---------- 阶段 1：拆分子问题 ----------
        split_plan = _split_question(
            client, question, max_subquestions=max_subquestions,
            library_context=library_context, temperature=temperature,
        )
        sub_objs: List[Dict[str, Any]] = split_plan.get("subquestions", [{
            "question": question, "suggested_queries": [], "depends_on": [],
        }])
        subquestions_text: List[str] = [s.get("question", "") for s in sub_objs]

        # ---------- 阶段 2：LLM 为每个子问题选择粒度 ----------
        granularities = _classify_subquestion_granularity(
            client, subquestions_text, library_context, temperature=temperature,
        )

        yield {
            "phase": "split",
            "subquestions": subquestions_text,
            "subobjects": sub_objs,
            "granularities": granularities,
            "execution_mode": split_plan.get("execution_mode", "parallel"),
            "rationale": split_plan.get("rationale", ""),
            "vector_mode": True,
        }

        # ---------- 阶段 3：逐个执行子问题检索 ----------
        all_sub_answers: List[str] = []
        all_sub_refs: List[Dict[str, Any]] = []
        seen_chunk_ids: set = set()

        for idx, sub_obj in enumerate(sub_objs, 1):
            sub_q = sub_obj.get("question", question)
            granularity = granularities[idx - 1] if idx - 1 < len(granularities) else "global"

            yield {"phase": "sub_start", "index": idx, "question": sub_q,
                   "granularity": granularity}

            # 根据粒度选择子流
            if granularity == "precise":
                sub_stream = ai_search_precise_vector_stream(
                    sub_q, registry, client, base_dir,
                    library_names=library_names, parallel=parallel,
                    temperature=temperature,
                )
            else:
                sub_stream = ai_search_global_vector_stream(
                    sub_q, registry, client, base_dir,
                    library_names=library_names, parallel=parallel,
                    temperature=temperature,
                )

            sub_answer_parts: List[str] = []
            sub_refs: List[Dict[str, Any]] = []
            try:
                for ev in sub_stream:
                    ph = ev.get("phase", "")
                    if ph == "plan":
                        yield {"phase": "sub_plan", "index": idx,
                               "queries": ev.get("queries", []),
                               "round": ev.get("round", 1)}
                    elif ph == "searching":
                        yield {"phase": "sub_searching", "index": idx,
                               "query": ev.get("query", ""),
                               "round": ev.get("round", 1),
                               "total_hits": ev.get("total_hits", 0),
                               "new_hits": ev.get("new_hits", 0)}
                    elif ph == "retrieval":
                        sub_refs = ev.get("references", [])
                        yield {"phase": "sub_retrieval", "index": idx,
                               "total_unique": ev.get("total_unique", 0),
                               "mini_count": ev.get("mini_count", 0)}
                    elif ph == "expanding":
                        yield {"phase": "sub_expanding", "index": idx,
                               "chunk_ids": ev.get("chunk_ids", []),
                               "round": ev.get("round", 1)}
                    elif ph == "expanded":
                        yield {"phase": "sub_expanded", "index": idx,
                               "expanded_texts": ev.get("expanded_texts", []),
                               "round": ev.get("round", 1)}
                    elif ph == "generating":
                        yield {"phase": "sub_generating", "index": idx}
                    elif ph == "reasoning":
                        yield {"phase": "sub_reasoning", "index": idx,
                               "delta": ev.get("delta", "")}
                    elif ph == "content":
                        delta = ev.get("delta", "")
                        if delta:
                            sub_answer_parts.append(delta)
                            yield {"phase": "sub_content", "index": idx, "delta": delta}
                    elif ph == "done":
                        pass  # 子流完成，不转发
                    elif ph == "error":
                        yield {"phase": "sub_error", "index": idx,
                               "error": ev.get("error", "")}
                        sub_answer_parts.append(f"[子问题检索失败: {ev.get('error', '')}]")
            except Exception as e:
                sub_answer_parts.append(f"[子问题异常: {e}]")
                yield {"phase": "sub_error", "index": idx, "error": str(e)}

            sub_answer = "".join(sub_answer_parts) or "[无答案]"
            all_sub_answers.append(sub_answer)

            # 合并引用（去重）
            for r in sub_refs:
                cid = r.get("chunk_id", "")
                if cid and cid not in seen_chunk_ids:
                    seen_chunk_ids.add(cid)
                    r = dict(r)
                    r["from_subquestion"] = idx
                    all_sub_refs.append(r)

            yield {"phase": "sub_done", "index": idx,
                   "answer": sub_answer, "ref_count": len(sub_refs)}

        # ---------- 阶段 4：汇总 ----------
        yield {"phase": "merging",
               "sub_count": len(subquestions_text),
               "total_refs": len(all_sub_refs)}

        # 构造汇总 context：子答案 + 引用列表
        parts = []
        for i, (sub_q, ans) in enumerate(zip(subquestions_text, all_sub_answers), 1):
            parts.append(f"【子问题 {i}】{sub_q}\n\n【子回答 {i}】\n{ans}")
        sub_ctx = "\n\n---\n\n".join(parts)

        ref_list = []
        for i, r in enumerate(all_sub_refs, 1):
            sf = r.get("source_file", "")
            heading = r.get("heading", "")
            lib = r.get("library", "")
            ref_list.append(f"[{i}] {lib} · {sf}"
                            f"{' · ' + heading if heading else ''}"
                            f" (来自子问题{r.get('from_subquestion', '')})")
        ref_ctx = "\n".join(ref_list) if ref_list else "(无引用)"

        summary_messages = [
            {"role": "system", "content": _SUMMARIZER_SYSTEM},
            {"role": "user", "content":
                f"【原始问题】\n{question}\n\n"
                f"【子问题与子回答】\n{sub_ctx}\n\n"
                f"【引用来源列表】\n{ref_ctx}\n\n"
                f"请基于以上子回答汇总生成最终答案。"},
        ]

        # 重新编号引用
        merged_refs = []
        for i, r in enumerate(all_sub_refs, 1):
            r = dict(r)
            r["index"] = i
            merged_refs.append(r)

        yield {"phase": "retrieval", "references": merged_refs,
               "total_unique": len(merged_refs),
               "sub_count": len(subquestions_text),
               "vector_mode": True}

        # 流式生成汇总答案
        yield from _stream_llm_answer(client, summary_messages, temperature)

    except DeepSeekError as e:
        yield {"phase": "error", "stage": "complex_vector",
               "error": f"DeepSeek 调用失败: {e}"}
    except Exception as e:
        yield {"phase": "error", "stage": "complex_vector",
               "error": f"complex 向量模式执行失败: {e}"}


def _classify_subquestion_granularity(
    client: DeepSeekClient,
    subquestions: List[str],
    library_context: str = "",
    temperature: float = 0.3,
) -> List[str]:
    """让 LLM 为每个子问题选择向量检索粒度（global 或 precise）。

    返回与 subquestions 等长的粒度列表，失败时全部回退到 "global"。
    """
    if not subquestions:
        return []
    if len(subquestions) == 1:
        # 单个子问题：简单判断，宽泛问题用 global，细节问题用 precise
        return ["global"]

    lib_hint = f"\n\n【数据来源】{library_context}" if library_context else ""
    sub_list = "\n".join(
        f"{i}. {q}" for i, q in enumerate(subquestions, 1)
    )
    messages = [
        {"role": "system", "content": _VECTOR_GRANULARITY_SYSTEM},
        {"role": "user", "content":
            f"以下{len(subquestions)}个子问题，请为每个选择粒度：\n{sub_list}{lib_hint}"},
    ]
    try:
        resp = client.chat(messages, temperature=temperature, max_tokens=400)
        result = _extract_json(resp.get("content", "")) or {}
        granularities = result.get("granularities") or []
    except Exception:
        granularities = []

    # 清洗：只接受 global / precise，不足补 global
    cleaned: List[str] = []
    for g in granularities:
        if isinstance(g, str) and g.strip().lower() in ("global", "precise"):
            cleaned.append(g.strip().lower())
        else:
            cleaned.append("global")
    # 补齐到与 subquestions 等长
    while len(cleaned) < len(subquestions):
        cleaned.append("global")
    return cleaned[:len(subquestions)]
