"""真正的 Agent 工作流：LLM 自主调用工具完成检索+生成（通用 RAG 模式）。

架构：
  - agent_workflow_stream_async：asyncio 异步核心。LLM 流式调用与工具执行
    全部经由线程池调度，同一轮内的多个工具调用并行执行（asyncio.gather），
    派遣多个子智能体时也天然并行。
  - agent_workflow_stream：同步生成器桥接（web_api 的 SSE 通道是同步的），
    在独立线程的事件循环中驱动异步核心，通过线程安全队列逐事件转发。
  - 子智能体（subagent.py）：带独立工具循环的真子智能体，由
    dispatch_subagent 工具派遣，自主精读主智能体指定的小范围 chunk。

工具集（与资料库领域无关，领域信息由库备注动态注入）：
  - list_libraries: 获取所有库的名称、备注、chunk数量
  - list_chunk_titles: 获取指定库的 chunk 标题列表
  - get_chunk: 获取指定 chunk 的完整文本（或前N字）
  - get_neighbors: 获取指定 chunk 前后相邻 chunk 的标题和预览
  - search_titles: 只在标题/文件名中检索
  - search: 按关键词/混合词/语义检索，返回命中位置前后片段
  - dispatch_subagent: 派遣子智能体精读指定 chunk
  - report_data_issue: 报告资料库内容问题
  - finish: 完成检索，开始生成最终答案
  - list_history_refs / filter_history_chunks: 多轮对话历史 chunk 复用

轮次与上下文控制：
  - 轮次预算：接近上限时只注入一次预算提示（不反复催促）；耗尽时明确
    告知模型"预算已用完，基于已有信息作答"，而非静默截断。
  - 上下文压缩：token 超阈值时用 LLM 摘要压缩中间历史；LLM 摘要失败时
    降级为机械压缩（逐条截断摘要），确保上下文必然收缩，不会静默沿用。
"""
import asyncio
import contextlib
import json
import os
import queue
import re
import sys
import threading
import time
from typing import Any, Dict, Iterator, List, Optional

from deepseek import DeepSeekClient
from library import LibraryRegistry, Library
from searcher import (
    parallel_search,
    search_semantic_progressive,
    search_multi_keywords,
    _HAN_PHRASE_RE,
)
from ai_search import (
    _load_chunk_text,
    _build_library_context,
    _build_mini_snippets,
    _build_extra_ctx_text,
    _extract_json,
    _build_history_text,
    _format_metadata_for_prompt,
)
from settings import SettingsStore
from userdata import auth_base_dir as _auth_base_dir


# ============================================================
#  文本相似度（用于 semantic_filter 的近似语义筛选）
# ============================================================

def _text_similarity(a: str, b: str) -> float:
    """计算两段文本的相似度（0-1）。

    策略：对中文按 bigram 计算 Jaccard 系数，对英文按 word 计算 Jaccard。
    这是字面相似度的近似，不调用 embedding，速度快。
    若需真正的语义相似度，应走 search 工具的 semantic 模式。
    """
    if not a or not b:
        return 0.0

    # 中文 bigram
    def chinese_bigrams(s):
        bgs = set()
        for phrase in re.findall(r'[\u4e00-\u9fff]+', s):
            if len(phrase) >= 2:
                for i in range(len(phrase) - 1):
                    bgs.add(phrase[i:i + 2])
        return bgs

    # 英文 word（小写化）
    def english_words(s):
        return set(w.lower() for w in re.findall(r'[a-zA-Z]+', s) if len(w) >= 2)

    a_bg = chinese_bigrams(a)
    b_bg = chinese_bigrams(b)
    a_w = english_words(a)
    b_w = english_words(b)

    # 合并特征
    a_feats = a_bg | a_w
    b_feats = b_bg | b_w

    if not a_feats or not b_feats:
        return 0.0

    inter = len(a_feats & b_feats)
    union = len(a_feats | b_feats)
    return inter / union if union > 0 else 0.0


# ============================================================
#  工具定义（OpenAI function-calling schema）
# ============================================================

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_libraries",
            "description": "获取资料库列表，包含库名、备注和chunk数量。若用户已指定检索库，仅返回这些选中的库；否则返回所有库。当你需要了解可检索的资料库时调用。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_chunk_titles",
            "description": "获取指定资料库的chunk标题列表，返回每个chunk的ID和标题。用于了解库的内容结构。",
            "parameters": {
                "type": "object",
                "properties": {
                    "library": {
                        "type": "string",
                        "description": "资料库名称",
                    },
                    "keyword": {
                        "type": "string",
                        "description": "可选：只返回标题包含此关键词的chunk，不传则返回全部",
                    },
                },
                "required": ["library"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_chunk",
            "description": "获取指定chunk的完整文本。当你需要阅读某个chunk的详细内容时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "library": {
                        "type": "string",
                        "description": "资料库名称",
                    },
                    "chunk_id": {
                        "type": "string",
                        "description": "chunk的ID，格式如 zone_001/chunk_000123",
                    },
                    "length": {
                        "type": "integer",
                        "description": "返回文本的最大字符数，默认10000",
                    },
                },
                "required": ["library", "chunk_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_neighbors",
            "description": "获取指定chunk前后相邻chunk的标题和预览（默认前1后1）。当你想了解某条资料的上下文（同篇章前后内容、相邻段落）时调用。比直接 get_chunk 多个 chunk 更省 token，因为只返回标题+150字预览。",
            "parameters": {
                "type": "object",
                "properties": {
                    "library": {
                        "type": "string",
                        "description": "资料库名称",
                    },
                    "chunk_id": {
                        "type": "string",
                        "description": "chunk的ID，格式如 zone_001/chunk_000123",
                    },
                    "window": {
                        "type": "integer",
                        "description": "前后各取多少个相邻chunk，默认1（前1后1共3个），最大5",
                    },
                },
                "required": ["library", "chunk_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_titles",
            "description": "只在 chunk 的标题(heading)和 chunk 文件名中检索关键词，返回标题命中的 chunk 列表（不检索正文）。适合快速定位章节、卷次、篇目标题，比 search 更轻量。当你要找的是某个章节/卷/篇的标题而非正文内容时优先使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "标题检索关键词（支持中文连续字串匹配，如书名、篇章名等连续文本）",
                    },
                    "libraries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要检索的资料库名称列表，不传则检索全部库",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量上限，默认30",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
                    "description": "在资料库正文内容中检索，返回命中的chunk及其片段。支持关键词检索、多关键词共现检索和语义向量检索三种模式。建议配合 title_filter 按来源/分类限定范围，避免同名或同类内容的干扰，大幅提高精确率。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索查询词",
                    },
                    "libraries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要检索的资料库名称列表，不传则检索全部库",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["keyword", "multi_keyword", "semantic"],
                        "description": "检索模式：keyword=单关键词检索，multi_keyword=多关键词共现检索（需提供queries），semantic=语义向量检索",
                    },
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "多关键词共现检索时的关键词列表（仅mode=multi_keyword时使用）",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量上限，默认20",
                    },
                    "title_filter": {
                        "type": "object",
                        "description": "标题筛选器：只保留标题（heading）满足条件的chunk。不传则不筛选标题。多个条件之间默认为AND（同时满足）。",
                        "properties": {
                            "contains": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "标题必须包含这些关键词中的至少一个（OR 关系）。例：['水利篇','第一章'] 表示标题含'水利篇'或'第一章'",
                            },
                            "contains_all": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "标题必须同时包含所有这些关键词（AND 关系）。例：['年报','工业'] 表示标题既要含'年报'又要含'工业'",
                            },
                            "excludes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "标题不能包含这些关键词。例：['附录','索引'] 表示排除含'附录'或'索引'的标题",
                            },
                            "regex": {
                                "type": "string",
                                "description": "标题正则匹配（Python re语法，忽略大小写）。例：'第[一二三四五]章' 匹配'第一章'、'第二章'等",
                            },
                        },
                    },
                    "semantic_filter": {
                        "type": "object",
                        "description": "语义筛选器：对初步检索结果用向量相似度做二次筛选，保留与给定文本语义相近的chunk。不传则不筛选。注意：此筛选会调用向量索引，比title_filter慢。",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "语义参考文本，筛选出与此文本语义相近的chunk。例：'某项政策的演变过程' 会保留语义接近该描述的chunk",
                            },
                            "min_score": {
                                "type": "number",
                                "description": "最低语义相似度阈值（0-1），默认0.3。越高越严格，0.5以上为强相关",
                            },
                            "top_n": {
                                "type": "integer",
                                "description": "语义筛选后保留的最大数量，默认10",
                            },
                        },
                    },
                    "tag_filter": {
                        "type": "object",
                        "description": "标签筛选器：按 chunk 的 tags 字段筛选结果（标签为从正文提取的人物/地名/机构/专有名词）。不传则不筛选。比 title_filter 更细粒度，因为 tags 是从正文提取的，能反映 chunk 的实际内容主题。",
                        "properties": {
                            "contains_any": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "chunk 的 tags 中含任一标签即保留（OR 关系）。例：['灌溉','水库'] 保留 tags 中含'灌溉'或'水库'的 chunk",
                            },
                            "contains_all": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "chunk 的 tags 必须同时含所有这些标签才保留（AND 关系）。例：['水利','投资'] 保留同时含'水利'和'投资'的 chunk",
                            },
                            "excludes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "chunk 的 tags 不能含这些标签。例：['草案','征求意见稿'] 排除 tags 中含'草案'或'征求意见稿'的 chunk",
                            },
                        },
                    },
                    "chunk_filter": {
                        "type": "object",
                        "description": "chunk 范围筛选器：把检索范围限定在指定的 chunk 列表或区间内（按 chunk 序号）。不传则检索全库。适合「在已知 chunk 范围内细化检索」的场景，比如 list_chunk_titles 已定位到某篇章后，只想在该篇章内检索。注意：此筛选与 title_filter 可叠加（AND 关系）。",
                        "properties": {
                            "library": {
                                "type": "string",
                                "description": "目标资料库名（必填，因为 chunk 序号是库内维度）",
                            },
                            "chunks": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "显式 chunk_id 列表（OR 关系，只保留这些 chunk）。例：['zone_001/chunk_000123','zone_001/chunk_000124']",
                            },
                            "ranges": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "chunk 序号区间列表（闭区间，按 chunk 文件名尾号解析）。支持 '001-399'、'100-200'、'001,005,010'（逗号列表）三种形式。例：['001-399'] 表示 chunk_000001 到 chunk_000399；['001,005,010'] 表示仅这 3 个 chunk。区间基于 zone_001。",
                            },
                        },
                    },
                },
                "required": ["query", "mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dispatch_subagent",
            "description": "派遣子智能体处理一组 chunk。子智能体是独立运行的智能体：它会拿到你指定的 chunk 清单（ID+标题+预览），自主决定阅读哪些 chunk、读多长、是否在范围内检索关键词，完成子任务后提交回答。适合：1) 主智能体已定位到小范围 chunk（如某篇章/章节），让子智能体精读并提取特定信息；2) 把多个相关 chunk 打包给子智能体做综合分析。注意：chunk_ids 必须是你已经通过 search/list_chunk_titles/get_chunk 等工具定位到的具体 chunk，不要凭空猜测。每次调用会启动一个独立的子智能体（拥有独立上下文和轮次预算），建议单次分配 1-5 个 chunk；同轮派遣多个子智能体可并行执行。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subtask": {
                        "type": "string",
                        "description": "给子智能体的子任务描述。必须明确指出要从 chunk 中提取什么信息。例：'从以下 chunk 中找出某事件的具体时间、地点和相关人物' / '总结以下 chunk 中关于某主题的关键信息'",
                    },
                    "chunks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "library": {"type": "string", "description": "chunk 所在的资料库名"},
                                "chunk_id": {"type": "string", "description": "chunk 的 ID，格式如 zone_001/chunk_000123"},
                            },
                            "required": ["library", "chunk_id"],
                        },
                        "description": "分配给子智能体的 chunk 列表，每个元素含 library 和 chunk_id。建议 1-5 个 chunk。",
                    },
                    "context_hint": {
                        "type": "string",
                        "description": "可选：给子智能体的额外上下文提示。例：'用户原始问题是 X，这部分 chunk 可能涉及 Y'，帮助子智能体聚焦",
                    },
                },
                "required": ["subtask", "chunks"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "report_data_issue",
            "description": "向质询报告页面报告资料库内容中存在的问题。当你在检索或阅读 chunk 时发现数据冲突、常识错误、前后矛盾、内容缺失等问题时调用此工具。报告会保存到质询报告页面供后续查看。仅在确实发现资料库内容问题时调用，不要用于报告你自己回答的不足。",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue_type": {
                        "type": "string",
                        "enum": ["数据冲突", "常识错误", "前后矛盾", "重要遗漏", "其他"],
                        "description": "问题类型：数据冲突=多来源对同一事物描述矛盾；常识错误=与公认事实不符；前后矛盾=同一文档内部自相矛盾；重要遗漏=关键信息缺失；其他=不属于上述类型的问题",
                    },
                    "description": {
                        "type": "string",
                        "description": "问题的详细描述，包括具体的位置、内容、为什么是问题。例：'某卷记载某事件发生于1998年，但同库另一处记载为1997年，存在年份矛盾'",
                    },
                    "chunk_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "相关 chunk 的 ID 列表（如 zone_001/chunk_000123），便于后续定位",
                    },
                    "library": {
                        "type": "string",
                        "description": "问题所在的资料库名称",
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["info", "warning", "error"],
                        "description": "严重程度：info=提示性（不影响检索结果）；warning=警告（可能影响准确性）；error=严重错误（明显数据错误）",
                    },
                },
                "required": ["issue_type", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "完成检索，开始生成最终答案。当你已经收集到足够信息时调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_history_refs",
            "description": "列出本会话之前轮次问答中引用过的 chunk 列表。当你判断新问题可能与之前讨论的话题相关、想复用历史 chunk 时调用。返回每个 chunk 的 chunk_id、library、heading 和所属问答轮次。注意：历史 chunk 仅作参考，新问题应优先独立检索，仅在确认与历史相关时复用。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_history_chunks",
            "description": "在历史引用过的 chunk 范围内做关键词检索，返回命中的 chunk 片段。适合新问题与历史话题相关、想在历史 chunk 中细化定位的场景。注意：此工具只在历史 chunk 范围内检索，不能替代全库检索；若历史 chunk 无法回答新问题，仍应调用 search 做全库检索。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索关键词",
                    },
                    "chunk_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要检索的历史 chunk_id 列表（可从 list_history_refs 获取）。不传则检索所有历史 chunk",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量上限，默认10",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# ============================================================
#  System Prompt（通用 RAG，不绑定具体资料领域；
#  领域信息由库备注通过【数据来源】动态注入）
# ============================================================

_AGENT_SYSTEM = """你是一个检索增强问答（RAG）智能体，通过调用工具在用户的资料库中查找信息并回答问题。

【工作流程】
1. 理解问题：明确要找什么信息、涉及哪些实体或主题
2. 了解资料范围：list_libraries 查看资料库（备注中通常说明资料领域）；list_chunk_titles / search_titles 了解库的内容结构与分类
3. 检索：用 search 在正文中检索（keyword=精确匹配专有名词；multi_keyword=多词共现；semantic=语义相似但表述不同）
   - chunk 标题(heading)通常含来源/分类/篇目信息，search 配合 title_filter 限定范围可避免同名或同类内容干扰，大幅提高精确率
   - 不确定分类时先 search_titles 定位相关篇章，再带 title_filter 检索正文
4. 精读：检索结果只是命中片段，需要完整内容时用 get_chunk 展开；想看前后文用 get_neighbors
   已定位到一小批 chunk 且需要提取复杂信息时，用 dispatch_subagent 派子智能体精读
5. 评估：信息是否足以回答问题？不足则换查询词、调整筛选条件继续检索
6. 完成：信息足够后调用 finish，进入最终答案生成

【多轮对话】本会话之前的问答历史已注入上下文，含历史引用过的 chunk：
- 新问题与历史相关时，可用 list_history_refs / filter_history_chunks 复用历史 chunk
- 新问题应优先独立检索（search）；历史 chunk 仅作参考，不要因为历史有 chunk 就不做新检索

【数据问题报告】检索中发现资料存在数据冲突/常识错误/前后矛盾/重要遗漏时，调用 report_data_issue 报告。仅报告资料本身的问题，不报告你自己回答的不足。

【工具调用规则】
- 必须通过 function-calling 通道调用工具（tool_calls 字段），不要用文本标签、伪代码或自然语言描述工具调用
- 每一轮只调用一个工具；不要输出任何叙述性文字（如"好的，我先查看…"），你的文字输出不会展示给用户
- 通常 2-4 轮检索即可获得足够信息，不要过度检索
- 信息已充分时直接调用 finish，不要解释
- 最终答案只在 finish 之后的生成阶段输出；引用资料时用 [n]（资料编号）标注"""


# ============================================================
#  工具执行器
# ============================================================

def _build_ref_hit_snippet(chunk_text: str, matched_words: List[str],
                           window: int = 200) -> str:
    """构造"命中点 ±window 字"的引用片段。

    在 chunk 文本中定位首个命中词，取命中点前后各 window 字作为片段，
    越界处用省略号标注，供引用列表展示更细粒度的来源内容。
    找不到命中词时返回空字符串（由调用方回退）。
    """
    if not chunk_text or not matched_words:
        return ""
    # window<=0 时无法构造有效片段，交由调用方回退逻辑兜底
    if window <= 0:
        return ""
    positions = []
    for w in matched_words:
        if not w:
            continue
        idx = chunk_text.find(w)
        if idx >= 0:
            positions.append(idx)
    if not positions:
        return ""
    pos = min(positions)
    start = max(0, pos - window)
    end = min(len(chunk_text), pos + window)
    snippet = chunk_text[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(chunk_text):
        snippet = snippet + "..."
    return snippet


class ToolExecutor:
    """执行 Agent 工具调用，返回结果文本。"""

    def __init__(self, registry: LibraryRegistry, base_dir: str,
                 library_names: Optional[List[str]] = None,
                 question: str = "",
                 client: Optional[Any] = None,
                 temperature: float = 0.3,
                 history: Optional[List[Dict[str, Any]]] = None):
        self.registry = registry
        self.base_dir = base_dir
        self.library_names = set(library_names) if library_names else None
        self.question = question
        self.client = client
        self.temperature = temperature
        self._chunk_ids_cache: Dict[str, List[str]] = {}
        self._tags_cache: Dict[str, List[str]] = {}
        # 并行工具执行时的共享状态锁（accessed/discovered/reported/subagent 记录）
        self._state_lock = threading.RLock()
        # 收集本次会话中报告的数据问题，供前端展示
        self.reported_issues: List[Dict[str, Any]] = []
        # 收集本次会话中派遣的子智能体执行记录，供前端展示
        self.subagent_records: List[Dict[str, Any]] = []
        # 追踪模型实际访问过的 chunk（通过 get_chunk/get_neighbors/dispatch_subagent）
        # 用于生成引用来源列表（只包含实际使用过的 chunk，而非全部检索命中）
        self.accessed_chunks: List[Dict[str, Any]] = []
        self._accessed_ids: set = set()
        # 追踪所有检索/浏览过的 chunk（通过 search/search_titles/get_chunk/get_neighbors）
        # 用于 finish 时汇总给模型参考，避免多轮检索后模型遗忘早期发现的 chunk
        # 与 accessed_chunks 的区别：discovered_chunks 包含检索命中但未展开的 chunk
        self.discovered_chunks: List[Dict[str, Any]] = []
        self._discovered_ids: set = set()
        # 本会话之前轮次的历史引用 chunk（从 history 提取，供 list_history_refs/filter_history_chunks 工具使用）
        # 格式：[{"chunk_id":..., "library":..., "heading":..., "source_file":..., "round": N}, ...]
        self.history_refs: List[Dict[str, Any]] = []
        if history:
            self._extract_history_refs(history)

    def _extract_history_refs(self, history: List[Dict[str, Any]]):
        """从精简历史中提取每轮 assistant 消息的 references，扁平化为 history_refs。"""
        round_idx = 0
        for m in history:
            role = m.get("role", "")
            if role == "user":
                round_idx += 1
            elif role == "assistant":
                refs = m.get("references", []) or []
                for r in refs:
                    cid = r.get("chunk_id", "")
                    lib = r.get("library", "")
                    if not cid or not lib:
                        continue
                    # 按 chunk_id 去重，保留最高轮次
                    existing = next((h for h in self.history_refs if h["chunk_id"] == cid and h["library"] == lib), None)
                    if existing:
                        existing["round"] = round_idx
                    else:
                        self.history_refs.append({
                            "chunk_id": cid,
                            "library": lib,
                            "heading": r.get("heading", ""),
                            "source_file": r.get("source_file", ""),
                            "round": round_idx,
                        })

    def execute(self, fn_name: str, args: Dict[str, Any]) -> str:
        """执行工具，返回结果文本。"""
        try:
            if fn_name == "list_libraries":
                return self._list_libraries()
            elif fn_name == "report_data_issue":
                return self._report_data_issue(
                    args.get("issue_type", "其他"),
                    args.get("description", ""),
                    chunk_ids=args.get("chunk_ids") or [],
                    library=args.get("library", ""),
                    severity=args.get("severity", "info"),
                )
            elif fn_name == "list_chunk_titles":
                return self._list_chunk_titles(
                    args.get("library", ""),
                    keyword=args.get("keyword"),
                )
            elif fn_name == "get_chunk":
                return self._get_chunk(
                    args.get("library", ""),
                    args.get("chunk_id", ""),
                    length=args.get("length", 10000),
                )
            elif fn_name == "get_neighbors":
                return self._get_neighbors(
                    args.get("library", ""),
                    args.get("chunk_id", ""),
                    window=min(int(args.get("window", 1) or 1), 5),
                )
            elif fn_name == "search_titles":
                return self._search_titles(
                    args.get("query", ""),
                    libraries=args.get("libraries"),
                    top_k=args.get("top_k", 30),
                )
            elif fn_name == "search":
                return self._search(
                    args.get("query", ""),
                    mode=args.get("mode", "keyword"),
                    libraries=args.get("libraries"),
                    queries=args.get("queries"),
                    top_k=args.get("top_k", 20),
                    title_filter=args.get("title_filter"),
                    semantic_filter=args.get("semantic_filter"),
                    tag_filter=args.get("tag_filter"),
                    chunk_filter=args.get("chunk_filter"),
                )
            elif fn_name == "dispatch_subagent":
                return self._dispatch_subagent(
                    args.get("subtask", ""),
                    args.get("chunks") or [],
                    context_hint=args.get("context_hint", ""),
                )
            elif fn_name == "finish":
                # 汇总已发现的 chunk 列表，帮助模型在最终生成时回顾所有检索到的资料
                # 避免多轮检索后模型遗忘早期发现的 chunk
                summary = self._build_discovered_summary()
                if summary:
                    return (
                        "检索完成，请根据已获取的信息生成最终答案。\n\n"
                        f"{summary}\n\n"
                        "以上是本次检索过程中发现的所有 chunk 汇总。"
                        "其中已通过 get_chunk/get_neighbors/dispatch_subagent 阅读过的 chunk 内容可直接引用，"
                        "未展开的 chunk 如需引用可再次调用 get_chunk 查看。"
                    )
                return "检索完成，请根据已获取的信息生成最终答案。"
            elif fn_name == "list_history_refs":
                return self._list_history_refs()
            elif fn_name == "filter_history_chunks":
                return self._filter_history_chunks(
                    args.get("query", ""),
                    chunk_ids=args.get("chunk_ids"),
                    top_k=args.get("top_k", 10),
                )
            else:
                return f"未知工具: {fn_name}"
        except Exception as e:
            return f"工具执行错误: {e}"

    def _list_history_refs(self) -> str:
        """列出本会话之前轮次引用过的 chunk，供模型判断是否复用。"""
        if not self.history_refs:
            return "本会话之前没有引用过任何 chunk，无需复用历史。"
        lines = [f"【历史引用 chunk 列表】（共 {len(self.history_refs)} 条，来自之前 {max(h['round'] for h in self.history_refs)} 轮问答）"]
        for i, h in enumerate(self.history_refs, 1):
            heading = h.get("heading", "")
            sf = h.get("source_file", "")
            disp = f" · {heading}" if heading else ""
            if sf:
                disp += f"（来源: {os.path.basename(sf)}）"
            lines.append(f"{i}. chunk_id={h['chunk_id']} | 库={h['library']}{disp} | 第{h['round']}轮")
        lines.append("")
        lines.append("提示：如需在历史 chunk 范围内细化检索，可调用 filter_history_chunks。")
        lines.append("如需查看某个历史 chunk 的完整内容，可直接调用 get_chunk。")
        return "\n".join(lines)

    def _filter_history_chunks(self, query: str,
                                chunk_ids: Optional[List[str]] = None,
                                top_k: int = 10) -> str:
        """在历史引用的 chunk 范围内做关键词检索，返回命中片段。

        复用现有 search 工具的 chunk_filter 机制，把历史 chunk 作为检索范围。
        """
        if not self.history_refs:
            return "本会话之前没有引用过任何 chunk，无法在历史范围内检索。请直接使用 search 工具做全库检索。"
        if not query or not query.strip():
            return "查询词不能为空。"

        # 确定检索范围：指定 chunk_ids 时只取这些，否则取全部历史 chunk
        if chunk_ids:
            target = [h for h in self.history_refs if h["chunk_id"] in set(chunk_ids)]
        else:
            target = list(self.history_refs)
        if not target:
            return "指定的 chunk_id 都不在历史引用中，请检查 chunk_id 或调用 list_history_refs 查看可用列表。"

        # 按库分组构造 chunk_filter，对每个库调用一次 search
        lib_chunks: Dict[str, List[str]] = {}
        for h in target:
            lib = h["library"]
            lib_chunks.setdefault(lib, []).append(h["chunk_id"])

        all_results = []
        for lib, cids in lib_chunks.items():
            chunk_filter = {"library": lib, "chunks": cids}
            try:
                sr = parallel_search(
                    self.registry, query,
                    libraries=[lib],
                    top_k=top_k,
                    chunk_filter=chunk_filter,
                )
                all_results.extend(sr.get("results", []))
            except Exception as e:
                continue

        if not all_results:
            return f"在历史 chunk 范围内未检索到与 '{query}' 相关的内容。请尝试使用 search 工具做全库检索。"

        # 去重 + 排序 + 截断
        seen = set()
        unique = []
        for h in all_results:
            cid = h.get("chunk_id", "")
            if cid in seen:
                continue
            seen.add(cid)
            unique.append(h)
        unique.sort(key=lambda x: x.get("score", 0), reverse=True)
        unique = unique[:top_k]

        lines = [f"【历史 chunk 范围内检索结果】（查询: '{query}'，命中 {len(unique)} 条）"]
        for i, h in enumerate(unique, 1):
            cid = h.get("chunk_id", "")
            lib = h.get("library", "")
            heading = h.get("heading", "")
            snippet = h.get("snippet", "")[:200]
            score = h.get("score", 0)
            disp_heading = f" · {heading}" if heading else ""
            lines.append(f"{i}. [{lib}]{disp_heading} (score={score})")
            lines.append(f"   chunk_id: {cid}")
            lines.append(f"   片段: {snippet}")
        lines.append("")
        lines.append("提示：如需查看完整内容可调用 get_chunk；如历史范围不足，请使用 search 做全库检索。")
        return "\n".join(lines)

    def _list_libraries(self) -> str:
        """获取库列表。

        若用户指定了检索库（library_names），只返回这些库并标注为"用户选中"。
        否则返回所有库。
        """
        all_libs = self.registry.list_libraries()
        if not all_libs:
            return "没有可用的资料库。"

        # 若用户选定了库，只列出选中的库
        if self.library_names:
            selected = [l for l in all_libs if l.name in self.library_names]
            if not selected:
                return f"用户选中的资料库（{', '.join(sorted(self.library_names))}）均不存在或未加载。"
            lines = ["用户选中的资料库（仅可检索这些库）："]
            for lib in selected:
                try:
                    chunk_ids = self._get_chunk_ids(lib.name)
                    chunk_count = len(chunk_ids)
                except Exception:
                    chunk_count = -1
                note = f"（备注：{lib.note}）" if lib.note else ""
                lines.append(f"- {lib.name}{note} [chunk数: {chunk_count}]")
            return "\n".join(lines)

        # 未指定库：列出全部
        lines = ["资料库列表："]
        for lib in all_libs:
            try:
                chunk_ids = self._get_chunk_ids(lib.name)
                chunk_count = len(chunk_ids)
            except Exception:
                chunk_count = -1
            note = f"（备注：{lib.note}）" if lib.note else ""
            lines.append(f"- {lib.name}{note} [chunk数: {chunk_count}]")
        return "\n".join(lines)

    def _list_chunk_titles(self, library_name: str,
                           keyword: Optional[str] = None) -> str:
        """获取指定库的 chunk 标题列表。"""
        if not library_name:
            return "参数错误：library 不能为空。请先调用 list_libraries 查看可用的库名。"
        lib = self.registry.get_library(library_name)
        if lib is None:
            return f"库不存在: {library_name}"

        chunk_ids = self._get_chunk_ids(library_name)
        if not chunk_ids:
            return f"库 {library_name} 中没有 chunk。"

        # 读取 heading
        mgr = lib.manager(self.base_dir)
        lines = [f"库 {library_name} 的 chunk 标题列表（共{len(chunk_ids)}条）："]
        count = 0
        for cid in chunk_ids:
            try:
                parts = cid.split("/")
                if len(parts) != 2:
                    continue
                zone_id, chunk_name = parts
                zone = mgr.get_zone(zone_id)
                chunk = zone.read_chunk(int(chunk_name.split("_")[1]))
                heading = chunk.get("heading", "")
                # 关键词过滤
                if keyword and keyword not in heading:
                    continue
                lines.append(f"- {cid} | {heading}")
                count += 1
                # 限制返回数量避免过长
                if count >= 100:
                    lines.append(f"...（还有更多，请用 keyword 参数缩小范围）")
                    break
            except Exception:
                continue
        if count == 0:
            if keyword:
                return f"库 {library_name} 中没有标题包含'{keyword}'的chunk。"
            return f"库 {library_name} 中没有可读取的 chunk。"
        lines.append("")
        lines.append("提示：标题中的来源/分类/篇目信息可作为 search 工具 title_filter 的筛选条件，提高检索精确率。")
        return "\n".join(lines)

    def _record_accessed_chunk(self, library_name: str, chunk_id: str,
                               snippet: str = "",
                               matched_words: Optional[List[str]] = None,
                               window: int = 200):
        """记录模型实际访问过的 chunk（用于生成引用来源列表）。

        只记录第一次访问，避免重复。引用来源只包含模型真正读取过内容的 chunk，
        而非全部检索命中的 chunk。

        snippet 参数可选，传入 chunk 文本的前 N 字作为预览片段，
        让前端引用列表能显示具体内容而非仅定位到 chunk 级别。

        matched_words / window：命中词与"命中点前后 window 字"片段窗口——
        若提供命中词，则 snippet 会用"命中点 ±window 字"的小片段替换，
        让引用列表颗粒度细化到命中位置附近，而非整块开头。
        未提供 matched_words 时，会反查已记录的 discovered_chunks 补充。
        """
        if not library_name or not chunk_id:
            return
        key = f"{library_name}/{chunk_id}"
        with self._state_lock:
            if key in self._accessed_ids:
                return
            self._accessed_ids.add(key)
        # 读取 chunk 元数据
        heading = ""
        source_file = ""
        text_offset = 0
        text_length = 0
        hit_snippet = ""
        # 命中词：优先用调用方传入，否则反查已发现的 chunk
        if not matched_words:
            for d in self.discovered_chunks:
                if d.get("library") == library_name and d.get("chunk_id") == chunk_id:
                    matched_words = d.get("matched_words") or []
                    if not hit_snippet:
                        hit_snippet = d.get("snippet", "") or ""
                    break
        try:
            lib = self.registry.get_library(library_name)
            if lib:
                mgr = lib.manager(self.base_dir)
                parts = chunk_id.split("/")
                if len(parts) == 2:
                    zone_id, chunk_name = parts
                    zone = mgr.get_zone(zone_id)
                    chunk = zone.read_chunk(int(chunk_name.split("_")[1]))
                    heading = chunk.get("heading", "") or ""
                    src = chunk.get("source", {}) or {}
                    source_file = src.get("file_name", "") or ""
                    text_offset = chunk.get("text_offset", 0) or 0
                    text_length = chunk.get("text_length", 0) or 0
                    chunk_text = chunk.get("text", "") or ""
                    # 未显式传 snippet 时，优先构造"命中点 ±window 字"小片段
                    if (not snippet) and matched_words and chunk_text:
                        snippet = _build_ref_hit_snippet(chunk_text, matched_words, window)
                    # 定位失败（如仅标题命中）时，优先用检索时记录的命中片段，
                    # 最后才回退 chunk 开头（可能与问题无关）
                    if not snippet and hit_snippet:
                        snippet = hit_snippet
                    if (not snippet) and chunk_text:
                        snippet = chunk_text[:200]
        except Exception:
            pass
        # index 计算 + append 必须在锁内，避免并行工具调用（dispatch_subagent / DSML）
        # 同时进入时读到相同 len 导致引用编号 [N] 重复
        with self._state_lock:
            self.accessed_chunks.append({
                "index": len(self.accessed_chunks) + 1,
                "library": library_name,
                "chunk_id": chunk_id,
                "source_file": source_file,
                "heading": heading,
                "hit_count": 1,
                "matched_words": matched_words or [],
                "snippet": snippet[:400] if snippet else "",
                "text_offset": text_offset,
                "text_length": text_length,
            })
        # 同步记录到 discovered_chunks（访问过的 chunk 也属于"已发现"）
        self._record_discovered_one(library_name, chunk_id, heading, source_file,
                                    matched_words=matched_words or [],
                                    snippet=snippet)

    def _record_discovered_one(
        self, library_name: str, chunk_id: str,
        heading: str = "", source_file: str = "",
        matched_words: Optional[List[str]] = None,
        snippet: str = "", score: Any = 0,
    ):
        """记录单个已发现的 chunk 到 discovered_chunks（去重）。

        用于 finish 时汇总给模型参考，避免多轮检索后模型遗忘早期发现的 chunk。
        与 accessed_chunks 的区别：discovered_chunks 包含检索命中但未展开的 chunk。
        """
        if not library_name or not chunk_id:
            return
        key = f"{library_name}/{chunk_id}"
        with self._state_lock:
            if key in self._discovered_ids:
                return
            self._discovered_ids.add(key)
            self.discovered_chunks.append({
                "chunk_id": chunk_id,
                "library": library_name,
                "heading": heading or "",
                "source_file": source_file or "",
                "matched_words": matched_words or [],
                "snippet": (snippet or "")[:200],
                "score": score,
            })

    def _record_discovered_chunks(self, results: List[Dict[str, Any]]):
        """批量记录检索结果到 discovered_chunks（去重）。

        用于 search / search_titles 工具调用后，把命中的 chunk 列表
        （包括命中但未展开的）记录下来，供 finish 时汇总给模型参考。
        """
        for h in results:
            cid = h.get("chunk_id", "")
            lib_name = h.get("library", "")
            if not cid or not lib_name:
                continue
            self._record_discovered_one(
                lib_name, cid,
                heading=h.get("heading", ""),
                source_file=h.get("source_file", ""),
                matched_words=h.get("matched_words", []),
                snippet=h.get("snippet", "") or h.get("sub_text", ""),
                score=h.get("score", 0),
            )

    def _build_discovered_summary(self, max_items: int = 50) -> str:
        """构建已发现 chunk 的汇总文本，供 finish 时返回给模型参考。

        格式：
            【已发现的 chunk 汇总】（共 X 条，显示前 Y 条）
            1. chunk_id=... | 库=... | 标题=... | 匹配词=[...]
            ...

        避免多轮检索后模型遗忘早期发现的 chunk。
        若 discovered_chunks 为空，返回空字符串。

        注意：对于已通过 get_chunk 访问过的 chunk，会标注其资料编号 [N]，
        提醒模型在最终答案中用 [N] 引用而非 chunk_id 中的数字。
        """
        if not self.discovered_chunks:
            return ""
        # 构建 chunk_key -> ref_idx 映射
        accessed_map = {}
        for c in self.accessed_chunks:
            key = f"{c['library']}/{c['chunk_id']}"
            accessed_map[key] = c.get("index")

        total = len(self.discovered_chunks)
        shown = min(total, max_items)
        lines = [f"【已发现的 chunk 汇总】（共 {total} 条，显示前 {shown} 条）"]
        for i, c in enumerate(self.discovered_chunks[:max_items], 1):
            cid = c.get('chunk_id', '')
            lib = c.get('library', '')
            key = f"{lib}/{cid}"
            ref_idx = accessed_map.get(key)
            if ref_idx is not None:
                parts = [f"{i}. [资料编号 {ref_idx}] chunk_id={cid} | 库={lib}（已展开阅读）"]
            else:
                parts = [f"{i}. chunk_id={cid} | 库={lib}"]
            heading = c.get("heading", "")
            if heading:
                parts.append(f"标题={heading}")
            matched = c.get("matched_words", [])
            if matched:
                parts.append(f"匹配词={matched}")
            snippet = c.get("snippet", "")
            if snippet:
                parts.append(f"片段={snippet[:80]}")
            lines.append(" | ".join(parts))
        if total > max_items:
            lines.append(f"...（还有 {total - max_items} 条未显示，可通过 get_chunk 查看具体内容）")
        lines.append("")
        lines.append("提示：最终答案中引用资料时，请使用已展开阅读的 [资料编号 N]，不要使用 chunk_id 中的数字。")
        return "\n".join(lines)

    def _get_chunk(self, library_name: str, chunk_id: str,
                   length: int = 10000) -> str:
        """获取 chunk 的完整文本。"""
        if not library_name or not chunk_id:
            return ("参数错误：library 和 chunk_id 不能为空。"
                    "请先通过 search / search_titles / list_chunk_titles 等工具获取有效的 chunk_id，"
                    "格式如 'zone_001/chunk_000123'，再调用 get_chunk。")
        text = _load_chunk_text(self.base_dir, library_name, chunk_id)
        if not text:
            return f"无法加载 chunk: {chunk_id}（库: {library_name}）"

        # 记录模型实际访问的 chunk（用于引用来源）
        # 不传 snippet：让 _record_accessed_chunk 反查检索时的命中词，
        # 自动构造"命中点 ±window 字"的片段，引用列表颗粒度更细
        self._record_accessed_chunk(library_name, chunk_id)

        # 获取分配的资料编号（与最终引用列表的 index 一致）
        ref_idx = None
        key = f"{library_name}/{chunk_id}"
        for c in self.accessed_chunks:
            if f"{c['library']}/{c['chunk_id']}" == key:
                ref_idx = c.get("index")
                break

        length = max(1, min(int(length), 20000))
        if len(text) > length:
            text = text[:length] + "\n...(文本已截断)"
        # 返回格式同时包含资料编号和 chunk_id
        # 资料编号 [N] 是引用时使用的编号，chunk_id 仅供再次调用工具时使用
        if ref_idx is not None:
            return f"[资料编号 {ref_idx} | chunk_id: {chunk_id} | 库: {library_name}]\n{text}"
        return f"[chunk_id: {chunk_id} | 库: {library_name}]\n{text}"

    def _search_titles(self, query: str,
                       libraries: Optional[List[str]] = None,
                       top_k: int = 30) -> str:
        """只在 chunk 标题(heading)和文件名中检索关键词。

        遍历指定库的所有 chunk，读取 heading 和 source_file，
        做中文连续字串匹配（query 整串出现即命中）。
        不读取正文，速度快，适合定位章节/卷/篇标题。
        """
        if not query:
            return "查询词不能为空"

        all_libs = self.registry.list_libraries()
        if libraries:
            libs = [l for l in all_libs if l.name in libraries]
        else:
            libs = all_libs

        if not libs:
            return "没有可检索的资料库。"

        # 提取查询中的中文连续字串作为匹配词
        # 如 "库A 水利篇" → ["库A", "水利篇"]
        match_words = [w for w in _HAN_PHRASE_RE.findall(query) if w]
        if not match_words:
            # 非中文（如英文/数字），按空格切分
            match_words = [w for w in query.split() if w]

        hits = []  # [(lib_name, cid, heading, source_file, matched_words, score)]
        for lib in libs:
            try:
                chunk_ids = self._get_chunk_ids(lib.name)
            except Exception:
                continue
            mgr = lib.manager(self.base_dir)
            for cid in chunk_ids:
                try:
                    parts = cid.split("/")
                    if len(parts) != 2:
                        continue
                    zone_id, chunk_name = parts
                    zone = mgr.get_zone(zone_id)
                    chunk = zone.read_chunk(int(chunk_name.split("_")[1]))
                    heading = chunk.get("heading", "") or ""
                    src = chunk.get("source", {}) or {}
                    source_file = src.get("file_name", "") or ""

                    # 标题+文件名拼接做匹配
                    haystack = f"{heading}\t{source_file}"
                    matched = [w for w in match_words if w in haystack]
                    if not matched:
                        continue
                    # 评分：连续匹配越多越长，分越高（与正文检索一致的 bonus 策略）
                    score = sum(haystack.count(w) * 100 for w in matched)
                    hits.append((lib.name, cid, heading, source_file,
                                 matched, score))
                except Exception:
                    continue

        if not hits:
            return f"标题中未检索到包含'{query}'的chunk。"

        # 按分数降序
        hits.sort(key=lambda x: x[5], reverse=True)
        top_k = max(1, min(int(top_k), 100))
        hits = hits[:top_k]

        # 记录到 discovered_chunks（标题命中也算"已发现"）
        for (lib_name, cid, heading, source_file, matched, score) in hits:
            self._record_discovered_one(
                lib_name, cid,
                heading=heading,
                source_file=source_file,
                matched_words=matched,
                snippet="",
                score=score,
            )

        lines = [f"标题检索结果（共{len(hits)}条，query='{query}'）："]
        for i, (lib_name, cid, heading, source_file, matched, score) in enumerate(hits, 1):
            parts = [f"{i}. chunk_id={cid} | 库={lib_name}"]
            if heading:
                parts.append(f"标题={heading}")
            if source_file:
                parts.append(f"来源={source_file}")
            parts.append(f"匹配={matched}")
            parts.append(f"分={score}")
            lines.append(" | ".join(parts))
        return "\n".join(lines)

    def _dispatch_subagent(self, subtask: str,
                           chunks: List[Dict[str, str]],
                           context_hint: str = "") -> str:
        """派遣子智能体处理一组 chunk。

        子智能体是带独立工具循环的真子智能体（subagent.run_subagent）：
        它拿到分配 chunk 的清单（ID+标题+预览）后，自主决定读哪些 chunk、
        读多长、是否在范围内检索关键词，完成后提交子任务回答。
        本方法只负责构造受限的回调（read_chunk / search_chunks，均限定
        在分配范围内，并通过 _get_chunk 记录引用来源），并汇总执行结果。

        多个 dispatch_subagent 在同一轮并行执行时，各自运行在独立线程中，
        每个子智能体拥有独立上下文与轮次预算。

        参数：
            subtask: 子任务描述（要从 chunk 中提取什么信息）
            chunks: [{"library": "xxx", "chunk_id": "zone_001/chunk_000123"}, ...]
            context_hint: 额外上下文提示

        返回：
            子智能体的回答文本（带 chunk_id 标注）
        """
        from subagent import run_subagent, DEFAULT_MAX_ROUNDS as SUBAGENT_MAX_ROUNDS
        if not subtask:
            return "子任务描述不能为空"
        if not chunks:
            return "chunks 列表不能为空"
        if self.client is None:
            return "子智能体不可用：未配置 DeepSeekClient"

        # 限制单次分配的 chunk 数量，避免上下文过大
        MAX_SUBAGENT_CHUNKS = 8
        if len(chunks) > MAX_SUBAGENT_CHUNKS:
            chunks = chunks[:MAX_SUBAGENT_CHUNKS]

        # 加载分配 chunk 的元数据（标题+预览），子智能体自行决定阅读策略
        allowed: List[Dict[str, str]] = []
        for c in chunks:
            lib_name = c.get("library", "")
            cid = c.get("chunk_id", "")
            if not lib_name or not cid:
                continue
            heading, preview = "", ""
            try:
                lib = self.registry.get_library(lib_name)
                if lib is not None:
                    mgr = lib.manager(self.base_dir)
                    parts = cid.split("/")
                    if len(parts) == 2:
                        zone = mgr.get_zone(parts[0])
                        chunk = zone.read_chunk(int(parts[1].split("_")[1]))
                        heading = chunk.get("heading", "") or ""
                        text = chunk.get("text", "") or ""
                        preview = text[:120].replace("\n", " ")
            except Exception:
                pass
            if not preview:
                # 元数据读取失败时兜底：直接加载正文取预览
                t = _load_chunk_text(self.base_dir, lib_name, cid)
                preview = (t or "")[:120].replace("\n", " ")
            allowed.append({
                "library": lib_name, "chunk_id": cid,
                "heading": heading, "preview": preview,
            })

        if not allowed:
            return "所有 chunk 参数无效（缺少 library 或 chunk_id），无法派遣子智能体。"

        cid_to_lib = {a["chunk_id"]: a["library"] for a in allowed}
        lib_chunks: Dict[str, List[str]] = {}
        for a in allowed:
            lib_chunks.setdefault(a["library"], []).append(a["chunk_id"])

        def _read(cid: str, length: int = 8000) -> str:
            """子智能体的 read_chunk 回调：限定分配范围，记录引用来源。"""
            lib = cid_to_lib.get(cid)
            if lib is None:
                return (f"chunk_id '{cid}' 不在本次分配清单中。"
                        f"可用 chunk_id: {', '.join(sorted(cid_to_lib))}")
            try:
                return self._get_chunk(lib, cid, length=max(1, min(int(length), 20000)))
            except Exception as e:  # noqa: BLE001 - 错误回传给子智能体自行调整
                return f"读取 chunk 失败: {e}"

        def _search_in_scope(query: str, top_k: int = 10) -> str:
            """子智能体的 search_chunks 回调：只在分配范围内做关键词检索。"""
            if not query or not query.strip():
                return "查询词不能为空"
            try:
                top_k = max(1, min(int(top_k), 30))
            except (TypeError, ValueError):
                top_k = 10
            parts = []
            for lib, cids in lib_chunks.items():
                try:
                    parts.append(self._search(
                        query, mode="keyword", libraries=[lib], top_k=top_k,
                        chunk_filter={"library": lib, "chunks": cids},
                    ))
                except Exception as e:  # noqa: BLE001
                    parts.append(f"库 {lib} 范围内检索失败: {e}")
            if not parts:
                return "分配范围内没有可检索的 chunk。"
            return "\n\n".join(parts)

        result = run_subagent(
            client=self.client,
            subtask=subtask,
            allowed_chunks=allowed,
            read_chunk=_read,
            search_chunks=_search_in_scope,
            question=self.question,
            context_hint=context_hint,
            temperature=self.temperature,
            max_rounds=SUBAGENT_MAX_ROUNDS,
        )

        # 记录子智能体执行情况，供前端展示
        tool_calls = result.get("tool_calls", [])
        read_ids = {tc.get("args", {}).get("chunk_id", "")
                    for tc in tool_calls if tc.get("tool") == "read_chunk"}
        read_ids.discard("")
        finish_reason = result.get("finish_reason", "")
        record = {
            "subtask": subtask,
            "chunks": [f"{a['library']}/{a['chunk_id']}" for a in allowed],
            "loaded_count": len(read_ids),
            "answer_length": len(result.get("answer", "")),
            "context_hint": context_hint,
            "rounds": result.get("rounds", 0),
            "tool_call_count": len(tool_calls),
            "finish_reason": finish_reason,
            "tool_calls": tool_calls,
        }
        with self._state_lock:
            self.subagent_records.append(record)

        status = "正常完成" if finish_reason == "finish" else (
            f"异常结束（{finish_reason or '未知'}）" if finish_reason == "error"
            else "轮次耗尽自动收尾")
        header = (f"【子智能体已完成】子任务: {subtask}\n"
                  f"分配 {len(allowed)} 个 chunk，执行 {result.get('rounds', 0)} 轮、"
                  f"调用工具 {len(tool_calls)} 次（实际阅读 {len(read_ids)} 个 chunk，{status}）\n\n"
                  "【子智能体回答】\n")
        return header + (result.get("answer") or "（子智能体未返回内容）")

    def _search(self, query: str, mode: str = "keyword",
                libraries: Optional[List[str]] = None,
                queries: Optional[List[str]] = None,
                top_k: int = 20,
                title_filter: Optional[Dict[str, Any]] = None,
                semantic_filter: Optional[Dict[str, Any]] = None,
                tag_filter: Optional[Dict[str, Any]] = None,
                chunk_filter: Optional[Dict[str, Any]] = None) -> str:
        """执行检索，返回格式化的结果。

        支持 title_filter / semantic_filter / tag_filter / chunk_filter 四种二次筛选。
        tag_filter 按 chunk 的 tags 字段筛选（jieba 提取的人物/地名/机构/专有名词）。
        chunk_filter 按 chunk_id 列表或序号区间筛选（限定检索范围）。
        """
        if not query and mode != "multi_keyword":
            return "查询词不能为空"

        # chunk_filter 预处理：解析 ranges 为显式 chunk_id 集合
        allowed_chunk_ids: Optional[set] = None
        if chunk_filter:
            allowed_chunk_ids = self._resolve_chunk_filter(chunk_filter)
            if allowed_chunk_ids is None:
                return f"chunk_filter 解析失败：{chunk_filter}"
            if not allowed_chunk_ids:
                return "chunk_filter 解析后为空集合，无 chunk 可检索"

        try:
            if mode == "semantic":
                sr = search_semantic_progressive(
                    self.registry, query, self.base_dir,
                    library_names=libraries,
                    parent_top_k=max(top_k, 10),
                    child_top_k=max(top_k // 4, 3),
                )
            elif mode == "multi_keyword":
                if not queries:
                    queries = [query] if query else []
                if not queries:
                    return "multi_keyword模式需要提供queries参数"
                sr = search_multi_keywords(
                    self.registry, queries, self.base_dir,
                    library_names=libraries,
                    top_k=top_k,
                )
            else:  # keyword
                sr = parallel_search(
                    self.registry, query,
                    library_names=libraries,
                    base_dir=self.base_dir,
                )
        except Exception as e:
            return f"检索失败: {e}"

        results = sr.get("results", [])
        if not results:
            return f"未检索到与'{query}'相关的内容。"

        # 二次筛选：chunk_filter（按 chunk_id 列表限定范围）
        filter_log = []
        if allowed_chunk_ids is not None:
            before = len(results)
            results = [h for h in results if h.get("chunk_id") in allowed_chunk_ids]
            filter_log.append(f"chunk_filter: {before}→{len(results)}")

        # 二次筛选：title_filter
        if title_filter:
            before = len(results)
            results = self._apply_title_filter(results, title_filter)
            filter_log.append(f"title_filter: {before}→{len(results)}")

        # 二次筛选：tag_filter（按 chunk 的 tags 字段）
        if tag_filter:
            before = len(results)
            results = self._apply_tag_filter(results, tag_filter)
            filter_log.append(f"tag_filter: {before}→{len(results)}")

        # 二次筛选：semantic_filter
        if semantic_filter and semantic_filter.get("text"):
            before = len(results)
            results = self._apply_semantic_filter(results, semantic_filter)
            filter_log.append(f"semantic_filter: {before}→{len(results)}")

        if not results:
            msg = f"检索到{sr.get('total_hits', 0)}条，但筛选后无剩余结果。"
            if filter_log:
                msg += f"（筛选过程: {'; '.join(filter_log)}）"
            return msg

        # 格式化结果
        total = sr.get("total_hits", len(results))
        header = f"检索结果（共{total}条"
        if filter_log:
            header += f"，筛选后{len(results)}条: {'; '.join(filter_log)}"
        header += f"，显示前{min(len(results), top_k)}条）："
        lines = [header]
        for i, h in enumerate(results[:top_k], 1):
            cid = h.get("chunk_id", "")
            lib_name = h.get("library", "")
            heading = h.get("heading", "")
            snippet = h.get("snippet", "")
            sub_text = h.get("sub_text", "")
            score = h.get("score", 0)
            hit_count = h.get("hit_count", 0)
            matched = h.get("matched_words", [])

            # 优先用 sub_text（语义检索），否则用 snippet
            display_snippet = sub_text or snippet
            if len(display_snippet) > 300:
                display_snippet = display_snippet[:300] + "..."

            parts = [f"{i}. chunk_id={cid} | 库={lib_name}"]
            if heading:
                parts.append(f"标题={heading}")
            if isinstance(score, float):
                parts.append(f"语义分={score:.3f}")
            else:
                parts.append(f"命中={hit_count}")
            if matched:
                parts.append(f"匹配词={matched}")
            # 若有 tags（tag_filter 加载过或检索结果自带），展示前 5 个
            tags_list = h.get("tags")
            if tags_list:
                parts.append(f"标签={tags_list[:5]}")
            # 注入元数据摘要（时代/主题/人物），让 LLM 基于元数据辅助判断
            md_summary = _format_metadata_for_prompt(h.get("metadata"))
            if md_summary:
                parts.append(f"元数据:{md_summary}")
            # 若有 semantic_filter，附加二次语义分
            sf_score = h.get("filter_semantic_score")
            if sf_score is not None:
                parts.append(f"筛选语义分={sf_score:.3f}")
            lines.append(" | ".join(parts))
            if display_snippet:
                lines.append(f"   片段: {display_snippet}")

        # 记录到 discovered_chunks（检索命中但未展开的 chunk 也算"已发现"）
        self._record_discovered_chunks(results[:top_k])

        return "\n".join(lines)

    def _apply_title_filter(self, results: List[Dict[str, Any]],
                            tf: Dict[str, Any]) -> List[Dict[str, Any]]:
        """按标题(heading)和源文件名(source_file)筛选结果。

        heading 是 chunk 级篇名（如"水利篇"），不含来源名；
        source_file 是源文件名（如"某年鉴2024.txt"），通常含来源名。
        过滤时同时检查两个字段：任一字段匹配即通过。

        支持：
          contains:     heading 或 source_file 含任一关键词（OR）
          contains_all: heading 或 source_file 含所有关键词（AND）
          excludes:     heading 和 source_file 都不含这些关键词
          regex:        heading 或 source_file 正则匹配（忽略大小写）
        """
        contains = tf.get("contains") or []
        contains_all = tf.get("contains_all") or []
        excludes = tf.get("excludes") or []
        regex_pat = tf.get("regex") or ""

        # 预编译正则
        regex_re = None
        if regex_pat:
            try:
                regex_re = re.compile(regex_pat, re.IGNORECASE)
            except re.error:
                regex_re = None

        def _match_field(text: str) -> bool:
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
        for h in results:
            heading = (h.get("heading") or "").strip()
            sf = h.get("source_file") or ""
            if sf:
                sf = os.path.basename(sf)
            if _match_field(heading) or _match_field(sf):
                filtered.append(h)
        return filtered

    def _load_chunk_tags(self, library: str, chunk_id: str) -> List[str]:
        """读取 chunk 的 tags 字段（带缓存）。

        parallel_search 返回结果不含 tags，需要从 chunk JSON 文件读取。
        同一会话内同一 chunk 重复读取时走缓存。
        """
        cache_key = f"{library}|{chunk_id}"
        if cache_key in self._tags_cache:
            return self._tags_cache[cache_key]

        tags: List[str] = []
        try:
            lib = self.registry.get_library(library)
            if lib is not None:
                mgr = lib.manager(self.base_dir)
                parts = chunk_id.split("/")
                if len(parts) == 2:
                    zone_id, chunk_name = parts
                    zone = mgr.get_zone(zone_id)
                    if zone is not None:
                        chunk_path = os.path.join(zone.chunks_dir, f"{chunk_name}.json")
                        if os.path.isfile(chunk_path):
                            with open(chunk_path, "r", encoding="utf-8") as f:
                                chunk = json.load(f)
                            tags = chunk.get("tags") or []
        except Exception:
            pass

        self._tags_cache[cache_key] = tags
        return tags

    def _apply_tag_filter(self, results: List[Dict[str, Any]],
                          tf: Dict[str, Any]) -> List[Dict[str, Any]]:
        """按 chunk 的 tags 字段筛选结果。

        支持：
          contains_any: tags 含任一标签（OR）
          contains_all: tags 含所有标签（AND）
          excludes:    tags 不含这些标签

        注意：parallel_search 返回结果不含 tags 字段，需要从 chunk 文件读取。
        """
        contains_any = tf.get("contains_any") or []
        contains_all = tf.get("contains_all") or []
        excludes = tf.get("excludes") or []

        if not (contains_any or contains_all or excludes):
            return results

        filtered = []
        for h in results:
            lib_name = h.get("library", "")
            cid = h.get("chunk_id", "")
            if not cid:
                continue
            tags = self._load_chunk_tags(lib_name, cid)
            tags_set = set(tags)

            # contains_any（OR）：tags 必须含任一标签
            if contains_any and not any(t in tags_set for t in contains_any):
                continue
            # contains_all（AND）：tags 必须含所有标签
            if contains_all and not all(t in tags_set for t in contains_all):
                continue
            # excludes：tags 不能含任一标签
            if excludes and any(t in tags_set for t in excludes):
                continue
            # 把 tags 附到结果上，供后续格式化展示
            h["tags"] = tags
            filtered.append(h)
        return filtered

    def _apply_semantic_filter(self, results: List[Dict[str, Any]],
                               sf: Dict[str, Any]) -> List[Dict[str, Any]]:
        """对结果做语义相似度二次筛选。

        策略：用 sf['text'] 作为查询文本，对 results 中每个 chunk 的
        snippet/正文计算向量相似度，按相似度降序，过滤低于 min_score 的，
        保留前 top_n 条。

        实现：复用 search_semantic_progressive 在指定 chunk 范围内检索。
        但更简单稳健的方式是用 embedding 直接算余弦相似度。
        """
        text = sf.get("text") or ""
        min_score = float(sf.get("min_score", 0.3))
        top_n = int(sf.get("top_n", 10))

        if not text or not results:
            return results

        # 收集候选 chunk_id 和库
        candidate_ids = [h.get("chunk_id") for h in results if h.get("chunk_id")]
        if not candidate_ids:
            return results

        # 用 searcher 提供的 chunk 文本 + 简单字面相似度（避免引入 embedding 依赖）
        # 对每个候选 chunk 的 snippet/heading 计算与 text 的字符重叠率作为近似语义分
        # 若需要真正的向量相似度，可调用 search_semantic_progressive 的子检索
        scored = []
        for h in results:
            cid = h.get("chunk_id", "")
            snippet = h.get("snippet") or h.get("sub_text") or ""
            heading = h.get("heading") or ""
            chunk_text = f"{heading} {snippet}"

            # 字面相似度：text 中的字符在 chunk_text 中的覆盖比例
            # 对中文按 bigram 计算 Jaccard 系数
            sim = _text_similarity(text, chunk_text)

            if sim >= min_score:
                h_copy = dict(h)
                h_copy["filter_semantic_score"] = round(sim, 3)
                scored.append(h_copy)

        # 按相似度降序，取 top_n
        scored.sort(key=lambda x: x.get("filter_semantic_score", 0), reverse=True)
        return scored[:top_n]

    def _report_data_issue(
        self,
        issue_type: str,
        description: str,
        chunk_ids: Optional[List[str]] = None,
        library: str = "",
        severity: str = "info",
    ) -> str:
        """向质询报告页面写入数据问题报告。

        保存到 InquiryStore，前端可在「质询报告」tab 查看。
        """
        if not description:
            return "描述不能为空"

        import time as _time
        from inquiry_store import InquiryStore

        # 规范化参数
        valid_types = {"数据冲突", "常识错误", "前后矛盾", "重要遗漏", "其他"}
        if issue_type not in valid_types:
            issue_type = "其他"
        valid_severities = {"info", "warning", "error"}
        if severity not in valid_severities:
            severity = "info"

        libs = [library] if library else (list(self.library_names) if self.library_names else [])

        # 解析每个引用 chunk 所属的库（优先显式 library 参数，
        # 其次会话历史引用记录），供质询报告页一键展开原文验证
        ref_lib_map = {h["chunk_id"]: h["library"] for h in self.history_refs}
        chunk_refs = []
        for cid in (chunk_ids or []):
            cid = (cid or "").strip()
            if not cid:
                continue
            chunk_refs.append({
                "chunk_id": cid,
                "library": library or ref_lib_map.get(cid, ""),
            })

        report = {
            "id": f"ai_{int(_time.time() * 1000)}",
            "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%S"),
            "question": self.question,
            "libraries": libs,
            "source": "ai_workflow",
            "issue_type": issue_type,
            "description": description,
            "chunk_ids": [r["chunk_id"] for r in chunk_refs],
            "chunk_refs": chunk_refs,
            "severity": severity,
            "findings_count": 1,
        }

        try:
            store = InquiryStore(self.base_dir)
            store.save(report)
            self.reported_issues.append(report)
            return f"已记录数据问题报告（类型: {issue_type}，严重度: {severity}）。报告已保存到质询报告页面。"
        except Exception as e:
            return f"报告保存失败: {e}"

    def _get_chunk_ids(self, library_name: str) -> List[str]:
        """缓存获取库的 chunk_id 列表。"""
        if library_name in self._chunk_ids_cache:
            return self._chunk_ids_cache[library_name]
        lib = self.registry.get_library(library_name)
        if lib is None:
            return []
        from searcher import _list_library_chunk_ids
        ids = _list_library_chunk_ids(lib, self.base_dir)
        self._chunk_ids_cache[library_name] = ids
        return ids

    def _resolve_chunk_filter(self, chunk_filter: Dict[str, Any]) -> Optional[set]:
        """把 chunk_filter 解析为显式 chunk_id 集合。

        支持：
          - chunks: 显式 chunk_id 列表
          - ranges: ['001-399']（区间）或 ['001,005,010']（列表）或两者混合

        返回：set(chunk_id) 或 None（解析失败）/ 空集（无 chunk）
        """
        if not chunk_filter:
            return None
        lib_name = chunk_filter.get("library", "")
        if not lib_name:
            return None
        explicit = set(chunk_filter.get("chunks", []) or [])
        ranges = chunk_filter.get("ranges", []) or []

        # 解析 ranges
        try:
            all_ids = self._get_chunk_ids(lib_name)
        except Exception:
            all_ids = []
        # 构建 chunk 序号 → chunk_id 映射
        # chunk_id 形如 'zone_001/chunk_000123'，序号取 000123 → 123
        seq_to_id: Dict[int, str] = {}
        for cid in all_ids:
            parts = cid.split("/")
            if len(parts) != 2:
                continue
            chunk_name = parts[1]  # chunk_000123
            m = re.search(r'chunk_(\d+)', chunk_name)
            if m:
                seq = int(m.group(1))
                seq_to_id[seq] = cid

        for r in ranges:
            r = str(r).strip()
            if not r:
                continue
            # 形式 1：'001-399' 区间
            if '-' in r:
                parts2 = r.split('-')
                if len(parts2) != 2:
                    continue
                try:
                    lo = int(parts2[0].strip())
                    hi = int(parts2[1].strip())
                except ValueError:
                    continue
                if lo > hi:
                    lo, hi = hi, lo
                for s in range(lo, hi + 1):
                    if s in seq_to_id:
                        explicit.add(seq_to_id[s])
            # 形式 2：'001,005,010' 逗号列表
            elif ',' in r:
                for p in r.split(','):
                    p = p.strip()
                    if not p:
                        continue
                    try:
                        s = int(p)
                    except ValueError:
                        continue
                    if s in seq_to_id:
                        explicit.add(seq_to_id[s])
            # 形式 3：单个序号 '010'
            else:
                try:
                    s = int(r)
                except ValueError:
                    continue
                if s in seq_to_id:
                    explicit.add(seq_to_id[s])

        return explicit

    def _get_neighbors(self, library_name: str, chunk_id: str,
                       window: int = 1) -> str:
        """获取指定 chunk 前后相邻的 chunk 标题和预览。

        参数：
            library_name: 库名
            chunk_id: 形如 'zone_001/chunk_000123'
            window: 前后各取多少个相邻 chunk，默认 1（前1后1）
        """
        if not library_name or not chunk_id:
            return ("参数错误：library 和 chunk_id 不能为空。"
                    "请先通过 search / search_titles / list_chunk_titles 等工具获取有效的 chunk_id，"
                    "格式如 'zone_001/chunk_000123'，再调用 get_neighbors。")
        try:
            all_ids = self._get_chunk_ids(library_name)
        except Exception as e:
            return f"获取 chunk 列表失败: {e}"
        if not all_ids:
            return f"库 '{library_name}' 无 chunk"

        # 找到当前 chunk 的位置
        try:
            idx = all_ids.index(chunk_id)
        except ValueError:
            return f"chunk_id '{chunk_id}' 不在库 '{library_name}' 中"

        # 记录模型查看的当前 chunk（用于引用来源）
        self._record_accessed_chunk(library_name, chunk_id)

        # 取前后 window 个
        start = max(0, idx - window)
        end = min(len(all_ids), idx + window + 1)
        neighbor_ids = all_ids[start:end]

        # 加载每个相邻 chunk 的 heading 和预览
        lines = [f"相邻 chunk 列表（当前 chunk 索引 {idx}，前后各 {window} 个）："]
        for i, nid in enumerate(neighbor_ids, start):
            parts = nid.split("/")
            if len(parts) != 2:
                continue
            zone_id, chunk_name = parts
            try:
                lib = self.registry.get_library(library_name)
                mgr = lib.manager(self.base_dir)
                zone = mgr.get_zone(zone_id)
                chunk = zone.read_chunk(int(chunk_name.split("_")[1]))
                heading = chunk.get("heading", "") or ""
                src = chunk.get("source", {}) or {}
                source_file = src.get("file_name", "") or ""
                # 预览前 150 字
                text = _load_chunk_text(self.base_dir, library_name, nid)
                preview = (text[:150] + "...") if text and len(text) > 150 else (text or "")
                preview = preview.replace("\n", " ")
                marker = " ← 当前" if nid == chunk_id else ""
                lines.append(f"[{i}] {nid} | 标题={heading} | 来源={source_file}{marker}")
                if preview:
                    lines.append(f"    预览: {preview}")
                # 记录相邻 chunk 到 discovered_chunks（当前 chunk 已由 _record_accessed_chunk 记录）
                if nid != chunk_id:
                    self._record_discovered_one(
                        library_name, nid,
                        heading=heading,
                        source_file=source_file,
                        snippet=preview,
                    )
            except Exception as e:
                lines.append(f"[{i}] {nid} | 加载失败: {e}")
        return "\n".join(lines)


# ============================================================
#  上下文压缩
# ============================================================

# 压缩触发阈值（估算 token 数）。
# DeepSeek V4 Flash/Pro 上下文已提升至 1M token，
# 预留 100K 给生成 + system，剩余 900K 中 800K 触发压缩，留 100K 余量。
# 可通过 settings.compress_threshold_tokens 覆盖。
COMPRESS_THRESHOLD_TOKENS = 800000
# 压缩时保留最近 N 条消息不压缩（确保近期上下文精确）
COMPRESS_KEEP_RECENT = 6
# 压缩摘要 max_tokens
COMPRESS_SUMMARY_MAX_TOKENS = 1500


def _load_compress_threshold() -> int:
    """从 settings 加载压缩阈值，允许运行时调整。"""
    try:
        from settings import SettingsStore
        store = SettingsStore(_auth_base_dir())
        v = store.get("compress_threshold_tokens", None)
        if v and isinstance(v, (int, float)) and v > 0:
            return int(v)
    except Exception:
        pass
    return COMPRESS_THRESHOLD_TOKENS


# ============================================================
#  DSML 伪标签解析（工程兜底）
# ============================================================

# 模型在多轮工具调用后偶尔会把工具调用当文本输出，形如：
#   <｜｜DSML｜｜tool_calls>
#   <｜｜DSML｜｜invoke name="search">
#   <｜｜DSML｜｜parameter name="query" string="true">先主 昭烈皇帝</｜｜DSML｜｜parameter>
#   <｜｜DSML｜｜parameter name="libraries" string="false">["二十四史"]</｜｜DSML｜｜parameter>
#   </｜｜DSML｜｜invoke>
#   </｜｜DSML｜｜tool_calls>
# 这通常是上下文退化或模型对 function-calling 通道失去信心的表现。
# 工程兜底：解析这种伪标签，转成标准 tool_calls 执行，避免浪费一轮检索。

# 全角竖线 DSML 标签检测
_DSML_DETECT_RE = re.compile(r"<｜｜DSML｜｜", re.DOTALL)


def _parse_dsml_tool_calls(text: str) -> List[Dict[str, Any]]:
    """从文本中解析 DSML 伪标签，返回标准 tool_calls 列表。

    返回格式与 OpenAI function-calling 一致：
        [{
            "id": "dsml_xxx",
            "type": "function",
            "function": {"name": "search", "arguments": "{...json...}"}
        }, ...]

    解析失败返回空列表（调用方回退到丢弃策略）。
    """
    if not text or not _DSML_DETECT_RE.search(text):
        return []

    tool_calls = []
    # 匹配每个 invoke 块：<｜｜DSML｜｜invoke name="xxx"> ... </｜｜DSML｜｜invoke>
    invoke_re = re.compile(
        r'<｜｜DSML｜｜invoke\s+name="([^"]+)"\s*>(.*?)</｜｜DSML｜｜invoke>',
        re.DOTALL,
    )
    # 匹配 parameter：<｜｜DSML｜｜parameter name="xxx" string="true|false">值</｜｜DSML｜｜parameter>
    param_re = re.compile(
        r'<｜｜DSML｜｜parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>(.*?)</｜｜DSML｜｜parameter>',
        re.DOTALL,
    )

    for m in invoke_re.finditer(text):
        fn_name = m.group(1).strip()
        body = m.group(2)
        if not fn_name:
            continue

        args: Dict[str, Any] = {}
        for pm in param_re.finditer(body):
            p_name = pm.group(1).strip()
            p_is_str = pm.group(2) == "true"
            p_val = pm.group(3).strip()
            if not p_name:
                continue
            # string=false 时尝试解析为 JSON（数组/对象/数字/布尔）
            if p_is_str:
                args[p_name] = p_val
            else:
                try:
                    args[p_name] = json.loads(p_val)
                except (json.JSONDecodeError, ValueError):
                    # 解析失败回退为字符串
                    args[p_name] = p_val

        tool_calls.append({
            "id": f"dsml_{len(tool_calls)}",
            "type": "function",
            "function": {
                "name": fn_name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        })

    return tool_calls


def _strip_dsml_from_text(text: str) -> str:
    """从文本中剥离所有 DSML 标签块，返回干净文本。"""
    if not text or not _DSML_DETECT_RE.search(text):
        return text
    # 剥离 <｜｜DSML｜｜tool_calls>...</｜｜DSML｜｜tool_calls> 整块
    text = re.sub(
        r'<｜｜DSML｜｜tool_calls>.*?</｜｜DSML｜｜tool_calls>',
        '',
        text,
        flags=re.DOTALL,
    )
    # 剥离未闭合的 tool_calls 块（流式时可能没收到闭合标签）
    text = re.sub(
        r'<｜｜DSML｜｜tool_calls>.*$',
        '',
        text,
        flags=re.DOTALL,
    )
    # 剥离所有 invoke 块
    text = re.sub(
        r'<｜｜DSML｜｜invoke\s+name="[^"]+"\s*>.*?</｜｜DSML｜｜invoke>',
        '',
        text,
        flags=re.DOTALL,
    )
    # 剥离孤立的 parameter 标签
    text = re.sub(
        r'<｜｜DSML｜｜parameter[^>]*>.*?</｜｜DSML｜｜parameter>',
        '',
        text,
        flags=re.DOTALL,
    )
    return text.strip()


class DSMLStreamFilter:
    """流式 DSML 过滤器，处理跨 delta 的 DSML 标签。

    解决问题：DSML 标签（如 <｜｜DSML｜｜tool_calls>）被分成多个 delta 传输时，
    逐 delta 检测会漏检，导致 DSML 文本泄漏到主输出。

    工作原理：
    - 维护一个 buffer，累积未确定性质的文本
    - 检测 DSML 开始标记 <｜｜DSML｜｜
    - 标记之前的内容安全 yield
    - 标记之后的内容进入 DSML 模式，等待结束标记 </｜｜DSML｜｜tool_calls>
    - 检查 buffer 末尾是否有部分开始标记（如 "<" 或 "<｜"），延迟 yield
    """

    DSML_START = "<｜｜DSML｜｜"
    DSML_END = "</｜｜DSML｜｜tool_calls>"

    def __init__(self):
        self.buffer = ""
        self.in_dsml = False
        # 收集的 DSML 文本（用于解析工具调用）
        self.dsml_text = ""

    def feed(self, delta: str):
        """喂入一个 delta，返回 (yield_chunks, dsml_completed)。
        yield_chunks: 应该 yield 给前端的干净文本列表
        dsml_completed: 如果为 True，表示一个 DSML 块已完成，self.dsml_text 可供解析
        """
        result = []
        dsml_completed = False
        self.buffer += delta

        while True:
            if self.in_dsml:
                # 在 DSML 块内，寻找结束标记
                idx = self.buffer.find(self.DSML_END)
                if idx >= 0:
                    # 找到结束标记，收集 DSML 文本，跳过整个块
                    self.dsml_text += self.buffer[:idx + len(self.DSML_END)]
                    self.buffer = self.buffer[idx + len(self.DSML_END):]
                    self.in_dsml = False
                    dsml_completed = True
                else:
                    # 还没找到结束标记：把 buffer 中确定不可能是结束标记
                    # 前缀的部分移入 dsml_text，保留"结尾恰好是 DSML_END
                    # 前 k 字符"的尾部继续累积——否则结束标记跨 delta 分片
                    # 时永远无法在 buffer 中凑齐，DSML 块会永不完成
                    keep = min(len(self.DSML_END) - 1, len(self.buffer))
                    while keep > 0 and self.buffer[-keep:] != self.DSML_END[:keep]:
                        keep -= 1
                    split_at = len(self.buffer) - keep
                    self.dsml_text += self.buffer[:split_at]
                    self.buffer = self.buffer[split_at:]
                    break
            else:
                # 不在 DSML 块内，寻找开始标记
                idx = self.buffer.find(self.DSML_START)
                if idx >= 0:
                    # 找到开始标记，yield 之前的内容
                    if idx > 0:
                        result.append(self.buffer[:idx])
                    self.dsml_text = self.buffer[idx:]
                    self.buffer = ""
                    self.in_dsml = True
                    # 立即检查 dsml_text 是否已包含结束标记
                    # （DSML 标签可能完整在一个 delta 里）
                    end_idx = self.dsml_text.find(self.DSML_END)
                    if end_idx >= 0:
                        # 已包含结束标记，提取结束标记之后的内容
                        after_end = self.dsml_text[end_idx + len(self.DSML_END):]
                        self.dsml_text = self.dsml_text[:end_idx + len(self.DSML_END)]
                        self.in_dsml = False
                        dsml_completed = True
                        if after_end:
                            self.buffer = after_end
                else:
                    # 没找到开始标记
                    # 检查 buffer 末尾是否有部分开始标记（如 "<"、"｜" 等）
                    safe_len = len(self.buffer)
                    for i in range(1, min(len(self.DSML_START), len(self.buffer)) + 1):
                        if self.buffer[-i:] == self.DSML_START[:i]:
                            safe_len = len(self.buffer) - i
                            break

                    if safe_len > 0:
                        result.append(self.buffer[:safe_len])
                        self.buffer = self.buffer[safe_len:]
                    # 保留可能的开始标记前缀，继续累积
                    break

        return result, dsml_completed

    def flush(self):
        """流结束时调用，返回剩余的安全内容。"""
        result = []
        if not self.in_dsml and self.buffer:
            result.append(self.buffer)
        elif self.in_dsml and self.buffer:
            # 流在 DSML 块内被截断：残余并入 dsml_text。
            # 结束标记未出现，块按未完成处理（不会触发工具调用解析）。
            self.dsml_text += self.buffer
        self.buffer = ""
        self.in_dsml = False
        return result

    def get_dsml_text(self):
        """获取收集的 DSML 文本（用于解析工具调用）。"""
        text = self.dsml_text
        self.dsml_text = ""
        return text


# ============================================================
#  最终生成阶段：叙述性过渡语识别
# ============================================================

# 模型偶尔会在最终生成阶段先输出一段叙述（"好的，我先查看……"）再给答案，
# 这类文本流经 content 通道但不是答案正文，应降级为 thinking 事件（仅调试可见）。
# 规则取向高查准率：宁可漏判（叙述短暂漏给前端），不可误判（答案正文被吞掉）。
# 因此只认"引导词+动作动词"的组合——单独的"我认为/首先/根据/检索"等都可能
# 是合法的答案开头，不视为叙述。
_NARRATION_RE = re.compile(
    r'^(?:'
    r'(?:好的|嗯|明白了|收到|okay)[，,、：: ]'
    r'|让我(?:来|先|去|再)?(?:看|查看|查|检索|搜索|分析|梳理|确认|读取|调用|检查|展开|试试)'
    r'|我(?:先|来|需要|应该|现在)(?:看|查看|查|检索|搜索|分析|梳理|确认|读取|调用|检查|展开|试试)'
    r'|接下来(?:我|要|需要|让|先|得)'
    r'|现在(?:我|要|需要|让|先)'
    r'|首先(?:我|需要|要|得|让)'
    r'|那么(?:我|要|需要|让|先)'
    r')'
)


def _looks_like_narration(text: str) -> bool:
    """判断最终生成阶段 content 的开头是否是叙述性过渡语（非答案正文）。"""
    if not text:
        return False
    first_line = text.split("\n", 1)[0].strip()
    if not first_line:
        return False
    return bool(_NARRATION_RE.match(first_line))


def _estimate_tokens(messages: List[Dict[str, Any]]) -> int:
    """粗略估算消息列表的 token 数。

    中文 ~1.5 字/token，英文 ~4 字符/token，
    综合按 字符数 / 1.8 估算（偏保守，宁可早压缩）。
    每条消息额外计 4 token 的结构开销。
    """
    total_chars = 0
    for m in messages:
        content = m.get("content") or ""
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total_chars += len(str(part.get("text", "")))
        # tool_calls 的 arguments 也算
        tcs = m.get("tool_calls") or []
        for tc in tcs:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            total_chars += len(fn.get("name", "")) + len(fn.get("arguments", ""))
        total_chars += 4  # 结构开销
    return int(total_chars / 1.8)


def _format_middle_for_compress(middle: List[Dict[str, Any]]) -> str:
    """把待压缩的中间消息格式化为文本（供 LLM 摘要）。"""
    parts = []
    for i, m in enumerate(middle):
        role = m.get("role", "?")
        content = m.get("content") or ""
        tcs = m.get("tool_calls") or []
        if tcs:
            tc_descs = []
            for tc in tcs:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                tc_descs.append(f"调用工具 {fn.get('name','')}({fn.get('arguments','')})")
            parts.append(f"[{i}] {role}: " + "; ".join(tc_descs) + (f"\n内容: {content}" if content else ""))
        else:
            parts.append(f"[{i}] {role}: {content}")
    middle_text = "\n\n".join(parts)

    # 截断超长 middle，避免压缩请求本身超限
    if len(middle_text) > 40000:
        middle_text = middle_text[:40000] + "\n...(已截断)"
    return middle_text


def _mechanical_digest(middle: List[Dict[str, Any]],
                       max_chars: int = 20000) -> str:
    """机械压缩摘要（不依赖 LLM）。

    逐条保留消息的角色、工具名和截断后的内容，作为 LLM 摘要失败时的
    降级方案：确保上下文必然收缩，且保住 chunk_id 等关键定位信息
    （工具结果的开头通常就是 chunk_id 行）。
    """
    lines = []
    for i, m in enumerate(middle):
        role = m.get("role", "?")
        content = m.get("content") or ""
        tcs = m.get("tool_calls") or []
        if tcs:
            descs = []
            for tc in tcs:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                descs.append(f"{fn.get('name','')}({fn.get('arguments','')})")
            line = f"[{i}] {role} 调用工具: " + "; ".join(descs)
            if content:
                line += f" | 内容: {content[:120]}"
        elif role == "tool":
            line = f"[{i}] 工具结果: {content[:200]}"
        else:
            line = f"[{i}] {role}: {content[:200]}"
        lines.append(line)
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...(机械摘要已截断)"
    return text


def _compress_conversation(
    client: DeepSeekClient,
    conversation: List[Dict[str, Any]],
) -> tuple:
    """压缩对话历史，返回 (新对话列表, 估算节省的 token 数, 摘要文本, 压缩方式)。

    策略：
      1. 保留 system + 前 2 条 (user 问题)
      2. 保留最后 COMPRESS_KEEP_RECENT 条消息不动
      3. 中间的消息（多为 assistant tool_calls + tool results）用 LLM 总结
      4. LLM 总结失败时降级为机械压缩（_mechanical_digest），确保上下文
         必然收缩——不会静默沿用超长对话导致后续请求超限

    压缩方式 method ∈ {"none", "llm", "mechanical"}。
    """
    if len(conversation) <= COMPRESS_KEEP_RECENT + 3:
        return conversation, 0, "", "none"

    # 第 0 条是 system，第 1 条是 user 问题，从第 2 条开始压缩
    head = conversation[:2]
    tail_keep = min(COMPRESS_KEEP_RECENT, len(conversation) - 2)
    tail = conversation[-tail_keep:]
    middle = conversation[2:len(conversation) - tail_keep]

    if not middle:
        return conversation, 0, "", "none"

    middle_text = _format_middle_for_compress(middle)

    summary_prompt = (
        "请将以下 Agent 工作流的对话历史压缩为简洁摘要，要求：\n"
        "1. 保留所有关键信息：检索到了哪些 chunk、关键事实、已确定的答案要点\n"
        "2. 保留工具调用的关键参数和返回的核心结果（不要丢失 chunk_id 和库名）\n"
        "3. 丢弃冗余的片段原文、重复信息\n"
        "4. 用条目式输出，每条一行\n"
        "5. 总长度控制在 1500 字以内\n\n"
        f"【待压缩的对话历史】\n{middle_text}"
    )

    summary = ""
    method = "llm"
    try:
        summary = client.ask(
            summary_prompt,
            system="你是上下文压缩助手，只做事实性摘要，不添加任何推断。",
            temperature=0.0,
            max_tokens=COMPRESS_SUMMARY_MAX_TOKENS,
        )
        if not summary or not summary.strip():
            raise ValueError("LLM 返回了空摘要")
    except Exception as e:
        # LLM 摘要失败 → 机械压缩兜底：逐条截断，保证上下文收缩
        method = "mechanical"
        summary = (
            f"[LLM 摘要失败（{e}），已降级为机械压缩：逐条保留截断摘要]\n"
            + _mechanical_digest(middle)
        )

    # 计算节省的 token
    old_tokens = _estimate_tokens(middle)
    new_tokens = _estimate_tokens([{"role": "user", "content": summary}])
    saved = max(0, old_tokens - new_tokens)

    # 构造新对话：system + user 问题 + 摘要(作为 user 消息) + 最近的 tail
    new_conv = head + [
        {"role": "user", "content": f"【历史对话摘要】\n{summary}"},
    ] + tail

    return new_conv, saved, summary, method


# ============================================================
#  Agent 工作流主入口（asyncio 异步核心 + 同步桥接）
# ============================================================

async def _wrap_sync_iter(sync_iter: Iterator[Any]):
    """把同步迭代器包装为异步迭代器。

    在独立线程中迭代同步生成器（如 client.chat_stream），通过
    loop.call_soon_threadsafe 把元素送入 asyncio.Queue，事件循环线程
    内不做阻塞 I/O。消费端提前退出（break/取消）时关闭底层生成器，
    中止网络读取。
    """
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()

    def _put(item: Any):
        try:
            loop.call_soon_threadsafe(q.put_nowait, item)
        except RuntimeError:
            pass  # 事件循环已关闭（任务取消后线程仍在泵送），丢弃残余事件

    def _pump():
        try:
            for item in sync_iter:
                _put(item)
        except BaseException as e:  # noqa: BLE001 - 异常原样传给消费端
            _put(e)
        finally:
            _put(_SENTINEL)

    threading.Thread(target=_pump, daemon=True,
                     name="llm-stream-pump").start()
    try:
        while True:
            item = await q.get()
            if item is _SENTINEL:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        # 消费端提前退出：关闭底层同步生成器以中止网络读取。
        # 生成器恰好在泵线程中执行时 close 会抛 ValueError，
        # 此时泵线程会在当前阻塞调用结束后自然结束。
        try:
            sync_iter.close()
        except (ValueError, RuntimeError, AttributeError):
            pass


async def _run_tool_calls_parallel(executor: "ToolExecutor",
                                   calls: List[tuple]) -> List[str]:
    """并行执行一轮内的多个工具调用，返回按原顺序排列的结果列表。

    calls: [(fn_name, args), ...]。单个工具的未捕获异常转为错误文本，
    不影响其他工具（executor.execute 内部亦有兜底）。
    """
    async def _one(fn_name: str, args: Dict[str, Any]) -> str:
        try:
            return await asyncio.to_thread(executor.execute, fn_name, args)
        except Exception as e:  # noqa: BLE001
            return f"工具执行错误: {e}"

    if len(calls) == 1:
        fn_name, args = calls[0]
        return [await _one(fn_name, args)]
    return list(await asyncio.gather(*(_one(fn, args) for fn, args in calls)))


# 轮次预算提示：剩余轮次 <= 此值时注入一次预算提示（只提示一次，不反复催促）
BUDGET_NOTICE_REMAINING = 2


async def agent_workflow_stream_async(
    question: str,
    registry: LibraryRegistry,
    client: DeepSeekClient,
    base_dir: str,
    library_names: Optional[List[str]] = None,
    temperature: float = 0.3,
    max_rounds: int = 10,
    extra_context: Optional[List[Dict[str, Any]]] = None,
    history: Optional[List[Dict[str, Any]]] = None,
):
    """Agent 工作流异步核心：LLM 自主调用工具完成检索+生成。

    单请求内不再同步阻塞：LLM 流式调用与工具执行均在线程池中调度，
    同一轮返回多个工具调用（含多个 dispatch_subagent）时并行执行。

    事件序列：
        {"phase":"tool_call","name":"search","arguments":{...}}
        {"phase":"tool_result","name":"search","result":"...","round":R}
        {"phase":"reasoning","delta":"..."}
        {"phase":"content","delta":"..."}
        {"phase":"rounds_exhausted","round":R}   # 轮次预算耗尽（温和收尾）
        {"phase":"done","usage":{...}}
        {"phase":"error","stage":"...","error":"..."}
    """
    # stop_reason ∈ {"finish", "no_tool", "dsml", "budget"}
    # budget = 轮次预算耗尽（最终生成时会明确告知模型基于已有信息作答）
    stop_reason = "budget"
    try:
        # 构造库背景信息（预检和后续流程都需要；可能读文件，放线程池）
        library_context = await asyncio.to_thread(
            _build_library_context, registry, library_names)

        # ---------- 阶段 0：问题预检（识别错别字/歧义/信息不足）----------
        # 避免对奇怪输入浪费算力检索，先让 LLM 判断问题是否可理解
        from ai_search import _check_question
        check = await asyncio.to_thread(
            _check_question, client, question,
            history=history, library_context=library_context)
        if not check.get("ok", True):
            yield {
                "phase": "clarify",
                "reason": check.get("reason", ""),
                "clarify": check.get("clarify", ""),
            }
            yield {"phase": "done", "skipped": True, "reason": "question_clarify"}
            return

        executor = ToolExecutor(registry, base_dir, library_names=library_names,
                                question=question, client=client, temperature=temperature,
                                history=history)

        # 构造初始消息
        lib_hint = f"\n\n【数据来源】{library_context}" if library_context else ""
        if library_names:
            lib_hint += f"\n用户指定检索库：{', '.join(library_names)}"

        # 注入用户从检索结果挑选的额外上下文
        extra_ctx_text = _build_extra_ctx_text(extra_context)

        # 构造历史问答文本（只要问题和最终回答，不含细节）
        history_text = _build_history_text(history)
        history_block = f"{history_text}\n\n" if history_text else ""

        conversation: List[Dict[str, Any]] = [
            {"role": "system", "content": _AGENT_SYSTEM},
            {"role": "user", "content":
                f"{extra_ctx_text}{history_block}{lib_hint}\n\n【问题】\n{question}"},
        ]

        # 加载压缩阈值（支持 settings 运行时配置）
        compress_threshold = _load_compress_threshold()

        # 连续无工具调用计数器：模型连续输出内容但不调工具时累加，
        # 超过阈值则强制进入最终生成（避免 narration 死循环占用所有轮次）
        no_tool_count = 0
        NO_TOOL_FORCE_FINISH = 2

        # DSML 解析失败计数器：反复输出无法解析的 DSML 时强制收尾
        dsml_fail_count = 0
        DSML_FAIL_FORCE_FINISH = 2

        # 工具循环阶段使用低温度：降低采样到训练分布尾部（DSML 伪标签格式）
        # 的概率。高温度会增加模型"回退"到非标准工具调用格式的风险。
        # 最终生成阶段仍用调用方传入的 temperature，保证答案流畅度。
        TOOL_LOOP_TEMPERATURE = 0.1

        # 轮次预算提示只注入一次（剩余不足时告知预算，不反复催促）
        budget_notice_sent = False

        # Agent 工具循环
        for round_idx in range(max_rounds):
            # 上下文压缩：每轮开始前检查 token 数，超阈值则压缩
            # （LLM 摘要失败时函数内部自动降级机械压缩，保证上下文收缩）
            est_tokens = _estimate_tokens(conversation)
            if est_tokens > compress_threshold:
                try:
                    new_conv, saved, summary, method = await asyncio.to_thread(
                        _compress_conversation, client, conversation)
                    if saved > 0:
                        conversation = new_conv
                        yield {
                            "phase": "compress",
                            "round": round_idx + 1,
                            "before_tokens": est_tokens,
                            "after_tokens": _estimate_tokens(conversation),
                            "saved_tokens": saved,
                            "summary_preview": summary[:500],
                            "method": method,
                        }
                except Exception as e:  # noqa: BLE001
                    # 压缩流程本身异常不中断工作流，继续用原对话
                    yield {
                        "phase": "compress",
                        "round": round_idx + 1,
                        "error": str(e),
                    }

            tool_calls = None
            content_buffer = ""

            async for event in _wrap_sync_iter(client.chat_stream(
                conversation,
                model=client.model,
                temperature=TOOL_LOOP_TEMPERATURE,
                tools=AGENT_TOOLS,
                # 注意：DeepSeek 思考模式（V4 Pro）不支持 tool_choice="required"，
                # 会报 "Thinking mode does not support this tool_choice"。
                # 因此使用 "auto"，依赖 DSML 静默兜底处理模型退化情况。
                tool_choice="auto",
                # 设置足够大的输出 token 上限，避免工具调用参数被截断
                # （截断会导致 finish_reason=length 而非 tool_calls，模型被迫退化为文本）
                max_tokens=8192,
            )):
                etype = event.get("type")
                if etype == "reasoning":
                    # 工具循环中的 reasoning 也是思考过程，不流给主输出
                    yield {"phase": "thinking", "kind": "reasoning", "delta": event.get("delta", "")}
                elif etype == "content":
                    delta = event.get("delta", "")
                    content_buffer += delta
                    # 实时检测 DSML 伪标签，命中片段不发给前端
                    # （整段会等流结束后用解析器处理）
                    if not _DSML_DETECT_RE.search(delta):
                        # 工具循环中的 content 是"思考过程"（如"好的，我先查看..."），
                        # 不作为最终答案流给主输出，改用 thinking 事件（仅调试模式可见）
                        yield {"phase": "thinking", "kind": "content", "delta": delta}
                elif etype == "tool_calls":
                    tool_calls = event.get("tool_calls", [])
                elif etype == "finish":
                    # 检查 finish_reason：如果是 length，说明输出被 max_tokens 截断
                    fr = event.get("finish_reason", "")
                    if fr == "length":
                        yield {
                            "phase": "dsml_recovered",
                            "round": round_idx + 1,
                            "recovered_count": 0,
                            "tool_names": [],
                            "_warning": f"finish_reason=length（输出被截断），当前 max_tokens=8192",
                        }

            # 后置处理：DSML 作为合法工具调用格式之一
            # 模型有时偏好用 <｜｜DSML｜｜tool_calls> 文本标签输出工具调用，
            # 我们接受这种格式，解析为标准 tool_calls 执行（与 function-calling 等价）
            dsml_detected = bool(_DSML_DETECT_RE.search(content_buffer))
            if dsml_detected:
                parsed_calls = _parse_dsml_tool_calls(content_buffer)
                if parsed_calls:
                    # DSML 优先：即使模型同时返回了 tool_calls，也以 DSML 解析结果为准
                    # （因为模型主动选择 DSML 时，function-calling 通道可能不稳定）
                    tool_calls = parsed_calls
                    # 丢弃 content_buffer：DSML 通常伴随大段叙事文本（违反系统提示的输出纪律），
                    # 若回灌对话会强化模型"输出叙事+DSML"的行为模式，导致后续轮次持续退化。
                    # 直接清空，让 assistant 消息的 content 为空，符合标准 function-calling 形态。
                    content_buffer = ""
                    # 通知前端：已通过 DSML 通道执行工具调用
                    yield {
                        "phase": "dsml_recovered",
                        "round": round_idx + 1,
                        "recovered_count": len(parsed_calls),
                        "tool_names": [tc["function"]["name"] for tc in parsed_calls],
                    }
                    dsml_fail_count = 0  # 解析成功，重置失败计数
                else:
                    # 解析失败：可能是 DSML 标签被截断（max_tokens）或格式错误
                    dsml_fail_count += 1
                    # 清空 content 避免污染对话
                    content_buffer = ""
                    # 连续解析失败超阈值 → 强制收尾，避免死循环
                    if dsml_fail_count >= DSML_FAIL_FORCE_FINISH:
                        yield {
                            "phase": "dsml_force_finish",
                            "round": round_idx + 1,
                            "warn_count": dsml_fail_count,
                        }
                        stop_reason = "dsml"
                        break  # 跳出工具循环，进入最终生成
                    # 否则注入提示让模型用 function-calling 或调 finish
                    # 注意：不提及 DSML 字样，避免模型"学到"这种格式可用
                    # 注意：不附带 tool_calls 字段——OpenAI/DeepSeek API 协议要求
                    # assistant 消息的 tool_calls 若存在必须是非空数组，空数组会触发
                    # "Invalid messages[N].tool_calls: empty array" 校验错误。
                    conversation.append({
                        "role": "assistant",
                        "content": "",
                    })
                    conversation.append({
                        "role": "user",
                        "content": (
                            "（系统提示：上一轮输出格式异常，工具调用未被识别。"
                            "请通过 function-calling 通道直接调用工具，或调用 finish 进入最终答案生成。"
                            "不要输出任何文本标签或叙述性文字。）"
                        ),
                    })
                    continue

            # 没有工具调用 → 模型可能输出叙述/思考或直接回答
            if not tool_calls:
                no_tool_count += 1

                # 连续无工具调用超阈值 → 强制进入最终生成
                if no_tool_count > NO_TOOL_FORCE_FINISH:
                    stop_reason = "no_tool"
                    break

                # 把模型输出的内容存入对话（作为 assistant 消息），
                # 然后注入纠正提示，让模型调工具或调 finish。
                # 不直接把 content 当最终答案——避免叙述性内容污染主输出。
                # 注意：不附带 tool_calls 字段——空数组会触发 API 校验错误。
                conversation.append({
                    "role": "assistant",
                    "content": content_buffer or "",
                })

                if not content_buffer:
                    # 空输出
                    prompt = (
                        "（系统提示：上一轮输出为空。请调用工具继续检索，"
                        "或调用 finish 进入最终答案生成。）"
                    )
                else:
                    # 有内容但没调工具——大概率是叙述性思考
                    prompt = (
                        "（系统提示：你输出了文字内容但没有调用工具。"
                        "在工具循环阶段，你的任何文字输出都不会展示给用户，"
                        "只有工具调用和 finish 之后的最终答案才会展示。"
                        "请直接调用工具继续检索，或调用 finish 进入最终答案生成。"
                        "不要输出思考过程、计划描述或叙述性文字。）"
                    )
                conversation.append({
                    "role": "user",
                    "content": prompt,
                })
                continue

            # 把 assistant 的 tool_calls 消息加入对话
            conversation.append({
                "role": "assistant",
                "content": content_buffer,
                "tool_calls": tool_calls,
            })

            # 模型成功调用了工具，重置无工具调用计数
            no_tool_count = 0

            # 解析全部调用并通知前端
            parsed_calls: List[tuple] = []
            has_finish = False
            for tc in tool_calls:
                func = tc.get("function", {})
                fn_name = func.get("name", "")
                args_str = func.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    args = {}

                if fn_name == "finish":
                    has_finish = True
                parsed_calls.append((fn_name, args))
                yield {
                    "phase": "tool_call",
                    "round": round_idx + 1,
                    "name": fn_name,
                    "arguments": args,
                }

            # 并行执行本轮所有工具调用（结果保持原顺序）。
            # 多个 dispatch_subagent 同轮派遣时即多个子智能体并行运行。
            results = await _run_tool_calls_parallel(executor, parsed_calls)

            # 按序回传结果、触发专属事件、把结果加入对话
            for (fn_name, _args), tc, result in zip(parsed_calls, tool_calls, results):
                # 通知前端工具结果
                yield {
                    "phase": "tool_result",
                    "round": round_idx + 1,
                    "name": fn_name,
                    "result": result[:2000],  # 限制传输大小
                }

                # 数据问题报告事件：前端展示提示
                if fn_name == "report_data_issue" and executor.reported_issues:
                    latest = executor.reported_issues[-1]
                    yield {
                        "phase": "data_issue_reported",
                        "round": round_idx + 1,
                        "issue_type": latest.get("issue_type", ""),
                        "description": latest.get("description", ""),
                        "library": (latest.get("libraries") or [""])[0] if latest.get("libraries") else "",
                        "severity": latest.get("severity", "info"),
                        "chunk_ids": latest.get("chunk_ids", []),
                        "report_id": latest.get("id", ""),
                    }

                # 子智能体派遣事件：前端展示执行情况
                if fn_name == "dispatch_subagent" and executor.subagent_records:
                    latest = executor.subagent_records[-1]
                    yield {
                        "phase": "subagent_dispatched",
                        "round": round_idx + 1,
                        "subtask": latest.get("subtask", ""),
                        "chunks": latest.get("chunks", []),
                        "loaded_count": latest.get("loaded_count", 0),
                        "answer_length": latest.get("answer_length", 0),
                        "context_hint": latest.get("context_hint", ""),
                        "rounds": latest.get("rounds", 0),
                        "tool_call_count": latest.get("tool_call_count", 0),
                        "finish_reason": latest.get("finish_reason", ""),
                    }

                # 把结果作为 tool 消息加入对话
                # 限制长度避免上下文爆炸（依赖上下文压缩机制处理超长对话）
                MAX_TOOL_RESULT_CHARS = 20000
                if len(result) > MAX_TOOL_RESULT_CHARS:
                    conv_result = result[:MAX_TOOL_RESULT_CHARS] + f"\n...(结果已截断，原始长度 {len(result)} 字)"
                else:
                    conv_result = result
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": conv_result,
                })

            # 如果调用了 finish，进入最终答案生成
            if has_finish:
                stop_reason = "finish"
                break

            # 轮次预算提示：剩余不足时只注入一次（告知预算，不反复催促）
            remaining = max_rounds - (round_idx + 1)
            if (0 < remaining <= BUDGET_NOTICE_REMAINING
                    and not budget_notice_sent):
                budget_notice_sent = True
                conversation.append({
                    "role": "user",
                    "content": (
                        f"（系统提示：检索轮次预算还剩 {remaining} 轮。"
                        "请评估已获取的信息是否足以回答问题："
                        "足够则调用 finish 进入最终答案生成；"
                        "不足则只检索最关键的信息。）"
                    ),
                })

        # 轮次预算耗尽：温和收尾——告知模型与前端，最终生成基于已有信息作答
        if stop_reason == "budget":
            yield {"phase": "rounds_exhausted", "round": max_rounds}

        # 最终生成（不带工具，纯流式输出）
        # 注入明确指令：基于已收集的信息生成最终答案
        # 构建资料编号映射表，让模型用 [n] 引用对应的 chunk
        ref_map_text = ""
        if executor.accessed_chunks:
            ref_lines = ["【资料编号映射表】（引用时必须使用此编号，不要使用 chunk_id 中的数字）"]
            for c in executor.accessed_chunks:
                idx = c.get("index")
                cid = c.get("chunk_id", "")
                heading = c.get("heading", "")
                sf = c.get("source_file", "")
                disp = f" · {heading}" if heading else ""
                if sf:
                    disp += f"（{os.path.basename(sf)}）"
                ref_lines.append(f"[{idx}] {cid}{disp}")
            ref_map_text = "\n".join(ref_lines) + "\n\n"

        # 预算耗尽时明确告知模型：基于已有信息作答并说明缺口（而非静默截断）
        budget_note = ""
        if stop_reason == "budget":
            budget_note = (
                f"6. 【注意】检索轮次已达上限（{max_rounds} 轮）。请基于已获取的信息作答；"
                "若信息不足以完整回答，请在答案中明确说明哪些部分未能查到，不要编造。\n"
            )

        conversation.append({
            "role": "user",
            "content": (
                f"{ref_map_text}"
                "请根据以上检索到的信息，直接回答用户的问题。"
                "要求：\n"
                "1. 只输出最终答案，不要输出工具调用、DSML标签或思考过程\n"
                "2. 引用资料时在关键信息后用 [n] 标注，n 必须是上方【资料编号映射表】中的编号（[1]、[2] 等）。"
                "绝对不要使用 chunk_id 中的数字作为引用编号，也不要自创编号。\n"
                "3. 如果信息不足，诚实说明哪些部分未找到\n"
                "4. 用清晰、结构化的方式呈现答案\n"
                "5. 【极重要】原文资料中可能含有 [数字] 形式的注解/脚注编号（如 [91]、[102]），"
                "这些是原文自带的标记，不是你的引用标记。绝对不要在答案中保留或复用这些编号，"
                "你只能使用上方【资料编号映射表】中列出的 [n] 编号进行引用。\n"
                f"{budget_note}"
            ),
        })

        # 最终生成阶段：支持 DSML 工具调用静默执行 + 重新生成
        # Pro 思考模式下，模型可能在最终生成阶段仍输出 DSML 工具调用
        # 这时静默执行工具调用，把结果加入对话，重新生成最终答案
        max_final_retries = 3  # 最多重试 3 次，避免死循环
        final_attempt = 0
        final_usage = None  # 保存最终生成的 usage，延迟到 retrieval 之后发送 done

        while final_attempt < max_final_retries:
            final_attempt += 1
            dsml_filter = DSMLStreamFilter()
            # 收集本轮的非 DSML 文本（可能是思考过程或正式答案）
            round_content = ""
            has_dsml_tool_call = False
            # 延迟 yield：前 N 字先累积，判断是思考过程还是正式答案
            pending_buffer = ""
            pending_resolved = False  # 是否已判断出内容类型
            is_thinking = False  # 当前内容是否是思考过程

            final_stream = _wrap_sync_iter(client.chat_stream(
                conversation, model=client.model, temperature=temperature,
                max_tokens=8192,
            ))
            async for event in final_stream:
                etype = event.get("type")
                if etype == "reasoning":
                    yield {"phase": "reasoning", "delta": event.get("delta", "")}
                elif etype == "content":
                    delta = event.get("delta", "")
                    # 用流式过滤器处理跨 delta 的 DSML 标签
                    chunks, dsml_completed = dsml_filter.feed(delta)

                    for chunk in chunks:
                        round_content += chunk
                        # 延迟判断：累积前 50 字，判断是叙述性过渡语还是正式答案；
                        # 无法确定时默认当答案（宁漏判不误判，避免吞掉答案正文）
                        if not pending_resolved:
                            pending_buffer += chunk
                            # 累积到足够长度或遇到换行，判断类型
                            if len(pending_buffer) >= 30 or '\n' in pending_buffer:
                                is_thinking = _looks_like_narration(pending_buffer)
                                pending_resolved = True
                                # 把累积的内容发出去
                                if is_thinking:
                                    yield {"phase": "thinking", "kind": "content", "delta": pending_buffer}
                                else:
                                    yield {"phase": "content", "delta": pending_buffer}
                                pending_buffer = ""
                        else:
                            # 已判断类型，直接发
                            if is_thinking:
                                yield {"phase": "thinking", "kind": "content", "delta": chunk}
                            else:
                                yield {"phase": "content", "delta": chunk}

                    # 如果 DSML 块完成，尝试解析工具调用
                    if dsml_completed:
                        dsml_text = dsml_filter.get_dsml_text()
                        parsed_calls = _parse_dsml_tool_calls(dsml_text)
                        if parsed_calls:
                            has_dsml_tool_call = True
                            # 静默执行工具调用（并行，保持顺序）
                            dsml_results = await _run_tool_calls_parallel(
                                executor,
                                [(tc["function"]["name"],
                                  _safe_json_loads(tc["function"].get("arguments", "{}")))
                                 for tc in parsed_calls])
                            for tc, result in zip(parsed_calls, dsml_results):
                                fn_name = tc["function"]["name"]
                                # 限制结果长度
                                if len(result) > 20000:
                                    result = result[:20000] + "\n...(结果已截断)"
                                # 把 DSML 工具调用转为标准格式加入对话
                                conversation.append({
                                    "role": "assistant",
                                    "content": "",
                                    "tool_calls": [{
                                        "id": tc.get("id", f"dsml_{final_attempt}"),
                                        "type": "function",
                                        "function": tc["function"],
                                    }],
                                })
                                conversation.append({
                                    "role": "tool",
                                    "tool_call_id": tc.get("id", f"dsml_{final_attempt}"),
                                    "content": result,
                                })
                                # 通知前端（调试模式可见）
                                yield {
                                    "phase": "dsml_recovered",
                                    "round": final_attempt,
                                    "recovered_count": 1,
                                    "tool_names": [fn_name],
                                    "_stage": "final_generation",
                                }
                elif etype in ("finish", "done"):
                    # flush 剩余的安全内容
                    for chunk in dsml_filter.flush():
                        round_content += chunk
                        if not pending_resolved:
                            pending_buffer += chunk
                        else:
                            if is_thinking:
                                yield {"phase": "thinking", "kind": "content", "delta": chunk}
                            else:
                                yield {"phase": "content", "delta": chunk}
                    # 流结束时仍未定性的累积内容：按答案输出
                    # （宁漏判不误判——短于 30 字且无换行的答案不能被吞掉）
                    if not pending_resolved and pending_buffer:
                        yield {"phase": "content", "delta": pending_buffer}
                        pending_buffer = ""
                        pending_resolved = True
                    final_usage = event.get("usage")
                    break

            # 关闭本轮 LLM 流迭代器（async for 提前 break 不会自动 close，
            # 残留的异步生成器会在事件循环关闭后无法 aclose 并告警）
            with contextlib.suppress(Exception):
                await final_stream.aclose()

            # 如果没有 DSML 工具调用，说明模型输出了正常答案，退出循环
            if not has_dsml_tool_call:
                break

            # 如果有 DSML 工具调用，但还没到最大重试次数，重新生成
            if final_attempt < max_final_retries:
                # 把本轮的非 DSML 文本（思考过程）加入对话
                if round_content:
                    # 更新最后一条 assistant 消息的 content
                    for msg in reversed(conversation):
                        if msg.get("role") == "assistant":
                            msg["content"] = round_content
                            break
                # 追加重新生成的提示
                conversation.append({
                    "role": "user",
                    "content": (
                        "（系统提示：已执行你请求的工具调用，结果已加入上下文。"
                        "现在请基于所有已获取的信息，直接输出最终答案。"
                        "不要再次调用工具，不要输出 DSML 标签，只输出答案文本。）"
                    ),
                })
            else:
                # 达到最大重试次数，强制结束
                yield {
                    "phase": "dsml_recovered",
                    "round": final_attempt,
                    "recovered_count": 0,
                    "tool_names": [],
                    "_warning": "最终生成阶段 DSML 工具调用达到最大重试次数，强制结束",
                }

        # 发送引用来源（模型实际访问过的 chunk）
        # 必须在 done 之前发送，前端收到 done 后会关闭流
        if executor.accessed_chunks:
            yield {
                "phase": "retrieval",
                "references": executor.accessed_chunks,
                "queries": [],
                "retrieval": {"total_unique": len(executor.accessed_chunks)},
            }
        yield {"phase": "done", "usage": final_usage}

    except Exception as e:  # noqa: BLE001
        yield {"phase": "error", "error": str(e), "stage": "agent_workflow"}


def _safe_json_loads(text: str) -> Dict[str, Any]:
    """解析工具调用参数 JSON，失败时返回空 dict。"""
    try:
        return json.loads(text) if isinstance(text, str) else (text or {})
    except (json.JSONDecodeError, ValueError, TypeError):
        return {}


def agent_workflow_stream(
    question: str,
    registry: LibraryRegistry,
    client: DeepSeekClient,
    base_dir: str,
    library_names: Optional[List[str]] = None,
    temperature: float = 0.3,
    max_rounds: int = 10,
    extra_context: Optional[List[Dict[str, Any]]] = None,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Iterator[Dict[str, Any]]:
    """Agent 工作流同步入口（web_api SSE 桥接）。

    在独立线程的事件循环中驱动 agent_workflow_stream_async，通过
    线程安全队列把事件逐个转发给同步消费端。单请求内部由 asyncio
    调度：LLM 调用与工具执行不阻塞事件循环，同轮多个工具（含多个
    子智能体）并行执行。消费端提前关闭（客户端断开）时取消异步
    任务，停止后端 LLM 调用。

    事件序列与 agent_workflow_stream_async 一致。
    """
    out_q: "queue.SimpleQueue" = queue.SimpleQueue()
    _DONE = object()
    loop = asyncio.new_event_loop()
    task_box: Dict[str, Any] = {}

    async def _drive():
        agen = agent_workflow_stream_async(
            question, registry, client, base_dir,
            library_names=library_names, temperature=temperature,
            max_rounds=max_rounds, extra_context=extra_context,
            history=history)
        try:
            async for ev in agen:
                out_q.put(ev)
        except Exception as e:  # noqa: BLE001
            out_q.put({"phase": "error", "error": str(e),
                       "stage": "agent_workflow"})
        finally:
            out_q.put(_DONE)
            # 异步生成器尚未耗尽（取消/异常）时显式关闭，
            # 触发其内部 finally（关闭底层 LLM 流）
            with contextlib.suppress(Exception):
                await agen.aclose()

    def _run_loop():
        asyncio.set_event_loop(loop)
        task_box["task"] = loop.create_task(_drive())
        try:
            loop.run_until_complete(task_box["task"])
        except (Exception, asyncio.CancelledError):  # noqa: BLE001 - 已作为 error 事件转发
            # CancelledError 继承自 BaseException：消费端断开导致的任务取消不算错误
            pass
        finally:
            loop.close()

    threading.Thread(target=_run_loop, daemon=True,
                     name="agent-workflow-loop").start()
    try:
        while True:
            item = out_q.get()
            if item is _DONE:
                break
            yield item
    finally:
        # 消费端提前关闭（客户端断开）：取消异步任务，停止后端 LLM 调用
        task = task_box.get("task")
        if task is not None and not task.done():
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(task.cancel)
