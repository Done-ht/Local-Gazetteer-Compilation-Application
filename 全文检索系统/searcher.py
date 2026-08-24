"""跨库并行搜索引擎。

在多个数据存储区（库）之间并行执行搜索，合并结果。
每个库独立查询（各自的 ZoneIndex/mmap/LRU 缓存互不干扰），用线程池并行。

结果格式设计（兼顾人类可读与程序化处理）：
{
    "query": "搜索词",
    "parallel": 4,
    "total_hits": 207,               # 命中 chunk 总数
    "searched_libraries": [          # 每个库的统计
        {"name": "郎溪县志", "note": "...", "hits": 207, "elapsed_ms": 12.3}
    ],
    "results": [                     # 命中详情，按相关度（命中词数）降序
        {
            "library": "郎溪县志",
            "library_note": "...",
            "chunk_id": "zone_001/chunk_000001",
            "source_file": "引子.docx",
            "source_sha256": "...",
            "matched_words": ["郎", "溪"],
            "hit_count": 2,
            "text_offset": 0,
            "snippet": "...引子 郎溪是..."
        }
    ]
}
"""
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any

from userdata import auth_base_dir as _auth_base_dir
from library import Library, LibraryRegistry
from metadata_stats import pick_dominant
from storage import ZoneManager
from indexer import MultiZoneIndex, ZoneIndex


# 中文连续字串正则（用于提取查询词组）
_HAN_PHRASE_RE = re.compile(r'[\u4e00-\u9fff]+')

# 括号组内词项分隔符：中英文逗号、顿号、竖线
_GROUP_SEPS = set("，,、|")
# 引号对（组内引号包裹的词项作为原子，引号内的分隔符不生效）
_QUOTE_PAIRS = {'"': '"', "'": "'", "“": "”", "『": "』", "「": "」"}


def _split_group_terms(content: str) -> List[str]:
    """切分括号组内的词项，支持引号原子与多分隔符。

    规则：
      - 分隔符：中英文逗号、顿号、竖线（(死，崩) / (死、崩) / (死|崩) 等价）
      - 引号原子：("5,40"，"1.5万") → ['"5,40"', '"1.5万"']，
        引号内的分隔符不切分。词项保留引号传给索引层（ZoneIndex.search
        把引号段作为整段精确短语匹配），显示层自行剥引号
    """
    terms: List[str] = []
    buf: List[str] = []
    open_q = ""      # 当前词项的起始引号（空 = 非引号词项）
    close_quote = None
    for ch in content:
        if close_quote is not None:
            if ch == close_quote:
                # 引号闭合：词项到此为止，紧随其后的内容另起一词项
                t = "".join(buf).strip()
                if t:
                    terms.append(f"{open_q}{t}{_QUOTE_PAIRS[open_q]}")
                buf = []
                open_q = ""
                close_quote = None
            else:
                buf.append(ch)
        elif ch in _QUOTE_PAIRS:
            if not open_q and not buf:
                open_q = ch
            close_quote = _QUOTE_PAIRS[ch]
        elif ch in _GROUP_SEPS:
            t = "".join(buf).strip()
            if t:
                terms.append(f"{open_q}{t}{_QUOTE_PAIRS.get(open_q, '')}" if open_q else t)
            buf = []
            open_q = ""
        else:
            buf.append(ch)
    t = "".join(buf).strip()
    if t:
        terms.append(f"{open_q}{t}{_QUOTE_PAIRS.get(open_q, '')}" if open_q else t)
    return terms


def _title_haystack(heading: str, source_file: str) -> str:
    """标题匹配范围：chunk 标题(heading) + 来源文件名。"""
    return f"{heading or ''}\t{source_file or ''}"


def apply_title_filter(results: List[Dict], title_groups: List[List[str]]) -> List[Dict]:
    """按标题组过滤检索结果（{} 语法）。

    组内任一词出现在 heading 或来源文件名中即该组通过；所有组都通过才保留。
    只在已召回的结果上做后过滤，代价 O(结果数)，无额外扫描。
    """
    if not title_groups:
        return results
    out = []
    for r in results:
        hay = _title_haystack(r.get("heading", ""), r.get("source_file", ""))
        if all(any(w in hay for w in group) for group in title_groups):
            out.append(r)
    return out


# 标题缓存文件名（库根目录级）
_TITLES_CACHE_FILE = "_titles.json"


def _load_title_cache(lib: Library, base_dir: str) -> List[Dict]:
    """加载库级标题缓存 [{chunk_id, heading, file_name, library}]。

    缓存按 zone 数 + chunk 数校验失效；缺失或过期时全量扫描 chunk 文件重建。
    标题只占数据的一小部分，缓存体量很小（每 chunk 仅 chunk_id + heading），
    首次构建一次 O(全部 chunk)，之后命中缓存为 O(1)。
    """
    mgr = lib.manager(base_dir)
    s = mgr.stats()
    cache_path = os.path.join(mgr.root, _TITLES_CACHE_FILE)
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if (cached.get("zone_count") == s["zone_count"]
                    and cached.get("total_chunks") == s["total_chunks"]
                    and cached.get("total_chars") == s["total_chars"]):
                return cached.get("titles", [])
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            pass

    titles: List[Dict] = []
    for zone in mgr.list_zones():
        for path in zone.iter_chunk_files():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    chunk = json.load(f)
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            src = chunk.get("source", {}) or {}
            titles.append({
                "chunk_id": chunk.get("chunk_id", ""),
                "heading": chunk.get("heading", "") or "",
                "file_name": src.get("file_name", "") or "",
                "zone_id": zone.zone_id,
            })
    try:
        tmp = cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"zone_count": s["zone_count"],
                       "total_chunks": s["total_chunks"],
                       "total_chars": s["total_chars"],
                       "titles": titles}, f, ensure_ascii=False)
        os.replace(tmp, cache_path)
    except OSError:
        pass  # 缓存写失败不影响检索
    return titles


def search_by_titles(
    registry: LibraryRegistry,
    title_groups: List[List[str]],
    base_dir: str,
    library_names: Optional[List[str]] = None,
    top_k: int = 50,
) -> Dict:
    """仅按标题检索（{} 语法单独使用时）：返回标题命中标题组的 chunk 列表。

    每个标题组内任一词出现在 heading 或来源文件名中即该组通过，
    所有组都通过的 chunk 才返回。评分与 agent 的 search_titles 一致：
    连续命中次数 × 100。
    """
    t0 = time.perf_counter()
    all_libs = registry.list_libraries()
    if library_names:
        libs = [l for l in all_libs if l.name in library_names]
    else:
        libs = all_libs

    results: List[Dict] = []
    for lib in libs:
        for t in _load_title_cache(lib, base_dir):
            hay = _title_haystack(t["heading"], t["file_name"])
            matched: List[str] = []
            score = 0
            ok = True
            for group in title_groups:
                group_matched = [w for w in group if w in hay]
                if not group_matched:
                    ok = False
                    break
                matched.extend(group_matched)
                score += sum(hay.count(w) * 100 for w in group_matched)
            if not ok or not matched:
                continue
            results.append({
                "library": lib.name,
                "library_note": lib.note,
                "chunk_id": t["chunk_id"],
                "source_file": t["file_name"],
                "source_file_path": "",
                "heading": t["heading"],
                "matched_words": sorted(set(matched)),
                "hit_count": sum(hay.count(w) for w in set(matched)),
                "snippet": "",
                "phrase_bonus": 0,
                "score": score,
                "title_matched": True,
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[:top_k]
    elapsed = (time.perf_counter() - t0) * 1000
    return {
        "query": " ∩ ".join("(" + "|".join(g) + ")" for g in title_groups),
        "title_groups": title_groups,
        "mode": "title",
        "parallel": 1,
        "total_hits": len(results),
        "searched_libraries": [{"name": l.name, "note": l.note} for l in libs],
        "elapsed_ms": round(elapsed, 2),
        "results": results,
    }


def _extract_metadata(src: dict) -> dict:
    """从 chunk 的 source 子对象提取元数据字段，供前端展示。

    优先使用 chunk 级 stats（chunk 本身的统计特征），
    仅当 chunk 级 stats 缺失时才回退到 doc_stats（文档级汇总）。
    chunk 级 stats 没有 dominant_dynasty/dominant_topic 字段，需从
    dynasty_scores/topic_scores 实时计算最高分项。

    时代字段整合逻辑：
    - 古代文本：dynasty_scores 有值 → 取最高分朝代（如"唐"）
    - 现代文本：dynasty_scores 为空或分数低，modern_era_scores 有值
      → 取最高分时期 + 年代范围（如"改革开放(1980s-)"）
    - 年代补充：若 dominant_decade 有值，作为辅助信息展示
    """
    stats = src.get("stats") or {}
    doc_stats = src.get("doc_stats") or {}

    # 文档级 time_span（"跨时代"标记）：县志/年鉴等通史性文档纵贯多个时代，
    # 单独归入任何朝代都是误判。metadata_stats 在 doc_stats 中标记后，
    # 这里读取并优先展示，避免前端误显示某个单一朝代。
    time_span = doc_stats.get("time_span")

    # chunk 级：从 dynasty_scores/topic_scores 取最高分作为主朝代/主主题
    # 信号弱（分数低于阈值）或前两名难分时不强行贴标签（pick_dominant 返回 None）
    chunk_dynasty_scores = stats.get("dynasty_scores") or {}
    chunk_topic_scores = stats.get("topic_scores") or {}
    chunk_dominant_dynasty = pick_dominant(chunk_dynasty_scores)
    chunk_dominant_topic = pick_dominant(chunk_topic_scores)

    # chunk 级现代时期 + 年代
    chunk_modern_era_scores = stats.get("modern_era_scores") or {}
    chunk_dominant_modern_era = pick_dominant(chunk_modern_era_scores)
    chunk_era_decade_range = (stats.get("modern_era_decade_range") or {}).get(chunk_dominant_modern_era)
    chunk_dominant_decade = stats.get("dominant_decade")

    # 整合"时代"字段：古代朝代优先，现代时期作为补充
    # 优先用 chunk 级 dynasty，其次 doc_stats 的 dominant_dynasty
    # 但若文档级标记为"跨时代"，强制清空朝代，避免误判
    dynasty = None if time_span else (chunk_dominant_dynasty or doc_stats.get("dominant_dynasty"))
    # 现代时期：chunk 级优先，回退文档级
    modern_era = chunk_dominant_modern_era or doc_stats.get("dominant_modern_era")
    era_decade_range = chunk_era_decade_range or doc_stats.get("modern_era_decade_range")
    dominant_decade = chunk_dominant_decade or doc_stats.get("dominant_decade")

    # 冲突仲裁：朝代和现代时期同时检测到时，比较分数决定取舍
    # 现代方志/年鉴中提到"乾隆年间"等历史词汇是正常引用，
    # 但如果现代时期关键词命中频率更高，说明是现代文档
    if dynasty and modern_era:
        # 获取分数：优先 chunk 级，回退 doc_stats
        chunk_dyn_score = (stats.get("dynasty_scores") or {}).get(chunk_dominant_dynasty, 0)
        doc_dyn_score = (doc_stats.get("dynasty_scores") or {}).get(dynasty, 0)
        dynasty_score = chunk_dyn_score or doc_dyn_score

        chunk_mod_score = (stats.get("modern_era_scores") or {}).get(chunk_dominant_modern_era, 0)
        doc_mod_score = (doc_stats.get("modern_era_scores") or {}).get(modern_era, 0)
        modern_score = chunk_mod_score or doc_mod_score

        # 现代时期分数 >= 朝代分数的 1.5 倍 → 判定为现代文本
        if modern_score > 0 and modern_score >= dynasty_score * 1.5:
            dynasty = None

    # 构造时代展示文本
    # 优先级：跨时代标记 > 朝代 > 现代时期 > 年代
    if time_span:
        # 跨时代文档：直接显示"跨时代"，不展示单一朝代
        era_label = time_span
    elif dynasty:
        # 古代文本：显示朝代名
        era_label = dynasty
    elif modern_era:
        # 现代文本：显示时期 + 年代范围
        era_label = f"{modern_era}({era_decade_range})" if era_decade_range else modern_era
    elif dominant_decade:
        # 仅有年代信息：显示主年代
        era_label = dominant_decade
    else:
        era_label = None

    topic = chunk_dominant_topic or doc_stats.get("dominant_topic")
    era_names = stats.get("era_names") or doc_stats.get("era_names") or []
    # 年号必须依附于朝代：没有朝代就没有年号
    # 跨时代文档和现代文本中"中和""普通"等年号多为常用词误识别，置空避免误导
    if not dynasty:
        era_names = []
    entity_density = stats.get("entity_density") or doc_stats.get("entity_density") or {}
    top_persons = stats.get("top_persons") or doc_stats.get("top_persons") or []

    return {
        "metadata": {
            "era": era_label,
            "time_span": time_span,  # 跨时代标记，供前端特殊展示
            "dynasty": dynasty,  # 保留原字段供后端过滤用
            "modern_era": modern_era,
            "decade_range": era_decade_range,
            "dominant_decade": dominant_decade,
            "topic": topic,
            "era_names": era_names,
            "entity_density": entity_density,
            "top_persons": top_persons,
            "relative_dir": src.get("relative_dir") or "",
        } if (stats or doc_stats) else None,
    }


def _compute_phrase_bonus(chunk_text: str, query: str, weight: int = 100) -> int:
    """计算词组连续匹配 bonus。

    对查询中的连续中文字串提取 bigram（2字组合），在 chunk 文本中
    统计连续出现次数。连续匹配越多、越长，bonus 越高。

    例：查询"经济发展" → bigram = ["经济","济发","发展"]
        chunk 有"经济发展"连续 → 3 bigram 全匹配 → bonus=300
        chunk 四字分散出现 → 0 bigram 匹配 → bonus=0

    这样"经济发展"四个字连在一起的 chunk 得分最高，优先级最高。
    """
    bonus = 0
    for phrase in _HAN_PHRASE_RE.findall(query):
        if len(phrase) < 2:
            continue
        # 提取 bigram
        for i in range(len(phrase) - 1):
            bigram = phrase[i:i + 2]
            count = chunk_text.count(bigram)
            if count:
                bonus += count * weight
    return bonus


def _load_chunk_for_snippet(zone_path: str, chunk_id: str, positions: List[int],
                            window: int = 40) -> str:
    """加载 chunk 文本，截取首个命中位置附近的片段。"""
    # chunk_id 形如 "zone_001/chunk_000001"
    try:
        parts = chunk_id.split("/")
        if len(parts) != 2:
            return ""
        zone_id, chunk_name = parts
        chunk_file = f"{chunk_name}.json"
        chunk_path = os.path.join(zone_path, "chunks", chunk_file)
        if not os.path.isfile(chunk_path):
            return ""
        with open(chunk_path, "r", encoding="utf-8") as f:
            chunk = json.load(f)
        text = chunk.get("text", "")
        if not text or not positions:
            return ""
        pos = min(positions)
        start = max(0, pos - window)
        end = min(len(text), pos + window)
        snippet = text[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        return snippet.replace("\n", " ")
    except Exception:
        return ""


def _search_single_library(lib: Library, base_dir: str, query: str,
                            chunk_filter: Optional[set] = None) -> Dict:
    """查询单个库，返回该库的结果 dict。

    chunk_filter: 若提供，只保留该集合中的 chunk_id（用于分块检索）。
    """
    t0 = time.perf_counter()
    mgr = lib.manager(base_dir)
    zones = mgr.list_zones()
    result: Dict[str, Dict[str, List[int]]] = {}

    for z in zones:
        # 使用缓存实例，避免重复 new + 读 manifest
        zi = ZoneIndex.get(z.index_dir)
        # 搜索是只读操作，不再每次触发全量 merge
        # 仅当索引目录完全为空（首次使用）时才构建一次
        if not os.path.exists(zi.manifest_path):
            zi.merge_zone_chunks(z.chunks_dir, z.zone_id)
        zi.ensure_offset_index(z.chunks_dir, z.zone_id)
        res = zi.search(query)
        for word, chunk_map in res.items():
            for cid, positions in chunk_map.items():
                # 分块过滤：只保留指定范围内的 chunk
                if chunk_filter is not None and cid not in chunk_filter:
                    continue
                result.setdefault(cid, {}).setdefault(word, []).extend(positions)

    elapsed = (time.perf_counter() - t0) * 1000

    # 构建 chunk 级结果
    chunk_results = []
    for cid, word_map in result.items():
        if not word_map:
            continue
        # 尝试定位 chunk 文件以取 source_file / snippet / phrase_bonus
        source_file = ""
        source_file_path = ""
        source_sha = ""
        text_offset = 0
        heading = ""
        snippet = ""
        phrase_bonus = 0
        all_positions: List[int] = []
        try:
            parts = cid.split("/")
            if len(parts) == 2:
                zone_id, chunk_name = parts
                zone = mgr.get_zone(zone_id)
                chunk_path = os.path.join(zone.chunks_dir, f"{chunk_name}.json")
                if os.path.isfile(chunk_path):
                    with open(chunk_path, "r", encoding="utf-8") as f:
                        chunk = json.load(f)
                    src = chunk.get("source", {})
                    source_file = src.get("file_name", "")
                    source_file_path = src.get("file_path", "")
                    source_sha = src.get("source_sha256", "")
                    text_offset = chunk.get("text_offset", 0)
                    heading = chunk.get("heading", "") or ""
                    for ps in word_map.values():
                        all_positions.extend(ps)
                    snippet = _load_chunk_for_snippet(
                        zone.path, cid, all_positions
                    )
                    # 计算词组连续匹配 bonus（连续词组得分最高）
                    chunk_text = chunk.get("text", "") or ""
                    phrase_bonus = _compute_phrase_bonus(chunk_text, query)
        except Exception:
            pass

        hit_count = sum(len(v) for v in word_map.values())
        chunk_results.append({
            "library": lib.name,
            "library_note": lib.note,
            "chunk_id": cid,
            "source_file": source_file,
            "source_file_path": source_file_path,
            "source_sha256": source_sha,
            "heading": heading,
            "matched_words": sorted(word_map.keys()),
            "hit_count": hit_count,
            "phrase_bonus": phrase_bonus,
            "score": hit_count + phrase_bonus,  # 综合分数：命中数 + 词组 bonus
            "text_offset": text_offset,
            "snippet": snippet,
            **_extract_metadata(src),
        })

    # 按综合分数降序（词组连续匹配优先，其次命中数）
    chunk_results.sort(key=lambda x: (x["score"], x["hit_count"]), reverse=True)

    return {
        "library": lib.name,
        "note": lib.note,
        "hits": len(chunk_results),
        "elapsed_ms": round(elapsed, 2),
        "results": chunk_results,
    }


def parallel_search(
    registry: LibraryRegistry,
    query: str,
    library_names: Optional[List[str]] = None,
    parallel: int = 4,
    base_dir: str = ".",
    chunk_filter: Optional[set] = None,
) -> Dict:
    """跨库并行搜索。

    library_names: 指定查询的库名列表；None 表示查全部库。
    parallel: 最大并行度。
    chunk_filter: 若提供，只保留该集合中的 chunk_id（分块检索时用）。
    返回合并后的结构化结果。
    """
    all_libs = registry.list_libraries()
    if library_names:
        libs = [l for l in all_libs if l.name in library_names]
        missing = set(library_names) - {l.name for l in libs}
        if missing:
            raise ValueError(f"库不存在: {sorted(missing)}")
    else:
        libs = all_libs

    if not libs:
        return {
            "query": query,
            "parallel": parallel,
            "total_hits": 0,
            "searched_libraries": [],
            "results": [],
        }

    # 并行查询
    lib_results: List[Dict] = []
    if len(libs) == 1 or parallel <= 1:
        for lib in libs:
            lib_results.append(_search_single_library(lib, base_dir, query, chunk_filter=chunk_filter))
    else:
        with ThreadPoolExecutor(max_workers=min(parallel, len(libs))) as pool:
            futures = {
                pool.submit(_search_single_library, lib, base_dir, query, chunk_filter): lib
                for lib in libs
            }
            # 保持库的注册顺序
            order = {lib.name: i for i, lib in enumerate(libs)}
            for fut in as_completed(futures):
                res = fut.result()
                lib_results.append(res)
            lib_results.sort(key=lambda r: order.get(r["library"], 0))

    # 合并（按综合分数排序：词组连续匹配优先）
    all_results: List[Dict] = []
    for lr in lib_results:
        all_results.extend(lr["results"])
    all_results.sort(key=lambda x: (x.get("score", x["hit_count"]), x["hit_count"]), reverse=True)

    return {
        "query": query,
        "parallel": parallel,
        "total_hits": len(all_results),
        "searched_libraries": [
            {
                "name": lr["library"],
                "note": lr["note"],
                "hits": lr["hits"],
                "elapsed_ms": lr["elapsed_ms"],
            }
            for lr in lib_results
        ],
        "results": all_results,
    }


# ============================================================
#  分块检索（大数据量优化）
# ============================================================

# 分块触发阈值：单库 chunk 总数超过此值才分块
PARTITION_THRESHOLD = 1800
# 每块最大 chunk 数（保留用于推算，实际块数由梯度表决定）
PARTITION_MAX_CHUNKS = 600
# 最多分块数
PARTITION_MAX_PARTS = 6

# 梯度分块表：[(chunk数上限, 块数), ...]
# 含义：chunk 数落在 (上一档, 本档] 时使用对应块数
# 默认 6 档：
#   ≤1800        → 1 块（不分）
#   1801-3000    → 2 块
#   3001-6000    → 3 块
#   6001-12000   → 4 块
#   12001-24000  → 5 块
#   >24000       → 6 块
PARTITION_GRADIENT = [
    (1800, 1),
    (3000, 2),
    (6000, 3),
    (12000, 4),
    (24000, 5),
    (10**9, 6),  # 兜底
]


def _load_partition_gradient():
    """从 settings 加载梯度表（可配置）。

    settings 中可配置：
      partition_threshold: 触发阈值（默认 1800）
      partition_max_parts: 最大块数（默认 6）
      partition_gradient: 自定义梯度表（JSON 列表，默认 None 用内置）
    """
    try:
        from settings import SettingsStore
        store = SettingsStore(_auth_base_dir())
        custom = store.get("partition_gradient", None)
        if custom and isinstance(custom, list) and len(custom) >= 2:
            # 校验并转换
            grad = []
            for item in custom:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    threshold, parts = int(item[0]), int(item[1])
                    grad.append((threshold, parts))
            if len(grad) >= 2:
                # 按 threshold 升序
                grad.sort(key=lambda x: x[0])
                return grad
        max_parts = store.get("partition_max_parts", PARTITION_MAX_PARTS)
        threshold = store.get("partition_threshold", PARTITION_THRESHOLD)
        # 用默认梯度但应用 max_parts 上限
        grad = [(threshold, 1)] + [
            (t, min(p, max_parts)) for t, p in PARTITION_GRADIENT[1:] if t > threshold
        ]
        # 兜底
        if not any(t == 10**9 for t, _ in grad):
            grad.append((10**9, max_parts))
        return grad
    except Exception:
        return PARTITION_GRADIENT


def _list_library_chunk_ids(lib: Library, base_dir: str) -> List[str]:
    """获取库内所有 chunk_id，按 (zone_id, chunk_name) 排序。

    chunk_id 形如 "zone_001/chunk_000123"，排序后跨 zone 拉平。
    """
    mgr = lib.manager(base_dir)
    ids: List[str] = []
    for z in mgr.list_zones():
        chunks_dir = z.chunks_dir
        if not os.path.isdir(chunks_dir):
            continue
        for fname in os.listdir(chunks_dir):
            if fname.endswith(".json"):
                # fname = "chunk_000123.json" → chunk_name = "chunk_000123"
                chunk_name = fname[:-5]
                ids.append(f"{z.zone_id}/{chunk_name}")
    # 跨 zone 拉平排序：按 zone_id 再按 chunk_name
    ids.sort()
    return ids


def _partition_chunk_ids(chunk_ids: List[str]) -> List[set]:
    """把 chunk_id 列表按梯度均分成若干份。

    规则：
      - 按梯度表决定块数（≤1800→1, ≤3000→2, ≤6000→3, ≤12000→4, ≤24000→5, >24000→6）
      - 连续切分（不跳着），按排序后的 chunk_id 均分
      - 块数从 settings.partition_gradient 读取（可配置）
    """
    n = len(chunk_ids)
    grad = _load_partition_gradient()
    # 找到第一个 threshold >= n 的档位
    parts = 1
    for threshold, p in grad:
        if n <= threshold:
            parts = p
            break
    if parts <= 1:
        return [set(chunk_ids)]

    # 均分
    size = (n + parts - 1) // parts
    result = []
    for i in range(parts):
        start = i * size
        end = min(start + size, n)
        result.append(set(chunk_ids[start:end]))
    return result


def parallel_search_partitioned(
    registry: LibraryRegistry,
    query: str,
    library_names: Optional[List[str]] = None,
    parallel: int = 4,
    base_dir: str = ".",
) -> Dict:
    """分块并行搜索。

    对每个库：
      1. 获取所有 chunk_id 并排序
      2. 总数 > 1800 → 按 chunk_id 均分（每块 <600，最多 4 块）
      3. 每块独立检索（用 chunk_filter 过滤）
      4. 库内合并重排

    跨库：每库独立分块，结果合并。

    返回结构与 parallel_search 一致，额外增加 partitions 字段描述分块信息。
    """
    all_libs = registry.list_libraries()
    if library_names:
        libs = [l for l in all_libs if l.name in library_names]
        missing = set(library_names) - {l.name for l in libs}
        if missing:
            raise ValueError(f"库不存在: {sorted(missing)}")
    else:
        libs = all_libs

    if not libs:
        return {
            "query": query,
            "parallel": parallel,
            "total_hits": 0,
            "searched_libraries": [],
            "partitions": [],
            "results": [],
        }

    # 为每个库计算分块
    lib_partitions: List[tuple] = []  # (lib, [chunk_filter_set, ...])
    partition_info: List[Dict] = []
    for lib in libs:
        chunk_ids = _list_library_chunk_ids(lib, base_dir)
        parts = _partition_chunk_ids(chunk_ids)
        lib_partitions.append((lib, parts))
        partition_info.append({
            "library": lib.name,
            "total_chunks": len(chunk_ids),
            "partition_count": len(parts),
            "partition_sizes": [len(p) for p in parts],
        })

    # 并行检索各分块
    # 任务粒度：(lib, chunk_filter) → 结果
    tasks: List[tuple] = []  # (lib_name, lib, chunk_filter)
    for lib, parts in lib_partitions:
        for pf in parts:
            tasks.append((lib.name, lib, pf))

    all_lib_results: Dict[str, List[Dict]] = {}  # lib_name → [分块结果...]
    if parallel <= 1 or len(tasks) == 1:
        for lib_name, lib, pf in tasks:
            r = _search_single_library(lib, base_dir, query, chunk_filter=pf)
            all_lib_results.setdefault(lib_name, []).append(r)
    else:
        with ThreadPoolExecutor(max_workers=min(parallel, len(tasks))) as pool:
            futures = {
                pool.submit(_search_single_library, lib, base_dir, query, pf): (lib_name, lib, pf)
                for lib_name, lib, pf in tasks
            }
            for fut in as_completed(futures):
                lib_name, lib, pf = futures[fut]
                r = fut.result()
                all_lib_results.setdefault(lib_name, []).append(r)

    # 按库的注册顺序合并各分块结果
    order = {lib.name: i for i, lib in enumerate(libs)}
    lib_merged: List[Dict] = []
    for lib in libs:
        parts_results = all_lib_results.get(lib.name, [])
        if not parts_results:
            continue
        # 合并各分块的 results
        merged_results: List[Dict] = []
        total_elapsed = 0.0
        for pr in parts_results:
            merged_results.extend(pr["results"])
            total_elapsed += pr["elapsed_ms"]
        # 库内重排
        merged_results.sort(
            key=lambda x: (x.get("score", x["hit_count"]), x["hit_count"]), reverse=True
        )
        lib_merged.append({
            "library": lib.name,
            "note": lib.note,
            "hits": len(merged_results),
            "elapsed_ms": round(total_elapsed, 2),
            "results": merged_results,
        })
    lib_merged.sort(key=lambda r: order.get(r["library"], 0))

    # 跨库合并
    all_results: List[Dict] = []
    for lr in lib_merged:
        all_results.extend(lr["results"])
    all_results.sort(key=lambda x: (x.get("score", x["hit_count"]), x["hit_count"]), reverse=True)

    return {
        "query": query,
        "parallel": parallel,
        "total_hits": len(all_results),
        "searched_libraries": [
            {
                "name": lr["library"],
                "note": lr["note"],
                "hits": lr["hits"],
                "elapsed_ms": lr["elapsed_ms"],
            }
            for lr in lib_merged
        ],
        "partitions": partition_info,
        "results": all_results,
    }


def format_search_result(result: Dict, top_k: int = 20, show_snippet: bool = True) -> str:
    """把搜索结果格式化成人类可读文本。"""
    lines: List[str] = []
    query = result["query"]
    total = result["total_hits"]
    libs = result["searched_libraries"]
    lib_names = ", ".join(l["name"] for l in libs) or "无"
    lines.append(f"[搜索] 查询: {query}")
    lines.append(f"[搜索] 并行度: {result['parallel']}, 查询库: {lib_names}")
    lines.append(f"[搜索] 总命中: {total} chunk")

    if libs:
        lines.append("")
        lines.append("=== 库统计 ===")
        for l in libs:
            lines.append(
                f"  {l['name']} ({l['note']}): "
                f"{l['hits']} chunk, {l['elapsed_ms']:.1f}ms"
            )

    if result["results"]:
        lines.append("")
        lines.append(f"=== 命中详情 (前 {min(top_k, len(result['results']))} 条) ===")
        for i, r in enumerate(result["results"][:top_k], 1):
            lines.append(f"  {i}. [{r['library']}] {r['chunk_id']}")
            lines.append(f"       源文件: {r['source_file']}")
            words = ", ".join(r["matched_words"])
            lines.append(f"       命中词: {words} ({r['hit_count']} 处)")
            lines.append(f"       偏移: {r['text_offset']}")
            if show_snippet and r.get("snippet"):
                lines.append(f"       片段: {r['snippet']}")

    return "\n".join(lines)


# ============================================================
#  多关键词共现查询（AND / N-of-M / 加权共现）
# ============================================================

def _compute_window_tightness(positions_per_query: Dict[str, List[int]]) -> int:
    """计算多关键词在同一 chunk 中的窗口紧密度。

    positions_per_query: {query_word: [位置列表]}
    返回紧密得分：关键词位置最大间距越小，得分越高。

    例：查询 [张飞, 卒]
        chunk1: 张飞@100, 卒@105 → 间距 5 → 高分
        chunk2: 张飞@100, 卒@5000 → 间距 4900 → 低分
    """
    if len(positions_per_query) < 2:
        return 0
    # 取每个查询词的首次出现位置
    first_positions = []
    for q, ps in positions_per_query.items():
        if not ps:
            return 0  # 缺一个查询词，紧密度为 0
        first_positions.append(min(ps))
    if len(first_positions) < 2:
        return 0
    first_positions.sort()
    # 窗口跨度 = 最大位置 - 最小位置
    span = first_positions[-1] - first_positions[0]
    # 跨度越小分越高：跨度 0 → 1000，跨度 100 → 900，跨度 1000 → 0
    return max(0, 1000 - span * 10)


def search_multi_keywords(
    registry: LibraryRegistry,
    queries: List[str],
    base_dir: str,
    library_names: Optional[List[str]] = None,
    parallel: int = 4,
    mode: str = "weighted",
    min_match: Optional[int] = None,
    chunk_filter: Optional[set] = None,
    top_k: int = 50,
) -> Dict:
    """多关键词共现查询：要求 chunk 同时包含多个关键词。

    Args:
        queries: 关键词列表（如 ["张飞", "卒", "三国志"]）
        mode: 检索模式
            - "and": 要求 chunk 同时包含全部关键词
            - "n_of_m": 要求 chunk 至少包含 min_match 个关键词
            - "weighted": 不强制阈值，按共现数²加权排序（默认）
        min_match: n_of_m 模式下的最少命中数；默认 ceil(len(queries)*0.6)
        chunk_filter: 若提供，只保留该集合中的 chunk_id
        top_k: 返回前 N 条结果

    Returns:
        {
            "query": "张飞 AND 卒 AND 三国志",
            "mode": "weighted",
            "total_hits": N,
            "results": [...],  # 每条多一个 cooccur_count 和 cooccur_score 字段
        }
    """
    import math
    t0 = time.perf_counter()
    queries = [q for q in queries if q and q.strip()]
    if not queries:
        return {"query": "", "mode": mode, "total_hits": 0, "results": [], "elapsed_ms": 0}
    if len(queries) == 1:
        # 单关键词退化为普通搜索
        result = parallel_search(registry, queries[0],
                                  library_names=library_names,
                                  parallel=parallel, base_dir=base_dir)
        result["mode"] = mode
        result["cooccur_query"] = queries
        return result

    # 默认 n_of_m 阈值
    if min_match is None:
        min_match = max(2, math.ceil(len(queries) * 0.6))
    min_match = max(1, min(min_match, len(queries)))

    # 各库并行检索每个关键词，然后合并
    all_libs = registry.list_libraries()
    if library_names:
        libs = [l for l in all_libs if l.name in library_names]
    else:
        libs = all_libs

    # 每个 chunk 的累积命中信息：cid -> {query_word: [positions], ...}
    # 同时记录 chunk 元信息
    chunk_data: Dict[str, Dict] = {}  # cid -> {lib, source_file, heading, positions_per_query}

    def _search_one_lib(lib: Library):
        """对单库执行多关键词检索，返回 {cid: {query: [positions]}}"""
        mgr = lib.manager(base_dir)
        zones = mgr.list_zones()
        lib_result: Dict[str, Dict] = {}
        for z in zones:
            zi = ZoneIndex.get(z.index_dir)
            if not os.path.exists(zi.manifest_path):
                zi.merge_zone_chunks(z.chunks_dir, z.zone_id)
            zi.ensure_offset_index(z.chunks_dir, z.zone_id)
            for q in queries:
                res = zi.search(q)
                for word, chunk_map in res.items():
                    for cid, positions in chunk_map.items():
                        if chunk_filter is not None and cid not in chunk_filter:
                            continue
                        if cid not in lib_result:
                            lib_result[cid] = {
                                "lib": lib,
                                "zone": z,
                                "positions_per_query": {},
                                "matched_words": set(),
                            }
                        lib_result[cid]["positions_per_query"].setdefault(q, []).extend(positions)
                        lib_result[cid]["matched_words"].add(word)
        return lib_result

    # 并行各库
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        futures = {pool.submit(_search_one_lib, lib): lib for lib in libs}
        for fut in as_completed(futures):
            try:
                lib_res = fut.result()
            except Exception:
                continue
            for cid, data in lib_res.items():
                if cid not in chunk_data:
                    chunk_data[cid] = data
                else:
                    # 合并（同一 chunk 跨库理论上不会出现，但保险起见）
                    for q, ps in data["positions_per_query"].items():
                        chunk_data[cid]["positions_per_query"].setdefault(q, []).extend(ps)
                    chunk_data[cid]["matched_words"] |= data["matched_words"]

    # 计算每个 chunk 的共现数与得分
    results: List[Dict] = []
    for cid, data in chunk_data.items():
        ppq = data["positions_per_query"]
        cooccur_count = len(ppq)  # 共现关键词数

        # 模式过滤
        if mode == "and" and cooccur_count < len(queries):
            continue
        if mode == "n_of_m" and cooccur_count < min_match:
            continue
        # weighted 模式无过滤

        # 加载 chunk 文本以计算 phrase_bonus 和 snippet
        lib = data["lib"]
        zone = data["zone"]
        chunk_text = ""
        source_file = ""
        source_file_path = ""
        source_sha = ""
        text_offset = 0
        heading = ""
        snippet = ""
        try:
            parts = cid.split("/")
            if len(parts) == 2:
                chunk_name = parts[1]
                chunk_path = os.path.join(zone.chunks_dir, f"{chunk_name}.json")
                if os.path.isfile(chunk_path):
                    with open(chunk_path, "r", encoding="utf-8") as f:
                        chunk = json.load(f)
                    chunk_text = chunk.get("text", "") or ""
                    src = chunk.get("source", {})
                    source_file = src.get("file_name", "")
                    source_file_path = src.get("file_path", "")
                    source_sha = src.get("source_sha256", "")
                    text_offset = chunk.get("text_offset", 0)
                    heading = chunk.get("heading", "") or ""
        except Exception:
            pass

        # hit_count = 所有关键词命中次数总和
        hit_count = sum(len(ps) for ps in ppq.values())
        # phrase_bonus：用所有查询词拼接计算（连续词组 bonus）
        phrase_bonus = 0
        if chunk_text:
            joined_query = " ".join(queries)
            phrase_bonus = _compute_phrase_bonus(chunk_text, joined_query)
        # snippet：取首个关键词的首个位置附近
        all_positions = []
        for ps in ppq.values():
            all_positions.extend(ps)
        if all_positions:
            snippet = _load_chunk_for_snippet(zone.path, cid, all_positions)

        # 共现得分：共现数² × 1000 + 窗口紧密度 + phrase_bonus*3 + hit_count
        cooccur_score = (
            cooccur_count * cooccur_count * 1000
            + _compute_window_tightness(ppq)
            + phrase_bonus * 3
            + hit_count
        )

        results.append({
            "library": lib.name,
            "library_note": lib.note,
            "chunk_id": cid,
            "source_file": source_file,
            "source_file_path": source_file_path,
            "source_sha256": source_sha,
            "matched_words": sorted(data["matched_words"]),
            "matched_queries": list(ppq.keys()),
            "hit_count": hit_count,
            "text_offset": text_offset,
            "heading": heading,
            "snippet": snippet,
            "phrase_bonus": phrase_bonus,
            "cooccur_count": cooccur_count,
            "cooccur_score": cooccur_score,
            "score": cooccur_score,  # 兼容下游排序
            **_extract_metadata(src),
        })

    # 按共现得分降序
    results.sort(key=lambda x: (x["cooccur_score"], x["cooccur_count"], x["hit_count"]), reverse=True)
    results = results[:top_k]

    elapsed = (time.perf_counter() - t0) * 1000
    query_str = " AND ".join(queries) if mode == "and" else " ∩ ".join(queries)

    return {
        "query": query_str,
        "cooccur_query": queries,
        "mode": mode,
        "min_match": min_match,
        "parallel": parallel,
        "total_hits": len(results),
        "searched_libraries": [{"name": l.name, "note": l.note} for l in libs],
        "elapsed_ms": round(elapsed, 2),
        "results": results,
    }


# ============================================================
#  关联词检索（同义词组 + 跨组共现加分）
# ============================================================

def search_related_keywords(
    registry: LibraryRegistry,
    groups: List[Any],
    base_dir: str,
    library_names: Optional[List[str]] = None,
    parallel: int = 4,
    chunk_filter: Optional[set] = None,
    top_k: int = 50,
) -> Dict:
    """关联词检索：每个 group 是一组同义词（组内任一命中都算该组命中），
    跨组共现越多得分越高。

    评分规则（满足"任意两组同时出现加分"）：
        基础分   = 共现组数² × 1000
        组对加成 = C(共现组数, 2) × 5000   # 任意两组共现各加 5000
        窗口紧密度 = 按各 group 首次命中位置计算的紧密度
        phrase_bonus = 所有同义词连续匹配 bonus
        hit_count = 所有同义词命中次数总和

    必需组（required）规则：
        - groups 元素支持 dict 格式 {"words":[...], "required":bool, "label":str}
        - 也兼容旧格式 ["w1","w2"]（默认 should）
        - chunk 必须命中所有 required 组才算相关，否则直接丢弃
        - 这样可避免跨朝代混入：required 组放问题主体（如刘备），should 组放同义动词（如死/崩）

    Args:
        groups: 同义词组列表，元素可为 dict 或 list，例如
            [
              {"words":["刘备","先主","先帝"], "required":true, "label":"问题主体"},
              {"words":["死","崩","卒"], "required":false, "label":"事件概念"}
            ]
            长度 1 时退化为单组检索（组内任一命中即可）。
        chunk_filter: 若提供，只保留该集合中的 chunk_id。
        top_k: 返回前 N 条。

    Returns:
        {
            "query": "刘备|先主|先帝 ∩ 死|崩|卒",
            "groups": [...],  # 标准化后的 groups（dict 格式）
            "total_hits": N,
            "results": [...],  # 每条含 group_hits / cooccur_groups / related_score / required_hit
            "required_indices": [0, ...],  # required 组下标列表（前端展示用）
            "filtered_by_required": N,  # 因未命中 required 组被丢弃的 chunk 数
        }
    """
    import math
    t0 = time.perf_counter()
    # 清洗：去空、去组内重复；标准化为 dict 格式
    cleaned_groups: List[Dict[str, Any]] = []
    for g in groups:
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
        seen = set()
        gs = []
        for w in words:
            w = (w or "").strip() if isinstance(w, str) else ""
            if w and w not in seen:
                seen.add(w)
                gs.append(w)
        if gs:
            cleaned_groups.append({"words": gs, "required": required, "label": label})

    if not cleaned_groups:
        return {"query": "", "groups": [], "total_hits": 0, "results": [],
                "searched_libraries": [], "elapsed_ms": 0,
                "required_indices": [], "filtered_by_required": 0}

    # 提取纯词列表供底层检索
    word_groups: List[List[str]] = [g["words"] for g in cleaned_groups]
    required_indices: List[int] = [i for i, g in enumerate(cleaned_groups) if g["required"]]

    if len(word_groups) == 1:
        # 单组：组内任一命中即算，退化为合并 OR 检索（仍按组内共现数排序）
        result = parallel_search(registry, " ".join(word_groups[0]),
                                  library_names=library_names,
                                  parallel=parallel, base_dir=base_dir)
        result["groups"] = cleaned_groups
        result["mode"] = "related_single"
        result["required_indices"] = required_indices
        result["filtered_by_required"] = 0
        return result

    all_libs = registry.list_libraries()
    if library_names:
        libs = [l for l in all_libs if l.name in library_names]
    else:
        libs = all_libs

    # 每个 chunk 的累积命中信息：cid -> {lib, zone, group_positions: {gi: [positions]}, matched_words: set}
    chunk_data: Dict[str, Dict] = {}
    # 各库命中统计（前端展示用）
    lib_hits: Dict[str, int] = {l.name: 0 for l in libs}
    lib_elapsed: Dict[str, float] = {l.name: 0.0 for l in libs}
    filtered_by_required = 0  # 因未命中 required 组被丢弃的 chunk 数

    def _search_one_lib(lib: Library):
        """对单库执行多组关联检索。"""
        _t0 = time.perf_counter()
        mgr = lib.manager(base_dir)
        zones = mgr.list_zones()
        lib_result: Dict[str, Dict] = {}
        for z in zones:
            zi = ZoneIndex.get(z.index_dir)
            if not os.path.exists(zi.manifest_path):
                zi.merge_zone_chunks(z.chunks_dir, z.zone_id)
            zi.ensure_offset_index(z.chunks_dir, z.zone_id)
            # 对每个 group 的每个同义词独立检索
            for gi, synonyms in enumerate(word_groups):
                for syn in synonyms:
                    res = zi.search(syn)
                    for word, chunk_map in res.items():
                        for cid, positions in chunk_map.items():
                            if chunk_filter is not None and cid not in chunk_filter:
                                continue
                            if cid not in lib_result:
                                lib_result[cid] = {
                                    "lib": lib,
                                    "zone": z,
                                    "group_positions": {},  # {gi: [positions]}
                                    "matched_words": set(),
                                    "group_matched_synonyms": {},  # {gi: set(synonyms)}
                                }
                            lib_result[cid]["group_positions"].setdefault(gi, []).extend(positions)
                            lib_result[cid]["matched_words"].add(word)
                            lib_result[cid]["group_matched_synonyms"].setdefault(gi, set()).add(syn)
        return lib_result

    # 并行各库
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        futures = {pool.submit(_search_one_lib, lib): lib for lib in libs}
        for fut in as_completed(futures):
            try:
                lib_res = fut.result()
            except Exception:
                continue
            for cid, data in lib_res.items():
                if cid not in chunk_data:
                    chunk_data[cid] = data
                else:
                    for gi, ps in data["group_positions"].items():
                        chunk_data[cid]["group_positions"].setdefault(gi, []).extend(ps)
                    chunk_data[cid]["matched_words"] |= data["matched_words"]
                    for gi, syns in data["group_matched_synonyms"].items():
                        chunk_data[cid]["group_matched_synonyms"].setdefault(gi, set()).update(syns)

    # 计算每个 chunk 的关联得分
    results: List[Dict] = []
    for cid, data in chunk_data.items():
        gp = data["group_positions"]
        n_groups_hit = len(gp)  # 共现组数
        if n_groups_hit == 0:
            continue

        # 必需组检查：所有 required 组必须都命中，否则丢弃
        # 这是过滤跨朝代混入的关键 —— required 组放问题主体（如"刘备"组）
        if required_indices:
            hit_group_indices = set(gp.keys())
            if not all(ri in hit_group_indices for ri in required_indices):
                filtered_by_required += 1
                continue

        lib = data["lib"]
        zone = data["zone"]
        chunk_text = ""
        source_file = ""
        source_file_path = ""
        source_sha = ""
        text_offset = 0
        heading = ""
        snippet = ""
        try:
            parts = cid.split("/")
            if len(parts) == 2:
                chunk_name = parts[1]
                chunk_path = os.path.join(zone.chunks_dir, f"{chunk_name}.json")
                if os.path.isfile(chunk_path):
                    with open(chunk_path, "r", encoding="utf-8") as f:
                        chunk = json.load(f)
                    chunk_text = chunk.get("text", "") or ""
                    src = chunk.get("source", {})
                    source_file = src.get("file_name", "")
                    source_file_path = src.get("file_path", "")
                    source_sha = src.get("source_sha256", "")
                    text_offset = chunk.get("text_offset", 0)
                    heading = chunk.get("heading", "") or ""
        except Exception:
            pass

        # hit_count = 所有同义词命中次数总和
        hit_count = sum(len(ps) for ps in gp.values())

        # phrase_bonus：每个同义词单独计算后取最大值（避免短词噪声）
        phrase_bonus = 0
        if chunk_text:
            for syns in word_groups:
                for syn in syns:
                    pb = _compute_phrase_bonus(chunk_text, syn)
                    if pb > phrase_bonus:
                        phrase_bonus = pb

        # 窗口紧密度：按各 group 首次命中位置计算
        first_positions: List[int] = []
        for gi, ps in gp.items():
            if ps:
                first_positions.append(min(ps))
        first_positions.sort()
        tightness = 0
        if len(first_positions) >= 2:
            span = first_positions[-1] - first_positions[0]
            tightness = max(0, 1000 - span * 10)

        # 计算共现的 should 组数（不含 required 组，用于加分）
        should_hit_count = sum(1 for gi in gp.keys() if gi not in required_indices)

        # 关联得分（收紧版）：
        # - 必需组命中是前提（已过滤），不再额外加分
        # - should 组共现：每多一个 should 组共现，大幅加分（核心相关性信号）
        # - 单 should 组命中（无其他共现）：大幅降权（避免单组高命中混入）
        pair_bonus = math.comb(should_hit_count, 2) * 5000
        related_score = (
            should_hit_count * should_hit_count * 1000   # should 共现组数平方放大
            + pair_bonus                                  # should 组对共现加成
            + tightness                                   # 窗口紧密度
            + phrase_bonus * 3
            + hit_count
        )

        # 若 should 组共现数 = 0（仅命中 required 组），大幅降权
        # —— 这种 chunk 可能只是提到了刘备但与"死亡"无关，相关性低
        if should_hit_count == 0:
            related_score = related_score // 100  # 降到 1/100

        # snippet：取首个 group 首个命中位置附近
        all_positions: List[int] = []
        for ps in gp.values():
            all_positions.extend(ps)
        if all_positions:
            snippet = _load_chunk_for_snippet(zone.path, cid, all_positions)

        # 构造组级命中信息（前端展示用，含 required 标记）
        group_hits = []
        for gi, ginfo in enumerate(cleaned_groups):
            hit_syns = sorted(data["group_matched_synonyms"].get(gi, set()))
            positions = gp.get(gi, [])
            group_hits.append({
                "group_index": gi,
                "synonyms": ginfo["words"],
                "required": ginfo["required"],
                "label": ginfo["label"],
                "hit_synonyms": hit_syns,
                "hit_count": len(positions),
            })

        results.append({
            "library": lib.name,
            "library_note": lib.note,
            "chunk_id": cid,
            "source_file": source_file,
            "source_file_path": source_file_path,
            "source_sha256": source_sha,
            "matched_words": sorted(data["matched_words"]),
            "hit_count": hit_count,
            "text_offset": text_offset,
            "heading": heading,
            "snippet": snippet,
            "phrase_bonus": phrase_bonus,
            "group_hits": group_hits,
            "cooccur_groups": n_groups_hit,
            "should_cooccur_groups": should_hit_count,
            "pair_bonus": pair_bonus,
            "tightness": tightness,
            "related_score": related_score,
            "required_hit": all(ri in gp for ri in required_indices),
            "score": related_score,  # 兼容下游排序
            **_extract_metadata(src),
        })

    # 按关联得分降序（should 共现多的优先）
    results.sort(key=lambda x: (
        x["should_cooccur_groups"],
        x["related_score"],
        x["cooccur_groups"],
        x["hit_count"],
    ), reverse=True)
    results = results[:top_k]

    elapsed = (time.perf_counter() - t0) * 1000
    # 显示用查询字符串："刘备|先主|先帝 ∩ 死|崩|卒"
    query_str = " ∩ ".join("|".join(g["words"]) for g in cleaned_groups)

    return {
        "query": query_str,
        "groups": cleaned_groups,
        "mode": "related",
        "parallel": parallel,
        "total_hits": len(results),
        "searched_libraries": [{"name": l.name, "note": l.note} for l in libs],
        "elapsed_ms": round(elapsed, 2),
        "results": results,
        "required_indices": required_indices,
        "filtered_by_required": filtered_by_required,
    }


# ============================================================
#  语义向量检索（bge-small-zh + Faiss HNSW）
# ============================================================
#
# 设计目标：
#   作为关键词检索的补充通道，解决以下召回盲区：
#   - 同义表述：用户问"逝世"，库内写"卒"/"崩"/"薨"
#   - 语义分散：用户问"农业发展"，库内分散在"耕作"/"水利"/"丰收"等
#   - 概念隐含：用户问"三国鼎立"，库内无此词但有相关史实描述
#
# 工作流程：
#   1. 查询文本 → bge-small-zh 编码为 384 维向量
#   2. Faiss HNSW 索引做近邻查询（毫秒级）
#   3. 按相似度阈值过滤，附加 chunk 元数据
#   4. 与关键词检索结果融合（加权排序）
#
# 索引构建时机：
#   导入完成后由 web_api 触发后台线程构建，期间不阻塞查询；
#   构建完成后自动热加载，下次查询即可命中。
# ============================================================


def _load_semantic_settings() -> Dict[str, Any]:
    """读取语义检索相关设置。"""
    try:
        from settings import SettingsStore
        store = SettingsStore(_auth_base_dir())
        return {
            "enabled": bool(store.get("semantic_enabled", True)),
            "top_k": int(store.get("semantic_top_k", 30)),
            "min_score": float(store.get("semantic_min_score", 0.30)),
            "fusion_weight": float(store.get("semantic_fusion_weight", 0.5)),
            "auto_build": bool(store.get("semantic_auto_build", True)),
        }
    except Exception:
        return {"enabled": True, "top_k": 30, "min_score": 0.30,
                "fusion_weight": 0.5, "auto_build": True}


def _attach_chunk_metadata(lib: Library, base_dir: str, cid: str,
                            score: float,
                            sub_id: str = "") -> Optional[Dict[str, Any]]:
    """为语义召回的 chunk 附加元数据（源文件、标题、片段等）。

    与关键词检索的结果结构保持一致，便于下游统一处理。

    Args:
        sub_id: 命中的子片段 ID（如 "zone_001/chunk_000001#2"）
                若提供，会从父 chunk 中切出对应子片段文本作为 sub_text 返回，
                前端可用下划线标出向量实际命中的段落。
    """
    try:
        mgr = lib.manager(base_dir)
        parts = cid.split("/")
        if len(parts) != 2:
            return None
        zone_id, chunk_name = parts
        zone = mgr.get_zone(zone_id)
        if zone is None:
            return None
        chunk_path = os.path.join(zone.chunks_dir, f"{chunk_name}.json")
        if not os.path.isfile(chunk_path):
            return None
        with open(chunk_path, "r", encoding="utf-8") as f:
            chunk = json.load(f)
        src = chunk.get("source", {})
        text = chunk.get("text", "") or ""

        # 提取向量命中的子片段文本
        # sub_id 形如 "zone_001/chunk_000001#2"，#后是子片段序号
        # 用 _split_into_subchunks 重新切父 chunk，取对应序号的子片段
        sub_text = ""
        if sub_id and "#" in sub_id:
            try:
                from faiss_index import _split_into_subchunks, DEFAULT_SUB_CHUNK_SIZE
                sub_idx = int(sub_id.rsplit("#", 1)[1])
                # 读取当前子分块大小配置（与构建时一致）
                sub_size = DEFAULT_SUB_CHUNK_SIZE
                try:
                    from settings import SettingsStore
                    store = SettingsStore(_auth_base_dir())
                    sub_size = int(store.get("semantic_sub_chunk_size", DEFAULT_SUB_CHUNK_SIZE))
                except Exception:
                    pass
                subchunks = _split_into_subchunks(text, max_size=sub_size)
                if 0 <= sub_idx < len(subchunks):
                    sub_text = subchunks[sub_idx]
            except Exception:
                pass

        # snippet 优先用子片段文本（向量实际命中的段落）
        # 截断到 200 字避免过长
        if sub_text:
            snippet = sub_text[:200].replace("\n", " ")
            if len(sub_text) > 200:
                snippet = snippet + "..."
        else:
            snippet = text[:80].replace("\n", " ")
            if len(text) > 80:
                snippet = snippet + "..."

        return {
            "library": lib.name,
            "library_note": lib.note,
            "chunk_id": cid,
            "source_file": src.get("file_name", ""),
            "source_file_path": src.get("file_path", ""),
            "source_sha256": src.get("source_sha256", ""),
            "matched_words": [],          # 语义召回无命中词
            "hit_count": 0,               # 语义召回无命中数
            "phrase_bonus": 0,
            "semantic_score": score,      # 语义相似度（0~1）
            "text_offset": chunk.get("text_offset", 0),
            "heading": chunk.get("heading", "") or "",
            "snippet": snippet,
            "sub_text": sub_text,         # 向量命中的子片段完整文本（前端下划线展示）
            "sub_id": sub_id,             # 子片段 ID
            "score": score,               # 综合分数（融合时会被覆盖）
            "channel": "semantic",        # 标识来源通道
            **_extract_metadata(src),
        }
    except Exception:
        return None


def search_semantic(
    registry: LibraryRegistry,
    query: str,
    base_dir: str,
    library_names: Optional[List[str]] = None,
    parallel: int = 4,
    top_k: Optional[int] = None,
    chunk_filter: Optional[set] = None,
) -> Dict:
    """跨库语义向量检索。

    Args:
        query: 查询文本（自然语言句子或关键词均可）
        library_names: 限定查询的库名；None 表示全部
        parallel: 并行度（语义检索本身很快，主要并行的是元数据装配）
        top_k: 单库召回条数；None 时用 settings.semantic_top_k
        chunk_filter: 若提供，只保留该集合中的 chunk_id（分块检索时用）

    Returns:
        与 parallel_search 兼容的结构，额外含 channel="semantic" 字段
        依赖未装 / 索引未就绪时返回空结果（total_hits=0），不报错
    """
    t0 = time.perf_counter()
    cfg = _load_semantic_settings()
    if not cfg["enabled"]:
        return {
            "query": query, "mode": "semantic",
            "total_hits": 0, "searched_libraries": [],
            "results": [], "elapsed_ms": 0,
            "semantic_available": False,
            "semantic_reason": "语义检索通道已在设置中关闭",
        }

    if top_k is None:
        top_k = cfg["top_k"]

    # 懒加载语义管理器（依赖未装时返回 unavailable）
    try:
        from semantic_manager import get_manager
        mgr_sem = get_manager(base_dir)
    except Exception as e:
        return {
            "query": query, "mode": "semantic",
            "total_hits": 0, "searched_libraries": [],
            "results": [], "elapsed_ms": 0,
            "semantic_available": False,
            "semantic_reason": f"语义模块加载失败：{e}",
        }

    if not mgr_sem.available():
        return {
            "query": query, "mode": "semantic",
            "total_hits": 0, "searched_libraries": [],
            "results": [], "elapsed_ms": 0,
            "semantic_available": False,
            "semantic_reason": mgr_sem.fail_reason(),
        }

    # 解析目标库
    all_libs = registry.list_libraries()
    if library_names:
        libs = [l for l in all_libs if l.name in library_names]
        missing = set(library_names) - {l.name for l in libs}
        if missing:
            raise ValueError(f"库不存在: {sorted(missing)}")
    else:
        libs = all_libs

    if not libs:
        return {
            "query": query, "mode": "semantic",
            "total_hits": 0, "searched_libraries": [],
            "results": [], "elapsed_ms": 0,
            "semantic_available": True,
        }

    # 并行查询各库
    def _query_one_lib(lib: Library) -> Dict:
        lib_root = lib.abs_path(base_dir)
        hits = mgr_sem.search(lib_root, query, top_k=top_k,
                              chunk_filter=chunk_filter)
        # 过滤低分 + 附加元数据
        results = []
        for h in hits:
            if h["score"] < cfg["min_score"]:
                continue
            meta = _attach_chunk_metadata(lib, base_dir, h["chunk_id"], h["score"],
                                          sub_id=h.get("sub_id", ""))
            if meta is not None:
                results.append(meta)
        return {
            "library": lib.name, "note": lib.note,
            "hits": len(results), "results": results,
        }

    lib_results: List[Dict] = []
    if len(libs) == 1 or parallel <= 1:
        for lib in libs:
            lib_results.append(_query_one_lib(lib))
    else:
        with ThreadPoolExecutor(max_workers=min(parallel, len(libs))) as pool:
            futures = {pool.submit(_query_one_lib, lib): lib for lib in libs}
            order = {lib.name: i for i, lib in enumerate(libs)}
            for fut in as_completed(futures):
                lib_results.append(fut.result())
            lib_results.sort(key=lambda r: order.get(r["library"], 0))

    # 跨库合并
    all_results: List[Dict] = []
    for lr in lib_results:
        all_results.extend(lr["results"])
    # 语义召回按相似度降序
    all_results.sort(key=lambda x: x.get("semantic_score", 0), reverse=True)

    elapsed = (time.perf_counter() - t0) * 1000
    return {
        "query": query, "mode": "semantic",
        "parallel": parallel,
        "total_hits": len(all_results),
        "searched_libraries": [
            {"name": lr["library"], "note": lr["note"], "hits": lr["hits"]}
            for lr in lib_results
        ],
        "elapsed_ms": round(elapsed, 2),
        "results": all_results,
        "semantic_available": True,
        "semantic_min_score": cfg["min_score"],
    }


def search_semantic_groups(
    registry: LibraryRegistry,
    semantic_groups: List[List[str]],
    base_dir: str,
    library_names: Optional[List[str]] = None,
    parallel: int = 4,
    top_k: int = 30,
    chunk_filter: Optional[set] = None,
) -> Dict:
    """语义同义词组检索：对 [A,B] 内每个词独立做向量召回，OR 合并取最高分。

    用途：
      用户输入 [死，崩，薨] 时，对"死""崩""薨"分别做语义检索，
      任一命中即算该组命中，按 chunk_id 聚合保留最高相似度。

    多组语义同义词时（如 [死，崩] [战役，交锋]）：
      组间是 AND 关系——chunk 必须在每组中都有命中（至少一个词命中），
      才算最终结果。这样能提高精确度，避免单一概念漂移召回。

    Args:
        semantic_groups: 语义同义词组列表，如 [["死","崩","薨"], ["战役","交锋"]]
        chunk_filter: 若提供，只保留该集合中的 chunk_id（与关键词检索交集时用）

    Returns:
        与 search_semantic 兼容的结构，额外含 semantic_groups 字段
    """
    t0 = time.perf_counter()
    cfg = _load_semantic_settings()
    if not cfg["enabled"]:
        return {
            "query": "", "mode": "semantic_groups",
            "total_hits": 0, "searched_libraries": [],
            "results": [], "elapsed_ms": 0,
            "semantic_available": False,
            "semantic_reason": "语义检索通道已在设置中关闭",
        }

    # 懒加载语义管理器
    try:
        from semantic_manager import get_manager
        mgr_sem = get_manager(base_dir)
    except Exception as e:
        return {
            "query": "", "mode": "semantic_groups",
            "total_hits": 0, "searched_libraries": [],
            "results": [], "elapsed_ms": 0,
            "semantic_available": False,
            "semantic_reason": f"语义模块加载失败：{e}",
        }

    if not mgr_sem.available():
        return {
            "query": "", "mode": "semantic_groups",
            "total_hits": 0, "searched_libraries": [],
            "results": [], "elapsed_ms": 0,
            "semantic_available": False,
            "semantic_reason": mgr_sem.fail_reason(),
        }

    # 解析目标库
    all_libs = registry.list_libraries()
    if library_names:
        libs = [l for l in all_libs if l.name in library_names]
        missing = set(library_names) - {l.name for l in libs}
        if missing:
            raise ValueError(f"库不存在: {sorted(missing)}")
    else:
        libs = all_libs

    if not libs:
        return {
            "query": "", "mode": "semantic_groups",
            "total_hits": 0, "searched_libraries": [],
            "results": [], "elapsed_ms": 0,
            "semantic_available": True,
        }

    # 清洗语义同义词组
    cleaned_groups = []
    for g in semantic_groups:
        words = [w.strip() for w in g if w and w.strip()]
        if words:
            cleaned_groups.append(words)

    if not cleaned_groups:
        return {
            "query": "", "mode": "semantic_groups",
            "total_hits": 0, "searched_libraries": [],
            "results": [], "elapsed_ms": 0,
            "semantic_available": True,
        }

    # 对每个组内的每个词独立做语义检索，结果按组聚合
    # group_results[i] = {chunk_id: {"score": max_score, "lib": Library, "sub_id": "..."}}
    group_results: List[Dict[str, Dict]] = []
    for words in cleaned_groups:
        group_hits: Dict[str, Dict] = {}
        for word in words:
            # 复用 search_semantic 的单库查询逻辑
            def _query_one_lib(lib: Library, w=word) -> List[Dict]:
                lib_root = lib.abs_path(base_dir)
                hits = mgr_sem.search(lib_root, w, top_k=cfg["top_k"],
                                      chunk_filter=chunk_filter)
                results = []
                for h in hits:
                    if h["score"] < cfg["min_score"]:
                        continue
                    meta = _attach_chunk_metadata(lib, base_dir,
                                                  h["chunk_id"], h["score"],
                                                  sub_id=h.get("sub_id", ""))
                    if meta is not None:
                        results.append(meta)
                return results

            # 并行查各库
            if len(libs) == 1 or parallel <= 1:
                for lib in libs:
                    for r in _query_one_lib(lib):
                        cid = r["chunk_id"]
                        # 组内 OR：保留最高相似度
                        if cid not in group_hits or r["semantic_score"] > group_hits[cid]["semantic_score"]:
                            group_hits[cid] = r
            else:
                with ThreadPoolExecutor(max_workers=min(parallel, len(libs))) as pool:
                    futures = [pool.submit(_query_one_lib, lib) for lib in libs]
                    for fut in as_completed(futures):
                        for r in fut.result():
                            cid = r["chunk_id"]
                            if cid not in group_hits or r["semantic_score"] > group_hits[cid]["semantic_score"]:
                                group_hits[cid] = r
        group_results.append(group_hits)

    # 组间 AND 交集：chunk 必须在每组中都有命中
    if len(group_results) == 1:
        # 单组：直接用该组结果
        final_hits = group_results[0]
    else:
        # 多组取交集
        common_cids = set(group_results[0].keys())
        for gr in group_results[1:]:
            common_cids &= set(gr.keys())
        final_hits = {}
        for cid in common_cids:
            # 取各组中最高分的最大值作为最终相似度
            best = None
            for gr in group_results:
                r = gr[cid]
                if best is None or r["semantic_score"] > best["semantic_score"]:
                    best = r
            final_hits[cid] = best

    # 转换为列表并按相似度降序
    all_results = sorted(final_hits.values(),
                         key=lambda x: x.get("semantic_score", 0),
                         reverse=True)[:top_k]

    # 各库命中统计（前端展示用）
    lib_hits: Dict[str, int] = {l.name: 0 for l in libs}
    for r in all_results:
        if r["library"] in lib_hits:
            lib_hits[r["library"]] += 1

    elapsed = (time.perf_counter() - t0) * 1000
    return {
        "query": " ∩ ".join("[" + "，".join(g) + "]" for g in cleaned_groups),
        "mode": "semantic_groups",
        "parallel": parallel,
        "total_hits": len(all_results),
        "searched_libraries": [
            {"name": name, "note": l.note, "hits": lib_hits.get(name, 0)}
            for l in libs for name in [l.name]
        ],
        "elapsed_ms": round(elapsed, 2),
        "results": all_results,
        "semantic_available": True,
        "semantic_min_score": cfg["min_score"],
        "semantic_groups": cleaned_groups,
    }


# ============================================================
#  五路融合检索（关键词 + 关联词 + 同义词 + 混合型 + 语义向量）
# ============================================================
#
# 融合策略：
#   1. 关键词检索（parallel_search / parallel_search_partitioned）
#      → 命中词数 + 词组连续 bonus
#   2. 关联词检索（search_related_keywords，单组时退化为同义词 OR）
#   3. 语义向量检索（search_semantic，相似度 0~1）
#   4. 多关键词共现（search_multi_keywords，组间共现² 加分）
#
# 评分归一化：
#   - 关键词分：归一化到 [0, 1]，公式 = (hit_count + phrase_bonus) / max_keyword_score
#   - 语义分：天然在 [0, 1]
#   - 融合分 = 语义分 × fusion_weight + 关键词分 × (1 - fusion_weight)
#   - 仅命中关键词或仅命中语义的 chunk，缺失通道按 0 处理
#
# 多通道去重：
#   - 按 chunk_id 去重，多通道命中时取融合分最高者
#   - 合并 matched_words、保留各通道来源标记
# ============================================================


def _normalize_keyword_scores(results: List[Dict]) -> List[Dict]:
    """把关键词检索的 score 归一化到 [0, 1]。

    使用 max-score 归一化（最简单的稳健方法）。
    """
    if not results:
        return results
    max_score = max(r.get("score", 0) for r in results) or 1
    for r in results:
        r["keyword_score_norm"] = r.get("score", 0) / max_score
    return results


def _fuse_results(keyword_results: List[Dict],
                   semantic_results: List[Dict],
                   fusion_weight: float = 0.5) -> List[Dict]:
    """融合关键词与语义检索结果。

    Args:
        keyword_results: 关键词检索结果（已归一化）
        semantic_results: 语义检索结果
        fusion_weight: 语义分权重，0=纯关键词，1=纯语义

    Returns:
        融合后的结果列表，按融合分降序
    """
    # 按 chunk_id 索引
    merged: Dict[str, Dict] = {}

    # 1. 关键词结果
    for r in keyword_results:
        cid = r["chunk_id"]
        kw_norm = r.get("keyword_score_norm", 0)
        fused = kw_norm * (1 - fusion_weight)
        item = dict(r)
        item["channels"] = ["keyword"]
        item["keyword_score_norm"] = kw_norm
        item["semantic_score"] = 0
        item["fused_score"] = fused
        item["score"] = fused  # 覆盖原 score，统一排序
        merged[cid] = item

    # 2. 语义结果
    for r in semantic_results:
        cid = r["chunk_id"]
        sem_score = r.get("semantic_score", 0)
        fused = sem_score * fusion_weight
        if cid in merged:
            # 已有关键词结果：合并
            item = merged[cid]
            item["semantic_score"] = sem_score
            item["fused_score"] = item.get("keyword_score_norm", 0) * (1 - fusion_weight) + fused
            item["score"] = item["fused_score"]
            item["channels"].append("semantic")
            # 合并 matched_words
            existing_words = set(item.get("matched_words", []))
            existing_words.update(r.get("matched_words", []))
            item["matched_words"] = sorted(existing_words)
            # 取较长的 snippet
            if len(r.get("snippet", "")) > len(item.get("snippet", "")):
                item["snippet"] = r["snippet"]
            # 保留向量命中的子片段（用于原文下划线展示）
            # 双通道命中时 keyword 侧无 sub_text，需从 semantic 侧补齐
            if r.get("sub_text") and not item.get("sub_text"):
                item["sub_text"] = r["sub_text"]
                if r.get("sub_id"):
                    item["sub_id"] = r["sub_id"]
        else:
            item = dict(r)
            item["channels"] = ["semantic"]
            item["keyword_score_norm"] = 0
            item["semantic_score"] = sem_score
            item["fused_score"] = fused
            item["score"] = fused
            merged[cid] = item

    # 3. 按融合分降序
    return sorted(merged.values(),
                  key=lambda x: x.get("fused_score", 0), reverse=True)


# ============================================================
#  父 chunk 级语义检索（大chunk模式）
# ============================================================

def search_semantic_parent(
    registry: LibraryRegistry,
    query: str,
    base_dir: str,
    library_names: Optional[List[str]] = None,
    parallel: int = 4,
    top_k: int = 20,
) -> Dict:
    """父chunk级语义检索（大chunk模式）。

    与 search_semantic 的区别：
      - 直接查父chunk索引（子片段向量池化得到），返回父chunk
      - 无 sub_text / sub_id（不定位到具体子片段）
      - 查询更快（父chunk数远少于子片段数）
      - 适合"宏观定位"场景，不适合"精确定位句子"

    返回结构与 search_semantic 兼容，但 results 中无 sub_text/sub_id 字段
    """
    t0 = time.perf_counter()
    cfg = _load_semantic_settings()
    if not cfg["enabled"]:
        return {
            "query": query, "mode": "semantic_parent",
            "total_hits": 0, "searched_libraries": [],
            "results": [], "elapsed_ms": 0,
            "semantic_available": False,
            "semantic_reason": "语义检索通道已在设置中关闭",
        }

    try:
        from semantic_manager import get_manager
        mgr_sem = get_manager(base_dir)
    except Exception as e:
        return {
            "query": query, "mode": "semantic_parent",
            "total_hits": 0, "searched_libraries": [],
            "results": [], "elapsed_ms": 0,
            "semantic_available": False,
            "semantic_reason": f"语义模块加载失败：{e}",
        }

    if not mgr_sem.available():
        return {
            "query": query, "mode": "semantic_parent",
            "total_hits": 0, "searched_libraries": [],
            "results": [], "elapsed_ms": 0,
            "semantic_available": False,
            "semantic_reason": mgr_sem.fail_reason(),
        }

    all_libs = registry.list_libraries()
    if library_names:
        libs = [l for l in all_libs if l.name in library_names]
        missing = set(library_names) - {l.name for l in libs}
        if missing:
            raise ValueError(f"库不存在: {sorted(missing)}")
    else:
        libs = all_libs

    if not libs:
        return {
            "query": query, "mode": "semantic_parent",
            "total_hits": 0, "searched_libraries": [],
            "results": [], "elapsed_ms": 0,
            "semantic_available": True,
        }

    def _query_one_lib(lib: Library) -> List[Dict]:
        lib_root = lib.abs_path(base_dir)
        hits = mgr_sem.search_parent(lib_root, query, top_k=cfg["top_k"])
        results = []
        for h in hits:
            if h["score"] < cfg["min_score"]:
                continue
            # 附加元数据（无 sub_text，大chunk模式不定位到子片段）
            meta = _attach_chunk_metadata(lib, base_dir,
                                          h["chunk_id"], h["score"],
                                          sub_id="")
            if meta is not None:
                # 移除 sub_text/sub_id（大chunk模式无意义）
                meta.pop("sub_text", None)
                meta.pop("sub_id", None)
                # 大chunk模式：snippet 展示父chunk前 300 字作为预览
                # （_attach_chunk_metadata 默认只取 80 字，对 10000 字父chunk太少）
                try:
                    mgr_lib = lib.manager(base_dir)
                    parts = h["chunk_id"].split("/")
                    if len(parts) == 2:
                        zone = mgr_lib.get_zone(parts[0])
                        if zone:
                            cp = os.path.join(zone.chunks_dir, f"{parts[1]}.json")
                            if os.path.isfile(cp):
                                with open(cp, "r", encoding="utf-8") as f:
                                    chk = json.load(f)
                                full = chk.get("text", "") or ""
                                if full:
                                    preview = full[:300].replace("\n", " ")
                                    if len(full) > 300:
                                        preview += "..."
                                    meta["snippet"] = preview
                                    meta["text_length"] = len(full)
                except Exception:
                    pass
                results.append(meta)
        return results

    # 并行查各库
    lib_results: List[Dict] = []
    if len(libs) == 1 or parallel <= 1:
        for lib in libs:
            res = _query_one_lib(lib)
            lib_results.append({"library": lib.name, "note": lib.note,
                                "hits": len(res), "results": res})
    else:
        with ThreadPoolExecutor(max_workers=min(parallel, len(libs))) as pool:
            futures = {pool.submit(_query_one_lib, lib): lib for lib in libs}
            order = {lib.name: i for i, lib in enumerate(libs)}
            for fut in as_completed(futures):
                res = fut.result()
                lib = futures[fut]
                lib_results.append({"library": lib.name, "note": lib.note,
                                    "hits": len(res), "results": res})
            lib_results.sort(key=lambda r: order.get(r["library"], 0))

    # 合并并按相似度降序
    all_results: List[Dict] = []
    for lr in lib_results:
        all_results.extend(lr["results"])
    all_results.sort(key=lambda x: x.get("semantic_score", 0), reverse=True)
    all_results = all_results[:top_k]

    elapsed = (time.perf_counter() - t0) * 1000
    return {
        "query": query, "mode": "semantic_parent",
        "parallel": parallel,
        "total_hits": len(all_results),
        "searched_libraries": [
            {"name": lr["library"], "note": lr["note"], "hits": lr["hits"]}
            for lr in lib_results
        ],
        "elapsed_ms": round(elapsed, 2),
        "results": all_results,
        "semantic_available": True,
        "semantic_min_score": cfg["min_score"],
    }


def search_semantic_groups_parent(
    registry: LibraryRegistry,
    semantic_groups: List[List[str]],
    base_dir: str,
    library_names: Optional[List[str]] = None,
    parallel: int = 4,
    top_k: int = 30,
    chunk_filter: Optional[set] = None,
) -> Dict:
    """父chunk级语义同义词组检索（大chunk模式的 [] 语法）。

    与 search_semantic_groups 的区别：
      - 使用父chunk索引（池化向量）而非子片段索引
      - 返回结果无 sub_text/sub_id
      - 适合"宏观定位"场景

    语义同义词组逻辑与 search_semantic_groups 一致：
      - 组内 OR：任一词命中即算该组命中
      - 组间 AND：chunk 必须在每组中都有命中
    """
    t0 = time.perf_counter()
    cfg = _load_semantic_settings()
    if not cfg["enabled"]:
        return {
            "query": "", "mode": "semantic_groups_parent",
            "total_hits": 0, "searched_libraries": [],
            "results": [], "elapsed_ms": 0,
            "semantic_available": False,
            "semantic_reason": "语义检索通道已在设置中关闭",
        }

    try:
        from semantic_manager import get_manager
        mgr_sem = get_manager(base_dir)
    except Exception as e:
        return {
            "query": "", "mode": "semantic_groups_parent",
            "total_hits": 0, "searched_libraries": [],
            "results": [], "elapsed_ms": 0,
            "semantic_available": False,
            "semantic_reason": f"语义模块加载失败：{e}",
        }

    if not mgr_sem.available():
        return {
            "query": "", "mode": "semantic_groups_parent",
            "total_hits": 0, "searched_libraries": [],
            "results": [], "elapsed_ms": 0,
            "semantic_available": False,
            "semantic_reason": mgr_sem.fail_reason(),
        }

    all_libs = registry.list_libraries()
    if library_names:
        libs = [l for l in all_libs if l.name in library_names]
        missing = set(library_names) - {l.name for l in libs}
        if missing:
            raise ValueError(f"库不存在: {sorted(missing)}")
    else:
        libs = all_libs

    if not libs:
        return {
            "query": "", "mode": "semantic_groups_parent",
            "total_hits": 0, "searched_libraries": [],
            "results": [], "elapsed_ms": 0,
            "semantic_available": True,
        }

    # 清洗语义同义词组
    cleaned_groups = []
    for g in semantic_groups:
        words = [w.strip() for w in g if w and w.strip()]
        if words:
            cleaned_groups.append(words)

    if not cleaned_groups:
        return {
            "query": "", "mode": "semantic_groups_parent",
            "total_hits": 0, "searched_libraries": [],
            "results": [], "elapsed_ms": 0,
            "semantic_available": True,
        }

    # 对每个组内的每个词独立做父chunk语义检索，结果按组聚合
    group_results: List[Dict[str, Dict]] = []
    for words in cleaned_groups:
        group_hits: Dict[str, Dict] = {}
        for word in words:
            def _query_one_lib(lib: Library, w=word) -> List[Dict]:
                lib_root = lib.abs_path(base_dir)
                hits = mgr_sem.search_parent(lib_root, w, top_k=cfg["top_k"],
                                             chunk_filter=chunk_filter)
                results = []
                for h in hits:
                    if h["score"] < cfg["min_score"]:
                        continue
                    meta = _attach_chunk_metadata(lib, base_dir,
                                                  h["chunk_id"], h["score"],
                                                  sub_id="")
                    if meta is not None:
                        meta.pop("sub_text", None)
                        meta.pop("sub_id", None)
                        results.append(meta)
                return results

            if len(libs) == 1 or parallel <= 1:
                for lib in libs:
                    for r in _query_one_lib(lib):
                        cid = r["chunk_id"]
                        if cid not in group_hits or r["semantic_score"] > group_hits[cid]["semantic_score"]:
                            group_hits[cid] = r
            else:
                with ThreadPoolExecutor(max_workers=min(parallel, len(libs))) as pool:
                    futures = [pool.submit(_query_one_lib, lib) for lib in libs]
                    for fut in as_completed(futures):
                        for r in fut.result():
                            cid = r["chunk_id"]
                            if cid not in group_hits or r["semantic_score"] > group_hits[cid]["semantic_score"]:
                                group_hits[cid] = r
        group_results.append(group_hits)

    # 组间 AND 交集
    if len(group_results) == 1:
        final_hits = group_results[0]
    else:
        common_cids = set(group_results[0].keys())
        for gr in group_results[1:]:
            common_cids &= set(gr.keys())
        final_hits = {}
        for cid in common_cids:
            best = None
            for gr in group_results:
                r = gr[cid]
                if best is None or r["semantic_score"] > best["semantic_score"]:
                    best = r
            final_hits[cid] = best

    all_results = sorted(final_hits.values(),
                         key=lambda x: x.get("semantic_score", 0),
                         reverse=True)[:top_k]

    lib_hits: Dict[str, int] = {l.name: 0 for l in libs}
    for r in all_results:
        if r["library"] in lib_hits:
            lib_hits[r["library"]] += 1

    elapsed = (time.perf_counter() - t0) * 1000
    return {
        "query": " ∩ ".join("[" + "，".join(g) + "]" for g in cleaned_groups),
        "mode": "semantic_groups_parent",
        "parallel": parallel,
        "total_hits": len(all_results),
        "searched_libraries": [
            {"name": name, "note": l.note, "hits": lib_hits.get(name, 0)}
            for l in libs for name in [l.name]
        ],
        "elapsed_ms": round(elapsed, 2),
        "results": all_results,
        "semantic_available": True,
        "semantic_min_score": cfg["min_score"],
        "semantic_groups": cleaned_groups,
    }


# 括号组匹配正则：() （） 圆括号关键词组；[] 【】 方括号语义组；{} ｛｝ 花括号标题组
_BRACKET_GROUPS_RE = re.compile(
    r"[（(]([^）)]*)[）)]"          # g1: 圆括号（关键词同义词）
    r"|\[([^\]]*)\]"                # g2: 英文方括号（语义同义词）
    r"|【([^】]*)】"                # g3: 中文方括号（语义同义词）
    r"|\{([^{}]*)\}"                # g4: 英文花括号（标题限定）
    r"|｛([^｛｝]*)｝"               # g5: 中文花括号（标题限定）
)


def _parse_query_groups_py(q: str) -> Optional[List[List[str]]]:
    """解析查询字符串为关联词组（与前端 parseQueryGroups 一致）。

    规则：
      - 无括号无空格 = 单词 → 返回 None（普通检索）
      - 多个词空格分隔 = 每词一组
      - (刘备，先主) 或 （刘备，先主） = 关键词同义词组
        （分隔符支持中英文逗号、顿号、竖线；引号内为原子词项）
      - [死，崩] 或 【死，崩】 = 语义同义词组（由 _parse_semantic_groups_py 处理）
      - {水利卷} 或 ｛水利卷｝ = 标题限定组（由 _parse_title_groups_py 处理）
      - 混合：刘备 (死，崩) [战役，交锋] {水利卷} 也支持
    """
    q = (q or "").strip()
    if not q:
        return None
    has_bracket = bool(re.search(r"[\[【]", q))
    has_paren = bool(re.search(r"[（(]", q))
    has_brace = bool(re.search(r"[{｛]", q))
    has_space = bool(re.search(r"\s", q))
    if not has_paren and not has_space and not has_bracket and not has_brace:
        return None
    if not has_paren and not has_bracket and not has_brace:
        words = [w for w in re.split(r"\s+", q) if w]
        # 每词一组（与前端 parseQueryGroups 一致；扁平字符串下游无法作为组处理）
        return [[w] for w in words] if len(words) >= 2 else None
    # 解析括号组（() [] 【】 {} ｛｝）+ 括号外独立词
    # 此函数仅返回 keyword_groups（圆括号内容 + 括号外独立词）
    # 方括号（语义）与花括号（标题）内容由各自的解析函数处理
    groups: List[List[str]] = []
    last_idx = 0
    for m in _BRACKET_GROUPS_RE.finditer(q):
        before = q[last_idx:m.start()].strip()
        if before:
            for w in re.split(r"\s+", before):
                if w:
                    groups.append([w])
        # m.group(1) = 圆括号内容（关键词同义词）
        # m.group(2)/m.group(3) = 方括号内容（语义同义词，跳过）
        # m.group(4)/m.group(5) = 花括号内容（标题限定，跳过）
        if m.group(1) is not None:
            syns = _split_group_terms(m.group(1))
            if syns:
                groups.append(syns)
        last_idx = m.end()
    after = q[last_idx:].strip()
    if after:
        for w in re.split(r"\s+", after):
            if w:
                groups.append([w])
    return groups if groups else None


def _parse_semantic_groups_py(q: str) -> Optional[List[List[str]]]:
    """解析查询字符串中的 [] 英文方括号或 【】 中文方括号语义同义词组。

    仅返回方括号内的内容，() {} 和括号外独立词由其他函数处理。

    例：
      "刘备 [死，崩，薨]" → [["死","崩","薨"]]
      "刘备 【死，崩，薨】" → [["死","崩","薨"]]
      "[战役，交锋] [死，崩]" → [["战役","交锋"], ["死","崩"]]
      "刘备 (死，崩)" → None（无方括号）
    """
    q = (q or "").strip()
    if not q:
        return None
    if not re.search(r"[\[【]", q):
        return None
    re_bracket = re.compile(r"\[([^\]]*)\]|【([^】]*)】")
    groups: List[List[str]] = []
    for m in re_bracket.finditer(q):
        content = m.group(1) if m.group(1) is not None else m.group(2)
        if content is not None:
            syns = _split_group_terms(content)
            if syns:
                groups.append(syns)
    return groups if groups else None


def _parse_title_groups_py(q: str) -> Optional[List[List[str]]]:
    """解析查询字符串中的 {} 英文花括号或 ｛｝ 中文花括号标题限定组。

    组内任一词出现在 chunk 标题(heading)或来源文件名中即通过该组。
    例：
      "{水利卷} 灌溉" → [["水利卷"]]
      "｛水利卷，水利志｝ 灌溉" → [["水利卷","水利志"]]
      "刘备 (死，崩)" → None（无花括号）
    """
    q = (q or "").strip()
    if not q:
        return None
    if not re.search(r"[{｛]", q):
        return None
    groups: List[List[str]] = []
    for m in _BRACKET_GROUPS_RE.finditer(q):
        content = m.group(4) if m.group(4) is not None else m.group(5)
        if content is not None:
            words = _split_group_terms(content)
            if words:
                groups.append(words)
    return groups if groups else None


def strip_title_groups_py(q: str) -> str:
    """从查询串中移除 {} 标题限定组，返回剩余部分（供纯关键词路径使用）。"""
    if not q:
        return q
    return _BRACKET_GROUPS_RE.sub(
        lambda m: m.group(0) if (m.group(4) is None and m.group(5) is None) else " ",
        q)


def parallel_search_fused(
    registry: LibraryRegistry,
    query: str,
    base_dir: str,
    library_names: Optional[List[str]] = None,
    parallel: int = 4,
    top_k: int = 20,
    enable_semantic: bool = True,
    chunk_filter: Optional[set] = None,
) -> Dict:
    """五路融合检索：关键词 + 关联词 + 同义词 + 混合型 + 语义向量。

    本函数是检索层的统一入口，自动判断：
      - 大库（chunk 数 > 阈值）→ 走分块检索
      - 小库 → 走普通并行检索
      - 语义通道可用 → 追加向量召回并融合

    Args:
        query: 查询文本（可含同义词组语法，如 "刘备 (死，崩)"）
        enable_semantic: 是否启用语义通道（False 时仅用关键词类检索）
        chunk_filter: 分块检索时传入的 chunk_id 范围

    Returns:
        与 parallel_search 兼容的结构，额外含：
        - channels: 各通道命中统计
        - semantic_available: 语义通道是否实际参与
    """
    t0 = time.perf_counter()
    cfg = _load_semantic_settings()

    # 0. 标题限定组（{} 语法）：从查询串剥离，避免 braces 进入关键词检索；
    #    纯标题查询直接走标题检索，混合查询在融合前按标题组过滤
    title_groups = _parse_title_groups_py(query)
    if title_groups:
        query = strip_title_groups_py(query).strip()
        if not query:
            # 纯 {} 查询：仅按标题检索（含标题缓存）
            result = search_by_titles(registry, title_groups, base_dir,
                                      library_names=library_names,
                                      top_k=top_k)
            result["semantic_available"] = False
            return result

    # 1. 关键词类检索（自动判断分块）
    #    解析查询语法：含括号 → 关联词组；多词空格 → 多关键词；否则单关键词
    groups = _parse_query_groups_py(query)
    if groups and len(groups) >= 2:
        # 多组关联词检索
        kw_result = search_related_keywords(
            registry, groups,
            library_names=library_names,
            parallel=parallel, base_dir=base_dir,
            chunk_filter=chunk_filter, top_k=top_k * 3,
        )
    else:
        # 单关键词 / 单组同义词 / 普通多词
        kw_result = parallel_search_partitioned(
            registry, query,
            library_names=library_names,
            parallel=parallel, base_dir=base_dir,
        )

    # 2. 语义检索（可选）
    sem_result = None
    sem_available = False
    if enable_semantic and cfg["enabled"]:
        sem_result = search_semantic(
            registry, query, base_dir,
            library_names=library_names,
            parallel=parallel,
            top_k=cfg["top_k"],
            chunk_filter=chunk_filter,
        )
        sem_available = sem_result.get("semantic_available", False)

    # 3. 融合
    kw_results = kw_result.get("results", [])
    _normalize_keyword_scores(kw_results)
    sem_count_fused = 0  # 实际参与融合的语义结果数（默认 0）

    if sem_available and sem_result:
        sem_results = sem_result.get("results", [])
        # 关键词 0 命中时，对语义结果应用更严格阈值
        # 避免低质量语义召回冒充检索结果（bge-small-zh 在 0.30 阈值下会放行弱相关结果）
        # 保留强语义匹配（如"逝世"→"卒"，分数通常 >0.5），过滤低分噪声
        if not kw_results:
            strict_min = max(cfg["min_score"], 0.45)
            sem_results = [r for r in sem_results
                           if r.get("semantic_score", 0) >= strict_min]
        # 标题限定（{} 语法）：语义召回同样按标题过滤
        if title_groups:
            sem_results = apply_title_filter(sem_results, title_groups)
        sem_count_fused = len(sem_results)  # 实际参与融合的语义结果数
        fused = _fuse_results(kw_results, sem_results,
                              fusion_weight=cfg["fusion_weight"])
    else:
        # 语义不可用：仅用关键词结果，score 保持归一化值
        fused = kw_results
        for r in fused:
            r["channels"] = ["keyword"]
            r["fused_score"] = r.get("keyword_score_norm", 0)

    # 标题限定（{} 语法）：融合结果按标题组过滤后再截断
    if title_groups:
        fused = apply_title_filter(fused, title_groups)

    # 截断到 top_k
    fused = fused[:top_k]

    elapsed = (time.perf_counter() - t0) * 1000

    # 4. 构造返回结构（与 parallel_search 兼容）
    # 统计各通道命中数
    kw_count = len(kw_results)
    # sem_count 反映实际参与融合的语义结果数（关键词 0 命中时可能被严格阈值过滤）
    sem_count = sem_count_fused
    fused_count = len(fused)
    both_count = sum(1 for r in fused if len(r.get("channels", [])) >= 2)

    # 合并 searched_libraries：同时展示关键词命中数和语义命中数
    # 避免前端显示"三库 0 命中"但实际有语义召回结果的混乱
    kw_libs_map = {l["name"]: l for l in kw_result.get("searched_libraries", [])}
    sem_libs_map = {}
    if sem_result:
        for l in sem_result.get("searched_libraries", []):
            sem_libs_map[l["name"]] = l
    merged_libs = []
    for name in list(kw_libs_map.keys()) + [n for n in sem_libs_map if n not in kw_libs_map]:
        kl = kw_libs_map.get(name, {})
        sl = sem_libs_map.get(name, {})
        merged_libs.append({
            "name": name,
            "note": kl.get("note") or sl.get("note", ""),
            "hits": kl.get("hits", 0),              # 关键词命中数
            "semantic_hits": sl.get("hits", 0),     # 语义召回数（原始，未过滤）
            "elapsed_ms": kl.get("elapsed_ms", 0),
        })

    return {
        "query": query,
        "parallel": parallel,
        "total_hits": fused_count,
        "searched_libraries": merged_libs,
        "results": fused,
        "elapsed_ms": round(elapsed, 2),
        "channels": {
            "keyword": kw_count,
            "semantic": sem_count,
            "fused": fused_count,
            "both": both_count,
        },
        "semantic_available": sem_available,
        "semantic_reason": sem_result.get("semantic_reason", "") if sem_result else "",
        "fusion_weight": cfg["fusion_weight"],
    }


# ============================================================
#  纯向量检索工作流（多轮 / 递进 / 精准）
# ============================================================
#
# 三个新函数支持纯向量检索模式：
#   - search_semantic_multi_round: 多轮多查询词向量检索，合并取最高分
#   - search_semantic_progressive: 大chunk→小chunk递进检索（global 模式用）
#   - search_semantic_precise: 小chunk→二次切分→二次向量精准检索
#
# 复用已有基础设施（不修改）：
#   - search_semantic / search_semantic_parent: 底层向量检索
#   - SemanticManager.search(): 带 chunk_filter 的子片段检索
#   - faiss_index.search_subchunks_of_text(): 二次切分向量化检索
#   - _attach_chunk_metadata(): 附加 chunk 元数据
#   - _load_semantic_settings(): 读取语义检索配置
# ============================================================


def search_semantic_multi_round(
    registry: LibraryRegistry,
    queries_per_round: List[List[str]],  # 每轮的查询词列表，如 [["刘备","曹操","赤壁"],["诸葛亮","周瑜","孙权"]]
    base_dir: str,
    library_names: Optional[List[str]] = None,
    parallel: int = 4,
    top_k_per_query: int = 20,
    chunk_level: str = "child",  # "child" 或 "parent"
) -> Dict:
    """多轮向量检索：对每轮的每个查询词独立做向量检索，合并所有结果。

    用途：
      多轮检索用于"主题延伸"场景。例如用户问"赤壁之战"，可分多轮：
        第 1 轮 [刘备, 曹操, 赤壁] —— 召回直接相关 chunk
        第 2 轮 [诸葛亮, 周瑜, 孙权] —— 召回关联人物 chunk
      每轮的每个查询词独立检索，结果按 chunk_id 聚合，取最高分。

    合并规则：
      - 同一 chunk 被多轮多次命中时，取最高分
      - 记录所有命中的查询词到 matched_queries
      - 记录命中的轮次数到 rounds_hit

    Args:
        queries_per_round: 每轮的查询词列表，外层为轮次，内层为该轮的查询词
        chunk_level: "child" 用小chunk索引（search_semantic，含 sub_text/sub_id），
                     "parent" 用大chunk索引（search_semantic_parent，无 sub_text）
        top_k_per_query: 单次检索召回条数

    Returns:
        与 search_semantic 兼容的结构，额外含：
        - rounds_info: 每轮每个查询词的命中情况统计
        - matched_queries: 每个 chunk 命中的查询词列表
        - rounds_hit: 每个 chunk 命中的轮次数
    """
    t0 = time.perf_counter()

    # 清洗查询词：去空、去组内重复
    rounds: List[List[str]] = []
    for qs in queries_per_round:
        seen = set()
        words = []
        for w in qs:
            w = (w or "").strip()
            if w and w not in seen:
                seen.add(w)
                words.append(w)
        if words:
            rounds.append(words)

    if not rounds:
        return {
            "query": "", "mode": "semantic_multi_round",
            "total_hits": 0, "searched_libraries": [],
            "results": [], "elapsed_ms": 0,
            "semantic_available": True,
            "rounds_info": [],
        }

    # 选择底层检索函数（chunk_level="parent" 用大chunk索引，否则用小chunk索引）
    if chunk_level == "parent":
        _search_fn = search_semantic_parent
    else:
        _search_fn = search_semantic

    # 逐轮逐词检索，结果按 chunk_id 聚合
    # chunk_id -> {"meta": result_dict, "matched_queries": [...], "rounds_hit": set}
    merged: Dict[str, Dict] = {}
    rounds_info: List[Dict] = []
    sem_available = True
    sem_reason = ""

    for round_idx, words in enumerate(rounds):
        for word in words:
            try:
                res = _search_fn(
                    registry, word, base_dir,
                    library_names=library_names,
                    parallel=parallel,
                    top_k=top_k_per_query,
                )
            except Exception as e:
                # 单词检索异常不中断整体流程，记录错误继续
                rounds_info.append({
                    "round": round_idx, "query": word,
                    "hits": 0, "error": str(e),
                })
                continue

            if not res.get("semantic_available", True):
                sem_available = False
                sem_reason = res.get("semantic_reason", "")

            hits = res.get("results", [])
            for r in hits:
                cid = r["chunk_id"]
                if cid not in merged:
                    merged[cid] = {
                        "meta": r,
                        "matched_queries": [],
                        "rounds_hit": set(),
                    }
                existing = merged[cid]
                # 同一 chunk 多次命中：取最高分
                if r.get("semantic_score", 0) > existing["meta"].get("semantic_score", 0):
                    existing["meta"] = r
                # 记录命中的查询词（去重）
                if word not in existing["matched_queries"]:
                    existing["matched_queries"].append(word)
                # 记录命中的轮次
                existing["rounds_hit"].add(round_idx)

            rounds_info.append({
                "round": round_idx, "query": word,
                "hits": len(hits),
            })

    # 组装最终结果列表
    all_results: List[Dict] = []
    for cid, data in merged.items():
        item = dict(data["meta"])
        item["matched_queries"] = data["matched_queries"]
        item["rounds_hit"] = len(data["rounds_hit"])
        item["rounds_hit_list"] = sorted(data["rounds_hit"])
        all_results.append(item)

    # 按相似度降序，多轮命中优先（多轮命中说明跨主题相关，更可能是核心 chunk）
    all_results.sort(
        key=lambda x: (x.get("rounds_hit", 0), x.get("semantic_score", 0)),
        reverse=True,
    )

    elapsed = (time.perf_counter() - t0) * 1000

    # 统计各库命中数（前端展示用）
    all_libs = registry.list_libraries()
    if library_names:
        libs = [l for l in all_libs if l.name in library_names]
    else:
        libs = all_libs
    lib_hits: Dict[str, int] = {l.name: 0 for l in libs}
    for r in all_results:
        lib_name = r.get("library", "")
        if lib_name in lib_hits:
            lib_hits[lib_name] += 1

    # 查询字符串：每轮用 [] 包裹，轮间用 ; 分隔
    query_str = " ; ".join("[" + "，".join(w) + "]" for w in rounds)

    return {
        "query": query_str,
        "mode": "semantic_multi_round",
        "chunk_level": chunk_level,
        "parallel": parallel,
        "total_hits": len(all_results),
        "searched_libraries": [
            {"name": l.name, "note": l.note, "hits": lib_hits.get(l.name, 0)}
            for l in libs
        ],
        "elapsed_ms": round(elapsed, 2),
        "results": all_results,
        "semantic_available": sem_available,
        "semantic_reason": sem_reason,
        "rounds_info": rounds_info,
    }


def search_semantic_progressive(
    registry: LibraryRegistry,
    query: str,
    base_dir: str,
    library_names: Optional[List[str]] = None,
    parallel: int = 4,
    parent_top_k: int = 20,
    child_top_k: int = 5,
) -> Dict:
    """大chunk→小chunk递进检索（global 模式用）。

    工作流程：
      1. 用查询词在大chunk索引上检索，取 parent_top_k 个父chunk
         （调用 search_semantic_parent 完成粗筛）
      2. 对每个命中的父chunk，用查询词在其子片段上检索，取 child_top_k 个子片段
         （通过 SemanticManager.search() 带 chunk_filter 限定到父chunk范围）
      3. 返回结果包含父chunk信息和子片段定位信息（sub_text/sub_id）

    用途：
      global 模式先粗筛定位到相关大chunk（语义覆盖广，召回率高），
      再精确定位到具体子片段段落（定位准确）。
      兼顾召回率和精确度，适合万字号大chunk库的检索场景。

    Returns:
        与 search_semantic 兼容的结构，results 中每条额外含：
        - parent_chunk_id: 子片段所属的父 chunk_id
        - parent_score: 父chunk的语义相似度
    """
    t0 = time.perf_counter()
    cfg = _load_semantic_settings()
    if not cfg["enabled"]:
        return {
            "query": query, "mode": "semantic_progressive",
            "total_hits": 0, "searched_libraries": [],
            "results": [], "elapsed_ms": 0,
            "semantic_available": False,
            "semantic_reason": "语义检索通道已在设置中关闭",
        }

    # 懒加载语义管理器
    try:
        from semantic_manager import get_manager
        mgr_sem = get_manager(base_dir)
    except Exception as e:
        return {
            "query": query, "mode": "semantic_progressive",
            "total_hits": 0, "searched_libraries": [],
            "results": [], "elapsed_ms": 0,
            "semantic_available": False,
            "semantic_reason": f"语义模块加载失败：{e}",
        }

    if not mgr_sem.available():
        return {
            "query": query, "mode": "semantic_progressive",
            "total_hits": 0, "searched_libraries": [],
            "results": [], "elapsed_ms": 0,
            "semantic_available": False,
            "semantic_reason": mgr_sem.fail_reason(),
        }

    # 解析目标库
    all_libs = registry.list_libraries()
    if library_names:
        libs = [l for l in all_libs if l.name in library_names]
        missing = set(library_names) - {l.name for l in libs}
        if missing:
            raise ValueError(f"库不存在: {sorted(missing)}")
    else:
        libs = all_libs

    if not libs:
        return {
            "query": query, "mode": "semantic_progressive",
            "total_hits": 0, "searched_libraries": [],
            "results": [], "elapsed_ms": 0,
            "semantic_available": True,
        }

    # 第一步：大chunk检索（粗筛定位到相关父chunk）
    parent_result = search_semantic_parent(
        registry, query, base_dir,
        library_names=library_names,
        parallel=parallel,
        top_k=parent_top_k,
    )

    if not parent_result.get("semantic_available", False):
        return {
            "query": query, "mode": "semantic_progressive",
            "total_hits": 0, "searched_libraries": [],
            "results": [], "elapsed_ms": 0,
            "semantic_available": False,
            "semantic_reason": parent_result.get("semantic_reason", ""),
        }

    parent_hits = parent_result.get("results", [])
    if not parent_hits:
        # 父chunk索引不可用或无命中 → 回退到普通向量检索，保证有结果
        fallback = search_semantic(
            registry, query, base_dir,
            library_names=library_names, parallel=parallel,
            top_k=parent_top_k * child_top_k,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        results = fallback.get("results", [])
        for r in results:
            r["parent_chunk_id"] = r.get("chunk_id", "")
            r["parent_score"] = r.get("semantic_score", 0)
        lib_hits: Dict[str, int] = {l.name: 0 for l in libs}
        for r in results:
            lib_name = r.get("library", "")
            if lib_name in lib_hits:
                lib_hits[lib_name] += 1
        return {
            "query": query, "mode": "semantic_progressive",
            "parallel": parallel,
            "total_hits": len(results),
            "searched_libraries": [
                {"name": l.name, "note": l.note, "hits": lib_hits.get(l.name, 0)}
                for l in libs
            ],
            "elapsed_ms": round(elapsed, 2),
            "results": results,
            "semantic_available": fallback.get("semantic_available", True),
            "semantic_min_score": cfg["min_score"],
            "parent_top_k": parent_top_k,
            "child_top_k": child_top_k,
            "parent_hits_count": 0,
            "fallback_to_basic": True,
        }

    # 第二步：对每个命中的父chunk做子片段检索
    # 通过 SemanticManager.search() 带 chunk_filter 限定到父chunk范围
    lib_map = {l.name: l for l in libs}

    def _search_children(lib: Library, parent_hit: Dict) -> List[Dict]:
        """对单个父chunk检索子片段，返回带 parent_chunk_id 的结果列表。"""
        lib_root = lib.abs_path(base_dir)
        parent_id = parent_hit["chunk_id"]
        parent_score = parent_hit.get("semantic_score", 0)
        try:
            # 直接在指定父 chunk 范围内做子片段向量查询（暴力精确）
            hits = mgr_sem.search_sub_in_parent(
                lib_root, query, parent_id, top_k=child_top_k,
            )
        except Exception:
            return []

        results = []
        for h in hits:
            if h["score"] < cfg["min_score"]:
                continue
            meta = _attach_chunk_metadata(
                lib, base_dir, h["chunk_id"], h["score"],
                sub_id=h.get("sub_id", ""),
            )
            if meta is not None:
                meta["parent_chunk_id"] = parent_id
                meta["parent_score"] = parent_score
                results.append(meta)
        return results

    # 构造任务列表：(lib, parent_hit)
    tasks: List[tuple] = []
    for ph in parent_hits:
        lib = lib_map.get(ph.get("library", ""))
        if lib is not None:
            tasks.append((lib, ph))

    # 并行检索各父chunk的子片段
    child_results: List[Dict] = []
    if parallel <= 1 or len(tasks) <= 1:
        for lib, ph in tasks:
            try:
                child_results.extend(_search_children(lib, ph))
            except Exception:
                continue
    else:
        with ThreadPoolExecutor(max_workers=min(parallel, len(tasks))) as pool:
            futures = {
                pool.submit(_search_children, lib, ph): (lib, ph)
                for lib, ph in tasks
            }
            for fut in as_completed(futures):
                try:
                    child_results.extend(fut.result())
                except Exception:
                    continue

    # 按子片段相似度降序
    child_results.sort(key=lambda x: x.get("semantic_score", 0), reverse=True)

    elapsed = (time.perf_counter() - t0) * 1000

    # 统计各库命中数
    lib_hits: Dict[str, int] = {l.name: 0 for l in libs}
    for r in child_results:
        lib_name = r.get("library", "")
        if lib_name in lib_hits:
            lib_hits[lib_name] += 1

    return {
        "query": query, "mode": "semantic_progressive",
        "parallel": parallel,
        "total_hits": len(child_results),
        "searched_libraries": [
            {"name": l.name, "note": l.note, "hits": lib_hits.get(l.name, 0)}
            for l in libs
        ],
        "elapsed_ms": round(elapsed, 2),
        "results": child_results,
        "semantic_available": True,
        "semantic_min_score": cfg["min_score"],
        "parent_top_k": parent_top_k,
        "child_top_k": child_top_k,
        "parent_hits_count": len(parent_hits),
    }


def search_semantic_precise(
    registry: LibraryRegistry,
    queries: List[str],  # 多个查询词
    base_dir: str,
    library_names: Optional[List[str]] = None,
    parallel: int = 4,
    initial_top_k: int = 20,
    subchunk_parts: int = 10,
    subchunk_top_k: int = 3,
) -> Dict:
    """精准模式向量检索（小chunk→二次切分→二次向量）。

    工作流程：
      1. 对每个查询词做小chunk向量检索（search_semantic），合并去重，
         取 initial_top_k 个chunk
      2. 对每个命中的chunk，用 faiss_index.search_subchunks_of_text 做二次切分检索：
         - 把长chunk切成 subchunk_parts 份，向量化后检索最相关的 subchunk_top_k 份
         - 对所有查询词都做一次，取最高分的子片段作为最终定位
      3. 返回结果包含 subchunk_text（二次切分命中的子片段文本）和 subchunk_score

    用途：
      精准模式在语义召回的基础上做二次定位，把万字号 chunk 进一步切分到段落级，
      定位到最相关的几百字片段，提高答案片段的精确度。

    Returns:
        与 search_semantic 兼容的结构，results 中每条额外含：
        - subchunk_text: 二次切分命中的子片段文本
        - subchunk_score: 二次切分命中相似度
        - subchunk_index: 子片段序号
        - subchunk_char_start/end: 子片段在父chunk中的字符位置
        - subchunk_matched_query: 二次切分命中所用的查询词
        - matched_queries: 该chunk命中的所有查询词列表
    """
    t0 = time.perf_counter()
    cfg = _load_semantic_settings()
    if not cfg["enabled"]:
        return {
            "query": "", "mode": "semantic_precise",
            "total_hits": 0, "searched_libraries": [],
            "results": [], "elapsed_ms": 0,
            "semantic_available": False,
            "semantic_reason": "语义检索通道已在设置中关闭",
        }

    # 清洗查询词
    queries = [q.strip() for q in queries if q and q.strip()]
    if not queries:
        return {
            "query": "", "mode": "semantic_precise",
            "total_hits": 0, "searched_libraries": [],
            "results": [], "elapsed_ms": 0,
            "semantic_available": True,
        }

    # 第一步：对每个查询词做小chunk向量检索，合并去重
    # chunk_id -> {"meta": result_dict, "matched_queries": [...]}
    merged: Dict[str, Dict] = {}
    sem_available = True
    sem_reason = ""

    for q in queries:
        try:
            res = search_semantic(
                registry, q, base_dir,
                library_names=library_names,
                parallel=parallel,
                top_k=initial_top_k,
            )
        except Exception:
            continue

        if not res.get("semantic_available", True):
            sem_available = False
            sem_reason = res.get("semantic_reason", "")

        for r in res.get("results", []):
            cid = r["chunk_id"]
            if cid not in merged:
                merged[cid] = {"meta": r, "matched_queries": [q]}
            else:
                existing = merged[cid]
                # 取最高分
                if r.get("semantic_score", 0) > existing["meta"].get("semantic_score", 0):
                    existing["meta"] = r
                if q not in existing["matched_queries"]:
                    existing["matched_queries"].append(q)

    if not merged:
        return {
            "query": " ; ".join(queries), "mode": "semantic_precise",
            "total_hits": 0, "searched_libraries": [],
            "results": [], "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            "semantic_available": sem_available,
            "semantic_reason": sem_reason,
        }

    # 取 initial_top_k 个 chunk（按小chunk相似度降序）
    initial_results = sorted(
        merged.values(),
        key=lambda x: x["meta"].get("semantic_score", 0),
        reverse=True,
    )[:initial_top_k]

    # 第二步：对每个 chunk 做二次切分检索
    try:
        import faiss_index
    except Exception as e:
        return {
            "query": " ; ".join(queries), "mode": "semantic_precise",
            "total_hits": 0, "searched_libraries": [],
            "results": [], "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            "semantic_available": sem_available,
            "semantic_reason": f"faiss_index 模块加载失败：{e}",
        }

    # 预计算 lib_map，避免在并行任务中重复 list_libraries
    all_libs = registry.list_libraries()
    if library_names:
        libs = [l for l in all_libs if l.name in library_names]
    else:
        libs = all_libs
    lib_map = {l.name: l for l in libs}

    def _refine_one(item: Dict) -> Optional[Dict]:
        """对单个 chunk 做二次切分检索，返回带 subchunk_text 的结果。"""
        meta = item["meta"]
        cid = meta["chunk_id"]
        lib_name = meta.get("library", "")
        lib = lib_map.get(lib_name)
        if lib is None:
            return None

        # 加载 chunk 全文（用于二次切分）
        chunk_text = ""
        try:
            mgr = lib.manager(base_dir)
            parts = cid.split("/")
            if len(parts) == 2:
                zone = mgr.get_zone(parts[0])
                if zone:
                    cp = os.path.join(zone.chunks_dir, f"{parts[1]}.json")
                    if os.path.isfile(cp):
                        with open(cp, "r", encoding="utf-8") as f:
                            chk = json.load(f)
                        chunk_text = chk.get("text", "") or ""
        except Exception:
            pass

        if not chunk_text:
            return None

        # 对所有查询词都做一次二次切分检索，取最高分子片段
        best_sub = None
        best_query = ""
        for q in item["matched_queries"]:
            try:
                subs = faiss_index.search_subchunks_of_text(
                    chunk_text, q,
                    n_parts=subchunk_parts,
                    top_k=subchunk_top_k,
                )
            except Exception:
                continue
            if not subs:
                continue
            # search_subchunks_of_text 返回按 score 降序，取首个
            top = subs[0]
            if best_sub is None or top.get("score", 0) > best_sub.get("score", 0):
                best_sub = top
                best_query = q

        if best_sub is None:
            return None

        result = dict(meta)
        result["subchunk_text"] = best_sub.get("text", "")
        result["subchunk_score"] = best_sub.get("score", 0.0)
        result["subchunk_index"] = best_sub.get("index", 0)
        result["subchunk_char_start"] = best_sub.get("char_start", 0)
        result["subchunk_char_end"] = best_sub.get("char_end", 0)
        result["subchunk_matched_query"] = best_query
        result["matched_queries"] = item["matched_queries"]
        # 综合分：小chunk相似度 × 0.5 + 二次切分相似度 × 0.5
        result["score"] = (
            meta.get("semantic_score", 0) * 0.5
            + best_sub.get("score", 0) * 0.5
        )
        return result

    # 并行做二次切分检索
    refined_results: List[Dict] = []
    if parallel <= 1 or len(initial_results) <= 1:
        for item in initial_results:
            try:
                r = _refine_one(item)
                if r is not None:
                    refined_results.append(r)
            except Exception:
                continue
    else:
        with ThreadPoolExecutor(max_workers=min(parallel, len(initial_results))) as pool:
            futures = [pool.submit(_refine_one, item) for item in initial_results]
            for fut in as_completed(futures):
                try:
                    r = fut.result()
                    if r is not None:
                        refined_results.append(r)
                except Exception:
                    continue

    # 按综合分降序
    refined_results.sort(key=lambda x: x.get("score", 0), reverse=True)

    elapsed = (time.perf_counter() - t0) * 1000

    # 统计各库命中数
    lib_hits: Dict[str, int] = {l.name: 0 for l in libs}
    for r in refined_results:
        lib_name = r.get("library", "")
        if lib_name in lib_hits:
            lib_hits[lib_name] += 1

    return {
        "query": " ; ".join(queries), "mode": "semantic_precise",
        "parallel": parallel,
        "total_hits": len(refined_results),
        "searched_libraries": [
            {"name": l.name, "note": l.note, "hits": lib_hits.get(l.name, 0)}
            for l in libs
        ],
        "elapsed_ms": round(elapsed, 2),
        "results": refined_results,
        "semantic_available": sem_available,
        "semantic_reason": sem_reason,
        "semantic_min_score": cfg["min_score"],
        "subchunk_parts": subchunk_parts,
        "subchunk_top_k": subchunk_top_k,
    }

