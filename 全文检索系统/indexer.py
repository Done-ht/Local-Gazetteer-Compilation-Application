"""倒排索引：词项偏移索引 + mmap 精准读取，查询延迟与数据量无关。

设计（O(1) 查询）：
1. 导入时为每个 chunk 生成独立 .idx（JSON: {word: [positions]}），与 chunk JSON
   一起进 staging、一起 move 到 chunks/。原子且崩溃安全。
2. chunk 提交后合并进 zone 级分桶索引：
   - 桶文件 bucket_XX.tsv（二进制追加，每行 word\tchunk_id\tpos1,pos2\n）
   - 偏移索引 bucket_XX.offsets.json（{word: [[offset, length], ...]}，全量加载内存）
     每条 [offset, length] 精确对应桶文件中的一行字节范围。
3. 查询：hash 定位桶 → 字典查偏移表 → mmap seek 读命中行 → O(1) 与总数据量无关。
4. LRU 缓存热点词查询结果，重复查询 < 0.01ms。
5. 合并幂等：_manifest.json 记录已索引 chunk_id，重复运行只处理未索引的。
6. 偏移索引缺失时自动 rebuild（向后兼容旧索引）。

分词：
- 中文逐字（每个汉字一个 token）
- 英文/数字连续串作为一个词
- 全部小写化

桶号 = md5(word)[:2]，共 256 桶。
"""
from __future__ import annotations

import hashlib
import json
import mmap
import os
import re
from collections import OrderedDict
from typing import List, Tuple, Dict, Iterator, Optional

# 连续英文/数字串
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
# 中文逐字
_HAN_RE = re.compile(r"[\u4e00-\u9fff]")

# 词元串（run）：字母/数字/中文 加连接符（小数点、百分号、连字符、斜杠）组成的
# 连续串，如 "5.40亿立方米" "3.5万" "COVID-19" "50%"。串内无空格，视为一个
# 整体短语做精确匹配，不拆成 "5"/"40"/"亿立方米" 独立词元。
# 注意：连接符不参与分词（tokenize 看不见它们），精确校验需读 chunk 原文。
_RUN_CONNECTORS = ".．%‰％-－/／"
_RUN_RE = re.compile(
    "[A-Za-z0-9_" + re.escape(_RUN_CONNECTORS) + r"\u4e00-\u9fff]+"
)
# 词元串内的子词元：字母数字串 或 单个汉字
_RUN_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
# 引号段（"…" '…' “…” 『…』 「…」）：整段作为精确短语，段内可含空格/逗号等
_EXACT_PHRASE_RE = re.compile(
    r'"([^"]*)"|“([^”]*)”|\'([^\']*)\'|『([^』]*)』|「([^」]*)」'
)

# LRU 搜索结果缓存上限（按词项缓存）
_SEARCH_CACHE_SIZE = 4096

# 短语精确校验（读 chunk 原文）的最大候选 chunk 数，防高频词元串退化
_PHRASE_VERIFY_CAP = 1000


def tokenize(text: str) -> List[Tuple[str, int]]:
    """分词，返回 [(token, char_position), ...]。position 为字符偏移。"""
    tokens: List[Tuple[str, int]] = []
    for m in _WORD_RE.finditer(text):
        tokens.append((m.group(0).lower(), m.start()))
    for m in _HAN_RE.finditer(text):
        tokens.append((m.group(0), m.start()))
    tokens.sort(key=lambda x: x[1])
    return tokens


def build_postings(text: str) -> Dict[str, List[int]]:
    """为一个 chunk 的文本构建 word -> [positions] 映射。"""
    postings: Dict[str, List[int]] = {}
    for word, pos in tokenize(text):
        postings.setdefault(word, []).append(pos)
    return postings


def _bucket_of(word: str) -> str:
    return hashlib.md5(word.encode("utf-8")).hexdigest()[:2]


# ============================================================
#  每 chunk 独立索引文件
# ============================================================

def chunk_index_path(chunk_json_path: str) -> str:
    """chunk_000001.json -> chunk_000001.idx"""
    base, ext = os.path.splitext(chunk_json_path)
    return base + ".idx"


def write_chunk_index(idx_path: str, postings: Dict[str, List[int]]) -> None:
    """写 per-chunk 索引文件（JSON）。"""
    tmp = idx_path + ".tmp"
    os.makedirs(os.path.dirname(idx_path), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(postings, f, ensure_ascii=False)
    os.replace(tmp, idx_path)


def read_chunk_index(idx_path: str) -> Dict[str, List[int]]:
    if not os.path.isfile(idx_path):
        return {}
    with open(idx_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
#  Zone 级合并索引（幂等 + 词项偏移索引）
# ============================================================

class ZoneIndex:
    """单个 zone 的合并倒排索引。

    目录布局：
        zone/_index/
            postings/
                bucket_00.tsv          # 二进制追加，每行 word\tchunk_id\tpos
                bucket_00.offsets.json # {word: [[offset, length], ...]}
                ...
            _manifest.json             # {"indexed_chunks": [...]}

    批量模式：设 _batch_mode=True 后，merge_zone_chunks 直接跳过，
    用于批量导入期间避免重复扫描；批量结束后调用方统一调一次。
    """

    # 类级实例缓存：按 index_dir 复用，避免重复 new + 读 manifest
    _instances: Dict[str, "ZoneIndex"] = {}

    @classmethod
    def get(cls, index_dir: str) -> "ZoneIndex":
        """获取或创建 ZoneIndex 实例（按 index_dir 缓存）。

        搜索/导入频繁创建 ZoneIndex 时复用同一实例，避免重复读 manifest。
        """
        inst = cls._instances.get(index_dir)
        if inst is None:
            inst = cls(index_dir)
            cls._instances[index_dir] = inst
        return inst

    @classmethod
    def invalidate(cls, index_dir: str) -> None:
        """使某 index_dir 的缓存失效（重建索引/删除库时调用）。

        会关闭该实例持有的所有 mmap 句柄，避免 Windows 上文件被占用。
        """
        inst = cls._instances.pop(index_dir, None)
        if inst is not None:
            for mm in inst._mmap_cache.values():
                try:
                    mm.close()
                except Exception:
                    pass
            inst._mmap_cache.clear()
            inst._offsets_cache.clear()
            inst._search_cache.clear()

    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.postings_dir = os.path.join(index_dir, "postings")
        self.manifest_path = os.path.join(index_dir, "_manifest.json")
        os.makedirs(self.postings_dir, exist_ok=True)
        # 运行时缓存
        self._offsets_cache: Dict[str, Dict[str, List[List[int]]]] = {}
        self._mmap_cache: Dict[str, mmap.mmap] = {}
        self._search_cache: OrderedDict = OrderedDict()
        # 批量模式 flag：True 时 merge_zone_chunks 直接跳过
        self._batch_mode = False

    def _bucket_path(self, bucket: str) -> str:
        return os.path.join(self.postings_dir, f"bucket_{bucket}.tsv")

    def _bucket_offsets_path(self, bucket: str) -> str:
        return os.path.join(self.postings_dir, f"bucket_{bucket}.offsets.json")

    # ---- manifest ----

    def _load_manifest(self) -> set:
        if not os.path.isfile(self.manifest_path):
            return set()
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("indexed_chunks", []))

    def _save_manifest(self, indexed: set) -> None:
        tmp = self.manifest_path + ".tmp"
        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {"indexed_chunks": sorted(indexed)},
                f,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        os.replace(tmp, self.manifest_path)

    # ---- 偏移索引加载/保存 ----

    def _load_offsets(self, bucket: str) -> Dict[str, List[List[int]]]:
        """加载某桶的偏移索引到内存（带缓存）。"""
        if bucket in self._offsets_cache:
            return self._offsets_cache[bucket]
        path = self._bucket_offsets_path(bucket)
        if not os.path.isfile(path):
            data: Dict[str, List[List[int]]] = {}
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        self._offsets_cache[bucket] = data
        return data

    def _save_offsets(self, bucket: str, offsets: Dict[str, List[List[int]]]) -> None:
        tmp = self._bucket_offsets_path(bucket) + ".tmp"
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(offsets, f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, self._bucket_offsets_path(bucket))
        self._offsets_cache[bucket] = offsets

    # ---- mmap ----

    def _get_mmap(self, bucket: str) -> mmap.mmap:
        """获取桶文件的 mmap 对象（带缓存）。"""
        if bucket in self._mmap_cache:
            return self._mmap_cache[bucket]
        path = self._bucket_path(bucket)
        if not os.path.isfile(path):
            # 空文件占位
            with open(path, "wb") as f:
                pass
        f = open(path, "rb")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        self._mmap_cache[bucket] = mm
        return mm

    def _invalidate_mmap(self, bucket: str) -> None:
        """写入后使该桶的 mmap 缓存失效（下次重新打开）。"""
        if bucket in self._mmap_cache:
            try:
                self._mmap_cache[bucket].close()
            except Exception:
                pass
            del self._mmap_cache[bucket]

    # ---- 合并写入 ----

    def _write_bucket_batch(
        self,
        bucket: str,
        items: List[Tuple[str, str, List[int]]],
    ) -> None:
        """把一批 (word, chunk_id, positions) 二进制追加进桶文件，同步更新偏移索引。

        items 中每个元素生成一行：word\tchunk_id\tpos1,pos2\n

        写入前会检查文件末尾是否以换行符结尾；若否（上次写入被中断导致行截断），
        先补一个换行符对齐行边界，避免后续行拼接到残留半行上。
        """
        offsets = self._load_offsets(bucket)
        path = self._bucket_path(bucket)
        # 写入前对齐：如果文件非空且末尾不是换行，补一个换行
        # 防止上次写入中断导致的行截断污染后续写入
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            with open(path, "rb") as _fchk:
                _fchk.seek(-1, 2)  # 定位到最后一个字节
                last_byte = _fchk.read(1)
                if last_byte != b"\n":
                    with open(path, "ab") as _falign:
                        _falign.write(b"\n")
        with open(path, "ab") as f:
            for word, cid, positions in items:
                pos_str = ",".join(str(p) for p in positions)
                line = f"{word}\t{cid}\t{pos_str}\n".encode("utf-8")
                offset = f.tell()
                f.write(line)
                offsets.setdefault(word, []).append([offset, len(line)])
        self._save_offsets(bucket, offsets)
        self._invalidate_mmap(bucket)

    def merge_chunk(self, chunk_id: str, postings: Dict[str, List[int]]) -> bool:
        """把一个 chunk 的 postings 合并进分桶索引。

        幂等：若 chunk_id 已在 manifest 中则跳过。
        返回是否实际写入了。
        """
        indexed = self._load_manifest()
        if chunk_id in indexed:
            return False
        by_bucket: Dict[str, List[Tuple[str, str, List[int]]]] = {}
        for word, positions in postings.items():
            b = _bucket_of(word)
            by_bucket.setdefault(b, []).append((word, chunk_id, positions))
        for bucket, items in by_bucket.items():
            self._write_bucket_batch(bucket, items)
        indexed.add(chunk_id)
        self._save_manifest(indexed)
        return True

    def merge_from_idx_file(self, chunk_id: str, idx_path: str) -> bool:
        """从 per-chunk .idx 文件合并。"""
        postings = read_chunk_index(idx_path)
        return self.merge_chunk(chunk_id, postings)

    def merge_zone_chunks(self, chunks_dir: str, zone_id: str,
                          progress_callback=None) -> Dict:
        """扫描 chunks_dir，把所有未索引的 .idx 合并进来。返回统计。

        写入时同步记录字节偏移到 bucket_XX.offsets.json。

        批量模式（_batch_mode=True）下直接跳过，由调用方在批量结束后
        统一调一次。返回 {"merged": 0, "skipped": 0, "batch_skipped": True}。

        progress_callback(current, total, message)：可选进度回调，
        在收集阶段每处理 50 个 chunk 调用一次，让调用方能上报进度。
        """
        if self._batch_mode:
            return {"merged": 0, "skipped": 0, "batch_skipped": True}
        indexed = self._load_manifest()
        merged = 0
        skipped = 0
        if not os.path.isdir(chunks_dir):
            return {"merged": 0, "skipped": 0}
        # 先扫描一次统计待合并总数（用于进度上报）
        all_names = sorted(n for n in os.listdir(chunks_dir) if n.endswith(".idx"))
        total_to_merge = len(all_names)
        # 收集所有新 chunk，按桶聚合
        new_by_bucket: Dict[str, List[Tuple[str, str, List[int]]]] = {}
        new_chunk_ids: List[str] = []
        for i, name in enumerate(all_names):
            chunk_id = f"{zone_id}/{os.path.splitext(name)[0]}"
            if chunk_id in indexed:
                skipped += 1
                continue
            postings = read_chunk_index(os.path.join(chunks_dir, name))
            for word, positions in postings.items():
                b = _bucket_of(word)
                new_by_bucket.setdefault(b, []).append((word, chunk_id, positions))
            new_chunk_ids.append(chunk_id)
            merged += 1
            # 每 50 个或最后一个，上报进度
            if progress_callback and (merged % 50 == 0 or i == total_to_merge - 1):
                try:
                    progress_callback(i + 1, total_to_merge, "indexing")
                except Exception:
                    pass
        # 按桶批量写入
        if progress_callback and new_by_bucket:
            try:
                progress_callback(0, len(new_by_bucket), "writing")
            except Exception:
                pass
        for bi, (bucket, items) in enumerate(new_by_bucket.items()):
            self._write_bucket_batch(bucket, items)
            if progress_callback and (bi % 10 == 0 or bi == len(new_by_bucket) - 1):
                try:
                    progress_callback(bi + 1, len(new_by_bucket), "writing")
                except Exception:
                    pass
        indexed.update(new_chunk_ids)
        self._save_manifest(indexed)
        return {"merged": merged, "skipped": skipped}

    def rebuild(self, chunks_dir: str, zone_id: str) -> Dict:
        """清空并全量重建 zone 级倒排索引（含偏移索引）。

        从 chunks_dir 的所有 chunk JSON 读取 text 重新构建 postings。
        用于删除 chunk 后重建，或从旧版（无偏移索引）升级。
        返回 {"merged": N}。
        """
        # 先关闭所有可能持有该 zone mmap 的实例（包括类级缓存中的旧实例），
        # 否则 Windows 上删除 bucket_XX.tsv 会因文件被占用而失败
        cls = type(self)
        old = cls._instances.pop(self.index_dir, None)
        if old is not None and old is not self:
            for mm in old._mmap_cache.values():
                try:
                    mm.close()
                except Exception:
                    pass
            old._mmap_cache.clear()
            old._offsets_cache.clear()
            old._search_cache.clear()
        # 关闭自身的 mmap
        for mm in self._mmap_cache.values():
            try:
                mm.close()
            except Exception:
                pass
        self._mmap_cache.clear()
        self._offsets_cache.clear()
        self._search_cache.clear()
        # 再清空现有分桶文件、偏移索引与 manifest
        if os.path.isdir(self.postings_dir):
            for name in os.listdir(self.postings_dir):
                p = os.path.join(self.postings_dir, name)
                if os.path.isfile(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

        indexed: set = set()
        merged = 0
        if not os.path.isdir(chunks_dir):
            self._save_manifest(indexed)
            return {"merged": 0}
        # 先收集所有 chunk 的 postings，按桶聚合
        all_by_bucket: Dict[str, List[Tuple[str, str, List[int]]]] = {}
        for name in sorted(os.listdir(chunks_dir)):
            if not (name.startswith("chunk_") and name.endswith(".json")):
                continue
            chunk_name = os.path.splitext(name)[0]
            chunk_id = f"{zone_id}/{chunk_name}"
            path = os.path.join(chunks_dir, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    chunk = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            text = chunk.get("text", "")
            postings = build_postings(text)
            for word, positions in postings.items():
                b = _bucket_of(word)
                all_by_bucket.setdefault(b, []).append((word, chunk_id, positions))
            indexed.add(chunk_id)
            merged += 1
        # 按桶批量写入
        for bucket, items in all_by_bucket.items():
            self._write_bucket_batch(bucket, items)
        self._save_manifest(indexed)
        return {"merged": merged}

    def cleanup_merged_idx(self, chunks_dir: str) -> int:
        """删除已合并进 zone 索引的 per-chunk .idx 文件，释放存储空间。

        仅删除 manifest 中已记录的 chunk 对应的 .idx。
        返回删除的文件数。
        """
        indexed = self._load_manifest()
        if not indexed or not os.path.isdir(chunks_dir):
            return 0
        removed = 0
        for name in os.listdir(chunks_dir):
            if not name.endswith(".idx"):
                continue
            chunk_name = os.path.splitext(name)[0]
            for cid in indexed:
                if cid.endswith("/" + chunk_name):
                    os.remove(os.path.join(chunks_dir, name))
                    removed += 1
                    break
        return removed

    # ---- 偏移索引完整性检测与自动重建 ----

    def ensure_offset_index(self, chunks_dir: str, zone_id: str) -> bool:
        """检测偏移索引是否完整，缺失则全量重建。

        返回是否执行了重建。
        判定：manifest 有记录但任意已存在桶缺少 offsets.json → 需要重建。
        """
        indexed = self._load_manifest()
        if not indexed:
            return False
        # 检查是否存在任何 offsets.json
        has_offsets = any(
            name.endswith(".offsets.json")
            for name in os.listdir(self.postings_dir)
            if os.path.isfile(os.path.join(self.postings_dir, name))
        ) if os.path.isdir(self.postings_dir) else False
        if has_offsets:
            return False
        # 缺少偏移索引，重建
        self.rebuild(chunks_dir, zone_id)
        return True

    # ---- 查询 ----

    def search_term(self, word: str) -> Dict[str, List[int]]:
        """搜索单词，返回 {chunk_id: [positions]}。

        优先用偏移索引 + mmap 精准读取（O(1)）；
        偏移索引缺失时回退线性扫描（O(n)，向后兼容）。
        结果走 LRU 缓存。
        """
        word = word.lower()
        # LRU 缓存命中
        cache_key = word
        if cache_key in self._search_cache:
            self._search_cache.move_to_end(cache_key)
            return self._search_cache[cache_key]

        bucket = _bucket_of(word)
        offsets = self._load_offsets(bucket)
        result: Dict[str, List[int]] = {}

        if offsets:
            # 快速路径：偏移索引精准读取
            path = self._bucket_path(bucket)
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                mm = self._get_mmap(bucket)
                for offset, length in offsets.get(word, []):
                    try:
                        line = mm[offset:offset + length].decode("utf-8")
                    except Exception:
                        continue
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) != 3:
                        # 偏移索引与文件不对齐（写入中断导致行错位），跳过
                        continue
                    _w, cid, pos_str = parts
                    # 校验读出的 word 与查询 word 一致；不一致说明偏移错位
                    if _w != word:
                        continue
                    # 容错解析 positions（个别非数字则跳过该 position）
                    positions = []
                    for p in pos_str.split(","):
                        if not p:
                            continue
                        try:
                            positions.append(int(p))
                        except ValueError:
                            continue
                    if positions:
                        result.setdefault(cid, []).extend(positions)
        else:
            # 回退路径：线性扫描（旧索引兼容）
            path = self._bucket_path(bucket)
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    for raw_line in f:
                        line = raw_line.decode("utf-8")
                        parts = line.rstrip("\n").split("\t")
                        if len(parts) != 3:
                            continue
                        w, cid, pos_str = parts
                        if w == word:
                            positions = []
                            for p in pos_str.split(","):
                                if not p:
                                    continue
                                try:
                                    positions.append(int(p))
                                except ValueError:
                                    continue
                            if positions:
                                result.setdefault(cid, []).extend(positions)

        # 写入 LRU 缓存
        self._search_cache[cache_key] = result
        self._search_cache.move_to_end(cache_key)
        if len(self._search_cache) > _SEARCH_CACHE_SIZE:
            self._search_cache.popitem(last=False)
        return result

    def search(self, query: str) -> Dict[str, Dict[str, List[int]]]:
        """搜索查询中的所有词项。返回 {word: {chunk_id: [positions]}}。

        匹配单位是"词元串"（run）：查询按空格和其它标点切成连续串，串内
        （字母/数字/中文/连接符）视为一个整体：
        - 单词元串（纯英文数字或单个汉字）：直接查倒排索引，与原逻辑一致
        - 纯中文多字串：短语匹配，只保留各字连续出现的位置（与原逻辑一致）
        - 混合串（如 "5.40亿立方米"、"3.5万"、"50%"）：整体精确匹配。
          先用倒排索引做位置对齐预筛（各子词元在算术偏移上对齐），
          若串内含连接符（"." 不进索引，无法区分 "5.40"/"5-40"/"5，40"），
          再加载候选 chunk 原文逐位置做精确子串校验。
        - 引号段（"…" ‘…’ 「…」等）：整段作为精确短语原子，段内可含
          空格、逗号等任意字符（用于同义词组语法 ("5,40"，"1.5万") 的引号词项）。
        """
        out: Dict[str, Dict[str, List[int]]] = {}
        seen_tokens = set()
        # 先切出引号段：段内整段精确匹配；段外按词元串处理
        last = 0
        segments: List[Tuple[str, bool]] = []
        for m in _EXACT_PHRASE_RE.finditer(query):
            segments.append((query[last:m.start()], False))
            phrase = next((g for g in m.groups() if g is not None), "")
            segments.append((phrase, True))
            last = m.end()
        segments.append((query[last:], False))

        for seg, is_exact in segments:
            if not is_exact:
                seg = seg.strip()
            if not seg:
                continue
            if is_exact:
                # 引号段：整段一个精确短语（保留段内全部字符，含空格/逗号）
                self._search_phrase(seg, out, seen_tokens)
                continue
            for run in _RUN_RE.findall(seg):
                toks = [(m.group(0).lower(), m.start())
                        for m in _RUN_TOKEN_RE.finditer(run)]
                if not toks:
                    continue
                # 词元串的有效范围：首词元起点到串尾（去头部连接符，保留尾部
                # 连接符，如 "%50" 取 "50"，"50%" 取 "50%"）
                run_exact = run[toks[0][1]:]
                self._search_phrase(run_exact, out, seen_tokens)
        return out

    def _search_phrase(self, phrase: str,
                       out: Dict[str, Dict[str, List[int]]],
                       seen_tokens: set) -> None:
        """匹配单个词元串/引号段（含连接符原文），结果写入 out。"""
        toks = [(m.group(0).lower(), m.start())
                for m in _RUN_TOKEN_RE.finditer(phrase)]
        if not toks:
            return
        has_gap = len(phrase) > sum(len(t) for t, _ in toks)

        if len(toks) == 1 and not has_gap:
            # 单词元：直接查倒排
            word = toks[0][0]
            if word not in seen_tokens:
                out[word] = self.search_term(word)
                seen_tokens.add(word)
            return

        # 多词元：位置对齐短语匹配（对含连接符的串是预筛）
        per_tok = [self.search_term(t) for t, _ in toks]
        common_cids = None
        for tm in per_tok:
            cids = set(tm.keys())
            common_cids = cids if common_cids is None else (common_cids & cids)
            if not common_cids:
                break
        phrase_result: Dict[str, List[int]] = {}
        verified_count = 0
        if common_cids:
            for cid in common_cids:
                pos_sets = [set(per_tok[i].get(cid, [])) for i in range(len(toks))]
                aligned: List[int] = []
                for p0 in per_tok[0].get(cid, []):
                    base = p0 - toks[0][1]
                    ok = True
                    for k in range(1, len(toks)):
                        if (base + toks[k][1]) not in pos_sets[k]:
                            ok = False
                            break
                    if ok:
                        aligned.append(base)
                if not aligned:
                    continue
                if has_gap:
                    # 含连接符：读原文逐位置精确校验。
                    # 高频词元串（如 "50%" 只有一个子词元 "50"，对齐不具区分度）
                    # 候选可能很多，加读取上限防退化；超限部分降级为仅索引对齐。
                    text = (self._load_chunk_text(cid)
                            if verified_count < _PHRASE_VERIFY_CAP else None)
                    if text is not None:
                        verified_count += 1
                        pl = phrase.lower()
                        aligned = [p for p in aligned
                                   if text[p:p + len(phrase)].lower() == pl]
                        if not aligned:
                            continue
                phrase_result[cid] = aligned
        key = phrase.lower()
        if key not in out:
            out[key] = phrase_result
        seen_tokens.update(t for t, _ in toks)
        seen_tokens.add(key)

    def _load_chunk_text(self, cid: str) -> Optional[str]:
        """按 chunk_id 读取该 chunk 的正文文本（供短语精确校验）。

        chunk_id 形如 "zone_001/chunk_000001"；chunks 目录是索引目录的兄弟目录。
        文件缺失/损坏时返回 None，调用方降级为仅索引对齐结果。
        """
        name = cid.rsplit("/", 1)[-1]
        path = os.path.join(os.path.dirname(self.index_dir),
                            "chunks", f"{name}.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                chunk = json.load(f)
            return chunk.get("text", "") or ""
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None


# ============================================================
#  多 zone 查询
# ============================================================

class MultiZoneIndex:
    """跨多 zone 查询，合并结果。"""

    def __init__(self, zone_index_dirs: List[str]):
        self.indexes = [ZoneIndex(d) for d in zone_index_dirs]

    def search(self, query: str) -> Dict[str, Dict[str, List[int]]]:
        """返回 {chunk_id: {word: [positions]}}。"""
        merged: Dict[str, Dict[str, List[int]]] = {}
        for idx in self.indexes:
            res = idx.search(query)
            for word, chunks in res.items():
                for cid, positions in chunks.items():
                    merged.setdefault(cid, {}).setdefault(word, []).extend(positions)
        return merged
