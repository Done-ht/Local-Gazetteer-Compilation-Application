"""Faiss HNSW 向量索引（单库级）。

职责：
    - 为单个库构建 / 加载 / 查询 HNSW 向量索引
    - chunk_id 与 faiss 内部行号的双向映射持久化
    - 状态机：idle → building → ready / failed，可被查询端热加载
    - 增量追加（导入新文件后追加向量）与全量重建

存储布局（每个库根目录下）：
    <lib_root>/_semantic/
        index.faiss              # HNSW 索引二进制
        chunk_ids.json           # 行号 → chunk_id 映射
        meta.json                # 元信息：维度、向量数、构建时间、模型名

设计要点：
    - 度量：内积（IP）。向量已 L2 归一化，IP = 余弦相似度
    - HNSW 参数：M=32, efConstruction=200, efSearch=64
      （10.5 万 chunk 单查询 < 50ms，召回率 > 95%）
    - 查询时只读不写，多线程并发安全
    - 构建时通过状态机保护：构建中查询走旧索引（若已 ready），
      构建完成原子切换
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from userdata import auth_base_dir as _auth_base_dir


# HNSW 默认参数（召回率与速度的平衡点）
# M：每个节点的最大邻居数，越大召回越高、内存占用越大
# ef_construction：构建时搜索宽度，越大构建越慢但索引质量越好
# ef_search：查询时搜索宽度，越大召回越高、查询越慢
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64

# 语义子分块大小（字符数）
# bge-small-zh 最大输入 512 token（约 500 中文字），超过会被截断丢失语义
# 父 chunk 向量化时切成 ≤此长度 的子片段，每个子片段独立向量化
# 查询时按父 chunk 聚合（取最高子片段得分），保留"最相关段落"信息
DEFAULT_SUB_CHUNK_SIZE = 500

# 句子分隔符（用于子切片时优先在句末切，避免切断句子）
# 按优先级排序：换行 > 句号 > 问号 > 感叹号 > 分号 > 空格
_SENTENCE_DELIMITERS = ["\n", "。", "！", "？", "；", " "]


def _split_into_subchunks(text: str, max_size: int = DEFAULT_SUB_CHUNK_SIZE) -> List[str]:
    """把长文本切成 ≤max_size 字符的子片段。

    策略：
      1. 优先在句子分隔符（换行/句号/问号等）处切，避免切断句子
      2. 累积到接近 max_size 时找最近的分隔点切
      3. 段落本身超过 max_size 时硬切（兜底）

    Args:
        text: 原始文本（父 chunk 的完整文本）
        max_size: 单个子片段最大字符数

    Returns:
        子片段文本列表；若原文 ≤max_size 则返回 [text]
    """
    if not text:
        return []
    if len(text) <= max_size:
        return [text]

    subchunks: List[str] = []
    pos = 0
    total = len(text)
    while pos < total:
        # 理想切点：pos + max_size
        end = pos + max_size
        if end >= total:
            subchunks.append(text[pos:])
            break

        # 在 [end - max_size//4, end] 范围内找最近的分隔符
        # 向前回溯 1/4 长度找切点，避免切得太短
        search_start = max(pos + max_size // 2, end - max_size // 4)
        best_cut = -1
        for delim in _SENTENCE_DELIMITERS:
            # 从 end 向前找分隔符
            cut = text.rfind(delim, search_start, end + 1)
            if cut > best_cut:
                best_cut = cut
                # 找到换行或句号就停止（已是较优切点）
                if delim in ("\n", "。"):
                    break

        if best_cut <= pos:
            # 没找到合适分隔符，硬切
            cut_pos = end
        else:
            # 切在分隔符之后（保留分隔符在当前片段末尾）
            cut_pos = best_cut + 1

        subchunks.append(text[pos:cut_pos])
        pos = cut_pos

    return subchunks


# 索引状态
STATUS_IDLE = "idle"            # 未构建（库刚创建或被清理）
STATUS_BUILDING = "building"    # 构建中（后台线程执行）
STATUS_READY = "ready"          # 已就绪（可查询）
STATUS_FAILED = "failed"        # 构建失败


class FaissIndex:
    """单个库的 Faiss HNSW 索引管理器。

    一个库对应一个 FaissIndex 实例。构建在后台线程执行，
    查询时若索引未就绪则返回空结果（不阻塞）。
    """

    def __init__(self, lib_root: str, dim: int = 384):
        """初始化。

        Args:
            lib_root: 库根目录（绝对路径）
            dim: 向量维度，默认 384（bge-small-zh）
        """
        self.lib_root = os.path.abspath(lib_root)
        self.dim = dim
        self.semantic_dir = os.path.join(self.lib_root, "_semantic")
        self.index_path = os.path.join(self.semantic_dir, "index.faiss")
        self.chunk_ids_path = os.path.join(self.semantic_dir, "chunk_ids.json")
        self.sub_to_parent_path = os.path.join(self.semantic_dir, "sub_to_parent.json")
        self.meta_path = os.path.join(self.semantic_dir, "meta.json")
        # 断点续建文件（构建中持久化进度，完成后删除）
        # vectors.part.f32：已向量化子片段的原始向量（float32 二进制，追加写）
        # sub_ids.part.json：已完成的子片段 ID 列表（顺序与 vectors.part.f32 对齐）
        # sub_to_parent.part.json：子片段→父chunk映射
        # parent_ids.part.json：父 chunk_id 顺序列表（构建开始时一次性写入）
        # build_state.json：构建状态（total/completed/dim/model_name/sub_chunk_size/started_at）
        self.vectors_part_path = os.path.join(self.semantic_dir, "vectors.part.f32")
        self.sub_ids_part_path = os.path.join(self.semantic_dir, "sub_ids.part.json")
        self.sub_to_parent_part_path = os.path.join(self.semantic_dir, "sub_to_parent.part.json")
        self.parent_ids_part_path = os.path.join(self.semantic_dir, "parent_ids.part.json")
        self.build_state_path = os.path.join(self.semantic_dir, "build_state.json")
        # 父 chunk 级向量索引（大chunk模式检索用）
        # 通过子片段向量池化生成，独立 FAISS 索引，查询时直接返回父chunk
        self.parent_index_path = os.path.join(self.semantic_dir, "index_parent.faiss")
        self.parent_ids_path = os.path.join(self.semantic_dir, "parent_ids.json")
        # 运行时：父 chunk 索引对象与映射
        self._parent_index = None
        self._parent_ids: List[str] = []              # 行号 → 父chunk_id
        self._parent_id_to_row: Dict[str, int] = {}    # 父chunk_id → 行号

        # 运行时状态
        self._status = STATUS_IDLE
        self._status_lock = threading.RLock()
        self._index = None              # faiss.Index 对象
        # 子片段相关：向量索引按子片段建，查询时聚合回父 chunk
        # _chunk_ids 存子片段 ID（如 "zone_001/chunk_000001#0"）
        # _sub_to_parent 存 子片段ID → 父chunk_id 映射
        # _parent_chunks 存父 chunk_id 集合（用于 needs_rebuild 比较）
        self._chunk_ids: List[str] = []  # 行号 → 子片段 ID
        self._chunk_id_to_row: Dict[str, int] = {}  # 子片段 ID → 行号
        self._sub_to_parent: Dict[str, str] = {}    # 子片段 ID → 父 chunk_id
        self._parent_chunks: List[str] = []         # 父 chunk_id 列表（排序）
        self._fail_reason: str = ""
        self._progress: Dict[str, Any] = {}  # 构建进度信息

        # 查询锁：与构建锁分离，构建时不阻塞查询（查询走旧索引）
        # 仅在 hot_reload 切换索引时短暂加锁
        self._query_lock = threading.RLock()

        # 启动时尝试加载已构建的索引
        self._try_load()

    # ----------------------------------------------------------
    #  状态查询
    # ----------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        """返回当前状态快照（前端展示用）。"""
        with self._status_lock:
            return {
                "status": self._status,
                "fail_reason": self._fail_reason,
                "vector_count": len(self._chunk_ids),       # 子片段数（=向量数）
                "parent_chunk_count": len(self._parent_chunks),  # 父 chunk 数
                "dim": self.dim,
                "progress": dict(self._progress),
                "index_exists": os.path.isfile(self.index_path),
            }

    def is_ready(self) -> bool:
        """索引是否就绪可查询。"""
        with self._status_lock:
            return self._status == STATUS_READY and self._index is not None

    def is_building(self) -> bool:
        with self._status_lock:
            return self._status == STATUS_BUILDING

    # ----------------------------------------------------------
    #  索引加载
    # ----------------------------------------------------------
    def _try_load(self) -> bool:
        """启动时尝试从磁盘加载已构建的索引。"""
        if not os.path.isfile(self.index_path) or not os.path.isfile(self.chunk_ids_path):
            return False
        try:
            import faiss
            import numpy as np
            with self._status_lock:
                # Windows 上 faiss.read_index 不支持含非 ASCII 字符的路径，
                # 改用 Python 读取字节后交给 faiss 反序列化（跨平台一致）
                # faiss.deserialize_index 要求 numpy 数组（而非 bytes）
                with open(self.index_path, "rb") as f:
                    blob = f.read()
                arr = np.frombuffer(blob, dtype=np.uint8)
                self._index = faiss.deserialize_index(arr)
                # 从索引对象读取实际维度（与构建时模型维度一致，
                # 避免硬编码 384 与实际模型维度不匹配）
                self.dim = int(self._index.d)
                with open(self.chunk_ids_path, "r", encoding="utf-8") as f:
                    self._chunk_ids = json.load(f)
                self._chunk_id_to_row = {cid: i for i, cid in enumerate(self._chunk_ids)}
                # 加载子片段 → 父 chunk 映射
                # 兼容旧索引：无映射文件时按"子片段 ID == 父 chunk_id"处理
                if os.path.isfile(self.sub_to_parent_path):
                    with open(self.sub_to_parent_path, "r", encoding="utf-8") as f:
                        self._sub_to_parent = json.load(f)
                else:
                    self._sub_to_parent = {cid: cid for cid in self._chunk_ids}
                # 推导父 chunk 列表（去重 + 排序）
                self._parent_chunks = sorted(set(self._sub_to_parent.values()))
                # 设置 HNSW 查询参数
                self._set_ef_search(self._index)
                self._status = STATUS_READY
            # 读取元信息
            meta = self._load_meta()
            print(f"[faiss] 加载已建索引：{self.lib_root} "
                  f"(向量数={len(self._chunk_ids)}, "
                  f"父chunk数={len(self._parent_chunks)}, "
                  f"构建于={meta.get('built_at', '?')})", flush=True)
            # 尝试加载父chunk索引（大chunk模式检索用，旧索引可能没有，忽略失败）
            self._try_load_parent_index()
            return True
        except Exception as e:
            print(f"[faiss] 加载索引失败 {self.lib_root}: {e}", flush=True)
            with self._status_lock:
                self._status = STATUS_FAILED
                self._fail_reason = f"加载已建索引失败：{e}"
            return False

    @staticmethod
    def _set_ef_search(index) -> None:
        """设置 HNSW 查询时的 efSearch 参数。"""
        try:
            import faiss
            # HNSW 索引对象的 efSearch 是可调参数
            index.hnsw.efSearch = HNSW_EF_SEARCH
        except Exception:
            pass  # 非 HNSW 索引或参数不存在，忽略

    def _load_meta(self) -> Dict:
        try:
            if os.path.isfile(self.meta_path):
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_meta(self, vector_count: int, model_name: str,
                   build_time_ms: float) -> None:
        meta = {
            "dim": self.dim,
            "vector_count": vector_count,
            "model_name": model_name,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "build_time_ms": round(build_time_ms, 1),
            "hnsw_m": HNSW_M,
            "hnsw_ef_construction": HNSW_EF_CONSTRUCTION,
            "hnsw_ef_search": HNSW_EF_SEARCH,
        }
        os.makedirs(self.semantic_dir, exist_ok=True)
        tmp = self.meta_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.meta_path)

    # ----------------------------------------------------------
    #  索引构建（在后台线程调用）
    # ----------------------------------------------------------
    def build(self, chunk_texts: List[Tuple[str, str]],
              embedder=None,
              progress_callback=None) -> bool:
        """全量构建 HNSW 索引。

        在后台线程调用。构建期间旧索引仍可查询，构建完成后原子切换。

        子分块策略：
          - 输入是父 chunk（如 10000 字），向量化前先切成 ≤500 字子片段
          - 每个子片段独立向量化，存入 HNSW 索引
          - 查询时按父 chunk 聚合（取最高子片段得分）
          - 这样既不浪费模型 512 token 上限，又能保留长 chunk 的完整语义

        Args:
            chunk_texts: [(parent_chunk_id, text), ...] 列表，按 chunk_id 排序
            embedder: Embedder 实例，None 时用共享实例
            progress_callback: 可选回调 fn(current, total, stage)
                              stage: "embedding" / "building" / "saving"

        Returns:
            True 成功，False 失败
        """
        from embedding import Embedder
        if embedder is None:
            embedder = Embedder.get_shared()

        # 检查依赖
        if not embedder.available():
            with self._status_lock:
                self._status = STATUS_FAILED
                self._fail_reason = embedder.fail_reason() or "向量模型不可用"
            return False

        # 读取子分块大小配置
        sub_chunk_size = self._load_sub_chunk_size()

        # 标记构建中
        with self._status_lock:
            if self._status == STATUS_BUILDING:
                # 已有构建在进行，跳过
                return False
            self._status = STATUS_BUILDING
            self._fail_reason = ""
            self._progress = {"current": 0, "total": len(chunk_texts),
                              "stage": "embedding", "started_at": time.time()}

        t0 = time.perf_counter()
        try:
            import faiss
            import numpy as np

            parent_total = len(chunk_texts)
            if parent_total == 0:
                # 空库：写一个空索引占位，避免反复触发构建
                os.makedirs(self.semantic_dir, exist_ok=True)
                self._save_index_to_disk(None, [], {}, "", 0.0)
                self._cleanup_partial_files()
                with self._status_lock:
                    self._status = STATUS_READY
                    self._parent_chunks = []
                    self._progress = {"current": 0, "total": 0,
                                      "stage": "done", "started_at": time.time()}
                return True

            # 1. 子切片：把每个父 chunk 切成 ≤sub_chunk_size 字的子片段
            # sub_id 形如 "zone_001/chunk_000001#0"，#后是子片段序号
            sub_ids: List[str] = []
            sub_texts: List[str] = []
            sub_to_parent: Dict[str, str] = {}
            parent_ids_ordered: List[str] = []
            for parent_id, text in chunk_texts:
                parent_ids_ordered.append(parent_id)
                parts = _split_into_subchunks(text, max_size=sub_chunk_size)
                if not parts:
                    # 空文本：仍占一个向量位，避免父 chunk 在索引中失踪
                    parts = [""]
                for i, sub_text in enumerate(parts):
                    sub_id = f"{parent_id}#{i}"
                    sub_ids.append(sub_id)
                    sub_texts.append(sub_text)
                    sub_to_parent[sub_id] = parent_id

            sub_total = len(sub_ids)

            # ===== 断点续建：检测并加载已完成的子片段 =====
            resumed_count = 0
            resumed_vecs = None
            if self._has_partial_build():
                try:
                    done_ids, done_map, done_parents, done_vecs = \
                        self._load_partial_vectors()
                    # 校验：父 chunk 集合必须与当前一致（避免库内容变化后续建错）
                    if done_parents == parent_ids_ordered and done_vecs.size > 0:
                        resumed_count = len(done_ids)
                        resumed_vecs = done_vecs
                        print(f"[faiss] 检测到未完成构建，断点续建："
                              f"已完成 {resumed_count}/{sub_total} 子片段",
                              flush=True)
                        # 更新进度
                        with self._status_lock:
                            self._progress["current"] = resumed_count
                            self._progress["total"] = sub_total
                            self._progress["stage"] = "embedding"
                            self._progress["resumed"] = True
                    else:
                        # 父 chunk 集合不一致（库内容已变化），丢弃旧断点
                        print(f"[faiss] 检测到旧断点但父chunk集合已变化，"
                              f"丢弃旧断点重新构建", flush=True)
                        self._cleanup_partial_files()
                except Exception as e:
                    print(f"[faiss] 加载断点失败，从头构建：{e}", flush=True)
                    self._cleanup_partial_files()

            # 若不是续建，初始化断点文件
            if resumed_count == 0:
                os.makedirs(self.semantic_dir, exist_ok=True)
                # 清空旧文件（避免追加到残留数据）
                for p in [self.vectors_part_path, self.sub_ids_part_path,
                          self.sub_to_parent_part_path]:
                    if os.path.isfile(p):
                        os.remove(p)
                # 写入父 chunk 列表（构建开始时一次性写入，续建时校验用）
                tmp = self.parent_ids_part_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(parent_ids_ordered, f, ensure_ascii=False)
                os.replace(tmp, self.parent_ids_part_path)
                # 写入初始构建状态
                self._write_build_state({
                    "total": sub_total,
                    "completed": 0,
                    "dim": self.dim,
                    "model_name": embedder.model_name,
                    "sub_chunk_size": sub_chunk_size,
                    "started_at": time.time(),
                    "done": False,
                })

            print(f"[faiss] 子切片完成：{parent_total} 父chunk → {sub_total} 子片段 "
                  f"(平均 {sub_total / max(1, parent_total):.1f} 片/chunk, "
                  f"上限={sub_chunk_size}字"
                  f"{f', 已续建 {resumed_count} 片' if resumed_count else ''})",
                  flush=True)

            # 2. 批量向量子片段（断点续建：跳过已完成的部分）
            def _emb_cb(cur, tot):
                with self._status_lock:
                    self._progress["current"] = cur
                    self._progress["total"] = tot
                    self._progress["stage"] = "embedding"
                if progress_callback:
                    try:
                        progress_callback(cur, tot, "embedding")
                    except Exception:
                        pass

            batch_size = 64
            new_vecs_list = []
            # 从 resumed_count 开始继续向量化
            for i in range(resumed_count, sub_total, batch_size):
                batch = sub_texts[i:i + batch_size]
                batch_ids = sub_ids[i:i + batch_size]
                batch_map = {sid: sub_to_parent[sid] for sid in batch_ids}
                v = embedder.encode(batch, show_progress=False)
                if v is None:
                    raise RuntimeError("向量化失败")
                new_vecs_list.append(v)
                # 即时持久化本批向量（断点续建的核心）
                self._append_batch_to_disk(v, batch_ids, batch_map)
                # 更新构建状态
                completed = min(i + batch_size, sub_total)
                self._write_build_state({
                    "total": sub_total,
                    "completed": completed,
                    "dim": self.dim,
                    "model_name": embedder.model_name,
                    "sub_chunk_size": sub_chunk_size,
                    "started_at": time.time() if resumed_count == 0 else
                                  (self._read_build_state() or {}).get("started_at", time.time()),
                    "done": False,
                })
                _emb_cb(completed, sub_total)

            # 合并向量：续建部分 + 新向量化部分
            if new_vecs_list:
                new_vecs = np.vstack(new_vecs_list).astype("float32")
                if resumed_vecs is not None and resumed_vecs.size > 0:
                    vecs = np.vstack([resumed_vecs, new_vecs]).astype("float32")
                else:
                    vecs = new_vecs
            else:
                vecs = resumed_vecs if resumed_vecs is not None else \
                       np.zeros((0, self.dim), dtype=np.float32)

            # 确保维度一致
            if vecs.shape[0] > 0 and vecs.shape[1] != self.dim:
                self.dim = int(vecs.shape[1])

            # 3. 构建 HNSW 索引
            with self._status_lock:
                self._progress["stage"] = "building"
                self._progress["current"] = 0
            if progress_callback:
                try:
                    progress_callback(0, sub_total, "building")
                except Exception:
                    pass

            # 度量：内积（向量已 L2 归一化，IP=余弦相似度）
            quantizer = faiss.IndexHNSWFlat(self.dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
            quantizer.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
            # 直接用 HNSW 作为顶层索引（无需 IVF 量化，10.5 万级规模 HNSW 足够快）
            index = quantizer
            index.add(vecs)

            # 4. 持久化最终索引
            with self._status_lock:
                self._progress["stage"] = "saving"
            if progress_callback:
                try:
                    progress_callback(sub_total, sub_total, "saving")
                except Exception:
                    pass

            elapsed = (time.perf_counter() - t0) * 1000
            self._save_index_to_disk(index, sub_ids, sub_to_parent,
                                      embedder.model_name, elapsed)

            # 5. 原子切换：替换内存中的索引对象
            with self._query_lock:
                self._set_ef_search(index)
                with self._status_lock:
                    self._index = index
                    self._chunk_ids = sub_ids
                    self._chunk_id_to_row = {cid: i for i, cid in enumerate(sub_ids)}
                    self._sub_to_parent = sub_to_parent
                    self._parent_chunks = parent_ids_ordered
                    self._status = STATUS_READY
                    self._progress = {"current": sub_total, "total": sub_total,
                                      "stage": "done",
                                      "elapsed_ms": round(elapsed, 1),
                                      "parent_count": parent_total,
                                      "started_at": time.time()}

            # 6. 清理断点文件
            self._cleanup_partial_files()

            # 7. 构建父 chunk 级向量索引（大chunk模式检索用）
            # 通过子片段向量最大池化生成父chunk向量
            # 策略：对每个父chunk的所有子片段向量取逐元素最大值
            #       保留最强语义信号，相比平均池化更能体现"是否包含某语义"
            t_pool = time.perf_counter()
            try:
                self._build_parent_index(vecs, sub_ids, parent_ids_ordered)
                pool_ms = (time.perf_counter() - t_pool) * 1000
                print(f"[faiss] 父chunk索引构建完成：{len(parent_ids_ordered)} 父chunk "
                      f"(池化耗时={pool_ms:.0f}ms)", flush=True)
            except Exception as e:
                # 父chunk索引构建失败不影响主索引使用
                print(f"[faiss] 父chunk索引构建失败（不影响小chunk模式）: {e}",
                      flush=True)

            print(f"[faiss] 索引构建完成：{self.lib_root} "
                  f"(父chunk={parent_total}, 子片段={sub_total}, "
                  f"用时={elapsed/1000:.1f}s"
                  f"{f', 含续建 {resumed_count} 片' if resumed_count else ''})",
                  flush=True)
            return True

        except Exception as e:
            import traceback
            traceback.print_exc()
            with self._status_lock:
                self._status = STATUS_FAILED
                self._fail_reason = f"构建失败：{e}"
                self._progress["stage"] = "failed"
            print(f"[faiss] 索引构建失败 {self.lib_root}: {e}", flush=True)
            print(f"[faiss] 已完成的向量已保留在磁盘，"
                  f"下次启动可从断点续建", flush=True)
            return False

    def _load_sub_chunk_size(self) -> int:
        """从 settings 读取子分块大小（默认 500）。"""
        try:
            from settings import SettingsStore
            store = SettingsStore(_auth_base_dir())
            return int(store.get("semantic_sub_chunk_size", DEFAULT_SUB_CHUNK_SIZE))
        except Exception:
            return DEFAULT_SUB_CHUNK_SIZE

    # ----------------------------------------------------------
    #  父 chunk 级向量索引（大chunk模式检索用）
    # ----------------------------------------------------------
    def _build_parent_index(self, sub_vecs: "np.ndarray", sub_ids: List[str],
                             parent_ids_ordered: List[str]) -> None:
        """构建父 chunk 级向量索引（最大池化）。

        策略：对每个父chunk的所有子片段向量取逐元素最大值
              保留最强语义信号，"父chunk是否包含某语义"的体现优于平均池化

        Args:
            sub_vecs: (N, dim) 子片段向量矩阵
            sub_ids: 长度 N 的子片段 ID 列表（与 sub_vecs 行对齐）
            parent_ids_ordered: 父 chunk_id 顺序列表（去重后）
        """
        import faiss
        import numpy as np

        # 建立父chunk → 子片段行号列表的映射
        parent_to_rows: Dict[str, List[int]] = {pid: [] for pid in parent_ids_ordered}
        for i, sid in enumerate(sub_ids):
            # sub_id 形如 "zone_001/chunk_000001#0"，父ID是去掉 #后缀
            pid = sid.rsplit("#", 1)[0] if "#" in sid else sid
            if pid in parent_to_rows:
                parent_to_rows[pid].append(i)

        # 对每个父chunk做最大池化
        parent_vecs = np.zeros((len(parent_ids_ordered), self.dim), dtype=np.float32)
        for j, pid in enumerate(parent_ids_ordered):
            rows = parent_to_rows[pid]
            if rows:
                # 取该父chunk所有子片段向量的逐元素最大值
                parent_vecs[j] = sub_vecs[rows].max(axis=0)
            # 无子片段的父chunk保持零向量（理论上不会出现）

        # L2 归一化（保证内积 = 余弦相似度）
        norms = np.linalg.norm(parent_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        parent_vecs = parent_vecs / norms

        # 构建 HNSW 索引
        parent_index = faiss.IndexHNSWFlat(self.dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
        parent_index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        parent_index.add(parent_vecs)
        self._set_ef_search(parent_index)

        # 持久化（与子片段索引相同的序列化策略，兼容中文路径）
        os.makedirs(self.semantic_dir, exist_ok=True)
        # 父索引
        blob = faiss.serialize_index(parent_index)
        with open(self.parent_index_path, "wb") as f:
            f.write(blob)
        # 父chunk_id 列表
        tmp = self.parent_ids_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(parent_ids_ordered, f, ensure_ascii=False)
        os.replace(tmp, self.parent_ids_path)

        # 加载到内存
        with self._query_lock:
            self._parent_index = parent_index
            self._parent_ids = parent_ids_ordered
            self._parent_id_to_row = {pid: i for i, pid in enumerate(parent_ids_ordered)}

    def _try_load_parent_index(self) -> bool:
        """启动时尝试加载已构建的父chunk索引。

        若磁盘上没有父索引文件，但子片段索引已就绪且有 sub_to_parent 映射，
        则从子片段索引中 reconstruct 向量，自动重建父 chunk 索引。
        """
        # 1. 磁盘文件存在 → 直接加载
        if os.path.isfile(self.parent_index_path) and \
           os.path.isfile(self.parent_ids_path):
            try:
                import faiss
                import numpy as np
                with open(self.parent_index_path, "rb") as f:
                    blob = f.read()
                arr = np.frombuffer(blob, dtype=np.uint8)
                parent_index = faiss.deserialize_index(arr)
                self._set_ef_search(parent_index)
                with open(self.parent_ids_path, "r", encoding="utf-8") as f:
                    self._parent_ids = json.load(f)
                self._parent_id_to_row = {pid: i for i, pid in enumerate(self._parent_ids)}
                self._parent_index = parent_index
                print(f"[faiss] 加载已建父chunk索引：{self.lib_root} "
                      f"(父chunk数={len(self._parent_ids)})", flush=True)
                return True
            except Exception as e:
                print(f"[faiss] 加载父chunk索引失败: {e}", flush=True)

        # 2. 磁盘文件不存在 → 尝试从子片段索引重建
        if self._index is not None and self._chunk_ids and self._sub_to_parent:
            try:
                self._rebuild_parent_index_from_sub()
                return True
            except Exception as e:
                print(f"[faiss] 自动重建父chunk索引失败: {e}", flush=True)

        self._parent_index = None
        self._parent_ids = []
        self._parent_id_to_row = {}
        return False

    def _rebuild_parent_index_from_sub(self) -> None:
        """从已加载的子片段索引中 reconstruct 向量，重建父 chunk 索引。

        HNSW 索引默认不支持 reconstruct，需先 make_direct_map。
        重建后持久化到磁盘，下次启动可直接加载。
        """
        import faiss
        import numpy as np

        t0 = time.perf_counter()
        # HNSW 需要开启 direct_map 才能 reconstruct
        try:
            self._index.make_direct_map()
        except Exception:
            pass  # 非 HNSW 索引可能不需要

        n = len(self._chunk_ids)
        sub_vecs = np.zeros((n, self.dim), dtype=np.float32)
        for i in range(n):
            sub_vecs[i] = self._index.reconstruct(i)

        # 推导父 chunk 列表（去重 + 排序）
        parent_ids_ordered = sorted(set(self._sub_to_parent.values()))

        # 复用已有的池化构建逻辑（含持久化）
        self._build_parent_index(sub_vecs, self._chunk_ids, parent_ids_ordered)
        pool_ms = (time.perf_counter() - t0) * 1000
        print(f"[faiss] 父chunk索引自动重建完成：{self.lib_root} "
              f"({len(parent_ids_ordered)} 父chunk, 耗时={pool_ms:.0f}ms)",
              flush=True)

    def is_parent_ready(self) -> bool:
        """父chunk索引是否就绪可查询。"""
        with self._status_lock:
            return self._status == STATUS_READY and self._parent_index is not None

    def search_parent(self, query_vec: "np.ndarray", top_k: int = 20) -> List[Dict[str, Any]]:
        """在父chunk级索引上做近邻查询（大chunk模式）。

        Args:
            query_vec: 查询向量 (dim,)
            top_k: 返回前 N 条

        Returns:
            [{"chunk_id": "zone_001/chunk_000001", "score": 0.85, "row": 12}, ...]
            直接返回父chunk_id，无需聚合
        """
        import numpy as np
        with self._query_lock:
            if self._parent_index is None or not self._parent_ids:
                return []
            q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
            scores, indices = self._parent_index.search(q, top_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._parent_ids):
                    continue
                results.append({
                    "chunk_id": self._parent_ids[idx],
                    "score": float(score),
                    "row": int(idx),
                })
            return results

    def search_sub_in_parent(self, query_vec: "np.ndarray",
                             parent_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """在指定父 chunk 范围内做子片段向量查询（暴力精确）。

        用于 progressive 检索第二步：父chunk粗筛后，精确定位子片段。
        通过 reconstruct 取出该父 chunk 所有子片段向量，做暴力余弦相似度。

        Args:
            query_vec: 查询向量 (dim,)
            parent_id: 父 chunk_id
            top_k: 返回前 N 条子片段

        Returns:
            [{"chunk_id": parent_id, "score": 0.85, "row": 12,
              "sub_id": "parent_id#2", "sub_score": 0.85}, ...]
        """
        import numpy as np
        with self._query_lock:
            if self._index is None:
                return []
            # 收集该父 chunk 的所有子片段行号
            sub_rows: List[int] = []
            for row, sid in enumerate(self._chunk_ids):
                if self._sub_to_parent.get(sid, sid) == parent_id:
                    sub_rows.append(row)
            if not sub_rows:
                return []
            # reconstruct 子片段向量
            try:
                self._index.make_direct_map()
            except Exception:
                pass
            q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
            results = []
            for row in sub_rows:
                vec = self._index.reconstruct(row)
                score = float(np.dot(q[0], vec))
                sid = self._chunk_ids[row]
                results.append({
                    "chunk_id": parent_id,
                    "score": score,
                    "row": int(row),
                    "sub_id": sid,
                    "sub_score": score,
                })
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

    # ----------------------------------------------------------
    #  断点续建：分批持久化 + 启动时检测未完成构建
    # ----------------------------------------------------------
    def _write_build_state(self, state: Dict[str, Any]) -> None:
        """写入构建状态文件（每批向量完成后调用）。"""
        os.makedirs(self.semantic_dir, exist_ok=True)
        tmp = self.build_state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
        os.replace(tmp, self.build_state_path)

    def _read_build_state(self) -> Optional[Dict[str, Any]]:
        """读取构建状态文件。不存在返回 None。"""
        if not os.path.isfile(self.build_state_path):
            return None
        try:
            with open(self.build_state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _has_partial_build(self) -> bool:
        """是否存在未完成的构建（断点续建前提）。"""
        state = self._read_build_state()
        if not state or state.get("done"):
            return False
        # 必须同时存在 vectors.part.f32 和 sub_ids.part.json
        return (os.path.isfile(self.vectors_part_path)
                and os.path.isfile(self.sub_ids_part_path))

    def _load_partial_vectors(self) -> Tuple[List[str], Dict[str, str], List[str], "np.ndarray"]:
        """加载已持久化的部分向量（断点续建用）。

        Returns:
            (sub_ids_done, sub_to_parent_done, parent_ids, vectors_array)
            vectors_array: shape=(n, dim), dtype=float32
        """
        import numpy as np
        with open(self.sub_ids_part_path, "r", encoding="utf-8") as f:
            sub_ids_done = json.load(f)
        with open(self.sub_to_parent_part_path, "r", encoding="utf-8") as f:
            sub_to_parent_done = json.load(f)
        with open(self.parent_ids_part_path, "r", encoding="utf-8") as f:
            parent_ids = json.load(f)
        # 读取已完成的向量二进制
        vecs = np.fromfile(self.vectors_part_path, dtype=np.float32)
        if vecs.size == 0:
            return sub_ids_done, sub_to_parent_done, parent_ids, np.zeros((0, self.dim), dtype=np.float32)
        # 推断维度：总元素数 / 已完成子片段数
        n = len(sub_ids_done)
        dim = vecs.size // n
        vecs = vecs.reshape(n, dim)
        # 同步维度
        if dim != self.dim:
            self.dim = dim
        return sub_ids_done, sub_to_parent_done, parent_ids, vecs

    def _append_batch_to_disk(self, batch_vecs: "np.ndarray",
                               batch_sub_ids: List[str],
                               batch_sub_to_parent: Dict[str, str]) -> None:
        """将一批向量追加到磁盘（断点续建的核心）。

        每批调用一次，确保即使中途崩溃也保留已完成的向量。
        """
        import numpy as np
        os.makedirs(self.semantic_dir, exist_ok=True)
        # 1. 追加向量二进制（float32，连续存储）
        with open(self.vectors_part_path, "ab") as f:
            batch_vecs.astype(np.float32).tofile(f)
            f.flush()
            os.fsync(f.fileno())
        # 2. 更新 sub_ids 列表（覆盖写，原子替换）
        existing_ids = []
        if os.path.isfile(self.sub_ids_part_path):
            try:
                with open(self.sub_ids_part_path, "r", encoding="utf-8") as f:
                    existing_ids = json.load(f)
            except Exception:
                existing_ids = []
        existing_ids.extend(batch_sub_ids)
        tmp = self.sub_ids_part_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(existing_ids, f, ensure_ascii=False)
        os.replace(tmp, self.sub_ids_part_path)
        # 3. 更新 sub_to_parent 映射
        existing_map = {}
        if os.path.isfile(self.sub_to_parent_part_path):
            try:
                with open(self.sub_to_parent_part_path, "r", encoding="utf-8") as f:
                    existing_map = json.load(f)
            except Exception:
                existing_map = {}
        existing_map.update(batch_sub_to_parent)
        tmp = self.sub_to_parent_part_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(existing_map, f, ensure_ascii=False)
        os.replace(tmp, self.sub_to_parent_part_path)

    def _cleanup_partial_files(self) -> None:
        """构建完成后清理 .part 文件和 build_state.json。"""
        for p in [self.vectors_part_path, self.sub_ids_part_path,
                  self.sub_to_parent_part_path, self.parent_ids_part_path,
                  self.build_state_path]:
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except Exception:
                pass

    def _save_index_to_disk(self, index, chunk_ids: List[str],
                            sub_to_parent: Dict[str, str],
                            model_name: str, build_time_ms: float) -> None:
        """原子写盘：先写临时文件再 rename，避免崩溃留下半成品。

        注意：Windows 上 faiss.write_index / read_index 通过 C++ IO 层
        打开文件，不支持含非 ASCII 字符（如中文库名）的路径。这里改为
        先用 faiss 序列化为字节，再用 Python 文件 IO 写盘，确保跨平台
        对中文路径的兼容性。

        Args:
            index: faiss 索引对象（None 表示空库）
            chunk_ids: 子片段 ID 列表（行号 → 子片段 ID）
            sub_to_parent: 子片段 ID → 父 chunk_id 映射
            model_name: 模型名
            build_time_ms: 构建耗时（毫秒）
        """
        os.makedirs(self.semantic_dir, exist_ok=True)
        # chunk_ids（子片段 ID 列表）
        tmp_ids = self.chunk_ids_path + ".tmp"
        with open(tmp_ids, "w", encoding="utf-8") as f:
            json.dump(chunk_ids, f, ensure_ascii=False)
        os.replace(tmp_ids, self.chunk_ids_path)
        # sub_to_parent 映射
        tmp_map = self.sub_to_parent_path + ".tmp"
        with open(tmp_map, "w", encoding="utf-8") as f:
            json.dump(sub_to_parent, f, ensure_ascii=False)
        os.replace(tmp_map, self.sub_to_parent_path)
        # faiss index：序列化为字节后用 Python 写盘
        if index is not None:
            import faiss
            blob = faiss.serialize_index(index)
            # faiss.serialize_index 返回 numpy 数组（uint8），
            # 用 tobytes() 转为原始字节后写入文件
            if hasattr(blob, "tobytes"):
                blob = blob.tobytes()
            tmp_idx = self.index_path + ".tmp"
            with open(tmp_idx, "wb") as f:
                f.write(blob)
            os.replace(tmp_idx, self.index_path)
        elif os.path.isfile(self.index_path):
            # 空库：移除旧索引
            try:
                os.remove(self.index_path)
            except Exception:
                pass
        # meta
        self._save_meta(len(chunk_ids), model_name, build_time_ms)

    # ----------------------------------------------------------
    #  查询
    # ----------------------------------------------------------
    def search(self, query_vec, top_k: int = 20) -> List[Dict[str, Any]]:
        """向量近邻查询（按父 chunk 聚合）。

        子片段索引查询后聚合回父 chunk：
          1. 向 HNSW 索引查 fetch_k = top_k × 3 个子片段候选
          2. 按 parent_id 聚合，同一父 chunk 取最高子片段得分
          3. 按聚合后分数降序，取前 top_k 个父 chunk

        这样既保证召回率（多取 3 倍候选），又避免同一父 chunk 重复出现。

        Args:
            query_vec: 已归一化的查询向量，shape=(dim,)
            top_k: 返回前 N 个父 chunk

        Returns:
            [{"chunk_id": "父chunk_id", "score": 0.85, "row": 12,
              "sub_id": "父chunk_id#2", "sub_score": 0.85}, ...]
            索引未就绪时返回空列表
        """
        if not self.is_ready():
            return []
        try:
            import numpy as np
            with self._query_lock:
                if self._index is None:
                    return []
                # faiss 要求 shape=(1, dim)
                q = np.asarray(query_vec, dtype="float32").reshape(1, -1)
                if q.shape[1] != self.dim:
                    return []
                # 多取 3 倍候选，保证聚合后仍有足够父 chunk
                fetch_k = min(top_k * 3, len(self._chunk_ids))
                if fetch_k <= 0:
                    return []
                scores, indices = self._index.search(q, fetch_k)

            # 按父 chunk 聚合：取最高子片段得分
            # 同一父 chunk 的多个子片段命中，只保留得分最高的那个
            parent_best: Dict[str, Dict[str, Any]] = {}
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._chunk_ids):
                    continue
                sub_id = self._chunk_ids[idx]
                parent_id = self._sub_to_parent.get(sub_id, sub_id)
                score_f = float(score)
                existing = parent_best.get(parent_id)
                if existing is None or score_f > existing["score"]:
                    parent_best[parent_id] = {
                        "chunk_id": parent_id,
                        "score": score_f,
                        "row": int(idx),
                        "sub_id": sub_id,         # 命中的子片段 ID
                        "sub_score": score_f,     # 子片段得分
                    }

            # 按聚合后分数降序，取前 top_k
            results = sorted(parent_best.values(),
                             key=lambda x: x["score"], reverse=True)
            return results[:top_k]
        except Exception as e:
            print(f"[faiss] 查询失败 {self.lib_root}: {e}", flush=True)
            return []

    # ----------------------------------------------------------
    #  清理
    # ----------------------------------------------------------
    def invalidate(self) -> None:
        """清理内存中的索引对象（删库 / 重建时调用）。"""
        with self._query_lock:
            with self._status_lock:
                self._index = None
                self._chunk_ids = []
                self._chunk_id_to_row = {}
                self._sub_to_parent = {}
                self._parent_chunks = []
                self._status = STATUS_IDLE
                self._fail_reason = ""
                self._progress = {}

    def remove_files(self) -> None:
        """删除磁盘上的索引文件（删库时调用）。"""
        self.invalidate()
        for p in (self.index_path, self.chunk_ids_path,
                  self.sub_to_parent_path, self.meta_path):
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except Exception:
                pass
        # 临时文件也清理
        for p in (self.index_path + ".tmp",
                  self.chunk_ids_path + ".tmp",
                  self.sub_to_parent_path + ".tmp",
                  self.meta_path + ".tmp"):
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except Exception:
                pass

    # ----------------------------------------------------------
    #  增量 / 重建判断
    # ----------------------------------------------------------
    def needs_rebuild(self, current_chunk_ids: List[str]) -> bool:
        """判断是否需要重建（父 chunk 集合变化时）。

        Args:
            current_chunk_ids: 当前库内全部父 chunk_id（按相同排序规则）

        Returns:
            True = 需要重建；False = 父 chunk 集合与索引一致
        """
        if not self.is_ready():
            return True
        with self._status_lock:
            # 比较父 chunk 列表（不是子片段列表）
            # 子片段数会因子切片策略变化，但父 chunk 集合稳定
            if self._parent_chunks != current_chunk_ids:
                return True
        return False


# ============================================================
#  二次切分向量化检索（精准模式用，无状态内存操作）
# ============================================================

def search_subchunks_of_text(text: str, query: str,
                              n_parts: int = 10, top_k: int = 3,
                              embedder=None) -> List[Dict[str, Any]]:
    """将长文本切成 n_parts 份，向量化后检索最相关的 top_k 份。

    精准模式的核心操作：把命中的父chunk（~10000字）切成约10份，
    每份独立向量化，再用查询向量做近邻检索，定位到最相关的段落。

    无状态、无持久化，纯内存操作。每次调用即时向量化。

    Args:
        text: 待切分的长文本（通常是父chunk全文）
        query: 查询文本
        n_parts: 切分份数（默认10）
        top_k: 返回前 N 份（默认3）
        embedder: Embedder 实例，None 时用共享实例

    Returns:
        [{"text": "子片段文本", "score": 0.85, "index": 0,
          "char_start": 0, "char_end": 500}, ...]
        按 score 降序
    """
    from embedding import Embedder
    if embedder is None:
        embedder = Embedder.get_shared()
    if not embedder.available() or not text.strip() or not query.strip():
        return []

    import numpy as np
    import faiss

    # 1. 切分文本：按 n_parts 均分，优先在句子边界切
    total_len = len(text)
    if total_len < n_parts * 50:
        # 文本太短，不值得切分，直接返回整段
        if total_len < 10:
            return []
        n_parts = max(1, total_len // 50)

    parts = []
    base_size = total_len // n_parts
    for i in range(n_parts):
        start = i * base_size
        end = (i + 1) * base_size if i < n_parts - 1 else total_len
        # 尝试在 [end-50, end+50] 范围找句子分隔符
        if end < total_len:
            search_start = max(start + 1, end - 50)
            best_cut = -1
            for delim in _SENTENCE_DELIMITERS:
                cut = text.rfind(delim, search_start, end + 50)
                if cut > best_cut:
                    best_cut = cut
                    if delim in ("\n", "。"):
                        break
            if best_cut > start:
                end = best_cut + 1
        chunk_text = text[start:end].strip()
        if chunk_text:
            parts.append({"text": chunk_text, "index": i,
                          "char_start": start, "char_end": end})

    if not parts:
        return []

    # 2. 向量化所有子片段
    texts = [p["text"] for p in parts]
    vecs = embedder.encode(texts, show_progress=False)
    if vecs is None:
        return []
    vecs = np.asarray(vecs, dtype=np.float32)
    # L2 归一化
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms

    # 3. 查询向量化
    qv = embedder.encode_query(query)
    if qv is None:
        return []
    qv = np.asarray(qv, dtype=np.float32).reshape(1, -1)
    qv = qv / (np.linalg.norm(qv) + 1e-8)

    # 4. 暴力近邻检索（子片段数少，无需 HNSW）
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    scores, indices = index.search(qv, min(top_k, len(parts)))

    # 5. 组装结果
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(parts):
            continue
        p = parts[idx]
        results.append({
            "text": p["text"],
            "score": float(score),
            "index": p["index"],
            "char_start": p["char_start"],
            "char_end": p["char_end"],
        })
    return results
