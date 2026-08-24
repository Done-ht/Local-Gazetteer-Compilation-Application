"""语义检索通道总管。

职责：
    - 进程级管理所有库的 FaissIndex 实例（懒创建、缓存）
    - 后台线程触发索引构建（导入完成后调用）
    - 提供跨库语义检索 API（被 searcher.py 调用）
    - 提供状态查询 API（被 web_api.py 调用，前端展示用）
    - 删库 / 重建时清理资源

设计要点：
    - 所有公开方法线程安全
    - 依赖缺失（sentence-transformers / faiss-cpu 未装）时优雅降级，
      所有检索返回空列表，状态查询返回 unavailable
    - 后台构建用 daemon 线程，进程退出时自动结束
    - 构建去重：同一库同时只有一个构建任务
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from embedding import Embedder
from faiss_index import FaissIndex, STATUS_BUILDING, STATUS_READY, STATUS_FAILED


class SemanticManager:
    """进程级语义检索管理器（单例）。"""

    _instance: Optional["SemanticManager"] = None
    _instance_lock = threading.Lock()

    def __init__(self, base_dir: str):
        """初始化。

        Args:
            base_dir: 数据根目录（库注册表所在目录）
        """
        self.base_dir = os.path.abspath(base_dir)
        # lib_name -> FaissIndex 实例
        self._indices: Dict[str, FaissIndex] = {}
        self._lock = threading.RLock()
        # 后台构建任务跟踪：lib_name -> Thread
        self._build_threads: Dict[str, threading.Thread] = {}
        # 全局开关（可被 settings 关闭）
        self._enabled = True
        # 共享 Embedder 实例（进程级单例），首次创建时从 settings 读取本地模型路径
        self._embedder = Embedder.get_shared(base_dir)

    # ----------------------------------------------------------
    #  单例
    # ----------------------------------------------------------
    @classmethod
    def get_shared(cls, base_dir: Optional[str] = None) -> "SemanticManager":
        with cls._instance_lock:
            if cls._instance is None:
                if base_dir is None:
                    raise ValueError("首次创建 SemanticManager 必须传 base_dir")
                cls._instance = cls(base_dir)
            return cls._instance

    @classmethod
    def reset_shared(cls) -> None:
        """重置单例（测试用）。"""
        with cls._instance_lock:
            cls._instance = None

    # ----------------------------------------------------------
    #  可用性
    # ----------------------------------------------------------
    def available(self) -> bool:
        """语义通道是否可用（依赖已装 + 全局开关开启）。"""
        if not self._enabled:
            return False
        return self._embedder.available()

    def fail_reason(self) -> str:
        """不可用时返回原因（前端展示用）。"""
        if not self._enabled:
            return "语义检索通道已在设置中关闭"
        return self._embedder.fail_reason()

    def set_enabled(self, enabled: bool) -> None:
        """开关语义通道。"""
        self._enabled = bool(enabled)

    # ----------------------------------------------------------
    #  库级索引访问
    # ----------------------------------------------------------
    def _get_or_create_index(self, lib_root: str) -> FaissIndex:
        """获取或创建库的 FaissIndex（懒创建，已加载则复用）。"""
        lib_root = os.path.abspath(lib_root)
        with self._lock:
            idx = self._indices.get(lib_root)
            if idx is None:
                idx = FaissIndex(lib_root)
                self._indices[lib_root] = idx
            return idx

    def get_index(self, lib_root: str) -> Optional[FaissIndex]:
        """获取已缓存的 FaissIndex（不存在返回 None）。"""
        lib_root = os.path.abspath(lib_root)
        with self._lock:
            return self._indices.get(lib_root)

    # ----------------------------------------------------------
    #  后台构建
    # ----------------------------------------------------------
    def trigger_build_async(self, lib_root: str, lib_name: str = "") -> dict:
        """异步触发某库的索引构建。

        - 若依赖未装 → 返回 started=False
        - 若该库已在构建中 → 跳过，返回 started=False
        - 若索引已就绪且 chunk 集合未变 → 跳过，返回 started=False
        - 否则启动 daemon 线程构建，立即返回 started=True

        Args:
            lib_root: 库根目录（绝对路径）
            lib_name: 库名（仅用于日志）

        Returns:
            {"started": bool, "reason": str}
        """
        if not self.available():
            return {"started": False, "reason": self.fail_reason() or "语义通道不可用"}

        idx = self._get_or_create_index(lib_root)
        if idx.is_building():
            return {"started": False, "reason": "该库向量索引正在构建中"}

        # 收集当前 chunk 列表
        chunk_texts = self._collect_chunk_texts(lib_root)
        if chunk_texts is None:
            return {"started": False, "reason": "收集 chunk 文本失败"}
        if not chunk_texts:
            return {"started": False, "reason": "该库尚无 chunk，请先导入文档"}

        # 检查是否需要重建
        current_ids = [c[0] for c in chunk_texts]
        if not idx.needs_rebuild(current_ids):
            return {"started": False, "reason": "向量索引已是最新，无需重建"}

        # 启动后台线程
        thread_name = f"semantic-build-{lib_name or lib_root}"
        # 防止线程名重复（Windows 上线程名长度有限）
        thread_name = thread_name[:60]
        th = threading.Thread(
            target=self._run_build,
            args=(lib_root, chunk_texts),
            name=thread_name,
            daemon=True,
        )
        with self._lock:
            self._build_threads[lib_root] = th
        th.start()
        print(f"[semantic] 已启动后台索引构建：{lib_name or lib_root} "
              f"(chunk 数={len(chunk_texts)})", flush=True)
        return {"started": True, "reason": ""}

    def _run_build(self, lib_root: str, chunk_texts: List[Tuple[str, str]]) -> None:
        """后台线程执行体。"""
        try:
            idx = self._get_or_create_index(lib_root)
            idx.build(chunk_texts, embedder=self._embedder)
        except Exception as e:
            print(f"[semantic] 后台构建异常 {lib_root}: {e}", flush=True)
        finally:
            with self._lock:
                self._build_threads.pop(lib_root, None)

    def resume_pending_builds(self, registry) -> int:
        """服务器启动时扫描所有库，续建未完成的向量索引构建。

        服务器异常重启后，若某库的向量索引构建中断（存在 .part 文件
        且 build_state.json 标记 done=false），自动从断点继续构建，
        避免用户辛苦等待的进度白白丢失。

        Args:
            registry: LibraryRegistry 实例，用于遍历所有库

        Returns:
            已启动续建的库数量
        """
        if not self.available():
            return 0
        resumed_count = 0
        for lib in registry.list_libraries():
            lib_root = lib.abs_path(self.base_dir)
            try:
                idx = self._get_or_create_index(lib_root)
                # 仅当索引未就绪且存在未完成断点时才续建
                if idx.is_ready():
                    continue
                if not idx._has_partial_build():
                    continue
                # 收集 chunk 文本并启动续建
                chunk_texts = self._collect_chunk_texts(lib_root)
                if not chunk_texts:
                    continue
                thread_name = f"semantic-resume-{lib.name}"[:60]
                th = threading.Thread(
                    target=self._run_build,
                    args=(lib_root, chunk_texts),
                    name=thread_name,
                    daemon=True,
                )
                with self._lock:
                    self._build_threads[lib_root] = th
                th.start()
                resumed_count += 1
                print(f"[semantic] 检测到未完成构建，已启动断点续建："
                      f"{lib.name} (chunk 数={len(chunk_texts)})",
                      flush=True)
            except Exception as e:
                print(f"[semantic] 续建启动失败 {lib.name}: {e}", flush=True)
        if resumed_count:
            print(f"[semantic] 共 {resumed_count} 个库启动了断点续建",
                  flush=True)
        return resumed_count

    def _collect_chunk_texts(self, lib_root: str) -> Optional[List[Tuple[str, str]]]:
        """收集库内所有 chunk 的 (chunk_id, text)。

        按 chunk_id 排序，与 searcher._list_library_chunk_ids 一致。
        """
        lib_root = os.path.abspath(lib_root)
        result: List[Tuple[str, str]] = []
        # 库根目录下的 zone_xxx/chunks/chunk_xxxxxx.json
        try:
            for entry in sorted(os.listdir(lib_root)):
                zone_dir = os.path.join(lib_root, entry)
                if not os.path.isdir(zone_dir):
                    continue
                if not entry.startswith("zone_"):
                    continue
                chunks_dir = os.path.join(zone_dir, "chunks")
                if not os.path.isdir(chunks_dir):
                    continue
                for fname in sorted(os.listdir(chunks_dir)):
                    if not fname.endswith(".json"):
                        continue
                    chunk_name = fname[:-5]  # 去 .json
                    chunk_id = f"{entry}/{chunk_name}"
                    try:
                        with open(os.path.join(chunks_dir, fname),
                                  "r", encoding="utf-8") as f:
                            chunk = json.load(f)
                        text = chunk.get("text", "") or ""
                        if text:
                            result.append((chunk_id, text))
                    except Exception:
                        continue
            # 跨 zone 排序（与 searcher._list_library_chunk_ids 一致）
            result.sort(key=lambda x: x[0])
            return result
        except Exception as e:
            print(f"[semantic] 收集 chunk 失败 {lib_root}: {e}", flush=True)
            return None

    # ----------------------------------------------------------
    #  查询
    # ----------------------------------------------------------
    def search(self, lib_root: str, query: str, top_k: int = 20,
               chunk_filter: Optional[set] = None) -> List[Dict[str, Any]]:
        """对单库执行向量近邻查询。

        Args:
            lib_root: 库根目录
            query: 查询文本
            top_k: 返回前 N 条
            chunk_filter: 若提供，只保留该集合中的 chunk_id

        Returns:
            [{"chunk_id": "...", "score": 0.85, "row": 12}, ...]
            不可用 / 未就绪 → 返回空列表
        """
        if not self.available() or not query.strip():
            return []
        # 用 _get_or_create_index 而非 get_index，确保服务器重启后
        # 首次查询时自动创建 FaissIndex 实例并触发从磁盘加载已建索引
        idx = self._get_or_create_index(lib_root)
        # 校验磁盘文件：防止内存状态与磁盘不一致导致查询崩溃
        if idx.is_ready():
            import os as _os
            if not _os.path.isfile(idx.index_path) or not _os.path.isfile(idx.chunk_ids_path):
                print(f"[semantic] 检测到磁盘索引文件丢失，清理内存缓存：{lib_root}",
                      flush=True)
                self.invalidate(lib_root)
                idx = self._get_or_create_index(lib_root)
        if not idx.is_ready():
            return []
        # 查询向量化
        qv = self._embedder.encode_query(query)
        if qv is None:
            return []
        # 多取一倍候选，便于 chunk_filter 过滤后仍有足够结果
        fetch_k = top_k * 2 if chunk_filter is not None else top_k
        hits = idx.search(qv, top_k=fetch_k)
        if chunk_filter is not None:
            hits = [h for h in hits if h["chunk_id"] in chunk_filter]
        return hits[:top_k]

    def search_parent(self, lib_root: str, query: str, top_k: int = 20,
                      chunk_filter: Optional[set] = None) -> List[Dict[str, Any]]:
        """对单库执行父chunk级向量查询（大chunk模式）。

        与 search() 的区别：直接查父chunk索引（池化向量），返回父chunk_id
        无需子片段聚合，查询更快，但无法定位到具体子片段位置。

        Args:
            lib_root: 库根目录
            query: 查询文本
            top_k: 返回前 N 条
            chunk_filter: 若提供，只保留该集合中的 chunk_id（父chunk_id）

        Returns:
            [{"chunk_id": "zone_001/chunk_000001", "score": 0.85, "row": 12}, ...]
            不可用 / 未就绪 / 无父chunk索引 → 返回空列表
        """
        if not self.available() or not query.strip():
            return []
        idx = self._get_or_create_index(lib_root)
        if idx.is_ready():
            import os as _os
            if not _os.path.isfile(idx.index_path) or not _os.path.isfile(idx.chunk_ids_path):
                self.invalidate(lib_root)
                idx = self._get_or_create_index(lib_root)
        if not idx.is_ready() or not idx.is_parent_ready():
            return []
        qv = self._embedder.encode_query(query)
        if qv is None:
            return []
        fetch_k = top_k * 2 if chunk_filter is not None else top_k
        hits = idx.search_parent(qv, top_k=fetch_k)
        if chunk_filter is not None:
            hits = [h for h in hits if h["chunk_id"] in chunk_filter]
        return hits[:top_k]

    def search_sub_in_parent(self, lib_root: str, query: str,
                             parent_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """在指定父 chunk 范围内做子片段向量查询（暴力精确）。

        用于 progressive 检索第二步：父chunk粗筛后，精确定位子片段。

        Returns:
            [{"chunk_id": parent_id, "score": 0.85, "row": 12,
              "sub_id": "parent_id#2", "sub_score": 0.85}, ...]
        """
        if not self.available() or not query.strip():
            return []
        idx = self._get_or_create_index(lib_root)
        if not idx.is_ready():
            return []
        qv = self._embedder.encode_query(query)
        if qv is None:
            return []
        return idx.search_sub_in_parent(qv, parent_id, top_k=top_k)

    def search_multi_libs(self, lib_roots: List[str], query: str,
                          top_k: int = 20,
                          chunk_filter: Optional[set] = None
                          ) -> Dict[str, List[Dict[str, Any]]]:
        """跨库向量查询。

        Returns:
            {lib_root: [hits...], ...}  仅包含有结果的库
        """
        if not self.available():
            return {}
        out: Dict[str, List[Dict[str, Any]]] = {}
        for lr in lib_roots:
            hits = self.search(lr, query, top_k=top_k, chunk_filter=chunk_filter)
            if hits:
                out[lr] = hits
        return out

    # ----------------------------------------------------------
    #  状态查询
    # ----------------------------------------------------------
    def status(self, lib_root: str) -> Dict[str, Any]:
        """查询单库索引状态（前端展示用）。

        会校验磁盘文件：如果内存状态是 ready 但磁盘索引文件已被外部删除，
        自动清理内存缓存并返回 idle，避免"幽灵就绪"状态。
        """
        if not self.available():
            return {
                "status": "unavailable",
                "fail_reason": self.fail_reason(),
                "vector_count": 0,
                "enabled": self._enabled,
            }
        # 用 _get_or_create_index 而非 get_index，确保服务器重启后
        # 首次状态查询时自动创建 FaissIndex 实例并触发从磁盘加载已建索引
        idx = self._get_or_create_index(lib_root)
        # 校验磁盘文件：防止内存状态与磁盘不一致
        # 场景：索引文件被外部进程删除（如测试脚本、手动清理），
        # 但服务器进程的 FaissIndex 实例仍缓存着旧的 ready 状态
        if idx.is_ready():
            import os as _os
            if not _os.path.isfile(idx.index_path) or not _os.path.isfile(idx.chunk_ids_path):
                # 磁盘文件已丢失，清理内存缓存
                print(f"[semantic] 检测到磁盘索引文件丢失，清理内存缓存：{lib_root}",
                      flush=True)
                self.invalidate(lib_root)
                idx = self._get_or_create_index(lib_root)
        s = idx.status()
        s["enabled"] = self._enabled
        return s

    def status_all(self, lib_roots: List[str]) -> List[Dict[str, Any]]:
        """查询所有库的索引状态。"""
        return [{"lib_root": lr, **self.status(lr)} for lr in lib_roots]

    # ----------------------------------------------------------
    #  清理
    # ----------------------------------------------------------
    def invalidate(self, lib_root: str) -> None:
        """清理内存中的索引对象（删库 / 重建时调用）。"""
        lib_root = os.path.abspath(lib_root)
        with self._lock:
            idx = self._indices.pop(lib_root, None)
        if idx is not None:
            idx.invalidate()

    def remove_files(self, lib_root: str) -> None:
        """删除库的索引文件 + 清理内存（删库时调用）。"""
        lib_root = os.path.abspath(lib_root)
        with self._lock:
            idx = self._indices.pop(lib_root, None)
        if idx is not None:
            idx.remove_files()

    def invalidate_all(self) -> None:
        """清理所有库的索引对象（服务器关闭时调用，可选）。"""
        with self._lock:
            for idx in self._indices.values():
                try:
                    idx.invalidate()
                except Exception:
                    pass
            self._indices.clear()


# ----------------------------------------------------------
#  便捷函数
# ----------------------------------------------------------
def get_manager(base_dir: Optional[str] = None) -> SemanticManager:
    """获取共享 SemanticManager 实例。"""
    return SemanticManager.get_shared(base_dir)
