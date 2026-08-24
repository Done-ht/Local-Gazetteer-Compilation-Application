"""语义向量模型封装（bge-small-zh）。

职责：
    - 懒加载 sentence-transformers 模型，首次调用时才加载
    - 批量把文本转成 384 维归一化向量（L2 范数=1）
    - 提供"是否可用"探测，让上层在依赖缺失时优雅降级

设计要点：
    - 模型实例进程级单例，避免重复加载（约 200MB 内存）
    - 加载用锁保护，多线程并发触发只加载一次
    - 全程纯 Python，不依赖 Docker / 任何容器
    - bge-small-zh 对中文短文本友好，384 维兼顾召回率与内存/磁盘占用
      （10.5 万 chunk × 384 维 × float32 ≈ 154MB）
    - 默认优先从本地模型目录加载，避免联网下载；未命中时才回退到 HuggingFace

用法：
    from embedding import Embedder
    emb = Embedder()                  # 不会立即加载模型
    if not emb.available():
        # 依赖未装 / 模型加载失败，跳过语义通道
        return []
    vecs = emb.encode(["文本1", "文本2"])  # 首次调用时加载模型
    qv = emb.encode_query("查询词")
"""
from __future__ import annotations

import os
import threading
from typing import List, Optional
from userdata import auth_base_dir as _auth_base_dir


# 默认 HuggingFace 模型名（本地未命中时回退）
DEFAULT_MODEL_NAME = "BAAI/bge-small-zh"

# 默认本地模型目录（相对项目根目录）
DEFAULT_LOCAL_MODEL_PATH = "models/bge-small-zh"

# bge 系列推荐：查询句前加 "为这个句子生成表示以用于检索相关文章："
# 可提升检索效果（仅查询侧加，文档侧不加）
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def _project_root() -> str:
    """返回项目根目录（embedding.py 所在目录）。"""
    return os.path.dirname(os.path.abspath(__file__))


def _resolve_model_path(model_path: Optional[str], base_dir: Optional[str] = None) -> Optional[str]:
    """解析模型路径。

    - 若传绝对路径且存在，直接返回
    - 若传相对路径，依次尝试 base_dir、项目根目录拼接
    - 都不存在返回 None
    """
    if not model_path:
        return None
    if os.path.isabs(model_path) and os.path.isdir(model_path):
        return os.path.abspath(model_path)

    candidates = []
    if base_dir:
        candidates.append(os.path.join(base_dir, model_path))
    candidates.append(os.path.join(_project_root(), model_path))

    for candidate in candidates:
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    return None


class Embedder:
    """向量模型懒加载封装（线程安全单例）。

    一个进程只需要一个 Embedder 实例；模型加载完毕后所有库共用。
    """

    _instance: Optional["Embedder"] = None
    _instance_lock = threading.Lock()

    def __init__(self, model_name: Optional[str] = None,
                 device: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 local_files_only: bool = False,
                 base_dir: Optional[str] = None):
        """构造时仅记录参数，不加载模型。

        Args:
            model_name: 模型名 / HuggingFace repo / 本地模型目录路径。
                        为 None 时自动检测默认本地模型目录，存在则离线加载；
                        不存在则回退到 HuggingFace 默认模型名。
            device: "cpu" / "cuda"，默认 None 自动选择（CPU 优先，
                    避免在没装 torch+cuda 的服务器上报错）
            cache_dir: 模型缓存目录，默认 None 用 HuggingFace 默认位置
                       （~/.cache/huggingface/hub）
            local_files_only: True 时强制只使用本地文件，禁止访问 HuggingFace 下载
            base_dir: 项目数据根目录，用于解析相对模型路径
        """
        # 强制 CPU：服务器场景下基本无 GPU，且 faiss-cpu 也是 CPU
        # 显式指定避免 torch 误用 cuda 引发依赖问题
        self.device = device or "cpu"
        self.cache_dir = cache_dir
        self.local_files_only = bool(local_files_only)
        self.base_dir = base_dir

        # 自动检测默认本地模型目录
        auto_local = _resolve_model_path(DEFAULT_LOCAL_MODEL_PATH, self.base_dir)

        if model_name:
            local_path = _resolve_model_path(model_name, self.base_dir)
            if local_path:
                # 显式指定了本地路径且存在：用本地
                self.model_name = local_path
                self._local_path = local_path
                self.local_files_only = True
            elif model_name == DEFAULT_MODEL_NAME and auto_local:
                # 显式指定的是默认 HuggingFace ID，且本地默认目录存在：优先离线加载
                self.model_name = auto_local
                self._local_path = auto_local
                self.local_files_only = True
            else:
                # 显式指定的是 HuggingFace ID 或本地路径不存在
                self.model_name = model_name
                self._local_path = None
        elif auto_local:
            # 未指定且默认本地目录存在：优先离线加载本地模型
            self.model_name = auto_local
            self._local_path = auto_local
            self.local_files_only = True
        else:
            # 未指定且本地目录不存在：回退 HuggingFace 默认模型
            self.model_name = DEFAULT_MODEL_NAME
            self._local_path = None

        self._model = None            # 延迟加载
        self._model_lock = threading.Lock()
        self._load_failed = False     # 加载失败标记，避免反复重试
        self._fail_reason: str = ""

    # ----------------------------------------------------------
    #  单例访问（便于全局复用）
    # ----------------------------------------------------------
    @classmethod
    def get_shared(cls, base_dir: Optional[str] = None,
                   model_name: Optional[str] = None) -> "Embedder":
        """获取进程级共享实例。

        首次调用时创建；后续调用复用已建实例。
        会从 settings 读取 semantic_model_path / semantic_model_name 配置；
        未配置时自动检测默认本地模型目录。
        """
        with cls._instance_lock:
            if cls._instance is None:
                resolved_model = model_name
                local_files_only = False
                if resolved_model is None:
                    if base_dir:
                        try:
                            from settings import SettingsStore
                            store = SettingsStore(_auth_base_dir())
                            local_path = store.get("semantic_model_path", "")
                            if local_path:
                                resolved_model = local_path
                                local_files_only = True
                            else:
                                resolved_model = store.get("semantic_model_name")
                        except Exception:
                            pass
                    # 仍未解析到模型时，交给 __init__ 自动检测本地目录
                    if resolved_model is None:
                        resolved_model = None  # 使用自动选择
                cls._instance = cls(
                    model_name=resolved_model,
                    local_files_only=local_files_only,
                    base_dir=base_dir,
                )
            return cls._instance

    @classmethod
    def reset_shared(cls) -> None:
        """重置共享实例（用于测试或切换模型）。"""
        with cls._instance_lock:
            cls._instance = None

    # ----------------------------------------------------------
    #  可用性探测
    # ----------------------------------------------------------
    def available(self) -> bool:
        """检查依赖是否安装（不会触发模型加载）。

        Returns:
            True = 依赖已装，可以调用 encode
            False = 缺少 sentence-transformers / faiss / numpy，上层应跳过语义通道
        """
        if self._load_failed:
            return False
        missing = []
        for mod in ["sentence_transformers", "faiss", "numpy"]:
            try:
                __import__(mod)
            except ImportError as e:
                missing.append(f"{mod}: {e}")
        if missing:
            # 记录具体缺失的模块，便于打包/部署时定位
            print(f"[embedding] 语义依赖不可用：{'; '.join(missing)}", flush=True)
            return False
        return True

    def fail_reason(self) -> str:
        """返回不可用的原因（前端展示用）。"""
        if self._load_failed and self._fail_reason:
            return self._fail_reason
        # 检查具体缺失的模块，给出精确错误信息
        missing = []
        for mod in ["sentence_transformers", "faiss", "numpy"]:
            try:
                __import__(mod)
            except ImportError as e:
                missing.append(f"{mod}（{e}）")
        if missing:
            return f"语义检索依赖不可用：{'; '.join(missing)}"
        return ""

    # ----------------------------------------------------------
    #  模型加载
    # ----------------------------------------------------------
    def _ensure_model(self):
        """首次调用时加载模型；失败则置位 _load_failed 不再重试。"""
        if self._model is not None:
            return self._model
        if self._load_failed:
            return None
        with self._model_lock:
            if self._model is not None:
                return self._model
            if self._load_failed:
                return None
            try:
                from sentence_transformers import SentenceTransformer
                kwargs = {"device": self.device}
                if self.cache_dir:
                    kwargs["cache_folder"] = self.cache_dir

                # 如果解析到了本地目录，直接加载本地模型；否则按 HuggingFace ID 加载
                if self._local_path:
                    model_id = self._local_path
                    kwargs["local_files_only"] = True
                    source_desc = f"本地路径 {model_id}"
                else:
                    model_id = self.model_name
                    kwargs["local_files_only"] = self.local_files_only
                    source_desc = model_id

                self._model = SentenceTransformer(model_id, **kwargs)
                # 触发一次空推理，确认模型真正可用
                _ = self._model.encode(["warmup"], show_progress_bar=False,
                                       convert_to_numpy=True)
                print(f"[embedding] 模型加载成功：{source_desc} "
                      f"(device={self.device}, dim={self.dim})",
                      flush=True)
            except Exception as e:
                self._load_failed = True
                self._fail_reason = f"模型加载失败：{e}"
                print(f"[embedding] 模型加载失败：{e}", flush=True)
                self._model = None
        return self._model

    # ----------------------------------------------------------
    #  向量化
    # ----------------------------------------------------------
    @property
    def dim(self) -> int:
        """向量维度（从模型动态获取；未加载时返回默认 384）。"""
        # 模型已加载时，从模型实例获取真实维度
        # （bge-small-zh 早期版本 384 维，v1.5 之后 512 维）
        if self._model is not None:
            try:
                # 新版 sentence-transformers 已重命名为 get_embedding_dimension
                if hasattr(self._model, "get_embedding_dimension"):
                    return int(self._model.get_embedding_dimension())
                return int(self._model.get_sentence_embedding_dimension())
            except Exception:
                pass
        return 384  # 默认值（未加载模型时用，避免强制加载）

    def encode(self, texts: List[str], batch_size: int = 64,
               show_progress: bool = False) -> "object":
        """批量把文档文本转为归一化向量。

        Args:
            texts: 文本列表
            batch_size: 单批大小（默认 64，平衡吞吐与内存）
            show_progress: 是否打印进度条（长任务时建议 True）

        Returns:
            numpy.ndarray，shape=(N, 384)，dtype=float32，已 L2 归一化
            失败时返回 None
        """
        model = self._ensure_model()
        if model is None or not texts:
            return None
        try:
            import numpy as np
            vecs = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2 归一化，配合 HNSW 内积度量
            )
            # 强制 float32（faiss 要求）
            if vecs.dtype != "float32":
                vecs = vecs.astype("float32")
            return vecs
        except Exception as e:
            print(f"[embedding] encode 失败：{e}", flush=True)
            return None

    def encode_query(self, query: str) -> "object":
        """把查询语句转为向量（bge 推荐加 query instruction）。

        Returns:
            numpy.ndarray，shape=(384,)，已归一化
            失败时返回 None
        """
        model = self._ensure_model()
        if model is None or not query:
            return None
        try:
            import numpy as np
            # bge 系列：查询侧加 instruction 提升召回
            q = f"{BGE_QUERY_INSTRUCTION}{query}"
            vec = model.encode(
                [q],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            vec = vec[0]
            if vec.dtype != "float32":
                vec = vec.astype("float32")
            return vec
        except Exception as e:
            print(f"[embedding] encode_query 失败：{e}", flush=True)
            return None


# ----------------------------------------------------------
#  模块级便捷函数
# ----------------------------------------------------------
def is_semantic_available() -> bool:
    """快速探测语义通道是否可用（不加载模型）。"""
    return Embedder().available()


def get_fail_reason() -> str:
    """获取语义通道不可用的原因（前端展示用）。"""
    return Embedder().fail_reason()
