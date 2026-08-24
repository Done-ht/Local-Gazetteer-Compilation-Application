"""系统设置持久化（独立模块，仅依赖标准库 + cryptography）。

普通设置保存到用户登录数据目录 biaoshifu（Windows 默认 <用户主目录>/biaoshifu，
跨应用共用；见 userdata.py）下的 `_settings.json`；
敏感字段（如 DeepSeek API Key）加密后单独保存到 `_secrets.json`，
与 `_settings.json` 物理隔离。

Windows 下使用 DPAPI 加密（绑定当前用户登录会话，无需额外密钥文件）；
非 Windows 平台使用 Fernet（密钥保存在 `_secret.key`，文件权限 0o600）。

当前管理 DeepSeek 相关配置：
    - deepseek_api_key   : API Key（敏感，单独加密存储）
    - deepseek_model     : 模型 id（deepseek-v4-flash / deepseek-v4-pro）
    - deepseek_base_url  : 自定义 API 地址（默认 https://api.deepseek.com）

用法：
    from settings import SettingsStore
    store = SettingsStore("/path/to/base_dir")

    store.set("deepseek_api_key", "sk-xxx")
    store.set("deepseek_model", "deepseek-v4-pro")

    key = store.get("deepseek_api_key")          # 原值
    safe = store.get_safe("deepseek_api_key")    # 脱敏：sk-***...***abcd
"""
from __future__ import annotations

import base64
import json
import os
import sys
import threading
from typing import Any, Dict, Optional


# 默认值
DEFAULTS: Dict[str, Any] = {
    "deepseek_api_key": "",
    "deepseek_model": "deepseek-v4-flash",
    "deepseek_base_url": "https://api.deepseek.com",

    # ===== 生成阶段 =====
    "gen_history_rounds": 6,        # 多轮对话历史轮数（1轮=2条消息）

    # ===== Agent 工作流（LLM 自主调用工具的循环次数上限）=====
    "agent_workflow_max_rounds": 15,  # 最大轮数，默认15，范围3-30

    # ===== 大段标签提取（jieba 词性过滤，导入时自动生成）=====
    # 标签存入 chunk JSON 的 tags 字段，用于辅助检索筛选
    "tag_top_k": 10,  # 每个 chunk 提取的标签数，默认10，范围0-30；设为0则禁用

    # ===== 分块检索梯度（按 chunk 数决定块数）=====
    "partition_threshold": 1800,  # 触发阈值：≤此值不分块
    "partition_max_parts": 6,     # 最大块数
    # 自定义梯度表（JSON 列表 [[threshold, parts], ...]， None=用内置默认）
    # 默认梯度：≤1800→1块, ≤3000→2块, ≤6000→3块, ≤12000→4块, ≤24000→5块, >24000→6块
    "partition_gradient": None,

    # ===== 语义检索通道（向量召回）=====
    # 基于 bge-small-zh 模型 + Faiss HNSW 索引，作为关键词检索的补充通道
    # 解决同义表述、语义分散、概念隐含等召回盲区
    # 依赖 sentence-transformers + faiss-cpu（未装时自动跳过，不影响主流程）
    "semantic_enabled": True,          # 语义通道总开关
    "semantic_top_k": 30,              # 单库语义召回条数（融合前）
    "semantic_min_score": 0.30,        # 最低相似度阈值（低于此分丢弃）
    "semantic_fusion_weight": 0.5,     # 融合权重：语义分 × 此权重 + 关键词分 × (1-权重)
                                       # 0=完全用关键词排序，1=完全用语义排序
    "semantic_auto_build": True,       # 导入完成后自动后台构建索引
    "semantic_sub_chunk_size": 500,    # 语义子分块大小（字符数）
                                       # 父 chunk 向量化时切成 ≤此长度 的子片段
                                       # bge-small-zh 最大输入 512 token（约 500 中文字）
                                       # 父 chunk 超过此长度会被切片后独立向量化
                                       # 查询时按父 chunk 聚合（取最高子片段得分）
    "semantic_model_path": "models/bge-small-zh",  # 本地向量模型目录（相对项目根目录）
                                                    # 存在时优先从该路径离线加载，避免访问 HuggingFace
}

# 视为敏感字段（读取时默认脱敏，单独加密存储）
SENSITIVE_KEYS = {"deepseek_api_key"}

SETTINGS_FILENAME = "_settings.json"
SECRETS_FILENAME = "_secrets.json"
KEY_FILENAME = "_secret.key"


# ============================================================
#  本地加密
# ============================================================

def _dpapi_encrypt(plain: bytes) -> bytes:
    """使用 Windows DPAPI 加密数据（绑定当前用户登录会话）。仅在 Windows 上可用。"""
    import ctypes
    from ctypes import wintypes

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [
            ("cbData", wintypes.DWORD),
            ("pbData", wintypes.LPBYTE),
        ]

    CryptProtectData = ctypes.windll.crypt32.CryptProtectData
    CryptProtectData.argtypes = [
        ctypes.POINTER(DATA_BLOB),
        wintypes.LPCWSTR,
        ctypes.POINTER(DATA_BLOB),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(DATA_BLOB),
    ]
    CryptProtectData.restype = wintypes.BOOL

    blob_in = DATA_BLOB(len(plain), ctypes.cast(plain, wintypes.LPBYTE))
    blob_out = DATA_BLOB()
    if not CryptProtectData(ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)):
        raise OSError("DPAPI 加密失败")
    encrypted = bytes(blob_out.pbData[:blob_out.cbData])
    ctypes.windll.kernel32.LocalFree(blob_out.pbData)
    return encrypted


def _dpapi_decrypt(encrypted: bytes) -> bytes:
    """使用 Windows DPAPI 解密数据。仅在 Windows 上可用。"""
    import ctypes
    from ctypes import wintypes

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [
            ("cbData", wintypes.DWORD),
            ("pbData", wintypes.LPBYTE),
        ]

    CryptUnprotectData = ctypes.windll.crypt32.CryptUnprotectData
    CryptUnprotectData.argtypes = [
        ctypes.POINTER(DATA_BLOB),
        wintypes.LPWSTR,
        ctypes.POINTER(DATA_BLOB),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(DATA_BLOB),
    ]
    CryptUnprotectData.restype = wintypes.BOOL

    blob_in = DATA_BLOB(len(encrypted), ctypes.cast(encrypted, wintypes.LPBYTE))
    blob_out = DATA_BLOB()
    if not CryptUnprotectData(ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)):
        raise OSError("DPAPI 解密失败，可能是当前用户会话变更或数据损坏")
    plain = bytes(blob_out.pbData[:blob_out.cbData])
    ctypes.windll.kernel32.LocalFree(blob_out.pbData)
    return plain


def _get_fernet(base_dir: str):
    """非 Windows 平台使用 Fernet；密钥保存在 _secret.key。"""
    from cryptography.fernet import Fernet
    key_path = os.path.join(base_dir, KEY_FILENAME)
    if os.path.isfile(key_path):
        with open(key_path, "rb") as f:
            key = f.read()
    else:
        key = Fernet.generate_key()
        os.makedirs(base_dir, exist_ok=True)
        tmp = key_path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(key)
        os.replace(tmp, key_path)
        if sys.platform != "win32":
            try:
                os.chmod(key_path, 0o600)
            except OSError:
                pass
    return Fernet(key)


def _encrypt_secret(value: str, base_dir: str) -> str:
    """加密敏感字符串，返回可写入 JSON 的文本。"""
    if sys.platform == "win32":
        encrypted = _dpapi_encrypt(value.encode("utf-8"))
        return base64.b64encode(encrypted).decode("ascii")
    else:
        f = _get_fernet(base_dir)
        token = f.encrypt(value.encode("utf-8"))
        return token.decode("ascii")


def _decrypt_secret(encoded: str, base_dir: str) -> str:
    """解密敏感字符串。"""
    if sys.platform == "win32":
        encrypted = base64.b64decode(encoded)
        return _dpapi_decrypt(encrypted).decode("utf-8")
    else:
        f = _get_fernet(base_dir)
        token = encoded.encode("ascii")
        return f.decrypt(token).decode("utf-8")


# ============================================================
#  设置存储
# ============================================================

class SettingsStore:
    """线程安全的设置存储。

    普通设置保存到 `_settings.json`；敏感字段加密后保存到 `_secrets.json`。
    读取时自动把旧版 `_settings.json` 中的明文敏感字段迁移到 `_secrets.json`。
    """

    def __init__(self, base_dir: str):
        self._base_dir = base_dir
        self._path = os.path.join(base_dir, SETTINGS_FILENAME)
        self._secrets_path = os.path.join(base_dir, SECRETS_FILENAME)
        self._lock = threading.Lock()

    @property
    def path(self) -> str:
        return self._path

    @property
    def secrets_path(self) -> str:
        return self._secrets_path

    # ----------------------------------------------------------
    #  读取
    # ----------------------------------------------------------
    def load_all(self) -> Dict[str, Any]:
        """读取全部普通设置（不含敏感字段解密），合并默认值。"""
        data = dict(DEFAULTS)
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                stored = json.load(f)
            if isinstance(stored, dict):
                for k, v in stored.items():
                    data[k] = v
        except (OSError, json.JSONDecodeError):
            pass
        return data

    def _load_secrets(self) -> Dict[str, str]:
        """读取加密存储的原始密文。"""
        if not os.path.isfile(self._secrets_path):
            return {}
        try:
            with open(self._secrets_path, "r", encoding="utf-8") as f:
                stored = json.load(f)
            if isinstance(stored, dict):
                return stored
        except (OSError, json.JSONDecodeError):
            pass
        return {}

    def _write_secrets(self, secrets: Dict[str, str]) -> None:
        """写入加密存储文件，并设置严格的文件权限。"""
        os.makedirs(self._base_dir, exist_ok=True)
        tmp = self._secrets_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(secrets, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self._secrets_path)
        if sys.platform != "win32":
            try:
                os.chmod(self._secrets_path, 0o600)
            except OSError:
                pass

    def _migrate_legacy_secret(self, key: str) -> None:
        """把 _settings.json 中的明文敏感字段迁移到 _secrets.json 并删除明文。"""
        settings_data = self.load_all()
        plain = settings_data.get(key)
        if not isinstance(plain, str) or not plain:
            return
        try:
            encrypted = _encrypt_secret(plain, self._base_dir)
        except Exception:
            return
        secrets = self._load_secrets()
        secrets[key] = encrypted
        self._write_secrets(secrets)
        if key in settings_data:
            del settings_data[key]
            self._write(settings_data)

    def _get_secret(self, key: str, default: Any = None) -> Any:
        """读取单个敏感字段（解密后的原值）。"""
        self._migrate_legacy_secret(key)
        secrets = self._load_secrets()
        encrypted = secrets.get(key)
        if not isinstance(encrypted, str) or not encrypted:
            return default
        try:
            return _decrypt_secret(encrypted, self._base_dir)
        except Exception:
            return default

    def get(self, key: str, default: Any = None) -> Any:
        """读取单个字段（原值）。"""
        if key in SENSITIVE_KEYS:
            return self._get_secret(key, default)
        return self.load_all().get(key, default)

    def get_safe(self, key: str, default: Any = None) -> Any:
        """读取单个字段，敏感字段返回脱敏值。"""
        value = self.get(key, default)
        if key in SENSITIVE_KEYS and isinstance(value, str) and value:
            return mask_secret(value)
        return value

    def safe_snapshot(self) -> Dict[str, Any]:
        """返回全部设置的脱敏快照（用于前端展示）。"""
        all_data = dict(DEFAULTS)
        settings_data = self.load_all()
        for k, v in settings_data.items():
            all_data[k] = v
        # 合并并解密敏感字段
        for key in SENSITIVE_KEYS:
            all_data[key] = self._get_secret(key, "")
        out = {}
        for k, v in all_data.items():
            if k in SENSITIVE_KEYS and isinstance(v, str) and v:
                out[k] = mask_secret(v)
            else:
                out[k] = v
        return out

    # ----------------------------------------------------------
    #  写入
    # ----------------------------------------------------------
    def set(self, key: str, value: Any) -> None:
        """更新单个字段并持久化。"""
        with self._lock:
            if key in SENSITIVE_KEYS:
                secrets = self._load_secrets()
                if isinstance(value, str) and value:
                    secrets[key] = _encrypt_secret(value, self._base_dir)
                else:
                    secrets.pop(key, None)
                self._write_secrets(secrets)
                # 同时清理 _settings.json 中可能存在的明文旧值
                settings_data = self.load_all()
                if key in settings_data:
                    del settings_data[key]
                    self._write(settings_data)
            else:
                all_data = self.load_all()
                all_data[key] = value
                self._write(all_data)

    def update(self, updates: Dict[str, Any]) -> None:
        """批量更新字段并持久化。"""
        if not updates:
            return
        with self._lock:
            secrets = self._load_secrets()
            settings_data = self.load_all()
            secret_changed = False
            setting_changed = False
            for key, value in updates.items():
                if key in SENSITIVE_KEYS:
                    if isinstance(value, str) and value:
                        secrets[key] = _encrypt_secret(value, self._base_dir)
                    else:
                        secrets.pop(key, None)
                    if key in settings_data:
                        del settings_data[key]
                        setting_changed = True
                    secret_changed = True
                else:
                    settings_data[key] = value
                    setting_changed = True
            if secret_changed:
                self._write_secrets(secrets)
            if setting_changed:
                self._write(settings_data)

    def remove(self, key: str) -> None:
        """删除单个字段（恢复为默认值）。"""
        with self._lock:
            if key in SENSITIVE_KEYS:
                secrets = self._load_secrets()
                secrets.pop(key, None)
                self._write_secrets(secrets)
            settings_data = self.load_all()
            if key in settings_data:
                del settings_data[key]
                self._write(settings_data)

    # ----------------------------------------------------------
    #  内部
    # ----------------------------------------------------------
    def _write(self, data: Dict[str, Any]) -> None:
        # 敏感字段永远不应写入 _settings.json（由 _write_secrets 单独加密存储）
        data = {k: v for k, v in data.items() if k not in SENSITIVE_KEYS}
        tmp = self._path + ".tmp"
        os.makedirs(self._base_dir, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self._path)


def mask_secret(value: str) -> str:
    """对 API Key 等敏感串做脱敏：保留前 3 位与后 4 位。"""
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:3]}{'*' * 6}{value[-4:]}"
