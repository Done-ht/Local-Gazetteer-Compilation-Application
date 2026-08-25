# -*- coding: utf-8 -*-
"""DocProof 服务器配置管理：密钥、token 限额、校对规则。

持久化到 %APPDATA%\\DocProof\\_server_config.json，敏感字段（API Key 等）
用 Windows DPAPI 加密后存储——密文绑定当前 Windows 用户账户，
换电脑/换用户无法解密。开发模式（未打包）下同样使用该路径。
"""
import base64
import json
import os
import sys
import threading

# 配置文件读写锁：防止多线程并发 save 损坏 JSON
_cfg_lock = threading.Lock()


def _user_config_dir() -> str:
    """返回用户级配置目录：%APPDATA%\\DocProof（Windows）；
    非 Windows 回退到 ~/.config/docproof。目录不存在时自动创建。

    仅用于服务端配置（_server_config.json）与日志；用户数据
    （_users.json / _sessions.json）见 USER_DATA_DIR。
    """
    appdata = os.environ.get("APPDATA")
    if appdata:
        d = os.path.join(appdata, "DocProof")
    else:
        d = os.path.join(os.path.expanduser("~"), ".config", "docproof")
    os.makedirs(d, exist_ok=True)
    return d


# 用户数据目录：存放 _users.json / _sessions.json。
# 优先用环境变量 DOCPROOF_USER_DATA_DIR 覆盖。
# 默认复用 ~/biaoshifu 的账号体系：两边的 _users.json/_sessions.json 结构与
# PBKDF2 哈希完全一致（见 users._hash_password），跨进程 .lock 也同名互斥，可直接共用；
# 该目录不存在时退回独立目录（全局配置同目录下的 users 子目录）。
def _default_user_data_dir() -> str:
    legacy = os.path.join(os.path.expanduser("~"), "biaoshifu")
    if os.path.isdir(legacy):
        return legacy
    return os.path.join(_user_config_dir(), "users")


USER_DATA_DIR = os.environ.get("DOCPROOF_USER_DATA_DIR", _default_user_data_dir())


# 打包后（PyInstaller frozen）：仍用 _APP_DIR 找旧版明文 config.json 做一次性迁移
if getattr(sys, "frozen", False):
    _APP_DIR = os.path.dirname(sys.executable)
else:
    _APP_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(_user_config_dir(), "_server_config.json")
# 旧版明文配置文件路径（仅用于首次启动时一次性迁移到新加密路径）
_LEGACY_CONFIG_PATH = os.path.join(_APP_DIR, "config.json")


# ---------- Windows DPAPI 敏感凭据加密 ----------
# 密文绑定当前 Windows 用户账户，换电脑/换用户无法解密。
# config.json 里只存 base64 密文（带前缀），人眼看不到原 key。
_DPAPI_PREFIX = "DPAPI:"
_SENSITIVE_KEYS = ("llm_api_key", "xf_appid", "xf_api_key", "xf_api_secret")


def _init_dpapi():
    """初始化 ctypes 与 DATA_BLOB 结构（wintypes 不自带 DATA_BLOB，必须自定义）。
    返回 (crypt32, kernel32, DATA_BLOB) 或 (None, None, None) 表示不可用。
    """
    try:
        import ctypes
        from ctypes import wintypes

        class DATA_BLOB(ctypes.Structure):
            # pbData 用 POINTER(c_char)：create_string_buffer 返回的 c_char_Array_N
            # 与之兼容，无需 cast
            _fields_ = [("cbData", wintypes.DWORD),
                        ("pbData", ctypes.POINTER(ctypes.c_char))]

        crypt32 = ctypes.windll.crypt32
        kernel32 = ctypes.windll.kernel32
        # 显式声明签名，避免 ctypes 默认 int 截断 64 位指针
        crypt32.CryptProtectData.argtypes = [
            ctypes.POINTER(DATA_BLOB), wintypes.LPCWSTR, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD,
            ctypes.POINTER(DATA_BLOB),
        ]
        crypt32.CryptProtectData.restype = wintypes.BOOL
        crypt32.CryptUnprotectData.argtypes = [
            ctypes.POINTER(DATA_BLOB), ctypes.POINTER(wintypes.LPWSTR),
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD,
            ctypes.POINTER(DATA_BLOB),
        ]
        crypt32.CryptUnprotectData.restype = wintypes.BOOL
        kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
        kernel32.LocalFree.restype = wintypes.HLOCAL
        return crypt32, kernel32, DATA_BLOB
    except Exception:
        return None, None, None


def _dpapi_protect(data):
    """用 Windows DPAPI 加密字符串，返回带前缀的 base64 密文；不可用时返回原文"""
    if not data:
        return data
    crypt32, kernel32, DATA_BLOB = _init_dpapi()
    if crypt32 is None:
        return data
    import ctypes
    try:
        blob = data.encode("utf-8")
        buf = ctypes.create_string_buffer(blob, len(blob))
        in_desc = DATA_BLOB(len(blob), buf)
        out_desc = DATA_BLOB()
        if not crypt32.CryptProtectData(
                ctypes.byref(in_desc), None, None, None, None, 0, ctypes.byref(out_desc)):
            return data
        try:
            raw = ctypes.string_at(out_desc.pbData, out_desc.cbData)
            return _DPAPI_PREFIX + base64.b64encode(raw).decode("ascii")
        finally:
            kernel32.LocalFree(out_desc.pbData)
    except Exception:
        return data


def _dpapi_unprotect(token):
    """解密 DPAPI 密文；无前缀视为旧明文原样返回（向后兼容）；解密失败返回空"""
    if not token or not token.startswith(_DPAPI_PREFIX):
        return token
    crypt32, kernel32, DATA_BLOB = _init_dpapi()
    if crypt32 is None:
        return ""
    import ctypes
    try:
        raw = base64.b64decode(token[len(_DPAPI_PREFIX):])
        buf = ctypes.create_string_buffer(raw, len(raw))
        in_desc = DATA_BLOB(len(raw), buf)
        out_desc = DATA_BLOB()
        if not crypt32.CryptUnprotectData(
                ctypes.byref(in_desc), None, None, None, None, 0, ctypes.byref(out_desc)):
            return ""
        try:
            plain = ctypes.string_at(out_desc.pbData, out_desc.cbData)
            return plain.decode("utf-8")
        finally:
            kernel32.LocalFree(out_desc.pbData)
    except Exception:
        return ""


def _encrypt_cfg(cfg: dict) -> dict:
    """返回一份副本，其中敏感字段已 DPAPI 加密（用于落盘）"""
    out = dict(cfg)
    for k in _SENSITIVE_KEYS:
        if out.get(k):
            out[k] = _dpapi_protect(out[k])
    return out


def _decrypt_cfg(cfg: dict) -> dict:
    """就地解密 cfg 中的敏感字段（从磁盘读出后调用）"""
    for k in _SENSITIVE_KEYS:
        v = cfg.get(k)
        if isinstance(v, str) and v.startswith(_DPAPI_PREFIX):
            cfg[k] = _dpapi_unprotect(v)
    return cfg

# DEAResource 参考资料目录（仅开发环境 bootstrap 时读取，打包后不适用）。
# 可用环境变量 DOCPROOF_DEARESOURCE_DIR 覆盖；默认取当前用户桌面下的 win/DEAResource。
DEARESOURCE_DIR = os.environ.get(
    "DOCPROOF_DEARESOURCE_DIR",
    os.path.expanduser("~/Desktop/win/DEAResource"),
)

# 预置规则开关：key -> 默认是否启用
# 默认全部关闭，意味着这些“风格类”问题不报；用户可按需打开。
PRESET_SWITCHES = {
    "check_cn_number_space": "数字与汉字之间空格",
    "check_en_number_space": "英文与汉字之间空格",
    "check_full_half_punct": "全角/半角标点混用",
    "check_redundant_space": "句中多余空格",
    "check_number_format": "数字格式（千分位/单位）",
}

DEFAULTS = {
    "llm_provider": "deepseek",
    "llm_api_key": "",
    "llm_base_url": "https://api.deepseek.com",
    "llm_model": "deepseek-v4-flash",
    "xf_appid": "",
    "xf_api_key": "",
    "xf_api_secret": "",
    "token_limit": 1000000,
    # 校对规则配置
    "rule_switches": {k: False for k in PRESET_SWITCHES},
    "custom_rules": [],
    # 二次复核：对每块检出的错误再做一轮交叉核对，降低误报、区分明显/存疑
    "enable_review": True,
    "review_context_chars": 800,
}

# AI 服务商预设：降低入手难度，选服务商即自动填 base_url 和推荐模型。
# 底层客户端为 OpenAI 兼容接口，任何兼容服务商均可使用。
# 注意：deepseek-chat / deepseek-reasoner 已于 2026-07-24 停用，
# DeepSeek 请用 V4 命名（flash 非思考 / pro 思考模式）。
PROVIDER_PRESETS = [
    {"name": "DeepSeek", "value": "deepseek",
     "base_url": "https://api.deepseek.com",
     "models": ["deepseek-v4-flash", "deepseek-v4-pro"],
     "default_model": "deepseek-v4-flash"},
    {"name": "OpenAI", "value": "openai",
     "base_url": "https://api.openai.com/v1",
     "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
     "default_model": "gpt-4o-mini"},
    {"name": "小米 MiMo", "value": "mimo",
     "base_url": "https://api.xiaomimimo.com/v1",
     "models": ["mimo-v2.5-pro", "mimo-v2.5"],
     "default_model": "mimo-v2.5-pro"},
    {"name": "自定义（OpenAI 兼容）", "value": "custom",
     "base_url": "", "models": [], "default_model": ""},
]
# 向后兼容：旧代码可能引用
DEEPSEEK_MODELS = ["deepseek-v4-flash", "deepseek-v4-pro"]

_cache = None


def _merge(target: dict, src: dict) -> dict:
    """浅层 deep merge：对 dict 字段递归合并，其余直接覆盖。
    用于保证新增的 rule_switches 子键在旧配置文件缺失时也能补回默认值。
    """
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(target.get(k), dict):
            merged = dict(target[k])
            for sk, sv in v.items():
                merged.setdefault(sk, sv)
            target[k] = merged
        else:
            target.setdefault(k, v)
    return target


def load() -> dict:
    """加载配置；文件不存在时返回默认值。

    - 敏感字段（API Key 等）在磁盘上是 DPAPI 密文，读出后自动解密；
    - 嵌套字段（如 rule_switches）做兼容合并：磁盘上缺失的子键用默认值补齐，
      避免旧 config 升级后丢失新增的预置开关；
    - 首次启动若新加密配置不存在但旧版明文 config.json 存在，自动迁移并加密落盘。
    """
    global _cache
    cfg = dict(DEFAULTS)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                disk = json.load(f)
            # 顶层标量字段直接覆盖
            for k, v in disk.items():
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    continue
                cfg[k] = v
            # 向后兼容：旧版 deepseek_* 键迁移到通用 llm_* 键
            if not cfg.get("llm_api_key") and disk.get("deepseek_api_key"):
                cfg["llm_api_key"] = disk["deepseek_api_key"]
            if not cfg.get("llm_base_url") and disk.get("deepseek_base_url"):
                cfg["llm_base_url"] = disk["deepseek_base_url"]
            if not cfg.get("llm_model") and disk.get("deepseek_model"):
                cfg["llm_model"] = disk["deepseek_model"]
            if not cfg.get("llm_provider"):
                cfg["llm_provider"] = "deepseek"
            # 嵌套 dict 字段做 merge，补齐缺失子键
            _merge(cfg, DEFAULTS)
            # 解密敏感字段（DPAPI 密文 → 明文，供运行时使用）
            _decrypt_cfg(cfg)
        except (json.JSONDecodeError, OSError):
            pass
    elif os.path.exists(_LEGACY_CONFIG_PATH):
        # 一次性迁移：旧版明文 config.json → 新加密 _server_config.json
        try:
            with open(_LEGACY_CONFIG_PATH, "r", encoding="utf-8") as f:
                disk = json.load(f)
            for k, v in disk.items():
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    continue
                cfg[k] = v
            if not cfg.get("llm_api_key") and disk.get("deepseek_api_key"):
                cfg["llm_api_key"] = disk["deepseek_api_key"]
            if not cfg.get("llm_base_url") and disk.get("deepseek_base_url"):
                cfg["llm_base_url"] = disk["deepseek_base_url"]
            if not cfg.get("llm_model") and disk.get("deepseek_model"):
                cfg["llm_model"] = disk["deepseek_model"]
            _merge(cfg, DEFAULTS)
            # 旧文件中敏感字段是明文，无需解密；直接加密落盘到新路径
            save(cfg)
        except (json.JSONDecodeError, OSError):
            pass
    _cache = cfg
    return cfg


def save(cfg: dict) -> None:
    """保存配置到 _server_config.json（敏感字段 DPAPI 加密后落盘）。
    加文件锁 + 原子写（临时文件 + os.replace），防并发损坏。
    """
    global _cache
    with _cfg_lock:
        _cache = dict(cfg)
        # 落盘前加密敏感字段；缓存里保留明文供运行时使用
        encrypted = _encrypt_cfg(cfg)
        tmp = CONFIG_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(encrypted, f, ensure_ascii=False, indent=2)
        os.replace(tmp, CONFIG_PATH)


# ---------- 校对规则 ----------

def get_rules(cfg: dict = None) -> dict:
    """从配置中提取规则配置，返回 {switches: dict, custom: list[str]}"""
    cfg = cfg if cfg is not None else (_cache or load())
    switches = dict(cfg.get("rule_switches") or {})
    # 补齐可能新增的预置开关（磁盘旧文件可能缺键）
    for k in PRESET_SWITCHES:
        switches.setdefault(k, False)
    custom = [r for r in (cfg.get("custom_rules") or []) if isinstance(r, str) and r.strip()]
    return {"switches": switches, "custom": custom}


def set_rules(cfg: dict, switches: dict, custom: list) -> dict:
    """把规则配置写回 cfg（不立即落盘），返回 cfg"""
    cfg["rule_switches"] = {k: bool(v) for k, v in switches.items()} if switches else {}
    cfg["custom_rules"] = [r.strip() for r in (custom or []) if isinstance(r, str) and r.strip()]
    return cfg


def load_rules_from_file(path: str) -> dict:
    """从预设 JSON 文件加载规则。
    文件格式：{"rule_switches": {...}, "custom_rules": [...]}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    switches = {k: bool(v) for k, v in (data.get("rule_switches") or {}).items()}
    custom = [r.strip() for r in (data.get("custom_rules") or []) if isinstance(r, str) and r.strip()]
    return {"switches": switches, "custom": custom}


def save_rules_to_file(path: str, switches: dict, custom: list) -> None:
    """把规则导出为预设 JSON 文件，便于在不同场景间复用"""
    payload = {
        "rule_switches": {k: bool(v) for k, v in (switches or {}).items()},
        "custom_rules": [r.strip() for r in (custom or []) if isinstance(r, str) and r.strip()],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def llm_configured(cfg: dict = None) -> bool:
    """AI 模型是否已配置（API Key 非空）"""
    cfg = cfg if cfg is not None else (_cache or load())
    return bool(cfg.get("llm_api_key"))


# 向后兼容别名
deepseek_configured = llm_configured


def xfyun_configured(cfg: dict = None) -> bool:
    cfg = cfg if cfg is not None else (_cache or load())
    return bool(cfg.get("xf_appid") and cfg.get("xf_api_key") and cfg.get("xf_api_secret"))


def _read_lines(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _parse_xfyun_secret(lines: list) -> dict:
    """解析讯飞密钥文件，兼容两种格式：
    1. 三行纯值，依次 APPID / APISecret / APIKey
    2. 键名行与值行交替（"APPID" / 值 / "APISecret" / 值 ...），或 "KEY: value" 同行
    """
    keys = {"APPID": "xf_appid", "APISECRET": "xf_api_secret", "APIKEY": "xf_api_key"}
    result, i = {}, 0
    while i < len(lines):
        line = lines[i]
        upper = line.upper().rstrip(":：").strip()
        if upper in keys:
            # "KEY: value" 同行，或键名单占一行、值在下一行
            value = line[len(upper):].lstrip(":：").strip()
            if not value and i + 1 < len(lines):
                i += 1
                value = lines[i]
            result[keys[upper]] = value
        i += 1
    if len(result) == 3:
        return result
    # 回退：按纯值三行处理，顺序 APPID / APISecret / APIKey
    if len(lines) >= 3:
        return {"xf_appid": lines[0], "xf_api_secret": lines[1], "xf_api_key": lines[2]}
    return {}


def bootstrap_from_dearesource(cfg: dict = None, save_to_file: bool = True) -> dict:
    """若配置为空，从 DEAResource 的密钥文件读取并填入。

    - OCRsecret-讯飞.txt：三行依次为 APPID / APISecret / APIKey
    - deepseek-api.txt：sk- 开头的一行
    """
    cfg = cfg if cfg is not None else load()

    if not llm_configured(cfg):
        path = os.path.join(DEARESOURCE_DIR, "deepseek-api.txt")
        if os.path.exists(path):
            for line in _read_lines(path):
                if line.startswith("sk-"):
                    cfg["llm_api_key"] = line
                    break
            else:
                lines = _read_lines(path)
                if lines:
                    cfg["llm_api_key"] = lines[0]

    if not xfyun_configured(cfg):
        path = os.path.join(DEARESOURCE_DIR, "OCRsecret-讯飞.txt")
        if os.path.exists(path):
            cfg.update(_parse_xfyun_secret(_read_lines(path)))

    if save_to_file:
        save(cfg)
    return cfg
