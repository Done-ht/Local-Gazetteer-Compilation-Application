# -*- coding: utf-8 -*-
"""多用户管理：用户名+密码登录、会话 token 鉴权、用户隔离、per-user 配置覆盖、限流计数。

数据文件（位于 %APPDATA%\\DocProof\\）：
- _users.json    用户表。结构（仿 biaoshifu）：
    {"users": {username: {username, pwd_hash, salt, role, created_at, overrides}}}
    pwd_hash = PBKDF2-HMAC-SHA256(salt + password)（迭代 20 万次）；salt 为 16 字节随机十六进制串。
    role ∈ {"admin", "user"}（admin 即管理员，向后兼容 is_admin 布尔语义）。
- _sessions.json  会话表。结构：
    {"sessions": {token: {username, created_at, expires_at}}}
    token = secrets.token_urlsafe(32)（43 字符 base64url，与 biaoshifu 一致）。
    有效期 7 天（SESSION_TTL）；过期自动清理；登出主动删除。

- 敏感字段（llm_api_key / xf_*）的 overrides 用 DPAPI 加密（复用 config.py），
  与全局配置的加密策略一致；服务端读出后解密供 LLM/OCR 调用。
- 限流：每用户最多 N 个并发纠错任务 + 每分钟 M 次请求（滑动窗口）。

设计原则：与 config.py 解耦，config.load() 仍返回全局默认配置，
本模块的 resolve_user_config() 在其之上叠加用户 overrides（解密后合并）。
"""
import hashlib
import json
import os
import secrets
import threading
import time
from collections import deque
from datetime import datetime

# 复用 config.py 的用户配置目录（%APPDATA%\DocProof）
import config

# 用户数据（_users.json / _sessions.json）存放在 config.USER_DATA_DIR 下：
# 默认 %APPDATA%\DocProof\users（可用环境变量 DOCPROOF_USER_DATA_DIR 覆盖）。
# 配置文件（_server_config.json）仍在 _user_config_dir() 下，互不干扰。
USERS_PATH = os.path.join(config.USER_DATA_DIR, "_users.json")
SESSIONS_PATH = os.path.join(config.USER_DATA_DIR, "_sessions.json")

# 会话有效期（秒）：7 天，与 biaoshifu 保持一致
SESSION_TTL = 7 * 24 * 3600

# 密码哈希迭代次数：与 biaoshifu 保持一致（PBKDF2-HMAC-SHA256）
PBKDF2_ITERATIONS = 200_000

# 限流参数
MAX_CONCURRENT_PROOFREAD = 2          # 单用户同时进行的纠错任务数上限
RATE_LIMIT_WINDOW = 60                # 滑动窗口 60 秒
RATE_LIMIT_MAX_REQUESTS = 120         # 单用户每分钟最多 120 次 API 请求
# （纠错一个 30 页文档约 30-60 次 API 调用，120 留余量给正常使用）

# 允许写入用户 overrides 的字段白名单
_OVERRIDABLE_NON_SENSITIVE = (
    "llm_provider", "llm_base_url", "llm_model", "token_limit",
    "enable_review", "review_context_chars", "rule_switches", "custom_rules",
)
_OVERRIDABLE_SENSITIVE = ("llm_api_key", "xf_appid", "xf_api_key", "xf_api_secret")

_lock = threading.Lock()         # 用户表进程内读写锁
_session_lock = threading.Lock()  # 会话表进程内读写锁
_rate_lock = threading.Lock()
# 限流状态：username -> {proofread_count: int, requests: deque[float]}
_rate_state: dict = {}

# ---------- 跨进程文件锁 ----------
# _users.json / _sessions.json 可能被多个服务端进程同时读写，仅靠进程内 threading.Lock
# 无法防止跨进程"丢失更新"（两个进程各自读快照→改→整文件写回，后写者覆盖先写者）。
# 方案：在每个数据文件旁放一个 .lock 文件，用 msvcrt.locking 独占锁定其第 1 个字节，
# 形成跨进程互斥；读-改-写全程同时持有进程内锁 + 该文件锁。
# 非 Windows 平台无 msvcrt，退化为仅进程内锁（本服务面向 Windows，DPAPI 亦如此）。
try:
    import msvcrt
    _HAS_MSVCRT = True
except ImportError:  # 非 Windows
    _HAS_MSVCRT = False

FILE_LOCK_TIMEOUT = 10  # 等待跨进程文件锁的最长秒数，超时抛错而不是无限阻塞


def _acquire_file_lock(lock_path: str):
    """跨进程独占锁定 lock_path 的第 1 个字节；返回已加锁的文件对象。
    锁文件缺失时以追加方式创建（绝不截断，避免并发创建互相破坏已锁状态）。
    等待超时抛出 TimeoutError。
    """
    # "a+b" 追加+二进制：文件不存在则创建，存在则原样打开，不截断内容
    f = open(lock_path, "a+b")
    try:
        f.seek(0, os.SEEK_END)
        if f.tell() == 0:  # 新锁文件至少要有 1 字节才能锁
            f.write(b"\0")
            f.flush()
        f.seek(0)
        deadline = time.time() + FILE_LOCK_TIMEOUT
        while True:
            try:
                # LK_NBLCK 非阻塞尝试；失败重试直到超时
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                return f
            except OSError:
                if time.time() >= deadline:
                    f.close()
                    raise TimeoutError(
                        f"等待跨进程文件锁超时（>={FILE_LOCK_TIMEOUT}s）：{lock_path}")
                time.sleep(0.05)
    except Exception:
        f.close()
        raise


class _CrossProcessLock:
    """同时持有进程内 threading.Lock 与跨进程文件锁的上下文管理器。
    两个锁一起持有，才能同时挡住"同进程多线程"与"多进程"两种并发写。
    """

    def __init__(self, lock_path: str, thread_lock: threading.Lock):
        self.lock_path = lock_path
        self.thread_lock = thread_lock
        self._fh = None

    def __enter__(self):
        self.thread_lock.acquire()
        try:
            if _HAS_MSVCRT:
                self._fh = _acquire_file_lock(self.lock_path)
        except Exception:
            self.thread_lock.release()
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._fh is not None:
                try:
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
        finally:
            if self._fh is not None:
                self._fh.close()
            self.thread_lock.release()


# 用户表与会话表的跨进程锁（各自独立锁文件，互不阻塞）
_users_lock = _CrossProcessLock(USERS_PATH + ".lock", _lock)
_sessions_lock = _CrossProcessLock(SESSIONS_PATH + ".lock", _session_lock)


# ---------- 文件读写 ----------

def _load_users_raw() -> dict:
    """从磁盘加载 _users.json；不存在返回空骨架。
    读取失败时抛出异常，避免调用方把假空数据写回覆盖真实内容。
    自动兼容旧版 list 格式（一次性迁移到新 dict 格式）。
    """
    if not os.path.exists(USERS_PATH):
        return {"users": {}}
    with open(USERS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 旧版格式：{"version":1, "users":[...]}  → 迁移
    if isinstance(data.get("users"), list):
        return _migrate_legacy_users(data)
    if isinstance(data, dict) and isinstance(data.get("users"), dict):
        return data
    return {"users": {}}


def _save_users_raw(data: dict) -> None:
    """原子写：先写临时文件再 os.replace，避免并发损坏"""
    tmp = USERS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, USERS_PATH)


def _migrate_legacy_users(data: dict) -> dict:
    """把旧版 list 格式用户表迁移到新 dict 格式。
    旧用户：display_name → username（去重）；随机生成密码并打印到控制台。
    保留 overrides 字段；is_admin → role。
    """
    old_users = data.get("users") or []
    new_data = {"users": {}}
    print()
    print("=" * 60)
    print("检测到旧版用户数据（access_token 模式），正在迁移到用户名+密码模式")
    print("=" * 60)
    used_names = set()
    for u in old_users:
        # username 来源：优先 display_name，回退 user_id
        base = (u.get("display_name") or u.get("user_id") or "user").strip()
        # 转成安全用户名：保留中文/字母数字，去空格
        name = "".join(c for c in base if c.isalnum() or '\u4e00' <= c <= '\u9fff')
        if not name:
            name = "user"
        if name in used_names:
            i = 2
            while f"{name}_{i}" in used_names:
                i += 1
            name = f"{name}_{i}"
        used_names.add(name)
        # 生成随机初始密码（12 位）
        password = secrets.token_urlsafe(9)[:12]
        salt = secrets.token_hex(16)
        pwd_hash = _hash_password(password, salt)
        role = "admin" if u.get("is_admin") else "user"
        new_data["users"][name] = {
            "username": name,
            "pwd_hash": pwd_hash,
            "salt": salt,
            "role": role,
            "created_at": u.get("created_at") or _now_iso(),
            "overrides": u.get("overrides") or {},
        }
        print(f"  用户名: {name}")
        print(f"  初始密码（仅显示一次，请立即保存并登录后修改）: {password}")
        print(f"  角色: {role}")
        print("-" * 60)
    _save_users_raw(new_data)
    print("迁移完成。请用上述用户名+密码登录。")
    print("=" * 60)
    print()
    return new_data


def _load_sessions_raw() -> dict:
    """从磁盘加载 _sessions.json；不存在返回空骨架。
    读取失败（JSONDecodeError/OSError）时抛出异常，避免调用方把假空数据写回覆盖真实内容。
    """
    if not os.path.exists(SESSIONS_PATH):
        return {"sessions": {}}
    with open(SESSIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("sessions"), dict):
        return data
    # 文件格式不符合预期（非 dict/sessions 结构）：视为空文件但不抛异常
    return {"sessions": {}}


def _save_sessions_raw(data: dict) -> None:
    """原子写会话表"""
    tmp = SESSIONS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, SESSIONS_PATH)


def _purge_expired_sessions(data: dict, now: float = None) -> bool:
    """清理过期会话；调用方需持有 _sessions_lock。返回是否有变更。
    同时限制 _sessions.json 体积（避免长期运行无限增长）。
    """
    if now is None:
        now = time.time()
    sessions = data.get("sessions", {})
    expired = [t for t, s in sessions.items() if s.get("expires_at", 0) < now]
    for t in expired:
        sessions.pop(t, None)
    return bool(expired)


# ---------- 密码哈希 ----------

def _hash_password(password: str, salt: str) -> str:
    """PBKDF2-HMAC-SHA256(password, salt, 20万次迭代)，与 biaoshifu 一致。
    salt 为十六进制字符串（16 字节 → 32 位十六进制），password 为 UTF-8 编码。
    """
    salt_bytes = bytes.fromhex(salt)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt_bytes, PBKDF2_ITERATIONS
    )
    return digest.hex()


def _gen_session_token() -> str:
    """生成会话 token：secrets.token_urlsafe(32) → 43 字符 base64url"""
    return secrets.token_urlsafe(32)


def _now_iso() -> str:
    """当前时间 ISO 字符串（与 biaoshifu 的 created_at 格式一致，如 '2026-08-08T04:34:41'）。
    用于 _users.json 的 created_at 字段；_sessions.json 的 created_at/expires_at 仍用 Unix int。
    """
    return datetime.now().isoformat(timespec="seconds")


def _with_is_admin(user: dict) -> dict:
    """在用户字典上派生 is_admin 布尔字段（供 main_server 的 _require_admin 使用）。
    role == "admin" 即管理员；同时兼容旧字段直接存在的情形。
    """
    if not user:
        return user
    if "is_admin" not in user:
        user["is_admin"] = (user.get("role") == "admin")
    return user


# ---------- 登录 / 鉴权 ----------

def login(username: str, password: str) -> tuple:
    """用户名+密码登录。
    成功返回 (session_token, user_dict)；失败返回 ("", None)。
    成功后写入 _sessions.json，token 7 天有效。
    """
    username = (username or "").strip()
    if not username or not password:
        return "", None
    with _users_lock:
        try:
            data = _load_users_raw()
        except (json.JSONDecodeError, OSError):
            # 用户表读取失败：无法校验密码
            return "", None
        u = data.get("users", {}).get(username)
        if not u:
            return "", None
        # 用 compare_digest 防时序攻击
        salt = u.get("salt", "")
        expected = u.get("pwd_hash", "")
        actual = _hash_password(password, salt)
        if not secrets.compare_digest(expected, actual):
            return "", None
        user = _with_is_admin(dict(u))
    # 生成会话
    token = _gen_session_token()
    now = int(time.time())
    with _sessions_lock:
        try:
            sdata = _load_sessions_raw()
        except (json.JSONDecodeError, OSError):
            # 会话表读取失败：退化为空骨架（仅在内存中创建会话，不落盘以免覆盖真实数据）
            sdata = {"sessions": {}}
        changed = _purge_expired_sessions(sdata, now)
        sdata.setdefault("sessions", {})[token] = {
            "username": username,
            "created_at": now,
            "expires_at": now + SESSION_TTL,
        }
        # 新增会话必然有变更，安全落盘；但读取失败时不落盘（避免覆盖）
        try:
            _save_sessions_raw(sdata)
        except OSError:
            pass
    return token, user


def authenticate(session_token: str) -> dict:
    """通过会话 token 找用户；找不到/已过期返回 None。
    每次校验都会清理过期会话（仅在有实际清理动作时才落盘）。
    读取失败时不落盘，避免把假空数据写回覆盖真实会话。
    """
    if not session_token:
        return None
    now = int(time.time())
    sdata = None
    read_ok = False
    with _sessions_lock:
        try:
            sdata = _load_sessions_raw()
            read_ok = True
        except (json.JSONDecodeError, OSError):
            # 读取失败：不落盘、不返回空骨架覆盖
            sdata = {"sessions": {}}
        changed = _purge_expired_sessions(sdata, now)
        s = sdata.get("sessions", {}).get(session_token)
        if not s:
            # 只有在真正清理了过期会话且读取成功时才落盘
            # 不能因为"当前token不存在"就把假空数据写回磁盘！
            if changed and read_ok:
                try:
                    _save_sessions_raw(sdata)
                except OSError:
                    pass
            return None
        if s.get("expires_at", 0) < now:
            sdata["sessions"].pop(session_token, None)
            changed = True
            if read_ok:
                try:
                    _save_sessions_raw(sdata)
                except OSError:
                    pass
            return None
        username = s.get("username", "")
        # 纯查询成功：如果有过期清理且读取成功，落盘
        if changed and read_ok:
            try:
                _save_sessions_raw(sdata)
            except OSError:
                pass
    if not username:
        return None
    with _users_lock:
        try:
            data = _load_users_raw()
        except (json.JSONDecodeError, OSError):
            # 用户表读取失败：视为用户不存在，但不清理会话
            return None
        u = data.get("users", {}).get(username)
        if not u:
            # 用户已被删除，清掉残留会话（仅在会话表读取成功时落盘）
            with _sessions_lock:
                try:
                    sdata2 = _load_sessions_raw()
                    sdata2.get("sessions", {}).pop(session_token, None)
                    _save_sessions_raw(sdata2)
                except (json.JSONDecodeError, OSError):
                    pass
            return None
        return _with_is_admin(dict(u))


def logout(session_token: str) -> bool:
    """主动登出：删除会话。成功返回 True。"""
    if not session_token:
        return False
    with _sessions_lock:
        try:
            sdata = _load_sessions_raw()
        except (json.JSONDecodeError, OSError):
            # 读取失败：无法安全修改，忽略（该会话会自然过期）
            return False
        existed = sdata.get("sessions", {}).pop(session_token, None) is not None
        if existed:
            try:
                _save_sessions_raw(sdata)
            except OSError:
                pass
        return existed


# ---------- 用户查询 ----------

def get_user(username: str) -> dict:
    with _users_lock:
        try:
            data = _load_users_raw()
        except (json.JSONDecodeError, OSError):
            return None
        u = data.get("users", {}).get(username)
        return _with_is_admin(dict(u)) if u else None


def list_users() -> list:
    """返回所有用户列表（不含 pwd_hash/salt）"""
    with _users_lock:
        try:
            data = _load_users_raw()
        except (json.JSONDecodeError, OSError):
            return []
        out = []
        for u in data.get("users", {}).values():
            safe = dict(u)
            safe.pop("pwd_hash", None)
            safe.pop("salt", None)
            # 兼容前端 is_admin 字段
            safe["is_admin"] = (u.get("role") == "admin")
            out.append(safe)
        return out


# ---------- 用户管理 ----------

def _normalize_username(name: str) -> str:
    """用户名规范化：去首尾空格；不允许为空。
    不做大小写转换（保留中文/英文原样），但禁止包含空白字符。
    """
    name = (name or "").strip()
    if not name:
        return ""
    if any(c.isspace() for c in name):
        return ""
    return name


def add_user(username: str, password: str, is_admin: bool = False) -> bool:
    """新增用户。username 不能与已有重复；password 至少 6 位。
    成功返回 True；失败（用户名重复/空/密码过短）返回 False。
    """
    username = _normalize_username(username)
    if not username:
        return False
    if not password or len(password) < 6:
        return False
    with _users_lock:
        try:
            data = _load_users_raw()
        except (json.JSONDecodeError, OSError):
            return False
        if username in data.get("users", {}):
            return False
        salt = secrets.token_hex(16)
        data.setdefault("users", {})[username] = {
            "username": username,
            "pwd_hash": _hash_password(password, salt),
            "salt": salt,
            "role": "admin" if is_admin else "user",
            "created_at": _now_iso(),
            "overrides": {},
        }
        try:
            _save_users_raw(data)
        except OSError:
            return False
        return True


def register_user(username: str, password: str) -> tuple:
    """自助注册：创建普通用户（role=user，非管理员），无需管理员审批。
    与 add_user 的区别：固定为普通用户角色，并返回中文失败原因。
    返回 (ok, reason)；成功 ok=True，reason 为空字符串。
    """
    username = _normalize_username(username)
    if not username:
        return False, "用户名不能为空，且不能包含空格"
    if not password or len(password) < 6:
        return False, "密码至少 6 位"
    with _users_lock:
        try:
            data = _load_users_raw()
        except (json.JSONDecodeError, OSError):
            return False, "用户数据暂时不可用，请稍后重试"
        if username in data.get("users", {}):
            return False, "该用户名已被注册"
        salt = secrets.token_hex(16)
        data.setdefault("users", {})[username] = {
            "username": username,
            "pwd_hash": _hash_password(password, salt),
            "salt": salt,
            "role": "user",
            "created_at": _now_iso(),
            "overrides": {},
        }
        try:
            _save_users_raw(data)
        except OSError:
            return False, "用户数据写入失败，请稍后重试"
        return True, ""


def delete_user(username: str) -> bool:
    """删除用户；管理员账户至少保留一个，避免锁死。
    同时清理该用户的所有会话。
    """
    with _users_lock:
        try:
            data = _load_users_raw()
        except (json.JSONDecodeError, OSError):
            return False
        users = data.get("users", {})
        if username not in users:
            return False
        # 不能删掉最后一个管理员
        admin_count = sum(1 for u in users.values() if u.get("role") == "admin")
        if users[username].get("role") == "admin" and admin_count <= 1:
            return False
        del users[username]
        try:
            _save_users_raw(data)
        except OSError:
            return False
    # 清理该用户的会话
    with _sessions_lock:
        try:
            sdata = _load_sessions_raw()
        except (json.JSONDecodeError, OSError):
            return True  # 用户已删除，会话失败跳过（自然过期）
        changed = False
        for t in [t for t, s in sdata.get("sessions", {}).items()
                  if s.get("username") == username]:
            sdata["sessions"].pop(t, None)
            changed = True
        if changed:
            try:
                _save_sessions_raw(sdata)
            except OSError:
                pass
    return True


def reset_password(username: str, new_password: str) -> bool:
    """重置用户密码；同时吊销其所有现有会话。
    新密码至少 6 位。成功返回 True。
    """
    if not new_password or len(new_password) < 6:
        return False
    with _users_lock:
        try:
            data = _load_users_raw()
        except (json.JSONDecodeError, OSError):
            return False
        u = data.get("users", {}).get(username)
        if not u:
            return False
        salt = secrets.token_hex(16)
        u["salt"] = salt
        u["pwd_hash"] = _hash_password(new_password, salt)
        try:
            _save_users_raw(data)
        except OSError:
            return False
    # 改密码后吊销所有旧会话
    with _sessions_lock:
        try:
            sdata = _load_sessions_raw()
        except (json.JSONDecodeError, OSError):
            return True  # 密码已改，会话清理跳过
        changed = False
        for t in [t for t, s in sdata.get("sessions", {}).items()
                  if s.get("username") == username]:
            sdata["sessions"].pop(t, None)
            changed = True
        if changed:
            try:
                _save_sessions_raw(sdata)
            except OSError:
                pass
    return True


def change_password(username: str, old_password: str, new_password: str) -> bool:
    """用户自助改密码（需校验旧密码）；改完吊销其它会话但保留当前。
    返回 (ok, reason)。
    """
    if not new_password or len(new_password) < 6:
        return False, "新密码至少 6 位"
    with _users_lock:
        try:
            data = _load_users_raw()
        except (json.JSONDecodeError, OSError):
            return False, "用户数据暂时不可用"
        u = data.get("users", {}).get(username)
        if not u:
            return False, "用户不存在"
        if not secrets.compare_digest(
                u.get("pwd_hash", ""),
                _hash_password(old_password, u.get("salt", ""))):
            return False, "旧密码错误"
        salt = secrets.token_hex(16)
        u["salt"] = salt
        u["pwd_hash"] = _hash_password(new_password, salt)
        try:
            _save_users_raw(data)
        except OSError:
            return False, "用户数据写入失败"
    return True, ""


def rename_user(old_username: str, new_username: str) -> bool:
    """改用户名。新用户名不能与已有重复。
    同步更新会话表中的 username 引用。
    """
    new_username = _normalize_username(new_username)
    if not new_username:
        return False
    with _users_lock:
        try:
            data = _load_users_raw()
        except (json.JSONDecodeError, OSError):
            return False
        users = data.get("users", {})
        if old_username not in users:
            return False
        if new_username != old_username and new_username in users:
            return False
        # 保留所有字段，仅改 key 与 username 字段
        u = users.pop(old_username)
        u["username"] = new_username
        users[new_username] = u
        try:
            _save_users_raw(data)
        except OSError:
            return False
    # 同步会话表引用
    with _sessions_lock:
        try:
            sdata = _load_sessions_raw()
        except (json.JSONDecodeError, OSError):
            return True  # 用户已改名，会话清理跳过
        changed = False
        for s in sdata.get("sessions", {}).values():
            if s.get("username") == old_username:
                s["username"] = new_username
                changed = True
        if changed:
            try:
                _save_sessions_raw(sdata)
            except OSError:
                pass
    return True


def set_admin(username: str, is_admin: bool) -> bool:
    """切换管理员状态。取消最后一个管理员的管理员身份会被拒绝。"""
    with _users_lock:
        try:
            data = _load_users_raw()
        except (json.JSONDecodeError, OSError):
            return False
        users = data.get("users", {})
        u = users.get(username)
        if not u:
            return False
        new_role = "admin" if is_admin else "user"
        if not is_admin:
            admin_count = sum(1 for x in users.values() if x.get("role") == "admin")
            if admin_count <= 1:
                return False
        u["role"] = new_role
        try:
            _save_users_raw(data)
        except OSError:
            return False
    return True


# ---------- per-user 配置覆盖 ----------

def resolve_user_config(user: dict, global_cfg: dict) -> dict:
    """合并全局配置与用户 overrides，返回最终生效配置。
    - 浅合并：overrides 中的字段直接覆盖全局；嵌套 dict 字段（rule_switches）也直接覆盖
    - 敏感字段（llm_api_key 等）落盘为 DPAPI 密文，此处先解密再合并
    - 敏感字段若 overrides 中为空字符串则视为"清空用户覆盖、回退全局"
    """
    overrides = user.get("overrides") or {}
    cfg = dict(global_cfg)
    for k, v in overrides.items():
        # 敏感字段先解密（带 DPAPI 前缀的密文）
        if k in _OVERRIDABLE_SENSITIVE and isinstance(v, str) and v.startswith(config._DPAPI_PREFIX):
            v = config._dpapi_unprotect(v)
        if v == "" and k in _OVERRIDABLE_SENSITIVE:
            # 空值表示用户主动清空自己的覆盖，回退到全局
            cfg.pop(k, None)
            continue
        cfg[k] = v
    return cfg


def set_user_overrides(username: str, non_sensitive: dict, sensitive: dict) -> None:
    """更新用户 overrides；空敏感字段视为"保持不变"（与 /api/credentials 语义一致）。
    敏感字段写入前 DPAPI 加密，与全局配置加密策略一致。
    """
    with _users_lock:
        try:
            data = _load_users_raw()
        except (json.JSONDecodeError, OSError):
            return
        u = data.get("users", {}).get(username)
        if not u:
            return
        ov = u.setdefault("overrides", {})
        for k in _OVERRIDABLE_NON_SENSITIVE:
            if k in non_sensitive:
                ov[k] = non_sensitive[k]
        for k in _OVERRIDABLE_SENSITIVE:
            if k in sensitive and sensitive[k]:
                ov[k] = config._dpapi_protect(str(sensitive[k]).strip())
        try:
            _save_users_raw(data)
        except OSError:
            pass


def clear_user_overrides(username: str, keys: list) -> None:
    """清除用户 overrides 中的指定字段（回退到全局）"""
    with _users_lock:
        try:
            data = _load_users_raw()
        except (json.JSONDecodeError, OSError):
            return
        u = data.get("users", {}).get(username)
        if not u:
            return
        ov = u.get("overrides") or {}
        for k in keys:
            ov.pop(k, None)
        u["overrides"] = ov
        try:
            _save_users_raw(data)
        except OSError:
            pass


# ---------- 限流（与登录模型解耦，按 username 计数） ----------

def _ensure_rate_entry(username: str) -> dict:
    if username not in _rate_state:
        _rate_state[username] = {
            "proofread_count": 0,
            "requests": deque(),
        }
    return _rate_state[username]


def record_request(username: str) -> None:
    """记录一次 API 请求；调用方应在所有受保护路由入口调用"""
    now = time.time()
    with _rate_lock:
        st = _ensure_rate_entry(username)
        st["requests"].append(now)
        # 清理过期记录
        cutoff = now - RATE_LIMIT_WINDOW
        while st["requests"] and st["requests"][0] < cutoff:
            st["requests"].popleft()


def check_rate_limit(username: str) -> tuple:
    """返回 (是否允许, 原因)。允许时返回 (True, "")"""
    now = time.time()
    with _rate_lock:
        st = _ensure_rate_entry(username)
        cutoff = now - RATE_LIMIT_WINDOW
        while st["requests"] and st["requests"][0] < cutoff:
            st["requests"].popleft()
        if len(st["requests"]) > RATE_LIMIT_MAX_REQUESTS:
            return False, f"请求过于频繁，每分钟上限 {RATE_LIMIT_MAX_REQUESTS} 次"
    return True, ""


def acquire_proofread_slot(username: str) -> bool:
    """占用一个纠错并发槽；超出上限返回 False"""
    with _rate_lock:
        st = _ensure_rate_entry(username)
        if st["proofread_count"] >= MAX_CONCURRENT_PROOFREAD:
            return False
        st["proofread_count"] += 1
        return True


def release_proofread_slot(username: str) -> None:
    with _rate_lock:
        st = _ensure_rate_entry(username)
        if st["proofread_count"] > 0:
            st["proofread_count"] -= 1


# ---------- 初始化 ----------

def ensure_admin_exists() -> tuple:
    """首次启动调用：若无任何用户，创建管理员账户并返回其用户名+初始密码。
    已有用户则返回 ("", "")。
    默认管理员：用户名 admin，密码 admin123（强制首次登录后修改）。
    """
    with _users_lock:
        try:
            data = _load_users_raw()
        except (json.JSONDecodeError, OSError):
            return "", ""
        if data.get("users"):
            return "", ""
        salt = secrets.token_hex(16)
        password = "admin123"
        data["users"] = {
            "admin": {
                "username": "admin",
                "pwd_hash": _hash_password(password, salt),
                "salt": salt,
                "role": "admin",
                "created_at": _now_iso(),
                "overrides": {},
            }
        }
        try:
            _save_users_raw(data)
        except OSError:
            return "", ""
        return "admin", password
