# -*- coding: utf-8 -*-
"""共用用户系统数据层：用户名+密码注册制、会话管理、管理员用户管理。

数据存储改到**跨应用共享目录** <用户主目录>/biaoshifu（可用环境变量 BIAOSHIFU_DIR 覆盖），
与"全文检索系统"（search 项目）共用同一组账号密码登录。

存储文件（与 search 项目 auth.py 格式完全一致，保证跨应用互通）：
    _users.json    用户表   {"users": {username: {username, pwd_hash, salt, role, created_at, ...}}}
    _sessions.json 会话表   {"sessions": {token: {username, created_at, expires_at}}}

密码哈希：PBKDF2-HMAC-SHA256，salt 与 pwd_hash 均为十六进制（与 search 项目同方案），
绝不明文落盘；会话 token 使用 secrets.token_urlsafe 随机生成，仅服务端保存。

- 首个注册的用户自动成为管理员（role="admin"），之后注册的都是普通用户
- 会话 token 持久化到 _sessions.json：服务重启后登录态不丢失，且与 search 项目互通
- 管理员可对普通用户执行 启用/禁用/删除/重置密码；不能对自己操作，且至少保留一个管理员
- 用户身份以用户名为唯一标识（user_id == username），与 search 项目的库级所有权模型一致

本应用自己的内容数据（上传/输出/任务）保存在本应用 data/<user_id>/ 下，
与共享账号数据分离，互不干扰。
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sys
import threading
import time


# ----------------------------------------------------------------------
# 数据目录：账号数据在共享 biaoshifu，内容数据在本应用 data/
# ----------------------------------------------------------------------

def _auth_base_dir() -> str:
    """解析跨应用共享登录数据目录（<用户主目录>/biaoshifu，可被环境变量覆盖）。

    与 search 项目的 userdata.auth_base_dir() 行为一致：
    优先级：环境变量 BIAOSHIFU_DIR > <用户主目录>/biaoshifu。
    """
    env = os.environ.get("BIAOSHIFU_DIR", "").strip()
    if env:
        d = os.path.abspath(os.path.expanduser(env))
    else:
        d = os.path.join(os.path.expanduser("~"), "biaoshifu")
    os.makedirs(d, exist_ok=True)
    return d


AUTH_DIR = _auth_base_dir()
USERS_PATH = os.path.join(AUTH_DIR, "_users.json")
SESSIONS_PATH = os.path.join(AUTH_DIR, "_sessions.json")

# 本应用自己的内容数据根目录（与共享账号数据分离）
if getattr(sys, "frozen", False):
    _BASE_DIR = os.path.dirname(sys.executable)
else:
    # users.py 位于 app/auth/，向上三级到项目根
    _BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(_BASE_DIR, "data")

PBKDF2_ITERATIONS = 200_000
SESSION_TTL = 7 * 86400          # 会话有效期（秒）
MIN_PASSWORD_LEN = 6
# 用户名：2-32 字符，仅字母/数字/下划线/中文（与共享账号体系一致，避免作为 owner 产生歧义）
USERNAME_RE = re.compile(r"^[\w\u4e00-\u9fff]{2,32}$")

_lock = threading.Lock()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ----------------------------------------------------------------------
# 持久化（与 search 项目 auth.py 的存储格式一致）
# ----------------------------------------------------------------------

def _load_users() -> dict:
    try:
        with open(USERS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "users" in data:
            return data
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return {"users": {}}


def _save_users(data: dict) -> None:
    os.makedirs(AUTH_DIR, exist_ok=True)
    tmp = USERS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, USERS_PATH)


def _load_sessions() -> dict:
    try:
        with open(SESSIONS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "sessions" in data:
            return data
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return {"sessions": {}}


def _save_sessions(data: dict) -> None:
    os.makedirs(AUTH_DIR, exist_ok=True)
    tmp = SESSIONS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, SESSIONS_PATH)


# ----------------------------------------------------------------------
# 密码哈希（PBKDF2-SHA256，salt/pwd_hash 均为 hex，与 search 项目互通）
# ----------------------------------------------------------------------

def _hash_password(password: str) -> dict:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return {"salt": salt.hex(), "pwd_hash": dk.hex()}


def _verify_password(password: str, salt_hex: str, pwd_hash: str) -> bool:
    try:
        salt = bytes.fromhex(salt_hex)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
        return hmac.compare_digest(dk.hex(), pwd_hash)
    except (ValueError, TypeError):
        return False


# ----------------------------------------------------------------------
# 用户信息转换
# ----------------------------------------------------------------------

def _full_user(username: str, u: dict) -> dict:
    """完整用户信息（含本应用需要的 user_id/is_admin/is_active）。"""
    return {
        "user_id": username,
        "username": u.get("username", username),
        "role": u.get("role", "user"),
        "is_admin": u.get("role") == "admin",
        "is_active": bool(u.get("is_active", True)),
        "created_at": u.get("created_at", ""),
    }


def _public_user(username: str, u: dict) -> dict:
    """对外暴露的用户信息（不含任何密码字段）。"""
    return {
        "user_id": username,
        "username": u.get("username", username),
        "is_admin": u.get("role") == "admin",
        "is_active": bool(u.get("is_active", True)),
        "created_at": u.get("created_at", ""),
        "last_login_at": u.get("last_login_at", ""),
    }


# ----------------------------------------------------------------------
# 注册 / 登录 / 登出
# ----------------------------------------------------------------------

def register(username: str, password: str) -> tuple:
    """注册新用户。首个注册的用户自动成为管理员。

    返回 (user_public_dict, error)；error 非 None 表示注册失败。
    """
    username = (username or "").strip()
    if not USERNAME_RE.match(username):
        return None, "用户名需为 2-32 个字符，且只能包含字母、数字、下划线或中文"
    if len(password) < MIN_PASSWORD_LEN:
        return None, f"密码长度至少 {MIN_PASSWORD_LEN} 位"
    with _lock:
        users = _load_users()
        if username in users.get("users", {}):
            return None, "用户名已存在，请换一个"
        role = "admin" if not users.get("users") else "user"   # 首个注册者 = 管理员
        salted = _hash_password(password)
        users.setdefault("users", {})[username] = {
            "username": username,
            "pwd_hash": salted["pwd_hash"],
            "salt": salted["salt"],
            "role": role,
            "is_active": True,
            "created_at": _now(),
            "last_login_at": "",
        }
        _save_users(users)
        return _public_user(username, users["users"][username]), None


def login(username: str, password: str) -> tuple:
    """登录。返回 (token, user_public_dict)；失败返回 (None, error)。"""
    username = (username or "").strip()
    if not username or not password:
        return None, "用户名或密码错误"
    with _lock:
        users = _load_users()
        u = users.get("users", {}).get(username)
        if not u:
            return None, "用户名或密码错误"
        if not _verify_password(password, u.get("salt", ""), u.get("pwd_hash", "")):
            return None, "用户名或密码错误"
        if not u.get("is_active", True):
            return None, "账号已被禁用，请联系管理员"
        token = secrets.token_urlsafe(32)
        now = int(time.time())
        sessions = _load_sessions()
        # 清理过期会话，避免文件无限膨胀
        sessions["sessions"] = {
            t: s for t, s in sessions.get("sessions", {}).items()
            if int(s.get("expires_at", 0)) > now
        }
        sessions.setdefault("sessions", {})[token] = {
            "username": username,
            "created_at": now,
            "expires_at": now + SESSION_TTL,
        }
        _save_sessions(sessions)
        u["last_login_at"] = _now()
        _save_users(users)
        return token, _public_user(username, u)
    return None, "用户名或密码错误"


def logout(token: str) -> None:
    if not token:
        return
    with _lock:
        sessions = _load_sessions()
        if token in sessions.get("sessions", {}):
            del sessions["sessions"][token]
            _save_sessions(sessions)


def authenticate(token: str) -> dict:
    """通过会话 token 找用户；无效或过期返回 None。

    token 与 search 项目共用 _sessions.json，因此 search 项目签发的会话在此同样有效。
    """
    if not token:
        return None
    now = int(time.time())
    with _lock:
        sessions = _load_sessions()
        s = sessions.get("sessions", {}).get(token)
        if not s:
            return None
        if int(s.get("expires_at", 0)) <= now:
            del sessions["sessions"][token]
            _save_sessions(sessions)
            return None
        username = s.get("username", "")
        users = _load_users()
        u = users.get("users", {}).get(username)
    if not u:
        return None
    return _full_user(username, u)


def get_user(user_id: str) -> dict:
    """按用户标识（即用户名）取用户。"""
    with _lock:
        users = _load_users()
        u = users.get("users", {}).get(user_id)
    if not u:
        return None
    return _full_user(user_id, u)


def list_users() -> list:
    with _lock:
        users = _load_users()
    result = []
    for name, u in sorted(users.get("users", {}).items()):
        result.append(_public_user(name, u))
    return result


def _count_admins(data: dict) -> int:
    return sum(
        1 for u in data.get("users", {}).values()
        if u.get("role") == "admin" and u.get("is_active", True)
    )


# ----------------------------------------------------------------------
# 管理员：用户管理
# ----------------------------------------------------------------------

def set_active(user_id: str, active: bool, operator_id: str) -> tuple:
    """启用/禁用用户。返回 (ok, error)。不能操作自己；不能禁用唯一管理员。"""
    if user_id == operator_id:
        return False, "不能对自己执行此操作"
    with _lock:
        users = _load_users()
        u = users.get("users", {}).get(user_id)
        if not u:
            return False, "用户不存在"
        if not active and u.get("role") == "admin" and _count_admins(users) <= 1:
            return False, "不能禁用唯一的管理员"
        u["is_active"] = bool(active)
        _save_users(users)
    return True, None


def remove_user(user_id: str, operator_id: str) -> tuple:
    """删除用户（含本应用内容数据目录）。返回 (ok, error)。不能删除自己。"""
    if user_id == operator_id:
        return False, "不能删除自己"
    with _lock:
        users = _load_users()
        if user_id not in users.get("users", {}):
            return False, "用户不存在"
        del users["users"][user_id]
        _save_users(users)
        # 同时清理该用户的所有会话
        sessions = _load_sessions()
        sessions["sessions"] = {
            t: s for t, s in sessions.get("sessions", {}).items()
            if s.get("username") != user_id
        }
        _save_sessions(sessions)
    # 删除本应用内容数据目录（上传/输出/任务）
    import shutil
    d = os.path.join(DATA_DIR, user_id)
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
    return True, None


def reset_password(user_id: str, new_password: str) -> tuple:
    """管理员重置密码。返回 (ok, error)。"""
    if len(new_password) < MIN_PASSWORD_LEN:
        return False, f"新密码长度至少 {MIN_PASSWORD_LEN} 位"
    with _lock:
        users = _load_users()
        u = users.get("users", {}).get(user_id)
        if not u:
            return False, "用户不存在"
        salted = _hash_password(new_password)
        u["pwd_hash"] = salted["pwd_hash"]
        u["salt"] = salted["salt"]
        _save_users(users)
    return True, None


def user_dir(user_id: str) -> str:
    """该用户在本应用的内容数据目录（自动创建，上传/输出/任务互不可见）。"""
    d = os.path.join(DATA_DIR, user_id)
    os.makedirs(os.path.join(d, "uploads"), exist_ok=True)
    os.makedirs(os.path.join(d, "outputs"), exist_ok=True)
    return d


# ----------------------------------------------------------------------
# 每用户并发任务数限制（讯飞模式：防单用户耗尽云端配额）
# ----------------------------------------------------------------------

# 单用户同时进行的识别任务数上限（配合页级并发：3 任务 × 5 页 = 15，低于讯飞 20 并发上限）
MAX_CONCURRENT_TASKS = 3

# user_id -> 正在进行的任务数（内存态，不落盘）
_concurrent: dict = {}


def acquire_slot(user_id: str, limit: int = None) -> bool:
    """尝试占用一个并发任务名额。超限返回 False。

    limit 可由调用方按配置覆盖（默认 MAX_CONCURRENT_TASKS）。
    """
    cap = limit if limit and limit > 0 else MAX_CONCURRENT_TASKS
    with _lock:
        n = _concurrent.get(user_id, 0)
        if n >= cap:
            return False
        _concurrent[user_id] = n + 1
        return True


def release_slot(user_id: str) -> None:
    with _lock:
        n = _concurrent.get(user_id, 0)
        if n > 1:
            _concurrent[user_id] = n - 1
        else:
            _concurrent.pop(user_id, None)
