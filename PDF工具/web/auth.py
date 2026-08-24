"""正式身份验证与用户管理。

多用户体系（库级所有权模型）：
- 未登录用户统一视为游客（GUEST），只能看到并读写"公共库"（owner == "guest" 的库）；
- 登录用户看到"公共库 + 自己名下的库"；
- 管理员（role == "admin"）可看到/管理所有库，可把库的所有权转移给其他用户（数据迁移）。

存储位置（在用户登录数据目录 biaoshifu 下，Windows 默认 <用户主目录>/biaoshifu，
跨应用共用同一组账号；见 userdata.py）：
    _users.json    用户表    {"users": {name: {name, pwd_hash, salt, role, created_at}}}
    _sessions.json 会话表    {"sessions": {token: {username, created_at, expires_at}}}

安全要点：
- 密码使用 PBKDF2-HMAC-SHA256（随机盐 + 20 万次迭代）哈希存储，绝不明文落盘；
- 会话 token 使用 secrets.token_urlsafe 随机生成，仅服务端保存（Cookie 中只存 token）；
- 密码比较使用 hmac.compare_digest 防时序攻击。
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from typing import Any, Dict, List, Optional

USERS_FILENAME = "_users.json"
SESSIONS_FILENAME = "_sessions.json"

# 游客身份标识：未登录用户统一视为 guest；owner == "guest" 的库为公共库
GUEST = "guest"

SESSION_TTL_SECONDS = 7 * 24 * 3600  # 会话有效期 7 天
PBKDF2_ITERATIONS = 200_000


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def hash_password(password: str, salt: Optional[bytes] = None) -> Dict[str, str]:
    """对密码做 PBKDF2 哈希，返回 {salt_hex, pwd_hash}。"""
    if salt is None:
        salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return {
        "salt": salt.hex(),
        "pwd_hash": digest.hex(),
    }


def verify_password(password: str, salt_hex: str, pwd_hash: str) -> bool:
    """校验密码，使用恒定时间比较。"""
    try:
        salt = bytes.fromhex(salt_hex)
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
        )
        return hmac.compare_digest(digest.hex(), pwd_hash)
    except (ValueError, TypeError):
        return False


class UserStore:
    """用户与会话存储，线程安全（每次读写都从磁盘加载/保存）。"""

    def __init__(self, base_dir: str):
        self._base_dir = base_dir
        self._users_path = os.path.join(base_dir, USERS_FILENAME)
        self._sessions_path = os.path.join(base_dir, SESSIONS_FILENAME)
        self._lock = threading.Lock()

    # ---------------- 持久化 ----------------

    def _load_users(self) -> Dict[str, Any]:
        try:
            with open(self._users_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "users" in data:
                return data
        except (OSError, json.JSONDecodeError, ValueError):
            pass
        return {"users": {}}

    def _save_users(self, data: Dict[str, Any]) -> None:
        os.makedirs(self._base_dir, exist_ok=True)
        tmp = self._users_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self._users_path)

    def _load_sessions(self) -> Dict[str, Any]:
        try:
            with open(self._sessions_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "sessions" in data:
                return data
        except (OSError, json.JSONDecodeError, ValueError):
            pass
        return {"sessions": {}}

    def _save_sessions(self, data: Dict[str, Any]) -> None:
        os.makedirs(self._base_dir, exist_ok=True)
        tmp = self._sessions_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self._sessions_path)

    # ---------------- 查询 ----------------

    def count_users(self) -> int:
        with self._lock:
            return len(self._load_users().get("users", {}))

    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """按用户名返回用户信息（不含密码字段）。"""
        if not username or username == GUEST:
            return None
        with self._lock:
            users = self._load_users().get("users", {})
            u = users.get(username)
        if not u:
            return None
        return {
            "username": u.get("username", username),
            "role": u.get("role", "user"),
            "created_at": u.get("created_at", ""),
        }

    def list_users(self) -> List[Dict[str, Any]]:
        with self._lock:
            users = self._load_users().get("users", {})
        result = []
        for name, u in sorted(users.items()):
            result.append({
                "username": u.get("username", name),
                "role": u.get("role", "user"),
                "created_at": u.get("created_at", ""),
            })
        return result

    def is_admin(self, username: str) -> bool:
        user = self.get_user(username)
        return bool(user and user.get("role") == "admin")

    def user_exists(self, username: str) -> bool:
        with self._lock:
            return username in self._load_users().get("users", {})

    # ---------------- 注册 / 登录 ----------------

    def register(self, username: str, password: str, role: Optional[str] = None) -> Dict[str, Any]:
        """注册新用户。

        - 首个注册用户自动成为管理员（role="admin"），用于系统初始化；
        - 之后注册的用户默认 role="user"（可通过 role 参数或管理员操作提升）。
        """
        username = (username or "").strip()
        if not username:
            raise ValueError("用户名不能为空")
        if len(username) > 32:
            raise ValueError("用户名最长 32 个字符")
        if not password or len(password) < 6:
            raise ValueError("密码至少 6 位")
        # 用户名只允许字母/数字/下划线/中文字符，避免作为 owner 时产生歧义
        import re
        if not re.fullmatch(r"[\w\u4e00-\u9fff]+", username):
            raise ValueError("用户名只能包含字母、数字、下划线或中文")
        with self._lock:
            users = self._load_users()
            if username in users.get("users", {}):
                raise ValueError(f"用户名已存在: {username}")
            if role is None:
                role = "admin" if not users.get("users") else "user"
            if role not in ("admin", "user"):
                raise ValueError(f"非法角色: {role}")
            salted = hash_password(password)
            users.setdefault("users", {})[username] = {
                "username": username,
                "pwd_hash": salted["pwd_hash"],
                "salt": salted["salt"],
                "role": role,
                "created_at": _now(),
            }
            self._save_users(users)
        return {
            "username": username,
            "role": role,
            "created_at": users["users"][username]["created_at"],
        }

    def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """校验用户名密码，成功返回用户信息，失败返回 None。"""
        username = (username or "").strip()
        if not username or not password:
            return None
        with self._lock:
            users = self._load_users().get("users", {})
            u = users.get(username)
        if not u:
            return None
        if not verify_password(password, u.get("salt", ""), u.get("pwd_hash", "")):
            return None
        return {
            "username": username,
            "role": u.get("role", "user"),
            "created_at": u.get("created_at", ""),
        }

    # ---------------- 会话 ----------------

    def create_session(self, username: str) -> str:
        """为用户创建会话 token 并持久化。"""
        token = secrets.token_urlsafe(32)
        now = int(time.time())
        with self._lock:
            sessions = self._load_sessions()
            # 清理过期会话，避免文件无限膨胀
            sessions["sessions"] = {
                t: s for t, s in sessions.get("sessions", {}).items()
                if int(s.get("expires_at", 0)) > now
            }
            sessions.setdefault("sessions", {})[token] = {
                "username": username,
                "created_at": now,
                "expires_at": now + SESSION_TTL_SECONDS,
            }
            self._save_sessions(sessions)
        return token

    def get_user_by_token(self, token: str) -> Optional[Dict[str, Any]]:
        """按会话 token 解析用户；token 无效或过期返回 None。"""
        if not token:
            return None
        now = int(time.time())
        with self._lock:
            sessions = self._load_sessions().get("sessions", {})
            s = sessions.get(token)
            if not s:
                return None
            if int(s.get("expires_at", 0)) <= now:
                return None
        return self.get_user(s.get("username", ""))

    def logout(self, token: str) -> bool:
        """注销会话，返回是否成功删除。"""
        if not token:
            return False
        with self._lock:
            sessions = self._load_sessions()
            if token in sessions.get("sessions", {}):
                del sessions["sessions"][token]
                self._save_sessions(sessions)
                return True
        return False

    # ---------------- 管理 ----------------

    def remove_user(self, username: str) -> bool:
        """删除用户账号（不影响其名下的库，库将保留但不可见，可先转移所有权）。"""
        if username == GUEST:
            return False
        with self._lock:
            users = self._load_users()
            if username not in users.get("users", {}):
                return False
            del users["users"][username]
            self._save_users(users)
            # 同时清理该用户的所有会话
            sessions = self._load_sessions()
            sessions["sessions"] = {
                t: s for t, s in sessions.get("sessions", {}).items()
                if s.get("username") != username
            }
            self._save_sessions(sessions)
        return True

    def set_role(self, username: str, role: str) -> Optional[Dict[str, Any]]:
        """修改用户角色（admin/user）。"""
        if role not in ("admin", "user"):
            raise ValueError(f"非法角色: {role}")
        with self._lock:
            users = self._load_users()
            u = users.get("users", {}).get(username)
            if not u:
                return None
            u["role"] = role
            self._save_users(users)
        return {
            "username": username,
            "role": role,
            "created_at": u.get("created_at", ""),
        }

    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """修改密码：需验证旧密码，新密码至少 6 位。"""
        if not new_password or len(new_password) < 6:
            raise ValueError("新密码至少 6 位")
        if self.authenticate(username, old_password) is None:
            return False
        salted = hash_password(new_password)
        with self._lock:
            users = self._load_users()
            u = users.get("users", {}).get(username)
            if not u:
                return False
            u["pwd_hash"] = salted["pwd_hash"]
            u["salt"] = salted["salt"]
            self._save_users(users)
        return True

    def admin_reset_password(self, username: str, new_password: str) -> bool:
        """管理员重置密码（不校验旧密码）。"""
        if not new_password or len(new_password) < 6:
            raise ValueError("新密码至少 6 位")
        salted = hash_password(new_password)
        with self._lock:
            users = self._load_users()
            u = users.get("users", {}).get(username)
            if not u:
                return False
            u["pwd_hash"] = salted["pwd_hash"]
            u["salt"] = salted["salt"]
            self._save_users(users)
        return True
