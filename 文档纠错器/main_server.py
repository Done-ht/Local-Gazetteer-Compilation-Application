# -*- coding: utf-8 -*-
"""DocProof 服务器入口：HTTP 服务 + 浏览器客户端分发。

端口设置（按优先级）：
    1. 命令行参数 --port / -p
    2. 环境变量 DOCPROOF_PORT
    3. 启动前控制台交互输入（默认 8000 回车确认）

监听地址：默认 --host 0.0.0.0（局域网可访问）；仅本机用 --host 127.0.0.1

登录模型：用户名+密码登录换取会话 token（7 天有效）；后续请求以
Authorization: Bearer <session_token> 携带。改密码或被删除会立即吊销会话。

路由（除标注外均需 Authorization: Bearer <session_token>）：
    GET  /                  返回浏览器客户端 _index.html（公开）
    GET  /api/health        健康检查：模型、是否已配置密钥（公开）
    POST /api/login         用户名+密码登录，返回 session_token + 用户信息（公开）
    POST /api/register      自助注册：创建普通用户（非管理员），成功后自动登录（公开）
    POST /api/logout        主动登出，删除当前会话（携带 token 即可）
    GET  /api/me            返回当前用户信息
    GET  /api/config        返回当前用户合并后的非敏感配置（密钥不返回）
    GET  /api/session/<id>  查询会话状态与页文本（校验归属）
    POST /api/config        保存非敏感配置（管理员写全局，普通用户写自己的 overrides）
    POST /api/credentials   设置/更新敏感凭据（同上分流）
    POST /api/upload        multipart 文件上传，返回 session_id + 页文本
    POST /api/proofread     NDJSON 流式纠错（按用户限流）
    POST /api/export        应用修正后导出文件下载（校验归属）
    POST /api/session/<id>/preview  应用修正后返回页文本预览（校验归属）
    GET  /api/users         列出所有用户（仅管理员）
    POST /api/users         新增用户：username + password + is_admin（仅管理员）
    DELETE /api/users/<username>  删除用户（仅管理员）
    POST /api/users/<username>/password  重置用户密码（仅管理员）
    POST /api/users/<username>/admin    切换管理员状态（仅管理员）
    POST /api/users/<username>/rename   改用户名（仅管理员）

依赖：仅 Python 标准库（http.server / email.parser 等）+ 已有 core/ 模块 + users.py。
"""
import argparse
import io
import ipaddress
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
import uuid
from email.parser import BytesParser
from email.policy import default as default_policy

# Windows 控制台默认 GBK 编码，遇到 ⚠ 等 Unicode 符号会 UnicodeEncodeError 崩溃
# 强制 stdout/stderr 用 UTF-8 + 行缓冲（PyInstaller 重定向到文件时默认全缓冲，会导致看不到输出）
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except (AttributeError, io.UnsupportedOperation):
        pass  # 某些环境（如重定向到文件）可能不支持 reconfigure
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, unquote
from logging.handlers import RotatingFileHandler

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import config
import users
from core import docload, exporter, ocr_xfyun, proofread
from core.deepseek import DeepSeekClient, DeepSeekError
from core.models import ErrorItem, Page, TokenUsage

# ---------- 日志：只落文件 + 按大小轮转，控制台保持静默 ----------
# 日志目录与配置文件同处 %APPDATA%\DocProof\logs，避免 EXE 所在目录只读。
_LOG_DIR = os.path.join(config._user_config_dir(), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_PATH = os.path.join(_LOG_DIR, "server.log")

# 单文件 2MB 上限，保留 5 份历史备份，总量上限约 12MB，不会持续膨胀。
_file_handler = RotatingFileHandler(
    _LOG_PATH, maxBytes=2 * 1024 * 1024, backupCount=5,
    encoding="utf-8",
)
_file_handler.setFormatter(logging.Formatter(
    "[%(asctime)s] %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
))
# 根 logger 只挂文件 handler，不向 stderr/stdout 输出，控制台静默
_root = logging.getLogger()
_root.setLevel(logging.INFO)
# 清掉 PyInstaller / 第三方库可能预设的 StreamHandler，确保真正静默
for _h in list(_root.handlers):
    _root.removeHandler(_h)
_root.addHandler(_file_handler)
logger = logging.getLogger("docproof.server")

# 客户端中途断连属于预期行为（关页/刷新/取消），单独收口避免堆栈噪音。
# 这三类对应：BrokenPipeError（写已关闭 socket）、ConnectionResetError（对端 RST）、
# ConnectionAbortedError（Windows WinError 10053，本机软件中止连接）。
DISCONNECTED_ERRORS = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)


class _ClientDisconnected(Exception):
    """客户端已断连的哨兵异常：由 emit 抛出，外层据此跳过后续写入与错误回传。"""

# 浏览器客户端页面（与 main_server.py 同目录）
INDEX_HTML_PATH = os.path.join(PROJECT_ROOT, "_index.html")

# 会话：session_id -> dict(username, doc_ctx, pages, file_type, source_path, created_at, last_access)
SESSIONS: dict = {}
SESSION_TTL = 2 * 3600  # 2 小时未访问自动清理
_session_lock = threading.Lock()


# ---------- SSRF 防护：校验 llm_base_url ----------

def _is_private_ip(host: str) -> bool:
    """判断 host 是否为内网/回环地址"""
    try:
        ip = ipaddress.ip_address(host)
        return (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_multicast or ip.is_reserved)
    except ValueError:
        # 域名：localhost 系列视为内网
        return host.lower() in ("localhost", "ip6-localhost", "ip6-loopback")


def validate_llm_base_url(url: str) -> tuple:
    """校验 base_url 是否安全；返回 (是否合法, 原因)。
    规则：
    - 必须是 http(s):// 开头
    - host 不能是内网/回环（防止服务器把带密钥的请求发到内网）
    - 特例：127.0.0.1 / localhost 在「本机调试」场景下允许，但仅当未配置 API Key 时
    """
    if not url:
        return False, "base_url 为空"
    if not url.startswith(("http://", "https://")):
        return False, "base_url 必须以 http:// 或 https:// 开头"
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if not host:
        return False, "base_url 无法解析 host"
    # 公网域名直接放行
    if not _is_private_ip(host):
        return True, ""
    # 内网/回环：仅当未配置 API Key 时允许（开发调试场景）
    cfg = config.load()
    if not cfg.get("llm_api_key"):
        return True, ""  # 本机调试，无密钥可泄露
    return False, f"已配置 API Key 时，base_url 不允许指向内网地址 {host}"


# ---------- 会话管理 ----------

def _purge_expired_sessions(now: float):
    """清理过期会话；调用方需持有 _session_lock"""
    expired = [sid for sid, s in SESSIONS.items()
               if now - s["last_access"] > SESSION_TTL]
    for sid in expired:
        s = SESSIONS.pop(sid, None)
        if s:
            _cleanup_session_files(s)


def _cleanup_session_files(session: dict):
    """删除会话关联的临时文件"""
    p = session.get("source_path")
    if p and p.startswith(tempfile.gettempdir()) and os.path.exists(p):
        try:
            os.remove(p)
        except OSError:
            pass


def _get_session(session_id: str) -> dict:
    with _session_lock:
        s = SESSIONS.get(session_id)
        if s:
            s["last_access"] = time.time()
        _purge_expired_sessions(time.time())
        return s


# ---------- multipart/form-data 解析 ----------

def parse_multipart(body: bytes, boundary: bytes):
    """解析 multipart/form-data，返回 {field_name: (filename or None, value_bytes)}"""
    # 用 email 模块解析 MIME multipart
    header = b"Content-Type: multipart/form-data; boundary=" + boundary + b"\r\n\r\n"
    msg = BytesParser(policy=default_policy).parsebytes(header + body)
    result = {}
    if not msg.is_multipart():
        return result
    for part in msg.get_payload():
        cd = part.get("Content-Disposition", "")
        name_match = re.search(r'name="([^"]*)"', cd)
        if not name_match:
            continue
        name = name_match.group(1)
        filename_match = re.search(r'filename="([^"]*)"', cd)
        filename = filename_match.group(1) if filename_match else None
        # part.get_payload(decode=True) 自动处理 base64/quoted-printable；此处通常就是原始字节
        value = part.get_payload(decode=True) or b""
        result[name] = (filename, value)
    return result


def _read_request_body(handler: BaseHTTPRequestHandler, max_bytes: int = 200 * 1024 * 1024):
    """读取请求体，限制最大 200MB"""
    length = int(handler.headers.get("Content-Length", 0))
    if length > max_bytes:
        raise ValueError(f"请求体过大: {length} bytes (上限 {max_bytes})")
    return handler.rfile.read(length) if length > 0 else b""


# ---------- 路由处理 ----------

def _send_json(handler: BaseHTTPRequestHandler, status: int, payload: dict, headers: dict = None):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    for k, v in (headers or {}).items():
        handler.send_header(k, v)
    handler.end_headers()
    handler.wfile.write(body)


def _send_error_json(handler: BaseHTTPRequestHandler, status: int, msg: str):
    _send_json(handler, status, {"error": msg})


def _send_ndjson_line(handler: BaseHTTPRequestHandler, payload: dict):
    """写一行 NDJSON；调用前需已开始流式响应"""
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    handler.wfile.write(line.encode("utf-8"))
    handler.wfile.flush()


# ---------- HTTP Handler ----------

def _public_user_info(user: dict) -> dict:
    """返回用户公开信息（不含 pwd_hash/salt）。
    兼容前端字段：username 既是登录账号也是展示名；is_admin 由 role 派生。
    """
    return {
        "username": user.get("username", ""),
        "display_name": user.get("username", ""),  # 前端历史用 display_name，这里同步为 username
        "is_admin": (user.get("role") == "admin") or bool(user.get("is_admin")),
        "role": user.get("role", "user"),
        "created_at": user.get("created_at", 0),
    }


class DocProofHandler(BaseHTTPRequestHandler):
    server_version = "DocProofServer/1.3"
    # 关闭默认日志（BaseHTTPRequestHandler.log_message 噪音大）
    def log_message(self, fmt, *args):
        logger.info("%s - %s", self.address_string(), fmt % args)

    # ---------- 鉴权 ----------
    # 公开路由：无需 token 即可访问
    PUBLIC_PATHS = {"/", "/index.html", "/favicon.ico", "/api/health", "/api/login"}

    def _auth(self):
        """从 Authorization 头提取并校验会话 token；返回 (user_dict, None) 或 (None, error_msg)。
        支持两种格式：Authorization: Bearer <session_token> 或 ?token=<session_token>（便于浏览器下载链接）
        """
        # 优先从 Header 取
        auth = self.headers.get("Authorization", "")
        token = ""
        if auth.startswith("Bearer "):
            token = auth[7:].strip()
        if not token:
            # 兜底从 query string 取（导出下载链接无法带 Header 时用）
            qs = urlparse(self.path).query
            if qs.startswith("token="):
                token = qs[6:]
        if not token:
            return None, "未登录或会话已过期（请重新登录）"
        user = users.authenticate(token)
        if not user:
            return None, "会话已过期或已被吊销，请重新登录"
        return user, None

    def _require_user(self):
        """校验会话 token 并记录限流；失败时已自动返回 401/429，成功返回 user_dict"""
        user, err = self._auth()
        if not user:
            _send_error_json(self, 401, err)
            return None
        # 限流：所有受保护路由统一计数（按 username）
        ok, reason = users.check_rate_limit(user["username"])
        if not ok:
            _send_error_json(self, 429, reason)
            return None
        users.record_request(user["username"])
        return user

    def _require_admin(self):
        """校验 token 且要求是管理员；失败自动返回错误"""
        user = self._require_user()
        if not user:
            return None
        if not user.get("is_admin"):
            _send_error_json(self, 403, "需要管理员权限")
            return None
        return user

    # ----- GET -----
    def do_GET(self):
        path = urlparse(self.path).path
        # 公开路由
        if path in ("/", "/index.html"):
            return self._serve_index()
        if path == "/favicon.ico":
            return self._send_favicon()
        if path == "/api/health":
            return self._handle_health()
        if path == "/api/login":
            return self._handle_login()
        # 以下均需鉴权
        if path == "/api/me":
            user = self._require_user()
            return self._handle_me(user) if user else None
        if path == "/api/config":
            user = self._require_user()
            return self._handle_get_config(user) if user else None
        if path == "/api/users":
            if self._require_admin():
                return self._handle_list_users()
            return None
        # /api/session/<id>：浏览器刷新后查询会话是否还活着 + 拉回页文本
        m = re.match(r"^/api/session/([^/]+)$", path)
        if m:
            user = self._require_user()
            return self._handle_get_session(m.group(1), user) if user else None
        _send_error_json(self, 404, f"未知的 GET 路径: {path}")

    # ----- POST -----
    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/login":
            return self._handle_login()
        if path == "/api/register":
            return self._handle_register()
        if path == "/api/logout":
            return self._handle_logout()
        if path == "/api/upload":
            user = self._require_user()
            return self._handle_upload(user) if user else None
        if path == "/api/proofread":
            user = self._require_user()
            return self._handle_proofread(user) if user else None
        if path == "/api/export":
            user = self._require_user()
            return self._handle_export(user) if user else None
        if path == "/api/credentials":
            user = self._require_user()
            return self._handle_set_credentials(user) if user else None
        if path == "/api/config":
            user = self._require_user()
            return self._handle_set_config(user) if user else None
        if path == "/api/users":
            if self._require_admin():
                return self._handle_create_user()
            return None
        # /api/session/<id>/preview
        m = re.match(r"^/api/session/([^/]+)/preview$", path)
        if m:
            user = self._require_user()
            return self._handle_preview(m.group(1), user) if user else None
        # /api/users/<username>/(password|admin|rename)
        m = re.match(r"^/api/users/([^/]+)/(password|admin|rename)$", path)
        if m:
            if self._require_admin():
                return self._handle_user_action(unquote(m.group(1)), m.group(2))
            return None
        _send_error_json(self, 404, f"未知的 POST 路径: {path}")

    # ----- DELETE -----
    def do_DELETE(self):
        path = urlparse(self.path).path
        m = re.match(r"^/api/users/([^/]+)$", path)
        if m:
            if self._require_admin():
                return self._handle_delete_user(unquote(m.group(1)))
            return None
        _send_error_json(self, 404, f"未知的 DELETE 路径: {path}")

    # ---------- 静态资源 ----------

    def _serve_index(self):
        if not os.path.exists(INDEX_HTML_PATH):
            _send_error_json(self, 500, f"客户端页面不存在: {INDEX_HTML_PATH}")
            return
        with open(INDEX_HTML_PATH, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _send_favicon(self):
        # 简单占位：返回 204 No Content，避免浏览器反复请求
        self.send_response(204)
        self.end_headers()

    # ---------- /api/health ----------

    def _handle_health(self):
        cfg = config.load()
        _send_json(self, 200, {
            "status": "ok",
            "model": cfg.get("llm_model", ""),
            "provider": cfg.get("llm_provider", ""),
            "llm_configured": bool(cfg.get("llm_api_key")),
            "ocr_configured": bool(cfg.get("xf_appid") and cfg.get("xf_api_key")
                                   and cfg.get("xf_api_secret")),
            "auth_enabled": True,  # 前端据此决定是否显示登录页
        })

    # ---------- /api/login (POST) ----------

    def _handle_login(self):
        """用户名+密码登录；成功返回 {session_token, user}。
        session_token 由前端保存，后续请求以 Authorization: Bearer <session_token> 携带。
        会话有效期 7 天；改密码或被删除会立即吊销。
        """
        data, err = _read_json_body(self)
        if err:
            _send_error_json(self, 400, err)
            return
        username = (data.get("username") or "").strip()
        password = data.get("password") or ""
        if not username or not password:
            _send_error_json(self, 400, "请输入用户名和密码")
            return
        token, user = users.login(username, password)
        if not token:
            # 统一返回 401，避免泄露用户名是否存在
            _send_error_json(self, 401, "用户名或密码错误")
            return
        _send_json(self, 200, {
            "session_token": token,
            "user": _public_user_info(user),
        })

    # ---------- /api/register (POST) ----------

    def _handle_register(self):
        """公开注册：创建普通用户（非管理员），无需管理员审批。
        成功后自动登录并返回 {session_token, user}，免去二次输入。
        """
        data, err = _read_json_body(self)
        if err:
            _send_error_json(self, 400, err)
            return
        username = (data.get("username") or "").strip()
        password = data.get("password") or ""
        ok, reason = users.register_user(username, password)
        if not ok:
            _send_error_json(self, 400, reason)
            return
        # 注册成功后直接登录，返回会话 token
        token, user = users.login(username, password)
        _send_json(self, 201, {
            "session_token": token,
            "user": _public_user_info(user),
        })

    # ---------- /api/logout (POST) ----------

    def _handle_logout(self):
        """主动登出：删除当前会话 token。即使 token 已过期也返回 200。"""
        # 从 Header 或 query 取 token（与 _auth 一致），不强制鉴权失败返回 401
        auth = self.headers.get("Authorization", "")
        token = auth[7:].strip() if auth.startswith("Bearer ") else ""
        if not token:
            qs = urlparse(self.path).query
            if qs.startswith("token="):
                token = qs[6:]
        if token:
            users.logout(token)
        _send_json(self, 200, {"status": "ok"})

    # ---------- /api/me (GET) ----------

    def _handle_me(self, user: dict):
        _send_json(self, 200, _public_user_info(user))

    # ---------- /api/config (GET) ----------

    def _handle_get_config(self, user: dict):
        """返回当前用户合并后的非敏感配置：模型、规则、复核、token 限额；密钥不返回"""
        global_cfg = config.load()
        cfg = users.resolve_user_config(user, global_cfg)
        # 标注哪些字段是用户自己的覆盖（前端可据此显示"恢复全局"按钮）
        overrides = user.get("overrides") or {}
        safe = {
            "llm_provider": cfg.get("llm_provider", "deepseek"),
            "llm_base_url": cfg.get("llm_base_url", ""),
            "llm_model": cfg.get("llm_model", ""),
            "token_limit": cfg.get("token_limit", 1000000),
            "enable_review": cfg.get("enable_review", True),
            "review_context_chars": cfg.get("review_context_chars", 800),
            "rule_switches": cfg.get("rule_switches", {}),
            "custom_rules": cfg.get("custom_rules", []),
            "provider_presets": config.PROVIDER_PRESETS,
            "preset_switches": config.PRESET_SWITCHES,
            "llm_configured": bool(cfg.get("llm_api_key")),
            "ocr_configured": bool(cfg.get("xf_appid") and cfg.get("xf_api_key")
                                   and cfg.get("xf_api_secret")),
            "is_admin": bool(user.get("is_admin")),
            "overridden_fields": sorted(overrides.keys()),
        }
        _send_json(self, 200, safe)

    # ---------- /api/session/<id> (GET) ----------

    def _handle_get_session(self, session_id: str, user: dict):
        """查询会话是否还在 + 返回页文本，供浏览器刷新后恢复状态。
        会话不存在或非本人会话均返回 404，避免泄露会话存在性。
        """
        s = _get_session(session_id)
        if not s or s.get("username") != user["username"]:
            _send_error_json(self, 404, "会话不存在或已过期")
            return
        _send_json(self, 200, {
            "session_id": session_id,
            "file_type": s["file_type"],
            "original_filename": s.get("original_filename", ""),
            "has_tables": s["doc_ctx"].has_tables if s.get("doc_ctx") else False,
            "page_count": len(s["pages"]),
            "pages": [{"page_num": p.page_num, "text": p.text} for p in s["pages"]],
            "created_at": s.get("created_at", 0),
        })

    # ---------- /api/config (POST) ----------

    def _handle_set_config(self, user: dict):
        """保存非敏感配置。
        - 管理员：写全局 config（影响所有未覆盖该字段的用户）
        - 普通用户：写自己的 overrides
        每次都校验 llm_base_url 防 SSRF。
        """
        try:
            body = _read_request_body(self)
            data = json.loads(body.decode("utf-8")) if body else {}
        except (json.JSONDecodeError, ValueError) as e:
            _send_error_json(self, 400, f"请求体解析失败: {e}")
            return

        # 校验 base_url（若提交了）
        if "llm_base_url" in data and data["llm_base_url"]:
            ok, reason = validate_llm_base_url(data["llm_base_url"])
            if not ok:
                _send_error_json(self, 400, f"base_url 不合法: {reason}")
                return

        if user.get("is_admin"):
            # 管理员写全局
            cfg = config.load()
            for k in ("llm_provider", "llm_base_url", "llm_model", "token_limit",
                      "enable_review", "review_context_chars"):
                if k in data:
                    cfg[k] = data[k]
            if "rule_switches" in data:
                cfg["rule_switches"] = {k: bool(v) for k, v in data["rule_switches"].items()}
            if "custom_rules" in data:
                cfg["custom_rules"] = [r for r in data["custom_rules"]
                                       if isinstance(r, str) and r.strip()]
            config.save(cfg)
            _send_json(self, 200, {"status": "ok", "scope": "global",
                                   "config_path": config.CONFIG_PATH})
        else:
            # 普通用户写自己的 overrides
            non_sensitive = {}
            for k in ("llm_provider", "llm_base_url", "llm_model", "token_limit",
                      "enable_review", "review_context_chars"):
                if k in data:
                    non_sensitive[k] = data[k]
            if "rule_switches" in data:
                non_sensitive["rule_switches"] = {k: bool(v) for k, v in data["rule_switches"].items()}
            if "custom_rules" in data:
                non_sensitive["custom_rules"] = [r for r in data["custom_rules"]
                                                 if isinstance(r, str) and r.strip()]
            users.set_user_overrides(user["username"], non_sensitive, {})
            _send_json(self, 200, {"status": "ok", "scope": "user_overrides"})

    # ---------- /api/credentials (POST) ----------

    def _handle_set_credentials(self, user: dict):
        """更新敏感凭据（llm_api_key / xf_*）。
        - 管理员：写全局（DPAPI 加密落盘）
        - 普通用户：写自己的 overrides（明文，因 token 已是访问凭据）
        """
        try:
            body = _read_request_body(self)
            data = json.loads(body.decode("utf-8")) if body else {}
        except (json.JSONDecodeError, ValueError) as e:
            _send_error_json(self, 400, f"请求体解析失败: {e}")
            return

        sensitive = {}
        for k in ("llm_api_key", "xf_appid", "xf_api_key", "xf_api_secret"):
            if k in data and data[k]:
                sensitive[k] = str(data[k]).strip()
        if not sensitive:
            _send_error_json(self, 400, "未提供任何凭据字段")
            return

        if user.get("is_admin"):
            cfg = config.load()
            for k, v in sensitive.items():
                cfg[k] = v
            config.save(cfg)
            _send_json(self, 200, {
                "status": "ok", "scope": "global",
                "updated": list(sensitive.keys()),
                "config_path": config.CONFIG_PATH,
                "encrypted": True,
            })
        else:
            users.set_user_overrides(user["username"], {}, sensitive)
            _send_json(self, 200, {
                "status": "ok", "scope": "user_overrides",
                "updated": list(sensitive.keys()),
            })

    # ---------- /api/upload ----------

    def _handle_upload(self, user: dict):
        ct = self.headers.get("Content-Type", "")
        m = re.search(r"boundary=(.+)", ct)
        if not m:
            _send_error_json(self, 400, "缺少 multipart boundary")
            return
        boundary = m.group(1).strip().strip('"').encode("utf-8")
        try:
            body = _read_request_body(self)
            fields = parse_multipart(body, boundary)
        except ValueError as e:
            _send_error_json(self, 400, str(e))
            return

        if "file" not in fields:
            _send_error_json(self, 400, "未上传 file 字段")
            return
        filename, file_bytes = fields["file"]
        if not filename:
            _send_error_json(self, 400, "文件名为空")
            return

        # 保存到临时文件（docload 需要 path；docx 导出也需要原文件）
        session_id = uuid.uuid4().hex[:16]
        suffix = os.path.splitext(filename)[1] or ".bin"
        tmp_path = os.path.join(tempfile.gettempdir(), f"docproof_{session_id}{suffix}")
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)

        # 加载文档（用当前用户合并后的配置判断 OCR 是否可用）
        try:
            global_cfg = config.load()
            cfg = users.resolve_user_config(user, global_cfg)
            needs = docload.needs_ocr(tmp_path)
            if needs and not (cfg.get("xf_appid") and cfg.get("xf_api_key")
                              and cfg.get("xf_api_secret")):
                _send_error_json(self, 400, "该文件需要 OCR，但服务器未配置讯飞密钥")
                return
            pages, ctx = docload.load_document(tmp_path)
            if ctx.needs_ocr:
                pages = self._run_ocr(tmp_path, cfg, ctx, file_bytes)
        except Exception as e:
            logger.exception("文档加载失败")
            _send_error_json(self, 500, f"文档加载失败: {e}")
            return

        # 注册会话——绑定 username 实现归属隔离
        with _session_lock:
            _purge_expired_sessions(time.time())
            SESSIONS[session_id] = {
                "username": user["username"],
                "doc_ctx": ctx,
                "pages": pages,
                "file_type": ctx.file_type,
                "source_path": tmp_path,
                "original_filename": filename,
                "created_at": time.time(),
                "last_access": time.time(),
            }

        _send_json(self, 200, {
            "session_id": session_id,
            "file_type": ctx.file_type,
            "has_tables": ctx.has_tables,
            "page_count": len(pages),
            "pages": [{"page_num": p.page_num, "text": p.text} for p in pages],
        })

    def _run_ocr(self, path: str, cfg: dict, ctx, file_bytes: bytes) -> list:
        """对图片/扫描版 PDF 跑 OCR"""
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        if ctx.file_type == "image":
            image_format = {"jpeg": "jpg"}.get(ext, ext)
            text = ocr_xfyun.ocr_image(file_bytes, cfg, image_format=image_format)
            return [Page(page_num=1, text=text)]
        # scanned_pdf
        return ocr_xfyun.ocr_pdf(path, cfg)

    # ---------- /api/proofread (NDJSON 流) ----------

    def _handle_proofread(self, user: dict):
        try:
            body = _read_request_body(self)
            data = json.loads(body.decode("utf-8")) if body else {}
        except (json.JSONDecodeError, ValueError) as e:
            _send_error_json(self, 400, f"请求体解析失败: {e}")
            return

        # 兼容两种入参：session_id（推荐）或直接传 pages
        pages = None
        if "session_id" in data:
            s = _get_session(data["session_id"])
            if not s or s.get("username") != user["username"]:
                _send_error_json(self, 404, "会话不存在或已过期，请重新上传文件")
                return
            pages = list(s["pages"])
        elif "pages" in data:
            pages = [Page(page_num=p["page_num"], text=p["text"]) for p in data["pages"]]
        else:
            _send_error_json(self, 400, "缺少 session_id 或 pages 字段")
            return
        if not pages:
            _send_error_json(self, 400, "没有可纠错的页")
            return

        # 用当前用户合并后的配置
        global_cfg = config.load()
        cfg = users.resolve_user_config(user, global_cfg)
        if not cfg.get("llm_api_key"):
            _send_error_json(self, 503, "服务器未配置 LLM API Key，请先在设置中填写")
            return

        # 限流：占用并发槽，超限拒绝
        if not users.acquire_proofread_slot(user["username"]):
            _send_error_json(self, 429, f"已有 {users.MAX_CONCURRENT_PROOFREAD} 个纠错任务"
                                       f"在运行，请等当前任务完成后再试")
            return

        # 客户端可覆盖以下字段
        rules = data.get("rules") or config.get_rules(cfg)
        review = bool(data.get("review", cfg.get("enable_review", True)))
        review_context = int(data.get("review_context", cfg.get("review_context_chars", 800)))
        token_limit = int(data.get("token_limit", cfg.get("token_limit", 1000000)))
        context_prev = data.get("context_prev", "") or ""
        context_next = data.get("context_next", "") or ""
        ocr_mode = bool(data.get("ocr_mode", False))
        # 可指定只纠错某些页（滑动式单页纠错）
        target_page_nums = data.get("page_nums")
        if target_page_nums:
            target_set = set(target_page_nums)
            pages = [p for p in pages if p.page_num in target_set]

        # 构造 LLM 客户端
        try:
            client = DeepSeekClient(
                api_key=cfg["llm_api_key"],
                base_url=cfg.get("llm_base_url", "https://api.deepseek.com"),
                model=cfg.get("llm_model", "deepseek-v4-flash"),
            )
        except DeepSeekError as e:
            users.release_proofread_slot(user["username"])
            _send_error_json(self, 503, str(e))
            return

        # 开始流式响应
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")  # 关闭反向代理缓冲
        self.end_headers()

        def emit(payload: dict):
            try:
                _send_ndjson_line(self, payload)
            except DISCONNECTED_ERRORS:
                # 抛哨兵异常而非 RuntimeError，便于外层精确捕获、避免落入通用 except 再尝试写入
                raise _ClientDisconnected()

        def on_status(msg):
            emit({"type": "status", "msg": msg})

        def on_progress(i, total, usage, total_usage):
            emit({"type": "progress", "chunk": i, "total": total,
                  "usage": vars(usage), "total_usage": vars(total_usage)})

        def on_review(i, total, confirmed, uncertain, rejected, excluded):
            emit({"type": "review", "chunk": i, "total": total,
                  "confirmed": confirmed, "uncertain": uncertain,
                  "rejected": rejected, "excluded": excluded})

        def on_limit_exceeded(total_usage):
            # 浏览器客户端无法弹窗，这里直接发 limit 事件让前端决定是否继续
            emit({"type": "limit", "total_usage": vars(total_usage)})
            return False  # 默认中止；浏览器可在收到 limit 后用新请求继续后续块

        try:
            errors, total_usage = proofread.proofread(
                pages, client, token_limit=token_limit,
                progress_cb=on_progress,
                on_limit_exceeded=on_limit_exceeded,
                rules=rules,
                review=review,
                review_context=review_context,
                review_cb=on_review,
                status_cb=on_status,
                context_prev=context_prev,
                context_next=context_next,
                ocr_mode=ocr_mode,
            )
        except (_ClientDisconnected, *DISCONNECTED_ERRORS):
            # 客户端已断连：仅记一行 info，不再尝试向其写入任何响应
            logger.info("客户端断开连接")
            return
        except DeepSeekError as e:
            try:
                emit({"type": "error", "msg": str(e)})
            except _ClientDisconnected:
                pass
            return
        except Exception as e:
            logger.exception("纠错异常")
            try:
                emit({"type": "error", "msg": f"纠错失败: {e}"})
            except _ClientDisconnected:
                pass
            return
        finally:
            # 无论正常结束、异常还是断连，都释放并发槽
            users.release_proofread_slot(user["username"])

        try:
            emit({"type": "done",
                  "errors": [vars(e) for e in errors],
                  "total_usage": vars(total_usage)})
        except _ClientDisconnected:
            pass

    # ---------- /api/session/<id>/preview ----------

    def _handle_preview(self, session_id: str, user: dict):
        s = _get_session(session_id)
        if not s or s.get("username") != user["username"]:
            _send_error_json(self, 404, "会话不存在或已过期")
            return
        try:
            body = _read_request_body(self)
            data = json.loads(body.decode("utf-8")) if body else {}
        except (json.JSONDecodeError, ValueError) as e:
            _send_error_json(self, 400, f"请求体解析失败: {e}")
            return

        corrections = data.get("corrections") or []
        pages = s["pages"]
        # 在副本上应用修正，便于前端预览
        text_by_page = {p.page_num: p.text for p in pages}
        for c in corrections:
            pn, start, end, repl = c[0], c[1], c[2], c[3]
            t = text_by_page.get(pn, "")
            text_by_page[pn] = t[:start] + repl + t[end:]
        _send_json(self, 200, {
            "pages": [{"page_num": p.page_num, "text": text_by_page[p.page_num]}
                      for p in pages],
        })

    # ---------- /api/export ----------

    def _handle_export(self, user: dict):
        try:
            body = _read_request_body(self)
            data = json.loads(body.decode("utf-8")) if body else {}
        except (json.JSONDecodeError, ValueError) as e:
            _send_error_json(self, 400, f"请求体解析失败: {e}")
            return

        sid = data.get("session_id")
        s = _get_session(sid) if sid else None
        if not s or s.get("username") != user["username"]:
            _send_error_json(self, 404, "会话不存在或已过期，请重新上传文件")
            return

        corrections = data.get("corrections") or []
        # corrections: [(page_num, start, end, replacement), ...]
        file_type = s["file_type"]
        original_filename = s.get("original_filename", "document")
        stem, _ = os.path.splitext(original_filename)
        out_dir = tempfile.mkdtemp(prefix="docproof_export_")
        out_path = os.path.join(out_dir, f"{stem}_已修正.docx")

        try:
            if file_type == "txt":
                out_path = os.path.join(out_dir, f"{stem}_已修正.txt")
                # 在页文本上应用修正
                pages = s["pages"]
                text_by_page = {p.page_num: p.text for p in pages}
                for c in corrections:
                    pn, start, end, repl = c[0], c[1], c[2], c[3]
                    t = text_by_page.get(pn, "")
                    text_by_page[pn] = t[:start] + repl + t[end:]
                ordered = [Page(page_num=p.page_num, text=text_by_page[p.page_num])
                           for p in pages]
                exporter.export_txt(ordered, out_path)
                mime = "text/plain; charset=utf-8"
            elif file_type == "docx":
                # 重新加载源 docx 以保证导出基于原始格式
                _, fresh_ctx = docload.load_document(s["source_path"])
                exporter.export_docx(fresh_ctx, corrections, out_path)
                mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            else:
                # pdf / scanned_pdf / image → docx
                pages = s["pages"]
                text_by_page = {p.page_num: p.text for p in pages}
                for c in corrections:
                    pn, start, end, repl = c[0], c[1], c[2], c[3]
                    t = text_by_page.get(pn, "")
                    text_by_page[pn] = t[:start] + repl + t[end:]
                ordered = [Page(page_num=p.page_num, text=text_by_page[p.page_num])
                           for p in pages]
                exporter.export_pdf_as_docx(ordered, [], out_path)
                mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        except Exception as e:
            logger.exception("导出失败")
            _send_error_json(self, 500, f"导出失败: {e}")
            return

        # 流式回传文件
        try:
            with open(out_path, "rb") as f:
                file_data = f.read()
        except OSError as e:
            _send_error_json(self, 500, f"读取导出文件失败: {e}")
            return
        finally:
            try:
                os.remove(out_path)
                os.rmdir(out_dir)
            except OSError:
                pass

        download_name = os.path.basename(out_path)
        # 文件名带中文，按 RFC 5987 用 filename* 编码
        from urllib.parse import quote
        encoded = quote(download_name)
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(file_data)))
        self.send_header("Content-Disposition",
                         f"attachment; filename=\"{encoded}\"; filename*=UTF-8''{encoded}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(file_data)

    # ---------- /api/users (管理员) ----------

    def _handle_list_users(self):
        _send_json(self, 200, {"users": users.list_users()})

    def _handle_create_user(self):
        """创建用户：username + password + is_admin。
        username 不能与已有重复；password 至少 6 位。
        """
        data, err = _read_json_body(self)
        if err:
            _send_error_json(self, 400, err)
            return
        username = (data.get("username") or "").strip()
        password = data.get("password") or ""
        is_admin = bool(data.get("is_admin", False))
        if not username:
            _send_error_json(self, 400, "缺少 username 字段")
            return
        if len(password) < 6:
            _send_error_json(self, 400, "密码至少 6 位")
            return
        ok = users.add_user(username, password, is_admin=is_admin)
        if not ok:
            _send_error_json(self, 400, "创建失败：用户名已存在或用户名非法（不能含空格）")
            return
        _send_json(self, 201, {
            "username": username,
            "is_admin": is_admin,
        })

    def _handle_delete_user(self, username: str):
        if not users.delete_user(username):
            _send_error_json(self, 400, "删除失败：用户不存在，或这是最后一个管理员账户（至少保留一个）")
            return
        _send_json(self, 200, {"status": "ok", "deleted": username})

    def _handle_user_action(self, username: str, action: str):
        """password / admin / rename 三种动作。
        password：管理员重置某用户密码（不需要旧密码），同时吊销其所有会话。
        """
        data, err = _read_json_body(self)
        if err:
            _send_error_json(self, 400, err)
            return
        if action == "password":
            new_pwd = data.get("password") or ""
            if len(new_pwd) < 6:
                _send_error_json(self, 400, "新密码至少 6 位")
                return
            if not users.reset_password(username, new_pwd):
                _send_error_json(self, 404, "用户不存在")
                return
            _send_json(self, 200, {"username": username, "password": new_pwd})
        elif action == "admin":
            is_admin = bool(data.get("is_admin", False))
            if not users.set_admin(username, is_admin):
                _send_error_json(self, 400, "操作失败：用户不存在，或这是最后一个管理员（不能取消）")
                return
            _send_json(self, 200, {"username": username, "is_admin": is_admin})
        elif action == "rename":
            new_name = (data.get("username") or data.get("display_name") or "").strip()
            if not new_name:
                _send_error_json(self, 400, "新用户名不能为空")
                return
            if not users.rename_user(username, new_name):
                _send_error_json(self, 400, "改名失败：用户不存在，或新用户名已被占用")
                return
            _send_json(self, 200, {"old_username": username, "username": new_name})


# ---------- 端口设置 ----------

class _SilentHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer 子类：覆写 handle_error，把客户端中途断连
    （关页/刷新/取消纠错）这类预期内的连接异常静默处理，只在日志里记一行，
    不向 stderr 打印堆栈，保持控制台干净。其他异常仍走默认 handle_error。"""

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, DISCONNECTED_ERRORS) or isinstance(exc, _ClientDisconnected):
            logger.info("客户端 %s 连接中断（预期内，已忽略）", client_address)
            return
        # 非断连异常：交给默认处理，但默认会 print 到 stderr，这里改为记日志
        logger.exception("处理请求时发生未捕获异常（来自 %s）", client_address)


def _read_json_body(handler, max_bytes: int = 64 * 1024):
    """读取并解析 JSON 请求体；返回 (data_dict, error_str)。
    请求体限 64KB（用户管理请求不应过大）。
    """
    try:
        body = _read_request_body(handler, max_bytes=max_bytes)
        data = json.loads(body.decode("utf-8")) if body else {}
        return data, None
    except (json.JSONDecodeError, ValueError) as e:
        return None, f"请求体解析失败: {e}"


def resolve_port(cli_port, env_port) -> int:
    """按优先级解析端口：CLI > 环境变量 > 交互输入（默认 8000）"""
    if cli_port:
        return cli_port
    if env_port:
        try:
            return int(env_port)
        except ValueError:
            print(f"[警告] 环境变量 DOCPROOF_PORT='{env_port}' 不是合法端口，忽略")
    # 交互输入
    print("=" * 60)
    print("DocProof 服务器启动 - 端口设置")
    print("=" * 60)
    while True:
        try:
            raw = input("请输入监听端口 [默认 8000，回车确认]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已取消")
            sys.exit(0)
        if not raw:
            return 8000
        if raw.isdigit() and 1 <= int(raw) <= 65535:
            return int(raw)
        print(f"[错误] '{raw}' 不是合法端口（1-65535），请重新输入")


def _get_lan_ip() -> str:
    """获取本机局域网 IPv4（用于打印访问 URL）；失败回退 127.0.0.1"""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 不真正发包，仅让 OS 决定出接口地址
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def main():
    parser = argparse.ArgumentParser(
        description="DocProof 文档纠错服务器（HTTP + 浏览器客户端，多用户隔离版）")
    parser.add_argument("--port", "-p", type=int, default=None,
                        help="监听端口（未指定时查 DOCPROOF_PORT 环境变量，再无则交互输入）")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="监听地址（默认 0.0.0.0 局域网可访问；仅本机用 127.0.0.1）")
    args = parser.parse_args()

    port = resolve_port(args.port, os.environ.get("DOCPROOF_PORT"))

    # 启动前加载配置（同时触发旧明文 config.json → 新加密 _server_config.json 的迁移）
    cfg = config.load()
    if not (cfg.get("llm_api_key") or cfg.get("xf_appid")):
        # 尝试从 DEAResource 自动导入（开发环境便利）
        try:
            config.bootstrap_from_dearesource(cfg)
        except Exception as e:
            logger.warning("从 DEAResource 导入密钥失败: %s", e)

    # 多用户：首启自动创建管理员账户并返回初始账号密码
    admin_user, admin_pwd = users.ensure_admin_exists()

    server = _SilentHTTPServer((args.host, port), DocProofHandler)
    server.daemon_threads = True

    # 计算访问 URL：0.0.0.0 时打印实际 LAN IP，便于手机/其他设备扫码访问
    lan_ip = _get_lan_ip() if args.host in ("0.0.0.0", "::", "") else args.host
    # 打印访问 URL（用户偏好：控制台输出访问 URL 便于直接打开）
    print()
    print("=" * 60)
    print("DocProof 服务器已启动（多用户隔离版 v1.4 - 用户名密码登录）")
    print(f"  监听地址:    {args.host}:{port}")
    if args.host in ("0.0.0.0", "::", ""):
        print(f"  本机访问:    http://127.0.0.1:{port}/")
        print(f"  局域网访问:  http://{lan_ip}:{port}/")
        print(f"               （同一 WiFi/网段下其他设备用此 URL）")
    else:
        print(f"  浏览器访问:  http://{args.host}:{port}/")
    print(f"  健康检查:    http://127.0.0.1:{port}/api/health")
    print(f"  全局配置:    {config.CONFIG_PATH}")
    print(f"  用户数据:    {users.USERS_PATH}")
    print(f"  会话数据:    {users.SESSIONS_PATH}（7 天有效期）")
    print(f"  日志文件:    {_LOG_PATH}（2MB 轮转，保留 5 份）")
    print(f"  敏感字段:    DPAPI 加密（绑定当前 Windows 用户）")
    if admin_user:
        print()
        print("  ⚠ 首次启动：已自动创建管理员账户")
        print(f"    用户名:    {admin_user}")
        print(f"    初始密码:  {admin_pwd}（建议登录后立即修改）")
        print(f"    在浏览器打开访问 URL，用上述账号密码登录。")
        print(f"    登录后可在「用户管理」中创建其他用户、重置密码。")
    if not cfg.get("llm_api_key"):
        print(f"  ⚠ 未配置 LLM API Key，管理员登录后在「设置」中填写")
    if args.host == "0.0.0.0":
        print(f"  ⚠ 监听 0.0.0.0：局域网内任何设备都能访问，请妥善保管账号密码")
    print("=" * 60)
    print("按 Ctrl+C 停止服务")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在停止服务器...")
        server.shutdown()
        # 清理会话临时文件
        with _session_lock:
            for s in SESSIONS.values():
                _cleanup_session_files(s)
            SESSIONS.clear()
        print("已停止")


if __name__ == "__main__":
    main()
