"""轻量 Web API（基于 Python 标准库 http.server，无额外依赖）。

启动前在控制台交互式设置端口（默认 20000，直接回车使用默认值；
若被占用则自动向后寻找可用端口）；
确认后启动服务，此后控制台不再输出，所有日志写入 output/log/server.log（10MB 轮转）。
所有响应均为 JSON。

接口列表：
    GET  /                              首页（接口说明）
    GET  /api/libraries                 列出所有库
    POST /api/libraries                 创建库        body: {"name":"...", "note":"..."}
    PATCH /api/libraries/{name}         修改备注      body: {"note":"..."}
    DELETE /api/libraries/{name}?yes=1  删除库
    GET  /api/stats?library=xxx         统计（可选指定库）
    GET  /api/search?query=xxx&libraries=A,B&parallel=4&top=20
    POST /api/import                    导入文件      body: {"files":["path1"],"library":"xxx","force":false}
    GET  /api/files?library=xxx&search=&ext=&page=1&page_size=50 列出库内源文件
    GET  /api/download?file_path=xxx&library=xxx  下载源文件到客户端浏览器
    GET  /api/file-chunks?library=xxx&sha256=xxx  获取源文件所有 chunk 拼接文本（预览用）
    DELETE /api/files?library=xxx&ext=txt   删除库内指定类型文件
    DELETE /api/files?library=xxx&sha=xxx   删除库内指定 SHA 文件
    GET  /api/verify?library=xxx        校验库
    POST /api/build-index               重建索引      body: {"library":"xxx"}
    POST /api/recover                   恢复事务      body: {"library":"xxx"}

用法：
    python web_api.py                       # 控制台交互设置端口（默认 20000，回车确认；占用则自动后移）
    python web_api.py --port 9000           # 跳过交互，直接用指定端口启动
    python web_api.py --no-dialog           # 跳过交互，用默认端口 20000 启动（后台服务用；占用则自动后移）

    默认监听 0.0.0.0，允许局域网内其他设备访问。
    如仅本机使用，改为 --host 127.0.0.1。

数据与日志布局：
    data/                所有数据（库、注册表、设置、会话），与代码分离
    data/libraries/      各库数据目录
    output/log/          日志文件（10MB 轮转，最多 5 个历史文件）
    启动后控制台不再输出，所有日志写入 output/log/server.log
"""
from __future__ import annotations

import argparse
from typing import Any, Dict
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from logging.handlers import RotatingFileHandler

from library import LibraryRegistry
from storage import ZoneManager
from settings import SettingsStore, mask_secret
from deepseek import DeepSeekClient, DeepSeekError, AVAILABLE_MODELS, PROVIDER_BASE_URLS, V4_FLASH, V4_PRO
from chat_store import ChatStore
from auth import UserStore, GUEST, SESSION_TTL_SECONDS
from userdata import auth_base_dir as _auth_base_dir


DEFAULT_PORT = 20000


# 控制台事件处理器引用（保持全局引用防止被垃圾回收）
_console_handler_ref = None


def _setup_console_ctrl_handler():
    """注册 Windows 控制台事件处理器，确保关闭窗口时能终止进程。

    CTRL_CLOSE_EVENT（点击窗口右上角 X 关闭）时 Windows 仅给约 5 秒
    清理时间，而 serve_forever() 阻塞在主线程中不会响应。
    通过 os._exit(0) 强制退出；所有工作线程均为 daemon，会随主进程退出。
    Ctrl+C 仍走 Python 默认路径触发 KeyboardInterrupt 实现优雅关闭。
    """
    global _console_handler_ref
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        HANDLER_ROUTINE = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
        CTRL_C_EVENT = 0
        CTRL_BREAK_EVENT = 1
        CTRL_CLOSE_EVENT = 2
        CTRL_LOGOFF_EVENT = 5
        CTRL_SHUTDOWN_EVENT = 6

        def _handler(ctrl_type):
            if ctrl_type in (
                CTRL_CLOSE_EVENT,
                CTRL_BREAK_EVENT,
                CTRL_LOGOFF_EVENT,
                CTRL_SHUTDOWN_EVENT,
            ):
                # 关闭窗口 / Ctrl+Break / 注销 / 关机：立即退出
                os._exit(0)
            # CTRL_C_EVENT: 交给 Python 默认处理（触发 KeyboardInterrupt）
            return False

        _console_handler_ref = HANDLER_ROUTINE(_handler)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.SetConsoleCtrlHandler(_console_handler_ref, True)
    except Exception:
        pass


def _is_port_in_use(port: int) -> bool:
    """检测端口是否被系统占用。

    分别检测 0.0.0.0 和 127.0.0.1 绑定，任一失败即认为占用。
    """
    import socket
    if not (1 <= port <= 65535):
        return True
    for host in ("0.0.0.0", "127.0.0.1"):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.settimeout(0.3)
            try:
                s.bind((host, port))
            except OSError:
                return True
    return False


def _find_available_port(start_port: int = DEFAULT_PORT) -> int:
    """从 start_port 开始向后寻找可用端口。"""
    port = start_port
    while port <= 65535:
        if not _is_port_in_use(port):
            return port
        port += 1
    raise RuntimeError("无可用端口（1-65535 均已占用）")


def _resolve_script_dir() -> str:
    """解析代码目录（web_api.py / exe 所在目录）。

    打包后（PyInstaller）：exe 所在目录；
    开发模式：脚本所在目录。
    代码、static 资源、output 日志都放在这里。
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _migrate_data_layout(script_dir: str, data_dir: str) -> None:
    """把旧布局（数据散落在代码目录）迁移到 data/ 子目录。仅迁移一次。

    旧布局：_libraries.json / _settings.json / 库目录都直接放在代码根。
    新布局：所有数据统一放在 data/ 下，库目录统一放在 data/libraries/<slug>/。
    迁移完成后写入 data/_migrated 标记，后续启动跳过。
    """
    marker = os.path.join(data_dir, "_migrated")
    if os.path.isfile(marker):
        return
    # 仅当旧布局存在（代码根有 _libraries.json）才迁移
    old_reg = os.path.join(script_dir, "_libraries.json")
    if not os.path.isfile(old_reg) and not os.path.isdir(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        with open(marker, "w", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%dT%H:%M:%S"))
        return
    os.makedirs(data_dir, exist_ok=True)
    # 1. 顶层 JSON 数据文件
    for fn in ["_libraries.json", "_settings.json", "_chat_sessions.json",
               "_import_batches.json", "_inquiries.json"]:
        src = os.path.join(script_dir, fn)
        dst = os.path.join(data_dir, fn)
        if os.path.isfile(src) and not os.path.isfile(dst):
            shutil.move(src, dst)
    # 2. 库目录：根据 _libraries.json 的 path 字段迁移到 data/libraries/<slug>
    reg_path = os.path.join(data_dir, "_libraries.json")
    if os.path.isfile(reg_path):
        with open(reg_path, "r", encoding="utf-8") as f:
            reg = json.load(f)
        libs_dir = os.path.join(data_dir, "libraries")
        os.makedirs(libs_dir, exist_ok=True)
        for d in reg.get("libraries", []):
            old_rel = d.get("path", "")
            if not old_rel or os.path.isabs(old_rel):
                continue
            # slug 取 path 末段（兼容 "2011郎溪年鉴" 和 "_libraries/郎溪县志_第一轮"）
            slug = old_rel.replace("\\", "/").rstrip("/").split("/")[-1]
            src_dir = os.path.join(script_dir, old_rel)
            dst_dir = os.path.join(libs_dir, slug)
            if os.path.isdir(src_dir) and not os.path.isdir(dst_dir):
                shutil.move(src_dir, dst_dir)
            elif os.path.isdir(src_dir) and os.path.isdir(dst_dir):
                # 目标已存在：合并（把 src 内容移入 dst）
                for name in os.listdir(src_dir):
                    shutil.move(os.path.join(src_dir, name), os.path.join(dst_dir, name))
                os.rmdir(src_dir)
            d["path"] = f"libraries/{slug}"
        with open(reg_path, "w", encoding="utf-8") as f:
            json.dump(reg, f, ensure_ascii=False, indent=2)
    # 3. 删除空的 _libraries/ 旧容器目录
    old_container = os.path.join(script_dir, "_libraries")
    if os.path.isdir(old_container) and not os.listdir(old_container):
        try:
            os.rmdir(old_container)
        except OSError:
            pass
    # 写标记
    with open(marker, "w", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%S"))


def _resolve_data_dir(script_dir: str) -> str:
    """解析数据根目录（与代码分离）。

    所有库数据、注册表、设置、会话都放在 <script_dir>/data/ 下。
    首次运行时自动把旧布局的数据迁移过来。
    """
    data_dir = os.path.join(script_dir, "data")
    _migrate_data_layout(script_dir, data_dir)
    return data_dir


# 代码目录（static 资源、output 日志在此）
SCRIPT_DIR = _resolve_script_dir()
# 数据根目录（库、注册表、设置、会话在此；与代码分离）
BASE_DIR = _resolve_data_dir(SCRIPT_DIR)
# 用户登录相关数据目录（<用户主目录>/biaoshifu，跨应用共用同一组账号；
# 与库数据 BASE_DIR 分离，不随 --data-dir 切换而变）
AUTH_DIR = _auth_base_dir()

# ============ 日志：输出到 output/log/server.log，10MB 轮转，不输出到控制台 ============
LOG_DIR = os.path.join(SCRIPT_DIR, "output", "log")
os.makedirs(LOG_DIR, exist_ok=True)
_logger = logging.getLogger("server")
_logger.setLevel(logging.INFO)
_log_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "server.log"),
    maxBytes=10 * 1024 * 1024,   # 10 MB
    backupCount=5,               # 保留 5 个历史文件，最多约 60MB
    encoding="utf-8",
)
_log_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
)
_logger.addHandler(_log_handler)
_logger.propagate = False  # 阻止日志向 root logger 传播（避免控制台输出）


class _LogStream:
    """把写入流的内容按行转发到日志文件（用于重定向 stdout/stderr）。

    这样所有模块（ai_search/embedding/faiss_index 等）的 print() 都会进入
    日志文件，控制台保持完全静默，避免长时运行控制台缓冲膨胀。
    """

    def __init__(self, level=logging.INFO):
        self._level = level
        self._buf = ""

    def write(self, s):
        if not s:
            return
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                _logger.log(self._level, "%s", line)

    def flush(self):
        if self._buf.strip():
            _logger.log(self._level, "%s", self._buf)
        self._buf = ""

    def isatty(self):
        return False


# 保留原始标准流引用：启动前的控制台交互（端口输入）需要真实控制台
_orig_stdout = sys.stdout
_orig_stderr = sys.stderr


def _redirect_stdio_to_log():
    """服务启动后调用：把 stdout/stderr 重定向到日志文件，控制台不再输出。

    启动前的端口交互（input/print）已在真实控制台完成，不受影响。
    """
    sys.stdout = _LogStream(logging.INFO)
    sys.stderr = _LogStream(logging.WARNING)


def _reconfigure_log_handler():
    """重新配置日志 handler 到当前 LOG_DIR（用于多实例数据目录切换）。

    模块加载时日志 handler 绑定到默认 LOG_DIR；当用户指定了自定义数据目录后，
    调用此函数把日志输出切换到 <data_dir>/output/log/server.log，实现多实例日志隔离。
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    for h in list(_logger.handlers):
        _logger.removeHandler(h)
        h.close()
    handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "server.log"),
        maxBytes=10 * 1024 * 1024,   # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    _logger.addHandler(handler)


def _log_exc(exc_type, exc_value, exc_tb):
    """全局未捕获异常写入日志文件，不输出到控制台。"""
    import traceback
    _logger.error("未捕获异常: %s", "".join(traceback.format_exception(exc_type, exc_value, exc_tb)))


sys.excepthook = _log_exc


def _resolve_static_path() -> str:
    """解析 static/index.html 的路径（基于代码目录，不基于数据目录）。

    打包后：先查 exe 同级 static/，再查 _MEIPASS/static/（PyInstaller 临时解压目录）。
    开发模式：脚本同级 static/。
    """
    if getattr(sys, "frozen", False):
        # 优先查 exe 同级目录的 static/
        local_static = os.path.join(SCRIPT_DIR, "static", "index.html")
        if os.path.isfile(local_static):
            return local_static
        # 回退到 PyInstaller 临时解压目录
        meipass_static = os.path.join(sys._MEIPASS, "static", "index.html")
        if os.path.isfile(meipass_static):
            return meipass_static
    return os.path.join(SCRIPT_DIR, "static", "index.html")


def _registry() -> LibraryRegistry:
    return LibraryRegistry(BASE_DIR)


def _settings() -> SettingsStore:
    return SettingsStore(AUTH_DIR)


def _chat_store() -> ChatStore:
    return ChatStore(BASE_DIR)


def _build_client_from_settings() -> DeepSeekClient:
    """根据当前设置构建 DeepSeek 客户端。未配置 api_key 时抛 ValueError。"""
    store = _settings()
    api_key = store.get("deepseek_api_key")
    if not api_key:
        raise ValueError("未配置 DeepSeek API Key，请先到「设置」页配置")
    model = store.get("deepseek_model") or V4_FLASH
    base_url = store.get("deepseek_base_url") or "https://api.deepseek.com"
    return DeepSeekClient(api_key=api_key, model=model, base_url=base_url)


def _ok(data, status=200):
    return status, {"ok": True, "data": data}


def _err(msg: str, status=400):
    return status, {"ok": False, "error": msg}


# ============================================================
#  身份验证：当前用户解析 / 权限校验
# ============================================================

def _auth_store() -> UserStore:
    """用户登录存储（<用户主目录>/biaoshifu，全局共享账号）。"""
    return UserStore(AUTH_DIR)


def _guest_user() -> dict:
    """游客身份（未登录）。"""
    return {"username": GUEST, "role": GUEST, "is_admin": False}


def _current_user(handler) -> dict:
    """从请求 Cookie 解析当前用户。未登录 / token 无效 / 过期 → 游客。"""
    token = ""
    try:
        from http.cookies import SimpleCookie
        cookie = handler.headers.get("Cookie", "")
        c = SimpleCookie()
        c.load(cookie)
        if "session" in c:
            token = c["session"].value
    except Exception:
        token = ""
    store = _auth_store()
    u = store.get_user_by_token(token)
    if not u:
        return _guest_user()
    return {
        "username": u["username"],
        "role": u["role"],
        "is_admin": u["role"] == "admin",
    }


def _session_cookie_header(token: str) -> str:
    """生成 Set-Cookie 头（HttpOnly，防 XSS 读取）。"""
    return (f"session={token}; Path=/; HttpOnly; SameSite=Lax; "
            f"Max-Age={SESSION_TTL_SECONDS}")


def _require_login(user) -> tuple | None:
    """需要登录才能操作。未登录返回错误响应，否则返回 None。"""
    if user["username"] == GUEST:
        return _err("请先登录", 401)
    return None


def _require_admin(user) -> tuple | None:
    """需要管理员权限。无任何账号时放行（首次引导期，供初始化管理员）。"""
    if _auth_store().count_users() == 0:
        return None
    if not user.get("is_admin"):
        return _err("需要管理员权限", 403)
    return None


def _get_lib(reg, name, user, write=False):
    """按身份获取库并校验可见/可写权限（同名不同属主时优先命中当前用户/公共库）。

    返回 (lib, None) 或 (None, error_response)。
    """
    lib = reg.get_library_for(name, user["username"], user["is_admin"])
    if lib is None:
        return None, _err(f"库不存在: {name}", 404)
    if write:
        if not lib.writable_by(user["username"], user["is_admin"]):
            return None, _err(f"无权操作库: {name}", 403)
    else:
        if not lib.visible_to(user["username"], user["is_admin"]):
            # 隐藏私有库的存在性
            return None, _err(f"库不存在: {name}", 404)
    return lib, None


def _check_libraries_visible(reg, names, user) -> tuple | None:
    """批量校验库名列表全部对当前用户可见。非法返回错误响应。"""
    for n in names:
        if not n:
            continue
        lib = reg.get_library_for(n, user["username"], user["is_admin"])
        if lib is None or not lib.visible_to(user["username"], user["is_admin"]):
            return _err(f"库不存在: {n}", 404)
    return None


def _lib_payload(lib, user, stats: dict | None = None) -> dict:
    """库信息序列化（含多用户字段）。stats 为可选的 mgr.stats() 结果。"""
    payload = {
        "id": lib.id,
        "name": lib.name,
        "note": lib.note,
        "path": lib.path,
        "created_at": lib.created_at,
        "folder": lib.folder,
        "owner": lib.owner,
        "is_public": lib.owner == GUEST,
        "can_edit": lib.writable_by(user["username"], user["is_admin"]),
    }
    if stats is not None:
        payload["stats"] = {
            "chars": stats["total_chars"],
            "chunks": stats["total_chunks"],
            "sources": stats["total_sources"],
            "zones": stats["zone_count"],
        }
    return payload


# ============================================================
#  接口处理函数
# ============================================================

def handle_list_libraries(method, path, query, body, user=None):
    reg = _registry()
    libs = reg.list_libraries_for(user["username"], user["is_admin"])
    out = []
    for lib in libs:
        mgr = lib.manager(BASE_DIR)
        s = mgr.stats()
        out.append(_lib_payload(lib, user, s))
    folders = reg.list_folders()
    return _ok({"libraries": out, "count": len(out), "folders": folders})


def handle_create_library(method, path, query, body, user=None):
    if method != "POST":
        return _err("需要 POST 方法", 405)
    name = body.get("name")
    note = body.get("note", "")
    if not name:
        return _err("缺少 name 参数")
    reg = _registry()
    try:
        # 游客创建的库归公共（所有游客公用）；登录用户创建的库归自己
        lib = reg.create(name, note=note, owner=user["username"])
    except ValueError as e:
        return _err(str(e))
    return _ok({"id": lib.id, "name": lib.name, "note": lib.note,
                "path": lib.abs_path(BASE_DIR), "owner": lib.owner})


def handle_update_library(method, path, query, body, name, user=None):
    if method != "PATCH":
        return _err("需要 PATCH 方法", 405)
    note = body.get("note")
    if note is None:
        return _err("缺少 note 参数")
    reg = _registry()
    lib, err = _get_lib(reg, name, user, write=True)
    if err:
        return err
    try:
        lib = reg.update_note(lib.name, note, owner=lib.owner)
    except ValueError as e:
        return _err(str(e), 404)
    return _ok({"name": lib.name, "note": lib.note})


def handle_move_library(method, path, query, body, name, user=None):
    """POST /api/libraries/{name}/move  body: {folder: "xxx"}
    移动库到文件夹。folder 为空字符串表示移到根级。
    """
    if method != "POST":
        return _err("需要 POST 方法", 405)
    folder = body.get("folder", "")
    reg = _registry()
    lib, err = _get_lib(reg, name, user, write=True)
    if err:
        return err
    try:
        lib = reg.move_library(lib.name, folder, owner=lib.owner)
    except ValueError as e:
        return _err(str(e), 404)
    return _ok({"name": lib.name, "folder": lib.folder})


def handle_folders(method, path, query, body, user=None):
    """文件夹管理（支持嵌套三层路径）。

    GET    /api/folders                  列出所有文件夹路径（含中间路径）
    POST   /api/folders   body:{name}    创建文件夹（name 可为 "A/B/C" 路径）
    POST   /api/folders/rename  body:{old, new}  重命名文件夹
    POST   /api/folders/delete  body:{name}      删除文件夹（库移到根级）

    多用户：文件夹是全局组织视图，游客只读；写操作（创建/重命名/删除/移动）需登录。
    """
    reg = _registry()
    if method == "GET":
        return _ok({"folders": reg.list_folders()})
    if method == "POST":
        err = _require_login(user)
        if err:
            return err
        # 子路由：rename / delete（文件夹路径可能含 /，不能放在 URL path 中）
        action = body.get("action", "").strip()
        if action == "rename":
            old = body.get("old", "").strip()
            new = body.get("new", "").strip()
            if not old or not new:
                return _err("缺少 old 或 new 参数")
            try:
                r = reg.rename_folder(old, new)
            except ValueError as e:
                return _err(str(e))
            return _ok(r)
        if action == "delete":
            name = body.get("name", "").strip()
            if not name:
                return _err("缺少 name 参数")
            try:
                r = reg.delete_folder(name)
            except ValueError as e:
                return _err(str(e))
            return _ok(r)
        if action == "move_folder":
            old = body.get("old", "").strip()
            new_parent = body.get("new_parent", "").strip()
            if not old:
                return _err("缺少 old 参数")
            try:
                r = reg.move_folder(old, new_parent)
            except ValueError as e:
                return _err(str(e))
            return _ok(r)
        # 默认：创建文件夹
        name = body.get("name", "").strip()
        if not name:
            return _err("缺少 name 参数")
        try:
            reg.create_folder(name)
        except ValueError as e:
            return _err(str(e))
        return _ok({"created": name})
    return _err("不支持的方法", 405)


# ============================================================
#  库级导入互斥锁
# ============================================================
# 同一库的 zone 事务文件（_transaction.json）不并发安全：两个请求同时
# 导入同一库会互相覆盖/删除事务文件（FileNotFoundError 等）。
# 以库根目录为粒度加锁，同一库的导入/恢复/删库重建串行执行，不同库并行。
_import_lib_locks: Dict[str, threading.Lock] = {}
_import_lib_locks_guard = threading.Lock()


def _lib_import_lock(lib_root: str) -> threading.Lock:
    """获取某个库的导入互斥锁（按库根路径）。"""
    with _import_lib_locks_guard:
        if lib_root not in _import_lib_locks:
            _import_lib_locks[lib_root] = threading.Lock()
        return _import_lib_locks[lib_root]


# ============================================================
#  身份验证 API：注册 / 登录 / 登出 / 当前用户
# ============================================================

def handle_auth_register(handler, method, path, query, body):
    """POST /api/auth/register  body: {username, password}
    注册账号。首个注册用户自动成为管理员。
    """
    if method != "POST":
        return _err("需要 POST 方法", 405)
    username = (body.get("username") or "").strip()
    password = body.get("password") or ""
    store = _auth_store()
    try:
        user = store.register(username, password)
    except ValueError as e:
        return _err(str(e))
    # 注册成功即登录
    token = store.create_session(user["username"])
    return _ok({"username": user["username"], "role": user["role"],
                "is_admin": user["role"] == "admin",
                "first_user": store.count_users() == 1},
               _set_session_cookie(handler, token))


def handle_auth_login(handler, method, path, query, body):
    """POST /api/auth/login  body: {username, password}
    登录，成功后设置会话 Cookie。
    """
    if method != "POST":
        return _err("需要 POST 方法", 405)
    username = (body.get("username") or "").strip()
    password = body.get("password") or ""
    store = _auth_store()
    user = store.authenticate(username, password)
    if user is None:
        return _err("用户名或密码错误", 401)
    token = store.create_session(user["username"])
    return _ok({"username": user["username"], "role": user["role"],
                "is_admin": user["role"] == "admin"},
               _set_session_cookie(handler, token))


def handle_auth_logout(handler, method, path, query, body):
    """POST /api/auth/logout  注销当前会话并清除 Cookie。"""
    if method != "POST":
        return _err("需要 POST 方法", 405)
    token = _token_from_cookie(handler)
    if token:
        _auth_store().logout(token)
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Set-Cookie", "session=; Path=/; HttpOnly; Max-Age=0")
    body_bytes = json.dumps({"ok": True, "data": {"logout": True}},
                            ensure_ascii=False).encode("utf-8")
    handler.send_header("Content-Length", str(len(body_bytes)))
    handler.end_headers()
    handler.wfile.write(body_bytes)
    return None


def handle_auth_me(method, path, query, body, user=None):
    """GET /api/auth/me  返回当前身份信息。"""
    store = _auth_store()
    user_count = store.count_users()
    return _ok({
        "username": user["username"],
        "role": user["role"],
        "is_admin": user["is_admin"],
        "is_guest": user["username"] == GUEST,
        "user_count": user_count,
        "has_admin": any(u["role"] == "admin" for u in store.list_users()),
    })


def _token_from_cookie(handler) -> str:
    try:
        from http.cookies import SimpleCookie
        c = SimpleCookie()
        c.load(handler.headers.get("Cookie", ""))
        if "session" in c:
            return c["session"].value or ""
    except Exception:
        pass
    return ""


def _set_session_cookie(handler, token: str) -> int:
    """写入会话 Cookie 到响应头。返回 200。"""
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Set-Cookie", _session_cookie_header(token))
    return 200


# ============================================================
#  数据迁移 API：复制公共库到我的库 / 转移所有权
# ============================================================

def handle_clone_library(method, path, query, body, name, user=None):
    """POST /api/libraries/{name}/clone  body: {new_name?, note?}
    把库复制一份到当前用户名下（"把公共库添加到自己的库"）。
    登录用户可复制公共库；管理员可复制任意库。
    """
    if method != "POST":
        return _err("需要 POST 方法", 405)
    err = _require_login(user)
    if err:
        return err
    reg = _registry()
    src, err = _get_lib(reg, name, user, write=False)
    if err:
        return err
    if not (src.owner == GUEST or user["is_admin"]):
        return _err("只能复制公共库到自己的名下", 403)
    try:
        lib = reg.clone_library(
            name,
            to_owner=user["username"],
            new_name=body.get("new_name") or None,
            note=body.get("note") or "",
            from_owner=src.owner,
        )
    except ValueError as e:
        return _err(str(e))
    return _ok({"id": lib.id, "name": lib.name, "path": lib.path,
                "owner": lib.owner})


def handle_transfer_library(method, path, query, body, name, user=None):
    """POST /api/libraries/{name}/transfer  body: {to_owner}
    管理员把库所有权转移给指定用户（或设为公共库 guest）。
    """
    if method != "POST":
        return _err("需要 POST 方法", 405)
    err = _require_admin(user)
    if err:
        return err
    to_owner = (body.get("to_owner") or "").strip()
    if not to_owner:
        return _err("缺少 to_owner 参数")
    if to_owner != GUEST:
        store = _auth_store()
        if not store.user_exists(to_owner):
            return _err(f"用户不存在: {to_owner}", 404)
    reg = _registry()
    lib = reg.get_library_for(name, user["username"], True)
    if lib is None:
        return _err(f"库不存在: {name}", 404)
    try:
        lib = reg.set_owner(lib.name, to_owner, from_owner=lib.owner)
    except ValueError as e:
        return _err(str(e), 404)
    return _ok({"name": lib.name, "owner": lib.owner})


def handle_auth_users(method, path, query, body, user=None):
    """GET /api/auth/users  管理员查看用户列表（含各用户库数量）。"""
    if method != "GET":
        return _err("需要 GET 方法", 405)
    err = _require_admin(user)
    if err:
        return err
    store = _auth_store()
    reg = _registry()
    usage = reg.list_owners_usage()
    users = []
    for u in store.list_users():
        users.append({
            "username": u["username"],
            "role": u["role"],
            "created_at": u["created_at"],
            "library_count": usage.get(u["username"], 0),
        })
    return _ok({"users": users, "count": len(users),
                "public_library_count": usage.get(GUEST, 0)})


def handle_auth_user_op(method, path, query, body, username, user=None):
    """管理员管理单个用户（仅 admin）。

    POST   /api/auth/users/{name}/role       {role: "admin"|"user"}
    POST   /api/auth/users/{name}/password   {new_password: "..."}
    DELETE /api/auth/users/{name}            删除用户（其名下库保留但仅管理员可见）
    """
    err = _require_admin(user)
    if err:
        return err
    store = _auth_store()
    parts = path.split("/")
    action = parts[-1] if len(parts) >= 5 else ""

    if method == "DELETE" and action == username:
        # 保护：不允许删除最后一个管理员，避免系统无人可管理
        target = store.get_user(username)
        if target is None:
            return _err(f"用户不存在: {username}", 404)
        if target["role"] == "admin":
            admins = [u for u in store.list_users() if u["role"] == "admin"]
            if len(admins) <= 1:
                return _err("不能删除最后一个管理员（会失去管理能力）", 400)
        ok = store.remove_user(username)
        if not ok:
            return _err(f"用户不存在: {username}", 404)
        return _ok({"deleted": username})

    if method == "POST" and action == "role":
        role = body.get("role")
        if role not in ("admin", "user"):
            return _err("role 必须是 admin 或 user")
        target = store.get_user(username)
        if target is None:
            return _err(f"用户不存在: {username}", 404)
        # 保护：不允许把最后一个管理员降级，避免系统无人可管理
        if target["role"] == "admin" and role == "user":
            admins = [u for u in store.list_users() if u["role"] == "admin"]
            if len(admins) <= 1:
                return _err("不能降级最后一个管理员（会失去管理能力）", 400)
        u = store.set_role(username, role)
        return _ok(u)

    if method == "POST" and action == "password":
        new_pw = (body.get("new_password") or "").strip()
        if len(new_pw) < 6:
            return _err("新密码至少 6 位")
        ok = store.admin_reset_password(username, new_pw)
        if not ok:
            return _err(f"用户不存在: {username}", 404)
        return _ok({"username": username, "updated": True})

    return _err("不支持的操作", 400)


# ============================================================
#  质询报告管理
# ============================================================

def handle_inquiries(method, path, query, body, user=None):
    """质询报告管理。

    GET    /api/inquiries              列出所有报告（需登录；报告可能含私有库信息）
    DELETE /api/inquiries              清空所有报告（仅管理员）
    """
    from inquiry_store import InquiryStore
    store = InquiryStore(BASE_DIR)
    if method == "GET":
        err = _require_login(user)
        if err:
            return err
        reports = store.list_all()
        return _ok({"reports": reports, "count": len(reports)})
    if method == "DELETE":
        err = _require_admin(user)
        if err:
            return err
        count = store.clear_all()
        return _ok({"cleared": count})
    return _err("不支持的方法", 405)


def handle_inquiry_op(method, path, query, body, report_id, user=None):
    """单个质询报告操作。

    GET    /api/inquiries/{id}   获取报告详情（需登录）
    DELETE /api/inquiries/{id}   删除报告（仅管理员）
    """
    from inquiry_store import InquiryStore
    store = InquiryStore(BASE_DIR)
    if method == "GET":
        err = _require_login(user)
        if err:
            return err
        r = store.get(report_id)
        if r is None:
            return _err(f"报告不存在: {report_id}", 404)
        return _ok(r)
    if method == "DELETE":
        err = _require_admin(user)
        if err:
            return err
        ok = store.delete(report_id)
        if not ok:
            return _err(f"报告不存在: {report_id}", 404)
        return _ok({"deleted": report_id})
    return _err("不支持的方法", 405)


def handle_inquiry_chunks(method, path, query, body, report_id, user=None):
    """GET /api/inquiries/{id}/chunks —— 批量取报告引用 chunk 的原文（快速验证）。

    报告中的 chunk_ids/chunk_refs 逐个解析出所属库并读取完整 chunk
    （text/heading/来源文件），前端在报告详情页一键展开原文，与 AI 的
    问题描述对照验证，无需手动检索定位。

    chunk 库解析顺序：chunk_refs[].library → 报告 libraries 逐一尝试。
    """
    if method != "GET":
        return _err("需要 GET 方法", 405)
    from inquiry_store import InquiryStore
    store = InquiryStore(BASE_DIR)
    err = _require_login(user)
    if err:
        return err
    r = store.get(report_id)
    if r is None:
        return _err(f"报告不存在: {report_id}", 404)

    reg = _registry()
    libs_visible = {l.name: l for l in reg.list_libraries_for(
        user["username"], user["is_admin"])}

    # 候选库：chunk_refs 的库优先，报告 libraries 兜底（cid -> 候选库列表）
    ref_libs = {}
    for ref in (r.get("chunk_refs") or []):
        cid = (ref.get("chunk_id") or "").strip()
        lib = (ref.get("library") or "").strip()
        if cid:
            ref_libs[cid] = [lib] if lib else []
    report_libs = [l for l in (r.get("libraries") or []) if l]

    def _read_chunk(lib_name: str, chunk_id: str):
        """在指定库中读取 chunk，返回 dict 或 None。"""
        lib = libs_visible.get(lib_name)
        if lib is None:
            return None
        parts = chunk_id.split("/")
        if len(parts) != 2:
            return None
        zone_id, chunk_name = parts
        mgr = lib.manager(BASE_DIR)
        zone = mgr.get_zone(zone_id)
        if zone is None:
            return None
        chunk_path = os.path.join(zone.chunks_dir, f"{chunk_name}.json")
        if not os.path.isfile(chunk_path):
            return None
        try:
            with open(chunk_path, "r", encoding="utf-8") as f:
                chunk = json.load(f)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        src = chunk.get("source", {}) or {}
        return {
            "chunk_id": chunk.get("chunk_id", chunk_id),
            "library": lib_name,
            "heading": chunk.get("heading", "") or "",
            "text": chunk.get("text", "") or "",
            "text_length": chunk.get("text_length", 0),
            "chunk_seq": chunk.get("chunk_seq"),
            "source_file": src.get("file_name", "") or "",
            "source_file_path": src.get("file_path", "") or "",
        }

    chunks_out = []
    unresolved = []
    seen = set()
    for cid in (r.get("chunk_ids") or []):
        cid = (cid or "").strip()
        if not cid or cid in seen:
            continue
        seen.add(cid)
        candidates = ref_libs.get(cid) or []
        found = None
        for lib_name in candidates + report_libs:
            if not lib_name:
                continue
            found = _read_chunk(lib_name, cid)
            if found is not None:
                break
        if found is not None:
            chunks_out.append(found)
        else:
            unresolved.append(cid)

    return _ok({
        "report_id": report_id,
        "chunks": chunks_out,
        "unresolved": unresolved,
    })


def handle_delete_library(method, path, query, body, name, user=None):
    if method != "DELETE":
        return _err("需要 DELETE 方法", 405)
    reg = _registry()
    lib, err = _get_lib(reg, name, user, write=True)
    if err:
        return err
    # 删除前先关闭该库所有 zone 的 mmap 缓存，避免 rmtree 时文件被占用（WinError 32）
    from indexer import ZoneIndex
    mgr = lib.manager(BASE_DIR)
    for z in mgr.list_zones():
        ZoneIndex.invalidate(z.index_dir)
    # 同步清理语义向量索引的内存缓存与磁盘文件，避免 rmtree 时 faiss 文件被占用
    lib_root = lib.abs_path(BASE_DIR)
    try:
        from semantic_manager import get_manager
        get_manager(BASE_DIR).remove_files(lib_root)
    except Exception as e:
        _logger.warning("清理语义索引失败 %s: %s", name, e)
    try:
        lib = reg.remove(lib.name, delete_data=True, owner=lib.owner)
    except ValueError as e:
        return _err(str(e), 404)
    except OSError as e:
        return _err(f"删除失败（文件被占用）: {e}", 500)
    return _ok({"deleted": lib.name, "path": lib.path})


def handle_stats(method, path, query, body, user=None):
    reg = _registry()
    lib_name = query.get("library")
    if lib_name:
        lib, err = _get_lib(reg, lib_name, user, write=False)
        if err:
            return err
        libs_to_show = [lib]
    else:
        libs_to_show = reg.list_libraries_for(user["username"], user["is_admin"])

    out = []
    for lib in libs_to_show:
        mgr = lib.manager(BASE_DIR)
        s = mgr.stats()
        # 读取文档级元数据缓存（朝代/主题/实体密度）
        doc_stats = {}
        doc_stats_path = os.path.join(mgr.root, "_doc_stats.json")
        if os.path.isfile(doc_stats_path):
            try:
                with open(doc_stats_path, "r", encoding="utf-8") as f:
                    doc_stats = json.load(f)
            except Exception:
                pass
        out.append({
            "name": lib.name,
            "note": lib.note,
            "path": lib.abs_path(BASE_DIR),
            "owner": lib.owner,
            "zone_count": s["zone_count"],
            "total_chars": s["total_chars"],
            "total_chunks": s["total_chunks"],
            "total_sources": s["total_sources"],
            "zones": s["zones"],
            "doc_stats": doc_stats,
        })
    return _ok({"libraries": out})


def handle_search(method, path, query, body, user=None):
    from searcher import (parallel_search, search_related_keywords,
                          parallel_search_fused, search_semantic_groups,
                          search_semantic_groups_parent)
    q = query.get("query", "")
    lib_param = query.get("libraries", "")
    library_names = [x for x in lib_param.split(",") if x] if lib_param else None
    try:
        parallel = int(query.get("parallel", "4"))
    except ValueError:
        parallel = 4
    try:
        top = int(query.get("top", "20"))
    except ValueError:
        top = 20
    # chunk_level：检索粒度切换
    #   child（默认）= 小chunk模式，子片段级语义检索，精确定位
    #   parent = 大chunk模式，父chunk级语义检索，宏观定位
    chunk_level = query.get("chunk_level", body.get("chunk_level") if isinstance(body, dict) else "child")
    if chunk_level not in ("child", "parent"):
        chunk_level = "child"

    # ===== 新版查询语法（前端 parseQueryGroups 解析）=====
    # 1. 纯关键词查询（query 参数）→ 纯字面匹配，不融合语义
    # 2. 结构化查询（keyword_groups + semantic_groups + title_groups）：
    #    - 仅 keyword_groups → 纯关键词同义词组检索
    #    - 仅 semantic_groups → 纯语义同义词组检索
    #    - 仅 title_groups（{} 语法）→ 标题检索（heading/文件名匹配）
    #    - 多者混合 → 组合检索，title_groups 作为标题过滤条件
    keyword_groups_raw = ""
    semantic_groups_raw = ""
    title_groups_raw = ""
    if isinstance(body, dict):
        keyword_groups_raw = body.get("keyword_groups", "")
        semantic_groups_raw = body.get("semantic_groups", "")
        title_groups_raw = body.get("title_groups", "")
    if not keyword_groups_raw:
        keyword_groups_raw = query.get("keyword_groups", "")
    if not semantic_groups_raw:
        semantic_groups_raw = query.get("semantic_groups", "")
    if not title_groups_raw:
        title_groups_raw = query.get("title_groups", "")

    # 兼容旧版 groups 参数（保留向后兼容）
    if not keyword_groups_raw and not semantic_groups_raw:
        groups_param = None
        if isinstance(body, dict) and body.get("groups"):
            groups_param = body.get("groups")
        else:
            gp_raw = query.get("groups", "")
            if gp_raw:
                try:
                    groups_param = json.loads(gp_raw)
                except (json.JSONDecodeError, TypeError):
                    groups_param = None
        if groups_param:
            keyword_groups_raw = json.dumps(groups_param)

    if not q and not keyword_groups_raw and not semantic_groups_raw and not title_groups_raw:
        return _err("缺少 query 或 keyword_groups/semantic_groups/title_groups 参数")

    # 解析 keyword_groups、semantic_groups 和 title_groups
    keyword_groups = None
    semantic_groups = None
    title_groups = None
    if keyword_groups_raw:
        try:
            parsed = json.loads(keyword_groups_raw) if isinstance(keyword_groups_raw, str) else keyword_groups_raw
            if isinstance(parsed, list):
                keyword_groups = [[str(x).strip() for x in g if str(x).strip()]
                                  for g in parsed if isinstance(g, list) and g]
                keyword_groups = [g for g in keyword_groups if g]
        except (json.JSONDecodeError, TypeError):
            pass
    if semantic_groups_raw:
        try:
            parsed = json.loads(semantic_groups_raw) if isinstance(semantic_groups_raw, str) else semantic_groups_raw
            if isinstance(parsed, list):
                semantic_groups = [[str(x).strip() for x in g if str(x).strip()]
                                   for g in parsed if isinstance(g, list) and g]
                semantic_groups = [g for g in semantic_groups if g]
        except (json.JSONDecodeError, TypeError):
            pass
    if title_groups_raw:
        try:
            parsed = json.loads(title_groups_raw) if isinstance(title_groups_raw, str) else title_groups_raw
            if isinstance(parsed, list):
                title_groups = [[str(x).strip() for x in g if str(x).strip()]
                                for g in parsed if isinstance(g, list) and g]
                title_groups = [g for g in title_groups if g]
        except (json.JSONDecodeError, TypeError):
            pass

    reg = _registry()
    # 多用户：library_names=None（搜全部）时收敛为当前用户可见的库；
    # 显式指定时校验每个库对当前用户可见
    if library_names:
        err_resp = _check_libraries_visible(reg, library_names, user)
        if err_resp:
            return err_resp
    else:
        library_names = [l.name for l in reg.list_libraries_for(user["username"], user["is_admin"])]
    try:
        # 分支 0：仅标题限定组（{} 语法单独使用）→ 标题检索
        if title_groups and not q and not keyword_groups and not semantic_groups:
            from searcher import search_by_titles
            result = search_by_titles(
                reg, title_groups,
                library_names=library_names,
                base_dir=BASE_DIR,
                top_k=top,
            )
            return _ok(result)

        # 分支 1：纯关键词查询（无任何括号语法）→ 纯字面匹配
        if q and not keyword_groups and not semantic_groups:
            # 标题限定（{} 与普通关键词混用时由前端拆分，此处兜底：
            # 未传 title_groups 时从 query 中解析 {}，并从 query 剥离只作过滤条件）
            if not title_groups:
                from searcher import _parse_title_groups_py
                title_groups = _parse_title_groups_py(q)
            if title_groups:
                from searcher import strip_title_groups_py
                q_search = strip_title_groups_py(q).strip()
                if not q_search:
                    # query 仅含 {}：转标题检索
                    from searcher import search_by_titles
                    result = search_by_titles(
                        reg, title_groups,
                        library_names=library_names,
                        base_dir=BASE_DIR,
                        top_k=top,
                    )
                    return _ok(result)
                q = q_search
            result = parallel_search(
                reg, q,
                library_names=library_names,
                parallel=parallel,
                base_dir=BASE_DIR,
            )
            # 标题限定（{} 与普通关键词混用时由前端拆分，此处兜底）
            if title_groups:
                from searcher import apply_title_filter
                result["results"] = apply_title_filter(result["results"], title_groups)
                result["total_hits"] = len(result["results"])
                result["title_groups"] = title_groups
            # 标注语义通道可用性（前端展示用，但不参与检索）
            try:
                from semantic_manager import get_manager
                mgr_sem = get_manager(BASE_DIR)
                result["semantic_available"] = mgr_sem.available()
            except Exception:
                result["semantic_available"] = False
            result["results"] = result["results"][:top]
            return _ok(result)

        # 分支 2：含 semantic_groups（[] 语法）的混合查询
        if semantic_groups:
            # 2a: 先做语义同义词组检索（根据 chunk_level 选择粒度）
            if chunk_level == "parent":
                sem_result = search_semantic_groups_parent(
                    reg, semantic_groups,
                    library_names=library_names,
                    parallel=parallel,
                    base_dir=BASE_DIR,
                    top_k=top * 3,
                )
            else:
                sem_result = search_semantic_groups(
                    reg, semantic_groups,
                    library_names=library_names,
                    parallel=parallel,
                    base_dir=BASE_DIR,
                    top_k=top * 3,  # 多取一些，便于和关键词取交集后仍够
                )

            if not sem_result.get("semantic_available", False):
                # 语义通道不可用：返回空结果 + 提示
                return _ok(sem_result)

            # 2b: 若同时有关键词组，做关键词检索并取交集
            if keyword_groups:
                kw_result = search_related_keywords(
                    reg, keyword_groups,
                    library_names=library_names,
                    parallel=parallel,
                    base_dir=BASE_DIR,
                    top_k=top * 3,
                )
                # 按 chunk_id 取交集
                kw_cids = {r["chunk_id"] for r in kw_result.get("results", [])}
                sem_results_filtered = [r for r in sem_result.get("results", [])
                                        if r["chunk_id"] in kw_cids]
                # 合并关键词命中信息到语义结果
                kw_map = {r["chunk_id"]: r for r in kw_result.get("results", [])}
                for r in sem_results_filtered:
                    kw = kw_map.get(r["chunk_id"])
                    if kw:
                        r["matched_words"] = kw.get("matched_words", [])
                        r["hit_count"] = kw.get("hit_count", 0)
                        r["channels"] = ["keyword", "semantic"]
                    else:
                        r["channels"] = ["semantic"]
                # 按语义相似度降序
                sem_results_filtered.sort(
                    key=lambda x: x.get("semantic_score", 0), reverse=True)
                # 标题限定（{} 语法）：交集结果按标题过滤
                if title_groups:
                    from searcher import apply_title_filter
                    sem_results_filtered = apply_title_filter(
                        sem_results_filtered, title_groups)
                sem_result["results"] = sem_results_filtered[:top]
                sem_result["total_hits"] = len(sem_results_filtered)
                sem_result["mode"] = "keyword_and_semantic"
                sem_result["keyword_groups"] = keyword_groups
                if title_groups:
                    sem_result["title_groups"] = title_groups
                # 合并 searched_libraries 关键词命中数
                kw_libs_map = {l["name"]: l for l in kw_result.get("searched_libraries", [])}
                for l in sem_result.get("searched_libraries", []):
                    kl = kw_libs_map.get(l["name"])
                    if kl:
                        l["keyword_hits"] = kl.get("hits", 0)
                return _ok(sem_result)
            else:
                # 2c: 仅语义同义词组检索（纯 [] 语法）
                if title_groups:
                    from searcher import apply_title_filter
                    sem_result["results"] = apply_title_filter(
                        sem_result.get("results", []), title_groups)
                    sem_result["total_hits"] = len(sem_result["results"])
                    sem_result["title_groups"] = title_groups
                sem_result["results"] = sem_result["results"][:top]
                return _ok(sem_result)

        # 分支 3：仅关键词同义词组（() 语法或多词空格）
        if keyword_groups:
            result = search_related_keywords(
                reg, keyword_groups,
                library_names=library_names,
                parallel=parallel,
                base_dir=BASE_DIR,
                # 标题过滤会丢结果，多取一些保证过滤后仍有 top 条
                top_k=top * 3 if title_groups else top,
            )
            if title_groups:
                from searcher import apply_title_filter
                result["results"] = apply_title_filter(result["results"], title_groups)
                result["total_hits"] = len(result["results"])
                result["title_groups"] = title_groups
            try:
                from semantic_manager import get_manager
                mgr_sem = get_manager(BASE_DIR)
                result["semantic_available"] = mgr_sem.available()
            except Exception:
                result["semantic_available"] = False
            result["results"] = result["results"][:top]
            return _ok(result)

        # 兜底（理论上不会走到）
        return _err("无法识别的查询参数组合")
    except ValueError as e:
        return _err(str(e))


def handle_semantic_search(method, path, query, body, user=None):
    """GET /api/semantic/search —— 纯语义向量检索（不走关键词检索）。

    用于在检索页单独验证向量召回效果，与 /api/search 的五路融合结果对比。

    参数：
        query=xxx        查询文本（自然语言句子或关键词均可）
        libraries=A,B    限定查询的库名（不传则查全部）
        parallel=4       并行度
        top=20           返回条数

    返回：与 /api/search 兼容的结构，额外标注 mode="semantic"
    """
    from searcher import search_semantic, search_semantic_parent
    from semantic_manager import get_manager
    q = query.get("query", "").strip()
    if not q:
        return _err("缺少 query 参数")
    lib_param = query.get("libraries", "")
    library_names = [x for x in lib_param.split(",") if x] if lib_param else None
    try:
        parallel = int(query.get("parallel", "4"))
    except ValueError:
        parallel = 4
    try:
        top = int(query.get("top", "20"))
    except ValueError:
        top = 20
    # chunk_level：检索粒度切换（child=小chunk子片段级，parent=大chunk父级）
    chunk_level = query.get("chunk_level", "child")
    if chunk_level not in ("child", "parent"):
        chunk_level = "child"

    reg = _registry()
    # 多用户：只检索当前用户可见的库
    if library_names:
        err_resp = _check_libraries_visible(reg, library_names, user)
        if err_resp:
            return err_resp
    else:
        library_names = [l.name for l in reg.list_libraries_for(user["username"], user["is_admin"])]
    # 执行前先检查目标库的向量索引状态
    # 若所有库都未就绪，直接返回不可用提示，避免空结果误导用户
    target_libs = [l for l in reg.list_libraries_for(user["username"], user["is_admin"])
                   if library_names is None or l.name in library_names]
    if not target_libs:
        return _ok({
            "query": q, "mode": "semantic",
            "total_hits": 0, "searched_libraries": [],
            "results": [], "elapsed_ms": 0,
            "semantic_available": False,
            "semantic_reason": "未选择任何库",
        })

    mgr_sem = get_manager(BASE_DIR)
    if not mgr_sem.available():
        return _ok({
            "query": q, "mode": "semantic",
            "total_hits": 0, "searched_libraries": [],
            "results": [], "elapsed_ms": 0,
            "semantic_available": False,
            "semantic_reason": mgr_sem.fail_reason(),
        })

    # 逐库检查索引状态：区分未构建 / 构建中 / 就绪
    not_ready_libs = []      # 未构建或构建中
    building_libs = []       # 正在构建中
    for lib in target_libs:
        lib_root = lib.abs_path(BASE_DIR)
        s = mgr_sem.status(lib_root)
        st = s.get("status", "idle")
        if st != "ready":
            not_ready_libs.append(lib.name)
            if st == "building":
                building_libs.append(lib.name)

    if not_ready_libs:
        # 所有库都未就绪 → 返回不可用提示
        if len(not_ready_libs) == len(target_libs):
            if building_libs:
                reason = f"所选库的向量索引正在构建中：{', '.join(building_libs)}。请等待构建完成后再试"
            else:
                reason = f"所选库尚未构建向量索引：{', '.join(not_ready_libs)}。请先在库管理页点击「构建向量索引」"
            return _ok({
                "query": q, "mode": "semantic",
                "total_hits": 0, "searched_libraries": [],
                "results": [], "elapsed_ms": 0,
                "semantic_available": False,
                "semantic_reason": reason,
            })
        # 部分库未就绪 → 仍执行，但在结果中标注（前端可展示）
        # 这里不阻断，让 search_semantic 正常执行（未就绪的库会返回空结果）

    try:
        if chunk_level == "parent":
            result = search_semantic_parent(
                reg, q, BASE_DIR,
                library_names=library_names,
                parallel=parallel,
                top_k=top,
            )
        else:
            result = search_semantic(
                reg, q, BASE_DIR,
                library_names=library_names,
                parallel=parallel,
                top_k=top,
            )
    except ValueError as e:
        return _err(str(e))
    # 截断到 top
    result["results"] = result["results"][:top]
    # 附加部分未就绪提示（若有）
    if not_ready_libs and result.get("semantic_available", True):
        result["partial_unavailable"] = not_ready_libs
    return _ok(result)


def handle_semantic_status(method, path, query, body, user=None):
    """GET /api/semantic/status —— 查询语义检索通道状态。

    可选参数：
        library=xxx  仅查指定库（不传则查所有库）

    返回：
        {
            "available": true/false,         # 通道是否可用（依赖+开关）
            "fail_reason": "",                # 不可用原因
            "libraries": [
                {
                    "library": "郎溪县志",
                    "lib_root": "C:/.../郎溪县志",
                    "status": "ready|building|idle|failed|unavailable",
                    "vector_count": 105000,
                    "progress": {...},         # 构建中时的进度
                    "fail_reason": ""
                }, ...
            ]
        }
    """
    try:
        from semantic_manager import get_manager
    except Exception as e:
        return _ok({"available": False, "fail_reason": f"语义模块加载失败：{e}",
                    "libraries": []})

    mgr_sem = get_manager(BASE_DIR)
    available = mgr_sem.available()
    fail_reason = mgr_sem.fail_reason() if not available else ""

    # 收集所有库的状态（仅当前用户可见的库）
    reg = _registry()
    lib_name = query.get("library")
    if lib_name:
        libs = [l for l in reg.list_libraries_for(user["username"], user["is_admin"])
                if l.name == lib_name]
    else:
        libs = reg.list_libraries_for(user["username"], user["is_admin"])

    out_libs = []
    for lib in libs:
        lib_root = lib.abs_path(BASE_DIR)
        s = mgr_sem.status(lib_root)
        out_libs.append({
            "library": lib.name,
            "lib_root": lib_root,
            "status": s.get("status", "idle"),
            "vector_count": s.get("vector_count", 0),
            "progress": s.get("progress", {}),
            "fail_reason": s.get("fail_reason", ""),
            "enabled": s.get("enabled", True),
        })

    return _ok({
        "available": available,
        "fail_reason": fail_reason,
        "libraries": out_libs,
    })


def handle_semantic_build(method, path, query, body, user=None):
    """POST /api/semantic/build —— 手动触发某库的语义索引构建。

    body: {"library": "xxx"}
    返回：{"triggered": true/false, "reason": "..."}
    """
    if method != "POST":
        return _err("需要 POST 方法", 405)
    lib_name = body.get("library")
    if not lib_name:
        return _err("缺少 library 参数")
    reg = _registry()
    lib, err = _get_lib(reg, lib_name, user, write=True)
    if err:
        return err

    try:
        from semantic_manager import get_manager
        from settings import SettingsStore
        store = SettingsStore(AUTH_DIR)
        auto_build = store.get("semantic_auto_build", True)
        if not store.get("semantic_enabled", True):
            return _ok({"triggered": False,
                        "reason": "语义检索通道已在设置中关闭"})
        mgr_sem = get_manager(BASE_DIR)
        if not mgr_sem.available():
            return _ok({"triggered": False,
                        "reason": mgr_sem.fail_reason()})
        ret = mgr_sem.trigger_build_async(
            lib.abs_path(BASE_DIR), lib_name=lib.name)
        return _ok({
            "triggered": ret.get("started", False),
            "reason": ret.get("reason", ""),
        })
    except Exception as e:
        return _err(f"触发构建失败：{e}", 500)


def handle_import(method, path, query, body, user=None):
    from importer import import_file
    from transaction import recover_all_zones
    if method != "POST":
        return _err("需要 POST 方法", 405)
    files = body.get("files", [])
    lib_name = body.get("library")
    force = body.get("force", False)
    if not files or not lib_name:
        return _err("缺少 files 或 library 参数")

    reg = _registry()
    lib, err = _get_lib(reg, lib_name, user, write=True)
    if err:
        return err
    mgr = lib.manager(BASE_DIR)

    lock = _lib_import_lock(mgr.root)
    with lock:
        recover_all_zones(mgr)

        # 展开目录，记录每个文件的 import_root（用于保留目录结构元数据）
        expanded = []
        file_import_roots = {}  # file -> import_root
        from extractor import supported
        for f in files:
            if os.path.isdir(f):
                for root, _d, names in os.walk(f):
                    for n in sorted(names):
                        full = os.path.join(root, n)
                        if supported(full):
                            expanded.append(full)
                            file_import_roots[full] = f
            elif os.path.isfile(f):
                expanded.append(f)

        results = []
        ok_n = skip_n = fail_n = 0
        # 批量导入：跳过每个文件的 merge，最后统一合并一次
        from indexer import ZoneIndex
        touched_zones = []  # 记录本次导入涉及到的 zone (index_dir, chunks_dir, zone_id)
        seen_zones = set()
        for f in expanded:
            r = import_file(mgr, f, force=force, base_dir=BASE_DIR, skip_merge=True,
                            import_root=file_import_roots.get(f))
            if r.get("ok"):
                ok_n += 1
                # 记录涉及的 zone，用于最后统一 merge
                try:
                    zone = mgr.get_zone(r["zone_id"])
                    if zone and zone.index_dir not in seen_zones:
                        seen_zones.add(zone.index_dir)
                        touched_zones.append((zone.index_dir, zone.chunks_dir, zone.zone_id))
                except Exception:
                    pass
                results.append({"file": f, "status": "ok",
                                "zone": r["zone_id"], "chunks": r["chunks_written"],
                                "chars": r["char_count"]})
            elif r.get("skipped"):
                skip_n += 1
                results.append({"file": f, "status": "skipped"})
            else:
                fail_n += 1
                results.append({"file": f, "status": "fail",
                                "error": r.get("error", "未知错误")})

        # 批量结束：对每个涉及到的 zone 统一合并一次索引
        merge_total = 0
        for index_dir, chunks_dir, zid in touched_zones:
            try:
                zi = ZoneIndex.get(index_dir)
                zi._batch_mode = False  # 确保非批量模式
                ms = zi.merge_zone_chunks(chunks_dir, zid)
                zi.cleanup_merged_idx(chunks_dir)
                merge_total += ms.get("merged", 0)
            except Exception as e:
                _logger.warning("merge 索引失败 %s: %s", index_dir, e)

        return _ok({"ok": ok_n, "skipped": skip_n, "fail": fail_n,
                    "total": len(expanded), "details": results,
                    "index_merged": merge_total})


def handle_upload_import(handler, user=None):
    """处理 multipart/form-data 文件上传 + 导入。

    表单字段：
        library   : 库名
        force     : "1" / "0"
        batch_id  : 可选，批次 ID（用于撤销）
        files[]   : 一个或多个文件

    上传的文件保存到临时目录，然后调用 import_file 导入，完成后清理临时文件。
    """
    from importer import import_file
    from transaction import recover_all_zones
    import tempfile
    import shutil
    from extractor import supported, SUPPORTED_EXTS

    ctype = handler.headers.get("Content-Type", "")
    if not ctype.startswith("multipart/form-data"):
        return _err("需要 multipart/form-data 请求", 415)

    # 简易 multipart 解析
    boundary = ctype.split("boundary=", 1)[1] if "boundary=" in ctype else ""
    if not boundary:
        return _err("缺少 boundary", 400)
    boundary_bytes = ("--" + boundary).encode("utf-8")

    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return _err("空请求体", 400)
    raw = handler.rfile.read(length)

    # 拆分 multipart parts
    parts = raw.split(boundary_bytes)
    lib_name = None
    force = False
    batch_id = ""
    uploaded = []  # [(filename, content_bytes)]

    for part in parts:
        if part in (b"", b"--", b"--\r\n", b"\r\n"):
            continue
        # 去掉开头的 \r\n
        if part.startswith(b"\r\n"):
            part = part[2:]
        # 去掉结尾的 \r\n
        if part.endswith(b"\r\n"):
            part = part[:-2]
        # 分离头部和内容
        try:
            header_block, content = part.split(b"\r\n\r\n", 1)
        except ValueError:
            continue
        header_str = header_block.decode("utf-8", errors="ignore")
        # 解析 Content-Disposition
        name = None
        filename = None
        for line in header_str.split("\r\n"):
            if line.lower().startswith("content-disposition:"):
                # 提取 name="..." 和 filename="..."
                import re as _re
                m_name = _re.search(r'name="([^"]*)"', line)
                m_file = _re.search(r'filename="([^"]*)"', line)
                if m_name:
                    name = m_name.group(1)
                if m_file:
                    filename = m_file.group(1)
        if name is None:
            continue
        if name == "library":
            lib_name = content.decode("utf-8", errors="ignore").strip()
        elif name == "force":
            force = content.decode("utf-8", errors="ignore").strip() in ("1", "true", "True")
        elif name == "batch_id":
            batch_id = content.decode("utf-8", errors="ignore").strip()
        elif name == "files" and filename:
            uploaded.append((filename, content))

    if not lib_name:
        return _err("缺少 library 参数")
    if not uploaded:
        return _err("未收到任何文件")

    reg = _registry()
    lib, err = _get_lib(reg, lib_name, user, write=True)
    if err:
        return err
    mgr = lib.manager(BASE_DIR)

    recover_all_zones(mgr)

    # 注册批次（用于撤销）
    if batch_id:
        with _import_batches_lock:
            _import_batches[batch_id] = {
                "cancelled": False,
                "shas": [],
                "library": lib_name,
                "total": len(uploaded),
            }

    # 保存到临时目录再导入
    tmp_dir = tempfile.mkdtemp(prefix="upload_")
    results = []
    ok_n = skip_n = fail_n = 0
    used_names = set()  # 已用文件名（小写），避免重名冲突
    # 批量导入：记录涉及的 zone，最后统一合并一次
    from indexer import ZoneIndex
    touched_zones = []
    seen_zones = set()
    batch_shas = []
    try:
        for filename, content in uploaded:
            # 检查撤销标志
            if batch_id:
                with _import_batches_lock:
                    if _import_batches.get(batch_id, {}).get("cancelled"):
                        # 撤销已 commit 的文件
                        _rollback_batch(mgr, batch_shas)
                        results.append({"file": filename, "status": "cancelled"})
                        continue
            # 安全文件名：只取 basename，避免路径穿越
            safe_base = os.path.basename(filename.replace("\\", "/"))
            # 跳过不支持的扩展名
            ext = os.path.splitext(safe_base)[1].lower()
            if ext not in SUPPORTED_EXTS:
                results.append({"file": filename, "status": "fail",
                                "error": f"不支持的类型: {ext or '(无扩展名)'}"})
                fail_n += 1
                continue
            # 处理重名：同名文件加序号后缀
            base_name, ext_name = os.path.splitext(safe_base)
            final_name = safe_base
            counter = 1
            while final_name.lower() in used_names:
                final_name = f"{base_name}_{counter}{ext_name}"
                counter += 1
            used_names.add(final_name.lower())
            tmp_path = os.path.join(tmp_dir, final_name)
            with open(tmp_path, "wb") as f:
                f.write(content)
            r = import_file(mgr, tmp_path, force=force, base_dir=tmp_dir, skip_merge=True)
            if r.get("ok"):
                ok_n += 1
                sha = r.get("source_sha256", "")
                if sha and batch_id:
                    batch_shas.append(sha)
                # 记录涉及的 zone，用于最后统一 merge
                try:
                    zone = mgr.get_zone(r["zone_id"])
                    if zone and zone.index_dir not in seen_zones:
                        seen_zones.add(zone.index_dir)
                        touched_zones.append((zone.index_dir, zone.chunks_dir, zone.zone_id))
                except Exception:
                    pass
                results.append({"file": filename, "status": "ok",
                                "zone": r["zone_id"], "chunks": r["chunks_written"],
                                "chars": r["char_count"],
                                "source_sha256": sha})
            elif r.get("skipped"):
                skip_n += 1
                results.append({"file": filename, "status": "skipped"})
            else:
                fail_n += 1
                results.append({"file": filename, "status": "fail",
                                "error": r.get("error", "未知错误")})
    finally:
        # 批量结束：先合并索引，再清理临时目录
        merge_total = 0
        for index_dir, chunks_dir, zid in touched_zones:
            try:
                zi = ZoneIndex.get(index_dir)
                zi._batch_mode = False
                ms = zi.merge_zone_chunks(chunks_dir, zid)
                zi.cleanup_merged_idx(chunks_dir)
                merge_total += ms.get("merged", 0)
            except Exception as e:
                _logger.warning("merge 索引失败 %s: %s", index_dir, e)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        # 更新批次 SHA 记录
        if batch_id:
            with _import_batches_lock:
                if batch_id in _import_batches:
                    _import_batches[batch_id]["shas"] = batch_shas

    return _ok({"ok": ok_n, "skipped": skip_n, "fail": fail_n,
                "total": len(uploaded), "details": results,
                "index_merged": merge_total, "batch_id": batch_id,
                "shas": batch_shas})


# ============================================================
#  导入流式进度 + 撤销
# ============================================================

import threading

# 活跃导入批次注册表：batch_id -> {cancelled, shas, library, mgr_root}
_import_batches: Dict[str, Dict[str, Any]] = {}
_import_batches_lock = threading.Lock()

# 持久化文件路径（用于服务器意外关闭后恢复/清理）
_IMPORT_BATCHES_FILE = "_import_batches.json"


def _persist_import_batches() -> None:
    """把当前所有未完成的导入批次持久化到磁盘。

    服务器意外关闭后，下次启动可通过此文件清理未完成的批次。
    已 done/cancelled 的批次不写入。
    """
    try:
        with _import_batches_lock:
            active = {
                bid: {
                    "library": b.get("library", ""),
                    "shas": list(b.get("shas", [])),
                    "total": b.get("total", 0),
                    "current": b.get("current", 0),
                    "cancelled": b.get("cancelled", False),
                    "done": b.get("done", False),
                    "started_at": b.get("started_at", 0),
                }
                for bid, b in _import_batches.items()
                if not b.get("done", False) and not b.get("cancelled", False)
            }
        path = os.path.join(BASE_DIR, _IMPORT_BATCHES_FILE)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(active, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        _logger.warning("持久化导入批次失败: %s", e)


def _load_and_cleanup_stale_import_batches() -> None:
    """服务器启动时调用：加载持久化文件，回滚所有未完成的批次。

    未完成 = 文件中记录的批次（服务器关闭时未 done/cancelled）。
    回滚 = 调用 _rollback_batch 删除已导入的 shas。
    """
    path = os.path.join(BASE_DIR, _IMPORT_BATCHES_FILE)
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        _logger.warning("读取持久化批次文件失败: %s", e)
        try:
            os.remove(path)
        except Exception:
            pass
        return
    if not data:
        try:
            os.remove(path)
        except Exception:
            pass
        return
    _logger.info("[启动清理] 检测到 %d 个未完成的导入批次，开始回滚...", len(data))
    from remover import remove_by_sha
    from indexer import ZoneIndex
    reg = _registry()
    for bid, b in data.items():
        lib_name = b.get("library", "")
        shas = b.get("shas", []) or []
        if not lib_name or not shas:
            continue
        lib = reg.get_library(lib_name)
        if lib is None:
            # 库已被删除，无需回滚
            continue
        mgr = lib.manager(BASE_DIR)
        cleaned = 0
        for sha in shas:
            try:
                remove_by_sha(mgr, sha)
                cleaned += 1
            except Exception as e:
                _logger.warning("[启动清理] 回滚 sha=%s... 失败: %s", sha[:12], e)
        # 失效所有 zone 索引缓存
        for z in mgr.list_zones():
            ZoneIndex.invalidate(z.index_dir)
        _logger.info("[启动清理] 批次 %s（库 %s）回滚 %d/%d 个文件", bid, lib_name, cleaned, len(shas))
    # 清理持久化文件
    try:
        os.remove(path)
    except Exception:
        pass
    _logger.info("[启动清理] 完成")


def _new_batch_id() -> str:
    import uuid
    return f"batch_{uuid.uuid4().hex[:12]}"


def handle_import_stream(handler, query, user=None):
    """GET /api/import/stream —— SSE 流式导入，推送进度 + 支持撤销。

    参数（query）：
        library      : 库名
        force        : "1"/"0"
        files        : JSON 编码的文件路径数组（path 模式）
        batch_id     : 客户端预生成的 batch_id（用于撤销）

    事件 phase：
        start    : {batch_id, total}
        progress : {current, total, file, status, chars}
        merging  : {zones}
        done     : {ok, skipped, fail, total, batch_id, shas}
        error    : {message}
    """
    import json as _json
    lib_name = (query.get("library") or "").strip()
    if not lib_name:
        handler._send(400, {"ok": False, "error": "缺少 library 参数"})
        return
    files_json = query.get("files", "[]")
    try:
        files = _json.loads(files_json) if isinstance(files_json, str) else files_json
    except _json.JSONDecodeError:
        files = []
    if not files:
        handler._send(400, {"ok": False, "error": "缺少 files 参数"})
        return
    force = query.get("force", "0") in ("1", "true", "True")
    batch_id = (query.get("batch_id") or "").strip() or _new_batch_id()

    reg = _registry()
    lib, err = _get_lib(reg, lib_name, user, write=True)
    if err:
        handler._send(err[0], err[1])
        return
    mgr = lib.manager(BASE_DIR)

    # 展开目录，记录每个文件的 import_root（用于保留目录结构元数据）
    from extractor import supported
    expanded = []
    file_import_roots = {}  # file -> import_root
    for f in files:
        if os.path.isdir(f):
            for root, _d, names in os.walk(f):
                for n in sorted(names):
                    full = os.path.join(root, n)
                    if supported(full):
                        expanded.append(full)
                        file_import_roots[full] = f
        elif os.path.isfile(f):
            expanded.append(f)

    if not expanded:
        handler._send(400, {"ok": False, "error": "没有可导入的文件"})
        return

    # 注册批次（用于撤销）
    with _import_batches_lock:
        _import_batches[batch_id] = {
            "cancelled": False,
            "shas": [],
            "library": lib_name,
            "total": len(expanded),
            "current": 0,
            "done": False,
            "started_at": time.time(),
        }
    # 持久化（服务器意外关闭后可清理）
    _persist_import_batches()

    gen = _build_import_stream_gen(mgr, expanded, force, batch_id, os.getcwd(),
                                   lib_name=lib_name, file_import_roots=file_import_roots)
    handler._send_sse_stream(gen())


def _trigger_semantic_build_after_import(lib_name: str, lib_root: str) -> Dict[str, Any]:
    """导入完成后异步触发语义向量索引构建。

    - 检查 settings.semantic_enabled 和 semantic_auto_build 开关
    - 依赖未装 / 已在构建中 / 索引已就绪时静默跳过
    - 构建在后台 daemon 线程执行，不阻塞当前请求
    - 返回触发结果（前端展示用）
    """
    try:
        from settings import SettingsStore
        store = SettingsStore(AUTH_DIR)
        if not store.get("semantic_enabled", True):
            return {"triggered": False, "reason": "语义检索通道已在设置中关闭"}
        if not store.get("semantic_auto_build", True):
            return {"triggered": False, "reason": "已禁用自动构建（可在设置中开启）"}
        from semantic_manager import get_manager
        mgr_sem = get_manager(BASE_DIR)
        if not mgr_sem.available():
            return {"triggered": False, "reason": mgr_sem.fail_reason()}
        ret = mgr_sem.trigger_build_async(lib_root, lib_name=lib_name)
        return {
            "triggered": ret.get("started", False),
            "reason": ret.get("reason", ""),
        }
    except Exception as e:
        return {"triggered": False, "reason": f"触发构建异常：{e}"}


def _build_import_stream_gen(mgr, files, force, batch_id, base_dir, lib_name="",
                             file_import_roots=None):
    """构造导入流式生成器（path 和 upload 共用）。

    files: 待导入文件绝对路径列表
    base_dir: import_file 的 base_dir 参数
    lib_name: 库名（用于触发语义索引构建，可选）
    file_import_roots: {file_path: import_root} 映射，用于保留目录结构元数据
    """
    def gen():
        from importer import import_file
        from transaction import recover_all_zones
        from indexer import ZoneIndex
        recover_all_zones(mgr)
        touched_zones = []
        seen_zones = set()
        ok_n = skip_n = fail_n = 0
        results = []
        batch_shas = []
        last_persist_count = 0  # 上次持久化时的 shas 数量
        try:
            yield {"phase": "start", "batch_id": batch_id, "total": len(files)}
            for i, f in enumerate(files):
                # 更新批次进度（供刷新页面后查询）
                with _import_batches_lock:
                    if batch_id in _import_batches:
                        _import_batches[batch_id]["current"] = i
                        _import_batches[batch_id]["shas"] = list(batch_shas)
                # 检查撤销标志
                with _import_batches_lock:
                    if _import_batches.get(batch_id, {}).get("cancelled"):
                        yield {"phase": "cancelled", "batch_id": batch_id,
                               "imported_shas": batch_shas,
                               "message": "已撤销导入，正在清理..."}
                        # 撤销已 commit 的文件
                        _rollback_batch(mgr, batch_shas)
                        with _import_batches_lock:
                            if batch_id in _import_batches:
                                _import_batches[batch_id]["done"] = True
                        # 撤销完成，从持久化中清除
                        _persist_import_batches()
                        yield {"phase": "done", "batch_id": batch_id,
                               "cancelled": True, "imported_shas": batch_shas}
                        return
                yield {"phase": "progress", "current": i + 1, "total": len(files),
                       "file": os.path.basename(f), "status": "processing"}
                r = import_file(mgr, f, force=force, base_dir=base_dir, skip_merge=True,
                                import_root=(file_import_roots or {}).get(f))
                if r.get("ok"):
                    ok_n += 1
                    sha = r.get("source_sha256", "")
                    if sha:
                        batch_shas.append(sha)
                    try:
                        zone = mgr.get_zone(r["zone_id"])
                        if zone and zone.index_dir not in seen_zones:
                            seen_zones.add(zone.index_dir)
                            touched_zones.append((zone.index_dir, zone.chunks_dir, zone.zone_id))
                    except Exception:
                        pass
                    results.append({"file": f, "status": "ok",
                                    "zone": r["zone_id"], "chunks": r["chunks_written"],
                                    "chars": r["char_count"]})
                    yield {"phase": "progress", "current": i + 1, "total": len(files),
                           "file": os.path.basename(f), "status": "ok", "chars": r.get("char_count", 0)}
                elif r.get("skipped"):
                    skip_n += 1
                    results.append({"file": f, "status": "skipped"})
                    yield {"phase": "progress", "current": i + 1, "total": len(files),
                           "file": os.path.basename(f), "status": "skipped"}
                else:
                    fail_n += 1
                    results.append({"file": f, "status": "fail",
                                    "error": r.get("error", "未知错误")})
                    yield {"phase": "progress", "current": i + 1, "total": len(files),
                           "file": os.path.basename(f), "status": "fail",
                           "error": r.get("error", "")}
                # 周期性持久化（每导入 10 个文件，或全部完成时）
                if len(batch_shas) - last_persist_count >= 10 or i == len(files) - 1:
                    with _import_batches_lock:
                        if batch_id in _import_batches:
                            _import_batches[batch_id]["shas"] = list(batch_shas)
                            _import_batches[batch_id]["current"] = i + 1
                    _persist_import_batches()
                    last_persist_count = len(batch_shas)
            # 合并索引
            yield {"phase": "merging", "zones": len(touched_zones)}
            merge_total = 0
            import queue as _q
            import threading as _th
            for zi_idx, (index_dir, chunks_dir, zid) in enumerate(touched_zones, 1):
                # 在子线程执行 merge，通过队列实时推送进度事件
                ev_q: _q.Queue = _q.Queue()
                merge_result_holder: Dict[str, Any] = {"ms": None, "err": None}
                def _run_merge(_zi=ZoneIndex.get(index_dir), _cd=chunks_dir,
                               _zid=zid, _q=ev_q, _holder=merge_result_holder,
                               _zi_idx=zi_idx):
                    try:
                        _zi._batch_mode = False
                        def _cb(cur, tot, stage):
                            try:
                                _q.put({"phase": "merging_progress",
                                        "zone_index": _zi_idx, "zone_id": _zid,
                                        "stage": stage, "current": cur, "total": tot})
                            except Exception:
                                pass
                        ms = _zi.merge_zone_chunks(_cd, _zid, progress_callback=_cb)
                        _zi.cleanup_merged_idx(_cd)
                        _holder["ms"] = ms
                    except Exception as e:
                        _holder["err"] = e
                    finally:
                        _q.put(None)  # 结束信号
                th = _th.Thread(target=_run_merge, daemon=True)
                th.start()
                # 实时推送进度事件
                while True:
                    try:
                        ev = ev_q.get(timeout=0.1)
                    except _q.Empty:
                        # 短暂无事件也 yield 一次心跳，保持 SSE 连接活跃
                        yield {"phase": "merging_heartbeat", "zone_index": zi_idx}
                        continue
                    if ev is None:
                        break
                    yield ev
                th.join(timeout=5)
                if merge_result_holder["err"]:
                    _logger.warning("merge 索引失败 %s: %s", index_dir, merge_result_holder['err'])
                elif merge_result_holder["ms"]:
                    merge_total += merge_result_holder["ms"].get("merged", 0)
            # 记录批次 SHA（撤销用）+ 标记完成
            with _import_batches_lock:
                if batch_id in _import_batches:
                    _import_batches[batch_id]["shas"] = batch_shas
                    _import_batches[batch_id]["current"] = len(files)
                    _import_batches[batch_id]["done"] = True
            # 已完成，从持久化文件中清除
            _persist_import_batches()
            # 触发语义向量索引后台构建（不阻塞，索引未就绪前可正常用关键词检索）
            sem_status = _trigger_semantic_build_after_import(
                lib_name, mgr.root,
            ) if lib_name else {"triggered": False, "reason": ""}
            yield {"phase": "done", "batch_id": batch_id,
                   "ok": ok_n, "skipped": skip_n, "fail": fail_n,
                   "total": len(files), "details": results,
                   "index_merged": merge_total, "shas": batch_shas,
                   "semantic_build": sem_status}
        except Exception as e:
            yield {"phase": "error", "message": str(e)}
        finally:
            # 保留批次记录供撤销（done 后仍可撤销）
            with _import_batches_lock:
                if batch_id in _import_batches:
                    _import_batches[batch_id]["shas"] = batch_shas
                    _import_batches[batch_id]["done"] = True
            # 更新持久化（异常退出时也保留状态供启动清理）
            _persist_import_batches()
    return gen


def handle_upload_stream(handler, user=None):
    """POST /api/upload/stream —— 文件上传 + SSE 流式导入进度。

    multipart/form-data 字段：
        library   : 库名
        force     : "1" / "0"
        batch_id  : 可选，客户端预生成的批次 ID（用于撤销）
        files[]   : 一个或多个文件

    响应：SSE 流（同 /api/import/stream 的事件序列）
    """
    from transaction import recover_all_zones
    import tempfile
    import shutil
    from extractor import supported, SUPPORTED_EXTS

    ctype = handler.headers.get("Content-Type", "")
    if not ctype.startswith("multipart/form-data"):
        handler._send(415, {"ok": False, "error": "需要 multipart/form-data 请求"})
        return

    boundary = ctype.split("boundary=", 1)[1] if "boundary=" in ctype else ""
    if not boundary:
        handler._send(400, {"ok": False, "error": "缺少 boundary"})
        return
    boundary_bytes = ("--" + boundary).encode("utf-8")

    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        handler._send(400, {"ok": False, "error": "空请求体"})
        return
    raw = handler.rfile.read(length)

    parts = raw.split(boundary_bytes)
    lib_name = None
    force = False
    batch_id = ""
    uploaded = []  # [(filename, content_bytes)]

    for part in parts:
        if part in (b"", b"--", b"--\r\n", b"\r\n"):
            continue
        if part.startswith(b"\r\n"):
            part = part[2:]
        if part.endswith(b"\r\n"):
            part = part[:-2]
        try:
            header_block, content = part.split(b"\r\n\r\n", 1)
        except ValueError:
            continue
        header_str = header_block.decode("utf-8", errors="ignore")
        name = None
        filename = None
        for line in header_str.split("\r\n"):
            if line.lower().startswith("content-disposition:"):
                import re as _re
                m_name = _re.search(r'name="([^"]*)"', line)
                m_file = _re.search(r'filename="([^"]*)"', line)
                if m_name:
                    name = m_name.group(1)
                if m_file:
                    filename = m_file.group(1)
        if name is None:
            continue
        if name == "library":
            lib_name = content.decode("utf-8", errors="ignore").strip()
        elif name == "force":
            force = content.decode("utf-8", errors="ignore").strip() in ("1", "true", "True")
        elif name == "batch_id":
            batch_id = content.decode("utf-8", errors="ignore").strip()
        elif name == "files" and filename:
            uploaded.append((filename, content))

    if not lib_name:
        handler._send(400, {"ok": False, "error": "缺少 library 参数"})
        return
    if not uploaded:
        handler._send(400, {"ok": False, "error": "未收到任何文件"})
        return

    reg = _registry()
    lib, err = _get_lib(reg, lib_name, user, write=True)
    if err:
        handler._send(err[0], err[1])
        return
    mgr = lib.manager(BASE_DIR)

    batch_id = batch_id or _new_batch_id()

    # 注册批次（用于撤销）
    with _import_batches_lock:
        _import_batches[batch_id] = {
            "cancelled": False,
            "shas": [],
            "library": lib_name,
            "total": len(uploaded),
            "current": 0,
            "done": False,
            "started_at": time.time(),
        }
    # 持久化（服务器意外关闭后可清理）
    _persist_import_batches()

    # 保存上传文件到临时目录，过滤不支持的扩展名
    tmp_dir = tempfile.mkdtemp(prefix="upload_stream_")
    expanded = []
    used_names = set()
    try:
        for filename, content in uploaded:
            safe_base = os.path.basename(filename.replace("\\", "/"))
            ext = os.path.splitext(safe_base)[1].lower()
            if ext not in SUPPORTED_EXTS:
                continue
            base_name, ext_name = os.path.splitext(safe_base)
            final_name = safe_base
            counter = 1
            while final_name.lower() in used_names:
                final_name = f"{base_name}_{counter}{ext_name}"
                counter += 1
            used_names.add(final_name.lower())
            tmp_path = os.path.join(tmp_dir, final_name)
            with open(tmp_path, "wb") as f:
                f.write(content)
            expanded.append(tmp_path)

        if not expanded:
            handler._send(400, {"ok": False, "error": "没有可导入的文件（类型不支持）"})
            return

        gen = _build_import_stream_gen(mgr, expanded, force, batch_id, tmp_dir,
                                       lib_name=lib_name)
        handler._send_sse_stream(gen())
    except Exception as e:
        # SSE 未启动时返回 JSON 错误
        handler._send(500, {"ok": False, "error": f"上传流式导入异常: {e}"})
    finally:
        # _send_sse_stream 阻塞至生成器耗尽后才执行到这里，此时可安全清理临时目录
        import shutil as _shutil
        _shutil.rmtree(tmp_dir, ignore_errors=True)


def _rollback_batch(mgr, shas: list):
    """撤销批次：按 SHA 逐个删除已导入的文件。"""
    from remover import remove_by_sha
    for sha in shas:
        try:
            remove_by_sha(mgr, sha)
        except Exception as e:
            _logger.warning("撤销删除失败 sha=%s...: %s", sha[:12], e)


def handle_import_cancel(method, path, query, body):
    """POST /api/import/cancel —— 撤销导入批次。

    body: {"batch_id": "..."}
    - 若导入正在进行：设置取消标志，导入循环中断并回滚已 commit 部分
    - 若导入已完成：直接按 SHA 删除所有已导入文件
    """
    if method != "POST":
        return _err("需要 POST 方法", 405)
    batch_id = (body.get("batch_id") or "").strip()
    if not batch_id:
        return _err("缺少 batch_id 参数")
    with _import_batches_lock:
        batch = _import_batches.get(batch_id)
    if batch is None:
        return _err(f"批次不存在或已过期: {batch_id}", 404)
    lib_name = batch.get("library", "")
    reg = _registry()
    lib = reg.get_library(lib_name)
    if lib is None:
        return _err(f"库不存在: {lib_name}", 404)
    mgr = lib.manager(BASE_DIR)
    shas = batch.get("shas", [])
    if not batch.get("cancelled"):
        # 标记取消（进行中的批次会中断循环）
        with _import_batches_lock:
            if batch_id in _import_batches:
                _import_batches[batch_id]["cancelled"] = True
        # 如果批次还在进行中，循环自己会回滚；这里处理已完成的批次
        # 判断是否已完成：检查 total 是否已处理完（简化：直接尝试删除）
    # 对于已完成批次，直接删除所有 SHA
    _rollback_batch(mgr, shas)
    # 清理批次记录
    with _import_batches_lock:
        _import_batches.pop(batch_id, None)
    # 同步更新持久化文件（移除已取消的批次）
    _persist_import_batches()
    return _ok({"batch_id": batch_id, "cancelled": True, "removed_shas": len(shas)})


def handle_import_batch_status(method, path, query, body):
    """GET /api/import/batch?batch_id=xxx —— 查询批次状态（用于刷新页面后恢复进度）。

    返回：{batch_id, total, current, cancelled, done, library, shas_count}
    """
    if method != "GET":
        return _err("需要 GET 方法", 405)
    batch_id = (query.get("batch_id") or "").strip()
    if not batch_id:
        return _err("缺少 batch_id 参数")
    with _import_batches_lock:
        batch = _import_batches.get(batch_id)
    if batch is None:
        return _err(f"批次不存在或已过期: {batch_id}", 404)
    return _ok({
        "batch_id": batch_id,
        "total": batch.get("total", 0),
        "current": batch.get("current", 0),
        "cancelled": batch.get("cancelled", False),
        "done": batch.get("done", False),
        "library": batch.get("library", ""),
        "shas_count": len(batch.get("shas", [])),
    })


def handle_export(method, path, query, body, user=None):
    """导出库数据为 JSON。

    GET /api/export?library=xxx&type=all|search&query=xxx
    - type=all: 导出整个库的所有 chunk
    - type=search&query=xxx: 导出搜索命中的 chunk

    返回 JSON 文件下载。
    """
    if method != "GET":
        return _err("需要 GET 方法", 405)
    lib_name = query.get("library")
    if not lib_name:
        return _err("缺少 library 参数")
    reg = _registry()
    lib, err = _get_lib(reg, lib_name, user, write=False)
    if err:
        return err
    mgr = lib.manager(BASE_DIR)

    export_type = query.get("type", "all")

    if export_type == "search":
        # 导出搜索结果
        q = query.get("query", "")
        if not q:
            return _err("type=search 时需要 query 参数")
        from searcher import parallel_search
        result = parallel_search(reg, q, library_names=[lib_name],
                                 parallel=1, base_dir=BASE_DIR)
        return _ok({
            "library": lib_name,
            "note": lib.note,
            "query": q,
            "total_hits": result["total_hits"],
            "results": result["results"],
        })

    # type=all: 导出全部 chunk
    import json as _json
    zones = mgr.list_zones()
    all_chunks = []
    for z in zones:
        for chunk_path in z.iter_chunk_files():
            try:
                with open(chunk_path, "r", encoding="utf-8") as f:
                    all_chunks.append(_json.load(f))
            except (OSError, _json.JSONDecodeError):
                continue
    return _ok({
        "library": lib_name,
        "note": lib.note,
        "zone_count": len(zones),
        "total_chunks": len(all_chunks),
        "chunks": all_chunks,
    })


# ============================================================
#  设置 / DeepSeek
# ============================================================

def handle_get_settings(method, path, query, body, user=None):
    """GET /api/settings —— 返回当前设置（脱敏）+ 可选模型清单 + 厂商地址表。"""
    store = _settings()
    return _ok({
        "settings": store.safe_snapshot(),
        "models": AVAILABLE_MODELS,
        "provider_base_urls": PROVIDER_BASE_URLS,
        "configured": bool(store.get("deepseek_api_key")),
        "can_edit": _auth_store().count_users() == 0 or user["is_admin"],
    })


def handle_update_settings(method, path, query, body, user=None):
    """POST /api/settings —— 保存设置。

    特殊处理：若 deepseek_api_key 传回的是脱敏值（含 *），则不覆盖原 key。
    允许传空串 "" 显式清空 api_key。

    除 deepseek_* 外，也接受 settings.DEFAULTS 中定义的检索参数字段（数字/字符串）。
    保存设置需要管理员权限（无任何账号时开放，用于首次引导）。
    """
    if method != "POST":
        return _err("需要 POST 方法", 405)
    err = _require_admin(user)
    if err:
        return err
    from settings import DEFAULTS, SENSITIVE_KEYS
    store = _settings()
    updates = {}
    for key in ("deepseek_model", "deepseek_base_url"):
        if key in body:
            val = body[key]
            if val is not None:
                updates[key] = str(val).strip()
    # api_key 单独处理
    if "deepseek_api_key" in body:
        val = body["deepseek_api_key"]
        if val is None:
            pass  # 不动
        elif val == "":
            updates["deepseek_api_key"] = ""  # 显式清空
        elif isinstance(val, str) and "*" in val:
            # 脱敏值原样传回，不覆盖
            pass
        else:
            updates["deepseek_api_key"] = val.strip()
    # 检索参数：凡是在 DEFAULTS 中声明的非敏感字段，按类型转换后写入
    for key, default in DEFAULTS.items():
        if key in SENSITIVE_KEYS:
            continue
        if key in ("deepseek_model", "deepseek_base_url", "deepseek_api_key"):
            continue  # 上面已处理
        if key not in body:
            continue
        val = body[key]
        if val is None or val == "":
            continue  # 留空=保持默认/不动
        if isinstance(default, bool):
            updates[key] = bool(val)
        elif isinstance(default, int):
            try:
                updates[key] = int(val)
            except (ValueError, TypeError):
                pass
        elif isinstance(default, float):
            try:
                updates[key] = float(val)
            except (ValueError, TypeError):
                pass
        else:
            updates[key] = val
    if updates:
        store.update(updates)
    return _ok({
        "settings": store.safe_snapshot(),
        "updated": list(updates.keys()),
    })


def handle_test_deepseek(method, path, query, body):
    """POST /api/settings/test —— 测试 DeepSeek 连接。

    body 可选：{"api_key":"...", "model":"...", "base_url":"..."}
    若提供则用临时值测试（不落盘）；否则用已保存的设置。
    """
    if method != "POST":
        return _err("需要 POST 方法", 405)
    try:
        # 优先用 body 里的临时值
        api_key = (body.get("api_key") or "").strip()
        model = (body.get("model") or "").strip()
        base_url = (body.get("base_url") or "").strip()
        # 如果传回的是脱敏值，则忽略、回退到已保存配置
        if api_key and "*" in api_key:
            api_key = ""
        if not api_key:
            api_key = _settings().get("deepseek_api_key") or ""
        if not api_key:
            return _err("未提供 API Key")
        if not model:
            model = _settings().get("deepseek_model") or V4_FLASH
        if not base_url:
            base_url = _settings().get("deepseek_base_url") or "https://api.deepseek.com"
        client = DeepSeekClient(api_key=api_key, model=model, base_url=base_url, timeout=30)
        result = client.ping()
        return _ok(result)
    except DeepSeekError as e:
        return _err(f"DeepSeek 调用失败: {e}")
    except ValueError as e:
        return _err(str(e))
    except Exception as e:
        return _err(f"测试失败: {e}")


def handle_repair_sources(method, path, query, body):
    """POST /api/repair-sources —— 补齐库中缺失的源文件副本并修正 chunk 元数据。

    body: {"library": "...", "source_dir": "..."}
      - library: 库名
      - source_dir: 包含原始源文件的目录（递归扫描）

    流程：
      1. 扫描库中所有 chunk 的 source，统计缺失副本的 SHA256 -> file_name 映射
      2. 递归扫描 source_dir，计算每个文件 SHA256，匹配缺失项
      3. 匹配成功的文件复制到 <库>/_sources/，并更新对应 chunk 的 source.file_path
      4. 同时更新 _dedup_index.json 中的 file_path

    返回 {"scanned": N, "missing": M, "repaired": K, "details": [...]}
    """
    if method != "POST":
        return _err("需要 POST 方法", 405)
    lib_name = (body.get("library") or "").strip()
    source_dir = (body.get("source_dir") or "").strip()
    if not lib_name or not source_dir:
        return _err("缺少 library 或 source_dir 参数")
    if not os.path.isdir(source_dir):
        return _err(f"source_dir 不存在或不是目录: {source_dir}")

    reg = _registry()
    lib, err = _get_lib(reg, lib_name, user, write=True)
    if err:
        return err
    mgr = lib.manager(BASE_DIR)
    sources_dir = os.path.join(mgr.root, "_sources")
    os.makedirs(sources_dir, exist_ok=True)

    # 1. 扫描所有 chunk，按 SHA 收集 file_path 信息
    from importer import _copy_source_to_lib
    from dedup import DedupIndex, compute_file_sha256
    from extractor import supported

    sha_to_chunks = {}  # sha -> [(zone, chunk_path, chunk_data)]
    for z in mgr.list_zones():
        if not os.path.isdir(z.chunks_dir):
            continue
        for name in sorted(os.listdir(z.chunks_dir)):
            if not name.endswith(".json"):
                continue
            cp = os.path.join(z.chunks_dir, name)
            try:
                with open(cp, "r", encoding="utf-8") as f:
                    c = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            src = c.get("source", {})
            sha = src.get("source_sha256", "")
            if not sha:
                continue
            sha_to_chunks.setdefault(sha, []).append((z, cp, c))

    # 2. 判断哪些 SHA 缺失副本（_sources/ 中没有对应文件）
    #    同时收集已存在副本的 basename -> sha 映射
    existing_basenames = set()
    for fn in os.listdir(sources_dir):
        full = os.path.join(sources_dir, fn)
        if os.path.isfile(full):
            existing_basenames.add(fn)

    missing_shas = {}  # sha -> file_name (用于显示)
    for sha, items in sha_to_chunks.items():
        # 检查该 SHA 是否在 _sources/ 有副本
        c = items[0][2]
        src = c.get("source", {})
        file_name = src.get("file_name", "")
        # _copy_source_to_lib 的命名规则：file_name 或 file_name_N
        # 简单判断：basename 在 _sources/ 中存在且 SHA 匹配即视为已有
        found = False
        for bn in existing_basenames:
            if bn == file_name or bn.startswith(os.path.splitext(file_name)[0]):
                full = os.path.join(sources_dir, bn)
                try:
                    if compute_file_sha256(full) == sha:
                        found = True
                        break
                except OSError:
                    pass
        if not found:
            missing_shas[sha] = file_name

    # 3. 递归扫描 source_dir，按 SHA 匹配补齐
    candidate_files = []
    for root, _d, names in os.walk(source_dir):
        for n in names:
            full = os.path.join(root, n)
            if supported(full) or os.path.isfile(full):
                candidate_files.append(full)

    repaired = 0
    details = []
    for cf in candidate_files:
        try:
            cf_sha = compute_file_sha256(cf)
        except OSError:
            continue
        if cf_sha not in missing_shas:
            continue
        # 匹配成功，复制到 _sources/
        file_name = missing_shas[cf_sha]
        try:
            rel_path = _copy_source_to_lib(mgr, cf, file_name, cf_sha)
        except Exception as e:
            details.append({"file": file_name, "sha": cf_sha[:12],
                            "status": "fail", "error": str(e)})
            continue
        # 更新所有对应 chunk 的 source.file_path
        for zone, cp, c in sha_to_chunks[cf_sha]:
            c["source"]["file_path"] = rel_path
            try:
                tmp = cp + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(c, f, ensure_ascii=False, indent=2)
                os.replace(tmp, cp)
            except OSError as e:
                details.append({"file": file_name, "sha": cf_sha[:12],
                                "status": "fail", "error": f"更新 chunk 元数据失败: {e}"})
                continue
        # 更新 dedup_index
        try:
            dedup = DedupIndex(mgr.root)
            existing = dedup.check(cf_sha)
            if existing:
                dedup.add(cf_sha, file_name, rel_path,
                          existing.get("zone_id", ""), existing.get("char_count", 0))
        except Exception:
            pass
        repaired += 1
        details.append({"file": file_name, "sha": cf_sha[:12],
                        "status": "ok", "copied_to": rel_path})
        # 从 missing_shas 移除已修复的
        del missing_shas[cf_sha]

    unrepaired = [{"file": fn, "sha": sha[:12]} for sha, fn in missing_shas.items()]

    return _ok({
        "scanned_chunks": sum(len(v) for v in sha_to_chunks.values()),
        "unique_sources": len(sha_to_chunks),
        "missing": len(unrepaired) + repaired,
        "repaired": repaired,
        "still_missing": unrepaired[:20],
        "details": details,
    })


def handle_chat_sessions(method, path, query, body, session_id=None, user=None):
    """会话管理：列出/创建/获取/删除会话。

    GET    /api/chat/sessions                 列出当前用户的会话
    POST   /api/chat/sessions                 创建会话 body:{title?,mode?,libraries?}
    GET    /api/chat/sessions/{id}            获取会话详情（含 messages）
    DELETE /api/chat/sessions/{id}            删除会话
    PATCH  /api/chat/sessions/{id}            更新会话元数据 body:{title?,mode?,libraries?}

    多用户：会话归属当前用户（游客=guest）；只能访问自己的会话，管理员可访问全部。
    """
    store = _chat_store()
    # 列出
    if method == "GET" and not session_id:
        if user["is_admin"]:
            return _ok({"sessions": store.list_sessions(owner=None)})
        return _ok({"sessions": store.list_sessions(owner=user["username"])})
    # 创建
    if method == "POST" and not session_id:
        title = (body.get("title") or "").strip()
        mode = body.get("mode") or "direct"
        libraries = body.get("libraries") or []
        s = store.create_session(title=title, mode=mode, libraries=libraries,
                                 owner=user["username"])
        return _ok(s)
    # 需要 session_id 的操作
    if not session_id:
        return _err("缺少 session_id", 400)
    # 校验会话归属（旧会话无 owner 视为游客会话；管理员可操作全部）
    s_owner = store.session_owner(session_id)
    if s_owner == "":
        return _err(f"会话不存在: {session_id}", 404)
    if not user["is_admin"] and s_owner != user["username"]:
        return _err(f"会话不存在: {session_id}", 404)
    if method == "GET":
        s = store.get_session(session_id)
        if s is None:
            return _err(f"会话不存在: {session_id}", 404)
        return _ok(s)
    if method == "DELETE":
        ok = store.delete_session(session_id)
        if not ok:
            return _err(f"会话不存在: {session_id}", 404)
        return _ok({"deleted": session_id})
    if method == "PATCH":
        s = store.update_session_meta(
            session_id,
            title=body.get("title"),
            mode=body.get("mode"),
            libraries=body.get("libraries"),
        )
        if s is None:
            return _err(f"会话不存在: {session_id}", 404)
        return _ok(s)
    return _err("不支持的方法", 405)


def _check_session_owner(store, session_id, user) -> tuple | None:
    """校验会话归属：返回 None=允许；否则错误响应。"""
    s_owner = store.session_owner(session_id)
    if s_owner == "":
        return _err(f"会话不存在: {session_id}", 404)
    if not user["is_admin"] and s_owner != user["username"]:
        return _err(f"会话不存在: {session_id}", 404)
    return None


def handle_chat_context(method, path, query, body, session_id, user=None):
    """管理会话的额外上下文（检索结果 → 注入对话模式）。

    POST   /api/chat/sessions/{id}/context   body: {chunks: [...]}  设置（覆盖）
    GET    /api/chat/sessions/{id}/context                                读取
    DELETE /api/chat/sessions/{id}/context                                清空

    chunks 每项: {chunk_id, library, heading, file_path, text, is_center?}
    """
    store = _chat_store()
    err = _check_session_owner(store, session_id, user)
    if err:
        return err
    s = store.get_session(session_id)
    if s is None:
        return _err(f"会话不存在: {session_id}", 404)
    if method == "GET":
        return _ok({"chunks": store.get_extra_context(session_id)})
    if method == "POST":
        chunks = body.get("chunks", [])
        if not isinstance(chunks, list):
            return _err("chunks 必须是数组")
        # 限制单条 text 长度，防止过大
        MAX_TEXT = 20000
        cleaned = []
        for c in chunks:
            if not isinstance(c, dict):
                continue
            t = c.get("text", "") or ""
            if len(t) > MAX_TEXT:
                t = t[:MAX_TEXT]
            cleaned.append({
                "chunk_id": c.get("chunk_id", ""),
                "library": c.get("library", ""),
                "heading": c.get("heading", ""),
                "file_path": c.get("file_path", ""),
                "text": t,
                "is_center": bool(c.get("is_center", False)),
                # 补存排序与窗口展示字段，避免会话重载后 UI 退化（"整块"化、顺序错乱）
                "chunk_seq": c.get("chunk_seq") or 0,
                "hit_positions": c.get("hit_positions") or [],
                "eff_window": c.get("eff_window") or 0,
            })
        store.set_extra_context(session_id, cleaned)
        return _ok({"count": len(cleaned)})
    if method == "DELETE":
        store.set_extra_context(session_id, [])
        return _ok({"cleared": True})
    return _err("不支持的方法", 405)


def handle_chat_export(method, path, query, body, session_id, user=None):
    """GET /api/chat/sessions/{id}/export —— 导出会话为 Markdown。"""
    store = _chat_store()
    err = _check_session_owner(store, session_id, user)
    if err:
        return err
    s = store.get_session(session_id)
    if s is None:
        return _err(f"会话不存在: {session_id}", 404)
    lines = []
    lines.append(f"# {s.get('title', '会话')}\n")
    lines.append(f"- 会话 ID：{s.get('id', '')}")
    lines.append(f"- 创建时间：{s.get('created_at', '')}")
    lines.append(f"- 更新时间：{s.get('updated_at', '')}")
    lines.append(f"- 检索模式：{s.get('mode', 'direct')}")
    libs = s.get("libraries", [])
    lines.append(f"- 关联库：{', '.join(libs) if libs else '（全部）'}\n")
    lines.append("---\n")
    for m in s.get("messages", []):
        role = m.get("role", "")
        ts = m.get("timestamp", "")
        if role == "user":
            lines.append(f"## 👤 用户 · {ts}\n")
            lines.append(m.get("content", "") + "\n")
        elif role == "assistant":
            lines.append(f"## 🤖 助手 · {ts}\n")
            if m.get("reasoning"):
                lines.append("<details><summary>思考过程</summary>\n")
                lines.append(m["reasoning"] + "\n")
                lines.append("</details>\n")
            lines.append(m.get("content", "") + "\n")
            refs = m.get("references", [])
            if refs:
                lines.append("**引用来源：**\n")
                for r in refs:
                    sf = r.get("source_file", "")
                    heading = r.get("heading", "")
                    cid = r.get("chunk_id", "")
                    lib = r.get("library", "")
                    lines.append(f"- [{r.get('index','')}] {lib} · {sf}"
                                 f"{' · ' + heading if heading else ''} ({cid})")
                lines.append("")
            lines.append("---\n")
    md = "\n".join(lines)
    return _ok({
        "markdown": md,
        "filename": f"chat_{s.get('title', 'session')}_{s.get('id','')}.md",
    })


def handle_suggest_topk(method, path, query, body, user=None):
    """GET /api/ai-search/suggest-topk?libraries=A,B —— 基于库 chunk 总数推荐引用条数。

    科学依据：
      - 检索结果相关性按 score 降序排列，前 N 条覆盖核心信息；
        N 过小漏答，N 过大引入噪声并占用 context 预算。
      - chunk 总数少时单条信息密度高，建议小 N；大库相关 chunk 多，
        适度放大 N 但需控制 context 总长（max_total_chars 上限 32000）。
      - 公式：clamp(round(log10(chunks+1) * 8), 5, 25)
        chunks=10  -> 8    chunks=100 -> 16
        chunks=432 -> 21   chunks=5000-> 25(封顶)
    """
    import math
    lib_param = query.get("libraries", "")
    library_names = [x for x in lib_param.split(",") if x] if lib_param else None
    reg = _registry()
    libs = reg.list_libraries_for(user["username"], user["is_admin"])
    if library_names:
        err_resp = _check_libraries_visible(reg, library_names, user)
        if err_resp:
            return err_resp
        libs = [l for l in libs if l.name in library_names]
    total_chunks = 0
    for lib in libs:
        mgr = lib.manager(BASE_DIR)
        total_chunks += mgr.stats()["total_chunks"]
    if total_chunks <= 0:
        suggested = 8
    else:
        suggested = max(5, min(25, round(math.log10(total_chunks + 1) * 8)))
    return _ok({
        "suggested_top_k": suggested,
        "total_chunks": total_chunks,
        "rationale": (
            f"基于 {total_chunks} 个 chunk 总数，按 log10 公式推荐 {suggested} 条。"
            f"小库信息密度高用小值，大库相关 chunk 多适度放大但封顶 25 避免噪声。"
        ),
    })


def handle_ai_search(method, path, query, body, user=None):
    """POST /api/ai-search —— 智能检索（一次性返回完整答案）。"""
    if method != "POST":
        return _err("需要 POST 方法", 405)
    question = (body.get("question") or "").strip()
    if not question:
        return _err("缺少 question 参数")
    libraries = body.get("libraries")  # list[str] 或 None
    parallel = int(body.get("parallel", 4))
    top_k = int(body.get("top_k", 20))
    temperature = float(body.get("temperature", 0.3))
    try:
        client = _build_client_from_settings()
    except ValueError as e:
        return _err(str(e))
    reg = _registry()
    if libraries:
        err_resp = _check_libraries_visible(reg, libraries, user)
        if err_resp:
            return err_resp
    else:
        libraries = [l.name for l in reg.list_libraries_for(user["username"], user["is_admin"])]
    from ai_search import ai_search
    try:
        result = ai_search(
            question, reg, client, BASE_DIR,
            library_names=libraries if libraries else None,
            parallel=parallel, top_k=top_k, temperature=temperature,
        )
    except DeepSeekError as e:
        return _err(f"DeepSeek 调用失败: {e}")
    except ValueError as e:
        return _err(str(e))
    return _ok(result)


# ============================================================
#  溯源：chunk 文本 + 打开文档
# ============================================================

def handle_get_chunk(method, path, query, body, user=None):
    """GET /api/chunk?library=xxx&chunk_id=zone_001/chunk_000123

    返回完整 chunk 文本及元数据，用于"展开 chunk 文字"溯源。
    """
    lib_name = query.get("library")
    chunk_id = query.get("chunk_id")
    if not lib_name or not chunk_id:
        return _err("缺少 library 或 chunk_id 参数")
    reg = _registry()
    lib, err = _get_lib(reg, lib_name, user, write=False)
    if err:
        return err
    mgr = lib.manager(BASE_DIR)
    parts = chunk_id.split("/")
    if len(parts) != 2:
        return _err(f"chunk_id 格式错误: {chunk_id}")
    zone_id, chunk_name = parts
    zone = mgr.get_zone(zone_id)
    if zone is None:
        return _err(f"zone 不存在: {zone_id}", 404)
    chunk_path = os.path.join(zone.chunks_dir, f"{chunk_name}.json")
    if not os.path.isfile(chunk_path):
        return _err(f"chunk 文件不存在: {chunk_id}", 404)
    try:
        with open(chunk_path, "r", encoding="utf-8") as f:
            chunk = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return _err(f"读取 chunk 失败: {e}")
    return _ok({
        "chunk_id": chunk.get("chunk_id"),
        "zone_id": chunk.get("zone_id"),
        "chunk_seq": chunk.get("chunk_seq"),
        "text": chunk.get("text", ""),
        "text_offset": chunk.get("text_offset", 0),
        "text_length": chunk.get("text_length", 0),
        "heading": chunk.get("heading", ""),
        "source": chunk.get("source", {}),
        "created_at": chunk.get("created_at", ""),
        "library": lib_name,
    })


def handle_file_chunks(method, path, query, body, user=None):
    """GET /api/file-chunks —— 获取某源文件对应的所有 chunk 文本（拼接预览用）。

    GET /api/file-chunks?library=xxx&sha256=xxx
      - library: 库名
      - sha256: 源文件 SHA256
      - file_path: 可选，当无 sha256 时按 file_path 匹配

    返回 {library, sha256, file_name, file_path, total_chunks, total_chars, text}
      - text: 所有 chunk 按 chunk_seq 升序拼接的纯文本
    """
    if method != "GET":
        return _err("需要 GET 方法", 405)
    lib_name = query.get("library")
    if not lib_name:
        return _err("缺少 library 参数")
    sha256 = (query.get("sha256") or "").strip()
    file_path = (query.get("file_path") or "").strip()
    file_name = (query.get("file_name") or "").strip()
    if not sha256 and not file_path and not file_name:
        return _err("缺少 sha256、file_path 或 file_name 参数")

    reg = _registry()
    lib, err = _get_lib(reg, lib_name, user, write=False)
    if err:
        return err
    mgr = lib.manager(BASE_DIR)

    from dedup import DedupIndex
    from remover import _iter_chunks


    # 优先用 sha256 查 dedup，缩小扫描 zone
    target_zone_id = None
    resolved_file_name = file_name or ""
    dedup_path = ""
    if sha256:
        info = DedupIndex(mgr.root).check(sha256)
        if info is None:
            return _err(f"未找到该文件: sha256={sha256}", 404)
        target_zone_id = info.get("zone_id", "")
        resolved_file_name = info.get("file_name", "") or resolved_file_name
        dedup_path = info.get("file_path", "")

    # 收集 chunk（同一文件所有 chunk 在同一 zone，但保险起见做兜底）
    zones = []
    if target_zone_id:
        try:
            zones = [mgr.get_zone(target_zone_id)]
        except Exception:
            zones = []
    if not zones:
        zones = mgr.list_zones()

    matched = []
    for zone in zones:
        for _path, chunk in _iter_chunks(zone):
            src = chunk.get("source", {})
            if sha256 and src.get("source_sha256", "") != sha256:
                continue
            src_file_path = src.get("file_path", "")
            if file_path:
                # 多种匹配方式：完全相等、正斜杠统一、反斜杠统一、basename 相等
                norm_a = file_path.replace("\\", "/")
                norm_b = src_file_path.replace("\\", "/")
                if norm_a != norm_b and file_path != src_file_path:
                    continue
            elif file_name:
                # 按 file_name 匹配（basename）
                src_basename = os.path.basename(src_file_path.replace("\\", "/"))
                if src_basename != file_name:
                    continue
            matched.append(chunk)

    if not matched:
        return _err("该文件无对应 chunk（可能已被删除）", 404)

    # 按 chunk_seq 升序拼接文本
    matched.sort(key=lambda c: c.get("chunk_seq", 0))
    # 去除 chunk 间重叠部分（overlap_prev 标记与前一块重叠字符数）
    parts = []
    total_chars = 0
    for i, c in enumerate(matched):
        text = c.get("text", "")
        if i > 0:
            ov = c.get("overlap_prev", 0)
            if ov > 0 and ov < len(text):
                text = text[ov:]
        parts.append(text)
        total_chars += c.get("text_length", 0)
    full_text = "".join(parts)

    if not resolved_file_name and matched:
        resolved_file_name = matched[0].get("source", {}).get("file_name", "")
    if not dedup_path and matched:
        dedup_path = matched[0].get("source", {}).get("file_path", "")

    return _ok({
        "library": lib_name,
        "sha256": sha256,
        "file_name": resolved_file_name,
        "file_path": dedup_path,
        "total_chunks": len(matched),
        "total_chars": total_chars,
        "text": full_text,
    })


def handle_chunks_around(method, path, query, body, user=None):
    """GET /api/chunks-around —— 获取选中 chunk 同文件前后 N 个 chunk。

    GET /api/chunks-around?library=xxx&chunk_id=zone_001/chunk_000123&around=2&window=0&matched_words=经济发展
      - library: 库名
      - chunk_id: 中心 chunk 的 id
      - around: 前后各取几个（默认 2，即前2+自身+后2=5个）
      - window: 每个 chunk 命中位置前后各取多少字（默认 0 = 返回整块全文）。
                >0 时中心 chunk 返回"命中点 ±window 字"的片段（省 token、颗粒更细）
      - matched_words: （可选）命中词，空格或逗号分隔，用于在 chunk 文本内定位命中点。
                未提供时若 window>0 则退回返回整块。

    返回 {chunks: [{chunk_id, chunk_seq, text, heading, hit_positions, ...}], center_seq}
    仅返回同一 source_sha256 的 chunk（不跨文件）。
    """
    if method != "GET":
        return _err("需要 GET 方法", 405)
    lib_name = query.get("library")
    if not lib_name:
        return _err("缺少 library 参数")
    chunk_id = (query.get("chunk_id") or "").strip()
    if not chunk_id or "/" not in chunk_id:
        return _err("缺少 chunk_id 参数（格式 zone_xxx/chunk_xxxxxx）")
    try:
        around = int(query.get("around", "2"))
    except ValueError:
        around = 2
    around = max(0, min(around, 10))  # 限制 0~10
    # 命中片段窗口：>0 时按命中小片段返回；0 保持整块全文（旧行为）
    try:
        window = int(query.get("window", "0"))
    except ValueError:
        window = 0
    window = max(0, min(window, 2000))  # 限制 0~2000 字
    # 命中词：用于在 chunk 内定位命中点
    mw_raw = (query.get("matched_words") or "").strip()
    matched_words = [w for w in
                     [x.strip() for x in mw_raw.replace(",", " ").split()] if w]

    reg = _registry()
    lib, err = _get_lib(reg, lib_name, user, write=False)
    if err:
        return err
    mgr = lib.manager(BASE_DIR)

    zone_id, chunk_name = chunk_id.split("/", 1)
    try:
        zone = mgr.get_zone(zone_id)
    except Exception:
        return _err(f"zone 不存在: {zone_id}", 404)

    # 读中心 chunk，拿到 source_sha256 和 chunk_seq
    chunk_path = os.path.join(zone.chunks_dir, f"{chunk_name}.json")
    if not os.path.isfile(chunk_path):
        return _err(f"chunk 不存在: {chunk_id}", 404)
    with open(chunk_path, "r", encoding="utf-8") as f:
        center_chunk = json.load(f)
    center_seq = center_chunk.get("chunk_seq", 0)
    target_sha = center_chunk.get("source", {}).get("source_sha256", "")
    if not target_sha:
        return _err("中心 chunk 无 source_sha256，无法扩展")

    # 遍历同 zone 的 chunk，筛选同 sha256 的，按 seq 排序
    from remover import _iter_chunks
    same_file_chunks = []
    for _p, c in _iter_chunks(zone):
        if c.get("source", {}).get("source_sha256", "") == target_sha:
            same_file_chunks.append(c)
    same_file_chunks.sort(key=lambda c: c.get("chunk_seq", 0))

    # 找中心 chunk 在同文件列表中的位置
    center_idx = None
    for i, c in enumerate(same_file_chunks):
        if c.get("chunk_seq", 0) == center_seq:
            center_idx = i
            break
    if center_idx is None:
        return _err("未找到中心 chunk 在同文件列表中的位置")

    # 截取前后 N 个
    start = max(0, center_idx - around)
    end = min(len(same_file_chunks), center_idx + around + 1)
    selected = same_file_chunks[start:end]

    # 返回精简字段
    out = []
    for c in selected:
        full_text = c.get("text", "") or ""
        text = full_text
        hit_positions = []
        # 命中片段窗口：仅在 window>0 且在该 chunk 内定位到命中点时切成"命中点 ±window 字"片段
        # 中心与相邻 chunk 都尝试定位（相邻块常含同词，同样省 token）；
        # 定位不到（如仅标题命中）则该块回退整块全文
        found_words = matched_words
        if window > 0 and found_words and full_text:
            positions = []
            for w in found_words:
                idx = full_text.find(w)
                if idx >= 0:
                    positions.append(idx)
            if positions:
                pos = min(positions)
                start = max(0, pos - window)
                end = min(len(full_text), pos + window)
                snippet = full_text[start:end]
                if start > 0:
                    snippet = "..." + snippet
                if end < len(full_text):
                    snippet = snippet + "..."
                text = snippet
                hit_positions = positions
        out.append({
            "chunk_id": c.get("chunk_id", ""),
            "chunk_seq": c.get("chunk_seq", 0),
            "text": text,
            "heading": c.get("heading", ""),
            "text_offset": c.get("text_offset", 0),
            "text_length": c.get("text_length", 0),
            "is_center": c.get("chunk_seq", 0) == center_seq,
            "hit_positions": hit_positions,
            "window": window if (window > 0 and hit_positions) else 0,
        })
    return _ok({
        "chunks": out,
        "center_seq": center_seq,
        "total_same_file": len(same_file_chunks),
    })


def _resolve_source_file(file_path: str, library: str = "") -> str | None:
    """解析源文件绝对路径。

    顺序：
      1. 相对 BASE_DIR
      2. 相对 cwd（即按绝对路径或当前工作目录）
      3. 回退到指定库的 _sources/ 目录按 basename 查找副本

    找不到返回 None。
    """
    if not file_path:
        return None
    candidates = [
        os.path.abspath(os.path.join(BASE_DIR, file_path)),
        os.path.abspath(file_path),
    ]
    abs_path = next((p for p in candidates if os.path.isfile(p)), None)
    # 回退：去指定库的 _sources/ 目录按 basename 查找副本。
    # 这样即使 chunk 的 source.file_path 是裸文件名（历史导入数据）或
    # 原始文件已被移动/删除，仍能找到导入时复制到存储区的副本。
    if abs_path is None and library:
        reg = _registry()
        lib = reg.get_library(library)
        if lib is not None:
            sources_dir = os.path.join(lib.abs_path(BASE_DIR), "_sources")
            base_name = os.path.basename(file_path.replace("\\", "/"))
            candidate_in_sources = os.path.join(sources_dir, base_name)
            if os.path.isfile(candidate_in_sources):
                abs_path = candidate_in_sources
    return abs_path


def handle_open_doc(method, path, query, body, user=None):
    """POST /api/open-doc —— 用默认应用打开文档，失败则在文件资源管理器中定位。

    body: {"file_path": "...", "fallback": "explorer" | "none", "library": "..."}
      - file_path: chunk 的 source.file_path（相对主程序目录的相对路径）
      - fallback: 打开失败时的回退动作，默认 "explorer"
      - library: 可选，库名，用于文件不存在时回退打开该库的数据存储区目录

    返回 {"action": "opened" | "explorer" | "failed", "abs_path": "...", "message": "..."}

    多用户：仅登录用户可触发（本机文件操作），且指定 library 时校验可见性。
    """
    if method != "POST":
        return _err("需要 POST 方法", 405)
    err = _require_login(user)
    if err:
        return err
    file_path = (body.get("file_path") or "").strip()
    if not file_path:
        return _err("缺少 file_path 参数")
    fallback = body.get("fallback", "explorer")
    library = body.get("library") or ""
    if library:
        reg = _registry()
        _, err = _get_lib(reg, library, user, write=False)
        if err:
            return err

    abs_path = _resolve_source_file(file_path, library)

    import subprocess
    import platform
    system = platform.system()

    if abs_path is None:
        # 文件不存在（常见于从浏览器上传的文件，临时目录已被清理）
        # 回退：打开数据存储区目录（如果有 library 参数），否则打开 BASE_DIR
        if fallback != "explorer":
            return _ok({
                "action": "failed",
                "abs_path": file_path,
                "message": "文件不存在（可能已被移动或删除，或来自已清理的临时上传目录）",
            })
        fallback_dir = BASE_DIR
        if library:
            reg = _registry()
            lib = reg.get_library(library)
            if lib is not None:
                fallback_dir = lib.abs_path(BASE_DIR)
        try:
            if system == "Windows":
                subprocess.Popen(f'explorer "{fallback_dir}"', shell=True)
            elif system == "Darwin":
                subprocess.Popen(["open", fallback_dir])
            else:
                subprocess.Popen(["xdg-open", fallback_dir])
            return _ok({
                "action": "explorer",
                "abs_path": fallback_dir,
                "message": f"源文件不存在，已打开数据存储区目录: {os.path.basename(fallback_dir)}",
            })
        except Exception as e:
            return _ok({
                "action": "failed",
                "abs_path": file_path,
                "message": f"文件不存在且回退打开目录失败: {e}",
            })

    # 1. 尝试用默认应用打开
    try:
        if system == "Windows":
            # 用 start 命令启动完全独立的进程（DETACHED），避免 HTTP 线程结束后影响子进程
            # os.startfile 在 Web 服务子线程中可能导致打开的程序随线程结束而被关闭
            subprocess.Popen(
                f'start "" "{abs_path}"',
                shell=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.DETACHED_PROCESS,
            )
            return _ok({
                "action": "opened",
                "abs_path": abs_path,
                "message": f"已用默认应用打开: {os.path.basename(abs_path)}",
            })
        elif system == "Darwin":
            subprocess.Popen(["open", abs_path])
            return _ok({
                "action": "opened",
                "abs_path": abs_path,
                "message": f"已用默认应用打开: {os.path.basename(abs_path)}",
            })
        else:
            subprocess.Popen(["xdg-open", abs_path])
            return _ok({
                "action": "opened",
                "abs_path": abs_path,
                "message": f"已用默认应用打开: {os.path.basename(abs_path)}",
            })
    except Exception as e:
        # 打开失败，回退到在文件资源管理器中定位
        if fallback != "explorer":
            return _ok({
                "action": "failed",
                "abs_path": abs_path,
                "message": f"打开失败: {e}",
            })
        try:
            if system == "Windows":
                # explorer /select,"path" 会打开资源管理器并选中文件
                subprocess.Popen(f'explorer /select,"{abs_path}"', shell=True)
            elif system == "Darwin":
                # macOS: 在 Finder 中显示
                subprocess.Popen(["open", "-R", abs_path])
            else:
                # Linux: 打开所在目录
                subprocess.Popen(["xdg-open", os.path.dirname(abs_path)])
            return _ok({
                "action": "explorer",
                "abs_path": abs_path,
                "message": f"打开失败已回退到文件资源管理器: {e}",
            })
        except Exception as e2:
            return _ok({
                "action": "failed",
                "abs_path": abs_path,
                "message": f"打开与回退均失败: {e2}",
            })


def handle_delete_files(method, path, query, body, user=None):
    from remover import remove_by_ext, remove_by_sha
    if method != "DELETE":
        return _err("需要 DELETE 方法", 405)
    lib_name = query.get("library")
    ext = query.get("ext")
    sha = query.get("sha")
    if not lib_name:
        return _err("缺少 library 参数")
    reg = _registry()
    lib, err = _get_lib(reg, lib_name, user, write=True)
    if err:
        return err
    mgr = lib.manager(BASE_DIR)

    if ext:
        result = remove_by_ext(mgr, ext)
    elif sha:
        result = remove_by_sha(mgr, sha)
    else:
        return _err("需要 ext 或 sha 参数")
    # 删除后使 ZoneIndex 缓存失效，避免下次搜索读到旧索引
    from indexer import ZoneIndex
    for z in mgr.list_zones():
        ZoneIndex.invalidate(z.index_dir)
    return _ok(result)


# ============================================================
#  批量删除（异步执行 + 进度查询，支持页面刷新后恢复进度）
# ============================================================

# 活跃删除批次注册表：batch_id -> {library, shas, total, current, done, cancelled, ...}
_delete_batches: Dict[str, Dict[str, Any]] = {}
_delete_batches_lock = threading.Lock()


def _new_delete_batch_id() -> str:
    import uuid
    return f"del_{uuid.uuid4().hex[:12]}"


def handle_batch_delete(method, path, query, body, user=None):
    """POST /api/files/batch-delete —— 异步批量删除文件。

    body: {"library":"xxx", "shas":["sha1","sha2",...]}
    返回：{batch_id, total}

    删除在后台线程执行；前端通过 GET /api/files/delete-status 查询进度。
    """
    if method != "POST":
        return _err("需要 POST 方法", 405)
    lib_name = (body.get("library") or "").strip()
    shas = body.get("shas") or []
    if not lib_name:
        return _err("缺少 library 参数")
    if not shas or not isinstance(shas, list):
        return _err("缺少 shas 参数或格式错误")
    reg = _registry()
    lib, err = _get_lib(reg, lib_name, user, write=True)
    if err:
        return err

    # 去重
    shas = list(dict.fromkeys([s for s in shas if s]))
    if not shas:
        return _err("shas 列表为空")

    batch_id = _new_delete_batch_id()
    with _delete_batches_lock:
        _delete_batches[batch_id] = {
            "library": lib_name,
            "shas": list(shas),
            "total": len(shas),
            "current": 0,
            "done": False,
            "cancelled": False,
            "ok_count": 0,
            "fail_count": 0,
            "errors": [],
            "removed_chunks": 0,
            "removed_chars": 0,
            "started_at": time.time(),
        }

    def _run_delete():
        from remover import remove_by_sha
        from indexer import ZoneIndex
        # 每次循环都新建 manager/registry，避免缓存陈旧
        try:
            reg_local = _registry()
            lib_local = reg_local.get_library(lib_name)
            if lib_local is None:
                with _delete_batches_lock:
                    if batch_id in _delete_batches:
                        _delete_batches[batch_id]["done"] = True
                        _delete_batches[batch_id]["errors"].append(
                            {"sha": "", "error": f"库不存在: {lib_name}"})
                return
            mgr = lib_local.manager(BASE_DIR)
            shas_list = list(shas)
            invalidated_zones = set()
            for i, sha in enumerate(shas_list):
                # 检查取消标志
                with _delete_batches_lock:
                    if _delete_batches.get(batch_id, {}).get("cancelled"):
                        break
                    if batch_id in _delete_batches:
                        _delete_batches[batch_id]["current"] = i
                try:
                    result = remove_by_sha(mgr, sha)
                    # 即时失效该 sha 所属 zone 的索引缓存
                    zones_affected = result.get("zones_affected", []) or []
                    for zid in zones_affected:
                        try:
                            z = mgr.get_zone(zid)
                            if z and z.index_dir not in invalidated_zones:
                                invalidated_zones.add(z.index_dir)
                                ZoneIndex.invalidate(z.index_dir)
                        except Exception:
                            pass
                    with _delete_batches_lock:
                        if batch_id in _delete_batches:
                            _delete_batches[batch_id]["ok_count"] += 1
                            _delete_batches[batch_id]["removed_chunks"] += result.get("removed_chunks", 0) or 0
                            _delete_batches[batch_id]["removed_chars"] += result.get("removed_chars", 0) or 0
                except Exception as e:
                    with _delete_batches_lock:
                        if batch_id in _delete_batches:
                            _delete_batches[batch_id]["fail_count"] += 1
                            _delete_batches[batch_id]["errors"].append(
                                {"sha": sha, "error": str(e)})
            # 最终再失效一次所有 zone（确保索引一致）
            for z in mgr.list_zones():
                ZoneIndex.invalidate(z.index_dir)
            with _delete_batches_lock:
                if batch_id in _delete_batches:
                    _delete_batches[batch_id]["current"] = len(shas_list)
                    _delete_batches[batch_id]["done"] = True
        except Exception as e:
            with _delete_batches_lock:
                if batch_id in _delete_batches:
                    _delete_batches[batch_id]["done"] = True
                    _delete_batches[batch_id]["errors"].append(
                        {"sha": "", "error": f"批量删除异常: {e}"})

    threading.Thread(target=_run_delete, daemon=True).start()
    return _ok({"batch_id": batch_id, "total": len(shas)})


def handle_delete_batch_status(method, path, query, body):
    """GET /api/files/delete-status?batch_id=xxx —— 查询批量删除进度。"""
    if method != "GET":
        return _err("需要 GET 方法", 405)
    batch_id = (query.get("batch_id") or "").strip()
    if not batch_id:
        return _err("缺少 batch_id 参数")
    with _delete_batches_lock:
        batch = _delete_batches.get(batch_id)
        if batch is None:
            return _err(f"批次不存在或已过期: {batch_id}", 404)
        return _ok({
            "batch_id": batch_id,
            "library": batch.get("library", ""),
            "total": batch.get("total", 0),
            "current": batch.get("current", 0),
            "done": batch.get("done", False),
            "cancelled": batch.get("cancelled", False),
            "ok_count": batch.get("ok_count", 0),
            "fail_count": batch.get("fail_count", 0),
            "removed_chunks": batch.get("removed_chunks", 0),
            "removed_chars": batch.get("removed_chars", 0),
            "errors": list(batch.get("errors", [])),
            "started_at": batch.get("started_at", 0),
        })


def handle_delete_cancel(method, path, query, body):
    """POST /api/files/delete-cancel —— 取消批量删除。

    body: {"batch_id":"..."}
    取消后批次停止处理下一个文件，已删除的不会回滚。
    """
    if method != "POST":
        return _err("需要 POST 方法", 405)
    batch_id = (body.get("batch_id") or "").strip()
    if not batch_id:
        return _err("缺少 batch_id 参数")
    with _delete_batches_lock:
        batch = _delete_batches.get(batch_id)
        if batch is None:
            return _err(f"批次不存在或已过期: {batch_id}", 404)
        batch["cancelled"] = True
        return _ok({"batch_id": batch_id, "cancelled": True})


def handle_delete_all_files(method, path, query, body, user=None):
    """POST /api/files/delete-all —— 删除库内全部文件（快速路径：删库重建）。

    body: {"library":"xxx"}
    流程：
      1. 记录原库的 name/note/folder
      2. 关闭所有 mmap 缓存
      3. 删除原库（数据 + 注册表项）
      4. 用同名同备注同文件夹重建空库
    返回：{name, note, folder, recreated: true}
    """
    if method != "POST":
        return _err("需要 POST 方法", 405)
    lib_name = (body.get("library") or "").strip()
    if not lib_name:
        return _err("缺少 library 参数")
    reg = _registry()
    lib, err = _get_lib(reg, lib_name, user, write=True)
    if err:
        return err
    old_note = lib.note or ""
    old_folder = lib.folder or ""
    old_owner = lib.owner

    # 关闭所有 mmap 缓存，避免文件占用
    from indexer import ZoneIndex
    mgr = lib.manager(BASE_DIR)
    for z in mgr.list_zones():
        ZoneIndex.invalidate(z.index_dir)
    # 同步清理语义向量索引的内存缓存与磁盘文件
    old_lib_root = lib.abs_path(BASE_DIR)
    try:
        from semantic_manager import get_manager
        get_manager(BASE_DIR).remove_files(old_lib_root)
    except Exception as e:
        _logger.warning("清理语义索引失败 %s: %s", lib_name, e)

    try:
        reg.remove(lib.name, delete_data=True, owner=lib.owner)
    except OSError as e:
        return _err(f"删除库失败（文件被占用）: {e}", 500)
    except ValueError as e:
        return _err(str(e), 404)

    # 重建同名同备注同文件夹（保留属主）的库
    try:
        new_lib = reg.create(lib_name, note=old_note, owner=old_owner)
    except ValueError as e:
        return _err(f"重建库失败: {e}", 500)
    if old_folder:
        try:
            reg.move_library(lib_name, old_folder, owner=old_owner)
        except Exception:
            pass
    return _ok({
        "name": new_lib.name,
        "note": new_lib.note,
        "folder": old_folder,
        "recreated": True,
    })


def handle_list_files(method, path, query, body, user=None):
    """列出库内所有源文件（基于 dedup 索引）。

    GET /api/files?library=xxx&search=关键词&ext=txt&page=1&page_size=50
    - search: 模糊匹配文件名（不区分大小写）
    - ext: 按扩展名筛选（如 txt/docx/pdf/md）
    - page/page_size: 分页
    """
    if method != "GET":
        return _err("需要 GET 方法", 405)
    lib_name = query.get("library")
    if not lib_name:
        return _err("缺少 library 参数")
    reg = _registry()
    lib, err = _get_lib(reg, lib_name, user, write=False)
    if err:
        return err
    mgr = lib.manager(BASE_DIR)

    from dedup import DedupIndex
    dedup = DedupIndex(mgr.root)
    all_files = dedup.all()  # {sha: {file_name, file_path, zone_id, imported_at, char_count}}

    search = query.get("search", "").lower().strip()
    ext_filter = query.get("ext", "").lower().strip().lstrip(".")
    try:
        page = max(1, int(query.get("page", "1")))
    except ValueError:
        page = 1
    try:
        page_size = max(1, min(500, int(query.get("page_size", "50"))))
    except ValueError:
        page_size = 50

    items = []
    for sha, info in all_files.items():
        fname = info.get("file_name", "")
        fpath = info.get("file_path", "")
        # 按文件名搜索
        if search and search not in fname.lower() and search not in fpath.lower():
            continue
        # 按扩展名筛选
        if ext_filter:
            fext = os.path.splitext(fname)[1].lower().lstrip(".")
            if fext != ext_filter:
                continue
        items.append({
            "sha256": sha,
            "file_name": fname,
            "file_path": fpath,
            "zone_id": info.get("zone_id", ""),
            "imported_at": info.get("imported_at", ""),
            "char_count": info.get("char_count", 0),
            "ext": os.path.splitext(fname)[1].lower().lstrip(".") or "(无)",
        })

    # 按导入时间倒序（新的在前）
    items.sort(key=lambda x: x["imported_at"], reverse=True)

    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = items[start:end]

    # 统计各类型数量
    ext_stats: dict = {}
    for it in items:
        ext_stats[it["ext"]] = ext_stats.get(it["ext"], 0) + 1

    return _ok({
        "library": lib_name,
        "files": page_items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
        "ext_stats": ext_stats,
    })


def handle_verify(method, path, query, body, user=None):
    from verifier import verify_all
    global _verify_running
    # 校验是重 I/O 操作（多线程并行读 chunk 会触发磁盘抖动、耗尽浏览器连接池）
    # 加并发锁：同一时刻只允许一个校验任务执行
    if _verify_running:
        return _err("已有校验操作正在进行，请等待完成", 409)
    if not _verify_lock.acquire(blocking=False):
        return _err("已有校验操作正在进行，请等待完成", 409)
    _verify_running = True
    try:
        lib_name = query.get("library")
        reg = _registry()
        if lib_name:
            # 单库校验（可写校验：verify 会读全量 chunk，重 IO）
            lib, err = _get_lib(reg, lib_name, user, write=True)
            if err:
                return err
            mgr = lib.manager(BASE_DIR)
            result = verify_all(mgr)
            return _ok({"libraries": [{
                "name": lib.name,
                "note": lib.note,
                "folder": lib.folder or "",
                "result": result,
            }]})
        # 全部库校验（仅当前用户可见的库）
        out = []
        for lib in reg.list_libraries_for(user["username"], user["is_admin"]):
            mgr = lib.manager(BASE_DIR)
            result = verify_all(mgr)
            out.append({
                "name": lib.name,
                "note": lib.note,
                "folder": lib.folder or "",
                "result": result,
            })
        return _ok({"libraries": out, "folders": reg.list_folders()})
    finally:
        _verify_running = False
        _verify_lock.release()


# 校验操作并发锁：与 _fix_lock 同机制，避免并发校验拖垮磁盘 IO
_verify_lock = threading.Lock()
_verify_running = False


# 修复操作全局锁：确保补抽标签/重建索引/构建向量索引/补统计 串行执行
# 这些操作涉及大量文件读写和 CPU 计算，并发会导致数据损坏
_fix_lock = threading.Lock()
_fix_running = False  # 标记是否有修复操作正在进行


def handle_verify_fix_stream(handler, query, user=None):
    """SSE 流式修复 API：串行执行选定的修复项。

    query 参数：
    - library: 库名（必填）
    - fixes: JSON 数组，如 ["tags","index","semantic","stats"]
    """
    global _fix_running
    lib_name = query.get("library", "")
    fixes_json = query.get("fixes", "[]")
    try:
        fixes = json.loads(fixes_json) if isinstance(fixes_json, str) else fixes_json
    except json.JSONDecodeError:
        fixes = []
    if not lib_name:
        handler._send(400, {"ok": False, "error": "缺少 library 参数"})
        return
    if not fixes:
        handler._send(400, {"ok": False, "error": "缺少 fixes 参数"})
        return

    reg = _registry()
    lib, err = _get_lib(reg, lib_name, user, write=True)
    if err:
        handler._send(err[0], err[1])
        return

    # 检查是否有修复操作正在进行（单线程）
    if _fix_running:
        handler._send(409, {"ok": False, "error": "已有修复操作正在进行，请等待完成"})
        return

    # 检查是否有导入操作正在进行
    with _import_batches_lock:
        for bid, b in _import_batches.items():
            if not b.get("done") and b.get("library") == lib_name:
                handler._send(409, {"ok": False, "error": "该库有导入操作正在进行，请等待完成"})
                return

    mgr = lib.manager(BASE_DIR)

    def gen():
        global _fix_running
        # 获取锁（单线程串行）
        if not _fix_lock.acquire(blocking=False):
            yield {"phase": "error", "error": "已有修复操作正在进行，请等待完成"}
            return
        _fix_running = True
        try:
            total = len(fixes)
            yield {"phase": "start", "total": total, "fixes": fixes}
            done = 0
            for fix_type in fixes:
                yield {"phase": "progress", "current": done + 1, "total": total,
                       "fix": fix_type, "status": "processing"}
                try:
                    if fix_type == "tags":
                        # _fix_tags 现为生成器：逐 chunk yield 进度，末尾 yield 结果 dict
                        result = None
                        for ev in _fix_tags(mgr):
                            if "ok" in ev:
                                result = ev
                            else:
                                yield {"phase": "sub_progress",
                                       "current": ev.get("current", 0),
                                       "total": ev.get("total", 0),
                                       "chunk_id": ev.get("chunk_id", ""),
                                       "tags_count": ev.get("tags_count", 0),
                                       "skipped": ev.get("skipped", False),
                                       "error": ev.get("error")}
                        if result is None:
                            result = {"ok": True, "fixed": 0, "skipped": 0,
                                      "failed": [], "action": "补抽标签"}
                    elif fix_type == "index":
                        result = _fix_index(mgr)
                    elif fix_type == "semantic":
                        result = _fix_semantic(mgr, lib_name)
                    elif fix_type == "stats":
                        result = _fix_stats(mgr)
                    else:
                        result = {"ok": False, "error": f"未知修复类型: {fix_type}"}
                except Exception as e:
                    result = {"ok": False, "error": str(e)}
                done += 1
                yield {"phase": "progress", "current": done, "total": total,
                       "fix": fix_type, "status": "done" if result.get("ok") else "fail",
                       "result": result}
            yield {"phase": "done", "total": total, "done": done}
        finally:
            _fix_running = False
            _fix_lock.release()

    handler._send_sse_stream(gen())


def _fix_tags(mgr):
    """补抽标签：给缺失 tags 的 chunk 补充提取标签。

    生成器模式：逐 chunk yield 进度，参照 handle_backfill_tags_stream。
    yield 字段：{current, total, chunk_id, tags_count, skipped, error(可选)}
    末尾 yield 最终结果 dict：{ok, fixed, skipped, failed, action}
    """
    from tagger import extract_tags
    from settings import SettingsStore
    store = SettingsStore(AUTH_DIR)
    top_k = int(store.get("tag_top_k", 8))

    # 先收集所有 chunk 文件路径，便于计算 total 与逐条进度
    chunk_files = []
    for zone in mgr.list_zones():
        for path in zone.iter_chunk_files():
            chunk_files.append(path)
    total = len(chunk_files)

    fixed = 0
    skipped = 0
    failed = []  # 记录读取/写入失败的 chunk，便于前端展示与日志定位
    for i, path in enumerate(chunk_files, 1):
        try:
            with open(path, "r", encoding="utf-8") as f:
                chunk = json.load(f)
            if chunk.get("tags"):
                skipped += 1
                yield {"current": i, "total": total,
                       "chunk_id": chunk.get("chunk_id", ""),
                       "tags_count": len(chunk.get("tags") or []), "skipped": True}
                continue
            text = chunk.get("text", "")
            if not text:
                skipped += 1
                yield {"current": i, "total": total,
                       "chunk_id": chunk.get("chunk_id", ""),
                       "tags_count": 0, "skipped": True}
                continue
            tags = extract_tags(text, top_k=top_k)
            chunk["tags"] = tags
            # 原子写入
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(chunk, f, ensure_ascii=False)
            os.replace(tmp, path)
            fixed += 1
            yield {"current": i, "total": total,
                   "chunk_id": chunk.get("chunk_id", ""),
                   "tags_count": len(tags), "skipped": False}
        except Exception as e:
            failed.append({"chunk": os.path.basename(path), "error": str(e)})
            # 记录失败 path 与异常，便于运维定位
            _logger.warning("_fix_tags 失败 %s: %s", path, e)
            yield {"current": i, "total": total,
                   "chunk_id": "", "tags_count": 0, "skipped": True, "error": str(e)}
            continue
    yield {"ok": True, "fixed": fixed, "skipped": skipped, "failed": failed,
           "action": "补抽标签"}


def _fix_index(mgr) -> dict:
    """重建倒排索引：全量重建所有 zone 的索引。"""
    from indexer import ZoneIndex
    total_rebuilt = 0
    for zone in mgr.list_zones():
        zi = ZoneIndex.get(zone.index_dir)
        stat = zi.rebuild(zone.chunks_dir, zone.zone_id)
        zi.cleanup_merged_idx(zone.chunks_dir)
        total_rebuilt += stat.get("merged", 0)
    return {"ok": True, "fixed": total_rebuilt, "action": "重建倒排索引"}


def _fix_semantic(mgr, lib_name) -> dict:
    """构建向量索引：触发异步构建。"""
    from semantic_manager import get_manager
    mgr_sem = get_manager()
    if not mgr_sem.available():
        reason = mgr_sem.fail_reason() or "未知原因"
        return {"ok": False, "error": f"向量索引不可用（{reason}）"}
    ret = mgr_sem.trigger_build_async(mgr.root, lib_name=lib_name)
    if ret.get("started"):
        return {"ok": True, "fixed": 0, "action": "构建向量索引（已异步触发）"}
    reason = ret.get("reason") or "无需重建"
    return {"ok": False, "error": f"未能启动向量索引构建：{reason}"}


def _fix_stats(mgr) -> dict:
    """补充元数据统计：给缺失 stats 的 chunk 补充朝代/主题/实体密度统计。

    同时更新库级 _doc_stats.json 汇总文件，供统计界面展示文档元数据。
    （导入时由 importer.py 写入该文件，但旧库在元数据功能上线前导入，
    需通过本函数重建。）
    """
    from metadata_stats import compute_chunk_stats, sliding_window_average, summarize_document_stats
    fixed = 0
    # 库级文档元数据汇总：file_name -> {朝代/主题/实体密度, relative_dir, char_count, chunk_count}
    doc_stats_path = os.path.join(mgr.root, "_doc_stats.json")
    all_doc_stats = {}
    if os.path.isfile(doc_stats_path):
        try:
            with open(doc_stats_path, "r", encoding="utf-8") as f:
                all_doc_stats = json.load(f)
        except Exception:
            all_doc_stats = {}
    for zone in mgr.list_zones():
        # 收集同文档的 chunk 统计，用于滑动平均
        # 注意：不跳过已有 stats 的 chunk，确保用户主动执行时全量重算
        # （用于补充新增字段如 top_persons，或重新生成统计）
        doc_chunks = {}  # file_name -> [(path, chunk_data), ...]
        for path in zone.iter_chunk_files():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    chunk = json.load(f)
                fn = chunk.get("source", {}).get("file_name", "")
                if fn not in doc_chunks:
                    doc_chunks[fn] = []
                doc_chunks[fn].append((path, chunk))
            except Exception:
                continue

        for fn, items in doc_chunks.items():
            if not items:
                continue
            # 对每个 chunk 做统计
            stats_list = []
            for path, chunk in items:
                text = chunk.get("text", "")
                stats = compute_chunk_stats(text)
                stats_list.append(stats)
            # 滑动平均 + 文档级汇总
            smoothed = sliding_window_average(stats_list, window=1)
            doc_stats = summarize_document_stats(stats_list)
            # 写回 chunk，并累计文档级字符数/目录信息
            doc_char_count = 0
            relative_dir = ""
            for (path, chunk), s in zip(items, smoothed):
                try:
                    chunk["source"]["stats"] = s
                    chunk["source"]["doc_stats"] = doc_stats
                    relative_dir = chunk["source"].get("relative_dir", "")
                    doc_char_count += chunk.get("text_length", len(chunk.get("text", "")))
                    tmp = path + ".tmp"
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump(chunk, f, ensure_ascii=False)
                    # 复用 transaction 的重试工具，避免杀软/索引服务短暂占用导致写回失败
                    from transaction import _replace_with_retry
                    _replace_with_retry(tmp, path)
                    fixed += 1
                except Exception:
                    continue
            # 更新库级汇总：与 importer.py 写入结构一致
            all_doc_stats[fn] = {
                **doc_stats,
                "relative_dir": relative_dir,
                "char_count": doc_char_count,
                "chunk_count": len(items),
            }
    # 写入库级 _doc_stats.json（原子写入，带重试）
    if fixed > 0:
        try:
            tmp = doc_stats_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(all_doc_stats, f, ensure_ascii=False, indent=2)
            from transaction import _replace_with_retry
            _replace_with_retry(tmp, doc_stats_path)
        except Exception:
            pass
    return {"ok": True, "fixed": fixed, "action": "补充元数据统计"}


def handle_build_index(method, path, query, body, user=None):
    from indexer import ZoneIndex
    if method != "POST":
        return _err("需要 POST 方法", 405)
    lib_name = body.get("library")
    if not lib_name:
        return _err("缺少 library 参数")
    # mode: "merge"（默认，增量合并） | "rebuild"（全量重建，清理损坏索引）
    mode = body.get("mode", "merge")
    reg = _registry()
    lib, err = _get_lib(reg, lib_name, user, write=True)
    if err:
        return err
    mgr = lib.manager(BASE_DIR)
    out = []
    for z in mgr.list_zones():
        zi = ZoneIndex(z.index_dir)
        if mode == "rebuild":
            # 全量重建：清空现有索引文件后从 chunks 重新构建
            # 用于修复索引文件损坏（写入中断导致行错位）
            stat = zi.rebuild(z.chunks_dir, z.zone_id)
            cleaned = zi.cleanup_merged_idx(z.chunks_dir)
            out.append({"zone": z.zone_id, "mode": "rebuild",
                        "merged": stat.get("merged", 0), "cleaned": cleaned})
        else:
            stat = zi.merge_zone_chunks(z.chunks_dir, z.zone_id)
            cleaned = zi.cleanup_merged_idx(z.chunks_dir)
            out.append({"zone": z.zone_id, "mode": "merge",
                        "merged": stat["merged"],
                        "skipped": stat["skipped"], "cleaned": cleaned})
    return _ok({"zones": out})


def handle_recover(method, path, query, body, user=None):
    from transaction import recover_all_zones
    if method != "POST":
        return _err("需要 POST 方法", 405)
    lib_name = body.get("library")
    if not lib_name:
        return _err("缺少 library 参数")
    reg = _registry()
    lib, err = _get_lib(reg, lib_name, user, write=True)
    if err:
        return err
    mgr = lib.manager(BASE_DIR)
    results = recover_all_zones(mgr)
    return _ok({"recovered": [{"zone": z, "action": a} for z, a in results]})


def _filter_refs_by_content(content, refs):
    """根据答案中实际出现的 [n] 引用标记过滤 references 列表。

    只保留模型在答案中真正引用过的条目。若答案中没有任何 [n] 标记
    （模型未引用任何资料），则保留全部 references，避免引用列表为空。

    注意：原文资料中可能含有 [数字] 形式的注解编号（如史书脚注号 [91]），
    这些不是模型引用标记。通过检查编号是否在 refs 范围内来区分：
    - 编号在 refs 范围内：视为有效引用
    - 编号超出 refs 范围：视为原文注解，忽略
    """
    if not content or not refs:
        return refs
    # refs 的有效编号集合
    valid_indices = {r.get("index") for r in refs if r.get("index") is not None}
    max_index = max(valid_indices) if valid_indices else 0

    cited = set()
    for m in re.finditer(r'\[(\d+)\]', content):
        try:
            n = int(m.group(1))
            # 只接受在 refs 范围内的编号，超出范围的视为原文注解号（如 [91]）
            if n in valid_indices:
                cited.add(n)
        except ValueError:
            pass
    if not cited:
        return refs
    filtered = [r for r in refs if r.get("index") in cited]
    # 若模型引用的编号全都不在 refs 中（幻觉），回退为全部保留
    return filtered if filtered else refs


def _clean_invalid_citations(content, refs):
    """清洗答案中不在 refs 列表范围内的 [n] 引用标记。

    模型可能保留原文注解号（如史书脚注 [91]）或幻觉输出超出范围的编号（如 [100]），
    这些编号与引用列表（通常是 [1]~[N]）不对应，会让用户产生困惑。

    处理策略：把不在 refs 中的 [n] 替换为不带方括号的纯数字（保留原文信息但不与引用列表混淆）。
    在 refs 中的 [n] 保持原样。

    返回清洗后的 content（字符串）。
    """
    if not content or not refs:
        return content
    valid_indices = {r.get("index") for r in refs if r.get("index") is not None}
    if not valid_indices:
        return content

    def _replace(m):
        n = int(m.group(1))
        if n in valid_indices:
            return m.group(0)  # 保留原样 [n]
        return str(n)  # 去掉方括号，仅保留数字

    return re.sub(r'\[(\d+)\]', _replace, content)


def handle_backfill_tags_stream(handler, query, user=None):
    """GET /api/backfill-tags/stream —— SSE 流式为旧库 chunk 补抽标签。

    参数（query）：
        library : 库名（必填）
        top_k   : 每个 chunk 提取的标签数（可选，默认从 settings 读取）
        force   : "1"/"0"，是否覆盖已有标签（默认 0，仅补抽 tags 缺失的 chunk）

    事件 phase：
        start    : {total}
        progress : {current, total, chunk_id, tags_count}
        done     : {updated, skipped, total}
        error    : {message}
    """
    lib_name = (query.get("library") or "").strip()
    if not lib_name:
        handler._send(400, {"ok": False, "error": "缺少 library 参数"})
        return

    reg = _registry()
    lib, err = _get_lib(reg, lib_name, user, write=True)
    if err:
        handler._send(err[0], err[1])
        return

    # 读取 top_k 配置
    try:
        from settings import SettingsStore
        store = SettingsStore(AUTH_DIR)
        top_k = int(store.get("tag_top_k", 10))
        top_k = max(0, min(top_k, 30))
    except Exception:
        top_k = 10
    if top_k == 0:
        handler._send(400, {"ok": False, "error": "tag_top_k 设为 0，标签提取已禁用"})
        return

    force = query.get("force", "0") in ("1", "true", "True")

    mgr = lib.manager(BASE_DIR)

    def _gen():
        from tagger import extract_tags
        import json as _json

        # 先收集所有 chunk 文件路径
        chunk_files = []
        for z in mgr.list_zones():
            chunks_dir = z.chunks_dir
            if not os.path.isdir(chunks_dir):
                continue
            for name in sorted(os.listdir(chunks_dir)):
                if name.endswith(".json") and name.startswith("chunk_"):
                    chunk_files.append((z, os.path.join(chunks_dir, name)))

        total = len(chunk_files)
        yield {"phase": "start", "total": total}

        updated = 0
        skipped = 0
        for i, (z, chunk_path) in enumerate(chunk_files, 1):
            try:
                with open(chunk_path, "r", encoding="utf-8") as f:
                    chunk = _json.load(f)
                # 已有标签且非 force 则跳过
                existing_tags = chunk.get("tags")
                if existing_tags and not force:
                    skipped += 1
                    yield {"phase": "progress", "current": i, "total": total,
                           "chunk_id": chunk.get("chunk_id", ""),
                           "tags_count": len(existing_tags), "skipped": True}
                    continue
                text = chunk.get("text", "")
                if not text or len(text) < 10:
                    skipped += 1
                    yield {"phase": "progress", "current": i, "total": total,
                           "chunk_id": chunk.get("chunk_id", ""),
                           "tags_count": 0, "skipped": True}
                    continue
                tags = extract_tags(text, top_k=top_k)
                chunk["tags"] = tags
                # 原子写回
                tmp = chunk_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    _json.dump(chunk, f, ensure_ascii=False, separators=(",", ":"))
                os.replace(tmp, chunk_path)
                updated += 1
                yield {"phase": "progress", "current": i, "total": total,
                       "chunk_id": chunk.get("chunk_id", ""),
                       "tags_count": len(tags), "skipped": False}
            except Exception as e:
                skipped += 1
                yield {"phase": "progress", "current": i, "total": total,
                       "chunk_id": "", "tags_count": 0, "skipped": True,
                       "error": str(e)}

        yield {"phase": "done", "updated": updated, "skipped": skipped,
               "total": total}

    handler._send_sse_stream(_gen())


# ============================================================
#  路由
# ============================================================

INDEX_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>全文检索系统 API</title>
<style>body{font-family:sans-serif;max-width:900px;margin:40px auto;padding:0 20px;color:#333}
h1{color:#2c3e50}table{border-collapse:collapse;width:100%}
td,th{border:1px solid #ddd;padding:8px;text-align:left}th{background:#f4f4f4}
code{background:#f4f4f4;padding:2px 6px;border-radius:3px}.m{color:#888;font-size:0.9em}</style>
</head><body>
<h1>全文检索系统 API</h1>
<p>所有响应均为 JSON，格式：<code>{"ok": true, "data": ...}</code> 或 <code>{"ok": false, "error": "..."}</code></p>
<table>
<tr><th>方法</th><th>路径</th><th>说明</th></tr>
<tr><td>GET</td><td><code>/api/libraries</code></td><td>列出所有库</td></tr>
<tr><td>POST</td><td><code>/api/libraries</code></td><td>创建库 body: {"name":"...","note":"..."}</td></tr>
<tr><td>PATCH</td><td><code>/api/libraries/{name}</code></td><td>修改备注 body: {"note":"..."}</td></tr>
<tr><td>DELETE</td><td><code>/api/libraries/{name}?yes=1</code></td><td>删除库</td></tr>
<tr><td>GET</td><td><code>/api/stats?library=xxx</code></td><td>统计（library 可选）</td></tr>
<tr><td>GET</td><td><code>/api/search?query=xxx&libraries=A,B&parallel=4&top=20</code></td><td>跨库并行搜索</td></tr>
<tr><td>POST</td><td><code>/api/import</code></td><td>导入文件 body: {"files":["path"],"library":"xxx","force":false}</td></tr>
<tr><td>DELETE</td><td><code>/api/files?library=xxx&ext=txt</code></td><td>删除库内文件（按扩展名/SHA）</td></tr>
<tr><td>GET</td><td><code>/api/verify?library=xxx</code></td><td>校验库</td></tr>
<tr><td>POST</td><td><code>/api/build-index</code></td><td>重建索引 body: {"library":"xxx"}</td></tr>
<tr><td>POST</td><td><code>/api/recover</code></td><td>恢复事务 body: {"library":"xxx"}</td></tr>
</table>
<h2>测试示例</h2>
<pre>
# 列出库
curl <span id="b"></span>/api/libraries

# 搜索
curl "<span id="b2"></span>/api/search?query=郎溪&parallel=4&top=5"

# 创建库
curl -X POST <span id="b3"></span>/api/libraries -H "Content-Type: application/json" -d '{"name":"新库","note":"备注"}'
</pre>
<script>
const u=location.origin;
['b','b2','b3'].forEach(id=>document.getElementById(id).textContent=u);
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # 不输出到控制台（防止长时运行控制台缓冲膨胀）；记录到日志文件
        _logger.info("%s %s", self.command, self.path)

    def _send(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse_stream(self, event_generator):
        """以 Server-Sent Events 方式持续推送事件。

        每个 event 被 JSON 序列化后写成 `data: {...}\n\n`。
        客户端断开（reader.cancel/页面刷新/关闭）时，wfile.write 会抛异常，
        此时显式 close 生成器，停止后端 LLM 调用，避免旧流继续运行串台新流。
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            for event in event_generator:
                data = json.dumps(event, ensure_ascii=False)
                self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # 客户端断开连接：显式关闭生成器，触发其 finally 块，停止后端 LLM 调用
            try:
                event_generator.close()
            except Exception:
                pass
        except Exception as e:
            err = json.dumps({"phase": "error", "error": str(e)},
                             ensure_ascii=False)
            try:
                self.wfile.write(f"data: {err}\n\n".encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass
            # 其他异常也关闭生成器
            try:
                event_generator.close()
            except Exception:
                pass

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _handle_download(self, query):
        """GET /api/download?file_path=xxx&library=xxx —— 下载源文件到客户端浏览器。

        服务器-客户端模式下，原 open-doc 仅在服务器本机打开文件，
        局域网客户端无法使用；改为通过 HTTP 下载文件到客户端。

        安全约束（多用户）：library 必填且必须是当前用户可见的库；
        解析后的文件路径必须位于该库根目录内，防止 ../ 逃逸下载任意文件。
        """
        file_path = (query.get("file_path") or "").strip()
        if not file_path:
            self._send(400, {"ok": False, "error": "缺少 file_path 参数"})
            return
        library = query.get("library") or ""
        if not library:
            self._send(400, {"ok": False, "error": "缺少 library 参数"})
            return
        reg = _registry()
        lib, err = _get_lib(reg, library, self._user, write=False)
        if err:
            self._send(err[0], err[1])
            return
        lib_root = os.path.abspath(lib.abs_path(BASE_DIR))
        abs_path = _resolve_source_file(file_path, library)
        if abs_path is None or not os.path.isfile(abs_path):
            self._send(404, {
                "ok": False,
                "error": "文件不存在（可能已被移动或删除，或来自已清理的临时上传目录）",
            })
            return
        # 防目录逃逸：解析结果必须落在该库根目录内（拒绝 ../ 或任意绝对路径）
        abs_path = os.path.abspath(abs_path)
        try:
            if os.path.commonpath([abs_path, lib_root]) != lib_root:
                self._send(403, {"ok": False, "error": "禁止下载库外文件"})
                return
        except ValueError:
            self._send(403, {"ok": False, "error": "禁止下载库外文件"})
            return

        file_size = os.path.getsize(abs_path)
        # 文件名编码处理：RFC 5987，兼容中文文件名
        base_name = os.path.basename(abs_path)
        try:
            ascii_name = base_name.encode("ascii", "ignore").decode("ascii") or "download"
        except Exception:
            ascii_name = "download"
        quoted_name = urllib.parse.quote(base_name)

        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header(
            "Content-Disposition",
            f'attachment; filename="{ascii_name}"; filename*=UTF-8\'\'{quoted_name}',
        )
        self.send_header("Content-Length", str(file_size))
        self.end_headers()
        # 流式写入，避免大文件一次性占用内存
        try:
            with open(abs_path, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except Exception as e:
            _logger.warning("下载文件写入失败 %s: %s", abs_path, e)

    def _handle_ai_search_stream(self, query):
        """处理 GET /api/ai-search/stream，返回 SSE 流。"""
        question = (query.get("question") or "").strip()
        if not question:
            self._send(400, {"ok": False, "error": "缺少 question 参数"})
            return
        lib_param = query.get("libraries", "")
        libraries = [x for x in lib_param.split(",") if x] if lib_param else None
        try:
            parallel = int(query.get("parallel", "4"))
        except ValueError:
            parallel = 4
        try:
            top_k = int(query.get("top_k", "20"))
        except ValueError:
            top_k = 20
        try:
            temperature = float(query.get("temperature", "0.3"))
        except ValueError:
            temperature = 0.3
        try:
            client = _build_client_from_settings()
        except ValueError as e:
            self._send(400, {"ok": False, "error": str(e)})
            return
        reg = _registry()
        user = self._user
        if libraries:
            err_resp = _check_libraries_visible(reg, libraries, user)
            if err_resp:
                self._send(err_resp[0], err_resp[1])
                return
        else:
            libraries = [l.name for l in reg.list_libraries_for(user["username"], user["is_admin"])]
        from ai_search import ai_search_stream
        # 语义检索模式开关
        semantic_enabled = (query.get("semantic") or "0").strip() in ("1", "true", "yes")

        def gen():
            for event in ai_search_stream(
                question, reg, client, BASE_DIR,
                library_names=libraries,
                parallel=parallel, top_k=top_k, temperature=temperature,
            ):
                yield event

        self._send_sse_stream(gen())

    def _handle_ai_search_agent_stream(self, query):
        """处理 GET /api/ai-search/agent/stream，返回 Agent 模式 SSE 流。

        Agent 模式：让 LLM 主导检索流程（规划 → 检索 → 评估 → 重试 → 生成），
        解决直接用整句检索时高频字（如"郎溪"）压倒核心词组（如"经济发展"）的问题。

        当单库 chunk > 1800 时，自动切换到分块检索工作流：
        每块由检索智能体产出事实性回答 → 总结智能体合并。
        """
        question = (query.get("question") or "").strip()
        if not question:
            self._send(400, {"ok": False, "error": "缺少 question 参数"})
            return
        lib_param = query.get("libraries", "")
        libraries = [x for x in lib_param.split(",") if x] if lib_param else None
        try:
            parallel = int(query.get("parallel", "4"))
        except ValueError:
            parallel = 4
        try:
            top_k = int(query.get("top_k", "20"))
        except ValueError:
            top_k = 20
        try:
            temperature = float(query.get("temperature", "0.3"))
        except ValueError:
            temperature = 0.3
        try:
            max_rounds = int(query.get("max_rounds", "2"))
        except ValueError:
            max_rounds = 2
        if max_rounds < 1:
            max_rounds = 1
        if max_rounds > 4:
            max_rounds = 4
        # 语义检索模式开关（叠加在关键词检索上，仅 keyword 模式生效）
        semantic_enabled = (query.get("semantic") or "0").strip() in ("1", "true", "yes")
        # 检索引擎切换：keyword=关键词模式 | semantic=向量语义模式
        # 由前端滑块控制，决定走关键词检索入口还是向量检索入口
        search_engine = (query.get("search_engine") or "keyword").strip()
        # 子模式：global / precise / complex（向量模式按子模式分发到不同入口函数）
        submode = (query.get("submode") or "global").strip()
        try:
            client = _build_client_from_settings()
        except ValueError as e:
            self._send(400, {"ok": False, "error": str(e)})
            return
        reg = _registry()
        user = self._user
        if libraries:
            err_resp = _check_libraries_visible(reg, libraries, user)
            if err_resp:
                self._send(err_resp[0], err_resp[1])
                return
        else:
            libraries = [l.name for l in reg.list_libraries_for(user["username"], user["is_admin"])]

        # Agent 工作流：LLM 自主调用工具完成检索+生成
        # 独立于 keyword/semantic 引擎和 global/precise/complex 子模式，
        # 也跳过分块检索（agent 自己决定检索范围）
        if submode == "agent_workflow":
            from agent_workflow import agent_workflow_stream as _awf
            # 接收前端可选传入的 extra_context（用户从检索结果挑选的资料）
            extra_context = None
            # 最大轮次：从设置读取（默认15，范围3-30）
            max_rounds = int(_settings().get("agent_workflow_max_rounds", 15))
            max_rounds = max(3, min(max_rounds, 30))
            def gen():
                for event in _awf(
                    question, reg, client, BASE_DIR,
                    library_names=libraries,
                    temperature=temperature,
                    max_rounds=max_rounds,
                    extra_context=extra_context,
                ):
                    yield event
            self._send_sse_stream(gen())
            return
        else:
            # reflexion 已移除：submode 仅支持 agent_workflow
            self._send(400, {"ok": False, "error": f"不支持的子模式: {submode}"})
            return

    def _handle_chat_send(self, query, session_id):
        """处理 GET /api/chat/sessions/{id}/send?question=...，返回 SSE 流。

        多轮对话：把会话历史拼进对话 messages，让模型能"接着问"。
        生成结束后把用户消息 + 助手消息持久化到会话。
        """
        question = (query.get("question") or "").strip()
        if not question:
            self._send(400, {"ok": False, "error": "缺少 question 参数"})
            return
        try:
            parallel = int(query.get("parallel", "4"))
        except ValueError:
            parallel = 4
        try:
            top_k = int(query.get("top_k", "20"))
        except ValueError:
            top_k = 20
        try:
            temperature = float(query.get("temperature", "0.3"))
        except ValueError:
            temperature = 0.3
        # mode: agent（精读+工具展开）| chat（纯对话）
        mode = (query.get("mode") or "agent").strip()
        # submode: global（全局检索）| precise（精准检索）| complex（复杂问题），仅 agent 模式用
        submode = (query.get("submode") or "global").strip()
        # effort: full | boost | standard | economy（agent 模式用）
        effort = (query.get("effort") or "standard").strip()
        # max_rounds：智能体工作流最大轮次（从设置读取，默认15，范围3-30）
        max_rounds = int(_settings().get("agent_workflow_max_rounds", 15))
        max_rounds = max(3, min(max_rounds, 30))
        # max_mini_chunks：agent 模式小 chunk 上限（用户可调）
        try:
            max_mini_chunks = int(query.get("max_mini_chunks", "100"))
        except ValueError:
            max_mini_chunks = 100
        max_mini_chunks = max(1, min(max_mini_chunks, 500))
        # semantic: 语义检索模式开关（1=开启，叠加在所有 agent 子模式上）
        semantic_enabled = (query.get("semantic") or "0").strip() in ("1", "true", "yes")
        # search_engine: 检索引擎切换（keyword=关键词模式，semantic=向量语义模式）
        # 由前端滑块控制，影响 agent 模式的检索入口函数选择
        search_engine = (query.get("search_engine") or "keyword").strip()

        store = _chat_store()
        session = store.get_session(session_id)
        if session is None:
            self._send(404, {"ok": False, "error": f"会话不存在: {session_id}"})
            return
        # 多用户：只能操作自己的会话（管理员可操作全部）
        user = self._user
        s_owner = store.session_owner(session_id)
        if s_owner == "" or (not user["is_admin"] and s_owner != user["username"]):
            self._send(404, {"ok": False, "error": f"会话不存在: {session_id}"})
            return
        libraries = session.get("libraries") or None
        if libraries:
            reg0 = _registry()
            err_resp = _check_libraries_visible(reg0, libraries, user)
            if err_resp:
                self._send(err_resp[0], err_resp[1])
                return
        history = session.get("messages", [])
        # 构造精简历史：只要问题和最终回答，不含 reasoning/工具调用等细节
        # 取最近 3 轮（6 条消息），供 agent 模式参考上下文
        slim_history = []
        recent_msgs = history[-6:] if len(history) > 6 else history
        for m in recent_msgs:
            role = m.get("role", "")
            content = m.get("content", "")
            if role in ("user", "assistant") and content:
                slim_history.append({
                    "role": role,
                    "content": content,
                    "references": m.get("references", []) if role == "assistant" else [],
                })
        # 读取用户从检索结果挑选的额外上下文（注入对话模式）
        extra_context = store.get_extra_context(session_id)

        try:
            client = _build_client_from_settings()
        except ValueError as e:
            self._send(400, {"ok": False, "error": str(e)})
            return
        reg = _registry()

        # 先持久化用户消息（让流式过程中前端能看到历史）
        user_msg = store.add_message(session_id, {"role": "user", "content": question})
        # 预占一条 assistant 消息，流式结束后回填
        assistant_msg = store.add_message(
            session_id, {"role": "assistant", "content": "",
                         "mode": mode},
        )

        if mode == "chat":
            from ai_search import simple_chat_stream as _stream_fn
        elif submode == "agent_workflow":
            from agent_workflow import agent_workflow_stream as _stream_fn
            mode = "agent"
        else:
            self._send(400, {"ok": False, "error": f"不支持的子模式: {submode}"})
            return

        def gen():
            # 收集最终答案、思考、引用
            final_content = ""
            final_reasoning = ""
            final_refs = []
            final_retrieval = {}
            final_queries = []
            # 工具调用记录（独立保存，不混入 queries，避免前端渲染 [object Object]）
            final_tool_calls = []
            try:
                if mode == "agent":
                    if submode == "agent_workflow":
                        # Agent 工作流：跳过分块检索，LLM 自主调用工具
                        for event in _stream_fn(
                            question, reg, client, BASE_DIR,
                            library_names=libraries,
                            temperature=temperature,
                            max_rounds=max_rounds,
                            extra_context=extra_context,
                            history=slim_history,
                        ):
                            if event.get("phase") == "done":
                                event = dict(event, message_id=assistant_msg["id"])
                            yield event
                            ph = event.get("phase")
                            if ph == "content":
                                final_content += event.get("delta", "")
                            elif ph == "reasoning":
                                final_reasoning += event.get("delta", "")
                            elif ph == "retrieval":
                                # 规划行动模式：在 done 之前发送引用来源
                                final_refs = event.get("references", [])
                                final_retrieval = event.get("retrieval", {})
                                final_queries = event.get("queries", [])
                            elif ph == "tool_call":
                                # 工具调用记录独立保存，不混入 queries
                                final_tool_calls.append({
                                    "tool": event.get("name"),
                                    "args": event.get("arguments", {}),
                                    "round": event.get("round", 0),
                                })
                else:
                    # chat 模式：纯对话，不检索（注入 extra_context 作为参考资料）
                    for event in _stream_fn(question, history, client,
                                            temperature=temperature,
                                            extra_context=extra_context):
                        if event.get("phase") == "done":
                            event = dict(event, message_id=assistant_msg["id"])
                        yield event
                        ph = event.get("phase")
                        if ph == "content":
                            final_content += event.get("delta", "")
                        elif ph == "reasoning":
                            final_reasoning += event.get("delta", "")
                # 流结束，根据答案中实际引用的 [n] 标记过滤 references
                # 注意：agent_workflow 模式不按 [n] 标记过滤
                #   - agent_workflow：后端通过 accessed_chunks 追踪模型实际访问的 chunk
                # 只有普通 agent 模式才按 [n] 标记过滤
                if submode != "agent_workflow":
                    filtered_refs = _filter_refs_by_content(final_content, final_refs)
                    if filtered_refs is not final_refs:
                        final_refs = filtered_refs
                        # 通知前端更新引用列表显示
                        yield {"phase": "refs_update", "references": final_refs}
                    # 清洗答案中不在 refs 列表范围内的 [n] 标记
                    # （如原文注解号 [91] 或模型幻觉编号 [100]），避免与引用列表混淆
                    cleaned_content = _clean_invalid_citations(final_content, final_refs)
                    if cleaned_content != final_content:
                        final_content = cleaned_content
                        # 通知前端用清洗后的内容替换显示
                        yield {"phase": "content_update", "content": final_content}
                # 流结束，回填 assistant 消息
                # 保存最终回答和思考过程（reasoning），供历史会话展示
                store.update_message(session_id, assistant_msg["id"], {
                    "content": final_content,
                    "references": final_refs,
                    "mode": mode,
                    "reasoning": final_reasoning,
                    "queries": final_queries,
                })
            except Exception as e:
                # 异常时也回填已生成的部分
                store.update_message(session_id, assistant_msg["id"], {
                    "content": final_content or f"（生成失败: {e}）",
                    "references": final_refs,
                    "mode": mode,
                    "error": str(e),
                    "reasoning": final_reasoning,
                    "queries": final_queries,
                })
            finally:
                # 客户端断开（页面刷新/关闭）时，Python 生成器会被 GeneratorExit 关闭，
                # 此时 try/except 都不会触发，但 finally 会执行。
                # 确保无论如何都回填，避免会话留下空 assistant 消息导致无法加载。
                try:
                    # 检查是否已回填（content 非空或已有 error），避免重复写入
                    msgs = store.get_messages(session_id)
                    existing = None
                    for m in msgs:
                        if m.get("id") == assistant_msg["id"]:
                            existing = m
                            break
                    if existing and not existing.get("content") and not existing.get("error"):
                        store.update_message(session_id, assistant_msg["id"], {
                            "content": final_content or "（生成被中断）",
                            "references": final_refs,
                            "mode": mode,
                            "interrupted": True,
                            "reasoning": final_reasoning,
                            "queries": final_queries,
                        })
                except Exception:
                    pass

        self._send_sse_stream(gen())

    def _parse(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
        return path, query

    def do_GET(self):
        path, query = self._parse()
        self._user = _current_user(self)
        if path == "/" or path == "":
            # 读取静态首页文件，找不到则回退到内置 INDEX_HTML
            html_path = _resolve_static_path()
            if os.path.isfile(html_path):
                with open(html_path, "rb") as f:
                    body = f.read()
            else:
                body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/auth/me":
            st, pl = handle_auth_me("GET", path, query, {}, self._user)
            self._send(st, pl)
            return
        if path == "/api/auth/users":
            st, pl = handle_auth_users("GET", path, query, {}, self._user)
            self._send(st, pl)
            return
        if path == "/api/libraries":
            st, pl = handle_list_libraries("GET", path, query, {}, self._user)
            self._send(st, pl)
            return
        if path == "/api/folders":
            st, pl = handle_folders("GET", path, query, {}, self._user)
            self._send(st, pl)
            return
        if path == "/api/stats":
            st, pl = handle_stats("GET", path, query, {}, self._user)
            self._send(st, pl)
            return
        if path == "/api/search":
            st, pl = handle_search("GET", path, query, {}, self._user)
            self._send(st, pl)
            return
        if path == "/api/verify":
            st, pl = handle_verify("GET", path, query, {}, self._user)
            self._send(st, pl)
            return
        if path == "/api/verify/fix/stream":
            handle_verify_fix_stream(self, query, self._user)
            return
        if path == "/api/export":
            st, pl = handle_export("GET", path, query, {}, self._user)
            self._send(st, pl)
            return
        if path == "/api/files":
            st, pl = handle_list_files("GET", path, query, {}, self._user)
            self._send(st, pl)
            return
        if path == "/api/settings":
            st, pl = handle_get_settings("GET", path, query, {}, self._user)
            self._send(st, pl)
            return
        if path == "/api/chunk":
            st, pl = handle_get_chunk("GET", path, query, {}, self._user)
            self._send(st, pl)
            return
        if path == "/api/download":
            self._handle_download(query)
            return
        if path == "/api/file-chunks":
            st, pl = handle_file_chunks("GET", path, query, {}, self._user)
            self._send(st, pl)
            return
        # 查询批量删除进度（刷新页面后恢复进度用）
        if path == "/api/files/delete-status":
            st, pl = handle_delete_batch_status("GET", path, query, {})
            self._send(st, pl)
            return
        if path == "/api/chunks-around":
            st, pl = handle_chunks_around("GET", path, query, {}, self._user)
            self._send(st, pl)
            return
        if path == "/api/semantic/status":
            st, pl = handle_semantic_status("GET", path, query, {}, self._user)
            self._send(st, pl)
            return
        if path == "/api/semantic/search":
            st, pl = handle_semantic_search("GET", path, query, {}, self._user)
            self._send(st, pl)
            return
        if path == "/api/ai-search/suggest-topk":
            st, pl = handle_suggest_topk("GET", path, query, {}, self._user)
            self._send(st, pl)
            return
        if path == "/api/ai-search/stream":
            self._handle_ai_search_stream(query)
            return
        if path == "/api/ai-search/agent/stream":
            self._handle_ai_search_agent_stream(query)
            return
        if path == "/api/import/stream":
            handle_import_stream(self, query, self._user)
            return
        # 旧库补抽 chunk 标签（SSE 流式进度）
        if path == "/api/backfill-tags/stream":
            handle_backfill_tags_stream(self, query, self._user)
            return
        # 查询批次状态（刷新页面后恢复进度用）
        if path == "/api/import/batch":
            st, pl = handle_import_batch_status("GET", path, query, {})
            self._send(st, pl)
            return
        # 质询报告列表
        if path == "/api/inquiries":
            st, pl = handle_inquiries("GET", path, query, {}, self._user)
            self._send(st, pl)
            return
        # /api/inquiries/{id}  获取质询报告详情
        if path.startswith("/api/inquiries/"):
            parts = path.split("/")
            # /api/inquiries/{id}/chunks  批量取引用 chunk 原文（快速验证）
            if len(parts) == 5 and parts[4] == "chunks":
                report_id = urllib.parse.unquote(parts[3])
                st, pl = handle_inquiry_chunks("GET", path, query, {}, report_id, self._user)
                self._send(st, pl)
                return
            if len(parts) == 4:
                report_id = urllib.parse.unquote(parts[3])
                st, pl = handle_inquiry_op("GET", path, query, {}, report_id, self._user)
                self._send(st, pl)
                return
        # 会话管理（GET）：列出 / 获取详情 / 导出 / 发消息（SSE）
        if path == "/api/chat/sessions":
            st, pl = handle_chat_sessions("GET", path, query, {}, user=self._user)
            self._send(st, pl)
            return
        # /api/chat/sessions/{id}
        if path.startswith("/api/chat/sessions/"):
            parts = path.split("/")
            # parts: ['', 'api', 'chat', 'sessions', '{id}', ...]
            if len(parts) >= 5:
                sid = parts[4]
                # /api/chat/sessions/{id}/export
                if len(parts) >= 6 and parts[5] == "export":
                    st, pl = handle_chat_export("GET", path, query, {}, sid, user=self._user)
                    self._send(st, pl)
                    return
                # /api/chat/sessions/{id}/send  (SSE 流式)
                if len(parts) >= 6 and parts[5] == "send":
                    self._handle_chat_send(query, sid)
                    return
                # /api/chat/sessions/{id}/context  读取额外上下文
                if len(parts) >= 6 and parts[5] == "context":
                    st, pl = handle_chat_context("GET", path, query, {}, sid, user=self._user)
                    self._send(st, pl)
                    return
                # /api/chat/sessions/{id}/messages/{msg_id}/export  单条消息导出
                if (len(parts) >= 7 and parts[5] == "messages"
                        and len(parts) >= 8 and parts[7] == "export"):
                    msg_id = parts[6]
                    store = _chat_store()
                    # 校验会话归属后导出
                    err_resp = _check_session_owner(store, sid, self._user)
                    if err_resp:
                        self._send(err_resp[0], err_resp[1])
                        return
                    r = store.export_single_message(sid, msg_id)
                    if r is None:
                        st, pl = _err(f"消息不存在: {msg_id}", 404)
                    else:
                        st, pl = _ok(r)
                    self._send(st, pl)
                    return
                # /api/chat/sessions/{id}
                st, pl = handle_chat_sessions("GET", path, query, {}, sid, user=self._user)
                self._send(st, pl)
                return
        self._send(404, {"ok": False, "error": f"未找到路径: {path}"})

    def do_POST(self):
        path, query = self._parse()
        self._user = _current_user(self)
        # 文件上传走 multipart 解析
        if path == "/api/upload":
            try:
                st, pl = handle_upload_import(self, self._user)
            except Exception as e:
                import traceback
                traceback.print_exc()
                st, pl = _err(f"上传处理异常: {e}", 500)
            self._send(st, pl)
            return
        # 文件上传 + SSE 流式进度（upload 模式进度条）
        if path == "/api/upload/stream":
            try:
                handle_upload_stream(self, self._user)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send(500, {"ok": False, "error": f"上传流式异常: {e}"})
            return
        body = self._read_body()
        if path == "/api/auth/register":
            st, pl = handle_auth_register(self, "POST", path, query, body)
            self._send(st, pl)
            return
        if path == "/api/auth/login":
            st, pl = handle_auth_login(self, "POST", path, query, body)
            self._send(st, pl)
            return
        if path == "/api/auth/logout":
            handle_auth_logout(self, "POST", path, query, body)
            return
        # /api/auth/users/{name}/role | /password  （仅管理员）
        if path.startswith("/api/auth/users/") and path != "/api/auth/users":
            parts = path.split("/")
            if (len(parts) >= 5 and parts[1] == "api" and parts[2] == "auth"
                    and parts[3] == "users"):
                uname = urllib.parse.unquote(parts[4])
                st, pl = handle_auth_user_op("POST", path, query, body, uname, self._user)
                self._send(st, pl)
                return
        if path == "/api/libraries":
            st, pl = handle_create_library("POST", path, query, body, user=self._user)
        elif path == "/api/import":
            st, pl = handle_import("POST", path, query, body, user=self._user)
        elif path == "/api/import/cancel":
            st, pl = handle_import_cancel("POST", path, query, body)
        elif path == "/api/files/batch-delete":
            st, pl = handle_batch_delete("POST", path, query, body, user=self._user)
        elif path == "/api/files/delete-cancel":
            st, pl = handle_delete_cancel("POST", path, query, body)
        elif path == "/api/files/delete-all":
            st, pl = handle_delete_all_files("POST", path, query, body, user=self._user)
        elif path == "/api/build-index":
            st, pl = handle_build_index("POST", path, query, body, user=self._user)
        elif path == "/api/recover":
            st, pl = handle_recover("POST", path, query, body, user=self._user)
        elif path == "/api/settings":
            st, pl = handle_update_settings("POST", path, query, body, user=self._user)
        elif path == "/api/settings/test":
            st, pl = handle_test_deepseek("POST", path, query, body)
        elif path == "/api/ai-search":
            st, pl = handle_ai_search("POST", path, query, body, user=self._user)
        elif path == "/api/open-doc":
            st, pl = handle_open_doc("POST", path, query, body, user=self._user)
        elif path == "/api/repair-sources":
            st, pl = handle_repair_sources("POST", path, query, body, user=self._user)
        elif path == "/api/semantic/build":
            st, pl = handle_semantic_build("POST", path, query, body, user=self._user)
        elif path == "/api/chat/sessions":
            st, pl = handle_chat_sessions("POST", path, query, body, user=self._user)
        elif path == "/api/folders":
            st, pl = handle_folders("POST", path, query, body, self._user)
        elif path == "/api/inquiries":
            st, pl = handle_inquiries("POST", path, query, body, self._user)
        else:
            parts = path.split("/")
            # /api/chat/sessions/{id}/context  设置额外上下文
            if (len(parts) == 6 and parts[1] == "api"
                    and parts[2] == "chat" and parts[3] == "sessions"
                    and parts[5] == "context"):
                sid = parts[4]
                st, pl = handle_chat_context("POST", path, query, body, sid, user=self._user)
            # /api/libraries/{name}/move  移动库到文件夹
            elif (len(parts) == 5 and parts[1] == "api"
                    and parts[2] == "libraries" and parts[4] == "move"):
                lib_name = urllib.parse.unquote(parts[3])
                st, pl = handle_move_library("POST", path, query, body, lib_name, user=self._user)
            # /api/libraries/{name}/clone  复制公共库到我的库（数据迁移）
            elif (len(parts) == 5 and parts[1] == "api"
                    and parts[2] == "libraries" and parts[4] == "clone"):
                lib_name = urllib.parse.unquote(parts[3])
                st, pl = handle_clone_library("POST", path, query, body, lib_name, user=self._user)
            # /api/libraries/{name}/transfer  管理员转移库所有权
            elif (len(parts) == 5 and parts[1] == "api"
                    and parts[2] == "libraries" and parts[4] == "transfer"):
                lib_name = urllib.parse.unquote(parts[3])
                st, pl = handle_transfer_library("POST", path, query, body, lib_name, user=self._user)
            else:
                st, pl = 404, {"ok": False, "error": f"未找到路径: {path}"}
        self._send(st, pl)

    def do_PATCH(self):
        path, query = self._parse()
        self._user = _current_user(self)
        body = self._read_body()
        parts = path.split("/")
        # /api/libraries/{name}
        if len(parts) == 4 and parts[1] == "api" and parts[2] == "libraries":
            name = urllib.parse.unquote(parts[3])
            st, pl = handle_update_library("PATCH", path, query, body, name, user=self._user)
        # /api/folders  文件夹操作统一走 POST + action（路径可能含 /，不放 URL）
        elif path == "/api/folders":
            st, pl = handle_folders("POST", path, query, body, self._user)
        # /api/inquiries/{id}  获取质询报告详情
        elif len(parts) == 4 and parts[1] == "api" and parts[2] == "inquiries":
            report_id = urllib.parse.unquote(parts[3])
            st, pl = handle_inquiry_op("GET", path, query, body, report_id, self._user)
        # /api/chat/sessions/{id}
        elif (len(parts) == 5 and parts[1] == "api"
              and parts[2] == "chat" and parts[3] == "sessions"):
            sid = parts[4]
            st, pl = handle_chat_sessions("PATCH", path, query, body, sid, user=self._user)
        else:
            st, pl = 404, {"ok": False, "error": f"未找到路径: {path}"}
        self._send(st, pl)

    def do_DELETE(self):
        path, query = self._parse()
        self._user = _current_user(self)
        body = self._read_body()
        parts = path.split("/")
        # /api/auth/users/{name}  删除用户（仅管理员）
        if (len(parts) == 5 and parts[1] == "api"
                and parts[2] == "auth" and parts[3] == "users"):
            uname = urllib.parse.unquote(parts[4])
            st, pl = handle_auth_user_op("DELETE", path, query, body, uname, self._user)
        # /api/libraries/{name}
        elif len(parts) == 4 and parts[1] == "api" and parts[2] == "libraries":
            name = urllib.parse.unquote(parts[3])
            st, pl = handle_delete_library("DELETE", path, query, body, name, user=self._user)
        # /api/folders  删除文件夹统一走 POST + action（路径可能含 /，不放 URL）
        elif path == "/api/folders":
            st, pl = handle_folders("POST", path, query, body, self._user)
        # /api/inquiries/{id}  删除单个质询报告
        elif len(parts) == 4 and parts[1] == "api" and parts[2] == "inquiries":
            report_id = urllib.parse.unquote(parts[3])
            st, pl = handle_inquiry_op("DELETE", path, query, body, report_id, self._user)
        # /api/inquiries  清空所有质询报告
        elif path == "/api/inquiries":
            st, pl = handle_inquiries("DELETE", path, query, body, self._user)
        elif path == "/api/files":
            st, pl = handle_delete_files("DELETE", path, query, body, user=self._user)
        # /api/chat/sessions/{id}/messages/{msg_id}  回退到某条消息之前
        elif (len(parts) == 7 and parts[1] == "api"
              and parts[2] == "chat" and parts[3] == "sessions"
              and parts[5] == "messages"):
            sid = parts[4]
            msg_id = parts[6]
            store = _chat_store()
            err_resp = _check_session_owner(store, sid, self._user)
            if err_resp:
                self._send(err_resp[0], err_resp[1])
                return
            s = store.truncate_after_message(sid, msg_id)
            if s is None:
                st, pl = _err(f"消息不存在: {msg_id}", 404)
            else:
                st, pl = _ok(s)
        # /api/chat/sessions/{id}
        elif (len(parts) == 5 and parts[1] == "api"
              and parts[2] == "chat" and parts[3] == "sessions"):
            sid = parts[4]
            st, pl = handle_chat_sessions("DELETE", path, query, body, sid, user=self._user)
        # /api/chat/sessions/{id}/context  清空额外上下文
        elif (len(parts) == 6 and parts[1] == "api"
              and parts[2] == "chat" and parts[3] == "sessions"
              and parts[5] == "context"):
            sid = parts[4]
            st, pl = handle_chat_context("DELETE", path, query, body, sid, user=self._user)
        else:
            st, pl = 404, {"ok": False, "error": f"未找到路径: {path}"}
        self._send(st, pl)


def _get_lan_ip() -> str:
    """获取本机局域网 IP 地址（用于展示给用户）。"""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _ask_startup_console(default_port: int = DEFAULT_PORT):
    """启动前在控制台交互式设置数据目录、端口、访问范围。

    返回 (data_dir, port, host) 或 None（用户取消）。
    在此阶段 stdout/stderr 尚未重定向到日志，提示语直接显示在控制台。
    """
    default_data = os.path.join(SCRIPT_DIR, "data")
    print("=" * 56)
    print("  全文检索系统 - 启动设置")
    print("=" * 56)
    # 1. 数据目录（默认用代码目录下的 data/；指定其他路径可启用独立数据区）
    print(f"  默认数据目录：{default_data}")
    print("  输入其他路径可启用独立数据区（多实例隔离）；直接回车用默认。")
    while True:
        try:
            raw = input("  数据目录 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  已取消。")
            return None
        if not raw:
            data_dir = default_data
            break
        if raw.lower() in ("q", "quit", "exit"):
            print("  已取消。")
            return None
        # 相对路径相对于代码目录解析
        data_dir = raw if os.path.isabs(raw) else os.path.join(SCRIPT_DIR, raw)
        data_dir = os.path.abspath(data_dir)
        break
    # 2. 端口
    print(f"  默认端口：{default_port}（直接回车使用默认值）")
    print("  可选：输入端口号后回车；输入 q 退出。")
    while True:
        try:
            raw = input("  监听端口 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  已取消。")
            return None
        if not raw:
            port = default_port
            break
        if raw.lower() in ("q", "quit", "exit"):
            print("  已取消。")
            return None
        try:
            port = int(raw)
        except ValueError:
            print(f"  ✗ 端口必须是数字，请重新输入（直接回车=默认 {default_port}）。")
            continue
        if not (1 <= port <= 65535):
            print("  ✗ 端口范围 1-65535，请重新输入。")
            continue
        break
    # 3. 访问范围：默认局域网
    while True:
        try:
            choice = input("  访问范围 [1]局域网（默认） [2]仅本机 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  已取消。")
            return None
        if not choice or choice == "1":
            host = "0.0.0.0"
            break
        if choice == "2":
            host = "127.0.0.1"
            break
        print("  ✗ 请输入 1 或 2（直接回车=局域网）。")
    is_default_data = (data_dir == default_data)
    print(f"  ✓ 数据目录={'(默认)' if is_default_data else data_dir}")
    print(f"  ✓ 端口={port}  范围={'局域网' if host == '0.0.0.0' else '仅本机'}")
    print("=" * 56)
    return (data_dir, port, host)


class _SilentHTTPServer(ThreadingHTTPServer):
    """静默 HTTP 服务器：异常写入日志文件，不输出到控制台。"""
    def handle_error(self, request, client_address):
        import traceback
        _logger.warning("请求处理异常 %s: %s",
                        client_address, traceback.format_exc())


def main():
    # 注册控制台事件处理器，确保关闭窗口（X 按钮）或 Ctrl+Break 能终止进程
    _setup_console_ctrl_handler()

    parser = argparse.ArgumentParser(description="全文检索系统 Web API")
    parser.add_argument("--host", default=None,
                        help="监听地址（不指定则由控制台交互选择；默认 0.0.0.0 局域网可访问，改为 127.0.0.1 则仅本机可访问）")
    parser.add_argument("--port", type=int, default=None,
                        help="监听端口（不指定则在控制台交互输入，默认 20000；占用则自动向后寻找可用端口）")
    parser.add_argument("--data-dir", default=None,
                        help="数据目录路径（默认用代码目录下的 data/；指定其他路径可启用独立数据区，实现多实例隔离）")
    parser.add_argument("--no-dialog", action="store_true",
                        help="跳过控制台交互，直接用默认数据目录和端口 20000 启动（用于后台服务/脚本；占用则自动向后寻找可用端口）")
    args = parser.parse_args()

    global BASE_DIR, LOG_DIR

    # 确定数据目录、端口、host：命令行显式指定 > 控制台交互 > 默认值
    # 此阶段 stdout/stderr 尚未重定向，print/input 直接走真实控制台
    if args.data_dir and args.port is not None and args.host is not None:
        # 全部命令行指定，跳过交互
        data_dir = os.path.abspath(args.data_dir)
        port, host = args.port, args.host
    elif args.no_dialog:
        data_dir = os.path.abspath(args.data_dir) if args.data_dir else BASE_DIR
        port = args.port if args.port is not None else DEFAULT_PORT
        host = args.host or "0.0.0.0"
    else:
        choice = _ask_startup_console(DEFAULT_PORT)
        if choice is None:
            print("  程序退出。")
            return
        data_dir, port, host = choice
        # 命令行显式参数优先于控制台输入
        if args.data_dir:
            data_dir = os.path.abspath(args.data_dir)
        if args.port is not None:
            port = args.port
        if args.host is not None:
            host = args.host

    # 应用数据目录：若与默认不同，切换 BASE_DIR 和 LOG_DIR，实现多实例隔离
    if data_dir != BASE_DIR:
        BASE_DIR = data_dir
        os.makedirs(BASE_DIR, exist_ok=True)
        # 日志目录跟着数据目录走，多实例日志也隔离
        LOG_DIR = os.path.join(BASE_DIR, "output", "log")
        _reconfigure_log_handler()

    # 若目标端口被占用，自动向后寻找可用端口
    if _is_port_in_use(port):
        new_port = _find_available_port(port)
        print(f"[提示] 端口 {port} 已被占用，自动切换到 {new_port}")
        port = new_port

    # 启动信息。先在真实控制台提示"正在启动"，避免后续加载阶段被误判为卡死。
    print("  正在启动服务（加载索引/恢复任务，约数秒）...")
    print(f"  日志文件: {os.path.abspath(LOG_DIR)}{os.sep}server.log")
    print("-" * 56)
    # 此后重定向 stdout/stderr 到日志文件，控制台不再输出运行期日志
    _redirect_stdio_to_log()

    _logger.info("[启动] 数据目录=%s 端口=%d 地址=%s", BASE_DIR, port, host)

    # 启动前清理上次未完成的导入批次（服务器意外关闭时残留的脏数据）
    _logger.info("[启动] 开始清理残留导入批次...")
    _load_and_cleanup_stale_import_batches()
    _logger.info("[启动] 残留导入批次清理完成")

    # 启动时检测并续建未完成的向量索引构建（断点续建）
    # 服务器异常重启后，已完成的向量化进度会保留在磁盘上，这里自动恢复
    _logger.info("[启动] 初始化语义管理器...")
    try:
        from semantic_manager import get_manager
        mgr_sem = get_manager(BASE_DIR)
        if mgr_sem.available():
            resumed = mgr_sem.resume_pending_builds(_registry())
            if resumed:
                _logger.info("[启动] 已自动恢复 %d 个库的向量索引构建任务", resumed)
        else:
            reason = mgr_sem.fail_reason() or "未知原因"
            _logger.warning("[启动] 语义检索通道不可用：%s", reason)
    except Exception as e:
        _logger.warning("[启动] 断点续建检测失败（不影响启动）: %s", e)
    _logger.info("[启动] 语义管理器初始化完成")

    _logger.info("[启动] 正在启动 HTTP 服务...")
    server = _SilentHTTPServer((host, port), Handler)
    _logger.info("[启动] HTTP 服务已创建，即将 serve_forever")
    lan_ip = _get_lan_ip()
    # 启动信息写入日志文件
    _logger.info("=" * 60)
    _logger.info("全文检索系统已启动")
    _logger.info("=" * 60)
    if host == "0.0.0.0":
        _logger.info("本机访问:   http://127.0.0.1:%d", port)
        _logger.info("局域网访问: http://%s:%d", lan_ip, port)
        _logger.info("（同一局域网内的其他设备可用上方地址访问）")
    else:
        _logger.info("访问地址: http://%s:%d", host, port)
    _logger.info("数据目录: %s", BASE_DIR)
    _logger.info("日志目录: %s", LOG_DIR)
    _logger.info("=" * 60)
    # 服务真正就绪：把访问地址回显到真实控制台，告知用户可访问
    # 此后控制台完全静默，所有后续输出仅写入日志
    _ready_msg = []
    if host == "0.0.0.0":
        _ready_msg.append(f"  本机访问:   http://127.0.0.1:{port}")
        _ready_msg.append(f"  局域网访问: http://{lan_ip}:{port}")
    else:
        _ready_msg.append(f"  访问地址: http://{host}:{port}")
    _ready_msg.append(f"  （首次启动需要初始化语义模型，约需 5-10 分钟，请耐心等待）")
    _ready_msg.append(f"  （Ctrl+C 停止服务；运行日志见 output/log/server.log）")
    _ready_msg.append("=" * 56)
    try:
        _orig_stdout.write("=" * 56 + "\n")
        _orig_stdout.write("  ✓ 服务已启动\n")
        for _line in _ready_msg:
            _orig_stdout.write(_line + "\n")
        _orig_stdout.flush()
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _logger.info("已停止（Ctrl+C）")
        server.server_close()


if __name__ == "__main__":
    main()
