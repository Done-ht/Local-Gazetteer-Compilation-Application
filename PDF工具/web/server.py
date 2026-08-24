"""Flask 应用 + 全部路由。

路由总览：
  GET  /                            首页（6 标签页，需登录）
  GET  /login                       登录/注册页
  POST /api/auth/login              登录（设置 cookie）
  POST /api/auth/logout             注销
  GET  /api/auth/me                 当前登录用户
  POST /api/auth/register           注册（首个用户自动管理员）
  GET  /api/auth/users              管理员：列出用户
  POST /api/auth/role               管理员：修改角色
  POST /api/auth/change_password    改自己密码
  POST /api/auth/reset_password     管理员：重置他人密码
  POST /api/auth/remove_user        管理员：删除用户
  GET  /api/status                 是否本机 + 主机名
  POST /api/local/list_dir         本机模式：列目录
  POST /api/local/quick_paths      本机模式：常用目录快速入口
  POST /api/local/file_info        本机模式：取 PDF 页数/加密状态
  POST /api/local/open_dir         本机模式：打开输出目录
  POST /api/upload                 远程模式：上传文件
  POST /api/{merge|split|...}/start  启动任务
  GET  /api/task/<id>/progress     SSE 进度流
  POST /api/task/<id>/cancel       取消任务
  GET  /api/download/<id>          远程模式：下载结果
  POST /api/task/<id>/cleanup      远程模式：清理临时文件
"""
import os
import sys
import io
import json
import uuid
import queue
import shutil
import zipfile
import tempfile
import time
import threading
from datetime import datetime

# 注入路径：项目根目录（用于 import app.*）与 web 目录（用于同级 import）
_WEB_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_WEB_DIR)
for _p in (_PROJECT_DIR, _WEB_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from flask import (Flask, request, jsonify, send_file, render_template,
                   Response, stream_with_context, redirect, make_response, g)

import local_detector
import path_browser
import open_dir
from task_manager import task_manager as tm, make_callbacks
from auth import UserStore
from userdata import auth_base_dir

from app.merge.merge_engine import MergeEngine
from app.split.split_engine import SplitEngine
from app.convert.convert_engine import ConvertEngine
from app.compose.compose_engine import ComposeEngine
from app.append.append_engine import AppendEngine
from app.insert.insert_engine import InsertEngine
from app.merge.sorter import sort_specified_order, sort_folder_order, sort_by_folder_order
from app.utils.pdf_utils import (
    get_pdf_page_count, is_valid_pdf, is_encrypted_pdf,
    get_pdf_info, get_pdf_bookmarks,
)

app = Flask(__name__, template_folder=os.path.join(_WEB_DIR, "templates"),
            static_folder=os.path.join(_WEB_DIR, "static"))
# 远程上传上限 2GB（本机模式不走上传）
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024
# 开发环境：模板自动重载（改 index.html 无需重启服务即可生效），静态文件不缓存（改 app.js/style.css 即时生效）
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# 远程模式上传文件存放区
UPLOAD_ROOT = os.path.join(tempfile.gettempdir(), "pdf_web_uploads")
os.makedirs(UPLOAD_ROOT, exist_ok=True)
# file_id -> {"path", "name", "owner", "created_at"}
UPLOADS: dict = {}
# 上传文件保留时长：2 小时未启动任务即视为废弃，可被清理
UPLOAD_TTL_SECONDS = 2 * 3600

# --------------------------------------------------------------------------- #
# 用户认证：跨应用共用 <用户主目录>\biaoshifu 下的账号数据
# --------------------------------------------------------------------------- #
user_store = UserStore(auth_base_dir())
SESSION_COOKIE = "pdf_session_token"
SESSION_MAX_AGE = 7 * 24 * 3600  # 7 天，与 auth.SESSION_TTL_SECONDS 对齐


def _current_user():
    """从 flask.g 取当前登录用户（由 before_request 注入），未登录返回 None。"""
    return getattr(g, "current_user", None)


def _current_username() -> str:
    """当前登录用户名；未登录返回空串。"""
    u = _current_user()
    return u.get("username", "") if u else ""


def _is_admin() -> bool:
    u = _current_user()
    return bool(u and u.get("role") == "admin")


def _check_task_owner(task):
    """越权校验：仅任务所有者或管理员可访问。失败返回 (resp, code) 或 None。"""
    me = _current_username()
    if not me:
        return jsonify({"error": "未登录"}), 401
    if task.owner and task.owner != me and not _is_admin():
        return jsonify({"error": "无权访问该任务"}), 403
    return None


def _resolve_uploads(file_ids, owner_check=True):
    """把 file_id 列表解析为 (paths, error_resp)。

    - 任一 file_id 不存在 → 返回 (None, (resp, 400))
    - owner_check=True 且存在不属于当前用户的 file_id → 返回 (None, (resp, 403))
    - 全部通过 → 返回 (paths, None)
    """
    me = _current_username()
    paths = []
    for fid in file_ids or []:
        item = UPLOADS.get(fid)
        if not item:
            return None, (jsonify({"error": f"文件不存在或已被清理: {fid}"}), 400)
        if owner_check and item.get("owner") and item["owner"] != me and not _is_admin():
            return None, (jsonify({"error": "无权使用他人上传的文件"}), 403)
        paths.append(item["path"])
    return paths, None


def _purge_expired_uploads():
    """清理超过 TTL 的废弃上传文件（启动任务时顺便调用一次）。"""
    now = time.time()
    expired = [fid for fid, it in UPLOADS.items()
               if now - it.get("created_at", now) > UPLOAD_TTL_SECONDS]
    for fid in expired:
        item = UPLOADS.pop(fid, None)
        if item:
            _safe_remove_file(item["path"])


def _safe_remove_file(path):
    """删除单个文件，忽略不存在。"""
    if not path:
        return
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


# 后台定期清理上传文件（每 30 分钟一次）
def _upload_gc_loop():
    while True:
        time.sleep(1800)
        try:
            _purge_expired_uploads()
        except Exception:
            pass


threading.Thread(target=_upload_gc_loop, daemon=True, name="pdf-upload-gc").start()


def _set_session_cookie(resp, token):
    resp.set_cookie(SESSION_COOKIE, token, max_age=SESSION_MAX_AGE,
                    httponly=True, samesite="Lax", path="/")


def _clear_session_cookie(resp):
    resp.set_cookie(SESSION_COOKIE, "", expires=0, path="/")


# 白名单：无需登录即可访问的路径
# _PUBLIC_PATHS 精确匹配；_PUBLIC_PREFIXES 前缀匹配
# 注意：/api/auth/login 与 /api/auth/register 是认证入口，需放行；
#       其余 /api/auth/* 接口（me/logout/users 等）需登录，由 before_request 注入 g.current_user。
_PUBLIC_PATHS = ("/login", "/api/auth/login", "/api/auth/register")
_PUBLIC_PREFIXES = ("/static/",)


@app.before_request
def _auth_guard():
    """全局鉴权：未登录拦截到 /login 或返回 401。"""
    path = request.path
    # 静态资源与认证相关路径放行
    if path in _PUBLIC_PATHS or any(path.startswith(p) for p in _PUBLIC_PREFIXES):
        return None
    token = request.cookies.get(SESSION_COOKIE, "")
    user = user_store.get_user_by_token(token) if token else None
    if user:
        g.current_user = user
        return None
    # 未登录：API 返回 401 JSON，页面重定向到登录页
    if path.startswith("/api/"):
        return jsonify({"error": "未登录或会话已过期", "need_login": True}), 401
    return redirect("/login")


# --------------------------------------------------------------------------- #
# 认证路由（登录 / 注销 / 注册 / 用户管理）
# --------------------------------------------------------------------------- #
@app.get("/login")
def login_page():
    # 已登录直接跳首页
    token = request.cookies.get(SESSION_COOKIE, "")
    if token and user_store.get_user_by_token(token):
        return redirect("/")
    return render_template("login.html")


@app.post("/api/auth/login")
def auth_login():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    user = user_store.authenticate(username, password)
    if not user:
        return jsonify({"error": "用户名或密码错误"}), 401
    token = user_store.create_session(user["username"])
    resp = jsonify({"user": user})
    _set_session_cookie(resp, token)
    return resp


@app.post("/api/auth/logout")
def auth_logout():
    token = request.cookies.get(SESSION_COOKIE, "")
    if token:
        user_store.logout(token)
    resp = jsonify({"ok": True})
    _clear_session_cookie(resp)
    return resp


@app.get("/api/auth/me")
def auth_me():
    user = _current_user()
    if not user:
        return jsonify({"error": "未登录"}), 401
    return jsonify({"user": user})


@app.post("/api/auth/register")
def auth_register():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    try:
        user = user_store.register(username, password)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    # 注册成功后自动登录
    token = user_store.create_session(user["username"])
    resp = jsonify({"user": user})
    _set_session_cookie(resp, token)
    return resp


@app.get("/api/auth/users")
def auth_users():
    user = _current_user()
    if not user or user.get("role") != "admin":
        return jsonify({"error": "需要管理员权限"}), 403
    return jsonify({"users": user_store.list_users()})


@app.post("/api/auth/role")
def auth_role():
    me = _current_user()
    if not me or me.get("role") != "admin":
        return jsonify({"error": "需要管理员权限"}), 403
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    role = data.get("role") or ""
    try:
        updated = user_store.set_role(username, role)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if not updated:
        return jsonify({"error": "用户不存在"}), 404
    return jsonify({"user": updated})


@app.post("/api/auth/change_password")
def auth_change_password():
    me = _current_user()
    if not me:
        return jsonify({"error": "未登录"}), 401
    data = request.get_json(silent=True) or {}
    old_pwd = data.get("old_password") or ""
    new_pwd = data.get("new_password") or ""
    try:
        ok = user_store.change_password(me["username"], old_pwd, new_pwd)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if not ok:
        return jsonify({"error": "原密码错误"}), 401
    return jsonify({"ok": True})


@app.post("/api/auth/reset_password")
def auth_reset_password():
    me = _current_user()
    if not me or me.get("role") != "admin":
        return jsonify({"error": "需要管理员权限"}), 403
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    new_pwd = data.get("new_password") or ""
    try:
        ok = user_store.admin_reset_password(username, new_pwd)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if not ok:
        return jsonify({"error": "用户不存在"}), 404
    return jsonify({"ok": True})


@app.post("/api/auth/remove_user")
def auth_remove_user():
    me = _current_user()
    if not me or me.get("role") != "admin":
        return jsonify({"error": "需要管理员权限"}), 403
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    if username == me["username"]:
        return jsonify({"error": "不能删除自己"}), 400
    if user_store.remove_user(username):
        return jsonify({"ok": True})
    return jsonify({"error": "用户不存在"}), 404


# --------------------------------------------------------------------------- #
# 基础路由
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    user = _current_user()
    return render_template("index.html",
                          username=user["username"],
                          is_admin=user.get("role") == "admin")


@app.get("/api/status")
def status():
    s = local_detector.get_status()
    s["is_local"] = local_detector.is_local_request(request.remote_addr)
    return jsonify(s)


def _require_local():
    """本机专属端点鉴权，失败返回 (resp, code) 或 None。"""
    if not local_detector.is_local_request(request.remote_addr):
        return jsonify({"error": "仅本机访问可操作文件系统"}), 403
    return None


@app.post("/api/local/list_dir")
def local_list_dir():
    deny = _require_local()
    if deny:
        return deny
    data = request.get_json(silent=True) or {}
    return jsonify(path_browser.list_dir(data.get("path", "")))


@app.post("/api/local/quick_paths")
def local_quick_paths():
    deny = _require_local()
    if deny:
        return deny
    return jsonify(path_browser.quick_paths())


@app.post("/api/local/file_info")
def local_file_info():
    deny = _require_local()
    if deny:
        return deny
    data = request.get_json(silent=True) or {}
    path = data.get("path", "")
    if not path or not os.path.isfile(path):
        return jsonify({"error": "文件不存在"}), 404
    try:
        if path.lower().endswith(".pdf"):
            info = get_pdf_info(path)
            info["encrypted"] = is_encrypted_pdf(path)
            info["valid"] = is_valid_pdf(path)
            return jsonify(info)
        # 图片等非 PDF 仅返回基本信息
        return jsonify({
            "path": path,
            "filename": os.path.basename(path),
            "page_count": 0,
            "encrypted": False,
            "valid": True,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/local/open_dir")
def local_open_dir():
    deny = _require_local()
    if deny:
        return deny
    data = request.get_json(silent=True) or {}
    return jsonify({"ok": open_dir.open_directory(data.get("path", ""))})


# --------------------------------------------------------------------------- #
# 远程模式：上传 / 下载 / 清理
# --------------------------------------------------------------------------- #
@app.post("/api/upload")
def upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "未收到文件"}), 400
    me = _current_username()
    uploaded = []
    now = time.time()
    for f in files:
        if not f or not f.filename:
            continue
        # 不用 secure_filename（会去掉中文），仅取 basename 防路径穿越
        safe_name = os.path.basename(f.filename)
        file_id = uuid.uuid4().hex[:12]
        dest = os.path.join(UPLOAD_ROOT, f"{file_id}_{safe_name}")
        f.save(dest)
        UPLOADS[file_id] = {"path": dest, "name": safe_name,
                            "owner": me, "created_at": now}
        uploaded.append({"file_id": file_id, "name": safe_name,
                         "size": os.path.getsize(dest)})
    return jsonify({"files": uploaded})


@app.get("/api/download/<task_id>")
def download(task_id):
    task = tm.get(task_id)
    if not task or not task.output_path:
        return jsonify({"error": "任务或输出不存在"}), 404
    # 越权校验：仅任务所有者或管理员可下载
    deny = _check_task_owner(task)
    if deny:
        return deny
    path = task.output_path
    if task.output_is_dir:
        # 拆分 single 模式输出多文件，打包 zip 下载
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(path):
                for fn in files:
                    full = os.path.join(root, fn)
                    zf.write(full, os.path.relpath(full, path))
        buf.seek(0)
        return send_file(buf, as_attachment=True,
                         download_name=os.path.basename(path) + ".zip")
    # range/extract 模式：engine 把单一 PDF 写到 output_dir 内，
    # 此时 output_path 仍是目录路径，需进入目录取唯一 PDF
    if os.path.isdir(path):
        pdfs = [f for f in os.listdir(path) if f.lower().endswith(".pdf")]
        if len(pdfs) == 1:
            path = os.path.join(path, pdfs[0])
        elif not pdfs:
            return jsonify({"error": "输出文件不存在"}), 404
        else:
            # 多个 PDF 回退为 zip（容错）
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for fn in pdfs:
                    full = os.path.join(path, fn)
                    zf.write(full, fn)
            buf.seek(0)
            return send_file(buf, as_attachment=True,
                             download_name=os.path.basename(path) + ".zip")
    if not os.path.isfile(path):
        return jsonify({"error": "输出文件不存在"}), 404
    return send_file(path, as_attachment=True, download_name=os.path.basename(path))


@app.post("/api/task/<task_id>/cleanup")
def cleanup_task(task_id):
    task = tm.get(task_id)
    if not task:
        return jsonify({"error": "任务不存在"}), 404
    deny = _check_task_owner(task)
    if deny:
        return deny
    return jsonify({"ok": tm.cleanup_task(task_id)})


# --------------------------------------------------------------------------- #
# 进度 SSE 与取消
# --------------------------------------------------------------------------- #
@app.get("/api/task/<task_id>/progress")
def task_progress(task_id):
    task = tm.get(task_id)
    if not task:
        return jsonify({"error": "任务不存在"}), 404
    # 越权校验：仅任务所有者或管理员可订阅进度
    deny = _check_task_owner(task)
    if deny:
        return deny

    def gen():
        while True:
            try:
                msg = task.q.get_nowait()
            except queue.Empty:
                if task.status == "running":
                    try:
                        msg = task.q.get(timeout=25)
                    except queue.Empty:
                        yield ": ping\n\n"
                        continue
                else:
                    # 任务已结束但队列无积压（被消费过），补发最终状态
                    yield from _emit_final(task)
                    return
            mtype = msg.get("type")
            if mtype == "progress":
                yield _sse("progress", {"current": msg["current"],
                                        "total": msg["total"],
                                        "message": msg["message"]})
            elif mtype == "done":
                yield _sse("done", {"output_path": msg.get("output_path", ""),
                                    "output_is_dir": msg.get("output_is_dir", False)})
            elif mtype == "error":
                yield _sse("error", {"message": msg["message"]})
            elif mtype == "cancelled":
                yield _sse("cancelled", {})
            elif mtype == "end":
                yield _sse("end", {})
                return

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return Response(stream_with_context(gen()), mimetype="text/event-stream",
                    headers=headers)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _emit_final(task):
    if task.status == "done":
        yield _sse("done", {"output_path": task.output_path,
                            "output_is_dir": task.output_is_dir})
    elif task.status == "error":
        yield _sse("error", {"message": task.error})
    elif task.status == "cancelled":
        yield _sse("cancelled", {})
    yield _sse("end", {})


@app.post("/api/task/<task_id>/cancel")
def cancel_task(task_id):
    task = tm.get(task_id)
    if not task:
        return jsonify({"error": "任务不存在"}), 404
    deny = _check_task_owner(task)
    if deny:
        return deny
    ok = tm.cancel(task_id)
    return jsonify({"ok": ok})


# --------------------------------------------------------------------------- #
# 六大功能启动路由
# --------------------------------------------------------------------------- #
def _merge_files(data, mode):
    """根据模式解析合并文件列表（本机模式支持三种排序，远程仅指定顺序）。

    远程模式下同时校验 file_id 归属。返回 (files, error_resp)。
    """
    if mode == "local":
        sort_mode = data.get("sort_mode", "specified")
        if sort_mode == "specified":
            return data.get("files", []), None
        if sort_mode == "folder":
            return sort_folder_order(data.get("folder_path", "")), None
        if sort_mode == "by_folder":
            return sort_by_folder_order(data.get("root_path", "")), None
        return [], None
    # 远程模式：解析 file_id 并校验归属
    paths, err = _resolve_uploads(data.get("files", []))
    if err:
        return [], err
    return paths, None


@app.post("/api/merge/start")
def merge_start():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "local")
    files, err = _merge_files(data, mode)
    if err:
        return err
    if not files:
        return jsonify({"error": "没有选择要合并的文件"}), 400
    fast_mode = data.get("fast_mode", True)

    if mode == "local":
        output_path = data.get("output_path", "")
        if not output_path:
            return jsonify({"error": "未指定输出路径"}), 400
        input_paths = []
    else:
        # 远程模式：再次解析以便记录 input_paths 用于清理（归属已校验）
        input_paths, _ = _resolve_uploads(data.get("files", []))
        output_path = os.path.join(UPLOAD_ROOT, f"out_{uuid.uuid4().hex[:8]}_"
                                               f"_合并结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")

    def engine_call(cancel_flag, q):
        progress_cb, cancel_check = make_callbacks(cancel_flag, q)
        return MergeEngine(progress_callback=progress_cb,
                           cancel_check=cancel_check).merge(files, output_path, fast_mode=fast_mode)

    _purge_expired_uploads()
    task_id = tm.start("merge", engine_call, output_path=output_path,
                      cleanup_paths=[output_path], input_paths=input_paths,
                      owner=_current_username())
    return jsonify({"task_id": task_id})


@app.post("/api/split/start")
def split_start():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "local")
    smode = data.get("split_mode", "single")

    if mode == "local":
        input_path = data.get("input_path", "")
        output_dir = data.get("output_dir", "")
        if not input_path:
            return jsonify({"error": "未选择输入 PDF"}), 400
        if not output_dir:
            return jsonify({"error": "未指定输出目录"}), 400
        input_paths = []
    else:
        file_ids = data.get("files", [])
        if not file_ids:
            return jsonify({"error": "未选择输入 PDF"}), 400
        paths, err = _resolve_uploads(file_ids)
        if err:
            return err
        input_path = paths[0]
        output_dir = os.path.join(UPLOAD_ROOT, f"split_{uuid.uuid4().hex[:8]}")
        os.makedirs(output_dir, exist_ok=True)
        input_paths = paths

    kwargs = {}
    if smode == "range":
        kwargs["range_text"] = data.get("range_text", "")
    elif smode == "extract":
        # 前端传 1-based 页码，转为 0-based
        kwargs["extract_pages"] = [p - 1 for p in data.get("extract_pages", [])]

    def engine_call(cancel_flag, q):
        progress_cb, cancel_check = make_callbacks(cancel_flag, q)
        return SplitEngine(progress_callback=progress_cb,
                           cancel_check=cancel_check).split(input_path, output_dir, smode, **kwargs)

    # single 模式输出多文件 → 走 zip 下载；range/extract 输出单一 PDF → 走文件下载
    is_dir = (smode == "single")
    _purge_expired_uploads()
    task_id = tm.start("split", engine_call, output_path=output_dir,
                      output_is_dir=is_dir, cleanup_paths=[output_dir],
                      input_paths=input_paths, owner=_current_username())
    return jsonify({"task_id": task_id})


@app.post("/api/convert/start")
def convert_start():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "local")
    dpi = int(data.get("dpi", 150))
    if dpi < 50 or dpi > 600:
        return jsonify({"error": "DPI 范围 50-600"}), 400

    if mode == "local":
        input_path = data.get("input_path", "")
        if not input_path:
            return jsonify({"error": "未选择输入 PDF"}), 400
        output_path = data.get("output_path", "")
        if not output_path:
            return jsonify({"error": "未指定输出路径"}), 400
        input_paths = []
    else:
        file_ids = data.get("files", [])
        if not file_ids:
            return jsonify({"error": "未选择输入 PDF"}), 400
        paths, err = _resolve_uploads(file_ids)
        if err:
            return err
        input_path = paths[0]
        # 取首个文件名构造输出名（归属已校验，可直接访问）
        first_name = UPLOADS[file_ids[0]]["name"]
        out_name = f"_{os.path.splitext(first_name)[0]}.docx"
        output_path = os.path.join(UPLOAD_ROOT, f"out_{uuid.uuid4().hex[:8]}_{out_name}")
        input_paths = paths

    def engine_call(cancel_flag, q):
        progress_cb, cancel_check = make_callbacks(cancel_flag, q)
        return ConvertEngine(progress_callback=progress_cb,
                             cancel_check=cancel_check).convert(input_path, output_path, dpi=dpi)

    _purge_expired_uploads()
    task_id = tm.start("convert", engine_call, output_path=output_path,
                      cleanup_paths=[output_path], input_paths=input_paths,
                      owner=_current_username())
    return jsonify({"task_id": task_id})


@app.post("/api/compose/start")
def compose_start():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "local")
    if mode == "local":
        files = data.get("files", [])
        output_path = data.get("output_path", "")
        if not output_path:
            return jsonify({"error": "未指定输出路径"}), 400
        input_paths = []
    else:
        file_ids = data.get("files", [])
        paths, err = _resolve_uploads(file_ids)
        if err:
            return err
        files = paths
        output_path = os.path.join(UPLOAD_ROOT, f"out_{uuid.uuid4().hex[:8]}_"
                                               f"合成结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        input_paths = list(files)

    if not files:
        return jsonify({"error": "没有选择要合成的图片"}), 400

    def engine_call(cancel_flag, q):
        progress_cb, cancel_check = make_callbacks(cancel_flag, q)
        return ComposeEngine(progress_callback=progress_cb,
                             cancel_check=cancel_check).compose(files, output_path)

    _purge_expired_uploads()
    task_id = tm.start("compose", engine_call, output_path=output_path,
                      cleanup_paths=[output_path], input_paths=input_paths,
                      owner=_current_username())
    return jsonify({"task_id": task_id})


@app.post("/api/append/start")
def append_start():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "local")
    if mode == "local":
        files = data.get("files", [])
        output_path = data.get("output_path", "")
        if not output_path:
            return jsonify({"error": "未指定输出路径"}), 400
        input_paths = []
    else:
        file_ids = data.get("files", [])
        paths, err = _resolve_uploads(file_ids)
        if err:
            return err
        files = paths
        output_path = os.path.join(UPLOAD_ROOT, f"out_{uuid.uuid4().hex[:8]}_"
                                               f"拼接结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        input_paths = list(files)

    if not files:
        return jsonify({"error": "没有选择要拼接的文件"}), 400

    def engine_call(cancel_flag, q):
        progress_cb, cancel_check = make_callbacks(cancel_flag, q)
        return AppendEngine(progress_callback=progress_cb,
                            cancel_check=cancel_check).append(files, output_path)

    _purge_expired_uploads()
    task_id = tm.start("append", engine_call, output_path=output_path,
                      cleanup_paths=[output_path], input_paths=input_paths,
                      owner=_current_username())
    return jsonify({"task_id": task_id})


@app.post("/api/insert/start")
def insert_start():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "local")
    insert_page = int(data.get("insert_page", 0))

    if mode == "local":
        base_pdf = data.get("base_pdf", "")
        files = data.get("files", [])
        output_path = data.get("output_path", "")
        if not base_pdf:
            return jsonify({"error": "未选择基础 PDF"}), 400
        if not output_path:
            return jsonify({"error": "未指定输出路径"}), 400
        input_paths = []
    else:
        base_id = data.get("base_pdf")
        if not base_id:
            return jsonify({"error": "未选择基础 PDF"}), 400
        # 校验基础 PDF 归属
        base_paths, err = _resolve_uploads([base_id])
        if err:
            return err
        base_pdf = base_paths[0]
        # 校验插入内容归属
        files, err = _resolve_uploads(data.get("files", []))
        if err:
            return err
        base_name = UPLOADS[base_id]["name"]
        out_name = f"_{os.path.splitext(base_name)[0]}_插入结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        output_path = os.path.join(UPLOAD_ROOT, f"out_{uuid.uuid4().hex[:8]}_{out_name}")
        input_paths = [base_pdf] + list(files)

    if not files:
        return jsonify({"error": "没有选择要插入的内容"}), 400

    def engine_call(cancel_flag, q):
        progress_cb, cancel_check = make_callbacks(cancel_flag, q)
        return InsertEngine(progress_callback=progress_cb,
                            cancel_check=cancel_check).insert(base_pdf, files, insert_page, output_path)

    _purge_expired_uploads()
    task_id = tm.start("insert", engine_call, output_path=output_path,
                      cleanup_paths=[output_path], input_paths=input_paths,
                      owner=_current_username())
    return jsonify({"task_id": task_id})


if __name__ == "__main__":
    # 直接运行 server.py 也可启动（推荐用 run_web.py）
    app.run(host="0.0.0.0", port=8000, threaded=True)
