# -*- coding: utf-8 -*-
"""讯飞模式 FastAPI 路由（从 ocr-web 的 http.server Handler 移植）。

鉴权统一走 app.auth.deps（与 paddle 模式共用账号系统）。
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from ..auth import users
from ..auth.deps import get_current_user, require_admin
from ..utils.config import load_config, save_config
from . import service
from .ocr_xfyun import OCRError, network_check

logger = logging.getLogger("xfyun")

router = APIRouter()

ALLOWED_EXT_EXTS = ".pdf .docx .jpg .jpeg .png .bmp .webp .tiff .tif".split()
SETTINGS_ALLOWED = ("app_id", "api_key", "api_secret", "api_type", "timeout", "retry",
                    "max_image_bytes", "page_concurrency", "concurrent_limit", "max_upload_bytes")


def _static_dir() -> str:
    """定位讯飞前端静态目录（支持打包环境）。"""
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
        candidates = [
            os.path.join(base, "_internal", "app", "xfyun", "static"),
            os.path.join(base, "_internal", "xfyun", "static"),
            os.path.join(base, "app", "xfyun", "static"),
            os.path.join(base, "xfyun", "static"),
        ]
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [os.path.join(base, "xfyun", "static")]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def _read_static(name: str) -> Optional[bytes]:
    p = os.path.join(_static_dir(), name)
    if not os.path.isfile(p):
        return None
    with open(p, "rb") as f:
        return f.read()


# ----------------------------------------------------------------------
# 页面
# ----------------------------------------------------------------------

@router.get("/")
async def index() -> Response:
    data = _read_static("index.html")
    if data is None:
        raise HTTPException(500, "前端页面缺失")
    return Response(content=data, media_type="text/html; charset=utf-8")


# ----------------------------------------------------------------------
# 公开接口
# ----------------------------------------------------------------------

@router.get("/api/network-check")
async def network_check_api() -> JSONResponse:
    """探测到讯飞服务器的网络连通性（不调用文字识别、不计费）。"""
    return JSONResponse({"ok": True, "check": network_check(load_config())})


# ----------------------------------------------------------------------
# 任务接口（需登录）
# ----------------------------------------------------------------------

@router.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
) -> JSONResponse:
    """上传文件并创建识别任务。"""
    filename = file.filename or "unnamed"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in service.ALLOWED_EXTS:
        raise HTTPException(400, f"不支持的文件类型 {ext}，支持：pdf / docx / jpg / jpeg / png / bmp / webp / tiff")
    cfg = load_config()
    max_upload = int(cfg.get("xf", {}).get("max_upload_bytes", 300 * 1024 * 1024))
    data = await file.read()
    if not data:
        raise HTTPException(400, "文件内容为空")
    if len(data) > max_upload:
        raise HTTPException(400, f"文件过大或为空（上限 {max_upload // (1024 * 1024)}MB）")
    try:
        created = service.create_task(user["user_id"], filename, data, cfg)
    except OCRError as e:
        raise HTTPException(400, str(e))
    return JSONResponse({"ok": True, **created})


class RetryBody(BaseModel):
    task_id: str = ""
    full: bool = False


@router.post("/api/retry")
async def retry_api(body: RetryBody, user: dict = Depends(get_current_user)) -> JSONResponse:
    try:
        result = service.retry_task(user["user_id"], body.task_id, full=bool(body.full), cfg=load_config())
    except OCRError as e:
        raise HTTPException(400, str(e))
    return JSONResponse({"ok": True, **result})


class CancelBody(BaseModel):
    task_id: str = ""


@router.post("/api/cancel")
async def cancel_api(body: CancelBody, user: dict = Depends(get_current_user)) -> JSONResponse:
    try:
        status = service.cancel_task(user["user_id"], body.task_id)
    except OCRError as e:
        raise HTTPException(400, str(e))
    return JSONResponse({"ok": True, "task_id": body.task_id, "status": status})


class DeleteBody(BaseModel):
    task_id: str = ""


@router.post("/api/delete")
async def delete_api(body: DeleteBody, user: dict = Depends(get_current_user)) -> JSONResponse:
    try:
        service.delete_task(user["user_id"], body.task_id)
    except OCRError as e:
        raise HTTPException(400, str(e))
    return JSONResponse({"ok": True, "task_id": body.task_id})


@router.get("/api/tasks")
async def list_tasks(user: dict = Depends(get_current_user)) -> JSONResponse:
    return JSONResponse({"ok": True, "tasks": service._load_tasks(user["user_id"])})


@router.get("/api/task/{task_id}")
async def get_task(task_id: str, user: dict = Depends(get_current_user)) -> JSONResponse:
    task = service._get_task(user["user_id"], task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    return JSONResponse({"ok": True, "task": task})


@router.get("/api/download/{task_id}")
async def download(task_id: str, format: str = "pdf",
                   user: dict = Depends(get_current_user)) -> Response:
    task = service._get_task(user["user_id"], task_id)
    if not task or task.get("status") != "done":
        raise HTTPException(404, "任务不存在或未完成")
    out_dir = os.path.join(users.user_dir(user["user_id"]), "outputs", task_id)
    name_map = {
        "pdf": ("识别结果.pdf", "application/pdf"),
        "docx": ("识别结果.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        "txt": ("识别结果.txt", "text/plain; charset=utf-8"),
    }
    if format not in name_map:
        raise HTTPException(400, "format 仅支持 pdf / docx / txt")
    fname, ctype = name_map[format]
    fpath = os.path.join(out_dir, fname)
    if not os.path.isfile(fpath):
        raise HTTPException(404, "输出文件不存在")
    with open(fpath, "rb") as f:
        data = f.read()
    base = os.path.splitext(task.get("filename", "result"))[0]
    dl_name = f"{base}-识别结果{os.path.splitext(fname)[1]}"
    ascii_name = dl_name.encode("ascii", "ignore").decode("ascii") or "result"
    return Response(
        content=data,
        media_type=ctype,
        headers={
            "Content-Disposition": f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(dl_name)}",
        },
    )


# ----------------------------------------------------------------------
# 讯飞配置（仅管理员）
# ----------------------------------------------------------------------

@router.get("/api/settings")
async def get_settings(_: dict = Depends(require_admin)) -> JSONResponse:
    cfg = load_config()
    return JSONResponse({"ok": True, "xf": cfg.get("xf", {})})


class SettingsBody(BaseModel):
    xf: dict = {}


@router.post("/api/settings")
async def post_settings(body: SettingsBody, _: dict = Depends(require_admin)) -> JSONResponse:
    xf_in = body.xf or {}
    new_xf = {
        k: v for k, v in xf_in.items()
        if k in SETTINGS_ALLOWED and isinstance(v, (str, int, float)) and str(v).strip() != ""
    }
    if new_xf:
        cfg = load_config()
        cfg.setdefault("xf", {}).update(new_xf)
        save_config(cfg)
        logger.info("管理员更新了讯飞配置（字段: %s）", ", ".join(sorted(new_xf)))
    cfg = load_config()
    return JSONResponse({"ok": True, "xf": cfg.get("xf", {})})
