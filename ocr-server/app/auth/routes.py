# -*- coding: utf-8 -*-
"""共用用户路由：注册 / 登录 / 登出 / 当前用户 / 管理员用户管理。

两个模式（paddle / xfyun）都挂载本路由。
"""
from __future__ import annotations

import os
import re
import sys

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import users
from .deps import get_current_user, require_admin, _extract_token

router = APIRouter()


# ----------------------------------------------------------------------
# 共用页面：登录/注册页、管理员用户管理页（两个模式都挂载）
# ----------------------------------------------------------------------

def _web_dir() -> str:
    """定位 app/web 前端资源目录（支持打包环境）。"""
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
        candidates = [
            os.path.join(base, "_internal", "app", "web"),
            os.path.join(base, "_internal", "web"),
            os.path.join(base, "app", "web"),
            os.path.join(base, "web"),
        ]
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [os.path.join(base, "web")]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")


def _read_page(name: str) -> HTMLResponse:
    p = os.path.join(_web_dir(), name)
    if not os.path.isfile(p):
        raise HTTPException(500, f"页面 {name} 缺失")
    with open(p, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@router.get("/login", response_class=HTMLResponse)
async def login_page() -> HTMLResponse:
    """登录/注册页（共用）。"""
    return _read_page("login.html")


@router.get("/admin", response_class=HTMLResponse)
async def admin_page() -> HTMLResponse:
    """管理员用户管理页（共用）。"""
    return _read_page("admin.html")


class UsernamePassword(BaseModel):
    username: str = ""
    password: str = ""


class ResetPasswordBody(BaseModel):
    password: str = ""


@router.post("/api/register")
async def register(body: UsernamePassword) -> dict:
    """注册新用户。系统内首个注册的用户自动成为管理员。"""
    username = (body.username or "").strip()
    password = body.password or ""
    if not users.USERNAME_RE.match(username):
        raise HTTPException(400, "用户名需为 2-32 个字符，且不能包含空格")
    if len(password) < users.MIN_PASSWORD_LEN:
        raise HTTPException(400, f"密码长度至少 {users.MIN_PASSWORD_LEN} 位")
    user, err = users.register(username, password)
    if err:
        raise HTTPException(400, err)
    return {"ok": True, "user": user, "first_user": user["is_admin"]}


@router.post("/api/login")
async def login(body: UsernamePassword) -> dict:
    """登录，返回会话 token（前端存 localStorage 并在请求头携带）。"""
    username = (body.username or "").strip()
    password = body.password or ""
    token, user = users.login(username, password)
    if not token:
        raise HTTPException(401, user or "用户名或密码错误")
    return {"ok": True, "token": token, "user": user}


@router.post("/api/logout")
async def logout(request: Request) -> dict:
    """登出：使当前会话 token 失效。"""
    users.logout(_extract_token(request))
    return {"ok": True}


@router.get("/api/me")
async def me(user: dict = Depends(get_current_user)) -> dict:
    """当前登录用户信息。"""
    return {
        "ok": True,
        "user": {
            "user_id": user["user_id"],
            "username": user["username"],
            "is_admin": bool(user.get("is_admin")),
            "is_active": bool(user.get("is_active", True)),
        },
    }


# ----------------------------------------------------------------------
# 管理员：用户管理
# ----------------------------------------------------------------------

@router.get("/api/admin/users")
async def admin_list_users(_: dict = Depends(require_admin)) -> dict:
    return {"ok": True, "users": users.list_users()}


@router.post("/api/admin/users/{user_id}/disable")
async def admin_disable_user(user_id: str, admin: dict = Depends(require_admin)) -> dict:
    ok, err = users.set_active(user_id, False, admin["user_id"])
    if not ok:
        raise HTTPException(400, err or "操作失败")
    return {"ok": True}


@router.post("/api/admin/users/{user_id}/enable")
async def admin_enable_user(user_id: str, admin: dict = Depends(require_admin)) -> dict:
    ok, err = users.set_active(user_id, True, admin["user_id"])
    if not ok:
        raise HTTPException(400, err or "操作失败")
    return {"ok": True}


@router.delete("/api/admin/users/{user_id}")
async def admin_delete_user(user_id: str, admin: dict = Depends(require_admin)) -> dict:
    ok, err = users.remove_user(user_id, admin["user_id"])
    if not ok:
        raise HTTPException(400, err or "删除失败")
    return {"ok": True}


@router.post("/api/admin/users/{user_id}/reset-password")
async def admin_reset_password(user_id: str, body: ResetPasswordBody,
                               admin: dict = Depends(require_admin)) -> dict:
    new_password = body.password or ""
    if len(new_password) < users.MIN_PASSWORD_LEN:
        raise HTTPException(400, f"新密码长度至少 {users.MIN_PASSWORD_LEN} 位")
    ok, err = users.reset_password(user_id, new_password)
    if not ok:
        raise HTTPException(400, err or "重置失败")
    return {"ok": True}
