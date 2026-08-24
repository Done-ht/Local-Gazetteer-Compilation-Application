# -*- coding: utf-8 -*-
"""FastAPI 鉴权依赖：两个模式共用。

- get_current_user：必须登录（401），被禁用账号拒绝（403）
- require_admin：必须管理员（403）
- 前端携带方式：请求头 X-Access-Token: <token> 或 Authorization: Bearer <token>
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, Request

from . import users


def _extract_token(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    t = request.headers.get("X-Access-Token", "")
    if t:
        return t.strip()
    return ""


def get_current_user(request: Request) -> dict:
    """依赖注入：当前登录用户 dict；未登录 401，被禁用 403。"""
    token = _extract_token(request)
    user = users.authenticate(token) if token else None
    if not user:
        raise HTTPException(401, "未登录或登录已过期，请先登录")
    if not user.get("is_active", True):
        raise HTTPException(403, "账号已被禁用，请联系管理员")
    return user


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """依赖注入：必须是管理员。"""
    if not user.get("is_admin"):
        raise HTTPException(403, "仅管理员可执行此操作")
    return user
