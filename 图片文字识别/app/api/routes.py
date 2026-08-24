"""FastAPI 路由：文件上传、任务查询、结果下载、配置、状态、二维码。

路由列表:
  GET  /                  主页面 index.html
  GET  /api/status        服务状态（并发/排队/引擎）
  POST /api/upload        上传文件，创建识别任务
  GET  /api/tasks         列出全部任务
  GET  /api/tasks/{id}    查询任务状态与进度
  GET  /api/download/{id} 下载结果文件
  GET  /api/download_zip  批量打包下载（zip）
  GET  /api/config        获取当前配置
  POST /api/config        更新配置
  GET  /api/qr            局域网访问地址二维码
"""
from __future__ import annotations

import io
import logging
import os
import sys
import time
from typing import List, Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request, Depends
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    FileResponse,
    Response,
    PlainTextResponse,
)
from pydantic import BaseModel

from ..auth.deps import get_current_user, require_admin
from ..utils import config as config_mod
from ..utils import task_dirs
from ..utils.config import load_config, save_config
from .concurrency import ConcurrencyManager
from .tasks import TaskManager

logger = logging.getLogger(__name__)

# 支持的文件扩展名（小写，含点）
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".pdf", ".docx"}

# 结果文件 media type 映射
MEDIA_MAP = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".json": "application/json",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


# ----------------------------------------------------------------------
# 静态资源目录定位
# ----------------------------------------------------------------------
def _web_dir() -> Optional[str]:
    """返回 Web 前端资源目录，找不到返回 None。"""
    if getattr(sys, "frozen", False):
        # PyInstaller onedir: _internal/app/web 或 _internal/web
        base = os.path.dirname(sys.executable)
        candidates = [
            os.path.join(base, "_internal", "app", "web"),
            os.path.join(base, "_internal", "web"),
            os.path.join(base, "app", "web"),
            os.path.join(base, "web"),
        ]
    else:
        # 开发环境: app/api/routes.py -> 上层 app/web
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [os.path.join(base, "web")]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def _read_web_file(name: str) -> Optional[bytes]:
    """读取 Web 资源文件内容。"""
    d = _web_dir()
    if d is None:
        return None
    p = os.path.join(d, name)
    if not os.path.isfile(p):
        return None
    with open(p, "rb") as f:
        return f.read()


# ----------------------------------------------------------------------
# 配置更新请求模型
# ----------------------------------------------------------------------
class ConfigUpdate(BaseModel):
    """配置更新请求（所有字段可选）。"""
    output_format: Optional[str] = None
    output_dir: Optional[str] = None
    render_dpi: Optional[int] = None
    max_concurrent: Optional[int] = None
    enable_layer2: Optional[bool] = None
    enable_layout: Optional[bool] = None
    # OCR 模型档位：tiny(最快/精度低) / small(默认) / medium(高精度/慢)
    # 仅 PP-OCRv6 有效。改后需重启服务（或等子进程按 batch_size 自然轮换）生效。
    ocr_model_tier: Optional[str] = None
    # 表格结构识别开关：true=启用SLANet（表格HTML结构，慢38-52s/页）
    # false=跳过（表格按普通文字OCR，快5-15s/页）。改后需重启生效。
    use_table_recognition: Optional[bool] = None


def _generate_output(
    tm: "TaskManager",
    task_id: str,
    result,
    fmt: str,
    is_pdf: bool,
) -> tuple:
    """按指定格式从已保存的 OCR 数据重新生成结果文件（识别后导出）。

    返回 (输出文件绝对路径, 输出文件名)。

    - PDF 任务：文本格式从 ocr_pages 重建 DocumentResult 生成；
      可搜索 PDF 走 merge_ocr_pages 合并路径（需原图，无法从 JSON 重建）。
    - 图像任务：复用内存中保存的完整 OCR 结果（TaskResult.document）。
    """
    from ..utils import output as out_mod

    work_dir = tm._work_dirs.get(task_id) or task_dirs.task_dir(task_id)
    dpi = 200
    if tm._pipeline is not None:
        try:
            dpi = int(tm._pipeline.cfg.get("render_dpi", 200))
        except Exception:
            pass

    if is_pdf:
        from ..core import task_processor
        if fmt == "searchable_pdf":
            out_path, _cnt = task_processor.merge_ocr_pages(task_id, result.source_name)
        else:
            doc = task_processor.build_document_result_from_ocr_pages(
                task_id, result.source_name
            )
            out_path = out_mod.save_output(doc, fmt, work_dir, render_dpi=dpi)
    else:
        doc = getattr(result, "document", None)
        if doc is None:
            raise HTTPException(
                409,
                "图像任务结果数据缺失，无法重新生成该格式，请下载已保存的结果文件",
            )
        out_path = out_mod.save_output(doc, fmt, work_dir, render_dpi=dpi)
    return out_path, os.path.basename(out_path)


# ----------------------------------------------------------------------
# 创建 FastAPI 应用
# ----------------------------------------------------------------------
def create_app(
    concurrency: ConcurrencyManager,
    task_manager: TaskManager,
    lan_url: str = "",
) -> FastAPI:
    """构建 FastAPI 应用并注册全部路由。

    参数:
        concurrency: 并发控制器
        task_manager: 任务管理器
        lan_url: 局域网访问地址（如 http://192.168.1.10:8000），供二维码接口使用
    """
    app = FastAPI(title="server-paddle OCR", version="1.0.0")
    app.state.concurrency = concurrency
    app.state.task_manager = task_manager
    app.state.lan_url = lan_url

    # 加载配置到应用作用域，供 upload 接口读取上传限制等参数
    cfg = load_config()

    def _guard_task(task_id: str, user: dict) -> None:
        """任务归属校验：无权限按 404 处理（不泄露存在性）。"""
        tm: TaskManager = app.state.task_manager
        if not tm.check_owner(task_id, user):
            raise HTTPException(404, f"任务不存在: {task_id}")

    # 在 FastAPI startup 事件中启动预约任务调度器
    # （此时事件循环已运行，asyncio.create_task 才能成功）
    @app.on_event("startup")
    async def _start_scheduler():
        # 先恢复历史任务状态，再启动调度器
        # 这样 scheduled 任务恢复后能被调度器正确激活
        restored = task_manager.restore_state()
        if restored > 0:
            logger.info("已恢复 %d 个历史任务", restored)
        task_manager.start_scheduler()

    @app.on_event("shutdown")
    async def _stop_scheduler():
        # 在事件循环中取消所有活跃的 run_task 协程
        # OCR 线程无法中断，但取消协程后 uvicorn 可以顺利退出
        task_manager.cancel_active_runners()
        task_manager.stop_scheduler()

    # ------------------------------------------------------------------
    # 主页面
    # ------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        data = _read_web_file("index.html")
        if data is None:
            return HTMLResponse(
                "<h1>server-paddle OCR</h1><p>未找到 Web 资源 index.html</p>",
                status_code=404,
            )
        return HTMLResponse(data.decode("utf-8"))

    @app.get("/style.css")
    async def style_css() -> Response:
        data = _read_web_file("style.css")
        if data is None:
            raise HTTPException(404, "style.css not found")
        return Response(content=data, media_type="text/css")

    @app.get("/app.js")
    async def app_js() -> PlainTextResponse:
        data = _read_web_file("app.js")
        if data is None:
            raise HTTPException(404, "app.js not found")
        return PlainTextResponse(data.decode("utf-8"), media_type="application/javascript")

    # ------------------------------------------------------------------
    # 服务状态
    # ------------------------------------------------------------------
    @app.get("/api/status")
    async def status(user: dict = Depends(get_current_user)) -> JSONResponse:
        cstatus = concurrency.get_all_status()
        # 引擎信息
        engine_info = {
            "provider": "paddle_local",
            "name": "本地 PaddleOCR (PP-OCRv6)",
            "available": False,
            "version": "",
        }
        try:
            import paddleocr
            engine_info["version"] = getattr(paddleocr, "__version__", "")
            engine_info["available"] = True
        except Exception:
            pass
        return JSONResponse({
            "service": "server-paddle OCR",
            "lan_url": app.state.lan_url,
            "concurrency": cstatus,
            "engine": engine_info,
            "supported_extensions": sorted(SUPPORTED_EXTS),
        })

    # ------------------------------------------------------------------
    # 上传限制（供前端预检）
    # ------------------------------------------------------------------
    @app.get("/api/limits")
    async def get_limits(user: dict = Depends(get_current_user)) -> JSONResponse:
        """返回当前上传限制配置，前端提交前据此预检。"""
        upload_limit = cfg.get("upload_limit", {})
        # 同时返回当前待处理任务数，便于前端判断是否还能提交
        cstatus = concurrency.get_all_status()
        pending_count = cstatus["running"] + cstatus["queued"] + cstatus["scheduled"]
        # 可用并发槽位：总槽位 - 运行中占用的槽位
        max_conc = concurrency.max_concurrent
        # 运行中任务占用的槽位数（多slot任务占多个）
        used_slots = sum(
            len(t.slots) if t.slots else (1 if t.slot is not None else 0)
            for t in concurrency._tasks.values()
            if t.status == "running"
        )
        available_slots = max(0, max_conc - used_slots)
        return JSONResponse({
            "max_files_per_batch": int(upload_limit.get("max_files_per_batch", 20)),
            "max_file_size_mb": int(upload_limit.get("max_file_size_mb", 500)),
            "max_batch_size_mb": int(upload_limit.get("max_batch_size_mb", 2048)),
            "max_pending_tasks": int(upload_limit.get("max_pending_tasks", 50)),
            "max_scheduled_tasks": int(upload_limit.get("max_scheduled_tasks", 20)),
            "current_pending": pending_count,
            "current_scheduled": cstatus["scheduled"],
            "max_concurrent": max_conc,
            "available_slots": available_slots,
        })

    # ------------------------------------------------------------------
    # 文件上传
    # ------------------------------------------------------------------
    @app.post("/api/upload")
    async def upload(
        files: List[UploadFile] = File(...),
        output_format: Optional[str] = Form(None),
        schedule_time: Optional[str] = Form(None),
        task_concurrency: Optional[int] = Form(None),
        user: dict = Depends(get_current_user),
    ) -> JSONResponse:
        """上传文件并创建任务。

        参数:
            files: 文件列表
            output_format: 输出格式
            schedule_time: 预约执行时间，ISO 8601 格式（如 "2026-07-26T03:30"）
                           不传或空字符串表示立即执行；超过 7 天会被拒绝
        """
        if not files:
            raise HTTPException(400, "未上传任何文件")

        # 读取上传限制配置
        upload_limit = cfg.get("upload_limit", {})
        max_files_per_batch = int(upload_limit.get("max_files_per_batch", 20))
        max_file_size_mb = int(upload_limit.get("max_file_size_mb", 500))
        max_batch_size_mb = int(upload_limit.get("max_batch_size_mb", 2048))
        max_pending_tasks = int(upload_limit.get("max_pending_tasks", 50))
        max_scheduled_tasks = int(upload_limit.get("max_scheduled_tasks", 20))

        # 校验 1：单批次文件数
        if len(files) > max_files_per_batch:
            raise HTTPException(
                400,
                f"单批次最多 {max_files_per_batch} 个文件，本次提交了 {len(files)} 个，"
                f"请分多次提交",
            )

        # 校验 2：单文件大小 & 批次总大小
        max_file_size_bytes = max_file_size_mb * 1024 * 1024
        max_batch_size_bytes = max_batch_size_mb * 1024 * 1024
        batch_total_size = 0
        for f in files:
            # UploadFile.size 在部分客户端可能为 None，先读内容再校验
            if f.size is not None:
                if f.size > max_file_size_bytes:
                    raise HTTPException(
                        400,
                        f"文件 {f.filename} 大小 {f.size / 1024 / 1024:.1f} MB "
                        f"超过单文件上限 {max_file_size_mb} MB，请拆分后提交",
                    )
                batch_total_size += f.size
        if batch_total_size > max_batch_size_bytes:
            raise HTTPException(
                400,
                f"批次总大小 {batch_total_size / 1024 / 1024:.1f} MB "
                f"超过上限 {max_batch_size_mb} MB，请分批提交",
            )

        # 解析预约时间
        scheduled_at: Optional[float] = None
        if schedule_time:
            import datetime
            try:
                # 兼容 "2026-07-26T03:30" 和 "2026-07-26 03:30" 两种格式
                dt = datetime.datetime.fromisoformat(schedule_time.replace("T", " "))
                # 当地时间转时间戳
                scheduled_at = dt.timestamp()
            except ValueError:
                raise HTTPException(400, f"预约时间格式错误: {schedule_time}")

            now = time.time()
            if scheduled_at < now:
                # 过去时间视为立即执行
                scheduled_at = None
            elif scheduled_at > now + 7 * 86400:
                raise HTTPException(400, "预约时间不能超过 7 天")

        # 校验 3：待处理任务总数（queued + running + scheduled）
        cstatus = concurrency.get_all_status()
        pending_count = cstatus["running"] + cstatus["queued"] + cstatus["scheduled"]
        if pending_count + len(files) > max_pending_tasks:
            raise HTTPException(
                400,
                f"当前待处理任务 {pending_count} 个 + 本次 {len(files)} 个 "
                f"将超过上限 {max_pending_tasks}，请等待部分任务完成后再提交",
            )

        # 校验 4：预约任务数（仅预约提交时检查）
        if scheduled_at is not None:
            if cstatus["scheduled"] + len(files) > max_scheduled_tasks:
                raise HTTPException(
                    400,
                    f"当前预约任务 {cstatus['scheduled']} 个 + 本次 {len(files)} 个 "
                    f"将超过上限 {max_scheduled_tasks}，请减少预约任务或等待已预约任务执行",
                )

        tm: TaskManager = app.state.task_manager
        # 生成本次上传的批次 ID（同一次上传的所有文件共享同一 batch_id）
        import uuid
        batch_id = uuid.uuid4().hex[:12]
        results = []
        for f in files:
            filename = f.filename or "unnamed"
            ext = os.path.splitext(filename)[1].lower()
            if ext not in SUPPORTED_EXTS:
                results.append({
                    "filename": filename,
                    "error": f"不支持的文件类型: {ext}",
                    "task_id": None,
                })
                continue
            # 读取内容
            try:
                data = await f.read()
            except Exception as e:
                results.append({
                    "filename": filename,
                    "error": f"读取文件失败: {e}",
                    "task_id": None,
                })
                continue
            if not data:
                results.append({
                    "filename": filename,
                    "error": "文件为空",
                    "task_id": None,
                })
                continue
            # 读取后再次校验大小（防止 f.size 为 None 时漏检）
            if len(data) > max_file_size_bytes:
                results.append({
                    "filename": filename,
                    "error": f"文件大小 {len(data) / 1024 / 1024:.1f} MB "
                             f"超过单文件上限 {max_file_size_mb} MB",
                    "task_id": None,
                })
                continue
            # 创建任务
            task_id = tm.new_task_id()
            # 记录任务归属用户（多用户隔离：列表/详情/下载按此过滤）
            tm.set_owner(task_id, user["user_id"])
            try:
                # prepare_upload 现在返回 task_id（PDF 上传后已立即拆分到 pdf_pages/，原文件已删除）
                tm.prepare_upload(task_id, filename, data)
            except Exception as e:
                results.append({
                    "filename": filename,
                    "error": f"保存文件失败: {e}",
                    "task_id": None,
                })
                continue
            # 预约任务：先同步注册为 scheduled 状态，确保上传返回时状态正确
            # （asyncio.create_task 不会立即执行，若不预注册，返回时状态还是 None）
            if scheduled_at is not None:
                concurrency.register_scheduled(
                    task_id, scheduled_at, batch_id=batch_id, source_name=filename,
                )
            # 调度后台执行（预约任务会先等待到点）
            # 用 _spawn_runner 注册，shutdown 时可统一取消
            # file_path 传空字符串：run_task 内部会从 source_dir 重建
            # （PDF 已拆分到 pdf_pages/，run_task 通过 task_dirs.get_total_pages 判定走 PDF 流程）
            tm._spawn_runner(
                task_id, "", filename, output_format,
                scheduled_at=scheduled_at, batch_id=batch_id,
                task_concurrency=task_concurrency or 1,
            )
            # 立即返回任务信息（含排队位置与原因）
            info = tm.get_task_info(task_id)
            results.append({
                "filename": filename,
                "task_id": task_id,
                "batch_id": batch_id,
                "status": info.status if info else "queued",
                "queue_position": concurrency.get_queue_position(task_id),
                "queue_reason": concurrency.get_queue_reason(task_id),
            })
        return JSONResponse({"tasks": results, "batch_id": batch_id})

    # ------------------------------------------------------------------
    # 任务列表
    # ------------------------------------------------------------------
    @app.get("/api/tasks")
    async def list_tasks(user: dict = Depends(get_current_user)) -> JSONResponse:
        tm: TaskManager = app.state.task_manager
        return JSONResponse({
            "tasks": tm.list_tasks(
                owner=user["user_id"],
                is_admin=bool(user.get("is_admin")),
            ),
            "concurrency": concurrency.get_all_status(),
        })

    # ------------------------------------------------------------------
    # 单个任务状态
    # ------------------------------------------------------------------
    @app.get("/api/tasks/{task_id}")
    async def get_task(task_id: str, user: dict = Depends(get_current_user)) -> JSONResponse:
        tm: TaskManager = app.state.task_manager
        _guard_task(task_id, user)
        info = tm.get_task_info(task_id)
        if info is None:
            raise HTTPException(404, f"任务不存在: {task_id}")
        result = tm.get_result(task_id)
        d = info.to_dict()
        d["queue_position"] = concurrency.get_queue_position(task_id)
        d["queue_reason"] = concurrency.get_queue_reason(task_id)
        # 返回任务级并发数，供前端展示
        meta = tm._task_meta.get(task_id, {})
        d["task_concurrency"] = int(meta.get("task_concurrency", 1))
        # 页级进度信息（基于 task_dirs 的 pdf_pages/ocr_pages 统计）
        # PDF 流程：total_pages=拆分后的总页数，completed_pages=已完成 OCR 的页数
        # 非 PDF 流程：total_pages=0，page_progress=0
        total_pages = task_dirs.get_total_pages(task_id)
        completed_pages = task_dirs.get_completed_pages(task_id)
        d["total_pages"] = total_pages
        d["completed_pages"] = completed_pages
        d["page_progress"] = (
            completed_pages / total_pages if total_pages > 0 else 0
        )
        if result is not None:
            d["source_name"] = result.source_name
            d["output_name"] = result.output_name
            d["output_format"] = result.output_format
            d["pages"] = result.pages
            # 完成时附带识别文字（截断防止响应过大）
            text = result.text or ""
            d["text"] = text if len(text) <= 20000 else (text[:20000] + "\n...[已截断]")
            d["text_truncated"] = len(result.text) > 20000
        return JSONResponse(d)

    # ------------------------------------------------------------------
    # 任务进度详情（页级进度）
    # ------------------------------------------------------------------
    @app.get("/api/tasks/{task_id}/progress")
    async def get_task_progress(task_id: str, user: dict = Depends(get_current_user)) -> JSONResponse:
        """获取任务的页级进度详情。

        返回:
            {task_id, status, total_pages, completed_pages,
             pending_pages, pending_page_nos, progress}
        """
        tm: TaskManager = app.state.task_manager
        _guard_task(task_id, user)
        info = tm.get_task_info(task_id)
        if info is None:
            raise HTTPException(404, f"任务不存在: {task_id}")
        total_pages = task_dirs.get_total_pages(task_id)
        completed_pages = task_dirs.get_completed_pages(task_id)
        pending_page_nos = task_dirs.get_pending_pages(task_id)
        progress = completed_pages / total_pages if total_pages > 0 else 0
        return JSONResponse({
            "task_id": task_id,
            "status": info.status,
            "total_pages": total_pages,
            "completed_pages": completed_pages,
            "pending_pages": len(pending_page_nos),
            "pending_page_nos": pending_page_nos,
            "progress": progress,
        })

    # ------------------------------------------------------------------
    # 重试中断/失败的任务
    # ------------------------------------------------------------------
    @app.post("/api/tasks/{task_id}/retry")
    async def retry_task(task_id: str, user: dict = Depends(get_current_user)) -> JSONResponse:
        """重试中断或失败的任务。

        前提：源文件仍在磁盘上（工作目录未被清理）。
        重试时重置状态为 queued，重新排队执行。
        """
        tm: TaskManager = app.state.task_manager
        _guard_task(task_id, user)
        info = tm.get_task_info(task_id)
        if info is None:
            raise HTTPException(404, f"任务不存在: {task_id}")
        if info.status != "error":
            raise HTTPException(400, f"任务当前状态为 {info.status}，只有失败/中断状态才能重试")
        ok = tm.retry_task(task_id)
        if not ok:
            raise HTTPException(400, "重试失败：源文件可能已被清理")
        info = tm.get_task_info(task_id)
        return JSONResponse({
            "task_id": task_id,
            "status": info.status if info else "queued",
            "message": "已重新提交，排队中",
            "queue_position": concurrency.get_queue_position(task_id),
            "queue_reason": concurrency.get_queue_reason(task_id),
        })

    @app.post("/api/tasks/{task_id}/pause")
    async def pause_task(task_id: str, user: dict = Depends(get_current_user)) -> JSONResponse:
        """暂停任务（支持 running/queued/scheduled）。

        running 状态会切断 OCR 推理并释放槽位，pages_dir 保留供恢复时断点续传。
        """
        tm: TaskManager = app.state.task_manager
        _guard_task(task_id, user)
        info = tm.get_task_info(task_id)
        if info is None:
            raise HTTPException(404, f"任务不存在: {task_id}")
        if info.status in ("done", "error", "paused"):
            raise HTTPException(400, f"任务状态 {info.status} 不可暂停")
        ok, reason = tm.pause_task(task_id)
        if not ok:
            raise HTTPException(400, reason)
        info = tm.get_task_info(task_id)
        return JSONResponse({
            "task_id": task_id,
            "status": info.status if info else "paused",
            "message": "已暂停",
        })

    @app.post("/api/tasks/{task_id}/resume")
    async def resume_task(task_id: str, user: dict = Depends(get_current_user)) -> JSONResponse:
        """恢复暂停的任务。

        pre_pause=running/queued → 重新排队（pipeline 自动断点续传）
        pre_pause=scheduled → 恢复预约等待
        """
        tm: TaskManager = app.state.task_manager
        _guard_task(task_id, user)
        info = tm.get_task_info(task_id)
        if info is None:
            raise HTTPException(404, f"任务不存在: {task_id}")
        if info.status != "paused":
            raise HTTPException(400, f"任务状态 {info.status} 不可恢复（仅 paused 可恢复）")
        ok, reason = tm.resume_task(task_id)
        if not ok:
            raise HTTPException(400, reason)
        info = tm.get_task_info(task_id)
        return JSONResponse({
            "task_id": task_id,
            "status": info.status if info else "queued",
            "message": "已恢复",
            "queue_position": concurrency.get_queue_position(task_id),
        })

    @app.delete("/api/tasks/{task_id}")
    async def delete_task(task_id: str, user: dict = Depends(get_current_user)) -> JSONResponse:
        """删除任务（含工作目录、结果文件、内存状态）。

        - running 状态：强制删除（切断 OCR 推理 + 取消协程 + 释放槽位）
        - scheduled 状态：唤醒等待协程让它退出
        - queued 状态：直接从队列移除
        - done/error 状态：清理工作目录和结果文件
        """
        tm: TaskManager = app.state.task_manager
        _guard_task(task_id, user)
        ok, msg = await tm.delete_task(task_id)
        if not ok:
            # 区分 404 和 400
            info = tm.get_task_info(task_id)
            if info is None:
                raise HTTPException(404, msg)
            raise HTTPException(400, msg)
        return JSONResponse({"task_id": task_id, "message": msg})

    # ------------------------------------------------------------------
    # 下载结果文件（识别后可指定 format 重新生成任意导出格式）
    # ------------------------------------------------------------------
    @app.get("/api/download/{task_id}")
    async def download(
        task_id: str,
        format: Optional[str] = None,
        user: dict = Depends(get_current_user),
    ) -> Response:
        """下载任务结果。

        识别后导出：通过可选参数 format 指定输出类型
        （searchable_pdf / docx / txt / markdown / json / original），
        从已保存的 OCR 数据重新生成，无需在识别前预选格式。
        缺省或与任务原格式一致时直接返回已保存的结果文件（快速路径）。
        """
        tm: TaskManager = app.state.task_manager
        _guard_task(task_id, user)
        result = tm.get_result(task_id)
        if result is None:
            raise HTTPException(404, "结果不存在或任务未完成")

        # 归一化格式
        fmt = (format or result.output_format or "original").strip().lower()
        is_pdf = task_dirs.get_total_pages(task_id) > 0
        if fmt == "original":
            if is_pdf or result.source_name.lower().endswith(".pdf"):
                fmt = "searchable_pdf"
            elif result.source_name.lower().endswith(".docx"):
                fmt = "docx"
            else:
                fmt = "txt"

        # 快速路径：请求格式与已保存结果一致，直接返回原文件
        if fmt == result.output_format and result.output_path and os.path.isfile(result.output_path):
            ext = os.path.splitext(result.output_name)[1].lower()
            media = MEDIA_MAP.get(ext, "application/octet-stream")
            return FileResponse(
                path=result.output_path,
                filename=result.output_name,
                media_type=media,
            )

        # 识别后按需重新生成
        out_path, out_name = _generate_output(tm, task_id, result, fmt, is_pdf)
        ext = os.path.splitext(out_name)[1].lower()
        media = MEDIA_MAP.get(ext, "application/octet-stream")
        return FileResponse(
            path=out_path,
            filename=out_name,
            media_type=media,
        )

    # ------------------------------------------------------------------
    # 批量打包下载（zip）
    # ------------------------------------------------------------------
    @app.get("/api/download_zip")
    async def download_zip(task_ids: str, user: dict = Depends(get_current_user)) -> Response:
        """把多个已完成任务的结果文件打包成 zip 下载。

        参数:
            task_ids: 逗号分隔的 task_id 列表（如 ?task_ids=id1,id2,id3）
                      或 "all" 表示打包当前用户所有已完成任务

        返回:
            application/zip 流
        """
        import zipfile
        tm: TaskManager = app.state.task_manager

        # 解析 task_ids（仅包含当前用户有权限的任务）
        if task_ids.strip().lower() == "all":
            # 当前用户所有已完成（done）且有权限的任务
            ids = [
                tid for tid, info in tm.concurrency._tasks.items()
                if info.status == "done" and tm.get_result(tid) is not None
                and tm.check_owner(tid, user)
            ]
        else:
            ids = [s.strip() for s in task_ids.split(",") if s.strip()]

        if not ids:
            raise HTTPException(400, "未提供有效的 task_id")

        # 收集结果文件，处理重名
        # 重名时加序号前缀（如 1_xxx.pdf, 2_xxx.pdf）
        entries = []  # [(output_path, archive_name)]
        used_names = set()
        missing = []
        for tid in ids:
            if not tm.check_owner(tid, user):
                continue  # 无权限的任务跳过（不报错，避免泄露存在性）
            res = tm.get_result(tid)
            if res is None or not res.output_path or not os.path.isfile(res.output_path):
                missing.append(tid)
                continue
            name = res.output_name or os.path.basename(res.output_path)
            # 处理重名
            final_name = name
            if final_name in used_names:
                base, ext = os.path.splitext(name)
                i = 2
                while f"{base}_{i}{ext}" in used_names:
                    i += 1
                final_name = f"{base}_{i}{ext}"
            used_names.add(final_name)
            entries.append((res.output_path, final_name))

        if not entries:
            raise HTTPException(404, "没有可下载的结果文件（任务可能未完成）")

        # 打包成 zip（内存中，避免临时文件清理问题）
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for path, name in entries:
                zf.write(path, name)
        buf.seek(0)

        # 生成 zip 文件名：OCR结果_时间戳.zip
        # Content-Disposition 头用 RFC 5987 格式编码中文文件名（latin-1 不支持中文）
        from datetime import datetime
        from urllib.parse import quote
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_name = f"OCR结果_{ts}.zip"
        zip_name_ascii = f"OCR_{ts}.zip"  # ASCII 回退名

        return Response(
            content=buf.getvalue(),
            media_type="application/zip",
            headers={
                "Content-Disposition": (
                    f"attachment; filename=\"{zip_name_ascii}\"; "
                    f"filename*=UTF-8''{quote(zip_name)}"
                ),
            },
        )

    # ------------------------------------------------------------------
    # 半成品导出（.ocr_draft ZIP，可用于跨机器续作）
    # ------------------------------------------------------------------
    @app.get("/api/tasks/{task_id}/export_draft")
    async def export_draft(task_id: str, include_source: bool = True,
                           user: dict = Depends(get_current_user)) -> Response:
        """导出任务半成品为 .ocr_draft ZIP 文件。

        参数:
            include_source: 是否打包源文件（true=跨机器续作，false=仅本机续作）
                            默认 true

        返回:
            application/zip 流（.ocr_draft 扩展名）

        适用场景：
          - 任务运行中导出当前进度，备份或迁移到其他机器
          - 任务中断/失败后导出已处理部分
          - 跨机器断点续传（include_source=true）
        """
        tm: TaskManager = app.state.task_manager
        _guard_task(task_id, user)
        info = tm.get_task_info(task_id)
        if info is None:
            raise HTTPException(404, f"任务不存在: {task_id}")
        try:
            zip_path, meta_data = tm.export_draft(task_id, include_source=include_source)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except Exception as e:
            logger.exception("导出半成品失败: %s", task_id)
            raise HTTPException(500, f"导出半成品失败: {e}")

        if not os.path.isfile(zip_path):
            raise HTTPException(500, "半成品文件生成失败")

        # 流式读取文件（避免大文件占用内存）
        from urllib.parse import quote
        zip_name = os.path.basename(zip_path)
        zip_name_ascii = f"draft_{task_id}.ocr_draft"
        with open(zip_path, "rb") as f:
            data = f.read()
        return Response(
            content=data,
            media_type="application/zip",
            headers={
                "Content-Disposition": (
                    f"attachment; filename=\"{zip_name_ascii}\"; "
                    f"filename*=UTF-8''{quote(zip_name)}"
                ),
                "X-Completed-Pages": str(meta_data.get("completed_pages", 0)),
                "X-Total-Pages": str(meta_data.get("total_pages", 0)),
            },
        )

    # ------------------------------------------------------------------
    # 提前导出 PDF（运行中任务把已处理页合并为最终 PDF）
    # ------------------------------------------------------------------
    @app.post("/api/tasks/{task_id}/finalize_partial")
    async def finalize_partial(task_id: str, user: dict = Depends(get_current_user)) -> JSONResponse:
        """把任务当前已处理页合并为最终 PDF，不影响任务继续运行。

        返回:
            {task_id, output_path, output_name, page_count, download_url}

        适用场景：
          - 任务运行中想提前看到部分结果
          - 任务因异常中断但已处理部分有用，需要导出 PDF

        注意：任务完成后（done）不应调用此接口，应直接下载最终结果。
        pages_dir 在任务完成时已被清理，旧 partial PDF 可能残留过期内容。
        """
        tm: TaskManager = app.state.task_manager
        _guard_task(task_id, user)
        info = tm.get_task_info(task_id)
        if info is None:
            raise HTTPException(404, f"任务不存在: {task_id}")
        # 任务已完成时拒绝 partial 导出：pages_dir 已被清理，
        # 旧 partial PDF 是过期内容（可能只有几页），应下载最终结果
        if info.status == "done":
            res = tm._results.get(task_id)
            if res and res.output_path:
                raise HTTPException(
                    400,
                    f"任务已完成，请直接下载最终结果（共 {res.pages} 页），"
                    f"无需导出部分内容",
                )
            raise HTTPException(400, "任务已完成，请直接下载最终结果")
        try:
            out_path, page_count = tm.finalize_partial(task_id)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except Exception as e:
            logger.exception("提前导出 PDF 失败: %s", task_id)
            raise HTTPException(500, f"提前导出失败: {e}")

        if not os.path.isfile(out_path):
            raise HTTPException(500, "PDF 文件生成失败")

        # 把 partial 结果注册到 _partial_results，供下载接口使用
        # 使用 task_id + 时间戳作为 partial_id，避免冲突
        partial_id = f"{task_id}_partial_{int(time.time())}"
        tm._partial_results[partial_id] = {
            "task_id": task_id,
            "output_path": out_path,
            "output_name": os.path.basename(out_path),
            "page_count": page_count,
            "created_at": time.time(),
        }
        # 持久化（重启后 partial 链接仍可用，直到工作目录被清理）
        tm._save_state(force=True)

        from urllib.parse import quote
        out_name = os.path.basename(out_path)
        # 同时返回基于 task_dirs 的页级进度统计，便于前端展示进度
        total_pages = task_dirs.get_total_pages(task_id)
        completed_pages = task_dirs.get_completed_pages(task_id)
        return JSONResponse({
            "task_id": task_id,
            "partial_id": partial_id,
            "output_path": out_path,
            "output_name": out_name,
            "page_count": page_count,
            "completed_pages": completed_pages,
            "total_pages": total_pages,
            "download_url": f"/api/download_partial/{partial_id}",
        })

    # ------------------------------------------------------------------
    # 下载提前导出的 PDF
    # ------------------------------------------------------------------
    @app.get("/api/download_partial/{partial_id}")
    async def download_partial(partial_id: str, user: dict = Depends(get_current_user)) -> FileResponse:
        """下载提前导出的 PDF 文件。"""
        tm: TaskManager = app.state.task_manager
        info = tm._partial_results.get(partial_id)
        if info is None:
            raise HTTPException(404, f"提前导出文件不存在: {partial_id}")
        _guard_task(info.get("task_id", ""), user)
        if not info.get("output_path") or not os.path.isfile(info["output_path"]):
            raise HTTPException(404, "文件已被清理（任务工作目录可能已删除）")
        return FileResponse(
            path=info["output_path"],
            filename=info.get("output_name") or os.path.basename(info["output_path"]),
            media_type="application/pdf",
        )

    # ------------------------------------------------------------------
    # 导入半成品（上传 .ocr_draft 续作）
    # ------------------------------------------------------------------
    @app.post("/api/upload_draft")
    async def upload_draft(
        file: UploadFile = File(...),
        output_format: Optional[str] = Form(None),
        task_concurrency: Optional[int] = Form(None),
        user: dict = Depends(get_current_user),
    ) -> JSONResponse:
        """上传半成品 ZIP 文件，解压并启动断点续传任务。

        参数:
            file: .ocr_draft 文件（含源文件和已处理单页）
            output_format: 输出格式（可选）

        返回:
            {task_id, source_name, completed_pages, total_pages, status}
        """
        tm: TaskManager = app.state.task_manager

        if not file.filename or not file.filename.lower().endswith(".ocr_draft"):
            raise HTTPException(400, "请上传 .ocr_draft 格式的半成品文件")

        # 读取上传文件到临时位置（避免大文件占用内存）
        import tempfile
        data = await file.read()
        if not data:
            raise HTTPException(400, "文件为空")

        # 校验大小（与源文件大小限制一致）
        upload_limit = cfg.get("upload_limit", {})
        max_file_size_mb = int(upload_limit.get("max_file_size_mb", 500))
        max_file_size_bytes = max_file_size_mb * 1024 * 1024
        if len(data) > max_file_size_bytes:
            raise HTTPException(
                400,
                f"半成品文件大小 {len(data) / 1024 / 1024:.1f} MB "
                f"超过上限 {max_file_size_mb} MB",
            )

        # 写入临时文件
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".ocr_draft")
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                f.write(data)
            # 调用 TaskManager 导入
            try:
                task_id, source_name, completed, total = tm.import_draft(
                    tmp_path, output_format=output_format,
                    task_concurrency=task_concurrency or 1,
                )
            except (FileNotFoundError, ValueError) as e:
                raise HTTPException(400, str(e))
            except Exception as e:
                logger.exception("导入半成品失败")
                raise HTTPException(500, f"导入失败: {e}")
        finally:
            # 临时文件已解压到工作目录，可以删除
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        info = tm.get_task_info(task_id)
        # 记录任务归属用户（导入的半成品任务同样按用户隔离）
        tm.set_owner(task_id, user["user_id"])
        return JSONResponse({
            "task_id": task_id,
            "source_name": source_name,
            "completed_pages": completed,
            "total_pages": total,
            "status": info.status if info else "queued",
            "message": f"已导入半成品，从第 {completed + 1} 页继续",
        })

    # ------------------------------------------------------------------
    # 配置查询 / 更新
    # ------------------------------------------------------------------
    @app.get("/api/config")
    async def get_config(user: dict = Depends(get_current_user)) -> JSONResponse:
        cfg = load_config()
        # 剔除讯飞密钥等敏感字段（讯飞凭证仅管理员在讯飞模式设置页可见）
        cfg.pop("xf", None)
        # 不返回敏感字段（服务端无凭证，但保持结构）
        return JSONResponse(cfg)

    @app.post("/api/config")
    async def update_config(body: ConfigUpdate, user: dict = Depends(require_admin)) -> JSONResponse:
        cfg = load_config()
        # 跟踪需要重启才生效的字段变更
        needs_restart = False
        restart_fields = []
        if body.output_format is not None:
            if body.output_format in ("original", "searchable_pdf", "markdown", "json", "txt", "docx"):
                cfg["output_format"] = body.output_format
        if body.output_dir is not None:
            cfg["output_dir"] = body.output_dir
        if body.render_dpi is not None:
            if 50 <= body.render_dpi <= 600:
                old_dpi = cfg.get("render_dpi", 200)
                if old_dpi != body.render_dpi:
                    needs_restart = True
                    restart_fields.append("渲染DPI")
                cfg["render_dpi"] = body.render_dpi
        if body.max_concurrent is not None:
            if 1 <= body.max_concurrent <= 16:
                old_max = cfg.get("max_concurrent", 3)
                if old_max != body.max_concurrent:
                    needs_restart = True
                    restart_fields.append("最大并发数")
                cfg["max_concurrent"] = body.max_concurrent
                # 同步更新并发管理器（仅字段，实际容量重启生效）
                concurrency.set_max_concurrent(body.max_concurrent)
        if body.enable_layer2 is not None:
            old_l2 = cfg.get("filter", {}).get("enable_layer2")
            if old_l2 != body.enable_layer2:
                needs_restart = True
                restart_fields.append("L1边过滤")
            cfg.setdefault("filter", {})["enable_layer2"] = body.enable_layer2
        if body.enable_layout is not None:
            old_layout = cfg.get("paddle", {}).get("enable_layout")
            if old_layout != body.enable_layout:
                needs_restart = True
                restart_fields.append("版面分析")
            cfg.setdefault("paddle", {})["enable_layout"] = body.enable_layout
        if body.ocr_model_tier is not None:
            tier = body.ocr_model_tier.lower()
            if tier in ("tiny", "small", "medium"):
                old_tier = cfg.get("paddle", {}).get("ocr_model_tier", "small")
                if old_tier != tier:
                    needs_restart = True
                    restart_fields.append(f"OCR模型档位({old_tier}→{tier})")
                cfg.setdefault("paddle", {})["ocr_model_tier"] = tier
        if body.use_table_recognition is not None:
            old_tbl = cfg.get("paddle", {}).get("use_table_recognition", True)
            if old_tbl != body.use_table_recognition:
                needs_restart = True
                restart_fields.append(
                    f"表格识别({'开' if body.use_table_recognition else '关'})"
                )
            cfg.setdefault("paddle", {})["use_table_recognition"] = body.use_table_recognition
        save_config(cfg)
        return JSONResponse({
            "ok": True, "config": cfg,
            "needs_restart": needs_restart,
            "restart_fields": restart_fields,
        })

    @app.post("/api/shutdown")
    async def shutdown_service(user: dict = Depends(require_admin)) -> JSONResponse:
        """关闭服务（配置更改后需重启时调用）。

        通过设置事件循环的关闭标志，让 uvicorn 在下一次循环时退出。
        前端调用后会显示"服务正在关闭"提示，用户需手动重新启动 exe。
        """
        import asyncio
        import os
        # 延迟 1 秒关闭服务，让响应先返回给客户端
        loop = asyncio.get_event_loop()
        def _shutdown():
            # Windows 下用 os._exit 直接退出进程
            # 不用 sys.exit 或 KeyboardInterrupt：在异步回调中抛异常会被 uvicorn 吞掉
            # os._exit 跳过 finally 清理，但配置已保存到磁盘，重启后自然恢复
            os._exit(0)
        loop.call_later(1.0, _shutdown)
        return JSONResponse({"ok": True, "message": "服务将在 1 秒后关闭"})

    # ------------------------------------------------------------------
    # 二维码（局域网访问地址）
    # ------------------------------------------------------------------
    @app.get("/api/qr")
    async def qr_code(user: dict = Depends(get_current_user)) -> Response:
        """生成局域网访问地址的二维码 PNG。

        qrcode 库未安装时返回 1x1 透明 PNG（而非 500），避免前端反复重试刷屏。
        前端检测到 Content-Length=0 或加载失败时会隐藏二维码图片。
        """
        url = app.state.lan_url or "http://127.0.0.1:8070"
        try:
            import qrcode
        except ImportError:
            # 返回 1x1 透明 PNG，避免 500 错误刷屏
            # (PNG 文件头 + IHDR + IDAT + IEND，宽高各 1 像素)
            transparent_png = (
                b"\x89PNG\r\n\x1a\n"
                b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
                b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
                b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
                b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            return Response(content=transparent_png, media_type="image/png")
        try:
            img = qrcode.make(url)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return Response(content=buf.getvalue(), media_type="image/png")
        except Exception as e:
            logger.warning("生成二维码失败: %s", e)
            raise HTTPException(500, "二维码生成失败: " + str(e))

    return app
