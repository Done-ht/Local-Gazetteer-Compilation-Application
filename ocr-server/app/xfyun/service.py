# -*- coding: utf-8 -*-
"""讯飞模式业务层：任务计划、断点续传、后台执行、持久化。

从 ocr-web/main_server.py 提取的纯业务逻辑（无 HTTP 层），
由 app/xfyun/routes.py（FastAPI）调用。
"""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
import shutil
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List

from ..auth import users
from .ocr_xfyun import OCRError, OcrLine, OcrPage, network_check, ocr_image, ocr_pdf_page, pdf_page_count
from .exporters import build_docx, build_pdf, build_txt

logger = logging.getLogger("xfyun")

ALLOWED_EXTS = {".pdf", ".docx", ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
_TASK_ID_RE = re.compile(r"^[a-f0-9]{16}$")

_uploads_lock = threading.Lock()

# 任务排队 / 取消（内存态，不落盘）
_cond = threading.Condition()   # 保护 _waiting 与 _cancel_flags
_waiting: list = []             # [(user_id, task_id), ...] 已排队、等待并发名额的任务
_cancel_flags: set = set()      # 已请求取消的 task_id（进行中的任务由执行线程检测后收尾）


class _TaskCanceled(Exception):
    """任务被用户取消。"""


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _task_id() -> str:
    return secrets.token_hex(8)


def _safe_filename(name: str) -> str:
    """去掉路径分隔符等危险字符，仅保留文件名。"""
    base = os.path.basename(name.replace("\\", "/"))
    base = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", base).strip()
    return base or "file"


# ----------------------------------------------------------------------
# 任务持久化（每用户一个 tasks.json）
# ----------------------------------------------------------------------

def _tasks_path(user_id: str) -> str:
    return os.path.join(users.user_dir(user_id), "tasks.json")


def _load_tasks(user_id: str) -> list:
    p = _tasks_path(user_id)
    if not os.path.isfile(p):
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_tasks(user_id: str, tasks: list) -> None:
    p = _tasks_path(user_id)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


def _get_task(user_id: str, task_id: str) -> dict:
    if not _TASK_ID_RE.match(task_id or ""):
        return None
    for t in _load_tasks(user_id):
        if t.get("task_id") == task_id:
            return t
    return None


def delete_task(user_id: str, task_id: str) -> None:
    """删除单条识别记录：移除 tasks.json 中的元数据，并清理磁盘上的
    上传源文件与输出（含逐页结果）。运行中的任务不允许删除。
    """
    if not _TASK_ID_RE.match(task_id or ""):
        raise OCRError("任务不存在")
    with _uploads_lock:
        tasks = _load_tasks(user_id)
        task = next((t for t in tasks if t.get("task_id") == task_id), None)
        if not task:
            raise OCRError("任务不存在")
        if task.get("status") in ("pending", "queued", "processing"):
            raise OCRError("任务正在运行，请先取消后再删除")
        tasks = [t for t in tasks if t.get("task_id") != task_id]
        _save_tasks(user_id, tasks)

    # 清理该任务在内存队列中的残留（防御性，正常已不在队列中）
    with _cond:
        _waiting[:] = [
            (u, t) for (u, t) in _waiting
            if not (u == user_id and t == task_id)
        ]
        _cancel_flags.discard(task_id)

    # 清理磁盘数据：上传源文件 + 输出（含逐页结果）
    udir = users.user_dir(user_id)
    for sub in ("uploads", "outputs"):
        p = os.path.join(udir, sub, task_id)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)


def _update_task(user_id: str, task_id: str, **fields) -> None:
    with _uploads_lock:
        tasks = _load_tasks(user_id)
        for t in tasks:
            if t.get("task_id") == task_id:
                t.update(fields)
                break
        _save_tasks(user_id, tasks)


# ----------------------------------------------------------------------
# 识别任务
# ----------------------------------------------------------------------

def _docx_plan_parts(path: str) -> tuple:
    """docx：提取文本 + 内嵌图片字节列表，返回 (text, [(image_bytes, ext), ...])。

    不在此处 OCR：图片按顺序进入处理计划，便于按页断点续传。
    """
    from docx import Document

    try:
        doc = Document(path)
        para_text = "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        raise OCRError(f"docx 解析失败: {e}")

    images: list = []
    try:
        from docx.enum.shape import WD_INLINE_SHAPE

        for shape in doc.inline_shapes:
            if shape.type != WD_INLINE_SHAPE.PICTURE:
                continue
            try:
                blob = shape._inline.graphic.graphicData.pic.blipFill.blip.embed
                image_part = doc.part.related_parts[blob]
                content_type = getattr(image_part, "content_type", "") or ""
                ext = "png"
                if "jpeg" in content_type:
                    ext = "jpg"
                elif "bmp" in content_type:
                    ext = "bmp"
                images.append((image_part.blob, ext))
            except Exception:
                continue
    except Exception:
        pass

    if not para_text.strip() and not images:
        raise OCRError("docx 中没有可识别的内容（无文字、无图片）")
    return para_text.strip(), images


def _build_plan(user_id: str, task_id: str, src_path: str, ext: str) -> list:
    """构建处理计划：每一项是一个待识别"页"（含类型与参数），按顺序执行即可断点续传。"""
    plan: list = []
    if ext in IMAGE_EXTS:
        plan.append({"kind": "image_file", "path": src_path})
    elif ext == ".pdf":
        total = pdf_page_count(src_path)
        plan = [{"kind": "pdf_page", "path": src_path, "page_index": i} for i in range(total)]
    elif ext == ".docx":
        text, images = _docx_plan_parts(src_path)
        if text:
            plan.append({"kind": "docx_text", "text": text})
        media_dir = os.path.join(os.path.dirname(src_path), "media")
        os.makedirs(media_dir, exist_ok=True)
        for i, (blob, img_ext) in enumerate(images):
            img_path = os.path.join(media_dir, f"img_{i}.{img_ext}")
            with open(img_path, "wb") as f:
                f.write(blob)
            plan.append({"kind": "image_file", "path": img_path})
    else:
        raise OCRError(f"不支持的文件类型: {ext}")
    if not plan:
        raise OCRError("没有可识别的页")
    return plan


def _process_plan_item(item: dict, cfg: dict, page_num: int) -> OcrPage:
    """执行计划中的一项，返回该页的 OcrPage。"""
    kind = item.get("kind")
    if kind == "image_file":
        with open(item["path"], "rb") as f:
            data = f.read()
        return ocr_image(data, cfg, page_num=page_num)
    if kind == "pdf_page":
        return ocr_pdf_page(item["path"], item["page_index"], cfg)
    if kind == "docx_text":
        return OcrPage(page_num=page_num, text=item.get("text", ""))
    raise OCRError(f"未知计划项: {kind}")


def _pages_dir(user_id: str, task_id: str) -> str:
    return os.path.join(users.user_dir(user_id), "outputs", task_id, "pages")


def _save_page_file(user_id: str, task_id: str, idx: int, page: OcrPage) -> None:
    """把一页结果立即落盘（断点续传的基础：已识别的页永不丢失）。"""
    d = _pages_dir(user_id, task_id)
    os.makedirs(d, exist_ok=True)
    if page.image_bytes:
        with open(os.path.join(d, f"page_{idx:04d}.jpg"), "wb") as f:
            f.write(page.image_bytes)
    meta = {
        "page_num": page.page_num,
        "text": page.text,
        "lines": [
            {"text": ln.text, "bbox": list(ln.bbox), "confidence": ln.confidence}
            for ln in page.lines
        ],
        "width": page.width,
        "height": page.height,
        "image": f"page_{idx:04d}.jpg" if page.image_bytes else None,
    }
    with open(os.path.join(d, f"page_{idx:04d}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)


def _count_pages(user_id: str, task_id: str) -> int:
    """统计已成功落盘的页数（以 json 完成标记为准）。"""
    d = _pages_dir(user_id, task_id)
    if not os.path.isdir(d):
        return 0
    return len([f for f in os.listdir(d) if f.endswith(".json")])


def _load_pages(user_id: str, task_id: str, total: int) -> List[OcrPage]:
    """从落盘文件组装全部页（生成最终输出用）。"""
    d = _pages_dir(user_id, task_id)
    pages: List[OcrPage] = []
    for idx in range(total):
        with open(os.path.join(d, f"page_{idx:04d}.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        image_bytes = None
        if meta.get("image"):
            with open(os.path.join(d, meta["image"]), "rb") as f:
                image_bytes = f.read()
        lines = [
            OcrLine(text=ln["text"], bbox=tuple(ln["bbox"]), confidence=ln.get("confidence", 0.0))
            for ln in meta.get("lines", [])
        ]
        pages.append(
            OcrPage(
                page_num=meta.get("page_num", idx + 1),
                text=meta.get("text", ""),
                lines=lines,
                image_bytes=image_bytes,
                width=meta.get("width", 0),
                height=meta.get("height", 0),
            )
        )
    return pages


def run_task(user_id: str, task_id: str, src_path: str, ext: str, cfg: dict, resume_from: int = 0) -> None:
    """后台执行识别任务，页级并发处理。

    断点续传语义：resume_from > 0 时保留已落盘页，只处理缺失页（按 page json 标记判断）；
    resume_from <= 0 时清空旧页全量重跑。
    并发满时任务保持 queued 排队，不直接失败；排队中或进行中均可被用户取消。
    """
    acquired = False
    try:
        # ---- 排队等待并发名额（并发满时进入等待队列，不直接失败）----
        # 每用户并发上限取配置 xf.concurrent_limit（管理员可在设置页调整）
        user_limit = int(cfg.get("xf", {}).get("concurrent_limit", users.MAX_CONCURRENT_TASKS))
        with _cond:
            while True:
                if task_id in _cancel_flags:
                    _update_task(user_id, task_id, status="canceled", error="任务已取消", resumable=False)
                    _cancel_flags.discard(task_id)
                    return
                if users.acquire_slot(user_id, user_limit):
                    acquired = True
                    break
                task = _get_task(user_id, task_id)
                if task and task.get("status") != "queued":
                    _update_task(user_id, task_id, status="queued")
                if (user_id, task_id) not in _waiting:
                    _waiting.append((user_id, task_id))
                if task_id in _cancel_flags:
                    continue  # 回到循环顶部，走取消分支
                _cond.wait()

        # 网络预检：断网快速失败，避免空耗等待超时（不产生 OCR 计费）
        probe = network_check(cfg)
        if not probe["ok"]:
            raise OCRError(f"无法连接讯飞 OCR 服务器（{probe['host']}），已暂停识别。请检查网络后点「继续识别」。")

        plan = _build_plan(user_id, task_id, src_path, ext)
        total = len(plan)
        pages_dir = _pages_dir(user_id, task_id)

        if resume_from <= 0:
            if os.path.isdir(pages_dir):
                shutil.rmtree(pages_dir, ignore_errors=True)
        os.makedirs(pages_dir, exist_ok=True)

        # 找出需要处理的页：已落盘 json 的页跳过（断点续传）
        to_do = [
            i for i in range(total)
            if not os.path.isfile(os.path.join(pages_dir, f"page_{i:04d}.json"))
        ]
        done = _count_pages(user_id, task_id)
        _update_task(
            user_id, task_id,
            status="processing", total=total, progress=done,
            completed=done, resumable=False, error="",
        )

        def process_one(idx: int) -> None:
            page = _process_plan_item(plan[idx], cfg, page_num=idx + 1)
            _save_page_file(user_id, task_id, idx, page)

        failed: dict = {}
        max_workers = max(1, int(cfg.get("xf", {}).get("page_concurrency", 5)))
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="xf-ocr") as ex:
            futures = {ex.submit(process_one, i): i for i in to_do}
            for fut in as_completed(futures):
                if task_id in _cancel_flags:
                    raise _TaskCanceled()
                idx = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    logger.warning("任务 %s 第 %d 页失败: %s", task_id, idx + 1, e)
                    failed[idx] = str(e)
                done = _count_pages(user_id, task_id)
                _update_task(user_id, task_id, status="processing", progress=done, completed=done)

        if failed:
            desc = "、".join(f"第{i + 1}页" for i in sorted(failed)[:10])
            more = f" 等 {len(failed)} 页" if len(failed) > 10 else ""
            first_err = next(iter(failed.values()))
            raise OCRError(f"{desc}{more}识别失败（已识别 {done}/{total} 页）。原因：{first_err[:120]}。可点「继续识别」重试失败页。")

        # 组装输出（PDF 导出失败不影响 docx/txt，note 里提示原因）
        out_dir = os.path.join(users.user_dir(user_id), "outputs", task_id)
        os.makedirs(out_dir, exist_ok=True)
        filename = _safe_filename(os.path.basename(src_path))
        note = ""

        pages = _load_pages(user_id, task_id, total)
        docx_bytes = build_docx(pages, filename)
        txt_str = build_txt(pages)
        docx_path = os.path.join(out_dir, "识别结果.docx")
        txt_path = os.path.join(out_dir, "识别结果.txt")
        with open(docx_path, "wb") as f:
            f.write(docx_bytes)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt_str)

        try:
            pdf_bytes = build_pdf(pages, filename)
            pdf_path = os.path.join(out_dir, "识别结果.pdf")
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)
        except Exception as e:
            logger.warning("任务 %s PDF 导出失败: %s", task_id, e)
            note = f"PDF 导出不可用：{e}"

        _update_task(
            user_id, task_id,
            status="done",
            progress=total,
            total=total,
            page_count=total,
            completed=total,
            resumable=False,
            filename=filename,
            note=note,
        )
    except _TaskCanceled:
        completed = _count_pages(user_id, task_id) if "pages_dir" in dir() else 0
        logger.info("任务 %s 已被用户取消（已识别 %d 页）", task_id, completed)
        _update_task(
            user_id, task_id,
            status="canceled", error="任务已取消",
            resumable=False,
        )
    except OCRError as e:
        completed = _count_pages(user_id, task_id) if "pages_dir" in dir() else 0
        logger.warning("任务 %s 失败（已识别 %d 页）: %s", task_id, completed, e)
        _update_task(
            user_id, task_id,
            status="error", error=str(e),
            resumable=completed > 0,
        )
    except Exception:
        completed = _count_pages(user_id, task_id) if "pages_dir" in dir() else 0
        logger.error("任务 %s 异常（已识别 %d 页）:\n%s", task_id, completed, traceback.format_exc())
        _update_task(
            user_id, task_id,
            status="error", error="识别过程中发生未知错误，详见服务端日志",
            resumable=completed > 0,
        )
    finally:
        if acquired:
            users.release_slot(user_id)
        _cancel_flags.discard(task_id)
        with _cond:
            _cond.notify_all()  # 唤醒排队中的任务重新竞争名额


def create_task(user_id: str, filename: str, file_bytes: bytes, cfg: dict) -> dict:
    """保存上传文件并创建任务，返回 {task_id, filename}。"""
    filename = _safe_filename(filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        raise OCRError(f"不支持的文件类型 {ext}，支持：pdf / docx / jpg / jpeg / png / bmp / webp / tiff")

    task_id = _task_id()
    udir = users.user_dir(user_id)
    src_dir = os.path.join(udir, "uploads", task_id)
    os.makedirs(src_dir, exist_ok=True)
    src_path = os.path.join(src_dir, filename)
    with open(src_path, "wb") as f:
        f.write(file_bytes)

    task = {
        "task_id": task_id,
        "filename": filename,
        "status": "pending",
        "progress": 0,
        "total": 0,
        "page_count": 0,
        "completed": 0,
        "resumable": False,
        "error": "",
        "note": "",
        "created_at": _now(),
    }
    with _uploads_lock:
        tasks = _load_tasks(user_id)
        tasks.insert(0, task)
        _save_tasks(user_id, tasks)

    threading.Thread(
        target=run_task,
        args=(user_id, task_id, src_path, ext, cfg),
        daemon=True,
    ).start()
    return {"task_id": task_id, "filename": filename}


def retry_task(user_id: str, task_id: str, full: bool = False, cfg: dict = None) -> dict:
    """重试失败任务（默认从断点续跑；full=True 全量重跑）。"""
    task = _get_task(user_id, task_id)
    if not task:
        raise OCRError("任务不存在")
    if task.get("status") not in ("error",):
        raise OCRError("只有失败的任务可以重试")
    filename = task.get("filename", "")
    src_path = os.path.join(users.user_dir(user_id), "uploads", task_id, filename)
    if not os.path.isfile(src_path):
        raise OCRError("原始文件不存在，无法重试")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        raise OCRError(f"不支持的文件类型 {ext}")
    resume_from = 0 if full else task.get("completed", 0)
    _update_task(
        user_id, task_id,
        status="pending", error="", note="",
        progress=resume_from, resumable=False,
    )
    threading.Thread(
        target=run_task,
        args=(user_id, task_id, src_path, ext, cfg or _get_cfg(), resume_from),
        daemon=True,
    ).start()
    return {"task_id": task_id, "resume_from": resume_from}


def cancel_task(user_id: str, task_id: str) -> str:
    """取消排队中或进行中的任务，返回最终状态。"""
    task = _get_task(user_id, task_id)
    if not task:
        raise OCRError("任务不存在")
    st = task.get("status")
    if st not in ("pending", "queued", "processing"):
        raise OCRError("只有排队中或进行中的任务可以取消")
    with _cond:
        _cancel_flags.add(task_id)
        _waiting[:] = [
            (uid, tid)
            for (uid, tid) in _waiting
            if not (uid == user_id and tid == task_id)
        ]
    if st in ("pending", "queued"):
        # 尚未占用名额：直接落定终态（执行线程醒来会确认 flag，不再改写）
        _update_task(user_id, task_id, status="canceled", error="任务已取消", resumable=False)
    with _cond:
        _cond.notify_all()  # 唤醒可能阻塞在等待队列中的执行线程，使其感知取消
    return "canceled"


def resume_stale_tasks(cfg: dict) -> None:
    """服务重启后恢复遗留任务（排队中/等待中/中断的进行中任务）。

    页级结果已落盘，处理中的任务按已完成的页数断点续跑；排队任务重新排队。
    """
    resumed = 0
    for u in users.list_users():
        user_id = u["user_id"]
        for t in _load_tasks(user_id):
            if t.get("status") not in ("pending", "queued", "processing"):
                continue
            task_id = t.get("task_id", "")
            filename = t.get("filename", "")
            src_path = os.path.join(users.user_dir(user_id), "uploads", task_id, filename)
            if not os.path.isfile(src_path):
                continue
            ext = os.path.splitext(filename)[1].lower()
            if ext not in ALLOWED_EXTS:
                continue
            resume_from = max(0, int(t.get("completed", 0) or 0))
            _update_task(user_id, task_id, status="queued", error="", note="")
            threading.Thread(
                target=run_task,
                args=(user_id, task_id, src_path, ext, cfg, resume_from),
                daemon=True,
            ).start()
            resumed += 1
    if resumed:
        logger.info("已恢复 %d 个遗留任务", resumed)


def _get_cfg() -> dict:
    from ..utils.config import load_config
    return load_config()
