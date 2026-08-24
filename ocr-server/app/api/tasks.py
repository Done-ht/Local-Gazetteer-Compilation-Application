"""任务管理：创建任务、运行识别、存储结果、清理。

每个上传文件对应一个任务：
  1. prepare_upload 保存上传文件到任务工作目录
  2. run_task 在并发控制下执行 pipeline.process_file
  3. 结果文件保留供下载，超时/超额自动清理

任务执行使用 run_in_executor，避免阻塞事件循环；
PaddleOCR 非线程安全，并发上限由 ConcurrencyManager 控制。
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .concurrency import ConcurrencyManager, TaskInfo
from ..core import task_processor
from ..utils import output as out_mod
from ..utils import task_dirs
from ..utils.progress_log import log_progress

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """任务结果元信息（供下载与展示）。"""
    task_id: str
    source_name: str  # 原始文件名
    output_path: str  # 结果文件绝对路径（供下载）
    output_name: str  # 下载文件名
    output_format: str  # 输出格式
    text: str = ""  # 识别文字（供前端预览）
    pages: int = 0  # 页数
    created_at: float = field(default_factory=time.time)
    # 完整的 OCR 结果（DocumentResult）：图像任务保留，用于识别后按需重新生成任意导出格式。
    # PDF 任务为 None（其结果从 ocr_pages 目录按需重建）。不参与 JSON 序列化。
    document: Optional[Any] = None


class TaskManager:
    """任务管理器。"""

    def __init__(
        self,
        concurrency: ConcurrencyManager,
        pipeline_factory: Callable[[], Any],
        output_dir: Optional[str] = None,
        keep_recent: int = 30,
    ) -> None:
        """
        参数:
            concurrency: 并发控制器
            pipeline_factory: 返回 Pipeline 实例的可调用对象（懒加载，首次用时构建）
            output_dir: 结果文件存放根目录，None 则用系统临时目录
            keep_recent: 保留最近完成的任务数（超出清理工作目录与结果文件）
        """
        self.concurrency = concurrency
        self._pipeline_factory = pipeline_factory
        self._pipeline: Any = None
        # 结果文件根目录：优先 output_dir，否则使用 task_dirs.DATA_DIR
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = task_dirs.DATA_DIR
        # 创建任务数据根目录与日志目录
        task_dirs.ensure_dirs()
        self.keep_recent = keep_recent
        # task_id -> 任务工作目录（含上传的源文件）
        self._work_dirs: Dict[str, str] = {}
        # task_id -> TaskResult
        self._results: Dict[str, TaskResult] = {}
        # 线程池：执行阻塞的 pipeline.process_file
        self._executor = ThreadPoolExecutor(
            max_workers=max(concurrency.max_concurrent, 1),
            thread_name_prefix="ocr-worker",
        )
        # 串行化 PaddleOCR 调用（PaddleOCR 非线程安全）
        self._ocr_lock = asyncio.Lock()
        # 进度卡死检测：连续 N 秒无进度更新视为卡死
        # 不再使用固定总超时——预约低峰期批量处理大 PDF（200+页）可能需要 30+ 分钟，
        # 只要进度在推进就应允许继续。仅当进度停滞（PaddleOCR 死锁）才终止。
        self.stall_timeout = 300  # 5 分钟无进度更新视为卡死
        self._progress_times: Dict[str, float] = {}  # task_id -> 上次进度更新时间
        # 卡死重建次数：连续 3 次卡死才标记 error，避免单次卡死误杀任务
        self._stall_counts: Dict[str, int] = {}
        # 预约任务调度器：每分钟检查一次是否有 scheduled 任务到点
        self._scheduler_task: Optional[asyncio.Task] = None
        # task_id -> 预约的 run_task 协程等待句柄（用于取消）
        self._scheduled_waiters: Dict[str, asyncio.Event] = {}
        # 已被用户删除的 scheduled 任务 id 集合
        # run_task 协程被唤醒后检查此集合，若在内则走删除分支而非激活分支
        self._deleted_tasks: set = set()
        # 已被用户暂停的 scheduled 任务 id 集合
        # run_task 协程被唤醒后检查此集合，若在内则走暂停分支退出（不报错）
        self._paused_marks: set = set()
        # 任务元信息：file_path / output_format（持久化与重试所需）
        self._task_meta: Dict[str, dict] = {}
        # 状态持久化文件：记录所有任务状态，服务重启后可恢复
        # scheduled/queued 重启后继续等待，running 标记为中断，done 保留下载链接
        self._state_file = os.path.join(self.output_dir, "_tasks_state.json")
        self._last_save_time = 0.0
        self._save_interval = 2.0  # 节流：最多每 2 秒写一次磁盘
        # 活跃的 run_task 协程句柄，shutdown 时统一取消，避免服务退不出来
        # （OCR 线程是 CPU 密集型，无法中断；取消协程后 fut 仍在跑，但服务可以退出）
        self._active_runners: Dict[str, asyncio.Task] = {}
        # 提前导出的 PDF 文件信息：partial_id -> {task_id, output_path, ...}
        # 任务运行中可多次提前导出，每个导出生成独立 partial_id
        self._partial_results: Dict[str, dict] = {}
        # 已取消/删除的任务集合：worker 线程通过 cancel_check 回调检测此集合，
        # 命中则提前退出循环，避免线程持续持有文件句柄导致目录无法删除
        self._cancelled_tasks: set = set()
        # 活跃的 executor future（task_id -> asyncio.Future），
        # delete_task/pause_task 设置取消标志后等待 fut 完成，确保线程退出再删文件
        self._active_futs: Dict[str, "asyncio.Future"] = {}
        # 任务当前正在 OCR 的页号（task_id -> page_no，-1 表示空闲），
        # 由 worker 线程通过 current_page_cb 上报，用于 stall 诊断时定位卡死页
        # 加锁保护：多进程并行时多个 worker 线程会并发写入
        self._current_pages: Dict[str, int] = {}
        self._current_pages_lock = threading.Lock()

    def start_scheduler(self) -> None:
        """启动预约任务调度器（在 FastAPI startup 事件中调用）。

        每 60 秒检查一次是否有 scheduled 状态的任务到达预约时间，
        到点则激活任务并调度执行。
        """
        if self._scheduler_task is not None:
            return

        async def _scheduler_loop():
            while True:
                try:
                    # 每 10 秒检查一次，确保预约任务到点后最多 10 秒内激活
                    await asyncio.sleep(10)
                    now = time.time()
                    # 找出到点的预约任务
                    due_ids = []
                    for tid, info in self.concurrency._tasks.items():
                        if (
                            info.status == "scheduled"
                            and info.scheduled_at is not None
                            and info.scheduled_at <= now
                        ):
                            due_ids.append(tid)
                    for tid in due_ids:
                        # 激活任务（转为 queued）并唤醒等待的 run_task
                        self.concurrency.activate(tid)
                        waiter = self._scheduled_waiters.pop(tid, None)
                        if waiter is not None:
                            waiter.set()
                except Exception:
                    logger.exception("预约任务调度器异常")

        self._scheduler_task = asyncio.create_task(_scheduler_loop())

    def _spawn_runner(self, *args, **kwargs) -> asyncio.Task:
        """创建 run_task 协程并注册到 _active_runners，shutdown 时可统一取消。

        协程结束后自动从 _active_runners 移除，避免内存累积。
        """
        task = asyncio.create_task(self.run_task(*args, **kwargs))
        # run_task 的第一个参数是 task_id，用于注册
        tid = args[0] if args else kwargs.get("task_id", "")
        self._active_runners[tid] = task

        def _cleanup(_t, _tid=tid):
            self._active_runners.pop(_tid, None)

        task.add_done_callback(_cleanup)
        return task

    def stop_scheduler(self) -> None:
        """停止调度器（在 shutdown 时调用）。"""
        if self._scheduler_task is not None:
            self._scheduler_task.cancel()
            self._scheduler_task = None

    def cancel_active_runners(self) -> None:
        """取消所有活跃的 run_task 协程（在 FastAPI shutdown 事件中调用）。

        必须在事件循环中调用，task.cancel() 才能生效。
        取消后 uvicorn 事件循环里没有待执行的协程，服务可以顺利退出。
        OCR 线程是孤儿，进程退出时由 OS 回收。
        """
        for tid, task in list(self._active_runners.items()):
            if not task.done():
                try:
                    task.cancel()
                except RuntimeError:
                    pass
        self._active_runners.clear()

    # ------------------------------------------------------------------
    # Pipeline 懒加载
    # ------------------------------------------------------------------
    def get_pipeline(self):
        """懒加载 Pipeline（首次调用时构建，加载 PaddleOCR 模型耗时较长）。"""
        if self._pipeline is None:
            self._pipeline = self._pipeline_factory()
        return self._pipeline

    # ------------------------------------------------------------------
    # 任务创建
    # ------------------------------------------------------------------
    def new_task_id(self) -> str:
        return uuid.uuid4().hex[:16]

    def set_owner(self, task_id: str, owner: str) -> None:
        """记录任务归属用户（列表/详情/下载按此过滤）。"""
        meta = self._task_meta.setdefault(task_id, {})
        meta["owner"] = owner
        try:
            self._save_task_meta(task_id)
        except Exception:
            pass

    def task_owner(self, task_id: str) -> Optional[str]:
        """返回任务归属用户；无归属（旧版遗留任务）返回 None。"""
        return self._task_meta.get(task_id, {}).get("owner")

    def check_owner(self, task_id: str, user: dict) -> bool:
        """当前用户是否有权访问该任务：管理员可访问全部；普通用户仅限本人任务。

        旧版遗留任务（无 owner）仅管理员可见，避免未登录时代的任务泄露给新用户。
        """
        if user.get("is_admin"):
            return True
        owner = self.task_owner(task_id)
        if owner is None:
            return False
        return owner == user.get("user_id")

    def prepare_upload(self, task_id: str, filename: str, data: bytes) -> str:
        """保存上传文件到任务工作目录，返回 task_id。

        对于 PDF 文件：保存后立即拆分到 pdf_pages/，原 PDF 被删除。
        对于非 PDF 文件（图片等）：直接保存到 source_dir/，不拆分。
        """
        # 创建任务目录结构（source/、pdf_pages/、ocr_pages/）
        task_dirs.create_task_dirs(task_id)
        self._work_dirs[task_id] = task_dirs.task_dir(task_id)
        # 防止路径穿越：仅用文件名
        safe_name = os.path.basename(filename)
        # 保存到 source_dir
        src_dir = task_dirs.source_dir(task_id)
        path = os.path.join(src_dir, safe_name)
        with open(path, "wb") as f:
            f.write(data)
        size_mb = len(data) / 1024 / 1024
        logger.info("上传文件: %s (%.1f MB) -> 任务 %s", safe_name, size_mb, task_id)
        log_progress(f"上传文件: {safe_name} ({size_mb:.1f} MB)")
        # PDF 文件：立即拆分到 pdf_pages/，拆分后原 PDF 被删除
        # 非 PDF 文件（jpg/png 等）不拆分，直接保存到 source_dir
        if safe_name.lower().endswith(".pdf"):
            try:
                total = task_processor.split_pdf_to_pages(path, task_id)
                logger.info("PDF 拆分完成: %s -> %d 页 (任务 %s)", safe_name, total, task_id)
                log_progress(f"PDF 拆分完成: {safe_name} ({total} 页)")
            except Exception as e:
                logger.exception("PDF 拆分失败: %s -> %s (%s)", safe_name, task_id, e)
                log_progress(f"PDF 拆分失败: {safe_name} ({e})")
                raise
        return task_id

    # ------------------------------------------------------------------
    # 任务执行
    # ------------------------------------------------------------------
    async def run_task(
        self,
        task_id: str,
        file_path: str,
        source_name: str,
        output_format: Optional[str] = None,
        scheduled_at: Optional[float] = None,
        batch_id: Optional[str] = None,
        task_concurrency: int = 1,
    ) -> None:
        """执行单个识别任务（并发控制 + 线程池 + 预约调度）。

        流程:
          1. 注册任务到并发控制器
             - 立即执行：register → queued
             - 预约执行：register_scheduled → scheduled，等待到点后 activate → queued
          2. await acquire 获取槽位（排队时阻塞，状态保持 queued）
          3. 在线程池执行 pipeline.process_file（不阻塞事件循环）
          4. 保存结果，更新任务状态
          5. release 释放槽位

        参数:
            batch_id: 批次 ID（同一次上传的所有文件共享），用于前端分组展示
        """
        # 对于非 PDF 文件（图片），从 source_dir 重建 file_path
        # prepare_upload 不再返回路径，run_task 的 file_path 参数仅为兼容保留
        # PDF 流程不使用 file_path（已拆分到 pdf_pages）
        if task_dirs.get_total_pages(task_id) == 0:
            reconstructed = os.path.join(task_dirs.source_dir(task_id), source_name)
            if os.path.isfile(reconstructed):
                file_path = reconstructed

        # 注册任务
        start_ts = time.time()
        if scheduled_at is not None and scheduled_at > time.time():
            # 预约任务：检查是否已由 upload 接口预注册为 scheduled
            # 如果未注册（如直接调用 run_task），才注册
            info = self.concurrency.get_status(task_id)
            if info is None:
                self.concurrency.register_scheduled(
                    task_id, scheduled_at, batch_id=batch_id, source_name=source_name,
                )
            # 记录任务元信息（持久化与重试用）
            self._task_meta[task_id] = {
                "file_path": file_path, "output_format": output_format,
                "source_name": source_name, "batch_id": batch_id,
                "task_concurrency": task_concurrency,
                # 保留 upload 接口预设的任务归属（多用户隔离）
                "owner": self._task_meta.get(task_id, {}).get("owner"),
            }
            self._save_task_meta(task_id)
            self._save_state()
            # 预约时间格式化为本地时间字符串
            from datetime import datetime as _dt
            sched_str = _dt.fromtimestamp(scheduled_at).strftime("%Y-%m-%d %H:%M:%S")
            logger.info("任务预约: %s -> %s (计划执行: %s)",
                        source_name, task_id, sched_str)
            log_progress(f"任务预约: {source_name} (计划执行: {sched_str})")
            # 创建等待事件，调度器到点时 set 唤醒
            waiter = asyncio.Event()
            self._scheduled_waiters[task_id] = waiter
            try:
                await waiter.wait()
            except asyncio.CancelledError:
                # 任务被取消（用户取消预约）
                self._scheduled_waiters.pop(task_id, None)
                self.concurrency.release(task_id, status="error", error="任务已取消")
                self._save_state()
                logger.info("预约任务已取消: %s", source_name)
                log_progress(f"预约任务已取消: {source_name}")
                return
            finally:
                self._scheduled_waiters.pop(task_id, None)
            # 被唤醒后检查：是到点激活，还是被用户删除/暂停？
            if task_id in self._deleted_tasks:
                self._deleted_tasks.discard(task_id)
                # delete_task 已清理工作目录/结果/元信息，这里只需释放并发槽位
                self.concurrency.release(task_id, status="error", error="任务已删除")
                self._save_state()
                logger.info("预约任务被删除: %s", source_name)
                log_progress(f"预约任务被删除: {source_name}")
                return
            if task_id in self._paused_marks:
                # 暂停：pause_task 已把状态置为 paused，协程直接退出即可
                # 不调用 release（pause 已处理状态转换和槽位释放）
                self._paused_marks.discard(task_id)
                self._save_state()
                logger.info("预约任务被暂停: %s", source_name)
                log_progress(f"预约任务被暂停: {source_name}")
                return
            logger.info("预约任务到点激活: %s", source_name)
            log_progress(f"预约任务到点激活: {source_name}")
        else:
            # 立即执行：注册为 queued
            self.concurrency.register(task_id, batch_id=batch_id, source_name=source_name)
            # 记录任务元信息（持久化与重试用）
            self._task_meta[task_id] = {
                "file_path": file_path, "output_format": output_format,
                "source_name": source_name, "batch_id": batch_id,
                "task_concurrency": task_concurrency,
                # 保留 upload 接口预设的任务归属（多用户隔离）
                "owner": self._task_meta.get(task_id, {}).get("owner"),
            }
            self._save_task_meta(task_id)
            self._save_state()
            logger.info("任务排队: %s -> %s", source_name, task_id)
            log_progress(f"任务排队: {source_name}")
        # 更新初始消息
        self.concurrency.update_progress(task_id, 0, 1, "等待处理槽位...")

        # 获取槽位（可能排队等待），同时分配实例池槽位号
        # task_concurrency > 1 时获取多个槽位，支持页级并行处理
        tc = max(1, min(task_concurrency, self.concurrency.max_concurrent))
        try:
            if tc > 1:
                slots, _immediate = await self.concurrency.acquire_many(task_id, tc)
                slot = slots[0]
            else:
                slot, _immediate = await self.concurrency.acquire(task_id)
                slots = [slot]
        except Exception as e:
            self.concurrency.release(task_id, status="error", error=str(e))
            return

        # 已获得槽位，开始处理
        self.concurrency.update_progress(task_id, 0, 1, "正在加载引擎并处理...")
        if tc > 1:
            logger.info("任务开始处理: %s (槽位 %s, %d进程并行)",
                        source_name, slots, tc)
            log_progress(f"任务开始: {source_name} (槽位 {slots}, {tc}进程并行)")
        else:
            logger.info("任务开始处理: %s (槽位 %d)", source_name, slot)
            log_progress(f"任务开始: {source_name} (槽位 {slot})")

        # 准备配置覆盖（输出格式）
        pipeline = self.get_pipeline()
        if output_format and output_format != pipeline.cfg.get("output_format"):
            pipeline.cfg["output_format"] = output_format
        # 输出目录固定到任务工作目录，避免污染源文件目录
        work_dir = self._work_dirs.get(task_id, task_dirs.task_dir(task_id))
        pipeline.cfg["output_dir"] = work_dir

        # 判断任务类型：PDF（已拆分到 pdf_pages）或图片（直接 OCR）
        is_pdf = task_dirs.get_total_pages(task_id) > 0

        # 进度回调（在线程中调用，仅更新数据，不涉及 await）
        # 同时记录更新时间，用于卡死检测
        # PDF 流程中 completed 来自 task_dirs.get_completed_pages（由 task_processor 维护）
        def progress_cb(cur: int, total: int, msg: str) -> None:
            self._progress_times[task_id] = time.time()
            self.concurrency.update_progress(task_id, cur, total, msg)

        # 取消检查回调：worker 线程在每页处理前调用，返回 True 表示任务已被删除/暂停
        # 命中后 process_task_pages 提前退出循环，释放文件句柄，让 delete_task 能清理目录
        def cancel_check() -> bool:
            return task_id in self._cancelled_tasks

        # 当前页号上报回调：worker 线程每页处理前调用，stall 诊断时用于定位卡死页
        # page_no=-1 表示该页处理结束（无论成功失败），清除标记避免脏数据
        def current_page_cb(tid: str, page_no: int) -> None:
            with self._current_pages_lock:
                self._current_pages[tid] = page_no

        # 在线程池执行阻塞调用
        loop = asyncio.get_running_loop()
        self._progress_times[task_id] = time.time()
        if is_pdf:
            # PDF 流程：task_processor.process_task_pages 处理所有未完成页
            fut = loop.run_in_executor(
                self._executor,
                self._process_sync,
                task_id,
                pipeline,
                slots,
                source_name,
                progress_cb,
                cancel_check,
                current_page_cb,
            )
        else:
            # 图片流程：保持原有 pipeline.process_file 调用
            # 文件路径从 source_dir 重建（prepare_upload 不再返回路径）
            img_path = os.path.join(task_dirs.source_dir(task_id), source_name)
            fut = loop.run_in_executor(
                self._executor,
                self._process_image_sync,
                pipeline,
                img_path,
                progress_cb,
                slot,
                slots if tc > 1 else None,
            )
        # 存储 future，delete_task/pause_task 设置取消标志后等待线程退出
        self._active_futs[task_id] = fut
        try:
            # 轮询检查完成或卡死，每 30 秒检查一次
            while not fut.done():
                await asyncio.sleep(30)
                if fut.done():
                    break
                # 输出内存与进度，方便定位内存累积
                try:
                    import psutil
                    mem_mb = psutil.Process().memory_info().rss / 1024 / 1024
                    info = self.concurrency.get_status(task_id)
                    logger.info(
                        "[监控] 任务 %s 进度 %s/%s 内存 %.0f MB",
                        task_id, info.progress if info else "?",
                        info.total if info else "?", mem_mb,
                    )
                except Exception:
                    pass
                last_progress = self._progress_times.get(task_id, 0)
                stalled = time.time() - last_progress > self.stall_timeout
                if stalled:
                    # === stall 诊断日志（在 kill 前收集，此时子进程状态最完整）===
                    # 目标：定位是哪一页卡死 + 子进程是否老化 + 主进程线程是否累积
                    with self._current_pages_lock:
                        current_page = self._current_pages.get(task_id, -1)
                    pids, child_mem, run_secs, page_cnts = self._collect_proc_diag(slots)
                    main_mem, thread_count, recv_line, recv_layout = self._collect_main_diag()
                    logger.error(
                        "[STALL诊断] task=%s source=%s 当前页=%d stall_count将=%d "
                        "卡死%ds | 子进程: pid=%s 运行%ss 处理%s页 RSS=%.0fMB | "
                        "主进程: RSS=%.0fMB 线程数=%d recv-line残留=%d recv-layout残留=%d",
                        task_id, source_name, current_page,
                        self._stall_counts.get(task_id, 0) + 1,
                        self.stall_timeout,
                        pids, run_secs, page_cnts, child_mem,
                        main_mem, thread_count, recv_line, recv_layout,
                    )
                    log_progress(
                        f"[STALL] {source_name} 第 {current_page} 页卡死 "
                        f"(pid={pids}, 子进程运行{run_secs}秒处理{page_cnts}页, "
                        f"主进程线程数={thread_count} recv残留={recv_line + recv_layout})"
                    )
                    logger.warning(
                        "任务进度卡死（%ds 无更新）: %s -> %s",
                        self.stall_timeout, source_name, task_id,
                    )
                    log_progress(f"任务卡死: {source_name} ({self.stall_timeout}s 无进度)")
                    # 卡死恢复策略：
                    #   1. 用 _kill_proc 切断子进程（不重启，避免 worker 线程用新进程继续处理）
                    #   2. 设置 _cancelled_tasks 让 worker 线程的 cancel_check 退出
                    #   3. 递增 stall_count，达到阈值则 release + return
                    #   4. release 后等待 worker 线程退出（与 delete_task 一致）
                    #
                    # 旧 Bug（导致信号量泄漏）：
                    #   - 用 _restart_proc（kill+重启），worker 线程用新进程继续处理，持有 slot 锁
                    #   - _restart_proc 抛异常时 stall_count 不递增，release 永不调用
                    #   - 不设 _cancelled_tasks，worker 线程不退出
                    #   - 不 pop _active_futs
                    #   最终信号量许可被永久占用，所有后续任务永远排队

                    # 1. kill 卡死的子进程（只 kill 不重启，切断 OCR 推理）
                    self._kill_ocr_slots(slots)

                    # 2. 递增 stall_count（无论 kill 是否成功都递增）
                    stall_count = self._stall_counts.get(task_id, 0) + 1
                    self._stall_counts[task_id] = stall_count
                    log_progress(f"已 kill 卡死子进程（第 {stall_count} 次）")

                    if stall_count >= 2:
                        # 连续 2 次卡死，标记 error 并释放槽位
                        logger.error(
                            "任务 %s 连续卡死 %d 次，标记为错误",
                            task_id, stall_count,
                        )
                        # 设置取消标记，让 worker 线程的 cancel_check 退出
                        self._cancelled_tasks.add(task_id)
                        # 释放信号量许可（让排队任务推进）
                        self.concurrency.release(
                            task_id, status="error", error="处理卡死",
                            message=f"进度卡死（重建 {stall_count} 次仍无进展）",
                        )
                        # 等待 worker 线程退出（与 delete_task 一致，10s 超时）
                        fut = self._active_futs.pop(task_id, None)
                        if fut is not None and not fut.done():
                            try:
                                await asyncio.wait_for(
                                    asyncio.shield(fut), timeout=10.0
                                )
                            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                                pass
                        # 清理取消标记和进度状态
                        self._cancelled_tasks.discard(task_id)
                        self._progress_times.pop(task_id, None)
                        self._stall_counts.pop(task_id, None)
                        with self._current_pages_lock:
                            self._current_pages.pop(task_id, None)
                        self._save_state()
                        return

                    # stall_count < 2：重置进度时间，给重建后的子进程 5 分钟窗口
                    self._progress_times[task_id] = time.time()
                    # 下次有新任务调用 ocr()/analyze() 时 _ensure_proc 会自动懒启动新子进程
            # 任务完成，获取结果
            if is_pdf:
                # PDF 流程：process_task_pages 返回 (success, completed_count)
                _success, _completed = fut.result()
                # success=False 且非用户取消/暂停：存在无法完成的页（如引擎级故障），
                # 不能照常合并输出——否则会把空白结果伪装成完成任务（2026-08-23 事故）
                if not _success and task_id not in self._cancelled_tasks:
                    raise RuntimeError(
                        f"存在无法完成的页面（仅完成 {_completed} 页），"
                        "任务按失败处理；详见 output/log/error_ocr.txt"
                    )
                # 根据用户选择的输出格式分流：
                #   - docx/txt/markdown/json：从 ocr_pages 重建 DocumentResult，调用 save_output
                #   - searchable_pdf/original：走原有 PDF 合并流程（叠加隐形文字层）
                fmt = pipeline.cfg.get("output_format") or "original"
                # PDF 源文件：original 等同于 searchable_pdf（还原为可搜索 PDF）
                if fmt == "original":
                    fmt = "searchable_pdf"

                if fmt in ("docx", "txt", "markdown", "json"):
                    self.concurrency.update_progress(
                        task_id, 0, 1, f"正在生成 {fmt.upper()} 文件...",
                    )
                    result = task_processor.build_document_result_from_ocr_pages(
                        task_id, source_name,
                    )
                    out_path = out_mod.save_output(
                        result, fmt, work_dir, render_dpi=200,
                    )
                    pages = len(result.pages)
                else:
                    # searchable_pdf：合并 ocr_pages 为最终可编辑 PDF
                    self.concurrency.update_progress(
                        task_id, 0, 1, "正在合并已识别页面...",
                    )
                    out_path, merged_count = task_processor.merge_ocr_pages(
                        task_id, source_name,
                        progress_cb=progress_cb,
                    )
                    result = None  # PDF 流程无 result 对象
                    pages = merged_count
            else:
                # 图片流程：返回 (result, out_path)
                result, out_path = fut.result()
                pages = len(result.pages) if result and result.pages else 0
        except asyncio.CancelledError:
            # 任务被取消（用户取消预约 / 暂停）
            # paused 标记：暂停场景，pause_task 已处理状态转换和槽位释放，这里不重复 release
            self._active_futs.pop(task_id, None)
            if task_id in self._paused_marks:
                self._paused_marks.discard(task_id)
                logger.info("任务已暂停（协程退出）: %s -> %s", source_name, task_id)
            else:
                # 真正的取消（用户取消预约等）
                self.concurrency.release(task_id, status="error", error="任务已取消")
            self._progress_times.pop(task_id, None)
            self._stall_counts.pop(task_id, None)
            with self._current_pages_lock:
                self._current_pages.pop(task_id, None)
            self._save_state()
            raise
        except Exception as e:
            logger.exception("任务处理失败: %s -> %s (%s)", source_name, task_id, e)
            log_progress(f"任务失败: {source_name} ({e})")
            self._active_futs.pop(task_id, None)
            self.concurrency.release(
                task_id, status="error", error=str(e),
                message=f"处理失败: {e}",
            )
            self._progress_times.pop(task_id, None)
            self._stall_counts.pop(task_id, None)
            with self._current_pages_lock:
                self._current_pages.pop(task_id, None)
            self._save_state()
            return

        # 保存结果（包在 try-except 中，避免后续处理异常导致槽位泄漏）
        try:
            if is_pdf:
                # PDF 流程：无 result 对象，pages 已在合并阶段确定
                text = ""
            else:
                # 图片流程：从 result 提取文本与页数
                text = self._extract_text(result, out_path, pipeline.cfg.get("output_format", "original"))
                pages = len(result.pages) if result and result.pages else 0
            out_name = os.path.basename(out_path) if out_path else source_name
            self._results[task_id] = TaskResult(
                task_id=task_id,
                source_name=source_name,
                output_path=out_path,
                output_name=out_name,
                output_format=pipeline.cfg.get("output_format", "original"),
                text=text,
                pages=pages,
                # 图像任务保留完整 OCR 结果，供识别后按需重新生成任意导出格式；
                # PDF 任务 result 为 None（从 ocr_pages 目录按需重建）
                document=result if result is not None else None,
            )
        except Exception as e:
            logger.exception("保存任务结果失败: %s -> %s (%s)", source_name, task_id, e)
            log_progress(f"保存结果失败: {source_name} ({e})")
            self._active_futs.pop(task_id, None)
            self.concurrency.release(
                task_id, status="error", error=str(e),
                message=f"保存结果失败: {e}",
            )
            self._progress_times.pop(task_id, None)
            self._stall_counts.pop(task_id, None)
            with self._current_pages_lock:
                self._current_pages.pop(task_id, None)
            self._save_state()
            return
        self.concurrency.set_result(task_id, out_path)
        # 释放槽位，唤醒下一个排队任务
        # 注意：set_result 只更新状态，不释放信号量；必须显式 release
        # 否则成功完成的任务会泄漏槽位，max_concurrent 个任务后所有后续任务永远排队
        self.concurrency.release(task_id)
        self._active_futs.pop(task_id, None)
        self._progress_times.pop(task_id, None)
        self._stall_counts.pop(task_id, None)
        with self._current_pages_lock:
            self._current_pages.pop(task_id, None)
        # 清理旧 partial PDF：任务运行中用户可能体验过"导出已识别部分"，
        # 留下 _partial_OCR.pdf（只有几页），任务完成后应清理避免混淆
        # pages_dir 已由 writer.close() 清理，这里只清理 partial 输出文件
        self._cleanup_partial_pdf(task_id)
        # 持久化最终结果（含下载链接，重启后仍可下载）
        self._save_state()
        # 任务完成日志：文件名、页数、总耗时
        elapsed = time.time() - start_ts
        logger.info(
            "任务完成: %s | %d 页 | 耗时 %.1fs | 输出: %s",
            source_name, pages, elapsed, out_name,
        )
        log_progress(
            f"任务完成: {source_name} | {pages} 页 | 耗时 {elapsed:.1f}s"
        )
        # 清理过期任务
        try:
            self.cleanup_old()
        except Exception:
            logger.debug("清理过期任务异常", exc_info=True)

    @staticmethod
    def _process_sync(task_id, pipeline, slots, source_name, progress_cb,
                      cancel_check=None, current_page_cb=None):
        """线程内同步执行 PDF 页级 OCR 处理流程。

        调用 task_processor.process_task_pages 完成：
          1. 获取未完成页号数组
          2. 按进程数分配任务
          3. 每进程 10 页后重新分配
          4. 循环直到未完成数组为空

        参数:
            cancel_check: 无参回调，返回 True 表示任务已被取消/删除，
                          process_task_pages 检测到后提前退出循环
            current_page_cb: 回调 (task_id, page_no) 上报当前正在处理的页号，
                             stall 触发时用于定位卡死页；page_no=-1 表示空闲

        返回 (success, completed_count)。
        """
        return task_processor.process_task_pages(
            task_id, pipeline, slots, source_name, progress_cb,
            cancel_check=cancel_check,
            current_page_cb=current_page_cb,
        )

    @staticmethod
    def _process_image_sync(pipeline, file_path: str, progress_cb, slot: int = 0,
                            slots=None):
        """线程内同步执行 pipeline.process_file（图片流程），传入实例池槽位号。

        slots 非空时启用多进程并行处理PDF（页级并行）。
        """
        if slots and len(slots) > 1:
            return pipeline.process_file(
                file_path, progress_cb=progress_cb, slot=slot, slots=slots
            )
        return pipeline.process_file(file_path, progress_cb=progress_cb, slot=slot)

    @staticmethod
    def _extract_text(result, out_path: str, fmt: str) -> str:
        """从结果中提取纯文本供前端预览。"""
        # 文本类输出直接读文件
        if out_path and os.path.isfile(out_path):
            ext = os.path.splitext(out_path)[1].lower()
            if ext in (".txt", ".md", ".json"):
                try:
                    with open(out_path, "r", encoding="utf-8") as f:
                        return f.read()
                except Exception:
                    pass
        # 二进制输出（pdf/docx）或读文件失败：从 DocumentResult 提取
        if result is None or not result.pages:
            return ""
        parts = []
        if getattr(result, "native_text", ""):
            parts.append(result.native_text)
        for page in result.pages:
            if page.skipped:
                continue
            parts.append(page.ocr_result.text)
        return "\n\n".join(p for p in parts if p)

    # ------------------------------------------------------------------
    # 结果查询
    # ------------------------------------------------------------------
    def get_task_info(self, task_id: str) -> Optional[TaskInfo]:
        return self.concurrency.get_status(task_id)

    def get_result(self, task_id: str) -> Optional[TaskResult]:
        return self._results.get(task_id)

    def get_task(self, task_id: str) -> Optional[dict]:
        """获取单个任务的详细信息（含页级进度）。

        返回包含任务状态、结果、页级进度（total_pages/completed_pages/page_progress）的字典。
        任务不存在时返回 None。
        """
        info = self.concurrency.get_status(task_id)
        if info is None:
            return None
        res = self._results.get(task_id)
        d = info.to_dict()
        d["source_name"] = (res.source_name if res else None) or info.source_name
        d["has_result"] = res is not None
        d["output_name"] = res.output_name if res else None
        d["pages"] = res.pages if res else 0
        meta = self._task_meta.get(task_id, {})
        d["task_concurrency"] = int(meta.get("task_concurrency", 1))
        # 页级进度信息（基于 task_dirs 的 pdf_pages/ocr_pages 统计）
        total_pages = task_dirs.get_total_pages(task_id)
        completed_pages = task_dirs.get_completed_pages(task_id)
        d["total_pages"] = total_pages
        d["completed_pages"] = completed_pages
        d["page_progress"] = (
            completed_pages / total_pages if total_pages > 0 else 0
        )
        return d

    async def delete_task(self, task_id: str) -> Tuple[bool, str]:
        """删除任务（含工作目录、结果文件、内存状态）。

        返回 (成功?, 原因说明)。
        - running 状态：切断 OCR 推理 + 设置取消标志 + 等待线程退出 + 取消协程 + 释放槽位
        - scheduled 状态：唤醒等待协程让它退出
        - queued 状态：直接从队列移除
        - done/error 状态：清理工作目录和结果文件

        异步方法：running 状态下需要 await executor future 等待 worker 线程退出，
        确保文件句柄全部释放后再删除目录，避免 Windows 文件锁导致 rmtree 残留。
        """
        info = self.concurrency.get_status(task_id)
        if info is None:
            return False, "任务不存在"
        if info.status == "running":
            # 强制删除运行中任务：
            # 1. 设置取消标志：worker 线程的 cancel_check 回调检测到后提前退出循环
            # 2. kill 全部槽位的 OCR 引擎（kill 子进程/重建实例），切断推理
            # 3. 等待 worker 线程退出（最多 10 秒），确保文件句柄释放
            # 4. 取消 run_task 协程（触发 CancelledError → release）
            # 5. 主动 release 释放槽位（不依赖协程时序，避免槽位泄漏）
            all_slots = list(info.slots) if info.slots else (
                [info.slot] if info.slot is not None else []
            )
            source_name = info.source_name or "未知"
            logger.warning(
                "强制删除运行中任务: %s (槽位 %s) -> %s", source_name, all_slots, task_id,
            )
            log_progress(f"强制删除运行中任务: {source_name}")
            # 先设置取消标志，让 worker 线程在下一个页面检查点退出
            self._cancelled_tasks.add(task_id)
            # kill 全部槽位（多槽位任务需 kill 所有子进程，否则线程继续处理）
            if all_slots:
                self._kill_ocr_slots(all_slots)
            # 标记删除，让 run_task 的 CancelledError 处理走删除分支
            self._deleted_tasks.add(task_id)
            # 等待 worker 线程退出：cancel_check 已设置 + 子进程已 kill，
            # 线程会在当前页 OCR 失败后检查到取消标志并退出
            fut = self._active_futs.get(task_id)
            if fut is not None and not fut.done():
                try:
                    await asyncio.wait_for(asyncio.shield(fut), timeout=10.0)
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                    # 超时或异常：线程仍在跑（可能卡在 OCR），继续删除目录
                    # rmtree 用 ignore_errors=True 容忍文件锁，残留文件由 OS 回收
                    logger.warning(
                        "等待 worker 线程退出超时(10s)，强制删除目录: %s", task_id,
                    )
            self._active_futs.pop(task_id, None)
            # 取消 run_task 协程
            runner = self._active_runners.get(task_id)
            if runner is not None and not runner.done():
                try:
                    runner.cancel()
                except RuntimeError:
                    pass
            # 主动释放槽位（run_task 的 CancelledError 也会 release，
            # 但重复 release 安全：第二次 had_slot=False，只更新状态）
            self.concurrency.release(
                task_id, status="error", error="任务已删除",
                message="运行中任务被用户删除",
            )
            self._progress_times.pop(task_id, None)
            self._stall_counts.pop(task_id, None)
            self._cancelled_tasks.discard(task_id)
        # scheduled 任务：唤醒等待协程，让它走删除分支退出
        elif info.status == "scheduled":
            waiter = self._scheduled_waiters.pop(task_id, None)
            if waiter is not None:
                # 不能直接 set，否则协程会继续走 register 流程
                # 用一个标记位让协程知道是被删除而非激活
                self._deleted_tasks.add(task_id)
                waiter.set()
        # 删除工作目录
        wd = self._work_dirs.pop(task_id, None)
        if wd and os.path.isdir(wd):
            try:
                shutil.rmtree(wd, ignore_errors=True)
            except Exception:
                pass
        # 删除结果文件（若不在工作目录内）
        res = self._results.pop(task_id, None)
        if res and res.output_path and os.path.isfile(res.output_path):
            if wd is None or not res.output_path.startswith(wd):
                try:
                    os.remove(res.output_path)
                except Exception:
                    pass
        # 删除任务元信息
        self._task_meta.pop(task_id, None)
        # 删除该任务关联的提前导出 PDF（partial_results）
        # 工作目录被删后这些文件已不存在，从内存中移除引用
        stale_partials = [
            pid for pid, info in self._partial_results.items()
            if info.get("task_id") == task_id
        ]
        for pid in stale_partials:
            self._partial_results.pop(pid, None)
        # 从并发管理器移除
        self.concurrency.remove(task_id)
        # 持久化状态
        self._save_state(force=True)
        return True, "已删除"

    # ------------------------------------------------------------------
    # 暂停 / 恢复
    # ------------------------------------------------------------------
    def pause_task(self, task_id: str) -> Tuple[bool, str]:
        """暂停任务（支持 running/queued/scheduled）。

        running 状态：
          1. 重启 OCR slot 切断正在进行的推理
          2. 取消 run_task 协程（触发 CancelledError）
          3. 并发控制器置 paused + 释放槽位（其他任务可推进）
          4. pages_dir 保留，恢复时自动断点续传

        queued 状态：直接置 paused + 从队列移除（不释放信号量，未获取过）
        scheduled 状态：唤醒等待协程让它退出，置 paused
        """
        info = self.concurrency.get_status(task_id)
        if info is None:
            return False, "任务不存在"
        source_name = info.source_name or "未知"

        # 调用并发控制器的 pause（处理状态转换 + 槽位释放）
        ok, reason, slots_to_kill = self.concurrency.pause(task_id)
        if not ok:
            return False, reason

        # running 任务：切断 OCR 推理 + 取消协程
        if slots_to_kill:
            logger.info(
                "暂停运行中任务: %s (槽位 %s) -> %s",
                source_name, slots_to_kill, task_id,
            )
            log_progress(f"暂停任务: {source_name}")
            self._kill_ocr_slots(slots_to_kill)
            runner = self._active_runners.get(task_id)
            if runner is not None and not runner.done():
                try:
                    runner.cancel()
                except RuntimeError:
                    pass
            self._progress_times.pop(task_id, None)
            self._stall_counts.pop(task_id, None)
        else:
            # queued / scheduled：只需唤醒等待协程退出
            logger.info("暂停任务: %s -> %s (原状态: %s)", source_name, task_id, info.pre_pause_status)
            log_progress(f"暂停任务: {source_name}")
            if info.pre_pause_status == "scheduled":
                waiter = self._scheduled_waiters.pop(task_id, None)
                if waiter is not None:
                    # 标记为暂停（非删除），让协程走暂停分支退出
                    self._paused_marks.add(task_id)
                    waiter.set()
            elif info.pre_pause_status == "queued":
                # queued 任务正在 await semaphore.acquire()，必须 cancel 才能退出
                # CancelledError 会触发 except 分支，但那里调用了 release(status=error)
                # pause 已把状态置为 paused，release 会覆盖回 error —— 需要用标记位跳过
                self._paused_marks.add(task_id)
                runner = self._active_runners.get(task_id)
                if runner is not None and not runner.done():
                    try:
                        runner.cancel()
                    except RuntimeError:
                        pass
            self._progress_times.pop(task_id, None)
            self._stall_counts.pop(task_id, None)

        self._save_state(force=True)
        return True, "已暂停"

    def resume_task(self, task_id: str) -> Tuple[bool, str]:
        """恢复暂停的任务，重新进入排队/调度。

        pre_pause=running/queued → 置 queued 重新排队（pipeline 断点续传）
        pre_pause=scheduled → 置 scheduled，重新创建 _spawn_runner 等待预约
        """
        info = self.concurrency.get_status(task_id)
        if info is None:
            return False, "任务不存在"
        if info.status != "paused":
            return False, f"任务状态 {info.status} 不可恢复（仅 paused 可恢复）"

        meta = self._task_meta.get(task_id, {})
        file_path = meta.get("file_path", "")
        # PDF 流程：file_path 已删除，用 pdf_pages 判断是否可恢复
        # 图片流程：file_path 仍存在于 source_dir
        is_pdf = task_dirs.get_total_pages(task_id) > 0
        if not is_pdf:
            if not file_path or not os.path.isfile(file_path):
                return False, "源文件不存在，无法恢复"

        ok, reason, new_status = self.concurrency.resume(task_id)
        if not ok:
            return False, reason

        source_name = meta.get("source_name") or info.source_name or "未知"
        output_format = meta.get("output_format")
        batch_id = meta.get("batch_id")
        scheduled_at = info.scheduled_at
        # 保留原任务级并发数，避免恢复后退化为单进程
        task_concurrency = int(meta.get("task_concurrency", 1))

        logger.info("恢复任务: %s -> %s (新状态: %s)", source_name, task_id, new_status)
        log_progress(f"恢复任务: {source_name}")

        # 重新创建 runner 协程
        # run_task 会检测到已有 task_id 已注册，直接走 acquire 流程
        # pipeline 会检测 pages_dir 已有页数，自动断点续传
        self._spawn_runner(
            task_id, file_path, source_name,
            output_format=output_format,
            scheduled_at=scheduled_at,
            batch_id=batch_id,
            task_concurrency=task_concurrency,
        )
        self._save_state(force=True)
        return True, "已恢复"

    def _kill_ocr_slot(self, slot: int) -> None:
        """仅 kill 指定槽位的 OCR 和版面分析子进程，不重启新进程。

        用 _kill_proc 代替 _restart_proc：
          - _restart_proc: kill + 启动新进程 + 等待模型加载 ready（10-20s/子进程）
          - _kill_proc: 仅 kill（<1s/子进程）

        delete_task / pause_task 只需切断推理让 worker 线程退出，不需要立即启动
        新子进程（任务已删除/暂停，新进程启起来也没用）。下次有新任务调用
        ocr()/analyze() 时 _ensure_proc 会自动懒启动新进程。

        属性访问路径与 stall 检测保持一致：
        pipeline._paddle._pool（SubprocessOCRProvider._pool = SubprocessOCRPool）
        pipeline._layout_pool（SubprocessLayoutPool）
        """
        if self._pipeline is None:
            return
        try:
            # === 1. Kill OCR 子进程 ===
            paddle_provider = getattr(self._pipeline, "_paddle", None)
            pool = getattr(paddle_provider, "_pool", None) if paddle_provider else None
            if pool is not None and slot < len(pool._procs):
                pool._kill_proc(slot)
                logger.warning("已 kill OCR 子进程 slot=%d", slot)
            else:
                # 同进程模式：无子进程可 kill，仅记录
                pass

            # === 2. Kill 版面分析子进程 ===
            layout_pool = getattr(self._pipeline, "_layout_pool", None)
            if layout_pool is not None and slot < len(layout_pool._procs):
                layout_pool._kill_proc(slot)
                logger.warning("已 kill 版面分析子进程 slot=%d", slot)
        except Exception as e:
            logger.error("kill OCR/Layout slot=%d 失败: %s", slot, e)

    def _kill_ocr_slots(self, slots: List[int]) -> None:
        """kill 多个槽位的子进程（仅 kill 不重启，<1s/slot）。

        多槽位任务（task_concurrency > 1）删除/暂停时需 kill 全部槽位，
        否则其余子进程仍存活，worker 线程继续处理页面持有文件句柄。
        """
        for slot in slots:
            self._kill_ocr_slot(slot)

    def _collect_proc_diag(self, slots: List[int]) -> Tuple[List[int], float, List[int], List[int]]:
        """收集 OCR/Layout 子进程诊断信息。

        用于 stall 触发时输出子进程状态，辅助判断是否为内存累积或子进程老化。

        返回:
            (pids, total_mem_mb, run_seconds_list, page_counts_list)
            - pids: 存活子进程的 pid 列表
            - total_mem_mb: 所有存活子进程 RSS 之和（MB）
            - run_seconds_list: 每个存活子进程已运行秒数
            - page_counts_list: 每个存活子进程已处理页数
        """
        pids: List[int] = []
        total_mem_mb = 0.0
        run_seconds: List[int] = []
        page_counts: List[int] = []
        try:
            import psutil
            now = time.time()
            # OCR 子进程
            pool = getattr(getattr(self._pipeline, "_paddle", None), "_pool", None)
            if pool is not None:
                for slot in slots:
                    if slot >= len(pool._procs):
                        continue
                    proc = pool._procs[slot]
                    if proc is None or proc.poll() is not None:
                        continue
                    pids.append(proc.pid)
                    start_t = pool._start_times[slot] if slot < len(pool._start_times) else 0.0
                    run_seconds.append(int(now - start_t) if start_t else -1)
                    page_counts.append(pool._page_counts[slot] if slot < len(pool._page_counts) else -1)
                    try:
                        total_mem_mb += psutil.Process(proc.pid).memory_info().rss / 1024 / 1024
                    except Exception:
                        pass
            # 版面分析子进程（合并统计）
            layout_pool = getattr(self._pipeline, "_layout_pool", None)
            if layout_pool is not None:
                for slot in slots:
                    if slot >= len(layout_pool._procs):
                        continue
                    proc = layout_pool._procs[slot]
                    if proc is None or proc.poll() is not None:
                        continue
                    pids.append(proc.pid)
                    start_t = layout_pool._start_times[slot] if slot < len(layout_pool._start_times) else 0.0
                    run_seconds.append(int(now - start_t) if start_t else -1)
                    page_counts.append(layout_pool._page_counts[slot] if slot < len(layout_pool._page_counts) else -1)
                    try:
                        total_mem_mb += psutil.Process(proc.pid).memory_info().rss / 1024 / 1024
                    except Exception:
                        pass
        except Exception as e:
            logger.debug("收集子进程诊断失败: %s", e)
        return pids, total_mem_mb, run_seconds, page_counts

    def _collect_main_diag(self) -> Tuple[float, int, int, int]:
        """收集主进程诊断信息。

        返回:
            (rss_mb, thread_count, recv_line_alive, recv_layout_alive)
            - rss_mb: 主进程 RSS（MB）
            - thread_count: 主进程总线程数
            - recv_line_alive: 阻塞在 OCR _recv 的 reader_thread 残留数
            - recv_layout_alive: 阻塞在 layout _recv 的 reader_thread 残留数

        残留 reader_thread 数异常增长（>10）指向 stall 根因：
        长时间运行后超时未退出的线程累积占 GIL，导致整体变慢直至卡死。
        """
        try:
            import psutil
            import threading as _threading
            proc = psutil.Process()
            rss_mb = proc.memory_info().rss / 1024 / 1024
            threads = _threading.enumerate()
            thread_count = len(threads)
            recv_line_alive = sum(1 for t in threads if t.name.startswith("recv-line-") and t.is_alive())
            recv_layout_alive = sum(1 for t in threads if t.name.startswith("recv-layout-") and t.is_alive())
            return rss_mb, thread_count, recv_line_alive, recv_layout_alive
        except Exception as e:
            logger.debug("收集主进程诊断失败: %s", e)
            return 0.0, -1, -1, -1

    def list_tasks(self, owner: str = None, is_admin: bool = False) -> List[dict]:
        """列出任务（最新在前）。

        owner 过滤：非管理员只看自己的任务；管理员看全部。
        旧版遗留任务（无 owner）仅管理员可见（避免未登录时代的任务泄露）。
        """
        items = []
        for tid, info in self.concurrency._tasks.items():
            if not is_admin:
                if self.task_owner(tid) != owner:
                    continue
            res = self._results.get(tid)
            d = info.to_dict()
            # 优先用结果中的 source_name（已完成任务），回退到任务信息的 source_name
            # 避免排队/处理中任务因无结果而丢失文件名展示
            d["source_name"] = (res.source_name if res else None) or info.source_name
            d["has_result"] = res is not None
            d["output_name"] = res.output_name if res else None
            d["pages"] = res.pages if res else 0
            # 返回任务级并发数，供前端展示（验证重试/恢复后是否保留）
            meta = self._task_meta.get(tid, {})
            d["task_concurrency"] = int(meta.get("task_concurrency", 1))
            # 返回页级进度信息（基于 task_dirs 的 pdf_pages/ocr_pages 统计）
            # total_pages: 拆分后的总页数；completed_pages: 已完成 OCR 的页数
            total_pages = task_dirs.get_total_pages(tid)
            completed_pages = task_dirs.get_completed_pages(tid)
            d["total_pages"] = total_pages
            d["completed_pages"] = completed_pages
            # 进度百分比：completed/total（total为0时为0）
            d["page_progress"] = (
                completed_pages / total_pages if total_pages > 0 else 0
            )
            items.append(d)
        items.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return items

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------
    def cleanup_old(self) -> int:
        """清理过期任务的工作目录与结果文件，保留最近 keep_recent 个已完成任务。

        返回清理的任务数。
        """
        # 收集已完成（done/error）的任务，按完成时间倒序
        done = [
            (tid, info)
            for tid, info in self.concurrency._tasks.items()
            if info.status in ("done", "error")
        ]
        done.sort(key=lambda x: x[1].finished_at or 0, reverse=True)
        removed = 0
        for tid, info in done[self.keep_recent:]:
            # 删除工作目录
            wd = self._work_dirs.pop(tid, None)
            if wd and os.path.isdir(wd):
                try:
                    shutil.rmtree(wd, ignore_errors=True)
                except Exception:
                    pass
            # 删除结果文件（可能在工作目录内，已随目录删除；外部则单独删）
            res = self._results.pop(tid, None)
            if res and res.output_path and os.path.isfile(res.output_path):
                # 若输出路径不在工作目录内才单独删
                wd = self._work_dirs.get(tid)
                if wd is None or not res.output_path.startswith(wd):
                    try:
                        os.remove(res.output_path)
                    except Exception:
                        pass
            # 删除该任务关联的提前导出 PDF（partial_results）
            stale_partials = [
                pid for pid, info in self._partial_results.items()
                if info.get("task_id") == tid
            ]
            for pid in stale_partials:
                self._partial_results.pop(pid, None)
            # 从并发管理器移除
            self.concurrency.remove(tid)
            removed += 1
        return removed

    def shutdown(self) -> None:
        """关闭线程池与调度器，取消所有活跃任务协程。

        OCR 线程是 CPU 密集型无法中断，但取消协程后服务可以退出
        （线程成为孤儿，进程退出时由 OS 回收）。

        若在事件循环中调用，会同步取消协程；若在循环外（main.py finally），
        cancel() 会安全地把请求加入队列（无需 await）。
        """
        self.stop_scheduler()
        # 取消所有活跃的 run_task 协程，避免 uvicorn 等待事件循环空不下来
        # task.cancel() 只标记取消请求，不阻塞，即使无事件循环也安全
        for tid, task in list(self._active_runners.items()):
            if not task.done():
                try:
                    task.cancel()
                except RuntimeError:
                    # 某些边界场景下 task 可能已完成或无事件循环，忽略
                    pass
        self._active_runners.clear()
        # 立即关闭线程池（不等待运行中的 OCR 线程）
        self._executor.shutdown(wait=False, cancel_futures=True)
        # 关闭子进程 OCR 池（避免子进程成为孤儿）
        if self._pipeline is not None:
            try:
                pool = getattr(self._pipeline, "_subprocess_pool", None)
                if pool is not None:
                    pool.shutdown()
            except Exception:
                pass
            # 阶段4：同时关闭子进程版面分析池（避免 layout 子进程成为孤儿）
            try:
                layout_pool = getattr(self._pipeline, "_layout_pool", None)
                if layout_pool is not None:
                    layout_pool.shutdown()
            except Exception:
                pass
        # 关闭前强制写入最终状态
        self._save_state(force=True)

    # ------------------------------------------------------------------
    # 状态持久化
    # ------------------------------------------------------------------
    def _save_state(self, force: bool = False) -> None:
        """把所有任务状态写入 JSON 文件，重启后可恢复。

        节流策略：非 force 模式下最多每 _save_interval 秒写一次磁盘，
        避免高频进度更新导致频繁 I/O。force=True 时立即写入（用于 shutdown）。

        持久化的任务：
          - scheduled / queued：重启后继续等待或重新排队
          - running：重启后标记为 interrupted（服务中断）
          - done / error：重启后保留结果，可下载
        """
        # 节流：非强制模式下限制写入频率
        now = time.time()
        if not force and (now - self._last_save_time) < self._save_interval:
            return
        self._last_save_time = now

        import json
        tasks_data = []
        for tid, info in self.concurrency._tasks.items():
            meta = self._task_meta.get(tid, {})
            res = self._results.get(tid)
            # 序列化任务信息
            entry = {
                "task_id": tid,
                "status": info.status,
                "progress": info.progress,
                "total": info.total,
                "message": info.message,
                "error": info.error,
                "created_at": info.created_at,
                "started_at": info.started_at,
                "finished_at": info.finished_at,
                "scheduled_at": info.scheduled_at,
                "batch_id": info.batch_id,
                "source_name": info.source_name,
                "pre_pause_status": info.pre_pause_status,
                # 元信息：重启后重新执行所需
                "file_path": meta.get("file_path"),
                "output_format": meta.get("output_format"),
                # 工作目录：优先 _work_dirs，回退到 task_dirs.task_dir
                "work_dir": self._work_dirs.get(tid) or task_dirs.task_dir(tid),
                # 任务级并发数：重试/恢复时必须保持，否则退化为单进程
                "task_concurrency": meta.get("task_concurrency", 1),
                # 任务归属用户（多用户隔离；旧任务无此字段）
                "owner": meta.get("owner"),
            }
            # 已完成任务附带结果信息（重启后仍可下载）
            if res is not None:
                entry["result"] = {
                    "output_path": res.output_path,
                    "output_name": res.output_name,
                    "output_format": res.output_format,
                    "text": res.text,
                    "pages": res.pages,
                }
            tasks_data.append(entry)

        state = {
            "version": 1,
            "saved_at": now,
            "tasks": tasks_data,
            # 提前导出的 PDF 链接（重启后仍可下载）
            "partial_results": {
                pid: info for pid, info in self._partial_results.items()
                if info.get("output_path") and os.path.isfile(info["output_path"])
            },
        }
        # 原子写入：先写临时文件再替换，避免写入中途崩溃导致文件损坏
        tmp_path = self._state_file + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self._state_file)
        except Exception:
            logger.debug("保存任务状态失败", exc_info=True)
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def _save_task_meta(self, task_id: str) -> None:
        """保存单个任务的元信息到任务目录下的 meta.json。

        与 _tasks_state.json 互为冗余：即使 _tasks_state.json 丢失或损坏，
        也能通过扫描任务目录下的 meta.json 恢复任务状态（含 task_concurrency）。

        每次任务创建/状态变更时调用，确保元信息持久化。
        """
        meta = self._task_meta.get(task_id)
        if meta is None:
            return
        info = self.concurrency.get_status(task_id)
        try:
            import json
            meta_data = {
                "task_id": task_id,
                "source_name": meta.get("source_name", ""),
                "output_format": meta.get("output_format"),
                "batch_id": meta.get("batch_id"),
                "task_concurrency": meta.get("task_concurrency", 1),
                "owner": meta.get("owner"),
                "status": info.status if info else "unknown",
                "progress": info.progress if info else 0,
                "total": info.total if info else 0,
                "created_at": info.created_at if info else 0,
                "finished_at": info.finished_at if info else None,
                "saved_at": time.time(),
            }
            # 添加结果信息（已完成任务）
            res = self._results.get(task_id)
            if res is not None:
                meta_data["result"] = {
                    "output_path": res.output_path,
                    "output_name": res.output_name,
                    "pages": res.pages,
                }
            meta_path = task_dirs.task_meta_path(task_id)
            os.makedirs(os.path.dirname(meta_path), exist_ok=True)
            tmp_path = meta_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, meta_path)
        except Exception:
            logger.debug("保存任务元信息失败 (task=%s)", task_id, exc_info=True)

    def _scan_and_recover_tasks(self) -> int:
        """扫描 DATA_DIR 下的任务目录，恢复 _tasks_state.json 中缺失的任务。

        场景：_tasks_state.json 丢失/损坏，或服务异常退出后状态文件未更新。
        通过扫描每个任务目录下的 meta.json + pdf_pages/ocr_pages 文件自动恢复。

        恢复规则：
          - 有 meta.json → 读取 task_concurrency、source_name 等
          - 无 meta.json → 从目录结构推断（source_name 未知，用 task_id）
          - 有最终 PDF (_xxx_OCR.pdf) → 标记 done
          - 有 pdf_pages 但无最终 PDF → 标记 interrupted（断点续传）
          - ocr_pages 完整 = pdf_pages → 标记 done（合并未完成）
          - 空目录 → 跳过
        """
        import json
        if not os.path.isdir(task_dirs.DATA_DIR):
            return 0

        # 收集 _tasks_state.json 中已有的 task_id
        existing_tids = set()
        try:
            if os.path.isfile(self._state_file):
                with open(self._state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                for entry in state.get("tasks", []):
                    tid = entry.get("task_id")
                    if tid:
                        existing_tids.add(tid)
        except Exception:
            pass

        recovered = 0
        now = time.time()
        for name in os.listdir(task_dirs.DATA_DIR):
            tid = name
            if tid in existing_tids:
                continue
            task_d = task_dirs.task_dir(tid)
            if not os.path.isdir(task_d):
                continue

            # 必须有 pdf_pages 目录才算有效任务（图片任务无 pdf_pages，跳过）
            pdf_pages = task_dirs.pdf_pages_dir(tid)
            if not os.path.isdir(pdf_pages):
                continue
            total_pages = task_dirs.get_total_pages(tid)
            if total_pages == 0:
                continue

            # 读取 meta.json（如果存在）
            meta_path = task_dirs.task_meta_path(tid)
            source_name = tid
            output_format = None
            batch_id = None
            task_concurrency = 1
            owner = None
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta_data = json.load(f)
                    source_name = meta_data.get("source_name", tid)
                    output_format = meta_data.get("output_format")
                    batch_id = meta_data.get("batch_id")
                    task_concurrency = int(meta_data.get("task_concurrency", 1))
                    owner = meta_data.get("owner")
                except Exception:
                    logger.warning("读取 meta.json 失败: %s", meta_path)

            # 检查是否有最终 PDF
            # 从目录中查找 _xxx_OCR.pdf 文件
            final_pdf = None
            for fname in os.listdir(task_d):
                if fname.endswith("_OCR.pdf") and fname.startswith("_"):
                    final_pdf = os.path.join(task_d, fname)
                    break

            completed_pages = task_dirs.get_completed_pages(tid)

            # 恢复元信息
            self._task_meta[tid] = {
                "file_path": "",
                "output_format": output_format,
                "source_name": source_name,
                "batch_id": batch_id,
                "task_concurrency": task_concurrency,
                "owner": owner,
            }
            self._work_dirs[tid] = task_d

            if final_pdf and os.path.isfile(final_pdf):
                # 有最终 PDF：标记为 done
                page_count = 0
                try:
                    import fitz
                    doc = fitz.open(final_pdf)
                    page_count = doc.page_count
                    doc.close()
                except Exception:
                    page_count = completed_pages
                self._results[tid] = TaskResult(
                    task_id=tid,
                    source_name=source_name,
                    output_path=final_pdf,
                    output_name=os.path.basename(final_pdf),
                    output_format=output_format or "original",
                    pages=page_count,
                )
                info = self.concurrency.register_scheduled(
                    tid, now, batch_id=batch_id, source_name=source_name,
                )
                info.status = "done"
                info.progress = page_count
                info.total = max(page_count, total_pages)
                info.finished_at = now
                info.message = "完成（扫描恢复）"
                recovered += 1
                logger.info(
                    "扫描恢复任务（已完成）: %s (%s) %d页",
                    tid, source_name, page_count,
                )
                log_progress(f"扫描恢复已完成任务: {source_name} ({page_count} 页)")
            elif completed_pages >= total_pages:
                # 所有页都已 OCR，但未合并：标记为 interrupted 让 run_task 合并
                self._spawn_runner(
                    tid, "", source_name, output_format,
                    batch_id=batch_id, task_concurrency=task_concurrency,
                )
                recovered += 1
                logger.info(
                    "扫描恢复任务（待合并）: %s (%s) %d/%d页",
                    tid, source_name, completed_pages, total_pages,
                )
                log_progress(
                    f"扫描恢复待合并任务: {source_name} ({completed_pages}/{total_pages} 页)"
                )
            else:
                # 部分完成：断点续传
                self._spawn_runner(
                    tid, "", source_name, output_format,
                    batch_id=batch_id, task_concurrency=task_concurrency,
                )
                recovered += 1
                logger.info(
                    "扫描恢复任务（断点续传）: %s (%s) %d/%d页",
                    tid, source_name, completed_pages, total_pages,
                )
                log_progress(
                    f"扫描恢复断点续传任务: {source_name} ({completed_pages}/{total_pages} 页)"
                )

        if recovered > 0:
            logger.info("扫描文件夹恢复 %d 个任务", recovered)
            log_progress(f"扫描文件夹恢复 {recovered} 个任务")
        return recovered

    def restore_state(self) -> int:
        """从磁盘加载任务状态，恢复 scheduled/queued/done/error 任务。

        在服务启动时（FastAPI startup 事件后）调用。
        返回恢复的任务数。

        恢复流程：
          1. 先扫描 DATA_DIR 下的任务目录，恢复 _tasks_state.json 中缺失的任务
             （应对状态文件丢失/损坏的场景，通过 meta.json + 文件结构恢复）
          2. 再从 _tasks_state.json 恢复已记录的任务

        恢复规则：
          - scheduled → 重新注册 scheduled，启动 run_task 协程等待到点
          - queued    → 重新注册 queued，启动 run_task 协程排队执行
          - running   → 标记为 interrupted（error 状态，特殊消息），用户可手动重试
          - done/error → 恢复结果信息，可下载/查看
          - 源文件不存在的任务跳过（无法恢复）
        """
        import json

        # 第一步：扫描文件夹恢复 _tasks_state.json 中缺失的任务
        scanned = self._scan_and_recover_tasks()

        if not os.path.isfile(self._state_file):
            return scanned
        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception:
            logger.exception("加载任务状态文件失败")
            return 0

        tasks = state.get("tasks", [])
        restored = 0
        now = time.time()
        loop = asyncio.get_event_loop()

        # 恢复提前导出的 PDF 链接（只恢复文件仍存在的）
        for pid, info in (state.get("partial_results") or {}).items():
            if not isinstance(info, dict):
                continue
            out_path = info.get("output_path")
            if out_path and os.path.isfile(out_path):
                self._partial_results[pid] = info

        for entry in tasks:
            tid = entry.get("task_id")
            if not tid:
                continue
            status = entry.get("status")
            file_path = entry.get("file_path")
            source_name = entry.get("source_name") or "unknown"
            batch_id = entry.get("batch_id")
            output_format = entry.get("output_format")
            scheduled_at = entry.get("scheduled_at")
            # 读取任务级并发数（重试/恢复时保持，避免退化为单进程）
            task_concurrency = int(entry.get("task_concurrency", 1))

            # 恢复工作目录映射：优先用持久化的 work_dir，回退到 task_dirs.task_dir
            work_dir = entry.get("work_dir") or task_dirs.task_dir(tid)
            if os.path.isdir(work_dir):
                self._work_dirs[tid] = work_dir
            else:
                # 任务目录不存在，可能已被清理
                work_dir = None

            # 恢复元信息
            self._task_meta[tid] = {
                "file_path": file_path,
                "output_format": output_format,
                "source_name": source_name,
                "batch_id": batch_id,
                "task_concurrency": entry.get("task_concurrency", 1),
                "owner": entry.get("owner"),
            }

            if status == "done":
                # 已完成任务：恢复结果信息
                res = entry.get("result")
                if res and res.get("output_path") and os.path.isfile(res["output_path"]):
                    self._results[tid] = TaskResult(
                        task_id=tid,
                        source_name=source_name,
                        output_path=res["output_path"],
                        output_name=res.get("output_name", source_name),
                        output_format=res.get("output_format", "original"),
                        text=res.get("text", ""),
                        pages=res.get("pages", 0),
                    )
                    # 恢复到并发管理器（状态 done）
                    info = self.concurrency.register_scheduled(
                        tid, now, batch_id=batch_id, source_name=source_name,
                    )
                    # 直接修正为 done 状态
                    info.status = "done"
                    info.progress = entry.get("progress", 1)
                    info.total = entry.get("total", 1)
                    info.finished_at = entry.get("finished_at")
                    info.message = "完成"
                    restored += 1
                else:
                    # 结果文件丢失，标记为错误
                    info = self.concurrency.register_scheduled(
                        tid, now, batch_id=batch_id, source_name=source_name,
                    )
                    info.status = "error"
                    info.error = "结果文件已丢失"
                    info.message = "结果文件已丢失（可能被清理）"
                    info.finished_at = now
                    restored += 1

            elif status == "error":
                # 失败任务：恢复错误信息
                info = self.concurrency.register_scheduled(
                    tid, now, batch_id=batch_id, source_name=source_name,
                )
                info.status = "error"
                info.error = entry.get("error") or "处理失败"
                info.message = entry.get("message") or "处理失败"
                info.progress = entry.get("progress", 0)
                info.total = entry.get("total", 0)
                info.finished_at = entry.get("finished_at") or now
                restored += 1

            elif status == "running":
                # 运行中任务被中断：分两种情况处理
                # 1. 输出文件已存在 → 上次处理已完成但状态未更新（服务在 set_result
                #    前崩溃），直接标记为 done，避免重新处理返回0页结果
                # 2. 输出文件不存在 → 从断点继续处理（PDF 检测 pdf_pages 已有页数）
                # PDF 流程：file_path 已删除（拆分后），用 pdf_pages 判断是否可恢复
                # 图片流程：file_path 仍存在于 source_dir
                is_pdf = task_dirs.get_total_pages(tid) > 0
                file_exists = bool(file_path and os.path.isfile(file_path))
                if is_pdf or file_exists:
                    # 计算输出文件路径（使用 task_dirs.final_pdf_path 统一管理）
                    out_path = task_dirs.final_pdf_path(tid, source_name)

                    if out_path and os.path.isfile(out_path):
                        # 输出文件已存在：上次处理已完成，直接标记为 done
                        page_count = 0
                        try:
                            import fitz
                            doc = fitz.open(out_path)
                            page_count = doc.page_count
                            doc.close()
                        except Exception as e:
                            logger.warning("读取已有PDF页数失败: %s", e)
                        self._results[tid] = TaskResult(
                            task_id=tid,
                            source_name=source_name,
                            output_path=out_path,
                            output_name=os.path.basename(out_path),
                            output_format=output_format or "original",
                            pages=page_count,
                        )
                        info = self.concurrency.register_scheduled(
                            tid, now, batch_id=batch_id, source_name=source_name,
                        )
                        info.status = "done"
                        info.progress = page_count
                        info.total = max(page_count, 1)
                        info.finished_at = now
                        info.message = "完成（恢复）"
                        restored += 1
                        logger.info(
                            "恢复中断任务（输出已存在，直接标记完成）: %s (%d页)",
                            tid, page_count,
                        )
                        log_progress(
                            f"恢复已完成任务: {source_name} ({page_count} 页)"
                        )
                    else:
                        # 输出文件不存在：从断点继续处理
                        # file_path 对 PDF 已无效，但 run_task 不再使用它（保留兼容）
                        self._spawn_runner(
                            tid, file_path, source_name, output_format,
                            batch_id=batch_id, task_concurrency=task_concurrency,
                        )
                        restored += 1
                        logger.info("恢复中断任务（断点续传）: %s (%s)", tid, source_name)
                        log_progress(f"恢复中断任务: {source_name}")
                else:
                    logger.warning("跳过中断任务（源文件丢失）: %s", tid)
                    log_progress(f"跳过中断任务（源文件丢失）: {source_name}")

            elif status == "paused":
                # 暂停任务：保持 paused 状态，等待用户手动恢复
                # 不自动恢复，避免服务重启后无预期开始处理
                info = self.concurrency.register_scheduled(
                    tid, now, batch_id=batch_id, source_name=source_name,
                )
                info.status = "paused"
                info.pre_pause_status = entry.get("pre_pause_status") or "queued"
                info.scheduled_at = scheduled_at
                info.progress = entry.get("progress", 0)
                info.total = entry.get("total", 0)
                info.message = "已暂停（服务重启后保留）"
                restored += 1
                logger.info("恢复暂停状态: %s (%s)", tid, source_name)

            elif status in ("scheduled", "queued"):
                # 排队/预约任务：重新注册并启动 run_task 协程
                # PDF 流程：file_path 已删除，用 pdf_pages 判断是否可恢复
                # 图片流程：file_path 仍存在于 source_dir
                is_pdf = task_dirs.get_total_pages(tid) > 0
                file_exists = bool(file_path and os.path.isfile(file_path))
                if not (is_pdf or file_exists):
                    logger.warning("跳过 %s 任务（源文件丢失）: %s", status, tid)
                    log_progress(f"跳过 {status} 任务（源文件丢失）: {source_name}")
                    continue

                if status == "scheduled" and scheduled_at:
                    # 预约时间已过：转为立即执行
                    if scheduled_at <= now:
                        logger.info("预约任务 %s 已到点，转为立即执行", tid)
                        log_progress(f"预约任务到点转立即执行: {source_name}")
                        self.concurrency.register(tid, batch_id=batch_id, source_name=source_name)
                        # 启动 run_task（无 scheduled_at，立即排队执行）
                        self._spawn_runner(
                            tid, file_path, source_name, output_format,
                            batch_id=batch_id, task_concurrency=task_concurrency,
                        )
                    else:
                        # 预约时间未到：重新注册为 scheduled
                        self.concurrency.register_scheduled(
                            tid, scheduled_at, batch_id=batch_id, source_name=source_name,
                        )
                        # 启动 run_task 协程等待到点
                        self._spawn_runner(
                            tid, file_path, source_name, output_format,
                            scheduled_at=scheduled_at, batch_id=batch_id,
                            task_concurrency=task_concurrency,
                        )
                    restored += 1
                else:
                    # queued 任务：重新注册并排队
                    self.concurrency.register(tid, batch_id=batch_id, source_name=source_name)
                    self._spawn_runner(
                        tid, file_path, source_name, output_format,
                        batch_id=batch_id, task_concurrency=task_concurrency,
                    )
                    restored += 1

        logger.info("已恢复 %d 个任务状态", restored)
        if restored > 0:
            log_progress(f"服务重启：已恢复 {restored} 个任务")
        return restored + scanned

    def retry_task(self, task_id: str) -> bool:
        """重试中断/失败的任务。

        前提：源文件仍在磁盘上（work_dir 未被清理）。
        重试时重置状态为 queued，重新排队执行。

        返回是否成功提交重试。
        """
        info = self.concurrency.get_status(task_id)
        if info is None:
            return False
        # 只允许中断/失败状态的任务重试
        if info.status not in ("error",):
            return False
        meta = self._task_meta.get(task_id)
        if not meta:
            return False
        # PDF 流程：file_path 已删除，用 pdf_pages 判断是否可重试
        # 图片流程：file_path 仍存在于 source_dir
        is_pdf = task_dirs.get_total_pages(task_id) > 0
        file_path = meta.get("file_path", "")
        if not is_pdf:
            if not file_path or not os.path.isfile(file_path):
                return False
        # 从并发管理器移除旧记录（避免状态冲突）
        self.concurrency.remove(task_id)
        # 重新注册并启动任务
        self.concurrency.register(
            task_id,
            batch_id=meta.get("batch_id"),
            source_name=meta.get("source_name"),
        )
        # 保留原任务级并发数，避免重试后退化为单进程
        task_concurrency = int(meta.get("task_concurrency", 1))
        self._spawn_runner(
            task_id, file_path, meta.get("source_name", "unknown"),
            meta.get("output_format"), batch_id=meta.get("batch_id"),
            task_concurrency=task_concurrency,
        )
        source_name = meta.get("source_name", "unknown")
        logger.info("任务重试: %s -> %s (并发=%d)", source_name, task_id, task_concurrency)
        log_progress(f"任务重试: {source_name}")
        return True

    # ------------------------------------------------------------------
    # 半成品导出 / 提前导出 / 导入半成品
    # ------------------------------------------------------------------
    def finalize_partial(self, task_id: str) -> Tuple[str, int]:
        """把任务当前已处理的单页合并为最终 PDF（提前导出）。

        调用 task_processor.merge_partial 合并 ocr_pages 中已完成的页。
        不影响任务继续运行：
          - 不清理 ocr_pages 目录（任务可能还在写入新页）
          - 不修改任务状态

        返回 (输出文件路径, 已合并页数)。
        """
        # 检查任务工作目录是否存在
        tdir = task_dirs.task_dir(task_id)
        if not os.path.isdir(tdir):
            raise FileNotFoundError("任务工作目录不存在或已被清理")
        meta = self._task_meta.get(task_id, {})
        source_name = meta.get("source_name") or "unknown.pdf"
        # 复用 pipeline 的 render_dpi 配置
        dpi = 200
        if self._pipeline is not None:
            dpi = int(self._pipeline.cfg.get("render_dpi", 200))

        def _progress_cb(cur: int, total: int, msg: str) -> None:
            self.concurrency.update_progress(task_id, cur, total, msg)

        out_path, page_count = task_processor.merge_partial(
            task_id, source_name, dpi=dpi,
            progress_cb=_progress_cb,
        )
        logger.info(
            "任务 %s 提前导出 %d 页到 %s",
            task_id, page_count, os.path.basename(out_path),
        )
        log_progress(f"提前导出: {source_name} ({page_count} 页)")
        return out_path, page_count

    def _cleanup_partial_pdf(self, task_id: str) -> None:
        """任务完成后清理旧的 partial PDF 文件。

        任务运行中用户可能点击过"导出已识别部分"，在任务目录下生成了
        _{name}_partial_OCR.pdf（只有几页）。任务完成后这个文件是过期的，
        应清理掉避免与最终结果混淆。
        """
        meta = self._task_meta.get(task_id, {})
        source_name = meta.get("source_name", "")
        if not source_name:
            return
        # partial 文件路径由 task_dirs.partial_pdf_path 统一管理
        partial_path = task_dirs.partial_pdf_path(task_id, source_name)
        if os.path.isfile(partial_path):
            try:
                os.remove(partial_path)
                logger.info("已清理旧 partial PDF: %s", partial_path)
            except Exception as e:
                logger.warning("清理 partial PDF 失败: %s", e)

    def export_draft(
        self, task_id: str, include_source: bool = True,
    ) -> Tuple[str, dict]:
        """把任务半成品打包为 .ocr_draft ZIP 文件。

        半成品格式（ZIP 容器，保持向后兼容）：
          meta.json              元信息（源文件名、DPI、总页数、已完成页数）
          source.<ext>           源 PDF（include_source=True 时打包）
          pages/ocr_XXXX.json    已处理单页 OCR 结果（精简模式，体积小）
          pages/page_XXXX.pdf    已处理单页 PDF（仅 include_source=False 时打包）

        新目录结构（ocr_pages/）导出时转换为 ZIP 中的旧命名（ocr_XXXX.json），
        保持半成品 ZIP 格式向后兼容。

        返回 (ZIP 文件路径, meta 信息字典)。
        """
        work_dir = self._work_dirs.get(task_id) or task_dirs.task_dir(task_id)
        if not os.path.isdir(work_dir):
            raise FileNotFoundError("任务工作目录不存在或已被清理")
        meta = self._task_meta.get(task_id, {})
        file_path = meta.get("file_path")
        source_name = meta.get("source_name") or os.path.basename(
            file_path or "unknown.pdf"
        )

        # 总页数：优先从 pdf_pages 获取（PDF 拆分后的页数）
        # 对于图片等非 PDF 文件，回退到 0
        total_pages = task_dirs.get_total_pages(task_id)

        # 扫描 ocr_pages 目录中已完成的页（同时存在 page_XXXX.pdf 和 page_XXXX.json）
        ocr_pages_d = task_dirs.ocr_pages_dir(task_id)
        completed_pages = task_dirs.get_completed_pages(task_id)
        pages_pdf_size = 0
        ocr_json_size = 0
        if os.path.isdir(ocr_pages_d):
            for name in os.listdir(ocr_pages_d):
                full = os.path.join(ocr_pages_d, name)
                if not os.path.isfile(full):
                    continue
                if name.startswith("page_") and name.endswith(".pdf"):
                    pages_pdf_size += os.path.getsize(full)
                elif name.startswith("page_") and name.endswith(".json"):
                    ocr_json_size += os.path.getsize(full)

        # 源文件路径：PDF 拆分后原文件已删除，需要时从 pdf_pages 重建
        source_path = None
        if file_path and os.path.isfile(file_path):
            source_path = file_path
        elif include_source and total_pages > 0:
            # PDF 已拆分：从 pdf_pages 重建源 PDF 到临时文件
            source_path = self._reconstruct_source_pdf(task_id, source_name)

        dpi = 200
        if self._pipeline is not None:
            dpi = int(self._pipeline.cfg.get("render_dpi", 200))

        # slim_mode：精简模式标记，导入时据此判断是否需要重新生成单页 PDF
        slim_mode = include_source

        meta_data = {
            "version": 2,
            "source_name": source_name,
            "render_dpi": dpi,
            "total_pages": total_pages,
            "completed_pages": completed_pages,
            "exported_at": time.time(),
            "include_source": include_source,
            "slim_mode": slim_mode,
        }

        # 打包 ZIP
        import zipfile
        import json as _json
        base_name = os.path.splitext(source_name)[0]
        out_name = f"_{base_name}_draft.ocr_draft"
        out_path = os.path.join(work_dir, out_name)
        tmp_path = out_path + ".tmp"

        with zipfile.ZipFile(
            tmp_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9,
        ) as zf:
            zf.writestr(
                "meta.json",
                _json.dumps(meta_data, ensure_ascii=False, indent=2),
            )
            if include_source and source_path and os.path.isfile(source_path):
                ext = os.path.splitext(source_name)[1] or ".pdf"
                zf.write(source_path, f"source{ext}")
            if os.path.isdir(ocr_pages_d):
                if slim_mode:
                    # 精简模式：只打包 OCR JSON（转换为旧命名 ocr_XXXX.json）
                    for page_no in task_dirs.list_ocr_pages(task_id):
                        json_path = task_dirs.ocr_json_path(task_id, page_no)
                        if os.path.isfile(json_path):
                            # 旧格式：ocr_XXXX.json
                            zf.write(json_path, f"pages/ocr_{page_no:04d}.json")
                else:
                    # 完整模式：打包所有 ocr_pages 文件（含 PDF 和 JSON）
                    for page_no in task_dirs.list_ocr_pages(task_id):
                        pdf_path = task_dirs.ocr_pdf_path(task_id, page_no)
                        json_path = task_dirs.ocr_json_path(task_id, page_no)
                        if os.path.isfile(pdf_path):
                            zf.write(pdf_path, f"pages/page_{page_no:04d}.pdf")
                        if os.path.isfile(json_path):
                            # 旧格式：ocr_XXXX.json
                            zf.write(json_path, f"pages/ocr_{page_no:04d}.json")
        os.replace(tmp_path, out_path)

        # 清理重建的临时源文件
        if (source_path and file_path and source_path != file_path
                and os.path.isfile(source_path)):
            try:
                os.remove(source_path)
            except Exception:
                pass

        # 日志
        final_size = os.path.getsize(out_path)
        final_mb = final_size / 1024 / 1024
        pdf_mb = pages_pdf_size / 1024 / 1024
        json_mb = ocr_json_size / 1024 / 1024
        logger.info(
            "任务 %s 导出半成品: %s (已完成 %d/%d 页, 精简=%s, "
            "单页PDF %.1fMB + JSON %.1fMB → ZIP %.1fMB)",
            task_id, out_name, completed_pages, total_pages, slim_mode,
            pdf_mb, json_mb, final_mb,
        )
        log_progress(
            f"导出半成品: {source_name} (已完成 {completed_pages}/{total_pages} 页, "
            f"体积 {final_mb:.1f}MB)"
        )
        return out_path, meta_data

    def _reconstruct_source_pdf(self, task_id: str, source_name: str) -> Optional[str]:
        """从 pdf_pages 重建源 PDF（用于导出半成品时获取源文件）。

        PDF 拆分后原文件已删除，导出半成品需要源文件时从此重建。
        返回重建的临时 PDF 路径，失败返回 None。
        """
        import fitz
        pdf_pages = task_dirs.list_pdf_pages(task_id)
        if not pdf_pages:
            return None
        # 重建到 source_dir 下
        out_path = os.path.join(task_dirs.source_dir(task_id), source_name)
        try:
            doc = fitz.open()
            for page_no in pdf_pages:
                page_path = task_dirs.page_pdf_path(task_id, page_no)
                if os.path.isfile(page_path):
                    src = fitz.open(page_path)
                    doc.insert_pdf(src)
                    src.close()
            doc.save(out_path, garbage=4, deflate=True)
            doc.close()
            logger.info("从 pdf_pages 重建源 PDF: %s (%d 页)", source_name, len(pdf_pages))
            return out_path
        except Exception as e:
            logger.warning("重建源 PDF 失败: %s", e)
            return None

    def import_draft(
        self,
        draft_path: str,
        output_format: Optional[str] = None,
        batch_id: Optional[str] = None,
        task_concurrency: int = 1,
    ) -> Tuple[str, str, int, int]:
        """导入半成品 ZIP，解压到新任务工作目录，启动断点续传。

        前提：半成品必须包含源文件（include_source=True 导出的）。
        导入后 task_processor 会检测 ocr_pages 已有页数，自动跳过已处理页。

        半成品 ZIP 格式（向后兼容）：
          pages/page_XXXX.pdf  → ocr_pages/page_XXXX.pdf
          pages/ocr_XXXX.json  → ocr_pages/page_XXXX.json（重命名）

        返回 (task_id, source_name, completed_pages, total_pages)。
        """
        import zipfile
        import json as _json
        import re as _re

        if not os.path.isfile(draft_path):
            raise FileNotFoundError(f"半成品文件不存在: {draft_path}")

        task_id = self.new_task_id()
        # 创建任务目录结构（source/、pdf_pages/、ocr_pages/）
        task_dirs.create_task_dirs(task_id)
        work_dir = task_dirs.task_dir(task_id)
        self._work_dirs[task_id] = work_dir

        # 解压到工作目录（临时解压，后续迁移到各子目录）
        try:
            with zipfile.ZipFile(draft_path, "r") as zf:
                zf.extractall(work_dir)
        except zipfile.BadZipFile as e:
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass
            self._work_dirs.pop(task_id, None)
            raise ValueError(f"半成品文件损坏或格式错误: {e}")

        # 读取 meta.json
        meta_path = os.path.join(work_dir, "meta.json")
        if not os.path.isfile(meta_path):
            shutil.rmtree(work_dir, ignore_errors=True)
            self._work_dirs.pop(task_id, None)
            raise ValueError("无效的半成品文件：缺少 meta.json")
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_data = _json.load(f)
        except Exception as e:
            shutil.rmtree(work_dir, ignore_errors=True)
            self._work_dirs.pop(task_id, None)
            raise ValueError(f"meta.json 解析失败: {e}")

        source_name = meta_data.get("source_name", "unknown.pdf")
        completed_pages = int(meta_data.get("completed_pages", 0))
        total_pages = int(meta_data.get("total_pages", 0))

        # 查找源文件（导出时存为 source.<ext>）
        source_ext = os.path.splitext(source_name)[1] or ".pdf"
        source_in_draft = os.path.join(work_dir, f"source{source_ext}")
        if not os.path.isfile(source_in_draft):
            shutil.rmtree(work_dir, ignore_errors=True)
            self._work_dirs.pop(task_id, None)
            raise FileNotFoundError(
                f"半成品缺少源文件 source{source_ext}。"
                "仅含源文件的半成品（include_source=true）才能导入续作。"
            )

        # 移动源文件到 source_dir
        src_dir = task_dirs.source_dir(task_id)
        file_path = os.path.join(src_dir, source_name)
        if source_in_draft != file_path:
            os.replace(source_in_draft, file_path)

        # 迁移 pages/ 内容到 ocr_pages/
        # 旧格式：pages/page_XXXX.pdf, pages/ocr_XXXX.json
        # 新格式：ocr_pages/page_XXXX.pdf, ocr_pages/page_XXXX.json
        old_pages_dir = os.path.join(work_dir, "pages")
        ocr_pages_d = task_dirs.ocr_pages_dir(task_id)
        if os.path.isdir(old_pages_dir):
            # 旧命名 ocr_XXXX.json 的正则
            ocr_json_re = _re.compile(r"ocr_(\d+)\.json$")
            for name in os.listdir(old_pages_dir):
                full = os.path.join(old_pages_dir, name)
                if not os.path.isfile(full):
                    continue
                if name.startswith("page_") and name.endswith(".pdf"):
                    # page_XXXX.pdf → ocr_pages/page_XXXX.pdf
                    shutil.move(full, os.path.join(ocr_pages_d, name))
                elif ocr_json_re.match(name):
                    # ocr_XXXX.json → ocr_pages/page_XXXX.json（重命名）
                    m = ocr_json_re.match(name)
                    page_no = int(m.group(1))
                    new_name = task_dirs.page_json_name(page_no)
                    shutil.move(full, os.path.join(ocr_pages_d, new_name))
            # 清理空的 pages/ 目录
            try:
                shutil.rmtree(old_pages_dir, ignore_errors=True)
            except Exception:
                pass

        # 精简模式恢复：检测是否缺少单页 PDF（slim_mode=true 导出的半成品）
        # 若缺少单页 PDF 但有 OCR JSON + 源文件，从源文件重新生成单页 PDF
        slim_mode = bool(meta_data.get("slim_mode", False))
        if slim_mode and os.path.isdir(ocr_pages_d):
            missing_pdfs = self._check_missing_page_pdfs(ocr_pages_d, completed_pages)
            if missing_pdfs:
                logger.info(
                    "精简模式恢复：从源文件重新生成 %d 个单页 PDF",
                    len(missing_pdfs),
                )
                log_progress(
                    f"导入半成品: 正在从源文件恢复 {len(missing_pdfs)} 个单页 PDF..."
                )
                self._regenerate_page_pdfs(file_path, ocr_pages_d, missing_pdfs)

        # 清理 meta.json
        try:
            os.remove(meta_path)
        except Exception:
            pass

        # 如果是 PDF：拆分到 pdf_pages/（拆分后原 PDF 被删除）
        if source_name.lower().endswith(".pdf"):
            try:
                task_processor.split_pdf_to_pages(file_path, task_id)
                logger.info("导入半成品 PDF 拆分完成: %s -> 任务 %s", source_name, task_id)
            except Exception as e:
                logger.exception("导入半成品 PDF 拆分失败: %s -> %s (%s)", source_name, task_id, e)

        # 记录元信息（file_path 对 PDF 已无效，保留用于兼容）
        self._task_meta[task_id] = {
            "file_path": file_path,
            "output_format": output_format,
            "source_name": source_name,
            "batch_id": batch_id,
            "task_concurrency": task_concurrency,
            "owner": self._task_meta.get(task_id, {}).get("owner"),
        }
        self._save_task_meta(task_id)
        # 注册为 queued，启动 run_task（task_processor 检测 ocr_pages 自动断点续传）
        self.concurrency.register(
            task_id, batch_id=batch_id, source_name=source_name,
        )
        self._save_state()
        self._spawn_runner(
            task_id, file_path, source_name, output_format,
            batch_id=batch_id, task_concurrency=task_concurrency,
        )

        logger.info(
            "导入半成品: %s -> 任务 %s (已完成 %d/%d 页, 断点续传)",
            source_name, task_id, completed_pages, total_pages,
        )
        log_progress(
            f"导入半成品: {source_name} (从第 {completed_pages + 1} 页继续)"
        )
        return task_id, source_name, completed_pages, total_pages

    @staticmethod
    def _check_missing_page_pdfs(
        pages_dir: str, expected_count: int,
    ) -> List[int]:
        """检查 pages_dir 中缺少哪些单页 PDF。

        返回缺少的页号列表（1-based）。
        """
        missing = []
        for page_no in range(1, expected_count + 1):
            page_file = os.path.join(pages_dir, f"page_{page_no:04d}.pdf")
            if not os.path.isfile(page_file):
                missing.append(page_no)
        return missing

    @staticmethod
    def _regenerate_page_pdfs(
        source_path: str, pages_dir: str, page_nos: List[int],
    ) -> None:
        """从源 PDF 重新生成单页 PDF。

        精简模式导入时调用：半成品只含 OCR JSON，不含单页 PDF。
        从源 PDF 复制对应页到独立文件，保留原始压缩数据。

        参数:
            source_path: 源 PDF 路径
            pages_dir: 单页文件目录
            page_nos: 需要生成的页号列表（1-based）
        """
        import fitz

        src_doc = fitz.open(source_path)
        try:
            for page_no in page_nos:
                src_page_idx = page_no - 1
                if src_page_idx >= src_doc.page_count:
                    logger.warning(
                        "源 PDF 页号越界: %d >= %d",
                        src_page_idx, src_doc.page_count,
                    )
                    continue
                page_file = os.path.join(pages_dir, f"page_{page_no:04d}.pdf")
                doc = fitz.open()
                try:
                    doc.insert_pdf(
                        src_doc,
                        from_page=src_page_idx,
                        to_page=src_page_idx,
                    )
                    doc.save(page_file, garbage=3, deflate=True)
                finally:
                    doc.close()
        finally:
            src_doc.close()
