"""并发控制：限制同时处理的 OCR 任务数。

PaddleOCR 模型非线程安全，多用户同时识别会崩溃或变慢。
用 asyncio.Semaphore 限制并发，超出排队等待并提示位置与原因。

设计：
  - register(task_id)  注册任务（初始 queued）
  - acquire(task_id)   异步获取槽位，等待期间任务状态保持 queued
                        并实时计算排队位置；获得槽位后置为 running
  - release(task_id)   释放槽位，唤醒下一个排队任务
  - get_queue_reason   返回排队的详细原因（前面有几个任务、预计等待）

队列顺序：按注册先后 FIFO。
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TaskInfo:
    """单个任务运行时信息。"""
    task_id: str
    # "scheduled" | "queued" | "running" | "paused" | "done" | "error"
    status: str
    progress: int = 0
    total: int = 0
    message: str = ""
    result_path: Optional[str] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    # 分配到的实例池槽位号（running 时非空，对应 PaddleLocalProvider 的 slot）
    # None 表示尚未分配（scheduled / queued / 已释放）
    slot: Optional[int] = None
    # 多槽位模式：任务可同时占用多个槽位（页级并行处理）
    # 单槽位模式（默认）下为空列表；多槽位模式下存所有已获取的槽位号
    # release 时释放全部 slots，slot 字段保留为 slots[0] 兼容旧逻辑
    slots: List[int] = field(default_factory=list)
    # 本任务请求的并发数（1=单槽位默认，>1=页级并行）
    task_concurrency: int = 1
    # 预约执行时间（Unix 时间戳，None 表示立即执行）
    # 用户可设置未来 7 天内的某个时间点；到点前任务保持 scheduled 状态，不占槽位
    scheduled_at: Optional[float] = None
    # 批次 ID：同一次上传的所有文件共享同一 batch_id
    # 前端按 batch_id 分组展示为"任务01/任务02"，展开后看到具体文件
    batch_id: Optional[str] = None
    # 源文件名（便于在批次内展示）
    source_name: Optional[str] = None
    # 暂停前的状态（用于恢复时回到正确流程）
    # running 暂停后恢复时走 queued → acquire 重新获取槽位，pipeline 自动断点续传
    # queued/scheduled 暂停后恢复时直接回到原状态
    pre_pause_status: Optional[str] = None

    def to_dict(self) -> dict:
        """转为可 JSON 序列化的字典。"""
        return {
            "task_id": self.task_id,
            "status": self.status,
            "progress": self.progress,
            "total": self.total,
            "message": self.message,
            "result_path": self.result_path,
            "error": self.error,
            "created_at": round(self.created_at, 3),
            "started_at": round(self.started_at, 3) if self.started_at else None,
            "finished_at": round(self.finished_at, 3) if self.finished_at else None,
            "scheduled_at": round(self.scheduled_at, 3) if self.scheduled_at else None,
            "batch_id": self.batch_id,
            "source_name": self.source_name,
        }


class ConcurrencyManager:
    """并发控制器：信号量 + FIFO 排队。"""

    def __init__(self, max_concurrent: int = 3) -> None:
        if max_concurrent < 1:
            max_concurrent = 1
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        # 全部任务表（含已完成，由 tasks 模块负责清理）
        self._tasks: Dict[str, TaskInfo] = {}
        # 排队任务的 FIFO 顺序（仅含 status=queued 的任务）
        self._queue_order: List[str] = []
        # 空闲槽位栈：[max-1, max-2, ..., 0]，LIFO 复用。
        # 每个槽位对应一个独立的 PaddleOCR 实例（由 init_ocr_pool 预创建），
        # 任务获取信号量后从此栈弹出一个槽位，完成后压回。
        self._free_slots: List[int] = list(range(max_concurrent - 1, -1, -1))

    # ------------------------------------------------------------------
    # 任务注册 / 查询
    # ------------------------------------------------------------------
    def register(
        self, task_id: str, batch_id: Optional[str] = None, source_name: Optional[str] = None
    ) -> TaskInfo:
        """注册新任务，初始状态 queued，加入排队队列。"""
        info = TaskInfo(
            task_id=task_id, status="queued", message="排队中",
            batch_id=batch_id, source_name=source_name,
        )
        self._tasks[task_id] = info
        self._queue_order.append(task_id)
        return info

    def register_scheduled(
        self, task_id: str, scheduled_at: float,
        batch_id: Optional[str] = None, source_name: Optional[str] = None,
    ) -> TaskInfo:
        """注册预约任务，初始状态 scheduled，不加入排队队列。

        预约时间到达前不占槽位，到点后由调度器调用 activate 转为 queued。
        """
        info = TaskInfo(
            task_id=task_id,
            status="scheduled",
            scheduled_at=scheduled_at,
            batch_id=batch_id,
            source_name=source_name,
            message=f"预约于 {time.strftime('%Y-%m-%d %H:%M', time.localtime(scheduled_at))} 执行",
        )
        self._tasks[task_id] = info
        return info

    def activate(self, task_id: str) -> bool:
        """把 scheduled 状态的任务转为 queued，加入排队队列。

        由调度器在预约时间到达时调用。返回是否成功激活。
        """
        info = self._tasks.get(task_id)
        if info is None or info.status != "scheduled":
            return False
        info.status = "queued"
        info.message = "预约时间到达，排队中"
        self._queue_order.append(task_id)
        return True

    def get_status(self, task_id: str) -> Optional[TaskInfo]:
        return self._tasks.get(task_id)

    def get_all_status(self) -> dict:
        """返回整体并发状态。"""
        running = sum(1 for t in self._tasks.values() if t.status == "running")
        queued = sum(1 for t in self._tasks.values() if t.status == "queued")
        scheduled = sum(1 for t in self._tasks.values() if t.status == "scheduled")
        return {
            "max_concurrent": self.max_concurrent,
            "running": running,
            "queued": queued,
            "scheduled": scheduled,
            "available": max(0, self.max_concurrent - running),
        }

    def get_queue_position(self, task_id: str) -> int:
        """返回该任务的排队位置（1=下一个执行），不在排队返回 0。"""
        info = self._tasks.get(task_id)
        if info is None or info.status != "queued":
            return 0
        pos = 0
        for tid in self._queue_order:
            if tid == task_id:
                return pos + 1
            t = self._tasks.get(tid)
            if t is not None and t.status == "queued":
                pos += 1
        return pos + 1

    def get_queue_reason(self, task_id: str) -> str:
        """返回该任务排队/等待的详细原因（供前端展示）。"""
        info = self._tasks.get(task_id)
        if info is None:
            return "任务不存在"
        if info.status == "running":
            return "正在处理中"
        if info.status in ("done", "error"):
            return "任务已结束"
        if info.status == "scheduled":
            # 预约任务：显示预约时间
            if info.scheduled_at:
                ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(info.scheduled_at))
                now = time.time()
                if info.scheduled_at > now:
                    remain = info.scheduled_at - now
                    if remain > 86400:
                        remain_str = f"{int(remain / 86400)} 天后"
                    elif remain > 3600:
                        remain_str = f"{int(remain / 3600)} 小时后"
                    else:
                        remain_str = f"{int(remain / 60)} 分钟后"
                    return f"预约于 {ts} 执行（{remain_str}）"
            return "预约任务，等待到点执行"
        # queued
        status = self.get_all_status()
        pos = self.get_queue_position(task_id)
        if status["running"] >= self.max_concurrent:
            return (
                f"当前已有 {status['running']} 个任务在处理（并发上限 "
                f"{self.max_concurrent}），本任务排在第 {pos} 位，"
                f"前面还有 {pos - 1} 个任务等待处理。"
            )
        return f"排队中，位置 {pos}，即将开始处理。"

    # ------------------------------------------------------------------
    # 槽位获取 / 释放
    # ------------------------------------------------------------------
    async def acquire(self, task_id: str) -> Tuple[int, bool]:
        """获取处理槽位并分配实例池槽位号。

        返回 (slot, immediate):
          - slot: 分配到的实例池槽位号（0~max_concurrent-1），传给 PaddleOCR
          - immediate: True=无需等待立即开始；False=曾排队等待过

        等待期间任务状态保持 queued，前端可轮询 get_queue_position。

        取消竞态防护：
          await semaphore.acquire() 返回后、设置 info.slots 之前若被 cancel，
          许可已被信号量扣减但 info.slots 为空，后续 release 不会归还许可。
          用 try/except asyncio.CancelledError 包裹，取消时立即归还许可。
        """
        info = self._tasks.get(task_id)
        if info is None:
            info = self.register(task_id)
        # 判断是否需要排队：当前运行数已达上限则需等待
        running = sum(1 for t in self._tasks.values() if t.status == "running")
        immediate = running < self.max_concurrent
        if not immediate:
            info.message = self.get_queue_reason(task_id)
        # 阻塞等待信号量（排队任务在此挂起）
        await self._semaphore.acquire()
        # 获取信号量后、设置 slots 之前的窗口期需防护取消竞态
        try:
            # 获得信号量，分配一个实例池槽位
            slot = self._free_slots.pop() if self._free_slots else 0
            info.slot = slot
            info.slots = [slot]
            info.status = "running"
            info.started_at = time.time()
            info.message = "开始处理"
            if task_id in self._queue_order:
                self._queue_order.remove(task_id)
            return slot, immediate
        except asyncio.CancelledError:
            # 取消竞态：许可已获取但任务被取消，立即归还
            self._semaphore.release()
            raise

    async def acquire_many(self, task_id: str, count: int) -> Tuple[List[int], bool]:
        """获取多个处理槽位（用于页级并行处理）。

        循环 acquire 信号量 count 次，每次弹出一个槽位号加入 slots 列表。
        等待期间更新进度消息（"等待 N 个槽位，已获取 M/N"）。

        返回 (slots, immediate):
          - slots: 分配到的槽位号列表，长度=count
          - immediate: True=无需等待立即获取全部；False=曾排队等待

        取消竞态防护：
          循环中途被 cancel 时，已获取的许可和槽位需立即归还，避免泄漏。
        """
        if count <= 1:
            slot, immediate = await self.acquire(task_id)
            return [slot], immediate

        info = self._tasks.get(task_id)
        if info is None:
            info = self.register(task_id)
        info.task_concurrency = count

        immediate = True
        acquired: List[int] = []
        try:
            for i in range(count):
                running = sum(1 for t in self._tasks.values() if t.status == "running")
                free = self.max_concurrent - running
                if free < count - i:
                    immediate = False
                    info.message = f"等待 {count} 个槽位，已获取 {i}/{count}"
                await self._semaphore.acquire()
                slot = self._free_slots.pop() if self._free_slots else 0
                acquired.append(slot)

            info.slots = acquired
            info.slot = acquired[0] if acquired else None
            info.status = "running"
            info.started_at = time.time()
            info.message = f"开始处理（{count} 进程并行）"
            if task_id in self._queue_order:
                self._queue_order.remove(task_id)
            return acquired, immediate
        except asyncio.CancelledError:
            # 取消竞态：归还已获取的全部许可和槽位
            for s in acquired:
                self._free_slots.append(s)
            for _ in acquired:
                try:
                    self._semaphore.release()
                except ValueError:
                    pass
            raise

    def release(
        self,
        task_id: str,
        status: str = "done",
        error: Optional[str] = None,
        message: str = "",
    ) -> None:
        """释放槽位，回收实例池槽位号，唤醒下一个排队任务。

        注意：只有获得过槽位（信号量）的任务才释放信号量。
        acquire 在等待信号量时被取消的任务，slot 为 None，不应释放信号量，
        否则会导致信号量计数超过 max_concurrent，引发并发超标和实例冲突。
        """
        info = self._tasks.get(task_id)
        if info is None:
            return
        # 释放所有已获取的槽位（兼容单slot和多slot模式）
        # 多slot模式：info.slots 有多个元素；单slot模式：info.slots=[slot] 或 []
        slots_to_release = list(info.slots) if info.slots else (
            [info.slot] if info.slot is not None else []
        )
        had_slot = len(slots_to_release) > 0
        # 回收实例池槽位号到空闲栈
        for s in slots_to_release:
            self._free_slots.append(s)
        info.slot = None
        info.slots = []
        info.status = status
        info.finished_at = time.time()
        if error:
            info.error = error
            info.message = message or f"失败: {error}"
        else:
            info.message = message or ("完成" if status == "done" else "已结束")
        if task_id in self._queue_order:
            self._queue_order.remove(task_id)
        # 每个已获取的槽位释放一个信号量许可
        for _ in slots_to_release:
            try:
                self._semaphore.release()
            except ValueError:
                # 信号量已满，忽略（避免重复 release 崩溃）
                pass

    # ------------------------------------------------------------------
    # 进度 / 结果更新
    # ------------------------------------------------------------------
    def update_progress(
        self, task_id: str, progress: int, total: int, message: str = ""
    ) -> None:
        info = self._tasks.get(task_id)
        if info is None:
            return
        info.progress = progress
        info.total = total
        if message:
            info.message = message

    def set_result(self, task_id: str, result_path: str) -> None:
        info = self._tasks.get(task_id)
        if info is None:
            return
        info.result_path = result_path
        info.status = "done"
        if info.total <= 0:
            info.total = 1
        info.progress = info.total
        info.finished_at = time.time()
        info.message = "完成"

    def remove(self, task_id: str) -> None:
        """从任务表移除（用于清理已完成任务）。运行中的任务不会被移除。"""
        info = self._tasks.get(task_id)
        if info is None:
            return
        if info.status == "running":
            return
        self._tasks.pop(task_id, None)
        if task_id in self._queue_order:
            self._queue_order.remove(task_id)

    # ------------------------------------------------------------------
    # 暂停 / 恢复
    # ------------------------------------------------------------------
    def pause(self, task_id: str) -> Tuple[bool, str, List[int]]:
        """暂停任务，释放槽位（让其他任务推进）。

        返回 (成功?, 原因, 需 kill 的槽位列表)。
        - running 状态：释放全部槽位 + 置 paused，记录 pre_pause_status=running
          恢复时走 queued → acquire，pipeline 检测 pages_dir 自动断点续传
          返回的 slots 用于上层调用 _kill_ocr_slots 切断所有子进程推理
        - queued 状态：从队列移除 + 置 paused，不释放信号量（未获取过）
        - scheduled 状态：置 paused，唤醒等待协程让它退出
        - done/error/paused：不允许暂停
        """
        info = self._tasks.get(task_id)
        if info is None:
            return False, "任务不存在", []
        if info.status in ("done", "error", "paused"):
            return False, f"任务状态 {info.status} 不可暂停", []

        pre = info.status
        info.pre_pause_status = pre
        info.status = "paused"
        info.message = "已暂停"
        # finished_at 不设置（暂停不算结束）

        slots_to_kill: List[int] = []
        if pre == "running":
            # 释放所有已获取的槽位（与 release 一致的回收逻辑）
            # 多slot模式：info.slots 有多个元素；单slot模式：info.slots=[slot] 或 []
            # 旧 Bug：只处理 info.slot（= slots[0]），多槽位任务会泄漏 N-1 个信号量许可
            slots_to_release = list(info.slots) if info.slots else (
                [info.slot] if info.slot is not None else []
            )
            if slots_to_release:
                slots_to_kill = list(slots_to_release)
                for s in slots_to_release:
                    self._free_slots.append(s)
                info.slot = None
                info.slots = []
            if task_id in self._queue_order:
                self._queue_order.remove(task_id)
            # 每个已获取的槽位释放一个信号量许可
            for _ in slots_to_release:
                try:
                    self._semaphore.release()
                except ValueError:
                    pass
        elif pre == "queued":
            # 排队中：从队列移除即可，未获取信号量无需释放
            if task_id in self._queue_order:
                self._queue_order.remove(task_id)
        # scheduled 状态的唤醒由 tasks.py 处理（通过 _scheduled_waiters）
        return True, "已暂停", slots_to_kill

    def resume(self, task_id: str) -> Tuple[bool, str, str]:
        """恢复暂停的任务。

        返回 (成功?, 原因, 恢复后状态)。
        - pre_pause=running → 置 queued 重新排队（pipeline 自动断点续传）
        - pre_pause=queued → 置 queued 重新排队
        - pre_pause=scheduled → 置 scheduled（保持原预约时间）

        恢复后状态由上层 _spawn_runner / register_scheduled 接管。
        """
        info = self._tasks.get(task_id)
        if info is None:
            return False, "任务不存在", ""
        if info.status != "paused":
            return False, f"任务状态 {info.status} 不可恢复（仅 paused 可恢复）", ""

        pre = info.pre_pause_status or "queued"
        info.pre_pause_status = None

        if pre == "scheduled":
            # 恢复为 scheduled 状态，等待到点执行
            info.status = "scheduled"
            info.message = "已恢复，等待预约时间"
            return True, "已恢复", "scheduled"
        else:
            # running / queued 都重新排队（running 走断点续传）
            info.status = "queued"
            info.message = "已恢复，排队中" if pre == "queued" else "已恢复，排队中（断点续传）"
            if task_id not in self._queue_order:
                self._queue_order.append(task_id)
            return True, "已恢复", "queued"

    # ------------------------------------------------------------------
    # 动态调整并发上限
    # ------------------------------------------------------------------
    def set_max_concurrent(self, n: int) -> None:
        """调整最大并发数，重建信号量和空闲槽位列表。

        关键：asyncio.Semaphore 的容量无法直接修改，必须重建。
        重建时需保留当前 running 任务已占用的许可数，避免新信号量容量超标
        或已占用许可丢失。

        旧 Bug：只更新 max_concurrent 字段，不重建信号量。导致：
          - /api/limits 报告 available = max_concurrent - running（基于新字段）
          - 但信号量容量仍是启动时的旧值
          - 字段 > 信号量容量时，available 虚高，任务排队不启动
          - 字段 < 信号量容量时，available 虚低，并发不足
        """
        if n < 1:
            n = 1
        old_max = self.max_concurrent
        self.max_concurrent = n

        # 统计当前 running 任务已占用的槽位数
        used_slots = 0
        for t in self._tasks.values():
            if t.status == "running":
                used_slots += len(t.slots) if t.slots else (1 if t.slot is not None else 0)

        # 重建信号量：新容量 = n，已用 = used_slots，可用 = n - used_slots
        # asyncio.Semaphore(initial) 初始值即为可用许可数
        available = max(0, n - used_slots)
        self._semaphore = asyncio.Semaphore(available)

        # 重建空闲槽位列表：从 running 任务的 slots 中排除已占用的
        used_slot_nums = set()
        for t in self._tasks.values():
            if t.status == "running":
                if t.slots:
                    used_slot_nums.update(t.slots)
                elif t.slot is not None:
                    used_slot_nums.add(t.slot)
        # 新的空闲槽位：0~n-1 中未被 running 任务占用的
        # LIFO 顺序（与 __init__ 一致）
        self._free_slots = [i for i in range(n - 1, -1, -1) if i not in used_slot_nums]

        logger.info(
            "并发上限调整: %d -> %d（running 占用 %d 槽位，可用 %d）",
            old_max, n, used_slots, available,
        )
