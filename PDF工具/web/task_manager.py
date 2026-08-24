"""后台任务管理：线程 + 进度队列 + 取消标志。

统一管理六大功能的异步执行：
- 在独立线程跑 engine（engine 是同步阻塞的，带 progress_callback/cancel_check）
- progress_callback 把进度推入任务专属队列，供 SSE 端点读取
- cancel_check 读取取消标志，engine 下次 _check_cancelled 时抛 CancelledException
- 取消时清理半成品输出文件

engine_call 闭包签名: engine_call(cancel_flag, q) -> result_path
内部应这样构造 engine：
    progress_callback = lambda c, t, m: q.put({"type":"progress","current":c,"total":t,"message":m})
    cancel_check = lambda: cancel_flag.is_set()
"""
import os
import uuid
import queue
import shutil
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, List


@dataclass
class Task:
    id: str
    kind: str                                  # merge/split/convert/compose/append/insert
    status: str = "running"                    # running/done/error/cancelled
    q: "queue.Queue" = field(default_factory=queue.Queue)
    cancel_flag: threading.Event = field(default_factory=threading.Event)
    result: Any = None                         # engine 返回的输出路径（或路径列表）
    error: str = ""
    output_path: str = ""                      # 主输出路径（打开目录 / 下载用）
    output_is_dir: bool = False                # 输出是否为目录（拆分模式输出多文件）
    cleanup_paths: List[str] = field(default_factory=list)  # 取消时清理的半成品
    temp_dir: str = ""                          # 远程模式临时目录，清理时整体删除
    input_paths: List[str] = field(default_factory=list)    # 远程模式上传的输入文件，清理时删除
    owner: str = ""                             # 任务所有者（用户名），用于多用户隔离
    thread: threading.Thread = None


def make_callbacks(cancel_flag: threading.Event, q: "queue.Queue"):
    """构造 progress_callback 与 cancel_check，桥接 engine 与任务队列。"""
    def progress_callback(current, total, message):
        q.put({"type": "progress", "current": current, "total": total, "message": message})
    def cancel_check():
        return cancel_flag.is_set()
    return progress_callback, cancel_check


class TaskManager:
    """全局任务注册表 + 后台线程调度。"""

    def __init__(self):
        self._tasks: dict = {}
        self._lock = threading.Lock()

    def start(self, kind: str, engine_call: Callable, *,
              output_path: str = "", output_is_dir: bool = False,
              cleanup_paths: list = None, temp_dir: str = "",
              input_paths: list = None, owner: str = "") -> str:
        """启动一个后台任务，返回 task_id。

        owner 为启动该任务的用户名，用于后续越权校验。
        """
        task_id = uuid.uuid4().hex[:12]
        task = Task(id=task_id, kind=kind)
        task.output_path = output_path
        task.output_is_dir = output_is_dir
        task.cleanup_paths = cleanup_paths or []
        task.temp_dir = temp_dir
        task.input_paths = input_paths or []
        task.owner = owner

        def runner():
            try:
                result = engine_call(task.cancel_flag, task.q)
                task.result = result
                task.status = "done"
                task.q.put({
                    "type": "done",
                    "result": result,
                    "output_path": task.output_path,
                    "output_is_dir": task.output_is_dir,
                })
            except Exception as e:
                if task.cancel_flag.is_set():
                    # engine 在取消检查时抛出 CancelledException，归为取消
                    task.status = "cancelled"
                    task.q.put({"type": "cancelled"})
                    for p in task.cleanup_paths:
                        _safe_remove(p)
                else:
                    task.status = "error"
                    task.error = str(e) or e.__class__.__name__
                    task.q.put({"type": "error", "message": task.error})
            finally:
                task.q.put({"type": "end"})

        thread = threading.Thread(target=runner, daemon=True, name=f"pdf-{kind}-{task_id}")
        task.thread = thread
        with self._lock:
            self._tasks[task_id] = task
        thread.start()
        return task_id

    def get(self, task_id: str):
        return self._tasks.get(task_id)

    def cancel(self, task_id: str) -> bool:
        """请求取消任务（实际中断在 engine 下次取消检查时发生）。"""
        task = self._tasks.get(task_id)
        if task and task.status == "running":
            task.cancel_flag.set()
            return True
        return False

    def cleanup_task(self, task_id: str) -> bool:
        """清理远程模式任务的临时输入文件与输出。"""
        task = self._tasks.get(task_id)
        if not task:
            return False
        for p in task.input_paths:
            _safe_remove(p)
        task.input_paths = []
        if task.temp_dir:
            _safe_remove(task.temp_dir)
            task.temp_dir = ""
        else:
            # 输出在 uploads 共享区，单独删
            _safe_remove(task.output_path)
        return True


def _safe_remove(path: str) -> bool:
    """安全删除文件或目录（忽略不存在和权限错误）。"""
    if not path:
        return False
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass
        return True
    except Exception:
        return False


# 全局单例
task_manager = TaskManager()
