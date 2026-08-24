"""子进程 OCR Worker。

通过子进程隔离 PaddleOCR 推理，彻底解决 PaddlePaddle C++ 内存池不释放问题。

实测数据（200页年鉴 PDF，DPI=200）：
  - 原方案（同进程）：内存从 400MB 涨到 4.2GB，每页涨约 20-50MB
  - auto_growth + ClearIntermediateTensor：基线仍在涨，15 页到 1.4GB
  - 子进程方案：每 N 页启动新子进程，主进程内存稳定在 356MB，零增长

原理：子进程退出时 OS 强制回收所有内存（包括 PaddlePaddle 的 NaiveAllocator
缓存的中间张量），是唯一能彻底解决 C++ 内存池不释放问题的方案。

调用方式：
  from .subprocess_ocr import SubprocessOCRPool
  pool = SubprocessOCRPool(batch_size=5)
  result = pool.ocr(image, slot=0)  # 同步接口

实现：
  - 每个 slot 维护一个子进程，处理 batch_size 页后重启
  - 通过 stdin/stdout 用 JSON + base64 通信（避免 pickle 安全问题）
  - 子进程内 PaddleOCR 实例复用，每页只做推理
"""
from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseProvider, OCRLine, OCRResult

logger = logging.getLogger(__name__)

# Worker 脚本路径（与本文件同目录）
_WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_worker_ocr.py")


def _kill_process_tree(proc: "subprocess.Popen") -> None:
    """强制杀死子进程及其全部后代。

    关键（修孤儿内存泄漏）：sys.executable 在 venv 下是 .venv\\Scripts\\python.exe
    （一个 7MB 的启动器 shim），它会再 spawn 真正的 Python312\\python.exe worker
    （1-2GB，持有 paddle 模型）。Popen.kill() 只杀启动器 shim，孙进程 worker
    成为孤儿继续占 CPU/内存——实测一台机器堆积 13GB 孤儿 worker 导致全部 90s
    超时。本函数用 Windows 原生 taskkill /T 杀整棵树，确保孙进程一起回收。

    优先级：taskkill /F /T（原生进程树杀）→ psutil 递归 kill → Popen.kill() 兜底。
    """
    pid = getattr(proc, "pid", 0)
    if not pid:
        try:
            proc.kill()
        except Exception:
            pass
        return
    # 1. 优先 taskkill /T：Windows 原生进程树杀，能杀到孙进程
    try:
        r = subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True, timeout=10,
        )
        if r.returncode == 0:
            return
    except Exception:
        pass
    # 2. 回退 psutil 递归 kill（跨平台）
    try:
        import psutil
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        parent.kill()
        return
    except Exception:
        pass
    # 3. 最后回退 Popen.kill()（只杀直接子进程，可能留孙进程孤儿）
    try:
        proc.kill()
    except Exception:
        pass


def _read_stream_safely(stream, timeout: float = 3.0, max_bytes: int = 4096) -> bytes:
    """带超时读取流内容（stderr），避免子进程退出时永久阻塞。

    子进程意外退出后，_recv 用 stderr.read() 读取错误信息定位原因。但 venv 的
    shim（.venv\\Scripts\\python.exe）会再 spawn 真 python，孙进程可能逃过 kill
    并一直占用 stderr 句柄 → read() 永不返回（无超时），且此刻正在持锁
    （_locks[slot] / _restart_locks[slot]）→ 该槽位被永久锁死，任务卡死。

    这里用线程 + Event 加超时：正常时读到 EOF/达到上限立即返回；异常时超时
    返回空串（放弃的 daemon 线程不占锁，随进程退出回收）。
    """
    import threading

    holder = {"data": b""}
    done_event = threading.Event()

    def _read():
        try:
            holder["data"] = stream.read(max_bytes)
        except Exception:
            holder["data"] = b""
        finally:
            done_event.set()

    read_thread = threading.Thread(target=_read, daemon=True, name="stderr-read")
    read_thread.start()
    done_event.wait(timeout=timeout)
    return holder["data"]


class SubprocessOCRPool:
    """子进程 OCR 实例池。

    每个 slot 对应一个长期运行的子进程，处理 batch_size 页后自动重启
    释放内存。线程安全：每个 slot 有独立锁。
    """

    def __init__(self, pool_size: int, paddle_config: Dict[str, Any],
                 batch_size: int = 5, cpu_threads: int = 0) -> None:
        # batch_size=5：每个子进程处理 5 页后重启，与 _worker_ocr.py 的内存监控间隔对齐
        # 每页都会检查内存阈值（1500MB），超限立即退出等待重启
        self._pool_size = pool_size
        self._paddle_config = paddle_config
        self._batch_size = max(1, batch_size)
        # 每子进程 CPU 线程数：0=自动（cpu_count // pool_size），>0 固定值
        self._cpu_threads = max(0, int(cpu_threads))
        # 每个 slot 的子进程句柄和锁
        self._procs: List[Optional[subprocess.Popen]] = [None] * pool_size
        self._locks: List[threading.Lock] = [threading.Lock() for _ in range(pool_size)]
        # 每个 slot 的重启锁：防止主线程 stall 检测与 worker 线程 ocr() 重试
        # 同时调用 _restart_proc 产生竞态（误 kill 新子进程 / 启动两个子进程 /
        # ready 响应被错误消费）
        self._restart_locks: List[threading.Lock] = [threading.Lock() for _ in range(pool_size)]
        # 每个 slot 已处理的页数（用于决定何时重启）
        self._page_counts: List[int] = [0] * pool_size
        # 每个 slot 的子进程启动时间（用于 stall 诊断时计算运行时长）
        self._start_times: List[float] = [0.0] * pool_size
        # 环境变量（必须在子进程启动前设置）
        self._env = self._build_env()

    def _build_env(self) -> Dict[str, str]:
        """构建子进程环境变量。

        CPU 线程分配（性能优化）：
          多并发时每个子进程默认开满所有核的 OpenMP 线程，N 个 OCR 子进程
          + N 个 layout 子进程同时满载 → 严重超订（上下文切换开销大）。
          这里按 pool_size（=max_concurrent）均分核数：
            threads = cpu_count // pool_size
          单并发时 = 核数（不限制，与旧行为一致）；5 并发 8 核时每子进程 1 线程，
          总线程 ≈ 2×核数（OCR+layout），接近但不严重超订。
          固定值优先：cpu_threads>0 时直接用，用户可精确控制。
        """
        env = os.environ.copy()
        # PaddlePaddle 内存分配器策略（必须在 import paddle 之前设置）
        # 官方推荐（PaddleOCR #11639 / 讨论 #14497）：naive_best_fit 在 CPU 模式下
        # 控制内存增长优于 auto_growth——auto_growth 的归还策略在 CPU 上偏弱，项目
        # 自测 15 页仍涨到 1.4GB。naive_best_fit 复用缓存块，配合 eager_delete 系列
        # 立即回收中间张量，把 Paddle 的 Tensor 缓存复用控制在低位。
        env["FLAGS_allocator_strategy"] = "naive_best_fit"
        env["FLAGS_eager_delete_scope"] = "True"            # 同步删除 scope
        env["FLAGS_eager_delete_tensor_gb"] = "0.0"          # 立即回收张量（阈值 0=立即）
        env["FLAGS_fast_eager_deletion_mode"] = "True"       # 快速 GC
        env["FLAGS_use_pinned_memory"] = "False"             # 关闭锁页内存，降 CPU 开销
        env["FLAGS_fraction_of_cpu_memory_to_use"] = "0.1"
        env["FLAGS_initial_cpu_memory_in_mb"] = "128"
        # CPU 线程数：OpenMP/MKL/OpenBLAS 在子进程 import paddle 前读取
        if self._cpu_threads > 0:
            threads = self._cpu_threads
        else:
            cpu_count = os.cpu_count() or 4
            threads = max(1, cpu_count // max(1, self._pool_size))
        env["OMP_NUM_THREADS"] = str(threads)
        env["MKL_NUM_THREADS"] = str(threads)
        env["OPENBLAS_NUM_THREADS"] = str(threads)
        # 标记子进程为 OCR worker 模式
        # 打包后 sys.executable 是 server-paddle.exe，会执行 main.py 的 main()
        # 设置此环境变量后，main.py 顶部检测到并跳转执行 _worker_ocr.py 主循环
        # 避免子进程输出端口提示、启动 uvicorn 等无关逻辑污染 JSON 通信通道
        env["_OCR_WORKER_MODE"] = "1"
        return env

    def _ensure_proc(self, slot: int) -> subprocess.Popen:
        """确保指定 slot 的子进程已启动。"""
        proc = self._procs[slot]
        if proc is not None and proc.poll() is None:
            return proc
        # 启动新子进程
        proc = subprocess.Popen(
            [sys.executable, _WORKER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._env,
            text=False,  # 用二进制模式，避免编码问题
            bufsize=0,
        )
        # 发送初始化配置
        init_msg = {"type": "init", "config": self._paddle_config}
        self._send(proc, init_msg)
        resp = self._recv(proc)
        if resp.get("type") != "ready":
            raise RuntimeError(f"子进程初始化失败: {resp}")
        self._procs[slot] = proc
        self._page_counts[slot] = 0
        self._start_times[slot] = time.time()
        logger.info("子进程 OCR worker %d 已启动 (pid=%d)", slot, proc.pid)
        return proc

    def _kill_proc(self, slot: int) -> None:
        """仅 kill 指定 slot 的子进程，不启动新进程。

        与 _restart_proc 的区别：
          - _restart_proc: kill 旧进程 + 启动新进程 + 等待新进程加载模型 ready（10-20s）
          - _kill_proc: 仅 kill 旧进程，不启动新进程（<1s）

        用途：delete_task / pause_task 只需切断 OCR 推理让 worker 线程退出，
        不需要立即启动新子进程（任务已被删除/暂停，新子进程启起来也没用）。
        下次 ocr() 调用时 _ensure_proc 会自动懒启动新进程。

        加 _restart_locks[slot] 锁，防止与 _restart_proc 竞态。
        """
        with self._restart_locks[slot]:
            old_proc = self._procs[slot]
            if old_proc is None:
                return
            try:
                self._send(old_proc, {"type": "exit"}, timeout=5.0)
            except Exception:
                pass
            try:
                old_proc.wait(timeout=3)
            except Exception:
                _kill_process_tree(old_proc)
            # 显式关闭管道：让阻塞在 readline() 的 reader_thread 收到 EOF 退出
            try:
                old_proc.stdin.close()
            except Exception:
                pass
            try:
                old_proc.stdout.close()
            except Exception:
                pass
            self._procs[slot] = None
            self._page_counts[slot] = 0
            self._start_times[slot] = 0.0
            logger.info("子进程 OCR worker %d 已 kill（不重启，下次懒启动）", slot)

    def _restart_proc(self, slot: int) -> subprocess.Popen:
        """重启指定 slot 的子进程，释放内存。

        加 _restart_locks[slot] 锁，防止主线程 stall 检测与 worker 线程 ocr()
        重试同时调用产生竞态：
          - 误 kill 新启动的子进程
          - 同时启动两个子进程，一个泄漏
          - ready 响应被错误消费

        kill 旧进程后显式关闭 stdout/stdin：让所有阻塞在 readline() 上的
        reader_thread 立即收到 EOF 返回 b"" 退出，避免僵尸线程累积。
        """
        with self._restart_locks[slot]:
            old_proc = self._procs[slot]
            if old_proc is not None:
                try:
                    self._send(old_proc, {"type": "exit"}, timeout=5.0)
                except Exception:
                    pass
                try:
                    old_proc.wait(timeout=5)
                except Exception:
                    _kill_process_tree(old_proc)
                # 显式关闭管道：让阻塞在 old_proc.stdout.readline() 的
                # reader_thread（来自 _recv 超时遗留）立即收到 EOF 退出
                try:
                    old_proc.stdin.close()
                except Exception:
                    pass
                try:
                    old_proc.stdout.close()
                except Exception:
                    pass
                self._procs[slot] = None
            logger.info("子进程 OCR worker %d 重启（释放内存）", slot)
            return self._ensure_proc(slot)

    @staticmethod
    def _send(proc: subprocess.Popen, msg: dict, timeout: float = 30.0) -> None:
        """发送 JSON 消息（以换行符分隔）。

        带超时保护（关键防死锁）：Windows 匿名管道缓冲仅数 KB，base64 图片
        消息可达 15MB+。若子进程卡死（paddle 推理死锁）不再消费 stdin，
        write() 会**永久阻塞**且无任何日志——faulthandler 实测长文档跑到
        某页时卡在 proc.stdin.write 30+ 分钟，_recv 的 90s 超时永远走不到。
        这里用线程 + Event 加超时，超时抛 RuntimeError，由上层 ocr() 捕获后
        重启子进程，与 _recv 超时机制对齐。

        参数:
            timeout: 写入超时秒数，默认 30s。正常写入 <1s，卡死时尽快暴露。
        """
        import threading
        data = (json.dumps(msg) + "\n").encode("utf-8")
        result_holder = {"err": None}
        done_event = threading.Event()
        proc_pid = getattr(proc, "pid", 0)

        def _write():
            try:
                proc.stdin.write(data)
                proc.stdin.flush()
            except Exception as e:
                result_holder["err"] = e
            finally:
                done_event.set()

        writer_thread = threading.Thread(
            target=_write, daemon=True,
            name=f"send-line-{proc_pid}",
        )
        writer_thread.start()
        if not done_event.wait(timeout=timeout):
            raise RuntimeError(
                f"子进程发送消息超时（{timeout}s），疑似子进程卡死不消费 stdin"
            )
        if result_holder["err"] is not None:
            raise result_holder["err"]

    @staticmethod
    def _recv(proc: subprocess.Popen, timeout: float = 90.0) -> dict:
        """接收一行 JSON 消息。

        跳过非 JSON 行（防御性：即使 worker 有残留 stdout 输出也不会崩溃）。
        子进程崩溃时 readline() 返回空，此时读取 stderr 获取崩溃信息。
        超时未收到响应视为子进程卡死（paddle 死锁），抛异常由上层重试。

        参数:
            timeout: 单次读取超时秒数，默认 90s（1.5 分钟）
                     单页 OCR 正常 1-10s，复杂版面 <60s，超过 90s 基本是死锁
                     旧值 180s 过大：3 次重试 = 540s 远超 stall_timeout=300s，
                     导致 stall 检测与 ocr() 重试必然产生竞态。
                     新值 90s：2 次重试 = 180s < 300s stall_timeout，避免竞态
        """
        import threading

        max_skips = 50  # 最多跳过 50 行非 JSON，防止死循环
        # 用 proc 的 pid 作为线程名标识，便于 stall 诊断时统计残留 reader_thread 数
        proc_pid = getattr(proc, "pid", 0)
        for _ in range(max_skips):
            # Windows pipe 不支持 select，用线程+Event 实现超时读取
            result_holder = {"line": None}
            read_event = threading.Event()

            def _read_line():
                try:
                    result_holder["line"] = proc.stdout.readline()
                except Exception:
                    result_holder["line"] = b""
                finally:
                    read_event.set()

            # 命名规则 recv-line-{pid}：诊断时可通过 threading.enumerate() 统计
            # 长时间运行后若残留 reader_thread 数量异常增长（>10），
            # 说明超时未退出的线程在累积，是潜在 stall 根因
            reader_thread = threading.Thread(
                target=_read_line, daemon=True,
                name=f"recv-line-{proc_pid}",
            )
            reader_thread.start()
            if not read_event.wait(timeout=timeout):
                # 超时：子进程卡死（paddle 死锁），reader_thread 仍在阻塞 readline
                # daemon=True 会随主进程退出，不会泄漏
                # 但若同一子进程连续多页接近超时，残留线程会累积占 GIL
                alive_count = sum(
                    1 for t in threading.enumerate()
                    if t.name.startswith("recv-line-") and t.is_alive()
                )
                logger.warning(
                    "[recv超时] pid=%d 超时%ss, 当前残留 reader_thread=%d",
                    proc_pid, timeout, alive_count,
                )
                raise RuntimeError(
                    f"子进程 OCR 响应超时（{timeout}s），疑似 paddle 死锁"
                )

            line = result_holder["line"]
            if not line:
                # 子进程已退出，读取 stderr 获取错误信息（带超时，防止孙进程
                # 占用 stderr 句柄导致无超时阻塞、槽位锁死）
                stderr_msg = ""
                try:
                    stderr_data = _read_stream_safely(proc.stderr)
                    if stderr_data:
                        stderr_msg = stderr_data.decode("utf-8", errors="replace")[-500:]
                except Exception:
                    pass
                returncode = proc.poll()
                raise RuntimeError(
                    f"子进程意外退出 (returncode={returncode}): {stderr_msg}"
                )
            try:
                return json.loads(line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # 非 JSON 行（残留输出），记日志并跳过
                logger.warning("跳过非 JSON 行: %s", line[:200])
                continue
        raise RuntimeError("连续 50 行非 JSON 输出，子进程通信异常")

    def ocr(self, image: np.ndarray, slot: int = 0, new_page: bool = False) -> dict:
        """对图片做 OCR，返回原始结果。

        返回格式与 PaddleOCR.ocr() 一致：[[box, (text, conf)], ...]

        参数:
            image: BGR numpy 数组
            slot: 实例池槽位号
            new_page: 是否为一页的页级边界调用。
                      **只有 new_page=True 的调用才计入子进程页数并触发 batch 重启。**

        重要（性能修复）：版面分析路径下，一页会调用多次 ocr()（每个文本区域一次
        区域 OCR + 一次整页补充 OCR）。旧逻辑按"调用次数"计数，batch_size=5 意味着
        一页 3-10 次调用后必然触发重启 → 几乎每页都重启子进程、反复重载模型
        （det+rec 模型加载约 5-6s），是单页 20-30s 的主因。
        现在改为按"真实页数"计数：只有每页的关键调用（整页补充 OCR / 整页回退 OCR）
        传 new_page=True，区域 OCR 不计，batch_size=5 = 每 5 页重启一次。

        子进程崩溃自动恢复：_recv 检测到子进程退出会抛 RuntimeError，
        这里捕获后重启子进程并重试本次 OCR，避免单次崩溃导致整个任务失败。
        最多重试 1 次（共 2 次尝试），与 pdf_handler 的单页重试配合形成两道防线。

        重试次数与超时的关系（关键防竞态）：
          _recv 超时 90s × 2 次尝试 = 180s < stall_timeout 300s
          确保 ocr() 重试在 stall 检测触发前完成，避免主线程与 worker 线程
          同时调用 _restart_proc 产生竞态。
          旧值 3 次重试 × 180s = 540s > 300s stall_timeout，必然竞态。
        """
        import time as _time
        with self._locks[slot]:
            # 图片编码为 base64（BGR 数组）—— 只编码一次，重试时复用
            t_enc_start = _time.time()
            img_bytes = image.tobytes()
            img_b64 = base64.b64encode(img_bytes).decode("ascii")
            t_enc = _time.time() - t_enc_start
            msg = {
                "type": "ocr",
                "image": img_b64,
                "shape": list(image.shape),
                "dtype": str(image.dtype),
            }

            last_error = None
            for attempt in range(2):
                try:
                    proc = self._ensure_proc(slot)
                    t_send_start = _time.time()
                    self._send(proc, msg)
                    resp = self._recv(proc)
                    t_comm = _time.time() - t_send_start
                    if resp.get("type") == "error":
                        # worker 内部 OCR 失败（如模型加载错误），不算崩溃
                        # 直接抛出由上层重试
                        raise RuntimeError(f"子进程 OCR 失败: {resp.get('message')}")
                    # 仅页级边界调用（new_page=True）参与计数并触发 batch 重启，
                    # 区域 OCR / 检测不计入，避免一页多次调用导致几乎每页重启子进程
                    if new_page:
                        self._page_counts[slot] += 1
                        if self._page_counts[slot] >= self._batch_size:
                            self._restart_proc(slot)
                    raw_result = resp.get("result", [])
                    logger.debug(
                        "子进程OCR[slot=%d]: 编码 %.2fs + 通信+推理 %.2fs = %.2fs | "
                        "图片 %dx%d | 原始结果 %d 项 | new_page=%s | 累计页数=%d",
                        slot, t_enc, t_comm, t_enc + t_comm,
                        image.shape[1], image.shape[0], len(raw_result) if raw_result else 0,
                        new_page, self._page_counts[slot],
                    )
                    return raw_result
                except RuntimeError as e:
                    last_error = e
                    # 子进程崩溃（意外退出/通信异常）：重启并重试
                    logger.warning(
                        "子进程 OCR slot=%d 第 %d 次尝试失败，重启子进程: %s",
                        slot, attempt + 1, e,
                    )
                    # 强制重启子进程（_ensure_proc 会重建）
                    try:
                        self._restart_proc(slot)
                    except Exception as restart_err:
                        logger.error("子进程重启失败: %s", restart_err)
                    if attempt < 1:
                        _time.sleep(1)
            # 2 次都失败，抛出异常由上层 pdf_handler 处理
            raise RuntimeError(
                f"子进程 OCR 连续 2 次失败 (slot={slot}): {last_error}"
            )

    def detect_boxes(self, image: np.ndarray, slot: int = 0) -> List:
        """仅检测文字框（不做识别）。

        不参与子进程页计数：detect 用于第二层过滤（每页可能多次/无），
        若计入页数会干扰 batch 重启节奏。页计数只由 ocr(new_page=True) 驱动。
        """
        with self._locks[slot]:
            proc = self._ensure_proc(slot)
            img_bytes = image.tobytes()
            img_b64 = base64.b64encode(img_bytes).decode("ascii")
            msg = {
                "type": "detect",
                "image": img_b64,
                "shape": list(image.shape),
                "dtype": str(image.dtype),
            }
            self._send(proc, msg)
            resp = self._recv(proc)
            if resp.get("type") == "error":
                raise RuntimeError(f"子进程检测失败: {resp.get('message')}")
            return resp.get("boxes", [])

    def shutdown(self) -> None:
        """关闭所有子进程。"""
        for slot, proc in enumerate(self._procs):
            if proc is not None:
                try:
                    self._send(proc, {"type": "exit"}, timeout=5.0)
                    proc.wait(timeout=3)
                except Exception:
                    _kill_process_tree(proc)
                self._procs[slot] = None
        logger.info("所有子进程 OCR worker 已关闭")


class SubprocessOCRProvider(BaseProvider):
    """子进程 OCR Provider，接口与 PaddleLocalProvider 兼容。"""

    name = "paddle-subprocess"

    def __init__(self, lang: str = "ch", use_gpu: bool = False,
                 ocr_version: str = "PP-OCRv6",
                 det_model_dir: str = "", rec_model_dir: str = "",
                 drop_score: float = 0.0,
                 det_db_unclip_ratio: float = 1.8, det_db_box_thresh: float = 0.5,
                 batch_size: int = 5, det_score_thresh: float = 0.3,
                 pool_size: int = 1, cpu_threads: int = 0) -> None:
        super().__init__()
        self.lang = lang
        self.use_gpu = use_gpu
        self.ocr_version = ocr_version
        self.det_model_dir = det_model_dir
        self.rec_model_dir = rec_model_dir
        self.drop_score = drop_score
        self.det_db_unclip_ratio = det_db_unclip_ratio
        self.det_db_box_thresh = det_db_box_thresh
        self.batch_size = batch_size
        self.det_score_thresh = det_score_thresh
        # 每子进程 CPU 线程数：0=自动（cpu_count // pool_size），>0 固定值
        self.cpu_threads = max(0, int(cpu_threads))
        # 进程池大小：必须等于 max_concurrent，每个并发任务用独立 slot
        # 之前硬编码 pool_size=1，导致 max_concurrent>1 时 slot=1 越界
        # 抛 IndexError: list index out of range
        self._pool_size = max(1, pool_size)
        self._pool: Optional[SubprocessOCRPool] = None
        self._init_failed = False

    def is_available(self) -> bool:
        """检查 paddleocr 是否安装（子进程会自己 import）。"""
        try:
            import paddleocr  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_pool(self) -> Optional[SubprocessOCRPool]:
        """懒初始化子进程池。"""
        if self._init_failed:
            return None
        if self._pool is not None:
            return self._pool
        try:
            from .paddle_local import _build_paddle_kwargs
            config = {
                "lang": self.lang,
                "ocr_version": self.ocr_version,
                "det_model_dir": self.det_model_dir,
                "rec_model_dir": self.rec_model_dir,
                "drop_score": self.drop_score,
                "det_db_unclip_ratio": self.det_db_unclip_ratio,
                "det_db_box_thresh": self.det_db_box_thresh,
            }
            # 复用 paddle_local 的参数构建逻辑（v5 参数名）
            kwargs = _build_paddle_kwargs(config)
            self._pool = SubprocessOCRPool(
                pool_size=self._pool_size,
                paddle_config=kwargs,
                batch_size=self.batch_size,
                cpu_threads=self.cpu_threads,
            )
            logger.info(
                "子进程 OCR 池已初始化（pool_size=%d, batch_size=%d, ocr=%s）",
                self._pool_size, self.batch_size, self.ocr_version,
            )
            return self._pool
        except Exception as e:
            logger.warning("子进程 OCR 池初始化失败: %s", e)
            self._init_failed = True
            return None

    def recognize(self, image: np.ndarray, slot: int = 0, new_page: bool = False) -> OCRResult:
        """识别图片中的文字。

        参数:
            image: BGR numpy 数组
            slot: 实例池槽位号
            new_page: 页级边界标记，True 时才计入子进程页数并触发 batch 重启。
                      由 handler 层在每页的整页识别调用（补充 OCR / 整页回退 OCR）传入，
                      区域 OCR 不传（保持 False），避免一页多次调用导致子进程频繁重启。

        子进程崩溃时自动重启并重试一次。崩溃信息记录到日志便于诊断。
        """
        h, w = image.shape[:2]
        pool = self._ensure_pool()
        if pool is None:
            # 初始化失败时返回空结果会把"环境损坏"伪装成"整页无文字"，
            # 导致整本输出空白但任务显示完成，因此必须上抛
            raise RuntimeError(
                "OCR 子进程池初始化失败，无法识别；"
                "请确认使用项目 .venv 启动且 paddleocr/paddlepaddle 版本匹配"
            )
        try:
            result = pool.ocr(image, slot=slot, new_page=new_page)
            lines = self._parse_result(result, w, h)
            return OCRResult(lines=lines, width=w, height=h, provider=self.name)
        except Exception as e:
            # 子进程可能崩溃，记录详细错误并尝试重启
            logger.error("子进程 OCR 识别失败（将重启子进程）: %s", e)
            try:
                pool._restart_proc(slot)
                # 重试一次
                result = pool.ocr(image, slot=slot, new_page=new_page)
                lines = self._parse_result(result, w, h)
                logger.info("子进程重启后重试成功")
                return OCRResult(lines=lines, width=w, height=h, provider=self.name)
            except Exception as e2:
                logger.error("子进程重启后仍失败: %s", e2)
                # 同上：失败不能伪装成空结果，上抛交给页级重试与任务熔断处理
                raise RuntimeError(f"子进程 OCR 重启重试后仍失败: {e2}") from e2

    def detect_boxes(self, image: np.ndarray, slot: int = 0):
        """检测文字框。"""
        pool = self._ensure_pool()
        if pool is None:
            return None
        try:
            boxes = pool.detect_boxes(image, slot=slot)
            return boxes
        except Exception as e:
            logger.warning("子进程检测失败: %s", e)
            return None

    def _parse_result(self, result, width: int, height: int) -> List[OCRLine]:
        """解析 OCR 结果为 OCRLine 列表。

        复用 paddle_local 的解析逻辑（子进程返回的格式与 PaddleOCR.ocr() 一致）。
        """
        from .paddle_local import PaddleLocalProvider
        # 借用 PaddleLocalProvider 的 _parse_result 静态逻辑
        # result 格式: [[box, (text, conf)], ...]
        lines = []
        for item in result:
            if not item or len(item) < 2:
                continue
            box = item[0]
            text_conf = item[1]
            if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
                text, conf = text_conf[0], text_conf[1]
            else:
                continue
            if conf < self.det_score_thresh:
                continue
            try:
                coords = [[float(p[0]), float(p[1])] for p in box]
            except (TypeError, ValueError, IndexError):
                continue
            lines.append(OCRLine(text=text, coords=coords, confidence=float(conf)))
        return lines

    def shutdown(self) -> None:
        """关闭子进程池。"""
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
