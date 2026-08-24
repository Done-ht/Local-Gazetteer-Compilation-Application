#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""多实例用户管理器。

维护多个用户数据区，每个用户绑定固定端口，实例常驻运行。
（新版多用户已内建在 web_api 单实例中：一个端口，登录后各看各的库，
 优先使用网页注册账号或 `python main.py user create`；本文件的
 多实例命令保留用于兼容旧版数据。）

【交互菜单模式】（推荐，适合非计算机用户）：
    python manager.py
    直接运行不带参数，会显示中文菜单，按数字选择即可。

【命令行模式】（适合脚本/高级用户）：
    python manager.py create <名字> [--port <端口>] [--host <地址>]
        创建新用户（旧版多实例模式），自动分配端口（从 20000 起向后寻找可用端口），建数据目录。
        新版建议改用：python main.py user create <用户名> --password <密码>

    python manager.py start <名字>
        启动该用户的检索服务实例（后台常驻）。

    python manager.py stop <名字>
        停止该用户实例。

    python manager.py restart <名字>
        重启该用户实例。

    python manager.py list
        列出所有用户及运行状态。

    python manager.py start-all
        启动所有未运行的用户实例。

    python manager.py stop-all
        停止所有运行中的用户实例。

    python manager.py remove <名字>
        删除用户（停止实例 + 删除数据目录，不可恢复）。

设计要点：
- 用户配置存储在 instances.json（用户名 → 端口/数据目录/host）
- 主数据区（名为 main）是默认共享实例，端口 20000（被占用时自动向后寻找可用端口）
- 服务器模式默认只启动一个 main 实例，多用户（账号+库归属）在服务内建
- 子进程通过 --no-dialog 启动，不弹交互，与 manager 窗口共存
- 子进程独立日志：<data_dir>/output/log/server.log
- 关闭 manager 窗口时，所有子进程同步结束
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from typing import Optional


# 打包成 exe 后，脚本实际位于 _internal/ 下，但数据/PID/配置应放在 exe 同级目录，
# 避免用户数据混入程序内部目录。
if getattr(sys, "frozen", False):
    _SCRIPT_DIR = os.path.dirname(sys.executable)
else:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_FILE = os.path.join(_SCRIPT_DIR, "instances.json")
WEB_API = os.path.join(_SCRIPT_DIR, "web_api.py")

# manager 自身日志（打包后放在 exe 同级 output/log/manager.log）
MANAGER_LOG_DIR = os.path.join(_SCRIPT_DIR, "output", "log")
MANAGER_LOG_FILE = os.path.join(MANAGER_LOG_DIR, "manager.log")


def _log(msg: str) -> None:
    """写一行 manager 日志，便于定位打包后子进程启动问题。"""
    try:
        os.makedirs(MANAGER_LOG_DIR, exist_ok=True)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(MANAGER_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{ts} {msg}\n")
    except Exception:
        pass

# 端口分配区间：主实例 20000，子用户从 20001 开始
MAIN_PORT = 20000
MAIN_NAME = "main"
MAIN_DATA_DIR = os.path.join(_SCRIPT_DIR, "data")


# ============================================================
#  配置读写
# ============================================================

def _load_config() -> dict:
    """读取用户配置。不存在则返回默认结构（含 main 主实例）。"""
    if os.path.isfile(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if "users" not in cfg:
                cfg["users"] = {}
            return cfg
        except (json.JSONDecodeError, OSError):
            pass
    # 默认配置：主实例
    return {
        "users": {
            MAIN_NAME: {
                "port": MAIN_PORT,
                "host": "0.0.0.0",
                "data_dir": MAIN_DATA_DIR,
            }
        }
    }


def _save_config(cfg: dict) -> None:
    """保存配置到 instances.json。"""
    tmp = CONFIG_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CONFIG_FILE)


def _get_user(cfg: dict, name: str) -> Optional[dict]:
    return cfg["users"].get(name)


# ============================================================
#  端口分配
# ============================================================

def _next_available_port(cfg: dict) -> int:
    """找出下一个可用端口（跳过配置中已分配的及系统已占用的）。"""
    used = {u["port"] for u in cfg["users"].values()}
    port = MAIN_PORT
    while port in used or _is_port_in_use(port):
        port += 1
        if port > 65535:
            raise RuntimeError("无可用端口（1-65535 均已占用）")
    return port


def _is_port_in_use(port: int) -> bool:
    """检测端口是否被系统占用（可能被其他程序占用）。

    分别检测 0.0.0.0 和 127.0.0.1 绑定，任一失败即认为占用。
    """
    import socket
    if not (1 <= port <= 65535):
        return True  # 非法端口视为占用
    for host in ("0.0.0.0", "127.0.0.1"):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.settimeout(0.3)
            try:
                s.bind((host, port))
            except OSError:
                return True
    return False


# ============================================================
#  进程管理
# ============================================================

def _pid_file(name: str) -> str:
    """用户实例的 PID 文件路径。"""
    return os.path.join(_SCRIPT_DIR, f".pid_{name}")


def _read_pid(name: str) -> Optional[int]:
    """读取用户实例的 PID。文件不存在或无效返回 None。"""
    pf = _pid_file(name)
    if not os.path.isfile(pf):
        return None
    try:
        with open(pf, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except (ValueError, OSError):
        return None


def _write_pid(name: str, pid: int) -> None:
    with open(_pid_file(name), "w", encoding="utf-8") as f:
        f.write(str(pid))


def _clear_pid(name: str) -> None:
    pf = _pid_file(name)
    if os.path.isfile(pf):
        try:
            os.remove(pf)
        except OSError:
            pass


def _is_pid_running(pid: int) -> bool:
    """检测进程是否在运行。"""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)  # 信号 0：不实际发送信号，仅检测进程是否存在
    except OSError:
        return False
    return True


def _is_user_running(name: str) -> bool:
    """用户实例是否在运行（PID 有效且进程存活）。"""
    pid = _read_pid(name)
    if pid is None:
        return False
    if _is_pid_running(pid):
        return True
    # 进程已死，清理残留 PID 文件
    _clear_pid(name)
    return False


# ============================================================
#  Windows Job Object + 控制台事件处理
#  确保关闭 manager 窗口（X 按钮 / Ctrl+C / Ctrl+Break）时
#  所有子进程同步终止，即使 manager 被强制结束也不会残留。
# ============================================================

_job_handle = None          # Windows Job Object 句柄
_console_handler_ref = None  # 保持控制台处理器引用，防止被垃圾回收


def _setup_process_management():
    """设置进程管理：Windows Job Object + 控制台关闭事件处理。

    在交互模式启动时调用一次，确保：
    1. 关闭 manager 窗口（X 按钮）或 Ctrl+C 时，所有子进程同步终止
    2. 即使 manager 被强制结束（如任务管理器），Job Object 也会杀掉子进程
    """
    import atexit
    atexit.register(_cleanup_on_exit, stop_all=True)
    if sys.platform == "win32":
        _setup_job_object()
        _setup_console_ctrl_handler()


def _setup_job_object():
    """创建 Windows Job Object，设置 KILL_ON_JOB_CLOSE 标志。

    子进程加入后，当 manager 进程退出（任何方式：正常退出、Ctrl+C、
    关闭窗口、任务管理器结束、崩溃），Job Object 句柄关闭，
    系统自动终止所有加入的子进程。
    """
    global _job_handle
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        JobObjectExtendedLimitInformation = 9

        class _BASIC_LIMIT(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_void_p),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", wintypes.ULARGE_INTEGER),
                ("WriteOperationCount", wintypes.ULARGE_INTEGER),
                ("OtherOperationCount", wintypes.ULARGE_INTEGER),
                ("ReadTransferCount", wintypes.ULARGE_INTEGER),
                ("WriteTransferCount", wintypes.ULARGE_INTEGER),
                ("OtherTransferCount", wintypes.ULARGE_INTEGER),
            ]

        class _EXTENDED_LIMIT(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BASIC_LIMIT),
                ("IoInfo", _IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        h = kernel32.CreateJobObjectW(None, None)
        if not h:
            _log(f"[job] CreateJobObjectW 失败: {ctypes.get_last_error()}")
            return
        info = _EXTENDED_LIMIT()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        size = ctypes.sizeof(_EXTENDED_LIMIT)
        if not kernel32.SetInformationJobObject(
            h, JobObjectExtendedLimitInformation, ctypes.byref(info), size
        ):
            _log(f"[job] SetInformationJobObject 失败: {ctypes.get_last_error()}")
            kernel32.CloseHandle(h)
            return
        _job_handle = h
        _log("[job] Job Object 已创建，子进程将随 manager 退出而终止")
    except Exception as e:
        _log(f"[job] 设置 Job Object 失败: {e}")


def _assign_to_job(proc):
    """将子进程加入 Job Object。"""
    if _job_handle is None:
        return
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        PROCESS_SET_QUOTA = 0x0100
        PROCESS_TERMINATE = 0x0001
        h = kernel32.OpenProcess(
            PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, proc.pid
        )
        if not h:
            _log(f"[job] OpenProcess({proc.pid}) 失败: {ctypes.get_last_error()}")
            return
        try:
            if not kernel32.AssignProcessToJobObject(_job_handle, h):
                _log(
                    f"[job] AssignProcessToJobObject({proc.pid}) 失败: "
                    f"{ctypes.get_last_error()}"
                )
        finally:
            kernel32.CloseHandle(h)
    except Exception as e:
        _log(f"[job] 分配进程到 Job Object 失败: {e}")


def _setup_console_ctrl_handler():
    """注册控制台事件处理器。

    CTRL_CLOSE_EVENT（点击窗口右上角 X 关闭）时 Windows 仅给约 5 秒
    清理时间。先快速发 CTRL_BREAK 让子进程优雅退出，等待 2 秒后
    强制退出；Job Object 会自动杀掉仍在运行的子进程。
    """
    global _console_handler_ref
    try:
        import ctypes
        from ctypes import wintypes

        HANDLER_ROUTINE = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
        CTRL_C_EVENT = 0
        CTRL_BREAK_EVENT = 1
        CTRL_CLOSE_EVENT = 2
        CTRL_LOGOFF_EVENT = 5
        CTRL_SHUTDOWN_EVENT = 6

        def _handler(ctrl_type):
            if ctrl_type in (
                CTRL_CLOSE_EVENT,
                CTRL_BREAK_EVENT,
                CTRL_LOGOFF_EVENT,
                CTRL_SHUTDOWN_EVENT,
            ):
                try:
                    # 快速发信号让子进程优雅退出（最多等 2 秒）
                    cfg = _load_config()
                    for nm in list(cfg.get("users", {}).keys()):
                        pid = _read_pid(nm)
                        if pid and _is_pid_running(pid):
                            try:
                                os.kill(pid, signal.CTRL_BREAK_EVENT)
                            except OSError:
                                pass
                    time.sleep(2)
                except Exception:
                    pass
                # 强制退出；Job Object 会自动杀掉所有子进程
                os._exit(0)
            # CTRL_C_EVENT: 交给 Python 默认处理（触发 KeyboardInterrupt）
            return False

        _console_handler_ref = HANDLER_ROUTINE(_handler)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.SetConsoleCtrlHandler(_console_handler_ref, True)
        _log("[job] 控制台事件处理器已注册")
    except Exception as e:
        _log(f"[job] 注册控制台事件处理器失败: {e}")


def _force_kill(pid: int):
    """强制终止进程（TerminationProcess / SIGKILL）。"""
    try:
        if sys.platform == "win32":
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            PROCESS_TERMINATE = 0x0001
            h = kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
            if h:
                kernel32.TerminateProcess(h, 1)
                kernel32.CloseHandle(h)
        else:
            os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def _start_process(name: str, user: dict) -> int:
    """启动用户实例子进程。返回 PID。

    若配置端口被其他程序占用，会自动向后寻找可用端口并更新 user['port'] 与配置。
    打包成 exe 后，manager.exe 会启动同目录下的 全文检索系统.exe；
    源码运行则继续调用 python web_api.py。
    """
    data_dir = user["data_dir"]
    port = user["port"]
    host = user["host"]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, "output", "log"), exist_ok=True)

    # 启动前检测端口占用
    if _is_port_in_use(port):
        # 可能是残留进程，先尝试清理同用户的旧实例
        if _is_user_running(name):
            _stop_process(name)
            time.sleep(1)
        # 若仍被占用，自动向后寻找可用端口
        if _is_port_in_use(port):
            cfg = _load_config()
            new_port = _next_available_port(cfg)
            print(f"[提示] 端口 {port} 已被占用，自动切换到 {new_port}")
            user["port"] = new_port
            cfg["users"][name]["port"] = new_port
            _save_config(cfg)
            port = new_port

    # 构造启动命令：打包后启动同目录主程序 exe，源码运行时启动 web_api.py
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(sys.executable)
        main_exe = os.path.join(exe_dir, "全文检索系统.exe")
        if not os.path.isfile(main_exe):
            raise RuntimeError(
                f"未找到主程序 {main_exe}，请确保 manager.exe 与 全文检索系统.exe 在同一目录"
            )
        cmd = [
            main_exe,
            "--data-dir", data_dir,
            "--port", str(port),
            "--host", host,
            "--no-dialog",
        ]
    else:
        cmd = [
            sys.executable, WEB_API,
            "--data-dir", data_dir,
            "--port", str(port),
            "--host", host,
            "--no-dialog",
        ]

    # 子进程与当前窗口共存：不使用 DETACHED_PROCESS，关闭 manager 窗口时子进程同步结束
    # Windows: 使用 CREATE_NEW_PROCESS_GROUP 便于后续统一发信号停止；
    #          同时把 stdout/stderr 重定向到子进程日志，便于捕获 0xc0000142 等启动错误。
    # POSIX: start_new_session=True
    kwargs = {}
    child_log = os.path.join(data_dir, "output", "log", "server_boot.log")
    try:
        os.makedirs(os.path.dirname(child_log), exist_ok=True)
    except Exception:
        child_log = os.devnull
    if sys.platform == "win32":
        # CREATE_NEW_PROCESS_GROUP：便于后续发信号停止
        # CREATE_NO_WINDOW：避免多个控制台实例争抢同一窗口，降低 0xc0000142 概率
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        )
    else:
        kwargs["start_new_session"] = True

    # 多实例同时启动时，Intel MKL/OpenMP 运行时容易冲突（libiomp5md.dll 重复初始化），
    # 导致子进程偶发 0xc0000142。设置这些环境变量可显著提高稳定性。
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # SEQUENTIAL 表示 MKL 不使用线程，避免 OpenMP 运行时冲突（无需额外 threading DLL）
    env["MKL_THREADING_LAYER"] = "SEQUENTIAL"
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    _log(f"[start] {name} cmd={' '.join(cmd)} cwd={_SCRIPT_DIR} child_log={child_log}")
    proc = subprocess.Popen(
        cmd,
        cwd=_SCRIPT_DIR,
        stdin=subprocess.DEVNULL,
        stdout=open(child_log, "a", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        env=env,
        **kwargs,
    )
    # 将子进程加入 Job Object（若已创建），确保 manager 退出时子进程同步终止
    _assign_to_job(proc)
    _write_pid(name, proc.pid)
    _log(f"[start] {name} pid={proc.pid}")
    return proc.pid


def _wait_user_ready(name: str, user: dict, timeout: int = 60) -> bool:
    """等待用户实例的 HTTP 服务就绪。返回是否成功。

    timeout 默认 60 秒，因为加载多个库的 faiss 索引可能需要 30-50 秒。
    """
    import urllib.request
    port = user["port"]
    url = f"http://127.0.0.1:{port}/api/libraries"
    _log(f"[wait] {name} url={url} timeout={timeout}s")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _is_user_running(name):
            _log(f"[wait] {name} process exited before ready")
            return False  # 进程已退出
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    _log(f"[wait] {name} ready")
                    return True
        except OSError as e:
            pass
        time.sleep(1)
    _log(f"[wait] {name} timeout after {timeout}s")
    return False


def _stop_process(name: str) -> bool:
    """停止用户实例。返回是否成功停止。

    先发送 CTRL_BREAK_EVENT（Windows）/ SIGTERM（POSIX）请求优雅退出，
    等待最多 5 秒；若仍未退出，则强制终止（TerminateProcess / SIGKILL）。
    """
    pid = _read_pid(name)
    if pid is None:
        return False
    if not _is_pid_running(pid):
        _clear_pid(name)
        return False
    try:
        if sys.platform == "win32":
            # Windows: SIGBREAK 等价于 Ctrl+Break，让进程优雅退出
            os.kill(pid, signal.CTRL_BREAK_EVENT)
        else:
            # POSIX: 先 SIGTERM，让进程优雅退出
            os.kill(pid, signal.SIGTERM)
    except OSError:
        pass
    # 等待进程退出（最多 5 秒）
    for _ in range(50):
        if not _is_pid_running(pid):
            break
        time.sleep(0.1)
    # 若仍未退出，强制终止
    if _is_pid_running(pid):
        _force_kill(pid)
        # 等待强制终止完成（最多 2 秒）
        for _ in range(20):
            if not _is_pid_running(pid):
                break
            time.sleep(0.1)
    _clear_pid(name)
    return not _is_pid_running(pid)


# ============================================================
#  命令实现
# ============================================================

def cmd_create(args) -> int:
    cfg = _load_config()
    name = args.name
    if name in cfg["users"]:
        print(f"[错误] 用户已存在: {name}")
        return 1

    # 分配端口
    if args.port:
        port = args.port
        # 检查是否与其他用户冲突
        for uname, u in cfg["users"].items():
            if u["port"] == port:
                print(f"[错误] 端口 {port} 已被用户 {uname} 占用")
                return 1
    else:
        port = _next_available_port(cfg)

    # 检查端口是否被系统其他程序占用
    if _is_port_in_use(port):
        print(f"[警告] 端口 {port} 当前已被占用（可能是其他程序），"
              f"启动时可能失败。如需指定其他端口，请用 --port 参数。")

    host = args.host or "127.0.0.1"  # 子用户默认仅本机访问，更安全
    data_dir = os.path.join(_SCRIPT_DIR, f"data_{name}")
    os.makedirs(data_dir, exist_ok=True)

    cfg["users"][name] = {
        "port": port,
        "host": host,
        "data_dir": data_dir,
    }
    _save_config(cfg)
    print(f"[创建] 用户: {name}")
    print(f"  端口: {port}")
    print(f"  地址: {host}")
    print(f"  数据目录: {data_dir}")
    print(f"\n  启动: python manager.py start {name}")
    return 0


def cmd_start(args) -> int:
    cfg = _load_config()
    name = args.name
    user = _get_user(cfg, name)
    if user is None:
        print(f"[错误] 用户不存在: {name}")
        return 1
    if _is_user_running(name):
        pid = _read_pid(name)
        print(f"[跳过] {name} 已在运行 (PID={pid}, 端口={user['port']})")
        return 0
    try:
        pid = _start_process(name, user)
    except RuntimeError as e:
        print(f"[失败] {name}: {e}")
        return 1
    # 等待 HTTP 服务就绪（最长 20 秒）
    if not _wait_user_ready(name, user, timeout=60):
        if not _is_pid_running(pid):
            print(f"[失败] {name} 启动后退出，请检查日志: "
                  f"{user['data_dir']}\\output\\log\\server.log")
        else:
            print(f"[超时] {name} 进程在运行但 HTTP 未就绪（可能仍在加载索引）")
            print(f"  日志: {user['data_dir']}\\output\\log\\server.log")
        _clear_pid(name)
        return 1
    host = user["host"]
    addr = "127.0.0.1" if host == "0.0.0.0" else host
    print(f"[启动] {name} PID={pid}")
    print(f"  访问: http://{addr}:{user['port']}")
    print(f"  （首次启动需要初始化语义模型，约需 5-10 分钟，请耐心等待）")
    return 0


def cmd_stop(args) -> int:
    cfg = _load_config()
    name = args.name
    user = _get_user(cfg, name)
    if user is None:
        print(f"[错误] 用户不存在: {name}")
        return 1
    if not _is_user_running(name):
        print(f"[跳过] {name} 未在运行")
        return 0
    if _stop_process(name):
        print(f"[停止] {name}")
        return 0
    else:
        print(f"[失败] {name} 停止失败，可能需要手动结束进程")
        return 1


def cmd_restart(args) -> int:
    cmd_stop(args)
    time.sleep(1)
    return cmd_start(args)


def cmd_list(args) -> int:
    cfg = _load_config()
    users = cfg["users"]
    if not users:
        print("[列表] 尚无用户")
        return 0
    print(f"[列表] 共 {len(users)} 个用户:")
    print(f"  {'名字':<12} {'端口':<7} {'地址':<14} {'状态':<8} {'数据目录'}")
    print(f"  {'-'*12} {'-'*7} {'-'*14} {'-'*8} {'-'*30}")
    for name in sorted(users.keys()):
        u = users[name]
        running = _is_user_running(name)
        status = "●运行中" if running else "○已停止"
        pid_str = f"(PID={_read_pid(name)})" if running else ""
        host_display = u["host"]
        print(f"  {name:<12} {u['port']:<7} {host_display:<14} {status}{pid_str:<12} {u['data_dir']}")
    return 0


def cmd_start_all(args) -> int:
    cfg = _load_config()
    started = 0
    skipped = 0
    failed = 0
    for name in sorted(cfg["users"].keys()):
        if _is_user_running(name):
            print(f"[跳过] {name} 已在运行")
            skipped += 1
            continue
        user = cfg["users"][name]
        try:
            pid = _start_process(name, user)
        except RuntimeError as e:
            print(f"[失败] {name}: {e}")
            failed += 1
            continue
        # 等待就绪
        if _wait_user_ready(name, user, timeout=20):
            print(f"[启动] {name} PID={pid} 端口={user['port']}")
            started += 1
        else:
            if not _is_pid_running(pid):
                print(f"[失败] {name} 启动后退出，请检查日志: "
                      f"{user['data_dir']}\\output\\log\\server.log")
            else:
                print(f"[超时] {name} HTTP 未就绪（可能仍在加载索引）")
            _clear_pid(name)
            failed += 1
    print(f"\n[汇总] 启动 {started} 个, 跳过 {skipped} 个, 失败 {failed} 个")
    return 0 if failed == 0 else 1


def cmd_stop_all(args) -> int:
    cfg = _load_config()
    stopped = 0
    for name in sorted(cfg["users"].keys()):
        if _is_user_running(name):
            if _stop_process(name):
                print(f"[停止] {name}")
                stopped += 1
    print(f"\n[汇总] 停止 {stopped} 个")
    return 0


def cmd_remove(args) -> int:
    cfg = _load_config()
    name = args.name
    user = _get_user(cfg, name)
    if user is None:
        print(f"[错误] 用户不存在: {name}")
        return 1
    if name == MAIN_NAME:
        print(f"[错误] 不能删除主实例 {MAIN_NAME}")
        return 1
    # 先停止
    if _is_user_running(name):
        if not args.yes:
            ans = input(f"用户 {name} 正在运行，确认停止并删除？(y/N) ").strip().lower()
            if ans not in ("y", "yes"):
                print("[取消] 已取消")
                return 0
        _stop_process(name)
    # 删除数据目录
    data_dir = user["data_dir"]
    if not args.yes:
        ans = input(f"确认删除数据目录 {data_dir}？（不可恢复）(y/N) ").strip().lower()
        if ans not in ("y", "yes"):
            print("[取消] 已取消删除数据，但用户配置已保留")
            return 0
    import shutil
    if os.path.isdir(data_dir):
        shutil.rmtree(data_dir, ignore_errors=True)
    # 删除配置
    del cfg["users"][name]
    _save_config(cfg)
    _clear_pid(name)
    print(f"[删除] 用户 {name} 及其数据已删除")
    return 0


# ============================================================
#  命令行参数
# ============================================================

def build_parser():
    import argparse
    p = argparse.ArgumentParser(
        prog="manager",
        description="多实例用户管理器（常驻模式）。直接运行不带参数进入交互菜单。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # 不带子命令时进入交互菜单
    sub = p.add_subparsers(dest="command", required=False)

    # create
    pc = sub.add_parser("create", help="创建新用户")
    pc.add_argument("name", help="用户名（唯一）")
    pc.add_argument("--port", type=int, default=None,
                    help="指定端口（不指定则自动从 20000 起向后寻找可用端口）")
    pc.add_argument("--host", default=None,
                    help="监听地址（默认 127.0.0.1 仅本机；0.0.0.0 局域网可访问）")
    pc.set_defaults(func=cmd_create)

    # start
    ps = sub.add_parser("start", help="启动用户实例")
    ps.add_argument("name", help="用户名")
    ps.set_defaults(func=cmd_start)

    # stop
    pst = sub.add_parser("stop", help="停止用户实例")
    pst.add_argument("name", help="用户名")
    pst.set_defaults(func=cmd_stop)

    # restart
    pr = sub.add_parser("restart", help="重启用户实例")
    pr.add_argument("name", help="用户名")
    pr.set_defaults(func=cmd_restart)

    # list
    pl = sub.add_parser("list", help="列出所有用户及状态")
    pl.set_defaults(func=cmd_list)

    # start-all
    psa = sub.add_parser("start-all", help="启动所有未运行的用户实例")
    psa.set_defaults(func=cmd_start_all)

    # stop-all
    pspa = sub.add_parser("stop-all", help="停止所有运行中的用户实例")
    pspa.set_defaults(func=cmd_stop_all)

    # remove
    prm = sub.add_parser("remove", help="删除用户（含数据，不可恢复）")
    prm.add_argument("name", help="用户名")
    prm.add_argument("-y", "--yes", action="store_true", help="跳过确认")
    prm.set_defaults(func=cmd_remove)

    return p


# ============================================================
#  交互菜单模式（适合非计算机用户）
# ============================================================

def _get_local_ip() -> str:
    """获取本机局域网 IP（用于显示服务器访问地址）。"""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def _print_user_table():
    """显示所有用户及访问地址的表格。"""
    cfg = _load_config()
    users = sorted(cfg["users"].keys())
    local_ip = _get_local_ip()
    print()
    print("=" * 66)
    print("  用户列表及访问地址")
    print("=" * 66)
    print(f"  {'名字':<10} {'端口':<7} {'状态':<8} {'访问地址'}")
    print(f"  {'-'*10} {'-'*7} {'-'*8} {'-'*34}")
    for name in users:
        u = cfg["users"][name]
        running = _is_user_running(name)
        status = "●运行中" if running else "○已停止"
        if running:
            if u["host"] == "0.0.0.0":
                addr = f"http://{local_ip}:{u['port']}  (局域网)"
            else:
                addr = f"http://127.0.0.1:{u['port']}  (仅本机)"
        else:
            addr = "-"
        print(f"  {name:<10} {u['port']:<7} {status:<8} {addr}")
    print()
    print("  提示：首次启动需要初始化语义模型（约需 5-10 分钟），")
    print("        页面可能短暂无响应，请耐心等待。")
    print()


def _ensure_main_user(cfg: dict, host: str) -> dict:
    """确保 main 用户存在且 host 配置正确。返回更新后的 main 用户配置。"""
    if MAIN_NAME not in cfg["users"]:
        cfg["users"][MAIN_NAME] = {
            "port": _next_available_port(cfg),
            "host": host,
            "data_dir": MAIN_DATA_DIR,
        }
        os.makedirs(MAIN_DATA_DIR, exist_ok=True)
    else:
        cfg["users"][MAIN_NAME]["host"] = host
    _save_config(cfg)
    return cfg["users"][MAIN_NAME]


def _cleanup_on_exit(stop_all: bool = True):
    """退出管理器时的清理工作。

    stop_all=True  停止所有运行中的用户实例（本机模式/强制退出时）
    stop_all=False 仅停止本次启动的实例（服务器模式保留后台服务时）
    """
    if not stop_all:
        return
    cfg = _load_config()
    running = [n for n in cfg["users"] if _is_user_running(n)]
    if not running:
        return
    print(f"[清理] 正在停止 {len(running)} 个实例（可能需要数秒，请稍候）...",
          flush=True)
    stopped = []
    for name in running:
        if _stop_process(name):
            stopped.append(name)
            print(f"  ✓ 已停止 {name}", flush=True)
        else:
            print(f"  ✗ {name} 停止失败", flush=True)
    if stopped:
        print(f"[清理] 已停止 {len(stopped)} 个实例: {', '.join(stopped)}",
              flush=True)


def _run_local_mode():
    """本机模式：启动 main 实例，显示本地链接，然后挂起等待关闭。"""
    cfg = _load_config()
    main_user = _ensure_main_user(cfg, "127.0.0.1")
    started_by_us = False
    if _is_user_running(MAIN_NAME):
        print(f"\n[提示] 主实例已在运行")
    else:
        try:
            pid = _start_process(MAIN_NAME, main_user)
        except RuntimeError as e:
            print(f"[失败] {MAIN_NAME}: {e}")
            return 1
        started_by_us = True
        time.sleep(1.5)
        if not _is_pid_running(pid):
            print(f"[失败] 启动失败，请查看日志："
                  f"{main_user['data_dir']}\\output\\log\\server.log")
            return 1
    print()
    print("=" * 50)
    print("  ✓ 本机模式已启动")
    print("=" * 50)
    print(f"  访问地址：http://127.0.0.1:{main_user['port']}")
    print("-" * 50)
    print("  ⚠ 关闭此窗口将停止服务，运行期间请保持此窗口打开")
    print("  提示：首次启动需要初始化语义模型（约需 5-10 分钟），")
    print("        页面可能短暂无响应，请耐心等待。")
    print("=" * 50)
    sys.stdout.flush()
    # 挂起，等待用户关闭窗口或 Ctrl+C
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n[退出] 正在停止服务...")
    finally:
        if started_by_us:
            _cleanup_on_exit(stop_all=True)
        print("再见！")
    return 0


def _start_user_with_retry(name: str, cfg: dict, allow_port_change: bool = True) -> bool:
    """启动用户实例，失败时支持更换端口重试。

    Args:
        name: 用户名
        cfg: 配置字典（会被修改并保存）
        allow_port_change: 是否允许失败时换端口

    Returns:
        True = 启动成功，False = 最终失败
    """
    while True:
        user = cfg["users"][name]
        try:
            pid = _start_process(name, user)
        except RuntimeError as e:
            print(f"[失败] {name}: {e}")
            _log(f"[fail] {name} start error: {e}")
            if not allow_port_change:
                return False
            new_port = _prompt_new_port(name, user["port"], cfg)
            if new_port is None:
                return False
            cfg["users"][name]["port"] = new_port
            _save_config(cfg)
            continue  # 重试
        # 等待就绪（加载多个库的 faiss 索引可能需要 30-50 秒）
        if _wait_user_ready(name, user, timeout=60):
            host = user["host"]
            addr = "127.0.0.1" if host == "0.0.0.0" else host
            print(f"[启动] {name} PID={pid}  http://{addr}:{user['port']}")
            _log(f"[ok] {name} PID={pid} port={user['port']}")
            return True
        # 超时或崩溃
        if not _is_pid_running(pid):
            print(f"[失败] {name} 启动后退出，请查看日志: "
                  f"{user['data_dir']}\\output\\log\\server.log")
            _log(f"[fail] {name} exited before ready, check {user['data_dir']}\\output\\log\\server.log")
        else:
            print(f"[超时] {name} HTTP 未就绪（可能仍在加载索引）")
            _log(f"[timeout] {name} http not ready after 60s")
        _clear_pid(name)
        if not allow_port_change:
            return False
        # 询问是否换端口重试
        try:
            ans = input(f"  是否更换端口重试？(y/N) > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if ans not in ("y", "yes"):
            return False
        new_port = _prompt_new_port(name, user["port"], cfg)
        if new_port is None:
            return False
        cfg["users"][name]["port"] = new_port
        _save_config(cfg)


def _prompt_new_port(name: str, current_port: int, cfg: dict) -> Optional[int]:
    """提示用户输入新端口。返回新端口号或 None（取消）。"""
    auto_port = _next_available_port(cfg)
    print(f"  当前端口：{current_port}  建议端口：{auto_port}")
    try:
        s = input(f"  输入新端口（回车用建议值 {auto_port}，输入 q 取消）> ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if s.lower() in ("q", "quit", "exit"):
        return None
    if not s:
        return auto_port
    try:
        port = int(s)
    except ValueError:
        print("[错误] 端口必须是数字")
        return None
    if port < 1 or port > 65535:
        print("[错误] 端口范围 1-65535")
        return None
    if _is_port_in_use(port):
        print(f"[错误] 端口 {port} 也被占用")
        return None
    return port


def _interactive_create_server():
    """服务器模式下创建新用户：多用户已在服务内建，引导通过网页注册。"""
    print()
    print("[多用户] 账号体系已内建在单个服务中，无需再创建独立实例。")
    print("  1. 打开上方访问地址，点击右上角「注册」即可自助创建账号")
    print("     （首个注册用户自动成为管理员）")
    print("  2. 或使用命令行：python manager.py 或源码目录下")
    print("     python main.py user create <用户名> --password <密码>")
    print()


def _run_server_mode():
    """服务器模式：单实例 + 内建多用户（一个端口，登录后各看各的库）。

    旧版多实例（每用户一个端口/数据目录）不再默认启用；
    若 instances.json 中已存在旧多实例配置，仍可通过命令行命令管理。
    """
    cfg = _load_config()
    main_user = _ensure_main_user(cfg, "0.0.0.0")
    print()
    print("[启动] 正在启动检索服务（多用户内建，登录后各看各的库）...")
    if _is_user_running(MAIN_NAME):
        print(f"[跳过] {MAIN_NAME} 已在运行")
    else:
        try:
            pid = _start_process(MAIN_NAME, main_user)
        except RuntimeError as e:
            print(f"[失败] {MAIN_NAME}: {e}")
            return 1
        if _wait_user_ready(MAIN_NAME, main_user, timeout=60):
            print(f"[启动] {MAIN_NAME} PID={pid} 端口={main_user['port']}")
        else:
            if not _is_pid_running(pid):
                print(f"[失败] 启动后退出，请查看日志: "
                      f"{main_user['data_dir']}\\output\\log\\server.log")
            else:
                print(f"[超时] HTTP 未就绪（可能仍在加载索引）")
            _clear_pid(MAIN_NAME)
            return 1
    # 显示访问地址
    local_ip = _get_local_ip()
    host = main_user.get("host", "0.0.0.0")
    addr = f"http://{local_ip}:{main_user['port']}" if host == "0.0.0.0" else f"http://127.0.0.1:{main_user['port']}"
    print()
    print("=" * 60)
    print(f"  访问地址：{addr}")
    print("=" * 60)
    print("  · 未登录（游客）：查看/维护公共库（所有游客公用）")
    print("  · 注册/登录后：拥有自己的库，可把公共库复制到名下")
    print("  · 首个注册用户自动成为管理员")
    print("  关闭此窗口即停止服务。")
    print()
    # 管理循环
    while True:
        print("-" * 50)
        print("  [r] 刷新状态")
        print("  [q] 退出并停止服务")
        print("-" * 50)
        try:
            choice = input("请选择 > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[退出] 正在停止服务...")
            _cleanup_on_exit(stop_all=True)
            return 0
        if choice == "r":
            print(f"[状态] 服务运行中：{addr}")
        elif choice in ("q", "quit", "exit"):
            print("[退出] 正在停止服务...")
            _cleanup_on_exit(stop_all=True)
            return 0
        else:
            print("[错误] 无效选择，请输入 r / q")


def _interactive_loop():
    """交互菜单主循环：启动即用模式。"""
    print()
    print("=" * 60)
    print("  ⚠ 关闭此窗口将停止所有服务，运行期间请保持此窗口打开")
    print("=" * 60)
    print("  请选择运行模式：")
    print("  [1] 本机使用（仅自己用，直接打开主数据区）")
    print("  [2] 服务器模式（多用户内建：一个端口，登录后各看各的库）")
    print("-" * 60)
    print("  提示：首次启动需要初始化语义模型（约需 5-10 分钟），请耐心等待。")
    try:
        choice = input("请选择 > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n再见！")
        return 0
    if choice == "1":
        return _run_local_mode()
    elif choice == "2":
        return _run_server_mode()
    else:
        print("[错误] 无效选择")
        return 1


class _Namespace:
    """简单的命名空间对象，用于给命令函数传 args。"""
    yes = False


def main():
    parser = build_parser()
    args = parser.parse_args()
    # 无子命令 → 进入交互菜单
    if args.command is None:
        # 设置进程管理（Job Object + 控制台事件处理），
        # 确保关闭窗口或 Ctrl+C 时所有子进程同步终止
        _setup_process_management()
        return _interactive_loop()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
