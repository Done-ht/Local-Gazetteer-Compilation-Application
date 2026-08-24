"""server-paddle 服务端入口。

启动流程:
  1. 加载配置，检测 PaddleOCR 环境
  2. 预创建 N 个 PaddleOCR 实例（N=最大并发数，约 30-60 秒）
  3. 启动 FastAPI 服务，监听 0.0.0.0:8000
  4. 控制台打印局域网访问地址
  5. 自动弹出浏览器打开引导页（显示局域网地址 + 二维码）

PaddleOCR 模型实例非线程安全，多并发共享同一实例会崩溃或结果错乱。
启动时预创建与最大并发数等量的独立实例组成实例池，每个并发槽位
持有独立实例，互不干扰。
"""
from __future__ import annotations

import logging
import os
import socket
import sys
import threading
import webbrowser

# ----------------------------------------------------------------------
# 子进程 OCR Worker 模式检测
# 打包后 sys.executable 是 server-paddle.exe，subprocess.Popen 启动子进程时
# 会执行整个 main.py（包括端口提示、uvicorn 启动等），导致子进程永远不响应
# JSON 协议，触发 300s 超时。
# 解决：subprocess_ocr.py 在子进程环境变量中设置 _OCR_WORKER_MODE=1，
# 主程序检测到后直接跳转执行 worker 主循环，跳过所有服务启动逻辑。
# 必须在最早期检测，避免执行任何 main.py 的初始化代码（如日志配置、端口提示）。
# ----------------------------------------------------------------------
if os.environ.get("_OCR_WORKER_MODE") == "1":
    from app.providers._worker_ocr import main as _worker_main
    _worker_main()
    sys.exit(0)

# ----------------------------------------------------------------------
# 子进程版面分析 Worker 模式检测
# 与 _OCR_WORKER_MODE 同理：SubprocessLayoutPool 启动子进程时设置
# _LAYOUT_WORKER_MODE=1，主程序检测到后跳转执行 _worker_layout.py 主循环，
# 跳过服务启动逻辑，避免污染 JSON 通信通道。
# 必须在 _OCR_WORKER_MODE 检测之后、任何 main.py 初始化代码之前检测。
# ----------------------------------------------------------------------
if os.environ.get("_LAYOUT_WORKER_MODE") == "1":
    from app.providers._worker_layout import main as _layout_worker_main
    _layout_worker_main()
    sys.exit(0)

# ----------------------------------------------------------------------
# CPU 线程数：不再限制（用户自行选择并发数）
# 之前设置 OMP_NUM_THREADS=1 把每实例线程压到 1，拖慢 OCR
# 现在让 PaddlePaddle 用默认线程数（=CPU 核数），通过 max_concurrent 控制总负载
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# PaddlePaddle 内存分配器策略：必须在 import paddle 之前设置
# 官方推荐（PaddleOCR #11639 / 讨论 #14497）：naive_best_fit + eager_delete 系列
# 在 CPU 模式下控制内存增长优于 auto_growth（auto_growth 归还策略在 CPU 上偏弱，
# 项目自测 15 页仍涨到 1.4GB）。配合 eager_delete 立即回收中间张量。
# 与子进程 worker（_worker_ocr.py / _worker_layout.py）的 _build_env 保持一致。
# 注：子进程模式下主进程跳过 PaddleOCR 实例池创建，此处仅为非子进程模式兜底。
os.environ.setdefault("FLAGS_allocator_strategy", "naive_best_fit")
os.environ.setdefault("FLAGS_eager_delete_scope", "True")
os.environ.setdefault("FLAGS_eager_delete_tensor_gb", "0.0")
os.environ.setdefault("FLAGS_fast_eager_deletion_mode", "True")
os.environ.setdefault("FLAGS_use_pinned_memory", "False")
os.environ.setdefault("FLAGS_fraction_of_cpu_memory_to_use", "0.1")
os.environ.setdefault("FLAGS_initial_cpu_memory_in_mb", "128")


# ----------------------------------------------------------------------
# 日志配置
# ----------------------------------------------------------------------
def _setup_logging() -> None:
    """配置日志：应用 INFO，第三方库 WARNING。

    只输出到日志文件（ocr_service.log），不输出到控制台。
    日志文件位于 output/log/ 目录下。

    日志文件使用 RotatingFileHandler 限制大小：单文件 10MB，保留 5 个备份，
    避免长期运行导致日志文件无限增长占用磁盘。

    注意：不使用 logging.basicConfig，因为它在根 logger 已有 handler 时
    是空操作。这里直接操作根 logger，确保 FileHandler 一定被添加。
    """
    from logging.handlers import RotatingFileHandler
    from app.utils.task_dirs import service_log_path, LOG_DIR

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = service_log_path()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    # 清除可能已存在的 handler，避免重复输出或被 uvicorn 预先配置
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    # 轮转日志：单文件 10MB，保留 5 个备份，总上限 60MB
    # 不添加 StreamHandler：日志只写文件，不输出到控制台
    file_handler = RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)
    # paddleocr/paddle 导入时会覆盖 root 为 WARNING，显式给 app logger 设 INFO
    # 保证 app 下所有子 logger（含版面诊断日志）的 INFO 不被过滤
    logging.getLogger("app").setLevel(logging.INFO)

    # 抑制第三方库的冗余日志
    # uvicorn.access 默认 INFO 会记录每个 HTTP 请求，刷屏严重且淹没应用日志
    # 提升到 WARNING：仅保留 5xx 错误请求，正常 200/4xx 不再记录
    for noisy in ("ppocr", "paddle", "paddlex", "matplotlib", "PIL",
                  "uvicorn.access", "httpcore", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # 立即写一条日志，验证文件写入是否正常
    logging.getLogger(__name__).info("日志系统初始化完成，日志文件: %s", log_path)


# ----------------------------------------------------------------------
# 局域网 IP 探测
# ----------------------------------------------------------------------
# VPN/虚拟网卡名称关键词（小写匹配）。命中即视为非真实物理网卡，跳过。
# VPN 虚拟网卡会接管外网路由，使 UDP 探测出口 IP 的方法错误返回 VPN 网卡 IP
# （手机不在 VPN 内，无法访问该 IP）。按网卡名过滤是主防线。
_VIRTUAL_ADAPTER_KEYWORDS = (
    "vpn", "virtual", "vmware", "virtualbox", "tap", "tun",
    "hyper-v", "vethernet", "loopback", "isatap", "teredo",
    "6to4", "docker", "wsl", "nat",
)


def _is_contiguous_netmask(val: str) -> bool:
    """判断 IPv4 字符串是否为合法连续子网掩码（本地化无关）。

    ipconfig 的掩码行标签随系统语言变化（英文 "Subnet Mask"、中文 "子网掩码"），
    不能靠关键词识别。改用值本身判断：合法连续掩码的 32 位形式必为 1*0*，
    其按位取反为 0*1*，满足 (inv & (inv+1)) == 0。
    """
    try:
        octets = [int(o) for o in val.split(".")]
    except ValueError:
        return False
    if len(octets) != 4 or any(o < 0 or o > 255 for o in octets):
        return False
    n = (octets[0] << 24) | (octets[1] << 16) | (octets[2] << 8) | octets[3]
    inv = (~n) & 0xFFFFFFFF
    return (inv & (inv + 1)) == 0


def _netmask_to_prefix(mask: str) -> int:
    """子网掩码转前缀长度；非法时回退 24。"""
    if not _is_contiguous_netmask(mask):
        return 24
    octets = [int(o) for o in mask.split(".")]
    n = (octets[0] << 24) | (octets[1] << 16) | (octets[2] << 8) | octets[3]
    return bin(n).count("1")


def _detect_lan_ip_from_adapters() -> str:
    """枚举真实物理网卡的 IPv4 地址，返回最佳局域网 IP；无候选返回空串。

    Windows 解析 ipconfig：按结构识别网卡块（顶格、含空格、以冒号结尾的行），
    块内第一个"非掩码"IPv4 为该网卡 IP，第一个"合法连续掩码"为子网掩码。
    完全不依赖本地化标签关键词（中英文系统均可）。
    """
    import re
    import subprocess

    if not sys.platform.startswith("win"):
        return ""

    try:
        # encoding="oem"：用控制台 OEM 代码页解码，正确显示中文本地化标签
        # （ASCII 网卡名和 IP 不受影响）；CREATE_NO_WINDOW 避免弹出黑窗
        out = subprocess.run(
            ["ipconfig"], capture_output=True, encoding="oem",
            timeout=5, creationflags=0x08000000,
        ).stdout
    except Exception:
        return ""

    ipv4_re = re.compile(r"(\d{1,3}(?:\.\d{1,3}){3})")
    candidates: list[tuple[str, str, int]] = []  # (网卡名, ip, 前缀长度)
    cur_name = ""
    pending_ip = None
    pending_mask = None
    for line in out.splitlines():
        stripped = line.strip()
        # 网卡标题行：顶格、含空格、以冒号结尾
        if line and not line[0].isspace() and stripped.endswith(":") and " " in stripped:
            if cur_name and pending_ip is not None:
                prefix = _netmask_to_prefix(pending_mask) if pending_mask else 24
                candidates.append((cur_name, pending_ip, prefix))
            cur_name = stripped[:-1]
            pending_ip = None
            pending_mask = None
            continue
        if not cur_name:
            continue
        m = ipv4_re.search(stripped)
        if not m or ":" not in stripped:
            continue
        val = m.group(1)
        if _is_contiguous_netmask(val):
            if pending_mask is None:
                pending_mask = val
        else:
            # 块内第一个非掩码 IPv4 为本网卡 IP（IPv4 地址行总在默认网关之前）
            if pending_ip is None:
                pending_ip = val
    if cur_name and pending_ip is not None:
        prefix = _netmask_to_prefix(pending_mask) if pending_mask else 24
        candidates.append((cur_name, pending_ip, prefix))

    # 过滤 + 打分挑选
    scored: list[tuple[int, str]] = []
    for name, ip, prefix in candidates:
        if ip.startswith("127.") or ip.startswith("169.254."):
            continue
        lname = name.lower()
        if any(k in lname for k in _VIRTUAL_ADAPTER_KEYWORDS):
            continue
        # /32 非 loopback 多为 VPN 点对点隧道，跳过（名称未命中时的兜底）
        if prefix == 32 and not ip.startswith("127."):
            continue
        if ip.startswith("192.168."):
            score = 100
        elif ip.startswith("10."):
            score = 80
        elif ip.startswith("172."):
            try:
                second = int(ip.split(".")[1])
                score = 60 if 16 <= second <= 31 else 10
            except (IndexError, ValueError):
                score = 10
        else:
            score = 10
        scored.append((score, ip))
    if not scored:
        return ""
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][1]


def _detect_lan_ip_by_udp() -> str:
    """兜底：UDP 连接公共 IP 探测出口 IP（不实际发包）。

    VPN 开启时会返回 VPN 虚拟网卡 IP（手机无法访问），仅作兜底。
    """
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 不实际发包，仅让 OS 决定出口 IP
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        if s is not None:
            try:
                s.close()
            except Exception:
                pass


def get_lan_ip() -> str:
    """获取本机局域网 IPv4 地址（用于生成可被其他设备访问的 URL）。

    优先策略：枚举所有网卡的 IPv4 地址，跳过 VPN/虚拟/loopback 网卡，从真实
    物理网卡（WiFi/以太网）中选取私网地址。这样即使开启 VPN（VPN 会接管外网
    路由，使旧的"连 8.8.8.8 探测出口"法返回 VPN 虚拟网卡 IP），也能正确返回
    真实 WiFi IP，保证手机能访问。

    回退：网卡枚举失败时用 UDP 探测出口 IP（VPN 开启时返回 VPN IP，仅兜底）。
    全部失败回退 127.0.0.1。
    """
    ip = _detect_lan_ip_from_adapters()
    if ip:
        return ip
    return _detect_lan_ip_by_udp()


# ----------------------------------------------------------------------
# 环境检测
# ----------------------------------------------------------------------
def _mode_display(mode: str) -> str:
    """运行模式的显示名（本地模式 / 联网模式）。"""
    return "本地模式" if mode == "paddle" else "联网模式"


def _check_env(mode: str = "paddle") -> None:
    """打印环境信息，便于诊断。"""
    print("=" * 64)
    print("  OCR 识别服务 启动中")
    print(f"  运行模式: {_mode_display(mode)}")
    print(f"  Python: {sys.executable}")
    print(f"          {sys.version.split()[0]}")
    root = os.path.dirname(os.path.abspath(__file__))
    print(f"  项目目录: {root}")
    from app.utils.config import config_path
    print(f"  配置文件: {config_path()}")
    if mode == "paddle":
        try:
            import paddleocr
            import paddle
            print(f"  PaddleOCR: {getattr(paddleocr, '__version__', '?')}"
                  f"  /  Paddle: {getattr(paddle, '__version__', '?')}")
            ver_major = int(str(getattr(paddleocr, "__version__", "2")).split(".")[0])
            if ver_major < 3:
                print("  [警告] PaddleOCR 版本过低，请升级: pip install paddlepaddle==3.2.0 paddleocr==3.3.2")
        except ImportError:
            print("  [警告] 未检测到 paddleocr/paddlepaddle，请先运行 setup.bat 安装依赖")
    else:
        try:
            import requests, fitz, docx, PIL  # noqa: F401
        except ImportError as e:
            print(f"  [警告] 缺少联网模式依赖: {e}（请运行 setup.bat 安装依赖）")
    print("=" * 64)


# ----------------------------------------------------------------------
# 构建应用
# ----------------------------------------------------------------------
def _build_app(lan_url: str, mode: str = "paddle"):
    """构建 FastAPI 应用。

    mode=paddle：本地 PaddleOCR 引擎（原有逻辑）+
                 共用用户系统（登录门禁 + 任务归属用户）
    mode=xfyun ：讯飞云端 OCR 引擎（从 ocr-web 移植）+ 共用用户系统
    两个模式都挂载共用登录/注册/用户管理路由，但引擎路由互斥——
    本次启动挂载哪个模式的路由，就只能是哪个软件，重启前不能切换。
    """
    from fastapi import FastAPI
    from app.auth.routes import router as auth_router

    # ------------------------------------------------------------------
    # xfyun 模式：不加载任何 paddle 模块（避免拖慢启动、占用内存）
    # ------------------------------------------------------------------
    if mode == "xfyun":
        from app.xfyun.routes import router as xfyun_router
        from app.xfyun import service as xf_service
        from app.utils.config import load_config

        cfg = load_config()
        app = FastAPI(title="OCR 识别服务（联网模式）")
        app.include_router(auth_router)
        app.include_router(xfyun_router)
        # 恢复上次运行遗留的排队/中断任务（页级断点续跑）
        try:
            xf_service.resume_stale_tasks(cfg)
        except Exception as e:
            print(f"  [警告] 恢复遗留任务失败: {e}")
        print()
        print("  联网模式就绪：上传 pdf / docx / 图片，调用云端 OCR")
        return app, None

    # ------------------------------------------------------------------
    # paddle 模式（原有逻辑）
    # ------------------------------------------------------------------
    from app.api.concurrency import ConcurrencyManager
    from app.api.routes import create_app
    from app.api.tasks import TaskManager
    from app.core.layout import init_layout_pool, layout_pool_size
    from app.core.pipeline import Pipeline
    from app.providers.paddle_local import init_ocr_pool, pool_size
    from app.utils.config import load_config

    cfg = load_config()
    max_concurrent = int(cfg.get("max_concurrent", 3))
    concurrency = ConcurrencyManager(max_concurrent=max_concurrent)

    # 不再限制 CPU 线程数（用户自行选择并发数，让 PaddleOCR 用满 CPU）
    # 之前的 _limit_cpu_threads 会把每实例线程数压到 1-2 个，拖慢 OCR
    # 现在让 paddle 用默认线程数（=CPU 核数），通过 max_concurrent 控制总负载

    # 预创建 N 个独立 PaddleOCR 实例（N=最大并发数），避免多线程共享崩溃。
    # 每个并发槽位对应一个独立实例，互不干扰。
    #
    # 内存优化（关键）：
    #   子进程模式下（use_subprocess_ocr=true，默认），OCR 推理走独立子进程，
    #   主进程的 _OCR_POOL 实例不会被使用。此时跳过 init_ocr_pool 可避免
    #   主进程白白加载 N 个 PaddleOCR 模型（每个约 200MB），显著降低主进程内存。
    #   子进程的实例池由 SubprocessOCRProvider 在首次任务时懒构建。
    #
    #   PPStructureV3（版面分析）在子进程模式下也在独立子进程中运行，
    #   主进程同样不需要实例池（与 OCR 子进程模式对齐）。
    pcfg = cfg.get("paddle", {})
    paddle_config = {
        "lang": pcfg.get("lang", "ch"),
        "use_gpu": pcfg.get("use_gpu", False),
        "ocr_version": pcfg.get("ocr_version", "PP-OCRv6"),
        "det_model_dir": pcfg.get("det_model_dir", ""),
        "rec_model_dir": pcfg.get("rec_model_dir", ""),
        "drop_score": pcfg.get("drop_score", 0.0),
        "det_db_unclip_ratio": pcfg.get("det_db_unclip_ratio", 1.8),
        "det_db_box_thresh": pcfg.get("det_db_box_thresh", 0.5),
    }
    use_subprocess = cfg.get("use_subprocess_ocr", True)
    if use_subprocess:
        # 子进程模式：跳过主进程 PaddleOCR 实例池创建
        # OCR 推理由独立子进程完成，主进程不需要 PaddleOCR 实例
        print(f"  OCR 模式：子进程隔离（跳过主进程 PaddleOCR 实例池创建）")
        print(f"  子进程将在首次任务时启动，每 {cfg.get('subprocess_batch_size', 5)} 页自动重启释放内存")
    else:
        # 同进程模式：主进程内创建 PaddleOCR 实例池
        print(f"  正在初始化 {max_concurrent} 个 OCR 实例（约需 30-60 秒）...")
        try:
            init_ocr_pool(max_concurrent, paddle_config)
        except Exception as e:
            # 实例池创建失败要优雅降级：不阻断服务启动，首次任务时会再尝试
            print(f"  [警告] OCR 实例池初始化失败: {e}")
        created = pool_size()
        if created > 0:
            print(f"  OCR 实例池就绪：{created}/{max_concurrent} 个实例")
        else:
            print("  [警告] 未创建任何 OCR 实例，请确认 paddleocr 已安装")

    # 预创建 N 个独立 PPStructureV3 实例（版面分析），与 OCR 池同样规模。
    # PPStructureV3 同样非线程安全，多并发共享会崩溃，需独立实例池。
    # 仅在配置启用版面分析时初始化；不可用时优雅降级（版面分析禁用，
    # 不阻断服务启动，回退整页 OCR）。
    #
    # 内存优化（阶段4）：
    #   子进程版面分析模式下（use_subprocess_layout=true，默认），PPStructureV3
    #   推理走独立子进程，主进程不需要 PPStructureV3 实例。此时跳过 init_layout_pool
    #   可避免主进程白白加载 N 个 PPStructureV3 模型（每个约 150MB），显著降低
    #   主进程内存。子进程的版面分析池由 SubprocessLayoutPool 在首次任务时懒构建。
    if pcfg.get("enable_layout", True):
        use_subprocess_layout = cfg.get("use_subprocess_layout", True)
        if use_subprocess_layout:
            # 子进程模式：跳过主进程 PPStructureV3 实例池创建
            # 版面分析推理由独立子进程完成，主进程不需要 PPStructureV3 实例
            print(f"  版面分析模式：子进程隔离（跳过主进程 PPStructureV3 实例池创建）")
            print(f"  子进程将在首次任务时启动，每 {cfg.get('subprocess_layout_batch_size', 5)} 页自动重启释放内存")
        else:
            # 同进程模式：主进程内创建 PPStructureV3 实例池
            print(f"  正在初始化 {max_concurrent} 个 PPStructureV3（版面分析）实例...")
            try:
                init_layout_pool(
                    max_concurrent,
                    lang=pcfg.get("lang", "ch"),
                    layout_score_thresh=0.3,
                    use_table_recognition=bool(pcfg.get("use_table_recognition", True)),
                )
            except Exception as e:
                # 版面分析池创建失败要优雅降级：不阻断服务启动
                print(f"  [警告] 版面分析实例池初始化失败: {e}")
            layout_created = layout_pool_size()
            if layout_created > 0:
                print(
                    f"  版面分析实例池就绪：{layout_created}/{max_concurrent} 个实例"
                )
            else:
                print(
                    "  [警告] 未创建任何版面分析实例，将降级为整页 OCR"
                )
    print()

    # Pipeline 工厂：懒加载，首次任务时构建（实例池已在上方预创建）
    task_manager = TaskManager(
        concurrency=concurrency,
        pipeline_factory=Pipeline,
        output_dir=cfg.get("output_dir", "") or None,
        keep_recent=30,
    )
    app = create_app(concurrency, task_manager, lan_url=lan_url)
    # 挂载共用用户系统路由（登录/注册/用户管理）
    app.include_router(auth_router)
    return app, task_manager


# ----------------------------------------------------------------------
# 浏览器自动打开引导页
# ----------------------------------------------------------------------
def _open_browser_delayed(url: str, delay: float = 1.5) -> None:
    """延迟打开浏览器，等服务启动后再打开。"""
    def _open():
        import time
        time.sleep(delay)
        try:
            webbrowser.open(url)
        except Exception:
            pass
    threading.Thread(target=_open, daemon=True).start()


# ----------------------------------------------------------------------
# 端口占用检测
# ----------------------------------------------------------------------
def _is_port_in_use(port: int) -> bool:
    """检测端口是否被占用（尝试 bind，失败即表示占用）。"""
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        s.bind(("0.0.0.0", port))
        return False
    except OSError:
        return True
    finally:
        if s is not None:
            try:
                s.close()
            except Exception:
                pass


# ----------------------------------------------------------------------
# 启动前端口设置交互
# ----------------------------------------------------------------------
def _prompt_port(default_port: int) -> int:
    """启动前交互式确认服务端口。

    流程：
      1. 显示当前默认端口（从 config.json 读取，默认 8070）
      2. 用户直接回车使用默认值，或输入新端口号
      3. 校验端口范围（1024-65535）和占用情况
      4. 输入 q 退出程序

    返回: 用户确认的端口号
    """
    print()
    print("=" * 64)
    print("  服务端口设置")
    print("=" * 64)
    print(f"  当前默认端口：{default_port}")
    print("  - 直接回车使用默认端口")
    print("  - 或输入新端口号（1024-65535）")
    print("  - 输入 q 退出程序")
    print()

    while True:
        try:
            raw = input(f"  请输入端口号 [默认 {default_port}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            # 非交互环境（如被重定向）或 Ctrl+C：直接使用默认值
            print()
            print(f"  使用默认端口：{default_port}")
            return default_port

        if not raw:
            # 回车：使用默认值
            chosen = default_port
        elif raw.lower() == "q":
            print("  用户取消，程序退出。")
            sys.exit(0)
        else:
            try:
                chosen = int(raw)
            except ValueError:
                print("  [错误] 端口号必须是整数，请重新输入。")
                continue
            if not (1024 <= chosen <= 65535):
                print(f"  [错误] 端口必须在 1024-65535 之间，当前输入 {chosen}，请重新输入。")
                continue

        # 端口占用检测
        if _is_port_in_use(chosen):
            print(f"  [错误] 端口 {chosen} 已被占用，请更换端口或关闭占用程序。")
            # 若是默认端口被占用，仍允许用户输入新端口
            continue

        # 二次确认
        print(f"  已选择端口：{chosen}")
        print()
        return chosen


# ----------------------------------------------------------------------
# 启动参数
# ----------------------------------------------------------------------
def _parse_args():
    """命令行参数：--mode 可覆盖 config.json 中的模式（便于脚本/测试）。"""
    import argparse
    parser = argparse.ArgumentParser(description="OCR 识别服务（本地 / 联网 双模式）")
    parser.add_argument(
        "--mode", choices=["paddle", "xfyun"], default=None,
        help="运行模式覆盖（默认取 config.json 的 mode，首次启动交互式选择）",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------
# 模式交互选择（每次启动都询问）
# ----------------------------------------------------------------------
def _prompt_mode(cfg: dict, override: str = None) -> str:
    """询问运行模式并持久化到 config.json。

    规则：
      1. --mode 命令行参数优先（覆盖，不询问）
      2. 其他情况：每次启动都在控制台询问选择 paddle / xfyun

    返回: "paddle" 或 "xfyun"
    """
    if override in ("paddle", "xfyun"):
        return override

    print()
    print("=" * 64)
    print("  请选择本次运行模式")
    print("=" * 64)
    print(f"  1) {_mode_display('paddle')} —— 本地识别（纯 CPU，无需联网）")
    print(f"  2) {_mode_display('xfyun')} —— 云端识别（需联网，消耗云端 API 额度）")
    print()
    print("  【提示】每次启动都会询问；如需脚本/静默运行，可用 --mode 参数指定。")
    print()
    while True:
        try:
            raw = input("  请选择模式 [1/2]（默认 1）: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            # 非交互环境（重定向/无控制台）：默认 paddle，避免卡死
            print()
            print("  [提示] 非交互环境，默认使用 paddle 模式")
            raw = "1"
        if raw in ("", "1", "paddle", "p"):
            mode = "paddle"
        elif raw in ("2", "xfyun", "x"):
            mode = "xfyun"
        else:
            print("  [错误] 请输入 1 或 2")
            continue
        break

    # 持久化到 config.json（仅作记录，不影响下次启动的询问）
    from app.utils.config import save_config
    if str(cfg.get("mode") or "").strip().lower() != mode:
        cfg["mode"] = mode
        try:
            save_config(cfg)
            print(f"  已记录运行模式 {mode!r} 到 config.json")
        except Exception as e:
            print(f"  [警告] 保存模式配置失败: {e}")
    return mode


# ----------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------
def main() -> None:
    args = _parse_args()
    _setup_logging()

    # 先确定运行模式（首次启动交互式选择并持久化）
    from app.utils.config import load_config, save_config
    cfg = load_config()
    mode = _prompt_mode(cfg, override=args.mode)

    _check_env(mode)

    # 先读取配置中的端口作为默认值
    default_port = int(cfg.get("port", 8070))

    # 启动前交互式确认端口（不运行服务直到端口确认）
    port = _prompt_port(default_port)

    # 端口与配置不同时保存到 config.json，下次启动作为默认值
    if port != default_port:
        cfg["port"] = port
        try:
            save_config(cfg)
            print(f"  已保存新端口 {port} 到 config.json")
        except Exception as e:
            print(f"  [警告] 保存端口配置失败: {e}")
        print()

    host = "0.0.0.0"
    lan_ip = get_lan_ip()
    lan_url = f"http://{lan_ip}:{port}"
    local_url = f"http://127.0.0.1:{port}"

    # 启动里程碑事件写入进度日志，便于用户在 ocr_progress.log 中
    # 看到完整的应用运行轨迹（启动/任务/页面/完成）
    try:
        from app.utils.progress_log import log_progress
        log_progress(f"=== 服务启动 (模式: {mode}, 访问地址: {lan_url}) ===")
    except Exception:
        pass

    print()
    print(f"  运行模式：{_mode_display(mode)}")
    print("  局域网访问地址（手机/平板扫码或输入）:")
    print(f"    {lan_url}")
    print("  本机访问地址:")
    print(f"    {local_url}")
    print()
    print("  按 Ctrl+C 停止服务")
    print()

    # 构建 FastAPI 应用
    app, task_manager = _build_app(lan_url, mode)

    # 延迟打开浏览器到引导页（本机）
    _open_browser_delayed(local_url, delay=1.5)

    # 启动 uvicorn
    # 关键：log_config 必须包含完整的 formatters/handlers/root 配置。
    # uvicorn 启动时会调用 logging.config.dictConfig(log_config)，
    # 如果 log_config 缺少 root 键，dictConfig 会清除 root logger 的现有 handlers
    # （包括 _setup_logging() 添加的 FileHandler），导致应用日志丢失。
    #
    # 不配置 console handler：日志只写文件，不输出到控制台。
    # log_level="info"：uvicorn 会用此参数设置根 logger 级别为 INFO。
    # uvicorn.access 的 INFO 日志通过 log_config 中 level=WARNING 抑制。
    try:
        import uvicorn
        from app.utils.task_dirs import service_log_path
        log_path = service_log_path()
        uvicorn.run(
            app, host=host, port=port, log_level="info",
            access_log=False,
            log_config={
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {
                    "default": {
                        "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        "datefmt": "%Y-%m-%d %H:%M:%S",
                    },
                },
                "handlers": {
                    "file": {
                        "class": "logging.handlers.RotatingFileHandler",
                        "filename": log_path,
                        "maxBytes": 10485760,
                        "backupCount": 5,
                        "encoding": "utf-8",
                        "formatter": "default",
                    },
                },
                "root": {
                    "level": "INFO",
                    "handlers": ["file"],
                },
                "loggers": {
                    "uvicorn": {"level": "INFO"},
                    "uvicorn.error": {"level": "INFO"},
                    "uvicorn.access": {"level": "WARNING"},
                    "ppocr": {"level": "WARNING"},
                    "paddle": {"level": "WARNING"},
                    "paddlex": {"level": "WARNING"},
                    "matplotlib": {"level": "WARNING"},
                    "PIL": {"level": "WARNING"},
                    "httpcore": {"level": "WARNING"},
                    "httpx": {"level": "WARNING"},
                },
            },
        )
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在关闭服务...")
    except Exception as e:
        # 捕获 uvicorn 运行时的致命错误（如子进程崩溃导致的主进程退出）
        # 写入文件便于诊断
        import traceback
        try:
            from app.utils.progress_log import log_progress
            log_progress(f"!!! 服务致命错误: {e}")
        except Exception:
            pass
        print(f"\n服务遇到致命错误: {e}")
        traceback.print_exc()
    finally:
        try:
            if task_manager is not None:
                task_manager.shutdown()
        except Exception:
            pass
        try:
            from app.utils.progress_log import log_progress
            log_progress("=== 服务停止 ===")
        except Exception:
            pass
        print("服务已停止")


if __name__ == "__main__":
    main()
