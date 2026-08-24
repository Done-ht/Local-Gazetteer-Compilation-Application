"""子进程 OCR Worker 脚本（PaddleOCR v6 / PP-OCRv6）。

通过 stdin/stdout 与主进程通信，JSON 协议：
  请求：{"type": "init|ocr|detect|exit", ...}
  响应：{"type": "ready|result|error", ...}

主进程启动后发送 init 配置，然后循环发送 ocr/detect 请求，
收到 exit 或 stdin 关闭时退出。退出时 OS 回收所有内存。

3.x 变更（v5 起，v6 沿用）：
  - 推理调用用 predict(img) 取代 ocr(img, cls=False)
  - 结果格式：predict() 返回 [OCRResult, ...]，OCRResult 继承 dict，
    通过 page['rec_texts'] / page['rec_scores'] / page['dt_polys'] 访问
  - 序列化输出仍为 [[box, [text, conf]], ...]，保持与主进程 _parse_result 兼容
"""
import os
import sys
import json
import base64
import logging

# PaddlePaddle 内存分配器策略：必须在 import paddle 之前设置
# 官方推荐（PaddleOCR #11639 / 讨论 #14497）：naive_best_fit + eager_delete 系列
# 在 CPU 模式下控制内存增长优于 auto_growth（auto_growth 归还策略在 CPU 上偏弱，
# 项目自测 15 页仍涨到 1.4GB）。naive_best_fit 复用缓存块，配合 eager_delete 立即
# 回收中间张量，把 Paddle 的 Tensor 缓存复用控制在低位。
# 父进程 _build_env 已设置同名变量（优先级高于 setdefault），此处 setdefault 仅为
# 直接运行 worker 脚本（测试/打包）时提供兜底。
#
# 内存上限参数（关键防泄漏）：
#   FLAGS_fraction_of_cpu_memory_to_use：单个 paddle 实例最多使用系统内存比例
#     - 0.1：16GB 系统下每个子进程上限 1.6GB，强制 paddle 频繁释放
#   FLAGS_initial_cpu_memory_in_mb：初始预分配内存，降低启动内存峰值
os.environ.setdefault("FLAGS_allocator_strategy", "naive_best_fit")
os.environ.setdefault("FLAGS_eager_delete_scope", "True")
os.environ.setdefault("FLAGS_eager_delete_tensor_gb", "0.0")
os.environ.setdefault("FLAGS_fast_eager_deletion_mode", "True")
os.environ.setdefault("FLAGS_use_pinned_memory", "False")
os.environ.setdefault("FLAGS_fraction_of_cpu_memory_to_use", "0.1")
os.environ.setdefault("FLAGS_initial_cpu_memory_in_mb", "128")

# 关键：在 import paddleocr 之前禁用其日志输出到 stdout
# PaddleOCR 的 DEBUG/INFO 日志（如 "ppocr DEBUG: Namespace(...)"）会污染
# stdout JSON 通信通道，导致主进程 _recv() 解析 JSON 失败：
# "Expecting value: line 1 column 2 (char 1)"
# 解决方案：
#   1. 把 root logger 输出到 stderr（避免污染 stdout）
#   2. PaddleOCR 的 print 直接输出到 stderr
logging.basicConfig(
    stream=sys.stderr,
    level=logging.WARNING,
    format="[worker] %(asctime)s %(levelname)s %(name)s: %(message)s",
)

import numpy as np

_ocr = None  # PaddleOCR 实例（init 后创建）


def send(msg: dict) -> None:
    """发送 JSON 消息到 stdout。"""
    sys.stdout.buffer.write((json.dumps(msg) + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


def recv() -> dict:
    """从 stdin 读取一行 JSON 消息。"""
    line = sys.stdin.buffer.readline()
    if not line:
        return {"type": "exit"}
    return json.loads(line.decode("utf-8"))


def init(config: dict) -> None:
    """初始化 PaddleOCR 实例。

    config 由主进程的 _build_paddle_kwargs 构建，已是 v5 参数名：
      lang, ocr_version, use_textline_orientation, text_rec_score_thresh,
      text_det_unclip_ratio, text_det_box_thresh, text_det_limit_side_len,
      text_det_limit_type, text_detection_model_dir, text_recognition_model_dir
    """
    global _ocr
    from paddleocr import PaddleOCR
    # 修正 frozen 环境的 __file__ 路径
    try:
        from app.providers.paddle_local import _fix_paddleocr_file_path
        _fix_paddleocr_file_path()
    except Exception:
        pass
    _ocr = PaddleOCR(**config)
    send({"type": "ready"})


def _log(msg: str) -> None:
    """诊断日志输出到 stderr，绝不污染 stdout JSON 通道。"""
    print(f"[worker] {msg}", file=sys.stderr, flush=True)


def do_ocr(msg: dict) -> None:
    """执行 OCR 识别。"""
    global _ocr
    if _ocr is None:
        send({"type": "error", "message": "OCR 未初始化"})
        return
    # 解码图片
    img_bytes = base64.b64decode(msg["image"])
    shape = tuple(msg["shape"])
    dtype = np.dtype(msg["dtype"])
    img = np.frombuffer(img_bytes, dtype=dtype).reshape(shape)
    # 调用 OCR：v5 用 predict(img)，不用 ocr(img)（已废弃，会输出 DeprecationWarning）
    # 关键：OCR 推理期间临时把 fd 1（stdout）重定向到 fd 2（stderr），
    # 防止 PaddleOCR/PaddlePaddle C++ 层直接 write(1, ...) 污染 JSON 通信通道。
    # send() 是唯一允许写 stdout 的函数，推理结束后恢复 fd 1。
    saved_fd = os.dup(1)
    ocr_error = None
    result = None
    try:
        os.dup2(2, 1)  # stdout -> stderr
        result = _ocr.predict(img)
    except Exception as e:
        ocr_error = e
    finally:
        os.dup2(saved_fd, 1)  # 恢复 stdout
        os.close(saved_fd)
    if ocr_error is not None:
        send({"type": "error", "message": str(ocr_error)})
        return
    # 诊断日志：OCR 原始结果信息（全部输出到 stderr，不污染 stdout）
    try:
        if not result:
            _log(f"OCR 返回空结果, type={type(result)}")
        else:
            first_page = result[0]
            # v5: OCRResult 是 dict 子类，用 page.get('rec_texts') 取行数
            texts = first_page.get("rec_texts") if isinstance(first_page, dict) else None
            n_lines = len(texts) if texts is not None else 0
            _log(f"OCR 返回 {len(result)} 页, 首页 {n_lines} 行, type={type(first_page).__name__}")
            if n_lines > 0:
                _log(f"首行: text={texts[0]!r}")
    except Exception as e:
        _log(f"诊断日志异常: {e}")
    # 结果序列化（numpy 数组转 list）
    serializable = _serialize_result(result)
    _log(f"序列化后 {len(serializable)} 项")
    send({"type": "result", "result": serializable})


def do_detect(msg: dict) -> None:
    """仅检测文字框。"""
    global _ocr
    if _ocr is None:
        send({"type": "error", "message": "OCR 未初始化"})
        return
    img_bytes = base64.b64decode(msg["image"])
    shape = tuple(msg["shape"])
    dtype = np.dtype(msg["dtype"])
    img = np.frombuffer(img_bytes, dtype=dtype).reshape(shape)
    # 同 do_ocr：推理期间重定向 stdout 到 stderr，防止 C++ 层污染 JSON 通道
    saved_fd = os.dup(1)
    det_error = None
    box_list = []
    try:
        os.dup2(2, 1)
        if hasattr(_ocr, "text_detector"):
            ret = _ocr.text_detector(img)
            if ret is None:
                boxes = []
            elif isinstance(ret, tuple) and len(ret) >= 1:
                boxes = ret[0]
            else:
                boxes = ret
            for box in boxes:
                try:
                    points = [[float(p[0]), float(p[1])] for p in box]
                    box_list.append(points)
                except (TypeError, ValueError, IndexError):
                    continue
    except Exception as e:
        det_error = e
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
    if det_error is not None:
        send({"type": "error", "message": str(det_error)})
        return
    send({"type": "result", "boxes": box_list})


def _serialize_result(result) -> list:
    """将 PaddleOCR v5 结果序列化为 JSON 可传输格式。

    v5 返回格式：[OCRResult, ...]
      OCRResult 继承 dict，通过 page['rec_texts'] / page['rec_scores']
      / page['dt_polys'] 访问数据。

    输出格式（保持与主进程 _parse_result 兼容）：
      [[box, [text, conf]], ...]
      box: [[x, y], [x, y], [x, y], [x, y]]  （4 个顶点）
      text: str
      conf: float

    单张图片只有一页，取 result[0] 遍历其中的行。
    """
    out = []
    if not result:
        return out
    # 取第一页
    first_page = result[0]
    if not first_page:
        return out

    # v5: OCRResult 是 dict 子类
    if isinstance(first_page, dict):
        texts = first_page.get("rec_texts") or []
        scores = first_page.get("rec_scores") or []
        polys = first_page.get("dt_polys") or []
        for i, poly in enumerate(polys):
            text = texts[i] if i < len(texts) else ""
            conf = float(scores[i]) if i < len(scores) else 0.0
            # poly 是 numpy 数组或 list，转为 JSON 可序列化的 [[x,y], ...]
            try:
                box_list = [[float(p[0]), float(p[1])] for p in poly]
            except (TypeError, ValueError, IndexError):
                continue
            out.append([box_list, [text, conf]])
        return out

    # 回退：2.x 列表格式 [[box, (text, conf)], ...]
    for item in first_page:
        if not item or len(item) < 2:
            continue
        box = item[0]
        text_conf = item[1]
        # box 是 numpy 数组，转为 list
        try:
            box_list = [[float(p[0]), float(p[1])] for p in box]
        except (TypeError, ValueError, IndexError):
            continue
        # text_conf 是 (text, conf) 元组
        if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
            text = str(text_conf[0])
            conf = float(text_conf[1])
        else:
            continue
        out.append([box_list, [text, conf]])
    return out


def _get_memory_mb() -> float:
    """返回当前进程 RSS 内存（MB），失败返回 0。"""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


# 内存阈值（MB）：超过此值主动退出，让主进程重启子进程释放内存。
# 设置为 2500MB（2.5GB）：
#   - 正常页面 OCR 内存 < 800MB
#   - 复杂页面（大表格/多栏）可能到 1.5-2GB
#   - 超过 2.5GB 基本是 paddle 内存泄漏，继续处理只会越涨越高
#   - 主动退出比被 OOM kill 更优雅，主进程会自动重启子进程
# 旧值 1500MB 在复杂年鉴/表格页面容易误触发，导致子进程频繁重启、
# 主进程写 pipe 时遇到 Broken pipe，任务进度卡死。
_MEMORY_LIMIT_MB = 2500


def main() -> None:
    """主循环：接收请求，处理，响应。

    内存自保护：每次处理完请求后检查 RSS，超过 _MEMORY_LIMIT_MB 主动退出。
    主进程的 _recv 检测到子进程退出会触发 _restart_proc，新子进程接管后续页。
    这是一道兜底防线：即使 paddle 内存上限参数失效，也不会出现单进程 5GB 的情况。
    """
    page_count = 0
    while True:
        try:
            msg = recv()
        except Exception as e:
            break
        msg_type = msg.get("type", "")
        if msg_type == "exit":
            break
        elif msg_type == "init":
            try:
                init(msg.get("config", {}))
            except Exception as e:
                send({"type": "error", "message": f"初始化失败: {e}"})
                break
        elif msg_type == "ocr":
            do_ocr(msg)
            page_count += 1
            # 每页检查内存阈值（防止单页内存暴涨超过限制）
            # 复杂版面/大表格单页可能涨 200-400MB，需及时检测
            mem_mb = _get_memory_mb()
            if mem_mb > _MEMORY_LIMIT_MB:
                _log(f"内存超限({mem_mb:.0f}MB > {_MEMORY_LIMIT_MB}MB)，主动退出等待重启")
                # 退出前发送通知：让主进程知道是"内存超限主动退出"而非"崩溃"
                # 主进程 _recv 收到此消息后可据此决定是否重试（特定页面可能持续触发内存超限）
                try:
                    send({"type": "exit", "reason": "memory_limit",
                          "mem_mb": round(mem_mb, 1), "page_count": page_count})
                except Exception:
                    pass
                break
            # 每 5 页输出一次内存日志（与 batch_size=5 对齐，重启前输出最终内存）
            if page_count % 5 == 0:
                _log(f"内存监控: 已处理 {page_count} 页, RSS={mem_mb:.0f}MB")
        elif msg_type == "detect":
            do_detect(msg)
        else:
            send({"type": "error", "message": f"未知消息类型: {msg_type}"})
    # 清理
    global _ocr
    _ocr = None


if __name__ == "__main__":
    main()
