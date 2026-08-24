"""子进程版面分析 Worker 脚本（PaddleOCR v6 / PPStructureV3）。

通过 stdin/stdout 与主进程通信，JSON 协议：
  请求：{"type": "init|analyze|exit", ...}
  响应：{"type": "ready|result|error", ...}

主进程启动后发送 init 配置（PPStructureV3 参数），然后循环发送 analyze 请求
（携带图片 base64 + 形状），子进程调用 PPStructureV3.predict() 推理后将版面区域
列表（含 type / bbox / 表格 HTML）序列化为 JSON 返回。收到 exit 或 stdin 关闭时退出。

3.x API 要点（与 2.x PPStructure 的差异）：
  - 类名：PPStructure → PPStructureV3
  - 构造参数全部更名：
      layout_model_dir → layout_detection_model_dir
      table_model_dir  → wired_table_structure_recognition_model_dir 等
      layout=True/table=True/ocr=False 已移除，改用 use_table_recognition 等开关
      show_log 已移除（3.x 严格校验未知参数）
  - 推理调用：predict(img) 取代 __call__(img)
    返回 [StructureV3Result, ...]，取 [0] 为单页结果
  - 结果格式：
      result["layout_det_res"]["boxes"] → [{coordinate, label, score}, ...]
        coordinate: [x1, y1, x2, y2]
        label: text/title/table/figure/formula/seal/...
      result["table_res_list"] → 表格识别结果列表
        每项含 table_region_id（对应 boxes 索引），html["pred"] 为 HTML 字符串

子进程退出时 OS 强制回收所有内存（包括 PaddlePaddle NaiveAllocator 缓存的
中间张量），彻底解决 PPStructureV3 长期运行内存不释放问题。
"""
import os
import sys
import json
import base64
import logging

# PaddlePaddle 内存分配器策略：必须在 import paddle 之前设置
# 与 _worker_ocr.py / SubprocessOCRPool._build_env 保持一致：
# naive_best_fit + eager_delete 系列（官方推荐，PaddleOCR #11639 / 讨论 #14497），
# 控制 PPStructureV3 内存增长。父进程 _build_env 已设同名变量（优先级更高），
# 此处 setdefault 仅为直接运行 worker 时提供兜底。
os.environ.setdefault("FLAGS_allocator_strategy", "naive_best_fit")
os.environ.setdefault("FLAGS_eager_delete_scope", "True")
os.environ.setdefault("FLAGS_eager_delete_tensor_gb", "0.0")
os.environ.setdefault("FLAGS_fast_eager_deletion_mode", "True")
os.environ.setdefault("FLAGS_use_pinned_memory", "False")
os.environ.setdefault("FLAGS_fraction_of_cpu_memory_to_use", "0.1")
os.environ.setdefault("FLAGS_initial_cpu_memory_in_mb", "128")

# 关键：在 import paddleocr 之前禁用其日志输出到 stdout
# PPStructureV3 的 DEBUG/INFO 日志会污染 stdout JSON 通信通道，导致主进程
# _recv() 解析 JSON 失败。所有日志输出到 stderr。
logging.basicConfig(
    stream=sys.stderr,
    level=logging.WARNING,
    format="[layout-worker] %(asctime)s %(levelname)s %(name)s: %(message)s",
)

import numpy as np

_layout = None  # PPStructureV3 实例（init 后创建）


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


def _log(msg: str) -> None:
    """诊断日志输出到 stderr，绝不污染 stdout JSON 通道。"""
    print(f"[layout-worker] {msg}", file=sys.stderr, flush=True)


def init(config: dict) -> None:
    """初始化 PPStructureV3 实例。

    config 字段：
      - lang: 语言（ch / en / ...），仅在无内置内部 OCR 模型时使用
      - text_detection_model_name/dir: 内部表格 OCR 检测模型（v6 small，可选）
      - text_recognition_model_name/dir: 内部表格 OCR 识别模型（v6 small，可选）
      - layout_detection_model_name/dir: 版面检测模型（PP-DocLayoutV3，可选）
      - table_classification_model_name/dir: 表格分类模型（可选）
      - wired_table_structure_recognition_model_name/dir: 有线表格结构识别（可选）
      - wireless_table_structure_recognition_model_name/dir: 无线表格结构识别（可选）
      - wired_table_cells_detection_model_name/dir: 有线单元格检测（可选）
      - wireless_table_cells_detection_model_name/dir: 无线单元格检测（可选）
      - layout_threshold: 版面检测置信度阈值（可选）

    注：PPStructureV3 3.7.0 不支持 ocr_version="PP-OCRv6"，故 config 不含
    ocr_version；内部表格 OCR 改用显式指定的 v6 small 模型目录。
    """
    global _layout
    from paddleocr import PPStructureV3

    # 修正 frozen 环境的 paddleocr.__file__ 路径
    # PPStructureV3 内部用 Path(__file__).parent 定位字典文件，
    # frozen 环境下 __file__ 是 PYZ 虚拟路径会导致定位失败
    try:
        from app.providers.paddle_local import _fix_paddleocr_file_path
        _fix_paddleocr_file_path()
    except Exception:
        pass

    lang = config.get("lang", "ch")
    layout_detection_model_dir = config.get("layout_detection_model_dir", "") or ""
    layout_detection_model_name = config.get("layout_detection_model_name", "") or ""
    # 内部表格单元格 OCR（v6 small，离线）
    text_detection_model_dir = config.get("text_detection_model_dir", "") or ""
    text_detection_model_name = config.get("text_detection_model_name", "") or ""
    text_recognition_model_dir = config.get("text_recognition_model_dir", "") or ""
    text_recognition_model_name = config.get("text_recognition_model_name", "") or ""
    # 表格结构识别开关（可配置）：false 时跳过 SLANet+RT-DETR，大幅降低 PPStructure 耗时
    # 实测：true 时每页 38-52s（SLANet 表格结构推理），false 时 5-15s
    # 关闭后表格区域仍被检测到，但只做普通文字 OCR（不生成 HTML 结构）
    use_table_recognition = bool(config.get("use_table_recognition", True))
    # 表格结构识别模型（5 个，离线）——仅在启用表格识别时加载
    _table_prefixes = [
        "table_classification",
        "wired_table_structure_recognition",
        "wireless_table_structure_recognition",
        "wired_table_cells_detection",
        "wireless_table_cells_detection",
    ]
    table_models = {}
    if use_table_recognition:
        for prefix in _table_prefixes:
            d = config.get(f"{prefix}_model_dir", "") or ""
            n = config.get(f"{prefix}_model_name", "") or ""
            if d and n:
                table_models[prefix] = (n, d)
    # 文档方向 / 文本行方向分类模型（离线，predict 时惰性创建）
    _aux_prefixes = [
        "doc_orientation_classify",
        "textline_orientation",
    ]
    aux_models = {}
    for prefix in _aux_prefixes:
        d = config.get(f"{prefix}_model_dir", "") or ""
        n = config.get(f"{prefix}_model_name", "") or ""
        if d and n:
            aux_models[prefix] = (n, d)
    layout_threshold = config.get("layout_threshold")

    # PPStructureV3 3.7.0 的 _SUPPORTED_OCR_VERSIONS 仅 v3/v4/v5，不支持
    # ocr_version="PP-OCRv6"（传了会抛 ValueError），故绝不传 ocr_version。
    # 内部表格 OCR 改用显式指定的 v6 small 模型目录（绕过版本校验）。
    # 显式指定内部 OCR 模型时 lang 会被忽略并告警，故仅在无内部 OCR 时传 lang。
    has_internal_ocr = bool(text_detection_model_dir and text_recognition_model_dir)

    kwargs: dict = dict(
        # 3.x 开关参数：仅启用版面检测 + 表格识别，关闭其余子任务降低开销
        use_doc_orientation_classify=False,  # 扫描件方向已正，跳过方向分类
        use_doc_unwarping=False,             # 跳过文档去弯曲
        use_textline_orientation=False,      # 跳过文本行方向分类
        use_table_recognition=use_table_recognition,  # 可配置：false 跳过 SLANet 大幅加速
        use_formula_recognition=False,       # 公式识别交给上层 OCR
        use_chart_recognition=False,          # 图表识别关闭（开销大且本服务不需要）
        use_seal_recognition=False,           # 印章识别关闭
        use_region_detection=False,           # 区域检测关闭（已有版面检测）
    )
    if not has_internal_ocr:
        # 无内置内部 OCR 模型：交给 PPStructureV3 按 lang 自动选（默认 v5_server）
        kwargs["lang"] = lang

    # 内部表格单元格 OCR 模型（v6 small，传 dir 时配套传 name）
    if text_detection_model_dir and text_detection_model_name:
        kwargs["text_detection_model_name"] = text_detection_model_name
        kwargs["text_detection_model_dir"] = text_detection_model_dir
    if text_recognition_model_dir and text_recognition_model_name:
        kwargs["text_recognition_model_name"] = text_recognition_model_name
        kwargs["text_recognition_model_dir"] = text_recognition_model_dir

    # 版面检测模型（独立于内部 OCR，传 dir 时配套传 name，与 yml 内 Global.model_name 一致）
    if layout_detection_model_dir:
        kwargs["layout_detection_model_dir"] = layout_detection_model_dir
        if layout_detection_model_name:
            kwargs["layout_detection_model_name"] = layout_detection_model_name
    if layout_threshold is not None:
        kwargs["layout_threshold"] = layout_threshold

    # 表格结构识别模型（5 个，离线。传 dir 时配套传 name）
    for prefix, (name, dir_) in table_models.items():
        kwargs[f"{prefix}_model_name"] = name
        kwargs[f"{prefix}_model_dir"] = dir_

    # 文档方向 / 文本行方向分类模型（离线。predict 时惰性创建，显式传 dir 避免锁机制）
    for prefix, (name, dir_) in aux_models.items():
        kwargs[f"{prefix}_model_name"] = name
        kwargs[f"{prefix}_model_dir"] = dir_

    _layout = PPStructureV3(**kwargs)
    send({"type": "ready"})


def _extract_table_htmls(result) -> dict:
    """从 PPStructureV3 结果中提取表格 HTML，返回 {box_index: html_str}。

    结果结构：
      result["table_res_list"] → 表格识别结果列表
      每项 table_res 含 table_region_id（对应 layout_det_res["boxes"] 的索引）
      table_res.html["pred"] → HTML 字符串

    table_res.html 属性调用 _to_html() 返回 {"pred": html_str} 字典。
    若表格识别失败或无表格，返回空字典。
    """
    table_htmls: dict = {}
    table_res_list = result.get("table_res_list") if isinstance(result, dict) else None
    if not table_res_list:
        return table_htmls
    for table_res in table_res_list:
        try:
            region_id = table_res.get("table_region_id")
            # table_res.html 是 HtmlMixin 的 property，返回 {"pred": html_str}
            html_dict = table_res.html
            html_str = html_dict.get("pred") if isinstance(html_dict, dict) else None
            if isinstance(html_str, str) and html_str and region_id is not None:
                table_htmls[int(region_id)] = html_str
        except Exception as e:
            _log(f"提取表格 HTML 失败: {e}")
    return table_htmls


def _sort_lines_reading_rows(lines: list) -> list:
    """行带感知排序：同一水平行带内的行按 x1 升序，行带之间按 y 升序。

    行带判定：按 y1 升序扫描，若该行 y 区间与当前行带的 y 区间
    重叠超过该行高度的 40%，视为同一行带（如同一行被拆成的左右两个框、
    表格同一行的多个格子，y 只差几像素）。正文逐行排列（行距 > 行高）
    时每行自成一个行带，行为与纯 y 排序一致。
    对齐上游 `_sort_column_rows` 的做法，修复块内仅按 y 排序导致的左右错位。
    """
    if len(lines) <= 1:
        return list(lines)

    def _yrange(ln) -> tuple:
        return ln["bbox"][1], ln["bbox"][3]

    items = sorted(lines, key=lambda ln: ln["bbox"][1])
    rows: list = []
    row_y1 = row_y2 = 0
    for ln in items:
        y1, y2 = _yrange(ln)
        lh = max(1, y2 - y1)
        if rows and y1 < row_y2 - 0.4 * lh and y2 > row_y1 + 0.4 * lh:
            rows[-1].append(ln)
            row_y1 = min(row_y1, y1)
            row_y2 = max(row_y2, y2)
        else:
            rows.append([ln])
            row_y1, row_y2 = y1, y2

    out: list = []
    for row in rows:
        row.sort(key=lambda ln: ln["bbox"][0])  # 行带内按 x1 升序
        out.extend(row)
    return out


def _extract_native_ocr_lines(result) -> list:
    """从 PPStructureV3 结果中提取原生 OCR 文本行（已按阅读顺序排列）。

    PPStructureV3.predict() 一次调用即完成：
      1. 版面检测 → layout_det_res
      2. 整页 OCR → overall_ocr_res（rec_texts / rec_boxes / rec_scores）
      3. OCR 行匹配到版面区域 + XY-cut 阅读顺序排序 → parsing_res_list

    本函数提取 overall_ocr_res 中的全部文本行，再按 parsing_res_list 的
    块阅读顺序重排，使输出行天然按"左栏→中栏→右栏、栏内从上到下"排列，
    无需上层再做 KMeans 栏缝检测 / 强制分栏 / 多栏排序。

    返回格式：[{text, bbox: [x1,y1,x2,y2], score}, ...]（阅读顺序）
    失败时返回空列表（上层回退到逐区域 OCR 旧路径）。
    """
    if not isinstance(result, dict):
        return []
    overall = result.get("overall_ocr_res")
    if not overall or not isinstance(overall, dict):
        return []
    rec_texts = overall.get("rec_texts") or []
    rec_scores = overall.get("rec_scores") or []
    rec_boxes = overall.get("rec_boxes")
    n = len(rec_texts)
    if n == 0 or rec_boxes is None:
        return []
    try:
        boxes_list = [[int(v) for v in b] for b in rec_boxes]
    except (TypeError, ValueError):
        return []
    scores_list = [float(s) for s in rec_scores] if len(rec_scores) == n else [0.0] * n
    # 收集所有行（中心点坐标用于匹配块）
    lines = []
    for i in range(n):
        x1, y1, x2, y2 = boxes_list[i]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        lines.append({
            "text": rec_texts[i], "bbox": [x1, y1, x2, y2],
            "score": scores_list[i], "cx": cx, "cy": cy, "used": False,
        })

    # parsing_res_list：XY-cut 排序后的块列表（阅读顺序）
    parsing = result.get("parsing_res_list") or []
    ordered: list = []
    for block in parsing:
        bb = getattr(block, "bbox", None) or (block.get("bbox") if isinstance(block, dict) else None)
        if not bb or len(bb) != 4:
            continue
        bx1, by1, bx2, by2 = [int(v) for v in bb]
        matched = [
            ln for ln in lines
            if not ln["used"] and bx1 <= ln["cx"] <= bx2 and by1 <= ln["cy"] <= by2
        ]
        matched = _sort_lines_reading_rows(matched)  # 块内行带感知排序（同行按 x 归位）
        for ln in matched:
            ln["used"] = True
            ordered.append({"text": ln["text"], "bbox": ln["bbox"], "score": ln["score"]})

    # 未匹配到任何块的行：按 (y, x) 顺序追加到末尾
    leftover = [ln for ln in lines if not ln["used"]]
    leftover.sort(key=lambda ln: (ln["bbox"][1], ln["bbox"][0]))
    for ln in leftover:
        ordered.append({"text": ln["text"], "bbox": ln["bbox"], "score": ln["score"]})
    return ordered


def do_analyze(msg: dict) -> None:
    """执行版面分析。

    输入：
      - image: base64 编码的 BGR numpy 数组
      - shape: [h, w, c]
      - dtype: numpy dtype 字符串（如 "uint8"）
    输出：
      - regions: [{type, bbox: [x1,y1,x2,y2], html}, ...]
        table 类型区域包含 html 字段（SLANet 识别结果）
      - ocr_lines: [{text, bbox, score}, ...] 原生 OCR 行（阅读顺序）
        PPStructureV3 一次 predict() 即完成整页 OCR + XY-cut 阅读顺序排序，
        上层直接使用可跳过逐区域 OCR（34 次→0 次）和自定义栏分析。
    """
    global _layout
    if _layout is None:
        send({"type": "error", "message": "PPStructureV3 未初始化"})
        return

    # 解码图片
    img_bytes = base64.b64decode(msg["image"])
    shape = tuple(msg["shape"])
    dtype = np.dtype(msg["dtype"])
    img = np.frombuffer(img_bytes, dtype=dtype).reshape(shape)

    # 关键：推理期间临时把 fd 1（stdout）重定向到 fd 2（stderr），
    # 防止 PPStructureV3 / PaddlePaddle C++ 层直接 write(1, ...) 污染 JSON 通信通道。
    # send() 是唯一允许写 stdout 的函数，推理结束后恢复 fd 1。
    saved_fd = os.dup(1)
    result = None
    analyze_error = None
    try:
        os.dup2(2, 1)  # stdout -> stderr
        # 3.x: predict() 返回 [StructureV3Result, ...]，取首页
        results = _layout.predict(img)
        result = results[0] if results else None
    except Exception as e:
        analyze_error = e
    finally:
        os.dup2(saved_fd, 1)  # 恢复 stdout
        os.close(saved_fd)

    if analyze_error is not None:
        send({"type": "error", "message": str(analyze_error)})
        return
    if result is None:
        send({"type": "result", "regions": []})
        return

    # 序列化 PPStructureV3 结果为 JSON 可传输格式
    # 结果结构：
    #   result["layout_det_res"]["boxes"] → [{coordinate, label, score}, ...]
    #     coordinate: [x1, y1, x2, y2]
    #     label: text/title/table/figure/formula/seal/...
    #   result["table_res_list"] → 表格 HTML（通过 _extract_table_htmls 提取）
    regions_out = []
    try:
        layout_det_res = result.get("layout_det_res") if isinstance(result, dict) else None
        boxes = []
        if layout_det_res is not None:
            boxes = layout_det_res.get("boxes") or []
        # 提取表格 HTML（{box_index: html_str}）
        table_htmls = _extract_table_htmls(result)

        for idx, box_info in enumerate(boxes):
            rtype = (box_info.get("label") or "text").lower()
            bbox = box_info.get("coordinate")
            if not bbox or len(bbox) != 4:
                continue
            try:
                bbox_list = [int(v) for v in bbox]
            except (TypeError, ValueError):
                continue
            # 提取表格 HTML（仅 table 类型区域，按 box 索引匹配）
            html = None
            if rtype == "table":
                html = table_htmls.get(idx)
            regions_out.append({
                "type": rtype,
                "bbox": bbox_list,
                "html": html,
            })
    except Exception as e:
        _log(f"解析版面分析结果异常: {e}")
        send({"type": "error", "message": f"解析结果失败: {e}"})
        return

    # 提取原生 OCR 行（PPStructureV3 整页 OCR + XY-cut 阅读顺序排序）
    # 一次 predict() 即完成全部 OCR，上层无需再逐区域调用 OCR 子进程
    ocr_lines_out: list = []
    try:
        ocr_lines_out = _extract_native_ocr_lines(result)
    except Exception as e:
        _log(f"提取原生 OCR 行失败（上层将回退逐区域 OCR）: {e}")

    _log(
        f"版面分析: {len(regions_out)} 个区域, "
        f"类型分布 {dict((t, sum(1 for r in regions_out if r['type'] == t)) for t in set(r['type'] for r in regions_out))}"
        f", 原生OCR行 {len(ocr_lines_out)}"
    )
    send({"type": "result", "regions": regions_out, "ocr_lines": ocr_lines_out})


def _get_memory_mb() -> float:
    """返回当前进程 RSS 内存（MB），失败返回 0。"""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


# 内存阈值（MB）：超过此值主动退出，让主进程重启子进程释放内存。
# PPStructureV3 比 PaddleOCR 内存占用低（无识别模型），但仍需防护：
#   - 正常页面 < 600MB
#   - 复杂版面（多栏 + 大表格 SLANet 推理）可能到 1.2-1.8GB
#   - 超过 2000MB 基本是内存泄漏，主动退出比被 OOM kill 更优雅
# 旧值 1200MB 在复杂表格页面容易误触发，导致版面分析子进程频繁重启。
_MEMORY_LIMIT_MB = 2000


def main() -> None:
    """主循环：接收请求，处理，响应。

    内存自保护：每次 analyze 后检查 RSS，超过 _MEMORY_LIMIT_MB 主动退出。
    主进程 _recv 检测到子进程退出会触发 _restart_proc，新子进程接管后续页。
    """
    page_count = 0
    while True:
        try:
            msg = recv()
        except Exception:
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
        elif msg_type == "analyze":
            do_analyze(msg)
            page_count += 1
            # 每页检查内存阈值（防止复杂表格 SLANet 推理单页内存暴涨）
            mem_mb = _get_memory_mb()
            if mem_mb > _MEMORY_LIMIT_MB:
                _log(f"内存超限({mem_mb:.0f}MB > {_MEMORY_LIMIT_MB}MB)，主动退出等待重启")
                # 退出前发送通知：让主进程知道是"内存超限主动退出"而非"崩溃"
                try:
                    send({"type": "exit", "reason": "memory_limit",
                          "mem_mb": round(mem_mb, 1), "page_count": page_count})
                except Exception:
                    pass
                break
            # 每 5 页输出一次内存日志
            if page_count % 5 == 0:
                _log(f"内存监控: 已处理 {page_count} 页, RSS={mem_mb:.0f}MB")
        else:
            send({"type": "error", "message": f"未知消息类型: {msg_type}"})

    # 清理
    global _layout
    _layout = None


if __name__ == "__main__":
    main()
