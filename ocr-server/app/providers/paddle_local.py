"""本地 PaddleOCR Provider（v6 / PP-OCRv6）。

基于 PaddleOCR 3.7.0（PP-OCRv6），纯 CPU 运行。
- 完整识别（det + rec）：作为独立 OCR 引擎使用
- 仅检测（det）：供双层漏斗过滤第二层使用

PaddleOCR 为可选依赖，未安装时 is_available() 返回 False，
上层自动降级为讯飞-only 模式。

3.x API 要点（与 2.x 的差异）：
  - 构造参数名全部更名：
      det_model_dir    → text_detection_model_dir
      rec_model_dir    → text_recognition_model_dir
      use_angle_cls    → use_textline_orientation
      drop_score       → text_rec_score_thresh
      det_db_unclip_ratio → text_det_unclip_ratio
      det_db_box_thresh   → text_det_box_thresh
  - ocr_version 参数：PP-OCRv3 / PP-OCRv4 / PP-OCRv5 / PP-OCRv6（默认 v6）
  - 推理调用：predict(img) 取代 ocr(img, cls=False)
    （ocr() 仍保留但已废弃，会输出 DeprecationWarning）
  - 结果格式：predict() 返回 [OCRResult, ...]，OCRResult 继承 dict，
    通过 page['rec_texts'] / page['rec_scores'] / page['dt_polys'] 访问数据
    （注意：getattr(page, 'rec_texts') 返回空字符串，必须用 dict 访问）
  - 3.5+ 引擎配置为 opt-in：不显式传 engine/engine_config 时默认行为与旧版
    一致（仍用飞桨框架推理），旧兼容参数继续生效。

v6 模型分档（PPLCNetV4 骨干，单模型支持 50 种语言）：
  tiny   det 1.9MB + rec 4.4MB   识别 73.5%  端侧/IoT
  small  det 9.6MB + rec 20.4MB  识别 81.3%  移动端/桌面（默认档，v5 mobile 继任）
  medium det 59.4MB + rec 73.3MB 识别 83.2%  服务端（+5.1% over v5_server）

离线运行支持：
  打包后通过 _get_bundled_models_dir() 定位应用内置模型目录，
  并把 det/rec 模型路径显式传给 PaddleOCR，避免首次运行联网下载。
  v6 模型缓存路径：~/.paddlex/official_models/PP-OCRv6_small_det/
  内置模型目录：paddleocr_models/PP-OCRv6_small_det/ 等
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseProvider, OCRLine, OCRResult

logger = logging.getLogger(__name__)

# 实例池：每个槽位持有独立 PaddleOCR 实例，避免多线程共享崩溃。
# PaddleOCR 模型实例非线程安全，多并发请求共享同一实例会崩溃或结果错乱。
# 池大小 = 最大并发数，每个并发槽位对应一个独立实例，互不干扰。
_OCR_POOL: List[Any] = []  # PaddleOCR 实例列表（按槽位索引）
_OCR_CONFIG: Optional[Tuple] = None  # 已初始化时的配置快照（供调试/比对）
_POOL_LOCK = threading.Lock()  # 保护池的创建，避免并发重复初始化
_POOL_SIZE = 0  # 池大小，0=未初始化


def _get_bundled_models_dir() -> Optional[str]:
    """返回打包内置的 PaddleOCR 模型根目录，无则返回 None。

    打包时模型放在 _internal/paddleocr_models/ 下，目录结构：
        paddleocr_models/
            PP-OCRv6_small_det/          # 文本检测模型（v6 small，默认档）
            PP-OCRv6_small_rec/          # 文本识别模型（v6 small，默认档）
            PP-OCRv6_medium_det/         # 文本检测模型（v6 medium，可选高精度）
            PP-OCRv6_medium_rec/         # 文本识别模型（v6 medium，可选高精度）
            PP-OCRv6_tiny_det/           # 文本检测模型（v6 tiny，可选轻量）
            PP-OCRv6_tiny_rec/           # 文本识别模型（v6 tiny，可选轻量）
            PP-DocLayout_plus-L/         # 版面分析模型（PPStructureV3 用）
            ...
    """
    if getattr(sys, "frozen", False):
        # PyInstaller onedir: _internal/paddleocr_models
        base = os.path.dirname(sys.executable)
        candidates = [
            os.path.join(base, "_internal", "paddleocr_models"),
            os.path.join(base, "paddleocr_models"),
        ]
    else:
        # 开发环境：项目根 / paddleocr_models（可选）
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidates = [os.path.join(base, "paddleocr_models")]

    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def _resolve_bundled_model(subpath: str) -> Optional[str]:
    """返回内置模型目录下指定子目录的完整路径，不存在返回 None。

    subpath 示例: "PP-OCRv6_small_det"
    """
    root = _get_bundled_models_dir()
    if root is None:
        return None
    full = os.path.join(root, subpath)
    return full if os.path.isdir(full) else None


def _get_paddleocr_pkg_dir() -> Optional[str]:
    """返回 paddleocr 包在磁盘上的真实目录（含 paddleocr.py）。

    PyInstaller frozen 环境下，import paddleocr 加载的是 PYZ 归档里的
    .pyc，其 __file__ 是虚拟路径（如 ...OCR-pdf.exe/PYZ-00.pyz/paddleocr/...），
    磁盘上不存在。PaddleOCR 内部用 Path(__file__).parent 定位字典文件
    （ppocr_keys_v1.txt 等）会失败。本函数返回 collect_all 放到磁盘上的
    真实包目录 _internal/paddleocr/，供调用方修正 __file__。
    """
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(sys.executable)
        candidates = [
            os.path.join(exe_dir, "_internal", "paddleocr"),
            os.path.join(exe_dir, "paddleocr"),
        ]
        for c in candidates:
            if os.path.isfile(os.path.join(c, "paddleocr.py")):
                return c
    return None


def _fix_paddleocr_file_path() -> None:
    """修正 frozen 环境 paddleocr.__file__ 指向磁盘真实路径。

    PaddleOCR 初始化时用 Path(__file__).parent 定位字典文件
    （ppocr_keys_v1.txt 等）。frozen 环境下 __file__ 是 PYZ 虚拟路径，
    Path(__file__).parent 指向不存在的目录，open() 字典失败 →
    本地 OCR 初始化失败或识别为空。
    本函数把 __file__ 改为磁盘上 _internal/paddleocr/paddleocr.py，
    使 Path(__file__).parent 正确定位到 _internal/paddleocr/。
    """
    if not getattr(sys, "frozen", False):
        return
    import paddleocr as _poc
    cur_file = getattr(_poc, "__file__", "") or ""
    if cur_file and os.path.isfile(cur_file):
        return  # 已是有效磁盘路径，无需修正
    pkg_dir = _get_paddleocr_pkg_dir()
    if pkg_dir:
        _poc.__file__ = os.path.join(pkg_dir, "paddleocr.py")
        logger.debug("已修正 paddleocr.__file__ -> %s", _poc.__file__)


def _read_model_name(model_dir: str) -> Optional[str]:
    """从模型目录的 inference.yml 读取 Global.model_name。

    paddleocr 3.7.0 要求：传 model_dir 时必须配套传匹配的 model_name，
    resolve_model_name 会校验 yml 内 Global.model_name 是否与传入的 model_name
    一致，不一致则报 "Model name mismatch"。本函数用于读取用户自配目录的模型名。
    """
    yml = os.path.join(model_dir, "inference.yml")
    if not os.path.isfile(yml):
        return None
    try:
        import yaml
        with open(yml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return (cfg.get("Global") or {}).get("model_name")
    except Exception:
        return None


def _build_paddle_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """根据配置字典构建 PaddleOCR v6 构造参数。

    cfg 字段:
        lang, ocr_version, det_model_dir, rec_model_dir,
        drop_score, det_db_unclip_ratio, det_db_box_thresh

    关键参数说明：
      ocr_version: "PP-OCRv6"（默认）/ "PP-OCRv5" / "PP-OCRv4" / "PP-OCRv3"
      text_det_limit_side_len: 检测前图像缩放尺寸（960 适合文档扫描件）
      text_det_limit_type: "max" 按长边限制（默认值适用文档）

    paddleocr 3.7.0 模型选择规则（关键）：
      - 若 text_detection/recognition_model_name 或 model_dir 任一非 None，
        则 lang / ocr_version 被忽略（并抛 UserWarning）。
      - 传 model_dir 时必须配套传匹配的 model_name，否则 resolve_model_name
        会用默认档名（v6 = medium）校验目录内 yml 的 Global.model_name，与
        small/tiny 目录不匹配时报 "Model name mismatch"。
      - ocr_version="PP-OCRv6" 在自动模式下恒映射到 medium 档（_get_ocr_model_names
        对 v6 只返回 medium），无法通过 ocr_version 选 small/tiny，必须显式传
        model_name + model_dir 指定档位。
    因此：内置/用户目录可用时，显式传 model_name + model_dir，且不再传
    lang/ocr_version（避免告警）；无显式目录时才传 lang/ocr_version 走自动下载。
    """
    lang = cfg.get("lang", "ch")
    ocr_version = cfg.get("ocr_version", "PP-OCRv6")
    # 模型档位：tiny(最快/精度低) / small(默认/平衡) / medium(高精度/慢)
    # 仅 v6 有效；v5/v4 固定 mobile 无分档。显式 det_model_dir/rec_model_dir 优先。
    ocr_model_tier = cfg.get("ocr_model_tier", "small")
    det_model_dir = cfg.get("det_model_dir", "")
    rec_model_dir = cfg.get("rec_model_dir", "")
    drop_score = cfg.get("drop_score", 0.0)
    det_db_unclip_ratio = cfg.get("det_db_unclip_ratio", 1.8)
    det_db_box_thresh = cfg.get("det_db_box_thresh", 0.5)

    kwargs: Dict[str, Any] = {
        # 不做文本行方向分类（扫描件方向已正，省去 cls 模型加载和推理）
        "use_textline_orientation": False,
        # 标题识别精进参数（3.x 参数名，v6 沿用）：
        #   text_rec_score_thresh  保留全部识别结果（默认 0.5 会丢弃低置信度行）
        #   text_det_unclip_ratio   扩大检测框，覆盖大字标题边缘字符
        #   text_det_box_thresh    降低框保留阈值，召回低对比度标题
        "text_rec_score_thresh": drop_score,
        "text_det_unclip_ratio": det_db_unclip_ratio,
        "text_det_box_thresh": det_db_box_thresh,
        # 检测前图像缩放：3.x 默认 limit_side_len=64 + limit_type="min"，
        # 对大图（3000px+）会缩到极小导致检测不到文字。
        # 显式设 960 + "max"（长边限制 960px），与 v4 默认行为一致。
        "text_det_limit_side_len": 960,
        "text_det_limit_type": "max",
    }

    # 解析 det 模型：优先用户配置目录，其次打包内置，最后 None（自动下载）
    det_dir: Optional[str] = det_model_dir or None
    det_name: Optional[str] = None
    if det_dir:
        det_name = _read_model_name(det_dir)
    else:
        det_subdir = _det_model_subdir(ocr_version, ocr_model_tier)
        bundled_det = _resolve_bundled_model(det_subdir)
        if bundled_det:
            det_dir = bundled_det
            det_name = det_subdir  # 内置模型子目录名即模型名
            logger.info("使用内置 det 模型: %s (tier=%s)", bundled_det, ocr_model_tier)

    # 解析 rec 模型：同上
    rec_dir: Optional[str] = rec_model_dir or None
    rec_name: Optional[str] = None
    if rec_dir:
        rec_name = _read_model_name(rec_dir)
    else:
        rec_subdir = _rec_model_subdir(ocr_version, ocr_model_tier)
        bundled_rec = _resolve_bundled_model(rec_subdir)
        if bundled_rec:
            rec_dir = bundled_rec
            rec_name = rec_subdir
            logger.info("使用内置 rec 模型: %s (tier=%s)", bundled_rec, ocr_model_tier)

    # 模型选择：det/rec 都有显式目录时，显式指定 name+dir（不传 lang/ocr_version，
    # 避免被忽略告警）；否则交给 paddleocr 按 lang/ocr_version 自动选择（联网下载默认档）。
    # 注：det 与 rec 必须同时显式或同时自动，否则 paddleocr 会因一方为 None 而用
    # 默认档名校验，触发 mismatch。内置模型成对存在，正常配置不会出现仅一方有的情况。
    if det_dir and det_name and rec_dir and rec_name:
        kwargs["text_detection_model_dir"] = det_dir
        kwargs["text_detection_model_name"] = det_name
        kwargs["text_recognition_model_dir"] = rec_dir
        kwargs["text_recognition_model_name"] = rec_name
    else:
        kwargs["lang"] = lang
        kwargs["ocr_version"] = ocr_version

    return kwargs


def _det_model_subdir(ocr_version: str, tier: str = "small") -> str:
    """根据 ocr_version 和 tier 返回内置 det 模型子目录名。

    tier（仅 v6 有效，v5/v4 无分档固定 mobile）：
      tiny   det 1.9MB  识别 73.5%  最快，端侧/IoT（CPU 极慢时降级用）
      small  det 9.6MB  识别 81.3%  默认档，CPU 推理快（v5 mobile 继任）
      medium det 59.4MB 识别 83.2%  高精度，CPU 推理慢约 3-5 倍
    用户可通过 config.json 的 paddle.ocr_model_tier 切换。
    显式 det_model_dir 优先级最高（绕过 tier）。
    """
    v = (ocr_version or "").upper().replace("-", "").replace("_", "")
    if "V6" in v:
        t = (tier or "small").lower()
        if t == "tiny":
            return "PP-OCRv6_tiny_det"
        if t == "medium":
            return "PP-OCRv6_medium_det"
        return "PP-OCRv6_small_det"  # small / 非法值默认
    if "V5" in v:
        return "PP-OCRv5_mobile_det"
    if "V4" in v:
        return "PP-OCRv4_mobile_det"
    return "PP-OCRv6_small_det"  # 默认 v6


def _rec_model_subdir(ocr_version: str, tier: str = "small") -> str:
    """根据 ocr_version 和 tier 返回内置 rec 模型子目录名。

    tier（仅 v6 有效）：
      tiny   rec 4.4MB   识别 73.5%  最快
      small  rec 20.4MB  识别 81.3%  默认档
      medium rec 73.3MB  识别 83.2%  高精度，内存占用大
    """
    v = (ocr_version or "").upper().replace("-", "").replace("_", "")
    if "V6" in v:
        t = (tier or "small").lower()
        if t == "tiny":
            return "PP-OCRv6_tiny_rec"
        if t == "medium":
            return "PP-OCRv6_medium_rec"
        return "PP-OCRv6_small_rec"  # small / 非法值默认
    if "V5" in v:
        return "PP-OCRv5_mobile_rec"
    if "V4" in v:
        return "PP-OCRv4_mobile_rec"
    return "PP-OCRv6_small_rec"  # 默认 v6


def _create_ocr_instance(cfg: Dict[str, Any]) -> Any:
    """创建单个 PaddleOCR 实例。失败时抛异常，由调用方决定如何处理。

    模型路径优先级（高 → 低）：
      1. 用户配置的 det_model_dir / rec_model_dir（精确指定）
      2. 打包内置模型 _internal/paddleocr_models/
      3. PaddleOCR 默认 ~/.paddlex/official_models/（首次运行联网下载）
    """
    # 延迟导入，避免未安装 paddleocr 时整个应用无法启动
    from paddleocr import PaddleOCR
    # frozen 环境下修正 __file__，使 PaddleOCR 内部能定位字典文件
    _fix_paddleocr_file_path()
    kwargs = _build_paddle_kwargs(cfg)
    return PaddleOCR(**kwargs)


def init_ocr_pool(size: int, config_kwargs: Dict[str, Any]) -> None:
    """初始化 OCR 实例池，创建 size 个独立实例。

    已初始化且大小一致时直接返回，避免重复创建。
    创建过程异常会被捕获并记日志，部分失败时池中实例数可能少于 size
    （调用方可通过 len(_OCR_POOL) 判断实际可用实例数）。
    """
    global _OCR_POOL, _OCR_CONFIG, _POOL_SIZE
    if size < 1:
        size = 1
    with _POOL_LOCK:
        if _OCR_POOL and _POOL_SIZE == size:
            return  # 已初始化且大小一致
        _OCR_POOL.clear()
        for i in range(size):
            try:
                instance = _create_ocr_instance(config_kwargs)
                _OCR_POOL.append(instance)
                logger.info(
                    "PaddleOCR 实例 %d/%d 创建成功（%s）",
                    i + 1, size, config_kwargs.get("ocr_version", "PP-OCRv6"),
                )
            except ImportError as e:
                # paddleocr 未安装：无需继续尝试，直接退出
                logger.warning("paddleocr 未安装，本地 OCR 不可用: %s", e)
                break
            except Exception as e:
                # 其它异常（如模型加载失败、OOM）：同样退出，避免重复同样错误
                logger.warning(
                    "PaddleOCR 实例 %d/%d 创建失败: %s", i + 1, size, e
                )
                break
        _POOL_SIZE = size
        _OCR_CONFIG = tuple(sorted(config_kwargs.items()))


def get_ocr_instance(slot: int) -> Any:
    """获取指定槽位的 OCR 实例（0~size-1）。

    槽位越界或池未初始化时返回 None，由调用方决定降级策略。
    """
    if not _OCR_POOL:
        return None
    if slot < 0 or slot >= len(_OCR_POOL):
        return None
    return _OCR_POOL[slot]


def pool_size() -> int:
    """返回当前实例池中可用实例数。"""
    return len(_OCR_POOL)


def rebuild_ocr_instance(slot: int) -> bool:
    """重建指定槽位的 OCR 实例，释放 PaddlePaddle C++ 内存池。

    PaddlePaddle 的 NaiveAllocator 分配的内存不会随 Python 对象 GC
    自动归还给 OS。日志实测：仅替换池中引用（旧实例等 GC）后，
    重建只释放 ~500MB，但下一页立刻又涨回去甚至更高（基线递增）。

    本方法在替换引用前显式做三件事：
      1. 调用 paddle.cuda.empty_cache() / paddle.framework.core 的清理 API
         （CPU 模式下清理中间张量缓存）
      2. del 旧实例并 gc.collect()，触发 Python 层析构
      3. 再次 gc.collect() 确保循环引用释放

    场景：
      - 卡死恢复：旧线程持有旧实例，直接替换让新任务用新实例
      - 长文档内存累积：每 N 页重建，释放 paddle 内存池

    返回 True 表示重建成功。
    """
    global _OCR_POOL
    with _POOL_LOCK:
        if slot < 0 or slot >= len(_OCR_POOL):
            logger.warning("重建失败：槽位 %d 越界（池大小 %d）", slot, len(_OCR_POOL))
            return False
        # 复用启动时的配置快照重建实例
        config_kwargs = dict(_OCR_CONFIG) if _OCR_CONFIG else {}
        if not config_kwargs:
            logger.warning("重建失败：无可用配置快照")
            return False
        try:
            # 1. 先显式释放旧实例的 PaddlePaddle C++ 内存池
            old_instance = _OCR_POOL[slot]
            _release_paddle_memory(old_instance)
            # 2. 替换池中引用并触发 GC
            _OCR_POOL[slot] = None  # 先置 None 断开引用
            del old_instance
            import gc
            gc.collect()
            # 3. 再次清理 paddle 全局内存池
            _release_paddle_global_cache()
            gc.collect()
            # 4. 创建新实例
            instance = _create_ocr_instance(config_kwargs)
            _OCR_POOL[slot] = instance
            logger.warning("已重建槽位 %d 的 OCR 实例（内存释放）", slot)
            return True
        except Exception as e:
            logger.error("重建槽位 %d 的 OCR 实例失败: %s", slot, e)
            # 失败时尝试恢复一个可用实例，避免槽位永久失效
            try:
                instance = _create_ocr_instance(config_kwargs)
                _OCR_POOL[slot] = instance
            except Exception:
                pass
            return False


def _release_paddle_memory(instance) -> None:
    """显式释放单个 PaddleOCR 实例持有的 C++ 内存。

    PaddleOCR 实例内部持有 PaddleInference 的 AnalysisPredictor，
    其 NaiveAllocator 分配的内存不会随 Python GC 归还 OS。

    关键 API：AnalysisPredictor.ClearIntermediateTensor()
    清理预测过程中产生的中间张量，释放 C++ 内存池中的缓存。
    """
    try:
        # PaddleOCR v5 的 predictor 在 paddlex_pipeline 下
        for attr_name in ("paddlex_pipeline", "ocr_engine", "text_detector",
                          "text_recognizer", "structure"):
            attr = getattr(instance, attr_name, None)
            if attr is None:
                continue
            # 找到内部的 predictor 对象（不同版本字段名不同）
            predictor = None
            for pred_attr in ("predictor", "_predictor", "config"):
                predictor = getattr(attr, pred_attr, None)
                if predictor is not None:
                    break
            if predictor is None:
                continue
            # 调用 ClearIntermediateTensor 清理中间张量（PaddleInference 官方 API）
            clear_fn = getattr(predictor, "clear_intermediate_tensor", None) or \
                       getattr(predictor, "ClearIntermediateTensor", None)
            if clear_fn:
                try:
                    clear_fn()
                    logger.info("已清理 predictor 中间张量（%s）", attr_name)
                except Exception as e:
                    logger.debug("清理 predictor 中间张量失败（%s）: %s", attr_name, e)
    except Exception:
        pass


def _release_paddle_global_cache() -> None:
    """清理 PaddlePaddle 全局内存池和中间张量缓存。

    PaddlePaddle 3.x 的 NaiveAllocator 会缓存中间张量不归还 OS，
    长文档处理时累积到 GB 级。这里调用所有已知的清理 API。

    关键：通过 FLAGS_allocator_strategy 设置为 auto_growth 可让分配器
    在张量释放后归还内存给 OS，但只能在启动时设置。
    运行时只能调用 empty_cache / ClearIntermediateTensor 尽力清理。
    """
    try:
        import paddle
        # GPU 模式清理显存（CPU 模式下也无害）
        try:
            paddle.cuda.empty_cache()
        except Exception:
            pass
        # CPU 模式：尝试通过 core 清理全局内存池
        core = getattr(paddle.framework, "core", None)
        if core is not None:
            # 尝试清理 DeviceContext 缓存
            try:
                # 某些版本有 SetEmptyMemoryPool 接口
                set_empty = getattr(core, "SetEmptyMemoryPool", None)
                if set_empty:
                    set_empty(True)
                    logger.info("已调用 core.SetEmptyMemoryPool(True)")
            except Exception:
                pass
    except Exception:
        pass


class PaddleLocalProvider(BaseProvider):
    """本地 PaddleOCR 提供商（PP-OCRv6）。"""

    name = "paddle_local"

    def __init__(
        self,
        lang: str = "ch",
        use_gpu: bool = False,
        ocr_version: str = "PP-OCRv6",
        det_model_dir: str = "",
        rec_model_dir: str = "",
        det_score_thresh: float = 0.3,
        # 识别丢弃阈值：PaddleOCR 默认 0.5 会把识别置信度低于该值的整行
        # （含 box）直接丢弃。大字标题因训练集样本少，置信度常落在 0.3~0.5
        # 区间被整行丢弃 → 标题残缺。设为 0 保留全部结果，交给上层按
        # det_score_thresh 自行过滤。
        drop_score: float = 0.0,
        # DB 检测框扩展比例：PaddleOCR 默认 1.5 对小字够用，但对大字标题
        # 边缘字符会切割。适度扩大到 1.8 可覆盖标题边缘，又不至于把相邻
        # 行并成一个框。
        det_db_unclip_ratio: float = 1.8,
        # DB 框保留阈值：PaddleOCR 默认 0.6 偏高，低对比度标题框会被丢弃。
        # 降到 0.5 提高召回，少量噪声由 det_score_thresh 兜底。
        det_db_box_thresh: float = 0.5,
    ) -> None:
        self.lang = lang
        self.use_gpu = use_gpu
        self.ocr_version = ocr_version
        self.det_model_dir = det_model_dir
        self.rec_model_dir = rec_model_dir
        self.det_score_thresh = det_score_thresh
        self.drop_score = drop_score
        self.det_db_unclip_ratio = det_db_unclip_ratio
        self.det_db_box_thresh = det_db_box_thresh
        # 初始化失败缓存：避免每张图都重新尝试加载模型（耗时数秒）
        # 注意：池由 init_ocr_pool 在服务启动时预创建；未预创建时首次调用
        # _get_instance 会自动初始化 size=1 的池（向后兼容单实例场景）
        self._init_failed = False
        self._init_error: Optional[str] = None

    # ------------------------------------------------------------------
    # BaseProvider 接口
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        try:
            import paddleocr  # noqa: F401
            return True
        except ImportError:
            return False

    def recognize(self, image: np.ndarray, slot: int = 0, new_page: bool = False) -> OCRResult:
        h, w = image.shape[:2]
        # new_page: 与 SubprocessOCRProvider 接口对齐，同进程模式无 batch 重启逻辑，
        # 该参数仅用于标记页边界（供子进程模式计数），此处忽略。
        ocr = self._get_instance(slot)
        if ocr is None:
            # 初始化失败时若返回空结果，会把环境问题伪装成"无文字"（同子进程模式），上抛
            raise RuntimeError(
                "OCR 引擎实例初始化失败（paddle_local）；"
                "请检查 paddleocr/paddlepaddle 安装完整性后重启服务"
            )
        result = self._run_ocr(ocr, image)
        lines = self._parse_result(result, w, h)
        return OCRResult(lines=lines, width=w, height=h, provider=self.name)

    # ------------------------------------------------------------------
    # 检测专用（供过滤第二层调用，只返回是否有文字框）
    # ------------------------------------------------------------------
    def detect_boxes(self, image: np.ndarray, slot: int = 0):
        """返回检测到的文字框列表，每个框为 4 个 [x,y] 顶点。

        参数:
            image: BGR numpy 数组
            slot: 实例池槽位号，默认 0（单实例场景兼容）

        返回值:
          - None: 检测器不可用（初始化失败），调用方应放行不阻断
          - []  : 检测器可用但未检测到文字框，调用方可拒绝
          - 非空: 检测到的文字框列表

        实现说明：
          3.7.0 的检测器位于 ocr.paddlex_pipeline._pipeline.text_det_model，
          直接调用仅做 det（跳过 rec），比完整 predict() 快约 4 倍。
          v5/3.3.2 曾用 ocr.text_detector，已废弃，_get_det_model 内置兼容回退。
        """
        ocr = self._get_instance(slot)
        if ocr is None:
            return None  # 不可用，让上层放行
        try:
            # 优先用底层 text_det_model（仅 det，最快）
            det_model = self._get_det_model(ocr)
            if det_model is not None:
                # 3.7.0：text_det_model(img) 返回迭代器，每项 dict 含 "dt_polys"
                # （单图也可直接传 img，内部按 batch 处理）
                ret = det_model(image)
                dt_polys = None
                if ret is not None:
                    try:
                        first = next(iter(ret))
                        if isinstance(first, dict):
                            dt_polys = first.get("dt_polys")
                    except StopIteration:
                        dt_polys = None
                if not dt_polys:
                    return []
                boxes: List[List[List[int]]] = []
                for box in dt_polys:
                    try:
                        points = [[int(float(p[0])), int(float(p[1]))] for p in box]
                    except (TypeError, ValueError, IndexError):
                        continue
                    boxes.append(points)
                return boxes
            # 回退：完整 predict（兼容无 det_model 属性的边界情况）
            result = self._run_ocr(ocr, image)
        except Exception as e:
            logger.warning("PaddleOCR 推理异常: %s", e)
            return None
        boxes = []
        for box, text, conf in self._iter_items(result):
            if conf < self.det_score_thresh:
                continue
            try:
                boxes.append([[int(float(p[0])), int(float(p[1]))] for p in box])
            except (TypeError, ValueError, IndexError):
                continue
        return boxes

    @staticmethod
    def _get_det_model(ocr) -> Any:
        """返回 PaddleOCR 实例内部的文本检测器（仅 det，跳过 rec）。

        3.7.0 路径：ocr.paddlex_pipeline._pipeline.text_det_model
          （paddlex_pipeline 是并行包装器 OCRPipeline，其 _pipeline 才是
           真正的 _OCRPipeline，text_det_model 是 TextDetRunnerPredictor）
        v5/3.3.2 兼容：ocr.text_detector（已废弃，保留回退）
        """
        try:
            px = getattr(ocr, "paddlex_pipeline", None)
            inner = getattr(px, "_pipeline", None) if px is not None else None
            det = getattr(inner, "text_det_model", None) if inner is not None else None
            if det is not None:
                return det
        except Exception:
            pass
        return getattr(ocr, "text_detector", None)

    # ------------------------------------------------------------------
    # 实例池访问（替代旧的 _get_ocr 单例懒加载）
    # ------------------------------------------------------------------
    def _build_config(self) -> Dict[str, Any]:
        """构建实例池初始化所需的配置字典。"""
        return {
            "lang": self.lang,
            "ocr_version": self.ocr_version,
            "det_model_dir": self.det_model_dir,
            "rec_model_dir": self.rec_model_dir,
            "drop_score": self.drop_score,
            "det_db_unclip_ratio": self.det_db_unclip_ratio,
            "det_db_box_thresh": self.det_db_box_thresh,
        }

    def _get_instance(self, slot: int = 0) -> Any:
        """获取指定槽位的 PaddleOCR 实例，失败后缓存失败状态不再重试。

        返回实例或 None（初始化失败）。

        - 服务启动时 init_ocr_pool 已预创建 N 个实例：直接按槽位取。
        - 未预创建（单实例场景/CLI/测试）：首次调用自动初始化 size=1 的池。
        - 槽位越界（池小于并发数）：回退到 0 号实例并告警，避免返回 None
          导致整页跳过；此为降级场景，正常情况下池大小 = 最大并发数。
        """
        # 之前已失败，直接返回 None
        if self._init_failed:
            return None
        instance = get_ocr_instance(slot)
        if instance is not None:
            return instance
        # 池未初始化：自动初始化 size=1 的池（向后兼容单实例场景）
        if not _OCR_POOL:
            try:
                init_ocr_pool(1, self._build_config())
            except Exception as e:
                self._init_failed = True
                self._init_error = str(e)
                logger.warning("PaddleOCR 自动初始化失败: %s", e)
                return None
            instance = get_ocr_instance(slot)
            if instance is not None:
                return instance
            # 池仍为空：初始化失败（如 paddleocr 未安装 / 模型加载异常）
            self._init_failed = True
            self._init_error = "实例池创建失败"
            logger.warning(
                "PaddleOCR 初始化失败（PP-OCRv6），本地 OCR/检测将不可用",
            )
            return None
        # 池已初始化但槽位越界：回退到 0 号实例（降级，避免整页跳过）
        logger.warning(
            "OCR 槽位 %d 越界（池大小 %d），回退到 0 号实例（可能共享，建议检查并发配置）",
            slot, len(_OCR_POOL),
        )
        return _OCR_POOL[0]

    # ------------------------------------------------------------------
    # 统一调用入口（v5 用 predict()，ocr() 已废弃）
    # ------------------------------------------------------------------
    def _run_ocr(self, ocr, image: np.ndarray):
        # predict() 是 v5 推荐方法，ocr() 会输出 DeprecationWarning
        return ocr.predict(image)

    # ------------------------------------------------------------------
    # 统一解析（v5 OCRResult dict 访问格式）
    # ------------------------------------------------------------------
    def _parse_result(
        self, result: Any, width: int, height: int
    ) -> List[OCRLine]:
        lines: List[OCRLine] = []
        for box, text, conf in self._iter_items(result):
            if conf < self.det_score_thresh:
                continue
            coords = [(int(p[0]), int(p[1])) for p in box]
            lines.append(
                OCRLine(text=text, coords=coords, confidence=float(conf))
            )
        return lines

    def _iter_items(self, result: Any):
        """迭代出 (box, text, conf) 三元组。

        v5 格式: result = [OCRResult, ...]
          OCRResult 继承 dict，通过 page['rec_texts'] / page['rec_scores']
          / page['dt_polys'] 访问数据。
          注意：getattr(page, 'rec_texts') 返回空字符串，必须用 dict 访问。
        """
        if not result:
            return
        for page in result:
            if page is None:
                continue
            # v5: OCRResult 是 dict 子类，用 dict 访问取数据
            texts = page.get("rec_texts") if isinstance(page, dict) else None
            polys = page.get("dt_polys") if isinstance(page, dict) else None
            if texts is not None and polys is not None:
                scores = page.get("rec_scores") or []
                for i, poly in enumerate(polys):
                    text = texts[i] if i < len(texts) else ""
                    conf = float(scores[i]) if i < len(scores) else 0.0
                    yield poly, text, conf
                continue
            # 回退：2.x 列表格式 [[box, (text, conf)], ...]
            if isinstance(page, list):
                for item in page:
                    if item is None:
                        continue
                    try:
                        box, tc = item
                        text, conf = tc
                        yield box, text, float(conf)
                    except (ValueError, TypeError):
                        continue
