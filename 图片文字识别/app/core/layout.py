"""文档版面分析模块（PaddleOCR v6 / PPStructureV3）。

基于 PaddleOCR 3.7.0 的 PPStructureV3（PP-DocLayout + SLANet），对文档图像做
版面切割：
  1. PP-DocLayout 检测 → 输出每个区域的 bbox + 类别（text/title/table/figure/...）
  2. XY-cut 递归切分 → 还原人类阅读顺序（多栏 / 嵌套标题正确处理）
  3. 按区域裁剪 → 交给上层 OCR 引擎逐块识别

PPStructureV3 仅做版面切割 + 表格结构识别，文本识别仍由现有 Provider 完成，
保证讯飞 / 本地两种模式都能复用版面能力。

3.x API 要点（与 2.x PPStructure 的差异）：
  - 类名：PPStructure → PPStructureV3
  - 构造参数全部更名（layout_model_dir → layout_detection_model_dir 等）
  - layout=True/table=True/ocr=False 已移除，改用 use_table_recognition 等开关
  - 推理调用：predict(img) 返回 [StructureV3Result]，取 [0]
  - 结果格式：result["layout_det_res"]["boxes"] → [{coordinate, label, score}]

XY-cut 算法说明：
  - 经典文档分析算法，递归地按 x / y 中位线切分区域集合
  - 切分时排除"跨越切分线"的大框（避免把跨栏大段落切断）
  - 无法再切时输出，天然处理"大文本块套小标题"的嵌套结构
  - 比一次性分桶（按列/按行）更鲁棒，是版面分析 30 年的标准基线

离线运行支持：
  打包后通过 _get_bundled_models_dir() 定位应用内置 layout 模型，
  避免首次运行联网下载。layout 模型名为 PP-DocLayout_plus-L。
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _get_bundled_models_dir() -> Optional[str]:
    """返回打包内置的 PaddleOCR 模型根目录，无则返回 None。

    与 paddle_local.py 中的同名函数保持一致路径策略。
    """
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
        candidates = [
            os.path.join(base, "_internal", "paddleocr_models"),
            os.path.join(base, "paddleocr_models"),
        ]
    else:
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidates = [os.path.join(base, "paddleocr_models")]

    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def _resolve_bundled_model(subpath: str) -> Optional[str]:
    """返回内置模型目录下指定子目录的完整路径，不存在返回 None。"""
    root = _get_bundled_models_dir()
    if root is None:
        return None
    full = os.path.join(root, subpath)
    return full if os.path.isdir(full) else None


def _get_paddleocr_pkg_dir() -> Optional[str]:
    """返回 paddleocr 包在磁盘上的真实目录（含 paddleocr.py）。

    与 paddle_local.py 中同名函数保持一致，用于修正 frozen 环境下
    paddleocr.__file__ 的 PYZ 虚拟路径问题。
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

    PPStructure 初始化时用 Path(__file__).parent 定位 table/layout
    字典文件，frozen 环境下 __file__ 是 PYZ 虚拟路径会导致定位失败。
    """
    if not getattr(sys, "frozen", False):
        return
    import paddleocr as _poc
    cur_file = getattr(_poc, "__file__", "") or ""
    if cur_file and os.path.isfile(cur_file):
        return
    pkg_dir = _get_paddleocr_pkg_dir()
    if pkg_dir:
        _poc.__file__ = os.path.join(pkg_dir, "paddleocr.py")
        logger.debug("已修正 paddleocr.__file__ -> %s", _poc.__file__)

# PPStructure 实例池：每个槽位持有独立 PPStructure 实例，避免多线程共享崩溃。
# PPStructure 实例本身非线程安全，多并发请求共享同一实例会崩溃或结果错乱。
# 池大小 = 最大并发数，每个并发槽位对应一个独立实例，互不干扰。
# 与 paddle_local.py 的实例池模式保持一致。
_LAYOUT_POOL: List[Any] = []  # PPStructure 实例列表（按槽位索引）
_LAYOUT_CONFIG: Optional[tuple] = None  # 已初始化时的配置快照（供调试/比对）
_LAYOUT_LOCK = threading.Lock()  # 保护池的创建，避免并发重复初始化
_LAYOUT_POOL_SIZE = 0  # 池大小，0=未初始化
_LAYOUT_INIT_FAILED = False  # 全局初始化失败标志（如 paddleocr 未安装）
_LAYOUT_INIT_ERROR: Optional[str] = None  # 全局初始化失败原因


def _create_layout_instance(
    lang: str, layout_score_thresh: float, use_table_recognition: bool = True
) -> Any:
    """创建单个 PPStructureV3 实例。失败时抛异常，由调用方决定如何处理。

    模型路径优先级（高 → 低）：
      1. 打包内置模型 _internal/paddleocr_models/
      2. PaddleOCR 默认 ~/.paddlex/official_models/（首次运行联网下载）

    参数:
      use_table_recognition: False 时跳过 SLANet 表格结构识别（每页省 30-40s），
        表格区域仍被检测但只做普通文字 OCR。默认 True 保持兼容。
    """
    # 延迟导入，避免未安装 paddleocr 时整个应用无法启动
    from paddleocr import PPStructureV3
    # frozen 环境下修正 __file__，使 PPStructureV3 内部能定位字典文件
    _fix_paddleocr_file_path()

    # PPStructureV3 3.7.0 的内部表格单元格 OCR：其 _SUPPORTED_OCR_VERSIONS 仅
    # v3/v4/v5，不支持 ocr_version="PP-OCRv6"（传了会抛 ValueError）。故改为
    # 显式指定 text_detection/recognition 的 model_name+dir 指向内置 v6 small
    # 模型（绕过版本校验，且离线可用）。显式指定内部 OCR 模型时 lang/ocr_version
    # 会被忽略并告警，故此时不传 lang/ocr_version。
    bundled_det = _resolve_bundled_model("PP-OCRv6_small_det")
    bundled_rec = _resolve_bundled_model("PP-OCRv6_small_rec")
    use_bundled_ocr = bool(bundled_det and bundled_rec)

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
    if not use_bundled_ocr:
        # 无内置 OCR 模型：交给 PPStructureV3 按 lang 自动选择（默认 v5_server，联网下载）
        kwargs["lang"] = lang

    # 版面检测置信度阈值（3.x 参数名 layout_threshold）
    if layout_score_thresh is not None:
        kwargs["layout_threshold"] = layout_score_thresh

    # 内部表格单元格 OCR 模型（v6 small，离线）
    if use_bundled_ocr:
        kwargs["text_detection_model_name"] = "PP-OCRv6_small_det"
        kwargs["text_detection_model_dir"] = bundled_det
        kwargs["text_recognition_model_name"] = "PP-OCRv6_small_rec"
        kwargs["text_recognition_model_dir"] = bundled_rec
        logger.info("使用内置 v6 small 模型作为 PPStructureV3 内部 OCR")

    # 版面检测模型（PP-DocLayoutV3，独立于内部 OCR，可同时显式指定）
    # 3.7.0 模型名为 PP-DocLayoutV3（3.3.x 旧名 PP-DocLayout_plus-L 已废弃）
    # 传 model_dir 时必须配套传匹配的 model_name（与目录内 inference.yml 的
    # Global.model_name 一致），否则 resolve_model_name 校验失败报 mismatch
    bundled_layout = _resolve_bundled_model("PP-DocLayoutV3")
    if bundled_layout:
        kwargs["layout_detection_model_name"] = "PP-DocLayoutV3"
        kwargs["layout_detection_model_dir"] = bundled_layout
        logger.info("使用内置 layout 模型: %s", bundled_layout)

    # 表格结构识别模型（5 个，离线打包）。use_table_recognition=True 时
    # PPStructureV3 需要以下模型，不显式指定会联网下载（百度 BOS 大文件
    # 下载不稳定，RT-DETR 约 124MB 经常中断），故全部内置到 paddleocr_models/。
    # 传 model_dir 时配套传 model_name（与 inference.yml 的 Global.model_name 一致）。
    # use_table_recognition=False 时跳过加载（省内存+加速启动）。
    if use_table_recognition:
        _TABLE_MODELS: list[tuple[str, str]] = [
            ("table_classification",                "PP-LCNet_x1_0_table_cls"),        # 表格分类 wired/wireless
            ("wired_table_structure_recognition",   "SLANeXt_wired"),                   # 有线表格结构
            ("wireless_table_structure_recognition","SLANet_plus"),                     # 无线表格结构
            ("wired_table_cells_detection",         "RT-DETR-L_wired_table_cell_det"),  # 有线单元格检测
            ("wireless_table_cells_detection",      "RT-DETR-L_wireless_table_cell_det"),# 无线单元格检测
        ]
        table_bundled = 0
        for param_prefix, model_name in _TABLE_MODELS:
            bundled = _resolve_bundled_model(model_name)
            if bundled:
                kwargs[f"{param_prefix}_model_name"] = model_name
                kwargs[f"{param_prefix}_model_dir"] = bundled
                table_bundled += 1
        if table_bundled:
            logger.info("已启用 %d/5 个内置表格结构识别模型", table_bundled)
    else:
        logger.info("表格结构识别已关闭（use_table_recognition=false），跳过 SLANet 模型加载")

    # 文档方向 / 文本行方向分类模型（离线）。虽然 use_doc_orientation_classify=False
    # 和 use_textline_orientation=False，但 layout_parsing / 内部 OCR 管线在 predict()
    # 时仍会惰性创建这两个模型（只是不执行推理）。不显式指定会触发 official_models
    # 锁文件机制，在打包/受限环境下可能因锁目录不可写而失败。故一并内置并显式传入。
    _AUX_MODELS: list[tuple[str, str]] = [
        ("doc_orientation_classify", "PP-LCNet_x1_0_doc_ori"),       # 文档方向分类
        ("textline_orientation",     "PP-LCNet_x1_0_textline_ori"),  # 文本行方向分类
    ]
    for param_prefix, model_name in _AUX_MODELS:
        bundled = _resolve_bundled_model(model_name)
        if bundled:
            kwargs[f"{param_prefix}_model_name"] = model_name
            kwargs[f"{param_prefix}_model_dir"] = bundled

    return PPStructureV3(**kwargs)


def init_layout_pool(
    size: int, lang: str = "ch", layout_score_thresh: float = 0.3,
    use_table_recognition: bool = True,
) -> None:
    """初始化版面分析实例池，创建 size 个独立 PPStructureV3 实例。

    已初始化且大小一致时直接返回，避免重复创建。
    创建过程异常会被捕获并记日志，部分失败时池中实例数可能少于 size
    （调用方可通过 len(_LAYOUT_POOL) 判断实际可用实例数）。
    PPStructureV3 不可用时优雅降级：池为空，版面分析自动禁用，不阻断启动。

    参数:
      use_table_recognition: False 时跳过 SLANet 表格结构识别（仅同进程模式生效）。
        子进程模式由 pipeline 传入 layout_config 控制。
    """
    global _LAYOUT_POOL, _LAYOUT_CONFIG, _LAYOUT_POOL_SIZE
    global _LAYOUT_INIT_FAILED, _LAYOUT_INIT_ERROR
    if size < 1:
        size = 1
    with _LAYOUT_LOCK:
        if _LAYOUT_POOL and _LAYOUT_POOL_SIZE == size:
            return  # 已初始化且大小一致
        _LAYOUT_POOL.clear()
        _LAYOUT_INIT_FAILED = False
        _LAYOUT_INIT_ERROR = None
        for i in range(size):
            try:
                instance = _create_layout_instance(
                    lang, layout_score_thresh, use_table_recognition
                )
                _LAYOUT_POOL.append(instance)
                logger.info(
                    "PPStructureV3 实例 %d/%d 创建成功", i + 1, size
                )
            except ImportError as e:
                # paddleocr 未安装：无需继续尝试，直接退出
                _LAYOUT_INIT_FAILED = True
                _LAYOUT_INIT_ERROR = f"paddleocr 未安装: {e}"
                logger.warning("PPStructureV3 不可用: %s", e)
                break
            except Exception as e:
                # 其它异常（如模型加载失败、OOM）：同样退出，避免重复同样错误
                _LAYOUT_INIT_FAILED = True
                _LAYOUT_INIT_ERROR = str(e)
                logger.warning(
                    "PPStructureV3 实例 %d/%d 创建失败: %s", i + 1, size, e
                )
                break
        _LAYOUT_POOL_SIZE = size
        _LAYOUT_CONFIG = (lang, layout_score_thresh, use_table_recognition)


def rebuild_layout_instance(slot: int) -> bool:
    """重建指定槽位的 PPStructureV3 实例，释放 paddle 内存池。

    与 paddle_local.rebuild_ocr_instance 配套使用：
    每 N 页同时重建 OCR + 版面分析实例，避免长文档累积内存。

    关键：PaddlePaddle 的 NaiveAllocator 内存不会随 Python GC 归还 OS。
    日志实测：仅替换池中引用（旧实例等 GC）后，重建几乎不降内存
    （4964MB→4968MB），必须显式调用 _release_paddle_memory 清理 C++ 中间张量。
    本方法复用 paddle_local 的清理流程：
      1. _release_paddle_memory(旧实例) — 清理 AnalysisPredictor 中间张量
      2. del 旧实例 + gc.collect() — 触发 Python 层析构
      3. _release_paddle_global_cache() — 清理全局内存池
      4. 创建新实例

    返回 True 表示重建成功。
    """
    global _LAYOUT_POOL
    with _LAYOUT_LOCK:
        if slot < 0 or slot >= len(_LAYOUT_POOL):
            logger.warning("重建失败：槽位 %d 越界（池大小 %d）", slot, len(_LAYOUT_POOL))
            return False
        if not _LAYOUT_CONFIG:
            logger.warning("重建失败：无可用配置快照")
            return False
        # 兼容 2 元组（旧版）和 3 元组（新增 use_table_recognition）
        if len(_LAYOUT_CONFIG) >= 3:
            lang, layout_score_thresh, use_tbl = _LAYOUT_CONFIG
        else:
            lang, layout_score_thresh = _LAYOUT_CONFIG
            use_tbl = True
        try:
            import gc
            # 1. 先显式释放旧实例的 PaddlePaddle C++ 内存池
            #    PPStructure 内部有 layout_detector / table_detector 等 predictor
            old_instance = _LAYOUT_POOL[slot]
            if old_instance is not None:
                from ..providers.paddle_local import (
                    _release_paddle_memory, _release_paddle_global_cache,
                )
                _release_paddle_memory(old_instance)
            # 2. 替换池中引用并触发 GC
            _LAYOUT_POOL[slot] = None  # 先置 None 断开引用
            del old_instance
            gc.collect()
            # 3. 再次清理 paddle 全局内存池
            from ..providers.paddle_local import _release_paddle_global_cache
            _release_paddle_global_cache()
            gc.collect()
            # 4. 创建新实例
            instance = _create_layout_instance(lang, layout_score_thresh, use_tbl)
            _LAYOUT_POOL[slot] = instance
            logger.warning("已重建槽位 %d 的 PPStructureV3 实例（内存释放）", slot)
            return True
        except Exception as e:
            logger.error("重建槽位 %d 的 PPStructureV3 实例失败: %s", slot, e)
            # 失败时尝试恢复一个可用实例，避免槽位永久失效
            try:
                instance = _create_layout_instance(lang, layout_score_thresh)
                _LAYOUT_POOL[slot] = instance
            except Exception:
                pass
            return False


def get_layout_instance(slot: int) -> Any:
    """获取指定槽位的 PPStructureV3 实例（0~size-1）。

    槽位越界或池未初始化时返回 None，由调用方决定降级策略。
    """
    if not _LAYOUT_POOL:
        return None
    if slot < 0 or slot >= len(_LAYOUT_POOL):
        return None
    return _LAYOUT_POOL[slot]


def layout_pool_size() -> int:
    """返回当前实例池中可用实例数。"""
    return len(_LAYOUT_POOL)


@dataclass
class LayoutRegion:
    """单个版面区域。"""

    # 区域类型：text / title / figure / table / header / footer / reference / equation
    type: str
    # bbox: [x_min, y_min, x_max, y_max]
    bbox: List[int]
    # 区域裁剪图（BGR numpy 数组），调用方按需取用
    crop: Optional[np.ndarray] = None
    # 区域在原图中的序号（XY-cut 排序后），从 0 开始
    order: int = 0
    # 表格 HTML（仅 type=table 且 SLANet 启用时填充，其他类型为 None）
    # 由 PPStructure 的 table 识别生成，包含 <table><tr><td> 结构
    html: Optional[str] = None


@dataclass
class LayoutResult:
    """单页版面分析结果。"""

    regions: List[LayoutRegion] = field(default_factory=list)
    # 是否成功执行版面分析（False 表示降级为整页 OCR）
    available: bool = False
    # 未执行 / 失败原因（用于日志）
    reason: str = ""
    # PPStructureV3 原生 OCR 行（整页 OCR + XY-cut 阅读顺序排序）
    # 每项 {text, bbox: [x1,y1,x2,y2], score}，已按阅读顺序排列。
    # 非空时上层直接使用，跳过逐区域 OCR 和自定义栏分析（KMeans/force_split/reorder）。
    # 为空时上层回退到逐区域 OCR 旧路径。
    native_ocr_lines: List[dict] = field(default_factory=list)


class LayoutAnalyzer:
    """版面分析器：封装 PPStructureV3，输出按阅读顺序排列的区域。

    线程安全说明：PPStructureV3 实例本身非线程安全，本类不持有自己的
    引擎实例，而是通过 init_layout_pool 预创建的实例池按 slot 取独立
    实例。每个并发槽位对应一个独立 PPStructureV3 实例，互不干扰。
    池由服务启动时（main.py）调用 init_layout_pool 初始化；未预创建
    时首次 analyze 会自动初始化 size=1 的池（向后兼容单实例场景）。

    子进程模式（推荐）：
      通过 set_subprocess_pool 注入 SubprocessLayoutPool 后，analyze 会
      优先走子进程路径，PPStructureV3 推理在独立子进程中执行，彻底隔离
      PaddlePaddle C++ 内存池。子进程退出时 OS 强制回收所有内存，
      解决 PPStructureV3 长期运行内存不释放 + 死锁问题。
      配置 use_subprocess_layout=true（默认）启用，pipeline._build_layout
      会自动注入子进程池。
    """

    def __init__(
        self,
        lang: str = "ch",
        # 版面模型阈值，低于该置信度的区域丢弃
        layout_score_thresh: float = 0.3,
        # XY-cut 切分阈值：区域数 <= 此值时不再递归
        min_regions_to_split: int = 1,
        # 切分线判定：区域跨越切分线的容差（像素），超过则视为跨栏不切断
        cross_tolerance_ratio: float = 0.02,
    ) -> None:
        self.lang = lang
        self.layout_score_thresh = layout_score_thresh
        self.min_regions_to_split = min_regions_to_split
        self.cross_tolerance_ratio = cross_tolerance_ratio
        # 自动初始化（单实例场景）失败缓存，避免每张图都重新尝试
        self._init_failed = False
        self._init_error: Optional[str] = None
        # 失败原因（供 analyze 读取后透传给 LayoutResult.reason）
        self._reason: str = ""
        # 子进程池（注入后优先使用，None 时走同进程模式）
        self._subprocess_pool: Optional["SubprocessLayoutPool"] = None

    def set_subprocess_pool(self, pool: "SubprocessLayoutPool") -> None:
        """注入子进程版面分析池，启用子进程模式。

        由 pipeline._build_layout 在配置 use_subprocess_layout=true 时调用。
        注入后 analyze 会优先走子进程路径，PPStructureV3 推理在独立子进程中执行。
        """
        self._subprocess_pool = pool

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        """是否可用（paddleocr 已安装且 PPStructureV3 可导入）。"""
        try:
            from paddleocr import PPStructureV3  # noqa: F401

            return True
        except ImportError:
            return False

    def analyze(self, image: np.ndarray, slot: int = 0, page_no=None) -> LayoutResult:
        """对一张图片做版面分析，返回按阅读顺序排列的区域列表。

        失败时返回 available=False，上层应降级为整页 OCR。

        参数:
            image: BGR numpy 数组
            slot: 实例池槽位号，从池中取对应 PPStructureV3 实例（默认 0，
                  单实例场景兼容；并发场景由上层传入任务分配的槽位号）

        执行路径：
          1. 子进程模式（_subprocess_pool 已注入）：调用 pool.analyze
             获取区域列表，失败时降级为整页 OCR
          2. 同进程模式：从实例池取 PPStructureV3 实例直接推理
        """
        # 子进程模式优先：PPStructureV3 推理在独立子进程中执行
        if self._subprocess_pool is not None:
            return self._analyze_subprocess(image, slot, page_no=page_no)
        return self._analyze_inproc(image, slot, page_no=page_no)

    def _analyze_subprocess(self, image: np.ndarray, slot: int, page_no=None) -> LayoutResult:
        """子进程模式：通过 SubprocessLayoutPool 执行版面分析。

        子进程返回的区域数据已是 dict 格式（type/bbox/html），无需调用
        PPStructureV3 实例。子进程崩溃/超时时由 pool.analyze 内部重试 2 次，
        仍失败则抛 RuntimeError，这里捕获后降级为整页 OCR（available=False）。

        子进程同时返回原生 OCR 行（ocr_lines），供上层跳过逐区域 OCR。
        """
        try:
            raw_regions, native_ocr_lines = self._subprocess_pool.analyze(image, slot)
        except Exception as e:
            logger.warning("子进程版面分析异常，降级为整页 OCR: %s", e)
            return LayoutResult(available=False, reason=f"子进程版面分析异常: {e}")

        if not raw_regions:
            return LayoutResult(available=False, reason="未检测到版面区域")

        return self._parse_raw_regions(
            raw_regions, image, page_no=page_no, native_ocr_lines=native_ocr_lines
        )

    def _analyze_inproc(self, image: np.ndarray, slot: int, page_no=None) -> LayoutResult:
        """同进程模式：从实例池取 PPStructureV3 实例直接推理。

        子进程模式不可用（配置关闭或初始化失败）时回退到此路径。
        3.x: 调用 predict(img) 返回 [StructureV3Result]，取 [0] 后用
        _result_to_regions 转换为统一的 [{type, bbox, html}, ...] 格式。
        """
        engine = self._get_instance(slot)
        if engine is None:
            return LayoutResult(available=False, reason=self._reason or "PPStructureV3 不可用")

        try:
            # 3.x: predict() 返回 [StructureV3Result, ...]，取首页
            results = engine.predict(image)
            layout_result = results[0] if results else None
        except Exception as e:
            logger.warning("PPStructureV3 版面分析异常: %s", e)
            return LayoutResult(available=False, reason=f"版面分析异常: {e}")

        if layout_result is None:
            return LayoutResult(available=False, reason="未检测到版面区域")

        # 结果转换为统一的 [{type, bbox, html}, ...] 格式
        raw_regions = self._result_to_regions(layout_result)
        if not raw_regions:
            return LayoutResult(available=False, reason="未检测到版面区域")

        # 提取原生 OCR 行（同进程模式同样从 predict() 结果中获取）
        native_ocr_lines = self._result_to_native_ocr(layout_result)
        return self._parse_raw_regions(
            raw_regions, image, page_no=page_no, native_ocr_lines=native_ocr_lines
        )

    @staticmethod
    def _result_to_regions(layout_result: Any) -> List[dict]:
        """将 PPStructureV3 的 StructureV3Result 转换为统一的区域列表。

        结果结构：
          layout_result["layout_det_res"]["boxes"] → [{coordinate, label, score}, ...]
            coordinate: [x1, y1, x2, y2]
            label: text/title/table/figure/formula/seal/...
          layout_result["table_res_list"] → 表格识别结果列表
            每项含 table_region_id（对应 boxes 索引），html["pred"] 为 HTML 字符串

        返回格式：[{type, bbox, html}, ...]
          - type: 区域类型（小写）
          - bbox: [x1, y1, x2, y2]（int 列表）
          - html: 仅 table 类型区域有值（SLANet 识别结果）
        """
        if not isinstance(layout_result, dict):
            return []

        # 提取 layout_det_res 中的区域列表
        layout_det_res = layout_result.get("layout_det_res")
        if not layout_det_res or not isinstance(layout_det_res, dict):
            return []
        boxes = layout_det_res.get("boxes") or []
        if not boxes:
            return []

        # 提取表格 HTML（{box_index: html_str}）
        table_htmls: dict = {}
        table_res_list = layout_result.get("table_res_list") or []
        for table_res in table_res_list:
            try:
                region_id = table_res.get("table_region_id")
                # table_res.html 是 HtmlMixin property，返回 {"pred": html_str}
                html_dict = table_res.html
                html_str = html_dict.get("pred") if isinstance(html_dict, dict) else None
                if isinstance(html_str, str) and html_str and region_id is not None:
                    table_htmls[int(region_id)] = html_str
            except Exception as e:
                logger.debug("提取表格 HTML 失败: %s", e)

        # 转换为统一格式
        regions: List[dict] = []
        for idx, box_info in enumerate(boxes):
            if not isinstance(box_info, dict):
                continue
            rtype = (box_info.get("label") or "text").lower()
            bbox = box_info.get("coordinate")
            if not bbox or len(bbox) != 4:
                continue
            try:
                bbox_list = [int(v) for v in bbox]
            except (TypeError, ValueError):
                continue
            html = table_htmls.get(idx) if rtype == "table" else None
            regions.append({"type": rtype, "bbox": bbox_list, "html": html})
        return regions

    @staticmethod
    def _result_to_native_ocr(layout_result: Any) -> List[dict]:
        """从 PPStructureV3 结果中提取原生 OCR 行（同进程模式）。

        与 _worker_layout.py 的 _extract_native_ocr_lines 逻辑一致：
        overall_ocr_res 提供全部文本行（text/bbox/score），
        parsing_res_list 提供 XY-cut 阅读顺序的块 bbox，用于重排行顺序。
        """
        if not isinstance(layout_result, dict):
            return []
        overall = layout_result.get("overall_ocr_res")
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
        lines = []
        for i in range(n):
            x1, y1, x2, y2 = boxes_list[i]
            lines.append({
                "text": rec_texts[i], "bbox": [x1, y1, x2, y2],
                "score": scores_list[i],
                "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2, "used": False,
            })
        parsing = layout_result.get("parsing_res_list") or []
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
            matched.sort(key=lambda ln: ln["bbox"][1])
            for ln in matched:
                ln["used"] = True
                ordered.append({"text": ln["text"], "bbox": ln["bbox"], "score": ln["score"]})
        leftover = [ln for ln in lines if not ln["used"]]
        leftover.sort(key=lambda ln: (ln["bbox"][1], ln["bbox"][0]))
        for ln in leftover:
            ordered.append({"text": ln["text"], "bbox": ln["bbox"], "score": ln["score"]})
        return ordered

    def _parse_raw_regions(
        self, raw_regions, image: np.ndarray, page_no=None,
        native_ocr_lines: Optional[List[dict]] = None,
    ) -> LayoutResult:
        """解析原始区域列表为 LayoutRegion，并执行 XY-cut 排序。

        子进程与同进程模式共用此逻辑：
          - 子进程返回：List[dict]（{type, bbox, html}）
          - 同进程返回：List[dict]（由 _result_to_regions 转换，结构一致）

        native_ocr_lines 非空时存入 LayoutResult，上层可直接使用跳过逐区域 OCR。
        """
        h, w = image.shape[:2]
        regions: List[LayoutRegion] = []
        for r in raw_regions:
            rtype = r.get("type", "text")
            bbox = r.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x_min, y_min, x_max, y_max = [int(v) for v in bbox]
            # 裁剪区域（边界保护）
            x_min = max(0, min(x_min, w - 1))
            x_max = max(x_min + 1, min(x_max, w))
            y_min = max(0, min(y_min, h - 1))
            y_max = max(y_min + 1, min(y_max, h))
            crop = image[y_min:y_max, x_min:x_max]
            # 提取表格 HTML（仅 table 类型区域）
            # 3.x: 子进程与同进程模式统一返回顶层 html 字段
            #   （子进程由 _worker_layout.py 序列化，同进程由 _result_to_regions 转换）
            region_html: Optional[str] = None
            if rtype == "table":
                html_val = r.get("html")
                if isinstance(html_val, str) and html_val:
                    region_html = html_val
            regions.append(
                LayoutRegion(
                    type=rtype, bbox=[x_min, y_min, x_max, y_max],
                    crop=crop, html=region_html,
                )
            )

        if not regions:
            return LayoutResult(available=False, reason="无有效区域")

        # XY-cut 排序
        ordered = self._xy_cut_sort(regions)
        for i, r in enumerate(ordered):
            r.order = i

        _ptag = f"P{page_no}" if page_no else "?"
        logger.info(
            "版面分析[%s]: %d 个区域，类型分布 %s",
            _ptag,
            len(ordered),
            {t: sum(1 for r in ordered if r.type == t) for t in set(r.type for r in ordered)},
        )
        # 版面坐标明细：输出每个区域的阅读顺序、类型与矩形边界框（原图坐标系）
        # 用于排查"串栏/错版"问题——核对每个 bbox 是否确实是正常的长方形、栏位是否规整
        for r in ordered:
            logger.info(
                "版面区域[%s][阅读序=%d] type=%-9s bbox=[x:%d~%d, y:%d~%d] w=%d h=%d",
                _ptag, r.order, r.type,
                r.bbox[0], r.bbox[2], r.bbox[1], r.bbox[3],
                r.bbox[2] - r.bbox[0], r.bbox[3] - r.bbox[1],
            )
        _native = native_ocr_lines or []
        if _native:
            logger.info(
                "版面分析[%s]: 原生OCR行 %d（PPStructureV3 整页OCR+XY-cut阅读顺序），"
                "上层将跳过逐区域OCR",
                _ptag, len(_native),
            )
        return LayoutResult(
            regions=ordered, available=True, native_ocr_lines=_native,
        )

    # ------------------------------------------------------------------
    # 实例池访问（替代旧的单例懒加载）
    # ------------------------------------------------------------------
    def _get_instance(self, slot: int = 0) -> Any:
        """获取指定槽位的 PPStructureV3 实例，失败后缓存失败状态不再重试。

        返回实例或 None（初始化失败）。

        - 服务启动时 init_layout_pool 已预创建 N 个实例：直接按槽位取。
        - 未预创建（单实例场景/CLI/测试）：首次调用自动初始化 size=1 的池。
        - 槽位越界（池小于并发数）：回退到 0 号实例并告警，避免返回 None
          导致整页跳过；此为降级场景，正常情况下池大小 = 最大并发数。
        """
        # 全局初始化已失败（如 paddleocr 未安装 / 模型加载异常）：直接返回 None
        if _LAYOUT_INIT_FAILED:
            self._reason = _LAYOUT_INIT_ERROR or "PPStructureV3 不可用"
            return None
        # 之前自动初始化已失败：直接返回 None
        if self._init_failed:
            self._reason = self._init_error or "PPStructureV3 不可用"
            return None
        instance = get_layout_instance(slot)
        if instance is not None:
            return instance
        # 池未初始化：自动初始化 size=1 的池（向后兼容单实例场景）
        if not _LAYOUT_POOL:
            try:
                init_layout_pool(1, self.lang, self.layout_score_thresh)
            except Exception as e:
                self._init_failed = True
                self._init_error = str(e)
                self._reason = self._init_error
                logger.warning("PPStructureV3 自动初始化失败: %s", e)
                return None
            instance = get_layout_instance(slot)
            if instance is not None:
                return instance
            # 池仍为空：初始化失败（paddleocr 未安装 / 模型加载异常）
            self._init_failed = True
            self._init_error = _LAYOUT_INIT_ERROR or "实例池创建失败"
            self._reason = self._init_error
            logger.warning("PPStructureV3 初始化失败，版面分析将不可用")
            return None
        # 池已初始化但槽位越界：回退到 0 号实例（降级，避免整页跳过）
        logger.warning(
            "Layout 槽位 %d 越界（池大小 %d），回退到 0 号实例（可能共享，建议检查并发配置）",
            slot, len(_LAYOUT_POOL),
        )
        return _LAYOUT_POOL[0]

    # ------------------------------------------------------------------
    # XY-cut 递归切分算法
    # ------------------------------------------------------------------
    def _xy_cut_sort(self, regions: List[LayoutRegion]) -> List[LayoutRegion]:
        """XY-cut 递归切分，返回按阅读顺序排列的区域。

        算法:
          1. 若区域数 <= min_regions_to_split，直接返回
          2. 计算所有区域的 x 跨度 / y 跨度
          3. 优先沿跨度更大的方向切分（多栏文档 x 跨度大，先按 x 切）
          4. 找到"空隙"作为切分线：遍历所有 bbox 边界，找到使两侧都有区域、
             且无区域跨越的最大空隙
          5. 排除跨越切分线的区域（它们属于另一侧或全局元素），递归处理两侧
          6. 无法找到切分线时，按 (y, x) 排序输出
        """
        if len(regions) <= self.min_regions_to_split:
            return list(regions)

        # 用当前区域集合的实际跨度作为 page_size（递归时子集尺寸已缩小）
        x_span = max(r.bbox[2] for r in regions) - min(r.bbox[0] for r in regions)
        y_span = max(r.bbox[3] for r in regions) - min(r.bbox[1] for r in regions)

        # 先尝试跨度更大的方向
        if x_span >= y_span:
            result = self._try_split(regions, axis=0, page_size=x_span)
            if result is not None:
                return result
            result = self._try_split(regions, axis=1, page_size=y_span)
            if result is not None:
                return result
        else:
            result = self._try_split(regions, axis=1, page_size=y_span)
            if result is not None:
                return result
            result = self._try_split(regions, axis=0, page_size=x_span)
            if result is not None:
                return result

        # 无法切分，按 (y, x) 排序兜底
        return sorted(regions, key=lambda r: (r.bbox[1], r.bbox[0]))

    def _try_split(
        self, regions: List[LayoutRegion], axis: int, page_size: int
    ) -> Optional[List[LayoutRegion]]:
        """尝试沿指定轴切分。

        axis: 0=x（垂直切分线，分左右），1=y（水平切分线，分上下）
        page_size: 页面在该轴上的尺寸（用于计算跨栏容差）
        返回切分后的有序列表，无法切分时返回 None。
        """
        tolerance = int(page_size * self.cross_tolerance_ratio)

        # 收集所有区域在该轴上的 [start, end]
        # axis=0 时用 bbox[0]/bbox[2]（x_min/x_max）
        # axis=1 时用 bbox[1]/bbox[3]（y_min/y_max）
        starts = [r.bbox[axis] for r in regions]
        ends = [r.bbox[axis + 2] for r in regions]

        # 候选切分线：所有 end 和 start 的中间点
        # 寻找最大空隙：按 start 排序，找相邻两区域间的最大间隙
        sorted_indices = sorted(range(len(regions)), key=lambda i: starts[i])
        sorted_starts = [starts[i] for i in sorted_indices]
        sorted_ends = [ends[i] for i in sorted_indices]

        # 计算累积：在位置 p 处，左侧有多少区域完全在 p 之前结束
        # 找一个 p 使得：左侧至少 1 个区域结束，右侧至少 1 个区域开始，
        # 且没有区域跨越 p（即所有区域的 [start,end] 要么全在 p 左，要么全在 p 右）
        # 简化版：找最大空隙
        best_gap = 0
        best_split = -1
        for i in range(len(regions) - 1):
            # 当前区域的最右端
            right_end = max(sorted_ends[: i + 1])
            # 下一区域的最左端
            left_start = sorted_starts[i + 1]
            gap = left_start - right_end
            if gap > best_gap and gap > tolerance:
                # 验证：没有区域跨越切分线
                split_pos = (right_end + left_start) / 2
                left_count = sum(1 for r in regions if r.bbox[axis + 2] <= split_pos + tolerance)
                right_count = sum(1 for r in regions if r.bbox[axis] >= split_pos - tolerance)
                # 跨越的区域（既不纯左也不纯右）排除，但要求两侧都至少有 1 个
                if left_count >= 1 and right_count >= 1:
                    best_gap = gap
                    best_split = split_pos

        if best_split < 0:
            return None

        # 分组：纯左 / 纯右 / 跨越（跨越的视为全局元素，按位置插入）
        left = []
        right = []
        cross = []
        for r in regions:
            if r.bbox[axis + 2] <= best_split + tolerance:
                left.append(r)
            elif r.bbox[axis] >= best_split - tolerance:
                right.append(r)
            else:
                # 跨越切分线的区域（如跨栏标题），单独处理
                cross.append(r)

        # 递归处理两侧
        left_sorted = self._xy_cut_sort(left) if left else []
        right_sorted = self._xy_cut_sort(right) if right else []

        # 跨越的区域：按其 y 中心插入到结果中合适的位置
        # （通常是跨栏标题 / 页眉，应出现在与其纵向位置对应的阅读流中）
        result: List[LayoutRegion] = []
        if not cross:
            result = left_sorted + right_sorted
        else:
            # 将 cross 按 y 排序，依次插入到 left_sorted + right_sorted 的合适位置
            # 简化策略：cross 区域先于两侧输出（页眉/跨栏标题通常在顶部）
            result = cross + left_sorted + right_sorted
        return result


# ======================================================================
# 子进程版面分析池（SubprocessLayoutPool）
# ----------------------------------------------------------------------
# 与 SubprocessOCRPool 对齐的实现：通过子进程隔离 PPStructureV3 推理，
# 彻底解决 PPStructureV3 长期运行内存不释放 + 死锁问题。
#
# 关键设计：
#   - 每个 slot 维护一个长期运行的子进程，处理 batch_size 页后自动重启
#   - 通过 stdin/stdout 用 JSON + base64 通信
#   - _recv 超时 90s（与 OCR 一致），避免 stall 检测与重试竞态
#   - _restart_proc 加锁，防止主线程 stall 检测与 worker 线程重试同时调用
#   - kill 旧进程后显式关闭 stdout/stdin，让 reader_thread 收到 EOF 退出
# ======================================================================
_WORKER_LAYOUT_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "providers", "_worker_layout.py",
)


def _kill_process_tree(proc) -> None:
    """强制杀死子进程及其全部后代（修孤儿内存泄漏，与 SubprocessOCRPool 对齐）。

    sys.executable 在 venv 下是 .venv\\Scripts\\python.exe（7MB 启动器 shim），
    它会再 spawn 真正的 Python312\\python.exe worker（1-2GB，持有 paddle 模型）。
    Popen.kill() 只杀启动器，孙进程 worker 成为孤儿继续占 CPU/内存。本函数用
    Windows 原生 taskkill /T 杀整棵树，确保孙进程一起回收。

    优先级：taskkill /F /T → psutil 递归 kill → Popen.kill() 兜底。
    """
    import subprocess as _sp
    pid = getattr(proc, "pid", 0)
    if not pid:
        try:
            proc.kill()
        except Exception:
            pass
        return
    # 1. 优先 taskkill /T：Windows 原生进程树杀，能杀到孙进程
    try:
        r = _sp.run(
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
    shim 会再 spawn 真 python，孙进程可能逃过 kill 并一直占用 stderr 句柄 →
    read() 永不返回（无超时），且此刻正在持锁 → 该槽位被永久锁死，任务卡死。
    这里用线程 + Event 加超时：超时返回空串，放弃的 daemon 线程不占锁。
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


class SubprocessLayoutPool:
    """子进程版面分析实例池。

    每个 slot 对应一个长期运行的子进程，处理 batch_size 页后自动重启
    释放内存。线程安全：每个 slot 有独立锁。

    与 SubprocessOCRPool 的差异：
      - 通信消息类型：init / analyze / exit（无 detect）
      - 返回数据：版面区域列表 [{type, bbox, html}, ...]
      - 内存阈值更低（2000MB vs 1500MB），因 PPStructureV3 内存占用较小
    """

    def __init__(self, pool_size: int, layout_config: Dict[str, Any],
                 batch_size: int = 5, cpu_threads: int = 0) -> None:
        # batch_size=5：每个子进程处理 5 页后重启，与 _worker_layout.py
        # 的内存监控间隔对齐
        self._pool_size = pool_size
        self._layout_config = layout_config
        self._batch_size = max(1, batch_size)
        # 每子进程 CPU 线程数：0=自动，>0 固定值
        self._cpu_threads = max(0, int(cpu_threads))
        # 每个 slot 的子进程句柄和锁
        self._procs: List[Any] = [None] * pool_size  # subprocess.Popen
        self._locks = [threading.Lock() for _ in range(pool_size)]
        # 重启锁：防止主线程 stall 检测与 worker 线程 analyze() 重试
        # 同时调用 _restart_proc 产生竞态
        self._restart_locks = [threading.Lock() for _ in range(pool_size)]
        # 每个 slot 已处理的页数（用于决定何时重启）
        self._page_counts: List[int] = [0] * pool_size
        # 每个 slot 的子进程启动时间（用于 stall 诊断时计算运行时长）
        self._start_times: List[float] = [0.0] * pool_size
        # 环境变量（必须在子进程启动前设置）
        self._env = self._build_env()

    def _build_env(self) -> Dict[str, str]:
        """构建子进程环境变量。

        与 SubprocessOCRPool._build_env 对齐，但使用 _LAYOUT_WORKER_MODE
        标记子进程为版面分析 worker 模式，让 main.py 跳转到 _worker_layout.py。

        CPU 线程分配：版面分析（PPStructureV3）是辅助任务（每页 0.4-3s），
        与 OCR 子进程共用 CPU。按 pool_size 的一半核数分配
        （threads = cpu_count // (2*pool_size)），避免 OCR+layout 双双满载超订。
        固定值 cpu_threads>0 时优先。
        """
        env = os.environ.copy()
        # PaddlePaddle 内存分配器策略（必须在 import paddle 之前设置）
        # 与 SubprocessOCRPool._build_env 对齐：naive_best_fit + eager_delete 系列
        # （官方推荐，见 PaddleOCR #11639 / 讨论 #14497），控制 PPStructureV3 内存增长。
        env["FLAGS_allocator_strategy"] = "naive_best_fit"
        env["FLAGS_eager_delete_scope"] = "True"
        env["FLAGS_eager_delete_tensor_gb"] = "0.0"
        env["FLAGS_fast_eager_deletion_mode"] = "True"
        env["FLAGS_use_pinned_memory"] = "False"
        env["FLAGS_fraction_of_cpu_memory_to_use"] = "0.1"
        env["FLAGS_initial_cpu_memory_in_mb"] = "128"
        # CPU 线程数：OpenMP/MKL/OpenBLAS 在子进程 import paddle 前读取
        if self._cpu_threads > 0:
            threads = self._cpu_threads
        else:
            cpu_count = os.cpu_count() or 4
            threads = max(1, cpu_count // max(2, 2 * self._pool_size))
        env["OMP_NUM_THREADS"] = str(threads)
        env["MKL_NUM_THREADS"] = str(threads)
        env["OPENBLAS_NUM_THREADS"] = str(threads)
        # 标记子进程为版面分析 worker 模式
        # 打包后 sys.executable 是 server-paddle.exe，会执行 main.py 的 main()
        # 设置此环境变量后，main.py 顶部检测到并跳转执行 _worker_layout.py 主循环
        env["_LAYOUT_WORKER_MODE"] = "1"
        # 同时清除 OCR worker 标记，避免误跳转
        env.pop("_OCR_WORKER_MODE", None)
        return env

    def _ensure_proc(self, slot: int):
        """确保指定 slot 的子进程已启动。"""
        import subprocess
        proc = self._procs[slot]
        if proc is not None and proc.poll() is None:
            return proc
        # 启动新子进程
        proc = subprocess.Popen(
            [sys.executable, _WORKER_LAYOUT_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._env,
            text=False,  # 用二进制模式，避免编码问题
            bufsize=0,
        )
        # 发送初始化配置
        init_msg = {"type": "init", "config": self._layout_config}
        self._send(proc, init_msg)
        resp = self._recv(proc)
        if resp.get("type") != "ready":
            raise RuntimeError(f"版面分析子进程初始化失败: {resp}")
        self._procs[slot] = proc
        self._page_counts[slot] = 0
        self._start_times[slot] = time.time()
        logger.info("子进程版面分析 worker %d 已启动 (pid=%d)", slot, proc.pid)
        return proc

    def _kill_proc(self, slot: int) -> None:
        """仅 kill 指定 slot 的子进程，不启动新进程。

        与 _restart_proc 的区别：
          - _restart_proc: kill 旧进程 + 启动新进程 + 等待新进程加载模型 ready（10-20s）
          - _kill_proc: 仅 kill 旧进程，不启动新进程（<1s）

        用途：delete_task / pause_task 只需切断 PPStructureV3 推理让 worker 线程退出，
        不需要立即启动新子进程。下次 analyze() 调用时 _ensure_proc 会自动懒启动。

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
            logger.info("子进程版面分析 worker %d 已 kill（不重启，下次懒启动）", slot)

    def _restart_proc(self, slot: int):
        """重启指定 slot 的子进程，释放内存。

        加 _restart_locks[slot] 锁，防止主线程 stall 检测与 worker 线程 analyze()
        重试同时调用产生竞态。kill 旧进程后显式关闭 stdout/stdin，让所有阻塞在
        readline() 上的 reader_thread 立即收到 EOF 返回 b"" 退出。
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
            logger.info("子进程版面分析 worker %d 重启（释放内存）", slot)
            return self._ensure_proc(slot)

    @staticmethod
    def _send(proc, msg: dict, timeout: float = 30.0) -> None:
        """发送 JSON 消息（以换行符分隔）。

        带超时保护（关键防死锁）：Windows 匿名管道缓冲仅数 KB，base64 图片
        消息可达 15MB+。若子进程卡死（paddle 推理死锁）不再消费 stdin，
        write() 会永久阻塞且无任何日志——与 OCR 子进程同款死锁，用线程 +
        Event 加超时，超时抛 RuntimeError，由上层 analyze() 捕获后重启子进程。
        """
        import threading
        import json as _json
        data = (_json.dumps(msg) + "\n").encode("utf-8")
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
            name=f"layout-send-{proc_pid}",
        )
        writer_thread.start()
        if not done_event.wait(timeout=timeout):
            raise RuntimeError(
                f"子进程版面分析发送消息超时（{timeout}s），疑似子进程卡死不消费 stdin"
            )
        if result_holder["err"] is not None:
            raise result_holder["err"]

    @staticmethod
    def _recv(proc, timeout: float = 90.0) -> dict:
        """接收一行 JSON 消息。

        跳过非 JSON 行（防御性：即使 worker 有残留 stdout 输出也不会崩溃）。
        子进程崩溃时 readline() 返回空，此时读取 stderr 获取崩溃信息。
        超时未收到响应视为子进程卡死（paddle 死锁），抛异常由上层重试。

        参数:
            timeout: 单次读取超时秒数，默认 90s
                     版面分析正常 1-5s，复杂表格 SLANet <30s，超过 90s 基本是死锁
                     与 OCR 子进程对齐：2 次重试 = 180s < stall_timeout 300s，避免竞态
        """
        import json as _json
        import threading as _threading

        max_skips = 50  # 最多跳过 50 行非 JSON，防止死循环
        # 用 proc 的 pid 作为线程名标识，便于 stall 诊断时统计残留 reader_thread 数
        proc_pid = getattr(proc, "pid", 0)
        for _ in range(max_skips):
            # Windows pipe 不支持 select，用线程+Event 实现超时读取
            result_holder = {"line": None}
            read_event = _threading.Event()

            def _read_line():
                try:
                    result_holder["line"] = proc.stdout.readline()
                except Exception:
                    result_holder["line"] = b""
                finally:
                    read_event.set()

            # 命名规则 recv-layout-{pid}：诊断时通过 threading.enumerate() 统计
            # 与 OCR 子进程的 recv-line- 前缀区分
            reader_thread = _threading.Thread(
                target=_read_line, daemon=True,
                name=f"recv-layout-{proc_pid}",
            )
            reader_thread.start()
            if not read_event.wait(timeout=timeout):
                # 超时：子进程卡死（paddle 死锁），reader_thread 仍在阻塞 readline
                # daemon=True 会随主进程退出，不会泄漏
                alive_count = sum(
                    1 for t in _threading.enumerate()
                    if t.name.startswith("recv-layout-") and t.is_alive()
                )
                logger.warning(
                    "[recv超时] layout pid=%d 超时%ss, 当前残留 reader_thread=%d",
                    proc_pid, timeout, alive_count,
                )
                raise RuntimeError(
                    f"子进程版面分析响应超时（{timeout}s），疑似 paddle 死锁"
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
                    f"子进程版面分析意外退出 (returncode={returncode}): {stderr_msg}"
                )
            try:
                return _json.loads(line.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                # 非 JSON 行（残留输出），记日志并跳过
                logger.warning("跳过非 JSON 行: %s", line[:200])
                continue
        raise RuntimeError("连续 50 行非 JSON 输出，子进程版面分析通信异常")

    def analyze(self, image: np.ndarray, slot: int = 0) -> tuple:
        """对图片做版面分析，返回 (区域列表, 原生OCR行)。

        返回格式：(regions, ocr_lines)
          - regions: [{type, bbox: [x1,y1,x2,y2], html}, ...]
            type: 区域类型（text/title/table/figure/...）
            bbox: 边界框
            html: 仅 table 类型区域有值（SLANet 识别结果）
          - ocr_lines: [{text, bbox, score}, ...] PPStructureV3 原生 OCR 行
            （整页 OCR + XY-cut 阅读顺序排序），非空时上层跳过逐区域 OCR。

        子进程崩溃自动恢复：_recv 检测到子进程退出会抛 RuntimeError，
        这里捕获后重启子进程并重试本次分析。最多重试 1 次（共 2 次尝试），
        与 OCR 子进程对齐：2 次尝试 × 90s = 180s < stall_timeout 300s，避免竞态。
        """
        import base64
        import time as _time

        with self._locks[slot]:
            # 图片编码为 base64 —— 只编码一次，重试时复用
            t_enc_start = _time.time()
            img_bytes = image.tobytes()
            img_b64 = base64.b64encode(img_bytes).decode("ascii")
            t_enc = _time.time() - t_enc_start

            msg = {
                "type": "analyze",
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
                        # worker 内部 PPStructureV3 失败（如模型加载错误），不算崩溃
                        raise RuntimeError(f"子进程版面分析失败: {resp.get('message')}")
                    self._page_counts[slot] += 1
                    regions = resp.get("regions", [])
                    ocr_lines = resp.get("ocr_lines", [])
                    logger.debug(
                        "子进程版面分析[slot=%d]: 编码 %.2fs + 通信+推理 %.2fs = %.2fs | "
                        "图片 %dx%d | 区域 %d 个 | 原生OCR行 %d",
                        slot, t_enc, t_comm, t_enc + t_comm,
                        image.shape[1], image.shape[0], len(regions), len(ocr_lines),
                    )
                    # 达到 batch_size 后重启释放内存
                    if self._page_counts[slot] >= self._batch_size:
                        self._restart_proc(slot)
                    return regions, ocr_lines
                except RuntimeError as e:
                    last_error = e
                    # 子进程崩溃（意外退出/通信异常）：重启并重试
                    logger.warning(
                        "子进程版面分析 slot=%d 第 %d 次尝试失败，重启子进程: %s",
                        slot, attempt + 1, e,
                    )
                    try:
                        self._restart_proc(slot)
                    except Exception as restart_err:
                        logger.error("子进程版面分析重启失败: %s", restart_err)
                    if attempt < 1:
                        _time.sleep(1)
            # 2 次都失败，抛出异常由上层 analyze 方法捕获后降级为整页 OCR
            raise RuntimeError(
                f"子进程版面分析连续 2 次失败 (slot={slot}): {last_error}"
            )

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
        logger.info("所有子进程版面分析 worker 已关闭")
