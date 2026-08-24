"""文档处理流水线（服务端版：仅本地 PaddleOCR）。

职责：
  1. 根据配置构建本地 PaddleOCR 引擎、过滤器、文档处理器。
  2. 调度单文件 / 批量文件处理。
  3. 调用输出转换器保存结果。

服务端版本固定使用 paddle_local 作为 OCR 引擎与第二层检测器，
不依赖讯飞云 OCR，所有计算在本地 CPU 完成，适合局域网离线运行。

内存优化：默认使用子进程模式（subprocess_ocr）隔离 PaddleOCR 推理，
每 N 页重启子进程释放 C++ 内存池。实测 200 页 PDF 内存稳定在 400MB，
而同进程模式会涨到 4GB+。配置 use_subprocess_ocr=false 可回退同进程模式。
"""
from __future__ import annotations

import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

from ..providers.base import BaseProvider
from ..providers.paddle_local import PaddleLocalProvider
from ..utils import output as out_mod
from ..utils.config import load_config
from .document.base import BaseHandler, DocumentResult
from .document.docx_handler import DocxHandler
from .document.image_handler import ImageHandler
from .document.pdf_handler import PdfHandler
from .filter import DualLayerFilter, FilterConfig
from .layout import LayoutAnalyzer

logger = logging.getLogger(__name__)

# 进度回调签名: (current, total, message)
ProgressCb = Callable[[int, int, str], None]


class Pipeline:
    """处理流水线（服务端版，仅本地 PaddleOCR）。"""

    def __init__(self, config: Optional[Dict] = None) -> None:
        self.cfg = config or load_config()
        self._paddle: Optional[BaseProvider] = None
        self._subprocess_pool = None  # 子进程池引用（用于 shutdown）
        self._layout_pool = None  # 子进程版面分析池引用（用于 shutdown）
        self._handlers: List[BaseHandler] = []
        self._build()

    # ------------------------------------------------------------------
    # 构建组件
    # ------------------------------------------------------------------
    def _build(self) -> None:
        # 服务端版本：provider_mode 固定为 paddle_local
        provider_mode = "paddle_local"
        # 构建 PaddleOCR（必须可用）
        paddle = self._build_paddle()
        if paddle is None or not paddle.is_available():
            raise RuntimeError(
                "PaddleOCR 不可用，请安装: "
                "pip install paddlepaddle==3.2.0 paddleocr==3.3.2"
            )
        ocr_engine: BaseProvider = paddle
        # 第二层检测复用 PaddleLocalProvider.detect_boxes
        detector = paddle
        enable_l2 = True

        # 过滤配置
        fcfg = self.cfg.get("filter", {})
        filter_cfg = FilterConfig(
            edge_density_threshold=fcfg.get("edge_density_threshold", 0.03),
            variance_threshold=fcfg.get("variance_threshold", 15.0),
            connected_min_area=fcfg.get("connected_min_area", 30),
            enable_layer2=fcfg.get("enable_layer2", True) and enable_l2,
        )
        self.flt = DualLayerFilter(filter_cfg, detector=detector)
        self.ocr_engine = ocr_engine
        self.detector = detector
        self.provider_mode = provider_mode

        # 版面分析器（可选，enable_layout=true 且 PPStructure 可用时启用）
        self._layout = self._build_layout()

        # 文档处理器
        dpi = self.cfg.get("render_dpi", 200)
        # 每 N 页重建 OCR 实例，释放 paddle 内存池累积的中间张量
        # 默认 10 页重建一次（日志实测 30 页太晚，内存已涨到 3GB+）
        # 0 表示不重建
        # 注意：子进程模式下由 SubprocessOCRPool 自动重启，此参数仅用于同进程模式
        rebuild_every = int(self.cfg.get("rebuild_ocr_every_pages", 10))
        # 子进程模式下不需要 rebuild_every（子进程自动重启释放内存）
        if self.cfg.get("use_subprocess_ocr", True):
            rebuild_every = 0
        # 版面分析后是否做整页补充识别（默认 True 保质量；追求速度可关，
        # 省去每页一次整页 OCR，约 6-10s/页，对版面规整文档影响小）
        supplement_ocr = bool(self.cfg.get("layout_supplement_ocr", True))
        self._handlers = [
            ImageHandler(ocr_engine, self.flt, layout_analyzer=self._layout,
                         supplement_ocr=supplement_ocr),
            PdfHandler(
                ocr_engine, self.flt,
                render_dpi=dpi, layout_analyzer=self._layout,
                rebuild_every_pages=rebuild_every,
                supplement_ocr=supplement_ocr,
            ),
            DocxHandler(ocr_engine, self.flt, layout_analyzer=self._layout,
                        supplement_ocr=supplement_ocr),
        ]

    def _build_paddle(self) -> Optional[BaseProvider]:
        if self._paddle is not None:
            return self._paddle
        pcfg = self.cfg.get("paddle", {})
        # 子进程模式：彻底隔离 PaddleOCR，每 N 页重启子进程释放 C++ 内存池
        # 实测：同进程模式 200 页涨到 4GB+，子进程模式稳定在 400MB
        use_subprocess = self.cfg.get("use_subprocess_ocr", True)
        if use_subprocess:
            try:
                from ..providers.subprocess_ocr import SubprocessOCRProvider
                # 子进程批次大小：每处理 batch_size 页重启一次子进程
                # 默认 5 页（与 _worker_ocr.py 内存监控间隔对齐）：
                #   - 每页检查内存阈值（1500MB），超限立即退出
                #   - 每 5 页输出内存日志，然后重启释放 paddle 内存池
                #   - 配合 FLAGS_fraction_of_cpu_memory_to_use=0.1，双重保险
                batch_size = int(self.cfg.get("subprocess_batch_size", 5))
                # 每子进程 CPU 线程数：0=自动（cpu_count//pool_size），>0 固定值
                cpu_threads = int(self.cfg.get("subprocess_cpu_threads", 0))
                # 进程池大小必须等于 max_concurrent，每个并发任务用独立子进程
                # 之前硬编码 pool_size=1，max_concurrent>1 时 slot 越界
                pool_size = max(1, int(self.cfg.get("max_concurrent", 1)))
                p = SubprocessOCRProvider(
                    lang=pcfg.get("lang", "ch"),
                    use_gpu=pcfg.get("use_gpu", False),
                    ocr_version=pcfg.get("ocr_version", "PP-OCRv6"),
                    det_model_dir=pcfg.get("det_model_dir", ""),
                    rec_model_dir=pcfg.get("rec_model_dir", ""),
                    det_score_thresh=pcfg.get("det_score_thresh", 0.3),
                    drop_score=pcfg.get("drop_score", 0.0),
                    det_db_unclip_ratio=pcfg.get("det_db_unclip_ratio", 1.8),
                    det_db_box_thresh=pcfg.get("det_db_box_thresh", 0.5),
                    batch_size=batch_size,
                    pool_size=pool_size,
                    cpu_threads=cpu_threads,
                )
                if p.is_available():
                    self._paddle = p
                    self._subprocess_pool = getattr(p, "_pool", None)
                    logger.info(
                        "使用子进程 OCR 模式（pool_size=%d, batch_size=%d）",
                        pool_size, batch_size,
                    )
                    return p
                logger.warning("子进程 OCR 不可用，回退同进程模式")
            except Exception as e:
                logger.warning("子进程 OCR 初始化失败，回退同进程模式: %s", e)
        # 同进程模式（回退）
        p = PaddleLocalProvider(
            lang=pcfg.get("lang", "ch"),
            use_gpu=pcfg.get("use_gpu", False),
            ocr_version=pcfg.get("ocr_version", "PP-OCRv6"),
            det_model_dir=pcfg.get("det_model_dir", ""),
            rec_model_dir=pcfg.get("rec_model_dir", ""),
            det_score_thresh=pcfg.get("det_score_thresh", 0.3),
            drop_score=pcfg.get("drop_score", 0.0),
            det_db_unclip_ratio=pcfg.get("det_db_unclip_ratio", 1.8),
            det_db_box_thresh=pcfg.get("det_db_box_thresh", 0.5),
        )
        if p.is_available():
            self._paddle = p
            return p
        return None

    def _build_layout(self) -> Optional[LayoutAnalyzer]:
        """构建版面分析器。

        启用条件：paddle.enable_layout=true（默认）且 PPStructureV3 可用。
        不可用时静默降级为整页 OCR（不报错，仅记日志）。

        子进程模式（推荐，默认启用）：
          配置 use_subprocess_layout=true 时，创建 SubprocessLayoutPool 并注入
          到 LayoutAnalyzer。PPStructureV3 推理在独立子进程中执行，彻底隔离
          PaddlePaddle C++ 内存池，解决长期运行内存不释放 + 死锁问题。
          与 OCR 子进程模式对齐：每 batch_size 页自动重启子进程释放内存。

        同进程模式（回退）：
          配置 use_subprocess_layout=false 时，PPStructureV3 在主进程中运行，
          依赖 init_layout_pool 预创建的实例池。长文档会累积内存，仅用于
          调试或子进程模式不可用时回退。
        """
        pcfg = self.cfg.get("paddle", {})
        if not pcfg.get("enable_layout", True):
            return None
        analyzer = LayoutAnalyzer(lang=pcfg.get("lang", "ch"))
        if analyzer.is_available():
            # 表格结构识别开关（可配置）：false 时跳过 SLANet+RT-DETR，PPStructure
            # 每页耗时从 38-52s 降至 5-15s。关闭后表格区域仍被检测，只做普通文字 OCR。
            use_table_recognition = bool(pcfg.get("use_table_recognition", True))
            # 子进程模式：创建 SubprocessLayoutPool 并注入到 analyzer
            # 默认启用（与 OCR 子进程模式对齐），可通过 use_subprocess_layout=false 关闭
            use_subprocess_layout = self.cfg.get("use_subprocess_layout", True)
            if use_subprocess_layout:
                try:
                    from .layout import SubprocessLayoutPool
                    # 解析内置模型路径（与 _create_layout_instance 逻辑一致）
                    # 3.7.0 版面检测模型名为 PP-DocLayoutV3（3.3.x 旧名 PP-DocLayout_plus-L）
                    # PPStructureV3 3.7.0 不支持 ocr_version="PP-OCRv6"，内部表格 OCR
                    # 改用显式指定的 v6 small 模型目录（绕过版本校验，离线可用）
                    from .layout import _resolve_bundled_model
                    layout_detection_model_dir = _resolve_bundled_model(
                        "PP-DocLayoutV3"
                    ) or ""
                    bundled_det = _resolve_bundled_model("PP-OCRv6_small_det") or ""
                    bundled_rec = _resolve_bundled_model("PP-OCRv6_small_rec") or ""
                    has_internal_ocr = bool(bundled_det and bundled_rec)
                    # 表格结构识别模型（5 个，离线）。百度 BOS 大文件下载不稳定，
                    # 故全部内置到 paddleocr_models/，显式传 dir 避免联网下载。
                    # use_table_recognition=false 时跳过模型加载（子进程也不会创建 SLANet）
                    _table_models = [
                        ("table_classification",                 "PP-LCNet_x1_0_table_cls"),
                        ("wired_table_structure_recognition",    "SLANeXt_wired"),
                        ("wireless_table_structure_recognition", "SLANet_plus"),
                        ("wired_table_cells_detection",          "RT-DETR-L_wired_table_cell_det"),
                        ("wireless_table_cells_detection",       "RT-DETR-L_wireless_table_cell_det"),
                    ]
                    table_model_paths = {
                        prefix: (_resolve_bundled_model(name) or "", name)
                        for prefix, name in _table_models
                    } if use_table_recognition else {}
                    # 文档方向 / 文本行方向分类模型（predict 时惰性创建，需离线）
                    _aux_models = [
                        ("doc_orientation_classify", "PP-LCNet_x1_0_doc_ori"),
                        ("textline_orientation",     "PP-LCNet_x1_0_textline_ori"),
                    ]
                    aux_model_paths = {
                        prefix: (_resolve_bundled_model(name) or "", name)
                        for prefix, name in _aux_models
                    }
                    layout_config = {
                        # 仅在无内置内部 OCR 时传 lang（让 PPStructureV3 自动选 v5_server）
                        "lang": pcfg.get("lang", "ch") if not has_internal_ocr else "",
                        # 内部表格单元格 OCR（v6 small，离线）
                        "text_detection_model_name": "PP-OCRv6_small_det" if bundled_det else "",
                        "text_detection_model_dir": bundled_det,
                        "text_recognition_model_name": "PP-OCRv6_small_rec" if bundled_rec else "",
                        "text_recognition_model_dir": bundled_rec,
                        # 版面检测模型（传 dir 时配套传 name，与 yml 内 Global.model_name 一致）
                        "layout_detection_model_name": "PP-DocLayoutV3" if layout_detection_model_dir else "",
                        "layout_detection_model_dir": layout_detection_model_dir,
                        # 表格结构识别开关（传给子进程 worker）
                        "use_table_recognition": use_table_recognition,
                        # 表格结构识别模型（5 个，离线）
                        **{
                            f"{prefix}_model_name": name if path else ""
                            for prefix, (path, name) in table_model_paths.items()
                        },
                        **{
                            f"{prefix}_model_dir": path
                            for prefix, (path, _) in table_model_paths.items()
                        },
                        # 文档方向 / 文本行方向分类模型（离线）
                        **{
                            f"{prefix}_model_name": name if path else ""
                            for prefix, (path, name) in aux_model_paths.items()
                        },
                        **{
                            f"{prefix}_model_dir": path
                            for prefix, (path, _) in aux_model_paths.items()
                        },
                        "layout_threshold": pcfg.get("det_score_thresh", 0.3),
                    }
                    batch_size = int(self.cfg.get("subprocess_layout_batch_size", 5))
                    pool_size = max(1, int(self.cfg.get("max_concurrent", 1)))
                    # 每子进程 CPU 线程数：0=自动（cpu_count//(2*pool_size)），>0 固定值
                    cpu_threads = int(self.cfg.get("subprocess_cpu_threads", 0))
                    pool = SubprocessLayoutPool(
                        pool_size=pool_size,
                        layout_config=layout_config,
                        batch_size=batch_size,
                        cpu_threads=cpu_threads,
                    )
                    analyzer.set_subprocess_pool(pool)
                    # 保存引用供 shutdown 时关闭子进程
                    self._layout_pool = pool
                    logger.info(
                        "使用子进程版面分析模式（pool_size=%d, batch_size=%d）",
                        pool_size, batch_size,
                    )
                except Exception as e:
                    logger.warning(
                        "子进程版面分析初始化失败，回退同进程模式: %s", e
                    )
            return analyzer
        logger.info("版面分析器不可用（PPStructureV3 未安装），降级为整页 OCR")
        return None

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def supported_extensions(self) -> Tuple[str, ...]:
        exts = []
        for h in self._handlers:
            exts.extend(h.extensions)
        return tuple(exts)

    def process_file(
        self,
        path: str,
        progress_cb: Optional[ProgressCb] = None,
        slot: int = 0,
        slots: Optional[List[int]] = None,
    ) -> Tuple[DocumentResult, str]:
        """处理单个文件，返回 (结果, 输出路径)。

        参数:
            path: 输入文件路径
            progress_cb: 进度回调
            slot: 实例池槽位号（单进程模式，默认0）
            slots: 多槽位列表（页级并行模式），传入时启用多进程并行处理PDF
                   每个槽位处理一个页范围，最后合并结果
        """
        handler = self._get_handler(path)
        if handler is None:
            raise ValueError(f"不支持的文件类型: {path}")
        if progress_cb:
            progress_cb(0, 1, f"开始处理: {os.path.basename(path)}")

        fmt = self.cfg.get("output_format", "original")
        out_dir = self.cfg.get("output_dir", "")
        dpi = self.cfg.get("render_dpi", 200)
        src_ext = os.path.splitext(path)[1].lower()

        # 增量写入路径：PDF 源 + 输出为 PDF（original 或 searchable_pdf）
        # 每页处理完立即落盘，中途崩溃也能保留已处理页
        use_incremental = src_ext == ".pdf" and fmt in ("original", "searchable_pdf")
        if use_incremental and slots and len(slots) > 1:
            # 多进程并行处理PDF：按页切分，每个slot处理一个页范围
            result, out_path = self._process_pdf_parallel(
                handler, path, progress_cb, out_dir, dpi, slots
            )
        elif use_incremental:
            result, out_path = self._process_pdf_incremental(
                handler, path, progress_cb, out_dir, dpi, slot
            )
        else:
            # 普通路径：全部处理完再统一输出
            result = handler.process(path, progress_cb=progress_cb, slot=slot)
            out_path = out_mod.save_output(result, fmt, out_dir, render_dpi=dpi)

        if progress_cb:
            progress_cb(1, 1, f"完成: {out_path}")
        return result, out_path

    def _process_pdf_incremental(
        self, handler, path: str, progress_cb, out_dir: str, dpi: int, slot: int = 0
    ) -> Tuple[DocumentResult, str]:
        """PDF 增量处理：每页处理完立即写入，中途崩溃不丢失已处理页。

        采用「单页独立文件 + 最终合并」方案（PagePdfWriter）：
          1. 每页 OCR 完成后保存为独立小 PDF（page_XXXX.pdf）
          2. 全部处理完成后用 insert_pdf 一次性合并

        相比旧 IncrementalPdfWriter 的优势（日志实测）：
          - 旧方案每 5 页 flush 随总页数增长恶化，末尾页单次 flush 15.5s
          - 新方案每页写入 ~13ms 恒定，合并 272 页仅 0.6s
          - 断点续传检查更简单：扫描 pages_dir 文件名即可

        支持断点续传：检测 pages_dir 已有页数，从下一页继续处理。
        """
        from ..utils.page_pdf_writer import PagePdfWriter

        writer = PagePdfWriter(path, out_dir, render_dpi=dpi)
        # 断点续传：已有页数作为起始页（跳过已处理页）
        start_page = writer.existing_pages()
        # 关键：_page_count 从已有页数开始，避免 append_page 覆盖阶段1的单页文件
        # 不这样做的话，断点续传的 page_no 会从 1 开始，覆盖阶段1的 page_0001.pdf
        writer._page_count = start_page

        # 若最终输出文件已存在且 pages_dir 为空，说明上次已合并完成
        # 直接返回已有文件，避免重复处理
        if start_page == 0 and os.path.isfile(writer.out_path):
            logger.info("输出文件已存在，跳过处理: %s", writer.out_path)
            # 从已有PDF读取实际页数，避免返回0页导致前端显示异常
            page_count = 0
            try:
                import fitz
                doc = fitz.open(writer.out_path)
                page_count = doc.page_count
                doc.close()
            except Exception as e:
                logger.warning("读取已有PDF页数失败: %s", e)
            from .document.base import PageResult
            pages = [PageResult(page_no=i + 1) for i in range(page_count)]
            return DocumentResult(source_path=path, pages=pages), writer.out_path

        def page_cb(page_result):
            # 每页处理完立即保存为独立 PDF 文件
            writer.append_page(page_result)
            if progress_cb:
                progress_cb(
                    writer.page_count, -1,
                    f"已写入 {writer.page_count} 页",
                )

        try:
            result = handler.process(
                path, progress_cb=progress_cb, page_cb=page_cb,
                slot=slot, start_page=start_page,
            )
            if progress_cb:
                progress_cb(-1, -1, f"正在合并 {writer.page_count} 页...")
            import time as _t
            t_merge_start = _t.time()
            out_path = writer.close()
            t_merge = _t.time() - t_merge_start
            # 合并耗时写入进度日志，便于诊断
            try:
                from ..utils.progress_log import log_progress
                log_progress(
                    f"PDF 合并完成: {writer.page_count} 页 | 耗时 {t_merge:.1f}s"
                )
            except Exception:
                pass
        except Exception:
            # 异常时保留已处理的单页文件（不合并，下次断点续传）
            # 只记录已处理页数，便于日志诊断
            logger.warning(
                "处理中断，已保存 %d 页单页文件到 %s",
                writer.page_count, writer.pages_dir,
            )
            raise
        return result, out_path

    def _process_pdf_parallel(
        self, handler, path: str, progress_cb, out_dir: str, dpi: int,
        slots: List[int],
    ) -> Tuple[DocumentResult, str]:
        """PDF 多进程并行处理：按页切分，每个 slot 处理一个页范围。

        流程:
          1. 获取 PDF 总页数，按 slots 数量切分页范围（尽量均匀）
          2. 用 ThreadPoolExecutor 并行处理各页范围，每个线程用一个 slot
          3. 每页处理完后通过 page_cb 写入共享 PagePdfWriter（显式 page_no 避免竞争）
          4. 全部完成后合并单页文件为最终 PDF

        线程安全保证:
          - PagePdfWriter.append_page 接收显式 page_no，不依赖 _page_count 递增
          - 每个线程处理不同的页范围，page_no 不会冲突
          - PagePdfWriter._ensure_src_doc 内部 fitz.open 非线程安全，
            但只在首次调用时打开，之后复用句柄；多线程同时 append_page
            会各自调用 _ensure_src_doc，可能重复打开 → 用锁保护
          - OCR 实例池每个 slot 独立，互不干扰
        """
        import fitz
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from ..utils.page_pdf_writer import PagePdfWriter
        from ..utils.progress_log import log_progress

        # 获取总页数
        doc = fitz.open(path)
        total_pages = doc.page_count
        doc.close()

        n = len(slots)
        # 切分页范围：均匀分配，余数分给前几个段
        # 例如 10页3进程 → [0,4), [4,7), [7,10)
        ranges = []
        base = total_pages // n
        remainder = total_pages % n
        start = 0
        for i in range(n):
            count = base + (1 if i < remainder else 0)
            if count > 0:
                ranges.append((start, start + count))
                start += count

        if not ranges:
            # 空PDF，直接返回
            writer = PagePdfWriter(path, out_dir, render_dpi=dpi)
            return DocumentResult(source_path=path, pages=[]), writer.close()

        log_progress(
            f"PDF 并行处理: {os.path.basename(path)}, {total_pages} 页, "
            f"{n} 进程, 页范围={ranges}"
        )
        if progress_cb:
            progress_cb(0, total_pages, f"并行处理 {total_pages} 页（{n} 进程）")

        writer = PagePdfWriter(path, out_dir, render_dpi=dpi)
        # 写入锁：保护 PagePdfWriter 的源文档句柄和文件写入
        write_lock = threading.Lock()
        # 进度计数器（线程间共享）
        progress_counter = {"done": 0}
        counter_lock = threading.Lock()

        def process_range(slot_idx: int, slot: int, page_start: int, page_end: int):
            """处理一个页范围 [page_start, page_end)。"""
            def page_cb(page_result):
                # 用源PDF页号作为输出页号（1基），保证合并顺序正确
                out_page_no = page_result.page_no
                with write_lock:
                    writer.append_page(page_result, page_no=out_page_no)
                # 更新进度
                with counter_lock:
                    progress_counter["done"] += 1
                    done = progress_counter["done"]
                if progress_cb:
                    progress_cb(done, total_pages,
                                f"并行处理 {done}/{total_pages} 页（进程 {slot_idx+1}/{n}）")

            # 调用 handler.process 处理指定页范围
            handler.process(
                path, progress_cb=None, page_cb=page_cb,
                slot=slot, start_page=page_start, end_page=page_end,
            )

        # 并行执行
        errors = []
        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = []
            for idx, (page_start, page_end) in enumerate(ranges):
                slot = slots[idx]
                fut = pool.submit(
                    process_range, idx, slot, page_start, page_end
                )
                futures.append(fut)
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    logger.error("并行处理页范围失败: %s", e)
                    errors.append(str(e))

        if errors:
            # 部分失败：仍尝试合并已完成的页
            logger.warning("并行处理有 %d 个错误，尝试合并已完成的页", len(errors))

        # 合并所有单页文件
        if progress_cb:
            progress_cb(-1, -1, f"正在合并 {writer.page_count} 页...")
        import time as _t
        t_merge_start = _t.time()
        out_path = writer.close()
        t_merge = _t.time() - t_merge_start
        log_progress(
            f"PDF 并行合并完成: {writer.page_count} 页 | 耗时 {t_merge:.1f}s"
        )

        # 从输出PDF读取页数构造结果
        page_count = 0
        try:
            doc = fitz.open(out_path)
            page_count = doc.page_count
            doc.close()
        except Exception:
            pass
        from .document.base import PageResult
        pages = [PageResult(page_no=i + 1) for i in range(page_count)]
        return DocumentResult(source_path=path, pages=pages), out_path

    def process_files(
        self,
        paths: List[str],
        progress_cb: Optional[ProgressCb] = None,
    ) -> List[Tuple[str, str, Optional[str]]]:
        """批量处理，返回 [(源文件, 输出路径, 错误信息)]。

        进度上报策略：
          - 进度条按文件级更新（文件 i 完成 → i/total）
          - 页级消息（如"识别 PDF 第 3/10 页"）透传到日志，让用户看到处理过程
        """
        results = []
        total = len(paths)
        for i, p in enumerate(paths):
            file_name = os.path.basename(p)
            # 用默认参数固定 i，避免闭包延迟绑定
            def file_progress(cur, page_total, msg, _i=i, _total=total, _name=file_name):
                if progress_cb:
                    if page_total > 0:
                        # 页级进度：折算到整体进度条
                        overall = (_i + cur / page_total) / _total * 100 if _total > 0 else 0
                        progress_cb(int(overall), 100, f"[{_name}] {msg}")
                    else:
                        # 中间状态消息，只显示不更新进度条
                        progress_cb(-1, -1, f"[{_name}] {msg}")

            try:
                if progress_cb:
                    progress_cb(i, total, f"开始处理 {i + 1}/{total}: {file_name}")
                _, out_path = self.process_file(p, progress_cb=file_progress)
                results.append((p, out_path, None))
            except Exception as e:
                logger.exception("处理失败: %s", p)
                results.append((p, "", str(e)))
        if progress_cb:
            progress_cb(total, total, "全部完成")
        return results

    def _get_handler(self, path: str) -> Optional[BaseHandler]:
        for h in self._handlers:
            if h.can_handle(path):
                return h
        return None
