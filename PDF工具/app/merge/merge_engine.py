import fitz
import os
from ..utils.pdf_utils import is_valid_pdf, is_encrypted_pdf


class CancelledException(Exception):
    """操作被取消"""
    pass


class MergeEngine:
    """PDF 合并引擎"""

    def __init__(self, progress_callback=None, cancel_check=None):
        """
        progress_callback: function(current, total, message) -> None
        cancel_check: function() -> bool, 返回 True 表示需要取消
        """
        self._progress_callback = progress_callback
        self._cancel_check = cancel_check

    def _check_cancelled(self):
        if self._cancel_check and self._cancel_check():
            raise CancelledException("操作已被取消")

    def _report(self, current: int, total: int, message: str = ""):
        self._check_cancelled()
        if self._progress_callback:
            self._progress_callback(current, total, message)

    def merge(self, input_files: list[str], output_path: str, fast_mode: bool = True) -> str:
        """
        合并多个 PDF 文件

        Args:
            input_files: 按顺序排列的 PDF 文件路径列表
            output_path: 输出文件路径
            fast_mode: 快速模式（流式拼接 + 关闭压缩），大文件场景显著提速

        Returns:
            输出文件的绝对路径

        Raises:
            FileNotFoundError: 输入文件不存在
            ValueError: 文件无效或加密
        """
        if not input_files:
            raise ValueError("没有选择要合并的文件")

        self._report(0, len(input_files), "正在验证文件...")

        valid_files = []
        for f in input_files:
            self._check_cancelled()
            if not os.path.isfile(f):
                raise FileNotFoundError(f"文件不存在: {f}")
            if not is_valid_pdf(f):
                raise ValueError(f"无效的 PDF 文件: {f}")
            if is_encrypted_pdf(f):
                raise ValueError(f"文件已加密，无法处理: {f}")
            valid_files.append(f)

        total_files = len(valid_files)

        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self._report(0, total_files, "开始合并...")

        if fast_mode:
            return self._merge_fast(valid_files, output_path, total_files)
        else:
            return self._merge_standard(valid_files, output_path, total_files)

    def _merge_fast(self, valid_files: list[str], output_path: str, total_files: int) -> str:
        """
        快速合并：流式拼接 + 关闭压缩
        - 直接以第一个文件为基础文档，避免从空文档累积重写
        - 保存时 garbage=0, deflate=False，跳过垃圾回收和流压缩
        """
        # 以第一个文件为基础打开（避免从空文档构建 + 大量 insert_pdf 累积）
        output_doc = fitz.open(valid_files[0])
        try:
            # 追加剩余文件
            for i, file_path in enumerate(valid_files[1:], start=1):
                self._check_cancelled()
                filename = os.path.basename(file_path)
                self._report(i, total_files, f"正在处理: {filename}")

                src_doc = fitz.open(file_path)
                try:
                    output_doc.insert_pdf(src_doc)
                finally:
                    src_doc.close()

            self._report(total_files, total_files, "正在保存合并文件...")
            # 关闭 garbage 和 deflate，大幅减少保存耗时
            output_doc.save(output_path, garbage=0, deflate=False)
        finally:
            output_doc.close()

        abs_path = os.path.abspath(output_path)
        self._report(total_files, total_files, f"合并完成: {os.path.basename(abs_path)}")
        return abs_path

    def _merge_standard(self, valid_files: list[str], output_path: str, total_files: int) -> str:
        """标准合并：从空文档构建，保存时压缩 + 垃圾回收（文件更小）"""
        output_doc = fitz.open()
        try:
            for i, file_path in enumerate(valid_files):
                self._check_cancelled()
                filename = os.path.basename(file_path)
                self._report(i, total_files, f"正在处理: {filename}")

                src_doc = fitz.open(file_path)
                try:
                    output_doc.insert_pdf(src_doc)
                finally:
                    src_doc.close()

            self._report(total_files, total_files, "正在保存合并文件...")
            output_doc.save(output_path, garbage=4, deflate=True)
        finally:
            output_doc.close()

        abs_path = os.path.abspath(output_path)
        self._report(total_files, total_files, f"合并完成: {os.path.basename(abs_path)}")
        return abs_path