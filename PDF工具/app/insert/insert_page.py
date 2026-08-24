import os
import time
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget,
    QListWidgetItem, QLabel, QProgressBar, QFileDialog,
    QGroupBox, QMessageBox, QLineEdit, QSpinBox, QFormLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from .insert_engine import InsertEngine, CancelledException
from ..compose.compose_engine import SUPPORTED_IMAGE_EXTS
from ..utils.pdf_utils import get_pdf_info


class InsertWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, base_pdf, insert_files, insert_page, output_path):
        super().__init__()
        self.base_pdf = base_pdf
        self.insert_files = insert_files
        self.insert_page = insert_page
        self.output_path = output_path
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        try:
            engine = InsertEngine(
                progress_callback=lambda c, t, m: self.progress.emit(c, t, m),
                cancel_check=lambda: self._cancel_requested
            )
            result = engine.insert(
                self.base_pdf, self.insert_files, self.insert_page, self.output_path
            )
            self.finished.emit(result)
        except CancelledException:
            self.error.emit("操作已取消")
        except Exception as e:
            self.error.emit(str(e))


class InsertPage(QWidget):
    """向 PDF 插入内容页面"""

    FILE_FILTER = "所有支持文件 (*.pdf *.jpg *.jpeg *.png *.bmp *.gif *.tiff *.tif *.webp);;PDF 文件 (*.pdf);;图片文件 (*.jpg *.jpeg *.png *.bmp *.gif *.tiff *.tif *.webp)"

    def __init__(self):
        super().__init__()
        self._base_pdf_path = ""
        self._base_page_count = 0
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # 标题
        title = QLabel("向 PDF 插入内容")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 5px 0;")
        layout.addWidget(title)

        info = QLabel("向 PDF 指定页面位置插入图片或 PDF 文件，原内容自动后移")
        info.setStyleSheet("color: #666;")
        info.setWordWrap(True)
        layout.addWidget(info)

        # ── 基础 PDF ──
        base_group = QGroupBox("基础 PDF 文档")
        base_layout = QVBoxLayout()

        file_row = QHBoxLayout()
        self.base_path = QLineEdit()
        self.base_path.setPlaceholderText("选择要插入内容的 PDF 文件...")
        self.base_path.setReadOnly(True)
        file_row.addWidget(self.base_path)
        self.btn_browse_base = QPushButton("浏览...")
        self.btn_browse_base.clicked.connect(self._browse_base)
        file_row.addWidget(self.btn_browse_base)
        base_layout.addLayout(file_row)

        self.base_info = QLabel("")
        base_layout.addWidget(self.base_info)

        # 插入位置
        pos_row = QHBoxLayout()
        pos_row.addWidget(QLabel("插入到第"))
        self.page_spin = QSpinBox()
        self.page_spin.setRange(1, 1)
        self.page_spin.setValue(1)
        self.page_spin.setToolTip("1 = 插在最前面")
        pos_row.addWidget(self.page_spin)
        pos_row.addWidget(QLabel("页之前（1 = 最前面）"))
        pos_row.addStretch()
        base_layout.addLayout(pos_row)

        base_group.setLayout(base_layout)
        layout.addWidget(base_group)

        # ── 要插入的内容 ──
        insert_group = QGroupBox("要插入的内容")
        insert_layout = QVBoxLayout()

        btn_row = QHBoxLayout()
        self.btn_add_files = QPushButton("添加文件")
        self.btn_remove = QPushButton("删除选中")
        self.btn_clear = QPushButton("清空列表")
        self.btn_add_files.clicked.connect(self._add_files)
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_clear.clicked.connect(self._clear_list)
        btn_row.addWidget(self.btn_add_files)
        btn_row.addWidget(self.btn_remove)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_clear)
        insert_layout.addLayout(btn_row)

        self.insert_list = QListWidget()
        self.insert_list.setSelectionMode(QListWidget.ExtendedSelection)
        insert_layout.addWidget(self.insert_list)

        self.insert_summary = QLabel("已选择 0 个文件")
        insert_layout.addWidget(self.insert_summary)

        insert_group.setLayout(insert_layout)
        layout.addWidget(insert_group)

        # ── 输出设置 ──
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("输出文件:"))
        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText("将自动生成默认路径...")
        output_layout.addWidget(self.output_path)
        self.btn_browse_output = QPushButton("浏览...")
        self.btn_browse_output.clicked.connect(self._browse_output)
        output_layout.addWidget(self.btn_browse_output)
        layout.addLayout(output_layout)

        # output_path 变化时自动更新按钮状态（避免边界情况导致不同步）
        self.output_path.textChanged.connect(self._update_ui_state)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        # 开始按钮 + 取消按钮
        action_layout = QHBoxLayout()
        self.btn_insert = QPushButton("开始插入")
        self.btn_insert.setEnabled(False)
        self.btn_insert.clicked.connect(self._start_insert)
        self.btn_insert.setStyleSheet(
            "QPushButton { background-color: #FF5722; color: white; "
            "padding: 8px 24px; font-size: 14px; font-weight: bold; }"
        )

        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setVisible(False)
        self.btn_cancel.clicked.connect(self._cancel_operation)
        self.btn_cancel.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; "
            "padding: 8px 24px; font-size: 14px; font-weight: bold; }"
        )

        action_layout.addStretch()
        action_layout.addWidget(self.btn_insert)
        action_layout.addWidget(self.btn_cancel)
        layout.addLayout(action_layout)

        self.setLayout(layout)

    # ── 基础 PDF ──────────────────────────────────

    def _browse_base(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 PDF 文件", "", "PDF 文件 (*.pdf)"
        )
        if not path:
            return
        self.base_path.setText(path)
        self._base_pdf_path = path
        try:
            info = get_pdf_info(path)
            self._base_page_count = info["page_count"]
            self.base_info.setText(
                f"文件名: {info['filename']}  |  总页数: {info['page_count']} 页"
            )
            self.page_spin.setRange(1, self._base_page_count + 1)
            self.page_spin.setValue(1)
            self._auto_set_output_path()
            self._update_ui_state()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法打开文件: {e}")

    # ── 插入内容 ──────────────────────────────────

    def _is_supported(self, path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext == ".pdf" or ext in SUPPORTED_IMAGE_EXTS

    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择要插入的文件", "", self.FILE_FILTER
        )
        if not files:
            return
        for file_path in files:
            if not self._is_supported(file_path):
                continue
            existing = False
            for i in range(self.insert_list.count()):
                if self.insert_list.item(i).data(Qt.UserRole) == file_path:
                    existing = True
                    break
            if existing:
                continue
            label = os.path.basename(file_path)
            ext = os.path.splitext(file_path)[1].lower()
            label += "  [PDF]" if ext == ".pdf" else "  [图片]"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, file_path)
            self.insert_list.addItem(item)
        # 添加文件时也尝试设置默认输出路径（确保任何操作顺序下按钮都能启用）
        self._auto_set_output_path()
        self._update_ui_state()

    def _remove_selected(self):
        for item in self.insert_list.selectedItems():
            self.insert_list.takeItem(self.insert_list.row(item))
        self._update_ui_state()

    def _clear_list(self):
        self.insert_list.clear()
        self._update_ui_state()

    # ── 输出路径 ──────────────────────────────────

    def _auto_set_output_path(self):
        if not self._base_pdf_path or self.output_path.text():
            return
        dir_name = os.path.dirname(self._base_pdf_path)
        base_name = os.path.splitext(os.path.basename(self._base_pdf_path))[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_path.setText(os.path.join(dir_name, f"_{base_name}_插入结果_{timestamp}.pdf"))

    def _browse_output(self):
        if self._base_pdf_path:
            default_dir = os.path.dirname(self._base_pdf_path)
            base_name = os.path.splitext(os.path.basename(self._base_pdf_path))[0]
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            default_path = os.path.join(default_dir, f"_{base_name}_插入结果_{timestamp}.pdf")
        else:
            default_path = ""
        path, _ = QFileDialog.getSaveFileName(
            self, "保存插入后的 PDF", default_path, "PDF 文件 (*.pdf)"
        )
        if path:
            if not path.lower().endswith(".pdf"):
                path += ".pdf"
            self.output_path.setText(path)
            self._update_ui_state()

    # ── 状态 ──────────────────────────────────────

    def _update_ui_state(self):
        insert_count = self.insert_list.count()
        self.insert_summary.setText(f"已选择 {insert_count} 个文件")
        has_base = bool(self._base_pdf_path)
        has_insert = insert_count > 0
        has_output = bool(self.output_path.text())
        self.btn_insert.setEnabled(has_base and has_insert and has_output)

    # ── 执行 ──────────────────────────────────────

    def _start_insert(self):
        if not self._base_pdf_path:
            QMessageBox.warning(self, "提示", "请选择基础 PDF 文件")
            return
        if self.insert_list.count() == 0:
            QMessageBox.warning(self, "提示", "请添加要插入的内容")
            return

        output_path = self.output_path.text().strip()
        if not output_path:
            # 兜底：自动生成默认路径（而不是弹警告让用户手动选）
            dir_name = os.path.dirname(self._base_pdf_path)
            base_name = os.path.splitext(os.path.basename(self._base_pdf_path))[0]
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(dir_name, f"_{base_name}_插入结果_{timestamp}.pdf")
            self.output_path.setText(output_path)

        insert_files = []
        for i in range(self.insert_list.count()):
            insert_files.append(self.insert_list.item(i).data(Qt.UserRole))

        insert_page = self.page_spin.value() - 1  # 转为 0-based

        self._set_ui_enabled(False)
        self.btn_cancel.setVisible(True)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("正在准备...")

        self.worker = InsertWorker(
            self._base_pdf_path, insert_files, insert_page, output_path
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, current, total, message):
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)
        self.status_label.setText(message)

    def _on_finished(self, result_path: str):
        self._set_ui_enabled(True)
        self.btn_cancel.setVisible(False)
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.status_label.setText("插入完成！")
        reply = QMessageBox.information(
            self, "插入完成", f"文件已保存到:\n{result_path}\n\n是否打开输出目录？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            os.startfile(os.path.dirname(result_path))

    def _on_error(self, error_msg: str):
        self._set_ui_enabled(True)
        self.btn_cancel.setVisible(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("插入失败")
        if "操作已取消" not in error_msg:
            QMessageBox.critical(self, "插入失败", error_msg)

    def _cancel_operation(self):
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.progress_bar.setVisible(False)
            self.status_label.setText("操作已取消")
            self._set_ui_enabled(True)
            self.btn_cancel.setVisible(False)
            output_path = self.output_path.text()
            if output_path and os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except Exception:
                    pass

    def _set_ui_enabled(self, enabled: bool):
        self.btn_browse_base.setEnabled(enabled)
        self.btn_add_files.setEnabled(enabled)
        self.btn_remove.setEnabled(enabled)
        self.btn_clear.setEnabled(enabled)
        self.btn_browse_output.setEnabled(enabled)
        self.btn_insert.setEnabled(enabled)
        self.btn_cancel.setEnabled(enabled)
        self.base_path.setEnabled(enabled)
        self.output_path.setEnabled(enabled)
        self.insert_list.setEnabled(enabled)
        self.page_spin.setEnabled(enabled)