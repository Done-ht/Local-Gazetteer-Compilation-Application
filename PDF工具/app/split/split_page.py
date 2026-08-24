import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QProgressBar, QFileDialog, QRadioButton, QButtonGroup,
    QGroupBox, QMessageBox, QLineEdit, QListWidget, QListWidgetItem,
    QSpinBox, QFormLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIntValidator

from .split_engine import SplitEngine, CancelledException
from ..utils.pdf_utils import get_pdf_info, get_pdf_bookmarks


class SplitWorker(QThread):
    """拆分后台工作线程"""
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, input_path, output_dir, mode, **kwargs):
        super().__init__()
        self.input_path = input_path
        self.output_dir = output_dir
        self.mode = mode
        self.kwargs = kwargs
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        try:
            engine = SplitEngine(
                progress_callback=lambda c, t, m: self.progress.emit(c, t, m),
                cancel_check=lambda: self._cancel_requested
            )
            result = engine.split(
                self.input_path, self.output_dir, self.mode, **self.kwargs
            )
            self.finished.emit(result)
        except CancelledException:
            self.error.emit("操作已取消")
        except Exception as e:
            self.error.emit(str(e))


class SplitPage(QWidget):
    """PDF 拆分页面"""

    def __init__(self):
        super().__init__()
        self._current_pdf_path = ""
        self._current_page_count = 0
        self._current_bookmarks = []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # 标题
        title = QLabel("PDF 拆分")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 5px 0;")
        layout.addWidget(title)

        # 文件选择
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("PDF 文件:"))

        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("选择要拆分的 PDF 文件...")
        self.file_path.setReadOnly(True)
        file_layout.addWidget(self.file_path)

        self.btn_browse = QPushButton("浏览...")
        self.btn_browse.clicked.connect(self._browse_file)
        file_layout.addWidget(self.btn_browse)
        layout.addLayout(file_layout)

        # 文件信息
        self.file_info_label = QLabel("")
        layout.addWidget(self.file_info_label)

        # 拆分模式
        mode_group = QGroupBox("拆分模式")
        mode_layout = QVBoxLayout()

        self.mode_radio_group = QButtonGroup()
        self.rb_single = QRadioButton("单页拆分（每页一个 PDF）")
        self.rb_range = QRadioButton("按范围拆分")
        self.rb_extract = QRadioButton("提取指定页合并为一个 PDF")
        self.rb_single.setChecked(True)

        self.mode_radio_group.addButton(self.rb_single, 1)
        self.mode_radio_group.addButton(self.rb_range, 2)
        self.mode_radio_group.addButton(self.rb_extract, 4)

        mode_layout.addWidget(self.rb_single)
        mode_layout.addWidget(self.rb_range)
        mode_layout.addWidget(self.rb_extract)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # 范围参数
        self._range_group = QGroupBox("范围设置")
        range_form = QFormLayout()

        self.range_input = QLineEdit()
        self.range_input.setPlaceholderText("例如: 1-3 5 8-10 或 1-3,5,8-10")
        self.range_input.setEnabled(False)
        range_form.addRow("页码范围:", self.range_input)
        self._range_group.setLayout(range_form)
        self._range_group.setEnabled(False)
        layout.addWidget(self._range_group)

        # 提取指定页参数
        self._extract_group = QGroupBox("提取设置")
        extract_form = QFormLayout()

        extract_input_layout = QHBoxLayout()
        self.extract_page_input = QLineEdit()
        self.extract_page_input.setPlaceholderText("例如: 1 3 5 7 或 1,3,5,7")
        self.extract_page_input.setEnabled(False)
        extract_input_layout.addWidget(self.extract_page_input)

        extract_form.addRow("提取页码:", extract_input_layout)
        self._extract_group.setLayout(extract_form)
        self._extract_group.setEnabled(False)
        layout.addWidget(self._extract_group)

        # 输出目录
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("输出目录:"))

        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText("选择输出目录...")
        output_layout.addWidget(self.output_dir)

        self.btn_browse_output = QPushButton("浏览...")
        self.btn_browse_output.clicked.connect(self._browse_output)
        output_layout.addWidget(self.btn_browse_output)
        layout.addLayout(output_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        # 开始按钮 + 取消按钮
        action_layout = QHBoxLayout()
        self.btn_split = QPushButton("开始拆分")
        self.btn_split.setEnabled(False)
        self.btn_split.clicked.connect(self._start_split)
        self.btn_split.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
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
        action_layout.addWidget(self.btn_split)
        action_layout.addWidget(self.btn_cancel)
        layout.addLayout(action_layout)

        self.setLayout(layout)

        # 连接信号
        self.mode_radio_group.buttonClicked.connect(self._on_mode_changed)

    def _on_mode_changed(self):
        self.range_input.setEnabled(self.rb_range.isChecked())
        self.extract_page_input.setEnabled(self.rb_extract.isChecked())
        self._range_group.setEnabled(self.rb_range.isChecked())
        self._extract_group.setEnabled(self.rb_extract.isChecked())

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 PDF 文件", "", "PDF 文件 (*.pdf)"
        )
        if path:
            self.file_path.setText(path)
            self._current_pdf_path = path
            try:
                info = get_pdf_info(path)
                self._current_page_count = info["page_count"]
                self._current_bookmarks = get_pdf_bookmarks(path)
                top_count = len([b for b in self._current_bookmarks if b["level"] == 1])
                bm_info = f"，{top_count} 个顶层书签" if top_count > 0 else "，无书签"
                self.file_info_label.setText(
                    f"文件名: {info['filename']}  |  总页数: {info['page_count']}{bm_info}"
                )

                # 每次选文档都更新默认输出目录（随文档变化）
                base_name = os.path.splitext(os.path.basename(path))[0]
                parent_dir = os.path.dirname(path)
                default_dir = os.path.join(parent_dir, f"_{base_name}_拆分结果")
                self.output_dir.setText(default_dir)

                self._update_ui_state()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法打开文件: {e}")

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.output_dir.setText(path)
            self._update_ui_state()

    def _update_ui_state(self):
        enabled = bool(self._current_pdf_path) and bool(self.output_dir.text())
        self.btn_split.setEnabled(enabled)

    def _start_split(self):
        if not self._current_pdf_path:
            QMessageBox.warning(self, "提示", "请选择要拆分的 PDF 文件")
            return
        if not self.output_dir.text():
            QMessageBox.warning(self, "提示", "请选择输出目录")
            return

        mode = "single"
        kwargs = {}

        if self.rb_single.isChecked():
            mode = SplitEngine.SPLIT_MODE_SINGLE
        elif self.rb_range.isChecked():
            mode = SplitEngine.SPLIT_MODE_RANGE
            range_text = self.range_input.text().strip()
            if not range_text:
                QMessageBox.warning(self, "提示", "请输入页码范围")
                return
            kwargs["range_text"] = range_text
        elif self.rb_extract.isChecked():
            mode = SplitEngine.SPLIT_MODE_EXTRACT
            page_text = self.extract_page_input.text().strip()
            if not page_text:
                QMessageBox.warning(self, "提示", "请输入要提取的页码")
                return
            try:
                # 统一支持英文逗号、中文逗号、空格作为分隔符
                normalized = page_text.replace("，", ",").replace(" ", ",")
                pages = [int(p.strip()) - 1 for p in normalized.split(",") if p.strip()]
                kwargs["extract_pages"] = pages
            except ValueError:
                QMessageBox.warning(self, "提示", "页码格式无效，请用逗号分隔，例如: 1,3,5,7")
                return

        self._launch_worker(mode, kwargs)

    def _launch_worker(self, mode: str, kwargs: dict):
        """启动后台拆分 worker"""
        self._set_ui_enabled(False)
        self.btn_cancel.setVisible(True)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("正在准备...")

        self.worker = SplitWorker(
            self._current_pdf_path, self.output_dir.text(), mode, **kwargs
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, current: int, total: int, message: str):
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)
        self.status_label.setText(message)

    def _on_finished(self, result_files: list[str]):
        self._set_ui_enabled(True)
        self.btn_cancel.setVisible(False)
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.status_label.setText(f"拆分完成，共生成 {len(result_files)} 个文件")

        reply = QMessageBox.information(
            self, "拆分完成",
            f"共生成 {len(result_files)} 个文件\n\n是否打开输出目录？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            os.startfile(self.output_dir.text())

    def _on_error(self, error_msg: str):
        self._set_ui_enabled(True)
        self.btn_cancel.setVisible(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("拆分失败")
        if "操作已取消" not in error_msg:
            QMessageBox.critical(self, "拆分失败", error_msg)

    def _cancel_operation(self):
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.progress_bar.setVisible(False)
            self.status_label.setText("操作已取消")
            self._set_ui_enabled(True)
            self.btn_cancel.setVisible(False)
            output_path = self.output_dir.text()
            if output_path and os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except Exception:
                    pass

    def _set_ui_enabled(self, enabled: bool):
        self.btn_browse.setEnabled(enabled)
        self.btn_browse_output.setEnabled(enabled)
        self.btn_split.setEnabled(enabled)
        self.btn_cancel.setEnabled(enabled)
        self.range_input.setEnabled(enabled and self.rb_range.isChecked())
        self.extract_page_input.setEnabled(enabled and self.rb_extract.isChecked())
        self._range_group.setEnabled(enabled and self.rb_range.isChecked())
        self._extract_group.setEnabled(enabled and self.rb_extract.isChecked())
        self.rb_single.setEnabled(enabled)
        self.rb_range.setEnabled(enabled)
        self.rb_extract.setEnabled(enabled)