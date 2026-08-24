import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QProgressBar, QFileDialog, QGroupBox, QMessageBox,
    QLineEdit, QSpinBox, QFormLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from .convert_engine import ConvertEngine, CancelledException
from ..utils.pdf_utils import get_pdf_info


class ConvertWorker(QThread):
    """转换后台工作线程"""
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, input_path, output_path, dpi):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.dpi = dpi
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        try:
            engine = ConvertEngine(
                progress_callback=lambda c, t, m: self.progress.emit(c, t, m),
                cancel_check=lambda: self._cancel_requested
            )
            result = engine.convert(self.input_path, self.output_path, self.dpi)
            self.finished.emit(result)
        except CancelledException:
            self.error.emit("操作已取消")
        except Exception as e:
            self.error.emit(str(e))


class ConvertPage(QWidget):
    """PDF 转 DOCX 页面"""

    def __init__(self):
        super().__init__()
        self._current_pdf_path = ""
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # 标题
        title = QLabel("PDF 转 DOCX")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 5px 0;")
        layout.addWidget(title)

        # 说明文字
        info = QLabel(
            "只支持文本还原（文本型 PDF 提取文字，扫描型 PDF 以图片嵌入 Word）"
        )
        info.setStyleSheet("color: #666; margin-bottom: 10px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        # 文件选择
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("PDF 文件:"))

        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("选择要转换的 PDF 文件...")
        self.file_path.setReadOnly(True)
        file_layout.addWidget(self.file_path)

        self.btn_browse = QPushButton("浏览...")
        self.btn_browse.clicked.connect(self._browse_file)
        file_layout.addWidget(self.btn_browse)
        layout.addLayout(file_layout)

        # 文件信息
        self.file_info_label = QLabel("")
        layout.addWidget(self.file_info_label)

        # 转换设置
        settings_group = QGroupBox("转换设置")
        settings_form = QFormLayout()

        dpi_layout = QHBoxLayout()
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(100, 300)
        self.dpi_spin.setValue(150)
        self.dpi_spin.setSuffix(" DPI")
        dpi_layout.addWidget(self.dpi_spin)
        dpi_layout.addWidget(QLabel("（扫描型 PDF 的渲染分辨率，越高越清晰但文件越大）"))
        settings_form.addRow("图片质量:", dpi_layout)

        settings_group.setLayout(settings_form)
        layout.addWidget(settings_group)

        # 输出设置
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("输出文件:"))

        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText("选择输出的 DOCX 文件路径...")
        output_layout.addWidget(self.output_path)

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
        self.btn_convert = QPushButton("开始转换")
        self.btn_convert.setEnabled(False)
        self.btn_convert.clicked.connect(self._start_convert)
        self.btn_convert.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; "
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
        action_layout.addWidget(self.btn_convert)
        action_layout.addWidget(self.btn_cancel)
        layout.addLayout(action_layout)

        self.setLayout(layout)

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 PDF 文件", "", "PDF 文件 (*.pdf)"
        )
        if path:
            self.file_path.setText(path)
            self._current_pdf_path = path
            try:
                info = get_pdf_info(path)
                self.file_info_label.setText(
                    f"文件名: {info['filename']}  |  总页数: {info['page_count']} 页"
                )
                # 自动设置输出路径
                dir_name = os.path.dirname(path)
                base_name = os.path.splitext(os.path.basename(path))[0]
                self.output_path.setText(os.path.join(dir_name, f"_{base_name}.docx"))
                self._update_ui_state()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法打开文件: {e}")

    def _browse_output(self):
        # 用已选 PDF 的文件名生成默认 docx 文件名
        if self._current_pdf_path:
            default_dir = os.path.dirname(self._current_pdf_path)
            base_name = os.path.splitext(os.path.basename(self._current_pdf_path))[0]
            default_path = os.path.join(default_dir, f"_{base_name}.docx")
        else:
            default_path = ""
        path, _ = QFileDialog.getSaveFileName(
            self, "保存转换后的 DOCX", default_path, "Word 文件 (*.docx)"
        )
        if path:
            if not path.lower().endswith(".docx"):
                path += ".docx"
            self.output_path.setText(path)
            self._update_ui_state()

    def _update_ui_state(self):
        enabled = bool(self._current_pdf_path) and bool(self.output_path.text())
        self.btn_convert.setEnabled(enabled)

    def _start_convert(self):
        if not self._current_pdf_path:
            QMessageBox.warning(self, "提示", "请选择要转换的 PDF 文件")
            return
        if not self.output_path.text():
            QMessageBox.warning(self, "提示", "请选择输出文件路径")
            return

        self._set_ui_enabled(False)
        self.btn_cancel.setVisible(True)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("正在准备...")

        self.worker = ConvertWorker(
            self._current_pdf_path,
            self.output_path.text(),
            self.dpi_spin.value()
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

    def _on_finished(self, result_path: str):
        self._set_ui_enabled(True)
        self.btn_cancel.setVisible(False)
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.status_label.setText("转换完成！")

        reply = QMessageBox.information(
            self, "转换完成",
            f"文件已保存到:\n{result_path}\n\n是否打开输出目录？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            os.startfile(os.path.dirname(result_path))

    def _on_error(self, error_msg: str):
        self._set_ui_enabled(True)
        self.btn_cancel.setVisible(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("转换失败")
        if "操作已取消" not in error_msg:
            QMessageBox.critical(self, "转换失败", error_msg)

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
        self.btn_browse.setEnabled(enabled)
        self.btn_browse_output.setEnabled(enabled)
        self.btn_convert.setEnabled(enabled)
        self.btn_cancel.setEnabled(enabled)
        self.output_path.setEnabled(enabled)
        self.dpi_spin.setEnabled(enabled)