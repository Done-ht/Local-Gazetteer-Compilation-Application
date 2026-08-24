import os
import time
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget,
    QListWidgetItem, QLabel, QProgressBar, QFileDialog,
    QRadioButton, QButtonGroup, QGroupBox, QMessageBox, QLineEdit,
    QAbstractItemView, QSplitter, QShortcut, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QKeySequence

from .append_engine import AppendEngine
from ..convert.convert_engine import CancelledException
from ..compose.compose_engine import SUPPORTED_IMAGE_EXTS
from ..utils.natural_sort import natural_sort


class AppendWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, files, output_path):
        super().__init__()
        self.files = files
        self.output_path = output_path
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        try:
            engine = AppendEngine(
                progress_callback=lambda c, t, m: self.progress.emit(c, t, m),
                cancel_check=lambda: self._cancel_requested
            )
            result = engine.append(self.files, self.output_path)
            self.finished.emit(result)
        except CancelledException:
            self.finished.emit("操作已取消")
        except Exception as e:
            self.error.emit(str(e))


class AppendDropList(QListWidget):
    delete_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_requested.emit()
        else:
            super().keyPressEvent(event)

    def get_ordered_files(self) -> list[str]:
        files = []
        for i in range(self.count()):
            files.append(self.item(i).data(Qt.UserRole))
        return files


class AppendPage(QWidget):
    """图片 + PDF 拼接页面"""

    FILE_FILTER = "所有支持文件 (*.pdf *.jpg *.jpeg *.png *.bmp *.gif *.tiff *.tif *.webp);;PDF 文件 (*.pdf);;图片文件 (*.jpg *.jpeg *.png *.bmp *.gif *.tiff *.tif *.webp)"

    def __init__(self):
        super().__init__()
        self._batches: list[dict] = []
        self._redo_stack: list[dict] = []
        self._next_batch_id = 0
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)

        title = QLabel("图片 + PDF 拼接")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 5px 0;")
        layout.addWidget(title)

        info = QLabel("支持混合添加 PDF 和图片，图片将转换为 PDF 页面后拼接")
        info.setStyleSheet("color: #666;")
        info.setWordWrap(True)
        layout.addWidget(info)

        # 排序模式
        mode_group = QGroupBox("排序模式")
        mode_layout = QVBoxLayout()
        self.mode_radio_group = QButtonGroup()
        self.rb_specified = QRadioButton("指定顺序（手动拖拽排列）")
        self.rb_folder = QRadioButton("文件夹内顺序（自然排序）")
        self.rb_specified.setChecked(True)
        self.mode_radio_group.addButton(self.rb_specified, 1)
        self.mode_radio_group.addButton(self.rb_folder, 2)
        mode_layout.addWidget(self.rb_specified)
        mode_layout.addWidget(self.rb_folder)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # 操作按钮
        btn_layout = QHBoxLayout()
        self.btn_add_files = QPushButton("添加文件")
        self.btn_add_folder = QPushButton("添加文件夹（含子目录）")
        self.btn_remove = QPushButton("删除选中")
        self.btn_clear = QPushButton("清空全部")
        self.btn_add_files.clicked.connect(self._add_files)
        self.btn_add_folder.clicked.connect(self._add_folder)
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_clear.clicked.connect(self._clear_list)
        btn_layout.addWidget(self.btn_add_files)
        btn_layout.addWidget(self.btn_add_folder)
        btn_layout.addWidget(self.btn_remove)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_clear)
        layout.addLayout(btn_layout)

        # 文件列表 + 操作历史
        splitter = QSplitter(Qt.Horizontal)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("文件列表（可拖拽排序）:"))
        self.file_list = AppendDropList()
        left_layout.addWidget(self.file_list)
        self.summary_label = QLabel("已选择 0 个文件")
        left_layout.addWidget(self.summary_label)
        splitter.addWidget(left_widget)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        hist_header = QLabel("操作历史（点击撤销）")
        hist_header.setStyleSheet("font-weight: bold;")
        right_layout.addWidget(hist_header)

        self._history_list = QListWidget()
        self._history_list.setAlternatingRowColors(True)
        self._history_list.itemDoubleClicked.connect(self._on_history_double_click)
        right_layout.addWidget(self._history_list)

        splitter.addWidget(right_widget)
        splitter.setSizes([550, 250])
        layout.addWidget(splitter)

        # 输出设置
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("输出文件:"))
        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText("将自动生成默认路径...")
        output_layout.addWidget(self.output_path)
        self.btn_browse_output = QPushButton("浏览...")
        self.btn_browse_output.clicked.connect(self._browse_output)
        output_layout.addWidget(self.btn_browse_output)
        layout.addLayout(output_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        action_layout = QHBoxLayout()
        self.btn_append = QPushButton("开始拼接")
        self.btn_append.setEnabled(False)
        self.btn_append.clicked.connect(self._start_append)
        self.btn_append.setStyleSheet(
            "QPushButton { background-color: #E91E63; color: white; "
            "padding: 8px 24px; font-size: 14px; font-weight: bold; }"
        )
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; "
            "padding: 8px 24px; font-size: 14px; font-weight: bold; }"
        )
        self.btn_cancel.setVisible(False)
        self.btn_cancel.clicked.connect(self._cancel_operation)
        action_layout.addStretch()
        action_layout.addWidget(self.btn_append)
        action_layout.addWidget(self.btn_cancel)
        layout.addLayout(action_layout)

        self.setLayout(layout)

        self.file_list.model().rowsMoved.connect(self._update_summary)
        self.file_list.delete_requested.connect(self._remove_selected)

        # 撤销/重做快捷键
        QShortcut(QKeySequence.Undo, self, self._undo_last_batch)
        QShortcut(QKeySequence("Ctrl+Y"), self, self._redo_last_batch)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self._redo_last_batch)
        QShortcut(QKeySequence.Redo, self, self._redo_last_batch)

    # ── 批量操作 ──────────────────────────────────

    def _add_batch(self, label, files, batch_type="add", removed_items=None):
        bid = self._next_batch_id
        self._next_batch_id += 1
        self._batches.append({
            "id": bid,
            "label": label,
            "files": list(files),
            "batch_type": batch_type,
            "removed_items": removed_items or []
        })
        self._rebuild_history_ui()

    def _undo_batch(self, batch_id):
        batch = next((b for b in self._batches if b["id"] == batch_id), None)
        if not batch:
            return
        if batch["batch_type"] == "add":
            batch_files = set(batch["files"])
            i = 0
            while i < self.file_list.count():
                if self.file_list.item(i).data(Qt.UserRole) in batch_files:
                    self.file_list.takeItem(i)
                else:
                    i += 1
        else:  # "delete"
            for item_data in batch.get("removed_items", []):
                item = QListWidgetItem(item_data["display_text"])
                item.setData(Qt.UserRole, item_data["file_path"])
                self.file_list.addItem(item)
        self._batches.remove(batch)
        self._redo_stack.append(batch)
        self._rebuild_history_ui()
        self._update_summary()
        if self.file_list.count() == 0:
            self.output_path.clear()

    def _redo_batch(self, batch_id):
        batch = next((b for b in self._redo_stack if b["id"] == batch_id), None)
        if not batch:
            return
        if batch["batch_type"] == "add":
            for file_path in batch["files"]:
                label = os.path.basename(file_path)
                ext = os.path.splitext(file_path)[1].lower()
                if ext == ".pdf":
                    label += "  [PDF]"
                else:
                    label += "  [图片]"
                item = QListWidgetItem(label)
                item.setData(Qt.UserRole, file_path)
                self.file_list.addItem(item)
        else:  # "delete"
            removed_paths = set(batch["files"])
            i = 0
            while i < self.file_list.count():
                if self.file_list.item(i).data(Qt.UserRole) in removed_paths:
                    self.file_list.takeItem(i)
                else:
                    i += 1
        self._redo_stack.remove(batch)
        self._batches.append(batch)
        self._rebuild_history_ui()
        self._update_summary()
        if self.file_list.count() == 0:
            self.output_path.clear()

    def _undo_last_batch(self):
        if self._batches:
            self._undo_batch(self._batches[-1]["id"])

    def _redo_last_batch(self):
        if self._redo_stack:
            self._redo_batch(self._redo_stack[-1]["id"])

    def _rebuild_history_ui(self):
        self._history_list.clear()
        for batch in reversed(self._batches):
            item = QListWidgetItem()
            item.setData(Qt.UserRole, batch["id"])

            frame = QFrame()
            frame_layout = QHBoxLayout(frame)
            frame_layout.setContentsMargins(6, 4, 6, 4)
            frame_layout.setSpacing(6)

            label = QLabel(batch["label"])
            label.setWordWrap(True)
            label.setStyleSheet("color: #333; font-size: 12px;")

            count_label = QLabel(f"({len(batch['files'])} 个)")
            count_label.setStyleSheet("color: #999; font-size: 11px;")

            undo_btn = QPushButton("撤销")
            undo_btn.setFixedSize(50, 24)
            undo_btn.setStyleSheet(
                "QPushButton { color: #d32f2f; border: 1px solid #d32f2f; "
                "border-radius: 3px; font-size: 11px; padding: 0; }"
                "QPushButton:hover { background-color: #ffebee; }"
            )
            bid = batch["id"]
            undo_btn.clicked.connect(lambda checked, bid=bid: self._undo_batch(bid))

            # 删除批次文字为红色
            if batch.get("batch_type") == "delete":
                label.setStyleSheet("color: #d32f2f; font-size: 12px;")

            frame_layout.addWidget(label, 1)
            frame_layout.addWidget(count_label)
            frame_layout.addWidget(undo_btn)

            frame.setStyleSheet(
                "QFrame { background-color: #fafafa; border: 1px solid #eee; "
                "border-radius: 3px; }"
            )

            item.setSizeHint(frame.sizeHint())
            self._history_list.addItem(item)
            self._history_list.setItemWidget(item, frame)

    def _on_history_double_click(self, item):
        batch_id = item.data(Qt.UserRole)
        redo_batch = next((b for b in self._redo_stack if b["id"] == batch_id), None)
        if redo_batch:
            self._redo_batch(batch_id)
        else:
            self._undo_batch(batch_id)

    # ── 文件操作 ──────────────────────────────────

    def _is_supported(self, path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext == ".pdf" or ext in SUPPORTED_IMAGE_EXTS

    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择文件", "", self.FILE_FILTER
        )
        if not files:
            return
        added = self._add_files_to_list(files)
        if added:
            self._add_batch(f"文件 × {len(added)}", added)

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹（含子目录）")
        if not folder:
            return
        files = []
        for dirpath, _, filenames in os.walk(folder):
            for f in filenames:
                if self._is_supported(os.path.join(dirpath, f)):
                    files.append(os.path.join(dirpath, f))
        if not files:
            QMessageBox.information(self, "提示", "文件夹中未找到支持的 PDF 或图片文件")
            return
        natural_sort(files)
        folder_name = os.path.basename(folder) or folder
        added = self._add_files_to_list(files)
        if added:
            self._add_batch(f"文件夹: {folder_name}", added)

    def _add_files_to_list(self, files: list[str]) -> list[str]:
        added = []
        for file_path in files:
            if not self._is_supported(file_path):
                continue
            existing = False
            for i in range(self.file_list.count()):
                if self.file_list.item(i).data(Qt.UserRole) == file_path:
                    existing = True
                    break
            if existing:
                continue
            label = os.path.basename(file_path)
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".pdf":
                label += "  [PDF]"
            else:
                label += "  [图片]"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, file_path)
            self.file_list.addItem(item)
            added.append(file_path)
        if added:
            self._update_summary()
            self._auto_set_output_path()
        return added

    def _remove_selected(self):
        selected = self.file_list.selectedItems()
        if not selected:
            return
        removed_items = []
        for item in selected:
            removed_items.append({
                "file_path": item.data(Qt.UserRole),
                "display_text": item.text()
            })
        removed_paths = {item.data(Qt.UserRole) for item in selected}
        for item in selected:
            self.file_list.takeItem(self.file_list.row(item))
        for batch in self._batches:
            batch["files"] = [f for f in batch["files"] if f not in removed_paths]
        self._batches = [b for b in self._batches if b["files"]]
        if removed_items:
            bid = self._next_batch_id
            self._next_batch_id += 1
            self._batches.append({
                "id": bid,
                "label": f"删除 × {len(removed_items)}",
                "files": [it["file_path"] for it in removed_items],
                "batch_type": "delete",
                "removed_items": removed_items
            })
        self._rebuild_history_ui()
        if self.file_list.count() == 0:
            self.output_path.clear()
        self._update_summary()

    def _clear_list(self):
        if self.file_list.count() == 0:
            return
        reply = QMessageBox.question(
            self, "确认清空", "确定要清空所有文件吗？\n（操作历史也将被清空）",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        self.file_list.clear()
        self._batches.clear()
        self._redo_stack.clear()
        self._rebuild_history_ui()
        self.output_path.clear()
        self._update_summary()

    # ── 输出路径 ──────────────────────────────────

    def _auto_set_output_path(self):
        if self.output_path.text():
            return
        files = self.file_list.get_ordered_files()
        if not files:
            return
        first_dir = os.path.dirname(files[0])
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_path.setText(os.path.join(first_dir, f"拼接结果_{timestamp}.pdf"))

    def _browse_output(self):
        files = self.file_list.get_ordered_files()
        default_dir = os.path.dirname(files[0]) if files else ""
        path, _ = QFileDialog.getSaveFileName(
            self, "保存拼接后的 PDF", default_dir, "PDF 文件 (*.pdf)"
        )
        if path:
            if not path.lower().endswith(".pdf"):
                path += ".pdf"
            self.output_path.setText(path)
            self._update_summary()

    # ── 汇总 / 执行 ──────────────────────────────

    def _update_summary(self):
        count = self.file_list.count()
        self.summary_label.setText(f"已选择 {count} 个文件")
        self.btn_append.setEnabled(count > 0 and bool(self.output_path.text()))

    def _start_append(self):
        files = self.file_list.get_ordered_files()
        output_path = self.output_path.text()
        if not files:
            QMessageBox.warning(self, "提示", "请添加文件")
            return
        if not output_path:
            QMessageBox.warning(self, "提示", "请选择输出文件保存路径")
            return

        self._set_ui_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("正在准备...")

        self.worker = AppendWorker(files, output_path)
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
        if result_path == "操作已取消":
            self.progress_bar.setVisible(False)
            self.status_label.setText("操作已取消")
            return
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.status_label.setText("拼接完成！")
        reply = QMessageBox.information(
            self, "拼接完成", f"文件已保存到:\n{result_path}\n\n是否打开输出目录？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            os.startfile(os.path.dirname(result_path))

    def _on_error(self, error_msg: str):
        self._set_ui_enabled(True)
        self.progress_bar.setVisible(False)
        if error_msg == "操作已取消":
            self.status_label.setText("操作已取消")
            return
        self.status_label.setText("拼接失败")
        QMessageBox.critical(self, "拼接失败", error_msg)

    def _cancel_operation(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.cancel()
            self.progress_bar.setVisible(False)
            self.status_label.setText("操作已取消")
            self._set_ui_enabled(True)
            output_path = self.output_path.text()
            if output_path and os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass

    def _set_ui_enabled(self, enabled: bool):
        self.btn_add_files.setEnabled(enabled)
        self.btn_add_folder.setEnabled(enabled)
        self.btn_remove.setEnabled(enabled)
        self.btn_clear.setEnabled(enabled)
        self.btn_append.setEnabled(enabled)
        self.btn_cancel.setVisible(not enabled)
        self.btn_browse_output.setEnabled(enabled)
        self.output_path.setEnabled(enabled)
        self.file_list.setEnabled(enabled)
        self._history_list.setEnabled(enabled)
        self.rb_specified.setEnabled(enabled)
        self.rb_folder.setEnabled(enabled)

    def keyPressEvent(self, event):
        if event.modifiers() == Qt.ControlModifier:
            if event.key() == Qt.Key_Z:
                self._undo_last_batch()
                return
            elif event.key() == Qt.Key_Y:
                self._redo_last_batch()
                return
        super().keyPressEvent(event)