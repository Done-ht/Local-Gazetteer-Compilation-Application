import os
import time
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget,
    QListWidgetItem, QLabel, QProgressBar, QFileDialog,
    QRadioButton, QButtonGroup, QGroupBox, QMessageBox, QLineEdit,
    QAbstractItemView, QSplitter, QFrame, QShortcut, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QKeySequence

from .merge_engine import MergeEngine, CancelledException
from .sorter import sort_folder_order, sort_by_folder_order, _scan_pdfs_recursive
from ..utils.natural_sort import natural_sort
from ..utils.pdf_utils import get_pdf_info


class MergeWorker(QThread):
    """合并后台工作线程"""
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, files, output_path, fast_mode=True):
        super().__init__()
        self.files = files
        self.output_path = output_path
        self.fast_mode = fast_mode
        self._cancel_requested = False

    def cancel(self):
        """请求取消操作"""
        self._cancel_requested = True

    def run(self):
        try:
            engine = MergeEngine(
                progress_callback=lambda c, t, m: self.progress.emit(c, t, m),
                cancel_check=lambda: self._cancel_requested
            )
            result = engine.merge(self.files, self.output_path, fast_mode=self.fast_mode)
            self.finished.emit(result)
        except CancelledException:
            self.error.emit("操作已取消")
        except Exception as e:
            self.error.emit(str(e))


class DropListWidget(QListWidget):
    """支持拖拽排序和 Delete 键删除的列表控件"""
    delete_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            self.delete_requested.emit()
        else:
            super().keyPressEvent(event)

    def get_ordered_files(self) -> list[str]:
        """获取当前列表中文件的路径（按显示顺序）"""
        files = []
        for i in range(self.count()):
            item = self.item(i)
            files.append(item.data(Qt.UserRole))
        return files


class MergePage(QWidget):
    """PDF 合并页面"""

    def __init__(self):
        super().__init__()
        self._batches: list[dict] = []           # 批量操作记录 [{id, batch_type, label, files/items}]
        self._redo_stack: list[dict] = []        # 重做堆栈
        self._next_batch_id = 0
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # 标题
        title = QLabel("PDF 合并")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 5px 0;")
        layout.addWidget(title)

        # 排序模式选择
        mode_group = QGroupBox("排序模式")
        mode_layout = QVBoxLayout()

        self.mode_radio_group = QButtonGroup()
        self.rb_specified = QRadioButton("指定顺序（手动拖拽排列）")
        self.rb_folder = QRadioButton("文件夹内顺序（自然排序）")
        self.rb_by_folder = QRadioButton("按文件夹顺序")
        self.rb_specified.setChecked(True)

        self.mode_radio_group.addButton(self.rb_specified, 1)
        self.mode_radio_group.addButton(self.rb_folder, 2)
        self.mode_radio_group.addButton(self.rb_by_folder, 3)

        mode_layout.addWidget(self.rb_specified)
        mode_layout.addWidget(self.rb_folder)
        mode_layout.addWidget(self.rb_by_folder)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # 操作按钮区域
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

        # 文件列表 + 操作历史（左右分栏）
        splitter = QSplitter(Qt.Horizontal)

        # 左侧：文件列表
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("文件列表（可拖拽排序）:"))

        self.file_list = DropListWidget()
        left_layout.addWidget(self.file_list)

        self.summary_label = QLabel("已选择 0 个文件，共 0 页")
        left_layout.addWidget(self.summary_label)

        splitter.addWidget(left_widget)

        # 右侧：操作历史（可滚动）
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

        # output_path 变化时自动更新按钮状态（避免边界情况导致不同步）
        self.output_path.textChanged.connect(self._update_summary)

        # 快速模式选项
        self.cb_fast_mode = QCheckBox(
            "快速模式（流式拼接 + 不压缩，大文件显著提速，输出文件略大）"
        )
        self.cb_fast_mode.setChecked(True)
        self.cb_fast_mode.setToolTip(
            "勾选：以第一个文件为基础追加，关闭垃圾回收和流压缩\n"
            "不勾：标准模式，输出文件更小但大文件较慢"
        )
        layout.addWidget(self.cb_fast_mode)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        # 合并按钮 + 取消按钮
        action_layout = QHBoxLayout()
        self.btn_merge = QPushButton("开始合并")
        self.btn_merge.setEnabled(False)
        self.btn_merge.clicked.connect(self._start_merge)
        self.btn_merge.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
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
        action_layout.addWidget(self.btn_merge)
        action_layout.addWidget(self.btn_cancel)
        layout.addLayout(action_layout)

        self.setLayout(layout)

        # 连接信号
        self.file_list.model().rowsMoved.connect(self._update_summary)
        self.file_list.delete_requested.connect(self._remove_selected)
        self.rb_specified.toggled.connect(self._on_mode_changed)
        self.rb_folder.toggled.connect(self._on_mode_changed)
        self.rb_by_folder.toggled.connect(self._on_mode_changed)

        # 撤销/重做快捷键
        QShortcut(QKeySequence.Undo, self).activated.connect(self._undo_last_batch)
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self._undo_last_batch)
        QShortcut(QKeySequence.Redo, self).activated.connect(self._redo_last_batch)
        QShortcut(QKeySequence("Ctrl+Y"), self).activated.connect(self._redo_last_batch)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self).activated.connect(self._redo_last_batch)

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Undo):
            self._undo_last_batch()
        elif event.matches(QKeySequence.Redo):
            self._redo_last_batch()
        else:
            super().keyPressEvent(event)

    # ── 批量操作 / 撤销机制 ──────────────────────────

    def _add_batch(self, label: str, files: list[str]):
        """记录一次批量添加操作"""
        batch_id = self._next_batch_id
        self._next_batch_id += 1
        batch = {"id": batch_id, "batch_type": "add", "label": label, "files": list(files)}
        self._batches.append(batch)
        self._rebuild_history_ui()

    def _undo_batch(self, batch_id: int):
        """撤销指定批次"""
        batch = next((b for b in self._batches if b["id"] == batch_id), None)
        if not batch:
            return

        if batch.get("batch_type") == "delete":
            # 恢复被删除的文件
            for item_data in batch["items"]:
                item = QListWidgetItem(item_data["display_text"])
                item.setData(Qt.UserRole, item_data["file_path"])
                self.file_list.addItem(item)
        else:
            # 从列表中移除该批次所有文件
            batch_files = set(batch["files"])
            i = 0
            while i < self.file_list.count():
                item = self.file_list.item(i)
                if item.data(Qt.UserRole) in batch_files:
                    self.file_list.takeItem(i)
                else:
                    i += 1

        self._batches.remove(batch)
        self._redo_stack.append(batch)
        self._rebuild_history_ui()
        self._update_summary()

        if self.file_list.count() == 0:
            self.output_path.clear()

    def _undo_last_batch(self):
        """撤销最后一个批次"""
        if not self._batches:
            return
        self._undo_batch(self._batches[-1]["id"])

    def _redo_batch(self, batch_id: int):
        """重做指定批次"""
        batch = next((b for b in self._redo_stack if b["id"] == batch_id), None)
        if not batch:
            return

        if batch.get("batch_type") == "add":
            # 重新添加文件
            for file_path in batch["files"]:
                try:
                    info = get_pdf_info(file_path)
                    item_text = f"{info['filename']}  ({info['page_count']} 页)"
                except Exception:
                    item_text = os.path.basename(file_path)
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, file_path)
                self.file_list.addItem(item)
        else:
            # 重新删除文件
            removed_paths = {item["file_path"] for item in batch["items"]}
            i = 0
            while i < self.file_list.count():
                item = self.file_list.item(i)
                if item.data(Qt.UserRole) in removed_paths:
                    self.file_list.takeItem(i)
                else:
                    i += 1

        self._redo_stack.remove(batch)
        self._batches.append(batch)
        self._rebuild_history_ui()
        self._update_summary()

        if self.file_list.count() == 0:
            self.output_path.clear()

    def _redo_last_batch(self):
        """重做最后一个撤销的批次"""
        if not self._redo_stack:
            return
        self._redo_batch(self._redo_stack[-1]["id"])

    def _rebuild_history_ui(self):
        """重建操作历史列表（可滚动）"""
        self._history_list.clear()

        # 按添加顺序显示（最新的在最上面）
        for batch in reversed(self._batches):
            item = QListWidgetItem()
            item.setData(Qt.UserRole, batch["id"])

            # 创建条目控件
            frame = QFrame()
            frame.setStyleSheet(
                "QFrame { background-color: #fafafa; border: 1px solid #eee; "
                "border-radius: 3px; margin: 2px; }"
            )
            row = QHBoxLayout(frame)
            row.setContentsMargins(8, 4, 8, 4)
            row.setSpacing(6)

            # 标签
            label = QLabel(batch["label"])
            label.setWordWrap(True)
            label.setStyleSheet("color: #333; font-size: 12px; border: none;")

            if batch.get("batch_type") == "delete":
                count = len(batch["items"])
                label.setStyleSheet("color: #d32f2f; font-size: 12px; border: none;")
            else:
                count = len(batch["files"])

            count_label = QLabel(f"({count} 个)")
            count_label.setStyleSheet("color: #999; font-size: 11px; border: none;")

            undo_btn = QPushButton("撤销")
            undo_btn.setFixedSize(50, 24)
            undo_btn.setStyleSheet(
                "QPushButton { color: #d32f2f; border: 1px solid #d32f2f; "
                "border-radius: 3px; font-size: 11px; padding: 0; }"
                "QPushButton:hover { background-color: #ffebee; }"
            )
            bid = batch["id"]
            undo_btn.clicked.connect(lambda checked, bid=bid: self._undo_batch(bid))

            row.addWidget(label, 1)
            row.addWidget(count_label)
            row.addWidget(undo_btn)

            item.setSizeHint(frame.sizeHint())
            self._history_list.addItem(item)
            self._history_list.setItemWidget(item, frame)

    def _on_history_double_click(self, item):
        """双击历史条目：如果在撤销栈中则重做，如果在重做栈中弹回"""
        batch_id = item.data(Qt.UserRole)
        # 检查是否在重做栈中
        redo_batch = next((b for b in self._redo_stack if b["id"] == batch_id), None)
        if redo_batch:
            self._redo_batch(batch_id)
        else:
            self._undo_batch(batch_id)

    # ── 文件操作 ─────────────────────────────────────

    def _on_mode_changed(self):
        self._update_summary()

    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择 PDF 文件", "", "PDF 文件 (*.pdf)"
        )
        if not files:
            return

        if not self.rb_specified.isChecked():
            QMessageBox.information(
                self, "提示",
                '当前为文件夹排序模式，请使用「添加文件夹」按钮'
            )
            return

        added = self._add_files_to_list(files, "指定顺序")
        if added:
            label = f"文件 × {len(added)}"
            self._add_batch(label, added)

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹（含子目录）")
        if not folder:
            return

        folder_name = os.path.basename(folder) or folder

        if self.rb_folder.isChecked():
            files = sort_folder_order(folder)
        elif self.rb_by_folder.isChecked():
            files = sort_by_folder_order(folder)
        else:
            # 指定顺序模式：递归扫描
            files = _scan_pdfs_recursive(folder)
            natural_sort(files)

        if not files:
            QMessageBox.information(self, "提示", f'文件夹「{folder_name}」中未找到 PDF 文件')
            return

        # 导入时使用自然排序，但指定顺序模式允许拖拽
        added = self._add_files_to_list(files, folder_name)
        if added:
            self._add_batch(f"文件夹: {folder_name}", added)

    def _add_files_to_list(self, files: list[str], source_label: str) -> list[str]:
        """添加文件到列表，返回实际新增的文件路径列表"""
        added = []
        for file_path in files:
            if not file_path.lower().endswith(".pdf"):
                continue
            # 检查是否已存在
            existing = False
            for i in range(self.file_list.count()):
                if self.file_list.item(i).data(Qt.UserRole) == file_path:
                    existing = True
                    break
            if existing:
                continue

            try:
                info = get_pdf_info(file_path)
                item_text = f"{info['filename']}  ({info['page_count']} 页)"
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, file_path)
                self.file_list.addItem(item)
            except Exception:
                item = QListWidgetItem(os.path.basename(file_path))
                item.setData(Qt.UserRole, file_path)
                self.file_list.addItem(item)
            added.append(file_path)

        if added:
            # 先设置默认输出路径，再更新汇总（汇总里会根据 output_path 启用按钮）
            self._auto_set_output_path()
            self._update_summary()

        return added

    def _remove_selected(self):
        """删除列表中选中的文件（同时生成可撤销的删除批次）"""
        selected = self.file_list.selectedItems()
        if not selected:
            return

        # 收集被删除的文件信息
        removed_items = []
        removed_paths = set()
        for item in selected:
            file_path = item.data(Qt.UserRole)
            display_text = item.text()
            removed_items.append({"file_path": file_path, "display_text": display_text})
            removed_paths.add(file_path)

        for item in selected:
            row = self.file_list.row(item)
            self.file_list.takeItem(row)

        # 从已有批次记录中移除被删除的文件
        for batch in self._batches:
            if batch.get("batch_type") == "add":
                batch["files"] = [f for f in batch["files"] if f not in removed_paths]
        # 移除空批次
        self._batches = [b for b in self._batches if b.get("batch_type") != "add" or b["files"]]

        # 记录删除批次（可撤销）
        batch_id = self._next_batch_id
        self._next_batch_id += 1
        self._batches.append({
            "id": batch_id,
            "batch_type": "delete",
            "label": f"删除 {len(removed_items)} 个文件",
            "items": removed_items
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

    # ── 输出路径 ─────────────────────────────────────

    def _auto_set_output_path(self):
        """自动生成默认输出路径"""
        if self.output_path.text():
            return  # 已有路径就不覆盖

        files = self.file_list.get_ordered_files()
        if not files:
            return

        first_dir = os.path.dirname(files[0])
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        default_path = os.path.join(first_dir, f"_合并结果_{timestamp}.pdf")
        self.output_path.setText(default_path)

    def _browse_output(self):
        files = self.file_list.get_ordered_files()
        if files:
            default_dir = os.path.dirname(files[0])
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            default_path = os.path.join(default_dir, f"_合并结果_{timestamp}.pdf")
        else:
            default_path = ""
        path, _ = QFileDialog.getSaveFileName(
            self, "保存合并后的 PDF", default_path, "PDF 文件 (*.pdf)"
        )
        if path:
            if not path.lower().endswith(".pdf"):
                path += ".pdf"
            self.output_path.setText(path)
            self._update_summary()

    # ── 汇总 / 执行 ──────────────────────────────────

    def _update_summary(self):
        files = self.file_list.get_ordered_files()
        total_pages = 0
        for f in files:
            try:
                info = get_pdf_info(f)
                total_pages += info["page_count"]
            except Exception:
                pass
        self.summary_label.setText(f"已选择 {len(files)} 个文件，共 {total_pages} 页")
        self.btn_merge.setEnabled(len(files) > 0 and bool(self.output_path.text()))

    def _start_merge(self):
        files = self.file_list.get_ordered_files()
        output_path = self.output_path.text().strip()

        if not files:
            QMessageBox.warning(self, "提示", "请添加要合并的 PDF 文件")
            return
        if not output_path:
            # 兜底：自动生成默认路径（而不是弹警告让用户手动选）
            first_dir = os.path.dirname(files[0])
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(first_dir, f"_合并结果_{timestamp}.pdf")
            self.output_path.setText(output_path)

        self._set_ui_enabled(False)
        self.btn_cancel.setVisible(True)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("正在准备...")

        self.worker = MergeWorker(files, output_path, fast_mode=self.cb_fast_mode.isChecked())
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
        self.status_label.setText("合并完成！")

        reply = QMessageBox.information(
            self, "合并完成",
            f"文件已保存到:\n{result_path}\n\n是否打开输出目录？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            os.startfile(os.path.dirname(result_path))

    def _on_error(self, error_msg: str):
        self._set_ui_enabled(True)
        self.btn_cancel.setVisible(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("合并失败")
        if error_msg != "操作已取消":
            QMessageBox.critical(self, "合并失败", error_msg)

    def _cancel_operation(self):
        """取消正在进行的合并操作"""
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.progress_bar.setVisible(False)
            self.status_label.setText("正在取消...")
            # 尝试删除部分输出文件
            output_path = self.output_path.text()
            if output_path and os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except Exception:
                    pass

    def _set_ui_enabled(self, enabled: bool):
        self.btn_add_files.setEnabled(enabled)
        self.btn_add_folder.setEnabled(enabled)
        self.btn_remove.setEnabled(enabled)
        self.btn_clear.setEnabled(enabled)
        self.btn_merge.setEnabled(enabled)
        self.btn_browse_output.setEnabled(enabled)
        self.output_path.setEnabled(enabled)
        self.file_list.setEnabled(enabled)
        self.rb_specified.setEnabled(enabled)
        self.rb_folder.setEnabled(enabled)
        self.rb_by_folder.setEnabled(enabled)
        self.btn_cancel.setEnabled(enabled)
        self._history_list.setEnabled(enabled)
        self.cb_fast_mode.setEnabled(enabled)