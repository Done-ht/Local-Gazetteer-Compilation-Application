import sys
from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QLabel, QStatusBar
)
from PyQt5.QtCore import Qt

from .merge.merge_page import MergePage
from .split.split_page import SplitPage
from .convert.convert_page import ConvertPage
from .compose.compose_page import ComposePage
from .append.append_page import AppendPage
from .insert.insert_page import InsertPage


class MainWindow(QMainWindow):
    """PDF 工具主窗口"""

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("PDF 工具 - 合并/拆分/转换/合成/拼接/插入")
        self.setMinimumSize(800, 600)
        self.resize(960, 720)

        # 中央部件
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        # 标签页
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #ccc;
                padding: 10px;
            }
            QTabBar::tab {
                padding: 10px 20px;
                font-size: 14px;
                min-width: 80px;
            }
            QTabBar::tab:selected {
                font-weight: bold;
                border-bottom: 2px solid #4CAF50;
            }
        """)

        self.merge_page = MergePage()
        self.split_page = SplitPage()
        self.convert_page = ConvertPage()
        self.compose_page = ComposePage()
        self.append_page = AppendPage()
        self.insert_page = InsertPage()

        self.tabs.addTab(self.merge_page, "合并")
        self.tabs.addTab(self.split_page, "拆分")
        self.tabs.addTab(self.convert_page, "转换")
        self.tabs.addTab(self.compose_page, "合成")
        self.tabs.addTab(self.append_page, "拼接")
        self.tabs.addTab(self.insert_page, "插入")

        layout.addWidget(self.tabs)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪 | 全程本地处理，文件不上传服务器")

        # 样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                padding: 6px 16px;
                border: 1px solid #ccc;
                border-radius: 3px;
                background-color: #fff;
            }
            QPushButton:hover {
                background-color: #e8e8e8;
            }
            QPushButton:disabled {
                color: #aaa;
                background-color: #f0f0f0;
            }
            QLineEdit {
                padding: 4px 8px;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
            QListWidget {
                border: 1px solid #ccc;
                border-radius: 3px;
                background-color: white;
            }
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 3px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 2px;
            }
        """)