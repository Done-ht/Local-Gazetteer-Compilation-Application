"""
PDF 工具 - Windows 桌面版
功能：PDF 合并（三种排序）/ 拆分（四种模式）/ 转 DOCX
技术栈：Python + PyQt5 + PyMuPDF + python-docx
"""

import sys
import os

# 确保当前目录在模块搜索路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 尽早初始化日志系统：必须早于任何业务模块导入，
# 否则第三方库（PyQt5 / PyMuPDF 等）的 print 输出无法被捕获。
from app.utils.logger import setup_logging
setup_logging()

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from app.main_window import MainWindow


def main():
    # 设置高 DPI 支持
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("PDF 工具")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()