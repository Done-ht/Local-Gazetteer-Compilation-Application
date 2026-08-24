# -*- mode: python ; coding: utf-8 -*-
# PDF 工具 · Web 服务版打包配置
# 用法：pyinstaller web_app.spec --noconfirm
# 产物：dist/PDF工具Web版.exe（单文件，双击即启动 Web 服务）

import os

# 项目根目录（spec 所在目录）与 web 目录
_PROJECT = os.path.dirname(os.path.abspath(SPEC))
_WEB = os.path.join(_PROJECT, "web")

# Web 版仅依赖引擎（fitz/docx/lxml/flask），无需 PyQt5 图形界面
a = Analysis(
    [os.path.join(_WEB, "run_web.py")],
    pathex=[_PROJECT, _WEB],
    binaries=[],
    datas=[
        (os.path.join(_WEB, "templates"), "templates"),
        (os.path.join(_WEB, "static"), "static"),
    ],
    hiddenimports=[
        # web 顶层模块（server 内为运行时注入 sys.path，需显式收集）
        "server",
        "local_detector",
        "path_browser",
        "open_dir",
        "task_manager",
        "auth",
        "userdata",
        "web_logger",
        # app 业务引擎与工具
        "app",
        "app.merge",
        "app.merge.merge_engine",
        "app.merge.sorter",
        "app.split",
        "app.split.split_engine",
        "app.convert",
        "app.convert.convert_engine",
        "app.compose",
        "app.compose.compose_engine",
        "app.append",
        "app.append.append_engine",
        "app.insert",
        "app.insert.insert_engine",
        "app.utils",
        "app.utils.pdf_utils",
        "app.utils.natural_sort",
        # Flask 相关（PyInstaller 钩子通常已覆盖，显式补全更稳妥）
        "flask",
        "werkzeug",
        "jinja2",
        "markupsafe",
        "itsdangerous",
        "click",
        "blinker",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "tkinter",
        "matplotlib",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="PDF工具Web版",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(_PROJECT, "1.ico") if os.path.exists(os.path.join(_PROJECT, "1.ico")) else None,
)
