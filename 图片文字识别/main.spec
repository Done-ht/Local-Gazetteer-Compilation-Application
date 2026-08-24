# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置（server-paddle 服务端版）。

采用 onedir 模式：
  - onefile 每次启动需解压到临时目录，大包（含 paddleocr/cv2）可能耗时 30-60 秒
  - onedir 无需解压，启动快，控制台可立即输出访问地址

关键：
  1. paddleocr / paddle（paddlepaddle）含大量动态 import 与 C 扩展，
     改用 collect_all 一次性收集 data + binaries + hiddenimports。
  2. paddleocr 内部用 sys.path.append + 顶层导入（from ppocr... / from tools...），
     通过 runtime_hooks（paddleocr_rt_hook.py）在启动时把 _internal/paddleocr/
     插入 sys.path，恢复顶层导入能力。
  3. PaddleOCR 3.x / PPStructureV3 默认从 ~/.paddlex/official_models/ 加载模型，
     首次运行联网下载；把已下载模型一起打包到 _internal/paddleocr_models/，支持离线运行。
  4. FastAPI / uvicorn / python-multipart / qrcode 也用 collect_all 收齐动态依赖。
  5. web/ 目录（index.html / style.css / app.js）作为数据文件打包到 _internal/app/web/。
  6. console=True：服务端需要控制台输出局域网访问地址与日志。

输出目录：dist/server-paddle/，入口：dist/server-paddle/server-paddle.exe
"""
import os
from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

# ----------------------------------------------------------------------
# 收集 paddleocr / paddle / fastapi / uvicorn 等完整包
# ----------------------------------------------------------------------
datas = []
binaries = []
hiddenimports = []

for pkg in (
    "paddleocr",
    "paddle",
    "imgaug",
    "imageio",
    "lmdb",
    "tifffile",
    # paddle.utils.cpp_extension 依赖 setuptools + Cython
    # 缺少 Cython/Utility/*.cpp 会导致 import paddle 失败
    "Cython",
    "setuptools",
    # Web 框架相关
    "fastapi",
    "uvicorn",
    "multipart",
    "anyio",
    "starlette",
    "qrcode",
):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
        print(f"[main.spec] collect_all({pkg}): "
              f"{len(d)} datas, {len(b)} binaries, {len(h)} hiddenimports")
    except Exception as e:
        print(f"[main.spec] collect_all({pkg}) skipped: {e}")

# imageio / imgaug 等需要 dist-info 元数据
for pkg in ("imageio", "imgaug", "tifffile", "lmdb", "fastapi", "uvicorn",
            "starlette", "anyio", "qrcode", "Cython", "setuptools"):
    try:
        datas += copy_metadata(pkg)
    except Exception as e:
        print(f"[main.spec] copy_metadata({pkg}) skipped: {e}")

# ----------------------------------------------------------------------
# skimage 子模块补丁：paddleocr 内部用到 PyInstaller 静态分析漏掉的子模块
# ----------------------------------------------------------------------
_skimage_subs = collect_submodules("skimage")
hiddenimports += _skimage_subs
print(f"[main.spec] collect_submodules(skimage): {len(_skimage_subs)} submodules")

# ----------------------------------------------------------------------
# 打包 PaddleOCR 模型文件，支持离线运行
# 模型源目录为项目本地 paddleocr_models/，含：
#   - PP-OCRv6 tiny/small/medium 三档 det+rec 模型
#   - PP-DocLayoutV3 版面检测模型
#   - 5 个表格结构识别模型（PP-LCNet_x1_0_table_cls / SLANeXt_wired /
#     SLANet_plus / RT-DETR-L_wired|wireless_table_cell_det）
#   - 2 个方向分类模型（PP-LCNet_x1_0_doc_ori / PP-LCNet_x1_0_textline_ori）
# ----------------------------------------------------------------------
_MODELS_SRC = os.path.join(os.path.dirname(os.path.abspath(SPEC)), "paddleocr_models")
_MODELS_DEST = "paddleocr_models"

if os.path.isdir(_MODELS_SRC):
    for sub in os.listdir(_MODELS_SRC):
        src = os.path.join(_MODELS_SRC, sub)
        if os.path.isdir(src):
            datas.append((src, os.path.join(_MODELS_DEST, sub)))
    print(f"[main.spec] 已包含 PaddleOCR 模型: {_MODELS_SRC}")
else:
    print(f"[main.spec] 警告: 未找到 PaddleOCR 模型目录 {_MODELS_SRC}，"
          f"打包后将依赖首次运行联网下载")

# ----------------------------------------------------------------------
# Web 前端资源（app/web 主页面 + app/xfyun/static 讯飞模式页面）
# ----------------------------------------------------------------------
_web_src = os.path.join(os.path.dirname(os.path.abspath(SPEC)), "app", "web")
if os.path.isdir(_web_src):
    datas.append((_web_src, os.path.join("app", "web")))
    print(f"[main.spec] 已包含 Web 资源: {_web_src}")
else:
    print(f"[main.spec] 警告: 未找到 Web 资源目录 {_web_src}")

_xf_static = os.path.join(os.path.dirname(os.path.abspath(SPEC)), "app", "xfyun", "static")
if os.path.isdir(_xf_static):
    datas.append((_xf_static, os.path.join("app", "xfyun", "static")))
    print(f"[main.spec] 已包含讯飞前端资源: {_xf_static}")
else:
    print(f"[main.spec] 警告: 未找到讯飞前端资源目录 {_xf_static}")

# ----------------------------------------------------------------------
# 中文字体文件：打包到 _internal/fonts/，确保任意 Windows 可用
# 英文版 Windows 无中文字体，不打包会导致 PDF 文字层丢失
# ----------------------------------------------------------------------
_FONT_DEST = "fonts"
_font_candidates = [
    ("C:/Windows/Fonts/simhei.ttf", "simhei.ttf"),      # 黑体（.ttf 支持子集化，首选）
    ("C:/Windows/Fonts/simkai.ttf", "simkai.ttf"),      # 楷体（.ttf 支持子集化）
    ("C:/Windows/Fonts/simsun.ttc", "simsun.ttc"),      # 宋体（.ttc 不支持子集化，备用）
    ("C:/Windows/Fonts/msyh.ttc", "msyh.ttc"),          # 微软雅黑（.ttc，备用）
]
_font_packed = 0
for src, name in _font_candidates:
    if os.path.isfile(src):
        datas.append((src, _FONT_DEST))
        _font_packed += 1
        print(f"[main.spec] 已包含字体: {name}")
    else:
        print(f"[main.spec] 跳过字体（本机不存在）: {name}")
if _font_packed == 0:
    print(f"[main.spec] 警告: 未找到任何中文字体，打包后英文版 Windows 将无法生成文字层")

# 显式补充隐藏导入
hiddenimports += [
    "ppocr",
    "ppstructure",
    # 共用用户系统（main.py 内函数级 import，需显式收集）
    "app.auth",
    "app.auth.routes",
    "app.auth.users",
    "app.auth.deps",
    # 讯飞云端模式（按模式挂载，需显式收集）
    "app.xfyun",
    "app.xfyun.routes",
    "app.xfyun.service",
    "app.xfyun.ocr_xfyun",
    "app.xfyun.exporters",
    # CV / 图像处理
    "cv2",
    "numpy",
    "PIL",
    "skimage",
    "scipy",
    # PDF / DOCX
    "fitz",
    "pymupdf",
    "docx",
    # Web 框架
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "multipart",
    "multipart.multipart",
    # 二维码
    "qrcode",
    "qrcode.main",
    "qrcode.image",
    "qrcode.image.pil",
    # 工具
    "requests",
    "pyclipper",
    "shapely",
    "setuptools",
    "psutil",
    # 标准库补丁
    "imghdr",
    "unittest.mock",
]

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    # paddleocr_rt_hook.py 在用户代码前执行，把 _internal/paddleocr/ 插入
    # sys.path，使 paddleocr 内部的 `from ppocr...` / `from ppstructure...` /
    # `from tools...` 顶层导入在 frozen 环境下也能成功。
    runtime_hooks=["paddleocr_rt_hook.py"],
    excludes=["PyQt5", "PyQt6", "PySide6", "tkinter", "test"],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="server-paddle",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # 服务端需要控制台输出访问地址
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="1.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="server-paddle",
)
