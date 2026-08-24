# -*- coding: utf-8 -*-
"""打包脚本：使用 PyInstaller 将系统打包为单文件夹模式（支持向量索引）。

生成的程序启动时在控制台交互式设置端口（默认 20000，回车确认；
若被占用则自动向后寻找可用端口），
确认后以服务器模式运行。服务启动后控制台不再输出，
所有日志写入 exe 同级 output/log/server.log（10MB 轮转）。

用法：
    python build.py

产物：
    dist/全文检索系统/           单文件夹模式（含 exe 与 _internal 依赖目录）
    dist/全文检索系统/全文检索系统.exe   主程序入口

运行方式：
    # 单实例模式（交互式设置端口）
    全文检索系统/全文检索系统.exe

运行时数据布局（exe 同级目录）：
    data/                主数据区（库、注册表、设置、会话）
    data/libraries/      各库数据目录
    output/log/          日志文件（10MB 轮转）

注意：分发时需拷贝整个"全文检索系统"文件夹（exe 与 _internal 缺一不可）。
"""
import os
import subprocess
import sys
import shutil
import importlib.util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_NAME = "全文检索系统"
ENTRY = "web_api.py"


def check_pyinstaller():
    """确认 PyInstaller 已安装。"""
    try:
        import PyInstaller  # noqa: F401
        print("[ok] PyInstaller 已安装")
    except ImportError:
        print("[installing] 正在安装 PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("[ok] PyInstaller 安装完成")


def ensure_optional_deps():
    """确保可选依赖（docx/pdf）已安装，缺失时自动安装。"""
    for mod, desc, pkg in [("docx", "Word(.docx)", "python-docx"), ("pypdf", "PDF", "pypdf")]:
        try:
            __import__(mod)
            print(f"[ok] {desc} 支持 (已安装 {mod})")
        except ImportError:
            print(f"[installing] 正在安装 {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            print(f"[ok] {pkg} 安装完成")


def collect_faiss_binaries():
    """
    收集 faiss 运行所需的全部二进制依赖。

    faiss 通过 wheel 安装时通常包含：
      - faiss 包目录下的 faiss.dll / _swigfaiss*.pyd / _faiss_example_external_module*.pyd
      - site-packages 根目录下的 faiss_cpu.libs/（含 libopenblas.dll 等）
    PyInstaller 的 --collect-all faiss 会漏掉其中部分 .dll，需要手动补充。

    返回 [(src_path, dest_dir), ...]：
      - src_path 可以是文件或目录
      - dest_dir 为打包后相对于 _internal 的目标路径
    """
    try:
        import faiss
    except ImportError:
        print("[warn] faiss 未安装，无法获取动态库路径")
        return []

    faiss_dir = os.path.dirname(faiss.__file__)
    site_packages = os.path.dirname(faiss_dir)
    print(f"[debug] faiss 包目录: {faiss_dir}")

    entries = []

    # 1) 收集 faiss 包目录下的所有 DLL / PYD（Windows）或 .so/.dylib（Unix）
    for root, dirs, files in os.walk(faiss_dir):
        for f in files:
            if f.endswith(('.pyd', '.dll', '.so', '.dylib')):
                full_path = os.path.join(root, f)
                # 保持与 faiss 包目录相同的相对结构
                rel_dir = os.path.relpath(root, faiss_dir)
                dest = os.path.join('faiss', rel_dir) if rel_dir != '.' else 'faiss'
                entries.append((full_path, dest))

    # 2) 收集 faiss_cpu.libs/ 目录（Windows wheel 带的 OpenBLAS / VC++ 运行时）
    faiss_libs_dir = os.path.join(site_packages, 'faiss_cpu.libs')
    if os.path.isdir(faiss_libs_dir):
        entries.append((faiss_libs_dir, 'faiss_cpu.libs'))

    if not entries:
        print("[warn] 未找到 faiss 相关动态库")
    else:
        print(f"[info] 发现 {len(entries)} 个 faiss 二进制依赖项")
    return entries

def build():
    check_pyinstaller()
    ensure_optional_deps()

    # 清理旧的构建产物
    for d in ["build", "dist"]:
        p = os.path.join(BASE_DIR, d)
        if os.path.isdir(p):
            try:
                shutil.rmtree(p)
                print(f"[clean] 已删除 {d}/")
            except PermissionError as e:
                print(f"[warn] 无法删除 {d}/：{e}")
                print("[hint] 该目录可能正被正在运行的程序占用（如已启动的 exe 或其日志文件），"
                      "请先关闭运行中的程序后重试。")
                sys.exit(1)
    spec = os.path.join(BASE_DIR, f"{APP_NAME}.spec")
    if os.path.isfile(spec):
        try:
            os.remove(spec)
        except PermissionError as e:
            print(f"[warn] 无法删除旧的 spec 文件：{e}")
            sys.exit(1)

    # 基础命令
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onedir",                  # 单文件夹模式（exe 与 _internal 依赖目录）
        "--console",                 # 保留控制台窗口：启动时交互输入端口
        "--name", APP_NAME,
        "--noconfirm",
        "--clean",
        # 包含 static 资源（Windows 用分号分隔）
        f"--add-data=static{os.pathsep}static",
        # 包含本地模型目录，确保打包后无需联网即可加载 bge-small-zh
        f"--add-data=models{os.pathsep}models",
        # 显式声明动态导入的可选依赖（PyInstaller 静态分析检测不到函数内 import）
        "--hidden-import=docx",
        "--hidden-import=pypdf",
        "--hidden-import=PyPDF2",
        # --collect-all 确保子模块/数据文件（如 docx templates）不遗漏
        "--collect-all", "docx",
        "--collect-all", "pypdf",
        # ========== 新增：语义检索核心依赖（必须显式收集）==========
        "--hidden-import=faiss",
        "--hidden-import=numpy",
        "--hidden-import=requests",
        "--hidden-import=sentence_transformers",
        "--hidden-import=transformers",
        "--hidden-import=torch",
        "--hidden-import=torch.nn",
        "--hidden-import=torch.utils.data",
        "--hidden-import=tiktoken",          # sentence-transformers / transformers 可能用到
        "--hidden-import=tiktoken_ext",      # tiktoken 扩展
        "--hidden-import=regex",             # transformers 依赖
        "--hidden-import=safetensors",       # transformers 模型加载
        "--hidden-import=tokenizers",        # transformers 分词器
        "--hidden-import=jieba",             # 中文分词（元数据统计/标签提取）
        "--hidden-import=jieba.posseg",      # jieba 词性标注（实体密度识别）
        "--hidden-import=jieba.analyse",     # jieba TF-IDF 关键词提取（标签提取）
        # --collect-all 能更完整地打包子模块和资源
        "--collect-all", "sentence_transformers",
        "--collect-all", "transformers",
        "--collect-all", "torch",
        "--collect-all", "faiss",
        "--collect-all", "numpy",
        "--collect-all", "tiktoken",
        "--collect-all", "tokenizers",
        "--collect-all", "safetensors",
        "--collect-all", "jieba",            # 收集 jieba 词典数据文件（dict.txt 等）
    ]

    # 处理 faiss 动态库（关键，否则报错找不到 DLL）
    faiss_entries = collect_faiss_binaries()
    if faiss_entries:
        for src, dest in faiss_entries:
            if os.path.isdir(src):
                cmd.append(f"--add-data={src}{os.pathsep}{dest}")
            else:
                cmd.append(f"--add-binary={src}{os.pathsep}{dest}")
            print(f"[info] 将 faiss 依赖 {src} -> {dest} 加入打包")
    else:
        print("[warn] 未找到 faiss 动态库，可能会导致运行时加载失败")

    # ========== 排除无关依赖 ==========
    # 本程序是 Web 服务，仅需 torch/transformers/sentence_transformers/faiss/numpy/
    # sklearn/scipy(后者为 sentence_transformers 运行时依赖)/jieba 等核心库。
    # 以下模块均非运行所需，排除后可大幅减小包体积并加速二进制依赖分析。
    exclude_modules = [
        # GUI 框架（Web 服务不需要）
        "matplotlib", "PySide6", "shiboken6", "PyQt5", "PyQt6", "qtpy",
        "tkinter", "_tkinter", "contourpy",
        # 数据分析（sklearn/scipy 为 sentence_transformers 依赖，必须保留）
        "pandas", "xarray", "zarr", "statsmodels", "patsy",
        # 绘图 / 可视化
        "bokeh", "plotly", "altair", "holoviews", "panel",
        # 图像处理（文本向量检索不需要）
        "skimage", "scikit-image", "imageio", "pywt", "PIL", "pillow",
        # 天文数据
        "astropy", "astropy_iers_data",
        # JIT 编译（体积大，运行时不需要）
        "numba", "llvmlite",
        # 并行/分布式计算
        "dask", "distributed", "cloudpickle",
        # 大数据存储
        "pyarrow", "tables", "pytables", "h5py",
        # 数据库 ORM
        "sqlalchemy",
        # 云服务 / AWS SDK
        "botocore", "boto3", "s3fs",
        # NLP（使用 jieba，不需要 nltk）
        "nltk",
        # Jupyter / IPython
        "jupyterlab", "notebook", "nbconvert", "nbformat", "jupyter_client",
        "jupyter_core", "ipykernel", "IPython", "jedi", "parso",
        "zmq", "pyzmq",
        # 调试 / 系统监控
        "debugpy", "psutil",
        # 文档工具
        "sphinx", "docutils", "babel",
        # 电子表格 I/O
        "openpyxl", "xlrd",
        # 代码工具
        "prompt_toolkit", "pyreadline3", "wcwidth", "Crypto", "Cython",
        "pygments", "gi",
        # torch 扩展（CPU 推理不需要）
        "torchvision", "torchaudio",
        # 压缩
        "lz4",
        # Web 框架（使用标准库 http.server）
        "tornado", "aiohttp",
        # 模板引擎（运行时不需要）
        "jinja2", "markdown", "mistune",
        # 终端美化
        "rich",
        # JSON schema 校验
        "jsonschema",
        # 注意：fsspec 此前被排除，但 sentence-transformers / transformers 运行时
        #       import fsspec（加载模型/hub），排除会导致语义检索依赖不可用，
        #       故必须保留。同时保留 yaml、charset_normalizer、chardet、tqmm、
        #       huggingface_hub、sympy、pydantic 等，否则会触发 ImportError
    ]
    for mod in exclude_modules:
        cmd.append(f"--exclude-module={mod}")

    cmd.append(ENTRY)

    print(f"\n[build] 开始打包主程序（包含向量检索）...")
    print(f"  命令: {' '.join(cmd)}\n")
    print("[注意] 由于包含 torch 和 transformers，最终包体积可能超过 2GB，请确保磁盘空间充足")

    # 避免 PyInstaller 收集 torch 等含 OpenMP 依赖库的子模块时，
    # 因多个 libiomp5md.dll 重复初始化而导致子进程崩溃。
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    result = subprocess.run(cmd, cwd=BASE_DIR, env=env)

    if result.returncode != 0:
        print("\n[fail] 主程序打包失败")
        sys.exit(1)

    exe_path = os.path.join(BASE_DIR, "dist", APP_NAME, f"{APP_NAME}.exe")
    if not os.path.isfile(exe_path):
        print(f"\n[fail] 未找到生成的 exe: {exe_path}")
        sys.exit(1)

    # 统计整个产物文件夹大小
    total_bytes = 0
    for root, dirs, files in os.walk(os.path.join(BASE_DIR, "dist", APP_NAME)):
        for f in files:
            try:
                total_bytes += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    size_mb = total_bytes / 1024 / 1024
    print(f"\n[ok] 主程序打包成功！")
    print(f"  程序: {exe_path}")
    print(f"  大小: {size_mb:.1f} MB（单文件夹，含全部依赖与本地模型）")

    # ============ 输出使用说明 ============
    print(f"\n{'='*60}")
    print(f"打包完成！")
    print(f"{'='*60}")
    print(f"\n  主程序（单文件夹模式，分发时拷贝整个文件夹）：")
    print(f"    \"{exe_path}\"")
    print(f"    启动后在控制台输入端口（默认 20000，回车确认；占用则自动后移）")
    print(f"    可选参数：--port 9000 / --no-dialog / --data-dir <路径>")
    print(f"\n  注意：向量索引已启用，默认从内置的 bge-small-zh 本地模型加载，")
    print(f"        无需联网下载。如需更换模型，可在 _settings.json 中修改 semantic_model_path。")


if __name__ == "__main__":
    build()