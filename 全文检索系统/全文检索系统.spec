# -*- mode: python ; coding: utf-8 -*-
import os

from PyInstaller.utils.hooks import collect_all

# faiss 的二进制文件位置随 Python 环境不同而不同，
# 这里从当前环境已安装的 faiss 包动态定位，不硬编码绝对路径。
import faiss

_faiss_dir = os.path.dirname(os.path.abspath(faiss.__file__))
_site_packages = os.path.dirname(_faiss_dir)

datas = [('static', 'static'), ('models', 'models')]
_faiss_cpu_libs = os.path.join(_site_packages, 'faiss_cpu.libs')
if os.path.isdir(_faiss_cpu_libs):
    datas.append((_faiss_cpu_libs, 'faiss_cpu.libs'))

binaries = []
for _name in ('faiss.dll', '_swigfaiss.pyd'):
    _p = os.path.join(_faiss_dir, _name)
    if os.path.isfile(_p):
        binaries.append((_p, 'faiss'))
hiddenimports = ['docx', 'pypdf', 'PyPDF2', 'faiss', 'numpy', 'requests', 'sentence_transformers', 'transformers', 'torch', 'torch.nn', 'torch.utils.data', 'tiktoken', 'tiktoken_ext', 'regex', 'safetensors', 'tokenizers', 'jieba', 'jieba.posseg', 'jieba.analyse']
tmp_ret = collect_all('docx')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pypdf')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('sentence_transformers')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('transformers')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('torch')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('faiss')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('numpy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('tiktoken')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('tokenizers')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('safetensors')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('jieba')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['web_api.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'PySide6', 'shiboken6', 'PyQt5', 'PyQt6', 'qtpy', 'tkinter', '_tkinter', 'contourpy', 'pandas', 'xarray', 'zarr', 'statsmodels', 'patsy', 'bokeh', 'plotly', 'altair', 'holoviews', 'panel', 'skimage', 'scikit-image', 'imageio', 'pywt', 'PIL', 'pillow', 'astropy', 'astropy_iers_data', 'numba', 'llvmlite', 'dask', 'distributed', 'cloudpickle', 'pyarrow', 'tables', 'pytables', 'h5py', 'sqlalchemy', 'botocore', 'boto3', 's3fs', 'nltk', 'jupyterlab', 'notebook', 'nbconvert', 'nbformat', 'jupyter_client', 'jupyter_core', 'ipykernel', 'IPython', 'jedi', 'parso', 'zmq', 'pyzmq', 'debugpy', 'psutil', 'sphinx', 'docutils', 'babel', 'openpyxl', 'xlrd', 'prompt_toolkit', 'pyreadline3', 'wcwidth', 'Crypto', 'Cython', 'pygments', 'gi', 'torchvision', 'torchaudio', 'lz4', 'tornado', 'aiohttp', 'jinja2', 'markdown', 'mistune', 'rich', 'jsonschema'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='全文检索系统',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='全文检索系统',
)
