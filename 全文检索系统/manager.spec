# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['manager.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'PySide6', 'shiboken6', 'contourpy', 'kiwisolver', 'dateutil', 'IPython', 'jedi', 'parso', 'prompt_toolkit', 'pyreadline3', 'wcwidth', 'Crypto', 'Cython', 'yaml', 'chardet', 'charset_normalizer', 'psutil', 'pygments', 'gi', 'torch', 'torchvision', 'torchaudio', 'sentence_transformers', 'transformers', 'tokenizers', 'safetensors', 'sklearn', 'scipy', 'faiss', 'numpy'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='manager',
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
    name='manager',
)
