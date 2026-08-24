"""PyInstaller 运行期 hook：让 paddleocr 内部以"顶层包"方式导入子包。

背景：
  paddleocr/paddleocr.py 在导入时会执行
      sys.path.append(os.path.join(__dir__, ''))
  随后用
      from ppocr.utils.logging import get_logger
      from tools.infer import predict_system
      from ppstructure.utility import init_args, ...
  把 ppocr / ppstructure / tools 当成"顶层包"导入。

  PyInstaller 静态分析无法跟踪这种 sys.path 动态修改 + 顶层导入，
  导致 frozen 环境下 `import paddleocr` 时找不到 ppocr / ppstructure / tools
  顶层模块，触发 ImportError → is_available() 返回 False →
  "选择了 paddle_local 但 PaddleOCR 不可用"。

修复：
  在 paddleocr 被导入之前，把 _internal/paddleocr/ 插入 sys.path，
  这样标准 PathFinder 能在磁盘上找到 ppocr/ ppstructure/ tools/ 三个子目录，
  把它们当作顶层包加载。collect_all("paddleocr") 已把全部 .py 作为数据文件
  放到该目录，磁盘上文件齐全，导入即可成功。

  必须作为 runtime_hooks 注册（在用户代码前执行），否则 paddleocr.py
  顶层 import 会先于本 hook 运行，仍然失败。
"""
from __future__ import annotations

import os
import sys


def _inject_paddleocr_dir() -> None:
    """把 paddleocr 包目录加入 sys.path，供 ppocr/ppstructure/tools 顶层导入。"""
    if not getattr(sys, "frozen", False):
        # 开发环境不需要：源码目录本身就在 sys.path 里
        return
    # onedir 模式：paddleocr 包位于 <exe_dir>/_internal/paddleocr/
    exe_dir = os.path.dirname(sys.executable)
    candidates = [
        os.path.join(exe_dir, "_internal", "paddleocr"),
        os.path.join(exe_dir, "paddleocr"),
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "ppocr")) and os.path.isdir(
            os.path.join(c, "ppstructure")
        ):
            if c not in sys.path:
                sys.path.insert(0, c)
            return


_inject_paddleocr_dir()
