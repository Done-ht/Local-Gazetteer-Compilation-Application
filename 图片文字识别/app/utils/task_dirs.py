"""任务目录管理器。

统一管理任务工作目录结构：
  output/
    data/                         # 所有任务数据根目录
      {task_id}/                  # 单个任务工作目录
        source/                   # 原始上传文件（拆分前的原件，拆分后删除）
        pdf_pages/                # 拆分后的单页 PDF（page_XXXX.pdf）
        ocr_pages/                # OCR 后的单页 PDF + JSON
          page_XXXX.pdf           # 含文字层的单页 PDF
          page_XXXX.json          # OCR 结果 JSON
        _{name}_OCR.pdf           # 最终合并的可编辑 PDF
        _{name}_partial_OCR.pdf   # 提前导出的 PDF
        meta.json                 # 任务元信息
    log/                          # 所有日志
      ocr_service.log             # 服务日志
      ocr_progress.log            # 进度日志
      error_ocr.txt               # OCR 错误日志（页数异常等）
"""
from __future__ import annotations

import os
import re
import sys
from typing import List, Optional


# 任务数据根目录（output/data），相对于服务根目录或 exe 同级目录
def _output_root() -> str:
    """返回 output 目录的绝对路径。

    优先级：
      1. 打包环境：exe 同级目录下的 output/
      2. 开发环境：server-paddle/output/
    """
    if getattr(os, "frozen", False):
        # 打包环境：exe 同级目录
        base = os.path.dirname(os.path.abspath(sys.executable))
    else:
        # 开发环境：server-paddle/ 目录
        # task_dirs.py 位于 app/utils/，向上两级到 server-paddle/
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, "output")


# 模块级常量，启动时确定
OUTPUT_ROOT = _output_root()
DATA_DIR = os.path.join(OUTPUT_ROOT, "data")
LOG_DIR = os.path.join(OUTPUT_ROOT, "log")


def ensure_dirs() -> None:
    """启动时调用，创建 output/data 和 output/log 目录。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def task_dir(task_id: str) -> str:
    """返回任务工作目录路径。"""
    return os.path.join(DATA_DIR, task_id)


def source_dir(task_id: str) -> str:
    """返回原始文件目录路径。"""
    return os.path.join(task_dir(task_id), "source")


def pdf_pages_dir(task_id: str) -> str:
    """返回拆分后的单页 PDF 目录路径。"""
    return os.path.join(task_dir(task_id), "pdf_pages")


def ocr_pages_dir(task_id: str) -> str:
    """返回 OCR 后的单页 PDF 目录路径。"""
    return os.path.join(task_dir(task_id), "ocr_pages")


def page_pdf_name(page_no: int) -> str:
    """返回单页 PDF 文件名（page_XXXX.pdf，4 位零填充）。"""
    return f"page_{page_no:04d}.pdf"


def page_json_name(page_no: int) -> str:
    """返回单页 OCR JSON 文件名（page_XXXX.json，4 位零填充）。"""
    return f"page_{page_no:04d}.json"


def page_pdf_path(task_id: str, page_no: int) -> str:
    """返回 pdf_pages 中指定页的 PDF 路径。"""
    return os.path.join(pdf_pages_dir(task_id), page_pdf_name(page_no))


def ocr_pdf_path(task_id: str, page_no: int) -> str:
    """返回 ocr_pages 中指定页的 PDF 路径。"""
    return os.path.join(ocr_pages_dir(task_id), page_pdf_name(page_no))


def ocr_json_path(task_id: str, page_no: int) -> str:
    """返回 ocr_pages 中指定页的 JSON 路径。"""
    return os.path.join(ocr_pages_dir(task_id), page_json_name(page_no))


# 页号正则：匹配 page_XXXX.pdf 或 page_XXXX.json 中的页号
_PAGE_RE = re.compile(r"page_(\d+)\.(?:pdf|json)$")


def list_pdf_pages(task_id: str) -> List[int]:
    """返回 pdf_pages 目录下所有页号（已排序）。

    用于确定总页数和拆分后的页范围。
    """
    d = pdf_pages_dir(task_id)
    if not os.path.isdir(d):
        return []
    pages = []
    for name in os.listdir(d):
        if name.endswith(".pdf"):
            m = _PAGE_RE.match(name)
            if m:
                pages.append(int(m.group(1)))
    return sorted(pages)


def list_ocr_pages(task_id: str) -> List[int]:
    """返回 ocr_pages 目录下所有已完成的页号（已排序）。

    完成定义：同时存在 page_XXXX.pdf 和 page_XXXX.json。
    只有 PDF 没有 JSON 视为不完整，不返回。
    """
    d = ocr_pages_dir(task_id)
    if not os.path.isdir(d):
        return []
    pages = []
    for name in os.listdir(d):
        if name.endswith(".json"):
            m = _PAGE_RE.match(name)
            if m:
                page_no = int(m.group(1))
                # 必须同时存在对应的 PDF 文件才算完成
                if os.path.isfile(os.path.join(d, page_pdf_name(page_no))):
                    pages.append(page_no)
    return sorted(pages)


def get_pending_pages(task_id: str) -> List[int]:
    """返回未完成 OCR 的页号列表（已排序）。

    对比 pdf_pages 和 ocr_pages，返回 pdf_pages 中存在但 ocr_pages 中不存在的页号。
    """
    pdf_pages = set(list_pdf_pages(task_id))
    ocr_pages = set(list_ocr_pages(task_id))
    pending = pdf_pages - ocr_pages
    return sorted(pending)


def get_total_pages(task_id: str) -> int:
    """返回 pdf_pages 中的总页数（拆分后的总页数）。"""
    return len(list_pdf_pages(task_id))


def get_completed_pages(task_id: str) -> int:
    """返回 ocr_pages 中已完成的页数。"""
    return len(list_ocr_pages(task_id))


def service_log_path() -> str:
    """返回服务日志路径（ocr_service.log）。"""
    return os.path.join(LOG_DIR, "ocr_service.log")


def progress_log_path() -> str:
    """返回进度日志路径（ocr_progress.log）。"""
    return os.path.join(LOG_DIR, "ocr_progress.log")


def error_log_path() -> str:
    """返回 OCR 错误日志路径。"""
    return os.path.join(LOG_DIR, "error_ocr.txt")


def task_meta_path(task_id: str) -> str:
    """返回任务元信息文件路径（meta.json）。

    保存在任务目录下，记录 task_concurrency、source_name 等，
    用于服务重启后扫描文件夹恢复任务状态。
    """
    return os.path.join(task_dir(task_id), "meta.json")


def log_error(task_id: str, message: str) -> None:
    """追加一条错误到 error_ocr.txt。"""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] 任务 {task_id}: {message}\n"
    try:
        with open(error_log_path(), "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def final_pdf_path(task_id: str, source_name: str) -> str:
    """返回最终合并的 PDF 路径（_{name}_OCR.pdf）。"""
    # 文件名以下划线开头，确保在目录中排在顶部
    base = os.path.splitext(os.path.basename(source_name))[0]
    return os.path.join(task_dir(task_id), f"_{base}_OCR.pdf")


def partial_pdf_path(task_id: str, source_name: str) -> str:
    """返回提前导出的 PDF 路径（_{name}_partial_OCR.pdf）。"""
    base = os.path.splitext(os.path.basename(source_name))[0]
    return os.path.join(task_dir(task_id), f"_{base}_partial_OCR.pdf")


def create_task_dirs(task_id: str) -> None:
    """创建任务所需的全部子目录。"""
    for d in [
        task_dir(task_id),
        source_dir(task_id),
        pdf_pages_dir(task_id),
        ocr_pages_dir(task_id),
    ]:
        os.makedirs(d, exist_ok=True)


def cleanup_task(task_id: str) -> None:
    """删除任务的整个工作目录。"""
    import shutil
    d = task_dir(task_id)
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
