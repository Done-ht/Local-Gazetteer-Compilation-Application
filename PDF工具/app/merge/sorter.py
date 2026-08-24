import os
from ..utils.natural_sort import natural_sort, natural_sort_key


def _scan_pdfs_recursive(root_path: str) -> list[str]:
    """递归扫描目录及子目录下所有 PDF 文件"""
    pdf_files = []
    for dirpath, _, filenames in os.walk(root_path):
        for f in filenames:
            if f.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(dirpath, f))
    return pdf_files


def sort_specified_order(files: list[str]) -> list[str]:
    """模式①：指定顺序 — 保持用户拖拽排列后的顺序"""
    return list(files)


def sort_folder_order(folder_path: str) -> list[str]:
    """
    模式②：文件夹内顺序（递归）
    - 扫描文件夹及所有子目录内的 PDF
    - 按自然排序
    """
    pdf_files = _scan_pdfs_recursive(folder_path)
    natural_sort(pdf_files)
    return pdf_files


def sort_by_folder_order(root_path: str) -> list[str]:
    """
    模式③：按文件夹顺序（递归两级）
    - 根目录散落的 PDF 排在最前
    - 子文件夹按名称自然排序，子文件夹内 PDF 自然排序
    - 子文件夹内的子文件夹内容合并到父文件夹中
    """
    # 收集根目录 PDF
    root_files = []
    for entry in os.scandir(root_path):
        if entry.is_file() and entry.name.lower().endswith(".pdf"):
            root_files.append(entry.path)
    natural_sort(root_files)

    # 收集子文件夹（仅一级）的 PDF
    dir_pdf_files: dict[str, list[str]] = {}
    for entry in os.scandir(root_path):
        if entry.is_dir():
            pdfs = []
            for dirpath, _, filenames in os.walk(entry.path):
                for f in filenames:
                    if f.lower().endswith(".pdf"):
                        pdfs.append(os.path.join(dirpath, f))
            if pdfs:
                natural_sort(pdfs)
                dir_pdf_files[entry.name] = pdfs

    # 按文件夹名称自然排序
    sorted_dirs = sorted(dir_pdf_files.keys(), key=natural_sort_key)
    result = list(root_files)
    for d in sorted_dirs:
        result.extend(dir_pdf_files[d])

    return result