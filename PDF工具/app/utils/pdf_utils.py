import fitz
import os


def get_pdf_page_count(path: str) -> int:
    """获取 PDF 文件页数"""
    doc = fitz.open(path)
    count = doc.page_count
    doc.close()
    return count


def is_valid_pdf(path: str) -> bool:
    """检查文件是否为有效 PDF"""
    if not os.path.isfile(path):
        return False
    try:
        doc = fitz.open(path)
        doc.close()
        return True
    except Exception:
        return False


def is_encrypted_pdf(path: str) -> bool:
    """检查 PDF 是否加密"""
    try:
        doc = fitz.open(path)
        encrypted = doc.is_encrypted
        doc.close()
        return encrypted
    except Exception:
        return False


def get_pdf_info(path: str) -> dict:
    """获取 PDF 基本信息"""
    doc = fitz.open(path)
    info = {
        "path": path,
        "filename": os.path.basename(path),
        "page_count": doc.page_count,
        "encrypted": doc.is_encrypted,
        "title": doc.metadata.get("title", ""),
        "author": doc.metadata.get("author", ""),
    }
    doc.close()
    return info


def get_pdf_bookmarks(path: str) -> list[dict]:
    """获取 PDF 书签（目录）结构"""
    doc = fitz.open(path)
    toc = doc.get_toc()
    doc.close()

    bookmarks = []
    for item in toc:
        level, title, page = item
        bookmarks.append({
            "level": level,
            "title": title,
            "page": page - 1,  # 转为 0-based
        })
    return bookmarks


def validate_page_range(page_text: str, total_pages: int) -> list[tuple[int, int]]:
    """
    解析页码输入字符串，返回 [(start, end), ...] 列表

    支持分隔符：英文逗号 `,`、中文逗号 `，`、空格（可混用）
    支持格式: "1-3,5,8-10" / "1-3 5 8-10" / "1-3，5，8-10"
    """
    if not page_text.strip():
        return []

    # 统一分隔符：把中文逗号和空格都替换为英文逗号
    normalized = page_text.replace("，", ",").replace(" ", ",")
    ranges = []
    parts = normalized.split(",")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str.strip())
            end = int(end_str.strip())
        else:
            start = int(part)
            end = start

        if start < 1 or end > total_pages or start > end:
            raise ValueError(f"无效页码范围: {part}（文档共 {total_pages} 页）")

        ranges.append((start - 1, end - 1))  # 转为 0-based
    return ranges