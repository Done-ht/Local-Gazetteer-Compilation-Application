"""本机目录浏览：列出指定目录下的子目录与受支持文件（PDF/图片）。

本机模式下，前端通过此模块浏览本机文件系统，选择 PDF/图片文件和输出目录，
避免走上传流程。复用 app.utils.natural_sort 保证 chapter2 排在 chapter10 之前。
"""
import glob
import os

from app.utils.natural_sort import natural_sort, natural_sort_key


# 与 compose_engine.SUPPORTED_IMAGE_EXTS 保持一致
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp"}


def _is_supported_file(name: str) -> bool:
    ext = os.path.splitext(name)[1].lower()
    return ext == ".pdf" or ext in SUPPORTED_IMAGE_EXTS


def list_dir(path: str) -> dict:
    """列出指定目录下的子目录与受支持文件。

    Returns:
        {
            "path": 当前绝对路径,
            "parent": 上一级目录（无可访问父目录则为 None）,
            "dirs": [子目录名...],          # 自然排序
            "files": [{name, path, ext, size, is_pdf, is_image}...],  # 自然排序
            "error": None 或错误信息
        }
    """
    result = {"path": None, "parent": None, "dirs": [], "files": [], "error": None}

    if not path:
        path = os.path.expanduser("~")

    try:
        path = os.path.abspath(path)
    except Exception as e:
        result["error"] = f"路径无效: {e}"
        return result

    result["path"] = path

    if not os.path.isdir(path):
        result["error"] = "路径不存在或不是目录"
        return result

    # 上一级目录（仅当可读时返回，供前端"返回上一级"按钮）
    parent = os.path.dirname(path)
    if parent and parent != path and os.access(parent, os.R_OK):
        result["parent"] = parent

    try:
        entries = os.listdir(path)
    except PermissionError:
        result["error"] = "无权限访问该目录"
        return result
    except Exception as e:
        result["error"] = f"读取目录失败: {e}"
        return result

    dirs = []
    files = []
    for name in entries:
        full = os.path.join(path, name)
        try:
            if os.path.isdir(full):
                dirs.append(name)
            elif os.path.isfile(full) and _is_supported_file(name):
                ext = os.path.splitext(name)[1].lower()
                try:
                    size = os.path.getsize(full)
                except OSError:
                    size = 0
                files.append({
                    "name": name,
                    "path": full,
                    "ext": ext,
                    "size": size,
                    "is_pdf": ext == ".pdf",
                    "is_image": ext in SUPPORTED_IMAGE_EXTS,
                })
        except OSError:
            # 遇到不可访问的符号链接等，跳过
            continue

    natural_sort(dirs)
    files.sort(key=lambda f: natural_sort_key(f["name"]))

    result["dirs"] = dirs
    result["files"] = files
    return result


def _resolve_known_dir(home: str, folder: str) -> str:
    """解析"桌面/文档/下载"等已知目录的真实路径，处理 OneDrive 重定向。

    探测顺序：~/OneDrive/<folder> → ~/OneDrive - <组织名>/<folder>（企业版）→ ~/<folder>。
    中文 Windows 资源管理器虽显示"桌面"，但真实目录名几乎总是 Desktop，故始终用英文目录名探测。
    返回首个存在且可读的路径；都不可用时返回空字符串。
    """
    candidates = [os.path.join(home, "OneDrive", folder)]
    # OneDrive 企业版：目录名形如 "OneDrive - 公司名"
    for od in glob.glob(os.path.join(home, "OneDrive -*")):
        candidates.append(os.path.join(od, folder))
    candidates.append(os.path.join(home, folder))
    for p in candidates:
        if os.path.isdir(p) and os.access(p, os.R_OK):
            return p
    return ""


def quick_paths() -> dict:
    """返回本机常用目录列表，供路径浏览器侧栏快速入口使用。

    跨平台探测主目录/桌面/文档/下载，过滤不存在或不可读的项。
    桌面/文档会优先检测 OneDrive 重定向（个人版与企业版），下载通常不被重定向。

    Returns:
        {"paths": [{"key","label","path","icon"}, ...]}
    """
    home = os.path.expanduser("~")
    result = []

    def add(key, label, path, icon):
        if path and os.path.isdir(path) and os.access(path, os.R_OK):
            result.append({"key": key, "label": label, "path": path, "icon": icon})

    add("home", "主目录", home, "🏠")
    add("desktop", "桌面", _resolve_known_dir(home, "Desktop"), "🖥️")
    add("documents", "文档", _resolve_known_dir(home, "Documents"), "📄")
    add("downloads", "下载", _resolve_known_dir(home, "Downloads"), "⬇️")

    return {"paths": result}
