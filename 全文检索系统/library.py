"""数据存储区（库）注册表管理。

每个"库"是一个独立的数据存储区根目录，拥有自己的 zone/索引/去重表。
库之间完全隔离，可以并行查询。

支持嵌套三层文件夹路径：folder 字段用 / 分隔，最多三层（如 "A/B/C"），
空字符串表示根级。

注册表存储在工作目录的 _libraries.json：
{
  "libraries": [
    {
      "id": "lib_001",
      "name": "郎溪县志",
      "note": "郎溪县志全文数据",
      "path": "datastore",
      "folder": "史料/正史",
      "created_at": "2026-07-23T..."
    },
    ...
  ],
  "_folders": ["史料", "史料/正史"]
}

本模块只负责库的元数据管理，不关心库内部的 zone/chunk 结构。
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict

from storage import ZoneManager


# 公共库属主标识：owner == "guest" 的库对所有游客可见可读写
PUBLIC_OWNER = "guest"


def is_public(owner: str) -> bool:
    """判断属主是否为公共库。"""
    return (owner or PUBLIC_OWNER) == PUBLIC_OWNER


@dataclass
class Library:
    id: str
    name: str
    note: str
    path: str           # 库根目录（绝对或相对工作目录）
    created_at: str
    folder: str = ""    # 所属文件夹路径（用 / 分隔，最多三层，空字符串=根级）
    owner: str = PUBLIC_OWNER  # 库属主：guest=公共库（所有人可读写）；否则为用户名

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Library":
        return cls(
            id=d.get("id", ""),
            name=d.get("name", ""),
            note=d.get("note", ""),
            path=d.get("path", ""),
            created_at=d.get("created_at", ""),
            folder=d.get("folder", ""),
            owner=d.get("owner") or PUBLIC_OWNER,
        )

    def visible_to(self, username: str, is_admin: bool = False) -> bool:
        """当前身份是否可见此库：公共库所有人可见；管理员可见全部。

        username: 当前用户名（游客传 PUBLIC_OWNER/"guest"）。
        """
        if is_admin:
            return True
        if is_public(self.owner):
            return True
        return self.owner == username

    def writable_by(self, username: str, is_admin: bool = False) -> bool:
        """当前身份是否可写此库：公共库所有人可写；管理员可写全部；其余仅属主。"""
        if is_admin:
            return True
        if is_public(self.owner):
            return True
        return self.owner == username

    def abs_path(self, base: str) -> str:
        """返回库的绝对路径。"""
        if os.path.isabs(self.path):
            return self.path
        return os.path.abspath(os.path.join(base, self.path))

    def manager(self, base: str) -> ZoneManager:
        """返回该库对应的 ZoneManager。"""
        return ZoneManager(self.abs_path(base))


class LibraryRegistry:
    """库注册表，管理 _libraries.json。"""

    def __init__(self, base_dir: str):
        self.base_dir = os.path.abspath(base_dir)
        self.registry_path = os.path.join(self.base_dir, "_libraries.json")
        self._data: Optional[Dict] = None

    # ---- 持久化 ----

    def _load(self) -> Dict:
        if self._data is not None:
            return self._data
        if not os.path.isfile(self.registry_path):
            self._data = {"libraries": []}
        else:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        return self._data

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        tmp = self.registry_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.registry_path)

    # ---- 查询 ----

    def list_libraries(self, owner: Optional[str] = None) -> List[Library]:
        """列出库。owner=None 返回全部；否则返回公共库 + 该用户名下的库。"""
        data = self._load()
        libs = [Library.from_dict(d) for d in data.get("libraries", [])]
        if owner is None:
            return libs
        return [l for l in libs if l.visible_to(owner, is_admin=False)]

    def list_libraries_for(self, username: str, is_admin: bool = False) -> List[Library]:
        """按身份列出可见库：管理员见全部，其余见公共库 + 自己的库。"""
        data = self._load()
        libs = [Library.from_dict(d) for d in data.get("libraries", [])]
        return [l for l in libs if l.visible_to(username, is_admin)]

    def get_library(self, name: str) -> Optional[Library]:
        """按名字查找库（全局首个匹配；CLI/管理用）。"""
        for lib in self.list_libraries():
            if lib.name == name:
                return lib
        return None

    def get_library_by_owner(self, name: str, owner: str) -> Optional[Library]:
        """按 (库名, 属主) 精确查找。"""
        owner = owner or PUBLIC_OWNER
        for lib in self.list_libraries():
            if lib.name == name and lib.owner == owner:
                return lib
        return None

    def get_library_for(self, name: str, username: str,
                        is_admin: bool = False) -> Optional[Library]:
        """按当前身份解析同名库：管理员取首个匹配；否则优先自己名下的，其次公共库。"""
        libs = [l for l in self.list_libraries() if l.name == name]
        if not libs:
            return None
        if is_admin:
            return libs[0]
        for lib in libs:
            if lib.owner == username:
                return lib
        for lib in libs:
            if is_public(lib.owner):
                return lib
        return None

    def _find(self, name: str, owner: Optional[str] = None) -> Optional[Library]:
        """查找库：owner 提供时按 (name, owner) 精确匹配，否则全局首名。"""
        if owner is None:
            return self.get_library(name)
        return self.get_library_by_owner(name, owner)

    def get_by_id(self, lib_id: str) -> Optional[Library]:
        for lib in self.list_libraries():
            if lib.id == lib_id:
                return lib
        return None

    def _next_id(self) -> str:
        libs = self.list_libraries()
        if not libs:
            return "lib_001"
        nums = []
        for lib in libs:
            m = re.search(r"(\d+)$", lib.id)
            if m:
                nums.append(int(m.group(1)))
        n = (max(nums) + 1) if nums else 1
        return f"lib_{n:03d}"

    # ---- 增删改 ----

    def create(self, name: str, note: str = "", path: Optional[str] = None,
               owner: str = PUBLIC_OWNER) -> Library:
        """创建一个新库。

        name: 库名（在该属主范围内唯一）
        note: 备注
        path: 库根目录；不指定则默认 <base_dir>/libraries/<owner>/<name_slug>
              （按属主隔离目录：guest=公共库放 libraries/guest/，用户放 libraries/<用户名>/；
                旧库 path 已存于注册表，不受影响）
        owner: 库属主（guest=公共库，所有游客公用；否则为用户名）
        """
        owner = owner or PUBLIC_OWNER
        if self.get_library_by_owner(name, owner) is not None:
            hint = "公共库名已存在" if owner == PUBLIC_OWNER else f"你已存在同名库"
            raise ValueError(f"{hint}: {name}")

        lib_id = self._next_id()
        if path is None:
            slug = _slugify(name)
            path = f"libraries/{owner}/{slug}"  # 统一用正斜杠，跨平台一致
            # 同属主内目录冲突（如 "A B" 与 "A_B" slug 相同）：追加序号
            base = path
            i = 2
            while os.path.exists(os.path.join(self.base_dir, path)):
                path = f"{base}_{i}"
                i += 1
        lib = Library(
            id=lib_id,
            name=name,
            note=note,
            path=path,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            owner=owner,
        )
        # 确保库目录存在
        abs_path = lib.abs_path(self.base_dir)
        os.makedirs(abs_path, exist_ok=True)
        ZoneManager(abs_path)  # 初始化目录结构

        data = self._load()
        data.setdefault("libraries", []).append(lib.to_dict())
        self._save()
        return lib

    def remove(self, name: str, delete_data: bool = True,
               owner: Optional[str] = None) -> Library:
        """删除库。

        delete_data=True 同时删除库目录下的所有数据文件。
        owner 指定时按 (name, owner) 精确定位（同名不同属主互不影响）。
        返回被删除的 Library。
        """
        lib = self._find(name, owner)
        if lib is None:
            raise ValueError(f"库不存在: {name}")

        if delete_data:
            abs_path = lib.abs_path(self.base_dir)
            if os.path.isdir(abs_path):
                import shutil
                shutil.rmtree(abs_path)

        data = self._load()
        data["libraries"] = [
            d for d in data.get("libraries", []) if d.get("id") != lib.id
        ]
        self._save()
        return lib

    def update_note(self, name: str, note: str,
                    owner: Optional[str] = None) -> Library:
        """更新库的备注。owner 指定时按 (name, owner) 精确定位。"""
        lib = self._find(name, owner)
        if lib is None:
            raise ValueError(f"库不存在: {name}")
        data = self._load()
        for d in data.get("libraries", []):
            if d.get("id") == lib.id:
                d["note"] = note
                break
        self._save()
        lib.note = note
        return lib

    # ---- 文件夹管理（支持嵌套三层路径）----

    def list_folders(self) -> List[str]:
        """列出所有文件夹路径（含中间路径，按字母序）。

        例如库归属 "A/B"，则返回 ["A", "A/B"]。
        这样前端可以直接根据路径构建树形结构。
        """
        data = self._load()
        folders = set()
        for f in data.get("_folders", []):
            if f:
                # 加入路径本身及其所有父路径
                parts = f.split("/")
                for i in range(1, len(parts) + 1):
                    folders.add("/".join(parts[:i]))
        for d in data.get("libraries", []):
            f = d.get("folder", "")
            if f:
                parts = f.split("/")
                for i in range(1, len(parts) + 1):
                    folders.add("/".join(parts[:i]))
        return sorted(folders)

    def create_folder(self, name: str) -> str:
        """创建文件夹（逻辑分组，支持嵌套三层路径）。"""
        name = _validate_folder_path(name)
        if not name:
            raise ValueError("文件夹名不能为空")
        data = self._load()
        folders = data.setdefault("_folders", [])
        if name not in folders:
            folders.append(name)
            self._save()
        return name

    def delete_folder(self, name: str) -> dict:
        """删除文件夹：把该文件夹下（含子文件夹）的库移到根级，并移除该文件夹及子文件夹。"""
        name = _validate_folder_path(name)
        if not name:
            raise ValueError("文件夹名不能为空")
        data = self._load()
        moved = 0
        prefix = name + "/"
        for d in data.get("libraries", []):
            f = d.get("folder", "")
            if f == name or f.startswith(prefix):
                d["folder"] = ""
                moved += 1
        # 移除 _folders 中该文件夹及其子文件夹
        folders = data.get("_folders", [])
        data["_folders"] = [f for f in folders if f != name and not f.startswith(prefix)]
        self._save()
        return {"deleted": name, "moved_to_root": moved}

    def rename_folder(self, old: str, new: str) -> dict:
        """重命名文件夹：更新该文件夹及其子文件夹下所有库的归属。"""
        old = _validate_folder_path(old)
        new = _validate_folder_path(new)
        if not old:
            raise ValueError("原文件夹名不能为空")
        if not new:
            raise ValueError("新文件夹名不能为空")
        if old == new:
            return {"renamed": old, "affected": 0}
        data = self._load()
        # 检查新名是否与已有文件夹冲突（完全匹配，不冲突子路径）
        existing = set(data.get("_folders", []))
        for d in data.get("libraries", []):
            existing.add(d.get("folder", ""))
        if new in existing:
            raise ValueError(f"文件夹路径已存在: {new}")
        prefix = old + "/"
        affected = 0
        for d in data.get("libraries", []):
            f = d.get("folder", "")
            if f == old:
                d["folder"] = new
                affected += 1
            elif f.startswith(prefix):
                # 子文件夹：替换前缀
                d["folder"] = new + "/" + f[len(prefix):]
                affected += 1
        folders = data.get("_folders", [])
        new_folders = []
        for f in folders:
            if f == old:
                new_folders.append(new)
            elif f.startswith(prefix):
                new_folders.append(new + "/" + f[len(prefix):])
            else:
                new_folders.append(f)
        data["_folders"] = new_folders
        self._save()
        return {"old": old, "new": new, "affected": affected}

    def move_library(self, lib_name: str, folder: str,
                     owner: Optional[str] = None) -> Library:
        """移动库到指定文件夹路径。空字符串表示移到根级。owner 指定时精确定位。"""
        lib = self._find(lib_name, owner)
        if lib is None:
            raise ValueError(f"库不存在: {lib_name}")
        folder = _validate_folder_path(folder)
        data = self._load()
        for d in data.get("libraries", []):
            if d.get("id") == lib.id:
                d["folder"] = folder
                break
        self._save()
        lib.folder = folder
        return lib

    def move_folder(self, old_path: str, new_parent: str) -> dict:
        """把一个文件夹（及其所有子文件夹和库）移动到另一个文件夹下。

        old_path: 要移动的文件夹路径（如 "A/B"）
        new_parent: 目标父文件夹路径（如 "X"）；空字符串表示移到根级

        返回 {old, new, affected_libs, affected_subfolders}

        规则：
        - 不能移动到自身下（如 "A" 不能移到 "A/B" 下）
        - 不能移动到自己的子文件夹下（如 "A/B" 不能移到 "A/B/C" 下，会形成环）
        - 新路径深度不能超过 3 层
        - 移动后：old_path 变为 new_parent + "/" + old_path 的末段
          子文件夹路径同步更新前缀
        """
        old_path = _validate_folder_path(old_path)
        new_parent = _validate_folder_path(new_parent)
        if not old_path:
            raise ValueError("要移动的文件夹路径不能为空")

        # 计算新路径
        old_last = old_path.split("/")[-1]
        new_path = (new_parent + "/" + old_last) if new_parent else old_last

        if old_path == new_path:
            return {"old": old_path, "new": new_path, "affected_libs": 0, "affected_subfolders": 0}

        # 检查环：不能移动到自身或自己的子文件夹下
        if new_parent == old_path or new_parent.startswith(old_path + "/"):
            raise ValueError("不能把文件夹移动到自身或其子文件夹下")

        # 检查新路径深度不超过3层
        new_depth = new_path.split("/")
        if len(new_depth) > 3:
            raise ValueError("移动后文件夹路径深度不能超过 3 层")

        # 检查新路径是否与已有文件夹冲突
        data = self._load()
        existing = set(data.get("_folders", []))
        for d in data.get("libraries", []):
            existing.add(d.get("folder", ""))
        if new_path in existing:
            raise ValueError(f"目标路径已存在文件夹: {new_path}")

        prefix = old_path + "/"
        affected_libs = 0
        affected_subfolders = 0

        # 更新库的 folder 字段
        for d in data.get("libraries", []):
            f = d.get("folder", "")
            if f == old_path:
                d["folder"] = new_path
                affected_libs += 1
            elif f.startswith(prefix):
                d["folder"] = new_path + "/" + f[len(prefix):]
                affected_libs += 1

        # 更新 _folders 列表
        folders = data.get("_folders", [])
        new_folders = []
        for f in folders:
            if f == old_path:
                new_folders.append(new_path)
                affected_subfolders += 1
            elif f.startswith(prefix):
                new_folders.append(new_path + "/" + f[len(prefix):])
                affected_subfolders += 1
            else:
                new_folders.append(f)
        data["_folders"] = new_folders

        self._save()
        return {"old": old_path, "new": new_path,
                "affected_libs": affected_libs, "affected_subfolders": affected_subfolders}

    def list_subfolders(self, parent: str = "") -> List[str]:
        """返回指定父文件夹下的直接子文件夹的完整路径。

        parent="" 返回根级下的子文件夹（一层路径）。
        parent="A" 返回 "A" 下的子文件夹（如 "A/B"）。
        只返回直接子级，不递归。
        """
        parent = _validate_folder_path(parent)
        all_folders = self.list_folders()
        result = []
        for f in all_folders:
            if not parent:
                # 根级：只返回一层路径
                if "/" not in f:
                    result.append(f)
            else:
                # 指定父级下：返回 "parent/X" 形式的直接子级
                if f.startswith(parent + "/"):
                    rest = f[len(parent) + 1:]
                    if "/" not in rest:
                        result.append(f)
        return sorted(result)

    def list_libraries_in_folder(self, folder: str = "", recursive: bool = False) -> List[Library]:
        """返回指定文件夹下的库。

        folder="" 返回根级库（folder 为空的库）。
        recursive=True 时返回该文件夹及其所有子文件夹下的库。
        recursive=False 时只返回直接归属该文件夹的库。
        """
        folder = _validate_folder_path(folder)
        libs = self.list_libraries()
        if not folder:
            # 根级：folder 为空的库
            return [l for l in libs if not l.folder]
        if recursive:
            prefix = folder + "/"
            return [l for l in libs if l.folder == folder or l.folder.startswith(prefix)]
        return [l for l in libs if l.folder == folder]

    def register_existing(self, name: str, note: str, path: str,
                          owner: str = PUBLIC_OWNER) -> Library:
        """把一个已有数据的目录注册为库（不创建新目录，不删数据）。

        用于迁移/接管已有 datastore。owner 指定库归属。
        """
        if self.get_library(name) is not None:
            raise ValueError(f"库名已存在: {name}")
        abs_path = path if os.path.isabs(path) else os.path.abspath(
            os.path.join(self.base_dir, path)
        )
        if not os.path.isdir(abs_path):
            raise ValueError(f"目录不存在: {abs_path}")
        lib_id = self._next_id()
        lib = Library(
            id=lib_id,
            name=name,
            note=note,
            path=path,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            owner=owner or PUBLIC_OWNER,
        )
        data = self._load()
        data.setdefault("libraries", []).append(lib.to_dict())
        self._save()
        return lib

    # ---- 多用户：所有权转移与库复制（数据迁移）----

    def set_owner(self, name: str, owner: str,
                  from_owner: Optional[str] = None) -> Library:
        """转移库所有权（数据迁移）：把库归给指定用户或设为公共库。

        仅修改注册表元数据，不移动数据目录。
        目标属主下若已存在同名库则报错（同名不同属主隔离）。
        from_owner 指定源属主时按 (name, from_owner) 精确定位。
        """
        lib = self._find(name, from_owner)
        if lib is None:
            raise ValueError(f"库不存在: {name}")
        owner = (owner or PUBLIC_OWNER).strip()
        if not owner:
            owner = PUBLIC_OWNER
        if owner != lib.owner and self.get_library_by_owner(name, owner) is not None:
            raise ValueError(f"目标属主下已存在同名库: {name}")
        data = self._load()
        for d in data.get("libraries", []):
            if d.get("id") == lib.id:
                d["owner"] = owner
                break
        self._save()
        lib.owner = owner
        return lib

    def clone_library(self, name: str, to_owner: str,
                      new_name: Optional[str] = None, note: str = "",
                      from_owner: Optional[str] = None) -> Library:
        """把库复制一份并归属到 to_owner 名下（"把公共库添加到自己的库"）。

        深拷贝整个库数据目录（含 zone/chunk/索引/语义索引），原库不受影响。
        新库注册表 owner = to_owner。

        Args:
            name: 源库名
            to_owner: 新库属主（用户名或 guest）
            new_name: 新库名；不指定则自动生成 "<源名>（副本）" 或追加序号
            note: 新库备注
            from_owner: 源库属主；提供时按 (name, from_owner) 精确定位，
                        避免同名库存在时解析到错误来源（隐私安全）
        """
        import shutil
        import uuid
        src = self._find(name, from_owner)
        if src is None:
            raise ValueError(f"库不存在: {name}")
        to_owner = (to_owner or PUBLIC_OWNER).strip()
        if not to_owner:
            to_owner = PUBLIC_OWNER
        # 确定新库名（在目标属主范围内避免重名；同名不同属主互不影响）
        existing = {l.name for l in self.list_libraries() if l.owner == to_owner}
        if not new_name or new_name in existing:
            base = new_name.strip() if new_name else f"{src.name}（副本）"
            new_name = base
            i = 2
            while new_name in existing:
                new_name = f"{base}{i}"
                i += 1
        # 新库目录：libraries/<目标属主>/<源slug>__clone_<短随机>（按属主隔离）
        src_abs = src.abs_path(self.base_dir)
        if not os.path.isdir(src_abs):
            raise ValueError(f"源库数据目录不存在: {src_abs}")
        dst_slug = _slugify(src.name) + "__clone_" + uuid.uuid4().hex[:8]
        dst_path = f"libraries/{to_owner}/{dst_slug}"
        dst_abs = os.path.join(self.base_dir, dst_path)
        try:
            shutil.copytree(src_abs, dst_abs)
        except OSError as e:
            raise ValueError(f"复制库数据失败: {e}")
        # 复制会带上源库的 _semantic 索引（faiss/chunk_ids 等基于库根的相对路径），
        # 新库语义索引直接可用。若源库 _dedup_index.json 存在，一并复制即可。
        lib_id = self._next_id()
        lib = Library(
            id=lib_id,
            name=new_name,
            note=note or src.note,
            path=dst_path,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            folder=src.folder,
            owner=to_owner,
        )
        data = self._load()
        data.setdefault("libraries", []).append(lib.to_dict())
        self._save()
        return lib

    def list_owners_usage(self) -> Dict[str, int]:
        """统计每个属主（含 guest 公共库）名下的库数量。"""
        from collections import Counter
        libs = self.list_libraries()
        counter = Counter(l.owner for l in libs)
        return dict(sorted(counter.items()))


def _slugify(name: str) -> str:
    """把名字转成安全的目录名。"""
    # 保留中文/字母/数字，其余替换为下划线
    slug = re.sub(r"[^\w\u4e00-\u9fff]+", "_", name.strip())
    return slug or "library"


def _validate_folder_path(folder: str) -> str:
    """校验文件夹路径并返回规范化后的路径。

    - 空字符串表示根级，合法
    - 用 / 分隔，最多三层
    - 每段不能为空、不能含 / 或 \\
    - 自动去除首尾空白和多余的 /
    """
    folder = (folder or "").strip()
    if not folder:
        return ""
    # 统一路径分隔符
    folder = folder.replace("\\", "/")
    # 去除首尾 /
    folder = folder.strip("/")
    if not folder:
        return ""
    parts = folder.split("/")
    if len(parts) > 3:
        raise ValueError("文件夹路径深度不能超过 3 层")
    cleaned = []
    for p in parts:
        p = p.strip()
        if not p:
            raise ValueError("文件夹路径中有空段")
        if "/" in p or "\\" in p:
            raise ValueError("文件夹名不能包含 / 或 \\")
        cleaned.append(p)
    return "/".join(cleaned)
