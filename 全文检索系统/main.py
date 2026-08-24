#!/usr/bin/env python
"""全文检索数据存储系统 - 主程序入口。

支持多个数据存储区（库），可并行跨库检索。

CLI 子命令：
    library create <name> --note <备注>   创建库
    library list                          列出所有库
    library remove <name>                 删除库（含数据）
    library notes <name> <备注>           修改库备注

    import <文件/目录> --library <库名>   导入文件到指定库
    search "关键词" --libraries A B --parallel 4   跨库并行检索
    verify --library <库名>               校验指定库
    recover --library <库名>              恢复残留事务
    remove --library <库名> --ext txt     删除库内指定类型文件
    stats [--library <库名>]              查看统计（不指定则列出全部库）
    build-index --library <库名>          重建索引
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from storage import ZoneManager
from library import LibraryRegistry, Library
from extractor import supported, SUPPORTED_EXTS
from userdata import auth_base_dir as _auth_base_dir


# 工作目录（注册表 _libraries.json 所在位置）= 代码目录下的 data/
# 库数据统一放在 data/ 下，与代码分离；用户登录数据存 <用户主目录>/biaoshifu（见 userdata.py）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(_SCRIPT_DIR, "data")
os.makedirs(BASE_DIR, exist_ok=True)
# 用户登录相关数据目录（<用户主目录>/biaoshifu，跨应用共用同一组账号）
AUTH_DIR = _auth_base_dir()


def _registry() -> LibraryRegistry:
    return LibraryRegistry(BASE_DIR)


def _get_lib_or_exit(reg: LibraryRegistry, name: str) -> Library:
    lib = reg.get_library(name)
    if lib is None:
        print(f"[错误] 库不存在: {name}")
        sys.exit(1)
    return lib


def _print_json(data) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


# ============================================================
#  library 管理
# ============================================================

def cmd_library(args) -> int:
    reg = _registry()
    action = args.action

    if action == "create":
        lib = reg.create(args.name, note=args.note or "", owner=getattr(args, "owner", None) or "guest")
        print(f"[创建库] {lib.name} (id={lib.id})")
        print(f"  备注: {lib.note or '(无)'}")
        print(f"  属主: {lib.owner}（guest=公共库，所有游客公用）")
        print(f"  路径: {lib.abs_path(BASE_DIR)}")
        return 0

    if action == "list":
        libs = reg.list_libraries()
        if not libs:
            print("[库列表] 尚无库，用 'library create <name>' 创建")
            return 0
        print(f"[库列表] 共 {len(libs)} 个库:")
        for lib in libs:
            mgr = lib.manager(BASE_DIR)
            s = mgr.stats()
            print(f"  {lib.name} (id={lib.id})  属主: {lib.owner}")
            print(f"    备注: {lib.note or '(无)'}")
            print(f"    路径: {lib.path}")
            print(f"    字符数: {s['total_chars']:,}, "
                  f"chunk: {s['total_chunks']}, "
                  f"源文件: {s['total_sources']}, "
                  f"zone: {s['zone_count']}")
        return 0

    if action == "remove":
        if not args.yes:
            print(f"[删除库] 确认删除库 '{args.name}' 及其所有数据？(y/N)", end=" ")
            try:
                ans = input().strip().lower()
            except EOFError:
                ans = "n"
            if ans not in ("y", "yes"):
                print("[删除库] 已取消")
                return 0
        lib = reg.remove(args.name, delete_data=True)
        print(f"[删除库] 已删除: {lib.name} (路径: {lib.path})")
        return 0

    if action == "notes":
        lib = reg.update_note(args.name, args.note)
        print(f"[修改备注] {lib.name}: {lib.note}")
        return 0

    if action == "transfer":
        # 数据迁移：把库所有权转移给指定用户（或设为公共库 guest）
        lib = reg.set_owner(args.name, args.owner)
        print(f"[转移所有权] {lib.name} -> {lib.owner}")
        return 0

    if action == "clone":
        # 数据迁移：复制库并归属到指定用户/公共库
        try:
            lib = reg.clone_library(args.name, to_owner=args.to,
                                    new_name=getattr(args, "name", None) or None)
        except ValueError as e:
            print(f"[错误] {e}")
            return 1
        print(f"[复制库] {args.name} -> {lib.name} (属主: {lib.owner})")
        print(f"  路径: {lib.path}")
        return 0

    print(f"[错误] 未知 library 操作: {action}")
    return 1


# ============================================================
#  用户管理（多用户账号体系）
# ============================================================

def cmd_user(args) -> int:
    from auth import UserStore
    store = UserStore(AUTH_DIR)
    action = args.action

    if action == "create":
        import getpass
        password = args.password
        if password is None:
            try:
                password = getpass.getpass("密码（至少 6 位）> ")
            except (EOFError, KeyboardInterrupt):
                print("\n[取消]")
                return 1
        try:
            # 首个用户自动为管理员；--admin 可显式指定管理员
            u = store.register(args.name, password,
                               role="admin" if args.admin else None)
        except ValueError as e:
            print(f"[错误] {e}")
            return 1
        print(f"[创建用户] {u['username']} 角色={u['role']}"
              + ("（首个用户，自动为管理员）" if u["role"] == "admin" and not args.admin else ""))
        return 0

    if action == "list":
        users = store.list_users()
        if not users:
            print("[用户列表] 暂无用户，用 'user create <name>' 创建（首个用户自动为管理员）")
            return 0
        usage = _registry().list_owners_usage()
        print(f"[用户列表] 共 {len(users)} 个用户:")
        for u in users:
            print(f"  {u['username']}  角色: {u['role']}  创建: {u['created_at']}"
                  f"  名下库数: {usage.get(u['username'], 0)}")
        print(f"  公共库（游客公用）: {usage.get('guest', 0)} 个")
        return 0

    if action == "remove":
        if not args.yes:
            print(f"[删除用户] 确认删除用户 '{args.name}'？（其名下库不会被删除，可先用 '"
                  f"library transfer {args.name} <新属主>' 转移）(y/N)", end=" ")
            try:
                ans = input().strip().lower()
            except EOFError:
                ans = "n"
            if ans not in ("y", "yes"):
                print("[删除用户] 已取消")
                return 0
        ok = store.remove_user(args.name)
        if not ok:
            print(f"[错误] 用户不存在: {args.name}")
            return 1
        print(f"[删除用户] 已删除: {args.name}")
        return 0

    if action == "set-role":
        u = store.set_role(args.name, args.role)
        if u is None:
            print(f"[错误] 用户不存在: {args.name}")
            return 1
        print(f"[设置角色] {u['username']} -> {u['role']}")
        return 0

    if action == "password":
        import getpass
        old = getpass.getpass("旧密码 > ")
        new = getpass.getpass("新密码（至少 6 位）> ")
        try:
            ok = store.change_password(args.name, old, new)
        except ValueError as e:
            print(f"[错误] {e}")
            return 1
        if not ok:
            print("[错误] 旧密码错误或用户不存在")
            return 1
        print(f"[修改密码] {args.name} 已更新")
        return 0

    print(f"[错误] 未知 user 操作: {action}")
    return 1


# ============================================================
#  import
# ============================================================

def cmd_import(args) -> int:
    from importer import import_file
    from transaction import recover_all_zones

    reg = _registry()
    lib = _get_lib_or_exit(reg, args.library)
    mgr = lib.manager(BASE_DIR)

    recovered = recover_all_zones(mgr)
    if recovered:
        print(f"[恢复] 检测到 {len(recovered)} 个残留事务，已处理：")
        for zid, res in recovered:
            print(f"  {zid}: {res}")

    raw_inputs = args.files
    base_dir = os.getcwd()
    files: list[str] = []
    file_import_roots: dict[str, str] = {}  # file -> import_root
    for p in raw_inputs:
        if os.path.isdir(p):
            for root, _dirs, names in os.walk(p):
                for n in sorted(names):
                    full = os.path.join(root, n)
                    if supported(full):
                        files.append(full)
                        file_import_roots[full] = p
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"[失败] 路径不存在: {p}")

    if not files:
        print("[导入] 未找到可导入的文件")
        return 0

    print(f"[导入] 库: {lib.name} | 共发现 {len(files)} 个可导入文件\n")

    ok_count = 0
    skip_count = 0
    fail_count = 0

    for f in files:
        if not os.path.isfile(f):
            print(f"[失败] 文件不存在: {f}")
            fail_count += 1
            continue
        if not supported(f):
            print(f"[失败] 不支持的文件类型: {f} (支持: {sorted(SUPPORTED_EXTS)})")
            fail_count += 1
            continue

        print(f"[导入] {f} ...")
        result = import_file(
            mgr, f,
            force=args.force,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            base_dir=base_dir,
            import_root=file_import_roots.get(f),
        )
        if result.get("ok"):
            ok_count += 1
            print(f"  -> 成功: zone={result['zone_id']} "
                  f"chunks={result['chunks_written']} "
                  f"chars={result['char_count']}")
        elif result.get("skipped"):
            skip_count += 1
            ex = result.get("existing", {})
            print(f"  -> 跳过（重复导入）: 已存在于 {ex.get('zone_id','?')}")
        else:
            fail_count += 1
            print(f"  -> 失败: {result.get('error', '未知错误')}")

    print(f"\n[汇总] 成功={ok_count} 跳过={skip_count} 失败={fail_count}")
    return 0 if fail_count == 0 else 1


# ============================================================
#  search（跨库并行）
# ============================================================

def cmd_search(args) -> int:
    from searcher import parallel_search, format_search_result

    reg = _registry()
    libs = reg.list_libraries()
    if not libs:
        print("[搜索] 尚无库，请先创建库并导入数据")
        return 0

    library_names = args.libraries if args.libraries else None
    parallel = args.parallel if args.parallel else 4

    try:
        result = parallel_search(
            reg, args.query,
            library_names=library_names,
            parallel=parallel,
            base_dir=BASE_DIR,
        )
    except ValueError as e:
        print(f"[错误] {e}")
        return 1

    if args.json:
        _print_json(result)
        return 0

    print(format_search_result(result, top_k=args.top))
    return 0


# ============================================================
#  verify
# ============================================================

def cmd_verify(args) -> int:
    from verifier import verify_all

    reg = _registry()
    lib = _get_lib_or_exit(reg, args.library)
    mgr = lib.manager(BASE_DIR)

    source_files = args.source if args.source else None
    results = verify_all(mgr, source_files)

    if not results:
        print(f"[校验] 库 {lib.name} 为空")
        return 0

    print(f"[校验] 库: {lib.name}")
    all_ok = True
    for r in results:
        chunk_ok = r["chunk_ok"]
        chunk_total = r["chunk_total"]
        cont_ok = r["continuity_ok"]
        src_ok = r["source_ok"]
        zone_ok = chunk_ok == chunk_total and cont_ok and src_ok
        status = "通过" if zone_ok else "失败"
        print(f"  [{status}] {r['zone_id']}: chunks {chunk_ok}/{chunk_total}, "
              f"连续性={'OK' if cont_ok else '异常'}, "
              f"源文件={'OK' if src_ok else '异常'}")
        if r["chunk_bad"]:
            for bad in r["chunk_bad"]:
                print(f"      坏块: {bad['chunk']} - {bad['reason']}")
        if r["continuity_issues"]:
            for issue in r["continuity_issues"]:
                print(f"      连续性: {issue}")
        if r["source_issues"]:
            for issue in r["source_issues"]:
                print(f"      源文件: {issue}")
        if not zone_ok:
            all_ok = False

    return 0 if all_ok else 1


# ============================================================
#  recover
# ============================================================

def cmd_recover(args) -> int:
    from transaction import recover_all_zones

    reg = _registry()
    lib = _get_lib_or_exit(reg, args.library)
    mgr = lib.manager(BASE_DIR)
    results = recover_all_zones(mgr)

    if not results:
        print(f"[恢复] 库 {lib.name}: 无残留事务")
        return 0

    print(f"[恢复] 库 {lib.name}: 处理了 {len(results)} 个残留事务：")
    for zid, res in results:
        print(f"  {zid}: {res}")
    return 0


# ============================================================
#  remove（删除库内文件）
# ============================================================

def cmd_remove(args) -> int:
    from remover import remove_by_ext, remove_by_sha

    reg = _registry()
    lib = _get_lib_or_exit(reg, args.library)
    mgr = lib.manager(BASE_DIR)

    if args.ext:
        result = remove_by_ext(mgr, args.ext)
        if result["removed_chunks"] == 0:
            print(f"[删除] 库 {lib.name}: 未找到扩展名为 {result['ext']} 的数据")
            return 0
        print(f"[删除] 库 {lib.name} | 按扩展名 {result['ext']}:")
        print(f"  删除 chunk 数: {result['removed_chunks']}")
        print(f"  删除源文件数: {result['removed_sources']}")
        print(f"  删除字符数: {result['removed_chars']:,}")
        print(f"  受影响 zone: {', '.join(result['zones_affected']) or '无'}")
        print(f"  已重建索引")
        return 0

    if args.sha:
        result = remove_by_sha(mgr, args.sha)
        if result["removed_chunks"] == 0:
            print(f"[删除] 库 {lib.name}: 未找到 SHA256 为 {args.sha} 的数据")
            return 0
        print(f"[删除] 库 {lib.name} | 按 SHA256 {args.sha}:")
        print(f"  删除 chunk 数: {result['removed_chunks']}")
        print(f"  删除字符数: {result['removed_chars']:,}")
        print(f"  已重建索引")
        return 0

    print("[删除] 请指定 --ext 或 --sha")
    return 1


# ============================================================
#  stats
# ============================================================

def cmd_stats(args) -> int:
    reg = _registry()
    libs = reg.list_libraries()
    if not libs:
        print("[统计] 尚无库")
        return 0

    if args.library:
        lib = _get_lib_or_exit(reg, args.library)
        libs_to_show = [lib]
    else:
        libs_to_show = libs

    for lib in libs_to_show:
        mgr = lib.manager(BASE_DIR)
        s = mgr.stats()
        print(f"=== 库: {lib.name} ({lib.note or '无备注'}) ===")
        print(f"  路径: {lib.abs_path(BASE_DIR)}")
        print(f"  Zone 数: {s['zone_count']}")
        print(f"  总字符数: {s['total_chars']:,}")
        print(f"  总 chunk 数: {s['total_chunks']}")
        print(f"  总源文件数: {s['total_sources']}")
        if s["zones"]:
            for z in s["zones"]:
                print(f"    {z['zone_id']}: chars={z['char_count']:,} "
                      f"chunks={z['chunk_count']} sources={z['source_count']} "
                      f"remaining={z['remaining']:,}")
        from dedup import DedupIndex
        dedup = DedupIndex(mgr.root)
        print(f"  去重索引: {dedup.count()} 条记录")
        print()
    return 0


# ============================================================
#  build-index
# ============================================================

def cmd_build_index(args) -> int:
    from indexer import ZoneIndex

    reg = _registry()
    lib = _get_lib_or_exit(reg, args.library)
    mgr = lib.manager(BASE_DIR)
    zones = mgr.list_zones()
    if not zones:
        print(f"[建索引] 库 {lib.name} 为空")
        return 0

    print(f"[建索引] 库: {lib.name}")
    for z in zones:
        zi = ZoneIndex(z.index_dir)
        stat = zi.merge_zone_chunks(z.chunks_dir, z.zone_id)
        cleaned = zi.cleanup_merged_idx(z.chunks_dir)
        print(f"  {z.zone_id}: 新合并 {stat['merged']} 个, "
              f"跳过 {stat['skipped']} 个, 清理 {cleaned} 个 .idx")
    print("[建索引] 完成")
    return 0


# ============================================================
#  argparse
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="search",
        description="全文检索数据存储系统（多库 + 并行检索）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ---- library ----
    plib = sub.add_parser("library", help="库管理（create/list/remove/notes）")
    lib_sub = plib.add_subparsers(dest="action", required=True)

    lc = lib_sub.add_parser("create", help="创建库")
    lc.add_argument("name", help="库名（唯一）")
    lc.add_argument("--note", default="", help="库备注")
    lc.set_defaults(func=cmd_library)

    ll = lib_sub.add_parser("list", help="列出所有库")
    ll.set_defaults(func=cmd_library)

    lr = lib_sub.add_parser("remove", help="删除库（含数据）")
    lr.add_argument("name", help="库名")
    lr.add_argument("-y", "--yes", action="store_true", help="跳过确认")
    lr.set_defaults(func=cmd_library)

    ln = lib_sub.add_parser("notes", help="修改库备注")
    ln.add_argument("name", help="库名")
    ln.add_argument("note", help="新备注")
    ln.set_defaults(func=cmd_library)

    ltr = lib_sub.add_parser("transfer", help="转移库所有权（数据迁移，仅改注册表）")
    ltr.add_argument("name", help="库名")
    ltr.add_argument("owner", help="新属主用户名（guest=设为公共库，所有游客公用）")
    ltr.set_defaults(func=cmd_library)

    lcl = lib_sub.add_parser("clone", help="复制库并归属到指定用户（数据迁移，深拷贝）")
    lcl.add_argument("name", help="源库名")
    lcl.add_argument("--to", required=True, help="新库属主（guest=公共库）")
    lcl.add_argument("--name", default=None, help="新库名（不指定自动生成）")
    lcl.set_defaults(func=cmd_library)

    # ---- user ----
    pu = sub.add_parser("user", help="用户管理（create/list/remove/set-role/password）")
    user_sub = pu.add_subparsers(dest="action", required=True)

    uc = user_sub.add_parser("create", help="创建用户（首个用户自动为管理员）")
    uc.add_argument("name", help="用户名（字母/数字/下划线/中文）")
    uc.add_argument("--password", default=None, help="密码（不指定则交互输入）")
    uc.add_argument("--admin", action="store_true", help="显式设为管理员")
    uc.set_defaults(func=cmd_user)

    ul = user_sub.add_parser("list", help="列出所有用户及名下库数")
    ul.set_defaults(func=cmd_user)

    urm = user_sub.add_parser("remove", help="删除用户（不影响其名下库）")
    urm.add_argument("name", help="用户名")
    urm.add_argument("-y", "--yes", action="store_true", help="跳过确认")
    urm.set_defaults(func=cmd_user)

    usr = user_sub.add_parser("set-role", help="设置用户角色（admin/user）")
    usr.add_argument("name", help="用户名")
    usr.add_argument("role", choices=["admin", "user"], help="目标角色")
    usr.set_defaults(func=cmd_user)

    upw = user_sub.add_parser("password", help="修改密码")
    upw.add_argument("name", help="用户名")
    upw.set_defaults(func=cmd_user)

    # ---- import ----
    pi = sub.add_parser("import", help="导入文件到指定库")
    pi.add_argument("files", nargs="+", help="文件/目录路径（目录递归扫描）")
    pi.add_argument("--library", required=True, help="目标库名")
    pi.add_argument("--force", action="store_true", help="绕过去重检查")
    pi.add_argument("--chunk-size", type=int, default=10000, help="每块字符数")
    pi.add_argument("--overlap", type=int, default=20, help="块间重叠字符数")
    pi.set_defaults(func=cmd_import)

    # ---- search ----
    ps = sub.add_parser("search", help="跨库并行检索")
    ps.add_argument("query", help="搜索关键词")
    ps.add_argument("--libraries", nargs="+", help="指定查询的库名（可多个，默认全部）")
    ps.add_argument("--parallel", type=int, default=4, help="并行度（默认 4）")
    ps.add_argument("--top", type=int, default=20, help="显示前 N 条命中")
    ps.add_argument("--json", action="store_true", help="输出 JSON 格式")
    ps.set_defaults(func=cmd_search)

    # ---- verify ----
    pv = sub.add_parser("verify", help="校验指定库")
    pv.add_argument("--library", required=True, help="库名")
    pv.add_argument("--source", nargs="+", help="源文件路径，用于源文件级校验")
    pv.set_defaults(func=cmd_verify)

    # ---- recover ----
    pr = sub.add_parser("recover", help="恢复残留事务")
    pr.add_argument("--library", required=True, help="库名")
    pr.set_defaults(func=cmd_recover)

    # ---- remove ----
    prm = sub.add_parser("remove", help="删除库内文件（按扩展名/SHA）")
    prm.add_argument("--library", required=True, help="库名")
    prm.add_argument("--ext", help="按扩展名删除，如 --ext txt")
    prm.add_argument("--sha", help="按源文件 SHA256 删除")
    prm.set_defaults(func=cmd_remove)

    # ---- stats ----
    pst = sub.add_parser("stats", help="查看统计")
    pst.add_argument("--library", help="指定库（默认列出全部）")
    pst.set_defaults(func=cmd_stats)

    # ---- build-index ----
    pbi = sub.add_parser("build-index", help="重建索引")
    pbi.add_argument("--library", required=True, help="库名")
    pbi.set_defaults(func=cmd_build_index)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
