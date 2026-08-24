#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""snap - 基于 git 的极简检查点工具

用法:
  snap list              列出所有检查点
  snap save <说明>       设置一个检查点
  snap prev              回退到上一个检查点
  snap go <编号>         回退到指定编号的检查点
  snap keep [N]          只保留最近 N 个检查点,丢弃更早的(默认 N=3)
  snap purge             立即回收被丢弃检查点占用的磁盘空间

选项:
  -y / --yes             跳过所有确认提示(脚本/演示中可用)

说明:
  每个"检查点"底层是一次 git 提交,记录当时工作区的完整快照。
  list 按时间从早到晚编号 1..N,回退后编号自动连续。
  prev / go 采用"时光倒流"语义:回退后,被跳过的检查点从历史移除。

  关于磁盘空间:
  prev/go 丢弃的检查点,其数据仍暂存在 .git 中(便于误退后用 git reflog
  找回,默认保留约 90 天)。如果想让这些空间立即释放,执行 snap purge。
  keep N 会丢弃较早的检查点并自动执行清理。
"""

import subprocess
import sys
import time
import shutil

# 定位 git:优先用系统 PATH 中的 git,找不到则回退到 Windows 默认安装路径
GIT = shutil.which("git") or r"C:\Program Files\Git\cmd\git.exe"

# 是否跳过确认(由命令行 -y 设置)
ASSUME_YES = False


def run(*args, check=True):
    """执行一条 git 命令并返回结果。check 为 True 时失败则报错退出。"""
    result = subprocess.run(
        [GIT, *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and result.returncode != 0:
        msg = (result.stderr or result.stdout).strip() or "未知错误"
        print(f"错误: {msg}", file=sys.stderr)
        sys.exit(1)
    return result


def ensure_repo():
    """确保当前目录是 git 仓库,否则自动初始化。"""
    r = subprocess.run(
        [GIT, "rev-parse", "--is-inside-work-tree"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        run("init")
        print("已初始化检查点系统。")


def is_dirty():
    """工作区是否有未提交改动(含未追踪文件)。"""
    return bool(run("status", "--porcelain").stdout.strip())


def snapshots():
    """返回当前历史中的全部检查点(从早到晚)。
    每项为 (编号, hash, 时间戳, 说明)。"""
    out = run("log", "--reverse", "--format=%H|%ct|%s").stdout
    result = []
    for i, line in enumerate(out.splitlines(), 1):
        if not line:
            continue
        h, ts, msg = line.split("|", 2)
        result.append((i, h, int(ts), msg))
    return result


def current_number():
    """当前 HEAD 对应的检查点编号;无提交时返回 0。"""
    r = run("rev-parse", "HEAD", check=False)
    if r.returncode != 0:
        return 0
    head = r.stdout.strip()
    for num, h, _, _ in snapshots():
        if h == head:
            return num
    return 0


def confirm(prompt):
    """y/n 确认。-y 模式下直接通过。"""
    if ASSUME_YES:
        return True
    while True:
        ans = input(f"{prompt} [y/n]: ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no", ""):
            return False


def handle_dirty(action):
    """工作区有改动时的交互处理。
    返回 True 表示可继续执行;False 表示应中止(用户取消或已另存)。"""
    print(f"\n工作区有未保存的改动,{action}会丢失这些改动。")
    if ASSUME_YES:
        print("(-y 模式:自动丢弃改动)")
        return True
    print("  1. 丢弃改动并继续")
    print("  2. 保存为新检查点(保存后中止本次操作)")
    print("  3. 取消")
    while True:
        c = input("请选择 [1/2/3]: ").strip()
        if c == "1":
            return True
        if c == "2":
            msg = input("新检查点说明: ").strip() or "自动保存"
            run("add", "-A")
            run("commit", "-m", msg, check=False)
            print("已保存为新检查点,未执行本次操作。")
            return False
        if c == "3":
            return False


def cmd_list():
    snaps = snapshots()
    if not snaps:
        print("暂无检查点。用 save 设置第一个。")
        return
    cur = current_number()
    print(f"{'编号':<6}{'时间':<20}说明")
    print("-" * 60)
    for num, _, ts, msg in snaps:
        t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        mark = "*" if num == cur else " "
        print(f"{mark}#{num:<5}{t:<20}{msg}")
    print(f"\n(* 当前所在检查点,共 {len(snaps)} 个)")


def cmd_save(message):
    if not is_dirty():
        print("工作区无改动,无需设置检查点。")
        return
    run("add", "-A")
    r = run("commit", "-m", message, check=False)
    if r.returncode != 0:
        print("没有可保存的改动。")
        return
    print(f"已设置检查点 #{len(snapshots())}: {message}")


def cmd_prev():
    cur = current_number()
    if cur <= 1:
        print("已在最早的检查点,没有上一个可回退。")
        return
    if is_dirty() and not handle_dirty("回退"):
        return
    # 重新读取(handle_dirty 若选"丢弃"会清空改动,状态需刷新)
    snaps = {n: h for n, h, _, _ in snapshots()}
    cur = current_number()
    if cur <= 1:
        print("已在最早的检查点。")
        return
    target = cur - 1
    print(f"将回退到 #{target}(当前 #{cur} 将被丢弃)。")
    if not confirm("确认回退?"):
        return
    run("reset", "--hard", snaps[target])
    print(f"已回退到检查点 #{target}。")


def cmd_go(n):
    snaps = {num: h for num, h, _, _ in snapshots()}
    if n not in snaps:
        max_n = max(snaps) if snaps else 0
        print(f"检查点 #{n} 不存在。可用编号: 1..{max_n}")
        return
    cur = current_number()
    if n == cur:
        print(f"当前已在检查点 #{n}。")
        return
    if is_dirty() and not handle_dirty("切换"):
        return
    snaps = {num: h for num, h, _, _ in snapshots()}
    if n not in snaps:
        print(f"检查点 #{n} 已不存在。")
        return
    if n < cur:
        desc = f"将回退到 #{n}(丢弃 #{n+1}..#{cur} 共 {cur - n} 个检查点)"
    else:
        desc = f"将前进到 #{n}"
    print(desc + "。")
    if not confirm("确认?"):
        return
    run("reset", "--hard", snaps[n])
    print(f"已切换到检查点 #{n}。")


def cmd_purge():
    """立即回收被丢弃检查点占用的磁盘空间。

    原理:先让 reflog 立即过期,再执行 git gc 把无引用的对象物理删除。
    执行后,被 prev/go 丢弃的检查点将无法再用 git reflog 找回。
    """
    if not confirm("purge 将永久删除被丢弃的检查点数据(无法再用 reflog 找回),继续?"):
        return
    # 让所有 reflog 条目立即过期(否则 gc 不会回收它们引用的对象)
    run("reflog", "expire", "--expire=now", "--all")
    # gc:物理删除无引用对象;--prune=now 表示立即清理(不等默认 2 周)
    r = run("gc", "--prune=now", check=False)
    print("已回收磁盘空间。")
    if r.stderr:
        # gc 进度信息走 stderr,不必展示给用户
        pass


def cmd_keep(n):
    """只保留最近 N 个检查点,丢弃更早的。

    用"孤儿分支重建"策略:把末尾 N 个提交嫁接到一个新根上,
    丢弃更早的提交。重建完成后立即回收磁盘空间。
    """
    snaps = snapshots()
    total = len(snaps)
    if total <= n:
        print(f"当前共 {total} 个检查点,未超过保留数 {n},无需清理。")
        return
    keep_from = total - n + 1
    drop_count = keep_from - 1
    target_hash = snaps[keep_from - 1][1]
    print(f"将保留最近 {n} 个检查点(#{keep_from}..#{total}),丢弃 #{1}..#{keep_from - 1} 共 {drop_count} 个。")
    if not confirm("确认?"):
        return

    cur_branch = run("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    tmp_branch = "snap-keep-tmp"
    # 在 target_hash 上创建孤儿分支,作为重建后的新历史起点
    run("checkout", "--orphan", tmp_branch, target_hash)
    run("commit", "-m", f"保留检查点 #{keep_from}(keep 重建)")
    # 依次嫁接 target_hash 之后的提交
    for i in range(keep_from, total):
        h = snaps[i][1]
        r = run("cherry-pick", h, check=False)
        if r.returncode != 0:
            # 冲突:回滚到原状态,删临时分支
            run("cherry-pick", "--abort", check=False)
            run("checkout", cur_branch)
            run("branch", "-D", tmp_branch, check=False)
            print(f"错误:重建历史时在 #{i+1} 遇到冲突,已中止,未做任何更改。", file=sys.stderr)
            sys.exit(1)
    # 重建成功:删除原分支,把临时分支改名为原分支
    run("branch", "-D", cur_branch)
    run("branch", "-m", cur_branch)
    print(f"已保留最近 {n} 个检查点。")
    # 自动回收被丢弃的早期提交空间
    run("reflog", "expire", "--expire=now", "--all")
    run("gc", "--prune=now", check=False)
    print("已自动回收磁盘空间。")


def main():
    global ASSUME_YES
    args = sys.argv[1:]
    if args and args[0] in ("-y", "--yes"):
        ASSUME_YES = True
        args = args[1:]
    ensure_repo()
    if not args:
        print(__doc__)
        return
    cmd, rest = args[0], args[1:]
    if cmd == "list":
        cmd_list()
    elif cmd == "save":
        if not rest:
            print("用法: snap save <说明>")
            sys.exit(1)
        cmd_save(" ".join(rest))
    elif cmd == "prev":
        cmd_prev()
    elif cmd == "go":
        if not rest:
            print("用法: snap go <编号>")
            sys.exit(1)
        try:
            cmd_go(int(rest[0]))
        except ValueError:
            print("编号必须是整数。")
            sys.exit(1)
    elif cmd == "keep":
        n = 3  # 默认保留 3 个
        if rest:
            try:
                n = int(rest[0])
            except ValueError:
                print("保留数量必须是整数。")
                sys.exit(1)
        if n < 1:
            print("保留数量至少为 1。")
            sys.exit(1)
        cmd_keep(n)
    elif cmd == "purge":
        cmd_purge()
    else:
        print(f"未知命令: {cmd}\n")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
