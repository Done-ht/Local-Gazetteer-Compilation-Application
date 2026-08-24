"""本机检测：通过请求来源 IP 判断是否为本机访问。

启动时枚举本机所有网卡 IP（含回环地址），运行时比对请求来源 IP。
本机访问 → 文件就地处理（直接读写本机路径）；远程访问 → 走上传/下载。
"""
import socket
import platform


def _collect_local_ips() -> set:
    """收集本机所有网卡 IP 地址（含回环地址）。"""
    ips = {"127.0.0.1", "::1"}
    try:
        _, _, addr_list = socket.gethostbyname_ex(socket.gethostname())
        for addr in addr_list:
            ips.add(addr)
    except Exception:
        pass
    # 兜底：getaddrinfo 再扫一次，覆盖 gethostbyname_ex 漏掉的情况
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            ips.add(info[4][0])
    except Exception:
        pass
    return ips


# 进程启动时一次性收集，后续请求直接比对
_LOCAL_IPS = _collect_local_ips()


def is_local_request(remote_addr):
    """判断请求来源 IP 是否为本机。

    Args:
        remote_addr: flask request.remote_addr

    Returns:
        bool
    """
    if not remote_addr:
        return False
    addr = remote_addr
    # 处理 IPv6 映射的 IPv4 地址，如 ::ffff:127.0.0.1
    if addr.startswith("::ffff:"):
        addr = addr[7:]
    return addr in _LOCAL_IPS


def get_status():
    """返回本机状态信息（不含 is_local，由请求级判断填充）。"""
    return {
        "hostname": platform.node(),
        "local_ips": sorted(_LOCAL_IPS),
    }
