"""网页版启动入口。

用法：
    python web/run_web.py
    python web/run_web.py --port 9000
    python web/run_web.py --host 127.0.0.1 --port 8000

控制台输出启动基础信息（访问 URL 等）；请求/运行日志写入日志文件
（exe 同目录 / 项目根目录下的 logs/app.log，5MB 轮转保留 3 份）。
"""
import os
import sys
import socket
import argparse

# 注入路径：项目根目录（import app.*）与 web 目录（import server）
_WEB_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_WEB_DIR)
for _p in (_PROJECT_DIR, _WEB_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 初始化文件日志（请求/运行日志落盘）；控制台仍可正常输出基础信息
from web_logger import setup_logging  # noqa: E402

setup_logging()

from server import app  # noqa: E402

# Flask 自身日志统一走 root → 文件日志
app.logger.handlers = []
app.logger.propagate = True


def _local_ipv4s():
    """枚举本机 IPv4 地址，用于打印内网访问 URL。"""
    ips = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            ip = info[4][0]
            if ":" not in ip and ip not in ips and not ip.startswith("169.254"):
                ips.append(ip)
    except Exception:
        pass
    return ips or ["127.0.0.1"]


def main():
    parser = argparse.ArgumentParser(description="PDF 工具网页版")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址，默认 0.0.0.0")
    parser.add_argument("--port", type=int, default=8000, help="监听端口，默认 8000")
    args = parser.parse_args()

    line = "=" * 64
    print(line)
    print("  PDF 工具 网页版")
    print(line)
    print("本机访问（文件就地处理，无需上传）：")
    print(f"    http://127.0.0.1:{args.port}")
    print(f"    http://localhost:{args.port}")
    ips = _local_ipv4s()
    real = [ip for ip in ips if ip != "127.0.0.1"]
    if real:
        print("内网访问（其他设备访问，走上传/下载）：")
        for ip in real:
            print(f"    http://{ip}:{args.port}")
    print("-" * 64)
    print("说明：检测到本机访问时直接读写本机路径，文件不经过上传；")
    print("     其他设备访问时走上传/下载流程。按 Ctrl+C 停止服务。")
    print(line)

    # threaded=True：每个请求独立线程，SSE 长连接不阻塞其他请求
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
