"""配置管理模块（服务端版：仅本地 PaddleOCR）。

负责加载 / 保存 config.json，提供全局配置访问。
服务端版本不包含讯飞 OCR 配置块，所有识别走本地 PaddleOCR。
"""
from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from typing import Any, Dict

# 配置文件存放位置：
# - 打包环境：exe 同级目录（用户可见可编辑，且可写）
# - 开发环境：server-paddle/ 项目根目录
if getattr(sys, "frozen", False):
    _CONFIG_DIR = os.path.dirname(sys.executable)
else:
    # config.py 位于 server-paddle/app/utils/config.py
    # 向上三层：utils -> app -> server-paddle
    _CONFIG_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
_CONFIG_PATH = os.path.join(_CONFIG_DIR, "config.json")

# 默认配置（服务端版）：仅本地 PaddleOCR，无讯飞凭证
_DEFAULT_CONFIG: Dict[str, Any] = {
    # OCR 提供商：服务端固定 paddle_local（保留字段以便前端展示）
    "ocr_provider": "paddle_local",
    # 输出格式：original | searchable_pdf | markdown | json | txt
    "output_format": "original",
    # 输出目录，空字符串表示与源文件同目录（服务端建议设为临时目录）
    "output_dir": "",
    # 双层漏斗过滤参数
    "filter": {
        # 第一层（OpenCV 统计特征）
        # 边缘密度 = Canny 边缘像素数 / 总像素数，低于阈值视为纯图
        "edge_density_threshold": 0.03,
        # 灰度方差低于阈值视为纯图
        "variance_threshold": 15.0,
        # 连通域最小面积，过小视为噪点
        "connected_min_area": 30,
        # 是否启用第二层 PaddleOCR 检测（服务端有 paddle 做 detector，默认 True）
        "enable_layer2": True,
    },
    # PaddleOCR 本地配置
    "paddle": {
        # 语言：ch | en | chinese_cht ...
        "lang": "ch",
        # 是否使用 GPU（CPU 模式设为 False，符合 8GB 内存纯 CPU 目标）
        "use_gpu": False,
        # OCR 版本：PP-OCRv6（默认）/ PP-OCRv5 / PP-OCRv4 / PP-OCRv3
        # v6 基于 PPLCNetV4，small 档识别精度 81.3%，单模型支持 50 种语言
        # paddleocr >= 3.7.0 + paddlepaddle >= 3.0.0
        "ocr_version": "PP-OCRv6",
        # 检测模型目录，空表示使用默认 PP-OCRv6_small_det
        "det_model_dir": "",
        # 识别模型目录，空表示使用默认 PP-OCRv6_small_rec
        "rec_model_dir": "",
        # 检测阈值，低于该置信度的框被丢弃
        "det_score_thresh": 0.3,
        # 识别丢弃阈值：0=保留全部（避免大字标题被默认 0.5 整行丢弃）
        "drop_score": 0.0,
        # DB 检测框扩展比例：1.8 覆盖大字标题边缘字符
        "det_db_unclip_ratio": 1.8,
        # DB 框保留阈值：0.5 召回低对比度标题
        "det_db_box_thresh": 0.5,
        # 是否启用版面分析（PPStructureV3）：按区域切割 + XY-cut 阅读顺序
        # true=多栏/嵌套标题正确排序；false=整页 OCR（更快但顺序可能乱）
        "enable_layout": True,
    },
    # 图片渲染 DPI（PDF 页面转图片用）
    "render_dpi": 200,
    # 服务端并发控制：同时处理的 OCR 任务数（PaddleOCR 非线程安全）
    "max_concurrent": 3,
    # 服务监听端口：启动时交互式确认，回车使用此默认值
    # 保存到 config.json 后下次启动会作为默认值显示
    "port": 8070,
    # 运行模式：paddle=本地 PaddleOCR；xfyun=讯飞云端 OCR
    # 空字符串=首次启动时交互式选择（选定后写入 config.json，重启前固定）
    "mode": "",
    # 讯飞云端 OCR 配置（xfyun 模式使用）
    "xf": {
        # 讯飞控制台凭证
        "app_id": "",
        "api_key": "",
        "api_secret": "",
        # standard=通用文字识别；llm=通用文档识别大模型
        "api_type": "llm",
        # 单张图片 base64 上限（字节），超限自动压缩
        "max_image_bytes": 4 * 1024 * 1024,
        # 单次请求超时（秒）
        "timeout": 60,
        # 网络错误/限流自动重试次数（0=关闭）
        "retry": 3,
        # 单任务内页级并发
        "page_concurrency": 5,
        # 每用户同时进行的识别任务数上限
        "concurrent_limit": 3,
        # 单文件上传上限（字节）
        "max_upload_bytes": 300 * 1024 * 1024,
    },
}


def load_config() -> Dict[str, Any]:
    """加载配置，文件不存在时写入默认配置并返回。"""
    if not os.path.exists(_CONFIG_PATH):
        save_config(_DEFAULT_CONFIG)
        return deepcopy(_DEFAULT_CONFIG)
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError):
        return deepcopy(_DEFAULT_CONFIG)
    # 合并默认值，确保新增字段存在
    return _merge(deepcopy(_DEFAULT_CONFIG), cfg)


def save_config(cfg: Dict[str, Any]) -> None:
    """保存配置到 config.json。"""
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并：override 覆盖 base。"""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def config_path() -> str:
    """返回配置文件路径。"""
    return _CONFIG_PATH
