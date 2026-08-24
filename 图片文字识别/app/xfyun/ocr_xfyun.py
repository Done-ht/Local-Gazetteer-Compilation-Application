# -*- coding: utf-8 -*-
"""讯飞 OCR 客户端（仅调用讯飞云服务）。

支持两种接口（config.json 的 xf.api_type 切换）：
- standard：通用文字识别（sf8e6aca1），ch_en_public_cloud
- llm：通用文档识别大模型（se75ocrbm）

统一输出结构化的行级结果（text + 像素坐标），供可搜索 PDF 定位文字层。
图片在调用前经 Pillow 压缩，保证 base64 不超过单图上限（默认 4MB），
并返回压缩后的图片字节与尺寸，可搜索 PDF 以压缩图做底图，坐标严格对齐。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, List, Optional, Tuple
from urllib.parse import urlencode

import requests

# ----------------------------------------------------------------------
# 数据结构
# ----------------------------------------------------------------------


@dataclass
class OcrLine:
    """单行识别结果。"""

    text: str
    # 包围盒 (x0, y0, x1, y1)，像素坐标，相对压缩后图片
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    confidence: float = 0.0


@dataclass
class OcrPage:
    """一页（或一张图）的识别结果。"""

    page_num: int
    text: str = ""
    lines: List[OcrLine] = field(default_factory=list)
    # 底图（压缩后 jpg 字节），可搜索 PDF 用它做背景页；文本型页可为 None
    image_bytes: Optional[bytes] = None
    width: int = 0
    height: int = 0


class OCRError(Exception):
    """OCR 调用异常（带中文说明）"""


class _RetryableError(Exception):
    """网络层/限流类错误：自动重试（不重新计费的未知结果，宁可重试避免整单浪费）。"""


# ----------------------------------------------------------------------
# 图片压缩：保证 base64 不超上限，返回 (jpg_bytes, width, height)
# ----------------------------------------------------------------------


def prepare_image(image_bytes: bytes, max_b64: int = 4 * 1024 * 1024) -> Tuple[bytes, int, int]:
    """将任意格式图片压缩为 jpg，保证 base64 编码后 ≤ max_b64。

    先按质量阶梯压缩，仍超限则等比缩放；最后兜底强制压到 30 质量。
    """
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")
    except Exception as e:
        raise OCRError(f"图片无法解析: {e}")

    w, h = img.size
    max_bytes = max_b64 * 3 // 4  # base64 上限折算回原始字节上限

    def encode(im: Image.Image, quality: int) -> Optional[bytes]:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
        data = buf.getvalue()
        return data if len(data) <= max_bytes else None

    # 质量阶梯
    for quality in (95, 85, 75, 65, 50):
        data = encode(img, quality)
        if data:
            return data, w, h

    # 缩放阶梯
    scale = 0.75
    while scale > 0.1:
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        small = img.resize((nw, nh), Image.LANCZOS)
        data = encode(small, 80)
        if data:
            return data, nw, nh
        scale *= 0.75

    # 兜底
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=30)
    return buf.getvalue(), w, h


# ----------------------------------------------------------------------
# 鉴权（HMAC-SHA256，两种接口共用签名规则，host/request-line 不同）
# ----------------------------------------------------------------------


def _build_auth_url(api_key: str, api_secret: str, sign_host: str, url_host: str, path: str) -> str:
    """生成带鉴权参数的请求 URL。

    sign_host：签名用 host（讯飞 llm 接口规定统一用 api.xf-yun.com）；
    url_host：实际请求域名（llm 接口为 cbm01.cn-huabei-1.xf-yun.com）。
    """
    date = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
    signature_origin = f"host: {sign_host}\ndate: {date}\nPOST {path} HTTP/1.1"
    signature_sha = hmac.new(
        api_secret.encode("utf-8"),
        signature_origin.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).digest()
    signature = base64.b64encode(signature_sha).decode("utf-8")
    authorization_origin = (
        f'api_key="{api_key}", algorithm="hmac-sha256", '
        f'headers="host date request-line", signature="{signature}"'
    )
    authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode("utf-8")
    params = {"host": sign_host, "date": date, "authorization": authorization}
    return f"https://{url_host}{path}?{urlencode(params)}"


def _extract_lines_standard(data: dict) -> List[OcrLine]:
    """解析通用文字识别响应：pages[].lines[].words/coord。"""
    lines: List[OcrLine] = []
    for page in data.get("pages", []):
        for line in page.get("lines", []):
            text = "".join(w.get("content", "") for w in line.get("words", []))
            coords = [(int(c.get("x", 0)), int(c.get("y", 0))) for c in line.get("coord", [])]
            if not text:
                continue
            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]
            bbox = (min(xs), min(ys), max(xs), max(ys)) if coords else (0, 0, 0, 0)
            lines.append(OcrLine(text=text, bbox=bbox, confidence=float(line.get("conf", 0.0))))
    return lines


def _extract_lines_llm(data: dict) -> List[OcrLine]:
    """递归解析大模型 OCR 响应：image[].content[][].textline。"""
    lines: List[OcrLine] = []

    def walk(node) -> None:
        if isinstance(node, dict):
            if node.get("type") == "textline":
                text = "".join(node.get("text", []))
                coords = [(int(c.get("x", 0)), int(c.get("y", 0))) for c in node.get("coord", [])]
                if text:
                    xs = [p[0] for p in coords]
                    ys = [p[1] for p in coords]
                    bbox = (min(xs), min(ys), max(xs), max(ys)) if coords else (0, 0, 0, 0)
                    lines.append(OcrLine(text=text, bbox=bbox, confidence=float(node.get("score", 0.0))))
            else:
                for v in node.values():
                    walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(data)
    return lines


# ----------------------------------------------------------------------
# 调用
# ----------------------------------------------------------------------


def _call(image_jpg: bytes, cfg: dict) -> List[OcrLine]:
    """按配置调用讯飞 OCR，返回行列表。网络错误/限流自动重试。"""
    retries = int(cfg.get("xf", {}).get("retry", 3))
    delay = 2.0
    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            return _call_once(image_jpg, cfg)
        except _RetryableError as e:
            last_err = e
            if attempt < retries:
                time.sleep(delay)
                delay *= 2  # 2s / 4s / 8s ...
    assert last_err is not None
    raise OCRError(f"OCR 请求网络错误（已自动重试 {retries} 次仍失败）：{last_err}")


def _call_once(image_jpg: bytes, cfg: dict) -> List[OcrLine]:
    """单次调用，不重试。网络/限流错误抛 _RetryableError，配置类错误抛 OCRError。"""
    xf = cfg.get("xf", {})
    app_id, api_key, api_secret = xf.get("app_id"), xf.get("api_key"), xf.get("api_secret")
    if not (app_id and api_key and api_secret):
        raise OCRError("讯飞 OCR 未配置：缺少 app_id / api_key / api_secret（见 config.json）")

    api_type = xf.get("api_type", "llm")
    timeout = float(xf.get("timeout", 30))
    b64 = base64.b64encode(image_jpg).decode("utf-8")

    if api_type == "llm":
        # 通用文档识别大模型（se75ocrbm）：请求域名 cbm01，签名 host 仍为 api.xf-yun.com
        url_host, sign_host, path = "cbm01.cn-huabei-1.xf-yun.com", "api.xf-yun.com", "/v1/private/se75ocrbm"
        url = _build_auth_url(api_key, api_secret, sign_host, url_host, path)
        body = {
            "header": {"app_id": app_id, "status": 0},
            "parameter": {
                "ocr": {
                    "result_option": "normal",
                    "result_format": "json",
                    "output_type": "one_shot",
                    "result": {"encoding": "utf8", "compress": "raw", "format": "json"},
                }
            },
            "payload": {
                "image": {"encoding": "jpg", "image": b64, "status": 0, "seq": 0}
            },
        }
        parse = _extract_lines_llm
    else:
        # 通用文字识别（sf8e6aca1）：域名与签名 host 均为 api.xf-yun.com
        url_host, sign_host, path = "api.xf-yun.com", "api.xf-yun.com", "/v1/private/sf8e6aca1"
        url = _build_auth_url(api_key, api_secret, sign_host, url_host, path)
        body = {
            "header": {"app_id": app_id, "status": 3},
            "parameter": {
                "sf8e6aca1": {
                    "category": "ch_en_public_cloud",
                    "result": {"encoding": "utf8", "compress": "raw", "format": "json"},
                }
            },
            "payload": {"sf8e6aca1_data_1": {"encoding": "jpg", "status": 3, "image": b64}},
        }
        parse = _extract_lines_standard

    try:
        resp = requests.post(
            url,
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
    except requests.RequestException as e:
        raise _RetryableError(e)

    if resp.status_code in (401, 403):
        raise OCRError(f"OCR 鉴权失败（HTTP {resp.status_code}）：{resp.text[:200]}，请检查讯飞密钥配置")
    if resp.status_code == 429 or resp.status_code >= 500:
        # 限流 / 服务端错误：可重试
        raise _RetryableError(f"HTTP {resp.status_code}: {resp.text[:120]}")
    if resp.status_code != 200:
        raise OCRError(f"OCR 请求失败（HTTP {resp.status_code}）：{resp.text[:200]}")

    try:
        result = resp.json()
    except ValueError:
        raise OCRError(f"OCR 响应不是合法 JSON：{resp.text[:200]}")

    header = result.get("header", {})
    if header.get("code") != 0:
        raise OCRError(f"OCR 服务返回错误 code={header.get('code')}: {header.get('message')}")

    text_b64 = result.get("payload", {}).get("result", {}).get("text", "")
    if not text_b64:
        return []
    try:
        data = json.loads(base64.b64decode(text_b64).decode("utf-8"))
    except Exception as e:
        raise OCRError(f"OCR 返回结果解析失败: {e}")
    return parse(data)


def ocr_image(image_bytes: bytes, cfg: dict, page_num: int = 1) -> OcrPage:
    """识别单张图片。自动压缩到接口限额。"""
    max_b64 = int(cfg.get("xf", {}).get("max_image_bytes", 4 * 1024 * 1024))
    jpg_bytes, w, h = prepare_image(image_bytes, max_b64)
    lines = _call(jpg_bytes, cfg)
    text = "\n".join(ln.text for ln in lines)
    return OcrPage(
        page_num=page_num,
        text=text,
        lines=lines,
        image_bytes=jpg_bytes,
        width=w,
        height=h,
    )


def pdf_page_count(path: str) -> int:
    """PDF 页数。"""
    try:
        import fitz
    except ImportError:
        raise OCRError("未安装 PyMuPDF(fitz)，无法处理 PDF；请运行 setup.bat 或 pip install -r requirements.txt")
    with fitz.open(path) as pdf:
        return len(pdf)


def ocr_pdf_page(path: str, page_index: int, cfg: dict) -> OcrPage:
    """渲染 PDF 的指定页并 OCR（page_index 从 0 起）。支持按页断点续跑。"""
    try:
        import fitz
    except ImportError:
        raise OCRError("未安装 PyMuPDF(fitz)，无法处理 PDF；请运行 setup.bat 或 pip install -r requirements.txt")

    # DPI 用顶层 render_dpi（与 paddle 模式共用同一配置项）
    dpi = int(cfg.get("render_dpi", 200))
    max_b64 = int(cfg.get("xf", {}).get("max_image_bytes", 4 * 1024 * 1024))
    with fitz.open(path) as pdf:
        if not (0 <= page_index < len(pdf)):
            raise OCRError(f"PDF 页索引越界: {page_index}")
        page = pdf[page_index]
        # 渲染，超限则自动降 dpi
        cur_dpi = dpi
        while True:
            pix = page.get_pixmap(dpi=cur_dpi)
            raw = pix.tobytes("jpeg")
            if len(raw) * 4 // 3 <= max_b64 or cur_dpi <= 60:
                break
            cur_dpi -= 30
        jpg_bytes, w, h = prepare_image(raw, max_b64)
        lines = _call(jpg_bytes, cfg)
        text = "\n".join(ln.text for ln in lines)
        return OcrPage(
            page_num=page_index + 1,
            text=text,
            lines=lines,
            image_bytes=jpg_bytes,
            width=w,
            height=h,
        )


def ocr_pdf(
    path: str,
    cfg: dict,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> List[OcrPage]:
    """整本 PDF 逐页 OCR（一次性，无断点）。需要按页续跑请用 pdf_page_count + ocr_pdf_page。"""
    total = pdf_page_count(path)
    pages: List[OcrPage] = []
    for i in range(total):
        pages.append(ocr_pdf_page(path, i, cfg))
        if progress_cb:
            progress_cb(i + 1, total)
    return pages


def network_check(cfg: dict) -> dict:
    """探测到讯飞 OCR 服务器的网络连通性（DNS + TCP 443 建连），不调用 OCR 接口、不产生计费。

    探测域名与实际请求域名一致（llm 接口为 cbm01.cn-huabei-1.xf-yun.com，
    standard 为 api.xf-yun.com），避免白名单网络下探了别的域名误判。
    返回 {"ok": bool, "host": str, "ms": int, "detail": str}。
    """
    api_type = cfg.get("xf", {}).get("api_type", "llm")
    host = "cbm01.cn-huabei-1.xf-yun.com" if api_type == "llm" else "api.xf-yun.com"
    t0 = time.time()
    try:
        with __import__("socket").create_connection((host, 443), timeout=3):
            pass
        ms = int((time.time() - t0) * 1000)
        return {"ok": True, "host": host, "ms": ms, "detail": ""}
    except Exception as e:
        ms = int((time.time() - t0) * 1000)
        return {"ok": False, "host": host, "ms": ms, "detail": str(e)}
