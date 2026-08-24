# -*- coding: utf-8 -*-
"""讯飞通用文字识别 Web API（sf8e6aca1）

严格按《OCR-通用文档识别-讯飞.txt》实现：
HMAC-SHA256 签名鉴权，POST https://api.xf-yun.com/v1/private/sf8e6aca1
"""
import base64
import hashlib
import hmac
import json
from datetime import datetime, timezone
from urllib.parse import urlencode

import requests

from .models import Page

HOST = "api.xf-yun.com"
PATH = "/v1/private/sf8e6aca1"
URL = f"https://{HOST}{PATH}"
# 图片 base64 后不超过 4MB
MAX_IMAGE_B64_SIZE = 4 * 1024 * 1024


class OCRError(Exception):
    """OCR 调用异常（带中文说明）"""


def make_signature(api_secret: str, date: str) -> str:
    """按文档规则计算 signature：hmac-sha256(signature_origin, apiSecret) 后 base64"""
    signature_origin = f"host: {HOST}\ndate: {date}\nPOST {PATH} HTTP/1.1"
    signature_sha = hmac.new(
        api_secret.encode("utf-8"),
        signature_origin.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).digest()
    return base64.b64encode(signature_sha).decode("utf-8")


def make_authorization(api_key: str, api_secret: str, date: str) -> str:
    """拼接 authorization_origin 并 base64"""
    signature = make_signature(api_secret, date)
    authorization_origin = (
        f'api_key="{api_key}", algorithm="hmac-sha256", '
        f'headers="host date request-line", signature="{signature}"'
    )
    return base64.b64encode(authorization_origin.encode("utf-8")).decode("utf-8")


def build_auth_url(api_key: str, api_secret: str) -> str:
    """生成带鉴权参数的请求 URL（date 为 RFC1123 GMT 格式）"""
    date = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
    params = {
        "host": HOST,
        "date": date,
        "authorization": make_authorization(api_key, api_secret, date),
    }
    return URL + "?" + urlencode(params)


def _parse_result_text(text_b64: str) -> str:
    """解析 payload.result.text：base64 解码 -> JSON -> pages/lines/words 按行拼接"""
    try:
        data = json.loads(base64.b64decode(text_b64).decode("utf-8"))
    except Exception as e:
        raise OCRError(f"OCR 返回结果解析失败: {e}")
    out_lines = []
    for page in data.get("pages", []):
        for line in page.get("lines", []):
            words = "".join(w.get("content", "") for w in line.get("words", []))
            if words:
                out_lines.append(words)
    return "\n".join(out_lines)


def ocr_image(image_bytes: bytes, cfg: dict, image_format: str = "jpg") -> str:
    """对单张图片做 OCR，返回识别文本。

    cfg 需含 xf_appid / xf_api_key / xf_api_secret。
    """
    appid, api_key, api_secret = cfg.get("xf_appid"), cfg.get("xf_api_key"), cfg.get("xf_api_secret")
    if not (appid and api_key and api_secret):
        raise OCRError("讯飞 OCR 未配置：缺少 appid / api_key / api_secret")

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    if len(image_b64) > MAX_IMAGE_B64_SIZE:
        raise OCRError("图片 base64 编码后超过 4MB，无法调用讯飞 OCR")

    body = {
        "header": {"app_id": appid, "status": 3},
        "parameter": {
            "sf8e6aca1": {
                "category": "ch_en_public_cloud",
                "result": {"encoding": "utf8", "compress": "raw", "format": "json"},
            }
        },
        "payload": {
            "sf8e6aca1_data_1": {
                "encoding": image_format,
                "status": 3,
                "image": image_b64,
            }
        },
    }

    try:
        resp = requests.post(
            build_auth_url(api_key, api_secret),
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
    except requests.RequestException as e:
        raise OCRError(f"OCR 请求网络错误: {e}")

    if resp.status_code in (401, 403):
        raise OCRError(f"OCR 鉴权失败（HTTP {resp.status_code}）：{resp.text[:200]}，请检查讯飞密钥配置")
    if resp.status_code != 200:
        raise OCRError(f"OCR 请求失败（HTTP {resp.status_code}）：{resp.text[:200]}")

    try:
        result = resp.json()
    except ValueError:
        raise OCRError(f"OCR 响应不是合法 JSON：{resp.text[:200]}")

    header = result.get("header", {})
    if header.get("code") != 0:
        raise OCRError(f"OCR 服务返回错误 code={header.get('code')}: {header.get('message')}")
    return _parse_result_text(result["payload"]["result"]["text"])


def _render_pdf_page(page, dpi: int) -> bytes:
    """PyMuPDF 渲染单页为 jpg bytes"""
    import fitz
    pix = page.get_pixmap(dpi=dpi)
    return pix.tobytes("jpg")


def ocr_pdf(path: str, cfg: dict, progress_cb=None, dpi: int = 150) -> list:
    """扫描版 PDF 逐页渲染为 jpg 后 OCR，返回 list[Page]。

    渲染图 base64 超 4MB 时自动降低 dpi 重渲染。
    """
    import fitz
    pages = []
    with fitz.open(path) as pdf:
        total = len(pdf)
        for i, page in enumerate(pdf):
            cur_dpi = dpi
            image_bytes = _render_pdf_page(page, cur_dpi)
            # base64 后超 4MB 则逐步降 dpi
            while len(base64.b64encode(image_bytes)) > MAX_IMAGE_B64_SIZE and cur_dpi > 50:
                cur_dpi -= 30
                image_bytes = _render_pdf_page(page, cur_dpi)
            text = ocr_image(image_bytes, cfg)
            pages.append(Page(page_num=i + 1, text=text))
            if progress_cb:
                progress_cb(i + 1, total)
    return pages
