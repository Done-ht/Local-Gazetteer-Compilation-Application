# -*- coding: utf-8 -*-
"""DocProof 核心模块 CLI 实测脚本（无 GUI）。

用法：.venv/Scripts/python.exe scripts/cli_test.py
"""
import logging
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
# Windows 控制台默认 GBK，强制 UTF-8 输出避免乱码/编码异常
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

import config
from core import docload, exporter, ocr_xfyun, proofread
from core.deepseek import DeepSeekClient, DeepSeekError

DATABASE_DIR = os.path.expanduser("~/Desktop/win/DEAResource/database")
YEARBOOK = os.path.join(DATABASE_DIR, "郎溪县钟桥街道2025年年鉴.docx")
DB_DOCX = os.path.join(DATABASE_DIR, "dp.docx")
OUT_DIR = os.path.join(PROJECT_ROOT, "test_output")


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_config():
    section("1. 配置 bootstrap（从 DEAResource 导入密钥）")
    cfg = config.bootstrap_from_dearesource()
    print(f"deepseek_configured: {config.deepseek_configured(cfg)}")
    print(f"xfyun_configured:    {config.xfyun_configured(cfg)}")
    key = cfg.get("deepseek_api_key", "")
    print(f"deepseek key 前缀:   {key[:6]}... (len={len(key)})")
    print(f"xf appid:            {cfg.get('xf_appid')}")
    print(f"token_limit:         {cfg.get('token_limit')}")
    return cfg


def test_load(path):
    section(f"加载文档: {os.path.basename(path)}")
    pages, ctx = docload.load_document(path)
    print(f"file_type={ctx.file_type} 页数={len(pages)} has_tables={ctx.has_tables} needs_ocr={ctx.needs_ocr}")
    for p in pages:
        print(f"  第{p.page_num}页({len(p.text)}字): {p.text[:50]!r}")
    return pages, ctx


def test_proofread(pages, cfg):
    section("3. 完整纠错（真实调用 DeepSeek）")
    client = DeepSeekClient(cfg["deepseek_api_key"])

    def on_progress(i, total, usage, total_usage):
        print(f"  块 {i}/{total}: 本次 {usage.total} tokens, 累计 {total_usage.total}")

    last_err = None
    for attempt in (1, 2):
        try:
            errors, usage = proofread.proofread(
                pages, client,
                token_limit=cfg.get("token_limit", 1000000),
                progress_cb=on_progress,
            )
            break
        except DeepSeekError as e:
            last_err = e
            print(f"  第 {attempt} 次调用失败: {e}" + ("，重试一次..." if attempt == 1 else ""))
    else:
        print(f"!! DeepSeek 调用失败: {last_err}")
        return None, None

    print(f"\n共检出 {len(errors)} 条错误，累计 tokens: {usage.total} "
          f"(prompt={usage.prompt_tokens}, completion={usage.completion_tokens})")
    for e in errors:
        print(f"  [{e.error_type}] p{e.page_num} @{e.offset_start}-{e.offset_end}")
        print(f"    原文: {e.original}")
        print(f"    建议: {e.suggestion}")
        print(f"    理由: {e.reason}")
    return errors, usage


def test_export(pages, ctx, errors):
    section("4. 导出验证")
    os.makedirs(OUT_DIR, exist_ok=True)
    corrections = [(e.page_num, e.offset_start, e.offset_end, e.suggestion) for e in errors]

    # 4a. docx 原位改写导出
    out_docx = os.path.join(OUT_DIR, "yearbook_corrected.docx")
    exporter.export_docx(ctx, corrections, out_docx)
    print(f"已导出: {out_docx}")

    # 重新加载验证
    pages2, _ = docload.load_document(out_docx)
    text2 = "".join(p.text for p in pages2)
    ok, fail = 0, 0
    for e in errors:
        if e.suggestion and e.suggestion in text2:
            ok += 1
        else:
            fail += 1
            print(f"  未生效: [{e.error_type}] {e.original!r} -> {e.suggestion!r}")
    print(f"docx 重新加载成功（{len(pages2)}页），correction 生效 {ok}/{len(errors)}")

    # 4b. txt 导出
    out_txt = os.path.join(OUT_DIR, "yearbook_merged.txt")
    exporter.export_txt(pages, out_txt)
    with open(out_txt, "r", encoding="utf-8") as f:
        txt = f.read()
    expected_txt = "\n\n".join(p.text for p in pages)
    print(f"已导出: {out_txt} ({len(txt)}字)，与页文本合并一致: {txt == expected_txt}")


def test_ocr_signature():
    section("5. 讯飞 OCR 签名自测（不真实调用）")
    signature = ocr_xfyun.make_signature(
        "apisecretXXXXXXXXXXXXXXXXXXXXXXX", "Wed, 11 Aug 2021 06:55:18 GMT")
    expected = "/mg2h9BCkespilZ94HUBaQVPq2v7PxYF90teTBlaxd8="
    print(f"计算值: {signature}")
    print(f"期望值: {expected}")
    assert signature == expected, "签名自测失败！"
    auth = ocr_xfyun.make_authorization(
        "apikeyXXXXXXXXXXXXXXXXXXXXXXXXXX",
        "apisecretXXXXXXXXXXXXXXXXXXXXXXX",
        "Wed, 11 Aug 2021 06:55:18 GMT")
    expected_auth = ("YXBpX2tleT0iYXBpa2V5WFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFgiLCBhbG"
                     "dvcml0aG09ImhtYWMtc2hhMjU2IiwgaGVhZGVycz0iaG9zdCBkYXRlIHJl"
                     "cXVlc3QtbGluZSIsIHNpZ25hdHVyZT0iL21nMmg5QkNrZXNwaWxaOTRIVU"
                     "JhUVZQcTJ2N1B4WUY5MHRlVEJsYXhkOD0i")
    print(f"authorization 匹配文档示例: {auth == expected_auth}")
    assert auth == expected_auth, "authorization 自测失败！"
    print("签名与 authorization 自测均通过 [OK]")


def test_robustness():
    section("6. 健壮性：加载 dp.docx")
    pages, ctx = docload.load_document(DB_DOCX)
    print(f"dp.docx: file_type={ctx.file_type} 页数={len(pages)} has_tables={ctx.has_tables}")
    total = sum(len(p.text) for p in pages)
    print(f"总字数: {total}")
    # 验证分块逻辑不抛异常
    chunks = proofread._split_chunks(pages)
    print(f"分块数: {len(chunks)}（每块≤{proofread.CHUNK_SIZE}字）")


def main():
    cfg = test_config()
    if not config.deepseek_configured(cfg):
        print("DeepSeek 未配置，终止")
        sys.exit(1)

    pages, ctx = test_load(YEARBOOK)
    errors, usage = test_proofread(pages, cfg)
    if errors is not None:
        test_export(pages, ctx, errors)
    test_ocr_signature()
    test_robustness()

    section("对照 db_error.txt 的三处隐蔽错误")
    print("错误1: 农业面积统计逻辑（再生稻重复统计）")
    print("错误2: '委托第三方运维…'主语缺失语病")
    print("错误3: '情暖夕阳'漏'红'字（应为'情暖夕阳红'）")
    if errors:
        hits = {"情暖夕阳": False, "运维": False, "再生稻": False}
        for e in errors:
            blob = e.original + e.suggestion + e.reason
            for k in hits:
                if k in blob:
                    hits[k] = True
        print(f"命中: 漏字={'是' if hits['情暖夕阳'] else '否'} "
              f"语病={'是' if hits['运维'] else '否'} "
              f"逻辑={'是' if hits['再生稻'] else '否'}")


if __name__ == "__main__":
    main()
