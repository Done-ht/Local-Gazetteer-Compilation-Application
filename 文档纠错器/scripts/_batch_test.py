# -*- coding: utf-8 -*-
"""DocProof 数据集批量测试脚本：对比 deepseek-v4-flash 与 deepseek-v4-pro。

用法：
    .venv/Scripts/python.exe scripts/_batch_test.py
    .venv/Scripts/python.exe scripts/_batch_test.py --cases 01,05,08,11,17
    .venv/Scripts/python.exe scripts/_batch_test.py --models flash,pro
    .venv/Scripts/python.exe scripts/_batch_test.py --no-review   # 关闭二次复核
    .venv/Scripts/python.exe scripts/_batch_test.py --typos       # 跑隐蔽错别字测试

输出：
    test_output/_batch_test_report.txt  （详细结果，下划线开头排在顶部）
    test_output/_batch_test_summary.csv （汇总表）
    控制台同步打印进度
"""
import argparse
import csv
import json
import logging
import os
import sys
import time
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")

import config
from core import docload
from core.deepseek import DeepSeekClient, DeepSeekError
from core.proofread import build_system_prompt, _split_chunks, _parse_response, _locate

DATASET_ROOT = os.path.expanduser("~/Desktop/数据集生产/冲突文档数据集")
TYPO_DIR = os.path.join(PROJECT_ROOT, "test_output", "_typos")
OUT_DIR = os.path.join(PROJECT_ROOT, "test_output")
os.makedirs(OUT_DIR, exist_ok=True)

# 模型显示名 -> 实际模型名
MODELS = {
    "flash": "deepseek-v4-flash",
    "pro": "deepseek-v4-pro",
}

# 案例分类（用于评估"应不应报错"）
# key=案例编号, value=(类别, 期望是否应检出错误)
# - 事实错误类：应检出（数据/史实/年代/官衔/讹误等）
# - 争议体例类：不应检出（仅观点分歧）
# - 夸大/讳饰/立场：表述问题，可能检出（视模型理解）
CASE_INFO = {
    "01": ("事实-数据冲突", True),
    "02": ("事实-文表不统一", True),
    "03": ("事实-表格内部矛盾", True),
    "04": ("事实-跨卷数据矛盾", True),
    "05": ("事实-史实错误", True),
    "06": ("事实-官衔混淆", True),
    "07": ("事实-学术史实错误", True),
    "08": ("事实-年代换算错误", True),
    "09": ("事实-与国史冲突", True),
    "10": ("事实-资料失真", True),
    "11": ("争议-犯罪分子入志", False),
    "12": ("争议-入志标准", False),
    "13": ("争议-在世人物入志", False),
    "14": ("夸大-奉贤县志", True),
    "15": ("立场-扑灭措辞", True),
    "16": ("讳饰-四讳问题", True),
    "17": ("争议-续修方式", False),
    "18": ("争议-通纪断代", False),
    "19": ("争议-市志概念", False),
    "20": ("争议-越境不书", False),
    "21": ("争议-年鉴化倾向", False),
    "22": ("事实-因袭旧志", True),
    "23": ("事实-孤证失误", True),
    "24": ("事实-传抄讹误", True),
}


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def list_cases(root):
    """列出所有案例编号及对应文件夹"""
    cases = []
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if os.path.isdir(full) and name[:2].isdigit():
            cases.append((name[:2], name, full))
    return cases


def load_case_docs(case_dir):
    """加载案例下所有 docx，返回 [(filename, pages, ctx, text), ...]"""
    docs = []
    for name in sorted(os.listdir(case_dir)):
        if not name.lower().endswith(".docx"):
            continue
        path = os.path.join(case_dir, name)
        try:
            pages, ctx = docload.load_document(path)
            text = "".join(p.text for p in pages)
            docs.append((name, pages, ctx, text))
        except Exception as e:
            print(f"  !! 加载失败 {name}: {e}")
    return docs


def proofread_single_doc(pages, client, model_name, with_review=False):
    """对单文档跑校对（直接使用项目核心逻辑）。

    返回 (errors, total_usage, elapsed_seconds)
    with_review=True 时启用二次复核。
    """
    from core.models import TokenUsage
    from core.proofread import _build_user_prompt, _review_chunk

    chunks = _split_chunks(pages)
    total_chunks = len(chunks)
    system_prompt = build_system_prompt({"switches": {}, "custom": []})

    errors, entities = [], []
    prev_tail = ""
    total_usage = TokenUsage()
    consumed = {}
    page_texts = {p.page_num: p.text for p in pages}

    t0 = time.time()
    for i, (page_num, chunk_text) in enumerate(chunks, start=1):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _build_user_prompt(chunk_text, entities, prev_tail)},
        ]
        content, usage = client.chat(messages)
        total_usage.add(usage)
        try:
            raw_errors, new_entities = _parse_response(content)
        except Exception:
            raw_errors, new_entities = [], []
        for e in new_entities:
            if isinstance(e, str) and e and e not in entities:
                entities.append(e)

        chunk_errors = []
        for raw in raw_errors:
            item = _locate(raw, page_num, page_texts, consumed)
            if item is None:
                continue
            chunk_errors.append(item)

        if with_review and chunk_errors:
            next_head = chunks[i][1] if i < len(chunks) else ""
            matched, rusage = _review_chunk(
                client, chunk_text, chunk_errors, prev_tail, next_head, 800)
            total_usage.add(rusage)
            for item, verdict, vreason in matched:
                v = (verdict or "").lower()
                if v == "reject":
                    item.confidence = "存疑"
                    if vreason:
                        item.reason = f"{item.reason}（复核疑似误报：{vreason}）"
                elif v == "uncertain":
                    item.confidence = "存疑"
                else:
                    item.confidence = "明确"
                errors.append(item)
        else:
            errors.extend(chunk_errors)
        prev_tail = chunk_text[-200:]

    elapsed = time.time() - t0
    return errors, total_usage, elapsed


def run_case_test(case_id, case_name, case_dir, model_key, api_key, with_review):
    """测试单个案例的所有文档，返回每文档的结果列表"""
    model_name = MODELS[model_key]
    client = DeepSeekClient(api_key, model=model_name)
    docs = load_case_docs(case_dir)
    results = []
    for doc_name, pages, ctx, text in docs:
        print(f"  [{model_key}] {case_id}/{doc_name} ({len(text)}字) ...", end=" ", flush=True)
        try:
            errors, usage, elapsed = proofread_single_doc(pages, client, model_name, with_review)
            print(f"{len(errors)}错 / {usage.total}tokens / {elapsed:.1f}s")
            results.append({
                "case_id": case_id,
                "case_name": case_name,
                "doc_name": doc_name,
                "chars": len(text),
                "model": model_key,
                "model_name": model_name,
                "errors_count": len(errors),
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total,
                "elapsed": round(elapsed, 1),
                "errors": [
                    {
                        "type": e.error_type,
                        "confidence": e.confidence,
                        "original": e.original,
                        "suggestion": e.suggestion,
                        "reason": e.reason,
                        "page": e.page_num,
                    } for e in errors
                ],
            })
        except DeepSeekError as e:
            print(f"FAIL: {e}")
            results.append({
                "case_id": case_id, "case_name": case_name, "doc_name": doc_name,
                "chars": len(text), "model": model_key, "model_name": model_name,
                "errors_count": -1, "prompt_tokens": 0, "completion_tokens": 0,
                "total_tokens": 0, "elapsed": 0,
                "errors": [], "error": str(e),
            })
        # 短暂休眠避免触发限流
        time.sleep(0.5)
    return results


def write_report(all_results, with_review, typo_results=None):
    """写出详细报告和 CSV 汇总"""
    report_path = os.path.join(OUT_DIR, "_batch_test_report.txt")
    csv_path = os.path.join(OUT_DIR, "_batch_test_summary.csv")

    # CSV 汇总：一行一个 (case, doc, model)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["案例ID", "案例名", "文档名", "字数", "模型", "检出数",
                    "prompt_tokens", "completion_tokens", "total_tokens",
                    "耗时(秒)", "类别", "应检出", "状态"])
        for r in all_results:
            cid = r["case_id"]
            cat, should = CASE_INFO.get(cid, ("未知", None))
            status = ""
            if should is True and r["errors_count"] > 0:
                status = "正确检出"
            elif should is False and r["errors_count"] == 0:
                status = "正确未报"
            elif should is False and r["errors_count"] > 0:
                status = f"误报{r['errors_count']}条"
            elif should is True and r["errors_count"] == 0:
                status = "漏报"
            w.writerow([cid, r["case_name"], r["doc_name"], r["chars"],
                        r["model"], r["errors_count"],
                        r["prompt_tokens"], r["completion_tokens"], r["total_tokens"],
                        r["elapsed"], cat, "是" if should else "否", status])

    # 详细报告
    lines = []
    lines.append("=" * 70)
    lines.append("DocProof 数据集批量测试报告")
    lines.append(f"二次复核：{'开启' if with_review else '关闭'}")
    lines.append(f"测试时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    # 按模型汇总
    for mk in MODELS:
        rs = [r for r in all_results if r["model"] == mk]
        if not rs:
            continue
        total_tokens = sum(r["total_tokens"] for r in rs)
        total_prompt = sum(r["prompt_tokens"] for r in rs)
        total_comp = sum(r["completion_tokens"] for r in rs)
        total_errors = sum(max(0, r["errors_count"]) for r in rs)
        total_chars = sum(r["chars"] for r in rs)
        elapsed_sum = sum(r["elapsed"] for r in rs)
        lines.append(f"\n## 模型 {mk} ({MODELS[mk]})")
        lines.append(f"  文档数：{len(rs)}")
        lines.append(f"  总字数：{total_chars}")
        lines.append(f"  总耗时：{elapsed_sum:.1f}s（平均 {elapsed_sum/len(rs):.1f}s/文档）")
        lines.append(f"  总 tokens：{total_tokens}（prompt={total_prompt}, completion={total_comp}）")
        lines.append(f"  平均 tokens/文档：{total_tokens/len(rs):.0f}")
        lines.append(f"  平均 tokens/千字：{total_tokens/(total_chars/1000):.0f}" if total_chars else "")
        lines.append(f"  检出错误总数：{total_errors}")

        # 按案例分类统计
        lines.append("\n  ### 按案例分类统计")
        lines.append(f"  {'案例':<6}{'类别':<22}{'应检出':<8}{'检出数':<8}{'状态':<14}{'tokens':<10}")
        for cid in sorted(set(r["case_id"] for r in rs)):
            cat, should = CASE_INFO.get(cid, ("未知", None))
            case_rs = [r for r in rs if r["case_id"] == cid]
            err_sum = sum(max(0, r["errors_count"]) for r in case_rs)
            tok_sum = sum(r["total_tokens"] for r in case_rs)
            if should is True and err_sum > 0:
                status = "正确检出"
            elif should is False and err_sum == 0:
                status = "正确未报"
            elif should is False and err_sum > 0:
                status = f"误报{err_sum}条"
            elif should is True and err_sum == 0:
                status = "漏报"
            else:
                status = "-"
            lines.append(f"  {cid:<6}{cat:<20}{'是' if should else '否':<6}{err_sum:<8}{status:<14}{tok_sum:<10}")

    # 模型对比
    lines.append("\n## 模型对比（同一文档）")
    lines.append(f"  {'案例':<6}{'文档':<40}{'flash tokens':<14}{'pro tokens':<12}{'flash错':<8}{'pro错':<8}")
    cases_docs = sorted(set((r["case_id"], r["doc_name"]) for r in all_results))
    for cid, dn in cases_docs:
        f = next((r for r in all_results if r["model"] == "flash" and r["case_id"] == cid and r["doc_name"] == dn), None)
        p = next((r for r in all_results if r["model"] == "pro" and r["case_id"] == cid and r["doc_name"] == dn), None)
        ft = f["total_tokens"] if f else "-"
        pt = p["total_tokens"] if p else "-"
        fe = f["errors_count"] if f else "-"
        pe = p["errors_count"] if p else "-"
        lines.append(f"  {cid:<6}{dn[:38]:<40}{ft!s:<14}{pt!s:<12}{fe!s:<8}{pe!s:<8}")

    # 错别字测试结果
    if typo_results:
        lines.append("\n" + "=" * 70)
        lines.append("隐蔽错别字测试结果")
        lines.append("=" * 70)
        for r in typo_results:
            lines.append(f"\n### {r['doc_name']} ({r['chars']}字, 共埋设 {r['typos_planted']} 处错别字)")
            for mk in MODELS:
                if mk not in r.get("models", {}):
                    continue
                m = r["models"][mk]
                lines.append(f"  [{mk}] 检出 {m['errors_count']} / {r['typos_planted']} 处, "
                             f"tokens={m['total_tokens']}, 耗时 {m['elapsed']}s")
                for e in m["errors"]:
                    lines.append(f"    - [{e['type']}/{e['confidence']}] {e['original']!r} → {e['suggestion']!r}")
                    lines.append(f"      理由：{e['reason']}")

    # 详细检出
    lines.append("\n" + "=" * 70)
    lines.append("各文档检出详情")
    lines.append("=" * 70)
    for r in all_results:
        cat, should = CASE_INFO.get(r["case_id"], ("未知", None))
        lines.append(f"\n## [{r['model']}] 案例 {r['case_id']}-{r['case_name']} / {r['doc_name']}")
        lines.append(f"  类别：{cat}（应检出：{'是' if should else '否'}）")
        lines.append(f"  字数：{r['chars']}，检出：{r['errors_count']} 条，"
                     f"tokens：{r['total_tokens']}（p={r['prompt_tokens']}, c={r['completion_tokens']}），"
                     f"耗时：{r['elapsed']}s")
        for i, e in enumerate(r["errors"][:20], 1):  # 最多列 20 条避免太长
            lines.append(f"  {i}. [{e['type']}/{e['confidence']}] {e['original']!r} → {e['suggestion']!r}")
            lines.append(f"     理由：{e['reason']}")
        if len(r["errors"]) > 20:
            lines.append(f"  ...（还有 {len(r['errors'])-20} 条省略）")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n报告已写出：{report_path}")
    print(f"汇总 CSV：{csv_path}")
    return report_path, csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="", help="逗号分隔的案例编号，留空=全部")
    parser.add_argument("--models", default="flash,pro", help="逗号分隔的模型，flash/pro")
    parser.add_argument("--no-review", action="store_true", help="关闭二次复核")
    parser.add_argument("--typos", action="store_true", help="只跑错别字测试")
    parser.add_argument("--with-typos", action="store_true", help="跑完案例后接着跑错别字测试，合并报告")
    args = parser.parse_args()

    cfg = config.bootstrap_from_dearesource()
    if not config.deepseek_configured(cfg):
        print("DeepSeek 未配置，终止")
        sys.exit(1)
    api_key = cfg["deepseek_api_key"]

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    with_review = not args.no_review

    if args.typos:
        # 仅跑错别字测试
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from _typo_generator import run_typo_test  # type: ignore
        typo_results = run_typo_test(api_key, models, with_review)
        write_report([], with_review, typo_results=typo_results)
        return

    # 选择案例
    all_cases = list_cases(DATASET_ROOT)
    if args.cases:
        wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
        all_cases = [c for c in all_cases if c[0] in wanted]
    print(f"将测试 {len(all_cases)} 个案例 × {len(models)} 个模型，二次复核：{'开' if with_review else '关'}")

    all_results = []
    for cid, cname, cdir in all_cases:
        cat, should = CASE_INFO.get(cid, ("未知", None))
        section(f"案例 {cid}：{cname}（{cat}，应检出：{'是' if should else '否'}）")
        for mk in models:
            rs = run_case_test(cid, cname, cdir, mk, api_key, with_review)
            all_results.extend(rs)

    typo_results = None
    if args.with_typos:
        section("隐蔽错别字测试")
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from _typo_generator import run_typo_test  # type: ignore
        typo_results = run_typo_test(api_key, models, with_review)

    write_report(all_results, with_review, typo_results=typo_results)


if __name__ == "__main__":
    main()
