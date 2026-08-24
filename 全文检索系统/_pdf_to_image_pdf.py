"""
将 OCR 版 PDF 转换为纯图片版 PDF。

原理：
  OCR 版 PDF = 扫描图片 + 叠加的可选中文字层。
  转换方法：把每一页渲染为高分辨率位图，再用位图重新组装成新 PDF。
  新 PDF 里只有图片，没有文字层，因此"变回了图片版本"。

依赖：
  pip install pymupdf pillow
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError as e:
    raise RuntimeError("请先安装 PyMuPDF：pip install pymupdf") from e

try:
    from PIL import Image
except ImportError as e:
    raise RuntimeError("请先安装 Pillow：pip install pillow") from e


# DPI：越大越清晰，但文件体积也越大。200 DPI 对中文县志/年鉴足够清晰。
DEFAULT_DPI = 200


def pdf_to_image_pdf(src_path: str | os.PathLike,
                     dst_path: str | os.PathLike,
                     dpi: int = DEFAULT_DPI,
                     overwrite: bool = False) -> dict:
    """把 src_path 的 PDF 渲染为纯图片版 PDF，写到 dst_path。

    返回统计信息字典：页数、源/目标文件大小。
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    if not src_path.exists():
        raise FileNotFoundError(f"源文件不存在：{src_path}")
    if dst_path.exists() and not overwrite:
        raise FileExistsError(f"目标文件已存在：{dst_path}（使用 -y 覆盖）")

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. 打开源 PDF，逐页渲染为 PIL.Image
    zoom = dpi / 72.0  # PDF 原生 72 DPI，zoom = 缩放比
    mat = fitz.Matrix(zoom, zoom)

    rendered_images: list[Image.Image] = []
    doc = fitz.open(src_path)
    try:
        total = len(doc)
        for i, page in enumerate(doc, 1):
            pix = page.get_pixmap(matrix=mat, alpha=False)
            # 把 pixmap 字节直接转成 PIL，避免写临时 PNG
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            rendered_images.append(img)
            if i % 10 == 0 or i == total:
                print(f"  渲染进度: {i}/{total} 页", flush=True)
    finally:
        doc.close()

    if not rendered_images:
        raise RuntimeError("PDF 没有可渲染的页面")

    # 2. 用 Pillow 把图片列表保存为新 PDF
    first, rest = rendered_images[0], rendered_images[1:]
    first.save(dst_path, "PDF", resolution=dpi, save_all=True, append_images=rest)

    src_size = src_path.stat().st_size
    dst_size = dst_path.stat().st_size
    return {
        "pages": len(rendered_images),
        "src_size_mb": round(src_size / 1024 / 1024, 2),
        "dst_size_mb": round(dst_size / 1024 / 1024, 2),
    }


def iter_ocr_pdfs(root: Path) -> list[Path]:
    """在 root 下递归找出所有 .pdf（默认处理全部，调用方也可只挑 OCR 标记的）。"""
    return sorted(p for p in root.rglob("*.pdf") if p.is_file())


def main():
    parser = argparse.ArgumentParser(
        description="把 OCR 版 PDF 转成纯图片版 PDF（去掉可选中文字层）"
    )
    parser.add_argument("input", nargs="?", help="输入文件或目录（目录则递归处理全部 PDF）")
    parser.add_argument("-o", "--output", help="输出文件或目录（默认与输入同目录，加 _纯图片 后缀）")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help=f"渲染 DPI，默认 {DEFAULT_DPI}")
    parser.add_argument("-y", "--overwrite", action="store_true", help="目标存在时直接覆盖")
    parser.add_argument("--ocr-only", action="store_true",
                        help="只处理文件名里包含 'OCR' 的 PDF（目录模式下默认开启）")
    parser.add_argument("--all", action="store_true",
                        help="目录模式下处理所有 PDF，不限 OCR 字样")
    args = parser.parse_args()

    # 无参数时，使用项目约定的"样本数据"目录
    if not args.input:
        default_root = Path(__file__).resolve().parent / "样本数据"
        if default_root.exists():
            args.input = str(default_root)
            print(f"未指定输入，自动使用样本数据目录：{default_root}")
        else:
            parser.print_help()
            sys.exit(1)

    input_path = Path(args.input).resolve()

    # ---------- 单个文件 ----------
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            print("错误：单个文件必须是 .pdf")
            sys.exit(1)
        dst = Path(args.output) if args.output else input_path.with_name(
            input_path.stem + "_纯图片.pdf"
        )
        print(f"处理单文件：{input_path.name}")
        print(f"输出到：{dst}")
        stat = pdf_to_image_pdf(input_path, dst, dpi=args.dpi, overwrite=args.overwrite)
        print(f"完成：{stat['pages']} 页 | 源 {stat['src_size_mb']} MB -> 目标 {stat['dst_size_mb']} MB")
        return

    # ---------- 目录模式 ----------
    if not input_path.is_dir():
        print(f"错误：输入路径不存在：{input_path}")
        sys.exit(1)

    output_dir = Path(args.output).resolve() if args.output else input_path
    pdfs = iter_ocr_pdfs(input_path)

    # OCR 过滤：默认目录模式下只处理文件名含 OCR 的，除非 --all
    ocr_only = args.ocr_only or (not args.all)
    if ocr_only:
        before = len(pdfs)
        pdfs = [p for p in pdfs if "ocr" in p.name.lower()]
        print(f"目录模式（仅 OCR 标记）：过滤 {before - len(pdfs)} 个非 OCR PDF，剩余 {len(pdfs)} 个")
    else:
        print(f"目录模式（全部 PDF）：共 {len(pdfs)} 个")

    if not pdfs:
        print("没有可处理的 PDF，退出。")
        return

    total_pages = 0
    ok = 0
    failed: list[tuple[str, str]] = []
    for i, src in enumerate(pdfs, 1):
        # 保持相对目录结构输出到 output_dir
        rel = src.relative_to(input_path)
        dst = output_dir / rel.with_name(rel.stem + "_纯图片.pdf")

        print(f"\n[{i}/{len(pdfs)}] {rel}")
        print(f"  输出：{dst}")
        try:
            stat = pdf_to_image_pdf(src, dst, dpi=args.dpi, overwrite=args.overwrite)
        except Exception as e:
            print(f"  失败：{e}")
            failed.append((str(rel), str(e)))
            continue

        total_pages += stat["pages"]
        ok += 1
        print(f"  OK {stat['pages']} 页 | {stat['src_size_mb']}MB -> {stat['dst_size_mb']}MB")

    print(f"\n===== 汇总 =====")
    print(f"成功：{ok}/{len(pdfs)} 个 | 总页数：{total_pages}")
    if failed:
        print(f"失败 {len(failed)} 个：")
        for name, err in failed:
            print(f"  - {name}: {err}")
    print(f"输出目录：{output_dir}")


if __name__ == "__main__":
    main()
