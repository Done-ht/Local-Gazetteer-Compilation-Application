"""OCR 提供商基类与数据结构。

所有 Provider（讯飞 / 本地 PaddleOCR / 未来百度腾讯等）均实现此接口，
保证上层流水线与具体厂商解耦。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class OCRLine:
    """单行识别结果。"""

    text: str
    # 4 个顶点坐标 [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]，左上角起顺时针
    coords: List[Tuple[int, int]] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """返回 (x_min, y_min, x_max, y_max) 包围盒。"""
        if not self.coords:
            return (0, 0, 0, 0)
        xs = [p[0] for p in self.coords]
        ys = [p[1] for p in self.coords]
        return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class OCRResult:
    """单张图片的 OCR 结果。"""

    lines: List[OCRLine] = field(default_factory=list)
    width: int = 0
    height: int = 0
    # 原始返回（供调试 / JSON 输出）
    raw: Optional[dict] = None
    # 提供商名称
    provider: str = ""
    # 逻辑行分组：每个子列表是同一逻辑行（如跨行标题）的物理行索引。
    # to_txt / to_markdown 等文本输出据此把同组物理行合并为一行；
    # searchable PDF 的文字层不合并，仍按物理行坐标定位，保证选中位置准确。
    line_groups: List[List[int]] = field(default_factory=list)
    # 表格区域列表（启用 SLANet 表格识别后填充）
    # 每项: {"bbox": [x1,y1,x2,y2], "html": "<table>...</table>"}
    # bbox 为原图像素坐标，html 为 SLANet 识别的表格 HTML 结构
    # PDF 渲染时用 insert_htmlbox 渲染表格，文字层跳过该区域避免重复
    tables: List[dict] = field(default_factory=list)

    @property
    def text(self) -> str:
        """拼接全部文本（按行），跨行标题等逻辑行合并为一行。"""
        if not self.line_groups:
            return "\n".join(line.text for line in self.lines)
        # 行索引 -> 组号；组 -> 合并后文本
        line_to_group: dict = {}
        group_text: dict = {}
        for gi, group in enumerate(self.line_groups):
            group_text[gi] = "".join(
                self.lines[i].text for i in group if 0 <= i < len(self.lines)
            )
            for i in group:
                line_to_group[i] = gi
        # 按物理行顺序输出：遇到组的首行输出合并文本，跳过组内其余行
        parts: List[str] = []
        seen_groups: set = set()
        for i, line in enumerate(self.lines):
            if i in line_to_group:
                gi = line_to_group[i]
                if gi not in seen_groups:
                    parts.append(group_text[gi])
                    seen_groups.add(gi)
            else:
                parts.append(line.text)
        return "\n".join(p for p in parts if p)


class BaseProvider:
    """OCR 提供商抽象基类。"""

    name: str = "base"

    def recognize(self, image: np.ndarray) -> OCRResult:
        """对一张图片执行 OCR 识别，返回 OCRResult。

        参数:
            image: BGR 或 RGB numpy 数组 (H, W, 3)
        """
        raise NotImplementedError

    def is_available(self) -> bool:
        """该提供商是否可用（依赖已安装 / 凭证已配置）。"""
        return True
