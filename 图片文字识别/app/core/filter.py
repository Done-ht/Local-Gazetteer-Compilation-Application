"""双层漏斗式过滤模块。

需求：
  先用 OpenCV 提取边缘密度、连通域形状、灰度方差三个统计特征，
  毫秒级排除纯图；然后仅对疑似图片跑 PaddleOCR 轻量检测模型
  （只检测不识别）做最终确认。

设计：
  Layer1（OpenCV）：纯统计特征，CPU 毫秒级，零依赖大模型。
  Layer2（PaddleOCR det）：仅加载 4.1MB PP-OCRv4 检测模型，
                           有框才调讯飞 OCR，无框直接跳过。

返回 (should_ocr: bool, reason: str)，reason 用于日志输出。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    """过滤配置。"""

    edge_density_threshold: float = 0.03
    # 黑白文字页面的边缘密度阈值（自适应：低饱和度占比高时启用）
    # 淡色扫描文字边缘可能较稀疏，阈值需更低
    edge_density_threshold_text: float = 0.01
    variance_threshold: float = 15.0
    connected_min_area: int = 30
    # 饱和度特征：判断页面是黑白文字 vs 彩色图片
    # 低饱和度像素（S<30）占比超过此阈值视为黑白文字页面
    low_sat_ratio_threshold: float = 0.9
    enable_layer2: bool = True


class DualLayerFilter:
    """双层漏斗过滤器。"""

    def __init__(
        self,
        config: FilterConfig,
        # PaddleOCR 检测器（仅 detect_boxes 方法），可选
        detector: Optional[object] = None,
    ) -> None:
        self.cfg = config
        self.detector = detector

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def should_ocr(self, image: np.ndarray, slot: int = 0) -> Tuple[bool, str]:
        """判断该图片是否需要走 OCR。

        参数:
            image: BGR numpy 数组
            slot: 实例池槽位号，传给 detector 选择对应 PaddleOCR 实例（默认 0）

        返回 (是否需要 OCR, 原因说明)。
        """
        # Layer 1：OpenCV 统计特征
        ok, reason = self._layer1_opencv(image)
        if not ok:
            return False, reason
        # Layer 2：PaddleOCR 检测
        if self.cfg.enable_layer2 and self.detector is not None:
            ok, reason = self._layer2_paddle(image, slot)
            if not ok:
                return False, reason
        return True, "通过双层过滤"

    # ------------------------------------------------------------------
    # Layer 1：OpenCV 统计特征
    # ------------------------------------------------------------------
    def _layer1_opencv(self, image: np.ndarray) -> Tuple[bool, str]:
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image.ndim == 3
            else image
        )
        # 特征 1：灰度方差 —— 纯色/渐变背景方差极低
        variance = float(gray.var())
        if variance < self.cfg.variance_threshold:
            return False, f"L1 拒绝: 灰度方差 {variance:.1f} < {self.cfg.variance_threshold}"

        # 特征 2：饱和度 —— 区分黑白文字页面 vs 彩色图片
        # 文字页面（白底黑字扫描/打印）饱和度极低；彩色风景图饱和度较高
        # 用低饱和度像素占比判断页面类型，自适应调整边缘密度阈值
        sat_low_ratio = 1.0  # 单通道图默认全部低饱和度
        sat_mean = 0.0
        if image.ndim == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            sat = hsv[:, :, 1]
            sat_mean = float(sat.mean())
            sat_low_ratio = float(np.sum(sat < 30)) / sat.size
        is_text_like = sat_low_ratio > self.cfg.low_sat_ratio_threshold

        # 特征 3：边缘密度 —— Canny 边缘像素占比
        # 自适应阈值：黑白文字页面用更低阈值（淡色扫描文字边缘稀疏），
        # 彩色图片用原阈值（彩色平滑图片大概率是风景图）
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / edges.size
        edge_threshold = (
            self.cfg.edge_density_threshold_text
            if is_text_like
            else self.cfg.edge_density_threshold
        )
        if edge_density < edge_threshold:
            page_type = "文字" if is_text_like else "彩色"
            return False, (
                f"L1 拒绝: 边缘密度 {edge_density:.4f} < {edge_threshold} "
                f"({page_type}页, 饱和度均={sat_mean:.1f}, 低饱和占比={sat_low_ratio:.2f})"
            )

        # 特征 4：连通域形状 —— 文字区域通常产生大量中小连通域
        # 使用二值化后的连通域分析
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        n_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        # stats[0] 是背景，排除
        if n_labels <= 1:
            return False, "L1 拒绝: 无前景连通域"
        areas = stats[1:, cv2.CC_STAT_AREA]
        # 统计满足最小面积的中小连通域数量
        small_count = int(np.sum(areas < self.cfg.connected_min_area))
        valid_count = int(np.sum(areas >= self.cfg.connected_min_area))
        # 文字图片通常 valid_count 较多；纯风景图大块连通域为主
        if valid_count < 3 and small_count < 20:
            return False, (
                f"L1 拒绝: 有效连通域 {valid_count} 个，小连通域 {small_count} 个"
            )
        return True, (
            f"L1 通过: 方差={variance:.1f}, 边缘={edge_density:.4f}, "
            f"饱和度均={sat_mean:.1f}(低占比{sat_low_ratio:.2f}), 连通域={valid_count}"
        )

    # ------------------------------------------------------------------
    # Layer 2：PaddleOCR 检测（只检测不识别）
    # ------------------------------------------------------------------
    def _layer2_paddle(self, image: np.ndarray, slot: int = 0) -> Tuple[bool, str]:
        try:
            boxes = self.detector.detect_boxes(image, slot=slot)  # type: ignore[union-attr]
        except Exception as e:
            logger.warning("L2 PaddleOCR 检测异常，放行交由后续 OCR: %s", e)
            return True, f"L2 异常放行: {e}"
        # None = 检测器不可用（初始化失败），放行不阻断
        if boxes is None:
            return True, "L2 跳过: PaddleOCR 检测器不可用，放行"
        # [] = 确实没检测到文字框，拒绝
        if not boxes:
            return False, "L2 拒绝: PaddleOCR 未检测到文字框"
        return True, f"L2 通过: 检测到 {len(boxes)} 个文字框"
