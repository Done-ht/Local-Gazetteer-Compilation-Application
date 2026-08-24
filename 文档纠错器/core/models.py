# -*- coding: utf-8 -*-
"""DocProof 核心数据模型"""
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Page:
    """一页文本（page_num 从 1 起）"""
    page_num: int
    text: str


@dataclass
class ErrorItem:
    """一条检出的错误；offset 为所在页文本内的局部偏移"""
    id: str
    page_num: int
    offset_start: int
    offset_end: int
    original: str
    suggestion: str
    error_type: str  # 错别字 / 语法 / 标点 / 逻辑
    reason: str
    confidence: str = "明确"  # 明确 / 存疑


@dataclass
class ConfirmRecord:
    """一次纠错确认记录（快照，便于撤销/审计）"""
    error: ErrorItem
    batch_id: str
    confirm_time: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


@dataclass
class TokenUsage:
    """一次/累计调用的 token 消耗"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total: int = 0

    def add(self, other: "TokenUsage") -> "TokenUsage":
        """累加另一个 TokenUsage，返回自身"""
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total += other.total
        return self
