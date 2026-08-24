# -*- coding: utf-8 -*-
"""OpenAI 兼容 API 客户端（默认 DeepSeek，可对接任意 OpenAI 兼容服务商）"""
import json

import requests

from .models import TokenUsage

BASE_URL = "https://api.deepseek.com"
# deepseek-chat / deepseek-reasoner 将于 2026-07-24 停用，改用 V4 命名
MODEL = "deepseek-v4-flash"


class DeepSeekError(Exception):
    """DeepSeek 调用异常（带中文说明）"""


class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str = BASE_URL, model: str = MODEL, timeout: int = 300):
        if not api_key:
            raise DeepSeekError("DeepSeek API key 未配置")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout  # 16k tokens 输出可能要 90-180s，给 300s 余量

    def chat(self, messages: list, temperature: float = 0.1, max_tokens: int = 16384):
        """调用 /chat/completions（强制 JSON 输出），返回 (content, TokenUsage)

        max_tokens 默认 16384：纠错任务单块常含 10-20 条错误，每条带 quote/original/
        suggestion/reason，输出 8-12k tokens 很常见；8192 时长文本会被截断导致
        JSON 解析失败。DeepSeek V4 上限 16384，配合 _parse_response 截断容错双保险。
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
        except requests.Timeout:
            raise DeepSeekError("DeepSeek 请求超时，请检查网络后重试")
        except requests.RequestException as e:
            raise DeepSeekError(f"DeepSeek 请求网络错误: {e}")

        if resp.status_code == 401:
            raise DeepSeekError("DeepSeek 鉴权失败（401）：API key 无效，请检查配置")
        if resp.status_code != 200:
            raise DeepSeekError(f"DeepSeek 请求失败（HTTP {resp.status_code}）：{resp.text[:300]}")

        try:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError) as e:
            raise DeepSeekError(f"DeepSeek 响应解析失败: {e}；原始响应: {resp.text[:300]}")

        usage_raw = data.get("usage", {})
        usage = TokenUsage(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
            total=usage_raw.get("total_tokens", 0),
        )
        return content, usage
