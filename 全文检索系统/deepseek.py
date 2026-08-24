"""DeepSeek API 客户端封装（独立模块，仅依赖 Python 标准库）。

兼容 OpenAI Chat Completions 协议，支持普通响应与流式响应（SSE）。

模型常量（2026-07 起 V4 系列，旧 deepseek-chat 已停用）：
    V4_FLASH  = "deepseek-v4-flash"   非思考模式，快速、低成本
    V4_PRO    = "deepseek-v4-pro"     思考模式，更强推理

用法：
    from deepseek import DeepSeekClient, V4_PRO

    client = DeepSeekClient(api_key="sk-xxx", model=V4_PRO)

    # 1. 普通对话
    resp = client.chat([{"role": "user", "content": "你好"}])
    print(resp["content"])

    # 2. 流式对话
    for chunk in client.chat_stream([{"role": "user", "content": "讲个故事"}]):
        if chunk["type"] == "content":
            print(chunk["delta"], end="", flush=True)

    # 3. 带上下文问答
    answer = client.ask("郎溪县的农业区划是怎样的？", context="...相关资料...")
"""
from __future__ import annotations

import json
import random
import socket
import time
import urllib.request
import urllib.error
from typing import Iterator, List, Dict, Optional, Any


# ============================================================
#  模型常量
# ============================================================
V4_FLASH = "deepseek-v4-flash"   # 非思考模式，快速
V4_PRO = "deepseek-v4-pro"       # 思考模式，推理更强

DEFAULT_MODEL = V4_FLASH
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_TIMEOUT = 60         # 普通请求
DEFAULT_STREAM_TIMEOUT = 300  # 流式请求（思考模型可能较慢）

# 失败自动重试（指数退避 + 随机抖动）。仅对瞬时性错误生效：
#   - NetworkError（连接失败、超时）
#   - APIError 且状态码在 RETRYABLE_STATUS_CODES 中（限流、网关类错误）
# 认证失败(401)、参数错误(400)等不可重试，立即抛出。
# 流式请求只在"尚未输出任何事件"时重试，避免调用方收到重复内容。
DEFAULT_MAX_RETRIES = 3          # 失败后最多重试次数（不含首次请求）
DEFAULT_RETRY_BASE_DELAY = 1.0   # 首次重试的延迟（秒），之后指数翻倍
RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})

# 可选模型清单（供 UI 选择）
# 只保留 DeepSeek 的两个预设（模型号变动频繁，其他模型由用户自行填写模型 ID）。
# 所有模型均需兼容 OpenAI Chat Completions 协议。
AVAILABLE_MODELS = [
    {"id": V4_FLASH, "name": "DeepSeek V4 Flash", "desc": "非思考 · 快速 · 低成本",
     "provider": "deepseek", "base_url": "https://api.deepseek.com"},
    {"id": V4_PRO, "name": "DeepSeek V4 Pro", "desc": "思考模式 · 推理更强",
     "provider": "deepseek", "base_url": "https://api.deepseek.com"},
]

# 常见厂商的 API 地址（供 UI 提供快捷填充，模型 ID 由用户自填）
PROVIDER_BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "openai":   "https://api.openai.com/v1",
    "kimi":     "https://api.moonshot.cn/v1",
    "glm":      "https://open.bigmodel.cn/api/paas/v4",
    "qwen":     "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "ernie":    "https://qianfan.baidubce.com/v2",
}


class DeepSeekError(Exception):
    """DeepSeek 调用异常基类。"""


class APIError(DeepSeekError):
    """API 返回的错误（含状态码与原始信息）。"""
    def __init__(self, message: str, status: int = 0, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


class NetworkError(DeepSeekError):
    """网络层错误（连接失败、超时等）。"""


class DeepSeekClient:
    """DeepSeek API 客户端。

    仅依赖标准库，使用 urllib 发起请求。
    线程安全：无实例可变状态（构造后即可在多线程中使用）。
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        stream_timeout: int = DEFAULT_STREAM_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    ):
        if not api_key:
            raise ValueError("api_key 不能为空")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.stream_timeout = stream_timeout
        self.max_retries = max(0, int(max_retries))
        self.retry_base_delay = max(0.0, float(retry_base_delay))

    # ----------------------------------------------------------
    #  普通对话
    # ----------------------------------------------------------
    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **extra,
    ) -> Dict[str, Any]:
        """发起一次 chat completions 请求，返回结构化结果。

        返回：
            {
                "content": "回复文本",
                "reasoning": "思考过程（若模型为思考模式，否则为空）",
                "model": "实际使用的模型",
                "usage": {...},
                "raw": {原始 JSON},
            }
        """
        payload = self._build_payload(
            messages, model=model, temperature=temperature,
            max_tokens=max_tokens, stream=False, **extra
        )
        raw = self._post_json_with_retry("/chat/completions", payload)
        return self._parse_completion(raw)

    # ----------------------------------------------------------
    #  流式对话
    # ----------------------------------------------------------
    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **extra,
    ) -> Iterator[Dict[str, Any]]:
        """流式 chat completions，逐 chunk 返回。

        当 extra 中包含 tools 时，会累积 tool_calls 分片，
        流结束后 yield {"type": "tool_calls", "tool_calls": [...]} 事件。

        每个 chunk 形如：
            {"type": "reasoning", "delta": "思考片段"}  # 思考模式才有
            {"type": "content", "delta": "回复片段"}
            {"type": "tool_calls", "tool_calls": [...]}  # 仅当模型调用工具时，流结束前 yield
            {"type": "done", "usage": {...} or None}
        """
        payload = self._build_payload(
            messages, model=model, temperature=temperature,
            max_tokens=max_tokens, stream=True, **extra
        )
        has_tools = "tools" in extra
        # 累积 tool_calls 分片：index -> {id, type, function: {name, arguments}}
        accumulated_tool_calls: Dict[int, Dict[str, Any]] = {}

        for event in self._post_stream_with_retry("/chat/completions", payload):
            etype = event.get("type")
            if has_tools and etype == "tool_calls_delta":
                # 累积 tool_calls 分片（按 index 拼接）
                for tc in event["delta"]:
                    idx = tc.get("index", 0)
                    if idx not in accumulated_tool_calls:
                        accumulated_tool_calls[idx] = {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    acc = accumulated_tool_calls[idx]
                    if tc.get("id"):
                        acc["id"] = tc["id"]
                    if tc.get("type"):
                        acc["type"] = tc["type"]
                    fn = tc.get("function", {})
                    if fn.get("name"):
                        acc["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        acc["function"]["arguments"] += fn["arguments"]
            elif etype in ("finish", "done"):
                # 流结束：如果有累积的 tool_calls，先 yield 出去
                if accumulated_tool_calls:
                    tool_calls_list = [
                        accumulated_tool_calls[i]
                        for i in sorted(accumulated_tool_calls.keys())
                    ]
                    yield {"type": "tool_calls", "tool_calls": tool_calls_list}
                yield event
            else:
                yield event

    # ----------------------------------------------------------
    #  便捷问答
    # ----------------------------------------------------------
    def ask(
        self,
        question: str,
        context: str = "",
        system: str = "",
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
    ) -> str:
        """单轮问答便捷封装，返回纯文本答案。

        Args:
            question: 用户问题
            context:  提供给模型的背景资料（如检索到的片段）
            system:   自定义 system prompt（默认根据是否有 context 自动构造）
        """
        messages = self._build_qa_messages(question, context, system)
        resp = self.chat(messages, model=model, temperature=temperature,
                         max_tokens=max_tokens)
        return resp["content"]

    def ask_stream(
        self,
        question: str,
        context: str = "",
        system: str = "",
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        """流式问答，与 chat_stream 同样的 chunk 格式。"""
        messages = self._build_qa_messages(question, context, system)
        yield from self.chat_stream(
            messages, model=model, temperature=temperature,
            max_tokens=max_tokens,
        )

    # ----------------------------------------------------------
    #  连通性测试
    # ----------------------------------------------------------
    def ping(self) -> Dict[str, Any]:
        """发一句最小对话，验证 api_key 与模型可用。

        返回 {"ok": True, "model": ..., "reply": "..."}，
        失败抛 DeepSeekError。
        """
        resp = self.chat(
            [{"role": "user", "content": "ping"}],
            max_tokens=16,
            temperature=0,
        )
        return {"ok": True, "model": resp["model"], "reply": resp["content"]}

    # ============================================================
    #  内部实现
    # ============================================================
    def _build_qa_messages(self, question: str, context: str, system: str) -> List[Dict]:
        if not system:
            if context:
                system = (
                    "你是一名严谨的资料分析助手。请仅根据下方提供的【参考资料】回答问题。"
                    "若资料不足以回答，请明确说明「资料中未提及」，不要编造内容。"
                    "回答时在关键信息后用 [n] 标注引用的第 n 条资料。"
                )
            else:
                system = "你是一名严谨的资料分析助手。请清晰、准确地回答问题。"
        messages: List[Dict] = [{"role": "system", "content": system}]
        if context:
            messages.append({
                "role": "user",
                "content": f"【参考资料】\n{context}\n\n【问题】\n{question}",
            })
        else:
            messages.append({"role": "user", "content": question})
        return messages

    def _build_payload(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        stream: bool,
        **extra,
    ) -> Dict[str, Any]:
        # 防御性清洗：移除 assistant 消息中空的 tool_calls 字段。
        # OpenAI/DeepSeek API 协议要求：tool_calls 若存在必须是非空数组，
        # 空数组会触发 "Invalid messages[N].tool_calls: empty array" 校验错误。
        # 此处作为最后一道防线，兜底处理上游代码可能的回归。
        sanitized_messages = []
        for m in messages:
            if m.get("role") == "assistant" and isinstance(m.get("tool_calls"), list):
                if not m["tool_calls"]:
                    # 空 tool_calls：移除该字段，仅保留 content
                    m = {k: v for k, v in m.items() if k != "tool_calls"}
            sanitized_messages.append(m)
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": sanitized_messages,
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(extra)
        return payload

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    # ----------------------------------------------------------
    #  失败重试（指数退避）
    # ----------------------------------------------------------
    def _is_retryable(self, exc: Exception) -> bool:
        """判断异常是否为瞬时性错误、值得重试。"""
        if isinstance(exc, NetworkError):
            return True
        if isinstance(exc, APIError):
            return exc.status in RETRYABLE_STATUS_CODES
        return False

    def _retry_delay(self, attempt: int) -> float:
        """第 attempt 次重试（从 1 起）前的等待秒数：指数退避 + 随机抖动。"""
        return self.retry_base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)

    def _post_json_with_retry(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """非流式请求 + 可重试错误的指数退避重试。"""
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                time.sleep(self._retry_delay(attempt))
            try:
                return self._post_json(path, payload)
            except DeepSeekError as e:
                if attempt >= self.max_retries or not self._is_retryable(e):
                    raise
        raise APIError("unreachable")  # 防御：循环必然 return 或 raise

    def _post_stream_with_retry(
        self, path: str, payload: Dict[str, Any]
    ) -> Iterator[Dict[str, Any]]:
        """流式请求 + 连接期重试。

        仅在尚未产出任何事件（连接建立/首个数据块失败）时重试；
        一旦开始输出，中途失败直接抛出——重试会导致调用方收到重复内容。
        """
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                time.sleep(self._retry_delay(attempt))
            produced = False
            try:
                for event in self._post_stream(path, payload):
                    produced = True
                    yield event
                return
            except DeepSeekError as e:
                if produced or attempt >= self.max_retries or not self._is_retryable(e):
                    raise

    # ----------------------------------------------------------
    #  HTTP 请求（单次，不带重试）
    # ----------------------------------------------------------
    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self.base_url + path
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                pass
            msg = self._extract_error_message(body) or f"HTTP {e.code}"
            raise APIError(msg, status=e.code, body=body)
        except urllib.error.URLError as e:
            raise NetworkError(f"连接 DeepSeek 失败: {e.reason}")
        except TimeoutError:
            raise NetworkError("请求超时")

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            raise APIError(f"响应非 JSON: {raw[:200]}", body=raw)

    def _post_stream(self, path: str, payload: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        url = self.base_url + path
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers=self._headers(),
        )
        try:
            resp = urllib.request.urlopen(req, timeout=self.stream_timeout)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                pass
            msg = self._extract_error_message(body) or f"HTTP {e.code}"
            raise APIError(msg, status=e.code, body=body)
        except urllib.error.URLError as e:
            raise NetworkError(f"连接 DeepSeek 失败: {e.reason}")
        except TimeoutError:
            raise NetworkError("请求超时")

        try:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="ignore").rstrip("\r\n")
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    yield {"type": "done", "usage": None}
                    return
                try:
                    obj = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                for event in self._parse_stream_chunk(obj):
                    yield event
            # 流自然结束（未见 [DONE]）
            yield {"type": "done", "usage": None}
        except (TimeoutError, socket.timeout) as e:
            # 读取阶段超时（与连接阶段的 TimeoutError 处理一致，转为 NetworkError）
            raise NetworkError(f"流式读取超时: {e}")
        finally:
            try:
                resp.close()
            except Exception:
                pass

    # ----------------------------------------------------------
    #  响应解析
    # ----------------------------------------------------------
    def _parse_completion(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """解析非流式响应。"""
        choices = raw.get("choices") or []
        content = ""
        reasoning = ""
        tool_calls = []
        finish_reason = ""
        if choices:
            msg = choices[0].get("message", {})
            content = msg.get("content", "") or ""
            # 思考模式可能有 reasoning_content 字段
            reasoning = msg.get("reasoning_content", "") or ""
            # 工具调用（function calling）
            tool_calls = msg.get("tool_calls", []) or []
            finish_reason = choices[0].get("finish_reason", "") or ""
        return {
            "content": content,
            "reasoning": reasoning,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
            "model": raw.get("model", ""),
            "usage": raw.get("usage", {}),
            "raw": raw,
        }

    def _parse_stream_chunk(self, obj: Dict[str, Any]) -> List[Dict[str, Any]]:
        """解析单个 SSE 数据块，可能产生 0~3 个事件。"""
        events: List[Dict[str, Any]] = []
        choices = obj.get("choices") or []
        if not choices:
            return events
        delta = choices[0].get("delta", {}) or {}
        # 思考片段（思考模式）
        reasoning_delta = delta.get("reasoning_content", "")
        if reasoning_delta:
            events.append({"type": "reasoning", "delta": reasoning_delta})
        # 正文片段
        content_delta = delta.get("content", "")
        if content_delta:
            events.append({"type": "content", "delta": content_delta})
        # 工具调用分片（流式 function calling，需由 chat_stream 累积拼接）
        tool_calls_delta = delta.get("tool_calls")
        if tool_calls_delta:
            events.append({"type": "tool_calls_delta", "delta": tool_calls_delta})
        finish = choices[0].get("finish_reason")
        if finish:
            events.append({
                "type": "finish",
                "finish_reason": finish,
                "usage": obj.get("usage"),
            })
        return events

    def _extract_error_message(self, body: str) -> str:
        """从 API 错误响应体中提取错误信息。"""
        if not body:
            return ""
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            return ""
        # OpenAI 风格：{"error": {"message": "..."}}
        err = obj.get("error")
        if isinstance(err, dict):
            return err.get("message", "") or ""
        if isinstance(err, str):
            return err
        # 直接 message
        if obj.get("message"):
            return obj["message"]
        return ""
