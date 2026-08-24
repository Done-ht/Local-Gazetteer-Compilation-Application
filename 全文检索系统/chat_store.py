"""智能问答会话持久化存储。

存储位置：base_dir/_chat_sessions.json
结构：
{
  "sessions": [
    {
      "id": "sess_20260723_abc123",
      "title": "郎溪县的经济发展",          # 自动取首条问题前 20 字
      "created_at": "2026-07-23T...",
      "updated_at": "2026-07-23T...",
      "mode": "agent",                    # direct | agent
      "libraries": ["郎溪县志"],           # 该会话绑定的库
      "messages": [
        {
          "id": "msg_xxx",
          "role": "user",
          "content": "介绍一下郎溪县的经济发展",
          "timestamp": "..."
        },
        {
          "id": "msg_xxx",
          "role": "assistant",
          "content": "根据档案资料...",     # 最终答案（不含思考过程）
          "reasoning": "思考过程...",       # 思考模式才有
          "references": [...],             # 引用列表
          "retrieval": {...},              # 检索统计
          "mode": "agent",                 # 该消息使用的检索模式
          "queries": [...],                # Agent 模式的查询词
          "timestamp": "..."
        }
      ]
    }
  ]
}

线程安全：每次读写都从磁盘加载/保存，避免多请求下数据不一致。
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional


SESSIONS_FILENAME = "_chat_sessions.json"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _gen_id(prefix: str = "sess") -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"{prefix}_{ts}_{short}"


class ChatStore:
    """会话存储，线程安全。"""

    def __init__(self, base_dir: str):
        self._base_dir = base_dir
        self._path = os.path.join(base_dir, SESSIONS_FILENAME)
        self._lock = threading.Lock()

    @property
    def path(self) -> str:
        return self._path

    def _load(self) -> Dict[str, Any]:
        try:
            with open(self._path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            if isinstance(data, dict) and "sessions" in data:
                return data
        except (OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError):
            pass
        return {"sessions": []}

    def _save(self, data: Dict[str, Any]) -> None:
        os.makedirs(self._base_dir, exist_ok=True)
        tmp = self._path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self._path)

    # ----------------------------------------------------------
    #  会话级操作
    # ----------------------------------------------------------
    def list_sessions(self, owner: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出会话（按 updated_at 降序），不含 messages 详情。

        owner=None 返回全部；否则仅返回该属主的会话（旧会话无 owner 视为游客会话）。
        """
        with self._lock:
            data = self._load()
        sessions = []
        for s in data.get("sessions", []):
            s_owner = s.get("owner") or "guest"
            if owner is not None and s_owner != owner:
                continue
            sessions.append({
                "id": s.get("id", ""),
                "title": s.get("title", ""),
                "created_at": s.get("created_at", ""),
                "updated_at": s.get("updated_at", ""),
                "mode": s.get("mode", "direct"),
                "libraries": s.get("libraries", []),
                "message_count": len(s.get("messages", [])),
            })
        sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return sessions

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            data = self._load()
        for s in data.get("sessions", []):
            if s.get("id") == session_id:
                return s
        return None

    def session_owner(self, session_id: str) -> str:
        """返回会话属主（旧会话无 owner 视为游客会话）。"""
        s = self.get_session(session_id)
        if s is None:
            return ""
        return s.get("owner") or "guest"

    def create_session(
        self,
        title: str = "",
        mode: str = "direct",
        libraries: Optional[List[str]] = None,
        owner: str = "guest",
    ) -> Dict[str, Any]:
        session = {
            "id": _gen_id("sess"),
            "title": title or "新会话",
            "created_at": _now(),
            "updated_at": _now(),
            "mode": mode,
            "libraries": libraries or [],
            "owner": owner or "guest",
            "messages": [],
        }
        with self._lock:
            data = self._load()
            data.setdefault("sessions", []).append(session)
            self._save(data)
        return session

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            data = self._load()
            before = len(data.get("sessions", []))
            data["sessions"] = [
                s for s in data.get("sessions", [])
                if s.get("id") != session_id
            ]
            if len(data["sessions"]) < before:
                self._save(data)
                return True
        return False

    def update_session_meta(
        self, session_id: str,
        title: Optional[str] = None,
        mode: Optional[str] = None,
        libraries: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            data = self._load()
            for s in data.get("sessions", []):
                if s.get("id") == session_id:
                    if title is not None:
                        s["title"] = title
                    if mode is not None:
                        s["mode"] = mode
                    if libraries is not None:
                        s["libraries"] = libraries
                    s["updated_at"] = _now()
                    self._save(data)
                    return s
        return None

    # ----------------------------------------------------------
    #  额外上下文（检索结果 → 注入对话模式）
    # ----------------------------------------------------------
    def set_extra_context(
        self, session_id: str, chunks: List[Dict[str, Any]],
    ) -> bool:
        """设置会话的额外上下文 chunk 列表（覆盖式）。

        chunks 每项结构：{chunk_id, library, heading, file_path, text, is_center?}
        传空列表等价于清空。
        """
        with self._lock:
            data = self._load()
            for s in data.get("sessions", []):
                if s.get("id") == session_id:
                    s["extra_context"] = list(chunks)
                    s["updated_at"] = _now()
                    self._save(data)
                    return True
        return False

    def get_extra_context(self, session_id: str) -> List[Dict[str, Any]]:
        """读取会话的额外上下文。无则返回 []。"""
        with self._lock:
            data = self._load()
        for s in data.get("sessions", []):
            if s.get("id") == session_id:
                return s.get("extra_context", []) or []
        return []

    # ----------------------------------------------------------
    #  消息级操作
    # ----------------------------------------------------------
    def add_message(
        self, session_id: str, message: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """追加一条消息到会话末尾。message 至少含 role/content。"""
        with self._lock:
            data = self._load()
            for s in data.get("sessions", []):
                if s.get("id") == session_id:
                    msg = dict(message)
                    msg.setdefault("id", _gen_id("msg"))
                    msg.setdefault("timestamp", _now())
                    s.setdefault("messages", []).append(msg)
                    s["updated_at"] = _now()
                    # 首条用户消息自动设为标题
                    if msg.get("role") == "user" and s.get("title", "新会话") == "新会话":
                        content = msg.get("content", "")
                        s["title"] = content[:20] + ("..." if len(content) > 20 else "")
                    self._save(data)
                    return msg
        return None

    def update_message(
        self, session_id: str, message_id: str, updates: Dict[str, Any],
    ) -> bool:
        """更新会话中某条消息的字段（用于流式生成结束后回填答案）。"""
        with self._lock:
            data = self._load()
            for s in data.get("sessions", []):
                if s.get("id") == session_id:
                    for m in s.get("messages", []):
                        if m.get("id") == message_id:
                            m.update(updates)
                            s["updated_at"] = _now()
                            self._save(data)
                            return True
        return False

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            data = self._load()
        for s in data.get("sessions", []):
            if s.get("id") == session_id:
                return s.get("messages", [])
        return []

    def truncate_after_message(
        self, session_id: str, message_id: str,
    ) -> Optional[Dict[str, Any]]:
        """回退到某条消息：删除该消息及其后所有消息，返回更新后的会话。

        用于"回退到之前的对话"——用户从某条 user 消息处重新提问。
        若 message_id 不存在，返回 None。
        """
        with self._lock:
            data = self._load()
            for s in data.get("sessions", []):
                if s.get("id") == session_id:
                    msgs = s.get("messages", [])
                    idx = None
                    for i, m in enumerate(msgs):
                        if m.get("id") == message_id:
                            idx = i
                            break
                    if idx is None:
                        return None
                    # 删除该消息及之后所有消息
                    s["messages"] = msgs[:idx]
                    s["updated_at"] = _now()
                    self._save(data)
                    return s
        return None

    def export_single_message(
        self, session_id: str, message_id: str,
    ) -> Optional[Dict[str, Any]]:
        """导出单条消息为 Markdown。返回 {markdown, filename} 或 None。"""
        with self._lock:
            data = self._load()
        for s in data.get("sessions", []):
            if s.get("id") != session_id:
                continue
            for m in s.get("messages", []):
                if m.get("id") != message_id:
                    continue
                lines = []
                lines.append(f"# 单条对话记录\n")
                lines.append(f"- 会话：{s.get('title', '')}（{s.get('id', '')}）")
                lines.append(f"- 消息 ID：{m.get('id', '')}")
                lines.append(f"- 时间：{m.get('timestamp', '')}")
                lines.append(f"- 角色：{m.get('role', '')}")
                lines.append("")
                lines.append("---\n")
                if m.get("reasoning"):
                    lines.append("<details><summary>思考过程</summary>\n")
                    lines.append(m["reasoning"] + "\n")
                    lines.append("</details>\n")
                lines.append(m.get("content", "") + "\n")
                refs = m.get("references", [])
                if refs:
                    lines.append("**引用来源：**\n")
                    for r in refs:
                        sf = r.get("source_file", "")
                        heading = r.get("heading", "")
                        cid = r.get("chunk_id", "")
                        lib = r.get("library", "")
                        lines.append(f"- [{r.get('index','')}] {lib} · {sf}"
                                     f"{' · ' + heading if heading else ''} ({cid})")
                    lines.append("")
                md = "\n".join(lines)
                ts = (m.get("timestamp", "") or "msg").replace(":", "-").replace("T", "_")
                return {
                    "markdown": md,
                    "filename": f"msg_{ts}.md",
                }
        return None

    def clear_all(self) -> int:
        """清空所有会话，返回删除数。"""
        with self._lock:
            data = self._load()
            n = len(data.get("sessions", []))
            data["sessions"] = []
            self._save(data)
            return n
