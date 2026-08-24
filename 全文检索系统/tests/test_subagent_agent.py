# -*- coding: utf-8 -*-
"""子智能体（带工具循环）单元测试：client 用脚本桩替换。"""
import unittest

from subagent import run_subagent


def tc(name, **args):
    """构造一条标准 tool_call。"""
    import json
    return {"id": f"call_{name}", "type": "function",
            "function": {"name": name,
                         "arguments": json.dumps(args, ensure_ascii=False)}}


class ScriptedClient:
    """按脚本逐轮返回 tool_calls 的 DeepSeekClient 桩。"""

    model = "stub-model"

    def __init__(self, rounds):
        self.rounds = list(rounds)  # 每轮的 tool_calls 列表
        self.stream_calls = 0
        self.chat_calls = []

    def chat_stream(self, messages, **kw):
        self.stream_calls += 1
        if self.stream_calls <= len(self.rounds):
            yield {"type": "tool_calls", "tool_calls": self.rounds[self.stream_calls - 1]}
        else:
            yield {"type": "content", "delta": "不应走到这里"}
        yield {"type": "done", "usage": None}

    def chat(self, messages, **kw):
        self.chat_calls.append(messages)
        return {"content": "收尾答案"}


CHUNKS = [
    {"library": "库A", "chunk_id": "zone_001/chunk_000001",
     "heading": "第一章", "preview": "正文预览..."},
    {"library": "库A", "chunk_id": "zone_001/chunk_000002",
     "heading": "第二章", "preview": "正文预览..."},
]


def make_callbacks():
    reads = []
    searches = []

    def read_chunk(cid, length=8000):
        reads.append((cid, length))
        return f"[chunk_id: {cid}]\n这是 {cid} 的正文内容"

    def search_chunks(query, top_k=10):
        searches.append((query, top_k))
        return f"检索结果（query={query}）"

    return read_chunk, search_chunks, reads, searches


class TestRunSubagent(unittest.TestCase):
    def test_read_then_finish(self):
        """子智能体先读 chunk 再 finish 提交回答。"""
        client = ScriptedClient([
            [tc("read_chunk", chunk_id="zone_001/chunk_000001", length=5000)],
            [tc("finish", answer="目标信息是 X（来源 zone_001/chunk_000001）")],
        ])
        read_chunk, search_chunks, reads, searches = make_callbacks()
        result = run_subagent(
            client=client, subtask="找出目标信息",
            allowed_chunks=CHUNKS, read_chunk=read_chunk,
            search_chunks=search_chunks, question="用户问题")
        self.assertEqual(result["finish_reason"], "finish")
        self.assertIn("目标信息", result["answer"])
        self.assertIn("zone_001/chunk_000001", result["answer"])
        self.assertEqual(result["rounds"], 2)
        self.assertEqual(reads, [("zone_001/chunk_000001", 5000)])
        self.assertEqual(searches, [])
        self.assertEqual(len(client.chat_calls), 0)  # 正常 finish 无收尾调用

    def test_search_then_read_then_finish(self):
        """范围内检索 → 定位 → 阅读 → finish 的完整工具链。"""
        client = ScriptedClient([
            [tc("search_chunks", query="关键词", top_k=5)],
            [tc("read_chunk", chunk_id="zone_001/chunk_000002")],
            [tc("finish", answer="结论")],
        ])
        read_chunk, search_chunks, reads, searches = make_callbacks()
        result = run_subagent(
            client=client, subtask="总结", allowed_chunks=CHUNKS,
            read_chunk=read_chunk, search_chunks=search_chunks)
        self.assertEqual(result["finish_reason"], "finish")
        self.assertEqual(result["rounds"], 3)
        self.assertEqual(searches, [("关键词", 5)])
        self.assertEqual(len(reads), 1)
        tools = [t["tool"] for t in result["tool_calls"]]
        self.assertEqual(tools, ["search_chunks", "read_chunk", "finish"])

    def test_max_rounds_fallback_generation(self):
        """轮次耗尽未 finish：降级为一次无工具收尾生成。"""
        client = ScriptedClient([
            [tc("read_chunk", chunk_id="zone_001/chunk_000001")],
            [tc("read_chunk", chunk_id="zone_001/chunk_000002")],
        ])
        read_chunk, search_chunks, reads, _ = make_callbacks()
        result = run_subagent(
            client=client, subtask="找出目标信息",
            allowed_chunks=CHUNKS, read_chunk=read_chunk,
            search_chunks=search_chunks, max_rounds=2)
        self.assertEqual(result["finish_reason"], "max_rounds")
        self.assertEqual(result["answer"], "收尾答案")
        self.assertEqual(len(client.chat_calls), 1)  # 收尾生成发生且仅一次
        # 收尾提示已注入对话
        final_messages = client.chat_calls[0]
        self.assertIn("轮次预算已用完", final_messages[-1]["content"])

    def test_unknown_tool_gets_error_text(self):
        """未知工具返回错误文本，子智能体可继续。"""
        client = ScriptedClient([
            [tc("search_full_library", query="x")],  # 不在受限工具集内
            [tc("finish", answer="答案")],
        ])
        read_chunk, search_chunks, reads, _ = make_callbacks()
        result = run_subagent(
            client=client, subtask="任务", allowed_chunks=CHUNKS,
            read_chunk=read_chunk, search_chunks=search_chunks)
        self.assertEqual(result["finish_reason"], "finish")
        self.assertEqual(result["answer"], "答案")

    def test_invalid_inputs(self):
        r = run_subagent(client=None, subtask="", allowed_chunks=CHUNKS,
                         read_chunk=lambda c, l: "", search_chunks=lambda q, k: "")
        self.assertEqual(r["finish_reason"], "error")
        self.assertIn("子任务", r["error"])

        class C:
            model = "m"

        r = run_subagent(client=C(), subtask="任务", allowed_chunks=[],
                         read_chunk=lambda c, l: "", search_chunks=lambda q, k: "")
        self.assertEqual(r["finish_reason"], "error")
        self.assertIn("chunk", r["error"])

        r = run_subagent(client=None, subtask="任务", allowed_chunks=CHUNKS,
                         read_chunk=lambda c, l: "", search_chunks=lambda q, k: "")
        self.assertIn("DeepSeekClient", r["error"])

    def test_dsml_fallback_parsed(self):
        """模型把工具调用输出成 DSML 伪标签时仍能解析执行。"""
        dsml = (
            '<｜｜DSML｜｜tool_calls>'
            '<｜｜DSML｜｜invoke name="read_chunk">'
            '<｜｜DSML｜｜parameter name="chunk_id" string="true">'
            'zone_001/chunk_000001</｜｜DSML｜｜parameter>'
            '</｜｜DSML｜｜invoke>'
            '</｜｜DSML｜｜tool_calls>'
        )

        class DSMLClient(ScriptedClient):
            def chat_stream(self, messages, **kw):
                self.stream_calls += 1
                if self.stream_calls == 1:
                    yield {"type": "content", "delta": dsml}
                else:
                    yield {"type": "tool_calls",
                           "tool_calls": [tc("finish", answer="答案")]}
                yield {"type": "done", "usage": None}

        client = DSMLClient([None])
        read_chunk, search_chunks, reads, _ = make_callbacks()
        result = run_subagent(
            client=client, subtask="任务", allowed_chunks=CHUNKS,
            read_chunk=read_chunk, search_chunks=search_chunks)
        self.assertEqual(result["finish_reason"], "finish")
        self.assertEqual(result["answer"], "答案")
        self.assertEqual(len(reads), 1)  # DSML 中的 read_chunk 被执行


class TestDispatchSubagentIntegration(unittest.TestCase):
    """ToolExecutor._dispatch_subagent 与真子智能体的集成（桩掉文件读取）。"""

    def _make_executor(self, client):
        from agent_workflow import ToolExecutor

        class Registry:
            def list_libraries(self):
                return []

            def get_library(self, name):
                return None  # 元数据读取走异常兜底路径

        return ToolExecutor(Registry(), base_dir=".", question="用户问题",
                            client=client, temperature=0.3)

    def test_dispatch_runs_subagent_loop_and_records(self):
        from unittest.mock import patch

        client = ScriptedClient([
            [tc("read_chunk", chunk_id="zone_001/chunk_000001")],
            [tc("finish", answer="提取结果（来源 zone_001/chunk_000001）")],
        ])
        executor = self._make_executor(client)

        with patch("agent_workflow._load_chunk_text",
                   return_value="这是 chunk 的正文内容，长度足够。"):
            out = executor._dispatch_subagent(
                "找出目标信息",
                [{"library": "库A", "chunk_id": "zone_001/chunk_000001"},
                 {"library": "库A", "chunk_id": "zone_001/chunk_000002"}],
                context_hint="关注时间地点")

        # 返回给主智能体的结果含执行摘要与子智能体回答
        self.assertIn("子智能体已完成", out)
        self.assertIn("提取结果", out)
        # 执行记录（供前端 subagent_dispatched 事件）
        self.assertEqual(len(executor.subagent_records), 1)
        rec = executor.subagent_records[0]
        self.assertEqual(rec["finish_reason"], "finish")
        self.assertEqual(rec["rounds"], 2)
        self.assertEqual(rec["loaded_count"], 1)  # 实际读了 1 个 chunk
        self.assertEqual(rec["tool_call_count"], 2)
        # 子智能体读过的 chunk 记入引用来源
        self.assertEqual(len(executor.accessed_chunks), 1)
        self.assertEqual(executor.accessed_chunks[0]["chunk_id"],
                         "zone_001/chunk_000001")

    def test_dispatch_rejects_bad_input(self):
        executor = self._make_executor(ScriptedClient([]))
        out = executor._dispatch_subagent("", [{"library": "A",
                                                "chunk_id": "x"}])
        self.assertIn("子任务描述不能为空", out)
        out = executor._dispatch_subagent("任务", [])
        self.assertIn("chunks 列表不能为空", out)
        out = executor._dispatch_subagent(
            "任务", [{"library": "", "chunk_id": ""}])
        self.assertIn("无效", out)


if __name__ == "__main__":
    unittest.main()
