# -*- coding: utf-8 -*-
"""Agent 工作流异步核心/同步桥接/并行工具执行单元测试（client 用桩替换）。"""
import asyncio
import threading
import time
import unittest

from agent_workflow import (
    _run_tool_calls_parallel,
    agent_workflow_stream,
    agent_workflow_stream_async,
)


def tc(name, args_json="{}"):
    return {"id": f"call_{name}", "type": "function",
            "function": {"name": name, "arguments": args_json}}


class StubRegistry:
    """空库 registry 桩：list_libraries 工具可正常执行。"""

    def list_libraries(self):
        return []

    def get_library(self, name):
        return None


class ScriptedClient:
    """脚本化的 DeepSeekClient 桩。

    - chat()：供 _check_question 预检，返回 ok=true
    - chat_stream(tools=...)：按脚本逐轮返回 tool_calls
    - chat_stream(无 tools)：最终生成阶段，返回固定答案
    """

    model = "stub-model"

    def __init__(self, tool_rounds):
        self.tool_rounds = list(tool_rounds)  # 每轮的 tool_calls 列表
        self.tool_stream_calls = 0
        self.final_stream_calls = 0
        self.chat_calls = 0

    def chat(self, messages, **kw):
        self.chat_calls += 1
        return {"content": '{"ok": true}'}

    def chat_stream(self, messages, **kw):
        if kw.get("tools"):
            self.tool_stream_calls += 1
            idx = self.tool_stream_calls - 1
            if idx < len(self.tool_rounds):
                calls = self.tool_rounds[idx]
                if calls is not None:
                    yield {"type": "tool_calls", "tool_calls": calls}
        else:
            self.final_stream_calls += 1
            yield {"type": "content", "delta": "最终答案"}
        yield {"type": "done", "usage": {"total_tokens": 1}}


def collect(gen):
    return list(gen)


class TestSyncBridgeEndToEnd(unittest.TestCase):
    def test_normal_flow_with_parallel_tools(self):
        """同步桥接端到端：一轮两个工具并行调用 → finish → 最终答案。"""
        client = ScriptedClient([
            [tc("list_libraries"), tc("list_libraries")],  # 第 1 轮：两个工具
            [tc("finish")],                                  # 第 2 轮：finish
        ])
        events = collect(agent_workflow_stream(
            "测试问题", StubRegistry(), client, base_dir=".",
            max_rounds=5))

        phases = [e["phase"] for e in events]
        # 两个工具调用的通知与结果按序成对出现
        self.assertEqual(phases.count("tool_call"), 3)
        self.assertEqual(phases.count("tool_result"), 3)
        first_round = [p for p in phases if p in ("tool_call", "tool_result")][:4]
        self.assertEqual(first_round, ["tool_call", "tool_call",
                                       "tool_result", "tool_result"])
        # 最终答案与收尾
        self.assertIn("content", phases)
        self.assertEqual(phases[-1], "done")
        content = "".join(e.get("delta", "") for e in events
                          if e["phase"] == "content")
        self.assertEqual(content, "最终答案")
        # 无错误、无轮次耗尽
        self.assertNotIn("error", phases)
        self.assertNotIn("rounds_exhausted", phases)

    def test_budget_exhaustion_graceful_finish(self):
        """轮次预算耗尽：发 rounds_exhausted 事件，最终生成仍完成。"""
        client = ScriptedClient([
            [tc("list_libraries")],  # 永不 finish
            [tc("list_libraries")],
            [tc("list_libraries")],
        ])
        events = collect(agent_workflow_stream(
            "测试问题", StubRegistry(), client, base_dir=".",
            max_rounds=3))
        phases = [e["phase"] for e in events]
        self.assertIn("rounds_exhausted", phases)
        self.assertEqual(phases[-1], "done")
        content = "".join(e.get("delta", "") for e in events
                          if e["phase"] == "content")
        self.assertEqual(content, "最终答案")
        self.assertNotIn("error", phases)

    def test_clarify_short_circuit(self):
        """预检不通过：clarify + done(skipped)，不进入工具循环。"""

        class ClarifyClient(ScriptedClient):
            def chat(self, messages, **kw):
                self.chat_calls += 1
                return {"content": '{"ok": false, "reason": "歧义", '
                                   '"clarify": "请补充"}'}

        client = ClarifyClient([])
        events = collect(agent_workflow_stream(
            "测试问题", StubRegistry(), client, base_dir="."))
        phases = [e["phase"] for e in events]
        self.assertEqual(phases, ["clarify", "done"])
        self.assertTrue(events[-1].get("skipped"))


class TestParallelToolExecution(unittest.TestCase):
    def test_results_ordered_and_concurrent(self):
        """多工具并行执行：结果保持原顺序，实际跑在不同线程。"""
        calls_lock = threading.Lock()
        seen_threads = []
        seen_names = []

        class RecordingExecutor:
            def execute(self, fn_name, args):
                with calls_lock:
                    seen_names.append(fn_name)
                    seen_threads.append(threading.current_thread().ident)
                time.sleep(0.15)  # 保证两个任务真正并发挂起
                return f"result:{fn_name}"

        results = asyncio.run(_run_tool_calls_parallel(
            RecordingExecutor(),
            [("tool_a", {}), ("tool_b", {}), ("tool_c", {})]))
        # 结果顺序与调用顺序一致
        self.assertEqual(results, ["result:tool_a", "result:tool_b",
                                   "result:tool_c"])
        # 并发执行：不同任务占用不同线程
        self.assertEqual(len(set(seen_threads)), 3)

    def test_single_call_fast_path(self):
        class Executor:
            def execute(self, fn_name, args):
                return "ok"

        results = asyncio.run(_run_tool_calls_parallel(
            Executor(), [("only_one", {})]))
        self.assertEqual(results, ["ok"])

    def test_tool_exception_isolated(self):
        """单个工具抛异常不影响其他工具，转为错误文本。"""

        class FlakyExecutor:
            def __init__(self):
                self.executed = []

            def execute(self, fn_name, args):
                self.executed.append(fn_name)
                if fn_name == "bad":
                    raise RuntimeError("boom")
                return f"ok:{fn_name}"

        ex = FlakyExecutor()
        results = asyncio.run(_run_tool_calls_parallel(
            ex, [("bad", {}), ("good", {})]))
        self.assertEqual(results[0], "工具执行错误: boom")
        self.assertEqual(results[1], "ok:good")
        self.assertEqual(ex.executed, ["bad", "good"])


class TestAsyncCore(unittest.TestCase):
    def test_async_generator_direct(self):
        """异步核心可直接 async-for 消费（供未来原生异步接入方使用）。"""

        async def run():
            client = ScriptedClient([[tc("finish")]])
            return [e async for e in agent_workflow_stream_async(
                "测试问题", StubRegistry(), client, base_dir=".")]

        events = asyncio.run(run())
        phases = [e["phase"] for e in events]
        self.assertIn("tool_call", phases)
        self.assertEqual(phases[-1], "done")


class TestBridgeCancellation(unittest.TestCase):
    def test_consumer_close_cancels_backend(self):
        """客户端断开（消费端提前 close）：异步任务被取消，桥接线程及时退出。"""
        release = threading.Event()

        class SlowClient(ScriptedClient):
            def chat_stream(self, messages, **kw):
                if kw.get("tools"):
                    # 先产出一个事件让工作流真正跑起来，随后挂起模拟 LLM 长响应
                    yield {"type": "reasoning", "delta": "思考中"}
                    release.wait(timeout=10)
                    yield {"type": "tool_calls",
                           "tool_calls": [tc("finish")]}
                    yield {"type": "done", "usage": None}
                else:
                    yield {"type": "content", "delta": "不应被生成"}
                    yield {"type": "done", "usage": None}

        client = SlowClient([])
        gen = agent_workflow_stream(
            "测试问题", StubRegistry(), client, base_dir=".")
        # 启动生成器并取第一个事件（预检→工具循环流的 reasoning）
        first = next(gen)
        self.assertEqual(first["phase"], "thinking")
        # 消费端关闭生成器（模拟 SSE 客户端断开），应立即返回
        gen.close()
        # 释放挂起的流，桥接线程应随即退出（任务已取消不再消费事件）
        release.set()
        deadline = time.time() + 5
        threads_alive = True
        while time.time() < deadline:
            threads_alive = any(t.name == "agent-workflow-loop" and t.is_alive()
                                for t in threading.enumerate())
            if not threads_alive:
                break
            time.sleep(0.05)
        self.assertFalse(
            threads_alive, "桥接事件循环线程在取消后 5 秒内未退出")
        # 后续生成未发生（取消生效）：最终答案阶段从未执行
        self.assertEqual(client.final_stream_calls, 0)

    def test_error_event_forwarded(self):
        """工作流内部异常转为 error 事件（不向同步端抛出）。"""

        class ExplodeClient(ScriptedClient):
            def chat_stream(self, messages, **kw):
                raise RuntimeError("llm boom")
                yield  # pragma: no cover

        events = collect(agent_workflow_stream(
            "测试问题", StubRegistry(), ExplodeClient([]), base_dir="."))
        phases = [e["phase"] for e in events]
        self.assertIn("error", phases)
        self.assertEqual(phases[-1], "error")
        self.assertIn("llm boom", events[-1]["error"])


if __name__ == "__main__":
    unittest.main()
