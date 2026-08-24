# -*- coding: utf-8 -*-
"""agent_workflow.py 上下文压缩单元测试（client 用桩替换）。"""
import unittest

from agent_workflow import (
    _compress_conversation,
    _estimate_tokens,
    _mechanical_digest,
)


class StubClient:
    """DeepSeekClient 桩：ask 返回固定摘要。"""

    def __init__(self, summary="这是摘要。"):
        self.summary = summary
        self.asked = []

    def ask(self, prompt, **kw):
        self.asked.append((prompt, kw))
        return self.summary


class FailClient:
    """DeepSeekClient 桩：ask 抛异常（模拟 LLM 摘要失败）。"""

    def ask(self, prompt, **kw):
        raise RuntimeError("api down")


def make_conversation(n=12):
    """构造 system + user + 若干 tool 消息的对话。

    工具结果内容足够长（600 字），确保机械压缩的逐条 200 字截断
    真的能收缩上下文（真实场景中压缩只在 token 远超阈值时触发）。
    """
    conv = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "问题"},
    ]
    for i in range(n - 2):
        conv.append({
            "role": "tool",
            "tool_call_id": f"call_{i}",
            "content": f"工具结果 {i}：" + "x" * 600,
        })
    return conv


class TestCompressConversation(unittest.TestCase):
    def test_short_conversation_unchanged(self):
        conv = make_conversation(8)  # <= COMPRESS_KEEP_RECENT + 3
        new_conv, saved, summary, method = _compress_conversation(StubClient(), conv)
        self.assertIs(new_conv, conv)
        self.assertEqual(saved, 0)
        self.assertEqual(summary, "")
        self.assertEqual(method, "none")

    def test_long_conversation_compressed(self):
        conv = make_conversation(12)
        client = StubClient("压缩后的摘要")
        new_conv, saved, summary, method = _compress_conversation(client, conv)
        self.assertEqual(summary, "压缩后的摘要")
        self.assertEqual(method, "llm")
        self.assertGreater(saved, 0)
        # 结构：system + user 问题 + 摘要(user) + 最近 tail
        self.assertEqual(new_conv[0]["role"], "system")
        self.assertEqual(new_conv[1]["role"], "user")
        self.assertIn("【历史对话摘要】", new_conv[2]["content"])
        self.assertEqual(new_conv[2]["role"], "user")
        self.assertLess(len(new_conv), len(conv))
        # 桩确实被调用（压缩请求发生）
        self.assertEqual(len(client.asked), 1)

    def test_compress_failure_falls_back_to_mechanical(self):
        """LLM 摘要失败时降级为机械压缩：上下文仍收缩，不再静默沿用原对话。"""
        conv = make_conversation(12)
        new_conv, saved, summary, method = _compress_conversation(FailClient(), conv)
        self.assertEqual(method, "mechanical")
        self.assertIsNot(new_conv, conv)          # 不再原样返回
        self.assertGreater(saved, 0)              # 上下文必然收缩
        self.assertLess(len(new_conv), len(conv))
        self.assertIn("机械压缩", summary)         # 摘要标注了降级方式
        self.assertIn("工具结果", summary)          # 保留了逐条截断的内容
        # 结构同样有效：system + user 问题 + 摘要(user) + 最近 tail
        self.assertEqual(new_conv[0]["role"], "system")
        self.assertEqual(new_conv[1]["role"], "user")
        self.assertEqual(new_conv[2]["role"], "user")

    def test_empty_llm_summary_falls_back_to_mechanical(self):
        """LLM 返回空摘要同样触发机械降级。"""
        conv = make_conversation(12)
        new_conv, saved, summary, method = _compress_conversation(StubClient("   "), conv)
        self.assertEqual(method, "mechanical")
        self.assertGreater(saved, 0)


class TestMechanicalDigest(unittest.TestCase):
    def test_digest_keeps_tool_names_and_chunk_ids(self):
        middle = [
            {"role": "assistant", "content": "",
             "tool_calls": [{"function": {"name": "search",
                                          "arguments": '{"query":"水利"}'}}]},
            {"role": "tool", "tool_call_id": "c1",
             "content": "chunk_id=zone_001/chunk_000123 | 库=县志\n片段内容..."},
            {"role": "user", "content": "系统提示文字"},
        ]
        digest = _mechanical_digest(middle)
        self.assertIn("search", digest)
        self.assertIn("水利", digest)
        self.assertIn("chunk_id=zone_001/chunk_000123", digest)

    def test_digest_capped(self):
        middle = [{"role": "tool", "tool_call_id": "c",
                   "content": "x" * 1000} for _ in range(100)]
        digest = _mechanical_digest(middle, max_chars=5000)
        self.assertLessEqual(len(digest), 5100)


class TestEstimateTokens(unittest.TestCase):
    def test_monotonic_with_content(self):
        small = [{"role": "user", "content": "x" * 100}]
        large = [{"role": "user", "content": "x" * 1000}]
        self.assertLess(_estimate_tokens(small), _estimate_tokens(large))

    def test_tool_calls_arguments_counted(self):
        plain = [{"role": "assistant", "content": ""}]
        with_calls = [{
            "role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": "search",
                                         "arguments": "q" * 500}}],
        }]
        self.assertGreater(
            _estimate_tokens(with_calls), _estimate_tokens(plain))


if __name__ == "__main__":
    unittest.main()
