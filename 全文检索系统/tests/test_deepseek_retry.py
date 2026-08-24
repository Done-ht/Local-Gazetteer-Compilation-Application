# -*- coding: utf-8 -*-
"""deepseek.py 重试机制单元测试（不发起真实网络请求）。"""
import unittest
from unittest import mock

import deepseek
from deepseek import APIError, DeepSeekClient, NetworkError


def make_client(max_retries=3):
    return DeepSeekClient(
        "sk-test",
        max_retries=max_retries,
        retry_base_delay=0.0,
        stream_timeout=1,
    )


COMPLETION = {
    "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
    "model": "test",
    "usage": {},
}


class TestChatRetry(unittest.TestCase):
    def setUp(self):
        # 测试中不真正 sleep
        self.sleep_patch = mock.patch.object(deepseek.time, "sleep")
        self.sleep_mock = self.sleep_patch.start()
        self.addCleanup(self.sleep_patch.stop)

    def test_network_error_retried_then_success(self):
        client = make_client()
        client._post_json = mock.Mock(
            side_effect=[NetworkError("boom"), COMPLETION])
        resp = client.chat([{"role": "user", "content": "q"}])
        self.assertEqual(resp["content"], "hi")
        self.assertEqual(client._post_json.call_count, 2)
        self.assertEqual(self.sleep_mock.call_count, 1)

    def test_retryable_status_retried(self):
        client = make_client()
        client._post_json = mock.Mock(
            side_effect=[APIError("rate limited", status=429), COMPLETION])
        resp = client.chat([{"role": "user", "content": "q"}])
        self.assertEqual(resp["content"], "hi")
        self.assertEqual(client._post_json.call_count, 2)

    def test_non_retryable_status_raises_immediately(self):
        client = make_client()
        client._post_json = mock.Mock(
            side_effect=APIError("bad key", status=401))
        with self.assertRaises(APIError):
            client.chat([{"role": "user", "content": "q"}])
        self.assertEqual(client._post_json.call_count, 1)
        self.sleep_mock.assert_not_called()

    def test_retries_exhausted(self):
        client = make_client(max_retries=2)
        client._post_json = mock.Mock(side_effect=NetworkError("down"))
        with self.assertRaises(NetworkError):
            client.chat([{"role": "user", "content": "q"}])
        # 首次 + 2 次重试
        self.assertEqual(client._post_json.call_count, 3)
        self.assertEqual(self.sleep_mock.call_count, 2)

    def test_zero_retries_single_attempt(self):
        client = make_client(max_retries=0)
        client._post_json = mock.Mock(side_effect=NetworkError("down"))
        with self.assertRaises(NetworkError):
            client.chat([{"role": "user", "content": "q"}])
        self.assertEqual(client._post_json.call_count, 1)


def gen_fail():
    raise NetworkError("connect failed")
    yield  # 使其成为生成器函数


def gen_ok():
    yield {"type": "content", "delta": "hi"}
    yield {"type": "done", "usage": None}


def gen_fail_after_first_event():
    yield {"type": "content", "delta": "partial"}
    raise NetworkError("read timeout")


class TestStreamRetry(unittest.TestCase):
    def setUp(self):
        self.sleep_patch = mock.patch.object(deepseek.time, "sleep")
        self.sleep_mock = self.sleep_patch.start()
        self.addCleanup(self.sleep_patch.stop)

    def _collect(self, client):
        return list(client.chat_stream([{"role": "user", "content": "q"}]))

    def test_stream_retried_before_first_event(self):
        client = make_client()
        # 第一次失败、第二次成功：side_effect 逐次返回
        client._post_stream = mock.Mock(side_effect=[gen_fail(), gen_ok()])
        events = self._collect(client)
        self.assertEqual(events[0], {"type": "content", "delta": "hi"})
        self.assertEqual(client._post_stream.call_count, 2)

    def test_stream_midstream_failure_not_retried(self):
        client = make_client()
        client._post_stream = mock.Mock(
            side_effect=[gen_fail_after_first_event(), gen_ok()])
        with self.assertRaises(NetworkError):
            self._collect(client)
        # 已输出事件后失败：不可重试（否则前端收到重复内容）
        self.assertEqual(client._post_stream.call_count, 1)

    def test_stream_non_retryable_not_retried(self):
        client = make_client()
        client._post_stream = mock.Mock(side_effect=[gen_fail(), gen_ok()])
        with mock.patch.object(
                DeepSeekClient, "_is_retryable", return_value=False):
            with self.assertRaises(NetworkError):
                self._collect(client)
        self.assertEqual(client._post_stream.call_count, 1)


class TestRetryDelay(unittest.TestCase):
    def test_exponential_backoff(self):
        client = make_client()
        client.retry_base_delay = 1.0
        with mock.patch.object(deepseek.random, "uniform", return_value=0.0):
            self.assertAlmostEqual(client._retry_delay(1), 1.0)
            self.assertAlmostEqual(client._retry_delay(2), 2.0)
            self.assertAlmostEqual(client._retry_delay(3), 4.0)


if __name__ == "__main__":
    unittest.main()
