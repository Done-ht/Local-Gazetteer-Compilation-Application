# -*- coding: utf-8 -*-
"""agent_workflow.py 启发式与 DSML 兜底解析单元测试。"""
import json
import unittest

from agent_workflow import (
    DSMLStreamFilter,
    _looks_like_narration,
    _parse_dsml_tool_calls,
    _strip_dsml_from_text,
)


class TestNarrationHeuristic(unittest.TestCase):
    """叙述性过渡语识别：高查准率，宁可漏判不可误判。"""

    NARRATION = [
        "好的，我先查看资料库。",
        "好的，让我检索一下刘备的相关内容",
        "让我查看先主传的内容",
        "我先检索一下相关信息",
        "接下来我需要搜索更多信息",
        "首先我要查看标题结构",
        "嗯，让我看看这个 chunk",
        "明白了，我现在来分析这些资料",
        "那么我先查看标题列表\n\n答案在后文",
    ]

    ANSWERS = [
        # "首先，"开头的结构化答案是合法答案开头
        "首先，郎溪县的农业区划分为三个部分。",
        # 人名开头（旧版硬编码 刘备/诸葛亮/曹操 场景）
        "刘备字玄德，涿郡涿县人。",
        "诸葛亮率军北伐，出祁山。",
        # "检索/根据/我认为"开头的合法答案（旧版会误吞）
        "检索到的资料表明，该事件发生于建安五年。",
        "根据《三国志》记载，刘备于章武三年去世。",
        "我认为这一记载存在矛盾。",
        "我觉得这个说法可以成立。",
        # markdown / 列表 / 编号开头
        "## 刘备生平",
        "1. 涿郡起兵",
        "- 三顾茅庐",
        "> 引用原文",
        # 空与纯空白
        "",
        "\n\n",
    ]

    def test_narration_detected(self):
        for text in self.NARRATION:
            self.assertTrue(
                _looks_like_narration(text), msg=f"应为叙述: {text!r}")

    def test_answers_not_swallowed(self):
        for text in self.ANSWERS:
            self.assertFalse(
                _looks_like_narration(text), msg=f"应为答案: {text!r}")


DSML_SAMPLE = (
    '我需要检索一下。\n'
    '<｜｜DSML｜｜tool_calls>\n'
    '<｜｜DSML｜｜invoke name="search">\n'
    '<｜｜DSML｜｜parameter name="query" string="true">先主 昭烈皇帝</｜｜DSML｜｜parameter>\n'
    '<｜｜DSML｜｜parameter name="libraries" string="false">["二十四史"]</｜｜DSML｜｜parameter>\n'
    '</｜｜DSML｜｜invoke>\n'
    '</｜｜DSML｜｜tool_calls>'
)


class TestDSMLFallback(unittest.TestCase):
    def test_parse_dsml_tool_calls(self):
        calls = _parse_dsml_tool_calls(DSML_SAMPLE)
        self.assertEqual(len(calls), 1)
        fn = calls[0]["function"]
        self.assertEqual(fn["name"], "search")
        args = json.loads(fn["arguments"])
        self.assertEqual(args["query"], "先主 昭烈皇帝")
        self.assertEqual(args["libraries"], ["二十四史"])  # string=false → JSON

    def test_parse_no_dsml(self):
        self.assertEqual(_parse_dsml_tool_calls("普通文本"), [])
        self.assertEqual(_parse_dsml_tool_calls(""), [])

    def test_strip_dsml(self):
        cleaned = _strip_dsml_from_text("前文。" + DSML_SAMPLE + "后文。")
        self.assertNotIn("DSML", cleaned)
        self.assertIn("前文。", cleaned)
        self.assertIn("后文。", cleaned)

    def test_stream_filter_split_deltas(self):
        """DSML 标签跨 delta 分片时不泄漏到主输出。"""
        f = DSMLStreamFilter()
        leaked = []
        completed = False
        # 把整段文本按 7 字符一片喂入（开始标记必然被切开）
        for i in range(0, len(DSML_SAMPLE), 7):
            chunks, comp = f.feed(DSML_SAMPLE[i:i + 7])
            leaked.extend(chunks)
            completed = completed or comp
        leaked.extend(f.flush())
        joined = "".join(leaked)
        self.assertNotIn("DSML", joined)
        self.assertIn("我需要检索一下。", joined)
        self.assertTrue(completed)
        dsml_text = f.get_dsml_text()
        self.assertIn("invoke", dsml_text)
        self.assertTrue(_parse_dsml_tool_calls(dsml_text))


if __name__ == "__main__":
    unittest.main()
