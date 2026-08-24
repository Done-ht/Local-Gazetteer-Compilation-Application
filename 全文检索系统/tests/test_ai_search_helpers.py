# -*- coding: utf-8 -*-
"""ai_search.py 共用辅助函数单元测试（不依赖真实检索/LLM）。"""
import unittest

import ai_search
from ai_search import (
    _build_extra_ctx_text,
    _build_vector_mini_chunks,
    _build_vector_references,
    _finalize_vector_hits,
    _stream_llm_answer,
    _summarize_round_hits,
    _warn_if_noisy,
)


class TestExtraCtxText(unittest.TestCase):
    def test_none_empty(self):
        self.assertEqual(_build_extra_ctx_text(None), "")
        self.assertEqual(_build_extra_ctx_text([]), "")

    def test_items_formatted(self):
        text = _build_extra_ctx_text([
            {"heading": "先主传", "file_path": r"data\二十四史\sgyy.txt",
             "text": "先主姓刘，讳备。"},
        ])
        self.assertIn("[用户选定资料 1] 先主传", text)
        self.assertIn("sgyy.txt", text)  # 用 basename 而非全路径
        self.assertIn("先主姓刘，讳备。", text)


class TestSummarizeRoundHits(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(_summarize_round_hits([]), "本轮无命中")

    def test_with_hits(self):
        hits = [
            {"heading": "先主传", "source_file": "a.txt"},
            {"heading": "", "source_file": "b.txt"},  # 回退 source_file
            {},  # 两者皆空 → 不进 top_headings
        ] * 3
        summary = _summarize_round_hits(hits)
        self.assertIn("共9条命中", summary)
        self.assertIn("先主传", summary)
        self.assertIn("b.txt", summary)


class TestFinalizeVectorHits(unittest.TestCase):
    def test_dedup_sort_truncate(self):
        hits = [
            {"chunk_id": "zone_001/chunk_000001", "score": 0.5,
             "matched_queries": ["a"]},
            {"chunk_id": "zone_001/chunk_000001", "score": 0.9,
             "matched_queries": ["b"]},   # 同 chunk 更高分，合并 matched_queries
            {"chunk_id": "zone_001/chunk_000002", "score": 0.7,
             "matched_queries": ["a"]},
            {"chunk_id": "", "score": 1.0},  # 无 id → 丢弃
        ]
        truncated, total_unique = _finalize_vector_hits(hits, max_mini_chunks=1)
        self.assertEqual(total_unique, 2)
        self.assertEqual(len(truncated), 1)  # 截断生效
        self.assertEqual(truncated[0]["chunk_id"], "zone_001/chunk_000001")
        self.assertEqual(truncated[0]["score"], 0.9)  # 保留最高分
        self.assertEqual(
            sorted(truncated[0]["matched_queries"]), ["a", "b"])


class TestMiniChunksAndReferences(unittest.TestCase):
    def test_mini_chunks_text_field_fallback(self):
        hits = [
            {"chunk_id": "zone_001/chunk_000001", "library": "二十四史",
             "source_file": "sanguo.txt", "heading": "先主传",
             "sub_text": "命中片段", "snippet": "备用片段",
             "matched_words": ["刘备"], "hit_count": 3},
        ]
        # 指定 sub_text
        mc = _build_vector_mini_chunks(hits, text_field="sub_text")[0]
        self.assertIn("命中片段", mc["mini_snippet"])
        # subchunk_text 缺失时回退 snippet
        mc2 = _build_vector_mini_chunks(hits, text_field="subchunk_text")[0]
        self.assertIn("备用片段", mc2["mini_snippet"])

    def test_mini_snippet_truncated_to_200(self):
        hits = [{"chunk_id": "c1", "library": "L", "sub_text": "字" * 500}]
        mc = _build_vector_mini_chunks(hits, text_field="sub_text")[0]
        self.assertLessEqual(len(mc["mini_snippet"]), 203)  # 200 + "..."

    def test_references_structure(self):
        hits = [{"chunk_id": "c1", "library": "L", "heading": "H",
                 "source_file": "f.txt", "source_file_path": "d/f.txt",
                 "matched_words": ["w"], "hit_count": 1,
                 "sub_text": "片段"}]
        mc = _build_vector_mini_chunks(hits, text_field="sub_text")
        refs = _build_vector_references(mc)
        self.assertEqual(refs[0]["index"], 1)
        self.assertEqual(refs[0]["chunk_id"], "c1")
        self.assertEqual(refs[0]["library"], "L")
        self.assertEqual(refs[0]["snippet"], mc[0]["mini_snippet"])


class TestWarnIfNoisy(unittest.TestCase):
    def test_below_threshold_no_warn(self):
        events = list(_warn_if_noisy(ai_search.EFFORT_WARN_THRESHOLD,
                                     ai_search.EFFORT_WARN_THRESHOLD))
        self.assertEqual(events, [])

    def test_above_threshold_warns(self):
        events = list(_warn_if_noisy(ai_search.EFFORT_WARN_THRESHOLD + 1, 10))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["phase"], "warn")
        self.assertEqual(events[0]["truncated_to"], 10)


class StubStreamClient:
    """chat_stream 桩：依次返回预设事件。"""

    def __init__(self, events):
        self.model = "stub"
        self._events = events

    def chat_stream(self, messages, **kw):
        yield from self._events


class TestStreamLLMAnswer(unittest.TestCase):
    def test_event_mapping(self):
        client = StubStreamClient([
            {"type": "reasoning", "delta": "思"},
            {"type": "content", "delta": "答"},
            {"type": "done", "usage": {"total": 1}},
        ])
        events = list(_stream_llm_answer(client, [], 0.3))
        self.assertEqual(events, [
            {"phase": "reasoning", "delta": "思"},
            {"phase": "content", "delta": "答"},
            {"phase": "done", "usage": {"total": 1}},
        ])


if __name__ == "__main__":
    unittest.main()
