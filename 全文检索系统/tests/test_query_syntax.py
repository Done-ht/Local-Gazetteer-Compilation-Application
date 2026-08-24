# -*- coding: utf-8 -*-
"""查询语法解析与标题过滤单元测试（{} 花括号 / 引号原子 / 多分隔符）。"""
import os
import unittest

from searcher import (
    _split_group_terms,
    _parse_query_groups_py,
    _parse_semantic_groups_py,
    _parse_title_groups_py,
    strip_title_groups_py,
    apply_title_filter,
)
from verifier import _verify_semantic


class TestSplitGroupTerms(unittest.TestCase):
    def test_comma_separators(self):
        self.assertEqual(_split_group_terms("死，崩，薨"), ["死", "崩", "薨"])
        self.assertEqual(_split_group_terms("死,崩"), ["死", "崩"])

    def test_extra_separators(self):
        self.assertEqual(_split_group_terms("死、崩|薨"), ["死", "崩", "薨"])

    def test_quoted_atom_keeps_quotes(self):
        # 引号原子：保留引号传给索引层做整段精确匹配（保留原引号字符）
        self.assertEqual(_split_group_terms('"5,40"，"1.5万"'),
                         ['"5,40"', '"1.5万"'])
        self.assertEqual(_split_group_terms('“5,40” x'), ['“5,40”', 'x'])

    def test_quoted_atom_comma_not_split(self):
        terms = _split_group_terms('"5,40"，5.40亿立方米')
        self.assertEqual(terms, ['"5,40"', "5.40亿立方米"])


class TestParseGroups(unittest.TestCase):
    def test_plain_multiword(self):
        self.assertEqual(_parse_query_groups_py("刘备 诸葛亮"),
                         [["刘备"], ["诸葛亮"]])

    def test_paren_group_with_pipe(self):
        self.assertEqual(_parse_query_groups_py("刘备 (死|崩)"),
                         [["刘备"], ["死", "崩"]])

    def test_braces_do_not_leak_into_keywords(self):
        # {} 内容不应进入 keyword_groups
        g = _parse_query_groups_py("灌溉 {水利卷}")
        self.assertEqual(g, [["灌溉"]])

    def test_title_groups(self):
        self.assertEqual(_parse_title_groups_py("灌溉 {水利卷，水利志}"),
                         [["水利卷", "水利志"]])
        self.assertEqual(_parse_title_groups_py("｛水利卷｝"), [["水利卷"]])
        self.assertIsNone(_parse_title_groups_py("灌溉"))

    def test_semantic_groups_with_new_seps(self):
        self.assertEqual(_parse_semantic_groups_py("[死、崩]"), [["死", "崩"]])
        self.assertEqual(_parse_semantic_groups_py("【死|崩】"), [["死", "崩"]])

    def test_strip_title_groups(self):
        self.assertEqual(strip_title_groups_py("灌溉 {水利卷}").strip(), "灌溉")
        self.assertEqual(
            strip_title_groups_py("刘备 (死，崩) {蜀书}").strip(), "刘备 (死，崩)")
        # () [] 内容不受影响
        self.assertEqual(strip_title_groups_py("刘备 (死，崩)").strip(),
                         "刘备 (死，崩)")

    def test_mixed_all_syntax(self):
        q = "刘备 (死，崩) {蜀书} [战役，交锋]"
        self.assertEqual(_parse_query_groups_py(q), [["刘备"], ["死", "崩"]])
        self.assertEqual(_parse_semantic_groups_py(q), [["战役", "交锋"]])
        self.assertEqual(_parse_title_groups_py(q), [["蜀书"]])


class TestApplyTitleFilter(unittest.TestCase):
    def _r(self, heading, source_file=""):
        return {"heading": heading, "source_file": source_file}

    def test_single_group(self):
        results = [self._r("蜀书·先主传"), self._r("吴书·吴主传")]
        out = apply_title_filter(results, [["蜀书"]])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["heading"], "蜀书·先主传")

    def test_group_or_semantics(self):
        results = [self._r("蜀书"), self._r("魏书"), self._r("吴书")]
        out = apply_title_filter(results, [["蜀书", "吴书"]])
        self.assertEqual({r["heading"] for r in out}, {"蜀书", "吴书"})

    def test_multi_group_and_semantics(self):
        results = [self._r("蜀书·先主传"), self._r("蜀书·后主传")]
        out = apply_title_filter(results, [["蜀书"], ["先主"]])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["heading"], "蜀书·先主传")

    def test_matches_source_file(self):
        results = [self._r("", "sanguozhi.txt"), self._r("", "hanshu.txt")]
        out = apply_title_filter(results, [["sanguozhi"]])
        self.assertEqual(len(out), 1)

    def test_empty_groups_noop(self):
        results = [self._r("x")]
        self.assertEqual(apply_title_filter(results, []), results)
        self.assertEqual(apply_title_filter(results, None), results)


class TestVerifySemanticEmpty(unittest.TestCase):
    def test_empty_storage_not_flagged(self):
        # 存储区为空（0 chunk）时不应报"向量索引未构建"
        class _FakeMgr:
            root = os.path.join(os.path.dirname(__file__), "_no_such_lib")

            def list_zones(self):
                return []

        class _FakeZone:
            meta = type("M", (), {"chunk_count": 0})()

        # 直接构造 0 chunk 场景：list_zones 返回空 → total_chunks = 0
        res = _verify_semantic(_FakeMgr())
        # 语义管理器不可用或空库都应视为 ok
        self.assertTrue(res["ok"])
        self.assertEqual(res["description"], "")


if __name__ == "__main__":
    unittest.main()
