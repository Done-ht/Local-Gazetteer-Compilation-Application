# -*- coding: utf-8 -*-
"""ZoneIndex.search 词元串整体匹配（混合串如 5.40亿立方米）单元测试。"""
import json
import os
import shutil
import tempfile
import unittest

from indexer import ZoneIndex, build_postings, write_chunk_index


def _make_zone(root, zone_id, texts):
    """构建测试 zone：<root>/<zone_id>/{chunks,_index}。texts: [(seq, text)]"""
    zone_path = os.path.join(root, zone_id)
    chunks_dir = os.path.join(zone_path, "chunks")
    index_dir = os.path.join(zone_path, "_index")
    os.makedirs(chunks_dir)
    os.makedirs(index_dir)
    for seq, text in texts:
        name = f"chunk_{seq:06d}"
        with open(os.path.join(chunks_dir, name + ".json"), "w",
                  encoding="utf-8") as f:
            json.dump({"chunk_id": f"{zone_id}/{name}", "text": text}, f,
                      ensure_ascii=False)
        write_chunk_index(os.path.join(chunks_dir, name + ".idx"),
                          build_postings(text))
    zi = ZoneIndex(index_dir)
    zi.merge_zone_chunks(chunks_dir, zone_id)
    return zi


class TestPhraseSearch(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.zi = _make_zone(self.root, "zone_001", [
            (1, "全县林地面积5.40亿立方米，森林覆盖率百分之六十。"),
            (2, "蓄积量5-40亿立方米与5，40亿立方米写法不一。"),
            (3, "共计蓄积5.40 亿立方米（有空格）。"),
            (4, "经济发展GDP增长快速。"),
            (5, "上升50%，增幅百分之五十。"),
            (6, "灌区design水量的50%达到设计标准。"),
        ])

    def tearDown(self):
        ZoneIndex.invalidate(self.zi.index_dir)
        shutil.rmtree(self.root, ignore_errors=True)

    def test_mixed_run_whole_match(self):
        """5.40亿立方米 整体匹配：只命中精确包含该串的 chunk。"""
        res = self.zi.search("5.40亿立方米")
        self.assertEqual(list(res.keys()), ["5.40亿立方米"])
        hit = res["5.40亿立方米"]
        self.assertIn("zone_001/chunk_000001", hit)
        # "5-40亿立方米" / "5，40亿立方米" 对齐但原文不同 → 校验剔除
        self.assertNotIn("zone_001/chunk_000002", hit)
        # "5.40 亿立方米"（含空格）不构成整体 → 剔除
        self.assertNotIn("zone_001/chunk_000003", hit)
        # 命中位置 = 精确串起点
        self.assertEqual(hit["zone_001/chunk_000001"], [6])

    def test_connector_variant_excluded(self):
        """同对齐不同连接符（5-40亿立方米）不算整体命中。"""
        res = self.zi.search("5-40亿立方米")
        hit = res["5-40亿立方米"]
        self.assertIn("zone_001/chunk_000002", hit)
        self.assertNotIn("zone_001/chunk_000001", hit)

    def test_percent_run(self):
        """50% 作为整体：命中两处精确 "50%"。"""
        res = self.zi.search("50%")
        hit = res["50%"]
        self.assertIn("zone_001/chunk_000005", hit)
        self.assertIn("zone_001/chunk_000006", hit)
        # 命中位置精确
        self.assertEqual(hit["zone_001/chunk_000005"], [2])

    def test_pure_chinese_phrase_unchanged(self):
        """纯中文短语行为与原逻辑一致（连续才算命中）。"""
        res = self.zi.search("经济发展")
        self.assertIn("经济发展", res)
        self.assertIn("zone_001/chunk_000004", res["经济发展"])

    def test_mixed_cn_en_run(self):
        """中文+英文混合串（经济GDP）整体匹配。"""
        res = self.zi.search("经济GDP")
        # "经济发展GDP" 中 "经济" 与 "GDP" 不相邻 → 不命中
        self.assertEqual(res.get("经济gdp", {}), {})

    def test_plain_alnum_unchanged(self):
        res = self.zi.search("GDP")
        self.assertIn("gdp", res)
        self.assertIn("zone_001/chunk_000004", res["gdp"])

    def test_quotes_are_emphasis(self):
        """引号视为强调：带引号与不带引号等价。"""
        res = self.zi.search('"5.40亿立方米"')
        self.assertEqual(list(res.keys()), ["5.40亿立方米"])
        self.assertIn("zone_001/chunk_000001", res["5.40亿立方米"])

    def test_multi_word_query(self):
        """空格分隔的多个词元串分别作为整体匹配。"""
        res = self.zi.search("5.40亿立方米 百分之六十")
        self.assertIn("5.40亿立方米", res)
        self.assertIn("百分之六十", res)
        self.assertIn("zone_001/chunk_000001", res["5.40亿立方米"])
        self.assertIn("zone_001/chunk_000001", res["百分之六十"])

    def test_quoted_atom_with_comma(self):
        """引号原子："5,40" 含逗号整段精确匹配（不被当作两个数字）。"""
        self.zi2 = _make_zone(self.root, "zone_002", [
            (1, "人口为5,40万人，另写作5.40万。"),
            (2, "人口为5 40万，写法有误。"),
        ])
        try:
            res = self.zi2.search('"5,40万"')
            self.assertEqual(list(res.keys()), ["5,40万"])
            hit = res["5,40万"]
            self.assertIn("zone_002/chunk_000001", hit)
            self.assertNotIn("zone_002/chunk_000002", hit)
            self.assertEqual(hit["zone_002/chunk_000001"], [3])
        finally:
            ZoneIndex.invalidate(self.zi2.index_dir)

    def test_quoted_phrase_with_space(self):
        """引号段可含空格：整段精确匹配。"""
        self.zi3 = _make_zone(self.root, "zone_003", [
            (1, "设计流量 design flow 达标。"),
            (2, "设计design流程flow混排。"),
        ])
        try:
            res = self.zi3.search('"design flow"')
            self.assertEqual(list(res.keys()), ["design flow"])
            hit = res["design flow"]
            self.assertIn("zone_003/chunk_000001", hit)
            self.assertNotIn("zone_003/chunk_000002", hit)
        finally:
            ZoneIndex.invalidate(self.zi3.index_dir)


if __name__ == "__main__":
    unittest.main()
