# -*- coding: utf-8 -*-
"""/api/search 处理器端到端测试：title_groups（{} 语法）分支与混合短语查询。"""
import json
import os
import shutil
import tempfile
import unittest

import web_api
from indexer import ZoneIndex, build_postings, write_chunk_index


def _make_lib(root, lib_name, items):
    """items: [(seq, text, heading, file)] → 建单 zone 库并注册。"""
    lib_dir = os.path.join(root, "data", lib_name)
    zone_id = "zone_001"
    zone_path = os.path.join(lib_dir, zone_id)
    chunks_dir = os.path.join(zone_path, "chunks")
    index_dir = os.path.join(zone_path, "_index")
    os.makedirs(chunks_dir)
    os.makedirs(index_dir)
    total_chars = 0
    for seq, text, heading, fname in items:
        name = f"chunk_{seq:06d}"
        with open(os.path.join(chunks_dir, name + ".json"), "w",
                  encoding="utf-8") as f:
            json.dump({
                "chunk_id": f"{zone_id}/{name}",
                "zone_id": zone_id,
                "chunk_seq": seq,
                "text": text,
                "text_length": len(text),
                "text_offset": 0,
                "heading": heading,
                "source": {"file_name": fname},
            }, f, ensure_ascii=False)
        write_chunk_index(os.path.join(chunks_dir, name + ".idx"),
                          build_postings(text))
        total_chars += len(text)
    with open(os.path.join(zone_path, "_zone.json"), "w", encoding="utf-8") as f:
        json.dump({"zone_id": zone_id, "char_count": total_chars,
                   "chunk_count": len(items), "source_count": len(items)},
                  f, ensure_ascii=False)
    ZoneIndex(index_dir).merge_zone_chunks(chunks_dir, zone_id)
    ZoneIndex.invalidate(index_dir)
    return lib_dir


class TestHandleSearchTitleGroups(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self._orig_base = web_api.BASE_DIR
        web_api.BASE_DIR = self.root
        _make_lib(self.root, "县志库", [
            (1, "全县林地蓄积5.40亿立方米，覆盖率60%。", "水利卷·蓄积篇", "xianzhi.txt"),
            (2, "文教学校数量统计。", "文教卷·学校篇", "xianzhi.txt"),
            (3, "灌区灌溉面积统计。", "水利卷·灌区篇", "xianzhi.txt"),
        ])
        self.reg = web_api.LibraryRegistry(self.root)
        self.reg.create("县志库", path=os.path.join("data", "县志库"))
        self.user = {"username": "tester", "is_admin": True}

    def tearDown(self):
        web_api.BASE_DIR = self._orig_base
        shutil.rmtree(self.root, ignore_errors=True)

    def _search(self, **params):
        st, payload = web_api.handle_search("GET", "/api/search",
                                            params, {}, user=self.user)
        self.assertEqual(st, 200, payload)
        return payload["data"] if isinstance(payload, dict) else payload

    def test_title_only_branch(self):
        d = self._search(title_groups=json.dumps([["水利"]]))
        self.assertEqual(d["mode"], "title")
        headings = {r["heading"] for r in d["results"]}
        self.assertEqual(headings, {"水利卷·蓄积篇", "水利卷·灌区篇"})

    def test_keyword_plus_title_filter(self):
        # 关键词"统计" 命中 2/3 两个 chunk，标题限定"水利"只保留 chunk 3
        d = self._search(
            keyword_groups=json.dumps([["统计"]]),
            title_groups=json.dumps([["水利"]]),
        )
        self.assertEqual([r["chunk_id"] for r in d["results"]],
                         ["zone_001/chunk_000003"])
        self.assertEqual(d["title_groups"], [["水利"]])

    def test_raw_query_with_braces_fallback(self):
        # 直接带 {} 的原始 query（API 兜底路径）
        d = self._search(query="统计 {水利}")
        self.assertEqual([r["chunk_id"] for r in d["results"]],
                         ["zone_001/chunk_000003"])

    def test_mixed_run_whole_phrase(self):
        # 5.40亿立方米 作为整体：只命中 chunk 1
        d = self._search(query="5.40亿立方米")
        self.assertEqual([r["chunk_id"] for r in d["results"]],
                         ["zone_001/chunk_000001"])
        self.assertEqual(d["results"][0]["matched_words"], ["5.40亿立方米"])


if __name__ == "__main__":
    unittest.main()
