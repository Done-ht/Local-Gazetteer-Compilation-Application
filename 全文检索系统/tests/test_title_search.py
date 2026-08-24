# -*- coding: utf-8 -*-
"""{} 标题检索（search_by_titles）端到端测试：临时库 + 标题缓存。"""
import json
import os
import shutil
import tempfile
import unittest

from indexer import ZoneIndex, build_postings, write_chunk_index
from library import Library, LibraryRegistry
from searcher import search_by_titles, _TITLES_CACHE_FILE


def _make_lib_with_zones(root, lib_name, zone_texts):
    """建临时库：<root>/data/<lib>/zone_xxx/chunks。zone_texts: {zone_id: [(seq, text, heading, file)]}"""
    lib_dir = os.path.join(root, "data", lib_name)
    for zone_id, items in zone_texts.items():
        zone_path = os.path.join(lib_dir, zone_id)
        chunks_dir = os.path.join(zone_path, "chunks")
        index_dir = os.path.join(zone_path, "_index")
        os.makedirs(chunks_dir)
        os.makedirs(index_dir)
        zi = ZoneIndex(index_dir)
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
                    "heading": heading,
                    "source": {"file_name": fname},
                }, f, ensure_ascii=False)
            write_chunk_index(os.path.join(chunks_dir, name + ".idx"),
                              build_postings(text))
            total_chars += len(text)
        # 写 zone 元数据（stats/缓存失效校验依赖 chunk_count）
        with open(os.path.join(zone_path, "_zone.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"zone_id": zone_id, "char_count": total_chars,
                       "chunk_count": len(items), "source_count": len(items)},
                      f, ensure_ascii=False)
        zi.merge_zone_chunks(chunks_dir, zone_id)
        ZoneIndex.invalidate(index_dir)
    return lib_dir


class TestSearchByTitles(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        _make_lib_with_zones(self.root, "测试库", {
            "zone_001": [
                (1, "灌区设计流量50立方米每秒。", "水利卷·灌区篇", "县志水利.txt"),
                (2, "文教事业概况。", "文教卷·学校篇", "县志文教.txt"),
                (3, "另一处水利内容。", "卷十二 水利", "县志旧版.txt"),
            ],
        })
        self.reg = LibraryRegistry(os.path.join(self.root, "registry"))
        self.reg.create("测试库", path=os.path.join("data", "测试库"))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_title_search_single_group(self):
        res = search_by_titles(self.reg, [["水利"]], self.root, top_k=10)
        self.assertEqual(res["mode"], "title")
        headings = {r["heading"] for r in res["results"]}
        self.assertEqual(headings, {"水利卷·灌区篇", "卷十二 水利"})

    def test_title_search_or_semantics(self):
        # 组内 OR：水利 或 文教
        res = search_by_titles(self.reg, [["水利", "文教"]], self.root, top_k=10)
        self.assertEqual(len(res["results"]), 3)

    def test_title_search_multi_group_and(self):
        # 两组 AND：标题须同时含 水利 和 灌区
        res = search_by_titles(self.reg, [["水利"], ["灌区"]], self.root, top_k=10)
        self.assertEqual([r["heading"] for r in res["results"]],
                         ["水利卷·灌区篇"])

    def test_title_matches_file_name(self):
        res = search_by_titles(self.reg, [["县志文教"]], self.root, top_k=10)
        self.assertEqual([r["source_file"] for r in res["results"]],
                         ["县志文教.txt"])

    def test_titles_cache_written_and_reused(self):
        search_by_titles(self.reg, [["水利"]], self.root, top_k=10)
        cache_path = os.path.join(self.root, "data", "测试库", _TITLES_CACHE_FILE)
        self.assertTrue(os.path.isfile(cache_path))
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        self.assertEqual(cached["total_chunks"], 3)
        self.assertEqual(len(cached["titles"]), 3)
        # 第二次查询走缓存，结果一致
        res2 = search_by_titles(self.reg, [["水利"]], self.root, top_k=10)
        self.assertEqual(len(res2["results"]), 2)


if __name__ == "__main__":
    unittest.main()
