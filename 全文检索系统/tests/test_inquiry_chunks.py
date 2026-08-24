# -*- coding: utf-8 -*-
"""质询报告快速验证接口测试：/api/inquiries/{id}/chunks。"""
import json
import os
import shutil
import tempfile
import unittest

import web_api


class TestInquiryChunks(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self._orig_base = web_api.BASE_DIR
        web_api.BASE_DIR = self.root
        # 建一个带 chunk 的库
        lib_dir = os.path.join(self.root, "data", "测试库")
        chunks_dir = os.path.join(lib_dir, "zone_001", "chunks")
        os.makedirs(chunks_dir)
        self.chunk_text = "全县林地蓄积5.40亿立方米。"
        with open(os.path.join(chunks_dir, "chunk_000001.json"), "w",
                  encoding="utf-8") as f:
            json.dump({
                "chunk_id": "zone_001/chunk_000001",
                "zone_id": "zone_001",
                "chunk_seq": 1,
                "text": self.chunk_text,
                "text_length": len(self.chunk_text),
                "heading": "水利卷·蓄积篇",
                "source": {"file_name": "xianzhi.txt"},
            }, f, ensure_ascii=False)
        self.reg = web_api.LibraryRegistry(self.root)
        self.reg.create("测试库", path=os.path.join("data", "测试库"))

        # 保存一份引用该 chunk 的质询报告（新格式：chunk_refs 带库定位）
        from inquiry_store import InquiryStore
        store = InquiryStore(self.root)
        store.save({
            "id": "inq_test_1",
            "timestamp": "2026-08-24T00:00:00",
            "question": "林地蓄积量是多少",
            "libraries": ["测试库"],
            "source": "ai_workflow",
            "issue_type": "数据冲突",
            "description": "两个 chunk 蓄积量数字不一致",
            "chunk_ids": ["zone_001/chunk_000001", "zone_001/chunk_000099"],
            "chunk_refs": [
                {"chunk_id": "zone_001/chunk_000001", "library": "测试库"},
            ],
            "severity": "warning",
        })
        self.user = {"username": "tester", "is_admin": True}

    def tearDown(self):
        web_api.BASE_DIR = self._orig_base
        shutil.rmtree(self.root, ignore_errors=True)

    def test_chunks_resolved_and_unresolved(self):
        st, payload = web_api.handle_inquiry_chunks(
            "GET", "/api/inquiries/inq_test_1/chunks", {}, {},
            "inq_test_1", user=self.user)
        self.assertEqual(st, 200, payload)
        data = payload["data"]
        self.assertEqual(len(data["chunks"]), 1)
        c = data["chunks"][0]
        self.assertEqual(c["library"], "测试库")
        self.assertEqual(c["heading"], "水利卷·蓄积篇")
        self.assertEqual(c["text"], self.chunk_text)
        # 不存在的 chunk 出现在 unresolved
        self.assertEqual(data["unresolved"], ["zone_001/chunk_000099"])

    def test_requires_login(self):
        st, _ = web_api.handle_inquiry_chunks(
            "GET", "/api/inquiries/inq_test_1/chunks", {}, {},
            "inq_test_1", user={"username": "guest", "is_admin": False})
        self.assertEqual(st, 401)

    def test_report_not_found(self):
        st, _ = web_api.handle_inquiry_chunks(
            "GET", "/api/inquiries/no_such/chunks", {}, {},
            "no_such", user=self.user)
        self.assertEqual(st, 404)


if __name__ == "__main__":
    unittest.main()
