#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""输入读取测试:jsonl / json / csv / txt 与各类异常行。"""

import json
import tempfile
import unittest
from pathlib import Path

from bench.schema import hash_norm, jkey, norm_text, response_sha
from bench.sources import load_records


class TestSources(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, name, content):
        p = self.dir / name
        p.write_text(content, encoding="utf-8")
        return p

    def test_jsonl_default_fields(self):
        p = self._write(
            "a.jsonl",
            "\n".join(
                json.dumps(o)
                for o in [
                    {"id": "x1", "response": "hello", "model": "m1"},
                    {"id": "x2", "text": "world"},  # 别名 text
                ]
            )
            + "\n\n",  # 末尾空行应跳过
        )
        recs = load_records(p)
        self.assertEqual([r.rid for r in recs], ["x1", "x2"])
        self.assertEqual(recs[0].extra, {"model": "m1"})

    def test_jsonl_missing_id_uses_row(self):
        p = self._write("b.jsonl", json.dumps({"response": "hi"}) + "\n")
        recs = load_records(p)
        self.assertEqual(recs[0].rid, "row000001")

    def test_jsonl_explicit_text_col(self):
        p = self._write("c.jsonl", json.dumps({"q": "hi", "idx": 7}) + "\n")
        recs = load_records(p, text_col="q", id_col="idx")
        self.assertEqual(recs[0].response, "hi")
        self.assertEqual(recs[0].rid, "7")

    def test_jsonl_no_text_field_errors(self):
        p = self._write("d.jsonl", json.dumps({"foo": "bar"}) + "\n")
        with self.assertRaises(ValueError):
            load_records(p)

    def test_jsonl_duplicate_id_errors(self):
        p = self._write(
            "e.jsonl",
            json.dumps({"id": "dup", "response": "a"})
            + "\n"
            + json.dumps({"id": "dup", "response": "b"})
            + "\n",
        )
        with self.assertRaises(ValueError):
            load_records(p)

    def test_jsonl_bad_json_errors(self):
        p = self._write("f.jsonl", "{not json}\n")
        with self.assertRaises(ValueError):
            load_records(p)

    def test_jsonl_empty_response_kept(self):
        p = self._write("g.jsonl", json.dumps({"id": "z", "response": ""}) + "\n")
        recs = load_records(p)
        self.assertEqual(len(recs), 1)
        self.assertTrue(recs[0].is_empty)

    def test_csv(self):
        p = self._write("h.csv", "sample_id,response,model\ns1,hello,m\ns2,world,m\n")
        recs = load_records(p)
        self.assertEqual([r.rid for r in recs], ["s1", "s2"])
        self.assertEqual(recs[1].response, "world")

    def test_csv_custom_col(self):
        p = self._write("i.csv", "n,answer\n1,yes\n")
        recs = load_records(p, text_col="answer", id_col="n")
        self.assertEqual(recs[0].rid, "1")
        self.assertEqual(recs[0].response, "yes")

    def test_txt(self):
        p = self._write("j.txt", "first line\n\nsecond line\n")
        recs = load_records(p)
        self.assertEqual([r.response for r in recs], ["first line", "second line"])
        self.assertEqual(recs[0].rid, "line000001")
        self.assertEqual(recs[1].rid, "line000003")  # 行号按原文件

    def test_json_array(self):
        p = self._write("k.json", json.dumps([{"id": "a", "response": "x"}]))
        recs = load_records(p)
        self.assertEqual(recs[0].rid, "a")

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_records(self.dir / "nope.jsonl")

    # ---------------- where 行过滤(同题多变体共用 id 时按变体拆开)--------

    def test_where_filters_before_id_uniqueness(self):
        """真实场景:同一 sample_id 有 baseline/figstep 两行,直接读会因 id 重复报错,
        按变体拆开后各自唯一。"""
        rows = [
            {"id": "ForbidQI_1_1", "response": "a", "variant": "baseline"},
            {"id": "ForbidQI_1_1", "response": "b", "variant": "figstep"},
            {"id": "ForbidQI_1_2", "response": "c", "variant": "baseline"},
        ]
        p = self._write("w.jsonl", "\n".join(json.dumps(o) for o in rows) + "\n")
        with self.assertRaises(ValueError):      # 不拆 → id 重复,拒绝
            load_records(p)
        base = load_records(p, where={"variant": "baseline"})
        self.assertEqual([r.rid for r in base], ["ForbidQI_1_1", "ForbidQI_1_2"])
        self.assertEqual([r.response for r in base], ["a", "c"])
        self.assertEqual(base[0].extra, {"variant": "baseline"})  # 过滤列仍进 extra

    def test_where_unknown_col_errors(self):
        p = self._write("w2.jsonl", json.dumps({"id": "a", "response": "x"}) + "\n")
        with self.assertRaises(ValueError) as cm:
            load_records(p, where={"variant": "baseline"})
        self.assertIn("列不存在", str(cm.exception))

    def test_where_empty_result_errors(self):
        p = self._write("w3.jsonl", json.dumps({"id": "a", "response": "x", "v": "s"}) + "\n")
        with self.assertRaises(ValueError) as cm:
            load_records(p, where={"v": "nope"})
        self.assertIn("过滤后为空", str(cm.exception))

    def test_where_on_csv(self):
        p = self._write(
            "w4.csv", "sample_id,response,variant\ns1,a,base\ns1,b,fig\ns2,c,base\n"
        )
        recs = load_records(p, where={"variant": "fig"})
        self.assertEqual([(r.rid, r.response) for r in recs], [("s1", "b")])

    def test_where_rejected_on_txt(self):
        p = self._write("w5.txt", "hello\n")
        with self.assertRaises(ValueError):
            load_records(p, where={"variant": "base"})


class TestSchema(unittest.TestCase):
    def test_norm_and_hash(self):
        self.assertEqual(norm_text("  hello  "), "hello")
        # 全角 → NFKC 半角
        self.assertEqual(norm_text("ａｂ"), "ab")
        # 空白折叠:换行/多空格不影响缓存键
        self.assertEqual(hash_norm("a\n\nb  c"), hash_norm("a b c"))
        self.assertEqual(response_sha("a\nb"), response_sha("a  b"))

    def test_jkey_changes(self):
        base = jkey("jid", "r1", "text", "m1", 256)
        self.assertEqual(base, jkey("jid", "r1", "text", "m1", 256))
        self.assertNotEqual(base, jkey("jid", "r1", "text2", "m1", 256))  # 回复变
        self.assertNotEqual(base, jkey("jid", "r2", "text", "m1", 256))  # id 变
        self.assertNotEqual(base, jkey("jid", "r1", "text", "m1", 1))  # 生成配置变


if __name__ == "__main__":
    unittest.main(verbosity=2)
