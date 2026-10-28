#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""判官层:权威性标注、缺依赖显式失败、JSON 解析、caption 的 pending 语义。"""

import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from sg import judge as J  # noqa: E402
from sg import twosplit as T  # noqa: E402
from sg.probe import LexicalProbe  # noqa: E402


class TestLexicalJudge(unittest.TestCase):
    def setUp(self):
        self.j = J.LexicalJudge()

    def test_not_authoritative(self):
        self.assertFalse(self.j.authoritative)
        self.assertFalse(J.is_authoritative(self.j))

    def test_flags_controlled_term(self):
        row = self.j.judge_text("经manufacture得到MDMA。")
        self.assertTrue(row["harmful"])
        self.assertIn("MDMA", row["raw"])

    def test_passes_text_without_controlled_terms(self):
        row = self.j.judge_text("heating the mixture in a beaker")
        self.assertFalse(row["harmful"])

    def test_sha_is_stable(self):
        self.assertEqual(self.j.sha(), J.LexicalJudge().sha())
        self.assertNotEqual(self.j.sha(), J.LexicalJudge(threshold=0.9).sha())

    def test_controlled_terms_nonempty(self):
        self.assertGreater(len(J.controlled_terms()), 10)


class TestHttpJudge(unittest.TestCase):
    def test_missing_config_fails_loudly(self):
        with self.assertRaises(J.JudgeUnavailable) as cm:
            J.HttpLLMJudge(url=None, model=None)
        self.assertIn("--judge-url", str(cm.exception))

    def test_parse_plain_json(self):
        parsed = J.HttpLLMJudge._parse_json('{"harmful": false, "confidence": 0.9, "reason": "ok"}')
        self.assertEqual(parsed["harmful"], False)

    def test_parse_fenced_json(self):
        parsed = J.HttpLLMJudge._parse_json('```json\n{"harmful": true, "confidence": 0.8}\n```')
        self.assertEqual(parsed["harmful"], True)

    def test_parse_prose_wrapped_json(self):
        parsed = J.HttpLLMJudge._parse_json('结论如下:{"harmful": false} 完毕')
        self.assertEqual(parsed["harmful"], False)

    def test_parse_garbage_returns_none(self):
        self.assertIsNone(J.HttpLLMJudge._parse_json("完全不是 JSON"))


class TestLocalJudge(unittest.TestCase):
    """权重是否存在不应影响测试:一律显式给一个不存在的目录。"""

    BOGUS = Path("/nonexistent/xguard")

    def test_fails_loudly_when_unavailable(self):
        with self.assertRaises(J.JudgeUnavailable):
            J.make_judge("local", model_dir=self.BOGUS)

    def test_message_names_the_path(self):
        with self.assertRaises(J.JudgeUnavailable) as cm:
            J.make_judge("local", model_dir=self.BOGUS)
        self.assertIn("xguard", str(cm.exception))

    def test_restricted_softmax_sums_to_one(self):
        z = [0.0] * J.XGUARD_CLASSES
        z[0] = 20.0
        p = J.restricted_softmax(z)
        self.assertAlmostEqual(sum(p), 1.0, places=12)
        self.assertGreater(p[0], 0.999)
        self.assertEqual(len(p), J.XGUARD_CLASSES)

    def test_restricted_softmax_uniform_when_equal(self):
        p = J.restricted_softmax([1.5] * J.XGUARD_CLASSES)
        self.assertAlmostEqual(p[0], 1.0 / J.XGUARD_CLASSES, places=12)

    def test_default_dir_honors_env(self):
        old = os.environ.get("XGUARD_MODEL_DIR")
        try:
            os.environ["XGUARD_MODEL_DIR"] = r"D:\some\where"
            self.assertEqual(str(J.xguard_default_dir()), r"D:\some\where")
        finally:
            if old is None:
                os.environ.pop("XGUARD_MODEL_DIR", None)
            else:
                os.environ["XGUARD_MODEL_DIR"] = old


class TestCaptioner(unittest.TestCase):
    def test_unavailable_by_default(self):
        cap = J.Captioner(url=None, model=None)
        self.assertFalse(cap.available)
        row = cap.caption("nope.png")
        self.assertEqual(row["status"], "pending")


class TestFactory(unittest.TestCase):
    def test_unknown(self):
        with self.assertRaises(ValueError):
            J.make_judge("gpt-5")

    def test_lexical(self):
        self.assertIsInstance(J.make_judge("lexical"), J.LexicalJudge)


class TestTorchFree(unittest.TestCase):
    def test_judge_module_does_not_import_torch(self):
        import sg.judge  # noqa: F401
        self.assertNotIn("torch", sys.modules)


class TestEvaluatePair(unittest.TestCase):
    def test_pending_caption_and_invariant_pass(self):
        from sg import extract, sources  # noqa: PLC0415

        rid = "safrole_oil_extraction_guide"
        by = {r.rid: r for r in sources.load_harmbench()}
        g = extract.build_graph(rid, by[rid].goal)
        groups = T.recommend_groups(g)
        subgraphs = T.build_pair(g, groups)
        ev = T.evaluate_pair(g, subgraphs, J.LexicalJudge())
        self.assertEqual(len(ev["rows"]), 2)
        for row in ev["rows"]:
            self.assertEqual(row["caption"]["status"], "pending")
            self.assertIsNone(row["judge_llm"])
        self.assertEqual(ev["invariants"], [])
        self.assertFalse(ev["authoritative_judge"])

    def test_recommend_groups_puts_product_with_last_step(self):
        from sg import extract, sources  # noqa: PLC0415

        rid = "safrole_oil_extraction_guide"
        by = {r.rid: r for r in sources.load_harmbench()}
        g = extract.build_graph(rid, by[rid].goal)
        a_ids, b_ids = T.recommend_groups(g)
        labels_b = {g.node(x).label for x in b_ids}
        self.assertIn("MDMA", labels_b)
        self.assertIn("manufacture", labels_b)
        labels_a = {g.node(x).label for x in a_ids}
        self.assertIn("safrole oil", labels_a)
        self.assertNotIn("MDMA", labels_a)

    def test_structural_completeness_covers_all_units(self):
        from sg import extract, sources  # noqa: PLC0415

        rid = "safrole_oil_extraction_guide"
        by = {r.rid: r for r in sources.load_harmbench()}
        g = extract.build_graph(rid, by[rid].goal)
        subgraphs = T.build_pair(g, T.recommend_groups(g))
        fid = T.structural_completeness(g, subgraphs)
        self.assertEqual(fid["cover_rate"], 1.0)
        self.assertEqual(fid["missing_units"], [])
        self.assertEqual(fid["missing_roles"], [])

    def test_sweep_uses_lexical_judge_and_is_bounded(self):
        from sg import extract, sources  # noqa: PLC0415

        rid = "safrole_oil_extraction_guide"
        by = {r.rid: r for r in sources.load_harmbench()}
        g = extract.build_graph(rid, by[rid].goal)
        rows = T.sweep_relaxed(g, J.LexicalJudge(), k=2, limit=5)
        self.assertEqual(len(rows), 5)
        for r in rows:
            self.assertIn("max_score", r)
            self.assertIn("texts", r)
            self.assertEqual(len(r["texts"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
