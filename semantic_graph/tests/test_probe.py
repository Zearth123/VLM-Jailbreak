#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探针层:占位探针的手算值、可加性/单调性、以及"真探针缺失必须显式失败"。"""

import math
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from sg import probe as P  # noqa: E402

CLOSE = 1e-9


def expected(weight: float) -> float:
    return 1.0 - math.exp(-weight / P.SATURATION)


class TestLexicalProbe(unittest.TestCase):
    def setUp(self):
        self.p = P.LexicalProbe()

    def test_empty_is_zero(self):
        self.assertEqual(self.p.score_text(""), 0.0)
        self.assertEqual(self.p.score_text("   "), 0.0)

    def test_entity_only_hand_value(self):
        # substance 权重 3.0 → 1 - exp(-0.75)
        self.assertAlmostEqual(self.p.score_text("ricin"), expected(3.0), places=9)

    def test_action_only_hand_value(self):
        # 动作权重 3.0 × ACTION_SCALE 0.125 = 0.375 → 1 - exp(-0.09375)
        self.assertAlmostEqual(self.p.score_text("make"), expected(3.0 * P.ACTION_SCALE), places=9)

    def test_combined_hand_value(self):
        self.assertAlmostEqual(
            self.p.score_text("ricin make"), expected(3.0 + 3.0 * P.ACTION_SCALE), places=9
        )

    def test_evasion_weight(self):
        self.assertAlmostEqual(
            self.p.score_text("without getting caught"), expected(P.EVASION_WEIGHT), places=9
        )

    def test_marker_dedup(self):
        self.assertEqual(self.p.score_text("make make make"), self.p.score_text("make"))

    def test_monotone_under_union(self):
        a = self.p.score_text("ricin")
        b = self.p.score_text("make")
        joined = self.p.score_text("ricin make")
        self.assertGreaterEqual(joined, a)
        self.assertGreaterEqual(joined, b)

    def test_content_dominates_framing_verb(self):
        """这是占位探针存在的前提:内容标记要压过框架动词,否则切分目标没意义。"""
        self.assertGreater(self.p.score_text("ricin"), self.p.score_text("make"))

    def test_markers_explainable(self):
        m = self.p.markers("ricin without getting caught at home")
        self.assertIn("entity:ricin", m)
        self.assertIn("evasion", m)
        self.assertIn("condition", m)
        self.assertAlmostEqual(sum(m.values()), self.p.weight("ricin without getting caught at home"))

    def test_describe_and_sha(self):
        d = self.p.describe()
        self.assertEqual(d["probe_id"], "lexical-v0")
        self.assertTrue(d["sha"])
        self.assertTrue(P.is_placeholder(self.p))


class TestFactory(unittest.TestCase):
    def test_lexical(self):
        self.assertIsInstance(P.make_probe("lexical"), P.LexicalProbe)

    def test_unknown(self):
        with self.assertRaises(ValueError):
            P.make_probe("gpt-4o")

    def test_xguard_missing_weights_fails_loudly(self):
        with self.assertRaises(P.ProbeUnavailable) as cm:
            P.make_probe("xguard", model_dir=Path("/nonexistent/xguard"))
        self.assertIn("找不到本地 guard 权重", str(cm.exception))

    def test_xguard_probe_fails_loudly(self):
        """没有 torch(或没有权重)时**必须**显式失败,不许静默降级,且要说清缺的是哪一样。"""
        try:
            import torch  # noqa: F401
        except ImportError:
            pass
        else:  # pragma: no cover - 只有装了 torch 的环境才会走到
            self.skipTest("本环境有 torch,跳过缺失依赖的断言")
        with self.assertRaises(P.ProbeUnavailable) as cm:
            P.make_probe("xguard")
        msg = str(cm.exception)
        if P.XGUARD_DIR.exists():
            self.assertIn("torch", msg)
        else:
            self.assertIn("找不到本地 guard 权重", msg)
            self.assertIn(P.XGUARD_DIR.name, msg)


class TestTorchFree(unittest.TestCase):
    def test_probe_module_does_not_import_torch(self):
        import sg.probe  # noqa: F401
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main(verbosity=2)
