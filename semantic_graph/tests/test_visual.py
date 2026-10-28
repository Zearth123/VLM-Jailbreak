#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图像注入层(A0 排版 / A1 图结构)的不变量。

最重要的两条:
  - A0 只画 `image_payload` 的原文片段 —— 渲染器不产生新内容;
  - A1 只画关系(方框/箭头/步序数字),**不写任何实体名字** —— 这是"身份不在 A1 里"的机械证明。
"""

import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from sg import extract, sources, twosplit as T, visual as V  # noqa: E402
from sg import project as P  # noqa: E402
from tests import workspace_tempdir  # noqa: E402

BY_ID = {r.rid: r for r in sources.load_harmbench()}
RID = "safrole_oil_extraction_guide"


def fixture():
    g = extract.build_graph(RID, BY_ID[RID].goal)
    pair = T.build_pair(g, T.recommend_groups(g))
    img_sg = next(sg for sg in pair if sg.channel != P.CARRIER_TEXT)
    txt_sg = next(sg for sg in pair if sg.channel == P.CARRIER_TEXT)
    return g, pair, img_sg, txt_sg


class TestSpecAndMetrics(unittest.TestCase):
    def test_condition_id_depends_on_spec(self):
        a = V.RenderSpec(arm="a0", font_pt=28).cid()
        b = V.RenderSpec(arm="a0", font_pt=12).cid()
        self.assertNotEqual(a, b)
        self.assertIn("renderer_version", V.RenderSpec().payload())

    def test_legibility_band_follows_patch_size(self):
        self.assertEqual(V.legibility_band(8), "below_patch")
        self.assertEqual(V.legibility_band(12), "below_patch")
        self.assertEqual(V.legibility_band(20), "borderline")
        self.assertEqual(V.legibility_band(28), "reliable")
        self.assertAlmostEqual(28 / V.PATCH, 2.0)

    def test_contrast_ratio_black_on_white_is_max(self):
        self.assertAlmostEqual(V.contrast_ratio(V._fg_level(1.0)), 21.0, places=1)
        self.assertLess(V.contrast_ratio(V._fg_level(0.45)), 5.0)


class TestA0(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g, cls.pair, cls.img_sg, cls.txt_sg = fixture()

    def test_deterministic(self):
        spec = V.RenderSpec(arm="a0", layout="caption", font_pt=28)
        a = V.render_a0(self.img_sg, spec, self.g.behavior_id)[0]
        b = V.render_a0(self.img_sg, spec, self.g.behavior_id)[0]
        self.assertEqual(a.sha256, b.sha256)

    def test_canvas_and_no_text_chunks(self):
        spec = V.RenderSpec(arm="a0", layout="table", font_pt=20)
        red = V.render_a0(self.img_sg, spec, self.g.behavior_id)[0]
        self.assertEqual(tuple(red.metrics["canvas"]), V.CANVAS)
        self.assertNotIn(b"tEXt", red.png)
        self.assertNotIn(b"iTXt", red.png)
        self.assertEqual(red.metrics["patch_grid"], [24, 24])

    def test_draws_only_payload_surfaces(self):
        """A0 画出来的字符串必须来自 image 载荷(碎片化时是它的子串)。"""
        want = [u["surface"] for u in self.img_sg.rewrite if u["channel"] == P.CARRIER_IMAGE]
        red = V.render_a0(self.img_sg, V.RenderSpec(), self.g.behavior_id)[0]
        for s in red.drawn_strings:
            self.assertTrue(any(s in w or w in s for w in want), f"{s!r} 不在载荷 {want} 里")
        # 文本片的片段绝不能被画进图里
        for u in self.txt_sg.rewrite:
            self.assertNotIn(u["surface"], red.drawn_strings)

    def test_refuses_empty_payload(self):
        with self.assertRaises(ValueError):
            V.render_a0(self.txt_sg, V.RenderSpec(), self.g.behavior_id)

    def test_fragment_reassembles(self):
        want = "".join(u["surface"] for u in self.img_sg.rewrite if u["channel"] == P.CARRIER_IMAGE)
        red = V.render_a0(self.img_sg, V.RenderSpec(fragment="split_within_image"),
                          self.g.behavior_id)[0]
        self.assertEqual("".join(red.drawn_strings), want)

    def test_all_layouts_render(self):
        for lay in ("caption", "axis_label", "table", "legend"):
            red = V.render_a0(self.img_sg, V.RenderSpec(layout=lay), self.g.behavior_id)[0]
            self.assertTrue(red.png)
            self.assertTrue(red.drawn_strings)

    def test_occlusion_reduces_visible_text(self):
        clean = V.render_a0(self.img_sg, V.RenderSpec(), self.g.behavior_id)[0]
        occ = V.render_a0(self.img_sg, V.RenderSpec(occlusion=0.3), self.g.behavior_id)[0]
        self.assertEqual(clean.metrics["text_occluded_frac"], 0.0)
        self.assertGreater(occ.metrics["text_occluded_frac"], 0.0)
        self.assertNotEqual(clean.sha256, occ.sha256)

    def test_rotation_changes_pixels(self):
        base = V.render_a0(self.img_sg, V.RenderSpec(), self.g.behavior_id)[0]
        rot = V.render_a0(self.img_sg, V.RenderSpec(rotation_deg=45), self.g.behavior_id)[0]
        self.assertNotEqual(base.sha256, rot.sha256)


class TestOnlyA0(unittest.TestCase):
    """A1(图结构型)已按决定删除;它的结论保留在 `visual.py` 的模块 docstring 里。"""

    def test_a1_is_gone(self):
        import sg.visual as Vmod
        self.assertFalse(hasattr(Vmod, "render_a1"))

    def test_unknown_arm_is_rejected(self):
        g, pair, img_sg, _ = fixture()
        with self.assertRaises(ValueError):
            V.sweep_specs("a1")
        with self.assertRaises(ValueError):
            V.render_condition(g, pair, V.RenderSpec(arm="a1"))

    def test_report_records_the_decision(self):
        g, pair, img_sg, _ = fixture()
        with workspace_tempdir() as td:
            res = V.write_condition_set(g, pair, V.sweep_specs("a0", quick=True), td)
            report = V.render_report(g, pair, res, created="t")
            self.assertIn("A1", report)
            self.assertIn("身份进不去", report)


class TestConditionSet(unittest.TestCase):
    def test_manifest_roundtrip(self):
        g, pair, img_sg, _ = fixture()
        with workspace_tempdir() as td:
            res = V.write_condition_set(g, pair, V.sweep_specs("a0", quick=True), td)
            rows = [json.loads(l) for l in Path(res["manifest"]).read_text(encoding="utf-8").splitlines() if l]
            self.assertEqual(len(rows), res["n_images"])
            self.assertTrue(rows)
            for r in rows:
                self.assertTrue(Path(td, r["png_path"]).exists())
                self.assertEqual(r["arm"], "a0")
                self.assertIn("patch_occupancy", r["metrics"])
                # 文本片必须是原文片段的拼接(与 project 层同一纪律)
                for part in r["text_piece"].split(P.SEP):
                    if part:
                        self.assertIn(part, g.goal)
                # 报告里必须能看到"云上还要跑什么"
            report = V.render_report(g, pair, res, created="t")
            self.assertIn("判据 B1", report)      # 云上要跑什么,必须写在报告里
            self.assertIn("patch", report)
            self.assertIn("身份进不去", report)   # A1 的结论必须留在报告里


if __name__ == "__main__":
    unittest.main(verbosity=2)
