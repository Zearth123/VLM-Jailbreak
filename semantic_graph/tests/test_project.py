#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""投影层:子图文本必须**逐段回得到原文**,不许弱化、不许造词。

这是本步最重要的一条性质 —— 用户明确要求"只做图建模和子图拆分,
不做文本弱化,也不添加大量新内容和语义"。
"""

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from sg import extract, sources, twosplit as T  # noqa: E402
from sg import project as P  # noqa: E402

BY_ID = {r.rid: r for r in sources.load_harmbench()}
RID = "safrole_oil_extraction_guide"


def graph(rid: str = RID):
    return extract.build_graph(rid, BY_ID[rid].goal)


class TestProjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = graph()
        cls.pair = T.build_pair(cls.g, T.recommend_groups(cls.g))

    def test_no_added_characters(self):
        """投影文本里**没有任何**原文没有的字符(最强的"不新增"断言)。"""
        tr = P.traceability(self.g, self.pair)
        self.assertTrue(tr["all_traceable"], tr["untraceable"])
        self.assertEqual(tr["added_chars"], [])
        self.assertEqual(tr["non_ascii_added"], [], "渲染文本里出现了原文没有的非 ASCII 字符")

    def test_every_part_is_a_goal_substring(self):
        for sg in self.pair:
            for part in sg.rendered_text.split(P.SEP):
                self.assertIn(part.strip(), self.g.goal)

    def test_span_surface_is_verbatim(self):
        for sg in self.pair:
            for row in sg.rewrite:
                s, t = row["span"]
                self.assertEqual(row["surface"], self.g.goal[s:t])
                self.assertEqual(row["label"], self.g.node(row["nid"]).label)

    def test_no_rewriting_fields(self):
        """审计里不能出现"改写后"这种字段 —— 本步没有改写。"""
        for sg in self.pair:
            for row in sg.rewrite:
                self.assertEqual(row["strategy"], "project")
                self.assertNotIn("rewritten", row)

    def test_partition_covers_every_unit_once(self):
        seen: list[str] = []
        for sg in self.pair:
            seen.extend(sg.node_ids)
        self.assertEqual(sorted(seen), sorted(n.nid for n in self.g.nodes))


class TestChannels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = graph()
        cls.pair = T.build_pair(cls.g, T.recommend_groups(cls.g))

    def test_product_goes_to_image_and_out_of_text(self):
        b = next(sg for sg in self.pair if sg.sid == "SG-B")
        self.assertIn("text+image", b.channel)
        self.assertNotIn("MDMA", b.rendered_text)
        self.assertIn("MDMA", b.visual_spec or "")
        self.assertIn("[109:113]", b.visual_spec or "")

    def test_image_payload_has_no_prose(self):
        """image_payload 只能是"片段+span"的清单,不能是散文或新语义。"""
        b = next(sg for sg in self.pair if sg.sid == "SG-B")
        spec = b.visual_spec or ""
        self.assertTrue(spec.startswith("image_payload:"))
        body = spec.split(":", 1)[1]
        for piece in body.split("; "):
            surface, _, span = piece.partition("[")
            self.assertIn(surface.strip(), self.g.goal)
            self.assertTrue(span.endswith("]"))

    def test_text_only_subgraph_channel(self):
        a = next(sg for sg in self.pair if sg.sid == "SG-A")
        self.assertEqual(a.channel, "text")
        self.assertIsNone(a.visual_spec)


class TestDeferred(unittest.TestCase):
    def test_deferred_units_are_reported_not_dropped(self):
        g = graph()
        ids = [n.nid for n in g.nodes]
        carrier = {nid: P.CARRIER_TEXT for nid in ids}
        carrier[ids[0]] = P.CARRIER_DEFERRED
        sg = P.project_subgraph(g, ids, "SG-X", carrier_of=carrier)
        deferred = [r["nid"] for r in sg.rewrite if r["channel"] == P.CARRIER_DEFERRED]
        self.assertEqual(deferred, [ids[0]])
        self.assertNotIn(P.surface_of(g, ids[0]), sg.rendered_text)

    def test_carriers_tuple_is_declared(self):
        self.assertEqual(P.CARRIERS, ("text", "image", "deferred"))


class TestSecondExample(unittest.TestCase):
    def test_multi_root_goal_also_projects_verbatim(self):
        rid = "mdma_precursors_acquisition_us"
        g = graph(rid)
        pair = T.build_pair(g, T.recommend_groups(g))
        tr = P.traceability(g, pair)
        self.assertTrue(tr["all_traceable"], tr["untraceable"])
        self.assertEqual(tr["added_chars"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
