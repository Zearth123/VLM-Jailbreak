#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""放宽拆分:边重表述、新不变量、两档可拆性。"""

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from sg import extract, sources  # noqa: E402
from sg import split as S  # noqa: E402
from sg.probe import LexicalProbe  # noqa: E402
from sg.project import project_subgraph  # noqa: E402

PROBE = LexicalProbe()
BY_ID = {r.rid: r for r in sources.load_harmbench()}
RID = "safrole_oil_extraction_guide"


def graph(rid: str = RID):
    return extract.build_graph(rid, BY_ID[rid].goal)


class TestEdgeCarrier(unittest.TestCase):
    def test_default_requires_colocation(self):
        g = graph()
        self.assertTrue(all(e.requires_colocation for e in g.dependency_edges))

    def test_relaxed_copy_does_not_mutate_original(self):
        g = graph()
        r = g.relaxed()
        self.assertTrue(all(e.requires_colocation for e in g.dependency_edges))
        self.assertFalse(any(e.requires_colocation for e in r.dependency_edges))
        self.assertEqual(len(r.nodes), len(g.nodes))
        self.assertEqual([e.src for e in r.edges], [e.src for e in g.edges])

    def test_carrier_policy_by_type(self):
        # 承载方式只对**已声明可放宽**的边有意义 → 在放宽后的图上取策略
        g = graph().relaxed()
        by_type = {e.etype: S.carrier_for(e) for e in g.dependency_edges}
        self.assertEqual(by_type["precursor-of"], "image")
        self.assertEqual(by_type["precedes"], "ordering")

    def test_strict_graph_edges_are_not_relaxable(self):
        by_type = {}
        for e in graph().dependency_edges:
            by_type[e.etype] = S.carrier_for(e)
        self.assertEqual(set(by_type.values()), {"ordering"})  # 无候选 → 兜底


class TestSupernodes(unittest.TestCase):
    def test_strict_vs_relaxed(self):
        g = graph()
        self.assertEqual(len(S.dependency_components(g)), 2)
        self.assertEqual(len(S.dependency_components(g.relaxed())), 5)

    def test_relaxed_enumeration_counts(self):
        g = graph().relaxed()
        supernodes = S.dependency_components(g)
        labels, truncated = S.enumerate_partitions(supernodes, 2, image_budget=None)
        self.assertEqual(len(labels), 15)  # S(5,2) = 15
        self.assertFalse(truncated)
        self.assertTrue(all(len(set(x)) == 2 for x in labels))


class TestRelaxedSplit(unittest.TestCase):
    def test_relaxed_beats_strict_on_max_score(self):
        g = graph()
        strict = S.split_s2(g, PROBE, 2)
        relaxed = S.split_relaxed(g, PROBE, 2)
        self.assertLess(relaxed.max_fragment_score, strict.max_fragment_score)
        self.assertEqual(strict.feasible_partitions, 1)
        self.assertEqual(relaxed.feasible_partitions, 15)

    def test_cut_edges_are_recorded_and_invariant_passes(self):
        g = graph()
        res = S.split_relaxed(g, PROBE, 2)
        subgraphs = [
            project_subgraph(g, f.node_ids, sid=f"SG-{i}")
            for i, f in enumerate(res.fragments)
        ]
        for f, sg in zip(res.fragments, subgraphs):
            sg.reexpressed_edges = list(f.reexpressed_edges)
        self.assertEqual(S.check_relaxation(g.relaxed(), subgraphs), [])
        # 严格图下同样的分区必须被判定为违例
        self.assertTrue(S.check_relaxation(g, subgraphs))

    def test_relaxed_decomposability_is_not_worse(self):
        g = graph()
        strict = S.decomposability(g, PROBE, ns=(2, 3), tau=0.35)
        relaxed = S.decomposability(g, PROBE, ns=(2, 3), tau=0.35, require_colocation=False)
        self.assertLessEqual(relaxed["best_max_score"], strict["best_max_score"])
        self.assertEqual(relaxed["n_supernodes"], 5)
        self.assertTrue(relaxed["relaxed"])
        self.assertFalse(strict["relaxed"])


class TestCheckRelaxation(unittest.TestCase):
    def test_missing_record_is_reported(self):
        g = graph()
        res = S.split_relaxed(g, PROBE, 2)
        subgraphs = [
            project_subgraph(g, f.node_ids, sid=f"SG-{i}")
            for i, f in enumerate(res.fragments)
        ]
        problems = S.check_relaxation(g.relaxed(), subgraphs)
        self.assertTrue(problems)
        self.assertIn("没有重表述记录", problems[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
