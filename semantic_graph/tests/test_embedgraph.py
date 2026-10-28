#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""向量建图与切分(`sg/embedgraph.py`)的测试。

用**合成 dump** 钉住:片段可回原文、亲和计算、切分约束、可切代价、投影与判据接线。
不需要任何模型;numpy 不可用时整组跳过。
"""

import json
import re
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from sg import embedgraph as EG  # noqa: E402
from sg import extract, sources, threshold as TH  # noqa: E402

RID = "safrole_oil_extraction_guide"
BY_ID = {r.rid: r for r in sources.load_harmbench()}
GOAL = BY_ID[RID].goal

# 让同一语义单元内的词向量相同,跨单元正交 —— 亲和矩阵因此可手算
VEC = {
    "extract": (1.0, 0.0, 0.0),
    "production": (0.0, 0.6, 0.8),
    "safrole": (0.0, 1.0, 0.0),
    "oil": (0.0, 1.0, 0.0),
    "sassafras": (0.0, 1.0, 0.0),
    "roots": (0.0, 1.0, 0.0),
    "MDMA": (0.0, 0.0, 1.0),
}
FILLER = (0.3, 0.3, 0.3)


def _np():
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        raise unittest.SkipTest("numpy 不可用")
    return np


def write_synthetic_dump(dirpath: Path, texts: list[dict]) -> Path:
    np = _np()
    vectors, offsets, text_ids, token_ids = [], [], [], []
    rows = []
    for ti, row in enumerate(texts):
        text = row["text"]
        words = [m for m in re.finditer(r"\S+", text)]
        rows.append({"text_index": ti, "rid": row["rid"], "text": text,
                     "n_tokens": len(words), "sha256": "x" * 16})
        for k, m in enumerate(words):
            vectors.append(VEC.get(m.group(0), FILLER))
            offsets.append([m.start(), m.end()])
            text_ids.append(ti)
            token_ids.append(1000 + k)
    dirpath.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dirpath / "emb.npz",
                        vectors=np.asarray(vectors, dtype="float32"),
                        offsets=np.asarray(offsets, dtype="int32"),
                        text_id=np.asarray(text_ids, dtype="int32"),
                        token_ids=np.asarray(token_ids, dtype="int32"))
    with open(dirpath / "emb.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (dirpath / "meta.json").write_text(json.dumps({"model_id": "synthetic", "layer": -1}),
                                       encoding="utf-8", newline="\n")
    return dirpath


def fixture():
    from tests import workspace_tempdir
    g = extract.build_graph(RID, GOAL)
    return g, workspace_tempdir


class TestDumpLoading(unittest.TestCase):
    def test_missing_dump_is_loud(self):
        from tests import workspace_tempdir
        with workspace_tempdir() as td:
            with self.assertRaises(EG.EmbedUnavailable):
                EG.load_embed_dump(Path(td) / "nope")

    def test_empty_jsonl_is_loud(self):
        from tests import workspace_tempdir
        with workspace_tempdir() as td:
            d = Path(td)
            _np()
            import numpy as np
            (d / "emb.jsonl").write_text("", encoding="utf-8")
            np.savez_compressed(d / "emb.npz", vectors=np.zeros((0, 3), "float32"),
                                offsets=np.zeros((0, 2), "int32"),
                                text_id=np.zeros((0,), "int32"))
            with self.assertRaises(EG.EmbedUnavailable):
                EG.load_embed_dump(d)

    def test_pick_text_requires_rid_when_ambiguous(self):
        from tests import workspace_tempdir
        with workspace_tempdir() as td:
            d = write_synthetic_dump(Path(td), [{"rid": "a", "text": "x y"},
                                                {"rid": "b", "text": "z"}])
            dump = EG.load_embed_dump(d)
            with self.assertRaises(EG.EmbedUnavailable):
                EG.pick_text(dump)
            self.assertEqual(EG.pick_text(dump, "b")[0], 1)


class TestFragments(unittest.TestCase):
    def test_text_equals_goal_span(self):
        g, tmp = fixture()
        with tmp() as td:
            dump = EG.load_embed_dump(write_synthetic_dump(Path(td), [{"rid": RID, "text": GOAL}]))
            frags = EG.fragments(g, dump, 0)
            self.assertTrue(frags)
            for f in frags:
                self.assertEqual(f.text, g.goal[f.span[0]:f.span[1]])

    def test_anchors_follow_spans(self):
        g, tmp = fixture()
        with tmp() as td:
            dump = EG.load_embed_dump(write_synthetic_dump(Path(td), [{"rid": RID, "text": GOAL}]))
            frags = EG.fragments(g, dump, 0)
            by_text = {f.text: f.nid for f in frags}
            self.assertEqual(by_text["MDMA"], "n003")
            self.assertEqual(by_text["extract"], "n004")
            # "safrole" 与 "oil" 是同一单元的两个词
            self.assertEqual(by_text["safrole"], "n001")
            self.assertEqual(by_text["oil"], "n001")

    def test_unmapped_words_are_counted(self):
        g, tmp = fixture()
        with tmp() as td:
            dump = EG.load_embed_dump(write_synthetic_dump(Path(td), [{"rid": RID, "text": GOAL}]))
            vg = EG.build(g, EG.fragments(g, dump, 0))
            self.assertGreater(vg.n_unmapped, 0)


class TestBuild(unittest.TestCase):
    def test_chain_edges_and_affinity(self):
        g, tmp = fixture()
        with tmp() as td:
            dump = EG.load_embed_dump(write_synthetic_dump(Path(td), [{"rid": RID, "text": GOAL}]))
            frags = EG.fragments(g, dump, 0)
            vg = EG.build(g, frags)
            self.assertEqual(len([e for e in vg.edges if e[3] == "chain"]), len(frags) - 1)
            # 两个 material 单元同向量 ⇒ 亲和 1.0;与 MDMA 正交 ⇒ 0.0
            self.assertAlmostEqual(vg.affinity[frozenset(("n001", "n002"))], 1.0, places=5)
            self.assertAlmostEqual(vg.affinity[frozenset(("n001", "n003"))], 0.0, places=5)
            # 同节点内部不产生亲和
            self.assertNotIn(frozenset(("n001", "n001")), vg.affinity)


class TestCut(unittest.TestCase):
    def test_groups_cover_all_nodes_once(self):
        g, tmp = fixture()
        with tmp() as td:
            dump = EG.load_embed_dump(write_synthetic_dump(Path(td), [{"rid": RID, "text": GOAL}]))
            vg = EG.build(g, EG.fragments(g, dump, 0))
            res = EG.cut(g, vg, k=2)
            self.assertTrue(res.groups, res.reason)
            flat = sorted(x for grp in res.groups for x in grp)
            self.assertEqual(flat, sorted(n.nid for n in g.nodes))

    def test_dependency_supernodes_stay_together(self):
        """依赖边不被切开:超级节点是原子单位。"""
        g, tmp = fixture()
        with tmp() as td:
            dump = EG.load_embed_dump(write_synthetic_dump(Path(td), [{"rid": RID, "text": GOAL}]))
            vg = EG.build(g, EG.fragments(g, dump, 0))
            res = EG.cut(g, vg, k=2)
            where = {nid: i for i, grp in enumerate(res.groups) for nid in grp}
            for e in g.dependency_edges:
                self.assertEqual(where[e.src], where[e.dst], f"{e.src}->{e.dst} 被切开")

    def test_infeasible_k_reports_reason(self):
        g, tmp = fixture()
        with tmp() as td:
            dump = EG.load_embed_dump(write_synthetic_dump(Path(td), [{"rid": RID, "text": GOAL}]))
            vg = EG.build(g, EG.fragments(g, dump, 0))
            res = EG.cut(g, vg, k=4)      # 只有 2 个超级节点
            self.assertFalse(res.groups)
            self.assertIn("不足", res.reason)


class TestCutCost(unittest.TestCase):
    def test_covers_every_dependency_edge(self):
        g, tmp = fixture()
        with tmp() as td:
            dump = EG.load_embed_dump(write_synthetic_dump(Path(td), [{"rid": RID, "text": GOAL}]))
            vg = EG.build(g, EG.fragments(g, dump, 0))
            costs = EG.cut_cost(g, vg)
            self.assertEqual(set(costs), {(e.src, e.dst) for e in g.dependency_edges})
            for v in costs.values():
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)

    def test_opposite_is_zero_cost(self):
        """同向强相似的依赖边可切代价最低(1 - 1 = 0)。"""
        g, tmp = fixture()
        with tmp() as td:
            dump = EG.load_embed_dump(write_synthetic_dump(Path(td), [{"rid": RID, "text": GOAL}]))
            vg = EG.build(g, EG.fragments(g, dump, 0))
            costs = EG.cut_cost(g, vg)
            self.assertAlmostEqual(costs[("n001", "n003")], 1.0, places=5)   # 正交 ⇒ 1.0


class TestProjection(unittest.TestCase):
    def test_subgraph_text_is_verbatim(self):
        from sg import project as P
        g, tmp = fixture()
        with tmp() as td:
            dump = EG.load_embed_dump(write_synthetic_dump(Path(td), [{"rid": RID, "text": GOAL}]))
            vg = EG.build(g, EG.fragments(g, dump, 0))
            res = EG.cut(g, vg, k=2)
            sgs = EG.to_subgraphs(g, res.groups)
            self.assertEqual(len(sgs), 2)
            self.assertEqual(P.traceability(g, sgs)["all_traceable"], True)
            for sg in sgs:
                self.assertTrue(sg.node_ids)


class TestRun(unittest.TestCase):
    def test_end_to_end_gate_is_untested(self):
        from tests import workspace_tempdir
        g, _ = fixture()
        with workspace_tempdir() as td:
            d = write_synthetic_dump(Path(td) / "dump", [{"rid": RID, "text": GOAL}])
            out = Path(td) / "report.md"
            res = EG.run(d, out, rid=RID, k=2, created="t")
            self.assertTrue(out.exists())
            self.assertEqual(res["gate"], "untested")     # 无读数 ⇒ 绝不 pass
            self.assertEqual(res["n_subgraphs"], 2)
            text = out.read_text(encoding="utf-8")
            self.assertIn("向量建图与切分", text)
            self.assertIn("untested", text)

    def test_determinism(self):
        from tests import workspace_tempdir
        g, _ = fixture()
        with workspace_tempdir() as td:
            d = write_synthetic_dump(Path(td) / "dump", [{"rid": RID, "text": GOAL}])
            a = EG.run(d, Path(td) / "a.md", rid=RID)
            b = EG.run(d, Path(td) / "b.md", rid=RID)
            self.assertEqual(a["cut"], b["cut"])
            self.assertEqual(a["affinity"], b["affinity"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
