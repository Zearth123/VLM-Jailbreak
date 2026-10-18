"""Semantic preservation invariants over benign, source-aligned examples."""

import copy
import unittest

from semantic_graph.schema import graph_payload, validate_graph
from semantic_split import split_record
from semantic_split.examples import benign_examples


class TestDecomposition(unittest.TestCase):
    def assert_preserved(self, row, result):
        original = validate_graph(graph_payload(row["graph"]), row["text"])
        node_index = {node["id"]: node for node in result["graph"]["nodes"]}
        edge_index = {edge["id"]: edge for edge in result["edge_index"]}
        nodes, edges = set(), set()
        for fragment in result["fragments"]:
            nodes.update(fragment["node_ids"])
            edges.update(fragment["edge_ids"])
            for eid in fragment["edge_ids"]:
                edge = edge_index[eid]
                self.assertIn(edge["source"], fragment["node_ids"])
                self.assertIn(edge["target"], fragment["node_ids"])
            for span in [*fragment["source_spans"], fragment["source_context"]]:
                self.assertEqual(span["quote"], row["text"][span["start"]:span["end"]])
        for relation in result["cross_relations"]:
            edges.add(relation["edge_id"])
            for endpoint in ("source", "target"):
                owners = [f["id"] for f in result["fragments"]
                          if relation[endpoint] in f["node_ids"]]
                self.assertEqual(owners, relation[endpoint + "_fragment_ids"])
        reconstructed = {
            "root_ids": result["graph"]["root_ids"],
            "nodes": [node_index[nid] for nid in nodes],
            "edges": [{k: edge_index[eid][k] for k in ("source", "target", "type", "evidence")}
                      for eid in edges],
        }
        self.assertEqual(validate_graph(graph_payload(reconstructed), row["text"]), original)
        self.assertEqual(result["text"], row["text"])

    def test_all_examples_reconstruct_the_entire_source_graph(self):
        for row in benign_examples():
            with self.subTest(row=row["id"]):
                result = split_record(row)
                self.assertEqual(result["split_status"], "ok")
                self.assert_preserved(row, result)

    def test_quantity_and_negation_stay_with_their_target(self):
        result = split_record(benign_examples()[0])
        first, second = result["fragments"]
        by_id = {n["id"]: n["surface_text"] for n in result["graph"]["nodes"]}
        self.assertEqual(len(result["fragments"]), 2)
        self.assertTrue({"两个", "红色杯子", "不要叠放"} <= {by_id[nid] for nid in first["node_ids"]})
        self.assertNotIn("不要叠放", {by_id[nid] for nid in second["node_ids"]})
        self.assertEqual([e["type"] for e in result["cross_relations"]], ["precedes"])

    def test_shared_entity_does_not_collapse_distinct_actions(self):
        result = split_record(benign_examples()[1])
        self.assertEqual(len(result["fragments"]), 2)
        shared = {n["surface_text"] for n in result["graph"]["nodes"] if n["id"] in result["shared_node_ids"]}
        self.assertEqual(shared, {"cup", "both actions must avoid soap"})
        self.assertEqual(result["constraint_scopes"][0]["fragment_ids"], ["f1", "f2"])
        self.assertFalse(any(e["type"] == "precedes" for e in result["edge_index"]))

    def test_request_frame_and_inner_activity_remain_separate(self):
        result = split_record(benign_examples()[2])
        self.assertEqual(len(result["fragments"]), 2)
        self.assertEqual(len(result["cross_relations"]), 1)
        self.assertEqual(result["cross_relations"][0]["type"], "patient")
        self.assertEqual(result["fragments"][0]["root_ids"], result["graph"]["root_ids"])

    def test_artifact_only_request_has_a_fragment(self):
        result = split_record(benign_examples()[3])
        self.assertEqual(len(result["fragments"]), 1)
        self.assertEqual(result["fragments"][0]["kind"], "artifact")
        self.assertEqual(result["coverage"]["nodes_preserved"], 3)

    def test_produced_artifact_is_local_and_purpose_activity_is_separate(self):
        text = "Write a list to organize books."
        row = {"id": "purpose", "text": text, "graph": {
            "root_ids": ["write", "list"], "nodes": [
                {"id": nid, "type": kind, "evidence": {"quote": quote, "occurrence": 1}}
                for nid, kind, quote in (("write", "action", "Write"), ("list", "artifact", "list"),
                                         ("organize", "action", "organize"), ("books", "entity", "books"))
            ], "edges": [
                {"source": source, "target": target, "type": kind,
                 "evidence": {"quote": text, "occurrence": 1}}
                for source, target, kind in (("write", "list", "produces"),
                                             ("write", "organize", "purpose"),
                                             ("organize", "books", "patient"))
            ]}}
        result = split_record(row)
        self.assertEqual(len(result["fragments"]), 2)
        self.assertEqual(len(result["fragments"][0]["root_ids"]), 2)
        self.assertEqual([e["type"] for e in result["cross_relations"]], ["purpose"])
        self.assert_preserved(row, result)

    def test_unattached_evidence_is_retained_and_flagged(self):
        row = benign_examples()[0]
        row["graph"]["edges"] = [e for e in row["graph"]["edges"] if e["type"] != "located_at"]
        result = split_record(row)
        self.assertEqual(result["split_status"], "partial")
        self.assertEqual(sum(f["kind"] == "context" for f in result["fragments"]), 1)
        self.assert_preserved(row, result)

    def test_unicode_offsets_and_repeated_occurrences(self):
        text = "🍵洗杯子，然后擦干杯子。"
        row = {"id": "unicode", "text": text, "graph": {
            "root_ids": ["wash", "dry"], "nodes": [
                {"id": nid, "type": kind, "evidence": {"quote": quote, "occurrence": occurrence}}
                for nid, kind, quote, occurrence in (
                    ("wash", "action", "洗", 1), ("cup1", "entity", "杯子", 1),
                    ("dry", "action", "擦干", 1), ("cup2", "entity", "杯子", 2))
            ], "edges": []}}
        for action, cup in (("wash", "cup1"), ("dry", "cup2")):
            row["graph"]["edges"].append({"source": action, "target": cup, "type": "patient",
                                            "evidence": {"quote": text, "occurrence": 1}})
        result = split_record(row)
        self.assertEqual(result["split_status"], "ok")
        cups = [n for n in result["graph"]["nodes"] if n["type"] == "entity"]
        self.assertEqual([n["evidence"]["start"] for n in cups], [2, 9])
        self.assert_preserved(row, result)

    def test_partial_source_is_never_promoted(self):
        row = benign_examples()[0]
        row.update(status="partial", issues=["Review scope."])
        result = split_record(row)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["issues"], row["issues"])
        self.assertEqual(result["split_status"], "partial")

    def test_failed_sources_are_not_fabricated_or_retried(self):
        for status in ("refused", "invalid", "error"):
            row = {"id": "failed", "text": "Draw a cup.", "status": status, "issues": [], "graph": None}
            result = split_record(row)
            self.assertEqual(result["status"], status)
            self.assertEqual(result["split_status"], "skipped")
            self.assertEqual(result["fragments"], [])

    def test_bad_evidence_and_bad_status_graph_pair_fail_explicitly(self):
        row = benign_examples()[0]
        row["graph"]["nodes"][0]["evidence"]["quote"] = "invented"
        self.assertEqual(split_record(row)["split_status"], "invalid")
        row = benign_examples()[0]
        row["status"] = "refused"
        self.assertEqual(split_record(row)["split_status"], "invalid")

    def test_missing_graph_requires_extraction(self):
        result = split_record({"id": "raw", "text": "Draw a cup."})
        self.assertEqual(result["split_status"], "missing_graph")
        self.assertEqual(result["fragments"], [])

    def test_canonical_ids_and_input_metadata_are_preserved(self):
        first = benign_examples()[0]
        first["metadata"] = {"note": "original"}
        saved = copy.deepcopy(first)
        second = copy.deepcopy(first)
        graph = second["graph"]
        mapping = {n["id"]: "renamed-" + n["id"] for n in graph["nodes"]}
        for node in graph["nodes"]:
            node["id"] = mapping[node["id"]]
        graph["root_ids"] = [mapping[r] for r in graph["root_ids"]]
        for edge in graph["edges"]:
            edge["source"], edge["target"] = mapping[edge["source"]], mapping[edge["target"]]
        graph["nodes"].reverse()
        graph["edges"].reverse()
        result = split_record(first)
        self.assertEqual(result, split_record(second))
        self.assertEqual(first, saved)
        self.assertEqual(result["metadata"], saved["metadata"])
        self.assertEqual(result, split_record(result))


if __name__ == "__main__":
    unittest.main()
