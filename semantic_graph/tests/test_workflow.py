"""Meaningful invariants using benign text and mocked HTTP. No paid calls."""

import copy
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from semantic_graph.client import APIError, DeepSeekClient, ModelConfig
from semantic_graph.evaluation import evaluate, repeat_consistency
from semantic_graph.io import load_goals, read_jsonl, select_samples, write_jsonl
from semantic_graph.pipeline import extract_goal, run_extraction
from semantic_graph.report import build_report
from semantic_graph.schema import GoalRecord, GraphError, align_evidence, graph_payload, validate_graph, validate_response

TEXT = "Wash the cup without soap."


def evidence(quote, occurrence=1):
    return {"quote": quote, "occurrence": occurrence}


def payload():
    return {"status": "ok", "issues": [], "graph": {
        "root_ids": ["a"], "nodes": [
            {"id": "a", "type": "action", "evidence": evidence("Wash")},
            {"id": "b", "type": "entity", "evidence": evidence("cup")},
            {"id": "c", "type": "constraint", "evidence": evidence("without soap")},
        ], "edges": [
            {"source": "a", "target": "b", "type": "patient", "evidence": evidence("Wash the cup")},
            {"source": "c", "target": "a", "type": "constraint_of", "evidence": evidence(TEXT)},
        ]}}


class FakeClient:
    config = ModelConfig()

    def __init__(self, responses=None, fatal=False):
        self.calls = 0
        self.responses = responses or [payload()]
        self.fatal = fatal

    def complete(self, messages):
        self.calls += 1
        if self.fatal:
            raise APIError("DeepSeek HTTP 401", fatal=True)
        response = self.responses[min(self.calls - 1, len(self.responses) - 1)]
        return {"content": json.dumps(response), "refusal": False, "finish_reason": "stop",
                "model": "fake-test-only", "usage": {"total_tokens": 1}}


class TestGrounding(unittest.TestCase):
    def test_unicode_and_repeated_quote_alignment(self):
        ev = align_evidence(evidence("茶", 2), "🍵茶和茶")
        self.assertEqual((ev["start"], ev["end"]), (3, 4))
        self.assertEqual(align_evidence(evidence("aa", 2), "aaa")["start"], 1)

    def test_missing_and_stale_source_evidence_rejected(self):
        with self.assertRaises(GraphError):
            align_evidence(evidence("boil water"), TEXT)
        with self.assertRaises(GraphError):
            align_evidence({**evidence("cup"), "start": 0}, TEXT)

    def test_preserves_negated_constraint_and_roundtrip(self):
        graph = validate_response(payload(), TEXT)["graph"]
        constraint = next(n for n in graph["nodes"] if n["type"] == "constraint")
        self.assertEqual(constraint["surface_text"], "without soap")
        self.assertEqual(validate_graph(graph_payload(graph), TEXT), graph)

    def test_model_node_numbering_does_not_change_graph(self):
        first = payload()["graph"]
        second = copy.deepcopy(first)
        second["nodes"].reverse()
        second["edges"].reverse()
        mapping = {"a": "x", "b": "z", "c": "y"}
        for n in second["nodes"]:
            n["id"] = mapping[n["id"]]
        second["root_ids"] = [mapping[r] for r in second["root_ids"]]
        for e in second["edges"]:
            e["source"], e["target"] = mapping[e["source"]], mapping[e["target"]]
        self.assertEqual(validate_graph(first, TEXT), validate_graph(second, TEXT))

    def test_duplicates_dangling_and_edge_evidence_rejected(self):
        for mutation in ("duplicate", "dangling", "evidence"):
            p = payload()
            if mutation == "duplicate":
                p["graph"]["nodes"].append(p["graph"]["nodes"][0])
            elif mutation == "dangling":
                p["graph"]["edges"][0]["target"] = "missing"
            else:
                p["graph"]["edges"][0]["evidence"] = evidence("Wash")
            with self.subTest(mutation=mutation), self.assertRaises(GraphError):
                validate_response(p, TEXT)

    def test_multiple_roots_do_not_require_temporal_edge(self):
        text = "Wash and dry."
        graph = {"root_ids": ["a", "b"], "nodes": [
            {"id": "a", "type": "action", "evidence": evidence("Wash")},
            {"id": "b", "type": "action", "evidence": evidence("dry")},
        ], "edges": []}
        validated = validate_graph(graph, text)
        self.assertEqual(len(validated["root_ids"]), 2)
        self.assertEqual(validated["edges"], [])

    def test_lists_quantities_locations_and_conditions(self):
        text = "List 2 cups in the kitchen when clean."
        graph = {"root_ids": ["a"], "nodes": [
            {"id": "a", "type": "action", "evidence": evidence("List")},
            {"id": "b", "type": "entity", "evidence": evidence("cups")},
            {"id": "q", "type": "quantity", "evidence": evidence("2")},
            {"id": "l", "type": "entity", "evidence": evidence("kitchen")},
            {"id": "c", "type": "condition", "evidence": evidence("when clean")},
        ], "edges": [
            {"source": "a", "target": "b", "type": "patient", "evidence": evidence("List 2 cups")},
            {"source": "q", "target": "b", "type": "quantity_of", "evidence": evidence("2 cups")},
            {"source": "a", "target": "l", "type": "located_at", "evidence": evidence("List 2 cups in the kitchen")},
            {"source": "c", "target": "a", "type": "condition_of", "evidence": evidence(text)},
        ]}
        self.assertEqual(len(validate_graph(graph, text)["nodes"]), 5)

    def test_refusal_and_partial_contract(self):
        r = validate_response({"status": "refused", "issues": ["unavailable"], "graph": None}, TEXT)
        self.assertIsNone(r["graph"])
        p = payload()
        p["status"] = "partial"
        with self.assertRaises(GraphError):
            validate_response(p, TEXT)

    def test_origin_is_separate_from_patient(self):
        text = "Take the cup from the shelf."
        graph = {"root_ids": ["a"], "nodes": [
            {"id": "a", "type": "action", "evidence": evidence("Take")},
            {"id": "b", "type": "entity", "evidence": evidence("cup")},
            {"id": "c", "type": "entity", "evidence": evidence("shelf")},
        ], "edges": [
            {"source": "a", "target": "b", "type": "patient", "evidence": evidence("Take the cup")},
            {"source": "a", "target": "c", "type": "origin", "evidence": evidence(text)},
        ]}
        self.assertIn("origin", [e["type"] for e in validate_graph(graph, text)["edges"]])

    def test_malformed_status_is_validation_failure(self):
        p = payload()
        p["status"] = []
        with self.assertRaises(GraphError):
            validate_response(p, TEXT)


class TestInputs(unittest.TestCase):
    def test_bom_quoted_multiline_and_filter_before_duplicate_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "goals.csv"
            path.write_text('Behavior,BehaviorID,FunctionalCategory,SemanticCategory\n"Wash, then\ndry.",x,standard,harmful\nOther,x,contextual,harmful\n', encoding="utf-8-sig", newline="")
            goals = load_goals(path, "standard")
            self.assertEqual(goals[0].text, "Wash, then\ndry.")
            with self.assertRaises(ValueError):
                load_goals(path)

    def test_bad_id_text_and_empty_filter_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            for rows in ([{"id": "x", "text": ""}], [{"id": "x", "text": "a"}] * 2, [{"text": "a"}]):
                write_jsonl(path, rows)
                with self.assertRaises(ValueError):
                    load_goals(path)
            write_jsonl(path, [{"id": "x", "text": "a"}])
            with self.assertRaises(ValueError):
                load_goals(path, "standard")

    def test_frozen_stratified_sample(self):
        from semantic_graph.io import QUOTAS
        goals = [GoalRecord(f"{category}{i}", "Wash.", category) for category in QUOTAS for i in range(8)]
        a, splits = select_samples(goals)
        b, other_splits = select_samples(list(reversed(goals)))
        self.assertEqual(a, b)
        self.assertEqual(splits, other_splits)
        self.assertEqual(sum(s == "dev" for s in splits.values()), 10)
        self.assertEqual(sum(s == "holdout" for s in splits.values()), 10)


class TestPipeline(unittest.TestCase):
    def test_cache_schema_repair_and_independent_repeat(self):
        goal = GoalRecord("x", TEXT)
        bad = payload()
        bad["graph"]["nodes"][0]["evidence"] = evidence("invented")
        client = FakeClient([bad, payload()])
        with tempfile.TemporaryDirectory() as tmp:
            first = extract_goal(goal, client, tmp)
            self.assertEqual(client.calls, 2)
            self.assertEqual(first["status"], "ok")
            self.assertTrue(extract_goal(goal, client, tmp)["cache_hit"])
            self.assertEqual(client.calls, 2)
            self.assertFalse(extract_goal(goal, client, tmp, 1)["cache_hit"])
            self.assertEqual(client.calls, 3)

    def test_no_refusal_loop(self):
        client = FakeClient([{"status": "refused", "issues": ["refused"], "graph": None}])
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(extract_goal(GoalRecord("x", TEXT), client, tmp)["status"], "refused")
            self.assertEqual(client.calls, 1)

    def test_auth_failure_stops_before_full_dataset(self):
        goals = [GoalRecord(str(i), TEXT) for i in range(20)]
        client = FakeClient(fatal=True)
        with tempfile.TemporaryDirectory() as tmp:
            rows, meta = run_extraction(goals, client, Path(tmp) / "run", Path(tmp) / "cache", workers=2, progress=lambda _: None)
            self.assertEqual(len(rows), 2)
            self.assertFalse(meta["complete"])
            self.assertEqual(build_report(Path(tmp) / "run")["n_unprocessed"], 18)

    def test_run_report_resume_and_fingerprint_guard(self):
        client = FakeClient()
        with tempfile.TemporaryDirectory() as tmp:
            out, cache = Path(tmp) / "run", Path(tmp) / "cache"
            run_extraction([GoalRecord("x", TEXT)], client, out, cache, progress=lambda _: None)
            metrics = build_report(out)
            self.assertEqual(metrics["coverage_ok"], 1)
            self.assertIsNone(metrics["fidelity"])
            self.assertIn("尚未完成", (out / "report.md").read_text(encoding="utf-8"))
            run_extraction([GoalRecord("x", TEXT)], client, out, cache, progress=lambda _: None)
            self.assertEqual(client.calls, 1)
            with self.assertRaises(ValueError):
                run_extraction([GoalRecord("other", TEXT)], client, out, cache)

    def test_viewer_does_not_execute_source_html(self):
        from semantic_graph.report import VIEWER
        # Source content is JSON-escaped and no innerHTML assignment is used.
        self.assertNotIn("innerHTML", VIEWER)
        value = json.dumps({"text": "</script><script>alert(1)</script>"}).replace("<", "\\u003c")
        self.assertNotIn("</script>", value)


class TestQuality(unittest.TestCase):
    def _reference(self):
        return {"id": "x", "text": TEXT, "split": "holdout", "reference_type": "human",
                "annotation": {"status": "adjudicated", "annotators": ["reviewer-a", "reviewer-b"]},
                "graph": validate_response(payload(), TEXT)["graph"]}

    def test_f1_exact_and_failed_predictions_remain_in_recall_denominator(self):
        gold = self._reference()
        prediction = {"id": "x", "text": TEXT, "graph": gold["graph"], "status": "ok"}
        self.assertEqual(evaluate([prediction], [gold])["micro"]["edges"]["f1"], 1)
        failed = {**prediction, "status": "error", "graph": None}
        score = evaluate([failed], [gold])["micro"]["nodes"]
        self.assertEqual(score["gold"], 3)
        self.assertEqual(score["recall"], 0)

    def test_pending_or_single_annotator_reference_rejected(self):
        for change in ("pending", "single"):
            gold = self._reference()
            if change == "pending":
                gold["annotation"]["status"] = "pending"
            else:
                gold["annotation"]["annotators"] = ["reviewer-a"]
            with self.assertRaises(ValueError):
                evaluate([], [gold])

    def test_consistency_rejects_cached_replays_as_independent_repeats(self):
        graph = self._reference()["graph"]
        row = {"id": "x", "text": TEXT, "graph": graph, "status": "ok", "repeat_id": 0}
        with self.assertRaises(ValueError):
            repeat_consistency([[row], [row]])
        repeated = {**row, "repeat_id": 1}
        self.assertEqual(repeat_consistency([[row], [repeated]])["valid_graph_agreement"], 1)


class TestTransport(unittest.TestCase):
    def test_unauthorized_does_not_retry_or_echo_secret(self):
        client = DeepSeekClient(ModelConfig(), api_key="test-secret")
        error = urllib.error.HTTPError("https://api.deepseek.com/models", 401, "unauthorized", {}, io.BytesIO(b"test-secret"))
        with patch("urllib.request.urlopen", side_effect=error) as call:
            with self.assertRaises(APIError) as caught:
                client.models()
            self.assertTrue(caught.exception.fatal)
            self.assertNotIn("test-secret", str(caught.exception))
            self.assertEqual(call.call_count, 1)

    def test_429_retries_then_recovers(self):
        client = DeepSeekClient(ModelConfig(), api_key="test-secret")
        error = urllib.error.HTTPError("https://api.deepseek.com/models", 429, "limited", {}, io.BytesIO())
        response = io.BytesIO(b'{"data":[{"id":"deepseek-flash"}]}')
        with patch("urllib.request.urlopen", side_effect=[error, response]) as call, patch("time.sleep"):
            self.assertEqual(client.models(), ["deepseek-flash"])
            self.assertEqual(call.call_count, 2)

    def test_credential_in_url_and_plain_http_rejected(self):
        for url in ("http://api.deepseek.com", "https://key@api.deepseek.com", "https://api.deepseek.com?key=x"):
            with self.assertRaises(ValueError):
                ModelConfig(base_url=url)


if __name__ == "__main__":
    unittest.main()

