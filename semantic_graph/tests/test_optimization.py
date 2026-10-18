"""Regression checks for scheduling, crash recovery, and source-only repairs."""

import copy
import json
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from semantic_graph.client import APIError, DeepSeekClient, ModelConfig
from semantic_graph.io import read_jsonl, recover_journal, write_jsonl
from semantic_graph.pipeline import cache_key, extract_goal, load_run, replay_run, run_extraction
from semantic_graph.report import build_report
from semantic_graph.schema import GoalRecord, GraphError, graph_payload, parse_response, result_record, validate_graph
from test_workflow import TEXT, FakeClient, evidence, payload


class TestLocalRepair(unittest.TestCase):
    def test_expands_exact_context_marks_partial_and_preserves_nodes(self):
        raw = payload()
        raw["graph"]["edges"][0]["evidence"] = evidence("cup")
        before = copy.deepcopy(raw)
        parsed = parse_response(raw, TEXT)
        self.assertEqual(parsed["status"], "partial")
        self.assertEqual(raw, before)
        self.assertEqual(parsed["repairs"][0]["after"]["quote"], "Wash the cup")
        self.assertEqual(parsed["graph"]["nodes"], parse_response(payload(), TEXT)["graph"]["nodes"])
        self.assertIn("without soap", [n["surface_text"] for n in parsed["graph"]["nodes"]])
        validate_graph(graph_payload(parsed["graph"]), TEXT)

    def test_repeated_unicode_context_keeps_correct_occurrence(self):
        text = "🍵 Wash cup. Wash cup."
        raw = {"status": "ok", "issues": [], "graph": {"root_ids": ["a"], "nodes": [
            {"id": "a", "type": "action", "evidence": evidence("Wash", 2)},
            {"id": "b", "type": "entity", "evidence": evidence("cup", 2)}], "edges": [
            {"source": "a", "target": "b", "type": "patient", "evidence": evidence("cup", 2)}]}}
        ev = parse_response(raw, text)["graph"]["edges"][0]["evidence"]
        self.assertEqual(ev, {"quote": "Wash cup", "occurrence": 2, "start": 12, "end": 20})

    def test_does_not_guess_types_endpoints_or_missing_evidence(self):
        for problem in ("type", "endpoint", "quote", "stale"):
            raw = payload()
            edge = raw["graph"]["edges"][0]
            edge["evidence"] = evidence("cup")
            if problem == "type":
                edge["type"] = "condition_of"
            elif problem == "endpoint":
                edge["target"] = "missing"
            elif problem == "quote":
                edge["evidence"] = evidence("invented")
            else:
                edge["evidence"]["start"] = 0
            with self.subTest(problem=problem), self.assertRaises(GraphError):
                parse_response(raw, TEXT)

    def test_local_repair_avoids_second_model_request(self):
        raw = payload()
        raw["graph"]["edges"][0]["evidence"] = evidence("cup")
        client = FakeClient([raw])
        with tempfile.TemporaryDirectory() as tmp:
            row = extract_goal(GoalRecord("x", TEXT), client, tmp)
        self.assertEqual(row["status"], "partial")
        self.assertEqual(client.calls, 1)


class TestJournal(unittest.TestCase):
    def test_recovers_only_unterminated_tail_including_split_unicode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_bytes(b'{"id":"first"}\n{"text":"\xe8\x8c')
            self.assertEqual(recover_journal(path), [{"id": "first"}])
            self.assertEqual(path.read_bytes(), b'{"id":"first"}\n')
            path.write_bytes(b'{"id":"first"}')
            self.assertEqual(recover_journal(path), [{"id": "first"}])
            self.assertTrue(path.read_bytes().endswith(b"\n"))

    def test_rejects_corrupt_complete_or_middle_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            for content in (b'bad\n', b'bad\n{"id":"x"}\n', b'[]'):
                path.write_bytes(content)
                with self.subTest(content=content), self.assertRaises(ValueError):
                    recover_journal(path)
                self.assertEqual(path.read_bytes(), content)

    def test_resume_uses_journal_without_cache_and_ignores_transport_changes(self):
        client = FakeClient()
        goal = GoalRecord("x", TEXT)
        with tempfile.TemporaryDirectory() as tmp:
            out, cache = Path(tmp) / "run", Path(tmp) / "cache"
            run_extraction([goal], client, out, cache, progress=lambda _: None)
            for entry in cache.iterdir():
                entry.unlink()
            with (out / "graphs.jsonl").open("ab") as stream:
                stream.write(b'{"id":')
            client.config = replace(client.config, timeout=12, retries=0)
            rows, meta = run_extraction([goal], client, out, cache, progress=lambda _: None)
            self.assertEqual(client.calls, 1)
            self.assertTrue(meta["complete"])
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(load_run(out)[0]), 1)

    def test_errors_are_retried_and_duplicate_attempts_compacted(self):
        goal = GoalRecord("x", TEXT)
        with tempfile.TemporaryDirectory() as tmp:
            out, cache = Path(tmp) / "run", Path(tmp) / "cache"
            client = FakeClient(fatal=True)
            run_extraction([goal], client, out, cache, progress=lambda _: None)
            journal = out / "graphs.jsonl"
            row = read_jsonl(journal)[0]
            write_jsonl(journal, [row, row])  # An interrupted retry left a second error.
            client.fatal = False
            rows, meta = run_extraction([goal], client, out, cache, progress=lambda _: None)
            self.assertEqual(rows[0]["status"], "ok")
            self.assertTrue(meta["complete"])
            self.assertEqual(len(read_jsonl(journal)), 1)


class TestScheduling(unittest.TestCase):
    def test_reuses_free_worker_while_first_task_is_still_running(self):
        progressed = threading.Event()
        waited_too_long = []
        graph = parse_response(payload(), TEXT)["graph"]

        def mocked_extract(goal, *_):
            if goal.id == "0" and not progressed.wait(timeout=2):
                waited_too_long.append(True)
            if goal.id == "2":
                progressed.set()
            return result_record(goal, "ok", graph, cache_hit=False)

        with tempfile.TemporaryDirectory() as tmp, patch("semantic_graph.pipeline.extract_goal", side_effect=mocked_extract), patch("semantic_graph.pipeline.write_jsonl", wraps=write_jsonl) as writer:
            out = Path(tmp) / "run"
            goals = [GoalRecord(str(i), TEXT) for i in range(6)]
            rows, _ = run_extraction(goals, FakeClient(), out, Path(tmp) / "cache", workers=2, progress=lambda _: None)
            self.assertEqual([r["id"] for r in rows], [g.id for g in goals])
            self.assertEqual([r["id"] for r in read_jsonl(out / "graphs.jsonl")], [g.id for g in goals])
            self.assertEqual(sum(call.args[0].name == "graphs.jsonl" for call in writer.call_args_list), 1)
        self.assertFalse(waited_too_long)

    def test_cache_key_ignores_transport_but_changes_with_request(self):
        goal, config = GoalRecord("x", TEXT), ModelConfig()
        self.assertEqual(cache_key(goal, config), cache_key(goal, replace(config, timeout=10, retries=0)))
        self.assertEqual(cache_key(goal, config), cache_key(goal, replace(config, temperature=0)))
        self.assertNotEqual(cache_key(goal, config), cache_key(goal, replace(config, temperature=1)))
        self.assertNotEqual(cache_key(goal, config), cache_key(goal, config, repeat_id=1))


class TestReplay(unittest.TestCase):
    def test_replay_does_not_call_provider_or_charge_archived_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "source", Path(tmp) / "replay"
            run_extraction([GoalRecord("x", TEXT)], FakeClient(), source, Path(tmp) / "cache", progress=lambda _: None)
            with patch.object(DeepSeekClient, "complete", side_effect=AssertionError("network forbidden")):
                rows, meta = replay_run(source, target)
            self.assertEqual(rows[0]["status"], "ok")
            self.assertEqual(meta["n_provider_calls"], 0)
            metrics = build_report(target)
            self.assertEqual(metrics["usage_this_run"], {})
            self.assertEqual(metrics["replayed"], 1)
            with self.assertRaises(ValueError):
                replay_run(target, Path(tmp) / "another")

    def test_replay_primes_current_cache_without_new_model_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target, cache = Path(tmp) / "source", Path(tmp) / "replay", Path(tmp) / "current-cache"
            goal = GoalRecord("x", TEXT)
            run_extraction([goal], FakeClient(), source, Path(tmp) / "original-cache", progress=lambda _: None)
            replay_run(source, target, cache)
            client = FakeClient(fatal=True)
            cached = extract_goal(goal, client, cache)
            self.assertTrue(cached["cache_hit"])
            self.assertEqual(cached["status"], "ok")
            self.assertEqual(client.calls, 0)


class TestMalformedTransport(unittest.TestCase):
    def test_non_string_content_is_reported_as_api_error(self):
        client = DeepSeekClient(ModelConfig(), api_key="test-secret")
        with patch.object(client, "_request", return_value={"choices": [{"message": {"content": []}}]}):
            with self.assertRaises(APIError):
                client.complete([])


if __name__ == "__main__":
    unittest.main()
