"""Graph-to-fragment workflow with benign inputs and simulated model responses."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from semantic_graph.cli import main
from semantic_graph.io import read_jsonl, write_jsonl
from semantic_graph.pipeline import run_extraction
from semantic_graph.schema import GoalRecord
from semantic_graph.tests.test_workflow import FakeClient, TEXT, payload
from semantic_split.pipeline import split_graph_run


class TestSplitIntegration(unittest.TestCase):
    def run_cli(self, args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main(args)

    def extract(self, directory, client, split=True, count=1):
        root = Path(directory)
        inputs, out = root / "input.jsonl", root / "run"
        write_jsonl(inputs, [{"id": f"text-{i}", "text": TEXT, "category": "benign"} for i in range(count)])
        args = ["extract", "--input", str(inputs), "--out", str(out),
                "--cache", str(root / "cache"), "--workers", "1"]
        if split:
            args.append("--split")
        with patch("semantic_graph.cli.DeepSeekClient", return_value=client):
            code = self.run_cli(args)
        return code, out

    def test_extract_calls_splitter_and_saves_all_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient()
            code, out = self.extract(tmp, client, count=2)
            self.assertEqual(code, 0)
            self.assertEqual(client.calls, 2)
            graphs = read_jsonl(out / "graphs.jsonl")
            fragments = read_jsonl(out / "fragments.jsonl")
            self.assertEqual([r["id"] for r in graphs], [r["id"] for r in fragments])
            for source, result in zip(graphs, fragments):
                self.assertEqual(result["graph"], source["graph"])
                self.assertEqual(result["text"], source["text"])
                self.assertEqual(result["split_status"], "ok")
                self.assertEqual(len(result["fragments"]), 1)
                self.assertEqual(result["coverage"]["nodes_preserved"], 3)
                self.assertNotIn("fragments", source)
            summary = json.loads((out / "split_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["source_complete"])
            self.assertEqual(summary["n_input"], 2)
            self.assertEqual(summary["n_fragments"], 2)
            self.assertEqual(summary["n_provider_calls"], 0)

    def test_existing_graph_only_workflow_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out = self.extract(tmp, FakeClient(), split=False)
            self.assertEqual(code, 0)
            self.assertTrue((out / "graphs.jsonl").exists())
            self.assertFalse((out / "fragments.jsonl").exists())

    def test_resuming_graph_run_with_split_does_not_request_model_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient()
            code, out = self.extract(tmp, client, split=False)
            self.assertEqual(code, 0)
            graphs = (out / "graphs.jsonl").read_bytes()
            self.assertEqual(client.calls, 1)
            code, _ = self.extract(tmp, client, split=True)
            self.assertEqual(code, 0)
            self.assertEqual(client.calls, 1)
            self.assertEqual((out / "graphs.jsonl").read_bytes(), graphs)
            self.assertEqual(read_jsonl(out / "fragments.jsonl")[0]["split_status"], "ok")

    def test_offline_split_is_repeatable_and_does_not_change_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out = self.extract(tmp, FakeClient(), split=False)
            self.assertEqual(code, 0)
            extraction = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
            with patch("urllib.request.urlopen", side_effect=AssertionError("unexpected network request")):
                self.assertEqual(self.run_cli(["split", "--run", str(out)]), 0)
                derived = {name: (out / name).read_bytes()
                           for name in ("fragments.jsonl", "split_summary.json")}
                self.assertEqual(self.run_cli(["split", "--run", str(out)]), 0)
            for name, content in extraction.items():
                self.assertEqual((out / name).read_bytes(), content)
            for name, content in derived.items():
                self.assertEqual((out / name).read_bytes(), content)

    def test_partial_and_refused_source_states_are_retained(self):
        partial = payload()
        partial.update(status="partial", issues=["Review scope."])
        refused = {"status": "refused", "issues": ["provider_refusal"], "graph": None}
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient(responses=[partial, refused])
            code, out = self.extract(tmp, client, count=2)
            self.assertEqual(code, 2)
            self.assertEqual(client.calls, 2)
            rows = read_jsonl(out / "fragments.jsonl")
            self.assertEqual([r["status"] for r in rows], ["partial", "refused"])
            self.assertEqual([r["split_status"] for r in rows], ["partial", "skipped"])
            self.assertEqual(rows[0]["issues"], ["Review scope."])
            self.assertEqual(rows[1]["fragments"], [])

    def test_fatal_error_keeps_unprocessed_inputs_in_split_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient(fatal=True)
            code, out = self.extract(tmp, client, count=3)
            self.assertEqual(code, 2)
            self.assertEqual(client.calls, 1)
            rows = read_jsonl(out / "fragments.jsonl")
            self.assertEqual([r["id"] for r in rows], ["text-0", "text-1", "text-2"])
            self.assertEqual([r["split_status"] for r in rows], ["skipped", "not_processed", "not_processed"])
            summary = json.loads((out / "split_summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["source_complete"])
            self.assertEqual(summary["n_input"], summary["n_output"])

    def test_offline_command_validates_source_fingerprint_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out = self.extract(tmp, FakeClient(), split=False)
            self.assertEqual(code, 0)
            inputs = read_jsonl(out / "inputs.jsonl")
            inputs[0]["text"] = "Different source."
            write_jsonl(out / "inputs.jsonl", inputs)
            self.assertEqual(self.run_cli(["split", "--run", str(out)]), 2)
            self.assertFalse((out / "fragments.jsonl").exists())

    def test_library_entry_point_uses_same_output_as_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            run_extraction([GoalRecord("benign", TEXT, "benign")], FakeClient(),
                           out, Path(tmp) / "cache", progress=lambda _: None)
            results, summary = split_graph_run(out)
            self.assertEqual(results, read_jsonl(out / "fragments.jsonl"))
            self.assertEqual(summary["split_statuses"], {"ok": 1})
            self.assertEqual(self.run_cli(["split", "--run", str(out)]), 0)
            self.assertEqual(results, read_jsonl(out / "fragments.jsonl"))


if __name__ == "__main__":
    unittest.main()
