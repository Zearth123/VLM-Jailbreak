"""Offline and mocked-extraction integration; no credentials or provider calls."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from semantic_graph.client import APIError, ModelConfig
from semantic_graph.io import read_jsonl, write_jsonl
from semantic_split.cli import main
from semantic_split.examples import benign_examples


class FakeClient:
    config = ModelConfig()

    def __init__(self, fatal=False, refused=False):
        self.calls, self.fatal, self.refused = 0, fatal, refused

    def complete(self, messages):
        self.calls += 1
        if self.fatal:
            raise APIError("DeepSeek HTTP 401", fatal=True)
        row = benign_examples()[0]
        payload = {k: row[k] for k in ("status", "issues", "graph")}
        if self.refused:
            payload = {"status": "refused", "issues": ["provider_refusal"], "graph": None}
        return {"content": json.dumps(payload), "finish_reason": "stop", "refusal": False,
                "model": "mock-only", "usage": {"total_tokens": 1}}


class TestCLI(unittest.TestCase):
    def run_cli(self, argv):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main(argv)

    def test_demo_and_graph_batch_are_entirely_offline(self):
        with tempfile.TemporaryDirectory() as tmp, patch("semantic_split.cli.DeepSeekClient", side_effect=AssertionError("network")):
            demo, batch = Path(tmp) / "demo", Path(tmp) / "batch"
            self.assertEqual(self.run_cli(["demo", "--out", str(demo)]), 0)
            self.assertEqual(self.run_cli(["split", "--input", str(demo / "inputs.jsonl"), "--out", str(batch)]), 0)
            self.assertEqual((demo / "fragments.jsonl").read_bytes(), (batch / "fragments.jsonl").read_bytes())
            summary = json.loads((batch / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["n_input"], 4)
            self.assertEqual(summary["n_fragments"], 7)

    def test_malformed_graph_does_not_drop_other_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp) / "input.jsonl", Path(tmp) / "out"
            rows = benign_examples()
            rows[0]["graph"]["nodes"][0]["evidence"]["quote"] = "invented"
            write_jsonl(source, rows)
            self.assertEqual(self.run_cli(["split", "--input", str(source), "--out", str(out)]), 2)
            results = read_jsonl(out / "fragments.jsonl")
            self.assertEqual([r["id"] for r in results], [r["id"] for r in rows])
            self.assertEqual(results[0]["split_status"], "invalid")
            self.assertTrue(all(r["split_status"] == "ok" for r in results[1:]))

    def test_plain_text_requires_explicit_extraction(self):
        with tempfile.TemporaryDirectory() as tmp, patch("semantic_split.cli._client", side_effect=AssertionError("network")):
            out = Path(tmp) / "out"
            self.assertEqual(self.run_cli(["split", "--text", "Draw a cup.", "--out", str(out)]), 2)
            self.assertFalse(out.exists())

    def test_explicit_extraction_and_cache_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient()
            for repeat in range(2):
                out = Path(tmp) / f"out{repeat}"
                with patch("semantic_split.cli._client", return_value=client):
                    self.assertEqual(self.run_cli([
                        "split", "--text", benign_examples()[0]["text"], "--extract",
                        "--cache", str(Path(tmp) / "cache"), "--out", str(out)]), 0)
                result = read_jsonl(out / "fragments.jsonl")[0]
                self.assertEqual(len(result["fragments"]), 2)
                self.assertEqual(result["cache_hit"], bool(repeat))
            self.assertEqual(client.calls, 1)

    def test_fatal_api_error_stops_calls_and_records_unprocessed_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp) / "input.jsonl", Path(tmp) / "out"
            write_jsonl(source, [{"id": f"text-{i}", "text": "Draw a cup."} for i in range(3)])
            client = FakeClient(fatal=True)
            with patch("semantic_split.cli._client", return_value=client):
                self.assertEqual(self.run_cli(["split", "--input", str(source), "--extract",
                                               "--cache", str(Path(tmp) / "cache"), "--out", str(out)]), 2)
            rows = read_jsonl(out / "fragments.jsonl")
            self.assertEqual(client.calls, 1)
            self.assertEqual([r["split_status"] for r in rows], ["skipped", "not_processed", "not_processed"])

    def test_refusal_remains_a_single_failed_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            client, out = FakeClient(refused=True), Path(tmp) / "out"
            with patch("semantic_split.cli._client", return_value=client):
                self.assertEqual(self.run_cli(["split", "--text", "Draw a cup.", "--extract",
                                               "--cache", str(Path(tmp) / "cache"), "--out", str(out)]), 2)
            self.assertEqual(client.calls, 1)
            self.assertEqual(read_jsonl(out / "fragments.jsonl")[0]["status"], "refused")

    def test_existing_output_is_checked_before_any_model_call(self):
        with tempfile.TemporaryDirectory() as tmp, patch("semantic_split.cli._client", side_effect=AssertionError("network")):
            out = Path(tmp)
            (out / "keep.txt").write_text("original", encoding="utf-8")
            self.assertEqual(self.run_cli(["split", "--text", "Draw a cup.", "--extract", "--out", str(out)]), 2)
            self.assertEqual((out / "keep.txt").read_text(encoding="utf-8"), "original")

    def test_duplicate_ids_fail_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp) / "input.jsonl", Path(tmp) / "out"
            write_jsonl(source, [benign_examples()[0], benign_examples()[0]])
            self.assertEqual(self.run_cli(["split", "--input", str(source), "--out", str(out)]), 2)
            self.assertFalse(out.exists())

    def test_json_and_multiline_utf8_text_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.json"
            source.write_text(json.dumps(benign_examples(), ensure_ascii=False), encoding="utf-8-sig")
            self.assertEqual(self.run_cli(["split", "--input", str(source), "--out", str(Path(tmp) / "json-out")]), 0)
            source = Path(tmp) / "paragraph.txt"
            text = benign_examples()[0]["text"] + "\n"
            source.write_text(text, encoding="utf-8-sig")
            with patch("semantic_split.cli._client", return_value=FakeClient()):
                out = Path(tmp) / "text-out"
                self.assertEqual(self.run_cli(["split", "--input", str(source), "--extract",
                                               "--cache", str(Path(tmp) / "cache"), "--out", str(out)]), 0)
            self.assertEqual(read_jsonl(out / "fragments.jsonl")[0]["text"], text)

    def test_malformed_status_is_reported_without_breaking_the_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp) / "input.jsonl", Path(tmp) / "out"
            rows = benign_examples()
            rows[0]["status"] = {"bad": "status"}
            write_jsonl(source, rows)
            self.assertEqual(self.run_cli(["split", "--input", str(source), "--out", str(out)]), 2)
            self.assertEqual(len(read_jsonl(out / "fragments.jsonl")), len(rows))
            self.assertEqual(read_jsonl(out / "fragments.jsonl")[0]["split_status"], "invalid")

    def test_stdin_credential_does_not_enter_any_saved_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, cache = Path(tmp) / "out", Path(tmp) / "cache"
            with patch("semantic_split.cli.sys.stdin", io.StringIO("example-secret\n")), patch(
                    "semantic_split.cli.DeepSeekClient", return_value=FakeClient()) as constructor:
                self.assertEqual(self.run_cli(["split", "--text", benign_examples()[0]["text"],
                                               "--extract", "--key-stdin", "--cache", str(cache),
                                               "--out", str(out)]), 0)
                self.assertEqual(constructor.call_args.kwargs["api_key"], "example-secret")
            for path in Path(tmp).rglob("*"):
                if path.is_file():
                    self.assertNotIn("example-secret", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
