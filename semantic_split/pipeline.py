"""Shared batch summary and offline postprocessing of semantic_graph runs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from semantic_graph.io import read_jsonl, write_json, write_jsonl
from semantic_graph.pipeline import load_run
from semantic_graph.schema import fingerprint

from . import SPLIT_VERSION, split_record


def summarize(inputs, rows):
    """Describe every input and result, including explicit upstream failures."""
    if len(inputs) != len(rows):
        raise ValueError("every input requires one split result")
    return {
        "split_version": SPLIT_VERSION, "method": "action-centered source graph decomposition",
        "input_sha256": fingerprint(inputs), "n_input": len(inputs), "n_output": len(rows),
        "n_fragments": sum(len(row["fragments"]) for row in rows),
        "split_statuses": dict(Counter(row["split_status"] for row in rows)),
        "source_statuses": dict(Counter(
            row.get("status") if isinstance(row.get("status"), str)
            else "missing" if row.get("status") is None else "malformed" for row in rows)),
        "note": "Coverage measures preservation of supplied graph elements, not semantic correctness.",
    }


def split_graph_run(run_dir):
    """Call the splitter on a saved benign graph run with zero model requests.

    Write only derived fragments.jsonl and split_summary.json beside graphs.jsonl.
    Rerunning refreshes these derived files without changing extraction artifacts,
    cache identities or metadata. Every original input receives one output row,
    including unprocessed inputs from an interrupted extraction.
    """
    root = Path(run_dir)
    sources, meta = load_run(root)
    inputs = read_jsonl(root / "inputs.jsonl")
    by_id = {row["id"]: row for row in sources}
    rows = []
    for original in inputs:
        source = by_id.get(original["id"])
        if source is None:
            result = split_record(original)
            result.update(split_status="not_processed",
                          split_issues=["Source extraction has not processed this input."])
        else:
            result = split_record(source)
        rows.append(result)
    summary = {
        **summarize(inputs, rows),
        "source_run": str(root.resolve()), "source_complete": meta.get("complete", False),
        "source_graphs_sha256": fingerprint(sources), "source_run_mode": meta.get("mode"),
        "n_provider_calls": 0,
    }
    write_jsonl(root / "fragments.jsonl", rows)
    write_json(root / "split_summary.json", summary)
    return rows, summary
