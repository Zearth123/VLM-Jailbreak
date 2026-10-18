"""Extraction, bounded retries, content-addressed cache, and incremental runs."""

from __future__ import annotations

import json
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

from . import PROMPT_VERSION, SCHEMA_VERSION
from .client import APIError, ModelConfig
from .io import read_jsonl, recover_journal, write_json, write_jsonl
from .prompt import SYSTEM_PROMPT
from .schema import STATUSES, GoalRecord, GraphError, fingerprint, graph_payload, parse_response, result_record, validate_graph

EXTRACTION_VERSION = "extractor-2"
TERMINAL_STATUSES = {"ok", "partial", "refused", "invalid"}


def _source_snapshot():
    root = Path(__file__).resolve().parent
    return {name: (root / name).read_text(encoding="utf-8")
            for name in ("__init__.py", "client.py", "io.py", "pipeline.py", "prompt.py", "schema.py")}


def cache_key(goal, config, repeat_id=0):
    return fingerprint({"goal": goal.to_dict(), "model": config.request_identity(), "prompt": SYSTEM_PROMPT,
                        "prompt_version": PROMPT_VERSION, "schema_version": SCHEMA_VERSION,
                        "extraction_version": EXTRACTION_VERSION, "repeat_id": repeat_id})


def _parse_completion(completion, text):
    if completion["refusal"] or completion["finish_reason"] == "content_filter":
        return {"status": "refused", "graph": None, "issues": ["provider_refusal"], "repairs": []}
    if completion["finish_reason"] != "stop":
        raise GraphError("completion did not finish normally (possibly truncated)")
    if not completion["content"].strip():
        raise GraphError("empty completion")
    return parse_response(json.loads(completion["content"]), text)


def extract_goal(goal, client, cache_dir, repeat_id=0):
    key = cache_key(goal, client.config, repeat_id)
    cache_path = Path(cache_dir) / f"{key}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("cache_key") != key or any(cached.get(k) != v for k, v in goal.to_dict().items()):
            raise ValueError("cache content/fingerprint mismatch")
        if cached.get("status") in TERMINAL_STATUSES:
            _validate_row(cached, goal.to_dict())
            return {**cached, "cache_hit": True}

    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({"source_text": goal.text}, ensure_ascii=False)}]
    attempts, issues, graph, status, fatal, repairs = [], [], None, "error", False, []
    started = time.monotonic()
    for attempt in range(2):  # One schema repair; no refusal workaround.
        try:
            completion = client.complete(messages)
        except APIError as exc:
            issues.append(str(exc))
            fatal = exc.fatal
            status = "error"
            break
        attempts.append(completion)
        try:
            parsed = _parse_completion(completion, goal.text)
            status, graph, issues = parsed["status"], parsed["graph"], parsed["issues"]
            repairs = parsed["repairs"]
            break
        except (json.JSONDecodeError, GraphError) as exc:
            status, issues = "invalid", [f"validation: {exc}"]
            if attempt == 0:
                messages.extend([
                    {"role": "assistant", "content": completion["content"]},
                    {"role": "user", "content": "The JSON/schema/source-span validation failed: " + str(exc)
                     + ". Correct the same extraction once. Use only original source evidence; do not answer or expand the source request."},
                ])
    result = result_record(goal, status, graph, issues, cache_key=key, cache_hit=False,
                           repeat_id=repeat_id, attempts=attempts, fatal_error=fatal, repairs=repairs,
                           elapsed_s=round(time.monotonic() - started, 3))
    write_json(cache_path, result)
    return result


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_extraction(goals, client, out, cache_dir, workers=2, repeat_id=0, progress=print):
    if workers < 1 or workers > 16 or repeat_id < 0:
        raise ValueError("workers must be 1..16 and repeat_id must be nonnegative")
    out = Path(out)
    if not goals or len({g.id for g in goals}) != len(goals):
        raise ValueError("goals must be nonempty and have unique ids")
    source_snapshot = _source_snapshot()
    implementation = fingerprint(source_snapshot)
    identity = {"input_fingerprint": fingerprint([g.to_dict() for g in goals]),
                "request_config": client.config.request_identity(), "prompt_sha256": fingerprint(SYSTEM_PROMPT),
                "implementation_sha256": implementation,
                "extraction_version": EXTRACTION_VERSION,
                "schema_version": SCHEMA_VERSION, "prompt_version": PROMPT_VERSION,
                "repeat_id": repeat_id}
    meta_path = out / "run.meta.json"
    previous = {}
    if out.exists() and any(out.iterdir()):
        if not meta_path.exists():
            raise ValueError("nonempty run directory has no run metadata; use a new output directory")
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        if any(previous.get(k) != v for k, v in identity.items()):
            raise ValueError("run fingerprint differs; use a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    expected = {g.id: g.to_dict() for g in goals}
    by_id = {}
    journal_path = out / "graphs.jsonl"
    for row in recover_journal(journal_path):
        _validate_row(row, expected.get(row.get("id")))
        if row["id"] in by_id and by_id[row["id"]]["status"] != "error":
            raise ValueError("duplicate completed id in journal")
        by_id[row["id"]] = row
    remaining = iter(g for g in goals if g.id not in by_id or by_id[g.id]["status"] == "error")
    meta = {**identity, "config": client.config.public(), "mode": "live",
            "started_utc": previous.get("started_utc", datetime.now(timezone.utc).isoformat()),
            "code_commit": _git_commit(), "n_input": len(goals), "n_processed": len(by_id), "complete": False,
            "note": "Machine extractions; human fidelity evaluation is separate."}
    write_json(meta_path, meta)
    write_json(out / "source.snapshot.json", source_snapshot)
    write_json(out / "prompt.snapshot.json", {"system": SYSTEM_PROMPT})
    write_jsonl(out / "inputs.jsonl", [g.to_dict() for g in goals])
    # Flush completed rows immediately; rewrite the ordered snapshot only once.
    # A crashed process can recover from this journal without relying on its cache.
    with journal_path.open("a", encoding="utf-8", newline="\n") as journal, ThreadPoolExecutor(max_workers=workers) as pool:
        pending, stopped = set(), False

        def fill_workers():
            while not stopped and len(pending) < workers:
                goal = next(remaining, None)
                if goal is None:
                    break
                pending.add(pool.submit(extract_goal, goal, client, cache_dir, repeat_id))

        fill_workers()
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            completed = [future.result() for future in done]
            stopped = stopped or any(row.get("fatal_error") for row in completed)
            for row in completed:
                by_id[row["id"]] = row
                journal.write(json.dumps(row, ensure_ascii=False) + "\n")
                journal.flush()
                progress(f"[{len(by_id)}/{len(goals)}] {row['id']}: {row['status']}" + (" (cache)" if row["cache_hit"] else ""))
            fill_workers()
    rows = [by_id[g.id] for g in goals if g.id in by_id]
    write_jsonl(journal_path, rows)
    meta.update(n_processed=len(rows), complete=len(rows) == len(goals) and not any(r.get("fatal_error") for r in rows),
                finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(meta_path, meta)
    return rows, meta


def replay_run(source, out, cache_dir=None):
    """Reinterpret archived responses in original order, with zero provider calls.

    Replay cannot test how the model would answer a changed repair prompt. It
    reports parser behavior on the responses that are actually available only.
    """
    rows, original_meta = load_run(source)
    if not original_meta.get("complete"):
        raise ValueError("replay requires a completed original run")
    if original_meta.get("mode") == "replay":
        raise ValueError("replay requires an original extraction run")
    if original_meta["schema_version"] != SCHEMA_VERSION or original_meta["prompt_sha256"] != fingerprint(SYSTEM_PROMPT):
        raise ValueError("replay requires matching schema and extraction prompt")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("replay output must be empty; preserve the original run")
    results = []
    config = ModelConfig(**{k: v for k, v in original_meta["config"].items() if k in ModelConfig.__dataclass_fields__})
    for row in rows:
        attempts = row.get("attempts", [])
        status, graph, issues, repairs, consumed = "invalid", None, ["No archived completion available."], [], []
        for completion in attempts[:2]:
            consumed.append(completion)
            try:
                parsed = _parse_completion(completion, row["text"])
                status, graph, issues, repairs = (parsed[k] for k in ("status", "graph", "issues", "repairs"))
                break
            except (json.JSONDecodeError, GraphError) as exc:
                issues = [f"validation: {exc}"]
        if not attempts:
            status, issues = "error", row.get("issues", issues)
        goal = GoalRecord(row["id"], row["text"], row["category"])
        results.append({**row, "status": status, "graph": graph, "issues": issues, "repairs": repairs,
                        "attempts": consumed, "cache_hit": False, "replayed": True,
                        "cache_key": cache_key(goal, config, original_meta["repeat_id"]),
                        "archived_attempts_available": len(attempts), "source_result_sha256": fingerprint(row)})
    source_snapshot = _source_snapshot()
    meta = {**original_meta, "mode": "replay", "source_run": str(Path(source).resolve()),
            "source_meta_sha256": fingerprint(original_meta), "extraction_version": EXTRACTION_VERSION,
            "implementation_sha256": fingerprint(source_snapshot), "n_provider_calls": 0,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "note": "Offline parser replay of archived completions, not an independent model run or fidelity evaluation."}
    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(out / "source.snapshot.json", source_snapshot)
    write_json(out / "run.meta.json", meta)
    write_jsonl(out / "inputs.jsonl", read_jsonl(Path(source) / "inputs.jsonl"))
    write_jsonl(out / "graphs.jsonl", results)
    if cache_dir is not None:
        for row in results:
            if row["status"] in TERMINAL_STATUSES:
                write_json(Path(cache_dir) / f"{row['cache_key']}.json", row)
    return results, meta


def _validate_row(row, expected):
    if expected is None or any(row.get(k) != expected[k] for k in ("id", "text", "category")):
        raise ValueError("unknown id or text/category mismatch in run")
    if row.get("status") not in STATUSES:
        raise ValueError("invalid result status")
    if (row["status"] in {"ok", "partial"}) != (row.get("graph") is not None):
        raise ValueError("result status/graph presence mismatch")
    if row.get("graph") is not None:
        row["graph"] = validate_graph(graph_payload(row["graph"]), row["text"])


def load_run(run_dir):
    root = Path(run_dir)
    meta = json.loads((root / "run.meta.json").read_text(encoding="utf-8"))
    rows = read_jsonl(root / "graphs.jsonl")
    snapshot = read_jsonl(root / "inputs.jsonl")
    if fingerprint(snapshot) != meta["input_fingerprint"]:
        raise ValueError("input snapshot fingerprint mismatch")
    expected = {g["id"]: g for g in snapshot}
    if len(expected) != meta["n_input"]:
        raise ValueError("input snapshot size mismatch")
    seen = set()
    for row in rows:
        rid = row["id"]
        if rid in seen:
            raise ValueError("duplicate id in run")
        _validate_row(row, expected.get(rid))
        seen.add(rid)
    return rows, meta

