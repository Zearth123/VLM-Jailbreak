"""Strict goal loading, official dataset preparation, and blind annotation packs."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path

from .schema import GoalRecord

HARM_BENCH_REPO = "centerforaisafety/HarmBench"
HARM_BENCH_PATH = "data/behavior_datasets/harmbench_behaviors_text_all.csv"
QUOTAS = {"chemical_biological": 4, "cybercrime_intrusion": 4, "illegal": 4,
          "harmful": 4, "misinformation_disinformation": 2, "harassment_bullying": 2}


def read_jsonl(path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{lineno}: expected object")
            rows.append(value)
    return rows


def recover_journal(path) -> list[dict]:
    """Read an append-only run journal, removing only an incomplete last write.

    A complete malformed record or malformed middle line is an error. A valid
    final record lacking its newline is retained and terminated before appending.
    """
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r+b") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            try:
                value = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if not line.endswith(b"\n") and not stream.read(1):
                    stream.seek(offset)
                    stream.truncate()
                    break
                raise ValueError(f"{path}: malformed complete journal record at byte {offset}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}: expected journal object at byte {offset}")
            rows.append(value)
            if not line.endswith(b"\n"):
                stream.write(b"\n")
    return rows


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load_goals(path, subset=None) -> list[GoalRecord]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            raw = list(csv.DictReader(stream))
    elif path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(raw, list):
            raise ValueError("JSON input must be an array")
    else:
        raw = read_jsonl(path)
    return _goal_records(raw, subset)


def _goal_records(raw, subset=None):
    if any(not isinstance(row, dict) for row in raw):
        raise ValueError("goal rows must be objects")
    if subset:
        raw = [r for r in raw if r.get("FunctionalCategory") == subset]
    seen, records = set(), []
    for index, row in enumerate(raw, 1):
        rid = row.get("BehaviorID", row.get("id"))
        text = row.get("Behavior", row.get("text"))
        category = row.get("SemanticCategory", row.get("category", "unknown"))
        if not isinstance(rid, str) or not rid.strip() or rid in seen:
            raise ValueError(f"row {index}: missing/non-string/duplicate id")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"row {index}: missing/empty/non-string text")
        if not isinstance(category, str) or not category:
            raise ValueError(f"row {index}: category must be a nonempty string")
        seen.add(rid)
        records.append(GoalRecord(rid, text, category))
    if not records:
        raise ValueError("no goals matched the input/filter")
    return records


def select_samples(goals, seed=42):
    rng = random.Random(seed)
    samples, splits = [], {}
    for category, quota in QUOTAS.items():
        choices = sorted((g for g in goals if g.category == category), key=lambda g: g.id)
        if len(choices) < quota:
            raise ValueError(f"not enough records in {category}")
        selected = rng.sample(choices, quota)
        for i, goal in enumerate(selected):
            samples.append(goal)
            splits[goal.id] = "dev" if i < quota // 2 else "holdout"
    return samples, splits


def prepare_dataset(out, source=None, seed=42):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("prepare output is not empty; use a new directory to preserve the frozen sample")
    out.mkdir(parents=True, exist_ok=True)
    provenance = {"repository": HARM_BENCH_REPO, "path": HARM_BENCH_PATH, "seed": seed}
    if source:
        content = Path(source).read_bytes()
        provenance["local_source"] = str(Path(source).resolve())
    else:
        headers = {"User-Agent": "VLM-Jailbreak-semantic-graph"}
        commit_url = f"https://api.github.com/repos/{HARM_BENCH_REPO}/commits/main"
        with urllib.request.urlopen(urllib.request.Request(commit_url, headers=headers), timeout=30) as response:
            commit = json.load(response)["sha"]
        url = f"https://api.github.com/repos/{HARM_BENCH_REPO}/contents/{HARM_BENCH_PATH}?ref={commit}"
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
            blob = json.load(response)
        content = base64.b64decode(blob["content"])
        provenance.update(commit=commit, blob_sha=blob["sha"], url=url)
    # Parse the actual bytes once and store the exact downloaded source.
    csv_rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline="")))
    provenance.update(source_sha256=hashlib.sha256(content).hexdigest(), all_records=len(csv_rows))
    raw_path = out / "harmbench_behaviors_text_all.csv"
    raw_path.write_bytes(content)
    goals = _goal_records(csv_rows, subset="standard")
    if len(goals) != 200:
        raise ValueError(f"expected standard=200, got {len(goals)}; inspect upstream dataset changes")
    samples, splits = select_samples(goals, seed)
    provenance["standard_categories"] = dict(Counter(g.category for g in goals))
    provenance["sample_ids"] = {g.id: splits[g.id] for g in samples}
    write_json(out / "manifest.json", provenance)
    write_jsonl(out / "standard200.jsonl", [g.to_dict() for g in goals])
    write_jsonl(out / "sample20.jsonl", [g.to_dict() for g in samples])
    for split in ("dev", "holdout"):
        write_jsonl(out / f"{split}10.jsonl", [g.to_dict() for g in samples if splits[g.id] == split])
    for name in ("annotator_a", "annotator_b", "adjudicated"):
        rows = [{**g.to_dict(), "split": splits[g.id], "reference_type": "human",
                 "annotation": {"status": "pending", "annotators": [], "notes": ""}, "graph": None}
                for g in samples]
        write_jsonl(out / "annotations" / f"{name}.jsonl", rows)
    return provenance

