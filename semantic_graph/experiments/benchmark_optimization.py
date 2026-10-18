"""Offline I/O and scheduler microbenchmarks; never instantiate an API client.

Run from the repository root:
python -B -m semantic_graph.experiments.benchmark_optimization
"""

from __future__ import annotations

import json
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from semantic_graph.client import ModelConfig
from semantic_graph.io import write_json, write_jsonl
from semantic_graph.pipeline import load_run, run_extraction
from semantic_graph.schema import GoalRecord, result_record

ROOT = Path(__file__).resolve().parents[1]


def benchmark():
    baseline, _ = load_run(ROOT / "runs" / "standard200")
    sizes = [len((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8")) for r in baseline]
    old_bytes = sum(size * (len(sizes) - i) for i, size in enumerate(sizes))
    new_bytes = 2 * sum(sizes)  # One journal append and one final ordered snapshot.
    old_times, new_times = [], []
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "graphs.jsonl"
        for _ in range(3):
            started = time.perf_counter()
            for i in range(1, len(baseline) + 1):
                write_jsonl(path, baseline[:i])
            old_times.append(time.perf_counter() - started)
            started = time.perf_counter()
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                for row in baseline:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
            write_jsonl(path, baseline)
            new_times.append(time.perf_counter() - started)
    io_result = {"records": len(baseline), "old_graph_bytes_written": old_bytes,
                 "new_graph_bytes_written": new_bytes, "byte_reduction": 1 - new_bytes / old_bytes,
                 "old_wall_s": old_times, "new_wall_s": new_times,
                 "median_wall_speedup": statistics.median(old_times) / statistics.median(new_times),
                 "scope": "Same archived rows; graph JSONL serialization and writes only; three trials."}

    delays = [0.09, 0.003, 0.003] * 8
    goals = [GoalRecord(str(i), "Read a book.") for i in range(len(delays))]

    class OfflineClient:
        config = ModelConfig()

    def task(goal, *_):
        time.sleep(delays[int(goal.id)])
        return result_record(goal, "refused", issues=["offline benchmark fixture"], cache_hit=False)

    old_times, new_times = [], []
    with tempfile.TemporaryDirectory() as tmp, patch("semantic_graph.pipeline.extract_goal", side_effect=task):
        for trial in range(3):
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=3) as pool:
                for i in range(0, len(goals), 3):
                    pending = [pool.submit(task, g) for g in goals[i:i + 3]]
                    for future in pending:
                        future.result()
            old_times.append(time.perf_counter() - started)
            started = time.perf_counter()
            run_extraction(goals, OfflineClient(), Path(tmp) / f"run{trial}", Path(tmp) / "cache",
                           workers=3, progress=lambda _: None)
            new_times.append(time.perf_counter() - started)
    scheduler = {"workers": 3, "task_delays_s": delays, "old_wall_s": old_times, "new_wall_s": new_times,
                 "median_wall_speedup": statistics.median(old_times) / statistics.median(new_times),
                 "scope": "Synthetic uneven delays. Old scheduler only vs new complete local pipeline; three trials. Not live API latency."}
    optimized, optimized_meta = load_run(ROOT / "runs" / "standard200-optimized")
    transitions = {}
    for before, after in zip(baseline, optimized):
        key = before["status"] + "->" + after["status"]
        transitions[key] = transitions.get(key, 0) + 1
    original_attempts = sum(len(r.get("attempts", [])) for r in baseline)
    consumed = sum(len(r.get("attempts", [])) for r in optimized)
    return {"provider_calls_this_optimization": 0, "io": io_result, "scheduler": scheduler,
            "replay": {"mode": optimized_meta["mode"], "records": len(optimized),
                       "baseline_attempts_archived": original_attempts, "replayed_attempts_consumed": consumed,
                       "archived_second_attempts_not_needed_by_parser": original_attempts - consumed,
                       "status_transitions": transitions,
                       "note": "Replay reuses available responses; it does not predict unseen model replies or establish semantic accuracy."}}


if __name__ == "__main__":
    output = ROOT / "experiments" / "optimization-2026-10-15.json"
    existing = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    result = {**existing, **benchmark()}
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False))
