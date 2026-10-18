"""Exact typed-span matching; refuses pending/unverified human gold."""

from __future__ import annotations

from itertools import combinations

from .schema import fingerprint, graph_payload, validate_graph


def graph_sets(graph):
    if graph is None:
        return set(), set(), set()
    keys = {n["id"]: (n["type"], n["evidence"]["start"], n["evidence"]["end"]) for n in graph["nodes"]}
    nodes = set(keys.values())
    edges = {(e["type"], keys[e["source"]], keys[e["target"]]) for e in graph["edges"]}
    roots = {keys[r] for r in graph["root_ids"]}
    return nodes, edges, roots


def prf(tp, predicted, gold):
    # Undefined precision/recall are null; never substitute a perfect score for no evidence.
    return {"tp": tp, "predicted": predicted, "gold": gold,
            "precision": tp / predicted if predicted else None,
            "recall": tp / gold if gold else None,
            "f1": 2 * tp / (predicted + gold) if predicted + gold else None}


def evaluate(predictions, references, split="holdout"):
    pred = {r["id"]: r for r in predictions}
    if len(pred) != len(predictions):
        raise ValueError("duplicate prediction ids")
    if len({r["id"] for r in references}) != len(references):
        raise ValueError("duplicate reference ids")
    refs = [r for r in references if split == "all" or r.get("split") == split]
    if not refs:
        raise ValueError("no reference rows in selected split")
    for ref in refs:
        annotation = ref.get("annotation", {})
        annotators = annotation.get("annotators", [])
        if (ref.get("reference_type") != "human" or annotation.get("status") != "adjudicated"
                or not isinstance(annotators, list)
                or not all(isinstance(a, str) and a.strip() for a in annotators)
                or len(set(annotators)) < 2):
            raise ValueError("human reference requires adjudicated status and at least two identified annotators")
        if ref.get("graph") is None:
            raise ValueError("pending/null human reference graph")
    totals = {name: [0, 0, 0] for name in ("nodes", "edges", "roots", "constraints")}
    details = []
    for ref in refs:
        rid = ref["id"]
        gold_graph = validate_graph(graph_payload(ref["graph"]), ref["text"])
        prediction = pred.get(rid)
        if prediction and prediction["text"] != ref["text"]:
            raise ValueError("prediction/reference source text mismatch")
        graph = prediction.get("graph") if prediction else None
        if graph is not None:
            graph = validate_graph(graph_payload(graph), ref["text"])
        ps, gs = graph_sets(graph), graph_sets(gold_graph)
        pairs = list(zip(ps, gs)) + [({n for n in ps[0] if n[0] == "constraint"}, {n for n in gs[0] if n[0] == "constraint"})]
        item = {"id": rid, "status": prediction["status"] if prediction else "missing"}
        for name, (p, g) in zip(totals, pairs):
            values = (len(p & g), len(p), len(g))
            totals[name] = [a + b for a, b in zip(totals[name], values)]
            item[name] = prf(*values)
        details.append(item)
    return {"reference_type": "human_adjudicated", "split": split, "n": len(refs),
            "matching": "exact node type and original character span; typed endpoints for relations",
            "micro": {name: prf(*counts) for name, counts in totals.items()}, "per_sample": details,
            "note": "Missing/failed predictions count as empty graphs. This metric does not independently establish semantic truth."}


def repeat_consistency(runs):
    if len(runs) < 2:
        raise ValueError("at least two runs required")
    by_run = [{r["id"]: r for r in rows} for rows in runs]
    ids = set(by_run[0])
    if any(set(run) != ids for run in by_run):
        raise ValueError("repeat runs must contain exactly the same ids")
    details, agreements, comparisons = [], 0, 0
    valid_agreements, valid_comparisons = 0, 0
    for rid in sorted(ids):
        rows = [run[rid] for run in by_run]
        if len({r["text"] for r in rows}) != 1:
            raise ValueError("repeat source text mismatch")
        if len({r.get("repeat_id", 0) for r in rows}) != len(rows):
            raise ValueError("repeat runs need distinct repeat_id values; cached replays are not independent calls")
        same = 0
        for a, b in combinations(rows, 2):
            equal = fingerprint({"status": a["status"], "sets": [sorted(s) for s in graph_sets(a["graph"])]}) == fingerprint({"status": b["status"], "sets": [sorted(s) for s in graph_sets(b["graph"])]})
            agreements += int(equal)
            same += int(equal)
            comparisons += 1
            if a["status"] in {"ok", "partial"} and b["status"] in {"ok", "partial"}:
                valid_comparisons += 1
                valid_agreements += int(equal)
        details.append({"id": rid, "pairwise_equal": same, "statuses": [r["status"] for r in rows]})
    return {"n": len(ids), "n_runs": len(runs), "pairwise_comparisons": comparisons,
            "exact_pairwise_agreement": agreements / comparisons if comparisons else None,
            "valid_pairwise_comparisons": valid_comparisons,
            "valid_graph_agreement": valid_agreements / valid_comparisons if valid_comparisons else None,
            "per_sample": details}

