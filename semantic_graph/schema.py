"""Graph contract, exact source alignment, and deterministic serialization."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from . import SCHEMA_VERSION

NODE_TYPES = {"entity", "action", "quantity", "condition", "constraint", "artifact"}
# Direction is always explicit; see SPEC.md for the source/target type contract.
EDGE_TYPES = {
    "agent", "patient", "instrument", "condition_of", "constraint_of",
    "quantity_of", "located_at", "origin", "purpose", "precedes", "produces",
}
STATUSES = {"ok", "partial", "refused", "invalid", "error"}


class GraphError(ValueError):
    pass


@dataclass(frozen=True)
class GoalRecord:
    id: str
    text: str
    category: str = "unknown"

    def to_dict(self) -> dict:
        return asdict(self)


def fingerprint(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _fields(obj: Any, required: set[str], optional: set[str], where: str) -> dict:
    if not isinstance(obj, dict):
        raise GraphError(f"{where}: expected object")
    missing, extra = required - obj.keys(), obj.keys() - required - optional
    if missing or extra:
        raise GraphError(f"{where}: missing={sorted(missing)}, unknown={sorted(extra)}")
    return obj


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GraphError(f"{where}: expected nonempty string")
    return value


def align_evidence(evidence: Any, text: str) -> dict:
    """Resolve an exact quote's 1-based, possibly overlapping occurrence."""
    obj = _fields(evidence, {"quote", "occurrence"}, {"start", "end"}, "evidence")
    quote = _string(obj["quote"], "evidence.quote")
    occurrence = obj["occurrence"]
    if type(occurrence) is not int or occurrence < 1:
        raise GraphError("evidence.occurrence must be a positive integer")
    start = -1
    for _ in range(occurrence):
        start = text.find(quote, start + 1)
        if start < 0:
            raise GraphError("evidence quote/occurrence does not exist in the original text")
    end = start + len(quote)
    # Stored graphs must also be checked, rather than silently repairing stale offsets.
    for key, expected in (("start", start), ("end", end)):
        if key in obj and (type(obj[key]) is not int or obj[key] != expected):
            raise GraphError(f"evidence.{key} disagrees with original text")
    return {"quote": quote, "occurrence": occurrence, "start": start, "end": end}


def _list(value: Any, where: str) -> list:
    if not isinstance(value, list):
        raise GraphError(f"{where}: expected list")
    return value


def validate_graph(raw: Any, text: str) -> dict:
    """Validate supported structure only; source spans cannot establish semantic truth."""
    obj = _fields(raw, {"root_ids", "nodes", "edges"}, set(), "graph")
    nodes = []
    ids = set()
    semantic_nodes = set()
    for index, item in enumerate(_list(obj["nodes"], "nodes")):
        n = _fields(item, {"id", "type", "evidence"}, set(), f"node[{index}]")
        nid = _string(n["id"], "node.id")
        if nid in ids:
            raise GraphError("duplicate node id")
        if not isinstance(n["type"], str) or n["type"] not in NODE_TYPES:
            raise GraphError("unknown node type")
        ev = align_evidence(n["evidence"], text)
        semantic_key = (n["type"], ev["start"], ev["end"])
        if semantic_key in semantic_nodes:
            raise GraphError("duplicate typed source span")
        semantic_nodes.add(semantic_key)
        ids.add(nid)
        nodes.append({"id": nid, "type": n["type"], "surface_text": ev["quote"], "evidence": ev})
    if not nodes:
        raise GraphError("a successful graph requires at least one node")
    by_id = {n["id"]: n for n in nodes}
    roots = _list(obj["root_ids"], "root_ids")
    if not roots or any(not isinstance(r, str) or r not in ids for r in roots):
        raise GraphError("roots must reference existing nodes")
    if len(set(roots)) != len(roots):
        raise GraphError("duplicate roots")
    if any(by_id[r]["type"] not in {"action", "artifact"} for r in roots):
        raise GraphError("roots must be actions or requested artifacts")

    edges = []
    edge_keys = set()
    for index, item in enumerate(_list(obj["edges"], "edges")):
        e = _fields(item, {"source", "target", "type", "evidence"}, set(), f"edge[{index}]")
        source, target, kind = e["source"], e["target"], e["type"]
        if not all(isinstance(x, str) for x in (source, target, kind)):
            raise GraphError("edge endpoints and type must be strings")
        if source not in ids or target not in ids or source == target:
            raise GraphError("dangling or self-referencing edge")
        if kind not in EDGE_TYPES:
            raise GraphError("unknown edge type")
        st, tt = by_id[source]["type"], by_id[target]["type"]
        if kind in {"agent", "patient", "instrument", "located_at", "origin", "purpose", "produces"} and st != "action":
            raise GraphError(f"{kind} source must be action")
        if kind == "produces" and tt != "artifact":
            raise GraphError("produces target must be artifact")
        if kind == "precedes" and (st != "action" or tt != "action"):
            raise GraphError("precedes requires two actions")
        if kind == "condition_of" and st != "condition":
            raise GraphError("condition_of source must be condition")
        if kind == "constraint_of" and st != "constraint":
            raise GraphError("constraint_of source must be constraint")
        if kind == "quantity_of" and st != "quantity":
            raise GraphError("quantity_of source must be quantity")
        if kind in {"condition_of", "constraint_of"} and tt not in {"action", "artifact", "entity"}:
            raise GraphError(f"{kind} target must be action/artifact/entity")
        key = (kind, source, target)
        if key in edge_keys:
            raise GraphError("duplicate edge")
        edge_keys.add(key)
        ev = align_evidence(e["evidence"], text)
        # Explicitly quoted relation context must include both endpoint spans.
        for nid in (source, target):
            endpoint = by_id[nid]["evidence"]
            if ev["start"] > endpoint["start"] or ev["end"] < endpoint["end"]:
                raise GraphError(f"edge[{index}] {kind}: evidence omits endpoint {nid} at [{endpoint['start']},{endpoint['end']}); use an exact source quote containing BOTH endpoints, or quote the entire source_text")
        edges.append({"source": source, "target": target, "type": kind, "evidence": ev})

    # Numbering returned by the model is arbitrary, so canonicalize it.
    nodes.sort(key=lambda n: (n["evidence"]["start"], n["evidence"]["end"], n["type"]))
    mapping = {n["id"]: f"n{i + 1}" for i, n in enumerate(nodes)}
    for n in nodes:
        n["id"] = mapping[n["id"]]
    for e in edges:
        e["source"], e["target"] = mapping[e["source"]], mapping[e["target"]]
    edges.sort(key=lambda e: (e["source"], e["type"], e["target"]))
    return {"root_ids": sorted(mapping[r] for r in roots), "nodes": nodes, "edges": edges}


def graph_payload(graph: dict) -> dict:
    """Remove derived surface text before revalidating a saved graph."""
    if not isinstance(graph, dict) or not {"root_ids", "nodes", "edges"} <= graph.keys():
        raise GraphError("saved graph must contain root_ids/nodes/edges")
    if not isinstance(graph["nodes"], list) or any(not isinstance(n, dict) or not {"id", "type", "evidence"} <= n.keys() for n in graph["nodes"]):
        raise GraphError("saved graph contains malformed nodes")
    return {
        "root_ids": graph["root_ids"],
        "nodes": [{k: n[k] for k in ("id", "type", "evidence")} for n in graph["nodes"]],
        "edges": graph["edges"],
    }


def validate_response(raw: Any, text: str) -> dict:
    obj = _fields(raw, {"status", "issues", "graph"}, set(), "response")
    if not isinstance(obj["status"], str) or obj["status"] not in {"ok", "partial", "refused"}:
        raise GraphError("model status must be ok/partial/refused")
    issues = _list(obj["issues"], "issues")
    if any(not isinstance(x, str) for x in issues):
        raise GraphError("issues must be strings")
    if obj["status"] == "refused":
        if obj["graph"] is not None:
            raise GraphError("refused response must have null graph")
        return {"status": "refused", "issues": issues, "graph": None}
    if obj["status"] == "partial" and not issues:
        raise GraphError("partial response requires a reason in issues")
    return {"status": obj["status"], "issues": issues, "graph": validate_graph(obj["graph"], text)}


def parse_response(raw: Any, text: str) -> dict:
    """Repair only exact-source edge context; never infer types or relationships.

    The strict validator remains the sole entry point for gold and saved graphs.
    A repaired model response is explicitly partial and carries an audit record.
    """
    try:
        return {**validate_response(raw, text), "repairs": []}
    except GraphError as original:
        original_error = original
    if not isinstance(raw, dict) or raw.get("status") not in ("ok", "partial"):
        raise original_error
    candidate = deepcopy(raw)
    graph = candidate.get("graph")
    if not isinstance(graph, dict) or not isinstance(graph.get("edges"), list):
        raise original_error
    # Validate the nodes and roots independently before using their positions.
    validate_graph({**graph, "edges": []}, text)
    positions = {n["id"]: align_evidence(n["evidence"], text) for n in graph["nodes"]}
    repairs = []
    for index, edge in enumerate(graph["edges"]):
        if not isinstance(edge, dict) or not isinstance(edge.get("source"), str) or not isinstance(edge.get("target"), str):
            raise original_error
        if edge["source"] not in positions or edge["target"] not in positions or "evidence" not in edge:
            raise original_error
        before = align_evidence(edge["evidence"], text)
        spans = [before, positions[edge["source"]], positions[edge["target"]]]
        start, end = min(s["start"] for s in spans), max(s["end"] for s in spans)
        if (start, end) != (before["start"], before["end"]):
            quote = text[start:end]
            occurrence, offset = 0, -1
            while offset < start:
                offset = text.find(quote, offset + 1)
                occurrence += 1
            after = {"quote": quote, "occurrence": occurrence}
            edge["evidence"] = after
            repairs.append({"edge_index": index, "kind": "expanded_exact_source_context",
                            "before": before, "after": align_evidence(after, text)})
    if not repairs:
        raise original_error
    # If ANY other contract error remains, reject the whole repair.
    result = validate_response(candidate, text)
    return {**result, "status": "partial", "repairs": repairs,
            "issues": [*result["issues"], "Relation evidence expanded locally from exact source positions; review required."]}


def result_record(goal: GoalRecord, status: str, graph=None, issues=None, **extra) -> dict:
    return {
        "schema_version": SCHEMA_VERSION, **goal.to_dict(), "status": status,
        "graph": graph, "issues": issues or [], **extra,
    }

