"""Deterministic action-centered decomposition; all evidence stays in the source.

Fragments are overlapping subgraphs, not independent paraphrases. Shared objects
and scoped modifiers may occur in several fragments. Relations between actions
remain explicit rather than merging every action through a shared object.
"""

from __future__ import annotations

from copy import deepcopy

from semantic_graph.schema import GraphError, fingerprint, graph_payload, validate_graph

from . import SPLIT_VERSION

ROLE_EDGES = {"agent", "patient", "instrument", "located_at", "origin", "produces"}
MODIFIER_EDGES = {"quantity_of", "condition_of", "constraint_of"}
SOURCE_STATUSES = {"ok", "partial", "refused", "invalid", "error"}


def _validate_input(row):
    if not isinstance(row, dict):
        raise ValueError("input record must be an object")
    for key in ("id", "text"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if "category" in row and not isinstance(row["category"], str):
        raise ValueError("category must be a string")
    issues = row.get("issues", [])
    if not isinstance(issues, list) or any(not isinstance(item, str) for item in issues):
        raise ValueError("issues must be a string list")
    status = row.get("status")
    if status is None and row.get("graph") is not None:
        status = "ok"
    if status is not None and (not isinstance(status, str) or status not in SOURCE_STATUSES):
        raise ValueError("unsupported source status")
    if status in {"refused", "invalid", "error"} and row.get("graph") is not None:
        raise ValueError("failed source status must have a null graph")
    if status in {"ok", "partial"} and row.get("graph") is None:
        raise ValueError("successful source status requires a graph")
    if status == "partial" and not issues:
        raise ValueError("partial source status requires issues")
    return status


def _modifier_closure(members, edges):
    members = set(members)
    while True:
        previous = len(members)
        for edge in edges:
            if edge["type"] in MODIFIER_EDGES and edge["target"] in members:
                members.add(edge["source"])
        if len(members) == previous:
            return members


def _source_spans(nodes, text):
    """Union node evidence intervals; keep disjoint intervals separate."""
    intervals = sorted((n["evidence"]["start"], n["evidence"]["end"]) for n in nodes)
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [{"start": start, "end": end, "quote": text[start:end]} for start, end in merged]


def _groups(graph):
    nodes, edges = graph["nodes"], graph["edges"]
    by_id = {node["id"]: node for node in nodes}
    groups = []
    for node in nodes:
        if node["type"] != "action":
            continue
        members = {node["id"]}
        members.update(edge["target"] for edge in edges
                       if edge["source"] == node["id"] and edge["type"] in ROLE_EDGES
                       and by_id[edge["target"]]["type"] != "action")
        groups.append((node["id"], "action", _modifier_closure(members, edges)))

    assigned = set().union(*(members for _, _, members in groups))
    for node in nodes:
        if node["id"] in graph["root_ids"] and node["id"] not in assigned:
            members = _modifier_closure({node["id"]}, edges)
            groups.append((node["id"], "artifact", members))
            assigned.update(members)

    # Preserve disconnected evidence as context, and flag it for review.
    remaining = set(by_id) - assigned
    for node in nodes:
        if node["id"] not in remaining:
            continue
        component, pending = set(), [node["id"]]
        while pending:
            current = pending.pop()
            if current not in remaining:
                continue
            remaining.remove(current)
            component.add(current)
            for edge in edges:
                if edge["source"] == current and edge["target"] in remaining:
                    pending.append(edge["target"])
                if edge["target"] == current and edge["source"] in remaining:
                    pending.append(edge["source"])
        groups.append((None, "context", component))
    order = {node["id"]: i for i, node in enumerate(nodes)}
    return sorted(groups, key=lambda item: min(order[nid] for nid in item[2]))


def _decompose(graph, text):
    nodes, edges = graph["nodes"], graph["edges"]
    edge_ids = [f"e{i + 1}" for i in range(len(edges))]
    fragments = []
    for anchor, kind, members in _groups(graph):
        selected = [node for node in nodes if node["id"] in members]
        spans = _source_spans(selected, text)
        start, end = spans[0]["start"], spans[-1]["end"]
        fragments.append({
            "id": f"f{len(fragments) + 1}", "kind": kind, "anchor_id": anchor,
            "root_ids": [rid for rid in graph["root_ids"] if rid in members],
            "node_ids": [node["id"] for node in selected],
            "edge_ids": [eid for eid, edge in zip(edge_ids, edges)
                         if edge["source"] in members and edge["target"] in members],
            "source_spans": spans,
            "display_text": " … ".join(span["quote"] for span in spans),
            "source_context": {"start": start, "end": end, "quote": text[start:end]},
        })

    membership = {node["id"]: [f["id"] for f in fragments if node["id"] in f["node_ids"]]
                  for node in nodes}
    internal = {eid for fragment in fragments for eid in fragment["edge_ids"]}
    cross = [{"edge_id": eid, **deepcopy(edge),
              "source_fragment_ids": membership[edge["source"]],
              "target_fragment_ids": membership[edge["target"]]}
             for eid, edge in zip(edge_ids, edges) if eid not in internal]
    scopes = []
    for node in nodes:
        if node["type"] != "constraint":
            continue
        targets = [edge["target"] for edge in edges
                   if edge["type"] == "constraint_of" and edge["source"] == node["id"]]
        scopes.append({"node_id": node["id"], "target_node_ids": targets,
                       "fragment_ids": [f["id"] for f in fragments
                                        if any(target in f["node_ids"] for target in targets)]})

    covered_nodes = {nid for fragment in fragments for nid in fragment["node_ids"]}
    covered_edges = internal | {edge["edge_id"] for edge in cross}
    if covered_nodes != set(membership) or covered_edges != set(edge_ids):
        raise AssertionError("decomposition lost source graph elements")
    return {
        "fragments": fragments,
        "edge_index": [{"id": eid, **deepcopy(edge)} for eid, edge in zip(edge_ids, edges)],
        "cross_relations": cross,
        "shared_node_ids": [nid for nid, owners in membership.items() if len(owners) > 1],
        "constraint_scopes": scopes,
        "coverage": {"nodes_preserved": len(covered_nodes), "nodes_total": len(nodes),
                     "edges_preserved": len(covered_edges), "edges_total": len(edges)},
    }


def split_record(row: dict) -> dict:
    """Split a source-aligned graph, retaining all input metadata and statuses.

    Invalid records produce explicit failures, so batch processing never silently
    drops a row. ``ok`` only describes structural preservation of the input graph.
    """
    result = deepcopy(row) if isinstance(row, dict) else {"input_record": deepcopy(row)}
    result.update(split_version=SPLIT_VERSION, split_status="invalid", split_issues=[],
                  fragments=[], edge_index=[], cross_relations=[], shared_node_ids=[],
                  constraint_scopes=[], coverage=None)
    try:
        status = _validate_input(row)
        if status in {"refused", "invalid", "error"}:
            result.update(split_status="skipped", split_issues=[f"source_status:{status}"])
            return result
        if row.get("graph") is None:
            result.update(split_status="missing_graph", split_issues=["Supply a graph or use --extract."])
            return result
        graph = validate_graph(graph_payload(row["graph"]), row["text"])
        result.update(graph=graph, status=status, source_graph_sha256=fingerprint(graph))
        result.update(_decompose(graph, row["text"]))
        issues = []
        if status == "partial":
            issues.append("Source graph is partial; its uncertainty is retained.")
        if any(fragment["kind"] == "context" for fragment in result["fragments"]):
            issues.append("Unanchored evidence retained as context; review its role and scope.")
        result.update(split_status="partial" if issues else "ok", split_issues=issues)
    except (GraphError, ValueError, TypeError, KeyError) as exc:
        result["split_issues"] = [str(exc)]
    return result
