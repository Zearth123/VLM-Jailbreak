"""Handwritten benign graphs for offline examples, not model predictions or gold."""

from __future__ import annotations


def _node(nid, kind, quote, occurrence=1):
    return {"id": nid, "type": kind, "evidence": {"quote": quote, "occurrence": occurrence}}


def _edge(source, target, kind, text):
    return {"source": source, "target": target, "type": kind,
            "evidence": {"quote": text, "occurrence": 1}}


def benign_examples():
    """Return fresh examples including explicit order, shared scope, and a frame."""
    text = "先把两个红色杯子放在桌上，不要叠放，然后拍摄杯子。"
    chinese = {
        "id": "cups-zh", "text": text, "category": "benign", "status": "ok", "issues": [],
        "graph": {"root_ids": ["place", "photo"], "nodes": [
            _node("place", "action", "放"), _node("photo", "action", "拍摄"),
            _node("cups", "entity", "红色杯子"), _node("table", "entity", "桌上"),
            _node("count", "quantity", "两个"), _node("neg", "constraint", "不要叠放"),
            _node("cups-again", "entity", "杯子", 2),
        ], "edges": [
            _edge("place", "cups", "patient", text),
            _edge("place", "table", "located_at", text),
            _edge("count", "cups", "quantity_of", text),
            _edge("neg", "place", "constraint_of", text),
            _edge("photo", "cups-again", "patient", text),
            _edge("place", "photo", "precedes", text),
        ]},
    }
    text = "Wash and dry the cup; both actions must avoid soap."
    shared = {
        "id": "shared-en", "text": text, "category": "benign", "status": "ok", "issues": [],
        "graph": {"root_ids": ["wash", "dry"], "nodes": [
            _node("wash", "action", "Wash"), _node("dry", "action", "dry"),
            _node("cup", "entity", "cup"),
            _node("neg", "constraint", "both actions must avoid soap"),
        ], "edges": [
            _edge("wash", "cup", "patient", text), _edge("dry", "cup", "patient", text),
            _edge("neg", "wash", "constraint_of", text),
            _edge("neg", "dry", "constraint_of", text),
        ]},
    }
    text = "说明如何清洗杯子。"
    frame = {
        "id": "frame-zh", "text": text, "category": "benign", "status": "ok", "issues": [],
        "graph": {"root_ids": ["explain"], "nodes": [
            _node("explain", "action", "说明"), _node("wash", "action", "清洗"),
            _node("cup", "entity", "杯子"),
        ], "edges": [
            _edge("explain", "wash", "patient", text),
            _edge("wash", "cup", "patient", text),
        ]},
    }
    text = "一张白色背景的海报。"
    artifact = {
        "id": "poster-zh", "text": text, "category": "benign", "status": "ok", "issues": [],
        "graph": {"root_ids": ["poster"], "nodes": [
            _node("poster", "artifact", "海报"), _node("count", "quantity", "一张"),
            _node("background", "condition", "白色背景"),
        ], "edges": [
            _edge("count", "poster", "quantity_of", text),
            _edge("background", "poster", "condition_of", text),
        ]},
    }
    for row in (chinese, shared, frame, artifact):
        row["annotation_origin"] = "handwritten_benign_example"
    return [chinese, shared, frame, artifact]
