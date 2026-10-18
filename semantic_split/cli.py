"""Offline graph decomposition and explicit opt-in extraction for benign text."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from semantic_graph.client import APIError, DeepSeekClient, ModelConfig
from semantic_graph.io import read_jsonl, write_json, write_jsonl
from semantic_graph.pipeline import extract_goal
from semantic_graph.schema import GoalRecord

from . import split_record
from .examples import benign_examples
from .pipeline import summarize

PACKAGE = Path(__file__).resolve().parent


def parser():
    p = argparse.ArgumentParser(description="无害文本的语义拆分：动作/产物片段、原文证据及跨片段关系。")
    sub = p.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="使用内置中英文无害示例离线运行，无需密钥")
    demo.add_argument("--out", type=Path, required=True)
    split = sub.add_parser("split", help="拆分已有语义图，或显式调用模型抽取无害文本")
    source = split.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="JSONL/JSON记录，或作为单条记录的UTF-8文本文件")
    source.add_argument("--text", help="一段待拆分的无害文本")
    split.add_argument("--id", default="text-1", help="--text 输入的记录标识")
    split.add_argument("--out", type=Path, required=True)
    split.add_argument("--extract", action="store_true", help="允许为缺少语义图的原始文本调用DeepSeek")
    split.add_argument("--cache", type=Path, default=PACKAGE / "cache")
    split.add_argument("--key-stdin", action="store_true", help="隐藏输入密钥，仅用于--extract")
    split.add_argument("--base-url", default="https://api.deepseek.com")
    split.add_argument("--model", default=ModelConfig().model)
    split.add_argument("--timeout", type=float, default=90)
    split.add_argument("--max-tokens", type=int, default=8192)
    return p


def _load(args):
    if args.text is not None:
        rows = [{"id": args.id, "text": args.text}]
    elif args.input.suffix.lower() == ".txt":
        rows = [{"id": args.input.stem, "text": args.input.read_text(encoding="utf-8-sig")}]
    elif args.input.suffix.lower() == ".json":
        rows = json.loads(args.input.read_text(encoding="utf-8-sig"))
    else:
        rows = read_jsonl(args.input)
    if not isinstance(rows, list) or not rows:
        raise ValueError("input must contain a nonempty list of records")
    seen = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"row {index}: expected object")
        rid = row.get("id")
        if not isinstance(rid, str) or not rid.strip() or rid in seen:
            raise ValueError(f"row {index}: missing/non-string/duplicate id")
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError(f"row {index}: missing/empty/non-string text")
        if not isinstance(row.get("category", "unknown"), str):
            raise ValueError(f"row {index}: category must be a string")
        seen.add(rid)
    return rows


def _needs_extraction(row):
    return row.get("graph") is None and row.get("status") is None


def _client(args):
    key = None
    if args.key_stdin:
        if sys.stdin.isatty():
            from getpass import getpass
            key = getpass("DeepSeek credential (hidden): ").strip()
        else:
            key = sys.stdin.readline().strip()
        if not key:
            raise ValueError("empty credential input")
    return DeepSeekClient(ModelConfig(base_url=args.base_url, model=args.model,
                                     timeout=args.timeout, max_tokens=args.max_tokens), api_key=key)


def _write_run(out, inputs, rows):
    summary = summarize(inputs, rows)
    write_jsonl(out / "inputs.jsonl", inputs)
    write_jsonl(out / "fragments.jsonl", rows)
    write_json(out / "summary.json", summary)
    return summary


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
            raise ValueError("output directory is not empty; choose a new --out directory")
        if args.command == "demo":
            inputs = benign_examples()
        else:
            if args.key_stdin and not args.extract:
                raise ValueError("--key-stdin requires --extract")
            inputs = _load(args)
            if not args.extract and any(_needs_extraction(row) for row in inputs):
                raise ValueError("raw text has no graph; use --extract or supply an existing graph")
        rows, client, stopped = [], None, False
        for row in inputs:
            source = row
            if args.command == "split" and args.extract and _needs_extraction(row):
                if stopped:
                    result = split_record(row)
                    result.update(split_status="not_processed", split_issues=["Stopped after a fatal API error."])
                    rows.append(result)
                    continue
                if client is None:
                    client = _client(args)
                goal = GoalRecord(row["id"], row["text"], row.get("category", "unknown"))
                source = {**row, **extract_goal(goal, client, args.cache)}
                stopped = bool(source.get("fatal_error"))
            rows.append(split_record(source))
        summary = _write_run(args.out, inputs, rows)
        print(json.dumps({"out": str(args.out.resolve()), **summary}, ensure_ascii=False))
        return 0 if all(row["split_status"] in {"ok", "partial"} for row in rows) else 2
    except (ValueError, APIError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
