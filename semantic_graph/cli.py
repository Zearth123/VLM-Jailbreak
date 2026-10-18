"""python -m semantic_graph: extraction, optional benign decomposition, and evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .client import APIError, DeepSeekClient, ModelConfig
from .evaluation import evaluate, repeat_consistency
from .io import load_goals, prepare_dataset, read_jsonl, write_json
from .pipeline import load_run, replay_run, run_extraction
from .report import build_report

PACKAGE = Path(__file__).resolve().parent


def parser():
    p = argparse.ArgumentParser(description="忠实于原文的语义图抽取；不补全过程、不执行输入目标。")
    sub = p.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="固定官方数据、20条样本及人工标注模板")
    prepare.add_argument("--out", type=Path, default=PACKAGE / "data" / "harmbench-v1")
    prepare.add_argument("--source", type=Path, help="可选本地官方CSV；默认下载固定到commit的官方数据")
    prepare.add_argument("--seed", type=int, default=42)
    extract = sub.add_parser("extract", help="调用DeepSeek并校验图，支持同配置续跑")
    extract.add_argument("--input", type=Path, required=True)
    extract.add_argument("--subset", choices=["standard", "contextual", "copyright"])
    extract.add_argument("--out", type=Path, required=True)
    extract.add_argument("--cache", type=Path, default=PACKAGE / "cache")
    extract.add_argument("--limit", type=int)
    extract.add_argument("--workers", type=int, default=2)
    extract.add_argument("--repeat-id", type=int, default=0)
    extract.add_argument("--split", action="store_true", help="为无害文本在建图后调用semantic_split生成语义片段")
    probe = sub.add_parser("probe", help="只检查鉴权与可用模型，不发起生成")
    for sp in (extract, probe):
        sp.add_argument("--base-url", default="https://api.deepseek.com")
        sp.add_argument("--model", default="deepseek-flash")
        sp.add_argument("--max-tokens", type=int, default=8192)
        sp.add_argument("--temperature", type=float, default=0)
        sp.add_argument("--timeout", type=float, default=90)
        sp.add_argument("--retries", type=int, default=2)
        sp.add_argument("--key-stdin", action="store_true", help="从标准输入首行读取密钥，仅保存在当前进程环境")
    report = sub.add_parser("report", help="不调用模型，重新生成统计和图浏览器")
    report.add_argument("--run", type=Path, required=True)
    replay = sub.add_parser("replay", help="离线重解析已保存响应，不调用模型；保留原始实验")
    replay.add_argument("--run", type=Path, required=True)
    replay.add_argument("--out", type=Path, required=True)
    replay.add_argument("--cache", type=Path, help="可选：将回放结果写入新版缓存，避免重复调用")
    split = sub.add_parser("split", help="离线调用semantic_split拆分已有的无害文本语义图")
    split.add_argument("--run", type=Path, required=True, help="包含graphs.jsonl与输入快照的建图目录")
    ev = sub.add_parser("evaluate", help="仅接受完成双人复核的人工参考图")
    ev.add_argument("--run", type=Path, required=True)
    ev.add_argument("--gold", type=Path, required=True)
    ev.add_argument("--split", choices=["dev", "holdout", "all"], default="holdout")
    ev.add_argument("--out", type=Path, required=True)
    consistency = sub.add_parser("consistency", help="比较不同repeat-id独立抽取的一致性")
    consistency.add_argument("--runs", nargs="+", type=Path, required=True)
    consistency.add_argument("--out", type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "prepare":
            meta = prepare_dataset(args.out, args.source, args.seed)
            print(json.dumps({"out": str(args.out), "categories": meta["standard_categories"],
                              "human_annotation": "pending"}, ensure_ascii=False))
        elif args.command in {"extract", "probe"}:
            if args.key_stdin:
                if sys.stdin.isatty():
                    from getpass import getpass
                    key = getpass("DeepSeek credential (hidden): ").strip()
                else:
                    key = sys.stdin.readline().strip()
                if not key:
                    raise ValueError("empty credential input")
                os.environ["DEEPSEEK_API_KEY"] = key
                del key
            config = ModelConfig(args.base_url, args.model, args.temperature, args.max_tokens, args.timeout, args.retries)
            client = DeepSeekClient(config)
            if args.command == "probe":
                print(json.dumps({"models": client.models()}, ensure_ascii=False))
            else:
                goals = load_goals(args.input, args.subset)
                if args.limit is not None:
                    if args.limit <= 0:
                        raise ValueError("limit must be positive")
                    goals = goals[:args.limit]
                rows, meta = run_extraction(goals, client, args.out, args.cache, args.workers, args.repeat_id,
                                            progress=lambda message: print(message, flush=True))
                metrics = build_report(args.out)
                outcome = {"run": str(args.out), "statuses": metrics["statuses"], "complete": meta["complete"]}
                split_failed = False
                if args.split:
                    from semantic_split.pipeline import split_graph_run
                    split_rows, split_summary = split_graph_run(args.out)
                    outcome.update(split_statuses=split_summary["split_statuses"],
                                   n_fragments=split_summary["n_fragments"])
                    split_failed = any(r["split_status"] not in {"ok", "partial"} for r in split_rows)
                print(json.dumps(outcome, ensure_ascii=False))
                if not meta["complete"] or any(r.get("fatal_error") for r in rows):
                    return 2
                if split_failed:
                    return 2
        elif args.command == "report":
            print(json.dumps(build_report(args.run), ensure_ascii=False))
        elif args.command == "replay":
            replay_run(args.run, args.out, args.cache)
            print(json.dumps(build_report(args.out), ensure_ascii=False))
        elif args.command == "split":
            from semantic_split.pipeline import split_graph_run
            rows, summary = split_graph_run(args.run)
            print(json.dumps(summary, ensure_ascii=False))
            if any(r["split_status"] not in {"ok", "partial"} for r in rows):
                return 2
        elif args.command == "evaluate":
            predictions, _ = load_run(args.run)
            result = evaluate(predictions, read_jsonl(args.gold), args.split)
            write_json(args.out, result)
            print(json.dumps(result["micro"], ensure_ascii=False))
        elif args.command == "consistency":
            runs = [load_run(path) for path in args.runs]
            if any(meta.get("mode") == "replay" for _, meta in runs):
                raise ValueError("offline replays are not independent repeats")
            fixed_keys = ("input_fingerprint", "config", "prompt_sha256", "schema_version", "prompt_version")
            if any(any(meta.get(k) != runs[0][1].get(k) for k in fixed_keys) for _, meta in runs[1:]):
                raise ValueError("repeat runs must have the same input, model, prompt, and schema")
            result = repeat_consistency([rows for rows, _ in runs])
            write_json(args.out, result)
            print(json.dumps({k: v for k, v in result.items() if k != "per_sample"}, ensure_ascii=False))
        return 0
    except (ValueError, APIError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

