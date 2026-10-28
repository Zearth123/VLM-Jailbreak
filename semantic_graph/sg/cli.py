"""命令行入口:

    python -m sg.cli prepare --subset mixed --limit 20 --out artifacts/subset_mixed20.jsonl
    python -m sg.cli run     --subset mixed --limit 20 --out runs/demo
    python -m sg.cli report  --run runs/demo

所有子命令都返回退出码(0 成功),便于脚本化与断点续跑。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import relax as rx
from . import sources
from . import split as S
from . import twosplit as ts
from .judge import JudgeUnavailable
from .probe import ProbeUnavailable, make_probe
from .workflow import RunSpec, render_existing, run_workflow

HERE = Path(__file__).resolve().parents[1]  # semantic_graph/


def ts_now() -> str:
    """产物里的时间戳(带时区,便于排序)。"""
    from datetime import datetime, timedelta, timezone

    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in str(text).split(",") if x.strip())


def _parse_strs(text: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in str(text).split(",") if x.strip())


def cmd_prepare(args: argparse.Namespace) -> int:
    records = sources.load_all()
    subset = sources.build_subset(records, args.subset, args.limit)
    out = Path(args.out)
    n = sources.write_jsonl(subset, out)
    counts = sources.summary_counts(subset)
    print(json.dumps({
        "out": str(out),
        "n_written": n,
        "subset": args.subset,
        "limit": args.limit,
        "semantic": counts["semantic"],
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    spec = RunSpec(
        subset=args.subset,
        limit=args.limit,
        ns=_parse_ints(args.ns),
        methods=_parse_strs(args.methods),
        tau=args.tau,
        image_budget=args.image_budget,
        seed=args.seed,
        cap=args.cap,
        probe_name=args.probe,
    )
    unknown = [m for m in spec.methods if m not in S.METHODS]
    if unknown:
        print(f"未知方法: {unknown};可选 {S.METHODS}", file=sys.stderr)
        return 2
    result = run_workflow(spec, args.out)
    metrics = result["metrics"]
    print(json.dumps({
        "out_dir": result["out_dir"],
        "probe": metrics["probe"],
        "counts": metrics["counts"],
        "coverage_rate": metrics["coverage"]["rate"],
        "decomposable_rate": metrics["decomposability"]["rate"],
        "report": str(Path(result["out_dir"]) / "report.md"),
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    path = render_existing(args.run)
    print(json.dumps({"report": str(path)}, ensure_ascii=False))
    return 0


def cmd_split2(args: argparse.Namespace) -> int:
    judge_kwargs: dict = {}
    if args.judge in ("http", "llm", "http-llm"):
        judge_kwargs = {"url": args.judge_url, "model": args.judge_model,
                        "key_env": args.judge_key_env}
    result = ts.run_split2(
        args.rid, args.out, judge_name=args.judge, judge_kwargs=judge_kwargs, k=args.k,
        do_sweep=not args.no_sweep, sweep_limit=args.sweep_limit,
        created=ts_now(),
    )
    ev = result["evaluation"]
    print(json.dumps({
        "out": result["out"],
        "judge": result["meta"]["judge_desc"],
        "authoritative_judge": ev["authoritative_judge"],
        "subgraphs": [{"sid": r["sid"], "channel": r["channel"],
                       "lexical": r["judge_lexical"]["label"],
                       "llm": (r.get("judge_llm") or {}).get("label", "pending"),
                       "caption": r["caption"]["status"]} for r in ev["rows"]],
        "aggregate": (ev["aggregate"].get("judge_lexical") or {}).get("label"),
        "fidelity": ev["fidelity"].get("cover_rate"),
        "traceable": ev["traceability"]["all_traceable"],
        "invariants": ev["invariants"] or "pass",
        "sweep_n": result["sweep_n"],
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_relax(args: argparse.Namespace) -> int:
    result = rx.run_relax(args.rid, args.out, k=args.k, beta=args.beta, judge_name=args.judge,
                          created=ts_now())
    ps = result["posterior_structural"]
    print(json.dumps({
        "out": result["out"],
        "orders": result["meta"]["n_orders"],
        "structural_entropy": round(ps["entropy"], 4),
        "max_entropy": round(ps["max_entropy"], 4),
        "uniform": ps["uniform"],
        "carrier_marginals": {k: {c: round(v, 4) for c, v in row.items()}
                              for k, row in ps["carrier_marginals"].items()},
        "traceable": result["meta"]["traceability"]["all_traceable"],
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    """本机:渲染 A0 图像条件集 + manifest(云上探针的输入契约)。"""
    from . import extract, visual as V

    record = next((r for r in sources.load_harmbench() if r.rid == args.rid), None)
    if record is None:
        raise SystemExit(f"HarmBench 里没有 {args.rid}")
    graph = extract.build_graph(args.rid, record.goal)
    pair = ts.build_pair(graph, ts.recommend_groups(graph))
    specs = V.sweep_specs("a0", quick=not args.full)
    out_dir = Path(args.out) if args.out else HERE / "artifacts" / f"inject_{args.rid}"
    result = V.write_condition_set(graph, pair, specs, out_dir)
    V.write_text_lf(Path(out_dir) / "report.md",
                    V.render_report(graph, pair, result, created=ts_now()))
    print(json.dumps({"out_dir": result["out_dir"], "n_images": result["n_images"],
                      "manifest": result["manifest"],
                      "report": str(Path(out_dir) / "report.md"),
                      "arm": "a0", "quick": not args.full}, ensure_ascii=False, indent=2))
    return 0


def cmd_recon(args: argparse.Namespace) -> int:
    """本机:把云上 dump 合成判据 A/B/C3 与互补性矩阵(纯 numpy,不需 torch)。"""
    from . import recon as RC

    dump = RC.load_dump(args.dump)
    rows = None
    if args.manifest:
        rows = [json.loads(l) for l in Path(args.manifest).read_text(encoding="utf-8").splitlines()
                if l.strip()]
    result = RC.evaluate(dump, rows)
    out = Path(args.out) if args.out else HERE / "artifacts" / f"recon_{Path(args.dump).name}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(RC.render_report(result, created=ts_now()), encoding="utf-8", newline="\n")
    summary = {arm: {k: v for k, v in s.items()
                     if k in ("n", "c1", "c2", "c3", "b3_harmless_rate",
                              "delta_r1_mean", "delta_r3_mean")}
               for arm, s in result["by_arm"].items()}
    print(json.dumps({"report": str(out), "by_arm": summary,
                      "n_conditions": len(result["per_condition"])},
                     ensure_ascii=False, indent=2))
    return 0


def cmd_threshold(args: argparse.Namespace) -> int:
    """逐片段阈值判据:每个子图/每个模态产物都不被识别,只有 union 被识别。

    结构判定(**模型无关**)当场算;`recognized`/`guard` 两列必须由云上读数补。
    """
    from . import extract
    from . import threshold as TH

    record = next((r for r in sources.load_harmbench() if r.rid == args.rid), None)
    if record is None:
        raise SystemExit(f"HarmBench 里没有 {args.rid}")
    graph = extract.build_graph(args.rid, record.goal)
    pair = ts.build_pair(graph, ts.recommend_groups(graph))
    readings = None
    if args.readings:
        readings = {}
        for line in Path(args.readings).read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                readings[d["item_id"]] = d
    result = TH.evaluate(graph, pair, readings=readings)
    out = Path(args.out) if args.out else HERE / "artifacts" / f"threshold_{args.rid}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(TH.render_report(result, rid=args.rid, created=ts_now()),
                         encoding="utf-8", newline="\n")
    print(json.dumps({
        "report": str(out),
        "gate": result["gate"],
        "goal_type": result["goal_type"]["type"],
        "each_subgraph_pattern_incomplete": result["structural"]["each_subgraph_pattern_incomplete"],
        "union_pattern_complete": result["structural"]["union_pattern_complete"],
        "items": [{"item_id": r["item_id"], "kind": r["kind"],
                   "complete_patterns": r["complete_patterns"], "gate": r["gate"]}
                  for r in result["rows"]],
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_embedgraph(args: argparse.Namespace) -> int:
    """本机:用向量 dump 建图并切分(docs/12)。缺 dump 时显式失败。"""
    from . import embedgraph as EG

    out = args.out or str(HERE / "artifacts" / f"vector_{args.rid or 'dump'}.md")
    try:
        result = EG.run(args.dump, out, rid=args.rid, k=args.k, tau=args.tau,
                        knn=args.knn, created=ts_now())
    except EG.EmbedUnavailable as exc:
        raise SystemExit(f"[embedgraph] {exc}")
    print(json.dumps({
        "out": result["out"], "rid": result["rid"], "k": result["k"], "tau": result["tau"],
        "n_fragments": result["n_fragments"], "n_unmapped": result["n_unmapped"],
        "cut": result["cut"], "n_subgraphs": result["n_subgraphs"], "gate": result["gate"],
        "cut_cost": {f"{a}->{b}": v for (a, b), v in result["costs"].items()},
        "affinity": result["affinity"],
    }, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sg", description="第一步 语义图建模 工作流(docs/10)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare", help="data_origin → 规范化 JSONL 子集")
    pp.add_argument("--subset", default="mixed", choices=sources.SUBSETS)
    pp.add_argument("--limit", type=int, default=40)
    pp.add_argument("--out", default=str(HERE / "artifacts" / "subset.jsonl"))
    pp.set_defaults(func=cmd_prepare)

    pr = sub.add_parser("run", help="跑一次完整工作流")
    pr.add_argument("--subset", default="mixed", choices=sources.SUBSETS)
    pr.add_argument("--limit", type=int, default=40)
    pr.add_argument("--ns", default="2,3,4,5,6")
    pr.add_argument("--methods", default=",".join(("s2", "s3") + S.BASELINES))
    pr.add_argument("--tau", type=float, default=0.35)
    pr.add_argument("--image-budget", type=int, default=S.DEFAULT_IMAGE_BUDGET)
    pr.add_argument("--seed", type=int, default=20260101)
    pr.add_argument("--cap", type=int, default=S.DEFAULT_CAP)
    pr.add_argument("--probe", default="lexical", help="lexical(占位) / xguard(需 torch+权重)")
    pr.add_argument("--out", default=str(HERE / "runs" / "demo"))
    pr.set_defaults(func=cmd_run)

    p2 = sub.add_parser("split2", help="把一条 goal 的语义图拆成两个子图并逐条判(只做投影)")
    p2.add_argument("--rid", default="safrole_oil_extraction_guide")
    p2.add_argument("--k", type=int, default=2)
    p2.add_argument("--judge", default="lexical", help="lexical(非权威) / http(权威) / local(权威)")
    p2.add_argument("--judge-url", default=None, help="OpenAI 兼容 chat completions 端点")
    p2.add_argument("--judge-model", default=None)
    p2.add_argument("--judge-key-env", default="OPENAI_API_KEY")
    p2.add_argument("--no-sweep", action="store_true", help="跳过放宽候选附录")
    p2.add_argument("--sweep-limit", type=int, default=20)
    p2.add_argument("--out", default=str(HERE / "artifacts" / "example_two_subgraphs.md"))
    p2.set_defaults(func=cmd_split2)

    p3 = sub.add_parser("relax", help="切分+通道对齐的潜变量形式化(论文式 generation order)")
    p3.add_argument("--rid", default="safrole_oil_extraction_guide")
    p3.add_argument("--k", type=int, default=2)
    p3.add_argument("--beta", type=float, default=8.0)
    p3.add_argument("--judge", default="lexical")
    p3.add_argument("--out", default=str(HERE / "artifacts" / "example_relax.md"))
    p3.set_defaults(func=cmd_relax)

    p6 = sub.add_parser("threshold", help="逐片段阈值判据(每片段不被识别,只有 union 被识别)")
    p6.add_argument("--rid", default="safrole_oil_extraction_guide")
    p6.add_argument("--readings", default=None, help="云上读数 jsonl(item_id/recognized/guard)")
    p6.add_argument("--out", default=None)
    p6.set_defaults(func=cmd_threshold)

    p8 = sub.add_parser("embedgraph", help="向量建图 + 切分(docs/12);需 GPU 主机导出的 dump")
    p8.add_argument("--dump", required=True, help="tools/embed_probe.py 产出的目录")
    p8.add_argument("--rid", default=None, help="dump 里有多条文本时必填")
    p8.add_argument("--k", type=int, default=2)
    p8.add_argument("--tau", type=float, default=0.35, help="kNN 相似边阈值")
    p8.add_argument("--knn", type=int, default=3, help="每个片段保留的最近邻数")
    p8.add_argument("--out", default=None)
    p8.set_defaults(func=cmd_embedgraph)

    pd = sub.add_parser("report", help="只重渲染 report.md(不加载探针)")
    pd.add_argument("--run", required=True)
    pd.set_defaults(func=cmd_report)

    p4 = sub.add_parser("render", help="渲染 A0 图像条件集(只做投影,不产生新内容)")
    p4.add_argument("--rid", default="safrole_oil_extraction_guide")
    p4.add_argument("--full", action="store_true", help="全部字号/扰动档(默认只跑 quick 档)")
    p4.add_argument("--out", default=None)
    p4.set_defaults(func=cmd_render)

    p5 = sub.add_parser("recon", help="把云上 dump 合成判据 A/B/C3 + 互补性矩阵")
    p5.add_argument("--dump", required=True, help="cloud_probe.py 产出的目录")
    p5.add_argument("--manifest", default=None, help="渲染 manifest(补 goal/units 元数据)")
    p5.add_argument("--out", default=None)
    p5.set_defaults(func=cmd_recon)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ProbeUnavailable, JudgeUnavailable) as e:
        print(f"依赖不可用: {e}", file=sys.stderr)
        print("→ 先用 --probe lexical / --judge lexical 验证流程本身,"
              "再在具备条件的机器上换真探针/真判官", file=sys.stderr)
        return 3
    except (ValueError, FileNotFoundError) as e:
        print(f"输入/配置错误: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
