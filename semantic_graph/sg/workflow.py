"""端到端工作流:data_origin → 规范化输入 → 语义图 → 切分扫描 → 指标与报告。

一个 run 目录(沿用 benchmark 的产物约定):

    input.jsonl          规范化后的输入快照(join key = rid / BehaviorID)
    graphs.jsonl         每目标一张语义图(R2 节点与边,带 span)
    units.jsonl          逐单元的表面文本与探针分数(audit 用)
    splits.jsonl         目标 × N × 方法 的切分结果
    decomposability.jsonl τ 门禁的可计算判定
    run.meta.json        参数、版本、探针 sha、计数
    metrics.json/.csv    汇总指标
    report.md            人类可读报告(每个占位数值都带标签)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import extract, sources
from . import metrics as M
from . import split as S
from .probe import Probe, make_probe
from .schema import GoalRecord, dump_jsonl, write_text_lf

RUN_SCHEMA = "sg-run-0.1"
CN_TZ = timezone(timedelta(hours=8))


@dataclass
class RunSpec:
    subset: str = "mixed"
    limit: int | None = 40
    ns: tuple[int, ...] = (2, 3, 4, 5, 6)
    methods: tuple[str, ...] = ("s2", "s3") + S.BASELINES
    tau: float = 0.35
    image_budget: int | None = S.DEFAULT_IMAGE_BUDGET
    seed: int = 20260101
    cap: int = S.DEFAULT_CAP
    probe_name: str = "lexical"
    extractor: str = "a5-template"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ns"] = list(self.ns)
        d["methods"] = list(self.methods)
        return d


def _now() -> str:
    return datetime.now(CN_TZ).isoformat(timespec="seconds")


def collect_unit_scores(graph, probe: Probe) -> list[dict]:
    texts = S.unit_texts(graph)
    out: list[dict] = []
    for n in graph.nodes:
        text = texts[n.nid]
        out.append({
            "behavior_id": graph.behavior_id,
            "nid": n.nid,
            "ntype": n.ntype,
            "label": n.label,
            "serializable": n.serializable,
            "span": list(n.span),
            "text": text,
            "score": round(probe.score_text(text), 6),
        })
    return out


def run_workflow(
    spec: RunSpec,
    out_dir,
    records: list[GoalRecord] | None = None,
    probe: Probe | None = None,
    run_id: str | None = None,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_records = records if records is not None else sources.load_all()
    if records is not None:
        # 显式传入 records 时视为"已选定的输入",不再按 subset 二次过滤(便于定向复现与测试)
        if not all_records:
            raise ValueError("records 为空")
        subset = list(all_records)
    else:
        subset = sources.build_subset(all_records, spec.subset, spec.limit)
    sources.write_jsonl(subset, out / "input.jsonl")

    probe = probe if probe is not None else make_probe(spec.probe_name)

    graphs = [extract.build_graph(r.rid, r.goal, spec.extractor) for r in subset]
    problems = {}
    for g in graphs:
        bad = extract.check_invariants(g)
        if bad:
            problems[g.behavior_id] = bad
    if problems:
        raise ValueError(f"图的不变量被破坏,拒绝继续: {json.dumps(problems, ensure_ascii=False)}")
    dump_jsonl([g.to_dict() for g in graphs], out / "graphs.jsonl")

    unit_scores: list[dict] = []
    for g in graphs:
        unit_scores.extend(collect_unit_scores(g, probe))
    dump_jsonl(unit_scores, out / "units.jsonl")

    splits = []
    for g in graphs:
        if not g.nodes:
            continue  # 没有单元的图无从切分;它们出现在覆盖率与 unsupported 清单里
        splits.extend(
            S.sweep(g, probe, ns=spec.ns, methods=spec.methods, seed=spec.seed,
                    image_budget=spec.image_budget, cap=spec.cap)
        )
    dump_jsonl([s.to_dict() for s in splits], out / "splits.jsonl")

    details = [
        S.decomposability(g, probe, ns=spec.ns, tau=spec.tau,
                          image_budget=spec.image_budget, cap=spec.cap)
        for g in graphs
    ]
    dump_jsonl(details, out / "decomposability.jsonl")

    deps_cut_total = sum(len(s.deps_cut) for s in splits)
    if deps_cut_total:
        raise ValueError(f"依赖边被切断 {deps_cut_total} 次 —— 违反 docs/10 §4 S2 约束 ①")

    params = {
        "subset": spec.subset,
        "limit": spec.limit,
        "ns": list(spec.ns),
        "methods": list(spec.methods),
        "tau": spec.tau,
        "image_budget": spec.image_budget,
        "seed": spec.seed,
        "cap": spec.cap,
        "extractor": spec.extractor,
    }
    metrics = M.compute_all(subset, graphs, unit_scores, splits, details, probe, params)
    M.write_metrics(metrics, out / "metrics.json")
    M.write_metrics_csv(metrics, out / "metrics.csv")

    meta = {
        "run_schema": RUN_SCHEMA,
        "run_id": run_id or out.name,
        "created": _now(),
        "spec": spec.to_dict(),
        "subset": spec.subset,
        "limit": spec.limit,
        "extractor": spec.extractor,
        "probe": probe.describe(),
        "cap": spec.cap,
        "image_budget": spec.image_budget,
        "n_input_records": len(subset),
        "n_graphs": len(graphs),
        "n_units": len(unit_scores),
        "n_splits": len(splits),
        "deps_cut_total": deps_cut_total,
        "docs": ["docs/10-semantic-graph-modeling.md", "docs/04-modules.md M1/M2", "docs/03-data-model.md"],
    }
    write_text_lf(out / "run.meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
    write_text_lf(out / "report.md", M.render_report(meta, metrics))

    return {"out_dir": str(out), "meta": meta, "metrics": metrics}


def load_run(run_dir) -> tuple[dict, dict]:
    run = Path(run_dir)
    meta = json.loads((run / "run.meta.json").read_text(encoding="utf-8"))
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    return meta, metrics


def render_existing(run_dir) -> Path:
    """只重渲染报告 —— 不 import torch,零 GPU 成本(benchmark 的 torch-free 约定)。"""
    run = Path(run_dir)
    meta, metrics = load_run(run)
    path = run / "report.md"
    write_text_lf(path, M.render_report(meta, metrics))
    return path
