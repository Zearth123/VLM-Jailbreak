"""指标与报告(docs/10 §1.2 的六条判据 + §8 的 P2 门禁口径)。

全部 torch-free、纯 stdlib —— 报告链可零 GPU 重算(沿用 benchmark 的结构性保证)。
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .probe import Probe, is_placeholder
from .schema import GoalGraph, GoalRecord, SplitResult, write_text_lf

BASELINE_METHODS = ("s6_none", "s6_all", "s6_sentence", "s6_contiguous", "s6_random")
# N 由结构决定的方法:报告里 N 记为 natural(split.py 的 N_FREE_METHODS)
NATURAL_N_METHODS = ("s3", "s6_all", "s6_sentence")


def split_key(method: str, n) -> tuple[str, object]:
    """报告归行键:结构决定 N 的方法 → natural;其余用请求的 N。"""
    return (method, "natural" if method in NATURAL_N_METHODS else n)


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    """Wilson 95% 区间(与 benchmark 的口径一致:小样本、极端比例下有界)。"""
    if n <= 0:
        return (math.nan, math.nan, math.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


# ---------------------------------------------------------------- 覆盖率与可拆性


def coverage_stats(records: list[GoalRecord], graphs: list[GoalGraph]) -> dict:
    by_semantic: dict[str, list[int]] = {}
    by_task: dict[str, int] = {}
    unsupported: dict[str, int] = {}
    n_with_nodes = 0
    n_decomposable = 0
    n_multi_root = 0

    for r, g in zip(records, graphs):
        has_nodes = len(g.nodes) > 0
        n_with_nodes += int(has_nodes)
        n_decomposable += int(g.decomposable)
        n_multi_root += int(g.multi_root)
        by_task[g.task_kind] = by_task.get(g.task_kind, 0) + 1
        if g.unsupported_reason:
            unsupported[g.unsupported_reason] = unsupported.get(g.unsupported_reason, 0) + 1
        slot = by_semantic.setdefault(r.semantic_category or "unknown", [0, 0])
        slot[0] += int(has_nodes)
        slot[1] += 1

    k, n = n_with_nodes, len(graphs)
    p, lo, hi = wilson_ci(k, n)
    return {
        "k": k,
        "n": n,
        "rate": round(p, 6),
        "ci95": [round(lo, 6), round(hi, 6)],
        "n_decomposable_units": n_decomposable,
        "decomposable_rate": round(n_decomposable / n, 6) if n else math.nan,
        "n_multi_root": n_multi_root,
        "by_semantic": {s: {"k": v[0], "n": v[1]} for s, v in sorted(by_semantic.items())},
        "by_task_kind": dict(sorted(by_task.items())),
        "unsupported_reasons": dict(sorted(unsupported.items())),
    }


# ---------------------------------------------------------------- 逐目标指标


def _group_by_goal(splits: list[SplitResult]) -> dict[str, list[SplitResult]]:
    out: dict[str, list[SplitResult]] = {}
    for s in splits:
        out.setdefault(s.behavior_id, []).append(s)
    return out


def split_rows(splits: list[SplitResult], n_total_goals: int) -> list[dict]:
    """按 (method, n) 聚合。s3 的 N 由阶段数自然给出 → 记为 natural。"""
    buckets: dict[tuple[str, object], list[SplitResult]] = {}
    for s in splits:
        buckets.setdefault(split_key(s.method, s.n), []).append(s)

    rows: list[dict] = []
    for (method, n), items in sorted(buckets.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        goals = {s.behavior_id for s in items}
        maxes = [s.max_fragment_score for s in items]
        means = [s.mean_fragment_score for s in items]
        aggs = [s.aggregate_score for s in items]
        deltas = [s.aggregate_score - s.max_fragment_score for s in items]
        frags = [len(s.fragments) for s in items]
        rows.append({
            "method": method,
            "n": n,
            "n_goals": len(goals),
            "coverage_of_goals": round(len(goals) / n_total_goals, 6) if n_total_goals else math.nan,
            "mean_fragments": round(sum(frags) / len(frags), 3) if frags else math.nan,
            "mean_max_fragment_score": round(sum(maxes) / len(maxes), 6) if maxes else math.nan,
            "mean_mean_fragment_score": round(sum(means) / len(means), 6) if means else math.nan,
            "mean_aggregate_score": round(sum(aggs) / len(aggs), 6) if aggs else math.nan,
            "mean_aggregate_delta": round(sum(deltas) / len(deltas), 6) if deltas else math.nan,
            "infeasible_goals": sum(1 for s in items if s.feasible_partitions == 0 and s.method == "s2"),
            "truncated_goals": sum(1 for s in items if s.truncated),
            "deps_cut_total": sum(len(s.deps_cut) for s in items),
        })
    return rows


def lift_rows(splits: list[SplitResult]) -> list[dict]:
    """`lift`(docs/10 §1.2):相对 (a) 不切 baseline(N=1) 与 (b) **最强** S6 基线的下降幅度。"""
    by_goal = _group_by_goal(splits)
    none_max: dict[str, float] = {}
    baseline_best: dict[str, float] = {}
    for gid, items in by_goal.items():
        none = [s for s in items if s.method == "s6_none"]
        if none:
            none_max[gid] = min(s.max_fragment_score for s in none)
        base = [s.max_fragment_score for s in items if s.method in BASELINE_METHODS]
        if base:
            baseline_best[gid] = min(base)

    rows: list[dict] = []
    buckets: dict[tuple[str, object], list[SplitResult]] = {}
    for s in splits:
        buckets.setdefault(split_key(s.method, s.n), []).append(s)
    for (method, n), items in sorted(buckets.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        lifts_none = [none_max[s.behavior_id] - s.max_fragment_score
                      for s in items if s.behavior_id in none_max]
        lifts_base = [baseline_best[s.behavior_id] - s.max_fragment_score
                      for s in items if s.behavior_id in baseline_best]
        rows.append({
            "method": method,
            "n": n,
            "lift_vs_none": round(sum(lifts_none) / len(lifts_none), 6) if lifts_none else math.nan,
            "lift_vs_best_baseline": round(sum(lifts_base) / len(lifts_base), 6) if lifts_base else math.nan,
        })
    return rows


def isolation_floor(unit_scores: list[dict]) -> dict:
    """**M2 的结构性下界**:任何分区都装得下那个最热的单元,所以

        max_fragment_score ≥ max(unit_score)

    于是"M2 单独就能把分数压到 τ 以下"这件事在多数目标上不成立 —— τ 门禁要靠 M3
    降低最热单元的分数(docs/10 §8 的门禁因此是 M2+M3 的联合门)。
    """
    by_goal: dict[str, list[float]] = {}
    for u in unit_scores:
        by_goal.setdefault(u["behavior_id"], []).append(u["score"])
    floors = {g: max(v) for g, v in by_goal.items() if v}
    vals = sorted(floors.values())
    return {
        "n_goals_with_units": len(floors),
        "mean_floor": round(sum(vals) / len(vals), 6) if vals else math.nan,
        "median_floor": round(vals[len(vals) // 2], 6) if vals else math.nan,
        "max_floor": round(vals[-1], 6) if vals else math.nan,
        "floors": {g: round(f, 6) for g, f in sorted(floors.items())},
    }


def decomposability_rows(details: list[dict], tau: float) -> dict:
    n = len(details)
    k = sum(1 for d in details if d["decomposable"])
    p, lo, hi = wilson_ci(k, n)
    return {
        "tau": tau,
        "n": n,
        "n_decomposable": k,
        "rate": round(p, 6),
        "ci95": [round(lo, 6), round(hi, 6)],
        "rows": details,
    }


def probe_artifact_rows(splits: list[SplitResult]) -> list[dict]:
    """P4 聚合器对照 + 占位探针伪影的直接证据。

    `s6_all`(每单元一片)在词袋探针下必然达到结构下界;若 S2 与它无差,
    说明**占位探针无法奖励"语义"切分** —— 这是探针的性质,不是图的结论。
    """
    by_goal = _group_by_goal(splits)
    rows: list[dict] = []
    for gid, items in sorted(by_goal.items()):
        s2 = [s for s in items if s.method == "s2"]
        s6all = [s for s in items if s.method == "s6_all"]
        s6none = [s for s in items if s.method == "s6_none"]
        if not s2 or not s6all:
            continue
        best_s2 = min(s.max_fragment_score for s in s2)
        best_all = min(s.max_fragment_score for s in s6all)
        agg = min(s.aggregate_score for s in s2)
        rows.append({
            "behavior_id": gid,
            "s2_best_max": round(best_s2, 6),
            "s6_all_best_max": round(best_all, 6),
            "s2_gain_vs_s6_all": round(best_all - best_s2, 6),
            "s2_aggregate": round(agg, 6),
            "s2_aggregate_delta": round(agg - best_s2, 6),
            "s6_none_max": round(min(s.max_fragment_score for s in s6none), 6) if s6none else None,
        })
    return rows


def compute_all(
    records: list[GoalRecord],
    graphs: list[GoalGraph],
    unit_scores: list[dict],
    splits: list[SplitResult],
    decomposability: list[dict],
    probe: Probe,
    params: dict,
) -> dict:
    cov = coverage_stats(records, graphs)
    dead = [g.behavior_id for g in graphs if not g.nodes]
    return {
        "params": params,
        "probe": {**probe.describe(), "is_placeholder": is_placeholder(probe)},
        "counts": {
            "n_goals": len(records),
            "n_graphs": len(graphs),
            "n_units": len(unit_scores),
            "n_splits": len(splits),
            "n_goals_without_units": len(dead),
        },
        "coverage": cov,
        "isolation_floor": isolation_floor(unit_scores),
        "splits": split_rows(splits, len(records)),
        "lift": lift_rows(splits),
        "p4_aggregate": probe_artifact_rows(splits),
        "decomposability": decomposability_rows(decomposability, params.get("tau", 0.35)),
        "goals_without_units": dead,
        "placeholder_caveat": (probe.caveat if is_placeholder(probe) else ""),
    }


# ---------------------------------------------------------------- 落盘


def write_metrics(metrics: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(path, json.dumps(metrics, ensure_ascii=False, indent=2))


def write_metrics_csv(metrics: dict, path) -> None:
    path = Path(path)
    rows = metrics["splits"]
    if not rows:
        write_text_lf(path, "")
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def render_report(meta: dict, metrics: dict) -> str:
    """人类可读报告。占位探针的每一个数字都带显式标签。"""
    lines: list[str] = []
    probe = metrics["probe"]
    counts = metrics["counts"]
    cov = metrics["coverage"]
    floor = metrics["isolation_floor"]
    dec = metrics["decomposability"]

    lines.append(f"# 语义图建模 run · {meta.get('run_id', '?')}")
    lines.append("")
    lines.append(f"- 生成时间:{meta.get('created', '?')}")
    lines.append(f"- 子集:`{meta.get('subset')}` / limit={meta.get('limit')} / 目标数 {counts['n_goals']}")
    lines.append(f"- 抽取器:`{meta.get('extractor')}`(docs/10 §3 A5 路线)")
    lines.append(f"- 探针:`{probe['probe_id']}` v{probe['version']}(kind={probe['kind']}, sha={probe['sha']})")
    lines.append(f"- 枚举上限 cap={meta.get('cap')} / 图像带宽预算={meta.get('image_budget')} / τ={dec['tau']}")
    lines.append("")
    if probe["is_placeholder"]:
        lines.append("> ⚠️ **占位探针**:" + probe["caveat"])
        lines.append("> 本报告所有 `*_score` 都是词袋加权的占位值,**不是** docs/10 §5 P1 的 `sensitivity`;")
        lines.append("> `lift` 与 P4 的结论在换用真探针之前不可引用。")
        lines.append("")

    lines.append("## 1 · 覆盖率(coverage)")
    lines.append("")
    lines.append(f"- 抽出 ≥1 个单元的占比:**{cov['k']}/{cov['n']} = {cov['rate']:.3f}**"
                 f"(95% CI {cov['ci95'][0]:.3f}–{cov['ci95'][1]:.3f})")
    lines.append(f"- 至少两个单元(可切):{cov['n_decomposable_units']}"
                 f"({cov['decomposable_rate']:.3f})")
    lines.append(f"- 多根目标(docs/10 §1.3 事实 3):{cov['n_multi_root']}")
    lines.append("")
    lines.append("| SemanticCategory | 有图 | 合计 |")
    lines.append("|---|---|---|")
    for s, v in cov["by_semantic"].items():
        lines.append(f"| `{s}` | {v['k']} | {v['n']} |")
    lines.append("")
    lines.append("| task_kind | 条数 |")
    lines.append("|---|---|")
    for t, v in cov["by_task_kind"].items():
        lines.append(f"| `{t}` | {v} |")
    lines.append("")
    if cov["unsupported_reasons"]:
        lines.append("未建图的原因:")
        lines.append("")
        for reason, v in cov["unsupported_reasons"].items():
            lines.append(f"- `{reason}`:{v}")
        lines.append("")

    lines.append("## 2 · M2 的结构性下界(isolation floor)")
    lines.append("")
    lines.append("任何分区都必然把「最热的那个单元」装进某个片段,故")
    lines.append("`max_fragment_score ≥ max(unit_score)`。")
    lines.append("")
    lines.append(f"- 下界均值 {floor['mean_floor']:.3f} / 中位 {floor['median_floor']:.3f} / 最大 {floor['max_floor']:.3f}")
    lines.append(f"- 若 τ = {dec['tau']},则**只要最热单元超过 τ,M2 单独就过不了门**,必须由 M3 降它。")
    lines.append("")

    lines.append("## 3 · 切分与 lift")
    lines.append("")
    lines.append("| method | N | 目标数 | 片段数(均) | 最大片段分数(均) | 聚合分数(均) | 聚合−最大(均) | lift vs 不切 | lift vs 最强基线 | 不可行 | deps 被切 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    lift_by_key = {(r["method"], str(r["n"])): r for r in metrics["lift"]}
    for r in metrics["splits"]:
        lf = lift_by_key.get((r["method"], str(r["n"])), {})
        lines.append(
            f"| `{r['method']}` | {r['n']} | {r['n_goals']} | {r['mean_fragments']:.2f} | "
            f"{r['mean_max_fragment_score']:.3f} | "
            f"{r['mean_aggregate_score']:.3f} | {r['mean_aggregate_delta']:+.3f} | "
            f"{lf.get('lift_vs_none', float('nan')):+.3f} | {lf.get('lift_vs_best_baseline', float('nan')):+.3f} | "
            f"{r['infeasible_goals']} | {r['deps_cut_total']} |"
        )
    lines.append("")
    lines.append("> `deps_cut` 必须恒为 0(依赖边永不切断,docs/10 §4 S2 约束 ①)。")
    lines.append(">")
    lines.append("> **`不可行` 怎么读**:S2 的可行分区存在的前提是 `N ≤ 超级节点数`(依赖合并后的可切单元数),")
    lines.append("> 且受图像带宽预算约束。不可行的目标会**退化为一超级节点一片**,因此其分数接近隔离下界 ——")
    lines.append("> 不能把该行的 N 当成「这个 N 有效」。`片段数(均)` 同时给出实际切成了几片。")
    lines.append("")

    lines.append("## 4 · 可拆性(τ 门禁,可计算定义)")
    lines.append("")
    lines.append(f"- τ = {dec['tau']};可拆 **{dec['n_decomposable']}/{dec['n']} = {dec['rate']:.3f}**"
                 f"(95% CI {dec['ci95'][0]:.3f}–{dec['ci95'][1]:.3f})")
    lines.append("- 定义(docs/10 §8.1 步骤 3):不存在任何满足依赖约束的分区使所有片段 ≤ τ → `non_decomposable`")
    lines.append("")

    lines.append("## 5 · P4 聚合器对照 —— 与占位探针的伪影")
    lines.append("")
    rows = metrics["p4_aggregate"][:10]
    if rows:
        lines.append("| BehaviorID | S2 最大 | s6_all 最大 | S2 相对 s6_all | 聚合分数 | 聚合−最大 |")
        lines.append("|---|---|---|---|---|---|")
        for r in rows:
            lines.append(f"| `{r['behavior_id']}` | {r['s2_best_max']:.3f} | {r['s6_all_best_max']:.3f} | "
                         f"{r['s2_gain_vs_s6_all']:+.3f} | {r['s2_aggregate']:.3f} | {r['s2_aggregate_delta']:+.3f} |")
        lines.append("")
    lines.append("**必须读懂的伪影**:占位探针是**加性词袋** —— 它只看标记集合,因此")
    lines.append("(a) 聚合分数必然 ≥ 最大片段分数(`聚合−最大 ≥ 0` 恒成立);")
    lines.append("(b) `s6_all`(每单元一片)已经达到隔离下界,于是 S2 相对它的增量必然 ≈ 0。")
    lines.append("这两条都是**探针的性质**,不是「语义切分无效」的证据;换 docs/10 §5 P1 的真探针后必须重跑。")
    lines.append("")
    if metrics["goals_without_units"]:
        lines.append("## 6 · 未产出任何单元的目标")
        lines.append("")
        for gid in metrics["goals_without_units"][:40]:
            lines.append(f"- `{gid}`")
        lines.append("")
    return "\n".join(lines) + "\n"
