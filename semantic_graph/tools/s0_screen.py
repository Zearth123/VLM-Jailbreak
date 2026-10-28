#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S0 屏幕:把"用编码器向量建图能不能帮上忙"的前置条件算成一张表。

回答三个问题(全部只读、纯 stdlib,不需要任何模型):

  1. **有害身份落在哪儿**(`threshold.goal_type`):E 实体 / R 关系 / P 参数。
     只有非 E 类才**可能**靠切文本让每片无害 —— E 类的身份就是那个名字,切到哪片都带着它。
  2. **结构上限**:给定目标,是否存在一个 k 划分,使每片都"角色不完整"而并集"模式完整"。
     这是**判据能接受的天花板**,与编码器无关。
  3. **判据可审计性**:符号代理(`HARM_PATTERNS`)只对含 product/material 的目标有信息量;
     对 R 类目标它恒为"不完整"⇒ 结论空转 ⇒ 这类目标只能靠**读数**判定。

产出 `artifacts/s0_screen.md` 与终端摘要。`C`(候选子集)是后续所有工作的工作面:
    C = { 类型 ≠ E ∧ 节点数 ≥ 2 }

    python tools/s0_screen.py --graphs runs/structured68/graphs.jsonl --out artifacts/s0_screen.md
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter
from typing import Any, Iterator

HERE = pathlib.Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from sg import extract, threshold as TH, twosplit as T  # noqa: E402
from sg.schema import GoalGraph  # noqa: E402

MAX_EXACT_NODES = 10


def set_partitions(items: list[str], k: int) -> Iterator[list[int]]:
    """把 items 划成恰好 k 个非空块的全部标号(Stirling 枚举,限制增长串)。"""
    n = len(items)
    if k <= 0 or k > n:
        return
    lab = [0] * n

    def rec(i: int, mx: int) -> Iterator[list[int]]:
        if i == n:
            if mx + 1 == k:
                yield list(lab)
            return
        for label in range(min(mx + 1, k - 1) + 1):
            lab[i] = label
            yield from rec(i + 1, max(mx, label))

    yield from rec(0, -1)


def structural_ceiling(graph: GoalGraph, k: int, *, require_union: bool = True) -> dict[str, Any]:
    """是否存在 k 划分,使「每片角色不完整」(且 `require_union` 时并集模式完整)。

    `require_union=False` 是给 R 类目标用的:它们的并集永远不含 product/material,
    符号并集判据对它们**无信息量**,能查的只有"每片不完整"这条必要条件的上限。
    """
    ids = [n.nid for n in graph.nodes]
    if len(ids) < k or len(ids) > MAX_EXACT_NODES:
        return {"k": k, "exact": False, "feasible": False, "n_partitions": 0, "witness": None}
    n = 0
    for lab in set_partitions(ids, k):
        n += 1
        groups = [[ids[i] for i in range(len(ids)) if lab[i] == L] for L in range(k)]
        if not all(TH.pattern_incomplete(graph, grp) for grp in groups):
            continue
        if require_union and not TH.complete_patterns(
                graph, [x for grp in groups for x in grp]):
            continue
        return {"k": k, "exact": True, "feasible": True, "n_partitions": n,
                "witness": groups}
    return {"k": k, "exact": True, "feasible": False, "n_partitions": n, "witness": None}


def screen(graphs: list[GoalGraph]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for g in graphs:
        gt = TH.goal_type(g)
        pair = T.build_pair(g, T.recommend_groups(g))
        res = TH.evaluate(g, pair)
        cal = TH.calibration(g, pair)
        ceil2 = structural_ceiling(g, 2)
        ceil3 = structural_ceiling(g, 3)
        each2 = structural_ceiling(g, 2, require_union=False)
        all_ids = [n.nid for n in g.nodes]
        rows.append({
            "rid": g.behavior_id,
            "type": gt["type"],
            "n_nodes": len(g.nodes),
            "n_dep_edges": gt["dependency_edges"],
            "identifying": gt["identifying_surfaces"][:2],
            "default_each_incomplete": res["structural"]["each_subgraph_pattern_incomplete"],
            "default_union_complete": res["structural"]["union_pattern_complete"],
            "union_proxy_informative": bool(TH.complete_patterns(g, all_ids)),
            "gate": cal["gate"],
            "ceil_k2": ceil2["feasible"],
            "ceil_k3": ceil3["feasible"],
            "each_ceiling_k2": each2["feasible"],
            "n_partitions_k2": ceil2["n_partitions"],
            "candidate": gt["type"] != "E" and len(g.nodes) >= 2,
        })
    return {"rows": rows, "n": len(rows)}


def summarise(s: dict[str, Any]) -> dict[str, Any]:
    rows = s["rows"]
    types = Counter(r["type"] for r in rows)
    nodes = Counter(r["n_nodes"] for r in rows)
    cand = [r for r in rows if r["candidate"]]
    ceil2 = [r for r in rows if r["ceil_k2"]]
    ceil3 = [r for r in rows if r["ceil_k3"]]
    return {
        "n": s["n"],
        "types": dict(types),
        "nodes": dict(sorted(nodes.items())),
        "n_candidate": len(cand),
        "candidate_types": dict(Counter(r["type"] for r in cand)),
        "ceil_k2": len(ceil2),
        "ceil_k2_types": dict(Counter(r["type"] for r in ceil2)),
        "ceil_k3": len(ceil3),
        "default_each_incomplete": sum(1 for r in rows if r["default_each_incomplete"]),
        "default_both_symbolic": sum(1 for r in rows
                                     if r["default_each_incomplete"] and r["default_union_complete"]),
        "union_proxy_informative": sum(1 for r in rows if r["union_proxy_informative"]),
        "candidate_and_ceil_k2": sum(1 for r in cand if r["ceil_k2"]),
        "candidate_and_each_ceiling": sum(1 for r in cand if r["each_ceiling_k2"]),
    }


def render(s: dict[str, Any], summ: dict[str, Any], source: str) -> str:
    L: list[str] = []

    def w(line: str = "") -> None:
        L.append(line)
    w("# S0 屏幕:编码器向量建图的前置条件")
    w()
    w(f"- 输入:`{source}`({summ['n']} 条)")
    w("- 判据口径:**校准后** union 侧以读数为准,符号模式只作诊断(`sg/threshold.calibration`)")
    w()
    w("## 1 · 有害身份落在哪儿(`threshold.goal_type`)")
    w()
    w("| 类型 | 条数 | 含义 |")
    w("|---|---|---|")
    mean = {"E": "身份在实体名 → 切文本不可能让每片无害",
            "R": "身份在关系 → 切断因果链即可",
            "P": "身份在参数 → 把参数与动作分开即可"}
    for t in ("E", "R", "P"):
        w(f"| {t} | {summ['types'].get(t, 0)} | {mean[t]} |")
    w()
    w("## 2 · 结构上限(与编码器无关)")
    w()
    w("「存在一个 k 划分,使每片角色不完整而并集模式完整」:")
    w()
    w("| k | 有解条数 | 有解条目的类型分布 |")
    w("|---|---|---|")
    w(f"| 2 | {summ['ceil_k2']} / {summ['n']} | {summ['ceil_k2_types']} |")
    w(f"| 3 | {summ['ceil_k3']} / {summ['n']} | — |")
    w()
    w("## 3 · 判据可审计性")
    w()
    w("`HARM_PATTERNS` 的每个模式都要求 `product` 或 `material` 角色。")
    w("对不含这类角色的目标,符号判据**恒为不完整**⇒ 并集侧信息量为零 ⇒ 只能靠读数判定。")
    w()
    w(f"- 并集符号代理**有信息量**的目标:{summ['union_proxy_informative']} / {summ['n']}"
      f"(其余 {summ['n'] - summ['union_proxy_informative']} 条的并集侧只能由读数确认)")
    w(f"- 默认配对下 `each_subgraph_pattern_incomplete` 通过:{summ['default_each_incomplete']} / {summ['n']}")
    w(f"- 默认配对下符号双条件同时通过:{summ['default_both_symbolic']} / {summ['n']}")
    w()
    w("## 4 · 候选子集 `C`(后续唯一的工作面)")
    w()
    w(f"`C = {{类型 ≠ E ∧ 节点数 ≥ 2}}` → **|C| = {summ['n_candidate']}**"
      f"(类型分布 {summ['candidate_types']})")
    w()
    w("| 口径 | 有解 |")
    w("|---|---|")
    w(f"| 符号双条件(每片不完整 ∧ 并集完整) | {summ['candidate_and_ceil_k2']} / {summ['n_candidate']} |")
    w(f"| **仅必要条件**(每片不完整),给 R 类用 | **{summ['candidate_and_each_ceiling']} / {summ['n_candidate']}** |")
    w()
    w("两类目标用**不同**的口径,不能混算:")
    w()
    w("- **E 类**:身份在实体名 → 任何文本片都带着名字 ⇒ 本篇路线**结构上排除**;")
    w("- **R 类**:符号并集代理恒空转 ⇒ 唯一有意义的门是**读数**(护栏/判官);")
    w("  符号上只要求「每片角色不完整」,这一条对绝大多数划分都成立,不构成约束。")
    w()
    w("**判读规则**(写死的门,避免事后解释):")
    w()
    w("- `|C| ≥ 5` 且 `C` 中「每片不完整」有解 > 0 → 进入向量建图(A 层),但**结论完全依赖读数**;")
    w("- `|C| < 5` → 以负结论收尾。")
    w()
    w("## 5 · 逐条明细")
    w()
    w("| BehaviorID | 类型 | 节点 | 依赖边 | 识别性表面 | 默认各片不完整 | 符号并集完整 | 上限k2 | 候选 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for r in s["rows"]:
        w(f"| `{r['rid']}` | {r['type']} | {r['n_nodes']} | {r['n_dep_edges']} | "
          f"{', '.join(r['identifying']) or '—'} | "
          f"{'是' if r['default_each_incomplete'] else '否'} | "
          f"{'是' if r['default_union_complete'] else '否'} | "
          f"{'有解' if r['ceil_k2'] else '无解'} | "
          f"{'是' if r['candidate'] else ''} |")
    w()
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default="runs/structured68/graphs.jsonl")
    ap.add_argument("--out", default="artifacts/s0_screen.md")
    args = ap.parse_args(argv)

    gpath = pathlib.Path(args.graphs)
    if not gpath.exists():
        raise SystemExit(f"缺少 {gpath};先跑 `python -m sg.cli run --subset structured --limit 68`")
    rows = [json.loads(l) for l in gpath.read_text(encoding="utf-8").splitlines() if l.strip()]
    graphs = [extract.build_graph(r["behavior_id"], r["goal"]) for r in rows]
    s = screen(graphs)
    summ = summarise(s)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(s, summ, str(gpath)), encoding="utf-8", newline="\n")

    print(json.dumps(summ, ensure_ascii=False, indent=2))
    print(f"\nwrote {out}")
    if summ["n_candidate"] < 5:
        print("判读:|C| < 5 → 以负结论收尾,不进入 A 层")
    elif summ["candidate_and_each_ceiling"] == 0:
        print("判读:C 中连「每片不完整」都无解 → 停在 B 层")
    else:
        print(f"判读:|C| = {summ['n_candidate']} → 进入 A 层;"
              "注意 R 类的并集侧只能由读数确认,符号门对它无信息量")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
