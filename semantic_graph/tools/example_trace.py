#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把一条 goal 的完整语义图建模过程逐阶段铺开,写成 markdown 追踪文件。

用途:给人读的"从原文本到完整语义图"的实例,也可作为回归基线。

    cd semantic_graph
    python tools/example_trace.py --rid safrole_oil_extraction_guide
    # → artifacts/example_safrole_oil_extraction_guide.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from sg import extract, sources  # noqa: E402
from sg import lexicon as L  # noqa: E402
from sg import metrics as M  # noqa: E402
from sg import split as S  # noqa: E402
from sg.probe import LexicalProbe  # noqa: E402
from sg.schema import SCHEMA_VER, write_text_lf  # noqa: E402

TAU = 0.35


def raw_csv_row(rid: str) -> str:
    """把 CSV 里的原始行(含引号)原样取出,展示数据格式这一步。"""
    text = sources.HARMBENCH.read_text(encoding="utf-8-sig")
    for line in text.splitlines():
        if line.rstrip().endswith("," + rid):
            return line
    return "(未找到原始行)"


def marker_matches(goal: str) -> list[tuple[int, int, str, str]]:
    out = []
    for kind, pat in L.PROCESS_MARKERS:
        m = pat.search(goal)
        if m:
            out.append((m.start(), m.end(), kind, m.group(0)))
    return sorted(out)


def explain_node(node, goal: str) -> str:
    surface = goal[node.span[0]:node.span[1]]
    if node.ntype == "action":
        for canon, pat, weight in L.ACTION_VERBS:
            if canon == node.label and pat.search(surface):
                return f"ACTION_VERBS[{canon}](权重 {weight}) 命中原文 {surface!r}"
    if node.ntype == "entity":
        for label, klass, pat in L.ENTITY_LEXICON:
            if label == node.label and pat.search(surface):
                return f"ENTITY_LEXICON[{label}](class={klass}) 命中原文 {surface!r}"
    if node.ntype == "constraint":
        return f"EVASION 正则命中原文 {surface!r}"
    if node.ntype == "condition":
        return f"CONDITION 正则命中原文 {surface!r}"
    if node.ntype == "quantity":
        return f"QUANTITY 正则命中原文 {surface!r}"
    if node.ntype == "artifact":
        cls = node.attrs.get("class")
        return f"{'STRUCTURE' if cls == 'structure' else 'CODE_ARTIFACT'} 正则命中原文 {surface!r}"
    return "(未知来源)"


def explain_edge(edge, graph) -> str:
    a, b = graph.node(edge.src), graph.node(edge.dst)
    table = {
        "patient": "动作之后最近的实体/结构 → patient",
        "instrument": "class=tool 的实体 → instrument",
        "condition-of": "quantity/condition 节点 → condition-of",
        "evades": "constraint(evasion)节点 → evades",
        "precursor-of": "前体实体出现在产物实体之前 → precursor-of(物料流)",
        "precedes": "同一子句内的动作按文本顺序串联 → precedes(先序)",
        "located-at": "位置类 condition → located-at",
        "agent": "施事 → agent",
    }
    return f"{table.get(edge.etype, '?')}  [{a.label} → {b.label}]"


def build_trace(rid: str, probe: LexicalProbe) -> str:
    record = next(r for r in sources.load_harmbench() if r.rid == rid)
    goal = extract.normalize(record.goal)
    graph = extract.build_graph(rid, goal)
    L_: list[str] = []

    def w(s: str = "") -> None:
        L_.append(s)

    w(f"# 语义图建模逐步追踪 · `{rid}`")
    w()
    w(f"> schema `{SCHEMA_VER}` · 抽取器 `{graph.extractor}` · 探针 `{probe.pid}` v{probe.version}(**占位**,不是 sensitivity)")
    w(">")
    w("> 本文由 `tools/example_trace.py` 从管线实际输出生成,所有 span 都可在原文上复核。")
    w()

    # ---- 0 原始数据
    w("## 0 · 原始数据(`data_origin/harmbench_behaviors_text_all.csv`)")
    w()
    w("```csv")
    w(raw_csv_row(rid))
    w("```")
    w()
    w(f"- `BehaviorID` = `{record.rid}`(join key)")
    w(f"- `FunctionalCategory` = `{record.functional_category}` · `SemanticCategory` = `{record.semantic_category}`")
    w(f"- `Tags` = `{record.tags!r}` · `ContextString` = `{record.context!r}`(空字符串,不是 NaN)")
    w()
    w("**规范化后**(`sources.GoalRecord`,JSONL 落盘格式):")
    w()
    w("```json")
    w(json.dumps(record.to_dict(), ensure_ascii=False, indent=2))
    w("```")
    w()

    # ---- 1 修辞判定
    w("## 1 · 修辞型判定(`extract.is_rhetorical`)")
    w()
    hits = [p.pattern for p in L.RHETORICAL if p.search(goal)]
    w(f"- 命中条数:**{len(hits)}**")
    w(f"- 结论:task_kind 不会被判成 `rhetorical`" if not hits else f"- 命中:{hits}")
    w()
    w("> 修辞型目标(纯文本生产,无过程结构)会在这里被排除并记 `unsupported_reason=rhetorical_text_production`。")
    w()

    # ---- 2 过程标记
    w("## 2 · 过程标记选择(`extract.find_marker`:取**最早出现**者)")
    w()
    w("| 起点 | 终点 | kind | 命中文本 |")
    w("|---|---|---|---|")
    for s, e, kind, text in marker_matches(goal):
        w(f"| {s} | {e} | `{kind}` | `{text}` |")
    w()
    marker = extract.find_marker(goal)
    if marker is None:
        w("- **无标记** → core 取整句")
    else:
        kind, m = marker
        w(f"- 胜出:`kind={kind}`,命中 `{m.group(0)}`,区间 ({m.start()}, {m.end()}) —— 起点最小者胜,同起点按 lexicon 顺序(更具体的在前)")
    w()

    # ---- 3 core 与子句
    w("## 3 · core 切分与子句划分(`cut_core` / `split_roots`)")
    w()
    w(f"- `core_text` = `{graph.core_text}`")
    w(f"- `core_span` = {graph.core_span} → 原文回查:`{goal[graph.core_span[0]:graph.core_span[1]]}`")
    w(f"- `roots`(子句森林,{len(graph.roots)} 个):")
    for i, r in enumerate(graph.roots):
        w(f"  - root{i}: `{r}`")
    w()
    w("> 指代型 core(`do this` 之类)会回退到整句的第一个子句;`core_text` 是**被请求的危害活动**,不是表面主谓宾。")
    w()

    # ---- 4 节点
    w(f"## 4 · 节点抽取(`build_graph` 的 ⑥ 步):{len(graph.nodes)} 个节点")
    w()
    w("| nid | 类型 | 标签 | span | 原文片段 | 属性 | serializable | 来源规则 |")
    w("|---|---|---|---|---|---|---|---|")
    for n in graph.nodes:
        surface = goal[n.span[0]:n.span[1]]
        w(f"| `{n.nid}` | {n.ntype} | `{n.label}` | ({n.span[0]}, {n.span[1]}) | `{surface}` | `{json.dumps(n.attrs, ensure_ascii=False)}` | {n.serializable} | {explain_node(n, goal)} |")
    w()
    w("> `serializable` 是**类型驱动初判**:`quantity` 或 `attrs.structural` → False,其余 True([docs/10 §9 R10](../../docs/10-semantic-graph-modeling.md) 归属未定案)。")
    w()

    # ---- 5 边
    w(f"## 5 · 边构建:{len(graph.edges)} 条边,其中**依赖边 {len(graph.dependency_edges)} 条**")
    w()
    w("| 源 | 边类型 | 目标 | 依赖边 | 生成规则 |")
    w("|---|---|---|---|---|")
    for e in graph.edges:
        w(f"| `{e.src}`({graph.node(e.src).label}) | **{e.etype}** | `{e.dst}`({graph.node(e.dst).label}) | {'✅ 是' if e.dependency else '否'} | {explain_edge(e, graph)} |")
    w()
    w(f"- `unsupported_reason` = `{graph.unsupported_reason}`")
    w(f"- 不变量自检 `check_invariants` = `{extract.check_invariants(graph)}`")
    w()

    # ---- 6 依赖合并
    sup = S.dependency_components(graph)
    w("## 6 · 依赖合并 → 超级节点(`split.dependency_components`)")
    w()
    w("> 依赖边(`precedes` / `precursor-of`)连通的节点必须同区 —— 这是 M2 的硬约束 ①。")
    w()
    for i, comp in enumerate(sup):
        labels = " + ".join(graph.node(x).label for x in comp)
        w(f"- supernode{i}: `{'`, `'.join(comp)}` = {labels}")
    w()
    w(f"- 可切单元数 = **{len(sup)}** → `N > {len(sup)}` 一定不可行")
    w()

    # ---- 7 S2 扫描
    w("## 7 · S2 约束优化分区(精确枚举 + `minimize max 片段分数`)")
    w()
    for n in (2, 3, 4):
        res = S.split_s2(graph, probe, n, image_budget=S.DEFAULT_IMAGE_BUDGET)
        w(f"### N = {n} · 可行分区数 = {res.feasible_partitions}{'(枚举被 cap 截断)' if res.truncated else ''}")
        w()
        if res.feasible_partitions == 0:
            w(f"> ⚠️ **不可行** —— 请求 N={n} 超过可切单元数({len(sup)}),或图像带宽预算卡住。")
            w("> 下表是**退化结果**(一超级节点一片),不能当成「满足 N 的分区」。")
            w()
        w("| 片段 | 单元 | 文本 | 分数 | deps_cut |")
        w("|---|---|---|---|---|")
        for f in res.fragments:
            w(f"| `{f.fid}` | `{'`, `'.join(f.node_ids)}` | `{f.text}` | {f.score:.4f} | {f.deps_cut or '—'} |")
        w()
        w(f"- `max_fragment_score` = {res.max_fragment_score:.4f} · `mean` = {res.mean_fragment_score:.4f}")
        w(f"- **P4 聚合分数**(片段裸拼接后再过探针)= {res.aggregate_score:.4f} → 聚合−最大 = {res.aggregate_score - res.max_fragment_score:+.4f}")
        w()

    # ---- 8 探针
    w("## 8 · 探针逐单元打分(占位探针的词袋贡献透明化)")
    w()
    w("| nid | 单元文本 | 标记贡献 | w | score = 1−exp(−w/4) |")
    w("|---|---|---|---|---|")
    texts = S.unit_texts(graph)
    for n in graph.nodes:
        t = texts[n.nid]
        mk = probe.markers(t)
        w(f"| `{n.nid}` | `{t}` | `{json.dumps(mk, ensure_ascii=False)}` | {sum(mk.values()):.3f} | {probe.score_text(t):.4f} |")
    w()
    floor = max(probe.score_text(texts[n.nid]) for n in graph.nodes)
    w(f"- **隔离下界** = max(单元分数) = **{floor:.4f}** → 任何分区的 `max_fragment_score` 都不低于它")
    w(f"- τ = {TAU} → 该目标{'**可拆**' if floor <= TAU else '**不可拆**(最热单元本身就超过 τ,M2 单独过不了门)'}")
    w()

    # ---- 9 可拆性与 lift
    dec = S.decomposability(graph, probe, ns=(2, 3, 4), tau=TAU)
    w("## 9 · τ 门禁的可计算判定 + 基线对照")
    w()
    w("```json")
    w(json.dumps(dec, ensure_ascii=False, indent=2))
    w("```")
    w()
    w("| 方法 | 请求 N | 实际片段数 | 最大片段分数 | lift vs 不切 | lift vs 最强基线 | deps_cut |")
    w("|---|---|---|---|---|---|---|")
    splits = S.sweep(graph, probe, ns=(2, 3, 4),
                     methods=("s2", "s3", "s6_none", "s6_all", "s6_sentence", "s6_contiguous", "s6_random"))
    best_base = min(s.max_fragment_score for s in splits if s.method.startswith("s6_") and s.method != "s6_none")
    none_max = next(s.max_fragment_score for s in splits if s.method == "s6_none")
    seen: dict[tuple[str, object], object] = {}
    for s in splits:
        seen[M.split_key(s.method, s.n)] = s
    for (method, n), s in sorted(seen.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        w(f"| `{method}` | {n} | {len(s.fragments)} | {s.max_fragment_score:.4f} | "
          f"{none_max - s.max_fragment_score:+.4f} | {best_base - s.max_fragment_score:+.4f} | {len(s.deps_cut)} |")
    w()

    # ---- 10 完整图
    w("## 10 · 完整语义图(落盘格式)")
    w()
    w("```json")
    w(json.dumps(graph.to_dict(), ensure_ascii=False, indent=2))
    w("```")
    w()
    w("---")
    w()
    w("- 相关设计:[docs/10-semantic-graph-modeling.md](../../docs/10-semantic-graph-modeling.md) · [docs/04-modules.md](../../docs/04-modules.md) M1/M2")
    w("- 相关实现:[../README.md](../README.md)")
    w()
    return "\n".join(L_)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成一条 goal 的逐步语义图追踪")
    ap.add_argument("--rid", default="safrole_oil_extraction_guide")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    out = Path(args.out) if args.out else HERE / "artifacts" / f"example_{args.rid}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = build_trace(args.rid, LexicalProbe())
    write_text_lf(out, text)
    print(json.dumps({"rid": args.rid, "out": str(out), "bytes": len(text.encode("utf-8"))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
