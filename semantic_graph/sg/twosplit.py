"""双子图拆分:把一张语义图切成两个子图,并逐条判"有没有有害意图"。

**范围(本模块的纪律)**:只做**图建模 + 子图拆分**。

  - 子图文本 = **原文 span 的拼接**(见 `project.py`),不弱化、不造词、不加框架;
  - 不做胶合句、不做图文组装(M6)、不做语境重构(M3)—— 那些步骤会引入新文字,不在本步;
  - 判官看到的因此是**原文片段本身**:它判有害就是有害,不靠措辞把判官哄过去。

四件事:
  1. `recommend_groups`  按"产物 + 生成它的那一步"切出 SG-B,其余为 SG-A(纯图规则)
  2. `build_pair`        两个子图各自投影;产物类单元按通道走 `image`
  3. `evaluate_pair`     逐子图判官 + 聚合器对照 + 结构完整性 + 放宽不变量 + 溯源自检
  4. `sweep_relaxed`     附录:放开依赖后 k 分区的全候选表
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import project as P
from . import split as S
from .judge import Captioner, Judge, is_authoritative, make_judge
from .schema import SCHEMA_VER, Subgraph, write_text_lf


def recommend_groups(graph) -> tuple[list[str], list[str]]:
    """按语义切:产物 + 生成它的那一步 → SG-B;其余 → SG-A。

    规则只用图的角色信息,不依赖 nid 编号,因此对任意目标可复现。
    """
    products = [n for n in graph.nodes if P.role_of(n) == "product"]
    actions = [n for n in graph.nodes if n.ntype == "action"]
    if not products or not actions:
        ordered = sorted(graph.nodes, key=lambda n: n.span[0])
        half = max(1, len(ordered) // 2)
        a = [n.nid for n in ordered[:half]]
        b = [n.nid for n in ordered[half:]] or [ordered[-1].nid]
        return a, b
    last_product = max(products, key=lambda n: n.span[1])
    prior = [a for a in actions if a.span[0] < last_product.span[0]]
    step = max(prior, key=lambda a: a.span[0]) if prior else actions[-1]
    b_ids = [n.nid for n in graph.nodes if n.nid == step.nid or P.role_of(n) == "product"]
    a_ids = [n.nid for n in graph.nodes if n.nid not in b_ids]
    return a_ids, b_ids


def default_image_units(graph, node_ids: list[str]) -> set[str]:
    """哪些单元应当由**图像**承载:产物 / 参数 / 结构类。

    这是"通道分配"的一次声明式选择(而不是文字加工):这些单元的内容留在像素上,
    它们的原文片段因此**不进**子图文本。
    """
    out: set[str] = set()
    for nid in node_ids:
        node = graph.node(nid)
        if P.role_of(node) == "product" or node.ntype == "quantity" or node.attrs.get("structural"):
            out.add(nid)
    return out


def build_pair(
    graph,
    groups: tuple[list[str], list[str]],
    image_units_b: set[str] | None = None,
) -> list[Subgraph]:
    """两个子图:A 全走文本;B 的产物类单元走图像(其余走文本)。"""
    a_ids, b_ids = groups
    img_b = default_image_units(graph, b_ids) if image_units_b is None else set(image_units_b)
    carrier_a = {nid: P.CARRIER_TEXT for nid in a_ids}
    carrier_b = {nid: (P.CARRIER_IMAGE if nid in img_b else P.CARRIER_TEXT) for nid in b_ids}
    return [
        P.project_subgraph(graph, a_ids, sid="SG-A", carrier_of=carrier_a, index=0),
        P.project_subgraph(graph, b_ids, sid="SG-B", carrier_of=carrier_b, index=1),
    ]


def _relaxed_edges(graph, subgraphs: list[Subgraph], relax_types: tuple[str, ...]) -> None:
    """把"被切开且允许重表述"的依赖边登记到承接它的子图上(就地写入)。

    判定基于**放宽后的图**,否则严格图里这些边仍要求同区,登记会静默失败(实测踩过)。
    """
    work = graph.relaxed(relax_types)
    where = {nid: sg.sid for sg in subgraphs for nid in sg.node_ids}
    for e in work.edges:
        if e.dependency and not e.requires_colocation and where.get(e.src) != where.get(e.dst):
            carrier = S.carrier_for(e)
            for sg in subgraphs:
                if where.get(e.src) == sg.sid or where.get(e.dst) == sg.sid:
                    if (e.src, e.dst, carrier) not in sg.reexpressed_edges:
                        sg.reexpressed_edges.append((e.src, e.dst, carrier))


def evaluate_pair(
    graph,
    subgraphs: list[Subgraph],
    judge: Judge,
    llm: Judge | None = None,
    captioner: Captioner | None = None,
    relax_types: tuple[str, ...] = ("precursor-of", "precedes"),
) -> dict[str, Any]:
    """逐子图判定 + 聚合器对照 + 结构完整性 + 不变量 + 溯源。"""
    _relaxed_edges(graph, subgraphs, relax_types)

    rows: list[dict[str, Any]] = []
    for sg in subgraphs:
        lex = judge.judge_text(sg.rendered_text) if sg.rendered_text else {
            "harmful": None, "score": 0.0, "label": "empty", "raw": "该子图无文本承载单元"}
        row: dict[str, Any] = {
            "sid": sg.sid,
            "channel": sg.channel,
            "n_units": len(sg.node_ids),
            "text": sg.rendered_text,
            "image_payload": sg.visual_spec,
            "judge_lexical": lex,
            "judge_llm": None,
            "caption": {"status": "pending", "reason": "本步不生成图像(M5b 未做)"},
            "reexpressed_edges": [list(x) for x in sg.reexpressed_edges],
        }
        if llm is not None and sg.rendered_text:
            row["judge_llm"] = llm.judge_text(sg.rendered_text)
        if captioner is not None and captioner.available and sg.visual_spec:
            row["caption"] = {"status": "pending", "reason": "尚未渲染图像(M5b 未做)"}
        sg.judge = {"lexical": lex, "llm": row["judge_llm"], "caption": row["caption"]}
        rows.append(row)

    aggregate_text = " ‖ ".join(sg.rendered_text for sg in subgraphs if sg.rendered_text)
    aggregate = {
        "text": aggregate_text,
        "judge_lexical": judge.judge_text(aggregate_text) if aggregate_text else None,
        "judge_llm": llm.judge_text(aggregate_text) if (llm is not None and aggregate_text) else None,
    }

    invariants = S.check_relaxation(graph.relaxed(relax_types), subgraphs)
    return {
        "rows": rows,
        "aggregate": aggregate,
        "fidelity": structural_completeness(graph, subgraphs),
        "traceability": P.traceability(graph, subgraphs),
        "invariants": invariants,
        "authoritative_judge": is_authoritative(llm) if llm is not None else False,
    }


def structural_completeness(graph, subgraphs: list[Subgraph]) -> dict[str, Any]:
    """规则化完整性代理:两个子图是否覆盖了全部单元与还原所需角色。

    **这不是"可还原性"的证明**,只防退化解(例如把内容整片丢掉):
    每一段投影文本都能回原文(由 `traceability` 保证),这里再看单元与角色是否齐。
    """
    covered = {nid for sg in subgraphs for nid in sg.node_ids}
    roles = {P.role_of(n) for n in graph.nodes if n.nid in covered}
    image_units = [r["nid"] for sg in subgraphs for r in sg.rewrite
                   if r["channel"] == P.CARRIER_IMAGE]
    return {
        "n_units": len(graph.nodes),
        "n_covered": len(covered),
        "missing_units": [n.nid for n in graph.nodes if n.nid not in covered],
        "roles_present": sorted(roles),
        "missing_roles": [r for r in ("material", "action", "product") if r not in roles],
        "image_units": image_units,
        "cover_rate": round(len(covered) / len(graph.nodes), 6) if graph.nodes else 0.0,
        "note": "规则化代理,不等于可还原性;真正的保真度需权威判官/人工",
    }


def sweep_relaxed(
    graph,
    judge: Judge,
    k: int = 2,
    relax_types: tuple[str, ...] = ("precursor-of", "precedes"),
    limit: int = 20,
) -> list[dict[str, Any]]:
    """放开依赖后的 k 分区全候选表:每个候选各自投影后过判官。**不做任何文字加工**。"""
    work = graph.relaxed(relax_types)
    supernodes = S.dependency_components(work)
    labels_list, truncated = S.enumerate_partitions(supernodes, k, image_budget=None)
    rows: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for labels in labels_list:
        groups: list[list[str]] = [[] for _ in range(k)]
        for i, lab in enumerate(labels):
            groups[lab].extend(supernodes[i])
        key = tuple(sorted(tuple(sorted(g)) for g in groups))
        if key in seen:
            continue
        seen.add(key)
        texts = []
        for gi, g in enumerate(groups):
            sg = P.project_subgraph(graph, g, sid=f"c{gi + 1}", index=gi)
            texts.append(sg.rendered_text)
        verdicts = [judge.judge_text(t) if t else {"harmful": None, "score": 0.0, "label": "empty"}
                    for t in texts]
        rows.append({
            "groups": ["+".join(graph.node(x).label for x in g) for g in groups],
            "texts": texts,
            "harmful_parts": sum(1 for v in verdicts if v["harmful"]),
            "max_score": round(max((v["score"] for v in verdicts), default=0.0), 6),
            "truncated": truncated,
        })
    rows.sort(key=lambda r: (r["harmful_parts"], r["max_score"]))
    return rows[:limit]


def render_artifact(rid: str, graph, subgraphs: list[Subgraph], evaluation: dict[str, Any],
                    sweep: list[dict[str, Any]], meta: dict[str, Any],
                    baseline: dict[str, Any] | None = None) -> str:
    L: list[str] = []

    def w(s: str = "") -> None:
        L.append(s)

    auth = evaluation.get("authoritative_judge", False)
    jd = meta.get("judge_desc", {})
    w(f"# 双子图拆分 · `{rid}`")
    w()
    w(f"- 生成时间:{meta.get('created', '?')} · schema `{meta.get('schema_ver', '')}`")
    w(f"- 判官:`{jd.get('judge_id')}` v{jd.get('version')}(kind={jd.get('kind')}, "
      f"authoritative={jd.get('authoritative')}, sha={jd.get('sha')})")
    w("- **范围**:只做图建模 + 子图拆分。子图文本 = **原文 span 的拼接**,"
      "不弱化、不造词、不加任务框架;胶合句与图文组装(M6)、语境重构(M3)都不在本步。")
    tr = evaluation.get("traceability", {})
    w(f"- 溯源自检:每段都能回原文 = **{tr.get('all_traceable')}**;新增字符 = "
      f"{tr.get('added_chars') or '无'}")
    w()
    if not auth:
        w("> ⚠️ 当前判官是**词法档(非权威)**。它判的就是**原文片段本身** ——")
        w("> 判有害即说明这些片段带词面痕迹,不存在「用措辞哄判官」的成分。")
        w()

    w("## 1 · 结论摘要")
    w()
    w("| 子图 | 通道 | 单元数 | 词法判官 | LLM 判官 | caption(L2) |")
    w("|---|---|---|---|---|---|")
    for r in evaluation["rows"]:
        lex = r["judge_lexical"]
        llm = r.get("judge_llm")
        w(f"| `{r['sid']}` | {r['channel']} | {r['n_units']} | "
          f"{lex['label']} | {llm['label'] if llm else 'pending'} | {r['caption']['status']} |")
    agg = evaluation["aggregate"]
    if agg.get("judge_lexical"):
        w(f"| **聚合器**(两子图裸拼接) | — | — | {agg['judge_lexical']['label']} | "
          f"{agg['judge_llm']['label'] if agg.get('judge_llm') else 'pending'} | — |")
    w()
    fid = evaluation["fidelity"]
    w(f"- 完整性代理 `cover_rate` = {fid['cover_rate']}(缺角色:{fid['missing_roles'] or '无'};"
      f"图像承载单元:{fid['image_units'] or '无'})")
    w(f"- 放宽不变量自检:{evaluation['invariants'] or '通过(每条被切依赖边都有重表述记录)'}")
    if baseline:
        w(f"- 基线对照(整条 goal 不切,直接判):`{baseline['judge_lexical']['label']}`"
          f"({baseline['judge_lexical']['raw']})")
    w()

    w("## 2 · 两个子图")
    w()
    for r in evaluation["rows"]:
        w(f"### `{r['sid']}`({r['channel']})")
        w()
        w("**投影文本**(原文片段,判官看的就是它):")
        w()
        w("```text")
        w(r["text"] or "(该子图无文本承载单元)")
        w("```")
        w()
        if r["image_payload"]:
            w(f"**图像承载清单(`image_payload`,只含原文片段与 span)**:`{r['image_payload']}`")
            w()
        if r["reexpressed_edges"]:
            w("**被切开但已重表述的依赖边**:")
            w()
            w("| 源 | 目标 | 承载方式 |")
            w("|---|---|---|")
            for src, dst, carrier in r["reexpressed_edges"]:
                w(f"| `{src}`({graph.node(src).label}) | `{dst}`({graph.node(dst).label}) | {carrier} |")
            w()
        w("**投影审计**(单元 → 通道 → 原文片段;无「改写后」字段 —— 本步不做文字加工):")
        w()
        w("| nid | 类型 | 角色 | 通道 | 原文片段 | span | 标签 |")
        w("|---|---|---|---|---|---|---|")
        for sg in subgraphs:
            if sg.sid != r["sid"]:
                continue
            for e in sg.rewrite:
                w(f"| `{e['nid']}` | {e['ntype']} | {e['role']} | {e['channel']} | "
                  f"`{e['surface']}` | ({e['span'][0]}, {e['span'][1]}) | `{e['label']}` |")
        w()

    w("## 3 · 聚合器对照(docs/08 §2.3)")
    w()
    w("```text")
    w(agg.get("text", ""))
    w("```")
    w()
    if agg.get("judge_lexical"):
        w(f"- 词法判官:{agg['judge_lexical']['label']}({agg['judge_lexical']['raw']})")
    if agg.get("judge_llm"):
        w(f"- LLM 判官:{agg['judge_llm']['label']}")
    w()
    w("> 这一项**不要求通过**:它是对「防御方不做跨片段聚合」这一前提的显式检验。")
    w("> 拼接用的是原文片段,拼接本身不产生新字符。")
    w()

    if sweep:
        w("## 4 · 附录:放开依赖后的 k 分区候选(逐候选投影 + 判官)")
        w()
        w(f"共 {len(sweep)} 条(前 {meta.get('sweep_limit', 20)} 条):")
        w()
        w("| 候选分组 | 有害片数 | 最大词法分 | 各片投影文本 |")
        w("|---|---|---|---|")
        for r in sweep:
            w(f"| {' ‖ '.join(r['groups'])} | {r['harmful_parts']} | {r['max_score']:.3f} | "
              + " ‖ ".join(f"`{t or '(空)'}`" for t in r["texts"]) + " |")
        w()

    w("## 5 · 本步不做什么(范围声明)")
    w()
    w("1. **不弱化文字、不换词、不加框架** —— 子图文本里任何一个字符都必须能按 span 回到原文;")
    w("2. **不做语境重构(M3)** —— 那属于 P3,是独立模块,会引入新文字,必须在它能被单独度量之后再做;")
    w("3. **不做组装与胶合(M6)**:没有胶合句,所以本产物不是可直接提交的输入;")
    w("4. **图像侧只到 `image_payload` 清单** —— 哪些单元的哪段原文要进图;渲染与 caption 检查未做;")
    w("5. **判官档可换但必须显式**:权威判官缺失时,本报告的 `harmless` 不作数。")
    w()
    w("---")
    w()
    w("- 方法:[docs/10-semantic-graph-modeling.md](../../docs/10-semantic-graph-modeling.md) §4(放宽拆分)、"
      "§4.12(潜变量形式化)")
    w("- 实现:[../README.md](../README.md)")
    w()
    return "\n".join(L)


def run_split2(
    rid: str,
    out,
    judge_name: str = "lexical",
    judge_kwargs: dict[str, Any] | None = None,
    use_llm: bool = True,
    k: int = 2,
    relax_types: tuple[str, ...] = ("precursor-of", "precedes"),
    do_sweep: bool = True,
    sweep_limit: int = 20,
    created: str = "",
) -> dict[str, Any]:
    """跑一次双子图拆分,产出 markdown + `subgraphs.jsonl` + `judge.jsonl`。"""
    from . import extract, sources

    record = next((r for r in sources.load_harmbench() if r.rid == rid), None)
    if record is None:
        raise ValueError(f"HarmBench 里没有 {rid}")
    graph = extract.build_graph(rid, record.goal)
    if len(graph.nodes) < 2:
        raise ValueError(f"{rid} 的图只有 {len(graph.nodes)} 个单元,无法切两个子图")

    judge: Judge = make_judge(judge_name, **(judge_kwargs or {}))
    llm: Judge | None = judge if (use_llm and judge_name.lower() in ("http", "llm", "http-llm")) else None

    groups = recommend_groups(graph)
    subgraphs = build_pair(graph, groups)
    evaluation = evaluate_pair(graph, subgraphs, judge, llm=llm, relax_types=relax_types)
    sweep = sweep_relaxed(graph, judge, k=k, relax_types=relax_types, limit=sweep_limit) if do_sweep else []
    baseline = {"judge_lexical": judge.judge_text(graph.goal),
                "judge_llm": llm.judge_text(graph.goal) if llm else None}

    meta = {
        "rid": rid,
        "created": created,
        "schema_ver": SCHEMA_VER,
        "judge_desc": judge.describe(),
        "k": k,
        "relax_types": list(relax_types),
        "groups": groups,
        "sweep_limit": sweep_limit,
        "scope": "graph modeling + subgraph splitting only (projection, no rewriting)",
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(out, render_artifact(rid, graph, subgraphs, evaluation, sweep, meta, baseline))
    sub_path = out.with_name(out.stem + ".subgraphs.jsonl")
    judge_path = out.with_name(out.stem + ".judge.jsonl")
    with open(sub_path, "w", encoding="utf-8", newline="\n") as f:
        for sg in subgraphs:
            f.write(json.dumps(sg.to_dict(), ensure_ascii=False) + "\n")
    with open(judge_path, "w", encoding="utf-8", newline="\n") as f:
        for r in evaluation["rows"]:
            f.write(json.dumps({"sid": r["sid"], **r["judge_lexical"]}, ensure_ascii=False) + "\n")
        if evaluation["aggregate"].get("judge_lexical"):
            f.write(json.dumps({"sid": "AGGREGATE", **evaluation["aggregate"]["judge_lexical"]},
                               ensure_ascii=False) + "\n")
        f.write(json.dumps({"sid": "WHOLE_GOAL", **baseline["judge_lexical"]},
                           ensure_ascii=False) + "\n")
    return {"out": str(out), "subgraphs_jsonl": str(sub_path), "judge_jsonl": str(judge_path),
            "meta": meta, "evaluation": evaluation, "sweep_n": len(sweep)}
