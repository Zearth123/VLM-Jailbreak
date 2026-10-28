"""用文本编码器的 token/片段向量建图,并在图上切出「逐片无害」的候选划分。

定位(docs/12)
--------------
  · 它只是**划分提议器**:替代手写的按类型策略;判据仍归 `sg/threshold.py`(护栏/判官读数);
  · 编码器必须**非对齐、冻结、只前向**(base LM / CLIP 文本塔),不得用 RLHF 对话模型
    —— 否则违反 docs/08 §2.1 的原意(被同一类安全训练卡住 + 不可复现);
  · 本模块**只用 numpy**,不加载任何模型;向量由 `tools/embed_probe.py` 在 GPU 主机产出。

dump 契约(与 `tools/cloud_probe.py` 同一风格)
---------------------------------------------
    dumps/<run>/emb.jsonl   {"text_index":0,"rid":..., "text":..., "n_tokens":N, "sha256":...}
    dumps/<run>/emb.npz     vectors (T,d) f32 | offsets (T,2) i32 | text_id (T,) i32 | token_ids (T,) i32
    dumps/<run>/meta.json   {model_id, revision, layer, pooling, dtype}

缺 dump / 缺 numpy 时**显式抛 `EmbedUnavailable`**,绝不返回占位向量。

一个必须写明的限度
------------------
切分的原子单位仍是 `Node`(它带 span,渲染要逐字回原文)。因此"片段图"在实际输出上退化为
**节点两两亲和矩阵 + 链边**:词级片段只在对齐到节点之前参与相似度聚合。若将来要支持
"切在节点内部",必须同时改 `Node.span` 语义与渲染纪律 —— 不在本模块范围内。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import project as P
from . import split as S
from . import threshold as TH
from .schema import GoalGraph, Subgraph, write_text_lf

EMB_SCHEMA = "eg-0.1"
DEFAULT_TAU = 0.35
DEFAULT_KNN = 3


class EmbedUnavailable(RuntimeError):
    """向量 dump 或 numpy 不可用。宁可失败,也不用占位向量假装跑过。"""


def _require_np():
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - 环境相关
        raise EmbedUnavailable(f"需要 numpy: {exc}") from exc
    return np


# ---------------------------------------------------------------- dump


def load_embed_dump(dump_dir: str | Path) -> dict[str, Any]:
    """读向量 dump;缺文件/格式不对 → `EmbedUnavailable`。"""
    np = _require_np()
    d = Path(dump_dir)
    jl, nz, meta = d / "emb.jsonl", d / "emb.npz", d / "meta.json"
    for p in (jl, nz):
        if not p.exists():
            raise EmbedUnavailable(
                f"缺少 {p};先在 GPU 主机跑 `python tools/embed_probe.py ...`")
    texts = [json.loads(l) for l in jl.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not texts:
        raise EmbedUnavailable(f"{jl} 为空")
    z = np.load(nz)
    for key in ("vectors", "offsets", "text_id"):
        if key not in z:
            raise EmbedUnavailable(f"{nz} 缺少键 {key}")
    vectors = z["vectors"]
    if vectors.ndim != 2 or vectors.size == 0:
        raise EmbedUnavailable(f"{nz} 的 vectors 形状异常: {getattr(vectors, 'shape', None)}")
    return {"texts": texts, "vectors": vectors, "offsets": z["offsets"],
            "text_id": z["text_id"], "token_ids": z.get("token_ids"),
            "meta": json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}}


def pick_text(dump: dict[str, Any], rid: str | None = None) -> tuple[int, dict[str, Any]]:
    """选一条文本:给 rid 就精确匹配;只有一条时直接用;否则要求 rid。"""
    texts = dump["texts"]
    if rid is not None:
        for i, t in enumerate(texts):
            if t.get("rid") == rid:
                return i, t
        raise EmbedUnavailable(f"dump 里没有 rid={rid}(有:{[t.get('rid') for t in texts][:5]})")
    if len(texts) == 1:
        return 0, texts[0]
    raise EmbedUnavailable(f"dump 里有 {len(texts)} 条文本,必须给 --rid")


# ---------------------------------------------------------------- 片段


@dataclass
class Fragment:
    fid: int
    nid: str | None
    span: tuple[int, int]
    text: str
    vec: Any

    def to_dict(self) -> dict[str, Any]:
        return {"fid": self.fid, "nid": self.nid, "span": list(self.span),
                "text": self.text}


def fragments(graph: GoalGraph, dump: dict[str, Any], text_index: int) -> list[Fragment]:
    """把 token 合并成**词级片段**(偏移有间隙就断开),向量取均值。

    返回的 `text` 必须逐字等于 `graph.goal[span]` —— 与 `project.py` 的溯源纪律一致。
    """
    np = _require_np()
    tid = dump["text_id"]
    sel = [i for i in range(len(tid)) if int(tid[i]) == text_index]
    if not sel:
        raise EmbedUnavailable(f"text_index={text_index} 没有对应 token")
    offs = dump["offsets"]
    vecs = dump["vectors"]

    out: list[Fragment] = []
    cur: list[int] = []

    def flush() -> None:
        if not cur:
            return
        s = int(min(offs[i][0] for i in cur))
        e = int(max(offs[i][1] for i in cur))
        if e <= s:
            cur.clear()
            return
        v = np.asarray(vecs[cur], dtype="float32").mean(axis=0)
        out.append(Fragment(fid=len(out), nid=None, span=(s, e),
                            text=graph.goal[s:e], vec=v))
        cur.clear()

    prev_end = -1
    for i in sel:
        a, b = int(offs[i][0]), int(offs[i][1])
        if b <= a:            # 零宽 token(special)
            continue
        if prev_end >= 0 and a > prev_end:   # 有空隙 ⇒ 词边界
            flush()
        cur.append(i)
        prev_end = max(prev_end, b)
    flush()
    for f in out:
        f.nid = _anchor(graph, f.span)
    return out


def _anchor(graph: GoalGraph, span: tuple[int, int]) -> str | None:
    """把片段对齐到语义单元:取**重叠最多**的节点。"""
    best, best_ov = None, 0
    for n in graph.nodes:
        lo = max(span[0], n.span[0])
        hi = min(span[1], n.span[1])
        ov = hi - lo
        if ov > best_ov:
            best, best_ov = n.nid, ov
    return best


# ---------------------------------------------------------------- 图


@dataclass
class VecGraph:
    frags: list[Fragment]
    edges: list[tuple[int, int, float, str]] = field(default_factory=list)  # (i,j,w,kind)
    affinity: dict[frozenset[str], float] = field(default_factory=dict)
    n_unmapped: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"n_fragments": len(self.frags), "n_edges": len(self.edges),
                "n_unmapped": self.n_unmapped,
                "n_node_pairs": len(self.affinity),
                "edges": [{"i": i, "j": j, "w": round(w, 4), "kind": k}
                          for i, j, w, k in self.edges]}


def _cos(a, b) -> float:
    import math
    na = float((a * a).sum()) ** 0.5
    nb = float((b * b).sum()) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return float((a * b).sum()) / (na * nb)


def build(graph: GoalGraph, frags: list[Fragment], *, tau: float = DEFAULT_TAU,
          knn: int = DEFAULT_KNN) -> VecGraph:
    """建图:**链边**(相邻片段,权重 1.0)+ **kNN 相似边**(cos ≥ tau)。

    节点两两亲和取**片段对上的最大 cos**(有任一强连接就算强连接)。
    """
    g = VecGraph(frags=frags)
    g.n_unmapped = sum(1 for f in frags if f.nid is None)
    n = len(frags)
    for i in range(n - 1):
        g.edges.append((i, i + 1, 1.0, "chain"))
    sims: list[list[tuple[float, int]]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if frags[i].nid is not None and frags[i].nid == frags[j].nid:
                continue                      # 同一节点内部不参与节点亲和
            c = _cos(frags[i].vec, frags[j].vec)
            sims[i].append((c, j))
            if c >= tau:
                g.edges.append((i, j, c, "knn"))
    for i in range(n):
        for c, j in sorted(sims[i], reverse=True)[:knn]:
            if c >= tau and not any((min(i, j), max(i, j)) == (min(a, b), max(a, b))
                                    for a, b, _, k in g.edges if k == "knn"):
                g.edges.append((i, j, c, "knn"))
    for i in range(n):
        for j in range(i + 1, n):
            ni, nj = frags[i].nid, frags[j].nid
            if ni is None or nj is None or ni == nj:
                continue
            key = frozenset((ni, nj))
            c = _cos(frags[i].vec, frags[j].vec)
            if c > g.affinity.get(key, -math.inf):
                g.affinity[key] = c
    return g


# ---------------------------------------------------------------- 切分


@dataclass
class CutResult:
    groups: list[list[str]]
    cut_weight: float
    n_partitions: int
    truncated: bool = False
    constraint_violated: bool = False
    best_violating: list[list[str]] | None = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.groups) and not self.constraint_violated

    def to_dict(self) -> dict[str, Any]:
        return {"groups": self.groups, "cut_weight": round(self.cut_weight, 4),
                "n_partitions": self.n_partitions, "truncated": self.truncated,
                "constraint_violated": self.constraint_violated,
                "reason": self.reason}


def _cut_weight(groups: Sequence[Sequence[str]], affinity: dict[frozenset[str], float]) -> float:
    where = {nid: gi for gi, grp in enumerate(groups) for nid in grp}
    return sum(w for pair, w in affinity.items()
               if len(pair) == 2 and len({where.get(x) for x in pair}) > 1)


def cut(graph: GoalGraph, vg: VecGraph, *, k: int = 2, cap: int = S.DEFAULT_CAP) -> CutResult:
    """在**依赖超级节点**上枚举 k 划分,最小化被切边权重和,并要求每片角色不完整。

    超级节点来自 `split.dependency_components` ⇒ 必须同区的依赖边自动不会被切开。
    """
    supers = S.dependency_components(graph)
    if len(supers) < k:
        return CutResult(groups=[], cut_weight=0.0, n_partitions=0,
                         reason=f"超级节点只有 {len(supers)} 个,不足 k={k}")
    labels_list, truncated = S.enumerate_partitions(supers, k, image_budget=None, cap=cap)
    if not labels_list:
        return CutResult(groups=[], cut_weight=0.0, n_partitions=0, truncated=truncated,
                         reason="没有可行的 k 划分")
    best, best_w = None, math.inf
    best_any, best_any_w = None, math.inf
    for labels in labels_list:
        groups: list[list[str]] = [[] for _ in range(k)]
        for i, lab in enumerate(labels):
            groups[lab].extend(supers[i])
        w = _cut_weight(groups, vg.affinity)
        if w < best_any_w:
            best_any, best_any_w = groups, w
        if not all(TH.pattern_incomplete(graph, grp) for grp in groups):
            continue
        if w < best_w:
            best, best_w = groups, w
    if best is None:
        return CutResult(groups=[], cut_weight=best_any_w, n_partitions=len(labels_list),
                         truncated=truncated, constraint_violated=True,
                         best_violating=best_any,
                         reason="所有划分都有子图自己凑齐了有害模式")
    return CutResult(groups=best, cut_weight=best_w, n_partitions=len(labels_list),
                     truncated=truncated)


def cut_cost(graph: GoalGraph, vg: VecGraph) -> dict[tuple[str, str], float]:
    """B 层:给每条依赖边一个「可切代价」= 1 − 节点亲和(越低越可切)。

    替代 `split.CARRIER_POLICY` 里按类型手写的策略;被切边仍需 `check_relaxation` 登记。
    """
    out: dict[tuple[str, str], float] = {}
    for e in graph.dependency_edges:
        a = vg.affinity.get(frozenset((e.src, e.dst)))
        out[(e.src, e.dst)] = 1.0 if a is None else round(1.0 - a, 6)
    return out


# ---------------------------------------------------------------- 输出


def to_subgraphs(graph: GoalGraph, groups: Sequence[Sequence[str]],
                 prefix: str = "SG-") -> list[Subgraph]:
    """把划分投影成 `Subgraph`(全 text 承载),供 `threshold.evaluate` 原封不动消费。"""
    out: list[Subgraph] = []
    order = sorted(range(len(groups)),
                   key=lambda i: min(graph.node(n).span[0] for n in groups[i]))
    for idx, gi in enumerate(order):
        out.append(P.project_subgraph(graph, list(groups[gi]), sid=f"{prefix}{chr(65 + idx)}",
                                      index=idx))
    return out


def render(graph: GoalGraph, vg: VecGraph, cut_res: CutResult,
           subgraphs: Sequence[Subgraph] | None = None, *, rid: str = "",
           costs: dict[tuple[str, str], float] | None = None, created: str = "") -> str:
    L: list[str] = []

    def w(line: str = "") -> None:
        L.append(line)

    w(f"# 向量建图与切分 · `{rid or graph.behavior_id}`")
    w()
    w(f"- 生成时间:{created or '?'}")
    w("- 方法:[docs/12](../../docs/12-vector-graph-splitting.md);判据仍由 `sg/threshold.py` 给")
    w(f"- 片段 {len(vg.frags)}(未对齐 {vg.n_unmapped})· 图边 {len(vg.edges)} · 节点对 {len(vg.affinity)}")
    w()
    w("## 1 · 片段与对齐(逐段可回原文)")
    w()
    w("| fid | 节点 | span | 片段 |")
    w("|---|---|---|---|")
    for f in vg.frags:
        w(f"| {f.fid} | `{f.nid or '—'}` | ({f.span[0]},{f.span[1]}) | `{f.text}` |")
    w()
    w("## 2 · 节点亲和(片段对上的最大 cos)")
    w()
    w("| 节点对 | 亲和 | 依赖边? | 可切代价 |")
    w("|---|---|---|---|")
    dep = {frozenset((e.src, e.dst)) for e in graph.dependency_edges}
    for pair, wv in sorted(vg.affinity.items(), key=lambda kv: -kv[1]):
        a, b = sorted(pair)
        c = (costs or {}).get((a, b), (costs or {}).get((b, a)))
        w(f"| `{a}`–`{b}` | {wv:.4f} | {'是' if pair in dep else '否'} | "
          f"{'—' if c is None else f'{c:.4f}'} |")
    w()
    w("## 3 · 切分结果")
    w()
    w(f"- 枚举到 {cut_res.n_partitions} 个划分(截断:{cut_res.truncated})")
    w(f"- 最优被切权重和:**{cut_res.cut_weight:.4f}**")
    if cut_res.constraint_violated:
        w(f"- **约束未满足**:{cut_res.reason};以下仅作诊断")
        if cut_res.best_violating:
            for gi, grp in enumerate(cut_res.best_violating):
                w(f"  - 诊断组 {gi}:{grp}")
    for gi, grp in enumerate(cut_res.groups):
        w(f"  - 组 {gi}:{grp}")
    w()
    if subgraphs:
        w("## 4 · 投影文本(判据的输入)")
        w()
        for sg in subgraphs:
            w(f"### `{sg.sid}`")
            w()
            w("```text")
            w(sg.rendered_text or "(空)")
            w("```")
            w()
        res = TH.evaluate(graph, list(subgraphs))
        w("## 5 · 判据(读数为 None ⇒ 全部 untested)")
        w()
        w("| 条目 | 角色 | 完整模式 | 判定 |")
        w("|---|---|---|---|")
        for r in res["rows"]:
            w(f"| `{r['item_id']}` | {', '.join(r['roles'])} | "
              f"{', '.join(r['complete_patterns']) or '—'} | {r['gate']} |")
        w()
        w(f"- 总判定:**{res['gate']}**(结构:各片不完整="
          f"{res['structural']['each_subgraph_pattern_incomplete']},"
          f"并集完整={res['structural']['union_pattern_complete']})")
        w()
    w("## 6 · 本模块不做什么")
    w()
    w("1. **不判害**:`recognized`/`guard` 必须由模型给;这里只出结构判定;")
    w("2. **不切节点内部**:原子单位仍是 `Node`,片段图退化为节点亲和矩阵;")
    w("3. **不训练**:向量只前向取,来自非对齐编码器;")
    w("4. 缺 dump 时**不产出任何结论**。")
    w()
    return "\n".join(L)


def run(dump_dir: str | Path, out: str | Path, *, rid: str | None = None, k: int = 2,
        tau: float = DEFAULT_TAU, knn: int = DEFAULT_KNN, cap: int = S.DEFAULT_CAP,
        created: str = "") -> dict[str, Any]:
    """一条 goal:dump → 片段 → 图 → 切分 → 投影 → 报告。"""
    from . import extract, sources

    dump = load_embed_dump(dump_dir)
    idx, row = pick_text(dump, rid)
    rid = rid or row.get("rid")
    rec = next((r for r in sources.load_harmbench() if r.rid == rid), None)
    if rec is None:
        raise EmbedUnavailable(f"HarmBench 里没有 {rid}")
    graph = extract.build_graph(rid, rec.goal)
    frags = fragments(graph, dump, idx)
    vg = build(graph, frags, tau=tau, knn=knn)
    cut_res = cut(graph, vg, k=k, cap=cap)
    subgraphs = to_subgraphs(graph, cut_res.groups) if cut_res.groups else []
    costs = cut_cost(graph, vg)
    text = render(graph, vg, cut_res, subgraphs, rid=rid, costs=costs, created=created)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(out, text)
    return {"out": str(out), "rid": rid, "k": k, "tau": tau,
            "n_fragments": len(vg.frags), "n_unmapped": vg.n_unmapped,
            "cut": cut_res.to_dict(), "costs": costs,
            "n_subgraphs": len(subgraphs),
            "gate": TH.evaluate(graph, subgraphs)["gate"] if subgraphs else None,
            "affinity": {f"{sorted(p)[0]}|{sorted(p)[1]}": round(v, 4)
                         for p, v in vg.affinity.items()}}
