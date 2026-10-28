"""M2 · 切分:triples → N fragments(docs/10 §4)。

路线:
  - **S2 约束优化分区(推荐)**:在依赖合并后的"超级节点"上**精确枚举** N 分区,
    目标 `minimize max fragment score`(等价于 docs/10 §4 S2 的 minimize max_i sensitivity),
    约束 ① 依赖边不跨区 ② 每区 unit 数上下界 ③ 每区不可文本化单元数 ≤ 图像带宽预算
  - **S3 过程阶段分层**:原料 / 操作 / 条件 / 产物 / 约束,`N` 由阶段数自然给出
  - **S6 零假设基线**:句子 / 连续 / 随机 / 不切 / 全切 —— 没有它们,`lift` 无法报告

所有方法共用同一不变量:**依赖边永不切断**;违例写进 `Fragment.deps_cut` 并由测试断言为空。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .probe import Probe
from .schema import Fragment, GoalGraph, Node, SplitResult

DEFAULT_IMAGE_BUDGET = 2  # 每片段允许的不可文本化单元上限(M4 带宽约束的代理,需扫描)
DEFAULT_CAP = 20000  # 分区枚举上限,防止组合爆炸
STAGE_ORDER = ("materials", "operation", "conditions", "product", "constraints")


@dataclass
class Partition:
    groups: list[list[str]]
    max_score: float
    mean_score: float
    fragments: list[Fragment]


# ---------------------------------------------------------------- 单元文本与依赖


def unit_text(node: Node) -> str:
    """单元的"可读表面文本" —— 探针与 M3 都看它。"""
    if node.ntype == "action":
        return node.label
    if node.ntype == "quantity":
        return f"quantity {node.label}"
    return f"{node.ntype} {node.label}"


def unit_texts(graph: GoalGraph) -> dict[str, str]:
    return {n.nid: unit_text(n) for n in graph.nodes}


def dependency_pairs(graph: GoalGraph) -> set[frozenset[str]]:
    """**必须同区**的依赖边对(`dependency and "co-location" in carriable_by`)。

    放宽与否由图本身承载(见 `GoalGraph.relaxed`),不在调用处另设开关 ——
    两处开关会互相抵消,这是实测踩过的坑。
    """
    return {frozenset((e.src, e.dst)) for e in graph.edges if e.requires_colocation}


def dependency_components(graph: GoalGraph) -> list[list[str]]:
    """把**必须同区**的依赖边连通的节点合成"超级节点" —— 切分的最小单位。"""
    parent: dict[str, str] = {n.nid: n.nid for n in graph.nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for e in graph.edges:
        if e.requires_colocation:
            union(e.src, e.dst)

    groups: dict[str, list[str]] = {}
    for n in graph.nodes:
        groups.setdefault(find(n.nid), []).append(n.nid)
    ordered = sorted(groups.values(), key=lambda g: min(_span_index(graph, x) for x in g))
    for g in ordered:
        g.sort(key=lambda x: _span_index(graph, x))
    return ordered


def _span_index(graph: GoalGraph, nid: str) -> int:
    return graph.node(nid).span[0]


def enforce_deps(graph: GoalGraph, groups: list[set[str]]) -> list[set[str]]:
    """合并所有被**必须同区**的依赖边横跨的组,直到不变量成立。"""
    deps = dependency_pairs(graph)
    changed = True
    while changed:
        changed = False
        for pair in deps:
            a, b = tuple(pair)
            ga = next((g for g in groups if a in g), None)
            gb = next((g for g in groups if b in g), None)
            if ga is None or gb is None or ga is gb:
                continue
            ga |= gb
            groups.remove(gb)
            changed = True
            break
    return groups


def deps_cut_in(graph: GoalGraph, groups: list[list[str]]) -> list[tuple[str, str]]:
    """**必须同区**的依赖边被切开的列表 —— 恒为空,非空即不变量违例。"""
    where = {nid: i for i, g in enumerate(groups) for nid in g}
    out: list[tuple[str, str]] = []
    for e in graph.edges:
        if e.requires_colocation and where.get(e.src) != where.get(e.dst):
            out.append((e.src, e.dst))
    return out


# 被切开的依赖边由谁承载(显式策略,别用"第一个可用"糊过去):
#   precursor-of:物料流 = 图上的箭头 → image
#   precedes    :先序 = 步骤编号/顺序词 + 胶合句 → ordering
CARRIER_POLICY = {"precursor-of": "image", "precedes": "ordering"}


def carrier_for(edge) -> str:
    """给被切开的依赖边定一个承载方式(按类型策略,不在候选里则退 glue → ordering)。"""
    preferred = CARRIER_POLICY.get(edge.etype, "glue")
    allowed = tuple(edge.carriable_by or ())
    if preferred in allowed:
        return preferred
    for c in ("image", "glue", "ordering"):
        if c in allowed:
            return c
    return "ordering"


def relaxed_cut_in(graph: GoalGraph, groups: list[list[str]]) -> list[tuple[str, str, str]]:
    """被切开、但**允许被重表述**的依赖边 → `(src, dst, 承载方式)`。"""
    where = {nid: i for i, g in enumerate(groups) for nid in g}
    out: list[tuple[str, str, str]] = []
    for e in graph.edges:
        if e.dependency and not e.requires_colocation and where.get(e.src) != where.get(e.dst):
            out.append((e.src, e.dst, carrier_for(e)))
    return out


# ---------------------------------------------------------------- 打分与片段构造


def build_fragments(graph: GoalGraph, groups: list[list[str]], probe: Probe) -> list[Fragment]:
    texts = unit_texts(graph)
    where = {nid: i for i, g in enumerate(groups) for nid in g}
    cut = deps_cut_in(graph, groups)
    relaxed = relaxed_cut_in(graph, groups)
    fragments: list[Fragment] = []
    for i, g in enumerate(groups):
        text = " ".join(texts[x] for x in g)
        my_cut = [c for c in cut if where.get(c[0]) == i or where.get(c[1]) == i]
        my_relaxed = [r for r in relaxed if where.get(r[0]) == i or where.get(r[1]) == i]
        fragments.append(
            Fragment(fid=f"f{i + 1}", node_ids=list(g), text=text,
                     score=probe.score_text(text), deps_cut=my_cut,
                     reexpressed_edges=my_relaxed)
        )
    return fragments


def _partition_key(groups: list[list[str]]) -> tuple:
    return tuple(sorted(tuple(sorted(g)) for g in groups))


def score_partition(graph: GoalGraph, groups: list[list[str]], probe: Probe) -> Partition:
    frags = build_fragments(graph, groups, probe)
    scores = [f.score for f in frags]
    return Partition(groups=groups, max_score=max(scores, default=0.0),
                     mean_score=sum(scores) / len(scores) if scores else 0.0, fragments=frags)


# ---------------------------------------------------------------- S2 精确枚举


def enumerate_partitions(
    supernodes: list[list[str]],
    k: int,
    image_budget: int | None = DEFAULT_IMAGE_BUDGET,
    n_min_units: int = 1,
    n_max_units: int | None = None,
    cap: int = DEFAULT_CAP,
    nonserializable: set[str] | None = None,
) -> tuple[list[list[int]], bool]:
    """把 m 个超级节点分成恰好 k 个非空组(Restricted growth strings)。

    返回 (分区列表, 是否因 cap 截断)。每个分区是长度为 m 的标签序列。
    """
    m = len(supernodes)
    if k <= 0 or m == 0 or k > m:
        return [], False
    sizes = [len(g) for g in supernodes]
    nser = nonserializable or set()
    nser_counts = [len([x for x in g if x in nser]) for g in supernodes]

    out: list[list[int]] = []
    truncated = False
    labels = [0] * m
    part_units = [0] * k
    part_nser = [0] * k

    def ok(i: int, lab: int) -> bool:
        if part_units[lab] + sizes[i] > (n_max_units or 10 ** 9):
            return False
        if image_budget is not None and part_nser[lab] + nser_counts[i] > image_budget:
            return False
        return True

    def rec(i: int, used: int) -> None:
        nonlocal truncated
        if truncated:
            return
        if i == m:
            if used != k:
                return
            if any(part_units[j] < n_min_units for j in range(k)):
                return
            out.append(list(labels))
            if len(out) >= cap:
                truncated = True
            return
        for lab in range(min(used, k - 1) + 1):
            if not ok(i, lab):
                continue
            labels[i] = lab
            part_units[lab] += sizes[i]
            part_nser[lab] += nser_counts[i]
            rec(i + 1, max(used, lab + 1))
            part_units[lab] -= sizes[i]
            part_nser[lab] -= nser_counts[i]

    rec(0, 0)
    return out, truncated


def split_s2(
    graph: GoalGraph,
    probe: Probe,
    n: int,
    image_budget: int | None = DEFAULT_IMAGE_BUDGET,
    cap: int = DEFAULT_CAP,
) -> SplitResult:
    """S2:精确枚举 + `minimize max fragment score`(并列时比 mean,再按分区字典序定序)。"""
    supernodes = dependency_components(graph)
    nser = {x.nid for x in graph.nodes if not x.serializable}
    labels_list, truncated = enumerate_partitions(
        supernodes, n, image_budget=image_budget, cap=cap, nonserializable=nser
    )
    if not labels_list:
        # 不可行:退化为"每个超级节点一个片段",让上层看到真实情况
        groups = [list(g) for g in supernodes]
        parts = score_partition(graph, groups, probe)
        aggregate_text = " | ".join(f.text for f in parts.fragments)
        return SplitResult(
            behavior_id=graph.behavior_id, method="s2", n=n, fragments=parts.fragments,
            aggregate_text=aggregate_text, aggregate_score=probe.score_text(aggregate_text),
            feasible_partitions=0, objective="min_max",
        )

    best: Partition | None = None
    best_key: tuple | None = None
    for labels in labels_list:
        groups: list[list[str]] = [[] for _ in range(n)]
        for i, lab in enumerate(labels):
            groups[lab].extend(supernodes[i])
        groups = [g for g in groups if g]
        part = score_partition(graph, groups, probe)
        key = (round(part.max_score, 9), round(part.mean_score, 9), _partition_key(groups))
        if best_key is None or key < best_key:
            best, best_key = part, key

    assert best is not None
    aggregate_text = " | ".join(f.text for f in best.fragments)
    return SplitResult(
        behavior_id=graph.behavior_id, method="s2", n=len(best.groups), fragments=best.fragments,
        aggregate_text=aggregate_text, aggregate_score=probe.score_text(aggregate_text),
        feasible_partitions=len(labels_list), objective="min_max", truncated=truncated,
    )


# ---------------------------------------------------------------- 放宽拆分(边重表述)


def split_relaxed(
    graph: GoalGraph,
    probe: Probe,
    k: int,
    relax_types: tuple[str, ...] = ("precursor-of", "precedes"),
    image_budget: int | None = DEFAULT_IMAGE_BUDGET,
    cap: int = DEFAULT_CAP,
) -> SplitResult:
    """放宽拆分:允许指定类型的依赖边被切开,条件是必须被 image/glue/ordering 重表述。

    与 S2 共用同一套精确枚举与 `minimize max 片段分数` 目标,唯一区别是超级节点按
    `require_colocation=False` 合并(即那些边不再强制同区)。
    """
    relaxed_graph = graph.relaxed(relax_types)
    supernodes = dependency_components(relaxed_graph)
    nser = {x.nid for x in relaxed_graph.nodes if not x.serializable}
    labels_list, truncated = enumerate_partitions(
        supernodes, k, image_budget=image_budget, cap=cap, nonserializable=nser
    )
    if not labels_list:
        groups = [list(g) for g in supernodes]
        parts = score_partition(relaxed_graph, groups, probe)
        aggregate_text = " | ".join(f.text for f in parts.fragments)
        return SplitResult(
            behavior_id=graph.behavior_id, method="relaxed", n=k, fragments=parts.fragments,
            aggregate_text=aggregate_text, aggregate_score=probe.score_text(aggregate_text),
            feasible_partitions=0, objective="min_max_relaxed", truncated=truncated,
        )

    best: Partition | None = None
    best_key: tuple | None = None
    for labels in labels_list:
        groups: list[list[str]] = [[] for _ in range(k)]
        for i, lab in enumerate(labels):
            groups[lab].extend(supernodes[i])
        groups = [g for g in groups if g]
        part = score_partition(relaxed_graph, groups, probe)
        key = (round(part.max_score, 9), round(part.mean_score, 9), _partition_key(groups))
        if best_key is None or key < best_key:
            best, best_key = part, key

    assert best is not None
    aggregate_text = " | ".join(f.text for f in best.fragments)
    return SplitResult(
        behavior_id=graph.behavior_id, method="relaxed", n=len(best.groups),
        fragments=best.fragments, aggregate_text=aggregate_text,
        aggregate_score=probe.score_text(aggregate_text),
        feasible_partitions=len(labels_list), objective="min_max_relaxed", truncated=truncated,
    )


def check_relaxation(graph: GoalGraph, subgraphs: list) -> list[str]:
    """不变量:每条**被切开的依赖边**必须出现在某个子图的 `reexpressed_edges` 里。

    返回问题清单(空 = 通过)。这是"放宽"取代"一律不许切"之后的新不变量。
    """
    problems: list[str] = []
    where: dict[str, str] = {}
    for sg in subgraphs:
        for nid in sg.node_ids:
            where[nid] = sg.sid
    reexpressed: set[tuple[str, str]] = set()
    for sg in subgraphs:
        for src, dst, _carrier in sg.reexpressed_edges:
            reexpressed.add((src, dst))
    for e in graph.edges:
        if not e.dependency:
            continue
        crossed = e.src in where and e.dst in where and where[e.src] != where[e.dst]
        if crossed:
            if e.requires_colocation:
                problems.append(f"必须同区的依赖边被切开: {e.src}->{e.dst}({e.etype})")
            elif (e.src, e.dst) not in reexpressed:
                problems.append(f"依赖边被切开但没有重表述记录: {e.src}->{e.dst}({e.etype})")
    return problems


# ---------------------------------------------------------------- S3 阶段分层


def stage_of(node: Node) -> str:
    if node.ntype == "quantity":
        return "materials"
    if node.ntype == "action":
        return "operation"
    if node.ntype == "condition":
        return "conditions"
    if node.ntype == "constraint":
        return "constraints"
    if node.attrs.get("class") == "precursor":
        return "materials"
    return "product"


def split_s3(graph: GoalGraph, probe: Probe) -> SplitResult:
    buckets: dict[str, set[str]] = {}
    for n in graph.nodes:
        buckets.setdefault(stage_of(n), set()).add(n.nid)
    groups = [buckets[s] for s in STAGE_ORDER if buckets.get(s)]
    groups = [g for g in enforce_deps(graph, groups) if g]
    groups = [sorted(g, key=lambda x: _span_index(graph, x)) for g in groups]
    groups = sorted(groups, key=lambda g: min(_span_index(graph, x) for x in g))
    part = score_partition(graph, groups, probe)
    aggregate_text = " | ".join(f.text for f in part.fragments)
    return SplitResult(
        behavior_id=graph.behavior_id, method="s3", n=len(part.groups), fragments=part.fragments,
        aggregate_text=aggregate_text, aggregate_score=probe.score_text(aggregate_text),
        feasible_partitions=1, objective="stage_layering",
    )


# ---------------------------------------------------------------- S6 零假设基线


def _clause_spans(graph: GoalGraph) -> list[tuple[int, int]]:
    if graph.core_span is None:
        return []
    base = graph.core_span[0]
    spans: list[tuple[int, int]] = []
    pos = 0
    for r in graph.roots:
        start = graph.core_text.find(r, pos)
        if start < 0:
            start = pos
        spans.append((base + start, base + start + len(r)))
        pos = start + len(r)
    return spans


def split_baseline(graph: GoalGraph, probe: Probe, method: str, n: int, seed: int = 20260101) -> SplitResult:
    """S6 基线。全部在**依赖闭合**的组上切,保证与 S2 同约束(否则比较不可比)。

    注意:`n` 是**请求的** N(报告按它归行),实际片段数由超级节点数钳制 —— 见 `n_fragments`。
    """
    supernodes = dependency_components(graph)
    m = len(supernodes)
    k = max(1, min(n, m))

    if method == "s6_none":
        groups = [set(x for g in supernodes for x in g)]
    elif method == "s6_all":
        groups = [set(g) for g in supernodes]
    elif method == "s6_contiguous":
        groups = [set() for _ in range(k)]
        for i, g in enumerate(supernodes):
            groups[min(i * k // m, k - 1)] |= set(g)
    elif method == "s6_random":
        rng = random.Random(seed + n)
        groups = [set() for _ in range(k)]
        for g in supernodes:
            groups[rng.randrange(k)] |= set(g)
    elif method == "s6_sentence":
        spans = _clause_spans(graph)
        groups = [set() for _ in range(max(1, len(spans)))]
        for g in supernodes:
            pos = _span_index(graph, g[0])
            idx = 0
            for i, (s, t) in enumerate(spans):
                if s <= pos < t:
                    idx = i
                    break
            groups[idx] |= set(g)
    else:
        raise ValueError(f"未知基线: {method}")

    groups = [g for g in enforce_deps(graph, [g for g in groups if g]) if g]
    groups = [sorted(g, key=lambda x: _span_index(graph, x)) for g in groups]
    groups = sorted(groups, key=lambda g: min(_span_index(graph, x) for x in g))
    part = score_partition(graph, groups, probe)
    aggregate_text = " | ".join(f.text for f in part.fragments)
    return SplitResult(
        behavior_id=graph.behavior_id, method=method, n=n, fragments=part.fragments,
        aggregate_text=aggregate_text, aggregate_score=probe.score_text(aggregate_text),
        feasible_partitions=1, objective="baseline", seed=seed,
    )


BASELINES = ("s6_none", "s6_all", "s6_sentence", "s6_contiguous", "s6_random")
# N 由结构决定(阶段数 / 超级节点数 / 子句数)的方法:只跑一次,报告里 N 记为 natural
N_FREE_METHODS = ("s3", "s6_all", "s6_sentence")
# 不切:其语义 N 就是 1
FIXED_N = {"s6_none": 1}
METHODS = ("s2", "s3") + BASELINES


def split(graph: GoalGraph, probe: Probe, method: str, n: int, seed: int = 20260101,
          image_budget: int | None = DEFAULT_IMAGE_BUDGET, cap: int = DEFAULT_CAP) -> SplitResult:
    if method == "s2":
        return split_s2(graph, probe, n, image_budget=image_budget, cap=cap)
    if method == "s3":
        return split_s3(graph, probe)
    if method in BASELINES:
        return split_baseline(graph, probe, method, n, seed=seed)
    raise ValueError(f"未知方法: {method}(可选 {', '.join(METHODS)})")


def sweep(
    graph: GoalGraph,
    probe: Probe,
    ns: tuple[int, ...] = (2, 3, 4, 5, 6),
    methods: tuple[str, ...] = ("s2", "s3") + BASELINES,
    seed: int = 20260101,
    image_budget: int | None = DEFAULT_IMAGE_BUDGET,
    cap: int = DEFAULT_CAP,
) -> list[SplitResult]:
    """`目标 × N × 方法` 的完整扫描 —— docs/10 §8.1 步骤 4。

    `s3` / `s6_none` / `s6_all` 的 N 不由参数决定(阶段数 / 1 / 超级节点数),只跑一次。
    """
    out: list[SplitResult] = []
    for method in methods:
        if method in N_FREE_METHODS:
            out.append(split(graph, probe, method, n=0, seed=seed,
                             image_budget=image_budget, cap=cap))
            continue
        if method in FIXED_N:
            out.append(split(graph, probe, method, n=FIXED_N[method], seed=seed,
                             image_budget=image_budget, cap=cap))
            continue
        for n in ns:
            out.append(split(graph, probe, method, n=n, seed=seed,
                             image_budget=image_budget, cap=cap))
    return out


# ---------------------------------------------------------------- 可拆性判定


def decomposability(
    graph: GoalGraph,
    probe: Probe,
    ns: tuple[int, ...] = (2, 3, 4, 5, 6),
    tau: float = 0.35,
    image_budget: int | None = DEFAULT_IMAGE_BUDGET,
    cap: int = DEFAULT_CAP,
    require_colocation: bool = True,
) -> dict:
    """docs/10 §8.1 步骤 3 的可计算定义:

    > 若**不存在**任何满足依赖约束的分区使所有片段 `sensitivity ≤ τ`,则标记 `non_decomposable`。

    在 S2 的精确枚举上判定,因此这是一个可复现的**定义**,不是启发式过滤。

    `require_colocation=False` 是**放宽档**:允许可重表述的依赖边被切开,于是可切单元数上升、
    `best_max_score` 通常下降。两档必须分开报告 —— 混在一起会让人以为"拆分本来就够"。
    """
    work = graph if require_colocation else graph.relaxed()
    supernodes = dependency_components(work)
    nser = {x.nid for x in work.nodes if not x.serializable}
    best_max = 1.0
    best_n: int | None = None
    total_feasible = 0
    truncated = False
    for n in ns:
        labels_list, trunc = enumerate_partitions(
            supernodes, n, image_budget=image_budget, cap=cap, nonserializable=nser
        )
        truncated = truncated or trunc
        total_feasible += len(labels_list)
        for labels in labels_list:
            groups: list[list[str]] = [[] for _ in range(n)]
            for i, lab in enumerate(labels):
                groups[lab].extend(supernodes[i])
            groups = [g for g in groups if g]
            part = score_partition(work, groups, probe)
            if part.max_score < best_max:
                best_max, best_n = part.max_score, n
    ok = best_n is not None and best_max <= tau
    return {
        "behavior_id": graph.behavior_id,
        "decomposable": ok,
        "relaxed": not require_colocation,
        "n_supernodes": len(supernodes),
        "best_max_score": round(best_max, 6),
        "best_n": best_n,
        "feasible_partitions": total_feasible,
        "truncated": truncated,
        "unsupported_reason": graph.unsupported_reason,
        "n_nodes": len(graph.nodes),
    }
