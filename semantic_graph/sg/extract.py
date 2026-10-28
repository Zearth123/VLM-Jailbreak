"""M1 · 语义分解:goal → 类型化属性图(docs/10 的 A5 + R2 路线)。

规则实现,零模型、零依赖、完全确定:
  1. 先判修辞型目标(无过程结构 → 不建图,显式标记,docs/10 §8.1 步骤 3)
  2. 按"最早出现的过程标记"切出 core(危害活动所在段),多根目标按子句分开
  3. 在文本上抽动作 / 实体 / 结构 / 参数 / 约束 / 条件,建 R2 的节点与边
  4. 依赖边(`precedes` / `precursor-of`)标记为 dependency=True → M2 不得切断

**已知边界(必须与 docs/10 §3 A5 一起读)**:
  - 只做模板化句面的头部覆盖;非模板化长句会退化为 `no_action_detected`
  - 动作/实体匹配是句面的,不做指代消解 —— 这是 A2(SRL)要补的尾
  - `serializable` 的类型驱动初判归属尚未定案(docs/10 §9 R10)
"""

from __future__ import annotations

import re

from . import lexicon as L
from .schema import DEPENDENCY_EDGE_TYPES, Edge, GoalGraph, GoalRecord, Node

_NODE_ORDER = {"action": 0, "artifact": 1, "entity": 2, "quantity": 3, "condition": 4, "constraint": 5}


def normalize(text: str) -> str:
    """折叠空白。不改大小写、不做改写 —— 保留原文 span 的可回溯性。"""
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------- 任务类型与 core


def is_rhetorical(goal: str) -> bool:
    return any(p.search(goal) for p in L.RHETORICAL)


def find_marker(goal: str) -> tuple[str, re.Match[str]] | None:
    """取**最早出现**的过程标记;同起点按 lexicon 列表顺序(更具体的在前)。"""
    best: tuple[int, int, str, re.Match[str]] | None = None
    for idx, (kind, pat) in enumerate(L.PROCESS_MARKERS):
        m = pat.search(goal)
        if m is None:
            continue
        key = (m.start(), idx)
        if best is None or key < (best[0], best[1]):
            best = (m.start(), idx, kind, m)
    if best is None:
        return None
    return best[2], best[3]


def split_roots(text: str) -> list[str]:
    """按连接词切子句(禁止按裸逗号切,docs/10 §10.1)。"""
    parts = [p.strip(" ,;.") for p in L.CLAUSE_SPLIT.split(text)]
    return [p for p in parts if p]


def cut_core(goal: str, match: re.Match[str], kind: str) -> tuple[str, tuple[int, int], list[str]]:
    """core = 过程标记之后的文本;同时返回该段的**子句划分**(多根目标用)。

    指向代词(`do this`)的 core 无内容 → 回退到第一个子句,但 roots 保留全部子句,
    否则多根目标会被误判成单根(docs/10 §1.3 事实 3)。
    """
    post = goal[match.end():].strip(" ,;:.")
    if not post or L.ANAPHORIC_CORE.search(post):
        # core 是 "do this" 这类指代 → 回退到**整句**的第一个子句,否则会切出一段空内容
        roots = split_roots(goal) or [goal]
        core = roots[0]
    else:
        roots = split_roots(post) or [post]
        core = post
    start = goal.find(core)
    if start < 0:
        start = match.end()
    core = core[: L.MAX_CORE_CHARS]
    span = (start, start + len(core))
    # 去掉 "how to " 这类残留前缀
    strip = re.match(r"^(?:how to|how and where to|successfully|quietly|illegally)\s+", core, re.IGNORECASE)
    if strip:
        core = core[strip.end():]
        span = (span[0] + strip.end(), span[1])
    return core.strip(), span, roots


# ---------------------------------------------------------------- 节点抽取


def _add(nodes: list[Node], ntype: str, label: str, span: tuple[int, int], attrs: dict) -> Node:
    nid = f"n{len(nodes) + 1:03d}"
    node = Node(nid=nid, ntype=ntype, label=label, span=span, attrs=attrs,
                serializable=default_serializable(ntype, attrs))
    nodes.append(node)
    return node


def default_serializable(ntype: str, attrs: dict) -> bool:
    """类型驱动初判(docs/10 §2 R2 / §7)。

    不可文本化的候选 = 结构 / 参数;实体名称、动作、条件、约束本身都能被文本完整承载。
    """
    if ntype == "quantity":
        return False
    if attrs.get("structural"):
        return False
    return True


def extract_actions(text: str, offset: int) -> list[tuple[str, tuple[int, int], float]]:
    out: list[tuple[str, tuple[int, int], float]] = []
    for canon, pat, weight in L.ACTION_VERBS:
        for m in pat.finditer(text):
            out.append((canon, (offset + m.start(), offset + m.end()), weight))
    out.sort(key=lambda x: x[1][0])
    return out


def build_manual_graph(behavior_id: str, goal: str,
                       units: list[tuple[str, str, tuple[int, int]]],
                       edges: list[tuple[int, int, str, bool]] | None = None,
                       task_kind: str = "manual") -> GoalGraph:
    """**手工声明**的语义图(P1 式构造,docs/07)。

    规则抽取只覆盖模板化句面(疑问句、修辞型目标会退化成单节点或空图),
    这时由人给定语义单元与边,再走**完全相同**的投影/切分/判据管线。

    `units`:`[(label, ntype, (start, end))]`,按顺序编号 `n001…`;
    span 必须落在 `goal` 内且非空白 —— 这是"逐字回原文"纪律的入口校验。
    `edges`:`[(i, j, etype, dependency)]`,`i`/`j` 是 units 的**下标**(从 1 开始)。
    """
    nodes: list[Node] = []
    for i, (label, ntype, span) in enumerate(units, start=1):
        s, e = int(span[0]), int(span[1])
        if not (0 <= s < e <= len(goal)):
            raise ValueError(f"unit {i} 的 span 越界: {(s, e)} / goal 长 {len(goal)}")
        if not goal[s:e].strip():
            raise ValueError(f"unit {i} 的 span 是空白: {(s, e)}")
        nodes.append(Node(nid=f"n{i:03d}", ntype=ntype, label=label, span=(s, e)))
    if not nodes:
        raise ValueError("manual graph 至少需要 1 个单元")
    out_edges: list[Edge] = []
    for i, j, etype, dep in (edges or []):
        if not (1 <= i <= len(nodes) and 1 <= j <= len(nodes)):
            raise ValueError(f"边端点越界: ({i}, {j}),只有 {len(nodes)} 个单元")
        out_edges.append(Edge(src=f"n{i:03d}", dst=f"n{j:03d}", etype=etype,
                              dependency=bool(dep)))
    first = nodes[0].span
    return GoalGraph(behavior_id=behavior_id, goal=goal, task_kind=task_kind,
                     core_span=first, core_text=goal[first[0]:first[1]],
                     roots=[n.nid for n in nodes], nodes=nodes, edges=out_edges,
                     extractor="manual")


def build_graph(behavior_id: str, goal: str, extractor: str = "a5-template") -> GoalGraph:
    goal = normalize(goal)

    if is_rhetorical(goal):
        return GoalGraph(
            behavior_id=behavior_id,
            goal=goal,
            task_kind="rhetorical",
            core_span=None,
            core_text="",
            roots=[],
            unsupported_reason="rhetorical_text_production",
            extractor=extractor,
        )

    marker = find_marker(goal)
    if marker is None:
        kind = "unknown"
        core, span = goal, (0, len(goal))
        roots = split_roots(goal) or [goal]
    else:
        kind, match = marker
        core, span, roots = cut_core(goal, match, kind)

    nodes: list[Node] = []
    edges: list[Edge] = []

    # --- 实体 / 结构 / 参数 / 约束 / 条件:在**整句**上抽(框架里的 artifact 也在其中)
    entities: list[Node] = []
    for label, klass, pat in L.ENTITY_LEXICON:
        for m in pat.finditer(goal):
            dup = [e for e in entities if e.label == label and e.span == (m.start(), m.end())]
            if dup:
                continue
            entities.append(
                _add(nodes, "entity", label, (m.start(), m.end()), {"class": klass})
            )

    structures: list[Node] = []
    for m in L.STRUCTURE.finditer(goal):
        structures.append(
            _add(nodes, "artifact", m.group(0).lower(), (m.start(), m.end()),
                 {"class": "structure", "structural": True})
        )

    if kind in ("code", "list"):
        for m in L.CODE_ARTIFACT.finditer(goal):
            _add(nodes, "artifact", m.group(0).lower(), (m.start(), m.end()), {"class": "code"})

    quantities: list[Node] = []
    for m in L.QUANTITY.finditer(core):
        quantities.append(
            _add(nodes, "quantity", m.group(0).strip(), (span[0] + m.start(), span[0] + m.end()),
                 {"source": "core"})
        )
    for m in L.QUANTITY.finditer(goal):
        if any(q.span == (m.start(), m.end()) for q in quantities):
            continue
        quantities.append(
            _add(nodes, "quantity", m.group(0).strip(), (m.start(), m.end()), {"source": "goal"})
        )

    constraints: list[Node] = []
    for m in L.EVASION.finditer(goal):
        constraints.append(
            _add(nodes, "constraint", m.group(0).strip().lower(), (m.start(), m.end()),
                 {"class": "evasion"})
        )

    conditions: list[Node] = []
    for m in L.CONDITION.finditer(goal):
        conditions.append(
            _add(nodes, "condition", m.group(0).strip().lower(), (m.start(), m.end()), {"class": "condition"})
        )

    # --- 动作:先 core,core 里没有再用整句;仍没有且存在 artifact → 记一条 produce 兜底
    actions = extract_actions(core, span[0])
    action_source = "core"
    if not actions:
        actions = extract_actions(goal, 0)
        action_source = "goal"
    if not actions:
        artifact = next((n for n in nodes if n.ntype == "artifact"), None)
        if artifact is not None:
            actions = [("produce", artifact.span, 2.0)]
            action_source = "fallback_produce"
    if not actions:
        return GoalGraph(
            behavior_id=behavior_id,
            goal=goal,
            task_kind=kind,
            core_span=span,
            core_text=core,
            roots=roots,
            nodes=nodes,
            edges=edges,
            unsupported_reason="no_action_detected",
            extractor=extractor,
        )

    # 动作节点:同一规范动作在同一子句中只建一个(取首次出现)
    action_nodes: list[Node] = []
    seen: set[tuple[str, int]] = set()
    for canon, aspan, weight in actions:
        root_idx = _root_index(roots, core, aspan[0] - span[0])
        if (canon, root_idx) in seen:
            continue
        seen.add((canon, root_idx))
        action_nodes.append(
            _add(nodes, "action", canon, aspan, {"weight": weight, "root": root_idx, "source": action_source})
        )

    primary = action_nodes[0]
    # patient:动作之后最近的实体/结构;没有则取全图第一个实体
    patient = next((e for e in entities if e.span[0] >= primary.span[0]), entities[0] if entities else None)
    if patient is not None:
        edges.append(Edge(primary.nid, patient.nid, "patient"))
    for s in structures:
        if s.nid != (patient.nid if patient else None):
            edges.append(Edge(primary.nid, s.nid, "patient"))
    for e in entities:
        if e is patient:
            continue
        if e.attrs.get("class") == "tool":
            edges.append(Edge(primary.nid, e.nid, "instrument"))
    for q in quantities:
        edges.append(Edge(primary.nid, q.nid, "condition-of", span=q.span))
    for c in conditions:
        edges.append(Edge(primary.nid, c.nid, "condition-of", span=c.span))
    for c in constraints:
        edges.append(Edge(primary.nid, c.nid, "evades", span=c.span))

    # precursor-of:前体类实体出现在产物类实体之前 → 物料流(依赖边)
    precs = [e for e in entities if e.attrs.get("class") == "precursor"]
    prods = [e for e in entities if e.attrs.get("class") in ("substance", "agent")]
    for p in precs:
        later = [q for q in prods if q.span[0] > p.span[0]]
        if later:
            edges.append(Edge(p.nid, later[0].nid, "precursor-of", dependency=True))

    # precedes:同一子句内的动作按文本顺序串联(依赖边)
    by_root: dict[int, list[Node]] = {}
    for a in action_nodes:
        by_root.setdefault(a.attrs.get("root", 0), []).append(a)
    for group in by_root.values():
        group.sort(key=lambda n: n.span[0])
        for a, b in zip(group, group[1:]):
            edges.append(Edge(a.nid, b.nid, "precedes", dependency=True))

    reason = None
    if len(nodes) < 2:
        reason = "insufficient_units"
    return GoalGraph(
        behavior_id=behavior_id,
        goal=goal,
        task_kind=kind,
        core_span=span,
        core_text=core,
        roots=roots,
        nodes=nodes,
        edges=edges,
        unsupported_reason=reason,
        extractor=extractor,
    )


def _root_index(roots: list[str], core: str, rel_pos: int) -> int:
    """把 core 内偏移映射到子句序号。"""
    pos = 0
    for i, r in enumerate(roots):
        start = core.find(r, pos)
        if start < 0:
            continue
        if start <= rel_pos <= start + len(r):
            return i
        pos = start + len(r)
    return 0


def extract_goal(record: GoalRecord) -> GoalGraph:
    return build_graph(record.rid, record.goal)


def extract_all(records: list[GoalRecord]) -> list[GoalGraph]:
    return [extract_goal(r) for r in records]


def check_invariants(graph: GoalGraph) -> list[str]:
    """结构性自检:节点/边引用完整、依赖边类型合法、span 在原文内。"""
    problems: list[str] = []
    ids = {n.nid for n in graph.nodes}
    if len(ids) != len(graph.nodes):
        problems.append("节点 id 重复")
    for e in graph.edges:
        if e.src not in ids or e.dst not in ids:
            problems.append(f"边引用不存在的节点: {e.src}->{e.dst}")
        if e.etype not in ("agent", "patient", "instrument", "precursor-of", "condition-of",
                           "precedes", "located-at", "evades"):
            problems.append(f"未知边类型: {e.etype}")
        if e.dependency and e.etype not in DEPENDENCY_EDGE_TYPES:
            problems.append(f"非依赖边被标记为依赖: {e.etype}")
    for n in graph.nodes:
        s, t = n.span
        if not (0 <= s < t <= len(graph.goal)):
            problems.append(f"span 越界: {n.nid} {n.span} / len={len(graph.goal)}")
    return problems
