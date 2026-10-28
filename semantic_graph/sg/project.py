"""子图投影:子图文本 = **原文 span 的拼接**。

范围纪律(本模块的唯一职责)
--------------------------
第一步只做**图建模 + 子图拆分**。所以本模块**不做**任何文字层面的加工:

  - ❌ 不做委婉化 / 去词面(那是 M3 语境重构,docs/04 M3,属于 P3)
  - ❌ 不写任务框架("实验记录""课程作业"之类都是新增语义)
  - ❌ 不造句、不解释、不加类别断言(例如把 MDMA 说成"胺类产物")
  - ✅ 只做投影:哪个单元进哪个子图、由哪个通道承载、以及**把该单元的原文片段取出来**

由此得到一条可机械检查的性质:

> `rendered_text` 里的每一段都必须能在 `graph.goal` 里按 span 找到。

`visual_spec` 是给渲染器的**结构化清单**(只含原文片段与 span),不是散文,也不是新内容。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .schema import GoalGraph, Node, Subgraph

SEP = " · "  # 片段分隔符:非字母字符,便于逐段回原文核对

# 承载通道。`deferred` = 本步不承载,记在账上等 M6 组装决定(M6 不在本步范围内)
CARRIER_TEXT = "text"
CARRIER_IMAGE = "image"
CARRIER_DEFERRED = "deferred"
CARRIERS = (CARRIER_TEXT, CARRIER_IMAGE, CARRIER_DEFERRED)


@dataclass
class ProjectedUnit:
    """一个单元的投影记录(可审计,无任何新增文字)。"""

    nid: str
    ntype: str
    role: str
    label: str
    span: tuple[int, int]
    surface: str
    channel: str
    sid: str
    strategy: str = "project"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["span"] = list(self.span)
        return d


def role_of(node: Node) -> str:
    """单元在句子里的角色(纯类型/角色判断,不含任何文案)。"""
    if node.ntype == "action":
        return "action"
    if node.ntype == "quantity":
        return "parameter"
    if node.ntype == "condition":
        return "condition"
    if node.ntype == "constraint":
        return "constraint"
    if node.ntype == "artifact":
        return "structure" if node.attrs.get("structural") else "artifact"
    klass = node.attrs.get("class", "")
    if klass == "precursor":
        return "material"
    if klass in ("substance", "agent"):
        return "product"
    return "tool"


def surface_of(graph: GoalGraph, nid: str) -> str:
    """单元的**原文片段**(按 span 取,不做任何改写)。"""
    node = graph.node(nid)
    return graph.goal[node.span[0]:node.span[1]]


def project_subgraph(
    graph: GoalGraph,
    node_ids: list[str],
    sid: str,
    carrier_of: dict[str, str] | None = None,
    index: int = 0,
) -> Subgraph:
    """把一个单元集合投影成一个子图。

    - `text` 承载 → 该单元的**原文片段**进入 `rendered_text`(按 span 顺序,`SEP` 连接)
    - `image` 承载 → 该单元的原文片段进 `visual_spec`(结构化清单,不是散文)
    - `deferred`  → 本步不承载,计入 `deferred_units`(不隐藏信息损失)
    """
    carrier_of = carrier_of or {nid: CARRIER_TEXT for nid in node_ids}
    ordered = sorted(node_ids, key=lambda x: graph.node(x).span[0])

    text_parts: list[str] = []
    image_units: list[ProjectedUnit] = []
    deferred: list[str] = []
    audit: list[dict[str, Any]] = []
    for nid in ordered:
        node = graph.node(nid)
        channel = carrier_of.get(nid, CARRIER_TEXT)
        surface = surface_of(graph, nid)
        unit = ProjectedUnit(nid=nid, ntype=node.ntype, role=role_of(node), label=node.label,
                             span=node.span, surface=surface, channel=channel, sid=sid)
        audit.append(unit.to_dict())
        if channel == CARRIER_TEXT:
            text_parts.append(surface)
        elif channel == CARRIER_IMAGE:
            image_units.append(unit)
        else:
            deferred.append(nid)

    rendered = SEP.join(text_parts)
    visual_spec = None
    if image_units:
        visual_spec = "image_payload: " + "; ".join(
            f"{u.surface}[{u.span[0]}:{u.span[1]}]" for u in image_units)
    # 纯图像子图 → rendered_text 保持**空字符串**(绝不写占位话术;展示层自己说明)
    channel = CARRIER_IMAGE if image_units and not text_parts else (
        "text+image" if image_units else CARRIER_TEXT)
    return Subgraph(
        sid=sid,
        node_ids=list(node_ids),
        channel=channel,
        rendered_text=rendered,
        visual_spec=visual_spec,
        rewrite=audit,                 # 沿用字段名:这里是**投影审计**,不是改写审计
        score_source="projection",
    )


def traceability(graph: GoalGraph, subgraphs: list[Subgraph]) -> dict[str, Any]:
    """可机械检查的溯源报告:渲染文本里有没有**不在原文里**的字。

    - `all_traceable`:每段(按 `SEP` 切)都必须是 goal 的子串
    - `untraceable`:违反者清单(恒为空;非空即实现出错)
    - `non_ascii_added`:渲染文本中出现、而原文里没有的字符(用于抓"偷偷加中文框架")
    """
    untraceable: list[str] = []
    for sg in subgraphs:
        for part in sg.rendered_text.split(SEP):
            part = part.strip()
            if not part:
                continue
            if part not in graph.goal:
                untraceable.append(f"{sg.sid}: {part!r}")
    rendered_all = "".join(sg.rendered_text for sg in subgraphs)
    added_chars = sorted({c for c in rendered_all if c not in graph.goal and not c.isspace()
                          and c != SEP.strip()})
    return {
        "all_traceable": not untraceable,
        "untraceable": untraceable,
        "non_ascii_added": [c for c in added_chars if ord(c) > 127],
        "added_chars": added_chars,
    }


def projection_rows(graph: GoalGraph, subgraphs: list[Subgraph]) -> list[dict[str, Any]]:
    """扁平化的投影表:单元 → 子图 → 通道 → 原文片段。"""
    rows: list[dict[str, Any]] = []
    for sg in subgraphs:
        for row in sg.rewrite:
            rows.append({**row, "sid": sg.sid})
    return rows
