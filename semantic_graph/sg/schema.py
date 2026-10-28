"""语义图建模数据模型(docs/10 §7 的接口增量,与 docs/03 的字段语义兼容)。

纯 stdlib —— 目的之一是让"改指标/看结果"零 GPU 成本,可被 torch-free 报告链复用。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VER = "sg-0.2"

# 节点类型(docs/10 §2 R2)
NODE_TYPES = ("entity", "action", "quantity", "condition", "constraint", "artifact")

# 边类型(docs/10 §2 R2)
EDGE_TYPES = (
    "agent",
    "patient",
    "instrument",
    "precursor-of",
    "condition-of",
    "precedes",
    "located-at",
    "evades",
)

# 依赖边:M2 切分时不得切断(docs/04 M2、docs/10 §4 S2 约束 ①)
DEPENDENCY_EDGE_TYPES = ("precedes", "precursor-of")

# 依赖边的"承载方式"(docs/10 §4 的放宽扩展):
#   co-location  两个端点必须落在同一子图(默认,最严)
#   image        允许切开,但关系必须画进图像(箭头/示意)
#   glue         允许切开,但关系必须写进胶合句
#   ordering     允许切开,但关系必须由顺序词承载(如"上一步/下一步")
# 只有当 "co-location" 在 carriable_by 里时,edge.dependency 才强制同区。
CARRIABLE_BY = ("co-location", "image", "glue", "ordering")
DEFAULT_CARRIABLE_BY = ("co-location",)

# 类型驱动的 serializable 初判(docs/10 §7 / §9 R10:归属 M1 还是 M4 尚未定案)
NON_SERIALIZABLE_TYPES = ("quantity",)


@dataclass
class Node:
    """语义单元。`span` 是原文偏移,用于审计与"图→文本"回溯(docs/10 §7)。"""

    nid: str
    ntype: str
    label: str
    span: tuple[int, int]
    attrs: dict[str, Any] = field(default_factory=dict)
    serializable: bool = True

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["span"] = list(self.span)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Node":
        d = dict(d)
        d["span"] = tuple(d["span"])
        return cls(**d)


@dataclass
class Edge:
    """边。

    `dependency=True` 表示该边**在默认承载方式(co-location)下**必须保持同区(docs/10 §4 S2)。
    放宽拆分时把 `carriable_by` 换成 `("image", "glue", "ordering")`,于是这条边可以被切开 ——
    但代价是它必须由图像/胶合句/顺序词重表述,记录在 `Fragment.reexpressed_edges` 里。
    """

    src: str
    dst: str
    etype: str
    dependency: bool = False
    span: tuple[int, int] | None = None
    carriable_by: tuple[str, ...] = DEFAULT_CARRIABLE_BY

    @property
    def requires_colocation(self) -> bool:
        return self.dependency and "co-location" in self.carriable_by

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["span"] = list(self.span) if self.span else None
        d["carriable_by"] = list(self.carriable_by)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Edge":
        d = dict(d)
        d["span"] = tuple(d["span"]) if d["span"] else None
        d["carriable_by"] = tuple(d.get("carriable_by") or DEFAULT_CARRIABLE_BY)
        return cls(**d)


@dataclass
class GoalGraph:
    """一条 goal 的语义图。`task_kind` 与 `unsupported_reason` 支撑可计算的排除逻辑。"""

    behavior_id: str
    goal: str
    task_kind: str
    core_span: tuple[int, int] | None
    core_text: str
    roots: list[str] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    unsupported_reason: str | None = None
    extractor: str = "a5-template"

    @property
    def multi_root(self) -> bool:
        """多根目标(docs/10 §1.3 事实 3):`N` 要按根统计,不跨根合并。"""
        return len(self.roots) > 1

    @property
    def decomposable(self) -> bool:
        """至少两个可切分的单元才算"可拆"。"""
        return self.unsupported_reason is None and len(self.nodes) >= 2

    @property
    def dependency_edges(self) -> list[Edge]:
        return [e for e in self.edges if e.dependency]

    @property
    def colocation_edges(self) -> list[Edge]:
        """默认承载方式下**必须同区**的边(放宽拆分只针对其余依赖边)。"""
        return [e for e in self.edges if e.requires_colocation]

    def relaxed(self, types: tuple[str, ...] = DEPENDENCY_EDGE_TYPES) -> "GoalGraph":
        """返回一份把指定类型依赖边改为可重表述的副本(不改原图)。

        对应 docs/10 §4 的放宽扩展:边不再强制同区,但必须由 image/glue/ordering 承载。
        """
        edges = [
            Edge(src=e.src, dst=e.dst, etype=e.etype, dependency=e.dependency, span=e.span,
                 carriable_by=("image", "glue", "ordering") if e.etype in types else e.carriable_by)
            for e in self.edges
        ]
        return GoalGraph(
            behavior_id=self.behavior_id, goal=self.goal, task_kind=self.task_kind,
            core_span=self.core_span, core_text=self.core_text, roots=list(self.roots),
            nodes=list(self.nodes), edges=edges, unsupported_reason=self.unsupported_reason,
            extractor=self.extractor,
        )

    def node(self, nid: str) -> Node:
        for n in self.nodes:
            if n.nid == nid:
                return n
        raise KeyError(nid)

    def to_dict(self) -> dict[str, Any]:
        return {
            "behavior_id": self.behavior_id,
            "goal": self.goal,
            "task_kind": self.task_kind,
            "core_span": list(self.core_span) if self.core_span else None,
            "core_text": self.core_text,
            "roots": list(self.roots),
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "unsupported_reason": self.unsupported_reason,
            "extractor": self.extractor,
            "schema_ver": SCHEMA_VER,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GoalGraph":
        return cls(
            behavior_id=d["behavior_id"],
            goal=d["goal"],
            task_kind=d["task_kind"],
            core_span=tuple(d["core_span"]) if d["core_span"] else None,
            core_text=d["core_text"],
            roots=list(d.get("roots", [])),
            nodes=[Node.from_dict(n) for n in d["nodes"]],
            edges=[Edge.from_dict(e) for e in d["edges"]],
            unsupported_reason=d.get("unsupported_reason"),
            extractor=d.get("extractor", "a5-template"),
        )


@dataclass
class Fragment:
    """切分产物。

    `deps_cut` 恒为空 —— 非空即为"同区不变量"违例(docs/10 §4 S2)。
    放宽拆分时被切开的依赖边不写进 `deps_cut`,而是写进 `reexpressed_edges`
    (三元组 `(src, dst, 承载方式)`),由 `split.check_relaxation` 校验。
    """

    fid: str
    node_ids: list[str]
    text: str
    score: float
    deps_cut: list[tuple[str, str]] = field(default_factory=list)
    reexpressed_edges: list[tuple[str, str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["deps_cut"] = [list(x) for x in self.deps_cut]
        d["reexpressed_edges"] = [list(x) for x in self.reexpressed_edges]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Fragment":
        d = dict(d)
        d["deps_cut"] = [tuple(x) for x in d.get("deps_cut", [])]
        d["reexpressed_edges"] = [tuple(x) for x in d.get("reexpressed_edges", [])]
        return cls(**d)


@dataclass
class SplitResult:
    behavior_id: str
    method: str
    n: int
    fragments: list[Fragment]
    aggregate_text: str
    aggregate_score: float
    feasible_partitions: int = 0
    objective: str = "min_max"
    seed: int | None = None
    truncated: bool = False

    @property
    def max_fragment_score(self) -> float:
        return max((f.score for f in self.fragments), default=0.0)

    @property
    def mean_fragment_score(self) -> float:
        return sum(f.score for f in self.fragments) / len(self.fragments) if self.fragments else 0.0

    @property
    def deps_cut(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for f in self.fragments:
            out.extend(f.deps_cut)
        return out

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fragments"] = [f.to_dict() for f in self.fragments]
        d["max_fragment_score"] = self.max_fragment_score
        d["mean_fragment_score"] = self.mean_fragment_score
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SplitResult":
        return cls(
            behavior_id=d["behavior_id"],
            method=d["method"],
            n=d["n"],
            fragments=[Fragment.from_dict(f) for f in d["fragments"]],
            aggregate_text=d["aggregate_text"],
            aggregate_score=d["aggregate_score"],
            feasible_partitions=d.get("feasible_partitions", 0),
            objective=d.get("objective", "min_max"),
            seed=d.get("seed"),
            truncated=d.get("truncated", False),
        )


@dataclass
class Subgraph:
    """一个语义子图 —— 双子图拆分法(docs/10 §4 放宽扩展)的最小单位。

    与 `Fragment` 的区别:`Fragment` 只有"单元 + 文本",`Subgraph` 还带
      - `channel`:该子图走哪个通道(text / image / text+image)
      - `rendered_text`:**投影文本** = 该子图 text 承载单元的**原文 span 拼接**;
        不弱化、不造词(见 `project.py`),因此逐段可回原文
      - `visual_spec`:图像通道的**结构化清单**(`image_payload: 片段[span]; …`),不是散文
      - `rewrite`:**投影审计**(单元 → 通道 → 原文片段 → span);字段名沿用,
        内容里没有"改写后"这种东西 —— 本步不做文字加工
      - `reexpressed_edges`:被切开但已由 image/ordering 承载的依赖边
      - `judge`:词法 / LLM / caption 三类判定结果
    """

    sid: str
    node_ids: list[str]
    channel: str = "text"
    rendered_text: str = ""
    visual_spec: str | None = None
    rewrite: list[dict[str, Any]] = field(default_factory=list)
    reexpressed_edges: list[tuple[str, str, str]] = field(default_factory=list)
    judge: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    score_source: str = "projection"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["reexpressed_edges"] = [list(x) for x in self.reexpressed_edges]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Subgraph":
        d = dict(d)
        d["reexpressed_edges"] = [tuple(x) for x in d.get("reexpressed_edges", [])]
        return cls(**d)


@dataclass
class GoalRecord:
    """规范化后的输入记录(sources.build_subset 的产物)。`rid` 即 join key。"""

    rid: str
    goal: str
    dataset: str
    functional_category: str = ""
    semantic_category: str = ""
    tags: str = ""
    context: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GoalRecord":
        d = dict(d)
        d.setdefault("extra", {})
        return cls(**d)


def dumps(obj: Any) -> str:
    """统一 JSON 序列化:UTF-8 直出、不转义非 ASCII。"""
    if hasattr(obj, "to_dict"):
        obj = obj.to_dict()
    return json.dumps(obj, ensure_ascii=False, sort_keys=False)


def write_text_lf(path, text: str) -> None:
    """以 LF 写文本。

    Windows 上 `open(..., "w")` 会把换行翻成 CRLF,产出与仓库约定不一致的产物;
    **所有落盘都走这里**,别直接用 Path.write_text。
    """
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def dump_jsonl(items: list[Any], path) -> int:
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(dumps(it) + "\n")
            n += 1
    return n


def load_jsonl(path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:  # 行号对定位很重要
                raise ValueError(f"{path}:{ln} 不是合法 JSON: {e}") from e
    return out
