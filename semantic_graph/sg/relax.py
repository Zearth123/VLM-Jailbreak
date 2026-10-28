"""潜变量形式化:把"切分 + 通道对齐"写成论文式的 **generation order**(docs/10 §4.12)。

参考
----
Lyu, Cohen & Titov, *A Differentiable Relaxation of Graph Segmentation and Alignment for AMR
Parsing*, EMNLP 2021(arXiv:2010.12676)。论文的做法:

  1. 对齐 `A ∈ {0,1}^{n×(m+1)}` 与切分 `S ∈ {0,1}^{m×(m+1)}` 都是**潜变量**,拼成一个
     generation order `O = [A; S]`;约束是(每个 token 只有一个起点 / 每个节点只有一个后继 /
     每个节点只被生成一次 / 生成图必须无环);
  2. 目标 `log Σ_O P(O) P(v|x,O) P(E|x,O,v)` 不可直接边缘化 → 用 VAE(ELBO +
     inference network `Q_φ(O|G,x)`);
  3. 为了让 `O` 的采样可导 → **stochastic softmax**(perturb-and-MAP:Gumbel 噪声 + 约束下取 MAP),
     并且把所有按 `O` 定义的表示写成**对软 `O` 也成立**的加权和;
  4. 论文的 `∅`(空对齐)与 `A` / `A^∞`(`A`:节点是子图**起点**;`A^∞`:节点**属于**哪个子图)之分。

本模块把它落到我们的设定
----------------------
我们的图本来就带 `span`(规则抽取,不像 AMR 那样需要对 token 做对齐),所以真正要诱导的潜变量是
**通道对齐**:每个单元由 text / image / glue / **null** 承载。这里的 `null` 正是论文的 `∅` ——
单元不进任何子图的可见内容,只由胶合句兜。于是:

    论文 A^∞(成员)⇄ 我们的 `sid_of`(单元属于哪个子图)
    论文 A (起点)⇄ 我们的子图锚点(`a()`)
    论文 S (后继)⇄ 子图内按 span 的后继(`s_edges()`)
    论文 ∅       ⇄ `carrier == "null"`(高信号约束迁出文本通道)

**与论文不同的两处(必须写清楚)**:
  - 我们的图只有 5–15 个节点,**精确边缘化是可行的** —— 所以产物里报的是
    真实后验(而不是论文那种必要的松弛);松弛三件套(温度 softmax / Gumbel / perturb-and-MAP)
    仍然实现了,并接受单测,因为(a)它是论文的核心贡献,(b)图变大时要用它。
  - 判官是**黑盒**(且当前是词法档),没有梯度环境 → 线性打分器用
    **reward-weighted** 更新(有限空间上的 REINFORCE 式更新),不是论文的端到端反向传播。
"""

from __future__ import annotations

import itertools
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from . import project as P
from . import split as S
from .judge import Judge, is_authoritative, make_judge
from .probe import LexicalProbe
from .schema import DEPENDENCY_EDGE_TYPES, GoalGraph, write_text_lf

CARRIERS = P.CARRIERS  # ("text", "image", "deferred")
NULL_CARRIER = P.CARRIER_DEFERRED
DEFAULT_CAP = 20000


# ---------------------------------------------------------------- 生成顺序(潜变量)


@dataclass(frozen=True)
class Order:
    """一个 generation order:切分 + 通道对齐 + 子图顺序。

    用 tuple 字段以便哈希(去重、当作字典键)。`node_order` 是**按 span 排序的单元序列**,
    组内顺序与 `S`(后继关系)都由它导出 —— 保持与原文顺序一致。
    """

    behavior_id: str
    sid_of: tuple[tuple[str, str], ...]
    carrier_of: tuple[tuple[str, str], ...]
    sequence: tuple[str, ...]
    node_order: tuple[str, ...] = ()

    # ---- 论文里的 A^∞ / A / S 三个对象
    def sid_map(self) -> dict[str, str]:
        return dict(self.sid_of)

    def carrier_map(self) -> dict[str, str]:
        return dict(self.carrier_of)

    def groups(self) -> list[list[str]]:
        out: dict[str, list[str]] = {sid: [] for sid in self.sequence}
        for nid in (self.node_order or tuple(n for n, _ in self.sid_of)):
            sid = dict(self.sid_of).get(nid)
            if sid in out:
                out[sid].append(nid)
        return [out[sid] for sid in self.sequence]

    def a(self) -> dict[str, str]:
        """每个子图的**起点单元**(论文的 A):组内 span 最小者。"""
        return {sid: grp[0] for sid, grp in zip(self.sequence, self.groups())}

    def a_inf(self) -> list[tuple[str, str]]:
        """成员关系(论文的 A^∞):本实现里就是 `sid_of`。"""
        return list(self.sid_of)

    def s_edges(self) -> list[tuple[str, str]]:
        """子图内后继(论文的 S):按 span 顺序串联。"""
        out: list[tuple[str, str]] = []
        for grp in self.groups():
            out.extend(zip(grp, grp[1:]))
        return out


def carriers_in_order(order: Order) -> dict[str, str]:
    return order.carrier_map()


def default_carriers(graph: GoalGraph) -> dict[str, tuple[str, ...]]:
    """每个单元的**候选承载**(= 论文里对齐分布的支撑集)。

    - 产物 / 参数 / 结构类:`text | image` —— 让松弛自己决定要不要走图像
    - 其余(含约束):`text`

    `deferred` 通道保留在 `CARRIERS` 里但**不进默认候选**:它意味着"本步不承载",
    属于 M6 组装的决定,不在"图建模 + 子图拆分"范围内。
    """
    out: dict[str, tuple[str, ...]] = {}
    for n in graph.nodes:
        role = P.role_of(n)
        if role == "product" or n.ntype == "quantity" or n.attrs.get("structural"):
            out[n.nid] = (P.CARRIER_TEXT, P.CARRIER_IMAGE)
        else:
            out[n.nid] = (P.CARRIER_TEXT,)
    return out


# 被切开的依赖边"必须由谁的承载来兜"(与 `split.CARRIER_POLICY` 保持一致):
#   precursor-of(物料流):要在图上画箭头 → 至少一个端点走 image 通道
#   precedes(先序):由子图顺序与胶合句承载 → **不约束端点**(顺序本身就存在)
EDGE_CARRIER_REQUIREMENT: dict[str, tuple[str, ...]] = {
    "precursor-of": (P.CARRIER_IMAGE,),
    "precedes": (),
}


def _carriers_ok(graph: GoalGraph, sid_of: dict[str, str], carrier_of: dict[str, str]) -> bool:
    """结构约束:

    - 必须同区的依赖边不得跨子图(枚举时已保证,这里再核一遍);
    - 可重表述的依赖边允许跨子图,但必须有人兜:按边类型的 `EDGE_CARRIER_REQUIREMENT`
      检查至少一个端点的承载落在要求集合里(要求集合为空则不约束)。
    """
    for e in graph.edges:
        if not e.dependency:
            continue
        if sid_of.get(e.src) == sid_of.get(e.dst):
            continue
        if e.requires_colocation:
            return False
        need = EDGE_CARRIER_REQUIREMENT.get(e.etype, ())
        if not need:
            continue
        if carrier_of.get(e.src) not in need and carrier_of.get(e.dst) not in need:
            return False
    return True


def enumerate_orders(
    graph: GoalGraph,
    k: int = 2,
    carriers: dict[str, tuple[str, ...]] | None = None,
    relax_types: tuple[str, ...] = DEPENDENCY_EDGE_TYPES,
    cap: int = DEFAULT_CAP,
) -> list[Order]:
    """枚举所有满足结构约束的 generation order(切分 × 通道对齐)。

    切分在**放宽后**的图上枚举:可重表述的依赖边不再强制同区,但切开后必须由非 text 承载
    (`_carriers_ok`)—— 这正是论文把"切分"与"对齐"耦合起来的地方。
    """
    carriers = carriers or default_carriers(graph)
    work = graph.relaxed(relax_types)
    supernodes = S.dependency_components(work)
    labels_list, _ = S.enumerate_partitions(supernodes, k, image_budget=None, cap=cap)
    node_order = tuple(n.nid for n in sorted(graph.nodes, key=lambda n: n.span[0]))
    combos = list(itertools.product(*[carriers[nid] for nid in node_order]))
    orders: list[Order] = []
    seen: set[tuple] = set()
    for labels in labels_list:
        groups: dict[str, list[str]] = {}
        for i, lab in enumerate(labels):
            groups.setdefault(f"sg{lab + 1}", []).extend(supernodes[i])
        sequence = tuple(sorted(groups, key=lambda sid: min(
            graph.node(x).span[0] for x in groups[sid])))
        sid_of = {nid: sid for sid, grp in groups.items() for nid in grp}
        for combo in combos:
            carrier_of = dict(zip(node_order, combo))
            if not _carriers_ok(work, sid_of, carrier_of):
                continue
            key = (tuple(sorted(sid_of.items())), tuple(sorted(carrier_of.items())))
            if key in seen:
                continue
            seen.add(key)
            orders.append(Order(
                behavior_id=graph.behavior_id,
                sid_of=tuple(sorted(sid_of.items())),
                carrier_of=tuple(sorted(carrier_of.items())),
                sequence=sequence,
                node_order=node_order,
            ))
            if len(orders) >= cap:
                return orders
    return orders


# ---------------------------------------------------------------- 松弛三件套


def softmax(xs: Iterable[float], tau: float = 1.0) -> list[float]:
    xs = list(xs)
    if not xs:
        return []
    tau = max(tau, 1e-9)
    m = max(xs)
    exps = [math.exp((x - m) / tau) for x in xs]
    z = sum(exps)
    return [e / z for e in exps] if z > 0 else [1.0 / len(xs)] * len(xs)


def gumbel_noise(rng: random.Random, size: int) -> list[float]:
    out = []
    for _ in range(size):
        u = min(max(rng.random(), 1e-12), 1 - 1e-12)
        out.append(-math.log(-math.log(u)))
    return out


def gumbel_softmax(logits: list[float], tau: float = 1.0,
                   rng: random.Random | None = None) -> tuple[list[float], list[int]]:
    """Gumbel-softmax:返回 (软权重, 硬 one-hot)。`tau→0` 时软权重趋近硬 one-hot。"""
    rng = rng or random.Random(0)
    noisy = [x + g for x, g in zip(logits, gumbel_noise(rng, len(logits)))]
    soft = softmax(noisy, tau)
    hard = [0] * len(soft)
    hard[max(range(len(soft)), key=lambda i: soft[i])] = 1
    return soft, hard


def perturb_and_map(
    orders: list[Order],
    score_fn: Callable[[Order], float],
    rng: random.Random | None = None,
    tau: float = 1.0,
    draws: int = 8,
) -> tuple[Order, list[float]]:
    """perturb-and-MAP(论文的 stochastic softmax 采样端)。

    对每条 order 的分数加 Gumbel 噪声后取 MAP;`draws` 次采样里取经验频率最高的那条,
    同时返回该条在**加噪分布**下的软权重(供 straight-through 之类的下游使用)。

    注意:约束是在 `enumerate_orders` 里就满足的,所以这里的 argmax 天然落在可行域内
    —— 论文用约束求解器,我们因为图小,直接精确枚举。
    """
    rng = rng or random.Random(0)
    scores = [score_fn(o) for o in orders]
    counts = [0] * len(orders)
    last_soft: list[float] = []
    for _ in range(max(1, draws)):
        noisy = [s + g for s, g in zip(scores, gumbel_noise(rng, len(scores)))]
        best = max(range(len(noisy)), key=lambda i: noisy[i])
        counts[best] += 1
        last_soft = softmax(noisy, tau)
    pick = max(range(len(counts)), key=lambda i: (counts[i], scores[i]))
    return orders[pick], last_soft


# ---------------------------------------------------------------- 精确后验(= A^∞ 的边际)


def posterior(
    orders: list[Order],
    reward: list[float],
    beta: float = 4.0,
) -> dict[str, Any]:
    """`P(O) ∝ exp(β·R(O))` 的**精确**归一化(图小,不需要松弛近似)。

    返回 MAP、后验熵(以 nat 计)、逐单元通道边际(论文 A^∞ 的软版本)、逐单元子图边际。
    """
    if len(orders) != len(reward):
        raise ValueError("orders 与 reward 长度不一致")
    if not orders:
        return {"n_orders": 0, "probs": [], "entropy": 0.0, "uniform": True,
                "map_index": None, "carrier_marginals": {}, "group_marginals": {}}
    mx = max(reward)
    exps = [math.exp(beta * (r - mx)) for r in reward]
    z = sum(exps)
    probs = [e / z for e in exps]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    n = len(probs)
    uniform = abs(entropy - math.log(n)) < 1e-9
    carrier_marg: dict[str, dict[str, float]] = {}
    group_marg: dict[str, dict[str, float]] = {}
    for o, p in zip(orders, probs):
        for nid, c in o.carrier_of:
            carrier_marg.setdefault(nid, {})
            carrier_marg[nid][c] = carrier_marg[nid].get(c, 0.0) + p
        for nid, sid in o.sid_of:
            group_marg.setdefault(nid, {})
            group_marg[nid][sid] = group_marg[nid].get(sid, 0.0) + p
    return {
        "n_orders": n,
        "probs": probs,
        "entropy": entropy,
        "max_entropy": math.log(n),
        "uniform": uniform,
        "map_index": max(range(n), key=lambda i: (probs[i], reward[i])),
        "carrier_marginals": carrier_marg,
        "group_marginals": group_marg,
        "reward_max": mx,
        "reward_min": min(reward),
    }


# ---------------------------------------------------------------- 线性打分器 + reward-weighted 拟合


FEATURES = ("probe_max", "carrier_cost", "cross_bonus", "span_compact", "image_share")


def text_visible_ids(order: Order, group: list[str]) -> list[str]:
    """**会在文本里出现**的单元:只有 `carrier == "text"` 的。

    这是本模块最容易搞错的一点(实测踩过):image / glue / null 承载的单元都不出现在子图文本里 ——
    image 的信息在像素上,glue 的在胶合句里,null 的干脆不出现。把它算成"可见",就相当于
    忘记图像通道存在的意义。
    """
    carrier = order.carrier_map()
    return [nid for nid in group if carrier.get(nid) == "text"]


def _projected_text(graph: GoalGraph, order: Order, group: list[str]) -> str:
    """一个子图的**投影文本** = 该组里 text 承载单元的原文片段拼接(无任何改写)。"""
    carrier = order.carrier_map()
    return P.project_subgraph(graph, group, "proj", carrier_of=carrier).rendered_text


def _text_score(graph: GoalGraph, group: list[str], order: Order, probe: LexicalProbe) -> float:
    text = _projected_text(graph, order, group)
    return probe.score_text(text) if text else 0.0


def features_of(order: Order, graph: GoalGraph, probe: LexicalProbe,
                work: GoalGraph | None = None) -> dict[str, float]:
    """把一条 order 变成特征(透明、可审计;不是学习出来的表示)。

    `work` 传入放宽后的图;不传则就地构造。**必须用放宽后的图**判断"哪条边被切开" ——
    在严格图里所有依赖边都 `requires_colocation`,于是 cross 恒为 0(实测踩过)。
    """
    work = work or graph.relaxed()
    carrier = order.carrier_map()
    groups = order.groups()
    # 与 `reward_structural` 用**同一把尺子**:投影后最显眼的那一片有多显眼
    probe_max = max((_text_score(graph, grp, order, probe) for grp in groups), default=0.0)
    carrier_cost = sum(1.0 for c in carrier.values() if c != P.CARRIER_TEXT) / max(1, len(carrier))
    image_share = sum(1.0 for c in carrier.values() if c == P.CARRIER_IMAGE) / max(1, len(carrier))
    sid_of = order.sid_map()
    cross = 0
    for e in work.edges:
        if e.dependency and not e.requires_colocation and sid_of.get(e.src) != sid_of.get(e.dst):
            if carrier.get(e.src) != P.CARRIER_TEXT or carrier.get(e.dst) != P.CARRIER_TEXT:
                cross += 1
    cross_bonus = cross / max(1, len(graph.dependency_edges))
    span_compact = 0.0
    for grp in groups:
        lo = min(graph.node(x).span[0] for x in grp)
        hi = max(graph.node(x).span[1] for x in grp)
        span_compact += (hi - lo)
    span_compact = 1.0 / (1.0 + span_compact / 100.0)
    return {"probe_max": probe_max, "carrier_cost": carrier_cost,
            "cross_bonus": cross_bonus, "span_compact": span_compact,
            "image_share": image_share}


def score_of(feat: dict[str, float], weights: dict[str, float]) -> float:
    return sum(weights.get(k, 0.0) * v for k, v in feat.items())


def reward_weighted_fit(
    feats: list[dict[str, float]],
    reward: list[float],
    iters: int = 200,
    eta: float = 0.5,
    tau: float = 0.25,
    standardize: bool = True,
) -> tuple[dict[str, float], list[dict[str, float]], dict[str, dict[str, float]]]:
    """有限空间上的 **reward-weighted** 更新(REINFORCE 式的线性打分器拟合)。

    `w ← w + η · Σ_i P_τ(i) · (R_i − R̄) · z_i`,其中 `P_τ` 是当前打分器在温度 `τ` 下的后验,
    `z` 是标准化后的特征(否则不同量纲的特征更新尺度差异过大,学不动 —— 实测踩过)。

    返回 `(权重, 历史, 标准化参数)`;`标准化参数` 要让下游能用同一把尺子给新 order 打分。
    它替代的是论文里「inference network + 反向传播」那一步 —— 我们没有梯度环境,也不假装有。
    """
    keys = tuple(FEATURES)
    n = max(1, len(feats))
    norm = {"mu": {}, "sd": {}}
    if standardize and feats:
        for k in keys:
            vals = [f.get(k, 0.0) for f in feats]
            mu = sum(vals) / len(vals)
            var = sum((v - mu) ** 2 for v in vals) / len(vals)
            norm["mu"][k] = mu
            norm["sd"][k] = math.sqrt(var) or 1.0
    else:
        norm["mu"] = {k: 0.0 for k in keys}
        norm["sd"] = {k: 1.0 for k in keys}

    def z_of(f: dict[str, float]) -> dict[str, float]:
        return {k: (f.get(k, 0.0) - norm["mu"][k]) / norm["sd"][k] for k in keys}

    Z = [z_of(f) for f in feats]
    weights = {k: 0.0 for k in keys}
    history: list[dict[str, float]] = []
    for _ in range(iters):
        logits = [score_of(z, weights) for z in Z]
        probs = softmax(logits, tau=tau)
        r_bar = sum(p * r for p, r in zip(probs, reward))
        for k in keys:
            grad = sum(p * (r - r_bar) * z.get(k, 0.0) for p, r, z in zip(probs, reward, Z)) / n
            weights[k] += eta * grad
        history.append({"weights": dict(weights),
                        "expected_reward": sum(p * r for p, r in zip(probs, reward))})
    return weights, history, norm


# ---------------------------------------------------------------- 渲染与两类 reward


def render_order(
    graph: GoalGraph,
    order: Order,
) -> list[dict[str, Any]]:
    """把一条 order 投影成一组子图(**只取原文 span,不做任何文字加工**)。"""
    out: list[dict[str, Any]] = []
    carrier = order.carrier_map()
    groups = order.groups()
    for i, sid in enumerate(order.sequence):
        grp = groups[i]
        sg = P.project_subgraph(graph, grp, sid, carrier_of=carrier, index=i)
        out.append({"sid": sid, "hero": sg, "carriers": {nid: carrier.get(nid) for nid in grp}})
    return out


def reward_lexical(graph: GoalGraph, orders: list[Order], judge: Judge) -> list[float]:
    """判官档 reward:每个子图的**投影文本**都过判官 = 1,否则 0。

    投影不做任何弱化,所以这一列直接回答"原文片段本身能不能过判官"。
    """
    out = []
    for o in orders:
        ok = True
        for row in render_order(graph, o):
            text = row["hero"].rendered_text
            if text and judge.judge_text(text)["harmful"]:
                ok = False
                break
        out.append(1.0 if ok else 0.0)
    return out


def reward_structural(graph: GoalGraph, orders: list[Order], probe: LexicalProbe) -> list[float]:
    """结构档 reward:投影后最显眼的那一片有多不显眼(`1 − max`)。

    与 `features_of` 共用 `_text_score`,口径是"**只算 text 承载**的单元" ——
    把产物放到图像通道,这一项就会显著降低,这正是图像通道在数学上的价值。
    """
    out = []
    for o in orders:
        worst = max((_text_score(graph, grp, o, probe) for grp in o.groups()), default=0.0)
        out.append(1.0 - worst)
    return out


# ---------------------------------------------------------------- 产物


def run_relax(
    rid: str,
    out,
    k: int = 2,
    beta: float = 8.0,
    judge_name: str = "lexical",
    fit_iters: int = 200,
    seed: int = 20260101,
    created: str = "",
) -> dict[str, Any]:
    """跑一次潜变量形式化:枚举 → 两类 reward → 精确后验 → 松弛采样 → 线性打分器拟合。

    渲染只做**投影**(原文 span 拼接),不做任何文字弱化或新增。
    """
    from . import extract, sources

    record = next((r for r in sources.load_harmbench() if r.rid == rid), None)
    if record is None:
        raise ValueError(f"HarmBench 里没有 {rid}")
    graph = extract.build_graph(rid, record.goal)
    probe = LexicalProbe()
    judge = make_judge(judge_name)

    orders = enumerate_orders(graph, k=k)
    if not orders:
        raise ValueError(f"{rid} 在 k={k} 下没有可行 order")
    r_struct = reward_structural(graph, orders, probe)
    r_lex = reward_lexical(graph, orders, judge)

    post_struct = posterior(orders, r_struct, beta=beta)
    post_lex = posterior(orders, r_lex, beta=beta)

    feats = [features_of(o, graph, probe) for o in orders]
    weights, history, norm = reward_weighted_fit(feats, r_struct, iters=fit_iters)

    rng = random.Random(seed)
    map_order = orders[post_struct["map_index"]] if post_struct["map_index"] is not None else orders[0]
    sampled, soft = perturb_and_map(orders, lambda o: score_of(
        {k: (features_of(o, graph, probe).get(k, 0.0) - norm["mu"][k]) / norm["sd"][k]
         for k in FEATURES}, weights), rng=rng, draws=16)

    # 溯源:投影文本必须逐段回得到原文
    map_subgraphs = [row["hero"] for row in render_order(graph, map_order)]
    trace = P.traceability(graph, map_subgraphs)

    meta = {
        "rid": rid,
        "created": created,
        "k": k,
        "beta": beta,
        "judge": judge.describe(),
        "authoritative_judge": is_authoritative(judge),
        "n_orders": len(orders),
        "weights": weights,
        "traceability": trace,
        "reference": "Lyu, Cohen & Titov (EMNLP 2021), arXiv:2010.12676",
    }
    text = render_relax_artifact(graph, orders, r_struct, r_lex, post_struct, post_lex,
                                 map_order, sampled, soft, weights, history, meta, norm)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(out, text)
    orders_path = out.with_name(out.stem + ".orders.jsonl")
    with open(orders_path, "w", encoding="utf-8", newline="\n") as f:
        for o, rs, rl, ps, pl in zip(orders, r_struct, r_lex, post_struct["probs"], post_lex["probs"]):
            f.write(json.dumps({
                "order": {"sid_of": list(o.sid_of), "carrier_of": list(o.carrier_of),
                          "sequence": list(o.sequence)},
                "reward_structural": round(rs, 6), "reward_lexical": round(rl, 6),
                "posterior_structural": round(ps, 6), "posterior_lexical": round(pl, 6),
            }, ensure_ascii=False) + "\n")
    return {"out": str(out), "orders_jsonl": str(orders_path), "meta": meta,
            "posterior_structural": {kk: vv for kk, vv in post_struct.items() if kk != "probs"},
            "posterior_lexical": {kk: vv for kk, vv in post_lex.items() if kk != "probs"}}


def render_relax_artifact(graph: GoalGraph, orders: list[Order], r_struct: list[float],
                          r_lex: list[float], post_struct: dict[str, Any],
                          post_lex: dict[str, Any], map_order: Order, sampled: Order,
                          soft: list[float], weights: dict[str, float],
                          history: list[dict[str, Any]], meta: dict[str, Any],
                          norm: dict[str, dict[str, float]] | None = None) -> str:
    L: list[str] = []

    def w(s: str = "") -> None:
        L.append(s)

    w(f"# 潜变量切分+对齐(论文式 generation order) · `{meta['rid']}`")
    w()
    w(f"- 生成时间:{meta.get('created', '?')} · k={meta['k']} · β={meta['beta']}")
    w(f"- 判官:`{meta['judge']['judge_id']}`(authoritative={meta['authoritative_judge']})")
    w("- 范围:只做**图建模 + 子图拆分**;子图文本是**原文 span 的拼接**(不弱化、不新增)")
    w(f"- 参考:Lyu, Cohen & Titov, *A Differentiable Relaxation of Graph Segmentation and "
      f"Alignment for AMR Parsing*, EMNLP 2021([arXiv:2010.12676](https://arxiv.org/abs/2010.12676))")
    w(f"- 枚举到的可行 order 数:**{meta['n_orders']}**(切分 × 通道对齐,已满足结构约束)")
    tr = meta.get("traceability", {})
    w(f"- 溯源自检:每段投影文本都能回原文 = **{tr.get('all_traceable')}**;"
      f"渲染文本里原文没有的字符 = {tr.get('added_chars') or '无'}")
    w()
    if not meta["authoritative_judge"]:
        w("> ⚠️ 当前判官是**词法档(非权威)**。因为投影不做任何弱化,"
          "`reward_lexical` 判的就是**原文片段本身** —— 它若判有害,说明这些片段确实带词面痕迹。")
        w()

    w("## 1 · 与论文的对应")
    w()
    w("| 论文 | 本实现 |")
    w("|---|---|")
    w("| `A^∞`:节点是否属于某个子图 | `Order.sid_of` |")
    w("| `A`:节点是否是子图**起点** | `Order.a()`(组内 span 最小者) |")
    w("| `S`:节点后继(子图内顺序) | `Order.s_edges()`(组内按 span 串联) |")
    w("| `O=[A;S]` 的结构约束 | `enumerate_orders`:划分 + 必须同区的边不跨区 + "
      "跨区边按其类型的 `EDGE_CARRIER_REQUIREMENT` 找人兜(`precursor-of` 要 image,`precedes` 由顺序承载) |")
    w("| `∅` 空对齐 | `deferred` 通道:本步不承载(留给 M6 组装),默认不启用 |")
    w("| VAE + inference network | 线性打分器 + `reward_weighted_fit`(有限空间上的 reward-weighted 更新) |")
    w("| stochastic softmax(perturb-and-MAP) | `gumbel_softmax` / `perturb_and_map`(已实现并单测) |")
    w("| 松弛的必要性 | **我们不需要**:图只有 "
      f"{len(graph.nodes)} 个节点 → 精确后验可行(报的就是真实后验) |")
    w()

    w("## 2 · 两类 reward 与后验")
    w()
    w("| reward | 定义 | max | min | 后验熵 | 最大熵 | 是否无区分力 |")
    w("|---|---|---|---|---|---|---|")
    w(f"| `reward_lexical` | 每个子图的投影文本都过判官 = 1 | {post_lex.get('reward_max', 0):.3f} | "
      f"{post_lex.get('reward_min', 0):.3f} | {post_lex.get('entropy', 0):.3f} | "
      f"{post_lex.get('max_entropy', 0):.3f} | {'**是**' if post_lex.get('uniform') else '否'} |")
    w(f"| `reward_structural` | `1 − 投影后最显眼那片的分数` | {post_struct['reward_max']:.3f} | "
      f"{post_struct['reward_min']:.3f} | {post_struct['entropy']:.3f} | "
      f"{post_struct['max_entropy']:.3f} | {'是' if post_struct['uniform'] else '否'} |")
    w()
    w(f"- **MAP order**(结构档后验最大者):后验概率 {post_struct['probs'][post_struct['map_index']]:.3f}")
    w()
    w(f"  - 子图划分:`{' ‖ '.join(' + '.join(graph.node(x).label for x in grp) for grp in map_order.groups())}`")
    w(f"  - 非 text 承载:" + (", ".join(
        f"{graph.node(nid).label}→{c}" for nid, c in map_order.carrier_of if c != P.CARRIER_TEXT) or "无"))
    for row in render_order(graph, map_order):
        w(f"  - 投影文本 `{row['sid']}`:`{row['hero'].rendered_text or '(空)'}`")
        if row["hero"].visual_spec:
            w(f"  - image payload `{row['sid']}`:`{row['hero'].visual_spec}`")
    w()
    same = (sampled.sid_of, sampled.carrier_of) == (map_order.sid_of, map_order.carrier_of)
    w(f"- **perturb-and-MAP 采样**(Gumbel,16 次):"
      f"`{' ‖ '.join(' + '.join(graph.node(x).label for x in grp) for grp in sampled.groups())}`"
      f"(与 MAP {'一致' if same else '不同 —— 后验确实有多个峰'})")
    w()

    w("## 3 · 通道对齐的边际(论文 A^∞ 的软版本)")
    w()
    w("| 单元 | 类型 | 角色 | " + " | ".join(CARRIERS) + " | 最可能 |")
    w("|---|---|---|" + "---|" * (len(CARRIERS) + 1))
    carr = post_struct["carrier_marginals"]
    for n in graph.nodes:
        row = carr.get(n.nid, {})
        best = max(row.items(), key=lambda kv: kv[1])[0] if row else "—"
        cells = " | ".join(f"{row.get(c, 0.0):.3f}" for c in CARRIERS)
        w(f"| `{n.nid}` {n.label} | {n.ntype} | {P.role_of(n)} | {cells} | **{best}** |")
    w()
    w("> 这张表就是「松弛到底选了什么」的答案:每一列是该单元走该通道的后验概率。")
    w()

    w("## 4 · 结构档后验最高的前 5 条 order")
    w()
    w("| # | 子图划分 | text 承载 | 非 text 承载 | `reward_structural` | 后验 |")
    w("|---|---|---|---|---|---|")
    order_idx = sorted(range(len(orders)), key=lambda i: -post_struct["probs"][i])[:5]
    for rank, i in enumerate(order_idx, 1):
        o = orders[i]
        groups = " ‖ ".join("+".join(graph.node(x).label for x in grp) for grp in o.groups())
        car = o.carrier_map()
        txt = ", ".join(graph.node(nid).label for nid, c in o.carrier_of
                        if c == P.CARRIER_TEXT) or "—"
        nontxt = ", ".join(f"{graph.node(nid).label}→{c}" for nid, c in o.carrier_of
                           if c != P.CARRIER_TEXT)
        w(f"| {rank} | {groups} | {txt} | {nontxt or '—'} | {r_struct[i]:.3f} | "
          f"{post_struct['probs'][i]:.3f} |")
    w()

    w("## 5 · 线性打分器(reward-weighted 拟合)")
    w()
    w("| 特征 | 权重(标准化尺度) | 标准化 μ | 标准化 σ |")
    w("|---|---|---|---|")
    for k_, v in weights.items():
        mu = (norm or {"mu": {}, "sd": {}})["mu"].get(k_, 0.0)
        sd = (norm or {"mu": {}, "sd": {}})["sd"].get(k_, 1.0)
        w(f"| `{k_}` | {v:+.4f} | {mu:.4f} | {sd:.4f} |")
    w()
    if history:
        head = " → ".join(f"{h['expected_reward']:.4f}" for h in history[:5])
        tail = " → ".join(f"{h['expected_reward']:.4f}" for h in history[-3:])
        w(f"- 期望 reward 轨迹:第 1–5 步 {head} … 最后 3 步 {tail}")
        w(f"- 拟合提升:`{history[0]['expected_reward']:.4f} → {history[-1]['expected_reward']:.4f}`"
          f"(+{history[-1]['expected_reward'] - history[0]['expected_reward']:.4f})")
    w()
    w("> 更新式 `w ← w + η·Σ_i P_τ(i)·(R_i − R̄)·z_i`(有限空间上的 REINFORCE 式),"
      "替代论文里「inference network + 反向传播」那一步 —— 我们没有梯度环境,也不假装有。")
    w(">")
    w("> `carrier_cost` 与 `image_share` 在本目标里数值相同,因为该目标唯一用到的非 text 承载就是 image;"
      "出现 `deferred` 承载时两者会分叉。")
    w()

    w("## 6 · 范围与诚实清单")
    w()
    w("**本步只做图建模 + 子图拆分**;子图文本是原文 span 的拼接,**不做**文字弱化、不加框架、不注入新语义"
      "(那属于 M3 语境重构,docs/04 M3 / P3)。因此:")
    w()
    w("1. **没有神经 inference network、没有端到端反向传播** —— 判官是黑盒、环境无 torch;")
    w("2. **`reward_lexical` 直接判原文片段**:它判有害,说明这些片段本身带词面痕迹 ——")
    w("   这是诚实的结果,而不是靠措辞把判官哄过去;")
    w("3. **图像侧尚未闭环**:`image` 承载要求图里真的把那条关系画出来,当前只有 `image_payload` 清单"
      "(哪些单元的哪段原文要进图),渲染器与 caption 检查仍缺;")
    w("4. 论文的收益来自「学出来的切分打败手工规则」,而我们**故意不学**(docs/08 §2.1:构建管线不许依赖对齐模型);"
      "本模块的用途是把设计里的潜变量结构写清楚,并让「选哪个切分/通道」变成有后验、可审计的决策。")
    w()
    return "\n".join(L)
