"""输入读取与"调整为合适的数据格式"(docs/10 §10.1 的数据陷阱都在这里处理)。

规则:
  - `harmbench_behaviors_text_all.csv` 是 UTF-8 **带 BOM** → `utf-8-sig`
  - 400 行数据实际占 1530 物理行 → 必须用 `csv` 模块,**禁止按行或裸逗号切**
  - `BehaviorID` 是稳定 join key;AdvBench 无 id → 用 goal 的 sha1 派生稳定 id
  - 子集筛选完全确定(按 rid 排序),不用随机种子,保证可复现
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from .schema import GoalRecord

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data_origin"
HARMBENCH = DATA_DIR / "harmbench_behaviors_text_all.csv"
ADVBENCH = DATA_DIR / "advbench_harmful_behaviors.csv"

# docs/10 §1.3 事实 2 的两组类别
STRUCTURED_CATEGORIES = ("cybercrime_intrusion", "chemical_biological")
RHETORICAL_CATEGORIES = ("misinformation_disinformation", "harassment_bullying")

SUBSETS = ("structured", "rhetorical", "mixed", "standard", "all", "advbench")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件: {path}")
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    # 去掉 BOM 可能残留的列名
    fixed: list[dict[str, str]] = []
    for r in rows:
        fixed.append({(k or "").lstrip("\ufeff"): (v or "") for k, v in r.items()})
    return fixed


def load_harmbench(path: Path | None = None) -> list[GoalRecord]:
    rows = _read_csv_rows(path or HARMBENCH)
    out: list[GoalRecord] = []
    seen: set[str] = set()
    for r in rows:
        rid = (r.get("BehaviorID") or "").strip()
        if not rid:
            raise ValueError("HarmBench 行缺 BehaviorID —— 不能用行号做键")
        if rid in seen:
            raise ValueError(f"BehaviorID 重复: {rid}")
        seen.add(rid)
        out.append(
            GoalRecord(
                rid=rid,
                goal=(r.get("Behavior") or "").strip(),
                dataset="harmbench",
                functional_category=(r.get("FunctionalCategory") or "").strip(),
                semantic_category=(r.get("SemanticCategory") or "").strip(),
                tags=(r.get("Tags") or "").strip(),
                context=(r.get("ContextString") or "").strip(),
            )
        )
    return out


def advbench_id(goal: str) -> str:
    """AdvBench 没有稳定 id → 由 goal 派生(内容变则 id 变,可审计)。"""
    return "advbench_" + hashlib.sha1(goal.strip().encode("utf-8")).hexdigest()[:12]


def load_advbench(path: Path | None = None) -> list[GoalRecord]:
    rows = _read_csv_rows(path or ADVBENCH)
    out: list[GoalRecord] = []
    seen: set[str] = set()
    for r in rows:
        goal = (r.get("goal") or "").strip()
        if not goal:
            raise ValueError("AdvBench 行缺 goal")
        rid = advbench_id(goal)
        if rid in seen:
            raise ValueError(f"AdvBench 派生 id 重复: {rid}")
        seen.add(rid)
        out.append(
            GoalRecord(
                rid=rid,
                goal=goal,
                dataset="advbench",
                functional_category="advbench",
                semantic_category="advbench",
                extra={"target": (r.get("target") or "").strip()},
            )
        )
    return out


def load_all() -> list[GoalRecord]:
    return load_harmbench() + load_advbench()


def build_subset(
    records: list[GoalRecord],
    subset: str = "mixed",
    limit: int | None = None,
) -> list[GoalRecord]:
    """筛选工作子集。全部按 `rid` 排序,结果确定可复现。"""
    if subset not in SUBSETS:
        raise ValueError(f"未知子集: {subset}(可选 {', '.join(SUBSETS)})")

    if subset == "all":
        picked = sorted(records, key=lambda r: r.rid)
    elif subset == "advbench":
        picked = sorted([r for r in records if r.dataset == "advbench"], key=lambda r: r.rid)
    elif subset == "standard":
        picked = sorted([r for r in records if r.functional_category == "standard"], key=lambda r: r.rid)
    elif subset == "structured":
        picked = sorted(
            [r for r in records
             if r.functional_category == "standard" and r.semantic_category in STRUCTURED_CATEGORIES],
            key=lambda r: r.rid,
        )
    elif subset == "rhetorical":
        picked = sorted(
            [r for r in records
             if r.functional_category == "standard" and r.semantic_category in RHETORICAL_CATEGORIES],
            key=lambda r: r.rid,
        )
    else:
        # mixed:结构化目标**优先全取**,再用其余 standard 行按 rid 顺序补齐(两段各自有序)
        structured = sorted(
            [r for r in records
             if r.functional_category == "standard" and r.semantic_category in STRUCTURED_CATEGORIES],
            key=lambda r: r.rid,
        )
        rest = sorted(
            [r for r in records
             if r.functional_category == "standard" and r.semantic_category not in STRUCTURED_CATEGORIES],
            key=lambda r: r.rid,
        )
        picked = structured + rest

    if limit is not None:
        if limit <= 0:
            raise ValueError("limit 必须为正")
        picked = picked[:limit]
    if not picked:
        raise ValueError(f"子集 {subset} 过滤后为空")
    return picked


def summary_counts(records: list[GoalRecord]) -> dict[str, dict[str, int]]:
    by_functional: dict[str, int] = {}
    by_semantic: dict[str, int] = {}
    for r in records:
        by_functional[r.functional_category] = by_functional.get(r.functional_category, 0) + 1
        by_semantic[r.semantic_category] = by_semantic.get(r.semantic_category, 0) + 1
    return {"functional": by_functional, "semantic": by_semantic}


def write_jsonl(records: list[GoalRecord], path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    return len(records)


def read_jsonl(path) -> list[GoalRecord]:
    out: list[GoalRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(GoalRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError) as e:
                raise ValueError(f"{path}:{ln} 记录不合法: {e}") from e
    if not out:
        raise ValueError(f"{path} 没有任何记录")
    return out
