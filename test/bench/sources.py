"""输入读取:jsonl / json 数组 / csv / 纯文本(一条一行)。

只负责把各种来源规范成 ResponseRecord 列表;不 import torch。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .schema import ResponseRecord

DEFAULT_TEXT_FIELDS = ("response", "text", "content", "answer", "output")
DEFAULT_ID_FIELDS = ("id", "rid", "sample_id", "index")


def _pick(keys, candidates, explicit=None):
    if explicit:
        if explicit not in keys:
            raise ValueError(f"指定字段 {explicit!r} 不在列/键 {sorted(keys)} 中")
        return explicit
    for c in candidates:
        if c in keys:
            return c
    return None


def load_records(path, text_col: str | None = None, id_col: str | None = None,
                 where: dict | None = None):
    """读取输入文件,返回 list[ResponseRecord]。

    支持:
      - .jsonl / .ndjson:每行一个对象,文本字段默认 response/text/content/answer/output
      - .json:对象数组,同上
      - .csv:文本字段必须用 --text-col 指定(或命中默认列名)
      - 其余后缀:纯文本,一条回复一行(空行跳过),id = 行号

    where: {列名: 期望值} 行过滤(仅对象格式),先过滤再做 id 唯一性检查 ——
    同一题常有多个变体共用 id(如 baseline/figstep),按变体拆开后才各自唯一。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"输入文件不存在: {p}")
    suf = p.suffix.lower()

    if suf in (".jsonl", ".ndjson"):
        objs = _iter_jsonl(p)
    elif suf == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{p}: .json 期望是对象数组")
        objs = data
    elif suf == ".csv":
        # utf-8-sig:带 BOM 的 CSV(Excel 导出常见)第一列列名不能被 BOM 污染
        with p.open(encoding="utf-8-sig", newline="") as f:
            objs = list(csv.DictReader(f))
    else:
        if where:
            raise ValueError("where 行过滤只支持对象格式(jsonl/json/csv),纯文本不支持")
        return _load_lines(p)

    if where:
        objs = _apply_where(list(objs), where, p)
    return _load_objects(objs, p, text_col, id_col)


def _apply_where(objs: list, where: dict, p: Path) -> list:
    """按 {列: 值} 过滤。键不存在 / 过滤后为空都直接报错 —— 静默给空集会让人跑出 0 条还不知为什么。"""
    sample = next((o for o in objs if isinstance(o, dict)), None)
    if sample is not None:
        missing = [k for k in where if k not in sample]
        if missing:
            raise ValueError(f"{p}: --where 的列不存在 {missing};现有列: {sorted(sample)}")
    kept = [
        o for o in objs
        if isinstance(o, dict) and all(str(o.get(k)) == str(v) for k, v in where.items())
    ]
    if not kept:
        raise ValueError(f"{p}: --where {where} 过滤后为空(0 条)")
    return kept


def _iter_jsonl(p: Path):
    with p.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{lineno} JSON 解析失败: {e}") from e


def _load_objects(objs, p: Path, text_col, id_col):
    records: list[ResponseRecord] = []
    seen_ids: set[str] = set()
    for i, obj in enumerate(objs, 1):
        if not isinstance(obj, dict):
            raise ValueError(f"{p}: 第 {i} 条不是对象: {type(obj).__name__}")
        keys = set(obj.keys())
        tf = _pick(keys, DEFAULT_TEXT_FIELDS, text_col)
        if tf is None:
            raise ValueError(
                f"{p}: 第 {i} 条没有文本字段(找过 {DEFAULT_TEXT_FIELDS}),"
                f"可用 --text-col 指定;现有键: {sorted(keys)}"
            )
        text = obj.get(tf)
        if text is None:
            raise ValueError(f"{p}: 第 {i} 条文本字段 {tf!r} 为 null")
        text = str(text)
        cf = _pick(keys, DEFAULT_ID_FIELDS, id_col)
        rid = str(obj[cf]) if cf else f"row{i:06d}"
        if rid in seen_ids:
            raise ValueError(f"{p}: 重复 id {rid!r}(第 {i} 条);id 必须唯一,否则缓存与分母会串")
        seen_ids.add(rid)
        extra = {k: v for k, v in obj.items() if k not in (tf, cf)}
        records.append(ResponseRecord(rid=rid, response=text, extra=extra))
    if not records:
        raise ValueError(f"{p}: 没有读到任何记录")
    return records


def _load_lines(p: Path):
    records = []
    with p.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            records.append(ResponseRecord(rid=f"line{lineno:06d}", response=line.rstrip("\n")))
    if not records:
        raise ValueError(f"{p}: 没有读到任何非空行")
    return records
