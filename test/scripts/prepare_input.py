#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 CSV / jsonl / json 转成**标准输入** jsonl:每行 {"id", "response", "extra"}。

这是数据准备工具,不是评测程序的一部分(不 import torch、不判定)。它只做三件事:
  1. 指定/识别文本列与 id 列(识别规则与 bench 完全一致,复用 bench.sources)
  2. 按列值过滤(同一题有多个变体共用 id 时,必须先按变体拆开)
  3. 严格校验(id 唯一、文本字段存在),并写出去

用法:
    PY=YuFeng/.venv/bin/python
    $PY scripts/prepare_input.py --in 原始.csv --out inputs/xxx.jsonl --where variant=baseline
    $PY scripts/prepare_input.py --in 原始.csv --out inputs/yyy.jsonl --where variant=figstep \\
         --id-col sample_id --text-col response
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.sources import load_records  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="转换成标准输入 jsonl(id + response + extra)")
    ap.add_argument("--in", dest="src", required=True, help="源文件:.csv / .jsonl / .json")
    ap.add_argument("--out", required=True, help="输出 jsonl 路径")
    ap.add_argument("--text-col", default=None, help="文本列(默认自动识别 response/text/…)")
    ap.add_argument("--id-col", default=None, help="id 列(默认自动识别 id/rid/sample_id/index)")
    ap.add_argument("--where", action="append", default=[], metavar="COL=VAL",
                    help="只保留该列等于该值的行,可重复;如 --where variant=baseline")
    ap.add_argument("--dry-run", action="store_true", help="只校验并打印摘要,不写文件")
    args = ap.parse_args(argv)

    where = {}
    for w in args.where:
        if "=" not in w:
            raise SystemExit(f"--where 需要 COL=VAL 形式: {w!r}")
        k, v = w.split("=", 1)
        where[k] = v

    recs = load_records(args.src, text_col=args.text_col, id_col=args.id_col, where=where or None)

    lens = [len(r.response) for r in recs]
    n_empty = sum(1 for r in recs if r.is_empty)
    print(f"[prepare] {args.src}"
          + (f"  where={where}" if where else "")
          + f"\n  记录 {len(recs)} 条 | id 唯一 ✓ | 例 id {recs[0].rid!r}"
          f"\n  回复长度: min {min(lens)} 中位 {int(statistics.median(lens))} max {max(lens)}"
          f"\n  空回复 {n_empty} 条(会被判定为 empty_response 并排除出分母)")
    if recs[0].extra:
        print(f"  extra 字段 {len(recs[0].extra)} 个: {sorted(recs[0].extra)[:8]}"
              f"{' …' if len(recs[0].extra) > 8 else ''}(原样保留进快照,不参与判定)")

    if args.dry_run:
        print("  --dry-run:未写文件")
        return 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(
                {"id": r.rid, "response": r.response, "extra": r.extra}, ensure_ascii=False
            ) + "\n")
    print(f"  → 已写出 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
