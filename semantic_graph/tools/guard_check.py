#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""护栏检查:把人工构造的每个条目逐条送进判官,输出"是否被识别为有害"的表。

用途
----
逐片段无害是**硬要求**:每个子图单独提交时,护栏必须判 `safe`。本脚本就是这个门的执行器,
可反复用于 P1 式的人工构造-检查-修订循环。

    # 1) 把待检条目写进 json:{"items":[{"id":"SG-A","text":"..."}]}
    # 2) 跑判官(密钥只从环境变量读,绝不写入文件)
    $env:AI_KEY = '...'
    python tools/guard_check.py --items artifacts/p1_items.json \
        --judge http --judge-url https://<host>/v1/chat/completions --judge-model <model> \
        --out artifacts/p1_guard.jsonl

    # 无端点时:只用占位词法判官(结论**不作数**,仅用于跑通流程)
    python tools/guard_check.py --items ... --judge lexical

诚实边界
--------
- `--judge http` 是**LLM-as-guard 代理**,不是 ShieldGemma / Llama-Guard 这类专用护栏;
  报告里必须写清是哪一个模型判的(config 里有 `model` 与 `prompt_sha`)。
- `--judge lexical` 是加性词袋占位,**不是 sensitivity**;它的 `safe` 不构成任何证据。
- 判官给 `unparsed` 时记 `unparsed`,**不得记为 safe**。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from sg import judge as J  # noqa: E402


def load_items(path: pathlib.Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) else data
    if not items:
        raise SystemExit(f"[guard_check] {path} 里没有 items")
    for i, it in enumerate(items):
        if "id" not in it or "text" not in it:
            raise SystemExit(f"[guard_check] 第 {i} 项缺 id/text")
    return items


def verdict(row: dict) -> str:
    """把判官结论压成门要的一句话:只有明确 harmless 才算 safe。"""
    if row.get("harmful") is True:
        return "unsafe"
    if row.get("harmful") is False:
        return "safe"
    return "unparsed"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True, help="jsonl 输出")
    ap.add_argument("--report", default=None, help="markdown 表(默认与 --out 同名 .md)")
    ap.add_argument("--judge", default="lexical", choices=("lexical", "http", "llm", "http-llm",
                                                           "local", "local-hf", "xguard"))
    ap.add_argument("--judge-url", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-key-env", default="AI_KEY")
    ap.add_argument("--judge-model-dir", default=None,
                    help="本地 guard 权重目录(默认读 XGUARD_MODEL_DIR 或工作区 .models/)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--created", default="")
    args = ap.parse_args(argv)

    kwargs = {}
    if args.judge in ("http", "llm", "http-llm"):
        kwargs = {"url": args.judge_url, "model": args.judge_model,
                  "key_env": args.judge_key_env}
    elif args.judge in ("local", "local-hf", "xguard"):
        kwargs = {"model_dir": args.judge_model_dir, "device": args.device}
    try:
        judge = J.make_judge(args.judge, **kwargs)
    except J.JudgeUnavailable as exc:
        raise SystemExit(f"[guard_check] 判官不可用:{exc}")

    items = load_items(pathlib.Path(args.items))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for it in items:
            res = judge.judge_text(it["text"])
            detail = {}
            try:
                detail = json.loads(res.get("raw") or "{}")
            except (TypeError, ValueError):
                detail = {}
            row = {"id": it["id"], "text": it["text"], "kind": it.get("kind", ""),
                   "harmful": res.get("harmful"), "label": res.get("label"),
                   "score": res.get("score"), "reason": res.get("raw"),
                   "cat": detail.get("cat"), "p_risk": detail.get("p_risk"),
                   "p_safe": detail.get("p_safe"), "top3": detail.get("top3"),
                   "gate": verdict(res)}
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  {row['id']:28s} {row['gate']:9s} {str(row['reason'])[:90]}")

    md = pathlib.Path(args.report) if args.report else out.with_suffix(".md")
    L = [f"# 护栏检查 · {judge.describe().get('judge_id')}", ""]
    desc = judge.describe()
    L += [f"- 判官:`{desc.get('judge_id')}` v{desc.get('version')}(kind={desc.get('kind')}"
          f"{', model=' + str(desc.get('model')) if desc.get('model') else ''})",
          f"- authoritative:**{desc.get('authoritative')}**",
          f"- 生成时间:{args.created or '?'}", "",
          "| 条目 | 类别 | 判定 | harmful | cat | p_risk | score | 理由 |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        pr = r.get("p_risk")
        prs = "—" if pr is None else f"{pr:.4f}"
        L.append(f"| `{r['id']}` | {r['kind'] or '—'} | **{r['gate']}** | {r['harmful']} | "
                 f"{r.get('cat') or '—'} | {prs} | {r['score']} | {str(r['reason'])[:120]} |")
    n_safe = sum(1 for r in rows if r["gate"] == "safe")
    n_unsafe = sum(1 for r in rows if r["gate"] == "unsafe")
    n_unparsed = sum(1 for r in rows if r["gate"] == "unparsed")
    L += ["", f"**safe {n_safe} / unsafe {n_unsafe} / unparsed {n_unparsed}**(共 {len(rows)})", ""]
    if not desc.get("authoritative"):
        L += ["> ⚠️ 非权威判官:上表的 `safe` **不构成任何证据**,见 `sg/judge.py` 的 caveat。", ""]
    md.write_text("\n".join(L), encoding="utf-8", newline="\n")
    print(f"\nwrote {out}\nwrote {md}")
    print(f"safe {n_safe} / unsafe {n_unsafe} / unparsed {n_unparsed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
