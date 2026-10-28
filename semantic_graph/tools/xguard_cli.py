#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YuFeng-XGuard-Reason-0.6B 的终端入口。

为什么需要这一层
----------------
模型目录里的 `xguard.py` 是**库**,而且它的默认路径是错的:

    MODEL_DIR = Path(__file__).parent / "models" / "YuFeng-XGuard-Reason-0.6B"

本机的权重就放在 `xguard.py` **同一个目录**里,没有 `models/` 子目录。所以必须显式传
`model_dir`。本脚本负责这件事,并把分类结果翻译成本仓库统一的判定格式,可直接喂
`tools/submission.py --guard`。

判定口径
--------
模型输出**首个 token** 就是风险类别 ID(`dw`、`ha`…),`sec` 表示 Safe;
`tokenizer_config.json` 的 `id2risk` 给出 ID → 全名映射。判定规则:

    top-1 类别 == `sec`            → safe
    其它任何类别                    → unsafe
    拿不到类别(解析/截断失败)     → unparsed(绝不算 safe)

    # 单条
    python tools/xguard_cli.py --text "hacking · geofencing"
    # 批量(与 guard_check.py / guard_local.py 同样的 items 格式)
    python tools/xguard_cli.py --items artifacts/local_guard_demo_items.json \
        --out artifacts/xguard_demo.jsonl
    # 带解释(多跑一次前向,标签仍取自不带解释的那次)
    python tools/xguard_cli.py --items ... --out ... --reason
    # 不带 torch 也能验:风险映射 + 条目装载 + 行格式
    python tools/xguard_cli.py --self-test
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parents[1]        # semantic_graph/
ROOT = HERE.parent                                        # 仓库根
DEFAULT_MODEL = ROOT / ".models" / "YuFeng-XGuard-Reason-0.6B"
SAFE_ID = "sec"

TIPS = ("需要 torch + transformers(本模型要求 transformers >= 4.51):\n"
        "  $py   = 'C:\\Users\\Zearth\\.dsh\\dsh-runtimes\\dsh-primary-runtime\\dependencies\\python\\python.exe'\n"
        "  $libs = 'D:\\VLMjalibreak\\.pylibs'\n"
        "  & $py -m pip install --upgrade --target $libs torch "
        "--index-url https://download.pytorch.org/whl/cpu\n"
        "  & $py -m pip install --upgrade --target $libs 'transformers>=4.51' accelerate\n"
        "  $env:PYTHONPATH = $libs")


class GuardUnavailable(RuntimeError):
    """缺 torch/transformers 或权重不可用。宁可失败,也不伪造判定。"""


# ---------------------------------------------------------------- 结果翻译


def is_safe_label(name: str) -> bool:
    """风险全名形如 `Safe-Safe` / `Crimes and Illegal Activities-Dangerous Weapons`。"""
    return name.strip().lower().startswith("safe")


def to_row(item_id: str, text: str, res: dict, *, kind: str = "", model: str = "",
           safe_id: str = SAFE_ID) -> dict:
    """把 `xguard.classify` 的返回值翻译成统一行。

    `risk_score` 为空 = 没拿到类别(预算不够/解析失败)⇒ `unparsed`。
    """
    risk: dict = res.get("risk_score") or {}
    tok: dict = res.get("token_score") or {}
    if not risk:
        return {"id": item_id, "text": text, "kind": kind, "harmful": None,
                "label": "unparsed", "score": None,
                "reason": str(res.get("response", ""))[:200],
                "categories": [], "gate": "unparsed", "model": model,
                "truncated": bool(res.get("truncated")), "token_score": tok}
    label, score = max(risk.items(), key=lambda kv: kv[1])
    harmful = not is_safe_label(label)
    return {"id": item_id, "text": text, "kind": kind, "harmful": harmful,
            "label": "unsafe" if harmful else "safe", "score": score,
            "reason": str(res.get("response", ""))[:200],
            "categories": [] if harmful is False else [label],
            "gate": "unsafe" if harmful else "safe", "model": model,
            "truncated": bool(res.get("truncated")), "token_score": tok,
            "risk_top": label}


EXPL_RE = re.compile(r"<explanation>(.*?)</explanation>", re.S)


def explanation_of(response: str) -> str:
    m = EXPL_RE.search(response or "")
    return (m.group(1).strip() if m else "")


# ---------------------------------------------------------------- IO


def load_items(path: pathlib.Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) else data
    if not items:
        raise SystemExit(f"[xguard] {path} 里没有 items")
    for i, it in enumerate(items):
        if "text" not in it:
            raise SystemExit(f"[xguard] 第 {i} 项缺 text")
        it.setdefault("id", f"item{i}")
    return items


def resolve_model_dir(p: str | pathlib.Path | None) -> pathlib.Path:
    d = pathlib.Path(p) if p else DEFAULT_MODEL
    if not (d / "config.json").exists():
        raise SystemExit(f"[xguard] 不是模型目录(缺 config.json):{d}")
    if not (d / "xguard.py").exists():
        raise SystemExit(f"[xguard] 缺 xguard.py:{d}")
    return d


def load_guard(model_dir: pathlib.Path, device: str, dtype: str):
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        raise GuardUnavailable(f"缺 torch/transformers({exc})\n{TIPS}") from exc
    import torch

    sys.path.insert(0, str(model_dir))
    import xguard  # 同目录里的官方封装

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    wants = [dtype] if dtype != "auto" else (["bfloat16", "float32"] if device != "cpu"
                                             else ["float32", "bfloat16"])
    last: Exception | None = None
    for name in wants:
        td = {"bfloat16": torch.bfloat16, "float32": torch.float32,
              "float16": torch.float16}[name]
        try:
            model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=td)
        except TypeError:                       # 老版本 transformers 用 torch_dtype
            model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=td)
        except Exception as exc:                # noqa: BLE001
            last = exc
            continue
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        return xguard, model, tok, device, name
    raise GuardUnavailable(f"加载失败:{last}\n{TIPS}")


# ---------------------------------------------------------------- 自检


def self_test() -> int:
    cases = [
        ({"risk_score": {"Safe-Safe": 0.99}, "response": "sec"}, "safe", False),
        ({"risk_score": {"Crimes and Illegal Activities-Dangerous Weapons": 0.97,
                         "Safe-Safe": 0.02}, "response": "dw"}, "unsafe", True),
        ({"risk_score": {"Cybersecurity-Hacker Attack": 0.8}}, "unsafe", True),
        ({"risk_score": {}}, "unparsed", None),
        ({"risk_score": {}, "truncated": True}, "unparsed", None),
    ]
    bad = 0
    for res, want_gate, want_h in cases:
        row = to_row("t", "x", res)
        ok = row["gate"] == want_gate and row["harmful"] is want_h
        bad += 0 if ok else 1
        print(f"  {'ok ' if ok else 'FAIL'} want={want_gate:9s} got={row['gate']:9s} "
              f"harmful={row['harmful']} risk_top={row.get('risk_top')}")
    expl = explanation_of("sec\n<explanation>It is a harmless keyword list.</explanation>")
    ok = expl.startswith("It is")
    bad += 0 if ok else 1
    print(f"  {'ok ' if ok else 'FAIL'} explanation parse -> {expl!r}")
    print(f"\n{len(cases) + 1 - bad}/{len(cases) + 1} 用例通过")
    return 0 if bad == 0 else 1


# ---------------------------------------------------------------- 主流程


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None, help=f"默认 {DEFAULT_MODEL}")
    ap.add_argument("--items", default=None, help="{\"items\":[{\"id\",\"text\"}]}")
    ap.add_argument("--text", action="append", default=None, help="单条文本,可重复")
    ap.add_argument("--out", default=None, help="jsonl 输出")
    ap.add_argument("--report", default=None, help="markdown(默认与 --out 同名 .md)")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--dtype", default="auto", choices=("auto", "bfloat16", "float32", "float16"))
    ap.add_argument("--policy", default=None, help="注入 chat template 的 policy 变量")
    ap.add_argument("--reason", action="store_true", help="额外跑一次前向取解释")
    ap.add_argument("--reason-tokens", type=int, default=96)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--self-test", action="store_true", help="不需要 torch")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    if args.items:
        items = load_items(pathlib.Path(args.items))
    elif args.text:
        items = [{"id": f"text{i}", "text": t} for i, t in enumerate(args.text)]
    else:
        ap.error("需要 --items 或 --text(或 --self-test)")

    model_dir = resolve_model_dir(args.model_dir)
    try:
        xguard, model, tok, device, dtype = load_guard(model_dir, args.device, args.dtype)
    except GuardUnavailable as exc:
        raise SystemExit(f"[xguard] {exc}")
    id2risk = xguard.get_id2risk(tok, model_dir)
    print(f"# model={model_dir.name} device={device} dtype={dtype} "
          f"n_categories={len(id2risk)} safe_id={SAFE_ID!r}", file=sys.stderr)

    rows: list[dict] = []
    for it in items:
        msgs = [{"role": "user", "content": it["text"]}]
        res = xguard.classify(model, tok, msgs, id2risk, max_new_tokens=1,
                              reason_first=False, policy=args.policy, topk=args.topk)
        row = to_row(it["id"], it["text"], res, kind=it.get("kind", ""),
                     model=model_dir.name)
        if args.reason:
            r2 = xguard.classify(model, tok, msgs, id2risk, max_new_tokens=args.reason_tokens,
                                 reason_first=True, policy=args.policy, topk=args.topk)
            row["explanation"] = explanation_of(r2.get("response", ""))
        row["latency_s"] = res.get("latency_s")
        rows.append(row)
        print(f"  {row['id']:26s} {row['gate']:9s} {str(row.get('risk_top', ''))[:52]} "
              f"p={row['score']} {row.get('explanation', '')[:40]}")

    out = pathlib.Path(args.out) if args.out else HERE / "artifacts" / "xguard_out.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    md = pathlib.Path(args.report) if args.report else out.with_suffix(".md")
    L = [f"# YuFeng-XGuard 判定 · `{model_dir.name}`", "",
         f"- 设备:{device} · dtype:{dtype} · top-1 类别:`{SAFE_ID}` = Safe",
         f"- 生成时间:{__import__('datetime').datetime.now().isoformat(timespec='seconds')}", "",
         "| 条目 | 判定 | 概率 | top-1 类别 | 文本 |", "|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| `{r['id']}` | **{r['gate']}** | {r['score']} | "
                 f"{str(r.get('risk_top', '—'))[:60]} | `{r['text'][:60]}` |")
    c = {k: sum(1 for r in rows if r["gate"] == k) for k in ("safe", "unsafe", "unparsed")}
    L += ["", f"**safe {c['safe']} / unsafe {c['unsafe']} / unparsed {c['unparsed']}**", "",
          "> `unparsed` 不算 safe。判定只对该模型成立,不外推到别的护栏。", ""]
    md.write_text("\n".join(L), encoding="utf-8", newline="\n")
    print(f"\nwrote {out}\nwrote {md}")
    print(f"safe {c['safe']} / unsafe {c['unsafe']} / unparsed {c['unparsed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
