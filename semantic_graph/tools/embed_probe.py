#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU 主机:把每条 goal 的 **token/片段向量** dump 出来,供本机 `sg/embedgraph.py` 建图。

为什么单独一个脚本
------------------
本机(办公机)没有 torch/CUDA,而"取隐藏状态"必须跑在有模型的机器上。契约与
`tools/cloud_probe.py` 同一风格:云上写 dump,本机使用向量,两边不共享显存。

    python tools/embed_probe.py --model hf-internal-testing/tiny-random-gpt2 \
        --jsonl artifacts/texts.jsonl --out dumps/embed --layer -1 --pooling mean

    输入 jsonl:{"rid": "...", "text": "..."}(每行一条)
    产出:
      emb.jsonl   {"text_index","rid","text","n_tokens","sha256"}
      emb.npz     vectors (T,d) f32 | offsets (T,2) i32 | text_id (T,) i32 | token_ids (T,) i32
      meta.json   {model_id, revision, layer, pooling, dtype, device, created}

治理约束(docs/08 §2.1 / docs/10 A7)
-----------------------------------
构建管线**不许依赖对齐模型**。本脚本因此默认拒绝 id 里带 instruct / chat / -it / guard 的模型;
确需消融时必须显式加 `--allow-aligned`,并会在 meta 里记 `aligned_override=true`。

**未在本机验证**:本机无 torch。缺依赖/权重时显式退出,绝不写占位向量。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import pathlib
import re
import sys

ALIGNED_HINTS = ("instruct", "chat", "-it", "_it", "guard", "rlhf", "dpo", "safety")


def _require(mod: str):
    try:
        return __import__(mod)
    except Exception as exc:  # pragma: no cover - 云上环境相关
        raise SystemExit(
            f"[embed_probe] 缺少依赖 {mod}: {exc}\n"
            "  云上需要: pip install torch transformers numpy\n"
            "  torch: pip install torch --index-url https://download.pytorch.org/whl/cu126"
        )


def load_texts(path: pathlib.Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"[embed_probe] 输入为空: {path}")
    for i, r in enumerate(rows):
        if not r.get("text"):
            raise SystemExit(f"[embed_probe] 第 {i} 行缺 text")
        r.setdefault("rid", f"row{i}")
    return rows


def looks_aligned(model_id: str) -> str | None:
    low = model_id.lower()
    for hint in ALIGNED_HINTS:
        if hint in low:
            return hint
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="**非对齐**编码器(CLIP 文本塔 / base LM)")
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=-1, help="取第几层 hidden(-1 = 最后一层)")
    ap.add_argument("--pooling", choices=("none", "mean", "last"), default="none",
                    help="none = 逐 token(本模块要的就是逐 token);mean/last 只用于池化向量")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--allow-aligned", action="store_true",
                    help="允许对齐模型(仅消融;默认拒绝,见 docs/08 §2.1)")
    args = ap.parse_args(argv)

    hint = looks_aligned(args.model)
    if hint and not args.allow_aligned:
        raise SystemExit(
            f"[embed_probe] 拒绝使用疑似对齐模型({hint} 命中 {args.model})。\n"
            "  构建管线不许依赖对齐模型(docs/08 §2.1);换 base/CLIP 文本塔,\n"
            "  或显式加 --allow-aligned 并把结果标成消融。"
        )

    torch = _require("torch")
    tr = _require("transformers")
    np = _require("numpy")

    rows = load_texts(pathlib.Path(args.jsonl))
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = tr.AutoTokenizer.from_pretrained(args.model)
    model = tr.AutoModel.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype)).to(args.device).eval()

    vectors, offsets, text_ids, token_ids = [], [], [], []
    meta_rows = []
    for ti, row in enumerate(rows):
        enc = tok(row["text"], return_offsets_mapping=True, return_tensors="pt",
                  truncation=True, max_length=args.max_tokens)
        offs = enc.pop("offset_mapping")[0].tolist()
        enc.pop("token_type_ids", None)
        enc = {k: v.to(args.device) for k, v in enc.items()}
        with torch.no_grad():
            res = model(**enc, output_hidden_states=True)
        h = res.hidden_states[args.layer][0]                    # (T, d)
        assert len(h) == len(offs), (len(h), len(offs))
        meta_rows.append({"text_index": ti, "rid": row["rid"], "text": row["text"],
                          "n_tokens": len(offs),
                          "sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest()[:16],
                          "truncated": bool(enc["input_ids"].shape[1] >= args.max_tokens)})
        vectors.append(h.float().cpu().numpy())
        offsets.extend([[int(a), int(b)] for a, b in offs])
        text_ids.extend([ti] * len(offs))
        token_ids.extend(enc["input_ids"][0].tolist())

    V = np.concatenate(vectors, axis=0).astype("float32")
    np.savez_compressed(out / "emb.npz", vectors=V,
                        offsets=np.asarray(offsets, dtype="int32"),
                        text_id=np.asarray(text_ids, dtype="int32"),
                        token_ids=np.asarray(token_ids, dtype="int32"))
    with open(out / "emb.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in meta_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"model_id": args.model, "revision": getattr(model.config, "_name_or_path", ""),
            "layer": args.layer, "pooling": args.pooling, "dtype": args.dtype,
            "device": args.device, "n_texts": len(rows), "n_tokens": int(V.shape[0]),
            "dim": int(V.shape[1]), "aligned_override": bool(args.allow_aligned and hint),
            "created": _dt.datetime.now().isoformat(timespec="seconds")}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                   encoding="utf-8", newline="\n")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
