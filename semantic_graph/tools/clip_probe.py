#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M1(CLIP 侧):文本向量 → 图像 的可反演性探针。

回答的问题:**"文本对应的向量序列能否倒过来生成对应的图像"**

三件事,全部只用 CLIP(不含靶子 LLaVA,那部分在云上跑):
  A. 相似度基准:排版图 / 真实自然图 / 空白噪声 分别能靠近目标文本嵌入到什么程度
  B. **梯度反演**:直接优化像素,使 `CLIP_img(I) ≈ CLIP_text(T)`,看
       - 能压到多高的余弦
       - 反演出来的图**长什么样**(存 PNG)
  C. 特异性:反演图到底还原了"这一条语义",还是只到"某类语义"——
       对多条目标文本做排序,报 top-1 命中率(CLIP 层的迷你 R2)

用法:
  set PYTHONPATH=D:\\VLMjalibreak\\.pylibs
  python semantic_graph\\tools\\clip_probe.py --steps 400
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

CLIP_ID = "openai/clip-vit-large-patch14-336"
CANVAS = 336  # LLaVA-1.5 / CLIP ViT-L/14-336 的输入边长:24x24 patch
FONT_PTS = (12, 16, 20, 28, 40)
FONT_CANDIDATES = ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
                   "C:/Windows/Fonts/DejaVuSans.ttf"]
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[1]           # semantic_graph/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ----------------------------------------------------------------- 数据

def harmbench():
    from sg import sources
    return list(sources.load_harmbench())


def build_targets(rid: str):
    """返回 (goal, 投影文本, [(nid,label,span,surface)])"""
    from sg import extract
    rec = next(r for r in harmbench() if r.rid == rid)
    g = extract.build_graph(rid, rec.goal)
    spans = [(n.nid, n.label, n.span, rec.goal[n.span[0]:n.span[1]]) for n in g.nodes]
    spans.sort(key=lambda x: x[2][0])
    proj = " ".join(s for _, _, _, s in spans)
    return rec.goal, proj, spans


# ----------------------------------------------------------------- 图像

def render_text(text: str, font_pt: int, rotate: float = 0.0, contrast: float = 1.0):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
    d = ImageDraw.Draw(img)
    font = None
    for p in FONT_CANDIDATES:
        if pathlib.Path(p).exists():
            font = ImageFont.truetype(p, font_pt)
            break
    if font is None:
        font = ImageFont.load_default(size=font_pt)
    lines, cur = [], ""
    for w in text.split():
        probe = (cur + " " + w).strip()
        if d.textlength(probe, font=font) <= CANVAS - 20:
            cur = probe
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    lh = font_pt + 6
    y = max(8, (CANVAS - lh * len(lines)) // 2)
    grey = int(round(20 + (255 - 20) * (1.0 - contrast)))
    for ln in lines:
        x = max(8, int((CANVAS - d.textlength(ln, font=font)) // 2))
        d.text((x, y), ln, font=font, fill=(grey, grey, grey))
        y += lh
    if rotate:
        img = img.rotate(rotate, expand=False, fillcolor=(255, 255, 255))
    return img


def blank():
    from PIL import Image
    return Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))


def noise(seed=0):
    import random
    from PIL import Image
    rng = random.Random(seed)
    img = Image.new("RGB", (CANVAS, CANVAS))
    img.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256))
                 for _ in range(CANVAS * CANVAS)])
    return img


# ----------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rid", default="safrole_oil_extraction_guide")
    ap.add_argument("--natural-dir", default=str(ROOT / "artifacts" / "inv_probe"))
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts" / "inv_probe"))
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--tv", type=float, default=0.0, help=">0 时加全变差正则(更自然的解)")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} | clip={CLIP_ID}")
    model = CLIPModel.from_pretrained(CLIP_ID).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_ID)
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)

    goal, proj_text, spans = build_targets(args.rid)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def t_emb(texts: list[str]):
        with torch.no_grad():
            inp = proc(text=texts, return_tensors="pt", padding=True, truncation=True)
            return model.get_text_features(**{k: v.to(dev) for k, v in inp.items()})

    def i_emb_pil(images: list):
        with torch.no_grad():
            inp = proc(images=images, return_tensors="pt")
            return model.get_image_features(**{k: v.to(dev) for k, v in inp.items()})

    def i_emb_tensor(img_t: "torch.Tensor"):
        return model.get_image_features(pixel_values=(img_t - mean) / std)

    def cos(a, b):
        return float(F.cosine_similarity(a, b, dim=-1).mean())

    # ---------- 文本目标
    texts = {"goal": goal, "projected_text": proj_text}
    for nid, label, span, surf in spans:
        texts[f"span:{label}"] = surf
    others = [r for r in harmbench() if r.rid != args.rid][:8]
    for r in others:
        texts[f"other:{r.rid[:26]}"] = r.goal
    texts["benign"] = "Describe how to bake bread at home step by step."

    temb = {k: t_emb([v])[0] for k, v in texts.items()}
    print(f"文本目标 {len(texts)} 条(含 8 条其它 goal + 1 条良性)")

    # ---------- A. 相似度基准
    cands: dict[str, object] = {}
    for key in ("goal", "projected_text"):
        for pt in FONT_PTS:
            cands[f"typographic:{key}:pt{pt}"] = render_text(texts[key], pt)
    for nid, label, span, surf in spans:
        for pt in (20, 28, 40):
            cands[f"typographic:span:{label}:pt{pt}"] = render_text(surf, pt)
    cands["typographic:goal:pt28:rot45"] = render_text(goal, 28, rotate=45)
    cands["typographic:goal:pt28:lowcontrast"] = render_text(goal, 28, contrast=0.25)
    nat = []
    for p in sorted(pathlib.Path(args.natural_dir).glob("natural_*.jpg")):
        try:
            cands[f"natural:{p.stem}"] = Image.open(p).convert("RGB")
            nat.append(f"natural:{p.stem}")
        except Exception as exc:  # noqa: BLE001
            print("skip", p.name, exc)
    cands["control:blank"] = blank()
    cands["control:noise"] = noise()

    iemb = {k: i_emb_pil([v])[0] for k, v in cands.items()}
    sims: dict[str, dict[str, float]] = {}
    for tk in ("goal", "projected_text"):
        for kind in ("typographic", "natural", "control"):
            rows = {k: round(cos(temb[tk], v), 4) for k, v in iemb.items() if k.startswith(kind + ":")}
            if rows:
                sims[f"{tk} vs {kind}"] = dict(sorted(rows.items(), key=lambda kv: -kv[1]))

    # 文本-文本自相似:给相似度一个量纲基准
    tkeys = list(temb)
    tt = {f"{a} | {b}": round(cos(temb[a], temb[b]), 4)
          for i, a in enumerate(tkeys) for b in tkeys[i + 1:]}
    tt_sorted = dict(sorted(tt.items(), key=lambda kv: -kv[1]))

    # ---------- B. 梯度反演
    def tv_loss(x):
        return (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean() + \
               (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()

    def invert(target_text: str, init: str, steps: int, tag: str):
        t = temb.get(target_text)
        if t is None:
            t = t_emb([target_text])[0]
        with torch.no_grad():
            if init == "noise":
                base = torch.rand(1, 3, CANVAS, CANVAS, device=dev)
            else:
                pil = Image.open(init).convert("RGB").resize((CANVAS, CANVAS))
                base = torch.tensor(list(pil.getdata()), device=dev, dtype=torch.float32)
                base = base.view(CANVAS, CANVAS, 3).permute(2, 0, 1).unsqueeze(0) / 255.0
        img = base.clone().requires_grad_(True)
        opt = torch.optim.Adam([img], lr=args.lr)
        hist = []
        for step in range(steps):
            opt.zero_grad()
            loss = -F.cosine_similarity(i_emb_tensor(img), t, dim=-1).mean()
            if args.tv > 0:
                loss = loss + args.tv * tv_loss(img)
            loss.backward()
            opt.step()
            with torch.no_grad():
                img.clamp_(0, 1)
            if step % max(1, steps // 8) == 0 or step == steps - 1:
                with torch.no_grad():
                    c = cos(i_emb_tensor(img), t)
                hist.append({"step": step, "cos": round(c, 4)})
        with torch.no_grad():
            final_cos = cos(i_emb_tensor(img), t)
            fin = (img[0].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
        path = out_dir / f"inverted_{tag}.png"
        Image.fromarray(fin).save(path)
        # 特异性:该反演图对各文本的相似度排序
        with torch.no_grad():
            ie = i_emb_tensor(img)
            rank = {k: round(cos(ie, v), 4) for k, v in temb.items()}
        rank_sorted = dict(sorted(rank.items(), key=lambda kv: -kv[1]))
        top1 = next(iter(rank_sorted))
        return {"tag": tag, "target": target_text[:60], "init": init, "steps": steps,
                "final_cos": round(final_cos, 4), "history": hist,
                "png": str(path), "top1": top1, "ranking": rank_sorted}

    inversions = []
    for target in ("goal", "projected_text"):
        for init_kind, init_val in (("noise", "noise"), ("natural", nat[0] if nat else "noise")):
            tag = f"{target}_{init_kind}"
            print(f"[invert] {tag} ...")
            inversions.append(invert(target, init_val if init_kind == "natural" else "noise",
                                     args.steps, tag))
    if args.tv > 0:
        print("[invert] goal_noise_tv ...")
        inversions.append(invert("goal", "noise", args.steps, "goal_noise_tv"))

    report = {
        "rid": args.rid, "clip_id": CLIP_ID, "device": dev, "canvas": CANVAS,
        "goal": goal, "projected_text": proj_text, "text_max_len": int(proc.tokenizer.model_max_length),
        "similarities": sims, "text_text_selfsim_top": dict(list(tt_sorted.items())[:12]),
        "inversions": inversions,
        "n_natural": len(nat),
    }
    out = ROOT / "artifacts" / "clip_probe.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 打印要点
    print("\n=== A. goal 对排版图(前 5) ===")
    for k, v in list(sims.get("goal vs typographic", {}).items())[:5]:
        print(f"  {v:+.4f}  {k}")
    print("=== A. goal 对自然图 ===")
    for k, v in list(sims.get("goal vs natural", {}).items())[:6]:
        print(f"  {v:+.4f}  {k}")
    print("=== A. goal 对控制图 ===")
    for k, v in sims.get("goal vs control", {}).items():
        print(f"  {v:+.4f}  {k}")
    print("=== A. 文本-文本自相似(基准) ===")
    for k, v in list(tt_sorted.items())[:5]:
        print(f"  {v:+.4f}  {k}")
    print("=== B. 反演结果 ===")
    for r in inversions:
        print(f"  {r['tag']:26s} cos={r['final_cos']:+.4f}  top1={r['top1']:28s} -> {pathlib.Path(r['png']).name}")
        print("       ranking:", {k: v for k, v in list(r["ranking"].items())[:4]})
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
