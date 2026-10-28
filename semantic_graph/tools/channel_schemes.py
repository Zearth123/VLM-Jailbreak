#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""枚举"子图 → 通道"的注入方案,并把每个条目物化(文本条目 / 图像条目)。

一个方案 = 若干条目,每条是 `(sid, node_ids, channel)`;`channel ∈ {text, image}`。
  · text 条目:投影文本 = 该组单元**原文 span** 的拼接(逐字,不改写)
  · image 条目:A0 排版渲染(A0 只画载荷片段;`drawn_strings` 是审计字段)

产物:
    artifacts/schemes_<rid>/scheme.json   每个方案的条目清单(文本 + 图片路径 + 分片数)
    artifacts/schemes_<rid>/scheme.md     对照表
    artifacts/schemes_<rid>/images/*.png  图像条目的渲染结果

    python tools/channel_schemes.py --rid dji_drone_geofencing_hack \
        --split "n003,n001|n002,n004" --out artifacts/schemes_dji

把本脚本产出的图片目录交给 `tools/l2_check.py` 就能测 caption(L2)。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from sg import extract, project as P, sources  # noqa: E402
from sg import visual as V  # noqa: E402

# 每个方案 = [(sid, [nid...], channel)]
SCHEMES: dict[str, list[tuple[str, list[str], str]]] = {
    # 双文本基线
    "s0_text_text": [("SG-A", [], "text"), ("SG-B", [], "text")],
    # 一片文本 + 一片图像
    "s1_text_image": [("SG-A", [], "text"), ("SG-B", [], "image")],
    "s2_image_text": [("SG-A", [], "image"), ("SG-B", [], "text")],
    # 双图像
    "s3_image_image": [("SG-A", [], "image"), ("SG-B", [], "image")],
}
# 需要逐单元指定归属的方案(跨通道分散)
SPLIT_SCHEMES: dict[str, list[tuple[str, list[str], str]]] = {
    "s4_cross": [("SG-A1", ["n003"], "text"), ("SG-A2", ["n001"], "image"),
                 ("SG-B1", ["n002"], "text"), ("SG-B2", ["n004"], "image")],
    "s5_verbs_text_nouns_image": [("SG-A", ["n003"], "text"),
                                  ("SG-B", ["n004"], "text"),
                                  ("SG-C", ["n002", "n001"], "image")],
}


def build_items(scheme: dict, groups: list[list[str]], graph, spec: V.RenderSpec,
                out_dir: pathlib.Path, rid: str) -> list[dict]:
    items: list[dict] = []
    for idx, (sid, explicit, channel) in enumerate(scheme["entries"]):
        ids = explicit or groups[idx]
        carrier = {nid: (P.CARRIER_IMAGE if channel == "image" else P.CARRIER_TEXT)
                   for nid in ids}
        sg = P.project_subgraph(graph, ids, sid=sid, carrier_of=carrier, index=idx)
        row: dict = {"sid": sid, "channel": channel, "node_ids": ids,
                     "text": sg.rendered_text}
        if channel == "image":
            imgs = V.render_a0(sg, spec, rid)
            paths = []
            for im in imgs:
                name = f"{scheme['name']}_{sid}_{im.sha256[:8]}.png"
                (out_dir / "images" / name).write_bytes(im.png)
                paths.append((pathlib.Path("images") / name).as_posix())
            row["images"] = paths
            row["drawn_strings"] = [s for im in imgs for s in im.drawn_strings]
            row["payload_surfaces"] = [u["surface"] for u in V._image_units(sg)]
        items.append(row)
    return items


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rid", default="dji_drone_geofencing_hack")
    ap.add_argument("--goal", default=None,
                    help="直接给 goal 文本;不在 HarmBench 里时用(P1 式手工目标)")
    ap.add_argument("--units", default=None,
                    help="手工声明的图(JSON):规则抽取覆盖不了时用,见 build_manual_graph")
    ap.add_argument("--split", required=True, help="按 | 分组的节点,如 n003,n001|n002,n004")
    ap.add_argument("--schemes", default="s0_text_text,s1_text_image,s2_image_text,s3_image_image,"
                                        "s4_cross,s5_verbs_text_nouns_image")
    ap.add_argument("--out", required=True)
    ap.add_argument("--font-pt", type=int, default=28)
    ap.add_argument("--layout", default="caption", choices=("caption", "axis_label", "table", "legend"))
    args = ap.parse_args(argv)

    if args.units:
        spec_json = json.loads(pathlib.Path(args.units).read_text(encoding="utf-8"))
        goal = spec_json.get("goal") or args.goal
        if not goal:
            raise SystemExit("--units 的 JSON 里需要 goal(或另外给 --goal)")
        graph = extract.build_manual_graph(
            spec_json.get("rid") or args.rid, goal,
            [(u[0], u[1], (u[2][0], u[2][1])) for u in spec_json["units"]],
            [tuple(e) for e in spec_json.get("edges", [])],
            task_kind=spec_json.get("task_kind", "manual"))
    elif args.goal:
        goal = args.goal
        graph = extract.build_graph(args.rid, goal)
    else:
        rec = next((r for r in sources.load_harmbench() if r.rid == args.rid), None)
        if rec is None:
            raise SystemExit(f"HarmBench 里没有 {args.rid};用 --goal 直接给文本")
        goal = rec.goal
        graph = extract.build_graph(args.rid, goal)
    groups = [[x for x in part.split(",") if x] for part in args.split.split("|")]
    known = {n.nid for n in graph.nodes}
    for g in groups:
        for nid in g:
            if nid not in known:
                raise SystemExit(f"未知节点 {nid};可选 {sorted(known)}")

    out = pathlib.Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    for old in (out / "images").glob("*.png"):
        old.unlink()
    spec = V.RenderSpec(arm="a0", layout=args.layout, font_pt=args.font_pt)

    definitions: list[dict] = []
    for name in [s.strip() for s in args.schemes.split(",") if s.strip()]:
        if name in SCHEMES:
            template = SCHEMES[name]
            entries = [(sid, ids, ch) for (sid, ids, ch) in template]
            scheme = {"name": name, "entries": entries}
        elif name in SPLIT_SCHEMES:
            scheme = {"name": name, "entries": SPLIT_SCHEMES[name]}
        else:
            raise SystemExit(f"未知方案 {name};可选 {sorted(list(SCHEMES) + list(SPLIT_SCHEMES))}")
        # 模板方案的 groups 顺序与 entries 顺序一致;显式方案自带 ids
        entries = scheme["entries"]
        if all(not ids for _, ids, _ in entries):
            if len(entries) != len(groups):
                raise SystemExit(f"{name} 需要 {len(entries)} 组,但 --split 给了 {len(groups)} 组")
            entries = [(sid, groups[i], ch) for i, (sid, _, ch) in enumerate(entries)]
        scheme["entries"] = entries
        items = build_items(scheme, groups, graph, spec, out, args.rid)
        definitions.append({"scheme": name, "items": items,
                            "n_items": len(items),
                            "n_images": sum(len(i.get("images", [])) for i in items)})

    payload = {"rid": args.rid, "goal": graph.goal, "split": groups,
               "render": {"layout": args.layout, "font_pt": args.font_pt,
                          "renderer_version": V.RENDERER_VERSION},
               "schemes": definitions}
    (out / "scheme.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                     encoding="utf-8", newline="\n")

    L = [f"# 通道注入方案 · `{args.rid}`", "",
         f"- 划分:`{args.split}`", f"- 渲染:A0 layout={args.layout} font_pt={args.font_pt}",
         f"- 目标:`{graph.goal}`", "",
         "| 方案 | 条目 | 通道 | 单元 | 内容(文本 / 图像载荷) |", "|---|---|---|---|---|"]
    for d in definitions:
        for it in d["items"]:
            content = (it["text"] if it["channel"] == "text"
                       else f"image: {', '.join(it.get('drawn_strings', []))}")
            L.append(f"| `{d['scheme']}` | `{it['sid']}` | {it['channel']} | "
                     f"{', '.join(it['node_ids'])} | `{content}` |")
    L += ["", "**图像条目**:`drawn_strings` 必须是该组单元原文片段的逐字集合;",
          "多出的字符串即实现错误(A0 只画载荷)。", ""]
    (out / "scheme.md").write_text("\n".join(L), encoding="utf-8", newline="\n")
    print(json.dumps({d["scheme"]: d["n_items"] for d in definitions}, ensure_ascii=False))
    print(f"wrote {out / 'scheme.json'}\nwrote {out / 'scheme.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
