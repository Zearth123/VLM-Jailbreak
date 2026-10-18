"""CPU-only coverage reports and a self-contained evidence/graph browser."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from .io import write_json
from .pipeline import load_run

VIEWER = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>语义图与原文证据</title>
<style>
body{font:15px/1.65 system-ui,sans-serif;background:#f2f5f7;color:#203342;margin:0}main{max-width:1300px;margin:auto;padding:28px}
h1{font-size:25px;margin:0 0 8px}p{margin:8px 0}.muted{color:#617586}select{width:100%;padding:12px;font:inherit;margin:16px 0;border:1px solid #b8c8d0;border-radius:8px}
.card{background:white;border-radius:12px;padding:20px;margin:14px 0;box-shadow:0 2px 10px #1831440b}#source{white-space:pre-wrap;font-size:17px}mark{background:#ffe39c}
#canvas{width:100%;min-height:200px}svg text{font:12px system-ui;pointer-events:none}.scroll{overflow:auto}table{width:100%;border-collapse:collapse}td,th{padding:9px;text-align:left;border-bottom:1px solid #e5ecef;vertical-align:top}button{font:inherit;border:0;border-radius:5px;padding:4px 8px;cursor:pointer;background:#e8f0f4;color:#214b60}.node{cursor:pointer}
</style><main><h1>目标文本 → 语义图</h1><p class="muted">机器抽取结果。证据可定位表示原文匹配通过；关系含义仍需人工复核。选择节点或关系可查看原文证据。</p>
<select id="pick" aria-label="选择样本"></select><div id="info"></div><div class="card"><strong>输入原文</strong><p id="source"></p><p id="evidence" class="muted"></p></div>
<div class="card scroll"><strong>图结构</strong><p class="muted">粗边框为根节点。连线完整含义见下方关系表。</p><svg id="canvas" role="img" aria-label="语义图"></svg></div>
<div class="card"><strong>关系与证据</strong><div class="scroll"><table><thead><tr><th>起点</th><th>关系</th><th>终点</th><th>原文证据</th></tr></thead><tbody id="edges"></tbody></table></div></div>
<div class="card"><strong>待复核问题</strong><p id="issues"></p></div></main>
<script id="data" type="application/json">__DATA__</script><script>
const rows=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id);
const colors={action:'#d5ecf7',entity:'#e5ecf4',artifact:'#d8eee2',quantity:'#f9e9c6',condition:'#e8dff5',constraint:'#f6dcdb'};
let current, sourceChars;
function evidence(ev){const t=sourceChars, s=$('source');s.replaceChildren();s.append(document.createTextNode(t.slice(0,ev.start).join('')));const m=document.createElement('mark');m.textContent=t.slice(ev.start,ev.end).join('');s.append(m,document.createTextNode(t.slice(ev.end).join('')));$('evidence').textContent=`字符范围 [${ev.start}, ${ev.end}) · 第 ${ev.occurrence} 次出现`;}
function svg(tag,attrs,text){const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e;}
function show(i){current=rows[i];sourceChars=Array.from(current.text);$('source').textContent=current.text;$('evidence').textContent='';$('info').textContent=current.id+' · '+current.category+' · '+current.status;$('issues').textContent=current.issues.join('；')||'抽取器未报告问题；尚需人工审核。';$('edges').replaceChildren();$('canvas').replaceChildren();if(!current.graph)return;
const g=current.graph, nodes=Object.fromEntries(g.nodes.map(n=>[n.id,n])), positions={}, counts=[0,0,0];
for(const n of g.nodes){const c=n.type==='action'?0:['entity','artifact'].includes(n.type)?1:2;positions[n.id]={x:35+c*365,y:55+counts[c]++*82};}
const h=Math.max(200,Math.max(...counts)*82+55);$('canvas').setAttribute('viewBox',`0 0 1100 ${h}`);
const defs=svg('defs',{}), marker=svg('marker',{id:'arrow',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:6,markerHeight:6,orient:'auto-start-reverse'});marker.append(svg('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:'#8b9da8'}));defs.append(marker);$('canvas').append(defs);
for(const e of g.edges){const a=positions[e.source], b=positions[e.target],right=a.x<=b.x,ax=a.x+(right?287:-2),ay=a.y+27.5,bx=b.x+(a.x===b.x?292:right?-7:292),by=b.y+27.5,cx=a.x===b.x?a.x+335:(ax+bx)/2;const p=svg('path',{d:`M${ax},${ay} C${cx},${ay} ${cx},${by} ${bx},${by}`,fill:'none',stroke:'#9db1bc','stroke-width':1.4,'marker-end':'url(#arrow)'});const title=svg('title',{},e.type);p.append(title);p.onclick=()=>evidence(e.evidence);$('canvas').append(p);const tr=document.createElement('tr');for(const v of [nodes[e.source].surface_text,e.type,nodes[e.target].surface_text]){const td=document.createElement('td');td.textContent=v;tr.append(td);}const td=document.createElement('td'),bt=document.createElement('button');bt.textContent=e.evidence.quote;bt.onclick=()=>evidence(e.evidence);td.append(bt);tr.append(td);$('edges').append(tr);}
for(const n of g.nodes){const p=positions[n.id],group=svg('g',{class:'node',tabindex:0,role:'button','aria-label':n.surface_text});group.append(svg('rect',{x:p.x,y:p.y,width:285,height:55,rx:9,fill:colors[n.type],stroke:g.root_ids.includes(n.id)?'#2f6377':'#bdcbd2','stroke-width':g.root_ids.includes(n.id)?3:1}));group.append(svg('text',{x:p.x+12,y:p.y+19},n.id+' · '+n.type));const label=n.surface_text.length>37?n.surface_text.slice(0,36)+'…':n.surface_text;group.append(svg('text',{x:p.x+12,y:p.y+40},label));group.append(svg('title',{},n.surface_text));group.onclick=()=>evidence(n.evidence);group.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();evidence(n.evidence);}};$('canvas').append(group);}}
rows.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${i+1}. ${r.id} [${r.status}]`;$('pick').append(o);});$('pick').onchange=e=>show(Number(e.target.value));if(rows.length)show(0);
</script></html>'''


def build_report(run_dir):
    import json

    rows, meta = load_run(run_dir)
    status = Counter(r["status"] for r in rows)
    categories = defaultdict(Counter)
    usage = Counter()
    models = set()
    for row in rows:
        categories[row["category"]]["total"] += 1
        categories[row["category"]][row["status"]] += 1
        for attempt in row.get("attempts", []):
            if attempt.get("model"):
                models.add(attempt["model"])
            if not row.get("cache_hit") and not row.get("replayed"):
                for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    usage[name] += attempt.get("usage", {}).get(name, 0) or 0
    n = meta["n_input"]
    metrics = {"n_input": n, "n_processed": len(rows), "n_unprocessed": n - len(rows),
               "complete": meta["complete"], "statuses": dict(status),
               "coverage_ok": status["ok"] / n if n else None,
               "coverage_including_partial": (status["ok"] + status["partial"]) / n if n else None,
               "refusal_rate": status["refused"] / n if n else None,
               "cache_hits": sum(bool(r.get("cache_hit")) for r in rows),
               "replayed": sum(bool(r.get("replayed")) for r in rows),
               "locally_repaired": sum(bool(r.get("repairs")) for r in rows),
               "usage_this_run": dict(usage), "response_models": sorted(models),
               "by_category": {c: dict(counts) for c, counts in sorted(categories.items())},
               "fidelity": None, "human_reference_status": "pending",
               "note": "Coverage measures structural validation, not semantic correctness or risk reduction."}
    write_json(Path(run_dir) / "metrics.json", metrics)
    lines = ["# 语义图抽取报告", "", "**本报告为机器抽取的工程统计。人工参考图尚未完成，忠实度、P/R/F1 待评估。**", "",
             *(["**本次为已有响应的离线解析回放，新增模型调用为 0；不构成独立重复实验。**", ""] if meta.get("mode") == "replay" else []),
             f"输入 {n} 条；已处理 {len(rows)} 条；未处理 {n - len(rows)} 条。",
             f"完整抽取且结构校验通过：{status['ok']}/{n}；部分抽取：{status['partial']}/{n}；拒答：{status['refused']}/{n}。",
             f"结构无效：{status['invalid']}；调用失败：{status['error']}；缓存命中：{metrics['cache_hits']}。", "",
             "原文证据匹配不能证明关系含义正确，不能据此报告切分收益或攻击成功率。", "",
             "| 类别 | 已处理 | 完整抽取 | 部分抽取 | 拒答 | 无效 | 错误 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for category, counts in sorted(categories.items()):
        values = [counts[k] for k in ("total", "ok", "partial", "refused", "invalid", "error")]
        lines.append("| " + category + " | " + " | ".join(map(str, values)) + " |")
    lines += ["", "## 复现信息", "", f"- 请求模型：`{meta['config']['model']}`；响应模型：`{', '.join(sorted(models)) or '无有效响应'}`。",
              f"- 图规范：`{meta['schema_version']}`；提示词：`{meta['prompt_version']}`。",
              f"- 本轮新增调用返回的 token 用量：`{dict(usage)}`（不含缓存历史用量，不等同最终账单）。",
              "- 图与原文交互检查：打开同目录的 `graph_viewer.html`。", "",
              "## 待复核记录", ""]
    problems = [r for r in rows if r["status"] != "ok"]
    for row in problems:
        issue = "; ".join(row["issues"]).replace("\n", " ")
        lines.append(f"- `{row['id']}` — {row['status']}: {issue}")
    if not problems:
        lines.append("抽取器未报告失败或部分结果；仍需人工复核全部参考样本。")
    (Path(run_dir) / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    safe_rows = [{k: r[k] for k in ("id", "text", "category", "status", "graph", "issues")} for r in rows]
    data = json.dumps(safe_rows, ensure_ascii=False).replace("<", "\\u003c")
    (Path(run_dir) / "graph_viewer.html").write_text(VIEWER.replace("__DATA__", data), encoding="utf-8")
    return metrics

