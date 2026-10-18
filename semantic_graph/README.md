# 语义图建模

本模块提供 `目标文本 → DeepSeek 结构化抽取 → 原文证据校验 → 类型化语义图 → 报告`。仅抽取输入已有的信息，不补全隐含过程。图结构见 [SPEC.md](SPEC.md)。Python 3.10+，运行与离线测试只使用标准库。

首轮 200 条实测和独立重复见 [2026-10-15 实验记录](experiments/2026-10-15.md)：80 条通过结构校验、103 条拒答、17 条结构无效。双人人工参考图仍待完成，不能据此报告忠实度。

性能优化与离线回放见 [优化记录](experiments/optimization-2026-10-15.md)：结果写入量减少 98.1%；回放得到 74 条完整图和 15 条待复核部分图，拒答 103 条、无效 8 条。回放不构成新的模型实验。

## 快速运行

以下命令在项目仓库根目录执行。

```powershell
python -m unittest discover -s semantic_graph/tests -v
python -m semantic_graph prepare
python -m semantic_graph probe --key-stdin
python -m semantic_graph extract --input semantic_graph/data/harmbench-v1/dev10.jsonl --limit 5 --out semantic_graph/runs/dev5 --key-stdin
```

可在当前终端配置 `DEEPSEEK_API_KEY`，或使用 `--key-stdin` 在终端隐藏输入。密钥只保存在当前进程环境中，不写入文件。不要将密钥放入命令行参数、提示词或 Git。运行目录、原始数据、缓存和 `.env` 已加入忽略规则。

默认接口 `https://api.deepseek.com`，模型 `deepseek-flash`，关闭思考、温度 0、JSON 对象输出、单次上限 8192 tokens、并发 2、超时 90 秒。`--base-url / --model / --temperature / --max-tokens / --timeout / --retries` 可配置。JSON 格式合法不能替代本地证据和关系检查。

默认模型与开关参考 [DeepSeek 官方接口说明](https://api-docs.deepseek.com/)、[JSON 输出](https://api-docs.deepseek.com/guides/json_mode/)、[思考模式](https://api-docs.deepseek.com/guides/thinking_mode/)。模型别名可能更新，运行记录同时保存请求模型、响应模型、完整提示词、配置、输入指纹、当前源码指纹和仓库提交号，不宣称云端权重可被冻结。

## 无害文本建图后直接拆分

对无害文本使用 `extract --split`，建图和证据校验完成后会直接调用 `semantic_split` 的 Python 函数，生成动作或产物片段：

```powershell
python -m semantic_graph extract --input semantic_split/benign_texts.jsonl --out semantic_graph/runs/benign --key-stdin --split
```

流程为 `无害文本 → 原有语义图抽取与校验 → semantic_split → 片段及跨片段关系`。输出目录保留原有 `graphs.jsonl`、报告及输入快照，并新增：

- `fragments.jsonl`：逐条片段、数量与约束的归属、共享对象、跨片段关系及原文证据。
- `split_summary.json`：记录数、片段数、拆分状态、输入指纹及来源建图目录。

已有无害文本建图目录可以直接离线调用，无需密钥或新增模型请求：

```powershell
python -m semantic_graph split --run semantic_graph/runs/benign
```

这一步不修改原图、抽取元数据或缓存；重复执行会刷新两份派生拆分文件。`extract --split` 也可接续相同配置的已有抽取，已完成记录无需重新请求模型。拆分独立于抽取指纹，不改变第一阶段的续跑身份。

所有输入在拆分文件中各保留一条结果；上游 `partial` 不提升为完整，`refused/invalid/error` 标记为 `skipped`，中途停止而未抽取的输入标记为 `not_processed`。含这些失败或未处理状态时命令返回码为 2；全部结果为 `ok/partial` 时返回 0。结构保留不代表人工确认的语义正确。

片段规范、离线示例和独立程序见 [semantic_split 使用说明](../semantic_split/README.md)。

## 数据与人工标注

`prepare` 从 HarmBench 官方仓库读取当前 commit，再按该 commit 下载数据，保存 commit、blob SHA、字节 SHA256 与筛选统计。也可以通过 `--source 官方.csv --out 新目录` 使用本地文件。为保护冻结样本，不会覆盖非空的准备目录。

输出 `standard200.jsonl`、`sample20.jsonl`、`dev10.jsonl`、`holdout10.jsonl` 和 `manifest.json`。样本按 6 类分层，网络、化生、非法、一般危害各 4 条，虚假信息、骚扰各 2 条，随机种子 42，各类均分开发/留出集。

本次人工标注统一使用仓库的 [annotations 目录](annotations/README.md)。`annotator_a.jsonl` 与 `annotator_b.jsonl` 为**独立、未填充机器图的盲标模板**，其中所有图仍待标注。重复的空白副本已删除。请两位项目成员独立填写 graph，再讨论分歧，将最终结果写入 `adjudicated.jsonl`。在新的输出目录运行 `prepare` 时，会为新样本生成对应的标注包。

```json
"reference_type": "human",
"annotation": {"status": "adjudicated", "annotators": ["成员A", "成员B"], "notes": "分歧与处理说明"}
```

图可以使用规范中的 `quote + occurrence`，字符位置会在评估时重新计算校验。`graph` 不得为 null。程序不能验证标注者身份或两人是否真正独立，团队必须按实际情况填写。机器抽取结果与助手查看结果均不能直接冒充双人复核参考图。

开发集用于修改规范和提示词；留出集在冻结版本后运行，结果不能反向用于调参。节点和关系采用严格类型及原文跨度匹配的 micro P/R/F1，同时报告根与约束召回。自由图结构的一致性不应直接套用分类 Kappa。

## 全量抽取与报告

```powershell
python -m semantic_graph extract --input semantic_graph/data/harmbench-v1/dev10.jsonl --out semantic_graph/runs/dev10 --key-stdin
python -m semantic_graph extract --input semantic_graph/data/harmbench-v1/holdout10.jsonl --out semantic_graph/runs/holdout-r0 --key-stdin
python -m semantic_graph extract --input semantic_graph/data/harmbench-v1/standard200.jsonl --out semantic_graph/runs/standard200-v2 --key-stdin
python -m semantic_graph report --run semantic_graph/runs/standard200
python -m semantic_graph replay --run semantic_graph/runs/standard200 --out semantic_graph/runs/replay-new --cache semantic_graph/cache
```

每个 run 保存输入快照、提示词、运行指纹、逐条图、结构覆盖率报告及 `graph_viewer.html`。打开 HTML 可选择目标、查看图、点击节点和关系核对原文证据。报告全链无需密钥、网络或 GPU。风险、攻击成功率和语义忠实度均不从结构覆盖率推导。

缓存按原文、ID、影响响应的模型配置、提示词、图规范、解析器版本及独立重复编号索引；修改超时或网络重试次数不会使其失效。相同输入与抽取实现可续跑，已完成记录直接从结果日志恢复，无需重新请求；仅丢弃异常中断留下的末尾未完成行，文件中部损坏会报错。失败调用可续跑重试。每条结果追加写入一次，结束时按输入顺序整理一次；并发任务完成后立即补位。报告和浏览器代码的更新不影响抽取续跑。输出目录的指纹不一致时须使用新目录。

关系证据已能精确定位、但未覆盖两端节点时，本地只将其扩大为覆盖原证据和两端节点的最小连续原文片段；不修改节点、类型或关系方向。结果一律标记 `partial`，记录修复前后证据，待人工复核。其他结构错误最多请求模型修复一次。模型拒答直接保留；鉴权、余额、模型名称等系统性错误发生后停止派发新任务，在途任务最多为配置的并发数。

`replay` 按已保存响应的原顺序验证新解析器，不调用模型、不计新增 tokens，也不充当独立重复。可用 `--cache` 把回放结果迁移到新版缓存。当前全量 200 条及两轮独立重复的旧响应已迁移；无需重新付费运行。原始 `standard200` 保留供对照，优化回放位于 `standard200-optimized`。

## 忠实度评估与独立重复

```powershell
python -m semantic_graph evaluate --run semantic_graph/runs/standard200-optimized --gold semantic_graph/annotations/adjudicated.jsonl --split holdout --out semantic_graph/runs/standard200-optimized/fidelity.json
python -m semantic_graph extract --input semantic_graph/data/harmbench-v1/holdout10.jsonl --repeat-id 1 --out semantic_graph/runs/holdout-r1 --key-stdin
python -m semantic_graph extract --input semantic_graph/data/harmbench-v1/holdout10.jsonl --repeat-id 2 --out semantic_graph/runs/holdout-r2 --key-stdin
python -m semantic_graph consistency --runs semantic_graph/runs/holdout-r0 semantic_graph/runs/holdout-r1 semantic_graph/runs/holdout-r2 --out semantic_graph/runs/holdout-consistency.json
```

未完成双人复核的参考图会被评估命令拒绝。缺失或失败的预测按空图参与召回率分母，不静默丢弃。重复编号必须不同，相同缓存的回放不计为独立重复。一致性报告同时给出全部结果状态一致率和有效图的一致率，避免“连续拒答”被误读为图稳定。

