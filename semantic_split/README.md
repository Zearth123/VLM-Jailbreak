# 无害文本的语义拆分

本模块实现 `无害文本 → 语义图 → 带原文证据的语义片段`。输入已有语义图时完全离线；输入纯文本时，通过显式指定 `--extract` 复用现有 DeepSeek 抽取器。Python 3.10+，仅依赖标准库。

语义片段按动作或明确产物组织，包含相关对象、数量、条件和约束。共享对象可以出现在多个片段中，片段之间的时序、目的、请求框架等关系单独保存。输出保留完整原文和原有元数据，可作为后续普通文本、场景或图文表达的结构输入。

## 直接运行离线示例

在仓库根目录执行：

```powershell
python -m semantic_split demo --out semantic_split/runs/demo
```

无需网络、密钥或 GPU。内置 4 条手写的中英文无害示例图，共生成 7 个片段。手写示例用于演示与测试，不是模型实测结果，也不是独立人工标注的 gold。

中文示例：

> 先把两个红色杯子放在桌上，不要叠放，然后拍摄杯子。

图中两个动作分别形成片段：

| 片段 | 包含的语义元素 | 跨片段关系 |
|---|---|---|
| 放置 | 放、两个、红色杯子、桌上、不要叠放 | 放置 → 拍摄：`precedes` |
| 拍摄 | 拍摄、第二处出现的杯子 | 保留前述时序关系 |

“两个”继续归属于杯子，“不要叠放”继续作用于放置。它们不会变成脱离对象或动作的独立碎片。

## 拆分自己的无害文本

直接输入一段文本：

```powershell
python -m semantic_split split --text "先清洗杯子，然后擦干杯子，不要使用洗洁精。" --extract --key-stdin --out semantic_split/runs/my-text
```

批量输入 JSONL：

```powershell
python -m semantic_split split --input semantic_split/benign_texts.jsonl --extract --key-stdin --out semantic_split/runs/my-batch
```

JSONL 每行一条记录，最少需要 `id` 和 `text`，例如：

```json
{"id":"text-1","text":"先清洗杯子，然后擦干杯子。","category":"benign"}
```

也接受 JSON 数组文件，或将 UTF-8 `.txt` 文件整体作为一条记录。文本中的空白、换行及 Unicode 字符保持原样，批量 ID 必须唯一。

`--extract` 会将输入文本发送给配置的 DeepSeek 接口，并可能产生模型费用。密钥可用 `DEEPSEEK_API_KEY` 环境变量配置；`--key-stdin` 在交互终端中隐藏输入，在管道中读取标准输入第一行。程序不把密钥写入文件。使用纯文本而未指定 `--extract` 时会提示缺少语义图。

模型、接口和请求限制可以配置：

```powershell
python -m semantic_split split --input semantic_split/benign_texts.jsonl --extract --model deepseek-flash --timeout 90 --max-tokens 8192 --key-stdin --out semantic_split/runs/custom-model
```

默认复用现有图抽取模块的模型配置与缓存身份。缓存位于 `semantic_split/cache/`，相同输入与配置可复用抽取结果。已有 `partial/refused/invalid/error` 记录按原状态处理，不会自动重抽。

## 从原有图建模入口调用

原有 `semantic_graph` 入口已直接集成此模块。对无害文本建图时增加 `--split`：

```powershell
python -m semantic_graph extract --input semantic_split/benign_texts.jsonl --out semantic_graph/runs/benign --key-stdin --split
```

已有的无害文本建图目录可以离线追加拆分：

```powershell
python -m semantic_graph split --run semantic_graph/runs/benign
```

这两个入口在建图目录内生成 `fragments.jsonl` 和 `split_summary.json`。重复执行可刷新派生文件，保留 `graphs.jsonl`、原始输入、抽取元数据及模型缓存。新抽取记录直接交给拆分函数，不需要启动第二个终端程序或额外请求模型。独立的 `python -m semantic_split` 入口仍将结果写入新的独立目录。

在 Python 中处理已有建图目录：

```python
from semantic_split.pipeline import split_graph_run

results, summary = split_graph_run("semantic_graph/runs/benign")
```

## 已有语义图的离线输入

输入可以直接使用现有 `semantic_graph` 格式的无害记录，包含 `id`、`text`、`graph`，以及可选的 `status`、`issues`、`category` 等字段：

```powershell
python -m semantic_split split --input semantic_split/runs/demo/inputs.jsonl --out semantic_split/runs/offline
```

没有 `--extract` 时不会创建模型客户端或发起模型调用。输入图会重新执行原文证据、节点类型、关系方向及端点检查；不合法的图会被记录为失败。

Python 接口：

```python
from semantic_split import split_record
from semantic_split.examples import benign_examples

record = benign_examples()[0]  # 换成自己的无害文本及其证据对齐语义图
result = split_record(record)
for fragment in result["fragments"]:
    print(fragment["id"], fragment["display_text"], fragment["node_ids"])
```

## 输出结构

每次运行输出到新的目录：

- `inputs.jsonl`：本次输入快照。
- `fragments.jsonl`：逐条拆分结果，保留原文、元数据和校验后的完整图。
- `summary.json`：输入数量、片段数量、上游状态和拆分状态统计。

输出目录非空时拒绝覆盖；请使用新的 `--out` 路径。

逐条结果的新增字段：

| 字段 | 含义 |
|---|---|
| `split_version` | 拆分算法版本 |
| `source_graph_sha256` | 校验后完整图的指纹，有效图才有此字段 |
| `split_status` / `split_issues` | 本次拆分状态与待复核原因 |
| `fragments` | 语义片段列表 |
| `edge_index` | 为原图关系附加稳定的 `e1/e2/...` 编号 |
| `cross_relations` | 无法归入任何单个片段的关系，保留证据和两端片段 ID |
| `shared_node_ids` | 被多个片段引用的对象或修饰节点 |
| `constraint_scopes` | 约束实际修饰的节点及其所在片段，不自行扩大作用范围 |
| `coverage` | 输入图节点及关系的保留数量 |

每个片段包含：

```text
id / kind / anchor_id       片段编号、类型、中心动作或产物
root_ids                    本片段包含的原图目标根
node_ids / edge_ids         在完整 graph 和 edge_index 中引用的节点与内部关系
source_spans                合并重叠后仍可不连续的原文证据，每段都有 quote/start/end
display_text                用省略号连接证据，仅供展示
source_context              覆盖本片段节点证据的最小连续原文范围
```

`source_context` 为核对上下文提供便利，可能包含其他片段的内容；`display_text` 也不是自然语言改写。程序的正式语义输出是节点分组及关系引用。不同片段可以共享证据，不要求对原文字符做互斥分区。后续使用时应通过 `node_ids` 和 `edge_ids` 读取完整语义，不应仅使用展示文字。

偏移采用 Python Unicode 字符的 `[start, end)` 区间。JavaScript 使用 UTF-16 码元索引时需转换。

## 拆分规则与状态

1. 每个明确动作建立一个片段，吸收其对象、参与者、工具、地点、来源和产物节点。
2. 数量、条件和约束沿原图中明确的修饰关系进入对应片段；共享修饰可以出现在多个片段中。
3. 动作间的关系保留在 `cross_relations` 中；并列动作不自动增加先后关系。
4. 尚未归入动作片段的根产物建立产物片段。
5. 剩余节点作为 `context` 片段保存，并标记 `partial`，提示复核其语义归属。
6. 所有有效输入图的节点、关系和根保持可重建。片段编号依据原文证据位置生成，不代表执行顺序。

| `split_status` | 含义 |
|---|---|
| `ok` | 输入图通过校验，全部图元素保留且未发现游离语义节点 |
| `partial` | 上游图为部分结果，或存在待复核的游离节点 |
| `skipped` | 保留上游 `refused/invalid/error`，不生成片段 |
| `missing_graph` | 库接口收到无图的纯文本，需要先抽取 |
| `invalid` | 记录或语义图未通过校验 |
| `not_processed` | 前序调用发生致命接口错误，本条尚未发起抽取 |

批次中结构错误的图不会导致其他记录被静默丢弃。接口鉴权、余额等致命错误会停止后续抽取，并为未处理记录保存显式状态。程序返回码 0 表示所有记录均为 `ok/partial`；其余结果或输入错误返回 2。

图结构保留率不等于语义正确率。程序依赖上游图是否忠实表达原文，无法从合法字符位置证明关系含义正确；当前图规范也没有独立的共指和颜色属性关系，不会推断原文没有给出的过程或自动补全遗漏属性。

## 验证

```powershell
python -m unittest discover -s semantic_split/tests -v
python -m unittest discover -s semantic_graph/tests -v
```

测试覆盖中文与英文、Unicode 与重复证据、数量与否定作用范围、共享实体与约束、产物与目的、请求框架、多目标关系、完整图重建，以及离线运行、模拟抽取、缓存、密钥处理、失败状态和批量输入。测试不调用真实模型。
