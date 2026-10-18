# 无害文本建模与语义拆分示例

这里公开 4 条手写无害文本的语义图及其拆分结果，共 7 个片段，包含中文、英文、共享对象、数量、否定约束、请求框架及产物场景。这些是演示与测试数据，不是模型实验结果，也不是另行准备的 80 条无害文本数据。

| 文件 | 内容 |
|---|---|
| `inputs.jsonl` | 原始无害文本及记录标识 |
| `graphs.jsonl` | 手写的证据对齐语义图 |
| `run.meta.json` | 示例来源、记录数量和输入指纹 |
| `fragments.jsonl` | 片段、原文证据、约束作用范围及跨片段关系 |
| `split_summary.json` | 4 条记录、7 个片段的拆分统计 |
| `manifest.json` | 数据文件大小与 SHA256 |

在仓库根目录离线复现，使用新的输出目录：

```powershell
python -m semantic_split split --input examples/benign-graphs/graphs.jsonl --out semantic_split/runs/public-demo
```

也可以通过图建模模块的入口刷新本目录中的两份派生拆分文件：

```powershell
python -m semantic_graph split --run examples/benign-graphs
```

程序只校验结构和原文证据、保存片段引用；结构校验通过不等于人工确认的语义准确。程序详情见 [semantic_split 说明](../../semantic_split/README.md)。
