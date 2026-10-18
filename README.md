# VLM-Jailbreak

## 语义图建模

第一阶段实现目标文本的忠实语义图抽取，使用 DeepSeek 接口，并通过原文证据定位与结构校验保存结果。

- [使用说明](semantic_graph/README.md)
- [图规范与接口](semantic_graph/SPEC.md)
- [首轮 200 条实测与后续安排](semantic_graph/experiments/2026-10-15.md)
- [性能优化与离线回放](semantic_graph/experiments/optimization-2026-10-15.md)

在仓库根目录运行 `python -m semantic_graph --help`。模块仅依赖 Python 标准库，人工参考标注与机器抽取结果分开保存。

原有 `test/bench` 保持为模型回复风险评测工具；语义图质量通过独立参考图评价。

## 无害文本的语义拆分

`semantic_split` 将证据对齐的语义图拆分为动作或产物片段，保留数量、否定、条件、共享对象与跨片段关系。已有图可离线处理；纯文本可显式使用 `--extract` 复用图抽取器。

- [程序说明、输入格式与输出字段](semantic_split/README.md)
- 离线示例：`python -m semantic_split demo --out semantic_split/runs/demo`
- 原有图建模入口：`python -m semantic_graph extract --input semantic_split/benign_texts.jsonl --out semantic_graph/runs/benign --key-stdin --split`
- 已有无害文本语义图：`python -m semantic_graph split --run semantic_graph/runs/benign`
- 测试：`python -m unittest discover -s semantic_split/tests -v`

可直接查看 [无害文本建模数据与拆分结果](examples/benign-graphs/README.md)，包含 4 条手写示例图和 7 个片段，无需模型密钥即可离线复现。

## 已有实验数据

[standard200](semantic_graph/runs/standard200) 保留原有 HarmBench 实验的 200 条输入、建模结果、报告、可视化页面及运行快照。建模状态为 80 条 `ok`、103 条 `refused`、17 条 `invalid`；这批历史数据与上面的无害文本示例分开存放。

已有拆分结果位于 [split-resultsl](semantic_graph/runs/standard200/split-resultsl)，目录名沿用本地名称。200 条输出中，73 条拆分完成、7 条部分完成、120 条跳过，共 206 个片段；详情见 [summary.json](semantic_graph/runs/standard200/split-resultsl/summary.json)。原始失败状态和拆分状态分别保留，上传不会重新调用模型或重新生成结果。
