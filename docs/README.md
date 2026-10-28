# 方案文档索引

黑盒跨模态语义拆分攻击 —— 设计与实施方案。

## 一句话定义

在**无梯度访问、单次提交**的约束下,把一条有害指令分解为不可文本化的信息与文本信息,分别安置在图像与文本通道,使输入在三级判据下均无害,而在目标模型的联合表征中具备攻击语义。

## 文档

| 文件 | 内容 |
|---|---|
| [**00-overview.md**](00-overview.md) | **总述 —— 全部设计的完整总结,建议从这里开始** |
| [01-threat-model.md](01-threat-model.md) | 约束、三级无害判据、威胁模型 |
| [02-mechanism.md](02-mechanism.md) | 信息从哪来、规避的是什么、编码期 vs 生成期 |
| [03-data-model.md](03-data-model.md) | 数据结构规范 |
| [04-modules.md](04-modules.md) | 模块分解 M1–M7 |
| [05-experimental-setup.md](05-experimental-setup.md) | 台架、四个实验条件、数据切分 |
| [06-roadmap.md](06-roadmap.md) | 分阶段计划 P0–P6 |
| [07-p1-spec.md](07-p1-spec.md) | **P1 执行规格(当前重点)** |
| [08-design-rationale.md](08-design-rationale.md) | 设计依据:原方案的问题与取舍 |
| [09-open-questions.md](09-open-questions.md) | 未决问题与风险 |
| [**10-semantic-graph-modeling.md**](10-semantic-graph-modeling.md) | **语义图建模(M1 / M2 / 探针)的备选实现方案 —— 进入 P2 前先读** |
| [**11-p4-experiment-protocol.md**](11-p4-experiment-protocol.md) | **P4 实验协议:以 LLaVA-1.5-7B 为靶重做三步 + 判据 A/B/C3 与互补性矩阵** |
| [**12-vector-graph-splitting.md**](12-vector-graph-splitting.md) | **向量建图与切分(10 的 A9):可行性判定 + 实现;先给出 E/R 类型与结构上限的实测界** |

## 当前状态

- 阶段:**P1 之前**,尚未开始任何实现
- 下一步:执行 P1-a(见 [07-p1-spec.md](07-p1-spec.md))
- 阻塞项:构建期是否允许查询目标模型(见 [09-open-questions.md](09-open-questions.md) R1)
- 方案补充:第 1 步(语义图建模)的备选路线与判据已整理为 [10-semantic-graph-modeling.md](10-semantic-graph-modeling.md),供 P2 采用
- 实验协议:靶子已定位为 LLaVA-1.5-7B(云端部署),三步重做与判据见
  [11-p4-experiment-protocol.md](11-p4-experiment-protocol.md);实现为 `semantic_graph/sg/{visual,recon}.py`
- 向量建图路线:见 [12-vector-graph-splitting.md](12-vector-graph-splitting.md)。**已测得的界**:
  68 条里 R 45 / E 23;E 类身份在实体名上,任何文本切分都不可能让每片无害;R 类 41 条为工作面,
  但其并集侧符号判据恒空转 ⇒ 结论只能靠读数。实现为 `semantic_graph/sg/embedgraph.py` 与
  `semantic_graph/tools/{s0_screen,embed_probe}.py`
  与 `semantic_graph/tools/cloud_probe.py`

## 建议阅读顺序

**先读 [00-overview.md](00-overview.md)** —— 它把整条设计线串成一篇,读完即可掌握全局。

之后按需深入:

```
00 总述
 ├─ 08 为什么这样设计(删了什么、为什么)   ← 建议紧接着读,防止重新引入已排除的组件
 ├─ 01 约束与三级判据 → 02 机制理论
 ├─ 04 模块细节 → 05 实验台架
 ├─ 10 语义图建模:备选路线 → 03 数据模型 / 09 未决问题   ← 进 P2 前先定这一步
 ├─ 11 P4 实验协议(LLaVA-1.5-7B 三步重做 + 判据)      ← 动手跑实验前读
 ├─ 12 向量建图与切分:A9 的实现 + 已测出的界          ← 想用编码器向量切分时先读
 └─ 06 阶段路线 → 07 P1 怎么执行            ← 当前动作
```

回查具体设计时按编号索引即可。

## 相关文件

- `intro.md` —— 原始设计草案(中文)。其中 GCG 后缀与 PGD 扰动两个组件已因黑盒约束删除,见 `08`。
- `data_origin/harmbench_behaviors_text_all.csv` —— 主数据集,见 `05`。
- `data_origin/advbench_harmful_behaviors.csv` —— 第二测试集,见 `05`。
- `semantic_graph/` —— 第 1 步(语义图建模)工作流的最小实现与测试,见其中的 `README.md`。
- `benchmark/` —— 评测台架相关文件(判官测试、Linux venv、本地 guard 权重)。**该目录已于 2026-10-15 删除,现位于 Recycle Bin,可从资源管理器恢复**;当前检出不含它,`bench` 包原本也缺失,见 `docs/10-semantic-graph-modeling.md` §5.7。
