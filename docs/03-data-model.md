# 03 · 数据模型

所有模块的公共接口。原方案没有数据模型,所以各阶段之间无法解耦、也无法做归因 —— 这是它最缺的东西。

---

## 1. 结构

```
AttackSpec
  behavior_id    : str            # HarmBench BehaviorID,全局唯一 join key
  goal           : str            # 原始有害指令
  fragments      : [Fragment]
  text_payload   : str            # 最终送给目标模型的文本
  image_payload  : Image          # 最终送给目标模型的图像
  trace          : {stage → config, seed, 中间产物, 耗时}

Fragment
  id             : str
  units          : [SemanticUnit]
  deps           : [fragment_id]  # 依赖边,切分时不得切断
  sensitivity    : float          # 探针测得的有害度
  serializable   : bool           # 文本化后信息是否完整保留

SemanticUnit
  id             : str
  triple         : (subj, action, obj)
  surface_text   : str            # 原始表述
  reframed_text  : str            # 语境重构后(非委婉化)
  visual_spec    : str | None     # 待渲染的视觉细节
  sensitivity    : float
```

---

## 2. 两个字段承载全部设计意图

### `sensitivity` —— 把"片段无害"从前提变成测量值

原方案把"拆分后的片段是无害的"当**前提**。这里把它变成**每个片段上一个可测量的量**,由探针分类器对每个 unit 单独测出。

这一个字段同时解决三件事:

1. 攻击侧的调参信号
2. 防御侧的对照
3. 论文核心图:片段数 $N$ vs 各片段 `sensitivity` 分布

### `serializable` —— 把模态分配从启发式变成可判定属性

`serializable = True` 表示该片段的信息在文本化后完整保留;`False` 表示文本化会丢失它。

M4 的分配规则直接由这个字段驱动:**`serializable = False` → 图像**。详见 [04-modules.md](04-modules.md) M4。

> 本节的字段语义不变。类型化属性图、`span`、探针版本等**接口增量的提案**见 [10-semantic-graph-modeling.md](10-semantic-graph-modeling.md) §7;`serializable` 归 M1 还是 M4 的职责划分见 [09](09-open-questions.md) R10。

---

## 3. 一处必须坚持的做法:按 `BehaviorID` 对齐

HarmBench 文件的 `BehaviorID`(如 `dimethylmercury_instructions_home`)是**稳定的全局键**。所有下游产物 —— 判官输出、中间结果、消融配置 —— 一律按它 join。

不要用行号或行为文本做键:

- 行号会因过滤/重排而错位
- 行为文本可能重复,且含逗号换行

---

## 4. 数据集注意事项

`harmbench_behaviors_text_all.csv` 与 `advbench_harmful_behaviors.csv` 的字段结构、解析陷阱与切分方案见 [05-experimental-setup.md](05-experimental-setup.md) §3。

---

## 相关文档

- 模块如何读写这些结构 → [04-modules.md](04-modules.md)
- 数据集与切分 → [05-experimental-setup.md](05-experimental-setup.md)
