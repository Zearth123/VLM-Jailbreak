# 05 · 实验台架

---

## 1. 台架结构:护栏段是攻击目标

由 [01](01-threat-model.md) §3 已确认:该设计规避的是**外部输入护栏**,不是目标模型自身的对齐 —— 对齐读的就是那个联合表征。因此台架**必须包含护栏段**,否则前提没有被检验。

```
(T, I) ──→ [护栏段] ──→ [目标 VLM] ──→ 输出 ──→ [HarmBench 判官]
             ↑ 攻击目标         ↑ 攻击目标(但 L3 不规避它)
```

## 2. 四个实验条件

| 条件 | 护栏段 | 作用 |
|---|---|---|
| **A** | 无 | 测攻击**是否有效**(能否让模型配合)。**这是总门** |
| **B** | 文本护栏过 `T` | 测对单模态文本护栏的规避 |
| **C** | 文本护栏过 `T` + `I` 的 caption | 测对 L2 级护栏的规避 |
| **D** | 多模态护栏(若有) | 上界对照 |

**逻辑关系:A 不成立,B/C/D 全部无意义。**

因此 P0/P1 只做 A。这也是为什么 P1 可以在没有任何护栏实现的情况下先跑 —— 见 [07-p1-spec.md](07-p1-spec.md)。

**护栏实现建议**:文本护栏用 Llama-Guard-3(公开、标准)。条件 C 的 captioner 用最强可用多模态模型(见 [01](01-threat-model.md) §2)。

---

## 3. 数据集

### 3.1 文件与结构

| 文件 | 行数 | 字段 |
|---|---|---|
| `harmbench_behaviors_text_all.csv` | 400 | `Behavior`, `FunctionalCategory`, `SemanticCategory`, `Tags`, `ContextString`, `BehaviorID` |
| `advbench_harmful_behaviors.csv` | 520 | `goal`, `target` |

**`harmbench_behaviors_text_all.csv` 是 UTF-8 带 BOM** —— 必须用 `utf-8-sig` 打开,否则首列名会变成 `﻿Behavior`。

### 3.2 类别分布(主数据集)

| `FunctionalCategory` | 数量 |
|---|---|
| `standard` | 200 |
| `copyright` | 100 |
| `contextual` | 100 |

| `SemanticCategory` | 数量 |
|---|---|
| `copyright` | 100 |
| `cybercrime_intrusion` | 67 |
| `illegal` | 65 |
| `misinformation_disinformation` | 65 |
| `chemical_biological` | 56 |
| `harassment_bullying` | 25 |
| `harmful` | 22 |

### 3.3 字段使用注意

| 字段 | 注意 |
|---|---|
| `ContextString` | **仅** 100 条 `contextual` 行有值;其余为空字符串,**不是 NaN** —— 不要依赖 `isna()` |
| `Tags` | **仅** 200 条 `standard` 行有值;同上 |
| `BehaviorID` | 稳定主键,**所有 join 都用它**(见 [03](03-data-model.md) §3) |

### 3.4 两个数据集的角色

- **HarmBench** —— 主数据集。类别体系完整,便于分组报告。
- **AdvBench** —— 第二测试集,分布不同,用于测泛化。两个都跑。

---

## 4. 数据切分

**必须切 dev/test,否则报的是对判官的过拟合,不是攻击效力。**

- dev 100 / test 300,按 `FunctionalCategory` 分层,保证三类在两边都有代表。
- 按 `BehaviorID` 切,记录切分清单,不做二次调整。

**切分的必要性来自黑盒约束**:所有调参都是经验性的,你会大量对着判官调。不切分,结果不可解释。

---

## 5. 判官伪影的控制

**这是本方案最容易被审稿人攻击的一点,必须在实验设计阶段就处理。**

风险:如果攻击靠"把信息藏在 caption 丢失的细节里"生效,那么 HarmBench 的判官(Llama-2-13b 通用模型)同样读不到那些细节。于是测到的 ASR 里,**有一部分是判官无能,不是攻击有效**。

**对照做法**:同一批输入,用弱判官与一个能看到图的**强判官**各判一次,报两者之差。

差距就是判官伪影的量。**这个对照不做,结果无法解释。**

---

## 6. 靶子选型

| 类型 | 示例 | 作用 |
|---|---|---|
| 早期弱模型 | LLaVA-1.5-7B | 验证攻击在原理上能否成立 |
| 当前模型 | Qwen2.5-VL-7B 或 InternVL2.5 | 验证迁移性 |

**两个都报才诚实。** 只报前者属于选择性呈现。

---

## 相关文档

- 威胁模型与三级判据 → [01-threat-model.md](01-threat-model.md)
- 分阶段计划 → [06-roadmap.md](06-roadmap.md)
- P1 用哪部分数据 → [07-p1-spec.md](07-p1-spec.md)
