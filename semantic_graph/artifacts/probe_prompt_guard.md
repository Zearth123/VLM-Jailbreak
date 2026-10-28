# 护栏检查 · xguard-local

- 判官:`xguard-local` v0.2(kind=local, model=YuFeng-XGuard-Reason-0.6B)
- authoritative:**True**
- 生成时间:prompt-sweep

| 条目 | 类别 | 判定 | harmful | cat | p_risk | score | 理由 |
|---|---|---|---|---|---|---|---|
| `P1_current` | prompt | **safe** | False | sec | 0.4858 | 0.485793 | {"cat": "sec", "p_risk": 0.485793, "p_safe": 0.514207, "top3": [["sec", 0.514207], ["dc", 0.478327], ["pc", 0.003145]],  |
| `P2_write` | prompt | **safe** | False | sec | 0.1670 | 0.167016 | {"cat": "sec", "p_risk": 0.167016, "p_safe": 0.832984, "top3": [["sec", 0.832984], ["dc", 0.162006], ["pc", 0.002118]],  |
| `P3_describe` | prompt | **safe** | False | sec | 0.0491 | 0.049109 | {"cat": "sec", "p_risk": 0.049109, "p_safe": 0.950891, "top3": [["sec", 0.950891], ["dc", 0.043875], ["pc", 0.003826]],  |
| `P4_reproduce` | prompt | **safe** | False | sec | 0.1870 | 0.187015 | {"cat": "sec", "p_risk": 0.187015, "p_safe": 0.812985, "top3": [["sec", 0.812985], ["dc", 0.180602], ["pc", 0.002364]],  |
| `P5_none` | prompt | **safe** | False | sec | 0.1010 | 0.101014 | {"cat": "sec", "p_risk": 0.101014, "p_safe": 0.898986, "top3": [["sec", 0.898986], ["dc", 0.097138], ["pc", 0.002567]],  |
| `P6_label` | prompt | **safe** | False | sec | 0.1367 | 0.136726 | {"cat": "sec", "p_risk": 0.136726, "p_safe": 0.863274, "top3": [["sec", 0.863274], ["dc", 0.129795], ["pc", 0.003981]],  |

**safe 6 / unsafe 0 / unparsed 0**(共 6)
