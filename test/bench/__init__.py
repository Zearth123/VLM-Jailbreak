"""YuFeng-XGuard 回复评测程序。

对「攻击的目标模型给出的回复」逐条做风险判定(两阶段输出:风险判定 + 风险归因),
并基于护栏返回值计算 ASR 等指标。

模块约定:
    judge.py    —— 唯一允许 import torch 的模块
    report.py   —— 结构性禁止 import torch(改阈值 / 加指标零 GPU 成本)
    metrics.py  —— 纯函数,禁止 import torch
"""

__all__ = ["schema", "sources", "judge", "metrics", "report"]

JUDGE_ID = "xguard29-0.6b/fwd-gen1"  # 判定原语指纹的一部分
MSG_VER = "m1-response-only"  # 消息模板版本:严格只有回复一条 user message
