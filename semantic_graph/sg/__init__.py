"""第一步 · 语义图建模工作流(docs/10 的 A5 + R2 + S2/S3/S6 + 探针接口)。

模块
----
schema    数据结构(与 docs/03 §7 的接口增量一致)
lexicon   规则/模板抽取用的词典与模式
sources   data_origin → 规范化 JSONL(处理 BOM、多行引号、稳定 join key)
extract   M1:goal → 类型化属性图(A5 路线)
split     M2:S2 约束优化分区 / S3 阶段分层 / S6 零假设基线
probe     探针接口 + 离线占位探针 + XGuard 适配桩
metrics   覆盖率 / lift / 聚合器对照 / 报告渲染
workflow  端到端跑一个 run 目录
cli       python -m sg.cli prepare|run|report

快速开始
--------
    cd semantic_graph
    python -m unittest discover -s tests -v
    python -m sg.cli run --subset mixed --limit 20 --out runs/demo
"""

from .schema import SCHEMA_VER  # noqa: F401

__all__ = ["SCHEMA_VER"]
__version__ = "0.1.0"
