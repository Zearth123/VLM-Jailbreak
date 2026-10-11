"""数据结构、文本规范化与缓存键。本模块禁止 import torch。"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

SEC = "sec"  # Safe-Safe;其余 28 类为风险

_WS = re.compile(r"\s+")


@dataclass
class ResponseRecord:
    """一条待评测的目标模型回复。"""

    rid: str
    response: str
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.response.strip()


def norm_text(s: str) -> str:
    """NFKC 规范化 + 去首尾空白。用于 sha1 与拒答词表匹配。"""
    return unicodedata.normalize("NFKC", s).strip()


def hash_norm(s: str) -> str:
    """在 norm_text 基础上折叠空白,保证换行/多空格的写法变化不影响缓存键。"""
    return _WS.sub(" ", norm_text(s))


def response_sha(s: str) -> str:
    return hashlib.sha1(hash_norm(s).encode("utf-8")).hexdigest()


def classes_sha(classes: list[str]) -> str:
    return hashlib.sha1(json.dumps(classes, ensure_ascii=False).encode("utf-8")).hexdigest()


def jkey(judge_id: str, rid: str, response: str, msg_ver: str, max_new_tokens: int) -> str:
    """缓存键。response 变了 / 生成配置变了 → 键变 → 自动重判;只重跑报告 → 键不变。"""
    h = hashlib.sha1()
    for part in (judge_id, str(rid), response_sha(response), msg_ver, str(max_new_tokens)):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# judgments.jsonl 每行字段(文档用途)
JUDGMENT_FIELDS = (
    "jkey",  # 缓存键
    "id",  # 输入记录 id
    "cat_id",  # 风险判定(首词类别 ID),如 "dw" / "sec"
    "top1_p",  # 风险判定分值 = 受限 softmax 下该类概率
    "z",  # 29 类首词 logits(float32),一切派生量的唯一真值来源
    "logZ",  # 全词表 logsumexp
    "m29",  # 29 类 token 占全词表分布的总质量(格式异常检出)
    "attribution",  # 风险归因全文(含 <explanation> 标签)
    "attr_tokens",  # 归因 token 数
    "attr_stop_reason",  # close_tag / eos / max_new_tokens / disabled
    "n_input_tokens",  # 渲染后输入长度
    "overlong",  # 输入超长,已排除出分母
    "t_gen_s",  # 生成本次判定耗时
    "err",  # 非 null 表示 judge_error / empty_response,排除出分母
)
