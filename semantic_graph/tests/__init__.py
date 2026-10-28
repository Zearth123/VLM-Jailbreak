"""语义图建模工作流的测试套件。

约定(与 benchmark/tests 一致):
  - 纯 stdlib、零 GPU、零网络
  - 断言尽量是**手算值或不变式**,而不是实现自身的回显

运行:
    cd semantic_graph
    python -m unittest discover -s tests -v
"""

import contextlib
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]  # semantic_graph/
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# 系统临时目录在本机沙箱下不可删除(WinError 5)→ 用工作区内的临时目录,
# 且清理必须 ignore_errors:沙箱可能连工作区内的目录删除也拒绝。
TMP_ROOT = HERE / "runs" / "_tmp"


@contextlib.contextmanager
def workspace_tempdir():
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TMP_ROOT / f"t_{uuid.uuid4().hex[:10]}"
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
