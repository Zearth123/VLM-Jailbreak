#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地护栏解析与人脸识别(`tools/guard_local.py`)的离线测试。

不加载任何模型:只钉住"家族识别 + 输出解析 + 缺口必须显式失败"。
真跑模型的那一步在 `tools/LOCAL_GUARD.md` 里用命令验证。
"""

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

sys.path.insert(0, str(HERE / "tools"))
import guard_local as GL  # noqa: E402


class TestFamily(unittest.TestCase):
    def test_model_ids(self):
        self.assertEqual(GL.family_of("Qwen/Qwen3Guard-Gen-0.6B"), "qwen3guard")
        self.assertEqual(GL.family_of("Qwen/Qwen3Guard-Gen-8B"), "qwen3guard")
        self.assertEqual(GL.family_of("meta-llama/Llama-Guard-3-1B"), "llama_guard")
        self.assertEqual(GL.family_of("ibm-granite/granite-guardian-3.0-2b"), "granite")
        self.assertEqual(GL.family_of("some/base-model"), "generic")

    def test_local_path_still_detected(self):
        self.assertEqual(GL.family_of(r"D:\models\Qwen3Guard-Gen-0.6B"), "qwen3guard")


class TestParse(unittest.TestCase):
    def test_qwen3guard_three_tiers(self):
        self.assertFalse(GL.parse("qwen3guard", "Safety: Safe\nCategories: None")["harmful"])
        self.assertTrue(GL.parse("qwen3guard", "Safety: Unsafe\nCategories: Violent")["harmful"])
        # Controversial 默认按 unsafe(硬要求是"必须 safe")
        self.assertTrue(GL.parse("qwen3guard", "Safety: Controversial\nCategories: PII")["harmful"])
        # 放宽策略
        self.assertFalse(GL.parse("qwen3guard", "Safety: Controversial",
                                  controversial="safe")["harmful"])

    def test_qwen3guard_categories(self):
        got = GL.parse("qwen3guard", "Safety: Unsafe\nCategories: Violent")
        self.assertEqual(got["categories"], ["Violent"])

    def test_llama_guard(self):
        self.assertEqual(GL.parse("llama_guard", "safe")["harmful"], False)
        got = GL.parse("llama_guard", "unsafe\nS2,S9")
        self.assertTrue(got["harmful"])
        self.assertEqual(got["categories"], ["S2", "S9"])
        self.assertFalse(GL.parse("llama_guard", "**safe**")["harmful"])   # 带 markdown

    def test_granite(self):
        self.assertTrue(GL.parse("granite", "Yes")["harmful"])
        self.assertFalse(GL.parse("granite", "No")["harmful"])

    def test_generic_unsafe_before_safe(self):
        """`unsafe` 里含 `safe`,顺序错了会把有害判成无害。"""
        self.assertTrue(GL.parse("generic", "unsafe")["harmful"])
        self.assertTrue(GL.parse("generic", "UNSAFE")["harmful"])
        self.assertFalse(GL.parse("generic", "This looks safe to me.")["harmful"])

    def test_unparsed_is_never_safe(self):
        for family in ("qwen3guard", "llama_guard", "granite", "generic"):
            got = GL.parse(family, "")
            self.assertIsNone(got["harmful"], family)
            self.assertEqual(got["label"], "unparsed", family)
            self.assertIsNone(GL.parse(family, "我想我帮不了你。")["harmful"], family)


class TestSelfTest(unittest.TestCase):
    def test_builtin_cases_pass(self):
        self.assertEqual(GL.self_test(), 0)


class TestNoTorchIsLoud(unittest.TestCase):
    def test_guard_unavailable_is_raised_not_faked(self):
        """没有 torch 时必须抛 GuardUnavailable,绝不能返回占位判定。"""
        try:
            import torch  # noqa: F401
        except Exception:
            pass
        else:
            self.skipTest("本机有 torch,不适用")
        items = [{"id": "x", "text": "hello"}]
        with self.assertRaises(GL.GuardUnavailable):
            GL.run("Qwen/Qwen3Guard-Gen-0.6B", items, family="qwen3guard",
                   device="cpu", dtype="float32", max_new_tokens=8,
                   controversial="unsafe", risk="harm", local_files_only=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
