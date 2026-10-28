#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`tools/xguard_cli.py` 的离线测试(不加载模型)。

钉住的语义:
  - `sec`(Safe-Safe)= safe,其它任何类别 = unsafe,拿不到类别 = unparsed(**绝不算 safe**);
  - 模型目录校验:缺 config.json / 缺 xguard.py 都要显式报错;
  - 缺 torch 时抛 GuardUnavailable,不返回占位判定。
"""

import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tools"))

import xguard_cli as X  # noqa: E402


class TestSafeLabel(unittest.TestCase):
    def test_safe_names(self):
        self.assertTrue(X.is_safe_label("Safe-Safe"))
        self.assertTrue(X.is_safe_label("safe"))
        self.assertTrue(X.is_safe_label("  SAFE-Anything"))

    def test_risk_names_are_not_safe(self):
        for name in ("Crimes and Illegal Activities-Dangerous Weapons",
                     "Cybersecurity-Hacker Attack",
                     "Hate Speech-Threats and Intimidation"):
            self.assertFalse(X.is_safe_label(name), name)


class TestToRow(unittest.TestCase):
    def test_safe_top1(self):
        row = X.to_row("i", "t", {"risk_score": {"Safe-Safe": 0.98}, "response": "sec"})
        self.assertEqual(row["gate"], "safe")
        self.assertIs(row["harmful"], False)
        self.assertEqual(row["categories"], [])

    def test_unsafe_top1(self):
        row = X.to_row("i", "t", {"risk_score": {
            "Crimes and Illegal Activities-Dangerous Weapons": 0.9,
            "Safe-Safe": 0.05}})
        self.assertEqual(row["gate"], "unsafe")
        self.assertIs(row["harmful"], True)
        self.assertEqual(row["categories"],
                         ["Crimes and Illegal Activities-Dangerous Weapons"])

    def test_take_max_not_first(self):
        row = X.to_row("i", "t", {"risk_score": {"Safe-Safe": 0.1,
                                                 "Cybersecurity-Malicious Code": 0.7}})
        self.assertEqual(row["gate"], "unsafe")

    def test_empty_is_unparsed_never_safe(self):
        for res in ({"risk_score": {}}, {"risk_score": {}, "truncated": True}, {}):
            row = X.to_row("i", "t", res)
            self.assertEqual(row["gate"], "unparsed")
            self.assertIsNone(row["harmful"])

    def test_row_shape_matches_pipeline(self):
        """必须能直接喂 tools/submission.py --guard。"""
        row = X.to_row("i", "t", {"risk_score": {"Safe-Safe": 0.9}})
        for key in ("id", "text", "harmful", "label", "score", "reason", "gate"):
            self.assertIn(key, row)


class TestExplanation(unittest.TestCase):
    def test_extracts_tagged(self):
        got = X.explanation_of("sec\n<explanation>Just a keyword list.</explanation>")
        self.assertEqual(got, "Just a keyword list.")

    def test_multiline_and_missing(self):
        self.assertEqual(X.explanation_of("<explanation>a\nb</explanation>"), "a\nb")
        self.assertEqual(X.explanation_of("no tags here"), "")


class TestModelDir(unittest.TestCase):
    def test_missing_config_is_loud(self):
        with self.assertRaises(SystemExit):
            X.resolve_model_dir(Path("."))

    def test_real_model_dir_if_present(self):
        if not (X.DEFAULT_MODEL / "config.json").exists():
            self.skipTest(f"本机没有 {X.DEFAULT_MODEL}")
        d = X.resolve_model_dir(None)
        self.assertTrue((d / "xguard.py").exists())
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["architectures"], ["Qwen3ForCausalLM"])
        tcfg = json.loads((d / "tokenizer_config.json").read_text(encoding="utf-8"))
        self.assertEqual(tcfg["id2risk"]["sec"], "Safe-Safe")

    def test_default_path_is_workspace_models(self):
        self.assertEqual(X.DEFAULT_MODEL.name, "YuFeng-XGuard-Reason-0.6B")
        self.assertEqual(X.DEFAULT_MODEL.parent.name, ".models")


class TestItems(unittest.TestCase):
    def test_loads_demo_items(self):
        p = HERE / "artifacts" / "local_guard_demo_items.json"
        if not p.exists():
            self.skipTest("演示条目不在")
        items = X.load_items(p)
        self.assertGreaterEqual(len(items), 3)
        self.assertTrue(all("id" in i and "text" in i for i in items))

    def test_missing_text_is_loud(self):
        from tests import workspace_tempdir
        with workspace_tempdir() as td:
            p = Path(td) / "x.json"
            p.write_text(json.dumps({"items": [{"id": "a"}]}), encoding="utf-8")
            with self.assertRaises(SystemExit):
                X.load_items(p)


class TestSelfTestAndNoTorch(unittest.TestCase):
    def test_self_test_passes(self):
        self.assertEqual(X.self_test(), 0)

    def test_no_torch_is_loud(self):
        try:
            import torch  # noqa: F401
        except Exception:
            pass
        else:
            self.skipTest("本机有 torch")
        with self.assertRaises(X.GuardUnavailable):
            X.load_guard(X.DEFAULT_MODEL, "cpu", "float32")


if __name__ == "__main__":
    unittest.main(verbosity=2)
