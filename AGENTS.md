# Repository Guidelines

## Project Structure & Module Organization

The active implementation lives in `semantic_graph/`. Core Python modules are in `semantic_graph/sg/`; the command-line entry point is `sg/cli.py`. Keep unit and end-to-end tests in `semantic_graph/tests/`, named after the module they exercise. Utility and GPU-host scripts belong in `semantic_graph/tools/`. Generated experiment output goes to `semantic_graph/runs/` or `semantic_graph/artifacts/`, not into the package.

Design documents in `docs/` are the source of truth. Start with `docs/00-overview.md`, then consult `docs/08-design-rationale.md` before changing architecture. Raw HarmBench and AdvBench inputs are under `data_origin/`; preserve their encoding and schema. `intro.md` is historical and may be superseded.

## Build, Test, and Development Commands

Run commands from `semantic_graph/`:

```bash
python -m unittest discover -s tests -v
python -m sg.cli run --subset structured --limit 68 --ns 2,3,4,5,6 --out runs/structured68
python -m sg.cli report --run runs/structured68
python -m sg.cli render --rid <BehaviorID> --out artifacts/inject_<name>
```

There is no build system. The main workflow is standard-library-only; image rendering requires Pillow, and reconstruction requires NumPy. `tools/cloud_probe.py` is intended for a configured GPU host, not the local test loop.

## Coding Style & Naming Conventions

Use four-space indentation, type hints, `snake_case` for functions and variables, `PascalCase` for classes, and uppercase constants. Follow the existing compact, PEP 8-style layout and module docstrings. No formatter or linter is configured, so keep imports grouped (stdlib before local), avoid unrelated reformatting, and preserve UTF-8/LF in `docs/` and UTF-8/CRLF in `intro.md`.

## Testing Guidelines

Tests use `unittest`; use `test_<behavior>` methods inside `Test<Component>` classes. Add regression tests for every behavioral change, especially span traceability, deterministic output, dependency-edge invariants, and explicit failure when optional models are unavailable. Run the full suite before submitting. Never report `LexicalProbe` results as authoritative model-sensitivity evidence.

## Data, Security & Design Invariants

Join artifacts using `BehaviorID`, never row indexes or behavior text. Read the HarmBench CSV with `utf-8-sig`. Do not add gradients or target-model queries: the method is black-box and single-shot. Projected text and rendered strings must come verbatim from source spans. Keep API keys in environment variables and never commit model weights, secrets, or private endpoint details.

## Commits & Pull Requests

No Git history is available in this checkout. Use short, imperative commit subjects (for example, `Add dependency-edge regression test`) and keep each commit focused. Pull requests should explain the design impact, list tests run, link the relevant `docs/` section or issue, and include sample reports or rendered images when outputs change.
