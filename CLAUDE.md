# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A research project on **adversarial robustness of vision-language models (VLMs)**, specifically the
attack side: a black-box, cross-modal semantic-decomposition attack pipeline, evaluated against
HarmBench. The deliverable is the attack pipeline itself; evaluation reuses the benchmark's own judge.

`docs/` is the source of truth for the design. Read `docs/00-overview.md` first — it is a complete
standalone summary of the whole design. `docs/README.md` indexes all thirteen documents (overview,
threat model, mechanism analysis, data model, module decomposition, experimental setup, roadmap,
the P1 execution spec, design rationale, open questions, the semantic-graph modeling options,
the P4 experiment protocol, and the vector-graph splitting route).

**Orientation for a new session:**

- `docs/08-design-rationale.md` explains why the design is what it is, and which components were
  deliberately removed. Read it early — it will stop you re-introducing rejected ideas.
- `docs/07-p1-spec.md` is the current active work. It is a **manual** pilot: no pipeline code exists
  and none is wanted yet. P1 exists to decide whether the mechanism works at all before any code
  is written.
- `docs/10-semantic-graph-modeling.md` is a **proposal, not a spec**: candidate routes for M1/M2 plus
  the P2 sensitivity probe, with a decision matrix and fallback ladder. Nothing in it is decided
  until P2 starts.
- `intro.md` is the original design draft. Parts of it are superseded — notably its GCG suffix and
  PGD perturbation components, both deleted because the attack is black-box. See `08`.

**There is a first implementation, and it is deliberately narrow.** `semantic_graph/` holds the
P2 step-1 workflow (M1 rule extraction + M2 splitting + a probe interface), pure stdlib — no torch,
no spaCy, no network. One exception: `sg/visual.py` (the image-channel renderer) needs Pillow, and
fails loudly without it:

- `cd semantic_graph && python -m unittest discover -s tests -v` — 249 tests, ~1.4 s, no GPU.
- `cd semantic_graph && python -m sg.cli run --subset structured --limit 68 --ns 2,3,4,5,6 --out runs/structured68`
- `cd semantic_graph && python -m sg.cli split2 --rid safrole_oil_extraction_guide --out artifacts/x.md`
  (splitting + channel assignment; subgraph text is a **verbatim projection of goal spans** — no
  paraphrase, no frames, no invented wording; `sg/project.py` enforces it)
- `cd semantic_graph && python -m sg.cli relax --rid safrole_oil_extraction_guide --out artifacts/x.md`
  (latent segmentation+alignment per arXiv:2010.12676: order enumeration, exact posterior, carrier marginals)
- `cd semantic_graph && python -m sg.cli render --rid ... --out artifacts/inject_<rid>`
  (A0 typographic image conditions: the payload's **verbatim goal spans** drawn as pixels;
  no paraphrase, no invented wording — `drawn_strings` is the audit field)
- `cd semantic_graph && python -m sg.cli recon --dump dumps/<run> --manifest ...`
  (folds the cloud dump into gates A/B/C3 + a complementarity matrix; numpy only)
- `cd semantic_graph && python -m sg.cli embedgraph --dump dumps/<run> --rid ... --k 2`
  (vector-graph route, `docs/12`: word fragments from a **non-aligned, frozen** encoder →
  chain+kNN graph → cut minimizing severed affinity; the node-pair affinity doubles as a per-edge
  cut cost. The dump must come from `tools/embed_probe.py` on the GPU host; missing dump raises
  `EmbedUnavailable`, never a placeholder. Encoders whose id matches instruct/chat/-it/guard are
  refused unless `--allow-aligned`, because `docs/08 §2.1` forbids aligned models in construction)
- `cd semantic_graph && python tools/s0_screen.py` — local, model-free screen: goal type (E/R/P),
  structural ceiling per k, and the candidate subset `C = {type≠E ∧ nodes≥2}`
  (measured: E 23 / R 45 / P 0, `|C| = 41`)
- `cd semantic_graph && python tools/cloud_probe.py --manifest ... --out dumps/<run> --stage llava`
  (runs **on the GPU host**, not here: no torch on this machine; it is the only script not verified locally)
- Its probe (`LexicalProbe`) is an explicit **placeholder**: a bag-of-words weight, *not* the
  `sensitivity` of `docs/10` §5 P1. Never quote a number derived from it as a P2 gate result; the
  run's `report.md` says so on every page.
- Experiment protocol for the LLaVA-1.5-7B target (criteria A vs B, the A0 arm, the deferred A2 arm,
  dump contract, gates) is `docs/11-p4-experiment-protocol.md`.

**The whole `benchmark/` tree was deleted on 2026-10-15 and sits in the Recycle Bin**
(`D:\$Recycle.Bin\...\$I*` entries reference `D:\VLMjalibreak\benchmark`), so it can be restored from
Explorer. What it used to hold, and is therefore *not* in the current checkout: `benchmark/tests/`
(the eval-harness tests, which import a `bench` package that was already missing),
`benchmark/config/refusal_lexicon.txt`, `xguard.py`, a Linux `.venv`, and the
`YuFeng-XGuard-Reason-0.6B` guard weights. Anything that needs the real `sensitivity` probe must
first re-obtain those weights. There is no build system and no git history — do not invent commands
beyond the two above.

## Non-obvious constraints on the design

These are settled and easy to violate by accident:

- **Black-box.** The attack method must not use gradients. This is what makes it transferable to
  closed models. Note this does *not* forbid the researcher from holding open weights — weight
  access is fine for mechanism analysis, just not as an input to the attack method itself.
- **Single-shot.** Build the input once, submit, judge. No querying the target during construction.
  See `docs/09-open-questions.md` R1 — this is the highest-priority unresolved fork.
- **Three-level harmlessness (L1/L2/L3).** Inputs must be harmless per-modality *and* harmless when
  explicitly combined in text form. The L2 captioner must be the strongest available — using a weak
  one inflates results.
- **The attack targets external input guardrails, not model alignment.** Alignment operates on the
  same joint representation the attack lives in. If a deployment has no external guard, this design
  has zero advantage. This premise must be stated in any writeup.

## Datasets

| File | Rows | Columns |
|---|---|---|
| `harmbench_behaviors_text_all.csv` | 400 | `Behavior`, `FunctionalCategory`, `SemanticCategory`, `Tags`, `ContextString`, `BehaviorID` |
| `advbench_harmful_behaviors.csv` | 520 | `goal`, `target` |

`harmbench_behaviors_text_all.csv` is UTF-8 **with a BOM** — open it with `utf-8-sig`, or the first
column name becomes `﻿Behavior`. Structure:

- `FunctionalCategory`: `standard` (200), `copyright` (100), `contextual` (100).
- `SemanticCategory`: `copyright` 100, `cybercrime_intrusion` 67, `illegal` 65,
  `misinformation_disinformation` 65, `chemical_biological` 56, `harassment_bullying` 25, `harmful` 22.
- `ContextString` is populated only on the 100 `contextual` rows; `Tags` only on the 200 `standard`
  rows. Elsewhere they are empty strings, not NaN — don't rely on `isna()`.
- `BehaviorID` (e.g. `dimethylmercury_instructions_home`) is the stable key. **Join everything on
  it** — judge outputs, intermediate artifacts, split manifests. Never on row index or behavior text.

`advbench_harmful_behaviors.csv` is plain ASCII, two columns: the harmful `goal` and an affirmative
`target` completion. Used as a second test set for generalization.

Data splits: dev 100 / test 300, stratified by `FunctionalCategory`, split by `BehaviorID`.

## Working notes

- `intro.md` is UTF-8 with **CRLF** line endings; preserve both when editing so diffs stay clean.
- `docs/` files are UTF-8 with LF.
- Anything measured against the HarmBench judge must also be measured against a stronger
  vision-capable judge and reported as a delta — the weak judge inflates ASR on this attack class.
