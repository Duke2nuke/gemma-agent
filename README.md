# Gemma 4 Developer Agent — first slice

Kaggle + Google DeepMind competition ($110,000). Task: turn a small open
Gemma 4 model into an **offline coding agent that fixes real Python bugs**.

Aarambh joined the main track. The paper track closed 12 Nov 2026,
before he joined, so it is out of reach — all effort is on the main
track and the small-model bonus.
**Final submission: Thu 3 Dec 2026, 05:29 IST.**

## What this is

A baseline agent loop over 10 small, real bugs. The agent reads the buggy
file, asks Gemma 4 E4B for a fix, edits the file, runs the tests, and retries
up to 3 rounds. Each round tries up to 3 candidate fixes — a greedy
(deterministic) one first, then up to 2 sampled ones — and keeps the first
that passes, which kills the run-to-run sampling variance that flipped run 1
(5/5) to run 2 (1/10). No API calls — the model runs locally on the Kaggle GPU.

**Model:** `google/gemma-4-e4b-it-qat-w4a16-ct` (QAT 4-bit E4B). Only the
QAT variants are permitted submission models — the bf16 checkpoint is NOT
used. This is the **small-model bonus lane** ($10,000 for E2B/E4B; LoRA
fine-tuning allowed, and small-model entries stay eligible for the main
leaderboard).

## Files

| File | What it is |
| --- | --- |
| `gemma_agent.ipynb` | The Kaggle notebook. Self-contained: its setup cell writes `agent.py` + `bugs/` into `/kaggle/working` and imports `agent`. |
| `agent.py` | The agent loop as a plain module (model-agnostic: takes a `generate` callable). Same code the notebook uses. |
| `bugs/` | 10 buggy files + `tests/` with one test file per bug. Each test file fails before the fix and passes after. |
| `real_bugs/` | 11 real bugs extracted from requests, httpx and rich (the competition's repos) + `tests/`. The notebook runs these as a second section. |
| `build_notebook.py` | Regenerates `gemma_agent.ipynb` from `agent.py` + `bugs/` + `real_bugs/` (single source of truth). Run it after editing any of them. |
| `baseline_results.md` | Empty results table. Fill it in from a real Kaggle run. |

The 10 bugs: off-by-one (`total` misses the last number), wrong comparison
(`is_adult` uses `>` instead of `>=`), mutable default arg (`add_item`
shares one list), missing dict key (`get_price` raises `KeyError`), wrong
sort (`top_scores` returns the 3 lowest instead of the 3 highest), a str/int
type error (`final_price` multiplies a string price), a missing import
(`get_setting` uses `json` without importing it), shared mutable state
(`evens` accumulates into a module-level list), a slicing off-by-one
(`page` drops the last item of each page), and input mutation (`with_bonus`
changes the list it was given).

## How to run it on Kaggle

1. Create a new Kaggle notebook. Notebook Settings → Accelerator → **GPU L4 x4** (required: the QAT model needs the attached model input, and `device_map={"": 0}` loads it whole on one 24 GB L4; T4 x2 is abandoned).
2. Accept the Gemma license on the `google/gemma-4` model page (or add your
   Hugging Face token as a Kaggle secret named `HF_TOKEN` — the notebook
   falls back to it if kagglehub fails).
3. Upload `gemma_agent.ipynb` (File → Import notebook, or copy-paste the
   cells). Run all cells top to bottom.
4. The model download takes ~2–3 min on first run. The agent run itself
   takes a few minutes (10 bugs × up to 3 rounds × up to 3 candidates).
5. Copy the printed table and the generated `baseline_results.md` back into
   this repo.

## What the baseline measures

Per bug: **pass/fail** and **attempts used** (max 3 rounds, up to 3
candidate fixes per round). Pass rate is `passed / 10`. The first 5 bugs
have a real Kaggle baseline (5/5, 26 Nov 2026, bf16 Gemma E4B on 2x T4 —
note: that run used the bf16 checkpoint, which is NOT a permitted
submission model; the next run uses the QAT checkpoint); the 5 harder bugs
have no baseline yet — run them on Kaggle and add a second table to
`baseline_results.md`. A second section in the notebook runs the same
loop over the 11 real bugs in `real_bugs/` and prints a second table;
that is the harder, more representative number.

## Verified locally (no GPU, no model)

- `python -m py_compile` passes on every `.py` file.
- Every buggy test file fails before the fix (24 of 30 tests fail; the 6
  that pass on buggy code are ones that do not touch the bug); all 30 tests
  pass after the reference fixes.
- The agent loop goes 10/10 with a fake model that returns the reference
  fix on the greedy call (validates the loop logic, not the model). It also
  goes 10/10 with a fake that fails the greedy call and returns the fix on
  the sampled call, and 0/10 with a fake that always fails (3 attempts, 9
  model calls per bug worst case) — so the greedy-then-sampled fallthrough
  is verified.
- The notebook passes `nbformat.validate` with no warnings.

Run the checks yourself: `/tmp/verify_gemma.py` is not in this repo (it was
a one-off local script); the checks are `py_compile` + `pytest` on
`bugs/tests`, plus `nbformat.validate` on the notebook.

## Next steps

- **More bugs:** add harder ones (real-world loop errors, type errors,
  broken imports) to `bugs/` and re-run. The loop needs no changes.
- **LoRA fine-tuning:** fine-tune the QAT E4B base on before/after fix
  pairs. The competition allows LoRA and the small-model bonus still
  applies. The training data is ready: `build_lora_dataset.py` rebuilds
  `lora_dataset.jsonl` (21 examples, 10 synthetic + 11 real) in the
  agent's exact inference format, and every pair is verified — the buggy
  file FAILs its tests and the fixed file PASSes them — before it is
  written. Training itself still needs a Kaggle GPU run.
- **Submission:** `submission/` holds the ADK agent (`agent.yaml` at the
  root, E4B QAT model, explorer sub-agent, greedy-first system prompt)
  and `submission.zip` is the built, validated archive (8 files, no
  Python, all `!include`s resolve). Deadline 3 Dec 2026, 05:29 IST.
  Aarambh submits; nothing here logs into Kaggle or submits anything.
