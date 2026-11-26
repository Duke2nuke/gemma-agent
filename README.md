# Gemma 4 Developer Agent — first slice

Kaggle + Google DeepMind competition ($110,000). Task: turn a small open
Gemma 4 model into an **offline coding agent that fixes real Python bugs**.

Aarambh joined both the main track and the paper track.
**Final submission: Thu 3 Dec 2026, 05:29 IST.**

## What this is

A baseline agent loop over 10 small, real bugs. The agent reads the buggy
file, asks Gemma 4 E4B for a fix, edits the file, runs the tests, and retries
up to 3 times. No API calls — the model runs locally on the Kaggle GPU.

**Model:** `google/gemma-4-e4b-it` (instruction-tuned E4B, 4.5B active
params). This is the **small-model bonus lane** ($10,000 for E2B/E4B; LoRA
fine-tuning allowed, and small-model entries stay eligible for the main
leaderboard and the paper track).

## Files

| File | What it is |
| --- | --- |
| `gemma_agent.ipynb` | The Kaggle notebook. Self-contained: its setup cell writes `agent.py` + `bugs/` into `/kaggle/working` and imports `agent`. |
| `agent.py` | The agent loop as a plain module (model-agnostic: takes a `generate` callable). Same code the notebook uses. |
| `bugs/` | 10 buggy files + `tests/` with one test file per bug. Each test file fails before the fix and passes after. |
| `build_notebook.py` | Regenerates `gemma_agent.ipynb` from `agent.py` + `bugs/` (single source of truth). Run it after editing either. |
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

1. Create a new Kaggle notebook. Notebook Settings → Accelerator → **GPU T4 x2**.
2. Accept the Gemma license on the `google/gemma-4` model page (or add your
   Hugging Face token as a Kaggle secret named `HF_TOKEN` — the notebook
   falls back to it if kagglehub fails).
3. Upload `gemma_agent.ipynb` (File → Import notebook, or copy-paste the
   cells). Run all cells top to bottom.
4. The model download takes ~2–3 min on first run. The agent run itself
   takes a few minutes (10 bugs × up to 3 attempts).
5. Copy the printed table and the generated `baseline_results.md` back into
   this repo.

## What the baseline measures

Per bug: **pass/fail** and **attempts used** (max 3). Pass rate is
`passed / 10`. The first 5 bugs have a real Kaggle baseline (5/5, 26 Nov
2026, Gemma E4B on 2x T4); the 5 harder bugs have no baseline yet — run
them on Kaggle and add a second table to `baseline_results.md`.

## Verified locally (no GPU, no model)

- `python -m py_compile` passes on every `.py` file.
- Every buggy test file fails before the fix (24 of 30 tests fail; the 6
  that pass on buggy code are ones that do not touch the bug); all 30 tests
  pass after the reference fixes.
- The agent loop goes 10/10 with a fake model that returns the reference fix
  (validates the loop logic, not the model).
- The notebook passes `nbformat.validate` with no warnings.

Run the checks yourself: `/tmp/verify_gemma.py` is not in this repo (it was
a one-off local script); the checks are `py_compile` + `pytest` on
`bugs/tests`, plus `nbformat.validate` on the notebook.

## Next steps

- **More bugs:** add harder ones (real-world loop errors, type errors,
  broken imports) to `bugs/` and re-run. The loop needs no changes.
- **LoRA fine-tuning:** fine-tune E4B on before/after fix pairs. The
  competition allows LoRA and the small-model bonus still applies.
- **Paper track:** paper topics include new tasks and eval datasets — this
  5-bug set is the seed of one. Log every attempt, not just passes.
- **Submission:** the final Kaggle submission is the notebook plus a short
  write-up, by 3 Dec 2026, 05:29 IST. Aarambh submits; nothing here logs
  into Kaggle or submits anything.
