"""Build gemma_agent.ipynb from the repo files (single source of truth).

The notebook is self-contained: its setup cell writes agent.py and the
bugs/ tree into /kaggle/working from the embedded FILES dict, then imports
agent. That way one notebook upload is enough to run it on Kaggle, and the
notebook can never drift from the files in this repo.

Run: python3 build_notebook.py
Output: gemma_agent.ipynb (same directory)
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent

FILES = {
    "agent.py": (ROOT / "agent.py").read_text(),
}
for bug in sorted((ROOT / "bugs").glob("bug_*.py")):
    FILES[f"bugs/{bug.name}"] = bug.read_text()
for test in sorted((ROOT / "bugs" / "tests").glob("*.py")):
    FILES[f"bugs/tests/{test.name}"] = test.read_text()

SETUP_CELL = '''import sys
from pathlib import Path

# Files embedded by build_notebook.py from the repo (agent.py + bugs/).
# Written to /kaggle/working so the notebook runs from a single upload.
FILES = ''' + json.dumps(FILES, indent=1) + '''

for rel, content in FILES.items():
    path = Path("/kaggle/working") / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)

sys.path.insert(0, "/kaggle/working")
import agent

# Smoke test: the harness works and the first bug really fails before any fix.
passed, out = agent.run_tests(
    Path("/kaggle/working/bugs/tests/test_bug_01_off_by_one.py"),
    cwd=Path("/kaggle/working/bugs"),
)
print("smoke test, bug_01 original:", "PASS (unexpected!)" if passed else "FAIL (expected)")
'''


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


cells = [
    md("""# Gemma 4 Developer Agent — first slice (baseline)

Turn Gemma 4 E4B into an offline coding agent that fixes real Python bugs.
The agent loop: read the buggy file, plan a fix, edit it, run the tests,
retry up to 3 rounds, record pass/fail per bug. Each round tries up to 3
candidate fixes — a greedy (deterministic) one first, then up to 2 sampled
ones — and keeps the first that passes. That kills the run-to-run sampling
variance that flipped run 1 (5/5) to run 2 (1/10) with identical settings.

**Model:** `google/gemma-4-e4b-it-qat-w4a16-ct` — the QAT 4-bit E4B
checkpoint. Only the QAT variants are permitted submission models, so the
bf16 checkpoint is NOT used. The 4-bit weights (~4.5 GB) load easily on
Kaggle's 2x T4 with `device_map="auto"`. This is the small-model bonus lane
($10,000 for E2B/E4B; small-model entries stay eligible for the main
leaderboard and the paper track). NOTE: the kagglehub handle below,
`google/gemma-4/transformers/e4b-it-qat-w4a16-ct`, follows the pattern of
the bf16 handle and is not yet confirmed on Kaggle — if kagglehub fails,
the Hugging Face fallback loads the same checkpoint by its HF id.

**Before running:**
1. Notebook Settings → Accelerator → **GPU T4 x2**.
2. Accept the Gemma license: on Kaggle go to the `google/gemma-4` model page
   and accept, or add your Hugging Face token as a Kaggle secret named
   `HF_TOKEN` (the notebook falls back to it if kagglehub fails).
3. Run all cells top to bottom. The run takes a few minutes after the model
   loads (~2-3 min download on first run).

**What the baseline measures:** pass/fail per bug and attempts used, over
10 small real bugs (off-by-one, wrong comparison, mutable default arg,
missing dict key, wrong sort, a str/int type error, a missing import, shared
mutable state leaking between calls, a slicing off-by-one, and a function
that mutates its input). Each bug has tests that fail before the fix
and pass after. This is the starting point — later slices add more bugs,
LoRA fine-tuning, and the paper-track write-up."""),
    code("""import torch

print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name}, {p.total_memory / 1e9:.1f} GB")
assert torch.cuda.is_available(), "Enable GPU in Notebook Settings (T4 x2)"
"""),
    code("""# Kaggle images usually have these, but install to be safe.
# compressed-tensors is needed to load the QAT w4a16 checkpoint.
%pip install -q -U transformers accelerate kagglehub compressed-tensors
"""),
    code(SETUP_CELL),
    code("""import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# QAT 4-bit E4B — the only permitted submission model (bf16 is NOT allowed).
MODEL_ID = "google/gemma-4-e4b-it-qat-w4a16-ct"

model_path = None
try:
    import kagglehub
    # Handle follows the bf16 pattern; unconfirmed on Kaggle — the HF
    # fallback below loads the same checkpoint if this fails.
    model_path = kagglehub.model_download(
        "google/gemma-4/transformers/e4b-it-qat-w4a16-ct"
    )
    print("downloaded via kagglehub:", model_path)
except Exception as e:
    print("kagglehub failed, falling back to Hugging Face:", e)

load_from = model_path or MODEL_ID
tokenizer = AutoTokenizer.from_pretrained(load_from)
model = AutoModelForCausalLM.from_pretrained(
    load_from,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
print("model loaded:", load_from)
"""),
    code("""def generate(prompt: str, greedy: bool = False) -> str:
    \"\"\"Gemma-backed generate() for agent.run_all.

    greedy=True: deterministic decoding (do_sample=False) — the first
    candidate of each round, which gives a stable baseline. greedy=False:
    sampled (temperature 0.7, top_p 0.95) — the fallback candidates that
    beat run-to-run sampling variance.
    \"\"\"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    gen_kwargs = {"max_new_tokens": 512}
    if greedy:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = 0.7
        gen_kwargs["top_p"] = 0.95
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )


print(generate("Write one line of Python that adds 1 and 2. Reply with code only."))
"""),
    code("""from pathlib import Path

results = agent.run_all(
    generate,
    Path("/kaggle/working/bugs"),
    Path("/kaggle/working/runs"),
    max_attempts=3,
)
print()
print(agent.results_table(results))
"""),
    code("""from pathlib import Path

md = (
    "# Baseline results\\n\\n"
    "Model: google/gemma-4-e4b-it-qat-w4a16-ct (QAT 4-bit, Kaggle 2x T4)\\n"
    "Date: (fill in)\\n\\n"
    + agent.results_table(results)
)
Path("/kaggle/working/baseline_results.md").write_text(md)
print(md)
print("saved to /kaggle/working/baseline_results.md — copy it back to the repo")
"""),
    md("""## Next steps

- **More bugs:** add harder ones (wrong loop bounds in real code, type errors,
  broken imports) to `bugs/` and re-run. The loop needs no changes.
- **LoRA fine-tuning:** fine-tune the QAT E4B base on before/after fix
  pairs (the competition allows LoRA; the small-model bonus still
  applies). Start from the fixes this agent produces, corrected by hand.
- **Paper track:** the paper topics include new tasks and eval datasets —
  this 10-bug set is the seed of one. Log every attempt, not just passes.
- **Submission:** the final competition submission is a submission.zip
  with an ADK agent.yaml — this notebook is dev tooling for the loop, not
  the submission. Deadline Thu 3 Dec 2026, 05:29 IST."""),
]

for i, cell in enumerate(cells):
    cell.setdefault("id", f"cell-{i:02d}")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = ROOT / "gemma_agent.ipynb"
out.write_text(json.dumps(notebook, indent=1))
print(f"wrote {out} ({len(cells)} cells)")
