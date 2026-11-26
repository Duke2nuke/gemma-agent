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
for bug in sorted((ROOT / "real_bugs").glob("real_*.py")):
    FILES[f"real_bugs/{bug.name}"] = bug.read_text()
for test in sorted((ROOT / "real_bugs" / "tests").glob("*.py")):
    FILES[f"real_bugs/tests/{test.name}"] = test.read_text()

SETUP_CELL = '''import sys
from pathlib import Path

# Files embedded by build_notebook.py from the repo (agent.py + bugs/).
# Written to /kaggle/working so the notebook runs from a single upload.
# ensure_ascii=False is REQUIRED here, not cosmetic: with the default
# ensure_ascii=True, astral characters (e.g. the emoji in the real_02
# super_len test) become \ud83d\ude00 surrogate-pair escapes in the
# cell source, which Python parses as lone surrogates — and
# Path.write_text then fails with UnicodeEncodeError on Kaggle.
# With ensure_ascii=False the literal character is embedded instead.
FILES = ''' + json.dumps(FILES, indent=1, ensure_ascii=False) + '''

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
bf16 checkpoint is NOT used. NOTE: the kagglehub handle is
`google/gemma-4/Other/gemma-4-e4b-it-qat-w4a16-ct` (the "Other"
framework, confirmed 26 Nov from the Kaggle model page — the
"transformers" handle 404s). The checkpoint is the full multimodal
Gemma4ForConditionalGeneration model, so the load uses
`attn_implementation="eager"` — its audio sub-model does not support
SDPA (that was the 23:21 ValueError).

**Before running:**
0. If you already ran an earlier version of this notebook in this
   session, **restart the kernel first** (Session → Restart, or Runtime →
   Restart) — the old model otherwise stays in GPU memory and the new
   load runs out of memory.
1. Notebook Settings → Accelerator → **GPU L4 x4** (recommended — the
   competition's own hardware, 96 GB, and the QAT load spikes past what
   2x T4 can hold). Attach the model as an input first: right panel →
   Input → Add input → Models → search "gemma-4" → pick the
   `e4b-it-qat-w4a16-ct` variant. The notebook loads the attached model
   first; without it, L4 has no internet so the kagglehub/HF download
   cannot work. (T4 x2 still works as a fallback: there the notebook
   downloads the model itself.)
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
and pass after. A second section runs the same loop over 11 real bugs
extracted from requests, httpx and rich — the competition's own repos —
which is the harder, more representative test. This is the starting
point — later slices add more bugs and LoRA fine-tuning."""),
    code("""import torch

print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name}, {p.total_memory / 1e9:.1f} GB")
assert torch.cuda.is_available(), "Enable GPU in Notebook Settings (L4 x4)"
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
# 1. Attached model input (the L4x4 path — L4 has no internet, so the
# model must be attached: right panel → Input → Add input → Models).
from pathlib import Path as _P
if _P("/kaggle/input").exists():
    for _d in sorted(_P("/kaggle/input").glob("**/config.json")):
        if "gemma" in str(_d).lower():
            model_path = str(_d.parent)
            print("using attached model input:", model_path)
            break
# 2. Kaggle download (works on T4x2, which has internet).
if model_path is None:
    try:
        import kagglehub
        # Confirmed 26 Nov from the Kaggle model page: the QAT checkpoint
        # is under the "Other" framework, variant gemma-4-e4b-it-qat-w4a16-ct.
        model_path = kagglehub.model_download(
            "google/gemma-4/Other/gemma-4-e4b-it-qat-w4a16-ct"
        )
        print("downloaded via kagglehub:", model_path)
    except Exception as e:
        print("kagglehub failed, falling back to Hugging Face:", e)

load_from = model_path or MODEL_ID
tokenizer = AutoTokenizer.from_pretrained(load_from)
model = AutoModelForCausalLM.from_pretrained(
    load_from,
    # Do NOT force bf16 here: it dequantizes the QAT 4-bit weights to
    # ~18 GB and the generate call OOMs on a T4. Let compressed-tensors
    # load the checkpoint in native 4-bit.
    # Load on CPU first: the compressed-tensors quantization/compression
    # step temporarily needs ~2x the model size, and doing it on GPU
    # OOMs (the 00:18 run died trying to allocate 6.2 GiB on GPU 1
    # during load, with 9.27 GiB already there). CPU RAM handles the
    # spike; the weights are dispatched to the GPUs already quantized.
    device_map="cpu",
    # MUST stay eager: Gemma4AudioModel (part of the full multimodal
    # Gemma4ForConditionalGeneration checkpoint) does not support SDPA —
    # transformers raises ValueError and tells you to use eager instead
    # (huggingface/transformers#28005). SDPA was the 23:21 crash.
    attn_implementation="eager",
    low_cpu_mem_usage=True,
)
model.eval()
# Dispatch the already-quantized weights to the GPUs. This both splits
# the model (~8 GiB on GPU 0, the rest on GPU 1) and attaches the
# _hf_hook generate() needs (it reads module._hf_hook.no_grad — the
# 23:52 crash was the hook missing on the top-level model).
from accelerate import dispatch_model, infer_auto_device_map
from types import SimpleNamespace
# dispatch_model() does not take max_memory (TypeError on the 00:26 run),
# so infer the map first, then dispatch with it.
device_map = infer_auto_device_map(model, max_memory={0: "8GiB", 1: "14GiB"})
model = dispatch_model(model, device_map=device_map)
if not hasattr(model, "_hf_hook"):
    model._hf_hook = SimpleNamespace(no_grad=True)
# OOM diagnosis (26 Nov): if the first param is bf16/fp32 instead of
# a 4-bit quantized weight, the checkpoint was dequantized on load and
# will OOM on a single T4. Print it so the run log shows it.
first = next(model.parameters())
total = sum(p.numel() for p in model.parameters())
print("model loaded:", load_from)
print(f"first param dtype: {first.dtype}, total params: {total / 1e9:.2f}B")
print("quantization_config:", getattr(model.config, "quantization_config", None))
for i in range(torch.cuda.device_count()):
    print(
        f"GPU {i} after load: "
        f"allocated {torch.cuda.memory_allocated(i) / 1e9:.1f} GB, "
        f"reserved {torch.cuda.memory_reserved(i) / 1e9:.1f} GB"
    )
"""),
    code("""def generate(prompt: str, greedy: bool = False, max_new_tokens: int = 512) -> str:
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
    torch.cuda.empty_cache()  # free fragmented memory before generate
    gen_kwargs = {"max_new_tokens": max_new_tokens}
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


print(generate("Write one line of Python that adds 1 and 2. Reply with code only.", max_new_tokens=64))
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

print("## Real bugs")
real_results = agent.run_all(
    generate,
    Path("/kaggle/working/real_bugs"),
    Path("/kaggle/working/runs_real"),
    max_attempts=3,
    pattern="real_*.py",
)
print()
print(agent.results_table(real_results))
"""),
    code("""from pathlib import Path

md = (
    "# Baseline results\\n\\n"
    "Model: google/gemma-4-e4b-it-qat-w4a16-ct (QAT 4-bit, Kaggle 2x T4)\\n"
    "Date: (fill in)\\n\\n"
    "## Dev bugs (10 synthetic)\\n\\n"
    + agent.results_table(results)
    + "\\n## Real bugs (11 from requests/httpx/rich)\\n\\n"
    + agent.results_table(real_results)
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
