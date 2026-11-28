"""Build train_lora.ipynb: LoRA fine-tune of the QAT E4B base.

DRAFT — never run on a GPU. The training step (PEFT on the compressed-
tensors QAT checkpoint, inside the multimodal Gemma4ForConditional-
Generation model) is the one part of this repo not yet verified in a
real session. Verify it in the Kaggle session before trusting a run:
if prepare_model_for_kbit_training fails on this model, stop and report
the error instead of working around it.

What it does, in order:
1. Loads the QAT 4-bit E4B checkpoint exactly like gemma_agent.ipynb
   (same discovery, device_map={"": 0}, eager, low_cpu_mem_usage).
2. Attaches a LoRA adapter (r=16) to the language-model projections.
3. Trains on lora_dataset.jsonl (21 verified pairs, embedded here, so
   no dataset upload is needed) with prompt tokens masked: the model
   learns "buggy file + tests + failure -> corrected file", the same
   format the agent uses at inference time.
4. Saves the adapter to /kaggle/working/adapters/main_lora and builds
   submission.zip with the adapter included (agent.yaml gains
   `adapter: main_lora`).

Data warning: 21 pairs is tiny. 3 epochs at r=16 is a deliberately
small dose — enough to teach the output format and the common fix
shapes, not enough to memorise the hidden tasks. If the post-training
check shows the model parroting these 21 fixes verbatim on new bugs,
the dose was too big; cut epochs to 1 or r to 8.

Run: python3 build_train_notebook.py
Output: train_lora.ipynb (same directory)
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent

DATASET = [json.loads(line) for line in (ROOT / "lora_dataset.jsonl").read_text().splitlines() if line.strip()]

SUBMISSION_FILES = {}
for p in sorted((ROOT / "submission").rglob("*")):
    if p.is_file():
        SUBMISSION_FILES[str(p.relative_to(ROOT / "submission"))] = p.read_text()

SETUP_CELL = '''import sys
from pathlib import Path

# Embedded by build_train_notebook.py: the verified 21-pair dataset and
# the submission/ source files. Written to /kaggle/working so the
# notebook runs from a single upload. ensure_ascii=False is REQUIRED:
# the real_02 test contains an emoji, which would otherwise become
# surrogate-pair escapes that break Path.write_text.
DATASET = ''' + json.dumps(DATASET, indent=1, ensure_ascii=False) + '''
SUBMISSION_FILES = ''' + json.dumps(SUBMISSION_FILES, indent=1, ensure_ascii=False) + '''

work = Path("/kaggle/working")
with open(work / "lora_dataset.jsonl", "w") as fh:
    for ex in DATASET:
        fh.write(json.dumps(ex, ensure_ascii=False) + "\\n")
for rel, content in SUBMISSION_FILES.items():
    path = work / "submission" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
print(f"dataset: {len(DATASET)} examples, submission files: {len(SUBMISSION_FILES)}")
'''

LOAD_CELL = '''import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Same load as gemma_agent.ipynb: QAT 4-bit E4B, single-device map.
MODEL_ID = "google/gemma-4-e4b-it-qat-w4a16-ct"

model_path = None
from pathlib import Path as _P
if _P("/kaggle/input").exists():
    for _d in sorted(_P("/kaggle/input").glob("**/config.json")):
        if "gemma" in str(_d).lower():
            model_path = str(_d.parent)
            print("using attached model input:", model_path)
            break
if model_path is None:
    try:
        import kagglehub
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
    # Do NOT force bf16: it dequantizes the QAT 4-bit weights to ~18 GB
    # and generate() OOMs. Let compressed-tensors load native 4-bit.
    # Single-device map: device_map="auto" on a 4-GPU node runs
    # accelerate dispatch and leaves the top-level model with a wrapped
    # forward but no _hf_hook, so generate() raises AttributeError.
    device_map={"": 0},
    # MUST stay eager: Gemma4AudioModel does not support SDPA.
    attn_implementation="eager",
    low_cpu_mem_usage=True,
)
model.eval()
first = next(model.parameters())
total = sum(p.numel() for p in model.parameters())
print("model loaded:", load_from)
print(f"first param dtype: {first.dtype}, total params: {total / 1e9:.2f}B")
print("has _hf_hook:", hasattr(model, "_hf_hook"))
print("hf_device_map:", getattr(model, "hf_device_map", None))
print("model.device:", model.device)
'''

TRAIN_CELL = '''import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset

# --- Prepare the quantized model for LoRA training -------------------------
# DRAFT-ONLY step: if prepare_model_for_kbit_training raises on this
# multimodal checkpoint, STOP and report the error. Do not work around it
# with a manual hook or a different load: a wrong preparation silently
# trains the wrong weights.
model.config.use_cache = False
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# --- Tokenized dataset: prompt masked, model learns the response -----------
MAX_LEN = 4096

class FixDataset(Dataset):
    def __init__(self, examples):
        self.rows = []
        for ex in examples:
            prompt_ids = tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]
            response_ids = tokenizer(ex["response"], add_special_tokens=False)["input_ids"]
            ids = (prompt_ids + response_ids)[:MAX_LEN]
            labels = ([-100] * len(prompt_ids) + response_ids)[:MAX_LEN]
            self.rows.append({"input_ids": ids, "labels": labels})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(batch):
    maxlen = max(len(r["input_ids"]) for r in batch)
    pad = tokenizer.pad_token_id or 0
    input_ids, labels, attn = [], [], []
    for r in batch:
        n = len(r["input_ids"])
        input_ids.append(r["input_ids"] + [pad] * (maxlen - n))
        labels.append(r["labels"] + [-100] * (maxlen - n))
        attn.append([1] * n + [0] * (maxlen - n))
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attn),
    }


train_ds = FixDataset(DATASET)
print(f"training examples: {len(train_ds)}")
print("max token length:", max(len(r["input_ids"]) for r in train_ds.rows),
      f"(truncated to {MAX_LEN} if larger — if any example is truncated, its")
print("response is cut mid-file and teaches a truncated output; lower the")
print("prompt size or raise MAX_LEN before training in that case.")

args = TrainingArguments(
    output_dir="/kaggle/working/lora_out",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=1,
    save_strategy="no",
    bf16=True,
    report_to=[],
)
trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collate)
trainer.train()
print("training done")
'''

SAVE_CELL = '''from pathlib import Path

# Save only the adapter (PEFT format the harness expects).
out = Path("/kaggle/working/adapters/main_lora")
out.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(out))
print("adapter saved to", out)
for p in sorted(out.iterdir()):
    print(f"  {p.name}: {p.stat().st_size / 1e6:.2f} MB")
assert (out / "adapter_config.json").exists(), "adapter_config.json missing"
assert (out / "adapter_model.safetensors").exists(), "adapter weights missing"
'''

ZIP_CELL = '''import shutil
import zipfile
from pathlib import Path

# Build submission.zip v3: submission/ sources + the trained adapter.
# agent.yaml gains `adapter: main_lora` (harness field for the PEFT
# adapter under adapters/<name>/). If the harness ever rejects the
# adapter line, the fallback is the same zip without it (base model).
work = Path("/kaggle/working")
src = work / "submission"
agent_yaml = (src / "agent.yaml").read_text()
assert "adapter:" not in agent_yaml, "agent.yaml already has an adapter line"
agent_yaml = agent_yaml.replace(
    "model: gemma-4-e4b-it-qat-w4a16-ct",
    "model: gemma-4-e4b-it-qat-w4a16-ct\\nadapter: main_lora",
    1,
)
(src / "agent.yaml").write_text(agent_yaml)

zip_path = work / "submission.zip"
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(src.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(src))
    for p in sorted((work / "adapters").rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(work))
with zipfile.ZipFile(zip_path) as z:
    bad = z.testzip()
    print("zip entries:")
    for info in z.infolist():
        print(f"  {info.filename} ({info.file_size} bytes)")
    print("bad file:", bad, "| size bytes:", zip_path.stat().st_size)
    assert bad is None
    assert any(n == "agent.yaml" for n in z.namelist()), "agent.yaml not at zip root"
    assert any("adapter_model.safetensors" in n for n in z.namelist()), "adapter missing from zip"
print("saved to /kaggle/working/submission.zip — download it and upload to Kaggle")
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
    md("""# Gemma 4 Developer Agent — LoRA fine-tuning (DRAFT)

Fine-tune the QAT 4-bit E4B base on the 21 verified before/after fix
pairs (`lora_dataset.jsonl`, embedded in this notebook), then build a
submission.zip that includes the trained adapter.

**Status: DRAFT — never run on a GPU.** The PEFT step on this QAT
multimodal checkpoint is unverified. In the session, run the cells one
at a time and check each print before continuing. If
`prepare_model_for_kbit_training` raises, STOP and report the error.

**Before running:**
0. Restart the kernel first if this session already ran a notebook —
   the old model otherwise stays in GPU memory.
1. Notebook Settings → Accelerator → **GPU L4 x4 (required)**. Attach
   the QAT model as an input first (right panel → Input → Add input →
   Models → `e4b-it-qat-w4a16-ct`). L4 has no internet, so the attached
   input is required. No other input is needed: the dataset and the
   submission sources are embedded in this notebook.
2. Accept the Gemma license on the `google/gemma-4` model page, or add
   a Hugging Face token as a Kaggle secret named `HF_TOKEN`.
3. Run the cells top to bottom. Training is small (21 pairs, 3 epochs,
   effective batch 8) and should take minutes, not hours.

**Data warning:** 21 pairs is tiny. This dose (r=16, 3 epochs) is meant
to teach the output format and common fix shapes. If the trained model
parrots these 21 fixes on new bugs, the dose was too big — cut to
1 epoch or r=8."""),
    code("""import torch

print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name}, {p.total_memory / 1e9:.1f} GB")
assert torch.cuda.is_available(), "Enable GPU in Notebook Settings (L4 x4)"
"""),
    code("""# peft is the only extra dependency. L4 sessions have NO internet, so
# it cannot be installed there — it must already be in the image.
# This cell checks that and stops with instructions if it is missing,
# instead of failing later inside the training cell.
try:
    import peft
    print("peft", peft.__version__, "- ready")
except ImportError:
    print("peft is NOT installed and L4 has no internet, so it cannot be installed here.")
    print("Options: run this notebook on a GPU session WITH internet (T4x2)")
    print("so that %pip install -q peft works, or attach peft as an input.")
    print("Do not continue to the training cell until peft imports.")
    raise SystemExit("peft missing")
"""),
    code(SETUP_CELL),
    code(LOAD_CELL),
    code(TRAIN_CELL),
    code(SAVE_CELL),
    code(ZIP_CELL),
    md("""## After training — evaluate before submitting

Do NOT submit this zip blind. Two checks, in order:
1. **Re-run the baseline notebook with the adapter loaded** (load the
   base model, then `PeftModel.from_pretrained(model,
   "/kaggle/input/.../main_lora")`) over the same 10 + 11 bugs. The
   adapter should not score *worse* than the base baseline.
2. **Submit to Kaggle** (1 submission/day, 2 final slots — keep one
   slot for the best version). The daily leaderboard is the real test;
   a local pass rate on 21 known bugs is not a leaderboard score.

If the adapter makes things worse, submit the base-model v2 zip
instead — it is already built and validated."""),
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

out = ROOT / "train_lora.ipynb"
out.write_text(json.dumps(notebook, indent=1))
print(f"wrote {out} ({len(cells)} cells)")
