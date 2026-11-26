"""Agent loop for the Gemma 4 Developer Agent entry.

The loop is model-agnostic: the caller passes a ``generate`` callable that
takes a prompt string and a ``greedy`` flag and returns model text. The
Kaggle notebook wires that callable up to a locally loaded Gemma 4 E4B
(QAT 4-bit) model. Nothing here makes network calls.

Why greedy first, then sampled: run 1 (5 easy bugs) was 5/5, run 2 (same
bugs + 5 harder) was 1/10 with identical prompt, loop and generation
settings — pure sampling variance. So each round now tries up to 3
candidate fixes: a greedy (deterministic) fix first, then up to 2 sampled
fixes (temperature 0.7). Each candidate is tested against the bug's tests
and the first one that passes is kept. A "round" is one attempt in the
results table (max 3 rounds per bug, so up to 9 model calls per bug worst
case).

Flow per bug:
    1. copy the buggy file + its tests into a fresh working directory
    2. run the tests (they should fail at first)
    3. per round: ask the model for a fix (greedy, then sampled), test
       each candidate, keep the first that passes
    4. retry up to ``max_attempts`` rounds, then record pass/fail
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BugResult:
    """One bug, one row of the baseline table."""

    name: str
    attempts: int = 0
    passed: bool = False
    log: list[str] = field(default_factory=list)


CODE_BLOCK_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Pull the first python code block out of model output.

    Falls back to the whole output if there is no fenced block, so a model
    that just prints code still works.
    """
    match = CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip() + "\n"
    return text.strip() + "\n"


def build_prompt(code: str, test_code: str, test_output: str) -> str:
    """Prompt the model with the buggy file, its tests, and the failure."""
    return (
        "You are fixing a small Python bug. Reply with ONLY the complete "
        "corrected file inside one ```python code block, nothing else.\n\n"
        "The file is supposed to do what its docstring says. "
        "The tests below describe the correct behaviour.\n\n"
        "BUGGY FILE:\n```python\n" + code + "```\n\n"
        "TESTS:\n```python\n" + test_code + "```\n\n"
        "FAILING TEST OUTPUT (last 40 lines):\n```\n"
        + "\n".join(test_output.strip().splitlines()[-40:])
        + "\n```\n\n"
        "CORRECTED FILE:\n```python\n"
    )


def run_tests(test_path: Path, cwd: Path, timeout: int = 120) -> tuple[bool, str]:
    """Run one test file with pytest. Returns (passed, combined output)."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(test_path)],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode == 0, proc.stdout + proc.stderr


def run_agent(
    generate,
    bug_file: Path,
    test_file: Path,
    work_root: Path,
    max_attempts: int = 3,
    candidates_per_round: int = 3,
) -> BugResult:
    """Run the fix loop for one bug. Edits only a copy, never the original.

    Each round tries up to ``candidates_per_round`` fixes: the first is
    greedy (deterministic), the rest are sampled. Each candidate is tested
    and the first one that passes wins the round. A round counts as one
    attempt in the results table.
    """
    name = bug_file.stem
    result = BugResult(name=name)
    workdir = work_root / name
    testdir = workdir / "tests"
    testdir.mkdir(parents=True, exist_ok=True)

    work_bug = workdir / bug_file.name
    work_test = testdir / test_file.name
    shutil.copy(bug_file, work_bug)
    shutil.copy(test_file, work_test)
    # conftest.py makes the bug module importable from the tests dir.
    conftest = test_file.parent / "conftest.py"
    if conftest.exists():
        shutil.copy(conftest, testdir / "conftest.py")

    code = work_bug.read_text()
    test_code = work_test.read_text()

    passed, output = run_tests(work_test, cwd=workdir)
    result.log.append("attempt 0 (original): " + ("PASS" if passed else "FAIL"))
    if passed:
        result.passed = True
        return result

    for attempt in range(1, max_attempts + 1):
        result.attempts = attempt
        prompt = build_prompt(code, test_code, output)
        last_fixed = None
        for cand in range(candidates_per_round):
            greedy = cand == 0
            mode = "greedy" if greedy else "sampled"
            try:
                fixed = extract_code(generate(prompt, greedy=greedy))
            except Exception as exc:  # model call failed, try next candidate
                result.log.append(
                    f"attempt {attempt} candidate {cand + 1} ({mode}): "
                    f"generate() raised {exc!r}"
                )
                continue
            last_fixed = fixed
            work_bug.write_text(fixed)
            passed, output = run_tests(work_test, cwd=workdir)
            result.log.append(
                f"attempt {attempt} candidate {cand + 1} ({mode}): "
                + ("PASS" if passed else "FAIL")
            )
            if passed:
                result.passed = True
                return result
        # Next round refines from the last candidate tried, as before.
        if last_fixed is not None:
            code = last_fixed
        work_bug.write_text(code)

    return result


def run_all(
    generate,
    bugs_dir: Path,
    work_root: Path,
    max_attempts: int = 3,
    candidates_per_round: int = 3,
) -> list[BugResult]:
    """Run the agent over every bug_*.py in bugs_dir (sorted by name)."""
    results = []
    for bug_file in sorted(bugs_dir.glob("bug_*.py")):
        test_file = bugs_dir / "tests" / f"test_{bug_file.stem}.py"
        if not test_file.exists():
            print(f"SKIP {bug_file.name}: no {test_file.name}")
            continue
        print(f"--- {bug_file.name} ---")
        res = run_agent(
            generate, bug_file, test_file, work_root, max_attempts, candidates_per_round
        )
        print(f"    {'PASS' if res.passed else 'FAIL'} after {res.attempts} attempt(s)")
        results.append(res)
    return results


def results_table(results: list[BugResult]) -> str:
    """Markdown table for baseline_results.md."""
    lines = [
        "| Bug | Attempts | Result |",
        "| --- | --- | --- |",
    ]
    for res in results:
        lines.append(f"| {res.name} | {res.attempts} | {'pass' if res.passed else 'fail'} |")
    passed = sum(r.passed for r in results)
    lines.append("")
    lines.append(f"Pass rate: {passed}/{len(results)}")
    return "\n".join(lines) + "\n"
