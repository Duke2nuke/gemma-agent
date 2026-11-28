"""Build a LoRA fine-tuning dataset from the 21 verified bugs.

Each example is one (prompt, fixed file) pair in exactly the agent's
inference format: the prompt is agent.build_prompt(buggy code, tests,
failing output) and the response is the complete fixed file in one
```python block -- the same shape agent.extract_code() parses at run
time. Training on this format teaches the model the task it will do
in the harness: read buggy file + tests + failure, reply with only
the corrected file.

Every fix is verified here, not trusted: the buggy file must FAIL its
tests and the fixed file must PASS them, in a fresh temp dir, before
the pair is written. Run: python3 build_lora_dataset.py
Writes: lora_dataset.jsonl (21 examples) + lora_dataset_build.log
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import agent  # noqa: E402

ROOT = Path(__file__).parent

# --- Synthetic fixes (one targeted replacement each) -----------------------

SYNTHETIC_FIXES = {
    "bug_01_off_by_one": ("range(len(xs) - 1)", "range(len(xs))"),
    "bug_02_wrong_comparison": ("return age > 18", "return age >= 18"),
    "bug_03_mutable_default": (
        "def add_item(item, cart=[]):\n    cart.append(item)",
        "def add_item(item, cart=None):\n    if cart is None:\n        cart = []\n    cart.append(item)",
    ),
    "bug_04_missing_key": ("return PRICES[item]", "return PRICES.get(item, 0)"),
    "bug_05_wrong_sort": (
        "return sorted(scores)[:3]",
        "return sorted(scores, reverse=True)[:3]",
    ),
    "bug_06_type_error": (
        'return CONFIG["price"] * (1 - CONFIG["discount_pct"] / 100)',
        'return float(CONFIG["price"]) * (1 - CONFIG["discount_pct"] / 100)',
    ),
    "bug_07_missing_import": (
        '"""Load a JSON string and return the value for a key."""',
        '"""Load a JSON string and return the value for a key."""\n\n\nimport json',
    ),
    "bug_08_shared_state": (
        "_seen = []\n\n\ndef evens(xs):\n    for x in xs:\n        if x % 2 == 0:\n            _seen.append(x)\n    return _seen",
        "def evens(xs):\n    out = []\n    for x in xs:\n        if x % 2 == 0:\n            out.append(x)\n    return out",
    ),
    "bug_09_slice_off_by_one": (
        "return items[start:start + size - 1]",
        "return items[start:start + size]",
    ),
    "bug_10_mutates_input": ("out = scores", "out = list(scores)"),
}

# --- Real-bug fixes (from real_bugs/README.md reference fixes) -------------

REAL_FIXES = {
    "real_01_requests_content_type": (
        """            key, value = param, True
            index_of_equals = param.find("=")
            if index_of_equals != -1:
                key = param[:index_of_equals].strip(items_to_strip)
                value = param[index_of_equals + 1 :].strip(items_to_strip)
            params_dict[key.lower()] = value""",
        """            index_of_equals = param.find("=")
            if index_of_equals != -1:
                key = param[:index_of_equals].strip(items_to_strip)
                value = param[index_of_equals + 1 :].strip(items_to_strip)
                params_dict[key.lower()] = value""",
    ),
    "real_02_requests_super_len": (
        """    total_length = None
    current_position = 0
""",
        """    total_length = None
    current_position = 0

    if isinstance(o, str):
        o = o.encode("utf-8")
""",
    ),
    "real_03_requests_netrc_auth": (
        "            if _netrc:",
        "            if _netrc and any(_netrc):",
    ),
    "real_04_httpx_primitive_value": (
        '    return str(value)',
        """    if isinstance(value, (str, int, float)):
        return str(value)
    raise TypeError(\n        f"Expected str, int, float, bool, or None. Got '{type(value).__name__}'."\n    )""",
    ),
    "real_05_httpx_text_chunker": (
        """        if self._chunk_size is None:
            return [content]""",
        """        if self._chunk_size is None:
            return [content] if content else []""",
    ),
    "real_06_httpx_no_proxy": (
        """import os
import typing
from urllib.request import getproxies""",
        """import ipaddress
import os
import typing
from urllib.request import getproxies""",
    ),
    "real_07_httpx_urlencode_space": (
        "    return _stdlib_urlencode(items)",
        """    return "&".join(
        f"{quote(str(key))}={quote(str(value))}" for key, value in items
    )""",
    ),
    "real_08_httpx_ipv4_regex": (
        r'IPv4_STYLE_HOSTNAME = re.compile(r"^[0-9]+.[0-9]+.[0-9]+.[0-9]+$")',
        r'IPv4_STYLE_HOSTNAME = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$")',
    ),
    "real_09_httpx_gen_delims": (
        """    parsed_path: str = quote(path, safe=SUB_DELIMS + ":@/")
    parsed_query: typing.Optional[str] = (
        None if query is None else quote(query, safe=SUB_DELIMS + "/?")
    )
    parsed_fragment: typing.Optional[str] = (
        None if fragment is None else quote(fragment, safe=SUB_DELIMS + "/?")
    )""",
        """    parsed_path: str = quote(path, safe=SUB_DELIMS + ":/[]@")
    parsed_query: typing.Optional[str] = (
        None if query is None else quote(query, safe=SUB_DELIMS + ":/?[]@")
    )
    parsed_fragment: typing.Optional[str] = (
        None if fragment is None else quote(fragment, safe=SUB_DELIMS + ":/?#[]@")
    )""",
    ),
    # real_10 / real_11 are handled in apply_fix by anchor edits: their
    # regex lines are backslash-heavy and exact-match patterns are fragile
    # across escaping layers. The anchor edits below are equivalent.
}

# real_06 needs a second edit (the no_proxy loop body) on top of the import.
REAL_06_LOOP = (
    """        elif hostname:
            # NO_PROXY=.google.com is marked as "all://*.google.com,
            #   which disables "www.google.com" but not "google.com"
            # NO_PROXY=google.com is marked as "all://*google.com,
            #   which disables "www.google.com" and "google.com".
            #   (But not "wwwgoogle.com")
            mounts[f"all://*{hostname}"] = None""",
    """        elif hostname:
            # NO_PROXY=.google.com is marked as "all://*.google.com,
            #   which disables "www.google.com" but not "google.com"
            # NO_PROXY=google.com is marked as "all://*google.com,
            #   which disables "www.google.com" and "google.com".
            #   (But not "wwwgoogle.com")
            # IPs and localhost are exact mounts, not domain suffixes.
            if hostname == "localhost":
                mounts[f"all://{hostname}"] = None
            else:
                try:
                    ip = ipaddress.ip_address(hostname)
                except ValueError:
                    mounts[f"all://*{hostname}"] = None
                else:
                    mounts[
                        f"all://[{hostname}]"
                        if ip.version == 6
                        else f"all://{hostname}"
                    ] = None""",
)


def apply_fix(name: str, code: str) -> str:
    if name == "real_10_rich_escape":
        # Add @ to the tag-start class: [a-z#\/] -> [a-z#@\/]
        anchor = "[a-z#"
        assert code.count(anchor) == 1, "real_10 anchor"
        i = code.index(anchor) + len(anchor)
        return code[:i] + "@" + code[i:]
    if name == "real_11_rich_markup_parse":
        # A tag cannot span '[': .*? -> [^[]*?
        anchor = "].*?)"
        assert code.count(anchor) == 1, "real_11 anchor"
        return code.replace(anchor, "][^[]*?)", 1)
    fixes = dict(SYNTHETIC_FIXES)
    fixes.update(REAL_FIXES)
    old, new = fixes[name]
    assert old in code, f"{name}: fix pattern not found"
    assert code.count(old) == 1, f"{name}: fix pattern matches {code.count(old)}x"
    code = code.replace(old, new)
    if name == "real_06_httpx_no_proxy":
        old2, new2 = REAL_06_LOOP
        assert old2 in code and code.count(old2) == 1, "real_06 loop pattern"
        code = code.replace(old2, new2)
    return code


import os

TEST_PYTHON = os.environ.get("GEMMA_TEST_PYTHON", sys.executable)


def run_tests(workdir: Path, test_file: Path) -> tuple[bool, str]:
    proc = subprocess.run(
        [TEST_PYTHON, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(test_file)],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc.returncode == 0, proc.stdout + proc.stderr


def main() -> None:
    cases = []
    for bug_file in sorted((ROOT / "bugs").glob("bug_*.py")):
        cases.append(("bugs", bug_file))
    for bug_file in sorted((ROOT / "real_bugs").glob("real_*.py")):
        cases.append(("real_bugs", bug_file))

    log: list[str] = []
    examples = []
    for dirname, bug_file in cases:
        name = bug_file.stem
        bug_dir = ROOT / dirname
        test_file = bug_dir / "tests" / f"test_{name}.py"
        conftest = bug_dir / "tests" / "conftest.py"
        code = bug_file.read_text()
        test_code = test_file.read_text()
        fixed = apply_fix(name, code)

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / name
            testdir = workdir / "tests"
            testdir.mkdir(parents=True)
            work_bug = workdir / bug_file.name
            work_test = testdir / test_file.name
            if conftest.exists():
                shutil.copy(conftest, testdir / "conftest.py")
            shutil.copy(test_file, work_test)

            work_bug.write_text(code)
            buggy_pass, buggy_out = run_tests(workdir, work_test)
            assert not buggy_pass, f"{name}: buggy version unexpectedly PASSES"

            work_bug.write_text(fixed)
            fixed_pass, _ = run_tests(workdir, work_test)
            assert fixed_pass, f"{name}: fixed version FAILS its tests"

            prompt = agent.build_prompt(code, test_code, buggy_out)
            examples.append(
                {
                    "bug": name,
                    "prompt": prompt,
                    "response": "```python\n" + fixed + "```",
                }
            )
            log.append(f"OK {name}: buggy FAIL -> fixed PASS")

    out = ROOT / "lora_dataset.jsonl"
    with out.open("w") as fh:
        for ex in examples:
            fh.write(json.dumps(ex) + "\n")
    (ROOT / "lora_dataset_build.log").write_text("\n".join(log) + "\n")
    print(f"wrote {out} with {len(examples)} examples")
    print("\n".join(log))


if __name__ == "__main__":
    main()
