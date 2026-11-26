# Real-bug evaluation set

11 real Python bugs, each extracted verbatim from the code of one of
the competition's four repositories (psf/requests, encode/httpx,
Textualize/rich) as it was immediately before the merged fix. Every
bug ships as a single standalone, stdlib-only file with 3 tests. The
tests fail on the buggy version and pass on the reference-fixed
version (reference fixes are kept outside this directory; see
Verification below).

No FastAPI bug is included: every small FastAPI fix found in the
history window either needs Pydantic/Starlette machinery
(`jsonable_encoder`, dependency utils) or touches many files, so none
of them extracts to a standalone stdlib-only file.

Run the suite with:

```bash
python -m pytest real_bugs/tests/ -v
```

## The bugs

| # | Source | Bug | Reference fix |
|---|---|---|---|
| real_01 | requests `utils._parse_content_type_header`, PR #7309 (2026-03-30) | A Content-Type parameter with no `=` (e.g. `; no_equals`) is stored as `True` instead of being ignored | Only store params that contain `=` |
| real_02 | requests `utils.super_len`, issue #6586 (2023-11-28) | A `str` body is measured in characters, not UTF-8 bytes, so Content-Length is too small for non-ASCII text | Encode `str` to UTF-8 first |
| real_03 | requests `utils.get_netrc_auth`, PR #7205 (2026-02-12) | An empty netrc entry `(None, None, None)` is truthy, so auth returns `(None, None)` instead of `None` | Require `any(_netrc)` |
| real_04 | httpx `utils.primitive_value_to_str`, PR #2523 (2022-12-30) | Any value is silently `str()`-ed, so `QueryParams({"a": b"bytes"})` gives `a=b'bytes'` | Raise `TypeError` for non str/int/float/bool/None |
| real_05 | httpx `decoders.TextChunker.decode`, issue #2998 (2023-12-11) | `decode("")` returns `[""]`, so iterating an empty response yields one empty string | Return `[]` for empty content |
| real_06 | httpx `utils.get_environment_proxies`, PR #2659 (2023-04-19) | Every NO_PROXY entry is treated as a domain suffix (`127.0.0.1` → `all://*127.0.0.1`) | Map IPs, IPv6 and `localhost` to exact mounts |
| real_07 | httpx `urls.QueryParams.__str__` / `urlparse.urlencode`, issue #2536 (2023-01-10) | stdlib `urlencode` encodes spaces as `+` and `/` as `%2F` | Custom `urlencode` using `quote` (`%20`, `/` safe) |
| real_08 | httpx `urlparse.encode_host`, PR #2886 (2023-10-10) | The IPv4-detection regex uses unescaped `.`, so `023b76x43144` is treated as an IPv4 address and rejected | Escape the dots (`\.`) |
| real_09 | httpx `urlparse.urlparse` quoting step, PR #2701 (2023-05-09) | The `safe` sets for path/query/fragment drop legal gen-delims (`:`, `[`, `]`, `@`, `#`), which then get percent-encoded | Per-component safe sets: path `:/[]@`, query `:/?[]@`, fragment `:/?#[]@` (plus `SUB_DELIMS`) |
| real_10 | rich `markup.escape`, v10.16.2 (2022-01-02) | The escape regex has no `@` in the tag-start class, so `[@click]` is not escaped | Add `@` to the class |
| real_11 | rich `markup._parse` / `RE_TAGS`, commit 71325bb (2022-02-10) | `.*?` in the tag regex spans across `[`, so `"[foo bar [baz] qux]"` parses as one tag | Use `[^[]*?` so a tag cannot span `[` |

## Verification

Each bug was verified locally: the distinguishing test(s) fail on the
buggy file and the full 3-test file passes on the reference-fixed
copy. The fixed copies live outside the repo (they must never ship
in `real_bugs/`). To re-verify, copy `real_bugs/` to a temp dir,
apply each reference fix from the table above, and run the suite
against both trees.
