AI-Disclosure: Claude (Opus 4.7, 1M context) authored this security review; Grant reviewed.

# Security Review — Slice 2 Step 1 (ADR-015 catalog scan-time helper)

**Date:** 2026-05-23
**Reviewer:** `security-auditor` (Opus 4.7, 1M context), model pinned at invocation.
**Target:** Uncommitted diff implementing slice-2 Step 1 of ADR-015 — NEW `comfyless/catalog.py` (single helper `scan_model_family(model_dir) -> Optional[str]`); MOD `test_mcp_server.py` (characterization tests).

---

## Round 1

### Summary

ADR-015 slice 2 Step 1 adds a new module `comfyless/catalog.py` containing one helper, `scan_model_family(model_dir) -> Optional[str]`, plus characterization tests in `test_mcp_server.py`. The helper opens `<model_dir>/model_index.json` (operator-installed contents, potentially attacker-influenced if an operator installs a malicious package) and returns `infer_model_family(_class_name)` or `None` on any malformed input. Threat model is stdio same-uid; the relevant adversary is an attacker who can place a hostile `model_index.json` under `--model-base`, and a misbehaving LLM agent at request time that can later cause this helper to be invoked (Step 2). The function is the only public surface added; module-level work is one stdlib import and one named import from `nodes.eric_diffusion_utils`.

Reviewer worked through trust boundaries (file content from operator-controlled tree, return value flowing to the future catalog), each error path (`isfile`, `open`, `json.load`, type validation), absence (size cap, encoding pin, allow-list of return strings, symlink discipline), and verified scope. The posture is good for Step 1: the helper genuinely fails closed to `None` for the catalog-builder caller, never raises a propagating exception in the documented modes, and adheres to Vision invariant 14 (no edits to `nodes/eric_diffusion_utils.py` or `comfyless/generate.py`). Findings below are gaps worth marking before Step 2 builds on top of this surface.

### Coverage

Reviewed:
- `comfyless/catalog.py:1-94` (entire new module)
- `nodes/eric_diffusion_utils.py:1-84` (caller of `infer_model_family`, callee of `detect_pipeline_class` — confirmed pre-existing behavior on `_class_name` content)
- `test_mcp_server.py:40-55, 1230-1378` (new test block + imports)
- `comfyless/__init__.py` (confirmed import-time shim behavior into which `catalog` plugs cleanly)

Not reviewed (out of scope per prompt):
- `docs/decisions/ADR-015-*` and `docs/vision/slice-2-mcp-catalog.md` (prior reviews cover the design; this review covers code)
- `comfyless/mcp_server.py` Step 3 wiring (later step)
- HTTP-transport concerns (explicitly out of scope per threat model)

### Findings

#### [HIGH-1] Module-import side effect: importing `comfyless.catalog` transitively imports torch
Location: `comfyless/catalog.py:44` (`from nodes.eric_diffusion_utils import infer_model_family`).
Risk: `nodes/eric_diffusion_utils.py` does `import torch` at module top level (line 14). Importing `comfyless.catalog` therefore performs a heavyweight, hard-to-reverse side effect at every consumer that touches the catalog surface — including, by Step 3, MCP server startup before `--catalog` flag handling and before any startup fail-closed validation completes. This couples spawn-time control flow to torch import success/failure, expands the attack surface a malicious sitecustomize/PYTHONPATH could reach via this otherwise-purely-stdlib module, and makes the catalog scan walker (Step 2) impossible to import in a context that wants only the family-classification primitive (e.g. a future audit tool). It also defeats the explicit "scan-time independence from the operator's diffusers installation" property the docstring at lines 56-60 advertises — the module silently still drags torch in.
Remediation: import `infer_model_family` lazily inside `scan_model_family`. Preserves the single-source-of-truth coupling and removes the import-time torch cost.

#### [MEDIUM-1] `open()` has no size limit; Step 2 will iterate this over every directory under `--model-base`
Location: `comfyless/catalog.py:84-85`.
Risk: A malicious or accidentally-bloated `model_index.json` (e.g. a multi-GB file or a deeply nested JSON document) under `--model-base` is parsed in full by `json.load`. At Step 1 the helper is called explicitly, so this is single-shot at most; but Step 2 will fan this out over every candidate directory at spawn. A single hostile package installed by the operator is enough to cause spawn-time OOM or extreme stall before fail-closed startup validation completes — and the failure mode looks like "MCP server hangs at startup" rather than a clean error. Even in a same-uid model the spawn-time DoS matters because the MCP surface's only fail-closed posture is at startup. Marking for Step 1 rather than Step 2 because the helper signature and the open() call are what Step 2 will rely on — pushing a size cap up to the helper is cheaper now than retrofitting later.
Remediation: bound the read explicitly. Smallest targeted change: replace lines 84-85 with `with open(index_path, "rb") as f: data = f.read(_MAX_INDEX_BYTES + 1)` and a length check before `json.loads(data)`, where `_MAX_INDEX_BYTES` is a module-level constant (e.g. 1 MiB — `model_index.json` is canonically tiny). On overflow return `None` like the other malformed cases.

#### [MEDIUM-2] Encoding not pinned: `open()` uses locale-dependent encoding
Location: `comfyless/catalog.py:84`.
Risk: `open(index_path)` opens in text mode with the platform's `locale.getpreferredencoding()`. On hosts where `LANG`/`LC_*` is unset or set to a non-UTF-8 value (a misconfigured systemd unit, a container without explicit locale, a launchd plist under macOS), a perfectly valid UTF-8 `model_index.json` containing non-ASCII bytes raises `UnicodeDecodeError`, which the existing `except` does catch — so the helper returns `None` and the directory becomes silently invisible to the catalog. That is fail-closed for the catalog (good) but the operator gets no diagnostic at startup: the model just vanishes from `list_models`. The docstring says the helper returns `None` "if the file is unreadable / malformed JSON / not UTF-8" but the actual behavior is "not in the platform default encoding."
Remediation: pass `encoding="utf-8"` to `open()`. JSON spec is UTF-8; pinning removes locale dependence and makes the docstring's promise literally true.

#### [INFO-1] `os.path.isfile()` follows symlinks; not a Step 1 finding given threat model
Location: `comfyless/catalog.py:81`.
Risk: `isfile` returns True for a symlink pointing at any regular file the process can read. Under same-uid stdio, the attacker is already the same user, so symlink redirection to read `/etc/passwd`-class files buys nothing — the contents would still have to parse as JSON with a `_class_name` field. The Vision invariant 17 "no-follow-symlinks" rule is documented as belonging to the Step 2 scan walker, not this helper. Recording so Step 2's walker doesn't lose track of it.
Remediation: none for Step 1. When the walker lands in Step 2, prefer `os.lstat` / `os.scandir(follow_symlinks=False)` for directory traversal, and consider an explicit `os.path.islink(index_path)` check before opening if Vision invariant 17 is meant to extend to the file itself.

#### [INFO-2] Adversarial `_class_name` content is a pre-existing concern on the load path
Location: `nodes/eric_diffusion_utils.py:45-51` (and now `comfyless/catalog.py:93`).
Risk: `infer_model_family` returns the lowercase-normalized `class_name` verbatim on no-match. A `_class_name` containing control characters, ANSI escapes, RTL-override Unicode (U+202E), zero-width joiners, or a 10 MB string passes through unchanged. Downstream this flows into log lines, the `list_models` MCP response, and — in Step 2 — possibly catalog name components. This is pre-existing on the load paths, so per prompt rule "if pre-existing it's not a Step 1 finding."
Remediation: none for Step 1. Step 2's `build_catalog()` should validate / reject family strings it intends to embed in catalog names, and Step 4's `list_models` formatter should sanitize for log/JSON output.

#### [INFO-3] No scope creep; Vision invariant 14 honored
Location: `comfyless/catalog.py` (new); `test_mcp_server.py` (mod).
Confirmation: Only the new module and the test file are edited. `nodes/eric_diffusion_utils.py` and `comfyless/generate.py` are unchanged.

### Round 1 verdict

**CHANGES_REQUIRED.** No CRITICAL.

- HIGH × 1 — module-import side effect drags torch into `comfyless.catalog`. Move the import inside `scan_model_family`.
- MEDIUM × 2 — (a) unbounded `open()` + `json.load()` size; cap at ~1 MiB before Step 2 fans this over every model dir. (b) `open()` without `encoding="utf-8"` makes the helper's contract locale-dependent.
- INFO × 3 — symlink handling deferred to Step 2 walker (acceptable), `infer_model_family` returning unsanitized strings is pre-existing on load paths (Step 2/4 to mitigate at catalog/output layer), scope honored.

---

## Round 2

**Date:** 2026-05-23
**Reviewer:** `security-auditor` (Opus 4.7, 1M context), model pinned at invocation.
**Target:** Round-2 re-review of the round-1 fold-in. Changes to `comfyless/catalog.py` (lazy import + size cap + explicit UTF-8 decode) and `test_mcp_server.py` (four new characterization tests A–D covering the three fixes).

### Summary

All three round-1 substantive findings (HIGH-1, MEDIUM-1, MEDIUM-2) are addressed in the amended code, not merely acknowledged in docstrings. Reviewer walked each remediation against the actual diff, verified the test that closes each one, and inspected the fold-in for new edge cases (BOM handling, off-by-one in the size check, partial reads on the binary path, subprocess test soundness, per-call cost of the lazy import). No new HIGH or MEDIUM findings emerge from the fold-in. Vision invariant 14 still honored — no edits to `nodes/eric_diffusion_utils.py` or `comfyless/generate.py`.

### Coverage

Reviewed:
- `comfyless/catalog.py:38-118` (full amended module)
- `test_mcp_server.py:1380-1457` (four new tests: bloated, just-under, UTF-8, subprocess)
- `comfyless/__init__.py:1-66` (confirmed shim-only; no torch import path)
- `nodes/eric_diffusion_utils.py:1-51` (re-confirmed `import torch` at module top — the import that the HIGH-1 fix gates)

### Verification per round-1 finding

**HIGH-1 ADDRESSED.** `from nodes.eric_diffusion_utils import infer_model_family` moved from module-top to inside `scan_model_family`. The module is now stdlib-only at import time (`json`, `os`, `typing`). Test D spawns a fresh interpreter with `sys.executable -c "import sys; import comfyless.catalog; print('torch_imported=' + str('torch' in sys.modules))"` and asserts the conjunctive `returncode == 0 and "torch_imported=False" in stdout`. Sound proof: `import comfyless.catalog` traverses only its own `comfyless/__init__.py` shim path (shim-only) plus stdlib — Python does not auto-import torch. The lazy import only fires on first call to `scan_model_family`, after which `sys.modules` caches the result; subsequent calls are O(1) dict lookups — no per-call gotcha for Step-2 fan-out.

**MEDIUM-1 ADDRESSED.** `_MAX_INDEX_BYTES = 1024 * 1024` defined; read is `f.read(_MAX_INDEX_BYTES + 1)` then `if len(data) > _MAX_INDEX_BYTES: return None`. Bound math verified: file of exactly 1 MiB returns `len(data) == 1 MiB` (not `>`, proceeds); 1 MiB + 1 byte returns `len(data) > 1 MiB` (rejects). No off-by-one. Test A writes ~1 MiB + ~45 bytes of JSON envelope + padding and asserts `None`. Test B writes a ~50 KB padded valid JSON and asserts `"qwen-image"`, proving the cap doesn't fire prematurely.

**MEDIUM-2 ADDRESSED.** Text-mode `open(index_path)` replaced with binary `open(index_path, "rb")` plus explicit `data.decode("utf-8")`. `UnicodeDecodeError` is in the caught exception tuple. Host locale is no longer in the contract. Test C writes a JSON document containing multi-byte UTF-8 (`café-é-emoji-🚀`) with `ensure_ascii=False` and confirms `"qwen-image"` — exercises the explicit decode path on non-ASCII bytes.

### New-edge-case sweep (no findings)

- **BOM handling.** Both old and new paths fail-close to `None` via `JSONDecodeError`/`UnicodeDecodeError` catch. Net behavior unchanged.
- **Partial reads at the cap boundary.** Multibyte UTF-8 straddling the cap → `UnicodeDecodeError` → caught → `None`. Fail-closed.
- **`os.path.isfile` on the path.** Still follows symlinks; round-1 INFO-1 correctly defers this to the Step 2 walker per Vision invariant 17. Unchanged.
- **Subprocess test environment dependencies.** `cwd=str(Path(__file__).parent)` puts the repo root on the spawned interpreter's `sys.path[0]`. A theoretical false-fail would require a user `sitecustomize.py` that imports torch globally — environment quirk, not a defect in the test.
- **Per-call cost of the lazy import.** Post-first-call, `sys.modules` dict lookup is sub-microsecond. Negligible vs the existing `os.path.isfile` + `open` syscalls in the body.
- **Diff scope.** Only `comfyless/catalog.py` and `test_mcp_server.py` are modified. Vision invariant 14 honored.

### Round 2 verdict

**CLEAN.** All round-1 findings (HIGH-1, MEDIUM-1, MEDIUM-2) properly addressed in code; characterization tests A–D exercise the fix surfaces; no new HIGH or MEDIUM findings introduced by the fold-in. Round-1 INFO items remain correctly deferred per the stated Step 2 / Step 2-4 disposition. Ready to commit.
