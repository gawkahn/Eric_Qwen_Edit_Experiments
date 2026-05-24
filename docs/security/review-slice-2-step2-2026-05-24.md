AI-Disclosure: Claude (Opus 4.7, 1M context) authored this security review; Grant reviewed.

# Security Review — Slice 2 Step 2 (ADR-015 catalog data structure + scan walker + manifest parser + build_catalog)

**Date:** 2026-05-24
**Reviewer:** `security-auditor` (Opus 4.7, 1M context), model pinned at invocation.
**Target:** Uncommitted diff implementing slice-2 Step 2 of ADR-015 — MOD `comfyless/catalog.py` (adds `CatalogEntry`, `CatalogDict`, `CatalogBuildError`, `normalize_name`, `_add_entry`, `_scan`, `_parse_manifest_entry`, `_parse_manifest`, `build_catalog`; `_MAX_MANIFEST_BYTES`, `_KINDS`, `_SCAN_DIR_TO_KIND`); MOD `test_mcp_server.py` (Step 2 section covering N1–N15, N24–N26, N28).
**ADR:** ADR-015 (accepted, rounds 1+2 CLEAN). Vision: `docs/vision/slice-2-mcp-catalog.md` (invariants 1–7, 15–17).

---

## Round 1

### Summary

Step 2 is the security keystone of ADR-015 slice 2 — every later slice's reference-resolution safety depends on this code being right. The diff adds (a) the `CatalogEntry` TypedDict, (b) NFC-normalizing `normalize_name`, (c) `_add_entry` with three layered collision rules (exact-name same-realpath alias allowed; exact-name different-realpath rejected; casefold collision across entries rejected), (d) `_scan` walking `--model-base` with `os.walk(followlinks=False)`, file-symlink skipping, `model_index.json` directory-as-model dispatch with `dirnames.clear()`, and the `loras/checkpoints/diffusion_models` parent-basename dispatch, (e) `_parse_manifest_entry` with explicit allowed-keys / required-fields / kind-enum / target_family-only-on-lora validation, HF-vs-path branching via `_is_hf_repo_id`, and a defense-in-depth `_within(--model-base)` after `realpath`, (f) `_parse_manifest` with a 1 MiB cap mirroring Step 1's MEDIUM-1 shape, (g) `build_catalog` orchestrating scan-then-manifest with no partial state on failure. `CatalogBuildError` subclasses `ValueError` so the Step 3 wiring can adapt to `click.BadParameter`.

Threat model is stdio same-uid; the catalog is built but not yet exposed to the agent (`list_*` lands in Step 4). The adversaries are (1) an operator who installed a malicious model package or wrote a poisoned manifest, and (2) implicit: the agent at Step 3+ when this catalog becomes the only thing standing between agent inputs and load operations.

Reviewer walked every trust boundary the build path crosses: untrusted file contents under `--model-base`, the manifest JSON, the manifest entry `target` field (path or repo ID), the HF cache resolver, the `_within` defense, and the operator stderr channel. Step-1 fixes (lazy-import discipline, size cap, explicit UTF-8 decode) all carry forward correctly into the new module-top imports and the new `_parse_manifest` reader. Vision invariants 1–7, 15–17 are implemented as text where the contract lives, not just as docstrings. Findings below are gaps, not show-stoppers — the fail-closed posture is correct, but one MEDIUM-severity NUL-byte gap and several lower-severity concerns are worth marking before Step 3 reaches into this surface.

### Coverage

Reviewed:
- `comfyless/catalog.py:1-534` (entire amended module — module-top imports, Step 1 helper, Step 2 additions)
- `test_mcp_server.py:1460-2043` (full Step 2 test block, N1–N15, N24–N26, N28, plus the carry-forward torch-import subprocess test)
- `comfyless/server.py:52-189` (`_within`, `_check_paths`, NUL-byte handling — to verify the build-time `_within` reuse is semantically correct and to compare NUL-byte coverage)
- `comfyless/mcp_server.py:482-501, 702-736` (NUL-byte defense pattern in request-side handlers — the established project pattern)
- `nodes/eric_diffusion_utils.py:1-155` (`infer_model_family`, `_is_hf_repo_id`, `resolve_hf_path` — to confirm callee contracts and HF cache-miss exception class)
- ADR-015 §1 catalog construction; §4 model-base relationship
- Vision invariants 1–7, 15–17; negative cases N1–N15, N24–N26, N28
- Step-1 review for HIGH-1 / MEDIUM-1 / MEDIUM-2 carry-forward verification

Not reviewed (out of scope per prompt):
- Step 3 `--catalog` click-flag wiring + `_StartupConfig.catalog` (lands Step 3)
- `list_models` / `list_loras` handlers (lands Step 4)
- HTTP-transport-only concerns (per ADR-015 INFO-3)
- `extract_params` reverse-lookup behavior (lands slice 4)
- The agent-facing uniform-error contract (lands slice 3)

### Findings

#### [MEDIUM-1] NUL byte in manifest `target` escapes as uncaught `ValueError` from `os.path.realpath`

Location: `comfyless/catalog.py` (`abs_path_real = os.path.realpath(abs_path)` in `_parse_manifest_entry`).

Risk: A manifest entry with `"target": "/legitimate/path\x00/etc/passwd"` reaches `os.path.realpath`, which raises `ValueError: embedded null byte` (CPython's behavior; not in the listed `OSError`/`json.JSONDecodeError` catch tuple of `_parse_manifest`). The exception propagates out of `_parse_manifest_entry`, past `_parse_manifest` (which only catches `OSError` on the initial `open`), and out of `build_catalog` as a raw `ValueError`, not as `CatalogBuildError`. Two practical consequences:

1. **Operator debugging is degraded.** The `CatalogBuildError` channel is the operator-facing startup channel per ADR-015 §1 — its job is naming the offending entry. A raw `ValueError: embedded null byte` traceback at spawn does not name which manifest entry triggered it.

2. **`CatalogBuildError` is the contract Step 3 will wrap into `click.BadParameter`.** A non-`CatalogBuildError` exception class escaping from `build_catalog` will look like a server crash to the click `main()` handler at Step 3, not like a structured "bad input" report. The Step 3 wiring will need to add a separate `ValueError` catch in addition to `CatalogBuildError`, which is exactly the kind of contract drift this ADR was designed to avoid.

Compare with the established project pattern: `comfyless/server.py:138-149` and `comfyless/mcp_server.py:482-501, 702-736` BOTH pre-check `\x00` in path-shaped fields explicitly before any `realpath` call, because the NUL-byte case is a known footgun. The catalog manifest parser is the same surface (operator-supplied path-shaped value reaching `realpath`) and should follow the same pattern.

Remediation: in `_parse_manifest_entry`, after the `target` type-check and before the `_is_hf_repo_id` branch, add `if "\x00" in target: raise CatalogBuildError(f"manifest entry {name!r}: 'target' contains a null byte")`. Smallest targeted change.

#### [INFO-1] Symlink discipline depends on `os.walk(followlinks=False)` semantics at the top-level walk-root child

Location: `comfyless/catalog.py` `_scan` directory dispatch.

Confirmation: `os.walk(model_base_real, followlinks=False)` does not descend into symlinked subdirectories that are direct children of the walk root, AND `model_base_real` itself is `os.path.realpath`-resolved at `build_catalog` so a symlinked `--model-base` is also defended. The realpath-then-basename-of-dirpath pattern at the `model_index.json` dispatch is correct. No code change for Step 2. If a future refactor replaces `os.walk` with a custom `os.scandir`-based walker, retest these invariants explicitly.

#### [INFO-2] Catalog names from filesystem basenames and manifest keys flow into operator stderr (escaped by `!r`) and will flow into `list_*` MCP JSON in Step 4 (not yet sanitized)

Location: `comfyless/catalog.py` `CatalogBuildError` messages; future Step-4 `list_*` formatter (not in this diff).

Risk: Operator-facing `CatalogBuildError` messages use `!r` formatting which escapes control characters and ANSI sequences — defense is in place at Step 2. Slice-4 `list_models` / `list_loras` JSON serialization needs its own decision (sanitize? reject at build time? document?) before agent-facing surface lands. Threat model fit at Step 2: same-uid operator with malicious directory name can already write `~/.bashrc`; no additional capability. Forward-pointer to slice 4: `list_*` formatter must address. No Step 2 code change.

Remediation: Capture as TECH_DEBT entry for the slice-4 Vision: list_* formatter must decide between sanitizing-on-output, rejecting-at-build-time, or documenting-as-operator-responsibility for catalog names containing control characters / RTL overrides / etc.

#### [INFO-3] First call to `_parse_manifest_entry` triggers torch import via the `nodes.eric_diffusion_utils` module-top `import torch`

Location: `comfyless/catalog.py` `_parse_manifest_entry`'s lazy `from nodes.eric_diffusion_utils import _is_hf_repo_id, resolve_hf_path`.

Risk: Step 1's HIGH-1 contract ("`comfyless.catalog` is stdlib-only at import time") is preserved at module-top. But the FIRST call to `_parse_manifest_entry` imports `nodes.eric_diffusion_utils`, which triggers its module-top `import torch`. This is acceptable under the current architecture (catalog built once at MCP spawn, torch will be needed shortly after by `_load_pipeline`). Not a defect; a property to be aware of for future tools that want to validate a manifest without importing torch (e.g. a CLI `comfyless catalog validate`).

Remediation: none. If a future tool needs torch-free manifest validation, inline `_is_hf_repo_id` into `comfyless/catalog.py` and lazy-import `resolve_hf_path` only inside the HF branch.

#### [INFO-4] `_within` re-realpath's both arguments; cosmetic double-stat per check

Location: `comfyless/catalog.py` `_parse_manifest_entry`; `comfyless/server.py:158-162`.

Confirmation: Semantically correct (realpath is idempotent), microsecond-level cost at realistic manifest sizes. Recording as a reference for the ADR-015 §4 end-state amendment's migration plan.

#### [INFO-5] Build-time `_within` check does NOT verify the manifest target's realpath EXISTS as a file/directory

Location: `comfyless/catalog.py` `_parse_manifest_entry` (no `os.path.exists` after `realpath`).

Risk: `_parse_manifest_entry` realpath-resolves the manifest target and `_within`-checks it, but does not assert the resolved path exists. A manifest can declare `{"my_model": {"target": "/under/model-base/but/does/not/exist", "kind": "model"}}` and `build_catalog` will succeed. The error surfaces only at request time when `generate` (Step 3 onward) attempts the load. This is the **correct posture** per ADR-015 §4: catalog build is a containment check, not a usability check. The Vision invariants (2) explicitly do not require an existence check. The load attempt at request time is the authority for "does this actually load." Adding an existence check at build time would also have a TOCTOU window vs request time. Under the future slice-3 uniform-error contract, the load-time failure already returns `"reference not available"`, the right posture.

Remediation: none. The current behavior is correct under the Vision. Documents the deliberate choice for the record.

### Verification per Step-1 carry-forward

**Step-1 HIGH-1 (lazy-import discipline) preserved.** Module-top imports of `comfyless/catalog.py` are `json, os, unicodedata, typing` only. All `nodes.eric_diffusion_utils` and `comfyless.server` imports occur inside function bodies. Test verifies `build_catalog(scan-only)` does not pull torch — sound proof that Step 1's contract carries forward into the larger module.

**Step-1 MEDIUM-1 (size cap) preserved and extended.** `_MAX_INDEX_BYTES` unchanged. `_MAX_MANIFEST_BYTES` follows the same `f.read(N+1); if len(data) > N: reject` pattern. Off-by-one verified by reading the test (manifest with `_MAX_MANIFEST_BYTES + 1` bytes is rejected; boundary tested).

**Step-1 MEDIUM-2 (explicit UTF-8 decode) preserved and extended.** `scan_model_family` unchanged. `_parse_manifest` uses `data.decode("utf-8")` explicitly, with `UnicodeDecodeError` in the catch tuple. Locale-independent.

**Vision invariant 14 honored.** No edits to `nodes/eric_diffusion_utils.py`, `comfyless/server.py`, `comfyless/generate.py`, or `comfyless/mcp_server.py` in this diff. Step 2 is isolated to `comfyless/catalog.py` + `test_mcp_server.py`.

### Round 1 verdict

**CHANGES_REQUIRED — minor.** No CRITICAL. No HIGH.

- MEDIUM × 1 — NUL byte in manifest `target` escapes as uncaught `ValueError`, degrading the operator-facing error channel and breaking the `CatalogBuildError`-only contract that Step 3 will wrap. Pre-check for `\x00` in `target` before calling `os.path.realpath`, matching the project's established pattern in `comfyless/server.py` and `comfyless/mcp_server.py`.
- INFO × 5 — symlink discipline depends on specific `os.walk` semantics (correct; document for future refactors); catalog names not sanitized before flowing to `list_*` JSON (slice-4 forward-pointer; TECH_DEBT capture); first manifest entry triggers torch import (architecture property); `_within` re-realpath's both args (cosmetic); build-time `_within` does not check existence (correct per Vision; document the choice).

Fold-in for the MEDIUM finding is a 2-line addition in `_parse_manifest_entry` and one negative test in `test_mcp_server.py` asserting `CatalogBuildError` (not bare `ValueError`) is raised on a manifest target containing `\x00`. Round 2 after fold-in is mechanical; not gated.

---

## Round 2

**Date:** 2026-05-24
**Reviewer:** `security-auditor` (Opus 4.7, 1M context), model pinned at invocation.
**Target:** Round-2 re-review of the round-1 MEDIUM-1 fold-in. Changes to `comfyless/catalog.py` (`_parse_manifest_entry` NUL-byte pre-check before `os.path.realpath`) and `test_mcp_server.py` (N4h negative test asserting `CatalogBuildError`, not bare `ValueError`).

### Summary

Verified the NUL-byte pre-check is positioned before every NUL-sensitive sink on the `target` flow (`_is_hf_repo_id`, `resolve_hf_path`, `os.path.realpath`), the error message names entry and issue, and the new test catches regression to bare `ValueError` via the `_assert_raises` exception-class semantics. Swept the rest of the manifest entry shape and `_parse_manifest`/`build_catalog` for parallel NUL surfaces; documented one INFO covering the operator-supplied `--catalog` path + `--model-base` flag values that share MEDIUM-1's shape but are out of scope for this fold-in (naturally handled by `click.Path` at the Step-3 wiring boundary).

### Verification of MEDIUM-1 fold-in

**Placement (in `_parse_manifest_entry`, execution order on `target`):**

1. Type / empty-string check (`isinstance(target, str)` + truthiness).
2. **NUL pre-check: `if "\x00" in target: raise CatalogBuildError(...)`** — newly added.
3. `_is_hf_repo_id(target)` — pure-Python string ops (`startswith`, `len`, indexing, `split`); NUL-safe even without the pre-check, but the pre-check rules out the value before reaching here.
4. `resolve_hf_path(target, allow_download=False)` — reached only if `_is_hf_repo_id` returned True; the pre-check has already excluded NUL strings.
5. `os.path.realpath(abs_path)` — the original sink MEDIUM-1 identified. Pre-check defends both this and any future NUL-sensitive call.

The pre-check is correctly positioned BEFORE every NUL-sensitive sink AND before the HF-vs-path branch — a NUL-containing string can't be misclassified into either branch in a way that bypasses the check.

**Error message:** `f"manifest entry {name!r}: 'target' contains a null byte"`. Names the entry (operator-debuggable) and the precise issue (`null byte`). `!r` formatting on `name` escapes control chars (Round-1 INFO-2 stance). Operator-stderr-grade.

**Test N4h catches a regression to bare `ValueError`:**

`_assert_raises` uses `except exc_type as e:` where `exc_type = CatalogBuildError`. `CatalogBuildError extends ValueError`. The class hierarchy means `except CatalogBuildError` matches `CatalogBuildError` and its subclasses ONLY — it does NOT match a bare `ValueError`. So if a regression removes the pre-check and `os.path.realpath` raises `ValueError: embedded null byte`:
- The `except CatalogBuildError` clause does not fire.
- The `except BaseException` fallback fires, emits `check(..., False, detail="got ValueError: ...")`, and returns before the `message_contains` check.
- Both N4h assertions fail visibly in the suite.

Regression detection confirmed.

### Sweep for new gaps introduced by the fold-in

Other manifest-entry strings reaching NUL-sensitive operations:

- `name` (top-level JSON key): flows into `normalize_name` (`unicodedata.normalize("NFC", s)` — NUL-safe), dict key insertion (NUL-safe), and `str.casefold()` (NUL-safe). Does NOT reach `os.path.realpath`, `os.path.isfile`, `open`, or any other NUL-sensitive sink. A NUL-containing entry name would produce a usable-but-weird catalog key; under stdio same-uid threat model and `!r` escaping in `CatalogBuildError` messages, this is operator self-harm and was explicitly accepted in Round-1 INFO-2 as a slice-4 forward-pointer.
- `kind`: gated by `kind not in _KINDS` membership check; NUL-containing kind would simply fail enum validation with `CatalogBuildError`. Not reachable to any path sink.
- `model_family`, `target_family`: stored verbatim on the `CatalogEntry`. Not used in any path operation in slice 2. Out of scope for NUL.

Parallel pre-existing surfaces (not introduced by fold-in, but worth recording):

#### [INFO-6] Operator-supplied `--catalog` and `--model-base` paths have pre-existing parallel NUL surfaces

Location: `comfyless/catalog.py` `_parse_manifest` `os.path.isfile(catalog_path)`; `build_catalog` `os.path.realpath(model_base)`.

Risk: A NUL-containing value for either CLI flag escapes `build_catalog` as a bare `ValueError`, not a `CatalogBuildError` — the same shape as MEDIUM-1 was for the `target` field. Pre-dates the slice-2 step-2 fold-in (entered when the CLI flags were introduced in their respective slices). Under stdio same-uid the operator can already cause arbitrary harm; this is a contract-cleanliness concern, not an exploit. The Step-3 click wiring (next step) will naturally cover this — click's `Path` and `File` types pre-validate at the argparse layer before `build_catalog` is even called.

Remediation: at Step 3 wiring, declare `--catalog` as `click.Path(file_okay=True, dir_okay=False, exists=False)` (or equivalent) so click rejects NUL-bytes before `build_catalog` runs. No Step-2 code change required.

### Round 2 verdict

**CLEAN.** MEDIUM-1 is addressed in code (NUL pre-check before `_is_hf_repo_id`, `resolve_hf_path`, and `os.path.realpath`). The error message names entry and issue. The N4h test catches a regression to bare `ValueError` via the `_assert_raises` exception-class semantics. One new INFO-6 recorded covering the pre-existing parallel surfaces on `--catalog` / `--model-base` CLI flags — out of scope for this fold-in, naturally handled by click's `Path` type at the Step-3 wiring boundary.

No new HIGH or MEDIUM. Slice-2 step-2 cleared for commit.
