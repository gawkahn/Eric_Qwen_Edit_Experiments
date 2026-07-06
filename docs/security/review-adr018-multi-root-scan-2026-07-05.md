# Security review — ADR-018 multi-root catalog scan implementation

**Date:** 2026-07-05
**Reviewer:** `security-auditor` subagent (Opus, `model: "opus"` at invocation)
**Scope:** ADR-018 implementation — `comfyless/catalog.py` (kind-typed recursive scan, byte-identical de-dup carve-out, resolve_reference union), `comfyless/server.py` (`_check_paths` union — project-mandated review surface, `run_server` extra roots), `comfyless/mcp_server.py` (`--lora-path`/`--transformer-path` spawn args, `_StartupConfig.all_roots`), `start-mcpo.sh`.
**Verdict:** **CLEAN** (0 CRITICAL / 0 HIGH / 0 MEDIUM / 1 LOW / 2 INFO)
**Companion:** `code-reviewer` (Opus) ran in parallel on the same diff — verdict APPROVED, 1 LOW finding (same substance as F-2 below).

**AI-Disclosure:** Claude (Opus, security-auditor subagent) authored the review; Claude (Fable 5) drove the slice; Grant reviewed.

## Fold status (all resolved before commit)

| Finding | Severity | Resolution |
|---|---|---|
| F-1 — `run_server` extra-root loop lacks NUL pre-check | LOW | **Fixed** — explicit `"\x00" in p` guard before `realpath`, `comfyless/server.py` (mirrors MCP startup path). |
| F-2 — byte-identical cross-kind stem collision shadows the other kind (scan-order-dependent kind) | INFO | **Fixed (strong option)** — same-stem cross-kind collisions now fail closed via `CatalogBuildError` *before* the `_same_file_content` carve-out, regardless of path/content. Same substance as code-reviewer finding 1; ADR-018 §2 amended; negative test added. |
| F-3 — scan-order nondeterminism confined to byte-identical case | INFO | **Confirmed, no change** — no-wrong-weight invariant holds under `os.walk` order nondeterminism. |

Post-fold regression: full 15-suite run green (1971 tests, 0 failures), including 40 new ADR-018 tests.

---

## Reviewer output (verbatim)

The trust boundary is the MCP tool surface: an LLM agent (untrusted) calls `generate` / cascade / `list_*` over stdio. The security invariant ADR-015 established and ADR-018 must preserve is that the agent supplies only *catalog names*, never filesystem paths — a path-shaped value is basename-stripped, looked up in a spawn-time catalog, and every resolved `abs_path` is re-checked `_within` an operator-chosen allowlist before load. ADR-018 widens that allowlist from `{model_base}` to `{model_base} ∪ lora_paths ∪ transformer_paths` and adds a recursive, kind-typed scanner plus a byte-identical de-dup carve-out. The ADR's load-bearing claim is "trust class unchanged — this widens WHICH operator-curated trees, not WHO chooses them." I verified that claim by tracing every root into the union and every agent value into the resolver; checked the `_within` prefix comparison for the `/a/b` vs `/a/bb` class of bug; audited the new scanner's symlink/TOCTOU posture against the existing `_scan`; stress-tested the `_same_file_content` error paths for false-positive aliasing; and checked the fail-closed posture of cross-kind overlap, root validation, and startup.

The union-plumbing is sound: `all_roots`/`extra_roots` are derived exclusively from spawn-time click options (`--lora-path`/`--transformer-path`, both `multiple=True`, `resolve_path=False`, realpath'd and `isdir`-validated at spawn), `model_base` is always the first element so the union can never be empty, and no request-time field is ever appended to the roots tuple. `_within` correctly appends `os.sep` before the prefix compare, so `/a/bb` cannot satisfy containment under `/a/b`. The new `_scan_kind_root` matches `_scan`'s symlink discipline (`followlinks=False` + `os.path.islink` file skip + `realpath` mint), and because `os.walk` never yields a dirpath reached through a symlinked component, every minted `abs_path` stays within its realpath'd root and passes the later `_within` net consistently. The byte-identical carve-out fails closed on every error path I traced. Findings below are all LOW/INFO; none block merge.

**F-1 [LOW] `run_server` extra-root loop lacks the NUL pre-check the MCP path has**
Location: `comfyless/server.py:754-761`
Risk: In `run_server`, each `--lora-path`/`--transformer-path` is fed straight into `os.path.realpath(p)` with no `"\x00" in p` guard, unlike the MCP path which pre-checks. A NUL in a root arg raises bare `ValueError: embedded null byte` instead of a clean error. This is spawn-time, operator-supplied, and fails closed (crash before `listen()`), so there is no exploit — it is a consistency gap against the established project pattern. Also note the daemon does not currently expose these flags via `generate.py` (`run_server` gets `()` defaults per ADR §4 deferral), so the loop is latent.
Remediation: add `if "\x00" in p:` guard at the top of the loop body, mirroring the model_base/existing pattern.

**F-2 [INFO] Byte-identical duplicate across a lora root and a transformer root shadows the other kind's name**
Location: `comfyless/catalog.py` (the `_same_file_content` carve-out preceded no cross-kind guard on the different-`abs_path` branch)
Risk: The cross-kind fail-closed guard only fired when the *same* `abs_path` was minted under two kinds. If `--lora-path` and `--transformer-path` each hold a byte-identical file with the same stem at *different* paths, the scan order (lora roots before transformer roots) means the name is retained as `kind:"lora"` and the transformer copy is aliased away. `list_transformers` then omits the name, and an agent referencing it as a transformer gets `KindMismatch` → the uniform "reference not available" error. No wrong bytes are ever served (content is byte-identical and `expected_kind` is enforced at `resolve_reference`), so this is an availability/discoverability quirk under operator-curated trees, not a containment or wrong-weight defect.
Remediation: none required for security; optionally treat differing-kind byte-identical collisions as a fail-closed `CatalogBuildError` for operator clarity. *(Adopted — see fold table.)*

**F-3 [INFO] Scan-order nondeterminism is confined to the harmless (byte-identical) case — confirmed, not a defect**
Risk: `os.walk` directory-entry order is OS-dependent, so "first-scanned wins" is nondeterministic. Verified this can only change *which path* backs a name when the two files are byte-identical (`_same_file_content` gate); any genuine content difference, or a directory-vs-file collision, fails closed via `CatalogBuildError`. Since aliased paths carry identical bytes, the agent is served identical content regardless of order.
Remediation: none.

**Error-path verification (no finding):** `_file_sha256` returns `None` (not a partial digest) on any `OSError`, and `_same_file_content` returns `ha is not None and ha == hb`, so an unreadable or mid-read-failing duplicate yields `False` → the collision falls through to `CatalogBuildError` (fail closed, must-not-alias satisfied). Size-gate is `try/except OSError → False`. No path produces a false-positive alias. `resolve_reference` union check and `_check_paths` union both always include `model_base` and cannot be handed agent data. `--default-model` containment remains deliberately `model_base`-only, correctly *not* widened to the union.

**Verdict: CLEAN.** The ADR-018 claim "trust class unchanged — this widens WHICH operator-curated trees can be loaded from, not WHO chooses them" holds in the implementation: every root originates from a spawn-time operator CLI arg, the union is derived once in `_StartupConfig.all_roots` / `run_server.extra_roots` and never touched by request data, `model_base` guarantees a non-empty union, `_within` has no prefix-truncation bug, the new recursive scanner preserves `_scan`'s symlink/realpath discipline, and all four fail-closed conditions (cross-kind overlap, byte-identical de-dup error paths, missing-root validation, startup) deny on error. No CRITICAL/HIGH/MEDIUM findings.
