# Security Review — Slice 4 Step 4: `extract_params` core (non-cascade)

**AI-Disclosure:** Claude (Opus 4.8, 1M context) authored this security review; Grant to review. L3 Red Zone per slice-4 Vision §Risk (first MCP tool reading a caller-supplied filesystem path).
**Date:** 2026-07-07
**Reviewers:** `security-auditor` (Opus) + `code-reviewer` (Opus), both invoked per the project CLAUDE.md review bar (caller-supplied file-read on the LLM-agent surface).
**Change under review:** `_handle_extract_params` + `_sidecar_name_index` / `_resolve_sidecar_ref` / `_render_extracted_params` + constants + registration/dispatch/audit in `comfyless/mcp_server.py`; tests in `test_mcp_server.py` ("Slice 4 step 2"). Governing spec: `docs/vision/slice-4-mcp-extract-params.md` (invariants 1-10, 13-15; enrichment 11-12 is a later step).
**Verdict:** **CLEAN at CRITICAL/HIGH/MEDIUM** after fold-ins. The two MEDIUMs the code-reviewer raised were both cascade-specific and are resolved by **descoping cascade to a follow-on step (4d)**; the LoRA-weight LOW is folded; residual INFOs accepted at the threat model.

---

## Threat model

Single-user desktop tool; the comfyless MCP server is spawned by/for a same-uid LLM agent over stdio. Sidecars are the user's own files under `--output-dir`. The boundary being defended: (a) no read outside `--output-dir`; (b) no server-side absolute path or directory component crossing back to the agent; (c) no enumeration/type oracle; (d) read-only / no model load / no network / no write.

## Security-auditor verdict (CLEAN at C/H/M)

Verified sound:
- **Path gates.** `resolved = os.path.realpath(path)` runs FIRST (NUL rejected before it), then (1) `.json`-on-resolved, (2) `_within(resolved, output_dir)`. Realpath-first genuinely defeats `legit.json → evil.png` (fails check-1) and `legit.json → /outside/x.json` (fails `_within`). `open()` targets the SAME `resolved` string that was gated. `.JSON` / trailing-dot / trailing-slash / `..` all fail closed on the case-sensitive filesystem.
- **No enumeration / type oracle.** Both gate failures raise the single constant `_SIDECAR_PATH_REJECT`, byte-identical, never echoing `resolved`. The gate is pure string logic (no `stat`), so files outside `--output-dir` are not probeable. Past-the-gate failures collapse to one sanitized `_SIDECAR_UNREADABLE` (no traceback, no abs path).
- **No-abs_path egress (non-cascade).** `_validate_params` is an allowlist normalizer; every path-typed schema field (`model`, `transformer_path`, `vae_path`, `text_encoder*`, `loras[].path`) is resolved-to-name or dropped; `output_path`/`savepath`/`lora_warnings` (non-schema) popped defensively; a catalog miss surfaces only `os.path.basename`, never the directory; the catalog's own `abs_path` is used only for internal index keys and never emitted.
- **Reverse-resolution safety.** `_resolve_sidecar_ref` performs NO filesystem operation on sidecar content (basename + dict lookup + family compare). Ambiguity unresolved by family → miss, never a wrong-name guess. A hit is always a live, replay-valid catalog name (its replay loads the catalog's file under `--model-base`, not an attacker directory).
- **Load-plane independence.** No pipeline load, no network, no write, no `catalog_db` reference; resolution runs against `cfg.catalog` only.
- **Audit hygiene.** The `extract_params` audit payload is `{"path": path[:256]}`; the returned params blob and any `prompt`/`negative_prompt` never reach the audit line.

Auditor findings: one **LOW** (cascade_config denylist re-injection) + two **INFO** (same-uid realpath→open TOCTOU; unbounded `json.load` size). See resolutions below.

## Code-reviewer verdict (CHANGES REQUESTED → resolved)

Confirmed the gate logic, top-level sanitization, reverse-resolution safety, and error/audit hygiene are sound. Two MEDIUMs, both cascade:

- **MEDIUM-1 — cascade_config abs-path egress (denylist over caller bytes).** The cascade branch copied `cascade_config` verbatim and only transformed `stage_*` / dropped `scaffolding_repo`, so a rogue non-stage path key would pass through — a divergence from the Grant-owned invariant 8 ("no absolute path in ANY field").
- **MEDIUM-2 — cascade branch matched no real sidecar; N16 gave false confidence.** Real CLI cascade sidecars are FLAT (`cascade.py:930-950` spreads the config at top level; MCP cascade writes no JSON sidecar). The nested-`cascade_config` branch was dead code against real sidecars; the synthetic N16 fixture masked it.

Plus **LOW** (unvalidated `loras[].weight` could echo a string) and two **INFO** (N-static scope; trailing-slash basename edge).

## Resolutions (folded before commit)

- **Both cascade MEDIUMs → resolved by DESCOPING cascade to step 4d.** The nested-`cascade_config` handling and its synthetic test were removed. A real flat cascade sidecar now has all its cascade/path keys dropped by `_validate_params` — **provably no abs-path egress** (new N16 asserts this against the real flat shape: top-level `stage_c`/`stage_b`/`config_source`/`output_path` → none survive, non-path params do). Flat `stage_*`→names resolution becomes step 4d, mirroring the slice-3 / slice-3b split. This removes the denylist entirely (MEDIUM-1) and replaces the false-confidence test with a real-shape leak-safety test (MEDIUM-2).
- **LoRA-weight LOW → folded.** `_render_extracted_params` coerces each `weight` to number-or-`None`; a non-numeric (e.g. path-shaped) weight is dropped to `None`. New test asserts a `"weight": "/abs/evil/path"` sidecar yields `{"name": ..., "weight": null}` with no `/abs/` in the response.
- **N-static INFO → folded.** The static-source check now covers `_handle_extract_params` and `_render_extracted_params` (call-form `name(`) and a line-based module `import argparse` check (robust against the docstring's "does NOT import argparse" mention).
- **Auditor LOW (cascade denylist) → moot** (cascade handling removed).
- **INFO accepted at threat model:** same-uid realpath→open TOCTOU (agent could read the target directly; a lower-privilege model would need `O_NOFOLLOW`+`fstat`); unbounded `json.load` (self-DoS on the agent's own file); trailing-slash `model` → empty basename (negligible for real sidecars). None cross a privilege boundary here.

## Proof

`test_mcp_server.py` "Slice 4 step 2" section: N1-N16 (incl. the real-flat-cascade leak-safety N16), N22-N24, family-disambiguation, the LoRA-weight coercion, and the widened static-source check. MCP suite 620→654; full 16-suite regression 0 failures.

## Verdict

**CLEAN at CRITICAL/HIGH/MEDIUM.** No exploitable issue and no broken invariant under the stated same-uid threat model after the cascade descope + weight-coercion fold-ins. Cascade sidecar *stage replay* is a documented follow-on (step 4d); the core non-cascade `extract_params` ships with the full no-abs_path-egress guarantee intact.
