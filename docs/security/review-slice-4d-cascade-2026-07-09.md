# Security Review — extract_params step 4d (flat Stable Cascade resolution)

**AI-Disclosure:** Reviewed by `security-auditor` (Opus 4.8) and `code-reviewer`
(Opus 4.8); record authored by Claude (Opus 4.8, 1M context); Grant reviewed.
**Date:** 2026-07-09
**Slice:** ADR-011 slice 4, step 4d (see `docs/vision/slice-4-mcp-extract-params.md` §"Step 4d").
**Surface:** `comfyless/mcp_server.py` — `_is_cascade_sidecar`,
`_render_extracted_cascade_params`, and the cascade branch in
`_handle_extract_params`. Red Zone L3 (LLM-agent-facing, caller-supplied file read).

## Scope reviewed

The new step-4d branch that renders a FLAT Stable Cascade sidecar (top-level
`stage_c`/`stage_b`/`stage_a`/`scaffolding_repo`/`config_source`/`output_path`
per `cascade.py` `_KNOWN_KEYS` :75-89 and dispatch :930-950) into an inline,
allowlisted `cascade_config` whose stage references are reduced to catalog
NAMES. Both reviewers traced every `_KNOWN_KEYS` field plus arbitrary injected
keys through the renderer.

## Threat model

A same-uid LLM agent (stdio) supplies a `.json` sidecar path under
`--output-dir`; the server reads the attacker-controllable JSON content and
returns parameters inline in the MCP response. Load-bearing invariant: **no
absolute path or directory may cross the boundary** — not in any params field,
not in `cascade_config`, not in a notice, not in enrichment.

## Findings

### [MEDIUM] dtype fields echoed arbitrary short strings, incl. `/`-paths — FIXED

`_CASCADE_DTYPE_FIELDS` (`prior_dtype`/`decoder_dtype`/`vae_dtype`) were
key-allowlisted but their VALUES were accepted as any string capped to 32 chars
(`cc_out[k] = v[:32]`), so a crafted `{"prior_dtype": "/mnt/secret/x"}` echoed an
absolute directory string back inside `cascade_config` — the one spot a
`/`-bearing value survived, inconsistent with the number-or-None coercion on the
numeric fields and a deviation from invariant 20. Mitigating: same-uid, echoes
the caller's own bytes (no server secret / real resolved path disclosed) and the
value is inert on replay (`_resolve_torch_dtype` rejects it). Both reviewers
raised it independently and recommended the same fix.

**Resolution (this commit):** dtype values are now **value-allowlisted** to the
exact set `cascade._resolve_torch_dtype` accepts
(`bf16`/`bfloat16`/`fp16`/`float16`/`half`/`fp32`/`float32`/`float`); any other
value is dropped, mirroring the numeric-field discipline. Negative test **N32b**
added (`test_mcp_server.py`): a `/`-bearing dtype is dropped with no `/mnt/`
egress, an unknown non-path dtype is also dropped, and a legit dtype in the same
sidecar still passes. Vision invariant 19 tightened to specify the value
allowlist. `comfyless/mcp_server.py` `_CASCADE_DTYPE_VALUES`.

### [INFO] `model_family` / `prompt` / `negative_prompt` pass through verbatim — DEFERRED (debt)

These free-string fields are re-emitted verbatim (family with a truthy guard;
prompts with `isinstance str`), so an abs-path-shaped value placed there echoes
back. Caller's own bytes (no real-path disclosure); these are free text / a
family label, not path-typed; and the behavior is **identical to the already-
reviewed non-cascade `_render_extracted_params` (:382-383)** — not introduced by
4d. Fixing only the cascade path would create asymmetry; the correct fix is a
shared bound across both renderers. Logged in `TECH_DEBT.md` (2026-07-09).

### [INFO] `extract_params` reads the sidecar via unbounded `json.load` — DEFERRED (debt)

No size ceiling on the sidecar parse; a same-uid actor dropping a very large
`.json` under `--output-dir` could cause a transient memory spike. Property of
the whole slice-4 read path, not the 4d branch; requires write access to the
output dir. Logged in `TECH_DEBT.md` (2026-07-09).

## Verified correct (no finding)

- Every `_KNOWN_KEYS` field: stages → catalog name / strict `owner/repo` HF
  passthrough / directory-stripped basename (never a dir); `scaffolding_repo`,
  `config_source`, `output_path`, and all runtime keys → never read → dropped;
  numerics → number-or-None; unknown injected keys (`"__x__"`) → never read.
  No field can egress a real abs path or directory.
- `cascade_config` is a strict positive allowlist; `"prior_steps":"/abs"` → None.
- `_is_cascade_sidecar` — both OR branches route to the leak-safe cascade
  renderer; an undetected cascade sidecar falls to `_validate_params` +
  `_render_extracted_params`, which drops all cascade keys — also leak-safe.
- Errors: single `_SIDECAR_PATH_REJECT` (realpath-first, path never echoed),
  single `_SIDECAR_UNREADABLE` (`from None`); no new type/enumeration oracle
  (all names already enumerable via `list_*`).
- Cascade branch never calls `_validate_params` / `cascade.validate_config`
  (read-only reporter) — enforced + tested (N17 call-form `getsource` check).
- Enrichment reuse: read-only, structurally fail-open, allowlisted to
  `classification`/`model_family`/`description` (trigger_words re-validated),
  never `abs_path`/`root`/`relative_path`; failure prints only the exception
  type name. Audit line echoes only `path[:256]`, never params/enrichment.
- `scaffolding_repo` dropped regardless of HF-repo-id vs crafted-abs-path form.

## Verdict

No CRITICAL/HIGH. The single MEDIUM (dtype egress) is fixed in this commit with
a negative test; two INFO items are pre-existing / whole-slice and logged as
tech debt. Step 4d is cleared for commit.
