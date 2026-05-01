# Slice 1 Vision — Minimal `generate` MCP tool (cascade-included, default-model-aware)

**Date:** 2026-04-30
**ADR:** [ADR-011](../decisions/ADR-011-comfyless-mcp-server.md) — see the 2026-04-30 Changelog amendment for the cascade-in-slice-1 + `--default-model` decisions reflected here.
**Status:** approved by Grant 2026-04-30; awaiting `/change-slice` to start the Change Contract.
**AI-Disclosure:** Claude (Opus 4.7, 1M context) authored; Grant reviewed and approved.

---

## Slice

Minimal `generate` MCP tool for comfyless. First substantive implementation slice of ADR-011. Lands the full path-validation, audit, and Red Zone scaffolding that all subsequent MCP tools (`extract_params`, `list_models`, `list_loras`, `iterate`, `edit` stub) will inherit.

## Posture

- **Boundary:** integration (LLM-agent → comfyless → diffusers/filesystem).
- **Risk factors:** external exposure (LLM input drives paths and parameters); near security truth (path allowlist + audit + ADR-011 §3 enforcement); broad impact (foundation for five follow-on tools).
- **Risk level:** **L3 (Red Zone).** Project CLAUDE.md mandates Red Zone status from the first commit of the LLM-agent surface.

## Intent

Land a minimal `generate` MCP tool in `comfyless/mcp_server.py` exposing a stdio MCP server with:

- Three spawn-time CLI args:
  - `--output-dir` (required) — write root.
  - `--model-base` (required) — load-allowlist root.
  - `--default-model` (optional) — fallback model when the agent omits the `model` field.
- `--mcp-max-iterations <int>` (per ADR-011 §3c) is declared/parsed at spawn time even though `iterate` lands later — keeps the spawn contract stable from slice 1 onward.
- Hard-coded `allow_hf_download=False` at every `resolve_hf_path` and `_load_pipeline` call site, with a regression test asserting the value is unbypassable from any MCP-reachable code path (including cascade-specific loaders).
- Audit-to-stderr discipline: one line per invocation, success and rejection alike, with `prompt` and `negative_prompt` redacted.
- Schema-validated tool boundary (existing `COMFYLESS_SCHEMA` + cascade schema, applied via existing `_validate_params`).
- Coverage for **all five families on day one**, including Stable Cascade dispatch with `cascade_config.stage_*` and `scaffolding_repo` allowlist enforcement.
- Tool-description text steering the agent toward the right family for non-default prompt shapes; slice 1 ships an initial draft and operator docs name a recommended `--default-model` value.

## Invariants (must always be true)

1. Server startup fails closed if `--output-dir` or `--model-base` is missing, non-existent, or not a directory. No env-var fallback. No default. (ADR-011 §3a)
2. Every `generate` tool call validates `model`, `transformer_path`, `vae_path`, `text_encoder_path`, `text_encoder_2_path`, `loras[].path`, and (when `model` resolves to cascade) `cascade_config.stage_c`, `cascade_config.stage_b`, `cascade_config.stage_a`, `cascade_config.scaffolding_repo` against `--model-base` after `os.path.realpath`; rejects on `_within` failure. (ADR-011 §3b + 2026-04-30 Changelog amendment)
3. Every `generate` tool call validates `output_path` / resolved-savepath against `--output-dir` after `os.path.realpath`; rejects on `_within` failure.
4. `resolve_hf_path` and `_load_pipeline` invocations from the MCP code path **always** receive `allow_download=False` / `allow_hf_download=False`. For cascade dispatch this also covers prior/decoder/vqgan loaders and any `scaffolding_repo` resolution. No `**kwargs` passthrough from MCP request to those calls. (ADR-011 §3 first exclusion)
5. Every tool invocation — success and rejection alike — emits one audit line on stderr (never stdout). Audit line redacts `prompt` and `negative_prompt`. (ADR-011 §3b)
6. `tools/list` advertises exactly **one** tool: `generate`. Other slots are not announced until their slices land.
7. stdout carries only MCP JSON-RPC frames; nothing else.
8. If `model` is absent in the tool input AND `--default-model` is configured at spawn, the server resolves the omitted field to the configured default path. The resolved default path goes through the SAME realpath + `_within(--model-base)` validation at request time as a caller-supplied `model`. No bypass. (2026-04-30 Changelog amendment)
9. If `model` is absent AND `--default-model` is NOT configured, the call returns a structured MCP error naming the missing field. No silent fallback to "first model in `--model-base`" or any other heuristic. (2026-04-30 Changelog amendment)
10. Server startup fails closed if `--default-model` is set but does not realpath-resolve to a real path under `--model-base` (covers nonexistent path, non-directory file, symlink escaping `--model-base`). (2026-04-30 Changelog amendment)

## Failure semantics

- **Fail-closed at startup:** missing/invalid `--output-dir` or `--model-base` → server exits non-zero before serving a single MCP request. Same for `--default-model` when set.
- **Fail-closed at request time:** any path-validation failure, schema-validation failure, or HF cache miss returns a structured MCP error to the agent and writes an audit line to stderr; no partial success, no fallback, no silent path normalization.
- **Default-model defense in depth:** even after startup validation accepts `--default-model`, every request that uses it re-runs realpath + `_within` at request time. Startup validation does not exempt request-time validation.
- **Partial success not possible:** either the PNG and sidecar both land under `--output-dir`, or neither does.
- **Audit-line write failure** does not block the request — but increments a counter for the next slice's security review.

## Out of scope (explicit)

- Other five tools (`edit` stub, `list_models`, `list_loras`, `extract_params`, `iterate`) — each is its own follow-up slice (renumbered slices 2–6 per the 2026-04-30 ADR Changelog amendment).
- HTTP/SSE MCP transport.
- Removing or deprecating the existing `--json` mode.
- Edit pipeline support beyond what the future `edit` stub slice will declare.
- The dep-bump for the `mcp` SDK (its own tiny-change slice — slice 0; prerequisite, not part of this Vision).
- Streaming progress notifications, MCP `resources/list` / `prompts/list` surfaces.
- Server-side prompt classifier / model-selector reasoning layer (deferred to post-LLM-judge per the 2026-04-30 amendment Layer 3).
- Short-name → path resolution for the `model` field (deferred to land alongside `list_models`).

## Negative cases (required)

- **N1–N3:** Server started without/with-nonexistent/with-non-directory `--output-dir` exits non-zero.
- **N4:** Same three for `--model-base`.
- **N5:** `generate` with `model="/etc/anything"` → MCP error, no model load, audit line written.
- **N6:** `generate` with `model="<inside model_base>/../../../etc/passwd"` → MCP error after realpath + `_within`.
- **N7:** `generate` with `model=<symlink inside model_base pointing outside>` → MCP error after symlink resolution.
- **N8:** `generate` with `output_path` outside `--output-dir` → MCP error, no write.
- **N9:** `generate` with `loras[0].path` outside `--model-base` → MCP error, no LoRA load.
- **N10:** `generate` against an HF repo ID with no local cache entry under `allow_download=False` → MCP error naming the missing repo, no network call.
- **N11:** Regression test — monkey-patch `resolve_hf_path` to record `allow_download` arg values; exercise `generate` (including cascade); assert every recorded value is `False`. Same for `_load_pipeline`.
- **N12:** Audit line on a successful `generate` does NOT contain prompt or negative_prompt text.
- **N13:** Audit line goes to stderr, not stdout (capture both; stdout = JSON-RPC frames only).
- **N14:** `generate`'s input schema does NOT accept `max_iterations` (that's an `iterate`-only field).
- **N15:** `generate` with no `model` field, `--default-model` configured → uses default model; success path with audit line.
- **N16:** `generate` with no `model` field, `--default-model` NOT configured → structured MCP error, no model load, audit line written.
- **N17:** Server started with `--default-model=/tmp/path-outside-model-base` → exits non-zero at startup.
- **N18:** Server started with `--default-model=<symlink under model_base pointing outside>` → exits non-zero at startup (same realpath rule as the model field).
- **N19:** `generate` with cascade selected, `cascade_config.stage_c` outside `--model-base` → structured MCP error after realpath + `_within`, no cascade load.
- **N20:** `generate` with cascade selected, `cascade_config.scaffolding_repo` outside `--model-base` → structured MCP error, no scaffolding load.
- **N21:** Cascade dispatch via `generate` honors `allow_hf_download=False` for prior/decoder/vqgan loaders too — extension of N11 to cover cascade-specific call sites.
- **N22:** Cascade dispatch via `generate` writes an audit line that retains `cascade_config.stage_*` paths (paths are not redacted; only `prompt` and `negative_prompt` are dropped per invariant 5).

## Proof hooks

- **Positive:** `python3 test_mcp_server.py` — exercises happy path against a small fixture model.
- **Server startup positive:** `python3 -m comfyless.mcp_server --output-dir /tmp/out --model-base /tmp/models </dev/null` exits 0 (clean shutdown on EOF).
- **Negative cases N1–N22** organized into sections inside `test_mcp_server.py` (no pytest dep — same `python3 test_<name>.py` invocation as the other 7 suites).
- **Existing 7 suites continue to pass — 732/732.** Proves no existing behavior was broken.

## Red Zone ownership

- **Path-allowlist enforcement** (including cascade `stage_*` / `scaffolding_repo`): owned by **Grant** — AI-generated only, not sole author.
- **`--default-model` startup-validation logic** (a fourth call site for the realpath + `_within` pattern; verifying it slots into existing helpers without divergence): owned by **Grant**.
- **`allow_hf_download=False` enforcement** (including cascade-specific call sites): owned by **Grant** — verifies the regression test catches the broader call-site set.
- **Audit log redaction discipline:** owned by **Grant** — signs off on the field list.
- **ADR-011 is the design source of truth** — any divergence reverts to ADR amendment before code lands.

## Pointers

- ADR: [ADR-011](../decisions/ADR-011-comfyless-mcp-server.md) (Status: accepted; 2026-04-30 amendment in Changelog).
- Security review (round 1 + round 2 CLEAN): [review-comfyless-mcp-server-2026-04-28.md](../security/review-comfyless-mcp-server-2026-04-28.md).
- Slice-0 prerequisite: `mcp[stdio]` dep-bump — separate tiny-change slice, not yet started.
- Slice plan: ADR-011 §3d ordered slice plan (renumbered 2026-04-30: cascade collapsed into slice 1; old 3–7 renumber 2–6).
