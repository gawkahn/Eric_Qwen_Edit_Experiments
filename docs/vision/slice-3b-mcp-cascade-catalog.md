# Slice 3b Vision — Migrate the MCP cascade handler to catalog-name reference resolution

**Date:** 2026-06-24
**ADR:** [ADR-015](../decisions/ADR-015-mcp-catalog-reference-resolution.md) (Status: accepted; rounds 1+2 CLEAN). Slice-3b row of the [ADR-015 §5](../decisions/ADR-015-mcp-catalog-reference-resolution.md) revised plan (deferred from slice 3 per OQ-C). Parent: [ADR-011](../decisions/ADR-011-comfyless-mcp-server.md). Builds on the slice-3 non-cascade migration ([slice-3-mcp-generate-catalog.md](slice-3-mcp-generate-catalog.md), IMPLEMENTED 2026-06-02) which it mirrors.
**Status:** IMPLEMENTED 2026-06-24 (commits `7a92e2f` step 1 + `48a0833` step 2 on `main`; both Opus reviews CLEAN, code-reviewer's one PathMoved-coverage finding folded). Proof: `test_mcp_server.py` 460→483; full 9-suite 1310→1333, 0 failures. Reviews: `docs/security/review-slice-3b-cascade-catalog-2026-06-24.md`; ADR-015 Changelog 2026-06-24 entry. APPROVED by Grant 2026-06-24 (OQ-1 resolved: extend `resolve_reference` `expected_kind` to accept a tuple of kinds — cascade stages resolve against `{model, transformer}`; OQ-2 resolved: `scaffolding_repo` dropped from the agent surface, server uses the cascade default, HF-cache-miss on the default surfaces as a load-time error not a reference oracle — acceptable).
**AI-Disclosure:** Claude (Opus 4.8, 1M context) authored; Grant reviewed and approved.

---

## Posture

> **Posture:** Boundary: `entrypoint` (the MCP `generate` tool's cascade dispatch path, `_handle_generate_cascade`). Risk factors: **external exposure** (LLM agent drives the cascade reference inputs and reads the response); **near security truth** (this is the ADR-015 catalog reference contract — the security keystone — and this slice extends the load-bearing uniform-error contract to the cascade path, closing the cascade-side enumeration oracle the slice-1 handler still has). **Risk level: L3 (Red Zone).**

## Slice

Makes the slice-2 catalog **load-bearing for the cascade path** of `generate`, completing the `generate`-tool migration that slice 3 began for non-cascade families. Until now `_handle_generate_cascade` has consumed the slice-1 raw-path contract: it accepts `cascade_config.stage_c` / `stage_b` / `stage_a` / `scaffolding_repo` as caller-supplied absolute paths or HF repo IDs, validates them with `_within(--model-base)`, and emits distinct `HFCacheMiss` / `PathAllowlist` errors to the agent. Slice 3b rewires it to mirror `_handle_generate`:

1. **Input** — `stage_c` / `stage_b` / `stage_a` arrive as opaque catalog names; a path-shaped value has its directory discarded and its basename resolved through the catalog (ADR-015 §2 basename-strip). Each resolves via `resolve_reference` against kind ∈ **{`model`, `transformer`}** — a cascade stage is a `transformer` when single-file (the common case; the live catalog confirms `stable_cascade_stage_{b,c}` are `transformer`) and a `model` when a diffusers tree.
2. **Resolution failure** of any stage reference, from any cause, returns the byte-identical `"reference not available"`; the fine cause (`UnknownName` / `KindMismatch` / `MalformedReference` / `PathMoved` / `WithinFailure`) lands only on the stderr audit line. **This closes the cascade-side reference oracle** — the slice-1 cascade handler's distinct `HFCacheMiss` / `PathAllowlist` agent-facing errors are an enumeration signal that this fold removes, extending the ADR-015 §2 step 2 / HIGH-1 commitment to the cascade path.
3. **`scaffolding_repo` is removed from the agent surface.** It is an architecture-config detail, not an aesthetic weight choice. Supplying it in `cascade_config` is rejected with a named-field contract error (the field name is public schema knowledge — no oracle). The server relies on `cascade.validate_config`'s hard-coded default (`stabilityai/stable-cascade`), operator-trusted analogously to `--default-model`, and is **not** routed through the agent-facing resolver. Removal — not silent-ignore — closes its input attack surface (mirrors the slice-3 `_GENERATE_REMOVED_FIELDS` rule).
4. **Output** (`resolved_params.cascade_config`) renders `stage_*` as catalog **names**, never absolute paths; `scaffolding_repo` is absent from the response. The optional `notices` array carries the path-discard INFO for any stage resolved from a path-shaped value (name-only — never the agent-supplied raw value, INFO-2).

The slice-1 `_within(--model-base)` net is **retained, not removed** — it runs at the cascade load boundary as request-time defense-in-depth on each resolved `abs_path` (ADR-015 §2 step 3; the §4 end-state that drops it stays deferred — MEDIUM-3 preconditions unmet). `allow_hf_download=False` stays hard-coded at every cascade load call site. The on-disk PNG redaction (`redact_metadata_for_png`, which already basenames `cascade_config.stage_*` / `scaffolding_repo`) is unchanged — that is a separate sink from the in-frame response renderer.

## Invariants (must always be true)

1. **No absolute path crosses the MCP boundary** — neither as cascade input nor in any response field. `resolved_params.cascade_config.stage_*` render as catalog names; `scaffolding_repo` is absent from the response; no `/`-bearing string appears under any response key.
2. **`stage_c` / `stage_b` / `stage_a` resolve via `resolve_reference`** against kind ∈ {`model`, `transformer`}. Path-shaped values are basename-stripped → resolved → INFO path-discard notice (name-only).
3. **Uniform error** — every cascade reference-resolution failure returns byte-identical `"reference not available"`; fine cause on the stderr audit line only.
4. **`scaffolding_repo` supplied by the agent is rejected** with a named-field contract `ValidationError` before any resolution; the server otherwise uses the cascade default (not agent-affectable).
5. **Load-boundary `_within(--model-base)`** re-check on every resolved stage `abs_path` immediately before `build_pipelines`.
6. **`allow_hf_download=False`** at every cascade load call site.
7. **On-disk PNG redaction unchanged** (`redact_metadata_for_png` cascade branch stays as is).
8. **Purely additive resolver change** — extending `expected_kind` to accept a tuple leaves every existing single-`str`/`None` caller (the entire slice-3 non-cascade path) byte-unaffected; the non-cascade `generate`, `list_*`, and `extract_params`/`iterate` slots are untouched.

## Failure semantics

Fail-closed. Any stage resolution failure → uniform error, no fallback to a stale/raw path, no partial load. Wrong-kind or unknown stage name is indistinguishable from a miss in the agent-facing frame. Removed-field (`scaffolding_repo`) → contract `ValidationError` before resolution. A cache-miss on the *default* scaffolding repo surfaces as an internal load-time error (sanitized), not as a reference-resolution oracle (it is not agent-affectable).

## Out of scope

- The slice-1 cascade **topology** (prior/decoder/vqgan assembly, dtype handling, dimension alignment) — untouched.
- `scaffolding_repo` as a future spawn-time operator flag — not now.
- Non-cascade `generate`, `list_*`, `extract_params` (slice 4), `iterate` (slice 5), `edit` stub (slice 6).
- Promoting the catalog to sole path-authority (ADR-015 §4 end-state) — preconditions tracked separately.

## Negative cases required

- `stage_c` = unknown name → `"reference not available"` (audit `UnknownName`).
- `stage_c` = a **lora** catalog name → `"reference not available"` (audit `KindMismatch`) — proves the kind-set excludes `lora`.
- `stage_c` = path-shaped (`/etc/passwd`) → basename-stripped, miss → uniform error; the raw value never appears in any notice or error.
- `stage_c` with an embedded NUL byte → uniform error (`MalformedReference`), not an exception.
- `scaffolding_repo` present in `cascade_config` → contract `ValidationError` naming the field; no resolution attempted.
- A valid `transformer`-kind stage name (`stable_cascade_stage_c`) and a valid `model`-kind stage name both resolve `ok`.
- Response `resolved_params.cascade_config` contains no `/`-bearing string under any key.
- Catalog hit whose `abs_path` was moved post-spawn → uniform error (`PathMoved`), no stale-path load.
- Byte-equality: the cascade uniform-error message is identical to the non-cascade one (extends keystone test N5).

## Proof hooks

- Positive: `./.venv/bin/python3 test_mcp_server.py` (new cascade-name cases; 0 failures) + the full 9-suite run stays green (`test_manual_loop.py`, `test_multistage.py`, `test_params_schema.py`, `test_cascade.py`, `test_machine_boundary_validator.py`, `test_iterate.py`, `test_samplers.py`, `test_server_robustness.py`, `test_mcp_server.py`).
- Negative: the nine cases above, each asserting the byte-identical uniform message and that the audit cause differs while the agent-facing frame does not.

## Red Zone ownership

MCP reference contract — owned by **Grant**. AI authors the diff; `code-reviewer` (Opus) + `security-auditor` (Opus) both required before commit; ADR-015 Changelog appended at closure; this Vision's Status flipped to IMPLEMENTED at closure.

## Open Questions

- **OQ-1 (resolved 2026-06-24)** — extend `resolve_reference`'s `expected_kind` to accept a tuple of kinds; cascade stages resolve against `{model, transformer}`. Backward-compatible (bare `str` / `None` unchanged).
- **OQ-2 (resolved 2026-06-24)** — `scaffolding_repo` dropped from the agent surface; server uses the cascade default; HF-cache-miss on that default is a load-time error, not a reference oracle. Acceptable.
