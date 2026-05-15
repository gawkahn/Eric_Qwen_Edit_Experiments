# ADR-012: One Canonical Machine-Boundary Validator for Comfyless

**Date:** 2026-05-04
**Status:** accepted
**AI-Disclosure:** Claude (Opus 4.7, 1M context) authored; Grant reviewed and accepted 2026-05-15.
**Cross-references:** ADR-001 (daemon socket security — validator plugs into the existing `_validate_request` shape), ADR-006 (`--json` bridge — explicit non-target; CLI / sidecar path stays permissive), ADR-009 (per-family defaults overlay — schema evolution rules unchanged), ADR-011 (MCP server — consumer; slice 1 imports the canonical validator from line one).
**Triggering review:** Codex code review 2026-05-01 — findings 02 F2 ("Server Request Schema Does Not Match COMFYLESS_SCHEMA"), 03 SF1 ("Server Schema Drift At IPC Boundary"), 04 Rec 2 ("Treat COMFYLESS_SCHEMA As A Real Contract Module"), 05 Gap 2 ("Server Schema vs COMFYLESS_SCHEMA"). Vision artifact: `docs/vision/slice-machine-boundary-validator.md` (committed `567fba3`).

---

## Context

The comfyless code today has three places that validate machine-boundary inputs against type rules:

1. `comfyless/server.py:_validate_request` — the daemon socket's IPC entrypoint validator.
2. `comfyless/iterate.py` — per-LoRA validation inside the iterate sweep.
3. (Future) `comfyless/mcp_server.py` request validators per ADR-011, slice 1+.

Each of these targets the same input shape — the same `loras[].path` strings, the same `cfg_scale` and `true_cfg_scale` numerics, the same `seed`/`steps`/`width`/`height` integers — but each implements its own type rules. Codex's 2026-05-01 review confirmed the rules have already drifted: `_validate_request` accepts `cfg_scale` as `float` only (rejecting `int`); `COMFYLESS_SCHEMA` declares the field as `(int, float)`; iterate's per-LoRA validation does not check `weight` types at all (TECH_DEBT H-3, 2026-04-23). When ADR-011's MCP server lands, it would be a fourth implementation of the same rules unless the architecture changes.

Two distinct concerns are tangled together in the current state and need to stay distinct in the resolution:

- **Type contract.** What types each named field accepts. The schema (`COMFYLESS_SCHEMA`) is the canonical declaration. Today the schema declares `(int, float)` for some fields, which collapses the contract: "any of these is fine." That is incoherent when the field actually has a defined downstream behavior — `cfg_scale` is a scalar multiplier on a tensor, mathematically `float`; `steps` is a discrete count, mathematically `int`. Typing a field as "either int or float" leaves callers and downstream code making different assumptions and creates a contract that doesn't match the behavior.

- **Boundary discipline.** What happens at each call site when a malformed value arrives. Three call sites today have three different answers: server `_validate_request` rejects with a structured error (fail-closed); iterate's LoRA loop silently allows whatever passes the existing string check; CLI `_validate_params` warn-and-keeps (permissive) so a sidecar from an older comfyless run can be replayed even if it has gone stale. The CLI's posture is correct for human replay; the server's posture is correct for machine-driven calls. The disagreement is an architectural rule made implicit, not designed in.

The Red Zone scope makes the type-confusion-via-`bool` issue worse: Python's `bool` is a subclass of `int`, so `isinstance(True, int)` is `True`. The current server validator's `isinstance(x, int)` checks accept `True`/`False` for fields like `seed` and `steps`. An LLM agent submitting `{"seed": True}` over MCP is not a hypothetical — it is a class of LLM serialization quirk that other tool surfaces have hit. Closing the loophole at every call site requires either three separate fixes (and a fourth when MCP lands) or one canonical validator that all sites import. The canonical-validator route is also what codex's recommendation 04 Rec 2 names directly: "treat COMFYLESS_SCHEMA as a real contract module."

This ADR records the architectural decision. The slice that implements it is captured in `docs/vision/slice-machine-boundary-validator.md` (vault: `Vision/Slice-Machine-Boundary-Validator.md`), which carries the per-invariant detail (9 invariants), the negative-case test matrix (N1–N30 covering bool rejection, numeric-width parity, LoRA validation, cross-call-site parity, purity, no-coercion, CLI preservation, regression baseline, float-rejection-for-int-canonical-fields), and the proof hooks. This ADR is the durable design rationale; the Vision is the per-slice planning artifact.

## Decision

Adopt one canonical machine-boundary validator function, derived from a tightened canonical schema, with an explicit boundary-asymmetry rule between machine and human entry points.

### 1. Single canonical function

Introduce `validate_machine_request(payload: dict) -> ValidationResult`. Final signature decided in the slice's Change Plan; result shape returns either a normalized payload (with the explicit cast applied where applicable per §3 below) or a structured error naming the offending field plus a machine-readable reason. One function, one import path. The Python module is `comfyless/params_validation.py` (final name in the Change Plan); both `comfyless/server.py:_validate_request` and the future `comfyless/mcp_server.py` request validator and `comfyless/iterate.py`'s per-LoRA validation import and call it. **No type predicate (`isinstance(x, int)`, `isinstance(x, float)`, etc.) appears at any call site outside the canonical validator.** All type checking goes through the one function.

### 2. Schema as single source of typing truth

`COMFYLESS_SCHEMA` declares ONE canonical type per numeric field. The prior `(int, float)` declarations on `cfg_scale`, `true_cfg_scale`, and LoRA `weight` collapse to single canonical types — `float` in all three cases (each field is multiplied by tensors downstream; `float` is the natural canonical, and matches ComfyUI's `FLOAT` declaration on the same fields). The canonical schema declaration is the source of truth; the validator reads from it; it does not re-declare.

The schema collapse is in scope of the validator slice — the validator can't be implemented or tested coherently against an "either is OK" schema declaration. `test_params_schema.py` (135 tests) gets minor updates wherever it asserts the literal `(int, float)` declaration shape; tests that exercise `_validate_params` *behavior* are unchanged because `_validate_params` stays permissive (warn-and-keep) on numeric inputs. The slice's commit body documents the test-count delta from the collapse.

### 3. Canonical-type-per-field with safe-cast in one direction only

For fields canonical-typed `float` (`cfg_scale`, `true_cfg_scale`, LoRA `weight`), the validator accepts the canonical type directly AND accepts `int` with an explicit `float(x)` cast at the validator boundary. The cast is mathematically a no-op for any value the diffusers stack consumes (int→float is lossless within the float64 mantissa precision), and the cast occurs only after `isinstance(x, int) and not isinstance(x, bool)` is verified — so there is no custom-`__float__` threat surface. Cast-after-typecheck is the validator's published contract; the cast is documented in the function's signature and the test matrix proves it (Vision N8/N9/N17 assert the validated payload's value is the cast `float`, not just that the input was accepted).

For fields canonical-typed `int` (`seed`, `steps`, `width`, `height`, `max_sequence_length`), the validator accepts the canonical type only. `float` is a structured rejection naming the offending field. **The validator NEVER calls `int()` on a float input** — float→int is the unsafe direction (silent truncation versus silent rounding versus refuse — the validator refuses to make that decision); the caller fixes the input. Vision N25–N29 cover the per-int-field rejection; N30 is the architectural runtime check that asserts `int()` is never invoked on a float-sourced value across the negative-case grid.

`bool` is rejected for every numeric field regardless of canonical type. This closes the `bool` is `int` subclass loophole at every call site at once (Vision N1–N7).

### 4. Boundary-asymmetry architectural rule

**Machine boundaries fail closed.** Daemon socket, MCP server, iterate sweep — each rejects malformed input with a structured error. No silent normalization, no field-default substitution, no partial acceptance. The one explicit cast (int→float for canonical-`float` fields) is part of the validator's published contract, not silent normalization.

**Human boundaries warn-and-keep.** CLI `_validate_params` (sidecar import, `--params` flag) preserves whatever the human or an older comfyless run produced. Warn loudly; keep the value. A sidecar from a 2026-04 run that contains `cfg_scale: 7` is still replayable when the schema collapses `cfg_scale` to canonical-`float` — a warning surfaces the type drift; the value passes through. The CLI's *acceptance set* does not shrink; only the *warning set* may grow.

This asymmetry is not a hack. It is the architectural rule that ADR-011 §3b (audit-line redaction) and §3e (MCP artifact redaction) both already cite: machine-driven inputs/outputs are higher-stakes (an LLM cannot self-correct from a warning the way a human can) and therefore demand strictness; human-driven inputs/outputs benefit from forgiveness (a user replaying yesterday's run shouldn't have to hand-edit a sidecar). The validator slice makes this rule explicit and codifies it as Vision invariant 7.

### 5. Pure function discipline

The validator is pure: no filesystem reads, no environment-variable reads, no network IO, no global state mutation. Validation results are deterministic given the input payload. This makes the function unit-testable in isolation, importable into in-process callers without side effects, and free of the "did it touch disk?" question that plagues ad-hoc validators (Vision N20/N21 assert these properties).

### 6. LoRA validation as a per-entry helper

LoRA entries are validated as a unit: `path` is a string; `weight` is `int` (validator-boundary cast to `float`) or `float` (pass-through), not `bool`, not `str`, not `None`; both fields must be present together. The same per-entry helper is called from machine-boundary validation AND from iterate's per-LoRA validation, with no third copy. Closes TECH_DEBT entry "loras[i]['weight'] not type-checked in `_validate_request`" (H-3, 2026-04-23) — the validator slice annotated that entry "Pending closure by:" 2026-05-04 and will flip it to `Resolved:` when the slice ships.

### 7. Slice mechanics

Implementation is gated on this ADR being accepted. Vision artifact: `docs/vision/slice-machine-boundary-validator.md`. The validator slice is itself L3 / Red Zone — the function it produces becomes security truth for every machine-facing surface — so the slice runs both `code-reviewer` (Opus, model pinned at invocation per global §5A) AND `security-auditor` (Opus, same) before commit, per project CLAUDE.md's review-bar entry that names the comfyless server's IPC validator as a §12-trigger surface.

The validator slice **must land before slice 1 of ADR-011** (the minimal `generate` MCP tool). ADR-011's 2026-05-04 Changelog amendment §(a) records this dependency. Slice 1's request validator imports `validate_machine_request` from line one — there is no transitional state where the MCP path uses a different type-checking shape.

## Alternatives Rejected

### A. Keep three separate validators with their own rules (status quo)

Rejected. The codex review's 2026-05-01 finding-set is direct evidence the status quo has already drifted (server schema ≠ COMFYLESS_SCHEMA — finding 02 F2; server schema drift at IPC boundary — finding 03 SF1; LoRA `weight` not type-checked — TECH_DEBT H-3). Adding a fourth implementation in the MCP server would compound the drift surface across two more entry points (ADR-011 lists six MCP tools; even if only `generate` and `iterate` validate request payloads in slice 1+, that is two more divergent type-rule sites). The validator slice closes the loop by removing the duplication's possibility, not just its current-day instances.

### B. Single validator class with sub-validators

A `MachineRequestValidator` class with `validate_seed`, `validate_cfg`, etc. methods, possibly with subclassing per tool. Rejected. Function-level granularity is sufficient for the scope (single-payload validation, no inheritance hierarchy, no plug-in points). A class adds ceremony — instantiation, method dispatch, `self` plumbing, subclass override surfaces — without adding capability. If future scope demands a class (per-tool validators with shared infrastructure across many MCP tools), refactoring from a function to a class is straightforward; pre-emptive class design is not. Vision invariant 1 is "exactly one function," not "exactly one validator object" — the function shape is the right scope for what's needed today.

### C. Make CLI strict like machine boundary

Unify the validators by making both fail-closed. Rejected. The CLI's permissive posture is a feature, not a bug: a user replaying a sidecar from an older comfyless run benefits from the validator absorbing minor type drift rather than rejecting the whole file. Hand-editing JSON for a one-off test should not be the path of least resistance for a human caller. The asymmetry is correct architecture for the two distinct caller classes; unifying would be the wrong unification.

### D. Strict typing with no coercion

Schema declares one canonical type; validator rejects the other type entirely. So `cfg_scale=7` (int) returns a structured error even though `7.0` would be accepted. Rejected. LLM agents naturally type `7` not `7.0` — JSON itself doesn't distinguish (Python's `json.loads` returns `int` for `"7"` and `float` for `"7.0"`); a tool surface that rejects on this distinction is unnecessarily friction-laden. The int→float direction is mathematically lossless within values the diffusers stack consumes, and the cast occurs post-typecheck in a verified-non-bool int — so the threat surface is essentially zero. Strict-with-no-coercion adds friction without security benefit.

### E. Float() everything (always coerce to float)

A simpler design: every numeric field becomes `float`, with `int()` coercion at downstream call sites if the consumer needs an int. Rejected. The unsafe direction (float→int, silent truncation versus rounding versus refuse) is exactly what the validator must not do. `steps=50.5` is a malformed input from any reasonable reading; the validator should refuse rather than silently floor or round or push the decision elsewhere. Pushing the `int()` decision to downstream code distributes the unsafe-direction problem across every consumer site rather than closing it once.

### F. Keep `(int, float)` widening

Schema declares "either is fine"; validator accepts either, no coercion, no safe-cast, no rejection on the unsafe direction. Rejected on architectural grounds. If a field has a defined downstream behavior, "either is fine" is an underspecified contract — downstream code that assumes one or the other gets occasional surprises (a buffer-size calculation that expects `float` and gets `int`; a range-check that expects `int` and gets `float`). The contract should match the actual behavior, and the actual behavior is canonical per field. Keeping the widening also keeps the codex 02 F2 finding open by definition — the validator can't assert "server validates the same shape COMFYLESS_SCHEMA declares" if the schema declaration is itself ambiguous.

### G. Schema versioning to track type-rule migrations

Migrate the schema with a version field; older sidecars validate against their declared version. Rejected as out of scope. ADR-009 owns schema evolution rules; this ADR is about the validator that consumes the schema's current state. The schema collapse from `(int, float)` to single canonical types is a one-time tightening, not the start of a migration regime. If future schema evolution becomes complex enough to warrant versioning, ADR-009 amendments handle it.

### H. Move all validation into the daemon, none in the entrypoints

Remove `_validate_request` from server, MCP, iterate; have the daemon process do all validation server-side. Rejected. In-process callers (the CLI auto-detect path that falls back to in-process when no socket is present, plus iterate's per-LoRA expansion which runs before any IPC) need validation too — the daemon-side approach would mean validation is skipped or duplicated for in-process and pre-IPC calls. A callable function imported by all entrypoints serves both transports without duplication and without a hidden in-process gap.

### I. Defer validator harmonization until after MCP slice 1

A timeline alternative: ship MCP slice 1 with its own validator, then unify later as a refactor. Rejected. Slice 1 is L3 / Red Zone; the validator it ships becomes security truth for the LLM-facing surface. Refactoring security truth after it has shipped is exactly the wrong order — the refactor itself becomes a Red Zone change, with a wider blast radius than landing the unified validator first. The validator slice is small; doing it first costs less than doing it later.

## Deferred / Out of Scope

- **CFG routing harmonization** (codex 01 F1, 04 Rec 1) — runtime-core cluster, gated on HTTP-transport readiness per ADR-011 §6 (2026-05-04 Changelog amendment §(c)).
- **`_load_pipeline()` parity between ComfyUI and comfyless** (codex 02 F1) — runtime-core cluster.
- **Multi-stage / UltraGen / save-node helper deduplication** (codex 01 F2, 01 F3, 01 F4, 04 Rec 4) — runtime-core cluster.
- **`is_hf_repo_id` / `resolve_hf_path` promotion to public API** (codex 03 SF3) — runtime-core cluster (the validator imports from the current `nodes/eric_diffusion_utils` path; promoting to a stable public surface is the runtime-core cluster's concern, not the validator's).
- **`--json` mode rewriting or removal** — preserved at zero further investment per ADR-011 §5; slice 1 marks it as legacy in source per ADR-011 2026-05-04 Changelog amendment §(b).
- **Changing CLI `_validate_params`** — explicitly preserved per Vision invariant 7. The schema collapse may grow the warning set; the acceptance set is unchanged.
- **Schema evolution rules** — ADR-009 unchanged. This ADR is about the validator that consumes the schema, not about how the schema evolves over time.
- **Edit-pipeline validation** (Qwen-Edit, Flux.2-Edit) — outside the validator slice's scope; covered by ADR-011 slice 6 (`edit` stub) and any future edit-CLI work.
- **MCP server implementation** — slice 1 of ADR-011; this ADR ships the validator substrate that slice 1 imports from.
- **Cascade-config field-by-field type rules** — Stable Cascade has its own schema per ADR-010 amendment 1; the canonical validator handles the cascade fields that flow through the standard `generate` payload (`stage_*`, `scaffolding_repo` get path validation per ADR-011 §3a/§3b) but does not own the inline cascade-JSON schema's per-field type rules. Unifying those is a future runtime-core-adjacent concern, not this slice.
- **Behavioral parity tests between CLI and server beyond the validator matrix** — full equivalence is a runtime-core-cluster concern; the validator matrix proves type-rule parity, not full behavioral parity.
- **Cross-call-site parity beyond the three sites named** — this ADR commits to server, iterate, and the future MCP server. Other in-process callers (test fixtures, debugging tools) MAY use the validator opportunistically; they are not required to and are not part of the slice's invariant set.

## Changelog

- **2026-05-04 (initial draft)**: ADR drafted in response to codex 2026-05-01 review's findings 02 F2 / 03 SF1 / 04 Rec 2 / 05 Gap 2 plus the 2026-04-23 LoRA-weight TECH_DEBT entry. Records the canonical-validator decision, the canonical-type-per-field schema collapse, the int→float safe-cast in one direction only, and the boundary-asymmetry rule (machine fail-closed; human warn-and-keep). Cross-referenced from ADR-011 2026-05-04 Changelog amendment §(a) as the prerequisite for slice 1. Vision artifact already committed (`567fba3`); slice implementation gated on this ADR being accepted. AI-Disclosure: Claude (Opus 4.7, 1M context) authored; Grant to review.
- **2026-05-15 (accepted)**: Grant accepted ADR as written; no design changes. Validator slice may now begin per the Vision artifact at `docs/vision/slice-machine-boundary-validator.md`. Slice is itself L3 / Red Zone — both `code-reviewer` and `security-auditor` (Opus, pinned at invocation per project CLAUDE.md review-bar) run before commit. Ordering remains: validator slice → ADR-011 slice 1 (`generate` MCP tool).
