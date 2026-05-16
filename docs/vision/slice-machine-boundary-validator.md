# Slice Vision — Machine-Boundary Validator Harmonization (pre-MCP)

**Date:** 2026-05-01
**ADR:** ADR-012 (to be authored before code lands; this Vision is the planning artifact). Cross-references ADR-001 (daemon socket security), ADR-006 (`--json` bridge), ADR-009 (per-family defaults overlay), ADR-011 (MCP server — consumer of this validator).
**Status:** approved 2026-05-15 — ADR-012 accepted; implementation underway via the validator slice.
**AI-Disclosure:** Claude (Opus 4.7, 1M context) authored; Grant reviewed and approved.
**Triggering review:** Codex code review 2026-05-01 — findings `02 F2`, `03 SF1`, `04 Rec 2`, `05 Gap 2`.

---

## Slice

Establish ONE canonical machine-boundary validator that owns input-typing for every machine-facing comfyless surface (daemon `_validate_request`, future MCP `generate` request validator, iterate's per-LoRA validation). Close the bool-as-int loophole and the numeric-width disagreement between `COMFYLESS_SCHEMA` and `comfyless/server.py:_validate_request`. CLI-side `_validate_params` may remain permissive (human replay) — machine boundaries fail closed.

Pre-MCP slice. Inserted between slice 0 (shipped: `mcp==1.27.0` dep-bump) and slice 1 (minimal `generate` MCP tool) to harden the input-fingerprint substrate before MCP code wraps it. Doing this during slice 1 would (a) violate SRR for slice 1 and (b) force every MCP test to be retested when the validator tightens underneath. Doing it after MCP ships means MCP would inherit a leaky validator and a third disagreeing schema.

## Posture

- **Boundary:** existing comfyless IPC (daemon socket) + future MCP stdio surface (substrate). Tightening of an in-place trust boundary; no new attack surface introduced.
- **Risk factors:** substrate of an L3 surface (slice 1 MCP); numeric type-confusion (`bool` is `int` subclass) actively bypasses today's server validation; LoRA weight unchecked.
- **Risk level:** **L3 (Red Zone).** The validator itself becomes security truth — every machine-facing input crosses it.
- **Lens:** team-portable (the rule "one validator owns machine-boundary input typing" is the kind a multi-developer team would adopt; not solo-defensible).

## Intent

Land in this slice:

- `comfyless/params_validation.py` (or a name finalized in the Change Plan) — a new module exposing a single function `validate_machine_request(payload: dict) -> ValidationResult` (exact signature decided in Change Plan; result shape returns either a normalized payload or a structured error naming the offending field + reason).
- The function derives its type rules from `COMFYLESS_SCHEMA` (canonical source) plus a small explicit list of runtime-only fields the daemon adds (`request_id`, etc.). One source of type truth.
- Strict numeric typing: `bool` rejected for any field typed `int` or `float` (closes the `bool`-is-`int`-subclass loophole). **Each numeric field in `COMFYLESS_SCHEMA` collapses to ONE canonical type** — the prior `(int, float)` declarations on `cfg_scale`, `true_cfg_scale`, and LoRA `weight` resolve to `float` (each is multiplied by tensors downstream; `float` is the natural canonical, and matches ComfyUI's `FLOAT` declaration on the same fields). For fields canonical-typed `float`, the validator accepts `int` from the caller and applies `float(x)` at the validator boundary — int→float is lossless for any value the diffusers stack consumes, and the cast happens AFTER `isinstance(x, int) and not isinstance(x, bool)` is verified, so there is no `__float__`-shenanigans surface. For fields canonical-typed `int` (`seed`, `steps`, `width`, `height`, `max_sequence_length`), `float` is a structured rejection — no silent truncation, no rounding decision pushed into the validator.
- COMFYLESS_SCHEMA tightening (in scope): the three `(int, float)` declarations collapse to single canonical types as above. The validator can't be implemented or tested coherently against an "either is OK" schema declaration; the canonical type is what the contract says. CLI `_validate_params` stays permissive (warn-and-keep) on numeric inputs regardless of the schema shape (invariant 7); the schema collapse does not alter its accept set, only the warning set may grow.
- LoRA entries validated as a unit: `path` is a string; `weight` is `int` (validator-boundary cast to `float`) or `float` (pass-through), not `bool`, not `str`, not `None`; both fields must be present together. Closes the existing TECH_DEBT entry on `loras[i]["weight"]`.
- `comfyless/server.py:_validate_request` rewritten as a thin wrapper that calls `validate_machine_request` and adapts its result into the existing server response shape. No type rules duplicated in `server.py`.
- `comfyless/iterate.py` (or wherever iterate's per-LoRA validation lives) calls the same canonical LoRA validation helper. No third copy of LoRA type rules.
- Test matrix proving canonical validator, server `_validate_request`, and iterate's LoRA validation accept/reject the SAME shapes for a fixture grid.

CLI-side `_validate_params` is **explicitly out of scope** for behavioral change — it stays permissive for human replay (sidecar import, `--params` flag). The architectural rule "machine boundaries fail closed; human boundaries warn-and-keep" is recorded in ADR-012.

## Invariants (must always be true)

1. Exactly **one** function defines machine-boundary input-type rules. Both `_validate_request` and the future MCP request validator import and call it. No type predicate (`isinstance(x, int)`, `isinstance(x, float)`, etc.) appears in `comfyless/server.py:_validate_request` or in `comfyless/mcp_server.py` — all type checking goes through the canonical validator.
2. `bool` is rejected for every field whose canonical type declaration is `int`. Specifically: `seed`, `steps`, `width`, `height`, `max_sequence_length` reject `True`/`False`.
3. `bool` is rejected for every field whose canonical type declaration includes `float`. Specifically: `cfg_scale`, `true_cfg_scale`, LoRA `weight` reject `True`/`False`.
4. **Canonical numeric type per field, with safe-cast in one direction only.** `COMFYLESS_SCHEMA` declares ONE canonical type per numeric field (`int` OR `float`, never `(int, float)`). For fields canonical-typed `float` (`cfg_scale`, `true_cfg_scale`, LoRA `weight`), the validator accepts the canonical type directly AND accepts `int` with an explicit `float(x)` cast at the validator boundary — int→float is lossless for these values, and the cast occurs only after `isinstance(x, int) and not isinstance(x, bool)` is verified, so there is no custom-`__float__` threat surface. For fields canonical-typed `int` (`seed`, `steps`, `width`, `height`, `max_sequence_length`), the validator accepts the canonical type only; `float` is a structured rejection naming the offending field. The validator NEVER calls `int()` on a float input — float→int is the unsafe direction (silent truncation vs. rounding) and the validator refuses to make that decision. `bool` remains rejected for every numeric field per invariants 2 and 3.
5. LoRA entries validated as a unit: `{"path": str, "weight": (int, float)}`. Either field missing → reject. Either field wrong type → reject. The same helper is called from machine-boundary validation AND from iterate's per-LoRA validation.
6. The validator is a pure function: no filesystem reads, no environment-variable reads, no network IO, no global state mutation. Validation results are deterministic given the input payload.
7. CLI-side `_validate_params` (sidecar import / `--params` flag) remains **permissive in posture** — warn-and-keep for human-replay use. Its acceptance set is unchanged: every input it previously kept, it still keeps. The schema-collapse change in invariant 4 may grow the set of inputs it warns about (e.g., a sidecar with `cfg_scale: 7` triggers a warning where it previously did not, because the schema now declares `cfg_scale` as canonical `float`), but the warn-and-keep behavior is preserved — `_validate_params` does not start rejecting inputs that the prior version accepted. This invariant captures both the architectural rule (machine boundaries fail closed; human boundaries warn-and-keep) and the explicit non-change to `_validate_params`'s acceptance set.
8. `COMFYLESS_SCHEMA` remains the single source of canonical-key + canonical-type declarations. The validator reads from it; it does not re-declare types.
9. Validator failure response names the offending field and the rule violated (e.g., `{"error": "invalid_type", "field": "steps", "reason": "bool not accepted for int field"}`). Audit-line consumers (slice 1 MCP) can format from this shape without parsing free-text.

## Failure semantics

- **Fail-closed at request time** for every machine-boundary call site: validation failure returns a structured error; no silent type coercion in the unsafe direction (no `int()` of a `float` input; no `float()` of a `bool`; no `int()` of a `bool`), no field default substitution, no partial acceptance ("we'll let `steps=True` through and use `1`"). The one explicit cast — `int → float` for fields canonical-typed `float` — is documented in invariant 4, type-checked before the cast, and not "silent" because it's part of the validator's published contract.
- **CLI side unchanged:** `_validate_params` still warns-and-keeps malformed values for human replay. The asymmetry is intentional and named in invariant 7.
- **Validator failure in iterate:** an invalid LoRA entry in an iterate request rejects the entire request, not just the malformed entry. Partial-success on iterate input is not introduced.
- **No exception propagation across the boundary:** the validator returns a result type; it never raises. Wrapping IPC code converts the result into the appropriate response shape (server JSON, MCP error frame).

## Out of scope (explicit)

- **CFG routing harmonization** (codex `01 F1`, `04 Rec 1`) — backlog, runtime-core cluster, gated on HTTP transport per separate ADR.
- **`_load_pipeline()` parity** (codex `02 F1`) — backlog, runtime-core cluster.
- **Multi-stage / UltraGen / save-node helper deduplication** (codex `01 F2`, `01 F4`) — backlog.
- **`is_hf_repo_id` promotion to public API** (codex `03 SF3`) — backlog, runtime-core cluster.
- **`--json` mode docstring warning** (codex `02 F3`) — slice 1 invariant, NOT this slice. Drives MCP-adjacent doc clarity, not validation logic.
- **Removing or rewriting `--json` mode** — separate ADR amendment if pursued.
- **Changing `_validate_params`** (the human-side validator) — explicitly preserved per invariant 7.
- **Schema versioning** — `COMFYLESS_SCHEMA` evolution rules (ADR-009) are unchanged.
- **Behavioral parity tests between CLI and server beyond the validator matrix** — full equivalence is a runtime-core-slice concern.
- **MCP server implementation** — slice 1; this slice ships the substrate only.

## Negative cases (required)

Each becomes a test in `test_machine_boundary_validator.py` (no pytest; same `python3 test_<name>.py` invocation as the other 7 suites).

**Bool-as-int rejection (closes the existing loophole):**

- **N1:** `validate_machine_request({"steps": True, ...})` → reject; field=`steps`, reason names `bool`.
- **N2:** Same for `seed=True`.
- **N3:** Same for `width=True`.
- **N4:** Same for `height=True`.
- **N5:** Same for `max_sequence_length=True`.

**Bool rejected for float-typed fields:**

- **N6:** `cfg_scale=True` → reject.
- **N7:** `true_cfg_scale=True` → reject.

**Numeric-width parity with COMFYLESS_SCHEMA (canonical-`float` fields, safe int→float cast):**

- **N8:** `cfg_scale=4` (int) → accept; assert the validated payload's `cfg_scale` is `4.0` (a `float`) — validator-boundary safe cast applied per invariant 4.
- **N9:** `true_cfg_scale=4` (int) → accept; assert validated `true_cfg_scale` is `4.0` (a `float`).
- **N10:** `cfg_scale=4.0` (float) → accept; pass-through (validated payload's `cfg_scale` remains `4.0`).
- **N11:** `true_cfg_scale=None` → accept (matches `COMFYLESS_SCHEMA` allowance for omitting CFG override).

**LoRA validation:**

- **N12:** `loras=[{"path": "/m/x.safetensors", "weight": "heavy"}]` → reject; field=`loras[0].weight`, reason names `str`.
- **N13:** `loras=[{"path": "/m/x.safetensors", "weight": True}]` → reject; field=`loras[0].weight`, reason names `bool`.
- **N14:** `loras=[{"path": "/m/x.safetensors"}]` → reject; field=`loras[0]`, reason names missing `weight`.
- **N15:** `loras=[{"weight": 0.8}]` → reject; field=`loras[0]`, reason names missing `path`.
- **N16:** `loras=[{"path": "/m/x.safetensors", "weight": 0.8}]` → accept.
- **N17:** `loras=[{"path": "/m/x.safetensors", "weight": 1}]` → accept; assert validated `loras[0].weight` is `1.0` (a `float`) — validator-boundary safe cast applied (LoRA `weight` is canonical-`float`).

**Cross-call-site parity (the matrix codex `05 Gap 2` calls for):**

- **N18:** Fixture grid of 12 payloads (mix of valid + invalid) — assert `validate_machine_request`, `_validate_request` (server), and iterate's per-LoRA validator return the same accept/reject decision for every fixture. Failure = type predicate has leaked back into a call site outside the canonical validator.
- **N19:** Static check — `grep -E "isinstance.*\b(int|float|bool|str)\b" comfyless/server.py:_validate_request comfyless/iterate.py:<lora_validation>` returns no matches. (Run as a test asserting the count is zero.)

**Purity / no side effects:**

- **N20:** `validate_machine_request` called with a fully invalid payload does NOT touch the filesystem (assert via `unittest.mock.patch` on `os.path.exists`, `open`, etc.).
- **N21:** Calling the validator twice with identical payloads returns identical result objects (deterministic).

**No silent coercion:**

- **N22:** `cfg_scale=4` is accepted as `4` (int) and forwarded as int. The validator does not silently convert to `4.0`. Downstream code receives the same numeric type the caller sent. This guards against a future "we'll just float() everything" regression.

**CLI-side preservation:**

- **N23:** Existing `test_params_schema.py` still passes — `_validate_params` behavior unchanged. Tests that depend on `_validate_params` warn-and-keep behavior for malformed sidecar values still pass.

**Float rejected for int-canonical fields (no float→int coercion):**

- **N25:** `steps=50.0` → reject; field=`steps`, reason names "float not accepted for int field."
- **N26:** `seed=42.0` → reject; same shape.
- **N27:** `width=1024.0` → reject.
- **N28:** `height=1024.0` → reject.
- **N29:** `max_sequence_length=256.0` → reject.
- **N30:** Architectural — across the full negative-case fixture grid, the validator NEVER invokes `builtins.int` on a value that was supplied as a `float`. Implemented as a runtime patch: `unittest.mock.patch("comfyless.params_validation.int", side_effect=...)` (or whatever import path the validator uses) records every `int()` call; assert the recorded set never contains a value whose source-type was `float`. Guards against a future "we'll just `int()` everything" regression that would silently truncate caller input.

**Regression baseline:**

- **N24:** All 7 existing test suites pass — 732/732 (or whatever current count is) modulo small adjustments to `test_params_schema.py` for the canonical-type schema collapse (see Proof hooks). Adjustments are described in the slice's commit body; net suite count documented at commit time.

## Proof hooks

- **`python3 test_machine_boundary_validator.py`** — new test suite covering N1–N22.
- **`python3 test_params_schema.py`** — must still pass after the schema-collapse-driven test updates. Tests that asserted the literal `(int, float)` declaration shape on `cfg_scale`, `true_cfg_scale`, or LoRA `weight` schema entries become tests asserting the canonical-`float` shape. Tests that exercise `_validate_params` *behavior* should be unchanged — `_validate_params` stays warn-and-keep on numeric inputs (invariant 7) and its acceptance set does not shrink. Test count may shift by a small number; the deviation from 135 is documented in the slice's commit body alongside the schema diff.
- **`python3 test_iterate.py`** — must still pass at 92/92 (proves iterate's per-LoRA validation was correctly migrated, not broken).
- **`python3 test_server_robustness.py`** — must still pass at 8/8 (proves server `_validate_request` rewrite did not break IPC behavior).
- **All 7 existing suites** — 732/732 (regression baseline).
- **`grep`-based static check** for `isinstance` predicates outside the canonical validator (N19) wired into the new test suite.

## Red Zone ownership

- **Validator function signature + numeric type rules:** owned by **Grant** — AI-generated only, not sole author. (This becomes security truth for every machine-facing surface.)
- **Server integration** (replacement of `_validate_request` body): owned by **Grant** — verifies the existing IPC response shape is preserved and no new field-naming divergence is introduced.
- **Iterate integration** (per-LoRA validation call site): owned by **Grant** — verifies iterate semantics (one bad LoRA → reject the whole request) match the existing behavior or are explicitly changed.
- **CLI non-change discipline** (invariant 7): owned by **Grant** — confirms `_validate_params` was not modified.
- **ADR-012 is the design source of truth** — drafted alongside this Vision, accepted before any implementation commit.

## Pointers

- **Triggering codex findings:** `02 F2` (server schema ≠ COMFYLESS_SCHEMA), `03 SF1` (server schema drift at IPC boundary), `04 Rec 2` (treat COMFYLESS_SCHEMA as a real contract module), `05 Gap 2` (server schema vs COMFYLESS_SCHEMA test matrix). Vault: `~/obsidian/vaults/vault1/10_Projects/Image_gen/Codex_code_review/`.
- **Plan response that drove this slice:** session 2026-05-01 — codex review disposition table.
- **Existing TECH_DEBT closed by this slice:** `loras[i]["weight"]` not type-checked in `_validate_request`.
- **Cross-referenced ADRs:** ADR-001 (daemon socket security — boundary owner), ADR-006 (`--json` bridge — explicit non-target), ADR-009 (per-family defaults overlay — the canonical schema's evolution rules unchanged), ADR-011 (MCP server — consumes this validator).
- **Successor slice:** Slice 1 (minimal `generate` MCP tool, ADR-011) — its request validator imports and calls `validate_machine_request` from line one.
- **Backlogged items not in this slice:** runtime-core cluster (CFG routing, `_load_pipeline()` parity, helper deduplication, `is_hf_repo_id` promotion, etc.) — gated on HTTP transport readiness per the planned ADR-011 Changelog amendment.
