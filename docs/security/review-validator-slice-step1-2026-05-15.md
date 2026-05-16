# Security Audit — Validator Slice Step 1 (ADR-012)

**Date:** 2026-05-15
**Reviewer:** `security-auditor` subagent (Opus, model pinned at invocation per project CLAUDE.md review-bar)
**Scope:** Step 1 of the machine-boundary validator slice — `comfyless/params_validation.py`, `test_machine_boundary_validator.py`, Vision status flip.
**AI-Disclosure:** Claude (Opus 4.7, 1M context) authored the audit; Grant reviewed.

---

## Verdict

**CLEAN with two MEDIUM observations and several INFO notes.**

The validator closes the bool-as-int loophole correctly, the int→float safe cast is sound for the realistic threat model, and purity holds. The two MEDIUM items are forward-looking (regression-resistance + a structural concern about a non-validator code path), not defects in the as-written module.

---

## Findings

### 1. Bool-before-int check ordering — CORRECT, but ordering invariant has no test

**Severity:** MEDIUM (regression-resistance gap)
**Location:** `comfyless/params_validation.py:115-129` (`_KIND_INT`), `131-147` (`_KIND_FLOAT`)
**Threat:** The current order — `isinstance(value, bool)` rejection BEFORE `isinstance(value, int)` accept — is correct on both numeric branches. The float branch even places the bool reject before the float/int branches, which is the strongest version. The loophole is closed today.

The exploit vector if a maintainer reverses the order: `isinstance(True, int)` returns True, so swapping the two lines silently re-opens the bool-as-int loophole. **The test suite catches this** via N1-N7 (each numeric field rejects `True`), so the runtime behavior is guarded. What is NOT guarded is a more subtle regression: a future contributor refactors `_check_field` to share a helper, accidentally drops one of the two bool rejects (say, in the float branch only). N6/N7 still cover both float fields, so this is well-covered for the current schema, but if a new canonical-float field is added to `_SCHEMA_KIND` without a corresponding `N6/N7`-style test, the bool-as-float loophole opens on the new field.

**Recommendation:** Add a parametric test that iterates over `_SCHEMA_KIND` itself (not a hand-rolled field list) and asserts `True`/`False` are rejected for every field whose kind is `_KIND_INT`, `_KIND_FLOAT`, or `_KIND_FLOAT_NONE`. This makes schema additions self-cover.

**Status:** ADDRESSED in step 1 fold-in — parametric test added.

---

### 2. int→float safe cast — SAFE for the threat model

**Severity:** INFO
**Location:** `params_validation.py:140-143`, `205-208`
**Threat:** ADR-012 §3's claim is correct for the realistic LLM-agent threat model. The cast happens only after `not isinstance(value, bool) and isinstance(value, int)` is verified. Within CPython, the only `int` subclasses in the stdlib are `bool` (explicitly rejected) and the long-deprecated `enum.IntEnum` / `enum.IntFlag` (which would be rejected at JSON-deserialization, not constructible from `json.loads`).

The numpy concern: `numpy.int64` does NOT inherit from Python's `int` — `isinstance(numpy.int64(5), int)` is `False` on all numpy versions ≥ 1.20. `pydantic`'s integer types likewise do not subclass `int`. `decimal.Decimal` does not subclass `int`. **No documented Python int subclass with overridden `__float__` exists in the dependencies the validator would realistically see.** The MCP transport feeds the validator via `json.loads`, which returns only `int`/`float`/`str`/`bool`/`None`/`list`/`dict` — no exotic numeric types.

The remaining residual risk: a future caller constructs the dict in-process (not via JSON) and supplies a class whose `__init_subclass__` made it an `int` subclass with a malicious `__float__`. This is structurally unreachable from the MCP / server / iterate paths today.

**Recommendation:** None; behavior matches ADR §3. INFO only — note in TECH_DEBT that if any in-process caller of `validate_machine_request` is added that does NOT route through JSON deserialization, this assumption should be re-verified.

---

### 3. No float→int — fully prevented

**Severity:** INFO
**Location:** `params_validation.py` entire module
**Threat:** N30's static `int(` source-grep correctly detects zero call sites in the module. There is no `dict()` copy semantics path that calls `int()` on values (the `validated[key] = value` and `cleaned = dict(entry)` lines just copy references). The `@dataclass(frozen=True)` machinery does not coerce field types — `dataclasses` stores annotations but does not validate them. No `int()` call can reach a caller-supplied float through any indirect path in this module.

The validator imports only `dataclasses` and `typing` (the `Any`/`Optional` types). Neither performs runtime coercion. The validator's runtime path cannot indirectly call `int()` on a float.

**Recommendation:** None. The static check is sufficient.

---

### 4. Purity — verified

**Severity:** INFO
**Location:** `params_validation.py`, imports at lines 17-20
**Threat:** Imports limited to `dataclasses` and `typing` — neither touches filesystem on import in CPython. The N20 mock list (`open`, `os.path.exists`, `os.path.realpath`, `os.stat`) is sufficient for the current source. The validator never constructs `pathlib.Path` objects (which can stat in some operations) and never imports `os` directly.

Future-regression risk: if a maintainer adds `from pathlib import Path` and constructs `Path(value)` for a path field, that's IO-free today but would set a pattern that drifts (a subsequent `Path(value).exists()` would not be caught by the N20 mock list because it would patch `pathlib.Path.exists`, not `os.path.exists`).

**Recommendation:** Extend N20 to also assert `pathlib.Path.exists`, `pathlib.Path.stat`, `pathlib.Path.is_file`, `pathlib.Path.is_dir`, and `pathlib.Path.resolve` are not called. INFO; preventive.

**Status:** ADDRESSED in step 1 fold-in — N20 mock list extended.

---

### 5. Structured error `field` value — attacker-controlled string never used as code

**Severity:** INFO
**Location:** `params_validation.py:178-214` (LoRA error construction)
**Threat:** The `field` value for LoRA errors interpolates `index` (an int controlled by the loop, not the caller) into a fixed format string `f"loras[{index}].weight"`. The caller cannot inject characters into `field` — the only path is through the loop index, which is `enumerate`-generated.

For the top-level error, `field` is always one of the keys in `_ALL_FIELDS` (validator-controlled) or the literal string `"<root>"`. The `reason` field uses `type(value).__name__` which is the class's `__name__` attribute (Python identifier syntax, not arbitrary string).

The `field` value cannot contain caller-controlled data. Audit log consumers downstream (ADR-011 §3b) can format this without risk of log injection from the validator's structured error.

**Recommendation:** None. The structured error shape is safe.

---

### 6. Information disclosure via `type(value).__name__` — acceptable for threat model

**Severity:** INFO
**Location:** `params_validation.py:112, 128, 146, 154, 162, 181, 197, 213`
**Threat:** Exposing `type(value).__name__` discloses the Python class name of the caller's input — e.g., `dict`, `list`, `int`, `NoneType`. For a malicious LLM caller sending a custom class instance, this could leak the class name (`MyCustomThing`). In the threat model, the caller is either the LLM agent itself (which already knows what it sent) or a malicious client driving the MCP server (same-uid threat model per ADR-011 / F-9 of the 2026-04-28 review — already has process-level access).

There is no information disclosed that the attacker did not already possess. The exposure is acceptable.

**Recommendation:** None.

---

### 7. Race conditions on module-level dicts

**Severity:** MEDIUM
**Location:** `params_validation.py:37-73` (`_SCHEMA_KIND`, `_RUNTIME_KIND`, `_ALL_FIELDS`)
**Threat:** The three module-level dicts are NOT frozen. They are constructed at import time and only read (via `_ALL_FIELDS.get(key)`) during validation. CPython's GIL guarantees a single `dict.get()` is atomic, so a concurrent reader cannot observe a torn read.

However, the dicts are publicly mutable. A test, a misbehaving import, or a future maintainer could call `_SCHEMA_KIND.pop("cfg_scale")` or `_SCHEMA_KIND["cfg_scale"] = _KIND_INT` and silently weaken the validator globally. This is not a thread-safety issue (the read is atomic) but it IS a module-integrity issue: there is no structural barrier preventing mutation of the canonical schema map at runtime.

For the threat model where the validator becomes security truth, "the canonical schema map is mutable from any module that can import it" is a weaker invariant than the rest of the design supports.

**Recommendation:** Convert `_SCHEMA_KIND`, `_RUNTIME_KIND`, `_ALL_FIELDS` to `types.MappingProxyType` views, making them read-only at the language level. This is a one-line tightening per dict, structural (T2-tier — the mutation becomes literally impossible, not policed), and aligns with the "frozen dataclass" choice already made for `ValidationResult`.

**Status:** ADDRESSED in step 1 fold-in — all three dicts wrapped in `MappingProxyType`.

---

### 8. Regression-resistance — most-likely future weakening

**Severity:** INFO
**Location:** future maintainers, schema additions
**Threat (most-likely regression scenarios, ranked):**

  1. **New canonical-float or canonical-int field added to `_SCHEMA_KIND` without a corresponding N1-N9-style test.** The current tests hard-code field names; adding `guidance_scale` (canonical float) to the schema would not cause a test failure even if the validator's bool-reject branch had been broken on float. → Mitigated by Finding 1's parametric test recommendation.

  2. **Refactor merges `_KIND_INT` and `_KIND_FLOAT` branches into a shared helper.** The bool-reject must remain in the shared path. → Caught by N1-N7 today, but a maintainer could refactor "for clarity" and weaken the ordering. The parametric test from Finding 1 catches this.

  3. **Schema versioning lands (currently out of scope) and `_ALL_FIELDS` becomes dynamic.** A future migration that reads schema from disk or env would break Invariant 6 (purity). → Caught by N20 today; recommend ADR-009 amendments that add schema versioning explicitly re-run security review.

  4. **`unknown keys pass through unchanged` is documented as out-of-scope.** A malicious caller can inject arbitrary unknown keys into the validated payload, which then flow downstream. Step 3 (server) and step 4 (iterate) must each defend against the unknown-key pass-through if they consume `validated[key]` for any key not in `_ALL_FIELDS`. This is correctly out-of-scope for step 1 per ADR-012 / Vision, but flag it for the auditor reviewing step 3.

---

## Note on broader slice (steps 3 and 4)

When step 3 (server) and step 4 (iterate) wire up, specifically check:

1. **Unknown-key pass-through.** `validated[key] = value` at line 247 forwards any unknown key unchanged. Step 3 / step 4 must NOT trust unknown keys — confirm `server.py:_handle_generate` / iterate's request handler ignore or explicitly allowlist fields beyond `_ALL_FIELDS`. Otherwise the validator's "structured pass-through" becomes a bypass for fields the downstream code does NOT validate.

2. **N18 grid extension.** The grid is single-site today (validator only). When steps 3/4 land, the grid must assert that `_validate_request(server)`, `validate_machine_request(canonical)`, and `iterate._validate_lora` all return identical accept/reject decisions for the full grid. Cross-call-site parity is the codex 05 Gap 2 ask; verify the grid grew to cover it.

3. **N19 static grep.** Skip-marked today. Confirm it activates and asserts ZERO `isinstance(...)` matches in `comfyless/server.py:_validate_request`'s body AND in iterate's per-LoRA validation block. The current `_validate_request` at `comfyless/server.py:123-160` has six `isinstance` predicates that must all be deleted by step 3.

4. **Server `\x00` null-byte check at `server.py:160`.** That check is outside the canonical validator. Step 3 must decide: does it migrate into the canonical validator (then it's universal), or stay in server (then it's server-specific defense). The "one source of truth" invariant favors the former; surface this explicitly when step 3 reviews.

5. **`validate_machine_request` returns a new dict, but `validated["loras"]` for non-list-typed `loras` would pass through unchanged.** Line 254 guards with `isinstance(validated["loras"], list)` — but `loras` is `_KIND_LIST`, so a non-list would have been rejected at line 158. Confirmed safe. Step 3 should not add a parallel guard.

6. **`field` value `"<root>"` in the non-dict rejection.** Audit-line consumers must not assume `field` always matches an entry in `_ALL_FIELDS`. Confirm step 3's audit-line formatter handles the `<root>` and `loras[N].path|weight` patterns without crashing.

---

## Step 1 fold-in summary

In response to the audit, step 1 lands with these fold-ins before commit (none change the design — all are regression-resistance tightenings the auditor recommended):

- Finding 1 → parametric `_SCHEMA_KIND` bool-reject test added.
- Finding 4 → N20 mock list extended to `pathlib.Path.{exists,stat,is_file,is_dir,resolve}`.
- Finding 7 → `_SCHEMA_KIND`, `_RUNTIME_KIND`, `_ALL_FIELDS` wrapped in `types.MappingProxyType`.

Plus parallel `code-reviewer` recommendations:
- Frozen-dataclass test tightened to catch `dataclasses.FrozenInstanceError` specifically.
- N30 static check converted from regex/triple-quote-toggle to `ast.parse` walk (structurally tight).
- N18 fixture grid expanded to cover all N1-N17 reject scenarios.
- Invariant 8 drift (code-reviewer F1) — deferred to step 2 by design (the schema lives in `generate.py` and step 2 is where the collapse happens). Step 2 will wire `_SCHEMA_KIND` to derive from `COMFYLESS_SCHEMA`.

Re-run of `test_machine_boundary_validator.py` + 7 regression suites required before commit.

---

## Files referenced

- `/home/gawkahn/projects/ai-lab/code/Eric_Qwen_Edit_Experiments/comfyless/params_validation.py`
- `/home/gawkahn/projects/ai-lab/code/Eric_Qwen_Edit_Experiments/test_machine_boundary_validator.py`
- `/home/gawkahn/projects/ai-lab/code/Eric_Qwen_Edit_Experiments/docs/decisions/ADR-012-machine-boundary-validator.md`
- `/home/gawkahn/projects/ai-lab/code/Eric_Qwen_Edit_Experiments/docs/vision/slice-machine-boundary-validator.md`
- `/home/gawkahn/projects/ai-lab/code/Eric_Qwen_Edit_Experiments/docs/security/review-comfyless-mcp-server-2026-04-28.md` (prior security review establishing the LLM-agent threat model)
- `/home/gawkahn/projects/ai-lab/code/Eric_Qwen_Edit_Experiments/comfyless/server.py` (for step-3 context, not in slice scope)
