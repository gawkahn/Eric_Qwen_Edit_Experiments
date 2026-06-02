# Slice 3 Vision — Migrate `generate` (+ cascade) to catalog-name reference resolution

**Date:** 2026-06-01
**ADR:** [ADR-015](../decisions/ADR-015-mcp-catalog-reference-resolution.md) (Status: accepted; rounds 1+2 CLEAN). Slice-3 row of the [ADR-015 §5](../decisions/ADR-015-mcp-catalog-reference-resolution.md) revised plan. Parent: [ADR-011](../decisions/ADR-011-comfyless-mcp-server.md). Builds on the slice-2 catalog ([slice-2-mcp-catalog.md](slice-2-mcp-catalog.md), IMPLEMENTED 2026-05-30) and slice 2b `list_transformers` (IMPLEMENTED 2026-05-31).
**Status:** APPROVED by Grant 2026-06-02 (open questions resolved: OQ-A `transformer`→name + drop `vae`/`text_encoder`/`text_encoder_2` from MCP schema; OQ-B rename set confirmed; OQ-C non-cascade `generate` only, cascade → new slice 3b; OQ-D `--default-model` stays operator-trusted path; cadence: multi-step, reviewed per step). Implementation may begin per the Change Plan.
**AI-Disclosure:** Claude (Opus 4.8, 1M context) authored; Grant reviewed and approved.

---

## Posture

> **Posture:** Boundary: integration (LLM-agent → comfyless `generate`/`generate_cascade` MCP request/response contract). Risk factors: **near security truth** (this slice ships the load-bearing uniform-error contract — ADR-015 §2 step 2, HIGH-1 — and removes absolute paths from the agent-facing input AND output of the shipped slice-1 contract); **external exposure** (the LLM agent drives the reference inputs and reads the response); **broad impact** (changes the shipped slice-1 `generate` contract that 8 other surfaces and the whole MCP test suite depend on).

## Slice

The slice that makes the slice-2 catalog **load-bearing for `generate`** (the non-cascade path only; OQ-C resolved → cascade migration is **new slice 3b**). Until now the catalog has been built-but-unconsumed (slice 2 invariant 14). Slice 3 rewires `_handle_generate` so that:

1. **Input** weight references arrive as **opaque catalog names**, not filesystem paths. A path-shaped value has its directory component discarded and its basename resolved through the catalog (ADR-015 §2 basename-strip rule).
2. **Resolution failure** of *any* reference, from *any* cause, returns **one identical agent-facing error** — `"reference not available"` — with the fine-grained cause (`UnknownName` / `PathMoved` / `HFCacheMiss` / `WithinFailure` / `MalformedReference`) written **only** to the stderr audit line (ADR-015 §2 step 2, HIGH-1). This is the load-bearing security commitment of the whole ADR.
3. **Output** (`resolved_params`) renders every weight reference as a **catalog name**, never an absolute path. A new optional `notices: [{level, message}]` array carries the path-discard INFO notice (ADR-015 §3).

This **closes the HF-cache enumeration oracle** recorded in `TECH_DEBT.md` Security 2026-05-17 — the slice-1 `HFCacheMiss` error class became distinguishable from `PathAllowlist`/`UnknownName`, letting an agent probe which HF repos / paths exist. Folding all causes into one frame removes the oracle.

The slice-1 path-allowlist machinery (`_check_paths` / `_within`) is **retained, not removed** — it runs at request time as defense-in-depth on the catalog-resolved `abs_path` (ADR-015 §2 step 3; the §4 end-state that drops it is explicitly deferred until its preconditions ship with negative-test coverage — MEDIUM-3).

## Four signals

- **Who** — the **LLM agent** at request time supplies reference values (`model`, `loras[].name`, `transformer`, cascade stages) and reads `resolved_params` + `notices`. Same-uid stdio trust boundary (MCP child process). The **operator** at spawn time still configures `--model-base` / `--catalog` / `--default-model` (unchanged by this slice except the default-model resolution note in OQ-D).
- **Data** — *read*: the in-memory catalog (`normalize_name → {abs_path, kind, model_family?, source}`) built at spawn; *resolved at request time*: a reference name → its server-side `abs_path` (the load boundary). **`abs_path` must never cross the MCP boundary in either direction** — not in the response, not in any error message, not in any notice text, not in the audit line. The **fine-grained failure cause** must never cross to the agent — stderr audit only. The **agent-supplied raw reference value `R`** must never round-trip into a notice or error message (INFO-2: notice text interpolates the resolved *catalog name* only).
- **Boundary** — `comfyless/mcp_server.py` (`_handle_generate`, and pending OQ-C `_handle_generate_cascade`; the input JSONSchema `_GENERATE_INPUT_SCHEMA`; the response builder); a new request-time **resolver helper** in `comfyless/catalog.py` (`resolve_reference` / equivalent — the catalog module has build + `normalize_name` but no lookup-and-resolve function yet). Out of scope: catalog *build* (slice 2, frozen); `list_*` tools (slices 2/2b, frozen); `extract_params` (slice 4); `iterate` (slice 5); `edit` (slice 6); the CLI `generate` path (still consumes raw paths — the §4 end-state CLI migration is deferred); `output_path`/`savepath` (write destinations under `--output-dir`, not weight references — unchanged, ADR-015 §3).
- **Failure** — **fail-closed**: every reference-resolution failure rejects with the uniform agent-facing error and loads nothing. Never fall back to a stale catalog `abs_path`, never proceed to `_load_pipeline` on a failed resolution. Partial success (some references resolved, one failed → proceed) must be impossible. Audit line on every invocation, success or rejection, with the fine-grained cause on rejection.

## Risk level

**L3 (Red Zone).** This slice changes the **shipped slice-1 `generate` contract** (the agent input/output boundary) and ships the **load-bearing uniform-error contract** that the ADR's whole threat model rests on. Getting the error-uniformity wrong reintroduces the enumeration oracle; leaking an `abs_path` into a response/notice/audit defeats the path-hiding the ADR exists to provide. Runs `code-reviewer` (Opus) **and** `security-auditor` (Opus) before commit, model pinned at invocation per global §5A; security-review output saved to `docs/security/review-slice-3-*-<YYYY-MM-DD>.md` and referenced in the commit body. ADR-015 §2/§3 is the design source of truth.

## Intent

Make weight references on the `generate` (and cascade) MCP surface opaque catalog names in both directions, with one uniform agent-facing error for every resolution failure and the fine-grained cause confined to the operator's stderr audit, so that no absolute path and no enumeration signal ever crosses to the LLM agent — while retaining the slice-1 `_within` allowlist as request-time defense-in-depth on the resolved path.

## Invariants (must always be true)

1. **Reference inputs are catalog names.** For every weight-reference field, the handler reduces a path-shaped value to `os.path.basename(R)`, NFC-normalizes it via the catalog's `normalize_name` (the *same* function slice 2 uses for keys — single canonical implementation, ADR-015 §2 step 1), and looks it up in `cfg.catalog`. A bare name is normalized and looked up directly. The agent-supplied directory component, if any, is **discarded**; the catalog's `abs_path` is authoritative.
2. **Uniform agent-facing resolution error (HIGH-1, load-bearing).** *All* reference-resolution failure causes return the **byte-identical** structured MCP error to the agent — error class and message both fixed — with **no** distinguishing detail. The operator-audit cause set as implemented in the Step-1 resolver (`ResolveCause`) is: catalog miss (`UnknownName`); a catalog *hit* of the wrong kind for the field (`KindMismatch`); malformed/illegal reference (`MalformedReference` — null byte, empty after basename-strip, forbidden char); a catalog hit whose `abs_path` no longer exists at request time (`PathMoved`); and a catalog hit that fails the request-time `_within` re-check (`WithinFailure`). The agent frame is identical across all of them. **Two refinements vs ADR-015 §2's enumerated list, both made in the resolver (flagged for security-auditor):** (a) **`KindMismatch` added** — folding wrong-kind into the uniform frame closes a mild oracle the ADR's five-cause list left open (without it, a lora name supplied as `model` would fall through to `_load_pipeline` and surface as a *different* `InternalError` frame); (b) **request-time `HFCacheMiss` is subsumed by `PathMoved`** — catalog entries store the already-resolved local cache path (build-time HF resolution, slice-2 invariant 6), so a post-spawn cache eviction makes that local path vanish, indistinguishable from any other moved target; the catalog does not retain the originating repo ID, so request-time HF re-resolution would be a no-op on an absolute local path. Build-time HF-cache-miss remains a slice-2 startup failure. The agent-facing uniformity (the load-bearing property) holds identically under both refinements.
3. **Fine-grained cause is operator-only.** The specific cause (one of the five in invariant 2) is written **only** to the stderr audit line, never to any agent-facing frame. (ADR-015 §2 step 2.)
4. **Request-time `_within` fails closed (MEDIUM-1).** The catalog-resolved `abs_path` is re-checked at request time: existence (`os.path.exists` → `PathMoved` on a vanished target) then `_within(--model-base)` (which `realpath`s → `WithinFailure` on an escape). On failure → reject with the invariant-2 uniform error; **never** fall back to the stale catalog path, **never** proceed to load. **Note (security-auditor + code-reviewer, 2026-06-02):** there is **no** request-time `resolve_hf_path` re-resolution — the catalog stores the already-resolved *local* cache path and retains no repo ID, so a post-spawn HF-cache eviction surfaces as `PathMoved` (the local path vanished), subsuming the ADR's separate request-time `HFCacheMiss` label (see invariant 2). The handler in Step 2 must **not** reintroduce a request-time HF re-resolve, or this subsumption breaks.
5. **`abs_path` never crosses the MCP boundary, either direction.** No response field (`resolved_params`, `notices`, `output_path` is a write-dest not a weight ref), no error message, no notice text serializes any catalog entry's `abs_path`. Weight-reference fields in `resolved_params` render as **catalog names**. (ADR-015 §3.)
6. **Path-discard INFO notice (INFO-2).** When a reference resolved from a path-shaped `R` (i.e. a directory component was discarded), the response `notices` array gains an INFO entry: `"reference '<name>' resolved via catalog; supplied path discarded — do not rely on paths for later actions."` The interpolated value is the **resolved catalog name only** — never the agent-supplied raw `R`, which may carry attacker-chosen directory text that must not round-trip into the agent transcript.
7. **Response gains `notices: [{level, message}]`** — optional array, absent or empty when there is nothing to report. Non-path generation parameters in `resolved_params` (`prompt`, `negative_prompt`, `seed`, `steps`, `cfg_scale`, `true_cfg_scale`, `sampler`, `scheduler`, `width`, `height`, `model_family`, LoRA `weight`) are **unchanged**.
8. **Input schema field renames** (ADR-015 §3; OQ-A/OQ-B resolved): `loras[].path` → `loras[].name`; `transformer_path` → `transformer` (catalog name, `kind:"transformer"` lookup); `model` keeps its key (already clean). **`vae_path` / `text_encoder_path` / `text_encoder_2_path` are DROPPED from the MCP `generate` schema** — they have no catalog kind and a raw-path carve-out would reintroduce the input attack surface ADR-015 removes (CLI retains them; future slice may add `vae`/`text_encoder` catalog kinds if a real MCP use case emerges). The JSONSchema `description` for each migrated field states it is a catalog name discoverable via `list_models` / `list_transformers`, and that a path-shaped value has its directory discarded and basename resolved. `additionalProperties:False` stays — the old `loras[].path`, `transformer_path`, `vae_path`, `text_encoder_path`, `text_encoder_2_path` keys now fail validation (intended contract break).
9. **Slice-1 path-allowlist retained as defense-in-depth.** `_check_paths` / `_within` are not deleted; they run on the resolved `abs_path` per invariant 4. The §4 end-state that promotes the catalog to sole authority and drops `--model-base` is explicitly NOT in this slice (MEDIUM-3 preconditions).
10. **Audit on every invocation**, success and rejection alike — one stderr line, never stdout. On rejection it carries the fine-grained cause (invariant 3). It does **not** carry any catalog entry's `abs_path` (carry-forward of slice-2 invariant 10 / slice-1 invariant 5).
11. **Traceback strip carries forward.** Any internal exception in resolution or load is caught, full-traceback'd to stderr, and converted to a sanitized MCP error via the existing `_sanitize_error` / `_MCPHandlerError` pattern. No traceback, no `.py:line`, no absolute path crosses the boundary. (Carry-forward slice-1 invariant 13 / slice-2 invariant 11.)
12. **stdout carries only MCP JSON-RPC frames.** (Carry-forward.)
13. **No argparse / no CLI dispatch** added. (Carry-forward slice-1 invariant 14 / slice-2 invariant 13.)
14. **`--default-model` stays an operator-trusted path (OQ-D resolved).** It remains an operator-supplied abs path on `_StartupConfig`, `_within`-checked at startup, used when the agent omits `model`. It **bypasses** the agent-facing catalog name resolver (it is operator config, not agent input) but still passes the request-time `_within` check (invariant 4) and the existing default-model-escape check (handler step 3). The agent-facing failure on a now-invalid default still obeys invariants 2–4 (uniform error, no `abs_path` leak).
15. **The resolver is a pure, reusable catalog function.** The request-time name→abs_path resolution + basename-strip + normalize + the five failure-cause discrimination live in `comfyless/catalog.py` (so slice 4 `extract_params` reverse-lookup and slice 5 `iterate` reuse one implementation), returning a typed result that distinguishes hit / each failure cause for the *handler* to map onto the uniform error + audit cause. The catalog module never raises an agent-facing string; it returns causes, the handler renders the uniform frame.

## Failure semantics

- **Fail-closed at request time** on every reference-resolution failure (invariant 2). Reject with the uniform error, load nothing, audit the fine-grained cause. No partial resolution proceeds.
- **Never fall back** to a stale/escaped catalog `abs_path` (invariant 4).
- **Traceback-strip** on any unexpected internal exception (invariant 11).
- **Audit-line write failure** does not block the response (mirrors slice-1/2; increments `_audit_write_failures`).

## Out of scope (explicit)

- **Catalog build** — slice 2, frozen. Slice 3 only *consumes* the catalog and adds a request-time resolver helper.
- **`list_models` / `list_loras` / `list_transformers`** — slices 2/2b, frozen.
- **Cascade migration** (`_handle_generate_cascade`: `cascade_config.stage_c/b/a` → `kind:"model"` names; `scaffolding_repo` kind decision) — **new slice 3b** (OQ-C resolved). Slice 3 leaves the cascade handler on its slice-1 raw-path contract verbatim.
- **`extract_params`** (slice 4), **`iterate`** (slice 5), **`edit`** (slice 6).
- **The CLI `generate` path** — still consumes raw paths. The ADR-015 §4 end-state (catalog as sole authority, CLI routed through it, `--model-base` + request-time `_within` dropped) is a future amendment gated on MEDIUM-3 preconditions; this slice keeps `_within` as the safety net.
- **`output_path` / `savepath`** — write destinations under `--output-dir`, not weight references; unchanged (ADR-015 §3 / deferred list).
- **New catalog kinds** (`vae`, `text_encoder`, `text_encoder_2`) — not minted by slice 2's build; their MCP-field disposition is OQ-A (drop vs raw-path carve-out), not "add a new kind" (that is a future slice if a real use case emerges).
- **Fuzzy / alias name resolution** — v1 is exact-name (+ basename-of-path), ADR-015 deferred list.
- **Hot-reload of the catalog** — ADR-015 deferred polish.

## Negative cases (required)

**Uniform-error contract (the load-bearing tests):**

- **N1** — `model` = a name **not** in the catalog → uniform `"reference not available"` error; stderr audit cause = `UnknownName`.
- **N2** — `model` = a name that **is** in the catalog but whose `abs_path` no longer exists / fails request-time `_within` (fixture: catalog entry pointing at a path removed/moved post-spawn) → **same** uniform error frame as N1; stderr cause = `PathMoved` or `WithinFailure`. Asserts no fallback to the stale path and no load.
- **N3** — `model` = a name that exists in the catalog as a **different kind** (e.g. a `kind:"lora"` name supplied in the `model` field) → **same** uniform error frame as N1; stderr cause = `KindMismatch`. (Proves the kind-oracle closure — a lora name and a nonexistent name are indistinguishable to the agent.) **Plus the oracle-closure equivalence:** a catalog entry whose backing path is absent at request time (the post-spawn HF-cache-eviction / `PathMoved` case) yields the **same** frame as N1 — byte-identical from the agent's view.
- **N4** — `model` containing a null byte (and: empty after basename-strip e.g. `"/foo/bar/"`; a `_FORBIDDEN_NAME_CHARS` codepoint) → **same** uniform error frame as N1; stderr cause = `MalformedReference`.
- **N5 (the keystone assertion)** — N1–N4 produce **byte-identical** agent-facing error class AND message. A single test compares the frames for equality. (Proves HIGH-1.)
- **N6** — For each of N1–N4, assert the corresponding stderr audit line **does** carry the distinct fine-grained cause (proves the cause is preserved for the operator, just not for the agent).

**Resolution + notice behavior:**

- **N7** — `model` = a bare catalog name (no separator) that exists → resolves; response `resolved_params.model` is the **catalog name**; `notices` has **no** path-discard entry (nothing was discarded).
- **N8** — `model` = a path-shaped value (`/some/dir/<name>`) whose basename is a catalog name → resolves via basename; `resolved_params.model` = the catalog name; `notices` contains the INFO path-discard entry.
- **N9 (INFO-2)** — The N8 notice message contains the resolved **catalog name** and does **NOT** contain the agent-supplied directory text (`/some/dir/`) anywhere. Assert the raw `R` substring is absent from the entire response JSON.
- **N10** — `loras[].name` (new key) resolves; `resolved_params.loras[].name` renders the catalog name; old `loras[].path` key → schema validation rejects (additionalProperties / required-key contract break).

**`abs_path`-never-leaks cases:**

- **N11** — A successful `generate` response: assert no `abs_path` / no absolute filesystem string (`/home/`, `/mnt/`, `--model-base` prefix) appears anywhere in `resolved_params`, `notices`, or any error.
- **N12** — Every uniform-error frame (N1–N4): assert no absolute path and no fine-grained cause string appears in the agent-facing message.
- **N13** — Audit lines for N1–N4: assert no catalog `abs_path` appears (carry-forward of slice-2 invariant 10).

**Cascade not-touched (OQ-C resolved → slice 3b):**

- **N14** — `_handle_generate_cascade` still consumes its slice-1 raw-path contract verbatim: its `cascade_config` schema and resolution behavior are **unchanged** by slice 3 (the cascade migration is slice 3b). Asserts slice 3 did not partially migrate cascade.

**Regression / carry-forward:**

- **N15** — `tools/list` still advertises the four tools (`generate`, `list_models`, `list_loras`, `list_transformers`); only `generate`'s `inputSchema`/`description` changed (the migration), the three `list_*` tools are byte-identical to slice 2b.
- **N16** — Traceback-strip: force an internal exception in the resolver → MCP error frame has no `Traceback`, no `.py:<digits>`, no `/home/`-prefixed path; full traceback on stderr.
- **N17** — Static-source check: no `import argparse` added; resolver lives in `comfyless/catalog.py`.

## Proof hooks

- **Positive:** `./.venv/bin/python3 test_mcp_server.py` — new sections cover name resolution, basename-strip, the path-discard notice, and a successful generate returning names.
- **Negatives N1–N17** organized as sections inside `test_mcp_server.py` (no pytest dep; `python3 test_<name>.py` invocation via `./.venv/bin/python3` per ADR-013).
- **The keystone N5** (four-frame byte-equality) and **N3** (HF-miss == unknown-name frame) are the tests that prove the oracle closure; they gate the `TECH_DEBT.md` 2026-05-17 resolution mark.
- **Existing suites continue to pass.** `test_mcp_server.py`'s generate-contract tests are **rewritten** for the names contract (the slice-1 path-based generate tests assert the old contract and must migrate); the other 8 suites are untouched and stay green. Final count updated in CLAUDE.md's suite-count line at closure.

## Red Zone ownership

- **The uniform-error contract** (all five causes → one agent frame; fine-grained cause stderr-only): owned by **Grant** — AI-generated only. The four-frame byte-equality test (N5) is the proof he signs off.
- **The `abs_path`-never-crosses-the-boundary guarantee** across response + notices + errors + audit: owned by **Grant**.
- **The request-time `_within` fail-closed-no-fallback rule** (invariant 4): owned by **Grant**.
- **The notice-text sanitization** (catalog name only, never raw `R` — INFO-2): owned by **Grant**.
- **ADR-015 §2/§3 is the design source of truth.** Any divergence reverts to an ADR amendment before code lands.

## Open questions — RESOLVED 2026-06-02

**OQ-A — Component override fields → RESOLVED: `transformer_path` → catalog name (`transformer`); DROP `vae_path` / `text_encoder_path` / `text_encoder_2_path` from the MCP `generate` schema.** Rationale: a raw-path carve-out reintroduces exactly the input attack surface ADR-015 exists to remove. They are power-user fields not exercised end-to-end on the MCP surface; the CLI retains them. A future slice may add `vae` / `text_encoder` catalog kinds if a real MCP use case emerges. Slice 3's surface is "names only, no raw paths." (Invariant 8, N10-adjacent schema-rejection cases.)

**OQ-B — Rename set → RESOLVED:** `loras[].path` → `loras[].name`; `transformer_path` → `transformer`; `model` unchanged; the three dropped component fields per OQ-A. Deliberate contract break — old keys fail `additionalProperties:False`. (Invariant 8.)

**OQ-C — Cascade scope → RESOLVED: non-cascade `generate` only in slice 3; cascade migration is new slice 3b.** `_handle_generate_cascade` keeps its slice-1 raw-path contract verbatim; slice 3b decides `scaffolding_repo`'s kind and migrates `stage_*` → `kind:"model"` names. Mirrors the slice-2 → 2b split; keeps the load-bearing uniform-error keystone focused on the common path. (Out-of-scope list; N14.)

**OQ-D — `--default-model` → RESOLVED: stays an operator-trusted server-side path.** Bypasses the agent-facing catalog name resolver (operator config, not agent input) but still passes request-time `_within` and the existing default-model-escape check. (Invariant 14.)

## Slice plan (cadence resolved 2026-06-02 — multi-step, reviewed per step)

Mirrors slice-2 discipline: each step is independently committable with its own review gate.

| Step | Scope | Review |
|---|---|---|
| 1 | **Request-time resolver in `comfyless/catalog.py`** — pure function: basename-strip → `normalize_name` → catalog lookup → request-time `realpath`+`_within` (+ HF re-resolve for HF-sourced entries), returning a typed result that distinguishes **hit** (with `abs_path`, resolved name, `path_was_discarded` flag) from each of the five failure causes. No agent-facing strings in the module — it returns causes. Unit tests in `test_mcp_server.py` (or a catalog test section). | code-reviewer + security-auditor (it is the new resolution trust point). |
| 2 | **Migrate `_handle_generate`** — input field rename/drop (invariant 8); route `model` / `transformer` / `loras[].name` through the Step-1 resolver; map every failure cause onto the **uniform** `"reference not available"` error (invariant 2) with fine-grained cause to the audit line (invariant 3); render `resolved_params` weight fields as names + add `notices` with the path-discard INFO (invariants 5–7). `--default-model` per invariant 14. | code-reviewer + security-auditor (changes the shipped contract + ships HIGH-1). |
| 3 | **Tests** — N1–N17 as sections in `test_mcp_server.py`; **rewrite** the slice-1 path-based generate tests to the names contract; keystone N5 (four-frame byte-equality) + N3 (HF-miss == unknown-name). Full suite green. | code-reviewer. |
| 4 | **Closure** — mark `TECH_DEBT.md` Security 2026-05-17 **Resolved** (oracle closed by N3/N5); ADR-015 Changelog "slice 3 IMPLEMENTED" entry; CLAUDE.md suite-count line update; Backlog + vault mirrors. | docs-only (Sonnet-delegable per §13). |
