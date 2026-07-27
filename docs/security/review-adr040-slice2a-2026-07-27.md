# Security review — ADR-040 slice 2a (code): `run_id` correlation primitive

AI-Disclosure: `security-auditor` and `code-reviewer` subagents (both Claude
Opus 5, invoked with an explicit `model: "opus"` override per Grant's
2026-07-27 ruling) authored the findings; Claude (Opus 5) drove the session and
folded them; Grant reviewed.

**Date:** 2026-07-27
**Subject:** `comfyless/generate.py`, `comfyless/refine.py`, `comfyless/cascade.py` (+ tests)
**Trigger:** Red Zone — `comfyless/refine.py` is a `_red-zone-paths.sh` path
(LLM output influencing generation params).
**Chain:** `review-adr040-revision-2026-07-27.md` (design) →
`review-adr040-slice1-2026-07-27.md` (slice 1) → this.
**Status:** all findings folded; slice committed.

## What the slice does

Introduces `mint_run_id()` (`uuid4().hex[:8]`) as the single minting helper;
mints one `run_id` per invocation in `generate`'s CLI path, `refine`, and
`cascade`; stamps it client-side into every on-disk record those runs write;
registers `run_id` + `iterate_batch_id` in `_SKIP_SIDECAR_KEYS` and `run_id` in
cascade's `_KNOWN_KEYS`.

## Threat model

(a) Untrusted LLM output (judge/planner) must not gain a new field it can read
or write; (b) caller-supplied sidecars, PNG chunks, cascade config files, and
MCP `cascade_config` are untrusted input crossing into records and agent-facing
responses; (c) the correlation id must not become a path, a replayable param, or
an agent-visible runtime detail; (d) slice 2b will use `run_id` in a directory
name, so the minting site's ordering relative to filesystem ops matters.

**Verdict: sound, no exploitable vulnerability.** One real defect (below), the
rest test-strength and promise-precision.

## Findings folded

### MEDIUM — `run_id` was registered as a known cascade key but never minted there

**Both reviewers caught this independently.** `validate_config` does not strip
unknown keys — it copies them (`cfg = dict(raw)`) and only warns. Dispatch then
builds `sidecar = dict(cfg)`. So a `run_id` in an operator's cascade config, or
in an **agent-supplied `cascade_config` over MCP** (which reaches
`validate_config` directly and flows into both the on-disk PNG metadata and the
agent response), would have been carried as authoritative run provenance —
and registering the key had simultaneously removed the `unknown keys ignored`
stderr audit line that previously flagged it, making the inheritance silent.

The contrast makes the rule explicit: `iterate_batch_id` is safe only because
the sidecar `update()` overwrites it unconditionally; `output_format`/`quality`
are safe only because they are explicitly popped, with a comment naming this
exact hazard. `run_id` was the one `_KNOWN_KEYS` provenance key that was
neither.

Latent today (nothing emitted a cascade `run_id`), live the moment cascade
minted one — which the ADR requires.

**Closed:** cascade mints its own beside `iterate_batch_id` and writes it into
the `update()` block, so it overwrites any inherited value. Completes D1b's
"every entrypoint". A behavioral test in `test_cascade.py` now carries a
FOREIGN `run_id` through the round-trip fixture and asserts every provenance key
in the update block is authoritative for this run.

### MEDIUM — `run_id: Optional[str] = None` was a weaker promise than the ADR states

D1b says `run_id` appears on **every** record. With an `Optional[...] = None`
default on `refine_loop` and `verdict_record` and an unconditional stamp, what
was actually delivered was "every record carries the *key*". Any caller omitting
it writes `"run_id": null` into every candidate sidecar and verdict — and since
the id's whole purpose is equality-grouping, that **collapses every unset run
into one bucket**, which is worse than a missing key. The two entrypoints also
disagreed: `generate.py` guards with `if run_id:`, refine did not.

**Closed:** made required and keyword-only on both, so pyright enforces it at
every call site rather than trusting each caller. The six test harnesses default
it explicitly.

### LOW — a source-count assertion that could not catch what it claimed

`_rid_src.count("_stamped(run_generation(") == 2` pinned today's state, not the
invariant: adding a **third, unstamped** call site leaves the wrapped count at 2
and passes green — precisely the "one path silently ends up without a
correlation id" failure the choke point exists to prevent. It was also brittle
to a pure reflow.

**Closed structurally rather than by a better count.** The `_stamped()` closure
was replaced with a single assignment after both paths join:

```
try:    out = run_generation(...)
except RefRefusedError:  ...;  out = run_generation(...)
out.metadata["run_id"] = run_id
return out
```

One assignment, one return — the choke point is now structural instead of a
comment claiming two sites were both remembered, and there is nothing left to
count.

### LOW — three tests weaker than they read

1. **A test asserting a property of its own literal.** `"/" not in run_id` on a
   value the test supplied: unfailable. It also named the wrong gate —
   `_assert_no_paths` is **key**-based, so a `/`-bearing value would pass it
   regardless. Replaced with the property that actually matters: `run_id` never
   enters the judge or duel payloads.
2. **Source-greps where behavioral harnesses already existed in the same file.**
   `_run_loop`, `_run_loop_p`, `_run_loop_h`, and `_run_loop_e(refuse_daemon=True)`
   all drive the real `refine_loop` through the real `_write_json`. Replaced
   with on-disk assertions on both `_generate_one` paths — normal and the
   RefRefused in-process fallback.
3. **A "end-to-end" test that reimplemented the filter it tested.**
   `test_params_schema.py` built its own comprehension over
   `_SKIP_SIDECAR_KEYS`, so deleting the real filter line in `_load_sidecar`
   would have left it green. Now writes a temp sidecar and calls `_load_sidecar`.
   Its negative control (an unknown key still warns) was the strongest thing in
   the original diff and is kept as-is.

### LOW — half the slice had no coverage

Both `generate.py` stamps (delegated and in-process) were untested, and nothing
asserted the actual correlation promise. **Closed:** a delegated-path test in
`test_ref_edit.py` drives the real `_delegate_to_server` against a stubbed
daemon and reads the sidecar off disk, plus a sweep-level test asserting N
sidecars share one `run_id` while carrying a separately-minted
`iterate_batch_id` — the "distinct fields, one helper" claim, previously
untested.

### LOW — the second mandatory registration duty was missing

D1b names two registrations; only `_SKIP_SIDECAR_KEYS` had landed.
**Closed:** `docs/vision/slice-4-mcp-extract-params.md` item 20 now names
`run_id`, and `test_mcp_server.py` has the ADR's named negative test. Both
reviewers independently verified the guarantee is **structural** — the
non-cascade branch normalizes through `_validate_params` (dropping every
non-`COMFYLESS_SCHEMA` key) and the cascade branch renders through a positive
allowlist — so the doc entry records an existing property rather than providing
one. It belongs in this commit because this slice is what makes a sidecar
legally able to carry the key.

## Verified clean

- **`run_id` cannot reach judge or planner context.** Every LLM payload builder
  (`build_judge_user_text`, `build_duel_user_text`, `history_record`,
  `history_error_record`) is constructed field-by-field and never touches
  `outcome.metadata`. Now pinned by test.
- **A caller cannot influence refine's `run_id`** — a seed image's `run_id` is
  dropped twice inside `gen._load_params`.
- **No aliasing hazard** in the in-place `out.metadata` mutation: the daemon
  path's dict is freshly `json.loads`-ed, and the cold path's is local to
  `generate()` with the PNG already written before return.
- **`_SKIP_SIDECAR_KEYS` weakens nothing.** Both consumers (`_load_sidecar`, the
  PNG chunk branch) are drop-then-`_validate_params`; no key in that set is ever
  *trusted*, only unwarned.
- **Nothing parses `iterate_batch_id`**, so shortening it to 8 hex is safe;
  values are compared for equality to group records.
- **`run_id` never reaches the PNG tEXt chunk** on any path — sidecar only,
  matching the existing `iterate_batch_id` behavior. Slice 2b must not assume a
  stray PNG can be correlated without its sidecar.
- **Mint placement sets 2b up correctly**: `run_id` is minted after every
  `return 2` validation path (so a rejected invocation mints nothing) and before
  the first filesystem operation on the output dir — which is where D1's
  exclusive create has to go.
- **Scope clean:** three source files, five test files, all within D1b. The
  `import uuid` removal in cascade.py is completion of this change (no remaining
  reference), not drive-by.
- The verdict-key allowlist in `test_refine.py` is still an exact-equality
  check, so admitting `run_id` forced the decision through review rather than
  silencing the control.

## Carried into slice 2b

- **`run_id` is a 32-bit non-secret and must never acquire capability
  semantics.** It is guessable in ~2^31 tries and appears in logs. If the
  derived run dir ever lands in a daemon root shared with another principal, the
  `exist_ok=False` exclusive create — not the id — must remain the guard.
- **`refine.py`'s unconditional `os.makedirs(args.output_dir, exist_ok=True)` is
  exactly what the design re-review warned would silence the assertion.** 2b
  must *replace* it for the derived case, not add the exclusive create beside it.
