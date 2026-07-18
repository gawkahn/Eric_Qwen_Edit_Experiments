# Slice DLW — daemon LoRA weight application — Vision

**Date:** 2026-07-17 · **Author:** Claude (Fable 5) · **Approved:** Grant ("hold for daemon-side fix - and do that now", 2026-07-17)
**Context:** TECH_DEBT 2026-07-17 "Daemon LoRA path misses weight application";
code-review finding 2 on the `_apply_loras` weight fix (`1f52672`). Red Zone:
`comfyless/server.py` (ADR-001 daemon trust model; LoRA-diff semantics from
`docs/security/review-slice-DQ-daemon-quant-2026-07-03.md` F2/F3/F4).

## 1. Vision

**Outcome when done:** daemon-served generations honor the requested LoRA
weight exactly like the CLI/MCP path: PEFT-backed adapters scaled by ONE
cumulative `set_adapters` call over the full active set after the LoRA
diff; weight-only changes on an already-loaded LoRA take effect on the
unquantized incremental path (today they are silently ignored — the diff
keys on path alone).

**Invariants (each with at least one negative test):**

1. **Shared application logic, not a fork** — the cumulative weight
   application (direct-merge exclusion via `is_direct_merge_adapter`,
   Kohya `_te` halves at the parent's weight, warn-don't-block on
   failure) is ONE function used by both `_apply_loras` and the daemon;
   behavior of the CLI path is unchanged (existing test_apply_loras
   suite stays green, unmodified assertions).
2. **Weight-only change applies (unquantized path)** — same path, new
   weight → no reload; the record updates and the cumulative call
   applies the new scale. NEGATIVE: old behavior (ignored change) must
   fail the new test.
3. **Direct-merge adapters keep their baked weight, loudly** — a
   weight-only change on a direct-merge adapter cannot take effect
   without a reload; the daemon warns (response `warnings` +
   log) and keeps serving (warn-don't-block). It must NOT silently
   pretend the new weight applied.
4. **Quantized-path semantics unchanged** — LoRA (path, weight) stays in
   the cache key; any change still evicts + reloads (DQ F2/F3/F4
   posture untouched); the fresh loads then get the cumulative weight
   call like everything else.
5. **No trust-model change** — no new request fields, no path-validation
   changes, no socket-protocol changes. The diff touches only the LoRA
   add-loop and the post-diff weight application.

**Out of scope:** node-stacker singleton fix (own TECH_DEBT entry);
client→server pause protocol; any cache-key change.

## 2. Change boundary / edit scope (hard)

May change: `comfyless/generate.py` (extract the shared
`apply_adapter_weights` helper from `_apply_loras` — mechanical,
behavior-preserving), `comfyless/server.py` (**Red Zone** — LoRA add-loop
+ post-diff weight call), `test_apply_loras.py` (helper-level tests),
`test_server_robustness.py` (daemon-side tests), `TECH_DEBT.md` (entry
Resolved), this doc, `docs/security/review-slice-DLW-*.md` (new).
Anything else → STOP and split.

## 3. Design (condensed)

- `generate.apply_adapter_weights(pipe, pairs, log_fn)` — the post-loop
  block of `_apply_loras` verbatim (filter direct-merge → append `_te`
  halves → one `set_adapters` → warn on failure); returns the warning
  string or None so the daemon can surface it in response `warnings`.
- Server add-loop: `continue` only when path loaded AND weight equal;
  path loaded + weight differs → PEFT adapter: update `loaded_loras`
  record; direct-merge: loud warning, record keeps baked weight.
- After the add-loop: `apply_adapter_weights(pipe, [(rec["adapter_name"],
  rec["weight"]) for rec in loaded_loras])` — cumulative over the full
  active set, so removals re-pin survivors too. Idempotent across
  requests.

## 4. Proof

`just tests` green; new negatives: weight-change-ignored (old behavior)
fails; direct-merge weight-change warns; cumulative call covers
survivors after a removal. security-auditor (Fable) delta review on the
diff (existing surface, DQ posture) saved to docs/security/; code-reviewer
(Fable) on the whole diff.

AI-Disclosure: Claude (Fable 5) authored; Grant approved the build direction 2026-07-17.
