# Security Review — `scripts/lora_audit.py` S2 (`--dry-load` mode)

**Date:** 2026-05-28
**Slice:** S2 of ADR-014 (`docs/decisions/ADR-014-lora-audit-tool.md`)
**Reviewer:** `security-auditor` agent (model: opus, per `feedback_agent_model_pin_broken`)
**Branch:** `lora-convert-scripts` (worktree `/home/gawkahn/projects/ai-lab/code/eric-lora-convert`)
**Diff scope:** `scripts/lora_audit.py`, `test_lora_audit.py`, NEW `test_dry_load_integration.py`
**AI-Disclosure:** Claude (Opus 4.7) authored review; Grant reviewed.

---

## Verdict

**CLEAN** after fold. Initial review flagged 1 HIGH (stdout-contract leakage via
unredirected `diffusers.AutoPipelineForText2Image.from_pretrained`), 1 MEDIUM
(symlink TOCTOU between scan and dry-load reopen), and 2 LOW findings. All
HIGH + MEDIUM and 1 LOW have been folded; the remaining LOW (unbounded
`StringIO` capture per file) is recorded in `TECH_DEBT.md` under Security.
Parallel `code-reviewer` (Opus) raised 1 HIGH (`applied_modules` non-null on
`loaded=False`) which has also been folded.

## Threat Model

S1 was read-only / header-only. S2 adds a **deserialization-and-VRAM** surface:
caller-supplied weight files under `--audit-root` are now passed to
`diffusers.pipe.load_lora_weights()`, which dispatches to
`safetensors.load_file` (mmap, no code execution) for `.safetensors` and to
`torch.load(weights_only=True)` for `.pt/.bin/.pth`. The `weights_only=True`
guard is gated on `torch >= 1.13` (`diffusers/models/model_loading_utils.py:182`);
the project's pinned torch satisfies this. **Pickle code-execution path is
closed at the currently-pinned versions.**

Same-uid attacker capabilities under `audit_root` extend from "force a
classification-error in S1" to "force a dry-load attempt with attacker-supplied
weight content in S2." Path containment (realpath + `relative_to(audit_root)`)
is preserved; the new TOCTOU window between scan and dry-load is narrowed by
the M-1 fold below.

## Findings (folded)

### HIGH H-1 (security) — `--print-manifest` stdout contract breach **— FOLDED**

`_load_dry_load_pipeline` invokes diffusers' `from_pretrained`, which prints
warnings (`"Some weights of … were not used …"`, accelerate device-map
messages) to stdout. The call was placed **outside** the existing
`contextlib.redirect_stdout(io.StringIO())` block, so when callers pass
`--print-manifest` (Vision §13 machine-caller contract: stdout is the manifest
JSON, nothing else), the from_pretrained chatter mixes with the manifest JSON
and breaks parser compatibility.

**Fix:** wrapped the `_run_dry_load` call in `scripts/lora_audit.py:main` with
`with contextlib.redirect_stdout(io.StringIO())`. The per-call buffer used by
`_dry_load_per_base` for `applied=(\d+)` parsing is unaffected (it nests
correctly under the outer redirect).

**Test added:** `test_dry_load_print_manifest_stdout_clean` —
patches `_load_dry_load_pipeline` and `load_lora_with_key_fix` to print
realistic-looking from_pretrained banner output, runs with
`--dry-load --print-manifest`, asserts captured stdout is parseable JSON
end-to-end.

### HIGH H-1 (code-review) — `applied_modules` on `loaded=False` **— FOLDED**

The direct-merge loader paths print `applied=N, skipped=M` AND return
`applied > 0`. When `applied == 0`, the loader prints `applied=0` and returns
`False`. The audit code unconditionally parsed the count and wrote
`{loaded: False, applied_modules: 0, reason: None}` — downstream readers can't
distinguish "loader ran cleanly and merged nothing" from "loader returned
False as a signal."

**Fix:** `applied_modules = applied_raw if loaded else None`. The downstream
catalog now reads `applied_modules` as **"what got merged on success"**;
`null` means no success path was taken.

**Test added:** `test_dry_load_loaded_false_nulls_applied_modules` —
loader returns False with `applied=0` print, asserts manifest shape is
`{loaded: false, applied_modules: null, reason: null}`.

### MEDIUM M-1 — Symlink TOCTOU scan→dry-load **— FOLDED**

`_passes_scan_containment` + `_open_no_follow` validated the realpath **at
scan time**. The dry-load loop reopens the file by name seconds-to-hours later;
a same-uid attacker can swap the file (or a containing directory's terminal
component) to a symlink whose realpath escapes `audit_root` in the gap.
Diffusers' loader would then read whatever the new realpath points to. No
code execution (weights_only=True), but the attacker can force a parse error
that may surface attacker-controlled bytes in `dry_load.reason`, or in the
pathological case where the substituted file is itself a valid LoRA, alter
`dry_load.loaded` / `applied_modules` for that entry.

**Fix:** before each `load_lora_with_key_fix` call, re-run
`_passes_scan_containment(abs_path, audit_root, recheck_warnings)`. On
failure: record `{loaded: false, applied_modules: null, reason:
"containment_changed"}` and skip the load. The inner TOCTOU between the
recheck's realpath and the loader's `open()` is the same same-uid residual
ADR §6 already accepts. This is the same Option-B narrowing pattern S4 will
use for `--delete` (ADR §9 F-5).

**Test added:** `test_dry_load_containment_recheck` — monkey-patches `_scan`
to swap `second.safetensors` to a symlink-to-outside between scan and
dry-load; asserts swapped file records `containment_changed` while the
non-swapped sibling loads normally.

### LOW L-1 — `local_files_only=True` enforcement future-proofing **— FOLDED**

`_load_dry_load_pipeline` hardcodes `local_files_only=True` positionally. A
future refactor that adds `**kwargs` could shadow it.

**Fix:** docstring update at `_load_dry_load_pipeline` explicitly
forbids `**kwargs` passthrough and caller-supplied `local_files_only`
arguments; the comment names the regression risk so future readers see the
intent.

### LOW L-2 — Unbounded `StringIO` buffer per dry-load file **— DEFERRED**

`_dry_load_per_base` captures the loader's stdout in an `io.StringIO()` per
file. No exploit at the currently-pinned loader (no print-loop path), but a
future loader regression could OOM the audit process. Recorded in
`TECH_DEBT.md` under Security with fix shape and trigger.

### LOW L-3 — `test_dry_load_integration.py` SKIP exits 0 **— DEFERRED**

CI doesn't exist for this repo today. When CI lands, the SKIP-exit-0 pattern
silently green-lights runs that should have been gated. Recorded in
`TECH_DEBT.md` under Security; fix is a 2-line change paired with the CI
config.

## INFO (verified clean, no action)

- **I-1 — `_parse_applied_modules` extracts only the count.** Captured stdout
  is discarded after regex match; nothing else from the buffer reaches the
  manifest. No path-info / key-name leakage via dry_load.
- **I-2 — TOML `[defaults] dry_load = true` promotion-only.** CLI `--dry-load`
  presence is checked first and short-circuits the config check. Config can
  only promote off→on; never demote on→off. Matches ADR §7.
- **I-3 — Per-LoRA fault isolation uses `except Exception`.** Preserves
  `KeyboardInterrupt` and `SystemExit` pass-through per Vision invariant 9 +
  global §0 rule 2.
- **I-4 — S1 boundary functions untouched.** `_passes_scan_containment`,
  `_open_no_follow`, `_validate_output_dir`, `_redact_argv`, `_resolve_bases`,
  `_classify_one` are unchanged from S1. No scope creep into security-truth
  surfaces.
- **I-5 — F-3 VRAM-cascade WARN.** Fires only on second-and-subsequent base
  failure. Test `test_dry_load_vram_cascade_warning` covers the 2-base case.
- **I-6 — Diffusers/torch deserialization safety is version-dependent.** The
  `weights_only=True` kwarg is gated on `torch >= 1.13`. Project's pinned
  torch satisfies this. **Flag for the next dep-bump review:** confirm
  `weights_only=True` survives any diffusers/torch upgrade.
- **I-7 — Convert / Delete rejections intact.** `--convert` and
  `--delete/--yes` still return EXIT_STARTUP_FAIL with the S2-specific
  message. No accidental enablement.

## Tests added (review-fold coverage)

| Test | Finding |
|---|---|
| `test_dry_load_print_manifest_stdout_clean` | HIGH H-1 (security) |
| `test_dry_load_loaded_false_nulls_applied_modules` | HIGH H-1 (code-review) |
| `test_dry_load_containment_recheck` | MEDIUM M-1 (security) |

90 / 0 with the augmented suite: 40 S1 + 32 S2-spec + 18 review-fold.

## Cross-references

- ADR-014 §7 (dry-load mechanism): `docs/decisions/ADR-014-lora-audit-tool.md:319-337`
- ADR-014 §15 row S2 (slice plan + reviewer cadence): `docs/decisions/ADR-014-lora-audit-tool.md:455-456`
- S1 security review (CLEAN baseline): `docs/security/review-lora-audit-s1-2026-05-25.md`
- TECH_DEBT entries from this review: `TECH_DEBT.md` § Security (L-2 stdout buffer, L-3 SKIP-exit)
- Project review-bar rules: `CLAUDE.md` § Review bar
