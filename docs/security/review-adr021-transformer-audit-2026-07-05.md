# Security review — ADR-021 transformer audit (design phase)

**Date:** 2026-07-05 · **Reviewer:** security-auditor (Opus, model pinned at invocation) · **Round 1 verdict: CHANGES REQUIRED** (1 HIGH / 3 MED / 2 LOW / 2 INFO — all folded same-day, see ADR-021 Changelog). Round 2 appended below when complete.

**AI-Disclosure:** Claude (Opus, security-auditor subagent) authored the review; Claude (Fable 5) drove the slice; Grant reviewed.

---

## Round 1 (verbatim)

### Summary

ADR-021 extends the accepted, shipped `scripts/lora_audit.py` (ADR-014) with `kind:"transformer"` manifest entries: a repeatable `--transformer-root`, header-only loading-prognosis reuse via `audit_single_files.py`, shape-fingerprint matching against configured bases, and bounded sampled-content duplicate detection. The threat model is the same-trust-zone MVP inherited from ADR-014 §6: an operator CLI over caller-supplied directory trees, no LLM/remote callers yet. The design correctly declares **no new write/delete surface** — the only genuinely new capability is bounded content reads.

Overall posture is strong and consistent with ADR-014's hard-won containment discipline. The reused `audit_single_files._read_tiny_tensor` already carries the `max(0, min(hi-lo, cap))` hostile-offset guard, and the fail-toward-inclusion choice on duplicate read-error is the correct direction for a catalog/audit tool (never silently drop a real model). The findings below are dominated by **absence** items: the design does not guard against overlapping roots (which silently defeats the "no deletes under a transformer root" hard invariant when a transformer root is nested under the LoRA audit root), uses a collision-prone `root` basename as the manifest discriminator, and leans on a forward-compat contract ("ignore unknown keys") that is weaker than what invariant-5 additivity actually requires. The new 1-MiB sampled-read path is described but its hostile-header guards are left implicit rather than specified.

### Findings

**[HIGH] F-1 — Nested roots defeat the "no deletes under a transformer root" hard invariant**
The design roots each `--transformer-root`'s containment "against its own root" but never asserts the transformer roots are **disjoint from `--audit-root`**. `--audit-root` is required and always drives the LoRA scan+delete. If an operator points `--transformer-root` at a subtree of `--audit-root` (e.g. `--audit-root /models --transformer-root /models/checkpoints`), the LoRA `rglob` walks the transformer files too, a garbage transformer file (zero-byte / truncated / unparseable header) is classified `deletable` **as a LoRA**, and `--delete --yes` unlinks it — an irreversible delete under a transformer root, which Vision invariant 7 declares must never happen. The same file also mints both a `kind:"lora"` and a `kind:"transformer"` entry. Silent invariant violation with data-loss consequence, triggerable by plausible same-trust misconfiguration, no guard in the design.
Remediation: startup check rejecting any `--transformer-root` equal to / ancestor of / descendant of the resolved `--audit-root`. ADR-014 Alternative D already establishes that irreversible deletion is the line where warn-don't-block yields to a hard guard.

**[MED] F-2 — `root` discriminator is a basename; collision breaks determinism and identity**
Two transformer roots with the same trailing directory name (e.g. `/mnt/a/checkpoints` and `/mnt/b/checkpoints`) both yield `root="checkpoints"`. The `(root_discriminator, relative_path)` sort key then collides for like-named files, making `files[]` ordering resolve by insertion order — a determinism-invariant violation — and making the `root`+`relative_path` identity ambiguous.
Remediation: collision-free discriminator — index of the root in the resolved `transformer_roots` array (or full resolved path) as sort key and entry field.

**[MED] F-3 — Additivity claim rests on a stronger contract than ADR-014 actually states**
ADR-014's written forward-compat rule is "ignore unknown *keys*." Insufficient here because transformer entries reuse **known** keys with the same value domain: `classification` carries `usable`/`unconvertable`, `sha256` is now `null` for a non-`zero_byte` entry. A v1 consumer iterating `files[]` without filtering on `kind` will ingest transformer entries as usable LoRAs and may choke on `sha256: null`.
Remediation: strengthen the contract — consumers MUST branch on `kind` and skip unknown kinds; dual proof hooks (kind-filtering consumer ignores transformers; naive-iterate consumer documented to misparse). Alternatively bump `audit_version` if an external consumer cannot be assumed to filter.

**[MED] F-4 — New 1-MiB sampled-read path (§4) does not specify hostile-header guards**
The duplicate-detection read is **new** code, not the reused `_read_tiny_tensor` (256 B cap + `max(0, min(hi-lo, cap))` guard). Without the guards, a caller-supplied header with `data_offsets` where `hi < lo` yields a negative read length; `fh.read(negative)` slurps to EOF — memory blowup. Degenerate case: offsets past EOF return empty bytes; empty-equals-empty could falsely pair equal if both sides read short.
Remediation: mandate `max(0, min(hi - lo, 1 MiB, remaining_file_bytes))` and specify that any short/empty read on either side makes the pair **not** byte-equal (fail toward inclusion). Note `build_param_dict_from_dir` is header-only.

**[LOW] F-5 — Overlapping transformer roots (nested or repeated) double-count files.** Duplicate `kind:"transformer"` entries for one physical file under different keys; no security breach (no write/delete on transformer roots) but inflates counts and interacts with F-2 determinism. Remediation: detect nested/duplicate transformer roots at startup.

**[LOW] F-6 — False shape-match yields an unvalidated `usable` the catalog may auto-load.** ≥90% header shape overlap classifies `usable` with no dry-load. Worst case today: a manual load fails later. Forward-looking: once the catalog auto-loads `usable` candidates, a shapes-only false match becomes an availability vector. Remediation: none for MVP; record a re-review trigger in Deferred (mirrors ADR-014 F-10 pattern).

**[INFO] F-7 — importlib load of `audit_single_files.py` is side-effect-free.** Verified: stdlib-only imports, constants/functions, `main()` behind `__main__` guard. Keep the guard; any future module-level side effect would execute on import.

**[INFO] F-8 — Fail-toward-inclusion on duplicate read-error is sound; false-duplicate risk is self-limited.** The 1-MiB-prefix comparison can in principle be fooled by identical prefixes with differing tails (outside threat model). `duplicate_of` marks only the crafted file's *own* inclusion (self-exclusion at most); it cannot exclude a different legitimate file. "Read error → not duplicate" is the correct fail-open for an audit tool.

### Verdict

**CHANGES REQUIRED.** F-1 (HIGH) is the blocker; F-2/F-3/F-4 (MED) fold before implementation — all cheap specify-in-the-ADR fixes. F-5–F-8 informational. Re-fire `security-auditor` on the amended ADR before code lands, per the ADR-014 precedent this thread follows.

---

## Round 2

*(pending — fired on the amended ADR after the round-1 fold-in.)*
