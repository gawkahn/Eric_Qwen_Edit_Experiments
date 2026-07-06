# Security review — ADR-021 transformer audit (design phase)

**Date:** 2026-07-05 · **Reviewer:** security-auditor (Opus, model pinned at invocation) · **Round 1: CHANGES REQUIRED** (1 HIGH / 3 MED / 2 LOW / 2 INFO — all folded same-day) · **Round 2: CLEAN** (all folds ADDRESSED; 3 INFO implementation notes NEW-1..NEW-3 carried into the implementation slice). ADR-021 Status → accepted.

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

## Round 2 (verbatim)

**Date:** 2026-07-05 · **Reviewer:** security-auditor (Opus, model pinned at invocation) · **Round 2 verdict: CLEAN.**

### Per-finding fold verification

**F-1 (HIGH) — root-disjointness startup invariant — ADDRESSED.** §1 rejects every `--transformer-root` that is equal to / ancestor of / descendant of the resolved `--audit-root`, exit 1, naming both paths. The three delete-crossing cases all close: **equal**, **descendant** (`--audit-root /models --transformer-root /models/checkpoints` — the original scenario), and **ancestor** (`--audit-root /models/checkpoints --transformer-root /models` — where a LoRA under audit-root would sit under a transformer root). The check operates on realpaths, and the pre-existing per-entry O_NOFOLLOW realpath-descendancy on the delete path backstops the symlink-nesting variant. The hard-block-over-warn choice is correctly justified by the ADR-014 Alternative D irreversible-deletion precedent (deliberate, documented deviation from warn-don't-block — see NEW-2).

**F-2 (MED) — root_index identity + sort key — ADDRESSED.** Pairwise-disjoint roots guarantee each physical file appears under exactly one root, so `(root_index, relative_path)` is unique; LoRA entries carry `-1` and cannot collide with transformer indices (≥0). CLI order is stable; the key is a total order. Same-basename roots resolve to distinct indices. Vision negative case 9 codifies it.

**F-3 (MED) — kind-branching contract + dual proof hooks — ADDRESSED.** "Consumers MUST branch on `kind`" with two hooks (kind-filtering consumer sees exactly the v1 LoRA view; naive-iterate consumer asserted to misparse). Keeping `audit_version` 1 is defensible on the stated assumption that the only v1 consumer is the in-repo ADR-022 catalog, with the external-consumer escape hatch on record.

**F-4 (MED) — hostile-header guard — ADDRESSED.** Verified `max(0, min(hi - lo, 1 MiB, file_size - (8 + header_len + lo)))` against the safetensors layout: `hi < lo` → floored to 0; `lo` past EOF → 0; hostile `header_len` → 0, with the inherited 100 MB header cap bounding the prefix parse. Empty-equals-empty closed by "short/empty/errored read on EITHER side → NOT byte-equal." Fail-toward-inclusion preserved; base side confirmed header-only. Vision negative case 10 codifies it.

**F-5 (LOW) — pairwise disjointness — ADDRESSED.** Kills duplicate-minting without a dedupe pass; reinforces F-2.

**F-6 (LOW/forward) — auto-load re-review trigger — ADDRESSED.** Deferred section records the trigger (ADR-014 F-10 pattern).

**F-7 / F-8 (INFO)** — no change required, unchanged and still sound.

### New concerns introduced by the folds (all INFO; implementation notes)

**NEW-1 — disjointness predicate mechanism unspecified (prefix vs path-component).** Naive `startswith` over-rejects sibling dirs sharing a name-prefix (`/a/checkpoints` vs `/a/checkpoints_old`) — errs fail-closed, not a hole. Implementation: compare on path-component boundaries (`os.path.commonpath` / `Path.is_relative_to`).

**NEW-2 — hard-block is a deliberate deviation from warn-don't-block.** Correct here (irreversible deletion), but a future maintainer applying the warn-habit could downgrade it and silently reopen F-1. Implementation: inline comment tying the abort to F-1 / Vision invariant 7.

**NEW-3 — "K=4 largest unique-shape tensors" lacks a tie-break** for equal-sized tensors; content comparison converges regardless, so purely a reproducibility note against invariant 10. Implementation: deterministic tie-break (e.g. tensor key name).

### Verdict

**CLEAN.** All round-1 findings folded correctly. The three NEW items are INFO-level implementation-hardening notes. Implementation may proceed; carry NEW-1..NEW-3 into the `code-reviewer` + `security-auditor` pass on the implementation slice, and ensure Vision proof hooks exercise the F-1 ancestor case and the F-4 empty-read path (negative cases 8, 10 already name them).

---

## Implementation-phase reviews (2026-07-06)

Both reviewers (Opus, model pinned) ran on the implementation diff; both returned **CHANGES REQUIRED** with complementary findings, all folded same-day; post-fold suite 197/197, full regression 2060 green.

**code-reviewer (CHANGES REQUIRED → folded):**
- **Finding 1 (MED):** `_base_transformer_index` skipped unreadable base shards per-shard, SHRINKING |B| and INFLATING `|T∩B|/|B|` — the inline comment claimed "conservative," which was inverted; a partial base could manufacture a false `usable`/`duplicate_of`. **Fixed:** any unreadable shard marks the whole base unavailable for transformer matching (empty index → files fall to `no_matching_base`/`format_unknown`, fail-toward-inclusion) + loud per-shard warning + covering test (9a).
- **Finding 2 (LOW):** `_nested`'s `commonpath` `ValueError` branch was fail-open. **Fixed:** returns True (overlap → abort); unreachable in production but the guard protects a HIGH invariant.
- Finding 3-8: APPROVED (duplicate-gate fidelity incl. first-byte-equal-base determinism; §2 mapping; §3 math; §4 guard; §5 manifest; §6 delete filter + size-cap exemption + fault isolation). Dead `_T_USABLE_VERDICTS` constant removed.
- **Finding 9 (LOW):** test gaps — AIO fixture, inconclusive-dup warning, same-basename roots, unreadable-base-shard. **All added** (9a-9d, +6 tests).
- NEW-1/NEW-2/NEW-3 verified landed.

**security-auditor (CHANGES REQUIRED → folded):**
- **F-T1 (MED):** the transformer path's 5 GB size-cap exemption reopened an UNBOUNDED header read in `_probe_safetensors_garbage` (`f.read(n)` on a caller-declared uint64 with no upper bound; the LoRA path had been implicitly bounded by the size cap). **Fixed:** 100 MB header cap before the read (mirrors `audit_single_files._header`), > cap → `unparseable_header`; negative test added (crafted 200 MB-declared header classified garbage with no large read); `_classify_transformer` docstring corrected.
- **F-T2 (INFO):** disjointness predicate is case-sensitive — correct on this ext4/mergerfs deployment; assumption recorded in `_check_root_disjointness` docstring.
- **F-T3 (INFO):** same fail-open `ValueError` branch as code-review finding 2 — closed by the same fold.
- **F-T4 (INFO):** malformed `data_offsets` in the dup-checker → per-file `error` entry via fault isolation (contained; no change).
- Verified: F-1 hard-block symlink/relative/trailing-slash resistant + pre-scan; F-4 formula exact; NEW-3 deterministic; delete kind-filter unreachable for transformers; zero-write property; per-file isolation.

**Post-fold status: both gates satisfied; slice committed.**
