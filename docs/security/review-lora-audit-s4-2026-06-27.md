**AI-Disclosure:** Claude (Opus 4.8, 1M context) authored this security review via the `security-auditor` task; Grant reviewed.

# Security Review — `scripts/lora_audit.py` S4 (`--delete` path)

**Date:** 2026-06-27
**Slice:** S4 of ADR-014 (`docs/decisions/ADR-014-lora-audit-tool.md` §9 Delete policy, §6 path-traversal, §3 deletable signatures)
**Branch / worktree:** `lora-convert-scripts` @ `/home/gawkahn/projects/ai-lab/code/eric-lora-convert`
**Threat model:** solo desktop / machine-caller tool; trust boundary is the caller-supplied `--audit-root` tree (untrusted contents under a trusted parent path); attacker = same-uid local write under `audit_root`. Not Red Zone today (ADR header F-10 risk-trigger has not fired — no LLM/remote caller supplies paths). The LLM-agent/remote-caller variant that would re-classify this to Red Zone is explicitly out of scope for this slice.

## Summary

S4 adds the first **file-deletion surface** to `lora_audit.py`: `--delete` unlinks files the classifier marked `deletable` (zero-byte / truncated / unparseable-header / unrecognized-garbage) from the caller-supplied `--audit-root` tree. The control (`_safe_unlink` / `_run_delete`) faithfully implements ADR §9's mandated `safe_unlink` shape and **fails closed on every branch traced**. The triple-gate holds: gate 1 (`classification == CLASS_DELETABLE` loop filter — never promotes), gate 3 (`--yes`/`confirmed`; absence = zero-I/O preview, no `input()` anywhere), gate 2 (`_passes_scan_containment` on the parent, then `O_RDONLY|O_NOFOLLOW|O_DIRECTORY|O_CLOEXEC` parent-fd open, then Linux `/proc/self/fd` realpath re-check), followed by the F-5 reclassify-before-unlink and the dir-fd-relative `os.unlink(path.name, dir_fd=parent_fd)`. The literal code order is gate1 → gate3 → gate2 (gate 3 must precede gate 2's filesystem I/O to honor the no-IO-without-`--yes` promise); the "all three required to unlink" invariant holds structurally. `parent_fd` is closed in a `finally` on every path; no containment failure is swallowed into a successful unlink; `relative_path` is always an rglob + `relative_to(audit_root)` result carrying no `..`. The named reclassify→unlink content TOCTOU (ADR §9 Option B) is **not silently widened**. **Verdict: CLEAN** — no HIGH/MEDIUM findings.

## Coverage

Reviewed:
- `scripts/lora_audit.py` — `_safe_unlink` / `_run_delete` (the S4 delete path), read in full.
- `scripts/lora_audit.py` — `_rel`, `_passes_scan_containment`, `_open_no_follow`, `_probe_safetensors_garbage`, `_probe_pickle_magic`, `_classify_deletable` (gate dependencies).
- `scripts/lora_audit.py` — `_scan` / `_classify_one` (origin of `relative_path`; confirms no `..`).
- `scripts/lora_audit.py` — `main()` delete wiring + removal of the S3 reject block; `--delete` / `--yes` flag definitions.
- `docs/decisions/ADR-014-lora-audit-tool.md` §6, §9, and the F-1..F-17 Changelog — the mandated control and accepted residuals.
- `test_lora_audit.py` — the six S4 tests + `_temp_lora_tree`.

Not re-audited (and why): S1/S2/S3 paths outside the delete surface are unchanged and were CLEAN across review rounds 1–3; `docs/security/review-lora-audit-2026-05-17.md` was not re-opened directly — its F-1..F-17 dispositions are transcribed into ADR §6/§9.

## Adversarial checks performed (same-uid attacker under `audit_root`)

- **Terminal-component symlink swap** — defeated by `os.unlink(name, dir_fd=parent_fd)` (never follows the terminal symlink; worst case removes the attacker's own symlink, not its target).
- **Intermediate-directory / parent symlink swap** — defeated by `O_NOFOLLOW` on the parent open + the `/proc/self/fd` realpath re-check under `audit_root`.
- **`..` traversal** — impossible: `relative_path` is built from `rglob` + `relative_to(audit_root)`, no `..` component survives.
- **TOCTOU between containment check and unlink** — narrowed by the dir-fd-relative unlink; the residual reclassify→unlink content window is named-and-accepted (ADR §9 Option B) and grants no capability the same-uid attacker lacks (they can already `unlink` directly).
- **No-promotion** — only `CLASS_DELETABLE` entries reach `_safe_unlink`; a forged-but-actually-usable file is rejected by the F-5 reclassify (test #32).

## Findings

**Verdict: CLEAN.** No HIGH / MEDIUM. Two LOW/INFO observations; neither blocks merge.

### [LOW — FOLDED] Absolute escape-target path reached the manifest `warnings[]`

The `/proc/self/fd` escape-rejection branch embedded the absolute *outside-root* `real_parent` (e.g. a swapped-symlink target) into the `delete_skipped_containment_failed` warning's `detail` — the F-8 incremental-disclosure leak class, relevant only if a manifest is shared off the single-user host. **Folded in this slice:** the detail is now a fixed `"parent fd realpath escaped audit_root"` token; the `file` field (via `_rel`) still identifies the entry. The **pre-existing S1 `_passes_scan_containment` instances** that embed `real` the same way are out of S4 scope and recorded as TECH_DEBT (2026-06-27) for later unification — under the stated single-user model, SA's disposition is "no action required."

### [INFO — no action] Post-unlink `os.close` failure would under-report a real deletion

If `os.unlink` succeeds but the `finally`'s `os.close(parent_fd)` raised, the `_run_delete` backstop would record `deleted=False, delete_reason=unlink_failed` for a file that was in fact removed. Fails in the safe direction (never claims deleted-when-present); `os.close` on a valid fd effectively never fails. No action warranted.

## Code-review fold (separate from this security review)

The `code-reviewer` (Opus) pass returned **APPROVED** with one LOW: the `_run_delete` backstop `except` emitted a stderr `delete_failed` line but did not append a `Warning_` to the manifest `warnings[]`, diverging from `_safe_unlink`'s own OSError contract. **Folded:** the backstop now appends `Warning_(_rel(path, ...), W_DELETE_FAILED, ...)`. Test #34 hardened to assert `delete_reason == unlink_failed` and that the failure surfaces as a manifest warning.

## Disposition

CLEAN. Both review LOWs folded in-slice; the S1 carry-over recorded as TECH_DEBT. Proof: `test_lora_audit.py` 163/163; full 10-suite regression green. Cleared to commit S4.
