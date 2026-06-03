**AI-Disclosure:** Claude (Opus 4.8, 1M context) authored this security review via the `security-auditor` task; Grant reviewed.

# Security Review — `scripts/lora_audit.py` S3 (`--convert` write path)

**Date:** 2026-06-02
**Slice:** S3 of ADR-014 (`docs/decisions/ADR-014-lora-audit-tool.md` §8, §10)
**Branch / worktree:** `lora-convert-scripts` @ `/home/gawkahn/projects/ai-lab/code/eric-lora-convert`
**Threat model:** solo desktop / machine-caller tool; trust boundary is the caller-supplied `--audit-root` tree (untrusted contents under a trusted parent path); attacker = same-uid local write under `audit_root`. Not Red Zone today (ADR header F-10 risk-trigger has not fired — no LLM/remote caller supplies paths).

## Summary

S3 adds the first **file-write surface** to `lora_audit.py`: `--convert` reads each `convertable` source LoRA, re-derives the live `ConversionPlan`, runs `convert_state_dict` in-process, and writes `<stem>.<target_family>.safetensors` via `safetensors.torch.save_file(tmp)` + `os.replace(tmp, target)` under either `audit_root` (default) or a `_validate_output_dir`-vetted `--output-dir`. The write target name always carries an inserted `.<target_family>` infix, the output base dir is containment-validated, `entry.relative_path` is an rglob result that already passed scan-time containment (no `..`), and a defensive post-join `resolve().relative_to(base_dir)` re-check runs before the collision check. `convert_state_dict`'s output is purely in-memory tensors — no path is derived from file *content*. `target_family` is a registered-plan literal (`diffusers_klein` / `diffusers_chroma`), not caller input. Overall posture is **CLEAN under the stated MVP threat model**; the residuals are the same same-uid TOCTOU races ADR §6/§8/§10 already name and accept, plus the recorded MEDIUM observation. No S1/S2 control was regressed.

## Coverage

Reviewed:
- `scripts/lora_audit.py` — `_convert_one` / `_run_convert` (the S3 write path), read in full.
- `scripts/lora_audit.py` — `--output-dir` / `--convert` / `--delete` flag definitions.
- `scripts/lora_audit.py` — `_validate_output_dir` (inherited S1 control gating `--output-dir`).
- `scripts/lora_audit.py` — `_scan` `.tmp` surfacing (F-2 Option A interaction with `.safetensors.tmp`).
- `scripts/lora_audit.py` — `--delete` gate still rejects; `--convert` wired under `redirect_stdout`.
- `nodes/eric_lora_format_convert_apply.py` — `find_matching_plan` / `convert_state_dict` (provenance of plan + in-memory-only output).
- `nodes/eric_lora_format_convert.py` — `ConversionPlan` / `register_plan` / `CONVERSION_PLANS` (target_family provenance).
- `nodes/eric_lora_format_convert_{flux,chroma}.py` — `target_family` literal values.
- `nodes/eric_qwen_edit_lora.py` — `_load_state_dict` deserialization safety (`weights_only=True`).
- ADR-014 §6/§8/§10; S1 review (2026-05-25); S2 review (2026-05-28) for the established threat model and F-2/F-8/F-12/F-16 controls.

## Findings

### [INFO] Source-clobber is structurally impossible — confirmed (Vision invariant 3)

`target_name = f"{rel.stem}.{target_family}.safetensors"` always inserts the `.<target_family>` infix, so `target_path != source_path` by name for any input. `target_family` is a registered-plan literal containing no `.`/`/` that could collapse the infix (`diffusers_klein`, `diffusers_chroma`); the registry `CONVERSION_PLANS` is import-time-only and not influenceable by a caller-supplied LoRA file. The `.tmp` is `target + ".tmp"` in the target directory and `os.replace(tmp, target)` only ever names `target`. The source is opened read-only by `_load_state_dict` and never written.

Residual (accepted, same as ADR §8): if a same-uid attacker pre-plants a symlink at the target name pointing at the source file, `os.replace(tmp, target)` replaces the *symlink* with the regular tmp file (rename onto the link name, not through it) — the source inode is untouched. The collision check (`target_path.exists()` follows symlinks) would in fact catch a pre-planted symlink first and skip as `collision`. Confirmed safe.

### [INFO] Write-path traversal / symlink escape — contained, residual is the accepted same-uid TOCTOU

`out_rel` is derived from `PurePosixPath(entry.relative_path)` whose `..`-freedom is guaranteed by scan-time `_passes_scan_containment` (realpath under `audit_root`). `base_dir` is `audit_root` (realpath-contained) or the `_validate_output_dir`-vetted `--output-dir`. The defensive `target_path.resolve().relative_to(base_dir.resolve())` re-check is belt-and-suspenders. The named residual — a same-uid attacker swapping an intermediate directory component under `output_dir` for a symlink-to-outside in the window between the `resolve()` re-check and `mkdir(parents=True)`/`save_file`/`os.replace` — is a genuine TOCTOU. Accepted under the MVP threat model, consistent with ADR §6: the same attacker can write outside `audit_root` directly; the audit tool grants no new capability. See MEDIUM below for the one place this matters more.

### [MEDIUM] `mkdir(parents=True)` + `os.replace` follow symlinked intermediate dirs under `--output-dir` — narrowing absent on the write path

With `--output-dir` pointed outside `audit_root` (interactive `--allow-output-outside-root`, or machine `--require-output-allowlist`), the per-file target is `output_dir/<relative_source_dir>/<name>`. `<relative_source_dir>` comes from the *source* tree (attacker-influenceable). If a same-uid actor pre-plants a symlinked intermediate directory under `output_dir` (e.g. `output_dir/sub -> /home/gawkahn/.ssh`), `mkdir(parents=True, exist_ok=True)` traverses it and `os.replace` writes the converted `.safetensors` through it — a write outside the validated base. The post-join `resolve().relative_to(base_dir)` re-check happens *before* `mkdir`, so a symlink swapped after that check but before `mkdir`/`replace` is not re-validated (TOCTOU); a symlink-to-outside present at check time *is* caught by the full-path `resolve()`, so the pure-swap-after-check is the live vector.

Assessment: same same-uid TOCTOU class as the INFO above, within the documented accepted residual (the attacker can write to `.ssh` directly). Flagged MEDIUM-not-INFO because (a) it is the one S3 vector where the blast radius leaves `audit_root` *and* the write content is attacker-influenced (a valid-looking `.safetensors`), and (b) the write path conspicuously lacks the `O_NOFOLLOW` narrowing the rest of the tool applies, so a reader could wrongly assume parity. Under the MVP threat model with `--output-dir` defaulting inside `audit_root`, **no fix is required before commit.**

Remediation (smallest, optional, only if hardening desired): create the target parent via `os.open(parent, O_NOFOLLOW|O_DIRECTORY|O_CLOEXEC)`-validated components or write via a dir-fd (`os.open` the validated parent, dir-fd-relative replace), mirroring the ADR §9 delete pattern. Do **not** undertake this as scope creep in S3. Recorded as a TECH_DEBT § Security entry (2026-06-02): "S3 convert write path lacks the O_NOFOLLOW/dir-fd intermediate-symlink narrowing"; trigger = F-10 risk-trigger fires, or S4 lands the dir-fd delete path.

### [INFO] Collision TOCTOU (`exists()` then `os.replace`) — residual is benign under same-uid model

`target_path.exists()` returns no-collision, then a concurrent writer creates `target`, then `os.replace` overwrites it unconditionally. Threat actor is a same-uid concurrent process (the only one that can write there); they can already overwrite the file directly, and the source is never the target so no source loss occurs. The window is the converted-sibling output only. Matches ADR §8's pre-write-check framing (the check is a courtesy no-overwrite guard, not a concurrency lock). Accepted.

### [INFO] `.tmp` orphan policy preserved (ADR §10 F-2 Option A) — no S1 regression

`_convert_one` cleans up only its own tmp on a failed write within the same call (`if tmp.exists(): tmp.unlink()` inside the write `except`), never at scan time. `_scan` surfaces stray `*.safetensors.tmp` as `stale_tmp_file` and continues — no unlink. Concurrent A-mid-write / B-scan is safe. No tool-internal delete exists outside the (still-rejected) `--delete` gate. Invariants 3/4 intact. (Note: when `--output-dir` points outside `audit_root`, an orphan tmp there is never scanned, so `stale_tmp_file` will not fire for output-dir orphans — a documented-by-omission gap the ADR §10 narrative does not address; benign.)

### [INFO] No new deserialization surface on the write path

`_convert_one` re-loads the source via `_load_state_dict` — the same surface S1/S2 vetted: `safetensors.load_file` (mmap, no exec) for `.safetensors`, `torch.load(weights_only=True)` for `.pt/.bin/.pth`. `convert_state_dict` operates on in-memory tensors and emits in-memory tensors; `safetensors.torch.save_file` is serialization-only. No new deserialization risk. (Carry-forward flag from S2 I-6 still applies: re-confirm `weights_only=True` survives any torch/diffusers bump.)

### [INFO] stdout/stderr leakage — WARN lines emit relative paths only (F-8 posture preserved)

All five WARN emitters (`convert_skipped_escape`, `convert_skipped_collision`, `convert_failed` ×2, backstop `convert_failed`) emit `out_rel` or `entry.relative_path` — both relative, no absolute prefix. Exception text is truncated to 200 chars. `_run_convert` runs under `redirect_stdout(io.StringIO())` so reused-loader stdout chatter cannot break `--print-manifest`. Consistent with F-8. Residual parity note: a torch/safetensors error message could embed an absolute filename in the truncated exception summary on stderr — same accepted residual as S2 I-1 (stderr, not stdout; same-uid only).

### [INFO] `target_family` and `convert_plan` provenance — not caller-controlled

`target_family` originates from `ConversionPlan.target_family`, a literal set only by import-time `register_plan` calls in `eric_lora_format_convert_{flux,chroma}.py` (`diffusers_klein`, `diffusers_chroma`). A caller-supplied LoRA file cannot register or mutate a plan. The values contain no path separators or `..`. The guard fails closed (`convert_failed`) if either field is absent. Confirmed safe.

### [INFO] `--delete` / `--yes` still reject; S1/S2 controls intact

`--delete`/`--yes` return `EXIT_STARTUP_FAIL` with the S3-specific message. `_validate_output_dir`, `_passes_scan_containment`, `_open_no_follow`, `_redact_argv`, the dry-load loop, and the manifest writer are unchanged from the S2 baseline. No scope creep into security-truth surfaces. `--output-dir` is newly *consumed* by the convert path but was already defined and validated in S1; S3 does not weaken `_validate_output_dir`.

## Verdict

**CLEAN — no CHANGES REQUIRED before commit.**

S3's write path is correctly contained: source-clobber is structurally impossible, `target_family` is non-caller-controlled, the post-join containment re-check and collision skip are present, and the `.tmp`/F-2 Option A concurrency posture is preserved without regressing any S1/S2 control. The one item rising above INFO — the write path's lack of `O_NOFOLLOW`/dir-fd narrowing on intermediate directories under an out-of-root `--output-dir` — is a same-uid TOCTOU that falls squarely within the residual ADR §6/§8 already names and accepts. It does not block merge under the MVP threat model and is recorded as TECH_DEBT (fix shape: dir-fd-validated parent write mirroring ADR §9; trigger: F-10 risk-trigger or S4 delete-path landing).

**Action items (non-blocking):**
1. ✅ Persist this review to `docs/security/review-lora-audit-s3-2026-06-02.md` and reference it in the S3 commit body.
2. ✅ Add the MEDIUM as a `TECH_DEBT.md` § Security entry.
3. ✅ S3 test coverage confirmed for: target-name infix prevents source==target; out-of-root `--output-dir` escape rejection; collision skip leaves source untouched; orphan `.tmp` surfaced not deleted; WARN lines carry relative paths only; `--convert --print-manifest` stdout stays JSON-only.
