---
title: Security review — LoRA audit tool S1 (post-implementation)
date: 2026-05-25
slice: S1 of ADR-014 (--scan-only / --print-manifest)
reviewer: security-auditor (Opus, Claude Code 2.1.117)
status: CLEAN — 1 LOW + 3 INFO; LOW remediated in this slice
ai-disclosure: Claude Opus 4.7 (security-auditor agent) authored; Grant reviewed.
---

## Scope

S1 of ADR-014 LoRA audit tool — required per ADR §15 F-12 (security-auditor
re-runs on each slice that touches the path policy or deletion gates).

Files reviewed:

- `scripts/lora_audit.py` (~770 lines) — implements `--scan-only` and
  `--print-manifest`. S2 (`--dry-load`), S3 (`--convert`), S4 (`--delete`)
  flags exist but reject at runtime in S1.
- `test_lora_audit.py` (~390 lines) — 28 assertions across 8 functions; all
  passing locally.

Authoritative context:

- ADR-014 — `docs/decisions/ADR-014-lora-audit-tool.md` (§6 path policy, §15
  F-1..F-17 remediations).
- Pre-implementation rounds — `docs/security/review-lora-audit-2026-05-17.md`
  (three rounds, CLEAN).
- Vision slice — `docs/vision/slice-lora-audit.md` (§13 machine-caller
  contract).

## Threat model

Solo desktop / machine-caller tool. Trust boundary: caller-supplied
`--audit-root` directory tree (untrusted contents under a trusted parent
path). Attacker model: same-uid write under `audit_root` during a scan
(documented residual per ADR §6/§9 — such an attacker can already unlink
files). §12 trigger is file writes from caller-supplied paths (S3) and
deletions (S4), neither implemented in S1 — `--dry-load`, `--convert`,
`--delete` reject at runtime (lines 878–886 of the pre-fold code).

S1 surface: read-only header parsing of files under `audit_root`, manifest
write to disk inside `audit_root` (or `--print-manifest` to stdout), config
read from `--config` or default. Weight tensors are not loaded except for
`convertable` probing via `_load_state_dict` (safetensors → `load_file`,
.pt/.bin/.pth → `torch.load(weights_only=True)`).

## What was checked

1. **F-1..F-17 closure against implemented code:**
   - **F-1** (TOCTOU resolve→open): `_passes_scan_containment` is the
     authoritative realpath check; `_open_no_follow` is the narrowing
     defense-in-depth. Residual same-uid TOCTOU documented in the function
     docstring; matches ADR §6 Option-B framing.
   - **F-2** (atomic-write `.tmp` cleanup): `_write_manifest_atomic` writes
     to `<target>.tmp` and `os.replace`s; stray `.tmp` files surface via
     scan loop with `W_STALE_TMP_FILE` code; no auto-cleanup.
   - **F-3** (VRAM partial-state cascade): N/A in S1 (dry-load not implemented).
   - **F-4** (folder_paths stub unconditional + test): lines 32–35 install
     the stub before any `from nodes.* import …`. `test_no_real_folder_paths_import`
     asserts the stub is in `sys.modules`. The fake-package shim
     additionally prevents `nodes/__init__.py` from running, which is
     broader than F-4 required.
   - **F-5** (delete-gate classification poisoning): N/A in S1 (delete not implemented).
   - **F-6** (`--output-dir` blacklist vs allowlist): ADDRESSED at design
     level. `_validate_output_dir` splits interactive
     `--allow-output-outside-root` (blacklist) from machine
     `--require-output-allowlist` (allowlist) modes.
   - **F-7** (`weights_only=True` / `add_safe_globals` / size cap): repo-wide
     grep for `add_safe_globals` returns zero matches. Repo-wide grep for
     `torch.load(...weights_only=False...)` in `scripts/` returns zero
     matches. `_SIZE_CAP_BYTES = 5 GB` enforced via `path.stat().st_size`
     before any open. One `torch.load(...)` without `weights_only=True`
     exists in `nodes/eric_qwen_edit_component_loader.py:245`, but that
     module is NOT in the audit tool's import chain (confirmed:
     `eric_qwen_edit_lora.py` imports only `os, folder_paths, torch, typing`
     and does not pull in the component loader).
   - **F-8** (argv redaction uniform): long-flags and `=`-form covered. The
     **`-o<value>` concatenated short-form was a LOW finding in this
     review**; remediated in the same slice (see "Findings" below).
   - **F-9** (realpath-semantics dependency): the code uses
     `os.path.realpath(path)` (line 528), preserving realpath semantics;
     the audit-root-rename race is closed by this choice.
   - **F-10** (Red-Zone re-classification trigger): header-level
     documentation; no code surface.
   - **F-11** (sandbox alternative L): documentation-only.
   - **F-12** (S1 reviewer cadence): this review is the realization of F-12.
   - **F-13** (warnings array sort): `sorted(warnings, key=lambda w: ((w.file or ""), w.code))`.
   - **F-14** (fd-based open does not protect path-based reused classifier):
     ADDRESSED via the `_open_no_follow` docstring honest framing and the
     skip path for intentional symlinks. Realpath at scan enumeration is
     the authoritative control, matching ADR §6.
   - **F-15** (Linux-only `/proc/self/fd`): line 567 gates the procfs
     re-check behind `if sys.platform == "linux":`. Non-Linux gets
     `O_NOFOLLOW` + realpath as authoritative.
   - **F-16** (dir-fd-relative unlink): N/A in S1 (delete not implemented).
   - **F-17** (default-deny + mutual-exclusion on output-dir): argparse
     `add_mutually_exclusive_group()` enforces flag exclusivity;
     `_validate_output_dir` exits 1 when outside-root with neither flag.

2. **Stdout discipline:** `contextlib.redirect_stdout(io.StringIO())` wraps
   the prepare/scan loop; nodes modules have no module-load prints;
   `--print-manifest` is the only stdout writer.

3. **Deserialization safety:** `_load_state_dict` reuse goes through
   `safetensors.load_file` (safe) or `torch.load(weights_only=True)` (safe);
   pickle magic probe is byte-level, not load.

4. **Path policy:** `_passes_scan_containment` uses `os.path.realpath` (the
   authoritative control); `_open_no_follow` correctly skips
   `path.is_symlink()` (matching ADR §6 framing — intentional symlinks
   inside root are allowed).

5. **Per-file fault isolation:** `_scan`'s outer try/except +
   `_classify_lora`'s "all bases errored" `RuntimeError` both route to
   `classification=error` with `exit_code=2`.

6. **Manifest determinism:** `sort_keys=True`, `files_sorted` by
   relative_path, `warnings_sorted` by `(file, code)`.

7. **Exit codes:** 0/1/2 mapped correctly; no path bypasses them.

## Findings

### [LOW] Short-option no-separator argv form (`-o/path.json`) bypassed `_redact_argv` — REMEDIATED in this slice

Location: `scripts/lora_audit.py:417-435` (`_redact_argv`). Cross-references F-8.

**Risk:** argparse accepts `-o/abs/path.json` as a single argv token
equivalent to `-o /abs/path.json`. The redactor checked `token in
_PATH_FLAG_NAMES` (false for `-o/...`) and the `=`-form branch (false — no
`=` in `-o/...`), so the literal absolute path was recorded verbatim in
`manifest.tool_invocation.argv_redacted`. A manifest shared for diagnostics
would then leak `/home/<username>` and any sensitive subdirectory names
through the short-option form — exactly the leakage F-8 set out to close.
Same vector exists for any other short-form flag if any are added later,
but `-o` is the only short form in `_PATH_FLAG_NAMES` today.

**Remediation:** added `_SHORT_PATH_FLAGS = ("-o",)` and a short-concat
detection branch in `_redact_argv` that emits `-o<redacted>` for any token
matching `-o<value>`. Negative test (`test_redact_argv_no_path_leak`)
asserts both the long-flag form and the short-concat form scrub the raw
path. Documented at top of `_SHORT_PATH_FLAGS` that any new short-form
path flag added in S2-S4 MUST be added there.

### [INFO] `_validate_output_dir` `..` check uses un-resolved path

Location: `scripts/lora_audit.py:364-389, 396-399`.

The `..`-component rejection inspects `str(output_dir).split(os.sep)` —
the user-supplied path before `.resolve()`. ADR §6 specifies that
`--output-dir` must not contain `..` components, which is what this gate
checks. Subtler concern: if a caller passes an absolute path with no `..`
but the path resolves outside the audit-root, the only remaining defense
for the blacklist branch is the system-directory blacklist — which is
ADR-acknowledged as gappy. Not a new finding vs. ADR; just confirming the
defense is what ADR §6 said it is (gappy foot-gun-dampener).

**Remediation:** none required if behavior matches ADR §6 intent (it does).

### [INFO] Blacklist with `Path("/")` deny-alled `--allow-output-outside-root` — REMEDIATED in this slice

Location: `scripts/lora_audit.py:377-387`.

`Path("/")` was in `_BLACKLIST`, and `real.is_relative_to(Path("/"))` is
True for any absolute resolved path. This meant `--allow-output-outside-root`
*always* failed with `[ERROR] --output-dir resolves to system-blacklisted
path:` regardless of the actual target. Safer than ADR §6 prescribed (ADR
allowed e.g. `/opt/somewhere`) but the documented "interactive convenience
escape" path was functionally non-existent in S1. No S1 user-facing harm
because `--convert` is rejected at runtime, but it would surface as a
functional regression in S3.

**Remediation:** removed `Path("/")` from `_BLACKLIST`. The remaining
entries (`/etc`, `/usr`, `/var`, `/sys`, `/proc`, `/dev`, `~/.ssh`,
`~/.gnupg`) still cover the documented foot-gun-dampener intent. Added a
comment at the blacklist tuple explaining why `Path("/")` is excluded so
future maintainers don't re-add it. New regression test
(`test_allow_output_outside_root_usable`) exercises the now-reachable
path.

### [INFO] `_open_no_follow` skip on `path.is_symlink()` is correct per ADR §6 framing

Location: `scripts/lora_audit.py:553-554`.

For paths that are symlinks (inside-root targets), `_open_no_follow`
returns True without opening — relying on `_passes_scan_containment`'s
realpath check as the authoritative control. This matches ADR §6's
reuse-only honest framing (the reused path-based classifier will re-open
the symlinked path regardless of any fd we hold).
`test_symlink_inside_resolves_normally` covers the positive case (symlink
to inside-root file appears classified as `usable`). What's not covered:
the negative case — a deliberate symlink whose target is *also* a symlink
(chain), or a symlink to a non-existent inside-root target (caught by the
`real.exists()` guard in `_passes_scan_containment`, but not
regression-tested). Not a security defect in S1.

**Remediation:** none required. Consider adding in S2 a
`test_symlink_chain_inside` that constructs `a -> b -> c.safetensors` (all
inside root) and asserts a single FileEntry appears.

## F-1..F-17 closure verification — confirmed CLEAN

All seventeen findings from the three pre-implementation review rounds
close cleanly against the S1 implementation, with the documented MVP
residuals (same-uid TOCTOU, blacklist gappiness) acknowledged in ADR and
not regressed in code. The S1 review fold (B1-B4 from `code-reviewer`
plus the LOW above) reinforces F-8 (argv redaction completeness), F-17
(default-deny coverage), and the F-6 blacklist semantics without
introducing new surface.

## Companion `code-reviewer` findings (also folded in this slice)

The `code-reviewer` (Opus) pass on the same files produced four BLOCKING
findings, all addressed in this slice. They are summarized here for the
audit-trail record (their detailed remediations land in the commit body
of `scripts/lora_audit.py` + `test_lora_audit.py`):

- **B1** — `--output-allowlist-prefix` resolved non-strict; could accept
  prefixes that don't exist at startup. Fixed: `pfx.resolve(strict=True)`
  + `is_dir()` check.
- **B2** — `.tmp` detection in `_scan` over-broad (would flag any
  `notes.tmp`, in-flight `.safetensors.tmp` from concurrent runs). Fixed:
  narrowed to `rel.endswith(".safetensors.tmp") or path.name ==
  "lora_audit.json.tmp"`.
- **B3** — argv redaction missed `-o<value>` short form. Fixed (see LOW
  above).
- **B4** — `Path("/")` deny-all in blacklist. Fixed (see second INFO above).

## Verdict

**CLEAN with one LOW (short-option argv leakage, REMEDIATED) and three
INFO observations (two REMEDIATED, one accepted).** All F-1..F-17
remediations from the three pre-implementation review rounds remain closed
against the S1 implementation. No exploitable surface identified in the
S1 slice. The S3 (`--convert`) and S4 (`--delete`) gates remain
unreachable until those slices are implemented; this review's findings
proactively close gaps that would otherwise surface there.
