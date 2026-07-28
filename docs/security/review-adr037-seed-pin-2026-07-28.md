# Security + code review — ADR-037 D5 amendment: pinning the edit seed

**Date:** 2026-07-28
**Scope:** `comfyless/refine.py` (`pin_seed_image`, its call site in `main`,
`refine_loop`'s `edit_source_image` contract, the edit-mode entry pre-check, the
reused-output-dir warning, the entry containment list, and the DELETED ADR-040
seed warning), `test_refine.py`, `docs/decisions/ADR-037-*.md`
**Red Zone:** yes — `comfyless/refine.py` is listed in
`scripts/git-policy/_red-zone-paths.sh`, so `security-auditor` was mandatory.
**AI-Disclosure:** Claude (Opus 5) authored; `code-reviewer` and
`security-auditor` both ran on Opus 5, verified by transcript grep (see below);
Grant reviewed.

## What the slice does

`--seed-image` was the last reference the refine loop read from an
operator-owned path after entry, and it is consumed on TWO channels: the judge's
anchor holds its decoded bytes for the whole run, while `current_source` names
its PATH and is re-opened by whoever generates on every pre-promotion iteration.
A mid-run swap therefore makes generation condition on new bytes while the judge
scores identity against old ones. The slice pins the seed by value into
`<run dir>/source/` at entry, exactly as ADR-038 D5's `pin_static_refs` pins
`--ref-image` into `refs/`, and deletes ADR-040's out-of-roots seed warning
because the daemon now only ever sees the loop-owned copy.

## Verified clean

- **The pin cannot be skipped.** `edit_source` is set only in the `edit_mode`
  branch and consumed unconditionally at the pin site; no flag combination
  bypasses it. Every post-pin `current_source` assignment is loop-owned (a
  `candidates/` file, or the D4 anchor copy). After this change **no
  operator-supplied seed path can reach the daemon's `_check_ref_paths`** — the
  auditor traced this exhaustively, which is what makes deleting the ADR-040
  warning safe rather than merely convenient.
- **Ordering.** The pin runs after the ADR-040 D1 exclusive
  `makedirs(mode=0o700, exist_ok=False)` — still the first filesystem operation
  on the run dir — and before `refine_loop`. `pin_seed_image` touches only
  `source/` and `pin_static_refs` only `refs/`, so neither depends on the other's
  ordering.
- **Non-edit *seed mode* is untouched**, as the amendment scopes it: there the
  image is a params source and never a ref, and it keeps `load_seed_image_capped`.
- **Pre-existing symlinks fail closed** in both directions (`rmtree` refuses a
  symlinked `source/`; `makedirs(exist_ok=True)` raises on a dangling one).

## Findings and disposition

### HIGH — the pinned copy was never the validated copy — FIXED

Both reviewers found this independently, and it is the finding that matters:
`load_ref_image_capped(path)` validates and hashes one set of bytes, then
`shutil.copyfile(path, dst)` re-reads the path from scratch. What landed in
`source/` had passed no cap, no allowlist, and no regular-file guard.
`copyfile` imposes no size limit and rejects FIFOs but not block/char devices,
so the write was bounded by nothing — a symlink to `/dev/zero` fills the disk.
The benign case is worse than the adversarial one: a file still being written
(editor save, sync client, camera dump — the very actors motivating the
amendment) yields an oversized copy that the daemon refuses at iteration 0 on
the loop's OWN artifact, which is the post-entry failure class the "verbatim,
never re-encode" rationale exists to prevent. The logged SHA-256 then described
bytes on nobody's disk. In short: the fix meant to remove a check-then-use
window reintroduced one — the window ADR-035 6d's review had explicitly signed
off as absent.

**Fixed:** re-read through `_read_ref_bytes_capped` (same cap, same
regular-file guard), refuse unless the SHA-256 equals the validated read, and
write with `os.open(..., O_WRONLY|O_CREAT|O_EXCL, 0o600)` — `copyfile`'s
`open(dst,'wb')` follows a symlink and truncates, so a planted `source/seed.png`
on a group-writable explicit `--output-dir` could redirect the write. The same
residual in `pin_static_refs` is filed in TECH_DEBT (2026-07-28) rather than
fixed here, to keep this diff's boundary at the seed.

### MEDIUM — the pin could delete the operator's own seed — FIXED

`--output-dir /runs/r1 --seed-image /runs/r1/source/seed.png` — re-running from
a previous run's pinned seed, a natural workflow — validated, then `rmtree`'d
the directory holding the operator's file, then failed to copy it: run refused
AND seed destroyed. Only reachable with an explicit `--output-dir`.
**Fixed:** refuse before the `rmtree` when the seed is inside the target,
using the same `server._within` the containment plane uses.

### MEDIUM — the reused-dir warning covered `refs/` only — FIXED

An edit run with no `--ref-image` creates `source/` and no `refs/`, so two runs
sharing an explicit `--output-dir` had a second, unwarned collision: run B's pin
deletes run A's seed mid-flight, and run A's next generation fails to read
`current_source` — reaching the in-process latch through the loop's own
directory. **Fixed:** the warning now fires on either directory.

### MEDIUM — the first decode still used the weaker loader — FIXED

The amendment claims the format allowlist, the regular-file guard, and a
SHA-256 as the gain, but the edit-mode entry pre-check ran FIRST, on the
operator's path, through `load_seed_image_capped` — so a non-allowlisted seed
still dispatched into PIL's plugin zoo (EPS shells out to Ghostscript) before
being refused, `--seed-image /path/to/fifo` blocked forever because the
`O_NONBLOCK`+`S_ISREG` guard lives only in the other loader, and PIL's raw error
text was echoed. The pin hardened what the loop USES but not what gets DECODED.
**Fixed:** the pre-check uses `load_ref_image_capped` too.

### LOW — `refine_loop`'s pinning invariant lived in a docstring — FIXED

Both reviewers flagged it. The property "`edit_source` names a loop-owned copy
and the anchor came from the allowlisted loader" was carried by prose plus
`main`'s call order. A future caller — the video orchestrator and the LLM
planner are the named candidates — passing a path alone would silently get the
caps-only anchor decode AND send an operator path to the daemon, restoring both
closed defects. **Fixed:** `refine_loop` raises on an `edit_source` with no
`edit_source_image`. The test harness now pins exactly as `main` does, so the
suite exercises the real entry path rather than a bypass.

### LOW — provenance and leftovers — FIXED

The pin log named only the copy, and after this slice nothing else records the
run's origin (the deleted ADR-040 warning was the last line echoing it, and
sidecars record the pinned path). A refusal after pinning also left verbatim
copies of the operator's photos in a run dir it never named. **Fixed:** the log
names the original, and the refusal names the run dir and says copies may be
there.

### Records — FIXED

`TECH_DEBT.md`'s 2026-07-27 entry now carries its `Resolved:` append per §12,
including the three ways the shipped fix deviates from the fix that entry
sketched. ADR-040's "no new move step is needed" sentence — which that entry
explicitly required correcting in this slice — is corrected, and ADR-040 gains a
superseding Changelog entry so its slice-2b/slice-3 records no longer read as
current. The vault `Comfyless_Refine.md` warn-vs-refuse section is rewritten.

## Accepted residuals

- **Exposure.** In the derived case a verbatim copy of what is often a private
  photograph is written into the daemon's output root, readable by any same-UID
  wire client (`0700` does not defend against that client). The seed joins
  `refs/` in ADR-040's accepted residual; now named in the ADR rather than
  implicit.
- **No retention policy.** Copies persist after the run and nothing sweeps them
  (ADR-040 Deferred). Bounded per run at 8 refs + 1 seed × 64 MB now that the
  unbounded-write path is closed; unbounded across runs.
- **`source/` absent from the entry containment list** was raised as INFO by
  both reviewers — safe transitively, but added anyway, since that list is
  stated as "prove, don't assume".
- **`pin_static_refs`' identical validate-then-copy window** — filed
  (TECH_DEBT 2026-07-28), not fixed here.

## Reviewer model verification

Per the standing instruction, both agents were invoked with an explicit
`model: "opus"` and their transcripts grepped afterwards: 46 and 44 turns
respectively, every one `"model":"claude-opus-5"`, no Fable fallback. Both flagged that the diff handed
to them went stale mid-review (the working tree gained docstring corrections);
findings were re-checked against the final tree before disposition.

## Verdict

No CRITICAL. One HIGH (found by both), three MEDIUM, three LOW — all fixed
in-slice. The daemon-side gate is untouched, the client grants nothing new, and
the change is a net narrowing: one fewer operator-owned path read after entry,
and every read of that path now goes through the stricter loader.
