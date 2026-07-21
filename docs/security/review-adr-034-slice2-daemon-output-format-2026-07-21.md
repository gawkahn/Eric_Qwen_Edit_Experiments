# Security Review — ADR-034 Slice 2 (daemon output-format)

**AI-Disclosure:** Claude (Fable 5, `security-auditor` agent) authored; Grant reviewed.
**Date:** 2026-07-21
**Change reviewed:** ADR-034 slice 2 — `--output-format`/`--quality` carried across the
daemon wire; `comfyless/server.py` resolves the `OutputFormat` server-side and owns the
on-disk extension for the O_EXCL reservation and the savepath-template branch.
**Trigger:** Red Zone (`comfyless/server.py` on `_red-zone-paths.sh`) — IPC + filesystem writes.

---

## Summary

Slice 2 carries two new wire fields (`output_format`, `quality`) into `server.py`, where the
daemon resolves an `OutputFormat` enum server-side and owns the on-disk extension for both the
O_EXCL auto-numbered reservation and the savepath-template branch. Threat model: a hostile or
buggy local wire client on the Unix socket (0700 dir, operator-spawned daemon) attempting
extension/path injection, malformed-value crashes of the accept loop, or collision/overwrite in
a shared `--output-dir`. Overall posture is good: the extension is provably enum-derived and
never a caller string, the O_EXCL image reservation and the final `_within` containment check
are intact, and every hostile-value path traced returns a structured error rather than escaping.
The one real weakening was **not the image file — it was the JSON sidecar**: splitting the
reservation namespace by extension while the sidecar namespace remained per-stem let
mixed-format runs silently clobber each other's provenance sidecars.

## Findings

### [MEDIUM] Cross-format stem collision silently overwrites the JSON provenance sidecar — FIXED

**Location:** `comfyless/server.py` (reservation loop) + `comfyless/generate.py` `_resolve_savepath` + the client sidecar write.

**Risk (as found):** The O_EXCL reservation claimed names in a **per-extension** namespace
(`comfyless{NNNN}.png` vs `.jpg` are distinct, so O_EXCL never sees the sibling), but the sidecar
namespace stayed **per-stem** (`splitext(output_path)[0] + ".json"`, opened `"w"`). Scenario, no
concurrency required: run 1 (png) → `comfyless0001.png` + `comfyless0001.json`; run 2 (jpeg) in
the same `--output-dir` reserves `comfyless0001.jpg` (counter starts at 1, `.jpg` free) and the
client then overwrites `comfyless0001.json` with run 2's metadata. Run 1's provenance is
destroyed, and `--params comfyless0001.json` replays run 2's params. For JPEG the sidecar is the
**only** provenance record (no tEXt chunk). Image files never collide — the review-parallel-daemon
Finding-1 invariant survives for the image but was displaced onto its metadata artifact, and the
ADR's proof hook ("two daemons, same dir — no collision, no overwrite") did not cover the
mixed-format case.

**Resolution (in-slice):**
- Daemon auto-numbered branch (`server.py`): after the O_EXCL image reservation, **atomically
  co-reserve `comfyless{NNNN}.json`** (a second `O_CREAT|O_EXCL`). On `FileExistsError` for the
  sidecar, release the image placeholder and advance the counter; on other `OSError`, release and
  return a structured error. Both placeholders (`_reserved`, `_reserved_sidecar`) are unlinked on
  generation failure.
- `_resolve_savepath` (`generate.py`, savepath / in-process branch): the counter now requires
  **both** `{stem}{NNNN}{ext}` and `{stem}{NNNN}.json` to be free. Non-atomic here, matching this
  path's pre-existing exists()-then-write guarantee; the daemon branch is atomic.
- Tests: `test_server_robustness.py` adds a mixed-format daemon reservation case (jpeg run
  reserves `.jpg` + `.json`; a following png run skips stem 0001, uses 0002).
  `test_output_format.py` adds the `_resolve_savepath` sidecar-skip case. ADR-034 proof hook
  amended to name the mixed-format case.

### [INFO] Wire allowlist accepts `""` for `output_format`

`""` passes the value check and folds to None (→ png) at `req.get("output_format") or None` — a
slightly wider wire surface than the CLI's `{png,jpeg,jpg}`, mirroring the existing `quant`
pattern. No path to the filesystem; benign. Noted so the widening is deliberate. No action.

### [INFO] Client writes the sidecar at a daemon-dictated path (pre-existing)

The client derives its `.json` write path from `resp["output_path"]` with no containment check —
a compromised daemon could direct a `.json` write anywhere the client user can write. The daemon
is operator-spawned behind a 0700 socket dir, so this is within the existing trust model; this
slice changes only the extension of that path. Candidate TECH_DEBT, not fixed in this slice.

## Verification of the reviewed questions

1. **Extension caller control — clean.** The only sources of `out_fmt.extension` are the two
   tuples in `_FORMATS` (`output_format.py`); the wire name is type-checked (`_KIND_STR`),
   allowlist-checked (`_validate_request`), alias-folded and membership-checked again in
   `resolve_output_format`. No caller string concatenates into the candidate filename or savepath
   extension. `resolve_output_format` is called server-side with `output_path=None`, so no caller
   path influences inference. No traversal/injection path.
2. **O_EXCL + `_within` — preserved for the image.** `O_CREAT|O_EXCL|O_WRONLY` unchanged,
   `FileExistsError`→advance / `OSError`→structured error, final `_within(output_path, output_dir)`
   runs on both branches, savepath expansion containment-checked before any `mkdir`. Same-format
   two-daemon guarantee unchanged; mixed-format now covered by the co-reservation fix.
3. **Boundary completeness — sound.** Wrong type (incl. null, bool, string) rejected by the
   canonical validator with a structured error before value checks; NaN/Infinity/0/negative/huge
   all fail `0.0 < ql <= 1.0` (NaN comparisons evaluate False → rejected); int quality safe-cast to
   float and propagated via `req.update(result.payload)`. `resolve_output_format` re-validates as
   defense-in-depth and its `ValueError` is caught. No bypass for a direct wire client.
4. **Cache-key exclusion — correct.** Format/quality affect only `_save_with_metadata`; the
   pipeline is format-agnostic and `out_fmt` is per-call, so a cached pipeline serving any format
   is right, not a wrong-pipeline serve. Same rationale as NAG; consistent with the "must NOT
   enter the schema" decision (`_SKIP_SIDECAR_KEYS`).
5. **Server-side resolve with `output_path=None` — confirmed.** No inference input, so the D2
   contradiction branch is unreachable server-side; no caller path influences format selection.
6. **Daemon survival — confirmed.** Hostile `output_format`/`quality` terminate at one of three
   structured-error returns (canonical validator, value checks, resolve catch); reservation-loop
   `OSError`s return `IOError` frames. No path from these fields to an uncaught accept-loop
   exception.

## Verdict

The Red Zone invariants the slice was gated on (enum-derived extension, O_EXCL atomicity, `_within`
containment, boundary completeness, accept-loop survival) all hold. The MEDIUM sidecar
stem-collision finding — an integrity regression against the "no silent overwrite" spirit of
review-parallel-daemon Finding 1 — was **fixed in-slice** (atomic `.json` co-reservation in the
daemon branch; sibling-existence check in `_resolve_savepath`; tests + ADR proof-hook amendment).
Cleared to commit.
