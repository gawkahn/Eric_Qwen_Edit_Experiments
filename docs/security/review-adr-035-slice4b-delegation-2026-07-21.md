# Security Review — ADR-035 slice 4b (daemon-authoritative ref delegation + RefPathError fallback)

**AI-Disclosure:** Reviewed by the `security-auditor` subagent (Claude, Fable) on
2026-07-21; triaged/remediated by Claude (Opus 4.8), Grant reviewed.
**Scope:** `comfyless/{server,generate}.py` + `test_ref_edit.py` (the delegation-seam
delta only; slice-4 daemon-side containment/decode/caps unchanged). **Surface:**
§12 Red Zone — Unix-socket IPC. **Design authority:** ADR-035 2026-07-21 slice-4b
Changelog + `docs/vision/slice-ref-image-daemon.md` §H.

## Verdict

**Contract HOLDS — no CRITICAL / HIGH / MEDIUM / LOW; all findings INFO.**

The revised seam: a `--ref-image` run delegates on the same predicate as any run;
the daemon's `_check_ref_paths` is the sole containment authority; a refusal comes
back as a **distinct wire `error_type: "RefPathError"`** which the client converts
to an in-process fallback (return `None` → main() runs locally with row-1 user
authority) with a loud warning. Model-path `PathError` and every other daemon
error still hard-fail (rc 1).

## Confirmed invariants

1. **Fallback scope.** Only `error_type == "RefPathError"` triggers the fallback;
   keyed on the wire field, never a message substring. `RefPathError` has exactly
   one emitter (`server.py` `_handle_connection`), and `_check_paths` (weight/model
   roots) runs **before** `_check_ref_paths`, so a `RefPathError` on the wire proves
   the model paths already passed containment — a weight-root violation can never
   reach the fallback. Missing/unknown `error_type` (ClientRecvError, legacy daemon)
   → rc 1 (fail-closed).
2. **No leak/bypass.** `_check_ref_paths` does realpath containment only — no
   `open()`, no read; refusal is strictly pre-decode and reserves no output. The
   in-process fallback re-ingests through `load_ref_image_capped`, so byte/pixel
   caps, format allowlist, and the `S_ISREG` guard still apply; only the
   *containment* gate is absent — exactly decision 7 row 1 (typed CLI path = user
   authority). Not a downgrade (that path was readable in-process pre-4b too).
3. **Trust class boundary-determined.** The deleted client pre-filter was not a
   trust assertion; nothing replaced it. No new client-asserted trust field on the
   wire. The daemon's containment decision remains unilateral.
4. **Slice-4 guarantees intact.** Server-side diff is a single string change
   (`"PathError"` → `"RefPathError"`) + comments. NUL defense, count cap, entry
   shape/mode allowlist, non-regular-file guard, decode caps, and cache key are all
   outside the diff and unchanged.
5. **No silent drop.** Fallback returns `None`; main() runs in-process passing
   `ref_images=` — the reference is honored, not dropped, with a loud warning.

## INFO findings (no blocking action)

- **In-process fallback shares the GPU the daemon still holds VRAM on** — possible
  self-OOM. Availability-only, warned, pre-existing. **Mitigated:** the warning now
  names the escape hatch (`--unload the daemon if it OOMs`).
- **Ref paths now cross the wire + land in the daemon refusal log even when refused**
  (was in-process-only pre-4b). Acceptable same-UID solo desktop; revisit if
  `--serve` ever grows a network transport (already a project Red Zone trigger).
- **Version skew:** a pre-4b daemon returns `PathError` for a ref refusal → 4b
  client hard-fails rc 1 instead of falling back. Fails closed; clears on daemon
  restart. (Also tracked in TECH_DEBT for the slice-4 version-skew silent-drop.)
- **Fallback keying spoofable only by a same-UID socket squatter** — outside the
  ADR-001/ADR-035 threat model; a spoofed `RefPathError` only demotes the run to
  in-process with the user's own arguments (no authority gained).
