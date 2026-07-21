AI-Disclosure: `security-auditor` subagent (Fable) authored the findings; Grant reviewed. Remediation authored by Claude (Opus 4.8), Grant reviewed.

# Security review — ADR-035 slice 2: reference-image ingestion helper

**Date:** 2026-07-21
**Surface:** `comfyless/ref_image.py` (new) — §12 image-ingestion trigger (untrusted local file content → PIL decode → model conditioning).
**Trust model:** CLI-local hostile file *content* (not hostile path containment — that is the caller's job per ADR-035 6b). No auth/PII/network/billing.
**Reviewer:** `security-auditor` (Fable), no-shell read-only review against ADR-035 decision 6 (6c/6d/6g) and `docs/vision/slice-ref-image-cli.md`.
**Verdict:** **Design contract met.** No CRITICAL/HIGH. Two LOW (one closed in-slice, one deferred to slice 4 with a TECH_DEBT precondition), two INFO.

---

## Design-contract verdict (auditor)

- **6d single-read / TOCTOU — met.** One `open`/`read(max_bytes+1)`, no prior `os.stat`, hash over `data`, decode over `io.BytesIO(data)`. No check-then-use window; decoded bytes are byte-identical to hashed bytes by construction.
- **6c allowlist — met.** `formats=list(formats)` passed to `Image.open`; Pillow pinned 12.3.0 (`formats=` supported since 8.0). First-frame-only holds — no `seek()` anywhere; `convert()` operates on frame 0.
- **Pixel cap before decode — met.** Check on the lazy `img.size` before `convert("RGB")`. PNG/JPEG/WEBP report size from the header without pixel decode.
- **6g error hygiene — met in substance,** with one narrow residual (Finding 1), closed in-slice.

**Test-suite adequacy (auditor):** negative cases are real, not vacuous. Byte cap exercised at both boundary sides with a secret-marker leak assertion; the pixel-cap test asserts the dimension message on a <200-byte forged PNG, distinguishing the pixel-cap branch from a decode failure (proving order); allowlist rejection proven on three real files (GIF/BMP/TIFF); sidecar-JSON 6g case asserts the secret string is absent; multi-frame WEBP asserts frame-0 pixels by color. `expect_error` correctly fails when nothing raises.

---

## Findings

### [LOW] Finding 1 — rewrapped PIL exception text can echo ≤4 bytes of file content (PNG chunk IDs) — **CLOSED in-slice**
`ref_image.py` open/decode rewraps originally interpolated `{e}`. Pillow's PNG plugin embeds a 4-byte chunk id read from the file into some messages (`PngImagePlugin.py:183` `broken PNG file (chunk {repr(cid)})`; `:222` `bad header checksum in {repr(cid)}`), so a mostly-valid PNG that fails chunk parsing during `.load()` could echo up to 4 attacker-positioned bytes. Auditor rated it a letter-of-6g deviation, **not a usable oracle** — the stated sidecar-JSON/secret threat lacks PNG magic and fails with a content-free `UnidentifiedImageError`.

**Verification during remediation:** at Pillow 12.3.0 the escaping paths were not reproducible from the CLI-reachable surface — corruption during `_open` re-wraps to a content-free `UnidentifiedImageError`; the byte-echoing `SyntaxError` only escapes via `.load()` on a narrowly-crafted trailing malformed chunk. The format strings do exist in 12.3.0 source, so the channel is real if narrow.

**Remediation (applied):** both rewraps now surface `type(e).__name__` only, never `str(e)`. The class name (`UnidentifiedImageError` vs `OSError` vs `DecompressionBombError`) is the actionable part and structurally cannot carry file bytes; the open branch retains the allowed-formats hint. New regression test: a broken-body PNG carrying an ASCII marker in a corrupt IDAT reaches the convert rewrap and the test asserts the marker is absent and only path + class appear.

### [LOW] Finding 2 — non-regular-file path (FIFO/device) hangs `open()` forever — **DEFERRED to slice 4 (TECH_DEBT precondition)**
`open(path, "rb")` on a FIFO blocks in `open(2)` until a writer appears; no cap is ever reached. Benign under this slice's CLI-local trust (operator self-harm, consistent with the repo's warn-don't-block posture). Becomes a **daemon DoS in slice 4**: the daemon decodes paths inside `ref_image_roots` (defaults to `--output-dir`, a tree lower-trust flows write into), so a same-UID `mkfifo output/kf_003.png` hangs the VRAM-holding daemon. An `fstat`+`S_ISREG` check does **not** close it (the hang is in `open` itself); the daemon-side fix needs `os.open(..., O_NONBLOCK)`.

**Disposition:** no code change this slice (auditor concurred). Logged in `TECH_DEBT.md` as a hard precondition of slice 4's daemon exposure so it cannot land silently.

### [INFO] MPO files dispatch through the "JPEG" allowlist entry
A JPEG carrying MPO markers returns an `MpoImageFile` even with `formats=["JPEG"]`. Not an allowlist escape — MPO frames are libjpeg-decoded and this helper never seeks, so first-frame-only and the header pixel cap hold. Noted in the 6c constant comment so the "never dispatches outside the allowlist" claim stays precise.

### [INFO] Pixel cap bounds pixel count, not decode-buffer bytes
A 16-bit RGBA PNG at the 67 Mi-px cap allocates ~537 MB source + ~201 MB RGB copy (~740 MB transient/image). Bounded; aggregate bounded by the reference-count cap of 8 (enforced at parse/wire, out of scope here). Intentional, mirrors `refine.py`. No change.

---

## Out of scope (confirmed, not flagged)
Path containment / `ref_image_roots` (caller's job, 6b — slice 3/4); reference-count cap =8 (parse slice 1 / wire slice 4); daemon NUL-byte defense 6e (slice 4).
