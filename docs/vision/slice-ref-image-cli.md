# Vision — Reference-image CLI: foreground qwen-edit (`comfyless/generate.py` + `comfyless/ref_image.py`)

**Date:** 2026-07-20 · **ADR:** ADR-035 (accepted) · **Risk:** L2

> **Posture:** Boundary: entrypoint (`--ref-image` CLI flag) + local file
> reads (untrusted image content → PIL decode → model conditioning, a §12
> trigger). Risk factors: arbitrary-user-file decode. No auth/PII/network. Trust
> model is CLI-local, same as `--seed-image` / `generate.py --params`. Daemon
> boundary and sidecar replay are **out of scope** for this slice (own Vision
> docs later — ADR-035 decisions 6a-wire / 7).

## Intent

`python -m comfyless.generate --model <qwen-edit ckpt> --prompt "…"
--ref-image car.jpg:vl` runs a real Qwen-Image-Edit-2511 edit from the CLI:
1–8 reference images, each with a per-image `MODE` selecting its conditioning
paths, foreground (in-process) only. This is the first consumer increment for
ADR-033 keyframe authoring — evolve keyframe N → N+1 with scene lock (`:both`)
or viewpoint change (`:vl`).

## Scope of THIS slice (ADR-035 implementation slices 1–3)

- **1. Schema + flag parsing** — `--ref-image PATH[:MODE]` (repeatable),
  `ref_images` schema key, `MODE` validation, ref-count cap.
- **2. Ingestion helper** — new `comfyless/ref_image.py`: the decode/cap/hash
  security core (ADR-035 decisions 6c/6d/6g).
- **3. Foreground execution** — qwen-edit family routing into
  `generate_qwen_edit` (mined from `nodes/eric_diffusion_manual_loop.py:2373`),
  `MODE` → per-image `vl_flags`/`ref_flags`, output-dim derivation, row-1 trust
  treatment + skip-delegation for typed paths outside roots.

Deferred to their own Vision slices: daemon `ref_image_roots` + wire strictness
+ NUL defense + cache-key pin (ADR-035 6a/6b-daemon/6e/6f-daemon/3-test,
decision 2 wire flag); sidecar recording + replay trust (decision 7).

## Decisions inherited from ADR-035 (settled, not re-opened here)

1. **`--ref-image PATH[:MODE]`**, argparse `action="append"`, last-colon split
   like `_parse_lora_arg` (`generate.py:2097`). `MODE ∈ {both,vl,ref}`,
   default `both`. Any other suffix is a **hard error** — a colon-bearing path
   must append an explicit `:MODE` (`photo:ref:both`).
2. **`MODE` → conditioning paths:** `both` = VL+Ref (scene lock, geometry
   follows), `vl` = VL only (semantic carries, geometry free — the car
   re-orientation case), `ref` = Ref only. No `none`; an ignored image is an
   omitted flag.
3. **Edit is generation with reference conditioning** — family-derived, no
   `--mode`/`--ref-mode`. qwen-edit routes to the edit pipeline; unsupported
   families take the drop path (interactive: loud warn + proceed).
4. **Ingestion caps** (decision 6): read-once ≤ 64 MB, decompressed-pixel cap,
   `formats=["PNG","JPEG","WEBP"]`, first frame only, SHA-256 over decoded
   bytes, errors name path+reason and never echo file content.
5. **Row-1 trust** (decision 7): a typed `--ref-image` is user authority — no
   containment gate, loud echo if outside known roots.

## Invariants (must always be true)

- **Bad `--ref-image` → exit ≠ 0 with the offending value named; no GPU
  touched, nothing written.** Unknown `MODE`, > 8 references, and a
  mode-stripped path that does not exist (while the full spec does) each name
  the specific fault.
- **Every reference decoded goes through `comfyless/ref_image.py`** — there is
  no second decode site in this slice. Oversize (byte cap) and
  decompression-bomb (pixel cap) are rejected **before** full decode; a
  non-allowlisted format is rejected by the allowlist, not by later failure.
- **Ingestion error strings contain the path and a reason, never file bytes**
  — a co-located sidecar JSON fed as a ref fails decode without leaking its
  contents.
- **SHA-256 is computed over the exact bytes decoded** (single-read design) —
  no check-then-use window between validation, hash, and decode.
- **`MODE` maps deterministically to flags:** `both`→`vl=T,ref=T`,
  `vl`→`vl=T,ref=F`, `ref`→`vl=F,ref=T`; the Nth `--ref-image` becomes
  "Picture N" (order preserved, `manual_loop.py:2409`).
- **A typed reference path outside `ref_image_roots` runs in-process** (skip
  daemon delegation — the `--output` precedent, `generate.py:2450`), so row-1
  user authority holds regardless of whether a daemon is up.
- **Pipeline class is selected once, at load, in `detect_pipeline_class`**
  (ADR-035 decision 3) — ref-image presence never triggers a `from_pipe`
  upgrade. (The daemon cache-key *test* pinning this lands with the daemon
  slice; the constraint binds here.)
- **Non-edit code paths are unchanged** — a `generate` invocation with no
  `--ref-image` behaves exactly as today; existing generate tests stay green.
- `local_files_only=True` everywhere.

## Failure semantics

Fail-closed at ingestion, warn-and-proceed at family mismatch (interactive
CLI only, per ADR-035 decision 2): any cap/format/parse violation exits nonzero
before generation with the fault named; a reference handed to a family that
cannot consume it emits a loud warning naming the family and proceeds without
that reference. No partial writes — output is produced only after all
references pass ingestion.

## Out of scope

Daemon path (`server.py`, `ref_image_roots`, wire strictness, NUL defense,
cache-key test), sidecar recording + replay trust, MCP `edit` surface,
`flux2klein`/other-family img2img execution, `vae_target_size`, batch
generation, ComfyUI nodes. `flux2klein` and other families parse `--ref-image`
(slice 1) but their *execution* is not wired here — a non-qwen-edit ref request
takes the documented drop path.

## Negative-case tests (minimum)

- Unknown `MODE` (`car.jpg:blah`) → rejected, value named, no GPU.
- 9 `--ref-image` flags → rejected, cap named.
- Colon filename without explicit mode (`frame:ref` where `frame` absent but
  `frame:ref` exists) → error names the full-spec file.
- Oversize file (> 64 MB) → rejected before full decode.
- Decompression bomb (small file, huge declared dimensions) → rejected by pixel
  cap before decode.
- Non-allowlisted format (e.g. a `.tiff` or a Ghostscript-triggering `.eps`) →
  rejected by the format allowlist.
- Sidecar JSON fed as `--ref-image` → decode fails, error carries no file bytes.
- SHA-256 determinism: same bytes → same hash; one-byte change → different hash.
- `both`/`vl`/`ref` → correct `(vl_flags, ref_flags)` per image.
- No `--ref-image` → generate path byte-identical to pre-slice behavior.

## Proof hooks

- `./.venv/bin/python3 test_params_schema.py` — flag parsing, `MODE`
  validation, ref-count cap, colon escape (slice 1).
- `./.venv/bin/python3 test_ref_image.py` (new) — bomb/oversize/format/hash/
  error-string ingestion cases (slice 2).
- `./.venv/bin/python3 test_manual_loop.py` — `MODE`→flags mapping, routing,
  output-dim derivation (slice 3), plus existing suites green.
- `python -m py_compile comfyless/generate.py comfyless/ref_image.py`
- Live smoke (**Grant**): a keyframe pair — `--ref-image kf.png:both` locks
  geometry; `--ref-image car.jpg:vl` re-orients. Verify by eye.

## Edit scope (hard)

`comfyless/generate.py`, `comfyless/params_schema.py`,
`comfyless/params_validation.py`, new `comfyless/ref_image.py`, new
`test_ref_image.py`, `test_params_schema.py`, `test_manual_loop.py`, this
Vision doc (+ vault mirror), ADR-035 Changelog append. Nothing else — the
daemon and replay slices are separate edit scopes.

**Lens:** team-portable — argparse convention (`--lora` mold), §12 ingestion
caps (refine.py mold), and family-derived routing all follow existing repo
conventions; nothing solo-specific.
