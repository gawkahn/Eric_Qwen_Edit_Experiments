# Slice NF4 — bitsandbytes NF4 single-file consumption — Vision

**Date:** 2026-07-17 · **Author:** Claude (Fable 5) · **Approved:** Grant (session directive 2026-07-17: "go do the loader slice", naming the two projectGaia files)
**Implements:** [ADR-019](../decisions/ADR-019-native-quantization-support.md) — the deferred bnb-NF4 single-file path. TECH_DEBT trigger ("projectGaia becomes a model Grant actively wants to run") FIRED 2026-07-17: `projectGaiaFlux1D_v20NF4UNETUncensored.safetensors` (transformer-only) and `projectGaiaFlux1D_v20NF4Uncensored.safetensors` (AIO) are the only released versions of the model.

---

## 1. Vision

**Outcome when done:** `--transformer <nf4-file>` on a Flux.1 base loads by
dequantizing bnb-NF4 to bf16 IN MEMORY (pure torch, NO bitsandbytes
dependency, no Blackwell requirement) and feeding the existing single-file
materialization tail. Both file shapes work: UNET-only (bare
`double_blocks.*`) and AIO (same under `model.diffusion_model.*` beside
t5xxl/clip/vae subtrees, which are dropped — base-model components fill
those slots as usual). `--quant fp8|nvfp4` composes on top (dequant→bf16
→ quantize_module requant), giving the OOM-relief story back.

**Measured format (both files, header-verified 2026-07-17):** per quantized
param, four keys: `<name>.weight` U8 `[ceil(numel/2), 1]` packed nibbles;
`<name>.weight.absmax` F32 `[numel/64]` per-block scales (NOT
double-quantized — no `nested_absmax` in these files); `<name>.weight.
quant_map` F32 `[16]` embedded codebook; `<name>.weight.bitsandbytes__nf4`
U8 serialized quant-state (JSON: shape, blocksize, dtype). 314 quantized
weights; norms/bias/scale keys are plain F32.

**Invariants (each gets at least one negative test):**

1. **Existing variants byte-identical** — classification and loading of
   ca/cb/cc/cq-a/cq-w/ci-w files, and of plain bf16 files, are unchanged.
   The NF4 sniff must not fire on any of them (NEGATIVE: fp8 fixtures still
   classify as before with NF4 code present).
2. **Two-point enforcement (PQ-2 pattern)** — the header classifier
   detects NF4 by marker suffixes without reading tensor data; the loader
   re-validates everything it consumes (sizes, dtypes, quant-state JSON
   bounds, finiteness) and raises `ScaledFp8FormatError` with an
   actionable message on any violation. No tensor-content trust from the
   header stage.
3. **Decode correctness is proven numerically** — the nibble unpack order,
   codebook lookup, and per-block absmax multiply are verified against
   hand-computed expected tensors in tests (the old `dequantize_nf4.py` is
   KNOWN BROKEN — its logic is NOT ported; the decode is written fresh
   from the measured format). A wrong-order unpack or off-by-one block
   test MUST fail (NEGATIVE: perturbed fixture ≠ expected).
4. **Bounded parsing** — the quant-state blob is size-capped before JSON
   parse (existing 4096-byte descriptor cap pattern); required fields
   allowlisted; unknown/absent → loud reject. Double-quantized states
   (`nested_absmax` present) reject loudly as unsupported, never
   half-decode.
5. **AIO subtree selection is exact** — from the AIO file, ONLY the
   transformer subtree is materialized; `text_encoders.*` / `vae.*` /
   other non-transformer roots are dropped before the dequant stage and
   never reach the model builder (NEGATIVE: bundled TE keys absent from
   the handed-over state dict).
6. **Fail loud, fail early** — a malformed NF4 file (missing absmax,
   size-mismatched packed data, non-finite codebook) raises at
   classify/load time with a message naming the defect; it must NOT fall
   through to the noisy standard-path failure chain that NF4 files hit
   today.

**Out of scope:** bnb int8 (`.SCB`) files; double-quantized NF4 (nested
absmax) — reject loudly; keeping weights NF4-resident at compute time (no
kernel exists in-repo; dequant-to-bf16 is the whole point); NVIDIA nvfp4
single-file (`weight_scale_2` layout — separate deferred item, unchanged);
non-transformer NF4 components (text encoders etc. — the two real files
only quantize the diffusion transformer); writing dequantized files to
disk (in-memory only).

## 2. Change boundary / edit scope (hard)

May change: `nodes/eric_diffusion_fp8_ops.py` (**Red Zone —
security-auditor gate**: classifier branch + NF4 dequant stage + subtree
filter), `nodes/eric_diffusion_utils.py` (route the new variant in
`_load_single_weights`; update `_diagnose_slot_mismatch` check-1a wording
if it would now mislead), `test_fp8_single_file.py`, `docs/vision/` (this
doc), `docs/decisions/ADR-019-*.md` (Changelog append), `TECH_DEBT.md`
(NF4 entry Resolved append), `docs/security/review-slice-NF4-*.md` (new),
`audit_single_files.py` (verdict text: BNB → supported note). Anything
else → STOP and split. `comfyless/generate.py` should need NO edits (the
variant routing lives below `load_component`); if that proves wrong, stop
and amend this doc first.

## 3. Design (condensed)

- **Classifier:** new marker sniff in `classify_fp8_single_file` — fires
  when header keys contain the `.weight.quant_map` + `.weight.absmax` +
  `.weight.bitsandbytes__nf4` (or `__fp4`) suffix family → returns
  `("bnb4", info)`. Runs BEFORE the fp8-dtype short-circuit (NF4 files
  carry no fp8 dtypes and today return `(None, {})`). `.SCB` (bnb int8)
  stays unrecognized → existing late diagnostic.
- **Loader:** new branch in `load_scaled_fp8_component` for `"bnb4"`,
  mirroring ci-w's unconditional-dequant posture (`dequant_fp8` flag
  irrelevant — there is no NF4 residency). Per quantized param: bounded
  quant-state parse → validate (blocksize>0, shape numel == absmax·blocksize
  coverage, packed byte count == ceil(numel/2), codebook len 16 finite) →
  unpack nibbles (bnb order: first element in HIGH nibble — verified
  numerically in tests) → codebook lookup → reshape to [nblocks, blocksize]
  → × absmax → reshape(quant-state shape) → cast to target dtype. Marker
  keys consumed; plain keys pass through with the existing `t.to(dtype)`.
- **AIO:** for the `"bnb4"` variant, after prefix strip, drop keys under
  known non-transformer roots (reuse the krea2 `_NON_TRANSFORMER_ROOTS`
  concept as a shared/duplicated tuple in fp8_ops) before dequant. The
  measured AIO file's `model.diffusion_model.` prefix is 72% dominant, so
  the existing peek/strip mechanism engages; the filter guarantees
  invariant 5 regardless.
- **Tail:** converge on the existing "bf16 dict → `from_single_file(dict,
  config=base)`" materialization — no new model-build code.
- **`--quant` composition:** untouched — generate.py already requants
  override components via `quantize_module` after load.

## 4. Build order (each a commit; conventional prefix; both trailers)

1. `docs:` this Vision + ADR-019 Changelog append (spec-first).
2. security-auditor (Fable) DESIGN review of this doc + the measured
   format → `docs/security/review-slice-NF4-bnb-single-file-2026-07-17.md`;
   binding requirements folded into step 3.
3. `feat:` classifier + loader + subtree filter + tests; code-reviewer
   (Fable) on the whole diff; auditor delta if scope moved.
4. Live proof: both projectGaia files generate on the Flux.1-dev base
   (Grant), plain and with `--quant fp8`. THEN TECH_DEBT NF4-entry
   Resolved append (a resolution recorded only after it is true).

## 5. Proof

- `just tests` battery green; NF4 tests are CPU-only (tiny synthetic
  fixtures via the `_mk` house pattern; `_expect_reject` negatives citing
  review findings; `_CaptureComp` asserts the handed-over bf16 dict).
- Numeric ground truth: fixture weights built by ENCODING known bf16
  values (codebook index + absmax chosen by hand), asserting decode
  round-trip equality — not by trusting the decoder under test.
- Live gate: projectGaia UNET + AIO both generate; header notice names the
  variant and quantized-param count.

## 6. Acceptance

- [ ] Battery green; every §1 invariant has a passing negative.
- [ ] security-auditor review saved + referenced in the feat commit body.
- [ ] code-reviewer run, findings addressed.
- [ ] TECH_DEBT NF4 entry Resolved; audit_single_files.py verdict updated.
- [ ] Live: both projectGaia files load and generate (Grant's gate).

AI-Disclosure: Claude (Fable 5) authored; Grant approved the build direction 2026-07-17.
