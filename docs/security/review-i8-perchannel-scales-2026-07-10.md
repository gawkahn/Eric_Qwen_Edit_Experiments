# Security review — int8-tensorwise per-output-channel scales (reqs 57-60)

AI-Disclosure: Claude (Opus 4.8) authored the change and ran the review; reviewer
agent `security-auditor` pinned to Opus; Grant reviewed.

**Date:** 2026-07-10
**Surface:** `nodes/eric_diffusion_fp8_ops.py` — caller-supplied weight-file content
parser (§12 trigger: custom parsing of file CONTENT fed into compute ops).
**Change under review:** widen the `ci-w` (int8-tensorwise) `weight_scale` gate from
scalar-only to scalar-or-per-output-channel `(rows, 1)`. Implements reqs 57-60 of
the Amendment 2026-07-10 in `review-slice-I8-int8-tensorwise-2026-07-08.md`.
**Verdict:** **No findings at CRITICAL/HIGH/MEDIUM.** One [INFO], no remediation
required. Shipped after one docstring correction raised by `code-reviewer`.

---

## Why the change exists

`greed_int8.safetensors` (community Krea checkpoint, 170 quantized Linears) refused
to load. Investigation showed the file is pristine and **our requirement was wrong**.
`int8_tensorwise` names comfy-kitchen's *layout class*, not the scale granularity.
Verified against `comfy-kitchen==0.2.18`, the version pinned by ComfyUI master's
`requirements.txt` (PR #14636 has merged, so this is shipped upstream behavior):

- `TensorWiseINT8Layout.quantize(..., per_channel: bool)` — *"If True and is_weight,
  use per-channel (row-wise) scaling."*
- `quantize_int8_rowwise` returns *"scales: Float32 tensor `[..., 1]` with per-row scales."*
- `dequantize_int8_simple(q, scale) -> q.float() * scale`
- `requantize_kwargs`: `per_channel = bool(is_weight and (convrot or params.scale.dim() > 0))`
- Upstream `ops.py` performs **no** shape validation on the popped `weight_scale`.

Our dequant was already numerically identical to upstream's; only the gates were
over-tight. The original req 50 generalized finding F3 (an *fp8* finding: no
per-channel scales) onto the int8 flavor, where it does not hold.

## Threat model (unchanged across this review chain)

Same-uid solo-desktop operator loading a community checkpoint they chose. safetensors
is unpickled and mmap-bounded. Ceiling = silent numerical corruption of the operator's
own generation, not RCE. Widening the shape allowlist adds no capability — the bytes
were always going to be multiplied into weights. The only question is whether the
multiply is row-correct.

## The hazard this had to contain

A `(rows, 1)` scale broadcasts row-wise (correct). A `(1, in_features)` scale
broadcasts **column-wise, silently, with no torch shape error** — wrong semantics,
corrupted weights. A bare `(rows,)` scale broadcasts along the *last* axis. Both are
exactly the silent-corruption class F2/F3 exist to prevent. Hence the shape is pinned
exactly to `{(), (1,), (rows, 1)}` rather than admitting any `numel() > 1` vector.

## Auditor findings (full output)

**No findings at CRITICAL/HIGH/MEDIUM. The change is sound.**

**1. Shape allowlist is airtight.** `_is_ci_scale_shape` accepts exactly
`{(), (1,), (rows, 1)}`. Enumerating every scale shape torch broadcasts against a
`(rows, in)` weight without raising:

| shape | disposition |
|---|---|
| `()`, `(1,)` | numel 1, uniform scalar, always row-correct — accepted |
| `(rows, 1)` | row-wise, correct — accepted |
| `(1, in)` (column-broadcast; the core hazard) | rejected for all `in > 1`; equals `(rows,1)` only when `rows==1 and in==1`, where it IS the single-channel scalar and is correct |
| `(rows,)` (last-axis broadcast) | never in allowlist — always rejected |
| `(in,)`, `(rows, in)`, `(1,1)` for `rows>1`, all ≥3-D | rejected |

Degenerate cases confirmed: `rows==1`, `in==1`, `rows==in` (square), `(1,1)`, and
`(rows,1)` with `rows` coincidentally equal to `in` all resolve to either
accepted-and-row-correct or loud reject. `(1,1)` on a multi-row weight is rejected
(fail-closed over-strictness, not a security issue).

**2. Two-point enforcement holds; no int8 reaches pass-through.** Classify (header
`w_shape[0]`) and load (actual `t.shape[0]`) call the same predicate on the same
file — they cannot disagree. If `load_scaled_fp8_component` is reached with
`variant="ci-w"` while bypassing classify (it takes `variant` as a param), the loader
independently re-asserts the full ci-w ruleset. The unconditional guard raises on ANY
`torch.int8` tensor reaching step-2 for ANY variant, so the pre-existing silent
integer→float `copy_()` channel (req 49) stays closed. An int8-typed `weight_scale`
is rejected by dtype at both gates.

**3. No fp8 gate loosened.** `_validate_scale` keeps `allowed_dtypes=(torch.float32,)`
default and the `numel() != 1` scalar-only check; all three call sites (incl. the DMR
requant) use the default, so a `(rows,1)` scale on any fp8 path is rejected by the
numel check — **F3 stands**. `_F32_MIN_NORMAL` hoisted to module scope is
byte-identical to the former local (`1.1754943508222875e-38` = 2⁻¹²⁶); no behavior
change. `_validate_ci_scale` is called only on the ci-w path.

**4. Value check is trap-free and upcast-ordered correctly.**
`bad = ~(torch.isfinite(v) & (v >= _F32_MIN_NORMAL))` rejects NaN (isfinite False),
±Inf (isfinite False), 0.0/negatives (below floor), and subnormals (below floor) for
every element. `NaN >= x` returns False rather than raising — no comparison trap.
Dtype is validated on the RAW tensor before the f32 upcast (I8-5 satisfied). bf16 and
f32 share the same 8-bit exponent range, so bf16 subnormals remain below the floor
after upcast and are caught.

**5. Error-message hygiene preserved.** Every key/name interpolation in the new code
passes through `_safe_name()`. The one raw value echoed is a numeric scale value, not
an attacker-controlled key name, consistent with the pre-existing `_validate_scale`
echo.

**[INFO] Finite-but-huge scale can still produce ±Inf dequantized weights.**
`_validate_ci_scale` bounds each element to finite-positive-normal but not its
magnitude; a large finite scale times int8 values can overflow to ±Inf in the f32
dequant product — self-inflicted corruption of the operator's own output.
*Remediation: none required.* Explicitly accepted in finding I8-10 as identical to the
shipped fp8 posture (F2 never capped magnitude) and consistent with the threat model.
Noted for completeness.

**Reqs 57-60 assessment:** nothing identified as wrong. The shape rule pins exactly
rather than admitting `numel()>1` (57); fp8 paths are provably untouched via the
separate validator (58); the value policy is genuinely elementwise (59); `convrot`
stays refused via the unchanged strict `{"format"}` allowlist (60).

## Code-review findings (`code-reviewer`, Opus)

**Approved after one fix.** Promise drift: the `_classify_ci` docstring still read
`descriptor + scalar {BF16,F32} weight_scale` — the exact rule req 57 widened.
Corrected to `{BF16,F32} scalar-or-(rows,1)` before commit. One optional coherence nit
(disambiguate the surviving bare `req 50` citations as the *dtype* clause) also
applied. Reviewer explicitly endorsed keeping `_validate_ci_scale` separate from
`_validate_scale` rather than sharing: "the stated intent — a future int8 edit cannot
loosen an fp8 call site — is sound and worth the ~15 duplicated lines."

## Net security posture

This change makes us **stricter than upstream ComfyUI**, which validates scale shape
not at all. Two previously-silent corruption paths — column-broadcast `(1, in)` scales
and a poisoned single row scale in an otherwise-valid vector — are now loud rejects
that ComfyUI itself does not have.
