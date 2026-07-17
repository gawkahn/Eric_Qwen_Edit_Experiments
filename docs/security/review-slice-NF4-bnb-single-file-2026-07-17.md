AI-Disclosure: security-auditor (Claude Fable 5) authored the review; Claude (Fable 5) transcribed; Grant reviews at merge. Requirements 67-84 (MUST) + 85-90 (SHOULD) are binding on the slice NF4 implementation.

# Design-phase security review — ADR-019 slice NF4 (bnb-NF4 single-file consumption)

Reviewed design: `docs/vision/slice-NF4-bnb-single-file.md` (2026-07-17).
Implementation note (2026-07-17, post-review): the real marker key carries a
`.quant_state.` segment — `<name>.weight.quant_state.bitsandbytes__nf4` —
the requirements' `.weight.bitsandbytes__nf4` shorthand refers to that
measured form.

## Summary

The slice adds a fourth quant-flavor branch to the repo's most exercised untrusted-content parser: `classify_fp8_single_file` gains a marker sniff (`bnb4`), and `load_scaled_fp8_component` gains an unconditional-dequant branch (nibble unpack → codebook lookup → per-block absmax multiply → bf16), plus an AIO subtree filter, converging on the existing materialization tail. The trust boundary is unchanged in kind but widened in surface: a fully attacker-controlled .safetensors file (civitai provenance) now drives a **second bounded JSON parse** (the bnb quant-state blob — the module docstring currently asserts "EXACTLY ONE bounded exception", which this slice breaks and must amend), a **new arithmetic decode path** whose inputs (codebook, absmax, blocksize, shape) are all attacker values, and a **new key-space transformation** (subtree filter stacked on prefix strip) that creates aliasing opportunities the existing variants don't have. The threat model worked: malicious/corrupted file → classify → load → dequant → `from_single_file` tail; failure modes sought were allocation bombs, silent numeric corruption (NaN/Inf/wrong-scale weights that generate garbage without erroring), key aliasing that swaps or smuggles tensors, downgrade/misroute of existing variants, and uncaught-exception classes escaping the `ScaledFp8FormatError` contract.

Checked: the full classifier chain and its ordering (entry gate at fp8_ops.py:171, control-char at :178, nvfp4 reject at :193, cq dispatch at :205), the `_classify_cq`/`_classify_ci` bounded-parse precedents (:339-391, :595-632), the loader's pairing/stage-2/tail structure (:855-1156) including the two spots the bnb4 branch must interact with (the prefix-strip dict comprehension at :860, which is **silently last-wins on collisions**, and the stage-2 non-float raw pass-through at :1049, which would forward stray U8 tensors into `from_single_file` unrefused), the routing caller (`eric_diffusion_utils.py:726-848`, dominant-prefix peek + variant dispatch), and the krea2 subtree-selection precedent (`eric_krea2_convert.py:194-220` — an **allowlist** extraction, not a denylist drop; the vision doc proposes the weaker of the two). One latent defect found in existing code (uncaught `RecursionError` from deep-nested descriptor JSON) applies with full force to the new parse and is a binding requirement for it.

## Coverage

Reviewed: the vision doc (full); `nodes/eric_diffusion_fp8_ops.py`:1-1176 (docstring/posture, `classify_fp8_single_file`, `_classify_cq`, `_classify_ci`, `_validate_scale`/`_validate_ci_scale`, `load_scaled_fp8_component` incl. ci-w branch, stage-2 tail, residency swap); `docs/security/review-slice-Cd-comfy-quant-2026-07-02.md` (full — bounded-descriptor precedent, reqs 11-20); `docs/security/review-slice-NV-nvfp4-merge-guard-2026-07-16.md` (numbering: chain ends at req 66); `nodes/eric_krea2_convert.py`:180-280; `nodes/eric_diffusion_utils.py`:726-855.

Not reviewed: the two real projectGaia files (measured-format claims taken from the vision doc as stated assumptions); bitsandbytes source (not a dependency — claims about bnb serialization conventions are named as assumptions where they bear on requirements); `comfyless/generate.py` `--quant` composition (declared no-edit).

## Findings

**[HIGH] Prefix-strip is last-wins; the subtree filter stacks a second aliasing layer on it**
Location: nodes/eric_diffusion_fp8_ops.py:860
Risk: an AIO file carrying real `model.diffusion_model.double_blocks.N...weight` families AND a top-level evil `double_blocks.N...weight` (or `...weight.absmax`) collides after strip; the dict comprehension silently keeps one, letting a crafted tensor shadow a legitimate one — or mispair an absmax with a different base's weight — with no error. The bnb4 branch multiplies the alias surface (four keys per family).
Remediation: req 77.

**[HIGH] Stage-2 tail passes non-float tensors through raw**
Location: nodes/eric_diffusion_fp8_ops.py:1049
Risk: a bnb4 file carrying a stray marker-less U8 tensor (or a smuggled packed weight whose marker was dropped by the subtree filter) would flow raw into `from_single_file` → `copy_()` casts packed nibbles into a float param as integer garbage, silently. The int8 refusal at :1043 covers I8 only.
Remediation: req 79.

**[MEDIUM] Denylist subtree filter is weaker than the krea2 precedent it cites**
Location: design (vision doc §3 "AIO"); precedent at nodes/eric_krea2_convert.py:212-220
Risk: `_NON_TRANSFORMER_ROOTS` is a fixed denylist; a bundled component under an unlisted root (`te.`, `clip_l.`, `text_model.`) is NOT dropped and reaches the model-build tail, whose behavior on unknown keys is converter-dependent. Invariant 5 as worded is an allowlist claim being implemented as a denylist.
Remediation: reqs 77/78 floor; req 85 upgrade. Verdict (b).

**[MEDIUM] Deep-nested descriptor JSON raises uncaught `RecursionError` — pre-existing, inherited by the new parse**
Location: nodes/eric_diffusion_fp8_ops.py:348, :605 (existing); new bnb4 quant-state parse (planned)
Risk: ~2000 bytes of `[[[[…` inside a ≤4096-byte blob exceeds the recursion limit inside `json.loads`; `RecursionError` is not caught by `except (UnicodeDecodeError, ValueError)` — classification crashes with a raw traceback instead of the contracted `ScaledFp8FormatError`.
Remediation: req 70 (binding for the NF4 parse); req 87 (existing sites — separate slice).

**[MEDIUM] Attacker JSON must never resolve dtypes or drive behavior beyond shape/blocksize**
Location: design (quant-state fields `dtype`, `quant_type`, future fields)
Risk: the obvious implementation of the `dtype` field is `getattr(torch, state["dtype"])` — arbitrary attribute access on the torch module from attacker strings; any behavior keyed on `quant_type` reintroduces the D5 class.
Remediation: req 71 scope lock.

**[INFO] Memory amplification is bounded once coverage equations hold** — worst-case dequant allocation is ~4× the packed bytes actually present (plus transient f32 ~8×), linear in file size, PROVIDED allocations are sized only after req 72's equations pass. The requirement ordering is load-bearing.

**[INFO] Scope check** — the vision doc's edit scope matches what the design needs; no scope creep detected. `generate.py` no-edit claim is consistent with the routing read.

## Requirements contract (continuing from req 66)

Assumption named once, applying throughout: the measured-format claims (four-key families, blocksize 64, F32 absmax `[numel/64]`, no nested absmax, packed `[ceil(numel/2), 1]`) are per the vision doc's 2026-07-17 header measurement; requirements are written to *verify*, not trust, each at load time.

### MUST (binding)

**67. Trigger is the bitsandbytes marker suffix ONLY.** The bnb4 sniff fires iff at least one header key ends with the `bitsandbytes__nf4` or `bitsandbytes__fp4` marker suffix. Presence of `.absmax` / `.quant_map` suffixes alone must never fire it. Negative tests: (a) a file with `.absmax`+`.quant_map` keys but no marker classifies exactly as today; (b) the full existing fixture battery — ca, cb, cc, cq-a, cq-w, ci-w, plain bf16, control-char reject, nvfp4 `weight_scale_2` reject — classifies byte-identically with the NF4 code present (invariant 1).

**68. Flavor mutual exclusion — hybrids reject, never branch-win.** A file containing any bnb4 marker together with ANY of: an fp8-dtype tensor, an I8 tensor, a `.comfy_quant` key, a ca/cb scale-suffix key, the `scaled_fp8` marker, or a `.weight_scale_2` key → `ScaledFp8FormatError` at classify, naming one key from each side. `.SCB`-marked bnb-int8 files with no 4-bit marker remain unrecognized (`(None, {})` → standard path) — negative test.

**69. Header-time family completeness (two-point stage 1).** For every base `B = <name>.weight` derived from a marker key: the header must contain `B` with dtype U8; `B + ".absmax"` F32 1-D; `B + ".quant_map"` F32 shape `(16,)`; and the marker blob itself U8, 1-D, `1 ≤ numel ≤ 4096`, checked BEFORE any tensor read. Rejections name base + defect. Dangling members reject. A base carrying both `__nf4` and `__fp4` markers rejects as ambiguous. Classification validates ALL marker families header-wide, including families under roots the loader will later drop.

**70. Bounded quant-state parse (load stage).** Re-assert U8 / 1-D / ≤4096 on the actual blob tensor before decode. Strict UTF-8 decode; `json.loads` wrapped so `UnicodeDecodeError`, `ValueError`, AND `RecursionError` all convert to `ScaledFp8FormatError`. JSON root must be a dict; non-dict roots each reject (list, scalar, null, trailing garbage, empty tensor).

**71. Strict field validation + scope lock.** Required fields with type checks that EXCLUDE bool: `shape` — list of 1-8 entries, each an int ≥ 1; `blocksize` — int ≥ 1. Any `nested_*` field (or nested-absmax-style tensor keys) → loud reject "double-quantized NF4 unsupported" — never half-decode. If `quant_type` is present it must literally equal `"nf4"` / `"fp4"` AND match the marker suffix; mismatch rejects. The `dtype` field, if read at all, is validated against a literal allowlist and DISCARDED — never resolved via `getattr(torch, ...)`, never selects the output dtype. Scope lock, docstring-recorded: the JSON drives exactly two consumed values — shape and blocksize; reading any further field for behavior requires a fresh security review.

**72. Coverage equations before any allocation sized by declared shape.** With `numel = prod(shape)`: `packed.numel() == ceil(numel/2)` and `absmax.numel() == ceil(numel/blocksize)`, both checked against the ACTUAL tensors before the decode allocates anything proportional to `numel`. Negative tests: declared numel ≠ 2×packed (both directions); huge declared shape over a tiny packed tensor; absmax off-by-one.

**73. Odd-numel padding is deterministic and inert.** `ceil` semantics; decode takes exactly the first `numel` unpacked values; the padding nibble's content must not influence the output (two fixtures differing only in the pad nibble decode equal).

**74. Codebook indices safe by construction; nibble order numerically proven.** Unpack via bitwise ops on the uint8 tensor (`b >> 4` FIRST element, `b & 0xF` second — bnb high-nibble-first), no signed cast before masking, indices provably in [0, 15]. Tests: byte `0xFF` → (15, 15); the ground-truth fixture uses an order-ASYMMETRIC pattern such that a low-nibble-first implementation fails.

**75. Codebook validation.** Exactly 16 elements, every element finite, `|v| ≤ 1.0`. NaN, Inf, wrong length, |v|>1 each reject.

**76. absmax validation: finite and ≥ 0; zero is LEGAL.** NaN/Inf/negative reject naming the first bad index. Zero accepted (all-zero block). This deliberate divergence from the F2 finite-positive-normal rule (unchanged for fp8/int8 scales) is documented at the check.

**77. Post-strip key-collision reject.** The bnb4 branch detects two distinct source keys mapping to the same post-strip name and rejects — never dict-comprehension last-wins. Negative test: prefixed real key + evil unprefixed twin.

**78. Subtree filter: bnb4-only, atomic, pre-dequant.** The non-transformer-root drop runs ONLY on the bnb4 branch (invariant-1 negative: an fp8 fixture with a `text_encoders.` key loads exactly as today). Full-key root-prefix matching; a quantized family drops atomically. Dropped keys never reach dequant or the model builder (invariant-5 negative via `_CaptureComp`).

**79. Residual non-float refusal in the bnb4 stage-2 tail.** Any remaining non-floating-point tensor rejects. Negative test: marker-less stray U8 tensor rejects instead of materializing.

**80. Unconditional dequant, zero residency.** bnb4 returns after the materialization tail regardless of `dequant_fp8`; never reaches the fingerprint/residency swap; all four family keys consumed (test asserts key absence + all-float dtypes).

**81. Two-point enforcement (PQ-2).** Every classify-time gate (67-69) re-asserted at load, so a direct `load_scaled_fp8_component(..., variant="bnb4")` call on a non-conforming file rejects without the classifier's help.

**82. Error + log hygiene.** Every rejection is `ScaledFp8FormatError`, actionable, key named via `_safe_name`; raw JSON never echoed beyond `_safe_name` truncation. One aggregate header notice (variant, marker flavor(s), quantized-param count); no per-layer lines.

**83. Docstring amendment.** The module docstring's "EXACTLY ONE bounded exception" claim amended to enumerate the SECOND bounded read (bnb quant-state blob) with its gates (69/70) and scope lock (71). Vision invariant-2 wording carries the same amendment.

**84. Test-coverage binding.** Every MUST gets at least one negative citing its req number in `test_fp8_single_file.py`, plus the invariant-1 regression battery of 67(b). Fixtures built by ENCODING known values, including one `bitsandbytes__fp4`-marker fixture.

### SHOULD (advisory)

**85.** Upgrade the subtree filter toward the krea2 allowlist pattern (extract by known Flux transformer roots) or keep the denylist with a loud notice listing surviving unknown roots.
**86.** Bound blocksize beyond >0: pin to measured 64 or power-of-two ≤ 4096.
**87.** Backport the `RecursionError` catch to `_classify_cq`/`_classify_ci` (pre-existing gap — separate trivial slice, not a drive-by).
**88.** Name the failed-prefix condition (bnb4 AIO reaching the tail still-prefixed) rather than letting the converter's KeyError surface.
**89.** Pin the packed shape-form `[N, 1]` (measured layout); widen with a fixture if a flat variant appears.
**90.** Peak-RSS courtesy: decode per-param, `del` consumed source tensors as you go.

## Explicit verdicts

**(a) nf4 + fp4 marker widening — APPROVED, conditioned** on the quant_type↔marker cross-check (71), an fp4-marker fixture (84), and the flavor-naming header notice (82) — fp4 acceptance rests on the named assumption that bnb fp4 serialization is structurally identical, which load-time validation verifies per-file rather than trusts.

**(b) subtree filter — denylist-drop APPROVED as floor,** given bnb4-only + atomic drop (78) + collision reject (77). Residual gap (smuggling under an unlisted root) bounded to "extra keys reach a non-strict converter"; req 85 recommends the allowlist upgrade. Shipping without 77/78/79 would not be approved.

**(c) classifier ordering — SAFE, entry condition load-bearing.** Marker-suffix-only trigger (67) means no existing family can fire the sniff; hybrids reject rather than branch-win (68), so sniff placement affects only which loud error a crafted hybrid gets — both orders fail closed. The one real behavior change is confined to bnb-marked files (today: `(None, {})` + noisy late death; after: classify then load-or-reject with a named defect). No downgrade path for existing variants provided 67(b)'s regression negatives are in the battery.

---

# Addendum (2026-07-17) — delta ruling on req 68 scoping

Trigger: the real AIO file carries its bundled T5-XXL in fp8 beside the NF4
transformer — req 68's whole-file hybrid exclusion rejected the exact file
the slice exists to load (the UNET-only file classified clean).

**Ruling (1): APPROVED — scope the req-68 hybrid scan to keys outside the
exempt roots,** with two binding conditions: (i) the scan-exempt list and
the req-78 drop list are the SAME constant; (ii) a load-time purity gate on
the surviving dict (req 81 amended) backs the exemption. Condition (ii)
closes a real alias: `first_stage_model.` / `cond_stage_model.` are both
exempt roots AND dominant-prefix candidates — a crafted dominant prefix
could strip an exempt subtree INTO the transformer namespace if the filter
ran post-strip.

**Ruling (2): marker families under droppable roots stay
validated-but-dropped** (req 69 unchanged — header-only, zero tensor
reads); the req-82 notice reports the SURVIVING count, with one aggregate
line for dropped-root families.

**Ruling (3): the `.SCB` negative is unchanged**; I8/`.SCB` tensors under
exempt roots in a bnb4 file are tolerated-and-dropped (positive test);
under the transformer namespace they still reject.

**Amended texts:** req 68 — scan scoped to non-exempt keys; PRECEDENCE:
any bnb4 marker present ⇒ bnb4-or-reject, never fall through to
cc/ca/cb/cq (the real AIO would otherwise classify "cc" via its fp8 TE).
Req 78 — subtree filter runs on RAW keys BEFORE the prefix strip (the
alias closure); same constant as the scan exemption. Req 81 — loader
re-runs the flavor scan on the surviving dict (authoritative copy).
Req 82 — surviving-family count + aggregate dropped-family line.

**Req 84 additions:** AIO-shape positive (bnb4 classify + TE/fp8 absent
from the handed-over dict); in-namespace fp8/cq negatives; the
first_stage_model-as-dominant-prefix alias negative; the never-cc
precedence pin; `.SCB`-under-exempt-root positive; malformed-family-under-
exempt-root negative; constant-unity check.

All implemented and tested (test_fp8_single_file.py 239→291).

# Compliance verification (2026-07-17, code-reviewer, Fable)

Per-requirement compliance review of the implementation returned CHANGES
REQUIRED with 3 MED + 3 LOW findings, all folded same-day: (1) the delta
req-81 load-stage purity gate covered only the dtype half of the flavor
scan — the foreign-marker-KEY half added via a shared `_bnb4_foreign_marker`
helper used by both enforcement points; (2) two amended-MUST behaviors were
untested — direct-loader purity negatives (both halves) and the
dropped-family notice/surviving-count test added; (3) `audit_single_files.py`
omitted the new BNB4 verdict from its report order and its docstring still
said "unsupported" — both fixed; (4-6) LOW: contract sub-case negatives
added (RecursionError, scalar root, bool blocksize, packed-over-declared,
flat-[N] pin, in-namespace I8, dangling-absmax-beside-valid-family),
classify-comment overclaim fixed, vision invariant-4 wording reconciled
with the req-71 scope lock. Everything else verified SATISFIED, including
both real-file end-to-end loads (UNET 33 s / AIO 35 s, 314 families,
11.90B-param FluxTransformer2DModel, finite bf16 weights) and the decode
goldens against bitsandbytes 0.49.2.
