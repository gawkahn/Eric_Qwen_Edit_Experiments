# Vision slice — transformer audit (`kind:"transformer"` in lora_audit)

**Date:** 2026-07-05
**Risk:** L3 (reads caller-supplied directory trees + bounded content reads of caller-supplied weight files; NO new write/delete surface)
**Status:** approved (Grant directed the work 2026-07-05: "Finish the lora audit script to identify any completely incompatible loras and transformers, which will not be included")
**AI-Disclosure:** Claude (Fable 5) authored; Grant reviewed.

## What must be true when done

1. `scripts/lora_audit.py` accepts `--transformer-root DIR` (repeatable) and emits `kind:"transformer"` manifest entries for every `.safetensors` under those trees — the forward-compat hook ADR-014 §5 reserved.
2. Each transformer entry carries: a loading-prognosis verdict (reused from `audit_single_files.py`), a family hint, shape-fingerprint matches against the configured `--base` set, and a `duplicate_of` determination against those bases.
3. The four-class taxonomy (`usable` / `convertable` / `unconvertable` / `deletable`) extends to transformers with transformer-specific reason codes; **completely incompatible transformers (unsupported quant formats, no matching base) classify `unconvertable` and are thereby excluded from catalog candidacy.**
4. Byte-duplicates of hf-local diffusers bases are detected (bounded sampled reads) and marked `duplicate_of: <base name>` — the catalog excludes them (Grant: "Do not include comfyui single-file models that are simply duplicates of the diffusion models in hf-local").
5. The manifest change is **additive only** — `audit_version` stays 1; existing `kind:"lora"` consumers are unaffected (ADR-014 §5 forward-compatibility rule).
6. LoRA-side behavior is byte-identical when `--transformer-root` is not passed.

## What must never happen

7. **No writes or deletes under any `--transformer-root`.** `--convert` and `--delete` do not apply to transformer entries in this slice — `--delete` remains scoped to `deletable` LoRAs under `--audit-root` only.
8. No unbounded reads: per-file content reads are capped (header cap 100 MB inherited; sampled-tensor reads ≤ 1 MiB per tensor, ≤ 4 tensors per base comparison).
9. No path escapes: each `--transformer-root` gets the same authoritative realpath-descendancy containment + `O_NOFOLLOW` narrowing as `--audit-root` (ADR-014 §6), rooted at itself.
10. Determinism (ADR-014 invariant 8) survives: transformer entries sort into the same `files[]` array by `relative_path`, keyed with a `root` discriminator.
11. A transformer classified `deletable` is REPORT-ONLY (invariant 7 above); no promotion path exists.

## Proof hooks

- Extend `test_lora_audit.py`: transformer minting, verdict→classification mapping (one per reason code), duplicate detection positive + negative (same-shape different-content = NOT duplicate), no-write/no-delete negatives, containment negative, determinism re-run, additive-schema check (a v1 lora-only parser ignores transformer entries).
- Full 15-suite regression green.

## Negative cases required

1. `--delete --yes` with a garbage file under `--transformer-root` → file survives; manifest records it `deletable` + `deleted: false`.
2. Transformer root symlink escaping its own root → excluded + warning (mirrors ADR-014 N-cases).
3. bnb-NF4 single-file → `unconvertable`, reason `quant_unsupported_bnb`.
4. Shape-matched but content-differing finetune → `duplicate_of: null`.
5. dtype-mismatched (fp8 repackage of a bf16 base) → NOT a duplicate (distinct artifact).
6. `--transformer-root` missing/not-a-dir → startup abort exit 1 (fail-closed).
7. Manifest determinism: two runs byte-identical modulo `audited_at`.
8. `--transformer-root` nested under / containing / equal to `--audit-root`, or two overlapping transformer roots → startup abort exit 1 (security-auditor F-1: the nested case would let `--delete --yes` unlink a garbage transformer file classified as a LoRA).
9. Two transformer roots sharing a basename → entries distinguished by `root_index`; determinism holds (F-2).
10. Crafted header with `data_offsets` `hi < lo` or past-EOF in the duplicate sampler → bounded zero-length read, pair NOT byte-equal, no memory blowup (F-4).

## Out of scope

- Transformer dry-load (actually loading 130 × GB-scale files) — deferred to the live gen-validation slice; prognosis is header/shape-level here.
- Conversion of transformers (`--convert` stays LoRA-only).
- The catalog DB itself (next slice; consumes this manifest).
- text-encoder / VAE auditing (only the two transformer trees Grant named).
