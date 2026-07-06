# ADR-021: Transformer audit — `kind:"transformer"` entries in `scripts/lora_audit.py`

**Date:** 2026-07-05
**Status:** proposed (implementation gated by `code-reviewer` + `security-auditor`, both Opus — same L3 read-surface class as ADR-014)
**Risk:** L3 — reads caller-supplied directory trees + bounded content reads of caller-supplied weight files. **No new write/delete surface** (strictly narrower than ADR-014, which this extends).
**Related:** ADR-014 (LoRA audit tool — the manifest contract, containment policy §6, and the `kind` forward-compat hook this fills), ADR-019 (quant formats whose verdicts this reuses), ADR-018 (multi-root scan for the *serving* side; this ADR is the *audit* side of the same trees).
**Vision:** `docs/vision/slice-transformer-audit.md` (approved 2026-07-05).
**AI-Disclosure:** Claude (Fable 5) authored; Grant reviewed.

---

## Context

The LoRA-catalog service needs transformer (single-file checkpoint) candidacy decided the same way LoRA candidacy is: a machine-readable manifest naming what is usable against the hf-local diffusers bases, what is completely incompatible (excluded), and what is garbage. ADR-014 shipped the LoRA MVP and reserved `kind:"transformer"` as the forward-compat hook; its Deferred section named this as "next slice in this thread." Grant's catalog directive (2026-07-05) adds one requirement ADR-014 didn't anticipate: **byte-duplicates of hf-local diffusers models must be detected and excluded** — the comfyui trees hold single-file repackages of models that already exist as diffusers trees in hf-local.

Scale reality (measured 2026-07-05): 59 files/760 GB under `checkpoints/`, 71 files/1.4 TB under `diffusion_models/`. Full-content hashing is off the table; every mechanism below is header-only or bounded-sample reads.

The classification brain already exists: `audit_single_files.py` (223 lines, reviewed under ADR-019 slice C-d) produces per-file loading-prognosis verdicts (`BNB`/`SVDQ`/`NVFP4`/`CQ-*`/`AIO`/`SCALED`/`PLAINFP8`/`HI-PREC`) plus a family hint, header-only. This ADR reuses it — the audit tool does not fork a second definition of "loadable" (ADR-014 Alternative A discipline).

## Decision

### 1. CLI surface

- `--transformer-root DIR`, repeatable. Each root is validated at startup exactly like `--audit-root` (realpath `strict=True`, must be a directory, fail-closed abort). Scanned recursively with ADR-014 §6 containment (authoritative per-entry realpath-descendancy against *its own* root + `O_NOFOLLOW` narrowing).
- LoRA behavior without the flag is byte-identical (invariant 6). `--dry-load`, `--convert`, `--delete` continue to apply to LoRA entries only; transformer entries are read-classify-report ONLY in this slice.

### 2. Classification reuse

`audit_single_files.py` is loaded via `importlib.util.spec_from_file_location` (same pattern as the `nodes/` modules in `lora_audit.py`; it lives at the repo root and imports nothing heavy). Its `audit_file(path) -> {verdict, detail, family, n_fp8}` is the prognosis primitive.

Verdict → taxonomy mapping:

| Prognosis verdict | Classification | Reason code |
|---|---|---|
| `HI-PREC`, `SCALED`, `PLAINFP8`, `CQ-FP8` **and** ≥1 base shape-match | `usable` | `prognosis_<verdict-lowercased>` |
| `AIO` **and** ≥1 base shape-match | `usable` | `aio_bundle` (caveat: pipeline-fallback path, unvalidated until dry-load/gen-test) |
| `BNB`, `SVDQ`, `NVFP4`, `CQ-<other>` | `unconvertable` | `quant_unsupported_<marker>` |
| loadable prognosis but **zero** base shape-matches | `unconvertable` | `no_matching_base` |
| family `"?"` and zero base matches | `unconvertable` | `format_unknown` |
| garbage signatures (zero-byte / truncated / unparseable header) | `deletable` (report-only) | existing ADR-014 §3 reason codes |
| per-file classification raise | `error` | (ADR-014 fault isolation, exit 2) |

`convertable` is intentionally unreachable for transformers — there are no registered transformer conversion plans, and pretending otherwise would fabricate a capability.

### 3. Base shape-fingerprint matching

For each configured `--base name=path` (ADR-014 §4 — unchanged ingestion), the audit builds the base's **shape multiset**: the multiset of `(shape-tuple, dtype-class)` over the base transformer's params (`build_param_dict_from_dir`, header-only over the shards; dtype-class collapses {bf16, fp16, fp32} → `float` so precision variants match, while fp8/quant dtypes keep their identity).

A transformer file T **matches** base B iff `|dit-keys(T) ∩ shapes(B)| / |shapes(B)| ≥ 0.90` — where `dit-keys(T)` restricts to the DiT component for `AIO` bundles (the `model.diffusion_model.`/`model.model.` groups from `_AIO_GROUPS`) and is all keys otherwise, and shape intersection is multiset intersection. Name-agnostic by design: single-file ComfyUI key naming vs diffusers naming differs per family, but shapes don't. The 0.90 threshold tolerates missing/extra buffers (pos-embeds, norms) without admitting cross-family false positives (verified in tests against synthetic bases; different DiT families differ in layer shape inventories long before 90%).

Manifest records `matched_bases: ["<name>", ...]` (possibly several — e.g. a Qwen-Image finetune matches both `Qwen-Image` and `Qwen-Image-2512` if both are configured and shape-identical).

### 4. Duplicate detection (the exclusion Grant mandated)

For each matched base B, T is a **duplicate** of B iff all of:
1. Shape-match per §3 with overlap ≥ 0.999 (a repackage carries the whole tensor set, not 90%);
2. dtype-identical on compared tensors (an fp8 repackage of a bf16 base is **not** a duplicate — it is a distinct, independently useful artifact; Vision negative case 5);
3. **Sampled content equality:** take the K = 4 largest tensors of T whose `(shape, dtype)` occurs **exactly once** in both T and B (unique-shape pairing sidesteps key-name mapping entirely); for each, read the first `min(1 MiB, tensor_size)` bytes from both sides; all pairs byte-equal → duplicate. Any read error → **not** duplicate (fail toward inclusion; a false "duplicate" silently drops a real model from the catalog, a false "distinct" merely lists a redundant one).

If fewer than 2 unique-shape pairs exist, duplicate detection is **inconclusive**: `duplicate_of: null` + warning `dup_check_inconclusive` (fail toward inclusion, loudly — `feedback_warn_dont_block`).

Bounded IO: ≤ 4 tensors × 1 MiB × 2 sides ≈ 8 MiB per (T, B) comparison, only for §3-matched pairs. Manifest field: `duplicate_of: "<base name>" | null`.

**Duplicate ⇒ classification stays `usable`** (it genuinely loads) **with `duplicate_of` set** — exclusion is the *catalog's* policy decision, recorded here as data, not enforced by the audit tool (single-responsibility: the audit reports truth; the catalog applies policy).

### 5. Manifest schema (additive; `audit_version` stays 1, `tool_version` → 0.2.0)

Transformer entries add to `files[]`:

```json
{
  "kind": "transformer",
  "root": "diffusion_models",
  "relative_path": "Wan2.2/wan2.2_t2v_high_noise_14B_fp16.safetensors",
  "sha256": null,
  "size_bytes": 28864978432,
  "classification": "usable",
  "reason": "prognosis_hi-prec",
  "prognosis": {"verdict": "HI-PREC", "detail": "bf16/fp16 — standard path", "family": "wan", "n_fp8": 0},
  "matched_bases": ["wan21_t2v"],
  "duplicate_of": null,
  "verdicts_by_base": {},
  "convert_plan": null,
  "convert_output": null,
  "error": null
}
```

- `root`: basename of the `--transformer-root` the file was found under (disambiguates `relative_path` across roots; the resolved root paths are recorded once in a new top-level `transformer_roots` array, path-redaction rules per ADR-014 F-8 applying to argv only).
- `sha256` is `null` for transformer entries by default — hashing 2.2 TB is a non-starter; a later slice may add `--hash-transformers` opt-in. (Schema rule "sha256 null only for zero_byte" is hereby amended: also null for `kind:"transformer"` unless opted in.)
- `verdicts_by_base` stays `{}` for transformers (that field is the LoRA shape-check contract; transformer matching lives in `matched_bases`).
- Sort key for `files[]` becomes `(root_discriminator, relative_path)` where LoRA entries carry `root_discriminator = ""` — existing LoRA ordering is unchanged (determinism + additivity both hold).

### 6. What is deliberately NOT here

- **No dry-load for transformers** — 130 × GB-scale pipeline loads is a multi-hour GPU job; the live gen-validation slice owns actually-loads truth. Prognosis + shape evidence only.
- **No `--convert` / `--delete` on transformer roots** (Vision invariants 7/11). `deletable` transformer entries are report-only; the operator deletes by hand if ever.
- **No text-encoder/VAE trees** — only the two trees Grant named.

## Alternatives Rejected

- **Full-content SHA-256 dedupe.** 2.2 TB ≈ hours of pure IO per audit run; sampled unique-shape-pair comparison gives byte-level evidence at ~8 MiB per matched pair. The residual risk (two files identical in their 4 sampled MiB-prefixes but differing elsewhere) requires an adversarially crafted file, which is outside the same-trust-zone MVP threat model (ADR-014 §6 posture).
- **Name-based dedupe** (match `z_image_bf16.safetensors` → `Z-Image-base` by string). Names lie constantly in these trees (`velvetChroma_v20FP16` is a finetune, not Chroma). Content evidence or nothing.
- **Key-name mapping per family for exact tensor pairing.** That's the loader's remap machinery (slice C); reusing it per-family for audit pairing couples the audit to per-family remap completeness. Unique-shape pairing is family-agnostic and sufficient.
- **Excluding duplicates at audit level** (classification `excluded`). The audit reports facts; catalog applies policy. Keeps the manifest reusable if the policy changes (e.g. Grant later wants fp8 duplicates listed as variants).
- **Extending ADR-014 via changelog instead of a new ADR.** The dedupe policy, shape-fingerprint thresholds, and schema additions are new decision surface with their own alternatives; ADR-014 is accepted/closed and its changelog should record *its* scope, not absorb a sibling design.

## Deferred / Out of Scope

- `--hash-transformers` opt-in full hashing (trigger: catalog wants provenance-grade identity).
- Transformer dry-load / live validation (gen-validation slice).
- LoHa investigation (still parked from ADR-014).

## Changelog

- **2026-07-05 (proposed):** Drafted after Vision approval, following Grant's catalog-service directive. Gated on `security-auditor` (Opus) design review before code (ADR-014 precedent for this surface class); `code-reviewer` + `security-auditor` on the implementation slice.
