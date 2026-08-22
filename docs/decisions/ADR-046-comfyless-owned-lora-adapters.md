# ADR-046 — comfyless-owned LoRA adapter subsystem (ADR-045 slice 3c)

Status:   accepted
Parent:   `docs/decisions/ADR-045-comfyless-diffusion-standalone-repo.md`
Vision:   `docs/vision/comfyless-diffusion-extraction.md` §Slice 3c

## Context

ADR-045 clears every third-party-authored line out of the code that moves to
`comfyless_diffusion`. Slices 3a (sigma schedules) and 3b (upscale-VAE decode)
are done. What remains is the largest and riskiest item, and the Vision said it
needed its own record: comfyless's **only** LoRA entry point is
`nodes/eric_qwen_edit_lora.py`, reached through four imports —
`comfyless/generate.py:66` and `:1333`, `comfyless/server.py:669` and `:830` —
for two names, `load_lora_with_key_fix` and `is_direct_merge_adapter`.

Measured 2026-08-22 with `git blame -w`: the module is 1,540 lines, **782
Eric-authored and 758 Grant-authored**, and the authorship is interleaved at
function granularity rather than file granularity:

| Function | Author | Lines | Disposition |
|---|---|---|---|
| `_make_adapter_name`, `_load_state_dict`, `_adapter_module_path` | Eric | 48 | rewrite |
| `_normalize_keys`, `_detect_adapter_type` | Eric | 105 | rewrite |
| `_load_lokr_adapter` (+`_peft`, `_direct`) | Eric, Grant edits | 228 | rewrite |
| `_load_loha_adapter` (+`_peft`, `_direct`) | Eric, Grant edits | 174 | rewrite |
| `_rename_lora_down_up` | Eric | 24 | rewrite |
| `_load_lora_adapter` (+`_peft`, `_direct`) | Eric, Grant edits | 234 | rewrite |
| `_set_adapters_safe` | Eric | 25 | **not ported** — node-only consumer |
| `get_lora_list`, `get_lora_full_path` | Eric | 39 | **not ported** — `folder_paths` (ComfyUI) |
| `load_lora_with_key_fix` | mixed (~50 Eric) | 199 | rewrite the Eric lines (docstring, fast path, error predicate, dispatch) |
| `_apply_te_lora`, `_decode_kohya_keys`, `_bake_lora_alpha_scales` | Grant | 281 | move verbatim |
| `unload_adapters`, `is_direct_merge_adapter`, `plan_match_model_names` | Grant | 154 | move verbatim |

This is the code behind every LoRA incident in the repo's history: the Krea
LoRA regression (buffer-blindness, 7cc99ab), the alpha-drop over-strength bug
that reads as noise (`_bake_lora_alpha_scales`), the LoKR alpha-sentinel
convention, the LoKR→LoRA flatten rescue, and the DMR quantized-merge
dispatcher (ADR-019). Unlike the sigma schedules there is no frozen golden —
the Vision's concern — so the proof has to be built before the rewrite.

**The corpus, measured 2026-08-22** (catalog DB, `kind='lora'`, not excluded,
310 entries, 308 readable safetensors headers):

| Format | Files | Notes |
|---|---|---|
| `lora_A`/`lora_B` | ~190 | `diffusion_model.` prefix dominates (Z-Image, Krea, Qwen, Klein); some `transformer.`, bare `transformer_blocks`, `base_model` |
| `lora_down`/`lora_up` (Kohya) | ~105 | mostly `lora_unet_*` (Flux 47, Chroma 8, SDXL 4); 11 carry `lora_te*` text-encoder keys; some `lora_transformer_*` |
| LoKR | 11 | Klein 4, Z-Image 3, Krea 3, Qwen 1 |
| LoHa | **0** | the LoHa path has no real-file exercise on this machine |
| pre-baked `.diff` | 2 | Krea, handled by the conversion plan |
| carry `.alpha` | 114 | the bake path is live on a third of the corpus |

## Decision

**Create `comfyless/core/lora_adapters.py`** as a fresh, single-author module
that is comfyless's LoRA subsystem. `comfyless/generate.py`, `comfyless/server.py`
and `scripts/lora_audit.py` import from it. The node pack keeps
`nodes/eric_qwen_edit_lora.py` untouched, exactly as 3a/3b left the node-side
originals — it still serves the LoRA stacker, UltraGen and the multistage nodes,
and after the split it is the node repository's problem.

### What moves verbatim and what is rewritten

Grant-authored functions move as written (they are not the attribution problem
and they are the incident-hardened code — rewriting them would be risk for no
gain): `_apply_te_lora`, `_decode_kohya_keys`, `_bake_lora_alpha_scales`,
`unload_adapters`, `is_direct_merge_adapter`, `plan_match_model_names`, and the
Grant-authored body of `load_lora_with_key_fix` (the compatibility pre-check,
the slice-4 conversion path, the 0-modules failure report).

Eric-authored functions are **re-implemented from the contracts below**, not
edited in place. The point of a contract is that it can be satisfied by a
different shape; the new module is organised differently where that is the
better design anyway:

- The three `_load_*_adapter_direct` functions are ~90% the same loop with a
  different delta formula. The new module has **one** direct-merge driver
  parameterised by adapter kind (delta function, rank rule, backup-attr
  family, `_type` tag), which is what `test_quant.py`'s DMR source guard
  already treats them as.
- Key-prefix normalisation is a table-driven pass, not an if-ladder.
- Public names drop the leading underscore where comfyless calls them
  (`load_state_dict`, `make_adapter_name`, `adapter_module_path`,
  `detect_adapter_type`, `normalize_keys`); underscore aliases are NOT kept —
  no consumer outside the node pack uses them, and the node pack keeps its own
  file.

### Behavioural contract (the spec the rewrite is held to)

Every item below is pinned by the equivalence harness. Where the original's
behaviour is an accident of implementation it is still preserved, because the
pixel matrix hashes exact bytes and "matching the original" is the slice.

1. **`make_adapter_name(filename)`** — strip the first matching extension from
   `(.safetensors, .bin, .pt, .pth)` case-insensitively; then replace every
   `.` and ` ` with `_`.
2. **`load_state_dict(path)`** — `.safetensors` via `safetensors.torch.load_file`;
   anything else via `torch.load(map_location="cpu", weights_only=True)`.
3. **`adapter_module_path(key)`** — cut the key at the first *table entry* found
   anywhere in it, scanning the table **in table order** (not by position in
   the key): `.lokr_w1 .lokr_w2 .lokr_t2 .lora_A.weight .lora_A.default.weight
   .lora_B.weight .lora_B.default.weight .lora_down.weight .lora_up.weight
   .hada_w1_a .hada_w1_b .hada_w2_a .hada_w2_b .alpha .diff .diff_b`. No entry
   → drop the last dotted component (or return the key when it has no dot).
4. **`detect_adapter_type(sd)`** — substring tests over all keys, priority
   `lokr` (`lokr_w1`|`lokr_w2`) > `loha` (`hada_w1_a`|`hada_w2_a`) > `lora`
   (`lora_A`|`lora_B`|`lora_down`|`lora_up`) > `"unknown"`.
5. **`normalize_keys(sd, model=None)`** — first drop keys starting with any of
   `text_encoder. text_encoder_2. lora_te1_ lora_te2_ lora_te_`; empty → return
   empty. *Without a model:* strip `transformer.` from the keys that carry it
   iff any key carries it. *With a model:* let P = adapter module paths of the
   filtered keys and M = the model's non-empty `named_modules()` names.
   (a) `P ∩ M ≠ ∅` → unchanged. (b) For each prefix in order `transformer.`,
   `diffusion_model.`, `model.diffusion_model.`, `model.` — if stripping it
   from the P entries that carry it yields a set intersecting M, strip it from
   every key that carries it and return. (c) Otherwise search for a prefix
   `pfx` such that some `p ∈ P` ends with some `m ∈ M` (`p` strictly longer)
   and more than 30% of P map into M under `pfx`; apply it. (d) Nothing → warn
   with samples and return the filtered dict unchanged. **Known deviation:**
   the original's step (c) inspects the first 20 of P and first 50 of M in
   *set iteration order*, which is hash-seed-dependent, so the candidate it
   finds first is not reproducible across processes. The new implementation
   searches the same space in **sorted** order. On every input where the
   original's answer is unique the two agree; the harness constructs one
   multi-candidate case to document the difference rather than hide it.
6. **`rename_lora_down_up(sd)`** — if any key contains `lora_down` or `lora_up`,
   rewrite `.lora_down.weight → .lora_A.weight` and `.lora_up.weight →
   .lora_B.weight`; otherwise return the same object.
7. **PEFT injection** (all three kinds) — `inject_adapter_in_model(cfg,
   transformer, adapter_name=, state_dict=sd)` then `set_peft_model_state_dict`,
   then set `transformer._hf_peft_config_loaded = True`; log `unexpected_keys`
   minus `.alpha` entries and `missing_keys`. Configs:
   - LoRA: `LoraConfig(r, lora_alpha, target_modules=["_dummy"])`, r = `shape[0]`
     of the first key (iteration order) containing `lora_A` with ndim ≥ 2, else
     64; alpha = the first single-element `.alpha` whose base has a paired
     `.lora_A.weight`, else `float(r)`. Input is first passed through
     `rename_lora_down_up`.
   - LoKR: `LoKrConfig(r=100000, alpha=100000.0, decompose_both=False,
     decompose_factor=-1, target_modules=["_dummy"])`.
   - LoHa: `LoHaConfig(r, alpha, target_modules=["_dummy"])`, r from the first
     key containing `.hada_w1_a` (`shape[1]`) or `.hada_w1_b` (`shape[0]`) with
     ndim ≥ 2, else 8; alpha = first single-element `.alpha`, else `float(r)`.
   The transformer is `pipe.transformer` or, failing that, `pipe.unet`.
8. **Direct merge** (one driver, three kinds) — `model_sd =
   merge_resolution_map(transformer)`; `refuse_unmergeable_base(...)` BEFORE
   the first write (all-or-nothing, ADR-019 req 65); group keys by
   `adapter_module_path`, param name = remainder after the dot. Per module:
   - LoRA: A from `lora_A.weight` else `lora_A.default.weight`, B likewise;
     delta = `(B.float() @ A.float()) * scale`; r = `A.shape[0]`.
   - LoKR: delta = `torch.kron(w1.float(), w2.float()) * scale`;
     r = `min(w1.shape)` if ndim ≥ 2 else 1.
   - LoHa: delta = `(w1_a.float() @ w1_b.float()) * (w2_a.float() @ w2_b.float())
     * scale`; r = `w1_b.shape[0]` if ndim ≥ 2 else 1.
   - scale = `(alpha/r) * weight` when a module-local `alpha` is present and
     r > 0; `weight` when alpha is absent or r ≤ 0.
   - target = `path + ".weight"` if in `model_sd`, else `path`, else skip.
     Shape mismatch → `reshape` to the target or skip on `RuntimeError`.
   - Every write goes through `apply_merge_delta(transformer, target, delta,
     backup, log_prefix)` with `backup = transformer._<kind>_backup_<name>`
     (created on demand). **No direct `param.data.add_`.**
   - `record_direct_merge` iff applied > 0; `transformer.peft_config[name] =
     {"_type": "<kind>_direct", "_applied_modules": n, "_weight": weight}`;
     `_hf_peft_config_loaded = True`; warn when skipped > applied; return
     `applied > 0`.
9. **Orchestrators** — LoRA: bake alpha → rename → `pipe.load_lora_weights`
   with the bare dict then with every key prefixed `transformer.`, each
   verified via `pipe.get_list_adapters()` (unverifiable → assume present) →
   PEFT injection → direct merge. LoKR: PEFT → (on `ValueError`/`RuntimeError`)
   direct → (0 applied) `flatten_lokr_to_lora_sd` rescue: exception or empty →
   False; else pop the stale `peft_config[name]` and route through the LoRA
   orchestrator. LoHa: PEFT → direct.
10. **`load_lora_with_key_fix`** — pre-check via `check_lora` (non-fatal,
    `min_compatibility` skip); slice-4 conversion path when the pre-check shows
    0 matched of > 0; fast path `pipe.load_lora_weights(path)`, re-raising
    unless the error is a `KeyError` or its text matches the fixable list
    (`Target modules…not found`, `No modules were targeted`, `state_dict`,
    `lora_A`, `lora_B`, `lokr`, `loha`, `hada_`, `PEFT backend`,
    `not implemented`, `Handling for key`; the substring tests that are
    case-insensitive in the original stay so); fallback: load → TE apply →
    Kohya decode → normalise → detect → dispatch; `"unknown"` raises
    `ValueError`; 0-applied reports FAILED and returns False.

### Proof — three layers, built before the swap

The harness exists so the rewrite can be wrong and get caught. Each layer runs
the **node-pack original and the new module side by side** on the same input
in the same process and compares outputs bitwise — not against frozen goldens,
because the original stays in the repo until slice 6 and live comparison is
stronger. (After the split the node repo keeps the differential; the new repo
keeps only the contract tests.)

**Layer A — the real corpus, key-space only, CPU.** Every readable LoRA
header on this machine (308 files) through `adapter_module_path` per key,
`detect_adapter_type`, `normalize_keys` (no-model mode), `rename_lora_down_up`
and `make_adapter_name`. Headers only — no tensors are read, so it runs in
seconds and covers every key convention the corpus actually contains. Skips
with a visible `SKIP` when the catalog DB is absent (CI).

**Layer B — synthetic models, full behaviour, CPU.** Tiny `nn.Module`
transformers in diffusers layout (`transformer_blocks.N.attn.to_{q,k,v,out.0}`,
`ff.net.*`) plus a `ScaledFp8Linear` variant so the DMR dispatcher path is
exercised; synthetic LoRA / LoKR / LoHa state dicts — with and without `.alpha`,
with `alpha ≠ rank`, in `lora_A/B` and `lora_down/up` naming, under each key
prefix, with `default.` PEFT-saved names, with shape-mismatch modules, with a
module absent from the model — driven through both implementations on two
deep-copies of the same seeded model. Compared: every parameter and buffer
bitwise, `peft_config` registry entries, the backup dicts, return values, and
the `_hf_peft_config_loaded` flag; then `unload_adapters` on both and compared
again (restore is Grant's code but it consumes the backup format the driver
wrote, so it is the cheapest place to catch a backup-shape drift). Negative
cases: a LoKR whose factorisation PEFT rejects must fall to direct merge in
both; a 0-module direct merge must return False in both; the DMR source guard
(no `param.data.add_`, `apply_merge_delta(` and `merge_resolution_map(`
present) is re-pointed at the new driver.

**Layer C — the pixel matrix, GPU.** `feat-lora-krea2` (lora_A/B, fast path)
is the only LoRA case today. Four cases are added so each real load path is
hashed end to end, on the reproducible models:

| Case | Model | File | Path exercised |
|---|---|---|---|
| `feat-lora-qwen-lightning` | Qwen-Image-2512 | `Qwen/accelerators/Qwen-Image-Lightning-8steps-V11-bf16` | `lora_down/up`, bare `transformer_blocks` prefix, `.alpha` → bake |
| `feat-lora-krea-kohya` | Krea-2-Turbo | `Krea/style/photo/emerald` | `lora_unet_*` Kohya with `.alpha` |
| `feat-lora-krea-lokr` | Krea-2-Turbo | `Krea/realism/realism_engine_krea2_v3.1` | LoKR, conversion-plan / flatten |
| `feat-lora-klein-lokr` | FLUX.2-klein-9B | `Flux.2 Klein 9B/concept/klein_snofs_v1_1` | LoKR on Klein, `lokr_to_lora_svd` |

`pre3c` is captured with the ORIGINAL code before the swap (the only time that
comparison exists), `post3c` after; all strict cases must match. If a LoKR
case proves non-reproducible across batches (SVD), it is demoted to
`strict=False` with the measurement recorded, not silently.

### Consumers re-pointed in this slice

- `comfyless/generate.py` — both imports.
- `comfyless/server.py` — both imports. **Red Zone path** (`_red-zone-paths.sh`);
  the change is two import lines, but it runs `security-auditor` and the commit
  references this ADR and the saved review.
- `scripts/lora_audit.py` — `load_lora_with_key_fix`, `unload_adapters`,
  `_load_state_dict` from the new module; the `folder_paths` stub and the fake
  `nodes` package stay only as long as anything else in that script needs them.
- `test_server_robustness.py` — the DLW section patches the module
  `server.py` actually imports from, so it must patch the new name.
- `comfyless/__init__.py` shims and the `sys.path` insert are **not** deleted
  here — Vision item 5 says deleted-not-ported, and after this slice nothing
  in `comfyless/` needs them, but every test suite that imports `nodes.*`
  still relies on `import comfyless` installing them. Their deletion is a
  slice of its own with the suites' imports as its scope (folded into slice 5
  or 7; TECH_DEBT records it).

## Alternatives Rejected

**Edit the Eric-authored lines in place and move the file.** `git blame` on
the new repo would still show 782 Eric lines on every untouched character.
The obligation is on the code, not on the commit that moved it.

**Port `_set_adapters_safe` / `get_lora_list` / `get_lora_full_path` for
completeness.** comfyless has its own `apply_adapter_weights` in `generate.py`
and never consults ComfyUI's `folder_paths`. Porting dead surface into the
package being cleaned is the wrong direction.

**Frozen goldens instead of side-by-side comparison.** Goldens were right for
3a/3b (pure functions, cheap to enumerate). Here the interesting behaviour is
mutation of a model under PEFT wrapping and quantized bases; freezing every
parameter of every case would be large, opaque, and would have to be
regenerated for any PEFT bump. While the original lives in the repo the live
differential is both stronger and smaller.

**Treat the LoHa path as dead and drop it.** Zero LoHa files exist on this
machine, but the format is real (LyCORIS), the node pack exposes it, and the
rewrite cost is one delta function inside the shared driver. Kept, proven on
synthetic files, and recorded as corpus-unexercised.

## Deferred / Out of Scope

- Deleting `_install_shims()` and the `sys.path` insert (see Consumers).
- `test_lora_alpha_bake.py`, `test_lora_convert_krea.py`,
  `test_lora_order_insensitive.py` keep importing the node-pack module: they
  test the copy the node pack runs. Re-pointing them is slice 7's
  suite-ownership decision, not this slice's.
- The Z-Image LoKR failure (TECH_DEBT 2026-07-06) is a behaviour of the
  original and is reproduced, not fixed — fixing it here would break the
  equivalence this slice is proving.
- `normalize_keys` step (c)'s deterministic candidate order is the one
  intentional behaviour change, recorded above; making the original match is
  out of scope (it is the node pack's copy).

## Changelog

- 2026-08-22 — accepted; supersedes the "needs its own ADR" placeholder in the
  Vision §Slice 3c.

AI-Disclosure: Claude (Fable 5) authored; Grant reviewed.
