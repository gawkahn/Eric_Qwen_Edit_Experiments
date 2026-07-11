# Security Review — Hunyuan-Image 2.1 refiner daemon parity (re-apply onto main)

**AI-Disclosure:** Claude (Opus 4.8) `security-auditor` authored; Grant reviewed.
**Date:** 2026-07-11
**Reviewer:** `security-auditor` (Opus), invoked during the Hunyuan-Image 2.1
base+refiner re-apply onto `main` (Vision: `docs/vision/epic-hunyuan-2-1-plus-enhancer.md`,
ADR-016 + ADR-025).
**Trigger (§12):** change to `comfyless/server.py` (IPC surface) + a new
caller-supplied path (`refiner_path`) fed to `from_pretrained`
(pickle-deserialization of model weights).

## Scope reviewed

- `comfyless/server.py` — `_PATH_FIELDS` + `refiner_path`; NUL defense;
  `_check_paths` containment loop incl. `refiner_path`; `_request_cache_key`;
  `_maybe_load_refiner`; `_evict_chain`; `_handle_generate` load / rollback /
  LoRA-reload / cache paths.
- `comfyless/params_validation.py` — SCHEMA_KIND / `_RUNTIME_KIND` typing,
  `_check_field`, `validate_machine_request`.
- `comfyless/hunyuan_chain.py` (new) — `load_refiner_pipeline` class-lock,
  `run_chain`.
- `nodes/eric_diffusion_utils.py` — `resolve_vae_tiling`, `detect_pipeline_class`,
  `_is_hf_repo_id`, `resolve_hf_path`.
- Full feature diff.

Out of scope: in-process ComfyUI node paths (operator trust domain, not behind
the socket boundary); `swap_sampler` / `_save_with_metadata` internals
(unchanged); test files.

## Conclusion: **CLEAN**

The critical path-containment invariant holds. No CRITICAL / HIGH / MEDIUM issues.

Confirmed:
- `refiner_path` is in the `_check_paths` validated-field loop and gets the
  identical `_within(root)` realpath+containment check as every other model
  path. A request with `refiner_path` outside `--model-base` (absolute-outside,
  `..` traversal, or symlink whose realpath escapes) is rejected with a
  `PathError` **before** `_handle_generate` → `_maybe_load_refiner` →
  `load_refiner_pipeline` runs. `_check_paths` validates `refiner_path`
  unconditionally for every generate request (not gated on family or cache
  state), so the check cannot be skipped by any downstream branch.
- `_maybe_load_refiner` raises `ValueError` when `refiner_path` is set on a
  non-`hunyuan-image` family (no silent fallback); passes
  `allow_hf_download=False`; `load_refiner_pipeline` hard-locks
  `_class_name == "HunyuanImageRefinerPipeline"`.
- IPC typing sound: `refiner_path`/`vae_tiling` `_KIND_STR`, `refiner_steps`
  `_KIND_INT`, `refiner_cfg` `_KIND_FLOAT` — non-string/type-confused values
  rejected by `validate_machine_request` before any load; `refiner_path` in
  `_PATH_FIELDS` gets NUL-byte rejection ordered before the realpath call.
- State consistency fail-closed: on refiner-load failure both the initial-load
  and LoRA-reload paths `del pipe` + `empty_cache` + return error **without**
  `server_state.update()`; config-change and LoRA-failure branches call
  `_evict_chain` (which `server_state.clear()`s). No reachable state where a
  `cache_key` bearing a non-empty `refiner_path` coexists with
  `refiner_pipeline is None` and a live `pipeline`. `_evict_chain` drops the
  refiner before the base.

## INFO observations (no action for this diff)

1. **TOCTOU symlink window** — `_check_paths` realpath (check) vs
   `from_pretrained` (load). A symlink swapped between check and load could
   redirect the load; identical to the long-standing window for
   `model`/`transformer_path`/every other model path, not introduced here, and
   out of scope for the single-user desktop model. If ever hardened, harden
   uniformly for all path fields.
2. **Absolute `refiner_path` in output metadata** (`generate.py`) — consistent
   with the pre-existing absolute `model`/`transformer_path`/`vae_path`/
   `text_encoder_path` metadata entries; caller supplied `refiner_path` itself,
   so not a new cross-boundary leak. Logged to TECH_DEBT: when the MCP surface
   returns refiner metadata to an untrusted agent, fold `refiner_path` into the
   same basename redaction the other path keys receive. (The MCP `generate`
   handler does not thread refiner today — `test_hunyuan` Inv 12 confirms.)
