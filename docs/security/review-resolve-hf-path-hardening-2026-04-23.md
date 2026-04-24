# Security Review — resolve_hf_path hardening slice

AI-Disclosure: Claude (Opus 4.7) authored; Grant reviewed.
Date: 2026-04-23
Scope: hardening diff across `nodes/eric_diffusion_utils.py` and `comfyless/generate.py` implementing Findings 1, 2, 3 from `docs/security/review-resolve-hf-path-2026-04-23.md`.
Reviewer: security-auditor subagent (Opus)

## Summary

The three MEDIUM findings from the prior review are closed with minimal, targeted changes. The `_is_hf_repo_id` heuristic is now self-sufficient (no longer load-bearing on `validate_repo_id`); the download branch prints a one-line `DOWNLOADING` notice to stderr that names the repo before `snapshot_download(path)` hits the network; and `_run_cli_mode` emits a symmetric PNG-sidecar warning when `--params` plus `--allow-hf-download` plus an HF-repo-ID `model` value appear together. No new attack surface, no PII in logs, and `trust_remote_code` remains absent codebase-wide so the threat model is still "weight substitution, not RCE." The Finding 2 warning fires from `resolve_hf_path` itself, which is correct for the CLI/node path where the resolver is always reached — every loader call-site funnels through the resolver body, so coverage is uniform. The Finding 3 warning is localised to `_run_cli_mode` where PNG `--params` is the only intake channel, which is the correct placement.

## Verification of prior findings

### Finding 1 — `_is_hf_repo_id` heuristic tightening

**Status:** CLOSED.

`nodes/eric_diffusion_utils.py` `_is_hf_repo_id` now performs the final check `parts[0] not in (".", "..") and parts[1] not in (".", "..")` after the length-2 / all-non-empty gate. Walking the specific test cases:

- `"foo/.."` — `parts = ["foo", ".."]`, second part equals `".."`, rejected. Closes the exact case from the prior finding.
- `"foo/."` — second part equals `"."`, rejected.
- `"../bar"` — rejected by earlier `startswith("../")`.
- `"./foo"` — rejected by earlier `startswith("./")`.
- `"./."` — prefix check `startswith("./")` fires first, rejected.
- `"foo/."` followed by whitespace (e.g. `"foo/. "`) — `parts = ["foo", ". "]`, second part is `". "` (with space), NOT equal to `"."`, so the heuristic ACCEPTS it. `validate_repo_id` in huggingface_hub will then reject it, raising `HFValidationError` (a `ValueError` subclass), caught by the `except Exception` and re-raised as `RuntimeError`. No traversal or path-manipulation risk, just a noisier error message than ideal. Not worth a follow-up fix today; note it for the LLM-agent-bridge hardening.
- `"foo/.hidden"` — second part is `".hidden"`, NOT equal to `"."` or `".."`, ACCEPTED. Correct — `.hidden` is a valid HF repo name prefix (rare but allowed).
- `".foo/bar"` — first part is `".foo"`, NOT equal to `"."`, ACCEPTED. Correct.

**Dependency on `validate_repo_id`:** The inline comment explicitly records that the local check is no longer load-bearing on it, and the trailing-dot-with-whitespace case above is the only remaining place the heuristic is weaker than `validate_repo_id` — and in that case the downstream call catches it. The finding is closed.

### Finding 2 — download-branch network warning

**Status:** CLOSED.

`nodes/eric_diffusion_utils.py` prints `[EricDiffusion] DOWNLOADING from HuggingFace: {path}` to `sys.stderr` unconditionally before the `snapshot_download(path)` call in the `allow_download=True` cache-miss branch. `import sys` is added at module imports.

**Every-path coverage:**

- **Cache-hit branch:** `local_dir` resolves from the offline-only probe, function returns without reaching the warning. No warning, no network — correct silence.
- **Cache-miss + `allow_download=False`:** `ValueError` raised, never reaches the warning. Correct.
- **Cache-miss + `allow_download=True`:** Warning fires, then `snapshot_download(path)` is called. This is the target branch. Correct.
- **Call-site re-entry:** every `resolve_hf_path` call-site (loaders in `eric_diffusion_loader.py`, `eric_diffusion_component_loader.py` × 5 paths, `eric_qwen_edit_loader.py`, `eric_qwen_image_loader.py`, `comfyless/generate.py:_load_pipeline` × 5 paths, `comfyless/generate.py:generate`, `comfyless/generate.py:_run_cli_mode`) funnels through the same resolver body — the warning cannot be bypassed by any call path.

**`snapshot_download(path, local_files_only=True)` and network access:** huggingface_hub's contract is that `local_files_only=True` does not make HTTP requests; on cache miss it raises `LocalEntryNotFoundError`. There is one known edge case in recent hub versions where an offline probe can do a single `HEAD` for ETag comparison on `revision="main"` (staleness check), but this is metadata-only, no file bytes, and is gated by `HF_HUB_OFFLINE=1` if the user sets it. Known upstream behaviour; the new warning does not make this worse. No gap.

**Token-exfil residual:** same as prior Finding 2 — `snapshot_download` will still attach `HF_TOKEN` to requests for gated repos. The warning lets the user see the repo name before the request, which is the proportional mitigation for a desktop tool. Sufficient.

### Finding 3 — PNG-sidecar replay warning

**Status:** CLOSED.

`comfyless/generate.py:_run_cli_mode` fires a stderr `WARNING:` when all three of `args.params`, `args.allow_hf_download`, and `_is_hf_repo_id(_model_val)` are truthy, positioned after the existing local-path warning and before the resolve loop. `_is_hf_repo_id` is imported from `nodes.eric_diffusion_utils`.

**Malicious-PNG scenario (matches prior Finding 3 threat):**

1. Attacker ships a PNG with `parameters` tEXt chunk containing `"model_path": "attacker/evil-repo"`.
2. User runs `python -m comfyless.generate --params evil.png --allow-hf-download`.
3. `_load_params_from_png` extracts the params, `_extract_eric_save_params` renames `model_path` → `model`.
4. `p["model"] = "attacker/evil-repo"` — passes `_is_hf_repo_id` check.
5. `args.params` truthy, `args.allow_hf_download` truthy, `_is_hf_repo_id("attacker/evil-repo")` truthy → warning fires with the repo name.
6. User sees `WARNING: --params supplied an HF repo ID under --allow-hf-download: attacker/evil-repo` before `resolve_hf_path` runs.
7. `resolve_hf_path` then prints its own `DOWNLOADING from HuggingFace` line on cache miss.

Two warnings, both stderr, naming the repo — the user has two clear chances to Ctrl-C. Correct.

**Override case** — `--params evil.png --override model=different-attacker/repo --allow-hf-download`: `args.params` truthy, `_model_val` is the override value, warning fires naming the override value. Technically the warning message says "--params supplied an HF repo ID" when the actual HF repo ID came from `--override`, but the warning output shows the exact repo that will be fetched, so the social-engineering path is still closed. Minor wording imprecision, not a security gap.

**Corollary** — `--override model=attacker/repo` without `--params`: `args.params` is falsy, warning does not fire. Intentional: direct user input via `--override` is the user's own choice, not an undeclared input channel. Consistent with prior review's scoping.

**Over-firing on benign replay:** User's own sidecar replayed with `--allow-hf-download` triggers the warning. Intentional noise — the PNG→model channel is the threat vector regardless of who produced the PNG. One stderr line per replay is acceptable tax.

**`--json` mode:** `_run_json_mode` does not pass `allow_hf_download` to `generate()`, which means it defaults to `False` — downloads are fail-closed on the JSON bridge. The PNG-warning is not needed there because the JSON input channel is explicit. Correct layering.

## New findings

### LOW — Finding 3 warning message wording imprecise in the `--override` PNG-sidecar case

**Location:** `comfyless/generate.py` PNG-sidecar warning block.
**Description:** When `--params evil.png --override model=repo/foo --allow-hf-download` is used, the warning says "`--params` supplied an HF repo ID" but the actual repo came from `--override`. The fetched value (`_model_val`) is shown correctly on the next line so the user still sees what will be pulled. Nit, not a security gap.
**Recommendation:** Either leave it or change wording to "`--params` or `--override` supplied an HF repo ID". One-word change; not blocking.

### INFO — `_is_hf_repo_id` is now imported as a private helper across module boundaries

**Location:** `comfyless/generate.py` imports.
**Description:** A leading-underscore name imported from one module into another is a Python convention violation, not a security issue. Future refactors that rename or change the helper's signature will silently break the comfyless import. The underlying check is small enough that duplicating it at the call site would be an alternative, but the naming-convention-only risk does not warrant action.
**Recommendation:** None. If either module is refactored later, consider renaming `_is_hf_repo_id` to `is_hf_repo_id` (public) in the same slice.

### INFO — `_run_cli_mode` warning does not cover `transformer_path` / `vae_path` / `text_encoder_*` override fields from the PNG

**Location:** `comfyless/generate.py` PNG-sidecar warning block.
**Description:** The PNG warning only checks `p["model"]`, not the component override paths. A PNG sidecar that sets `"transformer_path": "attacker/malicious-transformer"` in the `parameters` chunk would trigger an HF download of the component weight with only the `resolve_hf_path` one-line stderr notice, not the louder PNG-sidecar WARNING. The Finding 2 mitigation (DOWNLOADING line) still fires from inside `resolve_hf_path` at the resolve loop, so the user is NOT left blind — they will see `DOWNLOADING from HuggingFace: attacker/malicious-transformer`. Acceptable for current threat model (one warning per network-touching resolve, with the repo named). When the LLM-agent bridge lands, the PNG warning should be extended to cover all five path fields.
**Recommendation:** None for this slice. Note for the LLM-agent-bridge slice: widen the PNG warning to cover `transformer_path`, `vae_path`, `text_encoder_path`, `text_encoder_2_path`.

### INFO — `import sys` adds no new attack surface

**Location:** `nodes/eric_diffusion_utils.py` imports.
**Description:** Confirmed. `sys` is pure stdlib.

### INFO — `path`/`_model_val` printed verbatim to stderr is safe

**Location:** `nodes/eric_diffusion_utils.py` download-branch warning, `comfyless/generate.py` PNG warning.
**Description:** Both call sites use f-string interpolation, not `%`/`.format()`, so there is no format-string attack surface. The values are printed to stderr only — not written to a SQL database, not passed to `eval`, not fed to a shell command. Safe.

## Post-review resolution

All three prior-review findings closed in this slice. No new MEDIUM or HIGH findings; LOW/INFO items queued for the LLM-agent-bridge slice where the threat model elevates.

Test suites: 368/368 pass (manual-loop 186, multistage 141, samplers 41).

## Conclusion

**Ready to merge.** Findings 1, 2, and 3 from the prior review are each verifiably closed: the heuristic rejects `foo/..` and `a/.` shapes and no longer depends on `validate_repo_id`; every download-branch network call is preceded by a stderr warning naming the repo; and PNG-sidecar replay under `--allow-hf-download` surfaces the repo before resolution. No new MEDIUM or HIGH issues introduced. Follow-up INFO-level items (widen the PNG warning to component paths; tighten the wording for `--override`; make `_is_hf_repo_id` public) can ride with the LLM-agent-bridge slice since that's the commit where the threat model elevates and these become more than nits.
