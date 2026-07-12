# Security Review — comfyless prompt-enhancer (first trust_remote_code in repo)

**AI-Disclosure:** Claude (Opus 4.8) `security-auditor` authored; Grant reviewed.
**Date:** 2026-07-11
**Trigger (ADR-026 §8):** the enhancer's `hunyuan-reprompt` backend loads the
Tencent reprompt tokenizer, which requires `trust_remote_code=True` (custom
`HYTokenizer` via `auto_map`) — the **first** `trust_remote_code` in the repo.
Also reviewed: the `openai-endpoint` HTTP client and the offline CLI file I/O.
Governing spec: `docs/decisions/ADR-026-comfyless-prompt-enhancement.md` §8.

## Scope
`comfyless/enhance.py` (full), the `comfyless/generate.py` inline-enhance wiring,
recipes/registry, against ADR-026 §7/§8 and the reprompt-model facts
(`implementation_details.md` A8, reprompt investigation).

## Conclusion: CLEAN (after one MEDIUM folded)

The `trust_remote_code` gate satisfies ADR-026 §8: the model loads
`trust_remote_code=False` (transformers-native `hunyuan_v1_dense`); only the
tokenizer uses TRC; both loads are `local_files_only=True`; a hard hash gate runs
as a precondition before the tokenizer loads; drift/missing files fail closed;
the inline `generate.py` hook fails closed (never generates on an un-enhanced
prompt); `tomllib` config parsing carries no code-exec and fails closed; API keys
come only from `key_env` (env-var name) and are never logged or written to
sidecars; backends are selectable by NAME only (no raw-URL), containing the SSRF
surface against a future untrusted caller.

### Finding folded before close

**[MEDIUM] Hash pin scope too narrow — `auto_map` / config unpinned.** The initial
implementation pinned only `tokenization_hy.py`, but `trust_remote_code` executes
whatever `auto_map` in `tokenizer_config.json` names — a swapped config could
redirect execution to un-reviewed code while the pinned `.py` stayed identical,
violating §8's "silent swap detectable at load."
**Resolution:** `_verify_reprompt_tokenizer` now sha256-pins BOTH
`tokenization_hy.py` and `tokenizer_config.json`, and additionally asserts the
`auto_map` `AutoTokenizer` target equals the reviewed `tokenization_hy.HYTokenizer`.
Verified: the real model still loads; a tampered/missing/​redirected snapshot is
refused.

### Accepted residual (deferred, documented)

**[LOW] urllib forwards `Authorization` across redirects** (`enhance_openai_endpoint`,
`_resolve_endpoint_model`). Endpoints are operator-chosen localhost, never
caller-supplied, so no exposure today. Deferred (implementation_details.md A11);
add a header-stripping redirect handler if a non-localhost endpoint is ever used.

**[INFO]** HTTP error path echoes ≤300 B of the server response into the exception
(no key can leak — key is outbound-only; stderr stays local). TOCTOU between hash
check and tokenizer load (negligible on a single-user local tool).

## Pin record (for future model bumps — re-review + re-pin on any change)
- `tokenization_hy.py` sha256 `0c1fced82e7de447f956daea515486bccf2f8a4b06d3d228c6296ea53f54d3b7`
- `tokenizer_config.json` sha256 `560d14d33de1d2e090913620b89bf8377f0f791bd0656f793be6adcf346eee7a`
- reviewed `auto_map` target: `tokenization_hy.HYTokenizer`
