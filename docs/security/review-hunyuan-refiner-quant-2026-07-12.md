# Security Review — Hunyuan refiner fp8 quant threading (daemon)

**AI-Disclosure:** Claude (Opus 4.8) `security-auditor` authored; Grant reviewed.
**Date:** 2026-07-12
**Trigger:** change to `comfyless/server.py` (IPC surface) — `_maybe_load_refiner`
now threads the request's quant triple (`quant`/`quant_skip`/`quant_only`) into
`hunyuan_chain.load_refiner_pipeline` so the refiner is fp8-quantized like the base.

## Scope
`comfyless/server.py` (`_validate_request`, QUANT_MODES gate, `_maybe_load_refiner`,
`_request_cache_key`, `_evict_chain`, refiner rollback), `comfyless/params_validation.py`
(SCHEMA_KIND typing, quant_skip/only identifier guard, QUANT_MODES), `comfyless/hunyuan_chain.py`
refiner quant block. Context: enhance.py reprompt quant (operator-config surface, not wire).

## Conclusion: CLEAN

No CRITICAL/HIGH/MEDIUM. The refiner quant reuses the **already-validated** base
quant fields with no new trust boundary:

1. `quant` is type-checked (`_KIND_STR`) then **value-allowlisted** against
   `QUANT_MODES = ("none","fp8")` at the machine boundary (`server.py:187`) before
   `_handle_generate` runs — no unvalidated string reaches torchao via the refiner.
2. `quant_skip`/`quant_only` are enforced as bare identifiers (str, no NUL, no
   `/`\`\``, ≤32) at the boundary, and in `hunyuan_chain.py` are used only as
   set-membership filters over a hardcoded `("transformer",)` tuple — `getattr`
   never takes a request-supplied slot. No injection/traversal.
3. `_request_cache_key` already includes the quant triple + `refiner_path`; the
   refiner loads inside the same cache-keyed block and evicts refiner-then-base
   together, so no mismatched-precision refiner can be served.
4. Refiner quant-failure message mirrors the pre-existing base-load behavior over
   the same 0700 same-UID socket (caller already knows `refiner_path`). No new leak.

INFO (no action): `_maybe_load_refiner` reads `req.get("quant")` directly rather
than the `req_quant` local in `_handle_generate`; equal today because
`req.update(result.payload)` normalizes `req` first. Noted for a future refactor
that might stop mutating `req`.

trust_remote_code gate (`_verify_reprompt_tokenizer`) is intact — the reprompt
quant block sits between the `trust_remote_code=False` model load and the TRC
tokenizer load, not disturbing the hash-pin ordering.
