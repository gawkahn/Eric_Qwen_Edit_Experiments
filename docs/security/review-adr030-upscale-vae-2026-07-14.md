AI-Disclosure: Claude (Fable) authored this security review; Grant reviewed.

# Security Review — ADR-030 2× upscale-VAE decode (2026-07-14)

Reviewer: `security-auditor` (Fable). Trigger (§12): modifies the Unix-socket IPC
daemon `comfyless/server.py`, and loads a caller-supplied model path via
`resolve_hf_path` (new `_load_upscale_vae` in `comfyless/generate.py`).

## Scope

New opt-in `--upscale-vae PATH` (+ `--upscale-vae-subfolder`). Two new wire
fields (`upscale_vae_path`, `upscale_vae_subfolder`) accepted at the daemon
boundary flow into `_load_upscale_vae`, which resolves the path through
`resolve_hf_path` and calls `AutoencoderKLWan.from_pretrained(resolved,
subfolder=<sub>, local_files_only=True)`. Threat model: an untrusted JSON
request at the daemon socket (and, per the documented roadmap, the same request
shape reaching `--json`/MCP/mcpo agent surfaces). Core invariant checked: every
weight load is confined to operator-curated roots so caller-controlled
pickle/`torch.load` deserialization can never be pointed at an arbitrary
directory (server.py `_check_paths`, rated CRITICAL for `refiner_path`).

## Findings

### [HIGH] `upscale_vae_subfolder` escaped `_check_paths` root containment — RESOLVED
`comfyless/generate.py` `_load_upscale_vae`. The subfolder was a free-form `str`
(not in `_PATH_FIELDS`/`_check_paths`) concatenated into
`from_pretrained(subfolder=...)` after the containment check; a value like
`../../../../home/x` or an absolute path traversed out of the realpath-validated
root and would load config+weights (`.bin` → pickle) from anywhere on the
filesystem — reopening the arbitrary-directory-deserialization hole the
containment guard closes for the path itself. Same-UID socket (0700 dir) means
no privilege escalation today, but it defeats the operator's directory-
confinement policy and becomes CRITICAL the moment the `--json`/MCP/HTTP agent
surfaces make the request origin untrusted.

**Fix applied:** `_load_upscale_vae` now rejects any subfolder that is absolute
or whose realpath-joined location escapes the resolved root (also catches
symlink escapes), raising `ValueError` before `from_pretrained`. Regression test:
`test_params_schema.py` — traversal subfolders (`../../etc`, `/etc/passwd`,
`a/../../../b`) rejected with "relative subpath" before any load.

### [MEDIUM] MCP redaction map did not cover the upscale path — RESOLVED
`comfyless/mcp_server.py` `_MCP_PATH_TYPED_FIELDS` vs
`comfyless/generate.py` metadata. `generate()` records the caller's absolute
upscale-VAE path in metadata; the MCP PNG sink `redact_metadata_for_png`
basenames only fields in `_MCP_PATH_TYPED_FIELDS`, which lacked the new field —
so if/when the MCP surface forwards it, the full host path would be embedded in
the returned PNG. Latent today (MCP does not forward the field; MCP is deferred
per ADR-030), but one wiring step away.

**Fix applied:** added the (now canonical) key `upscale_vae_path` to
`_MCP_PATH_TYPED_FIELDS` so it is basenamed like `vae_path`.

## Cleared (no finding)

- `upscale_vae_path` containment: added to `_PATH_FIELDS` (NUL rejection before
  realpath) and the `_check_paths` field loop (absolute + realpath root-union,
  symlinks resolved); loaded only on the post-`_check_paths` path; repo-id form
  rejected as non-absolute. Tested.
- No daemon network fetch: `allow_download=False` + `local_files_only=True`.
- Accept-loop crash safety: both fields type-checked `str` by
  `validate_machine_request`; NUL pre-rejected for the path; load failures caught
  → `LoadError`, success-only cache assignment (no corrupt state).
- No new code-execution surface vs the existing `--vae` override path (same
  trust level; same `from_pretrained` deserialization).
- Scope: diff stays within declared edit scope; no auth/authz/crypto/billing
  surfaces touched.

## Disposition

Both findings fixed in the same slice before commit. No open security debt.
