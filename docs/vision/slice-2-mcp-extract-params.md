# Slice 2 Vision — `extract_params` MCP tool (JSON-sidecar reader)

**Date:** 2026-05-19
**ADR:** [ADR-011](../decisions/ADR-011-comfyless-mcp-server.md) — §2 (`extract_params` row), §3 second user-decided exclusion (PNG extraction NOT exposed; two ordered checks), and the 2026-04-28 security-auditor F-3 fold-in. **Now also governed by [ADR-015](../decisions/ADR-015-mcp-catalog-reference-resolution.md) §3 (extract_params reverse mapping: returns catalog names; basename fallback + INFO notice on miss).**
**Status:** **superseded by ADR-015 re-plan (2026-05-23) — never approved in this form.** Under ADR-015 §5 the slice plan is reordered: `extract_params` becomes **new slice 4**, after the catalog (new slice 2) and the `generate` migration to catalog names (new slice 3). The two ordered path checks on the *sidecar path argument* in this Vision (realpath-first `.json` + `_within(--output-dir)`) carry forward unchanged — ADR-011 §3 second exclusion is unaffected. What changes is the **return contract**: path-typed fields in the response are catalog names via reverse lookup, with a basename fallback + INFO notice on catalog miss (ADR-015 §3 / MEDIUM-2 fold-in). This file is kept as the draft history; a fresh slice-4 Vision will be authored when slice 4 is reached, incorporating ADR-015's return-contract changes and any portability notes (case-folding hosts, security-auditor INFO-4).
**AI-Disclosure:** Claude (Opus 4.7, 1M context) authored; Grant to review and approve.

---

## Posture

> **Posture:** Boundary: integration (LLM-agent → comfyless → filesystem read). Risk factors: external exposure (LLM input drives a filesystem read path); near security truth (path allowlist + traceback strip + enumeration-oracle avoidance); sensitive data (returns stored generation params; reads files under the user's output root).

## Slice

Second implementation slice of ADR-011 (renumbered: old slice 3 → 2 per the 2026-04-30 cascade-collapse). Adds the `extract_params` MCP tool: a **read-only** tool that takes a caller-supplied path to a JSON sidecar, validates it through two ordered path checks, reads + schema-normalizes it, and returns the normalized [`COMFYLESS_SCHEMA`] params blob inline in the MCP response frame. No model loading, no network, no filesystem write.

It is the first MCP tool to expose a **caller-supplied file-read path**. All path-validation, audit, and traceback-strip scaffolding it needs already exists from slice 1 (`comfyless/mcp_server.py`: `_within`, `_emit_audit_line`, `_sanitize_error`, `_StartupConfig`, `_list_tools_impl`, `_call_tool_impl` name-dispatch, `_MCPHandlerError`). Slice 2 adds one `Tool` to `_list_tools_impl`, one dispatch branch in `_call_tool_impl`, and one `_handle_extract_params` handler.

## Four signals

- **Who** — the LLM agent, via an MCP client over stdio (same-uid trust boundary from the parent-process spawn relationship, per ADR-011 §6). It supplies one input: a filesystem `path`. The boundary crossed: caller-supplied path → the server opens and reads that file.
- **Data** — reads a JSON file from disk; returns a `COMFYLESS_SCHEMA`-normalized params blob (prompt, negative_prompt, seed, steps, cfg, sampler, scheduler, model/lora references, dimensions, …). Must never expose: file contents outside `--output-dir`; PNG bytes; arbitrary non-schema JSON content; tracebacks, internal module names, or absolute paths in error frames; whether an out-of-root or wrong-type path exists (enumeration / type oracle).
- **Boundary** — lives entirely in `comfyless/mcp_server.py`. Out of scope: PNG metadata extraction (an ADR-level exclusion, not a deferral — adding it needs a new ADR amendment); `list_models` / `list_loras` / `iterate` / `edit` (later slices); any read outside `--output-dir`; network; model loading; any write.
- **Failure** — fail-closed. Any gate failure rejects with a generic MCP error and an audit line, before any file read. No partial returns. Tracebacks never cross the boundary.

## Risk level

**L3 (Red Zone).** ADR-011 is Red Zone from day one for the LLM-agent surface; this slice adds a caller-supplied **file-read** path, which independently trips the project CLAUDE.md §12 trigger ("file reads from paths driven by external input"). Runs `code-reviewer` (Opus) **and** `security-auditor` (Opus) before commit; security-review output saved to `docs/security/review-slice-2-mcp-extract-params-<YYYY-MM-DD>.md` and referenced in the commit body.

## Intent

Add a read-only `extract_params` MCP tool that returns the `COMFYLESS_SCHEMA`-normalized params from a JSON sidecar **only** when its resolved path ends in `.json` and resolves within `--output-dir`, leaking nothing about paths it refuses.

## Invariants (must always be true)

1. **Two ordered checks, realpath FIRST.** The handler computes `resolved = os.path.realpath(path)` *before* any name inspection, then: (check 1) `resolved` must end in `.json`; (check 2) `_within(resolved, cfg.output_dir)` must hold. Both gates run and must pass before the file is opened. Resolving first is what defeats a `legit.json → evil.png` symlink — inspecting the surface name first would let it slip past. (ADR-011 §3 second exclusion; F-3 fold-in.)
2. **PNG / non-`.json` files are never read by this tool.** There is no PNG-byte-parsing code path reachable from `extract_params`. A path whose *resolved* name does not end `.json` is rejected at check 1. (ADR-011 §3 — PNG extraction not exposed; decision, not deferral.)
3. **Null-byte path rejected before realpath.** A `path` containing a NUL byte is rejected with a generic error before `os.path.realpath` is called (realpath raises `ValueError` on NUL); no file is touched. (Mirrors the slice-1 `generate` null-byte gate.)
4. **No type / enumeration oracle on the two security gates.** A check-1 failure ("resolved name not `.json`") and a check-2 failure ("resolved path outside `--output-dir`") return the **same** generic rejection message; the message does not state which check failed and does not echo the resolved absolute path. An attacker cannot use the response shape to distinguish "exists but wrong type" from "outside root" from "nonexistent-outside-root," nor to enumerate the filesystem beyond `--output-dir`.
5. **Returned blob is `COMFYLESS_SCHEMA`-normalized, not a raw file echo.** Only schema-recognized fields are returned; keys in the file that are not part of `COMFYLESS_SCHEMA` are dropped. The tool is a generation-params reader, not a general-purpose JSON file reader — a `.json` under `--output-dir` containing unrelated content yields only its schema-recognized subset (often empty), never arbitrary content verbatim.
6. **Read-only.** No model loading, no network, no filesystem write. `resolve_hf_path`, `_load_pipeline`, `generate`, and `cascade.run_one` are never reached from this handler. (Distinguishes this tool's blast radius from `generate`'s.)
7. **One audit line per invocation, success and rejection alike, on stderr only.** The line carries tool name (`extract_params`), the requested input path (paths are operator-visible and NOT redacted, per slice-1 invariant 5 / N22), status, and elapsed seconds. The audit line **does not** include the returned params blob — so an extracted `prompt` / `negative_prompt` never lands in the server log (PII-in-logs avoidance, consistent with ADR-011 §3b). stdout is never written to.
8. **Traceback strip at the MCP boundary.** Any internal exception (JSON parse error, IO error, normalization crash) is caught and converted to a sanitized MCP error; the response frame never contains a Python traceback, `.py:<line>` patterns, absolute paths, or internal module names. The full traceback is written to stderr for the operator. (Carry-forward of slice-1 invariant 13, via the existing `_sanitize_error` / `_MCPHandlerError` machinery.)
9. **No argparse / no CLI dispatch.** The `extract_params` handler does not `import argparse` and does not call `_run_cli_mode`, `_apply_overrides`, or `_load_params_file` in `comfyless/generate.py`. Sidecar normalization is performed via a path that does not route through CLI dispatch. (Carry-forward of slice-1 invariant 14 — note the collision in Open Questions: `_load_params_file` is the CLI's sidecar reader and is the obvious-but-forbidden reuse.)
10. **`tools/list` advertises exactly TWO tools: `generate` and `extract_params`.** This updates slice-1 invariant 6 (which asserted exactly one). No other slots are announced until their slices land.
11. **stdout carries only MCP JSON-RPC frames.** (Carry-forward of slice-1 invariant 7.)

## Failure semantics

- **Fail-closed at the gates:** NUL byte, resolved-name-not-`.json`, or resolved-path-outside-`--output-dir` → generic MCP rejection + audit line, **before any file open**. No normalization, no partial return.
- **Sanitized errors past the gates:** a path that passes both gates but does not exist, is unreadable, contains malformed JSON, or parses to a non-object (top-level list / scalar) → sanitized MCP error (error class without traceback or absolute path) + audit line. Within-`--output-dir` existence is the user's own directory, so a "not found"-class message there is acceptable — but it still must not leak a traceback or absolute path (invariant 8).
- **No partial success:** the tool either returns a complete normalized blob or an error; never a half-parsed structure.
- **Audit-line write failure** does not block the response (mirrors slice-1; increments the existing `_audit_write_failures` counter).

## Out of scope (explicit)

- **PNG metadata extraction** — an ADR-011 §3 *exclusion*, not a deferral. Adding it requires a new ADR amendment naming the PNG-byte-parse threat surface. Not this slice, not a later slice without that amendment.
- The other four tools: `list_models`, `list_loras`, `iterate`, `edit` stub — each its own follow-up slice per ADR-011 §3d.
- Reading sidecars outside `--output-dir`, or from arbitrary roots (e.g. `--model-base`). Containment root for `extract_params` is `--output-dir` only.
- Reconsidering the `--output-dir` containment choice in light of MCP `generate` no longer writing sidecars (see Open Questions) — that is a discussion to settle *before* approval, not code in this slice.
- Re-validating the returned blob as a *generate-ready* request (that is `generate`'s own validation job when the agent feeds params back in).
- HTTP/SSE transport; streaming; `resources/list` / `prompts/list`.

## Negative cases (required)

- **N1 — wrong extension:** `path` resolving to `foo.png` under `--output-dir` → check-1 rejection, file never opened.
- **N2 — `.json → .png` symlink (the headline attack):** a symlink `legit.json` under `--output-dir` whose target resolves to `evil.png` → rejection (proves realpath runs **before** the suffix check; a surface-name check would pass it).
- **N3 — `.json → outside-root` symlink:** a symlink `inside.json` under `--output-dir` whose target resolves to `/etc/secret.json` (outside root) → check-2 rejection (proves realpath-then-`_within` ordering).
- **N4 — traversal / out-of-root `.json`:** `path = "<output_dir>/../../etc/passwd.json"` (or any real `.json` outside `--output-dir`) → check-2 rejection, file never opened.
- **N5 — NUL byte:** `path` containing `\x00` → rejection before realpath, file never touched.
- **N6 — happy path:** a valid `.json` sidecar inside `--output-dir` → returns the normalized blob; success audit line on stderr.
- **N7 — unknown keys dropped:** fixture sidecar with a junk top-level key (e.g. `"__exfil__": "secret"`) plus valid params → response contains the schema params and does **not** contain `__exfil__` (invariant 5).
- **N8 — malformed JSON:** a `.json` under `--output-dir` with invalid JSON → sanitized error (asserts no `Traceback`, no `.py:<digits>`, no absolute path), audit line.
- **N9 — non-object JSON:** a `.json` under `--output-dir` whose top level is a list or a string → sanitized error.
- **N10 — no oracle:** the error frame for an N1 (wrong-extension) rejection and an N4 (outside-root) rejection are byte-identical generic messages and neither echoes the resolved absolute path (invariant 4).
- **N11 — tools/list count:** `_list_tools_impl` now returns exactly `["generate", "extract_params"]`; updates the slice-1 N-equivalent that asserted a single tool.
- **N12 — static no-CLI-dispatch:** source check on `comfyless/mcp_server.py` — the `extract_params` handler and module contain no `import argparse`, and no calls to `_run_cli_mode` / `_apply_overrides` / `_load_params_file` (invariant 9 carry-forward).
- **N13 — audit on stderr, not stdout:** capture both streams across an `extract_params` call; stdout = JSON-RPC frame only; the audit line is on stderr.
- **N14 — within-root nonexistent:** a `.json` path that passes both gates but does not exist on disk → sanitized "not found"-class error, audit line, no traceback (invariant 8 holds even for within-root reads).
- **N15 — audit omits returned params:** the audit line for a successful N6 call does **not** contain the extracted `prompt` / `negative_prompt` text (invariant 7).
- **N16 — input schema:** an `extract_params` call missing `path`, or with `path` non-string, is rejected by input validation before any filesystem touch.

## Proof hooks

- **Positive:** `./.venv/bin/python3 test_mcp_server.py` — new `extract_params` section exercises N6 against a fixture sidecar written into a temp `--output-dir`.
- **Negatives N1–N16** organized as sections inside `test_mcp_server.py` (no pytest dep — same `python3 test_<name>.py` invocation as the other suites; run via `./.venv/bin/python3` per ADR-013).
- **Static-source checks (N12)** via `inspect.getsource` / `ast` on `comfyless.mcp_server`.
- **All nine existing suites continue to pass — 1008/1008** (the `test_mcp_server.py` count grows; CLAUDE.md's suite-count line is updated in the closure step). Proves no slice-1 behavior regressed and `generate` still advertises/dispatches correctly alongside the new tool.

## Red Zone ownership

- **The two ordered path checks** (realpath-first ordering; `.json`-on-resolved; `_within(--output-dir)`): owned by **Grant** — AI-generated only, not sole author.
- **Enumeration/type-oracle avoidance** (unified generic rejection, no resolved-path echo, invariant 4): owned by **Grant**.
- **Return-blob exposure decision** (full paths in-frame vs basename-redacted; drop-unknown-keys normalization, invariant 5): owned by **Grant** — this is the data-exposure boundary. Proposed default below in Open Questions.
- **ADR-011 is the design source of truth.** Any divergence reverts to an ADR amendment before code lands.

## Open questions (resolve before approval / in `/change-slice`)

1. **Stale containment rationale.** ADR-011 §3's second exclusion justifies the `--output-dir` containment with "the agent extracts params from sidecars *its own prior `generate` calls* produced." But the 2026-05-02 amendment (§2) made MCP-driven `generate` stop writing sidecars to disk (inline-only). So the sidecars under `--output-dir` are now CLI/human-produced (or user-placed), not MCP-produced. **The security mechanism is unaffected** — the two ordered checks stand regardless of who wrote the file — but the *use-case narrative* shifts to "read sidecars from human CLI runs in the shared output root." Proposed resolution: keep the tool and the `--output-dir` containment as-is (mechanism is sound), and add a one-line ADR-011 Changelog note recording that the rationale narrative is updated post-no-sidecar-amendment. Confirm with Grant.
2. **Return-blob redaction.** Should the returned params carry full paths, or basename-redacted paths like the §3e PNG embedding? Proposed default: **full paths in-frame**, consistent with `generate`'s N25 inline-blob precedent ("the in-frame blob is the agent's authoritative record; only durable/shareable artifacts — PNG chunks — and logs are redacted"). The frame is same-uid and ephemeral. Confirm vs. redacting to basenames.
3. **Normalization implementation (Change-Plan-level, flagged here for the invariant-9 collision).** `_load_params_file` in `generate.py` is the CLI's sidecar reader — exactly the natural reuse, and exactly what invariant 9 (carry-forward of slice-1 invariant 14) forbids the MCP path from calling. The Change Plan must choose: (a) extract the pure read+normalize logic into a shared helper both surfaces call (ADR-012-style harmonization), or (b) inline `json.load` + `COMFYLESS_SCHEMA` normalization in the handler. Option (a) is cleaner long-term but widens the slice; option (b) keeps the slice contained. Decide in `/change-slice`.

## Pointers

- ADR: [ADR-011](../decisions/ADR-011-comfyless-mcp-server.md) — §2 `extract_params` row, §3 second exclusion (two ordered checks), 2026-04-28 F-3 fold-in.
- Predecessor Vision (substrate + carry-forward invariants 6/7/13/14): [slice-1-mcp-generate.md](slice-1-mcp-generate.md).
- Validator harmonization precedent (for Open Question 3 option a): [slice-machine-boundary-validator.md](slice-machine-boundary-validator.md) / ADR-012.
- Slice plan: ADR-011 §3d (renumbered 2026-04-30 — `extract_params` is slice 2).
