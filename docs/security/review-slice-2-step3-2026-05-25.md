# Security review — ADR-015 slice 2 step 3 (`--catalog` flag + spawn-time wire-up)

**Date:** 2026-05-25
**Reviewer agent:** `security-auditor` (Opus)
**Reviewer model:** `claude-opus-4-7[1m]`. Invocation passed `model: "opus"` explicitly per the Claude Code 2.1.117 frontmatter-pin workaround documented in CLAUDE.md §5A.
**Slice docs:** `docs/vision/slice-2-mcp-catalog.md`, `docs/decisions/ADR-015-mcp-opaque-handles.md`
**AI-Disclosure:** Claude (Opus) authored; Grant reviewed.
**Verdict:** CLEAN — 6 INFOs, 0 BLOCK / HIGH / MEDIUM / LOW.

---

## Summary

ADR-015 slice 2 step 3 wires the spawn-time catalog into the MCP-stdio
LLM-agent surface: a new `--catalog` click option on `main()`, a
`catalog: dict` slot on `_StartupConfig`, an explicit NUL-byte pre-check
in `_validate_startup_args`, a lazy `comfyless.catalog.build_catalog`
call, and a `CatalogBuildError → click.BadParameter(param_hint=
"--catalog") from None` wrap.

The threat model is: the operator owns `--model-base`, `--output-dir`,
`--default-model`, and `--catalog` (process-spawn channel); the LLM
agent is adversarial input on the JSON-RPC tools/call channel. Step 3
sits entirely on the operator startup channel — the agent does not
observe stderr and cannot influence catalog contents. The change
touches one of three §12 security-review surfaces (`comfyless/
mcp_server.py` IPC + LLM agent surface) and is Red Zone per the
per-project CLAUDE.md.

## Coverage

Reviewed:
- `comfyless/mcp_server.py` — full file, focus on lines 300-414
  (`_StartupConfig` + `_validate_startup_args`) and 978-1066 (click
  decorators + `main()`)
- `comfyless/catalog.py` — full file as caller-contract reference
  (contents already audited at Step 2, NOT in scope here)
- `test_mcp_server.py` lines 2063-2298 (Step 3 test block + module
  hygiene tail)
- `comfyless/server.py:158-173` (`_within` + `_check_paths` semantics)

Not reviewed (and why):
- The inside of `_handle_generate`, `_handle_generate_cascade`,
  `_call_tool_impl`, `_list_tools_impl` body changes — out of scope
  per the audit brief (Step 3 does not modify them; I14 was confirmed
  by inspection of file state at HEAD, not by re-auditing logic).
- `docs/vision/slice-2-mcp-catalog.md` body — invariants quoted in the
  brief; invariants statements were trusted rather than re-read.
- The Step-1 / Step-2 already-cleared findings in `comfyless/
  catalog.py` — explicit out-of-scope per the brief.

## Findings

**[INFO] Catalog path is not required to live under `--model-base`**
Location: `comfyless/mcp_server.py:1015-1027` (option declaration),
`comfyless/mcp_server.py:398-406` (build call), `comfyless/catalog.py:
451-491` (`_parse_manifest` open).
Risk: `--catalog` accepts any filesystem path the operator can read.
An operator who pastes the wrong absolute path could feed the server
a manifest from outside `--model-base`. Because the manifest can only
declare entries whose realpath lands under `--model-base` (enforced in
`_parse_manifest_entry`), the worst outcome is "operator points at the
wrong JSON, startup fails closed with a message naming the file." The
catalog cannot grant access to any path the agent could not already
address via `--model-base`. This is the operator startup channel; the
operator is trusted. **No fix needed.** Requiring `--catalog` to live
under `--model-base` would only constrain operator ergonomics without
closing an attack — it might even hurt operations where manifests are
checkout-managed alongside service configs outside the model tree.

**[INFO] NUL pre-check is redundant against `main()` but load-bearing for direct in-process callers**
Location: `comfyless/mcp_server.py:387-391`.
Risk: `click.Path.convert` calls `os.fspath` / `os.stat` on the value,
which rejects NUL bytes at parse time with `ValueError: embedded null
byte` and surfaces as a non-zero exit before `_validate_startup_args`
is even called. The explicit pre-check is therefore dead code along
the CLI path. However, test `Step3 F5b` directly invokes
`_validate_startup_args(catalog="manifest\x00.json")` — the pre-check
is what produces the `click.BadParameter` there instead of the lazy
import + `os.path.isfile` raising a bare `ValueError`. The
defense-in-depth posture is correct and matches the established
`_handle_generate` NUL-gate pattern (server.py:138-149,
mcp_server.py:482-502 per Step-2 MEDIUM-1 fold). **No fix needed.**

**[INFO] `CatalogBuildError` messages echo operator-supplied paths/repo IDs to stderr**
Location: `comfyless/mcp_server.py:398-406`; messages defined in
`comfyless/catalog.py:237-241, 250-253, 363-440, 462-499`.
Risk: Catalog-layer messages name the offending manifest entry, the
manifest path itself, and (for HF entries) the repo ID. This is
intentional — the operator needs the offending name to debug startup
failures. The agent does NOT observe stderr (MCP stdio reserves
stdout for JSON-RPC; stderr is the operator console / process
supervisor log). Even if stderr leaked into a shared log surface, the
values echoed are operator-authored (manifest contents) or
operator-configured (`--catalog` path arg) — no agent-controlled
string can reach this code path in Step 3, because the agent has no
influence over spawn-time arguments. The Step 3 wrap correctly uses
`from None` to suppress the exception chain so click's pretty-printer
does not double-render the trace. **No fix needed.**

**[INFO] `_StartupConfig.catalog: dict` type erasure**
Location: `comfyless/mcp_server.py:300-328`.
Risk: The slot is typed as `dict` rather than `comfyless.catalog.
CatalogDict` to keep the module-top import surface stdlib-only
(consistent with Step 1's HIGH-1 fold and Step 2's lazy-import
discipline). Read-after-spawn callers (future Step 4 `list_models` /
`list_loras`, future Step 5 reference resolution in
`_handle_generate`) will need to add their own runtime checks if they
want to defend against malformed catalog shapes — but the only writer
is `build_catalog` and its return contract is fixed by the typed
`CatalogDict` (TypedDict, runtime equivalent to `dict`). No safety
regression. **No fix needed.**

**[INFO] Tool-surface invariant I14 is structurally guaranteed**
Location: `comfyless/mcp_server.py:422-428`.
Risk: `_list_tools_impl(cfg)` ignores `cfg` and returns a literal
one-element list with `name="generate"`. No catalog content can reach
the tool-list response in Step 3 because the function does not read
`cfg.catalog`. The error-response and audit-line code paths likewise
do not synthesize tool names from catalog content. Test `Step3 I14`
is a useful regression latch but the invariant is currently
structural — a future Step-4 edit is the only place where catalog
content could leak into the tool list, and that's out of slice-2-
step-3 scope. **No fix needed.**

**[INFO] F4 wording: "symlink-escapes" describes the failure cell, not the planted shape**
Location: `test_mcp_server.py:2146-2173` (Step 3 F4).
Risk: The Step-3 F4 case writes a manifest entry whose `target` is an
absolute path outside `--model-base`. It exercises the catalog-layer
`_within(abs_path_real, model_base_real)` check and confirms that the
`CatalogBuildError` propagates through `_validate_startup_args` and
surfaces as a non-zero click exit naming `--catalog`. It does NOT
plant an actual symbolic link whose realpath escapes — that case is
exercised at the catalog unit level (Step-2 tests). The original
inline comment "F4: manifest entry symlink-escapes" was slightly
misleading. Adequate proof of I7 wiring for the `outside-base` cell
of the truth table; the symlink-specific cell is covered by Step-2
unit tests. **Fixed:** comment updated 2026-05-25 to "manifest entry's
target realpath-escapes" with explicit note that the symlink variant
is exercised at the catalog-unit level.

**[INFO] `main()` callers pass `catalog` by keyword (click decorator binding)**
Location: `comfyless/mcp_server.py:1041-1066`.
Risk: Click invokes the decorated function by keyword from the
parsed param map; the parameter order in the function signature is
harmless even though `catalog` was inserted between `default_model`
and `mcp_max_iterations`. The only direct caller outside click is
the test file (Step 3 success-case blocks), and each invocation uses
keyword arguments. There is no positional callsite to break. **No fix
needed.**

## Conclusion

**No findings at CRITICAL / HIGH / MEDIUM / LOW severities.** Step 3 is
a clean wiring slice: the catalog is built once at spawn, held on
`_StartupConfig`, and the failure surface (`CatalogBuildError`) is
adapted to click's startup-error contract with `param_hint="--catalog"`
and `from None` chain suppression. The NUL pre-check is correctly
placed before the lazy import. The tool-surface invariant I14 holds
structurally. The 5 CliRunner negative cases plus the direct
in-process NUL case provide adequate proof of fail-closed wiring; the
catalog-layer's own truth-table cells were covered by Step-2 unit
tests.

Areas considered (each cleared):

1. Path-allowlist boundary on `--catalog` itself — operator-trusted
   channel; no containment check needed.
2. NUL ordering vs. lazy `build_catalog` call — correctly placed;
   defense-in-depth for non-CLI callers.
3. `CatalogBuildError` message content — operator-facing stderr; no
   agent-controlled input on this path.
4. `_StartupConfig.catalog` type erasure — no safety regression;
   matches established lazy-import discipline.
5. Audit-line discipline — Step 3 adds no new audit lines and does
   not modify existing ones (confirmed).
6. Tool-surface invariant — `_list_tools_impl` does not read
   `cfg.catalog` (structural guarantee).
7. Test fail-closed coverage — F1-F5b adequate; F4 comment wording
   was imprecise (folded).
8. `main()` signature ordering — keyword-bound by click; no
   positional callsite to break.
