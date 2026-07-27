AI-Disclosure: Claude (Opus 5, security-auditor agent) authored; Claude (Sonnet 5) requested and applied the recommended hardening; Grant reviewed.

# Security review — `comfyless/mcp_server.py` + `comfyless/catalog.py` pyright drawdown

Date: 2026-07-27
Scope: the final comfyless/ pyright drawdown slice (ADR-042) — `comfyless/
mcp_server.py` (18 errors → 0), `comfyless/catalog.py` (a `resolve_reference`
signature widening this slice needed), reviewed together with `comfyless/
server.py`'s already-reviewed fix (see `review-server-validate-request-
typecheck-2026-07-27.md`). All three are §12 Red Zone surfaces.

**Coverage caveat (this reviewer's own instance had no Bash tool):** the
review was against working-tree file content plus targeted greps matching a
detailed change description, not a literal `git diff`. No unmentioned hunk
was found in the regions read, but this is weaker than a diff-based review —
noted here per the reviewer's own request to record it.

## Change under review

- `mcp_server.py`: ~5 call sites of `catalog.resolve_reference()` narrowed
  via `assert rr.cause is not None` (failure branch) / `assert rr.abs_path
  is not None and rr.name is not None` (success branch), since
  `ResolveResult`'s `Optional`-typed fields carry a discriminated-union
  invariant pyright can't see across the `.ok` check. Also: a `.get()` →
  subscript fix in the per-LoRA loop (key presence already checked), a new
  `if cfg.catalog_db_path is None: raise ...` guard in `_db_family_names`
  (defense-in-depth, matching the identical guard in the sibling
  `_handle_search`), and 2 `PIL.Image.LANCZOS` → `Image.Resampling.LANCZOS`
  fixes (Pillow 12.3.0 stub-only rename, same runtime value).
- `catalog.py`: `resolve_reference`'s `raw_ref: str` widened to `raw_ref:
  object` — the function's own first statement is an unconditional
  `isinstance(raw_ref, str)` gate that already handles non-str input
  gracefully (`cause="MalformedReference"`, never raises); the annotation
  undersold the function's actual defensive contract.

## Findings (all INFO — no blocking finding, no P0)

1. **`resolve_reference` signature widening verified safe.** The
   `isinstance` gate is the unconditional first executable statement; all 5
   production call sites checked, none relied on the static `str` to catch
   anything the runtime gate doesn't already catch. `test_mcp_server.py`'s
   non-str/malformed negative test still passes.
2. **`ResolveResult`'s discriminated-union invariant verified true on every
   return path** (7 returns walked in `catalog.py`) — none of the 8 new
   asserts are attacker-trippable today.
3. **All 8 asserts verified inside the `except BaseException as e:
   _sanitize_error(...) → raise ValueError(safe)` wrapper** around every MCP
   tool dispatch (`_call_tool_impl`, `mcp_server.py` ~1743-1790) — confirmed
   for `_handle_generate`, `_handle_generate_cascade`, the LoRA loop, and
   `_db_family_names`'s only caller path. No bypass found.
4. **Hardening applied (not required, but adopted):** on the 4
   *failure*-branch sites (`if not rr.ok: assert rr.cause is not None; raise
   _reference_error(rr.cause)`), the reviewer noted an assert-based
   invariant break would surface to the agent as a distinguishable
   `internal_error: AssertionError` frame instead of the byte-identical
   `_UNIFORM_REFERENCE_ERROR` every other rejection produces — eroding the
   ADR-015 HIGH-1 oracle-closure property (keystone test N5) if
   `resolve_reference` ever regresses. Swapped all 4 to `raise
   _reference_error(rr.cause or "UnknownName")`, matching the `or {}` idiom
   already used twice elsewhere in this file (`mcp_server.py:1952/2249`) and
   the pattern the sibling server.py review established. The 4
   *success*-branch asserts (`assert rr.abs_path is not None and rr.name is
   not None`) were correctly left as asserts — no non-raising fallback is
   safe there (a silently-wrong path would reach `_check_paths`/load).
5. **`_db_family_names`'s new guard verified behaviorally inert today** —
   its one caller already performs the identical check with a byte-identical
   `_MCPHandlerError` message before ever reaching this function.
6. **`Image.LANCZOS` → `Image.Resampling.LANCZOS` verified behavior-neutral**
   — the surrounding DoS-bounding caps (pixel/byte ceilings, iteration cap,
   fail-closed raise) are untouched.
7. **Pre-existing, not introduced by this slice:** `server.py`'s
   `req.update(result.payload)` (line ~175) consumes the same
   `ValidationResult` dataclass's `Optional[dict] payload` unguarded, on the
   same no-exception-handler accept loop — mirror of the `.error` issue the
   earlier review fixed. Already recorded in the server.py review; restated
   here so it isn't lost when that file's count is next touched. Out of
   scope for this slice.

## Verdict

**Approved as-is**, with the failure-branch hardening (item 4) applied
before commit. `mise exec -- pyright comfyless/mcp_server.py comfyless/
catalog.py comfyless/server.py` — 0 errors. `test_mcp_server.py` 702/702,
`test_catalog_db.py` 125/125, `test_server_robustness.py` 202/202 — all
green, no regressions.
