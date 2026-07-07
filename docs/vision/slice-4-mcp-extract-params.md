# Slice 4 Vision — `extract_params` MCP tool (JSON-sidecar reader + catalog enrichment)

**Date:** 2026-07-07
**ADR:** [ADR-011](../decisions/ADR-011-comfyless-mcp-server.md) — §2 (`extract_params` row), §3 second exclusion (PNG NOT exposed; two ordered checks on the sidecar path argument), 2026-04-28 F-3 fold-in. [ADR-015](../decisions/ADR-015-mcp-catalog-reference-resolution.md) §3 (return catalog names; basename fallback + INFO notice on miss) / §5 slice-4. [ADR-022](../decisions/ADR-022-catalog-service.md) — the metadata DB as the enrichment source (a Changelog amendment extending its consumers to `extract_params`).
**Status:** accepted (Grant approved 2026-07-07; implement Step 4a first). Supersedes the draft [slice-2-mcp-extract-params.md](slice-2-mcp-extract-params.md) (never approved; its two ordered path checks carry forward verbatim, its return contract is replaced by the ADR-015 names contract + ADR-022 enrichment settled here).
**AI-Disclosure:** Claude (Opus 4.8, 1M context) authored; Grant to review and approve.

---

## Posture

> **Posture:** Boundary: integration (LLM-agent → comfyless → filesystem read + read-only metadata-DB read). Risk factors: external exposure (LLM input drives a filesystem read path); near security truth (path gates, enumeration/type-oracle avoidance, no-abs_path-egress); sensitive data (returns stored generation params + catalog metadata read from the user's output root and metadata DB).

## Slice

Fourth implementation slice of ADR-011 (per the ADR-015 §5 reorder; slices 1/2/2b/3/3b shipped). Adds the read-only `extract_params` MCP tool: it takes a caller-supplied path to a JSON sidecar, validates it through two ordered path checks, reads + schema-normalizes it, **reverse-resolves each weight reference to its live-catalog name, enriches those names with metadata-DB detail when a DB is present**, and returns the result inline in the MCP response frame. No model loading, no network, no filesystem write.

It is the first MCP tool to expose a **caller-supplied file-read path**. All path-validation, audit, and traceback-strip scaffolding already exists from slice 1 (`_within`, `_emit_audit_line`, `_sanitize_error`, `_StartupConfig`, `_list_tools_impl`, `_call_tool_impl` name-dispatch, `_MCPHandlerError`). The catalog name namespace, `_basename_or_repo_id`, and `_discard_notice`-shaped notices exist from slices 2/3. The metadata DB read surface (`catalog_db.connect_readonly` / `search`) exists from ADR-022 S5.

**Precursor step (4a).** Today the server consults the metadata DB only when the operator passes `--catalog-db`, even though `catalog_cli` builds it at a canonical default path (`catalog_db.DEFAULT_DB_PATH` = `~/.local/share/comfyless/catalog.sqlite`). Step 4a makes the server **auto-discover the DB at that default path** when `--catalog-db` is not given: file present → use it; absent → run without. This turns the DB from an opt-in flag into the normal case (enrichment and `search` become available whenever a DB has been built) without coupling the load path to it. Independently shippable, its own commit.

## Two-catalog architecture (settled — the design premise of this slice)

The project has two catalogs with **deliberately different cadences and trust levels**; this slice preserves the split rather than collapsing it (Grant, 2026-07-07):

- **`cfg.catalog`** (in-process `CatalogDict` from `comfyless.catalog.build_catalog`) is the **live load authority**. Built by a fast scan of `--model-base` (+ manifest) at *server spawn*, so it reflects the filesystem as it is now. `resolve_reference` uses it for name→abs_path with request-time `exists`/`_within` fail-closed checks; `abs_path` never crosses the MCP boundary. It is always present (generate needs it) and always fresh.
- **The metadata DB** (`comfyless.catalog_db`, SQLite at `DEFAULT_DB_PATH`) is the **offline-built, eventual-consistency metadata plane**. `catalog_cli build` produces it (audit + sha256 every file + optional civitai *network* enrichment). Same name namespace (built from the same `build_catalog` scan — `catalog_builder.py:209`), enriched with descriptions / `strength_rec` / `sampler_rec` / `trigger_words` / `usage_tips` / classification / families. "DB never in load path" is an ADR-022 invariant.

**Why the split holds (why `cfg.catalog` is not redundant):** (1) *Freshness* — a weight added after the last DB build is in the live scan but not the DB; a DB-backed load path would reject brand-new real files until a rebuild. (2) *Build cost* — the DB build hashes every file + optional network; far too heavy for every spawn. (3) *Trust surface* — the DB is writable by the offline pipeline and carries `excluded`/`stale` rows; the live scan is anchored to the actual filesystem under `--model-base`, the tighter reviewed boundary.

**Consequence for this slice:** **name resolution runs against the live `cfg.catalog`** (guarantees the returned name is fresh and replay-valid); **the DB is enrichment-only**, queried by the already-resolved name. The DB never participates in resolving or loading a reference.

## Four signals

- **Who** — the LLM agent, via an MCP client over stdio (same-uid trust boundary from the parent-process spawn relationship, ADR-011 §6). It supplies one input: a filesystem `path`. Boundary crossed: caller-supplied path → the server opens and reads that file.
- **Data** — reads a JSON sidecar from disk (under `--output-dir`); reads the metadata DB read-only (when present); returns a `COMFYLESS_SCHEMA`-normalized params blob whose weight references are **catalog names** (never abs_paths), optionally annotated with metadata-DB detail, plus a `notices` array. Must never expose: file contents outside `--output-dir`; PNG bytes; arbitrary non-schema JSON; any absolute path or directory; tracebacks / internal module names; whether an out-of-root or wrong-type path exists (enumeration / type oracle).
- **Boundary** — `comfyless/mcp_server.py` (one `Tool` added to `_list_tools_impl`, one dispatch branch in `_call_tool_impl`, one `_handle_extract_params` handler, its description/schema constants, and a small name-resolution + DB-enrichment helper) plus the step-4a startup change in `main()`/`_validate_startup_args`. Out of scope: PNG metadata extraction (ADR-level exclusion); `iterate` / `edit` (later slices); any read outside `--output-dir`; network; model loading; any write; any change to `resolve_reference` or the load path.
- **Failure** — fail-closed. Any gate failure rejects with a generic MCP error + audit line, before any file read. No partial returns. Tracebacks never cross the boundary. DB-read failure degrades to no-enrichment (never fails the call, never leaks a DB path).

## Risk level

**L3 (Red Zone).** ADR-011 is Red Zone from day one for the LLM-agent surface; this slice adds a caller-supplied **file-read** path (§12 trigger: "file reads from paths driven by external input"). Runs `code-reviewer` (Opus) **and** `security-auditor` (Opus) before commit; security output saved to `docs/security/review-slice-4-mcp-extract-params-<YYYY-MM-DD>.md` and referenced in the commit body. Step 4a (startup DB auto-discovery) runs `code-reviewer` (Opus); it changes what the server exposes (`search` + enrichment auto-available) so it also runs `security-auditor` (Opus).

## Intent

Add a read-only `extract_params` MCP tool that returns the `COMFYLESS_SCHEMA`-normalized params of a JSON sidecar **only** when its resolved path ends in `.json` and resolves within `--output-dir`, with every weight reference rendered as a live-catalog name (basename + INFO notice on catalog miss) and enriched with metadata-DB detail when available — leaking nothing about paths it refuses or resolves.

## Invariants (must always be true)

### Path gates (carried forward verbatim from the slice-2 draft; ADR-011 §3 / F-3)

1. **Two ordered checks, realpath FIRST.** `resolved = os.path.realpath(path)` is computed *before* any name inspection, then: (check 1) `resolved` must end in `.json`; (check 2) `_within(resolved, cfg.output_dir)` must hold. Both gates run and must pass before the file is opened. Resolving first defeats a `legit.json → evil.png` symlink.
2. **PNG / non-`.json` files are never read by this tool.** No PNG-byte-parsing path is reachable from `extract_params`; the handler never calls `_load_params` / `_load_params_from_png` (the CLI readers that dispatch to PNG). A path whose *resolved* name does not end `.json` is rejected at check 1.
3. **Null-byte path rejected before realpath.** A `path` containing a NUL byte is rejected with a generic error before `os.path.realpath` is called; no file is touched.
4. **No type / enumeration oracle on the two gates.** A check-1 failure and a check-2 failure return the **same** generic rejection; the message names neither which check failed nor the resolved absolute path.

### Read + normalize

5. **Sidecar read is JSON-only and routes through the pure normalizer.** The handler does `json.load` on the gated file and normalizes via `_validate_params` (the pure `COMFYLESS_SCHEMA` cleaner in `comfyless/generate.py`) — never `_load_params` / `_load_sidecar` / `_apply_overrides` / `_run_cli_mode`, and it does not `import argparse`. No CLI dispatch, no PNG path, no override machinery.
6. **Returned blob is `COMFYLESS_SCHEMA`-normalized, not a raw file echo.** Only schema-recognized fields survive; unknown top-level keys in the file are dropped (a `.json` under `--output-dir` with junk content yields only its schema subset, never arbitrary content verbatim).
7. **Read-only.** No model loading, no network, no filesystem write. `resolve_hf_path`, `_load_pipeline`, `generate`, `cascade.run_one` are never reached.

### Reference rendering — names, never paths (ADR-015 §3)

8. **No absolute path or directory crosses the boundary, in any field or notice.** Every path-typed field is either rewritten to a name/basename or dropped. Specifically the handler:
   - resolves `model` → live-catalog name (kinds `model`/`transformer`);
   - drops `transformer_path`, replacing it with `transformer` = the resolved name (omitted when absent);
   - rewrites `loras[]` to `[{name, weight, …enrichment}]`, dropping the `path` key; each `weight` is coerced to number-or-`None` so a non-numeric value can never smuggle a string across (code-reviewer slice-4 LOW);
   - **drops** `vae_path`, `text_encoder_path`, `text_encoder_2_path` (off the MCP surface per ADR-015 OQ-A, mirroring `_resolved_params_as_names`);
   - **drops** `output_path`, `savepath` (write destinations — abs paths, ADR-015 §3);
   - **drops** `lora_warnings` (its strings embed abs_paths — slice-3 MEDIUM-1).
   Every non-path field (`prompt`, `negative_prompt`, `seed`, `steps`, `cfg_scale`, `true_cfg_scale`, `width`, `height`, `sampler`, `schedule`, `model_family`, LoRA `weight`) passes through verbatim.

   **Cascade (Stable Cascade) sidecars are a follow-on — step 4d, NOT this core step.** The real on-disk cascade sidecar is FLAT: `cascade.py` `dispatch()` spreads the config at top level (`stage_c`/`stage_b`/`stage_a`/`scaffolding_repo`/`config_source`/`output_path` are top-level keys), never nested under a `cascade_config` object (verified `cascade.py:930-950`; the MCP cascade handler writes no JSON sidecar at all). Every such key is non-schema, so `_validate_params` drops them ALL — a cascade sidecar is therefore **safe** (no abs-path egress) but yields no stage-name replay until step 4d resolves the flat top-level `stage_*` against the catalog. Bundling cascade into the core step was rejected on review: the code-reviewer found the nested-`cascade_config` handling was dead code against real sidecars and its synthetic test gave false confidence; a denylist over caller-controlled cascade bytes also risked a future abs-path egress (both-reviewer finding). Mirrors the slice-3 (non-cascade) / slice-3b (cascade) split.
9. **Resolution runs against the LIVE `cfg.catalog`, never the DB.** For each weight reference the handler matches `os.path.basename(value)` against a kind-scoped index built from `cfg.catalog` entries' `os.path.basename(abs_path)` (basename-to-basename, so file extensions line up and manifest-renamed entries still match by real filename). When more than one entry shares a `(kind, basename)`, the sidecar's `model_family` disambiguates; if still ambiguous, it is treated as a miss (invariant 10). No filesystem operation is performed on any path taken from sidecar content. **Hit → the catalog name** (a live, replay-valid `resolve_reference` key).
10. **Catalog miss → basename + INFO notice, never the directory.** A reference not resolvable in the live catalog renders as `os.path.basename(value)` (or the value unchanged if it is an HF repo ID, via `_basename_or_repo_id`) plus one `notices` entry `{"level":"INFO","message":"reference not in catalog; returned as filename"}` (ADR-015 §3 / MEDIUM-2). The absolute directory never appears. The surfaced basename is the accepted one-bit signal (ADR-015 §3 rationale: the sidecar is user-produced under `--output-dir`; the same-uid agent could read it directly).

### Enrichment — DB is metadata-only (ADR-022)

11. **Enrichment is by resolved name, read-only, and never in the resolution path.** After a reference resolves to a live-catalog name, and only when a metadata DB is available (step 4a auto-discovery or explicit `--catalog-db`), the handler looks that exact name up in the DB (read-only connection) and attaches an allowlisted `metadata` block (from `classification`, `best_description`: `description`/`usage_tips`/`trigger_words`/`strength_rec`/`sampler_rec`/`model_name`/`source`) to that reference. The DB is queried by name only — it never resolves a path and never influences which name was chosen. A basename-fallback (miss) reference is **not** enriched.
12. **Enrichment strictly allowlists fields and never leaks a path.** The attached `metadata` block contains only the description/recommendation fields above; it never carries `abs_path`, `root`, `relative_path`, or any filesystem string. `trigger_words` is re-validated on read (list of ≤64 short strings) exactly as `_handle_search` does. DB unavailable, name-absent, or DB-read error → the reference simply carries no `metadata` block (invariant 8's no-leak guarantee holds regardless).

### Boundary hygiene

13. **One audit line per invocation, success and rejection alike, on stderr only.** Carries tool name (`extract_params`), the requested input `path` (operator-visible, slice-1 invariant 5), status, elapsed seconds. It does **not** include the returned params blob or any enrichment text — an extracted `prompt`/`negative_prompt` or a `trigger_words` list never lands in the log (PII-in-logs avoidance). stdout carries only MCP JSON-RPC frames.
14. **Traceback strip at the MCP boundary.** Any internal exception (JSON parse, IO, normalization, DB read) is caught and converted to a sanitized MCP error; the frame never contains a traceback, `.py:<line>`, an absolute path, or an internal module name. Full traceback to stderr for the operator.
15. **`tools/list` advertises `extract_params` alongside the existing tools.** The list becomes `generate`, `list_models`, `list_loras`, `list_transformers`, `extract_params`, and (conditionally, when a DB is available) `search`. `extract_params` is advertised unconditionally — it degrades gracefully without a DB (names still resolve via the live catalog; only enrichment is skipped).

### Step 4a (startup DB auto-discovery)

16. **Auto-discovery only fills an unset `--catalog-db`, read-only, fail-open.** When `--catalog-db` is not supplied and a readable file exists at `catalog_db.DEFAULT_DB_PATH`, the server adopts it; otherwise `catalog_db_path` stays `None`. An explicit `--catalog-db` is never overridden. Adoption performs the same read-only openability validation the explicit path already gets; a default file that fails validation is treated as absent (server still starts), never a hard startup failure. No write, no build, no network at startup.

## Failure semantics

- **Fail-closed at the gates:** NUL byte, resolved-name-not-`.json`, or resolved-path-outside-`--output-dir` → generic MCP rejection + audit line, **before any file open**.
- **Sanitized errors past the gates:** a path that passes both gates but does not exist, is unreadable, is malformed JSON, or parses to a non-object → sanitized MCP error (class only, no traceback/abs-path) + audit line.
- **Enrichment degrades, never fails:** any DB-read problem (no DB, name absent, corrupt row, connect error) yields a response with no `metadata` block on the affected reference — never an error, never a leaked DB path.
- **No partial success** on the params blob: complete normalized blob or an error, never a half-parsed structure.
- **Audit-line write failure** does not block the response (increments the existing `_audit_write_failures` counter).

## Out of scope (explicit)

- **PNG metadata extraction** — ADR-011 §3 *exclusion*, not a deferral; adding it needs a new ADR amendment naming the PNG-byte-parse threat surface.
- **Making the DB the load/resolution authority** — explicitly rejected above; `cfg.catalog` stays the live authority, DB stays enrichment-only.
- **DB *fuzzy* search as the resolver** — resolution is exact (name/basename+family) against the live catalog; `catalog_db.search`'s LIKE/FTS ranking is for agent browsing (`search` tool), not deterministic replay resolution.
- `iterate` (slice 5), `edit` stub (slice 6); reading sidecars outside `--output-dir`; HTTP/SSE transport; streaming; re-validating the returned blob as generate-ready (that is `generate`'s job when the agent replays).

## Negative cases (required)

Path gates (carried forward): **N1** wrong extension `foo.png` → check-1 reject, file never opened. **N2** `.json → .png` symlink → reject (realpath before suffix). **N3** `.json → /etc/secret.json` (outside root) symlink → check-2 reject. **N4** traversal `<out>/../../etc/passwd.json` → check-2 reject, file never opened. **N5** NUL byte → reject before realpath. **N10** the N1 and N4 rejection frames are byte-identical and neither echoes a resolved absolute path.

Read/normalize: **N6** happy path — valid `.json` under `--output-dir` → normalized blob, success audit on stderr. **N7** junk top-level key (`"__exfil__":"secret"`) → absent from response. **N8** malformed JSON → sanitized error (no `Traceback`, no `.py:<digits>`, no abs path). **N9** non-object JSON (top-level list/scalar) → sanitized error. **N-newkeys** — `_validate_params` is used, not `_load_sidecar`/`_load_params` (static-source check: handler + module contain no `import argparse`, no `_run_cli_mode`/`_apply_overrides`/`_load_params`/`_load_sidecar` call).

Names/no-leak (the load-bearing ones): **N11** a sidecar carrying abs `model`/`transformer_path`/`loras[].path`/`vae_path`/`text_encoder*`/`output_path`/`savepath`/`lora_warnings` → response body contains **no** abs directory string (`/mnt/…`, `--model-base`, `--output-dir` roots), no `abs_path`/`path`/`transformer_path`/`vae_path`/`output_path`/`savepath`/`lora_warnings` keys. **N12** in-catalog reference → rendered as its live-catalog name; that name is a valid `resolve_reference` key (round-trips). **N13** not-in-catalog reference → rendered as bare basename (no directory) + exactly one INFO notice `"reference not in catalog; returned as filename"`. **N14** HF-repo-id reference → passed through unchanged (no basename split), no notice. **N15** ambiguous `(kind, basename)` unresolved by family → treated as miss (basename + notice), never a wrong-name guess. **N16 (descope)** a REAL flat cascade sidecar (top-level `stage_*` / `config_source` / `output_path`) → all cascade/path keys dropped by the normalizer, **no abs path anywhere in the response**; non-path params (`prompt`/`seed`/`model_family`) survive. (Flat `stage_*` → names is step 4d.)

Enrichment: **N17** DB present + resolved LoRA name has a description row → response carries an allowlisted `metadata` block (e.g. `strength_rec`/`trigger_words`); the block contains no `abs_path`/`root`/`relative_path`. **N18** DB absent (no `--catalog-db`, no default file) → happy path still returns names, simply no `metadata` blocks; `extract_params` still advertised. **N19** DB present but resolved name absent from DB → no `metadata` block, no error. **N20** basename-fallback (miss) reference → never enriched. **N21** `trigger_words` stored as a non-list / oversized → read-boundary guard yields a capped list-of-short-strings or omits it (mirrors `_handle_search`).

Boundary: **N22** audit on stderr not stdout; stdout = JSON-RPC only. **N23** success audit line omits the extracted `prompt`/`negative_prompt` and any `trigger_words`. **N24** input schema — call missing `path`, or `path` non-string → rejected before any filesystem touch.

Step 4a: **N25** no `--catalog-db`, default file exists + readable → server adopts it; `search`/enrichment available. **N26** no `--catalog-db`, no default file → `catalog_db_path` stays `None`, server starts, `search` unadvertised, `extract_params` still advertised. **N27** explicit `--catalog-db` given → default-path discovery does not override it. **N28** default file present but fails read-only validation → treated as absent, server still starts (fail-open, no hard error).

## Proof hooks

- **Positive:** `./.venv/bin/python3 test_mcp_server.py` — new `extract_params` section exercises N6/N12/N17 against fixture sidecars + a fixture DB written into temp dirs; step-4a section exercises N25–N28.
- **Negatives N1–N28** as sections in `test_mcp_server.py` (script-style, no pytest; run via `./.venv/bin/python3` per ADR-013). Static-source checks (N-newkeys) via `inspect.getsource`/`ast`.
- **All sixteen suites continue to pass** (current 2259; `test_mcp_server.py` count grows). CLAUDE.md suite-count line updated in the closure step.

## Red Zone ownership

- **The two ordered path checks** (realpath-first; `.json`-on-resolved; `_within(--output-dir)`) and **enumeration/type-oracle avoidance** (unified generic rejection, no resolved-path echo) — owned by **Grant**; AI-generated only.
- **The no-abs_path-egress contract** across the names + enrichment rendering (invariants 8/10/12) — owned by **Grant**; this is the data-exposure boundary.
- **The two-catalog split** (live authority vs enrichment-only DB; resolution never through the DB) — decided by **Grant** 2026-07-07; this Vision records it.
- **ADR-011/015 are the design source of truth; ADR-022 governs the DB.** Any divergence reverts to an ADR amendment before code lands.

## Open questions

None blocking. Resolved during design: return contract = catalog names (ADR-015, not the draft's "full paths" OQ2); normalization = the pure `_validate_params` (draft OQ3 option b, no CLI reuse); containment root stays `--output-dir` (draft OQ1); DB behavior = degrade gracefully with default-path auto-discovery (Grant 2026-07-07); resolution shape = exact name/basename+family against the live catalog, not fuzzy DB search (Grant 2026-07-07).

## Pointers

- ADR: [ADR-011](../decisions/ADR-011-comfyless-mcp-server.md) §2/§3; [ADR-015](../decisions/ADR-015-mcp-catalog-reference-resolution.md) §3/§5; [ADR-022](../decisions/ADR-022-catalog-service.md) (enrichment consumer amendment).
- Predecessor Visions: [slice-1-mcp-generate.md](slice-1-mcp-generate.md) (scaffolding), [slice-3-mcp-generate-catalog.md](slice-3-mcp-generate-catalog.md) (`_resolved_params_as_names`, the names contract), draft [slice-2-mcp-extract-params.md](slice-2-mcp-extract-params.md) (path-gate invariants).
- Reused helpers: `_basename_or_repo_id`, `_resolved_params_as_names` / `_resolved_cascade_params_as_names` (rendering shape), `_discard_notice` (notice shape), `_handle_search` / `catalog_db.search` (enrichment projection + `trigger_words` guard), `_validate_params` (pure normalizer).
