# Slice 2 Vision — Catalog infrastructure + `list_models` / `list_loras` MCP tools

**Date:** 2026-05-23
**ADR:** [ADR-015](../decisions/ADR-015-mcp-catalog-reference-resolution.md) (Status: accepted; rounds 1+2 CLEAN). Slice-2 row of the [ADR-015 §5](../decisions/ADR-015-mcp-catalog-reference-resolution.md) revised plan. Parent: [ADR-011](../decisions/ADR-011-comfyless-mcp-server.md) (2026-05-23 Changelog).
**Status:** approved by Grant 2026-05-23 (open questions resolved: OQ1 new `comfyless/catalog.py` module; OQ2 manifest-only `target_family`; OQ3 shape-detection extraction in slice 2; OQ4 no `abs_path` in `list_*` audit). **Amended 2026-05-23 to add `kind:"transformer"` to the slice-2 catalog scope** (single-file safetensors DiT weights from `checkpoints/` and `diffusion_models/`); `list_transformers` becomes new slice 2b; VAE / `text_encoder` catalog kinds and cascade `scaffolding_repo` modeling deferred. Ready for `/change-slice`.
**AI-Disclosure:** Claude (Opus 4.7, 1M context) authored; Grant to review and approve.

---

## Posture

> **Posture:** Boundary: integration (operator → comfyless catalog build; LLM-agent → comfyless catalog read). Risk factors: near security truth (catalog is THE keystone for all post-slice-3 reference resolution); broad impact (every subsequent MCP slice consumes catalog correctness); external exposure (LLM-agent drives the read surface; operator-supplied manifest crosses the spawn boundary).

## Slice

The additive slice that builds the catalog declared by ADR-015 and exposes its read-only discovery surface to the agent via `list_models` and `list_loras`. **Does not touch the shipped slice-1 `generate` handler** — the catalog is constructed and held on `_StartupConfig`, but `generate` continues to consume its existing path-based contract until new-slice 3 migrates it. The catalog being built-but-unused-by-generate is deliberate: it lets slice 2 land the entire catalog-construction trust boundary (the auditor's keystone) in isolation, with its own review surface, before the contract change to shipped code lands in slice 3.

Slice 2 also lands the spawn-time `--catalog <abs-path.json>` flag and the build-time invariants from ADR-015 §1 (no-follow-symlinks scan; NFC + case-insensitive-collision-rejection normalization; collision fail-closed; manifest may declare HF repo IDs with cache-miss fail-startup).

The catalog's **kind enum** is `{"model", "lora", "transformer"}` in slice 2 (see invariant 16). `kind:"transformer"` covers single-file safetensors diffusion-DiT weights — the `transformer_path` reference shape on `generate`. These are dormant in slice 2 (built, stored, audited, allowlist-checked) but not exposed by either slice-2 tool; **slice 2b (new follow-up slice)** adds `list_transformers` to surface them. This split lands the keystone in one slice (catalog supports every reference shape `generate` truly uses) without bloating slice 2's tool surface. `vae` / `text_encoder` kinds and cascade `scaffolding_repo` modeling are out of scope (see below) and are addressed in their own future slices if/when they become real use cases.

## Four signals

- **Who** — the **operator** at spawn time (configures `--model-base`, optionally `--catalog`, optionally hand-writes a manifest for a packaged distribution); the **LLM agent** at request time (calls `list_models`/`list_loras` to enumerate references it can name). Same-uid trust boundary on both (stdio MCP child process; manifest is operator-authored).
- **Data** — *read at spawn*: the `--model-base` directory tree (scan), optional manifest file, local HF cache (via `resolve_hf_path(..., allow_download=False)` for HF-source manifest entries); *held in memory*: the catalog dict `{normalized_name → {abs_path, kind, model_family?, source}}`; *returned at request time*: catalog entries **without** `abs_path` — only `name` + `kind` + `model_family` (when known) + `source`. **`abs_path` is server-side state and never crosses the MCP boundary.** The operator's manifest absolute path is also server-side only — `list_*` responses must not echo it.
- **Boundary** — `comfyless/mcp_server.py` (`_StartupConfig` extended with `catalog`; click `main()` gains `--catalog`; `_list_tools_impl` grows from 1 to 3 tools; `_call_tool_impl` gains 2 dispatch branches; 2 new handlers `_handle_list_models` / `_handle_list_loras`). Catalog-build logic preferred in a **new `comfyless/catalog.py` module** for review-boundary cleanliness (Open Question 1). Out of scope: anything in `_handle_generate` / `_handle_generate_cascade` (slice 3); `extract_params`/`iterate`/`edit` (slices 4–6); the uniform-error contract (slice 3 — no reference-resolution failures are reachable in slice 2 because the list tools enumerate, they do not resolve caller-supplied names); the catalog-as-sole-authority end-state (deferred amendment).
- **Failure** — fail-closed at startup on every malformed-or-ambiguous condition (no partial catalog ever runs); fail-closed at request time via slice-1's traceback-strip + sanitized-error pattern; audit line on every `list_*` invocation success or rejection.

## Risk level

**L3 (Red Zone).** ADR-015's `security-auditor` review explicitly called this slice's surface the "new security keystone" (HIGH-2 framing in the round-1 review). Every later slice's safety depends on the catalog's correctness — getting collision detection, symlink handling, or `abs_path` containment wrong here would silently undermine slice-3+ regardless of how carefully those slices are written. Runs `code-reviewer` (Opus) **and** `security-auditor` (Opus) before commit, model pinned at invocation; security-review output saved to `docs/security/review-slice-2-mcp-catalog-<YYYY-MM-DD>.md` and referenced in the commit body. ADR-015 §1 is the design source of truth.

## Intent

Build the spawn-time, in-memory `name → entry` catalog from a no-follow-symlinks scan of `--model-base` plus an optional operator manifest, with fail-closed startup on every ambiguity (collisions, escapes, HF cache miss, malformed manifest), and expose its public surface — names + kinds + families only, **never `abs_path`** — read-only via the `list_models` and `list_loras` MCP tools.

## Invariants (must always be true)

1. **Catalog built once at server spawn**, from (a) a scan of `--model-base` and (b) an optional `--catalog <abs-path.json>` operator manifest. The catalog is held on `_StartupConfig` and is read-only after spawn (no per-request rebuild, no hot reload — deferred polish per ADR-015 deferred list).
2. **Every entry's `abs_path` is `realpath`-resolved AND `_within(--model-base)`-checked at build time.** A manifest entry whose target escapes `--model-base` (after realpath, including symlink resolution) fails startup naming the entry. (ADR-015 §1.)
3. **Names are normalized at build time** by Unicode NFC + a case policy: names are compared **case-sensitively** for the lookup key, AND the catalog **rejects at startup any two entries whose names are equal under case-insensitive equality** (defeats case-folding host collisions; the §1 HIGH-2 fold-in). The same normalization function is used at all catalog-key callsites in this slice (build + the `list_*` responses' name field), so future slices (slice 3 request-side normalization) reuse the single canonical implementation.
4. **Scan does NOT follow symlinks.** A symlinked file or directory under `--model-base` does not mint an independent catalog entry. Only the underlying non-symlink scan hit's `realpath` is recorded. (ADR-015 §1 HIGH-2 fold-in.)
5. **Collision rule — fail closed at startup**, naming the ambiguous name, when *either*: (a) two distinct scan realpaths derive the same normalized name, OR (b) a manifest name collides under §3 normalization with a scan-derived name pointing at a **different** realpath. A manifest entry that aliases an additional name to a path, or assigns a name to a path the scan also found at the same realpath, is harmless and allowed; silent shadowing of a distinct scanned realpath is forbidden. (ADR-015 §1.)
6. **Manifest may declare HF-repo-ID-sourced entries.** Catalog build resolves them via `resolve_hf_path(..., allow_download=False)` to obtain the local `abs_path`. A build-time HF cache miss for a manifest entry **fails startup** naming both the manifest entry name and the repo ID. Operator-visible startup channel only — not the future agent-facing uniform-error contract. (ADR-015 §1 MEDIUM-4 fold-in.)
7. **Server startup fails closed** if `--catalog` is supplied and: the value does not resolve to an existing regular file; the file is not valid JSON; any entry is malformed (missing required fields, wrong types, non-string name/target).
8. **`_list_tools_impl` advertises exactly THREE tools**: `generate`, `list_models`, `list_loras`. (Updates slice-1 invariant 6's count from 1 to 3.) `generate`'s schema and description are **unchanged** in this slice.
9. **`list_models` / `list_loras` response entries contain ONLY**: `name`, `kind` (`"model"` for `list_models`; `"lora"` for `list_loras`), `source` (`"scan"` | `"manifest"`), and — for `list_models` only — `model_family` when known from scan-time class detection (e.g. `qwen-image`, `flux2`). For `list_loras`, an optional `target_family` field is surfaced **only** when a manifest entry explicitly declared one; scan-derived LoRA entries omit `target_family` entirely (no inference from filesystem layout or weight introspection in this slice). **`list_models` returns only catalog entries with `kind:"model"`; `list_loras` returns only `kind:"lora"`.** Catalog entries with `kind:"transformer"` sit in the catalog but are **not exposed by either slice-2 tool** — they remain dormant until slice 2b adds `list_transformers`. **No `abs_path`. No `path`. No filesystem-string under any other key.** No code path serializes a catalog entry's `abs_path` into an MCP response.
10. **Audit on every `list_*` invocation** — success and rejection alike, one line on stderr, never stdout. Audit carries tool name + status + elapsed; does NOT include `abs_path` from any catalog entry the call touched. (Carry-forward of slice-1 invariant 5.)
11. **Traceback strip carries forward**: any internal exception in catalog build (manifest parse error, realpath/IO error, HF resolver internal error) and in the `list_*` handlers is caught, full-traceback'd to stderr, and converted to a sanitized MCP error via the existing `_sanitize_error` / `_MCPHandlerError` pattern. No traceback, no `.py:line`, no `--model-base` absolute path, no operator manifest absolute path crosses the MCP boundary. (Carry-forward of slice-1 invariant 13.)
12. **stdout carries only MCP JSON-RPC frames.** (Carry-forward of slice-1 invariant 7.)
13. **No argparse / no CLI dispatch.** `--catalog` is added to the existing click `main()` command in `mcp_server.py` alongside `--output-dir` / `--model-base` / `--default-model` / `--mcp-max-iterations`. No `import argparse`; no call into `_run_cli_mode` / `_apply_overrides` / `_load_params_file`. (Carry-forward of slice-1 invariant 14.)
14. **Slice 2 does NOT modify the shipped `generate` handler.** `_handle_generate` and `_handle_generate_cascade` continue to consume their existing path-based contract verbatim; the catalog is built but not yet consumed by `generate`. The existing 158 `test_mcp_server.py` cases (N1–N33 from slice-1) and the other 8 suites continue to pass — 1008/1008 (or higher, with the new slice-2 cases added). This invariant is the line that lets slice 2 ship without re-firing slice-1's security review on the generate contract.
15. **No partial catalog ever runs.** Either the complete catalog (scan + manifest) builds cleanly and the server proceeds to serve MCP requests, OR catalog build raises and the server exits non-zero before binding stdio. There is no fallback to a "scan only" mode when the manifest fails, and no "skip the broken entry" silent drop.
16. **Catalog `kind` enum is `{"model", "lora", "transformer"}`** in slice 2. Each catalog entry has exactly one `kind`. `kind:"model"` covers diffusers-pipeline directories (those containing `model_index.json`) and HF-repo-ID-sourced entries; `kind:"lora"` covers LoRA adapter single-file safetensors; `kind:"transformer"` covers single-file safetensors diffusion DiT weights (the `transformer_path` reference shape). `vae` / `text_encoder` / `text_encoder_2` and cascade `scaffolding_repo` are **not** kinds in slice 2 — they are deferred (see Out of scope below).
17. **Scan classification dispatch rule** (the only filesystem-shape-to-`kind` mapping in slice 2; no key inspection of weights at scan time):
    - A directory containing `model_index.json` → mint a single entry with `kind:"model"`, name = directory basename, abs_path = realpath of the directory; **do not descend** into it.
    - For any other directory, descend (subject to invariant 4's no-follow-symlinks).
    - A non-symlink `.safetensors` file whose **immediate parent directory basename is `loras`** → `kind:"lora"`, name = filename stem, abs_path = realpath of the file.
    - A non-symlink `.safetensors` file whose immediate parent directory basename is `checkpoints` **or** `diffusion_models` → `kind:"transformer"`, name = filename stem, abs_path = realpath of the file. (Anchored to this codebase's actual ComfyUI directory layout, per the 2026-05-23 amendment.)
    - Any other filesystem entry encountered during scan is **skipped**; if the operator wants it cataloged, they declare it via manifest with an explicit `kind`. The parent-directory dispatch is byte-exact on the directory basename (no NFC / case folding on the directory name — operators following the convention name dirs as `loras` / `checkpoints` / `diffusion_models` exactly).

## Failure semantics

- **Fail-closed at startup** on every condition in invariants 2, 5, 6, 7, and the case-insensitive-collision part of 3. Server exits non-zero before serving a single MCP request. Startup-failure messages on stderr name the offending field (entry name, repo ID, conflicting names) for operator debugging; this is the operator channel, distinct from the future agent-facing uniform-error contract (slice 3).
- **Fail-closed at request time** in the `list_*` handlers: any internal exception → sanitized MCP error + audit line; no partial result returned.
- **No partial success** at any catalog-build stage (invariant 15).
- **Audit-line write failure** does not block the response (mirrors slice-1; increments the existing `_audit_write_failures` counter).

## Out of scope (explicit)

- Migrating `generate` (and `_handle_generate_cascade`) to consume the catalog — **new-slice 3**.
- The uniform agent-facing error class for reference-resolution failures (ADR-015 §2 step 2 commitment) — lands with slice 3 when reference-resolution failures become reachable through `generate`. Slice 2's `list_*` tools do **not** resolve caller-supplied names; they enumerate, so no reference-resolution failure mode is reachable here.
- **`list_transformers` MCP tool** — slice 2's catalog stores `kind:"transformer"` entries (per invariants 16 + 17) but **does not expose them**. `list_transformers` is **new slice 2b**, sitting between slice 2 and slice 3.
- **`vae` / `text_encoder` / `text_encoder_2` catalog kinds.** These overrides exist in the slice-1 `generate` schema (`vae_path`, `text_encoder_path`, `text_encoder_2_path`) but are rare-use power-user fields whose end-to-end wiring isn't actively exercised in this codebase. Deferred to a future slice if a real use case emerges. Slice 3's `generate` migration will either keep these three fields as raw-path inputs with an explicit "not catalog-resolved" carve-out, or drop them from the MCP schema if they prove dead code — a slice-3 decision.
- **Cascade `scaffolding_repo` modeling.** The cascade fields `stage_c` / `stage_b` / `stage_a` map cleanly to `kind:"model"` (they are standalone model checkpoints; the slice-1 cascade handler already `_within(--model-base)`-validates them). `scaffolding_repo` is an HF repo (directory-shaped) that loads text encoder + tokenizer + VAE as a bundle — it doesn't fit cleanly under `kind:"model"` and may warrant a sub-tag. Deferred to slice 3 cascade-migration design; slice 2 scans cascade stages as `kind:"model"` only if they happen to live under a `model_index.json` directory (otherwise the operator declares them via manifest).
- `extract_params` (slice 4), `iterate` (slice 5), `edit` stub (slice 6).
- Catalog-as-sole-authority end-state and the CLI-path migration (ADR-015 §4 deferred).
- Hot-reload of the catalog without server restart (ADR-015 deferred polish).
- LoRA target-family **inference** from filesystem layout or weight introspection — `list_loras` carries a `target_family` hint only if it comes from the manifest (Open Question 2). Inference is a future slice.
- Civitai / external metadata layered on the catalog (Backlog Queued; future).
- Streaming progress notifications, MCP `resources/list` / `prompts/list` surfaces.

## Negative cases (required)

**Startup fail-closed cases:**

- **N1** — `--catalog /path/not/exist.json` → server exits non-zero.
- **N2** — `--catalog` pointing at a directory (not a file) → exits non-zero.
- **N3** — `--catalog` pointing at malformed JSON → exits non-zero.
- **N4** — Manifest entry missing required fields (e.g. no `target`) → exits non-zero, message names the malformed entry.
- **N5** — Manifest entry whose target absolute path escapes `--model-base` after `realpath` → exits non-zero naming the entry.
- **N6** — Manifest entry whose target is a symlink resolving outside `--model-base` → exits non-zero (proves realpath happens before the `_within` check at build time, mirroring the slice-1 `extract_params` realpath-first ordering).
- **N7** — Manifest entry naming an HF repo ID that is **not** present in the local HF cache → exits non-zero naming both the entry name and the repo ID (no network call attempted).
- **N8** — Two scanned weights under `--model-base` whose basenames normalize to the same name → exits non-zero naming the ambiguous name.
- **N9** — Manifest assigning a name that collides (under §3 normalization) with a scan-derived name pointing at a **different** realpath → exits non-zero naming the conflict.
- **N10** — Two entry names that case-insensitively collide (e.g. `Foo` and `foo` in the manifest, or one in scan and one in manifest at different realpaths) → exits non-zero (proves the case-insensitive-collision-rejection rule at startup).
- **N11** — Symlink under `--model-base` does NOT mint an independent catalog entry: fixture with `link.safetensors → real.safetensors`; assert the catalog contains exactly one entry, recorded under `real.safetensors`'s realpath, not two entries (proves no-follow-symlinks).

**Spawn-succeeds-cleanly cases:**

- **N12** — Manifest that aliases a scan-derived name to the **same realpath** (harmless alias): spawn succeeds, the catalog has the entry, no startup failure (proves §5's "same realpath alias allowed" carveout).
- **N13** — Empty `--model-base` (no scannable weights) + no `--catalog` → spawn succeeds with an empty catalog; `list_models` / `list_loras` return empty lists.
- **N14** — Spawn without `--catalog` (omitted entirely) → spawn succeeds; catalog is just the scan.
- **N15** — Manifest with no entries (empty object) → spawn succeeds; catalog is just the scan.

**No-abs_path-leak / response-shape cases:**

- **N16** — `list_models` MCP call returns a list of entries; assert every entry's keys are a subset of `{name, kind, model_family, source}`; assert NO `abs_path` / `path` / any absolute-filesystem-string under any other key.
- **N17** — `list_loras` MCP call returns `{name, kind:"lora", source, target_family?}` entries; assert no `abs_path` leak (and no inferred `target_family` if Open Question 2 is resolved manifest-only).
- **N18** — Construct a fixture catalog with at least one HF-source entry and at least one local-scan entry; assert both surface only their **names** (not the repo ID, not the local path) on the MCP response.
- **N19** — `list_models` audit line on stderr does NOT contain any catalog entry's `abs_path` or repo ID; carries only tool name + count + status + elapsed (proves invariant 10's audit-redaction extension).

**Tool surface cases:**

- **N20** — `tools/list` MCP call advertises exactly three tools: `generate`, `list_models`, `list_loras`. `generate`'s `inputSchema` and `description` are byte-identical to slice 1 (proves invariant 14's no-touch).
- **N21** — `generate`'s 158 existing tests in `test_mcp_server.py` continue to pass with the slice-2 changes applied (catalog built but unused by generate); the other 8 suites also continue at 1008/1008. (Proves slice 1 didn't regress.)

**Traceback strip / no-argparse cases:**

- **N22** — Force an internal exception in catalog build (e.g. monkey-patch the manifest reader to raise) — assert the server exits non-zero at startup (build failures kill startup, they don't propagate to MCP frames; the request-time traceback-strip path is exercised separately). Then force an internal exception in `_handle_list_models` (monkey-patch the catalog-iteration step) — assert the MCP error frame to the client contains no `Traceback`, no `.py:<digits>`, no absolute paths starting with `/home/`, `/root/`, `--model-base`, or the manifest path; full traceback present on stderr.
- **N23** — Static source check on `comfyless/mcp_server.py` and `comfyless/catalog.py` (Open Question 1): no `import argparse`; `--catalog` is declared as a click option on `main()` not via argparse.

**Transformer-kind classification cases (2026-05-23 amendment, invariants 16 + 17):**

- **N24** — A `.safetensors` file in `<model-base>/checkpoints/foo.safetensors` is scanned and mints a catalog entry with `kind:"transformer"`, name `"foo"`. Same for `<model-base>/diffusion_models/bar.safetensors` → `kind:"transformer"`, name `"bar"`. (Proves the dispatch rule covers both conventional dirs.)
- **N25** — A `.safetensors` file outside `loras/`, `checkpoints/`, or `diffusion_models/` (e.g. `<model-base>/random_dir/orphan.safetensors` or `<model-base>/orphan.safetensors` at the root) is **skipped** by the scan — no catalog entry minted. (Proves the "manifest-required for unconventional locations" rule.)
- **N26** — A manifest declares `{"my_transformer": {"target": "/abs/path/outside/conventional/dir.safetensors", "kind": "transformer"}}` (target inside `--model-base` but not under `loras/`/`checkpoints/`/`diffusion_models/`) → mints `kind:"transformer"` entry. (Proves manifest override for unconventional locations.)
- **N27** — `kind:"transformer"` catalog entries are present after spawn but do **not** appear in `list_models` or `list_loras` responses. Set up a fixture with at least one of each of the three kinds (model, lora, transformer); assert `list_models` returns the model entries only, `list_loras` returns the lora entries only, neither returns the transformer entry. (Proves invariant 9's per-tool kind filtering and the dormant-`transformer`-until-slice-2b deferral.)
- **N28** — Symlink at `<model-base>/checkpoints/link.safetensors → ../random/real.safetensors`: the link is NOT minted (no-follow-symlinks per invariant 4); the target `real.safetensors` is in `random/` so it is also skipped (per N25). Net catalog effect: zero entries. (Cross-validates invariant 4 + invariant 17's parent-directory dispatch on a symlinked path.)

## Proof hooks

- **Positive:** `./.venv/bin/python3 test_mcp_server.py` — new sections cover catalog build, `list_models`, `list_loras`, response shape, no-abs_path leak, audit line.
- **Negatives N1–N23** organized as sections inside `test_mcp_server.py` (no pytest dep; same `python3 test_<name>.py` invocation as the other suites; run via `./.venv/bin/python3` per ADR-013).
- **Static-source check (N23)** via `inspect.getsource` / `ast.parse` on `comfyless.mcp_server` and `comfyless.catalog` (if separate module).
- **Existing 9 suites continue to pass — 1008/1008** (the `test_mcp_server.py` count grows; CLAUDE.md's suite-count line is updated in the closure step). Proves no slice-1 behavior regressed.

## Red Zone ownership

- **Catalog build sequence** (scan-then-manifest; no-follow-symlinks; NFC + case-insensitive-collision-rejection normalization applied at build; per-entry build-time realpath + `_within`): owned by **Grant** — AI-generated only.
- **Fail-closed collision rules** (scan-internal + scan-vs-manifest distinct-realpath shadowing; case-insensitive collision): owned by **Grant**.
- **Manifest HF-source build-time policy** (cache miss fails startup naming entry + repo): owned by **Grant**.
- **The `abs_path`-never-crosses-MCP-boundary guarantee** in `list_models` / `list_loras`: owned by **Grant** — signs off that no response serializer leaks `abs_path` under any key, including the audit line.
- **ADR-015 is the design source of truth.** Any divergence reverts to an ADR amendment before code lands.

## Open questions — RESOLVED 2026-05-23

1. **Catalog module location → resolved: new `comfyless/catalog.py` module.** Build logic + name normalization + lookup helpers live in a dedicated module so `mcp_server.py` doesn't sprawl past 1004 lines and the security keystone has its own clean review boundary. Slice 3 will see a smaller diff against `mcp_server.py` as a side effect.
2. **`list_loras` `target_family` field → resolved: manifest-only.** Scan-derived `list_loras` entries surface only `{name, kind, source}`. The manifest schema includes an optional `target_family` field (string, e.g. `"qwen-image"`); when present, `list_loras` propagates it on the corresponding response entry. No inference from filesystem layout or weight introspection in this slice. Invariant 9 amended below to reflect this; the response-shape negative case (N17) asserts no inferred `target_family` on scan-derived entries.
3. **Loadable-shape detection → resolved: extract in slice 2.** Slice 2 includes the helper extraction from the existing `_load_pipeline` (in `comfyless/generate.py`) and LoRA loader (`nodes/eric_qwen_image_lora.py` / `nodes/eric_qwen_edit_lora.py`) into the new `comfyless/catalog.py`. No new shape heuristics. The extraction is part of slice 2's diff — the existing loader call sites continue to work via the new shared helper. Single combined review covers both the extraction (refactor-shaped) and the new catalog behavior.
4. **`abs_path` redaction in `list_*` audit → resolved: no `abs_path` (and no manifest absolute path) in the audit line.** Tighter than slice-1's `generate` audit, intentional — the catalog's whole purpose is `abs_path` containment, so the audit line carries only tool name + entry count + status + elapsed. Invariant 10 already commits to this; this resolution confirms it as Grant-owned Red Zone choice.

## Pointers

- ADR: [ADR-015](../decisions/ADR-015-mcp-catalog-reference-resolution.md) (Status: accepted; rounds 1+2 CLEAN).
- Parent ADR: [ADR-011](../decisions/ADR-011-comfyless-mcp-server.md) (2026-05-23 Changelog — slice plan reordered).
- Security review of ADR-015 design: [review-adr-015-catalog-reference-2026-05-22.md](../security/review-adr-015-catalog-reference-2026-05-22.md).
- Slice-2 implementation review target: `docs/security/review-slice-2-mcp-catalog-<YYYY-MM-DD>.md` (saved at slice close).
- Predecessor slice-1 Vision (substrate + carry-forward invariants 5/7/13/14): [slice-1-mcp-generate.md](slice-1-mcp-generate.md).
- INFO-4 operator-guidance note (case-folding host portability): incorporate into `comfyless/README.md` (or equivalent operator-facing doc) at slice closure.
