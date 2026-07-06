# ADR-018: Multi-root catalog scan — separate `--lora-path` / `--transformer-path` roots with recursive, kind-typed scanning

**Date:** 2026-06-27
**Status:** accepted (2026-07-05 — `code-reviewer` APPROVED + `security-auditor` CLEAN, both Opus; review saved to `docs/security/review-adr018-multi-root-scan-2026-07-05.md`)
**AI-Disclosure:** Claude (Opus 4.8, 1M context) authored; Grant reviewed.
**Relates to / extends:** [ADR-015](ADR-015-mcp-catalog-reference-resolution.md) (catalog reference resolution — single `--model-base` recursive scan + `--catalog` manifest; basename-strip + `_within` containment). ADR-011 (MCP surface). Orthogonal to ADR-017.

---

## Context

The comfyless catalog is built from a **single** `--model-base` recursive scan (+ optional `--catalog` manifest). Models, LoRAs, and transformers must all live under that one tree; LoRAs/transformers are found by the *immediate-parent-dir* convention (`loras/`→lora, `checkpoints/`+`diffusion_models/`→transformer, `comfyless/catalog.py` `_SCAN_DIR_TO_KIND`). Two problems surfaced operating it (2026-06-27):

1. **The model store and the LoRA/transformer store are different trees.** Clean diffusers models live in `.../ai-base/models/hf-local`; standalone LoRAs/transformers live in `.../ai-base/models/comfyui/models/{loras,checkpoints,diffusion_models}`; and `.../ai-base/models/{hf-cache,invokeai}` hold app caches/stores whose models are UUID/hash-named junk (duplicates or unused). Pointing `--model-base` at the common parent `.../models` ingests the junk; pointing it at `hf-local` drops the LoRAs/transformers. No single tree captures models + LoRAs + transformers while excluding the junk.
2. **The convention scanner is not recursive into nested layouts.** Real LoRAs are organized `loras/<base-model>/<type>/file.safetensors` (type = concept/action/character/…); transformers `checkpoints/<base-model>/…` or `diffusion_models/<base-model>/…`. The current `_SCAN_DIR_TO_KIND` parent-name match only mints a file whose *immediate* parent is `loras`/`checkpoints`/`diffusion_models`, so anything one level deeper is silently skipped.

Grant's resolution: an **include** model, not a configurable exclude — point the scanner at exactly the trees we want, with the kind of each extra tree fixed by which flag named it, and scan each recursively.

This change widens the **load-boundary path allowlist** (`_check_paths` in `comfyless/server.py`): today every resolved weight path (model / transformer / vae / each lora) must be `_within(--model-base)`. LoRAs/transformers under `comfyui/` are outside `hf-local`, so they would be scanned-but-rejected at generate time unless the allowlist also accepts the new roots. That allowlist edit is the load-bearing, security-reviewed part. **Trust class is unchanged:** the new roots are spawn-time, operator-supplied directories exactly like `--model-base`; the agent still supplies only catalog *names* (basename-stripped, `_within`-checked), never paths. This widens *which operator-curated trees* can be loaded from — not *who* chooses them.

## Decision

### 1. Two new spawn-time scan roots (operator-supplied, like `--model-base`)

- **`--lora-path DIR`** — a directory tree scanned **recursively**; **every** `.safetensors` under it (any depth) is minted as `kind:"lora"`. Repeatable (`--lora-path A --lora-path B`) — LoRAs may live in more than one tree.
- **`--transformer-path DIR`** — same, minting `kind:"transformer"`. Repeatable. Point it at the *specific* transformer trees — `--transformer-path …/comfyui/models/checkpoints --transformer-path …/comfyui/models/diffusion_models` — **not** at their common parent `comfyui/models`: that parent also contains `loras/`, `vae/`, `text_encoders/`, etc., so a parent-level root would mint every LoRA a second time as `kind:"transformer"` (a cross-kind overlap, which fails the build closed per §2) and ingest non-transformer weights. The kind is fixed by the flag, not the subdir name — which is exactly why the flag must name only trees that are wholly of that kind.

These are **kind-typed roots**: kind is determined by *which flag* declared the tree, NOT by `_SCAN_DIR_TO_KIND` parent-name matching (which stays as-is for the `--model-base` tree only). This is what makes nested `<model>/<type>/file` layouts work. `--model-base` retains its existing model_index.json→model + convention-dir behavior, unchanged.

### 2. Naming (resolves the bare-name constraint)

A catalog name cannot contain `/` (the resolver basename-strips path-shaped input — ADR-015 §2 step 1). So a recursively-scanned weight is keyed by its **filename stem** (`foo.safetensors` → `foo`), via the existing `normalize_name` + `_add_entry` gate.

**Collisions fail closed — with one carve-out for byte-identical duplicates.** Two weights anywhere under the kind's roots sharing a stem (different `abs_path`) raise `CatalogBuildError` at spawn, naming both — the existing no-partial-catalog invariant (ADR-015 / Vision invariant 5/15) — **unless the two files are byte-identical** (same size AND same SHA-256), in which case the first-scanned entry is retained and the later path is treated as a harmless alias ("pick one"). Rationale for the carve-out: the same accel-LoRA is routinely copied under several `<base-model>/` folders; failing spawn on a duplicate of *identical content* punishes a layout convention, not an ambiguity — whichever path is served, the agent gets the same bytes. Only a **genuine** clash (distinct content, or a model *directory* — directories are never content-equal) fails closed, so the agent can never be silently served the wrong weight. The equality check is size-gate-then-full-SHA-256 at spawn; the hash cost is accepted (collisions are rare). The server does not start with an ambiguous catalog; the operator disambiguates genuine clashes by renaming. **A path-flattened naming scheme (e.g. `<model>__<type>__<stem>`) is the documented fallback** if real trees collide often enough that renaming is impractical (see Deferred); it is not v1 because it makes every LoRA reference verbose.

**Cross-kind overlap fails closed.** If the same *name* is minted under two different kinds — whether the same `abs_path` (overlapping `--lora-path`/`--transformer-path` trees) or different paths (even byte-identical content) — that is a kind ambiguity: first-in-wins would make the entry's kind depend on scan order, and a legitimate request for the later kind would uniformly fail. `CatalogBuildError` at spawn, naming both kinds. The byte-identical "pick one" carve-out applies only *within* a kind. Same-path/same-kind re-mints (one tree named twice, or nested roots of the same kind) remain harmless aliases. *(Different-path case tightened per code-review F-1, 2026-07-05.)*

**Root validation fails closed.** Each `--lora-path` / `--transformer-path` is realpath-resolved and must be an existing directory at spawn; a missing or non-directory root raises `CatalogBuildError` (it must NOT silently scan-as-empty — `os.walk` on a nonexistent path yields nothing, which would fail *open* into a partial catalog).

### 3. Load-boundary allowlist widening

`_check_paths(req, roots)` (today `_within(model_base)` for model/transformer/vae/loras) accepts a path that is `_within` **any** of `{model_base} ∪ lora_paths ∪ transformer_paths`. The request-side resolver's containment (`resolve_reference`) is likewise evaluated against the union. Each root is independently realpath-resolved and validated at spawn (existing dir, NUL-byte pre-check) exactly as `--model-base` is. No root need be under another.

### 4. Scope of edit

- `comfyless/catalog.py` — `build_catalog(model_base, *, lora_paths, transformer_paths, catalog_path)`; a kind-typed recursive scan helper (reuses `_add_entry` for collision-safe merge; skips file symlinks + does not follow dir symlinks, per existing invariant 4).
- `comfyless/server.py` — `_check_paths` takes the root set; `_within`-union.
- `comfyless/mcp_server.py` — `_validate_startup_args` parses/validates `--lora-path`/`--transformer-path` (repeatable), threads the root set into catalog build, resolver containment, and `_check_paths`.
- `start-mcpo.sh` — pass `--model-base hf-local --lora-path comfyui/models/loras --transformer-path comfyui/models/checkpoints --transformer-path comfyui/models/diffusion_models` by default (per §1: never the `comfyui/models` parent).
- Tests — `test_mcp_server.py` / `test_machine_boundary_validator.py`: multi-root scan, nested-depth minting, stem-collision fail-closed, load-boundary acceptance of in-root paths + rejection of out-of-all-roots paths.

### 5. Invariants (enforced by the slice)

- Recursively-scanned weights are kind-typed by their declaring root; nesting depth is unbounded.
- Catalog names remain bare (no `/`); stem collisions across a kind's roots fail the build closed (no silent drop, no partial catalog).
- A weight loads iff its resolved `abs_path` is `_within` the union of all roots; nothing outside all roots is loadable (the agent still cannot supply a path).
- `--model-base` behavior is unchanged (models via model_index.json; the legacy convention-dir scan still applies to that tree only).
- The daemon (`server.py`) `_check_paths` and the MCP path share the widened allowlist (no divergence).

## Alternatives Rejected

- **Configurable exclude (`--scan-exclude hf-cache,invokeai`).** Grant's explicit call: prefer an allowlist (include) over a denylist — a denylist silently ingests any *new* junk dir added under the parent later. Include fails safe (a new tree appears only when named).
- **Single extra "aux" scan root using the existing parent-name convention.** Doesn't handle nested `<model>/<type>/file` (the parent is `character/`, not `loras/`), which is the actual layout.
- **Filesystem reorg (move/symlink LoRAs into `hf-local`).** Dir symlinks aren't followed (`followlinks=False`), and moving breaks the ComfyUI install that owns those trees.
- **Manifest enumeration** of every LoRA/transformer. Defeated by scale + nesting and runs against the "separate LoRA-catalog project will own this" direction (TECH_DEBT 2026-06-27).
- **Path-flattened names by default.** Verbose agent references for the common (no-collision) case; kept as the fallback.

## Deferred / Out of Scope

- **Path-flattened / disambiguated names** if stem collisions prove common — the fallback to §2.
- **Family metadata on LoRAs/transformers** (the "list LoRAs in family X" feature) — still deferred to the separate LoRA-catalog project (TECH_DEBT 2026-06-27).
- **CLI `comfyless.generate` multi-root** — this slice wires the MCP server (+ shared `server.py`/`catalog.py`); the standalone CLI keeps single `--model-base` until needed.
- **CACHEDIR.TAG auto-skip** — unnecessary under an include model.

## Changelog

- **2026-06-27 (proposed):** Authored after the `hf-local` model-base narrowing (ADR-017 docs-closure era) dropped LoRAs/transformers (they live under `comfyui/`, outside `hf-local`), and the nested `<model>/<type>/file` layout was confirmed to defeat the parent-name convention scanner. Decision: include-model multi-root scan with kind-typed recursive `--lora-path`/`--transformer-path` roots + a widened load-boundary allowlist. Implementation per §4, gated by `code-reviewer` + `security-auditor` (Opus); security review saved under `docs/security/`.

- **2026-07-05 (implementation refinements folded into §1/§2):** Three cases the implementation surfaced, resolved before code lands:
  1. **Byte-identical duplicate carve-out** ("pick one"): same stem + different paths + identical size/SHA-256 → harmless alias, first-scanned retained. Genuine content clashes still `CatalogBuildError`. (Duplicated accel-LoRAs under multiple `<base-model>/` folders are a layout convention, not an ambiguity.)
  2. **Cross-kind overlap fails closed:** same `abs_path` minted under two kinds (overlapping roots) → `CatalogBuildError`, since first-in-wins would make kind scan-order-dependent.
  3. **Root validation fails closed:** each extra root must realpath-resolve to an existing directory at spawn; `os.walk` on a missing path yields-nothing, which would otherwise fail *open* into a partial catalog.
  Also corrected §1's `--transformer-path` guidance: name the specific `checkpoints/` + `diffusion_models/` trees, never their `comfyui/models` parent (contains `loras/` → cross-kind overlap). This ADR is consumed by the LoRA-catalog service (Vision/ADR to follow) as its scan substrate.

- **2026-07-05 (accepted; implementation shipped):** Full §4 scope implemented — `catalog.py` (`_scan_kind_root`, pick-one aliasing, cross-kind + root-validation fail-closed, `resolve_reference` roots-union), `server.py` (`_check_paths` union, `run_server` extra roots), `mcp_server.py` (`--lora-path`/`--transformer-path` → `_StartupConfig.all_roots`, six call sites on the union; `--default-model` deliberately stays model_base-only per invariant 10), `start-mcpo.sh` defaults. 40 new tests (test_mcp_server 573, test_server_robustness 105); full regression 1971 green. Reviews: `code-reviewer` APPROVED (1 LOW — cross-kind guard widened to different-path collisions, folded into §2), `security-auditor` CLEAN (F-1 LOW NUL pre-check folded; F-2 INFO adopted as the same §2 tightening; F-3 INFO confirmed harmless). Review record: `docs/security/review-adr018-multi-root-scan-2026-07-05.md`. Status → accepted. CLI `comfyless.generate` multi-root remains deferred (run_server params wired but unexposed).

AI-Disclosure: Claude (Opus 4.8, 1M context) authored; reviewed by Grant.
