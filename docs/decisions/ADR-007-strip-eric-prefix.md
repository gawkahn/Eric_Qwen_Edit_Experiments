# ADR-007: Strip `Eric_` Prefix from Module, File, and Node Class Names

**Date:** 2026-04-23
**Status:** proposed (execution deferred — see "Timing" below)

---

## Context

The repository began as a fork of Eric Hiss's `Eric_Qwen_Edit_Experiments` ComfyUI
node pack. The original codebase used an `Eric_` / `eric_` / `EricXxx` prefix
across files, class names, `NODE_CLASS_MAPPINGS` keys, and display names as a
namespace / attribution convention — standard for custom-node packs in the
ComfyUI ecosystem.

The fork has since diverged substantially from the upstream:

- Added `comfyless/` — a full CLI + Unix socket daemon (`generate.py`, `server.py`,
  `--json` bridge) that is not a ComfyUI node at all.
- Added the generic `GEN_PIPELINE` loader/generate nodes with auto-detection for
  Flux.1, Flux.2, Flux.2-Klein, Chroma, Qwen-Image, SDXL, SD3.5, Pony, Illustrious,
  Z-Image, and AuraFlow.
- Added Flux.2-Klein edit node, LoRA-format conversion infrastructure (Flux /
  Chroma / Flux.2), HuggingFace repo-id resolution (`resolve_hf_path`),
  Spectrum acceleration integration, and the Wan2.1 2× upscale VAE.
- Governance scaffolding: ADRs, §12 security reviews, TECH_DEBT register,
  AI-disclosure trailer hook, CC BY-NC 4.0 inherited license, pre-commit hook.

At this point the `Eric_` prefix is noise rather than namespace. The project
needs its own identity, and ComfyUI users shouldn't see "Eric Diffusion Loader"
in their node menu for what is now substantially a different tool. Attribution
to the original author belongs in the license header and a `CHANGES.md`, not in
every file name and node label.

Waiting longer raises the migration cost: each additional node, each additional
caller, each additional embedded workflow adds work to the rename.

## Decision

### Scope — in

Strip `Eric_` / `Eric ` / `eric_` prefix (but not the `Qwen` or `Diffusion`
qualifiers that follow it) from:

1. **File names** in `nodes/` — 46 files, e.g. `nodes/eric_diffusion_loader.py`
   → `nodes/diffusion_loader.py`. Use `git mv` so history is preserved.
2. **Class definitions** — 44 `class Eric*` → `class *`, e.g. `EricDiffusionLoader`
   → `DiffusionLoader`.
3. **`NODE_CLASS_MAPPINGS` keys** in `nodes/__init__.py` — e.g.
   `"Eric Diffusion Loader"` → `"Diffusion Loader"`. Note: this breaks
   workflow JSON re-import (see Migration).
4. **`NODE_DISPLAY_NAME_MAPPINGS` values** — same treatment as mapping keys.
5. **Imports in `comfyless/`** — 4 `from nodes.eric_...` → `from nodes....` lines.
6. **`nodes/__init__.py` imports** — match renamed file names.
7. **`pyproject.toml`** — `name = "comfyui-eric-qwen-edit"` →
   `name = "comfyui-qwen-edit"`. The package has never been published to PyPI,
   so this costs nothing.
8. **Top-level `__init__.py` docstring title** — "Eric Qwen-Edit & Qwen-Image
   Nodes" → "Qwen-Edit & Qwen-Image Nodes" (cosmetic; attribution block in
   the same file stays per "Scope — keep" below).
9. **`CLAUDE.md`, `README.md`, and other in-repo docs** — references to old
   module names and node class names update to match.

### Scope — keep (attribution preservation under CC BY-NC 4.0)

10. **`LICENSE.txt`** — unchanged.
11. **Top-level `__init__.py` copyright / attribution block** — modified to the
    heavily-modified-fork pattern:

    ```python
    # Copyright (c) 2026 Eric Hiss — original work.
    # Copyright (c) 2026 Grant Kahn — modifications.
    # Licensed under CC BY-NC 4.0 (see LICENSE.txt).
    # Original source: https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
    # This version contains substantial modifications; see CHANGES.md.
    ```

    Rationale: CC BY 4.0 (the "BY" half of NC) requires (a) credit the creator,
    (b) link to the license, (c) **indicate that changes were made**. Both
    copyright lines are legally accurate — each party holds copyright in their
    own contributions. Dropping "Commercial dual license" from Eric's original
    wording is deliberate: that was Eric's own offer on his code, not ours to
    extend to the combined work.

12. **New file: `CHANGES.md`** — list the material divergences from upstream
    (comfyless CLI + daemon, generic GEN_PIPELINE loader, Flux/Chroma/Flux2
    families, LoRA-format conversion, `resolve_hf_path`, §12 security reviews,
    ADR/TECH_DEBT governance). Satisfies the "indicate that changes were made"
    clause cleanly.

### Scope — deferred

13. **GitHub remote rename (`Eric_Qwen_Edit_Experiments` → TBD).** Hold until
    every internal change is merged, tested, and stable for a few sessions. If
    the rename goes sideways we may revert the internal changes before the
    remote name is touched, which is much cleaner when `origin/main` is still
    under the old URL. GitHub's auto-redirect covers the gap for anyone who
    cloned during the transition.
14. **Comfyless → independent package.** Split `comfyless/` into its own
    deployable package (PyPI wheel or vendored directory) so it no longer
    requires running inside the repo path. Separate ADR when that lands;
    mentioned here only because it affects the target name.
15. **Workflow JSON files committed to the repo (`workflows/*.json`, embedded
    PNG workflows).** Don't rewrite in bulk — they'll be regenerated naturally
    during testing (see Migration).

## Alternatives Rejected

**Keep the `Eric_` prefix indefinitely.** The longer we wait, the larger the
rename. And the namespace justification has dissolved: most of the current code
isn't a derivative of Eric's work at file level.

**Rename files only, keep `NODE_CLASS_MAPPINGS` keys stable.** Would preserve
every existing workflow. Considered; rejected because it leaves half-renamed
output (ComfyUI node menu still shows "Eric ..."), and workflow JSON is
trivially monkey-patchable at the `type`/`title`/`Node name for S&R` fields.
Workflow PNGs the user actually needs to keep are in the hundreds, not
thousands, and most serve as recovery artifacts that get replaced on the next
run anyway.

**Rename incrementally over time.** Considered; rejected because
`nodes/__init__.py` couples all node registrations to one file, and `comfyless/`
imports specific `nodes.eric_*` modules. Any partial rename leaves the repo in
a broken state until the next commit lands. Full rename on a branch is cleaner.

**Do the rename in-place on `main` as a commit series.** Each commit is correct
in isolation (files renamed together with their imports), but no intermediate
commit on `main` works end-to-end in ComfyUI until the entire series lands.
Branch + test + merge fixes this.

**Squash the rename commits on merge.** A 46-file rename + 44 class renames +
imports + mappings + docs as one mega-diff is unreviewable. Preserve the commit
series so the merge reads as a sequence of small, obviously-correct changes.

## Execution Plan

When the time comes (see "Timing" below), execute in this order:

### Branch + worktree isolation

1. Spawn a background agent with `isolation: "worktree"` (the Agent tool
   parameter creates a separate git worktree on a new branch — the main
   working copy and the agent's copy do not collide). Branch name:
   `rename/strip-eric-prefix`.
2. Agent runs in the background (`run_in_background: true`) so parallel work
   can continue in the main working copy.
3. When the agent reports the branch is ready, pull it locally, run manual
   validation (below), and merge with `--no-ff` to preserve the commit series.

### Commit series on the branch

Each logical step is its own commit so the merge reads as a reviewable sequence:

1. **File renames** — `git mv nodes/eric_*.py nodes/*.py` for all 46 files.
   Leaves imports broken; commit anyway (history preservation requires the
   rename commit be standalone for git's rename-detection heuristic).
2. **Class renames + per-file imports** — global rename of `EricDiffusion*` /
   `EricQwen*` / `EricLora*` class names and fix `from nodes.xxx import` lines
   in each renamed file.
3. **`nodes/__init__.py` mapping update** — rewire imports to new module names;
   rewrite `NODE_CLASS_MAPPINGS` keys and `NODE_DISPLAY_NAME_MAPPINGS` values.
4. **`comfyless/` import fixes** — 4 lines in `server.py` and `generate.py`.
5. **`pyproject.toml` package name** — `name = "comfyui-qwen-edit"`.
6. **Attribution + `CHANGES.md`** — top-level `__init__.py` block rewrite +
   new `CHANGES.md` at repo root.
7. **Docs sweep** — `CLAUDE.md`, `README.md`, and any `docs/` references to
   old names. This is last because it's the most volatile.

Each commit must include the AI-disclosure trailer and pass the pre-commit
hook. Commit messages use the existing `refactor:` prefix (per CLAUDE.md repo
convention).

### Migration — workflow JSON

For user-saved workflow JSON files (the two the user has — one component-load,
one edit-node), produce a mechanical monkey-patch script that rewrites:

- `nodes[].type` — old class-mapping key → new key.
- `nodes[].properties["Node name for S&R"]` — same.
- `nodes[].title` — same, if it matches the old name.

Input: list of `old_key → new_key` pairs (derivable from the rename map) and
the workflow JSON files. Output: patched copies next to the originals; user
verifies by loading in ComfyUI.

No automatic rewrite of workflow JSON files inside the repo; those will be
regenerated naturally during post-rename testing.

### Validation (manual — there is no CI)

Before merging, on the branch:

1. `python -m py_compile nodes/*.py` — syntax sanity.
2. Run the three test suites:
   - `python3 test_manual_loop.py`  (186 tests)
   - `python3 test_multistage.py`   (141 tests)
   - `python3 test_samplers.py`     (41 tests)
   - Expect 368 tests, 0 failures.
3. Start ComfyUI with the branch checked out. Verify:
   - Console shows all nodes registered with new names (no import errors).
   - Node menu displays new names under the "Diffusion" / "Qwen" categories.
   - A patched workflow JSON loads cleanly.
   - A representative generate run completes end-to-end (one edit, one text-to-image).
4. Run comfyless CLI — `python -m comfyless.generate --help` and one real
   generate with a known-good prompt, confirm the PNG and sidecar write
   correctly.
5. Test `--params` replay on a freshly-saved PNG.

Only after all five validation steps succeed, merge into `main` with
`--no-ff`, push, then execute the GitHub remote rename (deferred — see above).

## Parallel Work During the Rename

When the agent is running the rename on its own branch, the main working copy
stays free for other work. Safe parallel work (no conflict with the rename):

**Always safe (pure research / docs / design):**
- SD3.5 TensorRT research.
- Latent upscaler research.
- Auto-refinement loop design.
- Any new TECH_DEBT or ADR authoring.

**Safe only if the rename is contained to the branch (i.e., main is untouched
except at merge time):** most code work is safe as long as the main-branch
diff is compatible with the rename at merge time. In practice this means
**any main-branch changes that touch `nodes/` or `comfyless/` files must be
merged into the rename branch before the final validation and merge to main**
(either by rebasing the rename branch, or by replaying the main-branch
changes onto the rename branch). The rename agent should complete in one
session, so this window is narrow.

## Timing

**Hold execution until:**

- Dequantize tools work is complete (`dequantize_nf4.py` fixed,
  `dequantize_comfy.py` built, both verified in ComfyUI).
- A couple more comfyless slices have landed, bringing the comfyless surface
  closer to final shape. Likely candidates: the schema refactor queued in
  Backlog, and the JSON-bridge `--json` mode stabilization once the CLI
  surface stops drifting.

Rationale: the rename touches almost every file in `nodes/` and the key
import lines in `comfyless/`. Doing it now means every subsequent feature
adds a small coordination tax (renaming anything the feature touches). Doing
it too late means the in-flight features each drag the full rename along.
The intermediate plateau — dequant done and comfyless settling — is where
the coordination cost is lowest.

**Trigger to revisit:** when the items above are done, or if the in-flight
work starts creating enough naming churn that waiting longer stops helping.

## Changelog

- **2026-04-23** — proposed. Initial spec written after scope / attribution
  / execution-pattern discussion. Awaiting dequant + comfyless settling
  before execution.

## AI-Disclosure

Claude (Opus 4.7) authored; Grant to review and sign off before execution.
