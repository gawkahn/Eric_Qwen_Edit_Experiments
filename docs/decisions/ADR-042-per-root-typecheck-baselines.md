# ADR-042 — Per-root pyright baselines; comfyless/ drawdown to an honest floor

Status:   accepted
Date:     2026-07-27

## Context

ADR-032 set a single global ratchet integer (`.claude/typecheck-baseline`,
1026) covering `comfyless/`, `nodes/`, `pipelines/`. That number has produced
zero opportunistic drawdown since 2026-07-16: any individual file fix moves
the visible count by a few, invisible against 1026, so there is no local
signal that a plane got cleaner. The three areas also have unrelated
character — `nodes/` (520) is the ComfyUI-facing node pack, `pipelines/`
(454) is diffusers subclassing (much of it fighting third-party stub gaps),
`comfyless/` (52) is the CLI, and every §12 Red Zone surface in this repo
(`server.py`, `mcp_server.py`, `refine.py`, `generate.py`) lives there. A
single ratchet cannot express "the Red Zone plane is clean" separately from
"the diffusers plane is still messy" — it can only get bigger or smaller as
one blob.

Grant's decision (2026-07-27): drive `comfyless/` toward its honest floor now
(the actively-developed, Red-Zone-heavy plane), and replace the single
integer with per-root baselines so future drawdown on `nodes/` or
`pipelines/` has the same local signal, without pretending those two are in
scope for this slice.

`comfyless/generate.py` (7 errors) is excluded from this slice — the ADR-040
session is actively editing it concurrently; touching it here risks a merge
collision on a file neither session owns exclusively this week. `refine.py`
already has 0 errors, so it needs no separate carve-out.

## Decision

1. **Per-root baseline file, same path.** `.claude/typecheck-baseline` moves
   from a bare integer to `root=count` lines, one per pyright root declared
   in `[tool.pyright] include`:

   ```
   comfyless=<measured at mechanism-slice commit>
   nodes=520
   pipelines=454
   ```

   Same file path (git history / the single-file assumption in existing
   tooling stays mostly intact), new format. A bare-integer file is no longer
   valid input to any of the three enforcement layers after this slice lands
   — all three move together in the same commit (see §2 of the companion
   mechanism slice).

2. **Enforcement compares per-root, blocks on ANY root regression.** The
   local hook, the git-policy pre-commit check, the git-policy range check,
   and the CI job all group pyright's per-file diagnostic count by top-level
   root (first path segment under the repo root: `comfyless`, `nodes`,
   `pipelines`) and compare each root's current count against that root's
   committed baseline independently. A commit that lowers `comfyless` from 52
   toward its floor while `nodes`/`pipelines` stay flat passes cleanly — that local
   signal is the entire point of this ADR. A commit that raises any one
   root's count blocks, same override path as before (`# user-approved` /
   `Policy-override:`).

3. **`comfyless/` target is NOT zero — the honest floor is stated, not
   implied.** Measured 2026-07-27, pyright 1.1.411: 52 errors across 10
   files (`generate.py`'s 7 excluded from this slice's scope — see Context).
   Investigation before writing fixes (required by the guardrail below —
   "narrow it, or is the bug real?" can't be answered by inspection alone)
   found the errors split three ways, not two:
   - **Genuinely fixable, most already found and fixed in this slice:**
     `cascade.py`'s 5 are pyright false positives from diffusers 0.39.0's
     lazy `__init__.py` (`reportPrivateImportUsage` — the symbols ARE
     re-exported at runtime via `_import_structure`/`__getattr__`, pyright's
     static reimport-block analysis just doesn't credit it as a *public*
     re-export). Fixed by importing from the defining submodule directly, as
     pyright's own message suggests — not a suppression. One of the four
     (`from diffusers.pipelines.wuerstchen import PaellaVQModel`) was a
     genuinely dead import path — `PaellaVQModel` moved to
     `diffusers.pipelines.deprecated.wuerstchen.modeling_paella_vq_model` at
     some prior diffusers bump and `_load_stage_a` (Stable Cascade Stage A
     single-file load) has been raising `ModuleNotFoundError` at that call
     site ever since, silently, because nothing exercises that path in the
     test suite. This is the exact shape §15 predicts: pyright caught a real
     runtime bug that 129 passing `test_cascade.py` cases did not. `catalog.py`,
     `catalog_builder.py`, `params_validation.py` are plain `Optional`
     narrowing — no suppression needed.
   - **Structural, left residual:** `__init__.py` (6) —
     `Cannot assign to attribute X for class "ModuleType"`. Pyright objecting
     to monkeypatching ComfyUI's `folder_paths` / `comfy.utils` /
     `comfy.model_management` modules at import time — the standard pattern
     for making a custom node pack import outside a live ComfyUI process. Not
     a bug; a typed stub module would silence it at real engineering cost for
     a cosmetic gain. Same precedent ADR-032 set for the `comfy.*`
     `reportMissingImports` cluster.
   - **Excluded from scope, not counted:**
     `integrations/openwebui/generate_image_tool.py` (4) — see Decision §4.
   Landing zone: **comfyless=6** if `enhance.py`'s `_BaseModelWithGenerate`
   overload-resolution errors (2, a `transformers` stub quirk on
   `model.to(device)` / `.generate()` — confirmed correct, running code) also
   resolve without suppression, plus whatever the `mcp_server.py` /
   `server.py` review turns up. The measured final number is recorded in the
   Changelog once slices 3-5 land — this ADR commits to the *shape* of the
   floor (structural residual only, no swept-under-the-rug suppressions), not
   a number fixed before the fixes were written.

4. **`comfyless/integrations/openwebui/generate_image_tool.py` is excluded
   from `[tool.pyright]` scope, not fixed.** It imports `fastapi`,
   `starlette`, `open_webui.models.users`, `open_webui.routers.files` — all
   of which resolve only inside the OpenWebUI container this file is
   installed into (see its own module docstring: "runs INSIDE the OpenWebUI
   runtime"). It has never run in this repo's own environment and pyright
   here is checking against the wrong runtime by construction. Excluding it
   is honest; leaving it in the baseline as permanent residual would hide a
   real "this repo cannot see the OWUI host API" fact behind a number that
   looks like debt. Added to `[tool.pyright] exclude` in `pyproject.toml`.

5. **ADR-032's Status is left `accepted`, not superseded.** Per-root
   baselines replace ADR-032's *aggregation* (one integer) but not its
   *posture* (ratchet, ADR-032 §Decision 3-5: tool choice, scope, the
   `comfy.*` missing-import precedent, the two enforcement layers). ADR-032
   is still the governing document for "why pyright, why ratchet, why not
   drive-to-zero on nodes/pipelines"; ADR-042 only changes the shape of the
   number ADR-032's mechanism reads. Referenced from ADR-032's Changelog
   below rather than marked superseded.

## Alternatives Rejected

- **Per-file baselines** — finer than needed: 72 files' worth of integers is
  maintenance overhead for little signal gain beyond what per-root already
  gives, and invites the same drift the single global integer suffered
  (nobody updates 72 numbers opportunistically; three roots is tractable).
- **JSON baseline file** — no behavioral gain over `root=count` text lines,
  and the existing hook/git-policy tooling is bash + `grep`/`tr`, not
  bash + `jq` for this file (jq is already a dependency for hook JSON
  parsing elsewhere, but adding it to the baseline-file path is unforced
  complexity for three key-value pairs).
- **Drive comfyless/ to literal zero** — `__init__.py`'s ModuleType
  complaints are the cost of the monkeypatch pattern every ComfyUI custom
  node pack uses to be importable outside a live ComfyUI process; a typed
  stub module is real work (a `comfy-stubs` package tracking ComfyUI's
  actual API surface) for a cosmetic win. Revisit only if this repo ever
  ships such a stub package for another reason.
- **Fix `generate.py` too, since it's only 7 errors** — explicitly
  out of scope: the ADR-040 session owns that file this week (see the
  file-lease table in the handoff this slice worked from). Picked up
  whenever comfyless/ drawdown resumes after ADR-040 lands.

## Deferred / Out of Scope

- `nodes/` (520) and `pipelines/` (454) per-root baselines are seeded at
  their current measured counts (no drawdown attempted this slice) — the
  mechanism now exists for either to be picked up as its own future slice.
- `generate.py`'s 7 errors — next comfyless/ drawdown slice once ADR-040
  lands and the file lease clears.
- A `comfy-stubs` typed stub package for the `__init__.py` monkeypatch
  cluster and the `comfy.*` `reportMissingImports` cluster ADR-032 already
  left residual.

## Changelog

- 2026-07-27 — accepted. Baseline mechanism converted to per-root; comfyless/
  drawdown slices (mechanism + 3 fix commits) tracked here as they land.
- 2026-07-27 — mechanism-slice code-reviewer clarifications on §Decision 1-2
  (no behavior change, wording only): (a) only the local hook
  (`.claude/hooks/require-typecheck-clean.sh`) and the CI `typecheck` job
  actually invoke pyright and group diagnostics by root, via the shared
  `scripts/typecheck-per-root.sh`. The two git-policy layers
  (`pc_baseline_no_increase` in `scripts/git-policy/_lib.sh`, called from
  `pre-commit-checks.sh` and `check-range.sh`) never run pyright — they only
  compare the committed baseline FILE's old vs new content per root, which is
  what catches a same-commit self-legalizing bump. (b) §Decision 1's "a
  bare-integer file is no longer valid input" describes the steady state
  going forward, not the transition commit itself: `pc_baseline_no_increase`
  and the local hook both deliberately treat a bare-integer HEAD (no `root=`
  lines) as "no prior baseline for any root" — nothing to compare, nothing
  blocks — which is exactly what let the mechanism-implementation commit land
  without needing a manual override.

AI-Disclosure: Claude (Sonnet 5) authored; Grant reviewed.
