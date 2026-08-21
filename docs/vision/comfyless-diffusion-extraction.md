# Vision — extracting `comfyless_diffusion`

Status: in progress — slices 1a and 1b done
Decision record: `docs/decisions/ADR-045-comfyless-diffusion-standalone-repo.md` (accepted 2026-08-20)

## Lens (global §1)

**Team-portable.** The output is a pip-installable, version-pinned package with
a console script and a declared dependency contract — the shape you could hand
to someone else without explanation. That is a deliberate upgrade from the
current solo-defensible arrangement (a `sys.path` insert plus an inherited
working directory), and it is the reason several steps below cost more than the
minimum that would work on this box alone.

## What must be true when done

1. `comfyless_diffusion` is a standalone repository containing the comfyless
   runtime and its core library, installable as a wheel, with `comfyless` as a
   console script.
2. **Zero third-party-authored lines.** `git log --format=%an | sort -u` on the
   new repository returns exactly one name, and `git blame` agrees across every
   file.
3. The node pack in this repository consumes core as an **exact-pinned** pip
   dependency (§11) and contains no copy of it.
4. `nodes/` imports nothing from `comfyless`; the dependency runs one way.
5. The `sys.path` insert in `comfyless/__init__.py` and the `_install_shims()`
   ComfyUI stubs are **deleted**, not ported.
6. The full test battery is green in both repositories, and in the new one it
   runs against the **installed artifact**, not the working tree.
7. Generation output is bit-identical to today for the same inputs.

## What must never happen

- **A slice that ends with a red battery.** Every slice below leaves the repo
  green. There is no "broken until slice N" window.
- **A silent change to generated images.** The reimplemented sigma schedules
  (slice 3) are the only real exposure; see its proof obligation.
- **A change to the PNG `parameters` chunk contract.** Every image ever
  generated replays through it. It stays byte-compatible.
- **A Red Zone path moving without its gate configuration.** All four move
  (§ADR-045); `scripts/git-policy/`, the pre-commit config, and CI move with
  them in the same slice.
- **A dependency floor entering either `pyproject.toml`** (§11).

## Proof hooks

| Hook | When |
|---|---|
| `just tests` (25 suites) | end of every slice |
| `just typecheck` (per-root baselines) | end of every slice |
| `python -m build` + install into a clean venv + battery from site-packages | slice 5 onward |
| `git log --format=%an \| sort -u` on the new repo | slice 6 |
| Numerical equivalence harness | slice 3 |
| One live generation, same seed, pixel-diff vs a pre-restructure baseline | slices 3 and 8 |

Capture the pixel baseline **before slice 1** — same seed, same model, same
params, saved outside the repo. It is the only end-to-end guard against a
behavioural regression that unit tests miss.

## Change boundary

**In scope:** `comfyless/`, the 14 core modules under `nodes/` and `pipelines/`,
`pyproject.toml`, `requirements.txt`, `uv.lock`, `.claude/typecheck-baseline`,
`scripts/git-policy/`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`,
`systemd/comfyless@.service`, the test battery, `CLAUDE.md`.

**Explicitly out of scope:** renaming Grant-authored ComfyUI node classes or
`NODE_CLASS_MAPPINGS` keys (deferred indefinitely per ADR-045); the 18 saved
workflow artifacts; the stale ComfyUI deployments; correcting copyright headers
in the node repository.

## Slice roadmap

The governing principle: **make the boundary real inside this repository first,
where the full battery can prove it — then the split is a file move.** Slices
1–5 never leave this repo. Only 6–8 cross the boundary.

This ordering has a second payoff. Once the restructure lands, everything
destined for the new repo lives under a single prefix (`src/comfyless/`), which
turns history extraction from a multi-path rewrite into a single-prefix
operation.

### Slice 1 — Create `comfyless/core/`, move the library modules

**SPLIT during execution.** Measured, the import rewrite was 51 files, not
"comfyless/ and a few node files". Executed as two slices:

- **1a (done)** — the 6 `pipelines/` modules (nag_* + krea2_identity_edit).
  10 files, 26 import lines.
- **1b (done)** — the `nodes/` library modules. Grew from 8 to **12**: the
  closed dependency cluster pulled in `eric_krea2_convert` (imported by both
  utils and fp8_ops) and the three `eric_lora_format_convert_{flux,chroma,krea}`
  variants, which `eric_lora_format_convert` imports as registration
  side-effects via `from . import X` — a form three successive grep patterns
  missed (the first omitted `from . import`, the second used `[a-z_]+` which
  excludes the digits in `fp8_ops`, the third filtered out `importlib`). Only
  a name-based search over all files found them. 52 files, 78 rewritten lines
  plus 6 hand-fixed dynamic loaders.

**Deleting the `sys.path` insert MOVED to slice 3.** It cannot happen here:
the only remaining `nodes.*` imports in comfyless are the three Eric-lineage
functions (`load_lora_with_key_fix` / `is_direct_merge_adapter`,
`build_sigma_schedule`, `decode_latents_with_upscale_vae_safe`). The last
coupling to `nodes/` IS the third-party-authored code, so slice 3 severs the
dependency and clears the license residue in one action.

Move the 14 verified-Grant modules into `comfyless/core/` with `git mv` (so
rename detection holds), rewrite imports across `comfyless/` and `nodes/`, and
delete the `_PROJECT_ROOT` `sys.path` insert that currently makes `nodes/`
reachable.

`_install_shims()` **stays** in this slice — `eric_diffusion_utils` and
`eric_diffusion_manual_loop` still import `comfy` / `folder_paths`, so the stubs
are still load-bearing. Slice 2 removes core's need for them; the stubs
themselves survive until slice 3, because `comfyless/generate.py` still imports
`nodes.eric_qwen_edit_lora`, which imports `folder_paths` at module level.

*Proof achieved:* 32/32 suites green after each of 1a and 1b; pixel manifest
ALL MATCH; typecheck diagnostics exactly conserved at 987 across both slices
(1a moved 360 pipelines->comfyless, 1b moved 82 nodes->comfyless), which is
independent evidence that nothing was created or lost. All 18 moved files
recorded by git as renames at 99-100% similarity, so history follows them.
*Revert:* one commit per sub-slice, moves plus import rewrites.

### Slice 2 — Cut core's ComfyUI dependency  ✅ done

Remove the direct `comfy` / `folder_paths` dependencies from the core modules
that carry them.

**Deleting `_install_shims()` MOVED to slice 3**, for exactly the reason the
`sys.path` insert did. Measured: with the stubs evicted and `comfy` /
`folder_paths` blocked at the meta-path, `import comfyless.core` succeeds but
`import comfyless.generate` fails — it pulls in `nodes.eric_qwen_edit_lora`,
which has a **module-level** `import folder_paths` (line 40).
`nodes/eric_qwen_image_multistage.py` additionally reaches for `comfy.utils`
and `comfy.model_management` inside functions. The shims exist to serve those
three `nodes.*` imports, so they cannot die before slice 3 severs them. Slice 2
is therefore scoped to *core*, which is the part the extraction actually needs.

**`folder_paths` needed no change.** `resolve_component_path`
(`eric_diffusion_utils.py:398`) already wraps its import in
`try/except Exception`, so with ComfyUI absent it degrades to a no-op and
returns the path unchanged. It is an optional host-integration hook, not a
dependency, and it stays for the node pack's benefit.

**`comfy.utils.bislerp` was a real dependency and is now ours.** The shim never
stubbed `bislerp`, only `ProgressBar` — so `upscale_flux_latents` /
`upscale_flux2_latents` raised `AttributeError` under comfyless and only ever
worked inside ComfyUI. (The battery hid this: `test_manual_loop.py` injects its
own fake `bislerp`.) Upstream `bislerp` is comfyanonymous's, ComfyUI commit
`34887b88`, **GPL-3.0** — incompatible with this repo's CC BY-NC / Commercial
dual license — so it could not be vendored. `comfyless/core` now carries its
own implementation, its behaviour matched to ComfyUI's by **black-box probing
rather than reading the source**: `align_corners=False` centre-aligned
coordinates, slerped direction, lerped magnitude, `v0`-verbatim on parallel
pairs, plain lerp on antiparallel pairs, zero vectors contributing no
direction, width pass before height.

*Deliberate divergence, pinned by `test_bislerp.py`:* the two agree to ~2e-06
(float32 rounding) on every input whose sample coordinates avoid exact integer
ties — which includes **every realistic latent upscale ratio** (measured: 2x,
1.5x and mixed ratios at C=16/64 are all tie-free, so the node pack's real
output is unchanged to float32 noise). At a tie, ComfyUI's two tap-index arrays
disagree by 2 and it emits a one-row **spike discontinuity** (3 -> 19, output
row 9, where `9.5 * 3/19 == 1.5` exactly: rows `[a, a, b]` come back
`a, a, ..., a, b, mix`). We treat that as an upstream bug and interpolate
smoothly. This is the one behaviour change in the slice, and it only fires
where upstream was wrong.

*Proof achieved:* with `comfy` / `folder_paths` blocked at the meta-path and no
shims installed, `import comfyless.core`, `...core.eric_diffusion_manual_loop`
and `...core.eric_diffusion_utils` all succeed. 33/33 suites green (the glob
picked up the new `test_bislerp.py`; 23 assertions, 20 of which need no
ComfyUI, and the 3 cross-checks run against a real ComfyUI when one is
importable and skip cleanly otherwise). Typecheck `comfyless` 455 -> 453,
`nodes` and `pipelines` unchanged — a drop, as predicted for a slice that only
removes imports. Pixel manifest compared against `manifest-pre1a.json`.

### Slice 3 — Clear the three third-party-authored functions  ← the delicate one

Reimplement in `comfyless.core`: `build_sigma_schedule` (88 lines, the
substantial one), the Eric-authored portion of `load_lora_with_key_fix`, and the
4 lines in `decode_latents_with_upscale_vae_safe`. comfyless switches to the
core versions. **The node pack keeps its originals, untouched.**

**Also lands here** (both inherited from earlier slices, same structural
reason — they exist only to serve the `nodes.*` imports this slice removes):
the `_PROJECT_ROOT` `sys.path` insert, and `_install_shims()` together with the
`import comfyless  # triggers _install_shims()` line at `generate.py:31`.

*Proof obligation, and it is strict:* a harness that runs old and new
`build_sigma_schedule` across a grid of `(num_steps, denoise, schedule, power)`
covering all five schedules and asserts elementwise equality within float
tolerance. If any pair diverges, that is a **behaviour change**, not a rounding
detail — it silently alters every replay of an existing sidecar using that
schedule, and must be escalated rather than absorbed.

*Also:* `git blame` over `comfyless/core/` reports one author.

### Slice 4 — Resolve the layering inversion

`nodes/eric_diffusion_generate.py` imports `hunyuan_chain` and
`family_defaults` from comfyless at three sites. Decide per module whether it
belongs in core or whether the node drops the dependency.

*Proof:* `grep -rE "^\s*(from|import) comfyless" nodes/ pipelines/` is empty.

### Slice 5 — Packaging: src layout, package data, console script

Move to `src/comfyless/`; declare the 10 recipe TOMLs, `examples/`, and
keyframe assets as package data; add `[project.scripts]`; fix or delete
`_PROJECT_ROOT`.

*Proof:* build a wheel, install it into a clean venv, and run the **entire
battery against the installed package** with the working tree off `sys.path`.
This is the slice that proves the whole premise — nothing has ever installed
this package before.

### Slice 6 — The split

Install `git-filter-repo` (absent today; `git subtree split` is a weaker
fallback). Extract `src/comfyless/` with history into `comfyless_diffusion`.
Write the fresh README and LICENSE (crediting Eric Hiss as the origin of the
node pack this runtime grew from, without presenting the repo as a fork).
Create the 3.14 venv, relock, run the battery.

*Proof:* `git log --format=%an | sort -u` returns one name; battery green on
3.14 against the installed artifact.
*Not reversible in the usual sense* — but the source repo is untouched until
slice 7, so the fallback is "delete the new repo and retry."

### Slice 7 — Node pack cutover

Delete the moved modules here; add the exact-pinned dependency; split
`.claude/typecheck-baseline`, `scripts/git-policy/_red-zone-paths.sh` (which
becomes empty — the node pack has no Red Zone surface), `.pre-commit-config.yaml`,
and the CI workflow between the two repos. Pin the node pack to Python 3.12.

*Proof:* battery green in both repos; node pack imports resolve from the
installed core; `just policy-test` passes in both.

### Slice 8 — Infrastructure cutover

`systemd/comfyless@.service` moves to the console script and the new path,
losing its `PYTHONPATH` line. Redeploy the ComfyUI custom_nodes copies (all
three are four months stale). Update `CLAUDE.md` in both repos, the vault
manuals, and the mcpo launcher.

*Proof:* daemon starts under systemd from the new unit; one live generation
through the CLI and one through the daemon; pixel-diff against the slice-0
baseline.

## Sequencing against the OS upgrade

These are independent programs that share a resource — your attention — and one
ordering constraint. The new repo targets Python 3.14 (ADR-045), which is
26.04's default. Slice 6 can therefore either precede the upgrade (using a
uv-managed 3.14 on 24.04, which the runbook's A1 already installs) or follow it.
Running slices 1–5 first is safe either way, since they stay on 3.12.

Recommendation: land slices 1–5 before the upgrade window, hold 6–8 until after.
The restructure is verifiable on a known-good OS; the split's first act is
creating a 3.14 environment, which is cheaper on an OS that ships it.

## Open questions

1. **`hunyuan_chain` and `family_defaults` (slice 4)** — core, or does the node
   drop them? Not yet examined.
2. **Does the daemon wire protocol cross the boundary?** `comfyless/server.py`
   moves wholesale, so probably not — but if the node pack ever spoke to the
   daemon, that becomes a versioned contract.
3. **Node-pack test battery.** Which of the 4 dynamically-importing suites
   (`test_krea2_identity`, `test_vae_override_class`, `test_dry_load_integration`,
   `test_flux2`) stay here. Must be read from imports, not grepped.
4. **Does the new repo publish to PyPI**, or stay a `git+https` pin? Only
   affects whether the distribution name needs to be globally unique.
