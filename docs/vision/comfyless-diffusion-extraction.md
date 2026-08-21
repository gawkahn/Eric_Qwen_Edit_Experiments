# Vision — extracting `comfyless_diffusion`

Status: proposed — slice 1 not started
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

Move the 14 verified-Grant modules into `comfyless/core/` with `git mv` (so
rename detection holds), rewrite imports across `comfyless/` and `nodes/`, and
delete the `_PROJECT_ROOT` `sys.path` insert that currently makes `nodes/`
reachable.

`_install_shims()` **stays** in this slice — `eric_diffusion_utils` and
`eric_diffusion_manual_loop` still import `comfy` / `folder_paths`, so the stubs
are still load-bearing. Removing them is slice 2.

*Proof:* battery green; `grep -r "sys.path.insert" comfyless/` empty.
*Revert:* single commit, pure moves plus import rewrites.

### Slice 2 — Cut core's ComfyUI dependency, delete the shims

Remove the direct `comfy` / `folder_paths` imports from the two core modules
that carry them, then delete `_install_shims()`.

*Proof:* `python -c "import comfyless.core"` succeeds in an interpreter where
`comfy` and `folder_paths` are provably absent; battery green.

### Slice 3 — Clear the three third-party-authored functions  ← the delicate one

Reimplement in `comfyless.core`: `build_sigma_schedule` (88 lines, the
substantial one), the Eric-authored portion of `load_lora_with_key_fix`, and the
4 lines in `decode_latents_with_upscale_vae_safe`. comfyless switches to the
core versions. **The node pack keeps its originals, untouched.**

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
