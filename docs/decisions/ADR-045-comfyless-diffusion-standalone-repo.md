# ADR-045 — Split comfyless + runtime core into a standalone repo

Status:   accepted

## Context

This repository is a public fork of `EricRollei/Eric_Qwen_Edit_Experiments`.
Eric Rollei (Eric Hiss) authored the first 27 commits beginning 2026-03-17;
Grant has authored 523. `LICENSE.txt` is a modified CC BY-NC 4.0 with mandatory
attribution (credit to Eric Hiss, a link to the original repository in
documentation, attribution comments in code, and a modification notice), plus a
commercial license by contact. Copyright is asserted as "Eric Hiss, all rights
reserved" in a per-file header carried by every file — including files Grant
wrote from scratch, where it propagated by header template.

`comfyless/` (22 modules, 22,502 lines) was created 2026-04-18 and is entirely
Grant's. It has since become the primary surface: CLI, IPC daemon, MCP server,
refinement loop, video program, catalog plane. The ComfyUI node deployments have
gone dormant in the meantime — measured 2026-08-20, all three installs are plain
directory copies with no `.git`, carrying 26 node files against the repo's 45 and
missing `eric_diffusion_generate.py` entirely. They predate 2026-04-15.

Three forces make the boundary worth drawing now:

1. **comfyless is becoming system infrastructure.** The LLM egress gateway
   migration (TECH_DEBT 2026-08-13), GPU-time reservation against the ai-stack
   scheduler, and possible adoption of local_agents' MCP server all treat
   comfyless as a component other systems invoke.
2. **local_agents is about to introduce per-process uid-based permission and
   sandboxing.** That work will pin comfyless's path, service, and process
   identity into configuration living outside this repository. TECH_DEBT
   2026-08-13 already records that the planned uid-scoped `nftables` rule denies
   comfyless by construction, so an explicit allowance is coming either way. It
   should name the final identity, not the current one.
3. **A rename of this repository was the original proposal** (strip "Eric" from
   all names). Measurement showed that to be the more expensive path for less
   benefit — see Alternatives Rejected.

### Measured boundary (2026-08-20)

Recorded so it does not have to be re-derived. The split criterion is
**mechanical**: does the module define ComfyUI node classes (`INPUT_TYPES` /
`RETURN_TYPES`)? It is not a provenance audit.

**Library layer — 100% Grant, no node classes, no intra-pack imports:**

| Module | Lines |
|---|---|
| `nodes/eric_diffusion_manual_loop.py` | 2812 |
| `nodes/eric_diffusion_utils.py` | 2228 |
| `nodes/eric_diffusion_samplers.py` | 530 |
| `nodes/eric_diffusion_scheduler.py` | 170 |

**Library layer, pulled lazily via the LoRA module — 100% Grant:**

| Module | Lines |
|---|---|
| `nodes/eric_lora_format_convert_apply.py` | 918 |
| `nodes/eric_lora_format_convert.py` | 459 |
| `nodes/eric_diffusion_lora_check.py` | 423 |
| `nodes/eric_diffusion_fp8_ops.py` | Red Zone (ADR-019) |

**`pipelines/` splits by direct comfyless import:**

- Core side: `krea2_identity_edit` (813) + NAG cluster `nag_flux2` (747),
  `nag_krea2` (516), `nag_flux` (498), `nag_zimage` (452), `nag_common` (102)
  = 3,128 lines.
- Node side: `pipeline_qwen_edit` (886), `spectrum_forward` (347),
  `spectrum_utils` (261), `pipeline_output`. `detect_pipeline_class` resolves
  through `diffusers`, not through `pipelines/`, so there is no hidden edge.

**Eric-authored code in comfyless's dependency closure — three functions:**

| Function | Size | Eric lines | comfyless import sites |
|---|---|---|---|
| `build_sigma_schedule` | 137 | 88 | 1 (`generate.py:1538`) |
| `load_lora_with_key_fix` | 200 | 51 | 2 |
| `decode_latents_with_upscale_vae_safe` | 112 | 4 | 1 |

~143 lines total. `eric_qwen_image_multistage` imports module-level from two
Eric-majority files (`eric_qwen_edit_utils`, `eric_qwen_image_generate`), so
importing `build_sigma_schedule` loads ~160 further Eric lines. That is the only
messy edge in the graph and it sits behind a single call site.

**Red Zone paths — all four land on the new side.** `comfyless/server.py`,
`comfyless/mcp_server.py`, `comfyless/refine.py`, and `nodes/eric_diffusion_fp8_ops.py`
(a weight-file content parser with no node class, therefore core). The node
repository retains zero Red Zone surfaces.

**Typecheck baseline (ADR-042 per-root):** `comfyless=13`, `nodes=520`,
`pipelines=454`. How the 520 and 454 divide across the boundary must be measured
during the split; it determines whether the new repo adopts a drive-to-zero or
ratchet posture.

**Layering inversion to fix:** `nodes/eric_diffusion_generate.py` currently
imports *from* comfyless (`hunyuan_chain`, `family_defaults`) at three sites.
The node layer depends on the CLI layer. The split inverts this correctly to
`nodes → core ← comfyless`.

## Decision

Extract comfyless and the runtime core into a new standalone repository. The
node pack remains in place, unrenamed, and consumes core as a pinned pip
dependency.

| Layer | Name |
|---|---|
| Repository + filesystem directory | `comfyless_diffusion` |
| Distribution (pip target) | `comfyless-diffusion` |
| Package path on disk | `src/comfyless/` (src layout) |
| Import package | `comfyless` (unchanged) |
| Core subpackage | `comfyless.core` |
| Console script | `comfyless` |
| Python target | **3.14** (see below) |

**Python 3.14.** The new repository targets Python 3.14, matching Ubuntu 26.04's
default, rather than carrying the current 3.12 pin forward. Verified 2026-08-20
against `uv.lock`: the entire locked tree is 3.14-clean — 22 packages ship
`cp314` wheels, 79 are pure-python, and the three that appear `cp310`-only
(`protobuf`, `safetensors`, `torchao`) are `abi3` stable-ABI. torch 2.11.0
publishes `cp314`. There is no dependency blocker.

The node repository stays on 3.12, because it runs inside ComfyUI's interpreter
alongside custom node packs whose readiness is not ours to determine. The split
boundary and the Python boundary are therefore the same line — which is an
argument for the split independent of the ones above. See
`docs/runbooks/ubuntu-26.04-upgrade.md` §A0.

**src layout.** The package lives at `src/comfyless/`, not at the repository
root. `src` is a container directory, not a package — renaming `comfyless/` to
`src/` outright would make the import `from src.generate import ...`, breaking
every reference the decision above exists to preserve, and `src` is not a
distributable name.

The layout is load-bearing here rather than cosmetic. Measured 2026-08-20:
nothing currently installs this package. There is no `comfyless` in
`.venv/lib/python3.12/site-packages` and no `.pth` file; every consumer resolves
it through `PYTHONPATH=<repo root>` (set explicitly by `systemd/comfyless@.service:48`)
or through the working directory. Because the node repository will pip-install
core, the packaged artifact must actually work — and it has never been built or
exercised. src layout keeps the working tree off `sys.path` so the test battery
runs against the installed artifact and packaging defects surface locally
instead of in the consumer.

The import package stays `comfyless` because renaming it would churn 173 Python
import references across 37 files, 78 documented `python -m comfyless.<mod>`
command lines, and 1,396 references across docs and config — roughly 1,650 edits
for no functional gain. Distribution name differing from import name is standard
Python practice.

**Dependency mechanism: pinned pip dependency, not a git submodule.** The node
pack's `requirements.txt` gains an exact pin
(`comfyless-diffusion @ git+https://github.com/gawkahn/comfyless_diffusion@<tag>`).
Local development uses an editable install so edits stay live without a release
step.

**Console script.** `[project.scripts]` is currently undeclared; everything runs
as `python -m comfyless.<mod>`. The split introduces a `comfyless` console
script so the systemd unit becomes `ExecStart=/…/.venv/bin/comfyless serve`
rather than an interpreter invocation with an embedded venv path. A stable
executable name is a better referent for a sandbox profile.

**Attribution: the new repository is cleared of third-party-authored code
before first commit.** The intent is a fresh project that acknowledges Eric
Hiss, not a second fork that carries his license terms. Those are different
outcomes and only one of them is available by choice of wording — so the code is
what changes, not the framing.

The obligation attaches to ~143 lines in three functions (measured above). If
they are carried, `comfyless-diffusion` is a derivative work and inherits the
CC BY-NC attribution terms, including the NonCommercial restriction — an
unhelpful posture for something becoming system infrastructure. If they are
cleared, the new repository is original work, its license is a free choice, and
credit to Eric Hiss becomes a courtesy acknowledgement rather than a license
condition.

Clearing is bounded and verifiable (`git blame` on the new repo reports zero
Eric-authored lines):

- `decode_latents_with_upscale_vae_safe` — 4 Eric lines of 112. Trivial.
- `load_lora_with_key_fix` — 51 Eric lines of 200; the other 149 are Grant's.
- `build_sigma_schedule` — 88 Eric lines of 137, one call site. Implements
  published formulations it names in its own docstring (Karras EDM rho-spacing,
  RES4LYF beta-CDF warp, arctan S-curve), so an independent implementation from
  the sources is straightforward rather than a rewrite-around.

**The node repository's licensing is untouched.** It remains a fork of
`EricRollei/Eric_Qwen_Edit_Experiments` under the existing dual license, with
per-file headers and attribution intact, honestly labelled as what it is. Only
the extracted repository is fresh.

**README and LICENSE for the new repository are written fresh**, crediting Eric
Hiss as the origin of the ComfyUI node pack this runtime grew out of, without
presenting the new repository as a fork of it. A pip dependency in the other
direction (node pack depends on core) does not require matching licenses.

*This records an engineering decision about which code ships where. It is not
legal advice; the licensing conclusion is Grant's to confirm.*

## Implementation hazards (measured, not anticipated)

**Package data.** 10 TOML files under `comfyless/recipes/` (7) and
`comfyless/judge_recipes/` (3) are loaded at runtime via
`Path(__file__).resolve().parent / "recipes"` (`enhance.py:138`). Wheels exclude
non-Python files unless declared as package data, so the refinement loop and the
ADR-039 duel primitive break on first install. `comfyless/examples/` JSON and
keyframe JPEGs are in the same class. Declaring package data is part of the
first slice, not a follow-up.

**`_PROJECT_ROOT` is wrong under any install.** `comfyless/__init__.py:23`
computes `Path(__file__).resolve().parent.parent` and inserts it into
`sys.path`. Under src layout that resolves to `src/`; under a site-packages
install, to `site-packages/`. It is silently wrong in both — no exception, just
a bad path.

**That hazard is also the simplification.** The `sys.path` insert, together with
`_install_shims()` stubbing `folder_paths` and `comfy`, is the entire mechanism
by which comfyless currently reaches `nodes/` and `pipelines/`. Once core is a
real subpackage, both are deleted rather than ported — core becomes an ordinary
import. Note that `eric_diffusion_utils` and `eric_diffusion_manual_loop` still
carry direct `comfy` / `folder_paths` imports, so the shim layer can only be
removed once those are cleaned; that cleanup belongs to the extraction slice.

## Alternatives Rejected

**Rename this repository in place (the original proposal).** Strip "Eric" from
176 files — 44 module filenames, 32 classes, 32 `NODE_CLASS_MAPPINGS` keys, 70
markdown files. Rejected: it is a larger blast radius for a smaller result. It
requires a provenance audit to decide which nodes may be renamed, breaks 18
saved workflow artifacts that serialize node keys, does nothing about the
license posture of the infrastructure half, and leaves comfyless inside a
repository whose directory name is what local_agents would pin. The mechanical
core/node split achieves the identity change where it matters and defers the
cosmetic rename indefinitely.

**Git submodule.** Verified that ComfyUI-Manager supports it — `git_helper.py:75`
clones with `recursive=True`, and `manager_core.py` runs
`submodule update --init --recursive` on update. Rejected nonetheless: a
submodule SHA is invisible to this project's dependency tooling (`osv-scanner`,
`deps-cve`, `deps-licenses`, `deps-report`, and the §11 exact-pin hook all read
`requirements.txt` / `pyproject.toml`), producing a supply-chain gate with a
blind spot at the one boundary that matters most. It also arrives empty on a
plain `git clone`, on GitHub "Download ZIP", and on copy-based deploys — and the
current ComfyUI deployment is copy-based.

**Monorepo with two packages.** Keeps one history and one test battery, but
leaves comfyless inheriting this repository's directory name, license posture,
and Red Zone gate configuration. Fails the primary goal.

**Vendored copy of core in the node repo.** Simplest to execute, reintroduces
exactly the drift the pin exists to prevent.

## Deferred / Out of Scope

- **Renaming Grant-authored node classes and modules.** No deadline once
  comfyless has moved; a separate low-stakes slice.
- **Correcting copyright headers in the node repository** on Grant-authored
  files that assert Eric Hiss's copyright by template propagation. The new
  repository gets correct headers by construction; the node repository's are a
  separate deliberate act.
- **The node repository's own quality-gate set.** With zero Red Zone paths it
  needs a lighter configuration than the one it inherits.
- **Splitting the test battery.** Suites import via
  `importlib.util.spec_from_file_location`, so ownership must be read from
  actual imports rather than grepped.
- **Advisory CI against core `main`** in the node repo, as early warning for
  pin-deferred breakage. Optional; likely more ceremony than a solo project
  warrants.
- **Stale ComfyUI deployments.** All three are four months behind, and
  `comfy-dev/basedir/custom_nodes/Eric_Qwen_Edit_Experiments` is an empty
  directory while six saved workflows in that tree still reference its nodes.
  Cleanup is unrelated to this decision.

## Changelog

- 2026-08-20 — Proposed. Boundary measured and recorded; naming resolved;
  submodule alternative tested against ComfyUI-Manager and rejected on
  dependency-tooling grounds.
- 2026-08-20 — Adopted src layout; recorded that nothing currently installs the
  package. Added measured packaging hazards (package data, `_PROJECT_ROOT`).
  Attribution decision changed from carry-with-headers to clear-before-first-commit,
  making the new repository original work with a freely chosen license; node
  repository licensing explicitly untouched.
- 2026-08-20 — **Accepted.** Verified that all 14 moving modules plus
  `comfyless/` are 100% Grant-authored (zero Eric commits, zero Eric blame
  lines), so the extracted history can be carried in full without importing
  third-party-authored code — history preservation and the clean-license goal
  are not in tension. Extraction plan: `docs/vision/comfyless-diffusion-extraction.md`.
- 2026-08-20 — Python target set to 3.14 for the new repository, 3.12 retained
  for the node pack; dependency tree verified 3.14-clean against `uv.lock`.

AI-Disclosure: Claude (Opus 5) authored; Grant reviewed.
