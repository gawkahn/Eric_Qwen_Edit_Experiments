# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Backlog
`~/obsidian/vaults/vault1/10_Projects/Image_gen/Backlog.md`

## Project documentation artifacts

**Vault project root:** `~/obsidian/vaults/vault1/10_Projects/Image_gen/`

Rules for ADRs, security reviews, and the tech debt register are in §12 of
`~/.claude/CLAUDE.md`. Local paths for this project:

- **Repo:** `docs/decisions/`, `docs/security/`, `docs/vision/`, `TECH_DEBT.md` at project root
- **Vault mirrors:** `Decisions/`, `Security/`, `Vision/`, `Tech_Debt.md`,
  `Backlog.md` (all under the Vault project root above)
- **Comfyless user docs (vault-ONLY, no repo copy — rewritten 2026-07-09):**
  `Comfyless_Manual.md` (main) + `Comfyless_Models.md` + `Comfyless_MCP.md` +
  `Comfyless_Catalog.md`, under the Vault project root. Obsidian wikilink
  conventions. Update these when user-facing comfyless behavior changes.

This is a solo personal project — `docs/` in the repo is the canonical store.
Vault copies are a personal reference mirror.

---

## Project Overview

ComfyUI custom node set wrapping two 20B-parameter Qwen models from Alibaba:
- **Qwen-Image-Edit-2511** — Image editing up to 17 MP
- **Qwen-Image-2512** — Text-to-image generation up to 50+ MP

This is a ComfyUI extension; there is no standalone executable, test suite, or build step. Development is done by editing node files and reloading ComfyUI.

## Development Workflow

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Package-manager split (pip for the node pack, uv for comfyless dev):**

This repo uses two tools deliberately:

- **ComfyUI node pack path** — `pip` is the convention. ComfyUI Manager installs custom node packs by running `pip install -r requirements.txt` inside ComfyUI's venv. `requirements.txt` is the canonical manifest for downstream users and must remain pip-compatible.
- **Comfyless dev path** — `uv` is the preferred tool for local development, testing, and reproducibility work. `pyproject.toml` is the human-edited source of truth for dep declarations; `uv.lock` is the machine-generated full transitive lock (kept in version control so `uv sync` is reproducible across machines). `.python-version` pins the interpreter.

Rules:
- **`pyproject.toml` and `requirements.txt` must agree on direct deps at all times** — both list the same 17 top-level pins in the same order (`torch`, `torchvision`, `torchao`, `diffusers`, `transformers`, `accelerate`, `peft`, `safetensors`, `pillow`, `numpy`, `mcp`, `click`, `scipy`, then the tokenizer backends `sentencepiece`, `protobuf`, `tiktoken`, `ftfy`). Any dep bump edits both. `torchvision` must track `torch`'s minor (2.11 ↔ 0.26). **Documented exception (ADR-045 slice 7):** `comfyless-diffusion==0.1.0` is
pyproject-only. It resolves from the local checkout via `[tool.uv.sources]`
because the repository has no remote yet, and a path source is not something
`requirements.txt` can express for ComfyUI Manager. It joins requirements.txt
as a `git+https` pin when the remote exists (TECH_DEBT 2026-09-09).
**Documented exception (ADR-033):** `av` (video encode) is pyproject-only — the node pack never imports it, so it deliberately does NOT appear in `requirements.txt`.
- **`uv.lock` is regenerated whenever `pyproject.toml` changes** — `uv lock` after the edit, then commit pyproject + requirements + lock together in one slice.
- **Do NOT edit `uv.lock` by hand.** It's machine output.
- Fresh dev setup: `uv sync` (creates `.venv` matching the lock). ComfyUI install still uses pip as before — no change for downstream users.

**Lint / syntax check (also gated in CI since 2026-07-16 — see §"Commit-time hooks & quality gates"):**
```bash
python -m py_compile nodes/<file>.py   # syntax check a single file
```

**Test suites (`just tests` locally; gated in CI since 2026-07-16):**
```bash
./.venv/bin/python3 test_multistage.py           # multistage infrastructure (nodes/)
./.venv/bin/python3 test_lora_alpha_bake.py      # LoRA alpha baking (nodes/)
./.venv/bin/python3 test_lora_adapters.py        # ADR-046 differential: comfyless.core vs the node original
./.venv/bin/python3 test_hunyuan.py              # differential: comfyless vs nodes kwargs
./.venv/bin/python3 test_fp8_single_file.py      # scaled-fp8 loader + DMR (node-pack half of the source guards)
./.venv/bin/python3 test_lora_convert_krea.py    # Krea-2 LoRA conversion + fp8-resident buffers
./.venv/bin/python3 test_lora_order_insensitive.py  # order-insensitive direct merge
./.venv/bin/python3 test_flux2.py                # LIVE GPU smoke — outside `just tests`
```

**ADR-045 slice 7 (2026-09-09) split the battery.** The other 29 suites moved
to `comfyless_diffusion` along with the code they exercise; run them there with
`just tests` on its own 3.14 venv. What remains here is the node pack's own
tests plus the five SIDE-BY-SIDE differentials, which stay in this repository
precisely because it is the one that has both halves — `nodes/` locally and
`comfyless` as an installed dependency. `just tests` is glob-based and picks up
whatever is present, so the list above is descriptive; the glob is
authoritative.
Suites run against this repo's uv-managed `.venv` — invoke via
`./.venv/bin/python3` (created by `uv sync`). `comfyless` resolves from the
installed dependency, NOT from this tree (ADR-045 slice 7), so the three
suites that read comfyless source do it through the import system via the
`cf_path()` helper in `comfy_stub.py`, never a repo-relative path. The
`tests/test_lora_format_convert*.py` suites under `tests/` remain outside the
battery — see the TECH_DEBT entry before pulling them in.

`test_flux2.py` is a live GPU smoke test that performs an actual Flux.2 generation — separate from the unit suites above. Run only when you need to verify end-to-end Flux.2 behavior.

## Git commit conventions for this repo

See the general `Git Commit Discipline` rule in `~/.claude/CLAUDE.md` for the cadence and staging rules that apply to every Claude session. The additions below are the repo-specific conventions layered on top.

**Commit message style** — matches the existing history shown by `git log --oneline`:

- Prefix: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `update:`, `deps:` for dependency bumps, `chore:`, `tool:` for standalone CLI tools, `workflows:` for workflow JSON artifacts (the enforced set — `scripts/git-policy/_lib.sh` `pc_conventional`)
- Imperative mood, lowercase after the prefix
- Short first line (≤72 chars), optional body explaining the _why_ not the _what_
- **Every AI-produced commit must include both trailers** (global §0 rule 6 + §7):
  ```
  AI-disclosure: Claude (Sonnet 4.6) authored; Grant reviewed.
  Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
  ```
  Use the tier that actually wrote the code. The `AI-disclosure:` line is enforced by the pre-commit hook in `.claude/hooks/check-ai-disclosure.sh`.

**Files that belong in commits:**

- Node files (`nodes/`), pipelines (`pipelines/`), tests (`test_*.py`), docs (`*.md`) — always committed
- Workflow JSON files (`workflows/*.png`, `*.json`) — commit in their own slice separate from code changes; they're artifacts, not logic, and should be readable in history as "workflows: …"
- Standalone CLI tools (`analyze_checkpoint.py`, `dequantize_nf4.py`) — commit individually with `tool:` prefix since they're self-contained utilities not imported by nodes
- `CLAUDE.md`, `.gitignore`, `requirements.txt`, `README.md` — committed with their respective content changes

**Files that must NEVER be committed:**

- `session-handoff-*.md` — ephemeral scratch notes from prior sessions; belong in `.gitignore`
- `memory/` directory — Claude's internal persistent memory system (user/feedback/project/reference notes); session-specific and should stay in `.gitignore`
- `api_keys.ini`, `.env` — already gitignored, never bypass
- `__pycache__/`, `*.pyc`, `*.pyo` — already gitignored

**Staging discipline for this repo specifically:**

- Before touching `eric_diffusion_*.py` or `eric_diffusion_advanced_*.py`, run `git diff <file>` on every file you intend to change so you see the full starting state
- The manual loop module `eric_diffusion_manual_loop.py` is large (>2600 lines) — when committing changes to it, write the commit message based on the _semantic_ change (which function/path you touched), not the line count
- The test files import from `nodes/eric_diffusion_manual_loop.py` via `importlib.util.spec_from_file_location` — they DON'T fail if the module file is missing at import time, but they fail silently with confusing errors. When committing a test-only change, verify the tested module is already in git, or bundle test+module changes in the same commit

**Remote sync cadence:**

The global `Git Commit Discipline` rule "Never push to remote without explicit user approval" still holds. The repo-specific addition is about *when* to seek that approval: **after a logical batch of related commits concludes, proactively ask whether to push the batch.** A logical batch is what would be one PR if PR requirements were active — typically 1–5 commits sharing a coherent purpose (a feature slice, a clean-up batch, a multi-commit fix, an ADR + matching slice-Vision update). Do **not** ask after every commit individually — that pattern was an overcorrection from the 2026-04-21 47-commit-drift incident and treats the remote as raw off-site backup rather than as a logical-batch boundary. (Updated 2026-05-04.)

- Default flow: commit → continue working until the batch concludes → ask the user "push `origin main` now?" → push on yes, hold on no.
- If the user's batch-approval message already said to push (e.g. "commit and push this batch"), that's the push approval — no second ask for that one.
- "Hold" on one batch does not extend to the next — ask again after the next batch concludes.
- **Drift floor (don't lose this):** the 47-commit drift on 2026-04-21 is still the worst case to avoid. If commit count grows past ~5 without a clean batch boundary in sight, surface that as a flag rather than continuing silently. A session that ends with a long uncommitted-and-unpushed batch should explicitly state where the natural batch boundary was missed.
- Force push, skipping hooks (`--no-verify`), and signing bypass remain separate per-invocation approvals regardless.

## Review bar (this project)

**§5 Red Zone (auth / PII / billing / audit):** Currently absent — no auth, no PII, no billing, no audit trail. Solo desktop tool.

**§12 security review triggers — already present:**

§12 is broader than §5 and this project already trips it on the surfaces below.
The file-scoped ones are mechanically gated by
`scripts/git-policy/_red-zone-paths.sh` (commit-policy layer, adopted
2026-07-16) — keep that list and this table in sync:

**ADR-045 slice 7 (2026-09-09): this repository no longer has a §12 surface.**
Every file in the table below moved to `comfyless_diffusion`, which carries its
own copy of the review bar and of `scripts/git-policy/_red-zone-paths.sh`. The
table is kept, marked, because `check-range` over a range reaching before the
split still gates these paths — see the historical-patterns note in that script.

| Surface (all MOVED to comfyless_diffusion) | Former path here | Trigger |
|---------|------|---------|
| Unix socket IPC server | `src/comfyless/server.py` | IPC (Unix sockets) — ADR-001 |
| MCP server | `src/comfyless/mcp_server.py` | LLM agent tool surface — ADR-011 |
| Refinement-loop judge/seed | `src/comfyless/refine.py` | LLM output influencing generation params — ADR-027 |
| HF repo ID resolution + download | `src/comfyless/core/eric_diffusion_utils.py` `resolve_hf_path` (function-scoped) | Loading model weights from caller-supplied paths |
| `--json` stdin/stdout bridge | `src/comfyless/generate.py` `_run_json_mode` (function-scoped) | Machine-facing interface |
| Scaled-fp8 / int8 file-content parser | `src/comfyless/core/eric_diffusion_fp8_ops.py` | Parsing caller-supplied weight-file CONTENT — ADR-019 |
| LoRA adapter subsystem | `src/comfyless/core/lora_adapters.py` | Daemon LoRA weight writes / backups / registry — ADR-046 |

**Debt: CLOSED — the claim here was wrong.** This paragraph asserted that no
§12 review existed for `resolve_hf_path` (caller-supplied model loading), and
carried that as backlog. Two reviews do exist and always did:
`docs/security/review-resolve-hf-path-2026-04-23.md` and
`review-resolve-hf-path-hardening-2026-04-23.md`, the second closing three
MEDIUM findings from the first. Corrected 2026-09-09 while writing the sibling
repo's CLAUDE.md, when the claim was checked against the directory instead of
copied forward. The `src/comfyless/server.py` half was likewise closed by
ADR-001 + `review-comfyless-server-2026-04-23.md` /
`review-comfyless-server-hardening-2026-04-23.md`. Both surfaces now live in
comfyless_diffusion; the reviews are in both repos' `docs/security/`.

**Surfaces that become Red Zone on scope change:**

- **`--json` bridge + LLM agent wiring** (Backlog) — once model output drives paths or parameters into `generate()`, this becomes a Red Zone surface: prompt injection, path traversal, actor identity. Treat any commit that wires this as Red Zone from day one, not after.
- **HTTP transport** — if `--serve` ever grows a network interface, that commit is Red Zone regardless of other scope.
- **Batch generation from external input** — file writes at scale from caller-supplied lists is a §12 trigger.

**Review rules:**

- **Every non-trivial code slice runs `code-reviewer` (Fable) before commit.** "Trivial" = single-line fix, pure doc edit, mechanical rename with no behavior change.
- **This repository has no current §12 surface** (ADR-045 slice 7 — see the table above). The `_red-zone-paths.sh` patterns are retained so `check-range` still gates commits from BEFORE the split, and they fail closed if a moved file ever reappears here. `security-auditor` triggers for those surfaces now live in `comfyless_diffusion`.
- **When the `--json` / LLM agent wiring lands:** write spec + ADR before code, run `security-auditor`, treat as Red Zone from the first commit.
- Trivial skip ask: `"Trivial — skip review? Change: <one-line summary>. Reply 'review' to run it anyway."` Do not self-decide.
- Pass `model: "fable"` explicitly at every Agent-tool invocation for reviewer agents (`code-reviewer`, `security-auditor`). The frontmatter pin is known-broken in Claude Code 2.1.117 — structural enforcement requires the invocation-time override.

## Commit-time hooks & quality gates

Three enforcement layers (quality-gate kit adoption 2026-07-16 — `secrets` +
`commit-policy` gates only; types/tests/sast/supply-chain NOT adopted yet, see
the kit README in `~/.claude/templates/quality-gate-kit-python-uv/`):

1. **Harness hook (AI-facing, earliest):** `.claude/settings.json` installs a
   `PreToolUse` hook on `Bash` that rejects `git commit -m "..."` calls whose
   message lacks an `AI-disclosure:` trailer (global §7 / §0 rule 6). Script:
   `.claude/hooks/check-ai-disclosure.sh`. Bypassable by editor commit (no
   `-m`); human-only commits use `AI-disclosure: none`.
2. **pre-commit layer (real git state, every committer):**
   `.pre-commit-config.yaml` + `scripts/git-policy/`. Enable once per clone:
   `uv run pre-commit install --hook-type pre-commit --hook-type commit-msg`
   (pre-commit is in the uv `dev` dependency group). Checks: conventional
   subject, AI-disclosure trailer, no pyproject dep floors, TECH_DEBT.md
   append-only, Red Zone spec(=ADR)/review references, gitleaks secret scan,
   config-file hygiene. Red Zone paths live in
   `scripts/git-policy/_red-zone-paths.sh` (keep in sync with the Review bar
   above; `_run_json_mode` / `resolve_hf_path` are function-scoped and NOT
   path-gated — see TECH_DEBT.md). Escapes: `Policy-override:` line in the
   message skips the Red Zone reference checks; smoke tests:
   `just policy-test`.
3. **CI mirror (authoritative once branch protection exists):**
   `.github/workflows/ci.yml`, six jobs — git-policy smoke tests, gitleaks,
   semgrep sast, supply-chain (sources/licenses/CVE), pyright ratchet, and
   the `just tests` battery on every push/PR; plus the commit-range policy
   check on PRs.
4. **Typecheck ratchet (ADR-032; per-root, ADR-042):** `.claude/typecheck-baseline`
   holds one `root=count` line per top-level pyright root (`comfyless`,
   `nodes`, `pipelines` at adoption of ADR-042) — each root may only go DOWN
   independently, not one combined integer. A second PreToolUse hook runs
   `scripts/typecheck-per-root.sh` (~11 s) before every `git commit` and
   blocks if ANY root's count is above HEAD's baseline for that root;
   same-commit baseline bumps are blocked at the git-policy layer too (also
   per root). Deliberate bump: `# user-approved` on the command /
   `Policy-override:` in the message. When you fix type errors, lower that
   root's line in `.claude/typecheck-baseline` in the same commit.

Toolchain pins: `mise.toml` (gitleaks, just, osv-scanner, node, pyright —
`mise trust ./mise.toml && mise install`). Recipes: `just secrets`,
`policy-test`, `sast`, `typecheck`, `tests`, `deps-cve`, `deps-licenses`,
`deps-verify-sources`, `deps-report`. The gitleaks baseline is 0 (history
measured clean at adoption; `.gitleaks.toml` has no allowlist — this repo's
tests embed no credential-shaped fixtures). License policy: ADR-031. CVE
ignores (torch/setuptools, no reachable fix): `osv-scanner.toml` + the
`deps-cve` recipe flags, tied to the next torch bump.

## Architecture

### ComfyUI Registration
`nodes/__init__.py` imports all node classes and defines `NODE_CLASS_MAPPINGS` (internal key → class) and `NODE_DISPLAY_NAME_MAPPINGS` (internal key → UI label). Adding a new node requires: create file in `nodes/`, import class in `nodes/__init__.py`, add both mappings.

### Node Groups & Naming Conventions
- `eric_qwen_edit_*.py` — Edit pipeline nodes (loader, inpaint, LoRA, spectrum, etc.)
- `eric_qwen_image_*.py` — Generation pipeline nodes (loader, UltraGen, ControlNet, etc.)
- `eric_qwen_*.py` — Shared utilities (prompt rewriter, VAE loader)

Each node file defines a class with:
- `CATEGORY`, `FUNCTION`, `RETURN_TYPES`, `RETURN_NAMES` class attributes
- `INPUT_TYPES(cls)` classmethod returning required/optional input dicts
- A main method matching `FUNCTION` that does the work

### Pipeline Objects Passed Between Nodes
Nodes communicate via typed pipeline dicts, not raw model objects:
- `"QWEN_EDIT_PIPELINE"` — `{"pipeline": <QwenEditPipeline>, "model_path": str, ...}`
- `"QWEN_IMAGE_PIPELINE"` — `{"pipeline": <QwenImagePipeline>, "model_path": str, "offload_vae": bool}`
- `"QWEN_CONTROLNET_PIPELINE"` — wraps ControlNet model reference

### Custom Pipeline (`pipelines/`)
`pipeline_qwen_edit.py` extends the diffusers `QwenImageEditPlusPipeline` with:
- Native resolution preservation (aligned to 32px, capped at `DEFAULT_MAX_PIXELS = 16 MP`)
- True CFG support (`true_cfg_scale`) with norm-preserving rescaling
- Dual conditioning: VL path (Qwen2.5-VL tokens) + VAE/ref path (pixel latents)
- Spectrum acceleration hooks (Chebyshev feature forecasting, CVPR 2026)

`spectrum_utils.py` / `spectrum_forward.py` — patch the transformer's forward pass for 3–5× speedup without retraining.

### Model Caching
Loaders use module-level cache dicts (in `eric_qwen_edit_loader.py` and `eric_qwen_image_loader.py`). The cache stores a single pipeline + its config key; a different config triggers eviction and reload. Functions `get_gen_pipeline_cache()` / `clear_gen_pipeline_cache()` are imported by the component loaders.

### LoRA Loading (Three-Tier Fallback)
See `eric_qwen_edit_lora.py` / `eric_qwen_image_lora.py`:
1. **Fast path** — `pipeline.load_lora_weights()` (PEFT / diffusers native)
2. **PEFT injection** — inject adapter layers manually, then load
3. **Direct merge** — load state dict and merge weights into model parameters

Supports LoRA, LoKR, and LoHa formats with auto prefix detection.

### Guidance: Embedding vs. True CFG
**Critical distinction** (documented in `DEV_NOTES.md`):
- Guidance-distilled models (Flux.1-dev, SD3.5-Medium): `guidance_scale` is fed as a transformer input embedding — one forward pass per step, requires trained distillation.
- Qwen-Image-2512: `transformer.config.guidance_embeds = False` — guidance embedding is dead code. Use `true_cfg_scale` (standard CFG, 2× forward passes). Official recommendation: 50 steps, `true_cfg_scale = 4.0`.

### UltraGen Multi-Stage
`eric_qwen_image_ultragen.py` runs up to 3 progressive upscale stages. Each stage independently controls: steps, CFG scale, denoise strength, sigma schedule (`linear`/`balanced`/`karras`), seed mode, and LoRA weight. ControlNet variant in `eric_qwen_image_ultragen_cn.py`.

## Generic Multi-Model Nodes (`GEN_PIPELINE` type)

Three new nodes in `nodes/eric_diffusion_*.py` support any diffusers text-to-image model without model-specific code:

| Node | File |
|------|------|
| Eric Diffusion Load Model | `eric_diffusion_loader.py` |
| Eric Diffusion Unload | `eric_diffusion_loader.py` |
| Eric Diffusion Generate | `eric_diffusion_generate.py` |
| (shared helpers) | `eric_diffusion_utils.py` |

**Auto-detection:** loader reads `model_index.json → _class_name`, maps to a short `model_family` string (`"qwen-image"`, `"flux"`, `"flux2"`, etc.), and dynamically instantiates the pipeline class via `getattr(diffusers, class_name)`. New model families in diffusers work automatically.

**`GEN_PIPELINE` dict:**
```python
{
    "pipeline":        <pipeline obj>,
    "model_path":      str,
    "model_family":    "qwen-image" | "flux" | "flux2" | ...,
    "offload_vae":     bool,
    "guidance_embeds": bool,   # from transformer.config.guidance_embeds
}
```

**CFG routing in generate node:**
- `qwen-image` → `true_cfg_scale` (double-pass CFG), negative prompt used
- `flux` / `flux2` → `guidance_scale` (guidance embedding, single pass), negative prompt ignored
- unknown → `inspect.signature(pipe.__call__)` introspection, passes only accepted params

Existing `QWEN_IMAGE_PIPELINE` / `QWEN_EDIT_PIPELINE` nodes are untouched and use their own cache. The new nodes have a separate cache in `eric_diffusion_utils.py`.

**Edit nodes for Flux:** Flux.2 has no native edit variant. Future options: image-to-image via reference latent conditioning, or dedicated nodes using Flux.2-Klein-9B.

## Key Files for Common Tasks

| Task | File |
|------|------|
| Add a new Edit node | `nodes/eric_qwen_edit_<name>.py` + register in `nodes/__init__.py` |
| Add a new Generation node | `nodes/eric_qwen_image_<name>.py` + register in `nodes/__init__.py` |
| Modify pipeline inference logic | `pipelines/pipeline_qwen_edit.py` |
| Change Spectrum acceleration | `pipelines/spectrum_utils.py`, `pipelines/spectrum_forward.py` |
| Modify LoRA loading | `nodes/eric_qwen_edit_lora.py` or `nodes/eric_qwen_image_lora.py` |
| Prompt rewriting / LLM API | `nodes/eric_qwen_prompt_rewriter.py` (reads `api_keys.ini`) |
| 2× VAE upscale (Wan2.1) | `nodes/eric_qwen_upscale_vae.py` |

## OpenWebUI integration (comfyless → mcpo → OWUI)

**The comfyless half of this now lives in the sibling repository** (ADR-045 slice 7); paths below are relative to `../comfyless_diffusion/`. What stays here is `start-mcpo.sh`.

`src/comfyless/integrations/openwebui/generate_image_tool.py` is a native OpenWebUI Tool (runs inside the OWUI container) that drives image generation from chat and renders results inline. It calls the comfyless MCP server through the **mcpo** OpenAPI bridge — launched via `start-mcpo.sh` at THIS repo's root, which since ADR-045 slice 8 spawns `comfyless-mcp` from the sibling's venv — override `COMFYLESS_REPO` to point it at a different comfyless_diffusion worktree (model-base = `hf-local`, the curated set; scanning the parent `.../models` also surfaces HF-cache snapshot-hash names). Tools exposed to the model: `generate_image`, `list_models`, `list_loras`, `list_transformers` (catalog names only, no paths). Requires a tool-calling model — gpt-oss works; roleplay-finetuned models (e.g. Dolphin-Venice) do not reliably emit tool calls. See ADR-017 and `src/comfyless/integrations/openwebui/README.md`.

The MCP server (`../comfyless_diffusion/src/comfyless/mcp_server.py`) caches one pipeline in-process and evicts + frees it on config change (mirrors the `server.py` daemon) so a long-lived server doesn't OOM across model switches; LoRAs are applied via the shared `generate._apply_loras`. See `docs/security/review-mcp-pipeline-cache-2026-06-27.md`.

## The comfyless daemon (systemd)

`systemd/comfyless@.service` lives here but runs the SIBLING repo's code: since
ADR-045 slice 8 its `ExecStart` is
`comfyless_diffusion/.venv/bin/comfyless --serve --device cuda:%i`, one instance
per GPU (`systemctl --user start comfyless@0`). It carries no `PYTHONPATH` —
the console script pins its own interpreter, which is why `.venv/bin` rather
than the CWD lands on `sys.path[0]`.

Reinstall after editing:
`cp systemd/comfyless@.service ~/.config/systemd/user/ && systemctl --user daemon-reload && systemctl --user restart comfyless@0 comfyless@1`

**Caution:** this repo's `.venv/bin` still carries all six `comfyless*` console
scripts via the local path dependency, so an older unit pointing here would
keep working while silently running a different transitive dependency tree.
A wrong unit fails quietly, not loudly. See
`docs/security/review-slice8-infra-cutover-2026-09-09.md`.

## Important Constraints

- All model loading uses `local_files_only=True` — no internet access during inference.
- Dimension alignment is 32px throughout; violating this causes transformer shape errors.
- `pipeline.vae.enable_tiling()` is always called on generation pipelines — required for >2 MP decode without OOM.
- The Edit pipeline takes a `Qwen2VLProcessor` (vision-language processor); the Generation pipeline does **not** — it uses only a tokenizer

