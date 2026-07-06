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
  `Backlog.md`, `Comfyless_Manual.md` (all under the Vault project root above)

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
- **`pyproject.toml` and `requirements.txt` must agree on direct deps at all times** — both list the same 17 top-level pins in the same order (`torch`, `torchvision`, `torchao`, `diffusers`, `transformers`, `accelerate`, `peft`, `safetensors`, `pillow`, `numpy`, `mcp`, `click`, `scipy`, then the tokenizer backends `sentencepiece`, `protobuf`, `tiktoken`, `ftfy`). Any dep bump edits both. `torchvision` must track `torch`'s minor (2.11 ↔ 0.26).
- **`uv.lock` is regenerated whenever `pyproject.toml` changes** — `uv lock` after the edit, then commit pyproject + requirements + lock together in one slice.
- **Do NOT edit `uv.lock` by hand.** It's machine output.
- Fresh dev setup: `uv sync` (creates `.venv` matching the lock). ComfyUI install still uses pip as before — no change for downstream users.

**Lint (no CI configured — run manually):**
```bash
python -m py_compile nodes/<file>.py   # syntax check a single file
```

**Test suites (no CI, run manually):**
```bash
python3 test_manual_loop.py                 # 186 tests: samplers, manual loop, encode helper, Qwen edit
python3 test_multistage.py                  # 141 tests: multistage infrastructure
python3 test_params_schema.py               # 193 tests: comfyless COMFYLESS_SCHEMA + adapters + krea routing/rebalance + Krea2 attention-backend pin + Z-Image base/Turbo name-hint detection & routing (ADR-009)
python3 test_cascade.py                     # 129 tests: comfyless Stable Cascade dispatch (ADR-010)
python3 test_machine_boundary_validator.py  # 130 tests: machine-boundary validator (ADR-012)
python3 test_iterate.py                     #  92 tests: comfyless --iterate (ADR-008)
python3 test_samplers.py                    #  41 tests: custom schedulers / sampler swap
python3 test_server_robustness.py           # 105 tests: comfyless IPC timeouts + BrokenPipe survival + device-keyed socket routing + server-side device pinning + atomic output reservation (ADR-020) + daemon quant carriage (ADR-019 slice DQ: validation, cache-key discrimination, quant forwarding, H-1 symlink refusal) + multi-root _check_paths union (ADR-018)
python3 test_mcp_server.py                  # 597 tests: comfyless MCP server (ADR-011 slice 1 + ADR-015 slice 2 catalog/list_models/list_loras + slice 2b list_transformers + slice 3 generate catalog-name migration + slice 3b cascade catalog-name migration + ADR-018 multi-root kind-typed scan + ADR-022 S5 catalog search/family filters)
python3 test_quant.py                       # 101 tests: fp8 quantize-on-load (ADR-019 slice A) — eligibility policy, cache-key discrimination, DMR dispatcher routing, boundary hygiene
python3 test_fp8_single_file.py             #  85 tests: ComfyUI scaled-fp8 single-file loader + DMR merge (ADR-019 slices C/C-d/DMR) — classifier variants, security-review negatives, ScaledFp8Linear numerics, dequant->merge->requant dispatcher
python3 test_lora_order_insensitive.py      #  26 tests: direct-merge LoRAs order-insensitive to PEFT wrapping + LoKR->LoRA flatten (LoKR-on-Z-Image rescue: reconstruction, wiring, alpha-sentinel guard)
python3 test_vae_override_class.py          #  10 tests: --vae override honors the checkpoint's own VAE class (cherry-picked from krea-testing ad6689e)
python3 test_lora_audit.py                  # 197 tests: scripts/lora_audit.py classify / manifest / dry-load / convert / delete (ADR-014 S1–S4) + transformer audit (ADR-021: prognosis mapping, shape match, sampled dedupe, root disjointness, report-only)
python3 test_lora_convert_krea.py           #  24 tests: Krea-2 LoRA format-conversion plan (krea_native → diffusers_krea)
python3 test_catalog_db.py                  # 125 tests: catalog DB metadata plane (ADR-022 S1-S5) — schema, FUSE guard, sanitizer, upsert/stale semantics, manifest kind-branch join, families/sidecar/exclusion/search, civitai enrichment (mocked network), load-plane independence
```
All sixteen suites run against the comfyless uv-managed `.venv` — invoke via `./.venv/bin/python3` (created by `uv sync` at the repo root; see ADR-013 for the dep-divergence rule). Total 2182 unit tests; expect 0 failures.

`test_flux2.py` is a live GPU smoke test that performs an actual Flux.2 generation — separate from the unit suites above. Run only when you need to verify end-to-end Flux.2 behavior.

## Git commit conventions for this repo

See the general `Git Commit Discipline` rule in `~/.claude/CLAUDE.md` for the cadence and staging rules that apply to every Claude session. The additions below are the repo-specific conventions layered on top.

**Commit message style** — matches the existing history shown by `git log --oneline`:

- Prefix: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `update:`, `tool:` for standalone CLI tools, `workflows:` for workflow JSON artifacts
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

§12 is broader than §5 and this project already trips it on three surfaces:

| Surface | File | Trigger |
|---------|------|---------|
| Unix socket IPC server | `comfyless/server.py` | IPC (Unix sockets) |
| HF repo ID resolution + download | `nodes/eric_diffusion_utils.py` `resolve_hf_path` | Loading model weights from caller-supplied paths |
| `--json` stdin/stdout bridge | `comfyless/generate.py` `_run_json_mode` | Machine-facing interface; future LLM agent tool surface |
| Scaled-fp8 file-content parser (ADR-019 slice C) | `nodes/eric_diffusion_fp8_ops.py` + slice-C detection/remap in `eric_diffusion_utils.py` | Custom parsing of caller-supplied weight-file CONTENT (header key patterns, scale tensors) fed into compute ops — see `docs/security/review-slice-C-fp8-single-file-2026-07-02.md` F4 |

**Debt:** No ADR or security review exists for `comfyless/server.py` (IPC) or `resolve_hf_path` (caller-supplied model loading). These should have had §12 reviews before the code landed. Backlogged — when either surface is next modified, write the missing review before touching the code.

**Surfaces that become Red Zone on scope change:**

- **`--json` bridge + LLM agent wiring** (Backlog) — once model output drives paths or parameters into `generate()`, this becomes a Red Zone surface: prompt injection, path traversal, actor identity. Treat any commit that wires this as Red Zone from day one, not after.
- **HTTP transport** — if `--serve` ever grows a network interface, that commit is Red Zone regardless of other scope.
- **Batch generation from external input** — file writes at scale from caller-supplied lists is a §12 trigger.

**Review rules:**

- **Every non-trivial code slice runs `code-reviewer` (Opus) before commit.** "Trivial" = single-line fix, pure doc edit, mechanical rename with no behavior change.
- **Any change to `comfyless/server.py`, `resolve_hf_path`, or `_run_json_mode` also runs `security-auditor` (Opus).** Output saved to `docs/security/review-<slug>-<YYYY-MM-DD>.md` and referenced in the commit body.
- **When the `--json` / LLM agent wiring lands:** write spec + ADR before code, run `security-auditor`, treat as Red Zone from the first commit.
- Trivial skip ask: `"Trivial — skip review? Change: <one-line summary>. Reply 'review' to run it anyway."` Do not self-decide.
- Pass `model: "opus"` explicitly at every Agent-tool invocation for reviewer agents (`code-reviewer`, `security-auditor`). The frontmatter pin is known-broken in Claude Code 2.1.117 — structural enforcement requires the invocation-time override.

## Commit-time hooks

`.claude/settings.json` installs a `PreToolUse` hook on `Bash` that rejects `git commit -m "..."` calls whose message lacks an `AI-disclosure:` trailer. Required per global §7 and §0 rule 6.

- Bypassable by editor commit (no `-m` flag).
- For human-only commits: `AI-disclosure: none`.
- Hook script: `.claude/hooks/check-ai-disclosure.sh` (committed; travels with repo).

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

`comfyless/integrations/openwebui/generate_image_tool.py` is a native OpenWebUI Tool (runs inside the OWUI container) that drives image generation from chat and renders results inline. It calls the comfyless MCP server through the **mcpo** OpenAPI bridge — launched via `start-mcpo.sh` at the repo root (model-base = `hf-local`, the curated set; scanning the parent `.../models` also surfaces HF-cache snapshot-hash names). Tools exposed to the model: `generate_image`, `list_models`, `list_loras`, `list_transformers` (catalog names only, no paths). Requires a tool-calling model — gpt-oss works; roleplay-finetuned models (e.g. Dolphin-Venice) do not reliably emit tool calls. See ADR-017 and `comfyless/integrations/openwebui/README.md`.

The MCP server (`comfyless/mcp_server.py`) caches one pipeline in-process and evicts + frees it on config change (mirrors the `server.py` daemon) so a long-lived server doesn't OOM across model switches; LoRAs are applied via the shared `generate._apply_loras`. See `docs/security/review-mcp-pipeline-cache-2026-06-27.md`.

## Important Constraints

- All model loading uses `local_files_only=True` — no internet access during inference.
- Dimension alignment is 32px throughout; violating this causes transformer shape errors.
- `pipeline.vae.enable_tiling()` is always called on generation pipelines — required for >2 MP decode without OOM.
- The Edit pipeline takes a `Qwen2VLProcessor` (vision-language processor); the Generation pipeline does **not** — it uses only a tokenizer

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **Eric_Qwen_Edit_Experiments** (4350 symbols, 7840 relationships, 293 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> Index stale? Run `node .gitnexus/run.cjs analyze` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? `npx gitnexus analyze` (npm 11 crash → `npm i -g gitnexus`; #1939).

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows. For regression review, compare against the default branch: `detect_changes({scope: "compare", base_ref: "main"})`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `query({search_query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method without first running `impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit changes without running `detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/Eric_Qwen_Edit_Experiments/context` | Codebase overview, check index freshness |
| `gitnexus://repo/Eric_Qwen_Edit_Experiments/clusters` | All functional areas |
| `gitnexus://repo/Eric_Qwen_Edit_Experiments/processes` | All execution flows |
| `gitnexus://repo/Eric_Qwen_Edit_Experiments/process/{name}` | Step-by-step execution trace |

## Cross-Repo Groups

This repository is listed under GitNexus **group(s): ai-lab** (see `~/.gitnexus/groups/`). For cross-repo analysis, use MCP tools `impact`, `query`, and `context` with `repo` set to `@<groupName>` or `@<groupName>/<memberPath>` (paths match keys in that group’s `group.yaml`). Use `group_list` / `group_sync` for membership and sync. From the project root: `node .gitnexus/run.cjs group list`, `node .gitnexus/run.cjs group sync <name>`, `node .gitnexus/run.cjs group impact <name> --target <symbol> --repo <group-path>` (the `.gitnexus/run.cjs` path is repo-root-relative).

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
