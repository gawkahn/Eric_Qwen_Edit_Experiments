# Documentation — which repository each document belongs to

ADR-045 slice 7 (2026-09-09) copied every document in `docs/` to the sibling
repository `comfyless_diffusion` wholesale rather than splitting them, because
the Red Zone commit gate refuses a commit whose cited ADR or security review is
not present. The copy was always meant to be interim. This file is the
classification that decides what each side keeps.

**Nothing has been deleted yet.** This table is the judgement; the deletion is a
separate mechanical pass, deliberately kept apart from it — a wrongly-deleted
ADR is invisible afterwards, while a duplicated one is merely untidy.

## The rule: reachability, not residence

A document is `BOTH` when the code it governs is in, or is directly reached
through, one of the nine `comfyless.core` modules the node pack actually
imports:

    comfyless.core.eric_diffusion_utils           (11 import sites in nodes/)
    comfyless.core.eric_diffusion_manual_loop     (5)
    comfyless.core.eric_diffusion_fp8_ops         (5)
    comfyless.core.eric_lora_format_convert_apply (3)
    comfyless.core.eric_diffusion_samplers        (3)
    comfyless.core.hunyuan_chain                  (2)
    comfyless.core.family_defaults                (1)
    comfyless.core.eric_lora_format_convert       (1)
    comfyless.core.eric_diffusion_lora_check      (1)

Living under `comfyless/core/` is NOT sufficient, and this is the correction
that matters most here. An earlier pass (2026-09-13, counts recorded in
TECH_DEBT) treated core-namespace membership as the test and put Krea2 identity
editing and the NAG subsystem in `BOTH` on that basis. They are not reachable
from the node pack: `grep -rn "krea2_identity\|nag_\|identity_edit" nodes/
pipelines/` returns nothing. Both now classify CORE-ONLY, which is the whole of
the difference between that pass's counts and this one's.

Judge by what a document constrains, not by how often its prose says
"comfyless". Several CFG-routing documents are CORE-ONLY because the applier
they describe has no node-side caller, and several comfyless-sounding
quantization documents are BOTH because `eric_diffusion_fp8_ops` is imported by
five node files.

## Buckets

| Bucket | Count | This repository |
|---|---|---|
| CORE-ONLY | 144 | **140 deleted 2026-09-15**; 4 retained by ruling (below) — governs only `src/comfyless/**` in the sibling |
| BOTH | 33 | **keep in both repos, permanently** |
| NODE-ONLY | 1 | keep — governs only the ComfyUI node pack |
| PROCESS | 17 | keep — governs the repository or the way work is done |
| **Total** | **195** | |

`BOTH` staying duplicated is a decision, not an oversight. Those documents
genuinely govern two codebases because the `eric_diffusion_*` node track imports
`comfyless.core` instead of reimplementing it.

`NODE-ONLY` has exactly one member, and the reason is historical: every §12 Red
Zone surface this project ever had was comfyless-side or in the shared
quant/LoRA core, and `docs/vision/` was written entirely during the comfyless
era. No security review in this repository takes a ComfyUI node file,
`pipelines/`, or workflow JSON as its subject.

## Ambiguity that does not change the outcome

Only the CORE-ONLY boundary is consequential, because CORE-ONLY is the only
bucket this repository deletes. A document argued between BOTH, NODE-ONLY and
PROCESS is kept either way — so do not spend review effort there.


## Retained despite CORE-ONLY (4) — ruled 2026-09-15

The reachability rule classifies these four CORE-ONLY and that classification
stands; the decision to keep them anyway is a separate judgement, recorded here
so the two are not confused. They were the four calls flagged as close when the
table was first written.

- `docs/decisions/ADR-030-comfyless-2x-upscale-vae-decode.md` — the decision is
  only the CLI flag, but the decoder helper originated in, and still lives at,
  `nodes/eric_qwen_upscale_vae.py`. Node-facing guidance for a technique
  validated as a default recommendation.
- `docs/decisions/ADR-046-comfyless-owned-lora-adapters.md` — the subsystem is
  comfyless-owned and the ADR explicitly leaves `nodes/eric_qwen_edit_lora.py`
  untouched, but this repository keeps the differential test
  `test_lora_adapters.py` that exists because of it.
- `docs/security/review-adr-009-cfg-aliasing-2026-07-24.md` and
  `docs/security/review-parity-slice1-shared-defaults-2026-07-25.md` — both
  review the applier inside `comfyless/core/family_defaults.py`. The module is
  node-imported, but `apply_family_defaults` has no node-side caller: nodes read
  only the `FAMILY_DEFAULTS` dict (`nodes/eric_diffusion_generate.py:428`).
  Strict reachability says delete; retained because the cost of being wrong is
  asymmetric.

## Where the deleted documents live

Every one of the 140 was verified present in `comfyless_diffusion` before
deletion — nothing was lost, only de-duplicated. References to them survive in
this repository's `TECH_DEBT.md`, `CLAUDE.md`, `implementation_details.md` and
`systemd/comfyless@.service`; those paths now resolve in the sibling repository,
not here. `TECH_DEBT.md` is append-only, so its historical citations are
deliberately left as written.

The Red Zone commit gate is unaffected. Since 2026-09-15 it resolves a cited
ADR or review against the tree of the commit being checked rather than the
current working tree, so every commit that cited one of these docs when it was
made still passes `check-range` — verified at 43 Red Zone blocks over all of
history, unchanged by this prune. That fix is what made the prune possible at
all; see `TECH_DEBT.md`.

## Two tracked documents are outside the 195

`git ls-files docs/` returns 197, not 195. The classification pass missed two:

- `docs/README.md` — this file, written by the pass itself.
- `docs/comfyless-stable-cascade.md` — user-facing comfyless CLI documentation,
  CORE-ONLY by subject but with **no copy in the sibling repository**, so
  deleting it would have destroyed it rather than de-duplicated it. Held out of
  the prune and moved to `comfyless_diffusion` in its own slice (`14b2edf`
  there), with its paths corrected for that repository's src layout; removed
  here afterwards. `git ls-files docs/` now returns 56.

## BOTH (33) — Keep — governs both codebases

- `docs/decisions/ADR-002-three-tier-lora-fallback.md` — The three-tier LoRA cascade + key translation governs nodes/eric_qwen_edit_lora.py and the comfyless-side loader equally; both halves still implement it.
- `docs/decisions/ADR-003-gen-pipeline-cfg-routing.md` — Defines the GEN_PIPELINE dict consumed by nodes/eric_diffusion_loader.py and the family-keyed _build_call_kwargs CFG table duplicated in the node generate node and comfyless/generate.py.
- `docs/decisions/ADR-004-true-cfg-vs-guidance-embeds.md` — Routes Qwen-Image through true_cfg_scale via read_guidance_embeds/_build_call_kwargs, which both nodes/eric_diffusion_generate.py and comfyless/generate.py execute.
- `docs/decisions/ADR-005-sampler-multistep-only.md` — Constrains the AB2/AB3 scheduler subclasses in comfyless.core.eric_diffusion_samplers, which nodes/eric_diffusion_{generate,multistage,ultragen}.py import via sampler_choices/swap_sampler.
- `docs/decisions/ADR-009-per-family-default-params.md` — FAMILY_DEFAULTS now lives in comfyless/core/family_defaults.py and is read by nodes/eric_diffusion_generate.py (Hunyuan refiner operating point) as well as the CLI precedence ladder.
- `docs/decisions/ADR-016-hunyuan-image-base-refiner-chain.md` — comfyless/core/hunyuan_chain.py load_refiner_pipeline/run_chain is imported by nodes/eric_diffusion_generate.py, and the ADR wires the ComfyUI Generate node's refiner_path input alongside the CLI flag.
- `docs/decisions/ADR-019-native-quantization-support.md` — Quantize-on-load and single-file quant consumption extend nodes/eric_diffusion_loader.py and the shared fp8 ops (comfyless.core.eric_diffusion_fp8_ops, imported by nodes/eric_qwen_edit_lora.py) as well as the CLI/MCP surface.
- `docs/decisions/ADR-025-hunyuan-image-cfg-routing.md` — Adds the third CFG routing shape (distilled_guidance_scale) and hunyuan family patterns to _build_call_kwargs in BOTH nodes/eric_diffusion_generate.py and comfyless/generate.py, plus the node's vae_tiling input.
- `docs/decisions/ADR-028-comfyless-sigma-schedule.md` — Wires --schedule into comfyless generate AND extends the shared build_sigma_schedule/SCHEDULE_NAMES engine (now comfyless/core/sigma_schedules.py) that the UltraGen/multistage nodes' sigma_schedule inputs run on.
- `docs/decisions/ADR-029-res-exponential-multistep-samplers.md` — Adds res_2m/res_3m scheduler subclasses to the shared sampler module (comfyless.core.eric_diffusion_samplers) that nodes/eric_diffusion_*.py expose through sampler_choices().
- `docs/security/review-hunyuan-refiner-quant-2026-07-12.md` — Reviewed the fp8 quant block inside hunyuan_chain.load_refiner_pipeline (shared core imported by nodes/eric_diffusion_generate.py) as well as the server.py threading.
- `docs/security/review-hunyuan-refiner-reapply-2026-07-11.md` — Reviewed the new hunyuan_chain.py refiner module (shared core reached from the node pack) plus its daemon wiring in server.py/params_validation.py.
- `docs/security/review-i8-perchannel-scales-2026-07-10.md` — Reviewed the int8-tensorwise weight_scale gate in eric_diffusion_fp8_ops.py, the shared weight-file parser both the node loaders and comfyless reach.
- `docs/security/review-krea2-comfy-convert-2026-07-07.md` — Reviewed eric_krea2_convert.py plus the Krea branch of eric_diffusion_fp8_ops.load_scaled_fp8_component and eric_diffusion_utils.py — all shared loader core.
- `docs/security/review-refine-family-defaults-2026-07-18.md` — Reviewed refine's family-defaults overlay together with the announce-gating change to infer_model_family in the shared eric_diffusion_utils.py.
- `docs/security/review-resolve-hf-path-2026-04-23.md` — Reviewed resolve_hf_path in eric_diffusion_utils.py side by side with every node loader that calls it and the comfyless generate/server callers.
- `docs/security/review-resolve-hf-path-hardening-2026-04-23.md` — Reviewed the resolve_hf_path hardening in eric_diffusion_utils.py, the shared model-path resolver used by node loaders and comfyless alike.
- `docs/security/review-slice-3c-lora-adapters-2026-08-22.md` — Line-by-line differential of comfyless/core/lora_adapters.py against the node-pack original nodes/eric_qwen_edit_lora.py — both halves examined.
- `docs/security/review-slice-C-fp8-single-file-2026-07-02.md` — Design review of the scaled-fp8 single-file parser and ScaledFp8Linear in eric_diffusion_fp8_ops.py, reached from the node loaders and comfyless.
- `docs/security/review-slice-Cd-comfy-quant-2026-07-02.md` — Reviewed the .comfy_quant descriptor parse/allowlist in the shared eric_diffusion_fp8_ops.py classifier.
- `docs/security/review-slice-DMR-quantized-merge-2026-07-03.md` — Reviewed apply_merge_delta dequant-merge-requant and the backup ledger in shared fp8_ops — the LoRA direct-merge path both surfaces use.
- `docs/security/review-slice-I8-int8-tensorwise-2026-07-08.md` — Reviewed the int8-tensorwise (ci-w) extension of _classify_cq/load_scaled_fp8_component in the shared fp8 loader.
- `docs/security/review-slice-NF4-bnb-single-file-2026-07-17.md` — Design review of bnb-NF4 single-file consumption in eric_diffusion_fp8_ops.py/eric_krea2_convert.py/eric_diffusion_utils.py, all shared loader core.
- `docs/security/review-slice-NV-nvfp4-merge-guard-2026-07-16.md` — Reviewed the _requant_config_matching_base class gate in shared fp8_ops and traced all four merge call sites including nodes/eric_qwen_edit_lora.py.
- `docs/security/review-slice-PQ-partial-quant-2026-07-07.md` — Reviewed partial-quant / naked-fp8 coexistence in the shared eric_diffusion_fp8_ops.py loader.
- `docs/security/review-slice-R1R2R3-dequant-nonweight-2026-07-07.md` — Reviewed non-weight fp8 upcast and dequant-to-bf16 load mode in shared fp8_ops plus the generate.py quant wiring.
- `docs/vision/epic-hunyuan-2-1-plus-enhancer.md` — Builds comfyless.core.hunyuan_chain (imported by nodes/eric_diffusion_generate.py) plus the hunyuan-image family CFG routing and refiner_path gate present in both the node generate path and the CLI/daemon.
- `docs/vision/slice-A-fp8-quant-load.md` — ADR-019 fp8 quantize-on-load: edit scope is eric_diffusion_utils.py + eric_diffusion_loader.py (loader-node quant dropdown) as well as the comfyless CLI/MCP `--quant` flag and the shared LoRA tier dispatch.
- `docs/vision/slice-C-comfy-fp8-single-file.md` — Builds eric_diffusion_fp8_ops.py (ScaledFp8Linear) and the single-file detection path in eric_diffusion_utils.py — both now comfyless.core modules imported by nodes/eric_diffusion_loader.py; test_fp8_single_file.py still lives in the node repo.
- `docs/vision/slice-DMR-quantized-lora-merge.md` — dequant→merge→requant dispatcher in eric_diffusion_fp8_ops.py plus the four merge sites including nodes/eric_qwen_edit_lora.py and eric_lora_format_convert_apply.py — shared LoRA direct-merge code reached from both surfaces.
- `docs/vision/slice-NF4-bnb-single-file.md` — Adds bnb-NF4 single-file dequant to eric_diffusion_utils.py + eric_diffusion_fp8_ops.py, the shared loader modules nodes/eric_diffusion_loader.py imports.
- `docs/vision/slice-NV-nvfp4-quant-load.md` — nvfp4 quantize-on-load in eric_diffusion_utils.py/eric_diffusion_fp8_ops.py plus the refuse_unmergeable_base entry gate wired into nodes/eric_qwen_edit_lora.py and eric_lora_format_convert_apply.py.
- `docs/vision/slice-krea2-support.md` — Adds Krea2Pipeline family detection to the shared detect_pipeline_class/infer_model_family in eric_diffusion_utils.py and rows to comfyless.core.family_defaults, both of which nodes/eric_diffusion_loader.py imports, plus the CFG routing branch.


## NODE-ONLY (1) — Keep — governs the node pack only

- `docs/decisions/ADR-007-strip-eric-prefix.md` — Renames nodes/ file names, node class definitions, NODE_CLASS_MAPPINGS keys and NODE_DISPLAY_NAME_MAPPINGS labels plus the node-pack package name.


## PROCESS (17) — Keep — governs the repository or the process

- `docs/decisions/ADR-013-comfyless-torch-divergence.md` — Dependency-pinning policy: comfyless's uv .venv/pyproject pin set may diverge from the pip requirements.txt ComfyUI Manager path, and names the test runner.
- `docs/decisions/ADR-031-license-policy.md` — Dependency license allowlist for the supply-chain gate (scripts/check_licenses.py) — repository policy, not either code surface.
- `docs/decisions/ADR-032-static-type-checking-pyright.md` — Pins pyright, sets the ratchet baseline posture and the CI types gate for the repo.
- `docs/decisions/ADR-042-per-root-typecheck-baselines.md` — Replaces the single pyright ratchet integer with per-root baselines in .claude/typecheck-baseline and the commit hook.
- `docs/decisions/ADR-045-comfyless-diffusion-standalone-repo.md` — The repository split itself — where code lives, attribution/licensing, and the node-vs-core boundary criterion.
- `docs/runbooks/ubuntu-26.04-upgrade.md` — Host OS upgrade runbook (driver/CUDA/mise/filesystem preflight) — machine infrastructure, not either code surface.
- `docs/security/review-redzone-gate-repair-2026-09-09.md` — Reviewed the repo's Red Zone commit-gate script (_red-zone-paths.sh patterns, list_red_zone_paths, smoke tests), not product code.
- `docs/security/review-slice-0-mcp-dep-2026-04-30.md` — Supply-chain review of the mcp==1.27.0 pin across pyproject.toml, requirements.txt and uv.lock — dependency policy only.
- `docs/security/review-slice-0c-2026-05-16.md` — Design-time supply-chain review of ADR-013's torch/CUDA dependency-manifest divergence.
- `docs/security/review-slice-3d-attribution-clearing-2026-09-09.md` — Reviewed stripping copyright/attribution headers across 24 files — a licensing/comment-only change with no behavior.
- `docs/security/review-slice-4-layering-inversion-2026-08-22.md` — Reviewed the ADR-045 git mv of hunyuan_chain.py and family_defaults.py into comfyless/core plus import rewrites — a repo-layering slice.
- `docs/security/review-slice-5-src-layout-2026-09-09.md` — Reviewed the ADR-045 src/ layout, hatchling build-system and console-script packaging move.
- `docs/security/review-slice-7-node-cutover-2026-09-09.md` — Reviewed the ADR-045 node-pack cutover commit and the missing gate wiring in the receiving repository.
- `docs/security/review-slice8-infra-cutover-2026-09-09.md` — Infra audit of the ADR-045 slice 8 cutover of systemd units and start-mcpo.sh to the sibling repo's venv.
- `docs/security/review-systemd-daemon-unit-2026-07-17.md` — Infra audit of the systemd/comfyless@.service user unit and its sandboxing directives — deployment config, no code.
- `docs/vision/comfyless-diffusion-extraction.md` — The ADR-045 repo-split plan itself — which modules move to comfyless_diffusion, git/CI/packaging mechanics, not any inference behavior.
- `docs/vision/slice-0c-cuda-torch-realignment.md` — Bumps torch/diffusers/transformers pins across pyproject.toml, requirements.txt and uv.lock — dependency-pinning policy, no code behavior change.


## CORE-ONLY (144) — Deleted from this repository 2026-09-15 — governs the sibling only

140 of the 144 below were removed in one mechanical pass. The four listed
under "Retained despite CORE-ONLY" above are still present here. The full
list is kept as the record of what was deleted and on what reasoning.

- `docs/decisions/ADR-001-daemon-socket-security.md` — Socket location, path allowlisting and request schema for the comfyless Unix-socket daemon (src/comfyless/server.py), which has no node-pack surface.
- `docs/decisions/ADR-006-comfyless-dual-mode-json-bridge.md` — Dual-mode CLI/--json stdin-stdout contract, sidecar replay and --params/--override precedence in comfyless/generate.py only.
- `docs/decisions/ADR-008-comfyless-iterate.md` — Adds the repeatable --iterate CLI flag and Cartesian sweep machinery to comfyless.generate/iterate.py.
- `docs/decisions/ADR-010-stable-cascade-json-config.md` — Special-case `comfyless --model stablecascade <config.json>` dispatch fork and comfyless/cascade.py; no node class exists for Cascade.
- `docs/decisions/ADR-011-comfyless-mcp-server.md` — Establishes src/comfyless/mcp_server.py as the LLM-agent tool surface, superseding the --json bridge for that role.
- `docs/decisions/ADR-012-machine-boundary-validator.md` — One canonical validator (comfyless/params_validation.py) shared by server.py, mcp_server.py and iterate.py at the machine boundary.
- `docs/decisions/ADR-014-lora-audit-tool.md` — Specifies the scripts/lora_audit.py CLI (audit roots, --convert/--delete, manifest contract), which now lives in the sibling repo.
- `docs/decisions/ADR-015-mcp-catalog-reference-resolution.md` — Catalog-name-only reference contract and build/collision rules for comfyless/catalog.py + the MCP request boundary.
- `docs/decisions/ADR-017-mcp-image-return-owui-integration.md` — Adds return_image/max_return_px/max_return_bytes to the MCP generate response and the OpenWebUI native tool that consumes it.
- `docs/decisions/ADR-018-multi-root-catalog-scan.md` — Adds --lora-path/--transformer-path scan roots and widens the daemon/MCP _check_paths load allowlist in comfyless/{catalog,server,mcp_server}.py.
- `docs/decisions/ADR-020-parallel-daemon-per-gpu.md` — Device-keyed socket naming and one daemon process per GPU for comfyless/server.py.
- `docs/decisions/ADR-021-transformer-audit.md` — Extends the scripts/lora_audit.py CLI with --transformer-root classification for the catalog service.
- `docs/decisions/ADR-022-catalog-service.md` — SQLite metadata plane (comfyless/catalog_db.py, catalog_builder.py, catalog_cli.py) plus its MCP search/list wiring.
- `docs/decisions/ADR-023-nag-negative-guidance.md` — NAG attention processors for krea (comfyless/core/pipelines/nag_krea2.py) wired to CLI --nag-* params; the ADR scopes v1 to "comfyless path only (ComfyUI nodes later)" and no node exposes nag_scale.
- `docs/decisions/ADR-024-nag-family-expansion.md` — Extends the same comfyless-only NAG ports to flux/flux2/klein/zimage (comfyless/core/pipelines/nag_*.py); a ComfyUI node surface for NAG is explicitly deferred.
- `docs/decisions/ADR-026-comfyless-prompt-enhancement.md` — New comfyless/enhance.py subsystem; the ADR notes the existing node rewriter is node-only and is not the thing being built.
- `docs/decisions/ADR-027-comfyless-refinement-loop.md` — The generate-judge-plan loop in comfyless/refine.py, driven by the CLI, catalog DB and daemon cache.
- `docs/decisions/ADR-030-comfyless-2x-upscale-vae-decode.md` — Adds the --upscale-vae flag, family gating and sidecar fields to comfyless generate; it reuses the already-shipped node decode helper unchanged.
- `docs/decisions/ADR-033-comfyless-video-generation.md` — Keyframe-anchored segment chaining and stitching in comfyless/video.py with its own CLI/plan surface.
- `docs/decisions/ADR-034-comfyless-output-format.md` — --output-format jpeg and the centralized comfyless/output_format.py save/metadata path across generate.py, server.py, mcp_server.py, cascade.py, refine.py.
- `docs/decisions/ADR-035-comfyless-reference-image-surface.md` — Defines the comfyless --ref-image PATH[:MODE] schema, ingestion caps and daemon ref-root containment; the node pack's edit nodes are untouched.
- `docs/decisions/ADR-036-flux2-klein-reference-conditioning.md` — Generalizes _resolve_ref_family_support in comfyless/generate.py so flux2/flux2klein refs thread through the generic call path.
- `docs/decisions/ADR-037-refine-loop-v2-trajectory-edit.md` — Climb-from-best, trajectory history and edit-mode refinement in comfyless/refine.py.
- `docs/decisions/ADR-038-refine-multi-reference-edit.md` — Operator-pinned multi-reference carriage and identity judging in comfyless/refine.py's edit loop.
- `docs/decisions/ADR-039-refine-v3-promotion-gate.md` — Pairwise promotion gate and typed plateau escape in the comfyless/refine.py judge/planner.
- `docs/decisions/ADR-040-loop-output-inside-daemon-roots.md` — Requires the refine loop's --output-dir to sit inside the daemon's ref roots; governs comfyless/refine.py plus server.py's _build_ref_roots interaction.
- `docs/decisions/ADR-041-catalog-semantic-offers.md` — Concept enrichment and query rewrite for the catalog FTS search feeding refine.search_loras (comfyless/catalog_enrich_concepts.py, catalog_db.py, refine.py).
- `docs/decisions/ADR-043-krea2-identity-edit.md` — core/pipelines module is unreachable from nodes/ — verified no import of krea2_identity_edit or nag_* in nodes/ or pipelines/ (rule: BOTH requires reachability through one of the 9 imported comfyless.core modules, not residence in the core namespace)
- `docs/decisions/ADR-044-krea2-identity-daemon-carriage.md` — Carries identity/ref_boost/grounding_px over the daemon wire and the MCP schema (comfyless/server.py, mcp_server.py, generate.py delegation gate).
- `docs/decisions/ADR-046-comfyless-owned-lora-adapters.md` — Creates comfyless/core/lora_adapters.py as comfyless's own LoRA subsystem and explicitly leaves nodes/eric_qwen_edit_lora.py untouched as the node repo's own module.
- `docs/security/review-2026-04-21-daemon-socket.md` — Design review of the comfyless Unix-domain-socket client/server daemon (socket auth, output-path containment) — now src/comfyless/server.py.
- `docs/security/review-adr-009-cfg-aliasing-2026-07-24.md` — Reviewed the cfg_scale/true_cfg_scale aliasing fix in generate._apply_family_defaults and refine._overlay_family_defaults — CLI/refine appliers the node pack never calls.
- `docs/security/review-adr-015-catalog-reference-2026-05-22.md` — Design review of the MCP opaque-catalog-name reference contract (comfyless/mcp_server.py, catalog resolution) replacing caller-supplied paths.
- `docs/security/review-adr-034-slice2-daemon-output-format-2026-07-21.md` — Reviewed server.py resolving OutputFormat server-side for the O_EXCL reservation and savepath template.
- `docs/security/review-adr-034-slice5-refine-output-format-2026-07-21.md` — Reviewed output-format/quality threading through comfyless/refine.py and the daemon wire builder.
- `docs/security/review-adr-035-reference-image-surface-2026-07-20.md` — Design review of the comfyless --ref-image surface: CLI flag, ref_images wire key, daemon _check_paths containment.
- `docs/security/review-adr-035-rereview-2026-07-20.md` — Re-review of the amended ADR-035 design against server.py root plumbing, generate.py delegation and video.py plan-mode workers.
- `docs/security/review-adr-035-slice1b-mcp-ref-images-leak-2026-07-21.md` — Reviewed the ref_images path leak in mcp_server.py extract_params.
- `docs/security/review-adr-035-slice2-ref-image-ingestion-2026-07-21.md` — Reviewed the new comfyless/ref_image.py ingestion helper (byte/pixel caps, decode).
- `docs/security/review-adr-035-slice4-daemon-2026-07-21.md` — Reviewed the daemon reference-image path across comfyless/{server,generate,params_validation,ref_image}.py.
- `docs/security/review-adr-035-slice4b-delegation-2026-07-21.md` — Reviewed the CLI-to-daemon ref delegation seam and RefPathError fallback in comfyless/{server,generate}.py.
- `docs/security/review-adr-035-slice5-replay-trust-2026-07-22.md` — Reviewed sidecar/PNG-metadata replay trust for ref images across refine.py, ref_image.py and generate.py.
- `docs/security/review-adr-037-critique-offers-2026-07-25.md` — Reviewed critique-driven LoRA offers in the comfyless/refine.py judge loop.
- `docs/security/review-adr-037-d2-amendment-2026-07-24.md` — Reviewed refine.py tie-promotion and no-op seed resample plus the judge_recipes TOML rubric.
- `docs/security/review-adr-037-d3-until-score-2026-07-24.md` — Reviewed the --until-score composite gate in comfyless/refine.py.
- `docs/security/review-adr-037-d5-anchor-2026-07-24.md` — Reviewed switching the edit-mode judge anchor to the operator's original seed in comfyless/refine.py.
- `docs/security/review-adr-037-design-2026-07-23.md` — Design review of refine loop v2 (trajectory + edit mode) in comfyless/refine.py.
- `docs/security/review-adr-037-lora-offers-2026-07-25.md` — Reviewed keyword LoRA offers and the plateau-reword rubric in comfyless/refine.py.
- `docs/security/review-adr-037-sliceA-implementation-2026-07-23.md` — Reviewed the refine.py trajectory-core implementation and its generic judge recipe.
- `docs/security/review-adr-037-sliceB-implementation-2026-07-23.md` — Reviewed the refine.py edit-mode refinement implementation and edit-generic recipe.
- `docs/security/review-adr-037-stagnation-escape-2026-07-24.md` — Reviewed the --explore-after stagnation seed escape in comfyless/refine.py.
- `docs/security/review-adr-038-design-2026-07-25.md` — Design review of multi-reference refinement in comfyless/refine.py.
- `docs/security/review-adr-038-impl-2026-07-25.md` — Reviewed the multi-reference refinement implementation in comfyless/refine.py.
- `docs/security/review-adr-039-d3-exhaustion-amendment-2026-07-27.md` — Reviewed the consecutive-failed-batch exhaustion rule in the refine.py promotion gate.
- `docs/security/review-adr-039-design-2026-07-25.md` — Design review of refine v3 banded promotion gate in comfyless/refine.py.
- `docs/security/review-adr-039-slice1-duel-primitive-2026-07-25.md` — Reviewed the duel primitive and duel-generic judge recipe in comfyless/refine.py.
- `docs/security/review-adr-039-slice2-promotion-gate-2026-07-25.md` — Reviewed the banded promotion gate in comfyless/refine.py.
- `docs/security/review-adr-039-slice3-seed-batch-2026-07-26.md` — Reviewed the sideways cap and seed-batch logic in comfyless/refine.py.
- `docs/security/review-adr-039-slice4-anchor-duel-2026-07-26.md` — Reviewed the plateau trigger and anchor-duel slice in comfyless/refine.py.
- `docs/security/review-adr-044-commit2-wire-carriage-2026-08-01.md` — Reviewed Krea-2 identity params crossing the daemon wire in params_validation.py/generate.py/server.py.
- `docs/security/review-adr-044-commit3-mcp-hardening-2026-08-01.md` — Reviewed the identity-param reject-at-entry hardening in comfyless/mcp_server.py.
- `docs/security/review-adr-044-identity-daemon-carriage-2026-08-01.md` — Design review of Krea-2 identity edit over the daemon wire and MCP tool surface (server.py, mcp_server.py).
- `docs/security/review-adr017-mcp-image-return-2026-06-25.md` — Reviewed the base64 image-return path in comfyless/mcp_server.py (_encode_return_image, audit-line hygiene).
- `docs/security/review-adr017-owui-tool-2026-06-26.md` — Reviewed comfyless/integrations/openwebui/generate_image_tool.py, the OWUI-side tool talking to the MCP server via mcpo.
- `docs/security/review-adr018-multi-root-scan-2026-07-05.md` — Reviewed multi-root catalog scanning in comfyless/catalog.py plus server.py/_check_paths and mcp_server.py spawn args.
- `docs/security/review-adr021-transformer-audit-2026-07-05.md` — Design review of transformer-kind manifest entries in the scripts/lora_audit.py catalog-audit CLI.
- `docs/security/review-adr030-upscale-vae-2026-07-14.md` — Reviewed the --upscale-vae wire fields at the daemon boundary and generate._load_upscale_vae subfolder containment.
- `docs/security/review-adr037-seed-pin-2026-07-28.md` — Reviewed pinning the edit seed in comfyless/refine.py (Red Zone judge/seed surface).
- `docs/security/review-adr040-revision-2026-07-27.md` — Design review of ADR-040 loop output inside daemon roots (server.py, refine.py, mcp_server.py, cascade.py).
- `docs/security/review-adr040-slice1-2026-07-27.md` — Reviewed the opt-in report_roots field on the daemon ping handler in server.py/params_validation.py.
- `docs/security/review-adr040-slice2a-2026-07-27.md` — Reviewed the run_id correlation primitive minted in generate.py, refine.py and cascade.py.
- `docs/security/review-adr040-slice2b-2026-07-27.md` — Reviewed the derived run directory and entry validation in refine.py/generate.py.
- `docs/security/review-adr040-slice3-2026-07-27.md` — Reviewed generate.refuse_out_of_roots_refs and its _run_cli_mode entry gate.
- `docs/security/review-adr041-slice1-2026-07-29.md` — Reviewed the instruction_template catalog column and refine.search_loras change.
- `docs/security/review-catalog-annotate-2026-07-06.md` — Reviewed the catalog DB write-back verbs (worklist/annotate/exclude) and URL sanitization in the catalog service.
- `docs/security/review-catalog-enrich-2026-07-06.md` — Reviewed comfyless/catalog_enrich.py civitai hash lookup and text sanitization.
- `docs/security/review-catalog-mcp-s5-2026-07-06.md` — Reviewed the MCP search tool, family filters and catalog_db.connect_readonly accessor.
- `docs/security/review-comfyless-mcp-server-2026-04-28.md` — Design review of the comfyless MCP server (ADR-011) over server.py/generate.py/cascade.py.
- `docs/security/review-comfyless-server-2026-04-23.md` — Reviewed comfyless/server.py, the Unix-socket IPC daemon implementation.
- `docs/security/review-comfyless-server-hardening-2026-04-23.md` — Reviewed the comfyless/server.py hardening slice closing the prior review's findings.
- `docs/security/review-enhancer-trust-remote-code-2026-07-11.md` — Reviewed comfyless/enhance.py (first trust_remote_code use) and its generate.py inline wiring.
- `docs/security/review-judge-recipe-2026-07-15.md` — Reviewed the judge-rubric-to-recipe-file change in comfyless/refine.py and judge_recipes/generic.toml.
- `docs/security/review-llm-max-tokens-cap-2026-07-20.md` — Reviewed the max_tokens cap on the enhance.py and refine.py LLM wires.
- `docs/security/review-lora-audit-2026-05-17.md` — Design review of ADR-014's scripts/lora_audit.py CLI auditor (convert/delete surfaces).
- `docs/security/review-lora-audit-s1-2026-05-25.md` — Reviewed scripts/lora_audit.py S1 (--scan-only / --print-manifest).
- `docs/security/review-lora-audit-s2-2026-05-28.md` — Reviewed scripts/lora_audit.py S2 (--dry-load mode).
- `docs/security/review-lora-audit-s3-2026-06-02.md` — Reviewed scripts/lora_audit.py S3 (--convert write path).
- `docs/security/review-lora-audit-s4-2026-06-27.md` — Reviewed scripts/lora_audit.py S4 (--delete path).
- `docs/security/review-mcp-path-leak-close-2026-08-01.md` — Reviewed closing the upscale_vae_path/refiner_path leak in comfyless/mcp_server.py.
- `docs/security/review-mcp-pipeline-cache-2026-06-27.md` — Reviewed the in-process pipeline cache and LoRA-apply fix in comfyless/mcp_server.py.
- `docs/security/review-mcp-server-catalog-typecheck-2026-07-27.md` — Reviewed the pyright-drawdown code changes in comfyless/mcp_server.py and catalog.py resolve_reference.
- `docs/security/review-parallel-daemon-2026-07-03.md` — Reviewed the one-daemon-per-GPU device-keyed socket design in comfyless/server.py.
- `docs/security/review-parity-slice1-shared-defaults-2026-07-25.md` — Reviewed unifying generate/refine family-default appliers into family_defaults.apply_family_defaults plus refine --schedule; only the CLI/refine callers use it.
- `docs/security/review-parity-slice2-wire-warnings-2026-07-25.md` — Reviewed generate.surface_wire_warnings and its CLI/refine daemon-branch call sites.
- `docs/security/review-pause-daemon-guard-2026-07-17.md` — Reviewed the daemon pause opt-out and client recv-error contract in server.py/generate.py/pause.py.
- `docs/security/review-rebalance-daemon-mcp-2026-06-27.md` — Reviewed threading the Krea rebalance knobs through comfyless/server.py and mcp_server.py wire surfaces.
- `docs/security/review-refinement-loop-2026-07-13.md` — Design review of the ADR-027 LLM-as-judge refinement loop (comfyless/refine.py).
- `docs/security/review-refinement-loop-slice3-2026-07-13.md` — Reviewed the refine.py loop controller slice.
- `docs/security/review-refinement-loop-slice4-2026-07-15.md` — Reviewed the refine.py seed-image entry slice.
- `docs/security/review-server-timeout-brokenpipe-2026-04-24.md` — Reviewed the timeout/BrokenPipe fix in the comfyless/server.py accept loop.
- `docs/security/review-server-validate-request-typecheck-2026-07-27.md` — Reviewed the pyright-driven fix to _validate_request in comfyless/server.py.
- `docs/security/review-slice-1-mcp-step1-2026-05-16.md` — Reviewed the ADR-011 MCP server skeleton in comfyless/mcp_server.py.
- `docs/security/review-slice-1-mcp-step2-2026-05-17.md` — Reviewed the MCP generate handler wiring in comfyless/mcp_server.py.
- `docs/security/review-slice-1-mcp-step3-2026-05-17.md` — Reviewed MCP cascade dispatch in mcp_server.py/cascade.py.
- `docs/security/review-slice-2-step1-2026-05-23.md` — Reviewed the new comfyless/catalog.py scan_model_family helper.
- `docs/security/review-slice-2-step2-2026-05-24.md` — Reviewed the comfyless/catalog.py data structure, scan walker, manifest parser and build_catalog.
- `docs/security/review-slice-2-step3-2026-05-25.md` — Reviewed the --catalog spawn-time wire-up in comfyless/mcp_server.py.
- `docs/security/review-slice-2-step4-2026-05-25.md` — Reviewed the list_models/list_loras MCP tools and catalog-name sanitization.
- `docs/security/review-slice-2b-list-transformers-2026-05-31.md` — Reviewed the list_transformers MCP tool in comfyless/mcp_server.py.
- `docs/security/review-slice-3-step1-2026-06-02.md` — Reviewed catalog.resolve_reference, the MCP name-to-path resolution point.
- `docs/security/review-slice-3-step2-2026-06-02.md` — Reviewed the _handle_generate catalog-name migration in mcp_server.py/server.py.
- `docs/security/review-slice-3b-cascade-catalog-2026-06-24.md` — Reviewed the MCP cascade generate catalog-name migration.
- `docs/security/review-slice-4-mcp-extract-params-2026-07-07.md` — Reviewed the extract_params core (non-cascade) in comfyless/mcp_server.py.
- `docs/security/review-slice-4-step3-enrichment-2026-07-09.md` — Reviewed extract_params catalog-DB enrichment in comfyless/mcp_server.py.
- `docs/security/review-slice-4a-catalog-db-autodiscover-2026-07-07.md` — Reviewed default catalog-DB auto-discovery in mcp_server.py/catalog_db.py.
- `docs/security/review-slice-4d-cascade-2026-07-09.md` — Reviewed flat Stable Cascade resolution in the MCP extract_params path.
- `docs/security/review-slice-DLW-daemon-lora-weights-2026-07-17.md` — Reviewed the daemon's weight-aware LoRA diff and apply_adapter_weights helper in server.py/generate.py.
- `docs/security/review-slice-DQ-daemon-quant-2026-07-03.md` — Design review of carrying quant/quant_skip/quant_only across the daemon wire (server.py, generate.py, params_validation.py).
- `docs/security/review-validator-slice-step1-2026-05-15.md` — Reviewed the ADR-012 request validator in comfyless/params_validation.py and its server.py call site.
- `docs/security/review-validator-slice-step3-2026-05-16.md` — Reviewed the ADR-012 validator step 3 wiring into comfyless/server.py.
- `docs/vision/epic-krea2-identity-edit.md` — core/pipelines module is unreachable from nodes/ — verified no import of krea2_identity_edit or nag_* in nodes/ or pipelines/ (rule: BOTH requires reachability through one of the 9 imported comfyless.core modules, not residence in the core namespace)
- `docs/vision/slice-1-mcp-generate.md` — Builds the MCP `generate` tool in comfyless/mcp_server.py over comfyless/generate.py; no node-pack surface.
- `docs/vision/slice-2-mcp-catalog.md` — Builds comfyless/catalog.py plus the list_models / list_loras MCP tools — the catalog plane, sibling-only.
- `docs/vision/slice-2-mcp-extract-params.md` — Adds the extract_params MCP tool reading comfyless generate JSON sidecars; MCP surface only.
- `docs/vision/slice-2b-mcp-list-transformers.md` — Adds the list_transformers MCP tool over the catalog; MCP surface only.
- `docs/vision/slice-3-mcp-generate-catalog.md` — Migrates comfyless `generate` + cascade to catalog-name reference resolution in comfyless/catalog.py and mcp_server.py.
- `docs/vision/slice-3b-mcp-cascade-catalog.md` — Migrates the MCP cascade handler to catalog-name resolution; MCP/catalog surface only.
- `docs/vision/slice-4-mcp-extract-params.md` — Second extract_params MCP tool slice with catalog enrichment; MCP surface only.
- `docs/vision/slice-DLW-daemon-lora-weights.md` — Fixes LoRA weight application in comfyless/server.py's daemon LoRA diff and the shared comfyless/generate.py _apply_loras; the node pack has its own eric_qwen_edit_lora.py path, untouched.
- `docs/vision/slice-DQ-daemon-quant.md` — Carries the quant triple over the daemon wire protocol and into comfyless/server.py's pipeline cache key; daemon/IPC only.
- `docs/vision/slice-NAG-krea2-negative-guidance.md` — core/pipelines module is unreachable from nodes/ — verified no import of krea2_identity_edit or nag_* in nodes/ or pipelines/ (rule: BOTH requires reachability through one of the 9 imported comfyless.core modules, not residence in the core namespace)
- `docs/vision/slice-NAG2-family-expansion.md` — core/pipelines module is unreachable from nodes/ — verified no import of krea2_identity_edit or nag_* in nodes/ or pipelines/ (rule: BOTH requires reachability through one of the 9 imported comfyless.core modules, not residence in the core namespace)
- `docs/vision/slice-catalog-service.md` — Designs the SQLite catalog at ~/.local/share/comfyless/catalog.sqlite and its build/enrich/search plane exposed via MCP; sibling-only metadata service.
- `docs/vision/slice-lora-audit.md` — Builds scripts/lora_audit.py (now in the sibling repo) as a standalone catalog-feeding CLI; explicitly out of scope: any change to the node LoRA modules, which it only imports.
- `docs/vision/slice-machine-boundary-validator.md` — Creates comfyless/params_validation.py as the single validator for the daemon socket and MCP request surfaces; machine-boundary code only.
- `docs/vision/slice-mcp-image-return-owui.md` — Adds optional base64 image return to the MCP generate response plus the OpenWebUI native tool in comfyless/integrations/openwebui/.
- `docs/vision/slice-parallel-daemon-per-gpu.md` — Device-keyed Unix sockets and one daemon instance per GPU in comfyless/server.py; daemon/IPC only.
- `docs/vision/slice-pause-sigint.md` — Adds comfyless/pause.py ^C pause/resume for foreground CLI generation; CLI signal handling only.
- `docs/vision/slice-ref-image-cli.md` — Adds the --ref-image CLI flag and comfyless/ref_image.py ingestion helper on the comfyless foreground path; the node pack has its own edit nodes.
- `docs/vision/slice-ref-image-daemon.md` — Extends the ref-image path across the comfyless/server.py Unix-socket wire with ref_image_roots and cache-key pinning; daemon/IPC only.
- `docs/vision/slice-transformer-audit.md` — Adds kind:"transformer" entries and --transformer-root to scripts/lora_audit.py feeding catalog candidacy; sibling tooling only.
- `docs/vision/slice-v5-keyframe-authoring-refine-v2.md` — Restructures comfyless/refine.py's judge/plan loop and adds comfyless/keyframe.py; refine-loop surface only.
- `docs/vision/slice-video-1-single-segment.md` — Builds comfyless/video.py single-segment video CLI; sibling-only module.
- `docs/vision/slice-video-2-chaining.md` — Adds plan.json chaining and atomic stitch to comfyless/video.py; sibling-only module.

