# ADR-013: Comfyless dep divergence from ComfyUI's torch pin

**Date:** 2026-05-16
**Status:** accepted (2026-05-16, after `security-auditor` round-2 returned `CLEAN`).
**AI-Disclosure:** Claude (Opus 4.7, 1M context) authored; Grant reviewed.
**Related:** slice-0c Vision (`docs/vision/slice-0c-cuda-torch-realignment.md`); ADR-011 (comfyless MCP server, establishes comfyless as a deploy surface separate from the ComfyUI node pack).

---

## Context

This codebase ships two consumption surfaces:

1. **The ComfyUI custom-node pack.** Installed by ComfyUI Manager via `pip install -r requirements.txt` into ComfyUI's own venv. Operators do not run our test suite; they load nodes into a running ComfyUI server. The requirements file is the canonical manifest for this path and must remain pip-compatible.
2. **Comfyless — the CLI, daemon, and MCP server.** A separate code surface under `comfyless/` introduced through ADR-006 (dual-mode CLI), ADR-011 (MCP server as LLM-agent transport), and ADR-012 (machine-boundary validator). Comfyless runs against the same node-pack code, but its lifecycle is a different one — long-lived daemon, MCP child-of-LLM-client, scripted callers — and it is the path the eight test suites (`test_manual_loop`, `test_multistage`, `test_params_schema`, `test_cascade`, `test_iterate`, `test_samplers`, `test_server_robustness`, `test_machine_boundary_validator`; 850 tests total) exercise.

Through slice 0 (pin floors → exact pins for the node pack, commit `909b228`) and slice 0b (`click==8.3.2` pin add, commit `3665461`), the test suites have been executed against the ComfyUI venv at `/home/gawkahn/projects/ai-lab/ai-stack-data/comfy-dev/run/venv/`. The "Package-manager split" section of project CLAUDE.md committed in those slices already establishes the *intent* that comfyless dev uses `uv` and the ComfyUI node-pack path uses `pip`. The missing pieces are: an actual uv-managed `.venv` to run tests against, and an ADR that names the architectural rule — comfyless's dep pin set is chosen for the comfyless deploy environment, not for ComfyUI Manager compatibility.

**This ADR establishes that rule.** It does not move any pin today; it lays the substrate so future comfyless dep bumps (driven by MCP-side needs, validator-side needs, LoRA-tooling needs, or new model-family additions) can land without re-litigating the architectural question of whether comfyless's deps must track ComfyUI's. They do not.

**Why now:**

- Slice 1 (MCP), Hunyuan-Image support, and LoRA-convert tooling are landing concurrently. Each is a plausible future source of a dep bump that ComfyUI Manager does not need (or does not want). Without this ADR, every such bump triggers a fresh "should we hold comfyless back?" discussion.
- The host upgraded to CUDA driver 13.2 (`nvidia-smi`: NVIDIA-SMI 595.58.03, Driver Version 595.58.03, CUDA Version 13.2). The slice-0c Vision was authored against the assumption that this might force a torch pin bump. PyPI pre-flight inspection (2026-05-16) shows it does not — see §1 below. But the architectural moment is real even though the version movement is not.

## Decision

### 1. CUDA target: cu130 — forced, not chosen

PyPI pre-flight inspection (2026-05-16): both `torch==2.11.0` and the latest stable `torch==2.12.0` (uploaded 2026-05-13) ship as cu130-class wheels on Linux x86_64. Both require:

- `cuda-toolkit==13.0.2`
- `nvidia-cudnn-cu13`, `nvidia-cusparselt-cu13`, `nvidia-nccl-cu13`, `nvidia-nvshmem-cu13`

No cu131+ wheel exists on PyPI. The host's CUDA 13.2 driver runs cu130 wheels via NVIDIA's forward-compat contract; that is what the existing ComfyUI venv has been doing.

Slice-0c Vision invariant 1 (`torch.version.cuda` starts with `"13."`) is satisfied by `torch==2.11.0+cu130` today. There is no version-bump action available that would change the CUDA target.

If PyPI ever stops shipping cu13x as default — or ships a cu131+ wheel for a torch version with concrete benefit — re-evaluate then. Not a today decision.

### 2. Comfyless dev/test environment: uv-managed `.venv` at the comfyless repo root

The comfyless dev environment is the uv-managed `.venv` created by `uv sync` from `pyproject.toml` and `uv.lock` at the comfyless repo root. The eight test suites are run from `./.venv/bin/python3`. The ComfyUI venv is no longer the test runner — project CLAUDE.md line 67 is updated in the same slice.

`pyproject.toml` is the human-edited source of truth for direct deps. `uv.lock` is the machine-generated full transitive lock with per-artifact integrity hashes; it is checked in. `.python-version` (if present in the worktree) pins the interpreter. No hand edits to `uv.lock`.

### 3. ComfyUI Manager path: unchanged

`requirements.txt` remains the canonical manifest for `pip install -r requirements.txt` inside ComfyUI's venv. ComfyUI Manager continues to install the node pack the way it always has; no operator workflow changes for that path.

The "Package-manager split" rule in project CLAUDE.md (committed slice 0) governs this: `pyproject.toml` and `requirements.txt` list the same direct pins in the same order at all times. Any future direct-dep movement edits both files in the same slice; `uv.lock` is regenerated in the same slice.

### 4. Comment block in `pyproject.toml` and `requirements.txt`

Both files gain a short comment block above the `torch==` line stating:

> The comfyless dev/test environment is the uv-managed `.venv` created by `uv sync`. The pin set below is chosen for the comfyless deploy environment, not for ComfyUI Manager compatibility. ComfyUI Manager installs the node pack via `pip install -r requirements.txt` and that path is supported for the node pack only — comfyless dev/test must use `uv sync` (not `pip install -r requirements.txt`) so the lockfile-hash integrity contract (see ADR-013 §5) holds. Pins in this file may diverge from ComfyUI's bundled torch in the future; when that happens, ComfyUI Manager installs may surface as a resolver conflict against ComfyUI core — that conflict is upstream's resolver, not a node-pack break. See `docs/decisions/ADR-013-comfyless-torch-divergence.md` for the architectural rule, and `pyproject.toml` / `uv.lock` for the comfyless-side dep contract.

Same comment text in both files. This is slice-0c Vision invariant 9 shape (b). The text addresses the two operator failure modes flagged by the security-auditor (F-3): (a) resolver-conflict triage in ComfyUI's venv if a future bump diverges, and (b) the explicit "DO NOT use `requirements.txt` to set up the comfyless dev/test environment" warning that preserves the lockfile-hash integrity story for the comfyless side.

### 5. Hash-pinning posture: uv.lock integrity hashes, no `--require-hashes` opt-in

`uv.lock` records SHA-256 integrity hashes for every direct and transitive artifact; `uv sync` verifies them at install time. This satisfies the integrity-against-republish concern called out in global §11 paragraph 4 for non-Red-Zone code.

`--require-hashes` (pip-side discipline) is **not** opted into. Reasons:

- The slice does not change the pip path. ComfyUI Manager installs from `requirements.txt` without hashes today; the slice preserves that behavior so operators are unaffected. Adding hashes to `requirements.txt` would break the ComfyUI Manager path.
- The comfyless dev path uses `uv sync`, which already hash-verifies via the lockfile.
- This is an L2 slice (dep manifest; no Red Zone surface). Global §11 paragraph 4 allows `--require-hashes` opt-in for Red-Zone-heavy projects; comfyless's Red Zone surface is the MCP request boundary (ADR-011, ADR-012), not the dep manifest.

**Load-bearing assumption (per security-auditor F-2):** this posture rests on `uv sync` enforcing per-artifact SHA-256 verification at install time as its default and non-manifest-configurable behavior. The integrity check matters because comfyless's runtime *consumers* of these deps include Red Zone surfaces — the MCP request handler (ADR-011 §3) and the machine-boundary validator (ADR-012) both load into the same process where `torch` / `diffusers` / `transformers` / etc. are imported. The dep manifest itself is L2, but the *integrity contract* is what keeps a compromised-republish attack from landing inside a Red Zone process. The lockfile-hash discipline is doing real supply-chain work, not just hygiene.

If a future `uv` release adds a `--no-verify-hashes` flag (or equivalent) and operator workflows adopt it, this chain silently breaks. The slice does not opt into `--require-hashes` as a belt-and-braces defense today because the L2 risk classification does not warrant the operator-friction cost; revisit the decision the moment uv's hash-verify default changes or a Red Zone code path begins doing dynamic dep loads.

### 6. Pin set today: unchanged ML-stack pins + scipy declared

The slice intentionally moves no ML-stack pin (`torch`, `diffusers`, `transformers`, `accelerate`, `peft` remain at slice-0b versions). One slice-time addition was approved by Grant during execution: `scipy==1.17.1` is declared as the 11th direct dep, matching the version implicitly satisfied by the ComfyUI venv's transitive graph. The slice's first `uv sync` surfaced that `nodes/eric_diffusion_manual_loop.py:624` imports `scipy.stats.beta` directly; the dep was never declared. The Changelog "2026-05-16 (slice-time scope expansion)" entry captures the full audit trail and Grant's approval.

The eleven direct pins as of slice 0c close:

```
torch==2.11.0
diffusers==0.37.1
transformers==5.5.3
accelerate==1.13.0
peft==0.18.1
safetensors==0.7.0
pillow==12.2.0
numpy==2.4.4
scipy==1.17.1   ← slice-time addition (declared, not bumped)
mcp==1.27.0
click==8.3.2
```

Order matches between `pyproject.toml` and `requirements.txt` (project CLAUDE.md rule). Slice-0c Vision invariant 5 ("monotonic or unchanged") is satisfied: every pre-existing pin is unchanged; scipy is a *new* declared pin, not a downgrade of any existing one.

Rationale for not bumping torch 2.11.0 → 2.12.0 today (the only ML-stack version-bump that was on the table):

- Same CUDA target (cu130). No invariant unlocked.
- triton 3.6 → 3.7 is dead code on this codebase (no `torch.compile` / inductor usage).
- cudnn / nccl / cusparselt micro-version bumps with no API surface we depend on.
- ~8 new NVIDIA-lib SHA-256 hashes to audit, real test-suite re-validation work, and a 3-day-old release with no in-the-wild regression signal.

The architectural rule lands now. The FIRST future slice with a concrete reason to walk forward picks up the version-bump cost then.

### 7. Live smoke (per Vision invariants 7 + 8)

Pre-realignment baseline PNG and post-realignment smoke PNG are both generated with:

- Model: `Qwen/Qwen-Image` (already cached in local HF at `/mnt/nvme-8tb/hf`)
- Resolution: 256×256
- Steps: 2
- Seed: 12345
- `true_cfg_scale`: 2.0
- Sampler: `default` (the CLI's `--sampler` choices are `{default, multistep2, multistep3}`; qwen-image uses its bundled DDPM-style scheduler so `default` is the appropriate value — corrected from the slice-0c Vision Changelog's "euler" wording which named a sampler the CLI does not accept)
- Schedule: `linear`
- Prompt: `"a red cube on a white table"`
- Negative prompt: `""`

The exact `comfyless` invocation goes in this ADR's Changelog when the baseline is captured. The baseline PNG's SHA-256 is recorded in the Changelog. Pixel-MSE on PIL RGB 0–255 is computed against the baseline; threshold ≤ 1.0 per Vision invariant 8. PNG binaries are NOT committed (size; the SHA is the audit anchor).

### 8. Order of operations (per global §12)

1. ADR-013 (this document) — Status: `proposed`.
2. `security-auditor` (Opus, `model: "opus"` at invocation per global §5A) reviews ADR-013. Output saved to `docs/security/review-slice-0c-2026-05-16.md`.
3. Iterate ADR if `CHANGES REQUIRED`. Re-fire until `CLEAN`. Status flips to `accepted`.
4. Pre-realignment baseline PNG captured from the existing ComfyUI venv. SHA-256 recorded in this ADR's Changelog. Pre-slice `sha256sum uv.lock` also recorded in the Changelog at this step — the post-slice value is recorded at step 6 for auditability of the "byte-identical (modulo uv migrations)" claim (security-auditor F-4).
5. `uv sync` stands up `./.venv`. Runtime check: `import torch; assert torch.version.cuda.startswith("13.")` exits 0.
6. **Lock-integrity gate (per security-auditor F-1):** run `uv lock --check` BEFORE the comment-block edit. Expected exit 0 (asserts the existing lock is in sync with `pyproject.toml`; comments don't change resolution so a clean pre-edit check is the right baseline). If the check exits non-zero before any edit, the slice STOPS and surfaces the unexpected drift to Grant for go/no-go before proceeding. Then add the comment block to `pyproject.toml` + `requirements.txt`. Re-run `uv lock`; inspect the post-edit `uv.lock` diff. Pure-metadata diffs (uv revision number, source URL canonicalization) are acceptable. **Any transitive-dep version movement or new SHA-256 hash for an unchanged version surfaces to Grant as a separate go/no-go BEFORE commit** — that exact case is what the lockfile-hash discipline is designed to catch and the slice will not silently pull it in under the "unchanged pins" framing. Record the post-edit `sha256sum uv.lock` in this ADR's Changelog alongside the pre-edit value.
7. CLAUDE.md line 67 updated (test-runner path + 8 suites + 850 tests).
8. All 8 test suites run against `./.venv`. Must pass 850/850. Failure aborts the slice without commit.
9. Live smoke generation. Pixel-MSE ≤ 1.0 vs baseline. Above-threshold failure surfaces to Grant for go/no-go.
10. `code-reviewer` (Opus) on each code-touching commit. Commit batch (3–5 commits, `feat(deps):` prefix). The closure commit body documents `~/.cache/uv` directory size growth and `.venv` directory size per slice-0c Vision invariant 12 (operator disk-cost visibility). Push approval requested at batch close.

**Note for future readers using §8 as a template (per security-auditor F-6):** this slice ships unchanged pins (slice shape A); `security-auditor` is correctly invoked once on the ADR design and not per code commit. The FIRST future slice that DOES move a pin layers `security-auditor` (Opus, `model: "opus"` at invocation) onto each code-touching commit IN ADDITION TO `code-reviewer`, per slice-0c Vision §Red Zone ownership ("supply-chain audit: every new direct-pin version brings new SHA-256 hashes … hash-mismatch = bail out"). Do not copy this §8 verbatim for a version-bump slice; copy with that addition.

## Alternatives Rejected

### A. Walk torch + diffusers + transformers + peft to latest in this slice

Rejected. Torch 2.11→2.12 buys no API surface we use; the other three bumps lack a concrete trigger today. Full-sweep cost is high (8+ new NVIDIA SHA-256 hashes, test-suite re-validation, brand-new release exposure) with no offsetting unlock. Future slices that have a concrete reason — Hunyuan needing a new diffusers pipeline class, LoRA-convert needing newer peft, MCP slice 2+ needing newer mcp SDK — pay the bump cost then.

### B. Pin the pytorch.org cu13x index in `pyproject.toml` via `[tool.uv.sources]` or `--index-url`

Rejected. PyPI default Linux wheel for `torch==2.11.0` resolves to the cu130 build cleanly; uv handles it without extra index config. Pinning the pytorch.org index makes `uv sync` operator-dependent (extra index URL, potential auth in corporate environments) for zero benefit at cu130. If PyPI ever stops shipping cu13x as default, the call to add the index pin is a future ADR amendment.

### C. Opt into `--require-hashes` for the comfyless dev/test surface

Rejected at this slice. The `uv.lock` integrity-hash discipline (uv sync verifies SHA-256 on install) is sufficient at L2 — global §11 paragraph 4 explicitly allows projects to skip `--require-hashes` outside Red-Zone heavy code. Adding it would require either (a) parallel hashed and unhashed `requirements.txt` files (operator confusion) or (b) hashing `requirements.txt` itself, which breaks ComfyUI Manager. Defense in depth here would change behavior the slice explicitly wants to preserve.

**Per security-auditor F-2 (folded into §5):** the lockfile-hash posture is doing real (not cosmetic) supply-chain work because comfyless's runtime consumers include Red Zone surfaces (ADR-011 MCP request handler, ADR-012 validator). Skipping `--require-hashes` here means trusting `uv sync`'s default integrity verification — it does not mean skipping integrity altogether. If a future uv release relaxes that default, this rejection is revisited.

### F. Split the direct-pin sets between `pyproject.toml` (exact, comfyless-deploy-chosen) and `requirements.txt` (floor-style, ComfyUI-Manager-friendly)

Rejected (per security-auditor F-5). The "Package-manager split" rule in project CLAUDE.md and slice 0's §11 hook (`block-pyproject-floors.sh`) both rest on lockstep agreement between the two files; splitting them re-introduces floor specifiers in `requirements.txt` which the hook would block, and changes the comfyless project from "one set of pins" to "two sets of pins with a divergence policy." That doubles the audit surface for every future bump. The comment block (§4) is the lighter-weight signal: same pins in both files, comment block in both files explaining the divergence rule, ADR for the architectural anchor. Revisit only if ComfyUI Manager resolver conflicts against ComfyUI core's deps actually force the split — a future ADR amendment, not this one.

### D. Keep comfyless tests running against the ComfyUI venv indefinitely

Rejected. The ComfyUI venv's pin set is governed by ComfyUI Manager's resolver and by ComfyUI core release cadence — neither under our control. The "Package-manager split" rule in project CLAUDE.md already declared the intent for divergence; this ADR lands the mechanism so future bumps don't have to negotiate "but ComfyUI doesn't want this." Solo-defensible lens (per global §1): the divergence is also team-portable — anyone forking comfyless for their own LLM agent integration benefits from comfyless's deps being independently versioned.

### E. Use a different ML backend (jax, ONNX Runtime, etc.) as part of the realignment

Rejected — explicitly out of scope per slice-0c Vision's "Out of scope" list. Backend switching is its own ADR if it ever surfaces.

## Deferred / Out of Scope

- **The torch 2.11.0 → 2.12.0 bump itself.** Future slice when a concrete benefit surfaces.
- **diffusers / transformers / peft bumps.** Each is its own future slice with `code-reviewer` (Opus) review at minimum and `security-auditor` (Opus) if the bump touches the MCP request boundary or validator path.
- **Multi-CUDA wheel matrix.** Today this codebase targets a single host. Multi-target wheel installs are a future concern.
- **Pinning to specific pytorch.org index URLs.** Future ADR amendment if needed (see alternative B).
- **HTTP/SSE MCP transport** — out of scope per ADR-011 §6 + 2026-05-04 amendment; gated on runtime-core cluster + comfyless failed-load resilience.
- **Containerized deploy environment for comfyless.** Future slice; would land its own dep pins via the container image's `FROM` digest pin (per global §11 container-image rule).
- **GPU-related deps** (`xformers`, `flash-attn`, `bitsandbytes`, `triton` as a direct dep): each is its own future slice per slice-0c Vision's "Out of scope."

## Changelog

- **2026-05-16 (initial draft)**: ADR drafted in response to slice-0c Vision approval. Slice shape A (unchanged pins, establish divergence only) pinned by Grant after PyPI pre-flight showed no cu131+ wheel exists and the only available torch bump (2.11.0 → 2.12.0) is mostly aesthetic on this codebase. Six open questions resolved per slice-0c Vision Changelog 2026-05-16. Status: `proposed`. Next step: `security-auditor` (Opus) review of this ADR design before code lands.

- **2026-05-16 (security-auditor round-1 fold-in)**: Round-1 review (saved to `docs/security/review-slice-0c-2026-05-16.md`) returned `CHANGES REQUIRED` with 1 HIGH + 2 MEDIUM + 3 INFO. All six folded:
  - **F-1 (HIGH)**: §8 step 6 now gates the slice on `uv lock --check` BEFORE the comment-block edit (asserts the existing lock is in sync with `pyproject.toml`; comments don't affect resolution so a clean pre-edit check is the right baseline). Any non-metadata transitive-graph delta after the post-edit `uv lock` re-run surfaces to Grant as a separate go/no-go before commit. The "byte-identical (modulo uv migrations)" claim is now a verified assertion, not a hope.
  - **F-2 (MEDIUM)**: §5 now names the load-bearing assumption — `uv sync` enforces per-artifact SHA-256 verification at install time as its default — and the runtime-consumer Red Zone surfaces (MCP handler per ADR-011, validator per ADR-012) that make the lockfile-hash discipline supply-chain work, not just hygiene. Alternative C amended in parallel.
  - **F-3 (MEDIUM)**: §4 comment-block text lengthened to: (a) name `uv sync` (not `pip install -r requirements.txt`) as the only correct way to set up the comfyless dev/test environment, and (b) signal that future-divergence resolver conflicts in ComfyUI's venv are upstream's resolver issue, not a node-pack break.
  - **F-4 (INFO)**: §8 step 4 / step 6 record pre-slice and post-slice `sha256sum uv.lock` in this ADR's Changelog when the slice closes — cheap belt-and-braces alongside the F-1 gate.
  - **F-5 (INFO)**: Alternative F (split direct-pin sets) added to Alternatives Rejected.
  - **F-6 (INFO)**: §8 trailing note states that future version-bump slices using §8 as a template must add `security-auditor` (Opus) per code-touching commit per slice-0c Vision §Red Zone ownership; this slice ships unchanged pins so `security-auditor` is correctly limited to the ADR design review.

  Re-firing `security-auditor` (Opus) round 2 on the amended ADR. Status flips to `accepted` if round 2 returns `CLEAN`.

- **2026-05-16 (security-auditor round 2 → CLEAN, status accepted)**: Round-2 review (appended to `docs/security/review-slice-0c-2026-05-16.md`) verified all six round-1 fold-ins are ADDRESSED. Round 2 surfaced one new INFO finding (F-7) — slice-0c Vision invariant 12 (operator disk-cost disclosure: `~/.cache/uv` growth + `.venv` size in the closure commit body) was not named in §8 step 10. Folded in the same edit (single sentence appended to §8 step 10). Verdict: **CLEAN**. Status flipped from `proposed` to `accepted`. Implementation may now begin per §8 starting at step 4 (pre-realignment baseline PNG + pre-slice `sha256sum uv.lock` capture).

- **2026-05-16 (§8 step 4 audit anchors captured)**:
  - **Pre-slice `uv lock --check`** — exit 0; 87 packages resolved against `pyproject.toml`. F-1 pre-edit gate: CLEAN.
  - **Pre-slice `sha256sum uv.lock`** — `885b8fd714761a70db0bb9126f3252596ced2d53afd912c9eb81aaa17107a969` (commit `c7fd91f` baseline; uv 0.11.7).
  - **Pre-realignment baseline PNG** — `/tmp/baseline-slice-0c-pre-realignment.png`, 62321 bytes, SHA-256 `f9e68ad6f3de3e7bbb3c34087201abe9cff3a30d05b08a6cd7603f150017b076`. Generated via `/home/gawkahn/projects/ai-lab/ai-stack-data/comfy-dev/run/venv/bin/python3 -m comfyless.generate --model /home/gawkahn/projects/ai-lab/ai-base/models/hf-local/Qwen-Image --prompt "a red cube on a white table" --negative-prompt "" --seed 12345 --steps 2 --true-cfg 2.0 --width 256 --height 256 --sampler default --schedule linear --output /tmp/baseline-slice-0c-pre-realignment.png`. Note: comfyless emitted "true_cfg_scale is passed as 2.0, but classifier-free guidance is not enabled since no negative_prompt is provided" — empty negative prompt disables CFG; the smoke still exercises the loader + transformer + VAE decode path deterministically. Sidecar JSON written to `/tmp/baseline-slice-0c-pre-realignment.json`. Binary PNG and sidecar NOT committed; SHA is the audit anchor.
  - **§7 sampler correction**: §7 originally named `sampler: euler` (carried verbatim from the slice-0c Vision Changelog 2026-05-16 Q6). The comfyless CLI accepts `{default, multistep2, multistep3}`; "euler" is not a valid value. qwen-image uses its bundled DDPM-style scheduler so `default` is the appropriate value. §7 amended to record the corrected sampler name alongside the inline note.

- **2026-05-16 (slice-time scope expansion — scipy added as 11th declared dep)**: `uv sync` to the new `.venv` proceeded without error; runtime check passed (`torch.version.cuda == "13.0"`, `torch.cuda.is_available() == True`, capability `(12, 0)`). First 8-suite run surfaced `ModuleNotFoundError: No module named 'scipy'` in `test_manual_loop` at the beta-sigma-schedule code path (`nodes/eric_diffusion_manual_loop.py:624`). `scipy` is genuinely used by the codebase (beta sigma schedules) and was implicitly satisfied by the ComfyUI venv's transitive graph (scipy 1.17.1 present there); it was never declared as a direct dep. This is exactly the latent-undeclared-dep failure mode the divergence slice was designed to flush out.

  Per global §0 rule 7 + §4 ("never add dependencies without explicit human approval") Grant was asked; approved adding `scipy==1.17.1` as the 11th declared direct dep (matches ComfyUI venv version). Added between `numpy` and `mcp` in both `pyproject.toml` and `requirements.txt` (math/numerics group). Project `CLAUDE.md` "Package-manager split" rule updated from "10 top-level pins" to "11 top-level pins" with `scipy` named in the in-line list. `uv lock` re-resolved cleanly (88 packages, scipy added; no other transitive movement). `uv sync` installed `scipy==1.17.1` only.

  Slice shape A framing is preserved — the five ML-stack pins (`torch`, `diffusers`, `transformers`, `accelerate`, `peft`) remain at slice-0b versions and were not bumped in this slice. The scipy addition is a "missing declared dep, version-matched to ComfyUI venv runtime" correction, not a version-bump of any existing pin. The divergence rule established by this ADR is what made the gap visible.

- **2026-05-16 (§8 step 6 + 8 + 9 closure data)**:
  - **F-1 post-edit `uv lock --check`** — exit 0 after the comment-block-only edit (no scipy yet). `uv.lock` SHA unchanged from pre-edit: `885b8fd714761a70db0bb9126f3252596ced2d53afd912c9eb81aaa17107a969`. Comment-block edit produced no transitive-graph delta, as the F-1 gate expected.
  - **Post-scipy `uv lock`** — 88 packages resolved (was 87); `Added scipy v1.17.1` is the only resolution change. Post-scipy `uv.lock` SHA: `e2b2230a9845afcd31786e668d639711dc0b3ef3515a09cc2c06d3896d80b28c`. The lock delta is bounded to the scipy addition; no other transitive movement.
  - **Test suite (§8 step 8)** — all 8 suites run from `./.venv/bin/python3` against the new `.venv`. Result: **850 passed, 0 failed** (`test_manual_loop` 186, `test_multistage` 141, `test_params_schema` 135, `test_cascade` 129, `test_machine_boundary_validator` 118, `test_iterate` 92, `test_samplers` 41, `test_server_robustness` 8). Vision invariant 6 satisfied.
  - **Live smoke (§8 step 9)** — same comfyless invocation as the baseline, run from `./.venv/bin/python3` against the new `.venv`. Output: `/tmp/post-slice-0c.png`, 256×256 PNG. **Pixel-MSE vs baseline (PIL RGB 0–255): 0.000000.** Max absolute per-channel delta: 0.0. Decoded pixel bytes byte-for-byte identical to baseline. PNG file SHAs differ only because the embedded comfyless tEXt-chunk metadata includes a generation timestamp. Vision invariants 7 + 8 satisfied (well below the ≤ 1.0 threshold).
  - **Disk-cost disclosure (invariant 12)** — `.venv` = 4.9 GB (full torch + cuda-13 wheel stack + scipy). `~/.cache/uv` = 5.3 GB (unchanged from pre-slice; uv hardlinked existing torch/cuda wheels from prior shared cache use, only scipy was newly downloaded). `uv` hardlink fallback warning emitted during sync (`.venv` on mergerfs at `/home/gawkahn/projects/...`, cache on ext4 at `~/.cache`); cross-filesystem hardlink not supported, full-copy fallback is expected and operationally benign.
