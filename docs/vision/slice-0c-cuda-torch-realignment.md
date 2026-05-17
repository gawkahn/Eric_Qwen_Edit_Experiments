# Slice 0c Vision — Comfyless CUDA / torch realignment from ComfyUI's pin

**Date:** 2026-05-16
**ADR:** to be drafted as part of this slice — "Comfyless dep divergence from ComfyUI's torch pin" (architectural moment when comfyless's deploy env stops tracking ComfyUI's torch choice).
**Status:** proposed — awaiting Grant's approval. Designed for execution in a parallel 4th Claude session (worktree of choice) running independently from slice 1, Hunyuan-Image, and LoRA-convert sessions.
**AI-Disclosure:** Claude (Opus 4.7, 1M context) authored; Grant reviewed.

---

## Slice

Bump comfyless's torch and ML-stack pins (`torch`, `diffusers`, `transformers`, `accelerate`, `peft`) from the ComfyUI-aligned slice-0-era pins to the greatest set of versions that:

1. ship wheels for the host's CUDA version (CUDA 13.2 target; CUDA 13.0 acceptable fallback if 13.2 wheels are unavailable or have known issues), AND
2. satisfy every downstream minimum-torch / minimum-transformers / etc. constraint declared by the other direct deps, AND
3. preserve every behavior asserted by the existing 850-test suite.

This is the architectural moment where comfyless's deploy environment stops being yoked to ComfyUI's. ComfyUI Manager continues to install the custom-node pack via `pip install -r requirements.txt` into its own venv; that path is unaffected. The comfyless-dev uv-managed `.venv` evolves on its own cadence after this slice.

## Posture

- **Boundary:** dep manifest (`pyproject.toml` + `requirements.txt` + `uv.lock`). Affects every consumer of torch and the surrounding ML stack.
- **Risk factors:** supply-chain shape (new versions = new SHA-256 hashes to audit); test-suite regression (subtle behavior changes in torch / diffusers / transformers / accelerate / peft); CUDA-driver-version mismatches surfacing at runtime, not install time.
- **Risk level:** **L2.** Touches dep manifest but no auth / PII / Red-Zone surface. §11 hash-pinning compliance is the governing rule; not §5 Red Zone.

## Intent

Land a coherent version bump for the ML stack:

- `torch` → a version whose `torch.version.cuda` reports a 13.x runtime matching (or the greatest compatible below) the host system's CUDA 13.2.
- `diffusers`, `transformers`, `accelerate`, `peft` → versions whose minimum-torch / minimum-transformers requirements are satisfied by the bumped torch. No downgrades from current.
- `safetensors`, `pillow`, `numpy`, `mcp`, `click` → unchanged unless one of them blocks the bump (none expected; safetensors is CUDA-agnostic, the rest are pure-Python).

The matrix walk is the implementation work; the Vision asserts the *constraints* the walk must satisfy, not the specific versions chosen.

ADR documents the divergence: from this slice forward, comfyless's torch pin is chosen for the comfyless deploy environment, not for ComfyUI Manager compatibility.

## Invariants (must always be true)

1. After `uv sync`, `python -c "import torch; print(torch.version.cuda)"` outputs a string starting with `"13."`. (Targets CUDA 13.x.)
2. After `uv sync`, `torch.cuda.is_available()` returns `True` and `torch.cuda.get_device_capability()` returns a capability tuple supported by the resolved torch wheel.
3. `uv.lock` is in sync with `pyproject.toml` (`uv lock --check` exits 0). No hand edits to the lock.
4. `pyproject.toml` and `requirements.txt` list the same direct pins in the same order (project CLAUDE.md rule). No floor-style specifiers (`>=`/`~=`/`^`/`*`) — §11 enforcement, hook-checked.
5. Every direct dep version is monotonic or unchanged versus the prior pin (no downgrades). Any downgrade requires explicit acknowledgement in the Change Contract and a recorded rationale in the ADR.
6. All 850 existing unit tests pass against the new `.venv` (the eight existing suites: `test_manual_loop`, `test_multistage`, `test_params_schema`, `test_cascade`, `test_iterate`, `test_samplers`, `test_server_robustness`, `test_machine_boundary_validator`). Zero failures, zero skips not already present in the baseline.
7. A live smoke generation (a small Qwen-Image or Flux family run with a fixed seed and a small step count) completes without `CUDA`, `cuDNN`, or library-load errors and writes a PNG. The exact model/prompt/seed is documented in the Change Contract.
8. Pre-realignment baseline image versus post-realignment smoke image, both rendered with identical seed/model/sampler/steps: similarity above a Change-Contract-declared threshold (default proposal: pixel-MSE on PIL pillow images ≤ 1.0 on `RGB` 0-255 scale; or LPIPS ≤ 0.05 — author of slice 0c picks one). Soft fail: surfaces to Grant for manual go/no-go, gets recorded in the ADR if accepted.
9. ComfyUI's custom-node-install path is NOT broken. Two acceptable shapes:
   - (a) `pip install -r requirements.txt` inside ComfyUI's venv installs the new pins without conflict against ComfyUI's other deps; OR
   - (b) `requirements.txt` and `pyproject.toml` carry an explicit comment block above the `torch==` line noting the comfyless dev pin diverges from ComfyUI's expected torch and instructing operators to install comfyless dev/test via `uv sync`, while ComfyUI Manager continues to install only ComfyUI-side deps. The Change Contract picks (a) or (b).
10. No security-truth surfaces added or removed. No auth, no PII, no path-allowlist, no audit-log changes.
11. The ADR is committed before the pin changes land (per project CLAUDE.md §12 order-of-operations).
12. Post-realignment, `~/.cache/uv` directory size growth and the `.venv` directory size are documented in the closure commit body. Operators have visibility into the disk cost.

## Failure semantics

- **Matrix-walk dead end** (no compatible quartet of `torch` + `diffusers` + `transformers` + `accelerate` + `peft` exists for the target CUDA): slice exits WITHOUT committing any pin changes. Surface the constraint conflict (e.g., "diffusers ≥0.41 requires transformers ≥6.0 which has no wheel for cu131"); ask Grant to relax (drop to CUDA 13.0; accept a downgrade of one ML lib; or postpone the slice).
- **Test-suite regression** (any of the 850 fails): slice does NOT commit. Surface the failing test and the suspected dep that caused the regression. Either (a) the slice author writes an ADR-amendment justifying a behavior-change test update, or (b) the slice rolls back the offending pin and re-walks the matrix.
- **Live smoke crash at CUDA driver level:** slice does NOT commit. Caught and surfaced; almost always a CUDA-driver / wheel-CUDA mismatch worth fixing before proceeding.
- **Image-similarity below threshold:** slice surfaces the regression with concrete numbers (pixel-MSE / LPIPS values + side-by-side images) and asks Grant for go/no-go. If accepted, the deviation is recorded in the ADR.
- **uv.lock merge conflict with concurrent slices:** slice 0c IS the merge-hotspot owner — it lands as a clean commit ahead of any other slice's dep-touching commits, OR it rebases on top of them once they land. Coordination through Grant; never `git push --force`.

## Out of scope

- Switching to a different ML backend (jax, ONNX Runtime, etc.). Comfyless stays on torch.
- Quantization or precision shifts (bf16 → fp8, fp16 → int8). Existing precision discipline preserved.
- Adding new GPU-related deps: `xformers`, `flash-attn`, `bitsandbytes`, `triton`. Each is its own future slice with its own §11 audit.
- ComfyUI-side dep changes. ComfyUI's venv evolves on its own cadence.
- HF cache layout changes (the bind-mount to `/mnt/nvme-8tb/hf` per filesystem note stays as-is).
- Performance benchmarking beyond the smoke-test sanity check. Throughput / VRAM profiling is a separate concern.
- Multi-GPU / distributed-training support.
- CUDA-version downgrade scenarios (target is to GO UP from 12.x to 13.x; rollback is the slice's failure-semantics concern, not its goal).
- ComfyUI Manager's pip-resolver behavior with the new pins (responsibility shifts to operator if shape (b) of invariant 9 is chosen).
- Pinning to specific PyTorch-index URLs (e.g. `https://download.pytorch.org/whl/cu131`) versus default PyPI wheels — the Change Contract picks one; the Vision allows either.

## Negative cases (required)

- **N1**: Pin torch to a version with no compatible cu13x wheel → `uv lock` fails with a clear "no matching distribution" error. Slice does not produce a partial pin.
- **N2**: Pin diffusers to a version whose minimum-torch exceeds the resolved torch → `uv lock` fails with a clear constraint conflict. Slice surfaces which dep is forcing the conflict.
- **N3**: After successful `uv lock` and `uv sync`, runtime check (`import torch; assert torch.version.cuda.startswith("13.")`) FAILS → slice does not declare success; either the wheel was the wrong tag (uv index misconfiguration) or torch resolved to a version older than expected.
- **N4**: Any of the 850 existing tests fails when run from the new `.venv` → test runner exits non-zero; slice does not commit.
- **N5**: `nvidia-smi` reports a CUDA driver version below the wheel's required minimum at runtime → caught at slice's startup-smoke step; surfaced before any pin commit lands.
- **N6**: Live smoke generation crashes inside the diffusion call (not at CUDA driver level, but at a higher API surface — e.g., `RuntimeError: expected scalar type Half but found BFloat16`) → slice does not commit; the version pair causing the regression is identified.
- **N7**: §11 hook (`block-pyproject-floors.sh`) catches an accidental `>=` slip → commit refused; slice fixes and re-stages. Asserts the hook is still functional after the bump.
- **N8**: `requirements.txt` and `pyproject.toml` end up disagreeing on order or specific versions → grep-based assertion in the slice's closing test catches the drift before commit.

## Proof hooks

- **Positive smoke (CUDA runtime):** `python -c "import torch; assert torch.version.cuda.startswith('13.'), torch.version.cuda; assert torch.cuda.is_available(); print(torch.cuda.get_device_capability())"` exits 0.
- **Full test baseline:** run all 8 existing suites against the new `.venv`:
  ```bash
  for suite in test_manual_loop test_multistage test_params_schema \
               test_cascade test_iterate test_samplers \
               test_server_robustness test_machine_boundary_validator; do
      ./.venv/bin/python3 ${suite}.py || exit 1
  done
  ```
  Expected: 850 / 850 passing, zero failures.
- **Live smoke (GPU):** a small generate call producing a PNG, with the model/prompt/seed/steps recorded in the Change Contract. Compare against the pre-realignment baseline image (saved as `docs/security/baseline-slice-0c-pre-realignment-<sha>.png` before any pin change lands).
- **Lock-sync check:** `uv lock --check` exits 0 with no proposed changes.
- **Manifest-agreement check:** `diff <(grep -E "^[a-z]" requirements.txt) <(awk -F'"' '/^    "/ {print $2}' pyproject.toml | sed 's/==.*//;s/$/==/' | xargs -I {} grep -E "^{}.*" requirements.txt)` (or equivalent shape) confirms identical order.
- **Reproducibility:** in a fresh shell, `rm -rf .venv && uv sync` reconstructs the exact environment recorded in `uv.lock` — verified by `pip freeze | sha256sum` matching a recorded hash.

## Red Zone ownership

This is L2 — not Red Zone in the §5 sense — but two §11 concerns require Grant's verification:

- **Supply-chain audit:** Every new direct-pin version brings new SHA-256 hashes recorded in `uv.lock`. Reviewer (security-auditor, Opus) verifies the hashes match what PyPI / `download.pytorch.org` publishes for those versions at the time of the bump. Hash-mismatch = bail out.
- **Hash-pin integrity:** `uv.lock` continues to record per-artifact integrity hashes. Slice 0c does NOT relax this; an `--require-hashes`-equivalent install discipline is preserved (the slice may opt into `uv pip install --require-hashes` in the Change Contract if Grant wants the stricter flavor — Red-Zone-style is documented in global §11 paragraph 4).

Owned by Grant — AI-generated only, not sole author. Reviewer cadence: `code-reviewer` (Opus) MANDATORY; `security-auditor` (Opus) MANDATORY for the supply-chain pinning surface.

## Pointers

- ADR for the divergence: to be drafted as `docs/decisions/ADR-013-comfyless-torch-divergence.md` (assumes ADR-013 is the next sequential number; verify at draft time). Drafted and accepted BEFORE pin changes land.
- Project CLAUDE.md "Package-manager split" rules — preserved by this slice; only the comfyless-dev (uv) path changes. The pip / ComfyUI-Manager path is governed by invariant 9.
- Pre-realignment baseline image — prepared and saved BEFORE any pin change. Referenced via SHA-256 in the post-realignment regression assertion. NOT committed as a binary artifact (size); the SHA goes in the ADR.
- Parallel-session orientation prompt — written separately for the slice-0c session by the slice-1 session that authored this Vision (this file). Includes the file-boundary list against the three concurrent sessions (slice 1 / Hunyuan / LoRA-convert).
- Filesystem note (global CLAUDE.md): host has CUDA 13.2; `/home/gawkahn/projects/` is mergerfs (no fcntl locks); HF cache lives at `/mnt/nvme-8tb/hf` (bind mount). The `.venv` lands on mergerfs by default and is expected to work; if `uv sync` or runtime imports wedge, relocate via `UV_PROJECT_ENVIRONMENT`.

## Coordination notes (parallel-session boundary)

This slice's edit scope is exclusively `pyproject.toml`, `requirements.txt`, `uv.lock`, plus the ADR, the Change-Contract artifact, and the security-review save. CLAUDE.md may be touched ONLY to update the test-runner invocation guidance (line 67 currently names the ComfyUI venv; this slice flips it to name the new `.venv`).

Files the concurrent sessions own that slice 0c MUST NOT TOUCH:
- slice 1 (MCP): `comfyless/mcp_server.py`, `test_mcp_server.py`, surgical bits of `comfyless/generate.py` / `comfyless/cascade.py`, `comfyless/README.md`, `docs/vision/slice-1-mcp-generate.md`
- Hunyuan: `nodes/eric_diffusion_loader.py`, `nodes/eric_diffusion_generate.py`, `comfyless/family_defaults.py`, and any new Hunyuan node files
- LoRA-convert: `scripts/` (or repo-root `tool:`-prefix CLIs)

`uv.lock` is the merge hotspot. Slice 0c rebases on top of any lock-touching commit that lands while it's in flight. The concurrent sessions DO NOT touch the lock unless they need a new dep — and if they do, they coordinate with slice 0c through Grant first.

When slice 0c lands, the concurrent sessions each run `uv sync` to refresh their respective `.venv`s and re-run their test suites to verify no regression from the bump.
