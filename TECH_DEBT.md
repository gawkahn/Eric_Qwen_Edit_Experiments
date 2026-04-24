# Tech Debt

Items here are conscious deferrals — known gaps with a recorded reason for not fixing now.
Format: **Item** — why deferred, what triggers revisiting.

---

## Security

**`SO_PEERCRED` check on daemon `--unload`** *(2026-04-21)*
Any local user can send a shutdown request to the comfyless daemon and kill it.
On a single-user workstation this is a minor nuisance with no data-loss risk.
Implement if the tool is ever deployed on shared infrastructure (lab server, cloud
dev box). Trigger: first report of someone being disrupted, or shared-machine deployment.
See ADR-001.

**Daemon inference timeout** *(2026-04-21)*
A hung or very long generation blocks all subsequent clients indefinitely.
A configurable server-side timeout (abort + structured error response after N seconds)
is the right fix. Deferred until the daemon is implemented and basic operation is
verified. Trigger: implementation of the daemon server module.
See ADR-001.

**Daemon request rate limiting / VRAM exhaustion** *(2026-04-21)*
A client can force repeated 20B-parameter model reloads (each takes minutes, uses
~40GB VRAM) by alternating model paths on every request. Reasonable bounds on max
steps, max dimensions, and minimum time between model swaps would mitigate this.
Low urgency on a single-user machine. Trigger: shared-machine deployment.
See ADR-001.

---

## Dependencies

**Hash-locked installs** *(2026-04-21)*
`pip install --require-hashes` with a generated lock file defends against the
"same version, different bytes" PyPI backend-compromise threat that plain `==` pins
don't catch. Only worth doing from project start — retrofitting requires hashes for
all 100–200 transitive deps, with per-platform lock files for wheels.
Trigger: any new greenfield project in this space; do it from day one there.
See `pyproject.toml` comments and `project_dependency_pin_strategy.md` memory.

---

## Sampler Coverage

**Heun / RK3 samplers** *(2026-04-21)*
Single-step higher-order methods (Heun, RK3, RK4) require 2+ model evaluations per
denoising step. The scheduler API (`set_timesteps` / `step`) only allows one model
call per step. Implementing these requires a full manual denoising loop that controls
model calls directly (Phase C / RES4LYF territory). The current Adams-Bashforth
multistep samplers (multistep2, multistep3) cover the available quality improvement
space within the scheduler API.
Trigger: Phase C manual denoising loop is implemented.
See `docs/decisions/ADR-005-sampler-multistep-only.md`.

---

## LoRA

**Text encoder LoRA support (`lora_te1_*` keys)** *(2026-04-21)*
Keys prefixed `lora_te1_*` are currently silently dropped during LoRA loading.
They need to be loaded onto the text encoder (T5 / CLIP). Affects Flux.1 LoRAs most
visibly. Queued in Backlog.
*Resolved: 2026-04-22 — `_apply_te_lora()` added to fallback + conversion paths in `eric_qwen_edit_lora.py`.*

**Skip unresolvable Kohya keys in `decode_kohya_to_bfl()`** *(2026-04-21)*
Keys like `distilled_guidance_layer` cannot be mapped and currently cause errors or
leave garbage in the converted dict. Should gracefully skip with a warning.
Queued in Backlog.
*Resolved: 2026-04-22 — filter against `named_parameters()` in `load_converted_lora` before `pipe.load_lora_weights`.*

---

## CFG Routing

### [Code] SD3 `max_sequence_length` not forwarded in CFG routing
- **Location:** `nodes/eric_diffusion_generate.py:214`, `comfyless/generate.py` sdxl/sd3/sd1/zimage block
- **Observed:** 2026-04-23 during zimage family support slice
- **Why not now:** SD3 default (256) works; only matters if longer prompts are needed; separate slice.
- **Suggested fix:** Pass `max_sequence_length` in the sdxl/sd3/sd1/zimage block (after checking `sig.parameters` like auraflow does).
- **Priority:** Low

### [Security] Missing §12 security review for comfyless Unix socket IPC server
- **Location:** `comfyless/server.py` — full IPC server using Unix sockets
- **Observed:** 2026-04-23 governance review (§12 trigger: IPC)
- **Why not now:** Server is working and single-user; no immediate threat model. Review should have happened before it shipped.
- **Suggested fix:** Write `docs/security/review-comfyless-server-<date>.md` and ADR before the next non-trivial change to `server.py`.
- **Trigger:** Any code change to `comfyless/server.py`.
- **Priority:** Medium
- **Resolved: 2026-04-23** — review written to `docs/security/review-comfyless-server-2026-04-23.md` (mirror: Obsidian `Security/Review-2026-04-23-Comfyless-Server.md`). Conclusion: acceptable for single-user threat model; 3 MEDIUM findings queued as follow-up hardening slice (see entry below). Network transport and `--json` bridge remain Red Zone triggers for a fresh ADR + review when they land.

### [Security] Missing §12 security review for `resolve_hf_path` (caller-supplied model weight loading)
- **Location:** `nodes/eric_diffusion_utils.py` `resolve_hf_path()`, called from all 5 loader nodes
- **Observed:** 2026-04-23 governance review (§12 trigger: loading model weights from caller-supplied paths)
- **Why not now:** Shipped without a review; function is straightforward (HF cache lookup + optional download). No known exploit path in current single-user context.
- **Suggested fix:** Write `docs/security/review-resolve-hf-path-<date>.md` before the next change that touches path resolution or download behaviour.
- **Trigger:** Any change to `resolve_hf_path` or the `allow_hf_download` flow.
- **Priority:** Medium
- **Resolved: 2026-04-23** — review written to `docs/security/review-resolve-hf-path-2026-04-23.md` (mirror: Obsidian `Security/Review-2026-04-23-Resolve-HF-Path.md`). Conclusion: sound fail-closed resolver; `trust_remote_code` absent codebase-wide (verified). 3 MEDIUM findings queued as follow-up hardening slice (see entry below). LLM-agent bridge promotes these to HIGH when it lands.

### [Security] comfyless server hardening — follow-up from §12 review (2026-04-23)
- **Location:** `comfyless/server.py`
- **Observed:** 2026-04-23 security review (`docs/security/review-comfyless-server-2026-04-23.md`)
- **Why not now:** Acceptable for single-user desktop threat model; review recommends fixes ride with the next server-touching commit, not as a standalone change.
- **Suggested fix:** (Finding #1) add `MAX_FRAME = 1 MiB` cap and `conn.settimeout(5.0)` in `_recv`; (Finding #2) verify/enforce 0700 mode + uid on `/tmp/comfyless-$UID/` after `mkdir`; (Finding #8) reject non-absolute model/component/LoRA paths in `_check_paths` to remove reliance on `realpath`'s relative-path behaviour.
- **Trigger:** Next non-trivial commit that touches `comfyless/server.py`.
- **Priority:** Medium
- **Resolved: 2026-04-23** — hardening slice applied. Findings 1/2/8 closed per recommendations. Re-review surfaced a new MEDIUM (H-2: embedded-NUL path crashes accept loop via `realpath` ValueError) which was also closed in the same slice via NUL rejection in `_validate_request` for path-shaped fields. Finding 1 residual (per-call vs wall-clock timeout) also closed with `time.monotonic()` deadline. Two new LOW items (H-1 symlink check, H-3 lora weight type) deferred — see entries below. Review: `docs/security/review-comfyless-server-hardening-2026-04-23.md`.

### [Security] `_socket_dir` should use `lstat()` to reject pre-planted symlink (H-1)
- **Location:** `comfyless/server.py` `_socket_dir`
- **Observed:** 2026-04-23 re-review of hardening slice (`docs/security/review-comfyless-server-hardening-2026-04-23.md`)
- **Why not now:** Same-uid threat model; `mkdir(exist_ok=True) + stat()` currently follows a pre-planted symlink. Low impact on solo desktop, MEDIUM if shared-machine deployment ever happens.
- **Suggested fix:** call `d.lstat()` first and reject if `stat.S_ISLNK(st.st_mode)` before the existing uid/mode checks on `d.stat()`. Two-line change.
- **Trigger:** Next non-trivial commit touching `comfyless/server.py` or any scope change to shared-machine deployment.
- **Priority:** Low (Medium on shared-machine scope change)

### [Code] `loras[i]["weight"]` not type-checked in `_validate_request` (H-3)
- **Location:** `comfyless/server.py` `_validate_request` loras loop
- **Observed:** 2026-04-23 re-review of hardening slice
- **Why not now:** No exploit path; malformed weight is caught by the outer `except` around LoRA load. Inconsistent with the rest of the schema's strict type-checking.
- **Suggested fix:** `if "weight" in lora and not isinstance(lora["weight"], (int, float)): return "loras[{i}].weight: expected float"` alongside the existing path check.
- **Trigger:** Next server-touching commit or schema tidy pass.
- **Priority:** Low

### [Security] resolve_hf_path hardening — follow-up from §12 review (2026-04-23)
- **Location:** `nodes/eric_diffusion_utils.py` (`resolve_hf_path`, `_is_hf_repo_id`) + `comfyless/generate.py` `_run_cli_mode`
- **Observed:** 2026-04-23 security review (`docs/security/review-resolve-hf-path-2026-04-23.md`)
- **Why not now:** Not exploitable-as-is; `allow_hf_download` defaults to False and `trust_remote_code` is absent. Hardening blocks the PNG-sidecar social-engineering path before the LLM-agent bridge lands.
- **Suggested fix:** (Finding #1) reject `foo/..` and `foo/.` in `_is_hf_repo_id`; (Finding #2) emit loud stderr warning naming the exact repo when `allow_hf_download=True` hits the network; (Finding #3) symmetric warning in `_run_cli_mode` when a `--params`-derived model value is an HF repo ID under `--allow-hf-download`.
- **Trigger:** Before wiring the `--json` LLM-agent bridge, or on next change to `resolve_hf_path`.
- **Priority:** Medium
- **Resolved: 2026-04-23** — hardening slice applied. All three findings closed per recommendations. Re-review surfaced no new MEDIUM/HIGH issues; LOW/INFO items (wider PNG warning covering component paths, `_is_hf_repo_id` public rename, `--override` wording precision) queued for the LLM-agent-bridge slice when the threat model elevates. Review: `docs/security/review-resolve-hf-path-hardening-2026-04-23.md`.

### [Security] Symmetric `resolve_hf_path` on Qwen component loaders
- **Location:** `nodes/eric_qwen_edit_component_loader.py`, `nodes/eric_qwen_image_component_loader.py`
- **Observed:** 2026-04-23 security review of `resolve_hf_path` (Out-of-scope section)
- **Why not now:** These loaders predate the HF-resolution work and currently fail-closed on repo IDs via `local_files_only=True`. Behaviour-change to add resolution; inconsistency not exploitable.
- **Suggested fix:** Thread `allow_hf_download` BOOLEAN + `resolve_hf_path` calls into both loaders to match the generic component loader pattern.
- **Trigger:** Next meaningful edit to either Qwen component loader.
- **Priority:** Low

### [Code] `--override key=value` syntax inconsistent with `--param <value>` CLI convention
- **Location:** `comfyless/generate.py` `_apply_overrides()`, argparse setup
- **Observed:** 2026-04-23 during `--params` image path work
- **Why not now:** Breaking change to the `--override` interface; needs a deprecation period or a single coordinated rename. The `--json` bridge mode also uses `--override` so both surfaces must change together.
- **Suggested fix:** Decide on one convention (`param=value` or `--param value`) and apply consistently. User preference is `param=value`; `--override` is the odd one out. Alternatively, accept both syntaxes in `_apply_overrides()` as a transition path.
- **Priority:** Low
