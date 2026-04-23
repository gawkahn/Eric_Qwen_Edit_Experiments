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
