# Tech Debt

Items here are conscious deferrals — known gaps with a recorded reason for not fixing now.
Format: **Item** — why deferred, what triggers revisiting.

---

## Security

**MCP server: HF-cache-hit vs cache-miss is observably distinguishable to the agent** *(2026-05-17)*
`comfyless/mcp_server.py:_handle_generate` step 4 (HF resolution) runs
BEFORE step 5 (path allowlist). An agent probing HF repo IDs can distinguish
three outcomes by error class — cached + inside `--model-base` (success);
cached + outside `--model-base` (`PathAllowlist`); not cached (`HFCacheMiss`).
The repo ID itself is suppressed from both error strings (colon-split + `from
None` cause-chain), but the per-class signal lets the agent enumerate which
HF repos are present in the local cache. Bounded by same-uid stdio MCP threat
model (the agent already runs generation; cache contents are not an
independent secret). Surfaced by slice-1 step-2 security-auditor F3 LOW.
Trigger to revisit: HTTP-transport ADR, multi-tenant agent surface, or any
threat-model change that elevates the agent from trusted-to-generate-on-this-
host. Fix shape: unify the agent-facing error class to a generic
"validation failed: model not available" for both miss-and-outside-base
cases while retaining the finer-grained `HFCacheMiss`/`PathAllowlist`
classes on the stderr audit line for operator visibility.
See `docs/security/review-slice-1-mcp-step2-2026-05-17.md` F3.
**2026-05-23 update:** ADR-015 (`docs/decisions/ADR-015-mcp-catalog-reference-resolution.md`) commits to a uniform agent-facing error class across **all** reference-resolution failures (catalog miss, catalog hit whose path moved, HF-cache miss, request-time `_within` failure) with fine-grained cause on stderr only — closing this oracle by construction. Trigger met; mark **Resolved: <date> — slice 3 shipped uniform-error contract** when ADR-015 slice 3 lands.
**Resolved: 2026-06-02** — ADR-015 slice 3 (step 1 resolver + step 2 `_handle_generate` migration) shipped the uniform agent-facing reference error (`"reference not available"`) for ALL causes, with the fine cause (`UnknownName`/`KindMismatch`/`MalformedReference`/`PathMoved`/`WithinFailure`) on the stderr audit line only. Request-time HF-cache-miss is now indistinguishable from any other failure (subsumed by `PathMoved`). Proof: `test_mcp_server.py` keystone N5 (four-frame byte-equality) + N3. Reviews: `docs/security/review-slice-3-step1-2026-06-02.md`, `docs/security/review-slice-3-step2-2026-06-02.md`.

**MCP server: daemon delegation deferred** *(2026-05-17)*
`comfyless/mcp_server.py:_call_tool_impl` (slice 1 step 2) runs generation
in-process only; it does NOT auto-detect and delegate to the running
Unix-socket daemon as ADR-011 §1 describes ("delegates to the running
daemon... fall back to in-process when no socket is present"). Reason:
the daemon's existing `_save_with_metadata` embeds full paths into PNG
tEXt chunks, which violates slice-1 invariant 12 (MCP-returned PNGs must
carry basenames only). Implementing daemon-side MCP awareness requires
either (a) a `caller=mcp` field in the daemon's wire protocol so the
daemon applies the redaction map server-side, OR (b) the MCP server
reading the daemon's output PNG and rewriting the tEXt chunk after the
daemon finishes. Both expand scope; user approved deferring to a future
slice + ADR-011 Changelog amendment. Trade-off: MCP-driven generations
do not share the daemon's cross-call model cache, so every MCP `generate`
reloads from disk. Bounded — LLM-driven workflows tend to be fewer larger
calls vs. interactive CLI use. Trigger to revisit: any user-reported
latency complaint about MCP-driven generations, OR an LLM-judge / auto-
refinement loop being wired (those workflows are call-count-heavy and
would benefit from caching). See ADR-011 §1, slice-1 invariant 12,
`docs/security/review-slice-1-mcp-step1-2026-05-16.md`.

**MCP server: TOCTOU between `realpath` and `_within` in `_validate_startup_args`** *(2026-05-16)*
`comfyless/mcp_server.py:_validate_startup_args` calls `os.path.realpath(default_model)` once,
then `_within(resolved_default, resolved_base)` which calls `realpath` again internally
(`comfyless/server.py:160-161`). An attacker with write access inside `--model-base` could
swap the symlink target between the two calls so the second `realpath` sees a different
post-resolution path than the first. Bounded by the same-uid stdio MCP threat model (the
attacker already has the daemon's privileges; ADR-011 §7 defers anything beyond same-uid).
Surfaced by slice-1 step-1 security-auditor F1 LOW. Trigger to revisit: any commit that
weakens the same-uid assumption — HTTP transport ADR, multi-tenant deployment, network
exposure. Fix shape: cache `os.path.realpath(p)` once and pass the cached value to a `_within`
form that does NOT re-resolve. See docs/security/review-slice-1-mcp-step1-2026-05-16.md F1.

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

**No automated test for SGM-prefix + missing-key loader path** *(2026-04-24)*
The `_load_stripped_in_memory` helper inside `_load_single_weights`
(`nodes/eric_diffusion_utils.py`) has a known dtype-divergence trap when a
checkpoint produces `missing_keys` under `load_state_dict(strict=False,
assign=True)`. Today's fix raises and falls through to the temp-file path on
that condition, but a regression here would only surface in live use with a
real Flux/Klein finetune fixture (multi-GB safetensors with the
`model.diffusion_model.` SGM prefix and an architecture mismatch vs the
diffusers config). The bug from `eb571a8` (2026-04-21) reached the user
because no test exercised this path. Trigger to revisit: any change to
`_load_stripped_in_memory`, `_peek_dominant_prefix` thresholds, or the
diffusers pin. Possible fix shapes: a tiny synthetic state-dict fixture that
omits known keys + a stub config; OR a smoke-only integration test gated by
an env var pointing at a real fixture path.

**Comfyless server failed-load / OOM-cascade recovery (regression)** *(2026-05-04)*
A failed model load (commonly OOM, also corrupted-checkpoint and similar)
sometimes leaves the daemon in a partial state where the *unload* path also
fails, which then makes the *next* load fail with OOM (residual VRAM not freed).
**Recurred 2026-05-01 despite belief the original fix landed in or before the
2026-04-23 hardening slice** — first investigation step is to identify what was
*thought* fixed vs. what *actually* shipped, before designing the next fix.
Likely touches `comfyless/server.py` model-cache eviction path. Stdio v1 MCP
(slice 1) inherits parent-process recovery semantics so a daemon crash is
observable and restartable by the client; this is daemon-hygiene affecting
regular comfyless use. **Hard precondition before any non-stdio MCP transport
ADR (HTTP/SSE) drafts** — same gate as the runtime-core cluster's HTTP-readiness
preconditions. Suggested fix shape (post-investigation): treat any load
exception as "eviction needed" and free the partial state before the next
request; possibly process-isolation per generation if VRAM accounting can't be
made reliable. May warrant its own ADR if the fix is non-trivial. Cross-refs
the existing "Daemon request rate limiting / VRAM exhaustion" entry above
(that one is preventive, this one is recovery). See Backlog Queued entry
"Comfyless server failed-load / OOM-cascade resilience" for the architectural
gate framing.

**Client-side recv timeout is a flat 600s ceiling** *(2026-04-24)*
`_CLIENT_RECV_TIMEOUT_SEC = 600.0` in `comfyless/server.py` is a compile-time
constant. Realistic tail: a 50-MP Qwen-Image-2512 run at 50 steps with tile-VAE
decode can approach 600s. If it trips, the user sees `request timed out before
newline` on the client even though the image is being saved on the server side.
Fix shape: either raise to 1800s or expose via env var `COMFYLESS_CLIENT_RECV_TIMEOUT`.
Trigger: first report of the ceiling tripping, OR first commit that adds a model
family with known-longer generation times. See
`docs/security/review-server-timeout-brokenpipe-2026-04-24.md` (LOW finding).

**Catalog-name allowlist intentionally narrow** *(2026-05-25)*
`comfyless/catalog.py:_FORBIDDEN_NAME_CHARS` rejects C0/C1 controls,
zero-width chars + LRM/RLM (U+200B-200F), bidi overrides (U+202A-202E),
LINE/PARAGRAPH SEPARATOR (U+2028-2029), and bidi isolates (U+2066-2069)
at catalog-build time. Codepoints NOT in the set that are plausible
agent-UX confusables: BOM / ZWNBSP (U+FEFF), SOFT HYPHEN (U+00AD),
MONGOLIAN VOWEL SEPARATOR (U+180E), INTERLINEAR ANNOTATION (U+FFF9-
U+FFFB). Under the same-uid stdio trust model, the omission is
aesthetic/UX (two visually-identical names mapping to distinct entries),
not an exploit surface — an adversary at the same uid can already plant
anything. Trigger to revisit: first slice-3 agent UX report of "two
catalog entries look identical to me but resolve differently"; or any
threat-model change to multi-tenant MCP transport. Fix shape: extend
the regex one-liner with `\ufeff\u00ad\u180e\ufff9-\ufffb` (no behaviour
on existing valid names). Surfaced by slice-2 step-4 security-auditor
INFO (2026-05-25). See
`docs/security/review-slice-2-step4-2026-05-25.md`.

---

## Dependencies

**Hash-locked installs** *(2026-04-21)*
`pip install --require-hashes` with a generated lock file defends against the
"same version, different bytes" PyPI backend-compromise threat that plain `==` pins
don't catch. Only worth doing from project start — retrofitting requires hashes for
all 100–200 transitive deps, with per-platform lock files for wheels.
Trigger: any new greenfield project in this space; do it from day one there.
See `pyproject.toml` comments and `project_dependency_pin_strategy.md` memory.

**Krea-2 runtime blocked on a diffusers release** *(2026-06-25)*
Krea-2 (`Krea-2-Raw`, `Krea-2-Turbo`) support landed code-first
(`docs/vision/slice-krea2-support.md`): family detection, `FAMILY_DEFAULTS`,
CFG routing, catalog classification, and MCP parity all work on the current
pin. But `Krea2Pipeline` exists **only on diffusers `main`** — no PyPI
release ships it (0.38.0 verified without it; transformers 5.5.3 already has
`Qwen3VLModel` and diffusers 0.37.1 already has `AutoencoderKLQwenImage`, so
the pipeline + `Krea2Transformer2DModel` are the only missing pieces).
Per decision (2026-06-25): no nightly/git pin (§11 exact-pin) and no
vendoring. `generate` therefore raises the existing "upgrade diffusers"
`ValueError` at load until a tagged release exports `Krea2Pipeline`.
Why not now: bumping diffusers to `main` violates §11 and would diverge from
ComfyUI's bundled stack (ADR-013 torch divergence concerns); the dep bump is
its own ADR'd slice.
Trigger: a tagged diffusers release exporting `Krea2Pipeline` (watch 0.39.0).
When met: a separate slice pins the new diffusers (+ matching torchvision if
needed), runs `uv lock`, updates `requirements.txt` + `pyproject.toml`
together, and smoke-tests Raw/Turbo generation.

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

### [Code] Krea-2-Turbo `mu` / timestep-shift (1.15) not exposed
- **Location:** `comfyless/family_defaults.py` (`krea-turbo`), `comfyless/generate.py` `_build_call_kwargs` krea branch
- **Observed:** 2026-06-25 during Krea-2 support slice
- **Why not now:** Krea's CLI recommends Turbo at `mu=1.15` (a FlowMatchEuler timestep-shift), but `COMFYLESS_SCHEMA` has no shift/`mu` knob; diffusers' `Krea2Pipeline` computes a dynamic shift from resolution by default, so Turbo still runs without it. Out of scope for the code-first slice.
- **Suggested fix:** Add an optional `shift`/`mu` schema key, forward it in the krea CFG branch after checking `inspect.signature(pipe.__call__)`, and set `krea-turbo` default to 1.15. Verify against the actual diffusers `Krea2Pipeline.__call__` signature once a release ships it.
- **Trigger:** Turbo output quality shortfall traced to timestep shift, OR the diffusers dep slice lands and the real `__call__` signature is inspectable.
- **Priority:** Low

### [Code] ComfyUI node-side `_build_call_kwargs` lacks krea routing
- **Location:** `nodes/eric_diffusion_generate.py` (the ComfyUI-node mirror of `comfyless/generate.py:_build_call_kwargs`)
- **Observed:** 2026-06-25 during Krea-2 support slice (code-reviewer note)
- **Why not now:** The Krea-2 slice was scoped to the comfyless surfaces (CLI/daemon/MCP) only; the ComfyUI node mirror was intentionally left untouched, so it has no `krea`/`krea-turbo` branch and would fall through to the unknown-family introspection path. Conscious deferral, not silent drift.
- **Suggested fix:** Add the same `("krea","krea-turbo")` guidance_scale branch to the node-side `_build_call_kwargs` if/when Krea-2 is exercised through the ComfyUI node UI.
- **Trigger:** Krea-2 used via the ComfyUI Eric Diffusion Generate node, OR the next deliberate change to `nodes/eric_diffusion_generate.py` CFG routing.
- **Priority:** Low

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
- **Pending closure by:** *(2026-05-04)* validator harmonization slice — `docs/vision/slice-machine-boundary-validator.md`, ADR-012 forthcoming. Vision invariant 5 unifies LoRA-weight validation across machine-boundary call sites; invariant 4 makes `weight` canonical-`float` with safe `int → float` cast at the validator boundary, structurally enforcing the type-check this entry calls for. Original suggested-fix shape is superseded by the validator slice's per-LoRA validation helper. Mark `Resolved:` here (per global §12) when the validator slice ships.
- **Resolved: 2026-05-16** — closed by ADR-012 validator slice (commits `58ef335`..`57bd650`). `validate_lora_entry` in `comfyless/params_validation.py` enforces both `path` and `weight` as required fields; canonical-`float` typing rejects `bool` and rejects `str`/`None`; safe `int → float` cast applied at the validator boundary. Called from `comfyless/server.py:_validate_request` (step 3) AND `comfyless/generate.py:_validate_iterate_value` lora_stack branch (step 4) — single source of truth across both machine-boundary call sites. The 2026-04-23 §12 review's finding 9 (bool-as-int subtype loophole) is also closed by the same slice — the canonical validator rejects `bool` BEFORE the `int` accept branch in `_KIND_INT` and `_KIND_FLOAT`. Cross-site parity proved by the N18 grid (38 fixtures) in `test_machine_boundary_validator.py`. Step-3 security review at `docs/security/review-validator-slice-step3-2026-05-16.md`.

### [Security] Server path-error audit log drops `prompt` only, not `negative_prompt`
- **Location:** `comfyless/server.py:302-303` (in `_handle_connection`'s path-error branch)
- **Observed:** 2026-05-16 step-3 security audit of the validator slice (`docs/security/review-validator-slice-step3-2026-05-16.md`, finding 8)
- **Why not now:** Pre-existing — not introduced by the validator slice; out of step-3 edit scope. ADR-011 §3b (round-1 fold-in F-4, 2026-04-28) committed to dropping BOTH `prompt` AND `negative_prompt` from machine-boundary audit lines; this `_handle_connection` path-error log line was written before ADR-011 landed and drops only `prompt`.
- **Suggested fix:** in the `redacted = {...}` dict comprehension at line 302, exclude `negative_prompt` alongside `prompt`. One-line change: `redacted = {k: v for k, v in req.items() if k not in ("prompt", "negative_prompt")}`. Add a regression test asserting both keys are absent from the audit-line content.
- **Trigger:** Next `_handle_connection`-touching commit; or before ADR-011 slice 1 (MCP `generate` tool) lands and the audit pattern becomes a published contract.
- **Priority:** Medium

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

---

## Stable Cascade

### [Code] Cascade decoder→vqgan dtype-mismatch hook
- **Location:** `comfyless/cascade.py` — `build_pipelines` vae_dtype block; `run_one` decoder call
- **Observed:** 2026-04-27 during cascade dtype-boundary debugging (commits `e791462`, `dd196c1`)
- **Why not now:** The walked path defaults all three stages to `bf16` (commit `dd196c1`), which makes every internal boundary same-dtype and avoids the issue. When a user explicitly sets `vae_dtype != decoder_dtype` in the JSON config, today's behavior is: emit a stderr advisory, recast `decoder_pipe.vqgan` to the requested dtype, and rely on the user accepting the risk that the deprecated `StableCascadeDecoderPipeline.__call__` may still emit "Input type X / bias type Y should be the same" mid-forward (the cast from decoder latents to vqgan input is *internal* to the decoder pipeline call and not interceptable from `run_one`). For the stated single-user threat model and SAI's recommended recipes, this is acceptable.
- **Suggested fix:** Register a `forward_pre_hook` on `decoder_pipe.vqgan.decode` (or wrap the bound method) that casts the incoming latents tensor to the vqgan's parameter dtype before the call. ~10 lines + 2 tests (one mismatched-dtype config that today warns-and-runs, one that today crashes; both should pass after the hook).
- **Trigger:** First user/test that requires a non-bf16 decoder + non-matching vae combo, OR any change to `comfyless/cascade.py`'s dtype handling block that revisits this code.
- **Priority:** Low
- **References:** ADR-010 second amendment (Changelog); cascade.py inline comment near `_DTYPE_DEFAULTS`.

### [Out-of-scope features for Cascade] *(reference only — not active debt)*
ADR-010's "Deferred / Out of Scope" section formally declares the following non-goals for v1; revisit only if empirical use surfaces demand:
- LoRA support for the Cascade prior or decoder (~40 community LoRAs vs. thousands for SDXL/Flux makes the integration cost disproportionate).
- Image-variation conditioning via `image=` to the prior (`feature_extractor` + `image_encoder` are deliberately left at `None`).
- Stage A weight-swap UI (the JSON field exists; no published Stage A variants are known to be worth swapping).
- ControlNet variants (the SAI repo carries a `controlnet/` directory; not wired).
- Lite-variant filename detection or warning (permissive, doc-only policy — by design).
- Other `--iterate` axes beyond `prompt` and `seed` for cascade dispatch (cfg, model, transformer, etc. are JSON-config concerns, not iterate axes — ADR-010 amendment 3).
