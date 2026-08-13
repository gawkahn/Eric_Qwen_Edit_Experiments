# Tech Debt

Items here are conscious deferrals — known gaps with a recorded reason for not fixing now.
Format: **Item** — why deferred, what triggers revisiting.

---

## ADR-035 reference-image edit

**qwen-edit foreground run does not arm ^C pause/resume (single ^C hard-aborts)** *(2026-07-21)*
Slice-3 `code-reviewer` (Fable) finding 7 (minor). The reference-image edit
branch in `generate()` calls `_run_qwen_edit_refs` → `generate_qwen_edit`
directly, bypassing the `sigint_pause` context the text2img path wraps its
`pipe.__call__` in (slice PAUSE). So a foreground edit loses pause-at-step and a
single ^C aborts instead of pausing. `generate_qwen_edit` accepts a `progress_cb`
that could drive the pause.
Why not now: out of ADR-035 slice-3 scope (execution wiring); pause is a UX
nicety, not correctness. Trigger: a later edit-UX slice, or whenever
`generate_qwen_edit`'s `progress_cb` is wired for progress reporting — do both
together. Fix: adapt `comfyless.pause.sigint_pause` to the manual loop via
`progress_cb`.

**`_REF_MODE_FLAGS[spec["mode"]]` KeyErrors on a malformed mode from a non-CLI caller** *(2026-07-21)*
Slice-3 `code-reviewer` (Fable) finding 8 (minor). `_run_qwen_edit_refs`
indexes `_REF_MODE_FLAGS` by `spec["mode"]` with no local guard. Safe today —
the only caller, the CLI, validates modes via `_validate_ref_image_specs`
before `generate()`. But slice 4 (daemon wire) and slice 5 (`--params` replay)
introduce callers whose `ref_images` specs are NOT CLI-validated.
Why not now: no unvalidated caller exists in slice 3; adding a redundant guard
now is dead code. Trigger: HARD precondition of BOTH slice 4 and slice 5 — each
must validate `ref_images[].mode ∈ {both,vl,ref}` at its own boundary before
forwarding to `generate()` (mirrors the mode check `_parse_ref_image` already
does for the CLI). Add to those slices' boundary checklists.
Partially resolved: 2026-07-21 — ADR-035 slice 4 closed the DAEMON-WIRE half.
`validate_ref_image_entry` (`comfyless/params_validation.py`), called from
`validate_machine_request`, rejects any `ref_images[].mode ∉ {both,vl,ref}` at
the canonical boundary before `generate()` runs (pinned in
`test_machine_boundary_validator.py`). The slice-5 `--params`-replay half is
still open — replay must validate `mode` at its own boundary before forwarding.

**Edit-run sidecar records ref provenance but `--params` replay silently drops it (until slice 5)** *(2026-07-21)*
Slice 3 records `ref_images` (path/mode/sha256) in the edit sidecar for a
truthful record (code review F1, resolved early), but a `--params` replay of
that sidecar drops `ref_images` via `_SKIP_SIDECAR_KEYS` and regenerates WITHOUT
references — with no warning, because slice 5's replay-trust treatment (the loud
"recorded ref dropped — re-supply --ref-image", moved-file / hash-mismatch
notices, outside-roots refusal) is not yet built. This is not a regression
(pre-slice-3 there was nothing to drop) but the silent half remains.
Why not now: replay TRUST is ADR-035 decision 7 / slice 5 by design; only the
safe RECORDING half was pulled into slice 3. Trigger: slice 5 — remove
`ref_images` from `_SKIP_SIDECAR_KEYS`, add the replay-side warnings + F4 echo +
outside-roots refusal. Fix: as specified in ADR-035 decision 7.

---

## Security

**MCP: `refiner_path` not covered by output-metadata basename redaction** *(2026-07-11)*
The Hunyuan-Image refiner writes an absolute `refiner_path` into the PNG/sidecar
metadata (`comfyless/generate.py`), consistent with the pre-existing absolute
`model`/`transformer_path`/`vae_path`/`text_encoder_path` entries. The caller
supplied `refiner_path` itself, so this is not a new cross-boundary leak on the
CLI/daemon path. The MCP `generate` handler does NOT thread refiner today
(`test_hunyuan` Inv 12), so nothing is exposed to an untrusted agent yet.
**Why not now:** no MCP refiner surface exists; adding a redaction hook for a key
that never crosses the MCP boundary would be dead code. **Trigger:** the slice
that threads refiner through the MCP `generate` handler — fold `refiner_path`
into the same basename redaction the other path keys receive. Surfaced by the
`security-auditor` re-apply review INFO-2, 2026-07-11
(`docs/security/review-hunyuan-refiner-reapply-2026-07-11.md`).

**extract_params: free-string fields (`model_family`/`prompt`/`negative_prompt`) echo verbatim** *(2026-07-09)*
Both `_render_extracted_cascade_params` (step 4d) and the core-step
`_render_extracted_params` (`comfyless/mcp_server.py:382-383`) re-emit these
free-text fields verbatim (family with a truthy-string guard; prompts with an
`isinstance str` guard). A crafted sidecar can therefore place an abs-path-shaped
string in one of them and have it echo back in the response.
**Why not now:** these are the same-uid caller's OWN sidecar bytes (no real
resolved path or server secret is disclosed — the agent could read the file
directly, ADR-015 §3), and they are semantically free text / a family label, not
path-typed reference fields. The behavior is identical in the already-reviewed
non-cascade renderer — 4d did not introduce it, and fixing only the cascade path
would create an asymmetry. **Trigger:** a shared bound/validation across BOTH
renderers (one slice), or any threat-model change that elevates the agent above
trusted-same-uid (HTTP transport, multi-tenant). Surfaced by step-4d
code-reviewer + security-auditor INFO, 2026-07-09
(`docs/security/review-slice-4d-cascade-2026-07-09.md`).

**extract_params: sidecar parsed with unbounded `json.load` (no size ceiling)** *(2026-07-09)*
`_handle_extract_params` (`comfyless/mcp_server.py`) reads the gated `.json`
sidecar with `json.load` and no size limit, so a same-uid actor who drops a very
large `.json` under `--output-dir` could cause a transient memory spike.
**Why not now:** same-uid, requires write access to the output dir, and it is a
property of the whole slice-4 read path rather than the 4d cascade branch;
no availability guarantee is currently in the threat model. **Trigger:** a
resource-bound pass over the MCP read surface, or any move toward an untrusted /
networked caller. Surfaced by step-4d security-auditor INFO, 2026-07-09
(`docs/security/review-slice-4d-cascade-2026-07-09.md`).

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

**LoRA audit: unbounded captured stdout per dry-load file** *(2026-05-28)*
`scripts/lora_audit.py:_dry_load_per_base` wraps each `load_lora_with_key_fix`
call in `contextlib.redirect_stdout(io.StringIO())` and parses the buffer for
`applied=(\d+)`. The `StringIO` is GC'd after the iteration, but during a
single file's load it grows unbounded — a pathological loader that prints in a
loop would consume RAM until OOM. Today's loader does not print in a loop and
there is no caller-controlled path into the loader's print behavior, so the
risk is residual. Surfaced by security-auditor S2 review L-2.
Trigger: any commit that adds caller-controlled formatting strings into the
loader's logging path, OR any reproducible loader regression that prints
unbounded output. Fix shape: cap StringIO writes via a custom file-like wrapper
that raises after N bytes; record `dry_load.reason = "loader_log_overflow"`.

**LoRA audit: `test_dry_load_integration.py` SKIP exits 0** *(2026-05-28)*
The gated E2E test returns exit code 0 on SKIP. There is no CI today, so a
"silently skipped" outcome is not detectable, but the moment CI lands the SKIP
path will report green for runs that should have been gated as "not exercised."
Fix shape: switch SKIP exits to 77 (autoconf convention) and have the CI
runner treat 77 as a distinct status. Surfaced by code-reviewer S2 review L-3.
Trigger: any CI being wired for this repo, OR any sibling test gaining the
same gating pattern. Fix is a 2-line change in `test_dry_load_integration.py`
plus a CI runner update.

**LoRA audit: `_passes_scan_containment` embeds absolute escape-target paths in manifest warnings** *(2026-06-27)*
`scripts/lora_audit.py:_passes_scan_containment` (S1) records the absolute
realpath of a symlink that escapes `audit_root` into the
`excluded_symlink_escape` warning's `detail` field (`f"realpath {real} not
under audit_root"`). When the escape target is outside `audit_root` (e.g. a
swapped-symlink target like `/home/gawkahn/.ssh`), that absolute path ships in
the manifest `warnings[]` — the same F-8 incremental-disclosure leak class the
project closed for argv. Bounded by the single-user same-uid threat model (SA
S4 review LOW: "no action required" under that model). S4's new `_safe_unlink`
already sanitizes its equivalent `/proc/self/fd` escape-rejection detail to a
fixed token; this entry tracks the **S1 carry-over** so the two paths can be
unified rather than left asymmetric. Trigger to revisit: the F-10 risk-trigger
(LLM-agent / remote caller supplies `--audit-root`, re-classifying to Red Zone
and making manifest-sharing real), OR any decision to share manifests off the
single-user host. Fix shape: replace `real`/`real_parent` in the detail string
with a fixed `"escaped audit_root"` token (the `file` field already identifies
the entry). Surfaced by security-auditor S4 review LOW
(`docs/security/review-lora-audit-s4-2026-06-27.md`).

**LoRA audit: convert write-path lacks O_NOFOLLOW/dir-fd intermediate-symlink narrowing** *(2026-06-02)*
`scripts/lora_audit.py:_convert_one` (S3 `--convert`) writes the converted
sibling via `target_path.parent.mkdir(parents=True, exist_ok=True)` +
`safetensors.torch.save_file(tmp)` + `os.replace(tmp, target)`. The output path
is containment-checked once with `target_path.resolve().relative_to(base_dir)`
*before* the `mkdir`/write, but the write itself does not use the
`O_NOFOLLOW`/dir-fd-relative narrowing that the *read* path (`_open_no_follow`)
and the planned `--delete` path (ADR §9) apply. A same-uid attacker who swaps an
intermediate directory under an out-of-`audit_root` `--output-dir` for a symlink
in the TOCTOU window between the resolve-check and `mkdir`/`os.replace` can
redirect the write outside the validated base. This is within the ADR §6/§8
accepted residual (the same attacker can write there directly; the tool grants
no new capability) and `--output-dir` defaults inside `audit_root`, so the MVP
posture holds. The convert source read (`_load_state_dict(source_path)`) is the
read-side analogue: unlike the S2 dry-load path (which re-runs
`_passes_scan_containment` per file, M-1), `_convert_one` does not re-check
source containment before re-reading — same accepted same-uid residual.
Surfaced by security-auditor S3 review (MEDIUM, accepted; no CHANGES REQUIRED)
and code-reviewer S3 review (LOW-3).
Why not now: out of S3's declared edit scope; same-uid TOCTOU is below the MVP
threat floor; closing it is a non-trivial dir-fd write rewrite that belongs with
the S4 delete-path hardening (which already adopts the dir-fd pattern).
Fix shape: open the validated target parent with
`os.open(parent, O_NOFOLLOW|O_DIRECTORY|O_CLOEXEC)`, re-check its
`/proc/self/fd` realpath against `base_dir`, and `os.replace` dir-fd-relative;
mirror the ADR §9 `safe_unlink` shape. Optionally re-run
`_passes_scan_containment` on the source before re-reading.
Trigger: the F-10 risk-trigger fires (LLM/remote caller supplies paths → whole
tool re-classifies to Red Zone, write-path narrowing becomes mandatory), OR S4
lands the dir-fd delete path (fold the convert write into the same pattern).
See `docs/security/review-lora-audit-s3-2026-06-02.md`.

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
Resolved: 2026-06-27 (krea-testing branch ONLY) — diffusers git-pinned to
`main` @ 29a59fd (0.39.0.dev0) via `[tool.uv.sources]`; safetensors bumped
0.7.0→0.8.0 (diffusers main requires >=0.8.0). §11 exact-pin satisfied by the
commit rev. Krea-2-Turbo (8 steps) and Krea-2-Raw (CFG + negative prompt)
both smoke-tested generating coherent images. This pin must NOT be merged to
`main` — `main` stays on the last tagged release until a PyPI build exports
`Krea2Pipeline` (still the trigger above for the real, mergeable bump). The
branch exists precisely to test Krea without an unreleased pin reaching main.
Resolved: 2026-07-03 (fully) — diffusers 0.39.0 released with Krea2 and
pinned (9e4a2f2): git pin + [tool.uv.sources] removed, manifests are
main-mergeable, 13 suites green on the release, Krea-2-Turbo live smoke OK.
The original trigger fired; nothing remains of this entry.

**Krea2 attention-backend pin — upstream workaround** *(2026-07-03)*
diffusers 0.39.0's `Krea2AttnProcessor` passes a bool key-padding mask
together with `enable_gqa=True` (48 q heads / 12 kv heads). PyTorch's fused
SDPA kernels reject that combination (flash: no arbitrary masks;
mem-efficient: no GQA), so backend auto-select silently falls back to MATH
and materializes the full S^2 attention matrix — measured ~91 GB of
transients at 2560x1440 (14912 tokens): instant OOM, quant-independent
(originally misattributed to --quant + direct-merge LoRAs). comfyless works
around it in `_pin_krea_attention_backend` (generate.py): krea/krea-turbo
transformers get `set_attention_backend("_native_cudnn")` — cuDNN runs the
same shapes fused at ~0.17 GB; verified end-to-end 2560x1440 x8 steps with
quant fp8 + 4-LoRA stack, peak 44.3 GB. A registry test in
test_params_schema.py catches an upstream rename of `_native_cudnn`.
Why not now (the real fix): the processor should expand kv heads or diffusers
should prefer cuDNN when a mask disqualifies flash — that's upstream's call.
Trigger: upstream fix to Krea2AttnProcessor (or attention auto-select)
lands in a pinned diffusers release → remove the pin helper + tests.
Related cosmetic issue, no action: under quantization_config diffusers skips
`_keep_in_fp32_modules`, so Krea2RMSNorm weights load bf16 and torch warns
"Mismatch dtype ... Cannot dispatch to fused implementation" once per run;
the norm computes fp32 and casts back either way — output unaffected.

**Krea2 distill LoRA bias deltas (.diff_b) not applied** *(2026-07-03)*
krea-native distillation LoRAs (krea2_turbo_lora_rank_64_bf16: 535 keys)
co-ship a `.diff_b` bias delta alongside each standalone module's lora_A/B
pair (img_in, final_layer.linear, time_embed.linear_{1,2}, time_mod_proj,
txt_in.linear_{1,2}). The conversion path applies the weight deltas (PEFT)
but no loader applies bias deltas — they are now dropped LOUDLY (was:
silent on the lora branch; the LoKR branch already warned). Measured on the
turbo file: |mean| ~5e-4, max 0.026 (time_mod_proj) — small corrections,
plausibly negligible next to the weight deltas that were the 2026-07-03
"terrible results" root cause (7 standalone modules unmapped; fixed).
Upstream diffusers 0.39.0's #14074 converter doesn't handle .diff_b either
(raises on leftovers).
Why not now: applying bias deltas means a hybrid path — PEFT for the
low-rank pairs plus a direct bias add with backup/restore + unload
semantics; its own slice if quality demands it.
Trigger: raw+turbo-LoRA output still visibly trails the dedicated Turbo
checkpoint (or ComfyUI's rendering of the same file) AFTER the standalone
weight-delta fix — that gap would implicate the biases.

**NVFP4 quantize-on-load blocked on a stable torch/torchao/mslk triad** *(2026-07-02)*
NVFP4 quantize-on-load for diffusion (ADR-019 slice A, nvfp4 half) is officially
supported and works — PyTorch's "Faster Diffusion on Blackwell" blog + diffusers
`TorchAoConfig` + the `sayakpaul/diffusers-blackwell-quants` recipes cover
QwenImage specifically (1.39–1.49× over bf16 at batch 1–8, 62→52 GB peak). But it
runs **only on the nightly triad**: torch `2.12.0.dev` + torchao `0.17.0.dev` +
mslk `2026.3.15`, all `cu130`. Our stack is torch `2.11.0` **stable**. The
2026-07-02 fp8 spike used stable torchao 0.17.0 installed `--no-deps`; the fast
`to_nvfp4` quantizer (routed through the MSLK kernel, pytorch/ao PR #4031) needs
the `mslk` wheel, which is nightly-only and version-locked to nightly torch/torchao.
**`mslk` IS a pip wheel** (`--pre --index-url .../whl/nightly/cu130`), NOT a
source build — so building/pinning our own kernel is unnecessary *and* wrong: there
is nothing to build, and self-maintaining it would just be nightly-equivalent code
under a different name, not a path to a stable pin.
Why not now: pinning the nightly triad drags torch 2.11 stable → 2.12 nightly across
the **whole** stack (every model + all 1412 tests), violates §11 exact-release-pin,
and crosses the same no-nightly-pin line as the Krea-2 entry above. This is the
identical situation, not a "maybe never" — the format/hardware are >1yr old but the
diffusers-eager + torchao + MSLK software integration is landing *now* on the
nightly→stable pipeline. Also gated on quality: nvfp4 needs `torch.compile` and
QwenImage is more quant-sensitive than Flux (LPIPS 0.41 vs 0.44) — a
measure-carefully feature, not a free win. Stable fp8 (proven, 40→20 GB) is the
near-term path.
Trigger: a stable torch ≥2.12/cu130 **and** matching stable torchao **and** stable
mslk all released (watch `pytorch/ao` releases + the stable cu130 wheel index). When
met: a slice pins the triad, `uv lock`, updates `requirements.txt` + `pyproject.toml`
together, adds `--quant nvfp4`, and smoke-tests QwenImage quality vs fp8.
See `docs/decisions/ADR-019-native-quantization-support.md` §Deferred,
`project_native_quant_support.md` memory.
Resolved: 2026-07-16 — trigger met on EASIER terms than recorded (no torch
2.12 bump needed): MSLK 1.1.x is the stable release line FOR torch 2.11.x,
so the current pins carry nvfp4. Side-session smoke on Blackwell sm_120
verified `NVFP4DynamicActivationNVFP4WeightConfig(use_triton_kernel=True)`
end-to-end on the repo's exact torch/torchao pins, incl. the negative
control (mslk absent → the known AssertionError). Slice NV: `mslk-cuda==
1.1.0` pinned, `--quant nvfp4` wired (Blackwell ≥10.0 gate, mslk
warn-fallback, weight-only family split, nvfp4-base direct-merge refusal —
security reqs 61-66). Live QUALITY smoke still owed — see the 2026-07-16
entry below.

**nvfp4 live quality smoke owed before recommending the mode** *(2026-07-16)*
Slice NV wired `--quant nvfp4` with unit coverage only — both GPUs were
busy on a long iterate run when it landed. Until a real-generation gate
passes, nvfp4 is wired-but-unvalidated: prefer `--quant fp8`. Expect
weights-only-style fiddling per family on first live runs (the fp8
rollout's pattern), plus these specific unknowns: does the quantize-on-load
inside `from_pretrained` run the mslk triton kernel on CUDA-resident
tensors (CPU-staged loads may need the non-triton path)? does the zimage
weight-only transfer hold for nvfp4? is dynamic-activation nvfp4 quality
acceptable on QwenImage (more quant-sensitive than Flux, LPIPS 0.41 vs
0.44)? The 1.39-1.49× throughput win also expects `torch.compile` — without
it nvfp4 is mainly a VRAM play (upstream: 62→52 GB).
Why not now: no free GPU.
Trigger: a GPU frees up. Gate per the handoff/vision: same prompt+seed
nvfp4 vs fp8 vs bf16 on QwenImage (detailed idiosyncratic prompts with
checkable anchors), plus a LoRA-via-PEFT run under nvfp4 (direct merge is
deliberately refused — reqs 61/65).
Resolved: 2026-07-17 — T1 gate ran (Grant live, nvfp4-smoke/RUNBOOK.md);
verdict: **functional, not recommended — fp8 stays the default**. Findings:
(1) first load crashed on the Qwen2.5-VL vision tower's non-/16 shapes →
shape screen (de837de); with it, CPU-staged quantize-on-load through the
mslk path works. (2) dyn-act nvfp4 on QwenImage is NOT acceptable —
pervasive granular noise — answering that unknown; per-mode weight-only
split added qwen-image (3dfc24d). (3) Weight-only nvfp4 on QwenImage is
usable but visibly lossy (blotchiness, smeared fine detail) and ~4.7×
SLOWER than bf16/fp8 (83.7s vs 17.8s @ 50 steps: no fused dequant kernel).
(4) Krea (dyn-act) is nvfp4's first real operating point: usable output
with drift, 15 vs 23 GB and 37 vs 58 s vs fp8. Still unrun from the
runbook: zimage weight-only transfer (T3), LoRA-under-nvfp4 PEFT arm (T4)
— revisit if nvfp4 ever graduates past niche-VRAM use; torch.compile and
calibrated offline nvfp4 remain the ADR-019 deferred paths to real parity.

**Daemon socket silently drops `quant` from hand-crafted clients** *(2026-07-02)*
Slice A registered `quant`/`quant_skip`/`quant_only` in `_RUNTIME_KIND`, so the
canonical validator type-accepts them on every machine boundary — including the
Unix-socket daemon, whose handler (`comfyless/server.py`) neither consumes nor
rejects them. The shipped CLI is covered (`generate.py` skips daemon delegation
when `--quant` is set, with a log line), but a hand-rolled socket client sending
`{"type": "generate", "quant": "fp8", ...}` validates cleanly and generates
UNQUANTIZED with no signal back to the caller. Surfaced by slice-A code-reviewer
F2 (MED).
Why not now: the fix is in `comfyless/server.py` — a project-mandated
security-review surface and an explicit STOP boundary for the slice-A autonomous
run (Vision §2). Rejecting or supporting quant at the daemon dispatch entry is
the same slice as wiring quant through the daemon protocol + cache key, which
needs its own `security-auditor` pass.
Trigger: the "quant over daemon" slice (protocol + cache key + explicit
reject-or-support at dispatch), OR any report of a socket client using quant.
Fix shape: daemon dispatch rejects `quant != "none"` with an explicit
"daemon does not support quant; run in-process" error until the protocol
carries it end-to-end.
See `docs/decisions/ADR-019-native-quantization-support.md`,
`docs/vision/slice-A-fp8-quant-load.md` §2.
Resolved: 2026-07-03 — slice DQ carries quant end-to-end over the daemon:
wire request sends the triple, `_validate_request` semantically rejects
unknown modes (light `QUANT_MODES` constant in `params_validation.py`, no
torch on the accept-loop path), `_request_cache_key` discriminates on the
triple and on the LoRA set when quant is active (quantized pipelines always
evict+reload on LoRA change — direct merges can't be removed incrementally),
and the client delegation-skip is gone. Security review:
`docs/security/review-slice-DQ-daemon-quant-2026-07-03.md`.

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

**DMR partial-merge state on mid-loop raise** *(2026-07-03)*
apply_merge_delta's per-target raise paths (non-finite delta, orphan fp8,
torchao all-zero, requant-scale validation) can fire mid-loop, leaving backup
entries for the successfully-merged prefix while `peft_config[adapter]` and
the LIFO ledger were never written (both happen after the loop). The orphan
backup dict is then invisible to unload_adapters — recovery is a pipeline
reload. Only reachable on adversarial/degenerate adapters (normal LoRAs never
fire these raises), and the pre-DMR behavior for the same inputs was a
whole-adapter refusal, so nothing regressed — but the failure state is
messier. Surfaced by the DMR code-review (finding 3, advisory).
Why not now: requires transactional merge (apply to a staging list, commit
after the loop) — real restructuring for an adversarial-only path.
Trigger: extending the DMR surface (new quantized reps), OR a real user
report of a partially-merged adapter.
See `docs/security/review-slice-DMR-quantized-merge-2026-07-03.md`.
*Amended 2026-07-16 (slice NV, security review req 65): the "new quantized
reps" trigger FIRED with nvfp4 — and worse than the entry assumed: under
`--quant nvfp4` the per-target refusal becomes the NORMAL flow for
direct-merge-only adapters, not adversarial-only, so partial merge would have
been routine. Closed for the unmergeable-rep class by
`refuse_unmergeable_base` (fp8_ops): all four merge call sites scan the
resolution map BEFORE the first mutation and refuse the whole adapter if any
target is a torchao rep the dispatcher would refuse — weights untouched
per-ADAPTER, daemon-cached pipelines clean by construction. The OTHER
mid-loop raise paths this entry lists (non-finite delta, orphan fp8,
requant-scale validation) remain per-target and adversarial-only — the
transactional-merge restructuring is still deferred; this entry stays open
for those, trigger unchanged.*

**Daemon LoRA lifecycle: merged adapters never unload; weight-only changes ignored** *(2026-07-02)*
Two defects in `comfyless/server.py`'s LoRA diff (`_handle_generate` ~392-444),
confirmed read-only during the krea-testing "regression" investigation:
(1) Dropped LoRAs are removed via `pipe.delete_adapters(adapter_name)` — fine
for PEFT-registered adapters, but tier-3 direct-merge LoRAs baked their delta
into `param.data` and register only a cosmetic `peft_config` entry;
`delete_adapters` strips the registration and reports success while the
merged weights persist in the model. The restoration backups
(`_lokr_backup_*` / `_loha_backup_*` / `_lora_backup_*` /
`_converted_lora_backup_*`) exist but the daemon never restores from them.
(2) The diff keys on `path` only — re-requesting the same LoRA at a
different weight hits `path in loaded_paths → continue`, silently keeping
the old weight.
Both push users into restart-the-daemon-between-runs (Grant's actual habit
during LoRA testing). Related: LoRA load failures are non-fatal warnings
that land only in the daemon log — the client CLI never sees them, so
"LoRA failed but run reported success" has now bitten twice (MCP: fixed
28fea0b; daemon log: 2026-07-02 incident).
Why not now: `comfyless/server.py` is a §12 security-review surface; the
fix (restore-from-backup on removal or evict-on-merged-adapter-drop; weight
in the diff key; client-side warning relay) is its own gated slice.
Trigger: the next server.py slice (e.g. quant-over-daemon, same file), OR
LoRA A/B testing friction getting raised again.
See `project_krea_lora_regression.md` memory.
Update 2026-07-03 (explicit re-deferral, not resolution): the trigger fired
(slice DQ touched server.py) and was consciously deferred as out of scope
(`docs/vision/slice-DQ-daemon-quant.md` §Out of scope). Slice DQ SIDESTEPS
both defects for QUANTIZED pipelines only — the LoRA (path, weight) set
joins the cache key when quant is active, so any LoRA change (including
weight-only) evicts and reloads instead of taking the broken diff path.
Unquantized daemon behavior is unchanged; both defects remain open there.
New trigger: next server.py slice that isn't already at capacity, OR
unquantized LoRA A/B friction raised again.

**bnb NF4 single-file support dropped — revisit trigger + pure-torch path** *(2026-07-02)*
ADR-019 dropped NF4 single-file consumption (near-zero collection volume). The
2026-07-02 collection audit found exactly TWO NF4 files, both the same model
(`projectGaiaFlux1D_v20NF4*`), which Grant is genuinely interested in but
agrees doesn't justify the slice alone.
Why not now: one model; the heavyweight path (bitsandbytes runtime) needs a
new dep, and the lightweight path needs careful verification.
**Cheap path when triggered:** NF4 is a fixed 16-value codebook + per-64-block
absmax (usually double-quantized). A pure-torch dequant-at-load is ~50 lines,
NO new dependency — in-memory NF4→bf16 feeding the standard loader (the
ADR-blessed "upcast on load" shim, NOT the dead offline-script direction).
CAUTION: the old `dequantize_nf4.py` failed for undiagnosed reasons; any
implementation needs a numeric cross-check against bitsandbytes' own
dequantize (one-off scratch install, not a pinned dep) before trusting it.
Resolved: 2026-07-17 — trigger fired (Grant actively wants projectGaia;
only released form is NF4). ADR-019 slice NF4 shipped same day (7171d71):
"bnb4" classifier branch + pure-torch dequant-to-bf16 in fp8_ops (no bnb
dep), goldens proven against bitsandbytes 0.49.2 (the old dequantize_nf4.py
was NOT ported — its undiagnosed failure is moot), security contract reqs
67-90 + delta addendum, both real files load (UNET + AIO, 314 families,
11.90B-param FluxTransformer2DModel). Live generation gate passed by Grant
2026-07-17 ("works" — Flux.1 quality itself underwhelming, but that's the
model, not the loader).
Trigger: a second wanted NF4 model appears, OR projectGaia becomes a model
Grant actually reaches for.
See `docs/decisions/ADR-019-native-quantization-support.md`,
`audit_single_files.py` (finds NF4 files by .quant_state/.absmax markers).

**Tier-3 (direct-merge) LoRAs incompatible with `--quant`** *(2026-07-02)*
Under a torchao-quantized base (`Float8Tensor` / `NVFP4Tensor` weights), the LoRA
loader's tier-3 fallback (direct state-dict merge into `weight.data`) cannot run —
you can't in-place-merge a bf16 delta into a quantized tensor subclass. ADR-019 §4
forces the PEFT adapter path (tiers 1/2, **unfused**; `fuse_lora()` also disabled)
under quant, and makes tier-3-only LoRAs **fail loud** ("load without `--quant`, or
use a PEFT-loadable version"). So a LoRA that loads *only* via tier-3 (exotic
LoKR/LoHa, or prefix layouts PEFT can't ingest) is unusable simultaneously with
quant.
Why not now: the PEFT path covers the common case; the real fix (dequantize the
affected base layers → merge → requantize) is genuine engineering worth deferring
until we know it's needed.
**Urgency is gated on a survey** — how many LoRAs in the actual collection need
tier-3 is unknown. Run the loader's format-detection over the `--lora-path` trees
and count tier-3 fallbacks: if it's a handful of exotic ones, fail-loud is fine
indefinitely; if tier-3 is common, promote the dequant→merge→requant path.
MUST-VERIFY separately (not this debt): that `load_lora_weights` on an fp8 base
works via the unfused adapter path on our stack — that's a slice-A acceptance check,
not a deferral.
Trigger: (a) the survey shows a meaningful fraction of the collection needs tier-3,
OR (b) a user hits the fail-loud in real use.
See `docs/decisions/ADR-019-native-quantization-support.md` §4,
`project_native_quant_support.md` memory. Related: the exotic-format LoRAs that drive
tier-3 are the same ones behind `project_lokr_alpha_convention.md` /
`project_civitai_orphaned_files.md`.
**Resolved: 2026-07-03 — trigger (b) fired** (Grant's Krea filter-bypass/.diff and
snofs/LoKR adapters are direct-merge-only AND he needs --quant fp8 for OOM relief).
Slice DMR shipped the dequant→merge→requant path: apply_merge_delta dispatcher in
eric_diffusion_fp8_ops covers plain params (byte-identical), torchao Float8Tensor
params (requant + Parameter swap), and ScaledFp8Linear buffers (per-tensor requant +
cache invalidation), with kind-tagged exact-restore backups and a LIFO unload guard.
Security-gated: docs/security/review-slice-DMR-quantized-merge-2026-07-03.md
(requirements 21-30). ADR-019 §4 amended in its Changelog.

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
- **Resolved: 2026-07-03** — landed with slice DQ (the trigger fired: server-touching commit). `_socket_dir` raises `RuntimeError` on a symlinked `/tmp/comfyless-$UID` before the uid/mode checks; negative test in `test_server_robustness.py` plants a symlink for a fake uid and asserts refusal. Flagged by slice-DQ review F8 so the deferral wouldn't roll forward again.

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

---

## MCP Catalog

### [Code] Family pairing for list_loras / list_transformers (MCP) *(2026-06-27)*
- **What:** `list_loras` / `list_transformers` cannot return a model-family for scan-derived entries — only manifest-declared entries carry `target_family` / `model_family`. So "list LoRAs in the same family as model X" is not answerable today.
- **Why not now:** A separate LoRA-catalog project is the intended source of this metadata; this server will most likely query that catalog (for models, transformers, AND LoRAs) rather than grow its own manifest or add inference.
- **Trigger:** The LoRA-catalog project landing / being wired into comfyless.

### [Code] Agent-facing LoRA-failure signal over MCP *(2026-06-27)*
- **What:** When a LoRA fails to apply over MCP, the warning is logged operator-side only (it embeds an absolute path, which must not cross the boundary per ADR-015); the agent gets no signal that a requested LoRA didn't apply.
- **Why not now:** Surfacing it needs name-based redaction of the warning; out of scope for the OOM/LoRA-apply fix.
- **Trigger:** An agent/UX need for LoRA application status, or the LoRA-catalog integration above.

### [Code] rebalance_weights boundary validation is listness-only + client-omission untested *(2026-06-27)*
- **What:** Three deferred items from the rebalance daemon/MCP wiring reviews (see `docs/security/review-rebalance-daemon-mcp-2026-06-27.md`): (1) `rebalance_weights` is validated as `_KIND_LIST` only — a list of non-numbers or wrong length passes the machine boundary and only fails deep in `_apply_krea_rebalance` (`torch.tensor`) as a caught `InferenceError`, less precise than `loras`' per-entry validation; (2) `rebalance_mult`/`rebalance_weights` accept NaN/Inf (Python `json.loads` allows them), yielding a degenerate image; (3) the client-side omit-`rebalance_weights`-when-None branch in `_delegate_to_server` has no test (no harness exists for the client delegate path).
- **Why not now:** All three fail closed / are self-inflicted, consistent with the project's warn/contain-don't-pre-block footgun-tolerance posture (`security-auditor` + `code-reviewer` both rated them INFO/nit, no blocking). Adding the client-delegate test needs a new socket-mock harness for a trivial branch.
- **Trigger:** A dedicated list-of-float validator kind being added (would naturally cover element-type + finiteness), OR the next change to `_delegate_to_server` (add the omission test then), OR an observed bad-input report.

---

## Parallel daemon (ADR-020)

### [Code] Hard-crash orphans burn auto-number counter slots *(2026-07-03)*
- **What:** The atomic auto-number reservation (`comfyless/server.py` `_handle_generate`, Finding 1 fix) leaves a 0-byte `comfylessNNNN.png` placeholder if the daemon dies **uncatchably** mid-generation (SIGKILL / OOM-kill) — `except Exception` cannot clean those up. The orphan permanently reserves that counter slot across restarts; repeated hard crashes make the counter creep upward and litter 0-byte PNGs.
- **Why not now:** Fail-safe in direction (never overwrites real data), bounded by crash count, and the orphans are visually obvious (0-byte files). The caught-failure path already cleans up. Out of scope for the Finding 1 slice.
- **Trigger:** Observed orphan accumulation in practice, OR any startup-sequence work on `run_server` — at which point a one-line sweep (`unlink` 0-byte `comfyless*.png` in `output_dir` before the accept loop) closes it. Flagged by the slice-3 `security-auditor` pass (`docs/security/review-parallel-daemon-2026-07-03.md`).

### [Code] savepath-template branch retains the concurrent-collision class *(2026-07-03)*
- **What:** Finding 1's atomic reservation covers only the **auto-number** branch. The `if savepath:` branch (user-supplied template) still resolves a path and hands it to `generate()` with no atomic reservation, so two daemons sharing `--output-dir` with the *same template and same params* (a template lacking `%seed%`/timestamp entropy) TOCTOU-overwrite each other exactly as Finding 1 described.
- **Why not now:** Naming here is user-controlled (add entropy or per-device templates); Finding 1 explicitly scoped only the auto-number counter; this slice did not touch the branch. Recorded so "auto-number is atomic" is not mistaken for "all output paths are collision-safe."
- **Trigger:** A user hitting template-collision in a parallel setup, OR extending atomic reservation to the template branch (would need to reserve the resolved path the same way, handling the template's own dir creation). Flagged by the slice-3 `security-auditor` pass.

## 2026-07-06 — transformer shape-multiset matcher cross-matches 3072-dim DiT families

**What:** ADR-021 §3's 0.90 shape-multiset overlap matched Qwen-Image
single-file transformers against the Chroma base (both 3072-dim inventories)
— `matched_bases` spans architecturally distinct families. Family assignment
is protected downstream (duplicate_of + sidecar-agreement pick in the catalog
builder), but the matcher itself is looser than ADR-021 §3's "different DiT
families differ long before 90%" claim on real data.
**Why not now:** catalog-side evidence precedence makes the assignment
correct; the matcher's usable/unconvertable boundary is unaffected for the
current population (files matching wrong-family bases also match their own).
**Trigger:** any file whose matched_bases is WRONG-family-only (would
classify usable against a base it can't load on), or transformer dry-load /
gen-validation surfacing a load failure on a matched base. Fix directions:
dtype-exact multiset, per-key-count weighting, or raising the threshold with
unique-shape anchors.

## 2026-07-06 — Z-Image-Turbo gets base-model family defaults (fried output)

**What:** `infer_model_family` maps both Z-Image-base and Z-Image-Turbo to
`"zimage"` (single `ZImagePipeline` class, NO `is_distilled` marker in
model_index.json — unlike Krea-2, whose flag drives the `krea`/`krea-turbo`
split). `FAMILY_DEFAULTS["zimage"]` (steps 30, cfg 4.0) therefore applies to
the Turbo distill, which needs ~8 steps / CFG 1.0 — produces garbage.
Found live during Phase-B catalog gen-validation (every Z-Image-Turbo run
fried); worked around with explicit `--steps 8 --cfg 1.0`.
**Why not now:** no robust discriminator: model_index is identical; the only
delta is scheduler `shift` (3.0 vs 6.0), a tuning value, not a distill
marker — a heuristic on it would be fragile. Proper fix is per-MODEL default
overrides (operator manifest or catalog `families`/models metadata), which is
a design slice, not a patch.
**Trigger:** next zimage-related slice, OR the first user hit of fried
Z-Image-Turbo output via MCP/OWUI (agents don't pass explicit steps), OR any
new distilled variant landing in hf-local with the same no-marker problem.

**Resolved: 2026-07-06** — `infer_model_family` now detects Z-Image-Turbo by `"turbo"` in the model path (scoped to the zimage family); new `zimage-turbo` FAMILY_DEFAULTS (8 steps / cfg 1.0) + routing in `_build_call_kwargs`. ADR-009 changelog 2026-07-06; tests in test_params_schema.py; verified end-to-end (Z-Image-Turbo → zimage-turbo → 8/1.0).

## 2026-07-06 — lora_audit under-reports convertibility: ignores diffusers' built-in Kohya converters

**What:** `scripts/lora_audit.py` classifies a LoRA `convertable` only when
(a) direct key-match against a base or (b) this repo's `find_matching_plan`
returns a plan. It does NOT account for **tier-1 of the runtime loader** —
`pipeline.load_lora_weights()`, which invokes diffusers' OWN Kohya converters
(`_convert_kohya_flux_lora_to_diffusers`, `_convert_kohya_flux2_lora_to_diffusers`,
and the SDXL/SD3 non-diffusers paths). Result: legit Kohya-format
(`lora_unet_*`) LoRAs for diffusers-supported archs are mislabeled
`unconvertable` and excluded from the catalog.
**Confirmed 2026-07-06 (CPU proof):** `Flux.1-dev/realistic/Alternative_Girls`
(912 Kohya keys, base_model=flux1) → `find_matching_plan` None, audit
`wrong_arch`/excluded — BUT `diffusers._convert_kohya_flux_lora_to_diffusers`
converts it cleanly to 988 valid `transformer.transformer_blocks...lora_A`
keys → loads via tier-1. **~54 Flux.1 Kohya LoRAs under loras/Flux.1-dev/**
are wrongly excluded this way (a large slice of Grant's usable Flux
collection). SD3.5-Large case (`Photorealistic-SD3.5-Large-LoRA`) is
adjacent but murkier — generic converter mangles SD3 keys; no dedicated SD3
Kohya converter in diffusers; needs a runtime load to confirm.
**Why not now:** correct fix is a design change to the audit's `convertable`
detection — add a diffusers-native-conversion probe (call the arch-appropriate
`_convert_kohya_*` on the state dict; if it yields diffusers-shaped keys,
classify `convertable` with a new reason like `kohya_diffusers_native`).
That's an ADR-021/ADR-014 amendment + code + review, not a patch. Also wants
a decision on the SD3 branch and whether to actually run diffusers'
conversion at catalog-build time vs. record-and-defer-to-runtime.
**Trigger:** next lora-audit slice, OR Grant wanting the ~54 Flux LoRAs as
catalog candidates now (interim workaround: they load fine at runtime today
by absolute path — only the CATALOG candidacy flag is wrong, not the
loader). Interim manual reclaim: `catalog_cli exclude --clear <name>` won't
help (audit sets excluded=1 on rebuild); would need an operator-include
override, which doesn't exist yet.

## 2026-07-06 — LoKR LoRAs fail to load on Z-Image (all load paths)

**What:** Standard LoRA (lora_A/B) loads fine on Z-Image; **LoKR (LyCORIS
Kronecker) does NOT.** Confirmed on 3 files (lora-anal, lora-blowjob2,
lora-fisting; w1=[4,4], w2=[3840,64]). Failure chain: (1) diffusers' native
`load_lora_weights` (the fast path that DOES understand Z-Image's key
mapping) has no LyCORIS/LoKR support — LoKR never uses the working path;
(2) the PEFT-injection fallback builds the LoKr adapter with a decompose
factor derived from the target dim, which does NOT match this file's stored
factorization (built w1=[60,60] vs stored [4,4]) → load_state_dict size
mismatch; (3) the direct-merge fallback (`reconstruct_lokr_delta`) can't map
the LoRA's `layers.N.attention.*` names onto Z-Image's actual param layout
(`all_final_layer.2-1.*`, `all_x_embedder.*`, resolution-keyed) → applied=0,
adapter NOT active. The failure is **silent** (non-fatal by design: warn +
generate without adapter).
**Why not now:** the working LoKR merge in this repo (reconstruct_lokr_delta
+ conversion plans) was built/validated for Flux/Klein arch (klein_snofs,
Realism_Engine) where w1/w2 are directly stored and the key-mapping is known.
Z-Image is a newer arch with a different internal param layout not wired into
the LoKR direct-merge path, AND the small decompose factor (4) isn't handled
by the PEFT reconstruction. Fixing = extend the LoKR key-mapping/factor
handling to Z-Image (arch-specific) — a real code slice with its own review.
**Trigger:** Grant wanting Z-Image LoKR LoRAs usable; OR any new arch that
ships LoKR LoRAs. Affected entries are marked `gen_tests.verdict='load_failed'`.

**Resolved: 2026-07-06** — `flatten_lokr_to_lora_sd` (nodes/eric_lora_format_convert_apply.py) added as a final fallback in `_load_lokr_adapter`: when PEFT + direct-merge both fail (0 modules), the LoKR is flattened to standard lora_A/lora_B via SVD (reuses reconstruct_lokr_delta + svd_compress_to_lora) and routed through diffusers' fast path, which handles Z-Image key mapping. Arch-agnostic; fires only on failure (Flux/Klein LoKRs unaffected). Verified end-to-end (3 dead Z-Image LoKRs now load + apply, coherent output); 13 unit tests (test_lora_order_insensitive.py) incl. wiring + alpha-sentinel guard. code-reviewer APPROVED after test-coverage fold.
**Residual (code-review finding 1, LOW):** the rescue is reached only when the PEFT inject RAISES. If a future LoKR mis-sizes but PEFT silently no-ops (reports incompatible keys without raising), the flatten never runs and the silent-failure returns. Closing it fully needs a post-PEFT-success 0-module verification. Trigger: a LoKR that silently no-ops through PEFT.

## 2026-07-06 — gen-validation judged output images, not adapter application (methodology gap)

**What:** The Phase-A/B/rerun LoRA gen-validation judged the OUTPUT images
(baseline vs lora) to assign verdicts. Because LoRA load failures are
**silent** (loader warns + generates WITHOUT the adapter — see the
LoKR/Z-Image entry above and `project_krea_lora_regression`), a failed load
produces lora.png ≈ baseline.png, which the harness scored as
"no_effect"/"inconclusive"/"pass" — masking the failure. 3 LoKR failures were
mislabeled this way and only caught when Grant hit the error interactively.
This is the THIRD time silent LoRA-load failure has bitten (MCP: fixed
28fea0b; daemon log; now gen-validation).
**Why not now:** the proper fix is to make LoRA load-failure LOUD — either
the loader/CLI returns a non-zero signal / structured "adapter not active"
the client surfaces, or the gen-validation harness parses the loader log for
"0 modules applied / NOT active" before trusting an A/B verdict. The latter
is a harness fix; the former is the real product fix (surface daemon LoRA
warnings client-side, already a Backlog candidate per the memory).
**Trigger:** next gen-validation batch (harness must gate on load success);
OR wiring the LLM iterative generator (it MUST know if a LoRA applied).

## 2026-07-06 — SVDQuant / Nunchaku (int4/fp4) single-file consumption unsupported

**What:** Community 4-bit SVDQuant checkpoints (MIT-Han-Lab **Nunchaku**) cannot be
loaded. Grant has `svdq-int4_r32-flux.1-dev`, `svdq-fp4_r32-flux.1-dev`, and
`svdq-fp4_r128-qwen-image` under `.../diffusion_models/Flux.1-dev/`. Header
`__metadata__` self-declares `model_class: NunchakuFluxTransformer2dModel`,
`quantization_config.method: "svdquant"` (weight+activation int4, group_size 64);
tensor layout is `qweight` (packed 4-bit) + `wscales` + `lora_down`/`lora_up`
(the low-rank SVD outlier branch) + `smooth`/`smooth_orig`. This is a distinct
format from ADR-019's fp8/GGUF: **not** a diffusers `quantization_config` and
**not** a `from_single_file` path — diffusers has zero svdquant/nunchaku
awareness (verified). It requires the separate **`nunchaku`** inference engine
(custom CUDA kernels): `NunchakuFluxTransformer2dModel.from_pretrained(path)`
swapped into a diffusers pipeline. Nunchaku upstream covers Flux.1 (dev/
schnell/kontext/fill), Qwen-Image, SANA, PixArt — so Grant's Flux + Qwen-Image
svdq files are in scope.
**Why not now:** heavyweight new dep with tight kernel/torch/CUDA coupling —
must verify a `nunchaku` wheel exists for the pinned stack (torch 2.11 / cu13 /
Blackwell) before committing to a pin (§11); this is the same release-cadence
risk that deferred nvfp4 (ADR-019). Blackwell is where Nunchaku's speedup is
largest, so the hardware fit is good IF the wheel lines up. New loader-routing
surface (svdq detection → nunchaku transformer class → pipeline assembly) and a
§12 security review (caller-supplied model surface). Would be its own ADR
(sibling to ADR-019), not a slice of it.
**Trigger:** Grant prioritizing the Flux/Qwen svdq files over other work AND a
`nunchaku` wheel confirmed for the pinned stack. "All the rage" in the community
(strong download momentum), so likely worth doing once the dep story is clean.
**Won't-do (Grant, 2026-07-06):** Grant is not interested in nunchaku support,
and the svdq files above were a mis-identification — not his actual targets
(see the INT8-ConvRot entry below). Entry retained per §12 (don't delete) as a
record that the format is unsupported; not on the roadmap.

## 2026-07-06 — Krea-2 single-file / GGUF checkpoints cannot be loaded (no converter)

**What:** Community Krea-2 finetunes shipped as single-file
(`krea2MuseByStable_v15TurboFp8.safetensors` + `.gguf`,
`krea2TurboUncensored_v1.safetensors`) fail to load. Two compounding reasons:
(1) `Krea2Transformer2DModel` has **no `from_single_file`** in diffusers 0.39.0
(not a `FromOriginalModelMixin` subclass — verified) — Krea-2 is too new, so
single-file loading (plain OR quantized) is entirely absent upstream; (2) the
files use **native Krea key names** (`blocks.N.attn.wq/wk`, `mlp.gate`,
`qknorm.knorm.scale`) vs diffusers' `transformer_blocks.N.attn.to_q` /
`text_fusion.layerwise_blocks` / `time_embed.*`, and diffusers'
`single_file_utils` has no Krea converter. Our loader's direct-state-dict path
(which already fp8-upcasts) fails on the key mismatch. **GGUF work does NOT help
here** — it's built on `from_single_file`, which Krea lacks; the `.gguf` is
strictly harder than the `.safetensors` (needs the same converter PLUS manual
GGUF dequant).
**Why not now:** the tractable path is a custom native-Krea→diffusers full-
transformer key converter (analogous to the LoRA converters but whole-model) run
before a direct load — moderate-to-large reverse-engineering; the `.safetensors`
is the target (fp8 already handled), the `.gguf` parks behind it. Alternatively
wait for diffusers to add Krea2 `from_single_file` upstream (free, not in our
control, plausible given Krea momentum).
**Trigger:** Grant wanting the Krea finetune specifically AND diffusers still
lacking Krea single-file support; re-check each diffusers bump (upstream may
close it for free).

## 2026-07-06 — INT8-ConvRot single-file consumption unsupported (Grant's actual target format)

**What:** Grant's two near-term target checkpoints are **INT8-ConvRot** format
(civitai.red model 2242173 "Dark Beast … int8 convrot 2 … krea2 aggressive
edition"; model 958009 "RedCraft … int8 convrot NSFW edition 2") — not yet
downloaded. ConvRot = **rotation-based plug-and-play quantization for diffusion
transformers** (QuaRot-family: fuse orthogonal rotations into the weights /
activations to kill outliers, then quantize; here INT8 tensorwise,
`convrot_groupsize` 256). Paper: arXiv:2512.03673
(https://arxiv.org/html/2512.03673v1, Grant-supplied). It is a **ComfyUI-core
format** — added in ComfyUI PR #14636 / commit `1a510f0`, with
`ComfyUI-INT8-Toolkit` (SparknightLLC) around it; metadata self-declares
`int8_tensorwise` + `convrot` (bool) + `convrot_groupsize`. **diffusers has zero
awareness of it** (verified) — so comfyless (diffusers-based) cannot load it
today. NOT nunchaku, NOT GGUF, NOT a `from_single_file` path.
**Why not now:** supporting it in comfyless = a custom single-file loader + a
**rotation-aware INT8 Linear** (apply the fused ConvRot rotation at compute, int8
GEMM, per the paper / ComfyUI's `int8_tensorwise`+convrot op). This is a sibling
to ADR-019 slice C (ported scaled-fp8 ops) but harder — the rotation matrices are
new machinery. **Decisive feasibility point (arXiv:2512.03673): PURE PyTorch, NO
custom CUDA kernels.** ConvLinear = group-wise Regular Hadamard Transform on
activations (block 256; matrices deterministic/constructed on the fly, done as
reshape+matmul, NOT stored) → int8-quantize rotated activations → int8 GEMM vs
per-channel-scaled int8 weights → dequant. "No specialized inference engine
needed; compatible with standard PyTorch." That is the decisive advantage over
nunchaku (kernels) and what makes it tractable — a ConvRotInt8Linear analogous to
slice C's ScaledFp8Linear plus the Hadamard step; likely NO new heavyweight dep.
Reference impls to port: ComfyUI core #14636 + INT8-Toolkit (paper ships no public
repo). §12 security review (caller-supplied model content parse). Own ADR (sibling
to ADR-019), not a slice of it. Paper tested FLUX.1-dev/schnell only (AdaLN DiTs).
**Extra caveat for the krea2 target (model 2242173):** it is a **Krea-2**
finetune, so even with INT8-ConvRot support it inherits the Krea-2 single-file
architecture gap (see the Krea entry below — no diffusers `from_single_file`,
native key names). That one model is blocked on TWO fronts; the RedCraft target
(958009) is likely a more standard base and blocked only on INT8-ConvRot.
**Trigger:** Grant downloading the files + deciding to invest; re-check whether a
diffusers/community loader lands first (ComfyUI-core status means momentum).

**Partial resolution: 2026-07-08 — UNROTATED int8_tensorwise now loads (ADR-019
slice I8, `ci-w`).** A real unrotated file (`x3n0_m4tr1xKrea2`, 224 I8 Linears +
BF16 scalar scales, descriptor exactly `{"format": "int8_tensorwise"}`) drove a
much smaller slice than this entry costed: no rotation ⇒ dequant is `W = int8 ×
scale`, loaded dequant-to-bf16 with NO Int8Linear (with `--quant fp8`, torchao
re-quantizes — the proven LoRA path). Security reqs 46-56
(`docs/security/review-slice-I8-int8-tensorwise-2026-07-08.md`). **What remains
open in THIS entry is exactly the ConvRot-ROTATED case** — descriptors carrying
`convrot`/`convrot_groupsize` now fail LOUDLY (strict field allowlist on int8
descriptors; convrot-prefixed fields reject on fp8 descriptors too, I8-4)
instead of being unloadable-with-a-confusing-error, but applying the rotation
at compute (ConvRotInt8Linear + Hadamard step) is still unbuilt. The costing
above stands for that case; trigger unchanged (a real rotated file + decision
to invest).

## 2026-07-06 — CORRECTION (after inspecting a real file): "int8 convrot" targets are actually scaled-fp8; the blocker is Krea keys, not the quant

**What the file actually is:** `redcraft22INT8Convrot_11INT8Native.safetensors`
(RedCraft, civitai 958009 — downloaded 2026-07-06 into checkpoints/Krea) was
inspected. **Despite the "INT8 ConvRot" filename, the tensors are plain
scaled-fp8:** 256 `F8_E4M3` weights + per-tensor F32 `weight_scale` + a
`comfy_quant` descriptor whose bytes are literally `{"format": "float8_e4m3fn"}`.
**Zero** int8 tensors, **zero** rotation/convrot/hadamard/smooth tensors. This is
exactly the **comfy_quant cq-w** variant comfyless ALREADY ships (ADR-019 slice
C-d) — verified the descriptor passes `_CQ_FORMAT_ALLOWLIST`
(`{"float8_e4m3fn","float8_e5m2"}`). So the quant is a **non-issue**.
**The real (and only) blocker:** the keys are **native Krea-2**
(`model.diffusion_model.blocks.N.attn.wq/wk/wv/wo`, `mlp.gate/up/down`, `mod.lin`,
`qknorm`) — NOT diffusers Krea2 (`transformer_blocks.N.attn.to_q`). The embedded
ComfyUI workflow's save-prefix is `Krea2_turbo`, confirming a Krea-2 base. So this
file is blocked ONLY on the **native-Krea→diffusers key converter** (the entry
above), same wall as `krea2MuseByStable_v15TurboFp8.safetensors`. comfyless has no
Krea key remap (verified — only family *detection* exists, no converter).
**Implication:** the "add INT8-ConvRot support" work is NOT needed for Grant's two
targets — they're fp8, already handled. **One piece of work — the Krea key
converter — unblocks the plain Krea fp8 file AND both "int8 convrot" downloads.**
The generic INT8-ConvRot entry above remains a valid FUTURE item (real int8-convrot
files exist upstream), but it is decoupled from these specific models. **Dark Beast
CONFIRMED (inspected 2026-07-06):** `darkBeastINT8Convrot2_darkBeastKREA2FP8.safetensors`
(22 GB) is ALSO fp8 + native-Krea, NO int8/convrot (its only "rot/conv" tensors are
VAE conv layers). It differs from RedCraft in two ways: (a) it's an **all-in-one
bundle** — `text_encoders.*` (Qwen3-VL-4B, 714 tensors) + `vae.*` (194) +
`model.diffusion_model.*` transformer (430); (b) the transformer fp8 is **plain-cast
(no scale, no comfy_quant)** → our loader already upcasts plain-cast fp8 to bf16, so
again the quant is a non-issue. So Dark Beast needs the Krea key converter PLUS
component-splitting (extract the transformer from the bundle; Krea-2-Turbo base
supplies TE/VAE, or use the bundled ones). **RedCraft is the clean first target**
(transformer-only, comfy_quant fp8, blocked ONLY on Krea keys). Both filenames'
"INT8 ConvRot" labels are misnomers — neither file contains int8 or rotation tensors.

## 2026-07-06 — UPDATE: diffusers PR #14126 adds Krea2 from_single_file — TESTED, converts our files' keys correctly

**Finding (tested 2026-07-06):** diffusers **PR #14126** ("Add from_single_file()
support for Ideogram4 and Krea2 transformers", open, opened 2026-07-06, fixes
#14122) adds `FromOriginalModelMixin` + a Krea2 single-file key converter to
`Krea2Transformer2DModel`. Pulled the PR head (still `0.39.0.dev0` — SAME base as
our pin, so API-compatible) into an isolated /tmp checkout, overrode only diffusers
via PYTHONPATH against the project .venv, and ran RedCraft through it:
- **Key conversion WORKS.** RedCraft's native ComfyUI keys
  (`model.diffusion_model.blocks.N.attn.wq/wk/wv/wo`, `mlp.*`, `mod.lin`) convert
  cleanly to diffusers (`transformer_blocks.N.attn.to_q…`). Model loads with all
  430 params, 0 meta/uninitialized, correct shapes. NO missing-key warnings. (Needs
  `low_cpu_mem_usage=False` to dodge a meta-tensor/fp8 cast NotImplementedError —
  a mechanical detail, not a key problem.)
- **fp8 scale is dropped.** diffusers reports every `.weight_scale` as "not used".
  Verified the correct dequant is `fp8 × weight_scale` (absmax 1.859, EXACT match to
  base Krea-2-Turbo `to_q`); diffusers keeps the raw fp8 (absmax 448) → wrong
  weights without our scaling. **Applying `fp8 × weight_scale` is exactly slice
  C-d's comfy_quant cq-w path.**
**Path forward (supersedes "write our own converter"):** the two halves compose —
**diffusers PR #14126 (key conversion) + our slice C-d (fp8 scale) = correct
RedCraft load.** Dark Beast (plain fp8-cast, no scale) needs only the key conversion
+ bundle-split. So the comfyless work becomes a modest INTEGRATION slice (route
native-Krea single-files through Krea2.from_single_file for keys, then apply our
comfy_quant scaling on the fp8 layers), NOT a from-scratch key converter.
**Options:** (a) wait for #14126 merge + diffusers release, then bump pin + integrate;
(b) vendor the PR's Krea2 converter function now (pure key-mapping) to unblock before
release, tracking upstream. Either way the earlier "custom Krea converter" plan is
retired. NOTE: isolated PR checkout left at scratchpad/diffusers-pr14126 for the
vendoring option.

## 2026-07-07 — Security: `connect_readonly` skips the FUSE guard on an explicit `--catalog-db` path

- **What:** `comfyless/catalog_db.connect_readonly` (unlike the writable `connect`, which
  calls `fs_is_fuse`) performs no FUSE/mergerfs check before opening. A WAL-mode SQLite
  read on a FUSE union can hang on fcntl byte-range locks (the documented environment
  foot-gun). Surfaced by the slice-4a security review (INFO-3,
  `docs/security/review-slice-4a-catalog-db-autodiscover-2026-07-07.md`).
- **Why not now:** Not reachable via slice-4a auto-discovery — `DEFAULT_DB_PATH` lives under
  `~/.local/share` (home ext4, not the `/home/gawkahn/projects` mergerfs union). Only an
  operator who explicitly points `--catalog-db` at a FUSE-backed path is exposed, and a hang
  there is a pre-existing gap in `catalog_db.py`, outside slice 4's edit scope.
- **Trigger:** Add a `fs_is_fuse` fail-closed guard to `connect_readonly` (mirroring `connect`)
  the next time `catalog_db.py` is opened for a deliberate change, OR if a user ever reports a
  startup/search hang with `--catalog-db` pointed at a mergerfs path.

## 2026-07-08 — Embedded checkpoint adapters are IGNORED (loud notice), not applied

- **What:** community Krea-2 checkpoints can pack an UN-MERGED adapter inside the
  checkpoint file (`x3n0_m4tr1xKrea2`: 256 lora_A/B pairs + one `.diff` under a bare
  `diffusion_model.` prefix, next to the `model.diffusion_model.` int8 base).
  `build_krea2_transformer` now strips these guaranteed-fatal key shapes with a loud
  WARNING and loads the BASE model — matching ComfyUI's checkpoint loader, which also
  ignores that half. The adapter's visual effect is therefore NOT reproduced.
- **Why not now:** applying it correctly needs a scaling decision (no alpha stored;
  B@A at 1.0 is a guess — a wrong scale is silent wrong-look output, the exact class
  we refuse), and the cleaner UX is extracting it to a standalone LoRA file the user
  stacks deliberately with a chosen weight. The direct-merge machinery for applying
  it already exists (apply_merge_delta / convert path).
- **Trigger:** Grant wants x3n0's embedded-adapter look specifically (compare with/
  without in ComfyUI first — if ComfyUI ignores it too, the "intended look" already
  IS the base), OR a second embedded-adapter checkpoint appears. Then: a small
  extract-embedded-adapter tool (scripts/lora_audit family) beats loader auto-apply.

## 2026-07-07 — dequant-fp8 routing gaps: text-encoder slot + directory-fallback (slice R1/R2/R3 INFO deferrals)

- **What:** two conscious scope limits from the R1/R2/R3 code review (both INFO, not blockers):
  (a) `_load_pipeline` wires `dequant_fp8` only for the transformer/unet override — a LARGE
  text-encoder override pointing at a native scaled-fp8 single file with `--quant` active would
  stay `ScaledFp8Linear`-resident and torchao would silently no-op on it, the exact class R3
  closes for the transformer; (b) `load_component`'s directory-override fallbacks
  (`_try_from_single_file` in the `subfolder`/`direct` branches) don't thread the flag — safe
  today because they never enter the scaled-fp8 loader, but a future re-route would drop it.
- **Why not now:** the security review (req 44) scoped R3 to the transformer single-file
  override — the only case in live use; no known native-fp8 TE files in the collection.
- **Trigger:** first native scaled-fp8 text-encoder file appears, OR `_try_from_single_file`
  is ever taught to route into the scaled-fp8 loader.

## 2026-07-07 — fp8-RESIDENT LoRA direct merge produces NaN (black images) for some Krea LoRAs; not root-caused

- **What:** merging converted krea LoRAs into an fp8-resident (`ScaledFp8Linear`) base via the
  DMR dequant→add→requant path reports success (`applied=256, skipped=0`) but generation renders
  all-black (`invalid value encountered in cast` = NaN in the decoded image). Observed live on
  `moodyKrea2Mix_v30` with `nicegirls_krea2` and `ultra_real_krea2_v1` (single AND stacked);
  `MysticXXX_KREA2_v1` and `snofs` merge fine on the same base. LoRA files themselves are clean
  (no NaN/Inf, unremarkable magnitudes — the failing ones are SMALLER than the working one), so
  the mechanism is in the resident-merge/runtime numerics, not the adapters. Needs a GPU
  reproduction to bisect (requant scale coarsening? weight-only dequant cache? activation
  overflow at 6144-dim `to_gate`?).
- **Why not now:** ADR-019 slice R1/R2/R3 (2026-07-07) routes the affected combination around the
  resident path entirely — `--quant fp8` now dequants the native file to bf16 and re-quantizes
  via torchao `Float8Tensor`, the representation the DMR merge is proven on. With that
  workaround live, the resident-path NaN stops blocking Grant's workflow.
- **Trigger:** anyone needs LoRAs on an fp8-RESIDENT base *without* `--quant` (VRAM-constrained
  case where the bf16 dequant spike is unaffordable), OR a torchao-path generation also renders
  black (would falsify the "resident-merge-specific" theory and reopen the diagnosis).

## 2026-07-07 — Converted Krea LoRAs drop `.diff_b` bias deltas and second-level fp8 scales (`weight_scale_2`)

- **What:** `convert_state_dict` (`nodes/eric_lora_format_convert_apply.py`) emits WEIGHT
  deltas only. On a converted Krea LoRA, bias deltas (`.diff_b`) and any second-level fp8
  scale (`weight_scale_2`) are counted and loudly warned but never merged; the scaled-fp8
  header loader also rejects `weight_scale_2` (`nodes/eric_diffusion_fp8_ops.py`). Surfaced by
  the code-review of the fp8-resident-Krea-LoRA fix (7cc99ab, ADR-019 Changelog 2026-07-07).
- **Why not now:** the in-hand community Krea LoRAs (snofs / lenovo / nicegirls) ship neither,
  so they apply fully; no LoRA in the collection exercises the path, and merging a bias delta
  or a second-level scale into a REQUANTIZED fp8 base needs its own DMR-style security pass
  (a new write target beyond the DMR-3 `.weight`-only merge surface).
- **Trigger:** a Krea-2 distill/turbo LoRA reported partially-applied (some modules active,
  bias/scale-bearing ones silently skipped), OR a `weight_scale_2` reject in the daemon log.

## 2026-07-08 — NAG v1 deferrals: compute sharing, other families, ComfyUI nodes, CFG+NAG

- **What:** NAG for Krea-2 (ADR-023) landed naive: (1) the batch-2 lanes recompute the image
  tokens twice per NAG'd step (~1.9-2.0x wall full-window) — the paper's image-token compute
  sharing (+87%-overhead variant) is not implemented; (2) only krea/krea-turbo are gated in —
  other distilled families (Flux Schnell-class, Qwen distills) could reuse the machinery with
  their own processors; (3) no ComfyUI node surface — comfyless/MCP only; (4) NAG + classic
  CFG (guidance_scale>0) routes to stock CFG with a warning instead of combining them the way
  the reference pipeline can (`do_true_cfg` + NAG simultaneously).
- **Why not now:** correctness first (ADR-023 decision 3) — the naive port is testable against
  the reference math and Grant's A/B; compute sharing changes the processor's q/k/v layout and
  deserves its own slice with a wall-clock benchmark. Other families and ComfyUI nodes have no
  user demand yet. CFG+NAG has no use case on the checkpoints in hand (Raw uses CFG, Turbo
  uses NAG).
- **Trigger:** NAG's ~2x wall cost bothers Grant on 8-step Turbo (→ compute sharing or a VSF
  arXiv:2508.10931 benchmark); a non-krea distilled checkpoint needs negative prompts (→
  family expansion); NAG wanted in the ComfyUI graph (→ node surface); a model where CFG and
  NAG both matter (→ combined mode).

**Partially resolved: 2026-07-09** — deferral (2) "other families" closed by ADR-024: NAG now
covers flux, flux2, flux2klein, zimage, zimage-turbo (pipelines/nag_{common,flux,flux2,zimage}.py;
family gate table in generate.py). Chroma stays deferred with a changed shape: it is de-distilled,
so the right fix is routing negatives to its real CFG, not NAG. Deferrals (1) compute sharing,
(3) ComfyUI nodes, and (4) CFG+NAG combined mode remain open, plus a new one: Flux.2
reference-image (kontext) inputs skip NAG loudly (HF2-1). Triggers unchanged.

## 2026-07-09 — `_abspath` in the daemon wire request does not expand `~`

- **What:** `_build_wire_request._abspath` (`comfyless/generate.py:1822`) is bare
  `os.path.abspath()`. A `~/...` path in a `--params` sidecar or an `--iterate` JSON list is
  not a repo ID, so `resolve_hf_path` passes it through untouched (`generate.py:2323`), and
  `abspath` then joins it onto the client's CWD — the daemon receives `<cwd>/~/projects/...`
  and rejects it via `_check_paths` ("outside the allowed roots"). Affects `model`,
  `transformer_path`, `vae_path`, `text_encoder{,_2}_path`, and `loras[].path`. CLI flags are
  unaffected only because the shell expands the tilde before argparse sees it.
- **Why not now:** hand-writing `/home/<user>/...` in the JSON is a complete workaround, and
  the fix touches the `--json`/wire boundary feeding the server's allowed-roots check, so it
  wants `code-reviewer` plus a decision on whether tilde expansion belongs client-side at all
  (client `$HOME` then picks which path the server opens — benign for a solo local tool,
  but exactly the input the root check exists to constrain).
- **Trigger:** next deliberate change to wire-request path handling, or the next time a
  hand-written sidecar / iterate list fails the allowed-roots check. Check whether
  `cascade.py` and `mcp_server.py` build path fields the same way — if so the fix is one
  shared helper, not three copies.

## 2026-07-10 — `--quant fp8` transformer quant destroys Z-Image-base (NOT Turbo)

- **What:** `--quant fp8` on a Z-Image model yields a structurally-correct but heavily
  speckled image (composition survives; every patch carries high-frequency noise).
  Root-caused by an A/B/C matrix on 2026-07-10: (1) `--transformer greed_int8.safetensors`
  onto `Z-Image-base` **without** `--quant` → clean; (2) stock `Z-Image-base` **with**
  `--quant fp8` and no override → garbage; (3) stock `Z-Image-base` alone → clean. So the
  fault is `--quant fp8` on this architecture, independent of the int8 single-file loader,
  the `--transformer` override path, and the checkpoint itself.
- **CORRECTION 2026-07-10 (same day):** the title above is too broad. **Z-Image-*Turbo* +
  `--quant fp8` works fine** — verified during NAG testing and re-verified after this entry
  was written. Same architecture, same recipe, opposite outcome. So the fault is NOT
  "quant on Z-Image" and NOT the module set. Turbo and base differ in exactly two ways:
  weights, and schedule (turbo `cfg 1.0 / 8 steps`; base `cfg 4.0 / 30 steps`).
- **CFG-amplification hypothesis — FALSIFIED 2026-07-10.** Predicted Turbo + quant + `--cfg 4.0`
  would speckle and base + quant + `--cfg 1.0` would be clean. Both predictions failed:
  Turbo@cfg4 is *blurry* (ordinary over-guidance on a distill, no speckle) and base@cfg1 is
  *still speckled*. **The schedule is not the variable — cfg and steps are both exonerated.**
  fp8 is a floating format whose relative precision is ~constant with magnitude, so the
  weight-outlier reasoning was measuring the wrong property for this quantizer anyway.
- **LOCALIZED 2026-07-10:** `--quant fp8 --quant-only transformer` on Z-Image-base **speckles**.
  So it is quantization of the `ZImageTransformer2DModel` denoiser. The `Qwen3Model` text
  encoder is **exonerated** (consistent with the operator's prior: quantizing this TE has
  never caused problems on any family). One run, not two.
- **Current best explanation: the recipe uses PER-TENSOR granularity.**
  `_torchao_fp8_config()` returns `Float8DynamicActivationFloat8WeightConfig()` whose default
  is `granularity=[PerTensor(), PerTensor()]` (torchao 0.17.0, confirmed by introspection) —
  per-tensor scales for BOTH the dynamic activations and the weights. Per-tensor *activation*
  scaling lets one outlier token set the scale for the whole tensor and crush the rest toward
  zero. Non-distilled models are known to carry heavier activation outliers than their
  distills, which fits base-breaks / Turbo-survives on identical modules. It also explains
  why the weight-side numbers looked innocent: per-tensor fp8 *weight* quant of a base tensor
  is cos 0.99965 vs the original — the damage is on the activation side, invisible without a
  forward pass. NOTE this remains a hypothesis: weight-vs-activation was NOT separated, and
  cannot be from CPU. `PerRow` (per-token activations, per-output-channel weights) is the
  standard remedy and changes both at once.
- **Fix is structurally safe** (checked, not assumed): `_merge_into_torchao`
  (`eric_diffusion_fp8_ops.py:1324`) is granularity-agnostic — it `dequantize()`s and
  requantizes through `_torchao_fp8_config()` itself, so it inherits any recipe change.
  `quant_cache_fragment` keys on quant MODE, not granularity, so no stale-cache hazard.
  fp8 already gates on compute capability >= 8.9, which is also PerRow's requirement.
  The real risk is behavioral: the recipe is shared by Qwen / Flux / Krea (currently
  verified-good) and by the DMR LoRA merge, so the change needs a before/after image matrix
  across families, not a drive-by.
- **Also found:** `--quant-only model` is a silent no-op footgun. `model` is not a
  `model_index.json` component, so it is ignored with a notice and `build_quant_config` then
  bails with "no eligible components" and loads UNQUANTIZED. A clean image results and the
  operator concludes quant was fine. Valid component names for Z-Image are `transformer`,
  `text_encoder`, `vae` (VAE refused by invariant). Consider promoting the notice to a loud
  warning, or failing when `--quant-only` selects nothing.
  Run those two before touching any code.
- **Weight-outlier hypothesis — WEAKENED (measured 2026-07-10):** base and turbo have nearly
  identical outlier severity across all 276 2-D weights (median `|w|max/rms` 17.6 vs 16.1,
  p95 52.3 vs 47.9). Base's single worst tensor is larger (47.8 vs 14.0 `|w|max`) but the
  distributions overlap heavily. Does not on its own explain clean-vs-garbage.
- **Original suspected cause — NOT RULED OUT, but no longer leading:** `_torchao_fp8_config()`
  (`nodes/eric_diffusion_utils.py:1557`) returns a bare
  `Float8DynamicActivationFloat8WeightConfig()` with **no `filter_fn`**, so
  torchao converts *every* `nn.Linear` — including `final_layer.linear` (tokens → output
  patches), the per-block `adaLN_modulation` projections, and the `t_embedder` /
  `cap_embedder` stacks. diffusers' own `ZImageTransformer2DModel` declares
  `_skip_layerwise_casting_patterns = ['t_embedder', 'cap_embedder']`, i.e. upstream
  considers those precision-sensitive. fp8-e4m3 has a 3-bit mantissa, and the recipe also
  quantizes **activations** dynamically. This would still be a contributing factor — a
  noisier per-pass eps is exactly what CFG then amplifies. Both the override path
  (`quantize_module`) and the
  `from_pretrained` path (`build_quant_config`) share this recipe, which matches the
  observation that the bug reproduces without any override.
- **Not the cause (ruled out numerically, same session):** int8 dequant is correct
  (cos 0.9998–0.9999/layer vs `Z-Image-base`, all 170 layers use the full ±127 range);
  the fused-qkv split is correct (0.9999 on-diagonal, ~0.07/0.00 off-diagonal); diffusers'
  Z-Image converter emits all 521 keys with 0 missing / 0 unexpected, prefixed or stripped;
  embedders and pad tokens match base at cos 1.000000; stacking fp8 on int8 costs almost
  nothing (cos 0.99945 vs 0.99965 for fp8-on-base).
- **Why not now:** the fix is a `filter_fn` (or `module_fqn_to_config`) excluding the
  precision-sensitive modules, but *which* modules is an empirical question needing a GPU
  A/B sweep per family, and `_torchao_fp8_config` is shared by every family plus the DMR
  LoRA merge path (`_merge_into_torchao`) — narrowing it silently changes quant behavior
  for Qwen/Flux/Krea, which are currently verified-good. Needs its own slice with a
  before/after image matrix, not a drive-by.
- **Trigger:** next `--quant` work, or the next report of degraded output under quant on any
  family. Until then `--quant fp8` should be treated as unsupported on Z-Image-base;
  Z-Image-Turbo + quant is verified working.
- **Answered 2026-07-10:** not a regression. Z-Image + quant HAS been exercised — on
  **Turbo**, during NAG testing and again after this entry was written, and it works. The
  one archived `zimage.json` sidecar carries no `quant` key, but that is a single run and
  proves nothing on its own. What was never exercised is **base + quant**, i.e. quant at
  `cfg 4.0 / 30 steps`. Nothing broke it; that combination never ran.

---

## Hunyuan-Image

**ComfyUI Generate node re-loads the refiner on every execution** *(2026-07-11)*
The `EricDiffusionGenerate` node calls `load_refiner_pipeline(...)` inline in its
`generate` method, so the refiner transformer + VAE are re-loaded from disk each
run (the base pipe is loader-cached; the shared text_encoder is injected, so it's
not a full reload). The comfyless in-process and daemon paths cache the refiner
(`_cached_pipeline["refiner_pipeline"]` / `server_state["refiner_pipeline"]`); the
node path does not. **Why not now:** intentional per ADR-016 Vision OQ4 — the node
exposes only `refiner_path` on the Generate node (not a separate refiner-loader
node), so there is no node-side cache slot to hold it; the operator's ComfyUI
session is the operator's own trust domain and the reload is a per-run latency
cost, not a correctness issue. **Trigger:** if the per-execution refiner reload
becomes a real workflow bottleneck, add a refiner cache to the loader node (mirror
the base-pipe cache) or a dedicated `EricDiffusion Load Refiner` node. Surfaced by
the re-apply `code-reviewer` observation, 2026-07-11.

**ComfyUI Generate node does not quantize the refiner** *(2026-07-12)*
`nodes/eric_diffusion_generate.py` calls `load_refiner_pipeline(...)` without
threading `quant` (defaults to "none"), so in the ComfyUI node path a
`--quant`-loaded base is paired with a full-precision refiner. The comfyless CLI
and daemon paths thread quant correctly (base+refiner+reprompt all fp8).
**Why not now:** the node has no quant handle at that point — the loader node
quantizes the base and the GEN_PIPELINE dict doesn't carry the quant mode
forward; the operator uses the comfyless CLI for VRAM-tight hunyuan runs.
**Trigger:** node-path refiner VRAM becomes a real constraint — thread the quant
mode through GEN_PIPELINE (or detect the base's torchao state) into the node's
refiner load. Surfaced with the refiner-quant slice, 2026-07-12.
Resolved: 2026-07-12 — MOOT. The refiner is now NOT fp8-quantized on ANY path
(it produces black output under fp8 — see next entry), so the node correctly
matches the CLI/daemon behavior. Re-opens only if refiner fp8 is ever fixed.

**HunyuanImage refiner is not fp8-safe (black output)** *(2026-07-12)*
Quantizing the refiner's transformer to fp8 — either torchao recipe,
dynamic-activation OR weight-only — produces all-black (NaN) output at 2K, even
though the BASE transformer quantizes cleanly (verified by isolation: base-only
fp8 = content; base+refiner fp8 = mean 0.0 black). `load_refiner_pipeline` now
leaves the refiner in bf16 under `--quant fp8` with a loud log; the base +
reprompt still quantize. **Why not now:** root cause is per-layer (some refiner
Linear(s) overflow fp8) and needs a component-level exclusion investigation
(which modules to skip), not a recipe flip. **Trigger:** someone wants the
refiner's ~34 GB back under quant — bisect the refiner's Linear modules under
fp8 to find the NaN source and add a targeted skip-set, then re-enable the
(already-threaded) quant path in `hunyuan_chain.load_refiner_pipeline`. The
quant params stay wired through generate/server for that future fix.

**Upscale-VAE decode round-trips the 20B transformer CPU↔GPU every call (ADR-030)** *(2026-07-14)*
`decode_latents_with_upscale_vae_safe` unconditionally offloads `pipe.transformer`
to CPU and back for each 2× upscale decode (to free VRAM for the Wan decode). In
the `--serve` daemon that means every `--upscale-vae` generation pays a ~40 GB
(bf16 20B) PCIe transfer each way — real seconds — partially undercutting the
speed win that motivates the feature. Restore is guaranteed in a `finally`, so
the cache stays correct; only latency is affected. The `.to("cpu")`/`.to(cuda)`
round-trip of a torchao-quantized (`--quant fp8`) transformer is also unverified.
**Why not now:** slice 1 prioritized correctness + the daemon cache design; the
offload is the proven-safe node behavior. **Trigger:** upscale becomes a hot
daemon path — make the offload conditional on free VRAM (skip when the Wan decode
fits alongside the resident transformer) or opt-in for the daemon, and add a
hot test for `--quant fp8` + `--upscale-vae`. Surfaced by code-review of ADR-030.

**Hunyuan-reprompt sampling-knob cfg passthrough is untested (2026-07-15)**
`enhance_hunyuan_reprompt` copies backend-cfg sampling knobs
(`temperature`/`top_p`/`top_k`/`repetition_penalty`/`min_p`) into `gen_kwargs`
raw — no `_coerce_sampling_value` and no test coverage for ANY knob on this path
(the endpoint path is well-covered; the hunyuan loop is not). Adding `min_p` to
the loop inherited this gap rather than creating it. **Why not now:** the min_p
slice scoped to the endpoint resolver + recipe validation, which the tests do
cover; the hunyuan path passes values straight to transformers `.generate()`,
which validates them itself, so a bad cfg value surfaces as a generate-time
error rather than silent corruption. **Trigger:** if the hunyuan cfg-knob path
grows a coercion/validation step or a new knob with non-obvious semantics, add a
gen_kwargs-capture test (mock `_load_reprompt`) proving each knob reaches
`.generate()` and that bad types are rejected. Surfaced by code-review of the
min_p wiring slice.

**Refinement-loop cold path has no root containment for seed component paths (2026-07-15)**
`comfyless/refine.py::run_generation` falls back to an in-process `gen.generate()`
when no daemon socket is present. On that COLD path, seed-supplied
`model`/`transformer_path`/`vae_path`/`text_encoder*_path`/`refiner_path`/
`upscale_vae_path` load directly from any local directory the seed image's
metadata names — the daemon's `_check_paths` root-union validation never runs.
This is the same accepted trust model as `generate --params` replay (the LLM
verifiably cannot reach `WorkingConfig.base`; only a user-chosen seed image can),
so slice 4 shipped it as-is with the F4 loud echo strengthened to FLAG each path
outside the roots ("loads on the cold path only"). **Why not now:** a fail-closed
cold-path containment gate is a trust-model change (it would also constrain the
existing fresh-`--prompt`/`--params` cold path), a decision that belongs to Grant,
not a mechanical slice. **Trigger:** the refinement loop is ever run against
seed images from a less-trusted source, OR the cold path becomes reachable by a
non-interactive caller — then add a `_within`-union gate to `run_generation`'s
cold branch (mirroring `server._check_paths`) and decide fail-closed vs warn.
Surfaced by security-auditor review of ADR-027 slice 4 (MEDIUM-4).

**Batch enhance CLI (`comfyless.enhance`) has no output-path containment (2026-07-16)**
`comfyless/enhance.py::_cli` writes its `--output` JSON list + `.provenance.json`
sidecar to whatever path is given — absolute, `../`, anywhere — with no
`_check_paths`-style containment. This is SAFE today: the batch file-in/file-out
CLI is operator-only trusted input (same as `comfygen --savepath`), the enhancer
has ZERO MCP surface, and the MCP-natural path is INLINE enhancement
(`enhance_prompt_list`, used by `generate --enhance-prompt`) which returns strings
and writes NOTHING. **Why not now:** no untrusted caller can reach the file-write
path; adding containment would be speculative. **Trigger:** if the batch
file-in/file-out enhance is ever wired as an MCP/agent tool (the wrong shape — the
inline path is what a chat agent wants), that commit turns `--output` into a
§12 untrusted-path-write surface: add `server._within`/`_check_paths`-style output
containment against an allowlisted root and treat the commit as Red Zone from day
one (mirrors the `--json` bridge / future-MCP treatment in the project Review bar).
The 2026-07-16 overwrite-confirmation guard (warn + interactive y/N) is orthogonal
and does not address containment.

**git-policy: function-scoped Red Zone surfaces not path-gateable (2026-07-16)**
The quality-gate kit's commit-policy layer (`scripts/git-policy/`, adopted
2026-07-16) gates Red Zone commits by *file path* (`_red-zone-paths.sh`). Two of
this repo's §12 surfaces are *function*-scoped: `comfyless/generate.py::
_run_json_mode` and `nodes/eric_diffusion_utils.py::resolve_hf_path`. Their host
files change in most feature slices for non-Red-Zone reasons, so listing them
would fire the ADR/review-reference gate constantly and train a `Policy-override:`
habit — they are deliberately NOT listed; enforcement for them remains T5
(CLAUDE.md Review bar + reviewer discipline). **Why not now:** path-based hooks
cannot see which function a diff touches; a diff-hunk-range parser is real work
disproportionate to a solo repo. **Trigger:** either function moves into its own
module (then add the path), or a commit slips through that changed one of them
without review — then either build hunk-range detection into
`pre-commit-checks.sh` or accept whole-file gating despite the friction.

**Two CVEs unfixable at 2026-07-16 CVE batch — accepted pending upstream (2026-07-16)**
The `deps:` CVE batch (pillow/mcp/click direct; urllib3/pyjwt/python-multipart/
pydantic-settings/idna/cryptography transitive) cleared every advisory with a
released fix. Two remain open: (1) **torch 2.11.0 CVE-2025-3000 /
GHSA-rrmf-rvhw-rf47** — no fixed release exists; (2) **setuptools 81.0.0
PYSEC-2026-3447** — fix is 83.0.0 but torch 2.11.0 declares `setuptools<82`,
so the fix is unreachable without a torch bump. **Why not now:** nothing to
bump to. **Trigger:** any torch release note (watch the ADR-013 torch-pin
lockstep: torchvision minor must move with it) — on the next torch bump, drop
both ignores from the deps-cve recipe/osv config and re-scan. Both are pinned
as documented ignores in the supply-chain gate so the CVE gate can be 0-red.

**tests/ subdir LoRA-convert suites outside the test battery (2026-07-16)**
`tests/test_lora_format_convert*.py` (5 suites + harness) are standalone-
runnable but are NOT in the `just tests` battery (root-only glob) and appear
in no gate. They predate ADR-013 and their docstrings reference the old
comfy-dev venv; whether they pass under the uv `.venv` is unverified.
**Why not now:** pulling them in blind could turn the new tests gate red on
an environment mismatch rather than a real regression. **Trigger:** next time
LoRA format conversion is touched — verify them against `./.venv/bin/python3`,
then either add a `tests/test_*.py` arm to the justfile recipe or record why
they stay manual. Surfaced by code-reviewer during the tests-gate slice.

**Connect-refused on a LIVE daemon still falls through in-process (2026-07-17)**
`_send_server_command` (comfyless/generate.py) treats connect/send `OSError`
as "daemon absent" → returns None → `_delegate_to_server` falls through to
in-process generation. A live daemon whose listen backlog is momentarily full
would be misread as absent, starting an in-process model load on a GPU whose
VRAM the daemon holds — the same hazard the 2026-07-17 recv-side fix closed
(ClientRecvError), surviving on the connect side. **Why not now:** requires a
concurrent client burst against a serial solo-desktop daemon; the recv-side
fix covers the observed incident. **Trigger:** any parallel-client work
against one daemon (batch drivers, MCP multi-session), or the next
`_send_server_command` change. Surfaced by code-reviewer during the
pause-daemon-guard slice.

**run_server unlinks a live daemon's socket without a liveness probe (2026-07-17)**
`run_server` (comfyless/server.py:933-934) unconditionally unlinks an existing
socket at startup. Starting `comfyless@N` (systemd unit, 2026-07-17) while a
manually-started daemon owns that device's socket steals the path: the manual
daemon is orphaned (unreachable, still holding VRAM) and its shutdown
`finally` later deletes the NEW daemon's socket. `systemctl stop`'s
`ExecStop --unload` can likewise cleanly unload a foreign daemon over IPC.
Availability-only, single-user. **Why not now:** the fix (connect-probe the
existing socket; refuse to start if it answers) modifies a Red Zone path and
takes the ADR-reference + security-auditor route — not a drive-by on the unit
slice. **Trigger:** first real dual-launch collision, or the next
`run_server` change. Surfaced by infra-auditor
(docs/security/review-systemd-daemon-unit-2026-07-17.md, SHOULD 2).

**Daemon LoRA path misses weight application + ignores weight-only changes** *(2026-07-17)*
The `_apply_loras` weight fix (cumulative `set_adapters`, this date) covers
the CLI/MCP paths only. `comfyless/server.py` (~:690-712) calls
`load_lora_with_key_fix` directly with no post-load `set_adapters` — daemon
LoRAs run at FULL trained strength regardless of the requested weight (the
exact bug just fixed elsewhere; Grant's mystic/mcnl noise repros ran through
the daemon). Worse, the LoRA diff keys on PATH only (`if lora_spec["path"]
in loaded_paths: continue`), so a weight-only change on an already-loaded
LoRA is silently ignored.
Why not now: `server.py` is a Red Zone path (security-auditor + saved review
required); folding it into the CLI-side fix slice would be scope creep into
a gated file (review finding 2, 2026-07-17).
Trigger: next `server.py` slice — treat as its FIRST item; until then,
LoRA-weight-sensitive work should run foreground (no daemon on that GPU).
Resolved: 2026-07-17 — slice DLW, same day: `apply_adapter_weights`
extracted as the shared cumulative scaler (generate.py), daemon add-loop
made weight-aware (PEFT weight-only changes apply in place; direct-merge
weight changes warn loudly and keep the baked weight; duplicate paths in
one request load once, last weight wins), one cumulative call over the
full active set per request. security-auditor delta PASS-with-conditions
(F10 fixed in-slice, F11 → its own TECH_DEBT entry below); code-reviewer
findings folded. See docs/security/review-slice-DLW-daemon-lora-weights-
2026-07-17.md + docs/vision/slice-DLW-daemon-lora-weights.md.

**Node LoRA stacker: per-adapter singleton set_adapters deactivates earlier stack entries** *(2026-07-17)*
`nodes/eric_diffusion_lora_stacker.py` (~:196-206) calls
`_set_adapters_safe(pipe, name, w)` per adapter; diffusers' `set_adapters`
REPLACES the active-adapter set, so in a multi-LoRA stack only the LAST
adapter stays active at stage 1 on real diffusers (review finding 5,
2026-07-17). The comfyless path now uses one cumulative call — same pattern
applies here.
Why not now: node-pack UI path, out of the comfyless fix slice's scope;
needs a ComfyUI-side test pass.
Trigger: next touch of the stacker node, or a user report of a multi-LoRA
stack where only one LoRA takes effect.

**Nonfinite LoRA weights pass the daemon boundary** *(2026-07-17)*
`json.loads` at the socket accepts `NaN`/`Infinity` and `validate_lora_entry`
type-checks weight as float with no `isfinite` gate. A same-uid client (or
the future model-driven `--json` client) sending `"weight": NaN` causes:
unquantized path — perpetual weight-change churn (`abs(rec - NaN) <= eps` is
always False) and `set_adapters(NaN)` garbage output; quantized path —
`NaN != NaN` makes the cache key never match → full evict+reload every
request (availability churn, DQ F7 class). `refine.py` already rejects
nonfinite at its LLM boundary (`parse_constant`); the daemon boundary is the
odd one out. (Security review DLW F11, 2026-07-17.)
Why not now: `params_validation.py` / the socket parse are outside slice
DLW's declared edit scope — folding them in would be scope creep into a
separately-gated boundary.
Trigger: BINDING precondition of the `--json`/LLM-agent wiring commit (Red
Zone per the Review bar), or the next `params_validation.py` slice,
whichever comes first. Fix: `math.isfinite` in `validate_lora_entry` +
`parse_constant` rejection at the socket `json.loads`.

**plan.json ingestion: hardening preconditions for the LLM planner (slice 6)** *(2026-07-20)*
`comfyless/video.py` `load_plan` validates ADR-012-style (byte cap, unknown
keys, types/ranges, keyframe decode) under a CLI-local trust model — plans
are user-authored today. The slice-V2 code review (Fable, 2026-07-20) flagged
what must land BEFORE an LLM emits plans: (1) keyframe path containment —
no allowed-roots check, any PIL-decodable file on the machine can become
video content (ADR-018 `_check_paths` union pattern); (2) duplicate JSON
keys silently last-win — reject via `object_pairs_hook`; (3) TOCTOU on the
byte cap — `getsize` then separate `open`; read capped bytes from the open
handle; (4) no aggregate resource cap — 200 segments × 100k frames permits a
~20M-frame plan; add total-frames/runtime cap; (5) error messages echo
caller-supplied paths verbatim into what will become agent transcripts.
Why not now: slice V2's declared trust model is CLI-local (Vision
`slice-video-2-chaining.md`); slice 6 is Red Zone with its own spec +
security-auditor gate where these belong.
Trigger: BINDING precondition of the ADR-033 slice-6 (LLM planner / MCP
exposure) commit — the security review for that slice must confirm all five.

**Unused `EricQwenEditInpaintTransfer` node — removal candidate** *(2026-07-20)*
`nodes/eric_qwen_edit_inpaint_transfer.py` is imported and registered
(`nodes/__init__.py:28`, `:63`) so it loads into the ComfyUI graph, but it is
Eric's original code untouched since the initial release commit (`79c12b9`)
and has never been used in practice — Grant never adopted Eric's inpaint node
nor built one. It carries a hardcoded `ref_flags=[True, False]`
(`inpaint_transfer.py:455-456`) that encodes an inpaint-specific dual-path
arrangement nothing else depends on. Surfaced while surveying the edit
surface for ADR-035; recorded there under Deferred / Out of Scope.
Why not now: removing a registered node changes the node-pack's public
surface (a workflow JSON referencing it would break on load), which is a
separate decision from the comfyless schema work ADR-035 covers. Deleting it
mid-ADR would also be exactly the "clean up while here" §4 forbids.
Trigger: the next deliberate node-pack surface slice (any commit that adds or
removes entries in `NODE_CLASS_MAPPINGS`). Fix: confirm no workflow JSON in
`workflows/` references it, then remove the module, both `__init__.py` lines,
and its display-name mapping in one slice.

**`--json` bridge + Stable Cascade output-format handling (ADR-034)** *(2026-07-20)*
ADR-034 slice 1 wired `--output-format`/`--quality` on the CLI in-process
path only. The `--json` bridge (`generate.py` `_run_json_mode`, a
function-scoped Red Zone surface) and the Stable Cascade dispatch
(`comfyless/cascade.py`) currently **reject** both flags loudly (structured
error / stderr) rather than honor them — a deliberate slice-1 stopgap so
neither silently emits PNG when jpeg was requested. Cascade support is
already scoped to ADR-034 slice 4; the `--json` bridge was never assigned to
any ADR-034 slice (gap surfaced by the slice-1 code review, 2026-07-20).
Why not now: the bridge is machine-facing Red Zone — adding format there is
its own slice with a spec + security-auditor gate (it changes the bridge
contract and the sidecar/tEXt provenance channel per ADR-034 D4), out of
slice 1's non-Red-Zone edit scope.
Trigger: the ADR-034 daemon slice (slice 2) or the LLM-agent-bridge slice,
whichever wires machine-facing output next. Fix: thread the resolved
OutputFormat through the bridge request contract + cascade `_save_with_metadata`,
replacing the rejections with real handling; add the bridge to ADR-034's
Proposed slices list.
Partial-resolution: 2026-07-21 — the **Stable Cascade half landed** in ADR-034
slice 4 (commit `9faa17a`): `cascade.py` dispatch now resolves
`--output-format`/`--quality` and saves format-aware; the slice-1 reject
stopgap in `generate.py` is gone. The **`--json` bridge (`_run_json_mode`)
half remains OPEN** and unchanged — still its own future Red Zone slice per the
Trigger above (spec + `security-auditor` gate; never assigned an ADR-034 slice).

**Stable Cascade generation broken in the pinned `./.venv` (diffusers 0.39.0 dropped Würstchen)** *(2026-07-21)*
Stable Cascade generation builds on `diffusers.pipelines.wuerstchen`
(`DDPMWuerstchenScheduler` + the Würstchen pipeline classes), which diffusers
REMOVED after ~0.37.x. The repo pins `diffusers==0.39.0` in `./.venv`
(`uv.lock`), where that module is gone, so `build_pipelines` for any cascade
config dies at import with `ModuleNotFoundError: No module named
'diffusers.pipelines.wuerstchen'`. Confirmed 2026-07-21: `./.venv` diffusers
0.39.0 fails; the comfy-dev venv (diffusers 0.37.1) still has it and cascade
runs there. The pyright baseline already flags these imports as unresolved
(`cascade.py` wuerstchen import + `DDPMWuerstchenScheduler`), and `test_cascade.py`
passes only because it deliberately never builds a real pipeline (dispatch /
path / save-helper logic only) — so the runtime break is invisible to CI.
Impact: cascade is a supported family whose live path works ONLY under an
interpreter with diffusers ≤ ~0.37.x (currently comfy-dev's venv, incidentally
ComfyUI's). ADR-034 slice 4's cascade output-format code is therefore
unit-tested but NOT live-validated in `./.venv`; validate it under comfy-dev.
Why not now: fixing it means either (a) vendoring/shimming the removed
Würstchen pipeline into the repo, (b) pinning a second diffusers for a
cascade-only extra, or (c) formally declaring cascade a comfy-dev-only family —
each is a deliberate dep-architecture decision (ADR-013 territory), out of
ADR-034's output-format scope.
Trigger: the next diffusers bump (re-check wuerstchen availability), any move to
run cascade from `./.venv`, or a decision to make CI actually exercise a cascade
build. Interim: `docs/comfyless-stable-cascade.md` now documents the comfy-dev
requirement inline.

**MCP `extract_params` leaks `ref_images` absolute paths (ADR-035 slice-1 regression)** *(2026-07-21)*
ADR-035 slice 1 made `ref_images` a recognized `SCHEMA_KIND` key. `mcp_server.py`
`_handle_extract_params` normalizes a sidecar with `_validate_params` ALONE
(deliberately bypassing `_SKIP_SIDECAR_KEYS`), and `_render_extracted_params`
neither resolves nor drops `ref_images` — so a sidecar under `--output-dir`
carrying `"ref_images":[{"path":"/abs/..."}]` survives normalization and its
absolute paths cross the MCP boundary verbatim, breaking that function's stated
"no absolute path or directory survives" invariant (code-reviewer Fable,
2026-07-21, slice-1 review Finding 2). Before slice 1 the key was dropped as
unknown. Not consumable for generation and low-exploitability today (no writer
records `ref_images` in a sidecar yet), but it becomes a LIVE path leak the
moment ADR-035 slice 5 starts recording `ref_images`.
Why not now: the fix is in `mcp_server.py`, a Red-Zone-gated path OUTSIDE
ADR-035 slice 1's edit scope — it needs its own slice with `security-auditor`
(repo review bar) + a pin test. Not folded into slice 1 per §4 edit-scope split.
Trigger: MUST close before ADR-035 slice 5 (sidecar recording) lands; addressed
immediately as slice 1b. Fix: filter `_SKIP_SIDECAR_KEYS` (or pop `ref_images`)
in/before `_handle_extract_params`, with a pin test mirroring the existing
"no absolute path survives" assertions.
Resolved: 2026-07-21 — slice 1b added `ref_images` to the drop-outright tuple in
`_render_extracted_params` (cascade render path structurally can't emit it —
allowlist). `security-auditor` (Fable) PASS,
`docs/security/review-adr-035-slice1b-mcp-ref-images-leak-2026-07-21.md`. Pin
tests (unit + end-to-end through `_handle_extract_params`) in `test_mcp_server.py`.

**MCP extract_params: type-mismatched non-path schema fields egress arbitrary strings** *(2026-07-21)*
Pre-existing (slice-4 vintage), surfaced by the slice-1b `security-auditor`
(Fable) review. `_validate_params` (`generate.py:171-177`) KEEPS values on type
mismatch (warn-and-keep), and `_render_extracted_params` passes non-path schema
fields through verbatim — so a crafted sidecar `{"steps":"/home/gawkahn/secret"}`
egresses an absolute-path string through a numeric field across the MCP boundary,
breaking the letter of "no absolute path or directory survives". Same class the
cascade render path already closed with number-or-None coercion
(`mcp_server.py` `_CASCADE_NUMERIC_FIELDS`) and LoRA-weight coercion. Also:
`model_family` is re-injected verbatim from the raw sidecar on both render paths
(arbitrary-string egress), inconsistent with the cascade dtype value-allowlist.
Exfiltration value is LOW (prompt/negative_prompt already pass arbitrary sidecar
text verbatim by design), which is why the auditor rated it MEDIUM, not a 1b
blocker, and explicitly said fixing it in 1b would be scope creep.
Why not now: out of ADR-035 slice-1b edit scope (the 1b fix is the `ref_images`
drop only); this is a distinct, pre-existing hardening on a Red-Zone path.
Trigger: next deliberate `mcp_server.py` extract-hardening slice, or when the MCP
output/opaque-handle work (ADR-034 slice 3 / MCP edit ADR) touches this surface.
Fix: in `_render_extracted_params`, coerce numeric/bool schema fields to
type-or-drop (mirror `_CASCADE_NUMERIC_FIELDS`), value-allowlist `model_family`,
or run `_validate_params` in a strict drop-on-mismatch mode for the extract path.
Needs `security-auditor` (Red-Zone path).

**ref-image ingestion: non-regular-file path hangs the decode (slice-4 daemon DoS precondition)** *(2026-07-21)*
Surfaced by the ADR-035 slice-2 `security-auditor` (Fable) review
(`docs/security/review-adr-035-slice2-ref-image-ingestion-2026-07-21.md`,
Finding 2, LOW). `load_ref_image_capped` (`comfyless/ref_image.py`) opens the
path with plain `open(path, "rb")`. On a FIFO this blocks in `open(2)` until a
writer appears (a `/dev/tty`-style path blocks in `read`), an indefinite hang
that no byte/pixel cap can reach. Benign under slice 2's CLI-local trust
(operator self-harm; the repo prefers warn-don't-block for user-typed paths).
Why not now: out of ADR-035 slice-2 edit scope (the helper's content-safety core
only). Becomes a real DoS at slice 4: the daemon decodes paths inside
`ref_image_roots`, which defaults to `--output-dir` — a tree lower-trust flows
write into — so a same-UID `mkfifo output/kf_003.png` hangs the VRAM-holding
daemon on the next chained keyframe request. An `os.fstat`+`S_ISREG` check does
NOT close it (the hang is in `open` itself, before any fstat); the fix is a
non-blocking open at the daemon decode site (`os.open(path, O_RDONLY|O_NONBLOCK)`
then reject non-regular / would-block).
Trigger: slice 4 (daemon `ref_image_roots` exposure) — this is a HARD
precondition of that slice, not an optional hardening. Do not land daemon
ref-image decode without a non-regular-file rejection at the decode site.
Fix: `O_NONBLOCK` open + `S_ISREG` check in the daemon's ref-image decode path;
mirror the existing `_PATH_FIELDS` NUL-byte pre-check (6e) that guards the same
boundary.
Resolved: 2026-07-21 — ADR-035 slice 4. `load_ref_image_capped`
(`comfyless/ref_image.py`) now opens with `os.open(path, O_RDONLY|O_NONBLOCK)`,
`fstat`s, and raises `RefImageError` on any non-`S_ISREG` target before the read
(fd never leaked/double-closed across the reject/success/error paths). Shared
decode site, so foreground benefits too. Negatives pinned in `test_ref_image.py`
(FIFO / directory / symlink-to-FIFO rejected; symlink-to-regular accepted).

**ref-image daemon: TOCTOU symlink-swap between containment and decode** *(2026-07-21)*
ADR-035 slice-4 `security-auditor` (Fable) LOW. `_check_ref_paths`
(`comfyless/server.py`) resolves each ref path with `realpath` at check time,
but `load_ref_image_capped` re-resolves at `os.open` time. An actor able to
write inside a ref root (note: the daemon's own `--output-dir` is always a ref
root) can swap an in-root symlink to an out-of-root target between check and
open, so the daemon reads + VAE-encodes a file outside every ref root; the pixels
land in the returned image. Why not now: under the same-UID solo model, anyone
who can plant symlinks in a ref root can already read the target directly — this
is a confused-deputy defense-in-depth gap, not a privilege boundary. It is the
same trust-class shift the ADR already wills to the MCP ADR (output-dir
read-back / cross-plant loop, ADR-035 Deferred). Trigger: when an agent-driven /
less-trusted transport fronts the daemon (the MCP edit ADR). Fix: write the
realpath back into `req["ref_images"][i]["path"]` after `_check_ref_paths`
passes (shrinks the race to a single component), or open-then-fstat containment
for full closure.

**ref-image daemon: slice-4 client → pre-slice-4 daemon silently drops refs** *(2026-07-21)*
ADR-035 slice-4 `code-reviewer` (Fable) LOW (operational). ADR-020 daemons are
long-lived and persist across upgrades. A slice-4 client that delegates a
`--ref-image` run to a daemon started from PRE-slice-4 code sends `ref_images`
on the wire; the old daemon passes it through (`ref_images` is `_KIND_LIST`
since slice 1 / unknown-key passthrough before) but its `_handle_generate` never
forwards refs to `generate()` → the image is generated WITHOUT references and
returns `ok` → silent keyframe drop, defeating the very invariant slice 3's
blanket no-delegate guard protected. There is no wire version/capability
handshake on the daemon socket. Why not now: the cheap mitigation (restart
daemons after upgrade before ref-delegated runs) matches the existing DLW
daemon-restart guidance; a capability gate is a clean separate slice. Trigger:
the next daemon-protocol slice, OR any report of a delegated ref run producing
a text2img result. Fix: client checks a daemon `ping`-reply capability/version
before including `ref_images`; until then, restart daemons on upgrade (add to
the Comfyless manual's daemon-restart note).

## 2026-07-22 — `ref_images` replay gate absent on the `--json` / MCP transports
ADR-035 slice-5 `code-reviewer` (Fable) INFO. The decision-7 file-derived
replay-trust gate (`_apply_replay_ref_trust` → `_gate_file_derived_refs`) lives
only on the interactive CLI (`_run_cli_mode`). `_run_json_mode` and the MCP
server are safe TODAY only because they never forward `ref_images` from params
into `generate()` — they drop it by omission, not by a gate. Whoever later
wires reference-image replay into the `--json` bridge or an MCP `edit`/`generate`
tool inherits the row-2 obligation and MUST route file-derived paths through the
same gate (outside-roots refusal + hash/moved warnings), NOT honor them as
literal paths. Why not now: those transports have no ref-image feature yet;
adding the gate pre-emptively would be dead code. Trigger: any commit that lets
`ref_images` reach `generate()` from `--json` stdin or an MCP tool argument.
Fix: call `_apply_replay_ref_trust` (or an MCP-appropriate equivalent with the
server's `ref_image_roots`) before those paths reach the decode site.

## 2026-07-24 — stagnation-resampled iterations are invisible to the planner's trajectory context
ADR-037 D2-addendum `code-reviewer` (Fable) SHOULD, accepted as documented
deferral. A stagnation escape changes prompt AND seed in one iteration, but
`history_record` carries no `seed_resampled` flag, so the D1 trajectory
context attributes the next score delta entirely to the prompt edit — the
planner can learn a false lesson ("that rewrite worked/failed") on every
escape iteration. Why not now: adding a field to judge-bound history touches
the F8-P surface and warrants its own review pass; the escape only fires on
already-stagnant runs where attribution value is low. Trigger: planner
visibly re-proposing/reverting edits it "learned" from resampled iterations,
OR the next history-shape slice. Fix: a boolean `seed_resampled` field in
`history_record` (loop-owned state, not filesystem drift — slice-2
forward-constraint permits it), plus rubric line telling the judge scores on
such iterations reflect a new sample, not the edit.

## 2026-07-25 — refine loop does not ACT on failed LoRA applications
Parity-audit slice 2 `code-reviewer` (Fable) NIT. `surface_wire_warnings`
returns the LoRA-failure count and refine now logs the warnings, but no
caller consumes the count: the loop's accounting (history record, verdict,
promotion) is unchanged, so the planner can re-propose a LoRA that its own
prior iteration failed to apply, and a score delta caused by a MISSING
adapter is attributable only by a human reading the log. Why not now:
feeding failure state into the planner's context or the promotion rule
changes decision-making on a Red Zone file and belongs in its own slice with
its own review (it also interacts with the pending `--pin-lora` work and the
v3 promotion gate). Trigger: observing the planner re-propose a failed LoRA,
OR the next refine loop-accounting slice. Fix: carry the count into the
history record (a `lora_failed: true` flag is path-free and F8-P-safe) and
add a rubric line telling the planner that a flagged iteration's scores do
not reflect the proposed LoRA.

## 2026-07-25 — `schedule` has no value allowlist at the machine boundary
Parity-slice-1 `security-auditor` (Fable) LOW. Pre-existing, surfaced because
the slice widens the population of schedule-carrying wire requests (refine now
always sends one). `params_validation` types `schedule` as `_KIND_STR` with no
value check, so a `--json`/MCP/sidecar-supplied name outside `SCHEDULE_NAMES`
passes the boundary, `_sigma_schedule_gate` treats it as non-linear,
`build_sigma_schedule` silently shapes it as linear — and the
"[comfyless] sigma schedule: {name} (flow-match...)" log line plus the sidecar
then RECORD a schedule that did not run. Integrity misreport, not an exploit;
no dispatch or eval on the string. The CLI is unaffected (argparse
choices-gated on both generate and refine). Why not now: it belongs with the
machine-boundary validator, not this refactor, and the fix has two shapes
worth choosing between deliberately. Trigger: next params-validation slice, OR
any report of a sidecar recording a schedule that didn't run. Fix: either add
`schedule` to the dtype-style value-allowlist mechanism in
`params_validation.py`, or have `build_sigma_schedule` return the EFFECTIVE
name so the log and sidecar say "linear (fallback from <name>)".

## 2026-07-25 — `refine_loop` trusts its `duel_band` argument (CLI-layer validation only)
ADR-039 slice-2 `security-auditor` (Fable) INFO. `main()` validates
`--duel-band` as finite and >= 0 and exits 2 otherwise, but `refine_loop`
itself re-checks nothing: a NaN passed programmatically makes every band test
False, so the run silently disables duels and reverts to the promotion rule
ADR-039 supersedes, with nothing in the log to say so. Latent today — `main()`
is the only caller and validates first. Why not now: a second validation layer
inside the loop is dead code until a second caller exists, and the right shape
depends on what that caller is (a machine boundary wants typed validation, not
a scattered isfinite). Trigger: the day `refine_loop` is reachable from the MCP
server or the `--json` bridge — both already flagged in CLAUDE.md as
Red-Zone-on-scope-change — or any second in-repo caller. Fix: validate
`duel_band` (and the other loop floats: weights, `until_composite`) at that
boundary, or move the checks into `refine_loop` and have `main()` surface them.

## 2026-07-25 — reviewer subagents have no shell, so no ADR-039 review ran `git diff`
Process debt, not code. All six `code-reviewer` / `security-auditor` passes
across ADR-039 slices 1 and 2 reported the same limitation: the agent had only
Read/Grep/Glob, could not run `git diff`, and therefore reviewed working-tree
FILE STATE rather than the change. Each flagged that it could not rule out
hunks elsewhere in the 3200-line `refine.py` or in other tracked files, and
asked the parent session to confirm `git diff --stat`. That confirmation is a
human/parent step that could be skipped silently, and it is exactly the check a
reviewer exists to perform. Why not now: it is an agent-definition change under
`~/.claude/agents/` affecting every project, not a change to this repo, and it
wants one deliberate slice with its own §10A commit. Trigger: the next Red Zone
review in any project, or the ADR-039 slice-3 review. Fix: add `Bash(git diff*)`
/ `Bash(git status*)` to the reviewer agents' tool lists, or have the parent
session write the diff to a file and hand the reviewer that path.

## 2026-07-26 — catalog truncates trained INSTRUCTION templates to 64 chars
Found while diagnosing why the refine loop never proposed the face-swap LoRAs
(Grant, 2026-07-26). `sanitize_trigger_words` caps each entry at
`TRIGGER_WORD_CAP` (64 B), which is right for trigger WORDS ("ohwx man",
"pixel art") but wrong for the edit-tool LoRAs whose civitai `trainedWords` is
a full instruction TEMPLATE. `bfs_head_v5_2511_merged_version_rank_16_fp16`
ships a ~380-char template ("head_swap: start with Picture 1 as the base
image, keeping its lighting, environment, and background. remove the head from
Picture 1 completely and replace it with the head from Picture 2, strictly
preserving the hair, eye color, nose structure of Picture 2. copy the direction
of the eye, head rotation, micro expressions from Picture 1 ...") and the
catalog stores `"head_swap: start with Picture 1 as the base image, keeping its
l"`. The planner sees that fragment, so even when the LoRA IS offered it cannot
reproduce the phrasing the LoRA was trained on — which is plausibly why the
2026-07-25 face-swap runs got nothing out of this class of LoRA. Note the
template is also, verbatim, the "name the specific features instead of
'maintain identity'" strategy in the Backlog idea of the same date.
Why not now: raising the cap is a catalog-plane change (schema value, sanitizer
bound, FTS content size, and the planner-visible payload budget all move
together), and it wants a deliberate decision about how much third-party text
may enter LLM context — the offers are third-party-sourced metadata, an F8-P
adjacent surface. Trigger: the face-swap end-to-end test, or the next catalog
slice. Fix: a separate longer cap for template-shaped trained words (e.g.
`TRIGGER_TEMPLATE_CAP` ~512 B, one per entry), or a dedicated
`instruction_template` description column that the planner sees whole.

## 2026-07-27 — daemon ValidationError refusals leave no server-side trace
Found by the ADR-040 slice-1 security review (MEDIUM). Every other
security-relevant refusal on the daemon surface logs before responding —
`PathError` and `RefPathError` both `_log(...)` the redacted request — but the
generic `ValidationError` branch in `_handle_connection` responds silently. A
caller probing the new `report_roots` flag across `generate` / `unload`
therefore leaves no record at all, which matters because ADR-040 D2a
deliberately accepts a residual (the daemon cannot discriminate an MCP caller
from a CLI caller) whose only cheap detective control is a log line. The
disclosure half was closed in slice 1 — a successful `report_roots` ping now
logs a count-only line — but the refusal half was left alone.
Why not now: the branch is generic, so logging there changes behavior for
EVERY validation error on the surface, not just this flag. That is a
volume/PII-shaped decision of its own (request bodies carry prompts, and the
existing path-error logs redact `prompt` explicitly for that reason), and
bundling it into a slice about a ping field would be the "clean up while here"
the constitution forbids. Trigger: the next `server.py` slice, or the first
time an operator needs to reconstruct who probed the daemon. Fix: log the
refusal with the same redaction the `PathError` branch uses
(`{k: v for k, v in req.items() if k != "prompt"}`), or add a targeted log for
value-check refusals only.

## 2026-07-27 — comfyless/ per-root pyright baseline mixes concurrently-edited files
ADR-042 chose per-ROOT (not per-file) pyright baselines. `comfyless/generate.py`
is under active concurrent edit by the ADR-040 session (file lease, see the
2026-07-27 typecheck-drawdown handoff) while this session drives the rest of
`comfyless/` down. Both sessions share one working tree, so a live
`pyright`/`just typecheck` run picks up whichever WIP edits are on disk at that
instant — observed directly during this slice: `generate.py`'s live count
moved 7→8 between the handoff being written and this slice's mechanism
commit, with no intervening commit on either side. The `comfyless` root total
therefore isn't purely a function of committed history; it can shift under a
session that touches nothing in its own lease.
Why not now: the alternative (per-file baselines) is the granularity ADR-042
explicitly rejected as unmaintainable at 72-files-repo scope, and the
concurrent-session pattern itself is expected/normal in this repo (see
`feedback_concurrent_sessions` memory). Splitting `comfyless/generate.py` into
its own baseline root just for this would special-case one file inside one
directory, which the pyright root-grouping mechanism (first path segment)
doesn't support without real complexity.
Trigger: a ratchet false-block where the blocked root's regression is
entirely inside a file the blocked commit never touched — check `git diff`
against the blocked root's files before assuming the commit under review
introduced the regression. Fix, if it recurs often: extend
`scripts/typecheck-per-root.sh`'s grouping to support a second-level
override (e.g. `comfyless/generate.py` as its own baseline key) so a leased
file can ratchet independently of the root it lives in.

## 2026-07-27 — comfyless/cascade.py's Stable Cascade imports are untested against real diffusers
Found by code-reviewer during the ADR-042 comfyless/ drawdown (verifying the
`cascade.py` import-path fixes). `test_cascade.py` replaces `build_pipelines`
wholesale with a mock (`test_cascade.py:659-660`), so its 152/152 green does
NOT exercise the real `from diffusers... import StableCascadePriorPipeline,
StableCascadeDecoderPipeline, DDPMWuerstchenScheduler` / `PaellaVQModel`
imports this slice fixed — pyright is the only thing that caught the dead
`PaellaVQModel` import path (see the ADR-040/ADR-042 handoff commit history),
and would be the only thing to catch the next one. Compounding factor: the
pinned diffusers 0.39.0 already carries `_last_supported_version = "0.35.2"`
on `StableCascadePriorPipeline` (`DeprecatedPipelineMixin`, warning-only, not
a raise, per code-reviewer) — this pipeline is upstream-abandoned, so its
internal import paths are more likely than most to shift or vanish on a
future diffusers bump, silently, since nothing here would catch it except a
pyright run.
Why not now: writing a real (non-mocked) smoke test for `build_pipelines`
needs real Stable Cascade weights on disk or a much heavier mock that
actually imports the real diffusers classes — a bigger lift than this
drawdown slice's scope (fixing existing pyright errors, not writing new
coverage). Trigger: the next diffusers version bump (re-run pyright on
`comfyless/cascade.py` specifically as part of that bump's proof, don't rely
on `test_cascade.py` staying green), or a Stable Cascade generation request
actually failing in practice. Fix: either a lightweight test that imports
(not mocks) `PaellaVQModel`/`StableCascadeDecoderPipeline`/etc. and asserts
they're still importable from the paths `cascade.py` uses, or accept the
residual and rely on the pyright ratchet catching it at the next diffusers
bump (this repo's precedent for "pyright is the safety net, not the test
suite" — see ADR-032's `comfy.*` missing-import cluster).

## 2026-07-27 — refine's --seed-image is the only reference not pinned into the run dir
Found independently by `code-reviewer` and `security-auditor` during the
ADR-040 slice 2b review (`docs/security/review-adr040-slice2b-2026-07-27.md`,
MEDIUM). `pin_static_refs` copies every `--ref-image` into `<run dir>/refs/`
and the ADR-038 D5 docstring gives the reason: a reference is consumed on TWO
channels — pinned bytes for the judge, and a PATH re-read every iteration by
whoever generates — so leaving the path unpinned reopens the TOCTOU that
amendment closed. `--seed-image` has that exact shape and is not pinned:
`source_img = load_seed_image_capped(edit_source)` pins the judge's comparison
anchor at loop entry, while `current_source` (the same operator path) is
re-opened by the daemon on every generation until the first promotion. A
mid-run replace/re-encode/truncate — concurrent session, editor save, sync
client — makes generation condition on new bytes while the judge scores
identity against old ones, silently breaking "scores describe the
generation's inputs". This is an INTEGRITY defect independent of ADR-040: it
exists daemon or no daemon.
Second, smaller consequence: because the seed is not relocated into the run
dir, an out-of-roots `--seed-image` under a daemon still latches the run
in-process (ADR-037 D5) and can OOM. Slice 2b WARNS about that at entry rather
than refusing — a deliberate divergence from D3a's refuse-at-entry ruling for
the sibling `generate --ref-image` surface, recorded in the ADR-040 Changelog.
Pinning closes both: the run dir is inside a ref root by construction under D1,
so the failure becomes unrepresentable rather than warned.
Why not now: it is a behavior change to ADR-037 D5's edit-source contract
(what `edit_source` points at, and what the judge's anchor is loaded from), so
it needs an ADR amendment and its own reviewable slice — not a bolt-on to a
slice whose boundary is D1+D3. ADR-040's "Alternatives Rejected" also asserts
`pin_static_refs` already covers this ("no new move step is needed"), which is
true for `--ref-image` and false for `--seed-image`; that sentence needs
correcting in the same slice.
Trigger: the next change to refine's edit-source handling, ADR-038/ADR-037
edit-ref work, or a live run where the seed image is observed to change
mid-run. Fix: after the exclusive `makedirs`, copy the seed to
`<run dir>/refs/seed<ext>` with the existing `shutil.copyfile` primitive and
point `edit_source` at the copy — only iteration 0 uses it as `current_source`
(from iteration 1 the source is a candidate already inside the run dir), and
the byte/pixel caps already ran at entry.
Resolved: 2026-07-28 — `pin_seed_image` (ADR-037 D5 amendment 2026-07-28;
`docs/security/review-adr037-seed-pin-2026-07-28.md`). Both consequences
closed: the TOCTOU (nothing reads the operator's path after entry) and the
out-of-roots latch (the daemon only sees the loop-owned copy, so ADR-040's
warning was deleted rather than kept). Three deviations from the fix sketched
above, each forced by a review finding: the copy lands in `<run dir>/source/`,
NOT `refs/`, because `pin_static_refs` opens with an unconditional
`rmtree(refs/)` and sharing the directory would make correctness depend on
call order; `shutil.copyfile` was the wrong primitive — it is a second,
uncapped, unguarded read, so the pinned bytes were never the validated bytes
and the write was bounded by nothing (both reviewers, HIGH) — replaced by a
capped guarded re-read, a SHA-256 equality check, and an `O_EXCL` 0600 write;
and "the caps already ran at entry" was true but weak — that entry check used
`load_seed_image_capped`, so the first decode of the operator's file skipped
the format allowlist and the regular-file guard, and it now uses
`load_ref_image_capped` too. ADR-040's "no new move step is needed" sentence
corrected in the same slice, as this entry required.

## 2026-07-28 — pin_static_refs has the same validate-then-copy window pin_seed_image just closed
Found by `code-reviewer` and `security-auditor` (both, HIGH) while reviewing
the ADR-037 D5 seed-pin amendment, against `pin_seed_image`. The finding
applies verbatim to its older sibling: `pin_static_refs` calls
`load_ref_image_capped(r.path)` to validate and hash one set of bytes, then
`shutil.copyfile(r.path, dst)` re-reads the path from scratch. The bytes that
land in `<run dir>/refs/ref_NN.ext` are therefore never the bytes that passed
the byte cap, the pixel cap, the format allowlist, or the regular-file guard,
and `StaticRef.sha256` — which decision 7's replay gate compares against —
describes a file we may no longer hold. `copyfile` imposes no size limit and
rejects FIFOs but not block/char devices, so a `--ref-image` that grows or is
swapped between the two reads produces an unbounded write into the run dir,
and the daemon then refuses the loop's own artifact mid-run at its own
`load_ref_image_capped` decode. `open(dst,'wb')` also follows a symlink and
truncates, so a planted `refs/ref_00.png` on a group-writable explicit
`--output-dir` redirects the write.
Why not now: `pin_seed_image` was the slice under review and its boundary was
the SEED. Changing `pin_static_refs` in the same commit would have mixed an
unreviewed change into a Red Zone diff whose scope both reviewers checked, and
ADR-035 slice 2's review explicitly signed off on "no check-then-use window"
for the ingestion path — so correcting that record is part of the work, not a
drive-by edit.
Trigger: the next change to ADR-038 static-ref handling, or any report of a
mid-run "reference exceeds cap" failure on a file that passed at entry.
Fix: the shape `pin_seed_image` now uses — re-read via
`_read_ref_bytes_capped`, refuse on SHA-256 mismatch against the validated
read, and write through `os.open(..., O_WRONLY|O_CREAT|O_EXCL, 0o600)` instead
of letting `copyfile` open the destination. Extracting one shared
pin-bytes-to-run-dir helper for both call sites is the natural form.

## 2026-07-28 — Kohya SDXL LoRAs stay `wrong_arch` after the ADR-014 native-convert amendment

The 2026-07-28 ADR-014 amendment routes non-diffusers LoRA key layouts through
the base family's own `LoraLoaderMixin.lora_state_dict()` before shape-matching,
which recovers the ~54 Kohya **Flux** false negatives (measured 494/494 layers,
100%). The ~13 Kohya **SDXL** files (pony / SDXL / Illustrious, `lora_te1_*` +
unet) are NOT recovered — they still measure 0% and still classify `wrong_arch`.
Two independent causes, both confirmed empirically against
`loras/pony/Cunnilingus Close Up V2.safetensors`:
1. `_convert_non_diffusers_lora_to_diffusers` emits `.lora.down.weight` /
   `.lora.up.weight`, which are absent from `_ADAPTER_SUFFIXES`
   (`nodes/eric_diffusion_lora_check.py:24` has `.lora_down.weight`, with an
   underscore). `_strip_adapter_suffix` therefore returns `sfx=None` for every
   key and ZERO layers are extracted — the match is 0% before naming is even
   consulted.
2. Deeper, and not fixed by (1): the converted keys carry flat block indexing
   (`unet.down_blocks.4.1.proj_in`) while the SDXL base index is
   `down_blocks.1.attentions.0.proj_in`. diffusers performs further remapping
   inside the UNet loader that has not been traced.
Also note the SDXL LoRAs carry text-encoder keys (`lora_te1_*`) that no
transformer/unet base index can ever match — the base is a unet dir, so those
layers are structurally unmatchable and any future fix must decide whether they
count toward `total_layers`.
Why not now: Grant's scope call on 2026-07-28. Cause (2) is an open-ended dig of
unknown depth, and holding the proven Flux fix behind it would keep ADR-041
(semantic LoRA offers) blocked on a corpus missing a third of its Flux LoRAs.
Trigger: the SDXL/pony/Illustrious families becoming load-bearing for refine or
MCP offers, or any report that pony LoRAs are missing from catalog search.
Fix: start at cause (1) — it is a two-token suffix addition and makes the real
failure in (2) measurable instead of masked. Then trace diffusers'
`unet.load_attn_procs` / `_maybe_expand_lora_state_dict` remapping and decide
whether to shape-match post-remap or index the unet under both conventions.

## 2026-07-28 — `R_ARCH_MISMATCH_DIFFUSERS_ONLY` is a dead reason-code constant

`scripts/lora_audit.py:121` declares
`R_ARCH_MISMATCH_DIFFUSERS_ONLY = "arch_mismatch_diffusers_only"` and nothing in
`scripts/`, `nodes/`, or `test_lora_audit.py` references it. It reads like it
was minted for exactly the Kohya-vs-diffusers case the 2026-07-28 amendment
addresses, but its name asserts a *mismatch* — the opposite of what that
amendment records (those files are `usable`), so it was deliberately not
repurposed; the amendment added `R_OK_NATIVE_CONVERT` instead.
Why not now: §4 edit-scope discipline — deleting or wiring an unrelated constant
inside the amendment's diff is a "clean up while here" change. It is inert, so
it costs nothing to leave.
Trigger: the next slice that touches the reason-code block, or the SDXL
follow-up above (which may legitimately want a genuine arch-mismatch code).
Fix: delete it, or wire it in the SDXL slice if a real
"converted, still mismatched" reason turns out to be wanted. Confirm the closed
reason-set docs in ADR-014 §3 agree either way.

## 2026-07-28 — native-convert coverage floor is a heuristic, not a coverage proof

The ADR-014 amendment guards partial conversions with
`_COVERAGE_FLOOR = 0.5` in `scripts/lora_audit.py._try_native_convert_match`:
the converted dict must retain at least half the source's adapter layers.
This catches the gross case (`code-reviewer` H-2 reproduced a stub returning
1 of 10 source layers scoring `key_match_pct 100.0` and being promoted to
`usable`), but it is NOT a conservation law. Converters that split fused
projections — Kohya `qkv` into `to_q`/`to_k`/`to_v`, so one source layer
becomes three — inflate `converted_layers` above `source_layers`. A file
whose converter understood only ~2 of 10 source layers can still emerge with
~6 converted layers and clear a floor of 5. The ratio is therefore a floor on
gross loss, not a measure of semantic coverage.
The compensating controls, none of them proofs either: base config ordering
(`flux` before `flux2`, so the correct converter usually wins first), the
strict verdict requirement on the converted path (OK / NORM_TARGETING only —
`DIM_MISMATCH`-at-50% is refused there), and `source_layers` /
`converted_layers` being recorded in the manifest's `native_convert` field so
the ratio is auditable rather than hidden.
Why not now: a sound measure needs per-converter knowledge of which SOURCE
keys were consumed. diffusers exposes that inconsistently — some converters
raise, some warn-and-drop (`_convert_kohya_flux2_lora_to_diffusers` logs
`remaining_keys`), some silently pass through. Reconstructing it per family is
the mapping-table coupling the amendment deliberately rejected.
Trigger: a LoRA showing up in catalog search under a family it does not
belong to, or any `native_convert` entry whose recorded
`converted_layers`/`source_layers` ratio looks implausible. Also revisit on
any diffusers bump that changes converter drop behaviour.
Fix: capture the converter's own warning stream, or diff the source key set
against the union of keys each converter's mapping consumed, and record true
consumed-fraction instead of the layer-count ratio.

## 2026-07-28 — `_resolve_lora_mixin` re-parses model_index.json per (file × base)

`scripts/lora_audit.py._try_native_convert_match` calls `_resolve_lora_mixin`
on every fall-through file for every base, and that helper re-opens
`<base>/../model_index.json`, re-parses it, and re-walks the pipeline class
MRO each time. With 9 configured bases and ~600 fall-through files that is
~5400 small JSON reads per run where 9 would do. `code-reviewer` flagged it
non-blocking.
Why not now: the reads are tiny and the fall-through path already loads full
LoRA state dicts, which dominates by orders of magnitude — the measurable cost
is noise against that. Adding a cache field to `BaseSpec` is a widening of the
reviewed diff for no observed win.
Trigger: audit runtime becoming a complaint, or the base count growing well
past 9.
Fix: resolve once in `_prepare_bases` and cache the mixin on `BaseSpec`
alongside `param_dict`, which is already populated there lazily.

## 2026-07-29 — the unpinned-tool guard covers `npx` but not `uvx`

`start-mcpo.sh:60` ran `exec uvx mcpo` for months. `uvx` resolves the tool AND
its dependency closure at LATEST on every invocation — structurally the same
floating-version footgun as `npx <tool>`, which §14 of the global constitution
already forbids and enforces with a T1 PreToolUse hook
(`~/.claude/hooks/block-unpinned-gitnexus.sh`, added 2026-07-22).
It bit on 2026-07-29: mcp 2.0.0 removed the `streamablehttp_client` symbol that
mcpo 0.0.20 imports, so a bridge that had worked for weeks failed at startup
with an ImportError after a reboot, having silently pulled an artifact nobody
reviewed. Exactly the supply-chain surface §11 exists to close.
Grant's read, and it is correct: "the fact it broke at all indicates a problem
we let slide before." The npx instance got a hook; the uvx instance did not,
because the hook was written around the specific TOOL (gitnexus) rather than
the PATTERN (unpinned ephemeral-environment runners).
Fixed in this repo: `start-mcpo.sh` now pins both sides —
`uvx --from "mcpo==$MCPO_VERSION" --with "mcp==$MCP_VERSION"`. Note that
pinning only the transitive (the first fix attempted) leaves the tool floating
and is NOT sufficient.
Why the remaining gap is not now: widening
`block-unpinned-gitnexus.sh` to cover `uvx`/`pipx run`/bare `npx` is a
user-scope T1 hook change that affects EVERY project and every session, and a
too-broad matcher would block legitimate one-off invocations. That needs its
own deliberate slice with its own negative tests, not a drive-by widening
during an unrelated feature.
Trigger: the next unpinned-runner surprise in ANY repo, or the next time
`~/.claude/hooks/` is touched. Repo-local scan is currently clean —
`start-mcpo.sh:60` was the only `uvx`/`npx`/`pipx run` call site here.
Fix: generalise the hook to a pattern matcher over ephemeral-runner commands
(`uvx`, `npx`, `pipx run`, `bunx`, `dlx`) that requires a version specifier in
the tool token, with an explicit allowlist escape; then re-scan every indexed
repo for call sites.

## 2026-08-01 — a delegated run that ERRORS loses its `edit_warnings`

**What:** `edit_warnings` is the wire channel that carries daemon-side notices
back to the client (`generate.py:1390-1395`), surfaced by
`surface_wire_warnings` (`generate.py:4030`). It rides in the response
`metadata` — and error responses carry no `metadata` at all
(`server.py:1103-1105`). So every warning raised before the failure point is
lost on a delegated run that raises.

Surfaced by the ADR-044 security review (Finding 3) while checking whether the
Krea-2 identity edit's warnings survive delegation. On the SUCCESS path they do,
which is why ADR-044 deletes its draft's client-side emission rather than adding
more of it.

**Why not now:** the complete fix is a general daemon→client notices channel
that is populated independently of success. That widens a Red Zone wire schema
(`comfyless/server.py`), and every family's warnings would then need auditing
for path disclosure — the daemon knows absolute paths the client may not, so a
warning that is safe in a daemon log is not automatically safe on the wire. Not
a rider on a feature slice.

**Trigger:** the second family that needs a warning to survive a FAILED
delegated run. Today an errored run surfaces its error, so the lost warning is
never the user's only signal.

## 2026-08-01 — `NaN` `ref_boost` / `nag_scale` reach the bias math unguarded

**What:** `json.loads` accepts the bare token `NaN`, and the canonical validator
type-checks floats without checking finiteness. A `NaN` `ref_boost` flows into
`math.log(max(ref_boost, 1e-4))` (`pipelines/krea2_identity_edit.py:280`) and
propagates through the attention bias: corrupted output for that one request, no
crash, no message. The `nag_*` quadruple has the same shape.

**Why not now:** this is pre-existing and family-wide, not an identity-edit
defect — fixing it properly means a finiteness predicate in the canonical
validator (ADR-012's owner of machine-boundary type predicates) applied to every
float field, which is its own slice with its own negative tests. Doing it inside
an identity slice would fix one call site and leave the pattern.

Related and still open: the nonfinite-weight boundary gate noted under the Qwen
LoRA work. Same root — float fields typed but not range-checked at the boundary.

**Trigger:** the finiteness slice for `--lora` weights; fold both in together.

## 2026-08-01 — the >2-source identity refusal fires AFTER the ~30 GB load

**What:** `_resolve_ref_family_support` (`comfyless/generate.py:1993-1999`)
refuses a third `--ref-image` on the krea identity path with an `InferenceError`
— correct, but it runs inside `generate()`, so on a delegated run the daemon has
already loaded the pipeline before refusing. The reference count is knowable
client-side for free.

**Why not now:** a latency nicety, not a safety property. The daemon must remain
the authoritative gate regardless (a client-side check can go stale, exactly as
documented for `refuse_out_of_roots_refs`), so any client-side version is a
second spelling of a rule that already exists — ADR-040 D3a's shape, and that
one was justified by converting a *crash* into a clean refusal, which this is
not.

**Trigger:** if a user hits it often enough to complain, or if a client-side
entry-gate slice is opened for another reason and this can ride along.

## 2026-08-01 — the cascade branch still accept-and-drops the WEIGHT-PATH fields

**What:** the MCP `generate` tool routes to `_handle_generate_cascade` purely on
`cascade_config` presence, and that routing happens BEFORE `_handle_generate`'s
field guards. ADR-044 commit 3 closed the reference/identity tuple on both
branches (`_GENERATE_UNSUPPORTED_REF_FIELDS` is now checked at cascade entry
too), but `_GENERATE_REMOVED_FIELDS` — the weight paths `transformer_path` /
`vae_path` / `text_encoder_path` / `text_encoder_2_path` — is still only checked
on the non-cascade branch. A payload `{prompt, cascade_config, vae_path: ...}`
is accepted, type-validated and silently dropped.

Found by `security-auditor` during the commit-3 review (Finding 3). The half
that made ADR-044's own proof hook false was fixed in that commit; this half is
pre-existing and predates it.

**Why not now:** closing it changes behaviour for a caller shape ADR-044 never
touched, on a Red Zone file whose scope extension Grant approved specifically as
"the reference/identity fields on the generate surface." Widening that after the
approval is the wrong order. It is also inert rather than dangerous today:
cascade dispatches to `comfyless.cascade`, never `generate()`, so a dropped
`vae_path` reaches no loader — this is an N1 consistency gap, not an exposure.

Related, same review, and the reason to do them together: `generate_cascade`
already emits INFO notices for dropped `quant` / `nag_scale`, so silent drops
are against its own convention.

**Trigger:** the next `mcp_server.py` slice that touches the cascade branch.
The clean fix is hoisting both rejection loops above the cascade/non-cascade
split rather than duplicating a second copy.

## 2026-08-01 — cascade-bypass field list, corrected and completed

**Amends the entry above** ("the cascade branch still accept-and-drops the
WEIGHT-PATH fields"), which named only four fields. `_GENERATE_REMOVED_FIELDS`
grew to seven the same day, so the recorded gap was narrower than the real one.
Flagged by `security-auditor` with the point that matters: the whole reason that
commit existed is that a previous list was incomplete, so leaving the paper
trail incomplete repeats the failure in the register rather than the code.

**The complete set of fields the cascade branch still accept-and-drops:**
`transformer_path`, `vae_path`, `text_encoder_path`, `text_encoder_2_path`,
`upscale_vae_path`, `upscale_vae_subfolder`, `refiner_path`.
(`ref_images` / `ref_boost` / `grounding_px` / `identity` ARE closed on the
cascade branch — that half shipped in ADR-044 commit 3.)

**Plus two fields that accept-and-drop on BOTH branches:** `refiner_steps` and
`refiner_cfg`. They are COMFYLESS_SCHEMA members that pass the payload filter
into `gen_params`, are never forwarded, are never rejected, and SURVIVE
`extract_params` outbound — so extracting a CLI refiner sidecar hands the agent
orphaned numerics a replay silently ignores. Non-path (int/float), and a
`generate(**gen_params)` refactor would forward them into a guaranteed no-op
because `refiner_path` is rejected and `generate()` gates the chain on it. Low
risk, same silent-drop pattern.

**Why not now:** unchanged from the entry above — closing the cascade half
changes behaviour for a caller shape ADR-044 never touched, after Grant approved
a narrower scope. `refiner_steps`/`refiner_cfg` are recorded here so the
deferral is a decision rather than an omission.

**Trigger:** the next `mcp_server.py` slice touching the cascade branch. Fix
shape: hoist both rejection loops above the cascade/non-cascade split rather
than adding a third copy, and add the two refiner numerics to both the removed
tuple and the extract pop list at the same time.

## 2026-08-01 — two daemons per device can silently evict each other's socket

**What:** `comfyless --serve` binds one socket per device
(`/run/user/1000/comfyless-cuda<N>.sock`) with no ownership check in either
direction, so a second daemon on the same device silently takes over:

1. **Startup steals.** `server.py` does `if sock_path.exists(): sock_path.unlink()`
   before `bind()`. A second `--serve` on the same device unlinks the first
   daemon's socket and binds its own. The first process keeps running, holding a
   deleted inode and its VRAM, and never receives another connection. Nothing is
   logged on either side.
2. **Shutdown steals.** The server-loop `finally` does
   `if sock_path.exists(): sock_path.unlink()` — by PATH, without checking the
   inode is still the one it bound. So the ORPHANED daemon, on exit, deletes the
   LIVE daemon's socket.
3. **`systemctl stop` is remote-control, not process control.** The unit's
   `ExecStop` is `comfyless.generate --unload --device cuda:%i`, which sends an
   `unload` request over the shared socket — to whoever owns it. `unload` returns
   `False`, stopping that server's loop (and triggering (2)). So stopping the
   systemd unit can shut down an unrelated manual daemon and delete its socket,
   while the systemd process itself is merely SIGTERM'd afterwards.

Hit live 2026-08-01 during the ADR-044 Part C smoke: `comfyless@1.service` was
running when a manual `--serve` was started on cuda:1 for its `--ref-root`. Both
processes ran; the manual one owned the socket; the systemd one was a VRAM-holding
ghost. Confirmed by socket mtime and `pgrep`.

**Why not now:** the fix is a genuine design question, not a patch — options are
an ownership check (bind to a temp path + atomic rename, verify inode before
unlink), an abstract-namespace socket, `SO_REUSEADDR`-style refusal ("a daemon is
already serving cuda:1 — stop it first"), or systemd socket activation so the
unit owns the socket and manual runs cannot bind at all. Refusing to start beats
stealing, but it changes the restart ergonomics the systemd unit depends on. That
deserves its own slice, and none of it is on the ADR-044 path.

**Trigger:** the next `--serve` / systemd-unit change, or the next time a run
inexplicably ignores a daemon flag (that symptom is almost always this). Interim
workaround: exactly one daemon per device — `systemctl --user stop comfyless@<N>`
BEFORE starting a manual one, and `pgrep -af "comfyless.generate --serve"` to
confirm.

## 2026-08-13 — single-entrypoint LLM proxy: token plumbing + `model` pins

**What:** an ai-stack-wide proxy/router is being wired to sit in front of every
LLM endpoint. When it lands, this repo needs a config sweep, one code decision,
and one deferred security item promoted. Inventory taken 2026-08-13 (gitnexus
index fresh at `85cc02b`); recorded now so the surface doesn't have to be
re-derived when the proxy exists.

**The contract is already written and accepted** (read both before doing any of
this): `AI_Stack/Decisions/ADR-008-LLM-Egress-Gateway.md` (accepted 2026-08-08)
and the negotiation channel `AI_Lab/contracts/llm_egress_gateway_bridge.md`.
Shape: nginx auth proxy on **:8100** (the only exposed port, one static bearer
via `Authorization: Bearer`, path-independent auth) → `vllm_router` on
127.0.0.1:8101 → vLLM backends on **127.0.0.1:8001-8036**. Consumers set
`base_url` to `http://<host>:8100/v1`. The allowlist is closed and exact: POST
`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, GET `/v1/models` —
nothing else. Both paths this repo uses are on it.

**Correction to the premise this entry was opened with — the ports do NOT
break.** D9 keeps the backends on 8001-8036 and only rebinds them from
`0.0.0.0` to `127.0.0.1`. comfyless is a **host process**, so loopback stays
reachable and *every current `enhancers.toml` URL keeps working through
cutover*. ADR-008's cutover analysis (D1) enumerates the consumers that break as
the **containerized** ones — OpenWebUI, SillyTavern, ComfyUI nodes — because a
container cannot reach a host loopback bind. Consequences worth being explicit
about: (a) nothing forces a migration on ai-stack's cutover schedule, so we
sequence it ourselves; (b) until we migrate we are a host-local path around the
gateway that the token does not govern — not a LAN exposure and not a new one
(I-GW11 accepts exactly this residual); (c) an elective migration that never
happens is how a port dictionary survives the project that was supposed to
delete it.

**But the host-process path is transitional, not durable** (Grant, 2026-08-13):
direct host→backend reachability is expected to be withdrawn, leaving the proxy
as the only ingress. **So this migration is elective in TIMING and mandatory in
OUTCOME** — plan it as work that will be forced, not work that might never be
needed. ADR-008's 2026-08-10 changelog already names the mechanism: host-local
isolation is "a packet-filter question, not a topology one," with a uid-scoped
`nftables` `owner`-match rule as the right instrument, scoped to local_agents
rather than closed in ADR-008 (which is why I-GW11 stays open). Two consequences:

- **That rule, as described, denies comfyless.** Its stated virtue is that it
  "can permit one broker while denying the rest of the box" — and comfyless *is*
  the rest of the box, same operator and same uid (`gawkahn`) as everything else
  here. Correct security outcome, but it means either comfyless has migrated to
  :8100 by then or it needs an explicit allowance.
- **It fails in the worst available shape.** A filtered connection and a dead
  backend are indistinguishable at our error boundary — both arrive as a connect
  failure and render as the same "cannot reach endpoint" string. The day
  host-local egress closes, comfyless reports every model as down, which reads
  as an ai-stack outage rather than a policy change. Notice turns a confusing
  incident into a one-line config edit; notice has been requested (item 3 of the
  bridge doc below).

**Bridge channel opened** (2026-08-13):
`AI_Lab/contracts/llm_gateway_bridge_image_gen.md` — ai-stack ↔ Image_gen,
separate from the CLOSED ai-stack ↔ local_agents egress bridge. Carries the
ComfyUI nil-exposure answer, the liveness request, and the notice request.
Both sibling sessions pinged directly (`ai-stack-aa`, `scheduler-bb`).

**ADR-008's one un-enumerable consumer is ours, and the answer is NIL.** The ADR
says at D1: "confirm whether any ComfyUI workflow or custom node calls a host
LLM endpoint. That is runtime workflow state, not visible in the configs, and it
is the one consumer this ADR cannot enumerate from the repo." Swept 2026-08-13:
this pack IS installed in the bridge-networked containers
(`ai-stack-data/{comfyui,comfy1}/custom_nodes/comfyui-eric-qwen-edit`, plus
`comfy-dev/basedir`), and it does ship three nodes that CAN call a host LLM —
`eric_qwen_prompt_rewriter.py` and the inpaint/controlnet variants importing its
`_resolve_api_key`, via an operator-typed `api_url` widget. But **no saved
workflow uses them, and no saved workflow under any ComfyUI tree contains an
OpenAI-style `http://host:port/v1` URL at all**; the repo's own
`workflows/*.{json,png}` artifacts are likewise clean. Their defaults
(`localhost:1234` LM Studio, `localhost:11434` Ollama) resolve *inside* the
container, so they could never have reached a host backend without someone
typing a host-reachable address, and nobody has. Zero exposure at cutover from
this repo. Re-run the sweep if that changes:
`grep -rlio "http://[a-z0-9._-]*:[0-9]\{2,5\}/v1" <comfy trees>/run/ComfyUI/user`.

**Already token-ready — config only, no code.** The `key_env` convention
(config holds the env-var NAME, the value comes from the environment, never the
command line) is implemented by all three LLM consumers, and all three read the
SAME registry via `enhance.load_backends`, so one `key_env` line per
`enhancers.toml` entry serves every caller:

- `comfyless/enhance.py` — `_post_chat` + `_resolve_endpoint_model`, Bearer on
  both `/chat/completions` and `/models`
- `comfyless/refine.py` — `_backend_key` / `_post_judge` (judge + ADR-039 duel);
  `key_env` with a literal `key` fallback
- `comfyless/catalog_enrich_concepts.py` — reuses `enhance._post_chat` + `key_env`
  (this is the LoRA-catalog LLM plane; it needs nothing new)

**Confirmed OUT of scope — verified, not assumed:**

- `comfyless/server.py` (daemon) makes zero LLM calls. Enhancement runs
  CLIENT-side in `generate.py` (~:4639) before dispatch, so the token lives in
  the CLI process env; rotating it needs no daemon restart.
- `comfyless/mcp_server.py` has no outbound LLM call — its tools are `generate`,
  `list_models`, `list_loras`, `list_transformers`, `extract_params`, `search`.
  It is called FROM a model, not calling one.
- `comfyless/video.py:799` spawns workers with `subprocess.Popen(cmd)` and no
  `env=`, so `--devices` workers inherit the token for free.
- `comfyless/catalog_enrich.py` is civitai, not an LLM — its own auth axis.

**Needs a code decision — the node pack.**
`nodes/eric_qwen_prompt_rewriter.py` (and the inpaint / controlnet rewriters that
import its `_resolve_api_key`) use an older, separate mechanism: `_ENV_KEY_MAP`
maps URL FRAGMENTS (`deepseek`/`openai`/`anthropic`) to env names, then falls
back to generic `ERIC_QWEN_API_KEY`, then `api_keys.ini`. Under a proxy the URL
matches no fragment and lands on the generic var — which works, but only by
accident. Decide: add a proxy fragment, or make the generic var the documented
path for proxied endpoints.

**The one that fails SILENTLY — `ids[0]` model selection.** Every
`enhancers.toml` entry today omits `model` and relies on
`_resolve_endpoint_model` doing `GET /v1/models` and taking `ids[0]`. That is
only safe because it is one server per port. A proxy fronting many models breaks
the assumption without erroring — you would enhance/judge with whatever model
sorts first. **This is the highest-risk item here precisely because nothing
raises.**

ADR-008 **D6/I-GW5 makes this strictly worse than "ambiguous"**: `GET
/v1/models` is served by the proxy from the **full static inventory** and
"enumerates every model the gateway is configured to route to, under the exact
name consumers must send, **regardless of whether that model is currently
running**." That is deliberate — it exists so OpenWebUI's model picker isn't
empty on a cold box — and D6 accepts the consequence in terms: "a pick can 503."
So `ids[0]` would not merely pick an arbitrary model, it would routinely pick a
**down** one and 503 on the first chat call. A listing hit is explicitly not
proof of readiness.

What to do INSTEAD of `ids[0]` is deliberately left open (Grant, 2026-08-13):
the proxy is getting its own semantics for naming a model when several are
served, so the replacement is a consumption decision that follows the proxy's
contract. An explicit per-entry `model = "..."` pin is the obvious candidate —
do not treat it as settled. What IS settled: `ids[0]` cannot survive, and
whatever replaces it lands in the same slice as the URL rewrite.

**The D7 availability contract is specified against an SDK we do not use.**
This is the largest code finding here and it is ours alone — the bridge
negotiated D7 with local_agents, whose consumer is the `openai` Python SDK, and
every mechanism in it is an SDK behaviour: `x-should-retry: false` short-circuits
`_should_retry`, `Retry-After` is honoured verbatim by `_calculate_retry_timeout`
for `0 < x ≤ 60`, and `x-model-warming` / `x-model-ready-eta-seconds` are read
off `.response.headers` (I-GW12 chose headers over body precisely because
headers survive all three of the SDK's error-construction branches). **comfyless
uses raw `urllib`**, which auto-retries nothing, honours `Retry-After` not at
all, and — in our handlers — *discards the headers entirely*: both
`enhance._post_chat` and `refine._post_judge` catch `urllib.error.HTTPError` and
keep only `e.code` plus 300 bytes of body. So today a warming model and a dead
model are the same `EnhanceError`/`RefineError` string, and the ETA the gateway
went out of its way to make readable is dropped on the floor. Consuming D7 means
reading `e.headers` explicitly at both sites. Note the ADR states the three
states as: not-running-and-nothing-will-wake-it → 503 + `x-should-retry: false`;
warming → 503 + `Retry-After` + the two `x-model-*` headers; proxy/router broken
→ 502. Genuine upstream 503s pass through untouched (I-GW7), and auth failures
are 401/403 and never 503 (I-GW6), so status alone cleanly separates "refused"
from "unavailable."

**Cold start cannot be absorbed, by anyone.** ADR-008 measured two 235B launches
at **1,239 s and 1,137 s (19-21 min)** and states plainly that no combination of
`Retry-After` (capped at 60 s) and SDK retries can hide that — the design
surfaces the condition instead. Our hardcoded timeouts are all far below it and
have no config surface: `enhance.py:452` `timeout=10` on the `/v1/models` GET,
`enhance.py:479` `timeout=120` chat, `refine.py:77` `JUDGE_HTTP_TIMEOUT = 120`
(judge + duel; a parameter, but never sourced from the registry). The concepts
enricher inherits enhance's values. These should become backend-cfg-readable in
the same slice. Sharpest consequence in refine: an endpoint failure charges the
F7 iteration counter, so a model that is merely *warming* could burn a run's
iteration budget while reporting a misleading outcome.

**Liveness gap — now confirmed against the built artifact.** `models.json`'s
entries carry exactly `id`, `object`, `created`, `owned_by`. **There is no state
or availability field**, so the gap described below is a property of the shipped
inventory, not an inference from the ADR.

A second argument arrived from the ai-stack model-bringup session (2026-08-13)
and it is stronger than the scheduling one: **models are mutually exclusive by
GPU capacity, permanently.** `deepseek-v4-flash-0731` (:8035) and its abliterated
sibling (:8036) each fill both cards (~187 GiB), so one running means the other
structurally cannot — and that generalises to any two two-card models, forever,
independent of any scheduler. So "the model you want is down because a different
one is up" is a standing property of a fleet with more models than cards, not
merely a warming state. Consequence for us: probing N models cannot distinguish
"not started" from "cannot start while its neighbour holds the cards," so no
amount of client-side cleverness reconstructs the signal. That session also
noted our own connect failures against :8036 earlier that day were its model
swaps — i.e. the "indistinguishable from an outage" failure mode below already
happened to us once, from a cause that was neither filtering nor a fault.

**SMOKED END-TO-END THROUGH THE LIVE GATEWAY, 2026-08-13.** A `[gateway-deepseek]`
entry was added to `enhancers.toml` (deliberately separate from the raw-backend
`[deepseek-v4-flash]`, which is untouched and still works) with the three changes
a gateway entry needs: `url = http://localhost:8100/v1`, an explicit
`model = "deepseek-v4-flash-0731-abliterated-nvfp4"`, and
`key_env = "COMFYLESS_LLM_TOKEN"`. Results:

- **The happy path works with no code change.** `python -m comfyless.enhance`
  produced correct enhanced prompts through the authenticated gateway. The
  `key_env` convention carries the bearer end-to-end exactly as designed —
  confirming the "already token-ready" claim above against a real enforcing
  proxy rather than against the source.
- **Auth boundary verified both directions:** `/v1/models` answers 401
  unauthenticated and 200 with the bearer.
- **Missing-token failure is BETTER than predicted.** The note further up warned
  the empty-key path would read as a bad token rather than a missing one. The
  gateway's own 401 body says *"Invalid or missing bearer token"*, so the
  ambiguity is named by the server. Our silent-empty behaviour is still worth
  fixing (a client-side "key_env NAME is set but the variable is empty" warning
  costs nothing) but it is not the trap it was described as.
- **The D7 down-model contract is live and correct.** Requesting
  `gemma4-31b-heretic` (in the inventory, not running) returns
  **503** + `x-should-retry: false` + `x-gateway-rendered: 1`, with
  `error.code = "model_not_running"` and `error.type = "server_error"` in the
  body. No `Retry-After` / `x-model-warming` — correct, since the warming branch
  is inert with no scheduler.
- **`error.code == "model_not_running"` is the discriminator to consume.** It is
  machine-readable and beats both prose-matching and header-sniffing for
  separating "this model is down" from every other 5xx.
- **Our header-blindness is confirmed, exactly as documented.** comfyless
  surfaces that 503 as `EnhanceError: openai-endpoint HTTP 503 from ... {body}`
  with the body truncated at 300 chars and **every header discarded**, so
  `x-should-retry: false` never reaches the operator and the prose is cut
  mid-sentence before the part telling them how to launch the model.

**Spec-vs-implementation divergence worth knowing (reported upstream):** ADR-008
D7's table says the not-running 503 body "names the model, says how to launch
it." **It does not name the model** — the rendered body is static and generic
("The requested model is not serving on this host..."). That is not sloppiness,
it is forced: the model name lives in the POST body, and D1 deliberately keeps
the proxy out of body parsing ("doing that in the proxy would mean parsing the
most attacker-controlled part of the request inside the component that holds the
auth decision"). So naming the model in the proxy-rendered 503 is unimplementable
without giving up the property D1 exists to protect. The implementation chose
correctly; the ADR text is what should move. **Consequence for us:** never expect
the error body to identify the failed model — we already know which model we
sent, so any "X isn't up" message must be built from our own request state.

**Explicit bring-up, and the liveness gap it runs into.** Grant's design intent
(2026-08-13): waking a model must be an EXPLICIT decision, never an automatic
consequence of asking for one that is down — with an offer of what to do instead
("Qwen 32B isn't up but Gemma 4 is, use that instead?"). ADR-008 leaves room for
this by construction: D7 ends "the gateway reports 'warming' and an ETA; the
scheduler owns 'make it running'," and lifecycle is explicitly out of that ADR.
Two things to work out on our side before the scheduler slice freezes its shape:

1. **We cannot currently offer an alternative, because nothing tells us what is
   up.** D6 deliberately removed liveness from `/v1/models` (full static
   inventory, running or not), so the only liveness signal in the whole contract
   is a 503 on an actual call. Probing every candidate to build an "is up" list
   is N requests to find one answer. If a cheap liveness signal is wanted — a
   field on the `/v1/models` entries, or a separate endpoint — that is a
   **request to raise on the bridge doc**, and it is much cheaper to raise
   before the scheduler slice than after. Note the allowlist is closed and
   exact, so a new endpoint is an ADR-008 change, not a client-side choice.
2. **Most of our LLM calls have no human attached.** An interactive
   "use Gemma instead?" prompt only makes sense on the foreground CLI; refine
   loops, the MCP server, video workers and `--iterate` batches are all
   non-interactive and need a *policy* instead (fail fast naming the model and
   what the operator should launch — which is exactly what the D7 not-running
   body carries). The repo already has the right precedent for the split:
   `comfyless/pause.py`'s `sigint_pause` no-op guards (no TTY / not main thread
   / detached stdin) are the shape to reuse rather than reinvent, and the same
   guards decide correctly here.

**Config surface to rewrite:** `enhancers.toml` (9 entries, localhost:8016-8022
plus the 2026-08-13 `deepseek-v4-flash` on :8036), `enhancers.example.toml`
(committed, :8016/:8017), `implementation_details.md:49`, and a stale-reading
fixture at `test_catalog_enrich_concepts.py:227`. `catalog_cli.py:78`'s
`--backend gemma-moe-nvfp4` default is a NAME, so it survives the port change
unless the proxy renames models.

**Security items the token reactivates:**

- **A11 promoted.** `implementation_details.md:139` defers openai-endpoint
  redirect hardening as LOW: urllib follows redirects and forwards
  `Authorization: Bearer` cross-host. Its trigger reads "if a non-localhost
  endpoint is ever configured" — that is TOO NARROW. A *localhost* proxy that
  302s leaks the token just as well. The real trigger is "once a real token is
  in play." Re-scope A11 and fix it in the token slice.
- `enhance._post_chat` echoes 300 bytes of the HTTPError body into the raised
  exception; a proxy 401 could reflect a token into a log. Low, but look.
- The token must never land in the committed `enhancers.example.toml`
  (`enhancers.toml` itself is gitignored).

**Process cost — not a config sweep.** `comfyless/refine.py` is a Red Zone path
in `scripts/git-policy/_red-zone-paths.sh`. Any token wiring that touches it
needs `security-auditor` plus a saved `docs/security/review-*.md` and an ADR
reference, or pre-commit rejects the commit. Budget for a spec-first slice.

**Why not now — REVISED 2026-08-13 (later the same day): the gateway is already
standing, so "untestable" no longer applies.** Measured directly:

    curl -o /dev/null -w '%{http_code}' http://localhost:8100/v1/models   # 401 (auth live)
    ss -ltn  →  127.0.0.1:8100, 172.17.0.1:8100, 192.168.1.153:8100   (proxy)
                127.0.0.1:8101                                        (router, correctly loopback)
                0.0.0.0:8036                                          (backends NOT yet migrated)

So D1/D2/D3/D9 are implemented and enforcing; what remains on ai-stack's side is
**slice 4, the backend loopback migration**, which is the step that does not
affect us (we are a host process). The only thing gating a test on our side is
the bearer, which is an operator secret under `${STATE_DIR}/secrets/` — ask
Grant for it rather than going looking.

**The model-naming question is effectively answered by implementation.** The
proxy's inventory is `infra/gateway/models.json`, generated from the ai-stack
README registry by `tools/scripts/gen-gateway-inventory.sh` (which also emits
the router's backend table, so the two cannot drift — never hand-edit either).
It currently holds **29 entries whose `id`s are exactly the README names**,
including `deepseek-v4-flash-0731-abliterated-nvfp4` and the four
`gemma4-*-heretic*` / three `qwen3-vl-*` ids we consume. So a per-entry
`model = "<registry name>"` pin is no longer a guess — it is reading the served
inventory. Confirm against the live `/v1/models` at migration time, since the
generator can add entries.

**And `ids[0]` is now concretely, not theoretically, wrong:** the first entry in
that inventory is `gpt-oss-120b`. Every backend in `enhancers.toml` that omits
`model` would silently resolve to it.

**Trigger — two of them, and the second is a deadline we do not control:**

1. **The gateway going live** is the earliest we *can* migrate. Not forced by
   ai-stack's cutover (our URLs survive it), so this one is a decision.
2. **Host-local egress restriction is when we *must* have migrated** — the
   uid-scoped nftables work above. Watch the bridge doc for notice; if it lands
   without warning the symptom is "every model is down."

Whenever it happens, the `ids[0]` replacement, `e.headers` consumption for D7,
the configurable timeouts, and the A11 redirect fix all belong in the SAME slice
as the URL rewrite. A gateway URL with the old `ids[0]` resolution and
header-blind error handling is the combination that fails quietly and looks like
a working migration.
