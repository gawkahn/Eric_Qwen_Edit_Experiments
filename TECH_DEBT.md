# Tech Debt

Items here are conscious deferrals — known gaps with a recorded reason for not fixing now.
Format: **Item** — why deferred, what triggers revisiting.

---

## Security

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

## 2026-07-10 — `--quant fp8` produces garbage output on Z-Image-base (NOT Turbo)

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
- **What remains:** base + fp8 is broken at every schedule; Turbo + fp8 is fine at every
  schedule; base without fp8 is fine. The difference is the WEIGHTS, or the per-component
  quant split. Not yet separated: `resolve_quant_components` quantizes **transformer AND
  text_encoder** (Z-Image's TE is a `Qwen3Model` → "large LM" role; VAE is never quantized
  by invariant). Degraded conditioning from a quantized TE is an untested cause of
  noise-like texture. One asymmetry noted, significance unknown: Turbo's transformer ships
  **F32** on disk, base's ships **BF16** (TE and VAE are BF16 in both).
- **Next, before any code change** (two runs, existing flags, no code):
  `--quant fp8 --quant-only transformer` and `--quant fp8 --quant-skip transformer` on
  Z-Image-base. Localizes the fault to the transformer or the text encoder.
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
