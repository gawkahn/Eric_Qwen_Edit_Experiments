# ADR-030 — Comfyless 2× upscale-VAE decode (spacepxl Wan2.1-VAE-upscale2x)

Status:   accepted

## Context

**Primary motivation: speed.** Generating at 2048² is ~4× the compute of 1024².
spacepxl's `Wan2.1-VAE-upscale2x` is a decoder-only finetune that emits 12
channels (= 3 × 2 × 2); `pixel_shuffle(2)` turns its output into a 2×-resolution
image in a single VAE pass. So a **stable ~1 MP latent decoded up to 2×** yields
near-2048 output at roughly quarter the generation cost — a large, LoRA-agnostic
win. Already implemented for the ComfyUI node path in
`nodes/eric_qwen_upscale_vae.py::decode_latents_with_upscale_vae_safe` and proven
a quality win; this ADR wires that same decode into the **comfyless** generate
path. The Wan2.1 and Qwen-Image VAEs share the same latent space (identical
`latents_mean`/`latents_std`), so no cross-space conversion is needed.

**How this surfaced (and a correction to the record).** The feature came out of
a 2026-07-14 investigation into an edge-corruption band on 2048² Krea-2-Raw gens
(`krea-sampler-test/`). Three candidate causes were tested:

- **Sampler/schedule — ruled out.** The band appears on `default` too.
- **VAE *normalization* — ruled out.** The Krea-2-Raw VAE config and the Wan2.1
  VAE config carry **byte-identical** `latents_mean`/`latents_std` across all 16
  channels (the Krea VAE is packaged Wan-latent-compatible by design), so the
  pipeline denormalizes identically regardless of which VAE is in `pipe.vae`. A
  "stamp base stats onto the override" fix would have been a no-op; abandoned
  before code.
- **High-res boundary instability — *falsified*.** Initially the leading
  hypothesis, but **2048² with no LoRAs is clean on both VAEs.** The actual cause
  is an **over-strength LoRA** (`lenovo_krea2` at weight 1.2): it injects latent
  instability that Raw's 52-step denoise amplifies (Turbo's ~8 steps do not — the
  turbo/raw distinction) and the `AutoencoderKLWan` decoder worsens at spatial
  boundaries. Reducing the weight to ~0.7 nearly clears it. So high resolution is
  *not* inherently unstable, and that is **not** the justification for this
  feature — speed is.

The upscale path is therefore a performance/ergonomics feature, not a bug fix.
The edge-band cause (LoRA strength) is a usage matter, tracked separately.

Model is already local: `hf-local/Wan2.1-VAE-upscale2x/diffusers/`
`Wan2.1_VAE_upscale2x_imageonly_real_v1` (`AutoencoderKLWan`, `out_channels: 12`,
`z_dim: 16`). No new download.

Lens (§1): solo-defensible AND team-portable — this is a general super-res
decode path, not a one-off.

## Decision

Add an opt-in `--upscale-vae PATH` flag to comfyless generate. When set:

1. Run the diffusion pipe with `output_type="latent"` (instead of the default
   decode-to-PIL at `generate.py:1577`).
2. Decode the packed latents with the existing node helper
   `decode_latents_with_upscale_vae_safe(latents, upscale_vae, pipe, height,
   width, ...)` — it already handles device pinning, transformer offload +
   guaranteed restore, and auto-tiling for large latents.
3. `pixel_shuffle(2)` inside that helper yields a **2× image**; convert its
   `[B, 2H, 2W, 3]` float tensor to the PIL that the rest of the save path
   expects.

**No decode math is reimplemented** — the helper is reused as-is. The pipe's
own VAE (`pipe.vae`, native or `--vae` override) is retained solely for its
`latents_mean`/`latents_std` config, which the helper reads.

**CLI resolution semantics (the load-bearing choice):** `--width/--height`
specify the **generation (latent) resolution**; the saved PNG is **2× each
dimension**. So `--width 1024 --height 1024 --upscale-vae <path>` → a clean
1024² latent → a 2048² PNG. This is exactly what sidesteps the high-res boundary
instability. The 2× factor and the "you asked for gen res, you got 2× output"
contract are logged and recorded in the sidecar.

**Model-family gating:** the upscale VAE consumes Qwen-layout packed latents in
the shared Qwen/Wan latent space. Allow it for `krea`, `krea-turbo`, and
`qwen-image`; **hard-error with a clear message** for `flux`/`flux2`/other
incompatible latent spaces rather than silently producing garbage. The raw
`wan` video pipeline is intentionally excluded — it does not emit Qwen-layout
packed `[B, seq, C*4]` latents and its family string doesn't resolve through
`_FAMILY_PATTERNS` (code-review).

**Metadata:** sidecar records `upscale_vae` (path), `upscale_vae_subfolder`, and
`upscale_factor: 2`, plus the note that `width`/`height` are pre-upscale gen res.

**Execution-path scope — slice 1 is in-process CLI only** (`generate.py` direct
path). The `--serve` daemon and MCP server cache one pipeline; they do not yet
cache/offload a *second* (upscale) VAE. Daemon + MCP support is a follow-on
slice (see Deferred).

## Alternatives Rejected

- **Normalization "fix" on the plain `--vae` Wan override.** No-op: stats are
  byte-identical (see Context). This was the original scope-2 direction and was
  killed by the numeric check before any code.
- **Generate at 2048 natively, then post-process the edge band** (crop, mask,
  denoise). Treats the symptom; the unstable high-res latent is the disease.
- **Post-hoc pixel upscaler (ESRGAN / Lanczos) on a 1024 decode.** Lower quality
  than a latent-native decode-time super-res; the whole point of the 12-channel
  decoder is that the SR is learned in latent space.
- **`upscale_between_stages` (decode→re-encode) as the final step.** That helper
  exists for *inter-stage* multistage upscaling; for a final image we want the
  direct decode (`decode_latents_with_upscale_vae_safe`), no re-encode round trip.

## Deferred / Out of Scope

- **Daemon (`--serve`) + MCP integration.** Needs upscale-VAE load/cache/offload
  in `server.py` and `mcp_server.py`. Follow-on ADR-030 slice 2.
- **Comfyless multistage / UltraGen inter-stage upscale** via
  `upscale_between_stages`. Separate.
- **Non-2× ratios / arbitrary target resolution.** The model is fixed 2×.
- **The edge-corruption band itself** is a **usage issue, not a code bug**:
  over-strength `lenovo_krea2` LoRA (weight 1.2), amplified by high step counts
  and the Wan decoder. Mitigation is LoRA weight (~0.7), not this feature.
  Orthogonal to the 2× decode; not addressed here.
- **`--json` / batch / iterate interaction** beyond passing the flag through.

## Open Questions

1. ~~**Resolution contract**~~ — **RESOLVED 2026-07-14: gen-res-in, 2×-out.**
   `--width 1024` generates a 1024² latent and saves a 2048² PNG.
2. ~~**Slice-1 path boundary**~~ — **RESOLVED 2026-07-14: slice 1 includes the
   `--serve` daemon** (the user's primary path). MCP (`mcp_server.py`) remains a
   small follow-on. Daemon work: forward `--upscale-vae` over the wire protocol,
   load + cache + offload the upscale VAE in the server alongside the pipeline;
   the shared decode lives in `generate()` so CLI and daemon use one code path.

## Changelog

- 2026-07-14 — Proposed. Authored after the 2026-07-14 Krea sampler/VAE
  investigation. Reference impl already present at
  `nodes/eric_qwen_upscale_vae.py`; model already local.
- 2026-07-14 — Corrected Context: the motivating edge band was traced to an
  over-strength LoRA (`lenovo_krea2` 1.2), **not** high-res boundary instability
  (2048² no-LoRA is clean). Feature motivation reframed to **speed** (1024 gen +
  2× decode ≈ 2048 quality at ~¼ cost), which is LoRA-agnostic. Resolution
  contract resolved (gen-res-in, 2×-out).
- 2026-07-14 — **Accepted.** Slice 1 includes the `--serve` daemon; MCP deferred.
  §12 note: this slice touches `comfyless/server.py` and loads a caller-supplied
  model path via `resolve_hf_path`, so it requires `security-auditor` +
  `code-reviewer` before commit (review saved under `docs/security/`).
- 2026-07-14 — **Implemented + reviewed.** security-auditor: 2 findings, both
  fixed in-slice — [HIGH] `upscale_vae_subfolder` traversal escaping
  `_check_paths` (now realpath-confined to the resolved dir before load);
  [MEDIUM] MCP redaction map missing the upscale path (added). Full review:
  `docs/security/review-adr030-upscale-vae-2026-07-14.md`. code-reviewer:
  canonical metadata key (`upscale_vae_path`, not `upscale_vae`, so `--params`
  replay works; `upscale_factor` excluded via `_SKIP_SIDECAR_KEYS`); upscale
  cache moved above the output reservation (a LoadError no longer orphans a
  reserved PNG); `upscale_vae_path` added to `_run_one`'s resolve loop (repo-id
  parity CLI↔daemon); `wan` dropped from the family gate. Daemon cache-lifecycle
  test added. All 18 unit suites green. Transformer-offload latency logged to
  TECH_DEBT (2026-07-14) as a follow-on.

AI-Disclosure: Claude (Fable) authored; Grant reviewed.
