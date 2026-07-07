# Security review — ComfyUI-native Krea-2 key converter (single-file load)

**AI-Disclosure:** security-auditor (Claude Opus 4.8, 1M context) authored the
review below; Grant reviewed. 2026-07-07.

**Surface (§12 Review bar):** custom parsing of caller-supplied weight-file
CONTENT on the `comfyless --transformer` single-file load path — same surface
as the ADR-019 slice-C scaled-fp8 parser
(`review-slice-C-fp8-single-file-2026-07-02.md`). Extends it with a new
key-name → key-name converter.

**Files reviewed:**
- `nodes/eric_krea2_convert.py` (new — markers, `is_krea2_comfy_checkpoint`,
  `convert_krea2_comfy_key`, `_convert_block_internal`,
  `convert_krea2_comfy_state_dict`, `reshape_to_model_shapes`)
- `nodes/eric_diffusion_fp8_ops.py` — the Krea branch in
  `load_scaled_fp8_component` (convert → `from_config` → numel-safe reshape →
  `load_state_dict(assign=True)` → missing/unexpected-key raise → fp32-norm
  restore)
- `nodes/eric_diffusion_utils.py:643-667` — the single production caller and
  the classify-gate; upstream control-char guard at `eric_diffusion_fp8_ops.py`
  runs before the branch.

## Verdict: CLEAN

No HIGH/MEDIUM/LOW findings. Two INFO defense-in-depth notes (below). The
change introduces **no primitive beyond the pre-existing "load a checkpoint
you chose" baseline**: the loaded object is used for image generation only —
no code execution, auth, or privileged sink. All failure paths on the new
branch are fail-closed.

## Threat model

Untrusted `.safetensors` checkpoint author → CLI `--transformer` single-file
load → parsed KEY NAMES remapped to diffusers names → model built via
`from_config` + `load_state_dict`. Sink is image-gen-only. Relevant question:
does the new code add a primitive worse than "a chosen weights file yields a
wrong/garbage image or a crash"? Conclusion: no.

## Per-vector confirmations

1. **Key-name injection — BLOCKED.** Every emitted target key is
   `<fixed-prefix>.<\d+-index>.<suffix>` where the prefix set is a closed
   literal list (`transformer_blocks.`, `text_fusion.`, `img_in.`,
   `final_layer.`, `time_embed.`, `txt_in.`, `time_mod_proj.`) and pass-through
   suffixes stay inside the matched namespace. No `.format()`, format-spec, or
   regex-group-to-target path; unmatched keys pass through unchanged and become
   `unexpected_keys` (now raised on). Worst case is "a param value you could
   have supplied pre-shaped anyway."
2. **Reshape loop — SAFE.** Allowlisted to `*scale_shift_table` targets that
   already exist as real model params, exact-numel match only, target shape
   drawn from the built model (not attacker), `reshape` (copy-safe) not `view`.
   numel-collision only reinterprets a flat tensor into a param the author
   could have supplied pre-shaped. No new OOM (tensor already materialized;
   architecture-bounded).
3. **Parse safety — PURE.** `eric_krea2_convert.py` is `re` + string/dict ops
   only: no file I/O, path handling, `eval`, `pickle`, or `torch.load`.
   `is_krea2_comfy_checkpoint` reads key strings only, never tensor values.
   Upstream load is `safetensors.load_file` / `weights_only=True`.
4. **`assign=True` + `strict=False` — FAIL-CLOSED.** Dropped unexpected keys
   are ignored input; the code now raises on BOTH `missing_keys` and
   `unexpected_keys`; PyTorch's own size-mismatch check catches wrong-numel
   tensors regardless of `strict`. No partial random-init model can load.
5. **MCP boundary — NOT CROSSED.** CLI `--transformer` path only; no MCP
   name-resolution/response code touched. Error strings pass through
   `_safe_name` (repr + 200-char cap).
6. **vs. slice-C contract — EQUIVALENT.** Control-char rejection is enforced
   both upstream (loader, before the branch) and now self-guarded inside
   `convert_krea2_comfy_state_dict` (INFO-1 folded). No value materialization
   before validation.

## INFO notes (defense-in-depth)

- **INFO-1 (FOLDED 2026-07-07):** the converter now rejects control-char key
  names itself (without echoing the raw key), so the module is safe even if a
  future caller imports it directly rather than via the slice-C-guarded loader.
- **INFO-2 (accepted):** the missing/unexpected-key raise + PyTorch's internal
  size check are the completeness guards; adequate for an image-gen-only sink.

## Code-review companion

`code-reviewer` (Opus) returned CHANGES REQUIRED; findings folded before
commit — see ADR-019 Changelog 2026-07-07 (fp32-norm restoration, reshape
allowlist, unexpected-key raise, added unit tests, and this saved artifact
which closes the review-artifact gap the reviewer flagged as #9).
