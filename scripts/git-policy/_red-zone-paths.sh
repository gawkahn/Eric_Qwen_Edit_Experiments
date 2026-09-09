#!/usr/bin/env bash
# Red Zone path list — Eric_Qwen_Edit_Experiments (quality-gate kit adoption
# 2026-07-16). Single source of truth for which files are "Red Zone" in the
# §12 sense; sourced by scripts/git-policy/_lib.sh. The redzone-spec /
# redzone-review checks fire when a commit touches any path this returns 0 for.
#
# Source of the list: project CLAUDE.md §"Review bar (this project)" + the
# 2026-07-16 kit-adoption handoff. Keep in sync with CLAUDE.md when surfaces
# are added.
#
# Deliberately NOT listed (function-scoped surfaces, path gating too coarse):
#   - src/comfyless/generate.py         — only `_run_json_mode` is the §12 surface;
#     the rest of the file changes in most feature slices.
#   - src/comfyless/core/eric_diffusion_utils.py — TWO function-scoped surfaces
#     live here, and naming only one of them is how a gap hides: `resolve_hf_path`
#     (caller-supplied model loading) AND the fp8/comfy_quant DETECTION + REMAP
#     half of the ADR-019 weight-file surface, whose other half
#     (eric_diffusion_fp8_ops.py) IS gated above. Same coarseness reasoning as
#     generate.py — the file changes in most feature slices. See TECH_DEBT.md
#     entry "git-policy: function-scoped Red Zone surfaces not path-gateable"
#     and its 2026-09-09 amendment.
#     (Path corrected 2026-09-09: this comment named the pre-slice-5
#     `nodes/eric_diffusion_utils.py`, which no longer exists.)

is_red_zone_path() {
    local path="$1"
    if [[ "$path" =~ (^|/)(src/)?comfyless/server\.py$ ]]; then return 0; fi          # Unix-socket IPC daemon
    if [[ "$path" =~ (^|/)(src/)?comfyless/mcp_server\.py$ ]]; then return 0; fi      # MCP server (LLM tool surface)
    if [[ "$path" =~ (^|/)(src/)?comfyless/refine\.py$ ]]; then return 0; fi          # ADR-027 judge/seed surfaces
    # ADR-045 slice 1b moved the weight-file content parser out of nodes/; the
    # gate kept naming the old path and was a silent no-op from that day until
    # 2026-09-09, when a Red Zone edit to the moved file committed without it
    # firing (security review of slice 3d). lora_adapters.py is added at the
    # same time: since ADR-046 every daemon LoRA weight write, backup and
    # registry mutation flows through it, and it was never gated at all.
    # All three historical spellings stay matched, for the same reason slice 5
    # kept `(src/)?`: a `check-range` over a range spanning either move must
    # still content-check the pre-move commits, and a file resurrected at an
    # old path must not slip the gate. Over-matching a path that no longer
    # exists costs nothing and fails closed.
    if [[ "$path" =~ (^|/)((src/)?comfyless/core|nodes)/eric_diffusion_fp8_ops\.py$ ]]; then return 0; fi  # weight-file content parser (ADR-019)
    if [[ "$path" =~ (^|/)(src/)?comfyless/core/lora_adapters\.py$ ]]; then return 0; fi          # LoRA weight writes / backups / registry (ADR-046)
    return 1
}

# The `(src/)?` in the regexes above is deliberate (code-reviewer, ADR-045
# slice 5): the pre-move pattern `(^|/)comfyless/server\.py$` matched BOTH the
# old and the new path by suffix, and rewriting it to `src/` only would have
# silently narrowed the gate — `check-range` over a commit range spanning the
# move would stop content-checking pre-move commits, and a file resurrected at
# the old path (a partial revert of this slice) would bypass the gate entirely.
# Match both spellings until the split makes the old one unreachable.

# Canonical list (one per line) — used by smoke tests / enumeration. Mirror the
# regexes above here in human-readable form.
list_red_zone_paths() {
    cat <<'EOF'
src/comfyless/server.py
src/comfyless/mcp_server.py
src/comfyless/refine.py
src/comfyless/core/eric_diffusion_fp8_ops.py
src/comfyless/core/lora_adapters.py
EOF
}
