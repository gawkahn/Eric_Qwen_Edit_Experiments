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
#   - comfyless/generate.py         — only `_run_json_mode` is the §12 surface;
#     the rest of the file changes in most feature slices.
#   - nodes/eric_diffusion_utils.py — only `resolve_hf_path` is the surface;
#     same reasoning. See TECH_DEBT.md entry "git-policy: function-scoped Red
#     Zone surfaces not path-gateable".

is_red_zone_path() {
    local path="$1"
    if [[ "$path" =~ (^|/)comfyless/server\.py$ ]]; then return 0; fi          # Unix-socket IPC daemon
    if [[ "$path" =~ (^|/)comfyless/mcp_server\.py$ ]]; then return 0; fi      # MCP server (LLM tool surface)
    if [[ "$path" =~ (^|/)comfyless/refine\.py$ ]]; then return 0; fi          # ADR-027 judge/seed surfaces
    if [[ "$path" =~ (^|/)nodes/eric_diffusion_fp8_ops\.py$ ]]; then return 0; fi  # weight-file content parser (ADR-019)
    return 1
}

# Canonical list (one per line) — used by smoke tests / enumeration. Mirror the
# regexes above here in human-readable form.
list_red_zone_paths() {
    cat <<'EOF'
comfyless/server.py
comfyless/mcp_server.py
comfyless/refine.py
nodes/eric_diffusion_fp8_ops.py
EOF
}
