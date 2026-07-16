# Quality-gate recipes (kit adoption 2026-07-16 — secrets + commit-policy only;
# types/tests/sast/supply-chain gates not adopted yet, see the kit README in
# ~/.claude/templates/quality-gate-kit-python-uv/ for the remaining gates).
# Requires: mise (pins in mise.toml), uv.

# --redact keeps secret values out of the output; 0-baseline hard gate.
# Secret scanning — gitleaks (pinned via mise) over all git-tracked history
secrets:
    mise exec -- gitleaks git . --config .gitleaks.toml --no-banner --redact

# Commit-policy smoke tests — the git-policy check functions + evil-merge e2e.
policy-test:
    bash scripts/git-policy/test-git-policy.sh
