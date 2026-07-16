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

# 0-baseline hard gate (7 findings triaged + suppressed inline 2026-07-16);
# scope is this repo's code roots (no src/). FPs: inline # nosemgrep + comment.
# Security static analysis (SAST) — semgrep, exact-pinned via uv
sast:
    uv run --with semgrep==1.169.0 semgrep scan \
      --config p/python --config p/security-audit --config p/secrets \
      --error --quiet --metrics=off comfyless/ nodes/ pipelines/
