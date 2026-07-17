# ADR-032 — Static type checking: pyright, ratchet posture (types gate)

Status:   accepted
Date:     2026-07-16

## Context

Global CLAUDE.md §15 requires every code repo to have a pinned static type
checker gated in CI; this repo has none (a standing gap — the pyright
diagnostics seen during dev are the Claude Code harness's LSP, unpinned and
ungated). The quality-gate kit's types gate (reference: local_agents ADR-008)
supplies the shape: pin the checker, measure a baseline, wire a recipe, gate
CI. The motivating incident behind §15 — a well-typed-but-wrong `SecretStr`
passed where a `str` was annotated, surviving 685 passing tests — is exactly
the class of defect a type checker catches statically and tests miss.

Measured baseline (2026-07-16, pyright 1.1.411, basic mode, venv-resolved):
**1026 errors** across 72 files — comfyless/ 52, nodes/ 520, pipelines/ 454.
Dominant rules: reportArgumentType 669, reportAttributeAccessIssue 94,
reportOptionalMemberAccess 63. A structural subset (~32 reportMissingImports)
is `comfy.*` / ComfyUI-host imports that only resolve inside a ComfyUI
install — unfixable from this repo without stub packages.

## Decision

1. **Tool: pyright 1.1.411**, pinned via repo `mise.toml` (`npm:pyright` +
   `node`), NOT a uv dep — same choice and reasons as the reference repo's
   ADR-008: it is the checker the harness LSP speaks, and this repo's risk
   profile (§12 surfaces feeding model weights and LLM output into compute)
   wants a *type* checker before a style linter.
2. **Scope: `comfyless/`, `nodes/`, `pipelines/`** (`[tool.pyright]` in
   pyproject.toml; this repo has no `src/`). Test suites stay out per §15 —
   they deliberately violate the contracts they test. `typeCheckingMode:
   basic`, pythonVersion 3.12 (matches `.python-version`).
3. **Posture: RATCHET, not drive-to-zero.** 1026 is a large baseline (§15:
   "a large one ratchets until it's under control"). The committed integer
   lives in `.claude/typecheck-baseline`; a commit may only keep the count
   equal or lower it. Drawdown happens opportunistically (lower the baseline
   in the same commit that fixes errors) or as dedicated slices — comfyless/
   (52) is the natural first drawdown target since it carries the §12
   surfaces.
4. **Enforcement layers:** local PreToolUse(Bash) ratchet hook
   `.claude/hooks/require-typecheck-clean.sh` (kit template; fails OPEN if
   pyright can't run — local convenience, ~11 s per commit) + authoritative
   CI `typecheck` job (fails CLOSED: pyright error count compared to the
   committed baseline).
5. **The `comfy.*` missing-import errors stay in the baseline count.** They
   are real "this repo cannot see the ComfyUI host API" facts; excluding the
   rule would also silence genuinely broken imports in comfyless/.

## Alternatives Rejected

- **mypy / ruff type rules** — no harness-LSP synergy; ruff is a linter, not
  a type checker (ADR-008 reasoning holds unchanged here).
- **Drive-to-zero before gating** — 1026 errors is weeks of drawdown; the
  gate's value is stopping NEW errors today.
- **Excluding `nodes/` to shrink the baseline** — hides half the codebase
  from the gate; the ratchet makes the big number harmless.
- **Stub package for `comfy.*`** — real work for cosmetic benefit; revisit
  if a nodes/ drawdown slice ever happens.

## Deferred / Out of Scope

- Baseline drawdown slices (comfyless/ first when picked up).
- Widening to the test suites (separate decision with its own baseline).
- `strict` mode anywhere.

## Changelog

- 2026-07-16 — accepted; baseline 1026 measured and committed with the gate.
- 2026-07-16 — ratchet hardened after code-reviewer MEDIUM: the local hook
  now reads the baseline from `git show HEAD:` (a same-commit bump cannot
  self-legalize new errors), and the git-policy layer
  (`pc_baseline_no_increase`, pre-commit + CI range check) blocks staged
  baseline increases outright. Deliberate increases use `# user-approved`
  (hook) / `Policy-override:` (git-policy). Count parsing anchored to
  pyright's summary-line shape in both the hook and the CI job.

AI-Disclosure: Claude (Fable 5) authored; Grant reviewed.
