# Security review — ADR-045 slice 7 (node-pack cutover)

AI-Disclosure: `security-auditor` (Claude Fable 5, transcript-confirmed, no
fallback) reviewed; Claude Opus 5 authored the slice and this write-up; Grant
reviewed.

Date: 2026-09-09
Trigger: the commit DELETES all five §12 Red Zone files from this repository.
The gate fired on those deletions — which is itself the first live proof of the
`--no-renames` repair made earlier the same day, since with rename detection on
the deletions would have been invisible.

## The finding that mattered

**[HIGH] The Red Zone surfaces moved to a repository with the gate SCRIPTS but
none of the WIRING.** `comfyless_diffusion` had `scripts/git-policy/` and a
`.pre-commit-config.yaml` copied in, and I recorded the policy layer as
adopted. It was not: no `pre-commit install` had been run, there was no dev
dependency group so `uv run pre-commit` could not even execute, there were no
`.claude/` harness hooks, no gitleaks pin, and no CI. A commit touching
`server.py`, `mcp_server.py`, `refine.py`, `eric_diffusion_fp8_ops.py` or
`lora_adapters.py` there faced nothing at all — and its first three commits
went through ungated.

This is the "checked by NEITHER repo" case, and it is worse than either half
suggests: this repo's gate correctly matches only historical paths now, so the
surfaces were genuinely unenforced everywhere.

**Disposition: FIXED** in `comfyless_diffusion` before this commit landed —
dev group with `pre-commit==4.6.0`, both hook stages installed, `.claude/`
settings + hooks copied, gitleaks pinned. Verified by observing the gate run on
that repo's next commit.

## Second finding

**[HIGH] `supply-chain` was disabled wholesale though two of its gates need no
dependency resolution.** `check_lock_sources.py` is pure stdlib over `uv.lock`
and osv-scanner reads the lockfile as a file — only the `uv run` wrapper and
the license step require a synced venv. Skipping the whole job left the tier-3
check — the ONLY automated guard on the local-path exception this slice adds —
enforced nowhere.

**Disposition: FIXED** — split into a new always-on `supply-chain-lock` job
running the tier-3 check under system Python and the osv lockfile scan; the
sync-dependent job stays skipped. Two errors in my first attempt at that job
were caught before commit: an unpinned `jdx/mise-action@v2` (violating §11) and
the wrong osv-scanner invocation.

**This job will be RED on arrival**, and deliberately so — see the CVE section.

## Third finding

**[MEDIUM] Dependency-confusion window.** `comfyless-diffusion` is an
unregistered PyPI name sitting in `[project.dependencies]`, and `pip` ignores
`[tool.uv.sources]`. Today it fails loudly. The `requirements.txt` breakage is
the funnel: a user whose node pack dies with `ModuleNotFoundError: comfyless`
will reasonably try `pip install comfyless-diffusion`, which would execute a
squatter's code inside ComfyUI's venv. **Disposition: RECORDED** — TECH_DEBT
2026-09-09 amended with an explicit prohibition on ever letting a bare name
reach requirements.txt or install docs, plus the cheap mitigation (reserve the
name).

**[MEDIUM] CI skip comments claimed coverage that did not exist** ("the suites
run in comfyless_diffusion's own CI") — that repo has no CI at all.
**Disposition: FIXED**, comments now state that nothing runs there.

## Assessed and accepted

- **The lock-source exception is as narrow as claimed** — name AND kind AND
  path must all match; any drift (a uv path normalization, a rename, a second
  path source) lands in the offender list, i.e. fails closed. Mutation-tested
  on both axes.
- **A local path source vs a git source:** a sha-pinned git source anchors
  content cryptographically; a path source anchors nothing and imports the
  sibling working tree including uncommitted state. But it fetches nothing over
  the network, and on a single-user workstation the sibling is inside the same
  trust domain — anything that can write it can write this repo. The residual
  cost is process, not boundary.
- **Historical Red Zone patterns are the right polarity** — verified on all
  three axes, including that they fired on this very commit.
- **`comfy_stub.cf_path()`** takes only hard-coded literals, fails loudly on a
  `__file__`-less module, and imports only the already-trusted dependency. One
  consequence worth knowing: under an editable install the AST source guards
  inspect the sibling's WORKING TREE, so a green guard here does not attest to
  any committed state of the other repo.
- **Jobs still providing real coverage:** `git-policy`, `commit-policy`,
  `sast`, `secrets` (full-history gitleaks — still covers all pre-split
  comfyless content, since history was not rewritten here), and the new
  `supply-chain-lock`.

## Out-of-band finding: three new CVEs

Wiring the lock-file job surfaced three advisories absent from
`osv-scanner.toml`: cryptography 49.0.0 (**8.2**, fixed in 50.0.0),
transformers 5.5.3 (**7.1**, fixed in 5.10.0), accelerate 1.13.0 (**7.1**, no
fix listed). Not caused by this slice — the gate had simply not been run during
17 idle days, and the job that would have shown it red is one of the skipped
three. Per §11 these are MUST bumps and separate slices; recorded as TECH_DEBT
2026-09-09 with a next-session trigger. The new job being red on arrival is the
correct signal, not a defect to paper over.

## Scope note

The `systemd/comfyless@.service` `Documentation=` line was edited here though
the unit belongs to slice 8. Doc-only, no security consequence, flagged for the
record.
