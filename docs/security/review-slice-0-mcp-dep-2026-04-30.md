AI-Disclosure: Claude (Opus 4.7) authored; Grant to review.

# Security Review — Slice 0: `mcp==1.27.0` Dep Bump

**Date:** 2026-04-30
**Commits reviewed:** `909b228` (slice 0 dep-bump) + `0742ee3` (CLAUDE.md pin-count fix)
**Reviewer:** `security-auditor` agent (Opus)

**Scope:** Dep bump only — `mcp==1.27.0` added to `pyproject.toml` line 23, `requirements.txt` line 9, and the regenerated `uv.lock`. No code, no imports.

**Verdict:** No blockers. Push slice 0 once findings M-1 / M-2 are filed (TECH_DEBT ok; ADR Changelog amendment for M-1 is the cleaner home).

---

## Findings

### BLOCKER
None.

### HIGH
None.

### MEDIUM

**M-1 — Hash-pinning enforcement is undocumented at both install paths.**
*Artifact:* `pyproject.toml`, `requirements.txt`, ADR-011 §7, `comfyless/README.md` (slice-1).
*Risk:* `uv.lock` records per-artifact `sha256` for every wheel and sdist (verified — see lines 715, 713 for the `mcp` wheel/sdist; transitives carry their own hashes). The lockfile is the integrity layer per global §11. However: `uv sync` defaults to honoring lockfile hashes, but `uv sync --frozen` is the discipline that fails-closed if `pyproject.toml` drifts from the lock; neither this rule nor a `pip install --require-hashes` equivalent for the ComfyUI Manager path is documented anywhere. Per global §11 ("For Red Zone code, install with `--require-hashes` or equivalent"), MCP is the substrate of the Red Zone surface in slice 1 and inherits the rule.
*Recommendation:* ADR-011 Changelog amendment naming (a) `uv sync --frozen` as the dev install command and (b) the user-visible install caveat — pip's `--require-hashes` requires a hashed `requirements.txt`, which a downstream `pip install -r requirements.txt` invocation does not produce from the current 9-line manifest. A `requirements-lock.txt` with `--hash=sha256:` lines exported from `uv.lock` (`uv export --format requirements-txt`) is the smallest delta. Defer to slice 1 if not landing now; file as TECH_DEBT.

**M-2 — `mcp[stdio]` extras-name drift in ADR-011 §7 and slice-1 vision doc.**
*Artifact:* `docs/decisions/ADR-011-comfyless-mcp-server.md` line 150.
*Risk:* `mcp` 1.27.0 publishes extras `cli`, `rich`, `ws` (no `stdio`); stdio is base. The ADR text "Install via the bare `mcp` package or the `mcp[stdio]` extra" is wrong. A future maintainer copying that line into a different manifest gets `WARNING: mcp 1.27.0 does not provide the extra 'stdio'` — which in some pip / uv resolver paths is a *non-fatal* warning, silently installing the bare package with the false impression that an extra resolved. Not a vulnerability today; documentation drift in a Red Zone ADR.
*Recommendation:* Changelog amendment to ADR-011 striking "or the `mcp[stdio]` extra" — leave the bare `mcp` install as the only documented form. Same edit covers the slice-1 vision doc.

### LOW

**L-1 — Transitive `cryptography==47.0.0` and `pyjwt[crypto]` enter dep closure (slice 1+ §5 trip).**
*Artifact:* `uv.lock` lines 697–712.
*Risk:* Slice 0 imports neither; slice 1+ MUST not import `pyjwt` or `cryptography` directly without re-tripping §5 Red Zone ownership (key handling, token validation). MCP itself uses `pyjwt` for the OAuth-resource-server path (HTTP transport only) — stdio transport does not exercise it. Per ADR-011 §6 ("v1 ships only stdio-transport"), neither library is reached on the slice-1 code path.
*Recommendation:* Note in slice-1 Vision: any `import jwt` or `from cryptography ...` re-fires `security-auditor` regardless of slice scope.

### INFORMATIONAL

**I-1 — Provenance.** `mcp` on PyPI is published from `modelcontextprotocol/python-sdk` (Anthropic-maintained). Upload metadata in `uv.lock` (`upload-time = "2026-04-02T..."`) is consistent with the SDK release cadence. PyPI does not yet enforce Sigstore for `mcp`; integrity via lockfile sha256 is the practical guarantee. No typosquat candidates on the same pull (no `mcp-*` lookalike in the resolved transitive set).

**I-2 — CVE check.** I cannot fetch live OSV/GHSA from this environment. Manual `pip-audit` on the resolved set (`mcp==1.27.0`, `cryptography==47.0.0`, `pydantic==2.13.3`, `starlette==1.0.0`, `uvicorn==0.46.0`, `httpx`, `pyjwt`, `python-multipart`, `anyio==4.13.0`) is required before push and SHOULD be re-run weekly until the slice-8 supply-chain cluster lands. Note: `python-multipart` has a CVE history (CVE-2024-24762, CVE-2024-53981) — confirm the lockfile-pinned version is post-`0.0.18`.

**I-3 — License posture.** `mcp` is MIT (per the python-sdk repo). Top transitives — `pydantic` MIT, `starlette` BSD-3, `uvicorn` BSD-3, `httpx` BSD-3, `anyio` MIT, `cryptography` Apache-2.0/BSD-3 dual, `pyjwt` MIT, `python-multipart` Apache-2.0, `sse-starlette` BSD-3, `jsonschema` MIT, `python-dotenv` BSD-3 — all permissive. No GPL/AGPL/LGPL surprises in the slice-0 closure.

**I-4 — `uv.lock` hash sufficiency.** Yes, for version-pinning intent (§11 main rule). Insufficient for `--require-hashes` discipline at the pip path; see M-1.

---

**Pull-back trigger:** None. Slice 0 is safe to push to `origin/main`. M-1 and M-2 are amendments to ADR-011, not slice-0 reverts.

**Files referenced:**
- `pyproject.toml`
- `requirements.txt`
- `uv.lock`
- `docs/decisions/ADR-011-comfyless-mcp-server.md`

---

## Addendum 2026-05-01 — I-2 closed

OSV.dev (`https://api.osv.dev/v1/query`, PyPI ecosystem) sweep performed against all 22 packages introduced by this slice at their lockfile-pinned versions:

```
mcp==1.27.0                          [clean]
cryptography==47.0.0                 [clean]
pydantic==2.13.3                     [clean]
pydantic-core==2.46.3                [clean]
starlette==1.0.0                     [clean]
uvicorn==0.46.0                      [clean]
httpx==0.28.1                        [clean]
httpx-sse==0.4.3                     [clean]
pyjwt==2.12.1                        [clean]
anyio==4.13.0                        [clean]
python-multipart==0.0.27             [clean]
sse-starlette==3.4.1                 [clean]
jsonschema==4.26.0                   [clean]
jsonschema-specifications==2025.9.1  [clean]
python-dotenv==1.2.2                 [clean]
pydantic-settings==2.14.0            [clean]
typing-extensions==4.15.0            [clean]
typing-inspection==0.4.2             [clean]
referencing==0.37.0                  [clean]
rpds-py==0.30.0                      [clean]
cffi==2.0.0                          [clean]
pycparser==3.0                       [clean]
```

22/22 clean. No open OSV / GHSA advisories. I-2 resolved. The auditor's recommendation to re-run weekly stands until the slice-8 supply-chain cluster (tracked in `local_agents`) lands automated CVE scanning.

`python-multipart==0.0.27` is past the `0.0.18` fix line for CVE-2024-24762 / CVE-2024-53981.

