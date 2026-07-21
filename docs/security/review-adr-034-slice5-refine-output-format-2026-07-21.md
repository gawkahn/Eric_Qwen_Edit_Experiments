AI-Disclosure: Claude (Fable 5) authored; Grant reviewed.

# Security Review — ADR-034 slice 5 (D7): refine output-format threading

Date: 2026-07-21
Reviewer: `security-auditor` (Fable)
Surface: `comfyless/refine.py` (Red Zone — LLM-as-judge output influencing
generation; seed-image ingestion; on `scripts/git-policy/_red-zone-paths.sh`)
Verdict: **Approve with findings** — no CRITICAL/HIGH. Two MEDIUM (both
warn-level provenance/consistency hardening) + one INFO. **All three addressed
in-slice** (see Resolution notes).

## Summary

Slice 5 (D7) threads `--output-format {png,jpeg,jpg}` + `--quality` through
`comfyless/refine.py`: an `OutputFormat` is resolved once in `main()`, passed to
`refine_loop` → `run_generation`; the canonical candidate path is
`output_dir/candidates/candidate_NN<ext>`; the daemon path sends the raw format
name + quality fraction via `_daemon_namespace` into the one canonical wire
builder (`generate._build_server_request`); the cold path forwards the object to
`generate()`. It also closes the slice-2 latent AttributeError on refine's
daemon path (`_daemon_namespace` did not supply the `output_format`/`quality`
fields `_build_server_request` now reads).

Threat model: the LLM judge/planner is the untrusted actor (its output must not
influence paths or extensions); the daemon is trusted-but-versioned; the
operator CLI is trusted; the `*.verdict.json`/judge context must remain
path-free.

Traced: (a) every source that can feed the on-disk extension (argparse
`choices` → `resolve_output_format` enum map `_FORMATS` — extension is an
enum-derived constant, never a caller string); (b) the wire fields and their
server-side value checks (`server._validate_request`) plus type checks
(`_RUNTIME_KIND`); (c) the daemon savepath re-rooting + `shutil.move`
re-normalization; (d) the audit artifacts (`verdict_record`, `_assert_no_paths`,
sidecar); (e) error paths (bad `--quality`/format fails closed before any
generation; daemon validation errors are fatal `RefineError`).

Posture is good. The LLM verdict schema (scores/critique/override_prompt/
lora_ops) has no channel into format or path; the extension cannot be
judge-influenced. No traversal is introduced.

## Findings

### [MEDIUM-1] Daemon version-skew can mislabel candidate bytes via the canonical move
Location: `comfyless/refine.py` run_generation daemon branch.
Risk: `shutil.move(out_path, canonical)` renamed whatever the daemon returned to
the client-derived `stem + ext` without checking that the daemon's extension
agreed. Against a long-lived pre-slice-2 daemon, `validate_machine_request`
passes unknown keys through and the old daemon ignores `output_format`, saving
PNG — which refine then renamed to `candidate_NN.jpg` (PNG content, `.jpg` name;
`--quality` silently dropped). A stale resident daemon is a demonstrated
operational reality in this deployment. Not adversarial (daemon is
operator-owned), but it silently defeats the format request and mislabels
provenance.
Remediation (recommended): compare the daemon's returned extension with the
expected one; on mismatch, warn loudly (warn-don't-block) rather than move
silently.
**Resolution (2026-07-21, fixed in-slice):** run_generation now compares
`os.path.splitext(out_path)[1].lower()` with the expected extension; on
mismatch it (a) emits a loud `[refine] WARNING: ... likely a stale daemon
(restart it to honor the format)` and (b) keeps the daemon's bytes **honestly
labeled** by moving to `stem + daemon_ext` (never renaming PNG bytes to `.jpg`).
Regression tests: `test_refine.py` MEDIUM-1 cases (skew warns + honest relabel;
matching ext → no warning, canonical `.jpg`).

### [MEDIUM-2] Format switch on a reused --output-dir leaves stale other-extension images beside fresh same-stem sidecars
Location: `comfyless/refine.py` run_generation / loop candidate write.
Risk: candidate names are deterministic (`candidate_NN`); a rerun into the same
`--output-dir` with a different `--output-format` overwrites
`candidate_NN.json`/`.verdict.json` while the prior run's `candidate_NN.png` (or
`.jpg`) survives, so the stem no longer identifies one image (sidecar describes
the new run, sits beside the old image; `winners/` accumulates a mixed pair).
Same hazard class as the slice-2 daemon MEDIUM (closed there with the dual
O_EXCL image+json reservation); refine had no equivalent. Partial mitigation
already present: `--seed-image` on a `.jpg` fails closed (generate.py D6 jpg
refusal).
Remediation (recommended): before generating, check for `stem` + the other
known extensions; if found, warn (or unlink).
**Resolution (2026-07-21, fixed in-slice):** run_generation scans the output dir
for `stem + <other _EXT_TO_NAME extension>` before generation and emits a loud
`[refine] WARNING: ... mispaired stem` when a stale sibling is found. Chose
**warn, not unlink** (warn-don't-block; the operator owns the files — deleting
their prior output silently is the more dangerous default). Regression tests:
`test_refine.py` MEDIUM-2 cases (stale sibling warns + is NOT deleted; clean dir
→ no warning).

### [INFO] Default CLI runs always emit output_format/quality on the wire
Location: `comfyless/refine.py` main() / `_daemon_namespace`.
Risk: `main()` always resolves an `OutputFormat` (never None), so every
CLI-driven daemon request now carries `output_format="png"` + `quality=0.7`; the
`None → omit` branch is reached only by programmatic callers. Harmless today
(daemon value-checks accept both; old daemons pass unknown keys through). The
`_daemon_namespace` docstring described a byte-for-byte path the CLI never takes.
**Resolution (2026-07-21, fixed in-slice):** docstring corrected to state that a
CLI-driven png run does send `output_format="png"`/`quality=0.7`, and the
`None`-omission branch is programmatic/test-only.

## Areas considered clean (no findings)

- **Path containment / traversal:** extension is an enum constant from
  `_FORMATS`; the flag is argparse-`choices`-restricted; the stem is
  code-generated; judge output has no format/path channel.
- **Wire validation:** both fields type-checked (`_RUNTIME_KIND`) and
  value-checked server-side (`server._validate_request`); no new unvalidated
  field, no nulls on the wire (None omits cleanly).
- **Fail-closed error paths:** bad `--quality`/format rejected before any
  generation.
- **Audit-trail integrity:** `verdict_record` stays path- and format-free;
  `_assert_no_paths` gate unaffected; `os.path.splitext` stem derivation is
  extension-agnostic.
- **D5/opaque-handle:** no new agent-facing filesystem detail (CLI/operator
  surface only; `mcp_server.py` untouched).
- **Scope:** all refine.py changes fall within the declared D7 edit scope.
