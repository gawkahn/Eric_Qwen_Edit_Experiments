# Security review — ADR-045 slice 3d (attribution clearing)

AI-Disclosure: `security-auditor` (Claude Fable 5, transcript-confirmed
`claude-fable-5`, no fallback) reviewed; Claude Opus 5 authored the slice and
this write-up; Grant reviewed and made the licensing calls.

Date: 2026-09-09
Slice: ADR-045 slice 3d — strip the Eric Hiss copyright/attribution header
template from the extracted package, and clear the ~28 functional lines the
ADR's original three-function clearing set missed.
Trigger: the change edits the CONTENT of §12 Red Zone files.

## Red Zone surfaces actually touched

`src/comfyless/server.py` and `src/comfyless/mcp_server.py` — **comment and
docstring text only**, verified line by line by the auditor.

`src/comfyless/refine.py` was **NOT** modified: it never carried the header.
My review brief claimed three Red Zone files; the auditor checked rather than
accepted that, and the record is corrected here. `refine.py` shows a
zero-length diff and contains no Eric attribution.

`src/comfyless/core/eric_diffusion_fp8_ops.py` (the weight-file content parser)
was also touched, comment-only — see the MEDIUM below.

## Verified clean

1. Every removed line across the 24 files is a `#` comment (copyright /
   license / URL) or an `Author:` line inside a module docstring. No import,
   statement, decorator, encoding cookie, `# noqa` or `# type:` directive was
   removed or reordered. All four shebang files keep `#!/usr/bin/env python3`
   as line 1. All 46 package files compile.
2. **Nothing reads these docstrings at runtime.** Grepped `src/`, `nodes/`,
   `pipelines/`, `scripts/` for `__doc__` consumption: none. In particular the
   MCP tool descriptions advertised to an LLM agent are hardcoded literals
   (`mcp_server.py:711+`), not docstring-derived, so the strip cannot change
   what the agent is told.
3. **Blast radius confined to `src/comfyless`.** `nodes/` and `pipelines/` are
   untouched and retain their headers across 37–38 files — the node pack
   remains a properly attributed licensed fork, which was the compliance
   requirement.
4. **The delta-shape guard is no weaker.** Rewriting "try `.reshape()`, catch
   RuntimeError, skip" as an explicit `numel` check yields an identical skip
   set for every producible input (both delta sources are dense: a safetensors
   `.diff` tensor or a dense matmul product). There is **no input the old path
   skipped that the new path merges** — the security-relevant direction. The
   one divergence is an OOM during reshape's contiguity copy, which the old
   code swallowed as "not applicable" and the new code propagates: fail-closed,
   and arguably a bug fix.
5. **No adapter registers that previously would not have.** The
   `hasattr` → `getattr(...) is None` change diverges only when `peft_config`
   is `None`, a state nothing in this repo or diffusers produces (diffusers
   `del`s the attribute; peft sets the underscored `_peft_config` on a
   different object). In that unreachable state the old code crashed *after*
   the weights were already merged, leaving merged-but-unregistered weights;
   the new code completes registration. Registration never gated the merge in
   either version.

## Findings and disposition

### MEDIUM — the Red Zone gate no-opped on a Red Zone file in this very diff

`scripts/git-policy/_red-zone-paths.sh` still matches
`nodes/eric_diffusion_fp8_ops.py`, a path deleted when slice 1b moved the fp8
weight-file parser to `comfyless/core/`. This diff edits that parser (comment
only), and the T1 gate did not fire — this review happened because it was
invoked by hand. `src/comfyless/core/lora_adapters.py` is un-gated for the same
reason.

**Disposition: PULLED FORWARD.** This is the registered TECH_DEBT of
2026-08-22 (amended 2026-09-09 for the src/ move), whose trigger was "no later
than slice 6". The auditor's point — that we now have an observed instance of
the gap failing to fire on a real Red Zone edit rather than a theoretical one —
is the argument for not waiting. Fixed in its own commit immediately after this
slice, per global §4 (the policy layer is its own change boundary), rather than
widened into this diff.

### INFO — post-merge registration now completes where it previously crashed

Recorded, no action. See "Verified clean" item 5.

### LOW (from `code-reviewer`, folded in before commit)

- The CPU-offload fallback was re-expressed with `filter(None, components)`,
  which tests truthiness where the original tested `is not None` — an empty
  `nn.ModuleList` would have been silently skipped. Unreachable with today's
  components, but an unfaithful translation. **Fixed.**
- `core/eric_diffusion_manual_loop.py` carried a comment describing this as "a
  CC BY-NC / Commercial dual-licensed package" — a license self-declaration the
  extracted repo will not carry. **Fixed** (reworded, comment-only). The tree
  now greps clean for `Hiss`, `EricRollei`, `Copyright`, `LICENSE.txt` and
  `CC BY-NC` under `src/comfyless`.
- The rewritten reshape guard had **no battery-level coverage** — the existing
  case exercised a different module's untouched copy. **Fixed:** three checks
  added to `test_fp8_single_file.py`, including a fixture assertion and a
  positive case proving the merge path is reached, so the negative case cannot
  pass vacuously.

## Conscious keep, recorded not fixed

Module *filenames* under `src/comfyless/core/` still carry the `eric_` prefix
(`eric_diffusion_*.py`, `eric_lora_format_convert*.py`, `eric_krea2_convert.py`).
That is naming, not license text, and outside slice 3d's declared boundary.
Noted here so slice 6 decides deliberately whether the names survive extraction
rather than discovering them in a grep.
