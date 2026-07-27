# Security review — ADR-040 revision (design stage, pre-code)

AI-Disclosure: `security-auditor` subagent authored the findings; Claude (Opus 5)
drove the session and folded them; Grant reviewed.

**Model actually used: `claude-fable-5` (40/40 assistant turns).** The agent was
invoked WITHOUT a `model:` argument, per Grant's standing 2026-07-27
no-elevation instruction. It ran on Fable regardless, via the `model:
claude-fable-5` frontmatter pin in `~/.claude/agents/security-auditor.md`.
CLAUDE.md §5A documents that pin as silently ignored in Claude Code 2.1.117;
that is **no longer true** in the current build. Consequence: omitting `model:`
is not a way to avoid Fable billing. Raised with Grant; unresolved at time of
writing, and deliberately NOT fixed by editing the agent file (he instructed
that those files stay untouched).

**Date:** 2026-07-27
**Subject:** `docs/decisions/ADR-040-loop-output-inside-daemon-roots.md`, the
same-day revision (Grant's four rulings) — re-review of a design that had
already been reviewed on 2026-07-26 and materially changed since.
**Trigger:** §12 — the implementing slice touches `comfyless/server.py`, a Red
Zone IPC surface (Unix socket daemon) per the project review bar.
**Status:** all findings folded into the ADR before acceptance.

## Scope reviewed

Full revised ADR and the revision diff; `comfyless/server.py` (`_within`,
`_check_paths`, `_check_ref_paths`, `_resolve_ref_roots`, ping handler, savepath
handling, atomic reservation); `comfyless/refine.py` (`run_generation`,
`pin_static_refs`, `_write_json`, `verdict_record`, `_assert_no_paths`);
`comfyless/generate.py` (`_SKIP_SIDECAR_KEYS`, `_validate_params`,
`_load_sidecar`, `_expand_savepath_template`, `_resolve_savepath`,
`_delegate_to_server`, the `RefPathError` fallback, iterate mint sites);
`comfyless/cascade.py` batch-id mint; `comfyless/mcp_server.py`
(`_render_extracted_params`, `extract_params` handler); ADR-037 D5's prohibited
alternatives verbatim.

Not reviewed: ADR-038/-035/-015 full texts (spot-verified through the code they
govern); `_run_json_mode` (outside this ADR's blast radius); test suites (design
stage, no code exists yet).

## Findings

### HIGH — D1's "no move step" claim rested on a misread of `_resolve_savepath`

`_resolve_savepath` (`generate.py:522-524`) appends the 4-digit counter
**unconditionally**, starting at `counter = 1`. The daemon's first file for stem
`candidate_00` is therefore always `candidate_000001.png` — the first slot, not a
fired collision (a collision yields `…0002`).

The revision had asserted the opposite as forensic evidence and concluded that
making the savepath relative removes the move step entirely. It does not: the
daemon still does not write the canonical `output_dir/stem<ext>` name that
ADR-034 D7, the stale-other-extension warning (`refine.py:1897-1905`), and the
candidate-as-next-edit-source contract depend on. `shutil.move`
(`refine.py:1953`) is what reconciles the two namespaces today.

Dropping the rename would have forced an unstated choice: adopt counter-suffixed
names everywhere (an ADR-034 D7 amendment with a consumer list, and a same-stem
regeneration inside one run then yields `…0002`, breaking the `candidate_NN` ↔
file mapping), or keep a rename. Worse, an implementer who dropped it while
leaving a canonical-name assumption in place would let a **stale canonical file
in a reused directory be judged and pinned as the new candidate** — the
judge-vs-generation divergence ADR-038 D5 closed. `run_id` uniqueness protects
derived dirs but not explicit `--output-dir` reruns.

**Closed:** the forensic sentence is corrected and now explains Grant's standing
side question (the daemon logs `…0001`, the run dir shows `candidate_00.png`,
same bytes). D1 removes the MIRROR and the cross-tree move, and keeps a
same-directory `os.replace`. A negative test asserts the judge reads the bytes
the daemon wrote in that iteration.

### MEDIUM — uniqueness applies to derived run dirs only; the named test overclaimed

`run_id` is appended only to *derived* dirs (D3: derivation applies only when no
`--output-dir` was given). Two concurrent runs handed the same **explicit**
`--output-dir` still share it, and `pin_static_refs`' unconditional
`shutil.rmtree(refs_dir)` (`refine.py:3155`) still destroys the live sibling's
pinned refs and any completed run's provenance — the 2026-07-26 HIGH, alive on
the explicit path. The slice-plan test asserted a property the design provides
only for derived dirs.

**Closed:** stated as an explicit residual (an operator who names the directory
owns its concurrency), test scoped to derived runs, and a loud entry warning
added when an explicit dir already contains another run's `refs/`
(warn-don't-block).

### MEDIUM — the exclusive create's ordering was unspecified and is load-bearing twice

`pin_static_refs` calls `os.makedirs(refs_dir, exist_ok=True)`
(`refine.py:3156`), which **creates parent directories**. If the run dir's
exclusive create is sequenced after it, pinning silently materializes the run
dir and the assertion can never fire — restoring silent sharing. Separately, the
run dir is derived from a value the daemon supplied over the wire; the fresh
exclusive create is what bounds all subsequent client writes, and the `rmtree`,
to a directory this invocation provably just made, even against a buggy or
hostile ping response.

**Closed:** D1 now requires the `exist_ok=False` create to be the FIRST
filesystem operation on the run dir, with the ordering pinned by test rather
than by comment.

### MEDIUM — D3a did not scope the entry check, nor name the escape

Delegation is already skipped entirely when `--output` is set
(`generate.py:3271-3273`), so gating the new check on "a socket exists" would
refuse runs that were never going to reach the daemon. Separately, refusal
replaces a fallback that genuinely works on a box with spare VRAM — a
deliberate policy change (Grant ordered it) that the text should own and give an
escape from, per warn-don't-block.

The reviewer independently confirmed the reversal is justified: the one-shot's
`RefPathError` → in-process fallback is real at `generate.py:3330-3339` and its
warning text matches the ADR's quotation.

**Closed:** the check fires only when a socket exists AND delegation is not
skipped; daemonless and `--output` paths untouched; the refusal message names
`--output` and `--ref-root` as escapes.

### LOW — D2a's test is a tripwire, not a control

The framing was already honest (better than most, per the reviewer), but one
sentence overclaimed: "the client that would leak never asks" holds only for
first-party in-repo clients. The daemon answers **any** same-UID caller that
sets the flag, and the negative test asserts the absence of a string in a file
that today contains no daemon-socket client code at all — no `socket_path`, no
`_send_server_command`. Today's exposure via the mcpo/OWUI chain is nil for the
same reason. The spec also did not say the flag is honored only on `ping`.

**Closed:** relabeled a regression tripwire in D2a's own words; `report_roots`
is schema-validated as bool and honored ONLY on `type: "ping"` (ValidationError
otherwise), with a matching negative test.

### LOW — body promised wire-value type-checking that no test covered

D3 commits to ADR-012-style checking of `output_dir` / `ref_image_roots` (str,
absolute, NUL-free) with pre-D2 fallback on failure, but the named tests covered
only the "reports nothing" case. That value feeds `makedirs` / `rmtree`
derivation.

**Closed:** negative test added for each malformed shape (non-str, relative,
NUL-bearing) behaving exactly as the pre-D2 case.

### LOW — D1b verified sound; one nit, one understatement

Verification recorded rather than a defect:

- **No-wire-change claim TRUE** — both sidecars are written client-side
  post-response, at `generate.py:3294-3307` and `refine.py:2456-2459`.
- **`_assert_no_paths` still holds** — it is key-based (`refine.py:1189-1202`);
  `run_id`'s key and hex value are path-free.
- **The `iterate_batch_id` skip-keys wart is real** — genuinely absent from
  `_SKIP_SIDECAR_KEYS` today. Adding both keys is correct and weakens nothing:
  they were never schema params, so `_validate_params` drops them regardless;
  the set only silences the warning.
- **The MCP claim was UNDERstated** — `run_id` cannot survive `extract_params`
  even unlisted: the non-cascade path normalizes through `_validate_params`
  (drops non-schema keys) and the cascade path renders from raw through an
  allowlist. The list entry documents a structural guarantee rather than
  providing one. ADR corrected to say so.
- **Mint-site inconsistency confirmed** — 36-char dashed at `generate.py:3894`,
  32-hex at `cascade.py:895`. Shortening to 8 hex is safe; grep confirms
  equality-grouping only, nothing parses the value.
- **Nit:** 8 hex is 32 bits of real randomness (uuid4's version nibble sits
  outside `hex[:8]`), so the 50% birthday point is ~77k same-stem runs, not the
  ~65k the ADR claimed. Corrected.

## Cleared without findings

- **ADR-037 D5 prohibited alternatives** — `report_roots` is a read-only
  disclosure request, not a trust-assertion field. No daemon-gate exemption, no
  root merging, no per-request ref-root extension (verified against
  ADR-037:193-196).
- **Write authority / root widening** — daemon write behavior untouched (D1
  removes mirror-tree writes; ping additions are read-only). `ref_image_roots`
  is not widened; the run dir nests inside `output_dir`, already a ref root.
- **Line references** — every load-bearing citation verified correct:
  `refine.py:1908`, `server.py:866-869`, `refine.py:3155`,
  `generate.py:120/3303/3331/3894`, `cascade.py:895`, `server.py:418`. No
  incorrect references found.

## Outcome

All findings folded into ADR-040 before acceptance; see its 2026-07-27 Changelog
entry. The reviewer's overall posture: the design is sound, grants no new write
authority, widens no roots, and smuggles in none of ADR-037 D5's prohibited
alternatives. The HIGH was a text/claim defect with a real implementation trap
behind it, not an unsafe design.

**Still required before code lands:** slice 1 touches `server.py` and therefore
needs a `security-auditor` pass on the CODE, not only this design pass.
