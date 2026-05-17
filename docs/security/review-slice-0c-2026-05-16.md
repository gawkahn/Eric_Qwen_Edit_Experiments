# Security Review — Slice 0c / ADR-013 (Comfyless torch divergence)

**AI-Disclosure:** Claude (Opus 4.7, 1M context) authored as security-auditor subagent; Grant reviewed (pending sign-off).
**Date:** 2026-05-16
**Scope:** Design-time supply-chain / §11 hash-pin integrity review of `docs/decisions/ADR-013-comfyless-torch-divergence.md` (proposed) and the slice-0c Vision (`docs/vision/slice-0c-cuda-torch-realignment.md`, approved 2026-05-16). No code yet for this slice; the architectural rule and the §8 ordered execution plan are what is being reviewed.
**Reviewer:** security-auditor (Opus, model pinned at invocation per global §5A).
**Risk level reviewed against:** L2 (per slice-0c Vision §Posture). The dep manifest is the boundary; comfyless's Red Zone surfaces (MCP request boundary, validator) exist downstream and are governed by ADR-011 / ADR-012 — they are not in scope for this review except where the dep-manifest choices reach into them.

> **Workflow note:** The security-auditor agent has read-only tools. The review content was produced by the agent; the parent session persisted it to disk. Any subsequent re-reviews after ADR amendments will append below the initial verdict line.

---

## Verdict (round 1)

**CHANGES REQUIRED** — 1 HIGH, 2 MEDIUM, 3 INFO. None of the findings invalidate the architectural rule (comfyless deps diverge from ComfyUI's via uv-managed `.venv`; `pyproject.toml` is SoT; `uv.lock` carries integrity hashes; `requirements.txt` stays pip-compatible for ComfyUI Manager). All findings are small ADR-text or §8-checklist additions that can be folded in this same session before status flips to `accepted`.

The §11 hash-pin posture is correctly preserved by the slice (lockfile-hash discipline via `uv sync`). The decision to **not** opt into `--require-hashes` is defensible at L2, but the rationale in §5 should be tightened — see F-2 below. The `--require-hashes`-rejected alternative (C) is honest about its tradeoff but understates one channel where the dep manifest *does* touch a Red Zone surface.

---

## Findings

### F-1 — HIGH: §8 ordered execution lacks a `uv lock --check` (or equivalent) gate before commit

**Location:** ADR-013 §8 Order of operations, steps 5–7. Slice-0c Vision proof-hook line 100 (`uv lock --check exits 0 with no proposed changes`).

**Risk:** The ADR repeatedly characterises slice shape A as "unchanged pins, establish divergence only." §6 says "all ten direct pins remain at their current values." §3 says `uv.lock` is the integrity-hash anchor and is regenerated when `pyproject.toml` changes. §8 step 6 says "uv.lock regenerated (no pin movement; lock content should be byte-identical to before, modulo any uv version migrations)."

What's missing: a positive assertion in the ordered execution plan that the lock content **was in fact** byte-identical (or, where it differs, that each delta is accounted for). The slice-0c Vision proof hook at line 100 names `uv lock --check exits 0 with no proposed changes`, but ADR-013 §8 does not actually require it. The risk vector is concrete:

  - The slice writes a comment block above `torch==` in both `pyproject.toml` and `requirements.txt`. Comments do not affect resolution; `uv lock` re-run after that edit should be a no-op on the dependency graph.
  - However: any uv version difference between the machine where `uv.lock` was last regenerated (slice 0b commit `3665461`) and the machine running slice 0c can rewrite metadata sections of the lock (uv revision number, source URL canonicalization, etc.) silently.
  - More importantly: any transitive dependency that has had a PyPI re-resolve since slice 0b (e.g. a `huggingface-hub` micro-version bump now within the existing constraint window) gets pulled into the new lock when uv re-resolves. The "unchanged direct pins" framing hides that, and the operator reading the ADR is led to believe nothing changes.

Without the `--check` gate, slice 0c can commit a lock with new transitive SHA-256s under the "no version movement" framing. That is exactly the supply-chain shape §11 paragraph 4 designs the lockfile to defend against — *if* the lock is verified.

**Remediation (ADR-side, smallest text):** Amend ADR-013 §8 step 6 to gate the regeneration on `uv lock --check` pre-edit, inspect the post-edit diff, and surface any non-metadata delta to Grant as a separate go/no-go.

**Status:** Pending ADR amendment.

---

### F-2 — MEDIUM: `--require-hashes` rationale in §5 elides one channel where the dep manifest *does* reach into a Red Zone surface

**Location:** ADR-013 §5 ("Hash-pinning posture") and Alternatives Rejected C.

**Risk:** §5 and Alternative C correctly cite global §11 paragraph 4: "§11 as written is version pinning … For Red Zone code, go further: install with `--require-hashes` … A project whose surface is Red-Zone-heavy may opt the whole repo into `--require-hashes` by default." Both reach the same conclusion: comfyless's dep manifest is not Red Zone, so `uv.lock` integrity hashes suffice.

The reasoning is correct as stated. But it elides one channel:

comfyless's runtime *is* a Red Zone surface (MCP request handler per ADR-011 §3, machine-boundary validator per ADR-012). The deps installed by `uv sync` are loaded into the same process that handles MCP `tools/call("generate", ...)` requests from an LLM agent. A compromised-republish attack on, say, `transformers==5.5.3` (re-upload with the same version string but different bytes) would land in the runtime executing those Red Zone code paths. The `uv.lock` SHA-256 catches this *if and only if* the install actually verifies hashes — which `uv sync` does by default, per uv's documented behavior, and the ADR rests on that contract.

The ADR should name this assumption explicitly. As written, a reader could conclude the manifest is so far from Red Zone that hash discipline is mainly hygiene. The reality is: the manifest is one trust hop from a Red Zone process, and the lockfile is doing real work (not just hygiene) by enforcing integrity at every install.

Additionally, the ADR does not state what fail-closed behavior `uv sync` exhibits on a hash mismatch — that is the actual control that matters here. If uv sync exits non-zero on hash mismatch (it does, per uv documentation), the system is hash-verified. If a future uv version added a `--no-verify-hashes` flag and the slice's reproduce script used it, the chain would silently break.

**Remediation (ADR-side):** Amend §5 + Alt C to name the runtime-consumer Red Zone and the `uv sync` hash-verify default as the load-bearing control.

**Status:** Pending ADR amendment.

---

### F-3 — MEDIUM: Comment block (§4) routes ComfyUI Manager operators to the ADR but doesn't warn them about transitive-graph divergence drift

**Location:** ADR-013 §4 ("Comment block in `pyproject.toml` and `requirements.txt`").

**Risk:** The comment is accurate but operator-incomplete. Today's pins are unchanged; the comment says they "may diverge in the future." What it doesn't say is what a ComfyUI Manager operator should observe / verify / report-on when divergence does happen later. Concretely, two failure modes a future operator could hit, neither of which is signposted:

  1. **Resolver conflict in ComfyUI's venv.** Once a future slice bumps `transformers` (or any other pin) ahead of ComfyUI's bundled choice, `pip install -r requirements.txt` inside ComfyUI's venv may succeed but downgrade ComfyUI core's pinned transformers — or fail with a conflict that an operator interprets as "the node pack is broken." The current comment gives them no anchor to triage this.
  2. **Operator copying `requirements.txt` into a non-ComfyUI venv.** A user installing comfyless dev via `pip install -r requirements.txt` instead of `uv sync` *today* will work fine because pins are unchanged. The moment pins diverge, the pip path loses the lockfile-hash integrity story (per F-2). The comment block says "comfyless dev/test environment is the uv-managed `.venv`" but does not say "DO NOT use `requirements.txt` to set up the comfyless dev/test environment" — leaving the operator to infer.

This isn't a security defect today (pins are unchanged; pip path has always been hash-less). It is a clarity gap that will become a security observability gap once divergence actually happens.

**Remediation (ADR-side):** Lengthen the §4 comment-block text with the two operator failure modes.

**Status:** Pending ADR amendment.

---

### F-4 — INFO: Pre/post `uv.lock` SHA not recorded in Changelog

**Severity:** INFO
**Location:** ADR-013 §6, §8 step 4 (baseline PNG SHA-256 recorded) and step 6 (lock regeneration).

The ADR already records the baseline PNG SHA-256 in the Changelog as the smoke-test anchor (§7, §8 step 4). A parallel discipline for `uv.lock` itself — record the pre-slice `sha256sum uv.lock` in the Changelog at step 5/6, alongside the post-slice value — makes the "byte-identical (modulo uv migrations)" claim auditable after the fact. This is INFO because F-1 already requires `uv lock --check` as the gate; the pre/post SHA recording is cheap belt-and-braces.

**Remediation:** Optional fold-in to §8 step 6.

**Status:** Optional.

---

### F-5 — INFO: Alternatives Rejected list is complete for L2; one additional alternative is worth naming for future readers

**Severity:** INFO
**Location:** ADR-013 Alternatives Rejected A–E.

The five alternatives are complete and well-reasoned for the architectural rule under L2. One alternative not named that future readers may ask about:

**F. Split `requirements.txt` and `pyproject.toml` direct-pin sets (intentional divergence, not lockstep).** The current "Package-manager split" rule in project CLAUDE.md (and §3 of this ADR) requires `pyproject.toml` and `requirements.txt` to list the same direct pins in the same order. A future ADR could split them — ComfyUI Manager gets pins compatible with ComfyUI core (with `>=` floors per upstream's expected behavior), while comfyless's `pyproject.toml` carries the strict exact pins. The ADR implicitly rejects this by saying the comment block is the shape (b) of slice-0c Vision invariant 9. Naming it explicitly as a rejected alternative — "rejected: keeps the manifest-agreement rule simple; revisit if ComfyUI Manager resolver conflicts force the issue" — helps the next slice that wonders why this isn't done.

**Remediation:** Optional addition to Alternatives Rejected.

**Status:** Optional.

---

### F-6 — INFO: §8 step 10 reviewer cadence is correct; one cross-check note for future amendments

**Severity:** INFO
**Location:** ADR-013 §8 step 10 ("`code-reviewer` (Opus) on each code-touching commit. Commit batch (3–5 commits, `feat(deps):` prefix).")

The reviewer cadence is correct — `code-reviewer` (Opus) on each code-touching commit, `security-auditor` already invoked on this ADR. One nuance worth recording for the next dep-bump slice (when pins actually move):

When a future slice bumps `torch` / `diffusers` / `transformers` / `accelerate` / `peft`, the per-commit `code-reviewer` should be supplemented with `security-auditor` (Opus) per slice-0c Vision §Red Zone ownership ("supply-chain audit: every new direct-pin version brings new SHA-256 hashes … hash-mismatch = bail out"). Slice-0c Vision line 111 already mandates this. ADR-013 §8 step 10 currently names only `code-reviewer` for the commit batch. Today's slice doesn't move pins so `security-auditor` is correctly limited to this ADR design review — but a future reader copying §8 as a template for an actual version-bump slice would miss the `security-auditor`-per-commit requirement.

**Remediation:** Optional clarification in §8 step 10.

**Status:** Optional; doesn't affect this slice's correctness, only future readers using §8 as a template.

---

## Cross-cuts (what is correctly handled, called out for the record)

1. **Order of operations (§8) respects global §12.** ADR drafted → security review → status flip → code. Step 1–3 are sequenced correctly. The `code-reviewer` (Opus) per code commit is consistent with project CLAUDE.md "Review rules."

2. **No security control is bypassed in §8.** The ordering does not skip the existing `block-pyproject-floors.sh` hook (no floors introduced; comment lines don't trigger it), does not skip the `check-ai-disclosure.sh` hook (commits get the trailer), and does not skip `code-reviewer` on the code-touching commits.

3. **`--require-hashes` rejection is legitimately L2 for the dep manifest.** Global §11 paragraph 4 allows it for non-Red-Zone projects. The remaining gap (F-2) is about framing, not about flipping the decision.

4. **PyPI default wheel resolution is correctly not pinned to pytorch.org index.** Alternative B's rationale (operator-dependent setup) is correct; pinning the index adds an attack surface (resolver fan-out to an additional registry) without offsetting benefit at cu130. If PyPI ever stops shipping cu13x, the calculus inverts.

5. **CLAUDE.md update at §8 step 7 is in scope per slice-0c Vision §Coordination notes.** That step touches CLAUDE.md line 67 only (test-runner path; suite count 7→8; test count 732→850). No drift.

6. **ComfyUI Manager path unchanged.** §3 preserves the pip-compatible `requirements.txt` shape. The comment block is non-semantic to pip's parser. The hash-less pip path is unchanged from today's posture (which is what F-2 is asking the ADR to name).

7. **MCP surface (Red Zone per ADR-011) is correctly out of scope.** This slice does not modify `comfyless/mcp_server.py` (does not exist yet — slice 1 work), validator (ADR-012, just landed), or any path-allowlist code. The dep manifest is the only boundary; the runtime consumers are governed by their own ADRs.

8. **`.venv` lives on mergerfs per slice-0c Vision Pointers; HF cache stays at bind-mount.** No filesystem-locking concern surfaced by the divergence rule itself. Per global Filesystem-constraint note, if `uv sync` or runtime imports wedge on `.venv` under mergerfs, the slice's fallback (`UV_PROJECT_ENVIRONMENT`) handles it operationally; not a security concern.

---

## What was NOT reviewed (per slice prompt)

- Specific dep versions (torch / diffusers / transformers / accelerate / peft / mcp / etc.) — pin set is unchanged per slice shape A; their security history is not the question this slice asks.
- Code (none exists yet for this slice).
- Operational concerns (test-runner path correctness, file-boundary discipline against the three concurrent sessions).
- ComfyUI Manager's resolver behavior on hypothetical future divergence (today's pins are aligned with ComfyUI's choice; no divergence to test).
- The MCP server, validator, or any Red Zone surface — governed by ADR-011 and ADR-012, which already have their own reviews.

---

## Summary table

| ID | Severity | Title | ADR section affected | Smallest fold-in |
|---|---|---|---|---|
| F-1 | HIGH | §8 plan lacks `uv lock --check` gate | §8 step 6 | Add lock-check + diff-inspection before commit |
| F-2 | MEDIUM | `--require-hashes` rationale elides Red Zone consumers | §5 + Alt C | Name the runtime-consumer Red Zone, name `uv sync` hash-verify as the load-bearing default |
| F-3 | MEDIUM | Comment block doesn't warn on future-divergence triage | §4 | Lengthen comment text with the two operator failure modes |
| F-4 | INFO | Pre/post `uv.lock` SHA not recorded | §8 step 6 | Optional Changelog discipline |
| F-5 | INFO | Alternative F (split pin sets) not named | Alternatives Rejected | Optional addition |
| F-6 | INFO | §8 reviewer cadence is correct but copy-template for next slice needs note | §8 step 10 | Optional note for future readers |

---

## Verdict line (round 1)

**CHANGES REQUIRED.** Fold F-1, F-2, F-3 into ADR-013 before flipping Status from `proposed` to `accepted`. F-4 / F-5 / F-6 are optional and may be merged or skipped at Grant's discretion. Re-fire `security-auditor` (Opus) on the amended ADR; expected verdict `CLEAN` once F-1 / F-2 / F-3 are addressed.

---

## Round 2 (2026-05-16)

**AI-Disclosure:** Claude (Opus 4.7, 1M context) authored as security-auditor subagent, round 2; Grant reviewed (pending sign-off).
**Scope:** Verify each of the six round-1 findings was faithfully folded into ADR-013; scan amended sections for new gaps; cross-check ADR §8 against slice-0c Vision invariants 1, 5, 6, 8, 9, 11, 12 and proof hooks (lines 89–102).

### Per-finding verdict

**F-1 (HIGH) — ADDRESSED.**
§8 step 6 now contains an explicit `uv lock --check` gate BEFORE the comment-block edit, with the right semantics: the pre-edit check asserts the existing lock is in sync with `pyproject.toml` (comments don't affect resolution so a clean pre-edit check is the right baseline), and a non-zero exit STOPS the slice and surfaces drift to Grant for go/no-go before any edits land. The post-edit re-run is bounded: "pure-metadata diffs (uv revision number, source URL canonicalization) are acceptable" — and crucially the surface is named explicitly: "Any transitive-dep version movement or new SHA-256 hash for an unchanged version surfaces to Grant as a separate go/no-go BEFORE commit." That last sentence is the exact control round 1 wanted — explicit, not implied. The Changelog fold-in entry mirrors the gate in plain language ("`byte-identical (modulo uv migrations)` claim is now a verified assertion, not a hope"). This also satisfies slice-0c Vision proof-hook line 100, which round 1 noted was named in the Vision but not wired into the ADR.

**F-2 (MEDIUM) — ADDRESSED.**
§5 paragraph "Load-bearing assumption (per security-auditor F-2)" names both required elements:
  - `uv sync`'s default per-artifact SHA-256 verification at install time as the load-bearing control (and the manifest-non-configurability that makes it load-bearing).
  - The runtime-consumer Red Zone surfaces — MCP request handler (ADR-011 §3) and machine-boundary validator (ADR-012) — both loading into the same process as `torch` / `diffusers` / `transformers`, which is what makes the lockfile-hash discipline "real supply-chain work, not just hygiene."
The "revisit the decision the moment uv's hash-verify default changes" trigger is also stated, which is the right hedge. Alternative C carries the cross-reference paragraph ("Per security-auditor F-2 (folded into §5)…") so a reader who jumps to alternatives gets routed back to the §5 framing. Both halves of F-2 are satisfied.

**F-3 (MEDIUM) — ADDRESSED.**
§4 comment-block text now contains both operator-failure-mode anchors round 1 required:
  - "comfyless dev/test must use `uv sync` (not `pip install -r requirements.txt`) so the lockfile-hash integrity contract … holds." — the explicit DO-NOT-use-requirements.txt warning.
  - "ComfyUI Manager installs may surface as a resolver conflict against ComfyUI core — that conflict is upstream's resolver, not a node-pack break." — the future-divergence resolver-triage signal.
Both failure modes named in the file every operator reads first. The text remains a single block in both `pyproject.toml` and `requirements.txt`, preserving the §4 "same comment text in both files" rule.

**F-4 (INFO) — ADDRESSED.**
§8 step 4 captures pre-slice `sha256sum uv.lock` in the Changelog at baseline time; §8 step 6 captures post-edit `sha256sum uv.lock` after the `uv lock` re-run. Both anchors are explicit, matching the baseline-PNG SHA discipline already used for the smoke image (§7). The Changelog fold-in entry confirms the pattern ("cheap belt-and-braces alongside the F-1 gate").

**F-5 (INFO) — ADDRESSED.**
Alternative F ("Split the direct-pin sets between `pyproject.toml` … and `requirements.txt`") added to Alternatives Rejected with the correct rationale: it would re-introduce floor specifiers in `requirements.txt` (collides with the `block-pyproject-floors.sh` hook from slice 0), it doubles the audit surface ("one set of pins" → "two sets of pins with a divergence policy"), and the comment block (§4) is the lighter-weight signal. Re-evaluation trigger named ("revisit only if ComfyUI Manager resolver conflicts against ComfyUI core's deps actually force the split — a future ADR amendment, not this one"). Faithful to F-5's intent.

**F-6 (INFO) — ADDRESSED.**
§8 trailing note ("Note for future readers using §8 as a template (per security-auditor F-6)") states the right two facts: today's slice ships unchanged pins (shape A) so `security-auditor` is correctly invoked once on the ADR design; the FIRST future version-bump slice that copies §8 as a template MUST layer `security-auditor` (Opus, `model: "opus"` at invocation) onto each code-touching commit IN ADDITION TO `code-reviewer`, per slice-0c Vision §Red Zone ownership. The "do not copy §8 verbatim" instruction prevents the failure mode round 1 worried about (template-copy strips the version-bump-specific reviewer-cadence requirement).

### Cross-cuts on amended sections

1. **Vision invariants 1, 5, 6, 8, 9 shape (b), 11 — all wired into §8.** Step 5 checks invariant 1 (`torch.version.cuda` starts with `"13."`). Step 6 + §6 satisfy invariant 5 (unchanged pins). Step 8 covers invariant 6 (850/850). Step 9 covers invariant 8 (pixel-MSE ≤ 1.0). §4 + Changelog Q4 lock invariant 9 shape (b). Steps 1–3 enforce invariant 11 (ADR before pin changes). Vision proof-hook line 100 (`uv lock --check exits 0`) now wired via F-1 fold-in at step 6.

2. **Vision invariant 12 — partially surfaced; not introduced as a new gap by the fold-ins.** Invariant 12 asks operators to document `~/.cache/uv` directory size growth and `.venv` directory size in the closure commit body. ADR §8 step 10 names "commit batch (3–5 commits, `feat(deps):` prefix)" but does not explicitly carry the disk-size-documentation requirement. This was already a gap in round 1 (not flagged because it's an operator/closure concern, not a supply-chain security finding) and the fold-ins did not introduce it. INFO-tier only; surfaces here for the record.

3. **Changelog fold-in entry is faithful to round 1.** The 2026-05-16 round-1 fold-in entry summarizes each finding's resolution in 1–2 sentences mapping back to the section that carries it. Re-fire instruction ("Re-firing `security-auditor` (Opus) round 2 on the amended ADR. Status flips to `accepted` if round 2 returns `CLEAN`.") is correct per global §12 / project CLAUDE.md "Review rules."

4. **No new control bypassed by the amendments.** The §8 ordering still respects `block-pyproject-floors.sh`, `check-ai-disclosure.sh`, and the `code-reviewer`-per-code-commit rule. The new `uv lock --check` gate is an additive control, not a relaxation of any existing one. F-2's "revisit if uv relaxes hash-verify default" is a future-watch item, not a present-day bypass.

5. **No drift in scope.** Edit scope per Vision Coordination notes is `pyproject.toml`, `requirements.txt`, `uv.lock`, ADR, security review, CLAUDE.md line 67. The amended ADR §8 does not authorize work outside that scope. No silent expansion into slice 1 / Hunyuan / LoRA-convert files.

6. **§5's "manifest is one trust hop from a Red Zone process" framing closes the round-1 framing gap.** Round 1 said the §5 reasoning was correct but elided the runtime channel; the amendment names it cleanly without overclaiming Red Zone status for the manifest itself.

### Remaining concerns

None at HIGH/MEDIUM. One INFO observation:

**F-7 INFO — Invariant 12 (operator disk-cost visibility) not surfaced in §8 step 10.** Slice-0c Vision invariant 12 asks the closure commit body to document `~/.cache/uv` growth and `.venv` size. §8 step 10 names commit cadence and `feat(deps):` prefix but does not name the disk-size disclosure. Smallest fix: a single-sentence note in step 10 ("closure commit body documents `~/.cache/uv` growth and `.venv` size per slice-0c Vision invariant 12"). This is optional; not a supply-chain security finding, and not in scope for a §11 review. Flagged for the record.

### Verdict line (round 2)

**CLEAN.** All six round-1 findings are faithfully folded. The F-1 gate is explicit on both pre-edit baseline and post-edit transitive-delta surfacing to Grant. F-2 names both the `uv sync` hash-verify default AND the runtime-consumer Red Zone channel. F-3 comment block warns on both failure modes. F-4 / F-5 / F-6 are present with the right rationale and template-copy guard. No new HIGH or MEDIUM gaps introduced by the amendments. The single INFO (F-7, invariant 12 disk-cost disclosure) is operator-visibility, not supply-chain integrity, and does not block status flip. ADR-013 is clear to flip Status from `proposed` to `accepted`.
