# Security review — ADR-044 commit 2 (daemon wire carriage)

**AI-Disclosure:** Claude (Fable 5) authored the review via the
`security-auditor` subagent; Claude (Opus 5) commissioned it and applied the
remediation; Grant reviewed.

**Date:** 2026-08-01
**Change under review:** ADR-044 commit 2 working-tree diff — `comfyless/params_validation.py`,
`comfyless/generate.py`, `comfyless/server.py`, plus three test suites.
**Red Zone surface:** `comfyless/server.py` (Unix-socket IPC daemon)
**Design ADR:** `docs/decisions/ADR-044-krea2-identity-daemon-carriage.md`
**Predecessor review (design stage):** `review-adr-044-identity-daemon-carriage-2026-08-01.md`

**Reviewer model provenance:** transcript records 40 tool uses across a session
whose assistant turns are all `claude-fable-5`. **No Fable→Opus fallback.**
A `code-reviewer` (Fable, also no fallback) reviewed the same diff in parallel
for correctness/test quality; its findings are folded in below where they touch
the same lines.

---

## Verdict

> The code honours every disposition the design review accepted. I confirm all
> five commissioned properties hold; the findings below are one observability
> gap in the new fail-open branch and several INFO-grade residuals, none of
> which I would block merge on given the accepted-design context — the MEDIUM
> should be fixed in this slice since it is one line inside new Red Zone code.

Threat model as stated: the Unix socket is a same-UID trust boundary (ADR-001)
whose validated request drives file reads, a shared ~30 GB pipeline cache, and
long-lived daemon state. The sharpest properties are (1) no request shape can
enter the identity dispatch without a literal opt-in, and (2) a failed identity
request cannot leave attention-processor residue serving subsequent requests.

## Coverage

Reviewed: ADR-044 and the design review in full; `comfyless/server.py` 141-230,
400-660, 650-1165; `comfyless/params_validation.py` 1-320, 424-520;
`comfyless/generate.py` 770-813, 1380-1435, 1661-1702, 1938-2100, 2240-2370,
2680-2700, 3310-3390, 3690-3750, 3990-4025, 4690-4760, 4835-4843;
`comfyless/refine.py` 1856-2000; `pipelines/krea2_identity_edit.py` (processor
class, apply/restore, `_cap_longest_side`, `_grounded_encode`);
`comfyless/ref_image.py` caps; the three touched test suites; `TECH_DEBT.md`.

Not reviewed, with the reviewer's reasons: the literal `git diff` (the agent has
no shell tool — it reviewed working-tree state against the ADR contract and read
every region the changes touch, but cannot mechanically rule out an unrelated
hunk in an unread area of those files); `comfyless/mcp_server.py` beyond greps
(commit 3 scope — greps confirm none of the three fields appear there yet);
diffusers `set_attn_processor` internals (third-party; the raises-or-completes
restore assumption is inherited from the design review, which scoped it the same
way); `test_params_schema.py` beyond greps.

---

## Findings

### [MEDIUM] Residue-check failure was completely silent — a fail-open backstop with zero observability

**Location:** `comfyless/server.py` (the residue check's `except Exception: pass`)

Any internal failure of the residue check — an `attn_processors` walk that
raises on an exotic or quantized transformer wrapper, or `_evict_chain` itself
raising mid-evict — left a possibly-corrupted pipeline cached and serving. That
is the exact silent-wrong-bias scenario the check exists to prevent, with no
trace that the backstop misfired. The swallow-to-protect-the-`InferenceError`
design is correct; the silence is not. The comment's justification ("an import
that fails means the identity path could not have run") covers the benign causes
but is asserted, not proven, for all causes — and this daemon's own precedent
(the `report_roots` residual, slice-1 review MEDIUM) is that an accepted residual
is tolerable only if OBSERVABLE.

**Remediation:** bind the exception and `_log` it before continuing.

**Disposition:** FIXED in this commit. The `except` now binds `_res_err` and logs
`identity residue check failed (...) — cached pipeline NOT verified`, citing the
`report_roots` precedent in the comment. A new negative test
(`test_server_robustness.py`) drives a transformer whose `attn_processors`
property raises and asserts both that the real `InferenceError` still returns and
that the swallowed failure is logged.

### [INFO] `grounding_px = 0` silently means "no VL downscale"

**Location:** `pipelines/krea2_identity_edit.py` (`if grounding_px and ...`),
bounded by `comfyless/ref_image.py` caps

A wire client sending `grounding_px: 0` (falsy) or an enormous int disables the
cap, feeding up-to-67 MP references (× the 2-source cap) into the Qwen3-VL
tower. Worst case is a CUDA OOM the daemon survives as `InferenceError` — not a
wedge. A negative value produces a 16×16 floor: wrong output, no crash. No value
reaches a filesystem path, and `_check_ref_paths` runs regardless of `identity`.
Same-UID boundary, availability-only, consistent with the warn-don't-block house
rule.

**Disposition:** ACCEPTED, no change. Recorded here rather than in TECH_DEBT
because the existing range-warning helper already covers the useful band; a
`grounding_px >= 0` value check can ride the finiteness slice if that lands.

### [INFO] The NaN / `Infinity` `ref_boost` deferral is safe — reviewer agrees

`json.loads` accepts `NaN`/`Infinity`; both pass `_KIND_FLOAT` and corrupt the
bias. But the corruption is confined to the requesting run (`_bias_cache` lives
on per-call processor objects removed with them), cannot crash the daemon
(converted to `InferenceError`), and cannot persist past the residue check.
Refine's own boundary already rejects non-finite floats (F6). Fixing it only at
the identity call site would leave the identical `nag_*` sites.

**Disposition:** deferral CONFIRMED sound; the family-wide TECH_DEBT entry
(2026-08-01) stands.

### [INFO] Interim MCP accept-and-drop window until commit 3 lands

Between commits 2 and 3, an MCP agent sending `identity`/`ref_boost`/
`grounding_px` gets them validated then silently dropped at the explicit-kwargs
call site. Not a regression — identical to pre-commit-2 behaviour for the
scalars, and `identity` was previously an unknown pass-through key. `identity`
cannot reach the daemon from MCP, so the drop is fail-closed.

**Disposition:** ACCEPTED. Reviewer's remediation adopted: **land and push
commits 2 and 3 as one batch so the window never exists on the remote.**

### [INFO] Test-file scope deviates from the ADR's declared commit-2 scope

The ADR declared `test_params_schema.py`; the diff instead put the validator
negatives in `test_machine_boundary_validator.py` (the more natural home) and
left `test_params_schema.py` untouched — whose Part-B `identity` allowlist entry
predates this diff and "stands unchanged" exactly as the ADR promises. Purely a
record-keeping mismatch, no security effect. The parallel `code-reviewer` raised
the same point as a low-severity boundary note.

**Disposition:** FIXED — ADR-044 Changelog records the substitution.

### [INFO] Legacy `--json` bridge still accept-and-drops the two scalars

`_run_json_mode` forwards none of the three, so a `--json` caller's `ref_boost`/
`grounding_px` are dropped before `generate()` and even the inert-tuning warning
never fires. Pre-existing since Part B, untouched by this diff, and structurally
harmless: `identity` cannot enter via `--json`, so no run on that surface can
consume the scalars anyway.

**Disposition:** ACCEPTED, no change. Folds into the mode's eventual retirement.

---

## Reviewer's answers to the commissioned questions

**1 — Fail-closed on every inbound path? YES.** Strict bool at the boundary
(`params_validation._check_field`): `null`, `0`, `"false"`, `"0"`, `[]` all
rejected as `invalid_type`, never coerced. Every socket request passes
`_validate_request` before dispatch. Daemon default False; `generate()` default
False; refine omits the field (`_daemon_namespace` has no `identity`); MCP and
`--json` never forward it; `--params` replay sources identity from `args`, never
`p_cur`. The dispatch gate downgrades `ref_kind` to the drop path on any falsy
identity, so nothing reaches `identity_edit_pipe_call` without a validated
literal `True`.

**2 — The residue check.** Sound, modulo the MEDIUM. It cannot mask the
`InferenceError` (the return is built after; all check exceptions swallowed),
cannot kill the accept loop by anything short of `BaseException`, evicts only on
a genuine `isinstance` match, and inspects the RIGHT object: the checked
`server_state["pipeline"]` is the identical object `generate()` ran on, identity
touches only its `transformer`, and a refiner cannot coexist with a krea family
(`_maybe_load_refiner` hard-rejects non-hunyuan). Partial `_evict_chain` failure
still drops the `"pipeline"` key first, so the next request reloads rather than
serving torn state. The suite's vacuity-guard premise test was called out
approvingly.

**3 — Cache-key exclusion.** Verified: none of the three appears in
`_request_cache_key` or in `_load_pipeline`'s arguments. The one indirect vector
hunted — `grounding_px` baked into the memoized VL processor — does not exist:
the memo is built without it and the cap is applied per call. The remaining break
vector is exactly processor residue, backstopped by commit 1 plus the error-path
check.

**4 — New compute/filesystem reach.** None. All three are scalars into
math/resize/dispatch; `_check_ref_paths` runs on `ref_images` presence
regardless of `identity`. `ref_boost` is clamped at `1e-4` before `log`;
`grounding_px` only ever shrinks; allocations are bounded by the 64 MB / 67 MP
loader caps × the 2-source identity cap. Worst hostile values cost one
recoverable `InferenceError`.

**5 — Deleting the client-side warning: verified safe.** Every identity notice
rides a wire channel — drop-warn, no-op, inert-tuning, no-LoRA, range +
ignored-knob, NAG-skip, rebalance-skip — all land in metadata, and
`surface_wire_warnings` prints them exactly once on delegated success. In-process
runs print directly and never call the surfacer, so there is no double-print in
either mode. The error-path loss is the documented residual with a live TECH_DEBT
entry.

**6 — Anything else block-worthy: no.** The reviewer flagged that the proof hooks
requiring a GPU (bit-identical delegated vs in-process run) are outside what
static review can discharge, and that **Grant's live smoke should cover that one
before the batch pushes.**

---

## Cross-review note: the two reviewers disagreed, and it mattered

The parallel `code-reviewer` raised a MEDIUM the security auditor did not: a
future author wrapping `remove_identity_processors` in `try/except` inside the
pipeline's `finally` — a plausible hardening edit, since an exception in a
`finally` masks the in-flight error — would turn a failed restore into a
**successful response with residue installed**, which the daemon's error-path
check never inspects. Every suite would stay green.

The security auditor, reasoning about the code as it stands, judged success-path
residue "not a real failure mode" — correct today, because the restore is a bare
call that does propagate.

Both are right in their own frame. Resolution taken: close it with a structural
test guard (an AST negative asserting the restore is not enclosed in a nested
`try`) rather than adding success-path residue checking to `server.py`. Adding
un-reviewed code to a Red Zone path *after* its security review is the thing to
avoid; the guard closes the named regression at zero Red Zone cost. Verified by
mutation: applying the exact refactor leaves the pre-existing guard legs passing
and trips only the new one.

---

## Disposition summary

| Severity | Finding | Disposition |
|---|---|---|
| MEDIUM | Residue-check failure silent (fail-open, unobservable) | FIXED — logs + negative test |
| INFO | `grounding_px = 0` disables the VL cap | Accepted, no change |
| INFO | NaN `ref_boost` | Deferral confirmed sound; TECH_DEBT stands |
| INFO | Interim MCP accept-and-drop window | Accepted — push commits 2+3 as one batch |
| INFO | Test-file scope substitution | FIXED — ADR Changelog |
| INFO | Legacy `--json` bridge drop | Accepted, no change |
| (cross) | Success-path residue after a hypothetical refactor | FIXED — AST guard, mutation-verified |

**No finding was rejected.** The `is True` gate hardening (raised by
`code-reviewer` into this lane) was also adopted: `server.py` now compares
identity by identity rather than truthiness, so it fails closed even if the
`_RUNTIME_KIND` registration is ever removed — the `report_roots` precedent.
