# Security review — parity slice 2: shared wire-warning surfacer (2026-07-25)

AI-Disclosure: security-auditor and code-reviewer subagents ran on Claude Fable 5
(`claude-fable-5`), both explicitly confirmed — no model fallback occurred.
Parent session (Fable 5) folded the findings; Grant owns the decision.

Scope: uncommitted diff extracting `generate.surface_wire_warnings()` (a
`_WIRE_WARNING_CHANNELS` table over nag/schedule/edit/lora) and calling it from
the CLI's `_delegate_to_server` plus refine's `run_generation` on both the
daemon and cold branches. `comfyless/refine.py` is a Red Zone path;
`generate.py` hosts function-scoped Red Zone surfaces (`_run_json_mode`,
`resolve_hf_path`) — both verified untouched.

Motivation: refine read ONLY `edit_warnings`, so a planner-added LoRA that
silently failed to apply was invisible to operator, loop, and judge — an
integrity problem in its own right (score drift with no attributable cause),
newly acute because the planner began proposing LoRAs the same day.

## The question that motivated the audit

The `lora_warnings` strings EMBED LoRA PATHS —
`generate.lora_failure_warnings` documents them as "operator-facing only;
they are dropped from the agent surface (ADR-015 MEDIUM-1)". This slice pipes
them into a Red Zone file whose F3 invariant is that paths never reach
LLM-visible surfaces. **Auditor verdict: PASS.** Both new call sites emit
solely through `run_generation`'s `log` callable, whose only production
binding is `print` (refine.py `log = print`). No LLM-visible surface reads
them or their metadata: `build_judge_user_text` takes prompt / name+weight
LoRA views / path-stripped offers / history and never `GenOutcome.metadata`;
`history_record` is path-free by construction; `verdict_record` uses only
verdict fields and stays `_assert_no_paths`-gated. `Candidate.metadata` is
stored but never read into any of those. Net: exactly one new sink, the
operator's stdout.

## Disposition summary (folded same day)

| Finding | Severity | Disposition |
|---|---|---|
| Daemon-supplied strings reach the operator TTY unsanitized; refine's exposure widened from one channel to four (ANSI/OSC injection by a divergent same-UID daemon — defense-in-depth, not a privilege boundary) | LOW | **Fixed** — `_sanitize_wire_warning` strips non-printable characters at the shared choke point the refactor created (covers all three call sites). |
| No per-channel cap: ~1 MiB of warning text per response × 100 iterations buries the score/PASS/ABORT lines (attention-DoS) | LOW | **Fixed** — 20 items × 500 chars per channel, with an explicit "... N more suppressed" line. |
| `_FORBIDDEN_CONTEXT_KEYS` gates KEYS, so `lora_warnings` values (path-bearing) would not trip the F3 backstop if a future slice passed metadata into a judge payload | INFO | **Fixed** — the four wire-warning keys added to the list. |
| Gen-metadata sidecar has always persisted path-bearing `lora_warnings` to disk | INFO | Status quo, unaffected — operator artifact, never re-read into configs, never judge-visible. |
| Returned LoRA count has no consumer: the planner can re-propose a LoRA its own prior iteration failed to apply | NIT (code review) | **Deferred with a named trigger** — TECH_DEBT 2026-07-25; loop/judge accounting changes decision-making on a Red Zone file and interacts with `--pin-lora` + the v3 gate. |
| My replacement `test_nag.py` N1 pin matched the surfacer's DEF line — deleting the CLI call site would have left it green (strict weakening of the pin it replaced) | SHOULD (code review) | **Fixed** — re-pinned on the stderr emit literal. |
| Cold-path duplicate emission (generate prints to stderr at origin; loop logs to stdout) | INFO | Accepted — different sinks; the loop log is the operator's record when output is captured. |

## Other verdicts (condensed)

**Emission bound:** the client reads daemon responses through `server._recv`,
which rejects frames over `_MAX_FRAME_BYTES` (1 MiB) — so the pre-fold worst
case was bounded but noisy; the per-channel cap now bounds it at the display
layer too.

**Scope:** `_run_json_mode` never reaches `_delegate_to_server`;
`resolve_hf_path` absent from the diff; `mcp_server.py` untouched and still
pops `lora_warnings` from the agent surface with its own nag surfacing — what
MCP sees is identical.

**Behavior preservation (code review):** generate's emitted lines are
byte-identical (same prefixes, same stderr stream, same nag→schedule→edit
order); `_report_lora_outcome` receives the same unmutated dict and returns
the same RC; no double emission.

Test state at fold: test_refine.py 422 passed, test_nag.py 106 passed, full
battery 29/29.
