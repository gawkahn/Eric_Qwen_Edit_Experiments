# Security review — ADR-040 slice 1 (code): opt-in `report_roots` on `ping`

AI-Disclosure: `security-auditor` subagent (Claude Opus 5) authored the security
findings; `code-reviewer` subagent (Claude Opus 5) authored the correctness
findings; Claude (Opus 5) drove the session and folded both; Grant reviewed.

**Model note.** Both reviewers were invoked with an explicit `model: "opus"`
override, per Grant's 2026-07-27 ruling. Context: earlier the same day a
reviewer invoked with NO `model:` argument ran on `claude-fable-5` anyway — the
agent-file frontmatter pin, which CLAUDE.md §5A documents as silently ignored,
is functional in the current Claude Code build. Omitting the argument is
therefore no longer a way to avoid Fable; passing `model: "opus"` explicitly is.
§5A and the related memory were deliberately left uncorrected at Grant's
direction pending his own verification.

**Date:** 2026-07-27
**Subject:** `comfyless/server.py`, `comfyless/params_validation.py` (+ tests)
**Trigger:** Red Zone — `comfyless/server.py` is a `_red-zone-paths.sh` path
(Unix-socket IPC daemon).
**Design review this builds on:** `review-adr040-revision-2026-07-27.md`
**Status:** all findings folded; slice committed.

## What the slice does

Adds an opt-in `report_roots: bool` to the daemon's `ping` request. When true,
the ping response gains `output_dir` and `ref_image_roots`. The field is
registered in the canonical validator (`_RUNTIME_KIND` → `_KIND_BOOL`) so type
rejection happens in `params_validation.py` per ADR-012; `server._validate_request`
adds a presence-based value check making it a `ValidationError` on any non-`ping`
request type.

## Threat model

A same-UID local caller on a 0700-dir / 0600 Unix socket. The daemon has no
authentication and **cannot discriminate an MCP caller from a CLI caller** —
ADR-040 D2a records this as an accepted residual. The realistic adversaries are
therefore (a) a future HTTP/mcpo bridge that blindly forwards a health check,
and (b) a future in-repo maintainer wiring the MCP plane to the daemon.

**Verdict: no exploitable vulnerability in this diff.** The findings below are
one fail-open-on-refactor coupling, one missing detective control, and coverage
gaps in the D2a tripwire that become live in slices 2/3.

## Findings folded

### MEDIUM — the disclosure gate was truthiness, and failed OPEN on de-registration

`if req.get("report_roots"):` accepts any truthy value. The only thing making it
bool-only was the `_RUNTIME_KIND` entry in a *different module* — and
`validate_machine_request` passes **unknown keys through unchanged**, so
removing or renaming that registration would have silently converted the gate to
any-truthy-string. Not exploitable as written; a fail-open coupling on a Red Zone
disclosure gate.

**Closed:** changed to `is True`. Identity comparison, so ADR-012's
"no isinstance outside the canonical validator" and the N19 grep invariant both
stay intact, and disclosure now requires a literal JSON `true` regardless of
registration state. Pinned by a source-level test.

### MEDIUM — the accepted residual was unobservable

D2a accepts that a determined caller can still ask for roots. The only cheap
detective control for that residual is a log line, and there was none: the
disclosure branch logged nothing. Every other security-relevant event on this
surface logs (`PathError`, `RefPathError`).

**Closed (disclosure half):** a successful `report_roots` ping now logs a
**count-only, path-free** line — the roots are already in the startup banner and
the log should not become a second copy of them.

**Deferred (refusal half):** the generic `ValidationError` branch is also
traceless, but it is generic — logging there changes behavior for every
validation error on the surface, a volume/PII-shaped decision of its own (request
bodies carry prompts, which the `PathError` log redacts explicitly). Recorded in
TECH_DEBT.md 2026-07-27 rather than bundled in.

### MEDIUM — the D2a tripwire missed the route slices 2/3 will actually take

The tripwire greps `mcp_server.py`, but that file already imports from
`comfyless.generate` and `comfyless.server` and calls `generate()` in-process.
Per D3a the shared entry-check helper — the thing that *sends* `report_roots` —
lands in `generate.py` in slice 3. If the ping were issued from inside
`generate()`, the MCP process would obtain the daemon's roots while the tripwire
stayed green. Compounding it: D3's refusal message is specified to name the
`--ref-root` to add, so root paths would ride out through the MCP error frame to
an LLM caller.

**Closed:** added a check that `generate()`'s source never contains
`report_roots`, asserted now while it is still free.

### LOW — the premise check was weaker than its own name

Asserting absence of `socket_path` and `_send_server_command` does not prove
"holds no daemon socket-client code" — a hand-rolled client using
`socket.socket(socket.AF_UNIX, ...)` with inline framing passes both.

**Closed:** premise set extended to `_delegate_to_server` (the likeliest route,
flagged independently by the code reviewer) and `AF_UNIX`.

### LOW — two tautological tests, flagged independently by both reviewers

1. `isinstance(reported_roots, list)` **cannot fail**: the value came back
   through `json.loads`, and JSON serializes a tuple as an array regardless of
   what the handler put in `resp`. It claimed to pin `list(ref_roots)` and
   pinned nothing.
2. Feeding the *reported* roots back into `_check_ref_paths` **cannot fail**:
   the test helper hands back exactly what it was given.

**Closed:** both deleted. Replaced with the invariant that actually matters — a
*wiring* assertion that `run_server` passes the resolved `ref_image_roots` (not
the raw spawn tuple) into `_handle_connection`'s single `ref_roots` parameter,
which then serves both the report and `_check_ref_paths`. One parameter feeding
both is what makes divergence unrepresentable; divergence is what would let a
client pass D3 entry validation and still be refused mid-run.

### LOW — incomplete type-rejection coverage

`None` (the value a sloppy client most likely sends) and a list were untested.
**Closed:** both added.

### LOW — the CLI-plane rule was absent from the one file both planes read

`params_validation.py` is where a future author adding a wire field looks, and
the D2a restriction lived only in `server.py`, the ADR, and a test in another
suite. **Closed:** stated at the registration site.

### INFO — an ADR framing correction, recorded not fixed

D2's "the disclosure is a non-event" holds for `output_dir` (already inferable
from `output_path` in every successful generate response) but **not** for
`--ref-root` members: the containment error strings echo only the client's own
path, never a root, so the additions are disclosed by nothing today.
`report_roots` is a genuinely new disclosure for them. Deliberately accepted;
recorded so a future reader does not inherit the stronger "already leaked
anyway" framing.

## Verified clean

- Plain ping discloses nothing on **every** path — `ParseError`,
  `ValidationError`, and the `_send_safe` peer-gone swallow.
- The `ValidationError` text leaks nothing: `req_type` is constrained to the
  three-value allowlist before interpolation; no path, no attacker data.
- The value check cannot be bypassed by key casing, duplicate JSON keys
  (`json.loads` keeps the last), or the `req.update(result.payload)` that runs
  before it. It sits above the `if req_type != "generate": return None` early
  return, so it fires for `unload` and `ping`.
- `unload` + `report_roots` fails closed **without** stopping the daemon — the
  error returns before the shutdown branch.
- Type safety stays owned by the canonical validator (ADR-012); `isinstance(1, bool)`
  is `False` so int `1` is rejected. No isinstance added to `server.py` — the N19
  AST invariant is intact.
- Report/gate parity is **structural, not merely tested**: one `ref_roots`
  parameter feeds both the report and `_check_ref_paths` 31 lines later;
  `run_server` realpaths `output_dir` before `_resolve_ref_roots`, which
  realpaths each addition. A post-startup symlink swap resolves identically on
  both sides.
- No accept-loop kill hazard (`list()` and `json.dumps` over `str` cannot
  raise); no heavy import added to the validation path (F1 intact).
- No new write authority, no root widening, no ADR-037 D5 prohibited
  alternative (no per-request root extension, no root merging, no wire trust
  field).
- Schema placement is correct: `_RUNTIME_KIND` only, so `report_roots` never
  reaches `COMFYLESS_SCHEMA`, the sidecar, the `params_schema.py` drift guard,
  or the MCP tool schema. `_GENERATE_INPUT_SCHEMA` is `additionalProperties: False`
  and the handler builds explicit kwargs, so a recognized-but-unused key cannot
  ride in.
- No existing client sends `ping` at all, so the "no existing client path
  changed" claim is safe but vacuous.
- Edit scope is clean — two source files, two test files, zero slice-2 bleed, no
  drive-by cleanup, no dependency changes.

## Carried into slice 2

- **The value check is presence-based**, so `report_roots: false` on a
  `generate` request is a hard `ValidationError`, not a no-op. Slice 2's request
  builder must add the key only on ping requests, never unconditionally from a
  variable — otherwise it surfaces as a mid-run daemon rejection.
- **`report_roots` has no user-facing documentation yet.** D2a says it is
  "documented as a CLI-plane request field"; its only written form today is the
  ADR and the code. The vault `Comfyless_Manual.md` update is slice 2's, as
  planned.
