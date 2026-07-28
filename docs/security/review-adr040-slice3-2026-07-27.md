# Security + code review — ADR-040 slice 3 (D3a), `generate` one-shot entry gate

**Date:** 2026-07-27
**Scope:** `comfyless/generate.py` (`refuse_out_of_roots_refs` + its call site in
`_run_cli_mode`, `_should_delegate_to_server` docstring, `--ref-root` help text),
`test_ref_edit.py`, `test_mcp_server.py`
**AI-Disclosure:** Claude (Opus 5) authored the slice; `code-reviewer` and
`security-auditor` both run on Opus 5 and verified as such by transcript grep
(53 / 51 turns, `"model":"claude-opus-5"` on every turn, no Fable fallback);
Grant reviewed.

## Why this slice had a review at all

`comfyless/generate.py` is NOT in `scripts/git-policy/_red-zone-paths.sh`, and
the function-scoped Red Zone surfaces in it (`_run_json_mode`,
`resolve_hf_path`) are untouched — so no hook required this. It was run anyway
because the slice consumes daemon wire input at a containment decision and adds
a new path-disclosure sink, which is the §12 trigger regardless of which file it
lands in. The code reviewer independently flagged the missing record as a
finding; this document closes it.

## What the slice does

`generate`'s one-shot CLI path pings the daemon for its roots (`report_roots`,
ADR-040 D2) and refuses an out-of-roots `--ref-image` at ENTRY, instead of
letting `_delegate_to_server` catch the daemon's `RefPathError` and fall back
in-process — a fallback that is fatal, not a degrade, against a warm daemon
still holding its pipeline's VRAM (the 2026-07-26 incident). Containment is not
re-derived: it consumes the same `query_daemon_roots` + `paths_outside_roots`
helpers slice 2b built for the refine loop, per D3a's "implement once, consume
twice".

## Verified clean (recorded, since what was checked and found sound is half the record)

- **The client check grants nothing and cannot be mistaken for the boundary.**
  `server._check_ref_paths` is untouched; nothing in the diff sends a root,
  extends one, or requests an exemption; the client's decision never reaches the
  wire. Every failure direction is a FALSE NEGATIVE (client passes, daemon still
  refuses), degrading exactly to pre-slice behavior — never a grant. The auditor
  enumerated and confirmed this for relative paths, `..`, trailing slashes,
  symlinks, prefix-siblings, colon-in-filename MODE parsing, non-existent paths,
  and empty root sets.
- **The compared string IS the sent string.** `os.path.abspath` at the gate is
  byte-identical to `_build_server_request`'s `_abspath`, empty-string guard
  included (that guard was added in response to the auditor's INFO — it was the
  one divergence found).
- **Ping response is only compared and `repr()`'d.** No join, `makedirs`,
  `open`, or `rmtree` on any reported value — a real narrowing versus slice 2b,
  where those values became a `makedirs` parent. `repr()` escapes C0 controls
  and Cf-category characters, so a hostile responder cannot drive terminal
  escapes or spoofed text through the refusal message.
- **Scope gate is correct and cheap-first:** specs → delegation predicate →
  socket → ping. An `--output` run neither refuses nor pings; a daemonless run
  is untouched.
- **Placement is genuinely pre-load.** The gate runs after
  `_apply_replay_ref_trust` (so it sees the refs actually headed for the wire)
  and before `_confirm_iteration`, `mint_run_id`, every `resolve_hf_path`, and
  every model load. Nothing is written and no GPU is touched before the refusal.
- **D2a holds.** The only wire traffic is the literal
  `{"type": "ping", "report_roots": True}` inside `query_daemon_roots`, never
  through `_build_server_request`, so the daemon's presence-based hard
  `ValidationError` cannot be tripped. `refuse_out_of_roots_refs` has exactly one
  caller and is unreachable from `_run_json_mode` and `mcp_server.py`.
- **`--iterate` invariance.** `_ITERATE_SHAPES` has no `ref_image` axis and
  `device` is not an iterable axis, so the reference set provably cannot vary
  across a sweep — one check per invocation is sound, not an optimization.

## Findings and disposition

### MEDIUM — the refusal's `--output` escape was inoperative whenever `--savepath` was set — FIXED

Both reviewers found this independently. `_should_delegate_to_server` is
`bool(args.savepath) or using_default_output`, and `--output` / `--savepath` are
independent flags, so on a `--savepath` run ADDING `--output` does not skip
delegation — the gate fires again with the identical message. Since `--savepath`
is the documented way to name output with a daemon running (`--output`'s own
help says so), that is the dominant configuration in which the refusal fires.
The operator follows the advice, dead-ends, and the only remaining offer is
`--ref-root <whole directory>` — precisely the ADR-035 Finding 6 breadth grant
the message's narrowest-first ordering exists to discourage. This inverts the
warn-don't-block intent.

**Fixed:** the escape clause branches on `args.savepath` and names the flag that
must GO ("replace --savepath with --output <path>"). Both branches are now
tested by DRIVING the escape to `rc is None`, not by grepping the message for
the flag name — which was the reviewer's second point about the original test.

### LOW — the pre-D2 daemon case silently disabled the gate — FIXED

`query_daemon_roots` returns `None` silently for a daemon predating D2, and the
caller returned `None` with no output: the operator got zero signal that entry
validation never ran, then discovered it via the OOM path this slice exists to
close. D3 makes the notice normative for the sibling surface and `refine` emits
it. **Fixed:** a NOTICE mirroring refine's text.

### LOW — the refusal asserted `output_dir` was a ref root without checking — FIXED

"move or copy the reference under a reported root (`<output_dir>` is one)"
treated one wire value as a member of another wire-supplied list. True for the
real server (`_resolve_ref_roots` unions it in), false for a skewed or hostile
responder — and this is the same misattribution slice 2b already paid for on the
refine side. **Fixed:** the destination is taken from the validated
`ref_image_roots` unless `output_dir` is demonstrably a member.

### LOW — the suggested `--ref-root` was lexical, so it would not fix a symlinked reference — FIXED

The daemon realpaths both sides (`_within`, `_resolve_ref_roots`), so granting
`dirname(abspath(ref))` does not contain a symlinked target and the daemon still
refuses — after the operator has restarted a 20B-parameter daemon, the most
expensive action the message suggests. Only `outside[0]` was named, so two
out-of-roots directories meant two restarts. **Fixed:** `dirname(realpath(...))`,
deduplicated across every offending path.

### LOW — the `except ValueError` fail-open was reachable from the untrusted `--params` channel — FIXED

The comment claimed unreachability because `main()` validates the specs. It
does — but `_apply_replay_ref_trust` REWRITES `args.ref_image` from a sidecar
after that validation and before this gate, so on a replay run the gate parses a
set `main()` never saw. A sidecar naming a missing file is kept (with a
`** MISSING **` warning) and re-injected; if a colon-named file exists in that
directory the colon-disambiguation branch raises, the gate swallowed it, and the
containment check was silently skipped for the whole run. Bounded (the same call
raises uncaught further down, so the run died on a traceback rather than
reaching the fallback) — but an untrusted metadata channel disabling a
containment gate is the wrong shape regardless. **Fixed:** fail CLOSED — print
the named error, exit 2. The comment now states the real reachability.

### LOW — new unannounced blocking round-trip to a serial daemon — FIXED

The daemon's accept loop is strictly serial, so the entry ping queues behind any
in-flight generation, with a 600 s client deadline — up to ten minutes of silent
hang in an environment where concurrent sessions against one daemon are normal.
It fails open correctly (timeout → `None` → proceed), so no legitimate run is
refused; the cost was an unattributable stall. **Fixed:** a line naming what is
being waited on, printed before the ping.

### PROMISE DRIFT — three operator-facing statements contradicted the code — FIXED

`_should_delegate_to_server`'s docstring ("There is NO client-side ref-root
gate: the CLI cannot know the daemon's --output-dir") sat directly above the
function that is exactly that, asserting the false premise ADR-040 was written
to retire. Same for the `--ref-root` `--help` text, which is where the refusal
message sends the operator, and the vault manual. **Fixed** in all three, each
scoped to "against a daemon that reports roots this is an entry refusal; a
pre-D2 daemon keeps the fallback". The new function's docstring also now states
in terms that `server._check_ref_paths` remains the authoritative gate and this
check may never be relied on as the boundary.

### PROMISE DRIFT — the call site was unpinned — FIXED

Every D3a test drove the helper directly, so deleting the two-line call site
left all of them green — while "refused at ENTRY" is a claim about placement.
**Fixed:** source-pinned checks over `inspect.getsource(cg._run_cli_mode)` for
the call, for the exact default-output literal it passes (a divergence there
silently breaks the `--output` scope gate), and for its position before
`_confirm_iteration` and `def _run_one`.

## Accepted residuals

- **Stale-ping window.** The check runs once at entry; a daemon restarted with
  narrower roots between the ping and the request re-opens the fallback path.
  ADR-sanctioned (D3 names it), with the ADR-037 D5 latch as backstop.
  Per-iteration pinging was rejected as noise.
- **The gate refuses even when the daemon holds no VRAM** (socket up, pipeline
  evicted) — a case where the fallback would have succeeded. D3a rules on this
  explicitly ("refusal replaces a fallback that genuinely works on a box with
  room to spare"), so it is a decision, not a defect.
- **`realpath` on reported roots is uncapped and untimed.** `_valid_reported_path`
  imposes no length or count cap, so within the 1 MiB frame a hostile responder
  can force ~20k `realpath` calls, and a root pointing into a dead FUSE mount
  blocks with no timeout (relevant: `/home/gawkahn/projects` is mergerfs). Both
  require a hostile same-UID daemon, outside the threat model.
- **A NUL in a typed `--ref-image` would escape as an uncaught `ValueError`**
  from `realpath` (which catches `OSError` only). Unreachable from a shell —
  `execve` argv cannot carry NUL — and the replay channel NUL-checks at
  `_gate_file_derived_refs`. Not closed; recorded.
- **The D2a tripwire is still name-enumeration**, and this slice makes the chain
  one wrapper longer (`refuse_out_of_roots_refs` → `query_daemon_roots` →
  literal). Adding the name is the right minimal move and was done; it does not
  make the guard structural. The auditor's suggestion for when this is next
  touched: pin the allowlisted set of symbols `mcp_server.py` imports from
  `comfyless.generate`, so any NEW import fails regardless of its name.

## Verdict

No CRITICAL or HIGH. No boundary violation, no new authority, no fail-open at
the daemon boundary. One MEDIUM (operator dead-end in the dominant
configuration) and six LOWs, all fixed in-slice; the remainder are recorded
residuals with named triggers.
