# Security review — ADR-040 slice 2b (code): derived run dir + entry validation

AI-Disclosure: `security-auditor` and `code-reviewer` subagents (both Claude
Opus 5, invoked with an explicit `model: "opus"` override per Grant's
2026-07-27 ruling) authored the findings; Claude (Opus 5) drove the session and
folded them; Grant reviewed.

**Date:** 2026-07-27
**Subject:** `comfyless/generate.py`, `comfyless/refine.py` (+ `test_refine.py`,
`test_mcp_server.py`)
**Trigger:** Red Zone — `comfyless/refine.py` is a `_red-zone-paths.sh` path
(LLM output influencing generation params; seed-image ingestion). This slice
additionally makes a value the DAEMON supplies over the wire the parent of an
`os.makedirs` and, transitively, of `pin_static_refs`' unconditional
`shutil.rmtree`.
**Chain:** `review-adr040-revision-2026-07-27.md` (design) →
`review-adr040-slice1-2026-07-27.md` (slice 1) →
`review-adr040-slice2a-2026-07-27.md` (slice 2a) → this.
**Status:** all findings folded or recorded; one MEDIUM deferred to its own
slice with a TECH_DEBT entry and an ADR Changelog note.

**Model-fallback verification** (per the standing check): both transcripts were
grepped for the model field after completion —
`grep -o '"model":"[^"]*"' <transcript> | sort | uniq -c` returned
`49 "model":"claude-opus-5"` and `52 "model":"claude-opus-5"` respectively.
No Fable turns, no silent fallback. The explicit `model: "opus"` override held
for every turn of both agents.

## What the slice does

`refine` asks the daemon for its roots with an opt-in
`{"type": "ping", "report_roots": true}` (the slice-1 wire field), derives its
run directory as `<daemon output_dir>/refine-<run_id>` when no `--output-dir`
was given, validates every path the daemon will read back at ENTRY through a
shared containment helper, stops sending an absolute savepath template (which
made the daemon mirror the whole path under its own root), and keeps a
same-directory `os.replace` to reconcile `_resolve_savepath`'s unconditional
counter suffix. `--output-dir` becomes optional; an explicit directory outside
the reported roots is an entry refusal, never a relocation.

New shared helpers in `generate.py` (slice 3 consumes the same ones for
`generate`'s one-shot path, per D3a): `DaemonRoots`, `_valid_reported_path`,
`query_daemon_roots`, `paths_outside_roots`.

## Threat model the auditor worked from

Three boundaries are touched:

(a) **Daemon → client over the wire.** The ping response is machine input that
becomes a filesystem path, the parent of `os.makedirs`, and transitively the
parent of `pin_static_refs`' unconditional `shutil.rmtree`. The daemon is
trusted-equivalent (same UID, same socket), so the realistic adversary is a
*buggy or version-skewed* daemon, not a hostile one — but ADR-012 discipline
says shape-check anyway.

(b) **Client → daemon.** The client-computed relative `savepath` is joined
under the daemon's root; the question is whether it can ever aim outside the
prefix the client validated.

(c) **Operator CLI → both planes.** `--output-dir`, `--seed-image`,
`--ref-image` are operator-typed paths, and D1 changes *where the loop puts the
operator's data* by default.

## Verified CLEAN

Recorded in full because "what was checked and found sound" is half of what an
auditability record is for.

- **Ping response as untrusted input — complete and fail-closed.**
  `_valid_reported_path` enforces `isinstance str` ∧ non-empty ∧
  `startswith("/")` ∧ NUL-free. `query_daemon_roots` requires `status == "ok"`,
  treats *both* fields missing as pre-D2 (silent `None`), and rejects the
  **whole report** on any single malformed member. There is **no partial
  trust**: every reject path is `return None`, handled identically to "no
  daemon". An empty `ref_image_roots` is rejected rather than
  accepted-as-empty — which matters, because `paths_outside_roots` with empty
  roots returns *everything* as outside; the two fail closed in the same
  direction. Nothing malformed can reach `makedirs` or `rmtree`. All wire text
  is `repr()`'d in logs.
- **Ordering — the exclusive create is genuinely first.** The auditor walked
  `main()` linearly: catalog build → judge backend/recipes → seed load (reads
  only) → config build (the run dir is not even computed yet) → `mint_run_id`
  → ping → derive → `paths_outside_roots` (pure `realpath`, non-mutating) →
  `os.makedirs(output_dir, mode=0o700, exist_ok=False)`. `pin_static_refs`,
  `refine_loop`'s `candidates/`/`winners/`, and `anchor/` are all strictly
  after. `exist_ok=False` creates the *leaf* exclusively, so a pre-placed
  directory **or symlink** raises `FileExistsError` → clean `return 2`. That is
  the property bounding the later `rmtree` to a directory this invocation
  provably created, and it holds.
- **Containment fidelity — no divergence.** `paths_outside_roots` defers to
  `server._within` by import rather than re-deriving it. Prefix-sibling
  (`/data/out` vs `/data/output`) and symlink-resolving-outside are both
  correctly outside, and tests pin both. `output_dir/anchor` is not in the
  checked list, but containment is transitive — it is created by `makedirs`
  inside an already-validated `output_dir` — so omitting it is not a gap.
- **The relative savepath cannot carry `..`.** The relpath is computed from the
  **realpath'd** values `_within` compared, so the numerator is by construction
  `== root` (→ `"."`, handled) or `startswith(root + os.sep)` (→ no leading
  `..`). `run_server` realpaths its own `output_dir` *before* reporting it, so
  `Path(daemon_output_dir) / rel` reconstructs `realpath(candidates_dir)`
  exactly. Even if a symlink swap broke that between the `realpath` calls, the
  daemon's own `_within(Path(expanded).parent, output_dir)` and the final
  `_within(output_path, output_dir)` reject it — two independent guards, both
  fail-closed.
- **D2a holds structurally.** `report_roots` appears as a literal *only* inside
  the `{"type": "ping", ...}` dict. `_build_server_request` has no
  `report_roots` key on any branch and never builds one from a variable, so the
  daemon's presence-based hard ValidationError cannot be tripped by a
  generate/unload request. `mcp_server.py` contains zero occurrences of
  `report_roots`, `socket_path`, or `_send_server_command` — no daemon
  wire-client code at all.
- **No new write authority or root widening.** Nothing in the diff sends a
  root, extends one, or asks for an exemption. `run_id` is used only as a
  dirname component and an equality-grouping field; nothing looks a run up by
  id and nothing gates on it. It acquires no capability semantics.
- **Mirror out, rename in — both halves right.** `_resolve_savepath` appends
  `{counter:04d}` unconditionally, so the daemon's returned name can never
  equal `canonical`; `_src != _dst` therefore always holds and `os.replace`
  always overwrites any stale canonical file. There is no path where a stale
  canonical survives to be judged as this iteration's candidate.
- **Scope.** The diff touches exactly the declared files; every hunk maps to
  D1, D1b's dirname consumption, D3, or D3a's shared helper. The
  function-scoped Red Zone surfaces `_run_json_mode` and `resolve_hf_path` are
  untouched. `--output-dir` losing `required=True` was checked against the rest
  of the repo — no caller or test depends on argparse rejecting its absence.

## Findings and disposition

### MEDIUM — the D2a tripwire stopped tripping (FIXED)

`test_mcp_server.py` D2a premise check. Slice 2b introduced
`query_daemon_roots` as a **named wrapper** that hides the `report_roots`
literal, so all three tripwire assertions stayed green if a future
`mcp_server.py` called it: the string-absence check (the literal lives in
`generate.py`), the premise symbol set (which did not list the new name), and
`inspect.getsource(generate)` (the helper is a sibling of `generate`, not
inside it). The slice-1 review explicitly anticipated this — *"D3a puts the
shared entry-check helper — the thing that SENDS report_roots — in
generate.py"* — but pinned the assertion to `generate()` because the helper did
not exist yet. It exists now and the tripwire was not moved with it. This was
the one place slice 2b materially reduced an existing guard.

**Fixed:** `query_daemon_roots` added to the premise symbol set, with a comment
stating that the set tracks the helper's *name*, not the wire literal — a new
wrapper needs a new entry or the tripwire silently stops tripping.

### MEDIUM — `--seed-image` is the one reference consumed on two channels with an unpinned path (DEFERRED, recorded)

`source_img = load_seed_image_capped(edit_source)` pins the judge's comparison
anchor to bytes at loop entry, while `current_source` — the same operator path
— is re-opened by the daemon on every generation until the first promotion. If
the seed file is replaced, re-encoded, or truncated mid-run (concurrent
session, editor save, sync client), generation conditions on the new bytes
while the judge scores identity against the old ones, silently breaking "scores
describe the generation's inputs". `pin_static_refs`' own docstring names
precisely this two-channel shape as the reason *every* `--ref-image` is pinned,
judge-marked or not. The seed image is the only reference left unpinned, and
the ADR's "Alternatives Rejected" claim that *"no new move step is needed"* is
true for `--ref-image` and false for `--seed-image`.

Note this is an **integrity** defect independent of ADR-040: it exists whether
or not a daemon is involved.

**Disposition:** deferred to its own slice — it is a behavior change to
ADR-037 D5's edit-source contract and belongs with an ADR amendment, not bolted
onto 2b. Recorded in TECH_DEBT.md and in the ADR-040 Changelog rather than left
as an implicit "the loop cannot". Both reviewers independently reached the same
recommendation.

### MEDIUM — D1's mirror-tree removal is scoped, and the third branch was untested (FIXED)

`daemon_output_dir` truthy **but** the run dir outside it is a real, reachable
production path: an explicit `--output-dir` inside a `--ref-root` that is not
under the daemon's `--output-dir`. It passes entry validation (ref roots ⊋
output dir), so the run proceeds — and still builds the absolute-path mirror
tree and still does the cross-tree `shutil.move`. The code cannot do better
(the daemon only writes under its own root), so this is scoping, not a defect —
but ADR-040 §Deferred says flatly *"D1 stops new ones being created"*, and the
branch had zero test coverage.

**Fixed:** a `_drive_daemon_savepath(foreign_root=True)` variant now pins all
three assertions (absolute template, mirror built, cross-tree move lands the
canonical name); a code comment at the branch states the scoping; and the ADR
Changelog narrows the claim.

### LOW — the containment refusal misattributed its cause in the derived case (FIXED)

The check deliberately runs on derived dirs too ("running it through the same
check is what proves that rather than assuming it"). But the message hardcoded
the explicit-`--output-dir` framing, so an operator who hit it in the derived
case — having already omitted the flag — was told to omit it, and pointed at
the very root the derived dir came from. Reachable only against a daemon
reporting an `output_dir` outside its own `ref_image_roots`: impossible for the
real server (`_resolve_ref_roots` always unions it in) and possible for a
version-skewed, buggy, or hostile responder — exactly the case the
untrusted-input doctrine defends. Fail-closed, so diagnostic quality only, but
it would have sent the operator toward adding a broad `--ref-root` to fix a
daemon bug.

**Fixed:** the message branches on `derived` and names the daemon-side
inconsistency as the cause. Pinned by test.

### LOW — the seed-image warning recommended the broadest possible root (FIXED)

The warning said `--ref-root {dirname(edit_source)}`. Seed images live in
`~/Pictures`, `~/Downloads`, or a project root; following that advice grants
any same-UID wire client read + VAE-encode over the entire tree for the
daemon's lifetime — the breadth ADR-035 Finding 6 warns about, which
`_resolve_ref_roots` warns about at runtime. The `--output-dir` refusal twelve
lines above gets this right and explicitly says "not a parent"; the seed
warning contradicted it, and a directory is the *coarsest* form of the
suggestion since `--ref-root` cannot name a single file.

**Fixed:** narrow-first ordering — "move or copy the seed under a reported
root" is now the lead fix, `--ref-root` is marked the broad fallback with the
breadth caveat carried over.

### LOW — derived run dirs held the operator's reference photos at umask-default modes (FIXED)

Under D1 the location is no longer operator-chosen. Every derived run leaves
`<daemon output_dir>/refine-<run_id>/refs/ref_NN.*` — verbatim copies of the
operator's private reference images — inside a directory that is by
construction a daemon ref root, with no retention bound. `makedirs` uses
`0o777 & ~umask` (typically 0755), so on a shared box the copies are
world-readable in a path the operator never named.

**Fixed:** `mode=0o700` on the DERIVED branch only. An explicit `--output-dir`
keeps default modes — the operator chose that path. Both branches pinned by
test. Retention/cleanup is left to the ADR's Deferred section beside the
mirror-tree cleanup item.

### LOW — the D1 dirname charset was normative but unenforced (FIXED)

D1 makes the charset normative (`[A-Za-z0-9_-]`, no separators, no operator
text interpolated unescaped). `RUN_DIR_STEM` is a literal, but `run_id` was
joined verbatim with nothing asserting its shape. Code-owned today, so not a
live hole — a normative invariant with no enforcement at the point that depends
on it.

**Fixed:** a charset test over `RUN_DIR_STEM` and 50 `mint_run_id()` draws,
referenced from a comment at the join site.

### INFO — `--output-dir ""` silently took the derive branch (FIXED)

Truthiness test, so `--output-dir "$UNSET_VAR"` fell through to derivation
instead of erroring. The derive branch is the safe one and the location is
loudly echoed, so impact was confined to surprise. Changed to `is not None`
with an explicit empty-string refusal; pinned by test.

### INFO — TOCTOU on the explicit-`--output-dir` branch only (ACCEPTED)

Between the realpath-based containment check and `os.makedirs(exist_ok=True)`,
a path could be replaced by a symlink resolving outside the roots; every
subsequent write — including `pin_static_refs`' `rmtree` — would then land
outside the validated prefix. Requires same-UID write access to the parent,
outside this project's threat model. The **derived branch is immune**
(`exist_ok=False` raises on a pre-existing symlink). Closing it would need
`O_DIRECTORY|O_NOFOLLOW` open-then-verify, disproportionate here. Recorded so
the residual is explicit rather than assumed absent.

### INFO — a malformed ping is indistinguishable from a pre-D2 daemon (ACCEPTED)

Both skip entry validation and both emit the loud NOTICE. This is the
ADR-sanctioned degrade and correct for the pre-D2 case; it does mean a daemon
returning garbage can suppress the client's entry check. Bounded: the client
then behaves exactly as before this slice (absolute savepath, ADR-037 D5 latch
as backstop), so no new capability is granted. Making the malformed branch exit
non-zero would break the pre-D2 compatibility D3 explicitly preserves.

### INFO — the client still moves whatever path the daemon returns (PRE-EXISTING)

`_src = os.path.abspath(resp["output_path"])` is taken on faith; a buggy daemon
returning an arbitrary path would have that file moved into the candidates dir.
Predates the diff, bounded by the daemon's own `_within(output_path,
output_dir)`. The slice *improves* this branch: `os.replace` on the
same-directory case is atomic and fails loudly if `_dst` is a directory,
whereas `shutil.move` would have silently moved the file *into* it. If slice 3
touches this path, a one-line client-side containment assertion on the relative
branch would close it.

## Test-quality findings (all fixed)

The code reviewer audited the new tests as tests, which caught one assertion of
mine that could not fail:

- **Unfailable:** *"a derived run never emits that warning (it cannot collide)"*
  — a freshly created dir cannot contain `refs/` regardless of whether the
  `not derived` gate exists; deleting the gate left it green. Replaced with the
  stronger property the gate actually implies: a derived dir that already
  exists FAILS on the exclusive create, so the warn-and-proceed path is
  unreachable there by construction.
- **Dead scaffolding:** `_q_roots(socket_exists=...)` patched `socket_path`,
  which `query_daemon_roots` never calls (removed); `_FakeGen.daemon_out_seen`
  was assigned and never asserted (removed).
- **Over-claiming label:** *"yields the realpath'd roots verbatim"* — realpath
  is the daemon's duty (D2), not this function's. Retitled so it cannot be read
  as coverage for something untested here.
- **Weak label:** the daemonless check asserted no notice but nothing about a
  ping actually being sent. `_run_main_2b` now captures every wire request and
  the check asserts the list is empty.
- **Stub drift, named:** `_drive_daemon_savepath` reimplements `server.py`'s
  join rather than driving it, so its assertions pin the *client's* template
  only. `server.py` is untouched by this slice, but a future change to its join
  would leave these green. Now stated in the helper's docstring.

## Reviewer disagreement worth recording

Both reviewers were asked the same open question — whether `--seed-image`
outside the roots should warn, refuse, or be pinned — and they split on the
middle option in a way that is informative:

- The **code reviewer** argued the warn is a *divergence D3a already ruled
  against for the sibling surface*: when slice 3 lands,
  `generate --ref-image /outside` will exit 2 while
  `refine --seed-image /outside` warns and proceeds — the "two different
  behaviors for one misconfiguration" D3a named. Its position: keep the warn
  for 2b, but the ADR Changelog entry recording the divergence is the one thing
  worth holding the commit on.
- The **security auditor** argued refusal would be the *worst* of the three:
  it blocks a configuration that works fine daemonless and on a roomy box, and
  its only escape is widening a ref root to a whole photo directory — the LOW
  above. Its position: the harm ADR-040 exists to prevent is mid-run discovery
  after model load, and the entry warning already closes that.

Both converged on pinning as the right long-term answer, and the auditor named
the reason that settles it: **the case for pinning does not rest on ADR-040 at
all** — it is the independent two-channel integrity defect above. Disposition
taken: keep the warn in 2b, record the divergence in the ADR Changelog, file
the pinning slice in TECH_DEBT.
