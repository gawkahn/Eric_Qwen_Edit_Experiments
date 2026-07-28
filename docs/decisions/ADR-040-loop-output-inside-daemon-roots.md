# ADR-040 — The loop's output directory lives inside the daemon's roots

Status:   accepted
AI-Disclosure: Claude (Fable 5) authored; revised by Claude (Opus 5); Grant reviewed.

## Context

Live finding, 2026-07-26, on the first end-to-end ADR-039 run. A refine edit run
with `--output-dir` outside the daemon's roots fails, and the failure mode is
worse than it looks:

```
[refine] daemon refused a reference path ("ref_images[1].path outside the
ref-image roots: '<repo>/07-26-26/hairstyle-v3-smoke/refs/ref_00.jpg'")
— running the REST of the run in-process.
...
torch.OutOfMemoryError: CUDA out of memory. Process <daemon> has 67.46 GiB in use.
```

Three facts combine:

1. **The daemon's ref allowlist is `{its --output-dir} ∪ --ref-root`**
   (`server.py` `_build_ref_roots`), disjoint from the weight-path allowlist.
2. **The loop writes files the daemon must READ BACK, under the loop's own
   `--output-dir`:** `refs/` (ADR-038 D5 pinned static refs) and
   `candidates/candidate_NN.png` — which in edit mode is ref 0 of the next
   iteration. Only the *generated* image moves out of the daemon's tree
   (`run_generation` re-roots the savepath template, then moves the result to
   `output_dir/stem<ext>`); the read-back files never move IN.
3. **The designed degrade does not hold on a warm single-GPU box.** ADR-037 D5's
   first-refusal latch drops to in-process for the rest of the run, but the
   daemon is still holding ~67 GiB, so the in-process load OOMs. The
   "recoverable" path is fatal in the configuration the daemon exists to serve.

So **every** refine edit run whose `--output-dir` is not inside a daemon root is
broken by construction, and the operator learns it mid-run, after model load and
the first generation. ADR-038's own text names this: *"a mid-RUN refusal is the
2026-07-24 incident we are avoiding."*

Grant's framing, which is the decision: *"The whole output-dir/savepath
distinction is supposed to be gone. The output dir should be in the savepath,
which is in the ref root by definition if there's a daemon running."*

The reason it was not caught earlier: ADR-037 D5 justified the latch on the
premise that *"the client cannot know its roots."* That premise is an artifact
of the wire protocol, not a fact — the daemon simply exposes no way to ask
(`ping` / `unload` / `generate` are the only request types; `status` is a
response field).

## Decision

### D1 — When a daemon serves the run, the loop's output directory is derived under the daemon's output root

`refine` resolves its run directory to `<daemon output_dir>/<run stem>-<run_id>`
when a daemon is reachable on the target device. `refs/` and `candidates/` are
then inside a ref root **by definition**, and the class of failure above becomes
unrepresentable rather than diagnosed.

**D1 also kills the path-mirror tree, which is the operator-visible half of the
bug** (Grant, 2026-07-27: he had complained about this behavior directly, and
separately wondered where the `…0001`-suffixed images the daemon reports were
going). `run_generation` passes refine's ABSOLUTE run path as `savepath`
(`refine.py:1908`), but the daemon treats `savepath` as a RELATIVE template —
`savepath.lstrip("/")`, then joined under its own root (`server.py:866-869`).
The daemon therefore writes a **full mirror of the absolute path underneath its
own output dir**, and refine moves the file back out afterwards. Measured on
2026-07-27: 31 directories under `$DAEMON_OUT/home/gawkahn/…`, holding 2 orphaned
candidate PNGs (2.9 MB) from runs that died before the move.

**The `…0001` naming is a separate mechanism, and conflating the two was an
error in this ADR's first revision** (caught by the 2026-07-27 security
re-review). `_resolve_savepath` (`generate.py:522-524`) appends the counter
**unconditionally** — it starts at `counter = 1` and returns
`f"{stem}{counter:04d}{extension}"` — so the daemon's FIRST file for stem
`candidate_00` is always `candidate_000001.png`. It is the first slot, not a
fired collision (a collision would be `…0002`). This resolves Grant's standing
side question about where those files were going: the daemon's log names
`candidate_000001.png` and the run dir shows `candidate_00.png`, and they are
the same bytes — `shutil.move` (`refine.py:1953`) reconciles the two namespaces.

**Therefore D1 removes the MIRROR, not the rename.** Making the savepath
relative stops the shadow tree and the cross-filesystem-tree move, but the
daemon still applies its counter suffix, so the canonical `output_dir/stem<ext>`
name that ADR-034 D7, the stale-other-extension warning
(`refine.py:1897-1905`), and the candidate-as-next-edit-source contract all
depend on is still not what the daemon writes. A **same-directory
`os.replace(daemon_name, canonical)` remains** — atomic, no longer crossing
trees, and ADR-034 D7 is untouched.

Deleting the rename outright was considered and rejected: it would force refine
to adopt counter-suffixed names everywhere (an ADR-034 D7 amendment with a
consumer list), and a same-stem regeneration inside one run would then yield
`…0002`, breaking the `candidate_NN` ↔ file mapping. Worse, any implementer who
dropped the rename while leaving a canonical-name assumption in place would let
a stale canonical file in a reused directory be judged and pinned as the new
candidate — the judge-vs-generation divergence ADR-038 D5 closed. The rename is
load-bearing; only its cross-tree scope was accidental.

**The run directory is uniquely named at session start, not deduplicated on
collision** (Grant's ruling, superseding the first draft's suffix scheme —
*"if I were building serious multiuser stuff here I'd be wanting truly unique
dirnames for every run, set at session start — like UUID-unique, not just
'check to see if it exists and append a string if it does'"*). `run_id` is
minted once per invocation (D1b) and appended to the stem, so collision is
structurally impossible rather than handled.

This subsumes the design review's HIGH finding rather than answering it with a
retry loop. That finding: D1 would make the run dir a deterministic function of
daemon state plus a stem, and this environment runs concurrent sessions against
a daemon built to serve them; two runs deriving the same stem would SHARE a run
dir, and `pin_static_refs` opens with an unconditional `shutil.rmtree(refs_dir)`
(`refine.py:3155`), so run B would delete and repin `refs/` while run A is
mid-iteration — run A then either dies on a missing `ref_00.jpg` or silently
reads run B's reference bytes, reopening the judge-vs-generation divergence
ADR-038 D5 closed. It would also destroy a COMPLETED prior run's pinned
provenance whenever a stem repeated. With a per-invocation `run_id` in the
name, no two runs can address the same directory in the first place.

`os.makedirs(..., exist_ok=False)` is retained, but demoted to an **assertion**
— it now catches only a `run_id` collision (astronomically unlikely) or a
logic error, and fails the run rather than suffixing. The stem is
`[A-Za-z0-9_-]` only — no path separators, no operator text interpolated
unescaped.

**The exclusive create MUST be the first filesystem operation on the run dir**
(re-review MEDIUM), before `pin_static_refs`, before `candidates/`, and before
any write derived from a wire value. Two things depend on the ordering, and
both fail silently if it slips. First, `pin_static_refs` calls
`os.makedirs(refs_dir, exist_ok=True)` (`refine.py:3156`), which **creates
parent directories** — if pinning runs first it materializes the run dir itself
and the assertion can never fire, restoring silent sharing. Second, the run dir
is derived from a value the DAEMON supplied over the wire (the D2 `output_dir`);
a fresh exclusive create is what bounds every subsequent client write — and
`pin_static_refs`' `rmtree` — to a directory this invocation provably just
made, even against a buggy or malicious ping response. The ordering is a named
test, not a code comment.

**Uniqueness is guaranteed for DERIVED run dirs only** (re-review MEDIUM).
Derivation applies only when no `--output-dir` was given (D3), so two concurrent
runs handed the *same explicit* `--output-dir` still share it, and
`pin_static_refs`' unconditional `shutil.rmtree(refs_dir)` still destroys the
live sibling's pinned refs. That is the prior review's HIGH surviving on the
explicit path, and this ADR does not close it — an operator who names the
directory owns its concurrency. What this ADR does add, per the warn-don't-block
preference: when an explicit `--output-dir` already contains a `refs/` from
another run, the loop warns loudly at entry and proceeds.

### D1b — `run_id` is the correlation primitive, and `iterate_batch_id` derives from the same minting helper

A `run_id` is minted **once per invocation, at process start**, in every
entrypoint (`generate`, `refine`, `cascade`), and stamped into every record the
run writes: the load-plane sidecar, the path-free `*.verdict.json`, and the
duel-arm records. It appears verbatim in the D1 run dirname, so the directory an
operator is looking at names the ID that ties its contents together.

This closes a real gap: `iterate_batch_id` (ADR-008) is minted only when
`--iterate` is given (`generate.py:3894`), so **every refine run today produces
candidate sidecars with no correlation key at all**.

**`run_id` and `iterate_batch_id` stay DISTINCT fields sharing one minting
helper — they are not aliased** (Grant raised the case that decides this: a
future judge run that itself drives `--batch`/`--iterate`). `run_id` identifies
the invocation; `iterate_batch_id` identifies a sweep *within* it. Nested, the
outer identity survives and both group correctly; aliased, the nesting case
loses one of them. Code reuse is the shared helper; correlation is `run_id`
being present on every record unconditionally. Collapsing them is a one-line
change later if nesting is ruled out.

**Format: 8 hex characters** (`uuid.uuid4().hex[:8]`), from one helper both
fields call. Short enough to live in a dirname an operator types and to read off
a log line. That is 32 bits of real randomness (uuid4's version nibble sits
outside `hex[:8]`), so a birthday collision needs ~77k runs sharing one stem
before it is even 50/50. 8 hex is adequate as the run-dir collision defense
**only because** D1's `exist_ok=False` assertion fails loudly on the residual
case — which is why that assertion's ordering is normative above rather than
best-effort.

This also resolves a pre-existing inconsistency: `iterate_batch_id` is minted as
a 36-char dashed UUID in `generate.py:3894` but a 32-char hex in
`cascade.py:895`. Both move to the helper. Nothing parses these values — they
are compared for equality to group records — so shortening is safe, and
historical sidecars carrying longer IDs never compare against new ones in a way
that matters.

**No wire change is required, and this is why the correlation work is not Red
Zone.** Both clients write their sidecars themselves after the daemon returns —
`generate.py:3303` patches the response metadata and dumps `{stem}.json`;
`refine.py:2459` does the same via `_write_json`. `iterate_batch_id` already
rides this path with an explicit "stamped client-side so downstream grouping
works without requiring a server change" note. `run_id` follows the identical
route: `server.py` is untouched.

Two registration duties, both mandatory or the field becomes noise:

- **`_SKIP_SIDECAR_KEYS` (`generate.py:120`)** gains `run_id` AND
  `iterate_batch_id`. The latter is a **pre-existing wart this slice closes**:
  it is absent today, so a `--params` replay of any `--iterate` sidecar already
  prints `schema: dropping unknown key 'iterate_batch_id'`. Non-schema
  provenance keys belong in that set — they are "known-and-intentional
  non-params", exactly its stated purpose.
- **The MCP `extract_params` drop-list** gains `run_id`, beside
  `iterate_batch_id` (`docs/vision/slice-4-mcp-extract-params.md` item 20).
  Runtime provenance must not surface in planner- or agent-facing params. The
  re-review verified this is **already structural** rather than list-dependent:
  the non-cascade path normalizes through `_validate_params` (which drops every
  non-schema key) and the cascade path renders from raw through an allowlist, so
  `run_id` could not survive `extract_params` even with no change. The list
  entry is therefore documentation of an existing guarantee — worth writing
  down, but it is not the thing keeping the key out.

`run_id` is path-free, so `_assert_no_paths` continues to hold over the verdict
records unchanged.

An explicit `--output-dir` remains an override for the daemonless case and for
operators who know what they are doing; when one is given AND a daemon is
serving the run, it is validated (D3) rather than silently trusted. The derived
path is LOUDLY ECHOED at entry (the existing F4 precedent), so the operator
always sees where pinned copies of their reference photos landed — under D1
that location is implicit where it used to be their choice.

### D2 — `ping` reports the daemon's output dir and ref roots

The smallest wire change that removes the false premise: the `ping` response
gains `output_dir` and `ref_image_roots`. Read-only, additive, no new request
type, no behavioral coupling — a client that ignores the new fields is
unaffected.

**The fields are OPT-IN, not the default ping response** (design review
MEDIUM). Request shape: `{"type": "ping", "report_roots": true}`. For today's
same-UID socket the disclosure is a non-event — generate responses already
return resolved output paths and error strings echo paths. The hazard is
that `ping` is precisely the request a future HTTP/mcpo bridge forwards as a
health check, which would make root enumeration — including a broad
`--ref-root` like `$HOME` — the default answer to the cheapest unauthenticated
call, and would hand the deferred D4 review a leak-by-default primitive it has
to remember to strip. An explicit, schema-validated flag means a blindly
forwarded plain ping discloses nothing.

A boolean "is this path inside a root?" op was considered and rejected: D1 needs
the actual value to derive under, and a boolean oracle is binary-searchable
anyway, so it buys nothing.

**The reported values are the REALPATH'D ones the gate actually compares
against**, not the spawn-time strings.

#### D2a — `report_roots` is CLI-plane only, and the MCP server is barred from it by test

Grant, 2026-07-27: *"Leaking real paths to an unauthenticated ping is exactly
what I was talking about when I said this was challenging for the MCP case."*
He ruled that this is handled **now**, sequenced into this ADR, not deferred
behind an integration of unknown timing.

The honest statement of the problem: **the daemon cannot discriminate an MCP
call from a CLI call.** Same UID, same socket, same request schema
(`server.py:418`). No authentication framework is being built here, and the
opt-in flag alone does not settle it — it stops a *blindly forwarded* health
check, but not a bridge that forwards the flag too.

What actually holds today is narrower and testable: **the MCP server has no
reason to ever ask.** D2 exists for the refine loop, a CLI-plane client.
ADR-015 already means the MCP surface trades in catalog NAMES, not paths, and
D4 extends that to reference images as handles. So:

- `report_roots` is documented as a **CLI-plane request field**. `refine` and
  `generate` may set it; `mcp_server.py` may not.
- `report_roots` is **schema-validated as a bool and honored ONLY on
  `type: "ping"`** (re-review LOW). On any other request type it is a
  `ValidationError`. Without this the flag is a free-floating field a future
  request type could inherit by accident.
- A **negative test asserts `mcp_server.py` never emits `report_roots`.**

**That test is a regression tripwire, not a control, and the difference
matters.** The daemon will answer *any* same-UID caller that sets the flag; the
test asserts the absence of a string in a file that today contains no
daemon-socket client code at all (no `socket_path`, no `_send_server_command` —
the MCP server reaches generation in-process). So it cannot stop a leak, it can
only catch the day someone adds wire-client code to that file without thinking
about roots. Today's real exposure through the mcpo/OWUI chain is nil for the
same reason.

This is a **residual risk, recorded as such, not a mitigation that closes the
hole.** A future caller that does forward the flag re-opens it. The bound on the
damage is that root enumeration is the whole disclosure — no write authority is
granted, and D1 removes the loop's need for the operator to name paths at all.

#### D2b — the intended structural fix is the `la mcp serve` boundary

Recorded explicitly because **this ADR is one of the documents the integration
will read**, and a residual risk that is only implied will not survive the
handoff.

The `local_agents` project is building an MCP server framework; once its first
server (Obsidian MCP) lands and the general shape is settled, the intent is to
run **our** MCP server under `la mcp serve`. That is the lever that can supply
the discrimination the daemon cannot: a distinct socket per plane, or a daemon
whose roots live inside an MCP-owned tree — i.e. D4's "handle, not path" answer
arriving as a process boundary instead of as validation logic.

Neither shape is chosen here, and this ADR does not block on that project's
timeline. Whoever wires that integration should treat D2a's test as the
**tripwire to revisit**: if the MCP plane ever legitimately needs roots, the
answer is the process boundary, **not** relaxing the test.

### D3 — Entry-time validation, never a mid-run refusal

With D2 available, the loop checks BEFORE the first generation that every path
the daemon will be asked to read — the pinned `refs/`, the candidates dir — is
inside a reported root.

**Which of relocate-or-refuse applies is normative, not a choice**
(design review MEDIUM — the first draft left it ambiguous and its own negative
tests contradicted its body). Derivation (D1) applies ONLY when no explicit
`--output-dir` was given. An explicit `--output-dir` outside the reported roots
is an ENTRY REFUSAL naming both fixes — the `--ref-root` to add and the derived
alternative — never a silent relocation of output the operator named.

**The client check must use the daemon's containment semantics exactly**
(design review MEDIUM): realpath both sides, then `path == root or
path.startswith(root + os.sep)` — `_within`'s rule. A plain `startswith`, or a
check without realpath, diverges from the gate: a symlinked run dir or a
prefix-sibling (`/data/out` vs `/data/output`) passes at entry and is refused
mid-run, and the incident this ADR exists to kill survives behind a green
entry check.

**A daemon that reports nothing gets a loud entry notice** (design review LOW):
nothing better is possible against a pre-D2 daemon, but silence is a choice.
The notice states that paths cannot be validated at entry, that a mid-run
refusal will fall back in-process, and that this may OOM — restart the daemon.

**Wire response fields are untrusted input** (design review LOW): the daemon is
trusted-equivalent, but per the ADR-012 machine-boundary discipline the client
type-checks `output_dir` / `ref_image_roots` (str / list-of-str, absolute,
NUL-free, `repr()` in logs) before any filesystem use, and on failure behaves as
the pre-D2 case. The ADR-037 D5 latch stays as the backstop
for a daemon that predates D2 or reports nothing, but it stops being the primary
mechanism.

**The latch's degrade is downgraded from "recoverable" to "best effort"** in the
documentation: on a box where the daemon holds most of VRAM, in-process
generation will OOM. That is not a regression introduced here; it is an
honest description of what the fallback can promise.

**The considered alternative, recorded rather than left silent** (design review
LOW): when the latch fires and the device's socket still exists — i.e. the
daemon is demonstrably warm and holding VRAM — refusing the rest of the run
with the `--ref-root` / `--unload` fix is strictly better than proceeding into
a probable OOM, which is what the code does today while printing "--unload it
if this run OOMs". This is NOT adopted in this ADR only because it changes
ADR-037 D5's stated contract, which is its own amendment; D3 reduces the
frequency to the stale-ping window and the pre-D2 case. Named so that silence
is not read as endorsement of the fatal path.

#### D3a — the entry check is a shared helper, and `generate`'s one-shot path consumes it too

The first draft deferred this (*"the one-shot case fails fast today because it
makes a single call; the loop is what accumulates cost"*). Grant challenged the
deferral on 2026-07-27 and it does not survive contact with the code.

**The one-shot's failure mode is identical, not milder.** `generate.py:3331`
catches `RefPathError` and falls back in-process — the same fallback that OOM'd
on 2026-07-26 — and its own warning text concedes it: *"the daemon still holds
its pipeline's GPU memory, so this in-process run shares the device — `--unload`
the daemon if it OOMs."* On a warm single-GPU box that is not a degrade, it is a
crash. The difference between one-shot and loop is **how much time is wasted
before crashing**, not whether the run survives.

The deferral was therefore reasoning about *cost of discovery* when the fact
that matters is *the fallback being fatal*. Leaving it deferred would also ship
two different behaviors for one misconfiguration — the kind of divergence that
gets re-litigated in a future session with no record of why.

Decision: the containment check is implemented **once**, as a shared helper over
(paths, reported roots) applying the D3 `_within` semantics, with no
loop-specific assumptions. The loop consumes it in slice 2 and the one-shot CLI
in slice 3. This is **sequenced, not deferred** — small slices per §3 SRR,
without leaving the trap in the CLI. The `Deferred` section keeps only genuinely
out-of-scope items.

**Scope: the check runs only when the daemon would actually serve the request**
(re-review MEDIUM). Delegation is already skipped entirely when `--output` is
set (`generate.py:3271-3273`), so gating on "a socket exists" would refuse runs
that were never going to reach the daemon. The condition is: a socket exists
for the device AND delegation is not skipped. The daemonless path and the
`--output` path are untouched.

**This is a deliberate policy change and the message must say so.** Refusal
replaces a fallback that genuinely works on a box with VRAM to spare — it is
only fatal against a warm daemon holding most of the device. Per the
warn-don't-block preference, the refusal names its escape: `--output` (which
forces in-process and skips delegation) or adding the `--ref-root` that would
make the reference legal. The operator keeps the ability to shoot themselves in
the foot; they just have to say so.

### D4 — External callers do not get a path surface at all

Out of scope for this ADR's implementation, recorded because D2 raises it.
Grant: *"I'm not confident that any amount of validation can make 'external user
can write a file into my filesystem' safe."* Agreed, and the repo already has the
precedent: the MCP surface refuses filesystem paths outright — models and LoRAs
cross it as catalog NAMES in both directions (ADR-015). The consistent answer for
reference images is a HANDLE to bytes already inside an area the daemon owns,
never a path the caller names and we validate. Two shapes to choose between when
that lands — a daemon spawned with its roots inside an MCP-owned tmpfs, or
forcing external callers daemonless — and it is Red Zone from the first commit
per the project review bar. **It gets its own ADR; nothing here may be read as
having decided it.**

## Alternatives Rejected

**Fail fast with a good error message and leave the operator to set
`--output-dir`.** This was the first proposal and it is strictly worse than
making it work: it keeps a trap in the CLI and merely labels it. Kept only as
the D3 fallback for the un-relocatable case.

**Copy/move stray reference images into the daemon's roots as a first op.**
Already what `pin_static_refs` does (ADR-038 D5) — it copies every `--ref-image`
into `<output-dir>/refs/`. The copy was never the problem; the destination was.
D1 fixes the destination, so no new move step is needed.

**Have the daemon accept a per-request ref-root extension.** Prohibited by
ADR-037 D5 and re-prohibited here: wire trust fields, daemon exemptions, and
root merging remain prohibited. The daemon's roots are spawn-time operator
arguments and stay that way.

## Deferred / Out of Scope

- **The external/LLM caller surface (D4)** — its own ADR.
- **Multi-daemon / per-device root divergence.** The loop targets one device per
  run; a run that spans devices would need per-device root resolution.
- **Cleaning up the existing mirror tree.** The 31 stale directories and 2
  orphaned PNGs under `$DAEMON_OUT/home/gawkahn/…` are operator data; D1 stops
  new ones being created but this ADR deletes nothing. Removal is an operator
  action, offered separately.
- **Collapsing `iterate_batch_id` into `run_id`.** D1b keeps them distinct to
  preserve the nesting case; if a sweep-inside-a-run is ever ruled out, the
  merge is a one-line change and its own slice.

*(Retro-fitting D3 to `generate`'s one-shot path was listed here in the first
draft. It is no longer deferred — see D3a; it is slice 3.)*

## Slice plan

1. **D2 wire field + D2a** — `ping` returns `output_dir` + `ref_image_roots`
   behind `report_roots`; `server.py` is Red Zone, so ADR (this) →
   `security-auditor` → code. Includes the negative test barring
   `mcp_server.py` from emitting `report_roots`.
2. **D1 + D1b + D3 in the loop** — mint `run_id`, derive the run dir under the
   daemon root, drop the savepath mirror and the post-hoc move, validate at
   entry via the shared helper, keep the latch as backstop. Register `run_id` +
   `iterate_batch_id` in `_SKIP_SIDECAR_KEYS` and `run_id` in the MCP
   `extract_params` drop-list. Update the vault manual with worked examples,
   including an edit run with `--ref-image` (flagged as missing since the
   ADR-038 review).
3. **D3a retrofit** — `generate`'s one-shot path consumes the same entry-check
   helper, so a bad `--ref-image` is refused before model load instead of
   falling into the in-process OOM.

Negative tests named up front: an explicit `--output-dir` outside the daemon's
roots is caught at ENTRY, before any model load or generation, and is REFUSED
rather than relocated; two DERIVED runs never share a run dir (distinct
`run_id`s, so `pin_static_refs`' `rmtree` cannot touch a foreign run's refs) and
an `exist_ok=False` collision FAILS rather than suffixing; the exclusive create
is the FIRST filesystem operation on the run dir, ordered before
`pin_static_refs` (whose `makedirs(exist_ok=True)` would otherwise materialize
it); a shared EXPLICIT `--output-dir` already holding another run's `refs/`
warns loudly and proceeds; a prefix-sibling path (`/data/out` vs `/data/output`)
fails entry validation, as does a symlinked dir that resolves outside; a derived
run dir always contains `refs/` and `candidates/`, carries `run_id` in its name,
and is echoed at entry; no daemon write lands outside the derived run dir (the
mirror tree is not recreated), the remaining rename is same-directory, and the
bytes the judge reads are the bytes the daemon wrote in THAT iteration (never a
stale canonical file in a reused directory); `run_id` appears in every sidecar
and verdict record and survives a `--params` replay without a `dropping unknown
key` warning, as does `iterate_batch_id`; `run_id` never appears in MCP
`extract_params` output; `mcp_server.py` never emits `report_roots`;
`report_roots` on a non-`ping` request type is a ValidationError; a ping
response with a malformed `output_dir` / `ref_image_roots` (non-str, relative,
NUL-bearing) behaves exactly as the pre-D2 case and never reaches `makedirs` or
`rmtree`; a daemon that reports no roots (pre-D2) still latches exactly as today
AND warns at entry; a plain `ping` without `report_roots` discloses no paths;
`ping` gaining fields does not change any existing client path; a one-shot
`generate` with an out-of-roots `--ref-image` is refused at entry rather than
falling back in-process, the refusal names `--output` and `--ref-root` as
escapes, and the check does NOT fire when `--output` already skips delegation;
and the daemonless run is unaffected by all of it.

## Changelog

- 2026-07-28 — **Vault user docs closed** (the slice-2 deliverable slice 2b did
  not do; outstanding since the ADR-038 review). Vault-only, no repo copy per
  the project CLAUDE.md. `Comfyless_Manual.md` gains five worked `--ref-image`
  examples — edit a daemon-written image with no flags, the out-of-roots
  refusal with its verbatim transcript and which of the three fixes to take,
  two-reference `:MODE` routing, replaying an edit sidecar, and chaining
  generate → edit-refine. The refusal transcript was rendered from the real
  code path, not transcribed, because a documented error message that does not
  match the emitted one is worse than none.
  Two staleness bugs found while writing, both predating this ADR: the manual
  claimed `--params` replay "deliberately does NOT replay `ref_images` yet
  (ADR-035 slice 5, open)" — slice 5 SHIPPED, and a comfyless sidecar does
  replay refs through its trust gate; only the Eric-Save PNG chunk drops them
  permanently, which is a different provenance question. And
  `Comfyless_Refine.md` still documented `--output-dir` as *required* and the
  daemon path as always a cross-tree move, both of which D1 changed in slice
  2b — it now documents omitting the flag as the recommended daemon path, the
  three-branch savepath behavior, and the `--seed-image` warn-vs-refuse
  divergence with its rationale and cheapest fix.
- 2026-07-27 — **Slice 3 shipped (D3a) — the slice plan is complete.**
  `generate`'s one-shot path calls `refuse_out_of_roots_refs`, which consumes
  the SAME `query_daemon_roots` + `paths_outside_roots` helpers slice 2b built
  (D3a's "implement once, consume twice" — there is still exactly one
  client-side spelling of containment, deferring to `server._within` by import).
  An out-of-roots `--ref-image` now exits 2 before any model load instead of
  falling into the in-process fallback that OOMs against a warm daemon. The
  scope gate is the normative half and is enforced cheap-first — specs →
  `_should_delegate_to_server` → socket → ping — so an `--output` run neither
  refuses nor pings, and a daemonless run is untouched.
  **The warn-vs-refuse divergence recorded in the slice-2b entry is now IN
  EFFECT, not pending:** `generate --ref-image /outside` exits 2 while
  `refine --seed-image /outside` warns and latches. Pinning the seed (which
  would close it, and independently closes the ADR-038 D5 TOCTOU) remains its
  own slice, gated on an ADR-037 D5 amendment, filed in TECH_DEBT.
  Full record: `docs/security/review-adr040-slice3-2026-07-27.md`.
  **The MEDIUM both reviewers found independently, because it is the shape of
  bug this ADR keeps producing: a refusal whose named escape does not work.**
  The message offered `--output` to force in-process, but
  `_should_delegate_to_server` is `bool(savepath) or default_output` and the two
  flags are independent — so on a `--savepath` run, adding `--output` changes
  nothing and the operator hits the identical refusal. `--savepath` is the
  documented way to name output with a daemon running, so that was the DOMINANT
  configuration, and the only offer left standing was `--ref-root <whole
  directory>` — the exact ADR-035 Finding 6 breadth grant the narrowest-first
  ordering exists to discourage. The escape now branches on `args.savepath` and
  names the flag that must GO. The tests were complicit and were rewritten:
  asserting the string `--output` appears in the message is an assertion about
  the implementation; driving the advised escape to `rc is None` is an assertion
  about the promise.
  **Placement is the promise, so placement is now pinned.** Every D3a test drove
  the helper directly, so deleting the call site left them all green while
  "refused at ENTRY" silently became false. Source-pinned checks now hold the
  call, the exact default-output literal it passes, and its position before
  `_confirm_iteration` and `def _run_one`.
  Smaller findings folded: the pre-D2 daemon case emitted no notice, so a
  skipped entry check was silent (D3 makes that notice normative for the sibling
  surface — `refine` had it, `generate` did not); the "copy it here" destination
  asserted `output_dir` was a member of `ref_image_roots` without checking, the
  same misattribution slice 2b already paid for; the suggested `--ref-root` was
  a LEXICAL dirname, which does not contain a symlinked reference the daemon
  realpaths, and named only the first offending directory; the `except
  ValueError` fail-open was documented as unreachable but is reachable via
  `_apply_replay_ref_trust`, which rewrites `args.ref_image` from an untrusted
  sidecar AFTER `main()` validated the typed specs — it now fails CLOSED; the
  new entry ping queues behind an in-flight generation on the daemon's serial
  accept loop with a 600 s deadline, so it now announces what it is waiting on;
  and three operator-facing statements (`_should_delegate_to_server`'s
  docstring, the `--ref-root` help text, the vault manual) still asserted "there
  is NO client-side ref-root gate — the CLI cannot know the daemon's
  --output-dir", the premise D2 retired. Accepted residuals, named: the
  stale-ping window (a daemon restarted between ping and request), refusal even
  when the daemon holds no VRAM (D3a rules on this explicitly), and the D2a
  tripwire remaining name-enumeration — one wrapper longer now, with the
  structural replacement recorded for when it is next touched.
- 2026-07-27 — **Slice 2b shipped (D1 + D3).** The loop pings for roots
  (`{"type": "ping", "report_roots": true}`, a literal on a ping request and
  nowhere else), derives `<daemon output_dir>/refine-<run_id>` when no
  `--output-dir` was given, validates at entry through a shared
  `paths_outside_roots` helper that defers to `server._within` by import rather
  than re-deriving containment (slice 3 consumes the same one, per D3a), sends
  the savepath template RELATIVE when the run dir is inside the daemon's output
  root, and keeps a same-directory `os.replace`. `--output-dir` becomes
  optional; an explicit dir outside the reported roots is an entry refusal.
  Full record: `docs/security/review-adr040-slice2b-2026-07-27.md`.
  **Scoping correction to D1's mirror-tree claim.** §Deferred says "D1 stops
  new ones being created"; that holds for run dirs under the daemon's OUTPUT
  dir, which is every derived run and any explicit dir inside it. A third
  branch is reachable and unfixable from the client: an explicit `--output-dir`
  inside a `--ref-root` but NOT under the daemon's output dir passes entry
  validation (ref roots ⊋ output dir) and still needs the absolute template,
  because the daemon only ever writes under its own root. That case still
  builds a mirror and still moves cross-tree. Now covered by a named test
  rather than left as the branch an implementer is likeliest to get wrong.
  **`--seed-image` outside the roots WARNS rather than refuses, and that is a
  divergence from D3a's ruling for the sibling surface — recorded, not
  implied.** When slice 3 lands, `generate --ref-image /outside` will exit 2
  while `refine --seed-image /outside` warns and latches. The distinction taken
  here: D3's refusal is normative for the loop-OWNED directories, while the
  seed is operator-typed input the loop does not relocate — and refusal's only
  escape would be `--ref-root <photo directory>`, the breadth exposure ADR-035
  Finding 6 warns about, since `--ref-root` cannot name a single file. Both
  reviewers agreed the real answer is to PIN the seed into the run dir exactly
  as `pin_static_refs` does for `--ref-image`, which makes the failure
  unrepresentable rather than warned. That is deferred to its own slice
  because it amends ADR-037 D5's edit-source contract, and it is filed in
  TECH_DEBT. The security auditor found the case for pinning does not rest on
  this ADR at all: the seed is consumed on TWO channels — pinned bytes for the
  judge's anchor (`load_seed_image_capped` at entry) and a PATH the daemon
  re-reads every pre-promotion iteration — so a mid-run swap makes generation
  condition on new bytes while the judge scores identity against old ones. That
  is the exact TOCTOU ADR-038 D5 closed for every other reference, and it
  exists daemon or no daemon. The ADR's "Alternatives Rejected" claim that
  `pin_static_refs` means "no new move step is needed" is true for
  `--ref-image` and false for `--seed-image`.
  **A guard this slice nearly broke and restored before commit:** D2a's
  negative test asserts `mcp_server.py` never emits `report_roots`. Slice 2b
  put the wire literal inside `query_daemon_roots`, a NAMED WRAPPER — so all
  three tripwire assertions (string absence, premise symbol set,
  `inspect.getsource(generate)`) stayed green if `mcp_server.py` called the
  helper. The slice-1 review predicted exactly this and pinned the assertion to
  `generate()` because the helper did not exist yet. The premise symbol set now
  lists `query_daemon_roots` and says in-comment that it tracks the helper's
  NAME, not the wire field.
  Smaller findings folded: the containment refusal now branches on `derived`
  (an operator who omitted `--output-dir` was being told to omit it, and
  pointed back at the root that failed — reachable only against a daemon
  reporting an `output_dir` outside its own `ref_image_roots`); derived run
  dirs are created `0o700` because under D1 the location is no longer the
  operator's choice and pinned copies of their private reference photos land
  in it (explicit dirs keep default modes — they chose that path); the seed
  warning now leads with "move it under a root" and marks `--ref-root` the
  broad fallback; `--output-dir ""` is refused rather than silently derived;
  and the D1 dirname charset, normative but previously unenforced, is pinned by
  a test over `RUN_DIR_STEM` and `mint_run_id`. Accepted residuals, named:
  a TOCTOU window on the EXPLICIT branch only (the derived branch is immune —
  `exist_ok=False` raises on a pre-existing symlink), and a malformed ping
  being indistinguishable from a pre-D2 daemon (both degrade to the ADR-037 D5
  latch, which is what D3 preserves on purpose).
- 2026-07-27 — **Slice 2a shipped (D1b only).** Slice 2 as planned (D1 + D1b +
  D3) was split for reviewability per §3 SRR: 2a is the correlation primitive,
  which needs no daemon interaction; 2b is the derived run dir, the mirror-tree
  removal, and entry validation. Same scope, two reviewable diffs. Both
  reviewers judged the split coherent.
  **Scoping correction to D1b's "every record":** the claim holds for the
  `generate` CLI mode, `refine`, and `cascade`. It does NOT cover
  `_run_json_mode` (the `--json` bridge writes no sidecar — metadata goes to
  stdout, so no on-disk record is left uncorrelated) or `comfyless/video.py`'s
  segment sidecars (never in this ADR's scope). Named here rather than left
  ambiguous.
  **A registration bug this slice introduced and closed before commit:**
  `run_id` was added to cascade's `_KNOWN_KEYS` without cascade also MINTING
  one. `validate_config` copies unknown keys rather than dropping them (warn
  only), and dispatch builds `sidecar = dict(cfg)` — so a replayed sidecar, or
  an agent-supplied `cascade_config` over MCP, would have inherited a FOREIGN
  run's correlation id as this run's provenance, and registering the key had
  also removed the `unknown keys ignored` audit line that previously flagged
  it. Both reviewers caught it independently. Cascade now mints its own and the
  sidecar `update()` block overwrites unconditionally — matching how
  `iterate_batch_id` is protected, and how `output_format`/`quality` are popped
  for the same reason. This also completes D1b's "every entrypoint".
  **`run_id` is REQUIRED, not `Optional[str] = None`** (code review): a None
  writes `"run_id": null` into every record, and since the id's whole purpose
  is equality-grouping, that collapses every unset run into one bucket — worse
  than a missing key. Keyword-only-without-default makes pyright enforce it at
  every call site.
  Tests moved from source-greps to behavioral where a harness already existed:
  the loop's records are now read off disk on both `_generate_one` paths
  (including the RefRefused in-process fallback), the delegated `generate`
  sidecar is asserted against a stubbed daemon, a sweep is asserted to share one
  `run_id` while carrying a separate `iterate_batch_id`, and `_load_sidecar` is
  exercised rather than its filter reimplemented. A source-count assertion was
  removed as unsound — it would have stayed green if a third, unstamped call
  site were added, which is precisely the failure the choke point exists to
  prevent.
- 2026-07-27 — Design security RE-review of the same-day revision
  (`security-auditor`, invoked WITHOUT a `model:` argument per Grant's standing
  no-elevation instruction — but the transcript shows it ran on
  **`claude-fable-5` for all 40 turns anyway**, via the agent file's frontmatter
  pin. That pin was documented as non-functional (CLAUDE.md §5A); it is
  evidently functional in the current Claude Code build, so omitting `model:`
  no longer avoids Fable. Flagged to Grant; not resolved here.) Findings folded
  before acceptance. **One HIGH, and it
  corrected a factual error introduced by the revision itself:** the revision
  claimed `candidate_000001.png` proved the daemon's collision counter had
  fired, and concluded D1 removes the move step entirely. `_resolve_savepath`
  (`generate.py:522-524`) appends the counter UNCONDITIONALLY from `counter = 1`,
  so `…0001` is the first slot, not a collision — and the daemon therefore still
  does not write the canonical name after D1. Removing the rename would have
  forced an unstated ADR-034 D7 amendment and, in one foreseeable
  implementation, re-opened the judge-vs-generation divergence ADR-038 D5
  closed. D1 now removes the MIRROR and keeps a same-directory `os.replace`.
  **Three MEDIUM:** run_id uniqueness applies to DERIVED dirs only, so a shared
  explicit `--output-dir` still carries the prior HIGH — now stated as a
  residual with an entry warning rather than implied closed, and the negative
  test scoped to derived runs; the `exist_ok=False` create must be the FIRST
  filesystem op (`pin_static_refs`' `makedirs(exist_ok=True)` would otherwise
  materialize the run dir and silence the assertion, and the fresh create is
  what bounds writes derived from a wire value); D3a's check must fire only when
  the daemon would actually serve the request (`--output` already skips
  delegation) and must name its escape, since refusal replaces a fallback that
  works on a roomy box. **Three LOW:** D2a's test relabeled a regression
  tripwire rather than a control (the daemon answers any same-UID caller;
  `mcp_server.py` holds no wire-client code at all today), plus `report_roots`
  honored only on `type: ping`; a negative test added for malformed ping wire
  values, which the body promised to type-check but no test covered; birthday
  arithmetic corrected to ~77k. The reviewer independently verified the
  no-wire-change claim, `_assert_no_paths` still holding, the `iterate_batch_id`
  skip-keys wart being real, every load-bearing line reference, that no ADR-037
  D5 prohibited alternative is smuggled in, and that no new write authority or
  root widening is granted. It also found D1b's MCP claim UNDERstated — `run_id`
  cannot survive `extract_params` even unlisted, since both paths already filter
  structurally.
- 2026-07-27 — Revised on Grant's rulings across four items, before any code.
  **D1:** the collision-suffix scheme is replaced by a per-invocation `run_id`
  in the dirname, minted at session start — *"truly unique dirnames for every
  run… not just 'check to see if it exists and append a string if it does'"*;
  `exist_ok=False` is demoted to an assertion. D1 also now states the
  consequence the first draft missed: it eliminates the absolute-path MIRROR
  TREE the daemon builds under its own output dir (measured: 31 dirs, 2 orphaned
  PNGs) and the post-hoc `shutil.move` — a behavior Grant had complained about
  directly, and the answer to where the `…0001`-suffixed images were going.
  **D1b (new):** `run_id` becomes the correlation primitive on every record,
  closing the gap that refine runs mint no `iterate_batch_id` and so have no
  correlation key at all; the two stay DISTINCT fields sharing one minting
  helper, because Grant named the nesting case (`--batch` inside a judge run)
  that aliasing would break. Verified to need NO wire change — both clients
  write sidecars themselves post-response — so the correlation half is not Red
  Zone. Registration duties recorded, including a pre-existing wart this closes:
  `iterate_batch_id` is absent from `_SKIP_SIDECAR_KEYS`, so `--params` replay
  of an `--iterate` sidecar already logs a spurious unknown-key drop.
  **D2a/D2b (new):** the MCP path leak is handled NOW rather than deferred
  behind the `local_agents` timeline. The daemon cannot discriminate MCP from
  CLI (same UID, same socket, same schema); no auth is built. Instead
  `report_roots` is CLI-plane only, enforced by a negative test barring
  `mcp_server.py` from emitting it, and the residual risk is recorded AS a
  residual. D2b names `la mcp serve` as the intended structural fix explicitly,
  because this ADR is a document that integration will read, and marks D2a's
  test as the tripwire to revisit. **D3a (new):** the `generate` one-shot
  deferral is REVERSED — the one-shot's fallback is the same fatal in-process
  OOM (`generate.py:3331`, whose own warning concedes it), so the deferral was
  reasoning about cost-of-discovery when the fatality of the fallback is what
  matters; the entry check becomes a shared helper and the retrofit is sequenced
  as slice 3. Slice plan now 3 slices; `Deferred` holds only genuinely
  out-of-scope items plus the mirror-tree cleanup (operator data, not deleted
  here).
- 2026-07-26 — Design security review (`security-auditor`, Fable, no fallback)
  folded before acceptance: one HIGH — the derived run dir plus
  `pin_static_refs`' unconditional `rmtree` would have made two concurrent runs
  deriving the same stem destroy each other's pinned refs, so exclusive
  creation and a defined stem charset are now normative. Three MEDIUM: the
  `ping` fields are opt-in behind `report_roots` so a future bridge cannot leak
  them by forwarding a health check; relocate-vs-refuse is now normative
  (derive only when no `--output-dir` was given, refuse an explicit one);
  and the client containment check must mirror `_within` exactly, realpath
  included. Plus three LOW (pre-D2 entry notice, the fail-fast alternative to
  the latch recorded, wire fields treated as untrusted). The reviewer confirmed
  D3 smuggles in none of ADR-037 D5's prohibited alternatives and D1 grants no
  new daemon-side write authority. Full record to be saved under
  `docs/security/` with the implementation slice.
- 2026-07-26 — Proposed, from the first live ADR-039 run. Grant chose "derive it
  automatically" over "fail fast" and directed that stray refs be moved into the
  roots rather than refused — which `pin_static_refs` already does, so D1 is the
  whole fix. D4 recorded as explicitly undecided.
