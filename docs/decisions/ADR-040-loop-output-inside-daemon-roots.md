# ADR-040 — The loop's output directory lives inside the daemon's roots

Status:   proposed
AI-Disclosure: Claude (Fable 5) authored; Grant reviewed.

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

`refine` resolves its run directory to `<daemon output_dir>/<run stem>` when a
daemon is reachable on the target device. `refs/` and `candidates/` are then
inside a ref root **by definition**, and the class of failure above becomes
unrepresentable rather than diagnosed.

**The derived run dir is created EXCLUSIVELY, and this is load-bearing**
(design review HIGH). Today the operator picks distinct `--output-dir`s, so
ADR-038's accepted `candidates/` collision residual is unlikely in practice. D1
would make the run dir a deterministic function of daemon state plus a stem —
and this environment runs concurrent sessions by design against a daemon built
to serve them. Two runs deriving the same stem would SHARE a run dir, and
`pin_static_refs` opens with an unconditional `shutil.rmtree(refs_dir)`
(`refine.py:3124`): run B would delete and repin `refs/` while run A is
mid-iteration, so run A either dies on a missing `ref_00.jpg` or silently reads
run B's reference bytes — reopening exactly the judge-vs-generation divergence
ADR-038 D5 closed. It would also delete a COMPLETED prior run's pinned
provenance whenever a stem repeats.

Therefore: the run directory is created with `os.makedirs(..., exist_ok=False)`
(O_EXCL-equivalent) and, on collision, gets a uniqueness suffix rather than
joining the existing directory. The stem is `[A-Za-z0-9_-]` only — no path
separators, no operator text interpolated unescaped.

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
- **Retro-fitting D3 to `generate`'s one-shot path.** The one-shot case fails
  fast today because it makes a single call; the loop is what accumulates cost
  before discovering the problem.

## Slice plan

1. **D2 wire field** — `ping` returns `output_dir` + `ref_image_roots`;
   `server.py` is Red Zone, so ADR (this) → `security-auditor` → code.
2. **D1 + D3 in the loop** — derive the run dir under the daemon root, validate
   at entry, keep the latch as backstop. Update the vault manual with worked
   examples, including an edit run with `--ref-image` (flagged as missing since
   the ADR-038 review).

Negative tests named up front: an explicit `--output-dir` outside the daemon's
roots is caught at ENTRY, before any model load or generation, and is REFUSED
rather than relocated; two runs deriving the same stem never share a run dir
(exclusive creation, and `pin_static_refs`' `rmtree` therefore cannot touch a
foreign run's refs); a prefix-sibling path (`/data/out` vs `/data/output`) fails
entry validation, as does a symlinked dir that resolves outside; a derived run
dir always contains `refs/` and `candidates/` and is echoed at entry; a daemon
that reports no roots (pre-D2) still latches exactly as today AND warns at
entry; a plain `ping` without `report_roots` discloses no paths; `ping` gaining
fields does not change any existing client path; and the daemonless run is
unaffected by all of it.

## Changelog

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
