# Security Re-Review — ADR-035 Reference-Image Surface (amended design)

**AI-Disclosure:** Claude (Fable 5, `security-auditor` agent) authored; Grant reviewed.
**Date:** 2026-07-20
**Artifact reviewed:** `docs/decisions/ADR-035-comfyless-reference-image-surface.md` (Status: proposed, post-amendment)
**Prior review:** `docs/security/review-adr-035-reference-image-surface-2026-07-20.md` (4 BLOCKS, 5 preconditions, 2 INFO)
**Scope excluded:** MCP surface / agent uploads (deferred to a later ADR).

---

## Summary

The amended ADR-035 was re-reviewed against the four blockers and five preconditions
of the prior review, then the amendments themselves were reviewed adversarially.
Threat model unchanged: (1) arbitrary user files → PIL decode → model conditioning,
(2) same-UID wire clients → daemon file reads, (3) attacker-craftable sidecar/PNG
metadata → replay-driven file reads. What was checked beyond the two documents: the
daemon's root plumbing and NUL defense (`comfyless/server.py:52-64,199-266,381,
934-1000`), the serial accept loop and client-controlled savepath templating within
`--output-dir` (`server.py:777-839,989-995`), `_request_cache_key`
(`server.py:397-448`), refine's echo/roots machinery and full-schema seed authority
(`comfyless/refine.py:1345-1512`), and — decisive for the new findings — the CLI's
transparent daemon delegation (`comfyless/generate.py:2444-2470`) and plan mode's
worker-subprocess structure (`comfyless/video.py:738,798-799`).

Verdict up front: **all four blockers are genuinely closed** — the amendments are
constraints an implementer following them literally would obey, not just added
prose — and all five preconditions plus the INFO are carried. However, the
amendments open two new seams of the same "implementer will improvise under
deadline" class the original blockers were blocked for, both one-to-two-sentence
fixes: decision 7's refusal set is undefined on exactly the cold path it targets,
and rows 1/3 of the trust table collide on the *default* execution path because the
interactive CLI silently delegates to a live daemon.

## Coverage

Reviewed:
- `docs/security/review-adr-035-reference-image-surface-2026-07-20.md` (full)
- `docs/decisions/ADR-035-comfyless-reference-image-surface.md` (full, amended text)
- `comfyless/server.py:40-279, 330-460, 770-850, 930-1001`
- `comfyless/refine.py:1340-1512`
- `comfyless/generate.py:2086-2180, 2440-2506, 3290-3335`
- `comfyless/video.py` (grep-level: worker dispatch at 738, 798-799 only)

Not reviewed (and why):
- MCP surface / agent uploads — out of scope per the brief and the ADR's Deferred section.
- `nodes/eric_diffusion_manual_loop.py` internals, `detect_pipeline_class` body — verified by the prior review; the amendment cites them consistently with what that review confirmed.
- `comfyless/params_schema.py`, `params_validation.py` bodies — characterized secondhand, consistent with server-side code read.

## 1. Blocker verification

**(a) `_check_paths` weight-root reuse → decision 6a: CLOSED.** ADR:251-266 names
the wrong-trust-class error explicitly, names *both* bad fixes from the review as
forbidden ("exempting `ref_images` from containment entirely, or adding
`output_dir` to the weight roots"), and specifies the replacement concretely:
`ref_image_roots = --output-dir ∪ --ref-root additions`, shared `_within`
mechanism, "allowlists are disjoint in purpose and never merged" (ADR:264-266). An
implementer following this literally cannot reproduce either failure branch, and
the default admits the first consumer (keyframes are prior outputs in output-dir —
confirmed compatible with `server.py:777-839`, where outputs land in `output_dir`
by construction). Closed. New issues it opens: Findings 2, 4, 5, 7.

**(b) Cache-key invariant → decision 3: CLOSED.** The claim is narrowed to its
true scope, the invariant is stated as a binding constraint in a block quote
(ADR:210-213: class selected exactly once in `detect_pipeline_class`, no
`from_pipe` on a cached pipeline), the pressure point (`FluxPipeline` vs
`FluxImg2ImgPipeline`) is named with the mandated resolution (drop path, no quiet
upgrade, ADR:215-222), the future-slice discriminator-in-same-commit rule travels
with the decision (ADR:223-225), and the pinning test in the
`test_server_robustness` NAG style is required (ADR:227-228). This matches
everything the review demanded, including the test. Verified consistent with
`_request_cache_key` (`server.py:397-448` — ref images correctly analogous to the
NAG exclusion at 411-415). Closed; no residual.

**(c) Cap-enforcement locus → decision 6b: CLOSED, with one wording defect.**
ADR:268-276 places the shared helper "in whichever process performs the decode —
the CLI process on foreground runs, the daemon's generate path on daemon runs,"
which answers the foreground question: on foreground runs the CLI process *is* the
decode site and therefore hosts the real gate; "a CLI-side pre-check for fast
feedback is UX, not the gate" (ADR:276) reads unambiguously only once you've
absorbed the prior sentence — the pre-check clause refers to daemon-carried runs.
The locus blocker is closed. But 6b's ordering clause ("after `ref_image_roots`
containment and before PIL touches the file," ADR:275-276) contradicts decision 7
row 1 for foreground typed paths — Finding 3.

**(d) Replay-side trust treatment → decision 7: CLOSED as to substance.** The
three-class table (ADR:326-330) supplies exactly what the review said was absent:
file-derived paths get mandatory F4 echo extending `_SEED_ECHO_PATH_FIELDS`
(verified real, `refine.py:1352`), and go *stricter* than the review's minimum —
refusal on the cold path with re-specification instructions (ADR:329, 332-337),
correctly justified by the same nobody-stumbles logic as strict MODE. The
cold-path no-gate fact it relies on is real (`refine.py:1488-1491`). Closed — but
the refusal set it names is undefined off-daemon (Finding 1) and the row 1/row 3
seam is under-specified (Finding 2).

## 2. Non-blocking findings — carriage check

| Review finding | Where | Status |
|---|---|---|
| Format allowlist + single-frame | 6c, ADR:281-285 (`formats=["PNG","JPEG","WEBP"]`, first-frame, EPS/Ghostscript rationale carried) | Resolved in text |
| NUL defense for `ref_images` | 6e, ADR:293-298, incl. the mirrored negative test | Resolved in text |
| TOCTOU single-read + SHA-256 | 6d, ADR:287-291 (read once, hash those bytes, decode those bytes) + SHA-256 pinned at ADR:306 | Resolved in text |
| Colon-filename escape + error naming | Decision 1, ADR:110-117 (documented `photo:ref:both` escape; error must name the full-spec file) | Resolved in text |
| Plan-mode hard-fail + applied/dropped sidecar field | Decision 2, ADR:131-146 (interactive warn / machine hard-fail or fail-the-segment; per-ref applied/dropped recorded in every mode) + decision 7 recording, ADR:305-308 | Resolved in text (locus gap → Finding 4) |
| Fingerprint INFO | Deferred, ADR:414-419 (unkeyed-SHA-256 content-confirmation + path disclosure, explicitly willed to the MCP ADR) | Carried as specified |

All six accounted for. None silently dropped.

## 3. Findings — adversarial review of the amendments themselves

### [HIGH] Finding 1 — Decision 7 row 2's refusal set is undefined on the only path it applies to

**Location:** ADR-035:329 ("`ref_image_roots` ∪ operator roots… refused on the
cold path"); ADR-035:263-265 (6a defines `ref_image_roots` only as daemon spawn
flags); `comfyless/refine.py:1486-1497`

6a defines `ref_image_roots` exclusively in daemon terms (`--output-dir` +
`--ref-root` *spawn* flags), but row 2's refusal executes on the **cold in-process
path** — refine seed entry, `--params` replay — where no daemon and no spawn flags
exist; refine's only available root set today is the weight/catalog roots union
(`refine.py:1486`), which is precisely the set 6a establishes ref images do *not*
live in. Followed literally, row 2 refuses every legitimately replayed keyframe
sidecar (refs live in the output dir, not the weight roots) — the same
over-refusal shape as original blocker (a), and the same predictable improvisation
under deadline: the implementer loosens row 2 back to warn-only, silently
reopening blocker (d). Additionally, "re-specify it on the command line" (ADR:329,
333-334) is a dead end unless `--ref-image` actually exists on the refine/replay
CLI surfaces — currently unstated.

**Remediation:** one added sentence in 6a or 7 defining the off-daemon root set —
e.g. "on foreground/cold-path runs, `ref_image_roots` = the invocation's output
directory ∪ `--ref-root` (also available as a CLI flag) ∪ the operator weight
roots" — and a clause noting the re-specification escape requires `--ref-image` on
every entry surface that can carry file-derived refs (refine included).

**Verdict: blocks acceptance** (textual amendment).

### [HIGH] Finding 2 — Rows 1 and 3 of the trust table collide on the default execution path

**Location:** ADR-035:326-330; `comfyless/generate.py:2444-2470` (CLI silently
delegates whenever `socket_path(args.device).exists()`, falls back in-process
otherwise); precedent at `generate.py:2450-2452`

Transparent daemon delegation makes "typed at the CLI" and "crossing the daemon
socket" overlapping, not disjoint, classes. Row 3 does state precedence
("regardless of how the client obtained the path") — so the composition is
*defined* — but the consequence is not acknowledged: the same interactively typed
`--ref-image ~/photos/car.jpg` succeeds when no daemon runs and is **refused by
`ref_image_roots` containment when a daemon happens to be up**, i.e. row 1's "no
containment gate… would break legitimate use" guarantee is false on any machine
with a live daemon, which is the normal state here (ADR-020 daemons are
long-lived). Two implementer failure modes follow: (i) "fix" the UX by carrying a
`typed_by_user` provenance field on the wire — a client-asserted trust claim any
wire client can forge, gutting row 3; (ii) users trained to work around refusals
by widening `--ref-root`. The codebase already contains the correct resolution
pattern: delegation is *skipped* when the request contains something the server
cannot honor (`--output`, `generate.py:2450-2452`).

**Remediation:** two sentences in decision 7: (1) a typed `--ref-image` outside
`ref_image_roots` causes the CLI to skip daemon delegation and run in-process (the
existing `--output` precedent), preserving row 1 authority and row 3 containment
simultaneously; (2) trust class is determined solely by which boundary the path
arrives through and is never carried or honored as a wire/request field.

**Verdict: blocks acceptance** (textual amendment).

### [MEDIUM] Finding 3 — 6b's ordering clause contradicts row 1 for foreground typed paths

**Location:** ADR-035:275-276 ("after `ref_image_roots` containment and before
PIL touches the file") vs ADR-035:328 (typed paths: "no containment gate")

A literal implementer of 6b applies `ref_image_roots` containment inside the
shared decode helper unconditionally — imposing containment on foreground typed
paths that decision 7 exempts; the resulting breakage pressures removal of the
containment call from the shared helper, weakening the daemon path where it is
mandatory.

**Remediation:** reword 6b to "after whatever containment treatment decision 7
assigns to the path's trust class" (or move containment explicitly outside the
shared cap/decode helper).

### [MEDIUM] Finding 4 — Decision 2's mode predicate has no enforcement locus at the daemon boundary

**Location:** ADR-035:131-142; `comfyless/video.py:738,798-799` (plan workers are
CLI subprocesses); `comfyless/generate.py:2455` (interactive CLI also arrives via
the socket)

Plan workers and interactive CLI runs reach the daemon over the same socket; the
daemon — which performs family routing and therefore *discovers* the drop — cannot
distinguish "plan/machine-driven" from "interactive," so the warn-vs-hard-fail
split is undecidable at the place the ADR implies it executes. Improvised fixes: a
wire flag defaulting to lenient (machine clients silently inherit
warn-and-proceed, recreating the corrupted-chain failure decision 2 exists to
prevent), or client-side-only enforcement that daemon-direct clients (MCP,
scripts) never see.

**Remediation:** one sentence: strictness crosses the wire as an explicit request
field whose **absent-value default is hard-fail** (fail closed); only the
interactive CLI, on a TTY, sets the lenient value; applied/dropped is recorded in
wire metadata and the sidecar in every mode regardless.

### [MEDIUM] Finding 5 — No per-request reference-count cap

**Location:** ADR-035:287-291 (6d), ADR-035:99-102 (repeatable flag, 1–N);
`comfyless/server.py:45` (1 MiB frame bounds path *count* only, to ~thousands),
`server.py:989-995` (serial loop bounds concurrency, not per-request N)

64 MB byte cap × N refs read-into-memory: a single wire request listing a few
hundred distinct in-root images (trivially satisfiable — the daemon's own output
dir fills with valid PNGs) drives multi-GB transient allocations plus N decoded
tensors inside the VRAM-holding daemon — the per-file caps compose into no
per-request bound, and `manual_loop.py`'s 1–N precedent states no N.

**Remediation:** pin an explicit per-request ref-count cap in decision 1 or 6
(single digits satisfies the first consumer; "Picture N" prompting degrades far
below any memory-relevant N anyway), enforced at the same site as the 6b caps,
with a negative test.

### [MEDIUM] Finding 6 — `--ref-root` breadth is unguarded

**Location:** ADR-035:263-265 (6a)

`--ref-root /` or `--ref-root ~` converts row 3's containment into a no-op: every
user-readable image on the machine (private photos, browser-cache images) becomes
readable — and its bytes VAE-encodable into shareable output — by *any* same-UID
wire client, silently and permanently for the daemon's lifetime. Unlike
`--model-base` (useful reads limited to weight-shaped trees), the ref surface
makes broad roots maximally exploitable, and the operator adding a root "to make
one command work" is the realistic path there.

**Remediation:** one sentence in 6a: at spawn, a ref root equal to `/`, a mount
root, or the user's home directory produces a loud warning naming the exposure
(warn-don't-block, per house posture); consider logging each configured ref root
at startup as `server.py:982-983` already does for extra roots.

### [INFO] Finding 7 — Output-dir read-back loop and shared-output-dir cross-reads

**Location:** ADR-035:263-266 (6a default); `comfyless/server.py:784-798`
(client-chosen filenames within output-dir), `:809-812` (two daemons canonically
share `--output-dir`); ADR-035:414-419 (Deferred)

With output-dir as a default ref root, any wire client can read back any decodable
image any session or daemon wrote there (including MCP-driven outputs), and
client-controlled savepath naming lets one flow plant an image another flow will
read by name. Under the solo same-UID model this is the intended chaining feature
and adds no authority; once agent-driven flows front the daemon it becomes a
cross-session read/plant channel. The 6c format allowlist usefully limits the
primitive to decodable images (sidecar JSONs in the same dir fail decode) —
provided decode-failure errors never echo file bytes.

**Remediation:** add this read-back item to the Deferred MCP-awareness note
(alongside the fingerprint item), plus one clause anywhere in 6: ref
decode/containment errors report path and reason only, never file content.

### [INFO] Finding 8 — A fourth trust class — LLM-planner-derived — is absent from decision 7's table

**Location:** ADR-035:326-330; ADR-027 F1 closed two-key planner-override
allowlist

The table covers typed / file-derived / wire. The refine loop's planner (LLM
output) is today barred from path-shaped keys by the closed two-key allowlist;
nothing in ADR-035 records that `ref_images` must not join that allowlist, so a
future refine slice could add planner-adjustable refs without tripping any
documented constraint — LLM-directed file reads with none of the three treatments
applying. Relatedly, the typed-vs-file-derived seam in refine needs one rule: a
typed `--ref-image` on the refine CLI **replaces** seed-derived `ref_images`
wholesale, never merges — otherwise a crafted sidecar's extra ref rides alongside
the user's typed one under row-1 coloration.

**Remediation:** two sentences in decision 7: `ref_images` is excluded from the
ADR-027 planner-override allowlist (adding it requires its own security review),
and typed refs replace rather than merge with file-derived refs.

## 4. Acceptance verdict

The four blockers are closed on the strict standard (literal compliance prevents
the original failure scenarios), and all five preconditions plus the INFO are
carried. **The ADR should not flip to `accepted` quite yet**: the two HIGH
findings sit *inside* the amended decisions 6a/7 — the cold-path refusal set is
undefined on the only path it governs, and the row 1/row 3 collision on the
transparent-delegation path invites exactly the improvised-policy failure the
original blockers were blocked for. Both are one-to-two-sentence textual
amendments in the same spirit as the last round; with those landed, `accepted` is
warranted.

Preconditions to record before the implementation slice (spec/TECH_DEBT, not
necessarily ADR-body):

1. **(ADR text, pre-accept)** Off-daemon `ref_image_roots` definition +
   `--ref-image` availability on every replay-capable entry surface — Finding 1.
2. **(ADR text, pre-accept)** Skip-delegation rule for typed refs outside roots +
   "trust class is never wire-asserted" — Finding 2.
3. 6b reword to defer containment to decision 7's class assignment — Finding 3.
4. Wire-carried strictness flag, absent = hard-fail — Finding 4.
5. Per-request ref-count cap with negative test — Finding 5.
6. `--ref-root` breadth warning at spawn — Finding 6.
7. Deferred-section additions (read-back loop; no-content-in-errors) and the
   decision-7 planner-allowlist exclusion + replace-not-merge rule — Findings 7–8.
