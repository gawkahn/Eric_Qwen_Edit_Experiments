AI-Disclosure: Claude (Opus 4.7, 1M context) authored this security review; Grant reviewed.

# Security Review — ADR-015 Catalog-Mediated Reference Resolution (design, pre-implementation)

**Date:** 2026-05-22
**Reviewer:** `security-auditor` (Opus 4.7), model pinned at invocation.
**Target:** `docs/decisions/ADR-015-mcp-catalog-reference-resolution.md` — a Red Zone design change to the comfyless MCP reference contract. Reviewed before code per project §12 order-of-operations.

---

## Round 1

### Summary

ADR-015 replaces the MCP weight-reference contract (caller-supplied absolute paths validated with realpath + `_within`) with opaque catalog names resolved by a spawn-time, in-memory `name → abs_path` catalog. It is a Red Zone design change reviewed before code per project §12. Posture: the core decision reduces the attack surface (no caller-controlled directory in the load decision; names not paths cross the boundary in both directions) and is the right direction. But the ADR's "miss-naming is not an enumeration oracle" claim (§2) contradicts an already-accepted live finding on this surface, and several catalog-construction trust boundaries (scan-time symlink following, name normalization on case-folding/Unicode filesystems, scan-vs-manifest collision interaction) are underspecified for the document that becomes the security keystone for slices 2-5. CHANGES REQUIRED: 2 HIGH, 3 MEDIUM, 3 INFO.

### Coverage

Reviewed:
- `docs/decisions/ADR-015-mcp-catalog-reference-resolution.md` (full) — the design under review.
- `docs/decisions/ADR-011-comfyless-mcp-server.md` (full) — parent contract, §3a/§3b/§3e, the 2026-05-02 artifact-policy amendment, the deferred image-upload-as-seed oracle discussion.
- `comfyless/mcp_server.py:1-1005` — shipped baseline: `_handle_generate` (esp. steps 4-5 HF-resolve-before-allowlist at 533-579), `_handle_generate_cascade`, `_GENERATE_INPUT_SCHEMA`, null-byte gate (482-502), `_sanitize_error`, `_MCPHandlerError`, `_StartupConfig`, `_resolve_mcp_output_path`.
- `comfyless/server.py:158-189` — `_within`, `_check_paths` (reused verbatim per §1).
- `nodes/eric_diffusion_utils.py:89-154` — `_is_hf_repo_id`, `resolve_hf_path` (HF-source catalog entries route through these).
- `TECH_DEBT.md` Security section — the 2026-05-17 HF-cache enumeration-oracle finding the ADR's §2 oracle argument must reconcile with.

Not reviewed (and why):
- `comfyless/cascade.py`, `comfyless/params_validation.py` internals — referenced by the design but their contracts are not changed by ADR-015; out of scope for a design review of the reference model.
- HTTP-transport concerns — out of scope per the stated threat model (stdio same-uid); noted where relevant.

### Findings

#### [HIGH-1] §2 "miss-naming is not an enumeration oracle" contradicts an accepted live finding on the same surface
Location: ADR-015 §2 step 2 (Miss); reconcile with `TECH_DEBT.md` Security 2026-05-17 and the baseline ordering at `comfyless/mcp_server.py:533-579`.
Risk: The ADR asserts a catalog miss reveals nothing "beyond the catalog's intended public contract" because membership is discoverable via `list_models`/`list_loras`. That is true *only if every loadable thing is in the catalog and the error class is uniform across all rejection causes*. The shipped code already has a documented oracle (TECH_DEBT 2026-05-17): a reference can resolve to three distinguishable error classes — success, `PathAllowlist` (cached + outside base), `HFCacheMiss` (not cached) — letting the agent enumerate the HF cache independent of the catalog. ADR-015 §2 step 3 keeps request-time `resolve_hf_path` + `_within` for HF-sourced catalog entries, so a catalog *hit* whose path moved or whose HF entry is mid-eviction still produces a *different* error than a clean miss (`unknown <kind>`). The "single closed set, uniform `unknown` error" mental model the ADR relies on does not hold against the code it is layered on; the residual oracle survives the redesign rather than being closed by it.
Remediation: Add a clause to §2 step 2/3 committing to a *single agent-facing error class* for all reference-resolution failures (catalog miss, catalog hit whose path vanished/moved, HF-cache miss on an HF-sourced entry, request-time `_within` failure), with the fine-grained class retained only on the stderr audit line — and explicitly state this closes (not merely co-exists with) the 2026-05-17 TECH_DEBT oracle, marking that entry's trigger as met. Drop or heavily qualify the "strictly safer than slice-1" sentence until the uniform-error commitment is in the text.

#### [HIGH-2] Catalog construction (§1) does not specify scan-time symlink handling or name normalization — the new keystone's trust boundary is underspecified
Location: ADR-015 §1 Construction (Scan / Collision rule / build-time `_within`).
Risk: §1 says scan derives names "from the directory/file basename" and `_within`-checks every entry's `abs_path` at build time, with fail-closed on two distinct paths deriving the same name. Three gaps: (1) **Scan-time symlink following** — if the scan walks symlinks under `--model-base`, a symlink whose *target* is in-base but whose *link path* basenames to a different name silently mints a catalog entry; worse, the build-time `_within` is on the realpath'd target, so an in-base symlink pointing at another in-base weight creates a second name for the same file (benign) but a symlink chain crafted before spawn could shape the name→path map an operator did not intend. The ADR must state whether the scan resolves symlinks, skips them, or follows-then-`_within`-checks the link itself. (2) **Name normalization** — basename collision detection is byte-exact, but the host filesystem may case-fold (`Model.safetensors`/`model.safetensors`) or the manifest may supply Unicode names that NFC/NFD-normalize to collide with a scanned name; two names the operator believes distinct collapse to one lookup key, or two distinct files pass the byte-exact collision gate yet collide at OS-lookup time — a confused-deputy load. (3) The collision rule covers "scan yields two paths → same name" but not "scan-derived name == manifest-assigned name for a *different* path" — §1 says the manifest "augments and overrides," which is silent override, exactly the "pick one" the rule forbids for the scan.
Remediation: In §1 commit to: scan does not follow symlinks (or follows then `_within`-checks the realpath AND records the realpath as `abs_path`); names are normalized (NFC + an explicit case policy) before collision detection and lookup, both at build and at request normalization in §2; and the manifest-overrides-scan case is either (a) an explicit, audit-logged override of a *named* entry only, or (b) subject to the same fail-closed collision rule when a manifest name shadows a distinct scanned path. Name these as build-time invariants the slice-2 reviewer checks.

#### [MEDIUM-1] Request-time `_within` "fires differently for catalog hits whose path moved" is a residual TOCTOU oracle and a fail-mode the ADR should pin
Location: ADR-015 §2 step 3; §4 (request-time `_within` retained as defense-in-depth).
Risk: Between spawn-time catalog build and request-time load, a catalog entry's realpath can change (drive remount — explicitly cited as a motivation; operator file move; symlink swap). On such a request the catalog *hits* (name known) but request-time `_within`/`realpath` *fails*, producing an error distinct from a clean `unknown <kind>` miss — a timing/error-class signal that a name is known-but-currently-unloadable. Folded into the HIGH-1 uniform-error fix, but it also raises a fail-open/closed question §2 leaves open: on request-time `_within` failure for a catalog hit, does the server fail closed (reject) — it must.
Remediation: §2 step 3 states explicitly: request-time `_within`/realpath failure on a catalog hit fails *closed* with the same uniform error class as a miss; never falls back to the stale catalog path. One sentence.

#### [MEDIUM-2] `extract_params` basename-fallback (§3) leaks "a weight with this basename was used" — acceptable, but the ADR should bound it
Location: ADR-015 §3 extract_params reverse mapping (Miss → return basename + notice).
Risk: When a sidecar references a weight not in the catalog (e.g. a human-CLI direct load), the handler returns the bare basename. This confirms to the agent that *some* weight with that filename was used outside the catalog — minor layout signal (filename, never directory). For the stated threat model (same-uid desktop) this is acceptable, but the ADR asserts "a basename is not a filesystem-layout leak" as if it leaks nothing, which is slightly too strong: it leaks the existence of an off-catalog weight name. The sidecar itself is under `--output-dir` and was produced by the user's own runs, so the agent could read it directly anyway — which is the actual reason this is acceptable.
Remediation: Reword §3's justification to the real one (sidecar is user-produced under `--output-dir`, already agent-readable) rather than "a basename is not a leak," and note the alternative (drop the field) as the tightening available if the off-catalog-existence signal is ever judged material.

#### [MEDIUM-3] §4 deferred end-state should gate the request-time `_within` removal NOW with a precondition
Location: ADR-015 §4 End-state; Alternatives Rejected B.
Risk: The end-state promotes the catalog to sole path-authority and drops `--model-base` as the containment root. The ADR marks this "deferred, viable" but does not state the precondition that must hold before the request-time `_within` net is removed — namely that the uniform-error commitment (HIGH-1) and the symlink/normalization invariants (HIGH-2) are *implemented and tested*, not just designed. Removing `_within` while those are only ADR text would leave the catalog as a single point of trust with no second check.
Remediation: Add to §4 a hard precondition: the catalog-as-sole-authority amendment may not drop request-time `_within` until (a) catalog build-time symlink/normalization invariants are enforced with negative tests and (b) the uniform-error contract is shipped.

#### [INFO-1] Preserved ADR-011 invariants — verified intact in the design
Location: ADR-015 §2 step 3, §3, slice plan.
Confirmation: The design preserves: `allow_hf_download=False` (HF-sourced catalog entries still route through `resolve_hf_path(..., allow_download=False)`); audit-on-every-call (catalog names are loggable, §2 INFO notices are additive); traceback-strip (`_sanitize_error` unchanged); no-CLI-dispatch (catalog is in-process, additive to `_StartupConfig`); output containment (`output_path`/`savepath` explicitly unchanged, still `--output-dir`-bound). No regression of these in the design.

#### [INFO-2] `notices`/INFO message text must be scrubbed of raw agent input at implementation time
Location: ADR-015 §2 Hit notice; §3 extract notice.
Risk: The notices quote the resolved catalog *name* (intended-public), which is correct. The *supplied* path the agent sent must never be echoed in the notice (it could contain attacker-chosen directory text that round-trips into the agent transcript and onward).
Remediation: Add a one-line implementation note to slice 3: notices interpolate the resolved *catalog name* only, never the agent-supplied raw reference value.

#### [INFO-3] HTTP-transport note (out of scope for now)
Location: ADR-015 §4; ADR-011 §6.
Risk: All findings are calibrated to stdio same-uid. Under HTTP transport the error-class oracle (HIGH-1) escalates from a same-uid agent learning local layout to a network actor enumerating the host — but ADR-011 §6 gates HTTP behind a separate ADR.
Remediation: Carry HIGH-1's uniform-error contract forward as a hard precondition into any future HTTP-transport ADR.

### Round 1 verdict

**CHANGES REQUIRED.** No CRITICAL; the core direction (names not paths, server-side catalog, basename-strip + INFO notice) is a genuine net reduction in attack surface and is approved in shape. HIGH-1 and HIGH-2 must be folded into the ADR before slice-2 code is authored. Status stays `proposed` until fold-in; re-fire on amended text advised (HIGH-1 changes a load-bearing claim).

---

## Round 2

**Date:** 2026-05-23
**Reviewer:** `security-auditor` (Opus 4.7, 1M context), model pinned at invocation.
**Target:** ADR-015 as amended by the 2026-05-22 fold-in entry. Re-fire per round-1 advisory.

### Summary

Round-1 findings were addressed substantively, not just acknowledged. The fold-in commits to the uniform-error contract as a *load-bearing* invariant (the right framing), pins the catalog-construction trust boundary (no-follow-symlinks, NFC + case-sensitive-with-collision-rejection, manifest-shadows-scan fail-closed), nails request-time `_within` to fail-closed with the uniform error, reframes the `extract_params` basename-fallback to the correct justification, gates the end-state on shipped+tested preconditions, carries the INFO-2 notice-text scrubbing as an implementation note, and carries INFO-3 HTTP-transport precondition. One new gap surfaced in the fold-in itself: §1 does not pin whether the manifest may declare HF-repo-ID-sourced entries and how a build-time HF cache miss on such an entry is handled. Minor and tractable.

### Verification of round-1 fold-in claims

| Round-1 finding | Required commitment | Amended ADR location | Verdict |
|---|---|---|---|
| HIGH-1 | Single uniform agent-facing error class for ALL failure causes; fine-grained cause on stderr only; closes TECH_DEBT 2026-05-17 | §2 step 2 explicit; lists all four causes; names the stderr-only audit cause; explicitly closes the TECH_DEBT entry; explicitly *replaces* the slice-1 framing | Addressed |
| HIGH-2 | Scan no-follow-symlinks; NFC + case policy at build AND at request normalization; unified scan-and-manifest collision rule | §1 bullets 2–5 explicit on all three; manifest-shadows-distinct-scanned-path is fail-closed; §2 step 1 commits to same normalization at request side | Addressed |
| MEDIUM-1 | Request-time `_within` failure on catalog hit fails closed with the uniform error; never falls back to stale path | §2 step 3 explicit ("rejects... never falls back to the stale catalog path and never proceeds to load") | Addressed |
| MEDIUM-2 | Reword justification to user-produced-under-output-dir reason; name the drop-field tightening | §3 reworded to the correct reason and explicitly names the tightening available | Addressed |
| MEDIUM-3 | End-state gated on shipped + tested preconditions | §4 explicit ("not merely committed in ADR text"; "have negative-test coverage") | Addressed |
| INFO-2 | Notices interpolate resolved catalog name only, never agent-supplied raw input | §2 step 2 Hit notice parenthetical carries this as implementation note for slice 3 | Addressed |
| INFO-3 | HTTP-transport carry-forward of the uniform-error contract | §4 names it a hard precondition for any future HTTP-transport ADR | Addressed |
| INFO-1 | No change required; preserved-invariants confirmation | Folded as confirmation only in Changelog | Addressed |

All round-1 findings landed as text in the ADR, not just in the Changelog summary. Spot-checks on §2 step 2 (the load-bearing change) confirm the uniform-error contract is stated where the contract lives, not only summarized in the fold-in entry.

### New findings introduced by the fold-in

#### [MEDIUM-4] §1 underspecifies whether the manifest may declare HF-repo-ID-sourced entries; build-time behavior on HF cache miss is unstated
Location: ADR-015 §1 (manifest description; build-time `_within` clause); reconcile with `nodes/eric_diffusion_utils.py:110-142` (`resolve_hf_path`).
Risk: §1 says "every catalog entry's `abs_path` is `realpath`-resolved and `_within(--model-base)`-checked at catalog-build time" and that scan already routes "from the repo ID for already-cached HF entries — no network." It does not say whether the *manifest* may name an entry whose target is an HF repo ID, nor what happens at build time if a manifest declares one and that repo ID is not in the local HF cache. `resolve_hf_path(..., allow_download=False)` raises `ValueError` on cache miss; if the catalog build is silent on this, the cases are (a) fail startup, (b) drop the entry and continue (introduces a silent omission whose only signal is in stderr), or (c) defer resolution to request time (changes the §1 invariant). Each is defensible, but slice-2 code will pick one and the security-keystone ADR should pin it.
Remediation: One clause in §1 stating either (a) the manifest accepts HF repo IDs and a build-time cache miss fails startup with the entry name, (b) the manifest is local-paths-only and HF entries are scan-only, or (c) HF entries deferred and resolved at request time with the uniform-error contract handling the cache-miss case. Whichever; pin it before slice 2 writes code.

#### [INFO-4] Case-sensitive-with-case-insensitive-collision-rejection has a surprising shape on case-folding host filesystems (portability nuance, not current-host risk)
Location: ADR-015 §1 name-normalization clause.
Confirmation: The current host is Linux ext4 + mergerfs (per CLAUDE.md filesystem constraint), which is case-sensitive — the policy is invisible there. On a hypothetical macOS HFS+/APFS or NTFS host the rule is correct; the portability surprise is just that an operator porting the package to a case-folding host may hit fail-closed on names that "look fine" on their shell but collide under the strict policy.
Remediation: When slice 2 Vision is drafted, add a one-line note directing the operator to disambiguate via manifest on case-folding hosts. No ADR change needed.

### Round 2 verdict

**CHANGES REQUIRED — minor.** No CRITICAL, no HIGH. All round-1 findings addressed in the right places. One new MEDIUM-4 (manifest+HF-source unspecified) tractable in one clause; pin it before slice 2 authors code. INFO-4 is portability-only. If MEDIUM-4 is folded inline, the ADR is CLEAN to flip Status from `proposed` to `accepted` without a round 3.
