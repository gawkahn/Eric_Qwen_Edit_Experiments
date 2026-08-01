# ADR-044 — Krea-2 identity edit over the daemon wire (ADR-043 Part C)

**Status:** accepted

**Context:**

ADR-043 Parts A + B shipped the Krea-2 identity edit as a **CLI-foreground-only**
path, GPU-validated to 0.53/255 mean abs diff against the reference port. Three
facts keep it off the daemon today:

1. `ref_boost` and `grounding_px` are `COMFYLESS_SCHEMA` members, so the
   canonical validator (ADR-012) **accepts** them on the wire — but `server.py`
   never forwards them to `generate()`. That is an accepted-and-dropped
   parameter: precisely the silent failure invariant N1 exists to prevent.
2. `identity` has **no schema key at all**, by deliberate decision (Grant,
   2026-07-31: entry mode, not a generation parameter), so it has no wire
   representation. A delegated krea + `--ref-image` run therefore enters
   `generate()` with `identity=False` and takes the loud drop path — fail-closed,
   which is why Part B was safe to ship incomplete.
3. `_krea2_identity_forces_in_process` (`comfyless/generate.py:3781`)
   consequently keeps both cases in-process. The cost is real: an in-process load
   beside a warm daemon means two ~30 GB pipelines resident, which on a
   single-GPU box is the 2026-07-26 warm-daemon crash shape, not a graceful
   degrade. The measured-best `ref_boost` is 1.25 — non-default — so *every run
   that matters* currently takes that path.

`comfyless/server.py` and `comfyless/mcp_server.py` are both in
`scripts/git-policy/_red-zone-paths.sh`. This ADR precedes the code and
`docs/security/review-adr-044-identity-daemon-carriage-2026-08-01.md` reviewed
this design at draft, before any code was written (§12 order). **Three of that
review's findings changed the design below, and one of them removed work rather
than adding it** — the draft proposed a client-side warning-emission mechanism
that duplicated a wire channel already in place.

**Decision:**

Part C lands as **three commits in a fixed order**. The pipeline-layer hardening
is a precondition, not a cleanup: the cache-key decision in commit 2 is only
correct once commit 1 holds.

---

### Commit 1 — pipeline-layer state hygiene (precondition)

*Scope: `pipelines/krea2_identity_edit.py`, `test_krea2_identity.py`. Not a Red
Zone path, but it is what makes the Red Zone change safe.*

**Capture the stock attention processors BEFORE any swap, and apply inside the
`try`.** Today `apply_identity_processors` runs at `krea2_identity_edit.py:660`,
*outside* the `try:` at 669, with `origin` assigned only from its return value.
Two ways residue survives: `set_attn_processor` failing after partial
application, and `remove_identity_processors` itself raising inside the
`finally`. `pipelines/nag_krea2.py:392-394` already defends against exactly this
and says so in a comment — ADR-043 claimed to follow "the nag_krea2.py mold" and
on this point did not.

Under Part B that residue dies with the CLI process. Under Part C it persists in
the daemon's pipeline cache, and the failure is not loud: a *different* sequence
length crashes, but the **same** resolution — the common case in an `--iterate`
sweep — silently applies a wrong attention bias with frozen
`text_len`/`src_len`/`tgt_len`. That is corrupted output with no error, which is
exactly the leak commit 2's cache-key exclusion assumes cannot happen.

Also amend the `krea2_identity_edit.py:383-386` docstring: "adds no durable
instance state" is **false**. The VL processor memo
(`self._krea2_identity_vl_processor` / `_krea2_identity_vl_encoder_id`, set at
400-413) persists on what is, under the unbound call, the daemon's cached
`Krea2Pipeline`. It is inert — stock code never reads those attributes and the
memo is keyed on `(id(text_encoder), id(tokenizer))`, both pinned by the cache
key — but it is the one deliberate exception and must be named as such, because
the next author reasoning about the cached object will trust that docstring.

### Commit 2 — daemon wire carriage (RED ZONE: `comfyless/server.py`)

*Scope: `comfyless/params_validation.py`, `comfyless/generate.py`,
`comfyless/server.py`, `test_params_schema.py`, `test_ref_edit.py`,
`test_server_robustness.py`.*

1. **`identity` becomes a wire field in `_RUNTIME_KIND`, not `SCHEMA_KIND`.**
   `_RUNTIME_KIND` (`comfyless/params_validation.py:128`) is the established home
   for wire-only, non-sidecar-shaped fields — `ref_dims_explicit`,
   `ref_drop_strict`, `report_roots`, `output_format`. Registering `identity`
   there gives it a canonical bool type check at the machine boundary **while
   preserving the 2026-07-31 decision exactly**: no `COMFYLESS_SCHEMA` key, no
   sidecar record, no `--params` replay. The runtime-only allowlist drift guard
   in `test_params_schema.py` stands unchanged.

   Registration widens acceptance nowhere — unknown keys already pass through
   (`params_validation.py:422-437`); it only adds a type check. **Absent =
   `False`** (fail-closed), matching `generate()`'s own default
   (`generate.py:2078`) and the `ref_drop_strict` absent-is-strict precedent. The
   client sends the field only when the flag is set, so a plain request stays
   byte-identical to today's.

2. **`ref_boost` / `grounding_px` ride the wire as ordinary schema params** —
   sourced from the merged params in `_build_server_request` (the NAG quadruple's
   shape, since they are sidecar-replayable for the same reason: they change
   output CONTENT) — and `server._handle_generate` forwards all three to
   `generate()`.

3. **`_build_server_request` has two callers, and the second one is easy to
   miss.** `comfyless/refine.py:1881` builds a synthetic `argparse.Namespace`
   carrying "just the attributes `_build_server_request` reads," and it has no
   `identity`. Reading `args.identity` directly would raise `AttributeError` on
   *every* refine daemon generation — the identical latent break ADR-034 slice 5
   shipped and had to patch on this exact path. The read is
   `getattr(args, "identity", False)`, and a test pins the builder against
   `_daemon_namespace`'s attribute set.

   Refine's resulting behaviour is correctly fail-closed: it cannot express
   `--identity`, sends nothing, the daemon defaults to False, and refine's forced
   `ref_drop_strict=True` (`refine.py:1984`) turns a krea reference into a hard
   error rather than a silent drop.

4. **All three stay OUT of `_request_cache_key`.** They select OUTPUT, not
   pipeline shape — the NAG (ADR-023) and `output_format` (ADR-034) precedent.
   This includes `identity`: it changes which `__call__` runs, not what is
   loaded. The loaded object is a stock `Krea2Pipeline` either way, because
   `Krea2IdentityEditPipeline.__call__` runs **unbound** on it (ADR-043's
   load-bearing mold). Keying on any of the three would evict and reload ~30 GB
   on every `ref_boost` tweak — exactly the sweep an `--iterate` loop performs.

   This is safe **only given commit 1** plus (5) below. Per-call attributes
   (`_guidance_scale`, `_interrupt`, `_num_timesteps`, scheduler timesteps) are
   re-set by every stock call, and `_bias_cache` lives on the per-call processor
   objects and is removed with them.

5. **The daemon fails closed on identity residue.** When `generate()` raises out
   of a request that had `identity=True`, `_handle_generate` inspects
   `transformer.attn_processors` for surviving `Krea2IdentityEditAttnProcessor`
   instances and **evicts the cache entry** if any are found, before returning
   the `InferenceError`. Today it returns without evicting
   (`server.py:1091-1107`). A residue check is chosen over an unconditional
   evict: it is cheap, precise, and does not force a 30 GB reload after an
   ordinary OOM.

6. **Retire both forcings; `_krea2_identity_forces_in_process` is deleted.**
   Identity runs delegate like any other run once the wire carries all three.

7. **Delete the client-side no-op warning at `generate.py:4743-4755` — do not add
   more client-side emission.** The draft of this ADR proposed relocating every
   identity warning client-side on the theory that a delegated run's warnings die
   in the daemon's stderr. That premise is **half-false**, and the review caught
   it: `edit_warnings` is already a wire warning channel
   (`generate.py:1390-1395`), every identity notice is already appended to it
   (2254 / 2275 / 2299 / 2344), and `surface_wire_warnings` already prints them
   on the client's stderr after a delegated run (4030-4033).

   So N1 is already satisfied on the success path by an existing mechanism, and
   client-side emission would *duplicate* every notice. The one client-side print
   that exists today is the `--identity`-with-no-refs no-op, and it exists
   **only** because `identity` could not reach the daemon at all. Once it can,
   the daemon emits that same notice into `edit_warnings` and the client-side
   copy becomes the double-print. Part C therefore removes it. Net: Part C adds
   no warning plumbing and deletes some.

   **Residual, accepted:** warnings are still lost when a delegated run *errors*,
   because error responses carry no `metadata` (`server.py:1103-1105`). An
   errored run surfaces its error, so the lost warning is not the user's only
   signal. TECH_DEBT entry; trigger = the second family that needs a warning on a
   failed delegated run.

### Commit 3 — MCP surface hardening (RED ZONE: `comfyless/mcp_server.py`)

*Scope: `comfyless/mcp_server.py`, `test_mcp_server.py`.*

**The MCP surface is NOT closed by construction, contrary to this ADR's draft.**
`@app.call_tool(validate_input=False)` (`mcp_server.py:2901`) is deliberate —
invariant 5 requires every invocation to emit an audit line, and the framework's
default validation short-circuits before the handler runs — but it means
`_GENERATE_INPUT_SCHEMA` is **advertisory, not enforced**. An agent-sent
`ref_boost` / `grounding_px` / `ref_images` passes `validate_machine_request`
(all three are `SCHEMA_KIND`) and passes the payload filter into `gen_params`
(2133). The *only* thing stopping it is that the explicit-kwargs `generate()`
call at 2161-2198 happens not to forward those names — one unpinned layer, and a
silent accept-and-drop on an agent surface, which is the same N1 failure this ADR
condemns on the daemon wire.

The `ref_images` case is the sharp one. It reaches `gen_params` fully validated,
and the MCP plane has **no counterpart to the daemon's `_check_ref_paths`
containment** (`server.py:508-523`) because it was never supposed to accept
references. A future maintainer converting that call to `generate(**gen_params)`
— a natural cleanup — opens agent-supplied absolute paths to in-process decode
and VAE-encode of any user-readable file, with nothing to fail.

So:

1. **Add `ref_images`, `ref_boost`, `grounding_px`, `identity` to
   `_GENERATE_REMOVED_FIELDS`** (`mcp_server.py:90-95`), so they are rejected
   loudly at the handler — the `vae_path` precedent, whose comment already
   states the rule: silently accepting a raw path "would reintroduce the
   caller-supplied-path input attack surface ADR-015 removes." Field names are
   public schema knowledge, so naming them in the error leaks nothing.
2. **Strip `ref_boost` / `grounding_px` from `_render_extracted_params`**
   (outbound, alongside the existing `ref_images` drop at `mcp_server.py:391`).
   These ship **together**: rejecting inbound without stripping outbound would
   make an innocent replay loop — agent calls `extract_params`, echoes the result
   to `generate` — fail on keys we handed it.
3. **Tripwires pin the call site, not just the schema.** The draft's proposed
   test (four names absent from `_GENERATE_INPUT_SCHEMA`) would pass while the
   surface stayed reachable. The real invariant is that the MCP `generate()` call
   site forwards none of the four, D2a-shaped.

**Opening the MCP surface to references remains out of scope, and that is a
decision, not a deferral.** It needs a reference-image *handle* scheme first:
ADR-015's rule is catalog NAMES, never absolute paths, in both directions, and
reference images are arbitrary user files with no catalog to launder through —
which is exactly why ADR-035 drops `ref_images` on the way out. That is its own
ADR.

**Alternatives Rejected:**

- **Give `identity` a `COMFYLESS_SCHEMA` key to get it on the wire.** Reverses
  Grant's 2026-07-31 decision, and would put it in every sidecar and make it
  `--params`-replayable ("a sidecar consumer doesn't care that the image was
  generated with `--identity`"). Unnecessary: `_RUNTIME_KIND` already carries
  wire-only fields, which is what this is.
- **Put the three in `_request_cache_key` "to be safe."** Not conservative —
  wrong. None changes pipeline shape, and the cost is a 30 GB evict + reload per
  tweak, on the exact loop the tuning workflow runs. The genuine hazard the
  instinct is reaching for is state residue, and residue is addressed at its
  source (commits 1 and 2.5), not by defeating the cache.
- **Client-side pre-delegation warning emission.** The draft's decision 5;
  removed. `edit_warnings` already does this, and adding a second path would
  double-print every notice — and the natural "fix" for that (suppressing the
  daemon-side appends) would break the sidecar record and refine's surfacing.
- **A general daemon→client notices channel for the error path.** The complete
  fix for the accepted residual, and the right eventual one, but it widens a Red
  Zone wire schema and every other family's warnings would then need auditing for
  path disclosure, since the daemon knows absolute paths the client may not.
  TECH_DEBT, not a Red Zone slice rider.
- **Hold `mcp_server.py` scope and pin the call site with tests only.** Tempting
  — it keeps commit 3 test-only. Rejected because Finding 5 is an *absence*
  finding: the containment gate the MCP plane lacks is exactly what makes a
  future refactor dangerous, and a test that pins today's kwargs does not stop
  someone who edits the kwargs. Rejecting at entry does.
- **A `ping` capability handshake for version skew.** A pre-Part-C daemon ignores
  `identity: true` and the run degrades to the drop path (hard error for machine
  callers, warn-and-drop surfaced via `edit_warnings` for a TTY). Not silent. A
  wire-schema widening on a Red Zone surface is not worth it for a same-UID,
  single-operator daemon whose operator restarts it themselves. Accepted residual.

**Deferred / Out of Scope:**

- MCP identity carriage, and the reference-image handle scheme it depends on.
- A general daemon→client notices channel for the error path (TECH_DEBT).
- ADR-043 D11 catalog carriage of the empirically-earned `ref_boost=1.25`.
- `--params` replay re-entering identity mode — unchanged from ADR-043, working
  as designed.
- `NaN` `ref_boost` reaching `math.log(max(ref_boost, 1e-4))`
  (`krea2_identity_edit.py:280`) — `json.loads` accepts `NaN`. Pre-existing and
  family-wide: it is equally a `nag_*` fix, so it is not an identity slice.
  TECH_DEBT.
- Moving the >2-source refusal client-side so it fires before the ~30 GB load
  (`generate.py:1993-1999`). A latency nicety, not a safety property; the daemon
  stays the authoritative gate either way. TECH_DEBT.
- Any `FAMILY_DEFAULTS` change.

**Proof hooks (each promise gets a negative case):**

| Promise | Negative case |
|---|---|
| The wire carries all three | A delegated identity run is bit-identical to the in-process run at the same seed / `ref_boost` / `grounding_px` |
| Absent `identity` = off | A wire request with no `identity` field takes the drop path, not the identity path |
| Not a generation parameter | `identity` absent from `COMFYLESS_SCHEMA` **and** `SCHEMA_KIND`; present in `_RUNTIME_KIND` |
| Cache key unaffected | `_request_cache_key` byte-equal across differing `ref_boost` / `grounding_px` / `identity` |
| **Cache hygiene (the one the design rests on)** | **After a delegated identity run FAILS, the next non-identity request from the same daemon is byte-identical to a fresh-daemon run** |
| Restore survives a partial swap | Force `remove_identity_processors` to raise; assert the next daemon request does not reuse the pipeline |
| N1 holds without new plumbing | Every identity warning reaches the client's stderr via `edit_warnings` on a delegated run |
| No double-print | Each notice appears exactly once — in-process AND delegated |
| refine unbroken | `_build_server_request` works against `_daemon_namespace`'s attribute set |
| Non-krea untouched | A delegated flux2 / qwen-edit reference run sends no `identity` and emits no identity notice |
| MCP unreachable | The MCP `generate()` call site forwards none of the four; a payload carrying them is REJECTED, not dropped |

**Changelog:**

- 2026-08-01 — ADR written before code, per §12. Security review:
  `docs/security/review-adr-044-identity-daemon-carriage-2026-08-01.md`
  (`security-auditor`, Fable, 73/73 turns on `claude-fable-5`, no fallback).
  **Three findings changed this document before any code was written:** the
  "closed by construction" MCP claim was false (`validate_input=False`), the
  identity processors install outside the `try` where NAG captures before the
  swap, and the proposed client-side warning relocation duplicated the existing
  `edit_warnings` channel — that decision was replaced by a deletion. No finding
  was rejected.
- 2026-08-01 — **Commit 3 landed; three deviations from the text above.**
  Reviews: `docs/security/review-adr-044-commit3-mcp-hardening-2026-08-01.md`
  (`security-auditor`, Fable, no fallback) plus a parallel `code-reviewer`
  (Fable, no fallback).

  1. **A NEW constant, not an extension of `_GENERATE_REMOVED_FIELDS`.** Clause
     1 above says "add … to `_GENERATE_REMOVED_FIELDS`" — **do not do that; the
     implementation is `_GENERATE_UNSUPPORTED_REF_FIELDS`.** That tuple's
     rejection message reads "reference weights by catalog name (see list_models
     / list_transformers)", which for `ref_boost=1.25` is actively wrong
     guidance on an agent surface: there is no catalog name to reach for, the
     surface simply does not exist. Two tuples with honest messages beat one
     tuple with a misleading one. Consolidate to a field→reason mapping only if
     a THIRD rejection category ever appears.
  2. **The cascade branch is closed too** — not in the text above, and it had to
     be. The `generate` tool routes to `_handle_generate_cascade` on
     `cascade_config` presence BEFORE `_handle_generate`'s guards, so this ADR's
     own proof hook ("a payload carrying them is REJECTED, not dropped") was
     false on half the surface it claims. `_GENERATE_UNSUPPORTED_REF_FIELDS` is
     now checked at cascade entry as well. `_GENERATE_REMOVED_FIELDS` shares the
     same bypass and was deliberately NOT hoisted — that is a behaviour change
     for a caller shape this ADR never touched, so it is TECH_DEBT.
  3. **The call-site tripwire is an AST guard, not a substring.** The first cut
     (`"generate(**" not in <source slice>`) is defeated by a line wrap, and it
     is the only automated barrier protecting fields that are NOT in a rejection
     tuple. It now walks the AST for a `**` starred kwarg and for each closed
     name, with a premise check that exactly one `generate()` call exists.

  **Left open by commit 3, and NOT part of this ADR's decision:** the same
  review found that `upscale_vae_path` / `upscale_vae_subfolder` /
  `refiner_path` — path-typed `COMFYLESS_SCHEMA` members with no relation to the
  identity edit — take the identical accept-and-drop route inbound AND survive
  `_render_extracted_params` outbound as verbatim absolute paths, contradicting
  that function's own "no absolute path survives" docstring. Different fields,
  different lineage (ADR-030 / ADR-016), pre-existing.

- 2026-08-01 — **Grant decided to close that leak immediately, in its own
  commit** rather than defer it, after the disclosure was measured (two
  `/home/gawkahn/...` paths crossed the agent boundary on a sidecar recording
  either field). Shipped alongside this ADR's three commits and pushed in the
  same batch. Not an ADR-044 decision — recorded here only because this
  Changelog is where the deferral was written down, so this is where its
  resolution belongs. Full detail:
  `docs/security/review-mcp-path-leak-close-2026-08-01.md`.

  Two things from that fix are worth carrying forward into any future work on
  this surface, because both are general:
  - **A field rejected on PRESENCE must also be absent from the generate
    RESPONSE.** `resolved_params` is the agent's authoritative record, so
    echoing it back is the obvious replay loop, and an empty string does not
    spare a presence check. The pop list is now sourced from the rejection
    tuples themselves so the two cannot drift apart.
  - **`generate()` records `ref_boost` / `grounding_px` / `upscale_vae_*`
    UNCONDITIONALLY**, not only when non-default. An earlier review asserted the
    opposite and reasoned from it; the code comment says otherwise in as many
    words. Verify this class of claim against the metadata block, not a review.

- 2026-08-01 — **Commit 2 landed; review findings folded in.** Reviews:
  `docs/security/review-adr-044-commit2-wire-carriage-2026-08-01.md`
  (`security-auditor`, Fable, no fallback) plus a parallel `code-reviewer`
  (Fable, no fallback). Four changes to what this ADR specified:

  1. **`server.py` gates `identity` with `is True`, not `bool()`.** The
     `_RUNTIME_KIND` registration is the only thing making the field bool-only,
     and `validate_machine_request` passes unknown keys through unchanged — so
     de-registering or renaming that entry would let `identity: "no"` ENABLE the
     mode under a truthiness gate. Identity comparison fails closed regardless of
     registration state. This is the `report_roots` precedent in the same file.
  2. **The residue check LOGS its own failure** instead of swallowing silently.
     It is a fail-open backstop, and what it fails open *into* is the
     silent-wrong-bias case it exists to prevent; this daemon's own precedent is
     that an accepted residual is tolerable only if observable.
  3. **Success-path residue checking was considered and NOT taken.** The
     `code-reviewer` found a regression the error-path check cannot see: wrapping
     `remove_identity_processors` in `try/except` inside the pipeline's `finally`
     — a plausible hardening edit — turns a failed restore into a SUCCESS
     response with residue installed, and every suite stays green. Checking
     residue on the success path too would close it structurally for ~3 lines.
     Rejected anyway: adding un-reviewed code to a Red Zone path *after* its
     security review is the thing to avoid, and an AST guard (the restore must
     not sit inside a nested `try`, so it can still raise) closes the named
     regression at zero Red Zone cost. Mutation-verified — the refactor leaves
     every pre-existing guard leg passing and trips only the new one. Revisit if
     a second reason to inspect the success path ever appears.
  4. **Test-file scope substitution.** This ADR declared `test_params_schema.py`;
     the validator negatives went to `test_machine_boundary_validator.py`
     instead — the natural home for `validate_machine_request` contracts — and
     `test_params_schema.py` is untouched, its Part-B `identity` allowlist entry
     standing unchanged exactly as promised above.

  Also noted by the auditor and adopted: **commits 2 and 3 push as ONE batch**,
  so the interim window where the MCP surface still accept-and-drops the three
  fields never exists on the remote.

- 2026-08-01 — **Grant accepted the `mcp_server.py` scope extension in commit 3**
  (reject-at-entry over hold-scope-with-tests), making Part C a two-Red-Zone-file
  slice rather than one. The deciding argument was Finding 5's shape: it is an
  *absence* finding — the MCP plane has no `_check_ref_paths` counterpart — and a
  test that pins today's call-site kwargs does not stop an author who edits those
  kwargs. Status → accepted; implementation may begin.

**AI-Disclosure:** Claude (Opus 5) authored; Claude (Fable 5) reviewed the design
via `security-auditor`; Grant reviewed.
