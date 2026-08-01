# Security review — ADR-044 Krea-2 identity edit over the daemon wire (Part C)

**AI-Disclosure:** Claude (Fable 5) authored the review via the `security-auditor`
subagent; Claude (Opus 5) commissioned it and verified its load-bearing findings
against the code; Grant reviewed.

**Date:** 2026-08-01
**Design under review:** `docs/decisions/ADR-044-krea2-identity-daemon-carriage.md`
(reviewed at draft, BEFORE any code — §12 order)
**Red Zone surfaces:** `comfyless/server.py` (Unix-socket IPC daemon),
`comfyless/mcp_server.py` (LLM-agent tool surface)

**Reviewer model provenance:** the agent transcript records 73/73 assistant turns
on `claude-fable-5`. **No Fable→Opus fallback occurred.**

**Independent verification by the commissioning session** — the four load-bearing
findings were re-checked directly against the code before the ADR was amended:

| Finding | Claim | Verified |
|---|---|---|
| 1 | `@app.call_tool(validate_input=False)` — input schema not enforced | ✅ `mcp_server.py:2901` |
| 2 | Identity processors applied OUTSIDE the `try`; NAG captures before the swap | ✅ `krea2_identity_edit.py:660-669` vs `nag_krea2.py:392-394` |
| 3 | `edit_warnings` already rides the wire and is surfaced client-side | ✅ `generate.py:2226/2254/4030`; error responses carry no `metadata` (`server.py:1103-1105`) |
| 4 | `refine._daemon_namespace` has no `identity` attribute | ✅ `refine.py:1881-1893` |

---

## Summary (reviewer's words)

> The core wire design is sound — `identity` is genuinely fail-closed at every
> caller, the reference containment gate is not bypassable by the identity path,
> and the cache-key exclusion is correct on pipeline-shape grounds. But two of
> the ADR's supporting claims are false as written: the MCP input schema is **not
> enforced at runtime** (the "two layers" are really one unpinned call site), and
> the identity pipeline module **does** leave durable state on the cached
> instance and installs its processors **outside** the try/finally, regressing
> from the NAG mold it cites. Both must be corrected before the code slice.

Threat model as stated by the reviewer: the daemon socket is a same-UID trust
boundary (ADR-001) whose validated request dict drives file reads (reference
images), model loads, and a long-lived shared pipeline cache; the MCP server is
an LLM-agent boundary where a hostile or confused agent supplies arbitrary tool
arguments. The two highest-stakes properties are (1) that a per-call mode cannot
leave residue on the cached pipeline serving subsequent unrelated requests, and
(2) that the MCP surface's "closed by construction" claim is actually structural.

## Coverage

Reviewed in full: `ADR-044`, `ADR-043`, `comfyless/params_validation.py`,
`comfyless/server.py`, `pipelines/krea2_identity_edit.py`. Reviewed in part:
`comfyless/generate.py` (lines 1390-1460, 1700-2410, 3300-3420, 3550-4100,
4700-4900), `comfyless/mcp_server.py` (80-120, 300-410, 655-780, 1680-1770,
1855-2330, 2880-2905), `comfyless/refine.py` (1855-2104),
`pipelines/nag_krea2.py` (install/restore sites).

Explicitly NOT reviewed, with the reviewer's reasons: `_load_ref_pils` byte/pixel
cap internals (ADR-035 territory, unchanged by this design); diffusers
`set_attn_processor` internals (third-party — the partial-application hazard is
inferred from this repo's own `nag_krea2.py:392-394` comment, which asserts it);
test suites and the epic Vision doc (background, not the change surface);
`comfyless/cascade.py` beyond the MCP dispatch shown (cascade cannot reach
`generate()`'s identity path).

## Verification of the six design claims

1. **Accepted-and-dropped today — TRUE.** `ref_boost`/`grounding_px` are
   `SCHEMA_KIND` (`params_validation.py:113-114`), so `validate_machine_request`
   accepts them on the wire, and `server._handle_generate`'s `generate()` call
   (`server.py:1023-1090`) forwards neither them nor `identity`.
2. **Wire plan viable — TRUE**, subject to Finding 4 (refine).
3. **Cache-key exclusion sound — TRUE on shape grounds.** `_request_cache_key`
   (`server.py:531-582`) excludes all three; `_load_pipeline`'s arguments
   (`server.py:715-731`) are unaffected; the identity edit swaps the *callable*
   (`krea2_identity_edit.py:769-778`), not the loaded object. **But the exclusion
   is only safe if per-call state provably cannot persist — see Finding 2.**
4. **Forcing exists as described — TRUE** (`generate.py:3781-3836`, consulted at
   `4738-4742`).
5. **"Warnings are lost on the delegated path" — PARTLY FALSE.** See Finding 3.
6. **"MCP closed by construction" — FALSE at runtime.** See Finding 1.

---

## Findings

### [HIGH] 1 — The MCP input schema is advisory, not enforced; decision 6's "closed by construction" claim is false

**Location:** `comfyless/mcp_server.py:2901` (`@app.call_tool(validate_input=False)`),
`1710-1713`, `2131-2136`, `2161-2198`

The ADR stated the call is rejected by `_GENERATE_INPUT_SCHEMA`'s
`additionalProperties: False` before the `set(COMFYLESS_SCHEMA)` filter. The
framework is explicitly registered with `validate_input=False` (deliberately —
invariant 5 requires every invocation to emit an audit line, and the framework's
default short-circuits before the handler runs). So no schema enforcement exists
on the call path.

An agent-sent `ref_boost` / `grounding_px` / `ref_images` passes
`validate_machine_request` (all three are `SCHEMA_KIND`), passes the payload
filter into `gen_params` (`mcp_server.py:2133`), and is stopped **only** by the
explicit-kwargs `generate()` call at `2161-2198`, which happens not to forward
them. `identity` passes as an unknown, type-checked key and is likewise silently
dropped. The proof hook "a payload carrying them is rejected" would be satisfied
by nothing, and the single actually-load-bearing layer is unpinned by any test.
This is itself a silent accept-and-drop on an agent surface — the exact N1
failure the ADR's own context condemns on the daemon wire.

**Remediation:** correct the ADR text (one layer, not two, and it is the call
site); add `ref_images`, `ref_boost`, `grounding_px`, `identity` to
`_GENERATE_REMOVED_FIELDS` (`mcp_server.py:90-95`) so they are rejected loudly at
the handler, matching the `vae_path` precedent. This is a deliberate, minimal
`mcp_server.py` behaviour change and must be declared in the ADR's edit scope
(the draft promised "no `mcp_server.py` behaviour change"). If scope must hold
instead, the tripwires must pin the `generate()` **call-site kwargs**, not schema
key absence, and the proof hook must be rewritten to what is true.

**Disposition:** ACCEPTED. ADR-044 decision 6 rewritten; the removed-fields
rejection is in scope, paired with the outbound strip in Finding 8.

### [HIGH] 2 — Identity processor install runs outside the try/finally, and the daemon keeps a possibly-corrupted pipeline cached

**Location:** `pipelines/krea2_identity_edit.py:660-669` vs
`pipelines/nag_krea2.py:392-398`; `comfyless/server.py:1091-1107`

This is the question-C hazard, and it is real. `apply_identity_processors`
executes *before* the `try:`, and `origin` is only assigned from its return
value. If `set_attn_processor` fails after partial application, nothing restores;
if `remove_identity_processors` in the `finally` itself raises, the installed
processors also persist. The NAG module defends against exactly this — "Capture
the stock processors BEFORE any swap so the finally below can restore even from a
partially-applied set_attn_processor" — and the identity module, which ADR-043
says follows "the nag_krea2.py mold," does not.

Under Part B the corrupted state dies with the process. Under Part C it persists
in the daemon: `_handle_generate` returns `InferenceError` **without evicting**
(`server.py:1091-1107`), so the next request served from the same cache entry
runs with stale `Krea2IdentityEditAttnProcessor`s carrying frozen
`text_len`/`src_len`/`tgt_len`. A different sequence length crashes loudly; the
**same** resolution — the common case in an `--iterate` sweep — silently applies
a wrong attention bias. Corrupted output, no error: precisely the leak the
cache-key exclusion assumes impossible.

**Remediation:** before delegation ships, (a) mirror NAG — capture `origin`
before the swap, apply inside the `try`; (b) require the daemon to fail closed on
identity-path residue: evict the cached pipeline when `generate()` raises out of
an identity request, or verify `transformer.attn_processors` types on the error
path. Negative test: force `remove_identity_processors` to raise, assert the next
daemon request does not reuse the pipeline.

**Disposition:** ACCEPTED as a hard precondition. Promoted to ADR-044 decision 1
— the pipeline-layer fix lands FIRST, in its own commit, before any wire change.
Residue check chosen over unconditional evict (cheaper and more precise).

### [MEDIUM] 3 — Decision 5's premise is half-false; as specified it double-prints every delegated identity warning

**Location:** `comfyless/generate.py:1390-1395` (`edit_warnings` is a wire warning
channel), `4030-4033` (`_delegate_to_server` surfaces it client-side),
`2341-2345` / `2249-2255` / `2293-2300` (identity warnings are appended to
`edit_warnings`), `refine.py:2037-2038`

The ADR asserted delegated-path identity warnings "are emitted in the daemon's
process, whose stderr is not the user's terminal" and are therefore lost. But
every one of those warnings also rides `edit_warnings` in the wire metadata,
which `surface_wire_warnings` already prints on the client's stderr after a
successful delegated run — ADR-043's own Changelog confirms this end-to-end.
Warnings are genuinely lost only when the run **fails** (error responses carry no
metadata).

Adding unconditional client-side pre-delegation emission would therefore
duplicate every notice on the success path — including the no-refs no-op notice,
which the client already prints at `generate.py:4753-4755` and which the daemon
will *also* emit into `edit_warnings` once `identity` rides the wire. That
contradicts the ADR's own "No double-print" proof hook, and the likely
implementer "fix" — suppressing the daemon-side `edit_warnings` appends — would
break the sidecar record and refine's surfacing: an N1 regression.

**Remediation:** specify the dedupe explicitly before code, or drop the
client-side emission.

**Disposition:** ACCEPTED, resolved by deletion rather than dedupe. ADR-044
decision 4 now REMOVES the client-side no-op print at `generate.py:4743-4755` —
it existed only because `identity` could not reach the daemon. Once the wire
carries it, the existing `edit_warnings` channel covers every identity notice
with no new plumbing. Residual (warnings lost on the ERROR path) accepted and
recorded in TECH_DEBT.

### [MEDIUM] 4 — `_build_server_request` gains new attribute reads; refine's synthetic Namespace does not supply them

**Location:** `comfyless/refine.py:1881-1893` (`_daemon_namespace`), `1872-1880`
(the documented ADR-034 precedent for exactly this break), `generate.py:3637`

Refine reuses the canonical wire builder through a minimal `argparse.Namespace`
carrying "just the attributes `_build_server_request` reads." It has no
`identity` attribute. If Part C reads `args.identity` rather than
`getattr(args, "identity", False)`, every refine daemon generation raises
`AttributeError` — the same latent break ADR-034 slice 5 shipped and had to patch
on this exact path. The ADR named neither refine nor `_daemon_namespace` as a
wire-builder caller. (`ref_boost`/`grounding_px` are sourced from merged params
`p`, which refine supplies, so only `identity` is exposed.)

The resulting behaviour is correctly fail-closed: refine cannot express
`--identity`, sends nothing, the daemon defaults to False, and refine's forced
`ref_drop_strict=True` (`refine.py:1984`) turns a krea reference into a hard
error — never a silent drop.

**Remediation:** ADR names both callers; implementation uses a defaulted
`getattr`; a test pins that `_build_server_request` works against
`_daemon_namespace`'s attribute set.

**Disposition:** ACCEPTED in full.

### [MEDIUM] 5 — Inbound `ref_images` reaches `gen_params` on the MCP plane, where no reference-containment gate exists

**Location:** `comfyless/mcp_server.py:2133` (the filter passes it — `ref_images`
∈ `COMFYLESS_SCHEMA`), `2161-2198` (only the call site drops it); contrast
`comfyless/server.py:508-523` (`_check_ref_paths` is daemon-only)

An absence finding. The daemon plane's containment (`ref_image_roots`) has no MCP
counterpart, because the MCP surface was never supposed to accept references —
but the payload filter does not exclude `ref_images`, so it sits validated inside
`gen_params` at the call site. A future maintainer converting the call to
`generate(**gen_params)` — a natural cleanup — would silently open agent-supplied
absolute paths to in-process decode / VAE-encode of any user-readable file, with
no `_check_ref_paths` equivalent and no test to fail. The ADR's proposed
schema-absence tripwires would all still pass.

**Remediation:** same one-line fix as Finding 1 (`_GENERATE_REMOVED_FIELDS`),
plus a tripwire pinning that the MCP `generate()` call site passes no
`ref_images` / `identity` kwarg.

**Disposition:** ACCEPTED. This is the strongest argument for taking the
`mcp_server.py` scope extension rather than holding scope.

### [LOW] 6 — "Adds no durable instance state" is false: the VL processor memo persists on the cached pipeline

**Location:** `pipelines/krea2_identity_edit.py:383-386` (the claim), `400-413`
(`self._krea2_identity_vl_processor` / `_krea2_identity_vl_encoder_id` set on
`self`, which under the unbound call IS the daemon's cached `Krea2Pipeline`)

Behaviourally benign today — stock code never reads these attributes, the memo is
keyed on `(id(text_encoder), id(tokenizer))`, and both components are pinned by
the cache key. But ADR-044's cache-key argument leans on a "no durable instance
state" invariant that is not literally true, and a future author trusting the
docstring will mis-reason about the cached object. (The reviewer notes this
review nearly did.)

**Remediation:** amend the docstring and the ADR's decision wording to name the
memo as the one deliberate, inert exception. No code-path change needed.

**Disposition:** ACCEPTED — docstring + ADR wording amended.

### [INFO] 7 — Version skew: a pre-Part-C daemon silently ignores `identity: true`

**Location:** `params_validation.py:433-437` (unknown keys pass through),
`server.py:1023-1090` (not forwarded)

An updated client against a stale daemon degrades to the drop path — a hard error
for machine callers, warn-and-drop for a lenient TTY, surfaced via
`edit_warnings`. Not silent, but run behaviour depends on daemon version with no
client-side detection. Consider a capability field on the `ping` response, or
record it as an accepted residual.

**Disposition:** ACCEPTED AS RESIDUAL, recorded in ADR-044 and TECH_DEBT. A
capability handshake is a wire-schema widening on a Red Zone surface and is not
worth it for a same-UID, single-operator daemon the operator restarts themselves.

### [INFO] 8 — `extract_params` returns `ref_boost`/`grounding_px` to the agent, which the generate tool then silently drops

**Location:** `comfyless/mcp_server.py:390-392` (drop list excludes them; they are
schema members and survive)

A replay loop hands the agent parameters it cannot use — the same
accepted-and-dropped shape, contained to the MCP notices channel. The
`_GENERATE_REMOVED_FIELDS` fix converts this into a loud contract error;
alternatively drop the two scalars in `_render_extracted_params`.

**Disposition:** ACCEPTED — **both**, and they must ship together. Rejecting
inbound without stripping outbound would make an innocent replay loop fail; the
pair keeps the surface coherent.

### [INFO] 9 — Minor daemon-side robustness notes on the new wire values

**Location:** `pipelines/krea2_identity_edit.py:280`
(`math.log(max(ref_boost, 1e-4))` — `json.loads` accepts `NaN`, which propagates
into the bias: single-request output corruption, no crash, same status quo as
`nag_*`); `generate.py:1993-1999` (the 3-reference hard error fires in the daemon
**after** the ~30 GB load; the client could pre-check the count for free).

**Disposition:** NOTED, both out of scope. The NaN path is pre-existing and
family-wide (it would be a `nag_*` fix too, not an identity one); the 3-ref
pre-check is a latency nicety, not a safety property, and the daemon remains the
authoritative gate either way. TECH_DEBT entries.

---

## Reviewer's answers to the commissioning questions

**A — does `_RUNTIME_KIND` registration expose `identity` unintentionally?**
`_RUNTIME_KIND` is consumed only through `_ALL_FIELDS` in
`validate_machine_request` (`params_validation.py:211-212, 434`), whose three
callers are `server._validate_request` (`server.py:156`), MCP `_handle_generate`
(`mcp_server.py:1950`), and MCP cascade (`mcp_server.py:2251`). Refine inherits it
only via the daemon socket. Registration **widens acceptance nowhere** — unknown
keys already pass through unchanged (`params_validation.py:422-437`) — it only
adds a bool type check on all three planes. The ADR-040 D2a concern (MCP
inheritance) is real, but its consequence here is silent accept-and-drop, not
exposure; the D2a-analogous remedy is the removed-fields rejection of Finding 1,
since a request-type value gate does not apply (`generate` is the honored type).

**B — is `absent = False` genuinely fail-closed at every caller?**
Yes. `generate()` defaults `identity=False` (`generate.py:2078`); the server will
read it with a False default; the CLI sends it only when the flag is set;
refine's `_daemon_namespace` never sets it (subject to Finding 4); MCP never
forwards it; a `--params` replay cannot re-enter the mode (identity is sourced
from `args`, not `p_cur` — `generate.py:4876-4879`). Refine additionally forces
`ref_drop_strict=True`, so absence yields a hard error on krea refs.

**C — can the three change pipeline shape, or leak state across cache hits?**
Shape-safety confirmed: none touches `_load_pipeline`'s arguments; the cached
object stays a stock `Krea2Pipeline`; `identity_edit_pipe_call` is unbound.
Per-call attributes (`_guidance_scale`, `_interrupt`, `_num_timesteps`, scheduler
timesteps) are re-set by every stock call. The two residues are the processor
install/restore gap (**Finding 2 — the highest-risk item; the design must not
ship without closing it**) and the benign-but-undocumented VL memo (Finding 6).
`_bias_cache` lives on the per-call processor objects and is removed with them.

**D — does the identity path change reference handling on the daemon?**
No bypass. `_check_ref_paths` runs in `_handle_connection` (`server.py:512`)
before `_handle_generate` for every generate request, keyed only on `ref_images`
presence — `identity` is not consulted and cannot route around it. Slot ORDER
survives end-to-end: JSON array → order-preserving validator
(`params_validation.py:476-482`) → `req.get("ref_images")` as-is
(`server.py:1080`) → `_load_ref_pils` preserves list order (`generate.py:1727`).
The 2-source cap is enforced in-daemon by `_resolve_ref_family_support`
(`generate.py:1993-1999`) as an `InferenceError`; the wire-level 8-cap
(`params_validation.py:470`) bounds memory before that.

**E — disclosure risk in client-side warning emission?**
None. Both builders are pure functions of client-held values —
`_krea2_identity_param_warnings` (`generate.py:1745-1781`) and
`_krea2_identity_ignored_knob_warnings` (`1784-1825`): no paths, no daemon state.
The daemon→client direction is unchanged and already control-char sanitized
(`_sanitize_wire_warning`, `generate.py:1406-1418`). The real problem with
decision 5 is duplication, not disclosure (Finding 3).

**F — is the MCP surface closed by construction?**
FALSE as stated (Finding 1). One enforced layer exists — the explicit-kwargs call
site — not two; the input schema is advisory under `validate_input=False`, and
the `set(COMFYLESS_SCHEMA)` filter passes `ref_boost`, `grounding_px`, **and
`ref_images`**. Other MCP paths: cascade validates and never forwards to the
identity path; `extract_params` leaks the two scalars outward (Finding 8) but its
outbound `ref_images` drop at `mcp_server.py:391` is confirmed; the legacy
`--json` bridge (`_run_json_mode`, `generate.py:3300-3415`) forwards none of the
four. No sidecar-replay path on the MCP surface bypasses the call site.

**G — what the design missed.**
Four preconditions before code: (1) close Finding 2 — install-inside-try plus
daemon fail-closed on identity residue — because the cache-key decision is only
correct conditional on it; (2) make the MCP claim true rather than tested-as-false
(Finding 1), declaring the small `mcp_server.py` scope extension honestly;
(3) specify the warning dedupe (Finding 3) so the implementer does not resolve the
contradiction by weakening N1; (4) name refine as a wire-builder caller
(Finding 4). Plus a proof hook the ADR lacked: **after a delegated identity run
FAILS, the next non-identity request from the same daemon must be proven
byte-identical to a fresh-daemon run** — that is the test that actually pins the
cache-hygiene invariant the whole design rests on.

---

## Disposition summary

| # | Severity | Finding | Disposition |
|---|---|---|---|
| 1 | HIGH | MCP input schema advisory, not enforced | Accepted — decision 6 rewritten, `_GENERATE_REMOVED_FIELDS` extension in scope |
| 2 | HIGH | Processor install outside `try`; daemon caches residue | Accepted — hard precondition, lands first as its own commit |
| 3 | MEDIUM | `edit_warnings` already solves N1; decision 5 would double-print | Accepted — resolved by DELETING the client-side print |
| 4 | MEDIUM | refine `_daemon_namespace` lacks `identity` | Accepted in full |
| 5 | MEDIUM | Inbound `ref_images` reaches MCP `gen_params` uncontained | Accepted — closed by the Finding 1 fix |
| 6 | LOW | VL memo is durable instance state | Accepted — docstring + ADR wording |
| 7 | INFO | Version skew vs a stale daemon | Accepted as residual; TECH_DEBT |
| 8 | INFO | `extract_params` emits keys the tool rejects | Accepted — outbound strip ships WITH the inbound rejection |
| 9 | INFO | NaN `ref_boost`; post-load 3-ref refusal | Noted, out of scope; TECH_DEBT |

**No finding was rejected.** Findings 1, 2, 3 and 4 were independently verified
against the code by the commissioning session before disposition.
