# Security review — MCP path-leak close (`upscale_vae_path` / `refiner_path`)

**AI-Disclosure:** Claude (Fable 5) authored the review via the
`security-auditor` subagent; Claude (Opus 5) commissioned it, independently
verified the central claim, and applied the remediation; Grant reviewed and
approved the fix scope.

**Date:** 2026-08-01
**Change under review:** `comfyless/mcp_server.py`, `test_mcp_server.py`,
`test_hunyuan.py`
**Red Zone surface:** `comfyless/mcp_server.py` (LLM-agent tool surface)
**Spec:** there is no ADR — this is a pre-existing defect with its own lineage
(ADR-030 upscale VAE, ADR-016 hunyuan refiner). The spec is the HIGH and MEDIUM
in `docs/security/review-adr-044-commit3-mcp-hardening-2026-08-01.md`, which
Grant explicitly approved fixing in a separate commit from ADR-044's three.

**Reviewer model provenance:** 37 tool uses, all assistant turns
`claude-fable-5`. **No Fable→Opus fallback.**

---

## Verdict

The two named findings are correctly fixed **at the two sinks the fix targeted**,
and the `test_hunyuan.py` relaxation is a legitimate net tightening. But the sink
sweep this commit was supposed to complete was **still incomplete**: the generate
RESPONSE renderer (`_resolved_params_as_names`) was not touched, and it emits —
live — keys the same tool now hard-rejects on presence.

> No live path-VALUE leak remains anywhere I could reach.

Same shell-less caveat as the prior two reviews: working-tree state was checked
against the described change rather than a literal `git diff`.

## Coverage

Reviewed: `comfyless/mcp_server.py` 40-210, 212-296, 298-475, 478-640, 738-870,
1639-1690, 1730-1810, 1991-2314, 2317-2426, 2647-2760, 2821-2887, 2893-3012;
`comfyless/params_schema.py` in full (complete path-typed inventory);
`comfyless/generate.py` 1609, 2047-2212, 2630-2739; `test_hunyuan.py` and
`test_mcp_server.py` in the changed regions; `TECH_DEBT.md`; the prior review.

Not reviewed: the literal `git diff` (no shell tool); `params_validation.py` and
`catalog_db` internals (unchanged, relied on the prior review's
characterization); the two test files outside the changed sections.

---

## Findings

### [MEDIUM] The generate RESPONSE still emitted the rejected keys — the replay trap this commit's own comment condemns

**Location:** `comfyless/mcp_server.py` `_resolved_params_as_names`;
`comfyless/generate.py` metadata block

Every successful MCP `generate` response's `resolved_params` — documented as the
agent's authoritative record — contained `ref_boost`, `grounding_px`,
`upscale_vae_path`, `upscale_vae_subfolder`. All are now rejected **on presence**
(`if _removed in arguments`), and an empty string does not spare you. An agent
echoing `resolved_params` back into `generate` gets a hard ValidationError:
exactly the "emitting a key the sibling tool refuses turns an innocent replay
loop into a hard error" failure this commit's own comment names as the reason the
outbound strip exists.

For `upscale_vae_path` / `upscale_vae_subfolder` **the trap was introduced by
this diff**; for `ref_boost` / `grounding_px` it dated to ADR-044 commit 3.

**The prior review's Q3 answer was factually wrong** — it asserted "the generate
response cannot echo the scalars because generate() records them only when
non-default." Recording is unconditional.

**Independently verified by the commissioning session** before acting, precisely
because it contradicted an earlier review: `generate.py`'s metadata block carries
the comment "Both are recorded UNCONDITIONALLY (not gated on ref_kind)" in as
many words. The reviewer is right and the earlier review was wrong.

**Secondary, latent:** the same renderer passed `upscale_vae_path` /
`refiner_path` through verbatim if ever non-empty — unreachable under MCP today,
but exactly the "two sinks, one implemented" shape this commit condemns, with a
docstring ("No abs_path crosses the boundary") papering over it.

**Disposition:** FIXED. The renderer now pops every field in both rejection
tuples plus `ref_boost` / `grounding_px`, **sourced from the tuples themselves**
rather than a hand-copied list so the rule cannot drift. Tests assert each field
individually, that no `/home/` path survives, and — as a set relation — that NO
member of either rejection tuple can appear in a response. That set-relation test
immediately caught `identity` missing from the first, hand-written version of the
pop list, which is the argument for writing it that way.

### [MEDIUM] Inbound closure half-done on the cascade branch; the TECH_DEBT register understated it

**Location:** cascade entry checks only `_GENERATE_UNSUPPORTED_REF_FIELDS`;
`TECH_DEBT.md`

`{prompt, cascade_config, refiner_path: "/abs/..."}` is still accepted,
type-validated and silently dropped — now for seven weight-path fields instead of
four. Inert today (cascade never reaches `generate()`; its extract renderer is
allowlist-built). But the TECH_DEBT entry named only the original four, so the
register claimed a narrower gap than exists — and, as the reviewer put it, the
whole reason this commit exists is that a previous list was incomplete.

**Disposition:** code deferral UNCHANGED (closing it changes behaviour for a
caller shape ADR-044 never touched, after a narrower scope was approved).
Register CORRECTED — a new append names all seven bypassed fields explicitly,
plus the `refiner_steps` / `refiner_cfg` pair below, with the fix shape recorded.

### [INFO] Residual accept-and-drop: `refiner_steps` / `refiner_cfg` — the complete remaining list

The reviewer's answer to "the complete list this time." After this change, every
path-typed `COMFYLESS_SCHEMA` member is handled in both directions on the
non-cascade branch. The only schema members left accept-and-drop are the two
refiner numerics: they pass the payload filter into `gen_params`, are never
forwarded, never rejected, and survive `extract_params` outbound — so extracting
a CLI refiner sidecar hands the agent orphaned numerics a replay silently
ignores. Non-path, and a `**gen_params` refactor would forward them into a
guaranteed no-op since `refiner_path` is rejected and `generate()` gates on it.

**Disposition:** recorded in TECH_DEBT so the deferral is a decision, not an
omission.

### [INFO] `test_hunyuan.py` relaxation — legitimate, with two named narrowings

The reviewer was asked to judge adversarially whether this was weakening a lock
to make the change pass. Verdict: the old lock (`"refiner" not in <module
source>`) was **provably not holding the invariant it claimed** — the HIGH leak
existed while it was green, because threading or leaking via schema-generic code
never needs the literal token. The AST replacement asserts "does NOT thread
refiner" at the point of consumption, which is strictly stronger where it
matters.

Two genuine narrowings were named: (1) scope shrank from the whole module to
`_handle_generate`; (2) the incidental tripwire on ever *advertising* refiner
fields was lost. Plus hygiene: a now-unused `mcp_src_step3` read and a comment
referencing an assertion that no longer exists.

**Disposition:** all four FIXED — the AST walk now covers the whole
`comfyless.mcp_server` module, the three weight-path fields were added to the
schema-absence loop, the dead read is gone, and the stale comment is rewritten.

### [INFO] `_MCP_PATH_TYPED_FIELDS` lacked `refiner_path`

Doubly latent (MCP cannot activate the refiner; the agent-facing base64 copy is
metadata-free), so it matters only for a PNG the operator shares. But this
commit's own comment says a path-typed key "belongs in BOTH places" while fixing
one of the sinks.

**Disposition:** FIXED — `refiner_path` added, with the latency noted in place.

---

## Reviewer's answers to the commissioned questions

**1 — Outbound completeness.** Complete for `extract_params` (both branches),
`list_*` / `search` (allowlisted projections, no path fields), PNG-via-return_image
(transport copy carries no text chunks), and error messages (field name only,
value-disclosure tested). Was NOT complete for the generate response body — the
MEDIUM above.

**2 — Test relaxation.** Legitimate, a tightening honestly labelled; the old lock
demonstrably co-existed with the leak.

**3 — Replay loop.** Fully closed through `extract_params` on both sidecar paths;
was NOT closed through the generate response. Now is.

**4 — `upscale_vae_subfolder`.** Rejecting it is correct, not overreach. Alone it
is inert — activation requires `upscale_vae_path` or a daemon-cached VAE the MCP
cache never sets — so it is only meaningful alongside a rejected field; accepting
it would be a guaranteed silent drop, and it is filesystem-shaped
(relative-subpath) input on a surface with no upscale capability.

**5 — Block-worthy.** Nothing at CRITICAL/HIGH. The reviewer asked for the
generate-response pops to ride in this same commit (same file, same spec,
partially caused by this diff) and for the TECH_DEBT append before merge. Both
done.

---

## Disposition summary

| Severity | Finding | Disposition |
|---|---|---|
| MEDIUM | Generate response echoed rejected keys (replay trap) | FIXED — pops sourced from the rejection tuples; set-relation test |
| MEDIUM | Cascade branch bypass; TECH_DEBT understated it | Code deferral unchanged; register CORRECTED with all seven fields |
| INFO | `refiner_steps` / `refiner_cfg` accept-and-drop | Recorded in TECH_DEBT as a decision |
| INFO | `test_hunyuan.py` narrowings + hygiene | FIXED — module-scope AST, schema-absence widened, dead read removed |
| INFO | `_MCP_PATH_TYPED_FIELDS` lacked `refiner_path` | FIXED |

**No finding was rejected.** The MEDIUM was independently verified against the
code before remediation, because it contradicted a claim in the prior review —
and the prior review was the one that was wrong.
