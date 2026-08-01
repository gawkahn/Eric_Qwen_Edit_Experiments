# Security review — ADR-044 commit 3 (MCP surface hardening)

**AI-Disclosure:** Claude (Fable 5) authored the review via the
`security-auditor` subagent; Claude (Opus 5) commissioned it, independently
verified the HIGH finding, and applied the remediation; Grant reviewed.

**Date:** 2026-08-01
**Change under review:** ADR-044 commit 3 — `comfyless/mcp_server.py`,
`test_mcp_server.py`
**Red Zone surface:** `comfyless/mcp_server.py` (LLM-agent tool surface)
**Design ADR:** `docs/decisions/ADR-044-krea2-identity-daemon-carriage.md`
**Prior reviews in this chain:** `review-adr-044-identity-daemon-carriage-2026-08-01.md`
(design), `review-adr-044-commit2-wire-carriage-2026-08-01.md`

**Reviewer model provenance:** 27 tool uses, all assistant turns
`claude-fable-5`. **No Fable→Opus fallback.** A parallel `code-reviewer` (Fable,
no fallback) covered correctness and test quality.

---

## Verdict

The core change is correct and correctly placed: the rejection runs as step 0 on
the raw argument dict, before `validate_machine_request`, before any catalog or
filesystem touch. It is **not** bypassable by key-case or whitespace variants
(the validator does not normalize keys, and non-canonical variants are filtered
out by the `set(COMFYLESS_SCHEMA)` gate) nor by aliasing (`_CLI_TO_CANONICAL` is
CLI-side; the wire surface takes canonical keys only). The vacuous
schema-absence test shape did not land — the tests pin behaviour and the call
site, keeping the schema check explicitly labelled as the weak pin.

But the reviewer found the closure **incomplete in two directions**, one of
which is a live outbound path disclosure that predates this commit.

Threat model applied: same-UID stdio MCP server; the agent is the adversary; the
two protected properties are (1) no agent-supplied filesystem path is ever
consumed or one-refactor-away from consumption without a containment gate, and
(2) no absolute path or filesystem fact crosses outbound to the agent.

## Coverage

Reviewed: `comfyless/mcp_server.py` 60-315, 330-570, 1085-1185, 1731-1835,
1890-2460, 2610-2965; `test_mcp_server.py` 4710-4736 and the new ADR-044 block;
`comfyless/params_schema.py` in full; `comfyless/params_validation.py` 400-494;
`comfyless/generate.py` 2260-2304; ADR-044 and both prior reviews in full.

Not reviewed, with reasons: the literal `git diff` (no shell tool in the review
environment — working-tree state was checked against the ADR contract, so an
unrelated hunk in an unread region could escape); `cascade.validate_config`'s
handling of keys nested inside `cascade_config` (none of the four has a
consumption path there); `_maybe_attach_return_image` / `_resolve_mcp_output_path`
(unchanged, no interaction with the four fields).

---

## Findings

### [HIGH] `extract_params` emits verbatim absolute paths via `upscale_vae_path` and `refiner_path`

**Location:** `comfyless/mcp_server.py` (the `_render_extracted_params` drop
list), `comfyless/params_schema.py:72-73, 101`

The outbound drop tuple was extended for `ref_boost`/`grounding_px`, but two
abs-path-bearing `COMFYLESS_SCHEMA` siblings are absent. A sidecar recording
`upscale_vae_path` (ADR-030) or `refiner_path` (ADR-016) survives
`_validate_params` normalization — both are schema `str` fields — is not
resolved to a catalog name, is not popped, and is **returned to the agent
verbatim**. That discloses the operator's filesystem layout across the agent
boundary, in direct contradiction of the function's own docstring invariant
("No absolute path or directory survives this boundary"), and is the exact shape
ADR-035 slice-1b treated as a regression when the field was `ref_images`.

Reviewer's stated assumption: CLI-generated sidecars share the MCP
`--output-dir` (the documented deployment) and at least one run used
`--upscale-vae` or `--refiner`.

**Independently verified by the commissioning session.** Both are
`COMFYLESS_SCHEMA` string members; neither appears in the drop list. Sharper
than the reviewer stated: `upscale_vae_path` IS handled in the *other* outbound
sink — it sits in `_MCP_PATH_TYPED_FIELDS` and is basenamed for PNG-metadata
redaction (invariant 12), with a comment saying explicitly "so it cannot leak
the host filesystem layout to an MCP agent." So the intent was recorded and one
of two sinks implements it. `refiner_path` is in neither.

**Remediation:** add `upscale_vae_path`, `upscale_vae_subfolder` and
`refiner_path` to the pop tuple (or reverse-resolve `refiner_path` to a catalog
name as `transformer_path` is), plus negative tests beside the `ref_images` one.

**Disposition:** NOT FIXED IN THIS COMMIT — raised with Grant as its own
decision. Different fields, different ADR lineage (ADR-030 / ADR-016),
pre-existing, and outside the `mcp_server.py` scope extension Grant approved
(which was specifically the reference/identity fields). Folding a HIGH-severity
fix for unrelated fields into a Red Zone commit after its scope was approved is
the wrong order of operations.

### [MEDIUM] Inbound `upscale_vae_path` / `refiner_path` take the same accept-and-drop route this commit exists to close

**Location:** `comfyless/mcp_server.py` (absent from `_GENERATE_REMOVED_FIELDS`;
pass the payload filter into `gen_params`; not forwarded at the call site)

Both are caller-supplied weight paths, `SCHEMA_KIND` members, validated by
`validate_machine_request`, present in `gen_params`, and silently dropped only
by the explicit-kwargs call site — the identical latent hazard as `ref_images`
(design review Finding 5). A `generate(**gen_params)` refactor would forward an
agent-supplied absolute path into a weight *load*, with no `_check_paths`
coverage (step 6 checks only model/transformer/loras). The
`_GENERATE_REMOVED_FIELDS` comment's own rationale — "silently accepting a raw
`vae_path` would reintroduce the caller-supplied-path input attack surface
ADR-015 removes" — applies verbatim.

**Disposition:** NOT FIXED — same reasoning as the HIGH; raised with Grant
together with it, since they are the same two fields on opposite sides.
Mitigated meanwhile by the AST splat guard added in this commit, which is now a
real barrier rather than a formatting-fragile substring.

### [MEDIUM] The cascade dispatch of the same tool bypassed both rejection tuples

**Location:** `comfyless/mcp_server.py` (routing on `cascade_config` presence,
before any field guard; `_handle_generate_cascade` ran `validate_machine_request`
with no removed/unsupported-field loop)

A payload `{prompt, cascade_config, ref_images: [...]}` — or any of the four, or
the weight-path tuple — was accepted, type-validated and silently dropped. **The
ADR proof hook "a payload carrying them is REJECTED, not dropped" was therefore
false on half the surface it claims to cover**, and the N1 silent-accept-and-drop
this commit condemns persisted on the same Red Zone tool. No consumption path
exists today (cascade never reaches `generate()`), so this is an invariant break
plus a latent gap, not a live exploit.

**Disposition:** FIXED for `_GENERATE_UNSUPPORTED_REF_FIELDS` — the loop now runs
at `_handle_generate_cascade` entry, with four new negative tests (one per
field). `_GENERATE_REMOVED_FIELDS` was deliberately NOT hoisted: it has always
shared this bypass, and closing it changes behaviour for a caller shape this
commit never touched. TECH_DEBT entry filed, recommending both loops be hoisted
above the cascade/non-cascade split rather than duplicated further.

### [INFO] Audit line retains a supplied `ref_images` path — consistent posture, not a finding

A rejected request's `ref_images` values land verbatim in the operator's stderr
audit line. This matches the pre-existing weight-path rejection posture exactly
(a rejected `vae_path` value is likewise retained), invariant 5's dropped-fields
rule is scoped to prompt content, the value is the caller's own input, and the
stream is operator-only and same-UID. The agent-facing error correctly discloses
the field name only — verified by test and by message construction.

Adjacent observation, pre-existing and not introduced here: generate's audit echo
is uncapped, unlike the flood-capped `list_*` / `search` / `extract_params`
payloads.

**Disposition:** ACCEPTED, no change. The commissioning session had reached the
same conclusion independently and scoped the no-path-oracle test to the
agent-facing error deliberately, saying so in the test comment.

### [INFO] Call-site tripwires were substring-based and formatting-fragile

`f"{_f}=" not in <slice>` and `"generate(**" not in <slice>` are defeated by
trivial reformatting (`generate( **gen_params)`, `kw = gen_params;
generate(**kw)`, building a kwargs dict). For the four rejected fields this is
only defence-in-depth — the behavioural entry-rejection is the real guard — but
**the splat check is the only automated barrier protecting the un-rejected
fields of the MEDIUM above**, which is where the fragility bites.

**Disposition:** FIXED. Converted to an AST walk over `_handle_generate`: a
premise check that exactly one `generate()` call exists, a per-field keyword
check, and a `kw.arg is None` check (that node IS the `**splat`). House
precedent: the AST guards in `test_krea2_identity.py`.

---

## Reviewer's answers to the remaining commissioned questions

**Breaking change (Q3) — the right call.** Rejecting the previously-ignored
fields beats the prior behaviour, which made an agent believe a reference edit
occurred when it did not: an error is better than a lie. The self-inflicted
replay loop is fully closed — `extract_params` strips all three schema members,
the cascade extract renderer is allowlist-built so none of the four can survive,
`list_models` / `list_loras` / `list_transformers` and `search` emit allowlisted
name/kind/metadata projections only, and the generate response cannot echo the
scalars because `generate()` records them only when non-default and the MCP plane
can no longer supply a non-default.

**`identity` (Q4) — handled consistently.** Presence-based rejection matches
`_GENERATE_REMOVED_FIELDS` semantics; even `identity: false` rejects, which is
fail-closed and simple. Nothing else on this surface depends on unknown-key
tolerance for it, and the general unknown-key regime (pass the validator, die at
the `COMFYLESS_SCHEMA` filter) is otherwise unchanged.

**Bypass analysis (Q1).** Not reachable by key-case or whitespace variants — the
validator performs no key normalization, and non-canonical variants are dropped
by the `set(COMFYLESS_SCHEMA)` gate. Not reachable by aliasing — the alias map is
CLI-side. The one bypass found was the cascade branch (MEDIUM above).

---

## Disposition summary

| Severity | Finding | Disposition |
|---|---|---|
| HIGH | `extract_params` leaks `upscale_vae_path` / `refiner_path` verbatim | **Raised with Grant** — pre-existing, unrelated fields, outside approved scope |
| MEDIUM | Same two fields accept-and-drop inbound | **Raised with Grant** — same decision |
| MEDIUM | Cascade branch bypassed both rejection tuples | FIXED for the ref tuple; weight-path tuple → TECH_DEBT |
| INFO | Audit line retains supplied ref path | Accepted — consistent with existing posture |
| INFO | Substring call-site tripwires | FIXED — AST guard |

**No finding was rejected.** The HIGH and its MEDIUM sibling are deferred to an
explicit decision rather than dismissed; the ADR Changelog records them as
knowingly left open, so they cannot be lost.
