# Security Review — ADR-027 Refinement Loop (LLM-as-Judge), design phase

**AI-Disclosure:** Reviewed by the `security-auditor` agent (Claude Fable 5, pinned per §5A); findings triaged and folded into ADR-027 by Claude Fable 5; Grant reviewed.
**Date:** 2026-07-13
**Subject:** `docs/decisions/ADR-027-comfyless-refinement-loop.md` (design, pre-code)
**Verdict:** **needs-changes** — loop shape and reuse map are sound and the reused components (ADR-015 catalog resolver, ADR-022 FTS search, daemon `_check_paths`) check out; but the central validation mitigation as written breaks the design's keystone "no path from the LLM" invariant and must be rewritten in the ADR before code. All binding findings folded into ADR-027 Changelog 2026-07-13 (post-review).

---

## Summary

`comfyless/refine.py` is a CLI generate→judge→plan→regenerate loop where a local
VLM judge's output (revised prompt, LoRA add/remove/reweight by catalog name)
flows back into `generate()`. Threat model: single-user local desktop, no
auth/PII/billing; the judge endpoint is user-configured at the same trust
boundary as the enhancer. The security-relevant boundaries are **LLM output →
generation parameters** and, secondarily, **seed image (possibly foreign) →
judge context and → extracted params**.

Verified-and-holds: `catalog.resolve_reference` hardening chain (basename-strip,
forbidden-char gate, kind enforcement, existence, union-`_within` containment);
FTS `search()` parameterization + MATCH-operator neutralization; daemon-side
`_check_paths` root containment; `resolve_hf_path` fail-closed without explicit
download opt-in; `--max-iterations` as a genuine hard loop bound.

## Findings and disposition

**[CRITICAL] F1 — `COMFYLESS_SCHEMA` is the wrong allowlist; it grants the planner path authority the ADR forbids.**
`COMFYLESS_SCHEMA` (`comfyless/params_schema.py`) contains `model`,
`transformer_path`, `vae_path`, `text_encoder_path`, `text_encoder_2_path`,
`refiner_path` (all `str`), and `loras` entries are `{path, weight}`
(`validate_lora_entry` requires `path`). `validate_machine_request` additionally
passes unknown keys through unchanged. Validating planner output against that
schema would accept LLM-supplied filesystem paths and component-load targets —
including pickle-bearing `.pt/.bin` on the in-process path where `_check_paths`
never runs — directly contradicting the ADR's "no path is ever accepted from the
LLM" invariant. **Disposition: FOLDED.** ADR now specifies a closed two-key
planner-override allowlist (`prompt`, `loras:[{name,action,weight}]`) enforced
*before* any schema machinery; `COMFYLESS_SCHEMA` validates only the merged
config, never gates planner output; `validate_machine_request` is explicitly not
the gate.

**[HIGH] F2 — Ambiguous resolution plane; the ADR-022 SQLite DB stores `abs_path` and is not the hardened resolver.**
A one-query `SELECT abs_path FROM entries WHERE name=?` would bypass every
hardening property the ADR claims. The structural test forbidding
`generate.py`/`server.py` from importing `catalog_db` does not cover a new
`refine.py`. **Disposition: FOLDED.** ADR now pins: refine.py builds the ADR-015
in-memory catalog and resolves names via `catalog.resolve_reference(...,
expected_kind="lora")`; `catalog_db` is metadata/FTS only; refine.py never reads
`entries.abs_path`/`root`/`relative_path`; add refine.py to the structural test
when code lands.

**[MEDIUM] F3 — `catalog_db.search()` returns `abs_path`; planner context must strip path fields.**
**Disposition: FOLDED.** ADR planner-context section now states search
results/catalog lookups shown to the LLM carry name + metadata only;
`abs_path`/`root`/`relative_path` stripped before prompt assembly (mirrors MCP
`list_*`).

**[MEDIUM] F4 — `--seed-image` metadata channel bypasses the opaque-handle discipline.**
A foreign image's embedded `comfyless` PNG chunk seeds the config with raw
`model`/`transformer_path`/`loras[].path` (kept by warn-and-keep) and possibly
HF repo IDs → a network fetch if `--allow-hf-download` is exposed.
**Disposition: FOLDED.** ADR now states: refine.py does not expose
`--allow-hf-download` in v1 (all HF resolution `allow_download=False`,
fail-closed); on seed-image entry, loudly echo the load-bearing extracted fields
before the first generation; seed params are user-initiated and deliberately
keep full schema authority (unlike planner output).

**[MEDIUM] F5 — No size bound on images sent to the judge.**
17–50+ MP candidates base64-encode to hundreds of MB per iteration over stdlib
`urllib`; `--seed-image` read unbounded; enhancer transport timeout unverified.
**Disposition: FOLDED.** ADR now pins: every judge image downscaled to a fixed
eval resolution (longest side ≤1536 px) and re-encoded before base64;
`--seed-image` byte-capped; explicit per-call timeout on the judge HTTP request.

**[MEDIUM] F6 — No numeric bounds on planner-supplied LoRA weights; JSON parse accepts NaN/Infinity.**
`json.loads` default `parse_constant` accepts `NaN`/`Infinity`, which LLMs emit;
`validate_lora_entry` accepts any float. **Disposition: FOLDED.** ADR now pins:
verdict JSON parsed rejecting non-finite constants; LoRA weights clamped/dropped
outside |w| ≤ 4; scores validated as ints 1–10 before the composite.

**[INFO] F7 — Verdict-JSON strictness = reject-unknown at every level.**
**Disposition: FOLDED** into the contract (unknown keys dropped loudly at every
level; `verdict` outside `{pass,revise}` → `revise`; malformed response consumes
an iteration and continues — the cap, not the parse, bounds the loop).

**[INFO] F8 — "worst case is a wasted iteration" is conditional; reward-hack channel is concrete.**
The blast-radius claim holds only after F1/F2/F4 fold in. Concrete reward-hack
channel: the planner's revised `prompt` enters the next judge call's context, so
an injected response can plant judge-directed text that self-passes the loop and
promotes an arbitrary candidate to `winners/`. **Disposition: FOLDED.** ADR
Security section now states `winners/` is a recommendation, never consumed
downstream by automation without a human look; the persisted `*.verdict.json`
audit trail is the detection mechanism.
