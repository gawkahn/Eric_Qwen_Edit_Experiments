# ADR-027: comfyless Iterative Refinement Loop (LLM-as-Judge)

**Date:** 2026-07-13
**Status:** accepted

---

## Context

The auto-refinement loop has lived in the backlog (Ideas → "Auto-refinement
loop") since the project's LLM-endgame direction was set: *generate → judge →
plan → regenerate, iterated to a quality threshold*. Until now the building
blocks were missing. They now all exist:

- **`generate()` + the per-GPU daemon** already key their pipeline cache on
  model + LoRA set + quant, so a **prompt-only** change reuses the warm
  pipeline while a **LoRA** change evicts and reloads. The loop's central
  performance assumption ("most iterations are prompt tweaks; don't reload")
  falls out of existing behavior — no new caching work.
- **`--params` + `--override`** replay is exactly the per-iteration mutation
  mechanism. `extract_params` already reads both a sidecar JSON and a PNG
  `comfyless` chunk, so an existing image can seed the loop.
- **The catalog DB** (ADR-022, live) carries real semantic fields per entry:
  `description`, `usage_tips`, `trigger_words`, `strength_rec`, `sampler_rec`,
  `nsfw_level`, plus an FTS search index over
  `(name, model_name, description, usage_tips, trigger_words)`. This is what
  makes catalog-driven LoRA reasoning real: the planner can look up "what does
  this LoRA do / its trigger / its strength range" and search by effect.
- **The enhancer's `openai-endpoint` backend** (ADR-026) already speaks
  OpenAI-compatible `/v1/chat/completions` over stdlib `urllib`, with error
  handling and batch handling. The judge reuses that transport pattern; the
  only new wire concern is the **vision content shape**
  (`content: [{type:text}, {type:image_url, image_url:{url:"data:image/png;base64,…"}}]`),
  since the enhancer path is text-only.

Design decisions gathered 2026-07-13 (Grant):

- **v1 planner authority: prompt + LoRA only.** Rewrite the prompt, add/remove
  keywords, add/remove/reweight catalog LoRAs. **No** base-model or transformer
  swap in v1 (search-space explosion + forced reloads + heaviest catalog-metadata
  dependence — deferred).
- **Two entry modes:** a fresh `--prompt`, *or* a `--seed-image` (+ optional
  `--params` sidecar) to refine a prior result further.
- **Judge backend is pluggable** (mirrors the enhancer's backend registry).
  v1 ships the `openai-endpoint` vision path, starting with the **Gemma dense
  31B endpoint (:8016)**. A Qwen-VL backend slots in when the ai-stack agent
  lands a launcher — config/flag switch, no loop change.
- **One combined judge+planner call** per iteration ("evaluate and suggest
  fixes") returning structured scores + critique + overrides.
- **Eval axes:** prompt-adherence and aesthetics.
- **CLI-first.** Not an MCP tool — the loop is *sequential feedback* (judge N,
  then choose params for N+1), which the MCP `generate`-in-a-loop shape already
  serves; a batch `iterate` MCP tool is the wrong shape and is parked.

This ADR is written before code per §12. Because the loop lets an LLM's output
select LoRAs and shape prompts fed into `generate()`, it trips the §12 review
trigger ("model output drives parameters into generate()"); a `security-auditor`
design review runs before any code and its findings fold back here.

## Decision

**New module `comfyless/refine.py`; CLI `python -m comfyless.refine`.** It
orchestrates existing subsystems; it does not reimplement generation, catalog
access, or LLM transport.

### Loop

A **greedy hill-climb** (not a beam — gens are expensive and the cap is small):

1. **Seed** the working config from either `--prompt` (fresh) or
   `--seed-image`/`--params` (via `extract_params`). On seed-image entry, the
   embedded PNG chunk / sidecar is a **second untrusted channel** (F4): refine.py
   does **not** expose `--allow-hf-download` in v1 (all HF resolution stays
   `allow_download=False`, fail-closed), and it loudly echoes the load-bearing
   extracted fields (`model`, `transformer_path`, all `loras[].path`) before the
   first generation. Seed params are user-initiated and deliberately keep full
   schema authority — unlike planner output, which is the constrained channel.
2. **Generate** one image through the normal `generate()` path (daemon-aware:
   prompt-only → warm pipeline; LoRA change → evict+reload). Write it to
   `candidates/` with its sidecar.
3. **Judge+plan** in one combined LLM call: image + original target prompt +
   current config + relevant catalog context → structured JSON. The image is
   **downscaled to a fixed judge-eval resolution (longest side ≤1536 px) and
   re-encoded before base64** (F5) — candidates can be 17–50+ MP, which would
   otherwise base64 to hundreds of MB per call; `--seed-image` reads are
   byte-capped and the judge HTTP request carries an explicit per-call timeout.
4. **Record** the verdict JSON beside the candidate. Track best-so-far by
   composite score.
5. **Stop** if `verdict == pass` (both axes ≥ threshold), or the iteration cap
   is hit, or no composite improvement for `--patience` iterations. Otherwise
   **apply** the planner's validated overrides and loop to step 2.
6. **Finalize:** copy the passing image (or, if none passed, the top-ranked
   candidate) into `winners/`.

### Judge+planner contract (closed override allowlist)

The combined call returns JSON validated before use:

```
{
  "scores": {"prompt_adherence": <int 1-10>, "aesthetics": <int 1-10>},
  "critique": {"prompt_adherence": <str>, "aesthetics": <str>},
  "verdict": "pass" | "revise",
  "overrides": {
    "prompt": <str, optional>,
    "loras": [ {"name": <catalog name>, "action": "add"|"remove"|"set_weight",
                "weight": <float, optional>} ]   // optional
  }
}
```

**Validation is a closed two-key allowlist enforced BEFORE any schema machinery
(security review F1).** `COMFYLESS_SCHEMA` is **not** the gate on planner output
— it *contains* path fields (`model`, `transformer_path`, `vae_path`,
`text_encoder_path`, `text_encoder_2_path`, `refiner_path`) and its `loras`
entries are `{path, weight}`; gating on it would let the LLM supply filesystem
paths, breaking the keystone invariant. `validate_machine_request` is likewise
not the gate (it passes unknown keys through). Instead:

- **Exactly two override keys are honored: `overrides.prompt` (str) and
  `overrides.loras` (list of `{name, action, weight}`).** Every other key — and
  every unknown key at any level of the verdict JSON — is dropped with a loud
  warning (warn-don't-block). `verdict` outside `{"pass","revise"}` → treated as
  `revise`. A malformed/unparseable judge response consumes an iteration and the
  loop continues (the cap, not the parse, bounds the loop) (F7).
- **LoRAs are referenced by catalog NAME only, resolved via the ADR-015
  in-memory resolver (F2).** refine.py builds the in-memory catalog at startup
  (`catalog.build_catalog` from the same operator roots the MCP server uses) and
  resolves every planner name through `catalog.resolve_reference(catalog, name,
  roots, expected_kind="lora")`. The planner-facing `name` is translated to a
  path **only after** that hardened resolution — the loop never accepts a path
  from the LLM and never reads `entries.abs_path`/`root`/`relative_path` from the
  ADR-022 SQLite DB (which is metadata/FTS only, explicitly not a load plane).
  An unresolvable name is dropped with a warning, not fabricated into a path.
  refine.py is added to the structural AST test that forbids load-plane column
  reads when code lands.
- **Numeric bounds (F6).** The verdict JSON is parsed rejecting non-finite
  constants (`parse_constant` raises on `NaN`/`Infinity`); LoRA `weight` outside
  `|w| ≤ 4` is clamped/dropped with a warning; `scores` are validated as ints in
  1–10 before entering the composite. `COMFYLESS_SCHEMA` validation still runs on
  the *merged* config afterward, as the normal generation-path check.

### Planner context (what the LLM sees)

- The original target prompt + the current effective config (prompt, LoRA set +
  weights).
- For every LoRA currently in play *and* any the planner is invited to consider:
  its catalog `description`, `usage_tips`, `trigger_words`, `strength_rec`. When
  the planner wants a LoRA by *effect* rather than name, the loop runs the
  catalog FTS search and offers the top matches (by name) — the planner still
  chooses from real catalog names.
- **Path fields are stripped before prompt assembly (F3).** Catalog lookups and
  FTS search results (`catalog_db.search()` returns `e.*`, including `abs_path`)
  are reduced to name + metadata only — `abs_path`/`root`/`relative_path` never
  enter the LLM's context, mirroring the MCP `list_*` opaque-handle behavior.

### Scoring / "good enough"

- **Pass** = both axes ≥ `--pass-threshold` (default **8/10**).
- **Ranking** (winner selection) = weighted composite, prompt-adherence
  weighted higher than aesthetics (default **0.6 / 0.4**) because aesthetics is
  the noisier, more subjective axis.
- **Cap** = `--max-iterations` (default **10**).
- **Early stop** = `--patience` iterations with no composite improvement
  (default **2**).
- Judge runs at **temperature 0** (low variance / reproducible scoring).

### Output layout

`--output-dir/` contains:
- `candidates/` — every generated image + its sidecar + a `*.verdict.json`
  (scores, critique, verdict, the overrides that produced the *next*
  candidate). This is the audit trail: Grant can re-judge by hand and see *why*
  the loop moved where it did.
- `winners/` — the passing image, or the top-ranked candidate if none passed.

### Reuse map (no reinvention)

| Need | Reused from |
|------|-------------|
| Generation + daemon cache | `generate()` / `comfyless/server.py` |
| Per-iteration mutation | `--params` / `--override` merge, `COMFYLESS_SCHEMA` |
| Seed-from-image | `extract_params` (sidecar + PNG chunk) |
| LoRA/transformer semantics | catalog DB (`descriptions`, FTS) |
| LLM transport + error handling | `openai-endpoint` pattern (ADR-026) |

## Security (§12)

The loop feeds LLM output back into `generate()`, so it is treated as a §12
machine-driven-params surface (not §5 Red Zone: no auth/PII/billing, single-user
local, no network egress introduced — the judge endpoint is user-configured,
same trust boundary as the enhancer).

Threats + mitigations:

- **LLM-selected identifier → path traversal** — eliminated by construction
  *once the closed two-key override allowlist (F1) and the pinned ADR-015
  resolution plane (F2) are implemented as specified above*: the planner's only
  path-adjacent output is a LoRA `name`, translated to a path solely by the
  hardened resolver; no path field is ever honored from the LLM.
- **Image-based prompt injection into the judge VLM** — an adversarial seed
  image (or any candidate) could carry text steering the judge's scores/overrides.
  With F1/F2/F4 in place the blast radius is a wasted local iteration, not a
  path/param escape; the cap bounds spend; the candidate audit trail exposes it.
  (This "worst case is a wasted iteration" claim is *conditional* on F1/F2/F4 —
  without the closed allowlist it would be false, which is why F1 is CRITICAL.)
- **Seed-image metadata channel (F4)** — a foreign image's PNG chunk can seed
  raw path/repo-ID params. Bounded by fail-closed HF resolution (no
  `--allow-hf-download` in v1) + the loud pre-generation echo of load-bearing
  fields.
- **Unbounded resource spend** — `--max-iterations` is a hard cap; `--patience`
  cuts thrash; judge images are downscaled + byte-capped with a per-call timeout
  (F5).
- **Reward-hacking** (planner games the judge) — accepted risk for v1. Concrete
  channel (F8): the planner's revised `prompt` enters the *next* judge call's
  context, so an injected response can plant judge-directed text that self-passes
  the loop and promotes an arbitrary candidate to `winners/`. Bounded because
  it's local, the human reviews `winners/`, and the per-candidate
  `*.verdict.json` audit trail is the detection mechanism. **`winners/` is a
  recommendation — never consumed downstream by automation without a human look.**
  Noted for the aesthetic-calibration follow-up.

`security-auditor` (Fable, pinned per §5A) reviewed this design before code; its
output is saved to `docs/security/review-refinement-loop-2026-07-13.md`. Verdict
was **needs-changes** on one CRITICAL (F1, the schema-as-gate error) plus one
HIGH and four MEDIUM findings; all are folded into the contract above and this
Security section. The binding invariants the review established:

1. Planner overrides are a **closed two-key allowlist** (`prompt`, `loras`) —
   `COMFYLESS_SCHEMA`/`validate_machine_request` are never the gate on planner
   output (F1).
2. LoRA name→path resolution goes through the **ADR-015 in-memory resolver
   only**; refine.py never reads the ADR-022 DB's `abs_path` (F2).
3. Path fields are stripped from all LLM-facing catalog context (F3).
4. Seed-image entry is fail-closed on HF download + echoes load-bearing fields
   (F4).
5. Judge images are downscaled + byte-capped with a per-call timeout (F5).
6. Verdict JSON rejects non-finite numbers; weights clamped to `|w| ≤ 4`; scores
   are ints 1–10 (F6).

## Alternatives Rejected

- **MCP `iterate` tool as the vehicle.** A Cartesian sweep commits to all param
  combinations up front with no feedback between cells; the refinement loop is
  sequential-feedback. Parked (backlog).
- **Beam search over candidates.** Better exploration, but multiplies gen cost
  against a deliberately small cap. Greedy-with-best-tracking fits the budget.
- **Separate judge and planner calls.** Cleaner (neutral judge blind to its own
  remedies) but doubles calls/iteration. Deferred; v1 is one combined call.
- **Full model/transformer-swap authority in v1.** Explodes the search, forces
  frequent reloads, and leans hardest on catalog-metadata completeness. Deferred.
- **MCP/agent exposure first.** The MCP judge entrypoint is an open ADR-011
  question (catalog-routed vs privileged) tangled with the deferred
  image-upload-as-seed security review. CLI-first sidesteps both.

## Deferred / Out of Scope

- **Model/transformer swap in the planner action space.** Trigger: v1 proves
  the loop and prompt+LoRA authority is demonstrably insufficient.
- **Qwen-VL judge backend.** Pluggable slot; lands when the ai-stack launcher
  is ready. Gemma dense :8016 is v1.
- **Reference-image aesthetic calibration.** v1 uses a written rubric; a generic
  rubric carries its own taste. Trigger: the loop's aesthetic picks diverge from
  Grant's on real runs.
- **MCP / LLM-agent exposure of the loop.** Gated on the ADR-011 judge-entrypoint
  decision + the image-upload-as-seed security review.
- **Writing AI-authored critiques back into the catalog** (`descriptions.source
  = 'ai_authored'` exists). Out of scope; the loop reads the catalog, it does
  not enrich it.

## Changelog

- 2026-07-13 — Initial. Proposed. Records the design gathered with Grant
  (prompt+LoRA authority; prompt-or-seed-image entry; pluggable judge starting
  Gemma dense :8016; one combined judge+planner call; greedy hill-climb; pass =
  both axes ≥8; cap 10; candidates/winners output).
- 2026-07-13 (post-review) — `security-auditor` (Fable) design review returned
  **needs-changes**: 1 CRITICAL (F1 — `COMFYLESS_SCHEMA` as the planner gate
  would accept LLM-supplied paths, breaking the keystone invariant), 1 HIGH (F2
  — resolution-plane ambiguity), 4 MEDIUM (F3 path-leak in planner context, F4
  seed-image metadata channel, F5 unbounded judge-image size, F6 numeric
  bounds), 2 INFO (F7 verdict strictness, F8 reward-hack channel). All folded
  into the Judge+planner contract, Loop, and Security sections above. Review:
  `docs/security/review-refinement-loop-2026-07-13.md`. Design is now
  code-ready pending Grant's sign-off on the revised contract.
- 2026-07-13 (accepted) — Grant signed off on the revised two-key-allowlist
  contract. Status → accepted. Implementation proceeds in slices: (1) verdict
  schema + judge client in isolation (F1/F6/F7 + vision endpoint call);
  (2) catalog-name resolution + planner-context assembly (F2/F3); (3) loop
  controller (greedy hill-climb, candidates/winners, daemon reuse); (4)
  seed-image entry (F4/F5).
- 2026-07-13 (slice 1 landed) — commits b0d3549 (this ADR + review) + 89ca9c5
  (`comfyless/refine.py` verdict boundary + `test_refine.py`, 72 tests). Both
  Fable reviewers confirmed the F1 keystone; folded critique allowlisting,
  huge-int OverflowError→RefineError, and the seed-image pixel guard. Pushed.
- 2026-07-13 (slice 2 landed) — catalog-name resolution + planner-context
  assembly. code-reviewer + security-auditor (both Fable) confirmed F2 (name→path
  ONLY via the ADR-015 resolver; the ADR-022 DB's abs_path is never read) and F3
  (closed allowlist projection; path columns never reach the planner) both hold.
  Folded: a structural AST guard so a future `SELECT abs_path`/`row["abs_path"]`
  regresses loudly (was the F2-disposition's promised enforcement); broadened
  `open_catalog_db` to warn-and-degrade on a corrupt/schema-mismatched DB instead
  of crashing. **Forward constraints for slices 3/4 (from the slice-2 reviews):**
  (a) `ResolvedLoraOp.abs_path` MUST NOT be serialized into any planner-visible
  artifact (sidecar, `*.verdict.json`, next-call context); (b) resolver notices
  carry `res.cause`/`path_was_discarded` — keep them operator/stderr-only, or
  flatten to a uniform "not resolvable" string before any text re-enters LLM
  context (else `PathMoved`/`WithinFailure` leak filesystem-drift state the search
  plane never exposes); (c) slice 4 seed-image LoRA refs can be path-shaped, so
  honor the `path_was_discarded` INFO notice there.

- 2026-07-13 (slice 3 landed) — greedy hill-climb loop controller
  (`refine_loop`), daemon-aware generation (`run_generation`), the combined
  judge+plan glue (`judge_candidate`), path-free `verdict_record`, and the CLI
  (fresh `--prompt` entry). `test_refine.py` 94 → 155 tests.
  **security-auditor (Fable): APPROVE** — F1/F2/F3/F5/F6 + forward-constraints
  (a)/(b) all verified to hold; the pass gate is numeric and lie-proof (F8).
  Review: `docs/security/review-refinement-loop-slice3-2026-07-13.md`.
  **code-reviewer (Fable): needs-changes** on correctness; all findings folded
  before commit:
    - **HIGH — daemon savepath re-rooting.** The daemon re-roots the savepath
      template under its OWN `--output-dir` ("the client never dictates paths"),
      so on the loop's primary (daemon) path the candidates landed outside the
      run's `candidates/` tree and the ADR §Output-layout audit trail was empty.
      Fixed: `run_generation` now MOVES the daemon's returned image to the
      canonical `output_dir/candidate_NN.png`, giving uniform daemon/cold naming.
    - **MEDIUM — judge model autodetect escaped F7.** `GET /models` raised
      `EnhanceError` (not `RefineError`), aborting the run mid-loop and skipping
      winner finalization. Fixed: the model id is resolved + cached ONCE at
      startup; `judge_candidate`'s fallback re-raises as `RefineError`.
    - **MEDIUM — FTS search-offers was dead code.** The planner saw only active
      LoRAs and could never ADD one (half the v1 authority). Fixed:
      `search_loras(target_prompt)` now feeds path-free add-candidates into the
      judge context each iteration.
    - **MEDIUM — `refine_loop` had no tests.** Added 20 loop tests via
      monkeypatched `run_generation`/`judge_candidate`: pass/cap/patience stops,
      F7 iteration consumption, seed pinning after iter 0, winner finalization.
    - **LOW** — `_post_judge` JSON decode moved outside the transport `try`
      (non-JSON/oversized body → `RefineError`, stays within F7);
      `run_generation` logs the unreachable-daemon fallback and raises on
      `status=ok`-with-no-path; the cold path forwards transformer/vae/te/refiner
      override fields (slice-4 landmine closed); the `key_env` registry
      convention is honored for the judge Authorization header; `main()` exits
      cleanly (not a traceback) on `RefineError` / bad `--lora` weight.
  **Constraint (a) clarification (auditor INFO-2):** "planner-visible artifact"
  in the slice-2 forward-constraint means the path-free `verdict.json` + the
  judge context ONLY. The load-plane `<stem>.json` sidecar legitimately carries
  `loras[].path` — it is the human's `--params` replay artifact and is never read
  back into judge context. **Binding on slice 4:** seed sidecars are ingested
  ONLY via the F4 trusted-human channel (loud echo of load-bearing fields,
  `path_was_discarded` honored); sidecar content never enters judge context.
  Slices remaining: (4) seed-image entry (F4/F5).

- 2026-07-15 (slice 4 landed) — seed-image entry. `--seed-image` (a prior
  comfyless PNG) + optional `--params` override sidecar seed the working config
  via `build_config_from_seed`; `--prompt` XOR `--seed-image` (mutually exclusive
  required group); `--model` optional (defaults to the seed's model). Seed params
  keep FULL schema authority (user-initiated) — extraction routes through
  `generate._load_params`/`_validate_params`. F5 gate (`load_seed_image_capped`)
  runs FIRST; path-shaped `loras[].path` refs are `.safetensors`-stripped then
  basename→catalog-resolved through the ADR-015 resolver (foreign dir dropped,
  `path_was_discarded` surfaced, forward-constraint (c)); the load-bearing path
  fields are loudly echoed before the first generation. `test_refine.py` 155→188.
  **security-auditor (Fable): APPROVE with conditions** — no CRITICAL/HIGH; F1/F2/
  F3/F5-at-entry/F6 + forward-constraints (b)/(c) all verified to hold. Four
  MEDIUMs, all folded before commit:
    - **F4 echo omitted `upscale_vae_path`** (a daemon-loaded weight field, ADR-030)
      → added to `_SEED_ECHO_PATH_FIELDS` with a keep-in-sync note vs
      `server._PATH_FIELDS`.
    - **Seed prompt enters judge context unbounded.** **Ruling (this slice):** the
      seed's embedded `prompt` becomes the judge TARGET and necessarily re-enters
      judge context every iteration — this is the ONE deliberate exemption to the
      slice-3 "sidecar content never enters judge context" constraint (the loop
      cannot judge without a target). It is now length-capped at
      `OVERRIDE_PROMPT_MAX_CHARS`, symmetric with the planner-override prompt, so a
      crafted chunk cannot inject megabytes of judge-directed text (F8).
    - **`--params` read escaped the F5 byte cap** → `_stat_within_bytes` gates it
      against `SEED_IMAGE_MAX_BYTES` before read.
    - **Cold path had no root containment for seed component paths** (same accepted
      `generate --params` trust model; the LLM cannot reach `base`) → the F4 echo
      now flags each path OUTSIDE the operator roots ("loads on the cold path
      only") so the human sees it before an unattended loop; the daemon path still
      hard-validates via `_check_paths`. A fail-closed cold-path gate is a
      trust-model change deferred to Grant (TECH_DEBT).
  **code-reviewer (Fable): needs-changes**, all folded:
    - **MEDIUM — `float(r.op.weight or 1.0)` rewrote a deliberate weight 0.0 to
      1.0** → `... if r.op.weight is not None else 1.0`; weight-0 test added.
    - **MEDIUM — cold path dropped the ADR-030 `upscale_vae_path`/
      `upscale_vae_subfolder`** (daemon replayed 2×, cold 1×) → both forwarded in
      the cold `gen.generate` call.
    - **LOWs** — `_extract` catch tuple gained `AttributeError`/`TypeError` +
      a non-dict guard (a list/string `--params` no longer tracebacks); malformed
      seed lora entries are dropped WITH a notice; seed mode logs that CLI gen
      flags are ignored; abspath-on-slash HF-repo-id caveat documented; trivial
      abspath test strengthened with a relative path.
  Reviews saved: `docs/security/review-refinement-loop-slice4-2026-07-15.md`.
  **ADR-027 is now feature-complete** (all four planned slices landed); deferred
  items live in TECH_DEBT (cold-path containment; aesthetic-calibration follow-up).

**AI-Disclosure:** Claude (Fable 5) authored the design record from a design
conversation with Grant; Grant reviewed. Slice 3 implementation + review
folding authored by Claude (Fable 5); Grant reviewed. Slice 4 implementation +
review folding authored by Claude (Fable 5); Grant reviewed.
