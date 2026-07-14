# ADR-026: comfyless Prompt Enhancement Subsystem

**Date:** 2026-07-11
**Status:** accepted

---

## Context

Long-caption image models degrade badly on short prompts. Hunyuan-Image 2.1
was trained on long structured captions and ships a dedicated "reprompt" LLM
(a 14 GB `hunyuan_v1_dense` model under `HunyuanImage-2.1/reprompt/`) precisely
to expand user prompts before generation — a faithful quality test of 2.1
requires it. Beyond Hunyuan, Grant wants a general CLI-side enhancement path:
run a growing collection of prompt-list JSONs through an LLM to get enhanced
prompts and variations, avoiding manual cut-and-paste through an external LLM UI.

Today comfyless has **no** enhancement path. The only LLM rewriter in the repo
is a ComfyUI node (`nodes/eric_qwen_prompt_rewriter.py`) — a stdlib-`urllib`
client against an OpenAI-compatible chat endpoint with a baked `SYSTEM_PROMPT_EN`.
It is node-only; nothing in `comfyless/` imports it. So the CLI enhancer is new
construction, not a wiring job.

Requirements gathered in design (2026-07-11):

- **Inline** enhancement during `generate`/`--iterate`: enhance the *effective*
  prompt at each iteration. `--prompt "X" --iterate lora L.json` enhances "X"
  once; `--iterate prompt prompts.json` enhances each prompt as it flows through.
- **Offline** list→list transform: prompt-list JSON in, enhanced prompt-list JSON
  out, directly `--iterate prompt`-able in one command, with variations and
  traceability back to source prompts.
- **Family-specific output grammar**: an enhanced Pony prompt (danbooru tag-soup)
  is a different artifact from an enhanced Qwen/Flux prompt (natural language) or
  a Hunyuan prompt (long structured caption).
- **Selectable transformation intent**: generic ("let the LLM decide"),
  preserve-subject-enhance-setting, vary-setting/vary-subject, etc.
- The local reprompt backend requires `trust_remote_code=True` (custom
  `HYTokenizer` via the tokenizer's `auto_map`) — the **first** `trust_remote_code`
  in this codebase, which has deliberately kept it absent (verified in the
  `resolve_hf_path` security review, 2026-04-23).

## Decision

**1. One enhancement core.** `enhance(text, backend, recipe, n) -> list[str]`.
Every surface calls it; inline uses `n=1`, offline uses `n=N`.

**2. Backend ⟂ recipe decoupling — two orthogonal selectors.**
- **Backend = *where/how* to call the LLM** (endpoint url/model/key, or a local
  model). Named entries in a registry, same pattern as `api_keys.ini`.
- **Recipe = *what* transformation** (system prompt + target family grammar +
  temperature). Named template files.

Orthogonal so the config never becomes N×M: any endpoint runs any recipe.
CLI: `--enhance-prompt <backend>` + `--enhance-recipe <recipe>`.

**3. Two backends.**
- `openai-endpoint` — stdlib `urllib` OpenAI-compatible client, logic ported from
  `eric_qwen_prompt_rewriter.py`. Uses the selected recipe's system prompt.
- `hunyuan-reprompt` — local 14 GB `hunyuan_v1_dense`, `trust_remote_code=True`
  (see §8). Its enhancement behaviour is baked into the fine-tuned weights, so it
  **ignores `--enhance-recipe`**.

Secrets and endpoint URLs live only in the backend registry — never in CLI args
and never written to sidecars.

**4. Recipes are small TOML files** carrying `system_prompt`, `target` (family
grammar), and `temperature`. At **generate-time** the family's default recipe is
auto-selected (same overlay pattern as `comfyless/family_defaults.py`), overridable
with `--enhance-recipe`. The **offline** transform has no model loaded, so the
recipe (or `--family`) is named explicitly. v1 ships `-generic`,
`-preserve-subject`, `-vary-setting` per supported family, seeded from the node's
`SYSTEM_PROMPT_EN`.

**5. Inline hook + per-run memoization.** Enhancement hooks in at the per-tuple
prompt-resolution seam in `comfyless/generate.py` (where `prompt` is resolved into
`call_kwargs` before the pipeline call). Results are **memoized per unique input
prompt within a run**, so a fixed prompt across a LoRA/transformer sweep is
enhanced once and held constant (clean A/B; no wasted LLM calls), while a
prompt-axis iteration enhances each distinct prompt. Both cases fall out of the
same code path.

**6. Offline subcommand.** `comfyless enhance in.json --backend <b> --recipe <r>
--variations N -o out.json`. Input is the existing `--iterate prompt` shape (a flat
JSON list of strings); output is a flat JSON list of length M×N, directly
`--iterate prompt`-consumable in one command. Variations lengthen the list — the
prompt axis stays flat, no nesting. A companion provenance sidecar
(`out.provenance.json`) maps each output entry → `{source_prompt, variation_index}`;
`--iterate` ignores it, so traceability is free.

**7. Sidecar / replay semantics.** Enhancement runs **once**, at generation time.
The metadata sidecar records the backend name, recipe name, the original prompt,
and the enhanced result. `--params <sidecar>` replay uses the **stored enhanced
prompt** and does **not** re-call the LLM — otherwise replay would be
non-reproducible. The enhancer is seedable, but the frozen sidecar is what
guarantees exact replay.

**8. `trust_remote_code` policy (first in repo).** Permitted **only** for the
vendored, on-disk reprompt tokenizer, and only under these constraints:
- Loaded `local_files_only=True` from a path already on disk — "remote" collapses
  to *that reviewed local snapshot*, never a fresh fetch.
- The executed code (`tokenization_hy.py`, and anything in its `auto_map`) is
  **reviewed** before enabling. First review (2026-07-11): a benign 298-line
  tiktoken wrapper — imports `base64`/`os`/`unicodedata`/`tiktoken`/`transformers`,
  plain `open()` for the BPE file, no `subprocess`/`eval`/`exec`/network.
- The reviewed `.py` is **hash-pinned** so a later silent swap on disk is
  detectable at load.
- `security-auditor` (Opus) review is required before the slice-6 backend lands.

## Alternatives Rejected

- **Raw-URL backend descriptor** (`--enhance-prompt localhost:8001/v1`) — cannot
  carry the model name, API-key reference, or system prompt; leaks the URL/secrets
  into the replayable sidecar; not team-portable. A raw-URL *shorthand* may be
  offered later as a documented convenience, but named registry entries are the
  primary interface.
- **Fusing backend+recipe into single named entries** — N backends × M recipes
  config explosion. Orthogonal selectors instead.
- **Bracket combinatorial dynamic prompting** (`[thin,heavy] man …`) — explosive,
  and `--iterate` already provides exact Cartesian product when wanted. Controlled
  LLM-selective enhancement (regional markers, deferred to slice 5c) is the better
  lever.
- **`--enhance-only` scatter of per-prompt sidecars** — forces a for-loop over a
  directory to replay. Replaced by the single-list offline transform (§6).
- **Skip the local reprompt, use only the API endpoint** — insufficient for a
  faithful Hunyuan 2.1 quality test, which is the stated goal.

## Deferred / Out of Scope

- **Regional `keep`/`enhance`/`vary` span markers** (slice 5c) — recipes cover the
  coarse "preserve X / enhance Y" policy in v1.
- Recipe authoring help beyond the three v1 generics.
- Image/vision-conditioned enhancement (the node's inpaint/ControlNet variants).
- MCP surfacing of enhancement — this ADR is CLI-scoped.

## Slice map (this ADR governs slices 4–6 of the Hunyuan epic)

- **Slice 4** — this ADR + `security-auditor` on the `trust_remote_code` decision.
- **Slice 5** — core + `openai-endpoint` backend + backend/recipe registries +
  inline `--enhance-prompt`/`--enhance-recipe` + three generic recipes per family.
  `code-reviewer`.
- **Slice 5b** — offline `comfyless enhance` list→list transform + `--variations`
  + provenance sidecar. `code-reviewer`.
- **Slice 5c** *(deferred)* — regional span markers.
- **Slice 6** — `hunyuan-reprompt` local backend. `security-auditor`.

## Changelog

- 2026-07-11 — proposed. Design settled in session; enhancement architecture
  (backend/recipe decoupling, inline memoization, offline list→list transform,
  sidecar/replay contract, `trust_remote_code` policy) captured before any code.
- 2026-07-11 — **accepted; implemented.** `comfyless/enhance.py` +
  `comfyless/recipes/*` + inline `--enhance-prompt` (`generate.py`) + offline
  `python -m comfyless.enhance`. Both backends live/mock-validated. Deviations
  from the proposal, all in `implementation_details.md`: A1 (uniform TOML backend
  entries, no "reserved name"); A5 (reprompt output language follows input); A9
  (inline recipe family-default is best-effort → `generic`; use `--enhance-recipe`
  or the offline `--family` for family-specific). Deferred: A10 (VRAM
  co-residency + daemon-path provenance; `free_reprompt_cache()` provided, offline
  transform is the recommended hunyuan batch path), A11 (redirect auth hardening,
  LOW), slice 5c regional markers. §8 trust_remote_code implemented + reviewed:
  `docs/security/review-enhancer-trust-remote-code-2026-07-11.md` (CLEAN after the
  auto_map/config-pin MEDIUM was folded). code-reviewer findings #3/#4/#6/#7 folded.

- 2026-07-14 — **amendment: endpoint sampling-knob precedence.** The
  `openai-endpoint` path previously read only `temperature`/`top_p` from the
  recipe and ignored the backend cfg's sampling knobs entirely, never sending
  `top_k`/`repetition_penalty` — so knobs set in `enhancers.toml` (e.g.
  `[qwen-vl] temperature=0.99, top_k, repetition_penalty`) were silently dead.
  New `_resolve_endpoint_sampling(recipe, cfg)` resolves every knob with
  precedence **recipe > cfg > default**: `temperature` always resolves
  (default 0.8, default moved out of `load_recipe` into the resolver so a
  cfg-level temperature can apply when the recipe omits it); `top_p`/`top_k`/
  `repetition_penalty` are emitted only when set (keeps a plain request
  OpenAI-standard; the last two are vLLM extensions). `load_recipe` now
  validates/coerces all four knobs if present. A bogus `batch` cfg key (the
  toggle is `batch_variations`) emits a loud stderr warning but does not block
  (warn-don't-block). Recipe knobs override backend defaults; recipes are the
  per-recipe source of truth, cfg the per-backend fallback. test_enhance 59→80.
  code-reviewer (Fable) run before commit.

**AI-Disclosure:** Claude (Opus 4.8) authored; Grant reviewed. 2026-07-14
sampling-precedence amendment: Claude (Fable) authored; Grant reviewed.
