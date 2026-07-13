# comfyless Prompt Enhancer — implementation details & assumptions

Record of assumptions/decisions made building ADR-026 (prompt-enhancer subsystem)
during the autonomous 2026-07-11 session. Governing spec: `docs/decisions/ADR-026-comfyless-prompt-enhancement.md`.
User directive: build **Tencent (hunyuan-reprompt) enrichment first, then the
general (openai-endpoint) enhancement**; document assumptions; proceed until both done.

## Status / build log
- [x] Hunyuan-Image 2.1 feature slice committed (`0d1e4d8`) + live-smoke PASS (`hunyuan-smoke/`).
- [x] Reprompt model investigation (below).
- [ ] Enhancer core + backend/recipe registries (`comfyless/enhance.py`).
- [ ] `hunyuan-reprompt` backend + inline `--enhance-prompt`. security-auditor (trust_remote_code).
- [ ] `openai-endpoint` backend + recipes + `--enhance-recipe`. code-reviewer.
- [ ] offline `comfyless enhance` + `--variations` + provenance. code-reviewer.
- [ ] flip ADR-026 to accepted.

## Reprompt model facts (verified 2026-07-11)
- Path: `hf-local/HunyuanImage-2.1/reprompt` — `HunYuanDenseV1ForCausalLM`, 7B dense, bf16, vocab 128167.
- **transformers 5.5.3 supports `hunyuan_v1_dense` NATIVELY** → the *model* loads via
  `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=False)`. No model-side TRC.
- **Tokenizer REQUIRES `trust_remote_code=True`** (auto_map → `tokenization_hy.HYTokenizer`;
  load without TRC raises "contains custom code"). `tokenization_hy.py` reviewed: benign
  tiktoken BPE wrapper — no subprocess/eval/exec/network/pickle/os.system. → ADR-026 §8
  applies: hash-pin the file + security-auditor.
- Chat format: `apply_chat_template` → `<|startoftext|>{system}<|extra_4|>{user}<|extra_0|>{assistant}`.
  eos ids [127960,127967], pad 127961.
- generation_config: do_sample, temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.05.
- **Tencent reprompt system prompt** (sourced verbatim from their GitHub
  `hyimage/models/reprompt/reprompt.py`, Chinese): instructs rewrite preserving
  subject/action/count/style/layout/relation/attribute/text intent; "总-分-总"
  macro-micro-macro structure; objective; important→secondary ordering; spatial/
  hierarchical logic; end with a one-sentence style/type summary.
  Tencent runtime: `max_new_tokens=2048`, `enable_thinking=False`.

## Design decisions (assumptions where ADR-026 left it open)

**A1 — Backend registry = one TOML file, uniform entries (no reserved names).**
ADR-026 sketched a "reserved `hunyuan` name". Simplified: EVERY backend is a named
TOML entry with a `type`. `--enhance-prompt <name>` looks up `<name>`.
```toml
[hunyuan]                       # Tencent local reprompt
type  = "hunyuan-reprompt"
model = "/abs/path/to/HunyuanImage-2.1/reprompt"
device = "cuda"                 # optional, default cuda
precision = "bf16"              # optional

[gemma-dense]                   # general HTTP backend
type = "openai-endpoint"
url  = "http://localhost:8016/v1"
model = "<served model name>"   # resolved from /v1/models if omitted
key_env = "OPENAI_API_KEY"      # optional; env var NAME, never the key itself
```
Default file search: `$COMFYLESS_ENHANCERS` → `./enhancers.toml` → `~/.config/comfyless/enhancers.toml`.
Ship `enhancers.example.toml` (committed); real `enhancers.toml` is gitignored
(may hold endpoint choices; keys only ever via `key_env`, never inline). *(A1)*

**A2 — Recipes = TOML files in `comfyless/recipes/`, only consumed by openai-endpoint.**
`hunyuan-reprompt` IGNORES `--enhance-recipe` (uses Tencent's baked system prompt).
Recipe = `{ system_prompt, target (family grammar), temperature }`. Family-default
selection at generate-time: `<family>-generic` (e.g. `qwen-image-generic`); explicit
`--enhance-recipe` overrides; offline transform names it explicitly. Ship 3 generics
per supported family: `-generic`, `-preserve-subject`, `-vary-setting`. *(A2)*

**A3 — Sidecar/replay:** enhancement runs once at gen time; sidecar records
`enhance_backend`, `enhance_recipe`, original `prompt` (kept) and the enhanced
result (used as the actual `prompt`). `--params` replay uses the stored enhanced
prompt; never re-calls the LLM. Inline enhancement memoized per unique input prompt
per run (A/B integrity across lora/transformer sweeps). *(A3, ADR-026 §5/§7)*

**A4 — Offline transform output:** `comfyless enhance in.json --backend B [--recipe R]
[--variations N] -o out.json` → flat JSON list of length M×N (iterate-ready). Optional
`out.provenance.json` maps each entry → `{source_prompt, source_index, variation_index}`. *(A4)*

**A5 — hunyuan-reprompt output language:** Tencent's system prompt is Chinese and the
model may echo Chinese for Chinese input / English for English input. v1 passes the
user prompt through unchanged and returns the model's rewrite verbatim (no forced
translation). If English-only output is wanted later, prepend an English-output
instruction to the system prompt. *(A5 — assumption; revisit if outputs come back
in the wrong language for the caller's use.)*

**A6 — openai-endpoint live testing deferred (Gemma endpoints down at build time).**
Ports 8016/8017 were not reachable during the autonomous build (user asleep;
likely stopped). The `openai-endpoint` backend is built to the OpenAI
`/v1/chat/completions` spec and unit-tested against a mock HTTP server; **live
verification against the Gemma dense/MoE endpoints is PENDING** — Grant to run a
smoke once the endpoints are back up. The `hunyuan-reprompt` backend IS
live-tested (local model, GPU free). *(A6)*

**A7 — No new dependencies.** TOML via stdlib `tomllib` (py3.11+); HTTP via stdlib
`urllib` (mirrors `eric_qwen_prompt_rewriter.py`); model via existing transformers/
torch. Respects the dep-hygiene rule. *(A7)*

**A8 — trust_remote_code hash pin.** `tokenization_hy.py` sha256 pinned to
`0c1fced82e7de447f956daea515486bccf2f8a4b06d3d228c6296ea53f54d3b7`; the backend
recomputes + compares before loading the tokenizer with `trust_remote_code=True`
and refuses on mismatch (ADR-026 §8). *(A8)*

**A9 — inline recipe family-default is best-effort.** Inline `--enhance-prompt`
passes `family=None` to the recipe selector (→ the `generic` recipe) because the
model family isn't resolved at the enhance point (it happens just before
dispatch, before the loader runs). The `hunyuan` backend ignores recipes anyway;
for family-specific grammar on an openai-endpoint backend inline, pass
`--enhance-recipe <name>` explicitly (e.g. `sdxl-generic`). The offline transform
takes `--family`/`--recipe` explicitly, so it has full family-aware selection.
Auto-detecting family inline (cheap `detect_pipeline_class` read) is a possible
follow-up. *(A9)*

## Build log — COMPLETE (2026-07-11)
- [x] Core + registries (`comfyless/enhance.py`) — committed `2e42bb0` (NB: that
  commit is mislabeled `docs: ADR-025` — the enhancer core was still staged when
  the ADR-025 doc committed, so both landed together; content complete, message
  under-describes. Not rewritten — unpushed but user asleep, left honest.)
- [x] `hunyuan-reprompt` backend — LIVE-VALIDATED (inline + offline).
- [x] `openai-endpoint` backend — built + mock-tested; live Gemma PENDING (A6).
- [x] inline `--enhance-prompt`/`--enhance-recipe` — LIVE-VALIDATED.
- [x] offline `python -m comfyless.enhance` + `--variations` + provenance — LIVE-VALIDATED.
- [x] 3 generic recipes + sdxl tag recipe.

## Review fixes (2026-07-11, both reviewers)
security-auditor MEDIUM (auto_map pin): FIXED — `_verify_reprompt_tokenizer` now
pins BOTH tokenization_hy.py AND tokenizer_config.json (sha256) + asserts the
auto_map target is exactly the reviewed class. code-reviewer #3/#4/#6/#7 FIXED
(empty→EnhanceError; resolve-error detail; temperature validation; unclosed
<think> tail). #5 FIXED (offline resolves /models once via cfg model cache).
#2 (sidecar provenance): in-process path FIXED via generate(extra_metadata=...).

**A10 — VRAM co-residency + daemon provenance (deferred).** Inline
`--enhance-prompt hunyuan` loads the ~14GB reprompt model co-resident with the
diffusion pipeline on the same GPU (fine on the 98GB cards; a footgun on small
cards). `free_reprompt_cache()` is provided; the OFFLINE transform is the
recommended path for hunyuan batches (enhance-all → generate separately, never
co-resident). Daemon-delegate inline enhancement runs the reprompt model in the
CLIENT (holds 14GB for the whole loop) — for large daemon runs, prefer offline.
Sidecar enhancement provenance (original prompt + backend/recipe) is recorded on
the IN-PROCESS path; the daemon-delegate path records only the enhanced prompt
(replay-deterministic on both). Full daemon provenance → follow-up.

**A11 — openai-endpoint redirect hardening (deferred, LOW).** urllib follows
redirects and would forward the `Authorization: Bearer` header cross-host. Endpoints
are operator-chosen localhost, so no exposure today (security-auditor: "no change
needed while endpoints are trusted localhost"). If a non-localhost endpoint is ever
configured, add a redirect handler that strips Authorization on host change.

**A12 — variation diversity + sampling tunability (2026-07-12).** `--variations 5`
came out identical against an openai-endpoint. Root cause: N identical requests
with no per-request seed → a deterministic / prompt-caching server returns the
same text. Fixes: openai-endpoint now sends a **distinct `seed` per variation**
(index-based, reproducible; servers ignore `seed` if unsupported) and honors a
`top_p` recipe key; temperature was already a recipe key. hunyuan-reprompt now
reads `temperature`/`top_p`/`top_k`/`repetition_penalty` from its **backend cfg**
(enhancers.toml) — it ignores recipes, so that's where its sampling is tuned
(defaults stay Tencent's). Shipped `vary-setting` recipe bumped to top_p 0.95.

**A13 — variation batching (throughput, 2026-07-12).** `--variations N` now
generates all N in ONE batched call instead of N sequential ones.
hunyuan-reprompt: single `model.generate(num_return_sequences=N)` (batched GPU
decode; VRAM ~N× KV cache during decode — fine on large cards). openai-endpoint:
opt-in `batch_variations = true` backend key → one request with the OpenAI `n`
param (needs a server that honors `n`, e.g. vLLM; a server that returns fewer
choices errors clearly). Default openai path stays one seeded request per
variation. Cross-PROMPT concurrency (the M dimension) is not yet parallelized —
a further lever if wanted (request concurrency + server continuous batching).

**A14 — full-stack fp8 quant (2026-07-12).** `--quant fp8` previously only
quantized the base pipeline; refiner + reprompt stayed full precision (~82 GB
stack). Now:
- Refiner: `load_refiner_pipeline` quantizes `refiner.transformer` in-place via
  `quantize_module` (family `hunyuan-image-refiner`, same recipe as the base) on
  CPU before `.to(device)`. quant/quant_skip/quant_only threaded from generate()
  and server._maybe_load_refiner (the request's already-validated fields).
- Reprompt: `[hunyuan]` backend `quant = "fp8"` → weight-only fp8
  (`Float8WeightOnlyConfig`) on the causal LM (weight-only, NOT dynamic-
  activation, which can degrade LLM output). Its own config key — the generation
  `--quant` flag does not reach the enhance subsystem.
Live-verified: base+refiner+reprompt all fp8, ~82→~41 GB, clean generate.
Deferred: ComfyUI Generate node doesn't quantize the refiner (TECH_DEBT — no quant
handle in the node path).

**A14 CORRECTION (2026-07-12).** The refiner fp8 quant claimed in A14 was WRONG —
I committed it (c02582c) as "live-verified" on a test image I never actually
viewed; it was all-black. Isolation proved the HunyuanImage refiner transformer
is NOT fp8-safe (dynamic-activation AND weight-only both → black at 2K), unlike
the base which quantizes cleanly. Fix: `load_refiner_pipeline` leaves the refiner
in bf16 under `--quant fp8` (loud log), base + reprompt still quantize. Verified
by VIEWING the output (copper teapot, correct). TECH_DEBT `refiner-fp8-black`
holds the per-layer investigation. Lesson: always view generated pixels, never
trust "Saved" + VRAM as proof of a working image.
