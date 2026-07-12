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
