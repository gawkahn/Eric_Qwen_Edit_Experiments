# ADR-008: Comfyless Iteration Mode (`--iterate`)

**Date:** 2026-04-23
**Status:** proposed

---

## Context

Running the same generation with variations (different prompts, seeds, LoRAs, sampler choices, etc.) is the most common multi-run workflow. Today the user drives this from bash with a `for` loop over `python -m comfyless.generate` invocations. That works but has three downsides:

1. **Cold-start tax per variation.** Even when the daemon is warm, each bash-loop iteration re-parses argparse, re-builds the request, re-opens the Unix socket, and re-sends metadata. For prompt-only sweeps, the marginal cost of an in-process iteration is much lower than one process per variation.

2. **No combinatorial safety net.** A bash `for p in "${prompts[@]}"; do for s in "${seeds[@]}"; do ...` can silently balloon to thousands of runs if the user doesn't count carefully. The tool has no visibility into "you just asked for 1200 generations."

3. **Metadata drift.** Bash-loop iterations don't share a common provenance marker — each PNG's sidecar records that one invocation but nothing about the sweep it belonged to. Replay and correlation become painful.

A first-class `--iterate` mode is also a prerequisite for the LLM-agent auto-refinement loop already sketched in Backlog → Ideas: the judge/planner loop needs a programmatic way to fan out N candidate generations from a single param snapshot with only the varied axis changing.

## Decision

### Invocation surface

Add a repeatable `--iterate <param> <file>` flag to `comfyless.generate`:

```bash
python -m comfyless.generate \
    --model /path/to/model \
    --cfg-scale 4.0 \
    --iterate prompt prompts.json
```

- **`<param>`** is the name of any param that can otherwise be set via CLI flag, `--params` sidecar, or `--override`. First release supports: `prompt`, `negative_prompt`, `model`, `transformer_path`, `vae_path`, `text_encoder_path`, `text_encoder_2_path`, `seed`, `cfg_scale`, `steps`, `sampler`, `width`, `height`, `lora`. Anything else returns an error.
- **`<file>`** is a flat JSON list of values, one per iteration. The element shape matches what that axis already expects — strings for prompts, ints for seeds, list-of-dicts for LoRA stacks (see examples below). Invalid file or wrong element shape = hard error, no generation.
- Repeatable: multiple `--iterate` flags stack, producing the **Cartesian product** of all iteration axes (every combination).

### File format — worked examples

Each axis defines its own element shape. The file is always a top-level JSON array; each element is substituted into the named param per iteration.

**Strings (`prompt`, `negative_prompt`, `sampler`):**

```json
[
  "a lone lighthouse on a stormy coast at dusk",
  "a bustling cyberpunk street market at night",
  "a quiet forest clearing in early spring"
]
```

**Integers (`seed`, `steps`, `width`, `height`):**

```json
[42, 1337, 9999, 100000]
```

**Floats (`cfg_scale`):**

```json
[3.5, 4.0, 4.5, 5.0]
```

**Paths (`model`, `transformer_path`, `vae_path`, `text_encoder_path`, `text_encoder_2_path`):**

Absolute local paths OR HuggingFace repo IDs. Per-iteration resolution flows through `resolve_hf_path` just like a single-gen invocation (see the hardening slice in `docs/security/review-resolve-hf-path-hardening-2026-04-23.md`), and if `--allow-hf-download` is set every cache-miss still prints the `[EricDiffusion] DOWNLOADING from HuggingFace: <repo>` stderr warning per fetch.

```json
[
  "/mnt/nvme-8tb/hf-local/Qwen-Image-2512",
  "black-forest-labs/FLUX.1-dev",
  "/mnt/nvme-8tb/hf-local/chroma1-base"
]
```

**LoRA stacks (`lora`):**

Each iteration item is a complete LoRA stack — a list of `{path, weight}` dicts to apply for that iteration. The empty list means "no LoRA." This lets one sweep compare stacks directly (no LoRA vs. LoRA A vs. A+detail vs. just B):

```json
[
  [],
  [{"path": "/loras/style_a.safetensors", "weight": 0.8}],
  [{"path": "/loras/style_a.safetensors", "weight": 0.8},
   {"path": "/loras/detail_boost.safetensors", "weight": 0.5}],
  [{"path": "/loras/style_b.safetensors", "weight": 1.0}]
]
```

Iteration mode's `lora` axis always replaces the full stack for that run — it does not merge with a `--lora` flag on the same invocation. If both are supplied, `--iterate lora` wins and `--lora` is ignored (with a stderr warning). Mixing stack iteration with single-LoRA CLI flags is too easy to get wrong.

### Parameter precedence

Unchanged from today except iteration values override everything else *per iteration*:

```
sidecar (--params) < --override key=value < explicit CLI flag < current --iterate value
```

Each generation in the sweep is equivalent to running a single-gen `comfyless.generate` with the iteration values patched in on top of the rest.

### Safety: max iterations + interactive confirmation

Two complementary gates:

1. **Hard cap `--max-iterations N`**: default **500**. Cartesian product of iteration axes is computed up front; exceeding `N` is a hard error, no generation. Provides a predictable non-interactive ceiling.

2. **Interactive confirmation over threshold**: if the computed total is **≥ 5 iterations**, print `Iteration inputs will result in X generations. Proceed? [y/N]` to stderr and read a line from stdin. Any non-`y`/`Y` answer aborts.

**`--yes`** flag skips the interactive prompt (for scripted / cron use) but still honors the hard cap.

Rationale for both: the cap catches "I accidentally loaded a 10,000-item prompt file," the prompt catches "I didn't realize `prompts × seeds × cfgs` came to 240."

### Output naming

Reuse the existing `_expand_savepath_template` (no new `%iter%` token). Comfyless already auto-increments filenames on collision (`foo.png`, `foo_0001.png`, `foo_0002.png`, …), so an iteration loop writing to a literal path already produces separate files. Users who want per-image distinguishing in the path template can already use `%seed%`, `%cfg%`, `%steps%`, `%sampler%`, `%date:spec%`, `%model%`, `%transformer%`, `%base_model%` — the iteration adds the axis values to those tokens automatically.

**New template token: `%input%`** — resolves to the stem of the first `--iterate` source file. Useful for grouping output: `--savepath %date%/%input%/gen` sorts all `prompts.json`-driven sweeps under `.../prompts/`. For multi-axis runs (`--iterate prompt X.json --iterate seed Y.json`), `%input%` uses the first `--iterate` flag's source; users who need per-axis stems get them via `%input_<param>%` tokens — e.g. `%input_prompt%` → `X`, `%input_seed%` → `Y`. If `%input%` is used without any `--iterate` flag, it expands to the empty string.

### Per-iteration sidecar and PNG metadata

Each generated image gets its own sidecar JSON and `parameters` tEXt chunk as today, recording the **fully resolved params for that iteration** (not the iteration list). Replay via `--params <iter-output.png>` is identical to replaying a non-iterated generation.

Add one new field to each sidecar / tEXt chunk: `"iterate_batch_id"` — a UUID generated once per `--iterate` invocation, shared across every image in the sweep, so downstream tools can group or dedupe by batch. The iteration axes themselves are *not* recorded in the individual sidecars (each image stands alone for replay) but can be recovered by grouping on `iterate_batch_id`.

### Interaction with `--json` mode

**Reject `--iterate` when `--json` mode is active in v1.** The JSON bridge's v1 contract is single-request; adding iteration semantics to the `--json` schema is a separate design decision that belongs in the LLM-agent-bridge work (Backlog → Queued → "JSON tool interface"). Error message: `"--iterate is not supported in --json mode; an iteration schema will be added to the JSON bridge contract in a future release"`.

### Interaction with the daemon

When `--iterate` is active and only prompt/seed/cfg-scale/steps/sampler/width/height vary (i.e., no iteration axis touches model/transformer/vae/text-encoder/LoRA paths), the daemon reloads nothing between iterations — it's exactly the warm-cache win that motivates the feature. When an axis *does* touch loadable weights, the server evicts and reloads per-iteration, matching the existing non-iterated behavior.

No new daemon command. The client fans out the loop locally and sends one `generate` request per iteration.

## Alternatives Rejected

**`--iterate` as a single flag with a whole-request JSON blob** (e.g., `--iterate runs.json` where `runs.json` is a list of full param dicts). Considered; rejected because it conflates "a sweep of one or two axes" with "a list of entirely separate runs." The list-of-dicts use case is better served by shell scripts or the future LLM-agent tool-call flow — `--iterate` is specifically for "hold everything else constant, vary this axis."

**Zip semantics as the default** (parallel tuples instead of Cartesian product). Considered; rejected as the default because Cartesian is the "parameter sweep" case users reach for first. Zip is a legitimate future addition (`--zip` flag or `--iterate-mode zip`) but splitting it out doesn't cost anything — Cartesian with N one-element lists is functionally a zip, and multi-axis matched tuples are rare enough to wait for explicit demand.

**Richer file formats (JSONPath, CSV, plain text)**. Considered; rejected for v1. Flat JSON list is the common denominator — users can produce it from their source of truth with one-line `jq`/`python -c` preprocessing. Adding JSONPath or CSV adapters should wait for evidence that the preprocessing step is actually burdensome.

**Self-describing file format** (e.g., `{"prompt": ["a", "b"]}` where the file names the axis and the CLI axis name becomes optional). Considered; rejected for v1 in favour of pure flat lists. The neutral form is marginally safer against "passed this file to the wrong axis" mistakes, but the flat form preserves reuse — the same `prompts.json` can be fed through `--iterate prompt` or (if the user ever wanted) piped through another mode without editing the file. If the misrouting problem actually surfaces, adding `{axis: [...]}` acceptance is a one-line isinstance check and can land in a later slice.

**Add `%iter%` / `%iter_idx%` tokens** to the savepath template. Rejected — the existing collision-auto-increment already gives distinct per-iteration filenames, and `%seed%` / `%cfg%` / other tokens cover the "distinguish by varied axis" case. `%iter%` would duplicate the collision-handling work with a less-informative value.

**Interactive prompt by default with no `--yes` escape**. Considered; rejected for scripted / cron use cases. `--yes` is the standard Unix idiom for non-interactive acceptance.

**Hard cap as an interactive prompt instead of a hard error**. Considered; rejected. The cap exists for scripted safety — something needs to fail-closed at some level independent of TTY presence. `--max-iterations` overrides if the user genuinely wants 1000.

**Store the iteration axes in each PNG sidecar**. Considered; rejected because it breaks the replay-per-image invariant (each sidecar would replay a whole sweep, not one image). The `iterate_batch_id` UUID is the minimal correlation primitive that preserves per-image replay while allowing downstream grouping.

## Deferred / Out of Scope

- **Zip-semantics mode** — a future `--zip` or `--iterate-mode zip` flag when matched-tuple use cases emerge.
- **Iteration under `--json` mode** — separate design when the LLM-agent-bridge slice lands (see Backlog → Queued → "JSON tool interface"). Likely shape: a new `type: "iterate"` request whose payload includes the axis lists.
- **Per-iteration progress reporting** — the comfyless daemon doesn't stream progress; iteration mode prints "iter i/N: ..." to stderr before each generation. A richer TUI/progress-bar UX is a separate slice.
- **Resume on failure** — if iteration 37 of 100 fails, v1 stops. A `--continue-on-error` mode (log the failure, continue) and a `--resume` mode (pick up from the last successful iteration) are both reasonable future slices; not in v1 because both require persistent iteration state.
- **Widening `--iterate` safety to `--json` sidecar-replay warnings** — the resolve_hf_path review (Finding 3 of `docs/security/review-resolve-hf-path-2026-04-23.md`) already covers the model-path-in-PNG case per individual generation. Iteration doesn't change the threat model because each iteration still resolves through the same code path.

## Execution Plan

Single slice, single commit (the feature is small enough that staged commits would fragment it):

1. Parse `--iterate <param> <file>` (repeatable), `--max-iterations N`, and `--yes` in argparse.
2. Add an early iteration-planning phase in `_run_cli_mode` (before any resolution or warnings): load each iteration file, validate it's a JSON list, compute the Cartesian product size, enforce the hard cap, emit the confirmation prompt if over threshold.
3. Extend `_expand_savepath_template` with `%input%` and `%input_<param>%` tokens (passed in from the caller; empty string when no iteration is active).
4. Fan out: iterate the Cartesian product, patch the varied params into `p` per step, run the existing generate path, collect per-image sidecar/PNG metadata (now including `iterate_batch_id`).
5. Print a final summary line to stderr: `iterate: N/N generations completed in T seconds` (or `i/N failed, aborting` on error).

Tests: add a negative-case test for the validator (invalid file format, unknown param name, cap exceeded) and a positive test that a 2-axis 3×3 sweep produces 9 output paths with distinct tokens. The existing manual-loop suite doesn't exercise `comfyless` directly, but the iteration entry-point lives in `_run_cli_mode` which is already covered by ad-hoc runs — add a small scripted test that iteration planning works without actually generating.

## Changelog

- **2026-04-23** — proposed. Initial spec after design discussion: Cartesian default, flat JSON list input, `%input%` token (source file stem), 500 hard cap + interactive confirm over 5 + `--yes` escape, rejected under `--json` mode in v1, per-image `iterate_batch_id` UUID for correlation.
- **2026-04-24** — amended. Added File-format section with worked examples for each element-shape (strings, ints, floats, paths, LoRA stacks). Added `lora` to the supported param list with stack-replacement semantics (iterated stack overrides `--lora`; warn if both are supplied). Added self-describing file format to Alternatives Rejected.

## AI-Disclosure

Claude (Opus 4.7) authored; Grant to review and sign off before execution.
