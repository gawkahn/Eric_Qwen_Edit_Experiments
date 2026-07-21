# ADR-035: comfyless Reference-Image Surface (unified generate/edit schema)

**Date:** 2026-07-20
**Status:** accepted

---

## Context

comfyless is generation-only today. Two consumers now need reference-image
input (Backlog, 2026-07-19):

1. **Video keyframe authoring** — ADR-033 roadmap item 5 is explicitly blocked
   on it. Authoring keyframe N+1 from keyframe N with scene lock is an edit
   operation, and the 2026-07-19 hot test authored its keyframe pair by hand
   through the ComfyUI node pack precisely because comfyless could not.
2. **Chat-driven editing** via MCP/OpenWebUI — deferred to a later ADR (see
   *Deferred*), but it constrains the schema shape chosen here.

The first consumer is the proving ground for this ADR; the second must not be
made harder by it.

### What already exists (verified 2026-07-20)

- **comfyless half-knows qwen-edit already.** `comfyless/family_defaults.py`
  carries a `qwen-edit` entry (`true_cfg_scale` 4.0, steps 30), and
  `comfyless/catalog_builder.py` maps the `qwenedit` / `qwen edit` aliases onto
  it. What is missing is any execution path.
- **Pipeline classes are auto-detected, not mapped.** `detect_pipeline_class`
  (`nodes/eric_diffusion_utils.py:148`) reads `_class_name` from the
  checkpoint's `model_index.json` and does a dynamic `getattr(diffusers, ...)`.
  `comfyless/generate.py:903` consumes it. There is no hardcoded class table to
  extend.
- **`grep QwenImageEditPlusPipeline comfyless/` returns zero matches.** Wiring
  qwen-edit into comfyless is new integration work, not an extension of
  something already half-present.
- **The node pack has the hard parts solved.** `generate_qwen_edit`
  (`nodes/eric_diffusion_manual_loop.py:2373`) already accepts 1–N reference
  images with independent per-image VL/ref toggles, where the Nth entry becomes
  "Picture N" in the VL processor's prompt template. `_resize_ref_for_qwen_edit`
  (`:2268`) and `_calculate_qwen_edit_dimensions` (`:2254`) encode non-obvious
  resize/dimension rules. These are to be mined, not reinvented.
- **Qwen-Edit conditions on each reference along two independent paths**
  (`manual_loop.py:2396-2405`), and which paths are active is the single most
  consequential edit control:
  - **VL path (semantic)** — the image feeds the text encoder as part of the
    fused prompt embedding. This is what makes compositional prompts such as
    "the outfit from Picture 2" resolve.
  - **Ref path (pixel)** — the image is VAE-encoded and its packed latents
    concatenate into the transformer's `hidden_states`, so the model attends to
    reference pixels directly.

  Both default ON per image, with `vl_flags` / `ref_flags` (one bool per
  reference, `manual_loop.py:2378`) disabling either path per slot. **Disabling
  the Ref path is how the output stops mirroring the input's geometry.**
  Grant's worked example: a 3/4-view car reference rendered with both paths on
  reproduces the 3/4 view; dropping the Ref path leaves the model knowing what
  the car is semantically without being pinned to its source pixels, which is
  what permits a profile or head-on re-render. Both paths off means the model
  never sees that image at all — the node pack treats this as a soft-validation
  warning (`eric_diffusion_advanced_edit.py:393-401`), not an error.
- **`--lora` is argparse `action="append"`** (`comfyless/generate.py:1912`),
  parsed by `_parse_lora_arg` (`:2097`). This is the house convention for
  repeatable inputs.
- **There is a precedent for image ingestion with security caps.**
  `build_config_from_seed` (`comfyless/refine.py:1374`) takes an image path and
  gates it through `load_seed_image_capped` (`:348`) *before* trusting any
  embedded metadata: 64 MB byte cap (`SEED_IMAGE_MAX_BYTES`, `:53`), a
  decompressed-pixel cap, a prompt char cap, LoRA references reduced to
  basename→catalog resolution rather than honored as literal paths, and a loud
  echo of any load-bearing path outside operator roots.
- **The daemon's path-containment gate is `_check_paths`**
  (`comfyless/server.py:221`), enforcing `_within` against the union of model
  base, LoRA roots, and transformer roots. Schema-layer path fields
  (`params_schema.py:122`) are plain strings with no validation of their own;
  containment is entirely this downstream boundary's job.

### Documentation drift found while surveying

ADR-011 (`docs/decisions/ADR-011-comfyless-mcp-server.md:197`) states that an
`edit` tool was added to the MCP surface as a validating stub, "so the slot is
reserved on `tools/list`" and the real implementation could land without a
tool-list version bump. **That stub was never implemented.** `_list_tools_impl`
(`comfyless/mcp_server.py:1360`) registers five tools — `generate`,
`list_models`, `list_loras`, `list_transformers`, `extract_params`. There is no
`edit` entry.

Consequence: the tool-list bump ADR-011 tried to pre-pay for is still owed, and
will have to be spent deliberately when the MCP edit surface lands. This is
documentation/implementation drift, not a security defect — nothing insecure is
exposed; the tool simply is not there.

---

## Decision

**1. Reference images are a general schema surface, not an edit-only one.**

A new repeatable `--ref-image` flag (argparse `action="append"`, mirroring
`--lora` at `generate.py:1912`) taking `PATH[:MODE]`, and a corresponding
`ref_images` schema key. Order is significant and preserved — it maps onto the
"Picture N" convention already established at `manual_loop.py:2409`.

`MODE` carries the per-image conditioning-path selection (decision 2a) and is
parsed by splitting on the **last** colon, exactly as `_parse_lora_arg`
(`generate.py:2097`) does for `path:weight` — so paths containing colons parse
correctly. An unrecognized mode suffix is a hard error, not a silent fallback:
unlike a LoRA weight, guessing wrong here silently changes what the model sees.

Strictness inverts the residual risk onto the three *valid* suffixes: a file
literally named `frame:ref` parses as path `frame` + mode `ref` and silently
opens a different existing file, or errors with a misleading not-found. Two
requirements close this (security review, 2026-07-20): the always-safe escape
is documented — a path containing colons appends an explicit `:MODE`, so
`photo:ref:both` is unambiguous — and when the mode-stripped path does not
exist but the full spec does exist as a file, the error message must say so by
name rather than reporting a bare not-found.

**2. There is no edit mode. Edit is generation with reference conditioning.**

No `--mode` / `--edit` flag. What a reference image *means* is derived from the
model family, which is itself derived from the checkpoint. Routing table:

| Family | Meaning of a reference image |
|---|---|
| `qwen-edit` | Edit source — instruction-following against the reference |
| `flux2klein` | img2img / reference conditioning (style and subject carry) |
| Other families | img2img **iff the auto-detected pipeline class natively accepts an image input** (decision 3 constrains what "accepts" may mean) |
| Unsupported | Reference is dropped — behaviour depends on invocation mode, below |

What "dropped" means depends on who is watching (security review finding,
2026-07-20):

- **Interactive CLI:** loud warning naming the family and stating the reference
  was dropped, then proceed — warn-don't-block. A silent drop is a defect.
- **Plan / `--json` / machine-driven modes:** **hard failure** (or
  fail-the-segment under a video plan). The first consumer is ADR-033 chained
  plan execution, which runs unattended: a dropped reference there means
  keyframe N+1 is generated ignoring keyframe N and the error propagates
  through the rest of the chain with nobody reading warnings. The decision 2a
  rationale applies verbatim — warn-don't-block protects against footguns a
  user can stumble into; in machine mode nobody stumbles.

In every mode, the sidecar records per reference whether it was **applied or
dropped** (decision 7). A sidecar listing references that had no effect is a
false provenance record, and replay would then replay a lie.

**The mode split is carried across the wire, fail-closed** (2026-07-20
re-review, Finding 4). The daemon performs family routing and so is where a
drop is *discovered*, but plan workers and interactive runs reach it over the
same socket (`video.py:738`, `generate.py:2455`) — the daemon cannot itself
tell "machine-driven" from "interactive." So strictness travels as an explicit
request field whose **absent-value default is hard-fail**. Only the interactive
CLI, on a TTY, sets the lenient (warn-and-proceed) value; every other client —
plan workers, scripts, a future agent bridge — inherits the strict default by
omission. A lenient-by-default wire flag would silently hand machine clients the
corrupted-chain failure this decision exists to prevent. Applied/dropped is
recorded in wire metadata and the sidecar in every mode regardless.

**2a. Per-image conditioning paths are exposed as `MODE`, defaulting to both.**

`--ref-image PATH:MODE` where `MODE` is one of:

| `MODE` | VL path | Ref path | Meaning |
|---|---|---|---|
| `both` (default) | on | on | Scene lock — output follows the reference's geometry |
| `vl` | on | off | Semantic only — subject/content carries, geometry does not |
| `ref` | off | on | Pixel only — structure carries without entering the prompt embedding |

Omitting `:MODE` selects `both`, preserving the node pack's defaults and making
the common case terse.

This is the control that determines how closely output mirrors input, and it is
the first consumer's core knob: keyframe evolution with scene lock is `both`;
changing a subject's viewpoint or orientation between keyframes is `vl`.
Exposing it is not optional — without it, comfyless can only produce edits that
reproduce the source geometry, which is precisely the case keyframe authoring
needs to escape.

There is deliberately no `none` mode, and any `MODE` value outside `both` / `vl`
/ `ref` is an error rather than a warning-and-default. This is a considered
departure from the project's general warn-don't-block posture, justified by the
difference in interaction model between the two surfaces:

In the node graph, "ignore this image" is a genuinely useful state — Grant's
usage was loading four images and toggling among them while experimenting, where
temporarily silencing one slot beats rewiring the graph. The node pack
accordingly warns rather than errors when both flags are off
(`eric_diffusion_advanced_edit.py:393-401`).

At a CLI there is no persistent wiring to preserve. Excluding a reference from
the set is a new invocation with that flag omitted, which is strictly less
typing than spelling out a mode that means "ignore this". The state is therefore
unreachable by accident — it can only be typed deliberately — and a silent
default would let a typo'd mode quietly change what the model sees. Warn-don't-
block protects against footguns a user can stumble into; this is not one.

Families whose reference conditioning has no such dual-path structure — for
example `flux2klein` — accept only `both`, and reject `vl` / `ref` with an error
naming the family. This is one of the few places the unified surface must admit
that families differ, and it is better surfaced as a hard error than silently
collapsed.

**3. Family routing is derived from the model, never overridden.**

*(Distinct from decision 2a's per-image `MODE`, which selects conditioning paths
within a family. This decision is about which family semantic applies at all.)*

There is no `--ref-mode` flag. This is load-bearing for daemon correctness:
`_request_cache_key` (`comfyless/server.py:397`) keys on `req["model"]` among
other fields, and pipeline class is a deterministic function of the model path
via `model_index.json`. Same path → same class, always. **The existing cache key
is therefore already sufficient and needs no discriminator.** An override that
changed pipeline class for the same weights would silently break that
invariant — which is the decisive argument against having one.

The security review (2026-07-20) verified this claim and found it **only
conditionally true**: it holds because — and only while — class selection
happens exactly once. So the condition is stated here as a binding
implementation constraint, not left as an implicit conclusion:

> **Invariant: pipeline class is selected exactly once, in
> `detect_pipeline_class` (`eric_diffusion_utils.py:148`), at load time. The
> presence of reference images must never trigger a class swap or an
> `AutoPipelineForImage2Image.from_pipe` conversion on a cached pipeline.**

The pressure point is the routing table's "other families" row: for most
families the `model_index.json` class is the text2img pipeline whose `__call__`
does not accept an image (`FluxPipeline` vs `FluxImg2ImgPipeline`). The
tempting implementation — converting to the img2img sibling at request time —
makes pipeline class a function of *(model path, ref-images-present)* and the
un-discriminated cache then serves a text2img pipeline to an img2img request or
vice versa. Instead: a family whose detected class does not accept an image
takes the drop path of decision 2. comfyless does **not** quietly upgrade it.
If a future slice wants sibling-class img2img for such families, that slice
MUST add a cache-key discriminator in the same commit — recorded here so the
constraint travels with the decision.

The implementation slice adds a test pinning this invariant, in the style of
the NAG cache-key pins in `test_server_robustness.py`.

**4. Strength is one flag with family-dependent meaning.**

`--strength` applies to img2img-style conditioning. On `qwen-edit`, where true
instruction editing does not consume a denoise strength, passing it produces
the same loud warning-and-proceed treatment as an unsupported reference image.

**5. Dispatch branches below the schema surface.**

One schema, one CLI surface, one sidecar shape. Underneath, dispatch routes to
`generate_qwen_edit` versus the generic loop, exactly as sampler selection
already branches. The resize and dimension rules are mined from
`_resize_ref_for_qwen_edit` / `_calculate_qwen_edit_dimensions`, not rewritten.

**6. Image ingestion extends — not merely mirrors — the `refine.py`
seed-image mitigations, with a dedicated ref-image root allowlist.**

*(Rewritten after the 2026-07-20 security review. The original text — "reference
paths pass through the existing `_check_paths` gate; no new containment
mechanism is invented" — was wrong twice over: it conflated two trust classes,
and it would have rejected every legitimate keyframe.)*

**6a. The daemon gets a separate `ref_image_roots` allowlist.** `_check_paths`
(`server.py:221`) validates against `{model_base} ∪ lora roots ∪ transformer
roots` — the set of trees the daemon may **deserialize model weights** from.
Reference images for the first consumer live in output/working directories,
which are in none of those roots; reusing the weight allowlist would reject
every legitimate keyframe, and the two obvious "fixes" are both security
regressions (exempting `ref_images` from containment entirely, or adding
`output_dir` to the weight roots — a tree that lower-trust flows write into,
reopening the invariant the 2026-06-01 refiner-path CRITICAL closed).

Image-read roots and weight-load roots are different permissions. The daemon
therefore validates `ref_images` against its own root set: **`ref_image_roots`
= its `--output-dir` plus explicit operator additions via a `--ref-root` spawn
flag.** The mechanism (`_within` union) is shared code; the allowlists are
disjoint in purpose and never merged. The default is exactly what keyframe
chaining needs — keyframes are prior segment outputs in the output dir.

**`ref_image_roots` is also defined off-daemon** (2026-07-20 re-review, Finding
1). `ref_image_roots` cannot be a daemon-only concept, because decision 7's
cold-path refusal executes where no daemon and no spawn flags exist — refine
seed entry, `--params` replay, foreground generation. On those paths
**`ref_image_roots` = the invocation's output directory ∪ `--ref-root` (exposed
as a CLI flag as well as a daemon spawn flag) ∪ the operator weight/catalog
roots** (the set refine already resolves against, `refine.py:1486`). Without
this, decision 7's refusal would reject every legitimately replayed keyframe
sidecar — refs live in the output dir, not the weight roots — recreating
blocker (a)'s over-refusal on the cold path.

**Guard `--ref-root` breadth** (re-review Finding 6). A `--ref-root` of `/`, a
mount root, or the user's home directory makes any user-readable image on the
machine readable by any same-UID wire client for the daemon's lifetime — and
VAE-encodable into shareable output. Unlike `--model-base` (useful reads are
limited to weight-shaped trees), the ref surface makes a broad root maximally
exploitable. At spawn, such a root produces a loud warning naming the exposure
and proceeds (warn-don't-block — this root is operator-typed, not attacker-
reachable); each configured ref root is logged at startup as `server.py:982`
already does for extra roots.

**6b. Caps are enforced in the process that decodes.** Paths, not bytes, cross
the socket, so for daemon requests the daemon opens and decodes the file — and
any same-UID process speaking the wire protocol directly bypasses the CLI
argument layer entirely. The byte cap, pixel cap, and format allowlist
therefore execute at the decode site: one shared helper (patterned on
`load_seed_image_capped`, `refine.py:348`) called in whichever process performs
the decode — the CLI process on foreground runs, the daemon's generate path on
daemon runs — after whatever containment treatment decision 7 assigns to the
path's trust class, and before PIL touches the file. Containment is *not*
unconditional inside this helper: the caps and format allowlist always run, but
a typed-at-CLI path (decision 7 row 1) carries no containment gate, so the
helper must not impose `ref_image_roots` on every path indiscriminately. A
CLI-side pre-check for fast feedback is UX, not the gate.

**6c. Format allowlist and single-frame semantics.** Reference images are
arbitrary user files, not comfyless-authored PNGs like refine's seed images —
so `Image.open` must not dispatch across PIL's full plugin zoo (the EPS plugin
shells out to Ghostscript; the rarely-exercised C decoders carry most of
Pillow's CVE history). Decode is pinned to an explicit allowlist —
`formats=["PNG", "JPEG", "WEBP"]` — and multi-frame containers contribute their
first frame only, so per-frame pixel accounting cannot be gamed. Keyframe
authoring needs nothing beyond these three formats.

**6d. Single-read design.** The file is read **once** into memory (the byte cap
makes this safe by construction); the sidecar hash (decision 7) is computed
over those bytes; those same bytes are decoded. Validation, hash, and decode
then all describe the same content — no check-then-use window between
`realpath`, hashing, and `Image.open`.

**6e. `ref_images` joins the daemon's NUL-byte defense.** `_PATH_FIELDS`
(`server.py:52`) covers scalar path fields and `loras[].path` ad hoc; a NUL in
an unchecked path raises inside `os.path.realpath` and kills the accept loop —
a one-request daemon DoS. The list-shaped `ref_images` field gets the same
pre-check, with a negative test mirroring the existing `loras[i].path` NUL
test.

**6f. Per-request reference-count cap (= 8).** 6d reads each reference wholly
into memory, so N references × the 64 MB byte cap compose into no aggregate
bound — a single wire request listing a few hundred in-root images (the
daemon's own output dir supplies them) drives multi-GB transient allocations
plus N decoded tensors inside the VRAM-holding daemon (re-review Finding 5).
The request is refused above **8** references. The bound is generous for the
first consumer — qwen-edit realistically uses 1–3, and "Picture N" prompting
degrades well below 8 — while capping the daemon's transient at 8 × 64 MB.
Enforced at the same site as the 6b caps, with a negative test.

**6g. Ingestion errors report path and reason only, never file content.**
Decode-failure, cap-exceeded, and containment-refusal messages name the
offending path and the reason; they never echo file bytes. This keeps the
format allowlist's usefulness intact (a co-located sidecar JSON fed as a ref
fails decode without leaking its contents) and denies an oracle that would turn
attacker-directed reads into attacker-*readable* reads (re-review Finding 7).

**7. Sidecars record refs fully; replay treats recorded paths as untrusted.**

*(Rewritten after the 2026-07-20 security review, which found the original text
specified only the recording side — the replay-side trust treatment is the
actual security content.)*

**Recording.** Each reference is recorded with its path, its SHA-256 (computed
over the exact bytes decoded, per decision 6d), its `MODE`, and whether it was
**applied or dropped** (decision 2). A moved reference at replay is a loud
warning; a hash mismatch is a louder one. Replay never attempts to relocate a
missing reference.

**Replay.** `refine.py`'s decisive mitigation for path-shaped metadata was
never honoring it as a path — LoRA references are reduced to basename and
re-resolved through the ADR-015 catalog (`refine.py:1443`). Reference images
have no catalog to launder through, so a replayed sidecar's `ref_images`
entries can only ever be honored as literal filesystem paths — and the metadata
that carries them is attacker-craftable (the F4 channel
`build_config_from_seed` documents). That channel is the JSON **sidecar** for
every output format, plus the PNG `tEXt` chunk when output is PNG; once JPEG
output exists (ADR-034), JPEG carries provenance in the sidecar *only* (no tEXt
chunk), so the sidecar is the sole and sufficient channel to treat as
untrusted. The three-trust-class table below is written in terms of the path's
*source* (typed / file-derived / wire), not the container it arrived in, so it
already covers both — this note exists only so the PNG-specific wording is not
mistaken for a PNG-specific threat. The content hash is no defense: the
attacker writes the hash to match. Without further treatment, a crafted sidecar
directs comfyless to read any user-readable file and VAE-encode its bytes into
the conditioning of an image that may later be shared.

Reference paths therefore carry one of three trust classes, each with its own
treatment:

| Source of the path | Treatment |
|---|---|
| Typed at the CLI (`--ref-image`) | User authority — no containment gate (it would add no security and break legitimate use); loud echo if outside known roots |
| File-derived (sidecar replay, `--params`, refine seed entry) | Mandatory F4 treatment: loud echo of every ref path with an outside-roots flag **before** generation (extending the `_SEED_ECHO_PATH_FIELDS` mechanism, `refine.py:1352`, to `ref_images`); a path outside `ref_image_roots` ∪ operator roots is **refused** on the cold path with instructions to re-specify it on the command line |
| Crossing the daemon socket | `ref_image_roots` containment (decision 6a), regardless of how the client obtained the path |

The refusal in the second row is stricter than warn-don't-block, justified the
same way as strict `MODE` validation: nobody stumbles into replaying a crafted
sidecar, and the attack only works if it is silent. Retyping the path at the
CLI converts it to the first row — deliberate user authority. The cold
in-process path has no `_check_paths` gate at all (`refine.py:1489`), which is
precisely why the file-derived row cannot rely on downstream containment.

**Rows 1 and 3 do not conflict — the CLI resolves the seam by choosing where to
run** (2026-07-20 re-review, Finding 2). The interactive CLI silently delegates
to a running daemon whenever one is up (`generate.py:2444`), and daemons are
long-lived here (ADR-020) — so a naive reading has the same typed
`--ref-image ~/photos/car.jpg` succeed daemonless and get refused (row 3
containment) when a daemon happens to be running, breaking row 1's authority in
the machine's normal state. The resolution is the pattern already in the
codebase for `--output` (`generate.py:2450`): **a typed `--ref-image` that
falls outside `ref_image_roots` causes the CLI to skip daemon delegation and
run in-process**, where row 1 grants user authority with no containment gate.
Row 3's containment then governs only paths that actually cross the socket, and
both guarantees hold simultaneously. **Trust class is determined solely by the
boundary a path arrives through — it is never carried or honored as a wire /
request field.** A `typed_by_user` flag on the wire would be a client-asserted
trust claim any same-UID client could forge, gutting row 3.

**Two boundaries this table deliberately does not open** (re-review Finding 8):

- **`ref_images` is excluded from the ADR-027 planner-override allowlist.** The
  refine loop's LLM planner is barred from path-shaped keys by that closed
  two-key allowlist (ADR-027 F1); adding `ref_images` to it would create a
  fourth, LLM-directed trust class with none of the three treatments above
  applying, and requires its own security review. Not in this ADR.
- **Typed refs replace file-derived refs wholesale; they never merge.** On the
  refine CLI, a typed `--ref-image` replaces the entire seed-derived
  `ref_images` set rather than appending to it. Merging would let a crafted
  sidecar's extra reference ride alongside the user's typed one under row-1
  coloration — a file-derived path masquerading as typed.

---

## Alternatives Rejected

### Separate edit and generate surfaces (the node-pack shape)

The ComfyUI node pack maintains a hard split: `QWEN_EDIT_PIPELINE` versus
`QWEN_IMAGE_PIPELINE` as distinct wire types, with separate loaders, separate
LoRA stackers, and separate generate nodes at every stage. The obvious move was
to mirror that structure in comfyless.

**Rejected — the split is an artifact of ComfyUI's host constraints, not a
property of diffusion.** ComfyUI's graph is statically typed on wires; a node
accepting `QWEN_EDIT_PIPELINE` cannot also accept `QWEN_IMAGE_PIPELINE`. That
single constraint forces the entire duplicated cascade. comfyless has no wires,
one argparse surface, one schema, one dispatcher. The forcing constraint is
absent.

Examined difference by difference, nothing survives to justify a split:

| Difference | Requires a split? |
|---|---|
| Pipeline class | **No** — `detect_pipeline_class` already resolves it from `model_index.json`; free |
| Reference images | **No** — an additional optional schema key, not a different schema |
| Output dims from ref aspect ratio | **No** — a family-conditional defaulting rule, which `family_defaults.py` already does |
| `true_cfg_scale` 4.0 / steps 30 | **No** — `family_defaults.py` *already carries the `qwen-edit` entry* |

Historical note, recorded so this is not re-litigated: the node-pack split was
retained at the time because the shape already existed, and the decision predates
comfyless entirely — the project was then a fork of Eric's nodes aimed at
generalizing across families for base-model generation. The split was never
evaluated against a CLI surface, because there was no CLI surface to evaluate it
against.

Accepted cost: unified surfaces degrade badly on wrong input if error handling
is sloppy. Mitigated by decision 2's requirement that unsupported combinations
warn loudly and name the family.

**Independent confirmation — the Style Transfer node is a preset, not a
mechanism.** `nodes/eric_qwen_edit_style_transfer.py` looks like a third
operation alongside generate and edit, but its entire substance is a default
arrangement of decision 2a's two knobs (`style_transfer.py:142-158`): the style
image is VL-only (`ref_style=False`, "preventing structure bleed") and the
content image is both, wrapped in style-oriented prompt templates. Under this
ADR it needs no special support and no preset machinery — it is
`--ref-image style.png:vl --ref-image content.png:both` with a style prompt.

This is the same ComfyUI artifact as the generate/edit split one level down: it
is a separate *node* because a graph needs a node to hold defaults, not because
it is a separate *operation*. Two independent instances of the same pattern is
the strongest evidence available that the node-pack topology should not be
carried across.

### A `--ref-mode` override

Rejected under decision 3 — it would break the daemon cache-key invariant by
letting pipeline class vary independently of model path.

### Numbered `--ref-image1 --ref-image2` flags

Rejected — caps arity arbitrarily, diverges from the established `--lora`
convention, and Qwen-Image-Edit-2511 is the *Plus* multi-reference variant where
1–N with meaningful ordering is the native shape.

---

## Deferred / Out of Scope

- **MCP edit surface.** ADR-011 and ADR-015 hold that no filesystem paths cross
  the MCP boundary in either direction, but an uploaded image to edit is
  inherently a path or blob. Resolving that — an opaque handle for uploads, a
  server-managed upload directory, or something else — is its own ADR, and
  carries ADR-011's deferred image-upload-as-seed security review with it. It
  also now owes a tool-list surface bump, since the reserved stub was never
  implemented. Nothing in the video roadmap is blocked on it.
  Two awareness items travel with it (security review 2026-07-20 + re-review,
  INFO):
  - The unkeyed SHA-256 of a reference file in sidecar metadata is a
    content-confirmation fingerprint, and absolute ref paths disclose filesystem
    layout — acceptable for a solo desktop tool, but the MCP ADR (where outputs
    become agent-visible) must inherit that trade-off deliberately, not by
    default.
  - **Output-dir read-back / cross-plant loop.** With the output dir a default
    ref root (decision 6a), any wire client can read back any decodable image
    any session or daemon wrote there, and client-chosen savepath naming
    (`server.py:784`) lets one flow plant an image another reads by name. Under
    the solo same-UID model this *is* the intended chaining feature and adds no
    authority; once agent-driven flows front the daemon it becomes a
    cross-session read/plant channel the MCP ADR must weigh. The 6c format
    allowlist already limits the primitive to decodable images (co-located
    sidecar JSONs fail decode).
- **Restoring the ADR-011 `edit` stub** — pointless as a separate act; the slot
  will be filled by the real implementation when the MCP ADR lands.
- **Edit support for families beyond `qwen-edit` and `flux2klein`** — the
  routing table admits them, but only these two are validated here.
- **Multi-reference semantics beyond positional "Picture N"** — no named or
  role-tagged references in this slice.
- **`vae_target_size`** — the style-transfer node's control over the resolution
  references are VAE-encoded at (`style_transfer.py:136`). Its own documentation
  states that the default (encode at output resolution) matches Edit-node
  behaviour and is best for high-res, and that a fixed value is "only useful at
  low output res". The first consumer authors keyframes at 1280×704, where the
  default is correct. Deferred until a concrete case appears.
- **Style-transfer prompt templates** (`STYLE_TEMPLATES`,
  `style_transfer.py:36`) — useful prompt content, but a prompt-library concern
  rather than a schema one. The flag combination they accompany is already
  expressible per decision 2a.
- **Retiring `EricQwenEditInpaintTransfer`** — registered at
  `nodes/__init__.py:28,63` and untouched since the initial release commit
  (`79c12b9`); never used in practice. Removing a registered node is a separate
  decision on the node-pack surface, not part of the comfyless schema work.
- **`resolve_hf_path` §12 security review** — long-standing debt, untouched by
  this ADR.

---

## Changelog

- 2026-07-20 — Proposed. Scope confirmed as CLI + daemon with MCP deferred;
  video keyframe authoring as the first consumer. Records the generate/edit
  split as explicitly rejected, and the ADR-011 `edit`-stub drift as found.
- 2026-07-20 — Added decision 2a (per-image VL/Ref conditioning paths as
  `PATH:MODE`) after Grant recalled the node-pack control governing how closely
  output mirrors the input image. Decision 1's flag shape widened from bare
  `PATH` to `PATH[:MODE]` to carry it; decision 3 retitled to "family routing"
  to avoid colliding with the new `MODE` term.
- 2026-07-20 — Recorded the Style Transfer node as a second instance of the
  rejected-split pattern (a preset over decision 2a's knobs, not a distinct
  operation). Deferred `vae_target_size`, the style prompt templates, and the
  question of retiring the unused inpaint-transfer node.
- 2026-07-20 — Confirmed strict `MODE` validation (error, not warning) with the
  interaction-model rationale recorded in decision 2a: the ignore-this-image
  state earns its keep in a persistent graph but not at a per-invocation CLI.
  Inpaint-transfer removal logged in `TECH_DEBT.md`. Design settled; next step
  is `security-auditor` per §12 before any implementation.
- 2026-07-20 — Amended against the security review
  (`docs/security/review-adr-035-reference-image-surface-2026-07-20.md`), which
  found four blockers, all textual: decision 6 rewritten (6a: dedicated
  `ref_image_roots` allowlist replacing the wrong reuse of the weight-root
  `_check_paths` set, which would have rejected every keyframe; 6b: caps
  enforced at the decode site, daemon-side for daemon requests; 6c: PIL format
  allowlist PNG/JPEG/WEBP + first-frame semantics; 6d: single-read design
  resolving TOCTOU and hashing together; 6e: NUL-byte defense for
  `ref_images`); decision 3's cache-key claim narrowed to its true scope and
  its condition stated as a binding invariant with a pinning test, including
  the no-`from_pipe`-upgrade rule; decision 2's drop behaviour split by
  invocation mode (interactive warn-and-proceed, plan/`--json` hard-fail) with
  per-ref applied/dropped recorded in sidecars; decision 7 rewritten around
  the three-trust-class table, refusing file-derived outside-roots paths on the
  cold path; decision 1 gains the colon-filename escape and error-naming
  requirement; MCP fingerprint awareness noted in Deferred.
- 2026-07-20 — **Accepted** after a re-review
  (`docs/security/review-adr-035-rereview-2026-07-20.md`) confirmed all four
  blockers closed and surfaced two new HIGHs inside the amendments, now landed:
  Finding 1 — `ref_image_roots` defined off-daemon (output dir ∪ `--ref-root`
  CLI flag ∪ weight/catalog roots) so decision 7's cold-path refusal doesn't
  reject every replayed keyframe, with `--ref-image` required on replay-capable
  surfaces; Finding 2 — a typed `--ref-image` outside roots skips daemon
  delegation and runs in-process (the `--output` precedent), and trust class is
  never a wire-asserted field. Plus the MEDIUM/INFO carries: 6b defers
  containment to decision 7's class assignment (Finding 3); the mode strictness
  split crosses the wire fail-closed, absent = hard-fail (Finding 4); a
  per-request ref-count cap of 8 (Finding 5, decision 6f); a `--ref-root`
  breadth warning (Finding 6); errors report path+reason only (Finding 7,
  decision 6g); output-dir read-back loop willed to the MCP ADR; and
  `ref_images` excluded from the ADR-027 planner allowlist with
  typed-replaces-file-derived (Finding 8, decision 7).
- 2026-07-20 — Coupling note with ADR-034 (JPEG output, adopted same session,
  sequenced first). Decision 7's replay-channel prose corrected: the
  attacker-craftable metadata channel is the JSON sidecar for all formats (plus
  the PNG tEXt chunk only when output is PNG) — not PNG-specific. The
  three-trust-class table was already source-based and needs no structural
  change. Reconciliation points for the later ref-image slices 4–5: ADR-034 D6
  (`--params` .png/sidecar dispatch, `generate.py:172`), D7 (refine canonical
  path), D4 (sidecar as the sole JPEG provenance channel) — all land in
  ADR-034's own slices and are consumed, not duplicated, by ref-image replay.

---

**AI-Disclosure:** Claude (Opus 4.8) authored; Grant reviewed.
