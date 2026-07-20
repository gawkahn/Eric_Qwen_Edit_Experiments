# Security Review — ADR-035 Reference-Image Surface (design review)

**AI-Disclosure:** Claude (Fable 5, `security-auditor` agent) authored; Grant reviewed.
**Date:** 2026-07-20
**Artifact reviewed:** `docs/decisions/ADR-035-comfyless-reference-image-surface.md` (Status: proposed)
**Trigger:** §12 — image ingestion from caller-supplied paths; paths crossing the CLI→daemon boundary.
**Scope excluded:** MCP surface / agent upload naming; ADR-011's deferred image-upload-as-seed review (both deferred to a later ADR).

---

## Summary

ADR-035 proposes a reference-image surface for comfyless: a repeatable
`--ref-image PATH[:MODE]` CLI flag and a `ref_images` schema key, with images read
from local paths, decoded via PIL, then VAE-encoded and/or fed to the Qwen-VL text
encoder; paths also cross the CLI→daemon Unix socket. The trust boundaries touched:
(1) arbitrary user files → PIL decode → model conditioning (new — refine's precedent
only ever decoded comfyless-authored PNGs), (2) client → daemon over the 0600 same-UID
socket, where `_check_paths` is today the only containment gate, (3) untrusted
sidecar/PNG-embedded metadata → replay (decision 7), which for ref images has no
catalog indirection to fall back on. The design gets a lot right (strict MODE
validation, mirroring the F5 byte+pixel gates, order-preserving append flags), but
three decisions are underspecified or wrong as written: decision 6's "reference paths
pass through the existing `_check_paths` gate, no new containment mechanism is
invented" conflates two different trust classes and likely breaks the first consumer;
decision 3's cache-key claim is only conditionally true and the ADR's own routing table
creates pressure to violate the condition it depends on; and decision 6/7 never say
**which process** enforces the decode caps or how replayed ref paths are treated.

Verified during review: the actual `_check_paths`/`_within` implementation and its root
set (`comfyless/server.py:214-266,381`), the null-byte defense and `_PATH_FIELDS`
(`server.py:52-60,199-206`), `_request_cache_key` (`server.py:397-448`), socket
permissions (`server.py:972-977`, 0600 in 0700 dir → same-UID clients only),
`detect_pipeline_class` (`nodes/eric_diffusion_utils.py:148-184`) to verify/refute
decision 3, `_parse_lora_arg` (`comfyless/generate.py:2097-2107`) for the colon-split
precedent, `_load_pipeline` (`generate.py:873-949`) for where class selection happens,
and the refine precedent the ADR mirrors (`comfyless/refine.py:48-96,348-375,1374-1512`)
including the cold-path no-gate comment at `refine.py:1488-1497`.

## Coverage

Reviewed:
- `docs/decisions/ADR-035-comfyless-reference-image-surface.md` (full, the artifact under review)
- `comfyless/server.py:42-60, 150-266, 278-448, 972-980`
- `comfyless/refine.py:40-96, 330-439, 1374-1533`
- `comfyless/generate.py:860-949, 2060-2160`
- `nodes/eric_diffusion_utils.py:120-208`

Not reviewed (and why):
- MCP surface / agent upload naming — out of scope per the review brief and the ADR's own Deferred section.
- `nodes/eric_diffusion_manual_loop.py:2373` (`generate_qwen_edit`) internals — the ADR mines it for resize math, not for the trust boundary; not load-bearing for this review.
- `comfyless/params_schema.py:122` — not read directly; the ADR's characterization ("plain strings, no validation, containment is downstream's job") is consistent with the server-side code that was read.
- `nodes/eric_qwen_edit_style_transfer.py`, `eric_diffusion_advanced_edit.py` — cited by the ADR for UX rationale only.

---

## Findings

### [HIGH] Decision 6 reuses `_check_paths` for a different trust class — and breaks the first consumer

**Location:** ADR-035 decision 6; `comfyless/server.py:221-266,381`

`_check_paths` validates against `{model_base} ∪ lora roots ∪ transformer roots` — the
allowlist of trees the daemon may **deserialize model weights** from. Reference images
for the first consumer (video keyframes) live in output/working directories, which are
in none of those roots, so as designed every legitimate daemon-side ref image is
rejected. The predictable failure mode at implementation time is one of two bad fixes:
(a) exempt `ref_images` from `_check_paths` entirely — giving any same-UID wire client
an unvalidated daemon-side file read whose decoded content flows into output pixels (a
confused-deputy read once agent-driven flows front the daemon), or (b) add `output_dir`
to the shared roots union — which silently also permits **model/LoRA loading** from the
output directory, a tree that lower-trust flows (MCP-driven generation) write into.
Both weaken the invariant the 2026-06-01 CRITICAL refiner-path finding was closed on.

"No new containment mechanism is invented" is the wrong decision: image-read roots and
pickle-load roots are different permissions and need separate allowlists (e.g. a
spawn-time ref-image root set, defaulting to output_dir + explicit operator additions),
even if the mechanism (`_within` union) is shared code.

**Verdict: BLOCKS** moving to accepted. The decision text must be amended before
implementation, or the first implementation slice will have to improvise exactly this
policy under deadline.

### [HIGH] Decision 3's cache-key claim is only conditionally true, and the ADR's routing table undermines the condition

**Location:** ADR-035 decision 3 + decision 2 routing table; `comfyless/server.py:397-448`; `nodes/eric_diffusion_utils.py:148-184`; `comfyless/generate.py:903`

Verified as far as it goes: `detect_pipeline_class` is deterministic over
`model_index.json` content, `_load_pipeline` selects the class exclusively from it, and
ref images/MODE/strength are per-call inputs (like prompt and the NAG params
deliberately excluded at `server.py:410-415`) that don't change pipeline shape — so
**for `qwen-edit` and any family whose detected class natively accepts image input, the
existing key is sufficient.**

But the claim breaks on the routing table's third row, "Other families → img2img where
the auto-detected diffusers pipeline accepts it": for most families the
`model_index.json` class is the text2img pipeline whose `__call__` does **not** accept
an image (e.g. `FluxPipeline` vs `FluxImg2ImgPipeline`). An implementer serving that row
will reach for `AutoPipelineForImage2Image.from_pipe` or a sibling-class swap at request
time — at which point pipeline class is a function of (model path, ref-images-present)
and the un-discriminated cache serves a text2img pipeline to an img2img request or vice
versa.

The ADR states the invariant's conclusion ("same path → same class, always") without
stating it as a **constraint on implementation**: class selection happens only in
`detect_pipeline_class` at load; ref-image presence must never trigger a class swap or
`from_pipe` conversion; families whose detected class rejects images take the
warn-and-drop path; any future sibling-class img2img slice must add a cache-key
discriminator in the same commit.

**Verdict: BLOCKS** until that invariant sentence (plus a test pinning it, in the style
of the NAG pin in `test_server_robustness`) is added. Cheap amendment, but load-bearing
— this is precisely the silent-wrong-pipeline failure the decision claims to have
designed away.

### [HIGH] Enforcement locus of the byte/pixel caps is unspecified — must be daemon-side, before decode

**Location:** ADR-035 decision 6; `comfyless/server.py:45,972-977`; `comfyless/refine.py:348-375`

Paths, not bytes, cross the socket, so the **daemon** opens and decodes the file. If the
F5-style byte/pixel gates run only in the CLI argument layer, any same-UID process
speaking the wire protocol directly (the MCP server, a script, a future agent bridge)
bypasses them, and a decompression bomb or 2-gigapixel TIFF is decoded inside the
long-lived, VRAM-holding daemon — the one process whose crash/OOM costs the most. The
ADR's own architecture note (`params_schema` fields are unvalidated strings; "containment
is entirely this downstream boundary's job") argues the same for caps: the gate belongs
at every decode site, which means in the daemon's generate path, with the CLI check at
most a courtesy duplicate.

**Verdict: BLOCKS** as a one-sentence amendment to decision 6: "caps are enforced in the
process that decodes, i.e. daemon-side for daemon requests."

### [HIGH] "Mirror refine.py" is insufficient for replay: ref paths in sidecars are untrusted metadata honored as literal paths, with no catalog indirection available

**Location:** ADR-035 decision 7; `comfyless/refine.py:1374-1512` (esp. `:1443-1471` LoRA basename→catalog, `:1488-1497` cold-path no-gate)

The refine precedent's key mitigation for path-shaped metadata was **never honoring it as
a path** — LoRA refs are reduced to basename and re-resolved through the ADR-015 catalog.
Reference images have no catalog; a replayed sidecar's `ref_images` entries can only be
honored as literal filesystem paths. That makes decision 7 a new channel where untrusted
embedded metadata (a PNG chunk anyone can craft — the F4 channel `build_config_from_seed`
documents) drives file reads: on the cold in-process path there is **no `_check_paths`
gate at all** (`refine.py:1489-1491` — the same hole applies here, verbatim), so a crafted
sidecar/seed image can direct comfyless to read any user-readable file (`~/.ssh/id_rsa`, a
browser profile) and VAE-encode its bytes into the conditioning of an image that may later
be shared. The exfiltration path is indirect and lossy, but the read itself is
attacker-directed. The content hash does not help — the attacker writes the hash to match.

Minimum design requirement, stated in the ADR: replayed/sidecar-sourced ref paths get the
mandatory F4-style loud echo with the outside-operator-roots flag (extend
`_SEED_ECHO_PATH_FIELDS`-equivalent to `ref_images`), and the seed/replay entry path
treats a ref path outside the roots as at least warn-loudly-before-first-generation,
ideally require re-specification on the command line.

**Verdict: BLOCKS** — decision 7 currently specifies only the recording side and
moved-file/hash-mismatch warnings; the replay-side trust treatment is the actual security
content and is absent.

### [MEDIUM] Byte + pixel caps are not enough for arbitrary user files — a decoder/format allowlist is needed

**Location:** ADR-035 decision 6; `comfyless/refine.py:348-375`

`load_seed_image_capped`'s gates (byte cap, header-size pixel gate before decode) handle
classic decompression bombs, and were adequate because seed images are in practice
comfyless-authored PNGs. Ref images are **arbitrary user files**, so `Image.open`
dispatches across PIL's full plugin zoo: the EPS plugin shells out to Ghostscript on load
(a historical RCE vector when gs is installed — assumption: gs presence on this desktop is
plausible), and the rarely-exercised C decoders (TIFF, FLI, etc.) carry most of Pillow's
CVE history. The caps bound resource use, not decoder attack surface. The design should
pin an explicit format allowlist at open time (`Image.open(path, formats=["PNG", "JPEG",
"WEBP"])` semantics — keyframe authoring needs nothing else) so exotic plugins are
unreachable. Also specify single-frame semantics (first frame of multi-frame containers)
so per-frame pixel accounting can't be gamed.

**Verdict:** TECH_DEBT precondition at minimum; recommend folding the allowlist sentence
into decision 6 before acceptance since it is one line and this is the concrete way ref
images differ from the refine precedent.

### [MEDIUM] `ref_images` must join the daemon's null-byte defense or a NUL in a path kills the accept loop

**Location:** `comfyless/server.py:52-60` (`_PATH_FIELDS` comment), `:199-206`

The server's own comment records why the NUL check exists: `os.path.realpath` raises on
NUL and the exception escapes `_check_paths` and kills the accept loop — a one-request
daemon DoS. `_PATH_FIELDS` covers scalar fields and `loras[].path` is handled ad hoc; a
list-shaped `ref_images` field is covered by neither. The ADR does not mention it
(absence finding).

**Verdict:** TECH_DEBT precondition — name it in the ADR/spec so the implementation
slice's negative tests include it (mirror the `loras[i].path` NUL test).

### [MEDIUM] TOCTOU between validation/hash and decode — resolve with a single-read design

**Location:** ADR-035 decisions 6–7; `comfyless/server.py:214-218`; `comfyless/refine.py:356-375`

Three separate touches of the same path are implied — `_check_paths` realpath, hash
computation (decision 7), and PIL open/decode — each a check-then-use window in which a
same-UID-writable path component can be symlink-swapped. Exposure is modest under the
current threat model (0600 socket, same-user attacker can already do worse), consistent
with the accepted model-path TOCTOU, and there is precedent for going further (the
slice-DQ H-1 symlink refusal). The clean design answer covers both the TOCTOU and hash
questions at once: read the file **once** into memory (the byte cap makes this safe by
construction), compute the sidecar hash over those bytes, and decode those same bytes —
validation, hash, and decode then all describe the same content with no window. Hash
algorithm should be pinned as SHA-256 (the mismatch warning is an integrity signal, not a
security control, but MD5/SHA-1 would invite collision-shaped confusion later).

**Verdict:** TECH_DEBT precondition; not a block.

### [MEDIUM] Last-colon `PATH:MODE` split silently reads the wrong file when a filename ends in `:both`, `:vl`, or `:ref`

**Location:** ADR-035 decision 1; `comfyless/generate.py:2097-2107`

Strict mode validation makes the unknown-suffix case safe (hard error), but inverts the
danger for the three valid suffixes: a file literally named `frame:ref` parses as path
`frame` + mode `ref` and silently opens a **different existing file** or errors
file-not-found — no fallback exists, unlike `_parse_lora_arg`'s float-parse fallback.
Attacker leverage is marginal (requires attacker-influenced filenames fed to the flag),
but the silent-wrong-file case is exactly the class decision 1 says it wants to avoid.
Two cheap amendments: document the always-safe escape (a path containing colons must
append an explicit `:MODE`, so `photo:ref:both` is unambiguous), and specify that when the
mode-stripped path does not exist but the full spec does exist as a file, the
error/warning must say so by name rather than reporting a bare not-found.

**Verdict:** TECH_DEBT precondition (documentation + error-message requirement in the spec).

### [MEDIUM] Warn-and-proceed on unsupported families corrupts unattended video chains and falsifies provenance

**Location:** ADR-035 decision 2 (routing table, "loud warning, then proceed without it")

The first consumer is ADR-033 chained plan execution — unattended, machine-driven. A
dropped reference there means keyframe N+1 is generated ignoring keyframe N, and the error
propagates through the rest of the chain with nobody watching the warning. The ADR itself
already articulates the right principle in decision 2a: warn-don't-block protects against
footguns a user can *stumble into* interactively; in machine/plan mode nobody stumbles —
the same reasoning that justified strict MODE errors justifies hard-fail (or at least
fail-the-segment) for a dropped ref in plan/`--json` mode, keeping warn-and-proceed for
the interactive CLI.

Related integrity requirement: the sidecar must record whether each ref was **applied or
dropped** — a sidecar listing refs that had no effect is a false provenance record, and
decision 7's replay semantics then replay a lie.

**Verdict:** TECH_DEBT precondition; the sidecar applied/dropped field should be added to
decision 7 before acceptance.

### [INFO] Content hash of private user files embedded in shareable output metadata

**Location:** ADR-035 decision 7

Sidecars (and PNG-embedded params) travel with generated images that get shared. An
unkeyed SHA-256 of the reference file lets anyone holding the sidecar confirm whether a
specific known image was the reference (content-confirmation fingerprint), and absolute
ref paths disclose filesystem layout — both consistent with what existing sidecars already
do for model paths, and acceptable for a solo desktop tool, but worth one sentence of
acknowledgment in the ADR so the MCP/upload ADR (where outputs are agent-visible) inherits
the awareness deliberately.

**Verdict:** no action required now; note it in the ADR's Deferred section for the MCP
follow-up.

### [INFO] Direct `--ref-image` at the CLI needs no containment check

**Location:** ADR-035 decision 6; `comfyless/refine.py:1489-1491`

No risk to add — a user typing a path at their own CLI is exercising user authority, same
as `--seed-image` or any file argument; a CLI-side containment gate would add no security
and break legitimate use. The cold-path `_check_paths` hole matters only for
**file-derived** ref paths (sidecar replay, `--params`, refine seed entry), which is the
decision-7 finding above, and for the daemon boundary, which is the decision-6 finding.
The ADR should state this split explicitly (typed-flag paths: no gate, loud echo only if
outside roots; file-derived paths: mandatory F4 treatment; daemon-crossing paths: dedicated
ref root allowlist) so the implementer doesn't apply one policy to all three.

**Verdict:** amendment recommended alongside the decision-6 and decision-7 findings.

---

## Bottom line

Four findings block moving to `accepted` as written — the `_check_paths` reuse (wrong
allowlist for the trust class, breaks the first consumer), the decision-3 invariant left
implicit while the routing table pressures against it, the unspecified enforcement locus
for the decode caps, and the absent replay-side trust treatment in decision 7. All four
are cheap textual amendments to the ADR, not redesigns — the overall shape (unified
surface, strict MODE, refine-derived caps, family-derived routing) is sound. The remaining
findings are preconditions to record in the ADR/spec or TECH_DEBT so the implementation
slice's negative tests cover them.
