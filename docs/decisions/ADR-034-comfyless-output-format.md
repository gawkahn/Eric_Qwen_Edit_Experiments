# ADR-034 — comfyless JPEG output format

Status:   accepted

## Context

comfyless writes PNG unconditionally. PNG is lossless and large; for
high-megapixel Qwen / Flux output the files are routinely tens of MB, which
is wasteful for casual browsing, sharing, and the refinement loop's
intermediate iterations. A `--output-format jpeg` option with a quality knob
is wanted.

The feature looks like a one-line flag. It is not, for three reasons the
survey turned up.

### 1. `.png` is hardcoded at ~10 independent literal sites

| Site | Role |
|------|------|
| `generate.py:471` | `_resolve_savepath()` auto-counter `{stem}{NNNN}.png` |
| `generate.py:2041` | argparse default `/tmp/comfyless.png` (sentinel-compared at `:3129`) |
| `generate.py:2149` | `--json` bridge output path |
| `server.py:818` | daemon **atomic** `O_CREAT\|O_EXCL` reservation `comfyless{NNNN}.png` |
| `mcp_server.py:2476` | MCP mirror of the above (non-atomic exists()-loop) |
| `cascade.py:467,489,526,528,573` | cascade numbering regex + `cascade_NNNN.png` |
| `refine.py:935` | judge canonical path `{stem}.png`, daemon output `shutil.move`d onto it |

`cascade.py:537` (`suffix = p.suffix or ".png"`) is the sole site that already
honors a caller-supplied extension.

### 2. Metadata carriage is uneven across paths

Metadata is embedded as a single PNG `tEXt` chunk keyed `"comfyless"`, built by
two near-duplicate helpers (`generate.py:477-503`, `cascade.py:587-608`), each
calling a bare `pil_image.save(path, pnginfo=...)` — no `format=`, no
`quality=`, no compression argument anywhere in the tree.

A JSON sidecar (`splitext(path)[0] + ".json"`) is *also* written, on four
paths: CLI in-process (`generate.py:3217`), CLI daemon-client
(`generate.py:2482`), the `--json` bridge (`generate.py:2195`), and cascade
(`cascade.py:950`). Because the sidecar uses `splitext`, it already lands
correctly next to a `.jpg` with no change.

Note the daemon: the daemon *process* writes no sidecar (`server.py:923`
returns metadata over the wire only), but the CLI **client** that talks to it
does (`generate.py:2482`). Daemon-backed runs therefore do produce `.png` /
`.json` pairs on disk. The distinction is internal and has no user-visible
consequence.

**The MCP server is the sole path that writes no sidecar** — there is no
`_write_sidecar` call in `mcp_server.py`; the tEXt chunk is its only metadata
record. JPEG has no tEXt chunk, so MCP-path JPEG output would carry no
provenance at all.

### 3. The output-path resolvers are gated Red Zone paths

`server.py:777-839` and `mcp_server.py:2439-2488` perform template de-rooting,
`_within` containment checks, null-byte gates, and — in the daemon's case — the
`O_EXCL` atomic reservation that closed Finding 1 of
`review-parallel-daemon-2026-07-03`. All three of `server.py`, `mcp_server.py`,
`refine.py` are on `scripts/git-policy/_red-zone-paths.sh`.

Any change to extension handling touches the reservation logic. A naive
"strip `.png`, append configured suffix" edit is exactly the shape of change
that reintroduces a collision or containment bug.

## Decision

Add `--output-format {png,jpeg}` (default `png`) and `--quality` (default
`0.7`) to comfyless, threaded through every output path.

**D1 — Centralize.** Introduce one resolution helper (proposed:
`comfyless/output_format.py`) exposing the configured extension and the PIL
`save()` kwargs. The ~10 hardcoded literals and both duplicated
`_save_with_metadata` helpers route through it. No call site composes an
extension by hand.

**D2 — Explicit flag wins; extension infers when the flag is absent.**
`--output foo.jpg` with no `--output-format` selects JPEG. With
`--output-format png` it is an **error**, not a silent rewrite — the daemon and
MCP paths generate filenames the caller does not control, so silent
extension-rewriting there would be a surprise. Mismatch is loud (§ user
preference: warn-don't-block does not apply; this is a contradiction, not a
footgun).

**D3 — `--quality` is a 0.0–1.0 float, mapped to PIL's 1–95 integer.**
`0.7` → `quality=70`. Values `>1.0` or `<=0.0` rejected. `--quality` is
ignored (with a warning) when format is `png`.

The fraction is the deliberate choice, and it is *not* an inconsistency with
`video.py:891`'s integer `--crf` — the two are different knobs that should not
be made to rhyme:

| | `--crf` (x264) | `--quality` (JPEG) |
|---|---|---|
| scale | 0–51 | 0.0–1.0 → 1–95 |
| direction | **lower = better** | **higher = better** |
| scope | temporal; bits allocated across frames by motion | per-image; DCT quantization matrix scale |
| default | 16 (near-visually-lossless) | 0.7 |

Both descend from block-based DCT quantization, which is why they feel like
one dial — but the **inverted direction is a live footgun**. Someone who
learns `--crf 16` and reaches for `--quality 16` would get near-garbage under
an integer scale. A 0.0–1.0 fraction is unambiguously "more is better" and is
therefore hard to confuse with CRF; that collision-avoidance is the primary
rationale, with familiarity from web APIs (`canvas.toBlob(cb, type, quality)`
takes the same 0–1 fraction) as secondary support. There is no principled
numeric conversion between the two scales and none should be implied.

Intended use is size management for still output. Note for the implementer:
quality is a *quality* target, not a size target — bytes vary with image
content, so there is no size guarantee at a given value, and PIL's useful
ceiling is ~95 (above that, size grows with little visible gain), which is why
1.0 maps to 95 rather than 100.

**D4 — Write the JSON sidecar on the MCP path.** This closes the one real
metadata gap (Context §2) and makes provenance format-independent. It is a
behavior change beyond the nominal scope of this ADR, justified as a
precondition rather than a drive-by: without it, MCP-path JPEG output is
unrecoverably lossy. No change needed for the daemon — its client already
writes one.

**D5 — The MCP sidecar carries FULL paths; it is NOT redacted.** This is a
deliberate divergence from the tEXt chunk, which stays redacted via
`redact_metadata_for_png` (`mcp_server.py:128-160`).

The reasoning: redaction enforces the opaque-handle principle *at the MCP
boundary* — the agent deals in catalog names, never absolute paths, in both
directions. The sidecar does not cross that boundary. It is a local file on
the operator's disk whose sole purpose is `--params` replay, and replay
requires real resolvable paths. A redacted sidecar would be a file that exists
only to be useless.

The two artifacts therefore have different audiences and correctly differ:

| Artifact | Audience | Paths |
|----------|----------|-------|
| tEXt chunk in the returned image | the agent / whoever the image is shared with — **travels** | redacted (N26-N29) |
| `.json` sidecar on disk | the operator, for replay — **stays local** | full |

**The load-bearing assumption is that the sidecar never becomes agent-readable.**
Today it holds: the MCP tool surface is `generate` / `list_models` /
`list_loras` / `list_transformers` — none return file contents, and the OWUI
tool has no filesystem read. If a future tool ever reads arbitrary paths, or
the output directory is exposed over a transport, this decision must be
revisited and the sidecar redacted or relocated. That contingency is the
security-auditor's primary question for slice 3, and belongs in TECH_DEBT.

**D6 — `--params` replay stays coherent.** `generate.py:174` dispatches on
`.png` → chunk, else → sidecar. A `.jpg` path currently falls into
`_load_sidecar` and JSON-parses JPEG bytes, producing a confusing decode error.
Add an explicit branch: image extensions other than `.png` raise a directed
error naming the `.json` sidecar. Negative test required.

**D7 — refine.py's canonical path follows the configured format.**
`refine.py:935` and the `shutil.move` at `:952` must use the resolved
extension. The judge's own `Image.open` is format-agnostic (PIL sniffs) and
needs no change; its in-memory PNG data-URI encode (`refine.py:381`) is a
transport detail and stays PNG.

## Alternatives Rejected

**EXIF / XMP metadata embedding in JPEG.** Would preserve single-file
provenance. Rejected: the sidecar already carries the full record on CLI paths,
D4 extends it to the rest, and EXIF `UserComment` round-tripping through PIL
adds an encoding surface (and a second redaction path to audit) for no gain
over a JSON file that is already the `--params` replay format.

**WebP / AVIF support.** Deferred. The format enum is designed to extend, but
each additional codec is its own quality-scale mapping and its own PIL
capability check. Not in this slice.

**Per-path format flags** (separate daemon/MCP/cascade settings). Rejected as
gratuitous surface area; one global default with per-invocation override is
sufficient.

**Extension-rewrite on mismatch instead of erroring (D2).** Rejected: on the
daemon and MCP paths the caller does not author the filename, so a rewrite is
invisible.

## Deferred / Out of Scope

- WebP, AVIF, lossless-JPEG variants.
- `COMFYLESS_SCHEMA` gaining an output-format key. Format is an *output*
  concern, not a generation parameter; it must NOT enter the replay params
  (it belongs alongside `output_path` / `savepath` in the non-schema set,
  `mcp_server.py:159-160`). Called out explicitly because adding it to the
  schema is the obvious-looking wrong move.
- Deduplicating the two `_save_with_metadata` helpers into one shared
  implementation. Tempting while here (§4 "never clean up while here") —
  separate slice.
- The non-atomic exists()-loop at `mcp_server.py:2476` where the daemon uses
  `O_EXCL`. Pre-existing inconsistency, noted for TECH_DEBT, not fixed here.

## Proposed slices

Sequenced so each is independently revertible. Slices 2, 3, 5 are Red Zone.

1. `output_format.py` helper + CLI flags + `generate.py` in-process path.
   PNG default proven byte-identical to today. **Non-Red-Zone.**
2. Daemon: wire field, `server.py` reservation extension. **Red Zone** —
   `security-auditor`, focus on the `O_EXCL` reservation and `_within`.
3. MCP: tool schema field, `_resolve_mcp_output_path`, sidecar + D5 redaction.
   **Red Zone** — `security-auditor`, focus on the new sidecar leak surface.
4. Cascade numbering + `_save_with_metadata`.
5. `refine.py` canonical path. **Red Zone.**
6. D6 `--params` negative branch + TECH_DEBT entries.

## Proof hooks

- PNG default byte-identical to pre-change output (regression guard).
- `--output-format jpeg` produces a decodable JPEG; sidecar present and
  replayable via `--params <stem>.json` on **all** paths including daemon+MCP.
- Negative: `--output foo.jpg --output-format png` errors.
- Negative: `--quality 1.5` and `--quality 0` rejected.
- Negative: `--params foo.jpg` gives the directed sidecar error, not a JSON
  decode traceback.
- MCP path: on-disk tEXt chunk stays redacted (N26-N29 unchanged) **while**
  the sidecar beside it carries full paths and round-trips through `--params`.
  Both halves asserted in the same test so the D5 divergence is pinned
  deliberately rather than drifting.
- Daemon concurrency: two daemons, same output dir, JPEG — no collision, no
  overwrite (extends the parallel-daemon reservation test).

## Open Questions

1. ~~D3's 0.0–1.0 scale vs `video.py --crf`'s integer.~~ **Resolved
   2026-07-20:** fraction confirmed. Not an inconsistency — different knobs,
   and the fraction actively guards against CRF's inverted scale. See D3.
2. Should `--output-format` be settable as a persistent default (config file /
   env var), or per-invocation only? Per-invocation assumed.
3. D4 writes a sidecar on the MCP path, which previously wrote none. Confirm
   that is wanted unconditionally rather than gated behind the JPEG case.
4. D5's threat model: confirm the MCP output directory is not, and is not
   planned to be, reachable by the agent or exposed over a transport. The
   full-path sidecar is safe only while that holds.

## Changelog

- 2026-07-20 — **Accepted and adopted by the ref-image session.** The
  concurrent work this ADR was blocked on is ADR-035 (reference-image surface,
  accepted same day); that session now owns both, resolving the overlapping-file
  contention by single ownership rather than cross-session coordination.
  Promoted **ahead of** ref-image implementation: the video keyframe-authoring
  loop (ADR-033) that ref-image exists to serve feeds keyframes into the video
  I2V chain, whose input-size limits require JPEG (the repo's own example
  keyframes are q92 4:4:4 JPEG because pre-commit blocks >500 KB PNG). An
  edit surface that emits only PNG wedges a manual convert step into that loop —
  so JPEG output is on the critical path, not adjacent to it. Slices 1–6
  unchanged. Open Questions Q3/Q4 remain **binding preconditions for slices 2–3**
  (daemon/MCP) and are not resolved by this adoption. Q2 (persistent default):
  per-invocation confirmed for now.
- 2026-07-20 — proposed. Not implemented; blocked pending concurrent
  session work on overlapping files.
- 2026-07-20 — corrected: an earlier draft claimed the daemon writes no
  sidecar. The daemon *client* does (`generate.py:2482`); daemon-backed runs
  produce `.png`/`.json` pairs. MCP is the only gap. D5 reversed from
  "redact the MCP sidecar" to "sidecar carries full paths" — redaction would
  defeat the sidecar's only purpose, and the sidecar does not cross the MCP
  boundary that redaction protects.
- 2026-07-20 — Q1 resolved: `--quality` stays a 0.0–1.0 fraction. D3 rewritten
  to record the CRF-vs-JPEG comparison and the inverted-scale rationale rather
  than treating the difference as debt.

AI-Disclosure: Claude (Opus 4.8) authored; Grant reviewed.
