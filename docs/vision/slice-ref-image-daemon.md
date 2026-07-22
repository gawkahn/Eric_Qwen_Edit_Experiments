# Vision — Reference-image daemon path (`comfyless/server.py` + wire + `ref_image.py` hardening)

**Date:** 2026-07-21 · **ADR:** ADR-035 (accepted) · **Risk:** L3 (§12 Red Zone — Unix-socket IPC)

> **Posture:** Boundary: the daemon Unix socket (`comfyless/server.py`) — a
> same-UID wire client can now list reference-image paths in a `generate`
> request, which the daemon `realpath`s, contains, opens, and VAE-encodes. This
> is a §12 IPC Red Zone surface and a new untrusted-file-read primitive behind
> the socket. `security-auditor` (Fable) reviews the diff before commit; ADR-035
> already carries two pre-code security reviews (`review-adr-035-*` +
> `review-adr-035-rereview-2026-07-20.md`) whose findings 6a–6g / decision 3 /
> decision 2-Finding-4 are the design authority implemented here. Sidecar
> **replay trust** (decision 7 cold-path refusal, F4 echo, `--params`) stays
> **out of scope** — slice 5.

## Intent

Let the persistent daemon (ADR-020) honor `--ref-image`, so a qwen-edit run that
today is forced in-process (`_should_delegate_to_server` skips delegation on any
`--ref-image`, slice 3) can reuse a resident 20B pipeline instead of paying the
30–90 s reload. The first consumer is ADR-033 keyframe authoring, which the ADR
names as relying on the daemon.

The daemon becomes a second decode site for untrusted image *content* and the
**first** site where a wire client — not just a CLI arg — supplies reference
paths. That is the entire security weight of this slice: containment against a
dedicated root set, wire-shape strictness, and a decode helper that cannot be
made to block the accept loop.

## Scope of THIS slice (ADR-035 implementation slice 4)

- **A. `ref_image.py` non-regular-file guard** — `load_ref_image_capped` opens
  with `O_NONBLOCK` and rejects anything that is not `S_ISREG` before reading, so
  a FIFO / device / directory path cannot hang `open()` forever and wedge the
  daemon (closes slice-2 LOW-2 / TECH_DEBT precondition 1). Benefits the
  foreground path too; the shared decode site stays single.
- **B. Wire-shape strictness (canonical validator)** — `validate_ref_image_entry`
  in `params_validation.py`, called from `validate_machine_request`, mirroring
  `validate_lora_entry`: each `ref_images` entry is a dict with `path: str` and
  `mode: str ∈ {both,vl,ref}`; the list is capped at 8 (decision 6f re-checked on
  the wire); a bad `mode` from a non-CLI caller is rejected here, not by a
  `KeyError` on `_REF_MODE_FLAGS` inside `generate()` (TECH_DEBT precondition 2).
- **C. NUL defense (`server._validate_request`)** — `ref_images[i].path` joins the
  existing scalar/`loras` NUL pre-check (6e), so a NUL never reaches
  `os.path.realpath` and kills the accept loop.
- **D. `ref_image_roots` containment (`server.py`)** — a **separate** allowlist
  from `_check_paths`'s weight roots (6a). `ref_image_roots` = the daemon's
  `--output-dir` ∪ `--ref-root` spawn additions. A new `_check_ref_paths(req,
  ref_roots)` validates every `ref_images[i].path` via the shared `_within`
  union; the two root sets are **disjoint in purpose and never merged**.
- **E. Daemon-side decode + run (6b)** — `_handle_generate` threads `ref_images`
  and `ref_dims_explicit` into its `generate()` call. Decode/caps run in the
  daemon process because `generate()` → `_run_qwen_edit_refs` →
  `load_ref_image_capped` executes there. Containment (D) runs at the wire
  boundary *before* generate; the decode helper deliberately does not re-impose
  it (6b).
- **F. Drop-strictness across the wire, fail-closed (decision 2, Finding 4)** — a
  reference handed to a family with no edit path is a loud warn-and-proceed for
  the interactive CLI but a **hard failure** for machine clients. Strictness
  travels as a wire field whose **absent value = strict**. Only the interactive
  CLI on a TTY sets the lenient value; every other client (plan workers, scripts,
  the video worker) inherits strict by omission.
- **G. Cache-key pin (decision 3)** — a `test_server_robustness` case asserting
  `_request_cache_key` is byte-identical with and without `ref_images`, in the
  style of the NAG pins. Ref presence must never swap pipeline class or trigger a
  `from_pipe` upgrade; the key already ignores `ref_images` — the test pins it.
- **H. Delegation seam (decision 7, Finding 2 — revised in slice 4b).**
  `_should_delegate_to_server` loses its `and not args.ref_image` blanket skip. A
  `--ref-image` run delegates on the **same rule as any run** (savepath /
  default-output, skip explicit `--output`); the **daemon is the authoritative
  containment gate**. A reference outside the daemon's `ref_image_roots`
  (`--output-dir` ∪ `--ref-root`) comes back as a distinct **`RefPathError`**, on
  which `_delegate_to_server` **falls back to in-process** (row-1 user authority)
  with a loud warning. Keyed on the error *type*, never a message substring, so a
  model-path `PathError` still hard-fails. **No client-side ref-root gate** — the
  CLI cannot know the daemon's `--output-dir`, and a client `--ref-root` would
  only work when it matched the daemon's config (removed as a client flag; it
  stays a `--serve` spawn flag). *(The original slice-4 text — delegate iff every
  typed ref is inside a CLI `--ref-root` — is superseded; it made keyframes in the
  output tree run in-process and required a redundant client flag.)*
- **I. `--ref-root DIR` — a `--serve` spawn flag only** (repeatable; revised in
  4b). Adds to the daemon's `ref_image_roots` (= `--output-dir` ∪ these). **Not a
  client flag** — the daemon is the authoritative gate and a client `--ref-root`
  would only work when it matched the daemon's config (see §H). A `--ref-root` of
  `/`, a mount root, or `$HOME` prints a loud breadth warning and proceeds
  (warn-don't-block — operator-typed, not attacker-reachable; re-review Finding
  6). Each configured ref root is logged at startup.

## Decisions inherited from ADR-035 (settled, not re-opened here)

1. **Image-read roots ≠ weight-load roots** (6a). Reusing `_check_paths`'s
   `{model_base} ∪ lora ∪ transformer` set would reject every legitimate keyframe
   (they live in output/working dirs) *and* adding `--output-dir` to the weight
   roots reopens the 2026-06-01 refiner-path CRITICAL. Disjoint allowlists.
2. **Caps enforce in the process that decodes** (6b). Paths cross the socket, so
   the daemon decodes; a same-UID wire client bypasses the CLI arg layer entirely,
   which is why the caps live at the decode site, not in argparse.
3. **Pipeline class is selected once** (decision 3 invariant). Ref presence never
   triggers a class swap or `AutoPipelineForImage2Image.from_pipe`; the
   un-discriminated cache key stays sufficient because — and only while — that
   holds. Pinned by G.
4. **Trust class is the boundary a path arrives through, never a wire field**
   (decision 7). No `typed_by_user` flag on the wire — a same-UID client could
   forge it. Row-3 containment (D) governs every path that crosses the socket,
   full stop.

## Invariants (must always be true)

- **A malformed `ref_images` wire entry is rejected at the boundary, never
  crashes the accept loop.** Not-a-dict, missing/`non-str` `path`, missing/bad
  `mode`, > 8 entries, and a NUL in a path each return a structured
  `ValidationError` and the daemon keeps serving. No `isinstance`
  predicate is added to `server._validate_request` (N19 invariant) — shape
  validity is the canonical validator's job (B); `server` owns only NUL (C) and
  containment (D).
- **`mode ∉ {both,vl,ref}` from any caller is refused before `generate()`** — the
  `_REF_MODE_FLAGS` lookup can never `KeyError`.
- **Every wire-supplied reference path is `_within` `ref_image_roots`** (D), which
  is disjoint from and never merged with the weight-root allowlist. A path outside
  is a `RefPathError` (distinct wire type, 4b), logged server-side (prompt
  redacted) before the reply, on which the client falls back to in-process.
- **A FIFO / device / directory / symlink-to-non-regular passed as a reference is
  rejected before any blocking read** (A) — `open()` never hangs the daemon.
- **The daemon decodes through `comfyless/ref_image.py` only** — same single
  decode site as foreground; byte cap, pixel cap, format allowlist, first-frame,
  SHA-256, and the no-file-bytes-in-errors rule all still hold, now in the
  VRAM-holding process.
- **`_request_cache_key` is identical with and without `ref_images`** (G) — ref
  presence changes output content, not pipeline shape; the daemon serves the same
  cached qwen-edit pipeline across ref and non-ref requests for the same model.
- **A non-qwen-edit family + refs hard-fails a machine request and warn-proceeds
  an interactive one** (F) — the wire strictness field defaults to strict on
  absence; the video worker and scripts get fail-the-segment, not a silent
  keyframe drift.
- **A `--ref-image` run delegates on the same rule as any run** (slice 4b); a
  reference the daemon refuses (`RefPathError`) falls back to in-process with a
  loud warning, never a silent drop and never a hard failure. A model-path
  `PathError` still hard-fails (fallback is keyed on the distinct error type).
- **A non-ref `generate` request — wire or in-process — is byte-for-byte
  unchanged.** Existing `test_server_robustness` / `test_mcp_server` /
  `test_params_schema` stay green.
- `local_files_only=True`; the daemon never fetches from the network.

## Failure semantics

Fail-closed at every new boundary: wire-shape violation → `ValidationError`;
ref-containment violation → `RefPathError` (server-logged, prompt redacted;
client falls back to in-process); model-path violation → `PathError`;
non-regular-file / cap / format / decode violation → `RefImageError` surfaced as
an `InferenceError` naming path + reason, never file bytes; family-mismatch drop
→ hard `InferenceError` for machine clients (strict default), loud warn-and-
proceed for the interactive CLI (lenient). Every failure returns a structured
reply and leaves the daemon serving; a failed run unlinks its reserved output
placeholders (existing path). No partial writes.

## Out of scope

Sidecar **replay trust** (decision 7: cold-path outside-roots refusal, F4 echo,
`ref_images` removal from `_SKIP_SIDECAR_KEYS`, moved-file / hash-mismatch
warnings) — slice 5. MCP `edit` surface + tool-list bump — its own ADR.
`flux2klein` / other-family img2img execution. The video worker's ref-request
construction (a later ADR-033 slice; it will build wire requests directly and
inherit F's strict default by omission). `resolve_hf_path` §12 review (standing
debt). No new dependency.

## Negative-case tests (minimum)

- `ref_images` entry not a dict / missing `path` / `path` not str → `ValidationError`, field named, daemon survives.
- `ref_images` entry `mode` = `"evil"` / missing `mode` → `ValidationError`, no `KeyError`.
- 9 `ref_images` entries on the wire → rejected, cap named.
- NUL in `ref_images[i].path` → rejected (mirrors the `loras[i].path` NUL test), accept loop survives.
- Ref path outside `ref_image_roots` → `RefPathError`; a path inside `--output-dir` and one inside a `--ref-root` both pass; a weight-root-only path is refused (roots disjoint). Client turns `RefPathError` into an in-process fallback; a model-path `PathError` stays a hard error.
- FIFO path (`os.mkfifo`) and directory path as a reference → `RefImageError`, no hang.
- `_request_cache_key(req)` == `_request_cache_key(req + ref_images)` (decision 3 pin).
- Drop strictness: non-qwen-edit family + refs with strict (absent field) → hard error; with lenient field → warn-proceed. (Predicate-level; full family routing is a live-GPU concern.)
- Delegation seam: `--ref-image` inside `--ref-root` → delegate True; outside all roots → False; no `--ref-root` → False (in-process, slice-3 parity).
- `--ref-root /` → breadth warning emitted; startup still proceeds.

## Proof hooks

- `./.venv/bin/python3 test_server_robustness.py` — wire shape, NUL, containment, count cap, cache-key pin, drop strictness (new cases).
- `./.venv/bin/python3 test_ref_image.py` — non-regular-file guard (new cases) + existing ingestion cases green.
- `./.venv/bin/python3 test_ref_edit.py` — updated delegation seam (roots-aware).
- `./.venv/bin/python3 test_params_schema.py` — `ref_images` entry validation via the canonical validator.
- `just tests` — full battery, 0 failures.
- `python -m py_compile comfyless/server.py comfyless/generate.py comfyless/ref_image.py comfyless/params_validation.py`
- Live smoke (**Grant**): start a daemon with `--ref-root <keyframe dir>`; a
  `--ref-image kf.png:both --ref-root <dir>` run delegates and reuses the resident
  pipeline; a typed ref outside the root runs in-process.

## Edit scope (hard)

`comfyless/server.py`, `comfyless/generate.py`, `comfyless/params_validation.py`,
`comfyless/ref_image.py`, `test_server_robustness.py`, `test_ref_image.py`,
`test_ref_edit.py`, `test_params_schema.py`, this Vision doc, `ADR-035` Changelog,
`TECH_DEBT.md` (close the two slice-4 preconditions). Anything else → STOP and
split.

## AI-Disclosure

Claude (Opus 4.8) authored; Grant reviewed.
