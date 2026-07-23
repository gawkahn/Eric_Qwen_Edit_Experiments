# Security Review — ADR-035 Slice 5 (reference-image replay trust)

**AI-Disclosure:** authored by the `security-auditor` subagent (spawned pinned
to `claude-fable-5`) and folded by Claude (Fable) in the main session; Grant
reviewed. **Model-fallback note (recorded 2026-07-23):** the auditor did NOT
run entirely on Fable. Its transcript
(`tasks/ad1f31f72a8c30166.output`) contains a literal harness fallback record —
`{"type":"fallback","from":{"model":"claude-fable-5"},"to":{"model":"claude-opus-4-8"}}`
(requestId `req_011CdJ6nFWxbd2QsVGccPpqP`) — that fired ONCE at the transition
from the evidence-gathering phase to the first analysis turn (immediately after
grepping the `_within` containment helper). Fable performed only the read-in
(diff, ADR decision-7 text, ref validators, `_within`); **every finding below,
including CRITICAL-1, was authored by `claude-opus-4-8` post-fallback.** No
`stop_reason`/refusal is recorded, so the cause is not proven from the
transcript, but the most plausible reading is Fable's dual-use safety layer
declining the offensive-security synthesis turn and the harness falling back to
Opus 4.8. This matters for the §5A review bar: the model that actually authored
the security truth was Opus 4.8, not the pinned Fable (per this environment,
Fable is Mythos-tier, above Opus 4.8 — i.e. a silent downgrade at the
truth-authoring turns). The companion `code-reviewer` run
(`tasks/a3290aff851ebca45.output`) ran fully on Fable, no fallback.

**Date:** 2026-07-22
**Slice:** ADR-035 slice 5 — replay trust for `ref_images` (decision 7, row 2).
**Surfaces:** `comfyless/generate.py` (`_gate_file_derived_refs`,
`_replay_ref_roots`, `_apply_replay_ref_trust`, `_SKIP_SIDECAR_KEYS`,
`_extract_eric_save_params`), `comfyless/ref_image.py` (`hash_ref_file`,
`_read_ref_bytes_capped`), `comfyless/refine.py` (`build_config_from_seed`).
Red Zone trigger: `refine.py` is a `_red-zone-paths.sh`-gated path; the whole
slice is decision-7 security content (attacker-craftable metadata directing
file reads into shared image conditioning).

## Threat model

A `--params` sidecar / comfyless PNG `tEXt` chunk is attacker-craftable
metadata. Without treatment a crafted sidecar directs comfyless to read any
user-readable file and VAE-encode its bytes into the conditioning of an image
the victim may later share. The recorded content hash is no defense (the
attacker writes it to match). The cold in-process replay path has no
`_check_paths` gate (decision 7), so containment must be enforced by the gate
itself.

## Contract verified

| # | Contract | Verdict |
|---|----------|---------|
| C1 | No file I/O on a ref path that fails containment | MET (no-read-on-refusal test pins it) |
| C2 | Outside-roots file-derived path never reaches generate/wire/decode | MET (hard `return 2` before I/O), after CRITICAL-1 fix |
| C3 | `p["ref_images"]` never reaches generate/wire from params on any path | MET (unconditional pop; Eric-Save + `--json` + MCP + refine all drop) |
| C4 | Trust class decided by SOURCE, never a forgeable field; typed replaces file-derived | MET |
| C5 | `PATH:MODE` re-injection cannot smuggle a path past the parser | MET (last-colon split + always-valid MODE) |
| C6 | `hash_ref_file` no weaker than `load_ref_image_capped`; no byte echo | MET (shared `_read_ref_bytes_capped`) |
| C7 | Refine seed refs never execute/forward; malformed entries don't crash echo | MET |
| C8 | Root composition does not silently widen | MET **after** CRITICAL-1/LOW-1 fixes |

## Findings (all folded in-slice before commit)

### CRITICAL-1 — replay roots derived from attacker-controlled sidecar weight paths
`_replay_ref_roots` originally added `dirname(realpath(val))` for each
`model`/`*_path` read from the merged params dict `p` — the same untrusted
sidecar carrying `ref_images`. A crafted sidecar could set an inert weight key
(e.g. `upscale_vae_path`) to a file inside the target directory, making that
directory a trusted root, then point `ref_images` at a secret there; the ref
passed `_within`, the SHA mismatch was only a warning, and the file was
VAE-encoded — the exact primitive decision 7 exists to close, working
end-to-end. The daemon path did not save it (weight-beside ref refused at the
socket → ungated in-process fallback ran it).
**Fix:** `_replay_ref_roots` now reads roots ONLY from operator sources —
explicit `--output` dir, `--ref-root`, and CLI-**typed** weight args (`args.*`,
row-1 authority) — never from `p`. Negative test added:
`seam: sidecar weight path CANNOT authorize a co-located ref`.

### MEDIUM-1 — terminal escape injection via raw echo of attacker paths
The gate echoed `abs_path` raw to stderr for every entry, including refused
(fully attacker-controlled) paths; only NUL was rejected, so ESC/CSI/OSC
sequences reached the operator's terminal at the "verify each" prompt.
**Fix:** all path echoes use `repr()` (generate.py gate + refine seed echo).
Test: `gate: echo escapes control chars (no raw ESC on stderr)`.

### LOW-1 — default `--output` sentinel made `/tmp` a replay root
On a replay without explicit `--output`, `/tmp` (world-writable, so a co-tenant
can plant referenced bytes) became a trusted root.
**Fix:** the `/tmp/comfyless.png` sentinel is excluded from `_replay_ref_roots`.
Test: `roots: default /tmp output sentinel is NOT a root (LOW-1)`.

## Residual risk (accepted / deferred)

- **TOCTOU (LOW):** containment resolves symlinks via `realpath` at check time;
  `hash_ref_file` / decode `open()` the unresolved path. A concurrent
  same-UID symlink swap could redirect the read. Bounded to a same-UID attacker
  racing a single-writer desktop tool — noted, not fixed. (Consistent with the
  slice-2 LOW-2 disposition.)
- **Path-existence oracle (INFO):** a refused path's `realpath` lstats its
  components (metadata only, no content read) — acceptable under the same-UID
  model.
- **`--json` / MCP forward-looking (INFO → TECH_DEBT):** those transports drop
  `ref_images` today by never forwarding it; there is no gate there. Whoever
  later wires `ref_images` into `--json`/MCP inherits the row-2 obligation.
