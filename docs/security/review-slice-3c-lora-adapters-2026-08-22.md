AI-Disclosure: Claude (Fable 5, `claude-fable-5`) authored this review via the `security-auditor` agent (no model override passed; frontmatter pin honoured); Grant reviewed.

# Security review — ADR-045 slice 3c: `comfyless/core/lora_adapters.py` as the daemon's LoRA subsystem

Spec: `docs/decisions/ADR-046-comfyless-owned-lora-adapters.md`
Trigger: `comfyless/server.py` is a Red Zone path (Unix-socket IPC daemon, ADR-001). The change to
it is two function-local import lines (`:669`, `:830`), but every LoRA a daemon client requests
now executes `comfyless/core/lora_adapters.py` — a rewrite of `nodes/eric_qwen_edit_lora.py` —
instead of the node-pack original. Reviewed as a line-by-line differential against the original
on the axes that matter at the socket boundary.

## Summary

Threat model: a client inside the socket trust boundary supplies a LoRA *path* (already validated
by `_check_paths` against allowlisted roots) whose *file content* is untrusted — arbitrary key
strings and tensor shapes flow into key normalisation, PEFT config construction, and direct weight
writes on the daemon's long-lived cached pipeline. The properties that matter are the ADR-019 DMR
invariants (single write path, `.weight`-only resolution map, all-or-nothing entry gate, LIFO
ledger, restorable backups), the `weights_only=True` deserialisation posture, and parity of
exception handling so a failed load neither reports success nor escapes as a daemon crash.

**Posture: the rewrite preserves every ADR-019 requirement it inherits and does not widen or
narrow any exception boundary; no new CRITICAL or HIGH risk.** The one MEDIUM is an *absence* —
the commit-policy Red Zone path list no longer gates the file where the DMR dispatcher actually
lives, and this slice adds a second un-gated file that performs all daemon weight writes.

What was checked: every `try/except` in both modules (catch sets, placement relative to the
direct-merge call); every write to model state (`_merge_direct` has exactly one,
`apply_merge_delta`; `unload_adapters` is verbatim); the order of `merge_resolution_map` →
`refuse_unmergeable_base` → first `apply_merge_delta`; the resolution of `target` exclusively
through `model_sd`; `load_state_dict` (`torch.load(..., weights_only=True)` preserved at
`lora_adapters.py:119`); `adapter_name` provenance on the daemon path (`server.py:847`
`sanitize_adapter_name` → `[a-zA-Z0-9_-]`, unchanged); `.item()` call sites and their guards;
`_discover_prefix` cost model; module-level import side effects (none: `os`, `typing`, `torch`
only; all heavy imports are function-local). Not run by the auditor: the test suites (read-only
tools) — the 256-assertion green result is the author's.

## Coverage

Reviewed: `comfyless/core/lora_adapters.py` (whole); `nodes/eric_qwen_edit_lora.py` (whole, as the
differential baseline); `comfyless/server.py:335-344, 660-680, 820-885` (sanitizer, both import
sites, LoRA add-loop); `comfyless/core/eric_diffusion_fp8_ops.py:1725-1790, 2031-2103`
(resolution map, entry gate, dispatcher head, restore, ledger); `comfyless/generate.py:66, 1282,
1333`; ADR-046 (whole); `test_lora_adapters.py` source guards G1–G8; `test_quant.py:397-408`;
`test_server_robustness.py:1315`; `scripts/git-policy/_red-zone-paths.sh:12-34`; TECH_DEBT entries
2026-07-03 (DMR partial merge), 2026-07-16 (function-scoped gating), 2026-08-22 (slice 3c);
`comfyless/core/__init__.py`.

Not reviewed: `apply_merge_delta` body below line 1830, `_merge_into_scaled_fp8`,
`_merge_into_torchao` (unchanged; covered by the DMR/NV review chain);
`eric_lora_format_convert_apply.py` and `check_lora` / `decode_kohya_to_bfl` (not in the slice;
called identically by both implementations).

## Findings

### [MEDIUM] Red Zone path gate no longer covers the code that performs the daemon's weight writes

Location: `scripts/git-policy/_red-zone-paths.sh:23,34` gates `nodes/eric_diffusion_fp8_ops.py`,
which no longer exists (moved in slice 1b); `comfyless/core/eric_diffusion_fp8_ops.py` and
`comfyless/core/lora_adapters.py` are un-gated.

Risk: the T1 commit-policy trigger for `security-auditor` + ADR reference on the scaled-fp8
parser / DMR dispatcher is now a no-op, and this slice adds a second file through which every
daemon LoRA write, backup and registry mutation flows, also outside the gate. A future edit to
either file can land with no review trigger other than T5 discipline. Not introduced by the
two-line `server.py` change, but slice 3c is the commit that makes `lora_adapters.py`
daemon-executed, and the CLAUDE.md review-bar table still names the stale `nodes/` path.

Remediation: separate policy slice (outside this slice's declared scope): replace the
`nodes/eric_diffusion_fp8_ops\.py` pattern with `comfyless/core/eric_diffusion_fp8_ops\.py`, add
`comfyless/core/lora_adapters\.py`, in both the regex function and `list_red_zone_paths`; update
the CLAUDE.md table; `just policy-test`. **Disposition: TECH_DEBT entry 2026-08-22 "the Red Zone
path gate names a file that no longer exists" added in this slice; trigger = next policy slice,
no later than slice 6.**

### [INFO] `test_quant.py` DMR source guard now asserts on code the daemon does not run

`test_quant.py:397-408` reads `nodes/eric_qwen_edit_lora.py`; it still passes but no longer
protects the daemon path. No current gap — `test_lora_adapters.py` G1–G4 duplicate the same checks
against `_merge_direct`, including the `refuse_unmergeable_base` < `apply_merge_delta` ordering.
Fold into slice 7's suite-ownership pass (ADR-046 §Deferred). **Disposition: ADR-046 amended to
say "duplicated", not "re-pointed".**

### [INFO] Mid-loop raise partial-merge state — carried unchanged, not widened

`lora_adapters.py:411` (`alpha.item()` on a multi-element `.alpha` → `RuntimeError`) and `:432`
(`apply_merge_delta` req-21 non-finite raise) exist at the same positions in the original. A raise
on the Nth module leaves 1..N-1 merged with a populated backup attr but no `peft_config` entry /
ledger record; on the daemon the partially-merged pipeline stays cached (`server.py:868` warns,
does not evict). This is the open TECH_DEBT entry 2026-07-03 (amended 2026-07-16), adversarial-only
by its framing. The rewrite neither fixes nor worsens it. **Disposition: none in this slice
(equivalence is the contract); existing trigger stands.**

### [INFO] `adapter_name` on the daemon path is sanitised before reaching `setattr` — unchanged

Backup attribute names are `f"_{kind}_backup_{adapter_name}"`, identical to the original. On the
daemon, `adapter_name = sanitize_adapter_name(Path(lora_path).stem)` restricts to `[a-zA-Z0-9_-]`;
the `_<kind>_backup_` prefix prevents collision with any real transformer attribute.
`generate.py:1282` (CLI) uses only `.replace(" ", "_").replace(".", "_")` — pre-existing, out of
scope, same prefix protection applies. **Disposition: none.**

### [INFO] `_discover_prefix` cost is bounded; sorted order is not a new attack surface

Reached only after TE filtering, the direct-intersection test and all four known prefixes miss.
Cost ≈ 20·depth·|P| + 20·|M| — linear in the hostile file's key count, within a small constant of
the original's 20·50·|P|. The sorted order lets an adversary choose which 20 paths are inspected,
but a discovered prefix is only applied if >30% of P map into real module names under it, and it
is used solely for key renaming — no code or attribute lookups derive from it. No DoS beyond "a
50k-key file costs 50k-key work", which `load_state_dict` already incurred. **Disposition: none.**

### [INFO] `server.py` import change: no boundary behaviour change; one hidden dependency removed

`lora_adapters.py` has no import-time side effects; `comfyless/core/__init__.py` is docstring-only.
The old import depended on `comfyless/__init__._install_shims()` having stubbed `folder_paths`; the
new one does not — the daemon no longer has an implicit dependency on a ComfyUI stub to load its
LoRA code. Both imports remain function-local in the same positions, so import-time failure modes
are unchanged. **Disposition: none.**

## ADR-019 requirements verified as preserved (differential)

- **req 21** (non-finite delta gate): every delta reaches `apply_merge_delta` (`:432`), which
  performs the `isfinite` check; no path bypasses it.
- **req 23 / DMR-3 / DMR-8** (`.weight`-only buffer filter; param-then-buffer precedence):
  `_merge_direct` resolves targets only via `merge_resolution_map(transformer)`; `_resolve_target`
  checks `path + ".weight"` then `path` against that map and nothing else. An adversarial
  `foo.weight_scale.diff` cuts at `.diff` → `foo.weight_scale` → neither candidate is in a map that
  excludes non-`.weight` buffers. `unload_adapters` uses the same map (verbatim).
- **req 24** (single write path): `_merge_direct` contains no `.data` access; the only write is
  `apply_merge_delta(...)`. The `param.data.copy_` at `:884` is the verbatim restore path inside
  `unload_adapters`, already covered by the DMR review. G2 negative guard re-pointed.
- **req 25** (LIFO ledger): `record_direct_merge` iff `applied > 0`; `warn_non_lifo_unload` runs on
  every direct-merge unload regardless of backup presence (verbatim).
- **req 29** (dequant-cache invalidation on restore): inside `restore_merge_backup`, untouched.
- **req 65** (all-or-nothing entry gate): `refuse_unmergeable_base` at `:398`, before
  `_group_by_module` and the loop; first `apply_merge_delta` at `:432`; called once per direct-merge
  attempt exactly as each original `_load_*_direct` did. Propagation path identical
  (→ `load_lora_with_key_fix` → `server.py:868 except Exception` → warning, pipeline untouched).
  G4 ordering guard present.
- **Backups consumable by `unload_adapters`**: `_<kind>_backup_<name>` matches what
  `unload_adapters` derives from `peft_config[name]["_type"] = "<kind>_direct"`; `kind.tag` ∈
  {`lora`, `lokr`, `loha`}, the original's three literals.
- **`weights_only=True`** for non-safetensors files: `:119`.

## Exception-boundary parity

Every `except` clause in the new module has a counterpart with the same catch set and position
relative to the direct-merge call. `_load_lokr_adapter` / `_load_loha_adapter`:
`(ValueError, RuntimeError)` around PEFT injection only; flatten rescue `except Exception` — as
original. `_load_lora_adapter`: `(ValueError, RuntimeError)` per pipeline attempt and around PEFT
injection; inner `except Exception` on `get_list_adapters` verification → `registered = True` (same
fail-open-on-verification as the original); `_merge_direct` outside any handler.
`load_lora_with_key_fix`: `except Exception` on pre-check and conversion path (verbatim);
`(ValueError, RuntimeError, KeyError)` on the fast path with `_is_fixable_load_error` reproducing
the original predicate term-for-term including which substrings are case-insensitive. No
`BaseException` / bare `except` anywhere in either module. No path returns `True` without either
PEFT success or `applied > 0`. Nothing the original let escape is now swallowed, and nothing the
original caught now escapes.

## Companion code review (`code-reviewer`, Fable 5, same day)

Numerics and routing confirmed operation-for-operation. Three promise-drift findings, all folded
before commit: (1) `_discover_prefix` scans all module names where the original capped at 50 —
ADR-046 #5 amended to state the superset explicitly and `test_lora_adapters.py` B6.4 pins the
expected divergence on a >50-module model; (2) the multi-candidate discovery case the ADR promised
was missing — B6.3 added; the LoKR "PEFT rejects → except branch" orchestrator case was not pinned
— B3.7b added (and revealed, pre-existing and preserved, that a rejected PEFT LoKR injection leaves
the module wrapped, so the direct merge finds no `.weight` and the flatten rescue is what runs);
(3) the DMR guard was duplicated rather than re-pointed — ADR wording corrected. B3.6 relabelled
(pipeline `RuntimeError` is swallowed by both orchestrators, pre-existing). Two stale comments
noted for a follow-up doc slice: `comfyless/core/eric_diffusion_lora_check.py:22` and
`comfyless/core/eric_lora_format_convert_apply.py:8` still name `eric_qwen_edit_lora`.
