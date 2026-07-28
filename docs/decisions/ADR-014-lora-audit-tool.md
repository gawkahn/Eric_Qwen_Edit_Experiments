# ADR-014: LoRA audit tool — `scripts/lora_audit.py`

**Date:** 2026-05-17
**Status:** accepted (2026-05-23, after `security-auditor` round-3 returned `CLEAN`; rounds 1 + 2 returned `CHANGES REQUIRED` and were folded — see Changelog and `docs/security/review-lora-audit-2026-05-17.md`)
**Risk:** L3 (file writes from caller-supplied paths AND deletions; per global §12 file-write trigger)
**Risk-trigger note (per security-auditor F-10):** L3 today — file writes from caller-supplied paths, no §5 Red Zone surface. **Re-classifies to Red Zone (§5-adjacent) the day any LLM-agent or remote caller can supply `--audit-root`, `--base`, or `--output-dir` values.** Any slice that wires those flags from LLM output or network input MUST write a new ADR + `security-auditor` review at Red Zone discipline (allowlist instead of audit-root containment, actor identity, etc.); ADR-014 does **not** authorize that transition. Mirrored to Backlog as a standing check.
**Related:**
- Vision slice: `docs/vision/slice-lora-audit.md` (approved 2026-05-17 by Grant).
- ADR-013 (comfyless dep divergence) — establishes the `./.venv/bin/python3` test runner this ADR's proof hooks invoke.
- ADR-012 (machine-boundary validator) — precedent for path-traversal and machine-caller posture in this codebase.
- ADR-001 (daemon socket security) — precedent for fail-closed startup discipline in tools that touch the filesystem from caller-supplied input.

**AI-Disclosure:** Claude (Opus 4.7, 1M context) authored; Grant reviewed.

---

## Context

The LoRA catalog project (separate repo, **Backlog → Queued**) needs an "only usable LoRAs" view of Grant's local LoRA tree. Several thousand `.safetensors` LoRAs accumulated over time from multiple training toolchains (kohya_ss, ai-toolkit, civitai exports, ComfyUI captures), some orphaned post-civitai-red split (`project_civitai_orphaned_files`), some in formats that this node pack can convert into a usable shape (BFL→diffusers via `eric_lora_format_convert_apply.py`), some that can never load (wrong architecture, truncated, zero-byte). Catalog cannot reasonably classify these in-band — that requires loading transformers, walking state-dict keys, running PEFT shape checks. The classifier already exists inside this repo's LoRA loader; the catalog needs a CLI front door to it that emits a machine-readable manifest.

This ADR specifies that CLI: **`scripts/lora_audit.py`** (MVP — adapters only). The Vision slice (`docs/vision/slice-lora-audit.md`) is the *why* and the invariants; this ADR is the *how* and the trade-off decisions. Implementation follows in slices via `/change-slice`.

**Why now:**
- The classifier infrastructure is in place (`check_lora()`, `find_matching_plan()`, the `LoRACheckResult` dataclass with `verdict ∈ {OK, NORM_TARGETING, DIM_MISMATCH, POOR_MATCH, WRONG_ARCH}`) and is already the same compatibility check the loader uses as its pre-flight gate. The audit tool reuses it, doesn't fork it.
- The LoRA catalog cannot start work until it knows the manifest shape it consumes; the contract has to land first.
- The parallel MCP slice 1 and Hunyuan-Image worktrees do not touch `scripts/`, `nodes/eric_diffusion_lora_check.py`, or `nodes/eric_lora_format_convert*.py`, so this slice can land in isolation on the `lora-convert-scripts` branch in the `eric-lora-convert` worktree.
- ADR-013 just established the per-worktree `.venv` story; this slice's proof hooks consume that substrate.

**§12 trigger surface (the reason this is L3):**
- The tool reads safetensors from caller-supplied directory trees (`--audit-root`).
- With `--convert` it writes new safetensors based on caller-supplied paths.
- With `--delete` it removes files based on caller-supplied paths (constrained to those it classified `deletable`).
- `.pt` / `.bin` / `.pth` LoRA files are pickle-bearing; deserialization safety is non-negotiable (`weights_only=True`).

All three trigger §12; none trip §5 (no auth, PII, billing, audit-truth). The Red-Zone-grade ADR + security-auditor discipline applies because of §12, per project CLAUDE.md Review-Bar table.

---

## Decision

### 1. File location and module layout

- New file: `scripts/lora_audit.py` at the worktree root (new directory; matches Grant's stated preference 2026-05-16: "scripts/ - in fact, I think the analyze_checkpoint.py and dequantize_nf4.py scripts belong there too but we don't have to move them now").
- New test file: `test_lora_audit.py` at the worktree root, matching the existing `test_*.py` integration-suite convention (the `tests/` subdirectory is for `test_lora_format_convert*` unit tests on `nodes/` modules and is not the right home for a CLI integration suite).
- New fixture tree: `tests/fixtures/lora_audit_tree/` — synthetic safetensors files generated in-place by the test's `setUpModule` so the worktree stays repo-clean.
- No changes to `nodes/__init__.py`, `pyproject.toml`, `requirements.txt`, `uv.lock`, or CLAUDE.md in this ADR's slice.

### 2. Reuse strategy

The script lives outside the `nodes/` package and consumes its public-shape functions by name. To make this work without refactoring `nodes/`:

```python
# At the top of scripts/lora_audit.py:
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from nodes.eric_diffusion_lora_check import (
    check_lora, build_param_dict_from_dir, LoRACheckResult,
)
from nodes.eric_lora_format_convert_apply import (
    find_matching_plan, convert_state_dict,
)
from nodes.eric_qwen_edit_lora import (
    load_lora_with_key_fix, _load_state_dict,
)
```

`sys.path.insert` is the entire integration cost. No new modules in `nodes/`, no promotion of helpers to a shared `runtime/` package. (Promotion is the Runtime-core cluster slice in Backlog → Queued; sequencing it under this slice would conflict with that slice's design when it lands.)

`folder_paths` is the one ComfyUI shim that `nodes/eric_qwen_edit_lora.py` imports at module level (line 40; `get_lora_list` etc. depend on it). Python executes module top-level code on first import, so **any** `from nodes.eric_qwen_edit_lora import …` triggers that line. There is no import surface that avoids it — **the stub is always installed, not conditionally** (per security-auditor F-4; the prior "spot-check / if needed" framing was a spec gap). The exact preamble, which MUST appear before any `from nodes.* import …`:

```python
import sys, types
_fp = types.ModuleType("folder_paths")
_fp.get_folder_paths = lambda _category: []
_fp.get_full_path = lambda _category, _name: None
sys.modules["folder_paths"] = _fp
# ... only now import from nodes.*
```

This installs the stub into `sys.modules` so the real `folder_paths` from any ComfyUI install on `sys.path` is never consulted — a path-source the audit tool has no reason to read. The order-sensitivity is closed by a test: `test_no_real_folder_paths_import` (S1 deliverable) asserts that after script load, `sys.modules["folder_paths"]` is the stub `ModuleType`, not a real ComfyUI module. Documented in the script's module docstring AND enforced by the test, not docstring alone.

### 3. Classification taxonomy

Four top-level classifications, each with a closed set of reason codes. The catalog matches on classification; reason codes are diagnostic.

| Classification | Reason codes | Semantics |
|---|---|---|
| `usable` | `ok`, `norm_targeting`, `dim_mismatch_partial` | At least one configured base reports `LoRACheckResult.verdict in {OK, NORM_TARGETING}` (NORM_TARGETING is a "loads-but-PEFT-fast-path-silently-drops-norm-layers" caveat; loader works via direct-merge path). `dim_mismatch_partial` covers the case where `key_match_pct >= 50%` AND `dim_ok_pct >= 90%` AND `verdict == DIM_MISMATCH` (a handful of dim mismatches against a mostly-matching base; the loader has historically loaded these). |
| `convertable` | `lokr_to_lora_svd`, `lora_qkv_split`, `lora_passthrough` | `find_matching_plan(state_dict, base_param_names)` returns a non-None plan for at least one configured base. Reason codes match the plan's per-module behaviour described in `nodes/eric_lora_format_convert_apply.py:24-39`. |
| `unconvertable` | `wrong_arch`, `poor_match_no_plan`, `loha_unsupported`, `format_unknown`, `arch_mismatch_diffusers_only` | Loader's compatibility check returns `WRONG_ARCH` / `POOR_MATCH` AND no `find_matching_plan` hit. `loha_unsupported` is the explicit LoHa-source case (apply module marks LoHa→standard not-implemented); a follow-up slice may transition this to `convertable_loha` (Vision §Out-of-scope, "next slice after MVP"). `format_unknown` covers `_detect_adapter_type()` returning `"unknown"`. `arch_mismatch_diffusers_only` is the case where the LoRA targets original-format `double_blocks`/`single_blocks` and no diffusers conversion plan covers it (e.g. Chroma1-HD with no registered plan). |
| `deletable` | `zero_byte`, `truncated_header`, `unparseable_header`, `unrecognized_extension_zero_content` | File is genuinely garbage. `truncated_header` = safetensors header length field declares N bytes but file is shorter. `unparseable_header` = header is N bytes but not valid JSON. `unrecognized_extension_zero_content` = `.pt`/`.bin`/`.pth` that fails `torch.load(weights_only=True)` with corruption signature (not just "wrong format"). |

A file may also receive `classification: "error"` (per-file fault isolation — Vision invariant 9) with reason `error: "<short message>"`. The classifier does NOT mark a file `deletable` because of any other failure mode; only the four signatures above.

Precedence order for files where multiple classifications could fit:
1. `deletable` (garbage signatures) checked first — short-circuit before any base comparison.
2. `usable` (verdict against any configured base is good) — wins over `convertable` because no transformation is needed.
3. `convertable` (a registered plan fits some configured base).
4. `unconvertable` (none of the above fits).

**Deserialization-safety assumption (per security-auditor F-7):** `torch.load(..., weights_only=True)` is sufficient for torch 2.11.0 (ADR-013 pin) against RCE-class threats — it restricts unpickling to a weight-shaped allowlist (tensors, dicts, lists, basic numerics). Two residual concerns are closed here:
- **`add_safe_globals` inheritance.** If any transitively-imported module (`nodes/eric_*`, diffusers, transformers, peft) calls `torch.serialization.add_safe_globals`, the audit tool's `weights_only=True` would inherit the widened allowlist. S1 proof hook: `grep -rn 'add_safe_globals' nodes/ scripts/` returns zero matches. If a future dep bump introduces such a call, the assumption is re-audited.
- **Disk-fill via maliciously-sized `.pt`.** `.pt` is a zip container; a crafted file can declare gigabytes of nominal content. Bounded by a **5 GB per-file size cap** (`Path.stat().st_size`) applied *before* `torch.load` is invoked; files over the cap are classified `error` with reason `size_cap_exceeded` and **never opened**. (A LoRA over 5 GB is itself suspicious in an adapter tree; the cap surfaces it rather than silently OOM-ing.) `safetensors` output has no pickle/RCE surface; its only risk is output disk-fill, already bounded by the conversion path's `target_rank=64` SVD cap.

### 4. Base-model specification

Bases are named (label) and pointed at a transformer subdirectory (path containing `*.safetensors` shards readable by `build_param_dict_from_dir()`). Two ingestion modes that compose:

**TOML config** (default: `~/.config/lora_audit.toml`, override via `--config`):
```toml
[bases]
klein = "/home/gawkahn/projects/ai-lab/ai-base/models/hf-local/Flux2-Klein-9B/transformer"
chroma_base = "/home/gawkahn/projects/ai-lab/ai-base/models/hf-local/Chroma1-base/transformer"
flux2_dev = "/home/gawkahn/projects/ai-lab/ai-base/models/hf-local/Flux2-dev/transformer"

# Optional defaults (any may be omitted):
[defaults]
dry_load = false           # global default; CLI --dry-load overrides
```

**CLI flags:**
- `--base name=/abs/path` (repeatable). Adds a base; if `name` already exists from config, fails at startup unless `--override-base name=/new/path` is used. Names must match `^[a-zA-Z0-9_-]+$`.
- `--config PATH` to point at a non-default config file (or `--no-config` to skip config-file ingestion entirely).

**Validation at startup (fail-closed):**
- Each base path is resolved with `Path.resolve(strict=True)` and rejected if non-existent or not a directory.
- Each base path is checked for at least one `*.safetensors` shard before scanning begins (cheap glob; not a load). Bases that fail this check are flagged at startup; the user can choose to proceed without them (per `--continue-on-bad-base`) or abort. Default: abort.
- Names must be unique across config + flags after `--override-base` resolution; collision = startup abort.

The combination of config and flags exists because:
- Grant runs the tool interactively against the same base set repeatedly; config saves typing (matches `feedback_shell_command_format` ergonomics).
- The catalog (machine caller) builds the base set per-invocation from its own state; it uses `--no-config --base name=path …` and never touches the config file.

### 5. Manifest schema v1

The manifest is one JSON file written atomically to the audit root (default name: `lora_audit.json`, override via `-o PATH` which must resolve to a descendant of the audit root unless `--output-dir` is also given for the convert path).

```json
{
  "audit_version": 1,
  "tool_version": "0.1.0",
  "audited_at": "2026-05-17T14:30:00Z",
  "audit_root": "/abs/path/to/scanned/tree",
  "bases": {
    "klein":       {"path": "/abs/path/...", "param_count": 1234, "dry_load_attempted": true},
    "chroma_base": {"path": "/abs/path/...", "param_count": 5678, "dry_load_attempted": false}
  },
  "tool_invocation": {
    "argv_redacted": ["scripts/lora_audit.py", "--audit-root", "<root>", "--dry-load"],
    "config_path": "/home/gawkahn/.config/lora_audit.toml",
    "config_sha256": "abc..."
  },
  "totals": {
    "files_scanned": 312,
    "usable": 198,
    "convertable": 47,
    "unconvertable": 65,
    "deletable": 2,
    "error": 0
  },
  "files": [
    {
      "kind": "lora",
      "relative_path": "subdir/klein_snofs_v1_1.safetensors",
      "sha256": "9e8d...",
      "size_bytes": 18874368,
      "classification": "usable",
      "reason": "ok",
      "verdicts_by_base": {
        "klein":       {"verdict": "OK",         "key_match_pct": 100.0, "dim_ok_pct": 100.0, "dry_load": {"loaded": true,  "applied_modules": 304}},
        "chroma_base": {"verdict": "WRONG_ARCH", "key_match_pct":   0.0, "dim_ok_pct":   0.0, "dry_load": {"loaded": false, "reason": "0 modules patched"}}
      },
      "convert_plan": null,
      "convert_output": null,
      "error": null
    },
    {
      "kind": "lora",
      "relative_path": "old/realism_v3.safetensors",
      "sha256": "...",
      "size_bytes": 167772160,
      "classification": "convertable",
      "reason": "lokr_to_lora_svd",
      "verdicts_by_base": {
        "klein":       {"verdict": "WRONG_ARCH", "key_match_pct": 0.0, "dim_ok_pct": 0.0, "dry_load": {"loaded": false, "reason": "..."}},
        "chroma_base": {"verdict": "WRONG_ARCH", "key_match_pct": 0.0, "dim_ok_pct": 0.0, "dry_load": {"loaded": false, "reason": "..."}}
      },
      "convert_plan": {
        "source_family": "bfl_chroma",
        "target_family": "diffusers_chroma",
        "target_base":   "chroma_base"
      },
      "convert_output": "old/realism_v3.diffusers.safetensors",
      "error": null
    },
    {
      "kind": "lora",
      "relative_path": "orphans/mystery.safetensors",
      "sha256": null,
      "size_bytes": 0,
      "classification": "deletable",
      "reason": "zero_byte",
      "verdicts_by_base": {},
      "convert_plan": null,
      "convert_output": null,
      "error": null
    }
  ],
  "warnings": [
    {"file": "weird/escape.safetensors", "code": "excluded_symlink_escape", "detail": "realpath outside audit_root"}
  ]
}
```

**Schema rules:**
- `audit_version: 1` is the contract. Schema-breaking changes bump it AND ship a migration note in this ADR's Changelog. **Forward-compatible changes** (new `kind`, new reason codes, new optional fields on entries) are NOT breaks — catalog parsers MUST ignore unknown keys per Vision invariant 12.
- `tool_version` is sourced from a `_TOOL_VERSION = "0.1.0"` constant in `scripts/lora_audit.py`. Bumps are ADR-Changelog entries.
- `audited_at` is ISO 8601 UTC. **Excluded from the determinism check** (Vision invariant 8 / negative case 7) via jq mask.
- `relative_path` is POSIX-style ("/" separator) relative to `audit_root`, sorted alphabetically across the `files` array (determinism enabler).
- `sha256` is `null` only for `deletable: zero_byte` (no bytes to hash); always present otherwise. Lowercase hex.
- `verdicts_by_base` is empty `{}` for `deletable` (no base comparison is run — files are short-circuited before base ingestion).
- `convert_plan` is `null` unless `classification == "convertable"`. `target_base` names which configured base the plan was matched against (always exactly one — first-match-wins, matches `find_matching_plan` semantics).
- `convert_output` is the **relative path** of the sibling file that would be / was written (depending on whether `--convert` was set). Always populated for `convertable`; the catalog reads it directly.
- `tool_invocation.argv_redacted` (per security-auditor F-8): **every path-shaped flag value** is replaced with the literal flag name + `<redacted>` (not just `--audit-root`). This covers `--config`, `--base name=PATH`, `--output-dir`, `--output-allowlist-prefix`, and `-o`. The resolved paths are recorded *only* in typed fields — `audit_root`, `bases.*.path`, the top-level `output_dir` (present when `--convert --output-dir` is used), and `tool_invocation.config_path`. The catalog reads paths from those typed fields, never from `argv_redacted`. This makes the redaction principle uniform and removes the "could leak" ambiguity, so a manifest shared for diagnostics doesn't leak `/home/<username>` or sensitive subdirectory names through the argv echo.
- `tool_invocation.config_sha256` records the SHA-256 of the config file contents (if any) so the catalog can detect "manifest from old config, base set has changed."
- `warnings` is a separate array for non-file-specific warnings (symlink escapes, base-model header parse failures, stale `.tmp` files, etc.). Per-file errors live in the file's `error` field. **The `warnings` array is sorted lexicographically by `(file, code)`** (per security-auditor F-13) so the determinism property (invariant 8) survives any future scan-parallelism optimization that would otherwise emit warnings in nondeterministic order.

**Key ordering rule:** All dict keys at every depth are sorted alphabetically via `json.dumps(..., sort_keys=True)`. Combined with relative-path sorting in `files[]`, this gives byte-identical output across runs (modulo `audited_at` and `audited_run` data the catalog masks).

### 6. Path-traversal and symlink policy

**Authoritative containment, plus fd-based defense-in-depth (per security-auditor F-1, F-14, F-15 — Grant's choice 2026-05-23, Option B).** The authoritative control is the realpath descendancy check at scan-enumeration time — portable, works everywhere, and is what the threat model relies on for the MVP same-trust-zone callers. The reused classifier (`check_lora` in `nodes/eric_diffusion_lora_check.py` and `_load_state_dict` in `nodes/eric_qwen_edit_lora.py`) re-opens files **by path**, so a validated fd cannot be threaded into the read without refactoring `nodes/` — forbidden by Vision invariant 7 (reuse via import only). Honest framing: **the fd-based open below NARROWS the terminal-symlink-swap window, it does not CLOSE the TOCTOU.** The residual same-uid race is explicitly accepted under the MVP threat model because an attacker with same-uid write access under `audit_root` can `os.unlink()` files directly — narrowing the audit-tool race grants the attacker no capability they lack.

The future LLM-agent / remote-caller case re-classifies to Red Zone per the header's risk-trigger note (F-10); when that transition lands, the *new* ADR mandates the proper closure — fd-accepting readers (a `nodes/` refactor folded into the Runtime-core cluster slice), sandboxing (Alternative L), or both. ADR-014 does not promise a closure the reuse-only constraint cannot deliver today.

**Pattern (defense-in-depth, applied universally):** open with `O_NOFOLLOW` so the final path component is never a followed symlink; on Linux re-derive the realpath via `/proc/self/fd/{fd}` and re-check descendancy as a **bonus** narrowing; on non-Linux the `O_NOFOLLOW` alone is the bonus, and the scan-enumeration realpath remains the authoritative control. **No Linux-only hard dependency in the authoritative control path** — reconciling F-15's "team-portable lens" finding against Alternative L's own reasoning for rejecting Linux-only sandboxing as the default.

```python
audit_root = Path(args.audit_root).resolve(strict=True)
if not audit_root.is_dir():
    sys.exit("[ERROR] audit-root must be a directory")

def passes_scan_containment(path: Path) -> bool:
    """AUTHORITATIVE control: per-entry realpath must descend from audit_root."""
    try:
        real = Path(os.path.realpath(path))            # follows symlinks
    except OSError:
        return False
    try:
        real.relative_to(audit_root)
    except ValueError:
        emit_warning("excluded_symlink_escape", path)  # realpath escaped the root
        return False
    if not real.exists():
        emit_warning("dangling_symlink", path)
        return False
    return True

def open_no_follow(path: Path):
    """DEFENSE-IN-DEPTH: O_NOFOLLOW open + (Linux-only) per-fd realpath re-check.

    Narrows the terminal-symlink-swap window between scan and read; does NOT
    close it (the reused classifier re-opens by path — see ADR §6 framing).
    The fd is closed after the bonus re-check; the path-based reader takes
    the path. The authoritative guarantee is passes_scan_containment() above.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as e:
        if e.errno == errno.ELOOP:
            emit_warning("excluded_symlink_escape", path)
        else:
            emit_warning("unreadable", path)
        return False
    try:
        if sys.platform == "linux":  # F-15: bonus only, not authoritative
            real_fd = Path(os.path.realpath(f"/proc/self/fd/{fd}"))
            try:
                real_fd.relative_to(audit_root)
            except ValueError:
                emit_warning("excluded_symlink_escape", path)
                return False
        return True
    finally:
        os.close(fd)

for path in audit_root.rglob("*"):
    if path.is_dir():
        continue
    if not passes_scan_containment(path):
        continue
    if not open_no_follow(path):       # narrows; not the authoritative control
        continue
    # Classifier reads by path (reuse-only). Authoritative containment was
    # the realpath check above; this read inherits its residual TOCTOU.
    classify(path)
```

The residual TOCTOU is the gap between `passes_scan_containment(path)` and `classify(path)`'s internal `open(path, "rb")` — an attacker with same-uid write under `audit_root` can swap a symlink in that window. **Accepted under the MVP threat model** because the same attacker can `os.unlink()` files directly, granting them no capability they lack. The `O_NOFOLLOW` open narrows the window further (terminal-component swaps fail loudly with `ELOOP`); the Linux-only `/proc/self/fd` re-check narrows it further still where available. None of the three is a guarantee in the reuse-only world; the realpath check is the authoritative posture and is portable.

**Realpath-semantics dependency (per security-auditor F-9).** Per-entry containment relies on `os.path.realpath` / `Path.resolve()` returning the realpath of the *current* on-disk state of the entry, not the state captured at audit-root-resolve time. This is what closes the "rename `audit_root` to `audit_root.bak` and symlink `audit_root -> /etc` mid-scan" race: per-entry realpath then resolves to `/etc/...`, the `relative_to(audit_root)` check raises `ValueError`, and the entry is skipped. **A future refactor that switches to a non-realpath check (e.g. string-prefix comparison after `os.path.normpath`) would silently re-open this window** — flagged here so it isn't lost.

**`--output-dir` policy (per security-auditor F-6 — dual-caller).** The default and the interactive convenience flag are separated from the machine-caller requirement:
- **Inside audit_root (default).** No flag needed; `--output-dir` resolving to a descendant of `audit_root` is the primary control. The blacklist below is not consulted.
- **Interactive Grant, outside root (`--allow-output-outside-root`).** A convenience escape that rejects paths containing `..` components and rejects a system-directory blacklist `{/, /etc, /usr, /var, /sys, /proc, /dev, ~/.ssh, ~/.gnupg}`. **The blacklist is explicitly acknowledged as gappy** (it cannot enumerate every catastrophic target) and exists only as a foot-gun-dampener for the interactive single-user case. It is NOT a security control the machine caller may rely on.
- **Machine caller, outside root (`--require-output-allowlist`).** The catalog MUST pass this flag, which *disables* `--allow-output-outside-root` and instead demands one or more `--output-allowlist-prefix /abs/path` (repeatable). Every output path must (a) be a descendant of some supplied allowlist prefix, (b) contain no `..` components, (c) emit `[WARN] writing_outside_audit_root: <path>` per file written. An allowlist (closed set of permitted roots) replaces the blacklist (open-ended set of forbidden roots) for the untrusted-caller path, per the "blacklists are an anti-pattern" finding. Full replacement of the interactive blacklist by the allowlist for all callers is deferred to a follow-up slice; MVP keeps the blacklist as the interactive-only convenience.

**Default-deny + mutual exclusion (per security-auditor F-17).** Two startup invariants close the failure modes a permissive default could create:
- An `--output-dir` resolving outside `audit_root` with **neither** `--allow-output-outside-root` nor `--require-output-allowlist` is a startup error (exit 1, `[ERROR] output-dir is outside audit-root; pass --allow-output-outside-root for interactive use or --require-output-allowlist + --output-allowlist-prefix for machine use`). No silent fallthrough.
- `--allow-output-outside-root` and `--require-output-allowlist` are **mutually exclusive**; passing both is a startup error. This prevents the machine caller who forgets `--require-output-allowlist` from silently being granted the weaker blacklist gate.

Config file path is resolved and read; the path itself is not constrained, but the *paths it specifies* (base paths) are subject to the same `Path.resolve(strict=True)` + existence check.

`--delete` targets are subject to a triple gate enforced in code order: (a) classification is `deletable`, (b) fd-based realpath is a descendant of `audit_root`, (c) `--yes` is set. All three required. See §9 for the additional pre-unlink re-classification (F-5).

### 7. Dry-load mechanism

**Trigger:** `--dry-load` flag (default off; `[defaults] dry_load = true` in config flips the default).

**Mechanism:** For each configured base, sequentially:
1. Load the **full pipeline** via the same code path that nodes use — concretely `diffusers.AutoPipelineForText2Image.from_pretrained(base.parent_dir, local_files_only=True, torch_dtype=torch.bfloat16)`. (Base.parent_dir = the parent of the transformer subdir; diffusers expects the model root containing `model_index.json`.)
2. For every LoRA in the audit tree whose shape-match against this base returned `verdict != WRONG_ARCH` (i.e. anything that's not 0%-match-no-plan), call `load_lora_with_key_fix(pipe, lora_path, adapter_name=<derived>)`. Record `loaded: True|False` and `applied_modules` count (the loader returns this in its log; the audit tool captures it via a context-manager log-shim).
3. Unload all adapters: iterate the just-loaded names and call `unload_adapters(pipe, adapter_names)`.
4. Delete the pipeline (`del pipe; torch.cuda.empty_cache()`) before the next base.

**Why full pipeline, not transformer-only:** `load_lora_with_key_fix` calls `pipe.load_lora_weights()` on the fast path and `pipe.get_list_adapters()` for verification — both pipeline-level operations, not transformer-attribute operations. A transformer-only load would force the audit tool to fork the loader's verification path, which violates Vision invariant 7 (reuse, don't reimplement). The cost is one full-pipeline load per base (seconds to tens of seconds depending on family); the benefit is loader-faithful dry-load semantics.

**Failure mode (base load fails, e.g. OOM):** that base's `dry_load_attempted: false` in the manifest, all of its `verdicts_by_base[base]` entries fall back to shape-match-only (no `dry_load` sub-object), warn loud on stderr (`[WARN] dry-load skipped for base 'klein': OOM. Falling back to shape-match.`), continue with the next base.

**VRAM partial-state inheritance (per security-auditor F-3).** diffusers `from_pretrained` partially loads weight shards into VRAM before raising on OOM / version mismatch. The exception unwinds the Python pipeline reference, but `torch.cuda.empty_cache()` only returns *pooled* memory to the allocator — it does not free fragmented or leaked allocations. This is the same partial-state-on-OOM failure mode already tracked as comfyless-server tech debt (Backlog "Comfyless server failed-load / OOM-cascade resilience"). The audit tool mechanically inherits it by using the same load surface. Consequence: after any base load failure, **subsequent base loads in the same process may OOM for reasons unrelated to their own footprint** — a cascade. MVP behavior: keep the sequential loop, but after any base failure emit an additional `[WARN] vram_cascade_possible: prior base load failed; subsequent base failures may be downstream effects, not the base's own problem`, so a degraded all-shape-match manifest is diagnosable rather than silently misleading. **Deferred (not out-of-scope):** per-base subprocess isolation (fork+exec a fresh Python per base load) closes the cascade by giving each base a clean VRAM process; this lands as a follow-up slice if real audit runs show the cascade bites. Cross-referenced to the Backlog OOM-resilience item.

**Failure mode (per-LoRA dry-load raises):** caught; that file's `verdicts_by_base[base].dry_load.loaded = false`, `reason = "<short traceback summary>"`. Run continues. Per-file fault isolation (Vision invariant 9) extends through dry-load.

**Performance note (advisory, not invariant):** dry-load is O(num_bases × num_loras); typical run on Grant's tree (~3 bases × ~500 LoRAs at ~5s per dry-load attempt) is ~2 hours. The tool emits progress to stderr (`[INFO] base 'klein' (2/3): loaded; dry-loading 487 candidates...`) so a wrapping script can observe liveness. Future optimization (transformer-only dry-load, parallel per-base, GPU-skip) is its own slice if the wall-clock cost matters.

### 8. Convert path output naming

For each `classification: "convertable"` file, with `--convert`:
- Output path: `<source_stem>.<target_family>.safetensors` in the same directory as the source (default), or in `<output_dir>/<relative_source_dir>/<source_stem>.<target_family>.safetensors` if `--output-dir` is given.
- `<target_family>` comes from `ConversionPlan.target_family` (e.g. `"diffusers_chroma"`, `"diffusers_klein"`).
- Pre-write check: target path must not exist. If it exists, skip with `[WARN] convert_skipped_collision: <relative_path>` AND record `convert_output: null` in the manifest entry with `reason: "collision"`. Source is never modified. There is no `--in-place` flag in this slice.
- Atomic write: `safetensors.torch.save_file(state_dict, target + ".tmp"); os.replace(target + ".tmp", target)`. POSIX-atomic within the same filesystem; across filesystems, fall back to `shutil.move()` (atomic-or-error semantics).
- Permissions: created file inherits umask; no explicit `chmod`. (Grant's setup does not require Unix-permission discipline on artifacts; if a deployment ever does, that's a future ADR.)
- The new file is NOT re-classified in the same run — it appears in the next audit run if `lora_audit.py` is invoked again. Recursive classification within a single run is out of scope (avoids "did we converge?" semantic complexity).

### 9. Delete policy

`--delete` is gated on three conditions, all required:
1. File classification is `deletable` (one of the four signatures in §3). This is the *primary* gate — the flag does not "promote" files into deletable.
2. `Path.resolve(strict=True)` of the file is a descendant of `audit_root`.
3. `--yes` is also passed.

Without `--yes`, the tool emits `[INFO] would delete N files (zero_byte: A, truncated_header: B, unparseable_header: C, unrecognized_extension_zero_content: D)`, prints the relative paths, and exits 0. No I/O.

**Pre-unlink re-classification (per security-auditor F-5, F-16 — Option B closure).** With `--yes`, before each `unlink()` the tool **re-runs the `deletable` signature checks on the file as it exists at delete time** (the manifest classification was computed at scan time; the file may have been content-swapped between scan and unlink). The deletable signatures are all cheap (zero-byte `stat`, header struct unpack, header JSON parse). If the re-classification no longer returns `deletable`, the delete is **skipped** and the manifest entry records `delete_skipped_classification_changed`. This **narrows** the content-substitution race (it does not close it — see residual below).

**Dir-fd-relative unlink (per security-auditor F-16).** The unlink itself uses `os.unlink(name, dir_fd=parent_fd)` with `parent_fd` obtained via `os.open(parent_dir, O_RDONLY | O_NOFOLLOW | O_DIRECTORY | O_CLOEXEC)` and the same realpath descendancy re-check from §6 applied to the parent before the unlink. This prevents a terminal-component symlink swap between the fd-check and the unlink syscall from targeting a different inode than the one validated. The `O_NOFOLLOW` on the parent open also prevents an intermediate-directory swap on the parent from redirecting the unlink.

```python
def safe_unlink(path: Path, audit_root: Path):
    parent = path.parent
    if not passes_scan_containment(parent):  # §6 authoritative control on parent
        return False
    try:
        parent_fd = os.open(parent, os.O_RDONLY | os.O_NOFOLLOW
                                    | os.O_DIRECTORY | os.O_CLOEXEC)
    except OSError:
        return False
    try:
        if sys.platform == "linux":
            real_parent = Path(os.path.realpath(f"/proc/self/fd/{parent_fd}"))
            try:
                real_parent.relative_to(audit_root)
            except ValueError:
                return False
        # Re-classify the file as it exists right now (F-5 narrowing)
        if reclassify_deletable(path) is None:
            emit_warning("delete_skipped_classification_changed", path)
            return False
        os.unlink(path.name, dir_fd=parent_fd)   # F-16: dir-fd-relative
        return True
    finally:
        os.close(parent_fd)
```

**Residual (named, accepted per Option B).** The reclassify→unlink sequence still has a content TOCTOU between the reclassify's path-based open and the dir-fd-relative unlink — an attacker with same-uid write under `audit_root` can swap the file content in that window. The window is small (microseconds in the absence of paging), and the same attacker can already `os.unlink()` files directly, so they gain no capability they lack. The proper closure (fd-accepting reclassifier or sandboxing) lands when the F-10 risk-trigger fires.

Manifest records each performed deletion as `error: null, deleted: true` (manifest entry stays; the file is gone). The catalog uses this to scrub the path from its index.

There is **no interactive prompt**. The machine-caller invariant (Vision invariant 13) requires `--yes` to be the only authorization signal. A future slice MAY add an `--interactive-confirm` flag explicitly opted into by interactive users, but it is not the default.

### 10. Atomic-write mechanism

Both the manifest and the converted-LoRA outputs use the `.tmp` → `os.replace()` idiom:

```python
def _write_atomic(target: Path, write_fn):
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        write_fn(tmp)
        os.replace(tmp, target)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
```

A SIGKILL between `write_fn(tmp)` and `os.replace(tmp, target)` leaves the `.tmp` file orphaned. **The tool does NOT clean these up at scan time (per security-auditor F-2, Option A — Grant's choice 2026-05-19).** Scan-time `.tmp` deletion would be unsafe under concurrent invocations: the catalog invokes this tool as a subprocess on every newly-downloaded LoRA, so invocation A may be mid-write to `foo.diffusers.safetensors.tmp` when invocation B scans the same tree — B must not mistake A's in-flight tmp for an orphan and unlink it. Instead, a stray `*.tmp` under the audit root is surfaced as `[WARN] stale_tmp_file: <relative_path>` (and recorded in the `warnings` array); the operator decides whether to remove it. The next legitimate convert of the same target either succeeds (target absent) or reports `convert_skipped_collision` on the *target* (never on the tmp). This keeps invariants 3 and 4 absolute — there is **no** tool-internal exception that deletes a file outside the `deletable` classification + `--yes` gate. Documented in the script's docstring.

### 11. Tool versioning and stderr format

- `_TOOL_VERSION = "0.1.0"` constant in `scripts/lora_audit.py`. Bumps land as ADR-Changelog entries when the schema or behavior changes in a way callers care about. SemVer-ish: minor bumps for additive schema changes, major for breaks (which also bump `audit_version`).
- Stderr lines follow `^\[(WARN|INFO|ERROR)\] <message>$`. ERROR is for startup-fail or per-file-classification-raise; WARN is for non-fatal anomalies (symlink escape, base load failure, convert collision); INFO is for progress (per-base load timing, per-file classification ticker every 100 files).
- No log file. The catalog captures stderr if it wants it.
- Stdout is **empty** unless `--print-manifest` is set (in which case it writes the manifest JSON to stdout instead of the file). This keeps the default case clean for shell-pipe consumption.

### 12. Exit codes

- `0` — scan completed, manifest written (or printed), every file got a classification (including `error`). Warnings may have been emitted.
- `1` — startup failure (bad config, bad audit root, bad base path, path traversal, missing required flag). Manifest is NOT written. Stderr explains.
- `2` — scan completed, manifest written, but at least one file got `classification: "error"`. The catalog can choose to fail or surface a warning. (This is distinct from `0` because the catalog may want to re-run later when transient errors clear.)

There is no `3` or higher; "error during scan that nonetheless wrote a manifest" maps to `2`. "Error so severe we can't write the manifest" maps to `1`.

### 13. Test fixtures and the test suite

- `tests/fixtures/lora_audit_tree/` is generated programmatically in `setUpModule()`. The fixtures are tiny (≤ 1 KB each) — handcrafted state dicts via `safetensors.torch.save_file`, plus zero-byte file via `Path.touch()`, plus truncated file via `Path.write_bytes(b'\x10\x00\x00...')` (header length declared as 16 but file is short).
- A tiny synthetic base-model `tests/fixtures/synthetic_base/transformer/` is generated similarly: a few `transformer_blocks.0.attn.to_q.weight` tensors of known shape so `build_param_dict_from_dir` works. The dry-load test cases use this base (no real diffusers pipeline load is performed in unit tests; the `--dry-load` mode is exercised end-to-end via a separate `test_dry_load_integration.py` that's skipped by default unless `LORA_AUDIT_DRY_LOAD_E2E=1`).
- Negative-case tests map 1:1 to Vision negative cases (§Negative cases required, 12 numbered items).
- Determinism test runs the script twice in a `subprocess.run` and diffs the JSON with `audited_at` and `tool_invocation.argv_redacted` masked.
- Per-file fault-isolation test asserts `exit_code == 2` and `manifest["files"][i]["classification"] == "error"`.
- Machine-caller test runs `subprocess.run([..., "--delete", "--yes"], stdin=subprocess.DEVNULL, capture_output=True, text=True)` and asserts no `EOFError`, exit code is documented value, stderr lines match the prefix regex, manifest parses.

### 14. Order of operations (per global §12)

1. Vision slice (`docs/vision/slice-lora-audit.md`) — **approved 2026-05-17.** ✓
2. ADR-014 (this document) — Status: `proposed`. ✓
3. Vault mirrors of Vision (done 2026-05-17) and this ADR (to `Decisions/ADR-014-LoRA-Audit-Tool.md` immediately after acceptance).
4. `security-auditor` (Opus, `model: "opus"` at invocation per `feedback_agent_model_pin_broken`) reviews ADR-014. Output saved to `docs/security/review-lora-audit-2026-05-17.md` and mirrored to `Security/Review-2026-05-17-LoRA-Audit.md`.
5. Iterate ADR if `CHANGES REQUIRED`. Re-fire `security-auditor` until `CLEAN`. Status flips to `accepted`.
6. `uv sync` in this worktree (one-time prerequisite per Vision invariant 11).
7. Code in slices via `/change-slice` (slice plan in §15 below). Each non-trivial slice runs `code-reviewer` (Opus). Slices touching write/delete also run `security-auditor` (Opus) per project CLAUDE.md Review-Bar.
8. Final regression: all eight existing test suites (850 tests) pass against `./.venv/bin/python3`. New `test_lora_audit.py` adds to the count.
9. Backlog update + Obsidian mirror of all artifacts. Push approval requested at batch close.

### 15. Slice plan

In dependency order; each slice is its own commit (or small set of commits) and runs the reviewer cadence above before its commit body lands. Commit prefix `tool:` per project CLAUDE.md "tool:" prefix history.

| Slice | Scope | Reviewer cadence |
|---|---|---|
| **S1: Skeleton + manifest + shape-match classify** | `scripts/lora_audit.py` with config + flag ingestion, the fd-based path-traversal policy (§6), the `folder_paths` stub preamble (§2), shape-match-only classification via `check_lora`, `convertable` detection via `find_matching_plan`, manifest write. No `--dry-load`, no `--convert`, no `--delete` yet. `test_lora_audit.py` skeleton with positive + path-traversal + manifest-determinism + per-file-fault-isolation + machine-caller-non-interactive + forward-compatibility + `test_no_real_folder_paths_import` tests. | `code-reviewer` + `security-auditor` (both Opus) — per security-auditor F-12, S1 lands the path-traversal + manifest + audit-root-containment surface that S2/S3/S4 depend on; reviewing only at the later slices opens a regression window in the load-bearing boundary code. Subsequent slices' `security-auditor` invocations focus on the *new* write/delete/dry-load surfaces, not on re-reviewing S1's boundary code. |
| **S2: Dry-load mode** | `--dry-load` path; sequential per-base full-pipeline load via diffusers; per-LoRA `load_lora_with_key_fix` call; integration test gated on `LORA_AUDIT_DRY_LOAD_E2E=1`. | `code-reviewer` + `security-auditor` (both Opus) — dry-load loads caller-supplied LoRA weights into the GPU, which expands the deserialization surface. |
| **S3: Convert path** | `--convert` writes sibling files atomically; collision skip; output-dir validation. Tests for no-overwrite + atomicity + output-dir traversal rejection. | `code-reviewer` + `security-auditor` (both Opus) — file writes from caller-supplied input. |
| **S4: Delete path** | `--delete` removes `deletable`-only files; `--yes` gate; preview mode. Tests for triple-gate + no-promotion + audit-root containment. | `code-reviewer` + `security-auditor` (both Opus) — file deletions from caller-supplied input. |
| **S5: Backlog close + Obsidian mirror + commit batch push** | Backlog entry transitions; vault mirrors confirmed for all four code commits' ADRs / security review; ask Grant before push. | None (mechanical). |

Five code-bearing slices + ADR + Vision = a single PR-equivalent batch by the time S5 closes. The push-cadence rule (`feedback_push_cadence`) means one ask at the end of S5, not after each slice.

---

## Alternatives Rejected

### A. Implement the classifier from scratch in `scripts/lora_audit.py` instead of reusing `check_lora` / `find_matching_plan`

Rejected. Violates Vision invariant 7. The classifier already exists, is tested, and is the same code the production loader uses as its pre-flight gate. Duplicating it creates two definitions of "usable" that will drift; the loader's definition is authoritative.

### B. Single fast-mode (shape-match only, no dry-load)

Rejected per Grant 2026-05-16: "experience suggests that the shape matching doesn't tell the whole story." Shape-match is the pre-filter (decides which dry-load attempts to make and is the fallback when dry-load can't run), but the authoritative answer requires actually attempting the load. Both modes exist; `--dry-load` defaults off but config can flip the default.

### C. Per-file JSON sidecars (`foo.safetensors.audit.json`) instead of a single top-level manifest

Rejected per Grant 2026-05-16 ("Single top-level JSON at audit-root"). The catalog wants one file to consume; distributing the data across thousands of sidecars makes catalog cold-start more expensive (filesystem walks the tree just to assemble what the audit tool already had in memory). The single-manifest choice is consistent with the existing `analyze_checkpoint.py` precedent of "one tool run = one report to stdout".

### D. Allow `--in-place` flag to overwrite source LoRAs after conversion

Rejected. The no-overwrite invariant (#2) and the "warn-don't-block" feedback (`feedback_warn_dont_block`) sit in tension here: warn-don't-block says "let the user shoot themselves in the foot when they really want to," but the foot-shooting here is *irreversible source corruption*. Conversion is lossy in some cases (LoKR→standard LoRA via SVD truncation; see `nodes/eric_lora_format_convert_apply.py` reconstruct_lokr_delta + SVD path). Overwriting source means the user loses the ability to compare against the pre-converted weights or re-convert with different parameters. The sibling-file convention is unambiguously safer; a future user-driven slice can add `--in-place` if Grant's experience shows the sibling files are operationally painful.

### E. Interactive confirm prompt before delete (instead of `--yes` flag)

Rejected per Vision invariant 13 (machine-caller contract). The catalog cannot answer interactive prompts; an interactive default would break the eventual product use case. `--yes` is the only authorization mechanism. Future slice MAY add `--interactive-confirm` opt-in, not opt-out.

### F. Transformer-only dry-load (skip VAE / text encoders) for speed

Deferred to a future slice. The dry-load mechanism uses full-pipeline load because `load_lora_with_key_fix` calls pipeline-level methods (`pipe.load_lora_weights`, `pipe.get_list_adapters`). Transformer-only would require forking the loader's verification path, violating Vision invariant 7. The performance cost (per §7) is acceptable for MVP; if Grant's experience shows otherwise, the optimization slice picks it up with whatever reuse strategy fits then.

### G. Inline transformer / checkpoint auditing in this slice

Rejected per Vision (slice splits explicitly). Same classification taxonomy, different operations and failure modes. The manifest's `kind: "lora"` field is the contract that lets transformer entries land in a later slice without breaking the catalog's parser written today.

### H. Run dry-load in parallel (multiple bases at once, multiple LoRAs per base)

Rejected for MVP. Bases are GB-scale loads; the typical GPU has room for one at a time. Per-base parallelism within a single load is what `load_lora_with_key_fix` already does internally. Multi-GPU dispatching is its own slice if Grant's setup grows that capability.

### I. Emit machine-readable progress on stdout (JSONL events) instead of stderr text

Rejected for MVP. Adds a second protocol surface to design, document, and lock in. The catalog can parse `[INFO]`/`[WARN]`-prefixed stderr lines with `re.match` for liveness signals; full event protocol is a future slice if the catalog needs it.

### J. Store the config file at a comfyless-controlled path (e.g. `comfyless/config/lora_audit.toml`)

Rejected. The tool is for the catalog and for Grant; comfyless is the LLM-driven inference path and a different consumer. Default `~/.config/lora_audit.toml` matches XDG-Base-Dir-Spec conventions and keeps the audit tool's config out of any deploy package. `--config` always overrides; `--no-config` always bypasses.

### K. Use `pickle.loads(safe_unpickler=...)` for `.pt` / `.bin` / `.pth` instead of `torch.load(weights_only=True)`

Rejected. `torch.load(weights_only=True)` is the project-wide convention (matches `nodes/eric_qwen_edit_lora.py:122` `_load_state_dict`) and is the documented PyTorch-blessed safe mode for deserializing checkpoints. Rolling our own `safe_unpickler` is more code, more attack surface, and diverges from the existing pattern.

### L. Run the audit tool entirely under a `bwrap` / `firejail` / `nsjail` sandbox by default (per security-auditor F-11)

Rejected for MVP. A sandbox means even a critical bug in the deserialization or path-handling code can't escape to write outside `audit_root` — genuine kernel-boundary defense-in-depth that would close whole classes of findings (F-1, F-2, F-5, F-6) below the application layer. Rejected because: (a) it's a *deployment* concern, not a tool-design concern — any operator can wrap the CLI without an ADR amendment; (b) `bwrap`/`firejail` are Linux-only, and the project's "team-portable" lens (global §1) would not accept a Linux-only hard dependency by default; (c) it's purely additive — layering it later costs nothing now. **Re-evaluate the moment the LLM-agent bridge lands** (the risk-trigger note in the header): at that point sandboxing moves from "nice operator option" to "should be the default," and that transition's ADR should make the call.

---

## Deferred / Out of Scope

### Out of scope (this slice, in scope for the final product — Vision §Boundary)

- **Transformer / checkpoint (base-model) auditing.** Same classification taxonomy, different operations. Manifest's `kind: "lora"` field is the forward-compatibility hook. Next slice in this thread after the LoRA MVP ships.
- **LoHa conversion investigation.** Vision named this as the next slice after MVP — enumerate LoHa LoRAs in Grant's tree, prototype reconstruction (`(w1_a @ w1_b) * (w2_a @ w2_b) * scale` + SVD-truncate, mirroring the LoKR path in `nodes/eric_lora_format_convert_apply.py`), decide whether to ship or formally close as "won't fix" with documented reasoning.
- **Machine-caller surface optimizations.** `--watch` mode (catalog post-download invocation), `--single-file` short-circuit for the common catalog case, possibly daemon mode. MVP CLI is the contract; these are *additional* surfaces over the same contract.

### Out of scope (period)

- Modifications to `nodes/eric_*lora*.py`, `nodes/eric_diffusion_lora_check.py`, `nodes/eric_lora_format_convert*.py`. Reuse via import only.
- Promotion of helpers from `nodes/` to a `runtime/` package — that's the Runtime-core cluster slice in Backlog → Queued, sequenced before HTTP-transport MCP work.
- Catalog UI / consumer code (separate project; this ADR ships only the manifest contract).
- Quantized-file dequantization (NF4, fp8) — superseded by Native-quant entry in Backlog.
- HTTP / network surface for the audit tool. CLI only; future ADR if needed.
- Multi-GPU / parallel-base dry-load.
- JSONL event protocol on stdout.

### Deferred to a future ADR amendment

- Tool-version bumps when schema or behavior changes (Changelog entry suffices for additive changes; new ADR for breaking changes).
- Permissions discipline on conversion outputs (`chmod` policy).
- Manifest signing (catalog verifying provenance against a known tool key).
- Network-mounted-storage support — current invariants assume single-host POSIX semantics for the audit root.

---

## Changelog

- **2026-05-17 (initial draft)**: ADR drafted following Vision approval (2026-05-17). Status `proposed`. Next step: `security-auditor` (Opus, `model: "opus"` at invocation per `feedback_agent_model_pin_broken`) review of this ADR design before any code lands. Slice plan in §15 is the implementation contract.

- **2026-05-19 (security-auditor round-1 fold-in)**: Round-1 review (saved to `docs/security/review-lora-audit-2026-05-17.md`, mirrored to vault `Security/Review-2026-05-17-LoRA-Audit.md`) returned `CHANGES REQUIRED` with 0 HIGH, 6 MEDIUM, 4 LOW, 3 INFO. All 13 folded:
  - **F-1 (MED)**: §6 replaced resolve-then-open with the fd-based `os.open(O_NOFOLLOW)` + `/proc/self/fd` realpath re-check pattern, closing the read/convert TOCTOU; residual intermediate-dir-symlink race named, not silently accepted.
  - **F-2 (MED)**: §10 — Grant chose Option A. Scan-time `.tmp` cleanup dropped entirely; stray tmps surfaced as `[WARN] stale_tmp_file` and recorded in `warnings`. Invariants 3/4 now have no tool-internal delete exception.
  - **F-3 (MED)**: §7 — VRAM partial-state cascade named, `[WARN] vram_cascade_possible` added after any base failure, per-base subprocess isolation added as Deferred, cross-referenced to the Backlog OOM-resilience item.
  - **F-4 (MED)**: §2 — `folder_paths` stub made unconditional; exact `sys.modules` preamble specified; `test_no_real_folder_paths_import` added as an S1 deliverable.
  - **F-5 (MED)**: §9 — pre-`unlink()` re-classification added; `delete_skipped_classification_changed` records a skipped delete when content was swapped between scan and delete.
  - **F-6 (MED)**: §6 — Grant chose the smaller MVP fix. Blacklist retained as interactive-only convenience under `--allow-output-outside-root`; machine caller passes `--require-output-allowlist` + `--output-allowlist-prefix` (allowlist, not blacklist). Full allowlist replacement deferred to a follow-up slice.
  - **F-7 (LOW)**: §3 — deserialization-safety assumption named: `add_safe_globals` static-grep proof hook + 5 GB per-file size cap (`size_cap_exceeded`, never opened) bounding `.pt` disk-fill.
  - **F-8 (LOW)**: §5 — `argv_redacted` redaction made uniform across all path-shaped flags; resolved paths live only in typed fields (`audit_root`, `bases.*.path`, `output_dir`, `config_path`).
  - **F-9 (LOW)**: §6 — realpath-semantics dependency named so a future non-realpath refactor doesn't silently re-open the audit-root-rename race.
  - **F-10 (LOW)**: header — risk-trigger note added: re-classifies to Red Zone when an LLM-agent / remote caller can supply `--audit-root` / `--base` / `--output-dir`; that transition needs its own ADR + Red-Zone review.
  - **F-11 (INFO)**: Alternative L (sandboxing) added to Alternatives Rejected, with re-evaluate-on-LLM-bridge note.
  - **F-12 (INFO)**: §15 — `security-auditor` (Opus) added to the S1 reviewer cadence since S1 lands the load-bearing boundary surface.
  - **F-13 (INFO)**: §5 — `warnings` array sorted lexicographically by `(file, code)` to preserve determinism (invariant 8) under any future scan-parallelism.

  Re-firing `security-auditor` (Opus) round 2 on the amended ADR. Status flips to `accepted` if round 2 returns `CLEAN`.

- **2026-05-23 (security-auditor round-2 fold-in)**: Round-2 review (appended to `docs/security/review-lora-audit-2026-05-17.md`, mirrored to vault) returned `CHANGES REQUIRED` with 9 ADDRESSED, 2 PARTIAL, 0 NOT ADDRESSED, 2 ADDRESSED-NEW-CONCERN. The headline catch: my round-1 F-1 fix did not compose with Vision invariant 7 (reuse-only) — the reused classifier (`check_lora`, `_load_state_dict`) re-opens files **by path**, so the validated fd from §6 was discarded before the read and the TOCTOU was re-introduced (F-14). F-16 found a parallel composition gap in the delete path. F-15 named a Linux-only-dep contradiction with Alternative L. F-17 named a default-deny gap on `--output-dir`. Four new findings, all folded:
  - **F-14 (MED) + F-1 (revisited):** §6 reframed per Grant's Option-B choice (2026-05-23). Honest framing: the **realpath descendancy check at scan enumeration is the authoritative containment** (portable, works everywhere); the fd-based `O_NOFOLLOW` open + per-fd `/proc/self/fd` re-check is **defense-in-depth that narrows the terminal-symlink-swap window, not a closure**. Residual same-uid TOCTOU named and accepted under the MVP same-trust-zone threat model (attacker with same-uid write under `audit_root` can already `unlink` directly). The proper closure (fd-accepting readers or sandboxing) is mandated by the F-10 risk-trigger when LLM/remote callers can supply paths — a new ADR at that transition. §6 code block re-written: `passes_scan_containment` (authoritative) + `open_no_follow` (narrowing).
  - **F-15 (LOW):** §6 names `/proc/self/fd` as a **Linux-only defense-in-depth bonus**, not the authoritative path. On non-Linux the `O_NOFOLLOW` alone is the bonus, and the realpath check (portable) remains the sole authoritative control — no Linux-only hard dependency in the authoritative path, reconciling against Alternative L's team-portable lens.
  - **F-16 (MED) + F-5 (revisited):** §9 adopts `os.unlink(name, dir_fd=parent_fd)` with parent opened `O_NOFOLLOW | O_DIRECTORY` (implementable in our own delete path, no `nodes/` change). The F-5 re-classification claim is downgraded from "closes" to "narrows" the content-substitution window; the residual is named and accepted under the same MVP threat model. Code block in §9 shows the full `safe_unlink` shape.
  - **F-17 (LOW):** §6 adds the two startup invariants: outside-root `--output-dir` with neither flag is a startup error; `--allow-output-outside-root` and `--require-output-allowlist` are mutually exclusive (prevents the machine caller from silently inheriting the weaker blacklist).

  Re-firing `security-auditor` (Opus) round 3 on the amended ADR. Status flips to `accepted` if round 3 returns `CLEAN`.

- **2026-05-23 (security-auditor round 3 → CLEAN, Status accepted)**: Round-3 review (appended to `docs/security/review-lora-audit-2026-05-17.md` and mirrored to vault `Security/Review-2026-05-17-LoRA-Audit.md`) verified all four round-2 fold-ins (F-14, F-15, F-16, F-17) are ADDRESSED, confirmed F-1..F-13 remain ADDRESSED after the round-2 edits, and found no new concerns. Verdict CLEAN. Status flipped from `proposed` to `accepted`. Implementation may now begin per §14 step 6 (`uv sync` in this worktree) and step 7 (S1 via `/change-slice`).

  Round-3 also surfaced one Backlog action (not an ADR change): the "Runtime-core cluster slice" and the future "LLM-agent bridge" Backlog items should carry an explicit cross-reference to this ADR's F-10 risk-trigger so the future Red-Zone transition cannot land without re-firing `security-auditor` at the proper discipline. Owned by the Backlog-close slice (task #11).

- **2026-06-27 (implementation closed — S1→S4 shipped, MVP complete)**: All four code-bearing slices from §15 are implemented on `lora-convert-scripts` and pass the reviewer cadence:
  - **S1** (skeleton + manifest + shape-match classify + §6 path-traversal policy) — `code-reviewer` + `security-auditor` CLEAN (`review-lora-audit-s1-2026-05-25.md`).
  - **S2** (`--dry-load` per-base full-pipeline load) — reviews CLEAN (`review-lora-audit-s2-2026-05-28.md`).
  - **S3** (`--convert` atomic sibling write + collision skip + output-dir validation) — reviews CLEAN (`review-lora-audit-s3-2026-06-02.md`).
  - **S4** (`--delete`: triple-gate + F-5 pre-unlink reclassify + F-16 dir-fd-relative unlink + preview mode) — `code-reviewer` (Opus) APPROVED, `security-auditor` (Opus) **CLEAN** (`review-lora-audit-s4-2026-06-27.md`). Both review LOWs folded in-slice: the F-8 escape-target leak (absolute outside-root path removed from the manifest `delete_skipped_containment_failed` detail; the identical pre-existing S1 `_passes_scan_containment` instances logged as TECH_DEBT 2026-06-27 for later unification) and the backstop-`Warning_` consistency gap.

  The MVP contract (adapters-only, `kind: "lora"`) is complete: a single top-level `lora_audit.json` manifest the catalog can ingest, with optional dry-load / convert / delete. **Out of scope and next in this thread:** transformer / checkpoint auditing (`kind: "transformer"` — the forward-compat hook is already in the schema) and the LoHa-conversion investigation, both per the Deferred section. This branch was rebased onto `main` on 2026-06-27 (linear; carried the `starlette → 1.3.1` CVE-2026-48710 lock bump through the merge).

- **2026-07-28 (amendment — non-diffusers key formats classify via the loader's own converter)**: The core decision does not change (shape-match against a per-family base index remains the classifier). What changes is **what counts as "the LoRA's keys"**: the shape-match now runs against the key layout the *runtime loader* will actually see, not the layout on disk.

  **Problem.** The catalog rebuild investigation (Backlog 2026-07-25) found 587 LoRAs excluded `wrong_arch`. 485 are Wan video and that exclusion is CORRECT — there is no image-plane base. But ~80 sit in *configured-base* families and are false negatives: they are **Kohya-format** files whose keys `_resolve_to_param_key` (`nodes/eric_diffusion_lora_check.py:174`) cannot resolve. That helper handles the Kohya *naming* convention (underscore-flattened, `lora_`/`unet_` prefix strip) but only as a **literal prefix strip** — it has no notion of the *structural* remap that BFL→diffusers requires (`double_blocks` → `transformer_blocks`, fused `qkv` split into `to_q`/`to_k`/`to_v`). So a `lora_unet_double_blocks_*` file scores 0% against a diffusers Flux base and falls through to `wrong_arch`. The dominant group is ~54 Flux LoRAs — **which comfyless loads successfully today**, because `load_lora_weights` converts them natively. They are working LoRAs that are simply invisible to catalog search, the MCP `list_loras` surface, and refine's planner offers. Against a searchable corpus of 241 non-excluded LoRAs, this is roughly a third more.

  **Decision.** In `_classify_lora`, when no base yields a usable verdict, **re-run the shape-match against the state dict as converted by that base family's own diffusers `LoraLoaderMixin.lora_state_dict()`** before falling through to the `find_matching_plan` convertable probe. The mixin is resolved *data-driven*, not from a hardcoded family table: read `<base>/../model_index.json` → `_class_name` → `getattr(diffusers, cls)` → walk `__mro__` for the `*LoraLoaderMixin`. This is the same `model_index.json` → `_class_name` detection the generic loader already uses (CLAUDE.md "Auto-detection"), so a new family gains conversion coverage with no audit-side edit. Verified to resolve for all 9 configured bases:

  | base | `_class_name` | mixin |
  |---|---|---|
  | `qwen-image` | `QwenImagePipeline` | `QwenImageLoraLoaderMixin` |
  | `flux` | `FluxPipeline` | `FluxLoraLoaderMixin` |
  | `flux2` | `Flux2Pipeline` | `Flux2LoraLoaderMixin` |
  | `flux2klein` | `Flux2KleinPipeline` | `Flux2LoraLoaderMixin` |
  | `krea` | `Krea2Pipeline` | `Krea2LoraLoaderMixin` |
  | `zimage` | `ZImagePipeline` | `ZImageLoraLoaderMixin` |
  | `chroma` | `ChromaPipeline` | `FluxLoraLoaderMixin` |
  | `sd3` | `StableDiffusion3Pipeline` | `SD3LoraLoaderMixin` |
  | `sdxl` | `StableDiffusionXLPipeline` | `StableDiffusionXLLoraLoaderMixin` (deferred — see below) |

  **Why delegate rather than write the mappings.** The Backlog named two options — teach the classifier the Kohya→diffusers key mappings, or add a post-mapping classification. Both were drafted as *audit-owned* mapping tables; both are rejected. A hand-written table is a second, divergent copy of logic diffusers already owns (`_convert_kohya_flux_lora_to_diffusers`, `_convert_non_diffusers_qwen_lora_to_diffusers`, …), it drifts silently on every diffusers bump, and — the disqualifying property — it lets the audit assert loadability the real loader does not deliver, or deny loadability it does. Delegating makes the audit's answer **the loader's answer by construction**, which is the only version of this classifier worth trusting. It also composes with Vision invariant 7 (reuse-only): `lora_state_dict()` is a public classmethod that accepts a plain in-memory state dict, needs no network, no weights, and no pipeline instantiation.

  **This is not a rubber stamp**, which is the property that makes it safe. Conversion is attempted, then the *existing* shape-matcher adjudicates the result; a file whose keys the converter does not recognize passes through unchanged and still fails the match. Measured on real files under the configured LoRA root:

  - **Kohya Flux** (`kdny_body_flux.safetensors`, `lora_unet_double_blocks_*`) → converts to `transformer.transformer_blocks.N.attn.to_q.…`; **494/494 layers resolve (100%)** against the `flux` base. Was `wrong_arch`.
  - **Wan negative control** (`SVI_v2_PRO_Wan2.2-I2V-A14B_LOW…`) → `FluxLoraLoaderMixin.lora_state_dict` returns it **unchanged** (`diffusion_model.blocks.*`); **0/400 resolve (0.0%)**, still `wrong_arch`. The converter declined to touch a foreign layout.
  - **Kohya SDXL** (`pony/…`) → **0%**, blocked. Deferred, see below.

  **Schema.** Additive and forward-compatible under the `audit_version: 1` contract (§"`audit_version: 1` is the contract" — new reason codes and new *optional* entry fields are explicitly not breaks; parsers ignore unknown keys per Vision invariant 12). Two additions:
  - New usable reason code `R_OK_NATIVE_CONVERT = "ok_native_convert"` — classification stays `usable` (these files load today; that is the whole finding).
  - New optional entry field `native_convert: {"mixin": str, "base": str, "verdict": {…}}` carrying the post-conversion `LoRACheckResult` dict. This preserves verdict granularity (OK / NORM_TARGETING / DIM_MISMATCH_PARTIAL survive conversion and must remain visible) without minting a combinatorial `<reason>_via_native_convert` code per existing reason.

  The pre-existing constant `R_ARCH_MISMATCH_DIFFUSERS_ONLY` is **dead** (declared at `scripts/lora_audit.py:121`, never referenced) and is deliberately *not* repurposed here: its name asserts a mismatch, the opposite of what this records. Left untouched per §4 edit-scope discipline; logged as a TECH_DEBT observation.

  **Cost.** None on the fast path. The header-only shape-match still adjudicates every file first; conversion is reached only in the branch that *already* loads the full state dict for the `find_matching_plan` probe (`_classify_lora`, no-usable-verdict path). The 241 currently-usable LoRAs never enter it.

  **Deferred — Kohya SDXL (~13 files: pony / SDXL / Illustrious).** In scope for the Backlog item, explicitly **out of scope for this amendment** (Grant's call, 2026-07-28), because it is blocked on two distinct causes rather than one:
  1. `_convert_non_diffusers_lora_to_diffusers` emits `.lora.down.weight` / `.lora.up.weight`, which are absent from `_ADAPTER_SUFFIXES` (`nodes/eric_diffusion_lora_check.py:24`, which has `.lora_down.weight`). Zero layers are extracted, so the match is 0% before naming is even considered.
  2. Deeper: the converted keys carry flat block indexing (`unet.down_blocks.4.1.proj_in`) while the SDXL base index is `down_blocks.1.attentions.0.proj_in`. Fixing (1) alone does not resolve them — diffusers performs further remapping inside the UNet loader that has not been traced.

  Shipping the flux-family fix now unblocks **ADR-041** (semantic LoRA offers), which must not be tuned over a corpus missing a third of its Flux LoRAs. SDXL becomes its own slice with the two blockers above as its starting point; filed to TECH_DEBT.

  **Proof obligations** (`test_lora_audit.py`, 197 tests today): real Kohya `lora_unet_double_blocks_*` key fixtures classifying `usable` / `ok_native_convert`; a Wan fixture that must STILL classify `wrong_arch` (the anti-rubber-stamp negative); mixin resolution driven from a synthetic `model_index.json` including the absent-class and no-mixin-in-MRO fallbacks; and an assertion that the fast path does not load a state dict for an already-usable file. **`_TOOL_VERSION` bumps `0.2.0 → 0.3.0`** (additive schema change, SemVer-ish per §"Tool-version bumps").

  Reviewer cadence: `code-reviewer` (Opus, `model: "opus"` at invocation). `scripts/lora_audit.py` is not in `scripts/git-policy/_red-zone-paths.sh` and this amendment adds no boundary surface — no `security-auditor` gate. **The fix is invisible until the operator re-runs `audit → catalog_cli build`**; that is an operator action, offered rather than performed.

- **2026-07-28 (code-reviewer fold-in — the amendment above overstated its own safety claim)**: `code-reviewer` (Opus; transcript verified 57/57 records `claude-opus-5`, no Fable fallback) returned **not approved** with 2 HIGH and 3 MEDIUM. All five folded, each reproduced before fixing. Two corrections to the entry above, which is left standing as written so the record shows what was claimed versus what the code enforced:

  - **H-1 (HIGH) — the probe corrupted the state dict it was handed.** `mixin.lora_state_dict(state_dict)` was called on the caller's live dict. diffusers mutates a dict argument in place — every one of the 25 `load_lora_weights` implementations copies first, with the comment "if a dict is passed, copy it instead of modifying it inplace"; the audit was the one caller that skipped that guard. Reproduced: `QwenImageLoraLoaderMixin.lora_state_dict` drained a 3-key dict to **0 keys**, and against the real Kohya fixture the drain is **30 of 30**. Because `qwen-image` is first in the base table, it ran before every other base's probe *and* before `find_matching_plan`, which consumes the same object — so a file that was `convertable` before this amendment would have been reported `unconvertable`. A silent demotion the amendment never claimed and no test covered. Fixed with a defensive copy at the call site; pinned by `test_native_convert_does_not_mutate_caller_state_dict`, which fails with "lost 30 of 30 keys" when the copy is removed.

  - **H-2 (HIGH) — "this is not a rubber stamp" was true only for the wholly-foreign case.** The claim above is **overstated as written**. It holds when a converter declines outright, but not under *partial* conversion: `_convert_kohya_flux2_lora_to_diffusers` drops unrecognised source keys with only a warning and returns what it understood, and `check_lora` computes `key_match_pct` over the POST-conversion dict — so everything dropped is invisible to the matcher. Reproduced: a Kohya Flux.1 LoRA probed against the flux2 converter had its `mod_lin` keys silently dropped; with the guard removed, a stub returning 1 of 10 source layers scores `key_match_pct 100.0`, verdict OK, and is promoted to `usable`. Config ordering (`flux` before `flux2`) masks this today, but ordering is user-editable and preserves consistency, not correctness. Two guards added: a `_COVERAGE_FLOOR = 0.5` on retained source adapter layers, and — on the converted path only — refusal of the `DIM_MISMATCH`-at-50% acceptance `_is_usable_verdict` allows on the direct path, since a partial conversion plus a partial dim match compounds two weak signals. `source_layers` / `converted_layers` are recorded in the `native_convert` field so the residual is auditable from the manifest.

    **Named residual, deliberately not closed:** the floor is a heuristic, not a conservation law. Converters that split fused projections (`qkv` → `to_q`/`to_k`/`to_v`) inflate the converted count, so the ratio cannot be a proof of coverage. Filed to TECH_DEBT rather than papered over.

  - **M-1 (MEDIUM) — the anti-rubber-stamp negative control did not test the mechanism.** `test_native_convert_wan_still_wrong_arch` asserted the outcome without establishing that a converter ever ran: the Wan fixture's keys (`diffusion_model.blocks.*`, no `.alpha`, no `.lora_down.weight`) trip no sniff in `FluxLoraLoaderMixin.lora_state_dict`, which returns the same object byte-identical. Deleting the entire probe left all four checks green — tautological. The evidence bullet above ("returns it **unchanged**") documents the same no-op and read it as a decline. Now the Wan case explicitly asserts and labels the passthrough, and a second control (`_write_kohya_foreign_lora` — Kohya-suffixed keys under foreign block names) engages the `is_kohya` branch, is rejected by the converter's own `Incompatible keys detected` guard, and the test asserts that engagement happened.

  - **M-2 (MEDIUM) — two escape paths violated "must never become an audit ERROR".** `_resolve_lora_mixin` was called outside the caller's try block, so anything it raised propagated to `_classify_one` and flipped the run to `classification: error` / exit 2. `json.load(...).get(...)` raises `AttributeError` on a `model_index.json` that parses to a non-dict (`[]`, `"abc"`, `3`, `null`) — not caught by `except (OSError, ValueError)` — and diffusers' `_LazyModule` raises `RuntimeError`, not `AttributeError`, when an export's backend is missing, so the `getattr` default did not cover it. Severity is scope: one malformed base config would have turned *every* fall-through file in the corpus into an error. All three hardened; four non-dict payloads added to `test_resolve_lora_mixin_fallbacks`.

  - **M-3 (MEDIUM) — declared boundary not updated.** "Out of scope (period)" still reads "Modifications to ... `nodes/eric_diffusion_lora_check.py` ... Reuse via import only", and Vision invariant 7 says the tool "does NOT refactor any module under `nodes/`" — yet the entry above claims it "composes with Vision invariant 7 (reuse-only)" while modifying that module. **Carve-out recorded here:** `check_lora` gains one optional trailing parameter, `state_dict=None`. Default `None` preserves the header-reading path exactly; all six existing callers pass by keyword, so no positional breakage. This *narrows* the reuse-only constraint to "no forking of the matcher" rather than removing it — and the alternative was worse: `tests/lora_test_harness.py` solves the same problem by writing a temp `.safetensors` and re-reading it, which would put tool-internal writes back into a scan path that §9/§10 deliberately cleared of them. One matcher, two input sources.

  **Manifest self-consistency note for catalog consumers:** a `native_convert` entry carries `classification: usable` while every value in its `verdicts_by_base` says `WRONG_ARCH` — no 0.2.0 entry could look like that. Consumers MUST key off `classification` / `reason`, never off the per-base verdicts.

  Post-fold-in proof: `test_lora_audit.py` 232 checks (197 before the amendment, 222 at first review submission), `just tests` 29/29 suites, all three pyright roots unchanged at `comfyless=13 nodes=520 pipelines=454`. Both new guards mutation-tested — reverting either turns its test red.
