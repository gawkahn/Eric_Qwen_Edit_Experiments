# Vision slice — `scripts/lora_audit.py` (LoRA audit / convert / prune tool)

**Date:** 2026-05-17
**Slice owner:** Grant (Eric_Qwen_Edit_Experiments, `lora-convert-scripts` branch in worktree `eric-lora-convert`)
**Risk:** **L3** (file writes from caller-supplied paths AND deletions; per global §12)
**Related ADRs:**
- ADR-013 (comfyless dep divergence) — establishes the `./.venv/bin/python3` test runner this slice's proof hooks invoke.
- ADR-014 (LoRA audit tool — to be drafted next from this Vision).
**AI-Disclosure:** Claude (Opus 4.7, 1M context) authored; Grant reviewed.

---

## Posture

> **Posture:** Boundary: entrypoint (new CLI) + persistence (filesystem write/delete on caller-supplied paths). Risk factors: broad impact (writes new safetensors + can delete user data), near security truth (the file-write surface is the same shape a future LLM-agent could drive, per project CLAUDE.md "Surfaces that become Red Zone on scope change").

**Dual lens (global §1):** team-portable. The classifier reuses public-shape functions from `nodes/eric_diffusion_lora_check.py` and `nodes/eric_lora_format_convert_apply.py`; the manifest schema is the contract handed to the future LoRA catalog project; any operator who forks this node pack benefits from the same audit tool. The work is not solo-defensible-only.

---

## Four signals

**Who**

Two distinct callers, with different posture implications:

- **MVP (this slice):** Solo desktop user (Grant) running `./.venv/bin/python3 scripts/lora_audit.py <dir>` interactively.
- **Final product (later slices, but the design today must not block):** The LoRA catalog application invokes this tool as a subprocess *every time it adds a newly-downloaded LoRA to its index*, parses the manifest JSON, and uses the classification to decide whether to surface or hide the file. The caller is machine, not human. Interactive prompts must not exist on any code path; exit codes must be stable; stderr is structured-or-quiet; the manifest schema must be forward-compatible so adding transformer entries (future slice) or new verdict codes (e.g. LoHa-supported) does not break a deployed catalog.
- Downstream consumer (both flows): the future LoRA catalog reads the manifest to build its "only usable LoRAs" view.
- **Implication for this slice's design:** the MVP must already satisfy the machine-caller contract (`--yes` non-interactive, stable exit codes, JSON-with-version manifest). What's deferred is *additional* tool surface (e.g. `--watch` mode for catalog post-download invocation, a `--single-file` flag, possibly a long-lived daemon mode) — not the contract itself.
- No multi-user / network / privilege boundary today. If the catalog ever runs the tool with credentials or against shared / network-mounted storage, that's a new ADR.

**Data**
- **Read:** safetensors headers (always — fast, no weight load) and weight tensors (only on `--dry-load` or `--convert`) of caller-supplied LoRA files; transformer-component safetensors headers of caller-supplied base-model dirs; optional TOML config at the user-supplied path.
- **Write (opt-in):** one top-level `lora_audit.json` at the audit root; with `--convert`, sibling files `<stem>.<target_family>.safetensors` next to each convertable source.
- **Delete (opt-in):** only files whose DELETABLE signature is confirmed (zero-byte, truncated safetensors header that fails to parse, or file extension we recognise but contents we cannot open).
- **Never executed:** `.pt` / `.bin` / `.pth` are pickle-bearing — load with `torch.load(..., weights_only=True)` exclusively (matches existing `_load_state_dict` convention in `nodes/eric_qwen_edit_lora.py:122`).

**Boundary**
- **In scope (this slice — MVP):** new `scripts/lora_audit.py` (LoRA adapters only); new `test_lora_audit.py` (repo root, matches existing `test_*.py` convention); new `docs/decisions/ADR-014-lora-audit-tool.md`; new `docs/security/review-lora-audit-2026-05-17.md`; Obsidian mirrors of the three docs; Backlog entry transition (LoRA-convert MVP shipped; LoRA-catalog Queued un-blocked for adapter-only consumption; follow-up slices named below).
- **Out of scope (this slice, but in scope for the final product — name as next slices in Backlog):**
  - **Transformer / checkpoint (base-model) auditing.** Same shape (classify USABLE / CONVERTABLE / UNCONVERTABLE / DELETABLE), different operations (format-detect via `analyze_checkpoint.py` precedent, no convert path in MVP, deletable signature = same garbage criteria). Slice splits because file sizes (GB vs MB), failure modes ("won't load as base" vs "won't load as adapter"), and conversion vocabulary differ. The manifest schema designed today MUST carry a `kind: "lora"` field so transformer entries (`kind: "transformer"`) land in the same manifest later without breaking the catalog's parser.
  - **LoHa-source conversion (investigation slice).** The apply module currently marks LoHa→standard not-implemented; MVP emits `unconvertable_loha_unsupported` rather than half-implement. The NEXT slice after MVP ships is an **investigation slice**: enumerate the LoHa LoRAs in Grant's tree (count, source toolchain, target families), prototype reconstruction (LoHa delta = `(w1_a @ w1_b) * (w2_a @ w2_b) * scale`; SVD-truncate to target rank like the LoKR path does), decide whether to ship a real conversion path or formally close as "won't fix" with documented reasoning. Manifest's `unconvertable_loha_unsupported` code must be allowed to transition to `convertable_loha` in a later manifest version without schema break.
  - **Machine-caller `--watch` / `--single-file` / daemon modes.** The catalog calls the tool once per newly-downloaded LoRA; an efficient single-file path is desirable but lands as a follow-up slice once the MVP contract is proven against Grant's full tree.
- **Out of scope (period — not this slice, not the product):**
  - Modifications to `nodes/eric_diffusion_lora_check.py`, `nodes/eric_lora_format_convert*.py`, `nodes/eric_qwen_edit_lora.py` (reuse via import only — any refactor of these belongs in the Runtime-core cluster slice in Backlog → Queued).
  - Catalog UI / consumer code (separate project; this slice ships only the manifest contract).
  - Dequantization of NF4 / fp8 quantized files (superseded by Native-quant entry in Backlog).
  - Changes to `comfyless/`, `nodes/__init__.py`, `pyproject.toml`, `requirements.txt`, `uv.lock`, or CLAUDE.md (the parallel MCP and Hunyuan worktrees own those surfaces today; pyproject / lock / CLAUDE.md are merge hotspots).
- **New deps:** none. The tool uses only the existing 11 pins (`torch`, `safetensors`, `peft`, plus the `nodes/eric_*` modules' transitive use of `numpy`, `scipy`). TOML config parsing uses stdlib `tomllib` (Python 3.11+; already required by the existing project).

**Failure**
- Path-traversal attempt on audit root, output dir, or config-file path → **fail-closed at startup** with a clear error; no scan begins.
- Symlink whose `realpath()` resolves outside the audit root → skip with loud warning; classify as `excluded_symlink_escape` in the manifest; never follow.
- Base-model dir doesn't contain a parseable transformer → skip that base, warn loud, classify all per-base verdicts for that base as `base_unavailable`; continue with remaining bases.
- Per-file classification raises → catch, log, set classification = `error`, include short reason in manifest entry; the run continues. (Per `feedback_warn_dont_block`: one bad LoRA doesn't kill the catalog scan.)
- `--dry-load` base load fails (OOM, missing weights, version mismatch) → skip dry-load for that base, fall back to shape-match for that base only, warn loud (also `feedback_warn_dont_block`).
- `--convert` output path collides with an existing file → skip with warning; never overwrite. (No `--in-place` flag in this slice; sibling-only.)
- `--delete` on a DELETABLE-classified file without `--yes` → print preview ("would delete N files"), exit 0, no I/O.
- `--delete` on any file NOT classified DELETABLE → ignored even with `--yes` (the flag does not promote files; it only acts on files the classifier already marked).
- `.pt` / `.bin` / `.pth` loaded with `weights_only=False` anywhere in this slice → invariant violation; a static grep is a proof hook (see below).

---

## Intent

A standalone CLI that classifies every LoRA file under a directory tree as **usable / convertable / unconvertable / deletable** against a configurable set of base diffusers models, and emits one top-level JSON manifest the future LoRA catalog ingests. Writes (conversion) and deletions are strictly opt-in, sibling-file-only, and bounded inside the audit root.

---

## Invariants (must always be true)

1. **Audit-root containment.** Every path the tool reads, writes, or deletes resolves (via `Path.resolve()`) to a descendant of the user-supplied audit root. Anything outside is hard-rejected before I/O.
2. **No-overwrite.** `--convert` never writes to a path that already exists; collisions skip with a warning. Source LoRA files are never modified by this tool. (No `--in-place` mode in this slice.)
3. **No-delete-without-classification.** `--delete` only removes files whose classification is `deletable`. The flag is a release valve on a classifier decision, not a path-driven `rm`.
4. **No-delete-without-confirmation.** Without `--yes`, `--delete` prints a preview and exits 0 with no I/O. With `--yes` it deletes; there is no further interactive prompt (warn-don't-block).
5. **Deserialization safety.** All `.pt` / `.bin` / `.pth` loads use `torch.load(..., weights_only=True)`. `.safetensors` headers are parsed via `safe_open` or raw struct reads. No `pickle.load`, no `torch.load(weights_only=False)`, anywhere.
6. **Symlink discipline.** Symlinks whose `realpath()` escapes the audit root are skipped and logged; the tool never crosses the boundary it was given.
7. **Reuse, don't reimplement.** Classification uses `check_lora()` from `nodes/eric_diffusion_lora_check.py` and `find_matching_plan()` / `convert_state_dict()` / `reconstruct_lokr_delta()` from `nodes/eric_lora_format_convert_apply.py`. No fork, no parallel classifier. (If imports prove painful from `scripts/`, the script prepends `sys.path.insert(0, <repo_root>)` at the top; it does NOT refactor any module under `nodes/`.)
8. **Manifest determinism.** Two runs on identical input (same files, same bases, same config) produce byte-identical `lora_audit.json` modulo the `audited_at` timestamp. (Keys sorted; file order sorted by relative path; no nondeterministic dict ordering surfaces.)
9. **Per-file fault isolation.** An exception classifying file *X* never aborts the scan; *X* is recorded as `classification: "error"` with a short reason and the loop continues.
10. **No regressions.** All eight existing test suites (850 tests per CLAUDE.md line 67) continue to pass with 0 failures against `./.venv/bin/python3`. The new suite adds to that count; it does not replace any existing suite.
11. **Worktree venv prerequisite.** Per ADR-013 §2, the test runner is the uv-managed `.venv` at the worktree root. `.venv` is gitignored and per-worktree — this worktree (`eric-lora-convert`) needs `uv sync` run once before the slice's proof hooks can execute. The Change Plan names this as step 0.
12. **Manifest is forward-compatible.** Every top-level manifest carries `audit_version: 1` (integer) and `tool_version: "..."` (string from a module constant). Every file entry carries `kind: "lora"` (MVP) — future slices add `kind: "transformer"` etc. without changing the schema. New verdict codes (e.g. promoting `unconvertable_loha_unsupported` → `convertable_loha` after the investigation slice) are additive, not renames. A deployed catalog parser written today against `audit_version: 1` continues to work against later manifests that contain additional kinds and additional verdict codes; it filters or ignores rather than crashing. (Schema-break changes bump `audit_version` and ship a migration note in the ADR Changelog.)
13. **Machine-caller contract.** Every CLI code path is non-interactive — no `input()`, no `getpass`, no `confirm` prompts. The `--yes` flag is the *only* way to authorize destructive operations; absence of `--yes` is preview-mode, never a prompt. Exit codes are documented and stable (`0` ok, `1` startup-fail, `2` per-file-errors-present). Warnings go to stderr in a line-prefixed format (`[WARN] ...`) so a subprocess wrapper can grep / line-split without parsing concerns.

---

## Failure semantics

- **Fail-closed at startup** on: path-traversal in audit root / output dir / config path; non-existent audit root; config-file parse error (TOML invalid, unknown keys); any `--base` flag whose target dir doesn't exist; any `--base` whose name collides with another `--base` name or config-file base name without an explicit override flag.
- **Warn-and-continue** on: per-file errors, per-base load failures under `--dry-load`, symlink-escape, output collisions under `--convert`, base-model header-parse failure. Each of these is recorded in the manifest with a specific reason code, so the catalog can surface it.
- **Exit code:** `0` on success (including warns), `1` on startup failure, `2` if any per-file classification raised (so a wrapping script knows the manifest contains `error` entries).
- **No partial writes.** Manifest is written atomically: write to `lora_audit.json.tmp` in the audit root, then `os.replace()`. Conversion output uses the same pattern (`<stem>.<target_family>.safetensors.tmp` → `os.replace()`) so an interrupted convert never leaves a half-written sibling.

---

## Out of scope

- Transformer / checkpoint (base-model) auditing — separate slice per the original brief's split recommendation. This slice audits **adapters only**.
- LoRA catalog code (UI, indexing, persistence) — separate project; this slice ships only the manifest contract.
- LoHa-source conversion. The apply module already marks LoHa→standard not-implemented; this slice emits `unconvertable_loha_unsupported` rather than half-implement.
- Quantized-file dequantization (NF4, fp8) — superseded by the Native-quant Backlog entry; out of scope here.
- Any refactor of `nodes/eric_*lora*.py`, `nodes/eric_diffusion_lora_check.py`, or `nodes/eric_lora_format_convert*.py`. Reuse via import only.
- Any change to `pyproject.toml`, `requirements.txt`, `uv.lock`, or CLAUDE.md (merge hotspots in this multi-worktree window).
- `comfyless/` and `nodes/__init__.py` — touched zero. The parallel MCP and Hunyuan sessions own those surfaces.
- A `--no-dry-load` / `--fast` mode that's a *separate* flag from a config knob — to keep the matrix small, the slice exposes `--dry-load` as an opt-in flag (default off, on when set; sticky-on if config sets it).
- HTTP / network exposure. CLI only. If a future caller wires this to an HTTP endpoint, that's a new ADR.
- Promotion of any function in `nodes/eric_*` to a `runtime/` shared module. That is the Runtime-core cluster slice in Backlog → Queued, sequenced before HTTP-transport MCP work.

---

## Negative cases required

Each gets a test in `test_lora_audit.py`; collectively they prove the invariants above.

1. **Path traversal (audit root).** `--audit-root /tmp/a` + a symlink under it pointing to `/etc/passwd` → symlink skipped, classification = `excluded_symlink_escape`. The destination file is never opened.
2. **Path traversal (output dir).** `--convert --output-dir /tmp/a/../../etc` → fails at startup before any conversion attempt.
3. **No-overwrite.** A pre-existing `foo.diffusers.safetensors` next to `foo.safetensors`; `--convert` runs → existing file untouched (compare sha256 before/after), entry in manifest marked `convert_skipped_collision`.
4. **No-delete-without-classification.** A perfectly-good usable LoRA under the audit root; `--delete --yes` runs → file untouched.
5. **No-delete-without-confirmation.** A zero-byte file under the audit root; `--delete` (no `--yes`) → file untouched, preview printed.
6. **Pickle safety.** A `.pt` fixture containing a state dict → loads successfully under `weights_only=True`. (No malicious-pickle test — out of scope to construct one; the static guarantee is the `weights_only=True` constant in source, checked by the grep proof hook.)
7. **Manifest determinism.** Run the tool twice on the same input; diff the manifests with `audited_at` masked → zero diff.
8. **Per-file fault isolation.** A truncated `.safetensors` (header says 1 GB, actual file is 100 bytes) co-located with good files → run completes, bad file marked `classification: "error"` with a short reason, good files classified normally, exit code = 2.
9. **Base-load failure under `--dry-load`.** A `--base` pointing to a directory that has `model_index.json` but no transformer shards → that base's verdicts fall back to shape-match with a warning, other bases proceed normally.
10. **Symlink that doesn't escape.** A symlink under the audit root pointing to another file *inside* the audit root → resolved, classified normally (positive case for the symlink rule).
11. **Machine-caller non-interactive.** Subprocess-invoke the tool with stdin closed (`subprocess.run(..., stdin=subprocess.DEVNULL)`) on a tree containing a deletable file under `--delete --yes` and a convertable file under `--convert`. Run completes with no `EOFError` / no hang / no prompt; exit code is one of the documented values; stderr lines all match `^\[(WARN|INFO|ERROR)\] `; manifest JSON parses with `audit_version == 1`, `tool_version` present, every file entry has `kind == "lora"`.
12. **Forward-compatibility smoke.** Parse a synthetic future manifest containing `kind: "transformer"` entries and a verdict code `convertable_loha` not known to MVP. The tool's manifest-validator (if any helper exists in this slice) must not crash on unknown kinds / unknown verdict codes — it ignores or passes through. (This protects MVP from regressing the contract when later slices land.)

---

## Proof hooks

Prerequisite step (one-time per worktree, before any proof-hook run):

```bash
cd /home/gawkahn/projects/ai-lab/code/eric-lora-convert && uv sync
```

Run from worktree root using the worktree's `.venv` per ADR-013 §2:

```bash
# Positive: scan a fixture tree, manifest is well-formed and matches snapshot.
./.venv/bin/python3 test_lora_audit.py

# Determinism: run twice, diff with timestamp masked.
TMPDIR=$(mktemp -d) && ./.venv/bin/python3 scripts/lora_audit.py --audit-root tests/fixtures/lora_audit_tree -o "$TMPDIR/a.json" && ./.venv/bin/python3 scripts/lora_audit.py --audit-root tests/fixtures/lora_audit_tree -o "$TMPDIR/b.json" && diff <(jq 'del(.audited_at)' "$TMPDIR/a.json") <(jq 'del(.audited_at)' "$TMPDIR/b.json")
# Expect: empty diff, exit 0.

# Regression: all eight existing suites still pass (850 tests).
./.venv/bin/python3 test_manual_loop.py && ./.venv/bin/python3 test_multistage.py && ./.venv/bin/python3 test_params_schema.py && ./.venv/bin/python3 test_cascade.py && ./.venv/bin/python3 test_machine_boundary_validator.py && ./.venv/bin/python3 test_iterate.py && ./.venv/bin/python3 test_samplers.py && ./.venv/bin/python3 test_server_robustness.py
```

Static guarantees on the new source file (greps must return zero matches):

```bash
grep -nE "torch\.load\([^)]*weights_only\s*=\s*False" scripts/lora_audit.py
grep -nE "pickle\.(load|loads)\(" scripts/lora_audit.py
```

Each numbered negative case above corresponds to one test function in `test_lora_audit.py` (`test_path_traversal_audit_root`, `test_no_overwrite`, `test_no_delete_without_classification`, etc.).

---

## Red Zone ownership

This slice has L3 risk but no §5 Red Zone element (no auth, no PII, no money, no audit trail). The Red-Zone-grade discipline applies because of §12 triggers (file writes from caller-supplied input, deletions):

- **Path / boundary policy (§12 file-write trigger):** authored by Claude, owned by Grant. Concretely: invariants 1, 2, 5, 6 above and the negative cases that prove them. Grant signs off on the ADR and the security review before code.
- **Deserialization safety (`weights_only=True`):** authored by Claude, owned by Grant. The static-grep proof hook locks this in; Grant signs off on the ADR amendment if it ever changes.
- **Conversion plan reuse:** authored by Claude, no new Red Zone surface — the existing `find_matching_plan` and `convert_state_dict` are reused untouched.

---

## Process expectations recap (lifted from the substrate brief)

1. This Vision → review by Grant → ADR-014 → `security-auditor` (Opus, `model: "opus"` at invocation per `feedback_agent_model_pin_broken`) → code in slices via `/change-slice`.
2. Each non-trivial slice runs `code-reviewer` (Opus); slices touching write/delete also run `security-auditor` (Opus).
3. AI-Disclosure trailer mandatory; pre-commit hook enforces.
4. Standalone-CLI commits use `tool:` prefix (matches `analyze_checkpoint.py` and `dequantize_nf4.py` history).
5. Push at PR-equivalent batch boundary, not per commit; ask Grant before pushing.
6. Backlog + Obsidian mirrors updated as the slice progresses; this Vision file mirrors to `Vision/Slice-LoRA-Audit.md` in the vault. Per `feedback_model_tier_delegation`, the vault-mirror copy is a Sonnet/Haiku-tier delegation when the slice closes.
