# Security review: ADR-014 LoRA audit tool (2026-05-17)

**Reviewer:** security-auditor (Opus, model=opus at invocation)
**Document under review:** docs/decisions/ADR-014-lora-audit-tool.md
**Verdict:** CHANGES REQUIRED
**Findings:** 0 HIGH, 6 MEDIUM, 4 LOW, 3 INFO

## Summary

ADR-014 specifies a CLI auditor over caller-supplied LoRA trees with two opt-in destructive surfaces (`--convert` writes, `--delete` removes) and a machine-caller contract that must hold from MVP. The design correctly identifies L3 risk, applies §12 discipline, reuses the production classifier rather than forking it (Vision invariant 7), names `weights_only=True` as load-bearing, gates deletion on a triple (classification + descendant + `--yes`), and gives the manifest forward-compatible shape (`audit_version`, `kind`, additive verdicts) so the catalog can adopt later kinds without re-parsing. The slice plan layers `security-auditor` onto S2/S3/S4 (the slices that load weights, write files, and delete files), which is the right granularity.

The headline gap is not in *what* the ADR prescribes but in *what it does not yet prescribe between resolve and use*. The path-traversal policy in §6 resolves with `Path.resolve(strict=False)` per-entry but does the open later, leaving a TOCTOU window an attacker with same-uid write access under `audit_root` (the documented threat model includes "caller-supplied trees") can exploit to make a post-resolve symlink point outside the root. The atomic-write idiom in §10 is correct in the single-writer case but the "scan-time cleanup of `*.tmp`" rule is unsafe under concurrent invocations (which the machine-caller framing makes plausible — the catalog "invokes this tool as a subprocess every time it adds a newly-downloaded LoRA"). The dry-load mechanism (§7) loads the diffusers pipeline via the same import surface that already has a known partial-state-on-OOM failure mode in the comfyless server (per the project's existing TECH_DEBT), and the ADR does not name what happens between a failed base load and the next base — `torch.cuda.empty_cache()` does not free unbacked allocations, so the next base may OOM for reasons unrelated to its own footprint. The reuse strategy in §2 also has a load-bearing latent dependency: `nodes/eric_qwen_edit_lora.py` does `import folder_paths` at module level (line 40), so any import from that file triggers it; the ADR acknowledges the stub-fallback but defers the decision to a runtime "spot-check," which is exactly the kind of "we'll figure it out when we write the code" framing that ADRs are supposed to close.

Recommendation overall: fold the six MEDIUM findings (which are concrete one-paragraph ADR amendments), re-fire round 2. None of the gaps invalidate the architectural rule or the slice plan; they tighten boundary semantics and concurrent-invocation behavior so the machine-caller contract holds in adversarial conditions, not just in the happy path.

## Findings

### F-1 (MEDIUM): TOCTOU window between `resolve()` and `open()`

**Location:** ADR §6 (path-traversal and symlink policy), particularly the `for path in audit_root.rglob("*")` loop.

**Issue:** The policy resolves each entry's realpath, checks descendancy via `relative_to(audit_root)`, then later passes the original `path` (or `real`) into `_read_safetensors_header`, `safetensors.torch.load_file`, `_load_state_dict` (which open the file). Between the resolve check and the open, a same-uid process with write access under `audit_root` can swap a symlink so the second resolution escapes. The ADR's threat model treats the audit tree as "caller-supplied" — meaning at minimum the catalog (machine caller) and at most a future LLM-agent (per project CLAUDE.md "Surfaces that become Red Zone on scope change") may seed it.

**Impact:** A malicious file under `audit_root` named `harmless.safetensors` that is a symlink resolving inside-root at check time but outside-root at read time would cause the auditor to read (and `--convert` would cause it to compute SVD over) a file outside the declared boundary. With `--delete`, the gate "classification is `deletable`" prevents arbitrary file deletion via this TOCTOU — but a `truncated_header` classification can be poisoned by mutating the file between header-read and unlink (see F-5), and `unlink()` operates on the path-as-given which may by then resolve elsewhere.

**Recommendation:** Amend §6 to specify the open-via-fd pattern: `fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)` then resolve `os.path.realpath(f"/proc/self/fd/{fd}")` and re-check descendancy on the resolved fd path; pass the fd (or `open(fd, closefd=False)`) to downstream readers. For unlink under `--delete`, use `os.unlink(path)` only after a final fd-based descendancy re-check (or accept the residual race and explicitly name it). Single-paragraph amendment in §6 stating this is the canonical pattern, with the `--delete` re-check called out explicitly.

### F-2 (MEDIUM): Atomic-write `.tmp` cleanup is unsafe under concurrent invocations

**Location:** ADR §10 ("A SIGKILL between `write_fn(tmp)` and `os.replace(tmp, target)` leaves the `.tmp` file orphaned. The tool detects and cleans up stray `*.tmp` files from prior interrupted runs at scan time…").

**Issue:** The Vision §Who explicitly names the catalog use case: "invokes this tool as a subprocess every time it adds a newly-downloaded LoRA to its index." That is a concurrent-invocation pattern. Invocation A may be mid-write to `foo.diffusers.safetensors.tmp` when invocation B (scanning the same tree) sees the `.tmp` file and unlinks it as "orphaned from a prior interrupted run." A's `os.replace()` then fails or, worse, succeeds against a now-stale fd, leaving the final target either missing or corrupt depending on filesystem semantics.

The "scan time, not subject to deletion-policy gates" framing in §10 actively undermines invariant 1 (audit-root containment in *deletion* context) and invariant 3 (no-delete-without-classification) — a `.tmp` file is not classified `deletable` but the tool deletes it anyway.

**Impact:** Corruption of a concurrent convert; silent loss of in-flight conversion work; manifest entry says `convert_output` exists but file does not. Plus: invariants 3 and 4 ("delete only on classification + `--yes`") have an exception (`.tmp` files) that the ADR introduces without naming.

**Recommendation:** Amend §10 with one of two options, your choice:
- **Option A (simplest):** drop scan-time cleanup. Stray `.tmp` files survive across runs and are surfaced as warnings (`[WARN] stale_tmp_file: <relative_path>`) — operator decides whether to remove them. The next legitimate invocation that needs to write the same target either succeeds (target doesn't exist) or sees a `convert_skipped_collision` on the target (not the tmp).
- **Option B (preserve cleanup):** use per-invocation tmp names (`target + f".tmp.{os.getpid()}.{uuid4().hex[:8]}"`) so a concurrent invocation cannot mistake another's tmp for an orphan. Scan-time cleanup then unlinks only tmp files whose embedded pid is not in `/proc` (Linux-specific; document as such).

Either way, the ADR should state explicitly: "scan-time `.tmp` cleanup is *not* subject to the delete triple-gate; this is the one documented exception to invariants 3 and 4."

### F-3 (MEDIUM): Dry-load failure does not name what happens between bases (VRAM partial-state inheritance)

**Location:** ADR §7 ("Failure mode (base load fails, e.g. OOM): that base's `dry_load_attempted: false` … continue with the next base"). Cross-references project Backlog "comfyless server has a known failed-load leaves partial state, next load OOMs" issue.

**Issue:** Diffusers `from_pretrained` partially loads weight shards into VRAM before raising on OOM or version mismatch. The exception unwinds the local pipeline reference but the partially-allocated CUDA buffers are not freed deterministically — `torch.cuda.empty_cache()` returns pooled memory to the allocator but does not free fragmented/leaked allocations. The known partial-state-OOM-cascade is already a project tech-debt item; the ADR mechanically inherits it by using the same load surface and does not name the inheritance.

**Impact:** A user who has 3 bases configured and the first one fails partway through loading may see all subsequent bases skip with `dry_load_attempted: false` due to cascading OOM, with stderr warnings that do not name the cascade cause. The manifest's verdicts then fall back to shape-match-only across the board, silently degrading the audit quality. Worst case: the partial state corrupts subsequent runs in the same Python process (less concerning here since the CLI exits at the end, but the catalog use case may keep invoking).

**Recommendation:** Amend §7 with one paragraph:
- State explicitly that base load failure may leave VRAM in a degraded state and that subsequent base loads in the same process should not be relied upon (current MVP keeps the sequential loop but emits an additional `[WARN] vram_cascade_possible: prior base load failed; subsequent base failures may be downstream effects, not the base's own problem` after any base failure).
- Cross-reference the Backlog item.
- Name as Deferred (not Out-of-scope) the option of per-base subprocess isolation (fork+exec a fresh Python process per base load) that would close the cascade. Future slice if the audit-tool runs reveal it bites.

### F-4 (MEDIUM): Reuse strategy's `folder_paths` stub is a runtime decision the ADR should close

**Location:** ADR §2 ("`load_lora_with_key_fix` and `_load_state_dict`. A spot-check during implementation confirms which import surface stays clean. **If `nodes/eric_qwen_edit_lora.py` cannot be imported without ComfyUI's `folder_paths`, the script will provide a minimal stub** … same pattern the existing tests use").

**Issue:** `nodes/eric_qwen_edit_lora.py:40` has `import folder_paths` at module level. Python's import machinery executes module top-level code on first import, so any `from nodes.eric_qwen_edit_lora import …` triggers that line. There is no spot-check that can come out the other way: the stub will be needed. Leaving the determination to implementation time is a small but real spec gap — the slice plan does not name the stub as a deliverable, and the test surface ("the script's module docstring documents this") is the wrong layer for a security-adjacent runtime substitution.

The stub itself is the right shape (`folder_paths.get_folder_paths = lambda _: []`), but if it's installed *after* an inadvertent import of something else that pulls in `nodes/eric_qwen_edit_lora` transitively, or if the script's module-load order shifts during refactor, the stub may not be in place. The pattern "set up stub *before* first import" is order-sensitive and not naturally enforced.

**Impact:** Today: the script crashes at startup with `ModuleNotFoundError: folder_paths` if the stub is misordered, which is loud and fail-closed (acceptable). Tomorrow, if the stub gets simplified or moved, a future contributor may break the order without an obvious test signal — and the script silently imports the real `folder_paths` from whatever ComfyUI install happens to be on `sys.path`, which is a path-source the audit tool has no reason to consult.

**Recommendation:** Amend §2 to state definitively: the stub IS installed (drop the "if" clause). Specify the exact preamble in the ADR (`sys.modules['folder_paths'] = types.ModuleType('folder_paths'); sys.modules['folder_paths'].get_folder_paths = lambda _: []` before the `nodes.*` imports). Add to the S1 test suite a `test_no_real_folder_paths_import` that asserts `import folder_paths` after script load returns the stub, not a real ComfyUI module — closes the order-sensitivity loop.

### F-5 (MEDIUM): Delete-gate classification can be poisoned mid-scan

**Location:** ADR §9 (triple gate: classification + descendant + `--yes`), interacting with §3 deletable signatures (`zero_byte`, `truncated_header`, `unparseable_header`, `unrecognized_extension_zero_content`).

**Issue:** The classification is computed at scan time by reading the safetensors header. The unlink happens later (after all classifications complete, presumably). Between the two, a same-uid attacker (or a careless concurrent process) can replace `harmless.safetensors` with content that *would* have classified `deletable` if scanned now, but the manifest entry — which is the gate input — already says `deletable` based on the prior scan. Conversely, a file legitimately classified `deletable` at scan time can be replaced with a valid-LoRA before unlink, and the tool deletes the now-valid file.

The first vector is benign for an attacker (they cause deletion of a file they themselves created and replaced). The second vector is the live concern: an attacker who can write under `audit_root` *and* who knows a benign `zero_byte` file is about to be deleted can race a sensitive file into its path between scan and unlink. The descendant re-check at unlink time (F-1) closes the path-substitution variant, but the content-substitution variant is open.

**Impact:** Loss of a file that wasn't `deletable` at the moment of deletion. Severity stays MEDIUM because the attack window is small (delete happens after the full scan), the attacker needs write access under `audit_root`, and the alternative (an attacker who has write access under `audit_root`) can already `unlink()` files directly.

**Recommendation:** Amend §9 with a single sentence: "Before `unlink()`, re-classify the file via the same `deletable` signature checks; if the classification has changed, skip the delete and record `delete_skipped_classification_changed` in the manifest." One classmethod call per to-be-deleted file, cheap (these are file-content cheap signatures: zero-byte stat, header struct unpack, header JSON parse). Closes the content-substitution race.

### F-6 (MEDIUM): `--output-dir` blacklist is operator-fragile and conflates defense-in-depth with the primary control

**Location:** ADR §6 ("`--allow-output-outside-root`, which itself rejects paths that contain `..` components or that resolve to a system directory blacklist `{/, /etc, /usr, /var, /tmp/.X*, ~/.ssh, ~/.gnupg}`"). §8 ("Output path: … `<output_dir>/<relative_source_dir>/<source_stem>.<target_family>.safetensors`").

**Issue:** Blacklists for system directories are operator-fragile (`/opt` is missing; `~/Library/Keychains` is missing for any future macOS use; `/proc` is missing; `/sys` is missing; what about `/dev/shm`?). More importantly, the blacklist is *opt-in to the foot-shoot*: when the user passes `--allow-output-outside-root`, they have already declared the primary descendant-of-audit-root invariant inapplicable. The blacklist then has to carry the entire weight of "but not somewhere catastrophic" — and a blacklist always has gaps.

The §6 framing says "the blacklist is a defense-in-depth; the primary check is descendant-of-audit-root" — which is the right framing when `--allow-output-outside-root` is NOT set. The framing is wrong when it IS set: in that case the blacklist *is* the only control, not defense-in-depth.

**Impact:** A user who passes `--allow-output-outside-root --output-dir /opt/somewhere` (or `/dev/shm` or `/proc`) gets no protection from the blacklist for paths the blacklist doesn't enumerate. The convert path then writes safetensors blobs into operator-unintended locations.

**Recommendation:** Amend §6 to flip the framing:
- When `--allow-output-outside-root` is NOT set: descendant-of-audit-root is the sole control, blacklist not consulted.
- When `--allow-output-outside-root` IS set: the path must (a) match an allowlist of "explicitly-allowed-outside-root" prefixes that the user supplies via `--output-allowlist-prefix /opt/lora-outputs` (repeatable), not a blacklist; (b) still reject `..` components in the supplied path; (c) emit `[WARN] writing_outside_audit_root: <path>` per file written. The blacklist is removed entirely (replaced by user-supplied allowlist) — blacklists are an anti-pattern for this exact reason.

If this is too invasive a change for the MVP, a smaller fix: state in §6 that `--allow-output-outside-root` is an MVP convenience that will be replaced by `--output-allowlist-prefix` in a follow-up slice, and ADD `--require-output-allowlist` as a flag the catalog (machine caller) must pass to require the allowlist mode. The blacklist stays as the interactive-Grant convenience; the catalog never sees it.

### F-7 (LOW): `torch.load(weights_only=True)` on torch 2.11.0 — assumption-check name needed

**Location:** ADR §3 footnote about `.pt`/`.bin`/`.pth`; cross-reference Alternative K rejection.

**Issue:** `weights_only=True` is the project convention and the right answer for torch 2.11.0 (per ADR-013 pin). It does cover the primary RCE vector by restricting unpickling to a small allowlist of "weight-shaped" classes (tensors, dicts, lists, basic numerics). It does NOT, however, cover:
- **Zip-bomb / disk-fill via `.pt`** — `.pt` files are zip containers; a maliciously-crafted file can declare gigabytes of nominal content and exhaust disk on extraction. `weights_only=True` does not enforce a size cap.
- **`torch.serialization.add_safe_globals` calls elsewhere in the process** — if any other module in the import chain has whitelisted additional classes via `add_safe_globals`, the audit tool's `weights_only=True` inherits the broader allowlist. The reuse strategy (§2) imports the entire `nodes.eric_diffusion_lora_check` + `nodes.eric_lora_format_convert_apply` + `nodes.eric_qwen_edit_lora` modules, which themselves import diffusers, transformers, peft. Whether any of those calls `add_safe_globals` at import is not audited.
- **`safetensors.torch.save_file` for output** — the contents are caller-controlled tensors. There is no RCE surface in safetensors serialization (the format is binary-only, no pickle), but a maliciously-crafted state dict could be sized to exhaust disk during write (the convert path computes SVD which can blow up rank — already covered by `target_rank=64` cap, defensive).

**Impact:** No RCE; bounded by `weights_only=True`. Disk-fill DoS via `.pt` is plausible but mitigated by the typical-LoRA size budget (a 1 GB `.pt` is suspicious in a LoRA tree and should be surfaced).

**Recommendation:** Amend §3 with one sentence stating the assumption: "`weights_only=True` is sufficient for torch 2.11.0 against RCE-class threats; no `add_safe_globals` call exists in the transitively-imported `nodes/eric_*` modules (verified by static grep, proof hook below). Disk-fill via maliciously-sized `.pt` is bounded by per-file size cap of 5 GB applied before `torch.load` is invoked; files larger than this are classified `error` with reason `size_cap_exceeded` and never opened." Add to S1 proof hooks: `grep -rn 'add_safe_globals' nodes/ scripts/` returns zero matches.

### F-8 (LOW): Manifest's `tool_invocation.config_path` is recorded but `argv_redacted` redaction is partial

**Location:** ADR §5 ("`tool_invocation.argv_redacted` excludes the `--audit-root` value … Excludes any flag value that could leak filesystem layout the catalog doesn't already know.").

**Issue:** The redaction policy is correct in shape but leaves several flag-values that *could* leak filesystem layout unredacted: `--config PATH`, `--base name=PATH` (where PATH is recorded in `bases.name.path` already but the argv also carries it), `--output-dir PATH`, `--output-allowlist-prefix PATH` (per F-6). The `config_path` field separately records the config path. The manifest is consumed by the catalog (same trust zone today; potentially a wider audience tomorrow), but if a user ever shares the manifest for diagnostic purposes ("can you look at my audit output?"), the absolute paths leak `/home/<username>`, may name a sensitive subdirectory, and reveal which bases the user has on disk.

**Impact:** Low. The threat model (solo desktop, machine caller in same trust zone) does not penalize this. Becomes more relevant if the catalog ever ships manifests to a remote service (LLM agent that "summarizes my LoRA collection" via uploaded manifest, e.g.).

**Recommendation:** Amend §5 with one sentence: "All path-shaped flag values in `argv_redacted` are replaced with the literal flag name + `<redacted>`; the resolved paths are recorded in `audit_root`, `bases.*.path`, and a new top-level `output_dir` field (when applicable). Catalog reads paths from those typed fields, not from `argv_redacted`." This makes the redaction principle uniform and removes "could leak" ambiguity.

### F-9 (LOW): No mention of TOCTOU on `Path.resolve(strict=True)` for `--audit-root` itself

**Location:** ADR §6 ("`audit_root = Path(args.audit_root).resolve(strict=True)`").

**Issue:** The audit root is resolved once at startup; subsequent operations use the resolved value as the containment boundary. If a same-uid process moves or replaces the audit root directory between startup-resolve and the scan loop, the boundary check uses a path that no longer matches what the kernel will reach via `path.resolve()` per-entry. Concretely: `audit_root` after resolve is `/home/user/loras`; attacker moves `/home/user/loras` to `/home/user/loras.bak` and creates a symlink `/home/user/loras -> /etc`. Per-entry `Path(...).resolve()` now produces paths under `/etc`, and the `relative_to(audit_root="/home/user/loras")` check passes for entries under the symlinked `/etc/something` because the kernel resolves `/home/user/loras/something` to `/etc/something` and the audit_root *string* still says `/home/user/loras`.

Actually re-reading: `Path.resolve()` returns the realpath, so `Path("/home/user/loras/foo").resolve()` would return `/etc/foo`, not a `/home/user/loras/...` path. The `relative_to(audit_root)` check would then raise `ValueError` and the entry would be skipped. So this particular race is closed by the realpath semantics — but only because of an interaction the ADR doesn't name.

**Impact:** Negligible today; the resolve-semantics happens to close the window. Worth naming so a future refactor doesn't switch to a non-realpath check and silently open it.

**Recommendation:** Amend §6 with a sentence: "Per-entry containment relies on `Path.resolve()` returning the realpath of the *current* state of the entry, not the state at audit-root-resolve time. Switching to a non-realpath check (e.g., string-prefix comparison after `os.path.normpath`) would re-open a TOCTOU window where the audit-root directory itself was renamed and symlinked mid-scan."

### F-10 (LOW): Risk classification (L3 vs L3-borderline) is defensible today, fragile tomorrow

**Location:** ADR header ("Risk: L3"); cross-reference Context (§12 triggers, not §5).

**Issue:** The L3 framing rests on the threat model "solo desktop tool; future machine-caller in same trust zone." The Vision §Who explicitly names the future LLM-agent driving paths into the tool as a Red-Zone-adjacent vector ("the file-write surface is the same shape a future LLM-agent could drive"). Project CLAUDE.md "Surfaces that become Red Zone on scope change" already names this transition: once model output drives paths, the surface becomes Red Zone.

The L3 classification is correct *for the MVP that ships only to Grant + the in-trust-zone catalog*. It is wrong the day the catalog grows an LLM-agent that decides which LoRAs to audit based on a chat instruction. The ADR's slice plan ends at S5 (Backlog close); it does not name a trigger that escalates the tool to Red Zone discipline.

**Impact:** No present-day exploit; future regression vector. Specifically: if a follow-up slice wires the audit tool's `--audit-root` to an LLM-supplied path without re-running `security-auditor` at L4/L5 discipline (allowlist instead of audit-root containment, etc.), the path-traversal posture above silently transitions from "adequate for trusted caller" to "inadequate for adversarial caller."

**Recommendation:** Amend the ADR header with a one-line "Risk-trigger note": "L3 today (file writes from caller-supplied paths, no §5 Red Zone surface). Re-classifies to Red Zone (§5-adjacent) the day any LLM-agent or remote caller can supply `--audit-root`, `--base`, or `--output-dir` values. Any slice that wires those flags from LLM output or network input MUST write a new ADR + `security-auditor` review at Red Zone discipline; ADR-014 does not authorize that transition." Mirror this to Backlog as a check.

### F-11 (INFO): Alternative-rejected list is solid; one missing alternative worth naming

**Location:** ADR Alternatives Rejected A–K.

**Issue:** None of A–K is unfaithful; each rejection rationale is load-bearing. One alternative not enumerated:

**L. Run the audit tool entirely under `bwrap`/`firejail`/`nsjail` sandbox by default.** A sandboxed audit means even a CRITICAL bug in the deserialization or path-handling code can't escape the sandbox to write outside `audit_root`. The cost is operator-fragility (`bwrap` is Linux-only; the wrap-script is a separate concern), the benefit is genuine defense-in-depth that closes whole classes of findings (F-1, F-2, F-5, F-6) at the kernel boundary instead of the application boundary.

Future readers will ask "why isn't this sandboxed?" — having the rejection documented saves the cycle.

**Recommendation:** Add Alternative L with a rejection rationale: rejected for MVP because (a) it's a deployment concern, not a tool-design concern; (b) it imposes a Linux-only operator burden the cross-platform-portability lens (project CLAUDE.md "team-portable") would not accept by default; (c) it's an additive control that can be layered later by any operator without ADR amendment. Re-evaluate if the LLM-agent bridge lands.

### F-12 (INFO): §15 slice plan reviewer cadence is correct; one cross-check on S1

**Location:** ADR §15 ("S1: Skeleton + manifest + shape-match classify … Reviewer cadence: `code-reviewer` (Opus)").

**Issue:** S1 does not run `security-auditor` per the plan. S1 ships the path-traversal policy (§6), the manifest schema (§5), and the audit-root containment invariants (Vision invariants 1, 6, 8, 12). All four are §12-trigger surfaces or Red-Zone-adjacent: the path-traversal policy is the load-bearing control for *all* subsequent slices, and S1 lands it without `security-auditor` review of the implemented code. If S1's implementation differs in subtle ways from the ADR's prescription (e.g., uses `os.path.normpath` instead of `Path.resolve(strict=False)`), the gap is not caught until S2/S3/S4 where `security-auditor` is in the loop — and by then the boundary semantics are baked in.

**Impact:** Process gap, not a code finding. S1 ships the controls the later slices rely on; reviewing the controls only when the later slices land means a control-layer regression in S1 won't surface until two commits later.

**Recommendation:** Amend §15's S1 row: "Reviewer cadence: `code-reviewer` + `security-auditor` (both Opus) — S1 implements the path-traversal + manifest + audit-root-containment surface that S2/S3/S4 depend on; reviewing only later opens a regression window. Subsequent slices' `security-auditor` invocations focus on the new write/delete surfaces, not on re-reviewing S1's boundary code."

### F-13 (INFO): Determinism property is well-specified; one nit on `verdicts_by_base` ordering

**Location:** ADR §5 ("All dict keys at every depth are sorted alphabetically via `json.dumps(..., sort_keys=True)`. Combined with relative-path sorting in `files[]`, this gives byte-identical output across runs").

**Issue:** `json.dumps(sort_keys=True)` sorts dict keys but does not sort *array* contents. The `files` array is explicitly sorted by `relative_path` (good). The `verdicts_by_base` is a dict (keys sorted by `sort_keys=True`, good). The `warnings` array is not described as sorted; if the warning-emission order is nondeterministic (e.g., per-file warnings emitted during multi-threaded scanning — the ADR is silent on whether scanning is threaded), the manifest is no longer byte-identical. The ADR doesn't claim multi-threading and the slice plan doesn't introduce it, but a future optimization that does will silently break invariant 8.

**Impact:** No present-day issue. Future-proofing concern.

**Recommendation:** Add to §5 schema rules: "`warnings` array is sorted lexicographically by `(file, code)`." This locks the determinism property for future evolution and is cheap.

## What the ADR gets right

- **§3 classification taxonomy is closed and precise.** The four top-level classifications + closed reason-code lists are exactly what a catalog parser needs; the precedence rules (deletable short-circuits before base comparison; usable wins over convertable) are unambiguous. The "no-promotion" framing for deletable is the right shape and matches Vision invariant 3.
- **Reuse strategy (§2) is honest about its costs.** The `sys.path.insert` is named as the entire integration cost; the alternative (promote to `runtime/`) is correctly deferred and the conflict with the Runtime-core cluster slice is flagged. The `folder_paths` shim approach is the right pattern even though F-4 wants it pinned more tightly.
- **`weights_only=True` enforcement is multi-layered.** Convention in source (`_load_state_dict` reuse), Alternative K rejection naming PyTorch-blessed safe mode, static-grep proof hook in the Vision. The triple coverage is the right pattern for Red-Zone-adjacent deserialization.
- **Delete triple-gate (§9) is explicit about code-order.** "All three required, in code order" closes the "gate bypass via early-return" failure mode; the no-promotion rule keeps `--yes` a release valve on classifier decisions rather than a path-driven `rm`. The exit-0-on-preview is correct machine-caller posture.
- **Manifest forward-compatibility (§5 + Vision invariant 12) is real, not aspirational.** `kind: "lora"`, `audit_version: 1`, additive verdict codes — all three are concrete contracts the catalog can program against today and continue to honor when LoHa/transformer/etc. land. The Alternatives Rejected list correctly closes "per-file sidecar JSON" (C) and "JSONL events on stdout" (I) as future-additive options.
- **Order of operations (§14) respects global §12.** Vision → ADR → security review → CLEAN flip → code, with explicit acknowledgement that the security-auditor invocation must pass `model: "opus"` per the known Claude Code 2.1.117 issue. This is the right ceremony.
- **§7 dry-load failure semantics extend Vision invariant 9 (per-file fault isolation) through the dry-load layer.** Each per-LoRA dry-load raise is caught, logged into the manifest entry, and the loop continues. This is the right shape for a several-hour run that mustn't be killed by one bad file.
- **§8 atomic-write idiom is correct in single-writer case.** `.tmp` → `os.replace()` is POSIX-atomic within a filesystem; cross-filesystem `shutil.move()` fallback is the right escape hatch. F-2 is about the *cleanup* policy, not the write policy itself.
- **§11 exit codes (0/1/2) are stable and documented**, and the stderr line-prefix format (`^\[(WARN|INFO|ERROR)\] `) gives a wrapping catalog a parse contract without committing to a JSONL event protocol prematurely (correctly deferred per Alternative I).

## What I am NOT auditing

- **Code: none exists yet.** This review is design-only. Every finding above is a request to amend the ADR's text; implementation-level findings will surface in S1/S2/S3/S4 `security-auditor` reviews (per F-12, S1's invocation is requested as an amendment).
- **Vault mirrors.** The persistence-to-vault step is operator action; the review file the user will save is what I produce here.
- **Existing test suites** (the 850 tests named in Vision invariant 10). I trust the count and did not exhaustively re-verify the test file enumeration.
- **The catalog application** (separate repo, not yet existing). The manifest contract is what I assessed; the catalog's parsing implementation is its own future review.
- **`scripts/analyze_checkpoint.py` and `scripts/dequantize_nf4.py`** — named as future-move-to-scripts candidates in Vision context but not in scope for this review.
- **ADR-013's `uv sync` posture** (already reviewed; CLEAN per `review-slice-0c-2026-05-16.md` round 2).
- **Dry-load actual behavior in real diffusers** — the integration test (`LORA_AUDIT_DRY_LOAD_E2E=1`) is gated; I am reviewing the ADR's specification of the behavior, not running the code.

## Recommendation

**Fold all six MEDIUM findings (F-1 through F-6) and the four LOW/INFO findings whose amendments are one-paragraph additions (F-7, F-8, F-9, F-10), re-fire round 2.** F-11/F-12/F-13 are optional fold-ins; Grant's discretion.

The expected round-2 verdict after fold-ins is CLEAN: the architectural rule (CLI auditor reusing the production classifier; manifest contract; triple-gated delete; atomic-write; non-interactive machine-caller contract) is sound. The amendments tighten boundary semantics (F-1, F-5), close concurrent-invocation gaps (F-2), name VRAM-cascade behavior (F-3), pin the `folder_paths` stub (F-4), replace the output blacklist with an allowlist mode (F-6), and document assumption-checks (F-7 through F-10) — none of these require structural ADR rework, all are one-paragraph-or-less amendments. The slice plan (§15) stands as-is modulo F-12's S1 amendment.

After CLEAN flip: ADR-014 Status → `accepted`, vault mirror, then S1 begins via `/change-slice` with `code-reviewer` + `security-auditor` per the amended §15.

---

## Round 2 (2026-05-19): fold-in verification

**Reviewer:** security-auditor (Opus, model=opus at invocation)
**Verdict:** CHANGES REQUIRED
**Fold-in status:** 9 ADDRESSED, 2 PARTIAL, 0 NOT ADDRESSED, 2 ADDRESSED-NEW-CONCERN

### Per-finding verification

- **F-1:** ADDRESSED, NEW CONCERN — §6 replaces resolve-then-open with `os.open(O_NOFOLLOW|O_CLOEXEC)` + `/proc/self/fd` realpath re-check, and the residual intermediate-dir-symlink race is correctly characterized ("an attacker who can swap an intermediate directory under `audit_root` can already `unlink()` directly — grants no capability the attacker lacks"). **But the fd is never actually consumed by the reused classifier** (see F-14) and `/proc/self/fd` is silently Linux-only (see F-15).
- **F-2:** ADDRESSED — §10 reflects Option A exactly: no tool-internal `.tmp` delete, stray tmps become `[WARN] stale_tmp_file` recorded in `warnings`, and the section explicitly states there is no tool-internal exception that deletes a file outside the `deletable` + `--yes` gate, so invariants 3/4 stay absolute. The unbounded-tmp-litter consequence is named and accepted.
- **F-3:** ADDRESSED — §7 names the VRAM partial-state inheritance, adds `[WARN] vram_cascade_possible`, cross-references the Backlog OOM-resilience item, lists per-base subprocess isolation as Deferred.
- **F-4:** ADDRESSED — §2 makes the `sys.modules["folder_paths"]` stub unconditional with the exact preamble before any `from nodes.*` import, adds `test_no_real_folder_paths_import`. Confirmed against source: `nodes/eric_qwen_edit_lora.py:40` is a bare module-level `import folder_paths`; pre-seeding `sys.modules` correctly short-circuits the import.
- **F-5:** ADDRESSED, NEW CONCERN — §9 adds pre-`unlink()` re-classification with `delete_skipped_classification_changed`, closing the content-substitution-then-delete-valid-file race. **But the re-classify-then-unlink sequence has its own residual TOCTOU and a path/fd consistency gap** (see F-16).
- **F-6:** PARTIALLY ADDRESSED — §6 implements the dual-caller split honestly (interactive blacklist de-rated as "NOT a security control the machine caller may rely on"; machine caller uses allowlist). **But mutual-exclusion and default-deny enforcement is under-specified** (see F-17): a machine caller that forgets `--require-output-allowlist` silently falls through to the weaker blacklist.
- **F-7:** ADDRESSED — §3 names the `weights_only=True`/torch-2.11.0 assumption, adds the `add_safe_globals` grep proof hook, bounds `.pt` disk-fill with a 5 GB pre-load size cap.
- **F-8:** ADDRESSED — §5 makes redaction uniform across all path-shaped flags, routing resolved paths only into typed fields.
- **F-9:** ADDRESSED — §6 adds the realpath-semantics-dependency paragraph.
- **F-10:** ADDRESSED — header gains the risk-trigger note re-classifying to Red Zone on LLM/remote caller path supply.
- **F-11:** ADDRESSED — Alternative L (sandbox) added with rejection rationale + re-evaluate-on-LLM-bridge note.
- **F-12:** ADDRESSED — §15 S1 row now lists `code-reviewer` + `security-auditor` (both Opus) with regression-window rationale.
- **F-13:** ADDRESSED — §5 sorts the `warnings` array by `(file, code)`.

### New concerns introduced by the amendments

- **F-14 (MEDIUM): The fd-based open from F-1 does not protect the reused classifier — the fd is opened, validated, then discarded and the file re-opened by path.** §6 says "caller reads via `os.fdopen(fd, ...)` and classifies," but the reused classifier reads by **path**: `check_lora(lora_path)` → `_read_safetensors_header(lora_path)` does `open(path, "rb")` (`eric_diffusion_lora_check.py:52, 214`); the conversion read uses `_load_state_dict(path)` → `load_file(path)` / `torch.load(path)` (`eric_qwen_edit_lora.py:122-127`). None accept an fd. Implemented flow: open fd → realpath-check fd → close/ignore fd → `check_lora(path)` re-opens `path` from scratch, re-introducing the TOCTOU F-1 set out to close. The ADR must either (a) thread `f"/proc/self/fd/{fd}"` as the path argument to each reader AND prove each accepts a procfs path (uncertain for `safetensors.load_file`), or (b) honestly downgrade F-1's claim from "closes the TOCTOU" to "narrows it." This is the F-1+F-5+F-9 composition failure: individually coherent, do not compose against the real path-based classifier.
- **F-16 (MEDIUM): F-5's re-classify-then-unlink has an unclosed TOCTOU and a path-vs-fd divergence.** §9 orders: re-classify (reads content) → fd containment re-check → `os.unlink(path)`. Three gaps: (1) content TOCTOU between re-classify read and the separate `unlink` syscall — an attacker can swap a valid file into the path in that window; (2) `unlink` operates on a *path*, re-resolving the terminal component, so a terminal-symlink swap after the fd-check targets a different inode — the correct primitive is `os.unlink(name, dir_fd=parent_fd)` with the parent opened `O_NOFOLLOW`, which §9 does not specify; (3) the re-classification re-opens by path (same F-14 root cause). The amendment overstates closure ("closes the content-substitution race") when it narrows it. MEDIUM for the same reason F-5 was (same-uid attacker can already unlink directly).
- **F-15 (LOW): `/proc/self/fd` is Linux-only and breaks the stated "team-portable" lens without naming it.** §6 hard-codes `os.path.realpath(f"/proc/self/fd/{fd}")`, absent on macOS/BSD. Vision §Posture + global §1 declare this slice team-portable; Alternative L rejected sandboxing *because* Linux-only deps fail the team-portable lens — yet F-1 introduces precisely a Linux-only hard dependency in the load-bearing containment control without naming the contradiction. Either name it as a deliberate Linux-only choice (with `F_GETPATH`/`fcntl` fallback for macOS) or reconcile against Alternative L's reasoning.
- **F-17 (LOW, folds into F-6): default-deny on output path is not stated.** §6 never states the fallthrough: an outside-root `--output-dir` with neither `--allow-output-outside-root` nor `--require-output-allowlist` must fail closed at startup, and the two flags must be mutually exclusive (else the machine caller who forgets the flag silently gets the blacklist). One sentence fixes it.

### Verdict rationale

CHANGES REQUIRED. Nine of thirteen fold-ins are clean and complete. The blocking items are real composition gaps round 1 could not have caught because the fd-based pattern text did not yet exist: F-14 shows the F-1 fd open is discarded before the reused path-based classifier re-opens the file, so the headline TOCTOU fix does not reach the read it claims to protect; F-16 shows F-5's re-classify-then-unlink narrows but does not close the content-substitution race and uses path-based unlink rather than dir-fd-relative unlink. Both block S1/S4 (load-bearing boundary controls). F-15 and F-17 are LOW. Re-fire round 3 after F-14/F-15/F-16/F-17 are folded — either by threading `/proc/self/fd/{fd}`-as-path through the readers (and proving acceptance) or by honestly downgrading F-1/F-5 closure claims to "narrows," plus the dir-fd-relative unlink for F-16.

---

## Round 3 (2026-05-23): Option-B fold-in verification

**Reviewer:** security-auditor (Opus, model=opus at invocation)
**Verdict:** CLEAN
**Fold-in status:** 4 ADDRESSED, 0 PARTIAL, 0 NOT ADDRESSED, 0 ADDRESSED-NEW-CONCERN

### Round-2 finding verification

- **F-14:** ADDRESSED — §6 reframes `passes_scan_containment` (realpath descendancy) as the authoritative containment control and demotes `open_no_follow` + `/proc/self/fd` re-check to "defense-in-depth that narrows the terminal-symlink-swap window, does NOT close it." Prose, code-block docstring ("the path-based reader takes the path"), and trailing comment ("Classifier reads by path (reuse-only). Authoritative containment was the realpath check above; this read inherits its residual TOCTOU.") together honestly name the composition with the reuse-only classifier. Residual same-uid race explicitly accepted under the MVP threat model and tied to the F-10 risk-trigger.
- **F-15:** ADDRESSED — `/proc/self/fd` gated behind `if sys.platform == "linux":` in both §6 `open_no_follow` and §9 `safe_unlink` code blocks. Authoritative control (realpath descendancy in `passes_scan_containment`) is portable; on non-Linux, `O_NOFOLLOW` alone is the narrowing bonus. Alternative-L team-portable framing explicitly reconciled in §6 prose ("No Linux-only hard dependency in the authoritative control path").
- **F-16:** ADDRESSED — §9 adopts `os.unlink(path.name, dir_fd=parent_fd)` with `parent_fd = os.open(parent, O_RDONLY | O_NOFOLLOW | O_DIRECTORY | O_CLOEXEC)`, gates parent through `passes_scan_containment(parent)`, on Linux additionally re-checks `/proc/self/fd/{parent_fd}` realpath descendancy. F-5 reclassification claim downgraded from "closes" to "narrows" in both §9 prose and inline comment; residual content-substitution TOCTOU named and accepted, with F-10 trigger for proper closure.
- **F-17:** ADDRESSED — §6 "Default-deny + mutual exclusion" subsection adds the two startup invariants: outside-root `--output-dir` with neither flag is exit 1; `--allow-output-outside-root` and `--require-output-allowlist` are mutually exclusive at startup. The smallest residual is "machine caller mistakenly passes `--allow-output-outside-root` instead of `--require-output-allowlist`" — a catalog-implementation contract bug the ADR §6 prose explicitly closes ("The catalog MUST pass this flag").

### Round-1 fold-in regression check

F-1 through F-13 all remain ADDRESSED after the round-2 edits:
- **F-1 (revisited):** Honestly downgraded from "closes" to "narrows" per F-14 fold-in — realpath check from §6 (already in round-1 amendment, now elevated to authoritative) does the actual containment work. No regression; the closure claim was the regression and it is corrected.
- **F-2..F-4, F-6..F-13:** Text unchanged. Intact.
- **F-5 (revisited):** Reclassification still present in §9 `safe_unlink`; claim downgraded from "closes" to "narrows" per F-16 fold-in. Control in place; framing now honest.
- **F-10:** Header risk-trigger note unchanged and explicitly referenced from §6 and §9 residual-acceptance paragraphs — F-10 routing is genuinely load-bearing, not aspirational.
- **F-11:** F-15 reconciliation strengthens Alternative L by quoting its team-portable rationale.

### New concerns introduced by round-2 amendments

None — Option B's framing composes cleanly. Specific traces:

1. **§6 code block correctly models the trust hierarchy.** `passes_scan_containment` documented as authoritative; `open_no_follow` documented as narrowing; fd closed in `finally` after Linux bonus check; scan loop's terminal comment names residual TOCTOU and ties to reuse-only constraint.
2. **§9 `safe_unlink` correctly uses dir-fd-relative unlink.** Trace: `passes_scan_containment(parent)` (authoritative on parent's realpath) → `os.open(parent, O_RDONLY | O_NOFOLLOW | O_DIRECTORY | O_CLOEXEC)` (no follow, directory-only) → Linux-only `/proc/self/fd/{parent_fd}` re-check → `reclassify_deletable(path)` narrowing → `os.unlink(path.name, dir_fd=parent_fd)` (relative to validated parent fd, so terminal-component symlink swap cannot redirect to different inode). `parent_fd` closed in `finally`.
3. **F-17 mutual-exclusion correctly prevents round-2 silent-fallthrough.** Catalog forgets both flags → invariant (1) exit 1; catalog passes both flags → invariant (2) exit 1; catalog passes `--allow-output-outside-root` alone → catalog contract bug ADR explicitly forbids. Closure adequate for MVP machine-caller contract.
4. **Linux-only defense-in-depth posture is honest.** §6 explicitly states no Linux-only hard dependency in authoritative path; both code blocks gate `/proc/self/fd` behind `sys.platform == "linux"`. Portable property is realpath descendancy check.
5. **Residual-accepted framing matches the F-10 risk-trigger and the chain is intact.** §6 and §9 reference "the proper closure (fd-accepting readers or sandboxing) lands when the F-10 risk-trigger fires." Header F-10 says new ADR + Red-Zone review required for LLM/remote caller transition. Alternative L re-evaluate-on-LLM-bridge note completes the loop. Chain: MVP same-trust-zone → authoritative + narrowing accepted → F-10 fires on LLM/remote caller → new ADR mandates closure (fd-accepting readers via Runtime-core slice OR sandboxing per Alternative L).

### Verdict rationale

CLEAN. Option B is correctly stated as a trade — closure traded for portability + reuse-only honesty + future-ADR-routing — and the residual is correctly bounded by the documented MVP same-trust-zone threat model with the F-10 trigger genuinely load-bearing for the future Red-Zone transition. The §6 and §9 code blocks model the trust hierarchy faithfully (authoritative + narrowing, not "two equal checks"); the Linux-only bonus is honestly fenced; the F-17 default-deny + mutual-exclusion invariants close the silent-fallthrough scenario from round 2. **Flip ADR-014 Status to `accepted`, mirror to vault, then begin S1 via `/change-slice` with `code-reviewer` + `security-auditor` (both Opus) per the §15 cadence.** The Backlog "Runtime-core cluster slice" and "LLM-agent bridge" items should both carry an explicit cross-reference to ADR-014 F-10 so the future Red-Zone transition cannot land without re-firing this review at the proper discipline.
