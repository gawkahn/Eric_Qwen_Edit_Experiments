# Security Review — resolve_hf_path (caller-supplied model loading)

AI-Disclosure: Claude (Opus 4.7) authored; Grant to review.

Date: 2026-04-23
Scope: `nodes/eric_diffusion_utils.py` (`resolve_hf_path` + `_is_hf_repo_id`) and all call sites
Reviewer: security-auditor subagent (Opus)

## Summary

`resolve_hf_path` is a small, fail-closed resolver that correctly defaults to `local_files_only=True`, rejects the obvious local-path shapes (`/`, `./`, `../`, drive letters), and requires a per-invocation `allow_hf_download=True` to ever touch the network. The code does not load Python from repos (`trust_remote_code` is absent from the entire codebase), does not leak HF tokens in exception messages beyond what huggingface_hub itself emits, and respects the project's "no internet during inference" invariant because `from_pretrained(..., local_files_only=True)` is still enforced by all five call sites after resolution. The primary real risk is **social-engineering-driven arbitrary repo download** via PNG-sidecar replay plus `--allow-hf-download`: there is no allowlist, so when the user opts into downloading, any `owner/repo` string in an attacker-crafted PNG or (future) LLM-agent message will be fetched and written into the HF cache; a related minor issue is that `_is_hf_repo_id` accepts `foo/..` (two non-empty parts) and sends it to `snapshot_download`, which happens to be caught today only because `huggingface_hub.validate_repo_id` rejects it.

## Call-site inventory

All confirmed `resolve_hf_path` call sites:

- `nodes/eric_diffusion_loader.py:130` — `EricDiffusionLoader.load_pipeline`, resolves `model_path` with `allow_download=allow_hf_download` (node-level BOOLEAN toggle).
- `nodes/eric_diffusion_component_loader.py:413–417` — `EricDiffusionComponentLoader.load_pipeline`, resolves `base_pipeline_path`, `transformer_path`, `vae_path`, `text_encoder_path`, `text_encoder_2_path` (five paths, one toggle).
- `nodes/eric_qwen_edit_loader.py:149` — `EricQwenEditLoader.load_pipeline`, resolves `model_path`.
- `nodes/eric_qwen_image_loader.py:204` — `EricQwenImageLoader.load_pipeline`, resolves `model_path`.
- `comfyless/generate.py:483,496–502` — `_load_pipeline`, resolves base model + 4 component paths.
- `comfyless/generate.py:652` — `generate()` re-resolves `model_path` (harmless re-entry: already-resolved local paths are returned unchanged by the `_is_hf_repo_id` check).
- `comfyless/generate.py:1223` — `_run_cli_mode` pre-resolves all path fields client-side before server delegation.

Component loaders for Qwen-Edit / Qwen-Image that do NOT call `resolve_hf_path`:
- `nodes/eric_qwen_edit_component_loader.py` — paths flow directly into `from_pretrained` / `_load_single_weights`. HF repo IDs silently fail-closed here via `local_files_only=True`, which is acceptable but inconsistent with the generic component loader.
- `nodes/eric_qwen_image_component_loader.py` — same.

Server: `comfyless/server.py` never calls `resolve_hf_path` directly. It enforces `_within(model_base)` before any path reaches `_load_pipeline`, and the client in `comfyless/generate.py:1216-1226` pre-resolves all HF IDs to local snapshot paths before sending, so the server never sees a repo ID on the wire. This is a correct layering.

## Findings

### 1. MEDIUM — `_is_hf_repo_id` accepts `foo/..` and `foo/.`

**Location:** `nodes/eric_diffusion_utils.py:88-102`

**Description:** The repo-ID heuristic rejects paths starting with `../` but does not reject the inverse case — a string whose *second* component is `..` or `.`. `_is_hf_repo_id("foo/..")` returns True because the function only checks:
- doesn't start with `/`, `./`, or `../`
- not a Windows drive letter
- has exactly 2 non-empty `/`-separated parts

So `foo/..`, `a/.`, `bar/..`, etc. all pass as "HF repo IDs" and get handed to `snapshot_download`. Today this is caught by huggingface_hub's `validate_repo_id`, which raises `HFValidationError` for these forms before any I/O. That is defense-in-depth from a dependency — not our code. If huggingface_hub's validation ever relaxes or a future code path bypasses it, these strings could reach cache-key logic or filesystem operations with a relative `..` component.

**Recommendation:** Add an explicit check that neither split-part equals `..` or `.` in `_is_hf_repo_id`:
```python
parts = path.split("/")
return (len(parts) == 2 and all(parts)
        and parts[0] not in (".", "..") and parts[1] not in (".", ".."))
```

### 2. MEDIUM — No repo allowlist — any `owner/repo` fetches when `allow_hf_download=True`

**Location:** `nodes/eric_diffusion_utils.py:105-145`; toggles at `nodes/eric_diffusion_loader.py:108-115`, `eric_diffusion_component_loader.py:386-393`, `eric_qwen_edit_loader.py:109-116`, `eric_qwen_image_loader.py:180-187`, `comfyless/generate.py:857-859`.

**Description:** When a user enables `allow_hf_download`, any string that parses as `owner/repo` will be fetched via `snapshot_download`. There is no positive allowlist (e.g. `{"Qwen/*", "black-forest-labs/*", "InstantX/*"}`) and no per-run confirmation. The threat is not remote code execution (no `trust_remote_code` anywhere in the repo, confirmed below), but rather:

- **Disk fill / cache pollution** — attacker-specified 30–50 GB model pulled to `/mnt/nvme-8tb/hf`.
- **Unexpected weight substitution** — an attacker who knows the user's prompt path can craft a PNG sidecar (`parameters` or `comfyless` tEXt chunk) whose `model` field is `attacker/looks-like-qwen-edit`; on replay with `--params <image.png>`, the user gets weights they didn't vet. Output drift, license poisoning, or intentionally compromised weight files (fine-tunes with backdoored embeddings) are all plausible.
- **Gated-repo token exposure** — `snapshot_download` automatically uses `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` from the environment. A crafted repo ID pointing at an attacker-controlled gated repo could cause huggingface_hub to attach the user's token on the request to an attacker's server. (Low likelihood: gated repos are owned by the listing user; but nothing in this code audits *what* the user's token is being sent to.)

`allow_hf_download` defaults to `False` on every call site — this is the primary mitigation. The risk surfaces when users flip it on and then later load PNGs/sidecars from untrusted sources.

**Recommendation (minimal):** Emit a loud one-line warning whenever `allow_hf_download=True` AND the resolver is about to hit the network (cache-miss branch at `nodes/eric_diffusion_utils.py:139-145`), naming the exact repo being downloaded. A user replaying a malicious sidecar will at least see `[EricDiffusion] DOWNLOADING from HuggingFace: attacker/malicious-repo` in stderr before the fetch begins. The warning in combination with the default-off toggle is proportional to the desktop-tool threat model; a positive allowlist is the correct answer only when the LLM-agent bridge lands (see Future scope).

### 3. MEDIUM — PNG sidecar replay is an undeclared input channel for model paths

**Location:** `comfyless/generate.py:122-186` (`_extract_eric_save_params`, `_load_params_from_png`, `_extract_comfyui_params`); flows into `_run_cli_mode:1134-1226`.

**Description:** The project CLAUDE.md names the `--json` bridge as a §12 trigger and a future Red Zone once LLM output drives paths. `--params <image.png>` is *already* a caller-supplied-path-into-loader channel: PIL's `info` dict is JSON-decoded and a `model` / `model_path` key flows straight into `resolve_hf_path` → `from_pretrained` with only the warning at `generate.py:1201-1212` as a check (and that warning is only emitted for local paths that don't exist — HF repo IDs pass silently). This is pre-existing, out of scope for `resolve_hf_path` itself, but it's the real-world vector that makes Finding 2 exploitable.

A malicious PNG on a public image-sharing site, when replayed by `python -m comfyless.generate --params downloaded.png --allow-hf-download`, will:
1. Load the embedded JSON.
2. Extract a `model_path` the user never typed.
3. Resolve via `resolve_hf_path` with `allow_download=True`.
4. Fetch arbitrary weights.
5. Generate with those weights, save to the user's output dir.

Note that the mitigation exists in `_run_cli_mode:1201-1212`: the warning "model path does not exist on this host" triggers for local paths, but only for local paths — HF repo IDs are skipped because the heuristic at `_is_local` returns False for `owner/repo` strings. So the user gets no warning for a repo-ID-in-PNG replay.

**Recommendation (minimal):** In `_run_cli_mode` after the existing local-path warning block (~line 1213), add a symmetric warning that fires when the `--params`-derived model value looks like an HF repo ID AND `--allow-hf-download` is set. One line, stderr-only, so users see what's being pulled in from the PNG.

### 4. LOW — `LocalEntryNotFoundError` catch assumes exact exception class (upstream API stability)

**Location:** `nodes/eric_diffusion_utils.py:122,129`

**Description:** The function catches `huggingface_hub.errors.LocalEntryNotFoundError` specifically. If a future huggingface_hub version renames, moves, or subclasses this exception, the cache-miss case falls through to the generic `Exception` catch at line 144 only when `allow_download=True` — but when `allow_download=False`, an un-caught `Exception` would propagate with whatever message huggingface_hub raises. Today that message is vetted; tomorrow it might include token material or URL query strings.

**Recommendation:** Broaden the first catch to `except Exception as e:` and keep fail-closed semantics: any exception during the cache-only probe is treated as a cache miss, which then raises `ValueError` at line 133 without leaking the underlying exception message. Change is one line.

### 5. LOW — Exception chaining exposes huggingface_hub error text, possibly including URL

**Location:** `nodes/eric_diffusion_utils.py:144-145`

**Description:** `RuntimeError(f"Failed to download {path!r} from HuggingFace: {e}") from e` concatenates the upstream exception message into the raised error. For network errors, huggingface_hub sometimes includes full URLs with query-string parameters (no credentials directly, but mirror / CDN hints and sometimes `HF_HUB_ENDPOINT` overrides). On a personal desktop this is acceptable. If the tool is ever run in a shared environment where logs are aggregated, the bare pass-through of upstream error text is the weak link.

**Recommendation:** Log the raw exception once to stderr for diagnostics, but raise only `RuntimeError(f"Failed to download {path!r} from HuggingFace (see stderr for details)")` to avoid leaking message detail to callers that may surface it to end-users / logs. Two-line change.

### 6. INFO — `trust_remote_code` is absent — verified

**Location:** Entire codebase (verified by reading call sites and `from_pretrained` / `from_single_file` invocations in `eric_diffusion_loader.py`, `eric_diffusion_component_loader.py`, `eric_qwen_edit_loader.py`, `eric_qwen_image_loader.py`, `eric_qwen_image_component_loader.py`, `eric_qwen_edit_component_loader.py`, `comfyless/generate.py`, `comfyless/server.py`, `nodes/eric_diffusion_utils.py`).

**Description:** No call site passes `trust_remote_code=True` (or any value). Diffusers defaults it to `False` and transformers defaults it to `False`. This means downloaded repos cannot execute Python code via `modeling_*.py` during load. This is the single most important invariant for Finding 2's threat model — arbitrary repo download is limited to weight-substitution, not RCE.

**Recommendation:** None. Record this invariant in the ADR that should accompany the landing of the LLM-agent bridge: if `trust_remote_code` is ever added, the entire threat model shifts, and an allowlist + sandbox become mandatory, not optional.

### 7. INFO — Symlink follow behavior — HF cache under `/mnt/nvme-8tb/hf` (bind-mounted ext4)

**Location:** `nodes/eric_diffusion_utils.py:125-128, 140-143`

**Description:** `snapshot_download` returns a path inside `HF_HUB_CACHE` (typically `.../snapshots/<rev>/`) whose contents are symlinks into `.../blobs/`. The returned `local_dir` passes the `os.path.isdir` check; downstream code calls `from_pretrained(local_dir, ...)` which follows the symlinks to the blob files. Because the cache is on the bind-mounted ext4 volume (`/mnt/nvme-8tb/hf`) and is owned by the invoking user, a symlink attack would require an attacker with write access to that volume — equivalent to local compromise, at which point the model cache is not the weakest link. The `os.path.isdir` check at lines 126 and 141 is defensive belt-and-suspenders; there is no `realpath` resolution, which would be slightly stronger but not meaningfully so on a user-owned cache.

**Recommendation:** None required for the current threat model. If the cache ever moves to a path writable by a different account (shared lab machine), add `os.path.realpath()` + containment check against `HF_HUB_CACHE`.

### 8. INFO — Server-side layering: `resolve_hf_path` is correctly NOT called inside the daemon

**Location:** `comfyless/server.py:141-157` (`_check_paths`); `comfyless/generate.py:1216-1226` (client pre-resolves).

**Description:** The server enforces `_within(model_base)` using `os.path.realpath`. If `resolve_hf_path` were called on the server, HF repo IDs would bypass the `_within` check (the resolved cache path is in `/mnt/nvme-8tb/hf`, not necessarily under `--model-base`). The current design side-steps this: the client resolves HF IDs to local paths before sending, and the server only sees real filesystem paths. This is correct. ADR-001 already documents the model-base constraint; the interaction with HF repo IDs should be appended to ADR-001's Changelog or to the server review (`docs/security/review-comfyless-server-2026-04-23.md`).

**Recommendation:** Note the HF-resolution-on-client invariant in the server's module docstring or ADR-001 Changelog (one sentence). No code change.

## Out of scope / Future scope

Flagged for future work, not blocking current merges:

- **LLM-agent bridge (`--json` → tool surface).** The project CLAUDE.md already names this as a Red Zone trigger. When model paths flow from LLM output into `resolve_hf_path`, the mitigations in Finding 2 (warning + default-off) are insufficient. An allowlist of repo-ID prefixes becomes mandatory, and `allow_hf_download` should be a server-configured policy, not a per-request flag. Write a spec + ADR + `security-auditor` review before the first commit that wires this.
- **HTTP transport for the daemon.** Project CLAUDE.md names this as Red Zone. When `--serve` grows a network listener, every finding above escalates and an HTTP-specific review is required.
- **Batch generation with caller-supplied lists.** Also named in CLAUDE.md as a §12 trigger. Path-traversal via batch items and cache-fill via many repos-per-batch both become live.
- **Symmetric `resolve_hf_path` in `eric_qwen_{edit,image}_component_loader.py`.** These two older loaders predate the HF-resolution work and silently fail-closed on repo IDs via `local_files_only=True`. Adding the resolver there would be behavior-changing; out of scope for this review but worth a TECH_DEBT entry so the inconsistency doesn't surprise someone later.

## Conclusion

`resolve_hf_path` as written is a sound fail-closed resolver for a solo desktop tool: default-off download, `trust_remote_code` absent codebase-wide, HF cache on a user-owned volume, and a server layer that correctly never sees repo IDs. It does not block merges. The three MEDIUM findings (`foo/..` in the heuristic, no allowlist, PNG-sidecar replay amplification) are each one- to five-line fixes that close the social-engineering path for PNG-driven arbitrary downloads before the LLM-agent bridge lands and turns these from MEDIUM into HIGH. Fix 1, 2, and 3 together as a single "`resolve_hf_path` hardening" slice; findings 4 and 5 (LOW) can ride in the same slice or queue.
