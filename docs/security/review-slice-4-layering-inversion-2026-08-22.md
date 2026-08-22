# Security review — ADR-045 slice 4 (hunyuan_chain + family_defaults → comfyless/core)

AI-Disclosure: Claude (Fable 5, `security-auditor`, no model override; transcript confirms `claude-fable-5` on every turn, no fallback) authored; Grant reviewed.

## Summary

The slice is a `git mv` of two library modules into `comfyless/core/` (100% similarity, per the diff headers) plus import-path rewrites in four code callers, four test suites, and three docs. Threat model: the two touched Red Zone files (`server.py` — socket-facing daemon; `refine.py` — LLM-judge output feeding generation params) must not gain any behavior change; the moved `hunyuan_chain.py` must keep its "no filesystem search, resolve through `resolve_hf_path`, class-locked refiner" posture; and `comfyless.core` must not begin importing the CLI/daemon layer (which would let library code reach a trust boundary it shouldn't own). Each was verified against the working tree rather than the ADR prose. This is a mechanical move; it checks out.

## Coverage

Reviewed:
- The full staged diff (307 lines, rename-aware)
- `comfyless/core/hunyuan_chain.py:1-243` — full file at new path
- `comfyless/core/family_defaults.py:1-267` — full file at new path
- `comfyless/core/__init__.py` — confirms no re-exports or eager imports
- `comfyless/server.py:617-622` — `_maybe_load_refiner` call site (import + `allow_hf_download=False` pin)
- `comfyless/refine.py:49-50, 3658-3740` — import + `_overlay_family_defaults` delegation
- `test_hunyuan.py:952-966, 1231-1324` — structural pins
- `scripts/git-policy/_red-zone-paths.sh`, `TECH_DEBT.md:2761-2783`
- Repo-wide greps: old import paths; `comfyless/core/**` imports of `comfyless.*`; `nodes/` and `pipelines/` imports of comfyless outside `core`

Not reviewed:
- `test_refine.py` runtime (red on `main` pre-slice; TECH_DEBT 2026-08-22 root-cause note is in this diff but is doc-only)
- `comfyless/mcp_server.py` — not in the diff; does not reference `hunyuan_chain` directly (refiner flows through `generate`)

## Findings

**(1) server.py / refine.py — import-path-only: CONFIRMED.**
`server.py:617` is the only changed line; the call at `:618-622` still passes `allow_hf_download=False` literally. `refine.py:49-50` is the only changed line; `_overlay_family_defaults` (`:3658`) still delegates to `apply_family_defaults` (`:3701`) with the `is_eligible` gate that keeps `refiner_steps`/`refiner_cfg` out of daemon requests unrequested. No IPC, parsing, or gate logic touched.

**(2) hunyuan_chain.py posture survives the move: CONFIRMED.**
- Source contains none of `os.listdir`, `Path.glob`, `.iterdir(`, `.scandir(`, `os.scandir`; only `import sys` / `typing` at module level.
- `load_refiner_pipeline` resolves via `resolve_hf_path(refiner_path, allow_download=allow_hf_download)` (`:100`) with `allow_hf_download: bool = False` default (`:62`); `local_files_only=True` on `from_pretrained` (`:119`).
- `_REFINER_PIPELINE_CLASS_NAME = "HunyuanImageRefinerPipeline"` lock (`:33`) and the hard `ValueError` on mismatch (`:104-109`) are intact; no fallback branch.
- No LoRA loader references, no scheduler mutation (Inv 7 / Inv 8 pins).
- `test_hunyuan.py:960` now opens `comfyless/core/hunyuan_chain.py`; the old path does not exist in the tree, so the structural pins cannot silently read a stale copy. The pin is cwd-relative — pre-existing, unchanged, and the battery runs from repo root.

**(3) No new import cycle / no CLI-layer import from core: CONFIRMED.**
Grep of `comfyless/core/**` for `from|import comfyless.(server|generate|refine|mcp_server|...)` is empty; every `comfyless.*` import under `core/` targets `comfyless.core.*`. `hunyuan_chain`'s one intra-package import is lazy (`comfyless.core.eric_diffusion_utils`, `:94`). `nodes/` and `pipelines/` import nothing from comfyless outside `comfyless.core` (the Vision's corrected proof grep is empty). No stale old-path imports remain in code; the two remaining prose mentions were `TECH_DEBT.md:658` (historical, left) and `comfyless/README.md:356` (dead relative link — fixed in this slice per `code-reviewer`).

**[INFO] Red Zone path gate still stale — pre-existing, not this slice.**
Location: `scripts/git-policy/_red-zone-paths.sh:23` — names `nodes/eric_diffusion_fp8_ops.py`, which no longer exists; already registered in `TECH_DEBT.md:2761` (2026-08-22, slice 3c review) with a trigger of "next policy slice, no later than slice 6." This slice does not move any gated file (`server.py` and `refine.py` remain at their gated paths, and both were correctly caught by the gate — that is why this review exists).

No findings at CRITICAL / HIGH / MEDIUM.

## Verdict

**APPROVE.** Mechanical move, verified rather than assumed: contents of both modules unchanged, Red Zone edits are single import lines with every security-relevant parameter (`allow_hf_download=False`, refiner class lock, refine overlay eligibility gate) intact, no core→CLI layering regression, and the `test_hunyuan.py` structural pins read the live moved file.

`code-reviewer` (Fable 5, same session) independently returned APPROVED on the code with one doc finding (the README link above), folded.
