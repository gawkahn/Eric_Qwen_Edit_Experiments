# Security Review — ADR-027 amendment: judge rubric → recipe file

**AI-Disclosure:** `security-auditor` (Claude Fable 5, pinned per §5A) authored;
Grant reviewed. Findings folded into `comfyless/refine.py` before commit.

**Date:** 2026-07-15
**Surface:** `comfyless/refine.py` — judge system-prompt composition
(`_JUDGE_OUTPUT_CONTRACT` / `_DEFAULT_JUDGE_RUBRIC` / `compose_judge_system_prompt`
/ `load_judge_recipe`), `--judge-recipe` flag, `judge_candidate`/`refine_loop`/
`main()` threading; new `comfyless/judge_recipes/generic.toml`.
**Threat model:** §12 (LLM-facing prompt in the refinement loop); single-user
local desktop. The recipe file sits at the same trust level as the enhancer
recipes (`comfyless/recipes/*.toml`) — operator-controlled local config. The LLM
judge is the untrusted actor; `parse_verdict` + the ADR-015 resolver are the
enforcement boundary and are UNCHANGED by this amendment.
**Verdict:** clean on the core property; two hardening items folded (one MEDIUM,
one INFO).

---

## The security property holds

The claimed property — a recipe cannot weaken the safety posture — is verified.
`compose_judge_system_prompt` unconditionally appends `_JUDGE_OUTPUT_CONTRACT` on
every path that reaches `judge_candidate`: `main()` composes via
`compose_judge_system_prompt(load_judge_recipe(...))`, and every default
(`judge_candidate`, `refine_loop`) is the pre-composed `JUDGE_SYSTEM_PROMPT`
constant, itself built through `compose_judge_system_prompt`. `load_judge_recipe`
returns only the rubric string; no path lets a raw recipe reach the judge without
the contract appended, and nothing in TOML content can remove or shadow it.

Critically, this matters less than it appears: the system prompt only *asks*.
`parse_verdict` (unchanged) is the real F1 gate — closed top-level allowlist,
closed override allowlist, path/control-char rejection on LoRA names, non-finite
rejection, weight clamp — and name→path still goes only through the ADR-015
resolver (F2). A maliciously-crafted rubric can at worst degrade scoring or waste
iterations (the existing, accepted F8 reward-hack channel); it cannot mint path
or parameter authority. The contract text still exactly matches the shape
`parse_verdict` reads.

## Findings (folded)

**[MEDIUM] Missing explicitly-named recipe silently fell back to generic** —
`--judge-recipe qwen-vl` with a typo or not-yet-created file would silently run
the generic rubric, invalidating an A/B between judge models with zero signal
(every other unusable-config branch fails loud). → `load_judge_recipe` now raises
`RefineError` when an explicitly named non-`generic` recipe is absent; only the
default `generic` degrades (loud stderr WARNING → built-in `_DEFAULT_JUDGE_RUBRIC`
when generic.toml itself is missing).

**[INFO] `--judge-recipe` name joined into a path unsanitized** — `../`/absolute
values could read an arbitrary `.toml` into the judge prompt. Operator-trust today
(same as `--judge-config`), but a tripwire if the loop is ever exposed to an agent
(ADR-027 defers that surface). → bare-name guard added: a name containing `/`,
`\`, or `os.sep` is rejected. Constraint noted for the future MCP-exposure slice.

## Answers to the review questions

1. Recipe cannot weaken posture — contract appended unconditionally, lives only in
   code. 2. `parse_verdict` + ADR-015 resolver unchanged, remain the real gate;
   a rubric instructing path/extra-key emission still hits the allowlists +
   name/control-char rejection + resolver containment. 3. Traversal existed
   (operator-trust-equivalent to `--judge-config`); now guarded anyway. 4. Trust
   level correctly identical to the enhancer recipes; no new boundary. 5. Malformed
   TOML / missing / blank `system_prompt` fail closed (RefineError); the one
   silent-degrade path (missing file) is now fail-closed for explicit names + loud
   for the generic default.

## Companion code-review (Fable) — folded

False "pin test" comment corrected (constant deliberately not pinned to the file);
recipe→loop→judge threading now asserted (was the one untested link);
`UnicodeDecodeError` added to the loader's fail-closed except tuple. Fallback
chain, contract un-strippability, back-compat default, and enhance-sibling
consistency all confirmed clean. `test_refine.py` 188→206.
