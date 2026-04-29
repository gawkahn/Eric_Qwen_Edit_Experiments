# Security Review — comfyless MCP server design (ADR-011)

**AI-Disclosure:** Claude (Opus 4.7, 1M context) authored as security-auditor subagent; Grant reviewed (pending sign-off).
**Date:** 2026-04-28
**Scope:** Design-time review of `docs/decisions/ADR-011-comfyless-mcp-server.md` before any code lands. No code yet.
**Reviewer:** security-auditor (Opus, model pinned at invocation per global §5A).

> **Workflow note:** The security-auditor agent has read-only tools (Read, Grep, Glob). The review content was produced by the agent and saved to disk by the parent session. This is the as-produced review; any subsequent re-reviews after ADR amendments will append below.

## Summary

The ADR is well-shaped: the four user-decided exclusions (no `allow_hf_download` exposure, no PNG `--params` extraction, `cascade_config` inline only, edit as a stub) close the largest threat-surface deltas that distinguish "MCP-driven LLM" from "human at a CLI." The thin-adapter posture (delegate to existing daemon → reuse `COMFYLESS_SCHEMA` validation, `_check_paths`, savepath template enforcement) is correct: the per-tool boundary work that already passed two rounds of review on the daemon does not need to be re-invented.

That said, there are **four MEDIUM design holes that should be folded into the ADR before code starts** and **one HIGH that needs an explicit decision**:

1. **HIGH** — Output-root declaration mechanism is undefined. ADR says outputs "must land under user-declared output roots" but does not name where the root is declared, who owns it, or what happens when none exists.
2. **MEDIUM** — `iterate` blast-radius cap is delegated to the agent's `max_iterations`, with no server-side absolute ceiling.
3. **MEDIUM** — `extract_params` JSON-only enforcement against the symlink/extension-trick threat is named in the prose but not specified as a code-level requirement.
4. **MEDIUM** — Audit log destination collides with stdio MCP transport (the existing `_log` pattern writes to stderr; that's fine, but the ADR does not say so, and it's the kind of gap that drifts in implementation).
5. **MEDIUM** — Prompt-injection threat surface in §3 omits `transformer_path`, `vae_path`, `text_encoder_path`, `text_encoder_2_path`, `cascade_config.stage_*` paths, and `cascade_config.scaffolding_repo`.

These are all **structural design decisions** that should be in the ADR, not "we'll figure it out in implementation." Each one is small text — a sentence to a paragraph in the ADR — and naming them now prevents the implementation slice from making them implicitly.

The two exclusions (HF download / PNG params) are **mostly** sufficient on the field side, but each needs a one-line code-level enforcement that the ADR should name explicitly: a hard-fail `allow_download=False` invocation that cannot be re-overridden, and a `.json`-suffix check on `extract_params` that runs after symlink resolution.

---

## Findings

### F-1 — HIGH: Output-root declaration mechanism is structurally underspecified

**Location:** ADR §3 third bullet ("Output-path containment (must land under user-declared output roots; no writes outside)").

**Risk:** The existing `comfyless/server.py` enforces output containment via a `--output-dir` CLI flag passed at `--serve` startup, validated as a real directory, and then referenced at every `_within(output_path, output_dir)` check. The MCP server's startup model is fundamentally different — it is **spawned as a child by the MCP client** (Claude Desktop, `local_agents`, Claude Code), which generally does not pass app-specific CLI flags. If the MCP server can't get its `--output-dir`, one of three things happens at impl time:

  - (a) it inherits a default like `$HOME` or `$PWD` → effectively no containment
  - (b) it reads an env var (`COMFYLESS_OUTPUT_ROOT`?) the user has to remember to set
  - (c) it requires CLI args at spawn → but MCP client config files (e.g. `claude_desktop_config.json`) do support args, this is the safe path, and not naming it bakes (a)/(b) into the implementation by default

The existing daemon's containment story falls apart silently if the output root is ambient. The ADR needs to name which one.

**Remediation (ADR-side, smallest text):** Add to §3, before the implementation order: "Output root MUST be declared as a CLI argument to `python -m comfyless.mcp_server` at MCP-client spawn time (e.g. `args: ['-m', 'comfyless.mcp_server', '--output-dir', '/abs/path']` in `claude_desktop_config.json`). The server fails to start if `--output-dir` is missing or is not an existing directory. No env-var fallback. No default. The same constraint applies to `--model-base` (path allowlist root for `model`, `loras[].path`, and component overrides), mirroring the existing daemon contract." Same hard-fail behavior the existing `_run_serve_mode` enforces.

This is also the right place to say: the MCP server's path-validation reuses the existing daemon's `_within(p, model_base)` and `_within(savepath, output_dir)` helpers verbatim — it does not re-implement them. That sentence prevents an implementer from writing a parallel and weaker check.

---

### F-2 — MEDIUM: `iterate` needs a server-side hard ceiling independent of agent input

**Location:** ADR §2 `iterate` row, and §3 (no mention of server-side ceiling).

**Risk:** ADR-008's `--max-iterations` is fail-closed for a human-driven CLI invocation: a person typed `--max-iterations 500` and the value is from their own shell history. In MCP, `max_iterations` is **agent-controlled input** like every other field. A runaway loop or a model that decides to "explore" can pass `max_iterations: 10000` (or 100000), and the cap never trips because the cap is what the agent set.

The existing daemon does not enforce an iteration ceiling — `_handle_generate` runs one generation per request. For the MCP path, one `tools/call("iterate", …)` becomes a server-side loop, and the loop bound comes from the request itself. That's exactly the pattern that turns a stuck agent into a 10,000-image folder + several days of GPU time.

**Remediation (ADR-side):** Add to §2 `iterate` row or §3 as a separate bullet: "The MCP server enforces an absolute hard ceiling on `iterate` totals that is independent of any value passed in the `max_iterations` field — a server-startup CLI flag (`--mcp-max-iterations`, default e.g. 100) that the agent cannot override. If the agent's request would expand to more than the server cap, the call fails with a structured MCP error naming both the requested total and the cap. The agent's `max_iterations` is honored as an upper bound it imposes on itself; the server cap is the floor below which the agent's request is allowed to operate."

Defense in depth: the existing `_ITERATE_CONFIRM_THRESHOLD` interactive prompt at 5 generations is a no-op in MCP (no human terminal). The server cap is what replaces it.

---

### F-3 — MEDIUM: `extract_params` `.json`-only enforcement needs to bind on the resolved path, not the surface name

**Location:** ADR §3 second user-decided exclusion ("extract_params accepts JSON sidecar paths only and rejects non-`.json` extensions").

**Risk:** The user's exclusion correctly closes the PNG-byte-parsing channel by not invoking `_load_params_from_png` from MCP. But the implementation phrasing "rejects non-`.json` extensions" is checking the surface name of the path. Two adjacent issues:

  1. **Symlink trick.** Agent passes `/output/legit.json` which is a symlink to `/output/evil.png`. If the implementation checks `path.lower().endswith(".json")` *before* `os.path.realpath`, the surface check passes and the resolver opens what's really a PNG. `_load_sidecar` calls `json.load(open(path))`, which will fail loudly on a PNG (good — JSON parse error), but the path crossed the check on the wrong basis. Lower-risk than the active `_load_params_from_png` exposure but not clean.
  2. **Path containment.** The ADR doesn't say `extract_params` paths must live under `--output-dir` or any other allowlisted root. An agent could call `extract_params("/etc/anything.json")`. Most files won't parse as JSON or as a valid params blob, but the read happens — small information leak (existence/contents of arbitrary `.json` files the daemon's UID can read).

**Remediation (ADR-side):** Tighten the §3 wording to: "`extract_params` does both checks: (a) the resolved path (after `os.path.realpath` and symlink resolution) must end in `.json`, and (b) the resolved path must be within `--output-dir` (the same containment root that `generate` writes to). Same-directory containment is the natural model — the agent extracts params from sidecars the agent's own prior `generate` calls produced." This also closes a future class of "agent reads attacker-planted `.json` from `/tmp`" without needing further ADR amendments.

---

### F-4 — MEDIUM: Audit-log destination is undefined; existing pattern won't survive stdio transport without an explicit choice

**Location:** ADR §3 third bullet ("Audit trail (every tool invocation logged with full input — leans on the existing PathError audit log pattern)").

**Risk:** I read `comfyless/server.py` to verify the existing pattern. The "audit log" today is exactly one line: `_log(f"PathError: {err} req={redacted!r}")` at line 321, where `_log` is `print(..., file=sys.stderr, flush=True)`. There is no audit log file. There is no rotation. There is no structured format. There is no record at all for *successful* `generate` calls. The pattern the ADR leans on is **a single stderr line on rejection, nothing on success**.

For an MCP server on stdio, **stderr is fine and is the right choice** (stdout is reserved for the JSON-RPC protocol; logging there breaks the wire), but the ADR needs to say so explicitly because:

  1. An implementer could mistakenly log to stdout (corrupts MCP wire protocol).
  2. The ADR claims "every tool invocation logged with full input" — the existing pattern logs only on PathError, not on success. Either the ADR's claim is wrong, or the implementation needs new code that doesn't exist today. Naming this gap up front is the difference between "lean on existing pattern" (true for rejections, false for successes) and "extend existing pattern" (the actual work).
  3. The `redacted = {k: v for k, v in req.items() if k != "prompt"}` line at server.py:320 is the existing prompt-redaction discipline. The ADR should adopt the same redaction for MCP audit lines and say so explicitly — otherwise an LLM-supplied prompt gets dumped verbatim to whatever process captures the MCP server's stderr (which on Claude Desktop is a log file the user may share with Anthropic for support). That is borderline PII-in-logs depending on what users prompt with.

**Remediation (ADR-side):** Add to §3 third bullet: "All audit lines write to stderr (stdout is reserved for the MCP JSON-RPC frame). One line per tool invocation: tool name, redacted input (prompts and negative_prompts dropped, paths and params kept), success/error class, elapsed seconds. Successful `generate` invocations are logged in addition to rejections — the existing pattern logs only rejections; this is an additive requirement on the new code path. No PII (prompt text, negative prompt text) appears in the audit line; the prompt is recoverable from the per-output sidecar JSON the user controls, not from server logs."

---

### F-5 — MEDIUM: Prompt-injection field list in §3 is incomplete

**Location:** ADR §3 fourth bullet ("Prompt-injection threat surface (model-output-derived field flowing into `model`, `lora.path`, `output_path`, savepath template)").

**Risk:** The four fields named are correct but not exhaustive. Reading `_GENERATE_OPTIONAL` in server.py and the cascade schema in `comfyless/cascade.py`, the agent can also drive:

  - `transformer_path`, `vae_path`, `text_encoder_path`, `text_encoder_2_path` — each gets `_within(p, model_base)`-validated and `resolve_hf_path`-resolved. Same threat class as `model`. The 2026-04-23 hardening review's INFO finding 3 ("widen the PNG warning to component paths") flagged exactly this when the LLM-agent surface lands.
  - `cascade_config.stage_c`, `cascade_config.stage_b`, `cascade_config.stage_a` — read in `cascade.py:_load_unet` / `_load_stage_a`. These are loaded via `StableCascadeUNet.from_pretrained(path, …)` and `from_single_file(path, …)`. The cascade module today does **not** apply `_check_paths`-style allowlisting to these fields — the daemon never sees them because the existing `--json` and CLI paths run cascade in-process. When MCP exposes `generate(model="stablecascade", cascade_config={…})`, those three path fields enter the LLM-controlled trust boundary for the first time.
  - `cascade_config.scaffolding_repo` — passed to `resolve_hf_path` with whatever `allow_download` flag the caller provides. The user-decided exclusion correctly forces `allow_download=False` for MCP; the field still needs path-allowlist enforcement (otherwise an agent passes `/etc` and gets a containment failure that may or may not be informative).
  - `negative_prompt` — text content, low risk in itself, but if the audit log captures it without redaction it joins the PII-in-logs concern from F-4.

**Remediation (ADR-side):** Update §3 fourth bullet to: "Prompt-injection threat surface — every model-output-derived field flowing into a path or weight reference: `model`, `transformer_path`, `vae_path`, `text_encoder_path`, `text_encoder_2_path`, `loras[].path`, `output_path` / `savepath` template, and inside `cascade_config`: `stage_c`, `stage_b`, `stage_a`, `scaffolding_repo`. Each goes through `_check_paths`-equivalent allowlisting against `--model-base` (loadable weights) or `--output-dir` (writes). The cascade fields specifically — currently in-process and never validated against a `model_base` — gain the same allowlist enforcement as the standard `model` field when reached via MCP."

---

### F-6 — MEDIUM: `allow_hf_download=False` enforcement needs to be unbypassable, not just "not exposed"

**Location:** ADR §3 first user-decided exclusion.

**Risk:** The exclusion says "`allow_hf_download` is NOT exposed on the MCP surface (no flag, no field). … `resolve_hf_path` is invoked with `allow_download=False`." This is correct in spirit. The implementation-level question is whether the MCP server's invocation of `resolve_hf_path(path, allow_download=False)` is **the only** path through `resolve_hf_path` from the MCP request. Looking at `comfyless/generate.py:_load_pipeline` (line 688): the function takes `allow_hf_download` as a kwarg and passes it through to **five** `resolve_hf_path` calls. The MCP server adapter must be careful that **no** kwarg propagation upstream from the MCP entry point can flip this to True. A future bug where the MCP server's `generate` adapter accepts a future-added parameter and forwards it to `_load_pipeline` re-opens the channel.

The cleanest enforcement is structural: the MCP-server entry point hard-codes `allow_hf_download=False` at every `resolve_hf_path` and `_load_pipeline` boundary, and a unit test asserts that `allow_hf_download=True` cannot be reached from any MCP code path. That's stronger than "we don't expose the field."

**Remediation (ADR-side):** Add to §3 first exclusion: "Implementation enforcement: the MCP-server adapter hard-codes `allow_hf_download=False` at every `resolve_hf_path` and `_load_pipeline` call site. There is no `**kwargs` passthrough from the MCP request to these calls. A regression test asserts that no MCP-reachable code path can set `allow_hf_download=True`."

This is preventive, not currently broken — but it's the kind of regression that happens once the codebase grows.

---

### F-7 — LOW: Stub `edit` tool advertising `NotImplementedError` is OK, with a small caveat

**Location:** ADR §2 `edit` row.

**Risk:** Advertising a tool that always returns an error is **not** a security risk in itself — MCP clients are designed to handle tool-call failures, and an LLM seeing "edit returned NotImplementedError" simply doesn't call it again in the same session. The minor concern is that if the stub's input schema includes `image_path` (declared per the ADR), a malicious or buggy agent could probe by calling `edit({"image_path": "/etc/passwd", ...})` and the stub's input-validation behavior matters: does the schema validator examine the path before returning the `NotImplementedError`, or does it short-circuit?

If the stub's first action is to validate the input against the schema (which is what the MCP SDK's tool-call wrapper does by default) and only then return `NotImplementedError`, the `image_path` is checked and rejected for non-allowlist values *before* the `NotImplementedError` fires. This is the correct order and produces the right behavior — the agent can't use the stub as a side-channel to probe path existence under the user's umask. If the stub short-circuits to `NotImplementedError` before validation, the schema is purely advisory and the tool slot becomes a no-op probe surface (low harm but inconsistent).

**Remediation (ADR-side):** Add to §2 `edit` row: "The stub validates input against the schema (via the MCP SDK's standard tool-call wrapper) before returning `NotImplementedError`. Schema validation is not skipped just because the implementation is a stub." One-line clarification.

---

### F-8 — LOW: Cross-tool composition (`list_loras` → `generate`) is safe under the existing checks

**Reasoning:** The chain `list_loras → pick a path → generate.loras[].path` is safe because:

  1. `list_loras` returns paths it discovered by scanning under `--model-base` (the same allowlist root). Anything it returns is by construction within `model_base`.
  2. `generate.loras[].path` is re-validated by `_check_paths` against the same `model_base` at the daemon boundary. The agent cannot append `/../../etc/foo` to a returned path and have it sneak through — the `_within` check after `realpath` rejects it.

So the path is enough; a "verified hash" marker would be defense-in-depth but is not load-bearing. The existing structural enforcement (allowlist root + `_within` realpath check) covers the chain. **No ADR change needed.**

The chain `extract_params → generate` is safe **conditional on F-3 being addressed**: once `extract_params` is bounded to read sidecars within `--output-dir`, the returned blob can only reflect what the agent's own prior `generate` produced, and the blob's `model` / `lora.path` fields will get re-validated at the `generate` boundary anyway. Without F-3, the blob can contain arbitrary attacker-controlled values from a `.json` the agent was tricked into reading; even so, the `generate` boundary validation catches them. So this chain is "safe under existing checks" with F-3 being defense-in-depth on the read side.

---

### F-9 — LOW: stdio same-uid threat model is sufficient; PPID enforcement is overkill

**Reasoning:** The ADR's claim that stdio inherits the same-uid threat model from the parent-process spawn is correct. A different-uid attacker cannot spawn the MCP server as a child of an MCP client running as the user — they would need same-uid capability already, which is the threat model the existing Unix-socket daemon defends against. PPID enforcement (refuse to run if PPID isn't an expected parent) would catch a same-uid attacker who execs the MCP server directly, but:

  - An attacker who already has same-uid execution can also `LD_PRELOAD` or `ptrace` the legitimate MCP server, so PPID checks are not load-bearing.
  - The MCP transport ships JSON-RPC frames over the inherited stdin/stdout. An attacker with same-uid shell access already has full control over what messages the server sees regardless of PPID.

**No ADR change needed.** The same-uid threat model is the correct boundary.

---

### F-10 — INFO: MCP Python SDK transitive dep footprint

**Reasoning:** The official MCP Python SDK (`mcp` on PyPI, maintained by Anthropic) is a slim package whose transitive deps are roughly: `pydantic`, `pydantic-core`, `anyio`, `sniffio`, `typing-extensions`, `httpx`/`httpx-sse` (for the SSE transport, optional), `starlette` + `uvicorn` (for the HTTP transport, optional), and `python-dotenv` (optional). For a stdio-only deployment, the load-bearing transitive set is `pydantic` (already widely used in the diffusers ecosystem), `anyio`, `sniffio`, and `typing-extensions` — all small, all well-known, all already likely in the dependency closure of `transformers` / `diffusers`.

The HTTP/SSE transport pulls in `starlette` and `uvicorn` if the install extras are enabled; the ADR specifies stdio-only for v1, so the `mcp[stdio]` extra (or the bare `mcp` package) is what should be pinned. Worth naming explicitly in the dep-bump slice so the install does not silently pull `uvicorn` into a desktop-tool dep tree.

**Remediation (ADR-side, optional):** Add a sentence to §7: "Install via the bare `mcp` package or `mcp[stdio]` extra; explicitly do NOT install `mcp[http]` or `mcp[all]` until/unless the HTTP transport ADR lands. This keeps `starlette`, `uvicorn`, and the ASGI stack out of the desktop tool's dep closure."

This is INFO not MEDIUM because the existing dep-bump discipline (lockstep `pyproject.toml` + `requirements.txt` + `uv.lock`, exact pins, audit) already catches a surprise transitive in the lockfile diff. The ADR sentence is preventive.

---

## Findings tally (round 1, 2026-04-28)

- **HIGH: 1** (F-1 output-root mechanism)
- **MEDIUM: 5** (F-2 iterate cap, F-3 extract_params binding, F-4 audit log, F-5 prompt-injection field list, F-6 allow_hf_download structural enforcement)
- **LOW: 3** (F-7 stub validation order, F-8 cross-tool chain note, F-9 PPID overkill)
- **INFO: 1** (F-10 SDK extras pin)

## Verdict (round 1)

**CHANGES REQUIRED.** The design is fundamentally sound — the thin-adapter model, the four user-decided exclusions, and the §3 Red Zone discipline are all the right structural choices. But four design decisions are currently underspecified in ways that will be made implicitly by the implementer if not named in the ADR first:

1. **F-1 (HIGH)** — Where does `--output-dir` come from? CLI arg at MCP-client spawn, hard-fail if missing. Same for `--model-base`. (One paragraph in §3.)
2. **F-2 (MEDIUM)** — `iterate` needs a server-side absolute cap independent of agent input. (One sentence in §2 or §3.)
3. **F-3 (MEDIUM)** — `extract_params` must check the resolved (post-realpath) path for `.json` suffix AND must require the path to be within `--output-dir`. (One sentence in §3.)
4. **F-4 (MEDIUM)** — Audit log writes to stderr, includes successful invocations, redacts prompts. The existing pattern only covers rejections — naming it as additive prevents the implementer from thinking it's already done. (One sentence in §3.)
5. **F-5 (MEDIUM)** — Prompt-injection field list in §3 should name `transformer_path`, `vae_path`, `text_encoder_path`, `text_encoder_2_path`, and the cascade `stage_*` + `scaffolding_repo` fields. (Updated bullet in §3.)
6. **F-6 (MEDIUM)** — `allow_hf_download=False` is enforced at every `resolve_hf_path` and `_load_pipeline` boundary in MCP code paths, with a regression test. (One sentence in §3 first exclusion.)

These are six small text additions. None changes the overall design; all are "the design says X, the ADR text needs to say X explicitly." After they are folded in and the user confirms, the ADR is ready to flip from `proposed` to `accepted` and the implementation can begin per the §3 ordered slice plan.

I would not flip `Status: accepted` on the as-written ADR. The HIGH on output-root mechanism is the gate — without it, the implementer has to make a structural security decision (env var vs CLI arg vs default) that should be the ADR's call, not the implementer's.

---

**Files referenced:**

- `/home/gawkahn/projects/ai-lab/code/Eric_Qwen_Edit_Experiments/docs/decisions/ADR-011-comfyless-mcp-server.md`
- `/home/gawkahn/projects/ai-lab/code/Eric_Qwen_Edit_Experiments/comfyless/server.py`
- `/home/gawkahn/projects/ai-lab/code/Eric_Qwen_Edit_Experiments/comfyless/generate.py`
- `/home/gawkahn/projects/ai-lab/code/Eric_Qwen_Edit_Experiments/comfyless/cascade.py`
- `/home/gawkahn/projects/ai-lab/code/Eric_Qwen_Edit_Experiments/nodes/eric_diffusion_utils.py`
- `/home/gawkahn/projects/ai-lab/code/Eric_Qwen_Edit_Experiments/docs/security/review-comfyless-server-hardening-2026-04-23.md`
- `/home/gawkahn/projects/ai-lab/code/Eric_Qwen_Edit_Experiments/docs/security/review-resolve-hf-path-hardening-2026-04-23.md`

---

## Round 2 (2026-04-28)

**Reviewer:** security-auditor (Opus, model pinned at invocation per global §5A).
**Scope:** Verify the round-1 fold-in additions in ADR-011 against the original findings.

### Finding-by-finding verification

- **F-1 (HIGH) — ADDRESSED.** §3a now mandates `--output-dir` and `--model-base` as required spawn-time CLI args with hard-fail on missing / non-existent / non-directory, no env-var fallback, no default, no `$HOME` inheritance. The text explicitly states "The MCP server reuses the existing daemon's `_within(p, root)` and path-validation helpers verbatim — it does not re-implement them." This is the structural commitment F-1 asked for.

- **F-2 (MEDIUM) — ADDRESSED.** §3c introduces `--mcp-max-iterations` (default 100) as a spawn-time CLI flag, explicitly independent of the agent's `max_iterations` field. The expansion check happens "after Cartesian product, before `--limit` truncation," and overage produces a structured MCP error naming both requested total and cap. The framing "agent's `max_iterations` is honored as an upper bound it imposes on itself; the server cap is the floor below which the agent's request is allowed to operate" matches the round-1 recommendation verbatim.

- **F-3 (MEDIUM) — ADDRESSED.** §3 second exclusion now specifies the two checks in the correct order: (1) resolved path post-realpath/symlink resolution ends in `.json` with explicit reasoning ("checking the surface name first would let a `legit.json → evil.png` symlink slip past"), (2) resolved path within `--output-dir`. Same-directory containment rationale stated. Both halves of the F-3 ask landed.

- **F-4 (MEDIUM) — ADDRESSED.** §3b audit-trail bullet covers all four asks: stderr only (with explicit "stdout is reserved for the MCP JSON-RPC frame; logging there would corrupt the protocol"), success + rejection logged, prompts and negative_prompts dropped, success-path audit named as "additive new code, not a free inheritance" beyond the existing PathError-only pattern. The PII-in-logs framing is explicit ("avoids PII-in-logs when the MCP client captures the server's stderr").

- **F-5 (MEDIUM) — ADDRESSED.** §3b prompt-injection list now names: `model`, `transformer_path`, `vae_path`, `text_encoder_path`, `text_encoder_2_path`, `loras[].path`, `output_path`, `savepath` template, and inside `cascade_config`: `stage_c`, `stage_b`, `stage_a`, `scaffolding_repo`. The line "the cascade `stage_*` and `scaffolding_repo` fields specifically — currently in-process and never validated against a `model_base` — gain the same allowlist enforcement as the standard `model` field when reached via MCP" closes the in-process trust boundary issue F-5 raised.

- **F-6 (MEDIUM) — ADDRESSED.** §3 first exclusion now requires hard-coded `allow_hf_download=False` at every `resolve_hf_path` and `_load_pipeline` call site, no `**kwargs` passthrough, with a concretely-described regression test (monkey-patch `resolve_hf_path` to record `allow_download` values, exercise every tool, assert every recorded value is `False`). Stronger than the round-1 ask — the test description is operationally specific.

- **F-7 (LOW) — ADDRESSED.** §2 `edit` row now contains: "The stub validates input against the schema (via the MCP SDK's standard tool-call wrapper) before returning `NotImplementedError` — schema validation is not skipped just because the implementation is a stub, so the stub cannot be used as a side-channel to probe path existence." Both the validation order and the rationale are present.

- **F-8 (LOW) — N/A.** Round-1 acknowledged no ADR change needed.

- **F-9 (LOW) — N/A.** Round-1 acknowledged no ADR change needed.

- **F-10 (INFO) — ADDRESSED.** §7 now states: "Install via the bare `mcp` package or the `mcp[stdio]` extra. **Explicitly do NOT install `mcp[http]` or `mcp[all]`**" with the rationale that those extras pull `starlette` + `uvicorn` + the ASGI stack into the dep closure. Matches the round-1 remediation verbatim.

### New round-2 findings

- **F-11 (round 2) — INFO: Duplicate paragraph artifact in §3d.** Lines 119-121 (in the pre-cleanup revision) repeated content already present in §3d, leftover from the round-1 fold-in restructuring. Not a security risk — both copies said the same thing — but the kind of artifact that signals incomplete editing of an authoritative document. Recommended deletion as documentation-hygiene; not a security blocker, does not gate `accepted`. **Resolved 2026-04-28** in the same edit pass that flipped Status to `accepted`; duplicate lines deleted.

### Findings tally (round 2)

- **HIGH: 0 unaddressed** (F-1 closed)
- **MEDIUM: 0 unaddressed** (F-2 through F-6 closed)
- **LOW: 0 new** (F-7 closed; F-8, F-9 N/A)
- **INFO: 1 new, 1 resolved** (F-11 duplicate-paragraph artifact, fixed in same edit)

### Verdict (round 2)

**CLEAN.** All round-1 HIGH and MEDIUM findings are addressed in the amended ADR text. The fold-ins are not weaker than the round-1 recommendations — in several places (F-3 ordering rationale, F-6 regression test specificity, F-4 PII framing) they are stronger or more operationally concrete. The one new finding (F-11) is a documentation-hygiene duplicate paragraph fixed in the same edit pass, not a security defect.

The ADR is ready to flip to `accepted` and implementation can begin per the §3d ordered slice plan: dep-bump (MCP SDK pin) → minimal `generate` tool first → `extract_params` → `list_models` / `list_loras` → `iterate` → `edit` stub. Each slice runs `code-reviewer` AND `security-auditor` (both Opus, model pinned at invocation) before commit.
