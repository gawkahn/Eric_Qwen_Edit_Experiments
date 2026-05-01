# ADR-011: Comfyless MCP Server as the LLM-Agent Calling Interface

**Date:** 2026-04-28
**Status:** accepted
**AI-Disclosure:** Claude (Opus 4.7, 1M context) authored; Grant reviewed.
**Supersedes (in part):** ADR-006 — the `--json` bridge as the LLM-agent transport. The dual-mode CLI / sidecar JSON / `--params` replay / stdout-stderr split parts of ADR-006 remain in force for non-LLM scripted callers.

---

## Context

ADR-006 (decision ~2026-04-14, retroactively documented 2026-04-21) specified comfyless's dual-mode design: a human CLI mode and a `--json` stdin/stdout bridge for LLM-agent callers. The alternatives explicitly considered and rejected at that time were "separate binaries", "HTTP API", "no contract versioning", and "merged stdout/stderr". The Model Context Protocol (MCP) does not appear in the alternatives — it was not on the radar when the decision was made, and no concrete agent harness existed yet that the bridge was being built against.

Two facts have changed since:

1. **`local_agents` exists.** The user's `local_agents` project — first cited in this codebase's Backlog and memory as "the LLM agent backbone" — is now an actual tool-calling harness that comfyless needs to integrate with. The `local_agents` preference, surfaced 2026-04-28, is MCP. Continuing the bridge would require new development for tool discovery, a multi-tool dispatch surface (`generate`, `list_models`, `list_loras`, `extract_params`, `iterate`), long-lived connection support, and JSONSchema-typed input introspection — all of which MCP provides natively and the current `--json` mode lacks.

2. **MCP is the standard tool-calling protocol for LLM clients.** Anthropic's SDK supports it natively. Claude Desktop, Claude Code, `local_agents`, and any other MCP-aware client consume MCP servers without bespoke per-tool integration. JSON-RPC over stdio (or HTTP/SSE for remote) carries `tools/list`, `tools/call`, and JSONSchema-typed inputs as first-class concerns. Protocol versioning is part of the initialize handshake — the contract-version field invented by ADR-006 is no longer needed on the LLM-agent side.

The remainder of ADR-006 — the dual-mode CLI/sidecar entry point, `--params` replay semantics, sidecar JSON discipline, and strict stdout/stderr split *within* `--json` mode for any caller still using it — is orthogonal to the calling-protocol choice and stays intact. What changes is *which transport the LLM agent uses to drive comfyless*.

The comfyless Unix-socket daemon (ADR-001, hardened per the 2026-04-23 §12 review and the 2026-04-24 timeout/BrokenPipe fix) already provides the long-lived in-process model cache. An MCP server is structurally a thin adapter on top of the existing socket-client code: incoming `tools/call` → translate to internal request → existing daemon path. No new model-loading machinery, no new IPC, no new caching layer. The hard work is at the same boundary either path requires: input validation, path allowlisting, audit trail, and output-path policy.

This is a §12 trigger and Red Zone from day one per the project CLAUDE.md ("When the `--json` / LLM agent wiring lands: write spec + ADR before code, run `security-auditor`, treat as Red Zone from the first commit"). Replacing the bridge with MCP does not change the Red Zone status; the threat model is the LLM driving paths and parameters into `generate()`, regardless of transport.

## Decision

Adopt MCP as the LLM-agent calling interface for comfyless.

### 1. Module: `comfyless/mcp_server.py`

A new module exposing a stdio-transport MCP server. It depends on the official MCP Python SDK (exact-pinned per global §11). The server is a thin adapter:

- Each MCP tool maps to an internal call into existing comfyless code (`generate.run_one`, the Unix-socket client wrappers, `extract_*` helpers, `cascade.run_one`).
- The server delegates to the running daemon under the same conditions the current CLI does (auto-detect, fall back to in-process when no socket is present).
- No new model-loading code paths; the MCP server has the same loader semantics, the same `COMFYLESS_SCHEMA` validation, and the same `family_defaults` overlay as the CLI.

Entry point conventionally: `python -m comfyless.mcp_server` (so it can be spawned as a stdio MCP server child by any client).

### 2. Tool surface (v1)

Six tools, JSONSchema-typed via the MCP SDK's tool-declaration helpers:

| Tool | Purpose | Inputs (sketch) | Output |
|---|---|---|---|
| `generate` | Text-to-image. Covers all GEN_PIPELINE families *and* Stable Cascade via the same tool with `model: "stablecascade"` plus a `cascade_config` field (an inline Cascade JSON-config object — file paths to cascade configs are not exposed via MCP for the same reasons §3 forbids PNG `--params`). | prompt, model, seed, width, height, steps, cfg, sampler, schedule, loras[], component overrides, output dir, savepath template | output path, resolved-params sidecar JSON, elapsed seconds |
| `edit` | **Stub in v1.** Schema declared (image_path, prompt, model, all standard generate params) so the tool slot is reserved and `tools/list` advertises it; implementation returns a `NotImplementedError`-shaped MCP error noting that the comfyless edit CLI does not exist yet. The slot exists to give the agent (and any prompt engineering around it) a stable surface to plan against without forcing a tool-list version bump when the real implementation lands. The stub validates input against the schema (via the MCP SDK's standard tool-call wrapper) before returning `NotImplementedError` — schema validation is not skipped just because the implementation is a stub, so the stub cannot be used as a side-channel to probe path existence. | image_path, prompt, model, all standard generate params | (v1: `NotImplementedError` MCP error; future: edited output path + sidecar) |
| `list_models` | Discover available models on disk + already-cached HF repos (no network calls — see §3) | optional family filter, optional path-root filter | list of `{path, model_family, recommended_dtype, source}` |
| `list_loras` | Discover available LoRAs | optional target-family filter | list of `{path, target_family, sha256, format}` |
| `extract_params` | Read sidecar from a JSON file. **PNG metadata extraction is NOT exposed via MCP** (see §3); JSON sidecars only. | path (must be a `.json` sidecar) | normalized COMFYLESS_SCHEMA params blob |
| `iterate` | Run an `--iterate` sweep (ADR-008 axes; cascade-supported subset for cascade dispatch per ADR-010 amendment 3). Input shape = the same fields `generate` accepts (the *base* config) **plus** an `axes` field declaring which params vary and across what values. Anything not in `axes` is held fixed at its base value. There is no separate "load base from sidecar/PNG" path — the agent constructs the full base inline. The MCP server enforces an absolute hard ceiling on `iterate` totals **independent of any value passed in `max_iterations`** — see §3 server-side iterate cap below. | full generate-shaped base config + `axes: {param: [values...]}` + `max_iterations` + `limit` + `batch` | list of `{output_path, sidecar, elapsed}` entries + summary |

Beyond the v1 stubbed `edit` slot, the other edit variants (Flux.2-Edit, future Cascade-Edit if any) are out of scope until the comfyless edit CLI surface lands.

### 3. Red Zone discipline

#### 3a. Server startup contract (output and model roots)

The MCP server is spawned as a child by the MCP client (Claude Desktop, `local_agents`, Claude Code, etc.). Every containment guarantee in this ADR rests on two roots being declared at spawn time, not inferred:

- **`--output-dir <abs-path>`** — the only directory the server may write into. All `generate`, `edit`, `iterate` outputs and all sidecar JSON writes must land under this root after `os.path.realpath` resolution. No symlink-escapes, no `..` segments, no temp-file fallbacks outside the root.
- **`--model-base <abs-path>`** — the allowlist root for every loadable-weight reference: `model`, `transformer_path`, `vae_path`, `text_encoder_path`, `text_encoder_2_path`, `loras[].path`, and the cascade `stage_c` / `stage_b` / `stage_a` fields. All paths get `os.path.realpath`-resolved and `_within(p, model_base)`-checked at the daemon boundary.

Both flags are required positional concerns at spawn:

```jsonc
// claude_desktop_config.json equivalent
"comfyless": {
  "command": "python",
  "args": ["-m", "comfyless.mcp_server",
           "--output-dir",  "/abs/path/outputs",
           "--model-base",  "/abs/path/models"],
}
```

The server **fails to start** if either flag is missing, points at a non-existent directory, or points at a non-directory. There is **no env-var fallback, no default, no `$HOME` inheritance.** The MCP server reuses the existing daemon's `_within(p, root)` and path-validation helpers verbatim — it does not re-implement them.

`--mcp-max-iterations <int>` — see §3c below — is an additional spawn-time flag controlling the absolute ceiling on `iterate` calls.

#### 3b. Per-tool validation, audit, and threat surface

- **Per-tool input validation** against `COMFYLESS_SCHEMA` and the cascade schema, applied at the MCP-call boundary before the request reaches `generate.run_one` or `cascade.run_one`. Existing `_validate_params` is the validator; the MCP adapter does not bypass it.
- **Output-path containment** — already covered structurally by §3a's `--output-dir` requirement.
- **Audit trail** — every tool invocation, success and rejection alike, writes one structured line to **stderr** (stdout is reserved for the MCP JSON-RPC frame; logging there would corrupt the protocol). Each line carries: tool name, redacted input (paths, params, dimensions, etc.; **prompts and negative_prompts are dropped**), success/error class, elapsed seconds. The existing `_log("PathError: ...", file=sys.stderr)` pattern in `comfyless/server.py:321` is the foundation; the MCP path *extends* it to log on success too — the existing pattern logs only rejections, so success-path audit is **additive new code**, not a free inheritance. Prompts are recoverable from the per-output sidecar JSON the user controls; they do not appear in server logs (avoids PII-in-logs when the MCP client captures the server's stderr).
- **Prompt-injection threat surface** — every model-output-derived field flowing into a path, weight reference, or filesystem destination:
  - `model`, `transformer_path`, `vae_path`, `text_encoder_path`, `text_encoder_2_path`, `loras[].path`
  - `output_path`, `savepath` template
  - inside `cascade_config`: `stage_c`, `stage_b`, `stage_a`, `scaffolding_repo`

  Each goes through `_check_paths`-equivalent allowlisting against `--model-base` (loadable weights) or `--output-dir` (writes) at the MCP-call boundary. The cascade `stage_*` and `scaffolding_repo` fields specifically — currently in-process and never validated against a `model_base` — gain the same allowlist enforcement as the standard `model` field when reached via MCP. This closes the existing in-process trust boundary that the CLI never had to defend.

#### 3c. Server-side iterate cap (independent of agent input)

ADR-008's `--max-iterations` is a fail-closed cap when a *human* typed the value. In MCP, `max_iterations` is **agent-controlled input** like every other field — a runaway loop or an exploring model can pass `max_iterations: 10000` and the cap never trips because the cap is what the agent set.

The MCP server enforces an **absolute hard ceiling** on `iterate` totals independent of any value passed in the `max_iterations` field, set by spawn-time CLI flag `--mcp-max-iterations <int>` (default **100**). If the agent's request would expand to more than the server cap (after Cartesian product, before `--limit` truncation), the call fails with a structured MCP error naming both the requested total and the cap. The agent's `max_iterations` is honored as an upper bound it imposes on itself; the server cap is the floor below which the agent's request is allowed to operate. The existing `_ITERATE_CONFIRM_THRESHOLD` interactive prompt (5+ generations) is a no-op in MCP (no human terminal); the server cap replaces it.

#### 3d. Order of operations (per global §12 and project CLAUDE.md)

1. ADR-011 (this document) reviewed and accepted.
2. `security-auditor` (Opus, model pinned at invocation per global §5A) review of the MCP server design *before* code is written. Output saved to `docs/security/review-comfyless-mcp-server-<YYYY-MM-DD>.md`.
3. Implementation slices in dependency order: dep-bump (MCP SDK pin) → minimal `generate` tool first (the riskiest one, lands all path-validation infrastructure) → `extract_params` → `list_models` / `list_loras` → `iterate` → `edit` stub.
4. Each slice runs `code-reviewer` (Opus) AND `security-auditor` (Opus) before commit.
5. The first commit that wires the MCP server lands with full Red Zone discipline — no "we'll harden it later" intermediate states.

**User-decided exclusions (2026-04-28 review of this ADR):**

- **`allow_hf_download` is NOT exposed on the MCP surface.** No flag, no field. The LLM cannot trigger HF network downloads via comfyless. Models passed to `generate`/`iterate` must be either absolute local filesystem paths OR HF repo IDs already present in the local HF cache (`resolve_hf_path` is invoked with `allow_download=False`). Resolution failures return an MCP error naming the missing repo; the LLM may then surface that to the user, but cannot download it itself. `list_models` does not perform any network calls — it scans local filesystem and already-cached HF entries only.

  *Implementation enforcement:* the MCP-server adapter hard-codes `allow_hf_download=False` at every `resolve_hf_path` and `_load_pipeline` call site. There is no `**kwargs` passthrough from the MCP request to these calls. A regression test asserts that no MCP-reachable code path can set `allow_hf_download=True` (test imports the MCP adapter, monkey-patches `resolve_hf_path` to record `allow_download` arg values, exercises every tool, asserts every recorded value is `False`).

- **PNG `--params` extraction is NOT exposed on the MCP surface.** No tool reads PNG metadata for the LLM. `extract_params` accepts JSON sidecar paths only with **two checks** applied in order:
  1. The **resolved** path (after `os.path.realpath` and symlink resolution) must end in `.json` — checking the surface name first would let a `legit.json → evil.png` symlink slip past.
  2. The resolved path must be within `--output-dir` (the same containment root that `generate` writes to). Same-directory containment is the natural model: the agent extracts params from sidecars its own prior `generate` calls produced, not from arbitrary `.json` files anywhere on the filesystem.

  Cascade `cascade_config` fields are inline JSON objects, not file paths. The LLM has no path through the MCP surface that performs PNG-byte parsing on a caller-supplied file. The CLI's `--params <png>` mode remains for human users; the LLM cannot trigger it.

These exclusions are decisions, not deferrals — adding either capability later requires a new ADR amendment that names the new threat surface and the mitigations.

The same security gates that the `--json` bridge would have needed apply here — they are not new work, just landing on a different transport.

### 4. ADR-006 disposition

ADR-006 is **superseded in part** by this ADR for the LLM-agent transport role only. Specifically superseded:

- "`--json` bridge mode" as the **planned** LLM-agent calling surface (the existing implementation is preserved; no further investment is planned in it as the LLM surface — see §5 below).
- Contract-version field as the LLM-side schema-drift detector (MCP's initialize handshake replaces this).

Specifically *not* superseded — these remain in force per ADR-006:

- Default human CLI mode (argparse, stderr progress, sidecar JSON on disk).
- `--params` / `--override` replay semantics.
- Sidecar JSON discipline (written alongside every output regardless of mode).
- Strict stdout/stderr split *within* `--json` mode for any caller still using it.

ADR-006 status flips to `accepted (superseded in part by ADR-011)` with a Changelog entry pointing here. Per global §12, the body of ADR-006 is not rewritten; only Status and Changelog are edited.

### 5. Existing `--json` mode preserved at zero further investment

The `_run_json_mode` code in `comfyless/generate.py` is left in place as-is. It does what it does today; no further development is planned for it as the LLM-agent surface. If a non-LLM scripted caller wants a stdin/stdout JSON contract, the existing mode covers it. If no caller materializes, removal is a separate future slice and a separate ADR amendment — not this one.

### 6. Transport: stdio first

v1 ships only stdio-transport MCP server. HTTP/SSE transport is out of scope. stdio inherits the same-uid threat model from the parent-process spawn relationship, mirroring what the existing Unix-socket daemon already enforces. HTTP transport changes the threat model materially (network exposure, authentication, TLS) and requires a separate ADR.

### 7. Dependency policy

The MCP Python SDK is added as an exact-pinned direct dep per global §11. `pyproject.toml` and `requirements.txt` are updated in lockstep (project CLAUDE.md rule: both list the same direct deps in the same order); `uv.lock` is regenerated with `uv lock`; the three files commit together as a single dep-bump slice before the first MCP server commit.

Install via the bare `mcp` package or the `mcp[stdio]` extra. **Explicitly do NOT install `mcp[http]` or `mcp[all]`** until/unless the HTTP-transport ADR lands — those extras pull `starlette` + `uvicorn` + the ASGI stack into the desktop tool's dep closure for no v1 benefit.

## Alternatives Rejected

### A. Continue building the `--json` bridge as the LLM-agent surface

Rejected. To reach feature parity with what MCP gives for free (tool discovery, multi-tool dispatch, JSONSchema-typed inputs, schema introspection, long-lived connection, protocol versioning) the bridge would need substantial new development. By the time those features land, the bridge would be a worse-engineered MCP server. Sunk cost on the existing `_run_json_mode` is small — it stays in place under §5 — and the marginal investment beyond what already exists is the relevant comparison. `local_agents`'s stated preference for MCP also weighs heavily here: building a non-MCP surface specifically for the only known LLM-agent caller misaligns the work.

### B. Custom HTTP API

Rejected. MCP supports HTTP/SSE transport already; rolling our own custom HTTP API would discard the standard tool-discovery and client-compatibility benefits. If HTTP transport becomes necessary, it is added to the existing MCP server, not built separately.

### C. Keep both `--json` bridge and MCP at parity

Rejected. Doubles maintenance for the LLM-agent caller surface and creates two paths for path validation, audit, and Red Zone enforcement to drift between. The bridge remains in place at zero further investment as a non-LLM-caller convenience (see §5); it does not need to be at parity.

### D. Wait for an "official" MCP implementation in diffusers / ComfyUI

Rejected. There is no such standard upstream. comfyless is the LLM-facing surface for this codebase; ComfyUI itself is not on the LLM tool-call path.

### E. Route the LLM through ComfyUI's HTTP API and skip comfyless

Rejected. comfyless exists specifically because the ComfyUI calling surface is awkward for scripted/agent use (workflow JSON, queue model, server-lifecycle entanglement, reload friction). Routing the LLM through ComfyUI recreates the problem comfyless was built to solve, and would couple the agent path to ComfyUI versioning.

### F. Single MCP tool with a method/dispatch field instead of multiple typed tools

Rejected. MCP's strength is per-tool JSONSchema. Collapsing five tools into one `comfyless` tool with a `method` field would discard the per-tool typing, force the agent to learn an internal dispatch convention, and lose the standard `tools/list` discovery surface. The cost of declaring five tools is small; the value of typed surfaces per tool is the entire point of using MCP.

## Deferred / Out of Scope

- **Edit tools** (Qwen-Edit, Flux.2-Edit, future Cascade-Edit if any) — comfyless edit CLI does not exist yet; adding edit MCP tools depends on that landing first.
- **Image-judge / planner tools** — those belong in `local_agents`, not comfyless. comfyless is the generation surface; judging an image and planning the next iteration are agent-side concerns. The MCP server is a *consumer-of-LLM-decisions* surface, not a *contains-LLM-judgment* surface.
- **HTTP/SSE MCP transport** — stdio is sufficient for v1.
- **Removing the existing `--json` bridge** — left in place at zero further investment; future deprecation is a separate decision and a separate ADR amendment.
- **LoRA registry / catalog tool** — depends on the LoRA registry / Civitai harness work (Backlog Queued). When that lands, a `query_lora_catalog` tool can be added.
- **Multi-server orchestration / cross-server transactions** — `local_agents` handles multi-server composition; comfyless ships its own server only.
- **Auth beyond same-uid** — the Unix-socket daemon enforces same-uid via 0700 socket dir + uid verification (per the 2026-04-23 hardening slice). MCP-over-stdio inherits this from the parent-process spawn relationship. Network-transport auth is out of scope until HTTP transport is on the roadmap.
- **Auto-refinement loop wiring** — Backlog Ideas; depends on this ADR landing first since the MCP server is the comfyless half of the loop.
- **Streaming progress notifications during long generations** — MCP supports notifications; whether to emit per-step progress is a polish decision for a later slice.
- **Resource exposure (`resources/list`)** — MCP servers can expose read-only resources alongside tools (e.g. last sidecar JSON, model cache state). Not in v1; revisit if a clear use case emerges.
- **Prompt templates (`prompts/list`)** — MCP servers can also publish prompt templates. Not in v1.

## Changelog

- **2026-04-28 (initial draft)**: Decision drafted in response to `local_agents` stating MCP as the clear preference. Supersedes the LLM-agent transport role of ADR-006 (the rest of ADR-006 — dual-mode CLI, sidecar, `--params` replay, stdout/stderr split for the existing bridge — remains in force). v1 tool surface defined: `generate`, `list_models`, `list_loras`, `extract_params`, `iterate`. stdio transport only. `security-auditor` (Opus) review queued before any code lands. Existing `--json` mode preserved at zero further investment. Implementation order: ADR-011 → security review → backlog re-spec → dep-bump slice (MCP SDK pin) → impl slices (one tool at a time, `generate` first).

- **2026-04-28 (review fold-in)**: User reviewed initial draft and made four scope changes:
  1. Add `edit` as a stub tool — schema declared so the slot is reserved on `tools/list`, implementation returns `NotImplementedError` until the comfyless edit CLI surface lands. Tool surface count goes from 5 to 6.
  2. Clarify `iterate` input shape — same fields as `generate` (the *base* config) plus an `axes` field; anything not in `axes` is held fixed at its base value. No separate "load base from sidecar/PNG" path; agent constructs the full base inline.
  3. `allow_hf_download` is NOT exposed at all on the MCP surface (no flag, no field). Local paths or already-cached HF repos only.
  4. PNG `--params` extraction is NOT exposed on the MCP surface. `extract_params` is JSON-only and rejects non-`.json` extensions. `cascade_config` is inline JSON only, not a file path.
  These four are decisions, not deferrals — adding any of them later requires a new ADR amendment naming the new threat surface and mitigations. Status remains `proposed` pending the security-auditor review; flips to `accepted` if the auditor returns clean.

- **2026-04-28 (security-auditor round-1 fold-in)**: Round-1 security review (saved to `docs/security/review-comfyless-mcp-server-2026-04-28.md`) returned `CHANGES REQUIRED` with 1 HIGH + 5 MEDIUM, all small ADR-text additions. Folded:
  - **F-1 (HIGH)**: §3a now mandates `--output-dir` and `--model-base` as required CLI args at MCP-client spawn time, hard-fail if missing/non-existent; no env-var fallback, no default. Reuses existing daemon's `_within(p, root)` helpers verbatim.
  - **F-2 (MEDIUM)**: §3c adds server-side `--mcp-max-iterations` cap (default 100) independent of agent-supplied `max_iterations`. Agent value is honored as a self-imposed upper bound; the server cap is the floor below which the agent's request is allowed to operate.
  - **F-3 (MEDIUM)**: §3 second exclusion now specifies `extract_params` runs `.json`-suffix check on the **resolved** path (post-realpath/symlink resolution) AND requires the resolved path to be within `--output-dir`.
  - **F-4 (MEDIUM)**: §3b audit-trail bullet now states: stderr only (stdout reserved for JSON-RPC), one line per invocation success or rejection, prompts/negative_prompts dropped (no PII-in-logs), success-path audit is additive new code beyond the existing PathError-only pattern.
  - **F-5 (MEDIUM)**: §3b prompt-injection field list expanded to name `transformer_path`, `vae_path`, `text_encoder_path`, `text_encoder_2_path`, and the cascade `stage_c`/`stage_b`/`stage_a`/`scaffolding_repo` fields explicitly. Notes that the cascade fields gain `model_base` allowlisting for the first time when reached via MCP.
  - **F-6 (MEDIUM)**: §3 first exclusion now requires hard-coded `allow_hf_download=False` at every `resolve_hf_path` and `_load_pipeline` call site in MCP code paths, with a regression test asserting no MCP-reachable code path can flip it to True.
  - **F-7 (LOW)**: §2 `edit` row now states the stub validates input against the schema before returning `NotImplementedError` (so the stub can't be used as a path-existence probe).
  - **F-10 (INFO)**: §7 now specifies install via `mcp` or `mcp[stdio]`; explicitly NOT `mcp[http]` / `mcp[all]` (keeps starlette/uvicorn out of the dep closure).
  - F-8, F-9 acknowledged as defense-in-depth notes, no ADR change needed.

  Re-firing security-auditor round 2 on the amended ADR. Status flips to `accepted` if round 2 returns CLEAN.

- **2026-04-28 (security-auditor round 2 → CLEAN, status accepted)**: Round-2 review (appended to `docs/security/review-comfyless-mcp-server-2026-04-28.md`) verified all six round-1 fold-ins (F-1 HIGH and F-2 through F-6 MEDIUM, plus F-7 LOW and F-10 INFO) are ADDRESSED. Round 2 surfaced one new INFO finding (F-11) — a duplicate-paragraph artifact in §3d left over from the round-1 restructure, marked non-blocking. Cleaned up in this same edit (lines 119-121 of the prior revision deleted). Verdict: **CLEAN**. Status flipped from `proposed` to `accepted`. Implementation may now begin per the §3d ordered slice plan: dep-bump (MCP SDK pin) → minimal `generate` tool → `extract_params` → `list_models` / `list_loras` → `iterate` → `edit` stub. Each slice runs `code-reviewer` AND `security-auditor` (both Opus, model pinned at invocation per global §5A) before commit.

- **2026-04-30 (default-model amendment + cascade-in-slice-1)**: Slice-1 Vision review surfaced that the LLM agent will most often call `generate` with just a prompt — no `model`, no `loras`, no `cfg`, no `steps`. Family-default machinery in `params_schema.py` already handles cfg/steps/sampler/scheduler defaults per-family; the gap is which family/model to pick when the agent doesn't specify. Three layered decisions, two of which land in slice 1:

  **Layer 1 — default-model fallback (slice 1).** Add `--default-model <abs-path>` as an *optional* spawn-time CLI flag alongside the required `--output-dir` and `--model-base` (§3a). If present, validated at startup with the same realpath + `_within(--model-base)` check; the server fails-closed at startup if the value is missing/non-existent/escapes `--model-base`. The `generate` tool's `model` input field becomes **optional**:
    - `model` present → existing path-allowlist flow (§3a / §3b), unchanged.
    - `model` absent + `--default-model` configured → server resolves the omitted field to the configured default; the resolved path goes through the SAME realpath + `_within` validation at request time (defense in depth — startup validation does not exempt request-time validation).
    - `model` absent + `--default-model` NOT configured → structured MCP error. **No silent "first model in dir" or any other heuristic. The ADR commits only to the mechanism, not to any specific default value.** Recommended `--default-model` values and their tradeoffs are documented in operator-facing docs (`comfyless/README.md`, updated in slice 1), not in the ADR body — different deployments pick different defaults.

  **Layer 2 — tool-description steering (slice 1).** The MCP `description` field on `generate` carries LLM-readable guidance pointing the agent at non-default models for specific prompt shapes (e.g. anime/manga → Illustrious/Pony, fastest-at-modest-quality → Stable Cascade, text-heavy/photoreal → Qwen-Image). The text is a string constant in `comfyless/mcp_server.py`, refinable across later slices without ADR amendment as the agent's behavior on real prompts gives signal.

  **Layer 3 — server-side prompt classifier (deferred).** A wrapper layer that reasons about the prompt and auto-selects model/family/loras requires labeled data on which models actually win on which prompt classes. Deferred until the LLM-as-judge loop (Backlog) produces signal. NOT a slice-1 concern.

  **Slice-1 scope expansion (cascade in, not slice 2).** Cascade load/gen times are fast enough and broadly capable enough that splitting it from slice 1 is artificial. Slice 1 now covers the full `generate` surface including `cascade_config` dispatch, with `cascade_config.stage_c` / `stage_b` / `stage_a` / `scaffolding_repo` gaining `_within(--model-base)` allowlist enforcement on the same code path as `model` / `transformer_path` / `vae_path` / `text_encoder_path` / `text_encoder_2_path` / `loras[].path`. The original §3d slice plan collapses: old slice 2 (cascade integration) merges into slice 1; old slices 3–7 renumber 2–6.

  **§2 amendment — `generate` row Inputs.** `model` is annotated as *optional (uses `--default-model` when omitted)*.

  **§3a amendment — third (optional) spawn-time flag.** `--default-model <abs-path>`. If present, must resolve to a real path under `--model-base`; server fails-closed at startup otherwise. If absent, `generate` calls without an explicit `model` field return a structured MCP error.

  **§3b amendment — default-resolution path validation.** The default path resolved from `--default-model` is treated identically to a caller-supplied `model` value: realpath + `_within(--model-base)` check at request time. Cascade `stage_*` and `scaffolding_repo` fields gain the same enforcement (already covered by the existing §3b prompt-injection field list — slice 1 implements what §3b already specifies).

  **Adjacent open question — NOT settled by this amendment.** The `model` field accepts paths or HF repo IDs. Future LLM-friendliness may want short-name resolution (`"flux2-klein-9b"`, `"cascade"`, `"qwen-image"`) handled server-side; defer to land alongside `list_models` so the agent has one discovery surface for short-name → path mapping. Future amendment, not slice 1.

  **No re-audit for this amendment.** The path-allowlist mechanism is unchanged — the default path goes through the same `realpath` + `_within` check as caller-supplied paths, and the cascade fields gain the existing rule rather than introduce a new one. Slice-1 implementation review (`code-reviewer` Opus + `security-auditor` Opus before commit per project CLAUDE.md, since the MCP server is a new surface) covers the design + implementation together. User approved this amendment without re-firing security-auditor on the ADR.

  Slice-1 Vision captured as `docs/vision/slice-1-mcp-generate.md` (mirrored to Obsidian) so future cold-start sessions can pick up without re-reading the session-dump transcript.
