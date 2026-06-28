AI-Disclosure: Claude (Opus 4.8) authored this security review; Grant reviewed.

# Security Review — Krea rebalance through daemon + MCP (2026-06-27)

Scope: §12 review triggered by changes to the Unix-socket IPC daemon
(`comfyless/server.py`) and the MCP agent surface (`comfyless/mcp_server.py`).
Reviewer: `security-auditor` (Opus 4.8). Verdict: **CLEAN — no blocking findings.**

## Summary

The change threads three Krea-2 conditioning-rebalance knobs (`rebalance` bool,
`rebalance_mult` float, `rebalance_weights` list-of-floats) from the CLI client
through the Unix-socket daemon (`server.py`) and the MCP agent surface
(`mcp_server.py`) into the already-reviewed `generate()` / `_apply_krea_rebalance`
math. Threat model: the daemon socket is local-uid-only (dir 0700 / socket
0600), so its untrusted actor is anything sharing the UID; the MCP `generate`
tool is the genuinely semi-trusted boundary (an LLM agent, reachable via mcpo →
OpenWebUI). The audit asks whether these three new fields open a DoS/OOM, a
path/file/exec surface, an abs-path egress, a cache-state-bleed, or a
validator-bypass. The overall posture is clean: the fields are scalars and a
short list of numbers that feed only in-memory conditioning tensor math; they
touch no path, no file destination, no model-load target, and neither cache key.

Each field was traced across all three transports. Validator: adding the keys to
`_RUNTIME_KIND` correctly routes them through `_ALL_FIELDS` so they are now
*known* keys that are type-checked (bool / float-with-int-cast / list) rather
than passed through unvalidated, and a malformed scalar is rejected before the
expensive load+generate path — a tightening, not a loosening. Data flow:
`rebalance_weights` reaches a tensor op, but the only allocation that scales with
the list is `torch.tensor(per_layer_weights)`, gated behind the
`len != n_layers` (n_layers ≈ 12) check, and the list is already bounded by JSON
parsing (the daemon caps frames at 1 MiB). Error paths: every downstream raise
(`encode_prompt`, the ndim/length guards, a bad-element `torch.tensor`) is caught
— daemon side by the `generate()` try/except that keeps the accept loop alive,
MCP side by `_call_tool_impl`'s `BaseException` catch that sanitizes the message
and writes the traceback only to stderr — so all failures fail closed. State:
rebalance is applied per-call into a freshly built `call_kwargs` and does not
mutate cached pipeline weights, so its deliberate exclusion from both cache keys
is correct and leaks nothing across requests. Egress: the fields are numeric,
appear in no redaction/removed-field set, and the only thing echoed to the agent
is the agent's own input values — no abs-path crosses the boundary. Version skew
fails safe: an old daemon treats the keys as unknown and ignores them; an old
client simply omits them.

## Findings

No findings at CRITICAL, HIGH, or MEDIUM severity.

### [INFO] `rebalance_weights` element types not validated at the boundary (listness only)
Location: `comfyless/params_validation.py` (`_KIND_LIST`); reaches
`comfyless/generate.py` `_apply_krea_rebalance` `torch.tensor(per_layer_weights)`.
A caller can send a 12-element list of non-numbers; `validate_machine_request`
accepts it (only checks `isinstance(list)`) and `validate_input=False` on the MCP
handler means the schema's `items: {type: number}` is advisory. The value reaches
`torch.tensor(...)` and raises — fails closed (daemon try/except + MCP
`BaseException` catch contain it; list is JSON-parse-bounded; the `len != n_layers`
guard rejects wrong-length lists cheaply). Robustness gap, not an exploit.
Optional remediation: element-numeric check before `torch.tensor`, or a
list-of-float validator kind.

### [INFO] `rebalance_mult` / `rebalance_weights` accept NaN/Inf via Python's non-standard JSON
`json.loads` accepts `Infinity`/`NaN` by default and `_KIND_FLOAT` accepts them.
An Inf/NaN multiplier/weight yields a degenerate (garbage/black) image — no OOM,
escape, or leak; cost is one self-induced wasted generation.
Optional remediation: reject non-finite floats with `math.isfinite` at the
rebalance consumption point.

Both INFO items are contained robustness gaps consistent with the project's
warn/contain-don't-pre-block footgun-tolerance posture; deferred to TECH_DEBT
rather than fixed in this slice.

## Verdict

Clean. The rebalance fields are numeric, in-memory-only conditioning knobs: they
influence no path, file destination, model-load target, exec surface, or cache
key; introduce no abs-path egress on the MCP boundary; the validator change is a
correct tightening; all downstream failure modes fail closed and are contained
without crashing the daemon; version skew is fail-safe. No blocking findings —
safe to merge from a §12 security standpoint.
