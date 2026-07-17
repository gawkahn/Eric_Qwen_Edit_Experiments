# Slice PAUSE — ^C pause/resume for foreground generation — Vision

**Date:** 2026-07-16 · **Author:** Claude (Fable 5) · **Requested:** Grant (2026-07-16: "^C gives you a 'paused, ^C again to exit' kind of response — I think vllm does it")

## 1. Vision

**Outcome when done:** during a foreground `comfyless generate` (including `--iterate`) denoising loop, the first ^C pauses at the next step boundary ("press Enter to resume, ^C to abort") instead of killing the run; a second ^C — before the boundary or while paused — aborts exactly like a bare ^C does today. Protects hours-long runs from a stray ^C and lends the terminal back mid-run without losing GPU state.

**Invariants (each with a negative test in `test_pause.py`):**
1. **The abort path is unchanged** — a determined double-^C (or ^C at the pause prompt) raises `KeyboardInterrupt` just like today. The feature only adds a stop between "running" and "dead".
2. **Never armed off the interactive CLI** — non-TTY stdin (daemon, MCP, `--json` under a parent process, redirected runs) and non-main threads are transparent no-ops; blocking on `input()` there would hang a server.
3. **Never armed on pipelines that don't accept `callback_on_step_end`** — signature-introspected, not assumed; unknown families keep today's behavior.
4. **No hook leakage** — an existing `callback_on_step_end` in call_kwargs is never clobbered; the injected callback is removed from the dict on every exit path (kwargs may be reused/recorded).
5. **Clean handler restore** — the process SIGINT handler is restored on normal, resume, and abort exits.

**Out of scope:** pause across the Hunyuan refiner chain (two-pipeline call — untouched, plain ^C behavior); Stable Cascade dispatch; between-generation pause in `--iterate` (a ^C there aborts as today); freeing VRAM while paused (this is a breakpoint, not a yield); daemon-side pause.

## 2. Change boundary / edit scope (hard)

`comfyless/pause.py` (new), `comfyless/generate.py` (wrap the two inference call sites — plain and NAG — in `sigint_pause`; no other change), `test_pause.py` (new, auto-joins the `just tests` glob), this doc. NOT `comfyless/server.py` / `mcp_server.py` (guarded out by invariant 2 instead).

## 3. Design

`sigint_pause(pipe_call, call_kwargs)` context manager: installs a SIGINT handler that sets a flag (second ^C raises); injects a `callback_on_step_end` that, when the flag is set, prints the pause banner, restores the DEFAULT SIGINT handler, and blocks on `input()` (so ^C at the prompt raises `KeyboardInterrupt` out through the pipeline call — the pre-existing abort semantics). The stock `pipe.__call__` signature stands proxy for the NAG wrappers — all four `pipelines/nag_*.py` `__call__`s accept `callback_on_step_end` (verified 2026-07-16).

## 4. Proof

`./.venv/bin/python3 test_pause.py` — 28 checks, no real signals needed (handler and callback invoked directly). Live TTY behavior verifiable on the next foreground run (GPUs busy today).

## 5. Review outcome (2026-07-16)

code-reviewer (Fable): APPROVED with two SHOULDs, both folded: (1) detached-stdin (`sys.stdin is None`) no-ops instead of raising at context entry; (2) a first ^C landing AFTER the final step boundary (VAE decode/postprocess — no boundary left) no longer silently evaporates: a loud "this generation had already finished; continuing (^C again to abort)" notice prints on exit. The notice-and-continue choice (vs re-raising to restore exact pre-slice abort) follows Grant's standing warn-don't-block preference — the finished image is kept; flag for revisit if a real run shows re-raise is wanted. NITs folded: handler swapped before the pause flag clears (a ^C in that gap now aborts as intended); pause banner gets a leading newline (tqdm line garbling). Not a Red Zone surface — no security-auditor gate.

AI-Disclosure: Claude (Fable 5) authored; Grant reviewed.
