# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
"""^C pause/resume for foreground comfyless generation (vllm-style).

First SIGINT during the denoising loop requests a pause; at the next
step-end boundary the loop blocks on stdin::

    [comfyless] ^C — pausing at the next step boundary (^C again to abort now)
    [comfyless] paused after step 12/50 — press Enter to resume, ^C to abort

A second SIGINT — either before the boundary is reached or while paused —
aborts via ``KeyboardInterrupt``, exactly what a single ^C did before this
module existed. So the feature only ever ADDS a stop between "running" and
"dead"; the abort path is unchanged.

Scope guards (all make :func:`sigint_pause` a transparent no-op):

- ``enabled=False`` — the explicit opt-out. The daemon (``server.py``)
  passes this: it handles requests on its MAIN thread and is normally run
  in a foreground terminal (TTY stdin), so neither implicit guard below
  covers it — without the opt-out, a stray ^C in the daemon's terminal
  would block the whole daemon on ``input()`` mid-generation, wedging
  every client (2026-07-17);
- not the main thread (``signal.signal`` would raise);
- stdin is not a TTY — covers detached daemons, the MCP stdio surface,
  ``--json`` mode driven by a parent process, and redirected/piped runs,
  where "block on input()" would hang forever;
- the target callable does not accept ``callback_on_step_end`` (unknown
  pipeline families) — introspected, never assumed;
- the caller already installed a ``callback_on_step_end`` — we never
  clobber another hook.

The pause holds all GPU/VRAM state — it is a breakpoint, not a yield. Its
use cases are "I need the terminal back for a second WITHOUT losing a long
run" and "protect an hours-long --iterate generation from a stray ^C".
"""

from __future__ import annotations

import inspect
import signal
import sys
import threading
from contextlib import contextmanager


def _supports_step_callback(pipe_call) -> bool:
    """True if `pipe_call` accepts diffusers' callback_on_step_end kwarg."""
    try:
        params = inspect.signature(pipe_call).parameters
    except (TypeError, ValueError):
        return False
    return "callback_on_step_end" in params


@contextmanager
def sigint_pause(pipe_call, call_kwargs: dict,
                 log_prefix: str = "[comfyless]",
                 enabled: bool = True,
                 _input=input, _isatty=None):
    """Arm ^C pause/abort around one pipeline call.

    ``pipe_call`` is the callable whose signature decides callback support
    (the stock ``pipe.__call__`` also stands proxy for the NAG wrappers —
    all four ``nag_*.nag_pipe_call`` targets accept ``callback_on_step_end``,
    verified 2026-07-16). ``call_kwargs`` is mutated in place: the callback
    is injected on entry and ALWAYS removed on exit, so a dict the caller
    reuses (or records) never leaks the hook.

    ``enabled=False`` is the explicit opt-out for non-interactive callers
    whose runtime shape the implicit guards can't see (the foreground-
    terminal daemon: main thread + TTY stdin, but blocking on input()
    would wedge every client).

    ``_input`` / ``_isatty`` exist for tests only.
    """
    if not enabled:
        yield
        return
    # getattr-with-default so a detached stdin (sys.stdin is None — spawned
    # without a console, embedded interpreter) no-ops instead of raising on
    # the attribute access itself (code review 2026-07-16, SHOULD 1).
    isatty = (_isatty if _isatty is not None
              else getattr(sys.stdin, "isatty", lambda: False))
    try:
        interactive = bool(isatty())
    except (AttributeError, ValueError):  # closed stdin
        interactive = False
    if (not interactive
            or threading.current_thread() is not threading.main_thread()
            or not _supports_step_callback(pipe_call)
            or "callback_on_step_end" in call_kwargs):
        yield
        return

    pause_requested = threading.Event()

    def _handler(signum, frame):
        if pause_requested.is_set():
            # Second ^C before the step boundary: the user is done waiting.
            raise KeyboardInterrupt
        pause_requested.set()
        print(f"\n{log_prefix} ^C — pausing at the next step boundary "
              f"(^C again to abort now)", file=sys.stderr, flush=True)

    def _cb(pipe, step, timestep, callback_kwargs):
        if pause_requested.is_set():
            # While blocked on input(), hand SIGINT back to the default
            # handler so ^C raises KeyboardInterrupt out of input() — it
            # propagates through the pipeline call and aborts, same as a
            # plain ^C did before this module. Handler is swapped BEFORE
            # the flag clears so a ^C in the gap aborts (as the user
            # intended) rather than re-arming a pause (review NIT 1).
            prev = signal.signal(signal.SIGINT, signal.default_int_handler)
            pause_requested.clear()
            try:
                total = getattr(
                    getattr(pipe, "scheduler", None), "timesteps", None)
                total_s = f"/{len(total)}" if total is not None else ""
                print(f"\n{log_prefix} paused after step {step + 1}{total_s} "
                      f"— press Enter to resume, ^C to abort",
                      file=sys.stderr, flush=True)
                _input()
                print(f"{log_prefix} resuming", file=sys.stderr, flush=True)
            finally:
                signal.signal(signal.SIGINT, prev)
        return callback_kwargs if callback_kwargs is not None else {}

    call_kwargs["callback_on_step_end"] = _cb
    prev_handler = signal.signal(signal.SIGINT, _handler)
    try:
        yield
        if pause_requested.is_set():
            # First ^C landed AFTER the final step-end callback (VAE decode/
            # postprocess) — no boundary is coming, so without this notice
            # the ^C would silently evaporate under --iterate. Warn-don't-
            # block (Grant's standing preference): the finished image is
            # kept, the run continues, and the message says exactly that
            # (code review 2026-07-16, SHOULD 2).
            print(f"\n{log_prefix} ^C landed after the final step boundary — "
                  f"this generation had already finished; continuing "
                  f"(^C again to abort)", file=sys.stderr, flush=True)
    finally:
        signal.signal(signal.SIGINT, prev_handler)
        call_kwargs.pop("callback_on_step_end", None)
