#!/usr/bin/env python3
"""Test harness for comfyless.pause — ^C pause/resume (vllm-style).

Covers the sigint_pause contract without sending real signals: the handler
and step-end callback are invoked directly (they are plain functions once
installed). Invariants, each with a negative case:

  1. No-op guards       — non-TTY stdin, worker thread, unsupported
                          pipeline signature, pre-existing callback: the
                          context must change NOTHING (no handler swap, no
                          kwargs mutation).
  2. Abort unchanged    — second ^C (before the boundary, or while paused)
                          raises KeyboardInterrupt exactly like a bare ^C
                          did before the module existed.
  3. Pause at boundary  — first ^C only sets a flag; the block happens in
                          the step-end callback, which restores the DEFAULT
                          SIGINT handler while blocked on input().
  4. Clean exit         — the SIGINT handler and call_kwargs are restored
                          on every exit path (normal, resume, abort).
"""

import signal
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from comfyless.pause import sigint_pause, _supports_step_callback

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


def _pipe_with_callback(prompt=None, callback_on_step_end=None, **kw):
    return None


def _pipe_without_callback(prompt=None, **kw):
    return None


_TTY = lambda: True
_NOTTY = lambda: False


# ──────────────────────────────────────────────────────────────────────
print("── invariant 1: no-op guards ──────────────────────────────────")

_orig_handler = signal.getsignal(signal.SIGINT)

kw = {"prompt": "x"}
with sigint_pause(_pipe_with_callback, kw, _isatty=_NOTTY):
    check("non-TTY: no callback injected (NEGATIVE)",
          "callback_on_step_end" not in kw, f"{kw.keys()}")
    check("non-TTY: SIGINT handler untouched (NEGATIVE)",
          signal.getsignal(signal.SIGINT) is _orig_handler)

kw = {"prompt": "x"}
with sigint_pause(_pipe_without_callback, kw, _isatty=_TTY):
    check("unsupported pipeline signature: no callback injected (NEGATIVE)",
          "callback_on_step_end" not in kw)
    check("unsupported pipeline: SIGINT handler untouched (NEGATIVE)",
          signal.getsignal(signal.SIGINT) is _orig_handler)

_sentinel_cb = object()
kw = {"prompt": "x", "callback_on_step_end": _sentinel_cb}
with sigint_pause(_pipe_with_callback, kw, _isatty=_TTY):
    check("pre-existing callback never clobbered (NEGATIVE)",
          kw["callback_on_step_end"] is _sentinel_cb)
check("pre-existing callback survives exit too",
      kw["callback_on_step_end"] is _sentinel_cb)

_thread_result = {}


def _in_thread():
    tkw = {"prompt": "x"}
    with sigint_pause(_pipe_with_callback, tkw, _isatty=_TTY):
        _thread_result["injected"] = "callback_on_step_end" in tkw


_t = threading.Thread(target=_in_thread)
_t.start()
_t.join()
check("worker thread: no-op (daemon surface can never arm it) (NEGATIVE)",
      _thread_result.get("injected") is False)

check("_supports_step_callback: introspection matches reality",
      _supports_step_callback(_pipe_with_callback) is True
      and _supports_step_callback(_pipe_without_callback) is False
      and _supports_step_callback(42) is False)

# Detached stdin (sys.stdin is None — no-console spawn): must no-op, not
# AttributeError at context entry (review SHOULD 1 NEGATIVE).
_saved_stdin = sys.stdin
try:
    sys.stdin = None
    kw = {"prompt": "x"}
    _entered = False
    with sigint_pause(_pipe_with_callback, kw):
        _entered = True
        check("sys.stdin=None: no callback injected (NEGATIVE)",
              "callback_on_step_end" not in kw)
    check("sys.stdin=None: context is a transparent no-op", _entered)
finally:
    sys.stdin = _saved_stdin


# ──────────────────────────────────────────────────────────────────────
print("── invariants 2+3: pause, resume, abort ───────────────────────")

kw = {"prompt": "x"}
_inputs = []
with sigint_pause(_pipe_with_callback, kw, _isatty=_TTY,
                  _input=lambda: _inputs.append("resumed")):
    _armed = signal.getsignal(signal.SIGINT)
    check("armed: SIGINT handler swapped in", _armed is not _orig_handler)
    _cb = kw.get("callback_on_step_end")
    check("armed: step-end callback injected", callable(_cb))

    # No pause requested → callback is a pass-through.
    _ck = {"latents": "L"}
    check("no pause requested: callback passes kwargs through unchanged",
          _cb(None, 0, 0, _ck) is _ck and _inputs == [])
    check("None callback_kwargs coerced to {} (diffusers contract)",
          _cb(None, 0, 0, None) == {})

    # First ^C: flag only, no raise (invariant 3).
    _raised = False
    try:
        _armed(signal.SIGINT, None)
    except KeyboardInterrupt:
        _raised = True
    check("first ^C sets the pause flag WITHOUT raising", not _raised)

    # Second ^C before the boundary: abort (invariant 2 NEGATIVE).
    _raised = False
    try:
        _armed(signal.SIGINT, None)
    except KeyboardInterrupt:
        _raised = True
    check("second ^C before the boundary raises KeyboardInterrupt "
          "(abort path unchanged) (NEGATIVE)", _raised)

    # The abort attempt above did NOT clear the flag — the next step
    # boundary pauses (matches 'the raise escaped the handler' semantics
    # only when it propagates; here we swallowed it, so simulate a fresh
    # pause request instead: clear by running the callback once).
    _during = {}

    def _probe_input():
        _during["handler"] = signal.getsignal(signal.SIGINT)
        return ""

    kw2 = {"prompt": "x"}
    with sigint_pause(_pipe_with_callback, kw2, _isatty=_TTY,
                      _input=_probe_input):
        _armed2 = signal.getsignal(signal.SIGINT)
        _cb2 = kw2["callback_on_step_end"]
        _armed2(signal.SIGINT, None)          # request pause
        _out = _cb2(None, 11, 0, {"latents": "L"})
        check("paused at boundary: input() ran under the DEFAULT SIGINT "
              "handler (so ^C there aborts)",
              _during.get("handler") is signal.default_int_handler)
        check("resume: callback returned kwargs and generation continues",
              _out == {"latents": "L"})
        check("resume: armed handler reinstalled after input()",
              signal.getsignal(signal.SIGINT) is _armed2)
        # Second pause/resume cycle works (flag re-armable).
        _armed2(signal.SIGINT, None)
        _cb2(None, 12, 0, {})
        check("second pause/resume cycle works",
              _during.get("handler") is signal.default_int_handler)

# ^C while paused: input() raises KeyboardInterrupt → propagates out of the
# callback (through the pipeline) and the context restores the original
# handler on the way out.
kw3 = {"prompt": "x"}


def _ki_input():
    raise KeyboardInterrupt


_raised = False
try:
    with sigint_pause(_pipe_with_callback, kw3, _isatty=_TTY,
                      _input=_ki_input):
        _armed3 = signal.getsignal(signal.SIGINT)
        _cb3 = kw3["callback_on_step_end"]
        _armed3(signal.SIGINT, None)
        _cb3(None, 5, 0, {})
except KeyboardInterrupt:
    _raised = True
check("^C while paused aborts (KeyboardInterrupt propagates) (NEGATIVE)",
      _raised)

# Terminal EOF while paused (stdin closed): EOFError propagates like the
# ^C abort — never swallowed — and the handler is still restored.
kw3b = {"prompt": "x"}


def _eof_input():
    raise EOFError


_raised = False
try:
    with sigint_pause(_pipe_with_callback, kw3b, _isatty=_TTY,
                      _input=_eof_input):
        _armed3b = signal.getsignal(signal.SIGINT)
        _armed3b(signal.SIGINT, None)
        kw3b["callback_on_step_end"](None, 5, 0, {})
except EOFError:
    _raised = True
check("EOF while paused propagates (abort-equivalent) (NEGATIVE)", _raised)

# Late ^C: the flag set AFTER the final step boundary must not silently
# evaporate — the context prints the truthful notice on normal exit and
# does NOT raise (warn-don't-block; review SHOULD 2).
import io as _io                                       # noqa: E402
import contextlib as _ctx                              # noqa: E402
kw4 = {"prompt": "x"}
_err = _io.StringIO()
with _ctx.redirect_stderr(_err):
    with sigint_pause(_pipe_with_callback, kw4, _isatty=_TTY):
        _armed4 = signal.getsignal(signal.SIGINT)
        _armed4(signal.SIGINT, None)   # ^C during decode: no boundary left
check("late ^C (after final boundary): no raise, run continues",
      True)
check("late ^C: truthful notice printed, not silently eaten (NEGATIVE)",
      "already finished" in _err.getvalue(), f"got {_err.getvalue()!r}")
check("late ^C: handler restored on exit",
      signal.getsignal(signal.SIGINT) is _orig_handler)


# ──────────────────────────────────────────────────────────────────────
print("── invariant 4: clean exit on every path ──────────────────────")

check("handler restored after normal exit",
      signal.getsignal(signal.SIGINT) is _orig_handler)
check("callback removed from kwargs after normal exit (no sidecar leak)",
      "callback_on_step_end" not in kw)
check("callback removed after abort exit too (NEGATIVE)",
      "callback_on_step_end" not in kw3)


# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print(f"  {passed} passed, {failed} failed")
print("─" * 50)
sys.exit(1 if failed else 0)
