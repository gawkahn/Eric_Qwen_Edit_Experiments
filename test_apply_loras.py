#!/usr/bin/env python3
"""Test harness for _apply_loras weight application (2026-07-17 fix).

The tier-1 fast path of load_lora_with_key_fix (pipe.load_lora_weights →
return True) never applied the user's `weight`, so every fast-path LoRA
silently ran at FULL trained strength — surfaced by the Qwen-Mystic/mcnl
noise investigation. _apply_loras now applies weights with ONE cumulative
pipe.set_adapters call (per-adapter singleton calls would deactivate every
earlier adapter — diffusers' set_adapters REPLACES the active set), and
excludes direct-merge adapters (weight baked at merge time).

Runs without GPU, models, or real LoRA files: load_lora_with_key_fix is
monkeypatched; the pipe is a recording stub.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from comfyless import generate as gen  # noqa: E402

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


class _Transformer:
    def __init__(self, peft_config=None):
        self.peft_config = peft_config or {}


class _StubPipe:
    def __init__(self, peft_config=None, set_adapters_raises=False,
                 list_adapters=None):
        self.transformer = _Transformer(peft_config)
        self.set_adapters_calls = []
        self._raise = set_adapters_raises
        self._list_adapters = list_adapters or {}

    def get_list_adapters(self):
        return self._list_adapters

    def set_adapters(self, names, adapter_weights=None):
        if self._raise:
            raise RuntimeError("no PEFT layers found")
        self.set_adapters_calls.append((list(names), list(adapter_weights)))


def _with_stub_loader(result=True, raises=None, per_path=None):
    """Patch gen.load_lora_with_key_fix; returns the restore handle.

    per_path: optional {basename_stem: outcome} where outcome is True/False
    or an Exception instance to raise — for mixed-outcome runs."""
    real = gen.load_lora_with_key_fix

    def _stub(pipe, lora_path, adapter_name, log_prefix="[t]", weight=1.0):
        if per_path is not None:
            outcome = per_path[adapter_name]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        if raises is not None:
            raise raises
        return result

    gen.load_lora_with_key_fix = _stub
    return real


# ──────────────────────────────────────────────────────────────────────
print("── _apply_loras: cumulative weight application ─────────────────")

_real = _with_stub_loader(result=True)
try:
    pipe = _StubPipe()
    outcomes = gen._apply_loras(pipe, [
        {"path": "/x/a.safetensors", "weight": 0.7},
        {"path": "/x/b.safetensors", "weight": 0.5},
    ])
    check("both adapters report applied",
          all(o["applied"] for o in outcomes), f"{outcomes}")
    check("EXACTLY ONE cumulative set_adapters call (singleton calls "
          "would deactivate earlier adapters — NEGATIVE)",
          len(pipe.set_adapters_calls) == 1,
          f"calls: {pipe.set_adapters_calls}")
    check("both names + both user weights in the one call",
          pipe.set_adapters_calls[0] == (["a", "b"], [0.7, 0.5]),
          f"got {pipe.set_adapters_calls[0]}")

    # Default weight: spec without "weight" → 1.0.
    pipe2 = _StubPipe()
    gen._apply_loras(pipe2, [{"path": "/x/c.safetensors"}])
    check("missing weight defaults to 1.0",
          pipe2.set_adapters_calls == [(["c"], [1.0])],
          f"got {pipe2.set_adapters_calls}")

    # Direct-merge adapters are excluded (weight baked at merge time).
    pipe3 = _StubPipe(peft_config={"d": {"_type": "lora_direct",
                                         "_weight": 0.7}})
    gen._apply_loras(pipe3, [
        {"path": "/x/d.safetensors", "weight": 0.7},
        {"path": "/x/e.safetensors", "weight": 0.4},
    ])
    check("direct-merge adapter excluded; PEFT adapter still scaled",
          pipe3.set_adapters_calls == [(["e"], [0.4])],
          f"got {pipe3.set_adapters_calls}")

    pipe4 = _StubPipe(peft_config={"f": {"_type": "lokr_direct",
                                         "_weight": 1.0}})
    gen._apply_loras(pipe4, [{"path": "/x/f.safetensors", "weight": 1.0}])
    check("all-direct-merge run → set_adapters never called (NEGATIVE)",
          pipe4.set_adapters_calls == [], f"got {pipe4.set_adapters_calls}")

    # set_adapters failure: warn, never raise; outcomes intact.
    pipe5 = _StubPipe(set_adapters_raises=True)
    out5 = gen._apply_loras(pipe5, [{"path": "/x/g.safetensors",
                                     "weight": 0.3}])
    check("set_adapters failure warns instead of raising (outcomes kept)",
          len(out5) == 1 and out5[0]["applied"] is True)
finally:
    gen.load_lora_with_key_fix = _real

# Mid-loop exception must NOT shift later specs' weights (review finding
# 3): the outcomes↔loras alignment is by index, so spec 3's weight has to
# land on spec 3's adapter even when spec 2 exploded.
_real = _with_stub_loader(per_path={
    "a2": True, "b2": RuntimeError("mid-loop boom"), "c2": True})
try:
    pipe8 = _StubPipe()
    out8 = gen._apply_loras(pipe8, [
        {"path": "/x/a2.safetensors", "weight": 0.7},
        {"path": "/x/b2.safetensors", "weight": 0.9},
        {"path": "/x/c2.safetensors", "weight": 0.5},
    ])
    check("mid-loop exception: later weight stays with ITS adapter, "
          "no index shift (NEGATIVE)",
          pipe8.set_adapters_calls == [(["a2", "c2"], [0.7, 0.5])]
          and out8[1]["applied"] is False,
          f"calls {pipe8.set_adapters_calls}")
finally:
    gen.load_lora_with_key_fix = _real

# Kohya TE half (review finding 1): "<name>_te" adapters loaded by
# _apply_te_lora must ride the cumulative call at the parent's weight —
# set_adapters REPLACES the active set on every component, so omitting
# them would silently deactivate the TE half.
_real = _with_stub_loader(result=True)
try:
    pipe9 = _StubPipe(list_adapters={"transformer": ["k"],
                                     "text_encoder": ["k_te"]})
    gen._apply_loras(pipe9, [{"path": "/x/k.safetensors", "weight": 0.6}])
    check("TE half adapter included at the parent's weight (finding 1)",
          pipe9.set_adapters_calls == [(["k", "k_te"], [0.6, 0.6])],
          f"got {pipe9.set_adapters_calls}")
finally:
    gen.load_lora_with_key_fix = _real

# Failed loads are excluded from the weight call.
_real = _with_stub_loader(result=False)
try:
    pipe6 = _StubPipe()
    out6 = gen._apply_loras(pipe6, [{"path": "/x/h.safetensors",
                                     "weight": 0.9}])
    check("0-module LoRA (applied=False) excluded from set_adapters "
          "(NEGATIVE)",
          pipe6.set_adapters_calls == [] and out6[0]["applied"] is False,
          f"calls {pipe6.set_adapters_calls}")
finally:
    gen.load_lora_with_key_fix = _real

_real = _with_stub_loader(raises=RuntimeError("boom"))
try:
    pipe7 = _StubPipe()
    out7 = gen._apply_loras(pipe7, [{"path": "/x/i.safetensors",
                                     "weight": 0.9}])
    check("loader exception → outcome False, no set_adapters (NEGATIVE)",
          out7[0]["applied"] is False and pipe7.set_adapters_calls == [])
finally:
    gen.load_lora_with_key_fix = _real


# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print(f"  {passed} passed, {failed} failed")
print("─" * 50)
sys.exit(1 if failed else 0)
