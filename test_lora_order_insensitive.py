#!/usr/bin/env python3
"""Order-insensitive direct-merge + pre-baked .diff passthrough.

Two Krea LoRA failure modes, each proven with a negative case:

  A. Direct-merge after PEFT wrapping — a converted (.diff) LoRA must apply
     onto a transformer whose modules a PRIOR PEFT LoRA already wrapped
     (weight moved to <base>.base_layer.weight).  resolve_merge_target must
     find the base_layer weight and _apply_converted_lora_as_delta must
     mutate it.  Negative: a module absent from the model is still skipped.

  B. Pre-baked .diff passthrough — convert_state_dict must re-emit a module
     that ships a fully-materialized `.diff` (e.g. Krea txtfusion.projector
     bypass) instead of dropping it as "unsupported".  Negative: a genuinely
     unsupported suffix (LoHa hada_*) is still skipped.

Pure CPU, no GPU / no 20B weights.  Run:
  ./.venv/bin/python3 test_lora_order_insensitive.py    (expect 0 failures)
"""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import comfyless  # installs folder_paths/comfy stubs so nodes/ imports cleanly

import torch
import torch.nn as nn
from peft import LoraConfig, inject_adapter_in_model

from comfyless.core.eric_lora_format_convert import ConversionPlan, RenameRule
from comfyless.core.eric_lora_format_convert_apply import (
    convert_state_dict,
    resolve_merge_target,
    resolve_restore_target,
    _apply_converted_lora_as_delta,
    flatten_lokr_to_lora_sd,
)
from nodes.eric_qwen_edit_lora import unload_adapters

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


# ── A tiny transformer-shaped module in diffusers Krea layout ──────────
class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_gate = nn.Linear(16, 16, bias=False)
        self.to_k = nn.Linear(16, 16, bias=False)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attn()


def fresh_model():
    m = nn.Module()
    m.transformer_blocks = nn.ModuleList([Block()])
    return m


print("── A. resolve_merge_target through PEFT wrapping ─────────────")

m = fresh_model()
names = {n for n, _ in m.named_parameters()}
check("A1 unwrapped resolves to .weight",
      resolve_merge_target(names, "transformer_blocks.0.attn.to_gate")
      == "transformer_blocks.0.attn.to_gate.weight")

inject_adapter_in_model(
    LoraConfig(r=4, target_modules=["to_gate", "to_k"], lora_alpha=4), m, "prior")
wrapped_names = {n for n, _ in m.named_parameters()}
check("A2 wrapped resolves to base_layer.weight",
      resolve_merge_target(wrapped_names, "transformer_blocks.0.attn.to_gate")
      == "transformer_blocks.0.attn.to_gate.base_layer.weight",
      f"got {resolve_merge_target(wrapped_names, 'transformer_blocks.0.attn.to_gate')}")
check("A3 absent module resolves to None",
      resolve_merge_target(wrapped_names, "transformer_blocks.0.attn.nonexistent")
      is None)


print("── A. direct merge applies onto a PEFT-wrapped transformer ───")

# Model already wrapped by a prior PEFT adapter (m from above).
base = "transformer_blocks.0.attn.to_gate"
bl_key = base + ".base_layer.weight"
before = dict(m.named_parameters())[bl_key].detach().clone()
delta = torch.full((16, 16), 0.25)
ok = _apply_converted_lora_as_delta(
    m, {base + ".diff": delta}, "realism", weight=2.0, log_prefix="[t]")
after = dict(m.named_parameters())[bl_key]
check("A4 merge reports success on wrapped model", ok is True)
check("A5 base_layer.weight received delta*weight",
      torch.allclose(after, before + delta * 2.0),
      f"max diff {(after - (before + delta*2.0)).abs().max().item()}")

# Negative: a .diff whose module doesn't exist must be skipped, not applied.
ok_none = _apply_converted_lora_as_delta(
    m, {"transformer_blocks.0.attn.ghost.diff": delta},
    "ghost", weight=1.0, log_prefix="[t]")
check("A6 unknown module → no-op (skipped)", ok_none is False)


print("── B. convert_state_dict passes through pre-baked .diff ──────")

# A Krea-shaped plan (rename diffusion_model.txtfusion. → text_fusion.)
plan = ConversionPlan(
    source_family="krea_native",
    target_family="diffusers_krea",
    rename_rules=[RenameRule("diffusion_model.txtfusion.", "text_fusion.")],
    qkv_splits=[],
    model_signature="to_gate",
)

proj_sd = {"diffusion_model.txtfusion.projector.diff": torch.zeros(1, 12)}
out = convert_state_dict(proj_sd, plan, log_prefix="[t]")
check("B1 .diff module re-emitted at renamed target",
      "text_fusion.projector.diff" in out,
      f"got keys {list(out)}")
check("B2 .diff tensor preserved", out.get("text_fusion.projector.diff") is not None
      and tuple(out["text_fusion.projector.diff"].shape) == (1, 12))

# Negative: a real LoHa (hada_*) module is still unsupported → dropped.
loha_sd = {
    "diffusion_model.txtfusion.x.hada_w1_a": torch.zeros(4, 8),
    "diffusion_model.txtfusion.x.hada_w1_b": torch.zeros(4, 8),
    "diffusion_model.txtfusion.x.hada_w2_a": torch.zeros(4, 8),
    "diffusion_model.txtfusion.x.hada_w2_b": torch.zeros(4, 8),
}
out2 = convert_state_dict(loha_sd, plan, log_prefix="[t]")
check("B3 LoHa still unsupported (dropped)", len(out2) == 0, f"got {list(out2)}")

# B4: a co-shipped bias delta (.diff_b) is not silently swallowed. The weight
# .diff is still emitted; the bias is surfaced (reviewer finding).
bias_sd = {
    "diffusion_model.txtfusion.projector.diff": torch.zeros(1, 12),
    "diffusion_model.txtfusion.projector.diff_b": torch.zeros(1),
}
out3 = convert_state_dict(bias_sd, plan, log_prefix="[t]")
check("B4 weight .diff still emitted alongside a .diff_b",
      "text_fusion.projector.diff" in out3
      and "text_fusion.projector.diff_b" not in out3,
      f"got {list(out3)}")


print("── C. unload restores after PEFT wrapper teardown ────────────")

# Merge a delta into a PEFT-wrapped module, then REMOVE the PEFT wrapper
# (so weight moves base_layer.weight → .weight), then unload the direct
# adapter. Restore must still find the weight and undo the delta.
mc = fresh_model()
key = "transformer_blocks.0.attn.to_gate"
orig = dict(mc.named_parameters())[key + ".weight"].detach().clone()

inject_adapter_in_model(
    LoraConfig(r=4, target_modules=["to_gate"], lora_alpha=4), mc, "prior")
d = torch.full((16, 16), 0.5)
_apply_converted_lora_as_delta(
    mc, {key + ".diff": d}, "surgical", weight=1.0, log_prefix="[t]")
# The backup was keyed on base_layer.weight (module was wrapped at merge).
check("C1 resolve_restore_target finds base_layer key while still wrapped",
      resolve_restore_target({n for n, _ in mc.named_parameters()},
                             key + ".base_layer.weight")
      == key + ".base_layer.weight")

# Tear down the PEFT wrapper: weight collapses base_layer.weight → .weight
# (carrying the merged delta with it).
gate = mc.transformer_blocks[0].attn.to_gate
merged_weight = gate.base_layer.weight.detach().clone()  # W0 + delta
# Replace wrapper with a plain Linear holding the merged weight.
plain = nn.Linear(16, 16, bias=False)
plain.weight.data.copy_(merged_weight)
mc.transformer_blocks[0].attn.to_gate = plain
names_now = {n for n, _ in mc.named_parameters()}
check("C2 wrapper torn down → weight back at .weight",
      key + ".weight" in names_now and key + ".base_layer.weight" not in names_now)

# Now unload the surgical (direct-merge) adapter — must restore original.
class _Pipe:
    pass
pp = _Pipe(); pp.transformer = mc
unload_adapters(pp, ["surgical"], log_prefix="[t]")
restored = dict(mc.named_parameters())[key + ".weight"]
check("C3 unload restored original weight despite wrapper teardown",
      torch.allclose(restored, orig),
      f"max diff {(restored - orig).abs().max().item()}")


# ── LoKR → standard-LoRA flatten (2026-07-06; LoKR-on-Z-Image rescue) ──
# A same-arch LoKR that the direct-merge can't place (e.g. Z-Image key
# mapping) is flattened to lora_A/lora_B via SVD so diffusers' fast path
# can load it. Verify: reconstruction accuracy, key format, passthrough,
# rank truncation, empty-on-no-lokr.
print("\n── flatten_lokr_to_lora_sd (LoKR-on-Z-Image rescue) ──")
torch.manual_seed(7)
_w1 = torch.randn(2, 2)
_w2 = torch.randn(4, 4)  # kron -> 8x8, full rank captured at r<=8
_sd = {
    "layers.0.attention.to_q.lokr_w1": _w1,
    "layers.0.attention.to_q.lokr_w2": _w2,
    "layers.0.other.lora_A.weight": torch.randn(4, 8),  # passthrough
}
_flat = flatten_lokr_to_lora_sd(_sd, target_rank=64)
_A = _flat.get("layers.0.attention.to_q.lora_A.weight")
_B = _flat.get("layers.0.attention.to_q.lora_B.weight")
check("flatten: emits lora_A/lora_B at the same module path",
      _A is not None and _B is not None)
check("flatten: lokr_w1/w2 keys removed",
      not any(".lokr_" in k for k in _flat))
check("flatten: non-LoKR keys pass through untouched",
      "layers.0.other.lora_A.weight" in _flat)
if _A is not None:
    _err = (_B.float() @ _A.float() - torch.kron(_w1, _w2)).abs().max().item()
    check("flatten: B@A reconstructs kron(w1,w2) (scale=1.0, no alpha)",
          _err < 1e-4, f"max err {_err:.2e}")
# rank truncation: 64x64 delta, cap at 16 → A rows <= 16
_flat2 = flatten_lokr_to_lora_sd(
    {"x.lokr_w1": torch.randn(4, 4), "x.lokr_w2": torch.randn(16, 16)},
    target_rank=16)
check("flatten: rank truncation honored (A rows <= target_rank)",
      _flat2["x.lora_A.weight"].shape[0] <= 16)
# no LoKR present → empty (caller treats as nothing to flatten)
check("flatten: returns {} when no LoKR modules present",
      flatten_lokr_to_lora_sd({"a.lora_A.weight": torch.randn(2, 4)}) == {})

# review finding 3: stored alpha sentinel (ai-toolkit ~1e10) MUST be ignored
# for full w1/w2 — else scale = 1e10/r produces pure noise. reconstruct_lokr_
# delta ignores alpha when neither w is decomposed; flatten relies on it.
_flat_a = flatten_lokr_to_lora_sd(
    {"m.lokr_w1": _w1, "m.lokr_w2": _w2, "m.alpha": torch.tensor(1e10)},
    target_rank=64)
_Aa = _flat_a["m.lora_A.weight"]
_Ba = _flat_a["m.lora_B.weight"]
_erra = (_Ba.float() @ _Aa.float() - torch.kron(_w1, _w2)).abs().max().item()
check("flatten: alpha sentinel (1e10) IGNORED for full w1/w2 (scale stays "
      "1.0, not noise)", _erra < 1e-4, f"max err {_erra:.2e}")

# review finding 4: decomposed-only module (no full w1/w2) is skipped, not
# guessed — and does not emit lora keys for that module.
_flat_dec = flatten_lokr_to_lora_sd(
    {"d.lokr_w1_a": torch.randn(2, 4), "d.lokr_w1_b": torch.randn(4, 2)})
check("flatten: decomposed-only LoKR module skipped (no full w1/w2)",
      _flat_dec == {})

# ── review finding 2: wiring — flatten fires ONLY after PEFT raises AND
# direct-merge applies 0; the stale peft_config marker is popped before the
# standard loader re-registers. Monkeypatch the tiers around the real
# _load_lokr_adapter dispatcher. ──
print("\n── LoKR flatten wiring (fires only on failure; marker pop) ──")
import nodes.eric_qwen_edit_lora as _lm


class _FakeTf(nn.Module):
    def __init__(self):
        super().__init__()
        self.peft_config = {}


class _FakePipe:
    def __init__(self):
        self.transformer = _FakeTf()


_lokr_sd = {"layers.0.attn.to_q.lokr_w1": torch.randn(2, 2),
            "layers.0.attn.to_q.lokr_w2": torch.randn(4, 4)}
_save = (_lm._load_lokr_adapter_peft, _lm._load_lokr_adapter_direct,
         _lm._load_lora_adapter)


def _peft_raise(*a, **k):
    raise RuntimeError("size mismatch")


# Case A: PEFT raises, direct applies 0 (and leaves a stale marker) → flatten.
_seen = {}


def _direct_zero(pipe, sd, name, lp, weight=1.0):
    pipe.transformer.peft_config[name] = {"_type": "lokr_direct",
                                          "_applied_modules": 0}
    return False


def _std_capture(pipe, sd, name, lp, weight=1.0):
    _seen["std"] = True
    _seen["marker_at_call"] = name in pipe.transformer.peft_config
    _seen["has_loraA"] = any(k.endswith(".lora_A.weight") for k in sd)
    return True


try:
    _lm._load_lokr_adapter_peft = _peft_raise
    _lm._load_lokr_adapter_direct = _direct_zero
    _lm._load_lora_adapter = _std_capture
    _rc = _lm._load_lokr_adapter(_FakePipe(), dict(_lokr_sd), "adap", "[t]", 1.0)
finally:
    (_lm._load_lokr_adapter_peft, _lm._load_lokr_adapter_direct,
     _lm._load_lora_adapter) = _save
check("wiring: PEFT-raise + direct-0 → flatten → standard loader invoked",
      _seen.get("std") is True)
check("wiring: standard loader receives flattened lora_A/B keys",
      _seen.get("has_loraA") is True)
check("wiring: stale peft_config marker POPPED before standard re-register",
      _seen.get("marker_at_call") is False)
check("wiring: rescue reports overall success", _rc is True)

# Case B: direct-merge applies >0 → flatten NOT reached (no behavior change
# for currently-working LoKRs).
_seen2 = {}
try:
    _lm._load_lokr_adapter_peft = _peft_raise
    _lm._load_lokr_adapter_direct = lambda *a, **k: True
    _lm._load_lora_adapter = lambda *a, **k: _seen2.setdefault("std", True)
    _rc2 = _lm._load_lokr_adapter(_FakePipe(), dict(_lokr_sd), "adap2", "[t]", 1.0)
finally:
    (_lm._load_lokr_adapter_peft, _lm._load_lokr_adapter_direct,
     _lm._load_lora_adapter) = _save
check("wiring: direct-merge success → flatten NOT reached (Flux LoKRs "
      "unaffected)", "std" not in _seen2 and _rc2 is True)


print(f"\n{passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
