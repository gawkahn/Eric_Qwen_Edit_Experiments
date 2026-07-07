#!/usr/bin/env python3
"""Unit tests for the Krea-2 LoRA conversion plan.

Exercises detection, plan matching, and the rename math on synthetic
state dicts (no real LoRA file, no GPU, no model load). Mirrors the
empirical verification done against MysticXXX_KREA2_v1.safetensors:
ai-toolkit/ComfyUI-native Krea keys → diffusers Krea2 layout.

Run: ./.venv/bin/python3 test_lora_convert_krea.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import comfyless  # noqa: F401  installs folder_paths / comfy shims
import torch

from nodes.eric_lora_format_convert import detect_lora_format
from nodes.eric_lora_format_convert_apply import find_matching_plan, convert_state_dict


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


def _lora_pair(base, out_dim=8, in_dim=16, rank=4):
    """Return {base.lora_A.weight, base.lora_B.weight} with small tensors."""
    return {
        f"{base}.lora_A.weight": torch.zeros(rank, in_dim),
        f"{base}.lora_B.weight": torch.zeros(out_dim, rank),
    }


# A representative ai-toolkit Krea-2 native state dict: one main block
# (all attn projections + gate + mlp) and one text-fusion block.
def _make_krea_sd():
    sd = {}
    for sub in ("wq", "wk", "wv", "wo", "gate"):
        sd.update(_lora_pair(f"diffusion_model.blocks.0.attn.{sub}"))
    for sub in ("gate", "up", "down"):
        sd.update(_lora_pair(f"diffusion_model.blocks.0.mlp.{sub}"))
    for sub in ("wq", "wk", "wv", "wo", "gate"):
        sd.update(_lora_pair(f"diffusion_model.txtfusion.layerwise_blocks.0.attn.{sub}"))
    # refiner_blocks (the other text-fusion sub-list) + an mlp module, so
    # both txtfusion sub-names and FF-under-txtfusion are exercised.
    for sub in ("wq", "wk", "wv", "wo", "gate"):
        sd.update(_lora_pair(f"diffusion_model.txtfusion.refiner_blocks.0.attn.{sub}"))
    for sub in ("gate", "up", "down"):
        sd.update(_lora_pair(f"diffusion_model.txtfusion.refiner_blocks.0.mlp.{sub}"))
    return sd


KREA_MODEL_PARAMS = [
    "transformer_blocks.0.attn.to_q.weight",
    "transformer_blocks.0.attn.to_gate.weight",
    "transformer_blocks.0.ff.gate.weight",
    "text_fusion.layerwise_blocks.0.attn.to_q.weight",
]


# ── Detection ──────────────────────────────────────────────────────────
print("── detect_lora_format ─────────────────────────────────────────")
krea_sd = _make_krea_sd()
check("krea-native keys detect as 'krea_native'",
      detect_lora_format(krea_sd.keys()) == "krea_native")

# Negative: a diffusers-DiT LoRA (transformer_blocks/to_q) must NOT be
# misread as krea_native.
diffusers_sd = _lora_pair("transformer_blocks.0.attn.to_q")
check("diffusers-DiT keys do NOT detect as krea_native (negative)",
      detect_lora_format(diffusers_sd.keys()) != "krea_native")

# Negative: an SD-UNet LoRA likewise.
sd_unet = _lora_pair("down_blocks.0.attentions.0.to_q")
check("sd-unet keys do NOT detect as krea_native (negative)",
      detect_lora_format(sd_unet.keys()) != "krea_native")

# Negative: the lexically closest neighbor — BFL `img_attn.qkv`. The
# `.attn.wq` marker must NOT fire on `qkv` (this is the regression that
# would break if the marker were ever loosened to `.attn.q`).
bfl = _lora_pair("double_blocks.0.img_attn.qkv")
check("bfl-original keys do NOT detect as krea_native (negative)",
      detect_lora_format(bfl.keys()) != "krea_native")

# Fallback path: a txtfusion-less, transformer-blocks-only Krea LoRA must
# still detect via the `.attn.wq` marker (not just via `txtfusion`).
blocks_only = _lora_pair("diffusion_model.blocks.0.attn.wq")
check("blocks-only Krea LoRA detects via .attn.wq fallback",
      detect_lora_format(blocks_only.keys()) == "krea_native")


# ── Plan matching ──────────────────────────────────────────────────────
print("\n── find_matching_plan ─────────────────────────────────────────")
plan = find_matching_plan(krea_sd, KREA_MODEL_PARAMS)
check("krea LoRA + krea model → krea plan matched",
      plan is not None and plan.target_family == "diffusers_krea")

# Negative: model_signature 'to_gate' absent → no match (don't apply a
# Krea plan to a non-Krea model).
plan_wrong_model = find_matching_plan(
    krea_sd, ["transformer_blocks.0.attn.to_q.weight"]  # no to_gate
)
check("krea LoRA + non-krea model (no to_gate) → no match (negative)",
      plan_wrong_model is None)


# ── fp8-resident buffer visibility (regression: LoRAs failed on the
#    scaled-fp8 Krea 'redcraft' checkpoint) ─────────────────────────────
# ScaledFp8Linear registers its `.weight` as a BUFFER (not a Parameter)
# and bias-free projections (Krea attn `to_gate`) carry no Parameter at
# all. The conversion hook in load_lora_with_key_fix builds the model-side
# name list the plan's `to_gate` signature is matched against — if it
# enumerates named_parameters() ONLY, the fp8 weights are invisible, the
# plan never matches, and every LoRA falls through to a 0-module merge.
# The fix adds named_buffers(). These checks pin that behavior.
from torch import nn


class _Fp8Stub(nn.Module):
    """Mirrors ScaledFp8Linear storage: weight is a buffer, no bias param."""
    def __init__(self, out_dim=16, in_dim=16):
        super().__init__()
        self.register_buffer("weight", torch.zeros(out_dim, in_dim))
        self.register_buffer("weight_scale", torch.ones(()))
        self.bias = None


def _build_fp8_krea_model():
    """A minimal diffusers-Krea-named module whose Linears are fp8 stubs."""
    attn = nn.Module()
    for name in ("to_q", "to_k", "to_v", "to_gate"):
        setattr(attn, name, _Fp8Stub())
    attn.to_out = nn.ModuleList([_Fp8Stub()])
    ff = nn.Module()
    for name in ("gate", "up", "down"):
        setattr(ff, name, _Fp8Stub())
    block = nn.Module()
    block.attn, block.ff = attn, ff
    model = nn.Module()
    model.transformer_blocks = nn.ModuleList([block])
    return model


print("\n── fp8-resident buffer visibility ─────────────────────────────")
# Drive the ACTUAL production helpers (not an inline reconstruction) so a
# revert of either buffer-aware fix fails this suite.
from nodes.eric_qwen_edit_lora import plan_match_model_names
from nodes.eric_lora_format_convert_apply import (
    resolve_merge_target, mergeable_target_names,
)

fp8_model = _build_fp8_krea_model()
params_only = [n for n, _ in fp8_model.named_parameters()]
plan_names = plan_match_model_names(fp8_model)  # production name list

check("fp8 weights hidden from named_parameters (to_gate absent)",
      not any("to_gate" in n for n in params_only),
      f"unexpectedly present: {[n for n in params_only if 'to_gate' in n][:3]}")
check("plan_match_model_names surfaces the fp8 to_gate weight",
      any("to_gate" in n for n in plan_names))

# Fix 1 — plan matching (eric_qwen_edit_lora.load_lora_with_key_fix):
# production name list → plan matches; params-only → None (the bug).
check("fp8-resident model + plan_match_model_names → krea plan matched",
      (lambda p: p is not None and p.target_family == "diffusers_krea")(
          find_matching_plan(krea_sd, plan_names)))
check("fp8-resident model + params-only → no match (reproduces the bug)",
      find_matching_plan(krea_sd, params_only) is None)

# Fix 2 — the load_converted_lora pre-filter drops any base whose weight
# isn't resolvable. It now builds its name set via mergeable_target_names
# (the ACTUAL production seam — reverting it to named_parameters() fails
# these checks). A bias-free ScaledFp8Linear (weight = buffer, no bias
# Parameter) must resolve there but NOT against named_parameters() alone —
# otherwise the standard-LoRA Krea path (lora_A/lora_B) is silently dropped.
merge_names = mergeable_target_names(fp8_model)  # production name set
gate_base = "transformer_blocks.0.attn.to_gate"
check("bias-free fp8 to_gate resolvable via mergeable_target_names (fix 2)",
      resolve_merge_target(merge_names, gate_base) is not None)
check("bias-free fp8 to_gate NOT resolvable via params-only (the bug)",
      resolve_merge_target(set(params_only), gate_base) is None)
# DMR-3 guard: the scale buffer must never be a merge target.
check("scale buffer excluded from mergeable_target_names (DMR-3 guard)",
      not any(n.endswith("weight_scale") for n in merge_names))


# ── Conversion / rename math ───────────────────────────────────────────
print("\n── convert_state_dict rename ──────────────────────────────────")
conv = convert_state_dict(krea_sd, plan, log_prefix="[test]")
bases = {k.rsplit(".lora_", 1)[0] for k in conv}

# Pure rename (qkv_splits=[]) must preserve key count exactly — a
# regression that attached a split spec would change the count.
check("pure rename preserves key count (no split)",
      len(conv) == len(krea_sd),
      f"{len(conv)} != {len(krea_sd)}")

expected = {
    "transformer_blocks.0.attn.to_q",
    "transformer_blocks.0.attn.to_k",
    "transformer_blocks.0.attn.to_v",
    "transformer_blocks.0.attn.to_out.0",
    "transformer_blocks.0.attn.to_gate",
    "transformer_blocks.0.ff.gate",
    "transformer_blocks.0.ff.up",
    "transformer_blocks.0.ff.down",
    "text_fusion.layerwise_blocks.0.attn.to_q",
    "text_fusion.layerwise_blocks.0.attn.to_out.0",
    "text_fusion.layerwise_blocks.0.attn.to_gate",
    # refiner_blocks sub-name + FF-under-txtfusion (.mlp.→.ff.)
    "text_fusion.refiner_blocks.0.attn.to_q",
    "text_fusion.refiner_blocks.0.ff.gate",
}
for e in sorted(expected):
    check(f"renamed → {e}", e in bases, f"missing; got sample {sorted(bases)[:3]}")

# No native token survives the rename.
native_tokens = (".wq", ".wk", ".wv", ".wo", ".mlp.", "diffusion_model",
                 "txtfusion", ".attn.gate")
leftover = [b for b in bases if any(t in b for t in native_tokens)]
check("no native tokens remain after rename (negative)",
      not leftover, f"leftover: {leftover[:5]}")

# text_fusion's layerwise_blocks sub-name is preserved (the bare 'blocks.'
# substring must not have been corrupted into 'transformer_blocks.').
check("text_fusion sub-block name preserved (layerwise_blocks intact)",
      any("text_fusion.layerwise_blocks.0" in b for b in bases))
check("no corrupted 'layerwise_transformer_blocks' (negative)",
      not any("layerwise_transformer_blocks" in b for b in bases))


print("\n──────────────────────────────────────────────────")
print(f"  {passed} passed, {failed} failed")
print("──────────────────────────────────────────────────")
sys.exit(0 if failed == 0 else 1)
