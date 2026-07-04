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
