# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Krea-2 conversion-plan tests (krea_native → diffusers_krea).

Covers the block/txtfusion renames plus the standalone-module mappings
added for distillation LoRAs (krea2_turbo_lora_rank_64_bf16: `first`,
`last.linear`, `tmlp.*`, `tproj.1`, `txtmlp.*`) and the loud .diff_b
bias-delta drop. Standalone targets are verified against a REAL (tiny)
`Krea2Transformer2DModel` instance, so an upstream module rename breaks
these tests instead of silently un-mapping the turbo LoRA again.

Run standalone: ./.venv/bin/python3 tests/test_lora_format_convert_krea.py
"""

from __future__ import annotations

import io
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path


# ── Project root on sys.path + ComfyUI shims (mirror lora_test_harness) ─

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "folder_paths" not in sys.modules:
    fp = types.ModuleType("folder_paths")
    fp.get_folder_paths = lambda *_a, **_k: []
    fp.get_full_path = lambda *_a, **_k: None
    sys.modules["folder_paths"] = fp

if "comfy" not in sys.modules:
    sys.modules["comfy"] = types.ModuleType("comfy")
if "comfy.utils" not in sys.modules:
    cu = types.ModuleType("comfy.utils")

    class _NoopProgressBar:
        def __init__(self, *_a, **_k): pass
        def update(self, *_a, **_k): pass
        def update_absolute(self, *_a, **_k): pass

    cu.ProgressBar = _NoopProgressBar
    sys.modules["comfy.utils"] = cu
    sys.modules["comfy"].utils = cu


import torch  # noqa: E402

from comfyless.core.eric_lora_format_convert import (  # noqa: E402
    apply_rename_rules,
    get_plan,
)
from comfyless.core.eric_lora_format_convert_apply import (  # noqa: E402
    convert_state_dict,
)


_PLAN = get_plan("krea_native", "diffusers_krea")


# ── Standalone-module mapping (must mirror the plan AND upstream's
# diffusers 0.39.0 _convert_non_diffusers_krea2_lora_to_diffusers) ──────
_STANDALONE_MAP = {
    "diffusion_model.first":                "img_in",
    "diffusion_model.last.linear":          "final_layer.linear",
    "diffusion_model.tmlp.0":               "time_embed.linear_1",
    "diffusion_model.tmlp.2":               "time_embed.linear_2",
    "diffusion_model.tproj.1":              "time_mod_proj",
    "diffusion_model.txtmlp.1":             "txt_in.linear_1",
    "diffusion_model.txtmlp.3":             "txt_in.linear_2",
    "diffusion_model.txtfusion.projector":  "text_fusion.projector",
}


def _tiny_krea2_transformer():
    """Instantiate a real Krea2Transformer2DModel at toy size (CPU).

    Module NAMES are size-independent, so this gives authoritative
    ground truth for rename targets without loading real weights.
    """
    from diffusers.models.transformers.transformer_krea2 import (
        Krea2Transformer2DModel,
    )
    return Krea2Transformer2DModel(
        in_channels=4,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
        timestep_embed_dim=8,
        text_hidden_dim=16,
        num_text_layers=3,
        text_num_attention_heads=2,
        text_num_key_value_heads=2,
        text_intermediate_size=32,
        num_layerwise_text_blocks=1,
        num_refiner_text_blocks=1,
        axes_dims_rope=(2, 3, 3),
    )


def test_krea_plan_is_registered():
    assert _PLAN is not None, "krea_native → diffusers_krea plan missing"
    assert _PLAN.qkv_splits == [], "Krea-2 must not declare qkv splits"


def test_block_attn_and_mlp_renames():
    sd = {
        "diffusion_model.blocks.3.attn.wq.lora_A.weight": 0,
        "diffusion_model.blocks.3.attn.wo.lora_B.weight": 0,
        "diffusion_model.blocks.3.attn.gate.lora_A.weight": 0,
        "diffusion_model.blocks.3.mlp.gate.lora_A.weight": 0,
    }
    renamed = apply_rename_rules(sd, _PLAN)
    assert "transformer_blocks.3.attn.to_q.lora_A.weight" in renamed
    assert "transformer_blocks.3.attn.to_out.0.lora_B.weight" in renamed
    assert "transformer_blocks.3.attn.to_gate.lora_A.weight" in renamed
    assert "transformer_blocks.3.ff.gate.lora_A.weight" in renamed


def test_txtfusion_renames():
    sd = {
        "diffusion_model.txtfusion.layerwise_blocks.0.attn.wk.lora_A.weight": 0,
        "diffusion_model.txtfusion.refiner_blocks.1.mlp.down.lora_B.weight": 0,
        "diffusion_model.txtfusion.projector.lora_A.weight": 0,
    }
    renamed = apply_rename_rules(sd, _PLAN)
    assert "text_fusion.layerwise_blocks.0.attn.to_k.lora_A.weight" in renamed
    assert "text_fusion.refiner_blocks.1.ff.down.lora_B.weight" in renamed
    assert "text_fusion.projector.lora_A.weight" in renamed


def test_standalone_module_renames():
    # The turbo-LoRA regression: these 7 were dropped as unresolvable.
    for src, tgt in _STANDALONE_MAP.items():
        sd = {f"{src}.lora_A.weight": 0, f"{src}.diff_b": 0}
        renamed = apply_rename_rules(sd, _PLAN)
        assert f"{tgt}.lora_A.weight" in renamed, \
            f"{src} did not rename to {tgt} (got {sorted(renamed)})"
        assert f"{tgt}.diff_b" in renamed, \
            f"{src}.diff_b suffix lost in rename"


def test_mlp_rule_does_not_touch_tmlp_txtmlp():
    # NEGATIVE: the generic `.mlp.` → `.ff.` rule must not corrupt the
    # standalone tmlp/txtmlp names (no leading dot before their 'mlp').
    sd = {
        "diffusion_model.tmlp.0.lora_A.weight": 0,
        "diffusion_model.txtmlp.1.lora_A.weight": 0,
    }
    renamed = apply_rename_rules(sd, _PLAN)
    assert not any(".ff." in k for k in renamed), sorted(renamed)


def test_standalone_targets_exist_on_real_krea2_model():
    # Ground truth: every mapped target must be a real Linear on the
    # diffusers model (catches upstream renames of these modules).
    model = _tiny_krea2_transformer()
    params = {n for n, _ in model.named_parameters()}
    for tgt in _STANDALONE_MAP.values():
        assert f"{tgt}.weight" in params, f"{tgt}.weight not on Krea2 model"
        # every standalone module in the turbo file except the projector
        # and blocks has a bias (they ship .diff_b for it)
    for tgt in ("img_in", "final_layer.linear", "time_embed.linear_1",
                "time_embed.linear_2", "time_mod_proj",
                "txt_in.linear_1", "txt_in.linear_2"):
        assert f"{tgt}.bias" in params, f"{tgt}.bias not on Krea2 model"


def test_turbo_shaped_dict_converts_fully_with_loud_diff_b_warning():
    # Miniature turbo LoRA: one block module + all 7 standalones with
    # lora_A/B pairs AND .diff_b bias deltas. Conversion must emit a
    # lora pair per module (nothing skipped) and WARN about the dropped
    # bias deltas rather than losing them silently.
    r, d_in, d_out = 2, 8, 8
    sd = {}
    modules = ["diffusion_model.blocks.0.attn.wq"] + [
        s for s in _STANDALONE_MAP if not s.endswith("projector")]
    for m in modules:
        sd[f"{m}.lora_A.weight"] = torch.zeros(r, d_in)
        sd[f"{m}.lora_B.weight"] = torch.ones(d_out, r)
        if m != "diffusion_model.blocks.0.attn.wq":
            sd[f"{m}.diff_b"] = torch.ones(d_out)
    buf = io.StringIO()
    with redirect_stdout(buf):
        out = convert_state_dict(sd, _PLAN)
    log = buf.getvalue()
    out_bases = {k.rsplit(".lora_", 1)[0] for k in out if ".lora_" in k}
    expected = {"transformer_blocks.0.attn.to_q"} | {
        _STANDALONE_MAP[m] for m in modules[1:]}
    assert out_bases == expected, f"got {sorted(out_bases)}"
    assert "bias delta" in log and "WARNING" in log, \
        f".diff_b drop not surfaced loudly; log was: {log!r}"


def _run_all() -> int:
    funcs = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for n, f in funcs:
        try:
            f()
            print(f"  PASS   {n}")
        except AssertionError as e:
            failed.append(n)
            print(f"  FAIL   {n}: {e}")
        except Exception as e:
            failed.append(n)
            print(f"  FAIL   {n}: {type(e).__name__}: {e}")
    print(f"\n{len(funcs) - len(failed)}/{len(funcs)} tests passed.")
    if failed:
        print(f"Failures: {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(_run_all())
