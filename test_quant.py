#!/usr/bin/env python3
"""Test harness for fp8 quantize-on-load (ADR-019 slice A).

Covers the Vision-slice invariants (docs/vision/slice-A-fp8-quant-load.md §1),
each with at least one negative case:

  1. Default path unchanged        — mode 'none' → no config, empty cache
                                     fragment (key byte-identical).
  2. VAE never quantized           — refused even via quant_only.
  3. CLIP excluded by default      — addressable only via quant_only.
  4. Quant state in the cache key  — fragments/keys discriminate mode and
                                     component sets.
  5. LoRA tier-3 loud fail         — guard_direct_merge raises actionable
                                     RuntimeError on a quantized base.
  6. Warn-and-fall-back            — unsupported device → None config +
                                     notice, never an exception.

Runs without GPU or loaded models: the guard tests use a fake torchao-
namespaced tensor subclass; the size-gate tests use sparse files (st_size
without disk cost). The one CUDA-positive build test self-skips (as PASS
with a note) on hosts without fp8-capable hardware.
"""

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn

# Load the utils module directly — dodges nodes/__init__.py (which imports
# every node class and expects a ComfyUI environment). Established pattern
# from the other suites.
_spec = importlib.util.spec_from_file_location(
    "edu", Path(__file__).parent / "nodes" / "eric_diffusion_utils.py")
edu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(edu)

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


# ──────────────────────────────────────────────────────────────────────
print("── classify_quant_role (role table from the real model zoo) ───")

_ROLE_CASES = [
    # denoisers
    ("transformer", "QwenImageTransformer2DModel", "denoiser"),
    ("transformer", "FluxTransformer2DModel", "denoiser"),
    ("transformer", "ChromaTransformer2DModel", "denoiser"),
    ("transformer", "SD3Transformer2DModel", "denoiser"),
    ("transformer", "AuraFlowTransformer2DModel", "denoiser"),
    ("transformer", "ZImageTransformer2DModel", "denoiser"),
    ("unet", "UNet2DConditionModel", "denoiser"),
    ("prior", "StableCascadeUNet", "denoiser"),
    ("decoder", "StableCascadeUNet", "denoiser"),
    # large-LM text encoders
    ("text_encoder", "Qwen2_5_VLForConditionalGeneration", "lm"),
    ("text_encoder", "Mistral3ForConditionalGeneration", "lm"),
    ("text_encoder", "Qwen3ForCausalLM", "lm"),
    ("text_encoder", "Qwen3Model", "lm"),
    ("text_encoder", "UMT5EncoderModel", "lm"),
    ("text_encoder_2", "T5EncoderModel", "lm"),
    ("text_encoder_3", "T5EncoderModel", "lm"),
    # CLIP-class
    ("text_encoder", "CLIPTextModel", "clip"),
    ("text_encoder_2", "CLIPTextModelWithProjection", "clip"),
    ("image_encoder", "CLIPVisionModelWithProjection", "clip"),
    # VAE-role (invariant 2 ground truth)
    ("vae", "AutoencoderKL", "vae"),
    ("vae", "AutoencoderKLQwenImage", "vae"),
    ("vae", "AutoencoderKLFlux2", "vae"),
    ("vqgan", "PaellaVQModel", "vae"),
    # non-modules
    ("scheduler", "FlowMatchEulerDiscreteScheduler", "other"),
    ("tokenizer", "Qwen2Tokenizer", "other"),
    ("processor", "Qwen2VLProcessor", "other"),
    ("guider", "AdaptiveProjectedMixGuidance", "other"),
    ("feature_extractor", None, "other"),
]
for name, cls, want in _ROLE_CASES:
    got = edu.classify_quant_role(name, cls)
    check(f"role({name}, {cls}) == {want}", got == want, f"got {got}")


# ──────────────────────────────────────────────────────────────────────
print("── resolve_quant_components (policy + overrides) ──────────────")


def _mk_model_dir(tmp, index, te_sizes=None):
    """Write model_index.json + optional sparse text-encoder weight files."""
    root = Path(tmp)
    with open(root / "model_index.json", "w") as f:
        json.dump(index, f)
    for comp, size in (te_sizes or {}).items():
        d = root / comp
        d.mkdir(exist_ok=True)
        p = d / "model.safetensors"
        with open(p, "wb") as f:
            f.seek(size - 1)
            f.write(b"\0")
    return str(root)


_QWEN_INDEX = {
    "_class_name": "QwenImagePipeline",
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "Qwen2_5_VLForConditionalGeneration"],
    "tokenizer": ["transformers", "Qwen2Tokenizer"],
    "transformer": ["diffusers", "QwenImageTransformer2DModel"],
    "vae": ["diffusers", "AutoencoderKLQwenImage"],
}

_FLUX_INDEX = {
    "_class_name": "FluxPipeline",
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "CLIPTextModel"],
    "text_encoder_2": ["transformers", "T5EncoderModel"],
    "tokenizer": ["transformers", "CLIPTokenizer"],
    "tokenizer_2": ["transformers", "T5TokenizerFast"],
    "transformer": ["diffusers", "FluxTransformer2DModel"],
    "vae": ["diffusers", "AutoencoderKL"],
}

with tempfile.TemporaryDirectory() as tmp:
    # No TE subfolder on disk → size unknown → treated as large (kept).
    mp = _mk_model_dir(tmp, _QWEN_INDEX)
    sel, notes = edu.resolve_quant_components(mp, _QWEN_INDEX)
    check("default set = denoiser + LM text encoder",
          sel == {"transformer": "denoiser", "text_encoder": "lm"},
          f"got {sel}")
    check("vae not in default set (invariant 2)", "vae" not in sel)
    check("unknown TE size treated as large (kept)", "text_encoder" in sel)

with tempfile.TemporaryDirectory() as tmp:
    mp = _mk_model_dir(tmp, _FLUX_INDEX)
    sel, notes = edu.resolve_quant_components(mp, _FLUX_INDEX)
    check("CLIP excluded by default (invariant 3)",
          "text_encoder" not in sel, f"got {sel}")
    check("T5-XXL LM kept alongside", "text_encoder_2" in sel)
    check("CLIP exclusion is announced (notice)",
          any("CLIP" in n for n in notes), f"notes: {notes}")

# Size gate: small TE drops out, big TE stays.
with tempfile.TemporaryDirectory() as tmp:
    mp = _mk_model_dir(tmp, _QWEN_INDEX,
                       te_sizes={"text_encoder": 1 * 1024**3})   # 1 GB
    sel, notes = edu.resolve_quant_components(mp, _QWEN_INDEX)
    check("small (1GB) text encoder excluded by size gate",
          "text_encoder" not in sel, f"got {sel}")
    check("size-gate exclusion is announced",
          any("small" in n for n in notes), f"notes: {notes}")

with tempfile.TemporaryDirectory() as tmp:
    mp = _mk_model_dir(tmp, _QWEN_INDEX,
                       te_sizes={"text_encoder": 3 * 1024**3})   # 3 GB
    sel, _ = edu.resolve_quant_components(mp, _QWEN_INDEX)
    check("large (3GB) text encoder kept by size gate",
          "text_encoder" in sel, f"got {sel}")

# quant_only semantics.
with tempfile.TemporaryDirectory() as tmp:
    mp = _mk_model_dir(tmp, _FLUX_INDEX)
    # NEGATIVE (invariant 2): vae refused even via quant_only.
    sel, notes = edu.resolve_quant_components(mp, _FLUX_INDEX, only=("vae",))
    check("quant_only=vae selects nothing (invariant 2 NEGATIVE)",
          sel == {}, f"got {sel}")
    check("vae refusal notice says NEVER",
          any("NEVER" in n for n in notes), f"notes: {notes}")

    # CLIP is opt-in-able.
    sel, _ = edu.resolve_quant_components(mp, _FLUX_INDEX,
                                          only=("text_encoder",))
    check("quant_only can address CLIP (opt-in)",
          sel == {"text_encoder": "clip"}, f"got {sel}")

    # Unknown + non-module names are refused with notices, not errors.
    sel, notes = edu.resolve_quant_components(
        mp, _FLUX_INDEX, only=("bogus", "scheduler", "transformer"))
    check("quant_only: unknown/non-module refused, valid kept",
          sel == {"transformer": "denoiser"}, f"got {sel}")
    check("quant_only: unknown name announced",
          any("bogus" in n for n in notes), f"notes: {notes}")

    # quant_skip removes from the default set.
    sel, _ = edu.resolve_quant_components(mp, _FLUX_INDEX,
                                          skip=("text_encoder_2",))
    check("quant_skip removes a default-set component",
          sel == {"transformer": "denoiser"}, f"got {sel}")
    sel, notes = edu.resolve_quant_components(mp, _FLUX_INDEX,
                                              skip=("nonexistent",))
    check("quant_skip: unknown name announced, set unchanged",
          "transformer" in sel and any("nonexistent" in n for n in notes),
          f"sel {sel} notes {notes}")

# Non-component top-level lists in model_index.json (Krea-2 regression:
# text_encoder_select_layers is a list of ints and crashed classification).
_KREA2_INDEX = {
    "_class_name": "Krea2Pipeline",
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "Qwen3VLModel"],
    "text_encoder_select_layers": [2, 5, 8, 11],
    "tokenizer": ["transformers", "Qwen2Tokenizer"],
    "transformer": ["diffusers", "Krea2Transformer2DModel"],
    "vae": ["diffusers", "AutoencoderKLQwenImage"],
    "is_distilled": True,
    "patch_size": 2,
}

with tempfile.TemporaryDirectory() as tmp:
    mp = _mk_model_dir(tmp, _KREA2_INDEX)
    sel, notes = edu.resolve_quant_components(mp, _KREA2_INDEX)
    check("Krea-2 index: int-list entry skipped, no crash",
          sel == {"transformer": "denoiser", "text_encoder": "lm"},
          f"got {sel}")
    check("Krea-2 index: select_layers never classified",
          "text_encoder_select_layers" not in sel, f"got {sel}")
    # Null class_name in a component pair (optional components like
    # safety_checker ship as [null, null]) stays non-crashing → 'other'.
    _NULLCOMP = dict(_FLUX_INDEX, safety_checker=[None, None])
    sel, _ = edu.resolve_quant_components(mp, _NULLCOMP)
    check("null-class component pair tolerated, not selected",
          "safety_checker" not in sel and "transformer" in sel, f"got {sel}")

# Slice DQ (daemon quant carriage): the machine boundary duplicates
# QUANT_MODES in params_validation so the daemon's validation path never
# imports this torch-heavy module (security review slice-DQ F1). Pin the sync.
from comfyless.params_validation import QUANT_MODES as _PV_QUANT_MODES
check("QUANT_MODES boundary constant in sync with eric_diffusion_utils",
      tuple(_PV_QUANT_MODES) == tuple(edu.QUANT_MODES),
      f"params_validation {_PV_QUANT_MODES!r} vs utils {edu.QUANT_MODES!r}")

# Slice DQ review F5: a separatorless '..' slips the slot-name validator but
# must stay inert — dict miss with an unknown-component notice, never a
# filesystem path.
with tempfile.TemporaryDirectory() as tmp:
    mp = _mk_model_dir(tmp, _FLUX_INDEX)
    sel, notes = edu.resolve_quant_components(mp, _FLUX_INDEX, only=("..",))
    check("quant_only='..' inert: nothing selected, unknown-component notice",
          sel == {} and any(".." in n and "unknown" in n for n in notes),
          f"sel {sel} notes {notes}")


# ──────────────────────────────────────────────────────────────────────
print("── build_quant_config (mode gate + hardware fallback) ─────────")

with tempfile.TemporaryDirectory() as tmp:
    mp = _mk_model_dir(tmp, _QWEN_INDEX)

    # Invariant 1: mode none → no config at all.
    cfg, comps, notes = edu.build_quant_config(mp, "none")
    check("mode 'none' → (None, {}, []) (invariant 1)",
          cfg is None and comps == {} and notes == [])

    # NEGATIVE: unknown mode raises (caller bug, not environment).
    try:
        edu.build_quant_config(mp, "int3")
        check("unknown mode raises ValueError (NEGATIVE)", False, "no raise")
    except ValueError as e:
        check("unknown mode raises ValueError (NEGATIVE)", "int3" in str(e))

    # Invariant 6 NEGATIVE: cpu device → warn-and-fall-back, never raise.
    cfg, comps, notes = edu.build_quant_config(mp, "fp8", device="cpu")
    check("fp8 on cpu → None config (fallback, no exception)",
          cfg is None and comps == {})
    check("cpu fallback is announced (invariant 6)",
          any("FALLING BACK" in n for n in notes), f"notes: {notes}")

    # CUDA positive — self-skips on non-fp8 hardware.
    _fp8_capable = (torch.cuda.is_available()
                    and torch.cuda.get_device_capability(0) >= (8, 9))
    if _fp8_capable:
        cfg, comps, notes = edu.build_quant_config(mp, "fp8", device="cuda")
        check("fp8 on cuda → PipelineQuantizationConfig",
              type(cfg).__name__ == "PipelineQuantizationConfig",
              f"got {type(cfg).__name__}")
        check("mapping covers exactly the selected components",
              set(cfg.quant_mapping) == set(comps), f"{set(cfg.quant_mapping)}")
        _te_cfg = cfg.quant_mapping.get("text_encoder")
        _tr_cfg = cfg.quant_mapping.get("transformer")
        check("transformers-lib component gets transformers TorchAoConfig",
              _te_cfg is not None
              and type(_te_cfg).__module__.startswith("transformers"),
              f"got {type(_te_cfg).__module__}")
        check("diffusers-lib component gets diffusers TorchAoConfig",
              _tr_cfg is not None
              and type(_tr_cfg).__module__.startswith("diffusers"),
              f"got {type(_tr_cfg).__module__}")
    else:
        for _n in ("fp8 on cuda → PipelineQuantizationConfig",
                   "mapping covers exactly the selected components",
                   "transformers-lib component gets transformers TorchAoConfig",
                   "diffusers-lib component gets diffusers TorchAoConfig"):
            check(f"{_n} (SKIPPED: no fp8-capable GPU)", True)


# ──────────────────────────────────────────────────────────────────────
print("── cache-key discrimination (invariants 1 + 4) ────────────────")

frag_none = edu.quant_cache_fragment("none")
frag_fp8 = edu.quant_cache_fragment("fp8")
check("fragment('none') is empty — default cache key byte-identical "
      "(invariant 1)", frag_none == "")
check("fragment('fp8') is non-empty", frag_fp8 != "")

_base = "/m/qwen_bf16_cuda_False_False_False"
check("bf16 vs fp8 loader keys differ (invariant 4)",
      _base + frag_none != _base + frag_fp8)
# NEGATIVE for invariant 4: same quant config → same key (no spurious miss).
check("identical quant config → identical fragment (no spurious eviction)",
      edu.quant_cache_fragment("fp8", skip=("a", "b"))
      == edu.quant_cache_fragment("fp8", skip=("b", "a")))
check("different skip sets → different fragments",
      edu.quant_cache_fragment("fp8", skip=("text_encoder",)) != frag_fp8)
check("different only sets → different fragments",
      edu.quant_cache_fragment("fp8", only=("transformer",)) != frag_fp8)
check("skip vs only same names → different fragments",
      edu.quant_cache_fragment("fp8", skip=("x",))
      != edu.quant_cache_fragment("fp8", only=("x",)))


# ──────────────────────────────────────────────────────────────────────
print("── LoRA guard under quant (invariant 5) ───────────────────────")


# Stand-ins for a torchao-quantized module. Detection keys off
# type(param.data).__module__ — a real torchao subclass survives
# nn.Parameter's .data getter via __torch_dispatch__ (proven live on GPU),
# but a naive as_subclass fake gets stripped, so the fake is duck-typed:
# is_quantized_module only touches .parameters() and .data.
class _FakeQuantData:
    pass


_FakeQuantData.__module__ = "torchao.quantization.fake_for_test"


class _FakeParam:
    def __init__(self):
        self.data = _FakeQuantData()


class _FakeQuantModule:
    def parameters(self):
        return iter([_FakeParam()])


_plain = nn.Linear(4, 4)
_quant = _FakeQuantModule()

check("is_quantized_module: plain module → False",
      edu.is_quantized_module(_plain) is False)
check("is_quantized_module: torchao-namespaced param → True",
      edu.is_quantized_module(_quant) is True)
check("is_quantized_module: None → False",
      edu.is_quantized_module(None) is False)

# Positive: guard is a no-op on an unquantized base.
try:
    edu.guard_direct_merge(_plain, "[t]", "LoKR adapter")
    check("guard no-ops on unquantized base", True)
except RuntimeError:
    check("guard no-ops on unquantized base", False, "raised unexpectedly")

# NEGATIVE (invariant 5): guard raises loud + actionable on quantized base.
try:
    edu.guard_direct_merge(_quant, "[t]", "LoKR adapter")
    check("guard raises on quantized base (invariant 5 NEGATIVE)",
          False, "did not raise")
except RuntimeError as e:
    msg = str(e)
    check("guard raises on quantized base (invariant 5 NEGATIVE)", True)
    check("guard message is actionable (--quant + PEFT + ADR-019)",
          "--quant" in msg and "PEFT" in msg and "ADR-019" in msg,
          f"msg: {msg[:120]}")

# Slice DMR (ADR-019 §4 amendment): the four direct-merge functions route
# EVERY weight write through the apply_merge_delta dispatcher, which owns
# the raise for unmergeable quantized reps — the entry-guard protection
# MOVED into the dispatcher (Vision invariant 7 / security review req 24).
# Source inspection so a refactor can't silently reintroduce a raw write.
_lora_src = (Path(__file__).parent / "nodes" / "eric_qwen_edit_lora.py").read_text()
_conv_src = (Path(__file__).parent / "nodes"
             / "eric_lora_format_convert_apply.py").read_text()
for fn, src in [("_load_lokr_adapter_direct", _lora_src),
                ("_load_loha_adapter_direct", _lora_src),
                ("_load_lora_adapter_direct", _lora_src),
                ("_apply_converted_lora_as_delta", _conv_src)]:
    body = src.split(f"def {fn}(")[1].split("\ndef ")[0]
    check(f"{fn} routes merges through apply_merge_delta (DMR)",
          "apply_merge_delta(" in body)
    check(f"{fn} has no direct param.data.add_ (req 24 NEGATIVE)",
          "param.data.add_" not in body)
    check(f"{fn} uses the filtered merge_resolution_map (req 23)",
          "merge_resolution_map(" in body)


# ──────────────────────────────────────────────────────────────────────
print("── machine boundary: quant fields (ADR-012 hygiene) ───────────")

import comfyless.generate  # noqa: F401,E402 — installs shims
from comfyless.params_validation import validate_machine_request  # noqa: E402

_REQ = {"type": "generate", "model": "/m", "prompt": "p"}

r = validate_machine_request({**_REQ, "quant": "fp8"})
check("quant str accepted", r.ok, f"{r.error}")
r = validate_machine_request({**_REQ, "quant": 1})
check("quant non-str rejected (NEGATIVE)",
      not r.ok and r.error["field"] == "quant", f"{r.error}")

r = validate_machine_request({**_REQ, "quant_skip": ["text_encoder"]})
check("quant_skip bare slot names accepted", r.ok, f"{r.error}")
r = validate_machine_request({**_REQ, "quant_skip": "text_encoder"})
check("quant_skip non-list rejected (NEGATIVE)", not r.ok, f"{r.error}")
r = validate_machine_request({**_REQ, "quant_skip": [3]})
check("quant_skip non-str entry rejected (NEGATIVE)",
      not r.ok and "quant_skip[0]" in r.error["field"], f"{r.error}")
r = validate_machine_request({**_REQ, "quant_only": ["../vae"]})
check("quant_only path-shaped entry rejected (NEGATIVE)",
      not r.ok and "not paths" in r.error["reason"], f"{r.error}")
r = validate_machine_request({**_REQ, "quant_only": ["a\x00b"]})
check("quant_only NUL entry rejected (NEGATIVE)",
      not r.ok and "NUL" in r.error["reason"], f"{r.error}")
r = validate_machine_request({**_REQ, "quant_skip": ["b\\c"]})
check("quant_skip backslash entry rejected (NEGATIVE)", not r.ok, f"{r.error}")
r = validate_machine_request({**_REQ, "quant_skip": [f"s{i}" for i in range(33)]})
check("quant_skip >32 entries rejected (NEGATIVE, reviewer F3)",
      not r.ok and "too many" in r.error["reason"], f"{r.error}")
r = validate_machine_request({**_REQ, "quant_only": [f"s{i}" for i in range(32)]})
check("quant_only exactly 32 entries accepted (cap boundary)", r.ok,
      f"{r.error}")


# ──────────────────────────────────────────────────────────────────────
print("── MCP surface: schema + pipeline cache key (invariant 4) ─────")

from comfyless.mcp_server import (  # noqa: E402
    _GENERATE_INPUT_SCHEMA, _pipeline_cache_key)

_props = _GENERATE_INPUT_SCHEMA["properties"]
check("MCP schema declares quant enum (derived from QUANT_MODES)",
      _props.get("quant", {}).get("enum") == list(_PV_QUANT_MODES))
check("MCP schema declares quant_skip as string array",
      _props.get("quant_skip", {}).get("items", {}).get("type") == "string")
check("MCP schema declares quant_only as string array",
      _props.get("quant_only", {}).get("items", {}).get("type") == "string")
check("MCP schema bounds quant lists (maxItems, reviewer F3)",
      _props["quant_skip"].get("maxItems") == 32
      and _props["quant_only"].get("maxItems") == 32)
check("MCP schema still rejects unknown fields",
      _GENERATE_INPUT_SCHEMA["additionalProperties"] is False)

_k_none = _pipeline_cache_key("/m", "", False, [], "none", (), ())
_k_fp8 = _pipeline_cache_key("/m", "", False, [], "fp8", (), ())
check("MCP cache key: none vs fp8 differ (invariant 4)", _k_none != _k_fp8)
check("MCP cache key: default arg equals explicit none (back-compat)",
      _pipeline_cache_key("/m", "", False, []) == _k_none)
check("MCP cache key: same quant config → same key (NEGATIVE: no spurious "
      "eviction)",
      _pipeline_cache_key("/m", "", False, [], "fp8", ("a", "b"), ())
      == _pipeline_cache_key("/m", "", False, [], "fp8", ("b", "a"), ()))
check("MCP cache key: skip sets discriminate",
      _pipeline_cache_key("/m", "", False, [], "fp8", ("text_encoder",), ())
      != _k_fp8)


# ──────────────────────────────────────────────────────────────────────
print("\n── family-aware fp8 recipe (weight-only vs dyn-activation) ────")

# Z-Image-base needs weight-only fp8: dynamic per-tensor activation quant
# destroys its output (speckle→NaN, confirmed 2026-07-10). Everything else,
# incl. the distilled zimage-turbo and an unknown family, stays on the fast
# dynamic-activation path so no verified-good family regresses.
check("recipe: zimage -> weight_only",
      edu._fp8_recipe_for_family("zimage") == "weight_only")
check("recipe: zimage-turbo -> dynamic_activation (fast path kept)",
      edu._fp8_recipe_for_family("zimage-turbo") == "dynamic_activation")
for _fam in ("qwen-image", "flux", "flux2", "krea", "krea-turbo", "sdxl",
             "chroma", None):
    check(f"recipe: {_fam!r} -> dynamic_activation (unchanged)",
          edu._fp8_recipe_for_family(_fam) == "dynamic_activation")

# The config objects match the recipe. Both are torchao configs; weight-only
# is a distinct class (no activation quant).
from torchao.quantization import (                      # noqa: E402
    Float8DynamicActivationFloat8WeightConfig as _DynAct,
    Float8WeightOnlyConfig as _WOnly,
)
check("config: zimage -> Float8WeightOnlyConfig",
      isinstance(edu._torchao_fp8_config("zimage"), _WOnly))
check("config: zimage-turbo -> DynamicActivation",
      isinstance(edu._torchao_fp8_config("zimage-turbo"), _DynAct))
check("config: default (None) -> DynamicActivation (back-compat)",
      isinstance(edu._torchao_fp8_config(), _DynAct))
check("config: unknown family -> DynamicActivation",
      isinstance(edu._torchao_fp8_config("nonesuch"), _DynAct))

# The merge-path consistency contract (eric_diffusion_fp8_ops._merge_into_
# torchao): it picks the recipe by reading `act_quant_kwargs` off the base
# tensor, so it must be True that the two configs differ ONLY there and agree
# on the stored fp8 weight. Prove the discriminator exists and splits the two.
import torch.nn as _nn                                  # noqa: E402
from torchao.quantization import quantize_ as _q        # noqa: E402


def _akw(cfg):
    m = _nn.Linear(64, 128, bias=False)
    _q(m, cfg)
    return getattr(m.weight, "act_quant_kwargs", "MISSING")


check("merge-discriminator: weight-only base has act_quant_kwargs=None",
      _akw(edu._torchao_fp8_config("zimage")) is None)
# The dyn-activation recipe hits torchao's _check_hardware_support (CUDA cc
# ≥8.9 / MI300+ / XPU) inside quantize_ — self-skip on incapable hardware
# (e.g. CI runners), same pattern as the fp8 CUDA-positive block above.
if (torch.cuda.is_available()
        and torch.cuda.get_device_capability(0) >= (8, 9)):
    check("merge-discriminator: dyn-activation base has act_quant_kwargs set",
          _akw(edu._torchao_fp8_config("flux")) is not None)
else:
    check("merge-discriminator: dyn-activation base has act_quant_kwargs set "
          "(SKIPPED: no fp8-capable GPU)", True)


# ──────────────────────────────────────────────────────────────────────
print("── slice NV: nvfp4 quantize-on-load (ADR-019, vision slice-NV) ─")

# Invariant 5: the mode exists at both boundaries (the sync test above
# already pins params_validation == utils; this pins the VALUE).
check("nvfp4 in QUANT_MODES", "nvfp4" in edu.QUANT_MODES,
      f"{edu.QUANT_MODES!r}")

# ── Invariant 2: hardware gate — nvfp4 needs Blackwell (>= 10.0), fp8
# keeps its >= 8.9 floor. Fake the CUDA probe so this runs anywhere.
_real_avail = torch.cuda.is_available
_real_cap = torch.cuda.get_device_capability
try:
    torch.cuda.is_available = lambda: True
    torch.cuda.get_device_capability = lambda idx=0: (8, 9)
    ok_fp8, _ = edu.quant_hardware_ok("fp8", "cuda")
    ok_nv, why_nv = edu.quant_hardware_ok("nvfp4", "cuda")
    check("cap 8.9: fp8 accepted (gate unchanged, invariant 1)", ok_fp8)
    check("cap 8.9: nvfp4 REJECTED (NEGATIVE, invariant 2)", not ok_nv,
          f"ok={ok_nv}")
    check("cap 8.9 nvfp4 rejection names the mode and the 10.0 floor",
          "nvfp4" in why_nv and "10.0" in why_nv, f"why: {why_nv}")
    torch.cuda.get_device_capability = lambda idx=0: (12, 0)
    ok_nv, _ = edu.quant_hardware_ok("nvfp4", "cuda")
    ok_fp8, _ = edu.quant_hardware_ok("fp8", "cuda:1")
    check("cap 12.0 (Blackwell sm_120): nvfp4 accepted", ok_nv)
    check("cap 12.0: fp8 accepted on indexed device", ok_fp8)
finally:
    torch.cuda.is_available = _real_avail
    torch.cuda.get_device_capability = _real_cap

ok_cpu, why_cpu = edu.quant_hardware_ok("nvfp4", "cpu")
check("nvfp4 on cpu rejected, message names the mode (NEGATIVE)",
      not ok_cpu and "nvfp4" in why_cpu, f"why: {why_cpu}")

# ── Invariant 4: recipe/config selection mirrors the fp8 family split.
from torchao.prototype.mx_formats import (               # noqa: E402
    NVFP4DynamicActivationNVFP4WeightConfig as _NVDynAct,
    NVFP4WeightOnlyConfig as _NVWOnly,
)
check("nvfp4 config: zimage -> NVFP4WeightOnlyConfig (family gate)",
      isinstance(edu._torchao_nvfp4_config("zimage"), _NVWOnly))
check("nvfp4 config: zimage-turbo -> DynamicActivation (fast path kept)",
      isinstance(edu._torchao_nvfp4_config("zimage-turbo"), _NVDynAct))
_nv_default = edu._torchao_nvfp4_config()
check("nvfp4 config: default (None) -> DynamicActivation",
      isinstance(_nv_default, _NVDynAct))
check("nvfp4 dynamic config routes through the mslk triton kernel",
      getattr(_nv_default, "use_triton_kernel", False) is True)

# ── Per-mode weight-only split (2026-07-17): dynamic-activation nvfp4
# destroyed QwenImage output in the T1 live smoke (pervasive granular
# noise) while fp8 dyn-act on the same transformer is bf16-indistinguishable
# — so qwen-image is weight-only under nvfp4 ONLY. The negative pins that
# the split is per-mode, not shared.
check("nvfp4 config: qwen-image -> NVFP4WeightOnlyConfig (T1 noise fix)",
      isinstance(edu._torchao_nvfp4_config("qwen-image"), _NVWOnly))
check("fp8 config: qwen-image stays DynamicActivation (NEGATIVE — split "
      "is per-mode)",
      isinstance(edu._torchao_fp8_config("qwen-image"), _DynAct))
check("recipe selector: ('nvfp4', 'qwen-image') -> weight_only",
      edu._quant_recipe_for_family("nvfp4", "qwen-image") == "weight_only")
check("recipe selector: ('fp8', 'qwen-image') -> dynamic_activation "
      "(NEGATIVE)",
      edu._quant_recipe_for_family("fp8", "qwen-image")
      == "dynamic_activation")
check("recipe selector: zimage weight_only under BOTH modes",
      edu._quant_recipe_for_family("fp8", "zimage") == "weight_only"
      and edu._quant_recipe_for_family("nvfp4", "zimage") == "weight_only")
check("recipe selector: qwen-edit unaffected (dyn-act both modes — no "
      "evidence yet)",
      edu._quant_recipe_for_family("nvfp4", "qwen-edit")
      == "dynamic_activation")
try:
    edu._quant_recipe_for_family("gguf", "qwen-image")
    check("recipe selector: unknown mode raises, never a silent fp8-set "
          "default (NEGATIVE)", False, "no raise")
except ValueError as _e:
    check("recipe selector: unknown mode raises, never a silent fp8-set "
          "default (NEGATIVE)", "gguf" in str(_e))

# Real CPU quantize under the new mapping: qwen-image nvfp4 is weight-only
# (act_quant_kwargs None), same CI-safe shape as the zimage arm above.
_mqw = _nn.Linear(64, 128, bias=False, dtype=torch.bfloat16)
check("quantize_module nvfp4/qwen-image: quantizes (returns True)",
      edu.quantize_module(_mqw, "nvfp4", family="qwen-image") is True)
check("quantize_module nvfp4/qwen-image: weight-only "
      "(act_quant_kwargs None)",
      type(_mqw.weight.data).__name__ == "NVFP4Tensor"
      and getattr(_mqw.weight.data, "act_quant_kwargs", "MISSING") is None,
      f"got {type(_mqw.weight.data).__name__}")

# Dispatcher: fp8 requests still get EXACTLY the fp8 classes (invariant 1).
check("dispatcher: ('fp8', zimage) -> Float8WeightOnlyConfig (unchanged)",
      isinstance(edu._torchao_quant_config("fp8", "zimage"), _WOnly))
check("dispatcher: ('fp8', None) -> Float8 DynamicActivation (unchanged)",
      isinstance(edu._torchao_quant_config("fp8", None), _DynAct))
check("dispatcher: ('nvfp4', None) -> NVFP4 DynamicActivation",
      isinstance(edu._torchao_quant_config("nvfp4", None), _NVDynAct))
try:
    edu._torchao_quant_config("gguf")
    check("dispatcher: unknown mode raises, never a silent fp8 default "
          "(NEGATIVE)", False, "no raise")
except ValueError as _e:
    check("dispatcher: unknown mode raises, never a silent fp8 default "
          "(NEGATIVE)", "gguf" in str(_e))

# Invariant 1: the fp8-facing strings render byte-identically to the
# pre-slice-NV hardcoded forms (mode-aware f-strings, pinned exactly).
_ok_cpu, _why = edu.quant_hardware_ok("fp8", "cpu")
check("fp8 cpu message byte-identical to pre-NV",
      _why == "fp8 quantization requires a CUDA device", f"got {_why!r}")
_real_avail = torch.cuda.is_available
_real_cap = torch.cuda.get_device_capability
try:
    torch.cuda.is_available = lambda: True
    torch.cuda.get_device_capability = lambda idx=0: (8, 0)
    _ok, _why = edu.quant_hardware_ok("fp8", "cuda")
    check("fp8 capability message byte-identical to pre-NV",
          _why == "fp8 GEMM needs compute capability >= 8.9, device has 8.0",
          f"got {_why!r}")
finally:
    torch.cuda.is_available = _real_avail
    torch.cuda.get_device_capability = _real_cap

import contextlib as _ctx                                # noqa: E402
import io as _io                                         # noqa: E402
_buf = _io.StringIO()
_mfp8 = _nn.Linear(64, 128, bias=False, dtype=torch.bfloat16)
with _ctx.redirect_stdout(_buf):
    edu.quantize_module(_mfp8, "fp8", family="zimage")
check("fp8 recipe notice byte-identical to pre-NV",
      "quant: fp8 recipe=weight_only (family 'zimage')" in _buf.getvalue(),
      f"got {_buf.getvalue()!r}")

# ── Invariant 3 NEGATIVE: missing mslk surfaces as ImportError from the
# config builder (probed BEFORE torchao's quantize-time AssertionError can
# kill a from_pretrained mid-load) and as warn-and-fall-back from
# build_quant_config.
import importlib.util as _ilu_mod                        # noqa: E402
_real_find_spec = _ilu_mod.find_spec


def _hide_mslk(name, *a, **k):
    if name == "mslk":
        return None
    return _real_find_spec(name, *a, **k)


try:
    _ilu_mod.find_spec = _hide_mslk
    try:
        edu._torchao_nvfp4_config()
        check("mslk hidden: _torchao_nvfp4_config raises ImportError "
              "(NEGATIVE)", False, "no raise")
    except ImportError as e:
        check("mslk hidden: _torchao_nvfp4_config raises ImportError "
              "(NEGATIVE)", "mslk" in str(e), f"msg: {e}")
    # Through build_quant_config: environment problem → None config +
    # loud notice, never an exception (invariant 6 of slice A extends).
    _real_hw = edu.quant_hardware_ok
    edu.quant_hardware_ok = lambda m, d: (True, "")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            mp = _mk_model_dir(tmp, _QWEN_INDEX)
            cfg, comps, notes = edu.build_quant_config(mp, "nvfp4")
            check("mslk hidden: build_quant_config falls back to "
                  "unquantized (NEGATIVE)", cfg is None and comps == {},
                  f"cfg {cfg} comps {comps}")
            check("mslk hidden: fallback is announced",
                  any("FALLING BACK" in n for n in notes), f"notes: {notes}")
    finally:
        edu.quant_hardware_ok = _real_hw
finally:
    _ilu_mod.find_spec = _real_find_spec

# ── quantize_module routes by mode (override path) — real CPU quantize;
# both nvfp4 recipes quantize weights on CPU (kernel choice matters only
# at forward time), verified 2026-07-16.
_m = _nn.Linear(64, 128, bias=False, dtype=torch.bfloat16)
check("quantize_module nvfp4/zimage: quantizes (returns True)",
      edu.quantize_module(_m, "nvfp4", family="zimage") is True)
check("quantize_module nvfp4/zimage: weight is NVFP4Tensor",
      type(_m.weight.data).__name__ == "NVFP4Tensor",
      f"got {type(_m.weight.data).__name__}")
check("quantize_module nvfp4/zimage: weight-only (act_quant_kwargs None)",
      getattr(_m.weight.data, "act_quant_kwargs", "MISSING") is None)
check("nvfp4-quantized module detected by is_quantized_module "
      "(LoRA PEFT-path gate)", edu.is_quantized_module(_m) is True)

# The dynamic-activation arm quantized fine on the CUDA-visible dev box
# (2026-07-16) but may hardware-probe on a GPU-less CI runner, where
# quantize_module's warn-and-skip contract returns False — self-skip there,
# same pattern as the fp8 dyn-act arm above.
_m2 = _nn.Linear(64, 128, bias=False, dtype=torch.bfloat16)
_ok2 = edu.quantize_module(_m2, "nvfp4", family=None)
if not _ok2 and not torch.cuda.is_available():
    check("quantize_module nvfp4 default family: quantizes "
          "(SKIPPED: no CUDA — torchao declined the dyn-act recipe)", True)
    check("quantize_module nvfp4 default: dynamic-activation "
          "(SKIPPED: no CUDA)", True)
else:
    check("quantize_module nvfp4 default family: quantizes (returns True)",
          _ok2 is True)
    check("quantize_module nvfp4 default: dynamic-activation "
          "(act_quant_kwargs set)",
          getattr(_m2.weight.data, "act_quant_kwargs", None) is not None)

# The merge-guard rationale, pinned (slice NV invariant 6): NVFP4Tensor
# ALSO carries act_quant_kwargs (None when weight-only), so the fp8_ops
# recipe sniff CANNOT discriminate fp8 from nvfp4 — the class gate in
# _requant_config_matching_base is load-bearing. If torchao ever drops
# the attribute this check flags the API drift before the merge path
# sees it (mirrors the fp8 attribute pin above).
check("NVFP4Tensor carries act_quant_kwargs (why the merge guard is a "
      "CLASS check, not an akw sniff)",
      hasattr(_m.weight.data, "act_quant_kwargs"))

# ── Invariant 7: cache-key fragments discriminate nvfp4 from fp8.
frag_nv = edu.quant_cache_fragment("nvfp4")
check("fragment('nvfp4') is non-empty", frag_nv != "")
check("fragment('nvfp4') != fragment('fp8') (no cross-mode cache hit)",
      frag_nv != edu.quant_cache_fragment("fp8"))


# ──────────────────────────────────────────────────────────────────────
print("── nvfp4 shape screen (16-block constraint, 2026-07-17 fix) ───")
# torchao hard-raises mid-from_pretrained on Linear weights whose last two
# dims aren't divisible by 16 (first live hit: Qwen2.5-VL vision tower,
# hidden 3420). The screen reads safetensors HEADERS ONLY and routes
# offenders through modules_to_not_convert; fp8 is deliberately unscreened.

from safetensors.torch import save_file as _st_save      # noqa: E402

_BAD_UP = "visual.blocks.0.mlp.up_proj.weight"     # [20, 16] → 20 % 16 != 0
_BAD_DOWN = "visual.blocks.0.mlp.down_proj.weight"  # [16, 20]
_GOOD = "layers.0.self_attn.q_proj.weight"          # [32, 16] → clean


def _mk_te_safetensors(comp_dir, extra=None):
    comp_dir.mkdir(parents=True, exist_ok=True)
    tensors = {
        _GOOD: torch.zeros(32, 16, dtype=torch.bfloat16),
        _BAD_UP: torch.zeros(20, 16, dtype=torch.bfloat16),
        _BAD_DOWN: torch.zeros(16, 20, dtype=torch.bfloat16),
        # Never flagged: non-2-D and non-".weight" entries are outside
        # torchao's Linear-only conversion even when their dims violate /16.
        "norm.weight": torch.zeros(20, dtype=torch.bfloat16),
        "patch.conv.weight": torch.zeros(20, 16, 3, 3, dtype=torch.bfloat16),
        "some.scale": torch.zeros(20, 20, dtype=torch.bfloat16),
    }
    tensors.update(extra or {})
    _st_save(tensors, str(comp_dir / "model.safetensors"))


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    _mk_te_safetensors(root / "text_encoder")
    bad, scanned, nfail = edu._nvfp4_incompatible_weight_keys(
        str(root), "text_encoder")
    check("scan flags exactly the 2-D .weight offenders (sorted)",
          bad == sorted([_BAD_UP, _BAD_DOWN]), f"got {bad}")
    check("scan counts the readable header, no failures",
          scanned == 1 and nfail == 0, f"got {scanned}, {nfail}")

    # Sharded component: keys union across files, both headers counted.
    _st_save({"shard2.linear.weight": torch.zeros(16, 20, dtype=torch.bfloat16)},
             str(root / "text_encoder" / "model-00002.safetensors"))
    bad2, scanned2, _ = edu._nvfp4_incompatible_weight_keys(
        str(root), "text_encoder")
    check("sharded scan unions offenders across files",
          bad2 == sorted([_BAD_UP, _BAD_DOWN, "shard2.linear.weight"]),
          f"got {bad2}")
    check("sharded scan counts both headers", scanned2 == 2, f"got {scanned2}")

    # Partial failure: one good shard + one garbage shard → failures
    # reported so callers warn about an INCOMPLETE screen (review finding 2).
    with open(root / "text_encoder" / "model-00003.safetensors", "wb") as f:
        f.seek(4096 - 1)
        f.write(b"\0")
    bad2b, scanned2b, nfail2b = edu._nvfp4_incompatible_weight_keys(
        str(root), "text_encoder")
    check("partial failure: good shards still scanned, failure counted "
          "(NEGATIVE)", scanned2b == 2 and nfail2b == 1 and bad2b == bad2,
          f"got {scanned2b}, {nfail2b}, {bad2b}")

    # Unreadable file (sparse garbage, the _mk_model_dir shape): screen
    # reports 0 scanned so callers WARN instead of assuming clean (NEGATIVE).
    garb = root / "text_encoder_garbage"
    garb.mkdir()
    with open(garb / "model.safetensors", "wb") as f:
        f.seek(4096 - 1)
        f.write(b"\0")
    bad3, scanned3, nfail3 = edu._nvfp4_incompatible_weight_keys(
        str(root), "text_encoder_garbage")
    check("unreadable header → no keys, 0 scanned, 1 failed (NEGATIVE)",
          bad3 == [] and scanned3 == 0 and nfail3 == 1,
          f"got {bad3}, {scanned3}, {nfail3}")

    check("missing component dir → no keys, no reads (NEGATIVE)",
          edu._nvfp4_incompatible_weight_keys(str(root), "nope")
          == ([], 0, 0))

# ── Wiring through build_quant_config (hardware mocked — CI-safe).
_real_hw = edu.quant_hardware_ok
setattr(edu, "quant_hardware_ok", lambda m, d: (True, ""))
try:
    with tempfile.TemporaryDirectory() as tmp:
        mp = _mk_model_dir(tmp, _QWEN_INDEX)
        root = Path(mp)
        _mk_te_safetensors(root / "text_encoder")
        (root / "transformer").mkdir()
        _st_save({"blocks.0.ff.weight": torch.zeros(32, 16, dtype=torch.bfloat16)},
                 str(root / "transformer" / "model.safetensors"))

        cfg, comps, notes = edu.build_quant_config(
            mp, "nvfp4", only=("text_encoder", "transformer"))
        check("nvfp4 build: config built with both components",
              cfg is not None and sorted(comps) == ["text_encoder",
                                                    "transformer"],
              f"comps {comps}")
        assert cfg is not None
        _te_cfg = cfg.quant_mapping["text_encoder"]
        _tr_cfg = cfg.quant_mapping["transformer"]
        check("nvfp4 build: offenders land in the TE's "
              "modules_to_not_convert (full keys incl. .weight)",
              _te_cfg.modules_to_not_convert == sorted([_BAD_UP, _BAD_DOWN]),
              f"got {_te_cfg.modules_to_not_convert}")
        check("nvfp4 build: clean component gets NO exclusions",
              _tr_cfg.modules_to_not_convert is None,
              f"got {_tr_cfg.modules_to_not_convert}")
        check("nvfp4 build: exclusion is announced in notices",
              any("not divisible by 16" in n and "text_encoder" in n
                  for n in notes), f"notes: {notes}")

        # fp8 on the SAME dir: unscreened — no exclusions, no notice
        # (NEGATIVE: proves the screen is nvfp4-scoped).
        cfg8, comps8, notes8 = edu.build_quant_config(
            mp, "fp8", only=("text_encoder", "transformer"))
        assert cfg8 is not None
        check("fp8 build on same weights: no modules_to_not_convert "
              "(NEGATIVE)",
              cfg8.quant_mapping["text_encoder"].modules_to_not_convert
              is None,
              f"got {cfg8.quant_mapping['text_encoder'].modules_to_not_convert}")
        check("fp8 build: no divisibility notice (NEGATIVE)",
              not any("divisible by 16" in n for n in notes8),
              f"notes: {notes8}")

        # Partial shard failure through the build: incomplete screen must
        # WARN even though other shards scanned fine (review finding 2).
        with open(root / "text_encoder" / "model-00009.safetensors",
                  "wb") as f:
            f.seek(4096 - 1)
            f.write(b"\0")
        cfgp, _, notesp = edu.build_quant_config(
            mp, "nvfp4", only=("text_encoder", "transformer"))
        check("nvfp4 build: partial shard failure → incomplete-screen WARN, "
              "config still built (NEGATIVE)",
              cfgp is not None
              and any("could not read 1 of the" in n for n in notesp),
              f"notes: {notesp}")

    # Family-detection failure under nvfp4: post-split this silently lands
    # on dynamic-activation — the known-bad recipe for qwen-image — so it
    # must WARN (review finding 2). Detection succeeds on the fake dirs
    # (it reads model_index.json), so force the failure.
    _real_detect = edu.detect_pipeline_class

    def _detect_boom(path):
        raise RuntimeError("forced detection failure")

    setattr(edu, "detect_pipeline_class", _detect_boom)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            mp = _mk_model_dir(tmp, _QWEN_INDEX)
            root = Path(tmp)
            (root / "transformer").mkdir()
            _st_save({"blocks.0.ff.weight": torch.zeros(
                          32, 16, dtype=torch.bfloat16)},
                     str(root / "transformer" / "model.safetensors"))
            cfgf, _, notesf = edu.build_quant_config(
                mp, "nvfp4", only=("transformer",))
            check("nvfp4 build: family-detection failure → loud routing "
                  "WARN (NEGATIVE)",
                  cfgf is not None
                  and any("weight-only routing not honored" in n
                          for n in notesf), f"notes: {notesf}")
            cfgf8, _, notesf8 = edu.build_quant_config(
                mp, "fp8", only=("transformer",))
            check("fp8 build: no routing WARN on detection failure "
                  "(NEGATIVE — fp8 recipe never depended on it here)",
                  not any("routing not honored" in n for n in notesf8),
                  f"notes: {notesf8}")
    finally:
        setattr(edu, "detect_pipeline_class", _real_detect)

    # Unscannable component under nvfp4: loud warning, load proceeds
    # (warn-don't-block — invariant 6).
    with tempfile.TemporaryDirectory() as tmp:
        mp = _mk_model_dir(tmp, _QWEN_INDEX,
                           te_sizes={"text_encoder": 4096})  # sparse garbage
        cfg, comps, notes = edu.build_quant_config(
            mp, "nvfp4", only=("text_encoder",))
        check("nvfp4 build: unreadable headers → WARN notice, config still "
              "built (warn-don't-block)",
              cfg is not None
              and any("could not read any safetensors header" in n
                      for n in notes),
              f"cfg {cfg} notes: {notes}")
finally:
    setattr(edu, "quant_hardware_ok", _real_hw)

# ── quantize_module (override path): same screen as a live filter_fn.
# Weight-only recipe quantizes on CPU (established above) — CI-safe.
_scr = _nn.Sequential(
    _nn.Linear(16, 16, bias=False, dtype=torch.bfloat16),   # clean
    _nn.Linear(10, 16, bias=False, dtype=torch.bfloat16),   # weight [16,10]
)
_buf = _io.StringIO()
with _ctx.redirect_stdout(_buf):
    _ok_scr = edu.quantize_module(_scr, "nvfp4", family="zimage")
check("quantize_module nvfp4: quantizes despite a bad-shape Linear "
      "(returns True)", _ok_scr is True)
check("quantize_module nvfp4: clean Linear quantized",
      type(_scr[0].weight.data).__name__ == "NVFP4Tensor",
      f"got {type(_scr[0].weight.data).__name__}")
check("quantize_module nvfp4: bad-shape Linear left in full precision",
      type(_scr[1].weight.data).__name__ != "NVFP4Tensor"
      and _scr[1].weight.dtype == torch.bfloat16,
      f"got {type(_scr[1].weight.data).__name__}")
check("quantize_module nvfp4: skip is announced with the module fqn",
      "not divisible by 16" in _buf.getvalue()
      and "(e.g. 1)" in _buf.getvalue(), f"got {_buf.getvalue()!r}")

# fp8 on the same bad shape: NOT screened — fp8 rowwise has no /16
# constraint and the validated path must stay byte-identical (NEGATIVE).
_m_bad8 = _nn.Linear(10, 16, bias=False, dtype=torch.bfloat16)
check("quantize_module fp8: bad-shape Linear still quantizes (NEGATIVE — "
      "screen is nvfp4-scoped)",
      edu.quantize_module(_m_bad8, "fp8", family="zimage") is True
      and "Float8" in type(_m_bad8.weight.data).__name__,
      f"got {type(_m_bad8.weight.data).__name__}")


# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print(f"  {passed} passed, {failed} failed")
print("─" * 50)
sys.exit(1 if failed else 0)
