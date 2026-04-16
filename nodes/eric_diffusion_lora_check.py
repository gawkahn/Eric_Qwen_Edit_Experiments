# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Diffusion LoRA Compatibility Checker

Validates a LoRA's key names and tensor dimensions against a loaded transformer
(for in-loader use) or a model directory's safetensors headers (standalone scan).

No imports from other eric_ modules — safe to import from anywhere.

Author: Eric Hiss (GitHub: EricRollei)
"""

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Adapter suffix table (mirrors _SUFFIXES in eric_qwen_edit_lora) ──────────

_ADAPTER_SUFFIXES = (
    # Longer/more-specific suffixes first to avoid partial matches
    ".lora_A.weight", ".lora_B.weight",
    ".lora_down.weight", ".lora_up.weight",
    # No-.weight variants (some training tools omit the .weight suffix)
    ".lora_A", ".lora_B",
    ".lora_down", ".lora_up",
    ".lokr_w1", ".lokr_w2", ".lokr_t2",
    ".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b",
    ".alpha", ".diff", ".diff_b",
)

# Layers that exist as model parameters but that PEFT won't inject LoRA into
# (not in diffusers' default target_modules list for Flux/Chroma).
_NORM_PATTERNS = ("norm.linear", "norm1.linear", "norm1_context.linear",
                  "norm_out.linear")


def _strip_adapter_suffix(key: str) -> Tuple[str, Optional[str]]:
    """Return (base_module_path, matched_suffix) with adapter suffix removed."""
    for sfx in _ADAPTER_SUFFIXES:
        idx = key.find(sfx)
        if idx >= 0:
            return key[:idx], sfx
    return key.rsplit(".", 1)[0] if "." in key else key, None


def _read_safetensors_header(path) -> Dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class LoRACheckResult:
    path: str
    total_layers: int   # unique target layers in the LoRA
    matched: int        # layers whose param exists in the model
    dim_ok: int         # matched layers with correct in/out dimensions
    norm_layers: int    # matched layers in norm.linear (PEFT will drop these)
    unmatched: int      # layers not found in model at all
    dim_mismatches: List[str] = field(default_factory=list)  # base keys with bad dims
    skipped: bool = False  # set True when loader skips this LoRA
    arch_hint: Optional[str] = None  # human-readable format detection for 0% matches

    @property
    def key_match_pct(self) -> float:
        return 100 * self.matched / self.total_layers if self.total_layers else 0.0

    @property
    def dim_ok_pct(self) -> float:
        return 100 * self.dim_ok / self.matched if self.matched else 0.0

    @property
    def verdict(self) -> str:
        if self.key_match_pct < 10:
            return "WRONG_ARCH"
        if self.key_match_pct < 50:
            return "POOR_MATCH"
        if self.dim_ok_pct < 90:
            return "DIM_MISMATCH"
        if self.norm_layers > 0:
            return "NORM_TARGETING"
        return "OK"

    def log_lines(self, prefix: str = "[LoRA-Check]") -> List[str]:
        name = Path(self.path).name
        lines = [
            f"{prefix} {name}: "
            f"keys={self.matched}/{self.total_layers} ({self.key_match_pct:.0f}%)  "
            f"dims={self.dim_ok_pct:.0f}%  verdict={self.verdict}"
        ]
        if self.verdict == "WRONG_ARCH":
            # Prefer the architecture-specific hint computed at check
            # time over the old generic "likely SD1/SDXL" message — that
            # was misdiagnosing original-format Flux/Klein LoRAs as a
            # different model family entirely.
            hint_text = self.arch_hint or (
                "Architecture mismatch — LoRA targets modules not "
                "present in the loaded model. Will produce no effect."
            )
            import textwrap
            wrapped = textwrap.wrap(hint_text, width=70)
            for i, segment in enumerate(wrapped):
                marker = "!! " if i == 0 else "   "
                lines.append(f"{prefix}   {marker}{segment}")
        elif self.verdict == "POOR_MATCH":
            lines.append(
                f"{prefix}   !! Only {self.matched}/{self.total_layers} layers "
                "found in model — possible wrong model family"
            )
        if self.norm_layers > 0:
            lines.append(
                f"{prefix}   !! {self.norm_layers} norm.linear target(s): "
                "PEFT fast-path silently drops these — use direct-merge path for full effect"
            )
        if self.dim_mismatches:
            lines.append(
                f"{prefix}   !! {len(self.dim_mismatches)} dimension mismatch(es) — "
                "LoRA trained on different model size?"
            )
            for k in self.dim_mismatches[:3]:
                lines.append(f"{prefix}      {k}")
        if self.skipped:
            lines.append(f"{prefix}   !! LoRA was SKIPPED (below compatibility threshold)")
        return lines


# ── Core check logic ──────────────────────────────────────────────────────────

def _build_param_dict_from_transformer(transformer) -> Dict[str, Tuple]:
    """Build {param_name: shape} from a loaded nn.Module.

    Normalises PEFT-wrapped parameter names: when a LoRA has already been
    loaded, PEFT replaces nn.Linear with a wrapper and named_parameters()
    returns 'foo.base_layer.weight' instead of 'foo.weight'.  Strip
    '.base_layer.' so the checker works against both clean and LoRA-loaded
    transformers.
    """
    pd = {}
    for n, p in transformer.named_parameters():
        norm = n.replace(".base_layer.", ".")
        pd[norm] = tuple(p.shape)
    pd["_norm_lookup"] = {k.replace(".", "_"): k for k in pd if not k.startswith("_")}
    return pd


def build_param_dict_from_dir(model_dir: str) -> Dict[str, Tuple]:
    """Build {param_name: shape} by reading safetensors headers from a model directory.

    Filters to transformer component keys only (strips 'transformer.' prefix).
    Fast — reads headers only, no weight data loaded.
    """
    pd = {}
    for shard in sorted(Path(model_dir).glob("*.safetensors")):
        header = _read_safetensors_header(shard)
        for k, v in header.items():
            shape = tuple(v["shape"])
            # Normalise: strip 'transformer.' prefix to match named_parameters() format
            norm_k = k[len("transformer."):] if k.startswith("transformer.") else k
            pd[norm_k] = shape
    pd["_norm_lookup"] = {k.replace(".", "_"): k for k in pd if not k.startswith("_")}
    print(f"[LoRA-Check] Indexed {len(pd)-1} params from {model_dir}")
    return pd


def _resolve_to_param_key(lora_base_key: str, param_dict: Dict) -> Optional[str]:
    """Map a LoRA base key to a model parameter key. Returns None if not found."""
    norm_lookup = param_dict.get("_norm_lookup", {})

    # Diffusers-style: has dots in the key
    if "." in lora_base_key:
        for prefix in ("transformer.", "diffusion_model.", "model.diffusion_model.", ""):
            if lora_base_key.startswith(prefix):
                candidate = lora_base_key[len(prefix):] + ".weight"
                if candidate in param_dict:
                    return candidate
        return None

    # Kohya-style: all underscores, no dots
    # Common prefixes after stripping 'lora_': transformer_, unet_, diffusion_model_
    stripped = lora_base_key[5:] if lora_base_key.startswith("lora_") else lora_base_key
    candidates = [stripped]
    for model_pfx in ("transformer_", "unet_", "diffusion_model_", "model_diffusion_model_"):
        if stripped.startswith(model_pfx):
            candidates.append(stripped[len(model_pfx):])
    for s in candidates:
        norm_key = s + "_weight"
        if norm_key in norm_lookup:
            return norm_lookup[norm_key]
    return None


def check_lora(lora_path, transformer=None, param_dict: Optional[Dict] = None,
               log_prefix: str = "[LoRA-Check]") -> LoRACheckResult:
    """Check a LoRA file's compatibility against a model.

    Provide either:
      transformer  — loaded nn.Module (pipe.transformer), fastest for in-loader use
      param_dict   — pre-built dict from build_param_dict_from_dir(), for batch scanning
    """
    if param_dict is None:
        if transformer is None:
            raise ValueError("Provide transformer or param_dict")
        param_dict = _build_param_dict_from_transformer(transformer)

    header = _read_safetensors_header(lora_path)

    # Group header keys by target layer, collect down/up shapes
    layers: Dict[str, Dict] = {}
    for k, v in header.items():
        base, sfx = _strip_adapter_suffix(k)
        if sfx is None:
            continue
        shape = tuple(v["shape"])
        slot = ("down" if any(t in sfx for t in ("lora_A", "lora_down", "hada_w1_a"))
                else "up" if any(t in sfx for t in ("lora_B", "lora_up", "hada_w1_b"))
                else "other")
        layers.setdefault(base, {})[slot] = shape

    matched = dim_ok = norm_count = unmatched = 0
    dim_mismatch_keys = []

    for base_key, parts in layers.items():
        model_key = _resolve_to_param_key(base_key, param_dict)
        if model_key is None:
            unmatched += 1
            continue

        matched += 1

        # Norm-targeting check
        if any(pat in model_key for pat in _NORM_PATTERNS):
            norm_count += 1
            dim_ok += 1  # parameter exists, dims are fine, but PEFT drops it
            continue

        model_shape = param_dict[model_key]
        if ("down" in parts and "up" in parts and len(model_shape) >= 2
                and len(parts["down"]) >= 2):
            lora_in  = parts["down"][1]
            lora_out = parts["up"][0]
            if lora_in == model_shape[-1] and lora_out == model_shape[0]:
                dim_ok += 1
            else:
                dim_mismatch_keys.append(
                    f"{base_key}: LoRA ({lora_in}→{lora_out}) vs model {model_shape}"
                )
        else:
            dim_ok += 1  # not enough info to check

    # When nothing matched, detect what architecture this LoRA actually
    # targets so we can give a more accurate hint than the generic
    # "likely SD1/SDXL".  Pattern-match on the LoRA's stripped key
    # paths AND the model's parameter paths to disambiguate "wrong
    # arch entirely" vs "right family but wrong layout convention."
    arch_hint = None
    if matched == 0 and layers:
        sample_lora = list(layers.keys())[:3]
        sample_model = [k for k in param_dict if not k.startswith("_")][:3]
        print(f"{log_prefix} 0% match — sample LoRA base keys:  {sample_lora}")
        print(f"{log_prefix} 0% match — sample model param keys: {sample_model}")

        # Architecture detection from LoRA's own key patterns.
        all_lora_paths = list(layers.keys())
        has_double_blocks = any("double_blocks" in p for p in all_lora_paths)
        has_single_blocks = any("single_blocks" in p for p in all_lora_paths)
        has_unet_blocks = any(
            ("up_blocks" in p) or ("down_blocks" in p) or ("mid_block" in p)
            for p in all_lora_paths
        )
        has_diffusers_flux = any(
            ("transformer_blocks" in p) or ("single_transformer_blocks" in p)
            for p in all_lora_paths
        )
        # Model side detection: do we KNOW we're loading into a diffusers
        # Flux/Flux2 model that has the reorganized structure?  If yes,
        # the user's "original Flux/Klein LoRA" needs conversion rather
        # than being plain incompatible.
        all_model_paths = [k for k in param_dict if not k.startswith("_")]
        model_is_diffusers_flux = any(
            ("transformer_blocks" in p) or ("single_transformer_blocks" in p)
            for p in all_model_paths
        )

        if (has_double_blocks or has_single_blocks) and model_is_diffusers_flux:
            arch_hint = (
                "Original Flux/Klein/Chroma LoRA format detected "
                "(targets `double_blocks`/`single_blocks`).  This is the "
                "ComfyUI-native layout — the diffusers model loaded here "
                "uses the reorganized `transformer_blocks` / "
                "`single_transformer_blocks` layout, with the fused QKV "
                "projection split into separate to_q/to_k/to_v.  The "
                "LoRA is NOT incompatible with the underlying model "
                "weights, just with the diffusers module structure.  "
                "load_lora_with_key_fix will attempt automatic in-memory "
                "conversion via a registered ConversionPlan (see "
                "nodes/eric_lora_format_convert*.py); this 0% pre-check "
                "result is expected and will be followed by a converted "
                "load attempt.  If you also see a 'Converted adapter "
                "loaded' line, the LoRA is active.  If conversion is "
                "skipped (no plan available), no plan is registered for "
                "this (LoRA family, model family) pair yet."
            )
        elif has_double_blocks or has_single_blocks:
            arch_hint = (
                "Flux/Klein/Chroma LoRA (original format with "
                "`double_blocks`/`single_blocks`).  The currently-loaded "
                "model doesn't have the matching module names — confirm "
                "you've loaded the right model family for this LoRA."
            )
        elif has_diffusers_flux and not model_is_diffusers_flux:
            arch_hint = (
                "Diffusers-format Flux/Flux2 LoRA, but the loaded model "
                "doesn't have the matching `transformer_blocks` modules.  "
                "Likely loaded into a non-Flux model family by mistake."
            )
        elif has_unet_blocks:
            arch_hint = (
                "SD1/SDXL/SD3-style UNet LoRA (targets `up_blocks` / "
                "`down_blocks` / `mid_block`).  Not compatible with the "
                "Flux/Qwen-family transformer architecture — wrong model "
                "type entirely."
            )
        else:
            arch_hint = (
                "Unrecognized LoRA architecture.  Sample target keys "
                "above don't match any known pattern (Flux/Klein/Chroma "
                "original or diffusers, SD UNet).  Could be a custom "
                "fine-tune format or a newly-supported architecture not "
                "yet recognized by this checker."
            )

    result = LoRACheckResult(
        path=str(lora_path),
        total_layers=len(layers),
        matched=matched,
        dim_ok=dim_ok,
        norm_layers=norm_count,
        unmatched=unmatched,
        dim_mismatches=dim_mismatch_keys,
        arch_hint=arch_hint,
    )

    return result


# ── Standalone scanner entry point ────────────────────────────────────────────

def _run_scanner():
    """CLI: python -m nodes.eric_diffusion_lora_check --base DIR [--hd DIR] loras..."""
    import argparse, sys

    ap = argparse.ArgumentParser(
        description="Check LoRA compatibility against Chroma/Flux model variants."
    )
    ap.add_argument("--base", metavar="DIR", help="Path to chroma1-base (or any model dir)")
    ap.add_argument("--hd",   metavar="DIR", help="Path to chroma1-hd (or other variant)")
    ap.add_argument("loras", nargs="+", help="LoRA safetensors files to check")
    args = ap.parse_args()

    models = {}
    if args.base:
        models["base"] = build_param_dict_from_dir(args.base)
    if args.hd:
        models["hd"]   = build_param_dict_from_dir(args.hd)
    if not models:
        ap.error("Provide at least --base or --hd")

    model_names = list(models)
    col_w = 40
    header = f"{'LoRA':<{col_w}}" + "".join(
        f"  {'keys%':>8}{'dims%':>8}{'verdict':>14}" for _ in model_names
    )
    subhdr = " " * col_w + "".join(
        f"  {n:>8}{'':>8}{'':>14}" for n in model_names
    )
    print()
    print(header)
    print(subhdr)
    print("-" * len(header))

    for lp in sorted(args.loras):
        p = Path(lp)
        row = f"{p.name[:col_w]:<{col_w}}"
        for n, pd in models.items():
            try:
                r = check_lora(p, param_dict=pd)
                row += f"  {r.key_match_pct:>7.1f}%{r.dim_ok_pct:>7.1f}%{r.verdict:>14}"
            except Exception as e:
                row += f"  {'ERR':>8}{'':>8}{str(e)[:14]:>14}"
        print(row)
        # Print warnings indented under each row
        for n, pd in models.items():
            try:
                r = check_lora(p, param_dict=pd)
                for line in r.log_lines(prefix=f"  [{n}]")[1:]:  # skip first summary line
                    print(line)
            except Exception:
                pass


if __name__ == "__main__":
    _run_scanner()
