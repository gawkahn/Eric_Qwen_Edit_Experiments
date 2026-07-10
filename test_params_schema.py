#!/usr/bin/env python3
"""Test harness for the comfyless params schema + validator + adapters.

Exercises the schema-focused logic added by the params-schema refactor
(2026-04-24):

  - COMFYLESS_SCHEMA self-consistency (every key used, CLI map valid).
  - _validate_params drop-unknown + keep-but-warn-on-type-mismatch.
  - _extract_eric_save_params emits canonical keys only, drops garbage.
  - _extract_comfyui_params emits canonical keys only from a synthetic
    graph.
  - _load_sidecar round-trip on the shipped qwen_image_hello_world.json
    example.

Runs without ComfyUI, GPU, or loaded diffusion models — the comfyless
package installs its own shims for folder_paths / comfy.utils so the
module imports cleanly.
"""

import io
import json
import sys
from contextlib import redirect_stderr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import comfyless.generate as g


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
print("── COMFYLESS_SCHEMA shape ─────────────────────────────────────")

schema = g.COMFYLESS_SCHEMA

# COMFYLESS_SCHEMA is wrapped in MappingProxyType (read-only) per ADR-012
# step-2 hardening — the structural-immutability check the step-2 reviewer
# called out (F2). MappingProxyType is a `Mapping` but NOT a `dict`
# subclass; tightening the assertion to Mapping accepts both shapes.
import collections.abc as _abc  # noqa: E402

check("schema is a Mapping (dict or MappingProxyType)",
      isinstance(schema, _abc.Mapping))
check("schema is non-empty",
      len(schema) > 0)

# Every entry is (type-or-tuple, default)
for key, entry in schema.items():
    check(f"entry[{key!r}] is 2-tuple",
          isinstance(entry, tuple) and len(entry) == 2,
          f"got {entry!r}")

# model + prompt are required (default = None)
check("model is required (default None)",
      schema["model"][1] is None)
check("prompt is required (default None)",
      schema["prompt"][1] is None)

# Required-by-prompt: all the params the contract says must be covered
_REQUIRED_KEYS = {
    "model", "prompt", "negative_prompt", "seed", "steps",
    "cfg_scale", "true_cfg_scale", "width", "height",
    "sampler", "schedule", "max_sequence_length",
    "transformer_path", "vae_path", "text_encoder_path",
    "text_encoder_2_path", "vae_from_transformer",
    "loras",
}
missing = _REQUIRED_KEYS - set(schema.keys())
check("schema covers every required canonical key",
      not missing,
      f"missing: {sorted(missing)}")


# ──────────────────────────────────────────────────────────────────────
print("\n── Schema self-consistency vs generate() + CLI merge ──────────")

# Load generate.py source for string-grep coverage proofs.
_source = Path(g.__file__).read_text()

# Every canonical key should appear somewhere in the source beyond just
# the schema definition itself — proves it's wired into either the
# generate() signature, the CLI merge, or sidecar building.
for key in schema:
    # Count occurrences; must be > 1 (the schema definition counts as one)
    count = _source.count(f"\"{key}\"")
    check(f"schema key {key!r} referenced outside the schema dict",
          count >= 2,
          f"only {count} occurrences (probably dead)")

# Every canonical key referenced in generate() and _build_call_kwargs
# kwargs-building call sites is in the schema.  Pull likely-canonical
# names out of the generate() signature:
import inspect as _inspect
_gen_sig = _inspect.signature(g.generate)
_generate_params = set(_gen_sig.parameters.keys())

# generate()'s first positional is model_path, not model (API surface
# uses _path suffix here for clarity).  Rest should line up:
_sig_to_schema_rename = {"model_path": "model", "loras": "loras"}
_generate_canonical = {
    _sig_to_schema_rename.get(p, p)
    for p in _generate_params
    if p not in {
        # Runtime-only params, not sidecar-shaped; documented in schema comment.
        "output_path", "precision", "device", "offload_vae",
        "attention_slicing", "sequential_offload", "allow_hf_download",
        # Krea conditioning rebalance — runtime-only CLI flags (in-process
        # path), recorded in metadata when active but not sidecar input params.
        "rebalance", "rebalance_mult", "rebalance_weights",
        "_cached_pipeline",
        # Quantize-on-load knobs (ADR-019 slice A): runtime-class like
        # precision — hardware/VRAM tradeoffs, declared in _RUNTIME_KIND,
        # not sidecar-persisted.
        "quant", "quant_skip", "quant_only",
        # MCP-internal call-shape flag; signals _save_with_metadata to apply
        # invariant-12 PNG redaction. Not user-facing; not a sidecar param.
        "mcp_caller",
    }
}
_missing_from_schema = _generate_canonical - set(schema.keys())
check("every generate()-signature param (minus runtime-only) is in schema",
      not _missing_from_schema,
      f"missing: {sorted(_missing_from_schema)}")


# ──────────────────────────────────────────────────────────────────────
print("\n── _CLI_TO_CANONICAL sanity ────────────────────────────────────")

cli_map = g._CLI_TO_CANONICAL

check("CLI map is a dict", isinstance(cli_map, dict))

# Every mapped target is a valid canonical schema key.
_bad_targets = {cli: canon for cli, canon in cli_map.items()
                if canon not in schema}
check("every CLI target is a valid canonical key",
      not _bad_targets,
      f"bad: {_bad_targets}")

# No duplicate canonical targets (two CLI flags mapping to the same
# canonical key is a bug unless intentional — surface it loudly).
_targets = list(cli_map.values())
_dupes = {t for t in _targets if _targets.count(t) > 1}
check("no duplicate canonical targets in CLI map",
      not _dupes,
      f"duplicates: {sorted(_dupes)}")

# Identity pairs (cli name == canonical name) are conceptually noise; the
# contract allows keeping a few for clarity (e.g. negative_prompt,
# vae_from_transformer, lora→loras is NOT identical).  Spot-check the
# known renames are present:
for cli, canon in [
    ("cfg", "cfg_scale"),
    ("true_cfg", "true_cfg_scale"),
    ("max_seq_len", "max_sequence_length"),
    ("transformer", "transformer_path"),
    ("vae", "vae_path"),
    ("te1", "text_encoder_path"),
    ("te2", "text_encoder_2_path"),
    ("lora", "loras"),
]:
    check(f"CLI map has {cli!r} → {canon!r}",
          cli_map.get(cli) == canon,
          f"got {cli_map.get(cli)!r}")


# ──────────────────────────────────────────────────────────────────────
print("\n── _validate_params behavior ───────────────────────────────────")


def _capture_stderr(fn, *a, **kw):
    buf = io.StringIO()
    with redirect_stderr(buf):
        result = fn(*a, **kw)
    return result, buf.getvalue()


# Empty dict passes through silently.
out, err = _capture_stderr(g._validate_params, {}, source="unit")
check("empty dict → empty dict",
      out == {})
check("empty dict produces no warnings",
      err == "",
      f"stderr={err!r}")

# Known key + correct type → untouched, no warning.
out, err = _capture_stderr(g._validate_params,
                           {"seed": 42, "steps": 10}, source="unit")
check("known key with correct type preserved",
      out == {"seed": 42, "steps": 10})
check("known-good values emit no warnings",
      err == "",
      f"stderr={err!r}")

# Unknown key → dropped + warning.
out, err = _capture_stderr(g._validate_params,
                           {"seed": 42, "garbage_field": "X"},
                           source="unit-src")
check("unknown key dropped",
      "garbage_field" not in out)
check("known key preserved alongside unknown drop",
      out.get("seed") == 42)
check("unknown-key warning mentions key",
      "garbage_field" in err,
      f"stderr={err!r}")
check("unknown-key warning mentions source",
      "unit-src" in err,
      f"stderr={err!r}")
check("unknown-key warning uses 'dropping' verb",
      "dropping" in err,
      f"stderr={err!r}")

# Type mismatch → KEPT + warning.
out, err = _capture_stderr(g._validate_params,
                           {"seed": "42"},   # str instead of int
                           source="type-src")
check("type-mismatch value preserved (no coercion)",
      out == {"seed": "42"})
check("type-mismatch warning fires",
      "seed" in err and "expected" in err,
      f"stderr={err!r}")
check("type-mismatch warning mentions source",
      "type-src" in err)
check("type-mismatch warning names expected type",
      "int" in err)
check("type-mismatch warning names actual type",
      "str" in err)

# Canonical type per ADR-012 (accepted 2026-05-15): cfg_scale is canonical
# float. _validate_params remains warn-and-keep (Vision invariant 7) — its
# acceptance set is unchanged from the prior (int, float) declaration, but
# its warning set grew: int inputs that previously silently passed now
# surface a type-mismatch warning. The value still flows through.
out, err = _capture_stderr(g._validate_params,
                           {"cfg_scale": 4}, source="unit")
check("int kept for cfg_scale (warn, value preserved — warn-set grew per ADR-012)",
      out == {"cfg_scale": 4} and "cfg_scale" in err)

out, err = _capture_stderr(g._validate_params,
                           {"cfg_scale": 4.5}, source="unit")
check("float accepted for cfg_scale (canonical type, no warning)",
      out == {"cfg_scale": 4.5} and err == "")

# bool is now rejected at the type-check level (closes the prior documented
# gap). _validate_params still keeps the value per invariant 7; the warn
# surfaces it for human attention.
out, err = _capture_stderr(g._validate_params,
                           {"cfg_scale": True}, source="unit")
check("bool kept for cfg_scale (warn — bool-as-int gap surfaced per ADR-012)",
      out == {"cfg_scale": True} and "cfg_scale" in err)

out, err = _capture_stderr(g._validate_params,
                           {"cfg_scale": "4.5"}, source="unit")
check("str rejected for cfg_scale (warning, value kept)",
      out == {"cfg_scale": "4.5"} and "cfg_scale" in err)

# Nullable: true_cfg_scale can be None (explicit "unset" signal).
out, err = _capture_stderr(g._validate_params,
                           {"true_cfg_scale": None}, source="unit")
check("None accepted for nullable true_cfg_scale",
      out == {"true_cfg_scale": None} and err == "")


# ──────────────────────────────────────────────────────────────────────
print("\n── _extract_eric_save_params canonical-only output ────────────")

_SAMPLE_ERIC_SAVE = {
    "model_path": "/x/model",
    "model_name": "model",
    "node_type": "EricDiffusionSave",
    "prompt": "hi",
    "negative_prompt": "bad",
    "seed": 42,
    "steps": 20,
    "cfg_scale": 3.5,
    "width": 1024,
    "height": 1024,
    "sampler": "default",
    "schedule": "linear",
    "sampler_s2": "ignore",
    "sampler_s3": "ignore",
    "garbage_field": 1,
    "loras": [{"path": "/x.safetensors", "weight": 0.8}],
}

out, err = _capture_stderr(
    g._extract_eric_save_params,
    json.dumps(_SAMPLE_ERIC_SAVE),
    "test.png",
)
check("eric-save output keys are subset of schema",
      set(out.keys()).issubset(set(schema.keys())),
      f"non-canonical keys leaked: {set(out.keys()) - set(schema.keys())}")
check("eric-save: model_path renamed to model",
      out.get("model") == "/x/model")
check("eric-save: model_path itself not in output",
      "model_path" not in out)
check("eric-save: garbage_field dropped",
      "garbage_field" not in out)
check("eric-save: node_type dropped",
      "node_type" not in out)
check("eric-save: model_name dropped",
      "model_name" not in out)
check("eric-save: sampler_s2/s3 dropped",
      "sampler_s2" not in out and "sampler_s3" not in out)
check("eric-save: loras dropped (unreplayable format)",
      "loras" not in out)
check("eric-save: canonical fields preserved",
      out.get("prompt") == "hi" and out.get("steps") == 20
      and out.get("cfg_scale") == 3.5)
check("eric-save: LoRA warning printed when chunk had loras",
      "LoRAs were active" in err,
      f"stderr={err!r}")

# Negative case: malformed JSON raises ValueError.
raised = None
try:
    g._extract_eric_save_params("{not valid", "test.png")
except ValueError as e:
    raised = e
check("eric-save: malformed JSON raises ValueError",
      raised is not None and "not valid JSON" in str(raised))


# ──────────────────────────────────────────────────────────────────────
print("\n── _extract_comfyui_params canonical-only output ───────────────")

# Minimal synthetic ComfyUI graph: KSampler + CLIPTextEncode + EmptyLatent
_SAMPLE_COMFYUI_GRAPH = {
    "1": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 123, "steps": 25, "cfg": 7.5,
            "scheduler": "karras",
            "positive": ["2", 0],
            "negative": ["3", 0],
        },
    },
    "2": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "a cat"},
    },
    "3": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "bad"},
    },
    "4": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 768, "height": 512},
    },
}

out, err = _capture_stderr(
    g._extract_comfyui_params,
    json.dumps(_SAMPLE_COMFYUI_GRAPH),
)
check("comfyui output keys are subset of schema",
      set(out.keys()).issubset(set(schema.keys())),
      f"non-canonical keys leaked: {set(out.keys()) - set(schema.keys())}")
check("comfyui: cfg renamed to cfg_scale",
      out.get("cfg_scale") == 7.5)
check("comfyui: cfg (non-canonical) not in output",
      "cfg" not in out)
check("comfyui: steps/seed/width/height/prompt/negative_prompt extracted",
      out.get("steps") == 25
      and out.get("seed") == 123
      and out.get("width") == 768
      and out.get("height") == 512
      and out.get("prompt") == "a cat"
      and out.get("negative_prompt") == "bad")
check("comfyui: schedule mapped to 'karras'",
      out.get("schedule") == "karras")

# Negative case: malformed JSON raises ValueError.
raised = None
try:
    g._extract_comfyui_params("{nope")
except ValueError as e:
    raised = e
check("comfyui: malformed JSON raises ValueError",
      raised is not None and "not valid JSON" in str(raised))


# ──────────────────────────────────────────────────────────────────────
print("\n── Regression smoke: example sidecar round-trip ────────────────")

_EXAMPLE_PATH = Path(__file__).parent / "comfyless" / "examples" / "qwen_image_hello_world.json"
out, err = _capture_stderr(g._load_sidecar, str(_EXAMPLE_PATH))
check("example sidecar loads",
      isinstance(out, dict) and len(out) > 0)
check("example sidecar: all resulting keys are canonical",
      set(out.keys()).issubset(set(schema.keys())),
      f"non-canonical: {set(out.keys()) - set(schema.keys())}")
check("example sidecar: produces no validator warnings",
      err == "",
      f"stderr={err!r}")
# Spot-check the round-trip preserved critical params:
check("example sidecar: prompt preserved",
      "golden retriever" in out.get("prompt", ""))
check("example sidecar: seed preserved",
      out.get("seed") == 42)
check("example sidecar: cfg_scale preserved",
      out.get("cfg_scale") == 4.0)


# ──────────────────────────────────────────────────────────────────────
print("\n── _apply_overrides validates too ──────────────────────────────")

out, err = _capture_stderr(
    g._apply_overrides,
    {"seed": 1},
    ["steps=20", "garbage=X"],
)
check("overrides: known key applied",
      out.get("steps") == 20)
check("overrides: unknown key dropped",
      "garbage" not in out)
check("overrides: warning for unknown key",
      "garbage" in err and "dropping" in err)

# Negative case: malformed --override raises.
raised = None
try:
    g._apply_overrides({}, ["no-equals-sign"])
except ValueError as e:
    raised = e
check("overrides: malformed spec raises ValueError",
      raised is not None and "key=value" in str(raised))


# ──────────────────────────────────────────────────────────────────────
print("\n── _explicit_override_keys (ADR-009) ───────────────────────────")

check("override-keys: None → empty set",
      g._explicit_override_keys(None) == set())
check("override-keys: empty list → empty set",
      g._explicit_override_keys([]) == set())

_keys = g._explicit_override_keys(
    ["cfg_scale=5", "garbage=X", "no-equals", "seed=42"]
)
check("override-keys: valid canonical keys captured",
      "cfg_scale" in _keys and "seed" in _keys)
check("override-keys: unknown canonical key filtered",
      "garbage" not in _keys)
check("override-keys: malformed spec (no =) ignored",
      "no-equals" not in _keys)


# ──────────────────────────────────────────────────────────────────────
print("\n── FAMILY_DEFAULTS shape (ADR-009) ─────────────────────────────")

from comfyless.family_defaults import FAMILY_DEFAULTS

check("FAMILY_DEFAULTS is a dict",
      isinstance(FAMILY_DEFAULTS, dict))
check("FAMILY_DEFAULTS is non-empty",
      len(FAMILY_DEFAULTS) > 0)

for fam, entry in FAMILY_DEFAULTS.items():
    check(f"FAMILY_DEFAULTS[{fam!r}] is a dict",
          isinstance(entry, dict))

# Every key in every family dict must be a canonical schema key — the
# overlay applier silently skips unknown keys, but unknown entries here
# are dead code that signals a family-defaults edit that drifted from
# the schema.
_bad_fam_keys: dict = {}
for fam, entry in FAMILY_DEFAULTS.items():
    bad = set(entry.keys()) - set(schema.keys())
    if bad:
        _bad_fam_keys[fam] = bad
check("every FAMILY_DEFAULTS key is in COMFYLESS_SCHEMA",
      not _bad_fam_keys,
      f"bad: {_bad_fam_keys}")

# Spot-check critical families' values are the documented model-card numbers.
check("qwen-image: true_cfg_scale=4.0 (model card)",
      FAMILY_DEFAULTS["qwen-image"].get("true_cfg_scale") == 4.0)
check("qwen-image: steps=50 (model card)",
      FAMILY_DEFAULTS["qwen-image"].get("steps") == 50)
check("sdxl: cfg_scale=7.0 (SAI recommendation)",
      FAMILY_DEFAULTS["sdxl"].get("cfg_scale") == 7.0)


# ──────────────────────────────────────────────────────────────────────
print("\n── _apply_family_defaults overlay (ADR-009) ────────────────────")

import tempfile
import shutil
import atexit

_tmpdirs_to_cleanup: list = []
atexit.register(
    lambda: [shutil.rmtree(d, ignore_errors=True) for d in _tmpdirs_to_cleanup]
)


def _make_fake_model(family_class_name: str) -> str:
    """Create a tempdir with a synthetic model_index.json.

    Returns the absolute path.  Cleanup is registered atexit so tests
    don't leak /tmp dirs across runs.
    """
    d = tempfile.mkdtemp(prefix="fam_defaults_test_")
    _tmpdirs_to_cleanup.append(d)
    with open(Path(d) / "model_index.json", "w") as f:
        json.dump({"_class_name": family_class_name}, f)
    return d


# detect_pipeline_class verifies the class is in the installed diffusers.
# These three are core to the project; if any are missing the install is
# broken and the test failure is a useful signal.
_TEST_FAMILIES = {
    "qwen-image": "QwenImagePipeline",
    "sdxl":       "StableDiffusionXLPipeline",
    "flux":       "FluxPipeline",
}
_paths = {fam: _make_fake_model(cls) for fam, cls in _TEST_FAMILIES.items()}

# 1. No explicit, no iterated → family default writes.
_p = {"model": _paths["sdxl"], "cfg_scale": 3.5}
_capture_stderr(g._apply_family_defaults, _p, set(), set())
check("overlay: sdxl writes cfg_scale=7.0 when key is not explicit",
      _p["cfg_scale"] == 7.0,
      f"got {_p.get('cfg_scale')!r}")

# 2. Explicit set → NOT clobbered.
_p = {"model": _paths["sdxl"], "cfg_scale": 3.5}
_capture_stderr(g._apply_family_defaults, _p, {"cfg_scale"}, set())
check("overlay: explicit cfg_scale preserved (not overwritten by family)",
      _p["cfg_scale"] == 3.5)

# 3. Iterated axis → NOT clobbered.
_p = {"model": _paths["sdxl"], "cfg_scale": 3.5}
_capture_stderr(g._apply_family_defaults, _p, set(), {"cfg_scale"})
check("overlay: iterated cfg_scale preserved",
      _p["cfg_scale"] == 3.5)

# 4. qwen-image writes true_cfg_scale + steps simultaneously.
_p = {"model": _paths["qwen-image"], "true_cfg_scale": None, "steps": 28}
_capture_stderr(g._apply_family_defaults, _p, set(), set())
check("overlay: qwen-image writes true_cfg_scale=4.0",
      _p["true_cfg_scale"] == 4.0)
check("overlay: qwen-image writes steps=50",
      _p["steps"] == 50)

# 5. Mixed: explicit one key, family fills the other.
_p = {"model": _paths["qwen-image"], "true_cfg_scale": 6.0, "steps": 28}
_capture_stderr(
    g._apply_family_defaults, _p, {"true_cfg_scale"}, set(),
)
check("overlay: mixed — explicit true_cfg_scale preserved at 6.0",
      _p["true_cfg_scale"] == 6.0)
check("overlay: mixed — non-explicit steps still gets family value 50",
      _p["steps"] == 50)

# ── distilled-schedule warning on --transformer override (2026-07-10) ────
# Family resolves from the BASE model path (ADR-009 name-hint); --transformer
# is never inspected. So a base-weights override against a *-Turbo directory
# silently inherits a few-step distilled schedule. FAMILY_DEFAULTS already
# records that the two are mutually destructive ("Base defaults (30/4.0)
# DESTROY this distill"), and the reverse costs denoising steps the weights
# need. Neither direction raises, so the only symptom is a degraded image.


def _make_fake_model_named(family_class_name: str, prefix: str) -> str:
    """Fake model whose DIRECTORY NAME carries a name hint (ADR-009)."""
    d = tempfile.mkdtemp(prefix=prefix)
    _tmpdirs_to_cleanup.append(d)
    with open(Path(d) / "model_index.json", "w") as f:
        json.dump({"_class_name": family_class_name}, f)
    return d


_zturbo = _make_fake_model_named("ZImagePipeline", "zimg_turbo_test_")
_zbase = _make_fake_model_named("ZImagePipeline", "zimg_base_test_")


def _warned(p, explicit=frozenset()):
    _, err = _capture_stderr(g._apply_family_defaults, p, set(explicit), set())
    return "WARNING" in err and "--transformer override inherited" in err


check("distilled-warn: turbo base + transformer override WARNS",
      _warned({"model": _zturbo, "transformer_path": "/x/greed_int8.safetensors"}))
check("distilled-warn: warning names the family and both schedule values",
      all(s in _capture_stderr(
          g._apply_family_defaults,
          {"model": _zturbo, "transformer_path": "/x/g.safetensors"},
          set(), set())[1]
          for s in ("zimage-turbo", "cfg_scale=1.0", "steps=8")))

# Negatives — the warning must not cry wolf.
check("distilled-warn: turbo base, NO override stays quiet (legit turbo use)",
      not _warned({"model": _zturbo}))
check("distilled-warn: non-distilled base + override stays quiet",
      not _warned({"model": _zbase, "transformer_path": "/x/g.safetensors"}))
check("distilled-warn: override + BOTH cfg and steps explicit stays quiet",
      not _warned({"model": _zturbo, "transformer_path": "/x/g.safetensors",
                   "cfg_scale": 4.0, "steps": 30},
                  explicit={"cfg_scale", "steps"}))
# ...but an inherited 8-step budget ruins the image even when cfg is chosen,
# so a partially-explicit schedule must STILL warn.
check("distilled-warn: only cfg explicit -> still warns (steps inherited)",
      _warned({"model": _zturbo, "transformer_path": "/x/g.safetensors",
               "cfg_scale": 4.0}, explicit={"cfg_scale"}))
check("distilled-warn: only steps explicit -> still warns (cfg inherited)",
      _warned({"model": _zturbo, "transformer_path": "/x/g.safetensors",
               "steps": 30}, explicit={"steps"}))

# Table sync: every distilled family is a real FAMILY_DEFAULTS row, and the
# membership is not derivable from cfg_scale (krea-turbo 0.0, zimage-turbo 1.0)
# — which is exactly why it is declared rather than inferred.
from comfyless.family_defaults import DISTILLED_FAMILIES  # noqa: E402

check("distilled-warn: DISTILLED_FAMILIES all present in FAMILY_DEFAULTS",
      all(f in g.FAMILY_DEFAULTS for f in DISTILLED_FAMILIES),
      f"orphans: {sorted(set(DISTILLED_FAMILIES) - set(g.FAMILY_DEFAULTS))}")
check("distilled-warn: membership is not a cfg_scale==0 rule",
      g.FAMILY_DEFAULTS["krea-turbo"]["cfg_scale"] == 0.0
      and g.FAMILY_DEFAULTS["zimage-turbo"]["cfg_scale"] == 1.0
      and {"krea-turbo", "zimage-turbo"} <= set(DISTILLED_FAMILIES))
check("distilled-warn: flux2klein excluded (ordinary 24-step schedule)",
      "flux2klein" not in DISTILLED_FAMILIES
      and g.FAMILY_DEFAULTS["flux2klein"]["steps"] == 24)

# 6. Unknown family → no-op (class not in diffusers).
_unknown = _make_fake_model("ThisPipelineClassDoesNotExist__zzz")
_p = {"model": _unknown, "cfg_scale": 3.5}
_capture_stderr(g._apply_family_defaults, _p, set(), set())
check("overlay: unknown class → no-op (no exception, schema default kept)",
      _p["cfg_scale"] == 3.5)

# 7. Missing model key → no-op.
_p = {"cfg_scale": 3.5}
_capture_stderr(g._apply_family_defaults, _p, set(), set())
check("overlay: missing model key → no-op",
      _p == {"cfg_scale": 3.5})

# 8. Missing model_index.json → no-op (empty tempdir).
_empty = tempfile.mkdtemp(prefix="fam_defaults_empty_")
_tmpdirs_to_cleanup.append(_empty)
_p = {"model": _empty, "cfg_scale": 3.5}
_capture_stderr(g._apply_family_defaults, _p, set(), set())
check("overlay: missing model_index.json → no-op",
      _p["cfg_scale"] == 3.5)

# 9. Log line format on apply: family name, idx prefix, "defaults applied".
_p = {"model": _paths["flux"]}
_, _err = _capture_stderr(
    g._apply_family_defaults, _p, set(), set(), idx=3,
)
check("overlay: log line names family",
      "family=flux" in _err,
      f"stderr={_err!r}")
check("overlay: log line includes idx",
      "iter 3" in _err,
      f"stderr={_err!r}")
check("overlay: log line includes 'defaults applied'",
      "defaults applied" in _err,
      f"stderr={_err!r}")

# 10. No applied keys (all explicit) → no log line.
_p = {"model": _paths["flux"], "cfg_scale": 1.0, "steps": 1}
_, _err = _capture_stderr(
    g._apply_family_defaults, _p, {"cfg_scale", "steps"}, set(),
)
check("overlay: silent when all family keys are explicit",
      _err == "",
      f"stderr={_err!r}")


# ──────────────────────────────────────────────────────────────────────
print("\n── Krea-2 family detection + defaults + routing ────────────────")

from nodes.eric_diffusion_utils import infer_model_family

# Detection: one Krea2Pipeline class → two families via is_distilled.
check("krea: Krea2Pipeline → 'krea' (single-arg form unchanged)",
      infer_model_family("Krea2Pipeline") == "krea")
check("krea: Krea2Pipeline + is_distilled=False → 'krea'",
      infer_model_family("Krea2Pipeline", False) == "krea")
check("krea-turbo: Krea2Pipeline + is_distilled=True → 'krea-turbo'",
      infer_model_family("Krea2Pipeline", True) == "krea-turbo")
# is_distilled only flips the krea family — never leaks onto other classes.
check("krea: is_distilled is a no-op for non-krea classes",
      infer_model_family("FluxPipeline", True) == "flux")

# ── zimage base vs turbo: name-hint discriminator (ADR-009 2026-07-06) ──
# Z-Image ships NO is_distilled marker; Turbo detected by "turbo" in path.
check("zimage: ZImagePipeline, no hint → 'zimage' (base)",
      infer_model_family("ZImagePipeline") == "zimage")
check("zimage: ZImagePipeline + Z-Image-base path → 'zimage'",
      infer_model_family("ZImagePipeline", False,
                         name_hint="/hf-local/Z-Image-base") == "zimage")
check("zimage-turbo: ZImagePipeline + Z-Image-Turbo path → 'zimage-turbo'",
      infer_model_family("ZImagePipeline", False,
                         name_hint="/hf-local/Z-Image-Turbo")
      == "zimage-turbo")
check("zimage-turbo: HF snapshot path with 'Turbo' detected",
      infer_model_family("ZImagePipeline", False,
                         name_hint="/hub/models--Tongyi--Z-Image-Turbo/"
                                   "snapshots/abc123") == "zimage-turbo")
check("zimage-turbo: case-insensitive 'TURBO'",
      infer_model_family("ZImagePipeline", False,
                         name_hint="/x/Z-IMAGE-TURBO") == "zimage-turbo")
# name_hint 'turbo' is scoped to zimage — never flips another family.
check("name_hint 'turbo' is a no-op for non-zimage classes",
      infer_model_family("FluxPipeline", False,
                         name_hint="/x/Flux-Turbo") == "flux")
check("zimage: empty name_hint (default) → base, never turbo",
      infer_model_family("ZImagePipeline", False, name_hint="") == "zimage")
# Family-default values: base holds, turbo is the empirically-validated pair.
check("zimage: cfg_scale=4.0 (base, Phase-A validated)",
      FAMILY_DEFAULTS["zimage"].get("cfg_scale") == 4.0)
check("zimage: steps=30 (base)",
      FAMILY_DEFAULTS["zimage"].get("steps") == 30)
check("zimage-turbo: cfg_scale=1.0 (single-pass; base 4.0 destroys distill)",
      FAMILY_DEFAULTS["zimage-turbo"].get("cfg_scale") == 1.0)
check("zimage-turbo: steps=8 (distilled)",
      FAMILY_DEFAULTS["zimage-turbo"].get("steps") == 8)

# Family-default values are the model-card numbers.
check("krea: cfg_scale=3.5 (Raw model card)",
      FAMILY_DEFAULTS["krea"].get("cfg_scale") == 3.5)
check("krea: steps=52 (Raw model card)",
      FAMILY_DEFAULTS["krea"].get("steps") == 52)
check("krea-turbo: cfg_scale=0.0 (Turbo, CFG disabled)",
      FAMILY_DEFAULTS["krea-turbo"].get("cfg_scale") == 0.0)
check("krea-turbo: steps=8 (Turbo model card)",
      FAMILY_DEFAULTS["krea-turbo"].get("steps") == 8)


# _build_call_kwargs routing — fake pipes so we don't need diffusers'
# Krea2Pipeline installed (it ships only on diffusers main).
class _FakeKreaPipe:
    def __call__(self, prompt, height, width, num_inference_steps, generator,
                 guidance_scale=None, negative_prompt=None,
                 max_sequence_length=None):
        pass


class _FakeKreaPipeNoNeg:
    def __call__(self, prompt, height, width, num_inference_steps, generator,
                 guidance_scale=None):
        pass


# Raw: real CFG via guidance_scale, NOT true_cfg_scale; negative prompt
# forwarded when accepted.
_kw = g._build_call_kwargs(
    _FakeKreaPipe(), "krea", False, "a cat", "blurry",
    1024, 1024, 52, 3.5, None, 512, None,
)
check("krea routing: guidance_scale=3.5 passed",
      _kw.get("guidance_scale") == 3.5)
check("krea routing: NOT true_cfg_scale (flux-like, not qwen)",
      "true_cfg_scale" not in _kw)
check("krea routing: negative_prompt forwarded when accepted + provided",
      _kw.get("negative_prompt") == "blurry")
check("krea routing: max_sequence_length forwarded when accepted",
      _kw.get("max_sequence_length") == 512)

# Turbo: cfg=0.0 passes through unchanged (CFG disabled, single pass).
_kw = g._build_call_kwargs(
    _FakeKreaPipe(), "krea-turbo", False, "a cat", "",
    2048, 2048, 8, 0.0, None, 512, None,
)
check("krea-turbo routing: guidance_scale=0.0 passed through",
      _kw.get("guidance_scale") == 0.0)
check("krea-turbo routing: no negative_prompt when none provided",
      "negative_prompt" not in _kw)

# Pipe that doesn't accept negative_prompt → it is not forwarded.
_kw = g._build_call_kwargs(
    _FakeKreaPipeNoNeg(), "krea", False, "a cat", "blurry",
    1024, 1024, 52, 3.5, None, 512, None,
)
check("krea routing: negative_prompt dropped when pipe doesn't accept it",
      "negative_prompt" not in _kw)
check("krea routing: max_sequence_length dropped when pipe doesn't accept it",
      "max_sequence_length" not in _kw)


# zimage-turbo routing — the load-bearing check: it MUST route through the
# guidance_scale branch, NOT the introspection fallback (which would emit
# true_cfg_scale and drop CFG). Regression guard for ADR-009 2026-07-06.
class _FakeZImagePipe:
    def __call__(self, prompt, height, width, num_inference_steps, generator,
                 guidance_scale=None, negative_prompt=None):
        pass


_kw = g._build_call_kwargs(
    _FakeZImagePipe(), "zimage-turbo", False, "a cat", "",
    1024, 1024, 8, 1.0, None, 512, None,
)
check("zimage-turbo routing: guidance_scale=1.0 passed (single-pass)",
      _kw.get("guidance_scale") == 1.0)
check("zimage-turbo routing: NOT true_cfg_scale (would drop via fallback)",
      "true_cfg_scale" not in _kw)
_kw = g._build_call_kwargs(
    _FakeZImagePipe(), "zimage", False, "a cat", "blurry",
    1024, 1024, 30, 4.0, None, 512, None,
)
check("zimage (base) routing: guidance_scale=4.0 + negative forwarded",
      _kw.get("guidance_scale") == 4.0 and _kw.get("negative_prompt") == "blurry")


# ──────────────────────────────────────────────────────────────────────
print("\n── Krea rebalance: _parse_rebalance_weights ───────────────────")

check("parse comma list", g._parse_rebalance_weights("1,2,3") == [1.0, 2.0, 3.0])
check("parse semicolons normalized to commas",
      g._parse_rebalance_weights("1;2;3") == [1.0, 2.0, 3.0])
check("parse skips empty fields",
      g._parse_rebalance_weights("1, ,2") == [1.0, 2.0])
check("parse empty string → None", g._parse_rebalance_weights("") is None)
check("parse whitespace-only → None", g._parse_rebalance_weights("   ") is None)
check("parse None → None", g._parse_rebalance_weights(None) is None)
check("parse the shipped default preset (12 values)",
      g._parse_rebalance_weights("1,1,1,1,1,1,1,2.5,5,1.1,4,1")
      == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5, 5.0, 1.1, 4.0, 1.0])

# Negative case: malformed (non-numeric) input fails loud, not silent.
_raised = False
try:
    g._parse_rebalance_weights("1,x,3")
except ValueError:
    _raised = True
check("parse malformed → ValueError (fail loud)", _raised)


# ──────────────────────────────────────────────────────────────────────
print("\n── Krea rebalance: _apply_krea_rebalance gain math ────────────")

import torch  # noqa: E402


class _FakeEncodePipe:
    """Minimal pipe exposing encode_prompt → (4-D embeds, mask)."""
    def __init__(self, embeds, mask):
        self._embeds, self._mask = embeds, mask
        self.seen = {}

    def encode_prompt(self, prompt, device=None, max_sequence_length=512):
        self.seen = {"prompt": prompt, "device": device,
                     "max_sequence_length": max_sequence_length}
        return self._embeds, self._mask


# (batch=1, seq=2, n_layers=3, dim=4) of ones → easy to verify per-layer gains.
_embeds = torch.ones(1, 2, 3, 4)
_mask = torch.ones(1, 2, dtype=torch.long)

# Per-layer weights [1,2,3] × global mult 2.0 → layers scale to 2,4,6.
_pipe = _FakeEncodePipe(_embeds, _mask)
_ck = g._apply_krea_rebalance(_pipe, {"prompt": "x", "height": 1024},
                              2.0, [1.0, 2.0, 3.0], 512, "cpu")
_out = _ck["prompt_embeds"]
check("rebalance: prompt popped from call_kwargs", "prompt" not in _ck)
check("rebalance: prompt_embeds set", "prompt_embeds" in _ck)
check("rebalance: mask passed through", _ck["prompt_embeds_mask"] is _mask)
check("rebalance: prompt forwarded to encode_prompt", _pipe.seen["prompt"] == "x")
check("rebalance: layer 0 gain 1×mult2 → 2", bool(torch.allclose(_out[:, :, 0, :], torch.full((1, 2, 4), 2.0))))
check("rebalance: layer 1 gain 2×mult2 → 4", bool(torch.allclose(_out[:, :, 1, :], torch.full((1, 2, 4), 4.0))))
check("rebalance: layer 2 gain 3×mult2 → 6", bool(torch.allclose(_out[:, :, 2, :], torch.full((1, 2, 4), 6.0))))

# No per-layer weights → just the global multiplier, all layers equal.
_pipe2 = _FakeEncodePipe(torch.ones(1, 2, 3, 4), _mask)
_ck2 = g._apply_krea_rebalance(_pipe2, {"prompt": "y"}, 3.0, None, 512, "cpu")
check("rebalance: multiplier-only scales uniformly by 3",
      bool(torch.allclose(_ck2["prompt_embeds"], torch.full((1, 2, 3, 4), 3.0))))

# dtype is preserved across the float() round-trip.
_pipe3 = _FakeEncodePipe(torch.ones(1, 2, 3, 4, dtype=torch.bfloat16), _mask)
_ck3 = g._apply_krea_rebalance(_pipe3, {"prompt": "z"}, 1.0, [1.0, 1.0, 1.0], 512, "cpu")
check("rebalance: output dtype preserved (bfloat16)",
      _ck3["prompt_embeds"].dtype == torch.bfloat16)

# Negative case: wrong-length weights → ValueError.
_raised = False
try:
    g._apply_krea_rebalance(_FakeEncodePipe(torch.ones(1, 2, 3, 4), _mask),
                            {"prompt": "x"}, 1.0, [1.0, 2.0], 512, "cpu")
except ValueError:
    _raised = True
check("rebalance: wrong-length weights → ValueError", _raised)

# Negative case: 3-D embeds (no layer axis) → ValueError, not silent miss-scale.
_raised = False
try:
    g._apply_krea_rebalance(_FakeEncodePipe(torch.ones(1, 2, 4), _mask),
                            {"prompt": "x"}, 1.0, None, 512, "cpu")
except ValueError:
    _raised = True
check("rebalance: 3-D embeds → ValueError (fail loud)", _raised)


# ──────────────────────────────────────────────────────────────────────
# Krea2 attention-backend pin (_pin_krea_attention_backend): diffusers
# 0.39.0's Krea2AttnProcessor passes bool mask + enable_gqa, which knocks
# SDPA onto the math backend (S^2 materialization → OOM at high res);
# comfyless pins the transformer to cuDNN for the krea families.
print("\n── krea attention-backend pin ─────────────────────────────────")


class _FakeBackendTransformer:
    def __init__(self, raise_on_set=False):
        self.backend = None
        self._raise = raise_on_set

    def set_attention_backend(self, name):
        if self._raise:
            raise ValueError("no such backend")
        self.backend = name


class _FakeBackendPipe:
    def __init__(self, transformer):
        if transformer is not None:
            self.transformer = transformer


_t = _FakeBackendTransformer()
check("krea family pins cuDNN backend",
      g._pin_krea_attention_backend(_FakeBackendPipe(_t), "krea") is True
      and _t.backend == "_native_cudnn", f"backend={_t.backend!r}")
_t = _FakeBackendTransformer()
check("krea-turbo family pins cuDNN backend",
      g._pin_krea_attention_backend(_FakeBackendPipe(_t), "krea-turbo") is True
      and _t.backend == "_native_cudnn", f"backend={_t.backend!r}")
_t = _FakeBackendTransformer()
check("non-krea family left untouched (NEGATIVE)",
      g._pin_krea_attention_backend(_FakeBackendPipe(_t), "flux2") is False
      and _t.backend is None, f"backend={_t.backend!r}")
check("pipe without transformer → no-op, no raise",
      g._pin_krea_attention_backend(_FakeBackendPipe(None), "krea") is False)
check("set_attention_backend failure → warn + False, never raises",
      g._pin_krea_attention_backend(
          _FakeBackendPipe(_FakeBackendTransformer(raise_on_set=True)),
          "krea") is False)
# The pinned name must exist in the installed diffusers backend registry —
# catches an upstream rename breaking the pin silently.
from diffusers.models.attention_dispatch import AttentionBackendName
check("'_native_cudnn' exists in diffusers backend registry",
      "_native_cudnn" in {x.value for x in AttentionBackendName.__members__.values()})


# ──────────────────────────────────────────────────────────────────────
# ── LoRA outcome plumbing + CLI loud reporting (ADR-015 2026-07-06) ──
print("\n── LoRA failure surfacing (formatter + CLI reporter) ──")

# lora_failure_warnings: one operator string per FAILED outcome (path-bearing).
_outcomes = [
    {"path": "/abs/a.safetensors", "adapter_name": "a", "applied": True,
     "reason": None},
    {"path": "/abs/b.safetensors", "adapter_name": "b", "applied": False,
     "reason": "0 modules applied (adapter not active)"},
]
_w = g.lora_failure_warnings(_outcomes)
check("lora_failure_warnings: one string per FAILED outcome only", len(_w) == 1)
check("lora_failure_warnings: operator string includes the path (operator-facing)",
      "/abs/b.safetensors" in _w[0])
check("lora_failure_warnings: applied outcome produces no warning",
      all("/abs/a.safetensors" not in s for s in _w))
check("lora_failure_warnings: all-applied → empty",
      g.lora_failure_warnings([_outcomes[0]]) == [])

# _report_lora_outcome: exit 3 + banner when warnings present, 0 when absent.
import io as _io
import contextlib as _cl
_buf = _io.StringIO()
with _cl.redirect_stderr(_buf):
    _rc = g._report_lora_outcome({"lora_warnings": ["LoRA not applied (x): /p"]})
check("_report_lora_outcome: returns exit code 3 on failure", _rc == 3)
check("_report_lora_outcome: prints a prominent banner to stderr",
      "DID NOT APPLY" in _buf.getvalue())
_buf2 = _io.StringIO()
with _cl.redirect_stderr(_buf2):
    _rc0 = g._report_lora_outcome({"seed": 1})  # no lora_warnings key
check("_report_lora_outcome: returns 0 when no lora_warnings", _rc0 == 0)
check("_report_lora_outcome: silent when nothing failed", _buf2.getvalue() == "")
_rc_empty = g._report_lora_outcome({"lora_warnings": []})
check("_report_lora_outcome: empty warnings list → 0", _rc_empty == 0)

# _iterate_combo_disposition: the --iterate sweep must treat exit-3 (soft
# LoRA failure, image written) as NON-FATAL, but abort on real errors
# (code-review finding 1). Pins the semantics.
check("iterate-disposition: rc 0 → 'ok' (continue)",
      g._iterate_combo_disposition(0) == "ok")
check("iterate-disposition: rc 3 (soft LoRA fail) → 'soft' (continue, flag)",
      g._iterate_combo_disposition(g._LORA_SOFT_FAIL_RC) == "soft")
check("iterate-disposition: rc 1 → 'fatal' (abort sweep)",
      g._iterate_combo_disposition(1) == "fatal")
check("iterate-disposition: rc 2 → 'fatal' (abort sweep)",
      g._iterate_combo_disposition(2) == "fatal")
check("iterate-disposition: exit-3 constant is distinct from hard errors",
      g._LORA_SOFT_FAIL_RC == 3 and g._LORA_SOFT_FAIL_RC not in (0, 1, 2))
# _report_lora_outcome returns exactly the soft-fail code (not a bare literal)
check("_report_lora_outcome: failure code IS _LORA_SOFT_FAIL_RC",
      g._report_lora_outcome({"lora_warnings": ["x"]}) == g._LORA_SOFT_FAIL_RC)


# ── quant is sidecar-replayable (2026-07-08) ───────────────────────────
# quant moved from _RUNTIME_KIND to SCHEMA_KIND: it affects output
# CORRECTNESS for some transformer/LoRA combos (fp8-resident merges go
# black; --quant fp8's dequant→torchao path works), so --params replay must
# reproduce it. These pin the three legs of the round-trip.
print("\n── quant sidecar round-trip ───────────────────────────────────")
import argparse as _qap
import tempfile as _qtf
import os as _qos

# Leg 1: a sidecar carrying the triple survives _validate_params intact.
_qd = _qtf.mkdtemp(prefix="quant_sidecar_test_")
_qp = _qos.path.join(_qd, "s.json")
with open(_qp, "w") as _qf:
    json.dump({"model": "/m", "prompt": "p", "quant": "fp8",
               "quant_skip": ["text_encoder"], "quant_only": []}, _qf)
_ql = g._load_params(_qp)
check("sidecar quant survives _load_params", _ql.get("quant") == "fp8",
      f"got {_ql.get('quant')!r}")
check("sidecar quant_skip survives _load_params",
      _ql.get("quant_skip") == ["text_encoder"])

# Leg 2: argparse defaults are None SENTINELS — an omitted --quant must not
# clobber the sidecar's value in the CLI merge (regressing these defaults
# to "none"/[] silently breaks --params replay).
_qargv, sys.argv = sys.argv, ["comfyless"]
try:
    _qargs = g._parse_args()
finally:
    sys.argv = _qargv
check("--quant argparse default is the None sentinel", _qargs.quant is None)
check("--quant-skip/--quant-only argparse defaults are None sentinels",
      _qargs.quant_skip is None and _qargs.quant_only is None)
check("_cli_value_for treats unset quant as not-explicit",
      g._cli_value_for(_qargs, "quant") is None)

# Positive precedence leg: an EXPLICIT `--quant none` is non-None ("none"),
# so it wins the merge over a sidecar's fp8 (review note, 2026-07-08).
_qargv, sys.argv = sys.argv, ["comfyless", "--quant", "none"]
try:
    _qargs2 = g._parse_args()
finally:
    sys.argv = _qargv
check("explicit --quant none is non-None and explicit in the merge",
      g._cli_value_for(_qargs2, "quant") == "none")

# Leg 3: quant keys are params, not skip-set metadata — sidecar load must
# NOT drop them the way it drops timestamp/elapsed.
check("quant keys are NOT in _SKIP_SIDECAR_KEYS",
      not ({"quant", "quant_skip", "quant_only"} & g._SKIP_SIDECAR_KEYS))
# Schema seeding gives copies, not the shared default list objects. The
# comprehension is inline in _run_cli_mode, so pin it at the source level
# (repo precedent: test_server_robustness's source-text checks) — reverting
# to `k: default` would share the schema's mutable [] across runs.
import inspect as _qinsp
_qsrc = _qinsp.getsource(g._run_cli_mode)
check("seeding comprehension copies list defaults (source pin)",
      "list(default) if isinstance(default, list) else default" in _qsrc)


# ── NAG quadruple is sidecar-replayable (ADR-023) ──────────────────────
# nag_scale/nag_tau/nag_alpha/nag_end joined SCHEMA_KIND with the quant
# precedent: NAG changes output CONTENT, so --params replay must reproduce
# it. Same three legs as quant, plus the schema defaults pin.
print("\n── NAG sidecar round-trip (ADR-023) ───────────────────────────")

# Leg 0: schema defaults are the documented dormant/paper values.
check("nag schema defaults are 0.0/2.5/0.25/1.0",
      (schema["nag_scale"][1], schema["nag_tau"][1],
       schema["nag_alpha"][1], schema["nag_end"][1]) == (0.0, 2.5, 0.25, 1.0))
check("nag kinds are all float",
      all(schema[k][0] is float
          for k in ("nag_scale", "nag_tau", "nag_alpha", "nag_end")))

# Leg 1: a sidecar carrying the quadruple survives _load_params intact.
_nd = _qtf.mkdtemp(prefix="nag_sidecar_test_")
_np = _qos.path.join(_nd, "s.json")
with open(_np, "w") as _nf:
    json.dump({"model": "/m", "prompt": "p", "nag_scale": 4.0,
               "nag_tau": 3.0, "nag_alpha": 0.5, "nag_end": 0.75}, _nf)
_nl = g._load_params(_np)
check("sidecar nag_scale survives _load_params", _nl.get("nag_scale") == 4.0,
      f"got {_nl.get('nag_scale')!r}")
check("sidecar nag_tau/alpha/end survive _load_params",
      (_nl.get("nag_tau"), _nl.get("nag_alpha"), _nl.get("nag_end"))
      == (3.0, 0.5, 0.75))

# Leg 2: argparse defaults are None SENTINELS — an omitted --nag-scale must
# not clobber a sidecar's value in the CLI merge.
_nargv, sys.argv = sys.argv, ["comfyless"]
try:
    _nargs = g._parse_args()
finally:
    sys.argv = _nargv
check("nag argparse defaults are None sentinels",
      _nargs.nag_scale is None and _nargs.nag_tau is None
      and _nargs.nag_alpha is None and _nargs.nag_end is None)
check("_cli_value_for treats unset nag_scale as not-explicit",
      g._cli_value_for(_nargs, "nag_scale") is None)

# Positive precedence leg: an EXPLICIT `--nag-scale 0` is non-None (0.0),
# so it wins the merge over a sidecar's 4.0 (turns NAG off on replay).
_nargv, sys.argv = sys.argv, ["comfyless", "--nag-scale", "0"]
try:
    _nargs2 = g._parse_args()
finally:
    sys.argv = _nargv
check("explicit --nag-scale 0 is non-None and explicit in the merge",
      g._cli_value_for(_nargs2, "nag_scale") == 0.0)

# Leg 3: nag keys are params, not skip-set metadata — sidecar load must
# NOT drop them the way it drops timestamp/elapsed.
check("nag keys are NOT in _SKIP_SIDECAR_KEYS",
      not ({"nag_scale", "nag_tau", "nag_alpha", "nag_end"}
           & g._SKIP_SIDECAR_KEYS))
# ...while the nag_warnings provenance list IS skip-set metadata (like
# lora_warnings): a replayed sidecar must not trip the unknown-key warning.
check("nag_warnings IS in _SKIP_SIDECAR_KEYS (provenance, not a param)",
      "nag_warnings" in g._SKIP_SIDECAR_KEYS)

# Family gating table (ADR-023 krea + ADR-024 expansion; loud warning on
# unsupported families and on CFG-owned configs).
print("\n── NAG family gating (_nag_gate, ADR-024 table) ───────────────")
check("krea accepts NAG at cfg 0", g._nag_gate("krea", 4.0) == (True, None))
check("krea-turbo accepts NAG at cfg 0",
      g._nag_gate("krea-turbo", 4.0, 0.0) == (True, None))
check("zimage-turbo accepts NAG at cfg 0",
      g._nag_gate("zimage-turbo", 4.0, 0.0) == (True, None))
check("zimage (base) accepts NAG at cfg 0",
      g._nag_gate("zimage", 4.0, 0.0) == (True, None))
check("flux accepts NAG (guidance embeds are not CFG)",
      g._nag_gate("flux", 4.0) == (True, None))
check("flux2 accepts NAG even at cfg>0 (guidance embed rides along)",
      g._nag_gate("flux2", 4.0, 4.0) == (True, None))
check("flux2klein accepts NAG",
      g._nag_gate("flux2klein", 4.0, 3.5) == (True, None))
_ng_off = g._nag_gate("krea-turbo", 1.0)
check("nag_scale<=1 is dormant (no warning)",
      _ng_off == (False, None), f"got {_ng_off!r}")
check("nag_scale None is dormant", g._nag_gate("krea", None) == (False, None))
# cfg-gated families: classic CFG owns the negative at cfg>0.
for _fam in ("krea", "krea-turbo", "zimage", "zimage-turbo"):
    _ng_cfg = g._nag_gate(_fam, 4.0, 1.0)
    check(f"{_fam} at cfg>0 skips NAG with a loud CFG warning",
          _ng_cfg[0] is False and bool(_ng_cfg[1]) and "CFG" in _ng_cfg[1],
          f"got {_ng_cfg!r}")
_ng_sdxl = g._nag_gate("sdxl", 4.0)
check("unsupported family skips NAG", _ng_sdxl[0] is False)
check("unsupported-family warning is loud and names the family",
      bool(_ng_sdxl[1]) and "sdxl" in _ng_sdxl[1],
      f"got {_ng_sdxl[1]!r}")
_ng_dormant = g._nag_gate("sdxl", 0.0)
check("dormant NAG on unsupported family raises no warning (NEGATIVE)",
      _ng_dormant == (False, None), f"got {_ng_dormant!r}")
check("gate table and dispatch map cover the same families",
      set(g._NAG_CFG_OWNS_NEGATIVE) == set(g._NAG_MODULES))


print("\n──────────────────────────────────────────────────")
print(f"  {passed} passed, {failed} failed")
print("──────────────────────────────────────────────────")
sys.exit(0 if failed == 0 else 1)
