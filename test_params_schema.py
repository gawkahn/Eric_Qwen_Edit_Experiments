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

check("schema is a dict",
      isinstance(schema, dict))
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
        "_cached_pipeline",
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

# Union types: cfg_scale accepts int OR float. bool is a Python subclass
# of int (isinstance(True, int) == True), so the schema does NOT
# explicitly exclude bool — a documented gap. The True-input assertion
# below pins the *current* accepted behavior so any future tightening
# (e.g. adding `and not isinstance(value, bool)` to _validate_params)
# fails this test deliberately rather than silently changing semantics.
out, err = _capture_stderr(g._validate_params,
                           {"cfg_scale": 4}, source="unit")
check("int accepted for cfg_scale (union type)",
      out == {"cfg_scale": 4} and err == "")

out, err = _capture_stderr(g._validate_params,
                           {"cfg_scale": 4.5}, source="unit")
check("float accepted for cfg_scale (union type)",
      out == {"cfg_scale": 4.5} and err == "")

# Documented gap: bool passes as int. If a future change tightens this
# (and breaks this test), update the schema, this test, AND the comment.
out, err = _capture_stderr(g._validate_params,
                           {"cfg_scale": True}, source="unit")
check("bool accepted for cfg_scale (documented gap; bool is int-subclass)",
      out == {"cfg_scale": True} and err == "")

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
print("\n──────────────────────────────────────────────────")
print(f"  {passed} passed, {failed} failed")
print("──────────────────────────────────────────────────")
sys.exit(0 if failed == 0 else 1)
