#!/usr/bin/env python3
"""Test suite for the canonical machine-boundary validator (ADR-012).

Covers the Vision's N1-N17 + N20-N21 + N25-N30 negative-case grid plus a
small positive-case baseline. N18 (cross-call-site parity) currently
exercises the canonical validator only; steps 3 and 4 of the slice extend
it to the server and iterate paths. N19 (grep static check) skip-marked
until those rewrites land. N22 omitted: superseded by ADR-012 §3, which
makes the int → float safe cast the validator's published contract on
canonical-float fields (see N8/N9/N17).
"""

import ast
import dataclasses
import sys
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import comfyless.params_validation as pv  # noqa: E402
from comfyless.params_validation import (
    ValidationResult,
    validate_lora_entry,
    validate_machine_request,
)


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


def base_payload(**overrides):
    """Minimal valid payload for a generate request. Fields override defaults."""
    base = {
        "type": "generate",
        "model": "/m/foo",
        "prompt": "a cat",
    }
    base.update(overrides)
    return base


# ──────────────────────────────────────────────────────────────────────
print("\n== Positive baseline ==")

result = validate_machine_request(base_payload())
check("baseline: minimal valid payload accepted",
      result.ok and result.payload["model"] == "/m/foo")

result = validate_machine_request(base_payload(
    seed=42, steps=28, width=1024, height=1024, cfg_scale=4.0,
    true_cfg_scale=None, max_sequence_length=512,
    sampler="default", schedule="linear",
    vae_from_transformer=False, loras=[],
))
check("baseline: full sidecar-shaped payload accepted",
      result.ok)

# ──────────────────────────────────────────────────────────────────────
print("\n== N1-N5: bool rejected for canonical-int fields ==")

for n, field in [
    ("N1", "steps"),
    ("N2", "seed"),
    ("N3", "width"),
    ("N4", "height"),
    ("N5", "max_sequence_length"),
]:
    result = validate_machine_request(base_payload(**{field: True}))
    check(f"{n}: {field}=True rejected",
          not result.ok
          and result.error["field"] == field
          and "bool" in result.error["reason"])

# ──────────────────────────────────────────────────────────────────────
print("\n== N6-N7: bool rejected for canonical-float fields ==")

for n, field in [
    ("N6", "cfg_scale"),
    ("N7", "true_cfg_scale"),
]:
    result = validate_machine_request(base_payload(**{field: True}))
    check(f"{n}: {field}=True rejected",
          not result.ok
          and result.error["field"] == field
          and "bool" in result.error["reason"])

# Parametric N1-N7: derive numeric-field list from SCHEMA_KIND + _RUNTIME_KIND
# itself so that a future field addition without a corresponding test cannot
# silently weaken the bool-reject branch. Closes step-1 security-audit finding 1
# and step-3 security-audit finding 9 (runtime-kind coverage extension).
print("\n== Parametric bool-reject coverage (security-audit F1 + F9) ==")

_NUMERIC_KINDS = {pv._KIND_INT, pv._KIND_FLOAT, pv._KIND_FLOAT_NONE}
# pv._RUNTIME_KIND is private to params_validation but the test legitimately
# reaches into it: any numeric field added to runtime-only schema must auto-
# cover via this loop (step-3 audit F9). Today _RUNTIME_KIND has no numeric
# fields (only str / bool) so the loop is preventive.
_PARAMETRIC_FIELDS = {
    **pv.SCHEMA_KIND, **pv._RUNTIME_KIND, **pv._MCP_TRANSPORT_KIND}
for field, kind in _PARAMETRIC_FIELDS.items():
    if kind not in _NUMERIC_KINDS:
        continue
    # Required fields (model, prompt) are always str; numeric fields are all
    # optional, so the base_payload override is sufficient.
    for bad in (True, False):
        result = validate_machine_request(base_payload(**{field: bad}))
        check(
            f"parametric: {field}={bad} rejected (kind={kind})",
            not result.ok
            and result.error["field"] == field
            and "bool" in result.error["reason"],
        )

# ──────────────────────────────────────────────────────────────────────
print("\n== N8-N11: canonical-float field acceptance + safe int→float cast ==")

result = validate_machine_request(base_payload(cfg_scale=4))
check("N8: cfg_scale=4 (int) accepted and cast to 4.0 (float)",
      result.ok
      and result.payload["cfg_scale"] == 4.0
      and isinstance(result.payload["cfg_scale"], float)
      and not isinstance(result.payload["cfg_scale"], bool))

result = validate_machine_request(base_payload(true_cfg_scale=4))
check("N9: true_cfg_scale=4 (int) accepted and cast to 4.0 (float)",
      result.ok
      and result.payload["true_cfg_scale"] == 4.0
      and isinstance(result.payload["true_cfg_scale"], float))

result = validate_machine_request(base_payload(cfg_scale=4.0))
check("N10: cfg_scale=4.0 (float) accepted and pass-through",
      result.ok and result.payload["cfg_scale"] == 4.0)

result = validate_machine_request(base_payload(true_cfg_scale=None))
check("N11: true_cfg_scale=None accepted (nullable per schema)",
      result.ok and result.payload["true_cfg_scale"] is None)

# ──────────────────────────────────────────────────────────────────────
print("\n== N12-N17: LoRA list per-entry validation ==")

result = validate_machine_request(base_payload(
    loras=[{"path": "/m/x.safetensors", "weight": "heavy"}]))
check("N12: loras[0].weight='heavy' (str) rejected",
      not result.ok
      and result.error["field"] == "loras[0].weight"
      and "str" in result.error["reason"])

result = validate_machine_request(base_payload(
    loras=[{"path": "/m/x.safetensors", "weight": True}]))
check("N13: loras[0].weight=True (bool) rejected",
      not result.ok
      and result.error["field"] == "loras[0].weight"
      and "bool" in result.error["reason"])

result = validate_machine_request(base_payload(
    loras=[{"path": "/m/x.safetensors"}]))
check("N14: loras[0] missing weight rejected",
      not result.ok
      and result.error["field"] == "loras[0].weight"
      and result.error["error"] == "missing_field")

result = validate_machine_request(base_payload(loras=[{"weight": 0.8}]))
check("N15: loras[0] missing path rejected",
      not result.ok
      and result.error["field"] == "loras[0].path"
      and result.error["error"] == "missing_field")

result = validate_machine_request(base_payload(
    loras=[{"path": "/m/x.safetensors", "weight": 0.8}]))
check("N16: loras[0] {path: str, weight: float} accepted",
      result.ok
      and result.payload["loras"][0]["weight"] == 0.8
      and isinstance(result.payload["loras"][0]["weight"], float))

result = validate_machine_request(base_payload(
    loras=[{"path": "/m/x.safetensors", "weight": 1}]))
check("N17: loras[0].weight=1 (int) accepted and cast to 1.0 (float)",
      result.ok
      and result.payload["loras"][0]["weight"] == 1.0
      and isinstance(result.payload["loras"][0]["weight"], float))

# ──────────────────────────────────────────────────────────────────────
print("\n== N18: cross-call-site fixture grid (validator-only in step 1) ==")

# Steps 3 and 4 of the slice extend this grid to also assert the server and
# iterate-LoRA paths return identical accept/reject decisions. In step 1 we
# only assert the canonical validator handles the grid consistently — which
# is trivially the case (single-site parity), but the fixture lives here
# already so the later steps amend it in place.

_GRID = [
    # (name,                       payload,                                              expected_ok)
    # Positive cases — accepted with cast where applicable
    ("valid-minimal",              base_payload(),                                        True),
    ("valid-int-cfg-cast",         base_payload(cfg_scale=4),                             True),
    ("valid-float-cfg",            base_payload(cfg_scale=4.0),                           True),
    ("valid-int-true-cfg-cast",    base_payload(true_cfg_scale=4),                        True),
    ("valid-true-cfg-none",        base_payload(true_cfg_scale=None),                     True),
    ("valid-lora-int-weight",      base_payload(loras=[{"path": "/x", "weight": 1}]),     True),
    ("valid-lora-float-weight",    base_payload(loras=[{"path": "/x", "weight": 0.8}]),   True),
    # N1-N5: bool rejected for canonical-int fields
    ("N1-reject-bool-steps",       base_payload(steps=True),                              False),
    ("N2-reject-bool-seed",        base_payload(seed=True),                               False),
    ("N3-reject-bool-width",       base_payload(width=True),                              False),
    ("N4-reject-bool-height",      base_payload(height=True),                             False),
    ("N5-reject-bool-max-seq",     base_payload(max_sequence_length=True),                False),
    # N6-N7: bool rejected for canonical-float fields
    ("N6-reject-bool-cfg",         base_payload(cfg_scale=True),                          False),
    ("N7-reject-bool-true-cfg",    base_payload(true_cfg_scale=True),                     False),
    # N12-N15: LoRA shape rejections
    ("N12-reject-lora-str-weight", base_payload(loras=[{"path": "/x", "weight": "h"}]),   False),
    ("N13-reject-lora-bool-weight",base_payload(loras=[{"path": "/x", "weight": True}]),  False),
    ("N14-reject-lora-no-weight",  base_payload(loras=[{"path": "/x"}]),                  False),
    ("N15-reject-lora-no-path",    base_payload(loras=[{"weight": 0.8}]),                 False),
    # N25-N29: float rejected for canonical-int fields
    ("N25-reject-float-steps",     base_payload(steps=28.0),                              False),
    ("N26-reject-float-seed",      base_payload(seed=42.0),                               False),
    ("N27-reject-float-width",     base_payload(width=1024.0),                            False),
    ("N28-reject-float-height",    base_payload(height=1024.0),                           False),
    ("N29-reject-float-max-seq",   base_payload(max_sequence_length=256.0),               False),
    # Additional shape edges
    ("reject-str-seed",            base_payload(seed="42"),                               False),
    ("reject-int-prompt",          base_payload(prompt=42),                               False),
    ("reject-non-dict-payload",    "not a dict",                                          False),
    ("reject-list-payload",        ["not", "a", "dict"],                                  False),
]

for name, payload, expected_ok in _GRID:
    result = validate_machine_request(payload)
    check(f"N18 grid: {name}", result.ok == expected_ok,
          detail=f"got ok={result.ok}, error={result.error}")

# ──────────────────────────────────────────────────────────────────────
print("\n== N18 cross-site: server._validate_request matches canonical ==")

# Activated step 3 — server.py:_validate_request now wraps the canonical
# validator. For every grid fixture, server and canonical must return the
# same accept/reject decision. Step 4 extends this to iterate's per-LoRA
# helper.
import comfyless.server as server_mod  # noqa: E402

for name, payload, expected_ok in _GRID:
    server_err = server_mod._validate_request(payload)
    server_ok = server_err is None
    canonical_ok = validate_machine_request(payload).ok
    check(
        f"N18 cross-site: {name} (server={server_ok}, canonical={canonical_ok})",
        server_ok == canonical_ok,
        detail=f"server_err={server_err!r}",
    )

# Server-only branches the canonical validator doesn't model (request type
# semantic check + required-field presence). The N18 grid can't cover these
# because they're server-specific by design; lock them with targeted asserts.
# Closes step-3 code-reviewer suggestion 1.
print("\n  -- server-only branches (canonical doesn't model these) --")

check("server: type=ping accepted (no further checks)",
      server_mod._validate_request({"type": "ping"}) is None)
check("server: type=unload accepted (no further checks)",
      server_mod._validate_request({"type": "unload"}) is None)
check("server: type=garbage rejected",
      server_mod._validate_request({"type": "garbage"})
      and "Unknown request type" in server_mod._validate_request({"type": "garbage"}))
check("server: generate missing model rejected",
      server_mod._validate_request({"type": "generate", "prompt": "hi"})
      == "Missing required field: 'model'")
check("server: generate missing prompt rejected",
      server_mod._validate_request({"type": "generate", "model": "/m/foo"})
      == "Missing required field: 'prompt'")

# Verify F6 fold-in: server's _validate_request mutates req in place with the
# validator's int→float safe cast. Confirms the cast actually propagates to
# downstream consumers, not just to result.payload.
_req_with_int_cfg = {
    "type": "generate", "model": "/m/foo", "prompt": "hi", "cfg_scale": 4,
}
server_mod._validate_request(_req_with_int_cfg)
check("server: int→float cast propagated to caller's req dict (F6 fold-in)",
      _req_with_int_cfg["cfg_scale"] == 4.0
      and isinstance(_req_with_int_cfg["cfg_scale"], float))

# ──────────────────────────────────────────────────────────────────────
print("\n== N19: zero isinstance(int|float|bool|str) in server._validate_request ==")

# Activated step 3. AST-walks server.py, finds the _validate_request
# FunctionDef, asserts its body contains no Call to isinstance whose
# second-arg names int / float / bool / str. The check is structurally
# tight — immune to formatting and comment shenanigans.

_FORBIDDEN_TYPES = {"int", "float", "bool", "str"}
_server_src = (Path(__file__).parent / "comfyless" / "server.py").read_text()
_server_tree = ast.parse(_server_src)

def _isinstance_violations_in_body(body):
    """Yield Call nodes inside `body` whose shape is isinstance(X, T1[, T2, ...])
    with at least one T in _FORBIDDEN_TYPES."""
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "isinstance"):
            continue
        if len(node.args) < 2:
            continue
        type_arg = node.args[1]
        # Direct: isinstance(x, str)
        type_names = []
        if isinstance(type_arg, ast.Name):
            type_names.append(type_arg.id)
        # Tuple union: isinstance(x, (int, float))
        elif isinstance(type_arg, ast.Tuple):
            for elt in type_arg.elts:
                if isinstance(elt, ast.Name):
                    type_names.append(elt.id)
        if any(t in _FORBIDDEN_TYPES for t in type_names):
            yield node

_violations_server = []
for node in ast.walk(_server_tree):
    if isinstance(node, ast.FunctionDef) and node.name == "_validate_request":
        _violations_server = list(_isinstance_violations_in_body(node.body))
        break

check(
    "N19: zero forbidden isinstance(int|float|bool|str) in server._validate_request",
    not _violations_server,
    detail=f"violations at lines: {[v.lineno for v in _violations_server]}",
)

# N19 second site: iterate's lora_stack branch in _validate_iterate_value.
# Walk generate.py's AST, find the FunctionDef, then find the If statement
# whose test is `expected == "lora_stack"`, then check its body.
_generate_src = (Path(__file__).parent / "comfyless" / "generate.py").read_text()
_generate_tree = ast.parse(_generate_src)
_violations_iterate = []
for node in ast.walk(_generate_tree):
    if isinstance(node, ast.FunctionDef) and node.name == "_validate_iterate_value":
        for sub in node.body:
            if not isinstance(sub, ast.If):
                continue
            test = sub.test
            # Match `expected == "lora_stack"`.
            if (isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "expected"
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq)
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == "lora_stack"):
                _violations_iterate = list(_isinstance_violations_in_body(sub.body))
                break
        break

check(
    "N19: zero forbidden isinstance(int|float|bool|str) in iterate lora_stack branch",
    not _violations_iterate,
    detail=f"violations at lines: {[v.lineno for v in _violations_iterate]}",
)

# ──────────────────────────────────────────────────────────────────────
print("\n== N18 cross-site iterate: _validate_iterate_value lora_stack matches validate_lora_entry ==")

# Activated step 4. Iterate's per-LoRA validator takes a list of LoRA dicts;
# the canonical helper takes single entries. Build a mini-grid of single-
# entry fixtures and assert iterate(_validate_iterate_value([entry], "lora_stack"))
# matches validate_lora_entry(entry, 0).ok for every case.
import comfyless.generate as gen_mod  # noqa: E402

_LORA_GRID = [
    # (name,                              entry,                                            expected_ok)
    ("valid: path + float weight",        {"path": "/x.sft", "weight": 0.8},                True),
    ("valid: path + int weight (cast)",   {"path": "/x.sft", "weight": 1},                  True),
    ("valid: path + zero weight",         {"path": "/x.sft", "weight": 0.0},                True),
    ("reject: missing weight",            {"path": "/x.sft"},                               False),
    ("reject: missing path",              {"weight": 0.8},                                  False),
    ("reject: bool weight",               {"path": "/x.sft", "weight": True},               False),
    ("reject: str weight",                {"path": "/x.sft", "weight": "heavy"},            False),
    ("reject: None weight",               {"path": "/x.sft", "weight": None},               False),
    ("reject: int path",                  {"path": 42, "weight": 0.8},                      False),
    ("reject: non-dict entry",            "not a dict",                                     False),
    # Extra-keys pass-through: unknown-key tightening is out of scope per
    # Vision; validate_lora_entry preserves unknown keys verbatim (e.g. 'rank',
    # 'alpha' kohya metadata). Lock the contract surface for iterate consumers.
    ("valid: extra keys preserved",       {"path": "/x.sft", "weight": 0.8, "rank": 64},    True),
]

for name, entry, expected_ok in _LORA_GRID:
    canonical_ok = validate_lora_entry(entry, 0).ok
    iterate_ok = gen_mod._validate_iterate_value([entry], "lora_stack")
    check(
        f"N18 iterate cross-site: {name}",
        canonical_ok == iterate_ok == expected_ok,
        detail=f"canonical={canonical_ok}, iterate={iterate_ok}, expected={expected_ok}",
    )

# ──────────────────────────────────────────────────────────────────────
print("\n== N20: validator does no filesystem IO ==")

# Patch every filesystem entry point a future maintainer might reach for.
# Closes step-1 security-audit finding 4 (extend N20 to pathlib methods).
with unittest.mock.patch("builtins.open") as mock_open, \
     unittest.mock.patch("os.path.exists") as mock_exists, \
     unittest.mock.patch("os.path.realpath") as mock_realpath, \
     unittest.mock.patch("os.stat") as mock_stat, \
     unittest.mock.patch("os.access") as mock_access, \
     unittest.mock.patch("os.listdir") as mock_listdir, \
     unittest.mock.patch("os.scandir") as mock_scandir, \
     unittest.mock.patch("pathlib.Path.exists") as mock_p_exists, \
     unittest.mock.patch("pathlib.Path.stat") as mock_p_stat, \
     unittest.mock.patch("pathlib.Path.is_file") as mock_p_is_file, \
     unittest.mock.patch("pathlib.Path.is_dir") as mock_p_is_dir, \
     unittest.mock.patch("pathlib.Path.resolve") as mock_p_resolve:
    validate_machine_request(base_payload())
    validate_machine_request("not even a dict")
    validate_machine_request(base_payload(steps=True))
    validate_machine_request(base_payload(
        loras=[{"path": "/x", "weight": 0.5}]))
    all_mocks = [
        ("open", mock_open),
        ("os.path.exists", mock_exists),
        ("os.path.realpath", mock_realpath),
        ("os.stat", mock_stat),
        ("os.access", mock_access),
        ("os.listdir", mock_listdir),
        ("os.scandir", mock_scandir),
        ("pathlib.Path.exists", mock_p_exists),
        ("pathlib.Path.stat", mock_p_stat),
        ("pathlib.Path.is_file", mock_p_is_file),
        ("pathlib.Path.is_dir", mock_p_is_dir),
        ("pathlib.Path.resolve", mock_p_resolve),
    ]
    called = [name for name, m in all_mocks if m.call_count > 0]
    check("N20: no filesystem call (open / os.* / pathlib.Path.*)",
          not called,
          detail=f"unexpected IO calls: {called}")

# ──────────────────────────────────────────────────────────────────────
print("\n== N21: deterministic — identical inputs → identical results ==")

p = base_payload(cfg_scale=4, seed=42)
r1 = validate_machine_request(p)
r2 = validate_machine_request(dict(p))  # copy so input identity differs
check("N21: same payload → same ValidationResult.ok",
      r1.ok and r2.ok and r1.ok == r2.ok)
check("N21: same payload → same payload contents",
      r1.payload == r2.payload)

p_bad = base_payload(steps=True)
e1 = validate_machine_request(p_bad)
e2 = validate_machine_request(dict(p_bad))
check("N21: same bad payload → same error dict",
      e1.error == e2.error)

# ──────────────────────────────────────────────────────────────────────
print("\n== N25-N29: float rejected for canonical-int fields (no float→int) ==")

for n, field in [
    ("N25", "steps"),
    ("N26", "seed"),
    ("N27", "width"),
    ("N28", "height"),
    ("N29", "max_sequence_length"),
]:
    result = validate_machine_request(base_payload(**{field: 50.0}))
    check(f"{n}: {field}=50.0 (float) rejected",
          not result.ok
          and result.error["field"] == field
          and "float" in result.error["reason"]
          and "int" in result.error["reason"])

# ──────────────────────────────────────────────────────────────────────
print("\n== N30: validator source contains no int() call (AST check) ==")

# Vision suggested mock-patching `int` to record calls. That is not viable:
# the validator uses `int` as the type argument to isinstance() throughout,
# and patching the name breaks isinstance() with TypeError. The intent —
# "the validator never converts a float input to an int" — is served
# equivalently by walking the validator's AST for any Call node whose
# func is Name("int"). Immune to comments, docstrings, and formatting.

src_text = (Path(__file__).parent / "comfyless" / "params_validation.py").read_text()
tree = ast.parse(src_text)
int_calls = []
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "int":
        int_calls.append(node.lineno)

check("N30: validator source contains zero int() call sites (AST)",
      not int_calls,
      detail=f"int() call lines: {int_calls}")

# ──────────────────────────────────────────────────────────────────────
print("\n== Extra: dataclass frozen-ness (sanity check on internal contract) ==")

r_ok = validate_machine_request(base_payload())
try:
    r_ok.ok = False  # type: ignore[misc]
    check("dataclass frozen: mutation raises FrozenInstanceError", False,
          detail="frozen dataclass allowed mutation")
except dataclasses.FrozenInstanceError:
    check("dataclass frozen: mutation raises FrozenInstanceError", True)

# Module-level dicts are read-only (MappingProxyType per security-audit F7).
try:
    pv.SCHEMA_KIND["cfg_scale"] = pv._KIND_INT  # type: ignore[index]
    check("schema map: mutation raises TypeError", False,
          detail="MappingProxyType allowed mutation")
except TypeError:
    check("schema map: mutation raises TypeError", True)

# ──────────────────────────────────────────────────────────────────────
print("\n== Krea rebalance runtime fields ==")

# rebalance (bool) — accepted true; non-bool rejected.
check("rebalance=True accepted",
      validate_machine_request(base_payload(rebalance=True)).ok)
check("rebalance='x' (non-bool) rejected",
      not validate_machine_request(base_payload(rebalance="x")).ok)

# rebalance_mult (canonical float) — int cast to float; non-numeric rejected.
_r = validate_machine_request(base_payload(rebalance_mult=2))
check("rebalance_mult=2 (int) accepted and cast to 2.0 (float)",
      _r.ok and isinstance(_r.payload["rebalance_mult"], float)
      and _r.payload["rebalance_mult"] == 2.0)
check("rebalance_mult='x' (non-numeric) rejected",
      not validate_machine_request(base_payload(rebalance_mult="x")).ok)

# rebalance_weights (list) — list accepted; non-list rejected.
check("rebalance_weights=[1.0,2.0] accepted",
      validate_machine_request(base_payload(rebalance_weights=[1.0, 2.0])).ok)
check("rebalance_weights='x' (non-list) rejected",
      not validate_machine_request(base_payload(rebalance_weights="x")).ok)


# ──────────────────────────────────────────────────────────────────────
print("\n──────────────────────────────────────────────────")
print(f"  {passed} passed, {failed} failed")
print("──────────────────────────────────────────────────")
sys.exit(0 if failed == 0 else 1)
