#!/usr/bin/env python3
"""Unit tests for the OpenWebUI image-gen tool's pure parse helpers.

The tool module (comfyless/integrations/openwebui/generate_image_tool.py) runs
inside the OWUI container and imports container-only deps (open_webui, fastapi,
aiohttp, starlette). We stub those so the module imports here, then exercise the
pure `_parse_loras` / `_parse_weights_csv` helpers that turn the flat chat-tool
string params into the MCP `loras` / `rebalance_weights` shapes.

Run: ./.venv/bin/python3 test_owui_tool.py
"""
import importlib.util
import sys
import types

# Stub OWUI-container-only deps so the tool module imports outside the container.
for _name in ["aiohttp", "fastapi", "starlette", "starlette.datastructures",
              "open_webui", "open_webui.models", "open_webui.models.users",
              "open_webui.routers", "open_webui.routers.files"]:
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["fastapi"].Request = object
sys.modules["fastapi"].UploadFile = object
sys.modules["starlette.datastructures"].Headers = object
sys.modules["open_webui.models.users"].Users = object
sys.modules["open_webui.routers.files"].upload_file_handler = lambda *a, **k: None

_spec = importlib.util.spec_from_file_location(
    "owui_tool", "comfyless/integrations/openwebui/generate_image_tool.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)


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


# ── _parse_loras ───────────────────────────────────────────────────────
print("== _parse_loras ==")
check("single name → weight defaults to 1.0",
      _m._parse_loras("a") == [{"name": "a", "weight": 1.0}])
check("name:weight parsed",
      _m._parse_loras("a:0.8") == [{"name": "a", "weight": 0.8}])
check("multi, mixed, whitespace-trimmed",
      _m._parse_loras("a:0.8, b , c:2")
      == [{"name": "a", "weight": 0.8}, {"name": "b", "weight": 1.0},
          {"name": "c", "weight": 2.0}])
check("empty string → empty list", _m._parse_loras("") == [])
check("name with colon in weight position only — rpartition picks last colon",
      _m._parse_loras("my:lora:0.5") == [{"name": "my:lora", "weight": 0.5}])
check("default weight is a float (JSON-number contract, not int)",
      isinstance(_m._parse_loras("a")[0]["weight"], float))
# Documented edge: a colon-bearing name whose trailing segment parses as a
# number is silently split (name='foo', weight=2024.0). Pinned so a refactor
# doesn't change it unnoticed; the docstring tells callers to add ':weight'.
check("colon-in-name with numeric tail is split (documented edge)",
      _m._parse_loras("foo:2024") == [{"name": "foo", "weight": 2024.0}])
# Negative weights pass through (server doesn't reject them either — consistent).
check("negative weight passes through unchanged",
      _m._parse_loras("a:-0.5") == [{"name": "a", "weight": -0.5}])

# Negative cases.
_raised = False
try:
    _m._parse_loras(":0.5")
except ValueError:
    _raised = True
check("empty name → ValueError", _raised)

_raised = False
try:
    _m._parse_loras("a:notanumber")
except ValueError:
    _raised = True
check("non-numeric weight → ValueError", _raised)


# ── _parse_weights_csv ─────────────────────────────────────────────────
print("\n== _parse_weights_csv ==")
check("comma + semicolon both split",
      _m._parse_weights_csv("1,2.5;3") == [1.0, 2.5, 3.0])
check("the 12-value Krea preset round-trips",
      _m._parse_weights_csv("1,1,1,1,1,1,1,2.5,5,1.1,4,1")
      == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5, 5.0, 1.1, 4.0, 1.0])
check("empty string → empty list", _m._parse_weights_csv("") == [])
check("blanks between separators skipped",
      _m._parse_weights_csv("1, ,2") == [1.0, 2.0])

_raised = False
try:
    _m._parse_weights_csv("1,x,3")
except ValueError:
    _raised = True
check("non-numeric token → ValueError", _raised)


print("\n──────────────────────────────────────────────────")
print(f"  {passed} passed, {failed} failed")
print("──────────────────────────────────────────────────")
sys.exit(0 if failed == 0 else 1)
