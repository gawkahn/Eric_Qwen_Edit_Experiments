#!/usr/bin/env python3
"""Tests for comfyless/refine.py slices 1-2 — ADR-027 LLM-as-judge.

CPU-only, no GPU, no model weights, no network. Slice 1: the security-critical
verdict parser (closed two-key allowlist F1, numeric bounds F6, reject-unknown
F7) and the judge request-building pieces (image downscale + payload F5). Slice 2:
catalog-name resolution (F2 — names resolved ONLY via the ADR-015 hardened
resolver) and path-stripped planner metadata (F3), incl. a structural AST guard
that refine.py never selects a load-plane column. The thin HTTP POST wrapper
(`_post_judge`) needs a live endpoint and is not tested.

The negative cases are the point: an LLM must not be able to smuggle a path
(via an override key, a LoRA `path` field, or an unresolved name), a non-finite
number, or an out-of-range weight/score past this boundary — and no filesystem
path may reach the planner's LLM context.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from comfyless import refine
from comfyless.refine import RefineError, parse_verdict

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


def raises(name, fn):
    """Assert fn() raises RefineError."""
    global passed, failed
    try:
        fn()
    except RefineError:
        passed += 1
        print(f"  PASS  {name}")
    except Exception as e:  # noqa: BLE001
        failed += 1
        print(f"  FAIL  {name}  raised {type(e).__name__}, want RefineError: {e}")
    else:
        failed += 1
        print(f"  FAIL  {name}  did not raise")


def has_notice(v, needle):
    return any(needle in n for n in v.notices)


# ── Happy path ───────────────────────────────────────────────────────────────
print("== valid verdict ==")
V = parse_verdict(json.dumps({
    "scores": {"prompt_adherence": 7, "aesthetics": 9},
    "critique": {"prompt_adherence": "missing the red hat", "aesthetics": "clean"},
    "verdict": "revise",
    "overrides": {
        "prompt": "a cat in a red hat, studio light",
        "loras": [
            {"name": "detail-tweaker", "action": "set_weight", "weight": 0.8},
            {"name": "film-grain", "action": "add"},
            {"name": "oversharpen", "action": "remove"},
        ],
    },
}))
check("scores parsed", V.prompt_adherence == 7 and V.aesthetics == 9)
check("verdict parsed", V.verdict == "revise")
check("override prompt parsed", V.override_prompt == "a cat in a red hat, studio light")
check("three lora ops", len(V.lora_ops) == 3)
check("set_weight op", V.lora_ops[0].name == "detail-tweaker"
      and V.lora_ops[0].action == "set_weight" and V.lora_ops[0].weight == 0.8)
check("add op no weight", V.lora_ops[1].action == "add" and V.lora_ops[1].weight is None)
check("clean verdict has no notices", V.notices == [], detail=str(V.notices))
check("critique retained", V.critique.get("aesthetics") == "clean")

print("== fenced/prose-wrapped JSON tolerated ==")
V = parse_verdict("Here is my assessment:\n```json\n"
                  '{"scores": {"prompt_adherence": 5, "aesthetics": 5}, "verdict": "revise"}'
                  "\n```\nHope that helps!")
check("fenced JSON parses", V.prompt_adherence == 5 and V.aesthetics == 5)

# ── F1: closed override allowlist — no path from the LLM ──────────────────────
print("== F1 closed allowlist: path-bearing override keys dropped ==")
V = parse_verdict(json.dumps({
    "scores": {"prompt_adherence": 6, "aesthetics": 6},
    "verdict": "revise",
    "overrides": {
        "prompt": "ok",
        "model": "/etc/passwd",
        "transformer_path": "/home/gawkahn/evil.pt",
        "vae_path": "../../secret",
        "loras": [{"name": "good-lora", "action": "add"}],
    },
}))
check("disallowed 'model' dropped", not hasattr(V, "model"))
check("only prompt+loras survive", V.override_prompt == "ok" and len(V.lora_ops) == 1)
check("model drop noticed", has_notice(V, "model"), detail=str(V.notices))
check("transformer_path drop noticed", has_notice(V, "transformer_path"))
check("vae_path drop noticed", has_notice(V, "vae_path"))

print("== F1: LoRA 'path' field cannot smuggle a path ==")
V = parse_verdict(json.dumps({
    "scores": {"prompt_adherence": 6, "aesthetics": 6},
    "verdict": "revise",
    "overrides": {"loras": [
        {"name": "legit", "action": "add", "path": "/home/gawkahn/models/evil.pt"},
    ]},
}))
check("lora op kept by name", len(V.lora_ops) == 1 and V.lora_ops[0].name == "legit")
check("no 'path' attribute on LoraOp", not hasattr(V.lora_ops[0], "path"))
check("lora path key drop noticed", has_notice(V, "path"), detail=str(V.notices))

print("== F1: a LoRA entry that is ONLY a path (no name) is dropped ==")
V = parse_verdict(json.dumps({
    "scores": {"prompt_adherence": 6, "aesthetics": 6},
    "verdict": "revise",
    "overrides": {"loras": [{"path": "/evil.pt", "action": "add"}]},
}))
check("nameless path-only lora dropped", len(V.lora_ops) == 0)
check("dropped-name noticed", has_notice(V, "name"), detail=str(V.notices))

print("== F7: critique is allowlisted (no raw payload carrier) ==")
V = parse_verdict(json.dumps({
    "scores": {"prompt_adherence": 6, "aesthetics": 6},
    "verdict": "revise",
    "critique": {
        "prompt_adherence": "good",          # known + str → kept
        "aesthetics": {"deep": "payload"},   # known key, non-str value → dropped
        "model": "/etc/passwd",              # unknown key → dropped
    },
}))
check("critique keeps known str keys", V.critique == {"prompt_adherence": "good"},
      detail=str(V.critique))
check("critique unknown key dropped", has_notice(V, "critique key 'model'"))
check("critique non-str value dropped", has_notice(V, "critique 'aesthetics'"))

V = parse_verdict(
    '{"scores": {"prompt_adherence": 5, "aesthetics": 5}, "verdict": "revise",'
    ' "critique": {"aesthetics": 1e400}}')
check("1e400 inside critique does not survive (non-str dropped)", V.critique == {},
      detail=str(V.critique))

print("== F7: unknown top-level + score keys dropped ==")
V = parse_verdict(json.dumps({
    "scores": {"prompt_adherence": 6, "aesthetics": 6, "vibes": 11},
    "verdict": "revise",
    "sneaky": {"allow_hf_download": True},
    "output_dir": "/tmp/pwn",
}))
check("unknown top-level 'sneaky' dropped", has_notice(V, "sneaky"))
check("unknown top-level 'output_dir' dropped", has_notice(V, "output_dir"))
check("unknown score 'vibes' dropped", has_notice(V, "vibes"))

# ── F6: numeric bounds ───────────────────────────────────────────────────────
print("== F6: non-finite JSON constants rejected ==")
raises("NaN score rejected", lambda: parse_verdict(
    '{"scores": {"prompt_adherence": NaN, "aesthetics": 5}, "verdict": "revise"}'))
raises("Infinity score rejected", lambda: parse_verdict(
    '{"scores": {"prompt_adherence": Infinity, "aesthetics": 5}, "verdict": "revise"}'))
raises("-Infinity weight rejected", lambda: parse_verdict(
    '{"scores": {"prompt_adherence": 5, "aesthetics": 5}, "verdict": "revise",'
    ' "overrides": {"loras": [{"name": "x", "action": "set_weight", "weight": -Infinity}]}}'))
raises("overflow float score (1e400 -> inf) rejected", lambda: parse_verdict(
    '{"scores": {"prompt_adherence": 1e400, "aesthetics": 5}, "verdict": "revise"}'))
raises("huge bare-int score (10**400) -> RefineError not OverflowError", lambda: parse_verdict(
    '{"scores": {"prompt_adherence": ' + "9" * 400 + ', "aesthetics": 5}, "verdict": "revise"}'))
raises("single missing score key rejected", lambda: parse_verdict(json.dumps(
    {"scores": {"prompt_adherence": 5}, "verdict": "revise"})))

print("== F6: score clamping and int coercion ==")
V = parse_verdict(json.dumps({"scores": {"prompt_adherence": 11, "aesthetics": 0},
                              "verdict": "revise"}))
check("score 11 clamped to 10", V.prompt_adherence == 10)
check("score 0 clamped to 1", V.aesthetics == 1)
check("clamp noticed", sum("clamped" in n for n in V.notices) == 2, detail=str(V.notices))
V = parse_verdict(json.dumps({"scores": {"prompt_adherence": 7.6, "aesthetics": 4.2},
                              "verdict": "revise"}))
check("float score rounded", V.prompt_adherence == 8 and V.aesthetics == 4)
check("float rounding noticed", sum("coerced to int" in n for n in V.notices) == 2,
      detail=str(V.notices))
V = parse_verdict(json.dumps({"scores": {"prompt_adherence": 7, "aesthetics": 5},
                              "verdict": "revise"}))
check("clean int score has no coercion notice", not any("coerced" in n for n in V.notices))

print("== F6: non-numeric / bool scores rejected ==")
raises("string score rejected", lambda: parse_verdict(json.dumps(
    {"scores": {"prompt_adherence": "high", "aesthetics": 5}, "verdict": "revise"})))
raises("bool score rejected", lambda: parse_verdict(json.dumps(
    {"scores": {"prompt_adherence": True, "aesthetics": 5}, "verdict": "revise"})))
raises("missing scores object rejected", lambda: parse_verdict(json.dumps(
    {"verdict": "revise"})))

print("== F6: LoRA weight clamping ==")
V = parse_verdict(json.dumps({"scores": {"prompt_adherence": 5, "aesthetics": 5},
    "verdict": "revise", "overrides": {"loras": [
        {"name": "hot", "action": "set_weight", "weight": 1e9},
        {"name": "cold", "action": "set_weight", "weight": -50},
    ]}}))
check("huge weight clamped to +4", V.lora_ops[0].weight == 4.0)
check("negative weight clamped to -4", V.lora_ops[1].weight == -4.0)
check("weight clamp noticed", sum("clamped" in n for n in V.notices) == 2)

V = parse_verdict(
    '{"scores": {"prompt_adherence": 5, "aesthetics": 5}, "verdict": "revise",'
    ' "overrides": {"loras": [{"name": "x", "action": "set_weight", "weight": '
    + "9" * 400 + '}]}}')
check("huge bare-int weight dropped → set_weight op dropped", len(V.lora_ops) == 0,
      detail=str([o.name for o in V.lora_ops]))
check("huge-int weight drop noticed", has_notice(V, "weight dropped"))

print("== LoRA op validation edge cases ==")
V = parse_verdict(json.dumps({"scores": {"prompt_adherence": 5, "aesthetics": 5},
    "verdict": "revise", "overrides": {"loras": [
        {"name": "no-weight", "action": "set_weight"},          # set_weight sans weight → dropped
        {"name": "bad-action", "action": "delete"},             # invalid action → dropped
        {"name": "  spacey  ", "action": "add"},                # trimmed, kept
        {"name": "strval", "action": "set_weight", "weight": "0.5"},  # non-numeric weight
    ]}}))
check("set_weight without weight dropped", all(o.name != "no-weight" for o in V.lora_ops))
check("invalid action dropped", all(o.name != "bad-action" for o in V.lora_ops))
check("valid add kept + name trimmed", any(o.name == "spacey" for o in V.lora_ops))
check("string weight → set_weight dropped",
      all(o.name != "strval" for o in V.lora_ops), detail=str([o.name for o in V.lora_ops]))

print("== defense-in-depth: LoRA names with path/control chars dropped ==")
V = parse_verdict(json.dumps({"scores": {"prompt_adherence": 5, "aesthetics": 5},
    "verdict": "revise", "overrides": {"loras": [
        {"name": "../../etc/passwd", "action": "add"},
        {"name": "sub/dir/lora", "action": "add"},
        {"name": "back\\slash", "action": "add"},
        {"name": "ctrl\tchar", "action": "add"},
        {"name": "clean-name", "action": "add"},
    ]}}))
check("only clean name survives path/control filter",
      [o.name for o in V.lora_ops] == ["clean-name"], detail=str([o.name for o in V.lora_ops]))
check("path/control drop noticed", sum("path/control" in n for n in V.notices) == 4)

# ── F7: verdict coercion, malformed responses ────────────────────────────────
print("== F7: verdict coercion + malformed handling ==")
V = parse_verdict(json.dumps({"scores": {"prompt_adherence": 5, "aesthetics": 5},
                              "verdict": "absolutely stunning"}))
check("unknown verdict coerced to revise", V.verdict == "revise")
check("verdict coercion noticed", has_notice(V, "revise"))
print("== override prompt: empty / whitespace / oversize ignored ==")
V = parse_verdict(json.dumps({"scores": {"prompt_adherence": 5, "aesthetics": 5},
    "verdict": "revise", "overrides": {"prompt": ""}}))
check("empty override prompt ignored", V.override_prompt is None and has_notice(V, "empty"))
V = parse_verdict(json.dumps({"scores": {"prompt_adherence": 5, "aesthetics": 5},
    "verdict": "revise", "overrides": {"prompt": "   \n\t "}}))
check("whitespace override prompt ignored", V.override_prompt is None)
V = parse_verdict(json.dumps({"scores": {"prompt_adherence": 5, "aesthetics": 5},
    "verdict": "revise", "overrides": {"prompt": "x" * (refine.OVERRIDE_PROMPT_MAX_CHARS + 1)}}))
check("oversize override prompt ignored", V.override_prompt is None and has_notice(V, "exceeds"))

raises("no JSON object at all", lambda: parse_verdict("I refuse to answer."))
raises("truncated JSON rejected", lambda: parse_verdict(
    '{"scores": {"prompt_adherence": 5, "aesthetics"'))
raises("JSON array (not object) rejected", lambda: parse_verdict("[1, 2, 3]"))

# ── F5: image downscale + payload building ───────────────────────────────────
print("== F5: image downscale + judge payload ==")
from PIL import Image  # noqa: E402

big = Image.new("RGB", (4000, 2000), "white")
small = refine.downscale_for_judge(big)
check("4000x2000 longest side -> 1536", max(small.size) == refine.JUDGE_MAX_PX)
check("aspect ratio preserved", abs(small.size[0] / small.size[1] - 2.0) < 0.01)
same = Image.new("RGB", (800, 600), "white")
check("under-cap image untouched", refine.downscale_for_judge(same).size == (800, 600))

uri = refine.image_to_data_uri(same)
check("data URI prefix", uri.startswith("data:image/png;base64,"))

payload = refine.build_judge_payload("gemma", "SYS", "score this", uri)
check("payload temperature 0", payload["temperature"] == 0.0)
check("system message present", payload["messages"][0]["content"] == "SYS")
content = payload["messages"][1]["content"]
check("user content is a vision array", isinstance(content, list) and len(content) == 2)
check("text part present", content[0]["type"] == "text" and content[0]["text"] == "score this")
check("image part carries data URI", content[1]["type"] == "image_url"
      and content[1]["image_url"]["url"].startswith("data:image/png;base64,"))

print("== F5: seed-image byte cap ==")
raises("oversize seed image rejected", lambda: refine.load_seed_image_capped(
    __file__, max_bytes=1))  # this test file is > 1 byte

print("== F5: seed-image pixel cap (decompression-bomb guard, before decode) ==")
import tempfile  # noqa: E402
with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as _tf:
    _bomb_path = _tf.name
Image.new("RGB", (200, 200), "white").save(_bomb_path)
# A generous byte cap but a pixel cap below the image's 40_000 px: must reject on
# pixels (proving the header gate fires before the full decode).
raises("over-pixel-cap seed image rejected", lambda: refine.load_seed_image_capped(
    _bomb_path, max_bytes=10 * 1024 ** 2, max_pixels=200 * 200 - 1))
img_ok = refine.load_seed_image_capped(_bomb_path, max_pixels=200 * 200)
check("at-pixel-cap seed image loads (RGB)", img_ok.mode == "RGB" and img_ok.size == (200, 200))
os.unlink(_bomb_path)

# ── Slice 2: catalog-name resolution (F2) + safe planner metadata (F3) ───────
import tempfile  # noqa: E402
from comfyless.refine import LoraOp  # noqa: E402
from comfyless.catalog import build_catalog  # noqa: E402
from comfyless import catalog_db  # noqa: E402

print("== F2: LoRA name → path via ADR-015 resolver (names only, never a path) ==")
_root = tempfile.mkdtemp()
_mb = os.path.join(_root, "mb"); os.makedirs(_mb)
_ld = os.path.join(_root, "loras"); os.makedirs(_ld)
_td = os.path.join(_root, "tf"); os.makedirs(_td)
open(os.path.join(_ld, "detail-tweaker.safetensors"), "wb").close()
open(os.path.join(_td, "some-transformer.safetensors"), "wb").close()
_cat = build_catalog(_mb, lora_paths=(_ld,), transformer_paths=(_td,))
_roots = (_mb, _ld, _td)

_res, _notes = refine.resolve_lora_ops(_cat, _roots, [
    LoraOp("detail-tweaker", "add"),
    LoraOp("nonexistent-lora", "add"),
    LoraOp("some-transformer", "set_weight", 0.8),  # exists but wrong kind
])
check("known lora resolves to exactly one op", len(_res) == 1)
check("resolved abs_path is real + under a root",
      _res[0].abs_path.endswith("detail-tweaker.safetensors")
      and os.path.exists(_res[0].abs_path), detail=str(_res))
check("resolved op carries the original action/weight", _res[0].op.action == "add")
check("unknown lora dropped with notice", any("nonexistent-lora" in n for n in _notes))
check("wrong-kind (transformer) dropped as KindMismatch",
      all("some-transformer" != r.resolved_name for r in _res)
      and any("some-transformer" in n and "KindMismatch" in n for n in _notes),
      detail=str(_notes))

print("== F3: catalog metadata is a path-stripped allowlist ==")
_dbp = os.path.join(_root, "cat.sqlite")
_conn = catalog_db.connect(_dbp)
_eid = catalog_db.upsert_entry(_conn, name="detail-tweaker", kind="lora",
    abs_path="/secret/place/detail-tweaker.safetensors", root="/secret",
    relative_path="detail-tweaker.safetensors")
_conn.execute("UPDATE entries SET model_family = ? WHERE id = ?", ("qwen-image", _eid))
catalog_db.upsert_description(_conn, entry_id=_eid, source="civitai_api",
    description="adds crisp fine detail to skin and fabric", usage_tips="use 0.6-0.9",
    trigger_words=["detailed"], strength_rec="0.7")  # trigger_words is a list
catalog_db.rebuild_fts(_conn)
_conn.commit()

_md = refine.lora_metadata(_conn, "detail-tweaker")
check("metadata present", _md is not None)
check("metadata has description", "fine detail" in (_md or {}).get("description", ""))
check("metadata has usage_tips/trigger/strength",
      "0.6-0.9" in (_md or {}).get("usage_tips", "")
      and "detailed" in (_md or {}).get("trigger_words", "")
      and "0.7" in (_md or {}).get("strength_rec", ""))
check("metadata has model_family", (_md or {}).get("model_family") == "qwen-image")
_SAFE_KEYS = set(refine._SAFE_ENTRY_FIELDS) | set(refine._SAFE_DESC_FIELDS)
check("metadata keys are a closed subset of the safe allowlist (F3)",
      set((_md or {}).keys()) <= _SAFE_KEYS, detail=str(set((_md or {}).keys()) - _SAFE_KEYS))
check("missing lora → None", refine.lora_metadata(_conn, "no-such-lora") is None)

print("== F3: search-by-effect returns path-stripped safe views ==")
_hits = refine.search_loras(_conn, "fine detail", limit=5)
check("search finds the lora by effect", any(h.get("name") == "detail-tweaker" for h in _hits))
check("search results are a closed subset of the safe allowlist (F3)",
      all(set(h.keys()) <= _SAFE_KEYS for h in _hits), detail=str(_hits))

print("== planner-loras assembly (known + unknown + no-DB) ==")
_pl = refine.assemble_planner_loras(_conn, ["detail-tweaker", "unknown-lora"])
check("known name enriched", "0.6-0.9" in _pl[0].get("usage_tips", ""))
check("unknown name → name-only", _pl[1] == {"name": "unknown-lora"})
check("assemble tolerates conn=None", refine.assemble_planner_loras(None, ["x"]) == [{"name": "x"}])
_conn.close()

print("== open_catalog_db degrades (no crash) on absent / corrupt DB ==")
check("absent DB → None", refine.open_catalog_db(os.path.join(_root, "nope.sqlite")) is None)
_corrupt = os.path.join(_root, "corrupt.sqlite")
with open(_corrupt, "wb") as _cf:
    _cf.write(b"this is definitely not a sqlite database header!!")
check("corrupt non-sqlite DB → None (not an uncaught crash)",
      refine.open_catalog_db(_corrupt) is None)
check("valid DB opens read-only", refine.open_catalog_db(_dbp) is not None)

print("== F2/F3 structural guard: refine.py never selects a load-plane column ==")
# ADR-027 F2 disposition promise: refine.py is held to a structural check that a
# future `SELECT abs_path` / `row["abs_path"]` shortcut regresses loudly, not just
# a behavioral test of today's functions. (refine.py legitimately holds abs_path
# on ResolvedLoraOp — a resolver-sourced LOAD target — and legitimately imports
# catalog_db, so the guard is COLUMN-shaped, not an import ban.)
import ast as _ast  # noqa: E402
_LOAD_PLANE_COLS = {"abs_path", "root", "relative_path"}
_src = (Path(__file__).parent / "comfyless" / "refine.py").read_text()
_tree = _ast.parse(_src)

# (a) no SQL string literal names a load-plane column
_sql_viol = []
for _n in _ast.walk(_tree):
    if isinstance(_n, _ast.Constant) and isinstance(_n.value, str):
        _low = _n.value.lower()
        if "select" in _low and "from" in _low:
            for _c in _LOAD_PLANE_COLS:
                if _c in _low:
                    _sql_viol.append((_c, _n.value[:70]))
check("no load-plane column in any SQL literal", _sql_viol == [], detail=str(_sql_viol))

# (b) no string-literal subscript reads a load-plane column (row["abs_path"] etc.)
_sub_viol = []
for _n in _ast.walk(_tree):
    if isinstance(_n, _ast.Subscript) and isinstance(_n.slice, _ast.Constant):
        if _n.slice.value in _LOAD_PLANE_COLS:
            _sub_viol.append(_n.slice.value)
check("no dict/row subscript reads a load-plane column", _sub_viol == [], detail=str(_sub_viol))

# (c) the safe-field allowlists are disjoint from load-plane + audit columns
_FORBIDDEN_ALL = _LOAD_PLANE_COLS | {
    "sha256", "size_bytes", "classification", "reason", "duplicate_of",
    "excluded_reason", "family_conflict"}
check("safe field allowlists disjoint from path/audit columns",
      not (_SAFE_KEYS & _FORBIDDEN_ALL), detail=str(_SAFE_KEYS & _FORBIDDEN_ALL))

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
