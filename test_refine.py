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

# ══════════════════════════════════════════════════════════════════════════════
#  Slice 3 — loop controller (greedy hill-climb) pure surface
# ══════════════════════════════════════════════════════════════════════════════
from comfyless.refine import (  # noqa: E402
    WorkingConfig, LoraSlot, Candidate, Verdict, LoraOp, ResolvedLoraOp,
    composite_score, verdict_passes, apply_overrides, build_judge_user_text,
    verdict_record, _assert_no_paths, DEFAULT_W_PA, DEFAULT_W_AES,
)

print("\n== slice 3: composite scoring + pass gate ==")
check("composite weights prompt-adherence above aesthetics",
      composite_score(10, 5, 0.6, 0.4) == 8.0,
      detail=str(composite_score(10, 5, 0.6, 0.4)))
check("composite uses module defaults",
      composite_score(10, 0) == DEFAULT_W_PA * 10 + DEFAULT_W_AES * 0)
# The pass gate is NUMERIC (both axes >= threshold); the advisory verdict string
# is deliberately ignored so a lying judge cannot self-promote (F8).
check("pass when both axes >= threshold",
      verdict_passes(Verdict(8, 8, "revise", {}, None, []), 8) is True)
check("advisory 'revise' does not block a numeric pass",
      verdict_passes(Verdict(9, 8, "revise", {}, None, []), 8) is True)
check("fail when one axis below threshold",
      verdict_passes(Verdict(10, 7, "pass", {}, None, []), 8) is False)
check("advisory 'pass' cannot self-promote past the numeric gate",
      verdict_passes(Verdict(3, 3, "pass", {}, None, []), 8) is False)

print("\n== slice 3: apply_overrides — prompt ==")
_base_cfg = WorkingConfig(prompt="orig", loras=[], base={"seed": 42})
_v_prompt = Verdict(5, 5, "revise", {}, "rewritten", [])
_c1 = apply_overrides(_base_cfg, _v_prompt, [], [])
check("override prompt replaces prompt", _c1.prompt == "rewritten")
_c2 = apply_overrides(_base_cfg, Verdict(5, 5, "revise", {}, None, []), [], [])
check("absent override prompt keeps prior prompt", _c2.prompt == "orig")
check("apply_overrides is pure — source cfg prompt unchanged",
      _base_cfg.prompt == "orig")

print("\n== slice 3: apply_overrides — LoRA add/remove/set_weight ==")
_cfg_x = WorkingConfig(prompt="p",
                       loras=[LoraSlot("x", "/root/x.safetensors", 1.0)],
                       base={"seed": 1})


def _rop(name, action, weight, path):
    return ResolvedLoraOp(op=LoraOp(name, action, weight),
                          resolved_name=name, abs_path=path)


# add a new one (default weight when None), remove an active one
_notices = []
_c = apply_overrides(
    _cfg_x, Verdict(5, 5, "revise", {}, None, []),
    [_rop("y", "add", None, "/root/y.safetensors"),
     _rop("x", "remove", None, "/root/x.safetensors")], _notices)
check("add appends the new LoRA, remove drops the active one",
      _c.lora_names() == ["y"], detail=str(_c.lora_names()))
check("add with no weight defaults to 1.0", _c.loras[0].weight == 1.0)
check("apply_overrides pure — source LoRA set unchanged",
      _cfg_x.lora_names() == ["x"])

# add an already-active LoRA → noticed no-op
_n2 = []
_c = apply_overrides(_cfg_x, Verdict(5, 5, "revise", {}, None, []),
                     [_rop("x", "add", 0.5, "/root/x.safetensors")], _n2)
check("add of an already-active LoRA is a no-op", _c.lora_names() == ["x"])
check("add of active keeps prior weight (no-op, not reweight)",
      _c.loras[0].weight == 1.0)
check("add-active emits a notice", any("already active" in n for n in _n2))

# remove a LoRA that isn't active → noticed no-op
_n3 = []
_c = apply_overrides(_cfg_x, Verdict(5, 5, "revise", {}, None, []),
                     [_rop("z", "remove", None, "/root/z.safetensors")], _n3)
check("remove of an inactive LoRA is a no-op", _c.lora_names() == ["x"])
check("remove-inactive emits a notice", any("not active" in n for n in _n3))

# set_weight on active → updates; on inactive → adds at that weight (+notice)
_n4 = []
_c = apply_overrides(_cfg_x, Verdict(5, 5, "revise", {}, None, []),
                     [_rop("x", "set_weight", 2.5, "/root/x.safetensors")], _n4)
check("set_weight updates an active LoRA's weight", _c.loras[0].weight == 2.5)
_n5 = []
_c = apply_overrides(_cfg_x, Verdict(5, 5, "revise", {}, None, []),
                     [_rop("w", "set_weight", 1.5, "/root/w.safetensors")], _n5)
check("set_weight on inactive LoRA adds it at that weight",
      _c.lora_names() == ["x", "w"] and _c.loras[1].weight == 1.5)
check("set_weight-inactive emits a notice", any("inactive" in n for n in _n5))

# insertion order preserved across a multi-op merge
_cfg_ab = WorkingConfig(prompt="p", loras=[
    LoraSlot("a", "/root/a.safetensors", 1.0),
    LoraSlot("b", "/root/b.safetensors", 1.0)], base={})
_c = apply_overrides(_cfg_ab, Verdict(5, 5, "revise", {}, None, []),
                     [_rop("c", "add", 1.0, "/root/c.safetensors")], [])
check("insertion order preserved on add", _c.lora_names() == ["a", "b", "c"])

print("\n== slice 3: WorkingConfig.to_generate_params ==")
_p = _cfg_x.to_generate_params()
check("to_generate_params sets prompt", _p["prompt"] == "p")
check("to_generate_params carries the LOAD path in loras",
      _p["loras"] == [{"path": "/root/x.safetensors", "weight": 1.0}])
check("to_generate_params copies base (mutation isolation)",
      (_p.__setitem__("seed", 999), _cfg_x.base["seed"] == 1)[1])

print("\n== slice 3: F3 — no load-plane path ever reaches the LLM or verdict.json ==")
# active LoRAs hold an abs_path internally; the judge text must render name+weight
# only. A regression that leaks the path fails HERE, loudly.
_jt = build_judge_user_text("target scene", _cfg_x,
                            [{"name": "x", "description": "a detail LoRA"}])
check("judge text omits abs_path of active LoRAs", "/root/" not in _jt)
check("judge text omits .safetensors basename", ".safetensors" not in _jt)
check("judge text carries the target prompt", "target scene" in _jt)
check("judge text carries safe catalog metadata", "a detail LoRA" in _jt)

# _assert_no_paths is the last structural gate — it must fire on any path key at
# any nesting depth, and pass a clean payload.
raises("_assert_no_paths fires on top-level abs_path",
       lambda: _assert_no_paths({"abs_path": "/x"}))
raises("_assert_no_paths fires on nested path key",
       lambda: _assert_no_paths({"a": [{"b": {"root": "/x"}}]}))
raises("_assert_no_paths fires on 'relative_path'",
       lambda: _assert_no_paths({"relative_path": "y"}))
try:
    _assert_no_paths({"name": "x", "weight": 1.0, "meta": [{"description": "d"}]})
    check("_assert_no_paths passes a clean payload", True)
except RefineError:
    check("_assert_no_paths passes a clean payload", False)

# a leaking search_offer (upstream projection regressed) is caught before the
# text is assembled — proves the gate protects independently of the projection.
raises("build_judge_user_text refuses a path-bearing search offer",
       lambda: build_judge_user_text(
           "t", _cfg_x, [], search_offers=[{"name": "n", "abs_path": "/leak"}]))

print("\n== slice 3: verdict_record is a path-free audit artifact ==")
_vr = Verdict(6, 7, "revise",
              {"prompt_adherence": "close", "aesthetics": "ok"},
              "try adding rim light",
              [LoraOp("detail-lora", "add", 0.8),
               LoraOp("x", "remove", None)])
_cand = Candidate(index=2, image_path="/root/candidate_02.png",
                  metadata={"seed": 42, "loras": [{"path": "/root/x.safetensors"}]},
                  verdict=_vr, composite=6.4)
_rec = verdict_record(_cand, DEFAULT_W_PA, DEFAULT_W_AES)
_recs = json.dumps(_rec)
check("verdict_record contains no filesystem path", "/root/" not in _recs)
check("verdict_record contains no .safetensors", ".safetensors" not in _recs)
check("verdict_record records the raw proposed LoRA NAMES (not paths)",
      [o["name"] for o in _rec["proposed_overrides"]["loras"]]
      == ["detail-lora", "x"])
check("verdict_record carries the numeric scores",
      _rec["scores"] == {"prompt_adherence": 6, "aesthetics": 7})
check("verdict_record records the composite + weights",
      _rec["composite"] == 6.4 and _rec["weights"]["prompt_adherence"] == DEFAULT_W_PA)
check("verdict_record keeps the advisory verdict string",
      _rec["verdict"] == "revise")
# defense in depth: if abs_path ever crept into the record it would raise here —
# verdict_record calls _assert_no_paths internally, so a clean call is the proof.

print("\n== slice 3: _daemon_namespace carries the wire-builder's attributes ==")
_ns = refine._daemon_namespace("cuda:1", "bf16", "/out/cand")
check("_daemon_namespace pins device/precision/savepath",
      _ns.device == "cuda:1" and _ns.precision == "bf16"
      and _ns.savepath == "/out/cand")
check("_daemon_namespace defaults rebalance off (no accidental Krea path)",
      _ns.rebalance is False and _ns.rebalance_weights is None)

# ══════════════════════════════════════════════════════════════════════════════
#  Slice 3 — refine_loop controller (monkeypatched generation + judge)
#
#  run_generation (daemon/cold) and judge_candidate pull torch / a live endpoint,
#  so the LOOP is exercised by replacing those two module-level callables with
#  fakes: a fake generator that writes a real tiny PNG to the canonical path and
#  reports a seed, and a fake judge that replays a scripted sequence of verdicts
#  (or raises). This pins the pass/cap/patience stop logic, F7 iteration
#  consumption, seed pinning, and winner finalization — the surface the code
#  review flagged as untested. (run_generation's own daemon-move glue stays
#  integration-tested by the hot GPU run, like _post_judge.)
# ══════════════════════════════════════════════════════════════════════════════
import tempfile as _tf  # noqa: E402
from PIL import Image as _PILImage  # noqa: E402


def _mkverdict(pa, aes, vstr="revise"):
    return Verdict(pa, aes, vstr, {}, None, [])


class _FakeGen:
    """Stands in for refine.run_generation: writes a 4px PNG at the canonical path
    and records the seed each call actually saw (to prove seed pinning)."""
    def __init__(self, seed=123):
        self.seed = seed
        self.seeds_seen = []

    def __call__(self, cfg, *, device, output_dir, stem, precision="bf16", log=print):
        self.seeds_seen.append(cfg.base.get("seed"))
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{stem}.png")
        _PILImage.new("RGB", (4, 4), (7, 8, 9)).save(path)
        return refine.GenOutcome(image_path=path, metadata={"seed": self.seed})


class _FakeJudge:
    """Replays a scripted list of Verdict (or Exception to raise) — the last entry
    repeats if the loop runs longer than the script."""
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def __call__(self, image, target_prompt, cfg, backend_cfg, planner_loras, **kw):
        item = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


def _run_loop(script, *, max_iter=10, patience=2, threshold=8, seed=123):
    d = _tf.mkdtemp(prefix="refine_loop_test_")
    fg, fj = _FakeGen(seed=seed), _FakeJudge(script)
    _rg, _jc = refine.run_generation, refine.judge_candidate
    refine.run_generation, refine.judge_candidate = fg, fj
    try:
        cfg = WorkingConfig(prompt="p", loras=[], base={"seed": -1})
        out = refine.refine_loop(
            cfg, target_prompt="a detailed test scene", catalog={}, roots=(),
            conn=None, backend_cfg={"url": "http://x", "model": "m"},
            output_dir=d, device="cuda", pass_threshold=threshold,
            max_iterations=max_iter, patience=patience, log=lambda *_a: None)
    finally:
        refine.run_generation, refine.judge_candidate = _rg, _jc
    return d, out, fg, fj


print("\n== slice 3: refine_loop — pass stops + finalizes the passing candidate ==")
_d, _o, _fg, _fj = _run_loop([_mkverdict(9, 9)], threshold=8)
check("pass on iter 0 stops immediately", _o.iterations == 1)
check("pass sets passed=True", _o.passed is True)
check("winner file exists", _o.winner_path and os.path.isfile(_o.winner_path))
check("winners/ holds exactly one image",
      os.listdir(os.path.join(_d, "winners")) == ["candidate_00.png"])
check("candidate image + sidecar + verdict.json all written",
      all(os.path.isfile(os.path.join(_d, "candidates", f))
          for f in ("candidate_00.png", "candidate_00.json",
                    "candidate_00.verdict.json")))

print("\n== slice 3: refine_loop — cap stop with best-composite winner ==")
_d, _o, _fg, _fj = _run_loop([_mkverdict(5, 5)], max_iter=3, patience=99)
check("runs to the iteration cap", _o.iterations == 3)
check("cap stop is not a pass", _o.passed is False)
check("winner is finalized even without a pass",
      _o.winner_path and os.path.isfile(_o.winner_path))
check("best composite recorded", _o.best_composite == 5.0)

print("\n== slice 3: refine_loop — patience stops the run early ==")
_d, _o, _fg, _fj = _run_loop(
    [_mkverdict(7, 7), _mkverdict(5, 5), _mkverdict(5, 5), _mkverdict(5, 5)],
    max_iter=10, patience=2)
check("patience halts after 2 non-improving iters", _o.iterations == 3)
check("best-so-far (iter 0) wins on a patience stop", _o.best_composite == 7.0)
check("winner basename is the best candidate (iter 0)",
      os.path.basename(_o.winner_path) == "candidate_00.png")

print("\n== slice 3: refine_loop — F7 malformed verdict consumes an iteration ==")
_d, _o, _fg, _fj = _run_loop([RefineError("bad json"), RefineError("bad json")],
                             max_iter=10, patience=2)
check("two unusable verdicts hit patience and stop", _fj.calls == 2)
check("no candidate ever passed → winners/ empty, winner_path None",
      _o.winner_path is None and _o.passed is False)
check("the parse failure is recorded in the candidate's verdict.json",
      json.load(open(os.path.join(_d, "candidates", "candidate_00.verdict.json")))
      .get("error", "").startswith("bad json"))
check("winners/ directory is empty after an all-failure run",
      os.listdir(os.path.join(_d, "winners")) == [])

print("\n== slice 3: refine_loop — seed pinned after iteration 0 ==")
_d, _o, _fg, _fj = _run_loop([_mkverdict(5, 5)], max_iter=3, patience=99, seed=777)
# iter 0 runs with the CLI seed (-1); every later iter must reuse the pinned seed
# from iter 0's metadata (777) so the hill-climb varies only prompt/LoRA.
check("iteration 0 uses the initial seed", _fg.seeds_seen[0] == -1)
check("iterations 1+ reuse the pinned seed",
      _fg.seeds_seen[1:] == [777, 777], detail=str(_fg.seeds_seen))

print("\n== slice 3: refine_loop — no-pass run finalizes top composite ==")
_d, _o, _fg, _fj = _run_loop(
    [_mkverdict(6, 6), _mkverdict(4, 4), _mkverdict(4, 4)], max_iter=10, patience=2)
check("winner is the highest-composite candidate", _o.best_composite == 6.0)
check("winner file is iter 0's candidate",
      os.path.basename(_o.winner_path) == "candidate_00.png")

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
