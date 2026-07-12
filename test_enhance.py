"""Unit tests for comfyless.enhance (ADR-026 prompt enhancer).

Mock-based — no GPU, no network. The hunyuan-reprompt live path is covered by a
manual smoke (implementation_details.md); here we mock the model + HTTP so the
suite runs anywhere. Run: ./.venv/bin/python3 test_enhance.py
"""
import io
import json
import sys
import tempfile
import types
from pathlib import Path

import comfyless.enhance as E

_passed = 0
_failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  {detail}")


def _write(tmp, name, text):
    p = Path(tmp) / name
    p.write_text(text)
    return str(p)


print("── Backend registry ──")
with tempfile.TemporaryDirectory() as tmp:
    good = _write(tmp, "e.toml",
                  '[hunyuan]\ntype="hunyuan-reprompt"\nmodel="/m"\n'
                  '[gpt]\ntype="openai-endpoint"\nurl="http://x/v1"\n')
    b = E.load_backends(good)
    check("loads valid registry", set(b) == {"hunyuan", "gpt"})
    check("preserves type", b["hunyuan"]["type"] == "hunyuan-reprompt")

    bad_type = _write(tmp, "bt.toml", '[x]\ntype="nonsense"\n')
    try:
        E.load_backends(bad_type); check("unknown type rejected", False)
    except E.EnhanceError:
        check("unknown type rejected", True)

    malformed = _write(tmp, "mal.toml", 'this is not = valid = toml [[[')
    try:
        E.load_backends(malformed); check("malformed rejected", False)
    except E.EnhanceError:
        check("malformed rejected", True)

    try:
        E.load_backends(str(Path(tmp) / "nope.toml")); check("missing file rejected", False)
    except E.EnhanceError:
        check("missing file rejected", True)

    empty = _write(tmp, "empty.toml", "# nothing\n")
    try:
        E.load_backends(empty); check("empty registry rejected", False)
    except E.EnhanceError:
        check("empty registry rejected", True)


print("── Recipes ──")
check("default recipe qwen-image", E.default_recipe_name("qwen-image") == "qwen-image-generic")
check("default recipe unknown → generic", E.default_recipe_name("unknown") == "generic")
check("default recipe None → generic", E.default_recipe_name(None) == "generic")
with tempfile.TemporaryDirectory() as tmp:
    _write(tmp, "generic.toml", 'system_prompt="G"\ntarget="nl"\ntemperature=0.8\n')
    _write(tmp, "sdxl-generic.toml", 'system_prompt="TAGS"\ntarget="tags"\n')
    r = E.load_recipe("generic", tmp)
    check("loads recipe", r["system_prompt"] == "G" and r["target"] == "nl")
    r2 = E.load_recipe("qwen-image-generic", tmp)  # absent → falls back to generic
    check("missing family recipe falls back to generic", r2["system_prompt"] == "G")
    r3 = E.load_recipe("sdxl-generic", tmp)
    check("sdxl recipe distinct", r3["system_prompt"] == "TAGS")
    check("recipe temperature default applied", E.load_recipe("sdxl-generic", tmp)["temperature"] == 0.8)
    nosys = _write(tmp, "nosys.toml", 'target="nl"\n')
    try:
        E.load_recipe("nosys", tmp); check("recipe missing system_prompt rejected", False)
    except E.EnhanceError:
        check("recipe missing system_prompt rejected", True)
    # a genuinely absent recipe with no generic fallback present
    with tempfile.TemporaryDirectory() as tmp2:
        try:
            E.load_recipe("ghost", tmp2); check("absent recipe (no fallback) rejected", False)
        except E.EnhanceError:
            check("absent recipe (no fallback) rejected", True)


print("── _clean_output ──")
check("extracts <answer>", E._clean_output("pre <answer>hello world</answer> post") == "hello world")
check("strips <think>", E._clean_output("<think>reasoning</think>final") == "final")
check("think then answer", E._clean_output("<think>x</think><answer>Y</answer>") == "Y")
check("unclosed answer tolerated", E._clean_output("<answer>truncated text") == "truncated text")
check("no wrapper passthrough", E._clean_output("  just text  ") == "just text")


print("── trust_remote_code hash pin ──")
with tempfile.TemporaryDirectory() as tmp:
    # wrong-content tokenizer → refuse
    (Path(tmp) / "tokenization_hy.py").write_text("# tampered")
    try:
        E._verify_reprompt_tokenizer(tmp); check("hash mismatch refused", False)
    except E.EnhanceError as e:
        check("hash mismatch refused", "does not match" in str(e))
    # missing tokenizer → refuse
    with tempfile.TemporaryDirectory() as tmp2:
        try:
            E._verify_reprompt_tokenizer(tmp2); check("missing tokenizer refused", False)
        except E.EnhanceError as e:
            check("missing tokenizer refused", "missing" in str(e))


print("── dispatch ──")
BK = {"hunyuan": {"type": "hunyuan-reprompt", "model": "/m"},
      "gpt": {"type": "openai-endpoint", "url": "http://x/v1", "model": "m"}}
try:
    E.enhance("x", "nope", backends=BK); check("unknown backend rejected", False)
except E.EnhanceError:
    check("unknown backend rejected", True)

# hunyuan dispatch → mock the backend fn
_orig_h = E.enhance_hunyuan_reprompt
E.enhance_hunyuan_reprompt = lambda text, cfg, n: [f"H:{text}"] * n
try:
    out = E.enhance("cat", "hunyuan", backends=BK, n=2)
    check("hunyuan dispatch + n", out == ["H:cat", "H:cat"])
    # recipe ignored for hunyuan (no recipe lookup crash even with bogus name)
    out2 = E.enhance("cat", "hunyuan", backends=BK, recipe_name="does-not-exist")
    check("hunyuan ignores recipe", out2 == ["H:cat"])
finally:
    E.enhance_hunyuan_reprompt = _orig_h


print("── openai-endpoint via mock urlopen ──")
import urllib.request as _u
_orig_urlopen = _u.urlopen


class _Resp:
    def __init__(self, payload): self._p = json.dumps(payload).encode()
    def read(self): return self._p
    def __enter__(self): return self
    def __exit__(self, *a): return False


_calls = []
def _mock_urlopen(req, timeout=None):
    url = req.full_url
    _calls.append((url, req.data))
    if url.endswith("/models"):
        return _Resp({"data": [{"id": "auto-model"}]})
    if url.endswith("/chat/completions"):
        body = json.loads(req.data)
        # echo the user content so we can assert the recipe system prompt rode along
        return _Resp({"choices": [{"message": {"content": "ENH:" + body["messages"][1]["content"]}}]})
    raise AssertionError("unexpected url " + url)


with tempfile.TemporaryDirectory() as tmp:
    _write(tmp, "generic.toml", 'system_prompt="SYS"\ntarget="nl"\ntemperature=0.7\n')
    _u.urlopen = _mock_urlopen
    try:
        # model omitted → resolved from /v1/models
        bk = {"g": {"type": "openai-endpoint", "url": "http://x/v1"}}
        out = E.enhance("a cat", "g", backends=bk, family="qwen-image",
                        recipes_dir=tmp, n=1)
        check("openai enhance returns content", out == ["ENH:a cat"])
        check("model auto-resolved from /models", any(u.endswith("/models") for u, _ in _calls))
        # system prompt from recipe present in the request body
        chat = [d for u, d in _calls if u.endswith("/chat/completions")][0]
        sent = json.loads(chat)
        check("recipe system_prompt used", sent["messages"][0]["content"] == "SYS")
        check("temperature from recipe", sent["temperature"] == 0.7)
        # n variations → n POST calls
        _calls.clear()
        out3 = E.enhance("a dog", "g", backends=bk, recipes_dir=tmp, n=3)
        posts = [u for u, _ in _calls if u.endswith("/chat/completions")]
        check("n=3 → 3 completion calls", len(posts) == 3 and len(out3) == 3)
    finally:
        _u.urlopen = _orig_urlopen

# HTTP error → EnhanceError with backend context
import urllib.error as _ue
def _err_urlopen(req, timeout=None):
    if req.full_url.endswith("/models"):
        return _Resp({"data": [{"id": "m"}]})
    raise _ue.HTTPError(req.full_url, 500, "boom", {}, io.BytesIO(b"server error"))
with tempfile.TemporaryDirectory() as tmp:
    _write(tmp, "generic.toml", 'system_prompt="SYS"\n')
    _u.urlopen = _err_urlopen
    try:
        E.enhance("x", "g", backends={"g": {"type": "openai-endpoint", "url": "http://x/v1", "model": "m"}},
                  recipes_dir=tmp)
        check("openai HTTP error → EnhanceError", False)
    except E.EnhanceError as e:
        check("openai HTTP error → EnhanceError", "500" in str(e))
    finally:
        _u.urlopen = _orig_urlopen


print("── fail-loud on empty / bad config (review fixes) ──")
# empty enhancement → EnhanceError (never degrade to empty prompt)
def _empty_urlopen(req, timeout=None):
    if req.full_url.endswith("/models"):
        return _Resp({"data": [{"id": "m"}]})
    return _Resp({"choices": [{"message": {"content": "   <answer></answer>  "}}]})
with tempfile.TemporaryDirectory() as tmp:
    _write(tmp, "generic.toml", 'system_prompt="S"\n')
    _u.urlopen = _empty_urlopen
    try:
        E.enhance("x", "g", backends={"g": {"type": "openai-endpoint", "url": "http://x/v1", "model": "m"}}, recipes_dir=tmp)
        check("empty enhancement → EnhanceError", False)
    except E.EnhanceError as e:
        check("empty enhancement → EnhanceError", "empty" in str(e))
    finally:
        _u.urlopen = _orig_urlopen
# non-numeric temperature → EnhanceError (not a raw ValueError)
with tempfile.TemporaryDirectory() as tmp:
    _write(tmp, "bad.toml", 'system_prompt="S"\ntemperature="hot"\n')
    try:
        E.load_recipe("bad", tmp); check("non-numeric temperature → EnhanceError", False)
    except E.EnhanceError as e:
        check("non-numeric temperature → EnhanceError", "temperature" in str(e))
# _clean_output drops an unclosed <think> tail
check("unclosed <think> tail dropped", E._clean_output("<think>reasoning without close") == "")
check("closed think then answer still works", E._clean_output("<think>a</think><answer>keep</answer>") == "keep")


print("── offline enhance_prompt_list ──")
_orig_h2 = E.enhance_hunyuan_reprompt
# deterministic mock: variant vi of prompt p -> "p#vi"
E.enhance_hunyuan_reprompt = lambda text, cfg, n: [f"{text}#{i}" for i in range(n)]
try:
    bk = {"h": {"type": "hunyuan-reprompt", "model": "/m"}}
    enh, prov = E.enhance_prompt_list(["a", "b"], "h", backends=bk, variations=3)
    check("list×variations length", len(enh) == 6 and len(prov) == 6)
    check("source-major variation-minor order", enh == ["a#0", "a#1", "a#2", "b#0", "b#1", "b#2"])
    check("provenance source_index", [p["source_index"] for p in prov] == [0, 0, 0, 1, 1, 1])
    check("provenance variation_index", [p["variation_index"] for p in prov] == [0, 1, 2, 0, 1, 2])
    check("provenance source_prompt", prov[4]["source_prompt"] == "b")
    enh1, _ = E.enhance_prompt_list(["x"], "h", backends=bk, variations=1)
    check("variations=1 → 1:1", enh1 == ["x#0"])
finally:
    E.enhance_hunyuan_reprompt = _orig_h2


print(f"\n{'='*50}\n  {_passed} passed, {_failed} failed\n{'='*50}")
sys.exit(1 if _failed else 0)
