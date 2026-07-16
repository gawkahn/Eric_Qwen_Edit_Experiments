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
    # ADR-026 (2026-07-14): load_recipe no longer force-defaults temperature —
    # the default (0.8) + recipe>cfg precedence moved to the endpoint resolver so
    # a backend cfg can supply any knob the recipe omits.
    check("recipe WITHOUT temperature leaves it unset (resolver defaults)",
          "temperature" not in E.load_recipe("sdxl-generic", tmp))
    check("recipe WITH temperature keeps the file value",
          E.load_recipe("generic", tmp)["temperature"] == 0.8)
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


print("── ADR-026: endpoint sampling precedence (recipe > cfg > default) ──")

# Resolver precedence
_rc = {"temperature": 0.95, "top_p": 0.9}
_cc = {"temperature": 0.99, "top_p": 0.5, "top_k": 20, "repetition_penalty": 1.05}
_s = E._resolve_endpoint_sampling(_rc, _cc)
check("recipe temperature overrides cfg", _s["temperature"] == 0.95)
check("recipe top_p overrides cfg", _s["top_p"] == 0.9)
check("cfg supplies top_k when recipe omits it", _s["top_k"] == 20)
check("cfg supplies repetition_penalty when recipe omits it", _s["repetition_penalty"] == 1.05)
check("cfg temperature used when recipe omits it",
      E._resolve_endpoint_sampling({}, {"temperature": 0.99})["temperature"] == 0.99)
_sn = E._resolve_endpoint_sampling({}, {})
check("no knobs set -> only temperature default 0.8", _sn == {"temperature": 0.8})
check("top_k/repetition_penalty NOT emitted unless set",
      "top_k" not in _sn and "repetition_penalty" not in _sn)
# min_p (vLLM extension) follows the same recipe>cfg>default emit-only-when-set rule
check("min_p NOT emitted unless set", "min_p" not in _sn)
check("recipe min_p overrides cfg",
      E._resolve_endpoint_sampling({"min_p": 0.05}, {"min_p": 0.2})["min_p"] == 0.05)
check("cfg supplies min_p when recipe omits it",
      E._resolve_endpoint_sampling({}, {"min_p": 0.1})["min_p"] == 0.1)

# Payload construction: mock the HTTP POST + model resolve, capture the wire payload
_captured_payloads = []
_orig_post, _orig_rm = E._post_chat, E._resolve_endpoint_model
E._post_chat = lambda endpoint, payload, key: (_captured_payloads.append(payload) or ["ENHANCED"])
E._resolve_endpoint_model = lambda url, key, m: (m or "test-model")
try:
    out = E.enhance_openai_endpoint(
        "a cat",
        {"type": "openai-endpoint", "url": "http://x/v1", "model": "M",
         "top_k": 20, "repetition_penalty": 1.05},
        {"system_prompt": "SP", "temperature": 0.95, "top_p": 0.9, "min_p": 0.05}, 1)
    check("endpoint returns enhanced text", out == ["ENHANCED"])
    _pay = _captured_payloads[-1]
    check("payload temperature = recipe 0.95", _pay["temperature"] == 0.95)
    check("payload top_p = recipe 0.9", _pay["top_p"] == 0.9)
    check("payload min_p = recipe 0.05", _pay["min_p"] == 0.05)
    check("payload top_k = cfg fallback 20", _pay["top_k"] == 20)
    check("payload repetition_penalty = cfg fallback 1.05", _pay["repetition_penalty"] == 1.05)
    check("payload carries recipe system prompt", _pay["messages"][0]["content"] == "SP")

    _captured_payloads.clear()
    E.enhance_openai_endpoint(
        "a cat", {"type": "openai-endpoint", "url": "http://x/v1", "model": "M"},
        {"system_prompt": "SP"}, 1)
    _pay2 = _captured_payloads[-1]
    check("clean run emits temperature default 0.8", _pay2["temperature"] == 0.8)
    check("clean run sends NO top_k/repetition_penalty/min_p (OpenAI-standard)",
          "top_k" not in _pay2 and "repetition_penalty" not in _pay2
          and "min_p" not in _pay2)
finally:
    E._post_chat, E._resolve_endpoint_model = _orig_post, _orig_rm

# Recipe sampling-knob type validation
with tempfile.TemporaryDirectory() as tmpc:
    for _label, _toml in (
        ("non-int-str top_k", 'system_prompt="S"\ntop_k="notint"\n'),
        # bool → int(True)=1 would be a drastic silent change; reject it.
        ("bool top_k", 'system_prompt="S"\ntop_k=true\n'),
        ("bool temperature", 'system_prompt="S"\ntemperature=true\n'),
        # min_p is a recognized knob → same bool rejection as the others.
        ("bool min_p", 'system_prompt="S"\nmin_p=true\n'),
        # non-integer float top_k must not silently truncate to 20.
        ("non-integer-float top_k", 'system_prompt="S"\ntop_k=20.5\n'),
    ):
        _write(tmpc, "bad.toml", _toml)
        try:
            E.load_recipe("bad", tmpc); check(f"recipe {_label} rejected", False)
        except E.EnhanceError:
            check(f"recipe {_label} rejected", True)
    _write(tmpc, "ok.toml",
           'system_prompt="S"\ntop_k=40\nrepetition_penalty=1.1\ntop_p=0.8\nmin_p=0.05\n')
    _ok = E.load_recipe("ok", tmpc)
    check("recipe coerces top_k to int",
          _ok["top_k"] == 40 and isinstance(_ok["top_k"], int))
    check("recipe coerces repetition_penalty to float", _ok["repetition_penalty"] == 1.1)
    check("recipe coerces min_p to float",
          _ok["min_p"] == 0.05 and isinstance(_ok["min_p"], float))
    # top_k as an integer-valued float (40.0) is accepted and narrowed to int.
    _write(tmpc, "okf.toml", 'system_prompt="S"\ntop_k=40.0\n')
    check("recipe integer-valued-float top_k accepted → int",
          E.load_recipe("okf", tmpc)["top_k"] == 40)

# cfg-sourced bad types raise EnhanceError (clean message), not a raw ValueError
for _cfgbad in ({"temperature": "hot"}, {"top_k": True}, {"top_p": "x"},
                {"min_p": True}):
    try:
        E._resolve_endpoint_sampling({}, _cfgbad)
        check(f"cfg bad type {_cfgbad} raises EnhanceError", False)
    except E.EnhanceError:
        check(f"cfg bad type {_cfgbad} raises EnhanceError", True)
    except Exception as _e:  # noqa: BLE001 — a raw ValueError here is the bug
        check(f"cfg bad type {_cfgbad} raises EnhanceError",
              False, f"got {type(_e).__name__}")

# Loud warning on the bogus 'batch' key (feedback: warn, don't block)
import io as _io
from contextlib import redirect_stderr as _rse
with tempfile.TemporaryDirectory() as tmpb:
    _reg = _write(tmpb, "enh.toml",
                  '[qwen]\ntype="openai-endpoint"\nurl="http://x/v1"\nbatch=5\n')
    _buf = _io.StringIO()
    with _rse(_buf):
        _b = E.load_backends(_reg)
    check("bogus 'batch' key still loads the backend (warn, not block)",
          "qwen" in _b)
    check("bogus 'batch' key emits loud warning naming batch_variations",
          "batch_variations" in _buf.getvalue(), _buf.getvalue())


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
E.enhance_hunyuan_reprompt = lambda text, cfg, n, device_override=None: [f"H:{text}"] * n
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


print("── openai variation diversity (seed per variation + top_p) ──")
with tempfile.TemporaryDirectory() as tmp:
    _write(tmp, "generic.toml", 'system_prompt="S"\ntemperature=0.9\ntop_p=0.95\n')
    _calls.clear()
    _u.urlopen = _mock_urlopen
    try:
        bk = {"g": {"type": "openai-endpoint", "url": "http://x/v1", "model": "m"}}
        E.enhance("a cat", "g", backends=bk, recipes_dir=tmp, n=3)
        posts = [json.loads(d) for u, d in _calls if u.endswith("/chat/completions")]
        check("distinct seed per variation", [p.get("seed") for p in posts] == [0, 1, 2])
        check("top_p from recipe sent", all(p.get("top_p") == 0.95 for p in posts))
        check("temperature from recipe sent", all(p.get("temperature") == 0.9 for p in posts))
        _calls.clear()
        E.enhance("a dog", "g", backends=bk, recipes_dir=tmp, n=1)
        p1 = [json.loads(d) for u, d in _calls if u.endswith("/chat/completions")][0]
        check("n=1 (inline) sends no seed", "seed" not in p1)
    finally:
        _u.urlopen = _orig_urlopen


print("── openai batch_variations (n-param throughput) ──")
def _batch_urlopen(req, timeout=None):
    _calls.append((req.full_url, req.data))
    if req.full_url.endswith("/models"):
        return _Resp({"data": [{"id": "m"}]})
    b = json.loads(req.data)
    nreq = b.get("n", 1)  # honor n → nreq distinct choices
    return _Resp({"choices": [{"message": {"content": f"V{k}:" + b["messages"][1]["content"]}} for k in range(nreq)]})
with tempfile.TemporaryDirectory() as tmp:
    _write(tmp, "generic.toml", 'system_prompt="S"\ntemperature=0.9\n')
    _calls.clear(); _u.urlopen = _batch_urlopen
    try:
        bk = {"g": {"type": "openai-endpoint", "url": "http://x/v1", "model": "m", "batch_variations": True}}
        out = E.enhance("a cat", "g", backends=bk, recipes_dir=tmp, n=4)
        posts = [json.loads(d) for u, d in _calls if u.endswith("/chat/completions")]
        check("batch: ONE request for N variations", len(posts) == 1 and posts[0].get("n") == 4)
        check("batch: N choices unpacked", out == ["V0:a cat", "V1:a cat", "V2:a cat", "V3:a cat"])
    finally:
        _u.urlopen = _orig_urlopen
def _ignores_n_urlopen(req, timeout=None):
    if req.full_url.endswith("/models"):
        return _Resp({"data": [{"id": "m"}]})
    return _Resp({"choices": [{"message": {"content": "only one"}}]})  # ignores n
with tempfile.TemporaryDirectory() as tmp:
    _write(tmp, "generic.toml", 'system_prompt="S"\n')
    _u.urlopen = _ignores_n_urlopen
    try:
        E.enhance("x", "g", backends={"g": {"type": "openai-endpoint", "url": "http://x/v1", "model": "m", "batch_variations": True}}, recipes_dir=tmp, n=3)
        check("batch: server ignoring n → clear error", False)
    except E.EnhanceError as e:
        check("batch: server ignoring n → clear error", "1 of 3" in str(e))
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
E.enhance_hunyuan_reprompt = lambda text, cfg, n, device_override=None: [f"{text}#{i}" for i in range(n)]
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


print("── offline concurrency (order-preserving) ──")
with tempfile.TemporaryDirectory() as tmp:
    _write(tmp, "generic.toml", 'system_prompt="S"\n')
    _u.urlopen = _mock_urlopen  # echoes "ENH:<user content>"
    try:
        bk = {"g": {"type": "openai-endpoint", "url": "http://x/v1", "model": "m"}}
        prompts = [f"p{i}" for i in range(8)]
        enh, prov = E.enhance_prompt_list(prompts, "g", backends=bk, recipes_dir=tmp, concurrency=4)
        check("concurrency preserves source order", enh == [f"ENH:p{i}" for i in range(8)])
        check("concurrency processes all + provenance ordered",
              len(enh) == 8 and [p["source_index"] for p in prov] == list(range(8)))
        enh2, _ = E.enhance_prompt_list(["a", "b"], "g", backends=bk, recipes_dir=tmp, variations=2, concurrency=2)
        check("concurrency + variations flat length", len(enh2) == 4)
    finally:
        _u.urlopen = _orig_urlopen

print("── hunyuan device override ──")
_cap = {}
_orig_h3 = E.enhance_hunyuan_reprompt
E.enhance_hunyuan_reprompt = lambda text, cfg, n, device_override=None: (_cap.update(dev=device_override), ["x"])[1]
try:
    E.enhance("p", "h", backends={"h": {"type": "hunyuan-reprompt", "model": "/m", "device": "cuda:0"}}, device="cuda:1")
    check("gen --device overrides hunyuan backend device", _cap.get("dev") == "cuda:1")
    _cap.clear()
    E.enhance("p", "h", backends={"h": {"type": "hunyuan-reprompt", "model": "/m", "device": "cuda:0"}})
    check("no override → backend device used (None passed, cfg wins downstream)", _cap.get("dev") is None)
finally:
    E.enhance_hunyuan_reprompt = _orig_h3


print("── reprompt fp8 quant plumbing ──")
import inspect as _insp
_esrc = _insp.getsource(E)
check("reprompt cache key includes quant", "precision}|{quant}" in _esrc)
check("reprompt fp8 = weight-only recipe", "Float8WeightOnlyConfig" in _esrc)
check("reprompt rejects non-fp8 quant (hard)", "unsupported (only 'fp8')" in _esrc)
check("reprompt fp8 application failure warns-and-skips (not raises)", "reprompt model left in" in _esrc)
check("enhance_hunyuan_reprompt reads cfg quant", 'cfg.get("quant"' in _esrc)
check("reprompt quant passed to loader", "_load_reprompt(model_dir, device, precision, quant)" in _esrc)


# ── _cli overwrite guard ─────────────────────────────────────────────────────
print("== _cli: warn/confirm on existing output ==")
import os as _os  # noqa: E402
import builtins as _builtins  # noqa: E402

_orig_input, _orig_stdin = _builtins.input, sys.stdin
_orig_lb, _orig_epl = E.load_backends, E.enhance_prompt_list
_epl_calls = []
def _fake_epl(prompts, backend, **kw):
    _epl_calls.append(list(prompts))
    # Real enhance_prompt_list returns (enhanced list, provenance LIST[dict]).
    return (["ENHANCED-" + p for p in prompts],
            [{"backend": backend, "src": p} for p in prompts])
def _tty(val):
    return types.SimpleNamespace(isatty=lambda: val)
def _boom(*_a):
    raise AssertionError("input() should not be called here")
try:
    E.load_backends = lambda *a, **k: {"x": {"type": "openai-endpoint", "url": "http://x/v1"}}
    E.enhance_prompt_list = _fake_epl
    with tempfile.TemporaryDirectory() as _td:
        _inp = _os.path.join(_td, "in.json")
        with open(_inp, "w") as _f:
            json.dump(["a cat", "a dog"], _f)
        _out = _os.path.join(_td, "out.json")

        # (1) existing output + interactive "n" → abort (exit 1), file untouched,
        # and the expensive enhancement is NOT run (guard is before the LLM call).
        with open(_out, "w") as _f:
            _f.write("SENTINEL")
        _epl_calls.clear()
        sys.stdin = _tty(True); _builtins.input = lambda *a: "n"
        _rc = E._cli([_inp, "--backend", "x", "-o", _out])
        check("interactive 'n' aborts with exit 1", _rc == 1)
        check("declined overwrite leaves the file untouched", open(_out).read() == "SENTINEL")
        check("declined overwrite skips the enhance run", _epl_calls == [])

        # (2) existing output + interactive "y" → overwrite
        _builtins.input = lambda *a: "y"
        _rc = E._cli([_inp, "--backend", "x", "-o", _out, "--no-provenance"])
        check("interactive 'y' proceeds (exit 0)", _rc == 0)
        check("confirmed overwrite replaces the file",
              json.load(open(_out)) == ["ENHANCED-a cat", "ENHANCED-a dog"])

        # (3) existing output + NON-interactive stdin → warn + proceed, no prompt
        with open(_out, "w") as _f:
            _f.write("SENTINEL2")
        sys.stdin = _tty(False); _builtins.input = _boom
        _rc = E._cli([_inp, "--backend", "x", "-o", _out, "--no-provenance"])
        check("non-interactive proceeds without prompting (exit 0)", _rc == 0)
        check("non-interactive overwrote the file", json.load(open(_out))[0] == "ENHANCED-a cat")

        # (3b) proceeding path WITH provenance → both files written, sidecar
        # content matches the enhance return (exercises the prov_path branch).
        _out3 = _os.path.join(_td, "out3.json")
        sys.stdin = _tty(True); _builtins.input = _boom  # fresh path, no prompt
        _rc = E._cli([_inp, "--backend", "x", "-o", _out3])
        _prov3 = _os.path.join(_td, "out3.provenance.json")
        check("provenance sidecar is written on the proceeding path",
              _rc == 0 and _os.path.exists(_out3) and _os.path.exists(_prov3))
        check("sidecar content matches the enhance provenance list",
              json.load(open(_prov3)) == [{"backend": "x", "src": "a cat"},
                                          {"backend": "x", "src": "a dog"}])

        # (4) fresh output (nothing to clobber) → no prompt at all
        _fresh = _os.path.join(_td, "fresh.json")
        sys.stdin = _tty(True); _builtins.input = _boom
        _rc = E._cli([_inp, "--backend", "x", "-o", _fresh, "--no-provenance"])
        check("no prompt when output does not exist (exit 0)", _rc == 0)
        check("fresh output was written", _os.path.exists(_fresh))

        # (5) an existing PROVENANCE sidecar alone also triggers the guard
        _out2 = _os.path.join(_td, "out2.json")
        with open(_os.path.join(_td, "out2.provenance.json"), "w") as _f:
            _f.write("OLDPROV")
        _seen = []
        sys.stdin = _tty(True); _builtins.input = lambda *a: (_seen.append(1) or "n")
        _rc = E._cli([_inp, "--backend", "x", "-o", _out2])
        check("existing provenance sidecar alone triggers the prompt",
              _seen == [1] and _rc == 1)
finally:
    _builtins.input, sys.stdin = _orig_input, _orig_stdin
    E.load_backends, E.enhance_prompt_list = _orig_lb, _orig_epl

print(f"\n{'='*50}\n  {_passed} passed, {_failed} failed\n{'='*50}")
sys.exit(1 if _failed else 0)
