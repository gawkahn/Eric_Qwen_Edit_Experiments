"""comfyless prompt-enhancement subsystem (ADR-026).

Backend ⟂ recipe decoupled. Two backends:

  - ``hunyuan-reprompt`` — local Tencent HunYuanDenseV1 reprompt model. Uses
    Tencent's baked Chinese system prompt; IGNORES ``--enhance-recipe``.
  - ``openai-endpoint`` — any OpenAI-compatible ``/v1/chat/completions`` server
    (LM Studio, vLLM, Gemma, …). Uses the selected recipe's system prompt.

Core entry point: ``enhance(text, backend_name, ...) -> list[str]``. Inline
callers pass ``n=1``; the offline transform passes ``n=N`` for variations.

Design + assumptions: ``implementation_details.md`` (A1-A8) and ADR-026.
Only stdlib (``tomllib``, ``urllib``) + existing transformers/torch — no new deps.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Tencent reprompt system prompt ───────────────────────────────────────────
# Verbatim from Tencent-Hunyuan/HunyuanImage-2.1 `hyimage/models/reprompt/
# reprompt.py` (sourced 2026-07-11). Instructs the model to rewrite an image
# prompt preserving subject/action/count/style/layout/relation/attribute/text
# intent, in a 总-分-总 (macro-micro-macro) structure, objective, important→
# secondary, spatial/hierarchical logic, ending with a one-sentence style summary.
_HUNYUAN_REPROMPT_SYSTEM = (
    "你是一位图像生成提示词撰写专家，请根据用户输入的提示词，改写生成新的提示词，"
    "改写后的提示词要求：1 改写后提示词包含的主体/动作/数量/风格/布局/关系/属性/文字等 "
    "必须和改写前的意图一致； 2 在宏观上遵循“总-分-总”的结构，确保信息的层次清晰；"
    "3 客观中立，避免主观臆断和情感评价；4 由主到次，始终先描述最重要的元素，再描述次要和背景元素；"
    "5 逻辑清晰，严格遵循空间逻辑或主次逻辑，使读者能在大脑中重建画面；"
    "6 结尾点题，必须用一句话总结图像的整体风格或类型。"
)

# trust_remote_code hash pin (ADR-026 §8). The reprompt tokenizer requires
# trust_remote_code=True. transformers decides WHICH file to execute from the
# `auto_map` in tokenizer_config.json — so §8's "reviewed + pinned, silent swap
# detectable" invariant requires pinning BOTH the executed .py AND the config
# that names it, and asserting the auto_map still points at the reviewed class.
# Both files reviewed 2026-07-11 (tokenization_hy.py = benign tiktoken wrapper;
# auto_map = tokenization_hy.HYTokenizer). Refuse to execute on any drift.
_REPROMPT_TOKENIZER_FILE = "tokenization_hy.py"
_REPROMPT_TOKENIZER_SHA256 = (
    "0c1fced82e7de447f956daea515486bccf2f8a4b06d3d228c6296ea53f54d3b7"
)
_REPROMPT_CONFIG_FILE = "tokenizer_config.json"
_REPROMPT_CONFIG_SHA256 = (
    "560d14d33de1d2e090913620b89bf8377f0f791bd0656f793be6adcf346eee7a"
)
# The only auto_map target we have reviewed. Module component must be the pinned
# tokenizer file's stem; class is HYTokenizer.
_REPROMPT_AUTO_MAP_TARGET = "tokenization_hy.HYTokenizer"

# Tencent reprompt runtime knobs (their reprompt.py + generation_config.json).
_REPROMPT_MAX_NEW_TOKENS = 2048
_REPROMPT_GEN = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
                     repetition_penalty=1.05)

_VALID_TYPES = ("hunyuan-reprompt", "openai-endpoint")


class EnhanceError(RuntimeError):
    """Raised on any enhancement failure. Carries the backend name so the
    caller can surface a loud, actionable message (ADR-026 — never silently
    proceed on an un-enhanced prompt without the caller deciding to)."""


# ── Backend registry ─────────────────────────────────────────────────────────
def _default_config_path() -> Optional[Path]:
    """Resolve the enhancer registry file: $COMFYLESS_ENHANCERS → ./enhancers.toml
    → ~/.config/comfyless/enhancers.toml. Returns the first that exists."""
    env = os.environ.get("COMFYLESS_ENHANCERS")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.append(Path.cwd() / "enhancers.toml")
    candidates.append(Path.home() / ".config" / "comfyless" / "enhancers.toml")
    for c in candidates:
        if c.is_file():
            return c
    return None


def load_backends(path: Optional[str] = None) -> Dict[str, dict]:
    """Load + validate the backend registry TOML → {name: cfg}.

    Each entry must have a ``type`` in _VALID_TYPES. Fail-closed on a malformed
    file or an unknown type (a typo shouldn't silently yield "no such backend").
    """
    p = Path(path) if path else _default_config_path()
    if p is None:
        raise EnhanceError(
            "no enhancer registry found (set $COMFYLESS_ENHANCERS, or create "
            "./enhancers.toml — see enhancers.example.toml)"
        )
    if not p.is_file():
        raise EnhanceError(f"enhancer registry not found: {p}")
    try:
        with open(p, "rb") as f:
            data = tomllib.load(f)
    except (tomllib.TOMLDecodeError, OSError) as e:
        raise EnhanceError(f"malformed enhancer registry {p}: {e}") from e
    backends: Dict[str, dict] = {}
    for name, cfg in data.items():
        if not isinstance(cfg, dict):
            continue  # skip top-level scalars, if any
        t = cfg.get("type")
        if t not in _VALID_TYPES:
            raise EnhanceError(
                f"enhancer {name!r}: type must be one of {_VALID_TYPES}, got {t!r}"
            )
        backends[name] = dict(cfg)
    if not backends:
        raise EnhanceError(f"enhancer registry {p} defines no backends")
    return backends


# ── Recipes ──────────────────────────────────────────────────────────────────
_RECIPES_DIR = Path(__file__).resolve().parent / "recipes"


def default_recipe_name(family: Optional[str]) -> str:
    """Family's default recipe name: ``<family>-generic`` (e.g.
    ``qwen-image-generic``). Falls back to ``generic`` when family is unknown."""
    if not family or family == "unknown":
        return "generic"
    return f"{family}-generic"


def load_recipe(name: str, recipes_dir: Optional[str] = None) -> dict:
    """Load a recipe TOML (``{system_prompt, target, temperature}``) by name.

    Falls back to the family-agnostic ``generic`` recipe if the named one is
    absent, and errors only if neither exists (so a family without a bespoke
    recipe still enhances)."""
    d = Path(recipes_dir) if recipes_dir else _RECIPES_DIR
    candidate = d / f"{name}.toml"
    if not candidate.is_file() and name != "generic":
        fallback = d / "generic.toml"
        if fallback.is_file():
            candidate = fallback
    if not candidate.is_file():
        raise EnhanceError(f"recipe {name!r} not found in {d}")
    try:
        with open(candidate, "rb") as f:
            r = tomllib.load(f)
    except (tomllib.TOMLDecodeError, OSError) as e:
        raise EnhanceError(f"malformed recipe {candidate}: {e}") from e
    if not r.get("system_prompt"):
        raise EnhanceError(f"recipe {candidate} missing 'system_prompt'")
    r.setdefault("temperature", 0.8)
    try:
        r["temperature"] = float(r["temperature"])
    except (TypeError, ValueError):
        raise EnhanceError(
            f"recipe {candidate}: temperature must be numeric, got "
            f"{r['temperature']!r}"
        )
    r.setdefault("target", "")
    return r


# ── Backend: hunyuan-reprompt (local Tencent model) ──────────────────────────
# Module-level cache: the reprompt model is ~14 GB; never reload per call.
_reprompt_cache: Dict[str, Any] = {}


def _verify_reprompt_tokenizer(model_dir: str) -> None:
    """Refuse to load unless BOTH the vendored tokenizer file AND the config
    that selects it (via ``auto_map``) match the reviewed snapshots, and the
    auto_map still names only the reviewed class (ADR-026 §8).

    This is what makes trust_remote_code=True safe: transformers executes
    whatever ``auto_map`` in tokenizer_config.json points at, so pinning the
    .py alone is insufficient — a silently-swapped config could redirect
    execution to un-reviewed code while the pinned .py stays byte-identical.
    We pin both files and additionally assert the auto_map target."""
    md = Path(model_dir)
    for fname, pin in ((_REPROMPT_TOKENIZER_FILE, _REPROMPT_TOKENIZER_SHA256),
                       (_REPROMPT_CONFIG_FILE, _REPROMPT_CONFIG_SHA256)):
        f = md / fname
        if not f.is_file():
            raise EnhanceError(
                f"reprompt {f} missing — cannot verify trust_remote_code pin "
                f"(ADR-026 §8)"
            )
        got = hashlib.sha256(f.read_bytes()).hexdigest()
        if got != pin:
            raise EnhanceError(
                f"reprompt {f} sha256 {got} does not match the reviewed pin "
                f"{pin} — refusing trust_remote_code execution (ADR-026 §8). If "
                f"this change is intentional, re-review the file and update the pin."
            )
    # Defense in depth: assert auto_map still names ONLY the reviewed class
    # (belt-and-suspenders over the config hash — makes the intent explicit and
    # gives a targeted error if a future config bump adds another target).
    try:
        cfg = json.loads((md / _REPROMPT_CONFIG_FILE).read_text(encoding="utf-8"))
        am = cfg.get("auto_map", {}).get("AutoTokenizer")
    except (OSError, json.JSONDecodeError, AttributeError) as e:
        raise EnhanceError(f"reprompt tokenizer_config.json unreadable: {e}") from e
    targets = am if isinstance(am, list) else [am]
    for t in targets:
        if t and t != _REPROMPT_AUTO_MAP_TARGET:
            raise EnhanceError(
                f"reprompt auto_map targets {t!r}, not the reviewed "
                f"{_REPROMPT_AUTO_MAP_TARGET!r} — refusing trust_remote_code "
                f"(ADR-026 §8)"
            )


def _load_reprompt(model_dir: str, device: str, precision: str, quant: str = "none"):
    """Load (model, tokenizer), cached by (model_dir, device, precision, quant)."""
    import torch  # local import — heavy dep, only when this backend runs
    from transformers import AutoModelForCausalLM, AutoTokenizer

    key = f"{model_dir}|{device}|{precision}|{quant}"
    if key in _reprompt_cache:
        return _reprompt_cache[key]

    _verify_reprompt_tokenizer(model_dir)  # HARD gate before TRC execution

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}.get(precision, torch.bfloat16)
    # Model: NATIVE (transformers ≥5.5 supports hunyuan_v1_dense) — no TRC.
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=dtype, local_files_only=True,
        trust_remote_code=False,
    )
    if quant and quant != "none":
        if quant != "fp8":
            raise EnhanceError(
                f"hunyuan-reprompt quant {quant!r} unsupported (only 'fp8')"
            )
        # Weight-only fp8 (safe for a causal LM; dynamic-activation fp8 can
        # degrade LLM output). Quantize on CPU BEFORE .to(device) so the fp8
        # weights (~7 GB, not 14) land on the GPU. Same torchao path as the
        # diffusion side, weight-only recipe.
        try:
            from torchao.quantization import quantize_, Float8WeightOnlyConfig
            quantize_(model, Float8WeightOnlyConfig())
        except Exception as e:
            # Warn-and-skip on APPLICATION failure (parity with the refiner's
            # quantize_module + feedback_warn_dont_block): a bf16 reprompt is
            # still full-quality enhancement, just more VRAM — don't abort. The
            # invalid-value check above (quant != "fp8") stays a hard error.
            print(f"[comfyless] WARNING: hunyuan-reprompt fp8 quantization "
                  f"failed ({e}) — reprompt model left in {precision}",
                  file=sys.stderr)
    model = model.to(device).eval()
    # Tokenizer: requires TRC (custom HYTokenizer via auto_map); gated by the
    # hash check above.
    tok = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=True,
    )
    _reprompt_cache[key] = (model, tok)
    return model, tok


def _clean_output(text: str) -> str:
    """Normalize a model's raw output into a bare prompt string.

    - Drops any ``<think>…</think>`` channel (defensive; the reprompt model with
      enable_thinking=False pre-closes an empty one, and abliterated OpenAI
      endpoints reason in plain content — but hardens against markup leakage).
    - Extracts the inner text of an ``<answer>…</answer>`` wrapper — the Tencent
      reprompt model wraps its rewrite in one. An unclosed ``<answer>`` (output
      truncated at max_new_tokens) still yields everything after the open tag.
    """
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # tolerate an unclosed <think> (generation truncated mid-reasoning) — drop
    # the tail so raw reasoning never leaks into the prompt (mirrors the
    # unclosed-<answer> tolerance below)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if m:
        text = m.group(1)
    else:
        # tolerate an unclosed <answer> (truncated generation)
        m2 = re.search(r"<answer>(.*)", text, flags=re.DOTALL)
        if m2:
            text = m2.group(1)
    return text.strip()


def enhance_hunyuan_reprompt(text: str, cfg: dict, n: int,
                             device_override: Optional[str] = None) -> List[str]:
    """Enhance via the local Tencent reprompt model. `n` variations via sampling.

    `device_override` (the generation `--device` for the inline path) wins over
    the backend cfg `device`, so the reprompt model co-locates with the run's GPU
    instead of a hardcoded one — required to run independent gens on two GPUs."""
    import torch
    model_dir = cfg.get("model")
    if not model_dir:
        raise EnhanceError("hunyuan-reprompt backend missing 'model' path")
    device = device_override or cfg.get("device", "cuda")
    precision = cfg.get("precision", "bf16")
    quant = cfg.get("quant", "none")  # e.g. "fp8" → weight-only fp8 (~14→7 GB)
    # Sampling knobs are tunable from the backend cfg (enhancers.toml) — raise
    # `temperature`/`top_p` for more diverse --variations. Defaults are
    # Tencent's. do_sample stays on so variations actually differ.
    gen_kwargs = dict(_REPROMPT_GEN)
    for _k in ("temperature", "top_p", "top_k", "repetition_penalty"):
        if _k in cfg:
            gen_kwargs[_k] = cfg[_k]
    try:
        model, tok = _load_reprompt(model_dir, device, precision, quant)
    except EnhanceError:
        raise
    except Exception as e:
        raise EnhanceError(f"hunyuan-reprompt load failed: {e}") from e

    messages = [
        {"role": "system", "content": _HUNYUAN_REPROMPT_SYSTEM},
        {"role": "user", "content": text},
    ]
    try:
        prompt_str = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        enc = tok(prompt_str, return_tensors="pt").to(device)
    except Exception as e:
        raise EnhanceError(f"hunyuan-reprompt chat-template failed: {e}") from e

    in_len = enc["input_ids"].shape[1]
    # Throughput: generate all `n` variations in ONE batched forward pass via
    # num_return_sequences (do_sample=True in gen_kwargs makes them differ)
    # instead of n sequential calls. VRAM scales ~n× the KV cache during decode
    # — fine for typical --variations on a large card; drop --variations if a
    # small card OOMs.
    try:
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=_REPROMPT_MAX_NEW_TOKENS,
                num_return_sequences=max(1, n), **gen_kwargs,
            )
    except Exception as e:
        raise EnhanceError(f"hunyuan-reprompt generation failed: {e}") from e
    out: List[str] = []
    for row in gen:  # gen shape [n, total_len]
        cleaned = _clean_output(tok.decode(row[in_len:], skip_special_tokens=True))
        if not cleaned:
            # fail loud rather than degrade to an empty prompt (ADR-026 —
            # never silently proceed on an un-enhanced prompt)
            raise EnhanceError("hunyuan-reprompt returned an empty enhancement")
        out.append(cleaned)
    return out


def free_reprompt_cache() -> None:
    """Drop any cached reprompt model(s) and free GPU memory.

    The ~14 GB reprompt model loaded for inline ``--enhance-prompt hunyuan`` is
    co-resident with the diffusion pipeline on the same GPU during a run (a VRAM
    footgun on small cards — see implementation_details.md A10). Callers that
    know enhancement is finished can reclaim it. For large batches the offline
    ``python -m comfyless.enhance`` transform is preferred: enhance all prompts,
    then generate separately, so the two models are never co-resident."""
    if not _reprompt_cache:
        return
    _reprompt_cache.clear()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


# ── Backend: openai-endpoint (OpenAI-compatible HTTP) ────────────────────────
def _resolve_endpoint_model(url: str, key: str, requested: str) -> str:
    """If no model configured, GET {url}/models and use the first served id."""
    if requested:
        return requested
    req = urllib.request.Request(url.rstrip("/") + "/models", method="GET")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise EnhanceError(
            f"openai-endpoint: no 'model' configured and GET {url}/models "
            f"failed: {e!r}"
        ) from e
    ids = [m["id"] for m in data.get("data", []) if "id" in m]
    if ids:
        return ids[0]
    raise EnhanceError(
        f"openai-endpoint: no 'model' configured and {url}/models returned "
        f"no models"
    )


def _post_chat(endpoint: str, payload: dict, key: str) -> List[str]:
    """POST one chat/completions request; return the message content of every
    returned choice (one for a plain request, N when the payload set `n`)."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(endpoint, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise EnhanceError(
            f"openai-endpoint HTTP {e.code} from {endpoint}: {detail}"
        ) from e
    except (urllib.error.URLError, OSError) as e:
        raise EnhanceError(f"openai-endpoint cannot reach {endpoint}: {e}") from e
    try:
        return [c["message"]["content"] for c in data["choices"]]
    except (KeyError, IndexError, TypeError) as e:
        raise EnhanceError(
            f"openai-endpoint response missing choices[].message.content: "
            f"{str(data)[:200]}"
        ) from e


def enhance_openai_endpoint(text: str, cfg: dict, recipe: dict, n: int) -> List[str]:
    """Enhance via an OpenAI-compatible chat endpoint using the recipe system prompt.

    `n` variations: by default `n` independent requests, each with a distinct
    seed (so a deterministic/caching server still returns different text). Set
    `batch_variations = true` on the backend to instead request all `n` in ONE
    call via the OpenAI `n` param — a big throughput win on a server that honors
    `n` (e.g. vLLM); a server that ignores it returns one choice and we error
    clearly rather than silently under-deliver."""
    url = cfg.get("url")
    if not url:
        raise EnhanceError("openai-endpoint backend missing 'url'")
    key = ""
    key_env = cfg.get("key_env")
    if key_env:
        key = os.environ.get(key_env, "")
    model = _resolve_endpoint_model(url, key, cfg.get("model", ""))
    # Cache the resolved id back so a multi-prompt offline batch (which calls
    # this per source prompt with the same cfg dict) resolves /models once.
    cfg["model"] = model
    temperature = float(recipe.get("temperature", 0.8))
    top_p = recipe.get("top_p")
    system_prompt = recipe["system_prompt"]
    endpoint = url.rstrip("/") + "/chat/completions"
    n = max(1, n)

    def _base_payload() -> dict:
        p = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "temperature": temperature,
            "stream": False,
        }
        if top_p is not None:
            p["top_p"] = float(top_p)
        return p

    def _finalize(contents: List[str]) -> List[str]:
        cleaned = []
        for c in contents:
            v = _clean_output(c or "")
            if not v:
                raise EnhanceError("openai-endpoint returned an empty enhancement")
            cleaned.append(v)
        return cleaned

    # Batch path: one request with n=N. Opt-in per backend (needs server `n`).
    if n > 1 and cfg.get("batch_variations"):
        payload = _base_payload()
        payload["n"] = n
        contents = _post_chat(endpoint, payload, key)
        if len(contents) < n:
            raise EnhanceError(
                f"batch_variations: server returned {len(contents)} of {n} "
                f"choices — this endpoint may not support the OpenAI 'n' param; "
                f"remove batch_variations from the backend config to use "
                f"per-request variations"
            )
        return _finalize(contents[:n])

    # Default path: n independent requests, distinct seed each so a
    # deterministic/prompt-caching server still returns different text.
    out: List[str] = []
    for i in range(n):
        payload = _base_payload()
        if n > 1:
            payload["seed"] = i
        out.extend(_finalize(_post_chat(endpoint, payload, key)[:1]))
    return out


# ── Dispatch ─────────────────────────────────────────────────────────────────
def enhance(
    text: str,
    backend_name: str,
    *,
    backends: Dict[str, dict],
    recipe_name: Optional[str] = None,
    recipes_dir: Optional[str] = None,
    family: Optional[str] = None,
    n: int = 1,
    device: Optional[str] = None,
) -> List[str]:
    """Enhance ``text`` via the named backend, returning ``n`` variants.

    - ``hunyuan-reprompt`` ignores recipe (baked Tencent system prompt). ``device``
      (the run's generation GPU on the inline path) overrides its cfg device so the
      local model co-locates with the run. openai-endpoint ignores ``device`` — it
      is an HTTP client to a separately-hosted server.
    - ``openai-endpoint`` selects the recipe: explicit ``recipe_name`` →
      family default (``<family>-generic``) → ``generic``.
    Raises EnhanceError (with the backend name) on any failure.
    """
    if backend_name not in backends:
        raise EnhanceError(
            f"unknown enhance backend {backend_name!r}; known: "
            f"{sorted(backends)}"
        )
    cfg = backends[backend_name]
    t = cfg.get("type")
    if t == "hunyuan-reprompt":
        return enhance_hunyuan_reprompt(text, cfg, n, device_override=device)
    if t == "openai-endpoint":
        rn = recipe_name or default_recipe_name(family)
        recipe = load_recipe(rn, recipes_dir)
        return enhance_openai_endpoint(text, cfg, recipe, n)
    raise EnhanceError(f"backend {backend_name!r}: unsupported type {t!r}")


# ── Offline list→list transform (ADR-026 §6) ─────────────────────────────────
def enhance_prompt_list(
    prompts: List[str],
    backend_name: str,
    *,
    backends: Dict[str, dict],
    recipe_name: Optional[str] = None,
    recipes_dir: Optional[str] = None,
    family: Optional[str] = None,
    variations: int = 1,
    concurrency: int = 1,
    device: Optional[str] = None,
) -> tuple:
    """Enhance a flat list of prompts → (flat enhanced list, provenance list).

    Output length is len(prompts) × variations, in source-major then
    variation-minor order, so the result is directly consumable by one
    ``--iterate prompt`` run. Provenance[i] = {source_prompt, source_index,
    variation_index} for enhanced[i].

    ``concurrency`` prompts are enhanced in parallel via a thread pool — a real
    throughput win for the openai-endpoint backend (HTTP is I/O-bound, the GIL is
    released during the request, and the server's continuous batching absorbs the
    concurrent load). Push it until the server OOMs. For the local hunyuan backend
    it just serializes on one GPU and adds memory pressure — keep concurrency=1.
    Output order is preserved regardless of completion order."""
    n = len(prompts)
    results: List[Optional[List[str]]] = [None] * n

    def _one(si: int) -> None:
        results[si] = enhance(
            prompts[si], backend_name, backends=backends, recipe_name=recipe_name,
            recipes_dir=recipes_dir, family=family, n=variations, device=device,
        )

    if concurrency <= 1:
        for si in range(n):
            _one(si)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [ex.submit(_one, si) for si in range(n)]
            for f in as_completed(futs):
                f.result()  # propagate the first EnhanceError (cancels the rest on exit)

    enhanced: List[str] = []
    provenance: List[dict] = []
    for si, variants in enumerate(results):
        for vi, v in enumerate(variants or ()):
            enhanced.append(v)
            provenance.append({
                "source_prompt": prompts[si], "source_index": si,
                "variation_index": vi,
            })
    return enhanced, provenance


def _cli(argv: Optional[List[str]] = None) -> int:
    """`python -m comfyless.enhance in.json --backend B [-o out.json] ...` —
    read a JSON list of prompts, write an iterate-ready JSON list of enhanced
    prompts (+ optional provenance sidecar)."""
    import argparse
    p = argparse.ArgumentParser(
        prog="comfyless.enhance",
        description="Offline prompt-list enhancer (ADR-026). Input and output "
                    "are flat JSON string lists, directly --iterate prompt-able.",
    )
    p.add_argument("input", help="input JSON file: a flat list of prompt strings")
    p.add_argument("--backend", required=True, help="enhancer backend name (from the registry)")
    p.add_argument("--recipe", default=None, help="recipe name (openai-endpoint only; default: family/generic)")
    p.add_argument("--family", default=None, help="target model family for default recipe selection")
    p.add_argument("--variations", type=int, default=1, help="variants per input prompt (default 1)")
    p.add_argument("--concurrency", "-j", type=int, default=1, metavar="N",
                   help="enhance N prompts in parallel (default 1). Throughput lever "
                        "for openai-endpoint backends — push it until the server OOMs. "
                        "Keep at 1 for the local hunyuan backend (single GPU).")
    p.add_argument("--device", default=None,
                   help="GPU for a local (hunyuan) backend, overriding its config "
                        "device (ignored by openai-endpoint).")
    p.add_argument("-o", "--output", default=None, help="output JSON file (default: <input>.enhanced.json)")
    p.add_argument("--config", default=None, help="enhancer registry path (default: registry search)")
    p.add_argument("--no-provenance", action="store_true", help="skip the .provenance.json sidecar")
    p.add_argument("--recipes-dir", default=None, help="override recipes directory")
    args = p.parse_args(argv)

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            prompts = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"error: cannot read input {args.input!r}: {e}", file=sys.stderr)
        return 2
    if not isinstance(prompts, list) or not all(isinstance(x, str) for x in prompts):
        print(f"error: {args.input!r} must be a JSON list of strings", file=sys.stderr)
        return 2
    if not prompts:
        print(f"error: {args.input!r} is an empty list", file=sys.stderr)
        return 2
    if args.variations < 1:
        print("error: --variations must be >= 1", file=sys.stderr)
        return 2
    if args.concurrency < 1:
        print("error: --concurrency must be >= 1", file=sys.stderr)
        return 2

    try:
        backends = load_backends(args.config)
        enhanced, provenance = enhance_prompt_list(
            prompts, args.backend, backends=backends, recipe_name=args.recipe,
            recipes_dir=args.recipes_dir, family=args.family,
            variations=args.variations, concurrency=args.concurrency,
            device=args.device,
        )
    except EnhanceError as e:
        print(f"error [{args.backend}]: {e}", file=sys.stderr)
        return 1

    out_path = args.output or (str(Path(args.input).with_suffix("")) + ".enhanced.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(enhanced, f, ensure_ascii=False, indent=2)
    msg = f"wrote {len(enhanced)} prompts ({len(prompts)}×{args.variations}) → {out_path}"
    if not args.no_provenance:
        prov_path = str(Path(out_path).with_suffix("")) + ".provenance.json"
        with open(prov_path, "w", encoding="utf-8") as f:
            json.dump(provenance, f, ensure_ascii=False, indent=2)
        msg += f"  (+ {prov_path})"
    print(msg, file=sys.stderr)
    return 0


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(_cli())
