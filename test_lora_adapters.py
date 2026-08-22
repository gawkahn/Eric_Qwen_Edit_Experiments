#!/usr/bin/env python3
"""ADR-046 (ADR-045 slice 3c): comfyless.core.lora_adapters == the original.

Runs the node-pack module (``nodes/eric_qwen_edit_lora.py``) and the
comfyless rewrite (``comfyless/core/lora_adapters.py``) SIDE BY SIDE on the
same inputs in the same process and compares outputs bitwise. No goldens:
while the original lives in the repo the live differential is stronger.

Layer A -- the real corpus, key space only.  Every readable LoRA header on
          this machine (catalog DB) through the pure key functions. Headers
          only, so it runs in seconds and covers every key convention the
          corpus contains. Prints SKIP (not PASS) when the catalog is absent.
Layer B -- synthetic models, full behaviour, CPU.  Tiny transformers
          (bf16 and fp8-resident), synthetic LoRA / LoKR / LoHa dicts with and
          without alpha, each key naming, shape mismatches, absent modules,
          PEFT-wrapped bases -- driven through both implementations on two
          copies of the same seeded model; every parameter, buffer, registry
          entry, backup dict, ledger and return value compared; then unloaded
          and compared again.
Guards  -- the DMR source guard re-pointed at the new driver; the Grant-
          authored functions the ADR says "move verbatim" are proven
          character-identical; the new module imports nothing from ComfyUI
          or nodes/.

Run:  ./.venv/bin/python3 test_lora_adapters.py    (expect 0 failures)
"""
from __future__ import annotations

import ast
import contextlib
import copy
import inspect
import io
import json
import os
import sqlite3
import struct
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import comfyless  # noqa: F401  -- installs the folder_paths shim nodes/ needs

import torch
import torch.nn as nn
from safetensors.torch import save_file

import nodes.eric_qwen_edit_lora as orig
import comfyless.core.lora_adapters as new
from comfyless.core import eric_diffusion_fp8_ops as fp8ops

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}"[:600])


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ═══════════════════════════════════════════════════════════════════════
#  Layer A — the real corpus, key space
# ═══════════════════════════════════════════════════════════════════════
print("── A. real corpus: pure key functions ───────────────────────────")

CATALOG = Path("~/.local/share/comfyless/catalog.sqlite").expanduser()


def _header_keys(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return [k for k in json.loads(f.read(n)) if k != "__metadata__"]


if not CATALOG.exists():
    print("  SKIP  catalog DB absent -- corpus layer not run on this machine")
else:
    db = sqlite3.connect(str(CATALOG))
    rows = db.execute("select abs_path from entries where kind='lora' and excluded=0").fetchall()
    n_files = keys_total = 0
    path_mismatch = type_mismatch = norm_mismatch = rename_mismatch = name_mismatch = 0
    type_counts: dict = {}
    for (p,) in rows:
        if not p.endswith(".safetensors") or not os.path.exists(p):
            continue
        try:
            keys = _header_keys(p)
        except Exception:
            continue
        if not keys:
            continue
        n_files += 1
        keys_total += len(keys)
        sd = {k: None for k in keys}
        for k in keys:
            if orig._adapter_module_path(k) != new.adapter_module_path(k):
                path_mismatch += 1
        t_o, t_n = orig._detect_adapter_type(sd), new.detect_adapter_type(sd)
        type_counts[t_o] = type_counts.get(t_o, 0) + 1
        type_mismatch += t_o != t_n
        with quiet():
            n_o, n_n = orig._normalize_keys(sd), new.normalize_keys(sd)
            r_o, r_n = orig._rename_lora_down_up(sd), new.rename_lora_down_up(sd)
        norm_mismatch += list(n_o) != list(n_n)
        rename_mismatch += list(r_o) != list(r_n)
        base = os.path.basename(p)
        name_mismatch += orig._make_adapter_name(base) != new.make_adapter_name(base)
    print(f"        {n_files} files, {keys_total} keys, formats {type_counts}")
    check("A1 corpus readable (>= 100 files)", n_files >= 100, f"{n_files}")
    check("A2 adapter_module_path identical on every corpus key", path_mismatch == 0, f"{path_mismatch} keys")
    check("A3 detect_adapter_type identical on every file", type_mismatch == 0, f"{type_mismatch}")
    check("A4 normalize_keys (no model) identical incl. key order", norm_mismatch == 0, f"{norm_mismatch}")
    check("A5 rename_lora_down_up identical incl. key order", rename_mismatch == 0, f"{rename_mismatch}")
    check("A6 make_adapter_name identical on every filename", name_mismatch == 0, f"{name_mismatch}")
    check("A7 corpus exercises lora AND lokr", {"lora", "lokr"} <= set(type_counts), f"{type_counts}")

print("── A'. synthetic key edge cases ─────────────────────────────────")
EDGE_KEYS = [
    "transformer_blocks.0.attn.to_q.lora_A.weight",
    "transformer_blocks.0.attn.to_q.lora_A.default.weight",
    "transformer_blocks.0.attn.to_q.lora_B.default.weight",
    "transformer_blocks.0.attn.to_q.lora_down.weight",
    "transformer_blocks.0.attn.to_q.alpha",
    "x.lokr_w1", "x.lokr_w2", "x.lokr_t2",
    "x.hada_w1_a", "x.hada_w2_b",
    "x.diff", "x.diff_b",
    "x.alpha.lokr_w1",            # table order: .lokr_w1 wins over .alpha
    "x.diff_b.lora_A.weight",     # .lora_A.weight wins over .diff / .diff_b
    "no_suffix_at_all",           # no dot -> the key itself
    "a.b.c.weight",               # no known suffix -> drop last component
    "",                           # empty key
    "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight",
    "diffusion_model.layers.3.attention.qkv.lora_A.weight",
]
check("A'1 adapter_module_path identical on edge keys",
      all(orig._adapter_module_path(k) == new.adapter_module_path(k) for k in EDGE_KEYS),
      str([(k, orig._adapter_module_path(k), new.adapter_module_path(k)) for k in EDGE_KEYS
           if orig._adapter_module_path(k) != new.adapter_module_path(k)]))
EDGE_NAMES = ["Style v1.1.safetensors", "x.SafeTensors", "x.pt", "x.pth", "x.pt.pth",
              "x.bin", "a.b.c", "plain", "spaces in name.v2.BIN", "", ".safetensors"]
check("A'2 make_adapter_name identical on edge names",
      all(orig._make_adapter_name(n) == new.make_adapter_name(n) for n in EDGE_NAMES),
      str([(n, orig._make_adapter_name(n), new.make_adapter_name(n)) for n in EDGE_NAMES
           if orig._make_adapter_name(n) != new.make_adapter_name(n)]))
for label, sd in [("empty", {}), ("lokr+lora", {"a.lokr_w1": 0, "b.lora_A.weight": 0}),
                  ("loha+lora", {"a.hada_w1_a": 0, "b.lora_B.weight": 0}),
                  ("downup", {"a.lora_down.weight": 0}), ("diff-only", {"a.diff": 0}),
                  ("unknown", {"a.weight": 0})]:
    check(f"A'3 detect_adapter_type({label}) identical",
          orig._detect_adapter_type(sd) == new.detect_adapter_type(sd))


# ═══════════════════════════════════════════════════════════════════════
#  Layer B — synthetic models, full behaviour
# ═══════════════════════════════════════════════════════════════════════
print("── B. synthetic models: side-by-side state comparison ───────────")

D = 16


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(D, D, bias=False)
        self.to_k = nn.Linear(D, D, bias=False)
        self.to_v = nn.Linear(D, D, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(D, D)])


class FF(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.ModuleList([nn.Linear(D, 2 * D), nn.Linear(2 * D, D)])


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attn()
        self.ff = FF()


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([Block(), Block()])
        self.proj_out = nn.Linear(D, D)
        self.register_buffer("pos_table", torch.zeros(4))   # non-.weight buffer (must be ignored)


def fresh_pair(seed=0, fp8=False):
    torch.manual_seed(seed)
    m = Tiny()
    if fp8:
        w = torch.randn(D, D).to(torch.float8_e4m3fn)
        m.transformer_blocks[0].attn.to_q = fp8ops.ScaledFp8Linear(
            w, torch.tensor(0.02), None, None)
    return copy.deepcopy(m), copy.deepcopy(m)


class FakePipe:
    """Just enough pipeline for the orchestrators.

    ``pipeline_error`` -- what load_lora_weights raises (None => records the
    call and returns, like diffusers on a dict it cannot map; then the
    registration check decides).
    """
    def __init__(self, transformer, pipeline_error=ValueError("No modules were targeted"),
                 registers=False):
        self.transformer = transformer
        self.pipeline_error = pipeline_error
        self.registers = registers
        self.load_calls = []
        self.deleted = []

    def load_lora_weights(self, sd, adapter_name=None):
        self.load_calls.append((type(sd).__name__, adapter_name))
        if self.pipeline_error is not None:
            raise self.pipeline_error

    def get_list_adapters(self):
        return {"transformer": [n for _, n in self.load_calls]} if self.registers else {}

    def delete_adapters(self, names):
        self.deleted.append(names)
        cfg = getattr(self.transformer, "peft_config", {})
        for n in ([names] if isinstance(names, str) else names):
            cfg.pop(n, None)


def snapshot(m):
    """Everything the loaders can touch on a transformer."""
    snap = {"params": {k: v.detach().clone() for k, v in m.named_parameters()},
            "buffers": {k: v.detach().clone() for k, v in m.named_buffers()},
            "modules": [(n, type(sub).__name__) for n, sub in m.named_modules()],
            "peft_loaded": getattr(m, "_hf_peft_config_loaded", None),
            "ledger": list(getattr(m, "_eric_direct_merge_order", [])),
            "peft_config": {}, "backups": {}}
    for name, cfg in (getattr(m, "peft_config", None) or {}).items():
        snap["peft_config"][name] = cfg if isinstance(cfg, dict) else (type(cfg).__name__, repr(cfg))
    for attr in vars(m):
        if "_backup_" in attr:
            b = getattr(m, attr)
            snap["backups"][attr] = {k: (v if isinstance(v, dict) else v.clone()) for k, v in b.items()}
    return snap


def _same_tensor(a, b):
    if not torch.is_tensor(a) or not torch.is_tensor(b):
        return a == b
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)


def diff(s1, s2):
    out = []
    for sec in ("params", "buffers"):
        if set(s1[sec]) != set(s2[sec]):
            out.append(f"{sec} names differ: {set(s1[sec]) ^ set(s2[sec])}")
            continue
        for k in s1[sec]:
            if not _same_tensor(s1[sec][k], s2[sec][k]):
                out.append(f"{sec}[{k}] differ")
    for sec in ("modules", "peft_loaded", "ledger", "peft_config"):
        if s1[sec] != s2[sec]:
            out.append(f"{sec}: {s1[sec]} != {s2[sec]}")
    if set(s1["backups"]) != set(s2["backups"]):
        out.append(f"backup attrs differ: {set(s1['backups'])} vs {set(s2['backups'])}")
    else:
        for attr in s1["backups"]:
            b1, b2 = s1["backups"][attr], s2["backups"][attr]
            if set(b1) != set(b2):
                out.append(f"{attr} keys differ")
            else:
                for k in b1:
                    v1, v2 = b1[k], b2[k]
                    if isinstance(v1, dict) or isinstance(v2, dict):
                        if not (isinstance(v1, dict) and isinstance(v2, dict)
                                and set(v1) == set(v2)
                                and all(_same_tensor(v1[x], v2[x]) for x in v1)):
                            out.append(f"{attr}[{k}] quant backup differs")
                    elif not _same_tensor(v1, v2):
                        out.append(f"{attr}[{k}] differs")
    return out


def run_pair(label, make_sd, call_orig, call_new, *, seed=0, fp8=False,
             expect=None, wrap_prior=False, pipe_kwargs=None):
    """Run both implementations on identical fresh models; compare everything."""
    m_o, m_n = fresh_pair(seed, fp8)
    if wrap_prior:
        from peft import LoraConfig, inject_adapter_in_model
        for m in (m_o, m_n):
            torch.manual_seed(99)
            inject_adapter_in_model(LoraConfig(r=4, lora_alpha=4, target_modules=["to_k"]), m, "prior")
    p_o, p_n = FakePipe(m_o, **(pipe_kwargs or {})), FakePipe(m_n, **(pipe_kwargs or {}))
    sd_o, sd_n = make_sd(), make_sd()
    before = snapshot(m_o)
    errs = []
    with quiet():
        torch.manual_seed(7)
        try:
            r_o = call_orig(p_o, sd_o)
        except Exception as e:  # noqa: BLE001
            r_o = ("raised", type(e).__name__)
        torch.manual_seed(7)
        try:
            r_n = call_new(p_n, sd_n)
        except Exception as e:  # noqa: BLE001
            r_n = ("raised", type(e).__name__)
    a, b = snapshot(m_o), snapshot(m_n)
    d = diff(a, b)
    check(f"{label}: return identical ({r_o!r})", r_o == r_n, f"orig={r_o!r} new={r_n!r}")
    check(f"{label}: transformer state identical", not d, "; ".join(d[:4]))
    check(f"{label}: pipeline call sequence identical", p_o.load_calls == p_n.load_calls,
          f"{p_o.load_calls} vs {p_n.load_calls}")
    if expect is not None:
        check(f"{label}: expected outcome {expect!r}", r_o == expect, f"got {r_o!r}")
    changed = bool(diff(before, a))
    return m_o, m_n, p_o, p_n, changed


def lora_sd(prefix="", rank=4, alpha=None, names=("lora_A.weight", "lora_B.weight"),
            modules=("transformer_blocks.0.attn.to_q", "transformer_blocks.1.attn.to_v"),
            dims=None, seed=11):
    torch.manual_seed(seed)
    sd = {}
    for i, mod in enumerate(modules):
        o, n = (dims or {}).get(mod, (D, D))
        sd[f"{prefix}{mod}.{names[0]}"] = torch.randn(rank, n)
        sd[f"{prefix}{mod}.{names[1]}"] = torch.randn(o, rank) * 0.1
        if alpha is not None and (alpha != "first" or i == 0):
            sd[f"{prefix}{mod}.alpha"] = torch.tensor(float(alpha if alpha != "first" else 2.0))
    return sd


def lokr_sd(prefix="", alpha=None, modules=("transformer_blocks.0.attn.to_q",), f=(4, 4), seed=12):
    torch.manual_seed(seed)
    sd = {}
    for mod in modules:
        sd[f"{prefix}{mod}.lokr_w1"] = torch.randn(*f)
        sd[f"{prefix}{mod}.lokr_w2"] = torch.randn(D // f[0], D // f[1]) * 0.1
        if alpha is not None:
            sd[f"{prefix}{mod}.alpha"] = torch.tensor(float(alpha))
    return sd


def loha_sd(prefix="", alpha=None, rank=4, modules=("transformer_blocks.0.attn.to_q",), seed=13):
    torch.manual_seed(seed)
    sd = {}
    for mod in modules:
        sd[f"{prefix}{mod}.hada_w1_a"] = torch.randn(D, rank)
        sd[f"{prefix}{mod}.hada_w1_b"] = torch.randn(rank, D)
        sd[f"{prefix}{mod}.hada_w2_a"] = torch.randn(D, rank)
        sd[f"{prefix}{mod}.hada_w2_b"] = torch.randn(rank, D) * 0.1
        if alpha is not None:
            sd[f"{prefix}{mod}.alpha"] = torch.tensor(float(alpha))
    return sd


W = 0.7
NAME = "adp"

# ── B1. direct merge, all three kinds ──────────────────────────────────
print("   B1 direct merge driver")
direct = {
    "lora": (lambda p, sd: orig._load_lora_adapter_direct(p, sd, NAME, "[t]", weight=W),
             lambda p, sd: new._merge_direct(p, sd, NAME, "[t]", W, new._LORA)),
    "lokr": (lambda p, sd: orig._load_lokr_adapter_direct(p, sd, NAME, "[t]", weight=W),
             lambda p, sd: new._merge_direct(p, sd, NAME, "[t]", W, new._LOKR)),
    "loha": (lambda p, sd: orig._load_loha_adapter_direct(p, sd, NAME, "[t]", weight=W),
             lambda p, sd: new._merge_direct(p, sd, NAME, "[t]", W, new._LOHA)),
}
cases = [
    ("B1.1 lora no alpha", "lora", lambda: lora_sd(), True),
    ("B1.2 lora alpha!=rank", "lora", lambda: lora_sd(alpha=2.0), True),
    ("B1.3 lora alpha on first module only", "lora", lambda: lora_sd(alpha="first"), True),
    ("B1.4 lora PEFT-saved .default. names", "lora",
     lambda: lora_sd(names=("lora_A.default.weight", "lora_B.default.weight")), True),
    ("B1.5 lora absent module (skipped)", "lora",
     lambda: lora_sd(modules=("transformer_blocks.0.attn.to_q", "transformer_blocks.9.attn.ghost")), True),
    ("B1.6 lora ALL modules absent -> False", "lora",
     lambda: lora_sd(modules=("ghost.a", "ghost.b")), False),
    ("B1.7 lora shape mismatch (skipped)", "lora",
     lambda: lora_sd(modules=("transformer_blocks.0.attn.to_q", "transformer_blocks.0.ff.net.0"),
                     dims={"transformer_blocks.0.ff.net.0": (D, D)}), True),
    ("B1.8 lora path ending in .weight (fallback target)", "lora",
     lambda: lora_sd(modules=("transformer_blocks.0.attn.to_q.weight",)), True),
    ("B1.9 lora with orphan alpha + incomplete module", "lora",
     lambda: {**lora_sd(), "transformer_blocks.1.attn.to_k.alpha": torch.tensor(4.0),
              "transformer_blocks.1.attn.to_k.lora_A.weight": torch.randn(4, D)}, True),
    ("B1.10 lokr no alpha", "lokr", lambda: lokr_sd(), True),
    ("B1.11 lokr alpha", "lokr", lambda: lokr_sd(alpha=8.0), True),
    ("B1.12 lokr 2x8 factors", "lokr", lambda: lokr_sd(f=(2, 8)), True),
    ("B1.13 lokr missing w2 -> False", "lokr",
     lambda: {"transformer_blocks.0.attn.to_q.lokr_w1": torch.randn(4, 4)}, False),
    ("B1.14 loha no alpha", "loha", lambda: loha_sd(), True),
    ("B1.15 loha alpha", "loha", lambda: loha_sd(alpha=1.0), True),
    ("B1.16 loha two modules", "loha",
     lambda: loha_sd(modules=("transformer_blocks.0.attn.to_q", "transformer_blocks.1.ff.net.1")), False),
]
for label, kind, mk, expect in cases:
    # B1.16: ff.net.1 is (D, 2D) while the LoHa factors give (D, D) -> reshape fails -> skipped;
    # to_q still applies, so the outcome is True. Keep the expectation honest:
    if label.startswith("B1.16"):
        expect = True
    run_pair(label, mk, direct[kind][0], direct[kind][1], expect=expect)

print("   B1 on an fp8-resident base (ADR-019 dispatcher path)")
for label, kind, mk in [("B1.17 lora -> ScaledFp8Linear target", "lora", lambda: lora_sd()),
                        ("B1.18 lokr -> ScaledFp8Linear target", "lokr", lambda: lokr_sd()),
                        ("B1.19 loha -> ScaledFp8Linear target", "loha", lambda: loha_sd())]:
    m_o, m_n, *_ = run_pair(label, mk, direct[kind][0], direct[kind][1], fp8=True, expect=True)
    check(f"{label}: fp8 weight still fp8 after merge",
          m_n.transformer_blocks[0].attn.to_q.weight.dtype == torch.float8_e4m3fn)

print("   B1 through a PEFT-wrapped base (base_layer.weight resolution)")
run_pair("B1.20 lora direct on a base a prior PEFT adapter wrapped", lambda: lora_sd(
    modules=("transformer_blocks.0.attn.to_k", "transformer_blocks.0.attn.to_q")),
    direct["lora"][0], direct["lora"][1], wrap_prior=True)

# ── B2. unload restores identically ───────────────────────────────────
print("   B2 direct merge then unload_adapters")
for label, kind, mk in [("B2.1 lora", "lora", lambda: lora_sd(alpha=2.0)),
                        ("B2.2 lokr", "lokr", lambda: lokr_sd(alpha=8.0)),
                        ("B2.3 loha", "loha", lambda: loha_sd())]:
    m_o, m_n, p_o, p_n, _ = run_pair(label + " merge", mk, direct[kind][0], direct[kind][1])
    pristine = snapshot(fresh_pair()[0])
    with quiet():
        orig.unload_adapters(p_o, [NAME], "[t]")
        new.unload_adapters(p_n, [NAME], "[t]")
    a, b = snapshot(m_o), snapshot(m_n)
    check(f"{label} unload: state identical", not diff(a, b), "; ".join(diff(a, b)[:3]))
    check(f"{label} unload: weights restored to pristine",
          all(_same_tensor(a["params"][k], pristine["params"][k]) for k in pristine["params"]))
    check(f"{label} unload: registry + backup + ledger cleared",
          NAME not in a["peft_config"] and not a["backups"] and NAME not in a["ledger"],
          f"{a['peft_config']} {list(a['backups'])} {a['ledger']}")
    check(f"{label} is_direct_merge_adapter agrees after unload",
          orig.is_direct_merge_adapter(p_o, NAME) == new.is_direct_merge_adapter(p_n, NAME) is False)

# ── B3. orchestrators through the fake pipe ───────────────────────────
print("   B3 orchestrators (pipeline fails -> PEFT -> direct)")
orch = {
    "lora": (lambda p, sd: orig._load_lora_adapter(p, sd, NAME, "[t]", weight=W),
             lambda p, sd: new._load_lora_adapter(p, sd, NAME, "[t]", weight=W)),
    "lokr": (lambda p, sd: orig._load_lokr_adapter(p, sd, NAME, "[t]", weight=W),
             lambda p, sd: new._load_lokr_adapter(p, sd, NAME, "[t]", weight=W)),
    "loha": (lambda p, sd: orig._load_loha_adapter(p, sd, NAME, "[t]", weight=W),
             lambda p, sd: new._load_loha_adapter(p, sd, NAME, "[t]", weight=W)),
}
_, _, p_o, p_n, _ = run_pair("B3.1 lora: pipeline raises -> PEFT injection", lambda: lora_sd(alpha=2.0),
                             *orch["lora"], expect=True)
check("B3.1 both tried bare then transformer-prefixed dicts",
      [c[1] for c in p_o.load_calls] == [NAME, NAME], str(p_o.load_calls))
check("B3.1 PEFT layers present (not a direct merge)",
      not orig.is_direct_merge_adapter(p_o, NAME) and not new.is_direct_merge_adapter(p_n, NAME))
run_pair("B3.2 lora: kohya lora_down/up naming + alpha (bake+rename first)",
         lambda: lora_sd(alpha=2.0, names=("lora_down.weight", "lora_up.weight")), *orch["lora"], expect=True)
run_pair("B3.3 lora: pipeline 'succeeds' but never registers -> falls through",
         lambda: lora_sd(), *orch["lora"], expect=True, pipe_kwargs={"pipeline_error": None})
run_pair("B3.4 lora: pipeline registers -> returns on first attempt",
         lambda: lora_sd(), *orch["lora"], expect=True,
         pipe_kwargs={"pipeline_error": None, "registers": True})
run_pair("B3.5 lora: nothing maps -> PEFT fails -> direct 0 -> False",
         lambda: lora_sd(modules=("ghost.x", "ghost.y")), *orch["lora"], expect=False)
run_pair("B3.6 lora: pipeline RuntimeError is swallowed by both orchestrators, load continues (pre-existing)",
         lambda: lora_sd(), *orch["lora"], pipe_kwargs={"pipeline_error": RuntimeError("CUDA out of memory")})
_, _, p_o, p_n, _ = run_pair("B3.7 lokr: PEFT vs direct (same path taken)", lambda: lokr_sd(alpha=8.0), *orch["lokr"])
check("B3.7 both landed on the same path (direct-merge flag agrees)",
      orig.is_direct_merge_adapter(p_o, NAME) == new.is_direct_merge_adapter(p_n, NAME))
# PEFT's default factoriser picks (4,4) for 16, so 2x2/8x8 factors fail in
# set_peft_model_state_dict with a size mismatch -> the except branch. NOTE what
# then happens (pre-existing, preserved): inject_adapter_in_model has ALREADY
# wrapped to_q into a LoKr layer, so the direct merge looks for `to_q.weight`,
# finds only `to_q.base_layer.weight`, applies 0, and the flatten rescue runs.
_, _, p_o, p_n, _ = run_pair("B3.7b lokr: factorisation PEFT rejects -> except branch (direct 0 -> flatten rescue)",
                             lambda: lokr_sd(f=(2, 2)), *orch["lokr"])
check("B3.7b both took the same branch (direct-merge flag agrees)",
      orig.is_direct_merge_adapter(p_o, NAME) == new.is_direct_merge_adapter(p_n, NAME))
check("B3.7b PEFT rejection reproduced: to_q is PEFT-wrapped (has base_layer) in both",
      hasattr(p_o.transformer.transformer_blocks[0].attn.to_q, "base_layer")
      and hasattr(p_n.transformer.transformer_blocks[0].attn.to_q, "base_layer"))
run_pair("B3.8 lokr: nothing maps -> direct 0 -> flatten rescue -> False",
         lambda: lokr_sd(modules=("ghost.x",)), *orch["lokr"], expect=False)
_, _, p_o, p_n, _ = run_pair("B3.9 loha: PEFT vs direct (same path taken)", lambda: loha_sd(alpha=1.0), *orch["loha"])
check("B3.9 both landed on the same path (direct-merge flag agrees)",
      orig.is_direct_merge_adapter(p_o, NAME) == new.is_direct_merge_adapter(p_n, NAME))
run_pair("B3.10 loha: nothing maps -> False", lambda: loha_sd(modules=("ghost.x",)), *orch["loha"], expect=False)

# ── B4. normalize_keys with a model ───────────────────────────────────
print("   B4 normalize_keys against a model")
model = fresh_pair()[0]
for label, sd in [
    ("B4.1 already matching", lora_sd()),
    ("B4.2 transformer. prefix", lora_sd(prefix="transformer.")),
    ("B4.3 diffusion_model. prefix", lora_sd(prefix="diffusion_model.")),
    ("B4.4 model.diffusion_model. prefix", lora_sd(prefix="model.diffusion_model.")),
    ("B4.5 model. prefix", lora_sd(prefix="model.")),
    ("B4.6 unknown prefix, unique discovery", lora_sd(prefix="weird.stuff.")),
    ("B4.7 no match at all -> unchanged", lora_sd(modules=("ghost.a", "ghost.b"))),
    ("B4.8 TE keys dropped", {**lora_sd(), "text_encoder.x.lora_A.weight": torch.zeros(1),
                              "lora_te1_x.lora_down.weight": torch.zeros(1)}),
    ("B4.9 only TE keys -> empty", {"lora_te_x.lora_down.weight": torch.zeros(1)}),
    ("B4.10 mixed: some prefixed, some bare", {**lora_sd(prefix="transformer."),
                                               **lora_sd(modules=("transformer_blocks.0.attn.to_k",))}),
]:
    with quiet():
        o, n = orig._normalize_keys(sd, model=model), new.normalize_keys(sd, model=model)
    check(label, list(o) == list(n) and all(o[k] is n[k] for k in o), f"{list(o)[:3]} vs {list(n)[:3]}")
check("B4.11 no-model mode identical (transformer. stripped)",
      list(orig._normalize_keys(lora_sd(prefix="transformer."))) == list(new.normalize_keys(lora_sd(prefix="transformer."))))

# ── B5. load_lora_with_key_fix end to end on real files ───────────────
print("   B5 load_lora_with_key_fix on temp safetensors files")
tmp = tempfile.mkdtemp(prefix="lora3c-")


def write(name, sd):
    p = os.path.join(tmp, name)
    save_file({k: v.contiguous() for k, v in sd.items()}, p)
    return p


files = {
    "plain_lora": write("plain.safetensors", lora_sd(alpha=2.0)),
    "kohya_lora": write("kohya.safetensors", lora_sd(prefix="transformer.", names=("lora_down.weight", "lora_up.weight"), alpha=2.0)),
    "lokr": write("lokr.safetensors", lokr_sd(alpha=8.0)),
    "loha": write("loha.safetensors", loha_sd()),
    "unknown": write("unknown.safetensors", {"transformer_blocks.0.attn.to_q.weight": torch.zeros(D, D)}),
    "ghost": write("ghost.safetensors", lora_sd(modules=("ghost.a",))),
}
keyfix = (lambda p, path: orig.load_lora_with_key_fix(p, path, NAME, "[t]", weight=W),
          lambda p, path: new.load_lora_with_key_fix(p, path, NAME, "[t]", weight=W))
FIXABLE = [ValueError("Target modules ['x'] not found in the base model"),
           ValueError("No modules were targeted"), RuntimeError("Error(s) in loading state_dict"),
           ValueError("lora_A shape"), ValueError("something lokr"), ValueError("HADA_ bad"),
           ValueError("PEFT backend is required"), RuntimeError("Not Implemented for this"),
           ValueError("Handling for key foo"), KeyError("transformer.x.lora_A.weight")]
for i, err in enumerate(FIXABLE):
    run_pair(f"B5.{i+1} fast path raises {type(err).__name__}({str(err)[:28]!r}) -> manual path",
             lambda: files["plain_lora"], *keyfix, pipe_kwargs={"pipeline_error": err})
run_pair("B5.11 non-fixable error re-raised by both", lambda: files["plain_lora"], *keyfix,
         pipe_kwargs={"pipeline_error": RuntimeError("CUDA out of memory")},
         expect=("raised", "RuntimeError"))
run_pair("B5.12 fast path succeeds -> True, nothing touched", lambda: files["plain_lora"], *keyfix,
         pipe_kwargs={"pipeline_error": None, "registers": True}, expect=True)
run_pair("B5.13 kohya down/up + transformer. prefix file", lambda: files["kohya_lora"], *keyfix, expect=True)
run_pair("B5.14 lokr file", lambda: files["lokr"], *keyfix)
run_pair("B5.15 loha file", lambda: files["loha"], *keyfix)
run_pair("B5.16 unknown format raises ValueError in both", lambda: files["unknown"], *keyfix,
         expect=("raised", "ValueError"))
run_pair("B5.17 nothing maps -> FAILED -> False", lambda: files["ghost"], *keyfix, expect=False)
run_pair("B5.18 min_compatibility skip identical", lambda: files["ghost"],
         lambda p, path: orig.load_lora_with_key_fix(p, path, NAME, "[t]", weight=W, min_compatibility=0.5),
         lambda p, path: new.load_lora_with_key_fix(p, path, NAME, "[t]", weight=W, min_compatibility=0.5),
         expect=False)
check("B5.19 is_direct_merge_adapter on a pipe without transformer -> False in both",
      orig.is_direct_merge_adapter(object(), NAME) is False and new.is_direct_merge_adapter(object(), NAME) is False)
check("B5.20 plan_match_model_names identical (params + buffers)",
      orig.plan_match_model_names(fresh_pair(fp8=True)[0]) == new.plan_match_model_names(fresh_pair(fp8=True)[0]))

# ── B6. the one recorded deviation: discovery is deterministic ─────────
print("   B6 prefix discovery determinism (ADR-046 #5 deviation)")
sd = lora_sd(prefix="pfxA.")
with quiet():
    outs = {tuple(new.normalize_keys(sd, model=model)) for _ in range(5)}
check("B6.1 new discovery gives one answer across repeated calls", len(outs) == 1)
check("B6.2 that answer strips the discovered prefix",
      next(iter(outs))[0].startswith("transformer_blocks."), str(next(iter(outs))[:2]))
# Multi-candidate: pfxA.* and pfxB.* each cover >30% of the paths. The original
# picks whichever set iteration order surfaces first; the new picks sorted-first.
multi = {**lora_sd(prefix="pfxA.", modules=("transformer_blocks.0.attn.to_q", "transformer_blocks.0.attn.to_k")),
         **lora_sd(prefix="pfxB.", modules=("transformer_blocks.1.attn.to_q", "transformer_blocks.1.attn.to_v"))}
with quiet():
    n_keys = list(new.normalize_keys(multi, model=model))
    o_keys = list(orig._normalize_keys(multi, model=model))
check("B6.3 multi-candidate: new strips the sorted-first prefix (pfxA.)",
      all(not k.startswith("pfxA.") for k in n_keys) and any(k.startswith("pfxB.") for k in n_keys), str(n_keys[:4]))
check("B6.3 multi-candidate: original's answer is one of the two (order-dependent, recorded not compared)",
      (all(not k.startswith("pfxA.") for k in o_keys) and any(k.startswith("pfxB.") for k in o_keys))
      or (all(not k.startswith("pfxB.") for k in o_keys) and any(k.startswith("pfxA.") for k in o_keys)), str(o_keys[:4]))


class Wide(nn.Module):
    """> 50 modules, so the original's 50-name window usually misses."""
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([Block() for _ in range(12)])   # 12 * 9 = 108 modules


torch.manual_seed(3)
wide = Wide()
wide_sd = lora_sd(prefix="weird.", modules=tuple(f"transformer_blocks.{i}.attn.to_q" for i in range(12)))
with quiet():
    n_wide = list(new.normalize_keys(wide_sd, model=wide))
    o_wide = list(orig._normalize_keys(wide_sd, model=wide))
check("B6.4 >50-module model: new strips the prefix deterministically",
      all(k.startswith("transformer_blocks.") for k in n_wide), str(n_wide[:2]))
check("B6.4 >50-module model: the EXPECTED divergence -- original may leave it unchanged (recorded)",
      all(k.startswith("transformer_blocks.") for k in o_wide) or all(k.startswith("weird.") for k in o_wide),
      str(o_wide[:2]))
print(f"        (original on the wide model: {'stripped' if o_wide[0].startswith('transformer_blocks.') else 'UNCHANGED -- divergence reproduced'})")

# ═══════════════════════════════════════════════════════════════════════
#  Guards
# ═══════════════════════════════════════════════════════════════════════
print("── G. structural guards ─────────────────────────────────────────")
src = (REPO / "comfyless" / "core" / "lora_adapters.py").read_text()
body = src.split("def _merge_direct(")[1].split("\ndef ")[0]
check("G1 _merge_direct routes every write through apply_merge_delta (DMR)", "apply_merge_delta(" in body)
check("G2 _merge_direct has no raw param.data.add_ (ADR-019 req 24 NEGATIVE)", "param.data.add_" not in body)
check("G3 _merge_direct uses merge_resolution_map (req 23)", "merge_resolution_map(" in body)
check("G4 _merge_direct refuses before the first write (req 65)",
      body.index("refuse_unmergeable_base(") < body.index("apply_merge_delta("))

for fn in ("_apply_te_lora", "_decode_kohya_keys", "_bake_lora_alpha_scales",
           "unload_adapters", "is_direct_merge_adapter", "plan_match_model_names"):
    check(f"G5 {fn} is character-identical to the node-pack original (moved verbatim)",
          inspect.getsource(getattr(orig, fn)) == inspect.getsource(getattr(new, fn)))

tree = ast.parse(src)
imported = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        imported |= {a.name for a in node.names}
    elif isinstance(node, ast.ImportFrom) and node.module:
        imported.add(node.module)
check("G6 new module imports nothing from nodes/ or ComfyUI",
      not any(m in ("folder_paths", "nodes", "comfy") or m.startswith(("nodes.", "comfy.")) for m in imported),
      str(sorted(imported)))
check("G7 new module has no Eric Hiss copyright header", "Copyright (c) 2026 Eric Hiss" not in src)
check("G8 comfyless/ no longer imports nodes.eric_qwen_edit_lora",
      not any("nodes.eric_qwen_edit_lora" in p.read_text()
              for p in (REPO / "comfyless").rglob("*.py")),
      str([str(p) for p in (REPO / "comfyless").rglob("*.py") if "nodes.eric_qwen_edit_lora" in p.read_text()]))

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
