#!/usr/bin/env python3
"""Tests for Krea-2 identity-preserving instruction edit — ADR-043.

CPU-only, no GPU, no model weights. Covers the epic's Part A proof hooks:
the `[text | source(1..N) | target]` position-id layout at BOTH n_src=1 and
n_src=2 including the frame-ORDER invariant (frame 1 = scene, frame 2 =
identity), `ref_boost` bias placement across a two-block source span and its
1.0 no-op, processor install/restore with cuDNN-pin inheritance (ADR-023
hazard H1), the D10 image processor built from a LIVE encoder's vision_config
plus the negative case that would have caught the --te1 defect, the
tokenizer/encoder token-id consistency warnings, and source-order/count
validation.

Sidecar/schema legs belong in test_params_schema.py; routing legs in
test_ref_edit.py; daemon carriage is Part C and not built.
"""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import PIL.Image

sys.path.insert(0, str(Path(__file__).parent))

from pipelines import krea2_identity_edit as kie
from diffusers.models.transformers.transformer_krea2 import Krea2Transformer2DModel


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


torch.manual_seed(0)
CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
print("\n── position ids: [text | source(1..N) | target] ───────────────")
# ---------------------------------------------------------------------------
TEXT, GH, GW = 3, 2, 2
GRID = GH * GW

one = kie.edit_position_ids(TEXT, GH, GW, 1, CPU)
check("n_src=1 length is text + 1*grid + grid",
      one.shape == (TEXT + GRID + GRID, 3), f"got {tuple(one.shape)}")
check("n_src=1 text rows sit at the origin",
      torch.equal(one[:TEXT], torch.zeros(TEXT, 3)))
check("n_src=1 source block carries frame 1",
      torch.all(one[TEXT:TEXT + GRID, 0] == 1).item())
check("n_src=1 target block carries frame 0",
      torch.all(one[TEXT + GRID:, 0] == 0).item())

two = kie.edit_position_ids(TEXT, GH, GW, 2, CPU)
check("n_src=2 length is text + 2*grid + grid",
      two.shape == (TEXT + 2 * GRID + GRID, 3), f"got {tuple(two.shape)}")
# THE order invariant: a refactor could silently swap these and every shape
# assertion would still pass. Frame 1 = scene, frame 2 = identity (ADR-043).
check("n_src=2 FIRST source block carries frame 1 (scene)",
      torch.all(two[TEXT:TEXT + GRID, 0] == 1).item())
check("n_src=2 SECOND source block carries frame 2 (identity)",
      torch.all(two[TEXT + GRID:TEXT + 2 * GRID, 0] == 2).item())
check("n_src=2 target still carries frame 0",
      torch.all(two[TEXT + 2 * GRID:, 0] == 0).item())
check("frames are strictly ordered scene<identity (not merely distinct)",
      two[TEXT, 0].item() < two[TEXT + GRID, 0].item())

# h/w coordinates repeat per block rather than running on.
h_first = two[TEXT:TEXT + GRID, 1].tolist()
h_second = two[TEXT + GRID:TEXT + 2 * GRID, 1].tolist()
check("each source block restarts its h coordinates", h_first == h_second,
      f"{h_first} vs {h_second}")
check("h coords span the grid height", sorted(set(h_first)) == list(range(GH)))
check("w coords span the grid width",
      sorted(set(two[TEXT:TEXT + GRID, 2].tolist())) == list(range(GW)))


# ---------------------------------------------------------------------------
print("\n── ref_boost bias placement ───────────────────────────────────")
# ---------------------------------------------------------------------------
T_LEN, S_LEN, G_LEN = 3, 8, 4          # S_LEN = 2 source blocks of 4
proc = kie.Krea2IdentityEditAttnProcessor(
    text_len=T_LEN, src_len=S_LEN, tgt_len=G_LEN, ref_boost=4.0)
bias = proc._bias(CPU, torch.float32)
total = T_LEN + S_LEN + G_LEN
rows0 = T_LEN + S_LEN

check("bias is a square (1,1,L,L) additive mask",
      bias.shape == (1, 1, total, total), f"got {tuple(bias.shape)}")
check("target->source logits get log(ref_boost)",
      abs(bias[0, 0, rows0, T_LEN].item() - math.log(4.0)) < 1e-6)
check("the boost covers the WHOLE two-block source span",
      torch.allclose(bias[0, 0, rows0:, T_LEN:rows0],
                     torch.full((G_LEN, S_LEN), math.log(4.0))))
check("target->text logits are untouched",
      torch.all(bias[0, 0, rows0:, :T_LEN] == 0).item())
check("target->target logits are untouched",
      torch.all(bias[0, 0, rows0:, rows0:] == 0).item())
check("source rows are untouched (only TARGET queries are boosted)",
      torch.all(bias[0, 0, T_LEN:rows0, :] == 0).item())
check("text rows are untouched",
      torch.all(bias[0, 0, :T_LEN, :] == 0).item())

# Negative: a boost below 1 must ATTENUATE, not silently clamp to zero.
weak = kie.Krea2IdentityEditAttnProcessor(
    text_len=T_LEN, src_len=S_LEN, tgt_len=G_LEN, ref_boost=0.5)
check("ref_boost < 1 produces a negative bias (attenuates the source)",
      weak._bias(CPU, torch.float32)[0, 0, rows0, T_LEN].item() < 0)

noop = kie.Krea2IdentityEditAttnProcessor(
    text_len=T_LEN, src_len=S_LEN, tgt_len=G_LEN, ref_boost=1.0)
check("ref_boost == 1.0 is an all-zero (no-op) bias",
      torch.all(noop._bias(CPU, torch.float32) == 0).item())

# The bool key-padding mask the transformer hands down must survive the merge.
bool_mask = torch.ones(1, 1, 1, total, dtype=torch.bool)
bool_mask[0, 0, 0, 1] = False           # pretend text token 1 is padding
merged = proc._merge_mask(bool_mask, torch.float32, CPU)
check("a bool key-padding mask merges into additive form",
      merged.dtype == torch.float32)
check("masked-out keys stay masked after the merge",
      merged[0, 0, rows0, 1].item() < -1e30)
check("the boost survives merging with a padding mask",
      abs(merged[0, 0, rows0, T_LEN].item() - math.log(4.0)) < 1e-6)
check("merge with no incoming mask returns the bias unchanged",
      torch.equal(proc._merge_mask(None, torch.float32, CPU), bias))


# ---------------------------------------------------------------------------
print("\n── processor install / restore (ADR-023 hazard H1) ────────────")
# ---------------------------------------------------------------------------
tiny = Krea2Transformer2DModel(
    in_channels=16,
    num_layers=2,
    attention_head_dim=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=64,
    timestep_embed_dim=16,
    text_hidden_dim=32,
    num_text_layers=2,
    text_num_attention_heads=2,
    text_num_key_value_heads=2,
    text_intermediate_size=48,
    num_layerwise_text_blocks=1,
    num_refiner_text_blocks=1,
    axes_dims_rope=(4, 2, 2),
).eval()

# Pin a backend on the stock processors, as _pin_krea_attention_backend does.
for _proc in tiny.attn_processors.values():
    _proc._attention_backend = "_native_cudnn"

before = dict(tiny.attn_processors)
origin = kie.apply_identity_processors(
    tiny, text_len=T_LEN, src_len=S_LEN, tgt_len=G_LEN, ref_boost=4.0)
after = dict(tiny.attn_processors)

installed = [n for n, p in after.items()
             if isinstance(p, kie.Krea2IdentityEditAttnProcessor)]
check("processors were installed on the main transformer blocks",
      len(installed) > 0)
check("ONLY transformer_blocks.* were replaced (hazard H2)",
      all(n.startswith("transformer_blocks.") for n in installed),
      f"stray: {[n for n in installed if not n.startswith('transformer_blocks.')]}")
check("every installed processor inherited the cuDNN pin (hazard H1)",
      all(after[n]._attention_backend == "_native_cudnn" for n in installed))
check("apply returned the ORIGINAL processors for restore",
      set(origin) == set(before))

kie.remove_identity_processors(tiny, origin)
restored = dict(tiny.attn_processors)
check("restore puts the stock processors back",
      not any(isinstance(p, kie.Krea2IdentityEditAttnProcessor)
              for p in restored.values()))
check("restore preserves the cuDNN pin",
      all(p._attention_backend == "_native_cudnn" for p in restored.values()))
# `set_attn_processor` pops from the dict it is handed — origin must survive.
kie.remove_identity_processors(tiny, origin)
check("restore is repeatable (origin dict is not consumed)",
      len(dict(tiny.attn_processors)) == len(before))


# ---------------------------------------------------------------------------
print("\n── ADR-044 commit 1: restore survives a PARTIAL apply ─────────")
# ---------------------------------------------------------------------------
# The hazard (ADR-044 security review, Finding 2). `apply_identity_processors`
# used to run OUTSIDE the try, with `origin` assigned only from its return
# value. If set_attn_processor raised part-way through, `origin` was never
# bound, the finally had nothing to restore from, and the identity processors
# stayed installed.
#
# In-process that residue dies with the CLI. Once Part C delegates, it lives in
# the DAEMON's cached pipeline: the stale processors carry frozen
# text_len/src_len/tgt_len, so the next request at a DIFFERENT resolution
# crashes loudly — but one at the SAME resolution (the --iterate sweep case)
# silently gets a wrong attention bias. Wrong output, no error.
_pa_stock = dict(tiny.attn_processors)
_pa_real_set = tiny.set_attn_processor


def _exploding_set(procs):
    """Apply the swap, THEN raise — the partial-application shape."""
    _pa_real_set(procs)
    raise RuntimeError("simulated mid-apply failure")


# The FIXED shape: capture first, apply inside the try, restore in the finally.
_pa_origin = dict(tiny.attn_processors)          # captured BEFORE any swap
try:
    tiny.set_attn_processor = _exploding_set     # type: ignore[method-assign]
    try:
        kie.apply_identity_processors(
            tiny, text_len=T_LEN, src_len=S_LEN, tgt_len=G_LEN, ref_boost=4.0)
    except RuntimeError:
        pass
    check("premise: the simulated failure DID leave residue installed",
          any(isinstance(p, kie.Krea2IdentityEditAttnProcessor)
              for p in dict(tiny.attn_processors).values()))
finally:
    tiny.set_attn_processor = _pa_real_set       # type: ignore[method-assign]
    kie.remove_identity_processors(tiny, _pa_origin)

check("a pre-captured origin restores from a PARTIAL apply",
      not any(isinstance(p, kie.Krea2IdentityEditAttnProcessor)
              for p in dict(tiny.attn_processors).values()))
check("the partial-apply restore preserves the cuDNN pin (hazard H1)",
      all(p._attention_backend == "_native_cudnn"
          for p in dict(tiny.attn_processors).values()))
check("the partial-apply restore returns the SAME processor set",
      set(dict(tiny.attn_processors)) == set(_pa_stock))

# The negative that pins WHY the capture must be separate: taking `origin` from
# the return value — the pre-ADR-044 shape — leaves nothing bound when the
# apply raises, so there is no restore path on exactly the failure needing one.
_pa_returned = None
try:
    tiny.set_attn_processor = _exploding_set     # type: ignore[method-assign]
    try:
        _pa_returned = kie.apply_identity_processors(
            tiny, text_len=T_LEN, src_len=S_LEN, tgt_len=G_LEN, ref_boost=4.0)
    except RuntimeError:
        pass
finally:
    tiny.set_attn_processor = _pa_real_set       # type: ignore[method-assign]
    kie.remove_identity_processors(tiny, _pa_origin)
check("NEGATIVE: the return value is unbound when apply raises "
      "(why __call__ must not depend on it)",
      _pa_returned is None)


# ---------------------------------------------------------------------------
print("\n── ADR-044 commit 1: __call__ install/restore STRUCTURE ────────")
# ---------------------------------------------------------------------------
# Structural, not behavioural, for the same reason as the unbound-call guard
# below: the defect is a statement's POSITION relative to a try block, and
# reproducing it behaviourally needs a real GPU generation. A refactor that
# hoists the apply back out of the try must fail a test.
import ast  # noqa: E402

_c1_src = Path(kie.__file__).read_text()
# Scope to the PIPELINE's __call__ — `Krea2IdentityEditAttnProcessor.__call__`
# is defined earlier in the module and a bare module-level search finds that one
# instead, which silently passes every check below with zero matches.
_c1_cls = next(n for n in ast.walk(ast.parse(_c1_src))
               if isinstance(n, ast.ClassDef)
               and n.name == "Krea2IdentityEditPipeline")
_c1_call = next(n for n in _c1_cls.body
                if isinstance(n, ast.FunctionDef) and n.name == "__call__")


def _c1_calls_to(node, name):
    return [c for c in ast.walk(node)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
            and c.func.id == name]


_c1_all_applies = _c1_calls_to(_c1_call, "apply_identity_processors")
check("guard premise: __call__ installs the processors exactly once",
      len(_c1_all_applies) == 1, f"found {len(_c1_all_applies)}")

# A qualifying Try: applies inside its BODY, restores in its FINALLY.
_c1_guarded = [
    t for t in ast.walk(_c1_call)
    if isinstance(t, ast.Try)
    and any(_c1_calls_to(s, "apply_identity_processors") for s in t.body)
    and any(_c1_calls_to(s, "remove_identity_processors") for s in t.finalbody)
]
check("the apply is INSIDE a try whose finally restores",
      len(_c1_guarded) == 1,
      f"{len(_c1_guarded)} qualifying try blocks")

# ...restoring from `origin` SPECIFICALLY. Without this leg, a refactor that
# leaves the (now dead) capture in place but restores from a variable bound to
# apply's return value passes every other check while reinstating the defect
# (code review 2026-08-01, advisory 1).
_c1_removes = [c for t in _c1_guarded for s in t.finalbody
               for c in _c1_calls_to(s, "remove_identity_processors")]
check("the finally restores from `origin`, not from some other binding",
      len(_c1_removes) == 1
      and len(_c1_removes[0].args) >= 2
      and isinstance(_c1_removes[0].args[1], ast.Name)
      and _c1_removes[0].args[1].id == "origin",
      f"restore args: {[ast.dump(a) for a in _c1_removes[0].args]}"
      if _c1_removes else "no restore call found")

# ...and the restore must be allowed to RAISE. This closes the one regression
# the layered residue design cannot otherwise see (code review 2026-08-01).
#
# The daemon's residue check (server._handle_generate) runs only in its EXCEPT
# branch, so it depends on a failed restore propagating out of __call__. Wrap
# this call in `try/except Exception: pass` — a plausible hardening edit, since
# an exception raised in a finally masks the in-flight original error — and a
# successful body with a FAILED restore returns a SUCCESS response with the
# identity processors still installed on the daemon's cached pipeline. Nothing
# inspects the success path, so every suite in this repo stays green while the
# invariant ADR-044's cache-key decision rests on is silently broken.
#
# `ast.walk` in the guard above would still find the call nested inside that
# new Try, which is exactly why this leg checks NESTING rather than presence.
_c1_fin_tries = [t for gt in _c1_guarded for s in gt.finalbody
                 for t in ast.walk(s) if isinstance(t, ast.Try)]
check("NEGATIVE: the restore is not wrapped in a nested try (it must raise, "
      "so the daemon's residue check can see the failure)",
      not _c1_fin_tries,
      f"{len(_c1_fin_tries)} nested try block(s) in the finally")

# ...and the capture precedes it. `origin` must come from attn_processors, not
# from the apply's return value.
_c1_origin_assigns = [
    n for n in ast.walk(_c1_call)
    if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == "origin" for t in n.targets)
    and "attn_processors" in ast.dump(n.value)
]
check("origin is captured from attn_processors, not from apply's return",
      len(_c1_origin_assigns) == 1,
      f"found {len(_c1_origin_assigns)}")
check("the capture happens BEFORE the apply",
      bool(_c1_origin_assigns)
      and _c1_origin_assigns[0].lineno < _c1_all_applies[0].lineno)
check("NEGATIVE: origin is never assigned from apply_identity_processors(...)",
      not [n for n in ast.walk(_c1_call)
           if isinstance(n, ast.Assign)
           and any(isinstance(t, ast.Name) and t.id == "origin"
                   for t in n.targets)
           and _c1_calls_to(n.value, "apply_identity_processors")])

# The ref_boost == 1.0 fast path must stay CONDITIONAL. Making the capture and
# install unconditional passes every other leg here and every behavioural check,
# yet kills the maskless path: an all-zero float attn_mask still changes SDPA
# dispatch, so the 1.0 case would stop being stock-identical. The zero-BIAS is
# pinned elsewhere; this pins never-INSTALLED (code review 2026-08-01,
# advisory 2).
check("the capture/install is gated on ref_boost (1.0 stays maskless)",
      bool(_c1_origin_assigns)
      and isinstance(_c1_origin_assigns[0].value, ast.IfExp)
      and "ref_boost" in ast.dump(_c1_origin_assigns[0].value.test))


# ---------------------------------------------------------------------------
print("\n── D10: image processor built from the LIVE encoder ───────────")
# ---------------------------------------------------------------------------
def _encoder(patch=16, temporal=2, merge=2, with_vision=True):
    vision = SimpleNamespace(patch_size=patch, temporal_patch_size=temporal,
                             spatial_merge_size=merge) if with_vision else None
    return SimpleNamespace(config=SimpleNamespace(vision_config=vision))


built = kie.build_vl_image_processor(_encoder())
check("processor takes patch_size from the encoder", built.patch_size == 16)
check("processor takes merge_size from spatial_merge_size", built.merge_size == 2)
check("processor takes temporal_patch_size from the encoder",
      built.temporal_patch_size == 2)
check("code-owned normalization constants are applied",
      list(built.image_mean) == [0.5, 0.5, 0.5]
      and list(built.image_std) == [0.5, 0.5, 0.5])

# THE regression that a checkpoint-directory implementation would not catch:
# both encoders on this box report identical geometry, so reading the wrong
# source passes by coincidence. A differing encoder must change the processor.
other = kie.build_vl_image_processor(_encoder(patch=14, merge=3, temporal=4))
check("a DIFFERENT encoder yields a DIFFERENT processor (the --te1 defect)",
      (other.patch_size, other.merge_size, other.temporal_patch_size) == (14, 3, 4),
      f"got {(other.patch_size, other.merge_size, other.temporal_patch_size)}")
check("the two processors genuinely differ",
      built.patch_size != other.patch_size)

try:
    kie.build_vl_image_processor(_encoder(with_vision=False))
    check("a text-only encoder is refused", False, "no error raised")
except ValueError as exc:
    check("a text-only encoder is refused with a named cause",
          "vision_config" in str(exc))

try:
    kie.build_vl_image_processor(
        SimpleNamespace(config=SimpleNamespace(
            vision_config=SimpleNamespace(patch_size=16))))
    check("a partial vision_config is refused", False, "no error raised")
except ValueError as exc:
    check("a partial vision_config names the missing fields",
          "temporal_patch_size" in str(exc) and "spatial_merge_size" in str(exc))


# ---------------------------------------------------------------------------
print("\n── token-id consistency: warn, never block ────────────────────")
# ---------------------------------------------------------------------------
class _Tok:
    def __init__(self, mapping):
        self._m = mapping

    def convert_tokens_to_ids(self, token):
        return self._m.get(token)


AGREEING = _Tok({"<|image_pad|>": 151655, "<|vision_start|>": 151652,
                 "<|vision_end|>": 151653})
CFG = SimpleNamespace(image_token_id=151655, vision_start_token_id=151652,
                      vision_end_token_id=151653)

check("matching tokenizer and encoder produce no warnings",
      kie.token_id_consistency_warnings(AGREEING, CFG) == [])

DISAGREEING = _Tok({"<|image_pad|>": 999, "<|vision_start|>": 151652,
                    "<|vision_end|>": 151653})
warns = kie.token_id_consistency_warnings(DISAGREEING, CFG)
check("a mismatched vision token warns", len(warns) == 1, f"got {warns}")
check("the warning names the offending token",
      warns and "<|image_pad|>" in warns[0])
check("the warning names --te1 as the likely cause",
      warns and "--te1" in warns[0])
check("a config without the ids is skipped rather than warned",
      kie.token_id_consistency_warnings(AGREEING, SimpleNamespace()) == [])


# ---------------------------------------------------------------------------
print("\n── source order and count validation ──────────────────────────")
# ---------------------------------------------------------------------------
norm = kie.Krea2IdentityEditPipeline._normalize_sources
red = PIL.Image.new("RGB", (8, 8), (255, 0, 0))
blue = PIL.Image.new("RGB", (8, 8), (0, 0, 255))

single = norm(red)
check("a bare PIL image becomes a one-element list", len(single) == 1)

pair = norm([red, blue])
check("two sources are kept in the given order (scene, identity)",
      pair[0].getpixel((0, 0)) == (255, 0, 0)
      and pair[1].getpixel((0, 0)) == (0, 0, 255))
check("MAX_SOURCES is 2", kie.MAX_SOURCES == 2)

try:
    norm([red, blue, red])
    check("a third source is refused", False, "no error raised")
except ValueError as exc:
    check("a third source is a HARD ERROR, never a silent drop",
          "at most 2" in str(exc) or "MAX" in str(exc).upper(), str(exc))
    check("the error explains the slot meanings",
          "scene" in str(exc) and "identity" in str(exc))

try:
    norm([])
    check("an empty source list is refused", False, "no error raised")
except ValueError as exc:
    check("an empty source list is refused with a cause", "no source" in str(exc))


# ---------------------------------------------------------------------------
print("\n── ADR-028 coupling + unbound entry ───────────────────────────")
# ---------------------------------------------------------------------------
import inspect  # noqa: E402

sig = inspect.signature(kie.Krea2IdentityEditPipeline.__call__).parameters
check("__call__ accepts sigmas= (ADR-028 sigma-schedule gate coupling)",
      "sigmas" in sig)
check("__call__ exposes ref_boost", "ref_boost" in sig)
check("__call__ exposes grounding_px", "grounding_px" in sig)
check("__call__ takes `image` so the ref surface can pass one or two",
      "image" in sig)
check("identity_edit_pipe_call runs the subclass body unbound",
      "Krea2IdentityEditPipeline.__call__"
      in inspect.getsource(kie.identity_edit_pipe_call))
check("defaults match the model card (4.0 / 768)",
      kie.DEFAULT_REF_BOOST == 4.0 and kie.DEFAULT_GROUNDING_PX == 768)


# ---------------------------------------------------------------------------
print("\n── unbound-call safety: no self.-dispatched subclass methods ───")
# ---------------------------------------------------------------------------
# The defect this pins cost the first GPU run (2026-07-31): `__call__` ran
# `self._normalize_sources(image)`, but under identity_edit_pipe_call `self` is
# a STOCK Krea2Pipeline, so diffusers' ConfigMixin.__getattr__ raised
# AttributeError. Every CPU test before this one exercised the BOUND path,
# where `self.` resolves fine — so no amount of behavioural testing on a real
# subclass instance could have caught it. This is a structural guard instead:
# any subclass-defined METHOD reached through `self` is a latent crash.
import ast  # noqa: E402

_src = Path(kie.__file__).read_text()
_tree = ast.parse(_src)
_cls = next(n for n in ast.walk(_tree)
            if isinstance(n, ast.ClassDef)
            and n.name == "Krea2IdentityEditPipeline")
_own_methods = {n.name for n in _cls.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
check("guard found the subclass's own methods",
      len(_own_methods) >= 6, f"found {sorted(_own_methods)}")

_violations = [
    f"{node.attr} (line {node.lineno})"
    for node in ast.walk(_cls)
    if isinstance(node, ast.Attribute)
    and isinstance(node.value, ast.Name) and node.value.id == "self"
    and node.attr in _own_methods
]
check("no subclass method is called via self. (unbound-call safe)",
      not _violations,
      "self.-dispatched: " + ", ".join(_violations) if _violations else "")

# ...and the positive: they ARE reached class-qualified. Pins the fix's shape,
# so a "helpful" refactor back to self. fails both legs, not neither.
_qualified = {
    node.attr
    for node in ast.walk(_cls)
    if isinstance(node, ast.Attribute)
    and isinstance(node.value, ast.Name)
    and node.value.id == "Krea2IdentityEditPipeline"
}
for _m in ("_normalize_sources", "_target_size_for", "_grounded_encode",
           "_encode_source_latents", "_identity_vl_processor",
           "_cap_longest_side"):
    check(f"{_m} is called class-qualified", _m in _qualified)


print("\n──────────────────────────────────────────────────")
print(f"  {passed} passed, {failed} failed")
print("──────────────────────────────────────────────────")
sys.exit(0 if failed == 0 else 1)
