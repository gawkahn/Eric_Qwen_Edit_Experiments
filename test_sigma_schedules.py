#!/usr/bin/env python3
"""Test harness for comfyless.core.sigma_schedules — the slice-3 equivalence gate.

`build_sigma_schedule` moved out of `nodes/eric_qwen_image_multistage.py` into
a from-scratch comfyless implementation (ADR-045 slice 3), so that the
extracted package carries no third-party lineage. The node pack keeps its own
copy and is not touched.

This is the STRICT proof the Vision demands. A sigma schedule is replay
infrastructure: every sidecar that records `schedule` re-derives its sigmas
here on replay, so a divergence does not surface as an error — it silently
produces a different image from the same recorded parameters. Divergence is
therefore escalated, never absorbed.

Invariants, each with a negative case:

  1. Golden equivalence  — every one of the 13,000 frozen cases
                           (`tests/golden/sigma_schedules.json.gz`, captured
                           from the node-pack original at de0145e) matches
                           elementwise, and matches BITWISE. Negative: the
                           golden verifies its own sha256 first, so a
                           tampered or truncated oracle fails loudly instead
                           of vacuously passing.
  2. Range contract      — output descends, spans exactly
                           [sigma_min, sigma_start], and has the documented
                           length. Negative: no value may escape the range.
  3. Shared start sigma  — at a given (num_steps, denoise) ALL schedules
                           begin at the same sigma. This is the property that
                           makes schedule a free knob in a sidecar replay.
                           Negative: they must nonetheless DIFFER in between,
                           or the warps are not doing anything.
  4. Warp ordering       — rho=7 (karras) concentrates toward low sigma more
                           than rho=3 (balanced), which beats linear.
                           Negative: an unrecognised name must fall back to
                           linear EXACTLY, not merely approximately.
  5. Signature contract  — `power` is accepted and ignored; degenerate step
                           counts stay well formed. Negative: bong_tangent at
                           count <= 2 must equal linear, where its two-stage
                           split would otherwise degenerate.
"""

import gzip
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from comfyless.core.sigma_schedules import SIGMA_SCHEDULES, build_sigma_schedule

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


def _raises_typeerror(steps):
    try:
        build_sigma_schedule(steps, 1.0, "linear")
    except TypeError:
        return True
    except Exception:
        return False
    return False


GOLDEN = Path(__file__).parent / "tests" / "golden" / "sigma_schedules.json.gz"


# ── 1. Golden equivalence ─────────────────────────────────────────────
print("\n=== 1. equivalence with the node-pack original ===")

blob = json.load(gzip.open(GOLDEN, "rt"))
meta, cases = blob["_meta"], blob["cases"]

# NEGATIVE: verify the oracle before trusting it. A truncated or edited
# golden would otherwise let this whole suite pass on fewer/altered cases.
digest = hashlib.sha256(
    json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
check("golden matches its recorded sha256", digest == meta["sha256"],
      f"{digest} != {meta['sha256']}")
check(f"golden holds the recorded {meta['cases']} cases",
      len(cases) == meta["cases"], f"got {len(cases)}")

worst = 0.0
worst_key = None
bitwise = 0
bad_length = []
for key, want in cases.items():
    steps, denoise, schedule, power = key.split("|")
    got = build_sigma_schedule(int(steps), float(denoise), schedule, float(power))
    if len(got) != len(want):
        bad_length.append(key)
        continue
    delta = max((abs(a - b) for a, b in zip(got, want)), default=0.0)
    if delta == 0.0:
        bitwise += 1
    if delta > worst:
        worst, worst_key = delta, key

check("every golden case has the original length", not bad_length,
      f"{len(bad_length)} mismatched, e.g. {bad_length[:3]}")
check(f"all {len(cases)} cases match elementwise (<=1e-12)", worst <= 1e-12,
      f"worst |diff|={worst:.3e} at {worst_key}")
# Drift tripwire. Equality is currently EXACT, not merely close. If a numpy or
# scipy bump perturbs a last bit this fails while the check above still
# passes — that is the signal to escalate and re-confirm intent, per the
# Vision's "escalate rather than absorb". beta57 is the likeliest trigger,
# since it routes through scipy.stats.beta.ppf.
# WHEN IT FIRES, DO NOT LOOSEN THIS ASSERTION. Dependency bumps are their own
# deliberate slice (global §11): re-capture the golden inside that bump slice
# and record in the commit body that replayed sigmas moved, so the change is
# on the record rather than absorbed by a widened tolerance.
check(f"all {len(cases)} cases match bitwise", bitwise == len(cases),
      f"{len(cases) - bitwise} cases differ in the last bits")

# Every schedule name is actually exercised by the golden.
covered = {k.split("|")[2] for k in cases}
check("golden covers all five schedules", covered == set(SIGMA_SCHEDULES),
      f"covered={sorted(covered)}")


# ── 2. Range contract ─────────────────────────────────────────────────
print("\n=== 2. range and ordering ===")

descending = True
in_range = True
lengths = True
for steps in (1, 2, 3, 7, 20, 50):
    for denoise in (0.05, 0.5, 0.999, 1.0):
        for sched in SIGMA_SCHEDULES:
            s = build_sigma_schedule(steps, denoise, sched)
            expect = steps if denoise >= 1.0 else max(1, int(round(steps * denoise)))
            lengths &= len(s) == expect
            descending &= all(a >= b for a, b in zip(s, s[1:]))
            # Bound against LINEAR's start, not this schedule's own first
            # element — otherwise a curve that started too high would be
            # measured against itself and pass regardless.
            lo = 1.0 / steps
            hi = build_sigma_schedule(steps, denoise, "linear")[0]
            in_range &= all(lo - 1e-12 <= v <= hi + 1e-12 for v in s)

check("length is round(num_steps * denoise)", lengths)
check("sigmas descend", descending)
check("NEGATIVE: no sigma escapes [sigma_min, sigma_start]", in_range)
check("last sigma is sigma_min",
      abs(build_sigma_schedule(20, 1.0, "karras")[-1] - 1.0 / 20) < 1e-12)
check("full denoise starts at sigma_max",
      abs(build_sigma_schedule(20, 1.0, "karras")[0] - 1.0) < 1e-12)


# ── 3. Shared start sigma ─────────────────────────────────────────────
print("\n=== 3. all schedules share a start sigma ===")

shared = True
differ = True
for steps in (12, 30, 41):
    for denoise in (0.35, 0.6, 0.85):
        firsts = [build_sigma_schedule(steps, denoise, s)[0] for s in SIGMA_SCHEDULES]
        shared &= max(firsts) - min(firsts) < 1e-12
        # NEGATIVE: identical endpoints must not mean identical curves.
        # Sample the MIDDLE of each curve, where the warps are furthest apart.
        curves = [build_sigma_schedule(steps, denoise, s) for s in SIGMA_SCHEDULES]
        mids = [c[len(c) // 2] for c in curves]
        differ &= (max(mids) - min(mids)) > 1e-6

check("start sigma is schedule-independent", shared)
check("NEGATIVE: the curves still differ in between", differ)


# ── 4. Warp ordering ──────────────────────────────────────────────────
print("\n=== 4. warp strength ordering ===")

# Area under the descending curve is a simple proxy for how fast a schedule
# drops toward low sigma: the harder the warp, the smaller the sum.
area = {s: sum(build_sigma_schedule(40, 1.0, s)) for s in SIGMA_SCHEDULES}
check("karras (rho=7) concentrates lower than balanced (rho=3)",
      area["karras"] < area["balanced"], f"{area['karras']:.4f} vs {area['balanced']:.4f}")
check("balanced concentrates lower than linear",
      area["balanced"] < area["linear"], f"{area['balanced']:.4f} vs {area['linear']:.4f}")

# NEGATIVE: unknown names fall back to linear EXACTLY.
lin = build_sigma_schedule(23, 0.7, "linear")
for bogus in ("cosine", "", "KARRAS", "bong-tangent", "linear "):
    check(f"unknown schedule {bogus!r} falls back to linear",
          build_sigma_schedule(23, 0.7, bogus) == lin)


# ── 5. Signature contract ─────────────────────────────────────────────
print("\n=== 5. signature and degenerate inputs ===")

base = build_sigma_schedule(16, 0.8, "karras")
check("power is accepted and ignored",
      all(build_sigma_schedule(16, 0.8, "karras", p) == base
          for p in (0.0, 0.5, 1.0, 2.0, 99.0)))
check("denoise > 1.0 behaves as full denoise",
      build_sigma_schedule(16, 4.0, "karras") == build_sigma_schedule(16, 1.0, "karras"))
check("tiny denoise still yields at least one sigma",
      len(build_sigma_schedule(50, 0.0001, "karras")) == 1)

# NEGATIVE: bong_tangent's two-stage split degenerates at count <= 2, so it
# must hand off to linear rather than emit a short or empty stage.
for steps, denoise in ((2, 1.0), (1, 1.0), (20, 0.1), (20, 0.05)):
    n = steps if denoise >= 1.0 else max(1, int(round(steps * denoise)))
    if n <= 2:
        check(f"bong_tangent falls back to linear at count={n}",
              build_sigma_schedule(steps, denoise, "bong_tangent")
              == build_sigma_schedule(steps, denoise, "linear"))
check("bong_tangent diverges from linear once the split is viable",
      build_sigma_schedule(20, 1.0, "bong_tangent")
      != build_sigma_schedule(20, 1.0, "linear"))
check("all outputs are plain floats",
      all(type(v) is float for v in build_sigma_schedule(9, 0.9, "beta57")))

# Degenerate num_steps. The golden grid starts at num_steps=1, so these were
# never sampled by it; they were confirmed identical to the node-pack original
# by differential testing before the two implementations parted company, and
# are frozen here because that original will not be available to compare
# against once the repos split.
# num_steps=0 divides by zero computing sigma_min; a negative count reaches
# np.linspace, which rejects it.
for bad_steps, want_exc in ((0, ZeroDivisionError), (-1, ValueError), (-7, ValueError)):
    raised = None
    try:
        build_sigma_schedule(bad_steps, 1.0, "linear")
    except Exception as exc:
        raised = type(exc)
    check(f"num_steps={bad_steps} raises {want_exc.__name__} as before",
          raised is want_exc, f"raised {raised}")
check("denoise=0 collapses to a single sigma",
      len(build_sigma_schedule(30, 0.0, "karras")) == 1)

# NEGATIVE: a non-integer step count must be rejected by EVERY schedule, not
# just the ones that happen to route through np.linspace. bong_tangent builds
# its curve with np.arange, which accepts floats, so without an explicit guard
# it alone returns a schedule where the other four raise — the schedules would
# stop answering identically, which is what the replay contract rests on.
for sched in SIGMA_SCHEDULES:
    raised = None
    try:
        build_sigma_schedule(12.5, 1.0, sched)
    except Exception as exc:
        raised = type(exc)
    check(f"non-integer num_steps rejected by {sched}", raised is TypeError,
          f"raised {raised}")
# Even an integer-VALUED float is rejected: the original raised on it, so
# coercing here would be a silent behaviour change, not a convenience.
check("integer-valued float is rejected too (no silent coercion)",
      _raises_typeerror(12.0))
check("a real int is of course accepted", not _raises_typeerror(12))


# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print(f"  {passed} passed, {failed} failed")
print("─" * 50)
sys.exit(1 if failed else 0)
