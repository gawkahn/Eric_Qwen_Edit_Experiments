#!/usr/bin/env python3
"""Pixel-exact regression baseline for the comfyless_diffusion extraction.

Reproducibility is MODEL-DEPENDENT (measured 2026-08-21):

  reproducible (0 differing px across runs):
      Krea-2-Turbo, Qwen-Image-2512, FLUX.2-klein-9B, Juggernaut-XL_v9
  NOT reproducible (whole-image drift, ~80-91% of px, max |d| 159-254):
      Z-Image-Turbo, Chroma1-Flash

Z-Image was bit-identical on back-to-back runs, then shifted ONCE between two
capture batches and has been stable since -- batch-stable, not call-stable.  So
a back-to-back probe CANNOT establish reproducibility; only comparison across
separated batches can.  Mechanism unexplained; do not assume it will not shift
again.

EXECUTION PATH IS PINNED.  `--savepath` delegates the run to the daemon when
one is alive and silently falls back in-process when it is not -- a hidden
variable that flipped mid-experiment on 2026-08-21 when the cuda:0 daemon died,
and the most likely cause of the "unexplained" Z-Image shift recorded below.
This harness passes an explicit `--output`, which `_should_delegate_to_server`
skips, so every case runs in-process regardless of daemon state.

Cases carry strict=True/False.  Only strict cases can fail a comparison;
non-strict ones are reported as informational drift.

--schedule / --sampler are INERT on SDXL (byte-identical output) -- that family
does not use the flow-match manual loop.  Sweeps therefore run on Krea-2-Turbo,
which is both reproducible and discriminating.

Compare PIXELS, never file hashes: the PNG `parameters` chunk embeds the
savepath, so file digests differ between runs whose pixels are identical.

Usage:
    python3 capture_baseline.py capture   > writes manifest-<label>.json
    python3 capture_baseline.py compare manifest-before.json manifest-after.json
"""
import subprocess, sys, json, glob, os, time, hashlib
import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY   = os.path.join(REPO, ".venv/bin/python3")
MB   = os.path.expanduser("~/projects/ai-lab/ai-base/models/hf-local")
OUT  = os.path.join(os.path.expanduser("~"), "comfyless-baseline-out")
HERE = os.path.join(REPO, "tests", "golden")

PROMPT = ("A weathered brass sextant resting on a folded nautical chart, beside "
          "exactly three tangerines and a cracked porcelain teacup. A handwritten "
          "paper label reads MERIDIAN 47. Late afternoon window light, deep "
          "shadows, fine dust in the air.")
SEED, STEPS, W, H = 77341, 8, 1024, 1024

def case(tag, model, *extra, steps=8, cfg=None, strict=True):
    return {"tag": tag, "model": model, "extra": list(extra), "steps": steps,
            "cfg": cfg, "strict": strict}

SWEEP = "Krea-2-Turbo"       # reproducible AND sensitive to schedule/sampler
SW_STEPS = 12
LORAS = os.path.expanduser("~/projects/ai-lab/ai-base/models/comfyui/models/loras")

CASES = []
# A - sigma schedule sweep (build_sigma_schedule wiring)
for s in ("linear", "balanced", "karras", "beta57", "bong_tangent"):
    CASES.append(case(f"sched-{s}", SWEEP, "--schedule", s, steps=SW_STEPS))
# B - sampler sweep (eric_diffusion_samplers + manual loop)
for s in ("default", "multistep2", "multistep3", "res_2m", "res_3m"):
    CASES.append(case(f"samp-{s}", SWEEP, "--sampler", s, steps=SW_STEPS))
# C - family routing (eric_diffusion_utils detection + CFG branch)
CASES += [
    case("fam-krea2-turbo", "Krea-2-Turbo",     steps=12),
    case("fam-flux2-klein", "FLUX.2-klein-9B",  steps=12),
    case("fam-sdxl-jugg",   "Juggernaut-XL_v9", steps=20),
    case("fam-qwen-2512",   "Qwen-Image-2512",  steps=12),
    # non-strict: real coverage of their routing paths, but drift-prone
    case("fam-zimage-turbo", "Z-Image-Turbo",  steps=8,  strict=False),
    case("fam-chroma-flash", "Chroma1-Flash",  steps=12, strict=False),
]
# D - feature surfaces whose modules are moving
CASES += [
    case("feat-lora-krea2", SWEEP,
         "--lora", f"{LORAS}/Krea/krea2_turbo_lora_rank_64_bf16.safetensors:0.7",
         steps=SW_STEPS),
    case("feat-nag-krea2",  SWEEP, "--nag-scale", "5.0", steps=SW_STEPS),
    case("feat-quant-fp8",  "Qwen-Image-2512", "--quant", "fp8", steps=12),
    # ADR-030 2x Wan decode — the surface slice 3b reimplements. Qwen-only
    # (the upscale VAE shares Qwen's latent space); output is 2048x2048.
    case("feat-upscale-vae", "Qwen-Image-2512",
         "--upscale-vae", f"{MB}/Wan2.1-VAE-upscale2x",
         steps=12),
]

def newest(tag):
    # explicit --output => exact path, no increment suffix, no glob race
    p = os.path.join(OUT, tag + ".png")
    return p if os.path.exists(p) else None

def pixhash(path):
    a = np.array(Image.open(path).convert("RGB"))
    return hashlib.sha256(a.tobytes()).hexdigest(), list(a.shape)

def capture(label):
    man = {"_meta": {"label": label, "captured": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "seed": SEED, "prompt": PROMPT,
                     "git_head": subprocess.run(["git","-C",REPO,"rev-parse","--short","HEAD"],
                                                capture_output=True,text=True).stdout.strip()},
           "cases": {}}
    os.makedirs(OUT, exist_ok=True)
    for c in CASES:
        tag = f"bl-{label}-{c['tag']}"
        cmd = [PY, "-m", "comfyless.generate", "--model", f"{MB}/{c['model']}",
               "--prompt", PROMPT, "--seed", str(SEED), "--steps", str(c["steps"]),
               "--width", str(W), "--height", str(H), "--device", "cuda:0",
               "--output", os.path.join(OUT, tag + ".png")] + c["extra"]
        if c["cfg"] is not None: cmd += ["--cfg", str(c["cfg"])]
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
        dt = round(time.time() - t0, 1)
        f = newest(tag)
        if r.returncode != 0 or not f:
            err = (r.stderr or r.stdout or "").strip().splitlines()
            man["cases"][c["tag"]] = {"status": "FAILED", "rc": r.returncode,
                                      "err": err[-3:], "secs": dt}
            print(f"  {c['tag']:22s} FAILED ({dt}s) rc={r.returncode}", flush=True)
            continue
        h, shape = pixhash(f)
        man["cases"][c["tag"]] = {"status": "ok", "model": c["model"],
                                  "extra": c["extra"], "steps": c["steps"],
                                  "strict": c["strict"],
                                  "pixel_sha256": h, "shape": shape,
                                  "file": os.path.basename(f), "secs": dt}
        print(f"  {c['tag']:22s} ok  {dt:6.1f}s  {h[:16]}", flush=True)
    p = os.path.join(HERE, f"manifest-{label}.json")
    json.dump(man, open(p, "w"), indent=2)
    print(f"\nmanifest -> {p}")
    return man

def compare(a, b):
    A, B = json.load(open(a))["cases"], json.load(open(b))["cases"]
    bad = 0
    for k in sorted(set(A) | set(B)):
        x, y = A.get(k), B.get(k)
        if not x or not y:            print(f"  {k:22s} MISSING on one side"); bad += 1; continue
        if x["status"] != "ok" or y["status"] != "ok":
            print(f"  {k:22s} not-ok: {x['status']}/{y['status']}"); bad += 1; continue
        if x["pixel_sha256"] != y["pixel_sha256"]:
            if x.get("strict", True):
                print(f"  {k:22s} PIXEL MISMATCH  (strict)"); bad += 1
            else:
                print(f"  {k:22s} drift (non-strict, informational)")
    print(f"\n{'ALL MATCH' if not bad else f'{bad} DIVERGENCE(S)'}")
    return 1 if bad else 0

if __name__ == "__main__":
    if sys.argv[1] == "capture": capture(sys.argv[2] if len(sys.argv) > 2 else "before")
    else: sys.exit(compare(sys.argv[2], sys.argv[3]))
