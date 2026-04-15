#!/usr/bin/env python3
"""
Quick standalone test for Flux.2-dev generation.
Run with the ComfyUI venv python:
  /home/gawkahn/projects/ai-lab/ai-stack-data/comfyui/run/venv/bin/python3 test_flux2.py
"""

import torch
from diffusers import Flux2Pipeline

MODEL_PATH = "/home/gawkahn/projects/ai-lab/ai-base/models/hf-local/Flux.2-dev"
OUTPUT_PATH = "/tmp/flux2_test.png"

PROMPT = "A red fox sitting in a snowy forest, photorealistic"
STEPS = 20
GUIDANCE = 4.0
WIDTH = 1024
HEIGHT = 1024
SEED = 42

print(f"Loading Flux2Pipeline from {MODEL_PATH} ...")
# device_map="auto" splits the model across GPUs automatically via accelerate.
# Do NOT call .to() when using device_map — accelerate manages placement.
pipe = Flux2Pipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    device_map="balanced",
)

if hasattr(pipe.vae, "enable_tiling"):
    pipe.vae.enable_tiling()

print(f"Generating {WIDTH}x{HEIGHT}, {STEPS} steps, guidance={GUIDANCE} ...")
exec_device = pipe._execution_device
print(f"Execution device: {exec_device}")
generator = torch.Generator(device=exec_device).manual_seed(SEED)

result = pipe(
    prompt=PROMPT,
    height=HEIGHT,
    width=WIDTH,
    num_inference_steps=STEPS,
    guidance_scale=GUIDANCE,
    generator=generator,
)

result.images[0].save(OUTPUT_PATH)
print(f"Saved: {OUTPUT_PATH}")
