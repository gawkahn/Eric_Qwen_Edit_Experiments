#!/usr/bin/env bash
# start-mcpo.sh — bring up the mcpo OpenAPI bridge in front of the comfyless
# MCP stdio server so OpenWebUI (and any OpenAPI tool client) can call it.
#
# Background: OpenWebUI cannot call an MCP *stdio* server directly; mcpo
# (MCP -> OpenAPI proxy) bridges it. The native OWUI image tool
# (comfyless/integrations/openwebui/generate_image_tool.py, ADR-017) posts to
# this bridge's /generate endpoint. See memory/reference_mcpo_openwebui_bridge.
#
# Run in a dedicated terminal to watch logs, or `nohup ./start-mcpo.sh >~/mcpo.log 2>&1 &`
# for a detached service. Every var below can be overridden from the environment.
set -euo pipefail

REPO="${REPO:-/home/gawkahn/projects/ai-lab/code/Eric_Qwen_Edit_Experiments}"
# hf-local (not its parent .../models): the parent also walks the HF hub cache,
# surfacing snapshot-hash-named dirs as catalog entries. hf-local is the curated
# set with human-readable names (Grant, 2026-06-26).
MODEL_BASE="${MODEL_BASE:-/home/gawkahn/projects/ai-lab/ai-base/models/hf-local}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/gawkahn/gen-output}"
HOST="${MCPO_HOST:-172.17.0.1}"   # docker bridge gateway: reachable from host + containers, not the wider LAN
PORT="${MCPO_PORT:-8090}"
GPU="${MCPO_GPU:-0}"              # image gen pinned to GPU0 (dolphin/vLLM lives on GPU1)

# PYTHONPATH is required: the .venv is not an editable install, so `comfyless`
# is not importable from mcpo's cwd without it.
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME=/mnt/nvme-8tb/hf
export HF_HUB_CACHE=/mnt/nvme-8tb/hf

echo "[start-mcpo] binding ${HOST}:${PORT}, GPU=${GPU}"
echo "[start-mcpo] model-base=${MODEL_BASE}"
echo "[start-mcpo] output-dir=${OUTPUT_DIR}"

exec uvx mcpo --host "$HOST" --port "$PORT" -- \
  "$REPO/.venv/bin/python3" -m comfyless.mcp_server \
    --model-base "$MODEL_BASE" \
    --output-dir "$OUTPUT_DIR"
