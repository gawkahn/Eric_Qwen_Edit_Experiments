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

# Default to the worktree this script lives in, so running it from any worktree
# (e.g. a branch with a different .venv) uses that worktree's code AND its
# .venv/bin/python3 — not a hardcoded path. Override REPO to point elsewhere.
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# hf-local (not its parent .../models): the parent also walks the HF hub cache,
# surfacing snapshot-hash-named dirs as catalog entries. hf-local is the curated
# set with human-readable names (Grant, 2026-06-26).
MODEL_BASE="${MODEL_BASE:-/home/gawkahn/projects/ai-lab/ai-base/models/hf-local}"
# ADR-018 kind-typed scan roots. LORA_PATH: every .safetensors under it (any
# depth) catalogs as a LoRA. TRANSFORMER_PATH_*: the two specific transformer
# trees — never their comfyui/models parent (it also contains loras/ etc.;
# cross-kind overlap fails the catalog build closed).
LORA_PATH="${LORA_PATH:-/home/gawkahn/projects/ai-lab/ai-base/models/comfyui/models/loras}"
TRANSFORMER_PATH_CKPT="${TRANSFORMER_PATH_CKPT:-/home/gawkahn/projects/ai-lab/ai-base/models/comfyui/models/checkpoints}"
TRANSFORMER_PATH_DIFF="${TRANSFORMER_PATH_DIFF:-/home/gawkahn/projects/ai-lab/ai-base/models/comfyui/models/diffusion_models}"
# ADR-022 S5: metadata DB enables the `search` tool + model_family filters.
# Read-only from the MCP server; lives OFF mergerfs (SQLite locking).
CATALOG_DB="${CATALOG_DB:-$HOME/.local/share/comfyless/catalog.sqlite}"
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
echo "[start-mcpo] lora-path=${LORA_PATH}"
echo "[start-mcpo] transformer-paths=${TRANSFORMER_PATH_CKPT} ${TRANSFORMER_PATH_DIFF}"
echo "[start-mcpo] output-dir=${OUTPUT_DIR}"

# --catalog-db only if the DB exists (spawn is fail-closed on a bad path;
# a fresh machine without a built catalog still gets a working server).
CATALOG_DB_ARGS=()
if [ -f "$CATALOG_DB" ]; then
  CATALOG_DB_ARGS=(--catalog-db "$CATALOG_DB")
  echo "[start-mcpo] catalog-db=${CATALOG_DB}"
else
  echo "[start-mcpo] catalog-db absent (${CATALOG_DB}) — search tool disabled"
fi

# PINNED (§11). `uvx mcpo` resolves mcpo AND its deps at LATEST on every run —
# the same floating-version footgun as `npx <tool>`, which §14 already blocks
# for gitnexus. It bit on 2026-07-29: mcp 2.0.0 dropped the
# `streamablehttp_client` symbol that mcpo 0.0.20 imports, so a machine that
# had worked for weeks failed at startup with an ImportError after a reboot,
# having pulled an artifact nobody reviewed.
#
# BOTH are pinned deliberately: `--from` fixes the tool, `--with` fixes the
# transitive that actually broke. Pinning only the transitive still leaves the
# tool floating. Bumping either is its own commit — verify the pair starts and
# serves /openapi.json before changing these numbers.
MCPO_VERSION="${MCPO_VERSION:-0.0.20}"
MCP_VERSION="${MCP_VERSION:-1.28.1}"   # NOT 2.x — see above
echo "[start-mcpo] pinned mcpo==${MCPO_VERSION} with mcp==${MCP_VERSION}"

exec uvx --from "mcpo==${MCPO_VERSION}" --with "mcp==${MCP_VERSION}" \
  mcpo --host "$HOST" --port "$PORT" -- \
  "$REPO/.venv/bin/python3" -m comfyless.mcp_server \
    --model-base "$MODEL_BASE" \
    --lora-path "$LORA_PATH" \
    --transformer-path "$TRANSFORMER_PATH_CKPT" \
    --transformer-path "$TRANSFORMER_PATH_DIFF" \
    "${CATALOG_DB_ARGS[@]}" \
    --output-dir "$OUTPUT_DIR"
