#!/usr/bin/env bash
# Serve Gemma on AMD Instinct MI300X via vLLM (ROCm), OpenAI-compatible on :8000.
#
# For the AMD AI Notebooks environment (ROCm 7.2 + vLLM 0.16.0 + PyTorch 2.9).
# The notebook `mi300x_gemma.ipynb` automates this + captures proof; this script is
# the terminal equivalent / reference for the launch command.
#
# Prereqs:
#   export HF_TOKEN=hf_xxx    # Gemma weights are gated — accept the license on HF first
#
# Budget note: the notebook GPU is capped at ~4 hrs / 24 hrs. Serve, capture proof,
# then stop the process to free the clock.
set -euo pipefail

MODEL="${MODEL:-google/gemma-3-12b-it}"   # bump to gemma-3-27b-it if budget allows
PORT="${PORT:-8000}"
MEM_UTIL="${MEM_UTIL:-0.90}"              # fraction of the 192 GB MI300X to reserve
MAX_LEN="${MAX_LEN:-4096}"               # our prompts are short; small KV = fast start

export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:?set HF_TOKEN to your Hugging Face read token}"

echo "== hardware =="
rocm-smi || true
python - <<'PY'
import torch, vllm
print("torch", torch.__version__, "| vllm", vllm.__version__, "| hip", torch.version.hip)
PY

echo "== serving ${MODEL} on :${PORT} =="
exec vllm serve "${MODEL}" \
  --port "${PORT}" \
  --gpu-memory-utilization "${MEM_UTIL}" \
  --max-model-len "${MAX_LEN}" \
  --served-model-name "${MODEL}"

# Then point GemmaJudge at it:
#   INFERENCE_BACKEND=mi300x  MI300X_BASE_URL=http://localhost:8000/v1  MODEL_ID=google/gemma-3-12b-it
#   python -m gemmajudge.demo --n 10
