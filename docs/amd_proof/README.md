# AMD Compute Proof

## Executed official-Gemma proof: AMD Instinct MI300X VF (`gfx942`)

[`mi300x/`](mi300x/) is a real GemmaJudge production-loop run captured on AMD Developer
Cloud on 2026-07-10. It uses official `google/gemma-3-*` model IDs after access was granted
on Hugging Face:

- Hardware: AMD Instinct MI300X VF (`gfx942`), 191.6875 GiB VRAM.
- Stack: ROCm/HIP 6.4, PyTorch 2.9.0.dev+rocm6.4, vLLM 0.1.dev.
- Attacker + Judge: `google/gemma-3-4b-it`.
- Target: `google/gemma-3-1b-it`.
- Serving: two local OpenAI-compatible vLLM endpoints on the same AMD GPU.
- Eval: 3 model-generated hallucination cases, ASR 100% (3/3), full pipeline wall clock
  55.31 seconds, with a 25-second client timeout per request.

The committed artifacts are:

- [x] `rocm_smi.txt` - `rocm-smi` plus `rocminfo` identity output showing AMD MI300X VF and
  `gfx942`.
- [x] `versions.txt` - ROCm, vLLM, PyTorch, GPU, and VRAM versions.
- [x] `serve_command.txt` - exact two-server launch commands.
- [x] `vllm_engine.log` and `vllm_target.log` - ROCm model loading and API-server startup.
- [x] `attacker_prompt.md` - the exact novelty-safe attacker prompt used.
- [x] `eval_result.json` - complete attacker output, target responses, verdicts, token usage,
  metrics, and repeat-score data.
- [x] `eval_summary.json` - clean machine-readable run summary.
- [x] `notes.md` - hardware and metric scope notes.
- [x] `proof.png` - terminal screenshot showing the hardware identity and run summary.

The generated attacks are raw model output. They are a small recorded demonstration, not a
benchmark or human-calibrated factuality set. Repeated temperature-0 scores show deterministic
score stability only; they do not establish judge correctness.

## Historical W7900 proof

[`w7900/`](w7900/) remains committed as earlier AMD Radeon PRO W7900 (`gfx1100`) evidence.
Its attacker prompt contained concrete examples that could be repeated by the model, so it is
historical provenance rather than the primary public demonstration. Do not use its 80% ASR or
134.72-second full-pipeline metric in current public materials.

## Reproduce on an AMD vLLM environment

1. Accept the Gemma Terms and authenticate locally with a read token. Never commit that token.
2. Clone the repository and install the Python dependencies required by the runner.
3. Start the two local servers using the commands in `mi300x/serve_command.txt`.
4. Run the production seam with the bounded-timeout runner:

```bash
python scripts/run_amd_proof.py \
  --engine-model google/gemma-3-4b-it \
  --target-model google/gemma-3-1b-it \
  --backend-label "AMD Instinct MI300X VF (gfx942, ROCm 6.4)" \
  --n 3 \
  --request-timeout 25 \
  --output docs/amd_proof/mi300x/eval_result.json
```

`--request-timeout` rejects values greater than 30 seconds. The full multi-stage pipeline can
be longer than an individual bounded request.
