# AMD Radeon PRO W7900 - GemmaJudge proof-of-compute (run notes)

- Hardware: AMD Radeon PRO W7900 (gfx1100 / RDNA3, 48 GB VRAM) + AMD EPYC 9334 32-core (see rocm_smi.txt)
- Stack: ROCm 7.2.1 + vLLM 0.16 (rocm721 build) + PyTorch 2.9.1 (see versions.txt)
- NOTE: this is the AMD Radeon (gfx1100) fallback proof, NOT MI300X. The Track-3 AMD-compute requirement is met via self-hosted Gemma on AMD ROCm; MI300X run is a separate follow-up.
- Attacker+Judge (load-bearing Gemma): google/gemma-3-4b-it
- Target (system-under-test): google/gemma-3-1b-it
- Serving: vLLM OpenAI-compatible, both models on the one 48 GB GPU, max-model-len 4096, --enforce-eager (see serve_command.txt)
- Measured throughput: 136.9 completion tok/s over a batch of 8 (engine gemma-3-4b-it)
- Eval: hallucination, 5 cases, ASR 80% (4/5 failed), wall 134.7s (full run in eval_result.json)
- F9b judge self-consistency: score stdev 0.00 across re-judges (perfectly stable)

Gemma runs in BOTH the attacker and judge roles on this AMD GPU - the load-bearing
AMD compute usage for Track 3. Launch command(s) are in serve_command.txt.