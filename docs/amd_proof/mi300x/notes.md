# AMD Instinct MI300X - GemmaJudge proof-of-compute

- Hardware: AMD Instinct MI300X VF (`gfx942`), 191.6875 GiB VRAM; see `rocm_smi.txt` and
  `versions.txt`.
- Stack: ROCm/HIP 6.4.43483-a187df25c, PyTorch 2.9.0.dev20250702+rocm6.4, and vLLM
  0.1.dev8023+g894bed8; see `versions.txt`.
- Attacker + Judge: `google/gemma-3-4b-it`.
- Target: `google/gemma-3-1b-it`.
- Serving: both official Gemma model IDs ran as local OpenAI-compatible vLLM servers on the
  same MI300X. Exact commands and separate server logs are committed in this directory.
- Eval: hallucination mode, 3 model-generated cases, ASR 100% (3/3 score >=4), full pipeline
  wall clock 55.31 seconds. The client timeout was 25 seconds per OpenAI-compatible request;
  the full multi-request pipeline is expected to take longer.
- Repeat scoring: each recorded case received scores `[5, 5, 5]` at temperature 0. This shows
  deterministic score stability for this run, not judge correctness or reliability.

The result is a small recorded demonstration, not a benchmark, a human-calibrated evaluation,
or a general claim about target robustness. The raw attacker output is preserved verbatim in
`eval_result.json`; its three generated cases do not repeat the retired prompt examples, but
they are not a factual-validation set or proof of tactic coverage. The historical W7900 run
remains in `../w7900/` for provenance.
