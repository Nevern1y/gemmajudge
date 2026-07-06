# AMD Compute Proof

This folder holds the evidence that Gemma runs on **AMD Instinct™ MI300X** for GemmaJudge.
It satisfies the Track 3 requirement: *"AMD compute usage is a requirement: projects that
do not demonstrate it will be disqualified."*

Populate by **end of Day 3** (see WORK_SPLIT.md):

- [ ] `vllm_rocm_config.*` — the exact vLLM + ROCm launch config used to serve Gemma on MI300X.
- [ ] `deploy_log.txt` — terminal log showing the model loading and serving on ROCm.
- [ ] `mi300x_screenshot.png` — screenshot showing the model running on AMD hardware
      (e.g. `rocm-smi` output + the served endpoint).
- [ ] `notes.md` — one paragraph: model id, VRAM footprint, measured throughput.

> These files are intentionally committed (the `.gitignore` `runs/` rule does not cover this folder).
