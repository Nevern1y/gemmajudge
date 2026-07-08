# AMD Compute Proof

> ## ✅ Committed proof-of-compute: AMD Radeon PRO W7900 (gfx1100)
>
> The Track-3 AMD-compute gate is satisfied by a **real GemmaJudge run on an AMD Radeon PRO
> W7900 (gfx1100 / RDNA3, 48 GB) via vLLM + ROCm 7.2** on AMD Developer Cloud. Artifacts are in
> [`w7900/`](w7900/): `rocm_smi.txt` (shows `gfx1100`), `versions.txt`, `serve_command.txt`,
> `vllm_engine.log` (+ `vllm_target.log`), `eval_result.json` (hallucination **ASR 80%**, judge
> self-consistency **stdev 0.00**, 136.9 tok/s), and `notes.md`. Gemma-3-4b-it ran as the
> attacker+judge and Gemma-3-1b-it as the target, both on the one 48 GB GPU.
>
> The **MI300X notebook / script below is the AMD Instinct reference path** (identical code,
> `INFERENCE_BACKEND=mi300x`) and has **not been executed yet** — every "MI300X" mention below
> refers to that reference runbook, not to the committed proof.

---

Evidence that Gemma runs on **AMD (self-hosted via vLLM + ROCm)** for GemmaJudge — satisfying
the Track 3 gate: *"AMD compute usage is a requirement: projects that do not demonstrate it
will be disqualified."*

## Environment (AMD AI Notebooks portal)

- **Access:** AMD AI Developer Program → *AMD AI Notebooks* → **Request Notebook**.
- **Runtime:** ROCm 7.2 + vLLM 0.16.0 + PyTorch 2.9.
- **Hardware:** AMD Instinct MI300X (192 GB) — one GPU per notebook. *(Reference target; the
  committed run above used an AMD Radeon PRO W7900 / gfx1100, 48 GB, from AMD Developer Cloud.)*
- **Budget:** ~**4 hrs per 24 hrs**. Don't idle. Serve → capture proof → stop.

## How to generate the proof (one notebook run)

1. Accept the **Gemma license** on Hugging Face for your chosen model and create a read
   token. Gemma weights are gated.
2. In the AMD Jupyter environment, open **[`mi300x_gemma.ipynb`](mi300x_gemma.ipynb)** and
   run the cells top to bottom. It:
   - prints `rocm-smi` + the ROCm/vLLM/PyTorch versions (hardware proof),
   - serves Gemma via vLLM (OpenAI-compatible) — the **Attacker + Judge**, and optionally a
     weak Gemma **target**, all on the one GPU,
   - runs a smoke inference + a throughput measurement,
   - runs the **full GemmaJudge loop** against the local endpoint and saves the result,
   - writes `notes.md`,
   - tears the servers down to stop the budget clock.
   (Terminal alternative: [`serve_gemma_mi300x.sh`](serve_gemma_mi300x.sh).)
3. **Screenshot** the `rocm-smi` cell and the ASR output — those are submission/demo visuals.
4. Copy the notebook's `amd_proof_artifacts/*` into this folder and commit.

## Artifacts committed (the DQ gate is green when these exist)

The committed set lives in [`w7900/`](w7900/) (real AMD Radeon PRO W7900 run):

- [x] `rocm_smi.txt` — `rocm-smi` output showing the AMD GPU (`gfx1100`).
- [x] `versions.txt` — ROCm / vLLM / PyTorch versions.
- [x] `vllm_engine.log` (+ `vllm_target.log`) — model loading + serving on ROCm.
- [x] `serve_command.txt` — the exact vLLM launch command(s) used.
- [x] `eval_result.json` — a real GemmaJudge run against the AMD endpoint.
- [x] `notes.md` — model id, VRAM footprint, measured throughput.
- [x] `proof.png` — screenshot used by the app's AMD Proof page.

> These files are intentionally committed (the `.gitignore` `runs/` rule does not cover this folder).
> Do **not** commit any HF token — it goes in the notebook's runtime env only.
