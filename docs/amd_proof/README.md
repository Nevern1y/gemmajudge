# AMD Compute Proof

Evidence that Gemma runs on **AMD Instinct™ MI300X** for GemmaJudge — satisfying the
Track 3 gate: *"AMD compute usage is a requirement: projects that do not demonstrate it
will be disqualified."*

## Environment (AMD AI Notebooks portal)

- **Access:** AMD AI Developer Program → *AMD AI Notebooks* → **Request Notebook**.
- **Runtime:** ROCm 7.2 + vLLM 0.16.0 + PyTorch 2.9.
- **Hardware:** AMD Instinct MI300X (192 GB) — one GPU per notebook.
- **Budget:** ~**4 hrs per 24 hrs**. Don't idle. Serve → capture proof → stop.

## How to generate the proof (one notebook run)

1. Accept the **Gemma license** on Hugging Face for your chosen model and create a read
   token. Gemma weights are gated.
2. In the AMD Jupyter environment, open **[`mi300x_gemma.ipynb`](mi300x_gemma.ipynb)** and
   run the cells top to bottom. It:
   - prints `rocm-smi` + the ROCm/vLLM/PyTorch versions (hardware proof),
   - serves Gemma via vLLM (OpenAI-compatible) — the **Attacker + Judge**, and optionally a
     weak Gemma **target**, all on the one MI300X,
   - runs a smoke inference + a throughput measurement,
   - runs the **full GemmaJudge loop** against the local endpoint and saves the result,
   - writes `notes.md`,
   - tears the servers down to stop the budget clock.
   (Terminal alternative: [`serve_gemma_mi300x.sh`](serve_gemma_mi300x.sh).)
3. **Screenshot** the `rocm-smi` cell and the ASR output — those are the deck visuals.
4. Copy the notebook's `amd_proof_artifacts/*` into this folder and commit.

## Artifacts to commit here (the DQ gate is green when these exist)

- [ ] `rocm_smi.txt` — `rocm-smi` output showing the MI300X.
- [ ] `versions.txt` — ROCm / vLLM / PyTorch versions.
- [ ] `vllm_engine.log` (+ `vllm_target.log`) — model loading + serving on ROCm.
- [ ] `serve_command.txt` — the exact vLLM launch command(s) used.
- [ ] `eval_result.json` — a real GemmaJudge run against the MI300X endpoint.
- [ ] `mi300x_screenshot.png` — `rocm-smi` + the served endpoint (for the deck).
- [ ] `notes.md` — model id, VRAM footprint, measured throughput (auto-written by the notebook).

> These files are intentionally committed (the `.gitignore` `runs/` rule does not cover this folder).
> Do **not** commit any HF token — it goes in the notebook's runtime env only.
