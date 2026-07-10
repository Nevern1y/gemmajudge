# Real Gemma runs (evidence)

These are **real** GemmaJudge runs — actual Gemma weights judging real model
responses, not the simulated demo. They were produced on **2026-07-07**.

## How they were produced

- **Engine (Attacker + Judge):** `gemma-3-27b-it`, served on a **Fireworks on-demand
  deployment** (dedicated **2× NVIDIA H100 80GB**, BF16). Gemma generates the
  adversarial prompts *and* scores every response — one open-weight family on both
  sides of the loop.
- **Backend note:** Fireworks has **no serverless Gemma** (all Gemma models are
  deploy-only), so the judge ran on a dedicated deployment. The *same*
  OpenAI-compatible code path runs Gemma **self-hosted on AMD** (vLLM + ROCm) — see
   [`../amd_proof/`](../amd_proof/) for the AMD proof-of-compute (the primary committed
   evidence is a real **AMD Instinct MI300X VF / gfx942** run).
- **Targets:** live Fireworks **serverless** frontier models
  (`gpt-oss-120b`, `glm-5p1`, `glm-5p2`, `deepseek-v4-pro`, `kimi-k2p6`).
- Total credit spend for all of this: **~$9** of the $50 grant.

## Files

| file | what it is |
|---|---|
| `leaderboard.json` | Full [`LeaderboardResult`](../../gemmajudge/schemas.py): one 8-prompt Gemma attack set run against 5 targets, every verdict + measured tokens. |
| `leaderboard_table.txt` | The ranked leaderboard as printed by `python -m gemmajudge.leaderboard_demo`. |
| `gemma27b_judge_vs_glm.txt` | A single `python -m gemmajudge.demo` run: Gemma-27B judge vs `glm-5p1`, including the **F9b judge self-consistency** pass. |

## Headline results (hallucination failure mode)

**Recorded demonstration leaderboard** — higher ASR = more hallucination-prone under this
eight-prompt Gemma attack set. This demonstrates the workflow; it is not a statistically
calibrated benchmark:

| # | target | ASR | failed |
|---|---|---|---|
| 1 | glm-5p1 | 25% | 2/8 |
| 2 | deepseek-v4-pro | 25% | 2/8 |
| 3 | glm-5p2 | 25% | 2/8 |
| 4 | kimi-k2p6 | 12% | 1/8 |
| 5 | gpt-oss-120b | 0% | 0/8 |

**Deterministic score stability (F9b):** re-judging showcase cases 3× at temperature 0 gave
`[5,5,5]`, `[1,1,1]`, `[1,1,1]` → **stdev 0.00**. This measures repeatability, not
correctness.

> Reproduce: set the engine env to a Gemma endpoint (Fireworks deployment or a
> self-hosted AMD vLLM endpoint) and run
> `python -m gemmajudge.leaderboard_demo --n 8 --targets <ids> --out out.json`.
