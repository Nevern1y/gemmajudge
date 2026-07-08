# GemmaJudge

**Adversarial LLM evaluation, powered by Gemma on AMD.**

GemmaJudge uses one open-weight model family (Gemma) in two adversarial roles — an
**Attacker** that generates targeted adversarial test cases for a chosen failure mode,
and a **Judge** that scores a target model's responses against a rubric — running the
closed loop with committed proof of Gemma self-hosted on AMD GPUs via vLLM + ROCm. The
live demo can use a managed OpenAI-compatible backend for uptime.

Built for the **AMD Developer Hackathon: ACT II — Track 3 (Unicorn)**.

> Status: engine, offline simulation, robustness leaderboard, and AMD proof-of-compute are
> done and committed. Built during the hackathon (6–11 Jul 2026).

## Hackathon submission artifacts

Track 3 (Unicorn) requires a GitHub repository, demo video, and slide deck PDF; a live
hosted URL is optional but recommended. The official participant guide explicitly says
**no Docker image is required for Track 3**. A `Dockerfile` is included only as a
reproducibility convenience, not as a submission gate.

- **GitHub repository:** <https://github.com/Nevern1y/gemmajudge>
- **Live demo URL:** <https://gemmajudge.streamlit.app/>
- **Slide deck PDF:** uploaded directly to the lablab.ai submission form.
- **AMD proof-of-compute:** [`docs/amd_proof/w7900/`](docs/amd_proof/w7900/)
- **Real Gemma leaderboard data:** [`docs/real_runs/`](docs/real_runs/)

## Why it exists

Every time a team ships or fine-tunes an LLM, they're guessing whether it now hallucinates
more, is easier to jailbreak, or has grown more biased. GemmaJudge closes an
**attacker + judge loop** from a single open-weight family, self-hosted on hardware you
control — so eval data never leaves your infra and the judge isn't a metered closed API.

## How it works

```
config (target endpoint, failure mode, N)
        │
        ▼
  Attacker (Gemma)  ──►  Target model  ──►  Judge (Gemma)  ──►  report
   adversarial            responses          score + reasoning     ASR, drill-down,
   test cases                                per case              cost meter, AMD panel
```

## Robustness leaderboard

The same loop scales past a single target: **one** Gemma-generated attack set, run
against **many** models, ranked by Attack Success Rate — a self-hosted, open-weight
red-team + judge that benchmarks any model's hallucination robustness on hardware you
control. This is a **real run** (Gemma-3-27B judge on Fireworks, 8 shared prompts):

| # | target | ASR | failed |
|---|---|---|---|
| 1 | glm-5p1 | 25% | 2/8 |
| 2 | deepseek-v4-pro | 25% | 2/8 |
| 3 | glm-5p2 | 25% | 2/8 |
| 4 | kimi-k2p6 | 12% | 1/8 |
| 5 | gpt-oss-120b | 0% | 0/8 |

Full data + methodology: [`docs/real_runs/`](docs/real_runs/). It's the **🏆 Robustness
leaderboard** tab in the app, and reproducible from the CLI:

```bash
python -m gemmajudge.leaderboard_demo --n 8 \
    --targets accounts/fireworks/models/glm-5p1,accounts/fireworks/models/gpt-oss-120b
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # fill in your keys — never commit .env
streamlit run app.py
```

Or with Docker (optional for Track 3; the `Dockerfile` is only for reproducible local
runs, not a required Unicorn-track artifact):

```bash
docker build -t gemmajudge .
docker run --rm -p 8501:8501 --env-file .env gemmajudge   # open http://localhost:8501
```

### Try it in 30 seconds — no keys, no network

A **simulated** backend runs the whole attack→run→judge loop offline, so you can see
the product before wiring up a real managed or self-hosted AMD backend:

```bash
python -m gemmajudge.demo --offline        # full report in your terminal
# or, in the UI: streamlit run app.py  → toggle "Simulated demo" (on by default)
```

> Offline mode is **clearly labeled SIMULATED** — it's for development and demo
> fallback only. The real submission includes committed Gemma-on-AMD proof; canned output is never
> presented as a real model run.

## The engine seam

The UI talks to the engine through exactly one function — `run_eval` — so the
frontend and backend evolve independently:

```python
from gemmajudge.orchestrator import run_eval      # async def run_eval(config) -> EvalResult
from gemmajudge.schemas import EvalConfig, FailureMode

result = await run_eval(EvalConfig(
    failure_mode=FailureMode.HALLUCINATION,
    n_cases=20,
    target_endpoint="http://localhost:8000/v1",
    target_model_id="my-weak-model",
))
result.attack_success_rate     # % of cases the target failed (score >= 4)
result.cases                   # attacker prompt <-> judge verdict, for the drill-down
result.cost, result.metrics    # cost meter + AMD/latency panel
result.consistency             # F9b judge self-consistency spread
```

## Configuration

All configuration is via environment variables — nothing is hardcoded. See
[`.env.example`](.env.example) for the full list. Two inference backends, selected by
`INFERENCE_BACKEND`:

- `fireworks` — managed OpenAI-compatible backend for the live demo; not the AMD proof path.
- `mi300x` — self-hosted OpenAI-compatible vLLM backend on an AMD GPU.
  The committed proof used an AMD Radeon PRO W7900 (gfx1100); the same backend serves
  an AMD Instinct MI300X unchanged.

## AMD compute proof

A **real GemmaJudge run on AMD** lives in [`docs/amd_proof/w7900/`](docs/amd_proof/w7900/):
Gemma-3-4b-it (attacker + judge) and Gemma-3-1b-it (target) self-hosted via **vLLM + ROCm 7.2
on an AMD Radeon PRO W7900 (gfx1100)** on AMD Developer Cloud — with committed `rocm-smi`
output, vLLM serving logs, the launch command, and a real `eval_result.json` (hallucination
**ASR 80%**, judge self-consistency **stdev 0.00**, 136.9 tok/s). The `mi300x` backend and the
[`docs/amd_proof/mi300x_gemma.ipynb`](docs/amd_proof/mi300x_gemma.ipynb) notebook are included
as the AMD Instinct reference path. This satisfies the Track 3 AMD-compute requirement.

## License

[Apache-2.0](LICENSE).

## Acknowledgements

Gemma is an open-weight model family by Google DeepMind. AMD (ROCm on AMD Developer Cloud)
and Fireworks AI provide the inference infrastructure.
