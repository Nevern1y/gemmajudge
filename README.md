# GemmaJudge

**Adversarial LLM evaluation, powered by Gemma on AMD.**

GemmaJudge uses one open-weight model family (Gemma) in two adversarial roles — an
**Attacker** that generates targeted adversarial test cases for a chosen failure mode,
and a **Judge** that scores a target model's responses against a rubric — running the
whole closed loop on AMD (self-hosted on MI300X via vLLM + ROCm, or Gemma on Fireworks'
AMD-hosted infrastructure).

Built for the **AMD Developer Hackathon: ACT II — Track 3 (Unicorn)**.

> Status: 🚧 in active development during the hackathon (6–11 Jul 2026).

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

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # fill in your keys — never commit .env
streamlit run app.py
```

### Try it in 30 seconds — no keys, no network

A **simulated** backend runs the whole attack→run→judge loop offline, so you can see
the product before wiring up Fireworks/MI300X:

```bash
python -m gemmajudge.demo --offline        # full report in your terminal
# or, in the UI: streamlit run app.py  → toggle "Simulated demo" (on by default)
```

> Offline mode is **clearly labeled SIMULATED** — it's for development and demo
> fallback only. The real submission runs Gemma on AMD; canned output is never
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

- `fireworks` — Gemma on Fireworks' AMD-hosted infra (powers the live demo URL).
- `mi300x` — Gemma self-hosted on AMD Dev Cloud MI300X via vLLM + ROCm.

## AMD compute proof

Proof that Gemma runs on AMD MI300X (vLLM/ROCm config + deploy logs/screenshot) lives in
[`docs/amd_proof/`](docs/amd_proof/). This is a Track 3 requirement.

## License

[Apache-2.0](LICENSE).

## Acknowledgements

Gemma is an open-weight model family by Google DeepMind. AMD Instinct™ MI300X + ROCm and
Fireworks AI provide the inference infrastructure.
