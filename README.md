# GemmaJudge

**Adversarial LLM evaluation, powered by Gemma on AMD.**

GemmaJudge uses one open-weight model family (Gemma) in two adversarial roles — an
**Attacker** that generates targeted adversarial test cases and a **Judge** that scores
target responses — with committed proof of Gemma self-hosted on AMD GPUs via vLLM + ROCm.

Built for the **AMD Developer Hackathon: ACT II — Track 3 (Unicorn)**.

> Status: engine, recorded-run app, demonstration leaderboard, CI regression gate,
> AMD proof-of-compute, and
> measured ROCm LoRA judge proof are done and committed. Built during the hackathon
> (6–11 Jul 2026).

## Hackathon submission artifacts

Track 3 (Unicorn) requires a GitHub repository, demo video, and slide deck PDF; a live
hosted URL is optional but recommended. This repo includes a `Dockerfile` for a reproducible
one-command run and the linux/amd64 platform required by the general container rules.

- **GitHub repository:** <https://github.com/Nevern1y/gemmajudge>
- **Live demo URL:** <https://gemmajudge.streamlit.app/>
- **Demo video:** uploaded directly to the lablab.ai submission.
- **Slide deck PDF:** uploaded directly to the lablab.ai submission form.
- **AMD proof-of-compute:** [`docs/amd_proof/mi300x/`](docs/amd_proof/mi300x/)
- **Fine-tuned judge proof:** [`docs/fine_tune_eval/`](docs/fine_tune_eval/)
- **Real Gemma leaderboard data:** [`docs/real_runs/`](docs/real_runs/)
- **Rules/compliance audit:** [`docs/HACKATHON_COMPLIANCE.md`](docs/HACKATHON_COMPLIANCE.md)
- **AI tooling disclosure:** [`AI_TOOL_DISCLOSURE.md`](AI_TOOL_DISCLOSURE.md)

## Why it exists

Every time a team ships or fine-tunes an LLM, they're guessing whether it now hallucinates
more, is easier to jailbreak, or has grown more biased. GemmaJudge closes an
**attacker + judge loop** from a single open-weight family, self-hosted on hardware you
control. When the loop is self-hosted, eval data stays on that infrastructure and the
judge is not a metered closed API.

## How it works

```
config (target endpoint, failure mode, N)
        │
        ▼
  Attacker (Gemma)  ──►  Target model  ──►  Judge (Gemma)  ──►  report
   adversarial            responses          score + reasoning     ASR, drill-down,
   test cases                                per case              cost meter, AMD panel
```

## Recorded demonstration leaderboard

The same loop scales past a single target: **one** Gemma-generated attack set, run
against **many** models, ranked by Attack Success Rate — an open-weight red-team + judge
workflow that can run on hardware you control. This is a **recorded real run**
(Gemma-3-27B judge on Fireworks, 8 shared
prompts), presented as a product demonstration rather than a statistically calibrated
benchmark:

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

Or with Docker:

```bash
docker build --platform linux/amd64 -t gemmajudge .
docker run --rm -p 8501:8501 gemmajudge   # open http://localhost:8501
# optional real backend: add --env-file .env
```

### Try it in 30 seconds — no keys, no network

The public Streamlit app is a no-key viewer over committed **real** GemmaJudge runs:
the MI300X ROCm proof in `docs/amd_proof/mi300x/eval_result.json` and the real Gemma-27B
leaderboard in `docs/real_runs/leaderboard.json`. No API key is needed to inspect the
product flow, metrics, prompts, target responses, and judge reasoning.

A clearly labeled simulated backend is still available as a development fallback:

```bash
python -m gemmajudge.demo --offline        # full report in your terminal
# or, in the UI: streamlit run app.py  → toggle "Simulated demo" (on by default)
```

> Offline mode is **clearly labeled SIMULATED** — it's for development fallback only.
> The submitted demo defaults to recorded real artifacts, not canned output.

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
result.consistency             # deterministic repeat-score spread
```

### Use it as a CI regression gate

The same real evaluation command can block a build when the target exceeds an allowed
ASR or failed-case budget. `--json` emits a machine-readable summary; exit code `1`
means a threshold failed and exit code `2` means configuration or arguments were invalid.

```bash
python -m gemmajudge.demo --n 20 --no-consistency \
    --max-asr 0.20 --max-failed-cases 4 --json > gemmajudge-summary.json
```

Configure the engine and target endpoints through `.env.example`; do not use the
simulated `--offline` backend as release evidence.

`result.metrics.wall_clock_seconds` measures the full multi-request evaluation path. Each
individual model request is independently capped at 25 seconds by default, and configuration
above 30 seconds is rejected.

## Configuration

All configuration is via environment variables — nothing is hardcoded. See
[`.env.example`](.env.example) for the full list. Two inference backends, selected by
`INFERENCE_BACKEND`:

- `fireworks` — optional private live backend for short demos; not the AMD proof path.
- `mi300x` — self-hosted OpenAI-compatible vLLM backend on an AMD GPU.
  The committed primary proof uses an AMD Instinct MI300X VF (gfx942); a historical
  Radeon PRO W7900 (gfx1100) artifact is retained for provenance.

The public URL does not require a live model endpoint. For a private live Gemma demo,
set `MODEL_ID` and endpoint credentials from `.env.example`; never commit real keys.

## AMD compute proof

A **real GemmaJudge run on AMD** lives in [`docs/amd_proof/mi300x/`](docs/amd_proof/mi300x/):
`google/gemma-3-4b-it` as attacker + judge and `google/gemma-3-1b-it` as target, self-hosted
through **vLLM + ROCm 6.4 on an AMD Instinct MI300X VF (`gfx942`, 191.6875 GiB)** on AMD
Developer Cloud. The artifact includes hardware/version output, exact commands, both vLLM
logs, the attacker prompt, a screenshot, and the full result JSON. The recorded three-case
demonstration produced **ASR 100% (3/3)** with a **55.31-second full pipeline wall clock**;
each OpenAI-compatible request was capped at **25 seconds**. All three recorded cases repeated
as `[5, 5, 5]` at temperature 0. That is deterministic score stability, not judge correctness,
reliability, benchmark performance, or a claim about general target robustness. The earlier
[`docs/amd_proof/w7900/`](docs/amd_proof/w7900/) run remains as historical evidence only.

## Fine-tuning pipeline

GemmaJudge also includes a measured ROCm LoRA judge fine-tune proof: dataset builder,
seed JSONL, training entrypoint, optional Fireworks conversion docs, direct-Transformers
fallback evaluation, and base-vs-tuned/variant selection. In the recorded 56-example
validation run, the tuned Gemma-3-4B judge improved JSON validity from **89.3% → 100.0%**,
pass/fail accuracy from **66.1% → 75.0%**, macro-F1 from **0.578 → 0.622**, and score MAE
from **1.38 → 1.30**. Two extra LoRA variants were trained and evaluated; the original
checkpoint remained the champion. See [`docs/fine_tune_eval/report.json`](docs/fine_tune_eval/report.json)
and [`docs/fine_tune_eval/README.md`](docs/fine_tune_eval/README.md). The adapter and
merged model artifacts are intentionally gitignored; only data schemas, scripts, docs,
and measured reports should be committed.

## License

[MIT](LICENSE), as required by the lablab.ai participation terms for this event.

## Acknowledgements

Gemma is an open-weight model family by Google DeepMind. AMD (ROCm on AMD Developer Cloud)
and Fireworks AI provide the inference infrastructure.

OpenAI-based coding assistance and Gamma were used during development and presentation
authoring. Scope and human verification are documented in
[`AI_TOOL_DISCLOSURE.md`](AI_TOOL_DISCLOSURE.md).
