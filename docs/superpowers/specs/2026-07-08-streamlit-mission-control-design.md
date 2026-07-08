# GemmaJudge Streamlit Mission Control Design

Date: 2026-07-08
Status: Approved for implementation planning

## Context

GemmaJudge is a hackathon project for AMD Developer Hackathon: ACT II, Track 3. The backend is already working and exposes one frontend seam: `run_eval(config: EvalConfig) -> EvalResult`. The Streamlit frontend should be rebuilt from scratch, not redesigned incrementally from the current baseline.

The product story is: Gemma acts in two adversarial roles. The Attacker generates prompts, the Target answers, and the Judge scores the response with reasoning and evidence. The loop reports hallucination risk, drill-down evidence, cost, latency, and AMD-backed compute details.

## Design Direction

The new frontend will use a **Mission Control Narrative** with a **red-team control room** visual style.

The app should feel like a security/evaluation console, not a generic Streamlit form. It should be dramatic enough for a hackathon demo, but still credible and restrained.

Core visual principles:

- Dark graphite / near-black base.
- AMD red for risk, failed cases, and critical evidence.
- Gemma blue for Attacker and Judge roles.
- Muted green for safe/passed results.
- Compact technical dashboard typography.
- Honest copy: no fake AMD claims, no "world first" claims, no simulated output presented as real.

## Primary Demo Goal

The first screen should answer three questions within 10 seconds:

- What is GemmaJudge? Gemma attacks, the target answers, Gemma judges.
- Why does it matter? The app shows concrete hallucination failures, not only aggregate scores.
- Why does AMD matter? The screen surfaces backend, model id, latency, measured tokens, cost, and proof links.

The recommended first-screen mode is **Hybrid**:

- Show the real committed leaderboard snapshot immediately.
- Show a compact live evaluation console on the same screen.
- Let the detailed forensic drill-down become the memorable demo moment after a run or target selection.

## App Structure

The new `app.py` should be organized as small UI functions around the frozen backend seam.

High-level flow:

```text
main()
  load Streamlit secrets into env
  load real leaderboard artifact
  resolve real/offline settings
  render app chrome and theme
  render mission header
  render main tabs
```

Main tabs:

- Mission Control
- Live Evaluation
- Robustness Leaderboard
- AMD Proof

The default tab is Mission Control. It should not feel empty or tab-first; it should contain the strongest parts of the product immediately.

## Mission Control Screen

Hero copy:

```text
GemmaJudge
Adversarial LLM evaluation powered by Gemma on AMD.

Gemma attacks. The target answers. Gemma judges.
The report shows exactly where the model hallucinated.
```

Pipeline cards:

```text
[Gemma Attacker] -> [Target Model] -> [Gemma Judge] -> [Risk Report]
```

Card meanings:

- Gemma Attacker: generates targeted adversarial prompts.
- Target Model: system under test.
- Gemma Judge: scores 1-5 with reasoning and evidence.
- Risk Report: ASR, drill-down, measured cost, AMD backend.

Leaderboard snapshot on the same screen:

```text
Real Gemma run
5 targets · 8 shared attacks · 82,941 measured tokens
Top ASR: 25%
Safest target: gpt-oss-120b, 0%
```

This snapshot uses `docs/real_runs/leaderboard.json` and should be labeled as a real Gemma run, not simulated.

## Live Evaluation Console

Live controls should appear in the main surface, not only in the sidebar.

Controls:

- Mode: Simulated demo or Real backend.
- Target endpoint.
- Target model id.
- Failure mode.
- Number of cases.
- Run evaluation button.

Behavior:

- If real backend is not configured, show a friendly setup card and keep simulated mode available.
- If simulated mode is active, show a loud `SIMULATED` banner and explain that it is illustrative only.
- If real mode is active, show backend and model metadata before and after the run.
- On any run failure, show a friendly error panel and keep any previous successful result visible.

## Risk Report

After a run, render a report section with:

- Attack Success Rate.
- Failed cases.
- Wall clock time.
- Throughput.
- Inference backend.
- Engine model id.
- Measured tokens.
- Cost.

Color rules:

- ASR 0-10%: green.
- ASR 10-30%: amber.
- ASR 30%+: red.
- Score 1-2: pass / green.
- Score 3: borderline / amber.
- Score 4-5: fail / red.

Charts:

- Score distribution chart on a 1-5 scale.
- Token usage chart split by attacker, target, and judge.

Use Plotly where useful, since it is already in requirements.

## Worst-Case Dossier

The most memorable demo block should be a forensic dossier for the worst case.

Select the case with the highest judge score:

```python
max(result.cases, key=lambda c: c.verdict.score)
```

Render it as:

```text
CASE tc_001 · SCORE 5/5 · CONFIDENT FALSE

Gemma attack:
...

Target response:
...

Gemma judge:
...

Evidence span:
...
```

Below the dossier, show the remaining cases in compact expanders or rows. Worst cases should sort first.

## Robustness Leaderboard

The leaderboard tab should expose the real committed Gemma run.

Show:

- Real run label.
- Engine model id.
- Backend.
- Number of targets.
- Number of shared attacks.
- Measured tokens.

Table columns:

- Rank.
- Target.
- ASR.
- Failed.
- Mean score.
- Time.

Interactions:

- Select a target and inspect that target's drill-down.
- If time allows, add an attack matrix with rows as attacks, columns as targets, and cells colored by score.

The attack matrix is optional. It is visually strong, but not required for the first complete implementation.

## AMD Proof Tab

This tab should make the AMD requirement visible and honest.

Show:

- AMD-backed evaluation path.
- MI300X via vLLM + ROCm.
- Fireworks path for live URL.
- Shared OpenAI-compatible client for both paths.
- Request timeout guardrail under the 30s rule.
- Max concurrency.
- Link/path to `docs/amd_proof/`.

If proof artifacts exist, show a checklist:

- `rocm_smi.txt`
- `versions.txt`
- `vllm_engine.log`
- `serve_command.txt`
- `eval_result.json`
- `mi300x_screenshot.png`
- `notes.md`

If artifacts are missing, show them as pending. Do not imply proof exists when files are not present.

## Error Handling

The frontend must be demo-safe.

Rules:

- `ConfigError` becomes a friendly setup panel with missing env var names only.
- API/model failures become an on-screen run failure card.
- Leaderboard missing or invalid data should not crash the app.
- Simulated mode should always be available as a zero-key fallback.
- Do not print secrets.
- Do not expose raw tracebacks to judges.

## Technical Constraints

Do not change backend schemas or the integration seam.

Allowed imports from the backend:

```python
from gemmajudge.config import ConfigError, load_settings
from gemmajudge.offline import OfflineEngineClient, OfflineTargetClient
from gemmajudge.orchestrator import run_eval
from gemmajudge.schemas import EvalConfig, EvalResult, FailureMode, LeaderboardResult, TargetReport
```

Allowed frontend/runtime dependencies:

- Streamlit.
- pandas.
- plotly.
- Python standard library.

Do not introduce a new frontend framework. Do not modify `schemas.py`.

## Testing

Update `tests/test_app.py` around the new UI.

Required checks:

- App loads without exception.
- Simulated run renders Mission Control and Risk Report.
- `SIMULATED` banner appears in simulated mode.
- Leaderboard tab renders the committed real run.
- Metrics include Attack Success Rate.

Keep the existing CI posture: no network calls in tests.

## Copy Rules

The app text should be English-only for the hackathon judges.

Tone:

- Precise.
- Confident.
- Honest.
- No unsupported claims.
- No simulated results presented as real Gemma/AMD runs.

## Success Criteria

The frontend is successful when:

- A judge understands the product in 10 seconds.
- A simulated run works with zero keys.
- The real leaderboard is visible without live infrastructure.
- The worst hallucination case is obvious and memorable.
- AMD/Gemma usage is visible and honest.
- The app no longer looks like default Streamlit.
- Existing backend tests remain green.
