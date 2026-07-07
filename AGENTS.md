# AGENTS.md — GemmaJudge build brief (read this first)

> **Purpose of this file:** you (the AI coding assistant / architect) are building the
> **backend** of GemmaJudge for a hackathon. This is your cold-start context. Read it top
> to bottom before writing any code. Everything here is fact-checked against the official
> rules. When in doubt, this file wins over memory.

---

## 0. TL;DR — what to do when you open this

1. You own the **backend/engine**. Your human teammate owns the **frontend** (Streamlit).
2. The contract between you two is **frozen** in `gemmajudge/schemas.py`. Do not change it
   without a note in your reply (the teammate builds UI against it).
3. Build order is in **§7**. Next file to write is almost certainly `gemmajudge/config.py`
   then `gemmajudge/client.py` (env-driven OpenAI-compatible client).
4. Never violate the **hard constraints in §5** (30s/request, AMD proof, no secrets,
   English-only, no hardcoded answers).
5. Check `git log --oneline` to see how far the last session got.

---

## 1. The hackathon (context)

- **Event:** AMD Developer Hackathon: ACT II. Runs LIVE **6–11 Jul 2026**. Deadline **11 Jul**
  (exact HH:MM is on the lablab dashboard → Event Schedule, local timezone).
- **Track chosen: Track 3 — Unicorn (Open Innovation).** "An original AI application that
  uses AMD compute resources. There is no fixed task."
- **Prize we target:** Track 3, 1st place = **$2,500** (2nd $1,500 / 3rd $1,000). This is the
  real anchor. There is NO officially-confirmed fixed cash "Best Use of Gemma" bonus — do
  not put a Gemma-bonus dollar figure anywhere.
- **What we submit (Track 3):** GitHub repo (Yes) + demo video (Yes) + slide deck PDF (Yes)
  + live demo URL (optional but recommended). **No Docker image is required for Track 3.**
- **Auto pre-screening inspects: repo + slide deck PDF + live URL. It does NOT process the
  video.** So repo/deck/URL are the priority; video is for the human-judge round only.
- **Judging criteria (4, treated co-equal; no official % published):** Creativity &
  Originality · Completeness · Product/Market Potential · Use of AMD Platforms.

## 2. The product (what we're building)

**GemmaJudge** — adversarial LLM evaluation using **one open-weight model family (Gemma) in
two roles**:

- **Attacker (Gemma):** generates targeted adversarial prompts for a chosen failure mode.
- **Judge (Gemma):** scores the target model's responses against a rubric (1–5, pass/fail,
  reasoning, evidence span).

The loop is fully automated:
```
config (target endpoint, failure mode, N)
   → Attacker generates N adversarial prompts
   → each prompt sent to the TARGET model (system-under-test)
   → Judge scores each response
   → report: Attack Success Rate, score distribution, drill-down, cost meter, AMD panel
```

**Why it's defensible (say this, not more):** a closed attacker+judge loop from a single
open-weight family, **self-hosted on AMD** so eval data never leaves your hardware and the
judge isn't a metered closed API. Do NOT claim "nobody does adversarial generation" —
Promptfoo already does; our angle is the self-hosted single-family closed loop on AMD.

- **P0 failure mode: hallucination only.** Jailbreak = P1, bias = P2. Do not add P0 scope.
- **Demo target (system-under-test):** a deliberately weak model (e.g. Gemma 3 4B) so
  failures are dramatic. Must be verified on Day 1 that it actually hallucinates on our
  seeded attacks (the "Day-1 spike").

## 3. Roles — who builds what

| You (AI architect) — **BACKEND** | Teammate (human) — **FRONTEND** |
|---|---|
| `schemas.py` (done), `config.py`, `client.py`, attacker, judge, async orchestrator, cost/token accounting, MI300X/ROCm + Fireworks wiring | Streamlit `app.py`, config form, results table, score-distribution chart, cost meter UI, drill-down panel, deck, video, submission |
| Expose ONE entrypoint the UI calls: `async def run_eval(config: EvalConfig) -> EvalResult` | Calls only `run_eval()`; never imports engine internals |

**The integration seam is `run_eval(config) -> EvalResult`.** Keep that signature stable.
The teammate builds the UI against `schemas.py` fixtures, so as long as you honor the
contract, the two halves join without rework.

## 4. Frozen data contracts (source of truth = `gemmajudge/schemas.py`)

- `FailureMode` enum: `HALLUCINATION` (P0), `JAILBREAK` (P1), `BIAS` (P2)
- `AttackCase`: `{id, prompt, rationale, targeted_weakness}` — Attacker output
- `JudgeVerdict`: `{test_id, target_response, score(1–5), passed, reasoning, evidence_span}` — Judge output
- `EvalConfig`: `{failure_mode, n_cases, target_endpoint, target_model_id}` — user input
- `EvalResult`: `{config, verdicts[]}` + `.attack_success_rate` property (fraction scored ≥4)

`passed == (score <= 2)`. ASR counts cases with `score >= 4`. These are already implemented
and tested — reuse, don't redefine.

## 5. HARD CONSTRAINTS — never violate (from official rules)

1. **Response time < 30s per request** (all-tracks rule). → async fan-out; never run a huge
   batch on the live path. Instrument wall-clock per run. Batch of ~20 is the demo size.
2. **AMD compute usage is REQUIRED or the project is disqualified.** Gemma must run on AMD:
   self-hosted on MI300X (vLLM+ROCm) and/or Gemma on Fireworks' AMD-hosted infra. Proof
   (config + logs + screenshot) must be committed under `docs/amd_proof/` by end of Day 3.
3. **No secrets in git.** `.env` is gitignored; read all keys from env. We supply our OWN
   Fireworks keys — the `FIREWORKS_*`/`ALLOWED_MODELS` harness injection is a **Track 1**
   mechanic and does NOT apply to Track 3.
4. **English-only outputs.**
5. **No hardcoded/cached answers** — evaluation uses unseen variants; the attacker must
   genuinely generate.

## 6. Inference backends (env-selected)

`INFERENCE_BACKEND` picks where Attacker+Judge Gemma runs:
- `fireworks` → Gemma on Fireworks (AMD-hosted). Powers the **live URL** (managed uptime).
  Exact available Gemma model IDs must be confirmed from Fireworks (launch day was 6 Jul —
  confirm and set `MODEL_ID`).
- `mi300x` → Gemma self-hosted on AMD Dev Cloud MI300X via vLLM+ROCm. Powers the
  **screenshottable AMD proof**. This path does NOT depend on launch-day availability
  (open weights), so it's the robust backend for the AMD requirement.

Both are **OpenAI-compatible**, so ONE client works for both — flip via env. The target
model is a separate OpenAI-compatible endpoint (`TARGET_ENDPOINT` / `TARGET_MODEL_ID`).
Full env list is in `.env.example`.

## 7. Backend build order (your roadmap)

Do these in order. Each should be small, typed, and testable.

- [x] **`schemas.py`** — frozen contracts (DONE, tested).
- [ ] **`config.py`** — load + validate env (`INFERENCE_BACKEND`, `MODEL_ID`, keys, target).
      Fail loudly with a clear message if required vars are missing. Never print secrets.
- [ ] **`client.py`** — one `AsyncOpenAI`-based client factory. Given the backend env, return
      a client pointed at Fireworks or MI300X. A separate factory for the target endpoint.
      Include token-usage capture (for the cost meter) from the API response `usage`.
- [ ] **`prompts/`** — `attacker_hallucination.md` + `judge_hallucination.md` system prompts.
      Attacker seeded with TruthfulQA-style few-shot; Judge carries the 1–5 rubric.
- [ ] **`attacker.py`** — `async def generate_attacks(cfg) -> list[AttackCase]`. Structured
      JSON output (json_schema), validated into `AttackCase`. Retry on invalid JSON.
- [ ] **`target.py`** — `async def query_target(cfg, prompt) -> str`. Just calls the SUT.
- [ ] **`judge.py`** — `async def judge(case, response) -> JudgeVerdict`. json_schema, temp 0.
- [ ] **`orchestrator.py`** — `async def run_eval(cfg) -> EvalResult`. asyncio fan-out over N
      cases with a concurrency cap; per-run wall-clock; aggregate into `EvalResult`. **This
      is the seam the UI calls.**
- [ ] **judge reliability (F9b, P0-min):** re-score 1–3 showcase cases 3× and expose the
      spread. Pre-compute off the live path so it doesn't touch the 30s budget.
- [ ] **cost accounting:** sum token usage → $ using a configurable price-per-token; expose
      on `EvalResult` for the UI meter. Any "vs provider X" number needs a cited source.

**P1 (only if ahead):** jailbreak mode (`prompts/*_jailbreak.md` + AdvBench seed), full
self-consistency (every case ×3), leaderboard data, SQLite run history, PDF export data.

## 8. Stack & conventions

- Python 3.11, `asyncio`. Client: `openai` SDK (OpenAI-compatible → works for Fireworks &
  vLLM). Validation: `pydantic` v2. UI (teammate): `streamlit` + `plotly`.
- Lint: `ruff` (config in `pyproject.toml`). Tests: `pytest` in `tests/`. CI runs both.
- Keep functions async and typed. Every new module gets at least one test. Don't break the
  green CI. No network calls in tests — mock the client.
- Commits end with the Co-Authored-By trailer (see git log for the format).

## 9. Repo map

```
gemmajudge/                 # repo root (this file is here)
├── AGENTS.md               # ← you are reading it
├── README.md               # public-facing
├── gemmajudge/
│   ├── schemas.py          # FROZEN contracts (done)
│   ├── config.py           # TODO next
│   ├── client.py           # TODO
│   ├── attacker.py judge.py target.py orchestrator.py   # TODO
│   └── prompts/            # TODO system prompts
├── tests/test_schemas.py   # extend as you add modules
├── docs/amd_proof/         # AMD screenshots/logs/config go here (DQ gate)
├── .env.example            # full env var list
└── pyproject.toml requirements.txt .github/workflows/ci.yml
```

## 10. Open items needing a human (don't block on these — note and proceed)

1. Exact deadline HH:MM (lablab dashboard) — paste into README/PRD when known.
2. Exact Gemma model IDs available on Fireworks post-launch — sets `MODEL_ID` for real.
3. Which weak model is the demo target — confirm it reliably hallucinates (Day-1 spike).

> Private planning docs (full PRD, work-split, hackathon PDF) live OUTSIDE this repo in the
> parent folder on the owner's machine. This AGENTS.md is the self-contained version — you
> do not need those files to proceed, but the owner can share them if a detail is missing.
