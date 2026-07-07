# Deploying the live demo URL

The auto-screener checks a **live URL** (one of the three inspected artifacts). Fastest
path: **Streamlit Community Cloud**, backed by **Fireworks** (managed uptime — no MI300X
4-hour budget worry). Keep MI300X for the screenshottable AMD proof (`docs/amd_proof/`).

## Streamlit Community Cloud (recommended)

1. Repo must be **public** on GitHub (Track 3 requires a public repo anyway).
2. Go to <https://share.streamlit.io> → **New app** → pick `Nevern1y/gemmajudge`,
   branch `main`, main file `app.py`.
3. In **Advanced settings → Secrets**, paste (these become env vars):
   ```toml
   INFERENCE_BACKEND = "fireworks"
   # Gemma is DEPLOY-ONLY on Fireworks (see the ⚠️ note below): MODEL_ID is your
   # on-demand deployment path, not a serverless model id.
   MODEL_ID = "accounts/<your-account>/deployments/<deployment-id>"
   FIREWORKS_API_KEY = "fw_..."
   FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
   TARGET_ENDPOINT = "https://api.fireworks.ai/inference/v1"
   TARGET_MODEL_ID = "accounts/fireworks/models/gpt-oss-120b"  # a serverless target
   TARGET_API_KEY = "fw_..."
   PRICE_PER_1K_PROMPT_TOKENS = "0"
   PRICE_PER_1K_COMPLETION_TOKENS = "0"
   PRICE_SOURCE = "Fireworks pricing 2026-07"
   ```
   ⚠️ **Verified 2026-07-07: Fireworks has NO serverless Gemma** — every Gemma model
   (`gemma-3-4b/12b/27b-it`, `gemma-4-*`) is **deploy-only** (dedicated GPU). To run
   Gemma as the engine you first create an on-demand deployment (Model Library →
   the Gemma model → **Create Deployment**), wait for **Ready** (H100 capacity can be
   scarce — retry if it says "no available capacity"), then use its
   `accounts/<acct>/deployments/<id>` path as `MODEL_ID`. Serverless models
   (`gpt-oss-120b`, `glm-5p1/2`, `deepseek-v4-pro`, `kimi-k2p6`) make good *targets*.
   Remember to **delete the deployment** when done — it bills per GPU-hour.
   (Streamlit maps `st.secrets` and env; our `config.load_settings()` reads env, and
   `python-dotenv` is a no-op there — the platform vars win.)
4. Deploy. First load installs `requirements.txt` (includes `streamlit`, `openai`,
   `pydantic`, `pandas`). The app reads the secrets above and runs real Fireworks calls.
   Even with **no** engine configured, the **🏆 Robustness leaderboard** tab always
   renders the committed real Gemma-3-27B run (`docs/real_runs/leaderboard.json`).
5. Verify from an incognito window before submitting.

**For the AMD story, prefer MI300X** (below): it is the DQ-gate proof *and* avoids the
Fireworks H100 capacity/billing dance. Fireworks is the convenience live backend only.

## Fallback: point the URL at the self-hosted MI300X endpoint

If a needed Gemma variant isn't on Fireworks, serve it on MI300X (see
`docs/amd_proof/`) and set `INFERENCE_BACKEND=mi300x`, `MI300X_BASE_URL=<public url>`.
Note the notebook's 4-hour/24-hour budget — Fireworks is the safer public backend.

## Smoke check before submitting

- Repo public, CI green on `main`.
- `python -m gemmajudge.demo --n 10` returns a real ASR with your keys.
- Live URL loads and completes a run in an incognito session.
- `docs/amd_proof/` has the MI300X screenshot/logs/config committed.
- `deck/GemmaJudge_deck.pdf` present. English-only outputs. No secrets committed.
