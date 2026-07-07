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
   MODEL_ID = "accounts/fireworks/models/gemma-4-31b-it"
   FIREWORKS_API_KEY = "fw_..."
   FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
   TARGET_ENDPOINT = "https://api.fireworks.ai/inference/v1"
   TARGET_MODEL_ID = "accounts/fireworks/models/gemma-3-4b-it"
   TARGET_API_KEY = "fw_..."
   PRICE_PER_1K_PROMPT_TOKENS = "0"
   PRICE_PER_1K_COMPLETION_TOKENS = "0"
   PRICE_SOURCE = "Fireworks pricing 2026-07"
   ```
   Confirmed available on Fireworks (2026-07-07): `gemma-4-31b-it` (strong,
   attacker+judge) and `gemma-3-4b-it` (weak target). Both serverless / pay-per-token.
   (Streamlit maps `st.secrets` and env; our `config.load_settings()` reads env, and
   `python-dotenv` is a no-op there — the platform vars win.)
4. Deploy. First load installs `requirements.txt` (includes `streamlit`, `openai`,
   `pydantic`, `pandas`). The app reads the secrets above and runs real Fireworks calls.
5. Verify from an incognito window before submitting.

**Model ids (confirmed 2026-07-07 on this account):** attacker+judge =
`gemma-4-31b-it`; weak target = `gemma-3-4b-it` (or `gemma-3-1b-it` for a weaker,
more dramatic target). Re-list any time with `python scripts/probe_fireworks.py`.

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
