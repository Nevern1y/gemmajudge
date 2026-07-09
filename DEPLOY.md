# Deploying the live demo URL

The auto-screener checks a **live URL** (one of the inspected artifacts). The public
Streamlit app is a zero-cost viewer over committed real artifacts: the self-hosted AMD
W7900 run and the real Gemma leaderboard. Keep live model endpoints private/local unless
you intentionally enable a short judging demo.

## Streamlit Community Cloud (recommended)

1. Repo must be **public** on GitHub (Track 3 requires a public repo anyway).
2. Go to <https://share.streamlit.io> → **New app** → pick `Nevern1y/gemmajudge`,
   branch `main`, main file `app.py`.
3. No secrets are required for the submitted public viewer. It will load:
   `docs/amd_proof/w7900/eval_result.json` and `docs/real_runs/leaderboard.json`.
4. Optional private live Gemma backend: in **Advanced settings → Secrets**, paste:
    ```toml
    INFERENCE_BACKEND = "fireworks"
    # Gemma attacker + judge deployment path.
    MODEL_ID = "accounts/<your-account>/deployments/<gemma-deployment-id>"
    FIREWORKS_API_KEY = "fw_..."
    FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
    PRICE_PER_1K_PROMPT_TOKENS = "0"
    PRICE_PER_1K_COMPLETION_TOKENS = "0"
    PRICE_SOURCE = "Fireworks pricing 2026-07"
    ```
    By default, the target model is `accounts/fireworks/models/gpt-oss-120b` on the
    same Fireworks key. Add `TARGET_MODEL_ID`, `TARGET_ENDPOINT`, or `TARGET_API_KEY`
    only if you want to override that target.
   ⚠️ Gemma on Fireworks is usually deployment-only. For cost control, enable this
   live backend only during judging windows or demos, then delete the deployment when
   done; it bills per GPU-hour. The committed AMD proof remains the primary evidence
   that GemmaJudge runs self-hosted on AMD via ROCm/vLLM.
   (Streamlit maps `st.secrets` and env; our `config.load_settings()` reads env, and
   `python-dotenv` is a no-op there — the platform vars win.)
5. Deploy. First load installs `requirements.txt` (includes `streamlit`, `openai`,
    `pydantic`, `pandas`). The app reads the secrets above and runs real Fireworks calls.
    Even with **no** engine configured, the **🏆 Robustness leaderboard** tab always
    renders the committed real Gemma-3-27B run (`docs/real_runs/leaderboard.json`).
6. Verify from an incognito window before submitting.

**For the AMD story, the self-hosted AMD run** is the DQ-gate proof. A managed backend is
only for live-demo uptime and is not presented as the AMD-compute proof.

## Fallback: point the URL at the self-hosted AMD endpoint

If the needed Gemma variant isn't on Fireworks, serve it on a self-hosted AMD GPU (see
`docs/amd_proof/`) and set `INFERENCE_BACKEND=mi300x`, `MI300X_BASE_URL=<public url>`.
Note the AMD notebook's 4-hour/24-hour budget — Fireworks is the safer public backend.

## Troubleshooting: `ImportError` on the live app after a push

Streamlit Cloud hot-reloads the main script on each push but keeps the Python process
alive, so a module that was **already imported** (e.g. `gemmajudge.schemas`) is served
from `sys.modules` and won't pick up *newly added* top-level names — you'll see
`ImportError: cannot import name '<New>' from 'gemmajudge.schemas'`. Fix: **Manage app
→ ⋮ → Reboot app** (a full process restart re-imports everything). Only needed when a
push adds new module-level exports; ordinary edits hot-reload fine.

## Smoke check before submitting

- Repo public, CI green on `main`.
- Live URL: <https://gemmajudge.streamlit.app/> loads from an incognito browser.
- Docker image builds successfully: `docker build -t gemmajudge .` and runs the submitted Streamlit app.
- Mission Control loads the recorded AMD proof and recorded leaderboard target from
  committed JSON artifacts.
- **🏆 Robustness leaderboard** tab shows the real Gemma run in an incognito session.
- `docs/amd_proof/w7900/` has the AMD (Radeon W7900) `rocm-smi` + vLLM logs + serve command + a real `eval_result.json` committed.
- Slide deck PDF uploaded to the lablab.ai submission form. English-only outputs. No secrets committed.
