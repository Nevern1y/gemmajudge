"""Probe Fireworks: verify the key, list Gemma model ids, and do one smoke call.

Answers the last launch-day unknown (PRD Q6: which Gemma variants + exact ids does
Fireworks expose) and confirms your $50 key works — before you wire up the demo.

Usage:
    export FIREWORKS_API_KEY=fw_xxx          # Windows PowerShell: $env:FIREWORKS_API_KEY="fw_xxx"
    python scripts/probe_fireworks.py
    python scripts/probe_fireworks.py --smoke accounts/fireworks/models/<some-gemma-id>
"""

from __future__ import annotations

import argparse
import os
import sys

BASE_URL = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")


def main() -> int:
    parser = argparse.ArgumentParser(description="List Gemma models on Fireworks + smoke test.")
    parser.add_argument("--smoke", metavar="MODEL_ID", help="send one chat request to this model")
    parser.add_argument("--all", action="store_true", help="list ALL models, not just Gemma")
    args = parser.parse_args()

    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        print("Set FIREWORKS_API_KEY first (see the module docstring).", file=sys.stderr)
        return 2

    try:
        from openai import OpenAI
    except ImportError:
        print("pip install openai>=1.40.0", file=sys.stderr)
        return 2

    client = OpenAI(base_url=BASE_URL, api_key=key, max_retries=0, timeout=25)

    try:
        models = client.models.list().data
    except Exception as exc:  # noqa: BLE001 - surface auth/network errors plainly
        print(f"Could not list models (check the key / base URL): {exc}", file=sys.stderr)
        return 1

    ids = sorted(m.id for m in models)
    shown = ids if args.all else [i for i in ids if "gemma" in i.lower()]
    print(f"{len(ids)} models total; showing {len(shown)}"
          f"{' (Gemma only)' if not args.all else ''}:\n")
    for i in shown:
        print(" ", i)
    if not shown:
        print("  (no Gemma models matched — re-run with --all to see everything)")

    if args.smoke:
        print(f"\nSmoke test -> {args.smoke}")
        try:
            r = client.chat.completions.create(
                model=args.smoke,
                messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                max_tokens=8,
                temperature=0,
            )
            print("  response:", r.choices[0].message.content.strip())
            print("  usage   :", r.usage)
            print("  [OK] key works, model reachable.")
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] smoke call failed: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
