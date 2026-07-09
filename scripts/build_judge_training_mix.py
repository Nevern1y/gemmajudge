"""Build a train-only oversampled judge SFT mix for calibration experiments."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def build_mix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mixed = list(rows)
    by_score: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        expected = _label(row)
        by_score[int(expected["score"])].append(row)
        by_mode[str(expected["failure_mode"])].append(row)

    for score, target in ((2, 16), (4, 16)):
        bucket = by_score[score]
        if not bucket:
            continue
        for idx in range(max(0, target - len(bucket))):
            mixed.append(_duplicate(bucket[idx % len(bucket)], f"scoremix_{score}_{idx + 1:02d}"))

    for mode, target in (("jailbreak", 40), ("bias", 40)):
        bucket = by_mode[mode]
        if not bucket:
            continue
        current = sum(1 for row in mixed if _label(row)["failure_mode"] == mode)
        for idx in range(max(0, target - current)):
            mixed.append(_duplicate(bucket[idx % len(bucket)], f"modemix_{mode}_{idx + 1:02d}"))
    return mixed


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores: Counter[int] = Counter()
    modes: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for row in rows:
        expected = _label(row)
        scores[int(expected["score"])] += 1
        modes[str(expected["failure_mode"])] += 1
        reasons[str(row.get("metadata", {}).get("mix_reason", "original"))] += 1
    return {
        "n": len(rows),
        "scores": dict(sorted(scores.items())),
        "modes": dict(sorted(modes.items())),
        "reasons": dict(sorted(reasons.items())),
    }


def _label(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(row["messages"][1]["content"])


def _duplicate(row: dict[str, Any], reason: str) -> dict[str, Any]:
    dup = json.loads(json.dumps(row))
    metadata = dup.setdefault("metadata", {})
    metadata["id"] = f"{metadata.get('id', 'row')}_{reason}"
    metadata["mix_reason"] = reason.rsplit("_", 1)[0]
    return dup


def main() -> int:
    parser = argparse.ArgumentParser(description="Build train-only judge mix JSONL.")
    parser.add_argument("--train", type=Path, default=Path("data/judge_train.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/judge_train_balanced.jsonl"))
    args = parser.parse_args()

    rows = [
        json.loads(line) for line in args.train.read_text(encoding="utf-8").splitlines() if line
    ]
    mixed = build_mix(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in mixed) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"source": summarize(rows), "mixed": summarize(mixed)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
