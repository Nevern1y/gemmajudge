"""Convert canonical GemmaJudge JSONL to Fireworks/OpenAI SFT chat JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def convert_row(row: dict[str, Any]) -> dict[str, Any]:
    messages = []
    for message in row["messages"]:
        role = message["role"]
        if role == "model":
            role = "assistant"
        messages.append({"role": role, "content": message["content"]})
    return {"messages": messages}


def convert_file(input_path: Path, output_path: Path) -> int:
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            dst.write(json.dumps(convert_row(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert GemmaJudge JSONL for Fireworks SFT.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    count = convert_file(args.input, args.output)
    print(f"wrote {count} Fireworks examples -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
