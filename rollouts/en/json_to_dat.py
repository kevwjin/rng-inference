#!/usr/bin/env python3
"""
Convert rollout JSON files into PGFPlots-friendly .dat tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List


def format_per_step_probs(per_step_probs: List[dict], min_int: int, max_int: int) -> str:
    lines = ["# per_step_probs", "step value prob"]
    for value in range(min_int, max_int + 1):
        for step_idx, step_probs in enumerate(per_step_probs, start=1):
            prob = step_probs.get(str(value), 0.0)
            lines.append(f"{step_idx} {value} {prob:.16g}")
    return "\n".join(lines)


def format_sampled_sequences(sampled_sequences: Iterable[Iterable[int]]) -> str:
    lines = ["# sampled_sequences", "sequence_idx step_idx value"]
    for seq_idx, sequence in enumerate(sampled_sequences):
        for step_idx, value in enumerate(sequence):
            lines.append(f"{seq_idx} {step_idx} {value}")
    return "\n".join(lines)


def convert_file(path: Path, output_dir: Path | None = None) -> Path:
    with path.open() as f:
        data = json.load(f)

    out_path = (output_dir / path.name if output_dir else path).with_suffix(".dat")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    contents = [
        f"# prompt: {data.get('prompt', '').strip()}",
        f"# steps: {data.get('steps')}",
        f"# rollouts: {data.get('rollouts')}",
        f"# min_int: {data.get('min_int')}",
        f"# max_int: {data.get('max_int')}",
    ]

    per_step_probs = data.get("per_step_probs")
    if per_step_probs:
        contents.append(format_per_step_probs(per_step_probs, data["min_int"], data["max_int"]))

    sampled_sequences = data.get("sampled_sequences")
    if sampled_sequences:
        contents.append(format_sampled_sequences(sampled_sequences))

    out_path.write_text("\n\n".join(contents) + "\n")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert rollout JSON files into .dat files.")
    parser.add_argument("paths", nargs="+", help="JSON files or directories to convert.")
    parser.add_argument("-o", "--output-dir", type=Path, help="Optional directory for .dat files.")
    return parser.parse_args()


def iter_sources(paths: Iterable[str]) -> Iterable[Path]:
    for entry in paths:
        path = Path(entry)
        if path.is_dir():
            yield from path.glob("*.json")
        else:
            yield path


def main() -> None:
    args = parse_args()
    targets = list(iter_sources(args.paths))
    if not targets:
        raise SystemExit("No JSON files found to convert.")

    for target in targets:
        out_path = convert_file(target, args.output_dir)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
