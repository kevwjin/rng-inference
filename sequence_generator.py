"""Batch random integer sequence generation via Ollama."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Sequence

DEFAULT_MODEL = "llama3.2:3b-instruct-q4_0"


@dataclass
class BatchResult:
    """Container for a single batch generation result."""

    index: int
    integers: List[int]


def build_prompt(count: int) -> str:
    """Create a prompt that requests space-separated integers."""
    return (
        f"Generate a sequence of {count} random integers "
        "between 0 and 100 inclusive. "
        "Separate the integers by a space ' ' character. "
        "DO NOT output anything else.\n"
    )


def run_ollama(prompt: str) -> subprocess.CompletedProcess[str]:
    """
    Invoke Ollama with the provided prompt and return the completed process.
    """
    return subprocess.run(
        ["ollama", "run", DEFAULT_MODEL],
        input=prompt,
        text=True,
        capture_output=True,
    )


def extract_sequence(response: str, expected_len: int) -> List[int]:
    response = response.strip()
    if not response:
        raise RuntimeError("model returned an empty response")

    parts = response.split(" ")
    if "" in parts:
        raise RuntimeError(
            f"model did not return space-separated integers: {response!r}"
        )

    if len(parts) != expected_len:
        raise RuntimeError(
            f"expected {expected_len} integers, got {len(parts)}: {response!r}"
        )

    integers: List[int] = []
    for part in parts:
        try:
            integers.append(int(part))
        except ValueError as e:
            raise RuntimeError(
                f"model response contained a non-integer token {part!r}: "
                "{response!r}"
            ) from e

    return integers


def generate_batch(i: int, seq_len: int) -> BatchResult:
    prompt = build_prompt(seq_len)
    process = run_ollama(prompt)
    if process.returncode != 0:
        stderr = process.stderr
        raise RuntimeError(
            f"Ollama returned non-zero exit code {process.returncode} "
            f"for batch {i}.\n"
            + (f"Stderr: {stderr}" if stderr else "")
        )

    seq = extract_sequence(process.stdout, expected_len=seq_len)
    if len(seq) != seq_len:
        raise RuntimeError(
            f"Expected {seq_len} integers in batch {i}, "
            f"but extracted {len(seq)}.\n"
            f"Extracted model response: {process.stdout.strip()}"
        )

    return BatchResult(index=i, integers=seq)


def run_batches(num_batches: int, seq_len: int) -> Sequence[BatchResult]:
    results: List[BatchResult] = []
    for i in range(1, num_batches + 1):
        results.append(generate_batch(i, seq_len))
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate random integer sequences")
    parser.add_argument(
        "--num-batches", "-b",
        type=int, default=1,
        help="Number of sequences to generate")
    parser.add_argument(
        "--sequence-length", "-n",
        type=int, default=10,
        help="Number of integers per sequence")
    args = parser.parse_args(argv)

    def validate_count(value: int, name: str) -> int:
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return value

    try:
        num_batches = validate_count(args.num_batches, "--num-batches")
        seq_len = validate_count(args.sequence_length, "--sequence-length")
    except ValueError as e:
        parser.error(str(e))

    try:
        batches = run_batches(num_batches, seq_len)
    except Exception as e:  # noqa: BLE001 - surface meaningful errors to CLI
        print(f"Error: {e}", file=sys.stderr)
        return 1

    seqs = [batch.integers for batch in batches]
    print(json.dumps(seqs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
