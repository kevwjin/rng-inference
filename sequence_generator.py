"""Batch random integer sequence generation via Ollama."""
from __future__ import annotations

import argparse
import json
import re
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
    raw_response: str


def build_prompt(count: int) -> str:
    """Create a prompt that nudges the model to emit strict JSON."""
    return (
        "You are a random number generator. Respond with valid JSON only.\n"
        "Return exactly this structure: {\"sequence\": [<numbers>]} where <numbers> is a list"
        f" of {count} random integers between 0 and 255 inclusive.\n"
        "Integers may repeat and must appear as bare numbers (not strings).\n"
        "Do not add backticks, markdown, or commentary — only raw JSON."
    )


def run_ollama(prompt: str, model: str = DEFAULT_MODEL, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    """Invoke Ollama with the provided prompt and return the completed process."""
    try:
        return subprocess.run(  # noqa: S603,S607 - intentional shell invocation
            ["ollama", "run", model],
            input=prompt,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("The 'ollama' executable was not found on PATH.") from exc


def extract_sequence(response: str, expected_count: int) -> List[int]:
    """Attempt to extract a list of integers from the model response."""
    response = response.strip()

    # First try strict JSON parsing.
    try:
        data = json.loads(response)
        sequence = data.get("sequence")
        if isinstance(sequence, list) and all(isinstance(item, int) for item in sequence):
            return sequence
    except json.JSONDecodeError:
        pass

    # Fallback: regex extract integers.
    integers = [int(match) for match in re.findall(r"\d+", response)]
    if expected_count:
        integers = integers[:expected_count]
    return integers


def validate_count(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def generate_batch(index: int, count: int, model: str, timeout: float | None) -> BatchResult:
    prompt = build_prompt(count)
    process = run_ollama(prompt, model=model, timeout=timeout)
    if process.returncode != 0:
        stderr = process.stderr.strip()
        raise RuntimeError(
            f"Ollama returned non-zero exit code {process.returncode} for batch {index}."
            + (f" Stderr: {stderr}" if stderr else "")
        )

    sequence = extract_sequence(process.stdout, expected_count=count)
    if len(sequence) != count:
        raise RuntimeError(
            f"Expected {count} integers in batch {index}, but extracted {len(sequence)}."
            "\nResponse was:\n"
            f"{process.stdout.strip()}"
        )

    if any(number < 0 or number > 255 for number in sequence):
        raise RuntimeError(
            f"Batch {index} produced integers outside the 0-255 range: {sequence}."
        )

    return BatchResult(index=index, integers=sequence, raw_response=process.stdout)


def run_batches(batch_count: int, count_per_batch: int, model: str, timeout: float | None) -> Sequence[BatchResult]:
    results: List[BatchResult] = []
    for batch_index in range(1, batch_count + 1):
        results.append(generate_batch(batch_index, count_per_batch, model, timeout))
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate batches of random integer sequences via Ollama.")
    parser.add_argument("--batches", type=int, default=1, help="Number of sequences to generate (default: 1)")
    parser.add_argument("--count", type=int, default=10, help="Integers per sequence (default: 10)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--timeout", type=float, default=None, help="Optional timeout per request in seconds")
    args = parser.parse_args(argv)

    try:
        batch_total = validate_count(args.batches, "--batches")
        count_per_batch = validate_count(args.count, "--count")
    except ValueError as exc:
        parser.error(str(exc))

    try:
        batches = run_batches(batch_total, count_per_batch, args.model, args.timeout)
    except Exception as exc:  # noqa: BLE001 - surface meaningful errors to CLI
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    payload = {"sequences": [batch.integers for batch in batches]}
    print(json.dumps(payload, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
