"""Batch random integer sequence generation via Ollama."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np

DEFAULT_MODEL = "llama3.2:3b-instruct-q4_0"
ARTIFACT_ROOT = Path("artifacts")

PromptBuilder = Callable[[int], str]
SequenceExtractor = Callable[[str, int], List[int]]


@dataclass
class BatchResult:
    """Container for a single batch generation result."""

    index: int
    integers: List[int]


def build_prompt_en(seq_len: int) -> str:
    """Create an English prompt that requests space-separated integers."""
    return (
        f"Write out a sequence of {seq_len} random integers "
        "between 1 and 100 inclusive. "
        "Separate the integers by a space ' ' character. "
        "DO NOT output anything else.\n"
    )


def build_prompt_zh(seq_len: int) -> str:
    """Create a Chinese prompt that requests space-separated integers."""
    return (
        f"写出 {seq_len} 个介于 1 到 100（包含）的随机整数。"
        "使用空格字符（' '）分隔这些整数。"
        "不要输出任何其他内容。\n"
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


def extract_sequence_en(response: str, expected_len: int) -> List[int]:
    response = response.strip()
    tokens = [token for token in response.split(" ") if token]

    integers: List[int] = []
    for token in tokens:
        try:
            integers.append(int(token))
        except ValueError as e:
            raise RuntimeError(
                f"model response contained a non-integer token {token!r}: "
                f"{response!r}"
            ) from e

    if len(integers) != expected_len:
        raise RuntimeError(
            f"expected {expected_len} integers, got {len(integers)}: "
            f"{response!r}"
        )

    return integers


def extract_sequence_zh(response: str, expected_len: int) -> List[int]:
    response = response.strip()

    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if len(lines) == expected_len:
        tokens = [line.split()[-1] for line in lines]

        integers: List[int] = []
        for token in tokens:
            try:
                integers.append(int(token))
            except ValueError as e:
                raise RuntimeError(
                    f"model response contained a non-integer token {token!r}: "
                    f"{response!r}"
                ) from e
        return integers

    # Fall back to the space-separated format shared with the English prompt.
    return extract_sequence_en(response, expected_len)


def generate_sequence(
    i: int,
    seq_len: int,
    prompt_builder: PromptBuilder,
    extractor: SequenceExtractor,
) -> BatchResult:
    prompt = prompt_builder(seq_len)
    process = run_ollama(prompt)
    if process.returncode != 0:
        stderr = process.stderr
        raise RuntimeError(
            f"Ollama returned non-zero exit code {process.returncode} "
            f"for batch {i}.\n"
            + (f"Stderr: {stderr}" if stderr else "")
        )

    seq = extractor(process.stdout, seq_len)
    if len(seq) != seq_len:
        raise RuntimeError(
            f"Expected {seq_len} integers in batch {i}, "
            f"but extracted {len(seq)}.\n"
            f"Extracted model response: {process.stdout.strip()}"
        )

    return BatchResult(index=i, integers=seq)


def validate_count(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def generate_sequences(
    num_seqs: int,
    seq_len: int,
    prompt_builder: PromptBuilder,
    extractor: SequenceExtractor,
) -> Sequence[BatchResult]:
    results: List[BatchResult] = []
    i = 1
    while len(results) < num_seqs:
        try:
            result = generate_sequence(i, seq_len, prompt_builder, extractor)
        except RuntimeError as e:
            print(f"Retrying batch {i} after error: {e}", file=sys.stderr)
            continue
        results.append(result)
        i += 1
    return results


def save_sequences(
    seqs: list[list[int]],
    prefix: str,
    seq_len: int,
    num_seqs: int,
) -> Path:
    base_dir = ARTIFACT_ROOT / f"{prefix}-{seq_len}"
    output_dir = base_dir / "llm"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}-{seq_len}-llm-{num_seqs}.npz"
    output_path = output_dir / filename
    seqs_np = np.asarray(seqs, dtype=np.int64)
    np.savez(output_path, sequences=seqs_np)
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate random integer sequences")
    parser.add_argument(
        "--prompt-language", "-l",
        choices=("english", "en", "chinese", "zh"),
        default="english",
        help="Prompt language to use for generation (default: english)",
    )
    parser.add_argument(
        "--sequence-length", "-n",
        type=int,
        default=10,
        help="Number of integers per sequence (default: 10)")
    parser.add_argument(
        "--num-sequences", "-s",
        type=int,
        default=1,
        help="Number of sequences to generate (default: 1)")
    parser.add_argument(
        "--save-npz",
        action="store_true",
        help="Save sequences as an NPZ file in a language and length-specific directory",
    )
    args = parser.parse_args(argv)

    try:
        num_seqs = validate_count(args.num_sequences, "--num-sequences")
        seq_len = validate_count(args.sequence_length, "--sequence-length")
    except ValueError as e:
        parser.error(str(e))

    if args.prompt_language in ("english", "en"):
        prompt_builder = build_prompt_en
        extractor = extract_sequence_en
    else:
        prompt_builder = build_prompt_zh
        extractor = extract_sequence_zh

    try:
        batches = \
            generate_sequences(num_seqs, seq_len, prompt_builder, extractor)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    prefix = "en" if args.prompt_language in ("english", "en") else "zh"

    seqs = [batch.integers for batch in batches]
    print(json.dumps(seqs))

    if args.save_npz:
        output_path = save_sequences(
            seqs,
            prefix,
            args.sequence_length,
            args.num_sequences,
        )
        print(f"Saved sequences to {output_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
