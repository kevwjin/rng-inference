"""Batch random integer sequence generation via Ollama."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable, List, Sequence

import numpy as np

DEFAULT_MODEL = "llama3.2:3b-instruct-q4_0"
PROGRESS_INTERVAL_NUMBERS = 1 << 13  # 8192
ARTIFACT_ROOT = Path("artifacts")

PromptBuilder = Callable[[int, list[int] | None], str]
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


def build_prompt_en_history(seq_len: int, last4: list[int] | None) -> str:
    """
    English prompt that conditions on the last 4 integers of the previous
    sequence.
    """
    seed_last4 = [57, 84, 83, 67]
    integers = seed_last4 if not last4 else last4[-4:]
    integers_text = " ".join(str(x) for x in integers)
    return (
        "Here are the last 4 numbers from "
        "an ongoing sequence of random integers "
        f"from 1 to 100 inclusive: {integers_text}.\n"
        f"Continue the sequence by generating EXACTLY {seq_len} "
        "additional integers (1 to 100 inclusive).\n"
        f"Output ONLY the {seq_len} integers in space-separated format.\n"
        "Your entire response MUST contain exactly "
        f"{seq_len} integers and exactly {seq_len - 1} spaces — "
        "no text, no punctuation, no line breaks, no extra characters."
    )


def build_prompt_zh(seq_len: int, last4: list[int] | None = None) -> str:
    """Create a Chinese prompt that requests space-separated integers."""
    return (
        f"写出 {seq_len} 个介于 1 到 100（包含）的随机整数。"
        "使用空格字符（' '）分隔这些整数。"
        "不要输出任何其他内容。\n"
    )


def build_prompt_zh_history(seq_len: int, last4: list[int] | None) -> str:
    """
    Chinese prompt that conditions on the last 4 integers of the previous
    sequence.
    """
    seed_last4 = [57, 84, 83, 67]
    integers = seed_last4 if not last4 else last4[-4:]
    integers_text = " ".join(str(x) for x in integers)
    return (
        "以下是一个由 1 到 100（包含）随机整数序列的最后 4 个数字：" 
        f"{integers_text}。\n"
        f"继续生成恰好 {seq_len} "
        "个额外的整数（介于 1 到 100 之间，包含上下界）。\n"
        f"仅输出这 {seq_len} 个整数，使用空格分隔。\n"
        "完整回复必须只包含 "
        f"{seq_len} 个整数和恰好 {seq_len - 1} 个空格——"
        "不得包含文本、标点、换行或其他字符。"
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

    # fall back to the space-separated format shared with the en prompt
    return extract_sequence_en(response, expected_len)


def generate_sequence(
    i: int,
    seq_len: int,
    prompt_builder: PromptBuilder,
    extractor: SequenceExtractor,
    last4: list[int] | None,
) -> BatchResult:
    prompt = prompt_builder(seq_len, last4)
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


def next_available_rep(
    prefix: str,
    seq_len: int,
    num_seqs: int,
    history: bool
) -> int:
    """
    Find the next rep index that avoids clobbering existing artifacts.

    Looks for files matching:
    - {prefix}-llm[-hist]-len{seq_len}-n{num_seqs}-rep<rep>.npz
    - legacy: {prefix}-{seq_len}-llm-{num_seqs}.npz (treated as rep=1)
    """
    if not ARTIFACT_ROOT.exists():
        return 1

    suffix = "-hist" if history else ""
    pattern = re.compile(
        rf"{re.escape(prefix)}-llm{suffix}-len{seq_len}-"
        rf"n{num_seqs}-rep(\d+)\.npz$"
    )
    legacy_pattern = None
    if not history:
        legacy_pattern = re.compile(
            rf"{re.escape(prefix)}-{seq_len}-llm-{num_seqs}\.npz$"
        )

    max_rep = 0
    for path in ARTIFACT_ROOT.glob("*.npz"):
        name = path.name
        m = pattern.match(name)
        if m:
            rep = int(m.group(1))
            if rep > max_rep:
                max_rep = rep
            continue
        if legacy_pattern and legacy_pattern.match(name):
            max_rep = max(max_rep, 1)

    return max_rep + 1


def save_checkpoint(
    seqs: list[list[int]],
    path: Path,
    seq_len: int,
    num_seqs: int,
) -> None:
    """
    Persist a checkpoint atomically so generation can be resumed.

    Stored fields:
    - sequences: int64 array of shape (generated, seq_len)
    - seq_len: scalar
    - target: scalar total sequences requested
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    seqs_np = np.asarray(seqs, dtype=np.int64)
    with tmp_path.open("wb") as f:
        np.savez(
            f,
            sequences=seqs_np,
            seq_len=np.int64(seq_len),
            target=np.int64(num_seqs),
        )
    tmp_path.replace(path)
    print(
        f"Checkpoint saved {len(seqs)}/{num_seqs} sequences -> {path}",
        file=sys.stderr,
        flush=True,
    )


def load_checkpoint(path: Path) -> list[list[int]]:
    data = np.load(path, allow_pickle=False)
    seqs = data["sequences"]
    return seqs.tolist()


def generate_sequences(
    num_seqs: int,
    seq_len: int,
    prompt_builder: PromptBuilder,
    extractor: SequenceExtractor,
    *,
    existing_results: list[BatchResult] | None = None,
    checkpoint_interval: int | None = None,
    checkpoint_path: Path | None = None,
    use_history: bool = False,
) -> Sequence[BatchResult]:
    results: List[BatchResult] = []
    if existing_results:
        results.extend(existing_results)
    i = len(results) + 1
    last4: list[int] | None = None
    if existing_results and use_history and results:
        tail_len = min(4, len(results[-1].integers))
        last4 = results[-1].integers[-tail_len:]
    while len(results) < num_seqs:
        try:
            result = generate_sequence(
                i,
                seq_len,
                prompt_builder,
                extractor,
                last4 if use_history else None,
            )
        except RuntimeError as e:
            print(f"Retrying batch {i} after error: {e}", file=sys.stderr)
            continue
        results.append(result)
        if use_history:
            tail_len = min(4, len(result.integers))
            last4 = result.integers[-tail_len:]
        numbers_generated = len(results) * seq_len
        total_numbers = num_seqs * seq_len
        if numbers_generated % PROGRESS_INTERVAL_NUMBERS == 0 or numbers_generated == total_numbers:
            print(
                f"Progress: {numbers_generated}/{total_numbers} numbers generated",
                file=sys.stderr,
                flush=True,
            )
        if (
            checkpoint_interval
            and checkpoint_path
            and (
                len(results) % checkpoint_interval == 0 or \
                len(results) == num_seqs
            )
        ):
            save_checkpoint(
                [batch.integers for batch in results],
                checkpoint_path,
                seq_len,
                num_seqs,
            )
        i += 1
    return results


def save_sequences(
    seqs: list[list[int]],
    prefix: str,
    seq_len: int,
    num_seqs: int,
    output_path: Path | None,
) -> Path:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        filename = f"{prefix}-{seq_len}-llm-{num_seqs}.npz"
        output_path = ARTIFACT_ROOT / filename
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
        default=16,
        help="Number of integers per sequence (default: 16)")
    parser.add_argument(
        "--num-sequences", "-s",
        type=int,
        default=1,
        help="Number of sequences to generate (default: 1)")
    parser.add_argument(
        "--rep",
        type=int,
        default=0,
        help="Repeat index for default filenames; "
             "0 chooses the next available rep automatically",
    )
    parser.add_argument(
        "--save-npz",
        action="store_true",
        help="Save sequences as "
             "artifacts/<lang>-llm[-hist]-len<length>-n<count>-rep<rep>.npz",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Explicit output .npz path (overrides default naming)",
    )
    parser.add_argument(
        "--history-prompts",
        action="store_true",
        help="Condition each prompt on the last 4 numbers of the previous sequence "
        "(first prompt seeds with 57 84 83 67)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume generation from a checkpoint if present",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help="Save a checkpoint every N sequences (0 disables checkpoints)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        help=(
            "Explicit checkpoint path (.npz); defaults to "
            "the output path with a '.checkpoint.npz' suffix"
        ),
    )
    args = parser.parse_args(argv)

    try:
        num_seqs = validate_count(args.num_sequences, "--num-sequences")
        seq_len = validate_count(args.sequence_length, "--sequence-length")
        rep_arg = validate_count(args.rep, "--rep") if args.rep else 0
    except ValueError as e:
        parser.error(str(e))

    if args.prompt_language in ("english", "en"):
        extractor = extract_sequence_en
        prefix = "en"
        prompt_builder = (
            build_prompt_en_history
            if args.history_prompts
            else (lambda n, _last4=None: build_prompt_en(n))
        )
    else:
        extractor = extract_sequence_zh
        prefix = "zh"
        prompt_builder = (
            build_prompt_zh_history
            if args.history_prompts
            else (lambda n, _last4=None: build_prompt_zh(n))
        )

    try:
        checkpoint_interval = None
        if args.checkpoint_interval > 0:
            checkpoint_interval = args.checkpoint_interval
        else:
            checkpoint_interval = None

        if args.output_path is not None:
            default_output_path = args.output_path
        else:
            rep = rep_arg or next_available_rep(
                prefix, args.sequence_length, args.num_sequences, args.history_prompts
            )
            hist_suffix = "-hist" if args.history_prompts else ""
            filename = (
                f"{prefix}-llm{hist_suffix}-len{args.sequence_length}-"
                f"n{args.num_sequences}-rep{rep}.npz"
            )
            default_output_path = ARTIFACT_ROOT / filename

        checkpoint_path = args.checkpoint_path
        if checkpoint_path is None and checkpoint_interval:
            checkpoint_path = default_output_path.with_suffix(
                default_output_path.suffix + ".checkpoint.npz"
            )

        existing_results: list[BatchResult] | None = None
        if args.resume and checkpoint_path and checkpoint_path.exists():
            existing_seqs = load_checkpoint(checkpoint_path)
            existing_results = [
                BatchResult(index=i, integers=seq)
                for i, seq in enumerate(existing_seqs, start=1)
            ]
            print(
                f"Resuming from checkpoint {checkpoint_path} "
                f"with {len(existing_results)}/{num_seqs} sequences",
                file=sys.stderr,
                flush=True,
            )
        elif args.resume and checkpoint_path:
            print(
                "Resume requested but checkpoint "
                f"{checkpoint_path} not found; starting fresh.",
                file=sys.stderr,
                flush=True,
            )

        batches = generate_sequences(
            num_seqs,
            seq_len,
            prompt_builder,
            extractor,
            existing_results=existing_results,
            checkpoint_interval=checkpoint_interval,
            checkpoint_path=checkpoint_path,
            use_history=args.history_prompts,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    seqs = [batch.integers for batch in batches]
    print(json.dumps(seqs))

    if args.save_npz:
        output_path = save_sequences(
            seqs,
            prefix,
            args.sequence_length,
            args.num_sequences,
            args.output_path or default_output_path,
        )
        print(f"Saved sequences to {output_path}", flush=True)
        if args.checkpoint_interval and checkpoint_path:
            # clean up checkpoint artifacts now that the final NPZ exists
            for candidate in (
                checkpoint_path,
                checkpoint_path.with_name(checkpoint_path.name + ".tmp"),
            ):
                if candidate.exists():
                    candidate.unlink()

    return 0


if __name__ == "__main__":
    sys.exit(main())
