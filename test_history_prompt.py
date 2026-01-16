"""Quick utility to probe history-conditioned prompts and view model outputs."""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from typing import List, Sequence

import subprocess

# Reuse the same default model as sample_llm.py
DEFAULT_MODEL = "llama3.2:3b-instruct-q4_0"


def run_ollama(prompt: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ollama", "run", DEFAULT_MODEL],
        input=prompt,
        text=True,
        capture_output=True,
    )


def extract_sequence_en(response: str, expected_len: int) -> List[int]:
    response = response.strip()
    tokens = [token for token in response.split() if token]
    integers: List[int] = []
    for token in tokens:
        try:
            integers.append(int(token))
        except ValueError as e:
            raise RuntimeError(
                f"model response contained a non-integer token {token!r}: {response!r}"
            ) from e
    if len(integers) != expected_len:
        raise RuntimeError(
            f"expected {expected_len} integers, got {len(integers)}: {response!r}"
        )
    return integers


def extract_sequence_zh(response: str, expected_len: int) -> List[int]:
    lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
    if len(lines) == expected_len:
        tokens = [line.split()[-1] for line in lines]
        integers: List[int] = []
        for token in tokens:
            try:
                integers.append(int(token))
            except ValueError as e:
                raise RuntimeError(
                    f"model response contained a non-integer token {token!r}: {response!r}"
                ) from e
        return integers
    return extract_sequence_en(response, expected_len)


def build_prompt_en(history: list[int], next_len: int, hist_len: int) -> str:
    seed_last = [57, 84, 83, 67][:hist_len] or [57]
    numbers = seed_last if not history else history[-hist_len:]
    numbers_text = " ".join(str(x) for x in numbers)
    return (
        "Here are the last 4 numbers from an ongoing sequence of random integers "
        f"from 1 to 100 inclusive: {numbers_text}.\n"
        f"Continue the sequence by generating EXACTLY {next_len} additional integers (1 to 100 inclusive).\n"
        f"Output ONLY the {next_len} integers in space-separated format.\n"
        "Your entire response MUST contain exactly "
        f"{next_len} integers and exactly {next_len - 1} spaces — no text, "
        "no punctuation, no line breaks, no extra characters."
    )


def build_prompt_zh(history: list[int], next_len: int, hist_len: int) -> str:
    seed_last = [57, 84, 83, 67][:hist_len] or [57]
    numbers = seed_last if not history else history[-hist_len:]
    numbers_text = " ".join(str(x) for x in numbers)
    return (
        "以下是一个由 1 到 100（包含）随机整数序列的最后 4 个数字："
        f"{numbers_text}。\n"
        f"继续生成恰好 {next_len} 个额外的整数（介于 1 到 100 之间，包含上下界）。\n"
        f"仅输出这 {next_len} 个整数，使用空格分隔。\n"
        "完整回复必须只包含 "
        f"{next_len} 个整数和恰好 {next_len - 1} 个空格——不得包含文本、标点、换行或其他字符。"
    )


def generate_history(length: int, seed: int | None) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(1, 100) for _ in range(length)]


@dataclass
class GenResult:
    prompt: str
    response: str
    parsed: list[int] | None
    error: str | None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test history-conditioned prompts.")
    parser.add_argument("--language", choices=["en", "zh"], default="en", help="Prompt language (default: en)")
    parser.add_argument("--sequence-length", type=int, default=16, help="Length of next sequence to request (default: 16)")
    parser.add_argument("--history-length", type=int, default=4, help="Length of history to include (default: 4)")
    parser.add_argument("--history", type=str, help="Explicit history numbers, space-separated (overrides --history-length)")
    parser.add_argument("--num-generations", type=int, default=4, help="How many generations to run (default: 4)")
    parser.add_argument("--mode", choices=["history", "plain"], default="history", help="Use history-conditioned prompt or plain prompt (default: history)")
    parser.add_argument("--seed", type=int, help="Optional seed for random history")
    args = parser.parse_args(argv)

    next_len = args.sequence_length
    history_list: list[int]
    if args.history:
        try:
            history_list = [int(tok) for tok in args.history.split() if tok]
        except ValueError as e:
            parser.error(f"Invalid --history numbers: {e}")
            return 1
    else:
        history_list = generate_history(args.history_length, args.seed)

    hist_len = max(1, args.history_length)
    build_prompt = build_prompt_en if args.language == "en" else build_prompt_zh
    extractor = extract_sequence_en if args.language == "en" else extract_sequence_zh

    results: list[GenResult] = []
    current_history = history_list
    for i in range(1, args.num_generations + 1):
        if args.mode == "history":
            prompt = build_prompt(current_history, next_len, hist_len)
        else:
            # Plain prompts as in sample_llm.py
            if args.language == "en":
                prompt = (
                    f"Write out a sequence of {next_len} random integers "
                    "between 1 and 100 inclusive. "
                    "Separate the integers by a space ' ' character. "
                    "DO NOT output anything else.\n"
                )
            else:
                prompt = (
                    f"写出 {next_len} 介于 1 到 100（包含 1 和 100）的随机整数。"
                    "使用空格字符（' '）分隔这些整数。"
                    "不要输出任何其他内容。\n"
                )
        proc = run_ollama(prompt)
        resp = proc.stdout.strip()
        parsed: list[int] | None
        err: str | None
        if proc.returncode != 0:
            parsed = None
            err = f"ollama exit {proc.returncode}: {proc.stderr}"
        else:
            try:
                parsed = extractor(resp, next_len)
                err = None
            except Exception as e:
                parsed = None
                err = str(e)
        results.append(GenResult(prompt, resp, parsed, err))
        # Update history for the next prompt if parsing succeeded
        if parsed:
            current_history = (current_history + parsed)[-hist_len:]

    for idx, res in enumerate(results, 1):
        print(f"=== Generation {idx} ===")
        print("Prompt:")
        print(res.prompt)
        print("Response:")
        print(res.response)
        if res.parsed is not None:
            print("Parsed:", res.parsed)
        if res.error:
            print("Error:", res.error)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
