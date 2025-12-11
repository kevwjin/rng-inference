"""
Measure generation reliability for EN/zh prompts at multiple lengths.

For each language ∈ {en, zh} and length ∈ {2,4,6,8,16,32}, run N=100 generations
using the exact prompt builders from sample_llm (no history). Count successes
(exact length AND all ints in [1,100]), length mismatches, and out-of-range ints.

Outputs a JSON summary with counts per (lang, length).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from sample_llm import (
    DEFAULT_MODEL,
    build_prompt_en,
    build_prompt_zh,
    extract_sequence_en,
    extract_sequence_zh,
)


PromptBuilder = Callable[[int], str]
Extractor = Callable[[str, int], List[int]]


@dataclass
class RunResult:
    success: int = 0
    length_mismatch: int = 0
    out_of_range: int = 0
    total: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "success": self.success,
            "length_mismatch": self.length_mismatch,
            "out_of_range": self.out_of_range,
            "total": self.total,
        }


def run_ollama(model: str, prompt: str) -> str:
    proc = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Ollama returned {proc.returncode}: {proc.stderr.strip()}"
        )
    return proc.stdout


def check_sequence(seq: List[int], expected_len: int) -> Tuple[bool, bool]:
    length_ok = len(seq) == expected_len
    range_ok = all(1 <= x <= 100 for x in seq)
    return length_ok, range_ok


def evaluate(model: str, lang: str, length: int, runs: int) -> RunResult:
    if lang == "en":
        prompt_fn: PromptBuilder = build_prompt_en
        extractor: Extractor = extract_sequence_en
    else:
        prompt_fn: PromptBuilder = build_prompt_zh
        extractor = extract_sequence_zh

    result = RunResult(total=runs)

    for _ in range(runs):
        prompt = prompt_fn(length)
        resp = run_ollama(model, prompt)
        try:
            seq = extractor(resp, length)
        except Exception:
            result.length_mismatch += 1
            continue
        length_ok, range_ok = check_sequence(seq, length)
        if length_ok and range_ok:
            result.success += 1
        else:
            if not length_ok:
                result.length_mismatch += 1
            if not range_ok:
                result.out_of_range += 1

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM generation reliability check.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Ollama model name (default from sample_llm)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Runs per (lang, length)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reliability_summary.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint JSON path to resume progress",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if present",
    )
    args = parser.parse_args()

    lengths = [2, 4, 6, 8, 16, 32]
    langs = ["en", "zh"]

    summary: Dict[str, Dict[int, Dict[str, int]]] = {}
    completed: Dict[str, Dict[int, int]] = {lang: {} for lang in langs}

    ckpt_path = args.checkpoint
    if ckpt_path is None and args.output:
        ckpt_path = args.output.with_suffix(args.output.suffix + ".checkpoint.json")

    if args.resume and ckpt_path and ckpt_path.exists():
        data = json.loads(ckpt_path.read_text())
        summary = data.get("summary", summary)
        completed = data.get("completed", completed)
        print(f"Resuming from checkpoint {ckpt_path}")

    for lang in langs:
        if lang not in summary:
            summary[lang] = {}
        for length in lengths:
            already = completed.get(lang, {}).get(length, 0)
            if already >= args.runs:
                print(f"Skipping {lang} len {length}; already completed {already}/{args.runs}")
                continue
            res = evaluate(args.model, lang, length, args.runs)
            summary[lang][length] = res.as_dict()
            completed[lang][length] = args.runs
            print(
                f"{lang} len {length}: "
                f"success={res.success}/{res.total}, "
                f"len_mismatch={res.length_mismatch}, "
                f"out_of_range={res.out_of_range}"
            )
            if ckpt_path:
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                ckpt_path.write_text(json.dumps({"summary": summary, "completed": completed}, indent=2))
                print(f"Saved checkpoint to {ckpt_path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
