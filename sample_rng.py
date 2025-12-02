"""Generate pseudorandom integer sequences for LSTM/LLM baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Sequence

import numpy as np

ARTIFACT_ROOT = Path("artifacts")
PREFIX = "na"
PROGRESS_INTERVAL_NUMBERS = 1 << 13  # 8192


def generate_sequences(
    num_seqs: int,
    seq_len: int,
    seed: int | None
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sequences = rng.integers(1, 101, size=(num_seqs, seq_len), endpoint=False)
    total_numbers = num_seqs * seq_len
    if total_numbers >= PROGRESS_INTERVAL_NUMBERS:
        # Print a single completion notice for RNG (generation is vectorized).
        print(
            f"Progress: {total_numbers}/{total_numbers} numbers generated",
            flush=True,
        )
    return sequences.astype(np.int64)


def next_available_rep(seq_len: int, num_seqs: int) -> int:
    """
    Find the next rep index that avoids clobbering existing RNG artifacts.

    Matches:
    - {PREFIX}-rng-len{seq_len}-n{num_seqs}-rep<rep>.npz
    - legacy: {PREFIX}-{seq_len}-rng-{num_seqs}.npz (treated as rep=1)
    """
    if not ARTIFACT_ROOT.exists():
        return 1

    pattern = re.compile(
        rf"{re.escape(PREFIX)}-rng-len{seq_len}-n{num_seqs}-rep(\d+)\.npz$"
    )
    legacy_pattern = re.compile(
        rf"{re.escape(PREFIX)}-{seq_len}-rng-{num_seqs}\.npz$"
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
        if legacy_pattern.match(name):
            max_rep = max(max_rep, 1)

    return max_rep + 1


def save_sequences(
    sequences: np.ndarray,
    seq_len: int,
    num_seqs: int,
    output_path: Path | None,
    rep: int,
) -> Path:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        filename = (
            f"{PREFIX}-rng-len{seq_len}-n{num_seqs}-rep{rep}.npz"
        )
        output_path = ARTIFACT_ROOT / filename
    np.savez(output_path, sequences=sequences)
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate pseudorandom integer sequences")
    parser.add_argument(
        "--sequence-length", "-n",
        type=int,
        default=16,
        help="Number of integers per sequence (default: 16)",
    )
    parser.add_argument(
        "--num-sequences", "-s",
        type=int,
        default=1,
        help="Number of sequences to generate (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional RNG seed for reproducibility",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=0,
        help="Repeat index for default filenames; 0 chooses the next available rep automatically",
    )
    parser.add_argument(
        "--save-npz",
        action="store_true",
        help="Save sequences as artifacts/<prefix>-rng-len<length>-n<count>-rep<rep>.npz",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Explicit output .npz path (overrides default naming)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Explicit output .npz path (overrides default naming)",
    )
    args = parser.parse_args(argv)

    rep_arg = args.rep
    if rep_arg < 0:
        parser.error("--rep must be non-negative")

    rep = rep_arg or next_available_rep(args.sequence_length, args.num_sequences)

    seqs = generate_sequences(args.num_sequences, args.sequence_length, args.seed)
    print(json.dumps(seqs.tolist()))

    if args.save_npz:
        output_path = save_sequences(
            seqs,
            args.sequence_length,
            args.num_sequences,
            args.output_path,
            rep,
        )
        print(f"Saved sequences to {output_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
