"""Generate pseudorandom integer sequences for LSTM/LLM baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

ARTIFACTS_ROOT = Path("artifacts")
PREFIX = "rng"


def generate_sequences(
    num_seqs: int,
    seq_len: int,
    seed: int | None
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sequences = rng.integers(1, 101, size=(num_seqs, seq_len), endpoint=False)
    return sequences.astype(np.int64)


def save_sequences(sequences: np.ndarray, seq_len: int, num_seqs: int) -> Path:
    base_dir = ARTIFACTS_ROOT / f"{PREFIX}-{seq_len}"
    output_dir = base_dir / "rng"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{PREFIX}-{seq_len}-rng-{num_seqs}.npz"
    output_path = output_dir / filename
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
        "--save-npz",
        action="store_true",
        help="Save sequences under artifacts/pr-<length>/rng/",
    )
    args = parser.parse_args(argv)

    seqs = generate_sequences(args.num_batches, args.sequence_length, args.seed)
    print(json.dumps(seqs.tolist()))

    if args.save_npz:
        output_path = save_sequences(seqs, args.sequence_length, args.num_batches)
        print(f"Saved sequences to {output_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
