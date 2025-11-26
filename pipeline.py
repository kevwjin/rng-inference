"""Orchestrate RNG dataset generation, LSTM training, sampling, and histogram updates."""
from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

ARTIFACT_ROOT = Path("artifacts")


@dataclass
class PipelineConfig:
    sequence_length: int
    num_sequences: List[int]
    epochs: int
    seed: Optional[int]
    num_sequences_to_sample: Optional[int]
    force: bool
    skip_sampling: bool


class PipelineError(RuntimeError):
    pass


def run_command(args: List[str]) -> None:
    """Run a subprocess command, raising an error if it fails."""
    proc = subprocess.run(args, text=True, capture_output=True)
    if proc.returncode != 0:
        message = (
            f"Command failed: {' '.join(args)}\n"
            f"Stdout:\n{proc.stdout}\nStderr:\n{proc.stderr}"
        )
        raise PipelineError(message)
    if proc.stdout:
        print(proc.stdout.strip())


def generate_rng_dataset(seq_len: int, num_sequences: int, seed: Optional[int]) -> Path:
    args = [
        "python",
        "sample_rng.py",
        "--sequence-length",
        str(seq_len),
        "--num-sequences",
        str(num_sequences),
        "--save-npz",
    ]
    if seed is not None:
        args.extend(["--seed", str(seed)])

    run_command(args)
    return ARTIFACT_ROOT / f"na-{seq_len}-rng-{num_sequences}.npz"


def train_lstm(npz_path: Path, epochs: int) -> Path:
    args = [
        "python",
        "train_lstm.py",
        "--data-path",
        str(npz_path),
        "--epochs",
        str(epochs),
    ]
    run_command(args)

    # The training script saves to artifacts/<prefix>-<seq_len>-lstm.pt.
    # Rename/copy to include dataset size for clarity.
    base_prefix = npz_path.stem.split("-")[:2]
    prefix = "-".join(base_prefix)
    default_checkpoint = ARTIFACT_ROOT / f"{prefix}-lstm.pt"
    if not default_checkpoint.exists():
        raise PipelineError(f"Expected checkpoint not found: {default_checkpoint}")

    dataset_size = npz_path.stem.split("-")[-1]
    target_checkpoint = ARTIFACT_ROOT / f"{prefix}-rng-{dataset_size}-lstm.pt"

    if target_checkpoint.exists():
        target_checkpoint.unlink()
    default_checkpoint.rename(target_checkpoint)
    return target_checkpoint


def sample_lstm(model_path: Path, seq_len: int, num_sequences: int) -> Path:
    args = [
        "python",
        "sample_lstm.py",
        "--model-path",
        str(model_path),
        "--sequence-length",
        str(seq_len),
        "--num-sequences",
        str(num_sequences),
        "--save-npz",
    ]
    run_command(args)

    stem = model_path.stem
    sample_path = ARTIFACT_ROOT / f"{stem}-sample-{num_sequences}.npz"
    if not sample_path.exists():
        raise PipelineError(f"Expected sample NPZ not found: {sample_path}")
    return sample_path


def update_histograms() -> None:
    args = ["python", "plot_histogram.py"]
    run_command(args)


def parse_num_sequences(values: Iterable[str]) -> List[int]:
    num_sequences = []
    for value in values:
        try:
            num = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid count: {value}") from exc
        if num <= 0:
            raise argparse.ArgumentTypeError("Counts must be positive integers")
        num_sequences.append(num)
    if not num_sequences:
        raise argparse.ArgumentTypeError("At least one value is required")
    return num_sequences


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Run RNG->LSTM pipeline experiments")
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=16,
        help="Sequence length for generated data (default: 16)",
    )
    parser.add_argument(
        "--num-sequences",
        nargs="+",
        required=True,
        help="List of RNG dataset sizes to generate and train on",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs for LSTM (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for RNG dataset generation (optional)",
    )
    parser.add_argument(
        "--num-sequences-to-sample",
        type=int,
        help="If set, sample this many sequences from each trained LSTM",
    )
    parser.add_argument(
        "--skip-sampling",
        action="store_true",
        help="Skip sampling stage even if --sample-count is provided",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run stages even if artifacts already exist (overwrite)",
    )
    args = parser.parse_args()

    num_sequences = parse_num_sequences(args.num_sequences)
    return PipelineConfig(
        sequence_length=args.sequence_length,
        num_sequences=num_sequences,
        epochs=args.epochs,
        seed=args.seed,
        num_sequences_to_sample=args.num_sequences_to_sample,
        force=args.force,
        skip_sampling=args.skip_sampling,
    )


def main() -> int:
    config = parse_args()

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    for num in config.num_sequences:
        print(f"=== Processing dataset size {num} ===")
        npz_path = generate_rng_dataset(config.sequence_length, num, config.seed)
        checkpoint_path = train_lstm(npz_path, config.epochs)

        if config.num_sequences_to_sample and not config.skip_sampling:
            sample_lstm(checkpoint_path, config.sequence_length, config.num_sequences_to_sample)

    update_histograms()
    print("Pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
